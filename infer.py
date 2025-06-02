import torch
from utils.phonemize.mixed_phon import smart_phonemize
import omegaconf
from models import (
    load_ASR_models,
    load_F0_models,
    load_KotoDama_Prompter,
    load_KotoDama_TextSampler,
    TextEncoder,
    ProsodyPredictor,
    StyleEncoder,
    StyleTransformer1d,
    Transformer1d,
    AudioDiffusionConditional,
    LogNormalDistribution,
    KDiffusion,
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    WavLMDiscriminator,
)
from utils.PLBERT.util import load_plbert
from modules.istftnet import Decoder
from text_utils import TextCleaner
from modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

text_cleaner = TextCleaner()


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


class SPipeline:

    def __init__(self, config_path):

        self.config = omegaconf.OmegaConf.load(config_path)
        asr_config = self.config.get("ASR_config", False)
        asr_path = self.config.get("ASR_path", False)
        # print(asr_config, asr_path)
        self.text_aligner = load_ASR_models(asr_path, asr_config)

        F0_path = self.config.get("F0_path", False)
        self.pitch_extractor = load_F0_models(F0_path)

        BERT_path = self.config.get("PLBERT_dir", False)
        self.bert = load_plbert(BERT_path)
        self.bert_encoder = torch.nn.Linear(
            self.bert.config.hidden_size, self.config.model_params.hidden_dim
        )

        self.prompter = load_KotoDama_Prompter(
            path=self.config.get("KotoDama_Prompter_path", False)
        )
        self.text_sampler = load_KotoDama_TextSampler(
            path=self.config.get("KotoDama_TextSampler_path", False)
        )

        self.decoder = Decoder(
            dim_in=self.config.model_params.decoder.get("hidden_dim", False),
            style_dim=self.config.model_params.decoder.get("style_dim", False),
            dim_out=self.config.model_params.decoder.get("n_mels", False),
            resblock_kernel_sizes=self.config.model_params.decoder.get(
                "resblock_kernel_sizes", False
            ),
            upsample_rates=self.config.model_params.decoder.get(
                "upsample_rates", False
            ),
            upsample_initial_channel=self.config.model_params.decoder.get(
                "upsample_initial_channel", False
            ),
            resblock_dilation_sizes=self.config.model_params.decoder.get(
                "resblock_dilation_sizes", False
            ),
            upsample_kernel_sizes=self.config.model_params.decoder.get(
                "upsample_kernel_sizes", False
            ),
            gen_istft_n_fft=self.config.model_params.decoder.get(
                "gen_istft_n_fft", False
            ),
            gen_istft_hop_size=self.config.model_params.decoder.get(
                "gen_istft_hop_size", False
            ),
        )

        self.text_encoder = TextEncoder(
            channels=self.config.model_params.get("hidden_dim", False),
            kernel_size=5,
            depth=self.config.model_params.get("n_layer", False),
            n_symbols=self.config.model_params.get("n_token", False),
        )

        self.predictor = ProsodyPredictor(
            style_dim=self.config.model_params.get("style_dim", False),
            d_hid=self.config.model_params.get("hidden_dim", False),
            nlayers=self.config.model_params.get("n_layer", False),
            max_dur=self.config.model_params.get("max_dur", False),
            dropout=self.config.model_params.get("dropout", False),
        )

        self.style_encoder = StyleEncoder(
            dim_in=self.config.model_params.get("dim_in", False),
            style_dim=self.config.model_params.get("style_dim", False),
            max_conv_dim=self.config.model_params.get("hidden_dim", False),
        )  # acoustic style encoder
        self.predictor_encoder = StyleEncoder(
            dim_in=self.config.model_params.get("dim_in", False),
            style_dim=self.config.model_params.get("style_dim", False),
            max_conv_dim=self.config.model_params.get("hidden_dim", False),
        )  # prosodic style encoder

        if self.config.model_params.multispeaker:
            transformer = StyleTransformer1d(
                channels=self.config.model_params.style_dim * 2,
                context_embedding_features=self.bert.config.hidden_size,
                context_features=self.config.model_params.style_dim * 2,
                **self.config.model_params.diffusion.transformer,
            )
        else:
            transformer = Transformer1d(
                channels=self.config.model_params.style_dim * 2,
                context_embedding_features=self.bert.config.hidden_size,
                **self.config.model_params.diffusion.transformer,
            )

        self.diffusion = AudioDiffusionConditional(
            in_channels=1,
            embedding_max_length=self.bert.config.max_position_embeddings,
            embedding_features=self.bert.config.hidden_size,
            embedding_mask_proba=self.config.model_params.diffusion.embedding_mask_proba,  # Conditional dropout of batch elements,
            channels=self.config.model_params.style_dim * 2,
            context_features=self.config.model_params.style_dim * 2,
        )

        self.diffusion.diffusion = KDiffusion(
            net=self.diffusion.unet,
            sigma_distribution=LogNormalDistribution(
                mean=self.config.model_params.diffusion.dist.mean,
                std=self.config.model_params.diffusion.dist.std,
            ),
            sigma_data=self.config.model_params.diffusion.dist.sigma_data,  # a placeholder, will be changed dynamically when start training diffusion model
            dynamic_threshold=0.0,
        )
        self.diffusion.diffusion.net = transformer
        self.diffusion.unet = transformer

        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiResSpecDiscriminator()

        self.wd = WavLMDiscriminator(
            self.config.model_params.slm.hidden,
            self.config.model_params.slm.nlayers,
            self.config.model_params.slm.initial_channel,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion_sampler = DiffusionSampler(
            self.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),  # empirical parameters
            clamp=False,
        )

        self.__load_models(self.config.get("model_path", False))

    def __load_models(self, model_path):
        model_dict = {
            "bert": self.bert,
            "bert_encoder": self.bert_encoder,
            "predictor": self.predictor,
            "decoder": self.decoder,
            "predictor_encoder": self.predictor_encoder,
            "style_encoder": self.style_encoder,
            "diffusion": self.diffusion,
            "text_aligner": self.text_aligner,
            "pitch_extractor": self.pitch_extractor,
            "mpd": self.mpd,
            "msd": self.msd,
            "wd": self.wd,
            "KotoDama_Prompt": self.prompter,
            "KotoDama_Text": self.text_sampler,
        }
        params_whole = torch.load(model_path, map_location="cpu")
        params = params_whole["net"]
        params = {k: v for k, v in params.items() if k in model_dict}
        for k in model_dict:
            try:
                model_dict[k].load_state_dict(params[k])
            except:
                from collections import OrderedDict

                state_dict = params[k]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("module.", "")
                    new_state_dict[name] = v
                model_dict[k].load_state_dict(new_state_dict, strict=False)
            print(f"Loaded {k} from {model_path}")

        for k in model_dict:
            model_dict[k].to(self.device)
        return model_dict

    def load_style(self, style_path):
        style = torch.load(style_path)
        return style

    def get_style(self, pack, phonemize):
        len_phonemize = len(phonemize)
        s_style = list(pack.keys())

        list_distance = [abs(len_phonemize - len(pack[s])) for s in s_style]
        min_distance = min(list_distance)
        min_index = list_distance.index(min_distance)
        style = s_style[min_index]
        return style

    def generate(
        self,
        text,
        pack,
        alpha=0.3,
        beta=0.7,
        diffusion_steps=5,
        embedding_scale=1.0,
        rate_of_speech=1.0,
    ):
        text = smart_phonemize(text)
        ref_s = self.get_style(pack, text)

        tokens = text_cleaner(text)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)

            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.diffusion_sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,  # reference from the same speaker as the embedding
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x = self.predictor.lstm(d)
            x_mod = self.predictor.prepare_projection(x)
            duration = self.predictor.duration_proj(x_mod)

            duration = torch.sigmoid(duration).sum(axis=-1) / rate_of_speech

            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))

            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)

            F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)

            out = self.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50]


if __name__ == "__main__":
    config_path = "config/config_inference.yaml"
    pipeline = SPipeline(config_path)
    # pipeline.__load_models("Style_Tsukasa_v02/Top_ckpt_24khz.pth")
    # text = "Hello, how are you?"
    # pack = {"0": "0"}
    # out = pipeline.generate(text, pack)
    # print(out.shape)

import torch
import omegaconf


class StyleTTS2:

    def __init__(self, config_path, checkpoint_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = omegaconf.OmegaConf.load(config_path)
        
        self.plbert = PLBERT(self.config.PLBERT_dir)


        
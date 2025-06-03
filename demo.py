from infer import SPipeline
import gradio as gr
import json
import os
import torch
import soundfile as sf

ROOT_DIR = "/home/ubuntu/app/Tsukasa_Speech/sayaka"
data = json.load(open("transcripts.json"))
transcripts = list(data.keys())

pipeline = SPipeline("config/config_inference.yaml")

packs = {}
for emotion in os.listdir(ROOT_DIR):
    if os.path.isdir(os.path.join(ROOT_DIR, emotion)):
        packs[emotion] = torch.load(
            f"{ROOT_DIR}/{emotion}/style.pt", map_location="cpu"
        )


def generate_audio(text, emotion, embscale):
    pack = packs[emotion]
    wav = pipeline.generate(text, pack, embedding_scale=embscale)
    sf.write("temp.wav", wav, 24000)
    return "temp.wav"


with gr.Blocks() as demo:
    gr.Markdown("Tsukasa Speech")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Text")
            emotion = gr.Dropdown(
                label="Emotion",
                choices=[
                    "neutral",
                    "happy",
                    "sorrow",
                    "anger",
                    "fear",
                    "surprise",
                    "disgust",
                ],
            )
            embscale = gr.Slider(
                label="Embedding Scale",
                minimum=1.0,
                maximum=5.0,
                value=1.0,
                step=0.1,
                interactive=True,
            )
            generate = gr.Button("Generate")
        with gr.Column():
            audio = gr.Audio(label="Audio")
    generate.click(generate_audio, inputs=[text, emotion, embscale], outputs=audio)


if __name__ == "__main__":
    demo.launch(share=True, show_api=True)

from kokoro import KPipeline
import soundfile as sf
import torch
import os
import json
import numpy as np
import soundfile as sf
import sounddevice as sd


BASE_PATH = "/home/azureuser/project/model_bases"
MODEL_DIR = "koroko/"
koroko_path = os.path.join(BASE_PATH, MODEL_DIR)
os.makedirs(koroko_path, exist_ok=True)


def dowload_koroko_model():
    pipeline = KPipeline(lang_code='a') 
    torch.save(pipeline.model, os.path.join(koroko_path, "pytorch_model.bin"))

    # Save config
    config = {
        "lang_code": "a",
        "voice": "af_heart"
    }
    json.dump(config, open(os.path.join(koroko_path, "config.json"), "w"))


def load_koroko_pipeline(model_path, config_path):

    with open(config_path) as f:
        config = json.load(f)

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    pipeline = KPipeline(lang_code=config["lang_code"])
    pipeline.model = model
    pipeline.model.eval()

    return pipeline, config


def infer_and_play(pipeline, config, test_text: str, rate: int = 24000):
    """Run inference, concatenate audio, and play it in terminal."""
    generator = pipeline(test_text, voice=config.get("voice", "af_heart"))

    audio_chunks = []
    for i, (gs, ps, audio) in enumerate(generator):
        print(i, gs, ps)
        audio_chunks.append(audio)

    full_audio = np.concatenate(audio_chunks)

    # Play audio
    sd.play(full_audio, samplerate=rate)
    sd.wait()  

    return full_audio



if __name__ == "__main__":
    config_path = os.path.join(koroko_path, "config.json")
    model_path = os.path.join(koroko_path, "pytorch_model.bin")

    pipeline, config = load_koroko_pipeline(model_path, config_path)
    test_text = "Hello world, this is Koroko speaking!"
    infer_and_play(pipeline, config, test_text)

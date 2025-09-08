from kokoro import KPipeline
import soundfile as sf
import torch
import os
import json
import numpy as np
import sounddevice as sd
from huggingface_hub import login
from dotenv import load_dotenv


load_dotenv(override=True)
login(token=os.environ["HF_TOKEN"])


BASE_PATH = "C:/Users/DeLL/Desktop/anything_py/talka-agent/model_bases"
MODEL_DIR = "kokoro/"
kokoro_path = os.path.join(BASE_PATH, MODEL_DIR)
os.makedirs(kokoro_path, exist_ok=True)


def dowload_kokoro_model():
    pipeline = KPipeline(lang_code='a') 
    torch.save(pipeline.model, os.path.join(kokoro_path, "pytorch_model.bin"))

    # Save config
    config = {
        "lang_code": "a",
        "voice": "af_heart"
    }
    json.dump(config, open(os.path.join(kokoro_path, "config.json"), "w"))


def load_kokoro_pipeline():
    config_path = os.path.join(kokoro_path, "config.json")
    model_path = os.path.join(kokoro_path, "pytorch_model.bin")

    with open(config_path) as f:
        config = json.load(f)

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    pipeline = KPipeline(lang_code=config["lang_code"])
    pipeline.model = model
    pipeline.model.eval()

    return pipeline, config


def speech_output(test_text: str):

    pipeline, config = load_kokoro_pipeline()
    generator = pipeline(test_text, voice=config.get("voice", "af_heart"), split_pattern=r'\n+')

    for i, (gs, ps, audio) in enumerate(generator):
        fade_len = min(100, len(audio))  # first 100 samples
        audio[:fade_len] *= np.linspace(0, 1, fade_len)
        
        sd.play(audio, samplerate=24000)
        sd.wait() 

if __name__ == "__main__":
    dowload_kokoro_model()
    
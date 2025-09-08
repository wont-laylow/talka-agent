from faster_whisper import WhisperModel, download_model, available_models
import os

BASE_PATH = "C:/Users/DeLL/Desktop/anything_py/talka-agent/model_bases"
MODEL_DIR = "whisper/"
whisper_path = os.path.join(BASE_PATH, MODEL_DIR)
os.makedirs(whisper_path, exist_ok=True)

print(f"Available Whisper Models: {available_models()}")

def download_whisper_model(model_size="medium.en"):
    model_path = download_model(model_size, output_dir=whisper_path)
    return model_path


def transcribe_audio(audio_path):
    model = WhisperModel(whisper_path, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    return transcription

if __name__ == "__main__":
    download_whisper_model("small.en")




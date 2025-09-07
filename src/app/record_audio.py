import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

def record_audio(filename="input.wav", samplerate=16000):
    print("Recording... press ENTER to stop.")
    frames = []

    def callback(indata, frames_count, time, status):
        if status:
            print(status)
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="int16", callback=callback)
    with stream:
        input()  #
    print("Recording stopped.")

    audio = np.concatenate(frames, axis=0)
    write(filename, samplerate, audio)
    print(f"Saved recording as {filename}")
    return filename
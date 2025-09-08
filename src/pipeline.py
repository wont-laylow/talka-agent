from src.app.brain import brain_reply
from src.app.record_audio import record_audio
from src.app.voice_output import speech_output
from src.app.transcribe import transcribe_audio
import time 


if __name__ == "__main__":
    while True:
        start_time = time.time()
        wav_file = record_audio("input.wav")   
        transcription = transcribe_audio(wav_file)     
        reply = brain_reply(transcription)    
        print(f"AI: {reply}")    
        speech_output(reply) 
        endtime = time.time()
        print(f"Total time taken: {((endtime - start_time)/60):2f} minutes")
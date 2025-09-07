from src.app.brain import brain_reply
from src.app.record_audio import record_audio
from src.app.voice_output import speech_output
from src.app.transcribe import transcribe_audio
import time 


if __name__ == "__main__":
    start_time = time.time()
    wav_file = record_audio("input.wav")   
    transcription = transcribe_audio(wav_file)     
    brain_reply = brain_reply(transcription)        
    speech_output(brain_reply) 
    endtime = time.time()
    print(f"Total time taken: {((endtime - start_time)/60):2f} minutes")
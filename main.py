# Use YAMNet for classifying audios like differentiating if an audio is a bird, human, or a dinosaur
import microphone_transcriber
import desktop_transcriber
from datetime import datetime
from scipy.io.wavfile import write
import os
from vad import setup_vad

# Main loop
def main():
    # Globalize VAD
    model, utils, device = setup_vad()

    start_new_transcript = ""
    while start_new_transcript not in ("y", "n"):
        start_new_transcript = input("Start new transcription? (y/n): ")
        
    if start_new_transcript == "y":
        if os.path.exists("transcript.txt"):
            os.remove("transcript.txt")
        print("[Starting new transcript]")
    else:
        print("[Continuing new transcript]")

    while True:
        # Make sure record() actually works not sequentially with transcribe(). It needs to keep recording despite how long whisper takes to finish transcribing
        audio = microphone_transcriber.record(model, utils, device)

        write("temp.wav", microphone_transcriber.SAMPLE_RATE, audio)

        # Good chance that transcribe() is going to take quite a few seconds to transcribe, so record() needs to work independently
        transcription = microphone_transcriber.transcribe("temp.wav")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}\n")
        with open("transcript.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}\n")

if __name__ == "__main__":
    main()

# Bug: VAD is loading twice, make vad.py so each module shares vad. If you ever call VAD on both at the same time, microphone + desktop, this may be a brief deadlock or a race condition to fix 
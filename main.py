from transcription import record, transcribe, SAMPLE_RATE
from datetime import datetime
from scipy.io.wavfile import write
import os

# Main loop
def main():
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
        audio = record()

        write("temp.wav", SAMPLE_RATE, audio)

        transcription = transcribe("temp.wav")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}\n")
        with open("transcript.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}\n")

if __name__ == "__main__":
    main()
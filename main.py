from transcription import record, transcribe, SAMPLE_RATE
from datetime import datetime
from scipy.io.wavfile import write

# Main loop
def main():
    while True:
        audio = record()

        write("temp.wav", SAMPLE_RATE, audio)

        transcription = transcribe("temp.wav")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}")
        with open("transcript.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}\n")

if __name__ == "__main__":
    main()
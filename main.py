# Use YAMNet for classifying audios like differentiating if an audio is a bird, human, or a dinosaur
import microphone_transcriber
import desktop_transcriber
from datetime import datetime
from scipy.io.wavfile import write
import os
import torch
from queue import Queue
import threading

# Forces record() to work without waiting for transcribe()
def record_worker(audio_queue, model, utils, device):
    while True:
        audio = microphone_transcriber.record(model, utils, device)
        audio_queue.put(audio)

# Main loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    model = model.to(device)

    audio_queue = Queue()
    threading.Thread(target=record_worker, args=(audio_queue, model, utils, device), daemon=True).start()

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
        audio = audio_queue.get()
        
        write("temp.wav", microphone_transcriber.SAMPLE_RATE, audio)

        transcription = microphone_transcriber.transcribe("temp.wav")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}\n")
        with open("transcript.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}\n")

if __name__ == "__main__":
    main()

# Use YAMNet for classifying audios like differentiating if an audio is a bird, human, or a dinosaur
from datetime import datetime
from scipy.io.wavfile import write
import os
import torch
from queue import Queue
import threading

# Initial setup
transcriber_mode = ""
start_new_transcript = ""

# Prompt input for recording type
while transcriber_mode not in ("1", "2"):
        transcriber_mode = input("Press [1] for Microphone, [2] for Desktop transcription: ")

if transcriber_mode == "1":
    import microphone_transcriber as transcriber
else:
    import desktop_transcriber as transcriber

# Prompt input for new transcription
while start_new_transcript not in ("y", "n"):
    start_new_transcript = input("Press [y] for new transcription, [n] for appending: ")
    
if start_new_transcript == "y":
    if os.path.exists("transcript.txt"):
        os.remove("transcript.txt")

# Forces record() to work without waiting for transcribe()
def record_worker(audio_queue, model, utils, device):
    while True:
        audio = transcriber.record(model, utils, device)
        audio_queue.put(audio)

# Main loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    model = model.to(device)

    audio_queue = Queue()

    threading.Thread(target=record_worker, args=(audio_queue, model, utils, device), daemon=True).start()

    while True:
        audio = audio_queue.get()
        
        write("temp.wav", transcriber.SAMPLE_RATE, audio)

        transcription = transcriber.transcribe("temp.wav")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}\n")
        with open("transcript.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}]:{transcription.text}\n")

if __name__ == "__main__":
    main()
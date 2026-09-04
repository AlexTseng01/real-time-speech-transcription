import os
from groq import Groq
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import time
from scipy.io.wavfile import write
from datetime import datetime
import torch

# Settings
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30 # ms
SILENCE_DURATION = 2.0

# Initial setup
load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
model = model.to(device)

# Runs the audio file on the whisper-large-v3-turbo model and gets a string output
def transcribe(audio_file):
    with open(audio_file, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file, 
            model="whisper-large-v3"
        )
    return transcription

# VAD Model expects float32 tensors, 16kHz, chunks of 512 samples (32ms)
def is_speech_silero(audio_int16_chunk):
    audio_float32 = torch.from_numpy(audio_int16_chunk.astype(np.float32) / 32768.0).to(device)
    with torch.no_grad():
        speech_prob = model(audio_float32, SAMPLE_RATE).item()

    return speech_prob > 0.3

# Well... what else is there to say here?
def record():
    frame_size = 512
    recording = [] # Contains small chunks of audio, each is FRAME_DURATION long
    speaking = False # Becomes true if speaking is detected
    silence_start = None # Remember when the silence begins

    # Turns on microphone 
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", blocksize=frame_size) as stream:
        while True:
            audio, overflowed = stream.read(frame_size) # Possibly blocking
            audio = audio[:, 0] # Removes the channel dimension to become 1D
            is_speech = is_speech_silero(audio) # VAD doing its job

            # VAD detects audio
            volume = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if is_speech:
                if not speaking:
                    speaking = not speaking
                recording.append(audio.copy())
                silence_start = None
            elif speaking:
                recording.append(audio.copy())
                if silence_start is None:
                    silence_start = time.time()
                else:
                    silence_elapsed = time.time() - silence_start
                    print(f"\r[Elapsed]: {silence_elapsed:.2f} / {SILENCE_DURATION:.2f} sec", end="", flush=True)

                    if silence_elapsed >= SILENCE_DURATION:
                        print("\r" + " " * 50 + "\r", end="", flush=True)
                        break
                    
    return np.concatenate(recording)
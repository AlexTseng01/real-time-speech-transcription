import os
from groq import Groq
from dotenv import load_dotenv
import numpy as np
import time
from scipy.signal import butter, sosfilt
import torch
import pyaudiowpatch as pyaudio

# Settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
FRAME_DURATION = 32
SILENCE_DURATION = 2.0
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000) # 512
THRESHOLD = 50

# Initial setup
load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
model = model.to(device)

# Runs the audio file on the whisper-large-v3 model and gets a string output
def transcribe(audio_file):
    with open(audio_file, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file, 
            model="whisper-large-v3",
            language="en",
            temperature=0
        )
    return transcription

# VAD Model expects float32 tensors, 16kHz, chunks of 512 samples (32ms)
def is_speech_silero(audio_int16_chunk):
    audio_float32 = torch.from_numpy(audio_int16_chunk.astype(np.float32) / 32768.0).to(device)
    with torch.no_grad():
        speech_prob = model(audio_float32, SAMPLE_RATE).item()
    return speech_prob > 0.3

# Record desktop audio
def record():
    recording = []
    speaking = False
    silence_start = None

    p = pyaudio.PyAudio()

    device = p.get_device_info_by_index(23)

    rate = int(device["defaultSampleRate"])
    channels = device["maxInputChannels"]
    chunk = 1536  # 32 ms at 48 kHz

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=device["index"],
        frames_per_buffer=chunk
    )

    try:
        while True:
            data = stream.read(chunk)

            audio = np.frombuffer(data, dtype=np.int16)
            audio = audio.reshape(-1, channels)
            audio = audio.mean(axis=1).astype(np.int16)
            audio = audio[::3]

            is_speech = is_speech_silero(audio)

            # VAD detects audio
            volume = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if is_speech:
                if not speaking:
                    print("Beginning of speech detected")
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
            # Try to skip transcribing any empty audios
            if is_empty_audio(audio):
                continue
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    # Append 0.1 seconds of empty audio to beginning
    recording = np.concatenate(recording)

    silence = np.zeros(
        int(SAMPLE_RATE * 0.1),
        dtype=np.int16
    )

    recording = np.concatenate([silence, recording])

    return recording

# Checks if an audio numpy array is below the volume threshold
def is_empty_audio(audio):
    average_volume = np.mean(np.abs(audio))
    return average_volume < THRESHOLD
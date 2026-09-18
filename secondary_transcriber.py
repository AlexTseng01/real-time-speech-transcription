import os
from groq import Groq
from dotenv import load_dotenv
import numpy as np
import time
import torch
import pyaudiowpatch as pyaudio

# Settings
SAMPLE_RATE = 16000
FRAME_DURATION = 32
SILENCE_DURATION = 2.0 # If silence is detected for this long, finish recording
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
THRESHOLD = 50 # If volume falls below this number, trash the audio
OUTPUT_DEVICE = "Headset Earphone (JBL Quantum One Chat) [Loopback]" # A secondary output device found in JBL headset products
WHISPER_MODEL = "whisper-large-v3-turbo"

# Initial setup
load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

model = utils = device = None

# Runs the audio file on the whisper-large-v3 model and gets a string output
def transcribe(audio_file):
    with open(audio_file, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file, 
            model=WHISPER_MODEL,
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

# Make sure your secondary output device actually exists
def find_device(p):
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["name"] == OUTPUT_DEVICE:
            return i
    return None

def find_output_device():
    try:
        p = pyaudio.PyAudio()
    except Exception as e:
        print(f"Could not initialize PyAudio: {e}")
        return None, None

    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)

            if info["name"] == OUTPUT_DEVICE:
                return p, i

        print(f"Output device not found: {OUTPUT_DEVICE}")
        p.terminate()
        return None, None

    except Exception:
        p.terminate()
        raise

# Record desktop audio
def record(vad_model, vad_utils, vad_device):
    global model, utils, device

    model = vad_model
    utils = vad_utils
    device = vad_device

    recording = []
    speaking = False
    silence_start = None

    p, device_index = find_output_device()

    if p is None:
        return None

    audio_device = p.get_device_info_by_index(device_index)

    rate = int(audio_device["defaultSampleRate"])
    channels = audio_device["maxInputChannels"]
    chunk = 1536

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=audio_device["index"],
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

            # Skip audio below some volume threshold
            if is_empty_audio(audio):
                continue
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    recording = np.concatenate(recording)

    # Add 0.5 seconds to the beginning of the audio
    silence = np.zeros(int(SAMPLE_RATE * 0.5),dtype=np.int16)

    recording = np.concatenate([silence, recording])

    return recording

# Checks if an audio numpy array is below the volume threshold
def is_empty_audio(audio):
    average_volume = np.mean(np.abs(audio))
    return average_volume < THRESHOLD
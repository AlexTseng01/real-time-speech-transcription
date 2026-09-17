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
SILENCE_DURATION = 2.0 # How long to wait until finish recording
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
THRESHOLD = 50 # Trashes any recordings that are below this volume
GAME_DEVICE_INDEX = 23 # JBL Quantum One Gaming
CHAT_DEVICE_INDEX = 24 # JBL Quantum One Chat

# Initial setup
load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

model = utils = device = None

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
def record(vad_model, vad_utils, vad_device):
    global model, utils, device

    model = vad_model
    utils = vad_utils
    device = vad_device

    recording = []
    speaking = False
    silence_start = None

    p = pyaudio.PyAudio()

    game_device = p.get_device_info_by_index(GAME_DEVICE_INDEX)
    chat_device = p.get_device_info_by_index(CHAT_DEVICE_INDEX)

    rate = int(game_device["defaultSampleRate"])
    chunk = 1536  # 32 ms at 48 kHz

    game_channels = game_device["maxInputChannels"]
    chat_channels = chat_device["maxInputChannels"]

    # Open desktop device JBL Quantum One Gaming
    game_stream = p.open(
        format=pyaudio.paInt16,
        channels=game_channels,
        rate=rate,
        input=True,
        input_device_index=23,
        frames_per_buffer=chunk
    )

    # Open desktop device JBL Quantum One Chat
    chat_stream = p.open(
        format=pyaudio.paInt16,
        channels=chat_channels,
        rate=rate,
        input=True,
        input_device_index=24,
        frames_per_buffer=chunk
    )

    try:
        while True:
            game_data = game_stream.read(chunk)
            chat_data = chat_stream.read(chunk)

            game_audio = np.frombuffer(game_data, dtype=np.int16)
            chat_audio = np.frombuffer(chat_data, dtype=np.int16)

            game_audio = game_audio.reshape(-1, game_channels)
            chat_audio = chat_audio.reshape(-1, chat_channels)

            game_audio = game_audio.mean(axis=1)
            chat_audio = chat_audio.mean(axis=1)

            # Combining Game + Chat audio
            audio = ((game_audio.astype(np.int32) + chat_audio.astype(np.int32)) / 2).astype(np.int16)

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
            # Try to skip transcribing any empty audios
            if is_empty_audio(audio):
                continue
    finally:
        game_stream.stop_stream()
        game_stream.close()

        chat_stream.stop_stream()
        chat_stream.close()

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
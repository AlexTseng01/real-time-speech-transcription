
# Real-time speech transcription POC

This program serves as a proof of concept for a larger project that I plan to develop. Its primary function is to transcribe speech in real time. In the future, this POC is intended to become a modular component of another project.
## Features

List of features:
- Real-time speech transcription
## Upcoming Features

List of upcoming features:
- Diarization
- Speaker identification
- Use semantics to determine wake-up
- Fine-tuning Silero VAD model until it can detect real speech up to 30 feet away
- Fine-tuning Whisper model for higher English-transcription accuracy
- Conversational interrupts
- (Pray) for more optimizations
## Issues

List of known issues:
- Silence is interpreted by Whisper as "Thank you" and "you"

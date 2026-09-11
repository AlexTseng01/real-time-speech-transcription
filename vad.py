import torch

# Avoid loading Silero VAD twice for no reason
def setup_vad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    model = model.to(device)

    return model, utils, device
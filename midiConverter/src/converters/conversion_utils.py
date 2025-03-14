import librosa
import numpy as np
import torch

from src.models.constants import DRUM_INSTRUMENTS


def extract_true_label(file_name: str) -> str:
    name_lower = file_name.lower()
    if "kick" in name_lower:
        return DRUM_INSTRUMENTS[4]
    elif "snare" in name_lower:
        return DRUM_INSTRUMENTS[0]
    elif "hi-hat" in name_lower or "hihat" in name_lower:
        return DRUM_INSTRUMENTS[3]
    elif "crash" in name_lower:
        return DRUM_INSTRUMENTS[1]
    elif "rack_tom" in name_lower:
        return DRUM_INSTRUMENTS[5]
    elif "floor_tom" in name_lower:
        return DRUM_INSTRUMENTS[2]
    elif "ride" in name_lower:
        return DRUM_INSTRUMENTS[6]
    return "unknown"


def set_length(audio: np.ndarray, desired_length: int) -> np.ndarray:
    if len(audio) > desired_length:
        audio = audio[:desired_length]
    else:
        audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')
    return audio


def classify_instrument_dl_mel(path, model, sr=22050, duration=2, use_ensemble=False, use_mel_model=False):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = set_length(y, sr * duration)
    
    if use_ensemble:
        pred_idx = model.predict(y, sr)
    elif use_mel_model:
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=256,
            hop_length=256,
            n_fft=2048
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        x = torch.tensor(mel_db, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(x)
            pred_idx = outputs.argmax(dim=1).item()
    else:
        cqt = librosa.cqt(y, sr=sr, hop_length=512)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        x = torch.tensor(cqt_db, dtype=torch.float).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            outputs = model(x)
            pred_idx = outputs.argmax(dim=1).item()

    label_map = {i: inst for i, inst in enumerate(DRUM_INSTRUMENTS)}
    return label_map[pred_idx]
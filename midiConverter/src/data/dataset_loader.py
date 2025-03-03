import os
import librosa
import numpy as np

from src.models.constants import DRUM_INSTRUMENTS, INSTRUMENTS_DATA_SET_PATH


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val < 1e-8:
        return audio
    return audio / max_val

def set_length(audio: np.ndarray, desired_length: int) -> np.ndarray:
    if len(audio) > desired_length:
        audio = audio[:desired_length]
    else:
        audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')
    return audio

def load_subset(subset_path: str, instruments: list[str], sr: int = 22050) -> list[tuple[np.ndarray, str]]:
    data = []
    desired_length = sr * 2
    for inst in instruments:
        folder_path = os.path.join(subset_path, inst)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                audio, _ = librosa.load(file_path, sr=sr, mono=True)
                audio = set_length(audio, desired_length)
                audio = normalize_audio(audio)
                data.append((audio, inst))
    return data

def load_dataset_with_splits(dataset_path: str = INSTRUMENTS_DATA_SET_PATH,
                             instruments: list[str] = DRUM_INSTRUMENTS,
                             sr: int = 22050):
    train_data = load_subset(os.path.join(dataset_path, "train"), instruments, sr)
    val_data   = load_subset(os.path.join(dataset_path, "val"), instruments, sr)
    test_data  = load_subset(os.path.join(dataset_path, "test"), instruments, sr)
    return train_data, val_data, test_data

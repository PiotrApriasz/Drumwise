import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

from src.data.feature_extraction.spectogram_generator import generate_cqt_spectrogram, generate_mel_spectrogram


class CqtDrumDataset(Dataset):
    def __init__(self, data, label_map, sr=22050):
        self.data = data
        self.sr = sr
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, label_str = self.data[idx]
        label = self.label_map[label_str]
        cqt_db = generate_cqt_spectrogram(y=audio, sr=self.sr)
        x = torch.tensor(cqt_db, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class MelDrumDataset(Dataset):
    def __init__(self, data, label_map, sr=22050):
        self.data = data
        self.sr = sr
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, label_str = self.data[idx]
        label = self.label_map[label_str]

        mel_db = generate_mel_spectrogram(y=audio, sr=self.sr)

        x = torch.tensor(mel_db, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
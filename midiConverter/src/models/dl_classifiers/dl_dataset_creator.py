import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


class DrumDataset(Dataset):
    def __init__(self, data, label_map, sr=22050):
        self.data = data
        self.sr = sr
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, label_str = self.data[idx]
        label = self.label_map[label_str]
        cqt = librosa.cqt(audio, sr=self.sr, hop_length=512)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        x = torch.tensor(cqt_db, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data.feature_extraction.spectogram_generator import generate_mel_spectrogram, generate_cqt_spectrogram
from src.models.dl_classifiers.classifiers.cnn_classifier import DrumCNN
from src.constants import DRUM_INSTRUMENTS, TRAINED_DL_MODELS_PATH, CQT_BEST_CNN_MODEL_NAME
from src.data.dataset.single_label_dataset_loader import load_dataset_with_splits
from src.models.dl_classifiers.dl_dataset_creator import CqtDrumDataset


class HeterogeneousCnnEnsemble:
    def __init__(self, model_paths, model_types, device=None):
        self.models = []
        self.model_types = model_types
        
        for path, model_type in zip(model_paths, model_types):
            model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for heterogeneous ensemble prediction")
    
    def preprocess_audio(self, audio, sr=22050, model_type='cqt'):
        if model_type == 'mel':
            mel_db = generate_mel_spectrogram(y=audio, sr=sr)
            return torch.tensor(mel_db, dtype=torch.float).unsqueeze(0)
        else:
            cqt_db = generate_cqt_spectrogram(y=audio, sr=sr)
            return torch.tensor(cqt_db, dtype=torch.float).unsqueeze(0)
    
    def predict(self, audio, sr=22050):
        all_logits = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                x = self.preprocess_audio(audio, sr, self.model_types[i])
                x = x.unsqueeze(0)
                logits = model(x)
                all_logits.append(logits)

        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        return torch.argmax(avg_logits, dim=1).item()
    
    def predict_proba(self, audio, sr=22050):
        all_probs = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                x = self.preprocess_audio(audio, sr, self.model_types[i])
                x = x.unsqueeze(0)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        return avg_probs.cpu().numpy()[0]

    def predict_with_confidence(self, audio, sr=22050):
        all_probs = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                x = self.preprocess_audio(audio, sr, self.model_types[i])
                x = x.unsqueeze(0)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        avg_probs = torch.mean(torch.stack(all_probs), dim=0)

        confidence, pred_class = torch.max(avg_probs, dim=1)
        
        return pred_class.item(), confidence.item()
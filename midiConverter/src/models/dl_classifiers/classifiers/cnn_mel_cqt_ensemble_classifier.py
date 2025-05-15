import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data.feature_extraction.spectogram_generator import generate_mel_spectrogram, generate_cqt_spectrogram
from src.models.dl_classifiers.classifiers.cnn_classifier import DrumCNN
from src.constants import DRUM_INSTRUMENTS, TRAINED_DL_MODELS_PATH, CQT_BEST_CNN_MODEL_NAME

label_map = {inst: i for i, inst in enumerate(DRUM_INSTRUMENTS)}

class CnnMelCqtEnsemble:
    def __init__(self, model_paths, model_types, device=None):
        self.models = []
        self.model_types = model_types
        
        for path, model_type in zip(model_paths, model_types):
            model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for cnn_mel_cqt_ensemble ensemble prediction")
    
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

    def predict_and_get_avg_logits(self, audio, sr=22050):
        all_logits_tensors = []
        with torch.no_grad():
            for i, model in enumerate(self.models):
                x = self.preprocess_audio(audio, sr, self.model_types[i])
                x = x.unsqueeze(0)
                logits = model(x)
                all_logits_tensors.append(logits)

        avg_logits = torch.mean(torch.stack(all_logits_tensors), dim=0)  # Shape: [1, num_classes]
        pred_class = torch.argmax(avg_logits, dim=1).item()
        return pred_class, avg_logits

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

    def evaluate_on_test_data(self, test_audio_data, criterion, sr=22050):
        for model in self.models:
            model.eval()

        total_loss = 0
        correct_predictions = 0
        total_samples = len(test_audio_data)

        all_true_labels = []
        all_predicted_labels = []
        all_avg_logits_outputs = []

        if total_samples == 0:
            return 0.0, 0.0, [], [], []

        with torch.no_grad():
            for audio_sample, true_label_str in test_audio_data:
                true_label_idx = label_map[true_label_str]
                all_true_labels.append(true_label_idx)

                predicted_class_idx, avg_logits = self.predict_and_get_avg_logits(audio_sample, sr)
                all_predicted_labels.append(predicted_class_idx)
                all_avg_logits_outputs.append(avg_logits.cpu().numpy())

                if predicted_class_idx == true_label_idx:
                    correct_predictions += 1

                true_label_tensor = torch.tensor([true_label_idx], dtype=torch.long)
                loss = criterion(avg_logits, true_label_tensor)
                total_loss += loss.item()

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        return average_loss, accuracy, all_true_labels, all_predicted_labels, all_avg_logits_outputs

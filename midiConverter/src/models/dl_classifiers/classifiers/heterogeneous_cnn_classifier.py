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


def evaluate_ensemble(model_paths):
    ensemble = HeterogeneousCnnEnsemble(model_paths, model_types=['cqt'] * len(model_paths))

    label_map = {inst: i for i, inst in enumerate(DRUM_INSTRUMENTS)}
    _, _, test_data = load_dataset_with_splits()
    test_dataset = CqtDrumDataset(test_data, label_map, sr=22050)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []

    for x_test, y_test in test_loader:
        pred = ensemble.predict(x_test)
        all_preds.append(pred)
        all_labels.append(y_test.item())

    test_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("Ensemble Test Accuracy:", test_acc)
    print("Confusion Matrix:\n", cm)

    print("\nPer-class accuracy:")
    for i, instrument in enumerate(DRUM_INSTRUMENTS):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        if class_total > 0:
            print(f"{instrument}: {class_correct / class_total:.4f}")

    return {
        "test_acc": test_acc,
        "confusion_matrix": cm
    }


# TODO: Below code move to another file

def train_ensemble(num_models=3):

    model_paths = []
    
    for i in range(num_models):
        print(f"\n--- Training Model {i+1}/{num_models} ---\n")

        model_path = os.path.join(TRAINED_DL_MODELS_PATH, f"ensemble_model_{i+1}.pth")
        best_model_path = os.path.join(TRAINED_DL_MODELS_PATH, f"best_ensemble_model_{i+1}.pth")

        torch.manual_seed(42 + i)
        np.random.seed(42 + i)

        from src.models.dl_classifiers.classifiers.cnn_classifier import train_cnn_classifier
        train_cnn_classifier(model_path, best_model_path)
        
        model_paths.append(best_model_path)
    
    return model_paths


def compare_ensemble_vs_single(ensemble_paths, single_model_path, device=None):

    ensemble = HeterogeneousCnnEnsemble(ensemble_paths, model_types=['cqt'] * len(ensemble_paths))

    
    single_model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
    single_model.load_state_dict(torch.load(single_model_path))
    single_model.eval()

    label_map = {inst: i for i, inst in enumerate(DRUM_INSTRUMENTS)}
    _, _, test_data = load_dataset_with_splits()
    test_dataset = CqtDrumDataset(test_data, label_map, sr=22050)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ensemble_preds = []
    single_preds = []
    all_labels = []
    
    for x_test, y_test in test_loader:

        ensemble_pred = ensemble.predict(x_test)
        ensemble_preds.append(ensemble_pred)

        with torch.no_grad():
            single_output = single_model(x_test)
            single_pred = torch.argmax(single_output, dim=1).item()
        single_preds.append(single_pred)
        
        all_labels.append(y_test.item())

    ensemble_acc = accuracy_score(all_labels, ensemble_preds)
    single_acc = accuracy_score(all_labels, single_preds)
    
    ensemble_cm = confusion_matrix(all_labels, ensemble_preds)
    single_cm = confusion_matrix(all_labels, single_preds)
    
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"Single Model Accuracy: {single_acc:.4f}")
    print(f"Improvement: {(ensemble_acc - single_acc) * 100:.2f}%")

    correct_ensemble = np.array(ensemble_preds) == np.array(all_labels)
    correct_single = np.array(single_preds) == np.array(all_labels)
    
    ensemble_better = np.logical_and(correct_ensemble, np.logical_not(correct_single))
    single_better = np.logical_and(correct_single, np.logical_not(correct_ensemble))
    
    print(f"\nCases where ensemble is correct but single model is wrong: {np.sum(ensemble_better)}")
    print(f"Cases where single model is correct but ensemble is wrong: {np.sum(single_better)}")

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(ensemble_cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=DRUM_INSTRUMENTS, 
                yticklabels=DRUM_INSTRUMENTS)
    plt.title(f"Ensemble Confusion Matrix (Acc: {ensemble_acc:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(single_cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=DRUM_INSTRUMENTS, 
                yticklabels=DRUM_INSTRUMENTS)
    plt.title(f"Single Model Confusion Matrix (Acc: {single_acc:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.tight_layout()
    plt.savefig("ensemble_vs_single.png", dpi=300)
    plt.close()
    
    return {
        "ensemble_acc": ensemble_acc,
        "single_acc": single_acc,
        "ensemble_cm": ensemble_cm,
        "single_cm": single_cm
    }


if __name__ == "__main__":
    model_paths = train_ensemble(num_models=3)

    ensemble_metrics = evaluate_ensemble(model_paths)

    compare_metrics = compare_ensemble_vs_single(model_paths,
                                                 os.path.join(TRAINED_DL_MODELS_PATH, CQT_BEST_CNN_MODEL_NAME))
    
    print("\nEnsemble training and evaluation complete!")
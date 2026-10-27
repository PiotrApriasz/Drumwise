import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix, f1_score
import os
import time

from src.baseline.data.dataset.dataset_augumentation import create_augmented_dataset
from src.baseline.data.feature_extraction.spectogram_generator import generate_mel_spectrogram
from src.constants import DRUM_INSTRUMENTS, INSTRUMENTS_DATA_SET_PATH
from src.baseline.data.dataset.single_label_dataset_loader import load_subset
from src.reproducibility import get_device

label_map = {inst: i for i, inst in enumerate(DRUM_INSTRUMENTS)}


# --- Dataset Classes ---

class BaseDrumDataset(Dataset):
    """Base dataset class for drum sounds."""

    def __init__(self, data, label_map, sr=22050):
        self.data = data
        self.sr = sr
        self.label_map = label_map
        if not data:
            print("Warning: Initializing BaseDrumDataset with empty data.")
        elif not isinstance(data[0], (tuple, list)) or len(data[0]) != 2:
            print(
                f"Warning: Data format might be incorrect. Expected list of (audio, label_str), got first element: {data[0]}")

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def _get_base_item(self, idx):
        """Helper to get audio and label."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.data)}")
        audio, label_str = self.data[idx]
        if label_str not in self.label_map:
            raise ValueError(f"Label '{label_str}' not found in label_map: {self.label_map.keys()}")
        label = self.label_map[label_str]
        return audio, label


class MelDrumDataset(BaseDrumDataset):
    """Dataset generating Mel spectrograms."""

    def __getitem__(self, idx):
        audio, label = self._get_base_item(idx)
        mel_db = generate_mel_spectrogram(y=audio, sr=self.sr)
        x = torch.tensor(mel_db, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# --- Training helpers ---

def compute_class_weights(labels, num_classes):
    """Inverse-frequency ("balanced") class weights: n_samples / (n_present_classes * count) (review item D5).

    Each present class then contributes the same total weight; absent classes get 0.
    """
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float)


class EarlyStopping:
    """Tracks the best validation score; an improving epoch resets the counter (review item B5)."""

    def __init__(self, patience):
        self.patience = patience
        self.best = float('-inf')
        self.epochs_no_improve = 0

    def step(self, score) -> bool:
        """Records one epoch and returns True when `score` is a new best."""
        if score > self.best:
            self.best = score
            self.epochs_no_improve = 0
            return True
        self.epochs_no_improve += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_no_improve >= self.patience


# --- Abstract Base Classifier ---

class BaseClassifier(nn.Module, ABC):
    def __init__(self, num_classes=len(DRUM_INSTRUMENTS), sr=22050, batch_size=8, num_workers=4):
        super().__init__()
        self.num_classes = num_classes
        self.sr = sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = self._build_model()
        self.label_map = label_map
        self.device = get_device()
        self.to(self.device)

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Subclasses must implement this to define the model architecture."""
        pass

    def forward(self, x):
        """Forward pass delegates to the underlying model."""
        return self.model(x)

    def _create_datasets(self, augment_factor=2):
        """Loads raw data and creates train and validation datasets. The test split is never loaded here."""
        train_data = load_subset(os.path.join(INSTRUMENTS_DATA_SET_PATH, "train"), DRUM_INSTRUMENTS, self.sr)
        val_data = load_subset(os.path.join(INSTRUMENTS_DATA_SET_PATH, "val"), DRUM_INSTRUMENTS, self.sr)
        if augment_factor > 0:
            print(f"Creating augmented dataset with factor {augment_factor}...")
            train_data_aug = create_augmented_dataset(train_data, sr=self.sr, augment_factor=augment_factor)
        else:
            train_data_aug = train_data
            print("Skipping augmentation.")

        train_dataset = MelDrumDataset(train_data_aug, self.label_map, sr=self.sr)
        val_dataset = MelDrumDataset(val_data, self.label_map, sr=self.sr)

        print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
        return train_dataset, val_dataset

    def _create_test_dataset(self):
        """Loads only the test split (review item B11: no augmented training set is built for evaluation)."""
        test_data = load_subset(os.path.join(INSTRUMENTS_DATA_SET_PATH, "test"), DRUM_INSTRUMENTS, self.sr)
        return MelDrumDataset(test_data, self.label_map, sr=self.sr)

    def _create_loader(self, dataset, shuffle):
        # macOS starts workers with spawn, which pickles the in-memory dataset; persistent workers do it once
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def _train_epoch(self, train_loader, optimizer, criterion):
        """Runs a single training epoch."""
        self.model.train()
        train_loss = 0
        epoch_start_time = time.time()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        epoch_duration = time.time() - epoch_start_time
        return train_loss / len(train_loader), epoch_duration

    def _evaluate(self, data_loader, criterion):
        """Evaluates the model on a given dataloader."""
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []
        all_outputs = []
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(x_batch)
                total_loss += criterion(outputs, y_batch).item()

                all_outputs.extend(outputs.cpu().numpy())
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0
        return avg_loss, accuracy, all_labels, all_preds, all_outputs

    def train_model(self, save_path, best_model_save_path, optimizer, criterion, scheduler=None,
                    epochs=50, patience=6, augment_factor=2, class_weighted=True):
        """Full training loop with validation, early stopping on macro-F1, and model saving."""

        train_dataset, val_dataset = self._create_datasets(augment_factor=augment_factor)
        train_loader = self._create_loader(train_dataset, shuffle=True)
        val_loader = self._create_loader(val_dataset, shuffle=False)

        if class_weighted and isinstance(criterion, nn.CrossEntropyLoss):
            train_labels = [self.label_map[label] for _, label in train_dataset.data]
            criterion.weight = compute_class_weights(train_labels, self.num_classes)
            print(f"Class weights: {criterion.weight.numpy().round(3).tolist()}")
        criterion.to(self.device)

        early_stopping = EarlyStopping(patience)
        best_val_acc = 0

        train_losses = []
        val_losses = []
        val_accs = []
        val_f1s = []
        epoch_times = []

        for epoch in range(epochs):
            train_loss, epoch_duration = self._train_epoch(train_loader, optimizer, criterion)
            train_losses.append(train_loss)
            epoch_times.append(epoch_duration)

            val_loss, val_acc, val_labels, val_preds, _ = self._evaluate(val_loader, criterion)
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}, Epoch Time: {epoch_duration:.2f}s")

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if early_stopping.step(val_f1):
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                torch.save(self.model.state_dict(), best_model_save_path)
                print(f"Saved best model (Val Macro-F1: {val_f1:.4f}) to {best_model_save_path}")

            print(f"Epochs without validation macro-F1 improvement: {early_stopping.epochs_no_improve}/{patience}")
            if early_stopping.should_stop:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(f"Training finished. Best validation macro-F1: {early_stopping.best:.4f}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved final model state to {save_path}")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "val_f1s": val_f1s,
            "best_val_f1": early_stopping.best,
            "best_val_acc": best_val_acc,
            "epoch_times": epoch_times
        }

    def evaluate_model(self, model_path, criterion):
        """Loads a trained model and evaluates it on the test set."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        criterion.to(self.device)

        test_loader = self._create_loader(self._create_test_dataset(), shuffle=False)
        test_loss, test_acc, all_labels, all_preds, all_outputs = self._evaluate(test_loader, criterion)

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.num_classes)))
        class_names = list(DRUM_INSTRUMENTS)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("\nPer-class accuracy:")
        for i, instrument in enumerate(class_names):
            class_correct = cm[i, i]
            class_total = cm[i, :].sum()
            accuracy = class_correct / class_total if class_total > 0 else 0
            print(f"{instrument}: {accuracy:.4f} ({class_correct}/{class_total})")

        return {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "labels": all_labels,
            "predictions": all_preds,
            "outputs": all_outputs
        }

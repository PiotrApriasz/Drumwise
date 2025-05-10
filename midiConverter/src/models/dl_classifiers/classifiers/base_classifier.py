import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
import os
import time  # Added for timing

from src.data.dataset.dataset_augumentation import create_augmented_dataset
from src.data.feature_extraction.spectogram_generator import generate_cqt_spectrogram, generate_mel_spectrogram
from src.constants import DRUM_INSTRUMENTS
from src.data.dataset.single_label_dataset_loader import load_dataset_with_splits

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


class CqtDrumDataset(BaseDrumDataset):
    """Dataset generating CQT spectrograms."""

    def __getitem__(self, idx):
        audio, label = self._get_base_item(idx)
        cqt_db = generate_cqt_spectrogram(y=audio, sr=self.sr)
        x = torch.tensor(cqt_db, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class MelDrumDataset(BaseDrumDataset):
    """Dataset generating Mel spectrograms."""

    def __getitem__(self, idx):
        audio, label = self._get_base_item(idx)
        mel_db = generate_mel_spectrogram(y=audio, sr=self.sr)
        x = torch.tensor(mel_db, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# --- Abstract Base Classifier ---

class BaseClassifier(nn.Module, ABC):
    def __init__(self, num_classes=len(DRUM_INSTRUMENTS), sr=22050, feature_type='cqt', batch_size=8, num_workers=4):
        super().__init__()
        self.num_classes = num_classes
        self.sr = sr
        self.feature_type = feature_type.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = self._build_model()
        self.label_map = label_map

        if self.feature_type not in ['cqt', 'mel']:
            raise ValueError("feature_type must be 'cqt' or 'mel'")

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Subclasses must implement this to define the model architecture."""
        pass

    def forward(self, x):
        """Forward pass delegates to the underlying model."""
        return self.model(x)

    def _get_dataset_class(self):
        """Returns the appropriate Dataset class based on feature_type."""
        if self.feature_type == 'cqt':
            return CqtDrumDataset
        elif self.feature_type == 'mel':
            return MelDrumDataset
        else:
            raise ValueError("Invalid feature_type specified.")

    def _create_datasets(self, augment_factor=2):
        """Loads raw data and creates train, validation, and test datasets."""
        train_data, val_data, test_data = load_dataset_with_splits()
        if augment_factor > 0:
            print(f"Creating augmented dataset with factor {augment_factor}...")
            train_data_aug = create_augmented_dataset(train_data, sr=self.sr, augment_factor=augment_factor)
        else:
            train_data_aug = train_data
            print("Skipping augmentation.")

        DatasetClass = self._get_dataset_class()
        train_dataset = DatasetClass(train_data_aug, self.label_map, sr=self.sr)
        val_dataset = DatasetClass(val_data, self.label_map, sr=self.sr)
        test_dataset = DatasetClass(test_data, self.label_map, sr=self.sr)

        print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset

    def _create_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """Creates DataLoaders for the datasets."""
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers) if train_dataset is not None else None
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers) if val_dataset is not None else None
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers) if test_dataset is not None else None
        return train_loader, val_loader, test_loader

    def _train_epoch(self, train_loader, optimizer, criterion):
        """Runs a single training epoch."""
        self.model.train()
        train_loss = 0
        epoch_start_time = time.time()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = self.model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        return train_loss / len(train_loader), epoch_duration

    def _evaluate(self, data_loader, criterion):
        """Evaluates the model on a given dataloader."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_outputs = []
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

                all_outputs.extend(outputs.cpu().numpy())

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy, all_labels, all_preds, all_outputs

    def train_model(self, save_path, best_model_save_path, optimizer, criterion, scheduler=None,
                    epochs=50, patience=6, augment_factor=2):
        """Full training loop with validation, early stopping, and model saving."""

        train_dataset, val_dataset, test_dataset = self._create_datasets(augment_factor=augment_factor)
        train_loader, val_loader, _ = self._create_dataloaders(train_dataset, val_dataset, test_dataset)

        best_val_acc = 0
        best_val_loss = float('inf')
        epochs_no_improve = 0

        train_losses = []
        val_losses = []
        val_accs = []
        epoch_times = []

        for epoch in range(epochs):
            train_loss, epoch_duration = self._train_epoch(train_loader, optimizer, criterion)
            train_losses.append(train_loss)
            epoch_times.append(epoch_duration)

            val_loss, val_acc, _, _, _ = self._evaluate(val_loader, criterion)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Epoch Time: {epoch_duration:.2f}s")

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_model_save_path)
                print(f"Saved best model (Val Acc: {best_val_acc:.4f}) to {best_model_save_path}")
                epochs_no_improve = 0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc <= best_val_acc and epoch > 0:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0

            print(f"Epochs without validation accuracy improvement: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved final model state to {save_path}")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "best_val_acc": best_val_acc,
            "epoch_times": epoch_times
        }

    def evaluate_model(self, model_path, criterion):
        """Loads a trained model and evaluates it on the test set."""
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        _, _, test_dataset = self._create_datasets(augment_factor=0)
        _, _, test_loader = self._create_dataloaders(None, None, test_dataset)

        test_loss, test_acc, all_labels, all_preds, all_outputs = self._evaluate(test_loader, criterion)

        cm = confusion_matrix(all_labels, all_preds)
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
            "confusion_matrix": cm,
            "labels": all_labels,
            "predictions": all_preds,
            "outputs": all_outputs
        }
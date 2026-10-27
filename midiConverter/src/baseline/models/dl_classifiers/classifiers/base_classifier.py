import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix, f1_score
import os
import time

from src.baseline.data.dataset.augmentation import AugmentConfig, augment_window, jitter_onset
from src.baseline.data.dataset.single_label_dataset_loader import Hit, load_subset
from src.baseline.data.windowing import WindowConfig, cut_window, peak_normalize
from src.baseline.models.dl_classifiers.classifiers.checkpoints import load_checkpoint, save_checkpoint
from src.constants import DRUM_INSTRUMENTS, INSTRUMENTS_DATA_SET_PATH
from src.data.features import LogMelConfig, log_mel
from src.reproducibility import get_device

label_map = {inst: i for i, inst in enumerate(DRUM_INSTRUMENTS)}


# --- Dataset ---

def window_features(window: np.ndarray, features: LogMelConfig) -> torch.Tensor:
    """Peak-normalizes one onset window and returns its ``(1, n_mels, frames)`` log-mel tensor."""
    return torch.from_numpy(log_mel(peak_normalize(window), features)).unsqueeze(0)


class WindowDataset(Dataset):
    """Log-mel features of the window around each hit's onset, optionally augmented online."""

    def __init__(self, hits: list[Hit], label_map, window: WindowConfig, features: LogMelConfig,
                 augment: AugmentConfig = None):
        if window.sr != features.sr:
            raise ValueError(f"window sr {window.sr} != feature sr {features.sr}")
        self.hits = hits
        self.label_map = label_map
        self.window = window
        self.features = features
        self.augment = augment
        self._rng_seed = None
        self._rng_state = None

    def __len__(self):
        return len(self.hits)

    def _rng(self) -> np.random.Generator:
        # One generator per process: DataLoader workers get different torch seeds, the main process uses the
        # seed set by set_seed(), and the stream continues across epochs.
        seed = torch.initial_seed() % 2 ** 32
        if seed != self._rng_seed:
            self._rng_seed, self._rng_state = seed, np.random.default_rng(seed)
        return self._rng_state

    def labels(self) -> list[int]:
        return [self.label_map[hit.label] for hit in self.hits]

    def __getitem__(self, idx):
        hit = self.hits[idx]
        onset = hit.onset
        if self.augment is not None:
            rng = self._rng()
            onset = jitter_onset(onset, self.window.sr, rng, self.augment.onset_jitter_s)
            window = cut_window(hit.audio, onset, self.window)
            other = self.hits[int(rng.integers(len(self.hits)))]
            window = augment_window(window, self.window.sr, rng, self.augment, other=(other.audio, other.onset))
        else:
            window = cut_window(hit.audio, onset, self.window)
        x = window_features(window, self.features)
        y = torch.tensor(self.label_map[hit.label], dtype=torch.long)
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


def band_statistics(dataset: WindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-band mean and std of the (un-augmented) features of a dataset."""
    features = torch.cat([dataset[i][0][0] for i in range(len(dataset))], dim=1)  # (n_mels, all frames)
    return features.mean(dim=1), features.std(dim=1)


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
    def __init__(self, num_classes=len(DRUM_INSTRUMENTS), window: WindowConfig = WindowConfig(),
                 features: LogMelConfig = LogMelConfig(), augment: AugmentConfig = AugmentConfig(),
                 batch_size=64, num_workers=0):
        super().__init__()
        self.num_classes = num_classes
        self.window = window
        self.features = features
        self.augment = augment
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

    @abstractmethod
    def model_config(self) -> dict:
        """Constructor arguments of the model, stored in the checkpoint."""
        pass

    def forward(self, x):
        """Forward pass delegates to the underlying model."""
        return self.model(x)

    def _load_split(self, split, augment=None) -> WindowDataset:
        hits = load_subset(os.path.join(INSTRUMENTS_DATA_SET_PATH, split), DRUM_INSTRUMENTS, self.window.sr)
        return WindowDataset(hits, self.label_map, self.window, self.features, augment)

    def _create_loader(self, dataset, shuffle):
        # macOS starts workers with spawn, which pickles the in-memory dataset; persistent workers do it once
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def save(self, path):
        save_checkpoint(path, self.model, self.model_config(), self.window, self.features)

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
                    epochs=60, patience=10, class_weighted=True):
        """Full training loop with validation, early stopping on macro-F1, and model saving."""
        print(f"Window: {self.window.pre_s * 1000:.0f} ms before to {self.window.post_s * 1000:.0f} ms after the onset; "
              f"augmentation: {'on' if self.augment else 'off'}")
        train_dataset = self._load_split("train", self.augment)
        val_dataset = self._load_split("val")
        print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

        # Input standardization is fitted on clean training windows and stored in the model's buffers.
        clean_train = WindowDataset(train_dataset.hits, self.label_map, self.window, self.features)
        mean, std = band_statistics(clean_train)
        self.model.input_norm.set_stats(mean, std)

        train_loader = self._create_loader(train_dataset, shuffle=True)
        val_loader = self._create_loader(val_dataset, shuffle=False)

        if class_weighted and isinstance(criterion, nn.CrossEntropyLoss):
            criterion.weight = compute_class_weights(train_dataset.labels(), self.num_classes)
            print(f"Class weights: {criterion.weight.numpy().round(3).tolist()}")
        criterion.to(self.device)

        early_stopping = EarlyStopping(patience)
        best_val_acc = 0

        train_losses = []
        val_losses = []
        val_accs = []
        val_f1s = []
        epoch_times = []
        learning_rates = []

        for epoch in range(epochs):
            learning_rates.append(optimizer.param_groups[0]["lr"])
            train_loss, epoch_duration = self._train_epoch(train_loader, optimizer, criterion)
            train_losses.append(train_loss)
            epoch_times.append(epoch_duration)

            val_loss, val_acc, val_labels, val_preds, _ = self._evaluate(val_loader, criterion)
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)

            print(
                f"Epoch {epoch + 1}/{epochs}, LR: {learning_rates[-1]:.2e}, Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}, "
                f"Epoch Time: {epoch_duration:.2f}s")

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if early_stopping.step(val_f1):
                best_val_acc = val_acc
                self.save(best_model_save_path)
                print(f"Saved best model (Val Macro-F1: {val_f1:.4f}) to {best_model_save_path}")

            print(f"Epochs without validation macro-F1 improvement: {early_stopping.epochs_no_improve}/{patience}")
            if early_stopping.should_stop:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(f"Training finished. Best validation macro-F1: {early_stopping.best:.4f}")

        self.save(save_path)
        print(f"Saved final model state to {save_path}")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "val_f1s": val_f1s,
            "best_val_f1": early_stopping.best,
            "best_val_acc": best_val_acc,
            "epoch_times": epoch_times,
            "learning_rates": learning_rates,
            "window": self.window.to_dict(),
            "features": self.features.to_dict(),
            "model_config": self.model_config(),
        }

    def evaluate_model(self, model_path, criterion):
        """Loads a trained model and evaluates it on the test set, using the window it was trained with."""
        checkpoint = load_checkpoint(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.window, self.features = checkpoint["window"], checkpoint["features"]
        self.model.eval()
        criterion.to(self.device)

        test_dataset = self._load_split("test")
        test_loader = self._create_loader(test_dataset, shuffle=False)
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

        # Crash, ride and the toms come from a single library kit, so accuracy per source shows kit dependence.
        per_source = {}
        for hit, label, pred in zip(test_dataset.hits, all_labels, all_preds):
            correct, total = per_source.get(hit.source, (0, 0))
            per_source[hit.source] = (correct + int(label == pred), total + 1)
        per_source_accuracy = {src: correct / total for src, (correct, total) in sorted(per_source.items())}
        print("\nPer-source accuracy:", {src: round(acc, 4) for src, acc in per_source_accuracy.items()})

        return {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "labels": all_labels,
            "predictions": all_preds,
            "outputs": all_outputs,
            "file_names": [hit.name for hit in test_dataset.hits],
            "per_source_accuracy": per_source_accuracy,
            "window": self.window.to_dict(),
        }

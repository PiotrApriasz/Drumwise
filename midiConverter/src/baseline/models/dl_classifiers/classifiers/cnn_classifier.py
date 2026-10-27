import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.constants import DRUM_INSTRUMENTS, CNN_DROPOUT_RATE, CNN_BASE_CHANNELS, CNN_MINIMUM_LEARNING_RATE, \
    CNN_WARMUP_EPOCHS
from src.baseline.data.windowing import WindowConfig
from src.baseline.models.dl_classifiers.classifiers.base_classifier import BaseClassifier
from src.baseline.models.dl_classifiers.classifiers.checkpoints import load_checkpoint
from src.data.features import LogMelConfig


class InputNorm(nn.Module):
    """Per-band standardization; the statistics are fitted on training data and saved with the model."""

    def __init__(self, n_bands):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_bands))
        self.register_buffer("std", torch.ones(n_bands))

    def set_stats(self, mean, std):
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-5))

    def forward(self, x):  # x: (batch, 1, bands, frames)
        return (x - self.mean[None, None, :, None]) / self.std[None, None, :, None]


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4, min_hidden=8):
        super().__init__()
        hidden = max(min_hidden, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = x.mean((2, 3))
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(batch, channels, 1, 1)


class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers, squeeze-and-excitation, then max pooling over frequency only.

    Time is never pooled: an onset window has only about 10 frames, and their order carries the attack.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(self.se(x))


class DrumCNN(nn.Module):
    """Classifies one onset window of log-mel frames. Global average pooling makes it work for any window length."""

    def __init__(self, num_classes=len(DRUM_INSTRUMENTS), n_mels=128, base_channels=CNN_BASE_CHANNELS,
                 dropout_rate=CNN_DROPOUT_RATE):
        super().__init__()
        c = base_channels
        self.input_norm = InputNorm(n_mels)
        self.blocks = nn.Sequential(ConvBlock(1, c), ConvBlock(c, 2 * c), ConvBlock(2 * c, 4 * c))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(4 * c, num_classes)

    def forward(self, x):
        x = self.blocks(self.input_norm(x))
        x = x.mean(dim=(2, 3))
        return self.classifier(self.dropout(x))


class DrumCNNClassifier(BaseClassifier):
    """CNN Classifier for Drum Sounds, using the BaseClassifier for training."""

    def __init__(self, window: WindowConfig = WindowConfig(), features: LogMelConfig = LogMelConfig(),
                 augment=None, batch_size=64, num_workers=0, base_channels=CNN_BASE_CHANNELS,
                 dropout_rate=CNN_DROPOUT_RATE):
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        self.n_mels = features.n_mels
        super().__init__(num_classes=len(DRUM_INSTRUMENTS), window=window, features=features, augment=augment,
                         batch_size=batch_size, num_workers=num_workers)

    def _build_model(self) -> nn.Module:
        return DrumCNN(**self.model_config())

    def model_config(self) -> dict:
        return {"num_classes": self.num_classes, "n_mels": self.n_mels, "base_channels": self.base_channels,
                "dropout_rate": self.dropout_rate}


def load_cnn(path, map_location="cpu"):
    """Returns ``(model, window, features)`` from a checkpoint saved by ``BaseClassifier.save``."""
    checkpoint = load_checkpoint(path, map_location)
    model = DrumCNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["window"], checkpoint["features"]


def build_cnn_optimization(classifier, lr, weight_decay, label_smoothing, epochs, warmup_epochs=CNN_WARMUP_EPOCHS):
    """Optimizer, loss and LR schedule shared by the trainer and the tuner (review item B4).

    AdamW with a linear warm-up over ``warmup_epochs``, then one cosine decay to ``CNN_MINIMUM_LEARNING_RATE``
    over the remaining epochs. The scheduler is stepped once per epoch.
    """
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    min_ratio = CNN_MINIMUM_LEARNING_RATE / lr
    warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))

    def lr_factor(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return optimizer, criterion, LambdaLR(optimizer, lr_factor)

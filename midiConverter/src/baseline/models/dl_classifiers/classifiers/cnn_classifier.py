import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.constants import DRUM_INSTRUMENTS, CNN_DROPOUT_RATE, CNN_NUM_FILTERS1, CNN_NUM_FILTERS2, \
    CNN_MINIMUM_LEARNING_RATE
from src.baseline.models.dl_classifiers.classifiers.base_classifier import BaseClassifier

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = x.mean((2, 3))
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1)
        return x * y


class DrumCNN(nn.Module):
    def __init__(self, num_classes=7, num_filters1=CNN_NUM_FILTERS1, num_filters2=CNN_NUM_FILTERS2,
                 dropout_rate=CNN_DROPOUT_RATE):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters1)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters2)
        self.se1 = SEBlock(num_filters2)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(num_filters2, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)

        self.adapool = nn.AdaptiveMaxPool2d((15, 7))
        self.fc1 = nn.Linear(128 * 15 * 7, 128)
        self.dropout = nn.Dropout(float(dropout_rate))
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se1(x)
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se2(x)
        x = self.pool(x)

        x = self.adapool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class DrumCNNClassifier(BaseClassifier):
    """CNN Classifier for Drum Sounds, using the BaseClassifier for training."""

    def __init__(self, sr=22050, batch_size=8, num_workers=4,
                 num_filters1=CNN_NUM_FILTERS1, num_filters2=CNN_NUM_FILTERS2, dropout_rate=CNN_DROPOUT_RATE):
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2
        self.dropout_rate = dropout_rate

        super().__init__(num_classes=len(DRUM_INSTRUMENTS), sr=sr,
                         batch_size=batch_size,
                         num_workers=num_workers)

    def _build_model(self) -> nn.Module:
        print("Building DrumCNN model...")
        return DrumCNN(num_classes=self.num_classes,
                       num_filters1=self.num_filters1,
                       num_filters2=self.num_filters2,
                       dropout_rate=self.dropout_rate)



def build_cnn_optimization(classifier, lr, weight_decay, label_smoothing, epochs):
    """Optimizer, loss and LR schedule shared by the trainer and the tuner (review item B4).

    The cosine schedule spans the whole run (T_max = epochs), so the learning rate decays once instead of
    oscillating with a fixed period.
    """
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=CNN_MINIMUM_LEARNING_RATE)
    return optimizer, criterion, scheduler

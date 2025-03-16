import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataset_loader import load_dataset_with_splits
from src.constants import DRUM_INSTRUMENTS
from src.models.dl_classifiers.classifiers.cnn_classifier import create_augmented_dataset, CqtDrumDataset, SEBlock


def objective(trial):

    num_filters1 = trial.suggest_int("num_filters1", 8, 32, step=8)
    num_filters2 = trial.suggest_int("num_filters2", 16, 64, step=16)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    class TunedDrumCNN(nn.Module):
        def __init__(self, num_classes=7):
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

            self.adapool = nn.AdaptiveMaxPool2d((15, 7))
            self.fc1 = nn.Linear(128 * 15 * 7, 128)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.se1(x)
            x = self.pool(x)

            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)

            x = self.adapool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

    model = TunedDrumCNN(num_classes=len(DRUM_INSTRUMENTS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    max_epochs = 10
    for epoch in range(max_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            val_out = model(x_val)
            preds = val_out.argmax(dim=1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)
    val_acc = correct / total

    return val_acc

if __name__ == "__main__":

    train_data, val_data, _ = load_dataset_with_splits()
    train_data_aug = create_augmented_dataset(train_data, sr=22050, augment_factor=2)
    train_dataset = CqtDrumDataset(train_data_aug, sr=22050)
    val_dataset = CqtDrumDataset(val_data, sr=22050)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Val Acc: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
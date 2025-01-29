import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data.dataset_loader import load_dataset_with_splits
from src.data.tools.dataset_augumentation import create_augmented_dataset
from src.models.constants import DRUM_INSTRUMENTS
from src.models.dl_models.dl_dataset_creator import DrumDataset


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
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.adapool = nn.AdaptiveMaxPool2d((15, 7))
        self.fc1 = nn.Linear(128 * 15 * 7, 128)
        self.dropout = nn.Dropout(0.287)
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


def train_cnn_classifier(model_path, best_model_path):
    best_val_acc = 0

    label_map = {inst: i for i, inst in enumerate(DRUM_INSTRUMENTS)}

    train_data, val_data, test_data = load_dataset_with_splits()
    train_data_aug = create_augmented_dataset(train_data, sr=22050, augment_factor=2)
    train_dataset = DrumDataset(train_data_aug, label_map, sr=22050)
    val_dataset = DrumDataset(val_data, label_map, sr=22050)
    test_dataset = DrumDataset(test_data, label_map, sr=22050)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000103, weight_decay=1.98e-05)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    patience = 6
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 50

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
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_out = model(x_val)
                val_loss += criterion(val_out, y_val).item()
                preds = val_out.argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model so far: {best_val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            outputs = model(x_test)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print("Test Accuracy:", test_acc)
    print("Confusion Matrix:\n", cm)

    torch.save(model.state_dict(), model_path)

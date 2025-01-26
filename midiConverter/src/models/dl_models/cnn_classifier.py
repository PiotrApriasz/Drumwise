import os
import torch
import random
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data.dataset_loader import load_dataset_with_splits  # nowy moduł z metodą load_dataset_with_splits
instruments = ["snare", "crash", "floor_tom", "hi-hat", "kick", "rack_tom", "ride"]
label_map = {inst: i for i, inst in enumerate(instruments)}

def time_shift(audio, sr, max_shift_s=0.1):
    max_shift = int(max_shift_s * sr)
    shift = random.randint(-max_shift, max_shift)
    if shift >= 0:
        audio = np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
    else:
        audio = np.pad(audio, (0, -shift), mode='constant')[-len(audio):]
    return audio

def gain_change(audio, min_gain_db=-6, max_gain_db=6):
    gain_db = random.uniform(min_gain_db, max_gain_db)
    factor = 10 ** (gain_db / 20.0)
    audio = audio * factor
    return np.clip(audio, -1.0, 1.0)

def add_noise(audio, noise_factor=0.005):
    noise_amp = noise_factor * np.random.random() * np.amax(np.abs(audio))
    noise = noise_amp * np.random.normal(size=len(audio))
    return audio + noise

def pitch_shift(audio, sr, semitones=2):
    shift = random.uniform(-semitones, semitones)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)

augmentations = [
    time_shift,
    gain_change,
    add_noise,
    pitch_shift
]

def apply_random_augmentations(audio, sr):
    num_aug = random.randint(1, 3)
    chosen_augs = random.sample(augmentations, num_aug)
    random.shuffle(chosen_augs)
    for aug in chosen_augs:
        if aug == pitch_shift:
            audio = pitch_shift(audio, sr=sr, semitones=2)
        elif aug == time_shift:
            audio = time_shift(audio, sr=sr, max_shift_s=0.1)
        elif aug == add_noise:
            audio = add_noise(audio, noise_factor=0.005)
        elif aug == gain_change:
            audio = gain_change(audio, min_gain_db=-6, max_gain_db=6)
    return audio

def create_augmented_dataset(original_data, sr=22050, augment_factor=1):
    new_data = []
    for audio, label in original_data:
        new_data.append((audio, label))
        for _ in range(augment_factor):
            augmented = apply_random_augmentations(audio.copy(), sr)
            new_data.append((augmented, label))
    return new_data

class DrumDataset(Dataset):
    def __init__(self, data, sr=22050):
        self.data = data
        self.sr = sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, label_str = self.data[idx]
        label = label_map[label_str]
        cqt = librosa.cqt(audio, sr=self.sr, hop_length=512)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        x = torch.tensor(cqt_db, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

train_data, val_data, test_data = load_dataset_with_splits()

train_data_aug = create_augmented_dataset(train_data, sr=22050, augment_factor=2)

train_dataset = DrumDataset(train_data_aug, sr=22050)  # augment=False, bo już mamy gotowe dane
val_dataset   = DrumDataset(val_data, sr=22050)
test_dataset  = DrumDataset(test_data, sr=22050)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)


class DrumCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.adapool = nn.AdaptiveMaxPool2d((15, 7))
        self.fc1 = nn.Linear(128 * 15 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.adapool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

if __name__ == "__main__":

    best_val_acc = 0  # Przechowujemy najlepszy val_acc
    best_model_path = "best_drum_cnn.pth"

    model = DrumCNN(num_classes=len(instruments))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    patience = 3
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
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Zapisano nowy najlepszy model z Val Acc: {best_val_acc:.4f}")

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

    torch.save(model.state_dict(), "drum_cnn.pth")
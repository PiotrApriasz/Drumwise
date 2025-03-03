import os
import torch
import pandas as pd
import librosa
import random
import numpy as np
import mido
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from src.models.constants import GROOVE_MIDI_DATASET_PATH
from src.models.dl_classifiers.classifiers.cnn_classifier import SEBlock

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

writer = SummaryWriter(log_dir="runs/experiment_1")

drum_map = {
    35: 0, 36: 0, 26: 0,
    38: 1, 37: 1, 40: 1,
    42: 2,
    44: 3,
    46: 4,
    41: 5, 43: 5,
    45: 6, 47: 6, 48: 6, 50: 6,
    49: 7, 57: 7, 55: 7, 52: 7,
    51: 8, 59: 8,
}

instrument_names = [
    'Kick', 'Snare', 'Closed Hi-Hat', 'Pedal Hi-Hat', 'Open Hi-Hat',
    'Floor Tom', 'Rack Tom', 'Crash', 'Ride'
]

NUM_INSTR = 9


def file_exists(row):
    audio_full_path = os.path.join(GROOVE_MIDI_DATASET_PATH, row['audio_filename'])
    midi_full_path = os.path.join(GROOVE_MIDI_DATASET_PATH, row['midi_filename'])
    return os.path.exists(audio_full_path) and os.path.exists(midi_full_path)


def my_collate_fn(batch):
    cqt_data_list = []
    labels_list = []
    events_list = []
    time_gt_list = []
    audio_path_list = []
    midi_path_list = []
    bpm_list = []
    for item in batch:
        cqt_data, labels, time_gt, events, audio_path, midi_path, bpm = item

        cqt_np = cqt_data.cpu().numpy()
        print("CQT mean:", np.mean(cqt_np), "std:", np.std(cqt_np))

        cqt_data_list.append(cqt_data)  # [1, n_bins, num_beats]
        labels_list.append(labels)  # [num_beats, NUM_INSTR]
        time_gt_list.append(time_gt)
        events_list.append(events)
        audio_path_list.append(audio_path)
        midi_path_list.append(midi_path)
        bpm_list.append(bpm)
    max_beats = max([t.shape[-1] for t in cqt_data_list])
    padded_cqt_list = []
    for t in cqt_data_list:
        pad_size = max_beats - t.shape[-1]
        padded = F.pad(t, (0, pad_size), "constant", 0)
        padded_cqt_list.append(padded)
    cqt_data_batch = torch.stack(padded_cqt_list, dim=0)

    max_labels_beats = max([l.shape[0] for l in labels_list])
    padded_labels_list = []
    for l in labels_list:
        pad_size = max_labels_beats - l.shape[0]
        padded = F.pad(l, (0, 0, 0, pad_size), "constant", 0)
        padded_labels_list.append(padded)
    labels_batch = torch.stack(padded_labels_list, dim=0)

    max_beats = max([t.shape[0] for t in time_gt_list])
    padded_time_gt_list = []
    for t in time_gt_list:
        pad_size = max_beats - t.shape[0]
        padded = F.pad(t, (0, 0, 0, pad_size), "constant", 0)
        padded_time_gt_list.append(padded)
    time_gt_batch = torch.stack(padded_time_gt_list, dim=0)

    return cqt_data_batch, labels_batch, time_gt_batch, events_list, audio_path_list, midi_path_list, bpm_list





def parse_drum_midi(midi_path, bpm, sr, hop_length):
    mid = mido.MidiFile(midi_path)
    events = []
    time_in_seconds = 0.0
    for msg in mid:
        time_in_seconds += msg.time
        if not msg.is_meta and msg.type == 'note_on' and msg.velocity > 0:
            note = msg.note
            if note in drum_map:
                instrument_id = drum_map[note]
                events.append((time_in_seconds, instrument_id))
    #print(f"DEBUG: Parsed events for {os.path.basename(midi_path)}: {events}")
    return events


def beat_sync_features(cqt_db, y, sr, hop_length, aggregate=np.median):

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    beat_sync = librosa.util.sync(cqt_db, beat_frames, aggregate=aggregate)
    return beat_sync, beat_frames, tempo


def compute_onset_time_labels(beat_frames, events, sr, hop_length):

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    num_beats = len(beat_times)
    time_gt = np.zeros((num_beats, 1), dtype=np.float32)

    if num_beats > 1:
        beat_durations = np.diff(beat_times)
        default_duration = np.median(beat_durations)
    else:
        default_duration = 0.5

    for i in range(num_beats):
        start_time = beat_times[i]
        end_time = beat_times[i+1] if i+1 < num_beats else start_time + default_duration

        beat_events = [e for e in events if start_time <= e[0] < end_time]
        if beat_events:
            onset_time = min(e[0] for e in beat_events)
            normalized = (onset_time - start_time) / (end_time - start_time)
            time_gt[i, 0] = normalized
        else:
            time_gt[i, 0] = 0.0
    return time_gt


class GrooveDataset(Dataset):
    def __init__(self, csv_path, split, sr=22050, n_bins=84, bins_per_octave=12, hop_length=512,
                 transform=None, styles=('funk', 'hiphop', 'pop', 'rock'), time_signature='4-4'):
        self.sr = sr
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.transform = transform
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        df = df.dropna(subset=['audio_filename', 'midi_filename'])
        df = df[df['audio_filename'].str.strip() != ""]
        df = df[df['midi_filename'].str.strip() != ""]
        df = df[df.apply(file_exists, axis=1)]
        df = df[(df['time_signature'] == time_signature) & (df['style'].isin(styles))]
        df.reset_index(drop=True, inplace=True)
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row['audio_filename']
        midi_path = row['midi_filename']
        bpm = row['bpm']
        y, sr = librosa.load(os.path.join(GROOVE_MIDI_DATASET_PATH, audio_path), sr=self.sr, mono=True)

        cqt = librosa.cqt(y, sr=sr, hop_length=self.hop_length,
                          n_bins=self.n_bins, bins_per_octave=self.bins_per_octave)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        cqt_db = np.expand_dims(cqt_db, axis=0)  # [1, n_bins, time_frames]

        beat_sync, beat_frames, tempo = beat_sync_features(cqt_db, y, sr, self.hop_length, aggregate=np.median)
        # beat_sync: [1, n_bins, num_beats]

        events = parse_drum_midi(os.path.join(GROOVE_MIDI_DATASET_PATH, midi_path), bpm, sr, self.hop_length)

        time_frames_full = cqt_db.shape[-1]
        labels = np.zeros((time_frames_full, NUM_INSTR), dtype=np.float32)
        for (time_s, instr_id) in events:
            frame_idx = int((time_s * sr) / self.hop_length)
            if 0 <= frame_idx < time_frames_full:
                labels[frame_idx, instr_id] = 1.0

        labels_beat = librosa.util.sync(labels.T, beat_frames, aggregate=np.max).T  # [num_beats, NUM_INSTR]

        time_gt = compute_onset_time_labels(beat_frames, events, sr, self.hop_length)  # [num_beats, 1]

        cqt_sync_tensor = torch.tensor(beat_sync, dtype=torch.float32)  # [1, n_bins, num_beats]
        labels_sync_tensor = torch.tensor(labels_beat, dtype=torch.float32)  # [num_beats, NUM_INSTR]
        time_gt_tensor = torch.tensor(time_gt, dtype=torch.float32)  # [num_beats, 1]

        #print(f"DEBUG: {audio_path} | {midi_path} | BPM: {bpm} | Beat-sync CQT shape: {cqt_sync_tensor.shape} | Beat-sync labels shape: {labels_sync_tensor.shape}")
        return cqt_sync_tensor, labels_sync_tensor, time_gt_tensor, events, audio_path, midi_path, bpm


class DrumCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.se1 = SEBlock(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.se2 = SEBlock(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.se3 = SEBlock(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.adapool = nn.AdaptiveAvgPool2d((15, 3))
        self.fc1 = nn.Linear(64 * 15 * 3, 128)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se1(x)
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se2(x)
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.se3(x)
        x = self.pool3(x)

        if x.device.type == "mps":
            x = x.cpu()
            x = self.adapool(x)
            x = x.to(torch.device("mps"))
        else:
            x = self.adapool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x


class DrumEndToEndModel(nn.Module):
    def __init__(self, num_instruments, hidden_size=128):
        super().__init__()
        self.feature_extractor = DrumCNNFeatureExtractor()
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=2,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.fc_onset = nn.Linear(hidden_size * 2, num_instruments)
        self.fc_time = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        batch, _, n_bins, num_beats = x.shape
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(2)
        x = x.permute(0, 1, 3, 2)
        x = x.repeat(1, 1, 1, 3)
        x = x.view(batch * num_beats, 1, n_bins, 3)

        features = self.feature_extractor(x)  # [batch*num_beats, 128]
        features = features.view(batch, num_beats, -1)  # [batch, num_beats, 128]

        seq_lengths = torch.full((batch,), num_beats, dtype=torch.long, device=features.device)
        packed = nn.utils.rnn.pack_padded_sequence(features, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        onset_pred = self.fc_onset(lstm_out)  # [batch, num_beats, num_instruments]
        time_pred = torch.relu(self.fc_time(lstm_out))  # [batch, num_beats, 1]
        return onset_pred, time_pred


def debug_end_to_end_model():
    dataset = GrooveDataset(csv_path=os.path.join(GROOVE_MIDI_DATASET_PATH, "info.csv"), split='train')
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=my_collate_fn)
    model = DrumEndToEndModel(num_instruments=NUM_INSTR, hidden_size=128)
    model.eval()
    for batch_idx, (cqt_data, labels, events, audio_paths, midi_paths, bpms) in enumerate(loader):
        print(f"\n=== Batch {batch_idx} ===")
        print("cqt_data shape:", cqt_data.shape)  # [batch, 1, n_bins, num_beats]
        onset_pred, time_pred = model(cqt_data)
        print("Onset prediction shape:", onset_pred.shape)  # [batch, num_beats, NUM_INSTR]
        print("Time prediction shape:", time_pred.shape)  # [batch, num_beats, 1]
        print("Audio files:", audio_paths)
        print("MIDI files:", midi_paths)

        features = model.feature_extractor(cqt_data.view(-1, 1, cqt_data.shape[2], cqt_data.shape[3]))
        print("Feature extractor output - mean:", features.mean().item(), "std:", features.std().item())

        sample_idx = 0
        beat_sync = cqt_data[sample_idx, 0].numpy()  # [n_bins, num_beats]
        onset_labels = labels[sample_idx].numpy()  # [num_beats, NUM_INSTR]
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(beat_sync, sr=22050, x_axis='time', y_axis='cqt_hz')
        plt.title("Beat-synchronous CQT")
        plt.colorbar(format="%+2.f dB")
        num_beats = beat_sync.shape[1]
        beat_times = librosa.frames_to_time(np.arange(num_beats), sr=22050, hop_length=512)
        onset_binary = (onset_labels.sum(axis=1) > 0).astype(int)
        plt.scatter(beat_times, np.full_like(beat_times, beat_sync.shape[0] - 1),
                    c=onset_binary, cmap='coolwarm', marker='o', label='Onset')
        plt.legend()
        plt.show()
        break

activation_dict = {}

def get_activation(name):
    def hook(model, input, output):
        activation_dict[name] = output.detach().cpu()
    return hook

def train_end_to_end_model():
    train_dataset = GrooveDataset(csv_path=os.path.join(GROOVE_MIDI_DATASET_PATH, "info.csv"), split='train')
    val_dataset = GrooveDataset(csv_path=os.path.join(GROOVE_MIDI_DATASET_PATH, "info.csv"), split='validation')
    test_dataset = GrooveDataset(csv_path=os.path.join(GROOVE_MIDI_DATASET_PATH, "info.csv"), split='test')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=my_collate_fn)

    print("Data loaders prepared")

    model = DrumEndToEndModel(num_instruments=NUM_INSTR, hidden_size=128).to(device)

    model.feature_extractor.conv1.register_forward_hook(get_activation('conv1'))
    model.feature_extractor.conv2.register_forward_hook(get_activation('conv2'))
    model.feature_extractor.conv3.register_forward_hook(get_activation('conv3'))
    model.feature_extractor.conv4.register_forward_hook(get_activation('conv4'))
    model.feature_extractor.conv5.register_forward_hook(get_activation('conv5'))
    model.feature_extractor.conv6.register_forward_hook(get_activation('conv6'))
    model.feature_extractor.fc1.register_forward_hook(get_activation('fc1'))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion_onset = nn.BCEWithLogitsLoss()
    criterion_time = nn.MSELoss()

    num_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    print("Starting training loop")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_onset_preds = []
        all_train_onset_labels = []
        all_train_time_preds = []
        all_train_time_labels = []

        batch_mse_list = []
        batch_mae_list = []
        total_beats = 0

        val_batch_mse = []
        val_batch_mae = []
        total_beats_val = 0

        test_batch_mse = []
        test_batch_mae = []
        total_beats_test = 0

        all_activations = []

        onset_means = []
        onset_medians = []
        onset_all = []  #

        for batch in train_loader:
            cqt_data, labels, time_gt, events, audio_paths, midi_paths, bpms = batch
            cqt_data = cqt_data.to(device)  # [batch, 1, n_bins, num_beats]
            labels = labels.to(device)      # [batch, num_beats, NUM_INSTR]
            time_gt = time_gt.to(device)    # [batch, num_beats, 1]

            optimizer.zero_grad()
            onset_pred, time_pred = model(cqt_data)
            loss_onset = criterion_onset(onset_pred, labels)
            min_seq_len = min(time_pred.size(1), time_gt.size(1))
            loss_time = criterion_time(time_pred[:, :min_seq_len, :], time_gt[:, :min_seq_len, :])
            loss = loss_onset + loss_time
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * cqt_data.size(0)

            onset_pred_bin = (torch.sigmoid(onset_pred) > 0.5).float().detach().cpu().numpy()
            onset_all.extend(onset_pred_bin.flatten())
            onset_means.append(np.mean(onset_pred_bin))
            onset_medians.append(np.median(onset_pred_bin))

            with torch.no_grad():
                activations = model.feature_extractor(cqt_data.view(-1, 1, cqt_data.shape[2], cqt_data.shape[3]))
                all_activations.append(activations.cpu())

            avg_loss = train_loss / len(train_dataset)
            writer.add_scalar("Loss/Train", avg_loss, epoch)
            writer.add_scalar("Onset/Mean", np.mean(onset_means), epoch)
            writer.add_scalar("Onset/Median", np.median(onset_medians), epoch)
            writer.add_histogram("Onset/Histogram", np.array(onset_all), epoch)

            activations_cat = torch.cat(all_activations, dim=0)
            writer.add_histogram("Activations/FeatureExtractor", activations_cat, epoch)

            if 'conv1' in activation_dict:
                writer.add_histogram("Activations/conv1", activation_dict['conv1'], epoch)

            cqt_np = cqt_data.cpu().numpy()
            writer.add_scalar("Input/CQT_mean", np.mean(cqt_np), epoch)
            writer.add_scalar("Input/CQT_std", np.std(cqt_np), epoch)

            onset_pred_bin = (torch.sigmoid(onset_pred) > 0.5).float().detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            all_train_onset_preds.append(onset_pred_bin)
            all_train_onset_labels.append(labels_np)
            all_train_time_preds.append(time_pred.detach().cpu().numpy())
            all_train_time_labels.append(time_gt.detach().cpu().numpy())
        train_loss /= len(train_dataset)
        all_train_onset_preds_flat = np.concatenate([pred.reshape(-1) for pred in all_train_onset_preds])
        all_train_onset_labels_flat = np.concatenate([lab.reshape(-1) for lab in all_train_onset_labels])
        train_f1 = f1_score(all_train_onset_labels_flat, all_train_onset_preds_flat, average='micro')
        train_accuracy = accuracy_score(all_train_onset_labels_flat, all_train_onset_preds_flat)

        #all_train_time_preds_flat = np.concatenate([tp.reshape(-1) for tp in all_train_time_preds])
        #all_train_time_labels_flat = np.concatenate([tl.reshape(-1) for tl in all_train_time_labels])
        #train_mse = mean_squared_error(all_train_time_labels_flat, all_train_time_preds_flat)
        #train_mae = mean_absolute_error(all_train_time_labels_flat, all_train_time_preds_flat)

        for tp, tl in zip(all_train_time_preds, all_train_time_labels):
            min_seq_len = min(tp.shape[1], tl.shape[1])
            tp_trim = tp[:, :min_seq_len, :]
            tl_trim = tl[:, :min_seq_len, :]
            mse_batch = mean_squared_error(tl_trim.flatten(), tp_trim.flatten())
            mae_batch = mean_absolute_error(tl_trim.flatten(), tp_trim.flatten())
            num_beats = tp_trim.shape[0] * tp_trim.shape[1]
            batch_mse_list.append(mse_batch * num_beats)
            batch_mae_list.append(mae_batch * num_beats)
            total_beats += num_beats

        train_mse = np.sum(batch_mse_list) / total_beats
        train_mae = np.sum(batch_mae_list) / total_beats

        if True:
            sample = train_loader.dataset[random.randint(0, len(train_loader.dataset) - 1)]
            sample_cqt, sample_labels, sample_time_gt, _, sample_audio, sample_midi, sample_bpm = sample
            sample_cqt = sample_cqt.unsqueeze(0).to(device)
            with torch.no_grad():
                sample_onset_pred, sample_time_pred = model(sample_cqt)
            log_sample_predictions(sample_cqt.squeeze(0), sample_labels, sample_time_gt, sample_onset_pred,
                                   sample_time_pred)

        print("Starting validation loop")

        # Walidacja
        model.eval()
        val_loss = 0.0
        all_val_onset_preds = []
        all_val_onset_labels = []
        all_val_time_preds = []
        all_val_time_labels = []
        with torch.no_grad():
            for batch in val_loader:
                cqt_data, labels, time_gt, events, audio_paths, midi_paths, bpms = batch
                cqt_data = cqt_data.to(device)
                labels = labels.to(device)
                time_gt = time_gt.to(device)
                onset_pred, time_pred = model(cqt_data)
                loss_onset = criterion_onset(onset_pred, labels)
                min_seq_len = min(time_pred.size(1), time_gt.size(1))
                loss_time = criterion_time(time_pred[:, :min_seq_len, :], time_gt[:, :min_seq_len, :])
                loss = loss_onset + loss_time
                val_loss += loss.item() * cqt_data.size(0)
                onset_pred_bin = (torch.sigmoid(onset_pred) > 0.5).float().detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                all_val_onset_preds.append(onset_pred_bin)
                all_val_onset_labels.append(labels_np)
                all_val_time_preds.append(time_pred.detach().cpu().numpy())
                all_val_time_labels.append(time_gt.detach().cpu().numpy())
        val_loss /= len(val_dataset)
        all_val_onset_preds_flat = np.concatenate([pred.reshape(-1) for pred in all_val_onset_preds])
        all_val_onset_labels_flat = np.concatenate([lab.reshape(-1) for lab in all_val_onset_labels])
        #all_val_time_preds_flat = np.concatenate([tp.reshape(-1) for tp in all_val_time_preds])
        #all_val_time_labels_flat = np.concatenate([tl.reshape(-1) for tl in all_val_time_labels])

        val_f1 = f1_score(all_val_onset_labels_flat, all_val_onset_preds_flat, average='micro')
        val_accuracy = accuracy_score(all_val_onset_labels_flat, all_val_onset_preds_flat)
        #val_mse = mean_squared_error(all_val_time_labels_flat, all_val_time_preds_flat)
        #val_mae = mean_absolute_error(all_val_time_labels_flat, all_val_time_preds_flat)

        for tp, tl in zip(all_val_time_preds, all_val_time_labels):
            min_seq_len = min(tp.shape[1], tl.shape[1])
            tp_trim = tp[:, :min_seq_len, :]
            tl_trim = tl[:, :min_seq_len, :]
            num_beats = tp_trim.shape[0] * tp_trim.shape[1]
            mse_batch = mean_squared_error(tl_trim.flatten(), tp_trim.flatten())
            mae_batch = mean_absolute_error(tl_trim.flatten(), tp_trim.flatten())
            val_batch_mse.append(mse_batch * num_beats)
            val_batch_mae.append(mae_batch * num_beats)
            total_beats_val += num_beats
        val_mse = np.sum(val_batch_mse) / total_beats_val
        val_mae = np.sum(val_batch_mae) / total_beats_val

        print(f"Epoch {epoch+1}:")
        print(f"  Train loss: {train_loss:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}, Time MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, Time MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")

        if len(val_dataset) > 0:
            rand_idx = random.randint(0, len(val_dataset) - 1)
            sample = val_dataset[rand_idx]
            audio_path_sample = sample[4]
            midi_path_sample = sample[5]
            bpm_sample = sample[6]
            print("\n=== Logging random validation sample ===")
            output_events = write_model_output_to_text(audio_path_sample, midi_path_sample, bpm_sample, model, device)
            for event in output_events:
                print(event)
            print("=== End logging validation sample ===\n")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model so far")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
        scheduler.step()

    print("Training finished")

    print("Testing the best model")

    # Testowanie
    model.eval()
    test_loss = 0.0
    all_test_onset_preds = []
    all_test_onset_labels = []
    all_test_time_preds = []
    all_test_time_labels = []
    with torch.no_grad():
        for batch in test_loader:
            cqt_data, labels, time_gt, events, audio_paths, midi_paths, bpms = batch
            cqt_data = cqt_data.to(device)
            labels = labels.to(device)
            time_gt = time_gt.to(device)
            onset_pred, time_pred = model(cqt_data)
            loss_onset = criterion_onset(onset_pred, labels)
            min_seq_len = min(time_pred.size(1), time_gt.size(1))
            loss_time = criterion_time(time_pred[:, :min_seq_len, :], time_gt[:, :min_seq_len, :])
            loss = loss_onset + loss_time
            test_loss += loss.item() * cqt_data.size(0)
            onset_pred_bin = (torch.sigmoid(onset_pred) > 0.5).float().detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            all_test_onset_preds.append(onset_pred_bin)
            all_test_onset_labels.append(labels_np)
            all_test_time_preds.append(time_pred.detach().cpu().numpy())
            all_test_time_labels.append(time_gt.detach().cpu().numpy())
    test_loss /= len(test_dataset)
    all_test_onset_preds_flat = np.concatenate([pred.reshape(-1) for pred in all_test_onset_preds])
    all_test_onset_labels_flat = np.concatenate([lab.reshape(-1) for lab in all_test_onset_labels])
    #all_test_time_preds_flat = np.concatenate([tp.reshape(-1) for tp in all_test_time_preds])
    #all_test_time_labels_flat = np.concatenate([tl.reshape(-1) for tl in all_test_time_labels])

    test_f1 = f1_score(all_test_onset_labels_flat, all_test_onset_preds_flat, average='micro')
    test_accuracy = accuracy_score(all_test_onset_labels_flat, all_test_onset_preds_flat)
    #test_mse = mean_squared_error(all_test_time_labels_flat, all_test_time_preds_flat)
    #test_mae = mean_absolute_error(all_test_time_labels_flat, all_test_time_preds_flat)
    for tp, tl in zip(all_test_time_preds, all_test_time_labels):
        min_seq_len = min(tp.shape[1], tl.shape[1])
        tp_trim = tp[:, :min_seq_len, :]
        tl_trim = tl[:, :min_seq_len, :]
        num_beats = tp_trim.shape[0] * tp_trim.shape[1]
        mse_batch = mean_squared_error(tl_trim.flatten(), tp_trim.flatten())
        mae_batch = mean_absolute_error(tl_trim.flatten(), tp_trim.flatten())
        test_batch_mse.append(mse_batch * num_beats)
        test_batch_mae.append(mae_batch * num_beats)
        total_beats_test += num_beats
    test_mse = np.sum(test_batch_mse) / total_beats_test
    test_mae = np.sum(test_batch_mae) / total_beats_test
    print(f"Test loss: {test_loss:.4f}, F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}, Time MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

    if len(test_dataset) > 0:
        rand_idx = random.randint(0, len(test_dataset) - 1)
        sample = test_dataset[rand_idx]
        audio_path_sample = sample[4]
        midi_path_sample = sample[5]
        bpm_sample = sample[6]
        print("\n=== Logging random test sample ===")
        output_events = write_model_output_to_text(audio_path_sample, midi_path_sample, bpm_sample, model, device)
        for event in output_events:
            print(event)
        print("=== End logging test sample ===\n")


from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error
import librosa.display


def test_on_recording(audio_path, midi_path, model, device, sr=22050, n_bins=84, bins_per_octave=12, hop_length=512,
                      aggregate=np.median):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_db = np.expand_dims(cqt_db, axis=0)  # [1, n_bins, time_frames]

    beat_sync, beat_frames, tempo = beat_sync_features(cqt_db, y, sr, hop_length, aggregate=aggregate)
    # beat_sync: [1, n_bins, num_beats]

    extractedTempo = tempo.item()

    events = parse_drum_midi(midi_path, extractedTempo, sr, hop_length)
    time_frames_full = cqt_db.shape[-1]
    labels_frame = np.zeros((time_frames_full, NUM_INSTR), dtype=np.float32)
    for (time_s, instr_id) in events:
        frame_idx = int((time_s * sr) / hop_length)
        if 0 <= frame_idx < time_frames_full:
            labels_frame[frame_idx, instr_id] = 1.0
    labels_beat = librosa.util.sync(labels_frame.T, beat_frames, aggregate=np.max).T  # [num_beats, NUM_INSTR]

    time_gt = compute_onset_time_labels(beat_frames, events, sr, hop_length)  # [num_beats, 1]

    num_beats = min(beat_sync.shape[-1], labels_beat.shape[0], time_gt.shape[0])
    beat_sync = beat_sync[:, :, :num_beats]
    labels_beat = labels_beat[:num_beats, :]
    time_gt = time_gt[:num_beats, :]

    cqt_sync_tensor = torch.tensor(beat_sync, dtype=torch.float32)  # [1, n_bins, num_beats]
    labels_sync_tensor = torch.tensor(labels_beat, dtype=torch.float32)  # [num_beats, NUM_INSTR]
    time_gt_tensor = torch.tensor(time_gt, dtype=torch.float32)  # [num_beats, 1]

    print(f"Audio: {audio_path}")
    print(f"MIDI: {midi_path}")
    print(f"BPM (parametryczne): 120, Detected tempo: {tempo.item():.2f} BPM")
    print(f"Beat-sync CQT shape: {cqt_sync_tensor.shape}")
    print(f"Beat-sync labels shape: {labels_sync_tensor.shape}")
    print(f"Time GT shape: {time_gt_tensor.shape}")

    cqt_sync_tensor = cqt_sync_tensor.unsqueeze(0).to(device)
    labels_sync_tensor = labels_sync_tensor.unsqueeze(0).to(device)
    time_gt_tensor = time_gt_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        onset_pred, time_pred = model(cqt_sync_tensor)

    onset_pred_bin = (torch.sigmoid(onset_pred) > 0.5).float()

    print(f"Predicted onset shape: {onset_pred.shape}")
    print(f"Predicted time shape: {time_pred.shape}")

    onset_pred_flat = onset_pred_bin.cpu().numpy().flatten()
    onset_gt_flat = labels_sync_tensor.cpu().numpy().flatten()
    f1 = f1_score(onset_gt_flat, onset_pred_flat, average='micro')
    acc = accuracy_score(onset_gt_flat, onset_pred_flat)

    time_pred_flat = time_pred.cpu().numpy().flatten()
    time_gt_flat = time_gt_tensor.cpu().numpy().flatten()
    mse = mean_squared_error(time_gt_flat, time_pred_flat)
    mae = mean_absolute_error(time_gt_flat, time_pred_flat)

    print(f"Test Onset F1: {f1:.4f}, Accuracy: {acc:.4f}")
    print(f"Test Time MSE: {mse:.4f}, MAE: {mae:.4f}")

    return onset_pred, time_pred, labels_sync_tensor, time_gt_tensor, extractedTempo


import pretty_midi


def convert_model_output_to_midi(audio_path, onset_pred, time_pred, bpm_gt, sr=22050, hop_length=512, threshold=0.5,
                                 output_midi_path="output.mid"):

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    onset_pred = onset_pred.squeeze(0)  # [num_beats, NUM_INSTR]
    time_pred = time_pred.squeeze(0)  # [num_beats, 1]

    num_beats = onset_pred.shape[0]

    if len(beat_times) > 1:
        beat_durations = np.diff(beat_times)
        default_duration = np.median(beat_durations)
    else:
        default_duration = 0.5

    index_to_midi = {0: 35, 1: 38, 2: 42, 3: 44, 4: 46, 5: 41, 6: 45, 7: 49, 8: 51}

    note_events = []
    for i in range(num_beats):
        if i < len(beat_times):
            beat_start = beat_times[i]
        else:
            beat_start = beat_times[-1]
        if i < len(beat_times) - 1:
            beat_duration = beat_times[i + 1] - beat_times[i]
        else:
            beat_duration = default_duration

        rel_time = time_pred[i, 0].item()
        onset_time = beat_start + rel_time * beat_duration

        for instr in range(NUM_INSTR):
            if onset_pred[i, instr] >= threshold:
                note = pretty_midi.Note(velocity=100, pitch=index_to_midi[instr], start=onset_time,
                                        end=onset_time + 0.1)
                note_events.append(note)

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm_gt)
    drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
    drum_instrument.notes = note_events
    pm.instruments.append(drum_instrument)
    pm.write(output_midi_path)
    print(f"MIDI file written to {output_midi_path}")


def write_model_output_to_text(audio_path, midi_path, bpm_gt, model, device,
                               output_txt_path="model_output.txt",
                               threshold=0.5, sr=22050, n_bins=84,
                               bins_per_octave=12, hop_length=512, default_duration=0.1):

    y, sr = librosa.load(os.path.join(GROOVE_MIDI_DATASET_PATH,audio_path), sr=sr, mono=True)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_db = np.expand_dims(cqt_db, axis=0)  # [1, n_bins, time_frames]

    beat_sync, beat_frames, tempo = beat_sync_features(cqt_db, y, sr, hop_length, aggregate=np.median)

    _, beat_frames_full = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames_full, sr=sr, hop_length=hop_length)

    num_beats = beat_sync.shape[-1]
    cqt_sync_tensor = torch.tensor(beat_sync[:, :, :num_beats], dtype=torch.float32).to(device)
    cqt_sync_tensor = cqt_sync_tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        onset_pred, time_pred = model(cqt_sync_tensor)
    onset_pred = onset_pred.squeeze(0)  # [num_beats, NUM_INSTR]
    time_pred = time_pred.squeeze(0)  # [num_beats, 1]


    onset_pred_bin = (torch.sigmoid(onset_pred) > threshold).float().cpu().numpy()  # [num_beats, NUM_INSTR]
    time_pred_np = time_pred.cpu().numpy()  # [num_beats, 1]

    index_to_instr = {0: "Kick", 1: "Snare", 2: "Closed Hi-Hat", 3: "Pedal Hi-Hat",
                      4: "Open Hi-Hat", 5: "Floor Tom", 6: "Rack Tom", 7: "Crash", 8: "Ride"}

    events_output = []
    num_beats_model = onset_pred_bin.shape[0]
    for i in range(num_beats_model):
        if i < len(beat_times) - 1:
            beat_start = beat_times[i]
            beat_duration = beat_times[i + 1] - beat_times[i]
        else:
            beat_start = beat_times[-1] if len(beat_times) > 0 else 0.0
            beat_duration = default_duration

        rel_onset = time_pred_np[i, 0]
        abs_onset = beat_start + rel_onset * beat_duration

        instruments = []
        for instr in range(NUM_INSTR):
            if onset_pred_bin[i, instr] >= threshold:
                instruments.append(index_to_instr.get(instr, f"Instr {instr}"))
        if instruments:
            event_str = f"Beat {i}: Onset at {abs_onset:.3f}s, Duration {default_duration:.3f}s, Instruments: {', '.join(instruments)}"
            events_output.append(event_str)

    with open(output_txt_path, "w") as f:
        f.write(f"Audio: {audio_path}\n")
        f.write(f"MIDI: {midi_path}\n")
        f.write(f"BPM from CSV: {bpm_gt}, Detected tempo: {float(tempo):.2f} BPM\n")
        f.write("Predicted Events:\n")
        for event in events_output:
            f.write(event + "\n")
    print(f"Model output written to {output_txt_path}")
    return events_output


def log_sample_predictions(cqt_sync_tensor, labels_sync_tensor, time_gt_tensor, onset_pred, time_pred, threshold=0.5):
    index_to_instr = {0: "Kick", 1: "Snare", 2: "Closed Hi-Hat", 3: "Pedal Hi-Hat",
                      4: "Open Hi-Hat", 5: "Floor Tom", 6: "Rack Tom", 7: "Crash", 8: "Ride"}

    onset_pred = onset_pred.squeeze(0)  # [num_beats, NUM_INSTR]
    time_pred = time_pred.squeeze(0)  # [num_beats, 1]

    labels_np = labels_sync_tensor.cpu().numpy()  # [num_beats, NUM_INSTR]
    time_gt_np = time_gt_tensor.cpu().numpy()  # [num_beats, 1]
    onset_pred_bin = (torch.sigmoid(onset_pred) > threshold).float().cpu().numpy()  # [num_beats, NUM_INSTR]
    time_pred_np = time_pred.cpu().numpy()  # [num_beats, 1]

    num_beats = labels_np.shape[0]
    print("Logging sample predictions:")
    for i in range(num_beats):
        gt_instr = [index_to_instr[j] for j in range(NUM_INSTR) if labels_np[i, j] > 0]
        pred_instr = [index_to_instr[j] for j in range(NUM_INSTR) if onset_pred_bin[i, j] > 0]
        gt_time = time_gt_np[i, 0] if i < time_gt_np.shape[0] else None
        pred_time = time_pred_np[i, 0]
        gt_time_str = f"{gt_time:.3f}" if gt_time is not None else "N/A"
        print(f"Beat {i}: GT Instruments: {gt_instr if gt_instr else 'None'}, GT Time: {gt_time_str} | "
              f"Pred Instruments: {pred_instr if pred_instr else 'None'}, Pred Time: {pred_time:.3f}")


if __name__ == "__main__":
    #debug_end_to_end_model()
    print("Pipeline started")
    train_end_to_end_model()
    print("Pipeline finished")

    audio_path = "/Users/piotrek/DataSets/grooveAndMidi/drummer3/session2/2_rock_100_beat_4-4.wav"
    midi_path = "/Users/piotrek/DataSets/grooveAndMidi/drummer3/session2/2_rock_100_beat_4-4.mid"

    #model = DrumEndToEndModel(num_instruments=NUM_INSTR, hidden_size=128).to(device)
    #model.load_state_dict(torch.load("best_model.pth", map_location=device))

    #onset_pred, time_pred, labels_sync_tensor, time_gt_tensor, bpm_gt = test_on_recording(audio_path, midi_path, model, device)

    #convert_model_output_to_midi(audio_path, onset_pred, time_pred, bpm_gt, sr=22050, hop_length=512,
                                 #output_midi_path="converted_output.mid")

    #events_output = write_model_output_to_text(audio_path, midi_path, bpm_gt, model, device)
    #for event in events_output:
        #print(event)
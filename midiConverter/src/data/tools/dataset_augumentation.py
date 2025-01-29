import os
import torch
import random
import optuna
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data.dataset_loader import load_dataset_with_splits
from src.models.constants import DRUM_INSTRUMENTS

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
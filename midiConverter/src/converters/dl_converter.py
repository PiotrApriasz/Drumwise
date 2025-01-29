import os

import torch
import librosa
import numpy as np

from src.models.constants import DRUM_INSTRUMENTS
from src.models.dl_models.cnn_classifier import DrumCNN, label_map


def set_length(audio: np.ndarray, desired_length: int) -> np.ndarray:
    if len(audio) > desired_length:
        audio = audio[:desired_length]
    else:
        audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')
    return audio

def classify_instrument_dl(path, model, sr=22050, duration=2):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = set_length(y, sr * duration)
    cqt = librosa.cqt(y, sr=sr, hop_length=512)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    x = torch.tensor(cqt_db, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        pred_idx = outputs.argmax(dim=1).item()
    inv_label_map = {v: k for k, v in label_map.items()}
    return inv_label_map[pred_idx]


def extract_true_label(file_name: str) -> str:
    name_lower = file_name.lower()
    if "kick" in name_lower:
        return "kick"
    elif "snare" in name_lower:
        return "snare"
    elif "hi-hat" in name_lower or "hihat" in name_lower:
        return "hi-hat"
    elif "crash" in name_lower:
        return "crash"
    elif "rack_tom" in name_lower:
        return "rack_tom"
    elif "floor_tom" in name_lower:
        return "floor_tom"
    elif "ride" in name_lower:
        return "ride"
    return "unknown"

if __name__ == "__main__":
    model_path = "best_drum_cnn.pth"
    loaded_model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    audio_files_path = "../../audio"
    correct_count = 0
    total_count = 0

    # Słownik zliczający błędy dla każdej etykiety
    incorrect_per_label = {label: 0 for label in DRUM_INSTRUMENTS}

    for file_name in os.listdir(audio_files_path):
        if file_name.endswith(".wav"):
            true_label = extract_true_label(file_name)
            if true_label == "unknown":
                continue
            audio_path = os.path.join(audio_files_path, file_name)
            predicted_label = classify_instrument_dl(audio_path, loaded_model)
            print(f"File: {file_name}")
            print(f"Predicted: {predicted_label}, True: {true_label}\n")
            total_count += 1
            if predicted_label == true_label:
                correct_count += 1
            else:
                incorrect_per_label[true_label] += 1

    print("\n--- PODSUMOWANIE ---")
    print(f"Poprawnych rozpoznań: {correct_count}")
    print(f"Błędnych rozpoznań:   {total_count - correct_count}")
    print(f"Łącznie porównano:    {total_count} plików")

    print("\n--- LICZBA BŁĘDÓW NA KLASĘ (TRUE LABEL) ---")
    for label, count in incorrect_per_label.items():
        print(f"{label}: {count}")

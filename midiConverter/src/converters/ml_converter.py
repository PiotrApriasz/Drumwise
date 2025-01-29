import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import joblib

from src.data.feature_extractor import extract_features_from_audio
from src.data.dataset_loader import set_length


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


def classify_instrument(path, model, saved_scaler, saved_label_encoder):

    y, sr = librosa.load(path, sr=22050, mono=True)

    y = set_length(y, 22050 * 2)

    instrument = path.split("/")[-1].split(".")[0]

    features = extract_features_from_audio(y, sr)

    # plt.figure(figsize=(8, 4))
    # plt.plot(features, marker='o')
    # plt.title("Feature Vector for " + instrument)
    # plt.xlabel("Feature Index")
    # plt.ylabel("Feature Value")
    # plt.grid(True)
    # plt.show()

    X_new = np.array([features])

    X_new_scaled = saved_scaler.transform(X_new)

    y_pred = model.predict(X_new_scaled)

    predicted_class = saved_label_encoder.inverse_transform(y_pred)

    return predicted_class[0]

if __name__ == "__main__":
    svm_model = joblib.load('../models/trained_models/ml_trained_models/stackingensemble.pkl')
    scaler = joblib.load('../models/trained_models/ml_trained_models/scaler.pkl')
    label_encoder = joblib.load('../models/trained_models/ml_trained_models/encoder.pkl')

    # just to test
    audio_files_path = "../../audio"

    correct_count = 0
    total_count = 0

    for file_name in os.listdir(audio_files_path):
        if file_name.endswith(".wav"):
            true_label = extract_true_label(file_name)
            audio_path = os.path.join(audio_files_path, file_name)

            print("\nChecking drum instrument: ", file_name)
            predicted_label = classify_instrument(audio_path, svm_model, scaler, label_encoder)
            print("Predicated drum instrument:", predicted_label)

            if true_label != "unknown":
                total_count += 1
                if predicted_label == true_label:
                    correct_count += 1

    print("\n--- SUMMARY ---")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions:   {total_count - correct_count}")
    print(f"Sum of predicated:    {total_count}")
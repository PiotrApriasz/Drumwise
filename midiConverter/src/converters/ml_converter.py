import os

import librosa
import numpy as np
import joblib

from src.data.feature_extractor import extract_features_from_audio
from src.data.dataset_loader import set_length


def classify_instrument(audio_path, model, saved_scaler, saved_label_encoder):

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    y = set_length(y, 22050 * 2)

    features = extract_features_from_audio(y, sr)

    X_new = np.array([features])

    X_new_scaled = saved_scaler.transform(X_new)

    y_pred = model.predict(X_new_scaled)

    predicted_class = saved_label_encoder.inverse_transform(y_pred)

    return predicted_class[0]

if __name__ == "__main__":
    svm_model = joblib.load('../models/trained_models/stackingensemble.pkl')
    scaler = joblib.load('../models/trained_models/scaler.pkl')
    label_encoder = joblib.load('../models/trained_models/encoder.pkl')

    # just to test
    audio_files_path = "../../audio"

    for file_name in os.listdir(audio_files_path):
        print("\nChecking drum instrument: ", file_name)
        audio_path = os.path.join(audio_files_path, file_name)
        predicted_label = classify_instrument(audio_path, svm_model, scaler, label_encoder)
        print("Predicated drum instrument:", predicted_label)
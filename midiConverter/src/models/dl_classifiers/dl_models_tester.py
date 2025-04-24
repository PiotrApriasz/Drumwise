import os

import torch

from src.converters.conversion_utils import extract_true_label, classify_instrument_dl
from src.constants import DRUM_INSTRUMENTS, TRAINED_DL_MODELS_PATH, MEL_BEST_CNN_MODEL_NAME, CQT_BEST_CNN_MODEL_NAME, \
    CUTTED_VAL_SETS_PATH
from src.models.dl_classifiers.classifiers.cnn_classifier import DrumCNN
from src.models.dl_classifiers.classifiers.heterogeneous_cnn_classifier import HeterogeneousCnnEnsemble

if __name__ == "__main__":
    use_ensemble = True
    
    if use_ensemble:
        model_paths = [
            os.path.join(TRAINED_DL_MODELS_PATH, MEL_BEST_CNN_MODEL_NAME),
            os.path.join(TRAINED_DL_MODELS_PATH, CQT_BEST_CNN_MODEL_NAME)
        ]
        model_types = ['mel', 'cqt']
        model = HeterogeneousCnnEnsemble(model_paths, model_types)
    else:
        model_path = os.path.join(TRAINED_DL_MODELS_PATH, MEL_BEST_CNN_MODEL_NAME)
        model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
        model.load_state_dict(torch.load(model_path))
        model.eval()

    audio_files_path = CUTTED_VAL_SETS_PATH
    correct_count = 0
    total_count = 0

    incorrect_per_label = {label: 0 for label in DRUM_INSTRUMENTS}

    for file_name in os.listdir(audio_files_path):
        if file_name.endswith(".wav"):
            true_label = extract_true_label(file_name)
            if true_label == "unknown":
                continue
            audio_path = os.path.join(audio_files_path, file_name)
            predicted_label = classify_instrument_dl(audio_path, model, use_ensemble=use_ensemble)
            print(f"File: {file_name}")
            print(f"Predicted: {predicted_label}, True: {true_label}\n")
            total_count += 1
            if predicted_label == true_label:
                correct_count += 1
            else:
                incorrect_per_label[true_label] += 1

    print("\n--- SUMMARY ---")
    print(f"Correct classifications: {correct_count}")
    print(f"Incorrect classifications: {total_count - correct_count}")
    print(f"Total files compared: {total_count}")
    print(f"Accuracy: {correct_count/total_count:.4f}")

    print("\n--- ERRORS PER CLASS (TRUE LABEL) ---")
    for label, count in incorrect_per_label.items():
        print(f"{label}: {count}")

import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data.dataset_loader import load_data_set
from src.data.feature_extractor import extract_features
from src.models.ml_models.standard_classifiers import train_standard_classifiers
from src.models.ml_models.easy_ensemble_classifiers import train_ensemble_classifiers
from src.models.ml_models.hyperparameter_tuner import test_best_models, tune_models
from src.models.ml_models.ensemble_classifiers import build_ensemble_classifiers


def prepare_final_data() -> tuple[np.ndarray, np.ndarray]:

    data = load_data_set()
    feats = extract_features(data)

    X = [f[0] for f in feats]
    y = [f[1] for f in feats]

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X_scaled, y_encoded


def run_ml_pipeline():

    X_scaled, y_encoded = prepare_final_data()

    print("\n--- Standard Classifiers ---")
    train_standard_classifiers(X_scaled, y_encoded)

    print("\n--- Ensemble Classifiers ---")
    train_ensemble_classifiers(X_scaled, y_encoded)

    print("\n--- Tuning Models ---")
    tuned_rf, tuned_svm = tune_models(X_scaled, y_encoded)

    print("\n--- Testing Tuned Models ---")
    test_best_models(X_scaled, y_encoded, tuned_rf, tuned_svm)

    print("\n--- Final Ensemble ---")
    build_ensemble_classifiers(X_scaled, y_encoded, tuned_rf, tuned_svm)


if __name__ == "__main__":
    run_ml_pipeline()

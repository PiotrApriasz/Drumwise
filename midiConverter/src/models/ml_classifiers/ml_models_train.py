from src.models.ml_classifiers.ml_dataset_creator import prepare_final_data
from src.models.ml_classifiers.classifiers.standard_classifiers import train_standard_classifiers
from src.models.ml_classifiers.classifiers.easy_ensemble_classifiers import train_ensemble_classifiers
from src.models.ml_classifiers.ml_hyperparameter_tuner import test_best_models, tune_models
from src.models.ml_classifiers.classifiers.ensemble_classifiers import build_ensemble_classifiers


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

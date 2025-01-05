import os
import librosa
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier


basic_data_set_path = "/Users/piotrek/DataSets/DrumInstrumentsDataSet"
drum_instruments = ["kick", "snare", "overheads", "toms"]

def load_data_set():
    data_set = []
    for drum_instrument in drum_instruments:
        for file in os.listdir(basic_data_set_path + "/" + drum_instrument):
            if file.endswith(".wav"):
                file_path = basic_data_set_path + "/" + drum_instrument + "/" + file
                data, sampling_rate = librosa.load(file_path, sr=22050, mono=True)
                data_set.append([data, drum_instrument])

    return data_set

def extract_features_from_audio(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr, axis=1)

    features = np.concatenate((mfcc_mean, spec_centroid_mean, zcr_mean), axis=0)

    return features

def extract_features(data):
    features = []
    for audio in data:
        features.append([extract_features_from_audio(audio[0], 22050), audio[1]])

    return features

def prepare_x_and_y_variables(features):
    X = []
    y = []

    for feat_vector, label in features:
        X.append(feat_vector)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y

def prepare_training_and_test_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

def prepare_final_data():
    loaded_data_set = load_data_set()
    extracted_features = extract_features(loaded_data_set)
    X, y = prepare_x_and_y_variables(extracted_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X_scaled, y_encoded

def train_standard_classifiers(X, y):
    X_train, X_test, y_train, y_test = prepare_training_and_test_data(X, y)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='rbf'),  # domyślny kernel to rbf
        "DecisionTree": DecisionTreeClassifier(),
        "kNN": KNeighborsClassifier(n_neighbors=5)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n=== {model_name} ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"{model_name}: CV Accuracy Mean={scores.mean():.4f}, Std={scores.std():.4f}")


def train_ensemble_classifiers(X, y):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== Wyniki Cross-Validation (5-Fold) dla modeli ensemblowych ===\n")
    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"{model_name}: CV Accuracy Mean={scores.mean():.4f}, Std={scores.std():.4f}")

    X_train, X_test, y_train, y_test = prepare_training_and_test_data(X, y)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n=== {model_name} (train/test split) ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def tune_models(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    rf_grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        scoring='accuracy',
        cv=skf,
        n_jobs=-1
    )

    rf_grid_search.fit(X, y)

    print("\n--- Random Forest ---")
    print("Best params:", rf_grid_search.best_params_)
    print("Best CV accuracy:", rf_grid_search.best_score_)

    best_rf = rf_grid_search.best_estimator_

    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale']
    }

    svm_grid_search = GridSearchCV(
        estimator=SVC(kernel='rbf', random_state=42),
        param_grid=svm_param_grid,
        scoring='accuracy',
        cv=skf,
        n_jobs=-1
    )

    svm_grid_search.fit(X, y)

    print("\n--- SVM (RBF) ---")
    print("Best params:", svm_grid_search.best_params_)
    print("Best CV accuracy:", svm_grid_search.best_score_)

    best_svm = svm_grid_search.best_estimator_

    return best_rf, best_svm

def test_best_models(X, y, tuned_rf, tuned_svm):
    X_train, X_test, y_train, y_test = prepare_training_and_test_data(X, y)

    tuned_rf.fit(X_train, y_train)
    y_pred_rf = tuned_rf.predict(X_test)

    print("\n--- Tuned RandomForest ---")
    print("Test accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    print(confusion_matrix(y_test, y_pred_rf))

    # Test SVM
    tuned_svm.fit(X_train, y_train)
    y_pred_svm = tuned_svm.predict(X_test)

    print("\n--- Tuned SVM ---")
    print("Test accuracy:", accuracy_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))
    print(confusion_matrix(y_test, y_pred_svm))

def build_and_test_final_ensemble(X, y, tuned_rf, tuned_svm):
    tuned_svm_with_proba = SVC(
        kernel='rbf',
        C=tuned_svm.get_params()['C'],
        gamma=tuned_svm.get_params()['gamma'],
        probability=True,
        random_state=42
    )

    tuned_svm_with_proba.fit(X, y)

    voting_clf = VotingClassifier(
        estimators=[
            ('rf', tuned_rf),
            ('svm', tuned_svm_with_proba)
        ],
        voting='soft'
    )

    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', tuned_rf),
            ('svm', tuned_svm_with_proba)
        ],
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba'
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in [('VotingEnsemble', voting_clf), ('StackingEnsemble', stacking_clf)]:
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"{name} - CV Accuracy Mean={scores.mean():.4f}, Std={scores.std():.4f}")

    X_train, X_test, y_train, y_test = prepare_training_and_test_data(X, y)

    for name, model in [('VotingEnsemble', voting_clf), ('StackingEnsemble', stacking_clf)]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n=== {name} (train/test split) ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


X_scaled, y_encoded = prepare_final_data()

print("\n--------------------------------------- Standard Classifiers ---------------------------------------")
train_standard_classifiers(X_scaled, y_encoded)
print("\n--------------------------------------- Easy Ensemble Classifiers ---------------------------------------")
train_ensemble_classifiers(X_scaled, y_encoded)
print("\n--------------------------------------- Tuned Classifiers ---------------------------------------")
tuned_rf, tuned_svm = tune_models(X_scaled, y_encoded)
print("\n--------------------------------------- Test Tuned Classifiers ---------------------------------------")
test_best_models(X_scaled, y_encoded, tuned_rf, tuned_svm)
print("\n--------------------------------------- FINAL ENSEMBLE: Voting & Stacking ---------------------------------------")
build_and_test_final_ensemble(X_scaled, y_encoded, tuned_rf, tuned_svm)


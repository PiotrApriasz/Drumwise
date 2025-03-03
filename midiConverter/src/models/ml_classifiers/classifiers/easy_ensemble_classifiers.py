import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_ensemble_classifiers(X: np.ndarray, y: np.ndarray) -> None:

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== Wyniki Cross-Validation (5-Fold) ===\n")
    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"{model_name}: CV Accuracy Mean={scores.mean():.4f}, Std={scores.std():.4f}")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=42
    )

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n=== {model_name} (train/test split) ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

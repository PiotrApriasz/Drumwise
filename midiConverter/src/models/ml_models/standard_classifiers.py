import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def train_standard_classifiers(X: np.ndarray, y: np.ndarray) -> None:
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='rbf'),
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

    print("\n=== Cross Validation (5-Fold) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"{model_name}: CV Accuracy Mean={scores.mean():.4f}, Std={scores.std():.4f}")

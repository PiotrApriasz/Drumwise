import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def build_ensemble_classifiers(X: np.ndarray, y: np.ndarray, tuned_rf, tuned_svm) -> None:

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

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in [('VotingEnsemble', voting_clf), ('StackingEnsemble', stacking_clf)]:
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"{name} - CV Accuracy Mean={scores.mean():.4f}, Std={scores.std():.4f}")

    # train/test split
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=42
    )

    for name, model in [('VotingEnsemble', voting_clf), ('StackingEnsemble', stacking_clf)]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n=== {name} (train/test split) ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
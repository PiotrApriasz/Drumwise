import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def tune_models(X: np.ndarray, y: np.ndarray):

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

    # SVM
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

def test_best_models(X: np.ndarray, y: np.ndarray, tuned_rf, tuned_svm) -> None:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=42
    )

    tuned_rf.fit(X_train, y_train)
    y_pred_rf = tuned_rf.predict(X_test)

    print("\n--- Tuned RandomForest ---")
    print("Test accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    print(confusion_matrix(y_test, y_pred_rf))

    tuned_svm.fit(X_train, y_train)
    y_pred_svm = tuned_svm.predict(X_test)

    print("\n--- Tuned SVM ---")
    print("Test accuracy:", accuracy_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))
    print(confusion_matrix(y_test, y_pred_svm))

    joblib.dump(tuned_svm, 'models/trained_models/tuned_svm.pkl')

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def calculate_basic_metrics(y_true, y_pred_classes):
    """
    Calculates basic classification metrics.
    Args:
        y_true: True labels.
        y_pred_classes: Predicted classes (argmax of model output).
    Returns:
        A dictionary of metrics.
    """
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred_classes, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred_classes, average='micro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred_classes, average='weighted', zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def calculate_per_class_metrics(y_true, y_pred_classes, class_names=None):
    """
    Calculates per-class precision, recall, F1-score, and support.
    Uses sklearn.metrics.classification_report.
    Args:
        y_true: True labels.
        y_pred_classes: Predicted classes.
        class_names: List of class names corresponding to label indices.
    Returns:
        A dictionary containing the classification report (parsed as a dict).
    """
    if class_names is None:
        # Attempt to infer class names if not provided, assuming labels are 0 to N-1
        unique_labels = np.unique(np.concatenate((y_true, y_pred_classes)))
        target_names = [f"class_{i}" for i in unique_labels]
    else:
        target_names = class_names

    report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True, zero_division=0)
    return {"classification_report": report}


def generate_confusion_matrix(y_true, y_pred_classes, num_classes=None):
    """
    Generates the confusion matrix.
    Args:
        y_true: True labels.
        y_pred_classes: Predicted classes.
        num_classes: Optional. The total number of classes. If provided, ensures the matrix has size num_classes x num_classes.
    Returns:
        A numpy array representing the confusion matrix.
    """
    if num_classes:
        labels = list(range(num_classes))
        cm = confusion_matrix(y_true, y_pred_classes, labels=labels)
    else:
        cm = confusion_matrix(y_true, y_pred_classes)
    return cm


# Example of how you might combine these:
def get_all_single_label_metrics(y_true, y_pred_classes, y_pred_proba=None, class_names=None, num_classes=None):
    """
    Calculates and combines all relevant single-label metrics.
    Args:
        y_true: True labels (list or np.array).
        y_pred_classes: Predicted classes (argmax of model output).
        y_pred_proba: Raw probability outputs from the model (optional, for future ROC/AUC).
        class_names: List of class names for reporting.
        num_classes: Total number of classes for consistent confusion matrix size.
    Returns:
        A comprehensive dictionary of all calculated metrics.
    """
    metrics = {}

    basic_metrics = calculate_basic_metrics(y_true, y_pred_classes)
    metrics.update(basic_metrics)

    per_class_report = calculate_per_class_metrics(y_true, y_pred_classes, class_names=class_names)
    metrics.update(per_class_report)  # This nests the report dict under "classification_report"

    # The confusion matrix is already calculated in base_classifier.evaluate_model
    # but if we want it consistently from here:
    # conf_matrix = generate_confusion_matrix(y_true, y_pred_classes, num_classes=num_classes)
    # metrics["confusion_matrix_calculated_here"] = conf_matrix
    # For now, we rely on the one from evaluate_model

    # Placeholder for metrics requiring y_pred_proba (e.g., ROC AUC)
    if y_pred_proba is not None:
        # Example: roc_auc = calculate_roc_auc_scores(y_true, y_pred_proba)
        # metrics.update(roc_auc)
        pass

    return metrics

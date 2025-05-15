import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, \
    roc_auc_score


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

def _softmax_numpy(logits, axis=1):
    """
    Computes softmax activations numerically stably.
    Args:
        logits: Numpy array of logits.
        axis: Axis or axes along which the softmax should be computed.
    Returns:
        Numpy array of probabilities.
    """
    logits = np.asarray(logits)
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)



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
        max_label = np.max(unique_labels) if len(unique_labels) > 0 else -1
        target_names = [f"class_{i}" for i in range(max_label + 1)]
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


def calculate_roc_auc_scores(y_true, y_pred_logits, num_classes=None, class_names=None, average_methods=None):
    """
    Calculates ROC AUC scores for multi-class classification.
    Args:
        y_true: True labels (1D array).
        y_pred_logits: Raw logit outputs from the model (2D array: n_samples, n_classes).
        num_classes: Total number of classes. Essential for roc_auc_score with 'ovr'/'ovo'.
        class_names: List of class names for per-class ROC AUC reporting.
        average_methods: List of averaging methods for roc_auc_score (e.g., ['macro', 'weighted']).
                         Defaults to ['macro', 'weighted'].
    Returns:
        A dictionary of ROC AUC scores.
    """
    if average_methods is None:
        average_methods = ['macro', 'weighted']

    roc_auc_metrics = {}

    if y_pred_logits is None:
        return roc_auc_metrics

    y_pred_logits = np.asarray(y_pred_logits)

    if y_pred_logits.ndim != 2 or y_pred_logits.shape[0] == 0:
        print("Warning: y_pred_logits not in expected format (n_samples, n_classes) for ROC AUC or no samples.")
        return roc_auc_metrics

    current_num_classes = y_pred_logits.shape[1]
    if num_classes is None:
        num_classes = current_num_classes
    elif num_classes != current_num_classes:
        print(
            f"Warning: Provided num_classes ({num_classes}) does not match y_pred_logits columns ({current_num_classes}). Using {current_num_classes}.")
        num_classes = current_num_classes

    if num_classes < 2:
        # ROC AUC is not well-defined or meaningful for single-class scenarios from multi-class outputs
        return roc_auc_metrics

    y_pred_proba = _softmax_numpy(y_pred_logits, axis=1)

    # Ensure y_true is 1D array
    y_true = np.asarray(y_true).ravel()

    unique_true_labels = np.unique(y_true)
    if len(unique_true_labels) < 2:
        print(
            f"Warning: ROC AUC may be ill-defined as y_true contains only {len(unique_true_labels)} unique class(es). Scores might be NaN or 0.5.")
        # Scikit-learn's roc_auc_score might still compute, often resulting in 0.5 for 'ovr' if only one class present.

    labels_param = list(range(num_classes))

    for avg_method in average_methods:
        try:
            roc_auc = roc_auc_score(
                y_true,
                y_pred_proba,
                multi_class='ovr',
                average=avg_method,
                labels=labels_param
            )
            roc_auc_metrics[f"roc_auc_ovr_{avg_method}"] = roc_auc
        except ValueError as e:
            roc_auc_metrics[f"roc_auc_ovr_{avg_method}"] = np.nan
            print(f"Could not calculate ROC AUC OVR ({avg_method}) for {num_classes} classes: {e}")

    if class_names and len(class_names) == num_classes:
        try:
            per_class_roc_auc = roc_auc_score(
                y_true,
                y_pred_proba,
                multi_class='ovr',
                average=None,  # Returns array of scores for each class
                labels=labels_param
            )
            for i, class_name in enumerate(class_names):
                safe_class_name = str(class_name).replace(' ', '_').replace('/', '_')  # Make it a safe key
                if i < len(per_class_roc_auc):
                    roc_auc_metrics[f"roc_auc_ovr_class_{safe_class_name}"] = per_class_roc_auc[i]
                else:
                    roc_auc_metrics[f"roc_auc_ovr_class_{safe_class_name}"] = np.nan
        except ValueError as e:
            print(f"Could not calculate per-class ROC AUC OVR for {num_classes} classes: {e}")
            for i, class_name in enumerate(class_names):  # Initialize with NaN if calculation fails
                safe_class_name = str(class_name).replace(' ', '_').replace('/', '_')
                roc_auc_metrics[f"roc_auc_ovr_class_{safe_class_name}"] = np.nan

    return roc_auc_metrics


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

    conf_matrix = generate_confusion_matrix(y_true, y_pred_classes, num_classes=num_classes)
    metrics["confusion_matrix_calculated_here"] = conf_matrix

    if y_pred_proba is not None:
        inferred_num_classes_for_roc = num_classes
        if inferred_num_classes_for_roc is None and hasattr(y_pred_proba, 'shape') and len(y_pred_proba.shape) == 2:
            inferred_num_classes_for_roc = y_pred_proba.shape[1]

        roc_class_names = class_names
        if class_names and inferred_num_classes_for_roc and len(class_names) != inferred_num_classes_for_roc:
            print(
                f"Warning: Length of class_names ({len(class_names)}) for ROC AUC does not match inferred num_classes ({inferred_num_classes_for_roc}). Per-class ROC AUC by name might be affected.")
            roc_class_names = None

        roc_auc_results = calculate_roc_auc_scores(
            y_true,
            y_pred_logits=y_pred_proba,
            num_classes=inferred_num_classes_for_roc,
            class_names=roc_class_names
        )
        metrics.update(roc_auc_results)

    return metrics


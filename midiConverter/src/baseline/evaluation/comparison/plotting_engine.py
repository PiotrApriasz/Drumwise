import numpy as np
import pandas as pd
import sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from src.constants import DRUM_INSTRUMENTS


def plot_accuracy_comparison(results_dict, metric='test_acc', 
                            title='Model Accuracy Comparison',
                            sort=True):
    """
    Creates a bar chart comparing accuracy across models.
    
    Args:
        results_dict: Dictionary of experiment results
        metric: Which metric to plot ('test_acc', 'f1_macro', etc.)
        title: Plot title
        sort: Whether to sort bars by performance
    """
    metrics_data = []
    
    for exp_name, data in results_dict.items():
        # Get the model and feature type for display
        model_type = data.get('model_type', exp_name.split('_')[0])
        feature_type = data.get('feature_type', exp_name.split('_')[1] if len(exp_name.split('_')) > 1 else 'unknown')
        
        # Look for metric in test_set_evaluation or directly in results
        metric_value = None
        if 'test_set_evaluation' in data and metric in data['test_set_evaluation']:
            metric_value = data['test_set_evaluation'][metric]
        elif metric in data:
            metric_value = data[metric]
            
        if metric_value is not None:
            metrics_data.append({
                'Model': f"{model_type.upper()}",
                'Feature': feature_type.upper(),
                'Label': f"{model_type.upper()} ({feature_type.upper()})",
                'Value': metric_value
            })
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(metrics_data)
    if sort and not df.empty:
        df = df.sort_values('Value', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df['Label'], df['Value'], color=plt.cm.viridis(np.linspace(0, 1, len(df))))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.3f}", ha='center', va='bottom')
    
    plt.xlabel('Model (Feature Type)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1*df['Value'].max())
    plt.tight_layout()
    
    return plt.gcf()


def plot_roc_curves(results_dict, class_idx=None, 
                   title="ROC Curve Comparison",
                   figsize=(12, 8)):
    """
    Plots ROC curves for multiple models, either for a specific class
    or averaged across all classes.
    
    Args:
        results_dict: Dictionary of experiment results
        class_idx: Index of class to plot, or None for macro-average
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    for exp_name, data in results_dict.items():
        model_type = data.get('model_type', exp_name.split('_')[0]).upper()
        feature_type = data.get('feature_type', exp_name.split('_')[1] if len(exp_name.split('_')) > 1 else 'unknown').upper()
        
        # Extract ROC data
        if 'test_set_evaluation' in data:
            eval_data = data['test_set_evaluation']
        else:
            eval_data = data
            
        if class_idx is None:
            # Plot macro-average ROC
            if 'roc_auc_ovr_macro' in eval_data:
                # We need to reconstruct FPR and TPR from raw prediction data
                y_true = np.array(eval_data.get('labels', []))
                y_pred = np.array(eval_data.get('outputs', []))
                
                if len(y_true) > 0 and len(y_pred) > 0:
                    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
                    roc_auc = eval_data.get('roc_auc_ovr_macro', 0)
                    plt.plot(fpr, tpr, lw=2, label=f'{model_type} ({feature_type}) - AUC: {roc_auc:.3f}')
        else:
            # Plot specific class ROC
            class_name = list(DRUM_INSTRUMENTS)[class_idx]
            roc_key = f"roc_curve_{class_name}"
            
            if roc_key in eval_data:
                roc_data = eval_data[roc_key]
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
                auc_value = roc_data['auc']
                plt.plot(fpr, tpr, lw=2, label=f'{model_type} ({feature_type}) - AUC: {auc_value:.3f}')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def plot_learning_curves(results_dict, metric='val_acc',
                         title='Learning Curves Comparison',
                         figsize=(12, 6)):
    """
    Plots learning curves (metric vs epoch) for multiple models.

    Args:
        results_dict: Dictionary of experiment results
        metric: Which training metric to plot ('val_acc', 'loss', etc.)
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)

    for exp_name, data in results_dict.items():
        model_type = data.get('model_type', exp_name.split('_')[0]).upper()
        feature_type = data.get('feature_type',
                                exp_name.split('_')[1] if len(exp_name.split('_')) > 1 else 'unknown').upper()

        if 'training_history' in data:
            history = data['training_history']
            if metric in history and len(history[metric]) > 0:
                epochs = range(1, len(history[metric]) + 1)
                plt.plot(epochs, history[metric], 'o-', label=f'{model_type} ({feature_type})')

    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def plot_class_performance(results_dict, metric='f1-score',
                           title='Class Performance Comparison',
                           figsize=(14, 8)):
    """
    Compares model performance on each class.

    Args:
        results_dict: Dictionary of experiment results
        metric: Which metric to plot ('precision', 'recall', 'f1-score')
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)

    # Prepare data structure
    class_metrics = {instrument: [] for instrument in DRUM_INSTRUMENTS}
    model_names = []

    for exp_name, data in results_dict.items():
        model_type = data.get('model_type', exp_name.split('_')[0]).upper()
        feature_type = data.get('feature_type',
                                exp_name.split('_')[1] if len(exp_name.split('_')) > 1 else 'unknown').upper()
        model_label = f"{model_type} ({feature_type})"
        model_names.append(model_label)

        # Extract classification report
        if 'test_set_evaluation' in data and 'classification_report' in data['test_set_evaluation']:
            report = data['test_set_evaluation']['classification_report']
        elif 'classification_report' in data:
            report = data['classification_report']
        else:
            continue

        # Get metric value for each class
        for instrument in DRUM_INSTRUMENTS:
            if instrument in report and metric in report[instrument]:
                class_metrics[instrument].append(report[instrument][metric])
            else:
                class_metrics[instrument].append(0)  # Default if missing

    # Convert to DataFrame for plotting
    df = pd.DataFrame(class_metrics, index=model_names)

    # Plot
    ax = df.plot(kind='bar', figsize=figsize, width=0.8)

    plt.title(title)
    plt.ylabel(metric.title())
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Drum Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_confusion_matrix_grid(results_dict, figsize=(18, 12), normalize=True):
    """
    Creates a grid of confusion matrices for visual comparison.

    Args:
        results_dict: Dictionary of experiment results
        figsize: Overall figure size
        normalize: Whether to normalize confusion matrices
    """
    n_models = len(results_dict)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (exp_name, data) in enumerate(results_dict.items()):
        if i >= len(axes):
            break

        model_type = data.get('model_type', exp_name.split('_')[0]).upper()
        feature_type = data.get('feature_type',
                                exp_name.split('_')[1] if len(exp_name.split('_')) > 1 else 'unknown').upper()

        # Extract confusion matrix
        cm = None
        if 'test_set_evaluation' in data:
            eval_data = data['test_set_evaluation']
            cm = eval_data.get('confusion_matrix')
            if cm is None and 'classification_report' in eval_data:
                cm = eval_data['classification_report'].get('confusion_matrix_calculated_here')

        if cm is None:
            cm = data.get('confusion_matrix')
            if cm is None and 'classification_report' in data:
                cm = data['classification_report'].get('confusion_matrix_calculated_here')

        if cm is not None:
            cm = np.array(cm)
            if normalize and cm.sum() > 0:
                cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                        xticklabels=DRUM_INSTRUMENTS, yticklabels=DRUM_INSTRUMENTS,
                        cmap='Blues', ax=axes[i])
            axes[i].set_title(f"{model_type} ({feature_type})")
            axes[i].set_ylabel('True label')
            axes[i].set_xlabel('Predicted label')

    # Hide unused subplots
    for i in range(len(results_dict), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig



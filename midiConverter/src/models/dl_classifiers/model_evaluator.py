import torch
import os
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from pathlib import Path

from src.constants import DRUM_INSTRUMENTS, TRAINED_DL_MODELS_PATH, CUTTED_VAL_SETS_PATH, SIMPLE_EVALUATION_RESULTS_PATH
from src.converters.conversion_utils import extract_true_label, classify_instrument_dl


def get_classifier_class(model_type: str):
    model_type = model_type.lower()

    if model_type == 'cnn':
        module_path = 'src.models.dl_classifiers.classifiers.cnn_classifier'
        class_name = 'DrumCNNClassifier'
    elif model_type == 'lstm':
        module_path = 'src.models.dl_classifiers.classifiers.lstm_classifier'
        class_name = 'DrumLSTMClassifier'
    elif model_type == 'transformer':
        module_path = 'src.models.dl_classifiers.classifiers.transformer_classifier'
        class_name = 'DrumTransformerClassifier'
    elif model_type == 'cnn_lstm':
        module_path = 'src.models.dl_classifiers.classifiers.cnn_lstm_classifier'
        class_name = 'DrumCnnLstmClassifier'
    else:
        raise ValueError(f"Unknown model type: {model_type}.")

    try:
        module = importlib.import_module(module_path)
        classifier_class = getattr(module, class_name)
        return classifier_class
    except ModuleNotFoundError:
        raise ImportError(f"Could not import module: {module_path}. Ensure the file exists.")
    except AttributeError:
        raise ImportError(f"Could not find class: {class_name} in module: {module_path}.")
    except Exception as e:
        raise RuntimeError(f"Error getting classifier class for {model_type}: {e}")


def get_model_implementation_class(model_type: str):
    model_type = model_type.lower()

    if model_type == 'cnn':
        module_path = 'src.models.dl_classifiers.classifiers.cnn_classifier'
        class_name = 'DrumCNN'
    elif model_type == 'lstm':
        module_path = 'src.models.dl_classifiers.classifiers.lstm_classifier'
        class_name = 'DrumLSTM'
    elif model_type == 'transformer':
        module_path = 'src.models.dl_classifiers.classifiers.transformer_classifier'
        class_name = 'DrumTransformer'
    elif model_type == 'cnn_lstm':
        module_path = 'src.models.dl_classifiers.classifiers.cnn_lstm_classifier'
        class_name = 'DrumCnnLstm'
    else:
        raise ValueError(f"Unknown model type: {model_type}.")

    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class
    except ModuleNotFoundError:
        raise ImportError(f"Could not import module: {module_path}. Ensure the file exists.")
    except AttributeError:
        raise ImportError(f"Could not find class: {class_name} in module: {module_path}.")
    except Exception as e:
        raise RuntimeError(f"Error getting model class for {model_type}: {e}")


def get_model_paths(model_type: str, feature_type: str):
    model_type_upper = model_type.upper()
    feature_type_upper = feature_type.upper()

    best_model_const_name = f"{feature_type_upper}_BEST_{model_type_upper}_MODEL_NAME"

    try:
        from src import constants
        best_model_filename = getattr(constants, best_model_const_name)
    except AttributeError:
        raise ValueError(f"Model filename constant not found in constants.py: {best_model_const_name}")

    best_path = os.path.join(TRAINED_DL_MODELS_PATH, best_model_filename)

    return best_path


def evaluate_on_test_set(model_type, feature_type, criterion=None):
    print(f"Evaluating {model_type.upper()} model with {feature_type.upper()} features on test set...")

    try:
        model_path = get_model_paths(model_type, feature_type)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        ClassifierClass = get_classifier_class(model_type)
        classifier = ClassifierClass(feature_type=feature_type)

        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        results = classifier.evaluate_model(model_path=model_path, criterion=criterion)

        print("\nTest Set Evaluation Results:")
        print(f"Test Accuracy: {results['test_acc']:.4f}")
        print(f"Test Loss: {results['test_loss']:.4f}")

        print("\nPer-class metrics:")
        cm = results['confusion_matrix']
        for i, instrument in enumerate(DRUM_INSTRUMENTS):
            class_correct = cm[i, i]
            class_total = cm[i, :].sum()
            accuracy = class_correct / class_total if class_total > 0 else 0
            print(f"{instrument}: Accuracy={accuracy:.4f} ({class_correct}/{class_total})")

        return results

    except Exception as e:
        print(f"Error during test set evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_on_audio_files(model_type, feature_type, use_ensemble=False):
    print(f"Evaluating {model_type.upper()} model with {feature_type.upper()} features on audio files...")

    try:
        if use_ensemble and model_type.lower() == 'heterogeneous':
            from src.models.dl_classifiers.classifiers.heterogeneous_cnn_classifier import HeterogeneousCnnEnsemble
            from src.constants import MEL_BEST_CNN_MODEL_NAME, CQT_BEST_CNN_MODEL_NAME

            model_paths = [
                os.path.join(TRAINED_DL_MODELS_PATH, MEL_BEST_CNN_MODEL_NAME),
                os.path.join(TRAINED_DL_MODELS_PATH, CQT_BEST_CNN_MODEL_NAME)
            ]
            model_types = ['mel', 'cqt']
            model = HeterogeneousCnnEnsemble(model_paths, model_types)
        else:
            model_path = get_model_paths(model_type, feature_type)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            ModelClass = get_model_implementation_class(model_type)
            model = ModelClass(num_classes=len(DRUM_INSTRUMENTS))
            model.load_state_dict(torch.load(model_path))
            model.eval()

        correct_count = 0
        total_count = 0

        incorrect_per_label = {label: 0 for label in DRUM_INSTRUMENTS}
        confusion = np.zeros((len(DRUM_INSTRUMENTS), len(DRUM_INSTRUMENTS)), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(DRUM_INSTRUMENTS)}

        for file_name in os.listdir(CUTTED_VAL_SETS_PATH):
            if file_name.endswith(".wav"):
                true_label = extract_true_label(file_name)
                if true_label == "unknown" or true_label not in DRUM_INSTRUMENTS:
                    continue

                audio_path = os.path.join(CUTTED_VAL_SETS_PATH, file_name)
                predicted_label = classify_instrument_dl(audio_path, model, use_ensemble=use_ensemble)

                print(f"File: {file_name}")
                print(f"Predicted: {predicted_label}, True: {true_label}\n")

                total_count += 1
                if predicted_label == true_label:
                    correct_count += 1
                else:
                    incorrect_per_label[true_label] += 1

                true_idx = label_to_idx[true_label]
                pred_idx = label_to_idx[predicted_label]
                confusion[true_idx, pred_idx] += 1

        accuracy = correct_count / total_count if total_count > 0 else 0

        print("\n--- SUMMARY ---")
        print(f"Correct classifications: {correct_count}")
        print(f"Incorrect classifications: {total_count - correct_count}")
        print(f"Total files compared: {total_count}")
        print(f"Accuracy: {accuracy:.4f}")

        print("\n--- ERRORS PER CLASS (TRUE LABEL) ---")
        for label, count in incorrect_per_label.items():
            print(f"{label}: {count}")

        class_metrics = []
        for i, instrument in enumerate(DRUM_INSTRUMENTS):
            class_correct = confusion[i, i]
            class_total = confusion[i, :].sum()
            class_accuracy = class_correct / class_total if class_total > 0 else 0
            class_metrics.append({
                'instrument': instrument,
                'accuracy': class_accuracy,
                'correct': int(class_correct),
                'total': int(class_total),
                'error_count': int(incorrect_per_label[instrument])
            })

        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'incorrect_count': total_count - correct_count,
            'total_count': total_count,
            'confusion_matrix': confusion,
            'class_metrics': class_metrics,
            'error_per_class': incorrect_per_label
        }

    except Exception as e:
        print(f"Error during audio file evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_confusion_matrix(confusion_matrix, labels=None, title=None, save_path=None):
    if labels is None:
        labels = DRUM_INSTRUMENTS

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if title:
        plt.title(title)
    else:
        plt.title('Confusion Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_evaluation_results(results, model_type, feature_type, eval_type):
    import json
    from datetime import datetime
    import numpy as np

    os.makedirs(SIMPLE_EVALUATION_RESULTS_PATH, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{model_type}_{feature_type}_{eval_type}_eval_{timestamp}"

    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj

    results_to_save = convert_numpy_types({k: v for k, v in results.items() if k != 'confusion_matrix'})

    if 'confusion_matrix' in results:
        results_to_save['confusion_matrix'] = convert_numpy_types(results['confusion_matrix'])

    json_path = os.path.join(SIMPLE_EVALUATION_RESULTS_PATH, f"{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    if 'confusion_matrix' in results:
        cm_path = os.path.join(SIMPLE_EVALUATION_RESULTS_PATH, f"{base_filename}_cm.png")
        title = f"{model_type.upper()} - {feature_type.upper()} Confusion Matrix ({eval_type})"
        visualize_confusion_matrix(
            results['confusion_matrix'],
            title=title,
            save_path=cm_path
        )

    print(f"Results saved to: {json_path}")
    if 'confusion_matrix' in results:
        print(f"Confusion matrix saved to: {cm_path}")

    return True


def run_evaluation(args):
    model_type = args.model_type.lower()
    feature_type = args.feature_type.lower()

    print(f"--- Model Evaluation Configuration ---")
    print(f"Model: {model_type.upper()}")
    print(f"Feature Type: {feature_type.upper()}")
    print(f"Evaluation Type: {'Test Set' if args.test_set else 'Audio Files'}")
    print("-------------------------------------")

    use_ensemble = False
    if model_type == 'heterogeneous':
        use_ensemble = True
        print("Using heterogeneous ensemble of models...")

    if args.test_set:
        criterion = torch.nn.CrossEntropyLoss()
        results = evaluate_on_test_set(model_type, feature_type, criterion)

        if results and args.save_results:
            save_evaluation_results(results, model_type, feature_type, "test_set")
    else:
        results = evaluate_on_audio_files(model_type, feature_type, use_ensemble)

        if results and args.save_results:
            save_evaluation_results(results, model_type, feature_type, "audio_files")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained deep learning models on test set or audio files.")

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['cnn', 'lstm', 'transformer', 'cnn_lstm', 'heterogeneous'],
                        help="Type of model to evaluate.")
    parser.add_argument('--feature-type', type=str, required=True,
                        choices=['cqt', 'mel'],
                        help="Type of feature extraction used ('cqt' or 'mel').")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test-set', action='store_true',
                       help="Evaluate on the test set (using the classifier's evaluate_model method).")
    group.add_argument('--audio-files', action='store_true',
                       help="Evaluate on audio files (using the classify_instrument_dl function).")

    parser.add_argument('--save-results', action='store_true',
                        help="Save evaluation results to disk.")

    args = parser.parse_args()

    print("Starting Model Evaluation...")
    results = run_evaluation(args)
    print("\nEvaluation Complete.")
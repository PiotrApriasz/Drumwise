from datetime import datetime
import json

import torch
import os
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from pathlib import Path

from src.constants import DRUM_INSTRUMENTS, TRAINED_DL_MODELS_PATH, CUTTED_VAL_SETS_PATH, EVALUATION_RESULTS_PATH, \
    MEL_BEST_CNN_MODEL_NAME, CQT_BEST_CNN_MODEL_NAME
from src.converters.conversion_utils import extract_true_label, classify_instrument_dl
from src.data.dataset.single_label_dataset_loader import load_dataset_with_splits
from src.evaluation.metrics.single_label_metrics import get_all_single_label_metrics
from src.models.dl_classifiers.classifiers.cnn_mel_cqt_ensemble_classifier import CnnMelCqtEnsemble


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


def perform_test_set_evaluation(model_type, feature_type, criterion=None):
    print(f"\n--- Starting Test Set Evaluation for {model_type.upper()} with {feature_type.upper()} ---")

    try:
        model_path = get_model_paths(model_type, feature_type)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found or path not applicable at: {model_path}")

        ClassifierClass = get_classifier_class(model_type)
        classifier = ClassifierClass(feature_type=feature_type)
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        base_eval_results = classifier.evaluate_model(model_path=model_path, criterion=criterion)
        print("Base Test Set Evaluation (from classifier.evaluate_model) Complete.")
        if 'test_acc' in base_eval_results: print(f"  Accuracy: {base_eval_results['test_acc']:.4f}")
        if 'test_loss' in base_eval_results: print(f"  Loss: {base_eval_results['test_loss']:.4f}")

        print("Calculating detailed single-label metrics for test set...")
        true_labels = base_eval_results.get('labels')
        pred_classes = base_eval_results.get('predictions')
        pred_proba = base_eval_results.get('outputs')

        if true_labels is None or pred_classes is None:
            print("Error: 'labels' or 'predictions' missing from test set results. Cannot calculate detailed metrics.")
            return base_eval_results  # Return what we have

        detailed_metrics = get_all_single_label_metrics(
            y_true=true_labels,
            y_pred_classes=pred_classes,
            y_pred_logits=pred_proba,
            class_names=list(DRUM_INSTRUMENTS),
            num_classes=len(DRUM_INSTRUMENTS)
        )
        print("Detailed metrics calculation for test set complete.")

        combined_test_metrics = {**base_eval_results, **detailed_metrics}
        return combined_test_metrics

    except Exception as e:
        print(f"Error during test set evaluation for {model_type} ({feature_type}): {e}")
        import traceback
        traceback.print_exc()
        return None


def perform_ensemble_test_set_evaluation(criterion):
    """
    Performs test set evaluation for the CnnEnsemble.
    This function is designed to be called from run_all_evaluations.
    Args:
        criterion: The loss function to use for evaluation.
    Returns:
        A dictionary containing evaluation results, or None on error.
    """
    print(f"\n--- Starting Test Set Evaluation for CNN ENSEMBLE ---")
    model_type_display = "_CNN_ENSEMBLE"

    try:
        print("Loading raw test audio data for ensemble...")

        dataset_splits = load_dataset_with_splits()

        if isinstance(dataset_splits, tuple) and len(dataset_splits) == 3:
            _, _, test_data_raw = dataset_splits
        elif isinstance(dataset_splits, dict) and 'test' in dataset_splits:
            test_data_raw = dataset_splits['test']
        else:
            raise ValueError(
                f"Unsupported format from load_dataset_with_splits for raw audio. Expected tuple of 3 or dict with 'test' key. Got: {type(dataset_splits)}")

        if not test_data_raw:
            print("Error: No raw test data loaded for the ensemble. `test_data_raw` is empty.")
            return None
        print(f"Loaded {len(test_data_raw)} raw audio samples for testing the ensemble.")

        # 2. Initialize your ensemble
        print("Initializing CNN Ensemble...")
        model_paths_ensemble = [
            os.path.join(TRAINED_DL_MODELS_PATH, MEL_BEST_CNN_MODEL_NAME),
            os.path.join(TRAINED_DL_MODELS_PATH, CQT_BEST_CNN_MODEL_NAME)
        ]

        for mp in model_paths_ensemble:
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Ensemble model component not found: {mp}")

        ensemble_component_feature_types = ['mel', 'cqt']
        ensemble_model = CnnMelCqtEnsemble(model_paths_ensemble, ensemble_component_feature_types)

        print("Ensemble initialized.")

        print("Evaluating ensemble on the loaded test data...")
        if not hasattr(ensemble_model, 'evaluate_on_test_data'):
            print(f"Error: The {type(ensemble_model).__name__} class does not have an 'evaluate_on_test_data' method.")
            return None


        avg_loss, accuracy, true_labels, pred_classes, raw_outputs = ensemble_model.evaluate_on_test_data(
            test_audio_data=test_data_raw,
            criterion=criterion
        )
        print(f"Ensemble Evaluation Complete. Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

        print("Calculating detailed single-label metrics for ensemble...")
        detailed_metrics = get_all_single_label_metrics(
            y_true=true_labels,
            y_pred_classes=pred_classes,
            y_pred_proba=raw_outputs,
            class_names=list(DRUM_INSTRUMENTS),
            num_classes=len(DRUM_INSTRUMENTS)
        )
        print("Detailed metrics calculation for ensemble complete.")

        results = {
            "model_type_display": model_type_display,  # For clarity in results
            "feature_type_display": "ensemble (mel+cqt)",  # Descriptive
            "test_loss": float(avg_loss) if hasattr(avg_loss, 'item') else float(avg_loss),
            "test_acc": float(accuracy) if hasattr(accuracy, 'item') else float(accuracy),
            "labels": [int(l) for l in true_labels] if true_labels is not None else [],
            "predictions": [int(p) for p in pred_classes] if pred_classes is not None else [],
            "outputs": [out.tolist() if hasattr(out, 'tolist') else out for out in
                        raw_outputs] if raw_outputs is not None else [],
            **detailed_metrics
        }

        if "confusion_matrix_calculated_here" in results and results["confusion_matrix_calculated_here"] is not None:
            cm_value = results["confusion_matrix_calculated_here"]
            results["confusion_matrix"] = cm_value.tolist() if hasattr(cm_value, 'tolist') else cm_value
        elif "confusion_matrix" in results and results[
            "confusion_matrix"] is not None:
            cm_value = results["confusion_matrix"]
            results["confusion_matrix"] = cm_value.tolist() if hasattr(cm_value, 'tolist') else cm_value

        return convert_numpy_types(results)

    except ImportError as e:
        print(f"Error during ensemble evaluation ({model_type_display}): {e} - Check imports for ensemble components.")
        return None
    except AttributeError as e:
        print(
            f"Error during ensemble evaluation ({model_type_display}): {e} - Likely an issue with a class method or attribute (e.g., 'evaluate_on_test_data' in ensemble).")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during ensemble test set evaluation for {model_type_display}: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_on_audio_files(model_type, feature_type):
    print(f"Evaluating {model_type.upper()} model with {feature_type.upper()} features on audio files...")

    use_ensemble = False
    try:
        if model_type.lower() == 'cnn_mel_cqt_ensemble':
            use_ensemble = True

            model_paths_ensemble = [
                os.path.join(TRAINED_DL_MODELS_PATH, MEL_BEST_CNN_MODEL_NAME),
                os.path.join(TRAINED_DL_MODELS_PATH, CQT_BEST_CNN_MODEL_NAME)
            ]

            for mp in model_paths_ensemble:
                if not os.path.exists(mp):
                    raise FileNotFoundError(f"Ensemble model component not found: {mp}")

            ensemble_component_feature_types = ['mel', 'cqt']
            model = CnnMelCqtEnsemble(model_paths_ensemble, ensemble_component_feature_types)
            print("Using cnn mel cqt ensemble for audio file evaluation.")
        else:
            model_path = get_model_paths(model_type, feature_type)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            ModelClass = get_model_implementation_class(model_type)
            model = ModelClass(num_classes=len(DRUM_INSTRUMENTS))
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"Loaded {model_type.upper()} model from {model_path} for audio file evaluation.")

        y_true_audio = []
        y_pred_audio = []

        print(f"Processing audio files from: {CUTTED_VAL_SETS_PATH}")
        audio_file_names = [f for f in os.listdir(CUTTED_VAL_SETS_PATH) if f.endswith(".wav")]
        if not audio_file_names:
            print("No .wav files found in CUTTED_VAL_SETS_PATH. Skipping audio file evaluation.")
            return None

        for file_name in audio_file_names:
            true_label = extract_true_label(file_name)
            if true_label == "unknown" or true_label not in DRUM_INSTRUMENTS:
                continue

            audio_path = os.path.join(CUTTED_VAL_SETS_PATH, file_name)
            use_mel = False
            if feature_type == 'mel':
                use_mel = True
            predicted_label = classify_instrument_dl(
                audio_path,
                model,
                use_mel_model=use_mel,
                use_ensemble=use_ensemble
            )

            y_true_audio.append(true_label)
            y_pred_audio.append(predicted_label)
            print(f"File: {file_name}, True: {true_label}, Predicted: {predicted_label}")

        if not y_true_audio:
            print("No valid audio files processed. Skipping audio file metrics calculation.")
            return None

        unique_true_labels = set(y_true_audio)
        unique_pred_labels = set(y_pred_audio)
        instrument_to_idx = {instrument: idx for idx, instrument in enumerate(DRUM_INSTRUMENTS)}

        # Convert string labels to numeric indices if needed
        if isinstance(y_true_audio[0], str):
            y_true_indices = [instrument_to_idx.get(label, -1) for label in y_true_audio]
            y_pred_indices = [instrument_to_idx.get(label, -1) for label in y_pred_audio]

            # Filter out any invalid labels (-1)
            valid_entries = [(true_idx, pred_idx) for true_idx, pred_idx in zip(y_true_indices, y_pred_indices) if true_idx != -1]
            if not valid_entries:
                print("No valid label pairs after conversion. Cannot calculate metrics.")
                return None

            y_true_filtered, y_pred_filtered = zip(*valid_entries)
        else:
            y_true_filtered = y_true_audio
            y_pred_filtered = y_pred_audio

        audio_eval_results = get_all_single_label_metrics(
            y_true=y_true_filtered,
            y_pred_classes=y_pred_filtered,
            class_names=list(DRUM_INSTRUMENTS),
            num_classes=len(DRUM_INSTRUMENTS)
        )

        print("Audio File Evaluation Complete.")
        if 'accuracy' in audio_eval_results: print(
            f"  Overall Accuracy on Audio Files: {audio_eval_results['accuracy']:.4f}")

        return audio_eval_results

    except Exception as e:
        print(f"Error during audio file evaluation for {model_type} ({feature_type}): {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_confusion_matrix(cm_data, labels=None, title=None, save_path=None):
    if cm_data is None:
        print(f"No confusion matrix data provided for title: {title}. Skipping visualization.")
        return
    if labels is None:
        labels = list(DRUM_INSTRUMENTS)

    plt.figure(figsize=(10, 8))
    sns.heatmap(np.array(cm_data), annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title if title else 'Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()
    else:
        plt.show()



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


def save_combined_results(overall_results_dict, model_type, feature_type):
    os.makedirs(EVALUATION_RESULTS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a dedicated directory for this evaluation run
    experiment_name = f"{model_type}_{feature_type}_{timestamp}"
    results_dir = os.path.join(EVALUATION_RESULTS_PATH, experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    base_filename = f"{model_type}_{feature_type}_all_evals"

    # Extract and save raw prediction data
    raw_predictions_info = {}

    # Process test set evaluation data
    if 'test_set_evaluation' in overall_results_dict and overall_results_dict['test_set_evaluation']:
        test_eval = overall_results_dict['test_set_evaluation']

        # Extract raw prediction data if available
        y_true = test_eval.get('labels')
        y_pred_classes = test_eval.get('predictions')
        y_pred_logits = test_eval.get('outputs')

        if y_true is not None and y_pred_classes is not None:
            # Create paths for raw prediction files
            y_true_path = os.path.join(results_dir, f"{base_filename}_y_true.npy")
            y_pred_classes_path = os.path.join(results_dir, f"{base_filename}_y_pred_classes.npy")

            # Save NumPy arrays
            np.save(y_true_path, np.array(y_true))
            np.save(y_pred_classes_path, np.array(y_pred_classes))

            # Store relative paths in info dictionary
            raw_predictions_info['test_set'] = {
                'y_true': os.path.basename(y_true_path),
                'y_pred_classes': os.path.basename(y_pred_classes_path),
            }

            if y_pred_logits is not None:
                y_pred_logits_path = os.path.join(results_dir, f"{base_filename}_y_pred_logits.npy")
                np.save(y_pred_logits_path, np.array(y_pred_logits))
                raw_predictions_info['test_set']['y_pred_logits'] = os.path.basename(y_pred_logits_path)

    # Add raw predictions info to results
    savable_results = convert_numpy_types(overall_results_dict)
    savable_results['raw_predictions_paths'] = raw_predictions_info
    savable_results['experiment_name'] = experiment_name

    # Save JSON results file
    json_path = os.path.join(results_dir, f"{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(savable_results, f, indent=2)
    print(f"Combined evaluation results saved to: {json_path}")

    # Also save a copy in the main directory for backward compatibility
    main_json_path = os.path.join(EVALUATION_RESULTS_PATH, f"{experiment_name}.json")
    with open(main_json_path, 'w') as f:
        json.dump(savable_results, f, indent=2)
    print(f"Reference copy saved to: {main_json_path}")

    # Save confusion matrix visualizations
    if 'test_set_evaluation' in savable_results and savable_results['test_set_evaluation']:
        test_cm = savable_results['test_set_evaluation'].get('confusion_matrix')
        if test_cm is not None:
            cm_path_test = os.path.join(results_dir, f"{base_filename}_test_set_cm.png")
            visualize_confusion_matrix(test_cm, title=f"{model_type.upper()} Test Set CM", save_path=cm_path_test)
        elif 'classification_report' in savable_results['test_set_evaluation'] and \
             'confusion_matrix_calculated_here' in savable_results['test_set_evaluation']['classification_report']:
            test_cm_alt = savable_results['test_set_evaluation']['classification_report']['confusion_matrix_calculated_here']
            cm_path_test_alt = os.path.join(results_dir, f"{base_filename}_test_set_detailed_cm.png")
            visualize_confusion_matrix(test_cm_alt, title=f"{model_type.upper()} Test Set Detailed CM", save_path=cm_path_test_alt)

    if 'audio_files_evaluation' in savable_results and savable_results['audio_files_evaluation']:
        audio_cm = savable_results['audio_files_evaluation'].get('confusion_matrix')
        if audio_cm is None and 'classification_report' in savable_results['audio_files_evaluation']:
             audio_cm = savable_results['audio_files_evaluation']['classification_report'].get('confusion_matrix_calculated_here')

        if audio_cm is not None:
            cm_path_audio = os.path.join(results_dir, f"{base_filename}_audio_files_cm.png")
            visualize_confusion_matrix(audio_cm, title=f"{model_type.upper()} Audio Files CM", save_path=cm_path_audio)

    print(f"All evaluation data saved to directory: {results_dir}")
    return json_path, results_dir


def run_all_evaluations(args):
    print(f"--- Starting All Evaluations: Model Type '{args.model_type}' ---")

    all_results = {}
    criterion = torch.nn.CrossEntropyLoss()

    current_model_type = args.model_type.lower()
    ensemble_model_name = 'cnn_mel_cqt_ensemble'

    print(f"\n--- Preparing for Test Set Evaluation ---")
    test_results = None
    if current_model_type == ensemble_model_name:
        print(f"Recognized ensemble model type: {current_model_type}. Running ensemble test set evaluation.")
        test_results = perform_ensemble_test_set_evaluation(criterion=criterion)
    else:
        print(f"Recognized single model type: {current_model_type}. Running standard test set evaluation.")
        effective_feature_type = args.feature_type.lower() if args.feature_type else None
        if not effective_feature_type and current_model_type != ensemble_model_name:
            print(
                f"Warning: feature_type not specified for single model {current_model_type}. This might be an issue for perform_test_set_evaluation.")

        test_results = perform_test_set_evaluation(
            model_type=current_model_type,
            feature_type=effective_feature_type,
            criterion=criterion
        )

        if test_results:
            all_results['test_set_evaluation'] = test_results
            print(f"Test Set Evaluation completed for {current_model_type}.")
            if 'test_acc' in test_results: print(f"  Accuracy: {test_results['test_acc']:.4f}")
        else:
            print(f"Test Set Evaluation FAILED or returned no results for {current_model_type}.")

        print(f"\n--- Preparing for Audio Files Evaluation ---")
        audio_eval_results = None
        if current_model_type == ensemble_model_name:
            print(f"Recognized ensemble model type: {current_model_type}. Running ensemble audio files evaluation.")
            audio_eval_results = evaluate_on_audio_files(
                model_type=current_model_type,
                feature_type=None
            )
        else:
            print(f"Recognized single model type: {current_model_type}. Running standard audio files evaluation.")
            effective_feature_type = args.feature_type.lower() if args.feature_type else None
            audio_eval_results = evaluate_on_audio_files(
                model_type=current_model_type,
                feature_type=effective_feature_type
            )

        if audio_eval_results:
            all_results['audio_files_evaluation'] = audio_eval_results
            print(f"Audio Files Evaluation completed for {current_model_type}.")
        else:
            print(f"Audio Files Evaluation FAILED or returned no results for {current_model_type}.")


    if all_results and args.save_results:
        print(f"\n--- Saving Combined Results for {current_model_type} ---")
        save_feature_type_tag = "ensemble" if current_model_type == ensemble_model_name else (
            args.feature_type.lower() if args.feature_type else "unknown_feature")

        json_path, results_dir = save_combined_results(
            overall_results_dict=all_results,
            model_type=current_model_type,
            feature_type=save_feature_type_tag
        )
        print(f"Results saved to: {results_dir}")
    elif not all_results:
        print("\nNo results generated to save.")

    print(f"\n--- All Evaluations Finished for Model Type '{args.model_type}' ---")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluations (Test Set and Audio Files) for DL models.")

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['cnn', 'lstm', 'transformer', 'cnn_lstm', 'cnn_mel_cqt_ens'],
                        help="Type of model to evaluate.")

    parser.add_argument('--feature-type', type=str, required=True,
                        choices=['cqt', 'mel', 'ensemble_meta'],
                        help="Feature type used. For 'ensemble', this might be a general descriptor or ignored if model handles features internally.")
    parser.add_argument('--save-results', action='store_true',
                        help="Save all evaluation results (JSON and CM plots) to disk.")

    args = parser.parse_args()

    effective_feature_type = args.feature_type
    if args.model_type.lower() == 'cnn_mel_cqt_ens' and args.feature_type == 'ensemble_meta':
        effective_feature_type = 'ensemble'

    run_all_evaluations(args)

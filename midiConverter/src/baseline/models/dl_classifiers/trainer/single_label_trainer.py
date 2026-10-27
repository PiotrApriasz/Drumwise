from datetime import datetime
import json

import torch.nn as nn
import os
import argparse
import importlib

from src import constants
from src.constants import TRAINED_DL_MODELS_PATH, EVALUATION_RESULTS_PATH, SEED, EPOCHS, PATIENCE, BATCH_SIZE, \
    NUM_WORKERS, AUGMENT, CNN_LEARNING_RATE, CNN_WEIGHT_DECAY, CNN_LABEL_SMOOTHING, BASELINE_PRE_ONSET_MS, \
    BASELINE_POST_ONSET_MS
from src.baseline.data.dataset.augmentation import AugmentConfig
from src.baseline.data.windowing import WindowConfig
from src.baseline.evaluation.model_evaluator_runner import convert_numpy_types
from src.baseline.models.dl_classifiers.classifiers.checkpoints import model_file_name
from src.baseline.models.dl_classifiers.classifiers.cnn_classifier import build_cnn_optimization
from src.data.features import LogMelConfig
from src.reproducibility import set_seed


def get_classifier_class(model_type: str):
    """Dynamically imports and returns the classifier class based on model_type."""
    model_type = model_type.lower()
    if model_type == 'cnn':
        module_path = 'src.baseline.models.dl_classifiers.classifiers.cnn_classifier'
        class_name = 'DrumCNNClassifier'
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


def get_model_paths(model_type: str, post_onset_ms: int, seed: int) -> tuple[str, str]:
    """Final and best model paths; the file names carry the window length and the seed."""
    model_type_upper = model_type.upper()

    final_model_const_name = f"{model_type_upper}_MODEL_NAME"
    best_model_const_name = f"BEST_{model_type_upper}_MODEL_NAME"

    try:
        final_model_filename = getattr(constants, final_model_const_name)
        best_model_filename = getattr(constants, best_model_const_name)
    except AttributeError as e:
        raise ValueError(f"Error: Model filename constants not found in constants.py for {model_type}. Missing constant: {e}")

    final_path = os.path.join(TRAINED_DL_MODELS_PATH, model_file_name(final_model_filename, post_onset_ms, seed))
    best_path = os.path.join(TRAINED_DL_MODELS_PATH, model_file_name(best_model_filename, post_onset_ms, seed))

    return final_path, best_path


def train_and_evaluate_model(args):
    """
    Orchestrates the training and evaluation of a specified model,
    using defaults from constants and allowing overrides from args.
    """
    MODEL_TYPE = args.model_type.lower()

    print("Loading configuration...")

    seed = args.seed if args.seed is not None else SEED
    epochs = args.epochs if args.epochs is not None else EPOCHS
    patience = args.patience if args.patience is not None else PATIENCE
    batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
    num_workers = args.num_workers if args.num_workers is not None else NUM_WORKERS
    augment = args.augment if args.augment is not None else AUGMENT
    pre_onset_ms = args.pre_onset_ms if args.pre_onset_ms is not None else BASELINE_PRE_ONSET_MS
    post_onset_ms = args.post_onset_ms if args.post_onset_ms is not None else BASELINE_POST_ONSET_MS

    # Model-specific hyperparameters
    if MODEL_TYPE == 'cnn':
        lr = args.lr if args.lr is not None else CNN_LEARNING_RATE
        weight_decay = args.weight_decay if args.weight_decay is not None else CNN_WEIGHT_DECAY
        label_smoothing = args.label_smoothing if args.label_smoothing is not None else CNN_LABEL_SMOOTHING

    set_seed(seed)

    try:
        FINAL_MODEL_PATH, BEST_MODEL_PATH = get_model_paths(MODEL_TYPE, post_onset_ms, seed)
    except ValueError as e:
        print(e)
        return

    features = LogMelConfig()
    window = WindowConfig(sr=features.sr, pre_s=pre_onset_ms / 1000, post_s=post_onset_ms / 1000)

    # --- Print Final Configuration ---
    print("--- Final Configuration ---")
    print(f"Model Type: {MODEL_TYPE.upper()}")
    print(f"Window: {pre_onset_ms} ms before to {post_onset_ms} ms after the onset")
    print(f"Seed: {seed}, Epochs: {epochs}, Patience: {patience}")
    print(f"Batch Size: {batch_size}, Num Workers: {num_workers}")
    print(f"LR: {lr}, Weight Decay: {weight_decay}, Label Smoothing: {label_smoothing}")
    print(f"Online augmentation: {augment}")
    print(f"Best Model Path: {BEST_MODEL_PATH}")
    print(f"Final Model Path: {FINAL_MODEL_PATH}")
    print("-------------------------")

    # --- Initialization ---
    print(f"\nInitializing {MODEL_TYPE.upper()} classifier with log-mel features...")
    try:
        ClassifierClass = get_classifier_class(MODEL_TYPE)
    except (ValueError, ImportError, RuntimeError) as e:
        print(f"Error: {e}")
        return

    classifier = ClassifierClass(
        window=window,
        features=features,
        augment=AugmentConfig() if augment else None,
        batch_size=batch_size,
        num_workers=num_workers
    )
    print(f"Trainable parameters: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")

    # --- Define Optimizer, Criterion, Scheduler ---
    if MODEL_TYPE == 'cnn':
        optimizer, criterion, scheduler = build_cnn_optimization(classifier, lr, weight_decay, label_smoothing, epochs)

    # --- Training ---
    print("\nStarting training...")
    try:
        train_history = classifier.train_model(
            save_path=FINAL_MODEL_PATH,
            best_model_save_path=BEST_MODEL_PATH,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=epochs,
            patience=patience
        )
        print("Training complete.")
        print(f"Best validation macro-F1 achieved: {train_history['best_val_f1']:.4f}")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Evaluation ---
    print(f"\nEvaluating the best model ({BEST_MODEL_PATH}) on the test set...")
    if not os.path.exists(BEST_MODEL_PATH):
         print(f"Error: Best model file not found at {BEST_MODEL_PATH}. Ensure training saved the model.")
         return

    try:
        eval_results = classifier.evaluate_model(
            model_path=BEST_MODEL_PATH,
            criterion=nn.CrossEntropyLoss(),  # unweighted, so test losses are comparable across runs
        )
        print("\nEvaluation complete.")
        print(f"Test Accuracy: {eval_results['test_acc']:.4f}")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

    # --- Saving Training History ---
    if 'train_history' in locals() and train_history:
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            training_history_filename = (f"{MODEL_TYPE}_mel_w{post_onset_ms}ms_s{seed}_training_history_"
                                         f"{timestamp}.json")
            training_history_path = os.path.join(EVALUATION_RESULTS_PATH,
                                                 training_history_filename)

            history_to_save = convert_numpy_types({**train_history, "seed": seed, "augment": augment})

            os.makedirs(EVALUATION_RESULTS_PATH, exist_ok=True)
            with open(training_history_path, 'w') as f:
                json.dump(history_to_save, f, indent=2)
            print(f"Training history saved to: {training_history_path}")
        except Exception as e:
            print(f"\nError saving training history: {e}")
            import traceback
            traceback.print_exc()


# --- Main execution block with argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a single label drum sound classifier using settings from constants.py and allowing overrides.")

    # Required arguments
    parser.add_argument('--model-type', type=str, required=True, choices=['cnn'],
                        help="Type of model to train (e.g., 'cnn').")

    # Optional hyperparameters (default=None means use value from constants.py)
    parser.add_argument('--post-onset-ms', type=int, default=None, help=f"Window length after the onset, the baseline's latency (default from constants: {BASELINE_POST_ONSET_MS}).")
    parser.add_argument('--pre-onset-ms', type=int, default=None, help=f"Window length before the onset (default from constants: {BASELINE_PRE_ONSET_MS}).")
    parser.add_argument('--epochs', type=int, default=None, help=f"Max epochs (default from constants: {EPOCHS}).")
    parser.add_argument('--patience', type=int, default=None, help=f"Patience for early stopping (default from constants: {PATIENCE}).")
    parser.add_argument('--batch-size', type=int, default=None, help=f"Batch size (default from constants: {BATCH_SIZE}).")
    parser.add_argument('--lr', type=float, default=None, help="Learning rate (default from model-specific constants).")
    parser.add_argument('--weight-decay', type=float, default=None, help="Weight decay (default from model-specific constants).")
    parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=None, help=f"Online augmentation (default from constants: {AUGMENT}).")
    parser.add_argument('--label-smoothing', type=float, default=None, help="Label smoothing factor (default from model-specific constants).")
    parser.add_argument('--num-workers', type=int, default=None, help=f"Number of DataLoader workers (default from constants: {NUM_WORKERS}).")
    parser.add_argument('--seed', type=int, default=None, help=f"Random seed (default from constants: {SEED}).")

    args = parser.parse_args()

    print("Starting Single Label Training Script...")
    train_and_evaluate_model(args)
    print("\nScript finished.")

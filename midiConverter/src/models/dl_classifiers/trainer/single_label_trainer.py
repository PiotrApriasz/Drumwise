import datetime
import json

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
import importlib
import sys

from src import constants
from src.constants import *
from src.evaluation.model_evaluator_runner import convert_numpy_types


def get_classifier_class(model_type: str):
    """Dynamically imports and returns the classifier class based on model_type."""
    model_type = model_type.lower()
    if model_type == 'cnn':
        module_path = 'src.models.dl_classifiers.classifiers.cnn_classifier'
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


def get_model_paths(model_type: str, feature_type: str) -> tuple[str, str]:
    """Constructs final and best model paths using constants."""
    model_type_upper = model_type.upper()
    feature_type_upper = feature_type.upper()

    final_model_const_name = f"{feature_type_upper}_{model_type_upper}_MODEL_NAME"
    best_model_const_name = f"{feature_type_upper}_BEST_{model_type_upper}_MODEL_NAME"

    try:
        final_model_filename = getattr(constants, final_model_const_name)
        best_model_filename = getattr(constants, best_model_const_name)
    except AttributeError as e:
        raise ValueError(f"Error: Model filename constants not found in constants.py for {model_type}/{feature_type}. Missing constant: {e}")

    final_path = os.path.join(TRAINED_DL_MODELS_PATH, final_model_filename)
    best_path = os.path.join(TRAINED_DL_MODELS_PATH, best_model_filename)

    return final_path, best_path


def train_and_evaluate_model(args):
    """
    Orchestrates the training and evaluation of a specified model,
    using defaults from constants and allowing overrides from args.
    """
    MODEL_TYPE = args.model_type.lower()
    FEATURE_TYPE = args.feature_type.lower()

    print("Loading configuration...")

    epochs = args.epochs if args.epochs is not None else int(EPOCHS)
    patience = args.patience if args.patience is not None else int(PATIENCE)
    batch_size = args.batch_size if args.batch_size is not None else int(BATCH_SIZE)
    num_workers = args.num_workers if args.num_workers is not None else int(NUM_WORKERS)
    augment_factor = args.augment_factor if args.augment_factor is not None else int(AUGUMENT_FACTOR)

    # Model-specific hyperparameters
    if MODEL_TYPE == 'cnn':
        lr = args.lr if args.lr is not None else float(CNN_LEARNING_RATE)
        weight_decay = args.weight_decay if args.weight_decay is not None else float(CNN_WEIGHT_DECAY)
        label_smoothing = args.label_smoothing if args.label_smoothing is not None else float(CNN_LABEL_SMOOTHING)

    try:
        FINAL_MODEL_PATH, BEST_MODEL_PATH = get_model_paths(MODEL_TYPE, FEATURE_TYPE)
    except ValueError as e:
        print(e)
        return

    # --- Print Final Configuration ---
    print(f"--- Final Configuration ---")
    print(f"Model Type: {MODEL_TYPE.upper()}")
    print(f"Feature Type: {FEATURE_TYPE.upper()}")
    print(f"Epochs: {epochs}, Patience: {patience}")
    print(f"Batch Size: {batch_size}, Num Workers: {num_workers}")
    print(f"LR: {lr}, Weight Decay: {weight_decay}, Label Smoothing: {label_smoothing}")
    print(f"Augmentation Factor: {augment_factor}")
    print(f"Best Model Path: {BEST_MODEL_PATH}")
    print(f"Final Model Path: {FINAL_MODEL_PATH}")
    print(f"-------------------------")

    # --- Initialization ---
    print(f"\nInitializing {MODEL_TYPE.upper()} classifier with {FEATURE_TYPE.upper()} features...")
    try:
        ClassifierClass = get_classifier_class(MODEL_TYPE)
    except (ValueError, ImportError, RuntimeError) as e:
        print(f"Error: {e}")
        return

    classifier = ClassifierClass(
        feature_type=FEATURE_TYPE,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # --- Define Optimizer, Criterion, Scheduler ---
    if MODEL_TYPE == 'cnn':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        scheduler = CosineAnnealingLR(optimizer, T_max=CNN_SCHEDULER_ITERATIONS_NUM, eta_min=CNN_MINIMUM_LEARNING_RATE)

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
            patience=patience,
            augment_factor=augment_factor
        )
        print("Training complete.")
        if 'best_val_acc' in train_history:
            print(f"Best validation accuracy achieved: {train_history['best_val_acc']:.4f}")
        else:
            print("Training finished, best validation accuracy key not found in history.")

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
            criterion=criterion, # Use the same criterion
        )
        print("\nEvaluation complete.")
        if 'test_acc' in eval_results:
            print(f"Test Accuracy: {eval_results['test_acc']:.4f}")
        else:
            print("Evaluation finished, test accuracy key not found in results.")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

    # --- Saving Training History ---
    if 'train_history' in locals() and train_history:
        try:
            training_history_filename = f"{MODEL_TYPE}_{FEATURE_TYPE}_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            training_history_path = os.path.join(EVALUATION_RESULTS_PATH,
                                                 training_history_filename)

            history_to_save = convert_numpy_types(train_history)

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
    parser.add_argument('--feature-type', type=str, required=True, choices=['cqt', 'mel'],
                        help="Type of input feature ('cqt' or 'mel').")

    # Optional hyperparameters (default=None means use value from constants.py)
    parser.add_argument('--epochs', type=int, default=None, help=f"Max epochs (default from constants: {EPOCHS}).")
    parser.add_argument('--patience', type=int, default=None, help=f"Patience for early stopping (default from constants: {PATIENCE}).")
    parser.add_argument('--batch-size', type=int, default=None, help=f"Batch size (default from constants: {BATCH_SIZE}).")
    parser.add_argument('--lr', type=float, default=None, help="Learning rate (default from model-specific constants).")
    parser.add_argument('--weight-decay', type=float, default=None, help="Weight decay (default from model-specific constants).")
    parser.add_argument('--augment-factor', type=int, default=None, help=f"Data augmentation factor (default from constants: {AUGUMENT_FACTOR}).")
    parser.add_argument('--label-smoothing', type=float, default=None, help="Label smoothing factor (default from model-specific constants).")
    parser.add_argument('--num-workers', type=int, default=None, help=f"Number of DataLoader workers (default from constants: {NUM_WORKERS}).")

    args = parser.parse_args()

    print("Starting Single Label Training Script...")
    train_and_evaluate_model(args)
    print("\nScript finished.")
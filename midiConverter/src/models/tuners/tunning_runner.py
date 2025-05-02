import torch
import optuna
import os
import argparse
import sys
import functools
import traceback
import importlib

from src.constants import *

DEFAULT_TUNING_EPOCHS = EPOCHS // 3 if EPOCHS > 5 else 10
DEFAULT_TUNING_PATIENCE = PATIENCE // 2 if PATIENCE > 2 else 3

DEFAULT_LR_LOW = 1e-5
DEFAULT_LR_HIGH = 1e-3
DEFAULT_WD_LOW = 1e-6
DEFAULT_WD_HIGH = 1e-4
DEFAULT_LS_LOW = 0.0
DEFAULT_LS_HIGH = 0.2
DEFAULT_DROPOUT_LOW = 0.2
DEFAULT_DROPOUT_HIGH = 0.5

def load_objective_function(model_type: str):
    """Dynamically loads the objective function for the given model type."""
    model_type = model_type.lower()
    try:
        if model_type == 'cnn':
            module = importlib.import_module(".cnn_objective", package="src.models.tuners")
            return module.objective_cnn
        else:
            raise ValueError(f"Unsupported model type for tuning: {model_type}")
    except ImportError as e:
        print(f"Error importing objective module for type '{model_type}': {e}")
        print("Ensure the corresponding objective file (e.g., cnn_objective.py) exists in src/models/tuners/")
        raise
    except AttributeError as e:
         print(f"Error: Objective function not found in module for type '{model_type}': {e}")
         raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning for DL models.")

    # Required arguments
    parser.add_argument('--model-type', type=str, required=True, choices=['cnn'], # Add 'lstm', etc. later
                        help="Type of model to tune.")
    parser.add_argument('--feature-type', type=str, required=True, choices=['cqt', 'mel'],
                        help="Type of input feature.")

    # Tuning process arguments
    parser.add_argument('--n-trials', type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument('--study-name', type=str, default=None, help="Optuna study name (for resuming/organization).")
    parser.add_argument('--storage', type=str, default=None, help="Optuna storage URL (e.g., 'sqlite:///tuning.db').")

    # Fixed parameters for the tuning run (defaults from constants)
    parser.add_argument('--epochs', type=int, default=DEFAULT_TUNING_EPOCHS, help=f"Epochs per trial (default: {DEFAULT_TUNING_EPOCHS}).")
    parser.add_argument('--patience', type=int, default=DEFAULT_TUNING_PATIENCE, help=f"Patience per trial (default: {DEFAULT_TUNING_PATIENCE}).")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f"Batch size for trials (default: {BATCH_SIZE}).")
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS, help=f"DataLoader workers (default: {NUM_WORKERS}).")
    parser.add_argument('--augment-factor', type=int, default=AUGUMENT_FACTOR, help=f"Augmentation factor (default: {AUGUMENT_FACTOR}).")
    parser.add_argument('--base-save-dir', type=str, default=TRIAL_ARTIFACTS_PATH, help="Base directory to save trial artifacts.")

    # Search space boundaries (optional overrides)
    parser.add_argument('--lr-low', type=float, default=DEFAULT_LR_LOW)
    parser.add_argument('--lr-high', type=float, default=DEFAULT_LR_HIGH)
    parser.add_argument('--wd-low', type=float, default=DEFAULT_WD_LOW)
    parser.add_argument('--wd-high', type=float, default=DEFAULT_WD_HIGH)
    parser.add_argument('--ls-low', type=float, default=DEFAULT_LS_LOW)
    parser.add_argument('--ls-high', type=float, default=DEFAULT_LS_HIGH)
    parser.add_argument('--dropout-low', type=float, default=DEFAULT_DROPOUT_LOW)
    parser.add_argument('--dropout-high', type=float, default=DEFAULT_DROPOUT_HIGH)

    args = parser.parse_args()

    # --- Load Specific Objective Function ---
    try:
        objective_func = load_objective_function(args.model_type)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

    print(f"--- Hyperparameter Tuning Run ---")
    print(f"Model: {args.model_type.upper()}, Features: {args.feature_type.upper()}")
    print(f"Trials: {args.n_trials}, Epochs/Trial: {args.epochs}, Patience/Trial: {args.patience}")
    if args.study_name: print(f"Study Name: {args.study_name}")
    if args.storage: print(f"Storage: {args.storage}")
    print(f"-------------------------------")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True
    )

    # --- Prepare Objective with Fixed Arguments ---
    objective_with_args = functools.partial(
        objective_func,
        feature_type=args.feature_type,
        args=args
    )

    # --- Run Optimization ---
    try:
        study.optimize(objective_with_args, n_trials=args.n_trials, timeout=None)
    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")
        traceback.print_exc()

    # --- Display Results (Readable Format) ---
    print("\n--- Tuning Results ---")
    try:
        print(f"Number of finished trials: {len(study.trials)}")
        best_trial = study.best_trial
        print("\n--- Best Trial Found ---")
        print(f"Value (Best Validation Accuracy): {best_trial.value:.5f}")
        print("\nBest Parameters (for easy copying to constants.py / .env):")

        model_prefix = args.model_type.upper()

        print("\n# --- Suggested constants ---")
        for key, value in best_trial.params.items():
            if key == 'lr':
                const_name = f"{model_prefix}_LEARNING_RATE"
                print(f"{const_name}={value:.8f}")
            elif key == 'weight_decay':
                const_name = f"{model_prefix}_WEIGHT_DECAY"
                print(f"{const_name}={value:.8f}")
            elif key == 'label_smoothing':
                 const_name = f"{model_prefix}_LABEL_SMOOTHING"
                 print(f"{const_name}={value:.4f}")
            elif key == 'dropout_rate':
                 print(f"# Suggested {model_prefix} Model Param: dropout_rate={value:.4f}")
            elif key.startswith('num_filters'):
                 print(f"# Suggested {model_prefix} Model Param: {key}={value}")
            else:
                 print(f"# Suggested {model_prefix} Param: {key.upper()}={value}")
        print("# -------------------------")

    except ValueError:
        print("No trials completed successfully or best trial not found.")
    except Exception as e:
        print(f"Error retrieving study results: {e}")
        traceback.print_exc()

    print("\nTuning script finished.")

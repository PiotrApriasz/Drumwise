import optuna
import os
import traceback

from src.baseline.data.dataset.augmentation import AugmentConfig
from src.baseline.data.windowing import WindowConfig
from src.baseline.models.dl_classifiers.classifiers.cnn_classifier import DrumCNNClassifier, build_cnn_optimization
from src.data.features import LogMelConfig
from src.reproducibility import set_seed


def objective_cnn(trial, args) -> float:
    """
    Optuna objective function specifically for the CNN model.
    """
    print(f"\n--- Starting CNN Trial {trial.number} ---")

    # --- Suggest Hyperparameters for CNN ---
    lr = trial.suggest_float("lr", args.lr_low, args.lr_high, log=True)
    weight_decay = trial.suggest_float("weight_decay", args.wd_low, args.wd_high, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", args.ls_low, args.ls_high)
    base_channels = trial.suggest_categorical("base_channels", [16, 24, 32, 48])
    dropout_rate = trial.suggest_float("dropout_rate", args.dropout_low, args.dropout_high)

    print(f"  CNN Params: lr={lr:.6f}, wd={weight_decay:.6f}, ls={label_smoothing:.3f}, "
          f"base_channels={base_channels}, dropout={dropout_rate:.3f}")

    # --- Initialization ---
    # Every trial starts from the same seed, so trials differ only in their hyperparameters
    set_seed(args.seed)
    features = LogMelConfig()
    classifier = DrumCNNClassifier(
        window=WindowConfig(sr=features.sr, pre_s=args.pre_onset_ms / 1000, post_s=args.post_onset_ms / 1000),
        features=features,
        augment=AugmentConfig() if args.augment else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        base_channels=base_channels,
        dropout_rate=dropout_rate
    )

    # --- Define Optimizer, Criterion, Scheduler (same setup as the trainer) ---
    optimizer, criterion, scheduler = build_cnn_optimization(classifier, lr, weight_decay, label_smoothing, args.epochs)

    # --- Training ---
    trial_save_dir = os.path.join(args.base_save_dir, "tuning_trials", f"cnn_w{args.post_onset_ms}ms_trial_{trial.number}")
    temp_best_path = os.path.join(trial_save_dir, "cnn_best.pth")
    temp_final_path = os.path.join(trial_save_dir, "cnn_final.pth")
    os.makedirs(trial_save_dir, exist_ok=True)

    try:
        print(f"  Starting CNN training for {args.epochs} epochs...")
        train_history = classifier.train_model(
            save_path=temp_final_path,
            best_model_save_path=temp_best_path,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=args.epochs,
            patience=args.patience
        )
        print(f"  CNN Trial {trial.number} finished training.")

        metric = train_history['best_val_f1']
        print(f"  CNN Trial {trial.number} result (Best Val Macro-F1): {metric:.4f}")
        return metric

    except Exception as e:
        print(f"Error during CNN Trial {trial.number}: {e}")
        traceback.print_exc() # Print full traceback for debugging
        raise optuna.exceptions.TrialPruned(f"CNN Trial failed: {e}")

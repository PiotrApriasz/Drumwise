import os
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")


def _path(env_name):
    """Reads a path from .env; relative paths are resolved against the repository root."""
    value = os.getenv(env_name)
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else REPO_ROOT / path)


DRUM_INSTRUMENTS = ["snare", "crash", "floor_tom", "hi-hat", "kick", "rack_tom", "ride"]

INSTRUMENTS_DATA_SET_PATH = _path("TRAINING_PATH")
VAL_SET_PATH = _path("VAL_SET_PATH")
CUTTED_VAL_SETS_PATH = _path("CUTTED_VAL_SETS_PATH")
TEST_TRACKS_PATH = _path("TEST_TRACKS_PATH")
TO_CONVERT_PATH = _path("TO_CONVERT_PATH")

CONVERTED_PATH = _path("CONVERTED_PATH")
TRAINED_DL_MODELS_PATH = _path("TRAINED_DL_MODELS_PATH")
TRIAL_ARTIFACTS_PATH = _path("TRIAL_ARTIFACTS_PATH")

TESTING_ONSETS_PATH = _path("TESTING_ONSETS_PATH")

EVALUATION_RESULTS_PATH = _path("EVALUATION_RESULTS_PATH")

# Model files get the window length appended, e.g. best_cnn_w100ms.pth (see checkpoints.model_file_name).
BEST_CNN_MODEL_NAME = os.getenv("BEST_CNN_MODEL_NAME", "best_cnn.pth")
CNN_MODEL_NAME = os.getenv("CNN_MODEL_NAME", "final_cnn.pth")

# Numeric settings are parsed here, once, so the rest of the code gets ints and floats.
SEED = int(os.getenv("SEED", "42"))

# Onset window of the two-stage baseline: from PRE ms before to POST ms after the onset.
BASELINE_PRE_ONSET_MS = int(os.getenv("BASELINE_PRE_ONSET_MS", "50"))
BASELINE_POST_ONSET_MS = int(os.getenv("BASELINE_POST_ONSET_MS", "100"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))  # windows are cheap; spawned workers cost more than they save
EPOCHS = int(os.getenv("EPOCHS", "60"))
PATIENCE = int(os.getenv("PATIENCE", "10"))
AUGMENT = os.getenv("AUGMENT", "1") == "1"

CNN_LEARNING_RATE = float(os.getenv("CNN_LEARNING_RATE", "0.001"))
CNN_DROPOUT_RATE = float(os.getenv("CNN_DROPOUT_RATE", "0.3"))
CNN_WEIGHT_DECAY = float(os.getenv("CNN_WEIGHT_DECAY", "0.01"))
CNN_LABEL_SMOOTHING = float(os.getenv("CNN_LABEL_SMOOTHING", "0.05"))
CNN_BASE_CHANNELS = int(os.getenv("CNN_BASE_CHANNELS", "32"))
CNN_WARMUP_EPOCHS = int(os.getenv("CNN_WARMUP_EPOCHS", "3"))
CNN_MINIMUM_LEARNING_RATE = float(os.getenv("CNN_MINIMUM_LEARNING_RATE", "1e-6"))

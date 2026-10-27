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

MEL_BEST_CNN_MODEL_NAME = os.getenv("MEL_BEST_CNN_MODEL_NAME")
CQT_BEST_CNN_MODEL_NAME = os.getenv("CQT_BEST_CNN_MODEL_NAME")
MEL_CNN_MODEL_NAME = os.getenv("MEL_CNN_MODEL_NAME")
CQT_CNN_MODEL_NAME = os.getenv("CQT_CNN_MODEL_NAME")

BATCH_SIZE = os.getenv("BATCH_SIZE")
NUM_WORKERS = os.getenv("NUM_WORKERS")
EPOCHS = os.getenv("EPOCHS")
PATIENCE = os.getenv("PATIENCE")
AUGUMENT_FACTOR = os.getenv("AUGUMENT_FACTOR")

CNN_LEARNING_RATE = os.getenv("CNN_LEARNING_RATE")
CNN_DROPOUT_RATE = os.getenv("CNN_DROPOUT_RATE")
CNN_WEIGHT_DECAY = os.getenv("CNN_WEIGHT_DECAY")
CNN_LABEL_SMOOTHING = os.getenv("CNN_LABEL_SMOOTHING")
CNN_SCHEDULER_ITERATIONS_NUM = 10
CNN_MINIMUM_LEARNING_RATE = 1e-6

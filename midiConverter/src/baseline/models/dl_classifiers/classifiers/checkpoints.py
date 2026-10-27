"""Self-describing checkpoints: weights plus the window and feature settings the model was trained with.

The evaluator and the converter read those settings back, so a model is always fed exactly the input it was
trained on (review item D3).
"""
import os

import torch

from src.baseline.data.windowing import WindowConfig
from src.data.features import LogMelConfig


def model_file_name(base_name: str, post_onset_ms: int, seed: int) -> str:
    """``best_cnn.pth`` -> ``best_cnn_w100ms_s42.pth``: one file per window length W and seed."""
    stem, ext = os.path.splitext(base_name)
    return f"{stem}_w{post_onset_ms}ms_s{seed}{ext}"


def save_checkpoint(path: str, model: torch.nn.Module, model_config: dict, window: WindowConfig,
                    features: LogMelConfig) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "window": window.to_dict(),
        "features": features.to_dict(),
    }, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Returns the saved dict with ``window`` and ``features`` turned back into config objects."""
    checkpoint = torch.load(path, map_location=map_location)
    if "state_dict" not in checkpoint:
        raise ValueError(f"{path} is a checkpoint of the old 2 s model (tag v0-single-label); retrain the baseline.")
    checkpoint["window"] = WindowConfig(**checkpoint["window"])
    checkpoint["features"] = LogMelConfig(**checkpoint["features"])
    return checkpoint

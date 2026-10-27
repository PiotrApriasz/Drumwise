import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds every random generator used in training (Python, NumPy, PyTorch CPU and MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """The Apple GPU when available, otherwise the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

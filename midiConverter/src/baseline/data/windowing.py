"""Fixed windows around a drum onset, shared by training, evaluation and the converter (review item D3).

A window runs from ``pre_s`` before the onset to ``post_s`` after it. In real time the past part is already
available, so ``post_s`` is the latency the two-stage baseline pays before it can classify a hit.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    sr: int = 22050
    pre_s: float = 0.05
    post_s: float = 0.10

    @property
    def pre_samples(self) -> int:
        return int(round(self.pre_s * self.sr))

    @property
    def length(self) -> int:
        return self.pre_samples + int(round(self.post_s * self.sr))

    def to_dict(self) -> dict:
        return asdict(self)


def find_onset(y: np.ndarray, threshold_db: float = -30.0) -> int:
    """Index of the first sample within ``threshold_db`` of the signal's peak (onset of a one-shot file)."""
    envelope = np.abs(y)
    peak = envelope.max() if len(envelope) else 0.0
    if peak <= 0:
        return 0
    return int(np.argmax(envelope >= peak * 10 ** (threshold_db / 20)))


def cut_window(y: np.ndarray, onset: int, config: WindowConfig) -> np.ndarray:
    """The window around sample ``onset``, zero padded where it reaches outside ``y``."""
    start = onset - config.pre_samples
    window = np.zeros(config.length, dtype=np.float32)
    src_start, src_end = max(start, 0), min(start + config.length, len(y))
    if src_end > src_start:
        window[src_start - start:src_end - start] = y[src_start:src_end]
    return window


def peak_normalize(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y)) if len(y) else 0.0
    return y / peak if peak > 1e-8 else y

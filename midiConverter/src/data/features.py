"""Causal log-mel features shared by the baseline and the frame-level models (docs/frame-level-plan.md).

Frames are not centred: frame ``k`` covers samples ``[k * hop, k * hop + n_fft)``, so a frame never contains audio
from after its end time. Compression is ``log(1 + magnitude)``; per-band standardization is fitted on training data
and applied inside the model, not here.
"""
from dataclasses import asdict, dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class LogMelConfig:
    sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 220
    n_mels: int = 128
    fmin: float = 30.0
    fmax: float = 11025.0

    def to_dict(self) -> dict:
        return asdict(self)

    def num_frames(self, num_samples: int) -> int:
        return 0 if num_samples < self.n_fft else 1 + (num_samples - self.n_fft) // self.hop_length


def log_mel(y: np.ndarray, config: LogMelConfig = LogMelConfig()) -> np.ndarray:
    """Returns a ``(n_mels, frames)`` float32 array of ``log(1 + mel magnitude)``."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(y, dtype=np.float32), sr=config.sr, n_fft=config.n_fft, hop_length=config.hop_length,
        n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax, power=1.0, center=False)
    return np.log1p(mel).astype(np.float32)

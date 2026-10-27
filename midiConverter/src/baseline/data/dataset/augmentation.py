"""Online augmentation of onset windows: every call draws new parameters, so every epoch sees new variants.

There is no pitch shift: pitch is the main cue between rack and floor tom. There is no gain change either, because
windows are peak-normalized before feature extraction.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt


@dataclass(frozen=True)
class AugmentConfig:
    onset_jitter_s: float = 0.010  # onset detector error at inference
    overlay_prob: float = 0.5  # decay of an earlier hit still ringing (dense playing)
    overlay_gain_db: tuple = (-30.0, -10.0)
    noise_prob: float = 0.5
    noise_snr_db: tuple = (20.0, 40.0)
    eq_prob: float = 0.5  # microphone and room colouring


def jitter_onset(onset: int, sr: int, rng: np.random.Generator, jitter_s: float) -> int:
    max_shift = int(jitter_s * sr)
    return onset + int(rng.integers(-max_shift, max_shift + 1))


def overlay_decay(window: np.ndarray, other_audio: np.ndarray, other_onset: int, sr: int,
                  rng: np.random.Generator, gain_db: tuple) -> np.ndarray:
    """Adds a piece of another hit's decay, 30-500 ms after that hit's onset, at ``gain_db`` relative to full scale."""
    start = other_onset + int(rng.integers(int(0.03 * sr), int(0.5 * sr)))
    tail = np.zeros_like(window)
    piece = other_audio[start:start + len(window)]
    tail[:len(piece)] = piece
    return window + tail * 10 ** (rng.uniform(*gain_db) / 20)


def add_noise(window: np.ndarray, rng: np.random.Generator, snr_db: tuple) -> np.ndarray:
    power = np.mean(window ** 2)
    if power <= 0:
        return window
    noise_power = power / 10 ** (rng.uniform(*snr_db) / 10)
    return window + rng.normal(0.0, np.sqrt(noise_power), len(window)).astype(np.float32)


def random_eq(window: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Either a low-pass between 3 and 10 kHz or a ±6 dB high shelf above 1-5 kHz."""
    if rng.random() < 0.5:
        sos = butter(2, rng.uniform(3000, 10000), btype="lowpass", fs=sr, output="sos")
        return sosfilt(sos, window).astype(np.float32)
    sos = butter(2, rng.uniform(1000, 5000), btype="highpass", fs=sr, output="sos")
    gain = 10 ** (rng.uniform(-6, 6) / 20)
    return (window + (gain - 1) * sosfilt(sos, window)).astype(np.float32)


def augment_window(window: np.ndarray, sr: int, rng: np.random.Generator, config: AugmentConfig,
                   other: tuple = None) -> np.ndarray:
    """Applies the random overlay, EQ and noise steps. ``other`` is ``(audio, onset)`` of a random other hit."""
    if other is not None and rng.random() < config.overlay_prob:
        window = overlay_decay(window, other[0], other[1], sr, rng, config.overlay_gain_db)
    if rng.random() < config.eq_prob:
        window = random_eq(window, sr, rng)
    if rng.random() < config.noise_prob:
        window = add_noise(window, rng, config.noise_snr_db)
    return window.astype(np.float32)

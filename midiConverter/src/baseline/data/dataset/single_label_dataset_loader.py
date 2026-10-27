import os
import re
from dataclasses import dataclass

import librosa
import numpy as np

from src.baseline.data.windowing import find_onset, peak_normalize


@dataclass
class Hit:
    """One labelled drum hit: peak-normalized mono audio around the hit and the onset's sample index in it."""
    audio: np.ndarray
    label: str
    onset: int
    source: str
    name: str


def source_of(file_name: str) -> str:
    """The dataset a training file comes from, read from its name (used for per-source evaluation)."""
    if re.match(r"^(DI|IN|MN|ST)_", file_name):
        return "library"
    if re.match(r"^(RealDrum|WaveDrum|TechnoDrum)", file_name):
        return "idmt"
    if file_name.startswith(("Bass Sample", "Snare Sample")):
        return "samplepack"
    return "other"


def load_hit(file_path: str, label: str, sr: int = 22050, keep_before_s: float = 0.1,
             keep_after_s: float = 1.0) -> Hit:
    """Loads one one-shot file and keeps ``keep_before_s`` before and ``keep_after_s`` after its onset.

    The kept tail is longer than any analysis window, so augmentation can borrow another hit's decay from it.
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    onset = find_onset(audio)
    start = max(onset - int(keep_before_s * sr), 0)
    audio = audio[start:onset + int(keep_after_s * sr)]
    name = os.path.basename(file_path)
    return Hit(peak_normalize(audio).astype(np.float32), label, onset - start, source_of(name), name)


def load_subset(subset_path: str, instruments: list[str], sr: int = 22050) -> list[Hit]:
    hits = []
    for inst in instruments:
        folder_path = os.path.join(subset_path, inst)
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".wav"):
                hits.append(load_hit(os.path.join(folder_path, file_name), inst, sr))
    return hits

import librosa
import numpy as np
import torch

from src.constants import DRUM_INSTRUMENTS
from src.baseline.data.windowing import WindowConfig, cut_window, find_onset, peak_normalize
from src.data.features import LogMelConfig, log_mel


def extract_true_label(file_name: str) -> str:
    name_lower = file_name.lower()
    if "kick" in name_lower:
        return DRUM_INSTRUMENTS[4]
    elif "snare" in name_lower:
        return DRUM_INSTRUMENTS[0]
    elif "hi-hat" in name_lower or "hihat" in name_lower:
        return DRUM_INSTRUMENTS[3]
    elif "crash" in name_lower:
        return DRUM_INSTRUMENTS[1]
    elif "rack_tom" in name_lower:
        return DRUM_INSTRUMENTS[5]
    elif "floor_tom" in name_lower:
        return DRUM_INSTRUMENTS[2]
    elif "ride" in name_lower:
        return DRUM_INSTRUMENTS[6]
    return "unknown"


def detect_bpm(audio, sr):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr,
                                             hop_length=512,
                                             aggregate=np.median)

    tempo_dp, beats_dp = librosa.beat.beat_track(onset_envelope=onset_env,
                                                 sr=sr,
                                                 start_bpm=60,
                                                 tightness=100)

    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    tempo_perc, beats_perc = librosa.beat.beat_track(y=y_percussive,
                                                     sr=sr,
                                                     start_bpm=60,
                                                     tightness=100)

    ac_tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]

    perc_onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
    perc_ac_tempo = librosa.feature.rhythm.tempo(onset_envelope=perc_onset_env, sr=sr)[0]

    tempo_estimates = [tempo_dp, tempo_perc, ac_tempo, perc_ac_tempo]

    tempo_estimates = [t.item() if isinstance(t, np.ndarray) else t for t in tempo_estimates]

    median_tempo = np.median(tempo_estimates)
    closest_idx = np.argmin([abs(t - median_tempo) for t in tempo_estimates])
    estimated_tempo = tempo_estimates[closest_idx]

    estimated_bpm = round(estimated_tempo)

    if estimated_bpm > 170:
        estimated_bpm = estimated_bpm / 2
    elif estimated_bpm < 70:
        estimated_bpm = estimated_bpm * 2

    if estimated_bpm < 60 or estimated_bpm > 200:
        estimated_bpm = 120

    return int(estimated_bpm)


def classify_window(window: np.ndarray, model, features: LogMelConfig) -> str:
    """Classifies one onset window (cut with ``cut_window``), the same input the model was trained on."""
    x = torch.from_numpy(log_mel(peak_normalize(window), features))[None, None]
    with torch.no_grad():
        pred_idx = model(x.to(next(model.parameters()).device)).argmax(dim=1).item()
    return DRUM_INSTRUMENTS[pred_idx]


def classify_hit_file(path, model, window: WindowConfig, features: LogMelConfig) -> str:
    """Classifies a one-shot file: finds its onset and cuts the same window as in training."""
    y, _ = librosa.load(path, sr=window.sr, mono=True)
    y = peak_normalize(y)
    return classify_window(cut_window(y, find_onset(y), window), model, features)

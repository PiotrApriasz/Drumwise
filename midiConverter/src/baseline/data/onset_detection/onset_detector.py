import numpy as np
import librosa
import pretty_midi
from scipy.io import wavfile


def detect_drum_onsets(audio, sr):
    y_harmonic, y_percussive = librosa.effects.hpss(audio)

    perc_onset_frames = librosa.onset.onset_detect(
        y=y_percussive,
        sr=sr,
        hop_length=128,
        delta=0.025,
        wait=1,
        pre_max=2,
        post_max=2,
        pre_avg=2,
        post_avg=2
    )

    y_low = librosa.effects.preemphasis(audio, coef=0.98)
    low_onset_frames = librosa.onset.onset_detect(
        y=y_low,
        sr=sr,
        hop_length=128,
        delta=0.06,
        wait=0
    )

    all_onset_frames = np.unique(np.concatenate([perc_onset_frames, low_onset_frames]))
    all_onset_frames.sort()

    onset_times = librosa.frames_to_time(all_onset_frames, sr=sr, hop_length=128)

    if len(onset_times) > 0:
        filtered_times = [onset_times[0]]
        for t in onset_times[1:]:
            if t - filtered_times[-1] >= 0.02:
                filtered_times.append(t)
        onset_times = np.array(filtered_times)

    first_energy = np.mean(np.abs(audio[:int(0.05 * sr)]))
    if first_energy > 0.05 and (len(onset_times) == 0 or onset_times[0] > 0.1):
        onset_times = np.insert(onset_times, 0, 0)

    return onset_times
import numpy as np
import librosa
import pretty_midi
from scipy.io import wavfile
def extract_drum_segments(audio, sr, onset_times, segment_duration=0.15):
    segments = []
    times = []

    audio = librosa.util.normalize(audio)

    for i, onset_time in enumerate(onset_times):
        start_sample = max(0, int(onset_time * sr) - int(0.005 * sr))

        if i < len(onset_times) - 1:
            next_onset = onset_times[i + 1]
            time_to_next = next_onset - onset_time

            if time_to_next < 0.1:
                duration = min(0.12, time_to_next * 0.85)
            elif time_to_next < 0.25:
                duration = min(0.18, time_to_next * 0.85)
            else:
                duration = min(0.25, time_to_next * 0.85)
        else:
            duration = 0.25

        duration = max(0.06, duration)

        end_sample = min(len(audio), start_sample + int(duration * sr))

        segment = audio[start_sample:end_sample]

        if len(segment) < int(0.03 * sr):
            continue

        fade_in_samples = int(0.002 * sr)
        fade_out_samples = int(0.005 * sr)
        if len(segment) > fade_in_samples + fade_out_samples:
            fade_in = np.linspace(0, 1, fade_in_samples)
            fade_out = np.linspace(1, 0, fade_out_samples)
            segment[:fade_in_samples] *= fade_in
            segment[-fade_out_samples:] *= fade_out

        segments.append(segment)
        times.append(onset_time)

    return segments, times


def filter_closely_spaced_onsets(onset_times, min_distance=0.05):
    if len(onset_times) <= 1:
        return onset_times

    filtered_onsets = [onset_times[0]]

    for i in range(1, len(onset_times)):
        current = onset_times[i]
        previous = filtered_onsets[-1]
        time_diff = current - previous

        if time_diff >= min_distance:
            filtered_onsets.append(current)
        elif time_diff >= 0.02:
            filtered_onsets.append(current)

    return np.array(filtered_onsets)
import numpy as np
import librosa


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

        # A copy, so the fades below cannot change `audio` and the next segment's attack (review item B8)
        segment = audio[start_sample:end_sample].copy()

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

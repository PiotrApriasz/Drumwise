import os
import numpy as np
import librosa
import pretty_midi
from scipy.io import wavfile

from src.baseline.data.onset_detection.onset_detector import detect_drum_onsets
from src.baseline.data.windowing import WindowConfig, cut_window
from src.baseline.models.dl_classifiers.classifiers.checkpoints import model_file_name
from src.baseline.models.dl_classifiers.classifiers.cnn_classifier import load_cnn
from src.constants import TRAINED_DL_MODELS_PATH, TO_CONVERT_PATH, BEST_CNN_MODEL_NAME, TESTING_ONSETS_PATH, \
    CONVERTED_PATH, BASELINE_POST_ONSET_MS, SEED
from src.data.features import LogMelConfig

from src.baseline.converters.conversion_utils import classify_window, detect_bpm


def estimate_velocity(segment, floor_db=-40.0):
    """MIDI velocity 1-127 from the segment's peak level in dB relative to full scale (review item B10).

    The track is peak-normalized before segmentation, so the loudest hit maps to 127 and a hit `floor_db`
    quieter (or less) maps to 1.
    """
    peak = float(np.max(np.abs(segment))) if len(segment) else 0.0
    if peak <= 0:
        return 1
    level = np.clip((20 * np.log10(peak) - floor_db) / -floor_db, 0.0, 1.0)
    return int(round(1 + 126 * level))


def create_midi_file(onset_times, instruments, output_path, bpm=120.0, velocities=None):

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    drum_track = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    instrument_to_note = {
        "kick": 36,
        "snare": 38,
        "hi-hat": 42,
        "crash": 49,
        "ride": 51,
        "floor_tom": 41,
        "rack_tom": 50
    }

    for i, (time, instrument) in enumerate(zip(onset_times, instruments)):
        if instrument in instrument_to_note:
            note_number = instrument_to_note[instrument]
            
            velocity = velocities[i] if velocities is not None else 100
            
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=note_number,
                start=time,
                end=time + 0.1
            )
            drum_track.notes.append(note)

    midi.instruments.append(drum_track)
    midi.write(output_path)
    
    return None


def convert_audio_to_midi(audio_path, output_midi_path, model, window: WindowConfig, features: LogMelConfig,
                          save_onsets=False):
    """Onset detection, then one window per onset (the same cut as in training) classified by the model."""
    audio, sr = librosa.load(audio_path, sr=window.sr, mono=True)
    audio = librosa.util.normalize(audio)

    bpm = detect_bpm(audio, sr)

    onset_times = detect_drum_onsets(audio, sr)
    windows = [cut_window(audio, int(round(t * sr)), window) for t in onset_times]

    # The velocity uses the part after the onset, at the level of the normalized track.
    velocities = [estimate_velocity(w[window.pre_samples:]) for w in windows]
    instruments = [classify_window(w, model, features) for w in windows]

    if save_onsets:
        os.makedirs(TESTING_ONSETS_PATH, exist_ok=True)
        for i, (w, onset_time, instrument) in enumerate(zip(windows, onset_times, instruments)):
            onset_file = os.path.join(TESTING_ONSETS_PATH, f"onset_{i:03d}_{onset_time:.3f}s_{instrument}.wav")
            wavfile.write(onset_file, sr, (w * 32767).astype(np.int16))

        print(f"Saved {len(windows)} onset files to {TESTING_ONSETS_PATH}/")

    create_midi_file(onset_times, instruments, output_midi_path, bpm=bpm, velocities=velocities)

    result = {
        "num_hits": len(onset_times),
        "detected_instruments": set(instruments),
        "output_path": output_midi_path,
        "detected_bpm": bpm,
        "onset_files_dir": TESTING_ONSETS_PATH if save_onsets else None
    }

    return result


if __name__ == "__main__":

    audio_to_convert = "TestTrack7"
    audio_to_convert_path = os.path.join(TO_CONVERT_PATH, f"{audio_to_convert}.wav")
    converted_audio_output_dir = os.path.join(CONVERTED_PATH, f"{audio_to_convert}.mid")

    model_path = os.path.join(TRAINED_DL_MODELS_PATH, model_file_name(BEST_CNN_MODEL_NAME, BASELINE_POST_ONSET_MS, SEED))
    model, window, features = load_cnn(model_path)

    result = convert_audio_to_midi(
        audio_path=audio_to_convert_path,
        output_midi_path=converted_audio_output_dir,
        model=model,
        window=window,
        features=features,
        save_onsets=True
    )

    print("Conversion complete!")
    print(f"Detected {result['num_hits']} drum hits")
    print(f"Instruments found: {', '.join(result['detected_instruments'])}")
    print(f"Detected BPM: {result['detected_bpm']}")
    print(f"MIDI file saved to: {result['output_path']}")
    if result['onset_files_dir']:
        print(f"Onset audio files saved to: {result['onset_files_dir']}/")

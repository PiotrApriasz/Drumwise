import os
import numpy as np
import librosa
import pretty_midi
from scipy.io import wavfile
import torch

from src.baseline.data.onset_detection.onset_detector import detect_drum_onsets
from src.baseline.data.onset_detection.onset_processor import extract_drum_segments
from src.baseline.models.dl_classifiers.classifiers.cnn_classifier import DrumCNN
from src.constants import DRUM_INSTRUMENTS, TRAINED_DL_MODELS_PATH, TO_CONVERT_PATH, MEL_BEST_CNN_MODEL_NAME, \
    CQT_BEST_CNN_MODEL_NAME, TESTING_ONSETS_PATH, CONVERTED_PATH

from src.baseline.converters.conversion_utils import classify_audio_dl, detect_bpm


def classify_drum_segments(segments, sr, model, use_mel_model=False):
    """Classifies segments in memory; no temporary WAV files (review item B12)."""
    return [classify_audio_dl(librosa.util.normalize(segment), model, sr=sr, use_mel_model=use_mel_model)
            for segment in segments]


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


def convert_audio_to_midi(audio_path, output_midi_path, model, sr=22050, use_mel_model=False, save_onsets=False):

    audio, sr = librosa.load(audio_path, sr=sr, mono=True)

    bpm = detect_bpm(audio, sr)

    onset_times = detect_drum_onsets(audio, sr)

    segments, onset_times = extract_drum_segments(audio, sr, onset_times)

    velocities = [estimate_velocity(segment) for segment in segments]

    instruments = classify_drum_segments(segments, sr, model, use_mel_model=use_mel_model)

    if save_onsets:
        os.makedirs(TESTING_ONSETS_PATH, exist_ok=True)
        for i, (segment, onset_time, instrument) in enumerate(zip(segments, onset_times, instruments)):
            onset_file = os.path.join(TESTING_ONSETS_PATH, f"onset_{i:03d}_{onset_time:.3f}s_{instrument}.wav")
            wavfile.write(onset_file, sr, (segment * 32767).astype(np.int16))
        
        print(f"Saved {len(segments)} onset files to {TESTING_ONSETS_PATH}/")


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

    use_mel_model = False

    model_name = MEL_BEST_CNN_MODEL_NAME if use_mel_model else CQT_BEST_CNN_MODEL_NAME
    model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
    model.load_state_dict(torch.load(os.path.join(TRAINED_DL_MODELS_PATH, model_name), map_location="cpu"))
    model.eval()

    result = convert_audio_to_midi(
        audio_path=audio_to_convert_path,
        output_midi_path=converted_audio_output_dir,
        model=model,
        use_mel_model=use_mel_model,
        save_onsets=True
    )

    print("Conversion complete!")
    print(f"Detected {result['num_hits']} drum hits")
    print(f"Instruments found: {', '.join(result['detected_instruments'])}")
    print(f"Detected BPM: {result['detected_bpm']}")
    print(f"MIDI file saved to: {result['output_path']}")
    if result['onset_files_dir']:
        print(f"Onset audio files saved to: {result['onset_files_dir']}/")

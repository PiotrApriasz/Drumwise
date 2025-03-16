import os
import numpy as np
import librosa
import pretty_midi
import tempfile
from scipy.io import wavfile
import torch
from src.models.dl_classifiers.classifiers.cnn_classifier import DrumCNN
from src.models.dl_classifiers.classifiers.heterogeneous_cnn_classifier import HeterogeneousEnsemble
from src.constants import DRUM_INSTRUMENTS, TRAINED_DL_MODELS_PATH, TO_CONVERT_PATH, MEL_BEST_CNN_MODEL_NAME, \
    CQT_BEST_CNN_MODEL_NAME, TESTING_ONSETS_PATH, CONVERTED_PATH

from src.converters.conversion_utils import set_length, classify_instrument_dl_mel



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
    
    first_energy = np.mean(np.abs(audio[:int(0.05*sr)]))
    if first_energy > 0.05 and (len(onset_times) == 0 or onset_times[0] > 0.1):
        onset_times = np.insert(onset_times, 0, 0)
    
    return onset_times


def extract_drum_segments(audio, sr, onset_times, segment_duration=0.15):
    segments = []
    times = []
    
    audio = librosa.util.normalize(audio)
    
    for i, onset_time in enumerate(onset_times):
        start_sample = max(0, int(onset_time * sr) - int(0.005 * sr))
        
        if i < len(onset_times) - 1:
            next_onset = onset_times[i+1]
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


def classify_drum_segments(segments, sr, model, use_ensemble=False, use_mel_model=False):
    instruments = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, segment in enumerate(segments):
            temp_file = os.path.join(temp_dir, f"segment_{i}.wav")
            normalized_segment = librosa.util.normalize(segment)
            wavfile.write(temp_file, sr, (normalized_segment * 32767).astype(np.int16))
            instrument = classify_instrument_dl_mel(temp_file, model, use_ensemble=use_ensemble, use_mel_model=use_mel_model)
            instruments.append(instrument)

    return instruments


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


def convert_audio_to_midi(audio_path, output_midi_path, model, sr=22050, use_mel_model=False,  use_ensemble=False, save_onsets=False):

    audio, sr = librosa.load(audio_path, sr=sr, mono=True)

    bpm = detect_bpm(audio, sr)

    onset_times = detect_drum_onsets(audio, sr)

    segments, onset_times = extract_drum_segments(audio, sr, onset_times)

    velocities = []
    for segment in segments:
        energy = np.sqrt(np.mean(segment**2))
        velocity = min(127, max(40, int(energy * 100 * 127)))
        velocities.append(velocity)

    instruments = classify_drum_segments(segments, sr, model, use_ensemble=use_ensemble)

    if save_onsets:

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

    use_ensemble = True
    use_mel_model = False

    if use_ensemble:
        model_paths = [
            os.path.join(TRAINED_DL_MODELS_PATH, MEL_BEST_CNN_MODEL_NAME),
            os.path.join(TRAINED_DL_MODELS_PATH, CQT_BEST_CNN_MODEL_NAME)
        ]
        model_types = ['mel', 'cqt']
        model = HeterogeneousEnsemble(model_paths, model_types)
    elif use_mel_model:
        model_path = os.path.join(TRAINED_DL_MODELS_PATH, MEL_BEST_CNN_MODEL_NAME)
        model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        model_path = os.path.join(TRAINED_DL_MODELS_PATH, CQT_BEST_CNN_MODEL_NAME)
        model = DrumCNN(num_classes=len(DRUM_INSTRUMENTS))
        model.load_state_dict(torch.load(model_path))
        model.eval()

    
    result = convert_audio_to_midi(
        audio_path=audio_to_convert_path,
        output_midi_path=converted_audio_output_dir,
        model=model,
        use_mel_model=use_mel_model,
        use_ensemble=use_ensemble,
        save_onsets=True
    )

    print(f"Conversion complete!")
    print(f"Detected {result['num_hits']} drum hits")
    print(f"Instruments found: {', '.join(result['detected_instruments'])}")
    print(f"Detected BPM: {result['detected_bpm']}")
    print(f"MIDI file saved to: {result['output_path']}")
    if result['onset_files_dir']:
        print(f"Onset audio files saved to: {result['onset_files_dir']}/")

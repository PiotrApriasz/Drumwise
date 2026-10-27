"""Tests for the onset-window baseline (docs/frame-level-plan.md, stage 1b): windows, features, augmentation,
model, checkpoints and the converter all use the same input."""
import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

from src.baseline.converters.conversion_utils import classify_hit_file, classify_window
from src.baseline.converters.dl_audio_to_midi_converter import convert_audio_to_midi
from src.baseline.data.dataset.augmentation import AugmentConfig, augment_window, jitter_onset
from src.baseline.data.dataset.single_label_dataset_loader import Hit, load_hit
from src.baseline.data.windowing import WindowConfig, cut_window, find_onset
from src.baseline.models.dl_classifiers.classifiers.base_classifier import WindowDataset, band_statistics, label_map
from src.baseline.models.dl_classifiers.classifiers.checkpoints import load_checkpoint, save_checkpoint
from src.baseline.models.dl_classifiers.classifiers.cnn_classifier import DrumCNN, load_cnn
from src.constants import DRUM_INSTRUMENTS
from src.data.features import LogMelConfig, log_mel

SR = 22050
WINDOW = WindowConfig(sr=SR, pre_s=0.05, post_s=0.10)
FEATURES = LogMelConfig()


def _hit_audio(onset=1000, length=SR, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(length, dtype=np.float32)
    decay = np.exp(-np.arange(length - onset) / (0.05 * SR))
    y[onset:] = (rng.normal(0, 1, length - onset) * decay).astype(np.float32)
    return y / np.abs(y).max()


def _hits(n=8):
    return [Hit(_hit_audio(seed=i), DRUM_INSTRUMENTS[i % 7], 1000, "library", f"h{i}.wav") for i in range(n)]


def test_find_onset_finds_the_attack():
    assert abs(find_onset(_hit_audio(onset=1000)) - 1000) <= 2
    assert find_onset(np.zeros(100)) == 0


def test_cut_window_aligns_the_onset_and_pads_outside_the_signal():
    y = _hit_audio(onset=500)
    window = cut_window(y, 500, WINDOW)
    assert len(window) == WINDOW.length
    assert window[WINDOW.pre_samples] == y[500]
    # the window starts before the signal, so its start is zero padding
    assert np.all(window[:WINDOW.pre_samples - 500] == 0)


def test_log_mel_is_causal():
    y = _hit_audio(length=SR // 2)
    frames = log_mel(y, FEATURES)
    assert frames.shape == (FEATURES.n_mels, FEATURES.num_frames(len(y)))
    changed = y.copy()
    cut = FEATURES.n_fft + 5 * FEATURES.hop_length  # end of frame 5
    changed[cut:] = 0
    np.testing.assert_allclose(log_mel(changed, FEATURES)[:, :6], frames[:, :6], atol=1e-6)


def test_augmentation_keeps_the_window_shape_and_bounds_the_jitter():
    rng = np.random.default_rng(0)
    config = AugmentConfig()
    for _ in range(50):
        assert abs(jitter_onset(1000, SR, rng, config.onset_jitter_s) - 1000) <= int(0.010 * SR)
        window = augment_window(cut_window(_hit_audio(), 1000, WINDOW), SR, rng, config, other=(_hit_audio(seed=1), 1000))
        assert window.shape == (WINDOW.length,) and window.dtype == np.float32 and np.isfinite(window).all()


@pytest.mark.parametrize("augment", [None, AugmentConfig()])
def test_window_dataset_items_have_the_feature_shape(augment):
    dataset = WindowDataset(_hits(), label_map, WINDOW, FEATURES, augment)
    x, y = dataset[3]
    assert x.shape == (1, FEATURES.n_mels, FEATURES.num_frames(WINDOW.length))
    assert y.item() == label_map[DRUM_INSTRUMENTS[3]]


def test_band_statistics_have_one_value_per_band():
    mean, std = band_statistics(WindowDataset(_hits(), label_map, WINDOW, FEATURES))
    assert mean.shape == std.shape == (FEATURES.n_mels,)
    assert (std > 0).all()


@pytest.mark.parametrize("post_s", [0.03, 0.10, 0.20])
def test_cnn_works_for_any_window_length_and_is_compact(post_s):
    window = WindowConfig(sr=SR, pre_s=0.05, post_s=post_s)
    model = DrumCNN().eval()
    x = torch.randn(2, 1, FEATURES.n_mels, FEATURES.num_frames(window.length))
    assert model(x).shape == (2, len(DRUM_INSTRUMENTS))
    assert sum(p.numel() for p in model.parameters()) < 500_000


def test_checkpoint_round_trip_restores_weights_and_input_settings(tmp_path):
    model = DrumCNN().eval()
    model.input_norm.set_stats(torch.full((128,), 0.5), torch.full((128,), 2.0))
    path = tmp_path / "best_cnn_w100ms_s42.pth"
    save_checkpoint(str(path), model, {"num_classes": 7, "n_mels": 128, "base_channels": 32, "dropout_rate": 0.3},
                    WINDOW, FEATURES)
    loaded, window, features = load_cnn(str(path))
    assert window == WINDOW and features == FEATURES
    x = torch.randn(1, 1, 128, 11)
    torch.testing.assert_close(loaded(x), model(x))


def test_old_checkpoints_are_rejected_with_a_clear_message(tmp_path):
    path = tmp_path / "old.pth"
    torch.save(DrumCNN().state_dict(), path)
    with pytest.raises(ValueError, match="retrain"):
        load_checkpoint(str(path))


def test_classification_of_a_window_and_of_a_file(tmp_path):
    model = DrumCNN().eval()
    assert classify_window(cut_window(_hit_audio(), 1000, WINDOW), model, FEATURES) in DRUM_INSTRUMENTS
    path = tmp_path / "kick1.wav"
    sf.write(path, _hit_audio(), SR)
    assert classify_hit_file(str(path), model, WINDOW, FEATURES) in DRUM_INSTRUMENTS
    hit = load_hit(str(path), "kick", SR)
    assert abs(hit.onset - 1000) <= 2  # the loader keeps 100 ms before the onset; here the file starts earlier


def test_converter_writes_one_note_per_detected_onset(tmp_path):
    track = np.zeros(3 * SR, dtype=np.float32)
    for start in (int(0.5 * SR), int(1.5 * SR), int(2.2 * SR)):
        track[start:start + SR // 2] += _hit_audio(onset=0, length=SR // 2, seed=start)
    audio_path, midi_path = tmp_path / "track.wav", tmp_path / "track.mid"
    sf.write(audio_path, track, SR)
    result = convert_audio_to_midi(str(audio_path), str(midi_path), DrumCNN().eval(), WINDOW, FEATURES)
    notes = pretty_midi.PrettyMIDI(str(midi_path)).instruments[0].notes
    assert len(notes) == result["num_hits"] >= 3

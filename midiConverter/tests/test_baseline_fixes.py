"""Smoke tests for the baseline fixes listed in docs/frame-level-plan.md (Known problems)."""
import importlib
import random

import numpy as np
import pytest
import torch

from src import constants
from src.baseline.converters.conversion_utils import classify_audio_dl
from src.baseline.converters.dl_audio_to_midi_converter import estimate_velocity
from src.baseline.data.dataset.dataset_augumentation import time_shift
from src.baseline.data.onset_detection.onset_processor import extract_drum_segments
from src.baseline.evaluation.comparison.calculation_engine import perform_mcnemar_test
from src.baseline.models.dl_classifiers.classifiers.base_classifier import EarlyStopping, compute_class_weights
from src.baseline.models.dl_classifiers.classifiers.cnn_classifier import DrumCNN
from src.reproducibility import set_seed

SR = 22050


def test_constants_are_parsed_to_numbers():
    assert isinstance(constants.EPOCHS, int)
    assert isinstance(constants.CNN_LEARNING_RATE, float)


@pytest.mark.parametrize("module", [
    "src.baseline.models.tuners.tunning_runner",  # B3: crashed on import
    "src.baseline.evaluation.comparison.plotting_engine",  # imported a non-existent `sns` module
])
def test_modules_import(module):
    importlib.import_module(module)


def test_time_shift_never_cuts_the_attack():  # B7
    audio = np.zeros(SR, dtype=np.float32)
    audio[0] = 1.0
    for _ in range(50):
        assert time_shift(audio, SR, max_shift_s=0.1).max() == 1.0


def test_extract_segments_leaves_the_input_untouched():  # B8
    audio = np.random.default_rng(0).uniform(-1, 1, SR).astype(np.float32)
    audio[0] = 1.0  # already peak-normalized, so the in-function normalization is a no-op
    original = audio.copy()
    segments, times = extract_drum_segments(audio, SR, np.array([0.1, 0.16, 0.5]))
    assert len(segments) == len(times) == 3
    np.testing.assert_array_equal(audio, original)


def test_velocity_is_in_midi_range_and_follows_loudness():  # B10
    loud, quiet = np.full(100, 1.0), np.full(100, 0.05)
    assert estimate_velocity(loud) == 127
    assert 1 <= estimate_velocity(quiet) < estimate_velocity(loud)
    assert estimate_velocity(np.zeros(100)) == 1


def test_classify_audio_dl_returns_an_instrument():  # B12
    model = DrumCNN(num_classes=len(constants.DRUM_INSTRUMENTS)).eval()
    hit = np.random.default_rng(0).normal(0, 0.1, int(0.15 * SR)).astype(np.float32)
    assert classify_audio_dl(hit, model, sr=SR) in constants.DRUM_INSTRUMENTS


def test_early_stopping_resets_on_improvement():  # B5
    stopping = EarlyStopping(patience=2)
    assert stopping.step(0.5) and stopping.epochs_no_improve == 0
    assert not stopping.step(0.4) and stopping.epochs_no_improve == 1
    assert stopping.step(0.6) and stopping.epochs_no_improve == 0
    stopping.step(0.6)
    stopping.step(0.1)
    assert stopping.should_stop


def test_class_weights_are_inverse_frequency():  # D5
    weights = compute_class_weights([0, 0, 0, 1], num_classes=3)
    assert weights[1].item() == pytest.approx(3 * weights[0].item())
    assert weights[2] == 0
    assert (weights[0] * 3).item() == pytest.approx((weights[1] * 1).item())  # equal total weight per class


def test_mcnemar_gives_p_values():  # B6
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    results = {
        "a": {"labels": labels, "predictions": [0, 1, 0, 1, 0, 1, 0, 1]},
        "b": {"labels": labels, "predictions": [1, 0, 1, 0, 0, 1, 0, 1]},
        "c": {"labels": labels[:4], "predictions": [0, 1, 0, 1]},
    }
    p = perform_mcnemar_test(results)
    assert 0 < p.loc["a", "b"] <= 1
    assert np.isnan(p.loc["a", "c"])  # different test sets are not compared


def test_set_seed_is_reproducible():  # D6
    set_seed(7)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(7)
    assert first == (random.random(), np.random.rand(), torch.rand(1).item())

# Plan: moving midiConverter to frame-level multi-label transcription

**Status: Proposed**, except the items under "Decided" below. It refines the "Models", "Data" and "Latency and evaluation" sections of [thesis-plan.md](thesis-plan.md).

## Why change

The review of 2026-10-27 found that the current single-label pipeline cannot support the thesis claims:

- **Data leakage.** Train, val and test are split by file, not by recording, and contain 114 groups of byte-identical files. All 140 files of `cuttedValSets` are also present in the training folders (112 in `test`, 112 in `val`, 28 in `train`). The reported 0.96–1.0 test accuracy is not trustworthy.
- **Train/inference mismatch.** The model is trained on 2 s one-shots with full decay. The converter feeds it 0.06–0.25 s faded segments padded with silence. No saved metric measures what the converter does.
- **One label per onset.** Simultaneous hits such as kick + hi-hat cannot be represented. The `TestTrack*.mid` files contain no simultaneous hits, so they do not show this.
- **Not causal.** HPSS, normalization, BPM and peak picking use the whole file, and the classifier needs a 2 s window.

Automatic drum transcription (ADT) usually solves this as **frame-level multi-label activation detection**: one sigmoid output per instrument per ~10 ms frame, followed by peak picking. This formulation is multi-label and can be made causal, and its latency is set by how many frames the model may look ahead. It is evaluated with per-class onset F1 (`mir_eval`). This plan adopts that formulation.

## Task definition

- **Input:** mono audio, 22050 Hz (enough for hi-hat energy up to ~11 kHz; 44.1 kHz is an option if cymbals suffer).
- **Features:** log-mel, `n_fft=1024`, `hop=220` (~10 ms, ~100 fps), 128 bands, `fmin=30`, `fmax=11025`, `log(1 + x)` compression. Frames are **not centered** (`center=False`, window ends at the frame time), so a frame never contains future audio.
- **Output:** a `(T, C)` matrix of per-frame onset probabilities.
- **Classes (decided 2026-10-27):** a fixed mapping from General MIDI pitches to classes, in one place (`labels.py`). Changing the vocabulary is a config change, not new code.
  - **Main results: 3 classes**, kick, snare and hi-hat (closed and open merged). This is the standard ADT setup, so the results can be compared with the literature and with IDMT-SMT-Drums.
  - **Simultaneous hits are not extra classes.** With multi-label output, hi-hat + snare and hi-hat + kick are two classes active in the same frame. They are evaluated separately (see Evaluation).
  - **First extension: open hi-hat** as a 4th class (GM 46 separate from 42/44). E-GMD and ENST annotate it.
  - Later and optional: toms and cymbals (5 classes), then the current 7 `DRUM_INSTRUMENTS` on own data.
- **Targets:** 1 at the onset frame and 0.5 at the two neighbouring frames (target widening). This tolerates ±10 ms annotation jitter.
- **Loss:** BCE with a per-class `pos_weight`, because onsets cover only 1–3 % of frames.
- **Decoding (peak picking):**
  - A frame `t` is an onset of class `c` when all of these hold:
    - `p[t, c] >= thr_c`
    - `p[t, c]` is the maximum of the past `w` frames
    - `p[t, c] > p[t+1, c]`, which needs one frame of lookahead
  - Same-class onsets closer than 30 ms are merged.
  - Thresholds are tuned per class on the validation set.
- **Latency budget:**
  - The algorithmic latency is one hop plus the lookahead frames of the model plus the one frame of the peak picker. For lookahead 0–2 frames this is about 20–40 ms.
  - Compute time per frame is measured separately.
  - Both fit the "classification may arrive 20–50 ms later" assumption.

## Data

| Dataset | Role | Notes |
|---|---|---|
| **E-GMD** (Callender et al. 2020) | Main training set | 444 h, 43 kits, audio aligned with MIDI including velocity. It is too large for local disk (~92 GB free) and for CPU training. Start with a **subset of about 20–50 h covering all kits**, using the official train/validation/test split, which is by session. |
| **IDMT-SMT-Drums** | External test (3 classes) | Real, techno and sampled kits with onset annotations. Only extracted "first hits" exist locally today. Download the full set with annotations. |
| **ENST-Drums** | External test (up to 7 classes, acoustic, multi-mic) | Three drummers, three kits. Split by drummer. Distributed on request for research. |
| **MDB Drums** | Optional external test | Drums in mixed music. Only if time allows. |
| **Own acoustic recordings** | Target-domain test, later fine-tuning | See below. Never mixed into training of the models that are reported on the own test set without a held-out split by session. |

### Own recordings

- **Single hits** (the 20 existing `valSets` takes plus new ones): already labelled. Use them in two ways:
  1. Synthetic multi-label mixes, built by overlaying hits at random onsets and gains. These are training data for domain adaptation.
  2. A clean per-class sanity test.
- **Grooves to a metronome**, played from a MIDI score:
  - Annotation: onset detection, then manual correction in Sonic Visualiser, with the score giving the classes.
  - These are the main target-domain test set.
  - Re-record `TestTrack*` so that they include simultaneous hits (kick + hi-hat, crash + kick).
- **Split by session** (recording day or take), never by file.

### Hygiene rules

- Every dataset loader returns `(audio_path, events, split)`, where `events` is a list of `(time_s, class)`. Splits come from the dataset (official split, drummer or session), never from a random file shuffle.
- A `check_leakage.py` script hashes audio across splits and fails if any file appears twice. It runs before every training.
- Features are cached to `.npy` per track and keyed by a hash of the feature config, so training does not recompute spectrograms.
- Augmentation runs online, in the dataset, with a seed:
  - gain
  - EQ / low-pass
  - additive noise
  - room impulse responses, which narrow the e-drum → acoustic gap
  - small time stretch

  No augmentation may remove the attack.

## Models to compare

All models share one interface: input `(B, T, F)`, output `(B, T, C)` logits, plus a `lookahead` attribute in frames and a `step(frame, state)` method for streaming (stage 6). They train on random excerpts of 2–4 s.

1. **Two-stage baseline (current code, repaired).** Onset detector + segment classifier. Retrain it on short post-onset windows (e.g. 100 ms), the same length as at inference, and evaluate it with the same transcription harness. It shows the cost of single-label decoding on real grooves, so it is a result, not a dead end.
2. **Causal CNN:** a frame classifier on a past-only context window (e.g. 25 frames back + `lookahead` frames). The current `DrumCNN` blocks (conv + BN + SE) can be reused with causal padding along time.
3. **CRNN:** causal conv front-end + unidirectional GRU. This is the classic ADT architecture (Vogl et al. 2017), here without the backward direction.
4. **TCN:** dilated causal 1-D convolutions over time.
5. **Causal Transformer:** conv front-end + Transformer encoder with a causal attention mask and a limited context.

Each model is trained with **3 seeds**. Lookahead is a controlled variable (0, 1, 2 frames), which gives an F1-versus-latency curve.

## Evaluation

The evaluation harness is built **before** the new models. It replaces accuracy as the main metric.

- **Transcription F1 per class:** `mir_eval.onset.f_measure` with a 50 ms window (standard), plus 20 ms to report timing precision. Report micro-F1 (TP/FP/FN summed over tracks) and the mean per-track F1, separately for each test set (E-GMD test, IDMT, ENST, own recordings).
- **Simultaneous hits:** F1 restricted to reference onsets that coincide within 20 ms with an onset of another class (hi-hat + snare, hi-hat + kick, kick + snare). Each combination is reported separately and compared with F1 on isolated hits. This shows directly what single-label decoding loses.
- **Timing error:** distribution of `est - ref` for matched onsets.
- **Latency:**
  - algorithmic latency, from hop and lookahead
  - measured compute time per frame and per 10 ms chunk on Apple Silicon CPU
  - end-to-end latency in the streaming simulation
- **Cost:** parameters, model size in MB, CPU % while streaming.
- **Significance:** paired Wilcoxon test over per-track F1 between models, with seeds averaged. This replaces McNemar on single-label predictions.
- **Offline vs streaming check:** the streaming run must give the same events as the offline run on the same file. This is a test.

The first output of the harness is an honest number for the **current** pipeline on the own `TestTrack*` files.

## What to keep, change or drop in the current code

| Keep (possibly moved) | Change | Drop or archive |
|---|---|---|
| `.env` + `constants.py` pattern (Docker already removed) | `spectogram_generator.py` → causal log-mel with cache | `src/depracated/` |
| `create_midi_file` and the GM note map | `base_classifier.py` train loop → generic frame-level trainer (seeds, BCE, checkpoint by val F1, history JSON) | Empty placeholder files (LSTM, Transformer, multilabel stubs) are replaced by the new modules |
| `DrumCNN` conv + SE blocks as building blocks | `model_evaluator_runner.py` → `transcription_eval.py` around `mir_eval` | `cnn_mel_cqt_ensemble_classifier.py` (CQT does not suit a causal 10 ms frame rate) |
| `single_label_metrics.py` for the baseline classifier only | `plotting_engine.py` → F1 per class, F1 vs latency, CPU vs F1 | `detect_bpm` from the real-time path (offline export only) |
| `onset_detector.py` + `onset_processor.py` for the baseline only | Tuner: one config object shared by tuner and trainer, so tuned values are actually used | `validation_sets_creator.py` once own data has proper annotations |

Before anything else, tag the current state (`v0-single-label`) so the baseline can always be reproduced.

### Proposed layout

```
midiConverter/src/
  data/
    datasets/        egmd.py, idmt.py, enst.py, own.py   -> (audio_path, events, split)
    labels.py        GM pitch -> class maps (3/5/7), events -> frame targets
    features.py      causal log-mel + .npy cache
    frame_dataset.py random excerpts, online augmentation
    synth_mix.py     multi-label mixes from own one-shots
    check_leakage.py
  models/frame/      causal_cnn.py, crnn.py, tcn.py, causal_transformer.py
  training/          train_frame.py, config.py
  inference/         peak_picking.py, transcriber.py, streaming.py
  evaluation/        transcription_eval.py, benchmark.py, comparison/
  baseline/          current single-label code, moved here and repaired
```

## Known problems in the current code

Found in the review of 2026-10-27, by reading the code and checking the data. Line numbers refer to that state. **Action** says what happens to each problem:
- **Fix (baseline):** fix it when the single-label code is repaired as the two-stage baseline.
- **Fix (shared):** fix it in code that the new pipeline keeps.
- **Obsolete:** the code is dropped or replaced, so no fix is needed. The lesson carries over to the new code.

### Data and methodology

| ID | Where | Problem | Action |
|---|---|---|---|
| D1 | `data/input/training/` | Splits are random by file. 114 groups of byte-identical files span splits, and near-duplicates (e.g. `DI_HiHat_Tip_1111.1/.3/.4`, `RealDrum01_xx`) appear in train, val and test. | Fix in stage 0. Split by recording; `check_leakage.py`. |
| D2 | `data/input/cuttedValSets/` | All 140 files are also in the training folders (112 in `test`, 112 in `val`, 28 in `train`), so the "real audio" evaluation is not independent of model selection. | Fix in stage 0. Remove the copies; keep the sets as own-kit test data only. |
| D3 | training vs `dl_audio_to_midi_converter.py` | Train/inference mismatch: training uses 2 s one-shots with full decay, while the converter feeds 0.06–0.25 s faded segments padded with silence. No saved metric measures the converter. | Fix (baseline): train on the same short post-onset windows as inference. |
| D4 | `TestTrack*.mid` | No simultaneous hits in any of the 6 test tracks, so they hide the single-label limitation. | Re-record with simultaneous hits (stage 5). |
| D5 | training split | Strong class imbalance (828 hi-hat vs 100 snare and 108 crash in train), without class weights or sampling. | Fix (baseline). The new trainer uses `pos_weight`. |
| D6 | whole `src/` | No random seeds anywhere. Three saved CQT results show identical real-audio confusion matrices, so they are one checkpoint evaluated three times, not three runs. | Fix (shared): seeds in every entry point; do not report those results as separate runs. |
| D7 | `validation_sets_creator.py:53` | Assumes exactly 7 onsets in a fixed order and only checks `< 8`. One extra onset shifts every later label. | Obsolete once own data has proper annotations. Check the existing `cuttedValSets` labels by ear before using them. |
| D8 | pipeline | Not causal: HPSS, normalization, BPM and peak picking use the whole file, and the classifier needs a 2 s window. | Obsolete in the new pipeline. The baseline stays offline and is reported as such. |

### Code bugs

| ID | Where | Problem | Action |
|---|---|---|---|
| B1 | `model_evaluator_runner.py:209` | The ensemble path calls `get_all_single_label_metrics(y_pred_proba=...)`, a keyword that does not exist. The `TypeError` is caught and the function returns `None`, so no ensemble result was ever saved. The outputs also have shape `(1, 7)` instead of `(7,)`. | Obsolete (ensemble dropped). Fix only if the ensemble is kept. |
| B2 | `dl_audio_to_midi_converter.py:85` | `use_mel_model` is not passed to `classify_drum_segments`, so a MEL model receives CQT input. | Fix (baseline). |
| B3 | `tunning_runner.py:12` | `EPOCHS` from `.env` is a string, so `EPOCHS > 5` raises `TypeError` on import. The tuner cannot start. | Fix (baseline) if the baseline is retuned. Lesson: parse config types in one place. |
| B4 | `cnn_objective.py`, `single_label_trainer.py` | Tuned `num_filters1/2` and `dropout_rate` are never used by the trainer, which falls back to 16/32 and the `.env` dropout. Tuning runs without a scheduler on 1/3 of the epochs; final training uses CosineAnnealing with `T_max=10` over 50 epochs, so the LR oscillates. | Fix (baseline). Lesson: one config object shared by tuner and trainer. |
| B5 | `base_classifier.py:219` | Early stopping counts the improving epoch itself as "no improvement" (the counter becomes 1). `best_val_loss` is tracked but unused. | Fix (baseline). The new trainer selects by val F1. |
| B6 | `calculation_engine.py:53` | `import statsmodels` does not load `statsmodels.stats`. The `AttributeError` is swallowed by a bare `except`, so every p-value is NaN. It also compares runs evaluated on different test sets (260 vs 372 samples). | Obsolete (replaced by the paired Wilcoxon test). |
| B7 | `dataset_augumentation.py:12` | A negative `time_shift` (up to 100 ms) cuts off the attack, which is the most informative part of a hit. | Fix (baseline). Rule for the new augmentation: never remove the attack. |
| B8 | `onset_processor.py:31` | `segment` is a view of `audio`, so the fades modify the original in place and can damage the attack of the next segment when onsets are close. | Fix (baseline). |
| B9 | `onset_processor.py:50` | In `filter_closely_spaced_onsets` both branches append the onset, so `min_distance` has no effect. The function is not called anywhere. | Fix or delete (baseline). |
| B10 | `dl_audio_to_midi_converter.py:82` | Velocity is `RMS * 12700` of a normalized segment, so it is almost always 127. | Fix (shared), in the kept MIDI writer. |
| B11 | `base_classifier.py:248` | `evaluate_model` rebuilds the whole augmented training set (with pitch shift) only to get the test set. | Fix (baseline). |
| B12 | `dl_audio_to_midi_converter.py:19` | Every segment is written to a temporary WAV and read back before classification. | Fix (baseline). It matters for the latency benchmark. |

## Stage 0 action plan (agreed 2026-10-27)

Goal: start stage 1 from clean, working code. Code that the new pipeline drops is **deleted, not fixed**. Only code that stays gets fixed. The pre-fix pipeline stays reachable through a tag, so the "before" number can still be measured in stage 1 (e.g. from a `git worktree` of the tag).

Each step is its own commit.

| Step | Work | Done when |
|---|---|---|
| A | Tag `v0-single-label` on the state before any cleanup. | The tag exists. |
| B1 | Delete dead code and generated junk: `src/depracated/`, the empty placeholder modules, `standard_audio_feature_extractor.py`, the MEL+CQT ensemble (B1), LSTM / Transformer / CNN-LSTM / scikit-learn entries in `.env`, `constants.py`, `run_script.py`, `script_running_constants.py` and the evaluator, `data/output/models/ml/`, `data/output/testingOnsets/`, `tempFiles/`. Trim `requirements.txt` (drop opencv, sphinx, torchvision, jupyter; add `mir_eval`; newer torch for MPS). | No reference to removed modules is left. |
| B2 | Move the single-label code to `src/baseline/` (a move-only commit, so later diffs stay readable). | Imports and commands updated; nothing else changed. |
| C | Native environment: `uv` venv with Python 3.12, install requirements, check imports and `torch.backends.mps`. | Every module imports; MPS is available. |
| D | Data (D1, D2): remove the `cuttedValSets` copies from `training/`; re-split the remaining files by source recording (group key from the filename); add `check_leakage.py`. | `check_leakage.py` passes; split sizes are recorded. |
| E | Fix the remaining "Fix (baseline)" and "Fix (shared)" items except D3 (B2–B12, D5, D6), with small pytest smoke tests for each fixed spot. | `pytest` passes. |

D3 (training on short windows) needs a new loader and a retrain, so it moves to stage 1b together with retraining the baseline.

## Stages and done criteria

| # | Stage | Done when |
|---|---|---|
| 0 | Clean up | Steps A–E of the stage 0 action plan above are done. |
| 1 | Evaluation harness | `transcription_eval.py` scores the current pipeline on `TestTrack*` with `mir_eval`, as it is today. The first honest baseline number is recorded. |
| 1b | Repaired baseline | The single-label code has the "Fix (baseline)" and "Fix (shared)" items done (D3, D5, B2–B5, B7–B12), is retrained on the cleaned split, and is scored by the same harness. Can run alongside stage 2. |
| 2 | Data pipeline | E-GMD subset and IDMT loaders, class maps, cached features. One batch of `(features, targets)` can be plotted and looks right. |
| 3 | First frame-level model | CRNN or causal CNN, 3 classes, trained on the E-GMD subset. F1 reported on E-GMD test, IDMT and own recordings. |
| 4 | Model comparison | All models × 3 seeds × lookahead {0, 1, 2}. F1, latency, CPU and size tables, plus significance tests. Then extend to 5/7 classes. |
| 5 | Domain gap | Own groove recordings annotated. Room-IR augmentation and synthetic mixes. Fine-tuning measured on a held-out own session. |
| 6 | Streaming simulation | The file is fed in 10 ms chunks through `step()`, and the events match the offline run. Latency and CPU are measured. |
| 7 | App | `sounddevice` input → model → `python-rtmidi` virtual port, recorded live in GarageBand. |

Stages 1 and 2 are independent and can run in parallel.

## Decided (2026-10-27)

1. **Compute:** native macOS on the M2 Pro (16 GB), training on the Apple GPU through PyTorch MPS. Docker is removed from the repository. A cloud GPU (Google or Azure) is used only if the Mac is too slow for the comparison runs. The run budget is cut from the full 45-run grid:
   - 5 models × 3 seeds at lookahead 1
   - a lookahead {0, 2} sweep for the two best models only

   That is about 27 runs. Inference and benchmarks always run natively, because CPU usage on Apple Silicon is a thesis metric.
2. **Classes:** 3 classes for the main results, simultaneous hits evaluated explicitly, and open hi-hat as the first extension (see Task definition).
3. **Lookahead:** the target use is recording MIDI while hearing the acoustic kit. A constant delay can be compensated, so jitter matters more than absolute latency. Lookahead of 0–2 frames is an experimental variable. A stricter budget would only apply to live triggering of GarageBand sounds, which is out of scope for now.

## Decisions needed

1. **Recording setup** for own grooves (one mic or several), which is still open in the thesis plan.
2. **Whether the repaired two-stage baseline stays** in the comparison (recommended), or the current code is archived.
3. **Whether GarageBand honours CoreMIDI timestamps** (to be checked in stage 7). This decides whether delay compensation happens in the app or by shifting the region.

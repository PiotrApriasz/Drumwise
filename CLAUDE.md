# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

Drumwise is a monorepo with three largely independent parts:

- `midiConverter/` — **the active area** (current branch `master-thesis-main`). A Python/PyTorch drum-audio → MIDI converter built for a Master's thesis at Warsaw University of Technology. Most work happens here.
- `backend/` — .NET 8 ASP.NET Core Minimal API (`Drumwise.sln`) for the drum-learning platform (homeworks, identity, MIDI-conversion endpoint). Dormant.
- `frontend/converter-admin-panel/` — Next.js 15 / React 19 / Tailwind 4 scaffold (essentially the create-next-app template).

The thesis direction (a real-time, causal, multi-label drum transcriber with live MIDI into GarageBand) and its open questions are in [docs/thesis-plan.md](docs/thesis-plan.md). Read it before proposing model or data changes.

`data/` (repo root) holds datasets, trained models and evaluation output.

## midiConverter

### Environment

The code runs natively on macOS (Apple Silicon) in a virtual environment. There is no Docker setup. Training is meant to use the Apple GPU through PyTorch MPS; a cloud GPU (Google or Azure) is the fallback if the Mac is too slow. All paths, model filenames and default hyperparameters come from the root `.env` (git-ignored) through `src/constants.py`. Relative paths in `.env` (e.g. `data/input/training`) are resolved against the repository root, so scripts work from any working directory.

```bash
cd midiConverter
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Run modules from `midiConverter/` with `python -m` so `src.*` imports resolve:

```bash
python run_script.py                           # interactive menu of the train/eval commands
python -m src.baseline.models.dl_classifiers.trainer.single_label_trainer --model-type cnn   # optional: --post-onset-ms --pre-onset-ms --seed --epochs --patience --batch-size --lr --weight-decay --label-smoothing --[no-]augment --num-workers
python -m src.baseline.evaluation.model_evaluator_runner --model-type cnn --save-results   # picks best_cnn_w<W>ms_s<seed>.pth via --post-onset-ms and --seed
python -m src.baseline.models.tuners.tunning_runner --model-type cnn --n-trials 50 [--storage sqlite:///tuning.db]
```

The canonical command strings live in `script_running_constants.py`. Run `python -m src.data.check_leakage` before any training: it fails if the same audio is in two splits or in a split and `CUTTED_VAL_SETS_PATH`. `pytest` (from `midiConverter/`) runs the smoke tests in `tests/`.

Splits in `TRAINING_PATH` are by source recording, not by file (see `src/baseline/data/tools/resplit_training_data.py`). The own-kit recordings are test-only and must not be copied into `training/`.

### Implementation status

Only the **CNN** classifier exists. The trainer, tuner and evaluator accept only `--model-type cnn`. The frame-level multi-label models from [docs/frame-level-plan.md](docs/frame-level-plan.md) are not written yet.

### Architecture

The current code is the single-label baseline (the two-stage system: onset detection, then one classification per onset). It lives in `src/baseline/`, and the paths below are relative to that folder; `src/constants.py`, `src/reproducibility.py` and `src/data/` (shared with the future frame-level models) sit outside it. The task is **single-label classification of drum hits** into the 7 classes of `DRUM_INSTRUMENTS` (`snare, crash, floor_tom, hi-hat, kick, rack_tom, ride`). Label indices follow that list's order everywhere. The design and the measurements behind it are in docs/frame-level-plan.md ("Stage 1b: baseline redesign").

- **Onset windows** (`data/windowing.py`): the model sees a fixed window from `BASELINE_PRE_ONSET_MS` before to `BASELINE_POST_ONSET_MS` (W) after the onset, never the whole file. `cut_window` is the single function used in training, in the own-kit evaluation and in the converter; `find_onset` locates the onset of a one-shot file. W is the latency the baseline pays.
- **Data loading**: `data/dataset/single_label_dataset_loader.py` reads `TRAINING_PATH/{train,val,test}/<instrument>/*.wav` at 22050 Hz mono and returns `Hit` objects (peak-normalized audio from 100 ms before to 1 s after the onset, the onset index, the source dataset).
- **Augmentation** (`data/dataset/augmentation.py`) is online, in `WindowDataset.__getitem__`: onset jitter, the decay of another random hit, noise, random EQ. No pitch shift.
- **Features** (`src/data/features.py`): causal log-mel (`n_fft=1024`, hop 220, 128 bands, `center=False`, `log(1 + x)`). Per-band standardization is fitted on the training windows and stored in the model (`InputNorm`).
- **Models**: `BaseClassifier` (an `nn.Module` and an ABC) owns datasets, the train loop, checkpointing and `evaluate_model`. It runs on MPS when available (`src/reproducibility.get_device`), weights the loss by inverse class frequency, and selects and early-stops on validation macro-F1. `DrumCNN` is three conv blocks (conv-BN-ReLU ×2, SE, frequency-only pooling) with global average pooling, ~0.3 M parameters. `build_cnn_optimization` (AdamW, warm-up, cosine) is shared by trainer and tuner. All numeric settings come typed from `src/constants.py`; `.env.example` lists them.
- **Checkpoints** (`models/dl_classifiers/classifiers/checkpoints.py`) store the weights plus the window and feature settings; `load_cnn` returns `(model, window, features)`, so evaluation and conversion always use the training input. File names carry W and the seed: `best_cnn_w100ms_s42.pth`. Checkpoints from before 2026-10-27 (plain state dicts) are rejected.
- **Conversion pipeline** (`converters/dl_audio_to_midi_converter.py`): load and normalize audio → `onset_detection/onset_detector.py` (librosa onsets) → one `cut_window` per onset → `conversion_utils.classify_window` → MIDI via `pretty_midi`, velocity from the window's peak level, BPM from `conversion_utils.detect_bpm`.
- **Evaluation** (`evaluation/model_evaluator_runner.py`) runs two passes per model: (1) the test split through `evaluate_model` (per-class and per-source accuracy) plus detailed metrics from `metrics/single_label_metrics.py`; (2) the own-kit hits in `CUTTED_VAL_SETS_PATH`, each cut with the same onset window (`conversion_utils.classify_hit_file`), labels from the file name. Crash, ride and the toms in the training data come from one library kit, so (2) is the number that shows generalization to another kit. `--save-results` writes JSON and confusion-matrix plots under `EVALUATION_RESULTS_PATH`; the trainer also saves a training-history JSON there.
- **Cross-model comparison** (`evaluation/comparison/`): `eval_data_tools.load_evaluation_results` reads those saved JSONs. `plotting_engine` draws accuracy, ROC, learning-curve, per-class and confusion-grid comparisons. `calculation_engine` runs McNemar significance tests.
- **Tuning**: `models/tuners/tunning_runner.py` runs Optuna over `cnn_objective.objective_cnn` and writes trial artifacts to `TRIAL_ARTIFACTS_PATH`.

## backend (.NET)

Clean-architecture layering: `api/Drumwise.API` (Minimal API endpoints in `Endpoints/`, registered via `ApplicationEndpoints.cs`, with a client API-key middleware) → `core/Drumwise.Features` (vertical slices: one MediatR command/query plus handler per file, with FluentValidation validators in `Validation/`) → `core/Drumwise.Application` (entities, interfaces, `Result`/`Error` types, MediatR pipeline behaviours) and `core/Drumwise.Infrastructure` (EF Core with two contexts: `AppIdentityDbContext` and `ApplicationDbContext`, each with its own migrations folder; ASP.NET Identity; Google Drive file storage; domain events dispatched by an EF interceptor). The API uses local SQLite files (`app.db`, `main.db`).

```bash
dotnet build backend/Drumwise.sln
dotnet run --project backend/api/Drumwise.API
```

## frontend

```bash
cd frontend/converter-admin-panel && npm run dev   # also: build, start, lint
```

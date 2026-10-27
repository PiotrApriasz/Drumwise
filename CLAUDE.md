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
python -m src.baseline.models.dl_classifiers.trainer.single_label_trainer --model-type cnn --feature-type cqt   # optional: --epochs --patience --batch-size --lr --weight-decay --label-smoothing --augment-factor --num-workers
python -m src.baseline.evaluation.model_evaluator_runner --model-type cnn --feature-type mel --save-results
python -m src.baseline.models.tuners.tunning_runner --model-type cnn --feature-type cqt --n-trials 50 [--storage sqlite:///tuning.db]
```

The canonical command strings live in `script_running_constants.py`. Run `python -m src.data.check_leakage` before any training: it fails if the same audio is in two splits or in a split and `CUTTED_VAL_SETS_PATH`. `pytest` (from `midiConverter/`) runs the smoke tests in `tests/`.

Splits in `TRAINING_PATH` are by source recording, not by file (see `src/baseline/data/tools/resplit_training_data.py`). The own-kit recordings are test-only and must not be copied into `training/`.

### Implementation status

Only the **CNN** classifier exists. The trainer, tuner and evaluator accept only `--model-type cnn`. The frame-level multi-label models from [docs/frame-level-plan.md](docs/frame-level-plan.md) are not written yet.

### Architecture

The current code is the single-label baseline. It lives in `src/baseline/`, and the paths below are relative to that folder. Only `src/constants.py` sits outside it. The task is **single-label classification of short drum hits** into the 7 classes of `DRUM_INSTRUMENTS` (`snare, crash, floor_tom, hi-hat, kick, rack_tom, ride`). Label indices follow that list's order everywhere. A full track is turned into MIDI by onset detection followed by per-segment classification.

- **Data loading**: `data/dataset/single_label_dataset_loader.py` reads `TRAINING_PATH/{train,val,test}/<instrument>/*.wav`, resamples to 22050 Hz mono, pads or trims to 2 s, and normalizes. Augmentation (`dataset_augumentation.py`) expands only the training split by `augment_factor`.
- **Features**: `feature_type` is `cqt` or `mel`. Spectrograms are computed on the fly in the Dataset `__getitem__` (`CqtDrumDataset` / `MelDrumDataset` in `base_classifier.py`, using `feature_extraction/spectogram_generator.py`). They are not precomputed.
- **Models**: `BaseClassifier` (an `nn.Module` and an ABC) owns dataset creation, dataloaders, the train loop with early stopping, best-model checkpointing, and `evaluate_model`. Subclasses only implement `_build_model()`. `DrumCNNClassifier` wraps `DrumCNN` (conv layers with SE blocks).
- **Model paths by naming convention**: trainer and evaluator resolve filenames with `getattr(constants, f"{FEATURE}_{MODEL}_MODEL_NAME")` and `f"{FEATURE}_BEST_{MODEL}_MODEL_NAME"`, joined to `TRAINED_DL_MODELS_PATH`. Adding a new model type means adding those env vars, the constants in `constants.py`, and a branch in `get_classifier_class` (duplicated in both `single_label_trainer.py` and `model_evaluator_runner.py`, plus `get_model_implementation_class` in the evaluator).
- **Conversion pipeline** (`converters/dl_audio_to_midi_converter.py`): load audio → `onset_detection/onset_detector.py` (librosa onsets) → `onset_processor.py` (filter closely spaced onsets, cut ~0.15 s segments) → classify each segment with the CNN → write MIDI with `pretty_midi`/`mido`, using BPM from `conversion_utils.detect_bpm`.
- **Evaluation** (`evaluation/model_evaluator_runner.py`) runs two passes per model: (1) the held-out test split through `evaluate_model` plus detailed metrics from `metrics/single_label_metrics.py` (per-class metrics, confusion matrix, ROC-AUC, PR curves); (2) real-audio evaluation over `CUTTED_VAL_SETS_PATH`, where the ground-truth label comes from the filename via `conversion_utils.extract_true_label`. `--save-results` writes JSON and confusion-matrix plots under `EVALUATION_RESULTS_PATH`. The trainer also saves a `*_training_history_*.json` there.
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

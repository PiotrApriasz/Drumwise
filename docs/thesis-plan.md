# Master's thesis plan: real-time drum audio → MIDI

Working notes on the thesis direction for `midiConverter/`. Sections are split by status: **Decided**, **Proposed** (to be confirmed), and **Open**. The topic is not formally approved yet and there is no deadline.

## Decided

- **Topic:** keep Drumwise/midiConverter (acoustic drum audio → MIDI), since the foundations already exist. The air-drumming idea is shelved.
- **Main direction:** a real-time audio → MIDI system for acoustic drums. The core is models that detect hits during playing and classify the instrument.
- **Practical extension:** record the recognized hits live onto a track in GarageBand.
- **Scientific contribution:** a comparison of AI models for drum instrument recognition. This is the background of the thesis.
- **Hardware:** an acoustic kit is available. An electronic kit may be added during the work.
- **Task formulation:** frame-level multi-label onset detection. Details are in [frame-level-plan.md](frame-level-plan.md).
- **Compute (2026-10-27):** everything runs natively on the Mac (M2 Pro, 16 GB), with training on the Apple GPU through PyTorch MPS. Docker is removed. If the Mac turns out too slow, a cloud GPU (Google or Azure) is considered.
- **Classes (2026-10-27):** the main results use 3 classes (kick, snare, hi-hat), with explicit evaluation of simultaneous hits (hi-hat + snare, hi-hat + kick). Open hi-hat is the first extension.
- **Lookahead (2026-10-27):** the target use is recording MIDI while hearing the acoustic kit, so a constant delay can be compensated. Lookahead of 0–2 frames (0–20 ms) is an experimental variable, and the F1-versus-latency curve is a result.

## Proposed (to be confirmed)

### GarageBand integration
- A standalone app exposing a **virtual CoreMIDI port**, rather than a plugin. As far as we know, GarageBand has no MIDI FX slot (to be verified).
- Prototype in Python: `sounddevice` for audio input and `python-rtmidi` for MIDI output.

### Latency and evaluation
- For recording, only the onset timestamp has to be accurate, and timing jitter matters more than a constant delay. A MIDI note-on carries the instrument, so the note can only be sent once the class is known. A constant offset can then be compensated by timestamping notes or by shifting the region.
- Metrics:
  - per-class F1 with a timing tolerance (`mir_eval`)
  - latency
  - CPU usage on Apple Silicon
  - model size

### Models
- Models must be **causal**. They may not look ahead past the current hit.
- Candidates:
  - a CNN on a short window around the onset
  - a unidirectional LSTM/GRU
  - a causal Transformer
  - short-window mel features instead of CQT
- **Multi-label** output (sigmoid + BCE per class), because simultaneous hits such as kick + hi-hat are the norm.

### Data
- **Domain gap:** Groove MIDI is recorded on an e-drum, while the target is an acoustic kit recorded with a microphone.
- Candidate datasets:
  - E-GMD (more hours and kit sounds)
  - ENST-Drums and IDMT-SMT-Drums as external test sets
  - own recordings
- Own acoustic recordings:
  - single hits of each kit piece. The labels come for free, and synthetic multi-label mixes can be built from them.
  - grooves played to a metronome following a MIDI score
  - optionally several microphones or triggers

### Cloud baseline
- Azure AI Foundry as a comparison variant. It should show that the cloud is not viable for real time because of network latency.

## Proposed order of work

1. **Offline:** compare causal, multi-label models, measuring latency and CPU.
2. **Streaming simulation:** feed the same model an audio file chunk by chunk.
3. **App:** a virtual MIDI port, tested in GarageBand.

A detailed proposal for moving to frame-level multi-label transcription (data, models, evaluation, what to keep from the current code) is in [frame-level-plan.md](frame-level-plan.md).

## Open questions

- Recording hardware: audio interface, number of inputs, single microphone or several.
- The exact set of models to compare.
- The final choice of data.
- The role of Foundry.
- Formal approval of the topic. No deadline is set.

## Current state of the code (as of 2026-10-27)

What `midiConverter/` actually does today. Note that this differs in places from the plan above.

- **Task:** single-label classification into 7 classes: `snare, crash, floor_tom, hi-hat, kick, rack_tom, ride` (order of `DRUM_INSTRUMENTS` in `src/constants.py`).
- **Model:** a CNN with SE blocks (`DrumCNNClassifier`) on CQT or mel spectrograms. Hyperparameters were tuned with Optuna. A MEL+CQT ensemble averages the logits of the two best CNNs. The LSTM, Transformer and CNN-LSTM classifier files are empty.
- **Results:** saved test-set accuracy is 0.92–1.0 depending on the run (`data/output/evaluation/`). These numbers are not trustworthy because of data leakage between splits (see [frame-level-plan.md](frame-level-plan.md)). The ~95.7% best validation accuracy from earlier notes is not recorded in the saved results.
- **Data:** about 2,900 one-shot WAV samples split into `train/val/test` folders per instrument (2150 / 371 / 372), padded or trimmed to 2 s. The **Groove MIDI Dataset is not integrated** (`GROOVE_MIDI_DATASET_PATH = "toImplement"`).
- **Conversion pipeline:** offline only. Librosa onset detection on a whole file is followed by classification of each ~0.15 s segment. The pipeline is neither causal nor streaming.
- **Bidirectional LSTM:** only present in the old end-to-end model under `src/depracated/`. It is not part of the current pipeline.
- **Stack:** PyTorch, librosa, pretty_midi/mido, run natively on macOS (Docker removed on 2026-10-27).

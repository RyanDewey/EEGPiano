# EEGPiano — Full Model

SSVEP-based brain-computer interface that plays piano using EEG and EMG. A Unicorn Hybrid Black headset streams EEG to LSL; phase-reversing checkerboard stimuli at **7.5, 10, and 12 Hz** drive steady-state visual evoked potentials (SSVEP); an FBTRCA classifier identifies which frequency you are attending to; and EMG clenches trigger the corresponding piano note.

---

## Modes

| Mode | What it does |
|------|-------------|
| **TRAIN** | Displays SSVEP stimuli while Lab Recorder captures EEG + markers to `.xdf`. Collect data to retrain the model. |
| **TEST** | Automated 6-trial × 10s accuracy test. Prints per-frequency accuracy to terminal. |
| **RUN** | Interactive piano — assign notes to frequencies by clicking, then play with your brain and EMG. |

---

## Quick start

```bash
# One-time setup
brew install blueutil
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
# 1. Set MODE in start.py ("TRAIN", "TEST", or "RUN")
python start.py
```

See `instructions.txt` for detailed setup, Lab Recorder workflow, and troubleshooting.

---

## Files

| File | Purpose |
|------|---------|
| `start.py` | Launcher — set `MODE` and `MODEL_FILE` here |
| `gui.py` | Unified GUI for all three modes |
| `live_pipeline.py` | Real-time FBTRCA classifier (TEST / RUN) |
| `EMG_live.py` | Live EMG SVM → LSL outlet `EMGControl` |
| `train.py` | Retrain FBTRCA from `.xdf` calibration files |
| `unicornlsl.py` | Unicorn Hybrid Black → LSL EEG stream |
| `fbtrca_model.py` | FBTRCA algorithm (filter-bank TRCA) |
| `fbtrca_model_751012.pkl` | Trained model (7.5 / 10 / 12 Hz) |
| `newGuiData/` | Training `.xdf` files (freq in filename) |

---

## Retraining

```bash
# Collect new .xdf files in TRAIN mode, save to newGuiData/
# Name files with the frequency: 7.5hz_run-001_eeg.xdf, 10hz_run-001_eeg.xdf, etc.

python train.py --validate   # LOFO cross-validation then saves fbtrca_model_751012.pkl
```

---

## Hardware

- gTec Unicorn Hybrid Black EEG headset
- macOS (tested), Windows / Linux should work
- Monitor ≥ 60 Hz (all three frequencies divide exactly into 60 Hz)

---

## LSL streams

| Stream | Type | Description |
|--------|------|-------------|
| `Unicorn` | EEG | 16-channel, 250 Hz (ch 0–7 = EEG µV) |
| `SSVEPMarkers` | Markers | Stimulus onsets, trial cues, session events |
| `SSVEPPredictions` | Predictions | Classified frequency (Hz) per stride |
| `EMGControl` | Markers | `-1` left clench, `0` rest, `+1` right clench |

#!/usr/bin/env python3
"""
live_pipeline.py — Real-time FBTRCA SSVEP classifier

Usage:
    python live_pipeline.py [MODE] [MODEL_FILE]

    MODE       : "TEST" or "RUN"  (default: "RUN")
    MODEL_FILE : path to .pkl     (default: "fbtrca_model_751012.pkl")

TEST mode: waits for SSVEPMarkers, tracks per-trial and session accuracy.
RUN  mode: streams predictions to SSVEPPredictions outlet, no scoring.
"""

import pickle
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
from scipy import signal
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop

from fbtrca_model import FBTRCA, class_to_freq_map

# ── CLI args ──────────────────────────────────────────────────────────────────
MODE       = sys.argv[1] if len(sys.argv) > 1 else "RUN"
MODEL_FILE = sys.argv[2] if len(sys.argv) > 2 else "fbtrca_model_751012.pkl"

if MODE not in ("TEST", "RUN"):
    print(f"[MODEL] Unknown mode '{MODE}' — defaulting to RUN")
    MODE = "RUN"

print(f"[MODEL] Mode: {MODE}  |  Model: {MODEL_FILE}")

# ── LSL prediction outlet ─────────────────────────────────────────────────────
pred_info = StreamInfo('SSVEPPredictions', 'Predictions', 1, 0, 'float32', 'ssvep_pred')
pred_outlet = StreamOutlet(pred_info)
print('[MODEL] Prediction outlet "SSVEPPredictions" ready.')

# ── Marker stream (TEST mode requires it; RUN mode tries once) ────────────────
markers = None

if MODE == "TEST":
    print('[MODEL] Waiting for SSVEPMarkers stream...')
    while True:
        ms = resolve_byprop('name', 'SSVEPMarkers', timeout=2)
        if not ms:
            ms = resolve_byprop('type', 'Markers', timeout=2)
        if ms:
            markers = StreamInlet(ms[0])
            print('[MODEL] Connected to marker stream.')
            break
        print('[MODEL] No marker stream yet, retrying...')
        time.sleep(1)
else:
    ms = resolve_byprop('name', 'SSVEPMarkers', timeout=2)
    if not ms:
        ms = resolve_byprop('type', 'Markers', timeout=2)
        ms = [s for s in (ms or []) if s.name() != 'EMGControl']
    if ms:
        markers = StreamInlet(ms[0])
        print('[MODEL] Connected to marker stream (optional, for stimulus gating).')
    else:
        print('[MODEL] No marker stream — running without stimulus gating.')

# ── EEG stream ────────────────────────────────────────────────────────────────
print('[MODEL] Waiting for EEG stream...')
while True:
    es = resolve_byprop('type', 'EEG', timeout=2)
    if es:
        eeg = StreamInlet(es[0])
        print('[MODEL] Connected to EEG stream.')
        break
    print('[MODEL] No EEG stream yet, retrying...')
    time.sleep(1)

# ── Load model ────────────────────────────────────────────────────────────────
with open(MODEL_FILE, 'rb') as f:
    state = pickle.load(f)

print(f'[MODEL] Loaded {MODEL_FILE}')

model         = state['model']
freqs         = state['freqs']
freq_to_class = state['freq_to_class']
sfreq         = state['sfreq']
WIN_SEC       = state['win_sec']
STRIDE_SEC    = state['stride_sec']
DROP_START    = state['drop_start_sec']
LINE_FREQ     = state['line_freq']
BANDPASS      = state['bandpass']
RESAMPLE_HZ   = state['resample_hz']

n_channels = state.get('n_channels', None)
if n_channels is None:
    _, first_template = model.models[0][0]
    n_channels = first_template.shape[0]

freq_map = class_to_freq_map(freq_to_class)

# ── EMG clench-gating inlet (RUN mode) ───────────────────────────────────────
emg_inlet      = None
last_clench_t  = -999.0
CLENCH_GATE_SEC = 2.0   # seconds to suppress predictions after a clench

if MODE == "RUN":
    emg_streams = resolve_byprop('name', 'EMGControl', timeout=1)
    if emg_streams:
        emg_inlet = StreamInlet(emg_streams[0])
        print('[MODEL] Connected to EMGControl — clench gating active.')

print(f'[MODEL] Classes: {model.K}  |  Frequencies: {sorted(freq_to_class.keys())}')
print(f'[MODEL] Window: {WIN_SEC}s  |  Stride: {STRIDE_SEC}s  |  Channels: {n_channels}')

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(epoch, sfreq, line_freq, bandpass, resample_hz):
    epoch = np.array(epoch, dtype=np.float64)
    # 1. CAR
    epoch = epoch - epoch.mean(axis=0, keepdims=True)
    # 2. Notch 60 Hz + harmonics
    nyq = sfreq / 2
    nf  = line_freq
    while nf < nyq:
        b, a  = signal.iirnotch(w0=nf, Q=30, fs=sfreq)
        epoch = signal.filtfilt(b, a, epoch, axis=-1)
        nf   += line_freq
    # 3. Bandpass
    sos   = signal.butter(4, [bandpass[0], bandpass[1]],
                          btype='bandpass', fs=sfreq, output='sos')
    epoch = signal.sosfiltfilt(sos, epoch, axis=-1)
    # 4. Resample
    if abs(sfreq - resample_hz) > 1e-9:
        n_out = int(round(epoch.shape[1] * (resample_hz / sfreq)))
        epoch = signal.resample(epoch, n_out, axis=-1)
        sfreq = float(resample_hz)
    return epoch, sfreq

# ── Rolling buffer ────────────────────────────────────────────────────────────

buffer_sec  = max(4.0, WIN_SEC + 2.0)
buffer_len  = int(buffer_sec * sfreq)
window_len  = int(WIN_SEC * sfreq)
drop_warmup = int(DROP_START * sfreq)
batch       = np.zeros((n_channels, buffer_len), dtype=np.float64)

samples_seen        = 0
next_predict_sample = 0
stride_len          = int(STRIDE_SEC * sfreq)
in_stimulus         = False

# ── TEST mode accuracy tracking ───────────────────────────────────────────────
current_target = None
trial_idx      = 0
trial_correct  = 0
trial_total    = 0
total_correct  = 0
total_total    = 0
per_freq       = defaultdict(lambda: [0, 0])
session_done   = False

MARKER_TARGET_BASE    = 100.0
MARKER_STIM_START     = 252.0
MARKER_STIM_STOP      = 253.0
MARKER_SESSION_END    = 255.0
MARKER_CALIB_START    = 280.0
MARKER_CALIB_CUE_BASE = 150.0
MARKER_CALIB_DONE     = 281.0

# ── Calibration state ────────────────────────────────────────────────────────
in_calibration      = False
calib_windows       = []
calib_labels        = []
calib_model         = None
calib_current_class = None
CALIB_WEIGHT        = 0.3   # fraction of score given to session-specific calib model
_FB_BANDS   = state.get('fb_bands',   [(5,45),(12,45),(18,45),(24,45),(30,45)])
_WEIGHT_EXP = state.get('weight_exp', 1.25)

print('[MODEL] Running — waiting for data...\n')

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:

    # ── Pull EEG chunk ────────────────────────────────────────────────────────
    chunk, _ = eeg.pull_chunk(timeout=0.0)
    if chunk:
        new_data = np.asarray(chunk, dtype=np.float64).T
        new_data = new_data[:n_channels, :]

        if new_data.shape[0] != n_channels:
            print(f'[MODEL] WARNING: channel mismatch — got {new_data.shape[0]}, '
                  f'expected {n_channels}. Skipping chunk.')
            continue

        new_samples   = new_data.shape[1]
        samples_seen += new_samples

        if new_samples >= buffer_len:
            batch = new_data[:, -buffer_len:]
        else:
            batch[:, :-new_samples] = batch[:, new_samples:]
            batch[:, -new_samples:] = new_data
    else:
        time.sleep(0.002)

    # ── Pull markers (non-blocking) ───────────────────────────────────────────
    if markers is not None:
        while True:
            m_sample, _ = markers.pull_sample(timeout=0.0)
            if m_sample is None:
                break
            m = float(m_sample[0])

            # Calibration markers checked first — calib cues (150+freq) overlap
            # with the TEST target cue range (100–200) so must take priority.
            if m == MARKER_CALIB_START:
                in_calibration      = True
                calib_windows       = []
                calib_labels        = []
                calib_current_class = None
                batch[:] = 0.0
                samples_seen        = 0
                next_predict_sample = 0
                print('[MODEL] Calibration started — collecting data...')

            elif in_calibration and MARKER_CALIB_CUE_BASE < m < MARKER_CALIB_CUE_BASE + 50:
                freq_val = m - MARKER_CALIB_CUE_BASE
                if freq_val in freq_to_class:
                    calib_current_class = freq_to_class[freq_val]
                    batch[:] = 0.0
                    samples_seen        = 0
                    next_predict_sample = 0
                    print(f'[MODEL] Calibration cue: {freq_val} Hz '
                          f'(class {calib_current_class})')

            elif m == MARKER_STIM_START:
                in_stimulus = True

            elif m == MARKER_STIM_STOP:
                in_stimulus = False
                if MODE == "TEST" and current_target is not None and trial_total > 0:
                    acc = 100 * trial_correct / trial_total
                    print(f'[MODEL] ── Trial {trial_idx} result: '
                          f'{trial_correct}/{trial_total} = {acc:.0f}%  '
                          f'(target {current_target} Hz)\n')

            elif MODE == "TEST" and 100 < m < 200:
                new_target = m - MARKER_TARGET_BASE
                if current_target is not None and trial_total > 0:
                    acc = 100 * trial_correct / trial_total
                    print(f'[MODEL] ── Trial {trial_idx} result: '
                          f'{trial_correct}/{trial_total} = {acc:.0f}%  '
                          f'(target {current_target} Hz)\n')
                current_target  = new_target
                trial_idx      += 1
                trial_correct   = 0
                trial_total     = 0
                samples_seen    = 0
                next_predict_sample = 0
                batch[:] = 0.0
                print(f'[MODEL] ── Trial {trial_idx} started  →  target: {current_target} Hz')

            elif m == MARKER_CALIB_DONE:
                in_calibration = False
                if len(calib_windows) >= len(freqs):
                    X_c = np.stack(calib_windows)
                    y_c = np.array(calib_labels, dtype=int)
                    print(f'[MODEL] Fitting calibration model on '
                          f'{len(y_c)} windows...')
                    try:
                        calib_model = FBTRCA(
                            sfreq=RESAMPLE_HZ,
                            filter_bands=_FB_BANDS,
                            weight_exp=_WEIGHT_EXP,
                            trca_reg=1e-6,
                        ).fit(X_c, y_c)
                        print('[MODEL] Calibration model ready — '
                              f'ensemble weight: {CALIB_WEIGHT:.0%} calib '
                              f'/ {1-CALIB_WEIGHT:.0%} base')
                    except Exception:
                        print('[MODEL] Calibration model fit failed:')
                        traceback.print_exc()
                        calib_model = None
                else:
                    print('[MODEL] Not enough calibration windows — skipping fit')
                calib_current_class = None

            elif MODE == "TEST" and m == MARKER_SESSION_END:
                print()
                print('[MODEL] ══════════════════════════════════════════')
                print('[MODEL]           SESSION COMPLETE')
                print('[MODEL] ══════════════════════════════════════════')
                if total_total > 0:
                    oa = 100 * total_correct / total_total
                    print(f'[MODEL] Overall accuracy: '
                          f'{total_correct}/{total_total} = {oa:.1f}%')
                print('[MODEL] Per-frequency breakdown:')
                for freq in sorted(per_freq):
                    c, t = per_freq[freq]
                    pa   = 100 * c / t if t > 0 else float('nan')
                    print(f'[MODEL]   {freq:5.1f} Hz  →  {c}/{t} = {pa:.0f}%')
                print('[MODEL] ══════════════════════════════════════════\n')
                session_done   = True
                current_target = None

    # ── Stride gate: predict every stride_len samples ─────────────────────────
    if not (samples_seen >= next_predict_sample + stride_len):
        continue
    next_predict_sample = samples_seen

    if samples_seen < window_len + drop_warmup:
        continue

    # Stimulus gating: TEST mode only — RUN mode predicts continuously
    if MODE == "TEST" and markers is not None and not in_stimulus:
        continue

    # ── EMG clench gating: drain EMGControl and skip post-clench windows ──────
    if emg_inlet is not None:
        while True:
            esamp, _ = emg_inlet.pull_sample(timeout=0.0)
            if esamp is None:
                break
            if float(esamp[0]) != 0.0:
                last_clench_t = time.time()
    if time.time() - last_clench_t < CLENCH_GATE_SEC:
        continue

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        buf_proc, epoch_sfreq = preprocess(
            batch,
            sfreq=sfreq,
            line_freq=LINE_FREQ,
            bandpass=BANDPASS,
            resample_hz=RESAMPLE_HZ,
        )
        win_resampled = int(WIN_SEC * epoch_sfreq)
        epoch = buf_proc[:, -win_resampled:]

        # 5. Detrend
        epoch = signal.detrend(epoch, axis=-1, type='linear')
        # 6. Per-channel normalize
        std   = epoch.std(axis=-1, keepdims=True) + 1e-12
        epoch = epoch / std
        # 7. Artifact rejection
        ptp = epoch.max(axis=-1) - epoch.min(axis=-1)
        if np.any(ptp > 100.0):
            print(f'[MODEL] artifact rejected (max ptp={ptp.max():.1f})')
            continue

        # During calibration: collect labeled window, don't predict
        if in_calibration:
            if calib_current_class is not None:
                calib_windows.append(epoch.copy())
                calib_labels.append(calib_current_class)
            continue

        # Predict — ensemble with calibration model if available
        if calib_model is not None:
            base_scores,  _ = model.predict_scores(epoch)
            calib_scores, _ = calib_model.predict_scores(epoch)
            # Softmax-normalize before blending so raw score magnitude
            # differences between models don't let one class dominate
            def _softmax(x):
                e = np.exp(x - x.max())
                return e / e.sum()
            fused           = (1 - CALIB_WEIGHT) * _softmax(base_scores) + \
                              CALIB_WEIGHT * _softmax(calib_scores)
            predicted_class = int(np.argmax(fused))
        else:
            predicted_class, _ = model.predict(epoch)
        predicted_freq = freq_map[predicted_class]

        pred_outlet.push_sample([float(predicted_freq)])

        if MODE == "TEST":
            if current_target is not None and not session_done:
                correct        = (predicted_freq == current_target)
                trial_correct += int(correct)
                trial_total   += 1
                total_correct += int(correct)
                total_total   += 1
                per_freq[current_target][0] += int(correct)
                per_freq[current_target][1] += 1
                mark = '✓' if correct else '✗'
                print(f'[MODEL] {mark}  predicted {predicted_freq:5.1f} Hz  '
                      f'(target {current_target:5.1f} Hz)  '
                      f'[{trial_correct}/{trial_total} this trial]')
            else:
                print(f'[MODEL] predicted {predicted_freq} Hz  (no active trial)')
        else:
            print(f'[MODEL] predicted {predicted_freq} Hz')

    except Exception:
        print('[MODEL] ERROR during prediction — skipping this window:')
        traceback.print_exc()

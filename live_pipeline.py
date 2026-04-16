# Live SSVEP pipeline — 4-frequency accuracy test
# Loads fbtrca_model_4freq.pkl (2 s window, 1 s stride)
# Tracks classification accuracy against cue markers from gui.py

import time
import pickle
import traceback
import numpy as np
from collections import defaultdict
from scipy import signal
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet
from fbtrca_model import FBTRCA, class_to_freq_map

MARKER_SESSION_END = 255.0
MARKER_TARGET_BASE = 100.0   # cue marker = 100 + freq
MARKER_STIM_START  = 252.0
MARKER_STIM_STOP   = 253.0

# ── LSL prediction outlet ─────────────────────────────────────────────────────
pred_info = StreamInfo('SSVEPPredictions', 'Predictions', 1, 0, 'float32', 'ssvep_pred')
pred_outlet = StreamOutlet(pred_info)
print('[MODEL] Prediction outlet "SSVEPPredictions" ready.')

# ── Wait for marker stream (from gui.py) ──────────────────────────────────────
print('[MODEL] Waiting for SSVEPMarkers stream...')
while True:
    ms = resolve_byprop('type', 'Markers', timeout=2)
    if ms:
        markers = StreamInlet(ms[0])
        print('[MODEL] Connected to marker stream.')
        break
    print('[MODEL] No marker stream yet, retrying...')
    time.sleep(1)

# ── Wait for EEG stream ───────────────────────────────────────────────────────
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
with open('fbtrca_model_751012.pkl', 'rb') as f:
    state = pickle.load(f)

print('[MODEL] Loaded fbtrca_model_751012.pkl')

model       = state['model']
freqs       = state['freqs']
freq_to_class = state['freq_to_class']
sfreq       = state['sfreq']
WIN_SEC     = state['win_sec']       # 2.0 s
STRIDE_SEC  = state['stride_sec']   # 1.0 s
DROP_START  = state['drop_start_sec']
DROP_END    = state['drop_end_sec']
LINE_FREQ   = state['line_freq']
BANDPASS    = state['bandpass']
RESAMPLE_HZ = state['resample_hz']
FB_BANDS    = state['fb_bands']
WEIGHT_EXP  = state['weight_exp']

n_channels = state.get('n_channels', None)
if n_channels is None:
    _, first_template = model.models[0][0]
    n_channels = first_template.shape[0]

freq_map = class_to_freq_map(freq_to_class)

print(f'[MODEL] Classes: {model.K}  |  Frequencies: {sorted(freq_to_class.keys())}')
print(f'[MODEL] Window: {WIN_SEC}s  |  Stride: {STRIDE_SEC}s  |  Channels: {n_channels}')

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(epoch, sfreq, line_freq, bandpass, resample_hz):
    epoch = np.array(epoch, dtype=np.float64)
    epoch = epoch - epoch.mean(axis=0, keepdims=True)

    nyq = sfreq / 2
    nf  = line_freq
    while nf < nyq:
        b, a  = signal.iirnotch(w0=nf, Q=30, fs=sfreq)
        epoch = signal.filtfilt(b, a, epoch, axis=-1)
        nf   += line_freq

    sos   = signal.butter(4, [bandpass[0], bandpass[1]],
                          btype='bandpass', fs=sfreq, output='sos')
    epoch = signal.sosfiltfilt(sos, epoch, axis=-1)

    if abs(sfreq - resample_hz) > 1e-9:
        n_out = int(round(epoch.shape[1] * (resample_hz / sfreq)))
        epoch = signal.resample(epoch, n_out, axis=-1)
        sfreq = float(resample_hz)

    # Detrend then normalize — matches offline windowize order
    epoch = signal.detrend(epoch, axis=-1, type='linear')
    std   = epoch.std(axis=-1, keepdims=True) + 1e-12
    epoch = epoch / std

    return epoch, sfreq

# ── Rolling buffer ────────────────────────────────────────────────────────────

# Keep extra context on each side of the window so the bandpass filter
# (0.5 Hz lowcut, period = 2 s) has room to settle before we crop.
FILTER_PAD_SEC = 2.0                             # seconds of padding each side
filter_pad     = int(FILTER_PAD_SEC * sfreq)

buffer_sec  = max(3.0, WIN_SEC + 1.0) + FILTER_PAD_SEC
buffer_len  = int(buffer_sec * sfreq)
window_len  = int(WIN_SEC * sfreq)
drop_warmup = int(DROP_START * sfreq)   # samples to skip at trial start
batch       = np.zeros((n_channels, buffer_len), dtype=np.float64)

samples_seen      = 0
last_predict_time = time.time()

# ── Accuracy tracking ─────────────────────────────────────────────────────────

current_target  = None   # frequency being cued this trial
trial_idx       = 0
trial_correct   = 0
trial_total     = 0
total_correct   = 0
total_total     = 0
per_freq        = defaultdict(lambda: [0, 0])  # freq → [correct, total]
session_done    = False

print('[MODEL] Running — waiting for trial cue markers from GUI...\n')

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:

    # ── Pull EEG chunk ────────────────────────────────────────────────────────
    chunk, _ = eeg.pull_chunk(timeout=0.0)
    if chunk:
        new_data    = np.asarray(chunk, dtype=np.float64).T
        new_data    = new_data[:n_channels, :]

        if new_data.shape[0] != n_channels:
            print(f'[MODEL] WARNING: channel mismatch — got {new_data.shape[0]}, expected {n_channels}. Skipping chunk.')
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

    # ── Pull marker (non-blocking) ────────────────────────────────────────────
    m_sample, _ = markers.pull_sample(timeout=0.0)
    if m_sample:
        m = float(m_sample[0])

        if 100 < m < 200:
            # New trial started: record target frequency
            new_target = m - MARKER_TARGET_BASE
            if current_target is not None and trial_total > 0:
                # Print result of previous trial before switching
                acc = 100 * trial_correct / trial_total
                print(f'[MODEL] ── Trial {trial_idx} result: '
                      f'{trial_correct}/{trial_total} = {acc:.0f}%  '
                      f'(target {current_target} Hz)\n')
            current_target = new_target
            trial_idx     += 1
            trial_correct  = 0
            trial_total    = 0
            samples_seen   = 0
            batch[:] = 0.0   # clear stale pre-trial data from buffer
            print(f'[MODEL] ── Trial {trial_idx} started  →  target: {current_target} Hz')

        elif m == MARKER_STIM_STOP and current_target is not None and trial_total > 0:
            acc = 100 * trial_correct / trial_total
            print(f'[MODEL] ── Trial {trial_idx} result: '
                  f'{trial_correct}/{trial_total} = {acc:.0f}%  '
                  f'(target {current_target} Hz)\n')

        elif m == MARKER_SESSION_END:
            # ── Print final summary ───────────────────────────────────────────
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

    # ── Predict (on stride boundary) ─────────────────────────────────────────
    now = time.time()
    if now - last_predict_time >= STRIDE_SEC:
        last_predict_time = now

        if samples_seen >= (window_len + drop_warmup):
            try:
                # Pull padded chunk so filters settle before the crop window
                padded_len = window_len + filter_pad
                padded     = batch[:, -padded_len:]
                padded, _  = preprocess(padded, sfreq=sfreq, line_freq=LINE_FREQ,
                                        bandpass=BANDPASS, resample_hz=RESAMPLE_HZ)
                # After resampling the pad may have changed length — crop from right
                crop = int(WIN_SEC * RESAMPLE_HZ)
                epoch = padded[:, -crop:]

                # Amplitude rejection — matches offline windowize (amp_threshold=100.0)
                ptp = epoch.max(axis=-1) - epoch.min(axis=-1)
                if np.any(ptp > 100.0):
                    print(f'[MODEL] Epoch rejected — artifact detected (max ptp={ptp.max():.1f})')
                    continue

                predicted_class, _ = model.predict(epoch)
                predicted_freq     = freq_map[predicted_class]

                pred_outlet.push_sample([float(predicted_freq)])

                # ── Score against current trial target ────────────────────────
                if current_target is not None and not session_done:
                    correct = (predicted_freq == current_target)
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

            except Exception:
                print('[MODEL] ERROR during prediction — skipping this window:')
                traceback.print_exc()

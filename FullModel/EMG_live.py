#!/usr/bin/env python3
"""
EMG_live.py  –  Live EMG inference → LSL outlet for piano navigation
Outputs to LSL stream "EMGControl" with a single float32 channel:
   -1.0  left-hand clench  (navigate left / previous note)
    0.0  no clench
   +1.0  right-hand clench (navigate right / next note)
"""

import collections
import numpy as np
import pandas as pd
import time
import signal
import threading

from scipy.signal import butter, filtfilt, iirnotch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, resolve_streams

# ── Stream config ─────────────────────────────────────────────────────────────
# Name of the EEG headset stream to EXCLUDE when searching for the EMG board.
# Both Unicorn and OpenBCI/BrainFlow broadcast type='EEG' — we must not
# accidentally connect the EMG classifier to the EEG headset.
EEG_HEADSET_NAME = 'Unicorn'

# ── Quit flag ─────────────────────────────────────────────────────────────────
quit_flag = threading.Event()

def handle_signal(signum, frame):
    print('\n[emglive] stop signal received')
    quit_flag.set()

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT,  handle_signal)

# ── Train model on startup ────────────────────────────────────────────────────

print('[emglive] loading training data and fitting model...')

emg_1_labels = pd.read_csv('../EMG_Data/EMG_1.csv')
emg_1        = pd.read_csv(
    '../EMG_Data/BrainFlow-RAW_2026-02-25_21-01-56_0.csv',
    header=None, sep=r'[,\t\s]+', engine='python'
)

timestamp = emg_1.iloc[:, -2]
signal_raw = emg_1.iloc[:, 1]

emg_df = pd.DataFrame({'timestamp': timestamp, 'signal': signal_raw})
emg_df['label'] = 'relax'

for i in range(len(emg_1_labels) - 1):
    start_t = emg_1_labels.loc[i,   'timestamp_unix']
    end_t   = emg_1_labels.loc[i+1, 'timestamp_unix']
    lab     = emg_1_labels.loc[i,   'label']
    mask    = (emg_df['timestamp'] >= start_t) & (emg_df['timestamp'] < end_t)
    emg_df.loc[mask, 'label'] = lab

def bandpass(sig, low=20, high=450, fs=1000, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)

emg_df['filtered'] = bandpass(emg_df['signal'].values)

window, step = 200, 50
X, y = [], []

for s in range(0, len(emg_df) - window, step):
    seg = emg_df.iloc[s:s + window]
    sig = seg['filtered'].values
    X.append([
        np.mean(np.abs(sig)),
        np.std(sig),
        np.max(sig) - np.min(sig),
        np.sqrt(np.mean(sig ** 2)),
        np.sum(np.abs(np.diff(sig)))
    ])
    y.append(seg['label'].mode()[0])

X = np.array(X)
y = np.where(np.array(y) == 'clench', 1, 0)

model = SVC(kernel='rbf', class_weight='balanced', probability=True)
model.fit(X, y)
print('[emglive] model trained.')

# ── LSL outlet ────────────────────────────────────────────────────────────────

out_info = StreamInfo(
    name='EMGControl',
    type='Markers',
    channel_count=1,
    nominal_srate=0,          # irregular / event-driven
    channel_format='float32',
    source_id='emg_control_01',
)
outlet = StreamOutlet(out_info)
print('[emglive] LSL outlet "EMGControl" ready.')

# ── Connect to EEG/EMG LSL inlet ──────────────────────────────────────────────

print('[emglive] searching for EMG/OpenBCI stream (excluding EEG headset)...')

# Log all visible streams so we know what's on the network.
def _log_streams():
    found = resolve_streams(wait_time=1.0)
    print('[emglive] ── Visible LSL streams ──────────────────────────────')
    for s in found:
        print(f'[emglive]   name={s.name()!r:25s} type={s.type()!r:12s} '
              f'ch={s.channel_count():3d}  fs={s.nominal_srate():.0f} Hz')
    print('[emglive] ────────────────────────────────────────────────────')

_log_streams()

chosen = None
while chosen is None and not quit_flag.is_set():
    # Prefer BrainFlow/OpenBCI by name so we never accidentally grab the EEG headset.
    candidates = resolve_byprop('name', 'BrainFlow', timeout=2.0)
    if not candidates:
        all_eeg = resolve_byprop('type', 'EEG', timeout=2.0) or []
        candidates = [s for s in all_eeg if s.name() != EEG_HEADSET_NAME]
    if candidates:
        chosen = candidates[0]
    else:
        print('[emglive] no EMG/OpenBCI stream found, retrying...')
        time.sleep(0.5)

if quit_flag.is_set():
    raise SystemExit(0)

inlet = StreamInlet(chosen)
print(f'[emglive] connected to: name={chosen.name()!r}  '
      f'ch={chosen.channel_count()}  fs={chosen.nominal_srate():.0f} Hz')

# ── Filter helpers ────────────────────────────────────────────────────────────

LIVE_FS = chosen.nominal_srate() or 250  # actual sample rate from the OpenBCI stream

def highpass(x, cutoff=20):
    b, a = butter(4, cutoff / (LIVE_FS / 2), btype='highpass')
    return filtfilt(b, a, x)

def notch(x, f0=60):
    b, a = iirnotch(f0 / (LIVE_FS / 2), 30)
    return filtfilt(b, a, x)

def extract_features(sig):
    return np.array([[
        np.mean(np.abs(sig)),
        np.std(sig),
        np.max(sig) - np.min(sig),
        np.sqrt(np.mean(sig ** 2)),
        np.sum(np.abs(np.diff(sig)))
    ]])

# ── Live inference loop ───────────────────────────────────────────────────────

chunk_size  = int(LIVE_FS * 0.5)
buffer_ch1  = collections.deque()
buffer_ch2  = collections.deque()
last_output = 0          # debounce: only push when value changes

print('[emglive] streaming — waiting for clenches...')

while not quit_flag.is_set():
    samples, _ = inlet.pull_chunk(max_samples=50)
    if not samples:
        time.sleep(0.005)
    if samples:
        arr = np.array(samples)
        buffer_ch1.extend(arr[:, 0])
        buffer_ch2.extend(arr[:, 1])

    if len(buffer_ch1) >= chunk_size and len(buffer_ch2) >= chunk_size:
        # Channel 1
        c1 = np.array([buffer_ch1.popleft() for _ in range(chunk_size)])
        c1 = notch(highpass(c1))
        p1 = model.predict(extract_features(c1))[0]

        # Channel 2
        c2 = np.array([buffer_ch2.popleft() for _ in range(chunk_size)])
        c2 = notch(highpass(c2))
        p2 = model.predict(extract_features(c2))[0]

        output = int(p2) - int(p1)   # -1, 0, or +1

        # Only push non-zero events (or when returning to 0 after an event)
        if output != 0 or last_output != 0:
            outlet.push_sample([float(output)])
            print(f'[emglive] pushed {output:+d}  (ch1={p1}, ch2={p2})')
        last_output = output

print('[emglive] exited cleanly.')

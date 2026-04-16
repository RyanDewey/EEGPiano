import time
from collections import deque

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop

# ==============================================================
# LOAD TRAINING DATA
# ==============================================================

emg_1_labels = pd.read_csv('EMG_Data/EMG_1.csv')
emg_1 = pd.read_csv(
    "EMG_Data/BrainFlow-RAW_2026-02-25_21-01-56_0.csv",
    header=None, sep=r'[,\t\s]+', engine='python'
)

# ==============================================================
# PREPROCESS TRAINING DATA
# ==============================================================

timestamp = emg_1.iloc[:, -2]
signal    = emg_1.iloc[:, 1]

emg_df = pd.DataFrame({"timestamp": timestamp, "signal": signal})
emg_df["label"] = "relax"

labels = emg_1_labels
for i in range(len(labels) - 1):
    start = labels.loc[i,   "timestamp_unix"]
    end   = labels.loc[i+1, "timestamp_unix"]
    lab   = labels.loc[i,   "label"]
    mask  = (emg_df["timestamp"] >= start) & (emg_df["timestamp"] < end)
    emg_df.loc[mask, "label"] = lab

def bandpass(sig, low=20, high=450, fs=1000, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)

emg_df["filtered"] = bandpass(emg_df["signal"].values)

# ==============================================================
# FEATURE EXTRACTION  (window=200 samples — must match live)
# ==============================================================

WINDOW = 200   # samples — keep in sync with chunk_size below
STEP   = 50

def extract_features(seg):
    """Compute the 5 time-domain EMG features used for training and inference."""
    return [
        np.mean(np.abs(seg)),
        np.std(seg),
        np.max(seg) - np.min(seg),
        np.sqrt(np.mean(seg ** 2)),
        np.sum(np.abs(np.diff(seg))),
    ]

X, y = [], []
for start in range(0, len(emg_df) - WINDOW, STEP):
    seg     = emg_df.iloc[start:start + WINDOW]["filtered"].values
    label   = emg_df.iloc[start:start + WINDOW]["label"].mode()[0]
    X.append(extract_features(seg))
    y.append(label)

X = np.array(X)
y = np.where(np.array(y) == "clench", 1, 0)

# ==============================================================
# TRAIN MODEL
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='rbf', class_weight='balanced', probability=True)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, pred))
print("\nClassification Report")
print(classification_report(y_test, pred))

# ==============================================================
# CONNECT TO LSL STREAM
# ==============================================================

print("Searching for EEG stream...")
streams = []
while not streams:
    streams = resolve_byprop('type', 'EEG', timeout=2.0)
    if not streams:
        print("No stream found, retrying...")
        time.sleep(0.5)

inlet = StreamInlet(streams[0])
info  = inlet.info()

# Read actual sampling rate from stream; fall back to 250 if unset
fs = float(info.nominal_srate())
if fs <= 0:
    fs = 250.0
    print(f"Warning: stream reported srate=0, defaulting to {fs} Hz")
else:
    print(f"Connected to: {info.name()}  |  fs = {fs} Hz")

# ==============================================================
# LIVE FILTER FUNCTIONS
# Note: training used bandpass(20–450 Hz) at 1000 Hz.
# At 250 Hz the Nyquist limit is 125 Hz, so we use a highpass
# + notch chain instead — the low-frequency and powerline
# artefact removal is equivalent for EMG purposes.
# ==============================================================

def highpass(x, cutoff, fs):
    b, a = butter(4, cutoff / (fs / 2), btype='highpass')
    return filtfilt(b, a, x)

def notch(x, f0, fs):
    b, a = iirnotch(f0 / (fs / 2), Q=30)
    return filtfilt(b, a, x)

# ==============================================================
# LIVE INFERENCE LOOP
# chunk_size MUST equal WINDOW so feature distributions match training.
# ==============================================================

# Probability threshold for a clench prediction to be accepted.
# 0.5 = default SVM boundary (no bias).
# Raise RIGHT_THRESHOLD toward 1.0 to require higher confidence
# before the right hand is counted as clenching (fewer false positives).
LEFT_THRESHOLD  = 0.5   # probability cutoff for ch1 (left hand)
RIGHT_THRESHOLD = 0.5   # probability cutoff for ch2 (right hand)

chunk_size = WINDOW   # 200 samples per inference step
buffer_ch1 = deque()
buffer_ch2 = deque()

# ── LSL outlet for EMG predictions ───────────────────────────
emg_pred_info = StreamInfo(
    name='EMGPred',
    type='Predictions',
    channel_count=1,
    nominal_srate=0,
    channel_format='float32',
    source_id='emg_pred',
)
emg_outlet = StreamOutlet(emg_pred_info)
print('[LSL] EMG prediction outlet "EMGPred" ready.')

print(f"Listening for EMG (chunk = {chunk_size} samples = {chunk_size/fs:.2f}s)...\n"
      f"Thresholds — left: {LEFT_THRESHOLD}  right: {RIGHT_THRESHOLD}\n"
      "Output: -1 = left clench, 0 = rest/both, +1 = right clench\n"
      "Press Ctrl+C to stop.\n")

while True:
    samples, _ = inlet.pull_chunk(max_samples=50)

    if samples:
        arr = np.array(samples)
        buffer_ch1.extend(arr[:, 0])   # ch0 → left-hand EMG
        buffer_ch2.extend(arr[:, 1])   # ch1 → right-hand EMG
    else:
        time.sleep(0.01)   # avoid busy-spinning when stream is idle
        continue

    if len(buffer_ch1) < chunk_size:
        continue

    # Consume exactly chunk_size samples from the front of each buffer
    chunk_ch1 = np.array([buffer_ch1.popleft() for _ in range(chunk_size)])
    chunk_ch2 = np.array([buffer_ch2.popleft() for _ in range(chunk_size)])

    # Filter
    chunk_ch1 = highpass(chunk_ch1, 20, fs)
    chunk_ch1 = notch(chunk_ch1, 60, fs)

    chunk_ch2 = highpass(chunk_ch2, 20, fs)
    chunk_ch2 = notch(chunk_ch2, 60, fs)

    # predict_proba returns [p_rest, p_clench]; index 1 is clench probability
    prob_ch1 = model.predict_proba(np.array([extract_features(chunk_ch1)]))[0][1]
    prob_ch2 = model.predict_proba(np.array([extract_features(chunk_ch2)]))[0][1]

    pred_ch1 = int(prob_ch1 >= LEFT_THRESHOLD)
    pred_ch2 = int(prob_ch2 >= RIGHT_THRESHOLD)

    # -1 = left clench, +1 = right clench, 0 = rest or both
    output = pred_ch2 - pred_ch1
    emg_outlet.push_sample([float(output)])
    print(output)

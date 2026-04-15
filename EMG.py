import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import pylsl

print("Got here")



# Import Data

emg_1_labels = pd.read_csv('EMG_Data/EMG_1.csv')
emg_8_labels = pd.read_csv('EMG_Data/EMG_8.csv')

emg_1 = pd.read_csv("EMG_Data/BrainFlow-RAW_2026-02-25_21-01-56_0.csv", header=None, sep=r'[,\t\s]+', engine='python')
emg_8 = pd.read_csv("EMG_Data/BrainFlow-RAW_2026-04-01_18-25-27_3.csv", header=None, sep=r'[,\t\s]+', engine='python')


# Model

# ==========================
# LOAD FILES
# ==========================

emg = emg_1
labels = emg_1_labels

# ==========================
# SELECT COLUMNS BY INDEX
# ==========================

# BrainFlow layout: timestamp usually second-to-last column
timestamp = emg.iloc[:, -2]

# first EXG channel
signal = emg.iloc[:, 1]

emg_df = pd.DataFrame({
    "timestamp": timestamp,
    "signal": signal
})

# ==========================
# ALIGN LABELS
# ==========================

emg_df["label"] = "relax"

for i in range(len(labels)-1):

    start = labels.loc[i, "timestamp_unix"]
    end = labels.loc[i+1, "timestamp_unix"]
    lab = labels.loc[i, "label"]

    mask = (emg_df["timestamp"] >= start) & (emg_df["timestamp"] < end)
    emg_df.loc[mask, "label"] = lab

# ==========================
# FILTER SIGNAL
# ==========================

def bandpass(signal, low=20, high=450, fs=1000, order=4):

    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

emg_df["filtered"] = bandpass(emg_df["signal"].values)


# ==========================
# FEATURE EXTRACTION
# ==========================

window = 200
step = 50

X = []
y = []

for start in range(0, len(emg_df)-window, step):

    segment = emg_df.iloc[start:start+window]
    sig = segment["filtered"].values

    features = [
        np.mean(np.abs(sig)),
        np.std(sig),
        np.max(sig)-np.min(sig),
        np.sqrt(np.mean(sig**2)),
        np.sum(np.abs(np.diff(sig)))
    ]

    label = segment["label"].mode()[0]

    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

y = np.where(y == "clench", 1, 0)

# ==========================
# TRAIN MODEL
# ==========================

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



# Live Stuff

from pylsl import StreamInlet, resolve_streams
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import joblib
import time

# ==============================
# CONNECT TO STREAM
# ==============================

from pylsl import resolve_byprop, StreamInlet
import time

streams = []
print("Searching for EEG stream...")

while not streams:
    streams = resolve_byprop('type', 'EEG', timeout=2.0)
    if not streams:
        print("No stream found, retrying...")
        time.sleep(0.5)  # small gap between attempts

print(f"✅ Connected to: {streams[0].name()}")
inlet = StreamInlet(streams[0])

# ==============================
# FILTER FUNCTIONS
# ==============================

fs = 250  # update if different

def highpass(x, cutoff, fs):
    b, a = butter(4, cutoff/(fs/2), btype='highpass')
    return filtfilt(b, a, x)

def notch(x, f0, fs):
    b, a = iirnotch(f0/(fs/2), 30)
    return filtfilt(b, a, x)

# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(sig):
    return np.array([[
        np.mean(np.abs(sig)),
        np.std(sig),
        np.max(sig) - np.min(sig),
        np.sqrt(np.mean(sig**2)),
        np.sum(np.abs(np.diff(sig)))
    ]])

# ==============================
# LIVE INFERENCE LOOP
# ==============================

chunk_size = int(fs * 0.5)  # 1 second of samples
buffer_ch1 = []
buffer_ch2 = []

print("Listening... press Stop (■) to stop\n")

while True:
    samples, _ = inlet.pull_chunk(max_samples=50)
    if samples:
        samples_array = np.array(samples)
        buffer_ch1.extend(samples_array[:, 0])  # channel 1
        buffer_ch2.extend(samples_array[:, 1])  # channel 2

    # once we have 1 second of data, run inference on both channels
    if len(buffer_ch1) >= chunk_size and len(buffer_ch2) >= chunk_size:
        # Process Channel 1
        chunk_ch1 = np.array(buffer_ch1[:chunk_size])
        buffer_ch1 = buffer_ch1[chunk_size:]

        # Filter channel 1
        chunk_ch1 = highpass(chunk_ch1, 20, fs)
        chunk_ch1 = notch(chunk_ch1, 60, fs)

        # Extract features and predict for channel 1
        features_ch1 = extract_features(chunk_ch1)
        prediction_ch1 = model.predict(features_ch1)[0]

        # Process Channel 2
        chunk_ch2 = np.array(buffer_ch2[:chunk_size])
        buffer_ch2 = buffer_ch2[chunk_size:]

        # Filter channel 2
        chunk_ch2 = highpass(chunk_ch2, 20, fs)
        chunk_ch2 = notch(chunk_ch2, 60, fs)

        # Extract features and predict for channel 2
        features_ch2 = extract_features(chunk_ch2)
        prediction_ch2 = model.predict(features_ch2)[0]

        # Print results
        output = 0
        if prediction_ch1 == 1:
            output -= 1
        if prediction_ch2 == 1:
            output += 1

        print(output)
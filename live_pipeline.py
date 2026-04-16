# This is the main file that is designed to handle the live EEG data
# Works with fbtrca_model.py file
# From Keira

# Necessary Imports

import time
import pickle
import numpy as np
from scipy import signal
from pylsl import StreamInlet, resolve_streams, resolve_byprop, StreamInfo, StreamOutlet
from fbtrca_model import FBTRCA, class_to_freq_map # fbtcra_model derived from Ryan's code should be in the same folder


# Finds all streams on PC for debugging purposes
streams = resolve_streams()
for s in streams:
    print(s.name(), s.type())

print("Looking for LSL streams...")

# Optional marker stream: try once
marker_stream = [s for s in streams if s.type() == "Markers"]
if not marker_stream:
    print("No marker stream found.")
    markers = None
else:
    markers = StreamInlet(marker_stream[0])
    print("Connected to marker stream!")

# ── LSL prediction outlet ─────────────────────────────────────────────────────
pred_info = StreamInfo(
    name='SSVEPPredictions',
    type='Predictions',
    channel_count=1,
    nominal_srate=0,          # irregular / event-driven
    channel_format='float32',
    source_id='ssvep_pred',
)
pred_outlet = StreamOutlet(pred_info)
print('[LSL] Prediction outlet "SSVEPPredictions" ready.')

# Required EEG stream: keep waiting until it appears
print("Waiting for EEG stream to resolve...")
while True:
    eeg_stream = resolve_byprop("type", "EEG", timeout=2)
    if eeg_stream:
        eeg = StreamInlet(eeg_stream[0])
        print("Connected to EEG stream!")
        break

    print("No EEG stream found yet. Retrying in 1 second...")
    time.sleep(1)

# Loading the FBTCRA model with pickle

with open("fbtrca_model.pkl", "rb") as f: # "rb" is read binary
    state = pickle.load(f)

if not state:
    print("Could not find fbtrca_model.pkl in directory.")
else:
    print("Loaded FBTRCA model.")
    

# Loading the trained model

model = state["model"]

# Loading the hyperparameters

freqs = state["freqs"]
freq_to_class = state["freq_to_class"]
sfreq = state["sfreq"]
WIN_SEC = state["win_sec"]
STRIDE_SEC = state["stride_sec"]
DROP_START_SEC = state["drop_start_sec"]
DROP_END_SEC = state["drop_end_sec"]
LINE_FREQ = state["line_freq"]
BANDPASS = state["bandpass"]
RESAMPLE_HZ = state["resample_hz"]
FB_BANDS = state["fb_bands"]
WEIGHT_EXP = state["weight_exp"]

# Making sure the FBTRCA model is trained on an 8 channel EEG

n_channels = state.get("n_channels", None)
if n_channels is None:
    _, first_template = model.models[0][0]
    n_channels = first_template.shape[0]

print(f"Expected channels: {n_channels}")

print("Number of classes:", model.K)

# Create the dictionary to map classes to frequencies

freq_map = class_to_freq_map(freq_to_class)

# Preprocessing helper function
# Takes the data in chunks (epoch) and uses hyperparameters defined in fbtrca_model to preprocess data
# Based on Ryan's code

def preprocess(epoch, sfreq, line_freq, bandpass, resample_hz):

    # Epoch is just a chunk of EEG data
    # It needs to be converted into a numpy array

    epoch = np.array(epoch, dtype=np.float64)

    # Remove average signal from epochs to try to extract key features

    epoch = epoch - epoch.mean(axis=0, keepdims=True)

    # Setting constants

    nyq = sfreq/2 # Nyquist frequency
    nf = line_freq #Notch frequency

    # Notch filter

    while nf < nyq:
        b, a = signal.iirnotch(w0=nf, Q=30, fs=sfreq) # Create notch filter
        epoch = signal.filtfilt(b, a, epoch, axis=-1) # Apply notch filter
        nf += line_freq

    # Bandpass filter

    sos = signal.butter(
        4, # Filter order
        [bandpass[0], bandpass[1]], # Lower and upper cutoff frequencies
        btype="bandpass",
        fs=sfreq,
        output="sos"
    )
    epoch=signal.sosfiltfilt(sos, epoch, axis=-1) # Apply bandpass filter

    # Changing the sampling rate if needed

    if abs(sfreq - resample_hz) > 1e-9: # Determine if resampling is needed
        n_out = int(round(epoch.shape[1] * (resample_hz / sfreq))) # Find difference in sampling rate
        epoch = signal.resample(epoch, n_out, axis=-1) # Perform resampling
        sfreq = float(resample_hz) # Update sampling rate
    
    # Returns final results

    return epoch, sfreq

# Setting constants for the batches

# Buffer must be at least as long as the model's analysis window.
# If buffer < WIN_SEC, numpy silently returns fewer samples than the model
# expects, which corrupts the spatial filter correlation without any error.
buffer = max(WIN_SEC + 1, 4)  # always at least WIN_SEC + 1 second of headroom
buffer_len = int(buffer * sfreq)
window_len = int(WIN_SEC * sfreq)

assert buffer_len >= window_len, (
    f"Buffer ({buffer}s = {buffer_len} samples) must be >= "
    f"WIN_SEC ({WIN_SEC}s = {window_len} samples). Increase buffer."
)

# Predict every STRIDE_SAMPLES new samples instead of every STRIDE_SEC wall
# seconds. LSL delivers data in irregular bursts, so wall-clock timing fires
# before/after the expected sample count — sample-count stride is exact.
STRIDE_SAMPLES = int(round(STRIDE_SEC * sfreq))

# Artifact rejection: after z-scoring, skip epochs where any channel has
# peak-to-peak > AMP_THRESHOLD std-devs. Matches offline windowize() behaviour.
AMP_THRESHOLD = 10.0

# Only push a prediction when the margin between the top-2 fused scores
# exceeds this fraction. Prevents noisy, low-confidence predictions from
# triggering wrong piano notes.
CONFIDENCE_THRESHOLD = 0.10   # tune based on live performance

# Creates an array of shape n_channels x buffer_len that's all zeroes

batch = np.zeros((n_channels, buffer_len), dtype=np.float64)

samples_seen = 0
last_predict_sample = 0

# Now, time for the rolling windows

while True:
    
    # Pulling new EEG samples

    chunk, timestamps = eeg.pull_chunk(timeout=0.0) # Shape of chunk is (n_samples, n_channels)

    if chunk:
        new_data = np.asarray(chunk, dtype=np.float64).T # Transpose turns it into (n_samples, n_channels)
        new_data = new_data[:n_channels, :]
    else:
        continue

    # Verify channel count for debugging purposes

    if new_data.shape[0] != n_channels:
        raise RuntimeError(f"Channel mismatch: got {new_data.shape[0]}, expected {n_channels}")
        continue

    new_samples = new_data.shape[1]

    samples_seen += new_samples

    # In the case that the new batch is larger than or equal to the max length of the buffer, just keep the newst part of the batch

    if new_samples >= buffer_len:
        batch = new_data[:, -buffer_len:]

    # If the batch is less than the size of the max buffer

    else:
        batch[:, :-new_samples] = batch[:, new_samples:] # Shifts current contents of the buffer to the left
        batch[:, -new_samples:] = new_data # Fills empty space in buffer with new data

    # Predict every STRIDE_SAMPLES new samples (sample-count stride, not wall clock)

    if samples_seen - last_predict_sample < STRIDE_SAMPLES:
        continue

    last_predict_sample = samples_seen

    # Don't make a new prediction until enough samples exist

    if samples_seen < window_len:
        continue

    # Extract the newest window

    epoch = batch[:, -window_len:]

    # Preprocess extracted window (average ref, notch, bandpass, resample)

    epoch, epoch_sfreq = preprocess(
        epoch,
        sfreq=sfreq,
        line_freq=LINE_FREQ,
        bandpass=BANDPASS,
        resample_hz=RESAMPLE_HZ,
    )

    # Remove linear trend per channel to match offline training.
    # The FBTRCA templates were built from detrended epochs in the notebook's
    # windowize() function — skipping this step causes a systematic mismatch
    # between the template and the live epoch that degrades spatial filter scores.

    epoch = signal.detrend(epoch, axis=-1, type="linear")

    # Per-channel z-score to match offline training.
    # Templates were built from z-scored epochs. Without this, raw amplitude
    # differences dominate the correlation, causing the spatial filter (w) —
    # which was optimised for z-scored data — to weight channels incorrectly.
    # Higher SSVEP frequencies are most affected because their raw amplitudes
    # are naturally smaller (~1/f), so the mismatch is proportionally larger.

    std = epoch.std(axis=-1, keepdims=True) + 1e-12
    epoch = epoch / std

    # Artifact rejection: skip epoch if any channel has a large peak-to-peak.
    # Matches the amp_threshold rejection in the notebook's windowize() function.
    # Blinks, jaw clenches, and eye movements produce large bursts that score
    # randomly and push wrong predictions through to the piano.

    ptp = epoch.max(axis=-1) - epoch.min(axis=-1)
    if np.any(ptp > AMP_THRESHOLD):
        print("[Pipeline] Epoch rejected — artifact detected, skipping prediction")
        continue

    # Run the model on the epoch

    predicted_class, fused_scores = model.predict(epoch)
    predicted_frequency = freq_map[predicted_class]

    # Only push if the classifier is confident.
    # When fused scores are near-equal the model is uncertain — forcing a
    # prediction at that point triggers the wrong piano note with high probability.

    sorted_scores = np.sort(fused_scores)[::-1]
    margin = (sorted_scores[0] - sorted_scores[1]) / (sorted_scores[0] + sorted_scores[1] + 1e-12)

    if margin < CONFIDENCE_THRESHOLD:
        print(f"[Pipeline] Low confidence ({margin:.3f} < {CONFIDENCE_THRESHOLD}) — skipping prediction")
        continue

    print(f"Predicted class: {predicted_class}, frequency: {predicted_frequency}, confidence margin: {margin:.3f}")

    # ── Push prediction over LSL ──────────────────────────────────────
    pred_outlet.push_sample([float(predicted_frequency)])
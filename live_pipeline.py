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

buffer = 3 # Seconds FIDDLE WITH THIS FOR TESTING PURPOSES!!!!!!!
buffer_len = int(buffer * sfreq)
window_len = int(WIN_SEC * sfreq)

# Creates an array of shape 8 x buffer_len that's all zeroes

batch = np.zeros((n_channels, buffer_len), dtype=np.float64) 

samples_seen = 0
next_predict_sample = 0   # sample-count threshold for next prediction (replaces wall-clock stride)
in_stimulus = False       # True only between a start marker (252.0) and end marker (253.0)
stride_len = int(STRIDE_SEC * sfreq)

START_MARKER = "252.0"
END_MARKER   = "253.0"

# Now, time for the rolling windows

while True:

    # Poll marker stream to update stimulus-gating state
    if markers is not None:
        while True:
            marker_sample, _ = markers.pull_sample(timeout=0.0)
            if marker_sample is None:
                break
            marker_val = str(marker_sample[0])
            if marker_val == START_MARKER:
                in_stimulus = True
            elif marker_val == END_MARKER:
                in_stimulus = False

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

    # Stride gate: fire a prediction every stride_len samples (sample-count based,
    # not wall-clock, so it stays aligned regardless of CPU load)
    if samples_seen >= next_predict_sample + stride_len:
        next_predict_sample = samples_seen

        # Don't predict until we have a full window
        if samples_seen < window_len:
            continue

        # Stimulus gating: skip if we're not inside a marked trial.
        # When no marker stream is present, always predict (in_stimulus starts False
        # but markers is None, so we skip gating entirely).
        if markers is not None and not in_stimulus:
            continue

        # Preprocess the FULL buffer first (reduces filter edge effects vs
        # preprocessing only the short extracted window)
        buf_proc, epoch_sfreq = preprocess(
            batch,
            sfreq=sfreq,
            line_freq=LINE_FREQ,
            bandpass=BANDPASS,
            resample_hz=RESAMPLE_HZ,
        )

        # After resampling the buffer may have changed length; recompute window_len
        resampled_window_len = int(round(WIN_SEC * epoch_sfreq))

        # Extract the newest window from the preprocessed buffer
        epoch = buf_proc[:, -resampled_window_len:]

        # Remove linear trend per channel (matches offline windowize)
        epoch = signal.detrend(epoch, axis=-1, type="linear")

        # Per-channel normalization (matches offline windowize)
        std = epoch.std(axis=-1, keepdims=True) + 1e-12
        epoch = epoch / std

        # Artifact rejection: discard noisy epochs (matches offline windowize)
        ptp = epoch.max(axis=-1) - epoch.min(axis=-1)
        if np.any(ptp > 100.0):
            print("Epoch rejected (artifact — peak-to-peak > 100)")
            continue

        # Run the model on the epoch
        predicted_class, fused_scores = model.predict(epoch)
        predicted_frequency = freq_map[predicted_class]
        print(f"Predicted class: {predicted_class}, frequency: {predicted_frequency}")

        # ── Push prediction over LSL ──────────────────────────────────────
        pred_outlet.push_sample([float(predicted_frequency)])
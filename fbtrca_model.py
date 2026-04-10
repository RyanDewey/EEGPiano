# This file contains the FBTRCA helper functions
# From Ryan
# https://colab.research.google.com/drive/1K30xQjg4hGGpdrbCor7qaIwLM5GHa7u8?usp=sharing

# Necessary imports

import numpy as np
from scipy import signal

# Returns frequency from class

def class_to_freq_map(freq_to_class):
    return {v: k for k, v in freq_to_class.items()}

# Computes correlation between two signals

def corrcoef_1d(a, b, eps=1e-12):
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float((a @ b) / denom)

# Creates a bandpass filter

def bandpass_sos(low, high, fs, order=4):
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

# Applies filter (forward and backward)

def apply_sos(X, sos):
    return signal.sosfiltfilt(sos, X, axis=-1)

# Finds best combination of channels to predict SSVEP frequencies

def trca_fit(X_class, reg=1e-6):
    """
    X_class: (n_trials, n_ch, n_times)
    """
    X_class = np.asarray(X_class, dtype=np.float64)
    n_trials, n_ch, n_times = X_class.shape

    template = X_class.mean(axis=0)
    Xc = X_class - X_class.mean(axis=2, keepdims=True)

    Q = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(n_trials):
        Xi = Xc[i]
        Q += Xi @ Xi.T

    Z = Xc.sum(axis=0)
    S = Z @ Z.T - Q

    Q = Q + reg * np.eye(n_ch)
    A = np.linalg.solve(Q, S)
    eigvals, eigvecs = np.linalg.eig(A) # Linear algebra :(
    w = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    w = w / (np.linalg.norm(w) + 1e-12)
    return w, template

# Compares EEG chunk to class template

def trca_score(epoch, w, template):
    y = w @ epoch
    yt = w @ template
    return corrcoef_1d(y, yt)

# Actual class that combines all the helper functions into the FBTRCA model

class FBTRCA:
    def __init__(self, sfreq, filter_bands, weight_exp=1.25, trca_reg=1e-6):
        self.sfreq = sfreq
        self.filter_bands = filter_bands
        self.weight_exp = weight_exp
        self.trca_reg = trca_reg
        self.sos = [bandpass_sos(lo, hi, sfreq) for lo, hi in filter_bands]
        self.band_weights = np.array([1/((i+1)**weight_exp) for i in range(len(filter_bands))], dtype=np.float64)
        self.models = None
        self.K = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.K = int(y.max()) + 1

        self.models = []  # [band][cls] -> (w, template)
        for bi, sos in enumerate(self.sos):
            Xb = apply_sos(X, sos)  # (n_trials,n_ch,n_times)
            band_models = []
            for k in range(self.K):
                Xk = Xb[y == k]
                if len(Xk) < 2:
                    raise ValueError(f"Class {k} has <2 trials; TRCA needs multiple trials/class.")
                w, templ = trca_fit(Xk, reg=self.trca_reg)
                band_models.append((w, templ))
            self.models.append(band_models)
        return self

    def predict_scores(self, epoch):
        epoch = np.asarray(epoch, dtype=np.float64)
        per_band = np.zeros((len(self.sos), self.K), dtype=np.float64)
        for bi, sos in enumerate(self.sos):
            eb = apply_sos(epoch, sos)  # (n_ch,n_times)
            for k in range(self.K):
                w, templ = self.models[bi][k]
                per_band[bi, k] = trca_score(eb, w, templ)
        fused = (self.band_weights[:, None] * (per_band ** 2)).sum(axis=0)
        return fused, per_band

    def predict(self, epoch):
        fused, _ = self.predict_scores(epoch)
        return int(np.argmax(fused)), fused
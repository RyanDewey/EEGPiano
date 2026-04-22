#!/usr/bin/env python3
"""
Train (or retrain) the FBTRCA model on all XDF calibration data in a directory.

Designed for 7.5 / 10 / 12 Hz SSVEP data collected with the Unicorn headset.

Usage:
    python train.py                          # uses ./newGuiData, saves fbtrca_model_751012.pkl
    python train.py --validate               # run leave-one-file-out accuracy check first
    python train.py --data_dir /path/to/xdf  # custom data directory
    python train.py --freqs 7.5 10.0 12.0   # restrict to specific frequencies
"""

import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
from scipy import signal

from fbtrca_model import FBTRCA

# ── Constants (must match live_pipeline.py) ───────────────────────────────────
LINE_FREQ        = 60.0
BANDPASS         = (0.5, 45.0)
RESAMPLE_HZ      = 250
WIN_SEC          = 2.0
STRIDE_SEC       = 1.0
DROP_START_SEC   = 1.0
DROP_END_SEC     = 0.5
FB_BANDS         = [(5,45),(12,45),(18,45),(24,45),(30,45)]
WEIGHT_EXP       = 1.25
EEG_CHANNEL_IDXS = list(range(8))
START_MARKERS    = ["252.0"]
END_MARKERS      = ["253.0"]
FREQ_REGEX       = r"(\d+(?:\.\d+)?)\s*hz"

try:
    import pyxdf
except ImportError:
    sys.exit("pyxdf is required: pip install pyxdf")


# ── XDF helpers ───────────────────────────────────────────────────────────────

def freq_from_name(path):
    m = re.search(FREQ_REGEX, os.path.basename(path).lower())
    if not m:
        raise ValueError(f"Can't parse freq from filename: {path}")
    return float(m.group(1))


def _safe_name(s): return s["info"].get("name", ["?"])[0]
def _safe_type(s): return s["info"].get("type", ["?"])[0]
def _ch_count(s):
    try:    return int(s["info"]["channel_count"][0])
    except: return 0


def pick_eeg_stream(streams):
    candidates = [
        s for s in streams
        if s.get("time_series") is not None
        and len(s.get("time_stamps", [])) > 0
        and np.asarray(s["time_series"]).ndim == 2
        and np.asarray(s["time_series"]).shape[0] > 0
        and np.asarray(s["time_series"]).shape[1] > 1
    ]
    if not candidates:
        raise ValueError("No EEG-like stream found.")
    return max(candidates, key=lambda s: len(s["time_series"]))


def pick_marker_stream(streams):
    candidates = [
        s for s in streams
        if ("marker" in _safe_name(s).lower() or
            "marker" in _safe_type(s).lower() or
            _ch_count(s) == 1)
        and len(s.get("time_series", [])) > 0
    ]
    if not candidates:
        raise ValueError("No marker stream found.")
    return max(candidates, key=lambda s: len(s["time_series"]))


def marker_values_to_strings(marker_stream):
    out = []
    for v in marker_stream["time_series"]:
        if isinstance(v, (list, tuple, np.ndarray)):
            out.append(str(v[0]) if len(v) else "")
        else:
            out.append(str(v))
    return np.array(out, dtype=object)


def build_trial_intervals(marker_labels, marker_times):
    intervals, active = [], None
    for label, t in zip(marker_labels, marker_times):
        if label in START_MARKERS and active is None:
            active = t
        elif label in END_MARKERS and active is not None:
            if t > active:
                intervals.append((active, t))
            active = None
    return intervals


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X, sfreq):
    X = X - X.mean(axis=0, keepdims=True)
    nyq, nf = sfreq / 2, LINE_FREQ
    while nf < nyq:
        b, a = signal.iirnotch(w0=nf, Q=30, fs=sfreq)
        X = signal.filtfilt(b, a, X, axis=-1)
        nf += LINE_FREQ
    sos = signal.butter(4, list(BANDPASS), btype="bandpass", fs=sfreq, output="sos")
    X = signal.sosfiltfilt(sos, X, axis=-1)
    if abs(sfreq - RESAMPLE_HZ) > 1e-9:
        n_out = int(round(X.shape[1] * (RESAMPLE_HZ / sfreq)))
        X = signal.resample(X, n_out, axis=-1)
        sfreq = float(RESAMPLE_HZ)
    return X, sfreq


def windowize(X, sfreq, amp_threshold=100.0):
    start  = int(round(DROP_START_SEC * sfreq))
    end    = X.shape[1] - int(round(DROP_END_SEC * sfreq))
    win    = int(round(WIN_SEC * sfreq))
    stride = int(round(STRIDE_SEC * sfreq))
    if end - start < win:
        raise ValueError("Not enough data after edge drops.")
    trials = []
    for s in range(start, end - win + 1, stride):
        epoch = X[:, s:s+win]
        epoch = signal.detrend(epoch, axis=-1, type="linear")
        std   = epoch.std(axis=-1, keepdims=True) + 1e-12
        epoch = epoch / std
        if np.any(epoch.max(axis=-1) - epoch.min(axis=-1) > amp_threshold):
            continue
        trials.append(epoch)
    if not trials:
        raise ValueError("All epochs rejected — check data quality.")
    return np.stack(trials, axis=0)


# ── File loading ──────────────────────────────────────────────────────────────

def load_file(path):
    streams, _ = pyxdf.load_xdf(path)
    eeg = pick_eeg_stream(streams)
    mkr = pick_marker_stream(streams)

    X = np.asarray(eeg["time_series"], dtype=np.float64)
    t = np.asarray(eeg["time_stamps"],  dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    X = X[:, EEG_CHANNEL_IDXS].T
    sfreq = 1.0 / np.median(np.diff(t))

    labels    = marker_values_to_strings(mkr)
    mk_times  = np.asarray(mkr["time_stamps"], dtype=np.float64)
    intervals = build_trial_intervals(labels, mk_times)
    if not intervals:
        raise ValueError("No trial intervals found (missing start/end markers).")

    pieces = []
    for t0, t1 in intervals:
        i0 = np.searchsorted(t, t0)
        i1 = np.searchsorted(t, t1, side="right")
        if i1 > i0:
            p, sf = preprocess(X[:, i0:i1], sfreq)
            pieces.append(p)
    if not pieces:
        raise ValueError("No data inside trial markers.")
    return np.concatenate(pieces, axis=1), sf


def load_dataset(paths, freq_to_class):
    Xs, ys, sfreq = [], [], None
    for p in paths:
        try:
            X, sf = load_file(p)
            windows = windowize(X, sf)
            cls = freq_to_class[freq_from_name(p)]
            ys.append(np.full(len(windows), cls, dtype=int))
            Xs.append(windows)
            sfreq = sf
            print(f"  OK   {os.path.basename(p):55s} {len(windows):3d} windows")
        except Exception as e:
            print(f"  SKIP {os.path.basename(p):55s} {e}")
    if not Xs:
        sys.exit("No valid files loaded — aborting.")
    return np.concatenate(Xs), np.concatenate(ys), sfreq


# ── LOFO cross-validation ─────────────────────────────────────────────────────

def lofo_validate(file_data, freq_to_class, freqs, sfreq):
    filter_bands = FB_BANDS
    valid = list(file_data.keys())
    results = []                                    # overall acc per fold
    per_class = {f: [] for f in freqs}             # freq → [acc per fold where that freq is held out]
    per_class_all = {f: {'correct': 0, 'total': 0} for f in freqs}  # pooled across all folds

    for held in valid:
        train_paths = [p for p in valid if p != held]
        counts = {}
        for p in train_paths:
            c = freq_to_class[freq_from_name(p)]
            counts[c] = counts.get(c, 0) + 1
        if any(counts.get(c, 0) == 0 for c in range(len(freqs))):
            continue

        Xs, ys = zip(*[file_data[p] for p in train_paths])
        X_tr, y_tr = np.concatenate(Xs), np.concatenate(ys)
        X_te, y_te = file_data[held]

        mdl   = FBTRCA(sfreq=sfreq, filter_bands=filter_bands,
                       weight_exp=WEIGHT_EXP, trca_reg=1e-6).fit(X_tr, y_tr)
        preds = np.array([mdl.predict(ep)[0] for ep in X_te])
        acc   = float((preds == y_te).mean())
        results.append(acc)

        # Accumulate per-class stats for this fold
        held_freq = freq_from_name(held)
        for cls_idx, freq in enumerate(freqs):
            mask = y_te == cls_idx
            if mask.any():
                cls_correct = int((preds[mask] == y_te[mask]).sum())
                cls_total   = int(mask.sum())
                per_class_all[freq]['correct'] += cls_correct
                per_class_all[freq]['total']   += cls_total
                per_class[freq].append(cls_correct / cls_total)

        status = "OK" if acc >= 0.5 else "BAD"
        print(f"  {os.path.basename(held):55s} {held_freq:5.1f} hz  "
              f"acc={acc*100:5.1f}%  {status}")

    if results:
        print(f"\n  Overall  (mean ± std across folds): "
              f"{np.mean(results)*100:.1f}% ± {np.std(results)*100:.1f}%"
              f"  ← realistic live estimate")
        print()
        print(f"  {'Freq':>6}   {'Pooled acc':>10}   {'Mean ± std (per fold)':>24}")
        print(f"  {'──────':>6}   {'──────────':>10}   {'────────────────────────':>24}")
        for freq in freqs:
            pooled = per_class_all[freq]
            if pooled['total'] > 0:
                pooled_acc = 100 * pooled['correct'] / pooled['total']
            else:
                pooled_acc = float('nan')
            folds = per_class[freq]
            if folds:
                fold_mean = np.mean(folds) * 100
                fold_std  = np.std(folds)  * 100
                fold_str  = f"{fold_mean:.1f}% ± {fold_std:.1f}%"
            else:
                fold_str = "n/a"
            print(f"  {freq:>5.1f} hz   {pooled_acc:>9.1f}%   {fold_str:>24}")

    return float(np.mean(results)) if results else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train FBTRCA model from XDF calibration data (7.5 / 10 / 12 Hz).")
    parser.add_argument("--data_dir", default="./newGuiData",
                        help="Root folder containing XDF files (searched recursively)")
    parser.add_argument("--output",   default="./fbtrca_model_751012.pkl",
                        help="Output pickle path (default: fbtrca_model_751012.pkl)")
    parser.add_argument("--validate", action="store_true",
                        help="Run leave-one-file-out validation before saving")
    parser.add_argument("--freqs",    nargs="+", type=float,
                        help="Only use these frequencies, e.g. --freqs 7.5 10.0 12.0")
    args = parser.parse_args()

    all_paths = sorted(glob.glob(
        os.path.join(args.data_dir, "**", "*.xdf"), recursive=True))
    if not all_paths:
        sys.exit(f"No .xdf files found under {args.data_dir}")

    valid_paths = []
    for p in all_paths:
        try:
            freq_from_name(p)
            valid_paths.append(p)
        except ValueError:
            print(f"  SKIP (no freq in name): {os.path.basename(p)}")

    if args.freqs:
        valid_paths = [p for p in valid_paths if freq_from_name(p) in args.freqs]

    freqs = sorted({freq_from_name(p) for p in valid_paths})
    if not freqs:
        sys.exit("No usable files after filtering.")

    freq_to_class = {f: i for i, f in enumerate(freqs)}
    print(f"\nFrequencies : {freqs}")
    print(f"Files found : {len(valid_paths)}\n")

    sfreq_val = None

    if args.validate:
        print("=" * 65)
        print("LEAVE-ONE-FILE-OUT CROSS-VALIDATION")
        print("=" * 65)
        file_data = {}
        for p in valid_paths:
            try:
                X, sf = load_file(p)
                windows = windowize(X, sf)
                cls = freq_to_class[freq_from_name(p)]
                file_data[p] = (windows, np.full(len(windows), cls, dtype=int))
                sfreq_val = sf
                print(f"  Loaded {os.path.basename(p):55s} {len(windows):3d} windows")
            except Exception as e:
                print(f"  SKIP   {os.path.basename(p):55s} {e}")
        print()
        lofo_validate(file_data, freq_to_class, freqs, sfreq_val)
        print()

    print("=" * 65)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 65)
    X_all, y_all, sfreq = load_dataset(valid_paths, freq_to_class)
    print(f"\nTotal windows : {X_all.shape}")

    model = FBTRCA(sfreq=sfreq, filter_bands=FB_BANDS,
                   weight_exp=WEIGHT_EXP, trca_reg=1e-6).fit(X_all, y_all)

    model_info = {
        "model":          model,
        "freqs":          freqs,
        "freq_to_class":  freq_to_class,
        "sfreq":          sfreq,
        "win_sec":        WIN_SEC,
        "stride_sec":     STRIDE_SEC,
        "drop_start_sec": DROP_START_SEC,
        "drop_end_sec":   DROP_END_SEC,
        "line_freq":      LINE_FREQ,
        "bandpass":       BANDPASS,
        "resample_hz":    RESAMPLE_HZ,
        "fb_bands":       FB_BANDS,
        "weight_exp":     WEIGHT_EXP,
    }
    with open(args.output, "wb") as f:
        pickle.dump(model_info, f)

    print(f"\nSaved → {args.output}  (classes: {freqs})")


if __name__ == "__main__":
    main()

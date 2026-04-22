#!/usr/bin/env python3
"""
EEGPiano — unified launcher
============================
Set MODE and MODEL_FILE below, then run:  python start.py

TRAIN  – EEG headset + stimulus GUI + (use Lab Recorder to capture .xdf)
TEST   – EEG headset + accuracy test GUI + live classifier
RUN    – EEG headset + interactive piano GUI + live classifier + EMG
"""

import subprocess
import threading
import time
import signal
import sys

MODE       = "RUN"                      # "TRAIN" | "TEST" | "RUN"
MODEL_FILE = "fbtrca_model_751012.pkl"   # pkl produced by train.py


# ── Output streaming ──────────────────────────────────────────────────────────

def stream_output(process, name):
    for line in process.stdout:
        print(f'[{name}] {line}', end='')


# ── Process registry ──────────────────────────────────────────────────────────

processes: dict = {}


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def shutdown(signum=None, frame=None):
    print('\n[start] shutting down...')

    # Step 1: Stop the headset process gracefully first.
    # unicornlsl.py catches SIGTERM and runs clean stop_acq + drain.
    lsl = processes.get('LSL')
    if lsl and lsl.poll() is None:
        print('[start] sending SIGTERM to LSL — waiting for headset to stop...')
        lsl.send_signal(signal.SIGTERM)
        try:
            lsl.wait(timeout=15)
            print('[start] LSL stopped cleanly')
        except subprocess.TimeoutExpired:
            print('[start] LSL timeout — force killing')
            lsl.kill()

    # Step 2: Terminate all remaining processes.
    for name, p in processes.items():
        if name == 'LSL':
            continue
        if p.poll() is None:
            print(f'[start] terminating {name}...')
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    print('[start] all done')
    sys.exit(0)


# ── Launch helper ─────────────────────────────────────────────────────────────

def launch(key, args):
    p = subprocess.Popen(
        ['python', '-u'] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    processes[key] = p
    threading.Thread(target=stream_output, args=(p, key), daemon=True).start()
    return p


# ── Build process set for chosen mode ────────────────────────────────────────

if MODE == "TRAIN":
    launch('LSL',   ['unicornlsl.py'])
    launch('GUI',   ['gui.py', 'TRAIN'])

elif MODE == "TEST":
    launch('LSL',   ['unicornlsl.py'])
    launch('GUI',   ['gui.py', 'TEST'])
    launch('MODEL', ['live_pipeline.py', 'TEST', MODEL_FILE])

elif MODE == "RUN":
    launch('LSL',   ['unicornlsl.py'])
    launch('GUI',   ['gui.py', 'RUN'])
    launch('MODEL', ['live_pipeline.py', 'RUN', MODEL_FILE])
    launch('EMG',   ['EMG_live.py'])

else:
    print(f'[start] Unknown MODE "{MODE}". Set MODE to "TRAIN", "TEST", or "RUN".')
    sys.exit(1)

print(f'[start] Mode={MODE}  |  Model={MODEL_FILE}  |  Launched: {list(processes.keys())}')

# ── Signal handling ───────────────────────────────────────────────────────────

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ── Watch: if any child exits, shut everything down ───────────────────────────

try:
    while True:
        for name, p in list(processes.items()):
            if p.poll() is not None:
                print(f'[start] {name} exited (code {p.returncode}), shutting down')
                shutdown()
        time.sleep(0.5)
except KeyboardInterrupt:
    shutdown()

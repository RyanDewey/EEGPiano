"""
piano_output.py
===============
Reads the EEG (SSVEPPredictions) and EMG (EMGPred) LSL streams and plays
a synthesized piano note for each unique EMG trigger.

Note map (4 combinations):
  EMG = -1 (left clench)  + EEG = 10 Hz  →  C4  (MIDI 60)
  EMG = -1 (left clench)  + EEG = 12 Hz  →  E4  (MIDI 64)
  EMG = +1 (right clench) + EEG = 10 Hz  →  G4  (MIDI 67)
  EMG = +1 (right clench) + EEG = 12 Hz  →  C5  (MIDI 72)
  EMG =  0 (rest)                         →  silence
"""

import time
import numpy as np
import pygame
from pylsl import StreamInlet, resolve_byprop

# ─── Note map ─────────────────────────────────────────────────────────────────

NOTE_MAP = {
    (-1, 10.0): 55,   # G3 — left  + 10 Hz
    (-1, 12.0): 57,   # A3 — left  + 12 Hz
    ( 1, 10.0): 59,   # B3 — right + 10 Hz
    ( 1, 12.0): 62,   # D4 — right + 12 Hz
}

NOTE_NAMES = {55: 'G3', 57: 'A3', 59: 'B3', 62: 'D4'}

SAMPLE_RATE   = 44100
TONE_DURATION = 800    # ms — note length before natural decay ends it

# ─── Audio synthesis ──────────────────────────────────────────────────────────

def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def make_piano_tone(midi_note: int, duration_ms: int = TONE_DURATION,
                    sample_rate: int = SAMPLE_RATE, volume: float = 0.7) -> pygame.mixer.Sound:
    """Synthesize a piano-like tone with harmonics and exponential decay."""
    freq = midi_to_hz(midi_note)
    n    = int(sample_rate * duration_ms / 1000)
    t    = np.linspace(0, duration_ms / 1000, n, endpoint=False)

    # Fundamental + harmonics (approximates a piano timbre)
    wave  = 1.00 * np.sin(2 * np.pi * 1 * freq * t)
    wave += 0.60 * np.sin(2 * np.pi * 2 * freq * t)
    wave += 0.30 * np.sin(2 * np.pi * 3 * freq * t)
    wave += 0.15 * np.sin(2 * np.pi * 4 * freq * t)
    wave += 0.07 * np.sin(2 * np.pi * 5 * freq * t)

    # Piano envelope: short linear attack then exponential decay
    attack_n  = int(0.005 * sample_rate)
    envelope  = np.exp(-t * 3.5)
    envelope[:attack_n] = np.linspace(0, 1, attack_n)

    wave = wave * envelope * volume
    wave = (wave * 32767).clip(-32767, 32767).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([wave, wave]))
    return pygame.sndarray.make_sound(stereo)


# ─── Init audio ───────────────────────────────────────────────────────────────

pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
pygame.mixer.set_num_channels(8)

print('[piano] Synthesizing tones...')
sounds = {key: make_piano_tone(midi) for key, midi in NOTE_MAP.items()}
print('[piano] Tones ready.')

# ─── Connect to LSL streams ───────────────────────────────────────────────────

print('[piano] Waiting for SSVEPPredictions stream (from live_pipeline.py)...')
eeg_streams = []
while not eeg_streams:
    eeg_streams = resolve_byprop('name', 'SSVEPPredictions', timeout=2.0)
    if not eeg_streams:
        print('[piano] SSVEPPredictions not found yet, retrying...')
        time.sleep(0.5)
eeg_inlet = StreamInlet(eeg_streams[0])
print('[piano] Connected to SSVEPPredictions.')

print('[piano] Waiting for EMGPred stream (from EMG_Live.py)...')
emg_streams = []
while not emg_streams:
    emg_streams = resolve_byprop('name', 'EMGPred', timeout=2.0)
    if not emg_streams:
        print('[piano] EMGPred not found yet, retrying...')
        time.sleep(0.5)
emg_inlet = StreamInlet(emg_streams[0])
print('[piano] Connected to EMGPred.')

# ─── Main loop ────────────────────────────────────────────────────────────────

last_eeg  = 10.0   # default to 10 Hz until first EEG prediction arrives
last_emg  = 0
active_ch = None   # the pygame channel currently playing

print('[piano] Running. Waiting for EMG triggers...')

while True:
    # Pull latest EEG prediction (non-blocking)
    eeg_sample, _ = eeg_inlet.pull_sample(timeout=0.0)
    if eeg_sample:
        last_eeg = float(eeg_sample[0])

    # Pull latest EMG prediction (non-blocking)
    emg_sample, _ = emg_inlet.pull_sample(timeout=0.0)
    if emg_sample:
        emg = int(round(float(emg_sample[0])))

        if emg != 0 and emg != last_emg:
            # Rising edge: new clench detected — play note
            key = (emg, last_eeg)
            if key in sounds:
                if active_ch is not None and active_ch.get_busy():
                    active_ch.stop()
                active_ch = sounds[key].play()
                midi = NOTE_MAP[key]
                print(f'[piano] ♪ {NOTE_NAMES[midi]} (MIDI {midi})  '
                      f'EMG={emg:+d}  EEG={last_eeg} Hz')

        elif emg == 0 and last_emg != 0:
            # Falling edge: clench released — fade out
            if active_ch is not None and active_ch.get_busy():
                active_ch.fadeout(150)
            active_ch = None

        last_emg = emg

    time.sleep(0.005)

#!/usr/bin/env python3
"""
gui.py — Unified SSVEP Piano GUI
==================================
Supports three modes via sys.argv[1]:

  TRAIN  – Phase-reversing stimulus display for EEG data collection.
           Use Lab Recorder to capture the LSL streams to .xdf.
           SPACE starts/stops stimulation.

  TEST   – Automated 6-trial accuracy test (6 × 10 s).
           Sends cue markers; live_pipeline.py scores predictions.
           SPACE starts a new session.

  RUN    – Interactive piano: click keys to assign them to frequencies,
           classifier predictions highlight the target, EMG plays the note.
           SPACE starts/stops stimulus.

Usage:
    python gui.py [TRAIN|TEST|RUN]
"""

import math
import os
import random
import string
import sys
import time

import numpy as np
import pygame
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

# ── Mode ──────────────────────────────────────────────────────────────────────
MODE = sys.argv[1].upper() if len(sys.argv) > 1 else "RUN"
if MODE not in ("TRAIN", "TEST", "RUN"):
    print(f"[GUI] Unknown mode '{MODE}' — defaulting to RUN")
    MODE = "RUN"

# ── Shared constants ──────────────────────────────────────────────────────────
FREQUENCIES      = [7.5, 10.0, 12.0]
FREQ_COLORS      = [(255, 165,   0), ( 80, 160, 255), (210,  80, 255)]  # orange, blue, violet
# Per-freq, per-side (0=left clench, 1=right clench) color shades
CHORD_COLORS     = [
    [(255, 165,   0), (255, 210, 110)],   # 7.5 Hz  left=orange,      right=amber
    [( 80, 160, 255), (100, 230, 255)],   # 10  Hz  left=blue,        right=cyan-blue
    [(210,  80, 255), (235, 150, 255)],   # 12  Hz  left=violet,      right=lavender
]
MAX_NOTES_PER_SLOT = 4   # max notes per (freq, clench) chord
NOTE_MIDI_DEFAULT = [55, 57, 59]   # G3, A3, B3

CHECKER_PX = 15
STIM_PX    = 210
STIM_GAP   = 100
BORDER_PX  = 5

SCREEN_W   = 1440
SCREEN_H   = 900
FULLSCREEN = False
TARGET_FPS = 240

PIANO_MIDI_START = 36
PIANO_MIDI_END   = 83
PIANO_H_FRAC     = 0.28
WHITE_SEMITONES  = {0, 2, 4, 5, 7, 9, 11}
BK_OFFSETS       = {1: 1.0, 3: 2.0, 6: 4.0, 8: 5.0, 10: 6.0}

BG_COLOR   = (18,  18,  28)
STATUS_BG  = (28,  28,  42)
TEXT_COLOR = (220, 220, 220)
DIM_COLOR  = (90,  90,  100)
STATUS_H   = 50

MARKER_EXP_START    = 252
MARKER_EXP_STOP     = 253
MARKER_SESSION_END  = 255
MARKER_TARGET_BASE  = 100
MARKER_CALIB_START  = 280
MARKER_CALIB_DONE   = 281
MARKER_CALIB_CUE_BASE = 150   # 150 + freq sent per calibration cue

CALIB_FREQ_SEC   = 30   # seconds spent on each frequency during calibration
CALIB_BREAK_SEC  = 10   # rest between frequencies

# ── TEST-only constants ───────────────────────────────────────────────────────
N_TRIALS       = 6
TRIAL_SEC      = 10.0
COUNTDOWN_SECS = 3
PAUSE_SECS     = 2

IDLE        = 'idle'
COUNTDOWN   = 'countdown'
RUNNING     = 'running'
PAUSE       = 'pause'
DONE        = 'done'
CALIBRATING = 'calibrating'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uid(n=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _draw_checkerboard(surface, rect, phase):
    """Phase-reversing black/white checkerboard inside rect."""
    x0, y0, w, h = rect.x, rect.y, rect.width, rect.height
    rows = math.ceil(h / CHECKER_PX)
    cols = math.ceil(w / CHECKER_PX)
    for r in range(rows):
        for c in range(cols):
            px = x0 + c * CHECKER_PX
            py = y0 + r * CHECKER_PX
            pw = min(CHECKER_PX, x0 + w - px)
            ph = min(CHECKER_PX, y0 + h - py)
            if pw <= 0 or ph <= 0:
                continue
            white = (r + c + phase) % 2 == 0
            pygame.draw.rect(surface, (255, 255, 255) if white else (0, 0, 0),
                             (px, py, pw, ph))


def _draw_checkerboard_idle(surface, rect):
    """Dim gray static checkerboard shown when stimulation is stopped."""
    x0, y0, w, h = rect.x, rect.y, rect.width, rect.height
    rows = math.ceil(h / CHECKER_PX)
    cols = math.ceil(w / CHECKER_PX)
    for r in range(rows):
        for c in range(cols):
            px = x0 + c * CHECKER_PX
            py = y0 + r * CHECKER_PX
            pw = min(CHECKER_PX, x0 + w - px)
            ph = min(CHECKER_PX, y0 + h - py)
            if pw <= 0 or ph <= 0:
                continue
            col = (70, 70, 80) if (r + c) % 2 == 0 else (45, 45, 55)
            pygame.draw.rect(surface, col, (px, py, pw, ph))


def _build_piano_sound(midi_note: int, duration: float = 1.0,
                       sr: int = 44100) -> pygame.mixer.Sound:
    """
    Synthesize a piano-like tone using additive synthesis + ADSR envelope.
    """
    freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    n    = int(sr * duration)
    t    = np.linspace(0.0, duration, n, endpoint=False)

    wave = (
        1.00 * np.sin(2 * np.pi * 1 * freq * t) +
        0.50 * np.sin(2 * np.pi * 2 * freq * t) +
        0.25 * np.sin(2 * np.pi * 3 * freq * t) +
        0.12 * np.sin(2 * np.pi * 4 * freq * t) +
        0.06 * np.sin(2 * np.pi * 5 * freq * t) +
        0.03 * np.sin(2 * np.pi * 6 * freq * t)
    )
    wave /= np.max(np.abs(wave))

    atk = int(0.005 * sr)
    dec = int(0.150 * sr)
    rel = int(0.500 * sr)
    sus = max(n - atk - dec - rel, 0)
    sus_lvl = 0.60

    env = np.concatenate([
        np.linspace(0.0,     1.0,     atk),
        np.linspace(1.0,     sus_lvl, dec),
        np.full(sus,         sus_lvl),
        np.linspace(sus_lvl, 0.0,     rel),
    ])[:n]

    wave = (wave * env * 0.80 * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)


def _get_sound(midi: int, cache: dict) -> pygame.mixer.Sound:
    if midi not in cache:
        cache[midi] = _build_piano_sound(midi)
    return cache[midi]


def _midi_to_name(midi: int) -> str:
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    return f'{note_names[midi % 12]}{octave}'


def _get_clicked_key(mx: int, my: int, piano) -> str | None:
    """Return note name of piano key at pixel (mx, my), or None."""
    if not piano.rect.collidepoint(mx, my):
        return None
    # Black keys sit on top of white keys — check first
    for bk in piano.black_keys:
        x = int(bk['x'])
        w = max(1, int(piano.bkw))
        h = int(piano.bkh)
        if x <= mx <= x + w and piano.rect.y <= my <= piano.rect.y + h:
            return _midi_to_name(bk['midi'])
    for wk in piano.white_keys:
        x = int(wk['x'])
        w = max(1, int(piano.wkw))
        if x <= mx <= x + w:
            return _midi_to_name(wk['midi'])
    return None


# ── TEST helper ───────────────────────────────────────────────────────────────

def make_trial_sequence():
    """Each of the 3 frequencies appears exactly twice, randomly ordered."""
    pool = FREQUENCIES * 2
    random.shuffle(pool)
    return pool


# ── Piano class ───────────────────────────────────────────────────────────────

class Piano:
    """Draws a multi-octave piano keyboard and manages key highlights."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.highlights: dict = {}   # midi → RGB

        n_white = sum(
            1 for m in range(PIANO_MIDI_START, PIANO_MIDI_END + 1)
            if m % 12 in WHITE_SEMITONES
        )
        self.wkw = rect.width / n_white
        self.wkh = rect.height
        self.bkw = self.wkw * 0.58
        self.bkh = self.wkh * 0.62

        self.white_keys: list = []
        w_idx = 0
        for midi in range(PIANO_MIDI_START, PIANO_MIDI_END + 1):
            if midi % 12 in WHITE_SEMITONES:
                self.white_keys.append({
                    'midi': midi,
                    'x':    rect.x + w_idx * self.wkw,
                })
                w_idx += 1

        wk_x = {wk['midi']: wk['x'] for wk in self.white_keys}

        self.black_keys: list = []
        for midi in range(PIANO_MIDI_START, PIANO_MIDI_END + 1):
            s = midi % 12
            if s in BK_OFFSETS:
                c_midi = (midi // 12) * 12
                if c_midi in wk_x:
                    bx = wk_x[c_midi] + BK_OFFSETS[s] * self.wkw - self.bkw / 2
                    self.black_keys.append({'midi': midi, 'x': bx})

    def key_center_x(self, midi: int) -> float:
        if midi % 12 in WHITE_SEMITONES:
            for wk in self.white_keys:
                if wk['midi'] == midi:
                    return wk['x'] + self.wkw / 2
        else:
            for bk in self.black_keys:
                if bk['midi'] == midi:
                    return bk['x'] + self.bkw / 2
        return float(self.rect.centerx)

    def draw(self, surface: pygame.Surface, small_font: pygame.font.Font):
        # White keys first
        for wk in self.white_keys:
            x   = int(wk['x'])
            w   = max(1, int(self.wkw))
            mid = wk['midi']
            col = self.highlights.get(mid, (235, 235, 235))
            pygame.draw.rect(surface, col,
                             (x, self.rect.y, w + 1, int(self.wkh)),
                             border_top_left_radius=6, border_top_right_radius=6)
            pygame.draw.rect(surface, (50, 50, 50),
                             (x, self.rect.y, w + 1, int(self.wkh)),
                             1, border_top_left_radius=6, border_top_right_radius=6)
            if mid % 12 == 0:
                octave = mid // 12 - 1
                lbl = small_font.render(f'C{octave}', True, (80, 80, 80))
                surface.blit(lbl, (
                    x + w // 2 - lbl.get_width() // 2,
                    self.rect.y + self.wkh - lbl.get_height() - 5,
                ))

        # Black keys on top
        for bk in self.black_keys:
            x   = int(bk['x'])
            w   = max(1, int(self.bkw))
            mid = bk['midi']
            col = self.highlights.get(mid, (25, 25, 25))
            pygame.draw.rect(surface, col,
                             (x, self.rect.y, w, int(self.bkh)),
                             border_bottom_left_radius=4, border_bottom_right_radius=4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    note_sounds: dict = {}

    flags  = pygame.FULLSCREEN if FULLSCREEN else 0
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
    pygame.display.set_caption(f'SSVEP Piano — {MODE}')
    clock  = pygame.time.Clock()

    # ── Startup splash ────────────────────────────────────────────────────────
    try:
        _logo_raw  = pygame.image.load(
            os.path.join(os.path.dirname(__file__), 'NeuroKeysLogo.jpeg')
        ).convert()
        _logo      = pygame.transform.smoothscale(_logo_raw, (SCREEN_W, SCREEN_H))
        _logo_rect = _logo.get_rect(topleft=(0, 0))
        _FADE_IN   = 0.8    # seconds
        _HOLD      = 1.4    # seconds
        _FADE_OUT  = 0.8    # seconds
        _t0 = time.perf_counter()
        _running = True
        while _running:
            for _ev in pygame.event.get():
                if _ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if _ev.type == pygame.KEYDOWN:
                    _running = False
            _elapsed = time.perf_counter() - _t0
            if _elapsed >= _FADE_IN + _HOLD + _FADE_OUT:
                break
            if _elapsed < _FADE_IN:
                _alpha = int(255 * _elapsed / _FADE_IN)
            elif _elapsed < _FADE_IN + _HOLD:
                _alpha = 255
            else:
                _alpha = int(255 * (1.0 - (_elapsed - _FADE_IN - _HOLD) / _FADE_OUT))
            _logo.set_alpha(max(0, min(255, _alpha)))
            screen.fill(BG_COLOR)
            screen.blit(_logo, _logo_rect)
            pygame.display.flip()
            clock.tick(60)
    except Exception:
        pass   # logo missing or load error — skip splash silently

    font_xl = pygame.font.SysFont('Arial', 80, bold=True)
    font_lg = pygame.font.SysFont('Arial', 32, bold=True)
    font_md = pygame.font.SysFont('Arial', 22)
    font_sm = pygame.font.SysFont('Arial', 14)

    # ── LSL marker outlet ──────────────────────────────────────────────────────
    m_info = StreamInfo(
        name='SSVEPMarkers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='float32',
        source_id=f'ssvep_{_uid()}',
    )
    marker_outlet = StreamOutlet(m_info)
    print('[GUI] Marker outlet "SSVEPMarkers" ready.')

    # ── Layout ────────────────────────────────────────────────────────────────
    piano_h    = int(SCREEN_H * PIANO_H_FRAC)
    piano_rect = pygame.Rect(0, SCREEN_H - piano_h, SCREEN_W, piano_h)
    piano      = Piano(piano_rect)

    stim_cy = (STATUS_H + piano_rect.y) // 2
    n       = len(FREQUENCIES)
    total_w = n * STIM_PX + (n - 1) * STIM_GAP
    x_start = (SCREEN_W - total_w) // 2 + STIM_PX // 2
    stim_cx = [x_start + i * (STIM_PX + STIM_GAP) for i in range(n)]

    key_cx_default = [int(piano.key_center_x(m)) for m in NOTE_MIDI_DEFAULT]

    stim_rects = [
        pygame.Rect(cx - STIM_PX // 2, stim_cy - STIM_PX // 2, STIM_PX, STIM_PX)
        for cx in stim_cx
    ]

    # Progress bar (TEST mode)
    BAR_H = 12
    BAR_Y = piano_rect.y - BAR_H - 8
    BAR_X = stim_rects[0].x
    BAR_W = stim_rects[-1].right - stim_rects[0].x

    # ── Mode-specific init ─────────────────────────────────────────────────────
    running_stim   = False
    target_idx     = -1
    prev_phases    = [0] * n
    t_stim         = time.perf_counter()

    # Calibration state (shared across all modes)
    in_calibration  = False
    calib_freq_idx  = 0
    calib_t         = 0.0

    # RUN-mode state
    # chord_map[(freq_idx, side)] → [midi, ...]  side: 0=left(-1), 1=right(+1)
    chord_map: dict      = {}
    selected_midi: int   = -1
    key_press_time: float = -1
    pred_inlet  = None
    emg_inlet   = None

    # TEST-mode state
    state      = IDLE
    trial_seq  = make_trial_sequence()
    trial_idx_t = 0
    state_t    = 0.0

    if MODE == "TRAIN":
        pass   # nothing extra needed

    elif MODE == "TEST":
        state      = IDLE
        trial_seq  = make_trial_sequence()
        trial_idx_t = 0
        state_t    = 0.0
        running_stim = False
        target_idx  = -1
        prev_phases = [0] * n
        t_stim      = time.perf_counter()

    elif MODE == "RUN":
        running_stim  = False
        target_idx    = -1
        chord_map     = {}
        selected_midi = -1
        key_press_time = -1

        pred_streams = resolve_byprop('name', 'SSVEPPredictions', timeout=1.0)
        if pred_streams:
            pred_inlet = StreamInlet(pred_streams[0])
            print('[GUI] Connected to SSVEPPredictions stream.')
        else:
            print('[GUI] No prediction stream found — highlight-only mode.')

        emg_streams = resolve_byprop('name', 'EMGControl', timeout=1.0)
        if emg_streams:
            emg_inlet = StreamInlet(emg_streams[0])
            print('[GUI] Connected to EMGControl stream.')
        else:
            print('[GUI] No EMG stream found — keyboard-only mode.')

    # ── Helper closures ───────────────────────────────────────────────────────
    def start_state(new_state):
        nonlocal state, state_t
        state   = new_state
        state_t = time.perf_counter()

    def send_target(tidx):
        nonlocal target_idx
        target_idx = tidx
        freq = FREQUENCIES[tidx]
        marker_outlet.push_sample([float(MARKER_TARGET_BASE + freq)])
        print(f'[GUI] Trial {trial_idx_t + 1}/{N_TRIALS} — target: {freq} Hz')

    def start_stim():
        nonlocal running_stim, t_stim, prev_phases
        running_stim = True
        t_stim       = time.perf_counter()
        prev_phases  = [0] * n
        marker_outlet.push_sample([float(MARKER_EXP_START)])
        print(f'[GUI] {MARKER_EXP_START} → Experiment START')

    def stop_stim():
        nonlocal running_stim
        running_stim = False
        marker_outlet.push_sample([float(MARKER_EXP_STOP)])
        print(f'[GUI] {MARKER_EXP_STOP} → Experiment STOP')

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        now_abs = time.perf_counter()
        elapsed = now_abs - state_t

        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    if running_stim:
                        stop_stim()
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_c:
                    # Start calibration — blocked during active TEST trial
                    test_busy = (MODE == "TEST" and state in (COUNTDOWN, RUNNING))
                    if not in_calibration and not test_busy:
                        in_calibration = True
                        calib_freq_idx = 0
                        calib_t        = time.perf_counter()
                        target_idx     = 0
                        if not running_stim:
                            start_stim()
                        marker_outlet.push_sample([float(MARKER_CALIB_START)])
                        marker_outlet.push_sample([float(MARKER_CALIB_CUE_BASE + FREQUENCIES[0])])
                        print(f'[GUI] Calibration started — {n} freqs × {CALIB_FREQ_SEC}s each')

                if event.key == pygame.K_SPACE:
                    if in_calibration:
                        pass   # SPACE blocked during calibration

                    elif MODE == "TRAIN":
                        running_stim = not running_stim
                        if running_stim:
                            start_stim()
                        else:
                            stop_stim()

                    elif MODE == "TEST":
                        if state in (IDLE, DONE):
                            nonlocal_trial_seq = make_trial_sequence()
                            trial_seq[:]  = nonlocal_trial_seq
                            trial_idx_t   = 0
                            target_idx    = -1
                            send_target(FREQUENCIES.index(trial_seq[0]))
                            start_state(COUNTDOWN)

                    elif MODE == "RUN":
                        running_stim = not running_stim
                        if running_stim:
                            start_stim()
                        else:
                            stop_stim()

                # RUN-only keyboard shortcuts
                if MODE == "RUN":
                    if event.key == pygame.K_0:
                        chord_map     = {}
                        selected_midi = -1
                        print('[GUI] All chord assignments cleared')

                    # ← / → — simulate left/right EMG clench on current target
                    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        if target_idx >= 0:
                            side      = 0 if event.key == pygame.K_LEFT else 1
                            side_name = 'Left' if side == 0 else 'Right'
                            midis     = chord_map.get((target_idx, side), [])
                            if midis:
                                for midi in midis:
                                    _get_sound(midi, note_sounds).play()
                                key_press_time = time.perf_counter()
                                names = '+'.join(_midi_to_name(m) for m in midis)
                                print(f'[SIM-EMG] {side_name} clench → {names}')
                            else:
                                print(f'[SIM-EMG] {side_name} clench — no notes assigned '
                                      f'to {FREQUENCIES[target_idx]} Hz {side_name.lower()}')
                        else:
                            print('[SIM-EMG] no target predicted yet — look at a stimulus first')

                    # 1–6 — simulate EEG prediction + EMG clench
                    # 1=7.5Hz left, 2=7.5Hz right, 3=10Hz left,
                    # 4=10Hz right, 5=12Hz left, 6=12Hz right
                    sim_keys = [pygame.K_1, pygame.K_2, pygame.K_3,
                                pygame.K_4, pygame.K_5, pygame.K_6]
                    for ki, k in enumerate(sim_keys):
                        if event.key == k:
                            fi   = ki // 2   # freq index 0/1/2
                            side = ki % 2    # 0=left, 1=right
                            midis = chord_map.get((fi, side), [])
                            target_idx = fi
                            if midis:
                                for midi in midis:
                                    _get_sound(midi, note_sounds).play()
                                key_press_time = time.perf_counter()
                                side_name = 'Left' if side == 0 else 'Right'
                                names = '+'.join(_midi_to_name(m) for m in midis)
                                print(f'[SIM] {FREQUENCIES[fi]} Hz {side_name} → {names}')

            if event.type == pygame.MOUSEBUTTONDOWN and MODE == "RUN":
                mx, my = event.pos
                shift  = pygame.key.get_mods() & pygame.KMOD_SHIFT

                clicked_stim = None
                for i, rect in enumerate(stim_rects):
                    if rect.collidepoint(mx, my):
                        clicked_stim = i
                        break

                if clicked_stim is not None and selected_midi != -1:
                    # click → left clench (side 0), Shift+click → right clench (side 1)
                    side      = 1 if shift else 0
                    side_name = 'right' if side == 1 else 'left'
                    slot      = chord_map.get((clicked_stim, side), [])
                    if selected_midi in slot:
                        print(f'[GUI] {_midi_to_name(selected_midi)} already in '
                              f'{FREQUENCIES[clicked_stim]} Hz {side_name} chord')
                    elif len(slot) >= MAX_NOTES_PER_SLOT:
                        print(f'[GUI] {FREQUENCIES[clicked_stim]} Hz {side_name} chord '
                              f'is full ({MAX_NOTES_PER_SLOT} notes max) — press 0 to reset')
                    else:
                        slot.append(selected_midi)
                        chord_map[(clicked_stim, side)] = slot
                        names = '+'.join(_midi_to_name(m) for m in slot)
                        print(f'[GUI] {FREQUENCIES[clicked_stim]} Hz {side_name}: {names}')
                    selected_midi = -1

                elif clicked_stim is not None and selected_midi == -1:
                    print('[GUI] Select a piano key first, then click box=left clench  '
                          'Shift+click box=right clench')

                else:
                    clicked_note = _get_clicked_key(mx, my, piano)
                    if clicked_note is not None:
                        clicked_midi = -1
                        for wk in piano.white_keys:
                            if _midi_to_name(wk['midi']) == clicked_note:
                                clicked_midi = wk['midi']
                                break
                        if clicked_midi == -1:
                            for bk in piano.black_keys:
                                if _midi_to_name(bk['midi']) == clicked_note:
                                    clicked_midi = bk['midi']
                                    break

                        if clicked_midi != -1:
                            # Check if this note is in any chord — if so preview it
                            found_freq = None
                            for (fi, si), midis in chord_map.items():
                                if clicked_midi in midis:
                                    found_freq = fi
                                    break
                            if found_freq is not None:
                                _get_sound(clicked_midi, note_sounds).play()
                                key_press_time = time.perf_counter()
                                target_idx = found_freq
                                print(f'[Mouse] Played: {clicked_note}')
                            else:
                                selected_midi = clicked_midi
                                print(f'[Mouse] Selected: {clicked_note} — '
                                      f'click freq box=left clench  Shift+click=right clench')

        # ── Calibration state machine ─────────────────────────────────────────
        # calib_freq_idx: current freq index (on break: stimulus is stopped,
        # target_idx = -1, waiting CALIB_BREAK_SEC before next freq starts)
        if in_calibration:
            calib_elapsed = time.perf_counter() - calib_t
            on_break = not running_stim and calib_freq_idx < n

            if on_break:
                if calib_elapsed >= CALIB_BREAK_SEC:
                    target_idx = calib_freq_idx
                    freq       = FREQUENCIES[calib_freq_idx]
                    start_stim()
                    calib_t = time.perf_counter()
                    marker_outlet.push_sample([float(MARKER_CALIB_CUE_BASE + freq)])
                    print(f'[GUI] Calibration freq {calib_freq_idx + 1}/{n}: {freq} Hz')
            else:
                if calib_elapsed >= CALIB_FREQ_SEC:
                    calib_freq_idx += 1
                    if calib_freq_idx >= n:
                        stop_stim()
                        marker_outlet.push_sample([float(MARKER_CALIB_DONE)])
                        in_calibration = False
                        calib_freq_idx = 0
                        target_idx     = -1
                        print('[GUI] Calibration complete — model update sent to pipeline')
                    else:
                        stop_stim()
                        target_idx = -1
                        calib_t    = time.perf_counter()
                        print(f'[GUI] Break — next freq in {CALIB_BREAK_SEC}s')

        # ── TEST state machine ────────────────────────────────────────────────
        if MODE == "TEST" and not in_calibration:
            if state == COUNTDOWN:
                if elapsed >= COUNTDOWN_SECS:
                    start_stim()
                    start_state(RUNNING)

            elif state == RUNNING:
                if elapsed >= TRIAL_SEC:
                    stop_stim()
                    trial_idx_t += 1
                    target_idx   = -1
                    if trial_idx_t >= N_TRIALS:
                        marker_outlet.push_sample([float(MARKER_SESSION_END)])
                        print(f'[GUI] Session complete — {N_TRIALS} trials done.')
                        start_state(DONE)
                    else:
                        start_state(PAUSE)

            elif state == PAUSE:
                if elapsed >= PAUSE_SECS:
                    send_target(FREQUENCIES.index(trial_seq[trial_idx_t]))
                    start_state(COUNTDOWN)

        # ── Phase computation & per-cycle markers ─────────────────────────────
        stim_now = time.perf_counter() - t_stim
        phases   = []
        for i, freq in enumerate(FREQUENCIES):
            phase = int(stim_now * freq * 2) % 2
            phases.append(phase)
            if running_stim and phase == 0 and prev_phases[i] != 0:
                marker_outlet.push_sample([float(freq)])
        if running_stim:
            prev_phases = phases[:]

        # ── RUN: poll prediction inlet ────────────────────────────────────────
        if MODE == "RUN" and pred_inlet is not None:
            sample, _ = pred_inlet.pull_sample(timeout=0.0)
            if sample is not None:
                predicted_freq = sample[0]
                if predicted_freq in FREQUENCIES:
                    pred_idx  = FREQUENCIES.index(predicted_freq)
                    has_notes = any(chord_map.get((pred_idx, s), []) for s in (0, 1))
                    if has_notes:
                        target_idx = pred_idx
                        print(f'[GUI] Predicted: {predicted_freq} Hz — waiting for EMG clench')

        # ── RUN: poll EMG inlet ───────────────────────────────────────────────
        if MODE == "RUN" and emg_inlet is not None:
            emg_sample, _ = emg_inlet.pull_sample(timeout=0.0)
            if emg_sample is not None:
                emg_val = emg_sample[0]
                if emg_val != 0.0 and target_idx >= 0:
                    side  = 0 if emg_val == -1.0 else 1
                    midis = chord_map.get((target_idx, side), [])
                    if midis:
                        for midi in midis:
                            _get_sound(midi, note_sounds).play()
                        key_press_time = time.perf_counter()
                        side_name = 'Left' if side == 0 else 'Right'
                        names = '+'.join(_midi_to_name(m) for m in midis)
                        print(f'[EMG] {side_name} clench → {names}')

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Piano highlights
        piano.highlights = {}
        if MODE in ("TRAIN", "TEST"):
            for i in range(n):
                piano.highlights[NOTE_MIDI_DEFAULT[i]] = FREQ_COLORS[i]
        elif MODE == "RUN":
            recently_played = (time.perf_counter() - key_press_time) < 0.3
            for (freq_idx, side), midis in chord_map.items():
                color = CHORD_COLORS[freq_idx][side]
                if freq_idx == target_idx and recently_played:
                    color = tuple(max(0, c - 60) for c in color)
                for midi in midis:
                    piano.highlights[midi] = color
            if selected_midi != -1:
                if selected_midi % 12 in WHITE_SEMITONES:
                    piano.highlights[selected_midi] = (255, 255, 255)
                else:
                    piano.highlights[selected_midi] = (90, 90, 90)

        piano.draw(screen, font_sm)

        # Connector lines
        if MODE in ("TRAIN", "TEST"):
            for i, cx in enumerate(stim_cx):
                line_col    = tuple(max(0, c - 80) for c in FREQ_COLORS[i])
                stim_bottom = stim_cy + STIM_PX // 2 + BORDER_PX + 2
                pygame.draw.line(screen, line_col,
                                 (cx, stim_bottom), (key_cx_default[i], piano_rect.y), 2)
        elif MODE == "RUN":
            stim_bottom = stim_cy + STIM_PX // 2 + BORDER_PX + 2
            for (freq_idx, side), midis in chord_map.items():
                cx       = stim_cx[freq_idx]
                line_col = tuple(max(0, c - 80) for c in CHORD_COLORS[freq_idx][side])
                for midi in midis:
                    kx = int(piano.key_center_x(midi))
                    pygame.draw.line(screen, line_col, (cx, stim_bottom), (kx, piano_rect.y), 2)

        # Stimulus boxes
        for i, rect in enumerate(stim_rects):
            is_target = (target_idx == i)

            if running_stim:
                _draw_checkerboard(screen, rect, phases[i])
            else:
                _draw_checkerboard_idle(screen, rect)

            bw        = BORDER_PX * 2 if is_target else BORDER_PX
            bord_rect = pygame.Rect(rect.x - bw, rect.y - bw,
                                    rect.w + bw * 2, rect.h + bw * 2)
            pygame.draw.rect(screen, FREQ_COLORS[i], bord_rect, bw)

            # Target indicator (TEST: show during COUNTDOWN + RUNNING)
            show_target_indicator = False
            if MODE == "TEST" and is_target and state in (COUNTDOWN, RUNNING):
                show_target_indicator = True
            elif MODE == "RUN" and is_target:
                show_target_indicator = True

            if show_target_indicator:
                gr = pygame.Rect(bord_rect.x - 4, bord_rect.y - 4,
                                 bord_rect.w + 8, bord_rect.h + 8)
                pygame.draw.rect(screen, (255, 255, 255), gr, 3)
                arrow = font_lg.render('▼ LOOK HERE ▼', True, FREQ_COLORS[i])
                screen.blit(arrow, (rect.centerx - arrow.get_width() // 2,
                                    rect.y - arrow.get_height() - 30))

            # Frequency label above box
            freq_txt = font_md.render(f'{FREQUENCIES[i]} Hz', True, FREQ_COLORS[i])
            screen.blit(freq_txt, (
                rect.centerx - freq_txt.get_width() // 2,
                rect.y - freq_txt.get_height() - 4,
            ))

            # Note label below box
            if MODE in ("TRAIN", "TEST"):
                note_txt = font_sm.render(_midi_to_name(NOTE_MIDI_DEFAULT[i]), True, TEXT_COLOR)
                screen.blit(note_txt, (rect.centerx - note_txt.get_width() // 2, rect.bottom + 6))
            else:
                for side, prefix in ((0, 'L:'), (1, 'R:')):
                    midis = chord_map.get((i, side), [])
                    label = prefix + ('+'.join(_midi_to_name(m) for m in midis) if midis else '—')
                    col   = CHORD_COLORS[i][side] if midis else DIM_COLOR
                    surf  = font_sm.render(label, True, col)
                    screen.blit(surf, (rect.centerx - surf.get_width() // 2,
                                       rect.bottom + 6 + side * 18))

        # ── TEST overlays ─────────────────────────────────────────────────────
        if MODE == "TEST":
            if state == RUNNING:
                fill = min(1.0, elapsed / TRIAL_SEC)
                pygame.draw.rect(screen, (50, 50, 60),
                                 (BAR_X, BAR_Y, BAR_W, BAR_H), border_radius=4)
                if fill > 0:
                    col = FREQ_COLORS[target_idx] if target_idx >= 0 else TEXT_COLOR
                    pygame.draw.rect(screen, col,
                                     (BAR_X, BAR_Y, int(BAR_W * fill), BAR_H), border_radius=4)
                secs_left = max(0.0, TRIAL_SEC - elapsed)
                timer_txt = font_sm.render(f'{secs_left:.1f}s remaining', True, DIM_COLOR)
                screen.blit(timer_txt, (
                    BAR_X + BAR_W // 2 - timer_txt.get_width() // 2,
                    BAR_Y - timer_txt.get_height() - 2,
                ))

            elif state == COUNTDOWN:
                cd = max(0, COUNTDOWN_SECS - int(elapsed))
                cd_surf = font_xl.render(str(cd) if cd > 0 else 'GO', True, (255, 255, 100))
                screen.blit(cd_surf, (
                    SCREEN_W // 2 - cd_surf.get_width() // 2,
                    stim_cy - cd_surf.get_height() // 2,
                ))

            elif state == DONE:
                overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))
                screen.blit(overlay, (0, 0))
                done_txt = font_lg.render('Session Complete — accuracy results in terminal',
                                          True, (100, 255, 100))
                restart  = font_md.render('Press SPACE to run again   |   ESC to quit',
                                          True, DIM_COLOR)
                screen.blit(done_txt, (SCREEN_W // 2 - done_txt.get_width() // 2,
                                       SCREEN_H // 2 - 40))
                screen.blit(restart,  (SCREEN_W // 2 - restart.get_width() // 2,
                                       SCREEN_H // 2 + 10))

        # ── TRAIN overlay (instructions when stopped) ─────────────────────────
        if MODE == "TRAIN" and not running_stim:
            panel_w, panel_h = 520, 160
            panel_x = SCREEN_W // 2 - panel_w // 2
            panel_y = stim_cy + STIM_PX // 2 + 30
            panel   = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 160))
            screen.blit(panel, (panel_x, panel_y))
            lines = [
                'TRAIN MODE',
                'SPACE = start / stop stimulus',
                'Open Lab Recorder before starting',
                'Name file: {freq}hz_run-XXX_eeg.xdf',
            ]
            for li, line in enumerate(lines):
                col  = TEXT_COLOR if li > 0 else (255, 200, 60)
                fnt  = font_md if li == 0 else font_sm
                surf = fnt.render(line, True, col)
                screen.blit(surf, (
                    panel_x + panel_w // 2 - surf.get_width() // 2,
                    panel_y + 14 + li * 34,
                ))

        # ── Calibration overlay ───────────────────────────────────────────────
        if in_calibration:
            calib_elapsed = time.perf_counter() - calib_t
            on_break      = not running_stim and calib_freq_idx < n

            if on_break:
                secs_left = max(0.0, CALIB_BREAK_SEC - calib_elapsed)
                fill      = min(1.0, calib_elapsed / CALIB_BREAK_SEC)
                col       = DIM_COLOR
                label     = f'Break  —  next freq in {secs_left:.0f}s'
            else:
                freq      = FREQUENCIES[calib_freq_idx]
                col       = FREQ_COLORS[calib_freq_idx]
                secs_left = max(0.0, CALIB_FREQ_SEC - calib_elapsed)
                fill      = min(1.0, calib_elapsed / CALIB_FREQ_SEC)
                label     = f'Calibrating {calib_freq_idx + 1}/{n}  —  {secs_left:.1f}s remaining'

            pygame.draw.rect(screen, (50, 50, 60),
                             (BAR_X, BAR_Y, BAR_W, BAR_H), border_radius=4)
            if fill > 0:
                pygame.draw.rect(screen, col,
                                 (BAR_X, BAR_Y, int(BAR_W * fill), BAR_H), border_radius=4)
            timer_txt = font_sm.render(label, True, col)
            screen.blit(timer_txt, (
                BAR_X + BAR_W // 2 - timer_txt.get_width() // 2,
                BAR_Y - timer_txt.get_height() - 4,
            ))

        # ── Status bar ────────────────────────────────────────────────────────
        pygame.draw.rect(screen, STATUS_BG, (0, 0, SCREEN_W, STATUS_H))
        bar_mid_y = STATUS_H // 2

        if in_calibration:
            _on_break = not running_stim and calib_freq_idx < n
            _elapsed  = time.perf_counter() - calib_t
            if _on_break:
                secs_c = max(0.0, CALIB_BREAK_SEC - _elapsed)
                col_c  = DIM_COLOR
                msg_c  = f'Rest  —  next frequency starts in {secs_c:.0f}s'
            else:
                freq_c = FREQUENCIES[calib_freq_idx]
                col_c  = FREQ_COLORS[calib_freq_idx]
                secs_c = max(0.0, CALIB_FREQ_SEC - _elapsed)
                msg_c  = f'Focus on  {freq_c} Hz  ({calib_freq_idx + 1}/{n})  —  {secs_c:.0f}s left'
            left_txt  = font_md.render('CALIBRATING', True, (255, 200, 60))
            mid_txt   = font_md.render(msg_c, True, col_c)
            right_txt = font_sm.render('ESC=quit', True, DIM_COLOR)
            screen.blit(left_txt,  (16, bar_mid_y - left_txt.get_height() // 2))
            screen.blit(mid_txt,   (SCREEN_W // 2 - mid_txt.get_width() // 2,
                                    bar_mid_y - mid_txt.get_height() // 2))
            screen.blit(right_txt, (SCREEN_W - right_txt.get_width() - 14,
                                    bar_mid_y - right_txt.get_height() // 2))

        elif MODE == "TRAIN":
            left_txt  = font_md.render('TRAIN MODE', True, TEXT_COLOR)
            state_str = 'RUNNING' if running_stim else 'STOPPED'
            state_col = (80, 255, 80) if running_stim else (255, 80, 80)
            mid_txt   = font_md.render(f'Stimulation: {state_str}', True, state_col)
            right_txt = font_sm.render('SPACE=start/stop  ESC=quit', True, DIM_COLOR)
            screen.blit(left_txt,  (16, bar_mid_y - left_txt.get_height() // 2))
            screen.blit(mid_txt,   (SCREEN_W // 2 - mid_txt.get_width() // 2,
                                    bar_mid_y - mid_txt.get_height() // 2))
            screen.blit(right_txt, (SCREEN_W - right_txt.get_width() - 14,
                                    bar_mid_y - right_txt.get_height() // 2))

        elif MODE == "TEST":
            left_txt = font_md.render('TEST MODE', True, TEXT_COLOR)
            screen.blit(left_txt, (16, bar_mid_y - left_txt.get_height() // 2))

            if state == IDLE:
                msg     = 'Press SPACE to begin  (6 trials × 10 s = 60 s total)'
                msg_col = DIM_COLOR
            elif state == COUNTDOWN:
                freq    = trial_seq[trial_idx_t]
                msg     = f'Trial {trial_idx_t + 1}/{N_TRIALS}  —  Focus on  {freq} Hz'
                msg_col = FREQ_COLORS[FREQUENCIES.index(freq)]
            elif state == RUNNING:
                freq    = trial_seq[trial_idx_t]
                msg     = f'Trial {trial_idx_t + 1}/{N_TRIALS}  —  RECORDING  —  target: {freq} Hz'
                msg_col = FREQ_COLORS[FREQUENCIES.index(freq)]
            elif state == PAUSE:
                nf      = trial_seq[trial_idx_t] if trial_idx_t < N_TRIALS else None
                msg     = (f'Trial {trial_idx_t}/{N_TRIALS} done  —  Next: {nf} Hz'
                           if nf else 'Last trial done')
                msg_col = TEXT_COLOR
            else:
                msg     = f'Done — {N_TRIALS} trials complete'
                msg_col = (100, 255, 100)

            mid_txt = font_md.render(msg, True, msg_col)
            screen.blit(mid_txt, (SCREEN_W // 2 - mid_txt.get_width() // 2,
                                  bar_mid_y - mid_txt.get_height() // 2))

            right_txt = font_sm.render('C=calibrate  SPACE=start  ESC=quit', True, DIM_COLOR)
            screen.blit(right_txt, (SCREEN_W - right_txt.get_width() - 14,
                                    bar_mid_y - right_txt.get_height() // 2))

        elif MODE == "RUN":
            left_txt = font_md.render('RUN MODE', True, TEXT_COLOR)
            screen.blit(left_txt, (16, bar_mid_y - left_txt.get_height() // 2))

            if target_idx >= 0:
                left_midis  = chord_map.get((target_idx, 0), [])
                right_midis = chord_map.get((target_idx, 1), [])
                parts = []
                if left_midis:
                    parts.append('L:' + '+'.join(_midi_to_name(m) for m in left_midis))
                if right_midis:
                    parts.append('R:' + '+'.join(_midi_to_name(m) for m in right_midis))
                note_str = '  '.join(parts) if parts else '—'
                t_str    = f'Target: {FREQUENCIES[target_idx]} Hz  ({note_str})'
                t_col    = FREQ_COLORS[target_idx]
            else:
                t_str = 'Target: none'
                t_col = DIM_COLOR
            mid_txt = font_md.render(t_str, True, t_col)
            screen.blit(mid_txt, (SCREEN_W // 2 - mid_txt.get_width() // 2,
                                  bar_mid_y - mid_txt.get_height() // 2))

            right_txt = font_sm.render(
                'C=calibrate  SPACE=start/stop  ←=left clench  →=right clench  select key → click=left  Shift+click=right  0=reset  ESC=quit',
                True, DIM_COLOR)
            screen.blit(right_txt, (SCREEN_W - right_txt.get_width() - 14,
                                    bar_mid_y - right_txt.get_height() // 2))

        pygame.display.flip()
        clock.tick(TARGET_FPS)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
SSVEP Piano Stimulus Presentation with LSL Marker Streaming
============================================================
Six phase-reversing checkerboard stimuli at 4, 6, 8, 10, 12, 15 Hz are
displayed above a piano keyboard.  Stimulus cycle-onset times are pushed
to a dedicated LSL marker outlet ("SSVEPMarkers") so they align precisely
with EEG data recorded from unicornlsl.py.

Controls
--------
SPACE    – Start / stop stimulation  (markers 252 / 253)
1–6      – Cue a target frequency     (marker 100 + freq)
ESC / Q  – Quit
"""

import math
import random
import string
import sys
import time

import numpy as np   
import pygame
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop


# ─── Display & Stimulus Config ────────────────────────────────────────────────

SCREEN_W      = 1440
SCREEN_H      = 900
FULLSCREEN    = False   # flip to True for actual experiments
TARGET_FPS    = 240     # high loop rate → accurate phase computation

FREQUENCIES   = [6, 7.5, 10, 12, 15, 20]           # Hz
NOTE_NAMES    = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4']
NOTE_MIDI     = [60,   62,   64,   65,   67,  69] # MIDI 60 = C4

# Per-frequency accent colours (borders / piano highlights / labels)
FREQ_COLORS = [
    (255,  70,  70),   # 4  Hz – Red
    (255, 165,   0),   # 6  Hz – Orange
    (120, 220,  80),   # 8  Hz – Green
    ( 80, 160, 255),   # 10 Hz – Blue
    (210,  80, 255),   # 12 Hz – Violet
    (255, 120, 180),   # 15 Hz – Pink
]

CHECKER_PX    = 15     # size of each checker square (pixels)
STIM_PX       = 180    # stimulus box edge length (pixels)
BORDER_PX     = 5      # normal border width; doubled for targeted stimulus

# ─── Piano Config ─────────────────────────────────────────────────────────────

PIANO_MIDI_START = 36   # C2
PIANO_MIDI_END   = 83   # B5  (4 full octaves)
PIANO_H_FRAC     = 0.28 # fraction of screen height occupied by keyboard

WHITE_SEMITONES  = {0, 2, 4, 5, 7, 9, 11}   # C D E F G A B
# Black-key centre position within an octave, in units of white-key width from C
BK_OFFSETS = {1: 1.0, 3: 2.0, 6: 4.0, 8: 5.0, 10: 6.0}

# ─── Colours ──────────────────────────────────────────────────────────────────

BG_COLOR      = (18,  18,  28)
STATUS_BG     = (28,  28,  42)
TEXT_COLOR    = (220, 220, 220)
DIM_COLOR     = ( 90,  90, 100)
STATUS_H      = 50     # pixels

# ─── LSL Marker Codes ─────────────────────────────────────────────────────────

# Stimulus cycle onset → marker value equals the frequency (4, 6, 8, 10, 12, 15)
MARKER_EXP_START   = 252
MARKER_EXP_STOP    = 253
MARKER_TARGET_BASE = 100   # target-cue marker = 100 + frequency


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _uid(n=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _draw_checkerboard(surface, rect, phase):
    """Phase-reversing black/white checkerboard inside *rect*.
    phase 0 → normal orientation; phase 1 → inverted.
    """
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


# ─── Piano ────────────────────────────────────────────────────────────────────

class Piano:
    """Draws a multi-octave piano keyboard and manages key highlights."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.highlights: dict[int, tuple] = {}   # midi → RGB

        n_white = sum(
            1 for m in range(PIANO_MIDI_START, PIANO_MIDI_END + 1)
            if m % 12 in WHITE_SEMITONES
        )
        self.wkw = rect.width / n_white   # white key width  (float)
        self.wkh = rect.height
        self.bkw = self.wkw * 0.58
        self.bkh = self.wkh * 0.62

        # Build white-key geometry
        self.white_keys: list[dict] = []
        w_idx = 0
        for midi in range(PIANO_MIDI_START, PIANO_MIDI_END + 1):
            if midi % 12 in WHITE_SEMITONES:
                self.white_keys.append({
                    'midi': midi,
                    'x':    rect.x + w_idx * self.wkw,
                })
                w_idx += 1

        wk_x = {wk['midi']: wk['x'] for wk in self.white_keys}

        # Build black-key geometry
        self.black_keys: list[dict] = []
        for midi in range(PIANO_MIDI_START, PIANO_MIDI_END + 1):
            s = midi % 12
            if s in BK_OFFSETS:
                c_midi = (midi // 12) * 12   # C of same octave
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
        # White keys (drawn first so black keys paint on top)
        for wk in self.white_keys:
            x   = int(wk['x'])
            w   = max(1, int(self.wkw))
            mid = wk['midi']
            col = self.highlights.get(mid, (235, 235, 235))
            pygame.draw.rect(surface, col,          (x, self.rect.y, w + 1, int(self.wkh)), border_top_left_radius=6, border_top_right_radius=6)
            pygame.draw.rect(surface, (50, 50, 50), (x, self.rect.y, w + 1, int(self.wkh)), 1, border_top_left_radius=6, border_top_right_radius=6)
            # Label every C
            if mid % 12 == 0:
                octave = mid // 12 - 1
                lbl = small_font.render(f'C{octave}', True, (80, 80, 80))
                surface.blit(lbl, (
                    x + w // 2 - lbl.get_width() // 2,
                    self.rect.y + self.wkh - lbl.get_height() - 5,
                ))

        # Black keys (painted over white keys)
        for bk in self.black_keys:
            x   = int(bk['x'])
            w   = max(1, int(self.bkw))
            mid = bk['midi']
            col = self.highlights.get(mid, (25, 25, 25))
            pygame.draw.rect(surface, col, (x, self.rect.y, w, int(self.bkh)), border_bottom_left_radius=4, border_bottom_right_radius=4)


# ─── Piano sound ────────────────────────────────────────────────────────────────────

def _build_piano_sound(midi_note: int, duration: float = 1.0,
                       sample_rate: int = 44100) -> pygame.mixer.Sound:
    """
    Synthesize a piano-like tone for *midi_note* using additive synthesis
    + an ADSR envelope.  Returns a ready-to-play pygame Sound.

    Frequency formula: f = 440 × 2^((midi − 69) / 12)
    """
    freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    n    = int(sample_rate * duration)
    t    = np.linspace(0.0, duration, n, endpoint=False)

    # Additive harmonics (approximate piano timbre)
    wave = (
        1.00 * np.sin(2 * np.pi * 1 * freq * t) +
        0.50 * np.sin(2 * np.pi * 2 * freq * t) +
        0.25 * np.sin(2 * np.pi * 3 * freq * t) +
        0.12 * np.sin(2 * np.pi * 4 * freq * t) +
        0.06 * np.sin(2 * np.pi * 5 * freq * t) +
        0.03 * np.sin(2 * np.pi * 6 * freq * t)
    )
    wave /= np.max(np.abs(wave))          # normalize to ±1

    # ADSR envelope
    atk = int(0.005 * sample_rate)        # 5 ms  – fast piano attack
    dec = int(0.150 * sample_rate)        # 150 ms decay
    rel = int(0.500 * sample_rate)        # 500 ms release tail
    sus = max(n - atk - dec - rel, 0)    # sustain fills the remainder
    sus_lvl = 0.60

    env = np.concatenate([
        np.linspace(0.0,     1.0,     atk),
        np.linspace(1.0,     sus_lvl, dec),
        np.full(sus,         sus_lvl),
        np.linspace(sus_lvl, 0.0,     rel),
    ])[:n]

    wave = (wave * env * 0.80 * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])   # pygame needs stereo
    return pygame.sndarray.make_sound(stereo)


def _get_sound(midi: int, cache: dict) -> pygame.mixer.Sound:
    if midi not in cache:
        cache[midi] = _build_piano_sound(midi)
    return cache[midi]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    pygame.init()

    # ── Audio ──────────────────────────────────────────────────────────────
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print('[Audio] Pre-building piano note sounds …')
    note_sounds: dict[int, pygame.mixer.Sound] = {}  # midi → Sound, built on demand
    print('[Audio] Ready.')

    flags = pygame.FULLSCREEN if FULLSCREEN else 0
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
    pygame.display.set_caption('SSVEP Piano – EEG Data Collection')
    clock = pygame.time.Clock()

    font_lg = pygame.font.SysFont('Arial', 28, bold=True)
    font_md = pygame.font.SysFont('Arial', 20)
    font_sm = pygame.font.SysFont('Arial', 14)

    # ── LSL prediction inlet (non-blocking) ───────────────────────────────
    pred_inlet = None
    pred_streams = resolve_byprop('name', 'SSVEPPredictions', timeout=1.0)
    if pred_streams:
        pred_inlet = StreamInlet(pred_streams[0])
        print('[GUI] Connected to SSVEPPredictions stream.')
    else:
        print('[GUI] No prediction stream found — running in manual mode only.')

    # ── LSL EMG control inlet (non-blocking) ──────────────────────────────────
    emg_inlet = None
    emg_streams = resolve_byprop('name', 'EMGControl', timeout=1.0)
    if emg_streams:
        emg_inlet = StreamInlet(emg_streams[0])
        print('[GUI] Connected to EMGControl stream.')
    else:
        print('[GUI] No EMG stream found — keyboard-only mode.')

    
    # ── LSL marker outlet ──────────────────────────────────────────────────
    m_info = StreamInfo(
        name='SSVEPMarkers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,            # irregular (event-driven)
        channel_format='float32',
        source_id=f'ssvep_{_uid()}',
    )
    marker_outlet = StreamOutlet(m_info)
    print('[GUI] Marker outlet "SSVEPMarkers" ready.')

    # ── Layout ────────────────────────────────────────────────────────────
    piano_h    = int(SCREEN_H * PIANO_H_FRAC)
    piano_rect = pygame.Rect(0, SCREEN_H - piano_h, SCREEN_W, piano_h)
    piano      = Piano(piano_rect)

    stim_area_top = STATUS_H
    stim_area_bot = piano_rect.y
    stim_cy       = (stim_area_top + stim_area_bot) // 2   # vertical centre

    # Spread stimuli evenly across the full screen width so they never overlap.
    # Connector lines drop diagonally to the actual piano key positions below.
    STIM_GAP  = 60   # pixels of clear space between adjacent stimulus boxes
    n         = len(FREQUENCIES)
    total_w   = n * STIM_PX + (n - 1) * STIM_GAP
    x_start   = (SCREEN_W - total_w) // 2 + STIM_PX // 2
    stim_cx   = [x_start + i * (STIM_PX + STIM_GAP) for i in range(n)]

    # Piano key anchor x-positions (used only for connector lines)
    key_cx = [int(piano.key_center_x(m)) for m in NOTE_MIDI]

    # Pre-build stimulus rects
    stim_rects = [
        pygame.Rect(cx - STIM_PX // 2, stim_cy - STIM_PX // 2, STIM_PX, STIM_PX)
        for cx in stim_cx
    ]

    # ── Application state ─────────────────────────────────────────────────
    running_stim   = False
    target_idx     = -1
    prev_phases    = [0] * len(FREQUENCIES)
    t0             = time.perf_counter()
    key_press_time = -1
    last_played_midi = -1
    selected_midi  = -1                    # ← add
    key_freq_map: dict[int, list[int]] = {}  # freq_index → [midi, midi] (up to 2 keys)

    while True:
        now = time.perf_counter() - t0

        # ── Event handling ────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                # Check if a stimulus box was clicked
                clicked_stim = None
                for i, rect in enumerate(stim_rects):
                    if rect.collidepoint(mx, my):
                        clicked_stim = i
                        break

                if clicked_stim is not None and selected_midi != -1:
                    keys = key_freq_map.get(clicked_stim, [])
                    if selected_midi in keys:
                        # Already assigned, do nothing
                        print(f'[GUI] {_midi_to_name(selected_midi)} already assigned to {FREQUENCIES[clicked_stim]} Hz')
                    elif len(keys) < 2:
                        keys.append(selected_midi)
                        key_freq_map[clicked_stim] = sorted(keys, key=lambda m: piano.key_center_x(m))
                        print(f'[GUI] Assigned {_midi_to_name(selected_midi)} to {FREQUENCIES[clicked_stim]} Hz')
                    else:
                        print(f'[GUI] {FREQUENCIES[clicked_stim]} Hz already has 2 keys — press 0 to reset')
                    selected_midi = -1

                elif clicked_stim is not None and selected_midi == -1:
                    print('[GUI] Click a piano key first, then a frequency box to assign it')

                else:
                    # Check if a piano key was clicked
                    clicked_note = _get_clicked_key(mx, my, piano)
                    if clicked_note is not None:
                        # Find the midi number
                        clicked_midi = -1
                        for wk in piano.white_keys:
                            if _midi_to_name(wk['midi']) == clicked_note:
                                clicked_midi = wk['midi']
                                break
                        for bk in piano.black_keys:
                            if _midi_to_name(bk['midi']) == clicked_note:
                                clicked_midi = bk['midi']
                                break

                        if clicked_midi != -1:
                            # Check if this midi is assigned to any frequency
                            assigned_freq = None
                            for freq_idx, keys in key_freq_map.items():
                                if clicked_midi in keys:
                                    assigned_freq = freq_idx
                                    break
                            if assigned_freq is not None:
                                _get_sound(clicked_midi, note_sounds).play()
                                key_press_time = time.perf_counter()
                                last_played_midi = clicked_midi 
                                target_idx = assigned_freq
                                print(f'[Mouse] Played: {clicked_note}')
                            else:
                                selected_midi = clicked_midi
                                print(f'[Mouse] Selected: {clicked_note} — now click a frequency box')

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_SPACE:
                    running_stim = not running_stim
                    if running_stim:
                        t0          = time.perf_counter()
                        now         = 0.0
                        prev_phases = [0] * len(FREQUENCIES)
                        marker_outlet.push_sample([float(MARKER_EXP_START)])
                        print(f'[GUI] {MARKER_EXP_START} → Experiment START')
                    else:
                        marker_outlet.push_sample([float(MARKER_EXP_STOP)])
                        print(f'[GUI] {MARKER_EXP_STOP} → Experiment STOP')

                if event.key == pygame.K_0:
                    key_freq_map = {}
                    selected_midi = -1
                    print('[GUI] All key assignments cleared')

                num_keys = [
                    pygame.K_1, pygame.K_2, pygame.K_3,
                    pygame.K_4, pygame.K_5, pygame.K_6,
                ]
                for i, k in enumerate(num_keys):
                    if event.key == k:
                        if selected_midi != -1:
                            keys = key_freq_map.get(i, [])
                            if selected_midi in keys:
                                print(f'[GUI] {_midi_to_name(selected_midi)} already assigned to {FREQUENCIES[i]} Hz')
                            elif len(keys) < 2:
                                keys.append(selected_midi)
                                key_freq_map[i] = sorted(keys, key=lambda m: piano.key_center_x(m))
                                print(f'[GUI] Assigned {_midi_to_name(selected_midi)} to {FREQUENCIES[i]} Hz')
                            else:
                                print(f'[GUI] {FREQUENCIES[i]} Hz already has 2 keys — press 0 to reset')
                            selected_midi = -1
                        else:
                            # Play both keys for this frequency
                            keys = key_freq_map.get(i, [])
                            for midi in keys:
                                _get_sound(midi, note_sounds).play()
                            if keys:
                                key_press_time = time.perf_counter()
                                last_played_midi = keys[0] if len(keys) == 1 else -1
                                target_idx = i

        # ── Compute phases & send onset markers ───────────────────────────
        phases = []
        for i, freq in enumerate(FREQUENCIES):
            # Phase toggles at `freq` Hz: 0 for first half-cycle, 1 for second
            phase = int(now * freq * 2) % 2
            phases.append(phase)
            # Rising edge of new cycle (1 → 0): one marker per cycle
            if running_stim and phase == 0 and prev_phases[i] != 0:
                marker_outlet.push_sample([float(freq)])

        if running_stim:
            prev_phases = phases[:]

        # ── Poll for incoming predictions ─────────────────────────────────
        if pred_inlet is not None:
            sample, _ = pred_inlet.pull_sample(timeout=0.0)
            if sample is not None:
                predicted_freq = sample[0]
                if predicted_freq in FREQUENCIES:
                    pred_idx = FREQUENCIES.index(predicted_freq)
                    keys = key_freq_map.get(pred_idx, [])
                    if keys:
                        target_idx = pred_idx   # just highlight, wait for EMG
                        last_played_midi = -1    # ← clear so no key darkens on prediction alone
                        key_press_time = -1      # ← clear so recently_played is False
                        print(f'[GUI] Predicted: {predicted_freq} Hz — waiting for EMG clench')

        # ── Poll EMG navigation ───────────────────────────────────────────────────
        if emg_inlet is not None:
            emg_sample, _ = emg_inlet.pull_sample(timeout=0.0)
            if emg_sample is not None:
                emg_val = emg_sample[0]
                if emg_val != 0.0:
                    # Find which frequency is currently targeted
                    if target_idx >= 0:
                        keys = key_freq_map.get(target_idx, [])
                        # keys are sorted left→right, so index 0 = left, 1 = right
                        if emg_val == -1.0 and len(keys) >= 1:
                            midi = keys[0]   # left clench → left key
                            _get_sound(midi, note_sounds).play()
                            key_press_time = time.perf_counter()
                            last_played_midi = midi  
                            print(f'[EMG] Left clench → {_midi_to_name(midi)}')
                        elif emg_val == 1.0 and len(keys) >= 2:
                            midi = keys[1]   # right clench → right key
                            _get_sound(midi, note_sounds).play()
                            key_press_time = time.perf_counter()
                            last_played_midi = midi  
                            print(f'[EMG] Right clench → {_midi_to_name(midi)}')
        
        # ── Draw ─────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Piano highlights: always show frequency colour on target keys
        recently_played = (time.perf_counter() - key_press_time) < 0.3
        piano.highlights = {}
        for freq_idx, keys in key_freq_map.items():
            for midi in keys:
                color = FREQ_COLORS[freq_idx]
                if recently_played:
                    if last_played_midi == midi:
                        color = tuple(max(0, c - 60) for c in color)
                    elif last_played_midi == -1 and freq_idx == target_idx:
                        color = tuple(max(0, c - 60) for c in color)
                piano.highlights[midi] = color
        if selected_midi != -1:
            if selected_midi % 12 in WHITE_SEMITONES:
                piano.highlights[selected_midi] = (255, 255, 255)  # white key → white
            else:
                piano.highlights[selected_midi] = (90, 90, 90)     # black key → dark gray
        piano.draw(screen, font_sm)

        # Connector lines: diagonal from stimulus bottom-centre → piano key top
        for freq_idx, keys in key_freq_map.items():
            cx = stim_cx[freq_idx]
            line_col = tuple(max(0, c - 80) for c in FREQ_COLORS[freq_idx])
            stim_bottom = stim_cy + STIM_PX // 2 + BORDER_PX + 2
            for midi in keys:
                kx = int(piano.key_center_x(midi))
                pygame.draw.line(screen, line_col, (cx, stim_bottom), (kx, piano_rect.y), 2)

        # Stimuli
        for i, rect in enumerate(stim_rects):
            # Checkerboard fill
            if running_stim:
                _draw_checkerboard(screen, rect, phases[i])
            else:
                # Idle: dim gray checkerboard (shows layout without flickering)
                _draw_checkerboard_idle(screen, rect)

            # Coloured border (thicker when this is the target)
            is_target  = (target_idx == i)
            bw         = BORDER_PX * 2 if is_target else BORDER_PX
            bord_rect  = pygame.Rect(rect.x - bw, rect.y - bw,
                                     rect.w + bw * 2, rect.h + bw * 2)
            pygame.draw.rect(screen, FREQ_COLORS[i], bord_rect, bw)

            # "TARGET" label above the border
            if is_target:
                tgt_txt = font_md.render('▼ TARGET ▼', True, FREQ_COLORS[i])
                screen.blit(tgt_txt, (
                    rect.centerx - tgt_txt.get_width() // 2,
                    rect.y - tgt_txt.get_height() - 28,
                ))

            # Frequency label above stimulus
            freq_txt = font_lg.render(f'{FREQUENCIES[i]} Hz', True, FREQ_COLORS[i])
            screen.blit(freq_txt, (
                rect.centerx - freq_txt.get_width() // 2,
                rect.y - freq_txt.get_height() - 4,
            ))

            # Note names below stimulus — show assigned keys if any
            assigned_keys = key_freq_map.get(i, [])
            if assigned_keys:
                note_label = ' / '.join(_midi_to_name(m) for m in assigned_keys)
            else:
                note_label = '—'
            note_txt = font_md.render(note_label, True, TEXT_COLOR)
            screen.blit(note_txt, (
                rect.centerx - note_txt.get_width() // 2,
                rect.bottom + 6,
            ))

        # ── Status bar ────────────────────────────────────────────────────
        pygame.draw.rect(screen, STATUS_BG, (0, 0, SCREEN_W, STATUS_H))

        state_str = 'RUNNING' if running_stim else 'STOPPED'
        state_col = (80, 255, 80) if running_stim else (255, 80, 80)
        s_left    = font_md.render(f'Stimulation: {state_str}', True, state_col)
        screen.blit(s_left, (16, STATUS_H // 2 - s_left.get_height() // 2))

        if target_idx >= 0:
            keys = key_freq_map.get(target_idx, [])
            note_str = ' / '.join(_midi_to_name(m) for m in keys) if keys else '—'
            t_str = f'Target: {FREQUENCIES[target_idx]} Hz  ({note_str})'
            t_col = FREQ_COLORS[target_idx]
        else:
            t_str = 'Target: none'
            t_col = DIM_COLOR
        s_mid = font_md.render(t_str, True, t_col)
        screen.blit(s_mid, (SCREEN_W // 2 - s_mid.get_width() // 2,
                             STATUS_H // 2 - s_mid.get_height() // 2))

        s_right = font_sm.render('SPACE = start/stop   click key → click box to assign   0 = reset   ESC = quit', True, DIM_COLOR)
        screen.blit(s_right, (SCREEN_W - s_right.get_width() - 14,
                               STATUS_H // 2 - s_right.get_height() // 2))

        pygame.display.flip()
        clock.tick(TARGET_FPS)


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


def _get_clicked_key(mx: int, my: int, piano: Piano) -> str | None:
    """
    Returns the note name (e.g. 'C4') of the piano key at pixel (mx, my),
    or None if the click is outside the keyboard.
    """
    if not piano.rect.collidepoint(mx, my):
        return None

    # Check black keys first — they sit on top of white keys
    for bk in piano.black_keys:
        x = int(bk['x'])
        w = max(1, int(piano.bkw))
        h = int(piano.bkh)
        if x <= mx <= x + w and piano.rect.y <= my <= piano.rect.y + h:
            return _midi_to_name(bk['midi'])

    # Then check white keys
    for wk in piano.white_keys:
        x = int(wk['x'])
        w = max(1, int(piano.wkw))
        if x <= mx <= x + w:
            return _midi_to_name(wk['midi'])

    return None


def _midi_to_name(midi: int) -> str:
    """Converts a MIDI number to a note name like 'C4'."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    return f'{note_names[midi % 12]}{octave}'

if __name__ == '__main__':
    main()

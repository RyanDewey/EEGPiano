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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    pygame.init()

    # ── Audio ──────────────────────────────────────────────────────────────
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print('[Audio] Pre-building piano note sounds …')
    note_sounds = [_build_piano_sound(m) for m in NOTE_MIDI]
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
    running_stim = False
    target_idx   = -1         # index into FREQUENCIES / NOTE_NAMES (-1 = none)
    prev_phases  = [0] * len(FREQUENCIES)
    t0           = time.perf_counter()
    key_press_time = -1

    while True:
        now = time.perf_counter() - t0

        # ── Event handling ────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                clicked_note = _get_clicked_key(mx, my, piano)
                if clicked_note is not None:
                    print(f'[Mouse] Key clicked: {clicked_note}')

            if event.type == pygame.KEYDOWN:

                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_SPACE:
                    running_stim = not running_stim
                    if running_stim:
                        t0           = time.perf_counter()
                        now          = 0.0
                        prev_phases  = [0] * len(FREQUENCIES)
                        marker_outlet.push_sample([float(MARKER_EXP_START)])
                        print(f'[GUI] {MARKER_EXP_START} → Experiment START')
                    else:
                        marker_outlet.push_sample([float(MARKER_EXP_STOP)])
                        print(f'[GUI] {MARKER_EXP_STOP} → Experiment STOP')

                # Keys 1–6 select/deselect target
                num_keys = [
                    pygame.K_1, pygame.K_2, pygame.K_3,
                    pygame.K_4, pygame.K_5, pygame.K_6,
                ]
                for i, k in enumerate(num_keys):
                    if event.key == k:
                        note_sounds[i].play()
                        key_press_time = time.perf_counter()
                        target_idx = i
                        key_press_time = time.perf_counter()
                        freq       = FREQUENCIES[i]
                        mv         = float(MARKER_TARGET_BASE + freq)
                        marker_outlet.push_sample([mv])
                        print(f'[GUI] {mv} → Target cue: {freq} Hz ({NOTE_NAMES[i]})')

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
            sample, _ = pred_inlet.pull_sample(timeout=0.0)  # non-blocking
            if sample is not None:
                predicted_freq = sample[0]
                if predicted_freq in FREQUENCIES:
                    pred_idx = FREQUENCIES.index(predicted_freq)
                    note_sounds[pred_idx].play()
                    key_press_time = time.perf_counter()
                    target_idx = pred_idx   # highlight the predicted key
                    print(f'[GUI] {predicted_freq} Hz → {NOTE_NAMES[pred_idx]}')

        # ── Poll EMG navigation ───────────────────────────────────────────────────
        if emg_inlet is not None:
            emg_sample, _ = emg_inlet.pull_sample(timeout=0.0)
            # EMG input currently disabled
        
        # ── Draw ─────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Piano highlights: always show frequency colour on target keys
        recently_played = (time.perf_counter() - key_press_time) < 0.3
        piano.highlights = {
            NOTE_MIDI[i]: tuple(max(0, c - 60) for c in FREQ_COLORS[i])
            if i == target_idx and recently_played else FREQ_COLORS[i]
            for i in range(len(FREQUENCIES))
        }
        piano.draw(screen, font_sm)

        # Connector lines: diagonal from stimulus bottom-centre → piano key top
        for i, cx in enumerate(stim_cx):
            line_col    = tuple(max(0, c - 80) for c in FREQ_COLORS[i])
            stim_bottom = stim_cy + STIM_PX // 2 + BORDER_PX + 2
            pygame.draw.line(screen, line_col,
                             (cx, stim_bottom), (key_cx[i], piano_rect.y), 2)

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

            # Note name below stimulus
            note_txt = font_md.render(NOTE_NAMES[i], True, TEXT_COLOR)
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
            t_str = f'Target: {FREQUENCIES[target_idx]} Hz  ({NOTE_NAMES[target_idx]})'
            t_col = FREQ_COLORS[target_idx]
        else:
            t_str = 'Target: none   (press 1–6 to select)'
            t_col = DIM_COLOR
        s_mid = font_md.render(t_str, True, t_col)
        screen.blit(s_mid, (SCREEN_W // 2 - s_mid.get_width() // 2,
                             STATUS_H // 2 - s_mid.get_height() // 2))

        s_right = font_sm.render('SPACE = start/stop   1–6 = target   ESC = quit',
                                  True, DIM_COLOR)
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

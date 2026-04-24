#!/usr/bin/env python3
"""
gui.py — Unified SSVEP Piano GUI
==================================
Supports four modes via sys.argv[1]:

  TRAIN   – Phase-reversing stimulus display for EEG data collection.
             Use Lab Recorder to capture the LSL streams to .xdf.
             SPACE starts/stops stimulation.

  TEST    – Automated 6-trial accuracy test (6 × 10 s).
             Sends cue markers; live_pipeline.py scores predictions.
             SPACE starts a new session.

  RUN     – Interactive piano: click keys to assign them to frequencies,
             classifier predictions highlight the target, EMG plays the note.
             SPACE starts/stops stimulus.

  COMPOSE – Record a note sequence played via EEG + EMG (or keyboard).
             Double-clench (both hands simultaneously) inserts a rest.
             ENTER plays back the recorded sequence.
             A music staff above the stimulus boxes shows the sequence.

Click the top-left mode label to cycle through all four modes.

Usage:
    python gui.py [TRAIN|TEST|RUN|COMPOSE]
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
_MODES = ["TRAIN", "TEST", "RUN", "COMPOSE"]
MODE = sys.argv[1].upper() if len(sys.argv) > 1 else "RUN"
if MODE not in _MODES:
    print(f"[GUI] Unknown mode '{MODE}' — defaulting to RUN")
    MODE = "RUN"

# ── Shared constants ──────────────────────────────────────────────────────────
FREQUENCIES      = [7.5, 10.0, 12.0]
FREQ_COLORS      = [(255, 165,   0), ( 80, 160, 255), (210,  80, 255)]
CHORD_COLORS     = [
    [(255, 165,   0), (255, 210, 110)],
    [( 80, 160, 255), (100, 230, 255)],
    [(210,  80, 255), (235, 150, 255)],
]
MAX_NOTES_PER_SLOT = 4
NOTE_MIDI_DEFAULT  = [55, 57, 59]

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
DIM_COLOR  = (90,  90, 100)
STATUS_H   = 50

MARKER_EXP_START    = 252
MARKER_EXP_STOP     = 253
MARKER_SESSION_END  = 255
MARKER_TARGET_BASE  = 100
MARKER_CALIB_START  = 280
MARKER_CALIB_DONE   = 281
MARKER_CALIB_CUE_BASE = 150

CALIB_FREQ_SEC   = 30
CALIB_BREAK_SEC  = 10

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

# ── COMPOSE constants ─────────────────────────────────────────────────────────
NOTE_DURATION     = 0.45   # seconds each note plays during playback
REST_DURATION     = 0.40   # seconds of silence for a rest
DOUBLE_CLENCH_MS  = 120    # max ms between two EMG events to count as simultaneous
STAFF_H           = 110    # height of the staff area in pixels
STAFF_LINE_GAP    = 14     # pixels between staff lines
STAFF_COLOR       = (160, 160, 180)
NOTE_HEAD_R       = 7      # radius of note-head circle
REST_COLOR        = (220, 180, 80)
NOTE_SCROLL_PX    = 36     # horizontal pixels per note slot on the staff


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uid(n=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _draw_checkerboard(surface, rect, phase):
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
    if not piano.rect.collidepoint(mx, my):
        return None
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


def make_trial_sequence():
    pool = FREQUENCIES * 2
    random.shuffle(pool)
    return pool


# ── Staff drawing ─────────────────────────────────────────────────────────────

# Map MIDI pitch to staff step relative to middle-C (MIDI 60 = C4).
# Staff step 0 = B4 (top of a typical treble-clef staff), each step = one diatonic note.
# We compute vertical position from a reference note on the staff.
# Reference: MIDI 64 = E4 sits on the first ledger line below the staff
# (bottom staff line = E4, step 0).
# Each diatonic step moves up by STAFF_LINE_GAP/2 pixels.

_CHROMATIC_TO_DIATONIC = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3,
                           6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6}

def _midi_to_staff_step(midi: int) -> int:
    """
    Returns diatonic staff step counted upward from E4 (MIDI 64).
    E4 = 0, F4 = 1, G4 = 2, A4 = 3, B4 = 4, C5 = 5, D5 = 6, E5 = 7 ...
    Negative values = below E4.
    """
    oct_ref  = 4
    note_ref = 4   # E in diatonic (C=0,D=1,E=2,F=3,G=4,A=5,B=6)
    pitch    = midi % 12
    octave   = midi // 12 - 1
    diatonic = _CHROMATIC_TO_DIATONIC[pitch]
    return (octave - oct_ref) * 7 + (diatonic - note_ref)


def _draw_staff(surface: pygame.Surface, staff_rect: pygame.Rect,
                sequence: list, playback_pos: int,
                font_sm: pygame.font.Font, font_md: pygame.font.Font):
    """
    Draw a 5-line treble-clef staff inside staff_rect, rendering the
    note sequence.  playback_pos is the index currently being played (-1 = none).
    """
    sx, sy, sw, sh = staff_rect.x, staff_rect.y, staff_rect.width, staff_rect.height

    # Background
    bg = pygame.Surface((sw, sh), pygame.SRCALPHA)
    bg.fill((12, 12, 22, 200))
    surface.blit(bg, (sx, sy))

    # 5 staff lines — centred vertically in the rect
    mid_y    = sy + sh // 2 + 4
    line_gap = STAFF_LINE_GAP
    line_ys  = [mid_y + (2 - i) * line_gap for i in range(5)]   # bottom to top

    for ly in line_ys:
        pygame.draw.line(surface, STAFF_COLOR, (sx + 2, ly), (sx + sw - 4, ly), 1)

    # ── Treble clef PNG ───────────────────────────────────────────────────────
    # The image is loaded once and cached in the function's attribute dict.
    if not hasattr(_draw_staff, '_clef_cache'):
        _draw_staff._clef_cache = {}
    if 'img' not in _draw_staff._clef_cache:
        try:
            raw = pygame.image.load(
                os.path.join(os.path.dirname(__file__), 'trebleclef.png')
            ).convert_alpha()
            _draw_staff._clef_cache['img'] = raw
        except Exception as e:
            print(f'[GUI] Could not load trebleclef.png: {e}')
            _draw_staff._clef_cache['img'] = None

    clef_img = _draw_staff._clef_cache['img']
    if clef_img is not None:
        # ── ADJUST HERE — clef size & position ───────────────────────────────
        clef_w      = 40          # width  to scale the PNG to (px)
        clef_h      = 90          # height to scale the PNG to (px)
        clef_left   = sx + 8     # horizontal position (px from left edge of screen)
        # Vertical: centres the clef image on the middle staff line (line_ys[2])
        # Increase the offset to move it down, decrease to move it up
        clef_top    = line_ys[2] - clef_h // 2 + 4   # +4 nudges down; change to taste
        # ─────────────────────────────────────────────────────────────────────
        scaled = pygame.transform.smoothscale(clef_img, (clef_w, clef_h))
        surface.blit(scaled, (clef_left, clef_top))

    # ── Label (top-left of staff area) ───────────────────────────────────────
    label = font_sm.render('COMPOSE  —  sequence', True, (130, 130, 160))
    surface.blit(label, (sx + 4, sy + 4))

    if not sequence:
        # ── ADJUST HERE — hint text position (shown when sequence is empty) ──
        hint_x = sx + 80           # horizontal offset from left edge of screen
        hint_y = sy + sh - 6      # vertical: pixels below the bottom of the staff rect
        # ─────────────────────────────────────────────────────────────────────
        hint = font_sm.render(
            'Play notes via EEG+EMG or keyboard (1–6).  '
            'Double-clench = rest.  ENTER = playback.  BKSP = undo.  DEL = clear.',
            True, (90, 90, 110))
        surface.blit(hint, (hint_x, hint_y))
        return

    # Note slots start after clef — push right of the clef image
    # ── ADJUST HERE — where notes begin horizontally on the staff ─────────────
    note_x0  = sx + 60   # increase if clef_w is larger and overlaps first note
    # ─────────────────────────────────────────────────────────────────────────
    max_visible = max(1, (sw - 70) // NOTE_SCROLL_PX)

    # Scroll so last notes are always visible
    start_idx = max(0, len(sequence) - max_visible)

    # Bottom staff line = E4 (step 0), each diatonic step = line_gap/2 px upward
    step_px = line_gap / 2.0
    e4_y    = line_ys[0]   # bottom line = E4

    for slot_i, seq_i in enumerate(range(start_idx, len(sequence))):
        event = sequence[seq_i]
        nx    = note_x0 + slot_i * NOTE_SCROLL_PX
        is_playing = (seq_i == playback_pos)
        is_last    = (seq_i == len(sequence) - 1 and playback_pos == -1)

        if event['type'] == 'rest':
            # Draw a quarter-rest rectangle symbol
            col = (255, 220, 80) if (is_playing or is_last) else REST_COLOR
            rw, rh = 8, 16
            rx = nx - rw // 2
            ry = int(mid_y - rh // 2)
            pygame.draw.rect(surface, col, (rx, ry, rw, rh), border_radius=2)
            pygame.draw.rect(surface, (255, 255, 255) if is_playing else col,
                             (rx, ry, rw, rh), 1, border_radius=2)
            # "Z" cross mark
            pygame.draw.line(surface, (18, 18, 28), (rx + 1, ry + 4), (rx + rw - 1, ry + 4), 2)
            pygame.draw.line(surface, (18, 18, 28), (rx + 1, ry + 9), (rx + rw - 1, ry + 9), 2)
            if is_playing:
                pygame.draw.circle(surface, (255, 255, 100), (nx, int(mid_y) - 28), 4)

        else:
            midis = event['midis']
            for midi in midis:
                step = _midi_to_staff_step(midi)
                ny   = int(e4_y - step * step_px)

                # Ledger lines if outside staff
                for ly in line_ys:
                    if abs(ny - ly) < 2:
                        break   # on a line — no ledger needed
                if ny < line_ys[-1] - 2:   # above top line
                    ll = line_ys[-1] - line_gap
                    while ll >= ny - 2:
                        pygame.draw.line(surface, STAFF_COLOR,
                                         (nx - NOTE_HEAD_R - 3, ll),
                                         (nx + NOTE_HEAD_R + 3, ll), 1)
                        ll -= line_gap
                elif ny > line_ys[0] + 2:   # below bottom line
                    ll = line_ys[0] + line_gap
                    while ll <= ny + 2:
                        pygame.draw.line(surface, STAFF_COLOR,
                                         (nx - NOTE_HEAD_R - 3, ll),
                                         (nx + NOTE_HEAD_R + 3, ll), 1)
                        ll += line_gap

                # Determine color by freq_idx if stored, else white
                freq_idx = event.get('freq_idx', -1)
                if is_playing:
                    col = (255, 255, 100)
                elif freq_idx >= 0:
                    col = FREQ_COLORS[freq_idx]
                else:
                    col = (220, 220, 220)

                # Filled note head
                pygame.draw.circle(surface, col, (nx, ny), NOTE_HEAD_R)
                # Stem up
                stem_top = ny - NOTE_HEAD_R - 22
                pygame.draw.line(surface, col, (nx + NOTE_HEAD_R - 1, ny),
                                 (nx + NOTE_HEAD_R - 1, stem_top), 2)

                # Sharp/flat accidental marker
                if midi % 12 not in WHITE_SEMITONES:
                    acc = font_sm.render('#', True, col)
                    surface.blit(acc, (nx - NOTE_HEAD_R - acc.get_width() - 1,
                                       ny - acc.get_height() // 2))

                if is_playing:
                    pygame.draw.circle(surface, (255, 255, 100, 100),
                                       (nx, ny), NOTE_HEAD_R + 4, 2)

            # Playback cursor glow
            if is_playing:
                pygame.draw.line(surface, (255, 255, 100, 160),
                                 (nx, line_ys[-1] - 6), (nx, line_ys[0] + 6), 1)

    # Bar line at end
    last_nx = note_x0 + (min(len(sequence), max_visible) - 1) * NOTE_SCROLL_PX
    if sequence:
        pygame.draw.line(surface, STAFF_COLOR,
                         (last_nx + NOTE_SCROLL_PX // 2, line_ys[-1]),
                         (last_nx + NOTE_SCROLL_PX // 2, line_ys[0]), 2)

    # Sequence count
    cnt = font_sm.render(f'{len(sequence)} events', True, (80, 80, 100))
    surface.blit(cnt, (sx + sw - cnt.get_width() - 6, sy + 4))


# ── Piano class ───────────────────────────────────────────────────────────────

class Piano:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.highlights: dict = {}

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
                self.white_keys.append({'midi': midi, 'x': rect.x + w_idx * self.wkw})
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
    global MODE

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
        _FADE_IN   = 0.8
        _HOLD      = 1.4
        _FADE_OUT  = 0.8
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
        pass

    font_xl = pygame.font.SysFont('Arial', 80, bold=True)
    font_lg = pygame.font.SysFont('Arial', 32, bold=True)
    font_md = pygame.font.SysFont('Arial', 22)
    font_sm = pygame.font.SysFont('Arial', 14)

    # ── LSL marker outlet ─────────────────────────────────────────────────────
    m_info = StreamInfo(
        name='SSVEPMarkers', type='Markers',
        channel_count=1, nominal_srate=0,
        channel_format='float32', source_id=f'ssvep_{_uid()}',
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

    BAR_H = 12
    BAR_Y = piano_rect.y - BAR_H - 8
    BAR_X = stim_rects[0].x
    BAR_W = stim_rects[-1].right - stim_rects[0].x

    # Staff rect (shown only in COMPOSE mode) — sits just above stim boxes
    staff_rect = pygame.Rect(
        0, STATUS_H,
        SCREEN_W, STAFF_H,
    )

    # ── Mode-specific init ────────────────────────────────────────────────────
    running_stim   = False
    target_idx     = -1
    prev_phases    = [0] * n
    t_stim         = time.perf_counter()

    in_calibration  = False
    calib_freq_idx  = 0
    calib_t         = 0.0

    chord_map: dict      = {}
    selected_midi: int   = -1
    key_press_time: float = -1
    pred_inlet  = None
    emg_inlet   = None

    state       = IDLE
    trial_seq   = make_trial_sequence()
    trial_idx_t = 0
    state_t     = 0.0

    # ── COMPOSE state ─────────────────────────────────────────────────────────
    # Each element: {'type': 'note', 'midis': [midi, ...], 'freq_idx': int}
    #            or {'type': 'rest'}
    compose_sequence: list = []
    compose_playback       = False      # True while playing back
    compose_pb_idx         = 0          # current playback index
    compose_pb_t           = 0.0        # time playback last advanced
    compose_pb_playing     = False      # True = currently playing a sound this slot
    # Double-clench detection
    _last_emg_left_t       = -999.0
    _last_emg_right_t      = -999.0

    if MODE == "RUN" or MODE == "COMPOSE":
        pred_streams = resolve_byprop('name', 'SSVEPPredictions', timeout=1.0)
        if pred_streams:
            pred_inlet = StreamInlet(pred_streams[0])
            print('[GUI] Connected to SSVEPPredictions stream.')
        else:
            print('[GUI] No prediction stream found.')

        emg_streams = resolve_byprop('name', 'EMGControl', timeout=1.0)
        if emg_streams:
            emg_inlet = StreamInlet(emg_streams[0])
            print('[GUI] Connected to EMGControl stream.')
        else:
            print('[GUI] No EMG stream found — keyboard-only mode.')

    # ── Mode label clickable rect ─────────────────────────────────────────────
    mode_label_rect = pygame.Rect(0, 0, 160, STATUS_H)

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

    def stop_stim():
        nonlocal running_stim
        running_stim = False
        marker_outlet.push_sample([float(MARKER_EXP_STOP)])

    def switch_mode():
        """Cycle to the next mode, resetting relevant state."""
        global MODE
        nonlocal running_stim, target_idx, state, trial_idx_t
        nonlocal selected_midi, chord_map, compose_sequence, compose_playback
        nonlocal pred_inlet, emg_inlet
        if running_stim:
            stop_stim()
        idx  = _MODES.index(MODE)
        MODE = _MODES[(idx + 1) % len(_MODES)]
        pygame.display.set_caption(f'SSVEP Piano — {MODE}')
        running_stim      = False
        target_idx        = -1
        selected_midi     = -1
        compose_sequence  = []
        compose_playback  = False
        state             = IDLE
        trial_idx_t       = 0
        print(f'[GUI] Switched to {MODE} mode')

        # Connect streams when entering RUN or COMPOSE
        if MODE in ("RUN", "COMPOSE") and pred_inlet is None:
            ps = resolve_byprop('name', 'SSVEPPredictions', timeout=0.5)
            if ps:
                pred_inlet = StreamInlet(ps[0])
        if MODE in ("RUN", "COMPOSE") and emg_inlet is None:
            es = resolve_byprop('name', 'EMGControl', timeout=0.5)
            if es:
                emg_inlet = StreamInlet(es[0])

    def compose_add_note(midis: list, freq_idx: int = -1):
        """Append a note event to the compose sequence (max 35 events)."""
        if not midis:
            return
        if len(compose_sequence) >= 35:
            print('[COMPOSE] Sequence full (35 events max)')
            return
        compose_sequence.append({'type': 'note', 'midis': list(midis), 'freq_idx': freq_idx})
        print(f'[COMPOSE] + note {[_midi_to_name(m) for m in midis]}')

    def compose_add_rest():
        if len(compose_sequence) >= 35:
            print('[COMPOSE] Sequence full (35 events max)')
            return
        compose_sequence.append({'type': 'rest'})
        print('[COMPOSE] + rest')

    def compose_start_playback():
        nonlocal compose_playback, compose_pb_idx, compose_pb_t, compose_pb_playing
        if not compose_sequence:
            return
        compose_playback    = True
        compose_pb_idx      = 0
        compose_pb_t        = time.perf_counter()
        compose_pb_playing  = False
        print('[COMPOSE] Playback started')

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        now_abs = time.perf_counter()
        elapsed = now_abs - state_t

        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    if running_stim:
                        stop_stim()
                    pygame.quit(); sys.exit()

                # ── COMPOSE-specific keys ─────────────────────────────────────
                if MODE == "COMPOSE":
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        if not compose_playback:
                            compose_start_playback()

                    elif event.key == pygame.K_BACKSPACE:
                        if compose_sequence and not compose_playback:
                            removed = compose_sequence.pop()
                            print(f'[COMPOSE] Undo last event ({removed["type"]})')

                    elif event.key == pygame.K_DELETE:
                        if not compose_playback:
                            compose_sequence.clear()
                            print('[COMPOSE] Sequence cleared')

                    elif event.key == pygame.K_SPACE:
                        running_stim = not running_stim
                        if running_stim:
                            start_stim()
                        else:
                            stop_stim()

                    elif event.key == pygame.K_0:
                        selected_midi = -1
                        print('[COMPOSE] Key selection cleared')

                    # Keyboard simulation: same 1-6 as RUN mode
                    sim_keys = [pygame.K_1, pygame.K_2, pygame.K_3,
                                pygame.K_4, pygame.K_5, pygame.K_6]
                    for ki, k in enumerate(sim_keys):
                        if event.key == k:
                            fi   = ki // 2
                            side = ki % 2
                            midis = chord_map.get((fi, side), [])
                            if midis:
                                target_idx = fi
                                for midi in midis:
                                    _get_sound(midi, note_sounds).play()
                                key_press_time = time.perf_counter()
                                if not compose_playback:
                                    compose_add_note(midis, fi)
                            elif not compose_playback:
                                # No chord assigned — add a generic rest as placeholder
                                compose_add_rest()

                    # R = insert rest manually
                    if event.key == pygame.K_r and not compose_playback:
                        compose_add_rest()

                else:
                    # ── Non-COMPOSE key handling ──────────────────────────────
                    if event.key == pygame.K_c:
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

                    if event.key == pygame.K_SPACE:
                        if in_calibration:
                            pass
                        elif MODE == "TRAIN":
                            running_stim = not running_stim
                            if running_stim:
                                start_stim()
                            else:
                                stop_stim()
                        elif MODE == "TEST":
                            if state in (IDLE, DONE):
                                new_seq = make_trial_sequence()
                                trial_seq[:] = new_seq
                                trial_idx_t  = 0
                                target_idx   = -1
                                send_target(FREQUENCIES.index(trial_seq[0]))
                                start_state(COUNTDOWN)
                        elif MODE == "RUN":
                            running_stim = not running_stim
                            if running_stim:
                                start_stim()
                            else:
                                stop_stim()

                    if MODE == "RUN":
                        if event.key == pygame.K_0:
                            chord_map     = {}
                            selected_midi = -1

                        sim_keys = [pygame.K_1, pygame.K_2, pygame.K_3,
                                    pygame.K_4, pygame.K_5, pygame.K_6]
                        for ki, k in enumerate(sim_keys):
                            if event.key == k:
                                fi   = ki // 2
                                side = ki % 2
                                midis = chord_map.get((fi, side), [])
                                target_idx = fi
                                if midis:
                                    for midi in midis:
                                        _get_sound(midi, note_sounds).play()
                                    key_press_time = time.perf_counter()

            # ── Mode label click (top-left corner) ────────────────────────────
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mode_label_rect.collidepoint(mx, my):
                    switch_mode()
                    continue   # skip further mouse handling this frame

            if event.type == pygame.MOUSEBUTTONDOWN and MODE in ("RUN", "COMPOSE"):
                mx, my = event.pos
                shift  = pygame.key.get_mods() & pygame.KMOD_SHIFT

                clicked_stim = None
                for i, rect in enumerate(stim_rects):
                    if rect.collidepoint(mx, my):
                        clicked_stim = i
                        break

                if clicked_stim is not None and selected_midi != -1:
                    side      = 1 if shift else 0
                    side_name = 'right' if side == 1 else 'left'
                    slot      = chord_map.get((clicked_stim, side), [])
                    if selected_midi in slot:
                        pass
                    elif len(slot) >= MAX_NOTES_PER_SLOT:
                        print(f'[GUI] Chord full ({MAX_NOTES_PER_SLOT} notes max) — press 0 to reset')
                    else:
                        slot.append(selected_midi)
                        chord_map[(clicked_stim, side)] = slot
                        names = '+'.join(_midi_to_name(m) for m in slot)
                        print(f'[GUI] {FREQUENCIES[clicked_stim]} Hz {side_name}: {names}')
                    selected_midi = -1

                elif clicked_stim is None:
                    clicked_note = _get_clicked_key(mx, my, piano)
                    if clicked_note is not None:
                        clicked_midi = -1
                        for wk in piano.white_keys:
                            if _midi_to_name(wk['midi']) == clicked_note:
                                clicked_midi = wk['midi']; break
                        if clicked_midi == -1:
                            for bk in piano.black_keys:
                                if _midi_to_name(bk['midi']) == clicked_note:
                                    clicked_midi = bk['midi']; break

                        if clicked_midi != -1:
                            found_freq = None
                            for (fi, si), midis in chord_map.items():
                                if clicked_midi in midis:
                                    found_freq = fi; break

                            if found_freq is not None:
                                _get_sound(clicked_midi, note_sounds).play()
                                key_press_time = time.perf_counter()
                                target_idx = found_freq
                                if MODE == "COMPOSE" and not compose_playback:
                                    compose_add_note([clicked_midi], found_freq)
                            else:
                                selected_midi = clicked_midi

        # ── Calibration state machine ─────────────────────────────────────────
        if in_calibration:
            calib_elapsed = time.perf_counter() - calib_t
            on_break = not running_stim and calib_freq_idx < n

            if on_break:
                if calib_elapsed >= CALIB_BREAK_SEC:
                    target_idx = calib_freq_idx
                    start_stim()
                    calib_t = time.perf_counter()
                    marker_outlet.push_sample([float(MARKER_CALIB_CUE_BASE + FREQUENCIES[calib_freq_idx])])
            else:
                if calib_elapsed >= CALIB_FREQ_SEC:
                    calib_freq_idx += 1
                    if calib_freq_idx >= n:
                        stop_stim()
                        marker_outlet.push_sample([float(MARKER_CALIB_DONE)])
                        in_calibration = False
                        calib_freq_idx = 0
                        target_idx     = -1
                    else:
                        stop_stim()
                        target_idx = -1
                        calib_t    = time.perf_counter()

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
                        start_state(DONE)
                    else:
                        start_state(PAUSE)
            elif state == PAUSE:
                if elapsed >= PAUSE_SECS:
                    send_target(FREQUENCIES.index(trial_seq[trial_idx_t]))
                    start_state(COUNTDOWN)

        # ── Phase computation ─────────────────────────────────────────────────
        stim_now = time.perf_counter() - t_stim
        phases   = []
        for i, freq in enumerate(FREQUENCIES):
            phase = int(stim_now * freq * 2) % 2
            phases.append(phase)
            if running_stim and phase == 0 and prev_phases[i] != 0:
                marker_outlet.push_sample([float(freq)])
        if running_stim:
            prev_phases = phases[:]

        # ── RUN: poll prediction & EMG ────────────────────────────────────────
        if MODE == "RUN" and pred_inlet is not None:
            sample, _ = pred_inlet.pull_sample(timeout=0.0)
            if sample is not None:
                predicted_freq = sample[0]
                if predicted_freq in FREQUENCIES:
                    pred_idx  = FREQUENCIES.index(predicted_freq)
                    has_notes = any(chord_map.get((pred_idx, s), []) for s in (0, 1))
                    if has_notes:
                        target_idx = pred_idx

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

        # ── COMPOSE: poll prediction & EMG ───────────────────────────────────
        if MODE == "COMPOSE":
            if pred_inlet is not None:
                sample, _ = pred_inlet.pull_sample(timeout=0.0)
                if sample is not None:
                    predicted_freq = sample[0]
                    if predicted_freq in FREQUENCIES:
                        pred_idx  = FREQUENCIES.index(predicted_freq)
                        has_notes = any(chord_map.get((pred_idx, s), []) for s in (0, 1))
                        if has_notes:
                            target_idx = pred_idx

            if emg_inlet is not None and not compose_playback:
                emg_sample, _ = emg_inlet.pull_sample(timeout=0.0)
                if emg_sample is not None:
                    emg_val = emg_sample[0]
                    now_t   = time.perf_counter()

                    if emg_val == -1.0:   # left clench
                        _last_emg_left_t = now_t
                    elif emg_val == 1.0:  # right clench
                        _last_emg_right_t = now_t

                    # Double-clench detection
                    dt = abs(_last_emg_left_t - _last_emg_right_t)
                    if dt < DOUBLE_CLENCH_MS / 1000.0 and emg_val != 0.0:
                        # Both hands clenched near-simultaneously → rest
                        _last_emg_left_t  = -999.0
                        _last_emg_right_t = -999.0
                        compose_add_rest()
                    elif emg_val != 0.0 and target_idx >= 0:
                        side  = 0 if emg_val == -1.0 else 1
                        midis = chord_map.get((target_idx, side), [])
                        if midis:
                            for midi in midis:
                                _get_sound(midi, note_sounds).play()
                            key_press_time = time.perf_counter()
                            compose_add_note(midis, target_idx)

            # ── COMPOSE playback tick ─────────────────────────────────────────
            if compose_playback and compose_sequence:
                ev          = compose_sequence[compose_pb_idx]
                slot_dur    = REST_DURATION if ev['type'] == 'rest' else NOTE_DURATION
                slot_elapsed = now_abs - compose_pb_t

                if not compose_pb_playing:
                    # Fire sound at start of slot
                    if ev['type'] == 'note':
                        for midi in ev['midis']:
                            _get_sound(midi, note_sounds).play()
                    compose_pb_playing = True

                if slot_elapsed >= slot_dur:
                    compose_pb_idx += 1
                    compose_pb_t    = now_abs
                    compose_pb_playing = False
                    if compose_pb_idx >= len(compose_sequence):
                        compose_playback = False
                        compose_pb_idx   = 0
                        print('[COMPOSE] Playback complete')

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # ── Staff (COMPOSE only) ──────────────────────────────────────────────
        pb_idx_display = compose_pb_idx if compose_playback else -1
        if MODE == "COMPOSE":
            _draw_staff(screen, staff_rect, compose_sequence,
                        pb_idx_display, font_sm, font_md)

        # Adjust stim_cy when staff is shown so boxes don't overlap it
        effective_stim_cy = stim_cy
        if MODE == "COMPOSE":
            staff_bottom     = staff_rect.y + staff_rect.height
            stim_top         = effective_stim_cy - STIM_PX // 2
            if stim_top < staff_bottom + 10:
                effective_stim_cy = staff_bottom + 10 + STIM_PX // 2

        effective_stim_rects = [
            pygame.Rect(cx - STIM_PX // 2, effective_stim_cy - STIM_PX // 2, STIM_PX, STIM_PX)
            for cx in stim_cx
        ]
        effective_stim_bottom = effective_stim_cy + STIM_PX // 2 + BORDER_PX + 2

        # Piano highlights
        piano.highlights = {}
        if MODE in ("TRAIN", "TEST"):
            for i in range(n):
                piano.highlights[NOTE_MIDI_DEFAULT[i]] = FREQ_COLORS[i]
        elif MODE in ("RUN", "COMPOSE"):
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

            # Highlight notes currently playing during playback
            if MODE == "COMPOSE" and compose_playback and compose_sequence:
                pb_ev = compose_sequence[compose_pb_idx] if compose_pb_idx < len(compose_sequence) else None
                if pb_ev and pb_ev['type'] == 'note':
                    for midi in pb_ev['midis']:
                        piano.highlights[midi] = (255, 255, 100)

        piano.draw(screen, font_sm)

        # Connector lines
        if MODE in ("TRAIN", "TEST"):
            for i, cx in enumerate(stim_cx):
                line_col = tuple(max(0, c - 80) for c in FREQ_COLORS[i])
                pygame.draw.line(screen, line_col,
                                 (cx, effective_stim_bottom), (key_cx_default[i], piano_rect.y), 2)
        elif MODE in ("RUN", "COMPOSE"):
            for (freq_idx, side), midis in chord_map.items():
                cx       = stim_cx[freq_idx]
                line_col = tuple(max(0, c - 80) for c in CHORD_COLORS[freq_idx][side])
                for midi in midis:
                    kx = int(piano.key_center_x(midi))
                    pygame.draw.line(screen, line_col,
                                     (cx, effective_stim_bottom), (kx, piano_rect.y), 2)

        # Stimulus boxes
        for i, rect in enumerate(effective_stim_rects):
            is_target = (target_idx == i)

            if running_stim:
                _draw_checkerboard(screen, rect, phases[i])
            else:
                _draw_checkerboard_idle(screen, rect)

            bw        = BORDER_PX * 2 if is_target else BORDER_PX
            bord_rect = pygame.Rect(rect.x - bw, rect.y - bw,
                                    rect.w + bw * 2, rect.h + bw * 2)
            pygame.draw.rect(screen, FREQ_COLORS[i], bord_rect, bw)

            show_target_indicator = False
            if MODE == "TEST" and is_target and state in (COUNTDOWN, RUNNING):
                show_target_indicator = True
            elif MODE in ("RUN", "COMPOSE") and is_target:
                show_target_indicator = True

            if show_target_indicator:
                gr = pygame.Rect(bord_rect.x - 4, bord_rect.y - 4,
                                 bord_rect.w + 8, bord_rect.h + 8)
                pygame.draw.rect(screen, (255, 255, 255), gr, 3)
                arrow = font_lg.render('▼ LOOK HERE ▼', True, FREQ_COLORS[i])
                screen.blit(arrow, (rect.centerx - arrow.get_width() // 2,
                                    rect.y - arrow.get_height() - 30))

            freq_txt = font_md.render(f'{FREQUENCIES[i]} Hz', True, FREQ_COLORS[i])
            screen.blit(freq_txt, (
                rect.centerx - freq_txt.get_width() // 2,
                rect.y - freq_txt.get_height() - 4,
            ))

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
                pygame.draw.rect(screen, (50, 50, 60), (BAR_X, BAR_Y, BAR_W, BAR_H), border_radius=4)
                if fill > 0:
                    col = FREQ_COLORS[target_idx] if target_idx >= 0 else TEXT_COLOR
                    pygame.draw.rect(screen, col, (BAR_X, BAR_Y, int(BAR_W * fill), BAR_H), border_radius=4)
                secs_left = max(0.0, TRIAL_SEC - elapsed)
                timer_txt = font_sm.render(f'{secs_left:.1f}s remaining', True, DIM_COLOR)
                screen.blit(timer_txt, (BAR_X + BAR_W // 2 - timer_txt.get_width() // 2,
                                        BAR_Y - timer_txt.get_height() - 2))
            elif state == COUNTDOWN:
                cd = max(0, COUNTDOWN_SECS - int(elapsed))
                cd_surf = font_xl.render(str(cd) if cd > 0 else 'GO', True, (255, 255, 100))
                screen.blit(cd_surf, (SCREEN_W // 2 - cd_surf.get_width() // 2,
                                      effective_stim_cy - cd_surf.get_height() // 2))
            elif state == DONE:
                overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))
                screen.blit(overlay, (0, 0))
                done_txt = font_lg.render('Session Complete — accuracy results in terminal', True, (100, 255, 100))
                restart  = font_md.render('Press SPACE to run again   |   ESC to quit', True, DIM_COLOR)
                screen.blit(done_txt, (SCREEN_W // 2 - done_txt.get_width() // 2, SCREEN_H // 2 - 40))
                screen.blit(restart,  (SCREEN_W // 2 - restart.get_width() // 2, SCREEN_H // 2 + 10))

        # ── TRAIN overlay ─────────────────────────────────────────────────────
        if MODE == "TRAIN" and not running_stim:
            panel_w, panel_h = 520, 160
            panel_x = SCREEN_W // 2 - panel_w // 2
            panel_y = effective_stim_cy + STIM_PX // 2 + 30
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
                screen.blit(surf, (panel_x + panel_w // 2 - surf.get_width() // 2,
                                   panel_y + 14 + li * 34))

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

            pygame.draw.rect(screen, (50, 50, 60), (BAR_X, BAR_Y, BAR_W, BAR_H), border_radius=4)
            if fill > 0:
                pygame.draw.rect(screen, col, (BAR_X, BAR_Y, int(BAR_W * fill), BAR_H), border_radius=4)
            timer_txt = font_sm.render(label, True, col)
            screen.blit(timer_txt, (BAR_X + BAR_W // 2 - timer_txt.get_width() // 2,
                                    BAR_Y - timer_txt.get_height() - 4))

        # ── Status bar ────────────────────────────────────────────────────────
        pygame.draw.rect(screen, STATUS_BG, (0, 0, SCREEN_W, STATUS_H))
        bar_mid_y = STATUS_H // 2

        # Mode label — clickable, top-left, with hover hint
        mx_now, my_now = pygame.mouse.get_pos()
        mode_hover     = mode_label_rect.collidepoint(mx_now, my_now)
        mode_colors    = {
            'TRAIN':   (255, 200,  60),
            'TEST':    (100, 200, 255),
            'RUN':     TEXT_COLOR,
            'COMPOSE': (120, 255, 160),
        }
        mode_col = mode_colors.get(MODE, TEXT_COLOR)
        if mode_hover:
            # Draw subtle highlight box behind label
            pygame.draw.rect(screen, (40, 40, 60), mode_label_rect, border_radius=4)
            next_mode = _MODES[(_MODES.index(MODE) + 1) % len(_MODES)]
            hint = font_sm.render(f'→ {next_mode}', True, (100, 100, 120))
            screen.blit(hint, (8, STATUS_H - hint.get_height() - 3))
        left_txt = font_md.render(f'{MODE} MODE', True, mode_col)
        screen.blit(left_txt, (16, bar_mid_y - left_txt.get_height() // 2))

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
            calib_lbl = font_md.render('CALIBRATING', True, (255, 200, 60))
            mid_txt   = font_md.render(msg_c, True, col_c)
            right_txt = font_sm.render('ESC=quit', True, DIM_COLOR)
            screen.blit(calib_lbl, (16 + left_txt.get_width() + 24,
                                    bar_mid_y - calib_lbl.get_height() // 2))
            screen.blit(mid_txt,   (SCREEN_W // 2 - mid_txt.get_width() // 2,
                                    bar_mid_y - mid_txt.get_height() // 2))
            screen.blit(right_txt, (SCREEN_W - right_txt.get_width() - 14,
                                    bar_mid_y - right_txt.get_height() // 2))

        elif MODE == "TRAIN":
            state_str = 'RUNNING' if running_stim else 'STOPPED'
            state_col = (80, 255, 80) if running_stim else (255, 80, 80)
            mid_txt   = font_md.render(f'Stimulation: {state_str}', True, state_col)
            right_txt = font_sm.render('SPACE=start/stop  C=calibrate  ESC=quit', True, DIM_COLOR)
            screen.blit(mid_txt,   (SCREEN_W // 2 - mid_txt.get_width() // 2,
                                    bar_mid_y - mid_txt.get_height() // 2))
            screen.blit(right_txt, (SCREEN_W - right_txt.get_width() - 14,
                                    bar_mid_y - right_txt.get_height() // 2))

        elif MODE == "TEST":
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
                msg     = (f'Trial {trial_idx_t}/{N_TRIALS} done  —  Next: {nf} Hz' if nf else 'Last trial done')
                msg_col = TEXT_COLOR
            else:
                msg     = f'Done — {N_TRIALS} trials complete'
                msg_col = (100, 255, 100)
            mid_txt   = font_md.render(msg, True, msg_col)
            right_txt = font_sm.render('C=calibrate  SPACE=start  ESC=quit', True, DIM_COLOR)
            screen.blit(mid_txt,   (SCREEN_W // 2 - mid_txt.get_width() // 2,
                                    bar_mid_y - mid_txt.get_height() // 2))
            screen.blit(right_txt, (SCREEN_W - right_txt.get_width() - 14,
                                    bar_mid_y - right_txt.get_height() // 2))

        elif MODE == "RUN":
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
            mid_txt   = font_md.render(t_str, True, t_col)
            right_txt = font_sm.render(
                'C=calibrate  SPACE=start/stop  select key → click=L  Shift+click=R  0=reset  ESC=quit',
                True, DIM_COLOR)
            screen.blit(mid_txt,   (16 + left_txt.get_width() + 24,
                                    bar_mid_y - mid_txt.get_height() // 2))
            screen.blit(right_txt, (SCREEN_W - right_txt.get_width() - 14,
                                    bar_mid_y - right_txt.get_height() // 2))

        elif MODE == "COMPOSE":
            n_ev    = len(compose_sequence)
            n_notes = sum(1 for e in compose_sequence if e['type'] == 'note')
            n_rests = n_ev - n_notes
            if compose_playback:
                msg     = f'▶  Playing back  ({compose_pb_idx + 1}/{n_ev})'
                msg_col = (120, 255, 160)
            elif n_ev == 0:
                msg     = 'Play notes to record.  Assign chords below first.'
                msg_col = DIM_COLOR
            else:
                full_str = '  ⚠ FULL' if n_ev >= 35 else f' / 35'
                msg     = f'{n_notes} note{"s" if n_notes != 1 else ""}  +  {n_rests} rest{"s" if n_rests != 1 else ""}  ({n_ev}{full_str})'
                msg_col = (120, 255, 160)
            mid_txt   = font_md.render(msg, True, msg_col)
            right_txt = font_sm.render(
                'ENTER=play  BKSP=undo  DEL=clear  R=rest  0=deselect key  SPACE=stim  1–6=sim  ESC=quit',
                True, DIM_COLOR)
            screen.blit(mid_txt,   (16 + left_txt.get_width() + 24,
                                    bar_mid_y - mid_txt.get_height() // 2))
            screen.blit(right_txt, (SCREEN_W - right_txt.get_width() - 14,
                                    bar_mid_y - right_txt.get_height() // 2))

        pygame.display.flip()
        clock.tick(TARGET_FPS)


if __name__ == '__main__':
    main()

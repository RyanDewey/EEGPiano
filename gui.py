#!/usr/bin/env python3
"""
SSVEP Accuracy Test — 7.5, 10, 12 Hz
======================================
Runs a 60-second automated accuracy test (6 trials × 10 s).
Each frequency is tested exactly twice; live_pipeline.py scores
predictions against the cue marker and prints overall accuracy at the end.

Controls
--------
SPACE   – Start test / restart after done
ESC / Q – Quit
"""

import math
import random
import string
import sys
import time

import pygame
from pylsl import StreamInfo, StreamOutlet

# ─── Display ──────────────────────────────────────────────────────────────────

SCREEN_W   = 1440
SCREEN_H   = 900
FULLSCREEN = False
TARGET_FPS = 240

# ─── Stimulus config ──────────────────────────────────────────────────────────

FREQUENCIES = [7.5, 10, 12]
NOTE_NAMES  = ['G3', 'A3', 'B3']
NOTE_MIDI   = [55,   57,   59]

FREQ_COLORS = [
    (255, 165,   0),   # 7.5 Hz – Orange
    ( 80, 160, 255),   # 10  Hz – Blue
    (210,  80, 255),   # 12  Hz – Violet
]

CHECKER_PX = 15
STIM_PX    = 210
STIM_GAP   = 100
BORDER_PX  = 5

# ─── Test protocol ────────────────────────────────────────────────────────────

N_TRIALS       = 6
TRIAL_SEC      = 10.0
COUNTDOWN_SECS = 3
PAUSE_SECS     = 2

MARKER_EXP_START   = 252
MARKER_EXP_STOP    = 253
MARKER_SESSION_END = 255
MARKER_TARGET_BASE = 100   # marker = 100 + freq

# States
IDLE = 'idle'; COUNTDOWN = 'countdown'; RUNNING = 'running'
PAUSE = 'pause'; DONE = 'done'

# ─── Piano ────────────────────────────────────────────────────────────────────

PIANO_MIDI_START = 36
PIANO_MIDI_END   = 83
PIANO_H_FRAC     = 0.28
WHITE_SEMITONES  = {0, 2, 4, 5, 7, 9, 11}
BK_OFFSETS       = {1: 1.0, 3: 2.0, 6: 4.0, 8: 5.0, 10: 6.0}

BG_COLOR   = (18,  18,  28)
STATUS_BG  = (28,  28,  42)
TEXT_COLOR = (220, 220, 220)
DIM_COLOR  = ( 90,  90, 100)
STATUS_H   = 50


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _uid(n=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _draw_checkerboard(surface, rect, phase):
    x0, y0, w, h = rect.x, rect.y, rect.width, rect.height
    for r in range(math.ceil(h / CHECKER_PX)):
        for c in range(math.ceil(w / CHECKER_PX)):
            px = x0 + c * CHECKER_PX;  py = y0 + r * CHECKER_PX
            pw = min(CHECKER_PX, x0 + w - px);  ph = min(CHECKER_PX, y0 + h - py)
            if pw <= 0 or ph <= 0: continue
            white = (r + c + phase) % 2 == 0
            pygame.draw.rect(surface, (255,255,255) if white else (0,0,0), (px,py,pw,ph))


def _draw_checkerboard_idle(surface, rect):
    x0, y0, w, h = rect.x, rect.y, rect.width, rect.height
    for r in range(math.ceil(h / CHECKER_PX)):
        for c in range(math.ceil(w / CHECKER_PX)):
            px = x0 + c * CHECKER_PX;  py = y0 + r * CHECKER_PX
            pw = min(CHECKER_PX, x0 + w - px);  ph = min(CHECKER_PX, y0 + h - py)
            if pw <= 0 or ph <= 0: continue
            col = (70,70,80) if (r+c)%2==0 else (45,45,55)
            pygame.draw.rect(surface, col, (px,py,pw,ph))


class Piano:
    def __init__(self, rect):
        self.rect = rect
        self.highlights = {}
        n_white = sum(1 for m in range(PIANO_MIDI_START, PIANO_MIDI_END+1)
                      if m%12 in WHITE_SEMITONES)
        self.wkw = rect.width / n_white
        self.wkh = rect.height
        self.bkw = self.wkw * 0.58
        self.bkh = self.wkh * 0.62
        self.white_keys = []
        wi = 0
        for midi in range(PIANO_MIDI_START, PIANO_MIDI_END+1):
            if midi%12 in WHITE_SEMITONES:
                self.white_keys.append({'midi': midi, 'x': rect.x + wi*self.wkw})
                wi += 1
        wk_x = {wk['midi']: wk['x'] for wk in self.white_keys}
        self.black_keys = []
        for midi in range(PIANO_MIDI_START, PIANO_MIDI_END+1):
            s = midi%12
            if s in BK_OFFSETS:
                cm = (midi//12)*12
                if cm in wk_x:
                    self.black_keys.append({'midi': midi,
                                            'x': wk_x[cm] + BK_OFFSETS[s]*self.wkw - self.bkw/2})

    def key_center_x(self, midi):
        if midi%12 in WHITE_SEMITONES:
            for wk in self.white_keys:
                if wk['midi'] == midi: return wk['x'] + self.wkw/2
        else:
            for bk in self.black_keys:
                if bk['midi'] == midi: return bk['x'] + self.bkw/2
        return float(self.rect.centerx)

    def draw(self, surface, small_font):
        for wk in self.white_keys:
            x, w, mid = int(wk['x']), max(1,int(self.wkw)), wk['midi']
            col = self.highlights.get(mid, (235,235,235))
            pygame.draw.rect(surface, col,          (x, self.rect.y, w+1, int(self.wkh)))
            pygame.draw.rect(surface, (50,50,50),   (x, self.rect.y, w+1, int(self.wkh)), 1)
            if mid%12 == 0:
                lbl = small_font.render(f'C{mid//12-1}', True, (80,80,80))
                surface.blit(lbl, (x+w//2-lbl.get_width()//2,
                                   self.rect.y+self.wkh-lbl.get_height()-5))
        for bk in self.black_keys:
            x, w, mid = int(bk['x']), max(1,int(self.bkw)), bk['midi']
            pygame.draw.rect(surface, self.highlights.get(mid,(25,25,25)),
                             (x, self.rect.y, w, int(self.bkh)))


# ─── Main ─────────────────────────────────────────────────────────────────────

def make_trial_sequence():
    """Each frequency tested exactly twice, in random order."""
    pool = FREQUENCIES * 2          # 6 entries — every freq appears exactly twice
    random.shuffle(pool)
    return pool


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H),
                                     pygame.FULLSCREEN if FULLSCREEN else 0)
    pygame.display.set_caption('SSVEP Accuracy Test')
    clock = pygame.time.Clock()

    font_xl = pygame.font.SysFont('Arial', 80, bold=True)
    font_lg = pygame.font.SysFont('Arial', 32, bold=True)
    font_md = pygame.font.SysFont('Arial', 22)
    font_sm = pygame.font.SysFont('Arial', 14)

    # ── LSL outlet ────────────────────────────────────────────────────────────
    marker_outlet = StreamOutlet(StreamInfo(
        'SSVEPMarkers', 'Markers', 1, 0, 'float32', f'ssvep_{_uid()}'))
    print('[LSL] Marker outlet "SSVEPMarkers" ready.')

    # ── Layout ────────────────────────────────────────────────────────────────
    piano_h    = int(SCREEN_H * PIANO_H_FRAC)
    piano_rect = pygame.Rect(0, SCREEN_H - piano_h, SCREEN_W, piano_h)
    piano      = Piano(piano_rect)

    stim_cy = (STATUS_H + piano_rect.y) // 2
    n       = len(FREQUENCIES)
    total_w = n * STIM_PX + (n-1) * STIM_GAP
    x_start = (SCREEN_W - total_w) // 2 + STIM_PX // 2
    stim_cx = [x_start + i*(STIM_PX+STIM_GAP) for i in range(n)]
    key_cx  = [int(piano.key_center_x(m)) for m in NOTE_MIDI]
    stim_rects = [pygame.Rect(cx-STIM_PX//2, stim_cy-STIM_PX//2, STIM_PX, STIM_PX)
                  for cx in stim_cx]

    # Progress bar geometry (drawn above piano)
    BAR_H  = 12
    BAR_Y  = piano_rect.y - BAR_H - 8
    BAR_X  = stim_rects[0].x
    BAR_W  = stim_rects[-1].right - stim_rects[0].x

    # ── Test state ────────────────────────────────────────────────────────────
    state          = IDLE
    trial_seq      = make_trial_sequence()
    trial_idx      = 0
    state_t        = 0.0     # time.perf_counter() when state started
    running_stim   = False
    target_idx     = -1
    prev_phases    = [0] * n
    t_stim         = time.perf_counter()

    def start_state(new_state):
        nonlocal state, state_t
        state   = new_state
        state_t = time.perf_counter()

    def send_target(tidx):
        nonlocal target_idx
        target_idx = tidx
        freq = FREQUENCIES[tidx]
        marker_outlet.push_sample([float(MARKER_TARGET_BASE + freq)])
        print(f'[GUI] Trial {trial_idx+1}/{N_TRIALS} — target: {freq} Hz')

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

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        now_abs = time.perf_counter()
        elapsed = now_abs - state_t   # seconds in current state

        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    if running_stim: stop_stim()
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:
                    if state in (IDLE, DONE):
                        trial_seq  = make_trial_sequence()
                        trial_idx  = 0
                        target_idx = -1
                        send_target(FREQUENCIES.index(trial_seq[0]))
                        start_state(COUNTDOWN)

        # ── State machine ─────────────────────────────────────────────────────
        if state == COUNTDOWN:
            if elapsed >= COUNTDOWN_SECS:
                start_stim()
                start_state(RUNNING)

        elif state == RUNNING:
            if elapsed >= TRIAL_SEC:
                stop_stim()
                trial_idx += 1
                target_idx = -1
                if trial_idx >= N_TRIALS:
                    marker_outlet.push_sample([float(MARKER_SESSION_END)])
                    print(f'[GUI] Session complete — {N_TRIALS} trials done.')
                    start_state(DONE)
                else:
                    start_state(PAUSE)

        elif state == PAUSE:
            if elapsed >= PAUSE_SECS:
                send_target(FREQUENCIES.index(trial_seq[trial_idx]))
                start_state(COUNTDOWN)

        # ── Phase computation ─────────────────────────────────────────────────
        stim_now = time.perf_counter() - t_stim
        phases = []
        for i, freq in enumerate(FREQUENCIES):
            phase = int(stim_now * freq * 2) % 2
            phases.append(phase)
            if running_stim and phase == 0 and prev_phases[i] != 0:
                marker_outlet.push_sample([float(freq)])
        if running_stim:
            prev_phases = phases[:]

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        piano.highlights = {NOTE_MIDI[i]: FREQ_COLORS[i] for i in range(n)}
        piano.draw(screen, font_sm)

        # Connector lines
        for i, cx in enumerate(stim_cx):
            line_col    = tuple(max(0, c-80) for c in FREQ_COLORS[i])
            stim_bottom = stim_cy + STIM_PX//2 + BORDER_PX + 2
            pygame.draw.line(screen, line_col, (cx, stim_bottom), (key_cx[i], piano_rect.y), 2)

        # Stimulus boxes
        for i, rect in enumerate(stim_rects):
            is_target = (target_idx == i)

            if running_stim:
                _draw_checkerboard(screen, rect, phases[i])
            else:
                _draw_checkerboard_idle(screen, rect)

            bw        = BORDER_PX * 2 if is_target else BORDER_PX
            bord_rect = pygame.Rect(rect.x-bw, rect.y-bw, rect.w+bw*2, rect.h+bw*2)
            pygame.draw.rect(screen, FREQ_COLORS[i], bord_rect, bw)

            if is_target:
                # White outer glow
                gr = pygame.Rect(bord_rect.x-4, bord_rect.y-4,
                                 bord_rect.w+8, bord_rect.h+8)
                pygame.draw.rect(screen, (255, 255, 255), gr, 3)
                # Arrow above box
                arrow = font_lg.render('▼ LOOK HERE ▼', True, FREQ_COLORS[i])
                screen.blit(arrow, (rect.centerx - arrow.get_width()//2,
                                    rect.y - arrow.get_height() - 30))

            freq_txt = font_md.render(f'{FREQUENCIES[i]} Hz', True, FREQ_COLORS[i])
            screen.blit(freq_txt, (rect.centerx - freq_txt.get_width()//2,
                                   rect.y - freq_txt.get_height() - 4))
            note_txt = font_sm.render(NOTE_NAMES[i], True, TEXT_COLOR)
            screen.blit(note_txt, (rect.centerx - note_txt.get_width()//2,
                                   rect.bottom + 6))

        # Progress bar (shown during RUNNING)
        if state == RUNNING:
            fill = min(1.0, elapsed / TRIAL_SEC)
            pygame.draw.rect(screen, (50, 50, 60), (BAR_X, BAR_Y, BAR_W, BAR_H), border_radius=4)
            if fill > 0:
                col = FREQ_COLORS[target_idx] if target_idx >= 0 else TEXT_COLOR
                pygame.draw.rect(screen, col,
                                 (BAR_X, BAR_Y, int(BAR_W * fill), BAR_H), border_radius=4)
            secs_left = max(0.0, TRIAL_SEC - elapsed)
            timer_txt = font_sm.render(f'{secs_left:.1f}s remaining', True, DIM_COLOR)
            screen.blit(timer_txt, (BAR_X + BAR_W//2 - timer_txt.get_width()//2,
                                    BAR_Y - timer_txt.get_height() - 2))

        # Countdown overlay
        if state == COUNTDOWN:
            cd = max(0, COUNTDOWN_SECS - int(elapsed))
            cd_surf = font_xl.render(str(cd) if cd > 0 else 'GO', True, (255, 255, 100))
            screen.blit(cd_surf, (SCREEN_W//2 - cd_surf.get_width()//2,
                                  stim_cy - cd_surf.get_height()//2))

        # Done overlay
        if state == DONE:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            done_txt = font_lg.render('Session Complete — accuracy results in terminal',
                                      True, (100, 255, 100))
            restart  = font_md.render('Press SPACE to run again   |   ESC to quit',
                                      True, DIM_COLOR)
            screen.blit(done_txt, (SCREEN_W//2 - done_txt.get_width()//2, SCREEN_H//2 - 40))
            screen.blit(restart,  (SCREEN_W//2 - restart.get_width()//2,  SCREEN_H//2 + 10))

        # ── Status bar ────────────────────────────────────────────────────────
        pygame.draw.rect(screen, STATUS_BG, (0, 0, SCREEN_W, STATUS_H))

        if state == IDLE:
            msg     = 'Press SPACE to begin accuracy test  (6 trials × 10 s = 60 s total)'
            msg_col = DIM_COLOR
        elif state == COUNTDOWN:
            freq = trial_seq[trial_idx]
            msg     = f'Trial {trial_idx+1}/{N_TRIALS}  —  Focus on  {freq} Hz'
            msg_col = FREQ_COLORS[FREQUENCIES.index(freq)]
        elif state == RUNNING:
            freq = trial_seq[trial_idx]
            msg     = f'Trial {trial_idx+1}/{N_TRIALS}  —  RECORDING  —  target: {freq} Hz'
            msg_col = FREQ_COLORS[FREQUENCIES.index(freq)]
        elif state == PAUSE:
            next_freq = trial_seq[trial_idx] if trial_idx < N_TRIALS else None
            msg     = (f'Trial {trial_idx}/{N_TRIALS} done  —  '
                       f'Next: {next_freq} Hz' if next_freq else 'Last trial done')
            msg_col = TEXT_COLOR
        else:
            msg     = f'Done — {N_TRIALS} trials complete'
            msg_col = (100, 255, 100)

        status_surf = font_md.render(msg, True, msg_col)
        screen.blit(status_surf, (SCREEN_W//2 - status_surf.get_width()//2,
                                  STATUS_H//2 - status_surf.get_height()//2))

        esc_hint = font_sm.render('ESC = quit', True, DIM_COLOR)
        screen.blit(esc_hint, (SCREEN_W - esc_hint.get_width() - 14,
                                STATUS_H//2 - esc_hint.get_height()//2))

        pygame.display.flip()
        clock.tick(TARGET_FPS)


if __name__ == '__main__':
    main()

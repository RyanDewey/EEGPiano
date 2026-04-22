#!/usr/bin/env python
# Unicorn2lsl streams data from a Unicorn Hybrid Black EEG system to LSL
#
# Copyright (C) 2022 Robert Oostenveld
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import serial
import struct
import string
import random
import time
import signal
import glob
import subprocess
import threading
import numpy as np
from pylsl import StreamInfo, StreamOutlet

HEADSET_MAC = 'a4-6d-d4-17-bf-88'
timeout  = 5
nchan    = 16
fsample  = 250

start_acq = [0x61, 0x7C, 0x87]
stop_acq  = [0x63, 0x5C, 0xC5]

# ── Quit flag ─────────────────────────────────────────────────────────────────
quit_flag = threading.Event()

def handle_signal(signum, frame):
    print('\n[unicornlsl] stop signal received')
    quit_flag.set()

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT,  handle_signal)

# ── Bluetooth reset ───────────────────────────────────────────────────────────
def reset_bluetooth():
    """
    Disconnect and reconnect the BT session to get a fresh serial port.
    Do NOT unpair — unpairing destroys the SPP profile registration that
    creates /dev/cu.UN-*, and blueutil --pair alone can't recreate it.
    """
    print('[unicornlsl] resetting Bluetooth connection...')
    subprocess.run(['blueutil', '--disconnect', HEADSET_MAC], capture_output=True)
    time.sleep(3.0)
    subprocess.run(['blueutil', '--connect', HEADSET_MAC], capture_output=True)
    time.sleep(3.0)

    print('[unicornlsl] waiting for serial port to appear...')
    for _ in range(60):  # wait up to 30 seconds
        ports = glob.glob('/dev/cu.UN-*')
        if ports:
            print('[unicornlsl] serial port ready: ' + ports[0])
            # The /dev/cu.* file can appear before the BT serial channel is usable.
            time.sleep(3.0)
            return ports[0]
        time.sleep(0.5)

    raise RuntimeError('serial port never appeared — is the headset on and paired?')

# ── Serial helpers ────────────────────────────────────────────────────────────
def open_serial(device):
    try:
        s = serial.Serial(device, 115200, timeout=timeout)
        print('[unicornlsl] connected to serial port ' + device)
        time.sleep(2.0)  # let RFCOMM negotiation settle
        s.reset_input_buffer()
        s.reset_output_buffer()
        time.sleep(0.5)
        return s
    except Exception as e:
        raise RuntimeError(f'cannot connect to serial port {device}: {e}')

def do_stop(s):
    """Send stop_acq and drain. stop_acq sends NO response — never read after it."""
    print('[unicornlsl] sending stop to headset...')
    try:
        s.write(bytes(stop_acq))
        s.flush()
    except Exception:
        pass
    time.sleep(1.0)
    try:
        s.reset_input_buffer()
        s.reset_output_buffer()
    except Exception:
        pass
    print('[unicornlsl] headset stopped')

def start_headset(s):
    """Try to get the headset into a clean stopped state, then start streaming."""
    print('[unicornlsl] ensuring headset is stopped...')
    try:
        s.write(bytes(stop_acq))
        s.flush()
    except Exception:
        pass

    time.sleep(4.0)
    s.reset_input_buffer()
    s.reset_output_buffer()
    time.sleep(0.5)

    print('[unicornlsl] starting headset...')
    s.write(bytes(start_acq))
    s.flush()
    time.sleep(0.5)

    response = s.read(3)
    print(f'[unicornlsl] raw start response: {response!r}')
    if response != b'\x00\x00\x00':
        raise RuntimeError('cannot start data stream (response: %s)' % response.hex())

    print('[unicornlsl] started Unicorn')

# ── Connect and start headset ─────────────────────────────────────────────────
outlet = None
s = None

device = reset_bluetooth()
s = open_serial(device)

try:
    try:
        start_headset(s)
    except Exception:
        print('[unicornlsl] first start failed, reconnecting once...')
        try:
            s.close()
        except Exception:
            pass

        time.sleep(2.0)
        device = reset_bluetooth()
        s = open_serial(device)
        start_headset(s)

    # ── LSL outlet ────────────────────────────────────────────────────────────
    lsl_name   = 'Unicorn'
    lsl_type   = 'EEG'
    lsl_format = 'float32'
    lsl_id     = ''.join(random.choice(string.digits) for _ in range(6))

    streaminfo = StreamInfo(lsl_name, lsl_type, nchan, fsample, lsl_format, lsl_id)
    outlet = StreamOutlet(streaminfo)
    print('[unicornlsl] started LSL stream: name=%s, type=%s, id=%s' % (lsl_name, lsl_type, lsl_id))

    # ── Sync to packet boundary ───────────────────────────────────────────────
    print('[unicornlsl] syncing to packet boundary...')
    while not quit_flag.is_set():
        b = s.read(1)
        if b == b'\xC0':
            b2 = s.read(1)
            if b2 == b'\x00':
                s.read(43)  # discard rest of this first packet
                break
    print('[unicornlsl] synced — streaming')

    # ── Main loop ─────────────────────────────────────────────────────────────
    # Unicorn packet layout (45 bytes total):
    #   [0:2]   start sequence  0xC0 0x00
    #   [2]     battery/status
    #   [3:27]  8 EEG channels, 3 bytes each (big-endian 24-bit signed)
    #   [27:33] accelerometer,  3 × 2 bytes  (little-endian 16-bit signed)
    #   [33:39] gyroscope,      3 × 2 bytes  (little-endian 16-bit signed)
    #   [39:43] sample counter  (little-endian 32-bit unsigned)
    #   [43:45] end sequence    0x0D 0x0A

    while not quit_flag.is_set():
        payload = s.read(45)

        if len(payload) < 45:
            continue  # read timed out briefly, retry

        if payload[0:2] != b'\xC0\x00':
            raise RuntimeError('invalid packet')
        if payload[43:45] != b'\x0D\x0A':
            raise RuntimeError('invalid packet')

        battery = 100 * float(payload[2] & 0x0F) / 15

        eeg = np.zeros(8)
        for ch in range(0, 8):
            eegv = struct.unpack('>i', b'\x00' + payload[(3 + ch*3):(6 + ch*3)])[0]
            if eegv & 0x00800000:
                eegv = eegv | 0xFF000000
            eeg[ch] = float(eegv) * 4500000. / 50331642.

        accel = np.zeros(3)
        accel[0] = float(struct.unpack('<h', payload[27:29])[0]) / 4096.
        accel[1] = float(struct.unpack('<h', payload[29:31])[0]) / 4096.
        accel[2] = float(struct.unpack('<h', payload[31:33])[0]) / 4096.

        gyro = np.zeros(3)
        gyro[0] = float(struct.unpack('<h', payload[33:35])[0]) / 32.8
        gyro[1] = float(struct.unpack('<h', payload[35:37])[0]) / 32.8
        gyro[2] = float(struct.unpack('<h', payload[37:39])[0]) / 32.8

        counter = struct.unpack('<L', payload[39:43])[0]

        dat = np.zeros(nchan)
        dat[0:8]   = eeg
        dat[8:11]  = accel
        dat[11:14] = gyro
        dat[14]    = battery
        dat[15]    = counter

        outlet.push_sample(dat)

        if (counter % fsample) == 0:
            print('[unicornlsl] received %d samples, battery %d %%' % (counter, battery))

except Exception as e:
    print('[unicornlsl] error:', e)

finally:
    if s is not None:
        try:
            do_stop(s)
        except Exception as e:
            print(f'[unicornlsl] warning during stop: {e}')

        try:
            s.close()
        except Exception as e:
            print(f'[unicornlsl] warning during close: {e}')

    if outlet is not None:
        try:
            del outlet
        except Exception as e:
            print(f'[unicornlsl] warning during outlet cleanup: {e}')

    print('[unicornlsl] exited cleanly — headset ready for next run')

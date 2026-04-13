### EEG Piano

SSVEP-based EEG data collection using a Unicorn Hybrid Black headset. `unicornlsl.py` streams raw EEG to LSL, and `ssvep_stimulus.py` displays phase-reversing checkerboard stimuli (6, 7.5, 10, 12, 15, 20 Hz) above a piano keyboard while pushing timing markers to a second LSL stream. Both streams are recorded together in Lab Recorder into a single `.xdf` file.

---

#### Setup (one time only)

**1. Install blueutil**

`blueutil` is required so that `unicornlsl.py` can automatically manage the Bluetooth connection to the headset without you having to touch System Settings each session.

```
brew install blueutil
```

**2. Create and activate a Python virtual environment**

```
python -m venv .venv
source .venv/bin/activate
```

You must activate the virtual environment every time you open a new terminal before running any scripts. Your terminal prompt will show `(.venv)` when it is active.

**3. Install Python dependencies**

```
pip install -r requirements.txt
```

**4. Install Lab Recorder**

Lab Recorder is the application that captures LSL streams and saves them as `.xdf` files.

1. Go to the releases page in your browser:
   ```
   open https://github.com/labstreaminglayer/App-LabRecorder/releases
   ```
2. Find the latest release and download the macOS asset — it will be a `.dmg` file named something like `LabRecorder-1.16.0-OSX64.dmg`.
3. Open the downloaded `.dmg`:
   ```
   open ~/Downloads/LabRecorder-*.dmg
   ```
4. Drag **LabRecorder.app** into your `/Applications` folder when the installer window opens.
5. Eject the disk image, then launch Lab Recorder from Applications to verify it opens. macOS may show a security warning the first time — if so, go to **System Settings → Privacy & Security** and click **Open Anyway**.

**5. Pair the Unicorn headset (Bluetooth)**

- Power on the Unicorn Hybrid Black headset.
- Open **System Settings → Bluetooth** on your Mac.
- Find the headset in the device list and click **Connect** to pair it.
- Once paired, it will show as "not connected" — this is expected. The scripts manage the connection automatically using `blueutil`. Do **not** unpair it; unpairing destroys the serial port profile that the scripts depend on.

---

#### Recording a single `.xdf` file — step by step

Do this once per recording session (or once per file if you want multiple separate files).

**Step 1 — Open a terminal and activate the virtual environment**

```
cd /path/to/EEGPiano
source .venv/bin/activate
```

**Step 2 — Power on the Unicorn headset**

Press the power button on the headset and wait for it to finish booting (the LED will stop flashing rapidly). Make sure it is charged — the battery percentage is printed to the terminal every second once streaming starts.

**Step 3 — Launch the scripts**

Run the following command from the `EEGPiano` directory:

```
python start.py
```

This launches three subprocesses simultaneously:
- `unicornlsl.py` — connects to the headset over Bluetooth, starts streaming 16-channel EEG data at 250 Hz to an LSL outlet named `Unicorn`.
- `ssvep_stimulus.py` — opens the stimulus window and creates a second LSL outlet named `SSVEPMarkers` for event markers.
- `live_pipeline.py` — runs the live FBTRCA classifier (optional for data collection; you can ignore its output during recording).
Note: start.py has a TRAINING_MODE variable that can toggle live_pipeline.py

All output from all three scripts is printed to this single terminal, prefixed with `[LSL]`, `[MODEL]`, or `[unicornlsl]`.

Wait until you see both of the following lines before proceeding:
```
[unicornlsl] streaming
[LSL] Marker outlet "SSVEPMarkers" ready.
```

The Bluetooth reset and serial port negotiation takes approximately 10–15 seconds. If the headset fails to connect on the first attempt, `unicornlsl.py` will automatically retry once.

**Step 4 — Open Lab Recorder**

Open the Lab Recorder application. You do not need to do anything special — just have it open and ready.

**Step 5 — Discover and select the LSL streams in Lab Recorder**

1. Click the **Update** button in Lab Recorder. It will scan the network for active LSL streams.
2. You should see two streams appear in the list:
   - `Unicorn` — type `EEG`, 16 channels, 250 Hz
   - `SSVEPMarkers` — type `Markers`, 1 channel, irregular rate
3. Check the checkbox next to **both** streams. If you record only one, the EEG data and the stimulus markers will not be in the same file and cannot be aligned later.

If a stream does not appear, click **Update** again. If it still does not appear after 30 seconds, check the terminal for error output from `start.py`.

**Step 6 — Set the output filename**

In Lab Recorder, set the filename field to whatever you want the `.xdf` file to be named. Use a descriptive name that includes the subject ID, session number, or condition, for example:

```
sub-01_session-1_ssvep.xdf
```

Choose the save location by clicking the folder icon next to the filename. Lab Recorder will not create directories for you — make sure the folder already exists.

**Step 7 — Start recording in Lab Recorder**

Click **Start** in Lab Recorder. The button will change to **Stop** and the status indicator will show that both streams are being recorded. From this point on, all EEG samples and all marker events are being written to the `.xdf` file.

**Step 8 — Run the stimulus protocol**

Switch to the stimulus window (it opened automatically in Step 3). The window shows six flickering checkerboard boxes at 6, 7.5, 10, 12, 15, and 20 Hz above a piano keyboard.

Use the following controls during the session:

| Key | Action | Marker sent to LSL |
|-----|--------|--------------------|
| `SPACE` | Start stimulation (checkerboards begin flickering) | `252` (experiment start) |
| `1` | Cue target: 6 Hz (C3) | `106.0` |
| `2` | Cue target: 7.5 Hz (E3) | `107.5` |
| `3` | Cue target: 10 Hz (G3) | `110.0` |
| `4` | Cue target: 12 Hz (C4) | `112.0` |
| `5` | Cue target: 15 Hz (E4) | `115.0` |
| `6` | Cue target: 20 Hz (G4) | `120.0` |
| Same number again | Clear target cue | (no marker) |
| `SPACE` again | Stop stimulation | `253` (experiment stop) |
| `ESC` / `Q` | Quit the stimulus window | — |

Typical trial procedure:
1. Press a number key to select which frequency the subject should attend to. The corresponding checkerboard gets a thick colored border and a "TARGET" label.
2. Press `SPACE` to start the stimulation. The checkerboards begin flickering and per-cycle onset markers are streamed to LSL at the rate of the stimulus frequency.
3. Wait for the desired trial duration (e.g. 4–10 seconds).
4. Press `SPACE` again to stop stimulation.
5. Repeat from step 1 for the next trial.

**Step 9 — Stop recording in Lab Recorder**

When your session is complete, click **Stop** in Lab Recorder. The `.xdf` file is now fully written and closed. Do not close Lab Recorder or kill the terminal before clicking Stop, or the file may be truncated.

**Step 10 — Shut down the scripts**

In the terminal running `start.py`, press `Ctrl+C`. The shutdown sequence runs automatically:
1. `SIGTERM` is sent to `unicornlsl.py`, which sends a stop command to the headset over serial and waits for it to finish (up to 15 seconds).
2. The stimulus window and live pipeline are then terminated.
3. The terminal will print `[start] all done` when everything has exited cleanly.

The headset stays paired and connected over Bluetooth — you do not need to go back into System Settings. To record another file, start again from Step 1 (or Step 3 if you are staying in the same terminal session with the virtual environment already active).

---

#### Recording multiple files in one session

You do not need to reconnect the headset between recordings. `unicornlsl.py` automatically performs a Bluetooth disconnect/reconnect cycle every time it starts, so restarting `start.py` is enough. The sequence for a second file is:

1. Stop the current `start.py` with `Ctrl+C` and wait for `[start] all done`.
2. In Lab Recorder, click **Stop** if you have not already, set a new filename.
3. Run `python start.py` again in the same terminal.
4. Wait for `[unicornlsl] streaming` and `[LSL] Marker outlet "SSVEPMarkers" ready.`
5. Click **Update** then **Start** in Lab Recorder.
6. Continue with Step 8 above.

---

#### What is inside the `.xdf` file

An `.xdf` file is a container that holds multiple synchronized time series. This project's files contain:

- **`Unicorn` stream** — 16 channels at 250 Hz: channels 0–7 are EEG (µV), channels 8–10 are accelerometer (g), channels 11–13 are gyroscope (°/s), channel 14 is battery (%), channel 15 is the hardware sample counter.
- **`SSVEPMarkers` stream** — irregular-rate float markers:
  - `252` — experiment started (SPACE pressed)
  - `253` — experiment stopped (SPACE pressed again)
  - `100 + freq` — target frequency cued (e.g. `110.0` = 10 Hz target)
  - `freq` — per-cycle stimulus onset at the given frequency (e.g. `6.0` once every ~167 ms while stimulation runs at 6 Hz)

You can open `.xdf` files in Python with the `pyxdf` library:

```python
import pyxdf
streams, header = pyxdf.load_xdf("sub-01_session-1_ssvep.xdf")
```

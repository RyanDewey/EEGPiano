import subprocess
import threading
import time
import signal
import sys

scripts = {
    "GUI":   "ssvep_stimulus.py",
    "LSL":   "unicornlsl.py",
    "MODEL": "live_pipeline.py"
}

def stream_output(process, name):
    for line in process.stdout:
        print(f"[{name}] {line}", end="")

processes = {}

for name, script in scripts.items():
    p = subprocess.Popen(
        ["python", "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    processes[name] = p
    threading.Thread(target=stream_output, args=(p, name), daemon=True).start()

def shutdown(signum=None, frame=None):
    print("\n[start] shutting down...")

    # Step 1: Stop the LSL/headset process gracefully first.
    # Send SIGTERM, which unicornlsl.py catches and uses to run its clean
    # shutdown (stop_acq + wait). Give it up to 15s to finish.
    lsl = processes.get("LSL")
    if lsl and lsl.poll() is None:
        print("[start] sending SIGTERM to LSL — waiting for headset to stop...")
        lsl.send_signal(signal.SIGTERM)
        try:
            lsl.wait(timeout=15)
            print("[start] LSL stopped cleanly")
        except subprocess.TimeoutExpired:
            print("[start] LSL timeout — force killing")
            lsl.kill()

    # Step 2: Terminate the remaining processes.
    for name, p in processes.items():
        if name == "LSL":
            continue
        if p.poll() is None:
            print(f"[start] terminating {name}...")
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    print("[start] all done")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# Wait for any process to exit on its own, then shut everything down.
try:
    while True:
        for name, p in processes.items():
            if p.poll() is not None:
                print(f"[start] {name} exited (code {p.returncode}), shutting down")
                shutdown()
        time.sleep(0.5)
except KeyboardInterrupt:
    shutdown()
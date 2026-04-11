import subprocess
import threading

scripts = {
    "GUI": "ssvep_stimulus.py",
    "LSL": "unicornlsl.py",
}

def stream_output(process, name):
    for line in process.stdout:
        print(f"[{name}] {line}", end="")

processes = []

for name, script in scripts.items():
    p = subprocess.Popen(
        ["python", "-u", script],  # -u ensures live output
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    processes.append((name, p))

    threading.Thread(
        target=stream_output,
        args=(p, name),
        daemon=True
    ).start()

# wait for all processes
for name, p in processes:
    p.wait()
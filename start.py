import subprocess

scripts = ["ssvep_stimulus.py", "unicornlsl.py"]
processes = [subprocess.Popen(["python", s]) for s in scripts]

# Optional: Wait for all to finish
for p in processes:
    p.wait()

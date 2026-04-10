### EEG Piano

Instructions for using unicornlsl.py to enable unicorn hybrid black to lsl outlet via macOS.

1. Create a virtual environment
```
python -m venv .venv
```

2. Activate virtual environment
```
source .venv/bin/activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Make sure unicorn headset is powered ON, and connect to it with your macbook, it will connect for a second and then say "not connected" but appear in "My Devices"

5. Run script
```
python unicornlsl.py
```

6. Navigate to Lab Recorder software, click "Update" and select the Unicorn stream

7. Begin recording!

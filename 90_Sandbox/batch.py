import subprocess
import time
import sys
from pathlib import Path

script_to_run = Path("LP_DFT_simSmall.py").resolve()

for i in range(100):
    print(f"Run {i+1}/100")
    result = subprocess.run([sys.executable, str(script_to_run)])
    # kleine Pause für OS + Filesystem
    time.sleep(0.1)   # 100 ms, bei Bedarf auch 0.5–1.0 s
    
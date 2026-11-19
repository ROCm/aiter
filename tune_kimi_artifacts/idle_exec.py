"""Check idleness inside the optional GPU lock before launching a workload."""

import os
from pathlib import Path
import subprocess
import sys
import time


wrapper = Path('/usr/local/bin/gpu-lock')
locked = wrapper.is_file() and os.access(wrapper, os.X_OK)
while True:
    check = subprocess.run(['rocm-smi', '--showpids'],
                           capture_output=True, text=True)
    print(check.stdout, flush=True)
    if check.returncode:
        print(check.stderr, flush=True)
        raise SystemExit(74)
    if 'No KFD PIDs currently running' in check.stdout:
        break
    if locked:
        # Release the installed wrapper's lock before waiting on another job.
        raise SystemExit(75)
    time.sleep(1)
raise SystemExit(subprocess.run(sys.argv[1:]).returncode)

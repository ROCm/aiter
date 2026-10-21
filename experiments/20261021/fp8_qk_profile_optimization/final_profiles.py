import subprocess
import sys
from pathlib import Path

work = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-profile-opt-20261021/release")
for mode, case, kind, tag in [
    ("row", "4", "att", "final"),
    ("row", "4", "inst", "final"),
    ("row", "4", "occ", "final"),
    ("row", "4", "vmem", "final"),
    ("row", "4", "l2", "final"),
    ("row", "0", "att", "final"),
    ("paged", "4", "att", "final"),
]:
    label = f"final_{mode}_{case}_{kind}"
    with (work / ("run_" + label + ".log")).open("w") as log:
        p = subprocess.run(
            [sys.executable, str(work / "collect.py"), mode, case, kind, tag],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(label, p.returncode, flush=True)
    p.check_returncode()

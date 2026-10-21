import subprocess
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
work = base / "fp8-qk-pv-opt-20261021/release"
old = base / "fp8-qk-profile-opt-20261021"
for p in (old / "release").iterdir():
    if p.is_dir() and p.name.startswith(("final_row_", "final_paged_")):
        target = work / p.name.replace("final_", "baseline_", 1)
        if not target.exists():
            target.symlink_to(p, target_is_directory=True)
for p in old.iterdir():
    if p.is_dir() and p.name.startswith("baseline_pr_"):
        target = work / p.name
        if not target.exists():
            target.symlink_to(p, target_is_directory=True)
for kind in ["inst", "occ", "vmem"]:
    if (work / ("baseline_row_0_" + kind)).exists():
        continue
    with (work / ("run_baseline_row_0_" + kind + ".log")).open("w") as log:
        subprocess.run(
            [sys.executable, str(work / "collect.py"), "row", "0", kind, "baseline"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
for mode, case, kind in [
    ("row", "4", "att"),
    ("row", "4", "inst"),
    ("row", "4", "occ"),
    ("row", "4", "vmem"),
    ("row", "4", "l2"),
    ("row", "0", "att"),
    ("row", "0", "inst"),
    ("row", "0", "occ"),
    ("row", "0", "vmem"),
    ("paged", "4", "att"),
]:
    label = f"final_{mode}_{case}_{kind}"
    with (work / ("run_" + label + ".log")).open("w") as log:
        p = subprocess.run(
            [sys.executable, str(work / "collect.py"), mode, case, kind, "final"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(label, p.returncode, flush=True)
    p.check_returncode()

import subprocess
import sys
from pathlib import Path

work = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-pv-opt-20261021/release")
for label, args in [
    ("full_final", ["full_benchmark.py", "final"]),
    ("shared_final", ["shared.py"]),
    ("audit_final", ["audit.py", "final"]),
    ("profiles_final", ["final_profiles.py"]),
    ("summarize_final", ["summarize.py"]),
    ("att_compare_final", ["att_compare.py"]),
    ("analysis_final", ["analyze_final.py"]),
]:
    path = work / (label + ".log")
    assert not path.exists(), label
    with path.open("w") as log:
        p = subprocess.run(
            [sys.executable, "-u", str(work / args[0]), *args[1:]],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print("DONE", label, p.returncode, flush=True)
    if p.returncode:
        print(path.read_text()[-4500:], flush=True)
        p.check_returncode()

import subprocess
import sys
from pathlib import Path

w = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-opt-20261021")
tag = sys.argv[1]
metrics = [
    ["SQ_INSTS_VALU", "SQ_INSTS_MFMA", "SQ_INSTS_LDS", "SQ_WAVES"],
    ["LDSBankConflict", "MeanOccupancyPerCU", "OccupancyPercent"],
]
for i, pmc in enumerate(metrics):
    out = w / ("profile_" + tag + "_" + str(i))
    cmd = [
        "rocprofv3",
        "--pmc",
        *pmc,
        "--kernel-include-regex",
        "fused_dsv41_fp8_qk_decode_64",
        "--kernel-iteration-range",
        "3-5",
        "-d",
        str(out),
        "-o",
        "counters",
        "--",
        sys.executable,
        str(w / "profile_workload.py"),
        tag,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (w / ("profile_" + tag + "_" + str(i) + ".log")).write_text(r.stdout + r.stderr)
    print("RUN", tag, i, r.returncode, r.stdout[-800:], r.stderr[-800:])
    if r.returncode:
        continue
    for p in out.rglob("*.csv"):
        print(p, p.read_text()[:3500])

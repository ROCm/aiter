import collections
import csv
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-pr6675-20261021"
work = base / "fp8-qk-profile-opt-20261021"
path = repo / "csrc/kernels/mla/opus/mla_v41_prefill_a8w8_16mx4_64nx1.hpp"
lines = path.read_text().splitlines()
selected = set()
for i, s in enumerate(lines):
    if any(
        w in s
        for w in [
            "tr_load",
            "W_K_PV",
            "compute_pv",
            "ds_read",
            "VEC_TR_V",
            "struct Traits",
            "struct traits",
            "mma_pv",
            "pv_mfma",
        ]
    ):
        selected.update(range(max(0, i - 5), min(len(lines), i + 12)))
print("SOURCE", path)
for i in sorted(selected):
    print(i + 1, lines[i])
print("HEAD", "\n".join(lines[:20]))
for p in (work / "baseline_pr_4_att_v2").glob("stats*.csv"):
    rows = list(csv.DictReader(p.open()))
    ops = collections.Counter()
    for r in rows:
        op = r["Instruction"].split()[0]
        if op.startswith(("v_mfma", "ds_")):
            ops[op] += int(r["Hitcount"])
    print("OPS", p.name, dict(ops))

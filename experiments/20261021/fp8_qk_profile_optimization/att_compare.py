import collections
import csv
import json
from pathlib import Path

work = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-profile-opt-20261021/release")
results = {}
for tag in ["baseline", "deferred_index", "wide_deferred_index", "final"]:
    root = work / f"{tag}_row_4_att"
    if not root.exists():
        continue
    path = next(root.glob("stats*.csv"))
    rows = list(csv.DictReader(path.open()))
    waves = [json.loads(p.read_text()) for p in root.glob("ui*/se*_wv*.json")]
    cats = collections.Counter()
    for r in rows:
        op = r["Instruction"].split()[0]
        if op == "s_waitcnt":
            op = "vmem_wait" if "vmcnt" in r["Instruction"] else "lds_wait"
        elif op.startswith("ds_"):
            op = "lds_issue"
        elif op.startswith("v_mfma"):
            op = "mfma_issue"
        elif op.startswith("v_"):
            op = "valu_issue"
        cats[op] += int(r["Latency"])
    total = sum(cats.values())
    result = dict(
        waves=len(waves),
        total_wave_cycles=sum(w["duration"] for w in waves),
        instruction_latency=total,
        latency_per_wave={k: v / len(waves) for k, v in cats.items()},
        latency_fraction={k: v / total for k, v in cats.items()},
        top_stalls=sorted(rows, key=lambda r: int(r["Stall"]), reverse=True)[:5],
    )
    results[tag] = result
    print(tag, json.dumps(result))
(work / "att_comparison.json").write_text(json.dumps(results, indent=2))

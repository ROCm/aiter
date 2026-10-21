import collections
import csv
import json
from pathlib import Path

work = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-pv-opt-20261021/release")
summary = {}
for path in work.glob("*att*/stats*.csv"):
    rows = list(csv.DictReader(path.open()))
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        op = r["Instruction"].split()[0]
        if op == "s_waitcnt":
            op += "_vmem" if "vmcnt" in r["Instruction"] else "_lds"
        elif op.startswith("v_mfma"):
            op = "mfma"
        elif op.startswith("ds_"):
            op = "lds"
        elif op.startswith("global_"):
            op = "vmem"
        elif op.startswith("v_"):
            op = "valu"
        for f in ["Hitcount", "Latency", "Stall", "Idle"]:
            agg[op][f] += int(r[f])
    data = {
        "aggregate": dict(agg),
        "top_stalls": sorted(rows, key=lambda r: int(r["Stall"]), reverse=True)[:8],
    }
    summary[str(path.relative_to(work))] = data
    print(
        "ATT",
        path.parent.name,
        path.name,
        "TOTAL_LATENCY",
        sum(int(r["Latency"]) for r in rows),
    )
    print(
        json.dumps(
            dict(sorted(agg.items(), key=lambda x: x[1]["Stall"], reverse=True)[:10])
        )
    )
for path in work.glob("*/*counter*.csv"):
    rows = list(csv.DictReader(path.open()))
    kernels = {}
    for r in rows:
        entry = kernels.setdefault(r["Kernel_Name"], {})
        entry[r["Counter_Name"]] = float(r["Counter_Value"])
        entry["duration_us"] = (
            int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        ) / 1000
        entry["lds_bytes"] = int(r["LDS_Block_Size"])
    summary[str(path.relative_to(work))] = kernels
    print("PMC", path.parent.name, json.dumps(kernels))
(work / "att_summary.json").write_text(json.dumps(summary, indent=2))

import collections
import csv
import json
from pathlib import Path

root = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-pv-opt-20261021")
work = root / "release"
results = {}
for tag, folder in [
    ("baseline", work / "baseline_row_0_att"),
    ("first_combination", root / "final_row_0_att"),
    ("final", work / "final_row_0_att"),
]:
    if not folder.exists():
        continue
    rows = list(csv.DictReader(next(folder.glob("stats*.csv")).open()))
    waves = [json.loads(p.read_text()) for p in folder.glob("ui*/se*_wv*.json")]
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
    top = sorted(rows, key=lambda r: int(r["Stall"]), reverse=True)[:5]
    first = next(i for i, r in enumerate(rows) if "v_mfma_scale" in r["Instruction"])
    results[tag] = {
        "waves": len(waves),
        "latency_per_wave": {k: v / len(waves) for k, v in cats.items()},
        "top_stalls": top,
        "qk_start": rows[max(0, first - 35) : first + 5],
    }
(work / "h64_att_comparison.json").write_text(json.dumps(results, indent=2))
for tag, data in results.items():
    print(
        "H64",
        tag,
        data["waves"],
        {
            k: data["latency_per_wave"][k]
            for k in ["vmem_wait", "lds_wait", "lds_issue", "valu_issue", "s_barrier"]
        },
    )
    print("TOP", json.dumps(data["top_stalls"]))
    if tag == "final":
        print("QK_START", json.dumps(data["qk_start"]))
contexts = json.loads((work / "final_stall_contexts.json").read_text())
for c in contexts[:4]:
    print("H128_STALL", c["stall"]["Vaddr"])
    print([(r["Vaddr"], r["Instruction"]) for r in c["instructions"]])

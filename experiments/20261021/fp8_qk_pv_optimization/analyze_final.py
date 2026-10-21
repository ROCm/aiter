import csv
import json
from pathlib import Path

work = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-pv-opt-20261021/release")
bench = json.loads((work / "benchmark_final.json").read_text())
assert bench["complete"]
summary = json.loads((work / "att_summary.json").read_text())
att = json.loads((work / "att_comparison.json").read_text())


def counters(prefix, match=None):
    result = {}
    for name, kernels in summary.items():
        if name.startswith(prefix) and "counter" in name:
            for kernel, data in kernels.items():
                if match is None or match in kernel:
                    result.update(
                        {k: v for k, v in data.items() if k not in ["duration_us"]}
                    )
    return result


rows = []
for r in bench["rows"]:
    m = r["median_us"]
    rows.append(
        {
            "name": r["name"],
            "case": r["case"],
            "median_us": m,
            "row_latency_reduction_pct": 100 * (1 - m["fp8_row"] / m["fp8old_row"]),
            "paged_latency_reduction_pct": 100
            * (1 - m["fp8_paged"] / m["fp8old_paged"]),
            "row_vs_pr_latency_pct": 100 * (m["fp8_row"] / m["pr_v41_full"] - 1),
            "dense_errors": r.get("dense_errors"),
            "pr_errors": r["pr_errors"],
            "previous_output_l2": r["fp8_vs_previous_output_l2"],
        }
    )
result = {
    "selection": json.loads((work / "selection.json").read_text()),
    "rows": rows,
    "counters": {
        label: counters(prefix, match)
        for label, prefix, match in [
            ("baseline", "baseline_row_4_", None),
            ("final", "final_row_4_", None),
            ("pr_attention", "baseline_pr_4_", "opus_mla_v41_prefill"),
            ("pr_q_prepare", "baseline_pr_4_", "q_prepare"),
            ("pr_epilogue", "baseline_pr_4_", "inverse_rope"),
            ("baseline_h64", "baseline_row_0_", None),
            ("final_h64", "final_row_0_", None),
        ]
    },
    "att": {
        tag: {
            k: info[k]
            for k in [
                "waves",
                "total_wave_cycles",
                "instruction_latency",
                "latency_per_wave",
                "latency_fraction",
                "top_stalls",
            ]
        }
        for tag, info in att.items()
        if tag in ["baseline", "final"]
    },
}
(work / "analysis_final.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
# Preserve exact final ISA context for the stalls discussed in the report.
path = next((work / "final_row_4_att").glob("stats*.csv"))
instructions = list(csv.DictReader(path.open()))
contexts = []
for stall in result["att"]["final"]["top_stalls"]:
    pc = int(stall["Vaddr"])
    contexts.append(
        {
            "stall": stall,
            "instructions": [
                r for r in instructions if pc - 100 <= int(r["Vaddr"]) <= pc + 64
            ],
        }
    )
(work / "final_stall_contexts.json").write_text(json.dumps(contexts, indent=2))

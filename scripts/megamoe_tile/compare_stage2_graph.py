"""Compare verified EPLB Graph summaries; refuse uncontrolled input changes.

Usage: python -m scripts.megamoe_tile.compare_stage2_graph BASELINE CANDIDATE
BASELINE may be MORI or an earlier candidate. The output ratio is baseline
Stage2 span / candidate Stage2 span. Stage1 is excluded from that ratio.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.megamoe_tile.stage2_graph_inputs import aggregate_input_identity, performance_exclusions


COMMON_CONTROLS = (
    "shape", "route_pattern", "seed", "max_routes_per_token_per_rank",
    "warmup", "iterations", "tail_iterations", "graph_comparison_statistic",
    "measurement", "graph_contains_output_clone", "graph_input_quantization",
    "activation_parameters", "mori_combine_quant_type", "candidate_rail_quant_type",
    "mori_blocks", "mori_rdma_blocks", "tune_sha256", "torch_version", "environment",
    "graph_check_routing_replay", "correctness_replays",
)
COMMON_SOURCES = (
    "op_tests/multigpu_tests/bench_megamoe_tile_ep16_dual_path.py",
    "op_tests/multigpu_tests/bench_megamoe_tile_ep16_two_kernel.py",
    "op_tests/multigpu_tests/bench_mega_moe_v2.py",
    "op_tests/multigpu_tests/megamoe_stage2_graph.py",
    "scripts/megamoe_tile/stage2_graph_inputs.py",
    "scripts/megamoe_tile/summarize_stage2_graph.py",
)


def compare_reports(baseline, candidate):
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("performance_eligible") is not True:
            raise ValueError(f"{name}: diagnostic/unverified run is not performance eligible")
        identity = report.get("input_identity")
        if not identity or identity.get("schema") != "stage2_actual_inputs_v1":
            raise ValueError(f"{name}: missing actual input identity; equal seed alone is insufficient")
        if identity != aggregate_input_identity(identity["ranks"], shape=report["shape"]):
            raise ValueError(f"{name}: inconsistent input identity/workload summary")
        exclusions = performance_exclusions(report, identity)
        if exclusions:
            raise ValueError(f"{name}: " + "; ".join(exclusions))
        if not identity["eplb"] or report["route_pattern"] != "cross_node":
            raise ValueError(f"{name}: only cross_node EPLB is a performance workload")
        if (report["warmup"], report["iterations"], report["tail_iterations"]) != (10, 40, 20):
            raise ValueError(f"{name}: formal Graph protocol must be 10/40/20")
        if report.get("input_integrity_passed") is not True:
            raise ValueError(f"{name}: missing input restoration/integrity checks")
        if not report.get("correctness") or any(
            not math.isfinite(check["rank_max_rel_l2"]) or check["rank_max_rel_l2"] >= check["threshold"]
            for check in report["correctness"]
        ):
            raise ValueError(f"{name}: numerical correctness is missing or failed")
        timing = report["timing"]
        if timing["world_size"] != 16 or timing["samples"] != 320 or timing["iterations"] != list(range(20, 40)):
            raise ValueError(f"{name}: incomplete EP16 tail20 evidence")
    if baseline["path"] not in ("mori", "candidate") or candidate["path"] != "candidate":
        raise ValueError("expected MORI/candidate control and candidate result")
    mismatches = [key for key in COMMON_CONTROLS
                  if key not in baseline or key not in candidate or baseline[key] != candidate[key]]
    for source in COMMON_SOURCES:
        left, right = baseline.get("source_sha256", {}).get(source), candidate.get("source_sha256", {}).get(source)
        if not left or not right or left != right:
            mismatches.append(source)
    # Stage2-only candidate tuning holds Stage1 configuration and source fixed.
    if baseline["path"] == "candidate":
        if baseline.get("stage1_workers") != candidate.get("stage1_workers"):
            mismatches.append("stage1_workers")
        for source in ("stage1.py", "stage1_abi.py"):
            key = "aiter/ops/flydsl/kernels/megamoe_tile/" + source
            if not baseline["source_sha256"].get(key) or baseline["source_sha256"].get(key) != candidate["source_sha256"].get(key):
                mismatches.append(key)
    if baseline["input_identity"] != candidate["input_identity"]:
        mismatches.append("actual input/weight bytes, device identity or expert workload")
    if mismatches:
        raise ValueError("uncontrolled A/B variables: " + ", ".join(mismatches))
    statistic = baseline["graph_comparison_statistic"]
    left = baseline["timing"]["metrics"]["stage2_span_us"][statistic]
    right = candidate["timing"]["metrics"]["stage2_span_us"][statistic]
    if not all(math.isfinite(value) and value > 0 for value in (left, right)):
        raise ValueError("Stage2 timings must be finite and positive")
    return {"input_sha256": baseline["input_identity"]["sha256"],
            "shape": baseline["shape"], "statistic": statistic,
            "baseline_path": baseline["path"], "baseline_stage2_us": left,
            "candidate_stage2_us": right, "speedup_baseline_over_candidate": left / right,
            "candidate_latency_change_percent": (right / left - 1) * 100,
            "baseline_min_sample": baseline["timing"].get("stage2_min_sample"),
            "candidate_min_sample": candidate["timing"].get("stage2_min_sample"),
            "stage2_configuration_changes": {
                key: {"baseline": baseline.get(key), "candidate": candidate.get(key)}
                for key in sorted(set(baseline) | set(candidate))
                if (key.startswith("candidate_") or key == "stage2_workers")
                and baseline.get(key) != candidate.get(key)},
            "scope": "same-replay GEMM2 start to combine end versus fused Stage2; Stage1 excluded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        result = compare_reports(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

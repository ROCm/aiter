# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Summarize recorded calibration outcomes without inventing missing results."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def search_result(record):
    result = dict(record)
    result["outcome"] = record.get("outcome", "incomplete_run")
    latency = record.get("winner_us")
    result["valid_winner"] = (
        result["outcome"] == "valid" and finite_number(latency) and latency > 0
    )
    if "numerical_error" in record:
        error = record["numerical_error"]
        result["valid_winner"] = result["valid_winner"] and (
            finite_number(error) and error <= record.get("error_gate", 0.1)
        )
    if not result["valid_winner"]:
        result["winner_us"] = None
        cause = (record.get("failure_cause") or result["outcome"]).replace("-", "_")
        result["failure_cause"] = (
            cause
            if cause
            in {
                "missing_reference",
                "screening",
                "normal_pair",
                "worker_death",
                "incomplete_run",
            }
            else "normal_pair" if result["outcome"] == "valid" else "unclassified"
        )
    return result


def comparison(reference, candidate):
    if reference is None or candidate is None:
        return {"available": False, "regret_percent": None, "reason": "missing_search"}
    if not reference["valid_winner"] or not candidate["valid_winner"]:
        return {"available": False, "regret_percent": None, "reason": "invalid_winner"}
    return {
        "available": True,
        "regret_percent": 100 * (candidate["winner_us"] / reference["winner_us"] - 1),
        "reason": None,
    }


def wall_totals(searches, model, config=None):
    totals = {}
    for mode in ("prune", "staged"):
        matching = [
            record
            for record in searches
            if record["model"] == model
            and record["mode"] == mode
            and (config is None or record["config"] == config)
        ]
        times = [
            record["search_wall_seconds"]
            for record in matching
            if finite_number(record.get("search_wall_seconds"))
            and record["search_wall_seconds"] >= 0
        ]
        totals[mode] = {
            "expected_searches": len(matching),
            "recorded_searches": len(times),
            "observed_seconds": sum(times) if times else None,
            "complete": bool(matching) and len(times) == len(matching),
        }
    return totals


def summarize(records):
    """Return per-search outcomes and paired comparisons for annotated manifest jobs."""
    searches = [search_result(record) for record in records]
    groups = {}
    for result in searches:
        key = (result["model"], result["token"], result["config"])
        groups.setdefault(key, {})[result["mode"]] = result
    rows = []
    causes = Counter()
    for (model, token, config), modes in groups.items():
        prune = modes.get("prune")
        staged = modes.get("staged")
        full = modes.get("full")
        rows.append(
            {
                "model": model,
                "token": token,
                "config": config,
                "prune": prune,
                "staged": staged,
                "full": full,
                "full_to_prune": comparison(full, prune),
                "prune_to_staged": comparison(prune, staged),
                "full_to_staged": comparison(full, staged),
            }
        )
        if prune and prune["valid_winner"] and staged and not staged["valid_winner"]:
            causes[staged["failure_cause"]] += 1
    return {
        "searches": searches,
        "paired_rows": rows,
        "prune_valid_staged_invalid": {
            "count": sum(causes.values()),
            "causes": dict(causes),
        },
        "wall_time_totals": {
            "default_minimax": wall_totals(searches, "minimax", "default"),
            "extended_gpt": wall_totals(searches, "gpt"),
        },
    }


def load_records(manifest, gpu_dir):
    records = json.loads(manifest.read_text())
    if gpu_dir is None:
        return records
    annotated = []
    for job in records:
        directory = gpu_dir / job["id"]
        result_path = directory / "search.json"
        task_path = directory / "search-task" / "task.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        task = json.loads(task_path.read_text()) if task_path.exists() else {}
        record = dict(job, **result)
        record["task"] = task
        record["result_path"] = str(result_path) if result_path.exists() else None
        if not result_path.exists() or (
            record.get("outcome") == "valid" and task.get("exit_code") != 0
        ):
            record["outcome"] = "incomplete_run"
        annotated.append(record)
    return annotated


def display(value):
    if value is None:
        return "unavailable"
    return str(value).replace("|", "\\|").replace("\n", " ")


def error_range(errors):
    if not errors:
        return None
    numbers = [float(error) for error in errors]
    return f"{min(numbers):.6g}–{max(numbers):.6g} ({len(numbers)} logged)"


def markdown_report(report):
    lines = [
        "# MXFP4 calibration results",
        "",
        (
            f"Recorded manifest searches: {len(report['searches'])}. Valid winners: "
            f"{sum(record['valid_winner'] for record in report['searches'])}."
        ),
        "",
        (
            "Regret is 100 × (candidate winner kernel-duration sum / reference winner "
            "kernel-duration sum − 1). Comparisons require matching valid finite winners. "
            "Full→prune measures M_est pruning; prune→staged measures stage screening. "
            "Missing full searches provide no full reference."
        ),
        "",
        "## Per-search evidence",
        "",
        (
            "Logged numerical errors below use the reference metric; CSV err1 is the "
            "separate rounded CSV field. Winner latency is the main-compatible kernel-duration "
            "sum. Tuning wall time and matched elapsed diagnostics are separate measurements."
        ),
        "",
        "| Search | Outcome | Valid winner | GEMM1 | GEMM2 | Winner sum (us) | Logged numerical errors | CSV err1 | Validations | Search wall (s) | Matched elapsed median (us) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["searches"]:
        winner = record.get("winner") or {}
        values = (
            record["id"],
            record["outcome"],
            record["valid_winner"],
            winner.get("kernelName1"),
            winner.get("kernelName2"),
            record.get("winner_us"),
            error_range(record.get("finite_reference_errors")),
            winner.get("err1"),
            record.get("validation_attempts"),
            record.get("search_wall_seconds"),
            record.get("matched_synchronized_elapsed_median_us"),
        )
        lines.append("| " + " | ".join(display(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Paired outcomes and regret",
            "",
            "| Model | Tokens | Configuration | Prune valid | Staged valid | Staged failure cause | Full→prune (%) | Prune→staged (%) | Full→staged (%) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["paired_rows"]:
        values = (
            row["model"],
            row["token"],
            row["config"],
            row["prune"]["valid_winner"] if row["prune"] else False,
            row["staged"]["valid_winner"] if row["staged"] else False,
            row["staged"].get("failure_cause") if row["staged"] else "incomplete_run",
            *(
                row[key]["regret_percent"]
                for key in ("full_to_prune", "prune_to_staged", "full_to_staged")
            ),
        )
        lines.append("| " + " | ".join(display(value) for value in values) + " |")
    invalid = report["prune_valid_staged_invalid"]
    lines.extend(
        [
            "",
            f"Prune-valid/staged-invalid outcomes: {invalid['count']}.",
            "",
            "| Failure cause | Count |",
            "| --- | --- |",
        ]
    )
    lines.extend(f"| {cause} | {count} |" for cause, count in invalid["causes"].items())
    lines.extend(
        [
            "",
            "## Tuning wall-time totals",
            "",
            (
                "Observed sums include failed searches with recorded finite wall times. "
                "A partial sum does not estimate missing work; default MiniMax and extended "
                "GPT-OSS results remain separate."
            ),
            "",
            "| Matrix | Mode | Recorded / expected | Observed sum (s) | Complete |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for matrix, modes in report["wall_time_totals"].items():
        for mode, total in modes.items():
            lines.append(
                f"| {matrix} | {mode} | {total['recorded_searches']} / "
                f"{total['expected_searches']} | {display(total['observed_seconds'])} | {total['complete']} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--gpu-dir",
        type=Path,
        help="Load each id/search.json and id/search-task/task.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(load_records(args.manifest, args.gpu_dir))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.markdown.write_text(markdown_report(report))


if __name__ == "__main__":
    main()

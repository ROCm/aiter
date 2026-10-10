# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Summarize recorded calibration outcomes without inventing missing results."""

import argparse
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path


def finite_number(value: "object") -> "bool":
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def search_result(record: "dict[str, object]") -> "dict[str, object]":
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


def comparison(
    reference: "dict[str, object] | None", candidate: "dict[str, object] | None"
) -> "dict[str, object]":
    if reference is None or candidate is None:
        return {"available": False, "regret_percent": None, "reason": "missing_search"}
    if not reference["valid_winner"] or not candidate["valid_winner"]:
        return {"available": False, "regret_percent": None, "reason": "invalid_winner"}
    return {
        "available": True,
        "regret_percent": 100 * (candidate["winner_us"] / reference["winner_us"] - 1),
        "reason": None,
    }


def wall_totals(
    searches: "list[dict[str, object]]", model: "str", config: "str | None" = None
) -> "dict[str, dict[str, object]]":
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


def rank_retention(
    prune_winner: "dict[str, object]", console: "str", screen_topk: "int" = 2
) -> "dict[str, object]":
    """Rank a prune winner within the staged log's screening contexts."""
    g1 = prune_winner["kernelName1"]
    g2 = prune_winner["kernelName2"]
    block_m = int(re.search(r"_a4w4_(\d+)x", g1).group(1))
    accumulation = "atomic" if "_atomic" in g2 else "non_atomic"
    scores = {"GEMM1": [], "GEMM2": []}
    for stage, name, score in re.findall(
        r"\[mxfp4-staged\] (GEMM[12]) (\S+) screening us=(\S+)", console
    ):
        pattern = r"_a4w4_(\d+)x" if stage == "GEMM1" else r"_bf16_t(\d+)x"
        match = re.search(pattern, name)
        if match is None or int(match.group(1)) != block_m:
            continue
        if (
            stage == "GEMM2"
            and ("atomic" if "_atomic" in name else "non_atomic") != accumulation
        ):
            continue
        latency = float(score)
        if math.isfinite(latency) and latency > 0:
            scores[stage].append((name, latency))
    ranks = []
    for stage, name in (("GEMM1", g1), ("GEMM2", g2)):
        ranked = sorted(scores[stage], key=lambda item: item[1])
        ranks.append(
            next((i for i, item in enumerate(ranked, 1) if item[0] == name), None)
        )
    validated = (g1, g2) in re.findall(
        r"\[mxfp4-staged\] (\S+) \+ (\S+) (?:us=|validation failed:)", console
    )
    return {
        "block_m": block_m,
        "accumulation_mode": accumulation,
        "screen_topk": screen_topk,
        "g1_rank": ranks[0],
        "g2_rank": ranks[1],
        "g1_successful_candidates": len(scores["GEMM1"]),
        "g2_successful_candidates": len(scores["GEMM2"]),
        "both_retained": all(
            rank is not None and rank <= screen_topk for rank in ranks
        ),
        "pair_validated": validated,
    }


def staged_fallback_groups(console: "str", screen_topk: "int" = 2) -> "int":
    """Count groups with logged fallback or returned pairs outside the shortlist."""
    groups = {
        (int(block_m), accumulation.replace("-", "_"))
        for block_m, accumulation in re.findall(
            r"\[mxfp4-staged\] exhaustive fallback BM(\d+) (atomic|non-atomic):",
            console,
        )
    }
    pairs = re.findall(
        r"\[mxfp4-staged\] (\S+) \+ (\S+) (?:us=|validation failed:)", console
    )
    for g1, g2 in pairs:
        retained = rank_retention(
            {"kernelName1": g1, "kernelName2": g2}, console, screen_topk
        )
        if not retained["both_retained"]:
            groups.add((retained["block_m"], retained["accumulation_mode"]))
    return len(groups)


def summarize(records: "list[dict[str, object]]") -> "dict[str, object]":
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
        if (
            prune
            and prune["valid_winner"]
            and staged
            and staged["valid_winner"]
            and "rank_retention" in staged
        ):
            rows[-1]["rank_retention"] = staged["rank_retention"]
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


def load_records(manifest: "Path", gpu_dir: "Path | None") -> "list[dict[str, object]]":
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
    groups = {}
    for record in annotated:
        key = (record["model"], record["token"], record["config"])
        groups.setdefault(key, {})[record["mode"]] = record
    for modes in groups.values():
        prune, staged = modes.get("prune"), modes.get("staged")
        if not (
            prune
            and staged
            and search_result(prune)["valid_winner"]
            and search_result(staged)["valid_winner"]
        ):
            continue
        console = gpu_dir / staged["id"] / "search-task" / "console.log.txt"
        compressed = Path(str(console) + ".gz")
        if compressed.exists():
            with gzip.open(compressed, "rt") as stream:
                text = stream.read()
        elif console.exists():
            text = console.read_text()
        else:
            continue
        staged["rank_retention"] = rank_retention(prune["winner"], text)
    return annotated


def display(value: "object") -> "str":
    if value is None:
        return "unavailable"
    return str(value).replace("|", "\\|").replace("\n", " ")


def error_range(errors: "object") -> "str | None":
    if not errors:
        return None
    numbers = [float(error) for error in errors]
    return f"{min(numbers):.6g}–{max(numbers):.6g} ({len(numbers)} logged)"


def markdown_report(report: "dict[str, object]") -> "str":
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
    retained = [row for row in report["paired_rows"] if "rank_retention" in row]
    if retained:
        lines.extend(
            [
                "",
                "## Prune winner screening ranks",
                "",
                (
                    "Ranks use successful staged screens, ascending score with original log "
                    "order for ties. GEMM1 ranks are reused per BM; GEMM2 ranks separate atomic "
                    "and non-atomic accumulation, combining reduce/scatter in the latter. "
                    "Pair validated means a returned validation outcome was logged, including failure."
                ),
                "",
                "| Model | Tokens | Configuration | BM | Accumulation | GEMM1 rank | GEMM2 rank | Both retained K2 | Pair validated |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in retained:
            rank = row["rank_retention"]
            values = (
                row["model"],
                row["token"],
                row["config"],
                rank["block_m"],
                rank["accumulation_mode"],
                rank["g1_rank"],
                rank["g2_rank"],
                rank["both_retained"],
                rank["pair_validated"],
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


def main() -> "None":
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

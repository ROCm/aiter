"""Read current named GPU evidence without claiming missing acceptance passed."""

import argparse
import json
import math
from pathlib import Path


def main() -> "None":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads((args.root / "fixtures.json").read_text())
    summary = []
    for fixture in fixtures:
        directory = args.root / "gpu" / fixture["id"]
        record = {
            "id": fixture["id"],
            "configuration": {key: fixture[key] for key in ("fp8", "fp4", "limit")},
            "actions": {},
        }
        for action in ("pair", "replay", "aot", "precompiled"):
            result_path = directory / f"{action}.json"
            task_path = directory / f"{action}-task" / "task.json"
            if not result_path.exists() or not task_path.exists():
                task = json.loads(task_path.read_text()) if task_path.exists() else {}
                record["actions"][action] = {
                    "status": (
                        "failed"
                        if task.get("exit_code") not in (None, 0)
                        else "incomplete"
                    ),
                    "exit_code": task.get("exit_code"),
                }
                continue
            result = json.loads(result_path.read_text())
            task = json.loads(task_path.read_text())
            status = (
                "incomplete"
                if task.get("exit_code") is None
                else "pass" if task["exit_code"] == 0 else "failed"
            )
            if action == "aot":
                if not result["results"] or not all(
                    item.get("compile_time") is not None for item in result["results"]
                ):
                    status = "failed"
                evidence = {
                    "jobs": len(result["jobs"]),
                    "wall_seconds": result["wall_seconds"],
                }
            else:
                error = result["cosine_diff"]
                latency = result["input_csv_kernel_sum_us"]
                if not (
                    math.isfinite(error)
                    and error <= 0.1
                    and math.isfinite(latency)
                    and latency > 0
                    and result["finite"]
                ):
                    status = "failed"
                if (result["kernelName1"], result["kernelName2"]) != (
                    fixture["kn1"],
                    fixture["kn2"],
                ):
                    status = "failed"
                if (
                    action == "precompiled"
                    and result["environment"].get("FLYDSL_RUNTIME_RUN_ONLY") != "1"
                ):
                    status = "failed"
                evidence = {
                    "cosine_diff": error,
                    "kernel_sum_us": latency,
                    "wall_seconds": result["wall_seconds"],
                    "elapsed_median_us": result[
                        "matched_synchronized_elapsed_median_us"
                    ],
                }
            record["actions"][action] = {
                "status": status,
                "exit_code": task.get("exit_code"),
                "commit": task["implementation_commit"],
                "result": str(result_path),
                "process_start_utc": task.get("process_start_utc"),
                "process_exit_utc": task.get("process_exit_utc"),
                "environment": result["environment"],
                "fingerprints": result.get("fingerprints"),
                "compile_jobs": result.get("jobs"),
                **evidence,
            }
        statuses = {item["status"] for item in record["actions"].values()}
        record["status"] = (
            "failed"
            if "failed" in statuses
            else "incomplete" if "incomplete" in statuses else "pass"
        )
        summary.append(record)
    args.output.write_text(
        json.dumps(
            {
                "fixture_count": len(summary),
                "passed": sum(item["status"] == "pass" for item in summary),
                "incomplete": sum(item["status"] == "incomplete" for item in summary),
                "failed": sum(item["status"] == "failed" for item in summary),
                "fixtures": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

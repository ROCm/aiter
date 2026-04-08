#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from tabulate import tabulate

API_BASE = "https://api.github.com"
GENERIC_LABELS = {
    "self-hosted",
    "linux",
    "windows",
    "macos",
    "x64",
    "x86_64",
    "arm64",
    "default",
    "ubuntu-latest",
    "ubuntu-22.04",
    "ubuntu-24.04",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query GitHub Actions jobs and print status summary."
    )
    parser.add_argument(
        "--repo", required=True, help="GitHub repository in owner/repo format."
    )
    parser.add_argument(
        "--workflows",
        required=True,
        help="Comma-separated workflow file names.",
    )
    parser.add_argument("--job", default="", help="Optional exact job name filter.")
    parser.add_argument(
        "--hours", type=int, default=24, help="Lookback window in hours."
    )
    parser.add_argument(
        "--runner-report",
        action="store_true",
        help="Print runner concurrency summary by runner label.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print markdown table summary.",
    )
    parser.add_argument(
        "--snapshot-in",
        default="",
        help="Optional path to pre-fetched snapshot JSON.",
    )
    parser.add_argument(
        "--snapshot-out",
        default="",
        help="Optional path to write fetched snapshot JSON.",
    )
    return parser.parse_args()


def iso_to_datetime(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_time(value: str):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def format_time(value: str):
    if not value:
        return "-"
    parsed = parse_time(value)
    if not parsed:
        return "-"
    return parsed.astimezone(timezone.utc).strftime("%m-%d %H:%M")


class RateLimitExceededError(RuntimeError):
    def __init__(self, reset_epoch: int | None):
        self.reset_epoch = reset_epoch
        super().__init__("GitHub API rate limit exceeded.")


def github_get(url: str, token: str, params=None):
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        response = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            params=params or {},
            timeout=30,
        )

        if response.status_code == 403:
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset_header = response.headers.get("X-RateLimit-Reset")
            reset_epoch = int(reset_header) if reset_header else None
            is_limited = remaining == "0" or "rate limit" in response.text.lower()
            if is_limited:
                if attempt < max_attempts and reset_epoch is not None:
                    wait_seconds = max(1, min(reset_epoch - int(time.time()) + 1, 30))
                    print(
                        f"[warn] GitHub API rate limited. "
                        f"Retrying in {wait_seconds}s (attempt {attempt}/{max_attempts})."
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RateLimitExceededError(reset_epoch)

        response.raise_for_status()
        return response.json()

    raise RuntimeError("Unexpected retry loop exit in github_get.")


def split_csv(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def workflow_file_from_path(path_value: str):
    normalized = path_value.split("@", 1)[0]
    return Path(normalized).name


def normalize_labels(raw_labels: Any):
    if not raw_labels:
        return []
    if isinstance(raw_labels, str):
        return split_csv(raw_labels)
    return [str(label).strip() for label in raw_labels if str(label).strip()]


def list_recent_runs(
    owner: str, repo: str, workflows: list[str], token: str, lookback: datetime
):
    workflow_set = set(workflows)
    page = 1
    while True:
        payload = github_get(
            f"{API_BASE}/repos/{owner}/{repo}/actions/runs",
            token,
            params={"per_page": 100, "page": page},
        )
        runs = payload.get("workflow_runs", [])
        if not runs:
            return

        stop = False
        for run in runs:
            created_at = iso_to_datetime(run["created_at"])
            if created_at < lookback:
                stop = True
                continue
            workflow_path = run.get("path", "")
            workflow_file = (
                workflow_file_from_path(workflow_path) if workflow_path else ""
            )
            if workflow_file in workflow_set:
                run["workflow_file"] = workflow_file
                yield run

        if stop:
            return
        page += 1


def list_jobs_for_run(owner: str, repo: str, run_id: int, token: str):
    page = 1
    while True:
        payload = github_get(
            f"{API_BASE}/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
            token,
            params={"per_page": 100, "page": page},
        )
        jobs = payload.get("jobs", [])
        if not jobs:
            return
        for job in jobs:
            yield job
        page += 1


def job_name_matches(filter_name: str, actual_name: str):
    if not filter_name:
        return True
    if actual_name == filter_name:
        return True
    if actual_name.startswith(f"{filter_name} ("):
        return True
    if actual_name.startswith(f"{filter_name} / "):
        return True
    return False


def row_to_dict(row: dict[str, Any]):
    return row


def row_from_dict(data: Any):
    if isinstance(data, list):
        padded = data + [""] * max(0, 13 - len(data))
        return {
            "workflow": padded[0] or "-",
            "job": padded[1] or "-",
            "runner": padded[2] or "-",
            "runner_group": padded[3] or "-",
            "status": padded[4] or "-",
            "conclusion": padded[5] or "-",
            "branch": padded[6] or "-",
            "run_url": padded[7] or "-",
            "created_at": "",
            "started_at": "",
            "completed_at": "",
            "html_url": padded[7] or "-",
            "labels": [],
        }
    return {
        "workflow": data.get("workflow", "-"),
        "job": data.get("job", "-"),
        "runner": data.get("runner", "-"),
        "runner_group": data.get("runner_group", "-"),
        "status": data.get("status", "-"),
        "conclusion": data.get("conclusion", "-"),
        "branch": data.get("branch", "-"),
        "run_url": data.get("run_url", "-"),
        "created_at": data.get("created_at", ""),
        "started_at": data.get("started_at", ""),
        "completed_at": data.get("completed_at", ""),
        "html_url": data.get("html_url", data.get("run_url", "-")),
        "labels": normalize_labels(data.get("labels", []) or []),
    }


def parse_row_created_at(row: dict[str, Any]):
    created_at = row.get("created_at")
    if not created_at:
        return None
    if isinstance(created_at, datetime):
        return (
            created_at
            if created_at.tzinfo is not None
            else created_at.replace(tzinfo=timezone.utc)
        )
    if not isinstance(created_at, str):
        return None
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def calculate_duration(started_at: str, completed_at: str):
    started = parse_time(started_at)
    completed = parse_time(completed_at)
    if not started or not completed:
        return "-"
    duration_seconds = (completed - started).total_seconds()
    if duration_seconds < 0:
        return "-"
    return format_duration_seconds(duration_seconds)


def queue_time_seconds(row: dict[str, Any], report_time: datetime):
    created = parse_time(row.get("created_at", ""))
    if not created:
        return None

    runner_name = row.get("runner") or "-"
    status = row.get("status", "-")
    if not runner_name or runner_name == "-":
        if status not in ("queued", "waiting"):
            return None
        queue_seconds = (report_time - created).total_seconds()
        return queue_seconds if queue_seconds >= 0 else None

    started = parse_time(row.get("started_at", ""))
    if not started:
        return None
    queue_seconds = (started - created).total_seconds()
    return queue_seconds if queue_seconds >= 0 else None


def calculate_queue_time(row: dict[str, Any], report_time: datetime):
    queue_seconds = queue_time_seconds(row, report_time)
    if queue_seconds is None:
        return "-"
    suffix = " (queuing)" if row.get("status") in ("queued", "waiting") else ""
    return f"{format_duration_seconds(queue_seconds)}{suffix}"


def average(values: list[float]):
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: list[float], percent: int):
    if not values:
        return None
    ordered = sorted(values)
    clamped_percent = max(0, min(percent, 100))
    position = (len(ordered) - 1) * clamped_percent / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def format_duration_seconds(seconds: float | None):
    if seconds is None or seconds < 0:
        return "-"
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h{minutes}m"
    return f"{minutes}m{secs}s"


def get_custom_runner_labels(row: dict[str, Any]):
    labels = normalize_labels(row.get("labels", []) or [])
    return [
        label
        for label in labels
        if label.lower() not in GENERIC_LABELS
        and not label.lower().startswith("ubuntu-")
    ]


def select_primary_runner_label(labels: list[str]):
    if not labels:
        return ""

    preferred = [
        label
        for label in labels
        if any(
            token in label.lower()
            for token in ("mi", "gpu", "runner", "aiter", "linux-")
        )
    ]
    candidates = preferred or labels
    return sorted(candidates, key=lambda value: (-len(value), value.lower()))[0]


def get_runner_label(row: dict[str, Any]):
    label = select_primary_runner_label(get_custom_runner_labels(row))
    if label:
        return label
    runner_name = row.get("runner") or "-"
    return runner_name if runner_name and runner_name != "-" else "unknown"


def runner_label_sort_key(label: str):
    lowered = label.lower()
    family_match = re.search(r"(mi\d+[a-z0-9x]*)", lowered)
    family = family_match.group(1) if family_match else "zzz"

    count = 0
    for pattern in (r"(\d+)\s*gpu", r"gpu[-_]?(\d+)", r"-(\d+)$"):
        match = re.search(pattern, lowered)
        if match:
            count = int(match.group(1))
            break

    return (family, count, lowered)


def get_concurrency_label(row: dict[str, Any]):
    return select_primary_runner_label(get_custom_runner_labels(row))


def analyze_concurrency(job_rows: list[dict[str, Any]], report_time: datetime):
    stats_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in job_rows:
        label = get_concurrency_label(row)
        if label:
            stats_by_label[label].append(row)

    results = {}
    for label in sorted(stats_by_label, key=runner_label_sort_key):
        label_rows = stats_by_label[label]
        events: list[tuple[datetime, int]] = []
        queue_times: list[float] = []
        durations: list[float] = []

        for row in label_rows:
            created = parse_time(row.get("created_at", ""))
            started = parse_time(row.get("started_at", ""))
            completed = parse_time(row.get("completed_at", ""))

            if started and completed and completed >= started:
                events.append((started, +1))
                events.append((completed, -1))
                durations.append((completed - started).total_seconds())
            elif started:
                events.append((started, +1))
                events.append((report_time, -1))
                durations.append((report_time - started).total_seconds())

            queue_seconds = queue_time_seconds(row, report_time)
            if queue_seconds is not None:
                queue_times.append(queue_seconds)

        if not events:
            results[label] = {
                "peak": 0,
                "avg_concurrent": 0.0,
                "total_jobs": len(label_rows),
                "avg_queue_seconds": average(queue_times),
                "p50_queue_seconds": percentile(queue_times, 50),
                "p99_queue_seconds": percentile(queue_times, 99),
                "avg_duration_seconds": average(durations),
            }
            continue

        events.sort(key=lambda item: (item[0], item[1]))
        concurrent = 0
        peak = 0
        time_weighted_sum = 0.0
        total_time = 0.0
        previous_time = events[0][0]

        for timestamp, delta in events:
            if concurrent > 0:
                elapsed = (timestamp - previous_time).total_seconds()
                if elapsed > 0:
                    time_weighted_sum += concurrent * elapsed
                    total_time += elapsed
            concurrent += delta
            peak = max(peak, concurrent)
            previous_time = timestamp

        results[label] = {
            "peak": peak,
            "avg_concurrent": (
                round(time_weighted_sum / total_time, 1) if total_time > 0 else 0.0
            ),
            "total_jobs": len(label_rows),
            "avg_queue_seconds": average(queue_times),
            "p50_queue_seconds": percentile(queue_times, 50),
            "p99_queue_seconds": percentile(queue_times, 99),
            "avg_duration_seconds": average(durations),
        }

    return results


def build_queue_distribution(queue_times: list[float]):
    if not queue_times:
        return []
    ranges = [
        ("< 1 min", 0, 60),
        ("1-5 min", 60, 300),
        ("5-15 min", 300, 900),
        ("15-30 min", 900, 1800),
        ("30-60 min", 1800, 3600),
        ("> 60 min", 3600, float("inf")),
    ]
    total = len(queue_times)
    buckets = []
    for label, lower, upper in ranges:
        count = sum(1 for value in queue_times if lower <= value < upper)
        percentage = round(count / total * 100, 1) if total else 0.0
        buckets.append([label, count, f"{percentage}%"])
    return buckets


def build_runner_report_rows(job_rows: list[dict[str, Any]], report_time: datetime):
    stats = defaultdict(
        lambda: {
            "total": 0,
            "running": 0,
            "queued": 0,
            "waiting": 0,
            "success": 0,
            "failure": 0,
            "cancelled": 0,
            "queue_samples": [],
            "duration_samples": [],
        }
    )

    for row in job_rows:
        label = get_runner_label(row)
        if label in ("unknown", "ubuntu-latest"):
            continue

        stat = stats[label]
        stat["total"] += 1
        status = row.get("status", "-")
        conclusion = row.get("conclusion", "-")

        if status == "completed":
            if conclusion == "success":
                stat["success"] += 1
            elif conclusion == "failure":
                stat["failure"] += 1
            elif conclusion in ("cancelled", "timed_out", "action_required"):
                stat["cancelled"] += 1
        elif status == "in_progress":
            stat["running"] += 1
        elif status == "queued":
            stat["queued"] += 1
        elif status == "waiting":
            stat["waiting"] += 1

        queue_seconds = queue_time_seconds(row, report_time)
        if queue_seconds is not None:
            stat["queue_samples"].append(queue_seconds)

        started = parse_time(row.get("started_at", ""))
        completed = parse_time(row.get("completed_at", ""))
        if started and completed and completed >= started:
            stat["duration_samples"].append((completed - started).total_seconds())

    summary_rows = []
    distribution_sections = []
    for label in sorted(stats, key=runner_label_sort_key):
        stat = stats[label]
        summary_rows.append(
            [
                label,
                stat["total"],
                stat["running"],
                stat["queued"],
                stat["waiting"],
                stat["success"],
                stat["failure"],
                stat["cancelled"],
                format_duration_seconds(average(stat["queue_samples"])),
                format_duration_seconds(percentile(stat["queue_samples"], 50)),
                format_duration_seconds(percentile(stat["queue_samples"], 90)),
                format_duration_seconds(percentile(stat["queue_samples"], 99)),
                format_duration_seconds(average(stat["duration_samples"])),
            ]
        )
        distribution_sections.append(
            (label, build_queue_distribution(stat["queue_samples"]))
        )

    return summary_rows, distribution_sections


def build_job_report_rows(job_rows: list[dict[str, Any]], report_time: datetime):
    rows = []
    for row in sorted(
        job_rows, key=lambda item: item.get("created_at", ""), reverse=True
    ):
        rows.append(
            [
                row.get("workflow", "-"),
                row.get("job", "-"),
                get_runner_label(row),
                row.get("runner", "-"),
                row.get("runner_group", "-"),
                row.get("status", "-"),
                row.get("conclusion", "-"),
                format_time(row.get("created_at", "")),
                format_time(row.get("started_at", "")),
                calculate_queue_time(row, report_time),
                calculate_duration(
                    row.get("started_at", ""), row.get("completed_at", "")
                ),
                row.get("branch", "-"),
                row.get("html_url") or row.get("run_url", "-"),
            ]
        )
    return rows


def main():
    args = parse_args()
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not args.snapshot_in and not token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required.")

    owner, repo = args.repo.split("/", 1)
    workflows = split_csv(args.workflows)
    if not workflows:
        raise RuntimeError("No workflows specified. Please pass --workflows.")

    lookback = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    report_time = datetime.now(timezone.utc)
    rate_limited = False
    rate_limit_reset_epoch = None

    all_rows: list[dict[str, Any]] = []
    if args.snapshot_in:
        snapshot_payload = json.loads(
            Path(args.snapshot_in).read_text(encoding="utf-8")
        )
        snapshot_generated_at = snapshot_payload.get("generated_at")
        if isinstance(snapshot_generated_at, str):
            try:
                parsed_report_time = datetime.fromisoformat(
                    snapshot_generated_at.replace("Z", "+00:00")
                )
                report_time = (
                    parsed_report_time
                    if parsed_report_time.tzinfo is not None
                    else parsed_report_time.replace(tzinfo=timezone.utc)
                )
            except ValueError:
                pass
        all_rows = [row_from_dict(item) for item in snapshot_payload.get("rows", [])]
    else:
        try:
            runs = list_recent_runs(owner, repo, workflows, token, lookback)
            for run in runs:
                run_id = run["id"]
                run_url = run["html_url"]
                branch = run.get("head_branch") or "-"
                workflow = run.get("workflow_file", "-")
                for job in list_jobs_for_run(owner, repo, run_id, token):
                    all_rows.append(
                        {
                            "workflow": workflow,
                            "job": job.get("name", "-"),
                            "runner": job.get("runner_name") or "-",
                            "runner_group": job.get("runner_group_name") or "-",
                            "status": job.get("status") or "-",
                            "conclusion": job.get("conclusion") or "-",
                            "branch": branch,
                            "run_url": run_url,
                            "created_at": job.get("created_at", ""),
                            "started_at": job.get("started_at", ""),
                            "completed_at": job.get("completed_at", ""),
                            "html_url": job.get("html_url", ""),
                            "labels": normalize_labels(job.get("labels", []) or []),
                        }
                    )
        except RateLimitExceededError as exc:
            rate_limited = True
            rate_limit_reset_epoch = exc.reset_epoch
            print("[warn] GitHub API rate limit exceeded during report generation.")
        except requests.HTTPError as exc:
            print(f"[warn] Failed to query workflow runs: {exc}")

    if args.snapshot_out:
        snapshot_payload = {
            "generated_at": report_time.isoformat(),
            "rows": [row_to_dict(row) for row in all_rows],
        }
        Path(args.snapshot_out).write_text(
            json.dumps(snapshot_payload, ensure_ascii=False), encoding="utf-8"
        )

    workflow_set = set(workflows)
    job_rows = [
        row
        for row in all_rows
        if row.get("workflow") in workflow_set
        and job_name_matches(args.job, row.get("job", ""))
        and (created_at := parse_row_created_at(row)) is not None
        and created_at >= lookback
    ]

    if not job_rows:
        print("No matching job records in the selected time window.")
        if rate_limited and rate_limit_reset_epoch:
            reset_time = datetime.fromtimestamp(rate_limit_reset_epoch, timezone.utc)
            print(
                f"[warn] Rate limit resets at {reset_time.isoformat()} (UTC). "
                "Re-run after reset for complete data."
            )
        return

    tablefmt = "github" if args.summary else "grid"

    if args.runner_report:
        concurrency = analyze_concurrency(job_rows, report_time)
        print("## Concurrency by Runner Label")
        if concurrency:
            print(
                tabulate(
                    [
                        [
                            label,
                            values["peak"],
                            values["avg_concurrent"],
                            values["total_jobs"],
                            format_duration_seconds(values["avg_queue_seconds"]),
                            format_duration_seconds(values["p50_queue_seconds"]),
                            format_duration_seconds(values["p99_queue_seconds"]),
                            format_duration_seconds(values["avg_duration_seconds"]),
                        ]
                        for label, values in sorted(
                            concurrency.items(),
                            key=lambda item: runner_label_sort_key(item[0]),
                        )
                    ],
                    headers=[
                        "runner_label",
                        "peak_concurrent",
                        "avg_concurrent",
                        "total_jobs",
                        "avg_queue",
                        "p50_queue",
                        "p99_queue",
                        "avg_duration",
                    ],
                    tablefmt=tablefmt,
                )
            )
        else:
            print(
                "No matching self-hosted runner labels found in the selected time window."
            )
    else:
        print("### Job Status Report")
        print(
            tabulate(
                build_job_report_rows(job_rows, report_time),
                headers=[
                    "workflow",
                    "job",
                    "runner_label",
                    "runner",
                    "runner_group",
                    "status",
                    "conclusion",
                    "created",
                    "started",
                    "queue",
                    "duration",
                    "branch",
                    "url",
                ],
                tablefmt=tablefmt,
            )
        )

    if rate_limited and rate_limit_reset_epoch:
        reset_time = datetime.fromtimestamp(rate_limit_reset_epoch, timezone.utc)
        print("")
        print(
            f"> NOTE: Partial data due to GitHub API rate limit. "
            f"Reset at {reset_time.isoformat()} (UTC)."
        )


if __name__ == "__main__":
    main()

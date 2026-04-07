#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from tabulate import tabulate

API_BASE = "https://api.github.com"
GENERIC_RUNNER_LABELS = {
    "self-hosted",
    "linux",
    "windows",
    "macos",
    "x64",
    "x86_64",
    "arm64",
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
        help="Print runner fleet summary grouped by runner label.",
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


def split_csv(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def workflow_file_from_path(path_value: str):
    normalized = path_value.split("@", 1)[0]
    return Path(normalized).name


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


def normalize_labels(raw_labels):
    if not raw_labels:
        return []
    if isinstance(raw_labels, str):
        return split_csv(raw_labels)
    return [str(label).strip() for label in raw_labels if str(label).strip()]


def record_from_job(workflow: str, branch: str, run_url: str, job: dict):
    return {
        "workflow": workflow,
        "job": job.get("name", "-"),
        "runner": job.get("runner_name") or "-",
        "runner_group": job.get("runner_group_name") or "-",
        "status": job.get("status") or "-",
        "conclusion": job.get("conclusion") or "-",
        "branch": branch or "-",
        "run_url": run_url or "-",
        "job_url": job.get("html_url") or run_url or "-",
        "created_at": job.get("created_at") or "",
        "started_at": job.get("started_at") or "",
        "completed_at": job.get("completed_at") or "",
        "labels": normalize_labels(job.get("labels")),
    }


def normalize_record(data: dict):
    return {
        "workflow": data.get("workflow", "-"),
        "job": data.get("job", "-"),
        "runner": data.get("runner", "-"),
        "runner_group": data.get("runner_group", "-"),
        "status": data.get("status", "-"),
        "conclusion": data.get("conclusion", "-"),
        "branch": data.get("branch", "-"),
        "run_url": data.get("run_url", "-"),
        "job_url": data.get("job_url") or data.get("run_url", "-"),
        "created_at": data.get("created_at", ""),
        "started_at": data.get("started_at", ""),
        "completed_at": data.get("completed_at", ""),
        "labels": normalize_labels(data.get("labels")),
    }


def load_snapshot_records(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_records = payload.get("rows") or payload.get("jobs") or []
    return [normalize_record(item) for item in raw_records]


def write_snapshot_records(path: str, records: list[dict]):
    snapshot_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": records,
    }
    Path(path).write_text(
        json.dumps(snapshot_payload, ensure_ascii=False), encoding="utf-8"
    )


def format_seconds(seconds: float | None):
    if seconds is None:
        return "-"
    total_seconds = int(round(seconds))
    if total_seconds < 0:
        return "-"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m{secs}s"


def average(values: list[float]):
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: list[float], p: int):
    if not values:
        return None
    sorted_values = sorted(values)
    index = min(int(len(sorted_values) * p / 100), len(sorted_values) - 1)
    return sorted_values[index]


def extract_runner_label(record: dict):
    labels = normalize_labels(record.get("labels"))
    custom_labels = [
        label
        for label in labels
        if label.lower() not in GENERIC_RUNNER_LABELS
        and not label.lower().startswith("ubuntu-")
    ]
    if not custom_labels:
        return ""

    preferred = [
        label
        for label in custom_labels
        if any(
            token in label.lower()
            for token in ("mi", "gpu", "runner", "aiter", "linux-")
        )
    ]
    candidates = preferred or custom_labels
    return sorted(candidates, key=lambda value: (-len(value), value.lower()))[0]


def runner_label_sort_key(label: str):
    lowered = label.lower()
    family_match = re.search(r"(mi\d+[a-z0-9x]*)", lowered)
    family = family_match.group(1) if family_match else lowered

    gpu_count = 0
    for pattern in (r"(\d+)\s*gpu", r"gpu[-_]?(\d+)", r"-(\d+)$"):
        match = re.search(pattern, lowered)
        if match:
            gpu_count = int(match.group(1))
            break

    return (family, gpu_count, lowered)


def queue_time_seconds(record: dict, report_time: datetime):
    created_at = parse_time(record.get("created_at", ""))
    if not created_at:
        return None

    runner = record.get("runner") or ""
    if runner and runner != "-":
        started_at = parse_time(record.get("started_at", ""))
        if not started_at:
            return None
        queue_seconds = (started_at - created_at).total_seconds()
        return queue_seconds if queue_seconds >= 0 else None

    if record.get("status") not in ("queued", "waiting"):
        return None

    queue_seconds = (report_time - created_at).total_seconds()
    return queue_seconds if queue_seconds >= 0 else None


def duration_seconds(record: dict, report_time: datetime):
    runner = record.get("runner") or ""
    if not runner or runner == "-":
        return None

    started_at = parse_time(record.get("started_at", ""))
    if not started_at:
        return None

    completed_at = parse_time(record.get("completed_at", ""))
    end_time = completed_at or report_time
    duration = (end_time - started_at).total_seconds()
    return duration if duration >= 0 else None


def analyze_runner_labels(records: list[dict], report_time: datetime):
    by_label = defaultdict(list)
    for record in records:
        label = extract_runner_label(record)
        if not label:
            continue
        by_label[label].append(record)

    report = {}
    for label, items in by_label.items():
        queue_samples = []
        duration_samples = []
        events = []

        for item in items:
            queue_sample = queue_time_seconds(item, report_time)
            if queue_sample is not None:
                queue_samples.append(queue_sample)

            duration_sample = duration_seconds(item, report_time)
            if duration_sample is not None:
                duration_samples.append(duration_sample)

            runner = item.get("runner") or ""
            started_at = parse_time(item.get("started_at", ""))
            if not runner or runner == "-" or not started_at:
                continue

            completed_at = parse_time(item.get("completed_at", ""))
            end_time = completed_at or report_time
            if end_time < started_at:
                continue

            events.append((started_at, 1))
            events.append((end_time, -1))

        peak = 0
        avg_concurrent = 0.0
        if events:
            events.sort(key=lambda entry: (entry[0], entry[1]))
            concurrent = 0
            weighted_sum = 0.0
            active_window = 0.0
            prev_time = events[0][0]

            for timestamp, delta in events:
                if concurrent > 0:
                    elapsed = (timestamp - prev_time).total_seconds()
                    if elapsed > 0:
                        weighted_sum += concurrent * elapsed
                        active_window += elapsed
                concurrent += delta
                peak = max(peak, concurrent)
                prev_time = timestamp

            if active_window > 0:
                avg_concurrent = weighted_sum / active_window

        report[label] = {
            "peak": peak,
            "avg_concurrent": round(avg_concurrent, 1),
            "total_jobs": len(items),
            "avg_queue_seconds": average(queue_samples),
            "p50_queue_seconds": percentile(queue_samples, 50),
            "p99_queue_seconds": percentile(queue_samples, 99),
            "avg_duration_seconds": average(duration_samples),
        }

    return report


def render_job_report(records: list[dict]):
    table = [
        [
            record["workflow"],
            record["job"],
            record["runner"],
            record["runner_group"],
            record["status"],
            record["conclusion"],
            record["branch"],
            record["run_url"],
        ]
        for record in records
    ]
    return tabulate(
        table,
        headers=[
            "workflow",
            "job",
            "runner",
            "runner_group",
            "status",
            "conclusion",
            "branch",
            "run_url",
        ],
        tablefmt="github",
    )


def render_runner_report(records: list[dict], report_time: datetime):
    label_summary = analyze_runner_labels(records, report_time)
    if not label_summary:
        return "No matching self-hosted runner label records in the selected time window."

    lines = [
        "| Runner Label | Peak Concurrent | Avg Concurrent | Total Jobs | Avg Queue | P50 Queue | P99 Queue | Avg Duration |",
        "|-------------|-----------------|----------------|------------|-----------|-----------|-----------|--------------|",
    ]
    for label in sorted(label_summary, key=runner_label_sort_key):
        stats = label_summary[label]
        lines.append(
            f"| `{label}` | {stats['peak']} | {stats['avg_concurrent']:.1f} | "
            f"{stats['total_jobs']} | {format_seconds(stats['avg_queue_seconds'])} | "
            f"{format_seconds(stats['p50_queue_seconds'])} | "
            f"{format_seconds(stats['p99_queue_seconds'])} | "
            f"{format_seconds(stats['avg_duration_seconds'])} |"
        )

    return "\n".join(lines)


def main():
    args = parse_args()
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token and not args.snapshot_in:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required.")

    owner, repo = args.repo.split("/", 1)
    workflows = split_csv(args.workflows)
    if not workflows:
        raise RuntimeError("No workflows specified. Please pass --workflows.")

    lookback = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    report_time = datetime.now(timezone.utc)
    rate_limited = False
    rate_limit_reset_epoch = None

    all_records = []
    if args.snapshot_in:
        all_records = load_snapshot_records(args.snapshot_in)
    else:
        try:
            runs = list_recent_runs(owner, repo, workflows, token, lookback)
            for run in runs:
                run_url = run["html_url"]
                branch = run.get("head_branch") or "-"
                workflow = run.get("workflow_file", "-")
                for job in list_jobs_for_run(owner, repo, run_id, token):
                    all_records.append(record_from_job(workflow, branch, run_url, job))
        except RateLimitExceededError as exc:
            rate_limited = True
            rate_limit_reset_epoch = exc.reset_epoch
            print("[warn] GitHub API rate limit exceeded during report generation.")
        except requests.HTTPError as exc:
            print(f"[warn] Failed to query workflow runs: {exc}")

    if args.snapshot_out:
        write_snapshot_records(args.snapshot_out, all_records)

    workflow_set = set(workflows)
    matching_records = [
        record
        for record in all_records
        if record["workflow"] in workflow_set and job_name_matches(args.job, record["job"])
    ]

    if args.runner_report:
        if not args.summary:
            print("=== Runner Fleet Summary ===")
        print(render_runner_report(matching_records, report_time))
        if rate_limited and rate_limit_reset_epoch:
            reset_time = datetime.fromtimestamp(rate_limit_reset_epoch, timezone.utc)
            print("")
            print(
                f"> NOTE: Partial data due to GitHub API rate limit. "
                f"Reset at {reset_time.isoformat()} (UTC)."
            )
        return

    if not matching_records:
        print("No matching job records in the selected time window.")
        if rate_limited and rate_limit_reset_epoch:
            reset_time = datetime.fromtimestamp(rate_limit_reset_epoch, timezone.utc)
            print(
                f"[warn] Rate limit resets at {reset_time.isoformat()} (UTC). "
                "Re-run after reset for complete data."
            )
        return

    if args.summary:
        print("=== Job Status Report ===")
    print(render_job_report(matching_records))


if __name__ == "__main__":
    main()

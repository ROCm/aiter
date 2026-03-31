#!/usr/bin/env python3
import argparse
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from tabulate import tabulate

API_BASE = "https://api.github.com"


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
        help="Print runner utilization summary.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print markdown table summary.",
    )
    return parser.parse_args()


def iso_to_datetime(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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

        # Handle API rate-limit gracefully with small bounded retry.
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
    # Example: ".github/workflows/pre-checks.yaml@refs/heads/main"
    normalized = path_value.split("@", 1)[0]
    return Path(normalized).name


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
            workflow_file = workflow_file_from_path(workflow_path) if workflow_path else ""
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


def main():
    args = parse_args()
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required.")

    owner, repo = args.repo.split("/", 1)
    workflows = split_csv(args.workflows)
    if not workflows:
        raise RuntimeError("No workflows specified. Please pass --workflows.")
    lookback = datetime.now(timezone.utc) - timedelta(hours=args.hours)

    job_rows = []
    runner_stats = defaultdict(lambda: defaultdict(int))
    rate_limited = False
    rate_limit_reset_epoch = None

    try:
        runs = list_recent_runs(owner, repo, workflows, token, lookback)
        for run in runs:
            run_id = run["id"]
            run_url = run["html_url"]
            branch = run.get("head_branch") or "-"
            workflow = run.get("workflow_file", "-")
            for job in list_jobs_for_run(owner, repo, run_id, token):
                if args.job and job["name"] != args.job:
                    continue

                runner_name = job.get("runner_name") or "-"
                runner_group = job.get("runner_group_name") or "-"
                status = job.get("status") or "-"
                conclusion = job.get("conclusion") or "-"

                job_rows.append(
                    [
                        workflow,
                        job["name"],
                        runner_name,
                        runner_group,
                        status,
                        conclusion,
                        branch,
                        run_url,
                    ]
                )

                key = (workflow, runner_name, runner_group)
                runner_stats[key]["total"] += 1
                runner_stats[key][conclusion] += 1
    except RateLimitExceededError as exc:
        rate_limited = True
        rate_limit_reset_epoch = exc.reset_epoch
        print("[warn] GitHub API rate limit exceeded during report generation.")
    except requests.HTTPError as exc:
        print(f"[warn] Failed to query workflow runs: {exc}")

    if not job_rows:
        print("No matching job records in the selected time window.")
        if rate_limited and rate_limit_reset_epoch:
            reset_time = datetime.fromtimestamp(rate_limit_reset_epoch, timezone.utc)
            print(
                f"[warn] Rate limit resets at {reset_time.isoformat()} (UTC). "
                "Re-run after reset for complete data."
            )
        return

    if args.summary and not args.runner_report:
        print("=== Job Status Report ===")
        print(
            tabulate(
                job_rows,
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
        )

    if args.runner_report:
        rows = []
        for (workflow, runner_name, runner_group), counts in sorted(
            runner_stats.items()
        ):
            rows.append(
                [
                    workflow,
                    runner_name,
                    runner_group,
                    counts.get("total", 0),
                    counts.get("success", 0),
                    counts.get("failure", 0),
                    counts.get("cancelled", 0),
                    counts.get("timed_out", 0),
                    counts.get("skipped", 0),
                ]
            )
        print("=== Runner Fleet Summary ===")
        print(
            tabulate(
                rows,
                headers=[
                    "workflow",
                    "runner",
                    "runner_group",
                    "total",
                    "success",
                    "failure",
                    "cancelled",
                    "timed_out",
                    "skipped",
                ],
                tablefmt="github",
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

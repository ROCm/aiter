#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
WORKFLOW_NAME = "Aiter Test"
FAILURE_PAT = re.compile(
    r"(FAILED|ERROR|Traceback|AssertionError|RuntimeError|Exception|"
    r"Process completed with exit code|timed out|Timeout|segmentation fault|core dumped)",
    re.IGNORECASE,
)
WORD_PAT = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "aiter",
    "test",
    "tests",
    "gpu",
    "linux",
    "runner",
    "runners",
    "standard",
    "multi",
    "single",
    "shard",
    "shards",
    "true",
    "false",
    "main",
    "build",
    "run",
    "runs",
    "workflow",
}


class CommandError(RuntimeError):
    def __init__(self, args: list[str], rc: int, out: str, err: str) -> None:
        super().__init__(f"{' '.join(args)} failed with exit code {rc}: {err.strip()}")
        self.args_list = args
        self.rc = rc
        self.out = out
        self.err = err


def run_cmd(
    args: list[str],
    *,
    check: bool = True,
    timeout: int = 90,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        cwd=str(cwd or ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise CommandError(args, proc.returncode, proc.stdout, proc.stderr)
    return proc


def gh_json(args: list[str], *, timeout: int = 90) -> Any:
    proc = run_cmd(args, timeout=timeout)
    return json.loads(proc.stdout or "null")


def clip(text: str, limit: int = 1600) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [truncated]"


def md_escape(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def run_id_of(run: dict[str, Any]) -> str:
    return str(run.get("databaseId") or run.get("id") or run.get("run_id") or "")


def normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "databaseId": run.get("databaseId") or run.get("id"),
        "conclusion": run.get("conclusion") or "",
        "createdAt": run.get("createdAt") or run.get("created_at") or "",
        "event": run.get("event") or "",
        "headBranch": run.get("headBranch") or run.get("head_branch") or "",
        "headSha": run.get("headSha") or run.get("head_sha") or "",
        "name": run.get("name") or run.get("display_title") or WORKFLOW_NAME,
        "status": run.get("status") or "",
        "url": run.get("url") or run.get("html_url") or "",
    }


def public_run_url(repo: str, run: dict[str, Any]) -> str:
    run_id = run_id_of(run)
    if not run_id:
        return run.get("url", "")
    return f"https://github.com/{repo}/actions/runs/{run_id}"


def run_summary(repo: str, run: dict[str, Any] | None) -> dict[str, Any] | None:
    if not run:
        return None
    return {
        "id": run_id_of(run),
        "url": public_run_url(repo, run),
        "created_at": run.get("createdAt", ""),
        "event": run.get("event", ""),
        "head_branch": run.get("headBranch", ""),
        "head_sha": run.get("headSha", ""),
        "status": run.get("status", ""),
        "conclusion": run.get("conclusion", ""),
    }


def list_runs(repo: str, lookback: int) -> list[dict[str, Any]]:
    data = gh_json(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--branch",
            "main",
            "--workflow",
            WORKFLOW_NAME,
            "--limit",
            str(lookback),
            "--json",
            "databaseId,name,conclusion,headBranch,headSha,createdAt,event,url,status",
        ],
        timeout=120,
    )
    return [normalize_run(item) for item in data]


def get_run(repo: str, run_id: str) -> dict[str, Any]:
    data = gh_json(["gh", "api", f"repos/{repo}/actions/runs/{run_id}"], timeout=90)
    return normalize_run(data)


def get_jobs(repo: str, run_id: str) -> list[dict[str, Any]]:
    data = gh_json(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"],
        timeout=120,
    )
    return data.get("jobs", [])


def get_artifacts(repo: str, run_id: str) -> list[dict[str, Any]]:
    data = gh_json(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"],
        timeout=120,
    )
    return data.get("artifacts", [])


def failed_steps(job: dict[str, Any]) -> list[str]:
    return [
        step.get("name", "")
        for step in job.get("steps", [])
        if step.get("conclusion") == "failure"
    ]


def is_aggregate_job(name: str) -> bool:
    lower = name.lower()
    return (
        lower == "check-signal"
        or "aiter test gate" in lower
        or "standard test results" in lower
        or "tuned op bench" in lower
    )


def classify_job(job: dict[str, Any]) -> str:
    name = job.get("name", "")
    steps = " ".join(failed_steps(job)).lower()
    lower = name.lower()
    if lower.startswith("build_aiter_wheels") or "build" in steps:
        return "Build"
    if "mori smoke" in steps:
        return "Multi-GPU smoke"
    if "multi-gpu" in lower:
        return "Multi-GPU"
    if "standard tests" in lower and "tests" in steps:
        return "Standard shard"
    if any(token in steps for token in ("download", "restore", "checkout", "setup")):
        return "Infra/setup"
    if is_aggregate_job(name):
        return "Aggregate"
    return "Test/unknown"


def job_summary(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job.get("id"),
        "name": job.get("name", ""),
        "type": classify_job(job),
        "status": job.get("status", ""),
        "conclusion": job.get("conclusion", ""),
        "url": job.get("html_url", ""),
        "started_at": job.get("started_at", ""),
        "completed_at": job.get("completed_at", ""),
        "failed_steps": failed_steps(job),
    }


def parse_standard_job(name: str) -> tuple[str, str, str, str] | None:
    match = re.search(r"\((linux-[^,]+),\s*([^,]+),\s*([0-9]+),\s*([0-9]+)\)", name)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), match.group(4)


def parse_multigpu_runner(name: str) -> str | None:
    match = re.search(r"\((linux-[^,\)]+)", name)
    return match.group(1) if match else None


def artifact_names_for_failed_jobs(failed_jobs: list[dict[str, Any]]) -> set[str]:
    names = {"aiter_shards"}
    for job in failed_jobs:
        job_name = job.get("name", "")
        parsed = parse_standard_job(job_name)
        if parsed:
            runner, _label, _total, shard = parsed
            names.add(f"standard-test-log-{runner}-shard-{shard}")
            continue
        runner = parse_multigpu_runner(job_name)
        if runner and "multi-gpu" in job_name.lower():
            names.add(f"multigpu-test-{runner}")
    return names


def safe_dir_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def download_artifacts(
    repo: str,
    run_id: str,
    out_dir: Path,
    wanted: set[str],
    available: list[dict[str, Any]],
) -> list[str]:
    messages: list[str] = []
    available_names = {item.get("name", "") for item in available if not item.get("expired")}
    for name in sorted(wanted & available_names):
        dest = out_dir / safe_dir_name(name)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            proc = run_cmd(
                ["gh", "run", "download", run_id, "--repo", repo, "--name", name, "--dir", str(dest)],
                check=False,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            messages.append(f"artifact {name}: download timed out")
            continue
        if proc.returncode != 0:
            messages.append(f"artifact {name}: {clip(proc.stderr or proc.stdout, 500)}")
    missing = sorted(wanted - available_names)
    for name in missing:
        messages.append(f"artifact {name}: not found")
    return messages


def download_job_logs(repo: str, failed_jobs: list[dict[str, Any]], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []
    for job in failed_jobs:
        job_id = str(job.get("id", ""))
        if not job_id:
            continue
        dest = out_dir / f"job_{job_id}.log"
        with dest.open("w", encoding="utf-8", errors="replace") as handle:
            try:
                proc = subprocess.run(
                    ["gh", "api", f"repos/{repo}/actions/jobs/{job_id}/logs"],
                    cwd=str(ROOT),
                    text=True,
                    stdout=handle,
                    stderr=subprocess.PIPE,
                    timeout=90,
                )
            except subprocess.TimeoutExpired:
                messages.append(f"job log {job_id}: download timed out")
                try:
                    dest.unlink()
                except FileNotFoundError:
                    pass
                continue
        if proc.returncode != 0:
            messages.append(f"job log {job_id}: {clip(proc.stderr, 500)}")
            try:
                dest.unlink()
            except FileNotFoundError:
                pass
    return messages


def read_text(path: Path, limit: int = 2_000_000) -> str:
    data = path.read_bytes()[:limit]
    return data.decode("utf-8", errors="replace")


def extract_snippets_from_text(text: str, source: str, max_snippets: int = 10) -> list[dict[str, str]]:
    snippets: list[dict[str, str]] = []
    lines = text.splitlines()
    current_test = ""
    for idx, line in enumerate(lines):
        test_match = re.search(r"Running test:\s*([^\s]+)", line)
        if test_match:
            current_test = test_match.group(1)
        if not FAILURE_PAT.search(line):
            continue
        start = max(0, idx - 2)
        end = min(len(lines), idx + 7)
        context = "\n".join(lines[start:end])
        snippets.append(
            {
                "source": source,
                "test": current_test,
                "line": str(idx + 1),
                "context": clip(context, 1200),
            }
        )
        if len(snippets) >= max_snippets:
            break
    return snippets


def collect_log_snippets(work_dir: Path) -> list[dict[str, str]]:
    snippets: list[dict[str, str]] = []
    for path in sorted((work_dir / "artifacts").rglob("latest_test.log")):
        snippets.extend(extract_snippets_from_text(read_text(path), str(path.relative_to(work_dir))))
    for path in sorted((work_dir / "job_logs").glob("*.log")):
        snippets.extend(extract_snippets_from_text(read_text(path), str(path.relative_to(work_dir)), max_snippets=5))
    return snippets[:30]


def read_shard_lists(work_dir: Path) -> dict[str, list[str]]:
    shard_lists: dict[str, list[str]] = {}
    for path in sorted((work_dir / "artifacts").rglob("aiter_shard_*.list")):
        match = re.search(r"aiter_shard_([0-9]+)\.list$", path.name)
        if not match:
            continue
        files = [item for item in read_text(path, limit=200_000).split() if item]
        shard_lists[match.group(1)] = files
    return shard_lists


def have_commit(sha: str) -> bool:
    if not sha:
        return False
    proc = run_cmd(["git", "cat-file", "-e", f"{sha}^{{commit}}"], check=False, timeout=30)
    return proc.returncode == 0


def commit_range(base_sha: str, head_sha: str) -> list[dict[str, Any]]:
    if not have_commit(head_sha):
        return []
    fmt = "%H%x1f%h%x1f%ci%x1f%an%x1f%s"
    if have_commit(base_sha):
        range_spec = f"{base_sha}..{head_sha}"
        proc = run_cmd(["git", "log", "--reverse", f"--format={fmt}", range_spec], check=False, timeout=120)
    else:
        proc = run_cmd(["git", "log", "--reverse", "-n", "50", f"--format={fmt}", head_sha], check=False, timeout=120)
    if proc.returncode != 0:
        proc = run_cmd(["git", "log", "--reverse", "-n", "50", f"--format={fmt}", head_sha], timeout=120)
    commits: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\x1f", 4)
        if len(parts) != 5:
            continue
        sha, short, date, author, subject = parts
        commits.append(
            {
                "sha": sha,
                "short": short,
                "date": date,
                "author": author,
                "subject": subject,
                "files": changed_files(sha),
            }
        )
    return commits


def changed_files(sha: str) -> list[str]:
    proc = run_cmd(["git", "show", "--name-only", "--format=", "--no-renames", sha], check=False, timeout=60)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def words_from_text(text: str) -> set[str]:
    return {
        item.lower()
        for item in WORD_PAT.findall(text)
        if len(item) >= 3 and item.lower() not in STOPWORDS
    }


def build_failure_profile(
    failed_jobs: list[dict[str, Any]],
    snippets: list[dict[str, str]],
    shard_lists: dict[str, list[str]],
) -> dict[str, Any]:
    text_parts: list[str] = []
    failed_test_files: set[str] = set()
    failed_shard_test_files: set[str] = set()
    hardware: set[str] = set()
    categories: set[str] = set()
    for job in failed_jobs:
        name = job.get("name", "")
        text_parts.append(name)
        text_parts.extend(failed_steps(job))
        parsed = parse_standard_job(name)
        if parsed:
            _runner, label, _total, shard = parsed
            hardware.add(label.lower())
            categories.add("standard")
            for file_path in shard_lists.get(shard, []):
                failed_shard_test_files.add(file_path)
        elif "multi-gpu" in name.lower():
            categories.add("multigpu")
            if "MI350" in name or "MI35" in name:
                hardware.add("mi350x")
                hardware.add("gfx950")
        if "build_aiter_wheels" in name:
            categories.add("build")
    for snippet in snippets:
        text_parts.append(snippet.get("context", ""))
        test = snippet.get("test", "")
        if test.startswith("op_tests/") or test.startswith("tests/"):
            failed_test_files.add(test)
    token_text = "\n".join(text_parts + sorted(failed_test_files) + sorted(failed_shard_test_files))
    tokens = words_from_text(token_text)
    for special in (
        "moe",
        "mxfp4",
        "mxfp8",
        "gemm",
        "a8w8",
        "blockscale",
        "gfx942",
        "gfx950",
        "mori",
        "allreduce",
        "custom_ar",
        "multigpu",
        "dpx",
        "triton",
        "flydsl",
        "mha",
        "mla",
        "pa_decode",
        "kvcache",
    ):
        if special in token_text.lower():
            tokens.add(special)
    return {
        "tokens": sorted(tokens),
        "failed_test_files": sorted(failed_test_files),
        "failed_shard_test_files": sorted(failed_shard_test_files),
        "hardware": sorted(hardware),
        "categories": sorted(categories),
    }


def score_commit(commit: dict[str, Any], profile: dict[str, Any]) -> tuple[int, list[str]]:
    subject = commit["subject"].lower()
    files = commit["files"]
    file_text = "\n".join(files).lower()
    haystack = subject + "\n" + file_text
    tokens: set[str] = set(profile["tokens"])
    failed_files: set[str] = set(profile["failed_test_files"])
    failed_shard_files: set[str] = set(profile.get("failed_shard_test_files", []))
    categories: set[str] = set(profile["categories"])
    hardware: set[str] = set(profile["hardware"])
    score = 0
    reasons: list[str] = []

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        if reason not in reasons:
            reasons.append(reason)

    exact = sorted(set(files) & failed_files)
    if exact:
        add(14, "changes failed test file(s): " + ", ".join(exact[:3]))

    shard_matches = sorted((set(files) & failed_shard_files) - set(exact))
    if shard_matches:
        add(5, "changes test file(s) from failed shard(s): " + ", ".join(shard_matches[:3]))

    matched_tokens = sorted(token for token in tokens if token in haystack)
    if matched_tokens:
        add(min(12, len(matched_tokens) * 2), "matches failure tokens: " + ", ".join(matched_tokens[:8]))

    if "build" in categories and any(
        needle in haystack
        for needle in ("setup.py", "pyproject", "cmake", "csrc", "jit", "aot", "composable_kernel", "build")
    ):
        add(8, "build-related change in failure window")

    if "standard" in categories and any(needle in haystack for needle in ("op_tests/", "tests/", "aiter/ops", "aiter/fused")):
        add(5, "touches standard-test surface")

    if "multigpu" in categories and any(
        needle in haystack
        for needle in (
            "mori",
            "allreduce",
            "custom_all_reduce",
            "custom_ar",
            "multigpu",
            "communication",
            "dpx",
            "runner",
            "aiter-test.yaml",
        )
    ):
        add(10, "touches multi-GPU/Mori/communication surface")

    if ("mi300x" in hardware or "gfx942" in hardware) and any(needle in haystack for needle in ("gfx942", "mi300", "gemm", "moe")):
        add(6, "matches MI300X/gfx942 failure scope")

    if ("mi35x" in hardware or "mi350x" in hardware or "gfx950" in hardware) and any(
        needle in haystack for needle in ("gfx950", "mi35", "mi350", "opus", "mha", "mori")
    ):
        add(6, "matches MI35X/MI350X/gfx950 failure scope")

    if any(needle in haystack for needle in ("moe", "mxfp4", "fused_moe", "mxmoe")):
        add(4, "MoE/MXFP4 related")
    if any(needle in haystack for needle in ("a8w8", "blockscale", "gemm")):
        add(3, "GEMM/a8w8 related")

    return score, reasons[:6]


def rank_candidates(commits: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for commit in commits:
        score, reasons = score_commit(commit, profile)
        if score <= 0:
            continue
        item = dict(commit)
        item["score"] = score
        item["reasons"] = reasons
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["score"], item["date"]))
    return ranked[:20]


def table_failed_jobs(jobs: list[dict[str, Any]]) -> str:
    rows = ["| Job | Type | Failed step(s) | Link |", "|---|---|---|---|"]
    for job in jobs:
        name = md_escape(job.get("name", ""))
        kind = classify_job(job)
        steps = md_escape(", ".join(failed_steps(job)) or "(none)")
        url = job.get("html_url") or ""
        rows.append(f"| {name} | {kind} | {steps} | [job]({url}) |")
    return "\n".join(rows)


def table_candidates(candidates: list[dict[str, Any]], repo: str) -> str:
    if not candidates:
        return "No commit candidates scored above zero from the available evidence."
    rows = ["| Score | Commit | Subject | Evidence |", "|---:|---|---|---|"]
    for item in candidates[:12]:
        url = f"https://github.com/{repo}/commit/{item['sha']}"
        reasons = md_escape("; ".join(item["reasons"]))
        rows.append(
            f"| {item['score']} | [{item['short']}]({url}) | "
            f"{md_escape(item['subject'])} | {reasons} |"
        )
    return "\n".join(rows)


def commit_summary(commit: dict[str, Any]) -> dict[str, Any]:
    return {
        "sha": commit["sha"],
        "short": commit["short"],
        "date": commit["date"],
        "author": commit["author"],
        "subject": commit["subject"],
        "files": commit.get("files", []),
        "score": commit.get("score"),
        "reasons": commit.get("reasons", []),
    }


def build_evidence_json(
    repo: str,
    target: dict[str, Any] | None,
    newer_runs: list[dict[str, Any]],
    previous_success: dict[str, Any] | None,
    failed_jobs: list[dict[str, Any]],
    root_failed_jobs: list[dict[str, Any]],
    snippets: list[dict[str, str]],
    shard_lists: dict[str, list[str]],
    commits: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
    download_messages: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repo": repo,
        "target_run": run_summary(repo, target),
        "newer_runs": [run_summary(repo, run) for run in newer_runs],
        "previous_success": run_summary(repo, previous_success),
        "failed_jobs": [job_summary(job) for job in failed_jobs],
        "root_failed_jobs": [job_summary(job) for job in root_failed_jobs],
        "failure_profile": profile,
        "snippets": snippets,
        "shard_lists": shard_lists,
        "commit_window": [commit_summary(commit) for commit in commits],
        "candidates": [commit_summary(commit) for commit in candidates],
        "download_messages": download_messages,
    }


def select_culprit(
    candidates: list[dict[str, Any]],
    commits: list[dict[str, Any]],
    previous_success: dict[str, Any] | None,
    profile: dict[str, Any],
    snippets: list[dict[str, str]],
    shard_lists: dict[str, list[str]],
    download_messages: list[str],
) -> dict[str, Any]:
    if not candidates:
        return {
            "schema_version": 1,
            "status": "no_candidate",
            "confidence": "none",
            "auto_revert_ready": False,
            "candidate": None,
            "reasons": ["no commit candidate scored above zero"],
            "blockers": [],
        }

    top = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0
    margin = top["score"] - second_score
    has_specific_evidence = bool(snippets or shard_lists or profile.get("failed_test_files"))
    changed_failed_test = any(reason.startswith("changes failed test file") for reason in top.get("reasons", []))
    blockers: list[str] = []
    warnings: list[str] = []
    reasons = list(top.get("reasons", []))

    if not previous_success:
        blockers.append("previous successful Aiter Test run was not found")
    if not has_specific_evidence:
        blockers.append("no shard list, failed test file, or log snippet was available")
    if download_messages:
        warnings.append("some logs or artifacts were unavailable")

    confidence = "low"
    if previous_success and has_specific_evidence:
        if len(commits) == 1:
            confidence = "high"
            reasons.append("the last-success to first-failure window contains a single commit")
        elif top["score"] >= 30 and margin >= 10:
            confidence = "high"
            reasons.append(f"top candidate leads the next candidate by {margin} points")
        elif changed_failed_test and top["score"] >= 24 and margin >= 5:
            confidence = "high"
            reasons.append("top candidate changes a failed test file and has a clear score margin")
        elif top["score"] >= 20 and margin >= 5:
            confidence = "medium"
            reasons.append(f"top candidate leads the next candidate by {margin} points")
        else:
            reasons.append("top candidate did not separate enough from the rest of the window")
    else:
        reasons.append("candidate ranking is report-only until failure-specific evidence is available")

    return {
        "schema_version": 1,
        "status": "selected" if confidence in {"high", "medium"} else "weak_candidate",
        "confidence": confidence,
        "auto_revert_ready": confidence == "high" and not blockers,
        "candidate": commit_summary(top),
        "second_score": second_score,
        "score_margin": margin,
        "commit_window_size": len(commits),
        "reasons": reasons,
        "blockers": blockers,
        "warnings": warnings,
    }


def render_culprit_summary(culprit: dict[str, Any], repo: str) -> str:
    candidate = culprit.get("candidate")
    lines = ["### Culprit Assessment", ""]
    lines.append(f"- Confidence: `{culprit.get('confidence', 'none')}`")
    lines.append(f"- Auto-revert ready: `{str(culprit.get('auto_revert_ready', False)).lower()}`")
    if candidate:
        url = f"https://github.com/{repo}/commit/{candidate['sha']}"
        lines.append(f"- Top candidate: [`{candidate['short']}`]({url}) {candidate['subject']}")
        lines.append(f"- Score: `{candidate.get('score')}`; margin: `{culprit.get('score_margin')}`")
    for reason in culprit.get("reasons", [])[:6]:
        lines.append(f"- Reason: {reason}")
    for blocker in culprit.get("blockers", [])[:6]:
        lines.append(f"- Blocker: {blocker}")
    for warning in culprit.get("warnings", [])[:6]:
        lines.append(f"- Warning: {warning}")
    return "\n".join(lines)


def render_evidence(
    repo: str,
    target: dict[str, Any] | None,
    newer_runs: list[dict[str, Any]],
    previous_success: dict[str, Any] | None,
    failed_jobs: list[dict[str, Any]],
    root_failed_jobs: list[dict[str, Any]],
    snippets: list[dict[str, str]],
    shard_lists: dict[str, list[str]],
    commits: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    download_messages: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Aiter Test failure evidence")
    lines.append("")
    lines.append(f"Repository: {repo}")
    if not target:
        lines.append("")
        if previous_success:
            lines.append("The latest completed Aiter Test run in the requested lookback succeeded.")
            lines.append(f"- Run: {run_id_of(previous_success)}")
            lines.append(f"- URL: {previous_success.get('url', '')}")
            lines.append(f"- Created: {previous_success.get('createdAt', '')}")
            lines.append(f"- Head SHA: {previous_success.get('headSha', '')}")
        else:
            lines.append("No completed failed Aiter Test run was found in the requested lookback.")
        if newer_runs:
            lines.append("- Newer runs not used:")
            for run in newer_runs[:8]:
                lines.append(
                    f"  - {run_id_of(run)} {run.get('createdAt', '')} "
                    f"{run.get('status', '')}/{run.get('conclusion', '')} {run.get('headSha', '')[:8]}"
                )
        return "\n".join(lines)
    lines.extend(
        [
            "",
            "## Target run",
            "",
            f"- Run: {run_id_of(target)}",
            f"- URL: {public_run_url(repo, target)}",
            f"- Created: {target.get('createdAt', '')}",
            f"- Event: {target.get('event', '')}",
            f"- Head SHA: {target.get('headSha', '')}",
            f"- Conclusion: {target.get('conclusion', '')}",
        ]
    )
    if newer_runs:
        lines.append("- Newer runs not used:")
        for run in newer_runs[:8]:
            lines.append(
                f"  - {run_id_of(run)} {run.get('createdAt', '')} "
                f"{run.get('status', '')}/{run.get('conclusion', '')} {run.get('headSha', '')[:8]}"
            )
    if previous_success:
        lines.extend(
            [
                "",
                "## Previous successful Aiter Test run",
                "",
                f"- Run: {run_id_of(previous_success)}",
                f"- URL: {previous_success.get('url', '')}",
                f"- Created: {previous_success.get('createdAt', '')}",
                f"- Head SHA: {previous_success.get('headSha', '')}",
            ]
        )
    else:
        lines.extend(["", "## Previous successful Aiter Test run", "", "Not found in the requested lookback."])
    lines.extend(["", "## Failed jobs", "", table_failed_jobs(failed_jobs)])
    lines.extend(["", "## Root failed jobs", "", table_failed_jobs(root_failed_jobs) if root_failed_jobs else "No non-aggregate failed job was found."])

    if shard_lists:
        lines.extend(["", "## Downloaded shard lists", ""])
        for shard, files in sorted(shard_lists.items(), key=lambda item: int(item[0])):
            lines.append(f"- Shard {shard}: {len(files)} file(s): {', '.join(files[:12])}{' ...' if len(files) > 12 else ''}")

    lines.extend(["", "## Failure snippets", ""])
    if snippets:
        for idx, snippet in enumerate(snippets[:12], start=1):
            label = f"{snippet['source']}:{snippet['line']}"
            test = f" ({snippet['test']})" if snippet.get("test") else ""
            lines.append(f"### Snippet {idx}: {label}{test}")
            lines.append("")
            lines.append("```text")
            lines.append(snippet["context"])
            lines.append("```")
            lines.append("")
    else:
        lines.append("No detailed failure snippets were extracted from logs or artifacts.")

    lines.extend(["", "## Commit window", ""])
    if commits:
        lines.append(f"{len(commits)} commit(s) between the previous success and the failed run.")
        for commit in commits[:40]:
            lines.append(f"- {commit['short']} {commit['date']} {commit['subject']}")
        if len(commits) > 40:
            lines.append(f"- ... {len(commits) - 40} more commit(s)")
    else:
        lines.append("No commit window was available.")

    lines.extend(["", "## Candidate commits", "", table_candidates(candidates, repo)])

    if download_messages:
        lines.extend(["", "## Log/artifact download issues", ""])
        for message in download_messages[:20]:
            lines.append(f"- {message}")
    return "\n".join(lines)


def render_fallback_report(
    evidence: str,
    target: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    culprit: dict[str, Any],
    repo: str,
) -> str:
    if not target:
        return (
            "## Aiter Test Failure Report\n\n"
            "No current completed `Aiter Test` failure was found in the requested lookback.\n\n"
            "<details><summary>Collected evidence</summary>\n\n"
            f"{evidence}\n\n"
            "</details>\n"
        )
    run_url = target.get("url", "")
    run_url = public_run_url(repo, target)
    lines = [
        "## Aiter Test Failure Report",
        "",
        f"Analyzed run [{run_id_of(target)}]({run_url}) at `{target.get('headSha', '')[:8]}`.",
        "",
        "This fallback report is generated from deterministic GitHub Actions metadata, downloaded logs/artifacts when available, and git history. The configured analysis agent can rewrite it with more narrative detail.",
        "",
    ]
    lines.extend([render_culprit_summary(culprit, repo), ""])
    candidate_table = table_candidates(candidates, repo)
    lines.extend(["### Candidate Commits", "", candidate_table, ""])
    lines.extend(["### Evidence", "", "<details><summary>Collected evidence</summary>", "", evidence, "", "</details>", ""])
    return "\n".join(lines)


def render_prompt(evidence: str) -> str:
    return (
        "You are aiter-bot analyzing ROCm/aiter CI. Produce a concise GitHub Actions "
        "step summary report in Markdown from the evidence below.\n\n"
        "Requirements:\n"
        "- Identify the latest completed failed Aiter Test run and whether newer runs are pending, queued, cancelled, or successful.\n"
        "- Separate root failures from aggregate failures such as Aiter Test Gate.\n"
        "- Explain the most likely failure reason from logs/artifacts. If logs are missing, say exactly what is missing.\n"
        "- Rank suspicious commits by evidence strength, not just recency. Include SHA, PR number from subject when present, and why it is suspicious.\n"
        "- Use the structured culprit assessment as the starting point. If it says low confidence, explain what evidence is missing.\n"
        "- Do not claim certainty when the evidence only supports a hypothesis.\n"
        "- Keep the report readable in a GitHub summary.\n\n"
        "<evidence>\n"
        f"{clip(evidence, 55000)}\n"
        "</evidence>\n"
    )


def choose_target_run(runs: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    target_index = None
    for idx, run in enumerate(runs):
        if run.get("status") != "completed":
            continue
        if run.get("conclusion") == "success":
            return None, runs[:idx], run
        if run.get("conclusion") == "failure":
            target_index = idx
            break
    if target_index is None:
        return None, [], None
    target = runs[target_index]
    newer = runs[:target_index]
    previous_success = None
    for run in runs[target_index + 1 :]:
        if run.get("status") == "completed" and run.get("conclusion") == "success":
            previous_success = run
            break
    return target, newer, previous_success


def find_previous_success(runs: list[dict[str, Any]], created_at: str) -> dict[str, Any] | None:
    return next(
        (
            run
            for run in runs
            if run.get("createdAt", "") < created_at
            and run.get("status") == "completed"
            and run.get("conclusion") == "success"
        ),
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="ROCm/aiter")
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    work_dir = Path(args.out)
    work_dir.mkdir(parents=True, exist_ok=True)

    download_messages: list[str] = []
    skip_downloads = os.environ.get("AITER_TEST_FAILURE_SKIP_DOWNLOADS", "").lower() in {"1", "true", "yes"}
    runs = list_runs(args.repo, args.lookback)
    if args.run_id:
        target = get_run(args.repo, args.run_id)
        if target.get("conclusion") != "failure":
            newer_runs = []
            previous_success = target if target.get("conclusion") == "success" else None
            target = None
        else:
            newer_runs = [run for run in runs if run.get("createdAt", "") > target.get("createdAt", "")]
            previous_success = find_previous_success(runs, target.get("createdAt", ""))
    else:
        target, newer_runs, previous_success = choose_target_run(runs)

    if target and not previous_success and args.lookback < 400:
        extended_runs = list_runs(args.repo, 400)
        previous_success = find_previous_success(extended_runs, target.get("createdAt", ""))

    failed_jobs: list[dict[str, Any]] = []
    root_failed_jobs: list[dict[str, Any]] = []
    snippets: list[dict[str, str]] = []
    shard_lists: dict[str, list[str]] = {}
    commits: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    profile: dict[str, Any] = {
        "tokens": [],
        "failed_test_files": [],
        "failed_shard_test_files": [],
        "hardware": [],
        "categories": [],
    }

    if target:
        run_id = run_id_of(target)
        jobs = get_jobs(args.repo, run_id)
        failed_jobs = [job for job in jobs if job.get("conclusion") == "failure"]
        root_failed_jobs = [job for job in failed_jobs if not is_aggregate_job(job.get("name", ""))]
        artifacts = get_artifacts(args.repo, run_id)
        if skip_downloads:
            download_messages.append("log and artifact downloads skipped by AITER_TEST_FAILURE_SKIP_DOWNLOADS")
        else:
            wanted_artifacts = artifact_names_for_failed_jobs(root_failed_jobs)
            download_messages.extend(download_artifacts(args.repo, run_id, work_dir / "artifacts", wanted_artifacts, artifacts))
            download_messages.extend(download_job_logs(args.repo, root_failed_jobs, work_dir / "job_logs"))
        snippets = collect_log_snippets(work_dir)
        shard_lists = read_shard_lists(work_dir)
        profile = build_failure_profile(root_failed_jobs, snippets, shard_lists)
        if previous_success:
            commits = commit_range(previous_success.get("headSha", ""), target.get("headSha", ""))
        else:
            commits = commit_range("", target.get("headSha", ""))
        candidates = rank_candidates(commits, profile)

    culprit = select_culprit(
        candidates,
        commits,
        previous_success,
        profile,
        snippets,
        shard_lists,
        download_messages,
    )

    evidence = render_evidence(
        args.repo,
        target,
        newer_runs,
        previous_success,
        failed_jobs,
        root_failed_jobs,
        snippets,
        shard_lists,
        commits,
        candidates,
        download_messages,
    )
    evidence_json = build_evidence_json(
        args.repo,
        target,
        newer_runs,
        previous_success,
        failed_jobs,
        root_failed_jobs,
        snippets,
        shard_lists,
        commits,
        candidates,
        profile,
        download_messages,
    )
    (work_dir / "evidence.md").write_text(evidence + "\n", encoding="utf-8")
    (work_dir / "evidence.json").write_text(json.dumps(evidence_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (work_dir / "culprit.json").write_text(json.dumps(culprit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (work_dir / "fallback_report.md").write_text(
        render_fallback_report(evidence, target, candidates, culprit, args.repo),
        encoding="utf-8",
    )
    if target:
        structured = (
            "# Structured culprit assessment\n\n"
            "```json\n"
            f"{json.dumps(culprit, indent=2, sort_keys=True)}\n"
            "```\n\n"
            f"{evidence}"
        )
        (work_dir / "prompt.md").write_text(render_prompt(structured), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

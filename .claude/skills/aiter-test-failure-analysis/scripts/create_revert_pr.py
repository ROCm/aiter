#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
DEFAULT_ASSIGNEE = "gyohuangxin"
REVERT_MARKER = "aiter-auto-revert"
LOW_CONFIDENCE_MARKER = "aiter-low-confidence-culprit"
FAST_RUNNERS = (
    "linux-aiter-mi35x-1",
    "linux-aiter-oci-mi300x-1",
    "linux-aiter-do-mi350x-8",
    "linux-aiter-do-mi350x-dpx-8",
)


class CommandError(RuntimeError):
    def __init__(self, args: list[str], rc: int, out: str, err: str) -> None:
        super().__init__(f"{' '.join(args)} failed with exit code {rc}: {err.strip() or out.strip()}")
        self.args_list = args
        self.rc = rc
        self.out = out
        self.err = err


def run_cmd(
    args: list[str],
    *,
    check: bool = True,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise CommandError(args, proc.returncode, proc.stdout, proc.stderr)
    return proc


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_path() -> Path | None:
    value = os.environ.get("GITHUB_STEP_SUMMARY", "")
    return Path(value) if value else None


def append_summary(markdown: str) -> None:
    path = summary_path()
    if path:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(markdown.rstrip() + "\n")
    print(markdown.rstrip())


def gh_env() -> dict[str, str]:
    env = dict(os.environ)
    token = env.get("AITER_BOT_TOKEN") or env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")
    if token:
        env["GH_TOKEN"] = token
    return env


def gh_json(args: list[str], *, timeout: int = 120) -> Any:
    proc = run_cmd(args, timeout=timeout, env=gh_env())
    return json.loads(proc.stdout or "null")


def md_escape(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def candidate_from_culprit(culprit: dict[str, Any]) -> dict[str, Any] | None:
    candidate = culprit.get("candidate")
    return candidate if isinstance(candidate, dict) else None


def commit_pr_number(candidate: dict[str, Any]) -> str:
    match = re.search(r"\(#([0-9]+)\)\s*$", str(candidate.get("subject") or ""))
    return match.group(1) if match else ""


def target_run_id(evidence: dict[str, Any]) -> str:
    target = evidence.get("target_run") or {}
    return str(target.get("id") or os.environ.get("GITHUB_RUN_ID") or "unknown")


def target_run_url(evidence: dict[str, Any]) -> str:
    target = evidence.get("target_run") or {}
    return str(target.get("url") or "")


def current_workflow_run_url(repo: str) -> str:
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    return f"https://github.com/{repo}/actions/runs/{run_id}" if run_id else ""


def evidence_profile(evidence: dict[str, Any]) -> dict[str, Any]:
    profile = evidence.get("failure_profile")
    return profile if isinstance(profile, dict) else {}


def failed_job_names(evidence: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for section in ("root_failed_jobs", "failed_jobs"):
        for job in evidence.get(section) or []:
            name = str(job.get("name") or "")
            if name:
                names.append(name)
    return names


def fast_runner_from_evidence(evidence: dict[str, Any]) -> str:
    for name in failed_job_names(evidence):
        for runner in FAST_RUNNERS:
            if runner in name:
                return runner
        match = re.search(r"\((linux-[^,\)]+)", name)
        if match and match.group(1) in FAST_RUNNERS:
            return match.group(1)

    profile = evidence_profile(evidence)
    hardware = {str(item).lower() for item in profile.get("hardware", [])}
    categories = {str(item).lower() for item in profile.get("categories", [])}
    tokens = {str(item).lower() for item in profile.get("tokens", [])}

    if hardware & {"mi300x", "gfx942"}:
        return "linux-aiter-oci-mi300x-1"
    if "multigpu" in categories or "mori" in tokens:
        return "linux-aiter-do-mi350x-8"
    if hardware & {"mi350x", "mi35x", "gfx950"}:
        return "linux-aiter-mi35x-1"
    return "linux-aiter-mi35x-1"


def fast_test_files_from_evidence(evidence: dict[str, Any]) -> list[str]:
    profile = evidence_profile(evidence)
    files = []
    candidates = list(profile.get("failed_test_files", [])) or list(
        profile.get("failed_shard_test_files", [])
    )
    for value in candidates:
        path = str(value).strip()
        if path.startswith(("op_tests/", "tests/")) and path not in files:
            files.append(path)
    return sorted(files)


def fast_mori_smoke_from_evidence(evidence: dict[str, Any]) -> bool:
    profile = evidence_profile(evidence)
    haystack = " ".join(
        [*failed_job_names(evidence)]
        + [str(item) for item in profile.get("tokens", [])]
        + [str(item) for item in profile.get("categories", [])]
    ).lower()
    return "mori" in haystack and ("multigpu" in haystack or "smoke" in haystack)


def commit_subject(sha: str) -> str:
    proc = run_cmd(["git", "show", "-s", "--format=%s", sha], timeout=60)
    return proc.stdout.strip()


def is_merge_commit(sha: str) -> bool:
    proc = run_cmd(["git", "rev-list", "--parents", "-n", "1", sha], timeout=60)
    return len(proc.stdout.strip().split()) > 2


def ensure_origin_main() -> None:
    run_cmd(["git", "fetch", "--no-tags", "--depth=1", "origin", "main"], timeout=180)


def has_commit(sha: str) -> bool:
    proc = run_cmd(["git", "cat-file", "-e", f"{sha}^{{commit}}"], check=False, timeout=60)
    return proc.returncode == 0


def fetch_commit(sha: str) -> bool:
    if has_commit(sha):
        return True
    run_cmd(["git", "fetch", "--no-tags", "--depth=2", "origin", sha], check=False, timeout=180)
    if has_commit(sha):
        return True
    run_cmd(["git", "fetch", "--no-tags", "origin", sha], check=False, timeout=180)
    return has_commit(sha)


def remote_branch_exists(branch: str) -> bool:
    proc = run_cmd(["git", "ls-remote", "--heads", "origin", branch], check=False, timeout=60)
    return bool(proc.stdout.strip())


def open_revert_pr_for_sha(repo: str, sha: str) -> dict[str, Any] | None:
    short = sha[:12]
    pulls = gh_json(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,body,headRefName,url",
        ],
        timeout=120,
    )
    for pull in pulls or []:
        haystack = "\n".join(
            str(pull.get(key) or "") for key in ("title", "body", "headRefName")
        )
        if sha in haystack or short in haystack:
            return pull
    return None


def open_low_confidence_issue_for_sha(repo: str, sha: str) -> dict[str, Any] | None:
    short = sha[:12]
    issues = gh_json(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,body,url",
        ],
        timeout=120,
    )
    for issue in issues or []:
        haystack = "\n".join(str(issue.get(key) or "") for key in ("title", "body"))
        if LOW_CONFIDENCE_MARKER in haystack and (sha in haystack or short in haystack):
            return issue
    return None


def git_dirty() -> bool:
    proc = run_cmd(["git", "status", "--porcelain"], timeout=60)
    return bool(proc.stdout.strip())


def make_branch(branch: str) -> None:
    run_cmd(["git", "checkout", "-B", branch, "origin/main"], timeout=120)


def revert_commit(sha: str) -> subprocess.CompletedProcess[str]:
    return run_cmd(["git", "revert", "--no-edit", sha], check=False, timeout=180)


def abort_revert_if_needed() -> None:
    git_dir = run_cmd(["git", "rev-parse", "--git-dir"], timeout=60).stdout.strip()
    if (ROOT / git_dir / "REVERT_HEAD").exists():
        run_cmd(["git", "revert", "--abort"], check=False, timeout=60)


def push_branch(branch: str) -> None:
    run_cmd(["gh", "auth", "setup-git"], env=gh_env(), timeout=60)
    run_cmd(["git", "push", "--set-upstream", "origin", branch], env=gh_env(), timeout=180)


def commit_github_author(repo: str, sha: str) -> str:
    try:
        commit = gh_json(["gh", "api", f"repos/{repo}/commits/{sha}"], timeout=120)
    except (CommandError, json.JSONDecodeError):
        return ""
    for key in ("author", "committer"):
        value = commit.get(key) if isinstance(commit, dict) else None
        login = str((value or {}).get("login") or "").strip()
        if login:
            return login
    return ""


def pull_request_author(repo: str, candidate: dict[str, Any]) -> str:
    pr_number = commit_pr_number(candidate)
    if not pr_number:
        return ""
    try:
        pull = gh_json(["gh", "api", f"repos/{repo}/pulls/{pr_number}"], timeout=120)
    except (CommandError, json.JSONDecodeError):
        return ""
    user = pull.get("user") if isinstance(pull, dict) else None
    return str((user or {}).get("login") or "").strip()


def responsible_login(repo: str, candidate: dict[str, Any], fallback: str) -> str:
    sha = str(candidate.get("sha") or "")
    return commit_github_author(repo, sha) or pull_request_author(repo, candidate) or fallback


def assign_issue(repo: str, issue_url: str, login: str, fallback: str) -> str:
    if not login:
        login = fallback
    issue_number = issue_url.rstrip("/").rsplit("/", 1)[-1]
    for assignee in (login, fallback):
        if not assignee:
            continue
        proc = run_cmd(
            [
                "gh",
                "issue",
                "edit",
                issue_number,
                "--repo",
                repo,
                "--add-assignee",
                assignee,
            ],
            check=False,
            env=gh_env(),
            timeout=120,
        )
        if proc.returncode == 0:
            return assignee
    return ""


def limited_items(values: object, limit: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values[:limit]]


def create_low_confidence_issue(
    repo: str,
    candidate: dict[str, Any],
    culprit: dict[str, Any],
    evidence: dict[str, Any],
    fallback_assignee: str,
) -> tuple[str, str]:
    sha = str(candidate.get("sha") or "")
    short = str(candidate.get("short") or sha[:12])
    subject = str(candidate.get("subject") or commit_subject(sha) if sha else "unknown subject")
    run_id = target_run_id(evidence)
    run_url = target_run_url(evidence)
    profile = evidence_profile(evidence)
    assignee = responsible_login(repo, candidate, fallback_assignee)
    reasons = limited_items(culprit.get("reasons"), limit=8)
    blockers = limited_items(culprit.get("blockers"), limit=8)
    warnings = limited_items(culprit.get("warnings"), limit=8)
    failed_jobs = failed_job_names(evidence)[:8]
    failed_tests = limited_items(profile.get("failed_test_files"), limit=12)
    failed_shard_tests = limited_items(profile.get("failed_shard_test_files"), limit=12)
    hardware = limited_items(profile.get("hardware"), limit=8)
    categories = limited_items(profile.get("categories"), limit=8)

    body_lines = [
        f"<!-- {LOW_CONFIDENCE_MARKER}: culprit={sha} -->",
        "",
        "The Aiter Test failure report selected this culprit candidate with low confidence, "
        "so the bot did not create an automatic revert PR. Please investigate whether this "
        "commit is related to the failing Aiter Test run.",
        "",
        "## Candidate",
        "",
        f"- Commit: `{sha}`",
        f"- Subject: {subject}",
        f"- Confidence: `{culprit.get('confidence', '')}`",
        f"- Auto-revert ready: `{str(culprit.get('auto_revert_ready', False)).lower()}`",
        f"- Score: `{candidate.get('score', '-')}`; margin: `{culprit.get('score_margin', '-')}`",
        f"- Source Aiter Test run: `{run_id}`",
    ]
    if run_url:
        body_lines.append(f"- Run URL: {run_url}")
    report_url = current_workflow_run_url(repo)
    if report_url:
        body_lines.append(f"- Failure report run: {report_url}")
    if commit_pr_number(candidate):
        body_lines.append(f"- Candidate PR: #{commit_pr_number(candidate)}")
    if assignee:
        body_lines.append(f"- Suggested owner: @{assignee}")

    body_lines.extend(["", "## Failure Scope", ""])
    if failed_jobs:
        body_lines.append("- Failed jobs: " + ", ".join(f"`{item}`" for item in failed_jobs))
    if failed_tests:
        body_lines.append("- Failed tests: " + ", ".join(f"`{item}`" for item in failed_tests))
    if failed_shard_tests:
        body_lines.append(
            "- Test files in failed shard(s), not necessarily failed: "
            + ", ".join(f"`{item}`" for item in failed_shard_tests)
        )
    if hardware:
        body_lines.append("- Hardware: " + ", ".join(f"`{item}`" for item in hardware))
    if categories:
        body_lines.append("- Categories: " + ", ".join(f"`{item}`" for item in categories))
    if not any((failed_jobs, failed_tests, failed_shard_tests, hardware, categories)):
        body_lines.append("- No structured failure scope was available.")

    body_lines.extend(["", "## Report Evidence", ""])
    for reason in reasons:
        body_lines.append(f"- Reason: {reason}")
    for blocker in blockers:
        body_lines.append(f"- Blocker: {blocker}")
    for warning in warnings:
        body_lines.append(f"- Warning: {warning}")
    if not any((reasons, blockers, warnings)):
        body_lines.append("- No additional evidence was recorded.")

    title = f"[Aiter Test] Low-confidence culprit candidate {short}: {subject}"
    proc = run_cmd(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            title[:250],
            "--body",
            "\n".join(body_lines),
        ],
        env=gh_env(),
        timeout=180,
    )
    issue_url = proc.stdout.strip()
    assigned = assign_issue(repo, issue_url, assignee, fallback_assignee)
    return issue_url, assigned


def create_pr(
    repo: str,
    branch: str,
    sha: str,
    run_id: str,
    run_url: str,
    assignee: str,
    evidence: dict[str, Any],
) -> str:
    subject = commit_subject(sha)
    title = f"[bot revert] Revert {sha[:12]}: {subject}"
    fast_runner = fast_runner_from_evidence(evidence)
    fast_test_files = fast_test_files_from_evidence(evidence)
    fast_mori_smoke = fast_mori_smoke_from_evidence(evidence)
    body_lines = [
        "<!-- aiter-revert-fast",
        f"failed_run_id: {run_id}",
        f"runner: {fast_runner}",
        f"test_files: {' '.join(fast_test_files)}",
        f"mori_smoke: {str(fast_mori_smoke).lower()}",
        "-->",
        "",
        f"<!-- {REVERT_MARKER}: culprit={sha} -->",
        "",
        "Auto-created from the structured Aiter Test failure culprit report.",
        "",
        f"- Culprit SHA: `{sha}`",
        f"- Source Aiter Test run: `{run_id}`",
    ]
    if run_url:
        body_lines.append(f"- Run URL: {run_url}")
    body_lines.extend(
        [
            "",
            "This PR was only created because `culprit.json` had `auto_revert_ready: true` and `confidence: high`.",
        ]
    )
    proc = run_cmd(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repo,
            "--base",
            "main",
            "--head",
            branch,
            "--title",
            title,
            "--body",
            "\n".join(body_lines),
            "--assignee",
            assignee,
        ],
        env=gh_env(),
        timeout=180,
    )
    return proc.stdout.strip()


def render_skip(reason: str, culprit: dict[str, Any], sha: str = "") -> str:
    rows = [
        "## Aiter Auto-Revert",
        "",
        f"Skipped: {reason}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| auto_revert_ready | `{str(culprit.get('auto_revert_ready', False)).lower()}` |",
        f"| confidence | `{md_escape(culprit.get('confidence', ''))}` |",
    ]
    if sha:
        rows.append(f"| culprit | `{sha}` |")
    candidate = candidate_from_culprit(culprit)
    if candidate:
        rows.append(f"| subject | {md_escape(candidate.get('subject', ''))} |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "ROCm/aiter"))
    parser.add_argument("--work-dir", default=os.environ.get("AITER_TEST_FAILURE_WORKDIR", ""))
    parser.add_argument("--assignee", default=DEFAULT_ASSIGNEE)
    parser.add_argument("--dry-run", action="store_true", help="validate gates without pushing or creating a PR")
    args = parser.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else ROOT
    culprit_path = work_dir / "culprit.json"
    evidence_path = work_dir / "evidence.json"
    if not culprit_path.exists() or not evidence_path.exists():
        append_summary(
            "## Aiter Auto-Revert\n\n"
            f"Skipped: missing structured report files in `{work_dir}`."
        )
        return 0

    culprit = load_json(culprit_path)
    evidence = load_json(evidence_path)
    candidate = candidate_from_culprit(culprit)
    sha = str((candidate or {}).get("sha") or "")
    if not candidate or not sha:
        append_summary(render_skip("no culprit candidate was selected", culprit))
        return 0
    if culprit.get("confidence") == "low":
        if args.dry_run:
            append_summary(
                "## Aiter Low-Confidence Culprit\n\n"
                f"Dry run passed. The script would create an investigation issue for `{sha}` "
                "instead of opening an automatic revert PR."
            )
            return 0
        duplicate_pr = open_revert_pr_for_sha(args.repo, sha)
        if duplicate_pr:
            append_summary(
                "## Aiter Low-Confidence Culprit\n\n"
                f"Skipped: duplicate open revert PR already exists for `{sha}`.\n\n"
                f"- PR: #{duplicate_pr.get('number')} {duplicate_pr.get('url')}"
            )
            return 0
        duplicate_issue = open_low_confidence_issue_for_sha(args.repo, sha)
        if duplicate_issue:
            append_summary(
                "## Aiter Low-Confidence Culprit\n\n"
                f"Skipped: duplicate open investigation issue already exists for `{sha}`.\n\n"
                f"- Issue: #{duplicate_issue.get('number')} {duplicate_issue.get('url')}"
            )
            return 0
        issue_url, assigned = create_low_confidence_issue(args.repo, candidate, culprit, evidence, args.assignee)
        assignment_line = f"- Assignee: @{assigned}" if assigned else "- Assignee: (assignment failed)"
        append_summary(
            "## Aiter Low-Confidence Culprit\n\n"
            f"Created investigation issue for `{sha}` because the culprit report confidence is low.\n\n"
            f"- Issue: {issue_url}\n"
            f"{assignment_line}"
        )
        return 0
    if culprit.get("auto_revert_ready") is not True or culprit.get("confidence") != "high":
        append_summary(render_skip("culprit report is not high-confidence auto-revert ready", culprit, sha))
        return 0

    duplicate = None if args.dry_run else open_revert_pr_for_sha(args.repo, sha)
    if duplicate:
        append_summary(
            "## Aiter Auto-Revert\n\n"
            f"Skipped: duplicate open revert PR already exists for `{sha}`.\n\n"
            f"- PR: #{duplicate.get('number')} {duplicate.get('url')}"
        )
        return 0

    run_id = target_run_id(evidence)
    branch = f"aiter-bot/revert-{sha[:12]}-run-{run_id}"

    if args.dry_run and has_commit(sha):
        if is_merge_commit(sha):
            append_summary(render_skip("merge commits are not auto-reverted yet", culprit, sha))
            return 0
        append_summary(
            "## Aiter Auto-Revert\n\n"
            "Dry run passed. The report is high-confidence auto-revert ready, "
            f"`{sha}` is not a merge commit, and the script would create branch `{branch}`."
        )
        return 0

    ensure_origin_main()
    if not fetch_commit(sha):
        append_summary(render_skip("culprit commit could not be fetched on the revert runner", culprit, sha))
        return 0
    if is_merge_commit(sha):
        append_summary(render_skip("merge commits are not auto-reverted yet", culprit, sha))
        return 0

    if args.dry_run:
        append_summary(
            "## Aiter Auto-Revert\n\n"
            "Dry run passed. The report is high-confidence auto-revert ready, "
            f"`{sha}` is not a merge commit, and the script would create branch `{branch}`."
        )
        return 0

    if remote_branch_exists(branch):
        append_summary(
            "## Aiter Auto-Revert\n\n"
            f"Skipped: remote branch `{branch}` already exists, but no matching open PR was found. "
            "Leaving it for manual cleanup to avoid overwriting a previous attempt."
        )
        return 0
    if git_dirty():
        append_summary(render_skip("working tree is dirty before revert", culprit, sha))
        return 0

    make_branch(branch)
    revert_proc = revert_commit(sha)
    if revert_proc.returncode != 0:
        abort_revert_if_needed()
        append_summary(
            "## Aiter Auto-Revert\n\n"
            f"Skipped: `git revert --no-edit {sha}` had conflicts or failed. No branch was pushed.\n\n"
            "```text\n"
            f"{(revert_proc.stderr or revert_proc.stdout).strip()[:3000]}\n"
            "```"
        )
        return 0

    push_branch(branch)
    pr_url = create_pr(args.repo, branch, sha, run_id, target_run_url(evidence), args.assignee, evidence)
    append_summary(
        "## Aiter Auto-Revert\n\n"
        f"Created revert PR for `{sha}`.\n\n"
        f"- Branch: `{branch}`\n"
        f"- PR: {pr_url}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CommandError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

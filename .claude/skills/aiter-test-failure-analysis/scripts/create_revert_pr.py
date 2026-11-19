#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
DEFAULT_ASSIGNEE = "gyohuangxin"
REVERT_MARKER = "aiter-auto-revert"


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


def target_run_id(evidence: dict[str, Any]) -> str:
    target = evidence.get("target_run") or {}
    return str(target.get("id") or os.environ.get("GITHUB_RUN_ID") or "unknown")


def target_run_url(evidence: dict[str, Any]) -> str:
    target = evidence.get("target_run") or {}
    return str(target.get("url") or "")


def commit_subject(sha: str) -> str:
    proc = run_cmd(["git", "show", "-s", "--format=%s", sha], timeout=60)
    return proc.stdout.strip()


def is_merge_commit(sha: str) -> bool:
    proc = run_cmd(["git", "rev-list", "--parents", "-n", "1", sha], timeout=60)
    return len(proc.stdout.strip().split()) > 2


def ensure_origin_main() -> None:
    run_cmd(["git", "fetch", "origin", "main"], timeout=180)


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


def create_pr(repo: str, branch: str, sha: str, run_id: str, run_url: str, assignee: str) -> str:
    subject = commit_subject(sha)
    title = f"Revert {sha[:12]}: {subject}"
    body_lines = [
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
    if culprit.get("auto_revert_ready") is not True or culprit.get("confidence") != "high":
        append_summary(render_skip("culprit report is not high-confidence auto-revert ready", culprit, sha))
        return 0

    if is_merge_commit(sha):
        append_summary(render_skip("merge commits are not auto-reverted yet", culprit, sha))
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
    if args.dry_run:
        append_summary(
            "## Aiter Auto-Revert\n\n"
            "Dry run passed. The report is high-confidence auto-revert ready, "
            f"`{sha}` is not a merge commit, and the script would create branch `{branch}`."
        )
        return 0

    ensure_origin_main()
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
    pr_url = create_pr(args.repo, branch, sha, run_id, target_run_url(evidence), args.assignee)
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

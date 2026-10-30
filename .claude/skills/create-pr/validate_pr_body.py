#!/usr/bin/env python3
"""Check a PR description with aiter-bot's own validator before posting it.

Loads .github/scripts/pr_self_review.py from origin/main (the version the bot runs) and calls
validate_author_answers(body, head_sha, changed_op_tests). Prints OK or the exact issues the bot
would report, plus the PR-specific question selected for the changed paths.

Usage:
  validate_pr_body.py BODY.md [--pr N] [--head SHA] [--repo ROCm/aiter] [--ref origin/main]
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("body")
    ap.add_argument(
        "--pr", type=int, help="read head SHA and changed files from GitHub"
    )
    ap.add_argument("--head", help="head SHA (default: PR head, else local HEAD)")
    ap.add_argument("--repo", default="ROCm/aiter")
    ap.add_argument("--ref", default="origin/main", help="where to read the bot script")
    args = ap.parse_args()

    source = run("git", "show", f"{args.ref}:.github/scripts/pr_self_review.py")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pr_self_review.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("pr_self_review", path)
        bot = importlib.util.module_from_spec(spec)
        # Its dataclasses resolve their module through sys.modules.
        sys.modules[spec.name] = bot
        spec.loader.exec_module(bot)

    if args.pr:
        info = json.loads(
            run(
                "gh",
                "pr",
                "view",
                str(args.pr),
                "--repo",
                args.repo,
                "--json",
                "headRefOid,files",
            )
        )
        head = args.head or info["headRefOid"]
        paths = [f["path"] for f in info["files"]]
    else:
        head = args.head or run("git", "rev-parse", "HEAD").strip()
        paths = run("git", "diff", "--name-only", f"{args.ref}...HEAD").split()

    changed_tests = [p for p in paths if p.startswith("op_tests/")]
    body = Path(args.body).read_text()
    issues, answers = bot.validate_author_answers(body, head, changed_tests)

    print(f"head: {head}")
    print(f"PR-specific question: {bot.select_question(paths)}")
    for name, text in answers.items():
        print(f"  {name}: {bot.word_count(text)} units")
    if issues:
        print("ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

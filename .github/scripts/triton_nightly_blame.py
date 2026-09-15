#!/usr/bin/env python
"""Rank the commits most likely to have broken a failing nightly test.

Reuses the import graph from select_triton_tests.py: a commit is a suspect
when it touched a file the failing test can reach (its wrapper, kernel body,
config or the test itself). Pure git + AST, no GPU.
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def load_selector():
    spec = importlib.util.spec_from_file_location(
        "sel", HERE / "select_triton_tests.py"
    )
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    sel.ROOT = ROOT
    return sel


def git(*args):
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def commits_since(since, until="HEAD"):
    """First-parent commits, newest first — on a PR-merge history these are
    the merged PRs."""
    sep = "\x01"  # not a splitlines() boundary, unlike \x1e
    fmt = sep.join(["%H", "%h", "%an", "%ae", "%ad", "%s"])
    raw = git("log", "--first-parent", "--date=short", f"--format={fmt}", f"{since}..{until}")
    out = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        sha, short, author, email, date, subject = line.split(sep)
        pr = re.search(r"#(\d+)", subject)
        out.append(
            {
                "sha": sha,
                "short": short,
                "author": author,
                "login": email.split("@")[0].split("+")[-1]
                if "users.noreply.github.com" in email
                else "",
                "date": date,
                "subject": subject,
                "pr": pr.group(1) if pr else "",
                "files": git("show", "--name-only", "--format=", sha).splitlines(),
            }
        )
    return out


def suspects(failing_tests, since):
    sel = load_selector()
    tests = sel.list_files(sel.TESTS, "test_*.py")
    sources = sel.list_files(sel.SRC, "*.py")
    imports = {f: sel.scan_imports(f) for f in sources + tests}

    # Everything each failing test depends on, plus the test file itself.
    blast = {}
    for t in failing_tests:
        blast[t] = sel.reachable(t, imports) | {t} if t in imports else {t}

    ranked = []
    for c in commits_since(since):
        hits = {}
        for t, deps in blast.items():
            overlap = sorted(set(c["files"]) & deps)
            if overlap:
                hits[t] = overlap
        if hits:
            ranked.append({**c, "hits": hits, "score": len(hits)})
    ranked.sort(key=lambda c: (-c["score"], c["date"]))
    return ranked


def markdown(ranked, failing_tests, since, limit=5):
    lines = []
    if not ranked:
        lines.append(
            f"No commit since `{since[:9]}` touched anything these tests import — "
            "look at the runner, the Triton pin or the image rather than aiter."
        )
        return "\n".join(lines)
    lines.append(f"Suspect commits since `{since[:9]}` (ranked by overlap with the failing tests' dependencies):\n")
    lines.append("| commit | PR | author | touched |")
    lines.append("|---|---|---|---|")
    for c in ranked[:limit]:
        touched = sorted({f for fs in c["hits"].values() for f in fs})
        shown = ", ".join(f"`{f}`" for f in touched[:3]) + (" …" if len(touched) > 3 else "")
        pr = f"#{c['pr']}" if c["pr"] else "—"
        who = f"@{c['login']}" if c["login"] else c["author"]
        lines.append(f"| `{c['short']}` | {pr} | {who} | {shown} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", required=True, help="last known-good SHA")
    ap.add_argument("--tests", required=True, help="file with failing test paths, one per line")
    ap.add_argument("--json-out", help="also write the ranked suspects as JSON")
    args = ap.parse_args()

    failing = [l.strip() for l in Path(args.tests).read_text().splitlines() if l.strip()]
    if not failing:
        print("no failing tests given")
        return
    try:
        ranked = suspects(failing, args.since)
    except Exception as exc:  # noqa: BLE001 -- triage must never break the report
        print(f"suspect ranking failed: {exc}")
        return
    print(markdown(ranked, failing, args.since))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([{k: v for k, v in c.items() if k != "files"} for c in ranked], indent=1)
        )


if __name__ == "__main__":
    main()

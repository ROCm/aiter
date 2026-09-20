"""Shared logic for the orchestration layer.

In python rather than shell, for reliability: shell grep/wc/ls can be hijacked by a
function or alias (grep here is wrapped by Claude Code as a function that silently returns
bad output when its underlying binary is unavailable), and a check that silently returns
an empty value is more dangerous than no check — it makes everything look like it passed.

This does only ops: whether an artifact exists, whether the format is valid, whether a gate
passed, what is missing. **It does not judge whether a finding is correct, nor touch
severity** — that is the review-pr skill's job.
"""

import json
import os
import pathlib
import re
import subprocess
import sys

FINDING_RE = re.compile(r"^(?:🔴|⚠️|📝)\s")
FIRE_RE = re.compile(r"^([A-Z0-9]+)\s+FIRE\s")

# What a report is made of. A missing required file is a collection failure; a missing
# optional file is just noted.
REQUIRED = [
    "card.md",
    "verdicts.txt",
    "answers.txt",
    "ai_diagnostic.txt",
    "core_files.txt",
    "refutations.txt",
    "independent.txt",
    "rules.txt",
    "pr_meta.json",
]
OPTIONAL = ["late_findings.txt"]


def read(p):
    try:
        return pathlib.Path(p).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def nonempty(p):
    try:
        return pathlib.Path(p).stat().st_size > 0
    except OSError:
        return False


def count_lines(p):
    t = read(p)
    return len([l for l in t.split("\n") if l.strip()]) if t else 0


def fire_ids(work):
    """Rule ids ruled FIRE in verdicts.txt and not marked 'not reported'."""
    out = []
    for l in read(os.path.join(work, "verdicts.txt")).split("\n"):
        m = FIRE_RE.match(l)
        if m and "-- not reported:" not in l:
            out.append(m.group(1))
    return out


def card_findings(work):
    return [
        l
        for l in read(os.path.join(work, "card.md")).split("\n")
        if FINDING_RE.match(l)
    ]


def pr_meta(work):
    try:
        d = json.loads(read(os.path.join(work, "pr_meta.json")) or "{}")
    except json.JSONDecodeError:
        return None, None
    num = d.get("number") or d.get("pr")
    title = (d.get("title") or "").replace("\n", " ").strip()
    return num, title


def run_gates(work, proj):
    """Run the seven gates, return [(name, exit_code, first_line, full)]. Trust no agent's self-report."""
    sk = os.path.join(proj, ".claude", "skills", "review-pr", "triage.py")
    W = lambda f: os.path.join(work, f)
    # A missing artifact should let the gate rule "this step was not done", not crash the script
    for f in [
        "answers.txt",
        "ai_diagnostic.txt",
        "core_files.txt",
        "verdicts.txt",
        "refutations.txt",
        "independent.txt",
        "late_findings.txt",
    ]:
        pathlib.Path(W(f)).touch(exist_ok=True)
    pathlib.Path(W("card.md")).touch(exist_ok=True)
    specs = [
        ("answers", ["answers", W("answers.txt")]),
        ("diagnostic", ["diagnostic", W("ai_diagnostic.txt")]),
        ("corefiles", ["corefiles", W("core_files.txt"), W("pr.diff"), proj]),
        ("ledger", ["ledger", W("rules.txt"), W("verdicts.txt"), W("pr.diff")]),
        (
            "card",
            [
                "card",
                W("card.md"),
                W("verdicts.txt"),
                W("ai_diagnostic.txt"),
                W("answers.txt"),
                W("pr.diff"),
                W("late_findings.txt"),
            ],
        ),
        (
            "refutations",
            ["refutations", W("refutations.txt"), W("pr.diff"), W("card.md")],
        ),
        ("independent", ["independent", W("independent.txt"), W("card.md")]),
    ]
    env = dict(os.environ, PYTHONUTF8="1")
    res = []
    for name, args in specs:
        p = subprocess.run(
            [sys.executable, sk] + args,
            cwd=proj,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        out = (p.stdout + p.stderr).strip().split("\n")
        res.append(
            (
                name,
                p.returncode,
                out[0] if out and out[0] else "(no output)",
                "\n".join(out[:10]),
            )
        )
    return res


def pr_state(pr, repo="ROCm/aiter"):
    """Whether the PR is currently open / merged / closed.

    The queue often comes from a stale review list: of 11 reviewed on 2026-09-15, 2 had
    already merged the same day, one of them carrying three reds confirmed by GPU runs. If a
    report does not state the status, the reader assumes it can still be stopped before merge
    — but those issues are already on main, changing them from "fix before merge" to
    "follow-up fixes".
    """
    import json
    import os
    import subprocess

    tok = ""
    try:
        with open(os.path.expanduser("~/.git-credentials"), encoding="utf-8") as fh:
            for line in fh:
                if "github.com" in line and ":" in line:
                    tok = line.split("://", 1)[1].split("@")[0].split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    cmd = ["curl", "-s", "-m", "15"]
    if tok:
        cmd += ["-H", f"Authorization: token {tok}"]
    cmd.append(f"https://api.github.com/repos/{repo}/pulls/{pr}")
    try:
        d = json.loads(
            subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
            or "{}"
        )
    except json.JSONDecodeError:
        return None
    if "state" not in d:
        return None
    if d.get("merged_at"):
        return "merged"
    return d["state"]

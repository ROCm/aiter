"""Guard against prompt / SKILL.md drift.

prompts/worker.md and prompts/refuter.md quote SKILL.md verbatim (the Step 8 Output rules and
the Step 7.7 text). If SKILL.md changes and the copies do not, they drift silently and the
agents start working from a stale standard. This check extracts each quoted block from SKILL.md
(between unique start/end anchors) and asserts it appears verbatim in the prompt. Any drift, or
a moved anchor, exits non-zero. Run it in preflight and at the start of run_one.sh.

Layout-independent: the prompts sit in `prompts/` next to this file, wherever it lives, and
SKILL.md is always at <repo>/.claude/skills/review-pr/SKILL.md.
"""

import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _skill_md():
    root = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    p = pathlib.Path(root, ".claude", "skills", "review-pr", "SKILL.md")
    return p.read_text(encoding="utf-8")


# (prompt file relative to HERE, start anchor, end anchor) — anchors are unique lines in SKILL.md.
CHECKS = [
    (
        "prompts/refuter.md",
        "## Step 7.7 — Independent refutation",
        "`rules.md` § Refutation has why.",
    ),
    (
        "prompts/worker.md",
        "**Output rules (strictly enforced):**",
        '- `⚠️ The benchmark may not include setup cost` — no "Author must" conclusion',
    ),
]


def extract(text, start, end):
    i = text.find(start)
    if i < 0:
        return None
    j = text.find(end, i)
    if j < 0:
        return None
    return text[i : j + len(end)]


def check_fail_classes():
    """run_one.sh tags each failure with `fail <class> ...`; _notify.py owns the class->owner
    map. Assert every class run_one can emit is known to _notify.py, so a failure never routes
    to nobody and run_one.sh / _notify.py cannot drift apart."""
    import importlib.util

    run_one = (HERE / "run_one.sh").read_text(encoding="utf-8")
    emitted = set(re.findall(r"\bfail\s+([a-z]+)\s+[0-9]", run_one))
    spec = importlib.util.spec_from_file_location("_notify", HERE / "_notify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    known = set(mod.CLASSES)
    unknown = emitted - known
    if unknown:
        print(f"\u274c run_one.sh emits fail classes unknown to _notify.py: {sorted(unknown)}")
        return 1
    print(f"\u2705 fail classes consistent: run_one {sorted(emitted)} \u2286 _notify.py {sorted(known)}")
    return 0


def main():
    skill = _skill_md()
    bad = 0
    for prompt, start, end in CHECKS:
        block = extract(skill, start, end)
        prompt_text = (HERE / prompt).read_text(encoding="utf-8")
        if block is None:
            print(f"❌ {prompt}: SKILL.md anchors not found — {start!r} .. {end!r}")
            bad += 1
        elif block in prompt_text:
            print(f"✅ {prompt}: quoted block matches SKILL.md verbatim ({len(block)} chars)")
        else:
            print(f"❌ {prompt}: quoted block DRIFTED from SKILL.md — re-copy the section verbatim")
            bad += 1
    bad += check_fail_classes()
    print(f"{'OK' if bad == 0 else 'DRIFT'}: {bad} drift(s)")
    return bad


if __name__ == "__main__":
    sys.exit(main())

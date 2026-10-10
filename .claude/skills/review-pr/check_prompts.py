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


WORKFLOW = HERE.parents[2] / ".github" / "workflows" / "aiter-review-bot.yml"


def _notify_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_notify", HERE / "_notify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_budget_fits():
    """The run budget and the job's timeout-minutes live in different files, and a budget that
    does not fit is silently useless: the job is killed mid-agent with no classified status --
    the exact failure the budget was added to prevent. Keep the two in step."""
    run_one = (HERE / "run_one.sh").read_text(encoding="utf-8")
    wf = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r'^: "\$\{AITER_RUN_BUDGET:=(\d+)\}"', run_one, re.M)
    j = re.search(r"^  review:.*?^    timeout-minutes: (\d+)", wf, re.M | re.S)
    if not m or not j:
        print("\u274c cannot read AITER_RUN_BUDGET or the review job's timeout-minutes")
        return 1
    budget, cap = int(m.group(1)), int(j.group(1)) * 60
    if budget >= cap:
        print(f"\u274c AITER_RUN_BUDGET={budget}s does not fit the review job's {cap}s cap")
        return 1
    print(f"\u2705 run budget fits the job cap: {budget}s < {cap}s")
    return 0


def check_owner_fallbacks():
    """Every message but two routes through _notify.py's class->owner map. The watchdog and the
    lost-review report post their own comments and carry their own copy of a handle, so they
    drift from the map in silence: the next owner updates _notify.py and still never hears that
    the runner is down, or that a review vanished."""
    wf = WORKFLOW.read_text(encoding="utf-8")
    classes = _notify_mod().CLASSES
    rows = [
        ("watchdog.py", "env",
         r'OWNER_OVERRIDE"\) or ""\)\.strip\(\) or "([A-Za-z0-9-]+)"'),
        ("lost_review.py", "flow",
         r'OWNER_OVERRIDE"\) or ""\)\.strip\(\) or "([A-Za-z0-9-]+)"'),
    ]
    bad = 0
    for filename, cls, pattern in rows:
        m = re.search(pattern, (HERE / filename).read_text(encoding="utf-8"))
        var, default = classes[cls][:2]
        if not m:
            print(f"\u274c cannot read {filename}'s owner fallback")
            bad += 1
            continue
        if m.group(1) != default:
            print(f"\u274c {filename} pages '{m.group(1)}', but _notify.py's {cls} class is "
                  f"'{default}'")
            bad += 1
            continue
        if not re.search(r"OWNER_OVERRIDE: \$\{\{ vars\." + var + r" \}\}", wf):
            print(f"\u274c {filename} defaults to _notify.py's {cls} owner but the workflow does "
                  f"not pass {var}")
            bad += 1
            continue
        print(f"\u2705 {filename} matches _notify.py's {cls} class: {default} via {var}")
    return 1 if bad else 0


def check_runner_label_exclusive():
    """boxBusy() decides 'is the box busy' by looking only at this workflow's runs. That holds
    only while nothing else targets the same runner label -- otherwise a healthy box doing other
    work reads as idle, and the watchdog pages the owner about it."""
    wf = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r"runs-on: \[self-hosted, ([A-Za-z0-9_-]+)\]", wf)
    if not m:
        print("\u274c cannot read the review job's runner label")
        return 1
    label = m.group(1)
    others = sorted(q.name for q in WORKFLOW.parent.glob("*.y*ml")
                    if q != WORKFLOW and label in q.read_text(encoding="utf-8"))
    if others:
        print(f"\u274c runner label '{label}' is also targeted by {others}; the watchdog's busy "
              "check only sees aiter-review-bot runs and would page about a busy, healthy box")
        return 1
    print(f"\u2705 runner label '{label}' is exclusive to aiter-review-bot.yml")
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
    bad += check_budget_fits()
    bad += check_owner_fallbacks()
    bad += check_runner_label_exclusive()
    print(f"{'OK' if bad == 0 else 'DRIFT'}: {bad} drift(s)")
    return bad


if __name__ == "__main__":
    sys.exit(main())

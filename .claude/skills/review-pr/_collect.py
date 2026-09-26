import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lib

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2:
    sys.exit("usage: collect.sh <WORK_DIR> [PR_number]")
W = sys.argv[1]
num, _ = _lib.pr_meta(W)
pr = sys.argv[2] if len(sys.argv) > 2 else (str(num) if num else None)
if not pr:
    sys.exit(
        f"cannot read PR number ({W}/pr_meta.json missing or corrupt); pass it explicitly: collect.sh {W} <PR_number>"
    )

missing = [f for f in _lib.REQUIRED if not _lib.nonempty(os.path.join(W, f))]
if missing:
    # Validate before making the dir: a refused collect must not leave an empty dir, or it
    # gets picked up by the report index.
    sys.exit(
        f"refusing to collect: {W} is missing required artifacts {', '.join(missing)}.\n"
        f"{W} is an incomplete WORK dir (fetch likely failed) -- re-run the review."
    )

D = HERE / "reports" / f"PR-{pr}"
D.mkdir(parents=True, exist_ok=True)

# SKILL.md's `not independently refuted` is a conditional annotation (written only "when
# there is no independent reader"). We split Step 7.7 to a separate agent, so when the worker
# wrote the card it did not yet know whether a refuter would run and had to annotate "none".
# If independent.txt exists, an independent refutation did happen and the suffix must come off,
# or the card is lying. Mechanically restore the state SKILL.md specifies; no review judgement.
_card = pathlib.Path(W) / "card.md"
if _lib.nonempty(os.path.join(W, "independent.txt")):
    _t = _card.read_text(encoding="utf-8")
    if "not independently refuted" in _t:
        _t2 = _t.replace(" — not independently refuted", "").replace(
            " -- not independently refuted", ""
        )
        _card.write_text(_t2, encoding="utf-8")
        print(
            "  removed 'not independently refuted' from the card (independent.txt exists)"
        )

copied = 0
for f in _lib.REQUIRED + _lib.OPTIONAL:
    src = os.path.join(W, f)
    if _lib.nonempty(src):
        shutil.copy2(src, D / f)
        copied += 1
    elif f in _lib.OPTIONAL:
        (D / f).unlink(missing_ok=True)


# GATES.txt is this report's credibility receipt; keep it with the artifacts — write failures in too
_proj = subprocess.run(
    ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
    capture_output=True,
    text=True,
    check=False,
).stdout.strip() or os.path.dirname(os.path.dirname(os.path.dirname(str(HERE))))
res = _lib.run_gates(W, _proj)
npass = sum(1 for _, rc, _, _ in res if rc == 0)
lines = [f"======== SEVEN GATES ({W}) ========"]
for name, rc, first, full in res:
    lines.append(
        f"  {'✅' if rc == 0 else '❌'} {name:<12} "
        + (
            first
            if rc == 0
            else f"exit={rc}\n" + "\n".join("       " + l for l in full.split("\n"))
        )
    )
lines.append(f"======== {npass} green / {len(res)-npass} red ========")
(D / "GATES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"  collected into {D}  ({copied} items, {npass}/7 green)")

# Refresh the aggregate report right after collect (a manual step gets forgotten, and a
# forgotten one leaves the report inconsistent with reality). _report.py is a dev-only tool
# and may not ship; skip it silently when absent.
_rep = HERE / "_report.py"
if _rep.is_file():
    subprocess.run([sys.executable, str(_rep)], check=False)
if npass != 7:
    print(
        f"  ⚠️ this report has {7-npass} gate(s) unpassed — the index marks it; do not treat it as a conclusion"
    )

# collect is the last automatic step; the next step, git commit, is manual, and "a manual
# step gets forgotten" is exactly why #4860 sat unstaged in the worktree. Remind explicitly
# and point at the breakpoint detector.
print(
    f"  ↳ not landed yet: git add reports/PR-{pr}/ && git commit to finish."
)

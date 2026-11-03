"""Drop the findings the independent refuter KILLED from the card, so the seven gates see only
what survived. In the interactive review flow a human removes KILLED findings by hand before
finalizing; run_one.sh calls this to do it headlessly -- without it, the very first review whose
refuter kills a finding (its designed job) fails triage.py's independent gate and loses the whole
review.

independent.txt has one `SURVIVED|KILLED -- ...` line per card finding, in the card's order
(triage.py's independent gate relies on that same 1:1 order). We map by position and, only when
the counts line up 1:1, remove the KILLED findings; on any mismatch we change nothing and let the
gate report it, rather than guess.

Usage: _apply_refutation.py <card.md> <independent.txt>
"""

import pathlib
import re
import sys

FIND_PREFIX = ("\U0001F534", "⚠", "\U0001F4DD")  # 🔴 ⚠️ 📝
VERDICT = re.compile(r"^\s*(SURVIVED|KILLED)\b")


def main():
    if len(sys.argv) < 3:
        print("usage: _apply_refutation.py <card.md> <independent.txt>"); return 2
    card_p = pathlib.Path(sys.argv[1])
    indep = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
    verdicts = [m.group(1) for m in (VERDICT.match(l) for l in indep.splitlines()) if m]

    lines = card_p.read_text(encoding="utf-8").splitlines(keepends=True)
    finding_lines = [i for i, l in enumerate(lines) if l.lstrip().startswith(FIND_PREFIX)]

    if not finding_lines or not verdicts:
        print("nothing to apply (no findings or no verdicts)"); return 0
    if len(verdicts) != len(finding_lines):
        # Ambiguous mapping -- do not guess which finding a verdict refers to. Leave the card as
        # is; the independent gate will report the count mismatch.
        print(f"skip: {len(finding_lines)} card findings vs {len(verdicts)} verdicts -- mismatch")
        return 0

    drop = {finding_lines[i] for i, v in enumerate(verdicts) if v == "KILLED"}
    if not drop:
        print("no KILLED findings; card unchanged"); return 0
    kept = [l for i, l in enumerate(lines) if i not in drop]
    card_p.write_text("".join(kept), encoding="utf-8")
    print(f"applied refutation: dropped {len(drop)} KILLED finding(s), kept {len(finding_lines) - len(drop)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

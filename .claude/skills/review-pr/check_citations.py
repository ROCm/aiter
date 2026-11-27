#!/usr/bin/env python3
"""Gate: every `file:line` a finding cites must point at what the finding says is there.

The other gates check that a finding which fired reaches the card. None of them check
that the card's citations are accurate. The evidence for this check is ATOM's, not
aiter's: a batch of seven real ATOM reviews produced twelve findings of which two
carried a citation defect -- one inflated a count, and one cited `envs.py:263` for a
construct that lives at :253, where :263 holds an unrelated env var entry. The mechanism
in both was sound. That rate has not been measured on aiter's own cards, and it may well
differ; what carries over is that a reviewer who clicks the line and finds different code
stops trusting the whole card, so a wrong citation costs more than the finding earns.

That is also why this does not gate: see main() for the reasoning, and _publish.py for
the one thing the verdicts do decide -- whether a finding may be posted inline.

Existence alone does not catch this: in the real case the file existed and the line was
in range. So each citation is also checked for proximity -- an identifier the finding
quotes in backticks must appear within PROXIMITY lines of the cited one. The tolerance
is deliberately loose enough for the line drift that review of a moving branch produces,
and tight enough for the ten-line miss that actually happened.

Usage:  check_citations.py <WORK_DIR> [TREE]
        TREE defaults to <WORK_DIR>/merge-target, falling back to <WORK_DIR>/head.
Exit 0 = every citation checks out (or the card cites nothing). Exit 1 = at least one
does not; the offending lines are printed.
"""

import pathlib
import re
import sys

PROXIMITY = 5  # lines; see module docstring
FINDING = re.compile(r"^(🔴|⚠️|📝)")
# path/to/file.ext:1234 -- the extension keeps 'PR#2695:' and bare 'note:7' out
CITATION = re.compile(r"\b([\w./-]+\.[A-Za-z]{1,6}):(\d+)\b")
# What a finding says is at the cited line, in the two shapes cards use:
# `an_identifier` in backticks, or "an excerpt of the line" in double quotes.
# Reading only the first misjudges every finding that quotes the text instead --
# a stale-docstring finding names the docstring, not a symbol.
QUOTED = re.compile(r"`([A-Za-z_][\w.]{2,})`")
EXCERPT = re.compile(r"[\"\u201c]([^\"\u201c\u201d]{12,})[\"\u201d]")


def trees_for(work: pathlib.Path, explicit: str | None):
    """The trees a citation may resolve against, in the order to try them.

    `head/` holds the PR's versions of the files it changed; `merge-target/` is the full
    tree at BASE_SHA, the code before this PR. A finding citing a line the PR added
    exists only in the first; a finding citing untouched context exists only in the
    second. One tree for everything gets one of those two wrong.

    It used to take merge-target for everything, and the reviewer found that on
    aiter#7303 itself: citations into files that PR grew reported `_publish.py:221 --
    file has 171 lines`, 171 being the length on base. Every citation was flagged, every
    anchor vetoed, and the review published with zero inline comments while all seven
    gates read green.
    """
    if explicit:
        p = pathlib.Path(explicit)
        return [p] if p.is_dir() else []
    return [work / n for n in ("head", "merge-target") if (work / n).is_dir()]


def check(card_text: str, trees, changed=()):
    """Yield one complaint string per bad citation.

    `changed` is the PR's changed paths; a bare basename resolves against those first.
    """
    for raw in card_text.split("\n"):
        if not FINDING.match(raw.strip()):
            continue
        quoted = set(QUOTED.findall(raw))
        excerpts = [e.strip() for e in EXCERPT.findall(raw)]
        for path, lineno in CITATION.findall(raw):
            # Try each tree in order: head/ for a file this PR changed, merge-target/
            # for context code it did not touch.
            f = next((t / path for t in trees if (t / path).is_file()), None)
            tree = next((t for t in trees if (t / path).is_file()), trees[0])
            if f is None:
                f = trees[0] / path
                # Cards usually cite a bare basename. Resolve it the way a reader does:
                # against the files this PR actually touched, not the whole tree -- a
                # repo this size has several `scheduler.py`, and only one is in the diff.
                base = pathlib.Path(path).name
                hits = [c for c in changed if pathlib.Path(c).name == base]
                if len(hits) != 1:
                    # Search the FULL tree, not trees[0]. head/ holds only the files this PR
                    # touched, so a basename search there can never find untouched context
                    # code -- and the miss is reported as "no such file in the tree", i.e. an
                    # invented path, when the file plainly exists and this PR simply did not
                    # change it. That hard verdict then costs the whole finding its anchor
                    # and its permalink (_publish._unflagged_cites), including a correct,
                    # fully-qualified in-diff citation sitting in the same sentence. So a
                    # bare-basename citation of real code switched anchoring off silently,
                    # which is the failure this checker exists to prevent.
                    tree = next((t for t in trees if t.name == "merge-target"), tree)
                    wide = tree
                    hits = [
                        str(m.relative_to(wide))
                        for m in wide.rglob(base)
                        if m.is_file()
                    ]
                if not hits:
                    # Nothing by that name anywhere: the path is invented.
                    yield "bad", f"{path}:{lineno} -- no such file in the tree"
                    continue
                if len(hits) != 1:
                    # The name is real but under-specified -- several files carry it and
                    # none is in the diff, so the finding is pointing at context code by
                    # bare basename. A reader cannot follow that, which is worth saying,
                    # but it does not make the finding wrong the way an invented path
                    # does. Counted on this tree: aiter has eight utils.py and seven
                    # quant.py, so a bare basename citation is routinely ambiguous here.
                    yield "note", (
                        f"{path}:{lineno} -- ambiguous, {len(hits)} files share that "
                        f"name and none is in this diff; cite the directory"
                    )
                    continue
                # hits[0] came either from `changed` (a path in the diff, present in head/)
                # or from the basename search (a path in merge-target/). Resolve against
                # whichever tree actually holds it rather than assuming one of them.
                f = next(
                    (t / hits[0] for t in trees if (t / hits[0]).is_file()), tree / hits[0]
                )
                if not f.is_file():
                    yield "bad", f"{path}:{lineno} -- {hits[0]} is in the diff but not the tree"
                    continue
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError as e:
                yield "bad", f"{path}:{lineno} -- unreadable ({e})"
                continue
            n = int(lineno)
            if not 1 <= n <= len(lines):
                yield "bad", f"{path}:{lineno} -- file has {len(lines)} lines"
                continue
            if not quoted and not excerpts:
                continue  # nothing quoted to locate; existence is all we can check
            lo, hi = max(1, n - PROXIMITY), min(len(lines), n + PROXIMITY)
            window = "\n".join(lines[lo - 1 : hi])
            if any(q in window for q in quoted):
                continue
            if any(e in window for e in excerpts):
                continue
            whole = "\n".join(lines)
            elsewhere = [q for q in quoted if q in whole]
            if not elsewhere and quoted:
                # Nothing the finding quotes appears in the cited file at all. This fell
                # through silently before, which inverted the severity: a symbol present
                # but far from the cited line got flagged, while one that is not in the
                # file is the stronger sign the citation points somewhere else. Found on
                # PR#2707, whose finding cited model_runner.py:4465 for `uses_router_gemm`
                # -- a symbol that lives in linear.py and appears nowhere in model_runner.
                # (ATOM's case; the same shape of defect is what this looks for here.)
                # Restricted to identifier-shaped tokens: a finding may legitimately quote
                # a value (`per_token`) that is not source text in that file.
                ident = [q for q in quoted if "_" in q or "." in q or len(q) > 12]
                if ident:
                    yield "note", (
                        f"{path}:{lineno} -- none of {sorted(ident)} appears anywhere in "
                        f"this file; check the citation points where the finding means"
                    )
                continue
            if elsewhere:
                where = []
                for q in elsewhere[:1]:
                    for i, l in enumerate(lines, 1):
                        if q in l:
                            where.append(f"`{q}` is at :{i}")
                            break
                yield "note", (
                    f"{path}:{lineno} -- none of {sorted(quoted)} within "
                    f"±{PROXIMITY} lines; {'; '.join(where)}"
                )


def main(argv):
    if len(argv) < 2:
        sys.exit("usage: check_citations.py <WORK_DIR> [TREE]")
    work = pathlib.Path(argv[1])
    card = work / "card.md"
    if not card.is_file():
        print("CITATIONS: no card.md to check")
        return 1
    trees = trees_for(work, argv[2] if len(argv) > 2 else None)
    if not trees:
        print("CITATIONS: no tree to resolve citations against")
        return 1
    diff = work / "pr.diff"
    changed = ()
    if diff.is_file():
        changed = re.findall(
            r"^\+\+\+ b/(.+)$",
            diff.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
    out = list(check(card.read_text(encoding="utf-8"), trees, changed))
    bad = [m for sev, m in out if sev == "bad"]
    notes = [m for sev, m in out if sev == "note"]
    # Only the deterministic failures gate. Proximity is a heuristic -- matching prose
    # against source lines is fuzzy, and two rounds of tightening it on seven real
    # reviews traded one class of false alarm for another (an excerpt rule that fixed a
    # stale-docstring finding broke three findings whose line numbers were exact). A
    # heuristic that withholds correct reviews costs more than the defect it catches, so
    # it prints for the human reading the card and does not fail the gate.
    for n in notes:
        print(f"CITATIONS NOTE: {n}")
    if bad:
        print(f"CITATIONS BAD: {len(bad)} citation(s) do not resolve")
        for b in bad:
            print(f"  {b}")
        return 1
    print(
        "CITATIONS: every cited file:line resolves"
        + (f" ({len(notes)} proximity note(s))" if notes else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

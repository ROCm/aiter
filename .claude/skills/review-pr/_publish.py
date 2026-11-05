"""Publish gate (phase-2): post a collected report to the PR, as aiter-bot, under the rules.

The card goes up as a PR *review* (`event: "COMMENT"`, never APPROVE/REQUEST_CHANGES -- this
bot is advisory and does not gate merges), with the findings that can be anchored attached to
the lines they are about and the rest left in the review summary. A finding that cites a span
(`path:1002-1012`) is anchored to the whole span, not to its first line: the reader otherwise
sees a comment pinned to one line of a block the finding is about, which reads as a claim
about that line. A citation the diff does not reach cannot be anchored at all, and instead
earns a blob permalink pinned to the head SHA in the summary, so the reader sees the code
without going to find it by hand. Measured on 14 real cards
(ATOM's, where this mechanism was built; the shape of an aiter card is the same): a card's
findings average 2081 characters in one top-level comment, a single finding averages 662, and
half of all findings cite a line that is inside the PR's diff. Those halves read far better
next to the code than buried in a wall of prose.

A finding is anchored only if BOTH hold:
  - the cited line is in the diff (GitHub rejects anything else), and
  - check_citations.py did not flag that citation (neither a hard failure nor a note).
The second is the one that matters. A top-level card that cites a wrong line is a flaw the
reader works around; an inline comment on the wrong line lands in the author's working view
and is visibly wrong, and a reviewer who gets burned once stops reading the rest. Anchoring is
a privilege a finding earns by having a citation that checks out -- when citations.txt is
missing or the checker could not run, nothing is anchored and the whole card stays in the
summary, which is exactly what this bot posted before. card.md itself is never trimmed: it is
the archived record and the gates check its shape; the split is a presentation decision made
at publish time.

Rules (after run_one, given 7 green gates):
  - Review line is 🔴 HIGH RISK  → POST too: the reviewer who triggered it must see the high-risk
    findings (still advisory, not a merge gate); a warning annotation also flags it in the checks UI
  - otherwise (✅ / ⚠️ / notes only) → POST: post card.md as a PR review
  - already posted (same head SHA, by marker) → SKIP: do not repost
  - PR already merged/closed         → SKIP: a post-merge follow-up should not go as a PR review
Publishing identity = the owner of AITER_BOT_TOKEN(_FILE) (aiter-bot). Dry-run by default; --post actually posts.

A marker line is embedded at the end of the review summary for followup/dedup:
  <!-- aiter-bot review pr=<pr> head=<sha> -->
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_citations  # noqa: E402  -- the one authority on whether a citation resolves

REPORTS = HERE / "reports"
REPO = os.environ.get("AITER_BOT_REPO") or os.environ.get(
    "GITHUB_REPOSITORY", "ROCm/aiter"
)
BOT = os.environ.get("AITER_BOT_NAME", "aiter-bot")
# The dedup marker is deliberately NOT built from BOT. It is a literal that has already been
# written into comments on live PRs; deriving it from an overridable env var would let a box
# with AITER_BOT_NAME set repost over every review the real bot already published.
MARKER_BOT = "aiter-bot"


def _token():
    t = os.environ.get("AITER_BOT_TOKEN")
    if t:
        return t.strip()
    tf = os.environ.get("AITER_BOT_TOKEN_FILE")
    if tf:
        try:
            m = re.findall(
                r"(?:ghp_|github_pat_)[A-Za-z0-9_]+",
                pathlib.Path(tf).read_text(encoding="utf-8"),
            )
            if m:
                return m[-1]
        except OSError:
            pass
    try:
        with open(os.path.expanduser("~/.git-credentials"), encoding="utf-8") as fh:
            for line in fh:
                if "github.com" in line and ":" in line:
                    return (
                        line.split("://", 1)[1].split("@")[0].split(":", 1)[1].strip()
                    )
    except OSError:
        pass
    return ""


TOK = _token()


def _api(path, method="GET", body=None):
    cmd = ["curl", "-s", "-m", "25", "-X", method]
    if TOK:
        cmd += ["-H", f"Authorization: token {TOK}"]
    cmd += ["-H", "Accept: application/vnd.github+json"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    cmd.append(f"https://api.github.com/repos/{REPO}/{path}")
    try:
        return json.loads(
            subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
            or "null"
        )
    except (json.JSONDecodeError, OSError):
        return None


def _api_raw(path, accept):
    """Same call as _api but returns the body unparsed (used for the .diff media type)."""
    cmd = ["curl", "-s", "-m", "25"]
    if TOK:
        cmd += ["-H", f"Authorization: token {TOK}"]
    cmd += ["-H", f"Accept: {accept}", f"https://api.github.com/repos/{REPO}/{path}"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""


def head_sha(pr):
    try:
        d = json.loads(
            (REPORTS / f"PR-{pr}" / "pr_meta.json").read_text(encoding="utf-8")
        )
        return d.get("headRefOid") or (d.get("head") or {}).get("sha") or ""
    except (OSError, json.JSONDecodeError):
        return ""


def severity(card):
    """Severity of the Review line: 'high' | 'ok' (anything not 🔴 is ok, i.e. postable)."""
    for ln in card.splitlines():
        if ln.startswith("Review"):
            if "🔴" in ln or "HIGH RISK" in ln.upper():
                return "high"
            return "ok"
    return "ok"


# ---------------------------------------------------------------- anchoring

FINDING = re.compile(r"^(🔴|⚠️|📝)\s")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
# A citation that names a span: `path:1002-1012`. Deliberately tight -- an ASCII hyphen with
# no whitespace around it. The loose form (spaces allowed, or an en/em dash) misreads ordinary
# prose: a real card writes "flash_kda.py:381 — 6 published tuning candidates", which a loose
# pattern reads as the span 381..6 and would hand to GitHub as a backwards range, failing the
# whole review including the summary. Counted over 221 real findings on this bot's own cards,
# the tight form matched 25 spans with no false positive; the loose form added exactly that
# one sentence and nothing true.
CITE_SPAN = re.compile(r"\b([\w./-]+\.[A-Za-z]{1,6}):(\d+)-(\d+)\b")
# Past this width a span is not a "section" any more -- it is a comment painted over most of
# a hunk, which says less about where the problem is than the first line alone does.
MAX_SPAN = 100
# How much of a finding the summary index shows. Long enough to recognise the finding,
# short enough that the index stays one line per entry -- which is the whole point of it.
CLAUSE = 110
# The card's status lines, by the prefix each starts with. aiter's SKILL.md writes these as
# `Review (advisory):`, `Validation (deterministic):` and `Perf (advisory|deterministic):` --
# the parenthetical says which kind of evidence is behind the line, so matching on the bare
# word keeps every variant. Perf has no counterpart on an ATOM card and is aiter's own; it
# belongs in the summary for the same reason Validation does, since "NOT RUN" and "NO
# REGRESSION" are different facts and neither is a finding.
STATUS_PREFIXES = ("Review", "Validation", "Perf")


def pr_diff(pr):
    """The PR's unified diff: archived next to the report if run_one left one, else fetched."""
    f = REPORTS / f"PR-{pr}" / "pr.diff"
    if f.is_file():
        return f.read_text(encoding="utf-8", errors="replace")
    return _api_raw(f"pulls/{pr}", "application/vnd.github.v3.diff") or ""


def commentable(diff_text):
    """-> {path: {new_line, ...}} -- the RIGHT-side lines a review comment may attach to.

    GitHub rejects a comment on a line outside the diff, so a finding about context code the
    PR does not touch simply cannot be anchored. That is correct, not a failure.
    """
    out, path, new = {}, None, 0
    # The final newline terminates the last line; it does not introduce an empty one. Keeping
    # it would invent a commentable line one past the end of the last hunk -- and a comment on
    # a line that is not in the diff is rejected by the API, taking the whole review with it.
    if diff_text.endswith("\n"):
        diff_text = diff_text[:-1]
    for ln in diff_text.split("\n"):
        if ln.startswith("diff --git "):
            path, new = None, 0
            continue
        if ln.startswith("+++ "):
            p = ln[4:].strip()
            path = p[2:] if p.startswith("b/") else (None if p == "/dev/null" else p)
            continue
        m = HUNK.match(ln)
        if m:
            new = int(m.group(1))
            continue
        if path is None or not new:
            continue
        # Added and context lines exist on the right side and carry a new-file line number;
        # deletions and the "\ No newline" marker do not.
        if ln.startswith("+") or ln.startswith(" ") or ln == "":
            out.setdefault(path, set()).add(new)
            new += 1
    return out


def flagged_citations(pr):
    """-> set of 'path:line' that check_citations.py would not vouch for, or None if unknown.

    Reuses the checker's own verdicts (run_one writes them to citations.txt) rather than
    re-deciding here: both halves count, the hard failures and the proximity notes. None
    means the checker never ran on this card -- and a citation nobody checked is not a
    citation that checks out, so the caller anchors nothing.
    """
    f = REPORTS / f"PR-{pr}" / "citations.txt"
    if not f.is_file():
        return None
    text = f.read_text(encoding="utf-8", errors="replace")
    if "CITATIONS: no card.md" in text or "CITATIONS: no tree" in text:
        return None
    if not text.strip():
        return None
    # A traceback is not a verdict. The checker runs under `|| true` with its output
    # tee'd here, so a crash leaves a Python traceback and nothing else -- which read
    # as "no citation was flagged" and let every finding anchor with no checking at
    # all. This module's own contract above is the opposite: if the checker could not
    # run, nothing is anchored. No verdict line IS that case, however the file got
    # that way. Measured on this bot's own PR, where citations.txt was such a crash.
    if not any(ln.startswith("CITATIONS") for ln in text.split("\n")):
        return None
    out = set()
    # Hard failures only. A `CITATIONS BAD:` entry means the citation resolves to
    # nothing -- absent file, or a line past the end of it -- so anchoring there would
    # fail at the API or point at empty space. A `CITATIONS NOTE:` says a quoted symbol
    # is not within five lines of the cited one: a quality hint the checker's own
    # docstring puts at roughly 50% precision, not evidence the line is wrong. Counting
    # notes was measured switching the feature off entirely on aiter#7303 -- five cited
    # findings, zero anchors, four of them vetoed by notes.
    in_bad = False
    for ln in text.split("\n"):
        if ln.startswith("CITATIONS BAD:"):
            in_bad = True
            continue
        if ln.startswith("CITATIONS"):
            in_bad = False
            continue
        if not (in_bad and ln.startswith("  ")):
            continue
        m = check_citations.CITATION.search(ln)
        if m:
            out.add(f"{m.group(1)}:{m.group(2)}")
    return out


def citations(finding):
    """-> [(path, start, end)] in the order the finding writes them; end == start for a
    plain `path:line`.

    Spans are matched first and blanked out of the text before the single-line pattern runs
    over what is left. Otherwise `path:1002-1012` is read twice -- once as the span and once
    as the plain citation `path:1002` -- and the finding would carry both the block it is
    about and a second claim about the block's first line.
    """
    out, rest = [], finding
    for m in CITE_SPAN.finditer(finding):
        a, b = int(m.group(2)), int(m.group(3))
        # A span that runs backwards is not a span; it is prose that put a number after a
        # hyphen. Keep its opening line as a plain citation rather than discard the finding's
        # only pointer -- and never let it reach GitHub as a range.
        out.append((m.start(), m.group(1), a, b if b > a else a))
        rest = rest[: m.start()] + " " * (m.end() - m.start()) + rest[m.end() :]
    for m in check_citations.CITATION.finditer(rest):
        n = int(m.group(2))
        out.append((m.start(), m.group(1), n, n))
    return [(path, a, b) for _, path, a, b in sorted(out)]


def resolve_path(path, cmap):
    """-> the changed file a citation names, or None.

    Cards cite a bare basename constantly. One that matches exactly one changed file is that
    file; one that matches several is not resolvable, and picking between them is precisely
    how a comment -- or a link -- lands on code the finding is not about.
    """
    if path in cmap:
        return path
    base = path.rsplit("/", 1)[-1]
    same = [k for k in cmap if k.rsplit("/", 1)[-1] == base]
    return same[0] if len(same) == 1 else None


def _unflagged_cites(finding, flagged):
    """-> the finding's citations, or None if it may not point at anything.

    A finding carrying ANY citation the checker flagged is not anchored and not linked at
    all, even if some other citation in the same sentence would resolve: the flag says this
    finding's pointing is unreliable, and inline -- or behind a link the reader follows --
    is exactly where unreliable pointing costs the most.
    """
    if flagged is None:
        return None
    cites = citations(finding)
    if any(f"{p}:{a}" in flagged for p, a, _ in cites):
        return None
    return cites


def anchor_for(finding, cmap, flagged):
    """-> (path, start, end) the finding may be attached to, or None. end == start is a
    single-line anchor; end > start is a span.
    """
    cites = _unflagged_cites(finding, flagged)
    if cites is None:
        return None
    for path, a, b in cites:
        hit = resolve_path(path, cmap)
        if not hit:
            continue
        lines = cmap[hit]
        if b > a:
            # Every line of the span must be commentable, not just the two ends. GitHub takes
            # a span only within one hunk, and `commentable` holds one contiguous run per
            # hunk -- so "every line present" is exactly "inside one hunk". Checking the ends
            # alone would accept a span that jumps the gap between two hunks, and that single
            # malformed entry fails the entire review, summary and all other comments with it.
            if b - a + 1 <= MAX_SPAN and all(n in lines for n in range(a, b + 1)):
                return hit, a, b
            # Only part of the span is inside the diff. Anchor the end that is, rather than
            # drop a real finding because its citation is wider than what the PR touched.
            if a in lines:
                return hit, a, a
            if b in lines:
                return hit, b, b
            continue
        if a in lines:
            return hit, a, a
    return None


def cite_text(path, a, b):
    """`path:12` or `path:12-18` -- how a human writes the anchor this bot computed."""
    return f"{path}:{a}" if b == a else f"{path}:{a}-{b}"


def comment_for(path, a, b, body):
    """-> one entry of the review's comments[].

    GitHub spells a span as start_line..line with a side for each end. start_line must be
    strictly below line, which is why a backwards or degenerate span never gets here: it was
    already collapsed to a single line in citations().
    """
    c = {"path": path, "line": b, "side": "RIGHT", "body": body}
    if b > a:
        c["start_line"] = a
        c["start_side"] = "RIGHT"
    return c


def anchor_label(c):
    """`path:12-18` for a comments[] entry, span or not -- used when folding anchors back
    into the summary, where the reader has only the text to go on."""
    return cite_text(c["path"], c.get("start_line", c["line"]), c["line"])


def permalink(path, a, b, sha):
    """A blob URL pinned to the head SHA. Pinned, not a branch: the branch moves, and a link
    that drifts off the code it was written about is the same defect as a wrong anchor,
    arriving later."""
    frag = f"#L{a}" if b == a else f"#L{a}-L{b}"
    return f"https://github.com/{REPO}/blob/{sha}/{path}{frag}"


def permalinks_for(finding, cmap, flagged, sha):
    """-> the blob permalinks a finding that could NOT be anchored has earned.

    GitHub will not take a comment on a line outside the diff, correctly, so such a finding
    stays in the summary as bare text the reader has to go and look up. A permalink is what
    can be offered instead -- GitHub expands one into the code it names.

    Two gates, and both exist so this never links to the wrong code. The checker's verdict
    applies exactly as it does to anchoring: a flagged finding gets no link. And the path has
    to resolve to a file this PR actually touched -- a blob URL built from an unresolved
    basename is a guess, and a guess that 404s is the better of its two outcomes.
    """
    cites = _unflagged_cites(finding, flagged)
    if cites is None or not sha:
        return []
    out = []
    for path, a, b in cites:
        hit = resolve_path(path, cmap)
        if not hit:
            continue
        u = permalink(hit, a, b, sha)
        if u not in out:
            out.append(u)
    return out


def first_clause(finding, cite=""):
    """The finding's opening clause, for the summary index."""
    s = FINDING.sub("", finding).strip()
    # Drop a leading [verified] / [inferred] tag, and the citation the index line already
    # carries -- both cost width and say nothing new there.
    s = re.sub(r"^\[[^\]]{0,60}\]\s*", "", s)
    if cite:
        # `(?:-\d+)?` is not decoration. When a finding cites `path:563-569` but only 563 is
        # inside the diff, the anchor -- and so `cite` -- is `path:563`, and stripping only
        # that leaves the index line opening with a bare "-569". Found on a real card the
        # first time this ran over one.
        s = re.sub(r"^`?" + re.escape(cite) + r"(?:-\d+)?`?[,:]?\s*", "", s)
    for sep in (" — ", " -- ", "; "):
        if sep in s[: CLAUSE + 40]:
            s = s.split(sep, 1)[0]
            break
    if len(s) > CLAUSE:
        s = s[:CLAUSE].rsplit(" ", 1)[0] + "…"
    return s.strip()


def _lines_starting(card, prefix):
    return [ln for ln in card.split("\n") if ln.startswith(prefix)]


def _line_starting(card, prefix):
    hits = _lines_starting(card, prefix)
    return hits[0] if hits else ""


# The inline index exactly as build_review writes it, so the degraded retry can remove it.
# Kept beside nothing else: if build_review's wording changes and this stops matching, the
# test below fails rather than the index silently surviving into a review with no anchors.
INDEX_BLOCK = re.compile(
    r"\n*\*\*\d+ finding\(s\) posted inline, on the lines they are about:\*\*\n(?:.*\n?)*?(?=\n\n|\Z)"
)


def build_review(pr, card, diff_text, flagged, sha):
    """-> the exact body of POST /pulls/{pr}/reviews.

    event is always COMMENT. This bot is advisory: APPROVE or REQUEST_CHANGES would make it a
    merge gate and would change what the workflow's permissions mean.
    """
    cmap = commentable(diff_text)
    comments, index, loose = [], [], []
    for ln in card.split("\n"):
        if not FINDING.match(ln):
            continue
        a = anchor_for(ln, cmap, flagged)
        if a:
            comments.append(comment_for(a[0], a[1], a[2], ln.strip()))
            cite = cite_text(*a)
            index.append(f"- `{cite}` — {first_clause(ln, cite)}")
        else:
            links = permalinks_for(ln, cmap, flagged, sha)
            # Each URL alone on its own line. GitHub expands a bare blob permalink into the
            # code it points at only when nothing shares the line with it; inside the
            # sentence it collapses to a shortened link, which tells the reader less than the
            # `path:line` already sitting there. The finding's own text is left exactly as
            # the card wrote it -- the link is in addition, never instead.
            loose.append("\n\n".join([ln.strip()] + links))

    parts = [p for p in (_line_starting(card, "## "),) if p]
    status = []
    for pref in STATUS_PREFIXES:
        status += _lines_starting(card, pref)
    if status:
        # One block, in card order: these lines belong together and a blank line between each
        # turns four facts into four paragraphs.
        parts.append("\n".join(status))
    if loose:
        parts.append("\n\n".join(loose))
    if index:
        # Not decoration. Without it a reader of the summary alone cannot tell whether three
        # findings exist elsewhere or none do -- and an invisible finding reads exactly like a
        # finding that was never made.
        parts.append(
            f"**{len(index)} finding(s) posted inline, on the lines they are about:**\n"
            + "\n".join(index)
        )
    ind = _line_starting(card, "_Independent refutation:")
    if ind:
        parts.append(ind)
    parts.append(f"<!-- {MARKER_BOT} review pr={pr} head={sha} -->")
    return {
        "event": "COMMENT",
        "commit_id": sha,
        "body": "\n\n".join(parts),
        "comments": comments,
    }


def review_payload(pr):
    d = REPORTS / f"PR-{pr}"
    return build_review(
        pr,
        (d / "card.md").read_text(encoding="utf-8").rstrip(),
        pr_diff(pr),
        flagged_citations(pr),
        head_sha(pr),
    )


def already_posted(pr, sha):
    """Whether aiter-bot already posted for this head (by marker).

    Both endpoints: the marker used to ride in an issue comment and now rides in a review
    summary. aiter's bot has been posting issue comments on live PRs, so a dedup check that
    only read the reviews endpoint would repost on top of every one of those.
    """
    marker = f"{MARKER_BOT} review pr={pr} head={sha}"
    for ep in (
        f"issues/{pr}/comments?per_page=100",
        f"pulls/{pr}/reviews?per_page=100",
    ):
        cs = _api(ep)
        if isinstance(cs, list) and any(marker in (c.get("body") or "") for c in cs):
            return True
    return False


def decide(pr):
    """-> (action, reason). action in POST | HOLD | SKIP."""
    d = REPORTS / f"PR-{pr}"
    card = d / "card.md"
    if not card.is_file():
        return "SKIP", f"reports/PR-{pr}/card.md does not exist (run run_one first)"
    pull = _api(f"pulls/{pr}")
    if isinstance(pull, dict) and (
        pull.get("merged_at") or pull.get("state") != "open"
    ):
        state = "merged" if pull.get("merged_at") else pull.get("state")
        return (
            "SKIP",
            f"PR already {state} (post-merge follow-up, should not go as a PR review)",
        )
    sha = head_sha(pr)
    if sha and already_posted(pr, sha):
        return "SKIP", f"same head {sha[:9]} already posted (marker hit)"
    # Post the card for EVERY verdict, including 🔴: the reviewer who asked for the review must
    # see it, and 🔴 is exactly what they most need to see. It stays advisory (the Review line
    # says so) and never gates merge; main() adds a warning annotation for the 🔴 ones.
    return "POST", "postable"


def do_post(pr):
    """Post the card as a review. Only ever reached from main() under --post."""
    payload = review_payload(pr)
    assert payload["event"] == "COMMENT", "this bot never approves or requests changes"
    # A review is all-or-nothing: GitHub rejects the whole call -- summary included -- if
    # any one comments[] entry names a line it will not accept. A span anchor is a second
    # way to produce exactly that: start_line..line must sit inside one hunk, and a range
    # built across a gap is refused like any bad line. The anchor set is computed from the
    # diff's own hunks, but an off-by-one there would lose the entire review,
    # which is strictly worse than the plain comment this replaced -- and that plain
    # comment is what aiter's bot posts today, so the degraded path must land a review
    # that is no worse than what it displaced. So: try anchored, and if that is refused,
    # send the same review with the anchors dropped. Same endpoint and the same single
    # call site either way, so the drift check in check_prompts still holds.
    r = None
    for attempt in ("anchored", "summary-only"):
        if attempt == "summary-only":
            # Fold the anchors back into the body before dropping them. A review that
            # silently sheds half its findings reads exactly like a review that only had
            # the other half -- the failure this project keeps finding in its own parts.
            payload = dict(payload)
            # Drop the inline index first. It says "N finding(s) posted inline, on the lines
            # they are about" -- true of the attempt GitHub just refused, false of the one
            # being sent, where comments[] is empty. Left in, the degraded review claims N
            # findings sit on lines that carry nothing and then prints the same N again
            # below: the reader is told to look somewhere there is nothing, which is the
            # inverse of the silence this fallback exists to prevent.
            payload["body"] = INDEX_BLOCK.sub("", payload["body"]).rstrip()
            payload["body"] = payload[
                "body"
            ] + "\n\n_The findings below could not be attached to their lines — GitHub " "refused the anchored review, so they are inlined here instead._\n\n" + "\n\n".join(
                f"`{anchor_label(c)}` — {c['body']}" for c in payload["comments"]
            )
            payload["comments"] = []
        r = _api(f"pulls/{pr}/reviews", method="POST", body=payload)
        if isinstance(r, dict) and r.get("html_url"):
            n = len(payload["comments"])
            how = (
                ""
                if attempt == "anchored"
                else "  [ANCHORS REFUSED -- folded into summary]"
            )
            return True, (
                f'{r["html_url"]}  ({n} inline, '
                f'{len(payload["body"])} chars in summary){how}'
            )
        if not payload["comments"]:
            break  # the summary-only attempt failed too; nothing left to degrade to
    # Both attempts refused. The fallback above covers a bad ANCHOR; it cannot cover a token
    # that may not create reviews at all. AITER_BOT_TOKEN has been posting issue comments,
    # which a fine-grained PAT can do with `Issues: write` alone -- creating a PR review needs
    # `Pull requests: write`. If the PAT is fine-grained and lacks it, every review fails here
    # and the card is lost, where before this change it would have been posted. There is
    # deliberately no third path that re-posts as an issue comment: that would be a second POST
    # call site, which is exactly what check_prompts.check_publish_review_is_advisory refuses,
    # and a bot with two publishing routes drifts into using the wrong one. Verify the token's
    # scope instead -- the failure is loud (a red step and the API message below), not silent.
    return False, str(r)[:200]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("prs", nargs="+", help="PR number(s)")
    ap.add_argument(
        "--post",
        action="store_true",
        help="actually post; default dry-run prints the decision only",
    )
    ap.add_argument(
        "--payload",
        action="store_true",
        help="print the exact JSON that would be POSTed to /pulls/<pr>/reviews (never posts)",
    )
    a = ap.parse_args(argv)
    print(
        f'=== publish gate  [{"POST" if a.post else "DRY-RUN"}]  identity={BOT}  repo={REPO} ==='
    )
    posted, failed = [], []
    for pr in a.prs:
        action, reason = decide(pr)
        if action == "POST" and a.post:
            ok, info = do_post(pr)
            print(f'  #{pr}: POST -> {"✅ "+info if ok else "❌ "+info}')
            (posted if ok else failed).append(pr)
            if (
                ok
                and severity(
                    (REPORTS / f"PR-{pr}" / "card.md").read_text(encoding="utf-8")
                )
                == "high"
            ):
                # extra signal in the checks UI; the card itself is already posted as a comment
                print(
                    f"::warning title=aiter-bot HIGH RISK::PR #{pr} — review flags HIGH RISK (advisory, not a merge gate)"
                )
        else:
            print(f"  #{pr}: {action} — {reason}")
        if a.payload and (REPORTS / f"PR-{pr}" / "card.md").is_file():
            print(json.dumps(review_payload(pr), indent=2, ensure_ascii=False))
    if failed:
        print(f"  ❌ failed to post: {failed}")
        # Exit non-zero, or a review that was refused is indistinguishable from one
        # that went up. The refusal this most likely is -- a token that may post issue
        # comments but not create reviews -- would otherwise be invisible to the caller
        # on every single run, and the card would be lost each time while the step
        # stayed green. The API message is printed above; this makes someone read it.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

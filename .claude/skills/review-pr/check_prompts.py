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
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _skill_md():
    """SKILL.md sits next to this file; do not ask git where the repo root is.

    `git rev-parse --show-toplevel` returns an empty string outside a checkout, and
    Path("", ".claude", ...) is then a RELATIVE path resolved against the caller's cwd.
    So the drift check died with FileNotFoundError on a clean `git archive HEAD` export
    -- which is what this PR's own description tells a reviewer to verify from, and the
    one place the check absolutely has to work. It passed everywhere it was tried
    because every trial was inside a git checkout.
    """
    return (HERE / "SKILL.md").read_text(encoding="utf-8")


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
        ("queue_gate.py", "env",
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


def check_no_self_trigger():
    """The bot's own comments are issue_comment events like any other. Without an explicit
    exclusion every card and queue notice starts a run of this workflow, which is gated off a
    moment later -- but only after taking a hosted-runner slot the real requests are waiting
    for. The guard is one line and leaves no trace when removed, so assert it."""
    wf = WORKFLOW.read_text(encoding="utf-8")
    if not re.search(r"github\.event\.comment\.user\.login != 'aiter-bot'", wf):
        print("\u274c the gate does not exclude the bot's own comments; every notice it posts "
              "will start a run that competes with real requests")
        return 1
    print("\u2705 the gate excludes the bot's own comments")
    return 0


def check_notified_handoff():
    """The report job stays quiet when the review job says it already spoke. That answer travels
    _notify.py -> step output -> job output -> the report job's env, and every link is a string
    that something else has to spell the same way. Break one and nothing errors: the flag simply
    arrives empty, the report fires on every failed review, and the duplicate notice looks like
    the bot being broken rather than the wiring being broken."""
    wf = WORKFLOW.read_text(encoding="utf-8")
    notify = (HERE / "_notify.py").read_text(encoding="utf-8")
    links = [
        ("_notify.py writes the flag", 'fh.write("notified=true' in notify),
        ("the review job has a step to write it", re.search(r"^\s+id: notify\s*$", wf, re.M)),
        ("the review job exports it",
         re.search(r"^\s+notified: \$\{\{ steps\.notify\.outputs\.notified \}\}", wf, re.M)),
        ("the report job reads it",
         re.search(r"^\s+NOTIFIED: \$\{\{ needs\.review\.outputs\.notified \}\}", wf, re.M)),
    ]
    broken = [name for name, found in links if not found]
    if broken:
        print("\u274c the notified handoff is broken at: %s" % "; ".join(broken))
        return 1
    print("\u2705 notified handoff intact: _notify.py -> step -> job output -> report")
    return 0


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


def check_publish_needs_post_flag():
    """_publish.py's only closure against posting with excess write access is its missing
    `--post` flag: on the self-hosted box it falls back to ~/.git-credentials, a human PAT
    with repo write access, and posts over plain curl -- so GITHUB_TOKEN permission scoping
    does not stop it from posting at all. Today the workflow grants the review job only
    `contents: read`, so that fallback never has write access to use. If the workflow is ever
    changed to grant `issues: write` or `pull-requests: write` anywhere, GITHUB_TOKEN stops
    being irrelevant to _publish.py's reach, and the invocation MUST pass --post explicitly
    -- or there is a silent gap between what the token in scope can do and what the one call
    site that would use it was actually told to do."""
    wf = WORKFLOW.read_text(encoding="utf-8")
    run_one = (HERE / "run_one.sh").read_text(encoding="utf-8")
    granted = re.search(r"^\s*(issues|pull-requests):\s*write\s*$", wf, re.M)
    invocations = re.findall(r"[^\n]*_publish\.py[^\n]*", wf) \
        + re.findall(r"[^\n]*_publish\.py[^\n]*", run_one)
    has_post = any("--post" in line for line in invocations)
    if granted and not has_post:
        print(f"\u274c aiter-review-bot.yml grants '{granted.group(0).strip()}' somewhere, "
              f"but _publish.py is never invoked with --post -- GITHUB_TOKEN scoping is no "
              f"longer the only thing standing between a job and a post, and _publish.py's "
              f"missing-flag default (dry-run) must now be an explicit match, not an "
              f"accident")
        return 1
    if granted:
        print("\u2705 aiter-review-bot.yml grants write, and _publish.py is invoked with "
              "--post to match it")
        return 0
    print("\u2705 aiter-review-bot.yml grants no issues/pull-requests write anywhere; "
          "_publish.py's missing --post is its only closure, as documented")
    return 0


def check_publish_review_is_advisory():
    """The card goes up as a PR review, and a review carries an `event`. COMMENT leaves the
    bot advisory; APPROVE or REQUEST_CHANGES would make it a merge gate -- which is a
    different thing from what the workflow's `contents: read` was reasoned about, and a
    change no one would see in a diff that only touches a JSON literal. Pin the event, and
    pin that the one POST in the file is the review call."""
    src = (HERE / "_publish.py").read_text(encoding="utf-8")
    bad = 0
    for verdict in ("APPROVE", "REQUEST_CHANGES"):
        if f'"{verdict}"' in src or f"'{verdict}'" in src:
            print(f"\u274c _publish.py names {verdict}; this bot is advisory and must only ever "
                  f"send event COMMENT -- a review that gates merge changes what the "
                  f"workflow's permissions mean")
            bad += 1
    posts = re.findall(r'_api\(\s*f?"([^"]+)"\s*,\s*method="POST"', src)
    if posts != ["pulls/{pr}/reviews"]:
        print(f"\u274c _publish.py's POST call sites are {posts}; expected exactly one, "
              f"pulls/{{pr}}/reviews -- the card is published as a review with inline "
              f"comments, not as a top-level issue comment")
        bad += 1
    if '"event": "COMMENT"' not in src:
        print("\u274c _publish.py does not set event COMMENT on the review it builds")
        bad += 1
    if not bad:
        print("\u2705 _publish.py posts exactly one review, event COMMENT, never APPROVE/"
              "REQUEST_CHANGES")
    return bad


def check_anchoring_has_a_citation_check():
    """_publish.py anchors a finding inline only when check_citations.py vouched for its
    citation; with no citations.txt it anchors nothing and the whole card falls back into
    the summary. That degradation is silent and looks identical to a card whose findings
    simply cite nothing in the diff. So the two halves must stay wired: run_one.sh has to
    run the checker, and _collect.py has to carry its verdict into the report dir
    _publish.py reads."""
    run_one = (HERE / "run_one.sh").read_text(encoding="utf-8")
    lib = (HERE / "_lib.py").read_text(encoding="utf-8")
    broken = []
    if not (HERE / "check_citations.py").is_file():
        broken.append("check_citations.py is missing")
    if "check_citations.py" not in run_one:
        broken.append("run_one.sh never runs check_citations.py")
    if 'tee "$W/citations.txt"' not in run_one:
        broken.append("run_one.sh does not write citations.txt")
    if "citations.txt" not in lib:
        broken.append("_lib.py does not collect citations.txt into the report dir")
    if broken:
        print("\u274c inline anchoring is wired to nothing: %s -- every finding would "
              "silently fall back to the summary" % "; ".join(broken))
        return 1
    print("\u2705 anchoring wired: run_one.sh -> citations.txt -> _lib.py OPTIONAL -> _publish.py")
    return 0


def check_range_anchor_shape():
    """A span anchor is a second way to fail an all-or-nothing call, and it is invisible in
    a diff that only edits a dict literal. GitHub spells a span as start_line..line with a
    side for each end; it refuses the whole review -- summary and every other comment with
    it -- if start_line is not strictly below line, if its side is missing, or if the span
    reaches outside the hunk. So build a review over a synthetic diff whose file has two
    hunks, and assert the shape of what comes out rather than the source that produced it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_publish", HERE / "_publish.py")
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)
    diff = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -10,2 +10,4 @@\n ten\n+eleven\n+twelve\n thirteen\n"
        "@@ -50,1 +50,2 @@\n fifty\n+fiftyone\n"
    )
    card = (
        "## card\n\n"
        "\u26a0\ufe0f x.py:11-13 is a span inside one hunk.\n"
        "\U0001f534 x.py:11-51 is a span across the gap between two hunks.\n"
        "\U0001f4dd x.py:900-910 is a span the diff never reaches.\n"
    )
    cs = pub.build_review("1", card, diff, set(), "a" * 40)["comments"]
    bad = []
    for c in cs:
        if "start_line" in c:
            if not c["start_line"] < c["line"]:
                bad.append(f"start_line {c['start_line']} is not below line {c['line']}")
            if c.get("start_side") != c.get("side"):
                bad.append(f"span on {c['path']} names side {c.get('side')!r} but "
                           f"start_side {c.get('start_side')!r}")
        elif "start_side" in c:
            bad.append(f"{c['path']}:{c['line']} carries start_side with no start_line")
    spans = [c for c in cs if "start_line" in c]
    if len(spans) != 1 or (spans and (spans[0]["start_line"], spans[0]["line"]) != (11, 13)):
        bad.append(f"expected exactly the one in-hunk span 11-13, got "
                   f"{[(c['start_line'], c['line']) for c in spans]} -- a span across the "
                   f"gap between hunks is refused like any line outside the diff")
    if bad:
        print("\u274c _publish.py builds a span anchor GitHub will not take: %s"
              % "; ".join(bad))
        return 1
    print("\u2705 span anchors well formed: start_line < line, matching sides, inside one hunk")
    return 0


def check_unanchorable_citations_are_linked():
    """The other half of the same change: a citation GitHub will not take a comment on gets
    a blob permalink in the summary instead of bare `path:line` text. It is pinned to the
    head SHA -- a branch link drifts off the code it was written about, which is a wrong
    anchor arriving late. And it is gated exactly as anchoring is, because a link to the
    wrong code is worse than no link."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_publish", HERE / "_publish.py")
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)
    diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -10,2 +10,3 @@\n ten\n+eleven\n twelve\n"
    card = "## card\n\n\U0001f4dd x.py:900 is context this PR does not touch.\n"
    sha = "b" * 40
    want = f"/blob/{sha}/x.py#L900"
    bad = []
    body = pub.build_review("1", card, diff, set(), sha)["body"]
    if want not in body:
        bad.append("an out-of-diff citation got no permalink pinned to the head SHA")
    if "x.py:900 is context" not in body:
        bad.append("the human-readable citation was replaced rather than added to")
    # flagged -> no link, and no checker at all -> no link
    if "/blob/" in pub.build_review("1", card, diff, {"x.py:900"}, sha)["body"]:
        bad.append("a citation check_citations.py flagged was still linked")
    if "/blob/" in pub.build_review("1", card, diff, None, sha)["body"]:
        bad.append("a card the checker never ran on was still linked")
    if bad:
        print("\u274c _publish.py's permalinks are not gated as anchors are: %s" % "; ".join(bad))
        return 1
    print("\u2705 out-of-diff citations linked to the head SHA, and only when the checker vouched")
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
    bad += check_no_self_trigger()
    bad += check_notified_handoff()
    bad += check_runner_label_exclusive()
    bad += check_publish_needs_post_flag()
    bad += check_publish_review_is_advisory()
    bad += check_anchoring_has_a_citation_check()
    bad += check_range_anchor_shape()
    bad += check_unanchorable_citations_are_linked()
    print(f"{'OK' if bad == 0 else 'DRIFT'}: {bad} drift(s)")
    return bad


if __name__ == "__main__":
    sys.exit(main())

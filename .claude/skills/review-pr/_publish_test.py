"""What must not regress in _publish.py.

Three things, and the first is the important one:

  1. The --post closure. _publish.py falls back to ~/.git-credentials on the self-hosted box,
     a human PAT with repo write access, so GITHUB_TOKEN scoping does not stop it from posting
     at all -- the missing --post flag is the only thing standing between a decision and a real
     post. Dry-run must reach no POST. And the review it posts must be event COMMENT: APPROVE
     or REQUEST_CHANGES would turn an advisory bot into a merge gate.
  2. The anchoring rule. A finding goes inline only if its cited line is inside the diff AND
     check_citations.py vouched for every citation it carries. An inline comment on the wrong
     line lands in the author's working view, where being wrong costs the most.
  3. The degrade-to-summary fallback. A review is all-or-nothing -- one anchor GitHub will not
     take fails the whole call, summary included, which is worse than the plain issue comment
     this replaced and which aiter's bot posts today. On a refusal the same review must go out
     again with the anchors folded into the body, not quietly shed.

Nothing here touches the network: _api is replaced in every test that could reach it.

Run:  python3 -m pytest .claude/skills/review-pr/_publish_test.py -q
"""

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _publish

DIFF = """diff --git a/aiter/ops/shuffle.py b/aiter/ops/shuffle.py
--- a/aiter/ops/shuffle.py
+++ b/aiter/ops/shuffle.py
@@ -20,6 +20,7 @@
     moe_shuffle_scale,
 )
+from aiter.ops.topk import _FLYDSL_USE_HERD
 from torch import nn
-gone = 1
 trailing = 2
diff --git a/docs/notes.md b/docs/notes.md
--- a/docs/notes.md
+++ b/docs/notes.md
@@ -100,2 +100,3 @@
 context
+added
 after
"""

CARD = """## aiter PR #9999 — a card

**One-sentence summary that the review body does not need.**

Review (advisory): ⚠️ NEEDS WORK
Validation (deterministic): N/A — no runtime surface changed
Perf (advisory): NOT RUN — no idle GPU on this box

⚠️ [verified] aiter/ops/shuffle.py:23 imports `_FLYDSL_USE_HERD` — **Author must** pin it.

📝 [verified] `elsewhere.py:900` is nowhere near this diff; it is context code.

🔴 shuffle.py:22 and helper.py:7 — one of these citations was flagged.

_Independent refutation: 3 judged, 0 killed, 3 defended — cross-family._
"""


@pytest.fixture
def report(tmp_path, monkeypatch):
    d = tmp_path / "PR-9999"
    d.mkdir()
    (d / "card.md").write_text(CARD, encoding="utf-8")
    (d / "pr.diff").write_text(DIFF, encoding="utf-8")
    (d / "citations.txt").write_text(
        "CITATIONS NOTE: helper.py:7 -- none of ['helper'] appears anywhere in this file\n"
        "CITATIONS: every cited file:line resolves (1 proximity note(s))\n",
        encoding="utf-8",
    )
    (d / "pr_meta.json").write_text('{"headRefOid": "deadbeef"}', encoding="utf-8")
    monkeypatch.setattr(_publish, "REPORTS", tmp_path)
    return d


# ------------------------------------------------------------ the --post closure


def test_dry_run_reaches_no_post(report, monkeypatch, capsys):
    calls = []

    def fake_api(path, method="GET", body=None):
        calls.append((method, path))
        return [] if method == "GET" else {"html_url": "http://x"}

    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999"])
    assert not [c for c in calls if c[0] == "POST"], calls
    assert "DRY-RUN" in capsys.readouterr().out


def test_payload_flag_alone_still_posts_nothing(report, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        _publish,
        "_api",
        lambda path, method="GET", body=None: calls.append((method, path)) or [],
    )
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--payload"])
    assert not [c for c in calls if c[0] == "POST"], calls
    # and it printed something a human can read as the literal request body
    assert '"event": "COMMENT"' in capsys.readouterr().out


def test_post_goes_to_the_reviews_endpoint_as_a_comment(report, monkeypatch):
    sent = {}

    def fake_api(path, method="GET", body=None):
        if method == "POST":
            sent["path"], sent["body"] = path, body
            return {"html_url": "http://x"}
        return []

    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--post"])
    assert sent["path"] == "pulls/9999/reviews"
    assert sent["body"]["event"] == "COMMENT"
    assert sent["body"]["commit_id"] == "deadbeef"
    assert all(c["side"] == "RIGHT" for c in sent["body"]["comments"])


def test_no_source_line_can_make_it_approve_or_request_changes():
    src = (HERE / "_publish.py").read_text(encoding="utf-8")
    for verdict in ("APPROVE", "REQUEST_CHANGES"):
        assert f'"{verdict}"' not in src and f"'{verdict}'" not in src


# ------------------------------------------------------------ anchoring


def test_commentable_counts_right_side_lines_only():
    c = _publish.commentable(DIFF)
    assert c["aiter/ops/shuffle.py"] == {20, 21, 22, 23, 24}
    assert c["docs/notes.md"] == {100, 101, 102}


def test_anchors_only_what_the_diff_contains(report, monkeypatch):
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    assert [(c["path"], c["line"]) for c in pl["comments"]] == [
        ("aiter/ops/shuffle.py", 23)
    ]
    # the context-code finding stays in the summary, in full
    assert "elsewhere.py:900" in pl["body"]


def test_a_flagged_citation_forfeits_the_inline_slot(report, monkeypatch):
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    # shuffle.py:22 IS in the diff, but the same finding also cites helper.py:7, which
    # check_citations flagged -- so the finding is not anchored at all.
    assert not any(c["line"] == 22 for c in pl["comments"])
    assert "helper.py:7" in pl["body"]


def test_no_citation_check_means_nothing_is_anchored(report, monkeypatch):
    (report / "citations.txt").unlink()
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    assert pl["comments"] == []
    assert _publish.flagged_citations("9999") is None


def test_summary_indexes_every_anchored_finding(report, monkeypatch):
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    # an invisible finding must not read like a finding that was never made
    assert "`aiter/ops/shuffle.py:23`" in pl["body"]
    assert "1 finding(s) posted inline" in pl["body"]


def test_summary_carries_every_status_line_aiter_cards_have(report, monkeypatch):
    """aiter's card writes `Review (advisory):` where ATOM's writes `Review:`, and adds a
    Perf line ATOM has no counterpart for. Matching ATOM's literal prefixes would drop all
    three, leaving a summary with no verdict in it at all."""
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    body = _publish.review_payload("9999")["body"]
    assert "Review (advisory): ⚠️ NEEDS WORK" in body
    assert "Validation (deterministic): N/A — no runtime surface changed" in body
    assert "Perf (advisory): NOT RUN — no idle GPU on this box" in body
    assert "aiter-bot review pr=9999 head=deadbeef" in body


def test_card_md_is_not_rewritten(report, monkeypatch):
    before = (report / "card.md").read_text(encoding="utf-8")
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.review_payload("9999")
    assert (report / "card.md").read_text(encoding="utf-8") == before


def test_dedup_marker_is_looked_for_in_reviews_too(report, monkeypatch):
    seen = []

    def fake_api(path, method="GET", body=None):
        seen.append(path)
        if path.startswith("pulls/9999/reviews"):
            return [{"body": "x <!-- aiter-bot review pr=9999 head=abc -->"}]
        return []

    monkeypatch.setattr(_publish, "_api", fake_api)
    assert _publish.already_posted("9999", "abc") is True
    assert any(p.startswith("pulls/9999/reviews") for p in seen)


def test_dedup_still_sees_the_issue_comments_the_bot_already_posted(report, monkeypatch):
    """aiter's bot is live and its markers are sitting in issue comments on real PRs. A
    dedup that only read the reviews endpoint would repost on top of every one of them."""
    seen = []

    def fake_api(path, method="GET", body=None):
        seen.append(path)
        if path.startswith("issues/9999/comments"):
            return [{"body": "card <!-- aiter-bot review pr=9999 head=abc -->"}]
        return []

    monkeypatch.setattr(_publish, "_api", fake_api)
    assert _publish.already_posted("9999", "abc") is True
    assert any(p.startswith("issues/9999/comments") for p in seen)


def test_an_overridden_bot_name_does_not_move_the_dedup_marker(report, monkeypatch):
    """BOT is env-overridable; the marker must not be, or a box with AITER_BOT_NAME set
    reposts over every review the real bot already published."""
    monkeypatch.setattr(_publish, "BOT", "someone-elses-bot")
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    assert "aiter-bot review pr=9999" in _publish.review_payload("9999")["body"]


# ------------------------------------------------------------ degrade to summary


def _refusing_api(reject_anchored=True):
    """A fake API that refuses any review carrying comments[], as GitHub does when one
    anchor names a line it will not accept."""
    sent = []

    def fake_api(path, method="GET", body=None):
        if method != "POST":
            return []
        sent.append(body)
        if reject_anchored and body.get("comments"):
            return {"message": "pull_request_review_thread.line must be part of the diff"}
        return {"html_url": "http://x/review"}

    return fake_api, sent


def test_a_refused_anchor_retries_the_same_review_without_anchors(report, monkeypatch, capsys):
    fake_api, sent = _refusing_api()
    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--post"])
    assert len(sent) == 2, "the refusal must be retried, not given up on"
    assert sent[0]["comments"] and sent[1]["comments"] == []
    out = capsys.readouterr().out
    assert "ANCHORS REFUSED -- folded into summary" in out
    assert "✅" in out


def test_the_degraded_review_still_carries_the_findings_it_could_not_anchor(report, monkeypatch):
    """The whole point: a review that silently sheds half its findings reads exactly like a
    review that only had the other half."""
    fake_api, sent = _refusing_api()
    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--post"])
    folded = sent[1]["body"]
    assert "_FLYDSL_USE_HERD" in folded, "the anchored finding vanished from the summary"
    assert "aiter/ops/shuffle.py:23" in folded
    # and the findings that were never anchored are still there too
    assert "elsewhere.py:900" in folded


def test_a_review_with_no_anchors_is_not_retried_twice(report, monkeypatch):
    """Nothing to degrade to: retrying an identical call just doubles a failure."""
    (report / "citations.txt").unlink()  # -> nothing anchors
    sent = []

    def fake_api(path, method="GET", body=None):
        if method != "POST":
            return []
        sent.append(body)
        return {"message": "some other failure"}

    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--post"])
    assert len(sent) == 1

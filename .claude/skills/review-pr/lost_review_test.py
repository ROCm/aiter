#!/usr/bin/env python3
"""Drive lost_review.py against a fake GitHub.

Same shape as the other two: the fake replaces only _get/_post, and the source is read from
lost_review.py rather than copied, so these cannot keep passing after it changes underneath.

Usage: python3 .claude/skills/review-pr/lost_review_test.py
"""

import contextlib
import io
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "lost_review.py"
REPO = "ROCm/aiter"


def load(src):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    mod = types.ModuleType("lost_review_under_test")
    mod.__file__ = str(SRC)
    exec(compile(src, str(SRC), "exec"), mod.__dict__)
    return mod


def drive(mod, scn):
    rec = {"posts": [], "notices": [], "warnings": [], "errors": [], "rc": None}

    class Fake(mod.GitHub):
        def _get(self, path):
            raise AssertionError("this script should never read: %s" % path)

        def _post(self, path, body, token):
            rec["posts"].append({"path": path, "body": body["body"], "token": token})
            return scn.get("post_status", 201)

    env = {"REVIEW_RESULT": "failure", "NOTIFIED": "", "PR": "42",
           "GITHUB_REPOSITORY": REPO, "OWNER_OVERRIDE": "",
           "RUN_URL": "https://github.com/ROCm/aiter/actions/runs/1"}
    env.update(scn.get("env", {}))

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rec["rc"] = mod.run(Fake("gh-token", "bot-token"), env)
    for line in out.getvalue().splitlines():
        for tag, key in (("::notice::", "notices"), ("::warning::", "warnings"),
                         ("::error::", "errors")):
            if line.startswith(tag):
                rec[key].append(line[len(tag):])
    return rec


CASES = [
    ("reports a review job that died without a word",
     {},
     lambda r: [
         (len(r["posts"]) == 1, "posted %d comments, want 1" % len(r["posts"])),
         (len(r["posts"]) == 1 and r["posts"][0]["path"] == "/repos/ROCm/aiter/issues/42/comments",
          "posted to the wrong path"),
         (len(r["posts"]) == 1 and r["posts"][0]["token"] == "bot-token",
          "did not post as aiter-bot"),
         (len(r["posts"]) == 1 and "@zufayu" in r["posts"][0]["body"], "paged nobody"),
         (r["rc"] == 1 and r["errors"], "did not fail the run"),
     ]),

    # The review job reports its own failures; speaking again would double-page the owner.
    ("stays quiet when the review job already reported",
     {"env": {"NOTIFIED": "true"}},
     lambda r: [
         (not r["posts"], "posted a second notice over the review job's own"),
         (r["rc"] == 0, "failed the run for a failure that was already reported"),
     ]),

    ("stays quiet after a review that succeeded",
     {"env": {"REVIEW_RESULT": "success"}},
     lambda r: [
         (not r["posts"], "reported a successful review as lost"),
         (r["rc"] == 0, "failed the run after a good review"),
     ]),

    ("stays quiet when the review never ran",
     {"env": {"REVIEW_RESULT": "skipped"}},
     lambda r: [
         (not r["posts"], "reported a review that was never meant to run"),
         (r["rc"] == 0, "failed the run for a skipped review"),
     ]),

    # A cancelled job is the shape a 24 h queue expiry leaves behind, and it reports nothing either.
    ("reports a cancelled review too",
     {"env": {"REVIEW_RESULT": "cancelled"}},
     lambda r: [
         (len(r["posts"]) == 1 and "cancelled" in r["posts"][0]["body"],
          "did not name what happened"),
         (r["rc"] == 1, "let a cancelled review pass as fine"),
     ]),

    ("names the override owner when the repo sets one",
     {"env": {"OWNER_OVERRIDE": "  gyohuangxin  "}},
     lambda r: [
         (len(r["posts"]) == 1 and "@gyohuangxin" in r["posts"][0]["body"], "ignored the override"),
         (len(r["posts"]) == 1 and "@zufayu" not in r["posts"][0]["body"], "paged the default too"),
     ]),

    ("still fails the run when the notice cannot be posted",
     {"post_status": 403},
     lambda r: [
         (any("403" in m for m in r["warnings"]), "swallowed the failed post"),
         (r["rc"] == 1, "a token problem would have hidden a lost review entirely"),
     ]),
]

MUTANTS = [
    ("the already-reported check",
     'if (env.get("NOTIFIED") or "").strip().lower() == "true":', "if False:",
     "stays quiet when the review job already reported"),
    ("the good-outcome check",
     'if result in ("", "success", "skipped"):', "if False:",
     "stays quiet after a review that succeeded"),
    ("failing the run",
     'print("::error::the review job ended as %s with nothing reported" % result)\n    return 1',
     "return 0",
     "reports a review job that died without a word"),
    ("the warning when the post is rejected",
     'print("::warning::could not post', 'print("::notice::could not post',
     "still fails the run when the notice cannot be posted"),
    ("the owner override",
     '(env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"', '"zufayu"',
     "names the override owner when the repo sets one"),
]


def suite(src):
    mod = load(src)
    out = {}
    for name, scn, check in CASES:
        try:
            bad = [why for ok, why in check(drive(mod, scn)) if not ok]
            out[name] = "; ".join(bad) if bad else None
        except Exception as e:                                  # noqa: BLE001 - report, not raise
            out[name] = "threw: %s: %s" % (type(e).__name__, e)
    return out


def main():
    src = SRC.read_text(encoding="utf-8")
    ok = bad = 0

    def t(name, err):
        nonlocal ok, bad
        if err:
            print("  ❌ %s — %s" % (name, err)); bad += 1
        else:
            print("  ✅ %s" % name); ok += 1

    print("[lost review behaviour]")
    for name, err in suite(src).items():
        t(name, err)

    print("[lost review guards bite]")
    for why, find, repl, breaks in MUTANTS:
        if find not in src:
            t("breaking %s is caught" % why, "lost_review.py no longer contains `%s`" % find)
            continue
        t('breaking %s turns "%s" red' % (why, breaks),
          None if suite(src.replace(find, repl, 1))[breaks]
          else "it stayed green, so that check proves nothing")

    print("=== %d green / %d red ===" % (ok, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Say so when a review ended without saying anything.

Everything that reports a failed review runs inside the review job: run_one.sh writes the status
and _notify.py posts it. `if: always()` covers a step that failed, but not a job that stopped
existing -- if the self-hosted runner loses contact mid-review the job is marked failed and not
one of its steps runs, so nothing is written and nothing is posted. From the PR it looks exactly
like a review that is still going, forever.

This runs on hosted infra after the review job, whatever became of it, and speaks only when the
review job ended badly AND left no sign that it had already reported. It is the same idea as the
watchdog, moved to the other end: the watchdog covers a review that never started, this covers
one that started and then disappeared.

Usage (from the report job): python3 .claude/skills/review-pr/lost_review.py
Reads REVIEW_RESULT, NOTIFIED, PR, RUN_URL, GH_TOKEN, AITER_BOT_TOKEN, OWNER_OVERRIDE and the
GITHUB_* that Actions sets.
"""

import os
import sys

from watchdog import GitHub      # sibling; sys.path[0] is this script's directory


def body(result, owner, run_url):
    return (
        "⚠️ **aiter-bot** — the review job ended as `%s` without posting anything, "
        "so there is no review and no failure report for this PR.\n\n"
        "Everything that reports a failure runs inside that job, which means a job that stops "
        "existing — the runner losing contact mid-review is the usual cause — cannot "
        "report itself. Nothing is wrong with this PR, and nothing was changed on it.\n\n"
        "@%s — please check the runner.%s\n\n"
        "<sub>Re-comment `@aiter-bot review` to try again. See "
        "`.claude/skills/review-pr/RUNNER-SETUP.md` section 5.</sub>"
        % (result, owner, ("\n\nRun: " + run_url) if run_url else ""))


def run(api, env):
    result = (env.get("REVIEW_RESULT") or "").strip()
    # The review job reports its own failures; only speak when it could not.
    if (env.get("NOTIFIED") or "").strip().lower() == "true":
        print("::notice::the review job already reported; staying quiet")
        return 0
    if result in ("", "success", "skipped"):
        print("::notice::review result %r needs no report" % result)
        return 0

    owner = (env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"
    status = api.comment(env["GITHUB_REPOSITORY"], env["PR"],
                         body(result, owner, (env.get("RUN_URL") or "").strip()))
    if not 200 <= status < 300:
        print("::warning::could not post the lost-review notice: %s" % status)
    print("::error::the review job ended as %s with nothing reported" % result)
    return 1


def main():
    api = GitHub(os.environ["GH_TOKEN"], os.environ.get("AITER_BOT_TOKEN", ""))
    return run(api, os.environ)


if __name__ == "__main__":
    sys.exit(main())

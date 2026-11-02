#!/usr/bin/env python3
"""Report a review that never got a runner.

This is the one failure the in-job notifier cannot reach. With no runner the review job stays
`queued`: run_one.sh never starts, so _notify.py never runs, GitHub sends no failure mail, and
the run reads as "in progress" indefinitely -- this box sat dead for days that way. The job
that runs this script lives on GitHub-hosted infra, so it is alive exactly when the box is not.

Its two silences carry as much weight as its alarm. A false page here would cost exactly what
the mis-triaged GLM pages cost, so it stays quiet when the review has started and when the
single runner is merely busy with another one.

`issue_comment` workflows only ever run from the default branch, so GitHub will not execute
this on a PR branch -- watchdog_test.py drives it instead, against a fake API and clock.

Usage (from the workflow): python3 .claude/skills/review-pr/watchdog.py
Reads STUCK_MINUTES, PR, GH_TOKEN, AITER_BOT_TOKEN, OWNER_OVERRIDE, and the GITHUB_* that
Actions sets for every step.
"""

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

WORKFLOW = "aiter-review-bot.yml"
POLL_SECONDS = 120
PAGE = 100


class GitHub:
    """The three calls this needs, with no dependency to install on the hosted runner.

    Transport sits behind _get/_post on purpose: the test substitutes those two and everything
    above them -- paging, URLs, which token posts the comment -- is the shipping code.
    """

    API = "https://api.github.com"

    def __init__(self, token, bot_token=""):
        self._token = token
        self._bot_token = bot_token or token

    def _request(self, path, method="GET", body=None, token=None):
        req = urllib.request.Request(
            self.API + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"authorization": "token %s" % (token or self._token),
                     "accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.load(r) if r.status != 204 else None
        except urllib.error.HTTPError as e:
            return e.code, None

    def _get(self, path):
        status, data = self._request(path)
        if status >= 400:
            raise RuntimeError("GET %s -> %s" % (path, status))
        return data

    def _post(self, path, body, token):
        status, _ = self._request(path, method="POST", body=body, token=token)
        return status

    def jobs(self, repo, run_id):
        return self._get("/repos/%s/actions/runs/%s/jobs?per_page=%d" % (repo, run_id, PAGE))["jobs"]

    def in_progress_runs(self, repo):
        """Every in_progress run of this workflow, not just the first page.

        A batch of reviews queues up as a long list of in_progress runs and the API returns the
        newest first -- so the one actually holding the runner is last, and a single page would
        miss exactly it.
        """
        out, page = [], 1
        while True:
            batch = self._get(
                "/repos/%s/actions/workflows/%s/runs?status=in_progress&per_page=%d&page=%d"
                % (repo, WORKFLOW, PAGE, page))["workflow_runs"]
            out += batch
            if len(batch) < PAGE:
                return out
            page += 1

    def comment(self, repo, pr, body):
        # Post as aiter-bot, like every other message this workflow sends.
        return self._post("/repos/%s/issues/%s/comments" % (repo, pr),
                          {"body": body}, self._bot_token)


def box_busy(api, repo, own_run_id):
    """Is the box working on some other review? The single runner serialises them, so waiting
    behind one is normal and must not page anyone."""
    for run in api.in_progress_runs(repo):
        if run["id"] == own_run_id:
            continue
        if any(j["name"] == "review" and j["status"] == "in_progress"
               for j in api.jobs(repo, run["id"])):
            return True
    return False


def notice(msg):
    print("::notice::%s" % msg)


def warning(msg):
    print("::warning::%s" % msg)


def failed(msg):
    print("::error::%s" % msg)
    return 1


def stuck_body(mins, owner):
    return (
        "⚠️ **aiter-bot** — no self-hosted runner claimed this review in "
        "%s min, so it has not started. Nothing is wrong with this PR.\n\n"
        "A queued job never fails on its own, so this notice is the only signal: the runner "
        "process is probably not running on the box — see "
        "`.claude/skills/review-pr/RUNNER-SETUP.md` section 5.\n\n"
        "@%s — please check the runner.\n\n"
        "<sub>A queued job expires after 24 h. Re-comment `@aiter-bot review` once it is "
        "back.</sub>" % (mins, owner))


def run(api, env, sleep=time.sleep, clock=time.monotonic):
    raw = env.get("STUCK_MINUTES", "")
    try:
        mins = float(raw)
    except ValueError:
        mins = float("nan")
    # A typo here makes the deadline expire before the first poll, so the loop never runs and
    # the alarm fires at once -- paging the runner owner about a box that is fine. That is the
    # false page this job exists to prevent, so fail as the configuration error it is.
    if not math.isfinite(mins) or mins <= 0:
        return failed("STUCK_MINUTES is not a usable number of minutes: %r" % raw)

    repo = env["GITHUB_REPOSITORY"]
    own_run_id = int(env["GITHUB_RUN_ID"])
    deadline = clock() + mins * 60

    # Poll rather than sleep the whole window: a watchdog that returns as soon as the review is
    # under way does not hold a hosted runner, which matters when a batch of reviews queues up
    # and each one carries its own watchdog.
    saw_review = False
    while clock() < deadline:
        sleep(POLL_SECONDS)
        review = next((j for j in api.jobs(repo, own_run_id) if j["name"] == "review"), None)
        if review is not None:
            saw_review = True
            if review["status"] != "queued":
                return 0                                    # ours started

    # The job this watchdog exists to watch was never in its own run's job list -- a rename, or
    # a matrix that suffixes job names. Treating that as "ours started" would report success
    # forever while watching nothing, so name the problem; it is a workflow-structure fault and
    # not a reason to page anyone about the box.
    if not saw_review:
        return failed("this run has no job named 'review', so the watchdog is watching nothing")

    # Ask about the box only now. Asking inside the loop and returning on the first yes abandons
    # the PR: the runner can die a minute later with nobody left watching, and a queued job
    # never fails on its own.
    if box_busy(api, repo, own_run_id):
        notice("the runner is busy with another review; queueing is expected")
        return 0

    # Still queued, with no review running anywhere: nothing is listening for this label.
    owner = (env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"
    shown = int(mins) if float(mins).is_integer() else mins
    status = api.comment(repo, env["PR"], stuck_body(shown, owner))
    if not 200 <= status < 300:
        warning("could not post the stuck-review notice: %s" % status)
    return failed("no runner claimed this review in %s min" % shown)


def main():
    api = GitHub(os.environ["GH_TOKEN"], os.environ.get("AITER_BOT_TOKEN", ""))
    return run(api, os.environ)


if __name__ == "__main__":
    sys.exit(main())

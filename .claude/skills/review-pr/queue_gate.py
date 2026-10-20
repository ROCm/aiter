#!/usr/bin/env python3
"""Decide whether to accept a review request, and tell the requester what to expect.

The box runs one review at a time and each takes the better part of an hour, so requests queue.
A queued GitHub job is cancelled after 24 h, and the notifier lives inside the job that never
ran -- so a request accepted into a queue deeper than the day can simply vanish, in silence,
from the point of view of the person who asked.

The fix is not to detect that afterwards but to stop accepting work that cannot be done: this
runs in the gate, on the hosted runner, before anything reaches the box. It answers the
requester, on their own PR, at the moment they ask -- rather than paging the runner's owner
some time later.

It also catches a runner that is not taking work at all. Nothing queued is moving and something
has been waiting a while means the box is dead or wedged; the two are indistinguishable from
here and have the same remedy, so the message does not pretend to tell them apart.

Usage (from the gate job): python3 .claude/skills/review-pr/queue_gate.py
Reads AUTHORIZED, PR, GH_TOKEN, AITER_BOT_TOKEN, OWNER_OVERRIDE, the tuning below, and the
GITHUB_* that Actions sets for every step. Writes `ok` to GITHUB_OUTPUT.
"""

import datetime
import math
import os
import sys
import time

# watchdog.py sits next to this file; sys.path[0] is the script's own directory, so this picks
# up the sibling rather than any installed package of the same name.
from watchdog import GitHub

AVG_REVIEW_MINUTES = 50      # measured: a full review is about this
MAX_WAIT_HOURS = 8           # refuse beyond this rather than accept it; see below
NO_RUNNER_MINUTES = 30       # nothing moving for this long, with work waiting, means nobody is taking it
RECHECK_SECONDS = 60         # before calling a box dead, see whether it was simply between reviews

# The limit is a wait, not a queue length, because the constraint it answers is a wait: GitHub
# cancels a job that has been queued for 24 h, silently. Eight hours leaves room for the estimate
# to be wrong by a factor of three and still not lose the request.


def _age_minutes(stamp, now):
    t = datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)
    return (now - t).total_seconds() / 60.0


def tunable(env, name, default):
    """Read a repo var that is a number, or fall back loudly.

    These arrive from `vars.*`, which is a text box: a typo, a stray unit, or a well-meant `0`
    all reach here as a string. The watchdog learned this the hard way with STUCK_MINUTES, where
    an unusable value made the deadline expire before the first poll and paged the runner's owner
    about a healthy box. An unusable value must never be more drastic than no value at all.
    """
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not math.isfinite(value) or value <= 0:
        print("::warning::%s=%r is not a usable number; using %s" % (name, raw, default))
        return default
    return value


def survey(api, repo, own_run_id, now):
    """How many reviews are running, and how long the waiting ones have waited."""
    running, waiting = 0, []
    for run in api.in_progress_runs(repo):
        if run["id"] == own_run_id:
            continue
        review = next((j for j in api.jobs(repo, run["id"]) if j["name"] == "review"), None)
        if review is None:
            continue
        if review["status"] == "in_progress":
            running += 1
        elif review["status"] == "queued":
            waiting.append(_age_minutes(run["created_at"], now))
    return running, waiting


def decide(running, waiting, cfg):
    """Pure: (running count, list of waiting ages in minutes) -> (verdict, ahead, eta_minutes).

    Verdicts: 'idle' (start now), 'queued' (start in eta), 'too-deep' (refuse), 'no-runner'.
    """
    ahead = running + len(waiting)
    if running == 0 and waiting and max(waiting) >= cfg["no_runner_minutes"]:
        return "no-runner", ahead, int(max(waiting))
    eta = ahead * cfg["avg_minutes"]
    if eta >= cfg["max_wait_minutes"]:
        return "too-deep", ahead, eta
    if ahead == 0:
        return "idle", 0, 0
    return "queued", ahead, eta


def _hours(minutes):
    return "%.0f min" % minutes if minutes < 90 else "%.1f h" % (minutes / 60.0)


def message(verdict, ahead, eta, owner):
    if verdict == "no-runner":
        return (
            "⚠️ **aiter-bot** — %d review(s) have been waiting %s and none has "
            "started, so the self-hosted runner is not picking up work. It is either down or "
            "wedged; both need the same thing, a look at the box.\n\n"
            "This request was **not queued**, because a job queued behind a stopped runner is "
            "cancelled after 24 h without a word.\n\n"
            "@%s — please check the runner, then re-comment `@aiter-bot review`.\n\n"
            "<sub>See `.claude/skills/review-pr/RUNNER-SETUP.md` section 5.</sub>"
            % (ahead, _hours(eta), owner))
    if verdict == "too-deep":
        return (
            "**aiter-bot** — the box reviews one PR at a time and is **%d deep** right now "
            "(roughly %s of work ahead).\n\n"
            "This request was **not queued**: a job that waits more than 24 h is cancelled "
            "silently, so saying no now is better than losing it later.\n\n"
            "Re-comment `@aiter-bot review` once the queue is shorter."
            % (ahead, _hours(eta)))
    return ("**aiter-bot** — queued behind %d review(s); expect to start in about %s. "
            "The box reviews one PR at a time." % (ahead, _hours(eta)))


def run(api, env, now=None, sleep=time.sleep):
    """Returns (ok, verdict). Posts at most one comment."""
    out = env.get("GITHUB_OUTPUT")

    def finish(ok, verdict):
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write("ok=%s\n" % ("true" if ok else "false"))
        print("::notice::queue gate: %s -> %s" % (verdict, "run" if ok else "not queued"))
        return ok, verdict

    # The authorization step owns that decision; this one only ever narrows it.
    if (env.get("AUTHORIZED") or "").strip().lower() != "true":
        return finish(False, "unauthorized")

    # Authorization fails closed; everything from here does not. Refusing every review because
    # some part of the queue check broke would turn one bug in this file into an outage of the
    # whole bot -- a worse failure than the one it guards against, which is a single request lost
    # in a long queue. So the whole of it degrades, not just the API call: a missing variable, an
    # unparseable one, a network error on the way out.
    try:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        cfg = {"avg_minutes": tunable(env, "AVG_REVIEW_MINUTES", AVG_REVIEW_MINUTES),
               "max_wait_minutes": tunable(env, "MAX_WAIT_HOURS", MAX_WAIT_HOURS) * 60,
               "no_runner_minutes": tunable(env, "NO_RUNNER_MINUTES", NO_RUNNER_MINUTES)}
        repo = env["GITHUB_REPOSITORY"]
        own = int(env["GITHUB_RUN_ID"])
        running, waiting = survey(api, repo, own, now)
        verdict, ahead, eta = decide(running, waiting, cfg)

        # One snapshot cannot tell a dead runner from the seconds between one review finishing
        # and the next being picked up: in both, nothing is running and something has been
        # waiting a while -- the waiting is long because it sat behind the review that just
        # ended, not because nobody is working. Look again before calling a healthy box dead. A
        # handoff closes in seconds; an outage does not.
        if verdict == "no-runner":
            pause = tunable(env, "RECHECK_SECONDS", RECHECK_SECONDS)
            sleep(pause)
            later = now + datetime.timedelta(seconds=pause)
            running, waiting = survey(api, repo, own, later)
            verdict, ahead, eta = decide(running, waiting, cfg)
            if verdict != "no-runner":
                print("::notice::the box was between reviews, not stopped")

        if verdict == "idle":
            return finish(True, verdict)    # starts at once; a comment would be noise

        owner = (env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"
        status = api.comment(repo, env["PR"], message(verdict, ahead, eta, owner))
        if not 200 <= status < 300:
            print("::warning::could not post the queue notice: %s" % status)
        return finish(verdict == "queued", verdict)
    except Exception as e:                                  # noqa: BLE001 - degrade, not crash
        print("::warning::the queue check failed (%s: %s); proceeding" % (type(e).__name__, e))
        return finish(True, "unmeasured")


def main():
    api = GitHub(os.environ["GH_TOKEN"], os.environ.get("AITER_BOT_TOKEN", ""))
    ok, _ = run(api, os.environ)
    return 0 if ok is not None else 1      # the gate's output carries the decision, not the exit


if __name__ == "__main__":
    sys.exit(main())

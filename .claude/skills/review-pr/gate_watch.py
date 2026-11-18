#!/usr/bin/env python3
"""Page when a review request sits at run-level `queued` with nothing to say for it.

The gate job runs on GitHub-hosted runners. When those starve -- org quota spent, or a long
hosted queue -- the run starts no job at all: no gate, no review, no watchdog, no report, no
mail. The request just sits `queued`, and every guard that could speak lives inside the very
run that cannot start. This workflow is the one watcher outside the runs it watches, and it
pages by failing its own job: a failed scheduled run notifies the workflow author, with zero
write permissions and no bot token anywhere in the file.

A `queued` run is not itself the verdict: a request past its gate reads `queued` too while
it waits for the single box. The gate job's state tells them apart -- past its gate, the
request is the box's queue and the in-run watchdog's to watch; short of it, nothing in the
run can ever start, which is the silence this watcher exists to break.

Exactly once per stuck request, with no state to keep: the cron fires every WINDOW_MINUTES
and a request pages only while its age is inside [STUCK_MINUTES, STUCK_MINUTES +
WINDOW_MINUTES) -- one poll wide. A late or skipped poll can miss a window entirely; that is
the silence that exists today, and it beats paging about the same stuck request every fifteen
minutes. STUCK_MINUTES and WINDOW_MINUTES are constants, not env: the window must stay one
poll wide, and an env override could drift from the cron without any guard seeing it.

Usage (from the workflow): python3 .claude/skills/review-pr/gate_watch.py
Reads GH_TOKEN, OWNER_OVERRIDE, and the GITHUB_* that Actions sets for every step.
"""

import datetime
import os
import sys

from watchdog import GitHub, WORKFLOW, PAGE, notice, failed

STUCK_MINUTES = 30    # a queued run this old is not waiting for anything but a hosted runner
WINDOW_MINUTES = 15   # one poll wide; must equal the cron interval in review-bot-gate-watch.yml


def _age_minutes(created_at, now):
    """Minutes since the run was created -- a run sits `queued` from creation until its gate
    job starts, so this is how long it has waited for a hosted runner."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (now - datetime.datetime.strptime(created_at, fmt)).total_seconds() / 60.0


def queued_runs(api, repo):
    """Every run of the review workflow the API still reports `queued`, not just the first
    page. `queued` is ambiguous -- a request past its gate reads `queued` too while its
    review waits for the single box -- so the caller tells the two apart by each run's gate
    job. The API returns the newest first, so under a burst the run stuck longest is on the
    last page."""
    out, page = [], 1
    while True:
        batch = api._get("/repos/%s/actions/workflows/%s/runs?status=queued&per_page=%d&page=%d"
                         % (repo, WORKFLOW, PAGE, page))["workflow_runs"]
        out += batch
        if len(batch) < PAGE:
            return out
        page += 1


def run(api, env, now):
    repo = env["GITHUB_REPOSITORY"]
    try:
        queued = queued_runs(api, repo)
        stuck = []
        for r in queued:
            age = _age_minutes(r["created_at"], now)
            if not STUCK_MINUTES <= age < STUCK_MINUTES + WINDOW_MINUTES:
                continue
            # `queued` alone cannot tell a starved gate from a request past its gate and
            # waiting for the single box -- the API reports both as `queued`. The gate
            # job's state can: past its gate, the request is the box's queue and the
            # in-run watchdog's to watch; short of it, nothing in the run can ever start.
            jobs = api.jobs(repo, r["id"])
            gate = next((j for j in jobs if j["name"] == "gate"), None)
            if gate is not None and gate["status"] == "completed":
                continue
            stuck.append((r, age))
    except Exception as e:                              # noqa: BLE001 - the watcher itself broke
        return failed("cannot read the queued runs: %s: %s" % (type(e).__name__, e))
    if not stuck:
        if queued:
            notice("%d run(s) queued, oldest %.0f min -- nothing newly stuck"
                   % (len(queued), max(_age_minutes(r["created_at"], now) for r in queued)))
        return 0
    owner = (env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"
    msg = ("%d review request(s) have waited up to %.0f min for a hosted runner -- the gate "
           "job never started, so nothing in those runs can speak. @%s -- check the "
           "hosted-runner quota." % (len(stuck), max(a for _, a in stuck), owner))
    for r, _ in stuck:
        print(r["html_url"])
    return failed(msg)


def main():
    api = GitHub(os.environ["GH_TOKEN"])
    # GitHub timestamps are naive UTC; match them rather than mix aware with naive.
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    return run(api, os.environ, now)


if __name__ == "__main__":
    sys.exit(main())
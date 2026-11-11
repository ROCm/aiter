#!/usr/bin/env python3
"""Drive queue_gate.py against a fake GitHub and a fixed clock.

Same shape as watchdog_test.py: the fake replaces only _get/_post, so paging, URLs and which
token posts the comment are the shipping code; the source is read from queue_gate.py rather
than copied, so these checks cannot keep passing after it changes underneath them.

Usage: python3 .claude/skills/review-pr/queue_gate_test.py
"""

import contextlib
import datetime
import os
import io
import sys
import tempfile
import types
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "queue_gate.py"
OWN_RUN = 111
REPO = "ROCm/aiter"
NOW = datetime.datetime(2026, 10, 9, 12, 0, 0, tzinfo=datetime.timezone.utc)


def load(src):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))        # queue_gate imports its sibling watchdog
    mod = types.ModuleType("queue_gate_under_test")
    mod.__file__ = str(SRC)
    exec(compile(src, str(SRC), "exec"), mod.__dict__)
    return mod


def stamp(minutes_ago):
    return (NOW - datetime.timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def drive(mod, scn):
    rec = {"posts": [], "notices": [], "warnings": [], "gets": 0, "ok": None,
           "verdict": None, "slept": 0, "passes": 0}

    class Fake(mod.GitHub):
        def _get(self, path):
            rec["gets"] += 1
            u = urllib.parse.urlparse(path)
            q = urllib.parse.parse_qs(u.query)
            if u.path.endswith("/jobs"):
                rid = int(u.path.split("/runs/")[1].split("/")[0])
                return {"jobs": scn.get("jobs", {}).get(rid, [])}
            if "/runs" in u.path:
                status = q["status"][0]
                if status == "success":             # completed reviews, read for the ETA median
                    runs = scn.get("done", [])      # the API filters these server-side
                    if runs == "boom":              # the history being unreadable, not empty
                        raise RuntimeError("GET %s -> 500" % path)
                else:
                    runs = scn.get("runs", [])
                    if runs == "boom":              # the API being unreadable, not empty
                        raise RuntimeError("GET %s -> 500" % path)
                    if callable(runs):              # the box changing between the two looks
                        if int(q["page"][0]) == 1:
                            rec["passes"] += 1
                        runs = runs(rec["passes"])
                    runs = [r for r in runs if r.get("status") == status]
                per, page = int(q["per_page"][0]), int(q.get("page", ["1"])[0])
                newest_first = sorted(runs, key=lambda r: -r["id"])
                return {"workflow_runs": newest_first[(page - 1) * per:page * per]}
            raise AssertionError("unexpected GET %s" % path)

        def _post(self, path, body, token):
            if scn.get("post_raises"):              # a network error on the way out, not a 4xx
                raise RuntimeError("connection reset")
            rec["posts"].append({"path": path, "body": body["body"], "token": token})
            return scn.get("post_status", 201)

    with tempfile.NamedTemporaryFile("w+", suffix=".out", delete=False) as fh:
        out_path = fh.name
    env = {"AUTHORIZED": "true", "PR": "42", "GITHUB_REPOSITORY": REPO,
           "GITHUB_RUN_ID": str(OWN_RUN), "OWNER_OVERRIDE": "", "GITHUB_OUTPUT": out_path}
    env.update(scn.get("env", {}))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rec["ok"], rec["verdict"] = mod.run(
            Fake("gh-token", "bot-token"), env, now=NOW,
            sleep=lambda s: rec.__setitem__("slept", rec["slept"] + s))
    for line in buf.getvalue().splitlines():
        if line.startswith("::notice::"):
            rec["notices"].append(line[10:])
        elif line.startswith("::warning::"):
            rec["warnings"].append(line[11:])
    rec["output_file"] = Path(out_path).read_text(encoding="utf-8")
    Path(out_path).unlink(missing_ok=True)
    return rec


GATE_DONE = [{"name": "gate", "status": "completed", "conclusion": "success"}]
GATE_QUEUED = [{"name": "gate", "status": "queued", "conclusion": None}]


def busy(n_running, n_waiting, waited=5, n_arriving=0):
    """n runs being served, n waiting at the box, n_arriving still queued at their own gate.

    A waiting review is a request past its gate and held for the box; an arriving one has not
    got that far -- a burst of requests queues whole runs at the hosted runner before any gate
    job is even assigned, and those requests are as ahead of the requester as any.
    """
    runs, jobs = [], {}
    rid = 200
    for _ in range(n_running):
        runs.append({"id": rid, "status": "in_progress", "created_at": stamp(waited + 50)})
        jobs[rid] = GATE_DONE + [{"name": "review", "status": "in_progress"}]
        rid += 1
    for _ in range(n_waiting):
        runs.append({"id": rid, "status": "in_progress", "created_at": stamp(waited)})
        jobs[rid] = GATE_DONE + [{"name": "review", "status": "queued"}]
        rid += 1
    for _ in range(n_arriving):
        runs.append({"id": rid, "status": "queued", "created_at": stamp(waited)})
        jobs[rid] = GATE_QUEUED + [{"name": "review", "status": "queued"}]
        rid += 1
    return {"runs": runs, "jobs": jobs}


def finished(minutes):
    """Completed runs whose review job took the given minutes, for the ETA to be priced by."""
    runs, jobs = [], {}
    rid = 500
    for m in minutes:
        runs.append({"id": rid, "status": "completed", "created_at": stamp(120)})
        jobs[rid] = [{"name": "review", "status": "completed", "conclusion": "success",
                      "started_at": stamp(120), "completed_at": stamp(120 - m)}]
        rid += 1
    return runs, jobs


def with_done(scn, minutes, refused=False):
    """A queue scene that also carries the completed reviews the ETA is priced by."""
    done_runs, done_jobs = finished(minutes)
    if refused:
        # a request the gate refused: the run concludes successfully with no review to time
        done_runs = [{"id": 600, "status": "completed", "created_at": stamp(130)}] + done_runs
        done_jobs = {600: [{"name": "review", "status": "completed", "conclusion": "skipped"}],
                     **done_jobs}
    out = dict(scn)
    out["done"] = done_runs
    out["jobs"] = {**scn.get("jobs", {}), **done_jobs}
    return out


CASES = [
    ("starts at once and says nothing when the box is free",
     busy(0, 0),
     lambda r: [
         (r["ok"] is True, "refused a request with an idle box"),
         (not r["posts"], "commented on a review that starts immediately"),
         ("ok=true" in r["output_file"], "did not tell the workflow to proceed"),
     ]),

    ("tells the requester where they are in the queue",
     busy(1, 2),
     lambda r: [
         (r["ok"] is True, "refused a request the box can still take"),
         (len(r["posts"]) == 1, "posted %d comments, want 1" % len(r["posts"])),
         (len(r["posts"]) == 1 and "queued behind 3" in r["posts"][0]["body"],
          "did not say how many are ahead: %s" % (r["posts"][0]["body"][:80] if r["posts"] else "")),
         (len(r["posts"]) == 1 and "2.5 h" in r["posts"][0]["body"], "did not give an ETA"),
     ]),

    # Ten requests fired together queued eight whole runs at the hosted runner before any gate
    # could run, and every gate that did read 'queued behind 1' -- the requests it could not see
    # were exactly the ones ahead of the requester.
    ("counts requests queued at the gate, not just at the box",
     busy(1, 0, n_arriving=2),
     lambda r: [
         (r["ok"] is True, "refused a request the box can still take"),
         (len(r["posts"]) == 1 and "queued behind 3" in r["posts"][0]["body"],
          "did not count the requests still queued at their own gate: %s"
          % (r["posts"][0]["body"][:80] if r["posts"] else "")),
     ]),

    # A request stuck at the hosted runner says nothing about the box: its gate job never ran,
    # so paging the runner's owner over it would send someone to a healthy box, and refusing
    # it would drop a valid request over a queue the box never saw.
    ("does not page anyone because requests are stuck at the gate",
     busy(0, 0, waited=45, n_arriving=3),
     lambda r: [
         (r["verdict"] != "no-runner", "called a starved gate a dead runner (%r)" % r["verdict"]),
         (r["ok"] is True, "dropped a request over a queue it cannot even see"),
         (not any("@zufayu" in p["body"] for p in r["posts"]),
          "paged the owner over hosted-runner starvation"),
     ]),

    # The constant said 50 min a review; the box has been doing them in about 20. Every notice
    # carried an estimate 2.5x high, and the wait limit fired 2.5x early.
    ("prices the queue by what the box actually did lately",
     with_done(busy(1, 2), [17, 18, 19, 20, 22, 26, 33], refused=True),
     lambda r: [
         (r["ok"] is True, "refused a request the box can still take"),
         (len(r["posts"]) == 1 and "60 min" in r["posts"][0]["body"],
          "did not price the queue by the measured median: %s"
          % (r["posts"][0]["body"][:80] if r["posts"] else "")),
         (len(r["posts"]) == 1 and "2.5 h" not in r["posts"][0]["body"], "priced it by the constant"),
     ]),

    ("keeps the constant when no review has ever finished",
     with_done(busy(1, 2), []),
     lambda r: [
         (r["ok"] is True, "refused a request over an unreadable history"),
         (len(r["posts"]) == 1 and "2.5 h" in r["posts"][0]["body"],
          "did not fall back to the constant: %s"
          % (r["posts"][0]["body"][:80] if r["posts"] else "")),
     ]),

    # The ETA is a courtesy: a history that cannot be read must not stop the review.
    ("still answers while the ETA cannot be measured",
     dict(with_done(busy(1, 2), [20]), done="boom"),
     lambda r: [
         (r["ok"] is True, "a failure to read past reviews stopped an authorized review"),
         (len(r["posts"]) == 1 and "2.5 h" in r["posts"][0]["body"],
          "did not fall back to the constant: %s"
          % (r["posts"][0]["body"][:80] if r["posts"] else "")),
         (any("recent review times" in w for w in r["warnings"]),
          "degraded without naming it: %s" % r["warnings"]),
     ]),

    # A queue deeper than the day is the thing that loses requests: GitHub cancels a job that
    # waits 24 h, and the notifier lives inside the job that never ran.
    ("refuses rather than accept a queue it cannot work through",
     busy(1, 9),
     lambda r: [
         (r["ok"] is False, "accepted a request into a queue it cannot work through"),
         ("ok=false" in r["output_file"], "told the workflow to proceed anyway"),
         (len(r["posts"]) == 1 and "not queued" in r["posts"][0]["body"],
          "did not tell the requester it was refused"),
         (len(r["posts"]) == 1 and "10 deep" in r["posts"][0]["body"], "did not say how deep"),
     ]),

    # Nothing moving with work waiting: dead or wedged, indistinguishable from here, same remedy.
    ("refuses and pages the owner when nothing is being served",
     busy(0, 3, waited=45),
     lambda r: [
         (r["ok"] is False, "queued a review behind a runner that is not working"),
         (len(r["posts"]) == 1 and "@zufayu" in r["posts"][0]["body"], "did not page the owner"),
         (len(r["posts"]) == 1 and "not queued" in r["posts"][0]["body"],
          "did not say the request was dropped"),
         (r["verdict"] == "no-runner", "verdict was %r" % r["verdict"]),
     ]),

    # One snapshot cannot tell a dead runner from the seconds between one review ending and the
    # next being picked up -- in both, nothing is running and something has been waiting a long
    # time, because it was waiting behind the review that just finished.
    ("does not call a box dead while it is between reviews",
     {"runs": lambda n: ([{"id": 200, "status": "in_progress", "created_at": stamp(55)}] if n == 1
                         else [{"id": 200, "status": "in_progress", "created_at": stamp(55)},
                               {"id": 201, "status": "in_progress", "created_at": stamp(105)}]),
      "jobs": {200: GATE_DONE + [{"name": "review", "status": "queued"}],
               201: GATE_DONE + [{"name": "review", "status": "in_progress"}]}},
     lambda r: [
         (r["verdict"] != "no-runner", "called a healthy box dead during a handoff"),
         (r["ok"] is True, "refused a review because of a handoff gap"),
         (not any("@zufayu" in p["body"] for p in r["posts"]), "paged the owner over a handoff"),
         (r["slept"] > 0, "decided without looking again"),
     ]),

    # Waiting is normal while someone is being served; only a stalled queue means trouble.
    ("does not cry wolf while a review is actually running",
     busy(1, 3, waited=45),
     lambda r: [
         (r["verdict"] != "no-runner", "called a working box dead"),
         (not any("@zufayu" in p["body"] for p in r["posts"]), "paged the owner over normal queueing"),
     ]),

    ("does not count its own run",
     {"runs": [{"id": OWN_RUN, "status": "in_progress", "created_at": stamp(1)}],
      "jobs": {OWN_RUN: GATE_DONE + [{"name": "review", "status": "queued"}]}},
     lambda r: [
         (r["verdict"] == "idle", "counted itself as someone else's review (%r)" % r["verdict"]),
         (not r["posts"], "commented because it saw itself waiting"),
     ]),

    ("leaves an unauthorized request alone",
     dict(busy(0, 0), env={"AUTHORIZED": "false"}),
     lambda r: [
         (r["ok"] is False, "let an unauthorized request through"),
         (not r["posts"], "replied to an unauthorized comment"),
         (r["gets"] == 0, "called the API for a request it had already refused"),
     ]),

    ("pages the override owner when the repo sets one",
     dict(busy(0, 3, waited=45), env={"OWNER_OVERRIDE": "  gyohuangxin  "}),
     lambda r: [
         (len(r["posts"]) == 1 and "@gyohuangxin" in r["posts"][0]["body"], "ignored the override"),
         (len(r["posts"]) == 1 and "@zufayu" not in r["posts"][0]["body"], "paged the default too"),
     ]),

    # A bug here must not become an outage: the queue check is a courtesy, authorization is not.
    ("lets an authorized review through when the queue cannot be measured",
     {"runs": "boom"},
     lambda r: [
         (r["ok"] is True, "a failure to measure the queue stopped an authorized review"),
         (r["verdict"] == "unmeasured", "verdict was %r" % r["verdict"]),
         (any("RuntimeError" in m for m in r["warnings"]),
          "degraded without naming the cause: %s" % r["warnings"]),
         (not r["posts"], "commented about a queue it could not read"),
     ]),

    # A repo var is a text box. An unusable value in it must never be more drastic than no value
    # at all -- the watchdog learned this with STUCK_MINUTES, where one made it page about a
    # healthy box. Here a bad one could refuse every review, or stop the bot outright.
    ("falls back to the default when a tuning value is unusable",
     dict(busy(0, 0), env={"MAX_WAIT_HOURS": "abc"}),
     lambda r: [
         (r["ok"] is True, "a typo in a repo variable stopped an authorized review"),
         (r["verdict"] == "idle", "verdict was %r, not the normal one" % r["verdict"]),
         (any("MAX_WAIT_HOURS" in m for m in r["warnings"]), "swapped in a default in silence"),
     ]),

    ("refuses nothing when a tuning value is zero",
     dict(busy(0, 0), env={"MAX_WAIT_HOURS": "0"}),
     lambda r: [
         (r["ok"] is True, "a zero in a repo variable refused a review on an idle box"),
         (not r["posts"], "told the requester an idle box was too deep"),
     ]),

    # The degrade has to cover the whole check, not just the measurement -- this one blows up
    # after the queue has been read, on the way out.
    ("lets the review through when posting the notice blows up",
     dict(busy(1, 2), post_raises=True),
     lambda r: [
         (r["ok"] is True, "an error after the queue was read stopped an authorized review"),
         (r["verdict"] == "unmeasured", "verdict was %r" % r["verdict"]),
     ]),

    ("still decides when the comment cannot be posted",
     dict(busy(1, 9), env={}, post_status=403),
     lambda r: [
         (any("403" in m for m in r["warnings"]), "swallowed the failed post"),
         (r["ok"] is False, "a token problem silently turned a refusal into an acceptance"),
     ]),
]

MUTANTS = [
    ("the stalled-queue check",
     'if running == 0 and waiting and max(waiting) >= cfg["no_runner_minutes"]:', "if False:",
     "refuses and pages the owner when nothing is being served"),
    ("looking twice before calling a box dead",
     'if verdict == "no-runner":\n            pause = tunable(env, "RECHECK_SECONDS", RECHECK_SECONDS)',
     'if False:\n            pause = tunable(env, "RECHECK_SECONDS", RECHECK_SECONDS)',
     "does not call a box dead while it is between reviews"),
    ("the wait limit",
     'if eta >= cfg["max_wait_minutes"]:', "if False:",
     "refuses rather than accept a queue it cannot work through"),
    ("skipping its own run",
     'if run["id"] == own_run_id:', "if False:",
     "does not count its own run"),
    ("the authorization short-circuit",
     'if (env.get("AUTHORIZED") or "").strip().lower() != "true":', "if False:",
     "leaves an unauthorized request alone"),
    ("staying quiet when the box is free",
     "if ahead == 0:", "if False:",
     "starts at once and says nothing when the box is free"),
    ("the degrade-on-error path",
     "except Exception as e:                                  # noqa: BLE001 - degrade, not crash",
     "except ZeroDivisionError as e:",
     "lets an authorized review through when the queue cannot be measured"),
    ("the fallback for an unusable tuning value",
     "if not math.isfinite(value) or value <= 0:", "if False:",
     "falls back to the default when a tuning value is unusable"),
    ("the owner override",
     '(env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"', '"zufayu"',
     "pages the override owner when the repo sets one"),
    ("counting a running review as work in progress",
     'if review is not None and review["status"] == "in_progress":', "if False:",
     "does not cry wolf while a review is actually running"),
    ("counting requests still queued at their own gate",
     "arriving += 1", "pass",
     "counts requests queued at the gate, not just at the box"),
    ("telling a starved gate from a dead box",
     'gate["status"] == "completed" and gate["conclusion"] == "success"', "True",
     "does not page anyone because requests are stuck at the gate"),
    ("pricing the queue by measured reviews",
     "typical = statistics.median(done) if done else AVG_REVIEW_MINUTES",
     "typical = AVG_REVIEW_MINUTES",
     "prices the queue by what the box actually did lately"),
    ("the ETA degrade path",
     "except Exception as e:  # the ETA is a courtesy, not a gate",
     "except ZeroDivisionError as e:  # the ETA is a courtesy, not a gate",
     "still answers while the ETA cannot be measured"),
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


def entry_point(src):
    """See watchdog_test.py: main() is where the environment variable names actually land."""
    mod = load(src)
    mod.GitHub = type("Fake", (mod.GitHub,), {
        "_get": lambda self, path: {"workflow_runs": []},
        "_post": lambda self, *a, **k: 201})
    saved = dict(os.environ)
    os.environ.update({"GH_TOKEN": "t", "AITER_BOT_TOKEN": "b", "PR": "42",
                       "GITHUB_REPOSITORY": REPO, "GITHUB_RUN_ID": str(OWN_RUN),
                       "AUTHORIZED": "false"})
    try:
        with contextlib.redirect_stdout(io.StringIO()):     # its workflow commands are not ours
            rc = mod.main()
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return [("main() runs and returns a status", rc == 0)]


def main():
    src = SRC.read_text(encoding="utf-8")
    ok = bad = 0

    def t(name, err):
        nonlocal ok, bad
        if err:
            print("  ❌ %s — %s" % (name, err)); bad += 1
        else:
            print("  ✅ %s" % name); ok += 1

    print("[queue gate behaviour]")
    for name, err in suite(src).items():
        t(name, err)

    print("[queue gate guards bite]")
    for why, find, repl, breaks in MUTANTS:
        if find not in src:
            t("breaking %s is caught" % why, "queue_gate.py no longer contains `%s`" % find)
            continue
        t('breaking %s turns "%s" red' % (why, breaks),
          None if suite(src.replace(find, repl, 1))[breaks]
          else "it stayed green, so that check proves nothing")

    print("[entry point]")
    try:
        for name, good in entry_point(src):
            t(name, None if good else "main() did not wire up")
    except Exception as e:                                  # noqa: BLE001 - report, not raise
        t("main() runs", "threw: %s: %s" % (type(e).__name__, e))

    print("=== %d green / %d red ===" % (ok, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

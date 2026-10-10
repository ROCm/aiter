#!/usr/bin/env python3
"""Drive queue_gate.py against a fake GitHub and a fixed clock.

Same shape as watchdog_test.py: the fake replaces only _get/_post, so paging, URLs and which
token posts the comment are the shipping code; the source is read from queue_gate.py rather
than copied, so these checks cannot keep passing after it changes underneath them.

Usage: python3 .claude/skills/review-pr/queue_gate_test.py
"""

import contextlib
import datetime
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
    rec = {"posts": [], "notices": [], "warnings": [], "gets": 0, "ok": None, "verdict": None}

    class Fake(mod.GitHub):
        def _get(self, path):
            rec["gets"] += 1
            u = urllib.parse.urlparse(path)
            q = urllib.parse.parse_qs(u.query)
            if u.path.endswith("/jobs"):
                rid = int(u.path.split("/runs/")[1].split("/")[0])
                return {"jobs": scn.get("jobs", {}).get(rid, [])}
            if "/runs" in u.path:
                runs = scn.get("runs", [])
                if runs == "boom":                  # the API being unreadable, not empty
                    raise RuntimeError("GET %s -> 500" % path)
                per, page = int(q["per_page"][0]), int(q["page"][0])
                newest_first = sorted(runs, key=lambda r: -r["id"])
                return {"workflow_runs": newest_first[(page - 1) * per:page * per]}
            raise AssertionError("unexpected GET %s" % path)

        def _post(self, path, body, token):
            rec["posts"].append({"path": path, "body": body["body"], "token": token})
            return scn.get("post_status", 201)

    with tempfile.NamedTemporaryFile("w+", suffix=".out", delete=False) as fh:
        out_path = fh.name
    env = {"AUTHORIZED": "true", "PR": "42", "GITHUB_REPOSITORY": REPO,
           "GITHUB_RUN_ID": str(OWN_RUN), "OWNER_OVERRIDE": "", "GITHUB_OUTPUT": out_path}
    env.update(scn.get("env", {}))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rec["ok"], rec["verdict"] = mod.run(Fake("gh-token", "bot-token"), env, now=NOW)
    for line in buf.getvalue().splitlines():
        if line.startswith("::notice::"):
            rec["notices"].append(line[10:])
        elif line.startswith("::warning::"):
            rec["warnings"].append(line[11:])
    rec["output_file"] = Path(out_path).read_text(encoding="utf-8")
    Path(out_path).unlink(missing_ok=True)
    return rec


REVIEW_RUNNING = [{"name": "review", "status": "in_progress"}]
REVIEW_QUEUED = [{"name": "review", "status": "queued"}]


def busy(n_running, n_waiting, waited=5):
    """n runs being served, n waiting, all other runs older than ours."""
    runs, jobs = [], {}
    rid = 200
    for _ in range(n_running):
        runs.append({"id": rid, "created_at": stamp(waited + 50)}); jobs[rid] = REVIEW_RUNNING
        rid += 1
    for _ in range(n_waiting):
        runs.append({"id": rid, "created_at": stamp(waited)}); jobs[rid] = REVIEW_QUEUED
        rid += 1
    return {"runs": runs, "jobs": jobs}


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

    # Waiting is normal while someone is being served; only a stalled queue means trouble.
    ("does not cry wolf while a review is actually running",
     busy(1, 3, waited=45),
     lambda r: [
         (r["verdict"] != "no-runner", "called a working box dead"),
         (not any("@zufayu" in p["body"] for p in r["posts"]), "paged the owner over normal queueing"),
     ]),

    ("does not count its own run",
     {"runs": [{"id": OWN_RUN, "created_at": stamp(1)}],
      "jobs": {OWN_RUN: REVIEW_QUEUED}},
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
         (any("could not measure" in m for m in r["warnings"]), "degraded without saying so"),
         (not r["posts"], "commented about a queue it could not read"),
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
    ("the owner override",
     '(env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"', '"zufayu"',
     "pages the override owner when the repo sets one"),
    ("counting a running review as work in progress",
     'if review["status"] == "in_progress":', "if False:",
     "does not cry wolf while a review is actually running"),
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

    print("=== %d green / %d red ===" % (ok, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

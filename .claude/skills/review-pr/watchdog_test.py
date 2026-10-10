#!/usr/bin/env python3
"""Drive watchdog.py against a fake GitHub and a fake clock.

The watchdog is the one guard that cannot be checked by running it: `issue_comment` workflows
only ever run from the default branch, so GitHub does not execute that job on a PR branch at
all. It would first run after merge, against a real stuck review -- the worst place to find out
it is wrong.

The fake replaces only _get/_post, so paging, URL construction and which token posts the comment
are all the shipping code. The source is read from watchdog.py rather than copied here, so these
checks cannot keep passing after it has changed underneath them.

Usage: python3 .claude/skills/review-pr/watchdog_test.py
"""

import contextlib
import io
import sys
import types
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "watchdog.py"
OWN_RUN = 111
REPO = "ROCm/aiter"


def load(src):
    mod = types.ModuleType("watchdog_under_test")
    mod.__file__ = str(SRC)
    exec(compile(src, str(SRC), "exec"), mod.__dict__)
    return mod


def drive(mod, scn):
    """Run watchdog.run() once; return everything it did."""
    rec = {"polls": 0, "posts": [], "notices": [], "warnings": [], "errors": [], "rc": None}

    class Fake(mod.GitHub):
        def _get(self, path):
            u = urllib.parse.urlparse(path)
            q = urllib.parse.parse_qs(u.query)
            if u.path.endswith("/jobs"):
                run_id = int(u.path.split("/runs/")[1].split("/")[0])
                if run_id != OWN_RUN:
                    return {"jobs": scn.get("other_jobs", {}).get(run_id, [])}
                rec["polls"] += 1
                # A watchdog that never returns would hang this test instead of failing it.
                assert rec["polls"] <= 60, "polled past any sane deadline"
                st = scn["self_status"](rec["polls"])
                return {"jobs": [] if st is None else [{"name": "review", "status": st}]}
            if "/runs" in u.path:
                live = scn.get("live_runs", [])
                live = live(rec["polls"]) if callable(live) else live
                per, page = int(q["per_page"][0]), int(q["page"][0])
                # Model what the real endpoint does: newest first, capped at per_page. A fake
                # that always returned everything would hide a paging bug entirely.
                newest_first = sorted(live, key=lambda r: -r["id"])
                return {"workflow_runs": newest_first[(page - 1) * per:page * per]}
            raise AssertionError("unexpected GET %s" % path)

        def _post(self, path, body, token):
            rec["posts"].append({"path": path, "body": body["body"], "token": token})
            return scn.get("post_status", 201)

    env = {"STUCK_MINUTES": "20", "PR": "42", "GITHUB_REPOSITORY": REPO,
           "GITHUB_RUN_ID": str(OWN_RUN), "OWNER_OVERRIDE": ""}
    env.update(scn.get("env", {}))

    clock = [1000.0]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rec["rc"] = mod.run(Fake("gh-token", "bot-token"), env,
                            sleep=lambda s: clock.__setitem__(0, clock[0] + s),
                            clock=lambda: clock[0])
    for line in out.getvalue().splitlines():
        for tag, key in (("::notice::", "notices"), ("::warning::", "warnings"),
                         ("::error::", "errors")):
            if line.startswith(tag):
                rec[key].append(line[len(tag):])
    return rec


QUEUED_FOREVER = lambda _n: "queued"
BUSY_ELSEWHERE = {"live_runs": [{"id": 222}],
                  "other_jobs": {222: [{"name": "review", "status": "in_progress"}]}}

CASES = [
    ("alarms when no runner ever claims the review",
     {"self_status": QUEUED_FOREVER},
     lambda r: [
         (len(r["posts"]) == 1, "posted %d notices, want 1" % len(r["posts"])),
         (len(r["posts"]) == 1 and r["posts"][0]["path"] == "/repos/ROCm/aiter/issues/42/comments",
          "posted to the wrong path"),
         (len(r["posts"]) == 1 and "no self-hosted runner claimed this review in 20 min"
          in r["posts"][0]["body"], "the notice does not say what happened"),
         (len(r["posts"]) == 1 and r["posts"][0]["token"] == "bot-token",
          "did not post as aiter-bot"),
         (r["rc"] == 1 and r["errors"], "the job did not fail, so the alarm is only a comment"),
     ]),

    ("stays silent once our own review starts",
     {"self_status": lambda n: "queued" if n == 1 else "in_progress"},
     lambda r: [
         (not r["posts"], "paged someone about a review that had already started"),
         (r["rc"] == 0, "failed the job for a review that had already started"),
         (r["polls"] == 2, "kept polling after ours started (%d polls)" % r["polls"]),
     ]),

    ("stays silent while the box is busy with another review",
     dict(self_status=QUEUED_FOREVER, **BUSY_ELSEWHERE),
     lambda r: [
         (not r["posts"], "paged someone for normal queueing behind the single runner"),
         (r["rc"] == 0, "failed the job for normal queueing behind the single runner"),
         (any("busy" in m for m in r["notices"]), "left no trace of why it went quiet"),
         (r["polls"] > 1, "stopped watching the moment it saw the box busy"),
     ]),

    # Seeing another review is not a reason to stop watching: the runner can die a minute later,
    # and then this PR sits `queued` with nobody left alive to notice.
    ("alarms when the busy runner dies mid-wait",
     {"self_status": QUEUED_FOREVER,
      "live_runs": lambda n: [{"id": 222}] if n <= 2 else [],
      "other_jobs": {222: [{"name": "review", "status": "in_progress"}]}},
     lambda r: [
         (len(r["posts"]) == 1, "went quiet for a runner that died while it was watching"),
         (r["rc"] == 1, "did not fail the run after the runner disappeared"),
     ]),

    # The watchdog finds its subject by job name. If that name stops matching, a missing job
    # must not read as "ours started" -- that is a monitor reporting green because it has gone
    # blind.
    ("fails loudly when its own run has no review job",
     {"self_status": lambda _n: None},
     lambda r: [
         (not r["posts"], "paged the runner owner about a workflow-structure problem"),
         (r["rc"] == 1, "reported success while watching nothing at all"),
         (any("review" in m for m in r["errors"]),
          "did not say what it could not find: %s" % r["errors"]),
     ]),

    # The single runner serialises reviews, so a batch piles up as a long list of in_progress
    # runs -- and the one actually holding the runner is the OLDEST, returned last.
    ("finds the busy runner under a large backlog",
     {"self_status": QUEUED_FOREVER,
      "live_runs": [{"id": 1000 + i} for i in range(120)],
      "other_jobs": {1000: [{"name": "review", "status": "in_progress"}]}},
     lambda r: [
         (not r["posts"], "paged someone although a review was running the whole time"),
         (any("busy" in m for m in r["notices"]), "did not recognise the box as busy"),
     ]),

    ("pages the override owner when the repo sets one",
     {"self_status": QUEUED_FOREVER, "env": {"OWNER_OVERRIDE": "  gyohuangxin  "}},
     lambda r: [
         (len(r["posts"]) == 1 and "@gyohuangxin" in r["posts"][0]["body"], "ignored the override"),
         (len(r["posts"]) == 1 and "@zufayu" not in r["posts"][0]["body"],
          "paged the default owner too"),
     ]),

    # A typo makes the deadline expire before the first poll, firing the alarm on the spot --
    # the false page this job exists to prevent, caused by the workflow rather than the box.
    ("does not page anyone when STUCK_MINUTES is unusable",
     {"self_status": QUEUED_FOREVER, "env": {"STUCK_MINUTES": ""}},
     lambda r: [
         (not r["posts"], "paged someone because of a workflow typo, not a dead runner"),
         (r["rc"] == 1, "a watchdog that cannot run must not pass for a healthy one"),
         (any("STUCK_MINUTES" in m for m in r["errors"]),
          "blamed the runner for a workflow typo: %s" % r["errors"]),
     ]),

    ("still fails the job when the notice cannot be posted",
     {"self_status": QUEUED_FOREVER, "post_status": 403},
     lambda r: [
         (any("403" in m for m in r["warnings"]), "swallowed the failed post"),
         (r["rc"] == 1, "a token problem would have hidden a dead runner entirely"),
     ]),
]

# Every check above passes against code that does nothing in the path it claims to guard, unless
# it is shown to go red when that path is broken. Break them on purpose.
MUTANTS = [
    ("the early return for a review that started",
     'if review["status"] != "queued":', "if False:",
     "stays silent once our own review starts"),
    ("the guard on an unusable STUCK_MINUTES",
     "if not math.isfinite(mins) or mins <= 0:", "if False:",
     "does not page anyone when STUCK_MINUTES is unusable"),
    ("paging past the first page of in_progress runs",
     "if len(batch) < PAGE:", "if True:",
     "finds the busy runner under a large backlog"),
    ("the busy-runner check",
     "if box_busy(api, repo, own_run_id):", "if False:",
     "stays silent while the box is busy with another review"),
    # Proves the busy answer reflects the box at the end of the window, not mid-wait.
    ("the box-busy answer at the end of the window",
     "if box_busy(api, repo, own_run_id):", "if True:",
     "alarms when the busy runner dies mid-wait"),
    ("the guard on a run with no review job",
     "if not saw_review:", "if False:",
     "fails loudly when its own run has no review job"),
    ("failing the job on alarm",
     'return failed("no runner claimed this review in %s min" % shown)', "return 0",
     "alarms when no runner ever claims the review"),
    ("the warning when the post is rejected",
     'warning("could not post', 'notice("could not post',
     "still fails the job when the notice cannot be posted"),
    ("the owner override",
     '(env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"', '"zufayu"',
     "pages the override owner when the repo sets one"),
]


def transport_checks(mod):
    """_get/_post are the one seam the fake above replaces -- and they are hand-written, where
    an SDK used to be. Drive them against a real socket so a wrong header name or a swallowed
    error status cannot hide behind a fake that never speaks HTTP."""
    import http.server
    import json as _json
    import threading

    seen = []

    def _lower(headers):   # urllib normalises header names; compare case-insensitively
        return {k.lower(): v for k, v in headers.items()}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):            # keep the suite's output clean
            pass

        def _reply(self, code, obj):
            body = _json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen.append(("GET", self.path, _lower(self.headers), None))
            if self.path.endswith("/boom"):
                return self._reply(404, {"message": "nope"})
            self._reply(200, {"jobs": [{"name": "review", "status": "queued"}]})

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            seen.append(("POST", self.path, _lower(self.headers),
                         _json.loads(self.rfile.read(n) or b"{}")))
            self._reply(201, {"id": 1})

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    api = mod.GitHub("read-token", "bot-token")
    api.API = "http://127.0.0.1:%d" % srv.server_address[1]
    out = []
    try:
        jobs = api.jobs(REPO, 7)
        out.append(("jobs() parses the response",
                    jobs == [{"name": "review", "status": "queued"}]))
        _, _, h, _ = seen[-1]
        out.append(("a read uses the workflow token", h.get("authorization") == "token read-token"))
        out.append(("a read asks for the v3 media type", "vnd.github" in h.get("accept", "")))
        status = api.comment(REPO, 42, "hello")
        verb, path, h, body = seen[-1]
        out.append(("comment() returns the status", status == 201))
        out.append(("comment() POSTs the body",
                    verb == "POST" and path.endswith("/issues/42/comments")
                    and body == {"body": "hello"}))
        out.append(("comment() posts as the bot", h.get("authorization") == "token bot-token"))
        try:
            api._get("/boom")
            out.append(("a failed read raises rather than returning None", False))
        except RuntimeError:
            out.append(("a failed read raises rather than returning None", True))
    finally:
        srv.shutdown()
    return out


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

    print("[watchdog behaviour]")
    for name, err in suite(src).items():
        t(name, err)

    print("[watchdog transport]")
    for name, good in transport_checks(load(src)):
        t(name, None if good else "see watchdog.py GitHub._get/_post")

    print("[watchdog guards bite]")
    for why, find, repl, breaks in MUTANTS:
        if find not in src:            # the script drifted; the mutation would test nothing
            t("breaking %s is caught" % why, "watchdog.py no longer contains `%s`" % find)
            continue
        t('breaking %s turns "%s" red' % (why, breaks),
          None if suite(src.replace(find, repl, 1))[breaks]
          else "it stayed green, so that check proves nothing")

    print("=== %d green / %d red ===" % (ok, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

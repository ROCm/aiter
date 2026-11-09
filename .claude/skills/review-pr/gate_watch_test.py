#!/usr/bin/env python3
"""Drive gate_watch.py against a fake GitHub and a fixed clock.

The gate watch is a scheduled workflow, so GitHub will not run it on a PR branch at all --
and the failure it exists to catch, hosted runners starved, cannot be reproduced on demand.
Drive the shipping script instead: the URL it queries, the paging, the window it pages
inside and the job it fails are all the shipping code.

The source is read from gate_watch.py rather than copied here, so these checks cannot keep
passing after it has changed underneath them.

Usage: python3 .claude/skills/review-pr/gate_watch_test.py
"""

import contextlib
import datetime
import io
import os
import sys
import types
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "gate_watch.py"
REPO = "ROCm/aiter"
NOW = datetime.datetime(2026, 11, 9, 12, 0, 0)


def load(src):
    mod = types.ModuleType("gate_watch_under_test")
    mod.__file__ = str(SRC)
    exec(compile(src, str(SRC), "exec"), mod.__dict__)
    return mod


def run_at(run_id, minutes):
    """A queued review run created `minutes` before the poll."""
    return {"id": run_id,
            "created_at": (NOW - datetime.timedelta(minutes=minutes)
                           ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "html_url": "https://github.com/%s/actions/runs/%d" % (REPO, run_id)}


def drive(mod, scn):
    """Run gate_watch.run() once; return everything it did."""
    rec = {"notices": [], "warnings": [], "errors": [], "lines": [], "rc": None}

    class Fake(mod.GitHub):
        def _get(self, path):
            u = urllib.parse.urlparse(path)
            q = urllib.parse.parse_qs(u.query)
            if not u.path.endswith("/runs"):
                raise AssertionError("unexpected GET %s" % path)
            if "/workflows/aiter-review-bot.yml/" not in u.path:
                raise AssertionError("watching the wrong workflow: %s" % u.path)
            if q["status"][0] != "queued":
                raise AssertionError("queried status=%s, want queued" % q["status"][0])
            queued = scn["queued"]
            if queued == "boom":
                raise RuntimeError("GET %s -> 503" % path)
            # Model what the real endpoint does: newest first, capped at per_page. A fake
            # that always returned everything would hide a paging bug entirely.
            per, page = int(q["per_page"][0]), int(q.get("page", ["1"])[0])
            newest_first = sorted(queued, key=lambda r: -r["id"])
            return {"workflow_runs": newest_first[(page - 1) * per:page * per]}

        def _post(self, path, body, token):
            raise AssertionError("gate_watch must not post anything: %s" % path)

    env = {"GITHUB_REPOSITORY": REPO, "OWNER_OVERRIDE": ""}
    env.update(scn.get("env", {}))

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rec["rc"] = mod.run(Fake("gh-token"), env, scn["now"])
    for line in out.getvalue().splitlines():
        for tag, key in (("::notice::", "notices"), ("::warning::", "warnings"),
                         ("::error::", "errors")):
            if line.startswith(tag):
                rec[key].append(line[len(tag):])
                break
        else:
            rec["lines"].append(line)
    return rec


CASES = [
    ("stays quiet when nothing is queued",
     {"queued": [], "now": NOW},
     lambda r: [
         (r["rc"] == 0, "failed the job with an empty queue"),
         (not r["errors"], "paged about an empty queue: %s" % r["errors"]),
     ]),

    ("stays quiet while requests are merely queueing",
     {"queued": [run_at(1, 5), run_at(2, 10)], "now": NOW},
     lambda r: [
         (r["rc"] == 0, "failed the job for requests still inside a normal queue"),
         (not r["errors"], "paged about a queue that is merely busy: %s" % r["errors"]),
         (any("2 run(s) queued" in m for m in r["notices"]),
          "left no trace of what it saw: %s" % r["notices"]),
     ]),

    ("pages when a request has sat queued past the threshold",
     {"queued": [run_at(1, 35)], "now": NOW},
     lambda r: [
         (r["rc"] == 1, "stayed green while a request starved"),
         (any("hosted runner" in m for m in r["errors"]),
          "the page does not say what happened: %s" % r["errors"]),
         ("https://github.com/%s/actions/runs/1" % REPO in r["lines"],
          "did not name the stuck run: %s" % r["lines"]),
     ]),

    # Exactly once per stuck request is the whole design: the window is one poll wide, so a run
    # an earlier poll already paged must not page again on every poll after it.
    ("pages once per stuck request, not once per poll",
     {"queued": [run_at(1, 35), run_at(2, 70)], "now": NOW},
     lambda r: [
         (r["rc"] == 1, "stayed green while a request starved"),
         (any("1 review request" in m for m in r["errors"]),
          "counted a run an earlier poll already paged: %s" % r["errors"]),
         ("https://github.com/%s/actions/runs/2" % REPO not in r["lines"],
          "named the already-paged run again: %s" % r["lines"]),
     ]),

    ("stays quiet about a run an earlier poll already paged",
     {"queued": [run_at(1, 70)], "now": NOW},
     lambda r: [
         (r["rc"] == 0, "paged about the same stuck run a second time"),
         (not r["errors"], "re-paged a run an earlier poll already paged: %s" % r["errors"]),
         (any("newly stuck" in m for m in r["notices"]),
          "left no trace of the run it chose not to re-page: %s" % r["notices"]),
     ]),

    ("pages the override owner when the repo sets one",
     {"queued": [run_at(1, 35)], "now": NOW, "env": {"OWNER_OVERRIDE": "  gyohuangxin  "}},
     lambda r: [
         (r["rc"] == 1 and any("@gyohuangxin" in m for m in r["errors"]), "ignored the override"),
         (r["rc"] == 1 and not any("@zufayu" in m for m in r["errors"]),
          "paged the default owner too"),
     ]),

    # A watcher that cannot see the queue must not pass for a healthy one: the silence it
    # would keep is the silence this workflow exists to break.
    ("fails loudly when the queue cannot be read",
     {"queued": "boom", "now": NOW},
     lambda r: [
         (r["rc"] == 1, "a blind watcher passed for a healthy one"),
         (any("cannot read" in m for m in r["errors"]),
          "did not say it could not see the queue: %s" % r["errors"]),
     ]),

    # A burst queues as a long list and the API returns the newest first -- so the run stuck
    # longest, the one worth paging about, is on the LAST page.
    ("finds the stuck request under a large backlog",
     {"queued": [run_at(1000 + i, 5) for i in range(119)] + [run_at(999, 35)], "now": NOW},
     lambda r: [
         (r["rc"] == 1, "missed the stuck request under the backlog"),
         (any("1 review request" in m for m in r["errors"]),
          "counted the whole backlog as stuck: %s" % r["errors"]),
         ("https://github.com/%s/actions/runs/999" % REPO in r["lines"],
          "did not name the stuck run: %s" % r["lines"]),
     ]),
]

# Every check above passes against code that does nothing in the path it claims to guard, unless
# it is shown to go red when that path is broken. Break them on purpose.
MUTANTS = [
    ("the stuck threshold",
     "STUCK_MINUTES = 30", "STUCK_MINUTES = 3000",
     "pages when a request has sat queued past the threshold"),
    ("the one-poll page window",
     "WINDOW_MINUTES = 15", "WINDOW_MINUTES = 90",
     "stays quiet about a run an earlier poll already paged"),
    ("failing the job when it pages",
     "return failed(msg)", "return 0  # failed(msg)",
     "pages when a request has sat queued past the threshold"),
    ("the owner override",
     '(env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"', '"zufayu"',
     "pages the override owner when the repo sets one"),
    ("the loud failure when the queue cannot be read",
     'return failed("cannot read the queued runs:', 'return 0  # cannot read the queued runs:',
     "fails loudly when the queue cannot be read"),
    ("paging past the first page of queued runs",
     "if len(batch) < PAGE:", "if True:",
     "finds the stuck request under a large backlog"),
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
    """main() is the one path the behaviour checks never touch: they call run() directly, so a
    mistyped environment variable or a changed constructor lives there unseen until the first
    real cron fire. Drive it against a quiet queue."""
    mod = load(src)
    seen = []

    class Fake(mod.GitHub):
        def _get(self, path):
            seen.append(path)
            return {"workflow_runs": []}

        def _post(self, *a, **k):
            raise AssertionError("gate_watch must not post anything")

    mod.GitHub = Fake
    saved = dict(os.environ)
    os.environ.update({"GH_TOKEN": "t", "GITHUB_REPOSITORY": REPO, "OWNER_OVERRIDE": ""})
    try:
        with contextlib.redirect_stdout(io.StringIO()):     # its workflow commands are not ours
            rc = mod.main()
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return [("main() runs and decides", rc in (0, 1)),
            ("main() built an API object that was actually used", bool(seen))]


def main():
    src = SRC.read_text(encoding="utf-8")
    ok = bad = 0

    def t(name, err):
        nonlocal ok, bad
        if err:
            print("  ❌ %s — %s" % (name, err)); bad += 1
        else:
            print("  ✅ %s" % name); ok += 1

    print("[gate watch behaviour]")
    for name, err in suite(src).items():
        t(name, err)

    print("[gate watch guards bite]")
    for why, find, repl, breaks in MUTANTS:
        if find not in src:            # the script drifted; the mutation would test nothing
            t("breaking %s is caught" % why, "gate_watch.py no longer contains `%s`" % find)
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
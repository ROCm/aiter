#!/usr/bin/env python3
"""Leave the server untouched for a bounded interval after the baseline fails."""

import json
import re
import sys
import time
from pathlib import Path

import psutil


def pending_builds(text):
    pending = set()
    for action, name in re.findall(r"(start|finish) build \[([^\]]+)\]", text):
        if action == "start":
            pending.add(name)
        else:
            pending.discard(name)
    return pending


def main():
    log, output = map(Path, sys.argv[1:])
    original = log.read_text(errors="replace")
    pending = pending_builds(original)
    if not pending:
        raise SystemExit(
            "No unfinished build at watchdog failure; do not assume a compiler timeout."
        )
    print("Unfinished builds at baseline failure:", sorted(pending), flush=True)
    start = time.monotonic()
    result = {
        "pending_at_failure": sorted(pending),
        "completed": False,
        "client_exited": False,
    }
    while time.monotonic() - start < 1200:
        text = log.read_text(errors="replace")
        remaining = pending & pending_builds(text)
        if not remaining and not result["completed"]:
            result.update(
                completed=True, seconds_after_failure=time.monotonic() - start
            )
            print("Interrupted build finished:", json.dumps(result), flush=True)
        clients = []
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                if any("lm_eval" in arg for arg in (proc.info["cmdline"] or [])[:3]):
                    clients.append(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if result["completed"] and not clients:
            result["client_exited"] = True
            break
        if re.search(
            r"MEMORY_VIOLATION|ASSERT_TRAP|Memory access fault by GPU|Traceback",
            text[len(original) :],
        ):
            result["fault_after_failure"] = True
            break
        time.sleep(10)
    (output / "build-observation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["completed"] or not result["client_exited"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

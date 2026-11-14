"""Sum token usage per workflow subagent of this DSH session.

Maps childId -> label/phase/workflow from the parent session's tool-workflow events,
then walks each child session for usage records.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

from compression import zstd

# The harness exports the path of the current session log; every sibling session lives beside it.
_jsonl = os.environ.get("DSH_SESSION_JSONL")
if not _jsonl:
    raise SystemExit("set DSH_SESSION_JSONL (the harness exports it) before running this tool")
PARENT = Path(_jsonl)
SESS = PARENT.parent.parent


def events(path):
    raw = zstd.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    for line in raw.splitlines():
        try:
            yield json.loads(line)
        except Exception:
            continue


def find_usage(obj, acc):
    """Recursively collect any {input/output/prompt/completion}_tokens style dicts."""
    if isinstance(obj, dict):
        keys = set(obj.keys())
        if {"input", "output"} <= keys and all(isinstance(obj.get(k), int) for k in ("input", "output")):
            acc["input"] += obj["input"]
            acc["output"] += obj["output"]
            return
        if "inputTokens" in keys or "outputTokens" in keys:
            acc["input"] += obj.get("inputTokens") or 0
            acc["output"] += obj.get("outputTokens") or 0
            return
        if "prompt_tokens" in keys or "completion_tokens" in keys:
            acc["input"] += obj.get("prompt_tokens") or 0
            acc["output"] += obj.get("completion_tokens") or 0
            return
        for v in obj.values():
            find_usage(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_usage(v, acc)


def main():
    meta = {}
    run_names = {}
    for ev in events(PARENT):
        d = ev.get("data") or {}
        if ev.get("type") == "tool-workflow/run-start":
            run_names[d.get("runId")] = d.get("name")
        if ev.get("type") == "tool-workflow/agent-start":
            meta[d.get("childId")] = (run_names.get(d.get("runId"), "?"), d.get("label"))

    rows = []
    for cid, (run, label) in meta.items():
        dirs = [p for p in SESS.iterdir() if p.is_dir() and p.name == cid]
        if not dirs:
            rows.append((run, label, 0, 0, "NO SESSION"))
            continue
        f = dirs[0] / "session.jsonl.zstd"
        acc = defaultdict(int)
        steps = 0
        for ev in events(f):
            if ev.get("type") == "step/start":
                steps += 1
            if ev.get("type") in ("step/end", "assistant/message", "turn/end", "usage"):
                find_usage(ev.get("data") or {}, acc)
        rows.append((run, label, acc["input"], acc["output"], f"{steps} steps"))

    by_run = defaultdict(lambda: [0, 0, 0])
    for run, label, i, o, note in rows:
        by_run[run][0] += 1
        by_run[run][1] += i
        by_run[run][2] += o
        print(f"{run:34} | {str(label)[:38]:38} | in {i:9,} | out {o:7,} | {note}")

    print("\n=== per workflow ===")
    ti = to = 0
    for run, (n, i, o) in by_run.items():
        print(f"{run:34} | {n:3} agents | in {i:10,} | out {o:8,}")
        ti += i
        to += o
    print(f"{'TOTAL':34} | {len(rows):3} agents | in {ti:10,} | out {to:8,}")


if __name__ == "__main__":
    main()

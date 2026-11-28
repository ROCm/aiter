"""Render a DSH subagent session (zstd-compressed JSONL) as a readable transcript.

usage: python dump_subagent_log.py <session-id-or-prefix> [more ids...]
Writes <workspace>/pr-review-env/logs/<label>.md and prints where it went.
"""
import json
import os
import sys
from pathlib import Path

from compression import zstd

# The harness exports the path of the current session log; every sibling session lives beside it.
_jsonl = os.environ.get("DSH_SESSION_JSONL")
if not _jsonl:
    raise SystemExit("set DSH_SESSION_JSONL (the harness exports it) before running this tool")
PARENT = Path(_jsonl)
SESS = PARENT.parent.parent
OUT = Path(os.environ.get("AGENT_LOOP_LOGS") or (Path.cwd() / "agent-logs"))


def load(path):
    raw = zstd.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    for line in raw.splitlines():
        try:
            yield json.loads(line)
        except Exception:
            continue


def clip(text, n):
    text = str(text)
    return text if len(text) <= n else text[:n] + f"\n... [+{len(text) - n} chars truncated]"


def render(sid, full=False):
    matches = [d for d in SESS.iterdir() if d.is_dir() and d.name.startswith(sid)]
    if not matches:
        return f"no session dir for {sid}"
    d = matches[0]
    f = d / "session.jsonl.zstd"
    out = [f"# subagent session {d.name}", ""]
    limit = 10**9 if full else 4000
    for ev in load(f):
        t = ev.get("type")
        data = ev.get("data") or {}
        if t == "user/message":
            out += ["## PROMPT", "", "```", clip(data.get("text") or data.get("content") or json.dumps(data), 20000), "```", ""]
        elif t == "assistant/message":
            txt = data.get("text") or data.get("content") or ""
            if isinstance(txt, list):
                txt = "\n".join(str(p.get("text", p)) for p in txt if isinstance(p, dict))
            if str(txt).strip():
                out += ["## ASSISTANT", "", clip(txt, limit), ""]
        elif t == "tool/call":
            name = data.get("name") or data.get("toolName")
            args = data.get("arguments") or data.get("args") or data.get("input") or {}
            out += [f"### -> tool `{name}`", "", "```", clip(json.dumps(args, ensure_ascii=False, indent=1), 3000), "```", ""]
        elif t == "tool/result":
            res = data.get("result") or data.get("output") or data
            out += ["### <- result", "", "```", clip(json.dumps(res, ensure_ascii=False)[:100000], limit), "```", ""]
        elif t in ("error", "agent/error", "step/error"):
            out += [f"### !! {t}", "", "```", clip(json.dumps(data, ensure_ascii=False), 4000), "```", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{d.name}.md"
    target.write_text("\n".join(out), encoding="utf-8")
    return f"{target}  ({target.stat().st_size} bytes, {len(out)} blocks)"


if __name__ == "__main__":
    for sid in sys.argv[1:]:
        print(render(sid))

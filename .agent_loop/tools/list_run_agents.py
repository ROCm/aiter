"""List the subagents a named workflow run actually launched, from the parent session log.

usage: python list_run_agents.py <workflow-name>
"""
import json
import os
import sys

from compression import zstd

p = os.environ["DSH_SESSION_JSONL"]
want = sys.argv[1] if len(sys.argv) > 1 else None

runs = {}
rows = []
for line in zstd.decompress(open(p, "rb").read()).decode("utf-8", errors="replace").splitlines():
    try:
        o = json.loads(line)
    except Exception:
        continue
    t, d = o.get("type"), (o.get("data") or {})
    if t == "tool-workflow/run-start":
        runs[d.get("runId")] = d.get("name")
    if t == "tool-workflow/agent-start":
        name = runs.get(d.get("runId"), "?")
        if want is None or name == want:
            rows.append((d.get("seq"), d.get("phase"), d.get("label"), d.get("childId")))

for seq, ph, label, cid in rows:
    print(f"seq {seq:>3}  {str(ph):<26} {str(label):<44} {cid}")
print(f"\ntotal agents: {len(rows)}")

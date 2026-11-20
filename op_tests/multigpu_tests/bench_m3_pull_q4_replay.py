# SPDX-License-Identifier: MIT
"""Fixed first-turn request replay; model outputs need not be deterministic.

Not an MLPerf/AgentX result. See docs/gfx1250-pull-q4-prefetch.md.
Usage: M3_REPLAY_DATASET=/path/to/dataset.jsonl python this.py TAG 96 SALT
"""

import asyncio
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

import aiohttp

tag = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 48
salt = sys.argv[3] if len(sys.argv) > 3 else "0"
root = pathlib.Path(os.environ.get("M3_REPLAY_OUTPUT", "."))
dataset = pathlib.Path(os.environ["M3_REPLAY_DATASET"])
pool = []
seen = set()
for line in dataset.open():
    d = json.loads(line)
    if d.get("role") == "user" and d["conversation_id"] not in seen:
        seen.add(d["conversation_id"])
        pool.append(d)
requests = []
for i, d in enumerate(pool[:count]):
    requests.append(
        dict(
            model="/models/MiniMax-M3-MXFP4",
            messages=[
                {
                    "role": "system",
                    "content": f"[benchmark salt {salt}, request {i:04d}]\n"
                    + d["system"],
                },
                {"role": "user", "content": d["content"]},
            ],
            tools=d["tools"],
            temperature=0,
            top_p=1,
            max_tokens=8192,
            seed=188,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
out = root / "results" / tag
out.mkdir(parents=True, exist_ok=False)
(out / "requests.sha256").write_text(
    hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
)


async def main():
    sem = asyncio.Semaphore(8)
    rows = []
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as sess:

        async def one(i, body):
            async with sem:
                t0 = time.perf_counter()
                first = None
                usage = None
                chunks = []
                error = None
                try:
                    async with sess.post(
                        os.environ.get(
                            "M3_REPLAY_API", "http://127.0.0.1:18890"
                        ).rstrip("/")
                        + "/v1/chat/completions",
                        json=body,
                    ) as r:
                        r.raise_for_status()
                        async for line in r.content:
                            if not line.startswith(b"data: "):
                                continue
                            data = line[6:].strip()
                            if data == b"[DONE]":
                                break
                            d = json.loads(data)
                            if d.get("usage"):
                                usage = d["usage"]
                            for c in d.get("choices", []):
                                delta = c.get("delta") or {}
                                if any(
                                    delta.get(k)
                                    for k in (
                                        "content",
                                        "reasoning_content",
                                        "reasoning",
                                        "tool_calls",
                                    )
                                ):
                                    if first is None:
                                        first = time.perf_counter()
                                    chunks.append(delta)
                except Exception as e:
                    error = repr(e)
                end = time.perf_counter()
                n = (usage or {}).get("completion_tokens")
                canonical = {"content": "", "reasoning": "", "tools": {}}
                for delta in chunks:
                    canonical["content"] += delta.get("content") or ""
                    canonical["reasoning"] += (
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    for call in delta.get("tool_calls") or []:
                        idx = call.get("index", 0)
                        t = canonical["tools"].setdefault(
                            idx, {"name": "", "arguments": ""}
                        )
                        for k in ("name", "arguments"):
                            t[k] += (call.get("function") or {}).get(k) or ""
                row = dict(
                    i=i,
                    ttft_s=first - t0 if first else None,
                    latency_s=end - t0,
                    tpot_ms=(end - first) * 1000 / (n - 1)
                    if first and n and n > 1
                    else None,
                    usage=usage,
                    error=error,
                    response_hash=hashlib.sha256(
                        json.dumps(canonical, sort_keys=True).encode()
                    ).hexdigest(),
                    response=canonical,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)

        begin = time.perf_counter()
        await asyncio.gather(*(one(i, b) for i, b in enumerate(requests)))
        elapsed = time.perf_counter() - begin
    tokens = sum((x["usage"] or {}).get("completion_tokens", 0) for x in rows)
    report = dict(
        tag=tag,
        kind="secondary first-turn replay, not recipe result",
        n=len(rows),
        failed=sum(x["error"] is not None for x in rows),
        duration_s=elapsed,
        output_tokens=tokens,
        tok_s=tokens / elapsed,
        ttft_p50_s=statistics.median(
            x["ttft_s"] for x in rows if x["ttft_s"] is not None
        ),
        tpot_p50_ms=statistics.median(
            x["tpot_ms"] for x in rows if x["tpot_ms"] is not None
        ),
    )
    (out / "rows.json").write_text(
        json.dumps(sorted(rows, key=lambda x: x["i"]), indent=2)
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


asyncio.run(main())

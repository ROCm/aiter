#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract prefill/decode MoE token counts from an ATOM trace."
    )
    parser.add_argument("trace", type=Path, help="*.pt.trace.json.gz file")
    parser.add_argument("output", type=Path, help="output JSON path")
    parser.add_argument("--aiter-commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with gzip.open(args.trace, "rt", encoding="utf-8", errors="replace") as handle:
        events = json.load(handle).get("traceEvents", [])

    counts: Counter[tuple] = Counter()
    labels: dict[tuple, str] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "user_annotation":
            continue
        name = str(event.get("name", ""))
        if name.startswith("prefill["):
            phase = "prefill"
        elif name.startswith("decode["):
            phase = "decode"
        elif name.startswith("eager_decode["):
            phase = "eager_decode"
        else:
            continue

        bs_match = re.search(r"bs=(\d+)(?:/(\d+))?", name)
        tok_match = re.search(r"tok=(\d+)", name)
        if bs_match is None or tok_match is None:
            continue

        scheduled_bs = int(bs_match.group(1))
        graph_bs = int(bs_match.group(2) or scheduled_bs)
        scheduled_tokens = int(tok_match.group(1))
        if phase == "prefill":
            query_tokens_per_request = None
            moe_tokens = scheduled_tokens
            use_cudagraph = False
        else:
            query_tokens_per_request, remainder = divmod(
                scheduled_tokens, scheduled_bs
            )
            if remainder:
                query_tokens_per_request = None
                moe_tokens = scheduled_tokens
            else:
                moe_tokens = (
                    graph_bs * query_tokens_per_request
                    if phase == "decode"
                    else scheduled_tokens
                )
            use_cudagraph = phase == "decode"

        key = (
            phase,
            scheduled_tokens,
            moe_tokens,
            scheduled_bs,
            graph_bs,
            query_tokens_per_request,
            use_cudagraph,
        )
        counts[key] += 1
        labels[key] = name

    records = []
    for key, calls in sorted(counts.items()):
        phase, scheduled_tokens, moe_tokens, scheduled_bs, graph_bs, q, use_cg = key
        records.append(
            {
                "phase": phase,
                "calls": calls,
                "scheduled_tokens": scheduled_tokens,
                "moe_tokens": moe_tokens,
                "scheduled_bs": scheduled_bs,
                "graph_bs": graph_bs,
                "query_tokens_per_request": q,
                "use_cudagraph": use_cg,
                "example_label": labels[key],
            }
        )

    payload = {
        "aiter_commit": args.aiter_commit,
        "trace_file": str(args.trace),
        "workload": {
            "model": args.model,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "requests": args.requests,
            "max_concurrency": args.max_concurrency,
            "speculative_decode": False,
        },
        "semantics": {
            "scheduled_tokens": "logical tokens selected by the scheduler",
            "moe_tokens": (
                "physical rows passed through the model and MoE after "
                "CUDAGraph padding"
            ),
            "calls": "model forward calls carrying this MoE token shape",
        },
        "records": records,
        "prefill_moe_tokens": sorted(
            {record["moe_tokens"] for record in records if record["phase"] == "prefill"}
        ),
        "decode_moe_tokens": sorted(
            {
                record["moe_tokens"]
                for record in records
                if record["phase"] in ("decode", "eager_decode")
            }
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Summarize an AITER_RECORD bundle without replaying it.

    python op_tests/replay/inspect_bundle.py <bundle-dir>

Prints the session (GPU, aiter commit, framework patches, configs) and the
recorded calls grouped by op, weight shape, M and the kernels aiter chose.
Needs only the standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _tensor_shape(encoded) -> tuple | None:
    if isinstance(encoded, dict) and "__tensor__" in encoded:
        return tuple(encoded["__tensor__"]["shape"])
    return None


def call_key(call: dict) -> tuple[str, str, str, str]:
    """(op, weights, M, kernels) for grouping calls."""
    args = call.get("args", {})
    if call.get("op") == "fused_moe":
        weights = _tensor_shape(args.get("w1"))
        tokens = _tensor_shape(args.get("hidden_states"))
    else:  # generic: first two tensor arguments
        shapes = [s for s in map(_tensor_shape, args.values()) if s is not None]
        tokens = shapes[0] if shapes else None
        weights = shapes[1] if len(shapes) > 1 else None
    m = str(tokens[0]) if tokens else "?"
    kernels = " | ".join(
        "+".join(d.get("kernels", [])) for d in call.get("dispatch", [])
    )
    return (
        call.get("op", "?"),
        str(list(weights)) if weights else "?",
        m,
        kernels or "<no dispatch>",
    )


def summarize(bundle: Path) -> tuple[dict, Counter, Counter]:
    session = json.loads((bundle / "session.json").read_text())
    groups: Counter = Counter()
    flags: Counter = Counter()
    calls_path = bundle / "calls.jsonl"
    if calls_path.exists():
        with open(calls_path) as fp:
            for line in fp:
                if not line.strip():
                    continue
                call = json.loads(line)
                key = call_key(call)
                groups[key] += 1
                if call.get("error"):
                    flags[(key, "error")] += 1
                if call.get("capturing"):
                    flags[(key, "capturing")] += 1
    return session, groups, flags


def _m_sort(value: str) -> tuple[int, str]:
    return (int(value), "") if value.isdigit() else (sys.maxsize, value)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("bundle", type=Path)
    args = p.parse_args(argv)
    if not (args.bundle / "session.json").exists():
        print(f"{args.bundle}: no session.json (not a bundle directory?)")
        return 2
    session, groups, flags = summarize(args.bundle)

    aiter = session.get("aiter", {})
    print(f"bundle:     {args.bundle}")
    print(
        f"created:    {session.get('created')}  host: {session.get('host')}  pid: {session.get('pid')}"
    )
    print(f"gpu:        {session.get('gpu')}")
    print(
        f"aiter:      {str(aiter.get('commit') or '?')[:12]}  dirty: {aiter.get('dirty')}  "
        f"torch: {session.get('torch')}  hip: {session.get('hip')}"
    )
    print(
        f"frameworks: {session.get('frameworks') or '-'}  distributed: {session.get('distributed') or '-'}"
    )
    env = session.get("env", {})
    print(
        f"env:        {len(env)} AITER_* vars"
        + (f": {', '.join(sorted(env))}" if env else "")
    )
    for op, info in session.get("ops", {}).items():
        print(f"op {op}:")
        print(f"  counters: {session.get('counters', {}).get(op)}")
        print(f"  patched:  {info.get('patched') or '{}'}")
        for env_var, cfg in info.get("configs", {}).items():
            print(f"  config:   {env_var} <- {cfg.get('source')}")
        for key in ("session_info_error", "config_error"):
            if key in info:
                print(f"  {key}: {info[key]}")

    print()
    print(f"{'op':<10} {'weights':<22} {'M':>6} {'calls':>6}  flags  kernels")
    for key in sorted(groups, key=lambda k: (k[0], k[1], _m_sort(k[2]), k[3])):
        op, weights, m, kernels = key
        marks = ",".join(
            f"{name}={flags[(key, name)]}"
            for name in ("error", "capturing")
            if flags[(key, name)]
        )
        print(
            f"{op:<10} {weights:<22} {m:>6} {groups[key]:>6}  {marks or '-':<5}  {kernels}"
        )
    print(f"\n{sum(groups.values())} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())

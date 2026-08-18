#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize rocprofv3 ATT stats CSV by instruction class."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict


def classify(instruction: str) -> str:
    op = instruction.strip().split(" ", 1)[0]
    if op.startswith("v_mfma"):
        return "mfma"
    if op == "s_barrier":
        return "barrier"
    if op == "s_waitcnt":
        return "waitcnt"
    if op == "buffer_inv":
        return "cache_invalidate"
    if "atomic" in op:
        return "atomic"
    if op.startswith(("global_load", "buffer_load", "flat_load")):
        return "global_load"
    if op.startswith(("global_store", "buffer_store", "flat_store")):
        return "global_store"
    if op.startswith("ds_read"):
        return "lds_read"
    if op.startswith("ds_write"):
        return "lds_write"
    if "branch" in op or op.startswith("s_cbranch"):
        return "branch"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--code-object", type=int, required=True)
    args = parser.parse_args()

    totals = defaultdict(
        lambda: {"hitcount": 0, "latency": 0, "stall": 0, "idle": 0}
    )
    with open(args.csv_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["CodeObj"]) != args.code_object:
                continue
            instruction = row["Instruction"]
            if instruction.startswith(";"):
                continue
            item = totals[classify(instruction)]
            for field in item:
                item[field] += int(row[field.capitalize()] or 0)

    overall = {
        field: sum(item[field] for item in totals.values())
        for field in ("hitcount", "latency", "stall", "idle")
    }
    result = {
        "code_object": args.code_object,
        "overall": overall,
        "classes": dict(sorted(totals.items())),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

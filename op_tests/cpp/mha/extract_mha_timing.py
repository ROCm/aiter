#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Extract kernel wall time (ms) from MHA benchmark stdout.

Usage: extract_mha_timing.py fwd|bwd < captured.txt

Prints a single timing value (e.g. 0.098). Uses the last line matching the
benchmark summary: ", <ms> ms, <TFlops> TFlops".
"""
from __future__ import annotations

import re
import sys

# CK prints: ", 0.098 ms, 1.37 TFlops, 1366.55 GB/s"
PAT = re.compile(r",\s*(\d+(?:\.\d+)?)\s+ms,\s*[\d.]+\s+TFlops", re.MULTILINE)
FALLBACK = re.compile(r",\s*(\d+(?:\.\d+)?)\s+ms,")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("fwd", "bwd"):
        print("usage: extract_mha_timing.py fwd|bwd < stdout", file=sys.stderr)
        sys.exit(2)
    text = sys.stdin.read()
    matches = list(PAT.finditer(text))
    if matches:
        print(matches[-1].group(1), end="")
        return
    matches = list(FALLBACK.finditer(text))
    if matches:
        print(matches[-1].group(1), end="")
        return
    sys.exit(4)


if __name__ == "__main__":
    main()

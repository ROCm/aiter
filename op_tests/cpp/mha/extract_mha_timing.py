#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Extract kernel wall time (ms) from MHA benchmark stdout. Usage: extract_mha_timing.py fwd|bwd < captured.txt"""
from __future__ import annotations

import re
import sys

PAT = re.compile(r",\s*(\d+\.\d+)\s+ms,\s*[\d.]+\s+TFlops")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("fwd", "bwd"):
        print("usage: extract_mha_timing.py fwd|bwd < stdout", file=sys.stderr)
        sys.exit(2)
    kind = sys.argv[1]
    text = sys.stdin.read()
    key = "fmha_fwd" if kind == "fwd" else "fmha_bwd"
    pos = text.rfind(key)
    if pos < 0:
        sys.exit(3)
    chunk = text[pos : pos + 50000]
    m = PAT.search(chunk)
    if not m:
        m = re.search(r",\s*(\d+\.\d+)\s+ms,", chunk)
    if not m:
        sys.exit(4)
    print(m.group(1), end="")


if __name__ == "__main__":
    main()

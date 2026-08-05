#!/usr/bin/env python3
"""Derive a modified copy of an ISA dump, for testing edits end to end.

    # negative control: perturb one accumulator, output must change
    python make_variant.py in.s out.s --drop-first-wmma

    # the DEVIATION-1 experiment: move the hoisted wait back after the WMMA block
    python make_variant.py in.s out.s --wait-after-wmma
"""
import argparse
import re
import sys
from pathlib import Path


def drop_first_wmma(lines):
    """Delete the first WMMA in the steady loop -- a wrong-answer control."""
    for i, l in enumerate(lines):
        if "v_wmma_scale" in l:
            return lines[:i] + lines[i + 1:], f"dropped WMMA at line {i+1}"
    raise SystemExit("no v_wmma_scale found")


def wait_after_wmma(lines):
    """Move the `s_wait_dscnt 0x0` that LLVM hoisted after the 1st WMMA.

    In the wide build the emitted order is
        s_wait_dscnt 0x13 / wmma#1 / s_wait_dscnt 0x0 / wmma#2..32
    so the latency-hiding window is 1 WMMA wide instead of 16. This pushes the
    wait down to just before the KSL1 block (after 16 WMMA).
    """
    out = list(lines)
    idx = [i for i, l in enumerate(out) if "v_wmma_scale" in l]
    if len(idx) < 17:
        raise SystemExit("fewer than 17 WMMA; not the wide steady loop")
    wait_i = None
    for i in range(idx[0], idx[1]):
        if re.search(r"s_wait_dscnt 0x0\b", out[i]):
            wait_i = i
            break
    if wait_i is None:
        raise SystemExit("no `s_wait_dscnt 0x0` between WMMA 1 and 2")
    line = out.pop(wait_i)
    idx = [i for i, l in enumerate(out) if "v_wmma_scale" in l]
    insert_at = idx[15] + 1  # after the 16th WMMA
    out.insert(insert_at, line)
    return out, f"moved s_wait_dscnt 0x0 from line {wait_i+1} to {insert_at+1}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--drop-first-wmma", action="store_true")
    p.add_argument("--wait-after-wmma", action="store_true")
    a = p.parse_args()

    lines = Path(a.src).read_text().split("\n")
    if a.drop_first_wmma:
        lines, what = drop_first_wmma(lines)
    elif a.wait_after_wmma:
        lines, what = wait_after_wmma(lines)
    else:
        raise SystemExit("pick an edit")
    Path(a.dst).write_text("\n".join(lines))
    print(f"{a.dst}: {what}")


if __name__ == "__main__":
    sys.exit(main())

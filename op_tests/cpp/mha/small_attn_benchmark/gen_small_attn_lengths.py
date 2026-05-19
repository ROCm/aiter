#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic per-batch sequence lengths for CK group-mode packed varlen.

Distribution (default ``iid``): each batch row gets an independent draw from the
discrete uniform on {2, 3, …, max_p} (inclusive). Same seed and arguments always
produce the same comma-separated lists for CK ``-s`` / ``-s_k`` and for JAX.

Usage:
  gen_small_attn_lengths.py <scenario:1|2> <batch> <max_P> <outfile> [--seed N]
  gen_small_attn_lengths.py --print <scenario> <batch> <max_P> [--seed N]

Library:
  from gen_small_attn_lengths import generate_logical_lengths
  sq, sk = generate_logical_lengths(1, 2048, 16, seed=6764)
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Literal

DEFAULT_SEED = 6764
Distribution = Literal["iid", "stratified"]


def generate_logical_lengths(
    scenario: int,
    batch: int,
    max_p: int,
    *,
    seed: int = DEFAULT_SEED,
    distribution: Distribution = "iid",
) -> tuple[list[int], list[int]]:
    """Return (seqlen_q, seqlen_k) length lists for one benchmark point.

    Parameters
    ----------
    scenario:
        1 — both Q and KV lengths uniform in [2, max_p].
        2 — Q fixed at 1; KV lengths uniform in [2, max_p].
    batch:
        Number of batch rows (length of each list).
    max_p:
        Upper bound P for the sweep row (max logical length).
    seed:
        RNG seed. **Reset on every call** — same as re-running this script for
        each P in the shell sweep.
    distribution:
        ``iid`` (default) — ``random.randint(2, max_p)`` per row (discrete uniform).
        ``stratified`` — each value in [2, max_p] appears equally often, order
        shuffled (still deterministic from ``seed``).

    Examples
    --------
    CK / JAX for scenario 1, B=2048, P=16::

        sq, sk = generate_logical_lengths(1, 2048, 16, seed=6764)
        # sq[0], sk[0] are lengths for batch row 0, etc.
    """
    if scenario not in (1, 2):
        raise ValueError("scenario must be 1 or 2")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if max_p < 2:
        raise ValueError("max_p must be >= 2")

    rng = random.Random(seed)

    def uniform_row_lengths() -> list[int]:
        if distribution == "iid":
            return [rng.randint(2, max_p) for _ in range(batch)]
        # Stratified discrete uniform: equal counts of each length in [2, max_p].
        values = list(range(2, max_p + 1))
        n_vals = len(values)
        base, rem = divmod(batch, n_vals)
        lengths: list[int] = []
        for i, v in enumerate(values):
            lengths.extend([v] * (base + (1 if i < rem else 0)))
        rng.shuffle(lengths)
        return lengths

    if scenario == 1:
        sq = uniform_row_lengths()
        sk = uniform_row_lengths()
    else:
        sq = [1] * batch
        sk = uniform_row_lengths()

    return sq, sk


def write_lengths(path: str, sq: list[int], sk: list[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(map(str, sq)) + "\n")
        f.write(",".join(map(str, sk)) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--print",
        action="store_true",
        help="print sq and sk comma lists to stdout (two lines) instead of writing a file",
    )
    p.add_argument("scenario", type=int, nargs="?")
    p.add_argument("batch", type=int, nargs="?")
    p.add_argument("max_p", type=int, nargs="?")
    p.add_argument("outfile", nargs="?")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--distribution",
        choices=("iid", "stratified"),
        default="iid",
        help="iid: randint per row (default, matches existing ck_pr_6764 benches)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.print:
        if args.scenario is None or args.batch is None or args.max_p is None:
            print("usage: gen_small_attn_lengths.py --print <1|2> <batch> <max_P> [--seed N]", file=sys.stderr)
            sys.exit(2)
        sq, sk = generate_logical_lengths(
            args.scenario, args.batch, args.max_p, seed=args.seed, distribution=args.distribution
        )
        print(",".join(map(str, sq)))
        print(",".join(map(str, sk)))
        return

    if args.scenario is None or args.batch is None or args.max_p is None or args.outfile is None:
        print(
            "usage: gen_small_attn_lengths.py <1|2> <batch> <max_P> <outfile> [--seed N]",
            file=sys.stderr,
        )
        sys.exit(2)

    sq, sk = generate_logical_lengths(
        args.scenario, args.batch, args.max_p, seed=args.seed, distribution=args.distribution
    )
    write_lengths(args.outfile, sq, sk)


if __name__ == "__main__":
    main()

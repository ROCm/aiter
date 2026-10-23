# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Write ``candidates.csv``: the pinned FlyDSL configurations the tuning sweeps time.

The bench scenario of the multi-GPU tests reads it with ``--candidates`` and
times each row as one more column next to the shipping engine and the stock
kernels (``flydsl_allreduce_bench.load_candidates`` has the format). The fit
scripts read the same file to map a column back to its knobs.

The fused grids depend on which row geometries the kernels can build, so they
are generated from the kernel modules rather than written by hand. Re-run this
after a change to those geometries, and check the file in::

    python3 op_tests/multigpu_tests/flydsl_allreduce_tuning/gen_candidates.py
    # or, in CI or tune.sh check: fail if the checked-in file is stale
    python3 op_tests/multigpu_tests/flydsl_allreduce_tuning/gen_candidates.py --check

Names are the column prefixes of the bench tables and the keys the fit scripts
report, so they stay stable: ``fly_1stage_b<block>_a<atoms>_g<cap>``,
``fly_int4[_ring]_b<block>_st<st>_g<cap>``,
``fused_fly_1stage[_k<split>]_b<block>_g<cap>`` and
``fused_fly_<mesh|ring>_a<atoms_per_row>_st<st>_g<cap>``.

Sets (``--candidate-set``):

* ``sweep``     -- every plain row; the data the plain fit is fitted from.
* ``validate``  -- the plain rows the shipped dispatcher can choose between
  (one-shot and mesh; the ring is never dispatched on xGMI).
* ``fused``     -- the fused one-shot grid, unsplit and split.
* ``fused_qr``  -- the fused mesh/ring grid.
* ``smoke``     -- a handful of rows of each, for ``tune.sh --smoke``.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

OUT = os.path.join(_HERE, "candidates.csv")
COLUMNS = (
    "name",
    "operation",
    "algorithm",
    "atoms",
    "grid_cap",
    "fanout",
    "block",
    "split",
    "super_tile",
    "atoms_per_row",
    "sets",
)

# Plain one-shot: ``(block, atoms, grid_cap)``, peer fanout.
ONESHOT_GRID = (
    (256, 1, 64),
    (256, 1, 128),
    (128, 1, 128),
    (64, 1, 64),
    (512, 1, 64),
    (512, 2, 64),
    (256, 2, 64),
    (256, 4, 64),
    (512, 4, 64),
    (512, 4, 128),
)
# Plain mesh and ring: ``(block, super_tile, grid_cap)``.
MESH_GRID = tuple((b, st, 128) for b in (256, 512) for st in (1, 8))
RING_GRID = tuple((b, st, 128) for b in (128, 256, 512) for st in (8, 16, 32))

# Widths the fused one-shot grid is generated for: which blocks exist is a
# function of the width, so the grid is the union over these, and the bench
# leaves a row out (n/a) at a width that cannot build it.
FUSED_HIDDENS = (2048, 3072, 4096, 5120, 6144, 7168, 8192)
# The grid cap is the only knob that moves the block count of a fused build:
# the tile is one token row.
FUSED_CAPS = (8, 32, 64, 128)
# A split build spends ``split`` workgroups per row, so 256 lets k2 match the
# unsplit 128.
FUSED_SPLIT_CAPS = (*FUSED_CAPS, 256)
# Fused mesh/ring: atoms per row x super-tile x grid cap.
FUSED_QR_ROW_ATOMS = (1, 2, 4)
FUSED_QR_CAPS = (32, 64, 128, 256)

SMOKE = {
    "fly_1stage_b256_a1_g128",
    "fly_int4_b256_st1_g128",
    "fused_fly_mesh_a1_st1_g128",
    "fused_fly_ring_a1_st8_g128",
}


def _row(name, operation, algorithm, sets, **knobs):
    row = dict.fromkeys(COLUMNS, "")
    row.update(name=name, operation=operation, algorithm=algorithm)
    row.update({k: str(v) for k, v in knobs.items()})
    if name in SMOKE:
        sets = (*sets, "smoke")
    row["sets"] = ";".join(sets)
    return row


def rows() -> list[dict]:
    from aiter.ops.flydsl.kernels.comm.one_shot_allreduce_rmsnorm import (
        fused_block_options,
        fused_split_options,
    )
    from aiter.ops.flydsl.quick_allreduce import ALGORITHMS

    out = []
    for b, a, g in ONESHOT_GRID:
        out.append(
            _row(
                f"fly_1stage_b{b}_a{a}_g{g}",
                "ar",
                "oneshot",
                ("sweep", "validate"),
                atoms=a,
                grid_cap=g,
                fanout="peer",
                block=b,
            )
        )
    for b, st, g in MESH_GRID:
        out.append(
            _row(
                f"fly_int4_b{b}_st{st}_g{g}",
                "ar",
                "mesh",
                ("sweep", "validate"),
                super_tile=st,
                grid_cap=g,
                block=b,
            )
        )
    for b, st, g in RING_GRID:
        out.append(
            _row(
                f"fly_int4_ring_b{b}_st{st}_g{g}",
                "ar",
                "ring",
                ("sweep",),
                super_tile=st,
                grid_cap=g,
                block=b,
            )
        )

    blocks = sorted(
        {b for h in FUSED_HIDDENS for b, _a in fused_block_options(h)}, reverse=True
    )
    fused_one = []
    for b in blocks:
        for g in FUSED_CAPS:
            fused_one.append(
                _row(
                    f"fused_fly_1stage_b{b}_g{g}",
                    "ar_rmsnorm",
                    "oneshot",
                    ("fused",),
                    grid_cap=g,
                    block=b,
                )
            )
    pairs = sorted(
        {(k, b) for h in FUSED_HIDDENS for k, b, _a in fused_split_options(h) if k > 1},
        key=lambda kb: (kb[0], -kb[1]),
    )
    for k, b in pairs:
        # The engine rounds a split build's cap to a multiple of ``split``, so
        # a cap below ``split`` would run as another row's build.
        for g in FUSED_SPLIT_CAPS:
            if g < k:
                continue
            fused_one.append(
                _row(
                    f"fused_fly_1stage_k{k}_b{b}_g{g}",
                    "ar_rmsnorm",
                    "oneshot",
                    ("fused",),
                    grid_cap=g,
                    block=b,
                    split=k,
                )
            )
    # Three rows of the one-shot grid in the smoke set.
    for r in fused_one[:3]:
        r["sets"] += ";smoke"
    out += fused_one

    for algorithm in ("mesh", "ring"):
        for apr in FUSED_QR_ROW_ATOMS:
            for st in ALGORITHMS[algorithm].super_tiles:
                for g in FUSED_QR_CAPS:
                    out.append(
                        _row(
                            f"fused_fly_{algorithm}_a{apr}_st{st}_g{g}",
                            "ar_rmsnorm",
                            algorithm,
                            ("fused_qr",),
                            super_tile=st,
                            grid_cap=g,
                            atoms_per_row=apr,
                        )
                    )
    return out


def render(table: list[dict]) -> str:
    buf = io.StringIO()
    buf.write(
        "# Generated by gen_candidates.py -- edit that file and re-run it, do not\n"
        "# edit this one. Format: flydsl_allreduce_bench.load_candidates.\n"
    )
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(table)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if candidates.csv differs from what would be written",
    )
    args = ap.parse_args()
    text = render(rows())
    if args.check:
        with open(OUT) as fh:
            if fh.read() != text:
                raise SystemExit(f"{OUT} is stale: re-run {os.path.basename(__file__)}")
        print(f"{OUT} is up to date")
        return
    with open(OUT, "w") as fh:
        fh.write(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

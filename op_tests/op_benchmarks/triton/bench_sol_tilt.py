# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time of the Sol-Attn Jensen cap and low-rank tilt wrappers on Wan 2.2 shapes (us).

One layer's self-attention per rank: the whole sequence, 40 heads split over the Ulysses degree,
head dim 128. sol_q_abs_max runs per Q kind, sol_jensen_cap per K kind, sol_tilt_planes per K/V
recipe and sol_tilt_factors on the per-tensor FP8 operands. The inputs are the unit tests'.

    python bench_sol_tilt.py
    python bench_sol_tilt.py --resolution 480p --ulysses 4 8 --op planes --rank 1 2 4
"""

import argparse
import functools

import pandas as pd

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.sol_tilt import (
    sol_jensen_cap,
    sol_q_abs_max,
    sol_tilt_factors,
    sol_tilt_planes,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_sol_tilt import (
    make_jensen_inputs,
    make_q_abs_max_inputs,
    make_tilt_factor_inputs,
    make_tilt_inputs,
)

# Latent tokens of a Wan 2.2 clip: 21 frames of 45 x 80 at 720p, of 30 x 52 at 480p.
WAN_TOKENS = {"720p": 21 * 45 * 80, "480p": 21 * 30 * 52}
WAN_HEADS = 40
Q_KINDS = ["fp8", "bf16", "mxfp8", "rotated"]
K_KINDS = ["fp8", "bf16", "mxfp8", "mxfp4", "mxfp6"]
RECIPES = [
    ("e4m3", "e4m3"),
    ("int8", "e4m3"),
    ("bf16", "bf16"),
    ("mxfp8", "e4m3"),
    ("e4m3", "mxfp6_fp6_p"),
    ("mxfp6", "mxfp6_fp6_p"),
    ("mxfp6", "mxfp4_fp6_p"),
    ("mxfp4", "mxfp4_fp6_p"),
]
PACKED = ("mxfp4", "mxfp6", "mxfp4_fp6_p", "mxfp6_fp6_p", "rotated")
OPS = ["q_abs_max", "jensen_cap", "planes", "factors"]


def _cases(op, tokens, heads, kv_tile, rank):
    """(recipe, generator, op) for each case of op; each generator builds one call's kwargs."""
    shape = {"batch": 1, "heads": heads}
    if op == "q_abs_max":
        for kind in Q_KINDS:
            yield kind, functools.partial(
                make_q_abs_max_inputs, kind, seqlen=tokens, **shape
            ), sol_q_abs_max
    elif op == "jensen_cap":
        for kind in K_KINDS:
            yield kind, functools.partial(
                make_jensen_inputs,
                kind,
                seqlen_q=tokens,
                seqlen_k=tokens,
                kv_tile=kv_tile,
                **shape,
            ), sol_jensen_cap
    elif op == "planes":
        for k_format, v_format in RECIPES:
            yield f"{k_format}/{v_format}", functools.partial(
                make_tilt_inputs,
                k_format,
                v_format,
                seqlen_q=tokens,
                seqlen_k=tokens,
                kv_tile=kv_tile,
                rank=rank,
                **shape,
            ), sol_tilt_planes
    else:
        yield "e4m3/e4m3", functools.partial(
            make_tilt_factor_inputs,
            seqlen_q=tokens,
            seqlen_k=tokens,
            kv_tile=kv_tile,
            rank=rank,
            **shape,
        ), sol_tilt_factors


def _kwargs_only(make):
    """make's call kwargs alone, as the pool's one rotating field."""
    out = make()
    return (out[0] if isinstance(out, tuple) else out,)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--resolution", nargs="+", choices=sorted(WAN_TOKENS), default=["720p"]
    )
    parser.add_argument(
        "--ulysses", nargs="+", type=int, default=[8], help="Ulysses degrees"
    )
    parser.add_argument("--op", nargs="+", choices=OPS, default=OPS)
    parser.add_argument("--rank", nargs="+", type=int, default=[2])
    parser.add_argument("--kv-tile", type=int, default=128, choices=[64, 128])
    args = parser.parse_args()

    packed = get_gfx() == "gfx950"
    bencher = RotatingBufferBench()
    results = []
    for resolution in args.resolution:
        tokens = WAN_TOKENS[resolution]
        for degree in args.ulysses:
            heads = WAN_HEADS // degree
            for op in args.op:
                ranks = args.rank if op in ("planes", "factors") else [None]
                for rank in ranks:
                    for recipe, make, fn in _cases(
                        op, tokens, heads, args.kv_tile, rank
                    ):
                        if not packed and any(f in recipe for f in PACKED):
                            continue
                        bencher.build_pool(
                            [functools.partial(_kwargs_only, make)], [["kwargs"]]
                        )
                        us = bencher.bench(lambda kwargs, fn=fn: fn(**kwargs)) * 1000
                        results.append(
                            {
                                "resolution": resolution,
                                "ulysses": degree,
                                "heads": heads,
                                "op": op,
                                "recipe": recipe,
                                "rank": rank if rank is not None else "-",
                                "us": round(us, 2),
                            }
                        )
    logger.info(
        "Sol tilt wrappers, kv_tile %d (us):\n%s",
        args.kv_tile,
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()

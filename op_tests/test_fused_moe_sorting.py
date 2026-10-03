# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf sweep for the fused MoE sorting kernels.

Two ops under test, both single-launch HIP kernels that also zero `moe_buf`:
  fused        fused_moe_sorting_fwd       sort only; caller runs its own router
  fused_topk   fused_moe_sorting_topk_fwd  router (sigmoid top-k) folded in

They are compared against the Opus and CK sorting backends on the same inputs,
and every candidate is checked bit-exact (atol=0) against the torch reference
shared with test_moe_sorting.py. The router inputs come from aiter's own
grouped_topk so fused_topk is judged against exactly what it replaces.
"""

import argparse
import itertools

import pandas as pd
import torch
from test_moe_sorting import (
    _compare_moe_sorting_outputs,
    _moe_sorting_roofline,
    run_torch_moe_sorting,
)

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.fused_moe_sorting import (
    fused_moe_sorting_fwd,
    fused_moe_sorting_is_supported,
    fused_moe_sorting_topk_fwd,
    fused_moe_sorting_topk_is_supported,
)
from aiter.ops.topk import grouped_topk
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

# Validated on MI355X only so far; the kernel is plain HIP with GFX9 DPP.
SUPPORTED_GFX = ["gfx950"]

# GLM-5.2 router: sigmoid scoring, one group, renormalised, scale 2.5.
ROUTER_N_GROUP = 1
ROUTER_TOPK_GROUP = 1
ROUTER_RENORM = True
ROUTER_SOFTMAX = False
ROUTER_SCALE = 2.5


def _alloc_outputs(token, topk, E, unit_size, model_dim, dtype):
    max_padded = int(token * topk + E * unit_size - topk)
    max_blocks = int((max_padded + unit_size - 1) // unit_size)
    return {
        "sorted_ids": torch.empty(max_padded, dtype=dtypes.i32),
        "sorted_weights": torch.empty(max_padded, dtype=dtypes.fp32),
        "sorted_expert_ids": torch.empty(max_blocks, dtype=dtypes.i32),
        "num_valid_ids": torch.empty(2, dtype=dtypes.i32),
        "moe_buf": torch.empty((token, model_dim), dtype=dtype),
    }


def _as_tuple(o):
    return (
        o["sorted_ids"],
        o["sorted_weights"],
        o["sorted_expert_ids"],
        o["num_valid_ids"],
        o["moe_buf"],
    )


@benchmark()
def test_fused_moe_sorting(token, E, topk, unit_size, model_dim, dtype):
    # GLM-5.2 routes over n_routed_experts=256 while the sort sees E=257 (the
    # shared expert is not routed), so the gating tile is [token, E-1] there.
    router_e = E - 1 if E == 257 else E
    gating = (torch.randn((token, router_e), dtype=dtypes.fp32) * 3.0).contiguous()
    topk_weights = torch.empty((token, topk), dtype=dtypes.fp32)
    topk_ids = torch.empty((token, topk), dtype=dtypes.i32)
    grouped_topk(
        gating,
        topk_weights,
        topk_ids,
        ROUTER_N_GROUP,
        ROUTER_TOPK_GROUP,
        ROUTER_RENORM,
        ROUTER_SOFTMAX,
        ROUTER_SCALE,
    )

    ref = run_torch_moe_sorting(topk_ids, topk_weights, E, unit_size, None, None)

    ws_size = aiter.moe_sorting_opus_get_workspace_size(token, E, topk, 0)
    opus_ws = torch.empty(ws_size, dtype=torch.uint8) if ws_size > 0 else None
    # Cross-block handshake word for the fused router; zero once, kernel
    # leaves it zero.
    sync = torch.zeros(2, dtype=dtypes.i32)

    outs = {}

    def opus():
        o = outs["opus"]
        aiter.moe_sorting_opus_fwd(
            topk_ids,
            topk_weights,
            o["sorted_ids"],
            o["sorted_weights"],
            o["sorted_expert_ids"],
            o["num_valid_ids"],
            o["moe_buf"],
            E,
            int(unit_size),
            None,
            None,
            opus_ws,
            0,
            None,
            None,
            None,
        )
        return _as_tuple(o)

    def ck():
        o = outs["ck"]
        aiter.moe_sorting_fwd(
            topk_ids,
            topk_weights,
            o["sorted_ids"],
            o["sorted_weights"],
            o["sorted_expert_ids"],
            o["num_valid_ids"],
            o["moe_buf"],
            E,
            int(unit_size),
            None,
            None,
            0,
        )
        return _as_tuple(o)

    def fused():
        o = outs["fused"]
        fused_moe_sorting_fwd(
            topk_ids,
            topk_weights,
            o["sorted_ids"],
            o["sorted_weights"],
            o["sorted_expert_ids"],
            o["num_valid_ids"],
            o["moe_buf"],
            E,
            int(unit_size),
            None,
            None,
        )
        return _as_tuple(o)

    def fused_topk():
        o = outs["fused_topk"]
        fused_moe_sorting_topk_fwd(
            gating,
            o["sorted_ids"],
            o["sorted_weights"],
            o["sorted_expert_ids"],
            o["num_valid_ids"],
            o["moe_buf"],
            E,
            topk,
            int(unit_size),
            ROUTER_RENORM,
            ROUTER_SOFTMAX,
            ROUTER_SCALE,
            None,
            None,
            sync,
        )
        return _as_tuple(o)

    candidates = {"opus": opus, "ck": ck}
    if fused_moe_sorting_is_supported(token, E, topk, int(unit_size)):
        candidates["fused"] = fused
    if fused_moe_sorting_topk_is_supported(token, E, topk, int(unit_size), router_e):
        candidates["fused_topk"] = fused_topk

    flops, nbytes = _moe_sorting_roofline(token, topk, E, model_dim, dtype)
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        outs[name] = _alloc_outputs(token, topk, E, unit_size, model_dim, dtype)
        outs[name]["sorted_ids"].fill_(-1)
        outs[name]["sorted_expert_ids"].fill_(-1)
        # Correctness on a poisoned buffer first, then timing.
        errs = _compare_moe_sorting_outputs(ref, fn(), topk, token)
        err = max(errs.values()) if errs else 0.0
        _, us = run_perftest(fn, num_rotate_args=1)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fused_moe_sorting unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        help="moe_buf data type.\n    e.g.: -d bf16",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 8, 16, 32, 64, 80],
        help="Number of tokens.\n    e.g.: -m 64",
    )
    parser.add_argument(
        "-e",
        "--expert",
        type=int,
        nargs="*",
        default=[257, 256, 128],
        help="Number of experts (paired with -t).\n    e.g.: -e 257",
    )
    parser.add_argument(
        "-t",
        "--topk",
        type=int,
        nargs="*",
        default=[8, 8, 8],
        help="Top-k per token (paired with -e).\n    e.g.: -t 8",
    )
    parser.add_argument(
        "-u",
        "--unit_size",
        type=int,
        nargs="*",
        default=[16, 32],
        help="Rows per expert block.\n    e.g.: -u 32",
    )
    parser.add_argument(
        "-md",
        "--model_dim",
        type=int,
        nargs="*",
        default=[6144],
        help="Model dimension (moe_buf width).\n    e.g.: -md 6144",
    )
    args = parser.parse_args()
    assert len(args.expert) == len(args.topk), "-e and -t are paired"

    df = []
    for dtype, (E, topk), unit_size, model_dim, token in itertools.product(
        args.dtype, zip(args.expert, args.topk), args.unit_size, args.model_dim, args.m
    ):
        df.append(test_fused_moe_sorting(token, E, topk, unit_size, model_dim, dtype))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fused_moe_sorting summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()

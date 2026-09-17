# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]

# Kimi-K3 fused MoE-front: [gate_up | router logits | routed]
# Mapping: m = tokens; E = 896 routed experts; topk = 16; one group (no group filter).
GATE_UP_WIDTH = 1536
NUM_EXPERTS = 896
ROUTED_WIDTH = 3584
FUSED_FRONT_WIDTH = GATE_UP_WIDTH + NUM_EXPERTS + ROUTED_WIDTH
NUM_EXPERT_GROUP = 1
TOPK_GROUP = 1
TOPK = 16
NEED_RENORM = True
ROUTED_SCALING_FACTOR = 1.0


def run_torch(gating_output, correction_bias):
    # Reference only: fp32 math. Not timed, not in the table.
    weights, ids = aiter.biased_grouped_topk_torch(
        gating_output,
        correction_bias,
        TOPK,
        NEED_RENORM,
        NUM_EXPERT_GROUP,
        TOPK_GROUP,
    )
    return weights * ROUTED_SCALING_FACTOR, ids


def _sort_by_id(weights, ids):
    order = torch.argsort(ids, dim=-1)
    return weights.gather(1, order), ids.gather(1, order)


@benchmark()
def test_biased_grouped_topk(m, dtype):
    torch.manual_seed(20260917)
    backing = torch.randn((m, FUSED_FRONT_WIDTH), dtype=dtype)
    gating_output = backing[:, GATE_UP_WIDTH : GATE_UP_WIDTH + NUM_EXPERTS]
    correction_bias = torch.randn((NUM_EXPERTS,), dtype=dtype)
    ref_w, ref_id = run_torch(gating_output, correction_bias)
    ref_w, ref_id = _sort_by_id(ref_w, ref_id)

    topk_weights = torch.empty((m, TOPK), dtype=dtypes.fp32)
    topk_ids = torch.empty((m, TOPK), dtype=dtypes.i32)

    def run_hip():
        aiter.biased_grouped_topk(
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            NUM_EXPERT_GROUP,
            TOPK_GROUP,
            NEED_RENORM,
            ROUTED_SCALING_FACTOR,
        )
        return topk_weights, topk_ids

    candidates = {
        "hip": run_hip,
    }

    # sigmoid + bias per expert, then topk sequential scans, then a short renorm.
    flops = m * (5 * NUM_EXPERTS + TOPK * NUM_EXPERTS + TOPK)
    nbytes = (
        m * NUM_EXPERTS * gating_output.element_size()
        + NUM_EXPERTS * correction_bias.element_size()
        + m * TOPK * (topk_weights.element_size() + topk_ids.element_size())
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (out_w, out_id), us = run_perftest(fn)
        out_w, out_id = _sort_by_id(out_w, out_id)
        err = checkAllclose(
            ref_w.to(dtypes.fp32),
            out_w.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: biased_grouped_topk weights",
        )
        checkAllclose(
            ref_id.to(dtypes.fp32),
            out_id.to(dtypes.fp32),
            rtol=0,
            atol=0,
            msg=f"{name}: biased_grouped_topk ids",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "biased_grouped_topk unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ],
        help="""Number of tokens (Kimi-K3 router m).
    e.g.: -m 32""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = [test_biased_grouped_topk(m, dtype) for m in args.m]
        df = pd.DataFrame(df)
        aiter.logger.info(
            "biased_grouped_topk kimi-k3 hip baseline summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()

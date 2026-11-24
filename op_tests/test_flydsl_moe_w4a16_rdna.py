# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""W4A16 MoE on the gfx115x FlyDSL WMMA kernels against a dequantized torch
reference.

The sweep covers the three routing paths: the row and top-k-sum kernels (up to
ROW_PATH_MAX_TOKENS tokens), the grouped GEMM with a separate top-k sum, and
GEMM1 with SiLU-and-multiply in its epilogue or as a separate activation.
"""

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

from aiter import dtypes, logger
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_w4a16_rdna import (
    ROW_PATH_MAX_TOKENS,
    flydsl_rdna_moe,
    flydsl_rdna_moe_supported,
    gate_up_tile_order,
    pack_w4a16_weights,
    pad_expert_stride,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

# Single elements of large outputs land just outside the tolerance
# (1 of 1048576 at 512 tokens, group 32 with zero points).
MAX_MISMATCH = 1e-4

# (hidden, intermediate, experts, top_k) of the MoE layers of
# Qwen3.5/3.6-35B-A3B, Qwen3-30B-A3B and Gemma-4-26B-A4B.
MODEL_SHAPES = [
    (2048, 512, 256, 8),
    (2048, 768, 128, 8),
    (2816, 704, 128, 8),
]


def quantize(n_out, n_in, num_experts, group_size, zero_points, device):
    """Random unsigned int4 weights; returns ``(q, scale, zp, dequant)``."""
    shape = (num_experts, n_out, n_in)
    q = torch.randint(0, 16, shape, device=device)
    groups = (num_experts, n_out, n_in // group_size)
    scale = torch.rand(groups, dtype=torch.float32, device=device) * 0.02 + 0.001
    scale = scale.to(dtypes.bf16)
    if zero_points:
        zp = torch.randint(0, 16, groups, device=device)
    else:
        zp = torch.full(groups, 8, device=device)
    dequant = (q - zp.repeat_interleave(group_size, dim=2)).float()
    dequant *= scale.float().repeat_interleave(group_size, dim=2)
    return q, scale, (zp if zero_points else None), dequant


def run_torch(x, d1, d2, topk_weights, topk_ids, round_gate_up, round_down):
    """Reference in f32. ``round_gate_up`` / ``round_down`` round the GEMM1 and
    per-expert GEMM2 outputs to the activation dtype, where the op stores them."""

    def rounded(t, enable):
        return t.to(x.dtype).float() if enable else t

    out = torch.zeros((x.size(0), d2.size(1)), dtype=torch.float32, device=x.device)
    for e in topk_ids.unique().tolist():
        token, slot = (topk_ids == e).nonzero(as_tuple=True)
        gate_up = rounded(x[token].float() @ d1[e].T, round_gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        act = (F.silu(gate) * up).to(x.dtype).float()
        down = rounded(act @ d2[e].T, round_down)
        out.index_add_(0, token, down * topk_weights[token, slot, None])
    return out


@benchmark()
def test_flydsl_moe_w4a16_rdna(
    tokens, hidden, inter, experts, top_k, group_size, zero_points, fused_silu, dtype
):
    device = "cuda"
    torch.manual_seed(0)
    q1, s1, zp1, d1 = quantize(
        2 * inter, hidden, experts, group_size, zero_points, device
    )
    q2, s2, zp2, d2 = quantize(hidden, inter, experts, group_size, zero_points, device)
    if fused_silu:
        order = gate_up_tile_order(2 * inter, device)
        q1, s1 = q1[:, order], s1[:, order]
        zp1 = None if zp1 is None else zp1[:, order]
    w1, c1 = map(pad_expert_stride, pack_w4a16_weights(q1, s1, zp1))
    w2, c2 = map(pad_expert_stride, pack_w4a16_weights(q2, s2, zp2))

    x = torch.randn((tokens, hidden), dtype=dtype, device=device)
    logits = torch.randn((tokens, experts), dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.softmax(logits, dim=-1).topk(top_k)
    topk_ids = topk_ids.to(torch.int32)
    # Up to ROW_PATH_MAX_TOKENS, GEMM2 sums the experts in f32.
    ref = run_torch(
        x,
        d1,
        d2,
        topk_weights,
        topk_ids,
        round_gate_up=not fused_silu,
        round_down=tokens > ROW_PATH_MAX_TOKENS,
    )

    def silu_and_mul(out, gate_up):
        gate, up = gate_up.float().chunk(2, dim=-1)
        out.copy_(F.silu(gate) * up)

    out = torch.empty((tokens, hidden), dtype=dtype, device=device)
    workspace = torch.empty(
        tokens * top_k * (3 * inter + hidden), dtype=dtype, device=device
    )
    _, us = run_perftest(
        flydsl_rdna_moe,
        out,
        x,
        w1,
        c1,
        w2,
        c2,
        topk_weights,
        topk_ids,
        workspace,
        group_size,
        experts,
        inter,
        None if fused_silu else silu_and_mul,
        zero_points,
        zero_points,
    )

    # Only the weights routed to are read.
    routed = min(tokens * top_k, experts)
    weight_bytes = routed * 3 * hidden * inter // 2
    flops = 2 * tokens * top_k * 3 * hidden * inter
    err = checkAllclose(
        ref,
        out.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2 * ref.std().item(),
        msg="flydsl_rdna_moe",
    )
    assert (
        err <= MAX_MISMATCH
    ), f"flydsl_rdna_moe: {err:.4%} of elements outside rtol/atol"
    return {
        "gfx": get_gfx_runtime(),
        "path": "row" if tokens <= ROW_PATH_MAX_TOKENS else "grouped",
        "flydsl_rdna us": us,
        "flydsl_rdna TFLOPS": flops / us / 1e6,
        "flydsl_rdna TB/s": weight_bytes / us / 1e6,
        "flydsl_rdna err": err,
    }


def main():
    if not flydsl_rdna_moe_supported():
        logger.warning("flydsl_rdna_moe unsupported on %s; skipping", get_gfx_runtime())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="W4A16 MoE on the gfx115x FlyDSL kernels",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16, dtypes.fp16],
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 2, ROW_PATH_MAX_TOKENS, ROW_PATH_MAX_TOKENS + 1, 64, 512],
    )
    parser.add_argument("-g", "--group_size", type=int, nargs="*", default=[32, 128])
    parser.add_argument(
        "-z", "--zero_points", type=dtypes.str2bool, nargs="*", default=[False, True]
    )
    parser.add_argument(
        "-f", "--fused_silu", type=dtypes.str2bool, nargs="*", default=[True, False]
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        rows = [
            test_flydsl_moe_w4a16_rdna(
                tokens, *shape, group_size, zero_points, fused_silu, dtype
            )
            for shape, tokens, group_size, zero_points, fused_silu in itertools.product(
                MODEL_SHAPES,
                args.tokens,
                args.group_size,
                args.zero_points,
                args.fused_silu,
            )
            # Both GEMMs need K to be a multiple of the group size.
            if shape[0] % group_size == 0 and shape[1] % group_size == 0
        ]
        logger.info(
            "flydsl_rdna_moe %s summary (markdown):\n%s",
            dtype,
            pd.DataFrame(rows).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""W4A16 MoE on the gfx115x FlyDSL WMMA kernels against a dequantized torch
reference.

The sweep covers the three routing paths: the row and top-k-sum kernels (up to
ROW_PATH_MAX_TOKENS tokens), the grouped GEMM with a separate top-k sum, and
GEMM1 with SiLU-and-multiply in its epilogue or as a separate activation.
Further cases cover routes without an expert (id -1) and shapes whose operands
do not fit the LDS of the row or top-k-sum kernels. The workspace starts out
as NaN, so rows the op does not write show up.
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
    TILE_N,
    flydsl_rdna_moe,
    flydsl_rdna_moe_gemm,
    flydsl_rdna_moe_path,
    flydsl_rdna_moe_supported,
    gate_up_tile_order,
    pack_w4a16_weights,
    pad_expert_stride,
    topk_weighted_sum,
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

# (hidden, intermediate, experts, top_k, group_size) off the default dispatch:
# GEMM1 of the row kernel needs 68 KiB of LDS (grouped path at 1 token), the
# top-k-sum kernel 128 KiB (row GEMM2 + topk_weighted_sum), and top_k above
# MAX_REDUCE_ROWS, an odd top_k and top_k = 1.
EDGE_SHAPES = [
    (2048, 2048, 8, 8, 2048),
    (2048, 8192, 8, 8, 128),
    (2048, 512, 16, 10, 128),
    (2048, 512, 16, 3, 128),
    (2048, 512, 16, 1, 128),
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
        if e < 0:
            continue
        token, slot = (topk_ids == e).nonzero(as_tuple=True)
        gate_up = rounded(x[token].float() @ d1[e].T, round_gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        act = (F.silu(gate) * up).to(x.dtype).float()
        down = rounded(act @ d2[e].T, round_down)
        out.index_add_(0, token, down * topk_weights[token, slot, None])
    return out


@benchmark()
def test_flydsl_moe_w4a16_rdna(
    tokens,
    hidden,
    inter,
    experts,
    top_k,
    group_size,
    zero_points,
    fused_silu,
    dtype,
    invalid_routes=False,
    padded=True,
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
    pad = pad_expert_stride if padded else (lambda t: t)
    w1, c1 = map(pad, pack_w4a16_weights(q1, s1, zp1))
    w2, c2 = map(pad, pack_w4a16_weights(q2, s2, zp2))

    x = torch.randn((tokens, hidden), dtype=dtype, device=device)
    logits = torch.randn((tokens, experts), dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.softmax(logits, dim=-1).topk(top_k)
    topk_ids = topk_ids.to(torch.int32)
    if invalid_routes:
        # As in an engine warm-up: all routes of token 0 and, with more than
        # one route, the last route of every token have no expert; their
        # weights stay nonzero.
        if top_k > 1:
            topk_ids[:, -1] = -1
        topk_ids[0] = -1
    path = flydsl_rdna_moe_path(
        tokens, hidden, inter, top_k, group_size, topk_weights.dtype
    )
    # The top-k-sum kernel sums the experts in f32.
    ref = run_torch(
        x,
        d1,
        d2,
        topk_weights,
        topk_ids,
        round_gate_up=not fused_silu,
        round_down=path != "reduce",
    )

    def silu_and_mul(out, gate_up):
        gate, up = gate_up.float().chunk(2, dim=-1)
        out.copy_(F.silu(gate) * up)

    # NaN, so rows the op does not write fail the finite check.
    out = torch.full((tokens, hidden), torch.nan, dtype=dtype, device=device)
    workspace = torch.full(
        (tokens * top_k * (3 * inter + hidden),), torch.nan, dtype=dtype, device=device
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
    routed = min(int((topk_ids >= 0).sum()), experts)
    weight_bytes = routed * 3 * hidden * inter // 2
    flops = 2 * tokens * top_k * 3 * hidden * inter
    # The mismatch allowance is for rounding; rows the op misses stay NaN.
    assert out.isfinite().all(), "flydsl_rdna_moe: non-finite output"
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
        "path": path,
        "flydsl_rdna us": us,
        "flydsl_rdna TFLOPS": flops / us / 1e6,
        "flydsl_rdna TB/s": weight_bytes / us / 1e6,
        "flydsl_rdna err": err,
    }


def check_rejects_raw_pointer_misuse():
    """Layouts the kernels cannot index are refused before a launch."""
    hidden, inter, experts, top_k, group = 256, 128, 4, 2, 32
    q1, s1, _, _ = quantize(2 * inter, hidden, experts, group, False, "cuda")
    q2, s2, _, _ = quantize(hidden, inter, experts, group, False, "cuda")
    order = gate_up_tile_order(2 * inter, "cuda")
    w1, c1 = pack_w4a16_weights(q1[:, order], s1[:, order], None)
    w2, c2 = pack_w4a16_weights(q2, s2, None)
    x = torch.randn((2, hidden), dtype=dtypes.bf16, device="cuda")
    ids = torch.zeros((2, top_k), dtype=torch.int32, device="cuda")
    weights = torch.ones((2, top_k), dtype=torch.float32, device="cuda")
    out = torch.empty((2, hidden), dtype=dtypes.bf16, device="cuda")
    ws = torch.empty(2 * top_k * (3 * inter + hidden), dtype=dtypes.bf16, device="cuda")
    args = (w1, c1, w2, c2, weights, ids, ws, group, experts, inter, None)
    args += (False, False)
    strided = torch.randn((2, 2 * hidden), dtype=x.dtype, device="cuda")[:, :hidden]
    cases = (
        ("contiguous", out, strided, ws),
        ("one device", out, x.cpu(), ws),
        ("output dtype", out, x.half(), ws.half()),
        ("workspace dtype", out.half(), x.half(), ws),
        ("shape", out[:, :TILE_N].contiguous(), x, ws),
        ("w1/w2_pairs required", out, x, ws),
    )
    for msg, bad_out, bad_x, bad_ws in cases:
        bad_args = args[:6] + (bad_ws,) + args[7:]
        if msg.endswith("required"):
            bad_args = bad_args[:-2]
        try:
            flydsl_rdna_moe(bad_out, bad_x, *bad_args)
        except ValueError as e:
            assert msg in str(e), e
        else:
            raise AssertionError(f"flydsl_rdna_moe accepted: {msg}")
    # No tokens: nothing to launch.
    flydsl_rdna_moe(out[:0], x[:0], *args[:4], weights[:0], ids[:0], *args[6:])
    # The other public entry points.
    gate_up = torch.empty((2 * top_k, 2 * inter), dtype=x.dtype, device="cuda")
    odd = torch.zeros((experts, c1[0].numel() + 1), dtype=c1.dtype, device="cuda")
    odd = odd[:, : c1[0].numel()].view(c1.shape)
    calls = (
        (
            "x_row_div",
            lambda: flydsl_rdna_moe_gemm(
                gate_up, x, w1, c1, None, ids.view(-1), None, 0, group, coef_pairs=False
            ),
        ),
        (
            "dword aligned",
            lambda: flydsl_rdna_moe_gemm(
                gate_up,
                x,
                w1,
                odd,
                None,
                ids.view(-1),
                None,
                top_k,
                group,
                coef_pairs=False,
            ),
        ),
        ("one output row", lambda: topk_weighted_sum(out[:1], gate_up, weights)),
        ("scale shape", lambda: pack_w4a16_weights(q2, s2.transpose(1, 2), None)),
    )
    for msg, call in calls:
        try:
            call()
        except ValueError as e:
            assert msg in str(e), e
        else:
            raise AssertionError(f"accepted: {msg}")
    # Valid calls: tensors by keyword, and no rows (nothing to launch).
    topk_weighted_sum(out=out, rows=gate_up, topk_weights=weights)
    flydsl_rdna_moe_gemm(
        gate_up[:0],
        x,
        w1,
        c1,
        None,
        ids.view(-1)[:0],
        None,
        top_k,
        group,
        coef_pairs=False,
    )


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
        nargs="+",
        default=[dtypes.bf16, dtypes.fp16],
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, ROW_PATH_MAX_TOKENS, ROW_PATH_MAX_TOKENS + 1, 64, 512],
    )
    parser.add_argument("-g", "--group_size", type=int, nargs="+", default=[32, 128])
    parser.add_argument(
        "-z", "--zero_points", type=dtypes.str2bool, nargs="+", default=[False, True]
    )
    parser.add_argument(
        "-f", "--fused_silu", type=dtypes.str2bool, nargs="+", default=[True, False]
    )
    args = parser.parse_args()
    check_rejects_raw_pointer_misuse()

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
        # Routes of id -1 (with zero points), the edge shapes (also with zero
        # points and a separate activation) and weights without the padded
        # expert stride.
        rows += [
            test_flydsl_moe_w4a16_rdna(
                tokens, *shape, group, invalid, fused_silu, dtype, invalid, padded
            )
            for (*shape, group), tokens, fused_silu, invalid, padded in (
                list(
                    itertools.product(
                        [(*MODEL_SHAPES[0], 128)],
                        [1, ROW_PATH_MAX_TOKENS, ROW_PATH_MAX_TOKENS + 1, 64],
                        [True, False],
                        [True],
                        [True],
                    )
                )
                + list(
                    itertools.product(
                        EDGE_SHAPES, [1, 64], [True, False], [False, True], [True]
                    )
                )
                + list(
                    itertools.product(
                        [(*MODEL_SHAPES[0], 128)], [1, 64], [True], [False], [False]
                    )
                )
            )
        ]
        logger.info(
            "flydsl_rdna_moe %s summary (markdown):\n%s",
            dtype,
            pd.DataFrame(rows).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()

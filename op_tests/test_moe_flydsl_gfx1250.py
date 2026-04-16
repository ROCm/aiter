# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for gfx1250 FlyDSL MOE kernels (fp4, fp8, a8w4, bf16, fp16).

All tests use fused_moe as the entry point, which dispatches to the
appropriate gfx1250 FlyDSL kernel based on quantization type and dtype.
"""

import sys
import itertools
import argparse
import logging

import torch
import pandas as pd

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx

from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    torch_moe_stage1,
    torch_moe_stage2,
)

torch.set_default_device("cuda")

GFX = get_gfx()
if GFX != "gfx1250":
    print(f"Skipping: gfx1250 required, got {GFX}")
    sys.exit(0)

try:
    from aiter.ops.flydsl.moe_kernels import _run_compiled  # noqa: F401

    _FLYDSL_OK = True
except ImportError:
    _FLYDSL_OK = False

if not _FLYDSL_OK:
    print("Skipping: FlyDSL runtime not available")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_ref_moe(
    input_bf16, w1_bf16, w2_bf16, topk_weights, topk_ids, activation, dtype
):
    """Compute full 2-stage MOE reference in fp32 from unquantized weights."""
    token_num, model_dim = input_bf16.shape
    E = w1_bf16.shape[0]
    topk = topk_ids.shape[1]
    inter_dim = w1_bf16.shape[1] // 2

    inp = input_bf16.to(torch.float32)
    w1 = w1_bf16.to(torch.float32)
    w2 = w2_bf16.to(torch.float32)

    act_fn = torch.nn.functional.silu if activation in (
        ActivationType.Silu, ActivationType.Swiglu
    ) else torch.nn.functional.gelu

    out1 = torch.zeros(token_num, topk, inter_dim, dtype=torch.float32, device="cuda")
    for e in range(E):
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        t_idx, s_idx = idx[:, 0], idx[:, 1]
        y = torch.mm(inp[t_idx], w1[e].T)
        gate = y[:, :inter_dim]
        up = y[:, inter_dim:]
        out1[t_idx, s_idx] = act_fn(gate) * up

    out2 = torch.zeros(token_num, model_dim, dtype=torch.float32, device="cuda")
    for e in range(E):
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        t_idx, s_idx = idx[:, 0], idx[:, 1]
        y = torch.mm(out1[t_idx, s_idx], w2[e].T)
        y = y * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out2.index_add_(0, t_idx, y)

    return out2.to(dtype)


def _per_1x32_fp8_quant_weight(w, block_size=32):
    """Quantize weight tensor to fp8 with per-32 E8M0 block scaling.

    Args:
        w: weight tensor (E, N, K) in bf16/fp16/fp32

    Returns:
        (w_fp8, w_scale) where w_fp8 has same shape as w in fp8 dtype,
        w_scale has shape (E, N, K//32) in fp8_e8m0 dtype.
    """
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    orig_shape = w.shape
    w_2d = w.reshape(-1, orig_shape[-1]).to(torch.float32)
    w_q, w_s = per_1x32_f8_scale_f8_quant(w_2d, scale_type=dtypes.fp8_e8m0)
    w_s = w_s.view(torch.uint8).view(dtypes.fp8_e8m0)
    return w_q.view(orig_shape), w_s.view(*orig_shape[:-1], orig_shape[-1] // block_size)


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

@benchmark()
def test_gfx1250_fmoe(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    activation,
    quant_type,
    wq_dtype,
    fmt_label,
):
    """Test a single gfx1250 FlyDSL MOE kernel configuration."""

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)

    input_fp = torch.randn((token, model_dim), dtype=dtype)
    w1_fp = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
    w2_fp = torch.randn((E, model_dim, inter_dim), dtype=dtype)
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input_fp, score, topk, True)

    # Reference: compute from full-precision weights
    out_ref = _torch_ref_moe(
        input_fp, w1_fp, w2_fp, topk_weights, topk_ids, activation, dtype
    )

    # --- Format-specific weight preparation ---
    if fmt_label in ("bf16", "fp16"):
        # Unquantized: pass bf16/fp16 weights directly
        w1_q, w2_q = w1_fp, w2_fp
        w1_scale, w2_scale = None, None

    elif fmt_label == "fp4":
        # MXFP4: per-32 block-scaled fp4x2
        w1_q, w1_scale = torch_quant(w1_fp, quant_dtype=dtypes.fp4x2)
        w2_q, w2_scale = torch_quant(w2_fp, quant_dtype=dtypes.fp4x2)
        w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
        w2_q = w2_q.view(E, model_dim, inter_dim // 2)

    elif fmt_label == "a8w4":
        # A8W4: fp4x2 weights, activation will be quantized to fp8 internally
        w1_q, w1_scale = torch_quant(w1_fp, quant_dtype=dtypes.fp4x2)
        w2_q, w2_scale = torch_quant(w2_fp, quant_dtype=dtypes.fp4x2)
        w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
        w2_q = w2_q.view(E, model_dim, inter_dim // 2)

    elif fmt_label == "fp8":
        # MXFP8: per-32 block-scaled fp8 for both activation and weight
        w1_q, w1_scale = _per_1x32_fp8_quant_weight(w1_fp)
        w2_q, w2_scale = _per_1x32_fp8_quant_weight(w2_fp)

    else:
        raise ValueError(f"Unknown format: {fmt_label}")

    # --- Call fused_moe ---
    out_ck, us = run_perftest(
        fused_moe,
        input_fp,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=quant_type,
        activation=activation,
        doweight_stage1=False,
        num_iters=5,
        num_warmup=2,
    )

    tflops = token * model_dim * inter_dim * 3 * topk * 2 / us / 1e6

    # Tolerance varies by format: quantized formats have larger errors
    if fmt_label in ("bf16", "fp16"):
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 0.1, 0.1

    err = checkAllclose(
        out_ref,
        out_ck,
        rtol=rtol,
        atol=atol,
        msg=f"gfx1250 {fmt_label}: {us:>8.2f} us, {tflops:>8.2f} tflops",
    )

    def calc_diff(x, y):
        x, y = x.double(), y.double()
        denom = (x * x + y * y).sum()
        if denom == 0:
            return 0.0
        return float(1 - 2 * (x * y).sum() / denom)

    logits_diff = calc_diff(out_ref, out_ck)
    if logits_diff > 1e-3:
        logging.warning(
            f"[{fmt_label}] logits_diff={logits_diff:.6f} is large"
        )

    return {"fmt": fmt_label, "us": us, "tflops": tflops, "err": err, "diff": logits_diff}


# ---------------------------------------------------------------------------
# Format configurations
# ---------------------------------------------------------------------------
# (label, quant_type, wq_dtype, activation, min_tokens, dtype_override)
FORMAT_CFGS = [
    ("bf16",  QuantType.No,       None,          ActivationType.Silu,    1,   dtypes.bf16),
    ("fp16",  QuantType.No,       None,          ActivationType.Silu,    1,   dtypes.fp16),
    ("fp4",   QuantType.per_1x32, dtypes.fp4x2,  ActivationType.Silu,    1,   dtypes.bf16),
    ("a8w4",  QuantType.per_1x32, dtypes.fp4x2,  ActivationType.Swiglu,  512, dtypes.bf16),
    ("fp8",   QuantType.per_1x32, dtypes.fp8,    ActivationType.Silu,    1,   dtypes.bf16),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test gfx1250 FlyDSL MOE kernels (fp4/fp8/a8w4/bf16/fp16) via fused_moe.",
)
parser.add_argument(
    "-f", "--format",
    type=str,
    choices=["bf16", "fp16", "fp4", "a8w4", "fp8", "all"],
    nargs="*",
    default=["all"],
    help="Data format(s) to test. Default: all",
)
parser.add_argument(
    "-dim",
    type=str,
    nargs="*",
    default=["7168,256"],
    help="model_dim,inter_dim pairs (e.g. -dim 7168,256 4096,1024)",
)
parser.add_argument(
    "-t", "--tokenNum",
    type=int,
    nargs="*",
    default=[1, 16, 32, 64, 128, 256, 512, 1024],
    help="Number of tokens to test",
)
parser.add_argument(
    "-e", "--expert",
    type=int,
    default=8,
    help="Number of experts",
)
parser.add_argument(
    "-k", "--topk",
    type=int,
    default=2,
    help="Top-k experts per token",
)

args = parser.parse_args()

# Parse dimension pairs
dim_pairs = []
for d in args.dim:
    parts = d.split(",")
    dim_pairs.append((int(parts[0]), int(parts[1])))

# Select format configs
if "all" in args.format:
    selected_fmts = FORMAT_CFGS
else:
    selected_fmts = [c for c in FORMAT_CFGS if c[0] in args.format]

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
df = []
for (fmt_label, q_type, wq_dtype, act_type, min_tok, dtype_override) in selected_fmts:
    for (model_dim, inter_dim) in dim_pairs:
        for m in args.tokenNum:
            if m < min_tok:
                continue
            # Dimension alignment: per_1x32 requires K divisible by 32
            if q_type == QuantType.per_1x32 and model_dim % 32 != 0:
                continue
            if q_type == QuantType.per_1x32 and inter_dim % 32 != 0:
                continue

            ret = test_gfx1250_fmoe(
                dtype_override,
                m,
                model_dim,
                inter_dim,
                args.expert,
                args.topk,
                act_type,
                q_type,
                wq_dtype,
                fmt_label,
            )
            df.append(ret)

if df:
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("gfx1250 FlyDSL MOE test summary:\n%s", df_md)
else:
    aiter.logger.info("No test cases matched the given parameters.")

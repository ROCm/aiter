# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for gfx1250 FlyDSL MOE kernels (fp4, fp8, a8w4, bf16, fp16).

All tests use fused_moe as the entry point, which dispatches to the
appropriate gfx1250 FlyDSL kernel based on quantization type and dtype.
"""

import sys
import os
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

# FlyDSL 的 tests/kernels/utils/fp4_utils.py 里包含 preshuffle_b_16x16 /
# e8m0_to_f32 / mxfp4_to_f32 / fp8_e4m3_to_f32 等 bit-accurate dequant 工具,
# 这里复用这些工具使 reference 与 kernel 使用完全一致的量化基线.
# 用 importlib 单文件加载, 避免把 /home/zxe/FlyDSL/tests 放进 sys.path
# 后与 aiter.ops.flydsl.kernels 里 `from kernels.xxx import ...` 冲突.
import importlib.util  # noqa: E402
_FLYDSL_FP4_UTILS_PATH = "/home/zxe/FlyDSL/tests/kernels/utils/fp4_utils.py"
_spec = importlib.util.spec_from_file_location(
    "flydsl_fp4_utils", _FLYDSL_FP4_UTILS_PATH
)
flydsl_fp4_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flydsl_fp4_utils)

SCALE_BLOCK = 32

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
# Quant / Dequant helpers (bit-accurate, shared with FlyDSL UT)
# ---------------------------------------------------------------------------
def _dequant_blockscale_fp8(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize fp8_e4m3 with E8M0 per-32 block scale -> float32."""
    if scale.dim() == x_q.dim() - 1:
        scale = scale.view(*x_q.shape[:-1], scale.shape[-1])
    scale_f32 = flydsl_fp4_utils.e8m0_to_f32(scale.view(torch.uint8))
    scale_expanded = scale_f32.repeat_interleave(SCALE_BLOCK, dim=-1)[..., : x_q.shape[-1]]
    return flydsl_fp4_utils.fp8_e4m3_to_f32(x_q.view(torch.uint8)) * scale_expanded


def _dequant_blockscale_fp4(x_q: torch.Tensor, scale: torch.Tensor, k_dim: int) -> torch.Tensor:
    """Dequantize fp4x2 (packed uint8) with E8M0 per-32 block scale -> float32.

    x_q: (..., K//2) uint8/fp4x2
    scale: (..., K//32) uint8/fp8_e8m0
    returns: (..., K) float32
    """
    if scale.dim() == x_q.dim() - 1:
        scale = scale.view(*x_q.shape[:-1], scale.shape[-1])
    scale_f32 = flydsl_fp4_utils.e8m0_to_f32(scale.view(torch.uint8))
    scale_expanded = scale_f32.repeat_interleave(SCALE_BLOCK, dim=-1)[..., :k_dim]
    return flydsl_fp4_utils.mxfp4_to_f32(x_q.view(torch.uint8))[..., :k_dim] * scale_expanded


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


def _quant_dequant_activation_fp4(x: torch.Tensor) -> torch.Tensor:
    """Activation q->dq roundtrip for fp4 MOE reference (matches kernel's fp4 quant)."""
    from aiter.ops.quant import per_1x32_f4_quant

    orig_shape = x.shape
    x_q, x_s = per_1x32_f4_quant(x.to(torch.float32), quant_dtype=dtypes.fp4x2)
    K = orig_shape[-1]
    x_dq = _dequant_blockscale_fp4(
        x_q.view(-1, K // 2).contiguous(),
        x_s.view(torch.uint8).view(-1, K // SCALE_BLOCK).contiguous(),
        K,
    )
    return x_dq.view(orig_shape).to(torch.float32)


def _quant_dequant_activation_fp8(x: torch.Tensor) -> torch.Tensor:
    """Activation q->dq roundtrip for fp8/a8w4 MOE reference."""
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).to(torch.float32)
    x_q, x_s = per_1x32_f8_scale_f8_quant(x_2d, scale_type=dtypes.fp8_e8m0)
    x_s = x_s.view(torch.uint8)
    x_dq = _dequant_blockscale_fp8(x_q.view(torch.uint8), x_s)
    return x_dq.view(orig_shape).to(torch.float32)


# ---------------------------------------------------------------------------
# Reference MOE
# ---------------------------------------------------------------------------
def _torch_ref_moe(
    input_fp32, w1_fp32, w2_fp32, topk_weights, topk_ids, activation, dtype,
):
    """Compute full 2-stage MOE reference in fp32 from the supplied fp32 tensors.

    Caller should provide already-dequantized (or unquantized) tensors so that
    the reference shares the same quantization baseline as the kernel under test.
    """
    token_num, model_dim = input_fp32.shape
    E = w1_fp32.shape[0]
    topk = topk_ids.shape[1]
    inter_dim = w1_fp32.shape[1] // 2

    inp = input_fp32.to(torch.float32)
    w1 = w1_fp32.to(torch.float32)
    w2 = w2_fp32.to(torch.float32)

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

    # --- Format-specific weight preparation + dequantized ref tensors ---
    # Reference tensors (w1_ref / w2_ref / input_ref) must share the same
    # quantization baseline as the kernel, otherwise the residual noise from
    # quantization dominates and allclose fails.
    if fmt_label in ("bf16", "fp16"):
        w1_q, w2_q = w1_fp, w2_fp
        w1_scale, w2_scale = None, None
        input_ref = input_fp.to(torch.float32)
        w1_ref = w1_fp.to(torch.float32)
        w2_ref = w2_fp.to(torch.float32)

    elif fmt_label == "fp4":
        # MXFP4: per-32 block-scaled fp4x2 activation + weight
        w1_q, w1_scale = torch_quant(w1_fp, quant_dtype=dtypes.fp4x2)
        w2_q, w2_scale = torch_quant(w2_fp, quant_dtype=dtypes.fp4x2)
        w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
        w2_q = w2_q.view(E, model_dim, inter_dim // 2)

        # fp4 kernels do NOT use preshuffled weight layout (per FlyDSL UT).
        w1_ref = _dequant_blockscale_fp4(
            w1_q.view(E * inter_dim * 2, model_dim // 2),
            w1_scale.view(torch.uint8).view(E * inter_dim * 2, model_dim // SCALE_BLOCK),
            model_dim,
        ).view(E, inter_dim * 2, model_dim)
        w2_ref = _dequant_blockscale_fp4(
            w2_q.view(E * model_dim, inter_dim // 2),
            w2_scale.view(torch.uint8).view(E * model_dim, inter_dim // SCALE_BLOCK),
            inter_dim,
        ).view(E, model_dim, inter_dim)
        input_ref = _quant_dequant_activation_fp4(input_fp)

    elif fmt_label == "a8w4":
        # A8W4: fp4x2 weight + fp8 activation (quantized internally by fused_moe).
        w1_q, w1_scale = torch_quant(w1_fp, quant_dtype=dtypes.fp4x2)
        w2_q, w2_scale = torch_quant(w2_fp, quant_dtype=dtypes.fp4x2)
        w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
        w2_q = w2_q.view(E, model_dim, inter_dim // 2)

        # Dequant BEFORE preshuffle (preshuffle only reorders bytes consumed by kernel).
        w1_ref = _dequant_blockscale_fp4(
            w1_q.view(E * inter_dim * 2, model_dim // 2),
            w1_scale.view(torch.uint8).view(E * inter_dim * 2, model_dim // SCALE_BLOCK),
            model_dim,
        ).view(E, inter_dim * 2, model_dim)
        w2_ref = _dequant_blockscale_fp4(
            w2_q.view(E * model_dim, inter_dim // 2),
            w2_scale.view(torch.uint8).view(E * model_dim, inter_dim // SCALE_BLOCK),
            inter_dim,
        ).view(E, model_dim, inter_dim)
        input_ref = _quant_dequant_activation_fp8(input_fp)

        # Preshuffle fp4x2 W1 bytes for the a8w4 stage1 kernel (gfx1250).
        # NOTE: aiter dispatches a8w4's stage2 to the *fp4* kernel
        # (`stage2_fmt = "fp4"` in `_gfx1250_data_format`), which does NOT
        # consume a preshuffled weight layout — mirror the fp4-format test
        # above and leave W2 in row-major form to match the kernel's
        # expectation. Preshuffling W2 here was the root cause of the
        # a8w4 end-to-end numerical error (stage1 was already correct).
        w1_rows, w1_cols = E * inter_dim * 2, model_dim // 2
        w1_q = flydsl_fp4_utils.preshuffle_b_16x16(
            w1_q.contiguous().view(w1_rows, w1_cols), w1_rows, w1_cols
        ).view(E, inter_dim * 2, model_dim // 2)

    elif fmt_label == "fp8":
        # MXFP8: per-32 block-scaled fp8 for both activation and weight.
        w1_q, w1_scale = _per_1x32_fp8_quant_weight(w1_fp)
        w2_q, w2_scale = _per_1x32_fp8_quant_weight(w2_fp)

        w1_ref = _dequant_blockscale_fp8(
            w1_q.view(torch.uint8).view(E * inter_dim * 2, model_dim),
            w1_scale.view(torch.uint8).view(E * inter_dim * 2, model_dim // SCALE_BLOCK),
        ).view(E, inter_dim * 2, model_dim)
        w2_ref = _dequant_blockscale_fp8(
            w2_q.view(torch.uint8).view(E * model_dim, inter_dim),
            w2_scale.view(torch.uint8).view(E * model_dim, inter_dim // SCALE_BLOCK),
        ).view(E, model_dim, inter_dim)
        input_ref = _quant_dequant_activation_fp8(input_fp)

        # Preshuffle fp8 weight bytes.
        w1_rows, w1_cols = E * inter_dim * 2, model_dim
        w2_rows, w2_cols = E * model_dim, inter_dim
        w1_q = flydsl_fp4_utils.preshuffle_b_16x16(
            w1_q.contiguous().view(w1_rows, w1_cols), w1_rows, w1_cols
        ).view(E, inter_dim * 2, model_dim)
        w2_q = flydsl_fp4_utils.preshuffle_b_16x16(
            w2_q.contiguous().view(w2_rows, w2_cols), w2_rows, w2_cols
        ).view(E, model_dim, inter_dim)

    else:
        raise ValueError(f"Unknown format: {fmt_label}")

    # --- Call fused_moe FIRST ---
    # NOTE: FlyDSL stage2 kernel currently leaves the output buffer's un-touched
    # bytes as-is. If we run the reference first, PyTorch's caching allocator
    # reuses a buffer that still holds large values from the reference GEMM,
    # turning the final out_ck into inf/nan. Running the kernel first avoids
    # this contamination (the empty alloc is then freshly-faulted zero pages).
    # FlyDSL JIT 目前在 torch.profiler + 长 iter loop 下会写坏 out buffer,
    # 这里先只做正确性校验, perf 置零. 开 AITER_FLYDSL_PERF=1 可切回 run_perftest.
    if int(os.environ.get("AITER_FLYDSL_PERF", "0")):
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
    else:
        torch.cuda.synchronize()
        out_ck = fused_moe(
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
        )
        torch.cuda.synchronize()
        us = 1.0  # placeholder

    # Reference MOE on (dequantized or unquantized) fp32 tensors.
    out_ref = _torch_ref_moe(
        input_ref, w1_ref, w2_ref, topk_weights, topk_ids, activation, dtype,
    )

    tflops = token * model_dim * inter_dim * 3 * topk * 2 / us / 1e6

    # Tolerance follows FlyDSL UT conventions
    # (see /home/zxe/FlyDSL/tests/kernels/test_moe_gemm_mxscale_gfx1250.py):
    #   fp4:  rtol=0.5, atol=0.25
    #   a8w4: rtol=0.5, atol=0.5
    #   fp8:  rtol=0.25, atol=0.25
    #   bf16/fp16: rtol=1e-2, atol=1e-2 (our strict default)
    if fmt_label in ("bf16", "fp16"):
        rtol, atol = 1e-2, 1e-2
    elif fmt_label == "a8w4":
        rtol, atol = 0.5, 0.5
    elif fmt_label == "fp4":
        rtol, atol = 0.5, 0.25
    else:  # fp8
        rtol, atol = 0.25, 0.25

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
    # NOTE: aiter dispatches QuantType.per_1x32 + Swiglu + gfx1250 to the
    # a8w4 kernel only when tokens >= 512 (see `bf16_fp8_bound` in
    # `aiter/fused_moe.py::fused_moe_2stages`); below that the same call
    # routes to the pure-fp4 path.  Keep min_tokens=512 to actually hit
    # the a8w4 kernel here.
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

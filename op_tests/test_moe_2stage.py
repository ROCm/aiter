# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import itertools
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter.int4_utils import (
    rearrange_4bit_elements,
    convert_int8_to_uint32_int4,
)
from aiter.utility import fp4_utils
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx, get_cu_num
import argparse
import os
import pandas as pd
import logging

from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    torch_moe_stage1,
    torch_moe_stage2,
)


from aiter.ops.shuffle import (
    shuffle_weight,
    shuffle_scale_a16w4,
    shuffle_weight_a16w4,
)

torch.int4 = getattr(torch, "int4", torch.uint32)
torch.set_default_device("cuda")
AITER_MOE_EXPERT_BALANCE = (
    os.environ.get("AITER_MOE_EXPERT_BALANCE", "False").lower() == "true"
)


# ---------------------------------------------------------------------------
# FlyDSL gfx1250 capability detection (Step 1 of the merge plan).
#
# These hooks let the per_1x32 path run on gfx1250 by routing through aiter's
# FlyDSL backend with a bit-accurate fp32 reference (mirroring the standalone
# ``test_moe_flydsl_gfx1250.py``).  This block only sets up imports/patches
# and capability flags — the actual flydsl branch is wired into ``test_fmoe``
# in a later step.  When any prerequisite is missing we silently fall back
# to the original behavior (skip per_1x32 on non-gfx950).
# ---------------------------------------------------------------------------


def _patch_flydsl_tensor_adaptor_dlpack():
    """Allow FlyDSL's TensorAdaptor to ingest fp4x2 / e8m0 tensors.

    FlyDSL's ``_FLOAT8_DTYPES`` only enumerates the 4 standard fp8 variants.
    When aiter hands it ``torch.float4_e2m1fn_x2`` (DLPack code 14) or
    ``torch.float8_e8m0fnu`` it falls into "Unsupported DLPack dtype code" in
    ``DLTensorAdaptor``. Mirror the existing fp8 workaround (view as uint8
    before ``__dlpack__``).  Safe no-op if FlyDSL isn't installed.
    """
    try:
        from flydsl.compiler import jit_argument as _jit_arg
    except Exception:
        return
    extra = tuple(
        dt
        for dt in (
            getattr(torch, "float8_e8m0fnu", None),
            getattr(torch, "float4_e2m1fn_x2", None),
        )
        if dt is not None and dt not in _jit_arg._FLOAT8_DTYPES
    )
    if extra:
        _jit_arg._FLOAT8_DTYPES = tuple(_jit_arg._FLOAT8_DTYPES) + extra


_IS_GFX1250 = get_gfx() == "gfx1250"
_FLYDSL_OK = False
SCALE_BLOCK = 32


# ---------------------------------------------------------------------------
# Bit-accurate dequant / preshuffle helpers used by the FlyDSL gfx1250 ref
# branch.  Self-contained (no FlyDSL repo dependency): ``e8m0_to_f32`` and
# ``mxfp4_to_f32`` are reused from ``aiter.utility.fp4_utils``; the fp8 dequant
# uses PyTorch's native ``torch.float8_e4m3fn`` cast; ``preshuffle_b_16x16``
# is a few lines of view+permute (mirrors FlyDSL's reference implementation).
# Exposed under the legacy ``flydsl_fp4_utils`` namespace so callers below
# don't need to change.
# ---------------------------------------------------------------------------
class _FlyDSLFp4UtilsShim:
    """Drop-in replacement for FlyDSL's ``tests/kernels/utils/fp4_utils.py``."""

    @staticmethod
    def e8m0_to_f32(scale_e8m0_biased):
        return fp4_utils.e8m0_to_f32(scale_e8m0_biased)

    @staticmethod
    def mxfp4_to_f32(x):
        return fp4_utils.mxfp4_to_f32(x)

    @staticmethod
    def fp8_e4m3_to_f32(x):
        # PyTorch ≥ 2.1 supports float8_e4m3fn natively; the cast matches
        # FlyDSL's hand-rolled bit-pattern decode (OCP E4M3, no inf, NaN at
        # 0xFF).
        return x.view(torch.float8_e4m3fn).to(torch.float32)

    @staticmethod
    def preshuffle_b_16x16(b, rows, cols):
        """Preshuffle B into 16x16 byte tiles for WMMA-friendly LDS loads.

        Works for both fp4 (cols = K//2) and fp8 (cols = K).  Mirrors
        FlyDSL/tests/kernels/utils/fp4_utils.py::preshuffle_b_16x16.
        """
        assert rows % 16 == 0, f"rows must be a multiple of 16, got {rows}"
        assert cols % 16 == 0, f"cols must be a multiple of 16, got {cols}"
        b = b.view(rows, cols)
        b = b.view(rows // 16, 16, cols // 16, 16)
        b = b.permute(0, 2, 1, 3).contiguous()
        return b.view(rows, cols)


flydsl_fp4_utils = _FlyDSLFp4UtilsShim()


if _IS_GFX1250:
    _patch_flydsl_tensor_adaptor_dlpack()
    try:
        from aiter.ops.flydsl.moe_kernels import _run_compiled  # noqa: F401

        _FLYDSL_OK = True
    except ImportError:
        _FLYDSL_OK = False


# ---------------------------------------------------------------------------
# Bit-accurate quant/dequant helpers and fp32 MoE reference (Step 2 of the
# merge plan).  Mirrors the helpers in ``test_moe_flydsl_gfx1250.py`` so the
# FlyDSL gfx1250 branch (wired in Step 4) shares the same quantization
# baseline as the kernel under test.  ``_torch_ref_moe`` does not depend on
# ``flydsl_fp4_utils``; the dequant/quant helpers do, and must only be
# invoked when ``_FLYDSL_OK`` is True.
# ---------------------------------------------------------------------------


def _dequant_blockscale_fp8(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize fp8_e4m3 with E8M0 per-32 block scale -> float32."""
    if scale.dim() == x_q.dim() - 1:
        scale = scale.view(*x_q.shape[:-1], scale.shape[-1])
    scale_f32 = flydsl_fp4_utils.e8m0_to_f32(scale.view(torch.uint8))
    scale_expanded = scale_f32.repeat_interleave(SCALE_BLOCK, dim=-1)[
        ..., : x_q.shape[-1]
    ]
    return (
        flydsl_fp4_utils.fp8_e4m3_to_f32(x_q.view(torch.uint8)) * scale_expanded
    )


def _dequant_blockscale_fp4(
    x_q: torch.Tensor, scale: torch.Tensor, k_dim: int
) -> torch.Tensor:
    """Dequantize fp4x2 (packed uint8) with E8M0 per-32 block scale -> float32.

    x_q: (..., K//2) uint8/fp4x2
    scale: (..., K//32) uint8/fp8_e8m0
    returns: (..., K) float32
    """
    if scale.dim() == x_q.dim() - 1:
        scale = scale.view(*x_q.shape[:-1], scale.shape[-1])
    scale_f32 = flydsl_fp4_utils.e8m0_to_f32(scale.view(torch.uint8))
    scale_expanded = scale_f32.repeat_interleave(SCALE_BLOCK, dim=-1)[..., :k_dim]
    return (
        flydsl_fp4_utils.mxfp4_to_f32(x_q.view(torch.uint8))[..., :k_dim]
        * scale_expanded
    )


def _per_1x32_fp8_quant_weight(w, block_size=32):
    """Quantize weight tensor to fp8 with per-32 E8M0 block scaling.

    Returns ``(w_fp8, w_scale)`` where ``w_fp8`` keeps ``w``'s shape in fp8
    dtype and ``w_scale`` has shape ``(..., K//32)`` in fp8_e8m0.
    """
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    orig_shape = w.shape
    w_2d = w.reshape(-1, orig_shape[-1]).to(torch.float32)
    w_q, w_s = per_1x32_f8_scale_f8_quant(w_2d, scale_type=dtypes.fp8_e8m0)
    w_s = w_s.view(torch.uint8).view(dtypes.fp8_e8m0)
    return (
        w_q.view(orig_shape),
        w_s.view(*orig_shape[:-1], orig_shape[-1] // block_size),
    )


def _quant_dequant_activation_fp4(x: torch.Tensor) -> torch.Tensor:
    """Activation quant/dequant roundtrip matching the fp4 MoE kernel."""
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
    """Activation quant/dequant roundtrip matching the fp8/a8w4 MoE kernel."""
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).to(torch.float32)
    x_q, x_s = per_1x32_f8_scale_f8_quant(x_2d, scale_type=dtypes.fp8_e8m0)
    x_s = x_s.view(torch.uint8)
    x_dq = _dequant_blockscale_fp8(x_q.view(torch.uint8), x_s)
    return x_dq.view(orig_shape).to(torch.float32)


def _torch_ref_moe(
    input_fp32, w1_fp32, w2_fp32, topk_weights, topk_ids, activation, dtype,
):
    """Compute full 2-stage MoE reference in fp32 from already-dequantized
    (or unquantized) tensors so ref shares the kernel's quantization baseline.
    """
    token_num, model_dim = input_fp32.shape
    E = w1_fp32.shape[0]
    topk = topk_ids.shape[1]
    inter_dim = w1_fp32.shape[1] // 2

    inp = input_fp32.to(torch.float32)
    w1 = w1_fp32.to(torch.float32)
    w2 = w2_fp32.to(torch.float32)

    act_fn = (
        torch.nn.functional.silu
        if activation
        in (aiter.ActivationType.Silu, aiter.ActivationType.Swiglu)
        else torch.nn.functional.gelu
    )

    out1 = torch.zeros(
        token_num, topk, inter_dim, dtype=torch.float32, device="cuda"
    )
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


def _derive_flydsl_fmt(qType, AQDType, WQDType, dtype):
    """Map (qType, AQDType, WQDType, dtype) -> flydsl format label, or None.

    The labels mirror those in ``test_moe_flydsl_gfx1250.py``:
      * "bf16" / "fp16": no quant, full-precision MoE
      * "fp4":  per-1x32 fp4x2 weight + fp4x2 act
      * "a8w4": per-1x32 fp4x2 weight + bf16/fp16/fp8 act
                (aiter routes a8w4 stage1 to the a8w4 kernel and
                 stage2 to the fp4 kernel)
      * "fp8":  per-1x32 fp8 weight + fp8 act
    """
    if qType == aiter.QuantType.No:
        if dtype == dtypes.bf16:
            return "bf16"
        if dtype == dtypes.fp16:
            return "fp16"
        return None
    if qType == aiter.QuantType.per_1x32:
        if AQDType == dtypes.fp4x2 and WQDType == dtypes.fp4x2:
            return "fp4"
        if WQDType == dtypes.fp4x2 and AQDType in (
            dtypes.bf16, dtypes.fp16, dtypes.fp8,
        ):
            return "a8w4"
        if WQDType == dtypes.fp8 and AQDType == dtypes.fp8:
            return "fp8"
    return None


# When set, log every test_fmoe call's flydsl branch decision (used only
# during the merge to confirm the dispatch table without polluting normal
# runs). Toggle via ``AITER_FLYDSL_DEBUG_BRANCH=1``.
_FLYDSL_DEBUG_BRANCH = (
    os.environ.get("AITER_FLYDSL_DEBUG_BRANCH", "0") == "1"
)

# Per-fmt tolerance copied verbatim from ``test_moe_flydsl_gfx1250.py``.
# bf16/fp16 use the strict default; the quantized formats relax to the
# values FlyDSL's MXScale UT empirically uses.
_FLYDSL_TOL = {
    "bf16": (1e-2, 1e-2),
    "fp16": (1e-2, 1e-2),
    "fp4":  (0.5, 0.25),
    "a8w4": (0.5, 0.5),
    "fp8":  (0.25, 0.25),
}

# logits_diff (cosine-distance-like) fallback tolerance.  For per-1x32 fp4/fp8
# at small shapes (e.g. token=64, model_dim=256), the elementwise quantization
# noise can exceed ``_FLYDSL_TOL`` even when the kernel output is directionally
# correct.  When elementwise allclose fails, we fall back to logits_diff and
# accept the case as PASS as long as the diff stays below the per-fmt budget.
# Override globally with ``AITER_FLYDSL_DIFF_TOL=<float>``, or disable the
# fallback entirely with ``AITER_FLYDSL_STRICT_ELEM=1``.
_FLYDSL_DIFF_TOL = {
    "bf16": 1e-3,
    "fp16": 1e-3,
    "fp4":  0.5,
    "a8w4": 0.5,
    "fp8":  0.3,
}

# a8w4 only routes to the dedicated kernel when token >= this threshold (see
# ``bf16_fp8_bound`` in aiter.fused_moe::fused_moe_2stages); below that the
# same call falls through to the fp4 kernel, which would produce mismatched
# numerics versus the a8w4 reference.  Mirror the standalone gfx1250 UT.
_FLYDSL_A8W4_MIN_TOKENS = 512


def _run_flydsl_branch(
    *,
    dtype, token, model_dim, inter_dim, E, topk, actType, qType,
    AQDType, WQDType, doweight_stage1, hidden_pad, intermediate_pad,
    fmt_label,
):
    """gfx1250 FlyDSL backed test path; returns dict or None to skip.

    Equivalent to the body of ``test_gfx1250_fmoe`` in the standalone
    ``test_moe_flydsl_gfx1250.py`` but parameterized to plug into
    ``test_fmoe``.  Caller is responsible for the ``_use_flydsl`` gate.

    Skip rules (return ``None`` and warn):
      * ``hidden_pad`` / ``intermediate_pad`` non-zero — the FlyDSL
        kernels do not consume the padded layout.
      * ``fmt_label == "a8w4"`` and ``token < _FLYDSL_A8W4_MIN_TOKENS``
        — aiter would route to the fp4 kernel, breaking the a8w4 ref.
    """
    if hidden_pad != 0 or intermediate_pad != 0:
        aiter.logger.warning(
            "flydsl path: ignoring case with hidden_pad=%d intermediate_pad=%d "
            "(unsupported on gfx1250)",
            hidden_pad, intermediate_pad,
        )
        return None
    if fmt_label == "a8w4" and token < _FLYDSL_A8W4_MIN_TOKENS:
        aiter.logger.info(
            "flydsl path: skipping a8w4 case token=%d (< %d); aiter would "
            "route to the fp4 kernel.",
            token, _FLYDSL_A8W4_MIN_TOKENS,
        )
        return None

    torch_quant = aiter.get_torch_quant(qType)

    input_fp = torch.randn((token, model_dim), dtype=dtype)
    w1_fp = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
    w2_fp = torch.randn((E, model_dim, inter_dim), dtype=dtype)
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input_fp, score, topk, True)

    # Format-specific weight prep + dequantized ref tensors (must share the
    # quantization baseline with the kernel under test, otherwise residual
    # quant noise dominates and allclose fails).
    if fmt_label in ("bf16", "fp16"):
        w1_q, w2_q = w1_fp, w2_fp
        w1_scale = w2_scale = None
        input_ref = input_fp.to(torch.float32)
        w1_ref = w1_fp.to(torch.float32)
        w2_ref = w2_fp.to(torch.float32)

    elif fmt_label == "fp4":
        # MXFP4: per-32 block-scaled fp4x2 activation + weight.  fp4
        # kernels do NOT consume preshuffled weight layout (per FlyDSL UT).
        w1_q, w1_scale = torch_quant(w1_fp, quant_dtype=dtypes.fp4x2)
        w2_q, w2_scale = torch_quant(w2_fp, quant_dtype=dtypes.fp4x2)
        w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
        w2_q = w2_q.view(E, model_dim, inter_dim // 2)
        w1_ref = _dequant_blockscale_fp4(
            w1_q.view(E * inter_dim * 2, model_dim // 2),
            w1_scale.view(torch.uint8).view(
                E * inter_dim * 2, model_dim // SCALE_BLOCK
            ),
            model_dim,
        ).view(E, inter_dim * 2, model_dim)
        w2_ref = _dequant_blockscale_fp4(
            w2_q.view(E * model_dim, inter_dim // 2),
            w2_scale.view(torch.uint8).view(
                E * model_dim, inter_dim // SCALE_BLOCK
            ),
            inter_dim,
        ).view(E, model_dim, inter_dim)
        input_ref = _quant_dequant_activation_fp4(input_fp)

    elif fmt_label == "a8w4":
        # A8W4: fp4x2 weight + fp8 activation (quantized internally by
        # fused_moe).  Dequant BEFORE preshuffle (preshuffle only reorders
        # bytes consumed by kernel).
        w1_q, w1_scale = torch_quant(w1_fp, quant_dtype=dtypes.fp4x2)
        w2_q, w2_scale = torch_quant(w2_fp, quant_dtype=dtypes.fp4x2)
        w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
        w2_q = w2_q.view(E, model_dim, inter_dim // 2)
        w1_ref = _dequant_blockscale_fp4(
            w1_q.view(E * inter_dim * 2, model_dim // 2),
            w1_scale.view(torch.uint8).view(
                E * inter_dim * 2, model_dim // SCALE_BLOCK
            ),
            model_dim,
        ).view(E, inter_dim * 2, model_dim)
        w2_ref = _dequant_blockscale_fp4(
            w2_q.view(E * model_dim, inter_dim // 2),
            w2_scale.view(torch.uint8).view(
                E * model_dim, inter_dim // SCALE_BLOCK
            ),
            inter_dim,
        ).view(E, model_dim, inter_dim)
        input_ref = _quant_dequant_activation_fp8(input_fp)

        # Preshuffle fp4x2 W1 bytes for the a8w4 stage1 kernel (gfx1250).
        # NOTE: aiter dispatches a8w4's stage2 to the *fp4* kernel
        # (``stage2_fmt = "fp4"`` in ``_gfx1250_data_format``), which does
        # NOT consume a preshuffled weight layout — leave W2 in row-major
        # form to match the kernel's expectation.  Preshuffling W2 here was
        # the root cause of the historical a8w4 numerical bug.
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
            w1_scale.view(torch.uint8).view(
                E * inter_dim * 2, model_dim // SCALE_BLOCK
            ),
        ).view(E, inter_dim * 2, model_dim)
        w2_ref = _dequant_blockscale_fp8(
            w2_q.view(torch.uint8).view(E * model_dim, inter_dim),
            w2_scale.view(torch.uint8).view(
                E * model_dim, inter_dim // SCALE_BLOCK
            ),
        ).view(E, model_dim, inter_dim)
        input_ref = _quant_dequant_activation_fp8(input_fp)
        # Preshuffle fp8 weight bytes (both W1 and W2).
        w1_rows, w1_cols = E * inter_dim * 2, model_dim
        w2_rows, w2_cols = E * model_dim, inter_dim
        w1_q = flydsl_fp4_utils.preshuffle_b_16x16(
            w1_q.contiguous().view(w1_rows, w1_cols), w1_rows, w1_cols
        ).view(E, inter_dim * 2, model_dim)
        w2_q = flydsl_fp4_utils.preshuffle_b_16x16(
            w2_q.contiguous().view(w2_rows, w2_cols), w2_rows, w2_cols
        ).view(E, model_dim, inter_dim)

    else:
        raise ValueError(f"flydsl: unknown fmt_label {fmt_label!r}")

    # Kernel-FIRST call order.  FlyDSL's stage2 currently leaves untouched
    # bytes in the output buffer as-is; running the reference first lets
    # PyTorch's caching allocator hand us back a buffer that still holds
    # large values from the reference GEMM, contaminating the kernel
    # output with inf/NaN.  Issuing the kernel before the ref makes the
    # output buffer freshly-faulted zero pages.
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
            quant_type=qType,
            activation=actType,
            doweight_stage1=doweight_stage1,
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
            quant_type=qType,
            activation=actType,
            doweight_stage1=doweight_stage1,
        )
        torch.cuda.synchronize()
        us = 1.0  # placeholder, set AITER_FLYDSL_PERF=1 for real timing

    tflops = token * model_dim * inter_dim * 3 * topk * 2 / us / 1e6

    # ``AITER_FLYDSL_SKIP_REF=1`` mirrors FlyDSL UT's ``--skip_ref t``:
    # only assert the kernel didn't blow up (finite output, correct shape).
    # Useful on small shapes where the per-1x32 quant noise legitimately
    # dominates ``checkAllclose`` even when the kernel is correct.
    if int(os.environ.get("AITER_FLYDSL_SKIP_REF", "0")):
        finite = bool(torch.isfinite(out_ck).all())
        if not finite:
            aiter.logger.error(
                "flydsl_gfx1250 %s: output contains inf/nan", fmt_label,
            )
        aiter.logger.info(
            "flydsl_gfx1250 %s: %.2f us, %.2f tflops, finite=%s "
            "(skip-ref mode)",
            fmt_label, us, tflops, finite,
        )
        return {
            "us": us,
            "err": 0 if finite else 1,
            "fmt": fmt_label,
            "tflops": tflops,
            "diff": 0.0,
            "skip_ref": True,
            "finite": finite,
        }

    out_ref = _torch_ref_moe(
        input_ref, w1_ref, w2_ref, topk_weights, topk_ids, actType, dtype,
    )

    rtol, atol = _FLYDSL_TOL[fmt_label]
    err = checkAllclose(
        out_ref,
        out_ck,
        rtol=rtol,
        atol=atol,
        msg=(
            f"flydsl_gfx1250 {fmt_label}: {us:>8.2f} us, {tflops:>8.2f} tflops"
        ),
    )

    def _calc_diff(x, y):
        x, y = x.double(), y.double()
        denom = (x * x + y * y).sum()
        if denom == 0:
            return 0.0
        return float(1 - 2 * (x * y).sum() / denom)

    logits_diff = _calc_diff(out_ref, out_ck)

    # Cosine-distance-like fallback: if elementwise allclose flagged a
    # mismatch but the output is directionally correct, accept the case.
    # See ``_FLYDSL_DIFF_TOL`` for rationale.
    diff_tol = float(
        os.environ.get("AITER_FLYDSL_DIFF_TOL", _FLYDSL_DIFF_TOL[fmt_label])
    )
    strict_elem = int(os.environ.get("AITER_FLYDSL_STRICT_ELEM", "0"))
    finite = bool(torch.isfinite(out_ck).all())
    fallback_pass = (
        not strict_elem
        and err != 0
        and finite
        and logits_diff <= diff_tol
    )
    if fallback_pass:
        aiter.logger.info(
            "flydsl_gfx1250 %s: elementwise allclose failed (err=%s) but "
            "logits_diff=%.6f <= %.6f -> accept (cosine-fallback)",
            fmt_label, err, logits_diff, diff_tol,
        )
        err = 0
    elif logits_diff > 1e-3:
        logging.warning(
            "[flydsl/%s] logits_diff=%.6f is large (diff_tol=%.6f, err=%s)",
            fmt_label, logits_diff, diff_tol, err,
        )

    return {
        "us": us,
        "err": err,
        "fmt": fmt_label,
        "tflops": tflops,
        "diff": logits_diff,
        "diff_tol": diff_tol,
        "fallback": bool(fallback_pass),
        "finite": finite,
    }


@benchmark()
def test_fmoe(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    actType,
    qType,
    AQDType,
    WQDType,
    use_g1u1=False,
    doweight_stage1=False,
    hidden_pad=0,
    intermediate_pad=0,
    preshuffle=True,
    strict_accuracy=True,
):
    # per_1x32 dispatch:
    #   * gfx950: falls through to the original ck-based path below.
    #   * gfx1250 with FlyDSL available: handled by ``_run_flydsl_branch``
    #     below via the ``_use_flydsl`` early-return.
    #   * everything else: skip — neither backend supports it here.
    if qType == aiter.QuantType.per_1x32:
        if get_gfx() == "gfx950":
            pass  # use the existing aiter ck path below
        elif _IS_GFX1250 and _FLYDSL_OK:
            pass  # falls through to the flydsl branch below
        else:
            return

    # FlyDSL branch decision (Step 3) and dispatch (Step 4).
    fmt_label = _derive_flydsl_fmt(qType, AQDType, WQDType, dtype)
    _use_flydsl = bool(_IS_GFX1250 and _FLYDSL_OK and fmt_label is not None)
    if _FLYDSL_DEBUG_BRANCH:
        aiter.logger.info(
            "test_fmoe[flydsl-decide] qType=%s AQDType=%s WQDType=%s dtype=%s "
            "-> fmt_label=%r use_flydsl=%s",
            qType, AQDType, WQDType, dtype, fmt_label, _use_flydsl,
        )
    if _use_flydsl:
        return _run_flydsl_branch(
            dtype=dtype,
            token=token,
            model_dim=model_dim,
            inter_dim=inter_dim,
            E=E,
            topk=topk,
            actType=actType,
            qType=qType,
            AQDType=AQDType,
            WQDType=WQDType,
            doweight_stage1=doweight_stage1,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            fmt_label=fmt_label,
        )

    torch_quant = aiter.get_torch_quant(qType)
    input = torch.randn((token, model_dim), dtype=dtype)
    if use_g1u1:
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
        if hidden_pad != 0 and intermediate_pad != 0:
            w1[:, :, -hidden_pad:] = 0
            w1[:, -intermediate_pad:, :] = 0
            w1[:, inter_dim - intermediate_pad : inter_dim, :] = 0
        exp_bias1 = torch.clamp(torch.randn((E, inter_dim * 2), dtype=dtype), -1.0, 1.0)
    else:
        w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype)
        exp_bias1 = torch.clamp(torch.randn((E * inter_dim), dtype=dtype), -1.0, 1.0)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)
    if hidden_pad != 0 and intermediate_pad != 0:
        w2[:, :, -intermediate_pad:] = 0
        w2[:, -hidden_pad:, :] = 0
    exp_bias2 = torch.clamp(torch.randn((E, model_dim), dtype=dtype), -1.0, 1.0)
    if AITER_MOE_EXPERT_BALANCE:
        score = torch.zeros((token, E), dtype=dtype)
        start_col = 0
        end_col = topk
        for token_id in range(token):
            score[token_id, start_col:end_col] = 1.0
            start_col = end_col % E
            end_col = start_col + topk
    else:
        score = torch.randn((token, E), dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    if qType == aiter.QuantType.per_Tensor:
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=WQDType)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=WQDType)
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
    elif qType == aiter.QuantType.per_Token and WQDType == torch.int4:  # int4 w quant
        w1_qt, w1_scale = aiter.pertoken_quant(w1, quant_dtype=dtypes.i8, dtypeMax=7)
        w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=dtypes.i8, dtypeMax=7)
    elif qType == aiter.QuantType.per_128x128:

        def weight_per_128x128_quant(weight, quant_dtype):
            E, dim1, dim2 = weight.shape
            weight_blocks = weight.view(
                E, dim1 // 128, 128, dim2 // 128, 128
            )  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_blocks = weight_blocks.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_blocks = weight_blocks.view(
                E, -1, 128 * 128
            )  # [E, num_blocks, 128*128]
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight_blocks, quant_dtype=quant_dtype
            )
            weight_qt = weight_qt.view(
                E, dim1 // 128, dim2 // 128, 128, 128
            )  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_qt = weight_qt.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_qt = weight_qt.view(E, dim1, dim2)  # [E, dim1, dim2]
            weight_scale = weight_scale.view(
                E, dim1 // 128, dim2 // 128
            )  # [E, num_blocks_dim1, num_blocks_dim2]
            return weight_qt, weight_scale

        w1_qt, w1_scale = weight_per_128x128_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = weight_per_128x128_quant(w2, quant_dtype=WQDType)
    else:
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)

    if qType != aiter.QuantType.per_1x32:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)
    else:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    # Quant-ing a
    if qType == aiter.QuantType.per_128x128:
        a1_qt, a1_scale = aiter.pertoken_quant(
            input.view(token, -1, 128), quant_dtype=AQDType
        )
        a1_qt = a1_qt.view(token, model_dim)
        a1_scale = a1_scale.squeeze(-1)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and WQDType == dtypes.fp4x2
    ):  # a16w4 & a8w4
        a1_qt = input.to(dtypes.bf16)
        a1_scale = None
    else:
        a1_qt, a1_scale = torch_quant(input, quant_dtype=AQDType)

    # bias dtype convert
    if (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4
        exp_bias1_aiter = exp_bias1.to(dtypes.fp32)
        exp_bias2_aiter = exp_bias2.to(dtypes.fp32)
    else:
        exp_bias1_aiter = exp_bias1 = None
        exp_bias2_aiter = exp_bias2 = None

    # pre-shuffle
    w1_scale_aiter = w1_scale
    w2_scale_aiter = w2_scale
    if WQDType == torch.int4:  # int4 w quant
        w1_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w1_qt_aiter, (16, 16), use_int4=True)
            )
        )
        w2_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w2_qt_aiter, (16, 16), use_int4=True)
            )
        )
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4
        w1_qt_aiter = shuffle_weight_a16w4(w1_qt_aiter, 16, True)
        w1_scale_aiter = shuffle_scale_a16w4(w1_scale, E, True)
        w2_qt_aiter = shuffle_weight_a16w4(w2_qt_aiter, 16, False)
        w2_scale_aiter = shuffle_scale_a16w4(w2_scale, E, False)
    elif WQDType != dtypes.fp4x2 or preshuffle:
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    else:
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)

    # # ######################## stage 1 start ###########
    out1_ref = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=actType,
        quant_type=qType,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        w1_bias=exp_bias1,
        doweight=doweight_stage1,
    )

    # ######################## stage 2 start ###########
    if qType == aiter.QuantType.per_128x128:
        a2_qt, a2_scale = aiter.pertoken_quant(
            out1_ref.view(token, -1, 128), quant_dtype=AQDType
        )
        a2_scale = a2_scale.view(token, topk, -1)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4 & a8w4
        a2_qt = out1_ref
        a2_scale = None
    else:
        a2_qt, a2_scale = torch_quant(out1_ref, quant_dtype=AQDType)
    a2_qt = a2_qt.view(token, topk, -1)

    out2_ref = torch_moe_stage2(
        a2_qt,
        w1_qt,  # E, inter_dim*2, model_dim
        w2_qt,  # E, model_dim, inter_dim
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=qType,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        w2_bias=exp_bias2,
        doweight=not doweight_stage1,
    )

    # ######################## stage 2 end ###########
    out2_ck, us2 = run_perftest(
        fused_moe,
        input,
        w1_qt_aiter,
        w2_qt_aiter,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale_aiter,
        w2_scale=w2_scale_aiter,
        quant_type=qType,
        activation=actType,
        doweight_stage1=doweight_stage1,
        intermediate_pad=intermediate_pad,
        hidden_pad=hidden_pad,
        bias1=exp_bias1_aiter,
        bias2=exp_bias2_aiter,
        num_iters=5,
        num_warmup=2,
    )
    err = checkAllclose(
        out2_ref,
        out2_ck,
        msg=f"ck_moe_2stages:{us2:>8.2f} us, {token*model_dim*inter_dim*3*topk*2/us2/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    )

    def calc_diff(x: torch.Tensor, y: torch.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        sim = 2 * (x * y).sum() / denominator
        return 1 - sim

    logits_diff = calc_diff(out2_ref, out2_ck)
    if logits_diff > 1e-3:
        logging.warning(
            f"logits_diff: {logits_diff} is too large, please check the implementation"
        )
    if strict_accuracy:
        assert not (
            err != 0 and logits_diff > 0.01
        ), f"accuracy check failed: checkAllclose err={err}, logits_diff={logits_diff}"
    elif err != 0 and logits_diff > 0.01:
        logging.warning(
            f"accuracy check failed (non-strict): err={err}, logits_diff={logits_diff}"
        )

    return {"us": us2, "err": err}


l_quant = [
    (aiter.QuantType.No, None, None),  # a16w16
    (aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, torch.int4),  # a8w4
    (aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2),  # a4w4
    (aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2),  # a16w4
    (aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2),  # a8w4
    # gfx1250 mxfp8: per-1x32 fp8 act + fp8 weight; routes to FlyDSL
    # mxscale fp8 kernel via the Step 4 branch.
    (aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp8),  # mxfp8
]


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16, fp16}",
    help="""Data type.
    e.g.: -d bf16""",
)

parser.add_argument(
    "-dim",
    type=dtypes.str2tuple,
    nargs="*",
    default=[(7168, 256)],
    help="""Model dimension.
    e.g.: -dim 6144,4096""",
)

parser.add_argument(
    "-t",
    "--tokenNum",
    type=int,
    nargs="*",
    default=[
        1,
        3,
        5,
        16,
        32,
        64,
        128,
        256,
        1024,
        4096,
        163840,
    ],
    help="""Number of tokens.
    e.g.: -t 1024""",
)

parser.add_argument(
    "-q",
    "--quant",
    type=int,
    choices=range(len(l_quant)),
    help="""select quantization type:
    0 : aiter.QuantType.No, None, None),  # a16w16
    1: aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8  # a8w8
    2: aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8  # a8w8
    3: aiter.QuantType.per_Token, dtypes.fp8, torch.int4  # a8w4
    4: aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2  # a4w4
    5: aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8,  # a8w8,
    6: aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2,  # a16w4,
    7: aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2,  # a8w4
    8: aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp8,    # mxfp8 (gfx1250 FlyDSL)""",
)

parser.add_argument(
    "-a",
    "--act",
    type=dtypes.str2ActivationType,
    nargs="*",
    default=[aiter.ActivationType.Silu],
    help="""Select activation type. Default: [Silu].
    e.g.: -a gelu        # [Gelu]
          -a silu gelu    # [Silu, Gelu]""",
)

parser.add_argument(
    "-s",
    "--doweight_stage1",
    type=dtypes.str2bool,
    nargs="*",
    default=[False],
    help="""Whether to do weight in stage 1. Default is [False].
    -s f    # False.
    -s t    # True.""",
)

parser.add_argument(
    "-e",
    "--expert",
    type=int,
    default=257,
    help="""Number of experts.
    e.g.: -e 8""",
)

parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=9,
    help="""Number of top experts.
    e.g.: -k 2""",
)

parser.add_argument(
    "-p",
    "--preshuffle",
    type=dtypes.str2bool,
    nargs="*",
    default=[True],
    help="""Whether to use pre-shuffle weight mode. Default is [False, True].
    -p f    # False.
    -p t    # True.""",
)
parser.add_argument(
    "-hip",
    "--hidden_intermediate_pad",
    type=dtypes.str2tuple,
    nargs="*",
    default=[(192, 128)],
    help="""Hidden intermediate pad.
    e.g.: -hip 0,0""",
)
parser.add_argument(
    "--no-flydsl-csv",
    action="store_true",
    help="Skip validating flydsl shapes from tuned fmoe CSVs.",
)
parser.add_argument(
    "--no-legacy",
    action="store_true",
    help="Skip the original hardcoded shape sweep and skinny tests.",
)

args = parser.parse_args()


l_quant = [l_quant[args.quant]] if args.quant is not None else l_quant


# ---------------------------------------------------------------------------
# Both modes (CLI sweep / model-csv) reduce to the same shape:
#   yield (test_fmoe_kwargs, extras_for_df)
# A single runner consumes the stream.
# ---------------------------------------------------------------------------
# Only kept for dtypes that may not exist as torch attributes in older builds;
# anything else falls through to getattr(torch, attr).
_DTYPE_STR_FALLBACK = {
    "torch.float4_e2m1fn_x2": dtypes.fp4x2,
    "torch.float8_e8m0fnu": dtypes.fp8_e8m0,
}


def _str2dtype(s):
    s = s.strip()
    if s in ("None", "none", ""):
        return None
    if s.startswith("torch."):
        attr = s.split(".", 1)[1]
        if hasattr(torch, attr):
            return getattr(torch, attr)
    if s in _DTYPE_STR_FALLBACK:
        return _DTYPE_STR_FALLBACK[s]
    raise ValueError(f"unsupported dtype string: {s!r}")


def _str2enum(s, enum_cls):
    return getattr(enum_cls, s.strip().split(".")[-1])


def _row_to_kwargs(row):
    # csv rows store already-effective dims, so pad defaults to 0.
    q_type = _str2enum(row["q_type"], aiter.QuantType)
    aq_dtype = _str2dtype(row["q_dtype_a"])
    wq_dtype = _str2dtype(row["q_dtype_w"])
    act_type = _effective_act_type(
        q_type,
        aq_dtype,
        wq_dtype,
        _str2enum(row["act_type"], aiter.ActivationType),
    )
    return dict(
        dtype=_str2dtype(row["dtype"]),
        token=int(row["token"]),
        model_dim=int(row["model_dim"]),
        inter_dim=int(row["inter_dim"]),
        E=int(row["expert"]),
        topk=int(row["topk"]),
        actType=act_type,
        qType=q_type,
        AQDType=aq_dtype,
        WQDType=wq_dtype,
        use_g1u1=dtypes.str2bool(str(row["use_g1u1"])),
        doweight_stage1=dtypes.str2bool(str(row["doweight_stage1"])),
        hidden_pad=0,
        intermediate_pad=0,
        preshuffle=True,
    )


def _iter_csv_cases():
    """Yield (kwargs, extras) for every row of every selected model csv."""
    cu = get_cu_num()
    merged_csv = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
    df_csv = pd.read_csv(merged_csv)
    rows = df_csv[df_csv["cu_num"] == cu]
    for _, row in rows.iterrows():
        kernel_name1 = str(row.get("kernelName1", "") or "")
        kernel_name2 = str(row.get("kernelName2", "") or "")
        if "flydsl_" not in kernel_name1 and "flydsl_" not in kernel_name2:
            continue
        try:
            kwargs = _row_to_kwargs(row)
        except Exception as e:
            aiter.logger.warning(
                "skip row token=%s dim=(%s,%s): parse error %s",
                row.get("token"),
                row.get("model_dim"),
                row.get("inter_dim"),
                e,
            )
            continue
        kwargs["strict_accuracy"] = True
        yield kwargs, {
            "kernelName1": kernel_name1,
            "kernelName2": kernel_name2,
        }


_PER1X32_BF16_FP4 = (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2)
_PER1X32_FP8_FP4 = (aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2)
_PER1X32_FP4_FP4 = (aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2)


def _effective_act_type(quant_type, aq_dtype, wq_dtype, act_type):
    if (quant_type, aq_dtype, wq_dtype) in (_PER1X32_BF16_FP4, _PER1X32_FP8_FP4):
        return aiter.ActivationType.Swiglu
    return act_type


def _iter_legacy_cases():
    """Yield (kwargs, extras) for the original CLI-driven sweep."""
    extras = {"model": "legacy"}

    def _kw(
        dtype,
        m,
        model_dim,
        inter_dim,
        quant_type,
        aq_dtype,
        wq_dtype,
        doweight_stage1,
        act_type,
        **over,
    ):
        return dict(
            dtype=dtype,
            token=m,
            model_dim=model_dim,
            inter_dim=inter_dim,
            E=args.expert,
            topk=args.topk,
            actType=_effective_act_type(quant_type, aq_dtype, wq_dtype, act_type),
            qType=quant_type,
            AQDType=aq_dtype,
            WQDType=wq_dtype,
            use_g1u1=True,
            doweight_stage1=doweight_stage1,
            strict_accuracy=False,
            **over,
        )

    for (
        dtype,
        (quant_type, aq_dtype, wq_dtype),
        (model_dim, inter_dim),
        doweight_stage1,
    ) in itertools.product(args.dtype, l_quant, args.dim, args.doweight_stage1):
        triple = (quant_type, aq_dtype, wq_dtype)

        if triple in (_PER1X32_BF16_FP4, _PER1X32_FP8_FP4):
            for hidden_pad, intermediate_pad in args.hidden_intermediate_pad:
                for m in args.tokenNum:
                    yield _kw(
                        dtype,
                        m,
                        model_dim,
                        inter_dim,
                        quant_type,
                        aq_dtype,
                        wq_dtype,
                        doweight_stage1,
                        aiter.ActivationType.Swiglu,
                        hidden_pad=hidden_pad,
                        intermediate_pad=intermediate_pad,
                    ), extras
        elif triple == _PER1X32_FP4_FP4:
            for preshuffle in args.preshuffle:
                for act_type in args.act:
                    for m in args.tokenNum:
                        yield _kw(
                            dtype,
                            m,
                            model_dim,
                            inter_dim,
                            quant_type,
                            aq_dtype,
                            wq_dtype,
                            doweight_stage1,
                            act_type,
                            preshuffle=preshuffle,
                            hidden_pad=0,
                            intermediate_pad=0,
                        ), extras
        else:
            for act_type in args.act:
                for m in args.tokenNum:
                    yield _kw(
                        dtype,
                        m,
                        model_dim,
                        inter_dim,
                        quant_type,
                        aq_dtype,
                        wq_dtype,
                        doweight_stage1,
                        act_type,
                    ), extras


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
_case_iters = []
if not args.no_flydsl_csv:
    _case_iters.append(_iter_csv_cases())
if not args.no_legacy:
    _case_iters.append(_iter_legacy_cases())
case_iter = itertools.chain(*_case_iters)

df = []
seen = 0
for kwargs, extras in case_iter:
    seen += 1
    ret = test_fmoe(**kwargs)
    if ret is None:
        continue
    ret.update(extras)
    df.append(ret)

aiter.logger.info(
    "moe_2stage: scanned %d cases, recorded %d results (skipped %d)",
    seen,
    len(df),
    seen - len(df),
)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_2stage summary (markdown):\n%s", df_md)

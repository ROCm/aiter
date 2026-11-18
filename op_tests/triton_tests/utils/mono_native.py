# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Frozen native excerpts for regression tests. Do not refactor these functions.
# See mono_native_manifest.json for source commits and excerpt hashes.

from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Resolve the frozen ATOM dtype binding for the only accepted target, gfx1250.
dtypes = SimpleNamespace(fp8=torch.float8_e4m3fn)

_QUANT_BLOCK_SIZE = 32


@triton.jit
def _mxfp8_quant_op(
    x_grouped,
    QUANT_AXIS: tl.constexpr,
    LOG2_DTYPE_MAX: tl.constexpr = 8,
    AMAX_FLOOR: tl.constexpr = 0.0,
):
    """Shared MXFP8 (1x32 e8m0) scale derivation.

    Given a fp32 tile where the QUANT_AXIS dim is sized QUANT_BLOCK_SIZE (=32),
    returns (scale_e8m0, quant_scale): the per-group uint8 e8m0 scale and the
    matching fp32 multiplicative scale. Both outputs keep QUANT_AXIS with size 1
    so they broadcast against the input for in-place quantization.
    AMAX_FLOOR > 0 floors each group's amax (none by default).
    """
    amax = tl.max(tl.abs(x_grouped), axis=QUANT_AXIS, keep_dims=True)
    if AMAX_FLOOR > 0:
        amax = tl.maximum(amax, AMAX_FLOOR)
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_i32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_p2 = amax_i32.to(tl.float32, bitcast=True)
    scale_unbiased = tl.log2(amax_p2).floor() - LOG2_DTYPE_MAX
    scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(tl.int32) + 127).to(tl.uint8)
    quant_scale = tl.exp2(-scale_unbiased)
    return scale_e8m0, quant_scale


_fused_dual_rmsnorm_mxfp8_quant_repr = make_kernel_repr(
    "_fused_dual_rmsnorm_mxfp8_quant_kernel",
    [
        "BLOCK_SIZE_KQ",
        "BLOCK_SIZE_KK",
        "QUANT_BLOCK_SIZE",
    ],
)


@triton.jit(repr=_fused_dual_rmsnorm_mxfp8_quant_repr)
def _fused_dual_rmsnorm_mxfp8_quant_kernel(
    q_ptr,
    k_ptr,
    gq_ptr,
    gk_ptr,
    yq_ptr,
    sq_ptr,
    yk_ptr,
    M,
    KQ,
    KK,
    stride_qm,
    stride_qn,
    stride_km,
    stride_kn,
    stride_yqm,
    stride_yqn,
    stride_sqm,
    stride_sqn,
    stride_ykm,
    stride_ykn,
    eps_q,
    eps_k,
    BLOCK_SIZE_KQ: tl.constexpr,  # power-of-2 covering full KQ
    BLOCK_SIZE_KK: tl.constexpr,  # power-of-2 covering full KK
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32 (MXFP8 group size)
    NUM_PRGMS,  # row-loop stride; runtime, see `_fused_rms_mxfp8_kernel`
    AMAX_FLOOR: tl.constexpr = 0.0,  # Q groups' amax floor (none by default)
):
    """One program per row: do Q-side RMSNorm+MXFP8 quant AND K-side RMSNorm
    (bf16 out) in one launch. Mirrors the CK `fused_qk_rmsnorm_group_quant`
    fusion topology but emits MXFP8 1x32 (e8m0) scales for Q directly."""
    row_start = tl.program_id(0)

    q_col_offsets = tl.arange(0, BLOCK_SIZE_KQ)
    q_mask = q_col_offsets < KQ
    k_col_offsets = tl.arange(0, BLOCK_SIZE_KK)
    k_mask = k_col_offsets < KK

    n_q_groups: tl.constexpr = BLOCK_SIZE_KQ // QUANT_BLOCK_SIZE

    for row_idx in tl.range(row_start, M, NUM_PRGMS, num_stages=2):
        # ===== Q side: RMSNorm + MXFP8 quant =====
        x_q = tl.load(
            q_ptr + row_idx * stride_qm + q_col_offsets * stride_qn,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        g_q = tl.load(gq_ptr + q_col_offsets, mask=q_mask, other=0.0).to(tl.float32)

        ss_q = tl.sum(x_q * x_q, axis=-1)
        norm_q = tl.math.rsqrt((ss_q / KQ) + eps_q)
        y_q_fp32 = x_q * norm_q * g_q

        y_q_2d = tl.reshape(y_q_fp32, (n_q_groups, QUANT_BLOCK_SIZE))
        scale_q_e8m0, quant_scale_q = _mxfp8_quant_op(
            y_q_2d, QUANT_AXIS=1, AMAX_FLOOR=AMAX_FLOOR
        )

        qx_q_2d = y_q_2d * quant_scale_q
        qx_q = tl.reshape(qx_q_2d, (BLOCK_SIZE_KQ,))
        y_q_fp8 = qx_q.to(yq_ptr.type.element_ty)

        tl.store(
            yq_ptr + row_idx * stride_yqm + q_col_offsets * stride_yqn,
            y_q_fp8,
            mask=q_mask,
        )

        q_group_offsets = tl.arange(0, n_q_groups)
        q_group_mask = q_group_offsets < (KQ // QUANT_BLOCK_SIZE)
        scale_q_flat = tl.reshape(scale_q_e8m0, (n_q_groups,))
        tl.store(
            sq_ptr + row_idx * stride_sqm + q_group_offsets * stride_sqn,
            scale_q_flat,
            mask=q_group_mask,
        )

        # ===== K side: RMSNorm only, bf16 out =====
        x_k = tl.load(
            k_ptr + row_idx * stride_km + k_col_offsets * stride_kn,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        g_k = tl.load(gk_ptr + k_col_offsets, mask=k_mask, other=0.0).to(tl.float32)

        ss_k = tl.sum(x_k * x_k, axis=-1)
        norm_k = tl.math.rsqrt((ss_k / KK) + eps_k)
        y_k_fp32 = x_k * norm_k * g_k
        y_k_out = y_k_fp32.to(yk_ptr.type.element_ty)

        tl.store(
            yk_ptr + row_idx * stride_ykm + k_col_offsets * stride_ykn,
            y_k_out,
            mask=k_mask,
        )


def fused_dual_rmsnorm_mxfp8_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps_q: float,
    eps_k: float | None = None,
    yq: torch.Tensor | None = None,
    sq: torch.Tensor | None = None,
    yk: torch.Tensor | None = None,
    amax_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused dual RMSNorm in a single Triton launch.

    - Q side: RMSNorm(q, q_weight, eps_q) -> MXFP8 (FP8 e4m3fn + uint8 e8m0 1x32).
    - K side: RMSNorm(k, k_weight, eps_k) -> bf16.

    Replaces the CK `fused_qk_rmsnorm_group_quant` kernel for the MXFP8 GEMM
    path on V4 (Task #77): one launch instead of two (fused_rms_mxfp8_quant +
    rmsnorm2d_fwd_), eliminating the ~6us/layer launch-overhead regression.

    Args:
        q: (M, KQ) bf16 or fp16 — Q-side input (e.g. q_lora).
        k: (M, KK) bf16 or fp16 — K-side input (e.g. kv_pre).
        q_weight: (KQ,) bf16 or fp16 — Q RMSNorm weight.
        k_weight: (KK,) bf16 or fp16 — K RMSNorm weight.
        eps_q: Q RMSNorm epsilon.
        eps_k: K RMSNorm epsilon; defaults to eps_q.
        yq, sq, yk: optional pre-allocated outputs.
        amax_floor: floor of each Q group's amax (0: none).

    Returns:
        yq (M, KQ) fp8 e4m3fn, sq (M, KQ // 32) uint8 e8m0, yk (M, KK) bf16.
    """
    assert q.dim() == 2, f"q must be 2D, got {q.dim()}"
    assert k.dim() == 2, f"k must be 2D, got {k.dim()}"
    M, KQ = q.shape
    Mk, KK = k.shape
    assert M == Mk, f"q rows {M} != k rows {Mk}"
    assert q_weight.shape == (KQ,), f"q_weight shape {q_weight.shape} != ({KQ},)"
    assert k_weight.shape == (KK,), f"k_weight shape {k_weight.shape} != ({KK},)"
    assert (
        KQ % _QUANT_BLOCK_SIZE == 0
    ), f"KQ={KQ} must be a multiple of {_QUANT_BLOCK_SIZE}"
    if eps_k is None:
        eps_k = eps_q

    Ns = KQ // _QUANT_BLOCK_SIZE
    BLOCK_SIZE_KQ = triton.next_power_of_2(KQ)
    BLOCK_SIZE_KK = triton.next_power_of_2(KK)

    if yq is None:
        yq = torch.empty((M, KQ), dtype=torch.float8_e4m3fn, device=q.device)
    if sq is None:
        sq = torch.empty((M, Ns), dtype=torch.uint8, device=q.device)
    if yk is None:
        yk = torch.empty((M, KK), dtype=k.dtype, device=k.device)

    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _fused_dual_rmsnorm_mxfp8_quant_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        yq,
        sq,
        yk,
        M,
        KQ,
        KK,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        yq.stride(0),
        yq.stride(1),
        sq.stride(0),
        sq.stride(1),
        yk.stride(0),
        yk.stride(1),
        eps_q,
        eps_k,
        BLOCK_SIZE_KQ=BLOCK_SIZE_KQ,
        BLOCK_SIZE_KK=BLOCK_SIZE_KK,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
        AMAX_FLOOR=amax_floor,
    )
    return yq, sq, yk


FP8_DTYPE = dtypes.fp8


FP8_MAX = tl.constexpr(torch.finfo(FP8_DTYPE).max)


FP8_TL_DTYPE = tl.constexpr(
    tl.float8e4b8 if FP8_DTYPE is torch.float8_e4m3fnuz else tl.float8e4nv
)


AMAX_FLOOR = tl.constexpr(1e-10)


@triton.jit
def ceil_pow2_code(value):
    bits = value.to(tl.uint32, bitcast=True)
    return (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)


@triton.jit
def quantize_fp8_kernel(X, Y, S, SIZE, DEQUANT: tl.constexpr):
    # SIZE is the token count. A `tl.constexpr` here builds one kernel per
    # prefill chunk length and per batch size, and it only bounds the tiles.
    groups = tl.program_id(0) * 32 + tl.arange(0, 32)
    offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + offsets, offsets < SIZE, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), AMAX_FLOOR)
    code = ceil_pow2_code(amax * (1.0 / FP8_MAX))
    scale = (code << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x / scale[:, None], -FP8_MAX), FP8_MAX).to(FP8_TL_DTYPE)
    if DEQUANT:
        tl.store(Y + offsets, q.to(tl.float32) * scale[:, None], offsets < SIZE)
    else:
        tl.store(Y + offsets, q, offsets < SIZE)
        tl.store(S + groups, code.to(tl.uint8), groups < SIZE // 32)


def _check_quant_input(x, group):
    if x.ndim < 1 or x.shape[-1] % group:
        raise ValueError(f"Last dimension must be divisible by {group}")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Quantization requires a floating-point CUDA/ROCm tensor")
    return x.contiguous()


def quantize_fp8(x: torch.Tensor, *, dequantize: bool = False):
    """E4M3, group32 E8M0 ceil scales; optionally return QAT values in x.dtype."""
    x = _check_quant_input(x, 32)
    output = torch.empty_like(x, dtype=x.dtype if dequantize else FP8_DTYPE)
    scales = (
        None
        if dequantize
        else torch.empty(
            (*x.shape[:-1], x.shape[-1] // 32),
            device=x.device,
            dtype=torch.float8_e8m0fnu,
        )
    )
    if x.numel():
        quantize_fp8_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            output,
            None if scales is None else scales.view(torch.uint8),
            x.numel(),
            dequantize,
        )
    return output if dequantize else (output, scales)

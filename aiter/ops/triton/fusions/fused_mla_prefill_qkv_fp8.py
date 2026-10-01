# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Builds the FP8 q/k/v an MLA prefill attention kernel takes in one pass,
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_mla_prefill_qkv_fp8 import (
    _fused_mla_prefill_qkv_fp8_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["fused_mla_prefill_qkv_fp8"]

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
_INT32_MAX = 2**31 - 1


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _check_out(name, t, shape, dtype, device):
    assert t.shape == shape, f"{name} shape mismatch: {tuple(t.shape)} vs {shape}"
    assert t.dtype == dtype, f"{name} dtype mismatch: {t.dtype} vs {dtype}"
    assert t.device == device, f"{name} must be on {device}"
    assert t.is_contiguous(), f"{name} must be contiguous"


def fused_mla_prefill_qkv_fp8(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    v: torch.Tensor,
    k_pe: torch.Tensor,
    num_heads_out: int,
    out_dtype: torch.dtype,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MLA prefill attends per head over the decompressed latent: q holds
    [nope | rope] per head, K is [k_nope | k_pe] with k_pe shared by every
    head, and V is per head. Kernels such as ``mla_prefill_ps_asm_fwd`` take
    these packed, in FP8, at a head count that is a multiple of 16, so models
    with fewer heads per rank (Kimi-K3 at TP8: 12) replicate-pad. In one launch:

        q_out[t, h] = q[t, h % H]
        k_out[t, h] = cat(k_nope[t, h % H], k_pe[t])
        v_out[t, h] = v[t, h % H]

    for h in [0, num_heads_out), cast to ``out_dtype`` without scaling (the
    same as ``tensor.to(out_dtype)``).

    Args:
        q: [T, H, nope + rope], fp16/bf16. Any strides; last dim dense.
        k_nope: [T, H, nope], same dtype. Any strides (typically a view of the
            kv_b_proj output); last dim dense.
        v: [T, H, v_dim], same dtype and layout rules as k_nope.
        k_pe: [T, 1, rope] or [T, rope], same dtype; last dim dense. rope may be 0.
        num_heads_out: output head count, >= H. Padded head h duplicates head h % H,
            so the real heads come first.
        out_dtype: torch.float8_e4m3fn or torch.float8_e4m3fnuz.
        q_out, k_out, v_out: optional contiguous destinations.

    Returns:
        (q_out, k_out, v_out), contiguous [T, num_heads_out, D].

    Constraints:
        nope, v_dim and (when non-zero) rope are powers of two; every offset
        fits in int32.
    """
    _LOGGER.info(
        "FUSED_MLA_PREFILL_QKV_FP8: q=%s k_nope=%s v=%s k_pe=%s heads_out=%d",
        tuple(q.shape),
        tuple(k_nope.shape),
        tuple(v.shape),
        tuple(k_pe.shape),
        num_heads_out,
    )
    if k_pe.dim() == 2:
        k_pe = k_pe.unsqueeze(1)
    assert q.dim() == k_nope.dim() == v.dim() == k_pe.dim() == 3, "expected 3-D inputs"
    T, H, qk_dim = q.shape
    nope, v_dim, rope = k_nope.shape[-1], v.shape[-1], k_pe.shape[-1]
    assert k_nope.shape[:2] == (T, H), f"k_nope shape mismatch: {tuple(k_nope.shape)}"
    assert v.shape[:2] == (T, H), f"v shape mismatch: {tuple(v.shape)}"
    assert k_pe.shape[:2] == (T, 1), f"k_pe shape mismatch: {tuple(k_pe.shape)}"
    assert qk_dim == nope + rope, f"q head dim {qk_dim} != nope {nope} + rope {rope}"
    assert (
        _is_pow2(nope) and _is_pow2(v_dim) and (rope == 0 or _is_pow2(rope))
    ), f"head dims must be powers of two: nope={nope} rope={rope} v={v_dim}"
    assert num_heads_out >= H, f"num_heads_out {num_heads_out} < heads {H}"
    assert q.dtype in (torch.float16, torch.bfloat16), f"unsupported dtype {q.dtype}"
    assert k_nope.dtype == v.dtype == k_pe.dtype == q.dtype, "inputs must share a dtype"
    assert out_dtype in _FP8_DTYPES, f"unsupported out_dtype {out_dtype}"
    assert q.is_cuda and all(t.device == q.device for t in (k_nope, v, k_pe))
    assert all(
        t.stride(-1) == 1 for t in (q, k_nope, v, k_pe)
    ), "last dim must be dense"

    qk_shape, v_shape = (T, num_heads_out, qk_dim), (T, num_heads_out, v_dim)
    if q_out is None:
        q_out = torch.empty(qk_shape, dtype=out_dtype, device=q.device)
    if k_out is None:
        k_out = torch.empty(qk_shape, dtype=out_dtype, device=q.device)
    if v_out is None:
        v_out = torch.empty(v_shape, dtype=out_dtype, device=q.device)
    _check_out("q_out", q_out, qk_shape, out_dtype, q.device)
    _check_out("k_out", k_out, qk_shape, out_dtype, q.device)
    _check_out("v_out", v_out, v_shape, out_dtype, q.device)
    if T == 0:
        return q_out, k_out, v_out

    def max_offset(t):
        return sum((s - 1) * st for s, st in zip(t.shape, t.stride()))

    assert (
        max(map(max_offset, (q, k_nope, v, k_pe, q_out, k_out, v_out))) <= _INT32_MAX
    ), "offsets exceed int32"

    config = _get_config()
    BLOCK_T = config.pop("BLOCK_T")
    heads_out_pow2 = triton.next_power_of_2(num_heads_out)
    grid = (triton.cdiv(T, BLOCK_T),)
    _fused_mla_prefill_qkv_fp8_kernel[grid](
        q,
        k_nope,
        v,
        k_pe,
        q_out,
        k_out,
        v_out,
        T,
        q.stride(0),
        q.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        v.stride(0),
        v.stride(1),
        k_pe.stride(0),
        NUM_HEADS=H,
        NUM_HEADS_OUT=num_heads_out,
        HEADS_OUT_POW2=heads_out_pow2,
        NOPE_DIM=nope,
        ROPE_DIM=rope,
        V_DIM=v_dim,
        BLOCK_T=BLOCK_T,
        NEED_MASK=(T % BLOCK_T != 0) or (heads_out_pow2 != num_heads_out),
        **config,
    )
    return q_out, k_out, v_out

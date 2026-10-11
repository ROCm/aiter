# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host wrapper for the standalone FlyDSL GDN K6 output kernel.

``chunk_fwd_o_flydsl`` is signature-compatible with the Triton
``chunk_fwd_o_opt_vk`` so the separate K5+K6 pipeline can swap one for the other
at the call site without reshaping anything.
"""

from __future__ import annotations

import torch

from ..triton._triton_kernels.gated_delta_rule.utils import (
    GatedDeltaRulePrefillMetadata,
    prepare_chunk_indices,
    prepare_rebased_cu_seqlens,
)
from . import linear_attention_prefill_kernels as _host
from .kernels.chunk_gated_delta_o import compile_chunk_gated_delta_o
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "chunk_fwd_o_flydsl",
    "flydsl_k6_supported",
    "is_flydsl_k6_unsupported",
]

_compiled_o_kernels: dict = {}

# CDNA3 / CDNA4 both provide the 16x16x16 bf16 MFMA the kernel is built on.
_SUPPORTED_ARCHS = ("gfx942", "gfx950")


def _default_bv(V: int) -> int:
    """One V-tile per CTA.

    A wider tile cuts the q/k/A re-read across V-tiles -- BV=128 moves 28% less
    traffic than BV=64 -- but costs LDS, and below gfx950's 160 KiB the CTAs per
    CU drop faster than the traffic saved. gfx942's 64 KiB only fits 64 anyway.
    """
    return min(128 if _host._GFX_ARCH == "gfx950" else 64, V)


def _default_nr_split(BV: int, BT: int) -> int:
    """Waves along N per 16 query rows.

    The kernel's LDS footprint is fixed by the tile, so once it is large enough
    to cap the CTAs per CU, the only way left to add resident waves is to widen
    the CTA. That is worth doing exactly when the tile is wide enough to keep
    the extra waves fed: BV=128 splits, narrower tiles already fit more CTAs.
    """
    if BV >= 128 and BV // 16 % 2 == 0 and BT // 16 % 2 == 0:
        return 2
    return 1


def is_flydsl_k6_unsupported(
    *,
    q: torch.Tensor,
    h: torch.Tensor,
    K: int,
    V: int,
    BV: int,
    chunk_size: int,
) -> str | None:
    """Why the standalone K6 kernel cannot serve this call, or None if it can."""
    if _host._GFX_ARCH not in _SUPPORTED_ARCHS:
        return f"the FlyDSL K6 kernel needs {' or '.join(_SUPPORTED_ARCHS)}; this device is {_host._GFX_ARCH}"
    if q.dtype is not torch.bfloat16:
        return f"the FlyDSL K6 kernel is bf16-only; q is {q.dtype}"
    if h.dtype is not torch.bfloat16:
        return (
            f"the FlyDSL K6 kernel reads a bf16 h snapshot; got {h.dtype}. "
            "Pass snapshot_dtype=torch.bfloat16 to K5."
        )
    if chunk_size != 64:
        return f"the FlyDSL K6 kernel fixes BT=64; got chunk_size={chunk_size}"
    if K % 64 or K > 256 or (K & (K - 1)):
        return f"the FlyDSL K6 kernel needs a power-of-two K in 64..256; got K={K}"
    if V % BV:
        return f"BV={BV} must divide V={V}"
    # k + h + v + A; q is register-resident (see the kernel's LDS budget note).
    lds_kib = (64 * K + BV * K + 64 * BV + 64 * 64) * 2 / 1024
    budget = 160.0 if _host._GFX_ARCH == "gfx950" else 64.0
    if lds_kib > budget:
        return (
            f"the FlyDSL K6 kernel needs {lds_kib:.0f} KiB LDS at BV={BV}, K={K}, "
            f"over the {budget:.0f} KiB {_host._GFX_ARCH} budget"
        )
    return None


def flydsl_k6_supported(
    *,
    q: torch.Tensor,
    h: torch.Tensor,
    K: int,
    V: int,
    BV: int | None = None,
    chunk_size: int = 64,
) -> bool:
    """Whether the standalone K6 kernel can serve this call."""
    return (
        is_flydsl_k6_unsupported(
            q=q, h=h, K=K, V=V, BV=BV or _default_bv(V), chunk_size=chunk_size
        )
        is None
    )


def chunk_fwd_o_flydsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
    BV: int | None = None,
    NR_SPLIT: int | None = None,
) -> torch.Tensor:
    """FlyDSL K6: o = scale * (exp(g) * q @ h^T + tril(gated q @ k^T) @ v_new).

    Args:
        q: [B, T, Hg, K] bf16 token-major
        k: [B, T, Hg, K] bf16 token-major
        v: [B, H, T_flat, V] bf16 head-major — K5's UNGATED v_new
        o: [B, T, H, V] pre-allocated output, written in place
        h: [NT_flat, H, V, K] bf16 — K5's per-chunk snapshot, K contiguous
        g: [B, H, T_flat] fp32 head-major cumulative gate, or None
        scale: query scale; defaults to K**-0.5
        use_exp2: when True, g is interpreted in log2 space

    Returns:
        o
    """
    B, T, Hg, K = q.shape
    H = v.shape[1]
    T_flat = v.shape[2]
    V = v.shape[-1]
    BT = chunk_size
    if BV is None:
        BV = _default_bv(V)
    if NR_SPLIT is None:
        NR_SPLIT = _default_nr_split(BV, BT)
    if scale is None:
        scale = K**-0.5

    reason = is_flydsl_k6_unsupported(q=q, h=h, K=K, V=V, BV=BV, chunk_size=BT)
    if reason is not None:
        raise NotImplementedError(f"chunk_fwd_o_flydsl: {reason}")

    assert q.is_contiguous() and k.is_contiguous(), "K6: q and k must be contiguous."
    assert v.is_contiguous(), "K6: v_new must be contiguous."
    assert h.is_contiguous(), "K6: the h snapshot must be contiguous."
    if g is not None:
        assert g.is_contiguous(), "K6: g must be contiguous head-major."
        assert (
            g.shape[-1] == T_flat and g.shape[-2] == H
        ), f"K6: g must be [.., H={H}, T_flat={T_flat}]; got {tuple(g.shape)}."

    # Chunk schedule, mirroring the Triton K6 so both kernels walk identical
    # (sequence, chunk) pairs.
    if cu_seqlens is not None:
        if prefill_metadata is not None:
            prefill_metadata.validate(
                cu_seqlens=cu_seqlens,
                chunk_size=BT,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                total_prefill_tokens=T,
                num_sequences=len(cu_seqlens) - 1,
            )
            schedule = prefill_metadata.get_chunk_schedule(
                BT,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
            )
            sequence_ids = schedule.sequence_ids
            chunk_ids = schedule.chunk_ids
            kernel_cu_seqlens = schedule.kernel_cu_seqlens
            index_stride = 1
        else:
            flat_chunk_indices = prepare_chunk_indices(
                cu_seqlens, BT, num_decodes, num_decode_tokens
            ).reshape(-1)
            sequence_ids = flat_chunk_indices
            chunk_ids = flat_chunk_indices[1:]
            kernel_cu_seqlens = prepare_rebased_cu_seqlens(
                cu_seqlens, num_decodes, num_decode_tokens
            )
            index_stride = 2
        grid_nt = len(sequence_ids) // index_stride
    else:
        sequence_ids = None
        chunk_ids = None
        kernel_cu_seqlens = None
        index_stride = 1
        grid_nt = (T + BT - 1) // BT

    is_varlen = cu_seqlens is not None
    use_g = g is not None

    dummy = torch.empty(1, device=q.device, dtype=torch.int32)

    def _i32(t):
        return t.to(torch.int32) if t is not None else dummy

    cache_key = (
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        float(scale),
        use_g,
        is_varlen,
        bool(use_exp2),
        index_stride,
        NR_SPLIT,
    )
    if cache_key not in _compiled_o_kernels:
        _compiled_o_kernels[cache_key] = compile_chunk_gated_delta_o(
            K=K,
            V=V,
            H=H,
            Hg=Hg,
            SCALE=float(scale),
            BT=BT,
            BV=BV,
            USE_G=use_g,
            IS_VARLEN=is_varlen,
            G_IS_LOG2_SCALED=bool(use_exp2),
            INDEX_STRIDE=index_stride,
            NR_SPLIT=NR_SPLIT,
        )
    launch_fn = _compiled_o_kernels[cache_key]

    _run_compiled(
        launch_fn,
        q,
        k,
        v,
        h,
        g if g is not None else dummy.to(torch.float32),
        o,
        _i32(kernel_cu_seqlens),
        _i32(sequence_ids),
        _i32(chunk_ids),
        T,
        T_flat,
        grid_nt,
        B * H,
        torch.cuda.current_stream(),
    )
    return o

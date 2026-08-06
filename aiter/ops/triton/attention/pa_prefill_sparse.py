# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse paged-prefill attention over two KV sources (prefix + extend) with
per-head sink bias -- gfx1250 (gluon) only.

Exposes ``pa_prefill_sparse`` -- grid ``(T, cdiv(H, BLOCK_H))``, one token
and BLOCK_H heads per CTA. Same grid as the decode kernel. No split-K:
prefill fills the GPU via the token dimension.

Gfx950(`mla_gluon`) and others(Triton `_sparse_attn_prefill_kernel`, they
take a 1-D ``(kv_indices, kv_indptr)`` pair over one pool.

``pa_prefill_sparse`` -- a single entry that dispatches on arch:

    gfx1250 -> gluon ``_pa_prefill_sparse``           (two sources, native)
    gfx950  -> gluon wrapper ``mla_gluon``            (single source)
    else    -> triton ``_sparse_attn_prefill_kernel`` (single source)
"""

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_prefill_sparse import (
    _pa_prefill_sparse as gluon_pa_prefill_sparse,
)
from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4 import (
    _sparse_attn_prefill_kernel,
)
from aiter.ops.triton.gluon.mla_gluon import (
    mla_gluon as gluon_mla_sparse_prefill,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

DEVICE_ARCH = arch_info.get_arch()

_LOGGER = AiterTritonLogger()


def pa_prefill_sparse(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor | None,
    kv_indices_extend: torch.Tensor | None,
    kv_indptr_extend: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    has_invalid: bool | None = None,
) -> torch.Tensor:
    """Sparse prefill attention over two KV sources with sink.

    Dispatches on arch: gfx1250 -> gluon (two sources), gfx950 -> `mla_gluon`,
    otherwise the Triton kernel. Only gfx1250 reads a second KV source; the
    other two serve the prefix source alone and reject a non-empty extend.

    Args:
        q:                 [T, H, D] BF16/FP16 -- queries.
        unified_kv:        [total_pages, D] -- prefix KV source (paged).
        kv_indices_prefix: [total_prefix] int32 -- flat per-token slot lists
            into unified_kv. ``-1`` sentinels skipped.
        kv_indptr_prefix:  [T+1] int32 -- true prefix sum.
        kv:                [total_tokens, D] -- extend KV source (this fwd's
            input K, not yet in paged buffer). ``None`` for no extend source.
        kv_indices_extend: [total_extend] int32 -- flat per-token row idx lists
            into kv. ``-1`` sentinels skipped. ``None`` for no extend source.
        kv_indptr_extend:  [T+1] int32 -- true prefix sum. ``None`` for none.
        attn_sink:         [H] fp32 -- per-head softmax-denom bias.
        softmax_scale:     float.

    Returns:
        [T, H, D] attention output, same dtype as q.
    """
    if DEVICE_ARCH == "gfx1250":
        if not q.is_cuda:
            raise RuntimeError("pa_prefill_sparse requires CUDA/HIP tensors")
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(f"pa_prefill_sparse expects fp16/bf16 q, got {q.dtype}")
        if unified_kv.dtype != q.dtype:
            raise RuntimeError(
                f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
            )
        if kv.dtype != q.dtype:
            raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")

        T, H, D = q.shape
        if has_invalid is None:
            avg_prefix_len = kv_indices_prefix.numel() / max(T, 1)
            has_invalid = not (0 < avg_prefix_len <= 16)
        _LOGGER.info(
            f"PA_PREFILL_SPARSE T={T} H={H} D={D} "
            f"prefix_indices={kv_indices_prefix.shape[0]} "
            f"extend_indices={kv_indices_extend.shape[0]}"
        )

        out = torch.empty_like(q)
        assert (
            kv_indices_prefix.dtype == torch.int32 and kv_indices_prefix.is_contiguous()
        )
        assert (
            kv_indptr_prefix.dtype == torch.int32 and kv_indptr_prefix.is_contiguous()
        )
        assert (
            kv_indices_extend.dtype == torch.int32 and kv_indices_extend.is_contiguous()
        )
        assert (
            kv_indptr_extend.dtype == torch.int32 and kv_indptr_extend.is_contiguous()
        )

        total_prefix_pages = unified_kv.shape[0]
        total_extend_tokens = kv.shape[0]
        USE_EXP2 = True
        block_d = triton.next_power_of_2(D)
        assert block_d == D

        if H >= 64:
            block_h = 64
            block_k = 32
            num_warps = 4
            waves_per_eu = 1
        elif H >= 32:
            block_h = 32
            block_k = 32
            num_warps = 2
            waves_per_eu = 1
        else:
            block_h = max(triton.next_power_of_2(min(H, 16)), 16)
            block_k = 16
            num_warps = 1
            waves_per_eu = 1
        grid = (T, triton.cdiv(H, block_h))

        gluon_pa_prefill_sparse[grid](
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            out,
            total_prefix_pages,
            total_extend_tokens,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            unified_kv.stride(0),
            unified_kv.stride(1),
            kv.stride(0),
            kv.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            H,
            D,
            float(softmax_scale),
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            BLOCK_K=block_k,
            HAS_INVALID=has_invalid,
            USE_EXP2=USE_EXP2,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )
        return out

    elif DEVICE_ARCH == "gfx950":
        kv_indices_prefix, kv_indptr_prefix = _prep_single_source(
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
        )
        out = torch.empty_like(q)
        gluon_mla_sparse_prefill(
            q,  # q_nope = combined-D query (RoPE folded in)
            None,  # q_pe unused in prefill mode
            unified_kv,  # kv_c
            out,  # o (written in place)
            kv_indices_prefix,  # page_table = ragged kv_indices
            kv_indptr_prefix,  # seq_info = ragged kv_indptr
            float(softmax_scale),
            min_kv_seq_len=float("inf"),  # skip min_kv_seq_len check
            has_pe=False,
            attn_sink=attn_sink.contiguous() if attn_sink is not None else None,
        )
        return out

    else:
        # Portable Triton fallback.
        kv_indices_prefix, kv_indptr_prefix = _prep_single_source(
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
        )
        attn_sink = attn_sink or torch.empty(1, device="cuda", dtype=torch.float32)
        has_attn_sink = attn_sink is not None
        num_queries, num_heads, head_dim = q.shape
        block_d = triton.next_power_of_2(head_dim)
        out = torch.empty_like(q)

        grid = lambda META: (
            num_queries,
            triton.cdiv(num_heads, META["BLOCK_H"]),
        )
        _sparse_attn_prefill_kernel[grid](
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            attn_sink,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            unified_kv.stride(0),
            unified_kv.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            num_heads,
            head_dim,
            unified_kv.shape[0],
            float(softmax_scale),
            HAS_ATTN_SINK=has_attn_sink,
            BLOCK_D=block_d,
        )
        return out


# ---------------------------------------------------------------------------
# Inputs preparation
# ---------------------------------------------------------------------------


def _prep_single_source(
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor | None,
    kv_indices_extend: torch.Tensor | None,
    kv_indptr_extend: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the KV pool and indices for the gfx950 / Triton kernels.

    Rejects an extend KV source outright: only the gfx1250 gluon kernel reads a
    second pool.
    """
    extend = (
        ("kv", kv),
        ("kv_indices_extend", kv_indices_extend),
        ("kv_indptr_extend", kv_indptr_extend),
    )
    provided = [name for name, tensor in extend if tensor is not None]
    if provided:
        raise NotImplementedError(
            f"pa_prefill_sparse got an extend KV source ({', '.join(provided)}), "
            f"but {DEVICE_ARCH} is served by a single-source kernel that reads "
            f"only unified_kv. Two KV sources are supported on gfx1250 only; "
            f"merge them into unified_kv, or pass None for the extend trio."
        )

    return (
        _as_int32_contiguous_1d(kv_indices_prefix),
        _as_int32_contiguous_1d(kv_indptr_prefix),
    )


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()

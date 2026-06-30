# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Some of the kernels are adapted from vLLM deepseek v4 sparse attention:
# https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py


import torch
import triton
from packaging.version import Version

from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4 import (
    _combine_topk_swa_indices_ragged_kernel,
    _compute_combined_lens_kernel,
    _compute_topk_lens_kernel,
    _pack_dense_prefix_to_ragged_kernel,
    _pack_global_topk_ragged_kernel,
    _sparse_attn_prefill_kernel,
)
from aiter.ops.triton.utils._triton import arch_info

# Gluon (CDNA4) variant — opt-in, gated on Triton ≥ 3.6 + arch=gfx950.
# Prefill is served by the unified `mla_gluon(..., has_pe=False)` entry (HAS_PE=False
# + 1-D VarLen + single-split fast path over the shared `_mla_gluon` kernel).
_TRITON_VERSION = Version(triton.__version__)
_TRITON_GE_36 = _TRITON_VERSION >= Version("3.6.0")
_arch = arch_info.get_arch()
_gluon_sparse_attn_prefill = None
if _TRITON_GE_36 and _arch == "gfx950":
    try:
        from aiter.ops.triton.gluon.mla_gluon import (
            mla_gluon as _gluon_sparse_attn_prefill,
        )
    except ImportError:
        pass  # symbol stays None (set above)


def _use_gluon(gluon_fn) -> bool:
    """Prefer the Gluon kernel when it is available"""
    return gluon_fn is not None


def _bh_grid(num_queries: int, num_heads: int):
    """Triton-fallback launch grid: one program per (query, BLOCK_H head-block)."""
    return lambda META: (
        num_queries,
        triton.cdiv(num_heads, META["BLOCK_H"]),
    )  # noqa: E731


# ---------------------------------------------------------------------------
# DSV4 sparse-MLA layout constants
# ---------------------------------------------------------------------------

_DSV4_SPARSE_NOPE_DIM = 448
_DSV4_SPARSE_ROPE_DIM = 64


def _validate_dsv4_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    assert (
        head_dim == nope_head_dim + rope_head_dim
    ), f"{op_name} expected head_dim={nope_head_dim + rope_head_dim}, got {head_dim}"
    assert (
        nope_head_dim == _DSV4_SPARSE_NOPE_DIM
        and rope_head_dim == _DSV4_SPARSE_ROPE_DIM
    ), (
        f"{op_name} expects {_DSV4_SPARSE_NOPE_DIM} NoPE dims and "
        f"{_DSV4_SPARSE_ROPE_DIM} RoPE dims"
    )


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = _as_int32_contiguous_1d(lengths)
    indptr = torch.empty(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    indptr[0] = 0
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


def _valid_lengths(indices_2d: torch.Tensor) -> torch.Tensor:
    """Per-row count of non-sentinel (>= 0) entries."""
    return (indices_2d >= 0).sum(dim=-1, dtype=torch.int32)


def _prep_attn_sink(
    attn_sink: torch.Tensor | None, device: torch.device
) -> tuple[torch.Tensor, bool]:
    """Normalize the optional per-head sink into ``(tensor, has_sink)``. When
    absent, a 1-element placeholder keeps the kernel arg a valid pointer (its use
    is gated by the HAS_ATTN_SINK constexpr)."""
    if attn_sink is None:
        return torch.empty(1, device=device, dtype=torch.float32), False
    return attn_sink.contiguous(), True


# ---------------------------------------------------------------------------
# Ragged-index builders
# ---------------------------------------------------------------------------


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a dense `[N, max_topk]` index matrix into `(flat, indptr)`.

    `lengths[i]` controls how many of the leading entries of row `i` are
    copied. Entries that fall outside `[0, num_rows)` (when
    `num_rows >= 0`) are mapped to `-1` so the consumer kernel can mask
    them safely.
    """
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert (
        lengths.numel() == indices.shape[0]
    ), f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"

    max_width = indices.shape[1]
    lengths = lengths.clamp(min=0, max=max_width).contiguous()

    indptr = _build_indptr_from_lengths(lengths)

    if indices.numel() == 0:
        flat = torch.empty(0, dtype=torch.int32, device=indices.device)
    else:
        flat = torch.empty(
            int(indptr[-1].item()), dtype=torch.int32, device=indices.device
        )
        if flat.numel() > 0:
            block_size = 128
            _pack_dense_prefix_to_ragged_kernel[
                (indices.shape[0], triton.cdiv(max_width, block_size))
            ](
                indices,
                lengths,
                indptr,
                flat,
                indices.stride(0),
                int(num_rows),
                max_width,
                BLOCK_SIZE=block_size,
            )

    return flat, indptr


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve per-token local top-k positions to global slot ids (CSR).

    `topk_indices[t, k]` is a position inside the request `token_to_req_indices[t]`;
    looking it up in that request's `block_table` produces a physical slot id
    in the unified KV pool. Returns `(global_topk_ragged, topk_indptr, topk_lens)`.
    """
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


def combine_topk_swa_indices_ragged(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interleave per-token top-k and sliding-window indices into one CSR.

    For each query position, emits `min(topk, (pos+1)//compress_ratio)`
    top-k slots followed by `min(window_size, pos+1)` SWA slots. The two
    halves live in disjoint logical ranges (`[batch*M, batch*M+N)` for
    top-k vs `[batch*M+N, batch*M+N+window)` for SWA) so a single bf16
    KV buffer indexed by these ids can hold both pools without aliasing.
    """
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    num_workers = 128
    _compute_combined_lens_kernel[(num_reqs, num_workers)](
        combined_lens,
        query_start_loc,
        seq_lens,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
    )

    combined_indptr = _build_indptr_from_lengths(combined_lens)
    combined_ragged = torch.empty(
        num_tokens * (topk + window_size),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if combined_ragged.numel() > 0:
        block = 128
        _combine_topk_swa_indices_ragged_kernel[
            (num_reqs, num_workers, triton.cdiv(topk + window_size, block))
        ](
            combined_ragged,
            combined_indptr,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            M,
            N,
            topk_indices.shape[-1],
            TOP_K=topk,
            COMPRESS_RATIO=compress_ratio,
            WINDOW_SIZE=window_size,
            BLOCK_SIZE=block,
        )
    return combined_ragged, combined_indptr, combined_lens


# ---------------------------------------------------------------------------
# Sparse attention — internal launchers
# ---------------------------------------------------------------------------


def _sparse_attn_prefill_ragged(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert q.is_cuda and kv.is_cuda and indices.is_cuda and indptr.is_cuda

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    attn_sink, has_attn_sink = _prep_attn_sink(attn_sink, q.device)

    num_queries, num_heads, head_dim = q.shape
    assert (
        indptr.numel() == num_queries + 1
    ), f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    _validate_dsv4_sparse_dims(
        head_dim, nope_head_dim, rope_head_dim, "sparse_attn_prefill"
    )

    block_d = triton.next_power_of_2(head_dim)
    out = torch.empty_like(q, dtype=torch.bfloat16)

    if _use_gluon(_gluon_sparse_attn_prefill):
        _gluon_sparse_attn_prefill(
            q,  # q_nope = combined-D query (RoPE folded in)
            None,  # q_pe unused in prefill mode
            kv,  # kv_c
            out,  # o (written in place)
            indices,  # page_table = ragged kv_indices
            indptr,  # seq_info = ragged kv_indptr
            float(scale),
            has_pe=False,
            attn_sink=attn_sink if has_attn_sink else None,
        )
        return out
    else:
        grid = _bh_grid(num_queries, num_heads)
        _sparse_attn_prefill_kernel[grid](
            q,
            kv,
            indices,
            indptr,
            attn_sink,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            num_heads,
            head_dim,
            kv.shape[0],
            float(scale),
            HAS_ATTN_SINK=has_attn_sink,
            BLOCK_D=block_d,
        )
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    ragged_indices: torch.Tensor | None = None,
    ragged_indptr: torch.Tensor | None = None,
) -> None:
    """DSV4 sparse-MLA prefill — bf16 KV with ragged per-query indices.

    Args:
        q: `[N, H, D]` queries (bf16). `D = nope_head_dim + rope_head_dim`.
        kv: `[N_kv, 1, D]` KV pool (bf16). The `1` lane dim is squeezed
            inside this wrapper.
        indices: `[N, max_topk]` dense indices into `kv`'s flat `[N_kv, D]`
            view, with `-1` sentinels. Ignored when `ragged_indices` and
            `ragged_indptr` are both supplied.
        topk_length: `[N]` per-row valid count for `indices` (used to
            shrink the dense → ragged conversion). May be `None` to recompute
            from `(indices >= 0).sum(-1)`.
        scale: softmax scale.
        head_dim, nope_head_dim, rope_head_dim: must be `(512, 448, 64)`.
        attn_sink: optional `[H]` per-head softmax-denom bias (fp32).
        output: `[N, H, D]` destination (any dtype). Filled in place.
        ragged_indices, ragged_indptr: optional CSR alternative to
            `(indices, topk_length)`. Preferred when callers already have the
            ragged form materialized (e.g. across CUDA-graph captures).
    """
    assert (
        kv.ndim == 3 and kv.shape[1] == 1
    ), f"sparse_attn_prefill expects kv=[skv,1,d], got {kv.shape}"
    kv = kv.squeeze(1)  # [N_kv, 1, D] -> [N_kv, D] for the ragged launcher
    _validate_dsv4_sparse_dims(
        head_dim, nope_head_dim, rope_head_dim, "sparse_attn_prefill"
    )

    if ragged_indices is None or ragged_indptr is None:

        indices_2d = indices.reshape(indices.shape[0], -1)

        ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
            indices_2d,
            topk_length if topk_length is not None else _valid_lengths(indices_2d),
            num_rows=kv.shape[0],
        )

    output_chunk = _sparse_attn_prefill_ragged(
        q=q,
        kv=kv,
        indices=ragged_indices,
        indptr=ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
    )
    output.copy_(output_chunk.to(output.dtype))

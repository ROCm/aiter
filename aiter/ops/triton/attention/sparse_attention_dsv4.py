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
    _sparse_attn_decode_kernel,
    _sparse_attn_prefill_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

# Gluon (CDNA4) variants — opt-in, gated on Triton ≥ 3.6 + arch=gfx950.
_TRITON_VERSION = Version(triton.__version__)
_TRITON_GE_36 = _TRITON_VERSION >= Version("3.6.0")
_arch = arch_info.get_arch()
_gluon_sparse_attn_prefill_kernel = None
_gluon_sparse_attn_decode_kernel = None
if _TRITON_GE_36 and _arch == "gfx950":
    try:
        from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_attention_dsv4 import (
            _sparse_attn_prefill_kernel as _gluon_sparse_attn_prefill_kernel,
            _sparse_attn_decode_kernel as _gluon_sparse_attn_decode_kernel,
        )
    except ImportError:
        _gluon_sparse_attn_prefill_kernel = None
        _gluon_sparse_attn_decode_kernel = None

# Buffer-load resource descriptors are 32-bit byte-addressed; the Gluon kernels
# keep all per-row offsets in int32, so they can only be used when the addressed
# pool (bf16 KV buffer / fp8 paged cache) fits within 2 GiB.
_BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


def _fits_buffer_descriptor(tensor: torch.Tensor) -> bool:
    return tensor.numel() * tensor.element_size() < _BUFFER_LIMIT_BYTES


# ---------------------------------------------------------------------------
# DSV4 sparse-MLA layout constants
# ---------------------------------------------------------------------------

_DSV4_SPARSE_NOPE_DIM = 448
_DSV4_SPARSE_ROPE_DIM = 64


def _is_fp8_fnuz() -> bool:
    """Return True on architectures that use ``float8_e4m3fnuz`` (e.g. gfx942).

    The decode kernel bitcasts a packed uint8 lane to fp8 to dequantize the
    NoPE half of the cache. The bitcast target dtype depends on the
    architecture's FP8 encoding (FNUZ on MI300X / gfx942 vs FN on gfx950+).
    """
    return get_fp8_e4m3_dtype() == torch.float8_e4m3fnuz


def _validate_dsv4_sparse_dims(
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    op_name: str,
) -> None:
    assert head_dim == nope_head_dim + rope_head_dim, (
        f"{op_name} expected head_dim={nope_head_dim + rope_head_dim}, got {head_dim}"
    )
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
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


# ---------------------------------------------------------------------------
# Ragged-index builders
# ---------------------------------------------------------------------------


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a dense ``[N, max_topk]`` index matrix into ``(flat, indptr)``.

    ``lengths[i]`` controls how many of the leading entries of row ``i`` are
    copied. Entries that fall outside ``[0, num_rows)`` (when
    ``num_rows >= 0``) are mapped to ``-1`` so the consumer kernel can mask
    them safely.
    """
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert lengths.numel() == indices.shape[0], (
        f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"
    )

    max_width = indices.shape[1] if indices.ndim == 2 else 0
    lengths = lengths.clamp(min=0, max=max_width).contiguous()

    indptr = torch.empty(indices.shape[0] + 1, dtype=torch.int32, device=indices.device)
    indptr[0] = 0
    torch.cumsum(lengths, dim=0, out=indptr[1:])

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

    ``topk_indices[t, k]`` is a position inside the request ``token_to_req_indices[t]``;
    looking it up in that request's ``block_table`` produces a physical slot id
    in the unified KV pool. Returns ``(global_topk_ragged, topk_indptr, topk_lens)``.
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

    For each query position, emits ``min(topk, (pos+1)//compress_ratio)``
    top-k slots followed by ``min(window_size, pos+1)`` SWA slots. The two
    halves live in disjoint logical ranges (``[batch*M, batch*M+N)`` for
    top-k vs ``[batch*M+N, batch*M+N+window)`` for SWA) so a single bf16
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
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim, nope_head_dim, rope_head_dim, "sparse_attn_prefill"
    )

    block_d = triton.next_power_of_2(head_dim)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    # Prefer the Gluon kernel, but it gathers KV via 32-bit buffer_load offsets,
    # so fall back to Triton when the KV pool exceeds the 2 GiB descriptor cap.
    if _gluon_sparse_attn_prefill_kernel is not None and _fits_buffer_descriptor(kv):
        kernel = _gluon_sparse_attn_prefill_kernel
    else:
        kernel = _sparse_attn_prefill_kernel
    grid = lambda META: (  # noqa: E731
        num_queries,
        triton.cdiv(num_heads, META["BLOCK_H"]),
    )
    kernel[grid](
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


def _sparse_attn_decode_ragged(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {main_cache.shape}"
    )
    assert main_indices.ndim == 1, (
        f"expected main_indices=[nnz], got {main_indices.shape}"
    )
    assert main_indptr.ndim == 1, f"expected main_indptr=[b+1], got {main_indptr.shape}"
    assert (
        q.is_cuda
        and main_cache.is_cuda
        and main_indices.is_cuda
        and main_indptr.is_cuda
    )

    main_indices = _as_int32_contiguous_1d(main_indices)
    main_indptr = _as_int32_contiguous_1d(main_indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert main_indptr.numel() == num_queries + 1, (
        f"expected main_indptr shape [{num_queries + 1}], got {main_indptr.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim, nope_head_dim, rope_head_dim, "sparse_attn_decode"
    )

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_indptr is not None
        assert extra_indices.ndim == 1, (
            f"expected extra_indices=[nnz], got {extra_indices.shape}"
        )
        assert extra_indptr.ndim == 1, (
            f"expected extra_indptr=[b+1], got {extra_indptr.shape}"
        )
        extra_indices = _as_int32_contiguous_1d(extra_indices)
        extra_indptr = _as_int32_contiguous_1d(extra_indptr)
        assert extra_indptr.numel() == num_queries + 1, (
            f"expected extra_indptr shape [{num_queries + 1}], got {extra_indptr.shape}"
        )
    else:
        # Pass valid (but unused-by-dead-code) pointers so the constexpr
        # branch in the kernel doesn't materialize.
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    out = torch.empty_like(q, dtype=torch.bfloat16)
    # Prefer the Gluon kernel, but its 32-bit buffer_load offsets require both
    # paged caches to fit within the 2 GiB descriptor cap; else fall back to
    # Triton.
    if (
        _gluon_sparse_attn_decode_kernel is not None
        and _fits_buffer_descriptor(main_cache)
        and _fits_buffer_descriptor(extra_cache)
    ):
        kernel = _gluon_sparse_attn_decode_kernel
    else:
        kernel = _sparse_attn_decode_kernel
    grid = lambda META: (  # noqa: E731
        num_queries,
        triton.cdiv(num_heads, META["BLOCK_H"]),
    )
    kernel[grid](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        scale,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=triton.next_power_of_2(nope_head_dim),
        ROPE_DIM=rope_head_dim,
        IS_FNUZ=_is_fp8_fnuz(),
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
        q: ``[N, H, D]`` queries (bf16). ``D = nope_head_dim + rope_head_dim``.
        kv: ``[N_kv, 1, D]`` KV pool (bf16). The ``1`` lane dim is squeezed
            inside this wrapper.
        indices: ``[N, max_topk]`` dense indices into ``kv``'s flat ``[N_kv, D]``
            view, with ``-1`` sentinels. Ignored when ``ragged_indices`` and
            ``ragged_indptr`` are both supplied.
        topk_length: ``[N]`` per-row valid count for ``indices`` (used to
            shrink the dense → ragged conversion). May be ``None`` to recompute
            from ``(indices >= 0).sum(-1)``.
        scale: softmax scale.
        head_dim, nope_head_dim, rope_head_dim: must be ``(512, 448, 64)``.
        attn_sink: optional ``[H]`` per-head softmax-denom bias (fp32).
        output: ``[N, H, D]`` destination (any dtype). Filled in place.
        ragged_indices, ragged_indptr: optional CSR alternative to
            ``(indices, topk_length)``. Preferred when callers already have the
            ragged form materialized (e.g. across CUDA-graph captures).
    """
    assert kv.ndim == 3 and kv.shape[1] == 1, (
        f"sparse_attn_prefill expects kv=[skv,1,d], got {kv.shape}"
    )
    _validate_dsv4_sparse_dims(
        head_dim, nope_head_dim, rope_head_dim, "sparse_attn_prefill"
    )

    if ragged_indices is None or ragged_indptr is None:

        indices_2d = indices.reshape(indices.shape[0], -1)

        ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
            indices_2d,
            topk_length
            if topk_length is not None
            else (indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=kv.shape[0],
        )

    attn_sink = attn_sink[:, q.shape[1]] if attn_sink else None
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


def sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    """DSV4 sparse-MLA decode — fp8_ds_mla paged cache with SWA + topk.

    The kernel walks two CSR-indexed sparse passes that share one running
    softmax statistic: the "main" pass over ``swa_k_cache`` (SWA) and an
    optional "extra" pass over ``kv_cache`` (top-k). When ``swa_only`` is
    True only the main pass runs.

    Args:
        q: ``[N, H, D]`` decode queries (bf16). ``D = 448 + 64 = 512``.
        kv_cache: optional ``[blocks, block_size, 576]`` (uint8 fp8_ds_mla)
            top-k cache. Required iff not ``swa_only``.
        swa_k_cache: ``[blocks, block_size, 576]`` (uint8 fp8_ds_mla) SWA cache.
        swa_only: when True, skip the extra (top-k) pass entirely.
        topk_indices: dense ``[N, topk_max]`` indices into ``kv_cache``'s flat
            slot space. Optional iff ``topk_ragged_*`` are supplied.
        topk_lens: ``[N]`` valid count for ``topk_indices``.
        swa_indices: dense ``[N, swa_max]`` indices into ``swa_k_cache``'s flat
            slot space.
        swa_lens: ``[N]`` valid count for ``swa_indices``.
        swa_ragged_indices, swa_ragged_indptr: optional CSR form of the SWA
            pass. Preferred when caller has them materialized.
        topk_ragged_indices, topk_ragged_indptr: optional CSR form of the top-k
            pass. Preferred when caller has them materialized.
        attn_sink: optional ``[H]`` per-head softmax-denom bias (fp32).
        scale: softmax scale.
        head_dim, nope_head_dim, rope_head_dim: must be ``(512, 448, 64)``.
        output: ``[N, H, D]`` destination (any dtype). Filled in place.
    """
    assert swa_k_cache.dtype == torch.uint8, (
        f"sparse_attn_decode expects uint8 SWA cache, "
        f"got {swa_k_cache.dtype}"
    )
    _validate_dsv4_sparse_dims(
        head_dim, nope_head_dim, rope_head_dim, "sparse_attn_decode"
    )

    main_indices = swa_indices.reshape(swa_indices.shape[0], -1)
    main_lens = swa_lens or (main_indices >= 0).sum(dim=-1, dtype=torch.int32)

    if swa_ragged_indices is None or swa_ragged_indptr is None:
        swa_ragged_indices, swa_ragged_indptr = build_ragged_indices_from_dense(
            main_indices,
            main_lens,
            num_rows=swa_k_cache.shape[0] * swa_k_cache.shape[1],
        )

    attn_sink = attn_sink[:, q.shape[1]] if attn_sink else None

    extra_cache = None
    if not swa_only:
        assert kv_cache is not None
        assert topk_indices is not None or (
            topk_ragged_indices is not None and topk_ragged_indptr is not None
        )
        assert kv_cache.dtype == torch.uint8, (
            f"sparse_attn_decode expects uint8 extra cache, "
            f"got {kv_cache.dtype}"
        )

        extra_cache = kv_cache
        if topk_indices is not None:
            extra_indices = topk_indices.reshape(topk_indices.shape[0], -1)
            extra_lens = topk_lens or (extra_indices >= 0).sum(dim=-1, dtype=torch.int32)

            if topk_ragged_indices is None or topk_ragged_indptr is None:
                topk_ragged_indices, topk_ragged_indptr = build_ragged_indices_from_dense(
                    extra_indices,
                    extra_lens,
                    num_rows=extra_cache.shape[0] * extra_cache.shape[1],
                )

    attn_out =  _sparse_attn_decode_ragged(
        q=q,
        main_cache=swa_k_cache,
        main_indices=swa_ragged_indices,
        main_indptr=swa_ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        extra_cache=extra_cache,
        extra_indices=topk_ragged_indices,
        extra_indptr=topk_ragged_indptr,
    )

    output.copy_(attn_out.to(output.dtype))

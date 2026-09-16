# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OPUS-based paged-attention decode for gfx950.

Follows the sp3 kernel ``PA_A16W16_*_1TG_4W_16mx1_64nx4``: one thread group of
4 waves per (sequence, kv-head), 16 query rows, waves split along the KV axis
for Q*K and along the head axis for P*V.

The user-facing entry is :func:`pa_decode_opus`; it forwards to the
JIT-compiled HIP kernel via :func:`pa_decode_opus_fwd`.

Two dtype configurations are compiled, both fixed at head dim ``128`` and
GQA ratio ``<= 16`` (``qlen * gqa`` under packed MTP; GQA 16 also
allows ``qlen`` 2..4 via a fused token loop):

* :func:`pa_decode_opus` -- ``bf16`` Q/K/V/O (A16W16, no KV quantization), on
  ``v_mfma_f32_16x16x32_bf16``. Page size ``16``. K cache packed as
  ``[num_blocks, num_kv_heads, 128/8, 16, 8]``, V cache as
  ``[num_blocks, num_kv_heads, 128, 16]``.
* :func:`pa_decode_opus_fp8` -- ``fp8`` e4m3 Q/K/V with ``bf16`` O (A8W8), on the
  full-rate ``v_mfma_f32_16x16x128_f8f6f4``. Page size ``16``. Same layouts with
  the pack factor ``x = 16 bytes / itemsize`` now ``16``, so the K cache is
  ``[num_blocks, num_kv_heads, 128/16, 16, 16]``.
* :func:`pa_decode_opus_a16w8` -- ``bf16`` Q against the same fp8 KV, no sink.
  Page sizes ``16`` and ``128``, including MTP when ``q`` is 4-D. Plain or
  transposed V, per-tensor or per-token KV scales.

In both cases ``x`` is ``16 bytes / itemsize``, the standard vLLM packing.

See ``aiter/csrc/include/pa_decode_opus.h`` for the C++ API.
"""

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard
from ..utility import dtypes

MD_NAME = "module_pa_decode_opus"

_HEAD_DIM = 128
_PAGE_SIZE = 16
_A16W8_PAGE_SIZES = (16, 128)
_MAX_GQA = 16


@compile_ops("module_pa_decode_opus", develop=True)
def pa_decode_opus_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


@compile_ops("module_pa_decode_opus", develop=True)
def pa_decode_opus_fp8_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
) -> None: ...


@compile_ops("module_pa_decode_opus", develop=True)
def pa_decode_opus_fp8_ps_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    split_o: torch.Tensor,
    split_lse: torch.Tensor,
    softmax_scale: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
) -> None: ...


@compile_ops("module_pa_decode_opus", develop=True)
def pa_decode_opus_a16w8_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    k_scale: float,
    v_scale: float,
    k_scale_map: torch.Tensor | None = None,
    v_scale_map: torch.Tensor | None = None,
) -> None: ...


@compile_ops("module_pa_decode_opus", develop=True)
def pa_decode_opus_a16w8_ps_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    split_o: torch.Tensor,
    split_lse: torch.Tensor,
    softmax_scale: float,
    k_scale: float,
    v_scale: float,
) -> None: ...


def _check_shapes(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    head_dim: int = _HEAD_DIM,
    page_sizes: tuple[int, ...] = (_PAGE_SIZE,),
    *,
    shuffled_v: bool = False,
) -> None:
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_decode_opus requires gfx950, got {gfx}")

    if q.size(-1) != head_dim:
        raise RuntimeError(
            f"this entry point is compiled for head_dim={head_dim}, got {q.size(-1)}"
        )
    page_size = (
        v_cache.size(2) * v_cache.size(4)
        if shuffled_v or v_cache.dim() == 5
        else v_cache.size(-1)
    )
    if page_size not in page_sizes:
        raise RuntimeError(
            f"this entry point compiles page sizes {page_sizes}, got {page_size}"
        )

    # 4-D q is MTP and carries a token dim between batch and heads, so read the head
    # dims from the tail. qlen is 1 on the 3-D decode shape.
    num_heads, num_kv_heads = q.size(-2), k_cache.size(1)
    qlen = q.size(1) if q.dim() == 4 else 1
    if num_heads % num_kv_heads != 0:
        raise RuntimeError(
            f"num_heads={num_heads} not divisible by num_kv_heads={num_kv_heads}"
        )
    # Packed MTP keeps every (token, head) pair in one 16-row tile. GQA 16 with
    # qlen 2..4 instead loops one token per tile over a shared KV walk.
    gqa = num_heads // num_kv_heads
    if qlen * gqa > _MAX_GQA:
        if gqa != _MAX_GQA or qlen > 4:
            raise RuntimeError(
                f"qlen * GQA ratio must be <= {_MAX_GQA} (or GQA=={_MAX_GQA} "
                f"and qlen<=4 for the token loop), got {qlen} * {gqa}"
            )


def _pa_decode_opus_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return out if out is not None else torch.empty_like(q)


@torch_compile_guard(mutates_args=["out"], gen_fake=_pa_decode_opus_fake)
def pa_decode_opus(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged-attention decode over a block table, backed by the OPUS gfx950 kernel.

    The trailing ``out`` keyword is an aiter-only convenience for callers that
    want to reuse a pre-allocated output buffer; pass ``None`` (the default) to
    have one allocated for you.

    Args:
      q:            ``[batch, num_heads, 128]`` bf16 query for decode, or
                    ``[batch, qlen, num_heads, 128]`` for multi-token prediction.
                    The 4-D form is tail-causal: query token ``i`` attends the
                    first ``context_len - qlen + 1 + i`` KV tokens. One MFMA tile
                    row goes to each (token, head) pair, so ``qlen * gqa <= 16``.
      k_cache:      ``[num_blocks, num_kv_heads, D/K_PACK, PAGE, K_PACK]`` bf16 key cache.
      v_cache:      ``[num_blocks, num_kv_heads, 128, 16]`` bf16 value cache.
      block_tables: ``[batch, max_blocks_per_batch_row]`` int32 page indices.
      context_lens: ``[batch]`` int32 KV length per batch row.
      softmax_scale: float scalar applied to the QK^T scores.
      out:          Optional output buffer, shaped like ``q``.

    Returns:
      ``out``, shaped like ``q``, bf16.
    """
    if q.dtype != torch.bfloat16:
        raise RuntimeError(f"pa_decode_opus expects bf16 q, got {q.dtype}")
    if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise RuntimeError(
            f"KV cache dtype mismatch: k_cache={k_cache.dtype}, "
            f"v_cache={v_cache.dtype}, q={q.dtype}"
        )
    _check_shapes(q, k_cache, v_cache)

    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={tuple(q.shape)} dtype={q.dtype}"
        )

    # qlen == 1 is decode wearing a token dim: the causal window covers the whole
    # context, so the mask can never fire. Drop the dim and take the decode kernel,
    # which peels one tile instead of two.
    q_in, out_in = q, out
    if q.dim() == 4 and q.size(1) == 1:
        q_in, out_in = q.squeeze(1), out.squeeze(1)

    pa_decode_opus_fwd(
        q_in,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out_in,
        float(softmax_scale),
    )
    return out


def _pa_decode_opus_fp8_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    return torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_pa_decode_opus_fp8_fake)
def pa_decode_opus_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """A8W8 paged-attention decode: fp8 Q/K/V through ``v_mfma_f32_16x16x128_f8f6f4``.

    Scales are per-tensor and in the dequant direction, i.e. the kernel reconstructs
    ``x_true ~= x_fp8 * x_scale``. They cost nothing at runtime: ``q_scale * k_scale``
    rides on the softmax temperature and ``v_scale`` on the final ``1/l``.

    Args:
      q:            ``[batch, num_heads, 128]`` fp8 query, already quantized.
      k_cache:      ``[num_blocks, num_kv_heads, 128/16, 16, 16]`` fp8 key cache.
      v_cache:      ``[num_blocks, num_kv_heads, 128, 16]`` fp8 value cache.
      block_tables: ``[batch, max_blocks_per_batch_row]`` int32 page indices.
      context_lens: ``[batch]`` int32 KV length per batch row.
      softmax_scale: float scalar applied to the QK^T scores.
      q_scale, k_scale, v_scale: per-tensor dequant scales.
      out:          Optional ``[batch, num_heads, 128]`` bf16 output buffer.

    Returns:
      ``out`` (``[batch, num_heads, 128]`` bf16).
    """
    if q.dtype != dtypes.fp8:
        raise RuntimeError(f"pa_decode_opus_fp8 expects {dtypes.fp8} q, got {q.dtype}")
    if k_cache.dtype != dtypes.fp8 or v_cache.dtype != dtypes.fp8:
        raise RuntimeError(
            f"KV cache dtype mismatch: k_cache={k_cache.dtype}, "
            f"v_cache={v_cache.dtype}, expected {dtypes.fp8}"
        )
    _check_shapes(q, k_cache, v_cache)

    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    elif out.shape != q.shape or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={tuple(q.shape)} dtype={torch.bfloat16}"
        )

    pa_decode_opus_fp8_fwd(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out,
        float(softmax_scale),
        float(q_scale),
        float(k_scale),
        float(v_scale),
    )
    return out


# The shared reduce kernel is instantiated for a fixed set of head counts.
_REDUCE_NUM_HEADS = frozenset({1, 2, 4, 8, 10, 16, 32, 40, 48, 64, 96, 128})


class PaDecodeOpusPsPlan:
    """Work queue and partial buffers for one persistent-decode shape.

    Built separately from the launch because the schedule depends only on the context
    lengths: a caller whose lengths are stable can build it once, and a benchmark can
    keep the metadata kernel out of the timed region.
    """

    __slots__ = (
        "context_lens",
        "final_lse",
        "kv_indices",
        "kv_indptr",
        "qo_indptr",
        "reduce_final_map",
        "reduce_indptr",
        "reduce_partial_map",
        "split_lse",
        "split_o",
        "work_indptr",
        "work_info",
        "work_metadata_ptrs",
    )

    def __init__(self, **kw):
        for name, value in kw.items():
            setattr(self, name, value)


def pa_decode_opus_ps_plan(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int = _HEAD_DIM,
    page_size: int = _PAGE_SIZE,
) -> PaDecodeOpusPsPlan:
    """Compact the block table to CSR and build the work queue for it.

    The metadata kernel addresses pages through ``kv_indptr``/``kv_indices``, so the
    rectangular block table is compacted down to the pages each row actually uses.
    ``kv_granularity`` is pinned to the page size, which is what makes the work items'
    kv_start/kv_end plain page indices -- the kernel reads them that way.
    """
    from .attention import get_pa_metadata_info_v1, get_pa_metadata_v1

    if num_heads not in _REDUCE_NUM_HEADS:
        raise RuntimeError(
            f"the shared reduce kernel has no instantiation for num_heads={num_heads}; "
            f"supported: {sorted(_REDUCE_NUM_HEADS)}"
        )

    device = block_tables.device
    batch = context_lens.numel()
    lens = context_lens.to(torch.int32)
    pages = (lens + page_size - 1) // page_size
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    torch.cumsum(pages, 0, out=kv_indptr[1:])
    # Gather each row's used pages into one flat list.
    col = torch.arange(block_tables.size(1), device=device)
    kv_indices = (
        block_tables[col.unsqueeze(0) < pages.unsqueeze(1)].to(torch.int32).contiguous()
    )
    # Decode: one query token per request.
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)

    buffers = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in get_pa_metadata_info_v1(batch, num_kv_heads)
    ]
    (
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = buffers

    get_pa_metadata_v1(
        qo_indptr,
        kv_indptr,
        lens,
        num_heads // num_kv_heads,
        num_kv_heads,
        False,
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=page_size,
        block_size=page_size,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=-1,
    )

    num_partial = reduce_partial_map.size(0)
    return PaDecodeOpusPsPlan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        context_lens=lens,
        work_metadata_ptrs=work_metadata_ptrs,
        work_indptr=work_indptr,
        work_info=work_info,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        split_o=torch.empty(
            (num_partial, 1, num_heads, head_dim), dtype=torch.float32, device=device
        ),
        split_lse=torch.empty(
            (num_partial, 1, num_heads, 1), dtype=torch.float32, device=device
        ),
        final_lse=torch.empty((batch, num_heads), dtype=torch.float32, device=device),
    )


def pa_decode_opus_fp8_ps(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    plan: PaDecodeOpusPsPlan,
    softmax_scale: float,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Persistent A8W8 decode: one workgroup per CU, draining ``plan``'s work queue.

    Same numerics as :func:`pa_decode_opus_fp8`, different work distribution. Work
    items the metadata marked as splits go through the shared reduce kernel; the rest
    write ``out`` in place, so the reduce only touches rows that were actually split.
    """
    from .attention import pa_reduce_v1

    if q.dtype != dtypes.fp8:
        raise RuntimeError(
            f"pa_decode_opus_fp8_ps expects {dtypes.fp8} q, got {q.dtype}"
        )
    if k_cache.dtype != dtypes.fp8 or v_cache.dtype != dtypes.fp8:
        raise RuntimeError(
            f"KV cache dtype mismatch: k_cache={k_cache.dtype}, "
            f"v_cache={v_cache.dtype}, expected {dtypes.fp8}"
        )
    _check_shapes(q, k_cache, v_cache)

    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)

    pa_decode_opus_fp8_ps_fwd(
        q,
        k_cache,
        v_cache,
        plan.kv_indptr,
        plan.kv_indices,
        plan.context_lens,
        out,
        plan.work_indptr,
        plan.work_info,
        plan.split_o,
        plan.split_lse,
        float(softmax_scale),
        float(q_scale),
        float(k_scale),
        float(v_scale),
    )
    pa_reduce_v1(
        plan.split_o,
        plan.split_lse,
        plan.reduce_indptr,
        plan.reduce_final_map,
        plan.reduce_partial_map,
        1,
        out,
        plan.final_lse,
    )
    return out


def _check_a16w8(q, k_cache, v_cache):
    if q.dtype != torch.bfloat16:
        raise RuntimeError(f"pa_decode_opus_a16w8 expects bf16 q, got {q.dtype}")
    if k_cache.dtype != dtypes.fp8 or v_cache.dtype != dtypes.fp8:
        raise RuntimeError(
            f"KV cache dtype mismatch: k_cache={k_cache.dtype}, "
            f"v_cache={v_cache.dtype}, expected {dtypes.fp8}"
        )
    _check_shapes(q, k_cache, v_cache, page_sizes=_A16W8_PAGE_SIZES)


def _split_kv_scale(scale, scale_map):
    if scale_map is not None:
        return 1.0, scale_map
    if isinstance(scale, torch.Tensor):
        if scale.dim() == 0 or scale.numel() == 1:
            return float(scale.detach().float().reshape(()).item()), None
        return 1.0, scale
    return float(scale), None


def pa_decode_opus_a16w8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    k_scale: float | torch.Tensor = 1.0,
    v_scale: float | torch.Tensor = 1.0,
    out: torch.Tensor | None = None,
    k_scale_map: torch.Tensor | None = None,
    v_scale_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """A16W8 decode: bf16 Q against the fp8 KV cache, no attention sink.

    Same KV layouts and the same bandwidth as :func:`pa_decode_opus_fp8` -- Q is under
    1% of the traffic. The matrix core needs both operands the same width, so the
    kernel quantizes Q to fp8 once per query row on the way in; there is no ``q_scale``
    argument because it derives and applies that itself.

    Page size is ``16`` or ``128``. 3-D ``q`` is decode; 4-D
    ``[batch, qlen, num_heads, 128]`` is multi-token prediction with a tail-causal
    mask. Packed MTP needs ``qlen * gqa <= 16``; GQA 16 with ``qlen`` 2..4
    loops one query token per tile over a shared KV walk.

    ``v_cache`` may be the plain 4-D layout ``[blocks, kv_heads, D, PAGE]`` or the
    transposed 5-D layout ``[blocks, kv_heads, PAGE/x, D, x]``. ``k_scale`` /
    ``v_scale`` may be a per-tensor float or a per-token map
    ``[blocks, kv_heads, PAGE, 1]``.
    """
    _check_a16w8(q, k_cache, v_cache)
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)

    k_scale, k_scale_map = _split_kv_scale(k_scale, k_scale_map)
    v_scale, v_scale_map = _split_kv_scale(v_scale, v_scale_map)
    if (k_scale_map is None) != (v_scale_map is None):
        raise RuntimeError("k and v per-token scale maps must be supplied together")
    if k_scale_map is not None:
        k_scale_map = k_scale_map.contiguous()
        v_scale_map = v_scale_map.contiguous()

    # qlen == 1 is decode wearing a token dim: drop it so decode keeps the single
    # peeled tile rather than MTP's extra masked one.
    q_in, out_in = q, out
    if q.dim() == 4 and q.size(1) == 1:
        q_in, out_in = q.squeeze(1), out.squeeze(1)

    pa_decode_opus_a16w8_fwd(
        q_in,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out_in,
        float(softmax_scale),
        float(k_scale),
        float(v_scale),
        k_scale_map,
        v_scale_map,
    )
    return out


def pa_decode_opus_a16w8_ps(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    plan: PaDecodeOpusPsPlan,
    softmax_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Persistent A16W8: :func:`pa_decode_opus_a16w8` on ``plan``'s work queue.

    MTP (4-D ``q``) is not supported here: ``split_o`` / ``split_lse`` and
    ``pa_reduce_v1`` are indexed by query head, not by (token, head). Use
    :func:`pa_decode_opus_a16w8` for ``qlen > 1``.
    """
    from .attention import pa_reduce_v1

    if q.dim() == 4:
        raise RuntimeError(
            "pa_decode_opus_a16w8_ps does not support MTP; "
            "use pa_decode_opus_a16w8 for qlen > 1"
        )

    _check_a16w8(q, k_cache, v_cache)
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)

    pa_decode_opus_a16w8_ps_fwd(
        q,
        k_cache,
        v_cache,
        plan.kv_indptr,
        plan.kv_indices,
        plan.context_lens,
        out,
        plan.work_indptr,
        plan.work_info,
        plan.split_o,
        plan.split_lse,
        float(softmax_scale),
        float(k_scale),
        float(v_scale),
    )
    pa_reduce_v1(
        plan.split_o,
        plan.split_lse,
        plan.reduce_indptr,
        plan.reduce_final_map,
        plan.reduce_partial_map,
        1,
        out,
        plan.final_lse,
    )
    return out


_GPTOSS_HEAD_DIMS = (64, 128)
# gpt-oss ships 256-token pages; smaller pages use the same cache layout.
_GPTOSS_PAGE_SIZES = (16, 128, 256)


@compile_ops("module_pa_decode_opus", develop=True)
def pa_decode_opus_gptoss_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
) -> None: ...


@compile_ops("module_pa_decode_opus", develop=True)
def pa_decode_opus_gptoss_ps_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    sink: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    split_o: torch.Tensor,
    split_lse: torch.Tensor,
    softmax_scale: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
) -> None: ...


def _check_gptoss(q, k_cache, v_cache, sink, *, allow_shuffled_v=False):
    if q.dtype not in (dtypes.fp8, torch.bfloat16):
        raise RuntimeError(f"gpt-oss q must be {dtypes.fp8} or bf16, got {q.dtype}")
    if k_cache.dtype != dtypes.fp8 or v_cache.dtype != dtypes.fp8:
        raise RuntimeError(
            f"KV cache must be {dtypes.fp8}, got k={k_cache.dtype} v={v_cache.dtype}"
        )
    # 4-D q is MTP: heads sit at -2, not 1 (which is qlen).
    num_heads = q.size(-2)
    if sink.dtype != torch.float32 or sink.dim() != 1 or sink.size(0) != num_heads:
        raise RuntimeError(
            f"sink must be fp32 [num_heads={num_heads}], got {sink.dtype} "
            f"{tuple(sink.shape)}"
        )
    head_dim = q.size(-1)
    if head_dim not in _GPTOSS_HEAD_DIMS:
        raise RuntimeError(
            f"gpt-oss head_dim must be one of {_GPTOSS_HEAD_DIMS}, got {head_dim}"
        )
    shuffled_v = v_cache.dim() == 5
    if shuffled_v:
        if not allow_shuffled_v:
            raise RuntimeError("shuffled V is only supported by ordinary gpt-oss decode")
        if (
            q.dtype != dtypes.fp8
            or q.dim() not in (3, 4)
            or (q.dim() == 4 and q.size(1) != 1)
            or tuple(q.shape[-2:]) != (64, 128)
        ):
            raise RuntimeError("shuffled V requires single-token FP8 Q64/D128")
        if k_cache.dim() != 5 or tuple(k_cache.shape[1:]) != (4, 8, 128, 16):
            raise RuntimeError("shuffled V requires K [num_blocks, 4, 8, 128, 16]")
        if tuple(v_cache.shape) != (k_cache.size(0), 4, 8, 128, 16):
            raise RuntimeError("shuffled V must be [num_blocks, 4, 8, 128, 16]")
        if not k_cache.is_contiguous() or not v_cache.is_contiguous():
            raise RuntimeError("shuffled V and K must be contiguous")
    _check_shapes(
        q, k_cache, v_cache, head_dim, _GPTOSS_PAGE_SIZES, shuffled_v=shuffled_v
    )


def pa_decode_opus_gptoss(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale: float,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """gpt-oss decode: 64/128-dim heads, fp8 KV, learned per-head attention sink.

    ``q`` may be fp8 (A8W8, ``q_scale`` applies) or bf16 (A16W8, ``q_scale`` ignored --
    the kernel derives one per query row); its dtype picks the path. 3-D ``q`` is
    decode; 4-D ``[batch, qlen, num_heads, head_dim]`` is multi-token prediction with a
    tail-causal mask (query ``i`` attends ``context_len - qlen + 1 + i`` KV tokens).
    One MFMA tile row is a (token, head) pair, so ``qlen * gqa <= 16`` -- GQA 8
    therefore allows ``qlen`` 2, while GQA 16 allows only 1. Sink is one logit per query head, shared
    across those tokens.

    ``sink`` is one fp32 logit per query head in the same scaled-logit domain as
    ``(q.k) * softmax_scale``, the convention the gfx1250 kernel uses. It behaves as a
    KV column with no value: it enters the softmax denominator and nothing else.

    ``v_cache`` normally has shape ``[num_blocks, num_kv_heads, D, page]``.
    Single-token A8W8 Q64/KV4/D128/page128 also accepts contiguous shuffled V
    ``[num_blocks, 4, 8, 128, 16]``, with axes ``[page_id, head, token/16, D, 16]``.
    The caller supplies this layout; this function does not shuffle the cache.
    """
    _check_gptoss(q, k_cache, v_cache, sink, allow_shuffled_v=True)
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)

    # qlen == 1 is decode wearing a token dim: drop it so decode keeps the single
    # peeled tile rather than MTP's extra masked one.
    q_in, out_in = q, out
    if q.dim() == 4 and q.size(1) == 1:
        q_in, out_in = q.squeeze(1), out.squeeze(1)

    pa_decode_opus_gptoss_fwd(
        q_in,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out_in,
        sink,
        float(softmax_scale),
        float(q_scale),
        float(k_scale),
        float(v_scale),
    )
    return out


def pa_decode_opus_gptoss_ps(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    plan: PaDecodeOpusPsPlan,
    sink: torch.Tensor,
    softmax_scale: float,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Persistent gpt-oss decode, on the same work queue as the other ps entries.

    MTP (4-D ``q``) is not supported here: ``split_o`` / ``split_lse`` and
    ``pa_reduce_v1`` are indexed by query head, not by (token, head). Use
    :func:`pa_decode_opus_gptoss` for ``qlen > 1``.
    """
    from .attention import pa_reduce_v1

    if q.dim() == 4:
        raise RuntimeError(
            "pa_decode_opus_gptoss_ps does not support MTP; "
            "use pa_decode_opus_gptoss for qlen > 1"
        )

    _check_gptoss(q, k_cache, v_cache, sink)
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)

    pa_decode_opus_gptoss_ps_fwd(
        q,
        k_cache,
        v_cache,
        plan.kv_indptr,
        plan.kv_indices,
        plan.context_lens,
        out,
        sink,
        plan.work_indptr,
        plan.work_info,
        plan.split_o,
        plan.split_lse,
        float(softmax_scale),
        float(q_scale),
        float(k_scale),
        float(v_scale),
    )
    pa_reduce_v1(
        plan.split_o,
        plan.split_lse,
        plan.reduce_indptr,
        plan.reduce_final_map,
        plan.reduce_partial_map,
        1,
        out,
        plan.final_lse,
    )
    return out


__all__ = [
    "PaDecodeOpusPsPlan",
    "pa_decode_opus",
    "pa_decode_opus_a16w8",
    "pa_decode_opus_a16w8_fwd",
    "pa_decode_opus_a16w8_ps",
    "pa_decode_opus_a16w8_ps_fwd",
    "pa_decode_opus_fp8",
    "pa_decode_opus_fp8_fwd",
    "pa_decode_opus_fp8_ps",
    "pa_decode_opus_fp8_ps_fwd",
    "pa_decode_opus_fwd",
    "pa_decode_opus_gptoss",
    "pa_decode_opus_gptoss_fwd",
    "pa_decode_opus_gptoss_ps",
    "pa_decode_opus_gptoss_ps_fwd",
    "pa_decode_opus_ps_plan",
]

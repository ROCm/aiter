# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fp8 unified-attention backend for gfx950.

Adapts the vendored ``flash_attn_dualwave_swp`` fp8 kernel
(``kernels/flash_attn_fp8_gfx950.py``) to the ``unified_attention`` calling
convention vLLM/SGLang import directly, so a supported gfx950 fp8 paged call
routes here instead of Triton.

Dispatch is a pure predicate returning ``None`` when it can't serve the
config, falling through to Triton unchanged (matches
``flydsl_flash_attn_varlen_func`` / the flydsl branch in
``ops/gemm_op_a8w8.py``) -- deliberately no dispatch env var, since a code gate
states an arch/shape-scoped choice more honestly than another undocumented
backend flag. Force Triton by setting ``_FLYDSL_UNIFIED_ATTN_ARCH`` to ``False`` in
the Triton module. The one tuning knob is the split-count cap,
``AITER_UNIFIED_ATTN_MAX_KV_SPLITS`` (see ``_MAX_SEGMENTS``).

Two tiers chosen at call time from the machine-fill deficit (``_split_count``
and the gate in ``flydsl_unified_attention``), not from all-decode-ness: an
underfilled launch (few workgroups each serially scanning full KV depth)
takes packed + split-K, partitioning each sequence's KV across split
workgroups and combining partials; a full-pass batch stays on the
single-pass packed kernel. Split-K also serves the low-chunk mixed case
because the combine rebases its O write on ``cu_seqlens_q``
(``DualwaveSplitKCombineContext.init_descriptors``), so unequal query
lengths combine correctly.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch

from .kernels.flash_attn_dualwave_common import dualwave_splitk_workspace_elems
from .kernels.flash_attn_fp8_gfx950 import build_flash_attn_dualwave_swp_fp8_module
from .utils import is_flydsl_available

__all__ = ["flydsl_unified_attention"]


# Page size is structural, not a builder parameter: the paged path addresses KV
# in BLOCK_N-sized pages and BLOCK_N is pinned at 64 by the MFMA tile.
_PAGE_SIZE = 64

# Head dim is fixed by the kernel (it raises on anything else).
_HEAD_DIM = 128


def _target_num_prgms() -> int:
    """Split-K fill target: a launch with num_2d_prgms base workgroups is "full"
    at num_2d_prgms >= this value.

    Both full-chip gfx950 parts (MI350X, MI355X) are 256 CUs, so this is a
    device CU-count query, not a hardcoded arch constant -- it also handles
    CU-partitioned modes (CPX/NPS) where fewer CUs are exposed. Falls back to
    256 (the full-chip count) if the query fails.
    """
    try:
        from aiter.ops.triton.utils.device_info import get_num_sms

        return get_num_sms()
    except Exception:  # noqa: BLE001
        return 256


_TARGET_NUM_PRGMS = _target_num_prgms()


def _env_max_kv_splits(default: int = 16) -> int:
    """Auto-dispatch split-count cap, overridable by environment.

    Split-K selection cannot be tuned perfectly from shape alone (a lesson from
    CK), so the cap is an override knob rather than a derived guess.
    ``AITER_UNIFIED_ATTN_MAX_KV_SPLITS`` raises or lowers it; a non-integer or
    non-positive value is ignored and the default is used.
    """
    raw = os.environ.get("AITER_UNIFIED_ATTN_MAX_KV_SPLITS")
    if raw is None:
        return default
    try:
        n = int(raw)
    except ValueError:
        return default
    return n if n >= 1 else default


# Cap on the auto-dispatched split count. Default 16: no regression across
# tested decode shapes while capturing most of the long-context win. >16
# helps only very long contexts and regresses short ones, since _split_count
# keys on machine fill, not KV depth (counts are verified correct to 128).
# Raise via AITER_UNIFIED_ATTN_MAX_KV_SPLITS for long-context workloads.
_MAX_SEGMENTS = _env_max_kv_splits(16)

# block_m, and with it the largest GQA group that can pack into the M dimension.
# _make_dualwave_swp_fp8_traits asserts block_m % gqa_group_size == 0.
_BLOCK_M = 256

# The block table is staged through a fixed LDS window of PAGED_BT_LDS_SIZE=2048
# entries. Past that the stager writes only `local_tile < segment_tiles` slots
# and silently drops the remaining page ids -- wrong output, not a fault -- so
# the KV length must be capped here: 2048 pages * 64 tokens = 131072 tokens.
_MAX_KV_TILES = 2048

_FP8_DTYPE = torch.float8_e4m3fn


@lru_cache(maxsize=64)
def _get_kernel(
    num_heads: int,
    num_kv_heads: int,
    causal: bool,
    out_dtype_str: str,
    use_sinks: bool,
    num_kv_splits: int = 1,
):
    """Build (and cache) the paged+varlen fp8 launcher.

    Keyed on head counts, mask mode, output dtype, and split count; every
    other builder argument is pinned by the support gate or left at default.

    ``num_kv_splits`` selects the tier: at 1, the builder auto-enables
    M-dimension packing for GQA > 1:1 (``gqa_pack_m=None``) and builds the
    single-pass kernel; at > 1, ``gqa_pack_m`` defaults to unpacked, so the
    decode tier passes True explicitly to get the packed+split-K binary.
    ``GQA_PACK_M`` and ``NUM_KV_SPLITS`` both key the JIT cache so the two
    binaries cannot alias.

    ``causal``, ``out_dtype_str``, ``use_sinks`` key the cache too since each
    changes compiled code (mask path, store packer, sink init/epilogue).

    Do not key on batch/seqlen/device: ``_run_compiled`` memoizes the
    compiled function on the returned closure, so a finer key would defeat
    that cache and recompile per shape.
    """
    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        causal=causal,
        dtype_str="fp8",
        out_dtype_str=out_dtype_str,
        use_sinks=use_sinks,
        num_kv_heads=num_kv_heads,
        varlen=True,
        paged=True,
        num_kv_splits=num_kv_splits,
        gqa_pack_m=True if num_kv_splits > 1 else None,
    )


def _split_count(num_2d_prgms: int) -> int:
    """Split count from the machine-fill deficit; 1 means single-pass.

    The measured no-oversubscribe rule (NOT Triton's ceil/round-up/MIN=8): the
    largest power of two such that the launch does not oversubscribe the CU
    count, capped at _MAX_SEGMENTS. No MIN floor -- b=9 needs 4, below Triton's
    floor of 8. Below 2 there is no split to take, so the caller falls back to
    single-pass.

    Verified against the measured best split count: num_2d = 4*b for a
    b-sequence GQA-16 decode gives b=7 -> 8, b=8 -> 8, b=9 -> 4.
    """
    n = 1
    while n * 2 <= _MAX_SEGMENTS and num_2d_prgms * (n * 2) <= _TARGET_NUM_PRGMS:
        n *= 2
    return n if n >= 2 else 1


def _strides_ok(
    q, out, k, v, block_table, num_query_heads, num_kv_heads, head_size
) -> bool:
    """Check the exact layout the kernel's two scalar strides imply.

    The launcher takes one ``stride_q_n`` and one ``stride_kv_n`` and derives
    every offset from them, so anything else must be declined rather than
    silently mis-addressed.
    """
    if q.stride(2) != 1 or q.stride(1) != head_size:
        return False
    if out.stride(2) != 1 or out.stride(1) != head_size:
        return False
    # One stride_q_n argument serves the Q read, the O write, and BOTH buffer
    # num_records bounds (init_descriptors), so Q and O must agree on it.
    if q.stride(0) != out.stride(0):
        return False
    # `_run_compiled` launches on `q.reshape(-1)` / `out.reshape(-1)`, baking
    # the tensor's full memref into the kernel cache signature. A padded
    # (non-flattenable) layout would make reshape return a silent COPY --
    # the kernel writes the copy, the caller's real `out` stays untouched.
    # Requiring stride(0) == flattened row size restricts acceptance to
    # layouts where reshape(-1) is a view; padded layouts decline and fall
    # through to Triton.
    if q.stride(0) != num_query_heads * head_size:
        return False
    if out.stride(0) != num_query_heads * head_size:
        return False
    page_row = num_kv_heads * head_size
    for t in (k, v):
        if t.stride(3) != 1 or t.stride(2) != head_size:
            return False
        if t.stride(1) != page_row or t.stride(0) != _PAGE_SIZE * page_row:
            return False
    return block_table.stride(1) == 1


def _dispatch_mode_ok(window_size, block_table, shuffled_kv_cache, skip_reduce) -> bool:
    """Paged, full-window, non-shuffle, non-reduce. The reduce/shuffle flags
    belong to Triton layouts this kernel does not read; the paged path is the
    only one wired here. (Causal and non-causal are both built.)"""
    return (
        window_size[0] < 0
        and block_table is not None
        and not shuffled_kv_cache
        and not skip_reduce
    )


def _page_geometry_ok(block_size, max_seqlen_k) -> bool:
    """Structural page shape, not tunable: fixed page size, KV within the staged
    block-table window."""
    return (
        block_size == _PAGE_SIZE
        and (max_seqlen_k + _PAGE_SIZE - 1) // _PAGE_SIZE <= _MAX_KV_TILES
    )


def _dtypes_ok(q, k, v, out, cu_seqlens_q, seqused_k, block_table) -> bool:
    """fp8 QKV, bf16/f16 output (both pack to 2 bytes), int32 index tensors."""
    return (
        q.dtype == _FP8_DTYPE
        and k.dtype == _FP8_DTYPE
        and v.dtype == _FP8_DTYPE
        and out.dtype in (torch.bfloat16, torch.float16)
        and cu_seqlens_q.dtype == torch.int32
        and seqused_k.dtype == torch.int32
        and block_table.dtype == torch.int32
    )


def _descales_ok(q_descale, k_descale, v_descale) -> bool:
    """Per-tensor fp8 descales are mandatory: the kernel reads all three to form
    c_logit_scale and the V dequant."""
    return all(
        d is not None and d.dtype == torch.float32 and d.numel() == 1
        for d in (q_descale, k_descale, v_descale)
    )


def _geometry_ok(
    head_size, num_query_heads, num_kv_heads, num_queries_per_kv, cu_seqlens_q, num_seqs
) -> bool:
    """Fixed head dim; GQA group divides block_m for M-packing; cu_seqlens covers
    every sequence."""
    return (
        head_size == _HEAD_DIM
        and num_query_heads % num_kv_heads == 0
        and _BLOCK_M % num_queries_per_kv == 0
        and cu_seqlens_q.numel() == num_seqs + 1
    )


def _no_unsupported_features(
    softcap, alibi_slopes, qq_bias, q_scales, output_scale
) -> bool:
    """Features the kernel has no path for; declining beats silently dropping."""
    return (
        softcap == 0
        and alibi_slopes is None
        and qq_bias is None
        and q_scales is None
        and output_scale is None
    )


def _sinks_ok(sinks, num_query_heads) -> bool:
    """Sinks (per-head [num_query_heads] fp32) are served only single-split; the
    dispatch gate refuses split-K when sinks is set, so an accepted call always
    lands single-split."""
    return sinks is None or (
        sinks.dtype == torch.float32 and sinks.numel() == num_query_heads
    )


def _supported(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_k,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    num_kv_heads,
    block_size,
    num_queries_per_kv,
    num_seqs,
    q_scales,
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
    shuffled_kv_cache,
    skip_reduce,
) -> bool:
    """Whether this exact configuration can be served. Kept separate from the
    marshalling so it can be unit-tested against meta tensors, with no GPU."""
    if not is_flydsl_available():
        return False

    head_size = q.shape[-1]
    num_query_heads = q.shape[1]

    return (
        _dispatch_mode_ok(window_size, block_table, shuffled_kv_cache, skip_reduce)
        and _page_geometry_ok(block_size, max_seqlen_k)
        and _dtypes_ok(q, k, v, out, cu_seqlens_q, seqused_k, block_table)
        and _descales_ok(q_descale, k_descale, v_descale)
        and _geometry_ok(
            head_size,
            num_query_heads,
            num_kv_heads,
            num_queries_per_kv,
            cu_seqlens_q,
            num_seqs,
        )
        and _no_unsupported_features(
            softcap, alibi_slopes, qq_bias, q_scales, output_scale
        )
        and _sinks_ok(sinks, num_query_heads)
        and _strides_ok(
            q, out, k, v, block_table, num_query_heads, num_kv_heads, head_size
        )
    )


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    """fp8 buffers are passed to flydsl as int8 views (the kernel builds i8-typed
    descriptors so DMA and register loads share one byte view)."""
    return t.view(torch.int8) if t.dtype == _FP8_DTYPE else t


def _cu_seqlens_kv(seqused_k: torch.Tensor, num_seqs: int) -> torch.Tensor:
    """Lengths -> cumulative offsets.

    ``unified_attention`` passes per-sequence KV *lengths*; the kernel wants
    cumulative offsets and recovers the length as ``cu[i+1] - cu[i]``. Under
    paging the absolute bases are unobservable -- init_gmem_offsets drops
    kv_tok_base from the KV offset and the paged path builds per-page
    descriptors instead of a whole-tensor view -- so a synthesized cumsum is
    exactly equivalent to the real one.
    """
    cu = torch.zeros(num_seqs + 1, dtype=torch.int32, device=seqused_k.device)
    torch.cumsum(seqused_k, 0, out=cu[1:])
    return cu


def _scaled_q_descale(q_descale: torch.Tensor, softmax_scale: float, head_size: int):
    """Fold an arbitrary softmax scale into the Q descale.

    The kernel takes no runtime softmax scale: init_descale bakes in
    ``rsqrt(head_dim) * log2e`` and multiplies it by ``q_descale * k_descale``
    to form ``c_logit_scale``. Since that's a product, scaling q_descale by
    ``softmax_scale / rsqrt(head_dim)`` yields the requested scale with no
    kernel change.

    Sound only while q_descale reaches c_logit_scale and nothing else (holds
    today: exactly two uses, load and multiply) and c_logit_scale is applied
    only to a logit DIFFERENCE, which is what a softmax scale is. A test
    guards the invariant.

    For the near-universal ``softmax_scale == 1/sqrt(head_size)`` the ratio
    is exactly 1 and the tensor passes through untouched.
    """
    ratio = float(softmax_scale) * math.sqrt(head_size)
    if abs(ratio - 1.0) <= 1e-6:
        return q_descale
    return q_descale * ratio


def flydsl_unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    *,
    num_kv_heads,
    block_size,
    num_queries_per_kv,
    num_seqs,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    shuffled_kv_cache=False,
    skip_reduce=False,
):
    """Run unified attention on the FlyDSL fp8 gfx950 kernel.

    The positional parameters mirror ``unified_attention`` exactly so the hook is
    a mechanical forward. The keyword-only block is quantities the caller has
    already derived; recomputing them here would duplicate its layout unpacking.

    Returns ``out`` (written in place) if this configuration is supported, or
    ``None`` so the caller falls through to Triton.
    """
    if not _supported(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        seqused_k,
        max_seqlen_k,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        num_kv_heads,
        block_size,
        num_queries_per_kv,
        num_seqs,
        q_scales,
        alibi_slopes,
        output_scale,
        qq_bias,
        sinks,
        shuffled_kv_cache,
        skip_reduce,
    ):
        return None

    num_query_heads = q.shape[1]
    out_dtype_str = "f16" if out.dtype == torch.float16 else "bf16"

    # Tier selection. Split-K fires when ALL hold; otherwise single-pass.
    #  - max_seqlen_k > _PAGE_SIZE * 20 (1280): below this the combine pass
    #    isn't amortized (measured crossover).
    #  - num_2d_prgms < target: the machine is underfilled, the only regime
    #    split-K helps -- a high-chunk mixed batch (long prefill) fills the
    #    machine on its own and stays single-pass; only low-chunk mixes and
    #    all-decode route to split-K.
    #  - sinks is None: split-K + sinks is refused by the builder (exp(sink)
    #    would be double-counted across split combines).
    # The combine rebases its O write on cu_seqlens_q, so varlen batches with
    # unequal query lengths combine correctly under the same dense workspace.
    #
    # num_2d_prgms is the single-pass base workgroup count: num_kv_heads *
    # sum_i ceil(query_len_i / BLOCK_Q) (packed per-sequence work, not the
    # dense grid which over-counts padding). BLOCK_Q is block_m / GQA group.
    #
    # The exact sum needs per-sequence query lengths on cu_seqlens_q (device);
    # reading via .item() forces a ~29us host sync into the dispatch path, so
    # it's avoided wherever the branch can be decided from host scalars alone:
    #  - all-decode (max_seqlen_q == 1): sum is exactly num_seqs, no read
    #    needed -- this is the shallow split-K regime where the sync used to
    #    dominate kernel time.
    #  - otherwise, a host-side lower bound max(num_seqs, ceil(total_q /
    #    BLOCK_Q)) already fills the machine for a real prefill/high-chunk
    #    mix, so single-pass is decided without a read.
    #  - only a genuine low-chunk mix, where that lower bound underfills,
    #    needs the exact per-sequence sum and takes the device read.
    # Every branch matches the exact-sum decision; the read is skipped only
    # where it cannot change the outcome.
    num_kv_splits = 1
    if max_seqlen_k > _PAGE_SIZE * 20 and sinks is None:
        block_q = _BLOCK_M // num_queries_per_kv
        if max_seqlen_q == 1:
            num_q_blocks = num_seqs
        else:
            total_q = q.shape[0]
            lower_bound = max(num_seqs, (total_q + block_q - 1) // block_q)
            if num_kv_heads * lower_bound >= _TARGET_NUM_PRGMS:
                # Provably full at single-pass; exact value only gates <target.
                num_q_blocks = lower_bound
            else:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                num_q_blocks = int(
                    ((seqlens_q + (block_q - 1)) // block_q).sum().item()
                )
        num_2d_prgms = num_kv_heads * num_q_blocks
        if num_2d_prgms < _TARGET_NUM_PRGMS:
            num_kv_splits = _split_count(num_2d_prgms)

    kernel = _get_kernel(
        num_query_heads,
        num_kv_heads,
        bool(causal),
        out_dtype_str,
        sinks is not None,
        num_kv_splits,
    )

    workspace = None
    if num_kv_splits > 1:
        # fp32 partial workspace: O_partial + Mrow + Lrow, sized for the dense
        # [batch, max_seqlen_q(==1), heads, ...] layout the store/combine use.
        ws_elems = dualwave_splitk_workspace_elems(
            num_seqs, num_query_heads, int(max_seqlen_q), num_kv_splits, _HEAD_DIM
        )
        workspace = torch.empty(ws_elems, device=q.device, dtype=torch.float32)

    with torch.cuda.device(q.device.index):
        kernel(
            # .reshape(-1) relies on _strides_ok already restricting Q/O to
            # flattenable layouts (see _strides_ok), so this is always a
            # view here, never the silent copy that would occur otherwise.
            _as_i8(q).reshape(-1),
            _as_i8(k).reshape(-1),
            _as_i8(v).reshape(-1),
            out.reshape(-1),
            num_seqs,
            # grid.y is ceil(seq_len / BLOCK_Q), so this must be the MAX q len;
            # per-sequence trimming comes from cu_seqlens_q via the active guard.
            int(max_seqlen_q),
            # The KV stride is the within-page row stride, not the page stride.
            k.stride(1),
            q.stride(0),
            workspace=workspace,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=_cu_seqlens_kv(seqused_k, num_seqs),
            q_descale=_scaled_q_descale(q_descale, softmax_scale, q.shape[-1]),
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table.reshape(-1),
            block_table_stride=int(block_table.stride(0)),
            sink=None if sinks is None else sinks.reshape(-1),
            stream=torch.cuda.current_stream(q.device),
        )
    return out

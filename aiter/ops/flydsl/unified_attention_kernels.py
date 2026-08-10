# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fp8 unified-attention backend for gfx950.

Adapts the vendored ``flash_attn_dualwave_swp`` fp8 kernel
(``kernels/flash_attn_fp8_gfx950.py``) to the ``unified_attention`` calling
convention that vLLM and SGLang import directly, so a supported gfx950 fp8
paged call routes here instead of Triton.

Dispatch follows the in-tree convention for a new backend competing for an
existing op: a pure code predicate that returns ``None`` when it cannot serve
the configuration, letting the caller fall through to Triton unchanged (see
``flydsl_flash_attn_varlen_func`` in ``fmha_kernels.py`` and the flydsl branch
in ``ops/gemm_op_a8w8.py``). There is deliberately no environment variable:
aiter already carries 19 undocumented backend-gate vars, and this path is arch-
and shape-scoped, so a code gate states the choice more honestly. For an A/B,
patch ``_FLYDSL_UNIFIED_ATTN_ARCH`` in the Triton module to ``False``.

Two tiers. Prefill and mixed batches take the single-pass packed kernel. The
decode-only case (``max_seqlen_q == 1``) is machine-underfilled -- a handful of
workgroups each serially scan the full KV depth while most CUs sit idle -- so it
takes packed + split-K, which partitions each sequence's KV across split
workgroups and combines the partials. The tier is chosen at call time from the
machine-fill deficit; see ``_split_count`` and the gate in
``flydsl_unified_attention``.

The split-K tier is confined to ``max_seqlen_q == 1`` on purpose. The split-K
workspace and its combine kernel address O DENSELY by ``batch_idx *
max_seqlen_q`` with no ``cu_seqlens_q``, so they are correct only when every
sequence's query length equals ``max_seqlen_q`` -- exactly the all-decode case.
A mixed/prefill varlen batch (unequal query lengths) would be miscombined, so it
stays on single-pass regardless of fill. Serving the low-chunk mixed case would
need a varlen-q-aware workspace, which is out of scope here.
"""

from __future__ import annotations

import math
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

# gfx950/MI355X CU count. This is the split-K fill target: a launch with
# num_2d_prgms base workgroups is "full" at num_2d_prgms >= _TARGET_NUM_PRGMS.
# An arch constant, not a runtime GPU query -- the whole path is gfx950-gated
# (the Triton hook only calls in on gfx950), so the CU count is known.
_TARGET_NUM_PRGMS = 256

# The prototype only measured split counts in {2, 4, 8}; do not emit 16+ on
# unmeasured shapes.
_MAX_SEGMENTS = 8

# block_m, and with it the largest GQA group that can pack into the M dimension.
# _make_dualwave_swp_fp8_traits asserts block_m % gqa_group_size == 0.
_BLOCK_M = 256

# The block table is staged through a fixed LDS window of PAGED_BT_LDS_SIZE=2048
# entries. Past that the stager writes only `local_tile < segment_tiles` slots
# and silently drops the remaining page ids -- wrong output, not a fault -- so
# the KV length must be capped here. 2048 pages * 64 tokens = 131072 tokens,
# which is exactly the model's maximum context rather than comfortably above it.
_MAX_KV_TILES = 2048

_FP8_DTYPE = torch.float8_e4m3fn


@lru_cache(maxsize=64)
def _get_kernel(num_heads: int, num_kv_heads: int, causal: bool, out_dtype_str: str,
                use_sinks: bool, num_kv_splits: int = 1):
    """Build (and cache) the paged+varlen fp8 launcher.

    Keyed on the head counts, the mask mode, the output dtype and the split
    count. Every other builder argument is either pinned by the support gate
    (head_dim, dtype, varlen, paged) or left at its default.

    ``num_kv_splits`` selects the tier. At 1 the builder auto-enables
    M-dimension packing for GQA > 1:1 (``gqa_pack_m=None``) and builds the
    single-pass kernel. At > 1 packing must be forced on explicitly:
    ``gqa_pack_m`` defaults to unpacked for split-K (the committed default kept
    the packed+split-K prototype off), so the decode tier passes it True to get
    the packed+split-K binary. ``GQA_PACK_M`` and ``NUM_KV_SPLITS`` both key the
    JIT cache, so the two binaries cannot alias.

    ``causal``, ``out_dtype_str`` and ``use_sinks`` are runtime-selectable, so
    each must key the cache: they change compiled code (the mask path, the store
    packer, and the sink init/epilogue term).

    Do not key on batch/seqlen/device: ``_run_compiled`` memoizes the compiled
    function on the returned closure, so a finer key would defeat that cache and
    recompile per shape.
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
    count, capped at 8. No MIN floor -- b=9 needs 4, below Triton's floor of 8.
    Below 2 there is no split to take, so the caller falls back to single-pass.

    Verified against the committed A/B winners (39f7e68de): num_2d = 4*b for a
    b-sequence GQA-16 decode gives b=7 -> 8, b=8 -> 8, b=9 -> 4, matching the
    measured best split count at each.
    """
    n = 1
    while n * 2 <= _MAX_SEGMENTS and num_2d_prgms * (n * 2) <= _TARGET_NUM_PRGMS:
        n *= 2
    return n if n >= 2 else 1


def _strides_ok(q, out, k, v, block_table, num_kv_heads, head_size) -> bool:
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
    page_row = num_kv_heads * head_size
    for t in (k, v):
        if t.stride(3) != 1 or t.stride(2) != head_size:
            return False
        if t.stride(1) != page_row or t.stride(0) != _PAGE_SIZE * page_row:
            return False
    return block_table.stride(1) == 1


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
        # Dispatch mode. Both causal and non-causal are built (causal keys the
        # kernel cache); the paged path is the only one wired here, and the two
        # "reduce"/shuffle flags belong to Triton layouts we do not read.
        window_size[0] < 0
        and block_table is not None
        and not shuffled_kv_cache
        and not skip_reduce
        # Page geometry. Both are structural, not tunable.
        and block_size == _PAGE_SIZE
        and (max_seqlen_k + _PAGE_SIZE - 1) // _PAGE_SIZE <= _MAX_KV_TILES
        # Dtypes. Output packs via cvt_pk_bf16_f32 (bf16) or cvt_pkrtz (f16);
        # both store 2 bytes, so the store layout is identical.
        and q.dtype == _FP8_DTYPE
        and k.dtype == _FP8_DTYPE
        and v.dtype == _FP8_DTYPE
        and out.dtype in (torch.bfloat16, torch.float16)
        and cu_seqlens_q.dtype == torch.int32
        and seqused_k.dtype == torch.int32
        and block_table.dtype == torch.int32
        # Per-tensor fp8 descales are mandatory on this path -- the kernel reads
        # all three unconditionally to form c_logit_scale and the V dequant.
        and q_descale is not None
        and k_descale is not None
        and v_descale is not None
        and q_descale.dtype == torch.float32
        and k_descale.dtype == torch.float32
        and v_descale.dtype == torch.float32
        and q_descale.numel() == 1
        and k_descale.numel() == 1
        and v_descale.numel() == 1
        # Geometry. The GQA group must divide block_m for M-dimension packing.
        and head_size == _HEAD_DIM
        and num_query_heads % num_kv_heads == 0
        and _BLOCK_M % num_queries_per_kv == 0
        and cu_seqlens_q.numel() == num_seqs + 1
        # Features with no kernel support. Declining beats silently dropping.
        and softcap == 0
        and alibi_slopes is None
        and qq_bias is None
        and q_scales is None
        and output_scale is None
        # Sinks are served (per-head [num_query_heads] fp32) but only on the
        # single-split path. The adapter never builds split-K (see module
        # docstring), so any accepted sinks call is single-split; the guard is
        # defensive in case that ever changes.
        and (sinks is None
             or (sinks.dtype == torch.float32 and sinks.numel() == num_query_heads))
        and _strides_ok(q, out, k, v, block_table, num_kv_heads, head_size)
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
    to form ``c_logit_scale``. Because that is a product, scaling q_descale by
    ``softmax_scale / rsqrt(head_dim)`` yields exactly the requested scale with
    no kernel change.

    This is sound only while q_descale reaches c_logit_scale and nothing else.
    That holds today (the loaded scalar has exactly two uses: the load and the
    multiply), and c_logit_scale is only ever applied to a DIFFERENCE of logits,
    which is what a softmax scale is. The V descale is loaded separately and
    cannot be affected. A test guards the invariant.

    For the near-universal ``softmax_scale == 1/sqrt(head_size)`` the ratio is
    exactly 1 and the tensor passes through untouched, so the common path costs
    nothing.
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
        q, k, v, out, cu_seqlens_q, seqused_k, max_seqlen_k, causal, window_size,
        block_table, softcap, q_descale, k_descale, v_descale, num_kv_heads,
        block_size, num_queries_per_kv, num_seqs, q_scales, alibi_slopes,
        output_scale, qq_bias, sinks, shuffled_kv_cache, skip_reduce,
    ):
        return None

    num_query_heads = q.shape[1]
    out_dtype_str = "f16" if out.dtype == torch.float16 else "bf16"

    # Tier selection. Split-K only fires when ALL hold; otherwise single-pass.
    #  - max_seqlen_q == 1 (all-decode): the ONLY case the dense-q split-K
    #    workspace/combine addresses correctly (they have no cu_seqlens_q, so a
    #    mixed/prefill varlen batch would be miscombined). This is the necessary
    #    condition that keeps split-K inside the Phase-1-proven-correct envelope.
    #  - max_seqlen_k > 512: below this the combine pass does not amortize (it is
    #    also Triton's own force_2d_decode cutoff).
    #  - num_2d_prgms < target: the machine is underfilled, which is the only
    #    regime split-K helps. A full launch would only add combine overhead.
    #  - sinks is None: split-K + sinks is refused by the builder (exp(sink)
    #    would be counted once per split in the combine denominator).
    #
    # num_2d_prgms is the single-pass base workgroup count: num_kv_heads *
    # sum_i ceil(query_len_i / BLOCK_Q). For all-decode every query_len is 1 and
    # BLOCK_Q >= 1, so each sequence contributes exactly one q-block and this is
    # num_kv_heads * num_seqs.
    num_kv_splits = 1
    if max_seqlen_q == 1 and max_seqlen_k > _PAGE_SIZE * 8 and sinks is None:
        num_2d_prgms = num_kv_heads * num_seqs
        if num_2d_prgms < _TARGET_NUM_PRGMS:
            num_kv_splits = _split_count(num_2d_prgms)

    kernel = _get_kernel(num_query_heads, num_kv_heads, bool(causal), out_dtype_str,
                         sinks is not None, num_kv_splits)

    workspace = None
    if num_kv_splits > 1:
        # fp32 partial workspace: O_partial + Mrow + Lrow, sized for the dense
        # [batch, max_seqlen_q(==1), heads, ...] layout the store/combine use.
        ws_elems = dualwave_splitk_workspace_elems(
            num_seqs, num_query_heads, int(max_seqlen_q), num_kv_splits, _HEAD_DIM)
        workspace = torch.empty(ws_elems, device=q.device, dtype=torch.float32)

    with torch.cuda.device(q.device.index):
        kernel(
            _as_i8(q).view(-1),
            _as_i8(k).view(-1),
            _as_i8(v).view(-1),
            out.view(-1),
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
            block_table=block_table.view(-1),
            block_table_stride=int(block_table.stride(0)),
            sink=None if sinks is None else sinks.view(-1),
            stream=torch.cuda.current_stream(q.device),
        )
    return out

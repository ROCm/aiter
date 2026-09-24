# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 top-k-index decode block scoring: bf16 index query x fp8 cache.

This is the block-score half of the indexer; the top-k selector is elsewhere.
The incumbent pa_sparse_block_score_decode cannot serve this call -- it needs an
fp8 query with matching dtypes on both sides.
"""

import contextlib
import math

import torch

from aiter.jit.core import compile_ops


@compile_ops(
    "module_topk_index_score",
    fc_name="topk_index_score_decode",
    ffi_type="ctypes",
)
def _topk_index_score_raw(
    q_idx: int,
    key_cache_idx: int,
    score: int,
    block_table: int,
    seq_lens: int,
    q_numel: int,
    key_cache_numel: int,
    score_numel: int,
    block_table_numel: int,
    batch: int,
    num_chunks: int,
    chunk_blocks: int,
    stride_q_n: int,
    stride_q_h: int,
    stride_ik_blk: int,
    stride_s_h: int,
    stride_s_b: int,
    stride_bt_b: int,
    sm_scale: float,
    block_size: int,
    head_dim: int,
    num_idx_heads: int,
    query_len: int,
    aux_k: int,
    cert_bypass: int,
) -> None: ...


def _ptr(t) -> int:
    """Device address of a tensor; 0 is the null the kernels test for."""
    return 0 if t is None else t.data_ptr()


def _addressed_span(t) -> int:
    """Elements a view can ADDRESS from its base pointer: 1 + sum((n-1)*stride).
    """
    if t is None or t.numel() == 0:
        return 0
    return 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))


# Build constants. Not defaults: the LDS staging layout and invariant I5's 1.00
# L1 accesses per 128 B line are both derived for 128-byte rows.
OPUS_HEAD_DIM = 128
OPUS_BLOCK_SIZE = 128

# Columns of the MFMA, each holding one (token, head) pair.
OPUS_MFMA_COLS = 16

# Cells the C++ table BUILDS, and the subset dispatch may SELECT. Both sets are
# mirrored in topk_index_score.hpp and must be edited together.
OPUS_BUILT_CELLS = ((1, 1), (1, 2), (1, 4), (1, 8), (4, 1), (4, 2), (4, 4))
# All seven built cells are certified. Q = 2 is the default MTP width and Q = 8
# the model's full MTP depth; H = 2 is a deliberate refusal, and (4, 8) is
# H*Q = 32, past the 16 MFMA columns.
OPUS_CERTIFIED_CELLS = ((1, 1), (1, 2), (1, 4), (1, 8), (4, 1), (4, 2), (4, 4))

# The one tuned template axis. Both legs of the deployed cell carry phase-1
# correctness evidence; 3 is the accepted performance point on the traced shape.
OPUS_AUX_K_BUILT = (0, 3)

# Blocks per workgroup on the block axis. Swept 1..128 over the whole 80-point
# surface on three exclusive nodes, then validated on a held-out set sharing no
# shape with it; see _grid_lever for the derivation and the invalidation
# conditions. This is a MEASURED constant, not a default.
OPUS_CHUNK_BLOCKS = 4

# The banded arm: a working set inside the LLC band takes a larger chunk and no
# cache hint. DISABLED -- it failed its own pre-registered unanimity clause at
# two band edges. Kept with its derivation so it is not re-fitted from scratch.
OPUS_LLC_BAND_ENABLED = False

OPUS_LLC_BAND_LO_MIB = 128
OPUS_LLC_BAND_HI_MIB = 256
OPUS_LLC_BAND_CHUNK_BLOCKS = 24
OPUS_LLC_BAND_AUX_K = 0
OPUS_INDEX_K_PAGE_BYTES = OPUS_BLOCK_SIZE * OPUS_HEAD_DIM  # fp8: 1 byte/element

# AUX_K for a bucket with no tuning-table entry. See _aux_k_for.
OPUS_AUX_K_DEFAULT = 3

# Waves per SIMD the accepted build achieves. Informational: the grid lever no
# longer reads device capacity.
OPUS_WAVES_PER_CU = 8

# --------------------------------------------------------------------------- #
# Launch policy
# --------------------------------------------------------------------------- #
OPUS_CERT_BYPASS_TOKEN = 0x5339C0DE

_CERT_ACK = (
    "I am producing per-cell bitwise certification evidence and I accept that "
    "this bypasses the N16 dispatch-policy gate only"
)

# None = the hook is off. A list = the hook is on, and receipts go there.
_cert_receipts = None


@contextlib.contextmanager
def certification_bypass_n16(acknowledgement):
    """Scoped, acknowledged bypass of N16 -- for certification harnesses ONLY.
    """
    global _cert_receipts
    if acknowledgement != _CERT_ACK:
        raise ValueError(
            "certification_bypass_n16: refused. The acknowledgement must be the "
            "exact string at aiter/ops/topk_index_score.py:_CERT_ACK. It is "
            "verbose so that enabling this is deliberate and greppable."
        )
    if _cert_receipts is not None:
        raise RuntimeError(
            "certification_bypass_n16: already active; nesting is not permitted "
            "(a nested block would restore the wrong state on exit)"
        )
    _cert_receipts = []
    try:
        yield _cert_receipts
    finally:
        _cert_receipts = None

# Tuning table: bucket -> AUX_K, deliberately EMPTY. The surface-wide result is
# carried by OPUS_AUX_K_DEFAULT; a bucket here would be a second place to look.
_OPUS_AUX_TUNING_TABLE: dict = {}

_RESIDENT_WG_CACHE: dict = {}


def _resident_workgroups(device) -> int:
    """Workgroups resident across the device: CU count x waves per CU.

    The CU count is a device property, read once per device -- a host query at
    capture time, never a device-memory read, so cudagraph invariant G2 holds.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _RESIDENT_WG_CACHE.get(index)
    if cached is None:
        cached = (
            torch.cuda.get_device_properties(index).multi_processor_count
            * OPUS_WAVES_PER_CU
        )
        _RESIDENT_WG_CACHE[index] = cached
    return cached


def _grid_lever(max_blk: int, batch: int, capacity: int) -> tuple[int, int]:
    """Chunk count and chunk size for the block axis.
    """
    if _in_llc_band(max_blk, batch):
        chunk_blocks = min(OPUS_LLC_BAND_CHUNK_BLOCKS, max(1, max_blk))
    else:
        chunk_blocks = min(OPUS_CHUNK_BLOCKS, max(1, max_blk))
    num_chunks = math.ceil(max_blk / chunk_blocks)
    return num_chunks, chunk_blocks


def _in_llc_band(max_blk: int, batch: int) -> bool:
    """Is the index-K working set inside the last-level-cache band?
    """
    if not OPUS_LLC_BAND_ENABLED:
        return False
    mib = batch * max_blk * OPUS_INDEX_K_PAGE_BYTES / (1024.0 * 1024.0)
    return OPUS_LLC_BAND_LO_MIB <= mib <= OPUS_LLC_BAND_HI_MIB


def _aux_k_for(num_idx_heads: int, query_len: int, batch: int, max_blk: int) -> int:
    """AUX_K for a bucket: the tuning table if it has an entry, else
    """
    pinned = _OPUS_AUX_TUNING_TABLE.get(
        (num_idx_heads, query_len, batch, max_blk))
    if pinned is not None:
        return pinned
    # The banded arm changes BOTH knobs together. They were fitted together and
    # neither is valid alone: at chunk_blocks 24 the aux3 cache policy is a
    # LOSS, which is exactly why the band takes the control leg.
    if _in_llc_band(max_blk, batch):
        return OPUS_LLC_BAND_AUX_K
    return OPUS_AUX_K_DEFAULT


def _fp8_e4m3fnuz():
    # Absent on some torch builds; None simply means the fnuz arm is
    # unreachable, and the pin below still rejects it as "not e4m3fn".
    return getattr(torch, "float8_e4m3fnuz", None)


def _check_devices(*tensors) -> None:
    """D02 (review): every tensor is forwarded as a raw data_ptr, so nothing
    """
    named = [(n, t) for n, t in tensors if t is not None]
    for n, t in named:
        if not t.is_cuda:
            raise ValueError(
                "topk_index_score_decode: {} is on {}; every tensor must be a "
                "CUDA tensor because they are forwarded as raw pointers".format(n, t.device)
            )
    devs = {t.device for _, t in named}
    if len(devs) != 1:
        raise ValueError(
            "topk_index_score_decode: tensors span more than one device (%s); "
            "the launch takes one stream and cannot straddle them"
            % sorted(str(d) for d in devs)
        )


def _check_dtypes(q_idx, key_cache_idx, score) -> None:
    """Dtype routing. Every arm either accepts or raises -- the space is covered.
    """
    fnuz = _fp8_e4m3fnuz()
    if q_idx.dtype != torch.bfloat16:
        if q_idx.dtype == torch.float8_e4m3fn or (
            fnuz is not None and q_idx.dtype == fnuz
        ):
            raise ValueError(
                "topk_index_score_decode: q_idx is fp8; this operator takes an "
                "UNQUANTIZED bf16 q_idx. The fp8-Q operator is "
                "pa_sparse_block_score_decode -- use it instead of this path."
            )
        raise ValueError(
            f"topk_index_score_decode: q_idx must be bf16, got {q_idx.dtype}"
        )

    if key_cache_idx.dtype == torch.bfloat16:
        raise ValueError(
            "topk_index_score_decode: bf16 key_cache_idx is not supported "
            "(deliberate exclusion, integration.md ledger D0). No bf16-K "
            "implementation of this operator exists in aiter; this rejection is "
            "the documented fallback, not a missing feature."
        )
    if key_cache_idx.dtype != torch.float8_e4m3fn:
        if fnuz is not None and key_cache_idx.dtype == fnuz:
            raise ValueError(
                "topk_index_score_decode: key_cache_idx is float8_e4m3fnuz; this "
                "path is built for float8_e4m3fn only. The two differ in exponent "
                "bias and NaN handling, so dispatching fnuz here would return "
                "wrong numbers rather than fail."
            )
        raise ValueError(
            "topk_index_score_decode: key_cache_idx must be float8_e4m3fn, got "
            f"{key_cache_idx.dtype}"
        )

    if score.dtype != torch.float32:
        raise ValueError(
            f"topk_index_score_decode: score must be fp32, got {score.dtype}"
        )

def topk_index_score_decode_supported(
    q_idx,
    key_cache_idx,
    score,
    query_len: int = 1,
    max_seq_len: int = 0,
):
    """Can this operator serve these inputs? Returns ``(bool, reason)``.
    """
    fnuz = _fp8_e4m3fnuz()
    if getattr(q_idx, "dtype", None) != torch.bfloat16:
        return False, "q_idx must be bf16 (this operator never quantises Q)"
    if key_cache_idx.dtype == torch.bfloat16:
        return False, ("bf16 key_cache_idx is not supported (ledger D0); use "
                       "the Triton index-score path, which serves both dtypes")
    if key_cache_idx.dtype != torch.float8_e4m3fn:
        if fnuz is not None and key_cache_idx.dtype == fnuz:
            return False, "key_cache_idx is float8_e4m3fnuz; built for e4m3fn only"
        return False, "key_cache_idx must be float8_e4m3fn, got %s" % (
            key_cache_idx.dtype,)
    if score.dtype != torch.float32:
        return False, "score must be fp32"
    if q_idx.dim() != 3 or key_cache_idx.dim() != 3 or score.dim() != 3:
        return False, "q_idx, key_cache_idx and score are 3-D"
    if not (q_idx.is_contiguous() and key_cache_idx.is_contiguous()):
        return False, "q_idx and key_cache_idx must be contiguous"
    if score.stride(2) != 1:
        return False, "score must be contiguous along the block axis"
    # D03/D01 mirrored: the core refuses these, so the predicate must too or it
    # routes a caller into a ValueError it was added to prevent.
    if score.stride(0) == 0 or score.stride(1) == 0:
        return False, "score has a zero stride on the head or row axis; outputs alias"
    if _addressed_span(score) * score.element_size() >= 2**32:
        return False, "score exceeds the 32-bit buffer-descriptor extent"
    # D02 mirrored: every tensor is forwarded as a raw pointer.
    for _n, _t in (("q_idx", q_idx), ("key_cache_idx", key_cache_idx),
                   ("score", score)):
        if not _t.is_cuda:
            return False, "%s is on %s; every tensor must be a CUDA tensor" % (
                _n, _t.device)
    if len({q_idx.device, key_cache_idx.device, score.device}) != 1:
        return False, "tensors span more than one device"
    total_q, num_idx_heads, head_dim = q_idx.shape
    block_size = key_cache_idx.size(1)
    if head_dim != OPUS_HEAD_DIM or block_size != OPUS_BLOCK_SIZE:
        return False, "built for head_dim %d and block_size %d, got %d and %d" % (
            OPUS_HEAD_DIM, OPUS_BLOCK_SIZE, head_dim, block_size)
    if key_cache_idx.size(2) != head_dim:
        return False, "key_cache_idx head dim must match q_idx"
    if query_len < 1 or total_q % query_len != 0:
        return False, "q_idx rows must be a positive multiple of query_len"
    if num_idx_heads * query_len > OPUS_MFMA_COLS:
        return False, "num_idx_heads * query_len exceeds the %d MFMA columns" % (
            OPUS_MFMA_COLS,)
    cell = (num_idx_heads, query_len)
    if cell not in OPUS_BUILT_CELLS:
        return False, "no build for (num_idx_heads, query_len) = %s" % (cell,)
    if cell not in OPUS_CERTIFIED_CELLS:
        return False, "cell %s is built but not certified (N16)" % (cell,)
    if max_seq_len < 1:
        return False, "max_seq_len is required so the grid is fixed at capture"
    # D07 / rule A3 (review, upheld at YELLOW): the core rejects a mismatched
    # score outer shape at its own shape check, and this predicate did not, so
    # it answered "supported" for a case the core then raised on -- the exact
    # failure it exists to prevent.
    if score.size(0) != num_idx_heads or score.size(1) != total_q:
        return False, "score must be [num_idx_heads, total_q, S], got %s for " \
                      "num_idx_heads=%d total_q=%d" % (
                          tuple(score.shape), num_idx_heads, total_q)
    if key_cache_idx.size(0) < 1:
        return False, "key_cache_idx holds no pages"
    max_blk = math.ceil(max_seq_len / block_size)
    if score.size(2) < max_blk:
        return False, "score width is below cdiv(max_seq_len, %d)" % block_size
    return True, "supported"

def topk_index_score_decode(
    q_idx,
    key_cache_idx,
    score,
    block_table,
    seq_lens,
    sm_scale: float,
    query_len: int = 1,
    max_seq_len: int = 0,
    aux_k=None,
):
    """Score every block of the fp8 index key cache against a bf16 index query.

    Args:
        q_idx: ``[num_reqs * query_len, num_idx_heads, head_dim]`` **bf16**,
            contiguous. Never quantized -- an fp8 MFMA would round it.
        key_cache_idx: ``[num_pages, 128, 128]`` **float8_e4m3fn**, contiguous.
        score: ``[num_idx_heads, num_reqs * query_len, S]`` fp32, written in
            place. Blocks past ``cdiv(seq_len, 128)`` are left untouched, so
            pre-fill with ``-inf`` unless the consumer bounds each row.
        block_table: ``[num_reqs, >= max_block]`` int32.
        seq_lens: ``[num_reqs]`` int32.
        sm_scale: applied in-kernel as ``sm_scale * log2e`` AFTER the dot.
        query_len: query tokens per request; ``num_idx_heads * query_len`` must
            fit the MFMA's 16 columns.
        max_seq_len: upper bound on the context, required. The grid comes from
            this, never from ``seq_lens``, so it is fixed at capture.
        aux_k: index-K load cache policy. ``None`` takes the tuning table,
            whose default is ``OPUS_AUX_K_DEFAULT = 3`` -- see ``_aux_k_for``
            for the surface-wide evidence behind that default and
            the sweep that produced it for its
            validity range. Pass 0 or 3 explicitly to pin a leg.

    Returns:
        ``score``, written in place.
    """
    _check_dtypes(q_idx, key_cache_idx, score)
    _check_devices(
        ("q_idx", q_idx), ("key_cache_idx", key_cache_idx), ("score", score),
        ("block_table", block_table), ("seq_lens", seq_lens),
    )

    if q_idx.dim() != 3 or key_cache_idx.dim() != 3 or score.dim() != 3:
        raise ValueError(
            "topk_index_score_decode: q_idx, key_cache_idx and score are 3-D"
        )
    if not (q_idx.is_contiguous() and key_cache_idx.is_contiguous()):
        raise ValueError(
            "topk_index_score_decode: q_idx and key_cache_idx must be contiguous"
        )
    if score.stride(2) != 1:
        raise ValueError(
            "topk_index_score_decode: score must be contiguous along the block axis"
        )
    # D03 / rule D8 (review, upheld): stride(2) == 1 was the ONLY output-layout
    # gate, so an expanded view -- torch.empty(1,1,1).expand(4,1,1) -- passed with
    # stride(0) == 0 and every head stored to one address concurrently. A zero
    # stride on either non-contiguous axis means two logical outputs share one
    # physical element, which no amount of care inside the kernel can make safe.
    if score.stride(0) == 0 or score.stride(1) == 0:
        raise ValueError(
            "topk_index_score_decode: score has a zero stride on the head or row "
            "axis (strides %s), so different outputs alias one element; pass a "
            "materialised tensor rather than an expanded view" % (score.stride(),)
        )
    # D01 (review): the addressed span is forwarded as the buffer-descriptor
    # extent and the kernel narrows span*4 to 32 bits, so a span at or past
    # 2**30 float32 elements wraps the extent to zero and silently drops every
    # store. Refused here, where the number is still a Python int and the
    # arithmetic cannot wrap.
    _span_bytes = _addressed_span(score) * score.element_size()
    if _span_bytes >= 2**32:
        raise ValueError(
            "topk_index_score_decode: score spans %d bytes, which does not fit "
            "the 32-bit buffer-descriptor extent the kernel uses; the limit is "
            "%d bytes" % (_span_bytes, 2**32 - 1)
        )
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError(
            "topk_index_score_decode: block_table and seq_lens must be int32"
        )
    if seq_lens.dim() != 1 or seq_lens.stride(0) != 1:
        # The kernel builds the seq_lens descriptor as batch*sizeof(int) and
        # indexes it by request, so a strided or multi-dim seq_lens would be
        # read at the wrong offsets. Same class as the span defect above:
        # refused rather than mis-addressed.
        raise ValueError(
            "topk_index_score_decode: seq_lens must be 1-D and contiguous"
        )
    if block_table.dim() != 2 or block_table.stride(1) != 1:
        raise ValueError(
            "topk_index_score_decode: block_table must be 2-D with contiguous rows"
        )

    total_q, num_idx_heads, head_dim = q_idx.shape
    block_size = key_cache_idx.size(1)
    if key_cache_idx.size(2) != head_dim:
        raise ValueError(
            "topk_index_score_decode: key_cache_idx head dim must match q_idx"
        )
    if head_dim != OPUS_HEAD_DIM or block_size != OPUS_BLOCK_SIZE:
        raise ValueError(
            f"topk_index_score_decode: built for head_dim {OPUS_HEAD_DIM} and "
            f"block_size {OPUS_BLOCK_SIZE}, got {head_dim} and {block_size}"
        )
    if score.size(0) != num_idx_heads or score.size(1) != total_q:
        raise ValueError(
            "topk_index_score_decode: score must be [num_idx_heads, total_q, S]"
        )

    if total_q % query_len != 0:
        raise ValueError(
            f"topk_index_score_decode: q_idx rows {total_q} not a multiple of "
            f"query_len {query_len}"
        )
    num_reqs = total_q // query_len

    # query_len >= 1, num_idx_heads * query_len <= 16 and "is this cell built"
    # are NOT re-checked here. The C entry checks all three before it can reach a
    # launch, and it is reachable directly through ctypes, so its copy is the one
    # that has to exist. topk_index_score_decode_supported() keeps them so a
    # caller can still ASK without raising.
    cell = (num_idx_heads, query_len)
    cert_token = 0
    if cell not in OPUS_CERTIFIED_CELLS:
        # N16. The cell IS built -- that is what keeps the gap visible -- but it
        # has no per-cell bitwise evidence at the accepted config, so it may not
        # be dispatched on the 'same code path, compile-time constants only'
        # argument alone. The C++ entry refuses it again at layer C4.
        if _cert_receipts is None:
            raise ValueError(
                f"topk_index_score_decode: variant built but not certified -- "
                f"(num_idx_heads, query_len) = {cell} has no per-cell bitwise "
                f"evidence at the accepted config. Certified cells: "
                f"{OPUS_CERTIFIED_CELLS}."
            )
        # Certification mode: produce the evidence. The receipt is written
        # below, once aux_k has been resolved -- at this point it may still be
        # None (the tuning-table default is applied later), and a receipt naming
        # the wrong leg would be worse than none.
        cert_token = OPUS_CERT_BYPASS_TOKEN

    if seq_lens.size(0) != num_reqs or block_table.size(0) != num_reqs:
        raise ValueError(
            "topk_index_score_decode: seq_lens and block_table need one entry "
            "per request"
        )

    # N13: an empty batch is a no-op, and no zero-extent grid reaches capture.
    if num_reqs == 0:
        return score

    # KEPT even though the C entry would also reject it, via num_chunks == 0.
    # Omitting max_seq_len is the likeliest caller mistake here, and the C
    # message ("batch/num_chunks/chunk_blocks must be positive") would send a
    # reader to look at the wrong thing.
    if max_seq_len < 1:
        raise ValueError(
            "topk_index_score_decode: pass max_seq_len so the launch dimensions "
            "are fixed at capture"
        )
    max_blk = math.ceil(max_seq_len / block_size)
    if score.size(2) < max_blk:
        raise ValueError(
            f"topk_index_score_decode: score width {score.size(2)} is below "
            f"cdiv(max_seq_len, {block_size}) = {max_blk}"
        )
    if block_table.size(1) < max_blk:
        raise ValueError(
            f"topk_index_score_decode: block_table width {block_table.size(1)} is "
            f"below cdiv(max_seq_len, {block_size}) = {max_blk}"
        )

    if not math.isfinite(sm_scale):
        # Mirrors the C-side guard. A non-finite scale multiplies every dot, so
        # it NaNs the whole score buffer and the top-k consumer then ranks that
        # -- a silently-wrong-answer path, not a crash path.
        raise ValueError(
            f"topk_index_score_decode: sm_scale must be finite, got {sm_scale}"
        )

    if aux_k is None:
        aux_k = _aux_k_for(num_idx_heads, query_len, num_reqs, max_blk)
    # aux_k is not re-checked here: the C entry rejects anything outside the
    # built set, and it is the layer a ctypes caller cannot bypass.

    num_chunks, chunk_blocks = _grid_lever(
        max_blk, num_reqs, _resident_workgroups(q_idx.device)
    )

    if cert_token:
        # Receipt, with the leg resolved. One per call that actually used the
        # hook, so the certification record can name exactly which evidence was
        # produced under bypass.
        _cert_receipts.append(
            {"cell": list(cell), "aux_k": aux_k, "bypassed_gate": "N16",
             "batch": num_reqs, "max_blk": max_blk}
        )

    _topk_index_score_raw(
        _ptr(q_idx),
        _ptr(key_cache_idx),
        _ptr(score),
        _ptr(block_table),
        _ptr(seq_lens),
        # Descriptor extents: the ADDRESSED SPAN of each view, never numel().
        # See _addressed_span -- a short descriptor drops accesses silently.
        _addressed_span(q_idx),
        _addressed_span(key_cache_idx),
        _addressed_span(score),
        _addressed_span(block_table),
        num_reqs,
        num_chunks,
        chunk_blocks,
        q_idx.stride(0),
        q_idx.stride(1),
        key_cache_idx.stride(0),
        score.stride(0),
        score.stride(1),
        block_table.stride(0),
        float(sm_scale),
        block_size,
        head_dim,
        num_idx_heads,
        query_len,
        aux_k,
        cert_token,
    )
    return score

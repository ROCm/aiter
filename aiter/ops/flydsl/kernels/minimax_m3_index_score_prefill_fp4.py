# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 lightning-indexer PREFILL block scores, MXFP4 index cache.

The prefill scorer is bound by its score stores, not by reading K
(`minimax_m3_index_score_prefill`), so halving the K bytes buys nothing there
and a native MXFP4 kernel would mean reworking its LDS pipeline for no gain.
This entry instead dequantizes the pages the batch references into a compact
fp8 staging buffer and hands that to the fp8 prefill scorer unchanged. The
scores are therefore bit for bit those of "dequantize to fp8, then score",
which is also what the fp8 index cache's contract amounts to.

Exactness. e2m1 x 2^e is exactly representable in e4m3 when the biased e8m0
scale lies in [119, 133] (every element is then a multiple of 2^-9, the
smallest e4m3 subnormal, and at most 6 * 2^6 = 384 <= 448). Outside that
window the staged value saturates to +-448 or flushes towards zero; the fp8
index cache, which casts unit-scale, has the same range limit, so this is not
a new one. e8m0 bytes 0 and 255 are outside the contract, as in the decode
scorer.

Cost. Every page the batch scores is read once (8.5 KB) and written once
(16 KB) per call, before the scorer runs; a serving caller pays it once per
layer. The staging buffer is `batch * max_block` pages of 16 KB.

Layout, dtypes and strides of `k_data`/`k_scale` are the decode scorer's; see
`minimax_m3_index_score_fp4`.
"""

import torch
import triton
import triton.language as tl

from aiter import dtypes
from aiter.ops.flydsl.kernels.minimax_m3_index_score import (
    HEAD_DIM,
    PAGE,
    _validate_arch,
    _validate_rows,
    _validate_tensor,
)
from aiter.ops.flydsl.kernels.minimax_m3_index_score_fp4 import _validate_cache
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    alloc_score,
    score_prefill_flydsl,
)

__all__ = ["alloc_score", "dequant_pages_fp8", "score_prefill_flydsl_fp4"]

TOK_BLOCK = 32  # tokens one program dequantizes
NUM_WARPS = 4
FP8_MAX = tl.constexpr(448.0)


@triton.jit
def _e2m1_to_f32(code):
    """4-bit e2m1 codes (int32) to f32: sign, 2 exponent bits, 1 mantissa bit;
    0 and 0.5 are the subnormals."""
    mag = code & 7
    ex = mag >> 1
    man = (mag & 1).to(tl.float32)
    pow2 = (1 << tl.maximum(ex - 1, 0)).to(tl.float32)
    val = tl.where(ex == 0, 0.5 * man, (1.0 + 0.5 * man) * pow2)
    return tl.where(code >= 8, -val, val)


@triton.jit
def _dequant_pages_kernel(
    k_data,
    k_scale,
    block_table,
    seq_lens,
    out,
    stride_kd,
    stride_ks,
    stride_bt,
    max_block,
    q_in,
    q_out,
    stride_qn,
    stride_qh,
    n_q_rows,
    n_heads,
    K_PROGRAMS,
    PAGE_C: tl.constexpr,
    TOKS: tl.constexpr,
    DIM: tl.constexpr,
    HAS_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
):
    """One program: TOKS tokens of one (request, logical page) -> fp8.

    Output slot is `b * max_block + p`, so the staged block table is an
    arange. A page past the request's length is left alone, which is why the
    staging buffer is `empty`: the scorer never reads those pages' contents
    into a live score.

    With HAS_Q the grid also carries programs past the K ones that widen an e4m3
    query to bf16 (`Q_ROWS` rows each, row = token * heads + head): the scorer
    takes a bf16 query and rounds it back to e4m3 itself, which is the identity
    for a value that came from e4m3. Folding it into this launch costs no launch.
    """
    if HAS_Q and tl.program_id(0) >= K_PROGRAMS:
        qp = (tl.program_id(0) - K_PROGRAMS) * tl.num_programs(1) + tl.program_id(1)
        row = qp.to(tl.int64) * Q_ROWS + tl.arange(0, Q_ROWS)[:, None]
        col = tl.arange(0, DIM)[None, :]
        ok = row < n_q_rows
        src = (row // n_heads) * stride_qn + (row % n_heads) * stride_qh + col
        x = tl.load(q_in + src, mask=ok, other=0.0).to(tl.float32)
        tl.store(q_out + row * DIM + col, x.to(tl.bfloat16), mask=ok)
        return
    p = tl.program_id(0) // (PAGE_C // TOKS)
    tile = tl.program_id(0) % (PAGE_C // TOKS)
    b = tl.program_id(1)
    length = tl.load(seq_lens + b)
    if p * PAGE_C >= length:
        return
    page = tl.load(block_table + b.to(tl.int64) * stride_bt + p).to(tl.int64)
    tok = tile * TOKS + tl.arange(0, TOKS)[:, None, None]
    grp = tl.arange(0, DIM // 32)[None, :, None]  # e8m0 group of 32 elements
    j = tl.arange(0, 16)[None, None, :]  # byte within the group's 16
    byte = tl.load(k_data + page * stride_kd + tok * (DIM // 2) + grp * 16 + j).to(
        tl.int32
    )
    # One scale load per (token, group), broadcast over the group's 16 bytes.
    e8m0 = tl.load(k_scale + page * stride_ks + tok * (DIM // 32) + grp).to(tl.int32)
    scale = (e8m0 << 23).to(tl.float32, bitcast=True)
    # Low nibble is the even element, high nibble the odd one.
    lo = _e2m1_to_f32(byte & 0xF) * scale
    hi = _e2m1_to_f32(byte >> 4) * scale
    v = tl.minimum(tl.maximum(tl.join(lo, hi), -FP8_MAX), FP8_MAX)
    v = tl.reshape(v, (TOKS, DIM))
    tok2 = tile * TOKS + tl.arange(0, TOKS)[:, None]
    dst = (
        (b.to(tl.int64) * max_block + p) * (PAGE_C * DIM)
        + tok2 * DIM
        + tl.arange(0, DIM)[None, :]
    )
    tl.store(out + dst, v.to(out.dtype.element_ty))


Q_ROWS = 16  # query rows one widening program converts


def dequant_pages_fp8(
    k_data, k_scale, block_table, seq_lens, max_block, out=None, q=None, q_out=None
):
    """Dequantize the pages `block_table` names into fp8 `[batch*max_block, 128, 128]`.

    Slot `b * max_block + p` holds request b's logical page p, for p below its
    page count; the rest is left uninitialised.

    `q` (e4m3 `[rows, heads, 128]`) and `q_out` (contiguous bf16, same shape) ask
    the same launch to widen the query as well.
    """
    batch = seq_lens.shape[0]
    shape = (batch * max_block, PAGE, HEAD_DIM)
    if out is None:
        out = torch.empty(shape, dtype=dtypes.fp8, device=k_data.device)
    elif (
        out.dtype != dtypes.fp8
        or out.shape[0] < shape[0]
        or tuple(out.shape[1:]) != shape[1:]
        or not out.is_contiguous()
        or out.device != k_data.device
    ):
        raise ValueError(
            f"workspace: expected contiguous fp8 [>= {shape[0]}, 128, 128]"
        )
    k_programs = max_block * (PAGE // TOK_BLOCK)
    has_q = q is not None
    n_rows = q.shape[0] * q.shape[1] if has_q else 0
    q_chunks = -(-n_rows // Q_ROWS)
    q_programs = -(-q_chunks // batch)
    with torch.cuda.device(k_data.device):
        _dequant_pages_kernel[(k_programs + q_programs, batch)](
            k_data,
            k_scale,
            block_table,
            seq_lens,
            out,
            k_data.stride(0),
            k_scale.stride(0),
            block_table.stride(0),
            max_block,
            q if has_q else out,
            q_out if has_q else out,
            q.stride(0) if has_q else 0,
            q.stride(1) if has_q else 0,
            n_rows,
            q.shape[1] if has_q else 1,
            k_programs,
            PAGE_C=PAGE,
            TOKS=TOK_BLOCK,
            DIM=HEAD_DIM,
            HAS_Q=has_q,
            Q_ROWS=Q_ROWS,
            num_warps=NUM_WARPS,
        )
    return out


_IDENTITY = {}


def _identity_table(batch, max_block, device):
    """`[batch, max_block]` arange, the staged block table. Memoised: it only
    depends on the two bounds."""
    key = (batch, max_block, device)
    if key not in _IDENTITY:
        _IDENTITY[key] = torch.arange(
            batch * max_block, dtype=torch.int32, device=device
        ).view(batch, max_block)
    return _IDENTITY[key]


def score_prefill_flydsl_fp4(
    idx_q,
    k_data,
    k_scale,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    max_seq_len,
    sm_scale,
    out=None,
    workspace=None,
    q_scale: float = 1.0,
):
    """Block scores for a ragged prefill batch over an MXFP4 index cache.

    `score_prefill_flydsl` with the cache replaced by `k_data`/`k_scale`; every
    argument it shares keeps its contract. `workspace` optionally supplies the
    fp8 staging buffer (see `dequant_pages_fp8`); otherwise one is allocated.
    The staged pages are consumed by this call, not kept.

    `idx_q` is bf16 (rounded to e4m3 inside the scorer, as for an fp8 cache) or
    already e4m3 (`dtypes.fp8`, what the index-Q writer emits for q8kv4), in which
    case it is widened to a bf16 staging query in the same launch that dequantizes
    the pages -- exact, and the scorer's own rounding then gives back the same
    bytes, so the scores are those of the e4m3 query as stored. `q_scale` (default
    1.0, positive) multiplies the scores, folded into `sm_scale`.
    """
    dev = idx_q.device
    _validate_arch(dev)
    q_fp8 = idx_q.dtype == dtypes.fp8
    if q_fp8:
        _validate_tensor(idx_q, "idx_q", (dtypes.fp8,), 3, dev, 16)
        if idx_q.shape[2] != HEAD_DIM:
            raise ValueError(f"index head_dim must be {HEAD_DIM}, got {idx_q.shape[2]}")
        _validate_rows(idx_q, "idx_q")
    scaled = sm_scale * q_scale
    if not scaled > 0 or scaled == float("inf"):
        raise ValueError("sm_scale * q_scale must be finite and positive")
    _validate_cache(k_data, k_scale, dev)
    _validate_tensor(block_table, "block_table", (torch.int32,), 2, dev)
    _validate_tensor(seq_lens, "seq_lens", (torch.int32,), 1, dev)
    if not isinstance(max_seq_len, int) or max_seq_len < 1:
        raise ValueError(f"max_seq_len must be a positive int, got {max_seq_len}")
    batch = cu_seqlens_q.shape[0] - 1
    max_block = -(-max_seq_len // PAGE)
    if (
        batch < 1
        or seq_lens.shape[0] < batch
        or block_table.shape[0] < batch
        or block_table.shape[1] < max_block
        or block_table.stride(1) != 1
        or block_table.stride(0) < block_table.shape[1]
    ):
        raise ValueError(
            "block_table/seq_lens must cover every request and max_seq_len"
        )
    q_wide = (
        torch.empty(idx_q.shape, dtype=torch.bfloat16, device=dev) if q_fp8 else None
    )
    staged = dequant_pages_fp8(
        k_data,
        k_scale,
        block_table,
        seq_lens[:batch],
        max_block,
        workspace,
        q=idx_q if q_fp8 else None,
        q_out=q_wide,
    )
    return score_prefill_flydsl(
        q_wide if q_fp8 else idx_q,
        staged,
        _identity_table(batch, max_block, dev),
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        max_seq_len,
        scaled,
        out=out,
    )

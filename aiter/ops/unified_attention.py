# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from ..jit.core import compile_ops


# -----------------------------------------------------------------------------
# Triton combine kernel for the CK split-KV workspace layout.
#
# Algorithmically identical to the FlashDecoding `reduce_segments` kernel in
# `aiter.ops.triton._triton_kernels.attention.unified_attention`, just adapted
# to the CK kernel's workspace layout:
#   * o_acc / lse_acc are head-major ([H, S, T, D] / [H, S, T]) instead of
#     token-major.
#   * `lse_acc` is natural-log (m + log(l)) — a single fused tensor instead
#     of separate `segm_max` / `segm_expsum`.
#   * `o_acc` is already normalized by its per-split `l`, so the equivalent
#     of `segm_expsum` is implicitly 1.
#   * Empty splits are encoded by lse == -inf (host pre-fills lse_acc with
#     -inf; the CK kernel only overwrites real splits). No seq_lens-derived
#     mask is needed.
# -----------------------------------------------------------------------------
@triton.jit
def _fast_exp(x):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    return tl.math.exp2(x * RCP_LN2)


@triton.jit
def _reduce_segments_ck_layout(
    output_ptr,       # [num_tokens, num_query_heads, head_size]  (bf16/fp16)
    o_acc_ptr,        # [num_query_heads, num_splits, num_tokens, head_size]  fp32
    lse_acc_ptr,      # [num_query_heads, num_splits, num_tokens]             fp32
    num_tokens,       # int (runtime, stride multiplier)
    num_splits,       # int (runtime, stride multiplier + active-split mask)
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    NUM_SPLITS_PADDED: tl.constexpr,
):
    query_token_idx = tl.program_id(0)
    query_head_idx  = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)

    s_idx      = tl.arange(0, NUM_SPLITS_PADDED)
    split_mask = s_idx < num_splits

    lse_offset = (
        query_head_idx.to(tl.int64) * (num_splits * num_tokens)
        + s_idx.to(tl.int64) * num_tokens
        + query_token_idx
    )
    lse = tl.load(lse_acc_ptr + lse_offset, mask=split_mask, other=float("-inf"))

    is_empty = lse == float("-inf")
    lse_safe = tl.where(is_empty, -1.0e38, lse)
    lse_max  = tl.max(lse_safe)

    weight = _fast_exp(lse - lse_max)
    weight = tl.where(is_empty, 0.0, weight)
    weight_sum = tl.sum(weight)
    weight_sum_safe = tl.where(weight_sum == 0.0, 1.0, weight_sum)

    o_offset = (
        query_head_idx.to(tl.int64) * (num_splits * num_tokens * HEAD_SIZE)
        + s_idx[:, None].to(tl.int64) * (num_tokens * HEAD_SIZE)
        + query_token_idx * HEAD_SIZE
        + offs_d[None, :]
    )
    o = tl.load(
        o_acc_ptr + o_offset,
        mask=split_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    o = o * weight[:, None]
    acc = tl.sum(o, axis=0) / weight_sum_safe

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + offs_d
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


# -----------------------------------------------------------------------------
# JIT-compiled C++ kernel entry point. This is the raw 1:1 binding to the
# pybind symbol `unified_attention_fwd` exposed by `module_unified_attention`.
# All caller-facing code should go through `unified_attention_fwd` below.
# -----------------------------------------------------------------------------
def _gen_unified_attention_fwd_kernel_fake(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_len: torch.Tensor,
    mask_type: int,
    scale_s: float,
    scale: float,
    scale_k: float,
    scale_v: float,
    scale_out: float,
    cache_ptr_int32_overflow_possible: bool = False,
    num_splits: int = 1,
    o_acc_workspace: Optional[torch.Tensor] = None,
    lse_acc_workspace: Optional[torch.Tensor] = None,
    q_descale: float = 1.0,
    k_descale: float = 1.0,
    v_descale: float = 1.0,
    max_seqlen_q_override: int = 0,
) -> None:
    return None


@compile_ops(
    "module_unified_attention",
    fc_name="unified_attention_fwd",
    gen_fake=_gen_unified_attention_fwd_kernel_fake,
)
def _unified_attention_fwd_kernel(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_len: torch.Tensor,
    mask_type: int,
    scale_s: float,
    scale: float,
    scale_k: float,
    scale_v: float,
    scale_out: float,
    cache_ptr_int32_overflow_possible: bool = False,
    num_splits: int = 1,
    o_acc_workspace: Optional[torch.Tensor] = None,
    lse_acc_workspace: Optional[torch.Tensor] = None,
    q_descale: float = 1.0,
    k_descale: float = 1.0,
    v_descale: float = 1.0,
    max_seqlen_q_override: int = 0,
) -> None: ...


# -----------------------------------------------------------------------------
# Transparent split-KV plumbing.
#
# When the caller opts in (allow_splitkv=True, default) and doesn't pass an
# explicit num_splits / workspace pair, the wrapper:
#   1. picks num_splits from a cheap CTA-occupancy heuristic,
#   2. allocates FP32 (o_acc, lse_acc) workspaces if num_splits > 1,
#   3. runs the kernel with gridDim.z == num_splits,
#   4. merges the per-split partials into `output` with an LSE combine.
# -----------------------------------------------------------------------------
_NUM_CUS_CACHE: dict[Optional[int], int] = {}


def _num_cus(device: torch.device) -> int:
    """Cached multi_processor_count lookup."""
    key = device.index if device.type == "cuda" else None
    cached = _NUM_CUS_CACHE.get(key)
    if cached is None:
        cached = torch.cuda.get_device_properties(device).multi_processor_count
        _NUM_CUS_CACHE[key] = cached
    return cached


def _pick_num_splits(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    seq_lens: torch.Tensor,
) -> int:
    """Pick KV-splits to oversubscribe CTAs ~4x the device's CU count.

    Cost model (pure-CPU, no device sync — safe under CUDA graph capture):
        base_ctas   = num_kv_heads * q_tiles
        target_ctas = num_cus * 4
        raw_splits  = ceil(target_ctas / base_ctas)
        num_splits  = clamp(next_pow2(raw_splits), 1, 128)

    Why 4x oversubscription + cap 128 (was 2x + cap 16):
      * Mirrors the Triton FlashDecoding heuristic in `select_3d_config`
        (`target_num_prgms = cu_count * 4`, `min(num_segments, 128)`),
        so the two backends pick comparable split counts at the same
        shape and we're A/B-comparing kernels, not heuristics.
      * The 2x + cap-16 ceiling capped a 304-CU MI300X at only 16x8 =
        128 CTAs for b=1 GQA-8 decode (~42% occupancy on MI300X, ~50%
        on the 256-CU MI355X) — leaving ~half the device idle and
        bottlenecking on serial per-split KV traversal at long sk.
        At cap=64 the b=1 hd=128 sk=128K decode still ran 0.52x
        Triton; at cap=128 it reaches 0.81x — the gap was per-split
        work, not occupancy.
      * Workspace stays small: o_acc is fp32 [num_q_heads, num_splits,
        total_q, head_size]. At cap=128, batch=4, hd=128, GQA-8
        q_heads=64 → 64*128*4*128*4 ≈ 16 MiB. Combine cost scales
        linearly with num_splits per (token, head); the Triton
        combine kernel reads each split's o_acc/lse_acc exactly once.
      * Rounding to next_pow2 matches the combine kernel's
        NUM_SPLITS_PADDED = next_pow2(num_splits), so we don't pay
        for masked-out lanes — and stays compatible with the
        existing CK launch path (gridDim.z = num_splits is unchanged).

    `q_tiles` is approximated from `avg_q` to follow the C++
    select_config tile-tier ladder; mixed prefill+decode batches may
    be slightly over- or under-estimated but the clamp absorbs the
    error.

    A split that ends up with zero KV pages (because num_splits > the
    seq's KV-page count) is harmless: the kernel writes -inf to that
    split's lse_acc, and _combine_splits drops -inf rows from the
    merge. We intentionally do not read seq_lens off the device here
    to keep the wrapper compatible with CUDA-graph capture and avoid
    per-call syncs.
    """
    env = os.environ.get("AITER_UA_FORCE_SPLITS")
    if env is not None:
        # Caller is responsible for passing a power-of-2 value (or 1).
        return max(1, int(env))

    total_q       = query.shape[0]
    num_q_heads   = query.shape[1]
    num_kv_heads  = key_cache.shape[2]
    num_seqs      = seq_lens.shape[0]
    if num_seqs <= 0:
        return 1
    num_qpkv = num_q_heads // num_kv_heads

    # Estimate q_tiles from avg_q (no device sync). Mirrors the C++
    # select_config tile-tier ladder closely enough for the heuristic.
    avg_q = total_q // num_seqs
    if num_qpkv == 1:
        # d=128 MHA: kBlockQ ∈ {16, 32, 128, 256}
        if avg_q <= 16:
            kBlockQ = 16
        elif avg_q <= 32:
            kBlockQ = 32
        elif avg_q <= 128:
            kBlockQ = 128
        else:
            kBlockQ = 256
    else:
        # d=64 GQA-8: kBlockQ ∈ {2, 8, 16}
        kBlockQ_tiny  = 16 // num_qpkv   # 2
        kBlockQ_small = 64 // num_qpkv   # 8
        if avg_q <= kBlockQ_tiny:
            kBlockQ = kBlockQ_tiny
        elif avg_q <= kBlockQ_small:
            kBlockQ = kBlockQ_small
        else:
            kBlockQ = 128 // num_qpkv    # 16 (decode_*_m128 / prefill share kBlockQ=16)

    q_tiles     = max(1, (total_q + kBlockQ - 1) // kBlockQ)
    base_ctas   = num_kv_heads * q_tiles
    num_cus     = _num_cus(query.device)

    # Prefill-regime saturation guard. The 4x-CU oversubscription rule
    # is calibrated for the decode regime (small total_q, base_ctas
    # well below num_cus, sk that dominates per-CTA work). For prefill
    # (avg_q in the thousands) base_ctas typically already saturates
    # the device with q-tiles alone, and splitting K just multiplies
    # the combine kernel's per-(token,head) reduce + the workspace
    # alloc, without adding useful parallelism — the per-CTA K-traversal
    # was already short relative to combine's overhead.
    #
    # Measured on a vLLM-captured production trace (d=128, GQA-6, FP8,
    # sq=sk=1000): the unguarded heuristic picks splits=4 for b=4 and
    # splits=2 for b=8, costing 2.6x / 1.85x in CK time vs splits=1.
    # Both shapes already have base_ctas >= num_cus (500 and 1000
    # against 256 on MI355X), so the device is already saturated by Q
    # tiles alone — the K-split fan-out is pure overhead.
    #
    # Decode is unaffected: avg_q == 1 keeps the predicate false on
    # every batch in the production trace (the largest base_ctas a
    # decode batch hits at avg_q=1 is num_kv_heads * ceil(batch/kBlockQ_tiny)).
    #
    # Chunked prefill (small avg_q on top of long sk) is also unaffected
    # because base_ctas stays well below num_cus when q_tiles per
    # sequence is small.
    if avg_q > 8 and base_ctas >= num_cus:
        return 1

    target_ctas = num_cus * 4
    # ceil-div so a single under-saturated base_ctas still gets bumped
    # to the next power-of-2 splits tier instead of rounding down to 0.
    raw_splits  = (target_ctas + base_ctas - 1) // max(1, base_ctas)
    # Round up to next pow2 so combine's NUM_SPLITS_PADDED == num_splits
    # (no masked-out reduce lanes wasted).
    pow2_splits = 1
    while pow2_splits < raw_splits:
        pow2_splits <<= 1
    return max(1, min(128, pow2_splits))


def _combine_splits(
    output: torch.Tensor,
    o_acc: torch.Tensor,
    lse_acc: torch.Tensor,
) -> None:
    """FlashDecoding-style LSE merge via a Triton kernel.

    This is a thin wrapper around `reduce_segments_ck_layout`, which is the
    CK-layout sibling of `reduce_segments` (the kernel that Triton-UA's 3D
    path uses). Algorithmically identical — both fuse the LSE rescale +
    weighted sum into a single Triton launch — so combine-step overhead is
    the same on both backends, eliminating it as a confounder when
    comparing CK vs Triton attention-kernel performance.

    o_acc   : [nhead, num_splits, total_q, hdim]  fp32, contiguous,
                                                  per-split-normalized
    lse_acc : [nhead, num_splits, total_q]        fp32, contiguous,
                                                  natural-log (m + log(l))
    output  : [total_q, nhead, hdim]              bf16/fp16 (CK output dtype)

    Per (t, h):
        lse_max  = max_s lse[h, s, t]
        w[s]     = exp(lse[h, s, t] - lse_max)   ( -inf rows -> 0 )
        out[t,h] = sum_s o_acc[h, s, t, :] * w[s] / sum_s w[s]
    """
    num_q_heads, num_splits, num_tokens, head_size = o_acc.shape
    head_size_padded   = triton.next_power_of_2(head_size)
    # NUM_SPLITS_PADDED is the `tl.arange` upper bound — only it needs to
    # be a power of 2. `num_splits` itself is the actual stride multiplier
    # and the kernel masks the [num_splits, NUM_SPLITS_PADDED) tail.
    num_splits_padded  = triton.next_power_of_2(num_splits)
    grid = (num_tokens, num_q_heads)
    _reduce_segments_ck_layout[grid](
        output_ptr=output,
        o_acc_ptr=o_acc,
        lse_acc_ptr=lse_acc,
        num_tokens=num_tokens,
        num_splits=num_splits,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        NUM_SPLITS_PADDED=num_splits_padded,
        num_warps=2 if num_splits >= 4 else 1,
        num_stages=1,
    )


# -----------------------------------------------------------------------------
# Public entry point.
#
# Default behavior (allow_splitkv=True, no explicit overrides): the wrapper
# picks num_splits, allocates the FP32 workspaces if needed, runs the kernel,
# and merges the per-split partials into `output`. The caller sees the same
# API as before — just with split-KV applied opportunistically.
#
# Explicit-override path: if the caller passes num_splits > 1 OR provides
# workspaces, the wrapper bypasses the heuristic and forwards the explicit
# values straight to the kernel. In that case the caller owns the combine.
# Set `allow_splitkv=False` to disable the transparent path entirely.
# -----------------------------------------------------------------------------
def unified_attention_fwd(
    output: torch.Tensor,           # [num_tokens, num_heads_q, head_size]
    query: torch.Tensor,            # [num_tokens, num_heads_q, head_size]
    key_cache: torch.Tensor,        # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,      # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,     # [num_seqs, max_num_blocks_per_seq]
    seq_lens: torch.Tensor,         # [num_seqs]
    query_start_len: torch.Tensor,  # [num_seqs + 1]
    mask_type: int,                 # 0: no mask, 2: causal
    scale_s: float,
    scale: float,
    scale_k: float,
    scale_v: float,
    scale_out: float,
    cache_ptr_int32_overflow_possible: bool = False,
    # Opt-in for transparent split-KV. Default-on. When True the wrapper picks
    # num_splits, allocates workspaces, and combines into `output`. Set False
    # to force the single-launch path regardless of shape.
    allow_splitkv: bool = True,
    # Explicit overrides. Non-default values bypass the transparent path and
    # the caller is responsible for the combine (when num_splits > 1).
    num_splits: int = 1,
    # o_acc_workspace  : float32 [num_q_heads, num_splits, num_tokens, head_size]
    # lse_acc_workspace: float32 [num_q_heads, num_splits, num_tokens]
    o_acc_workspace: Optional[torch.Tensor] = None,
    lse_acc_workspace: Optional[torch.Tensor] = None,
    # Per-tensor FP8 descales — mirror Triton unified_attention's q_scale,
    # k_scale, v_scale device-tensor arguments but passed as float32 scalars
    # here. The kernel folds q_descale*k_descale into the softmax scale and
    # applies v_descale once to o_acc outside the K/V loop. For non-FP8
    # dtypes leave these at 1.0f (the default) and the kernel is a no-op
    # w.r.t. these arguments.
    q_descale: float = 1.0,
    k_descale: float = 1.0,
    v_descale: float = 1.0,
    # Optional caller-side override of max_seqlen_q used by C++ select_config.
    # 0 (default) keeps the conservative `num_tokens` heuristic. Pass the real
    # per-seq max when known (e.g. uniform-sq benchmarks) to enable the
    # tighter decode_d{64,128}_m{16,32,128} tiers instead of falling through
    # to prefill_d{64,128}.
    max_seqlen_q: int = 0,
) -> None:
    explicit_override = (
        num_splits > 1
        or o_acc_workspace is not None
        or lse_acc_workspace is not None
    )

    if explicit_override or not allow_splitkv:
        # Explicit-override or opt-out: forward straight to the kernel.
        _unified_attention_fwd_kernel(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            query_start_len,
            mask_type,
            scale_s,
            scale,
            scale_k,
            scale_v,
            scale_out,
            cache_ptr_int32_overflow_possible,
            num_splits,
            o_acc_workspace,
            lse_acc_workspace,
            q_descale,
            k_descale,
            v_descale,
            max_seqlen_q,
        )
        return

    # Transparent split-KV: heuristic picks num_splits and the wrapper owns
    # the workspace + combine.
    chosen = _pick_num_splits(query, key_cache, seq_lens)
    if chosen <= 1:
        _unified_attention_fwd_kernel(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            query_start_len,
            mask_type,
            scale_s,
            scale,
            scale_k,
            scale_v,
            scale_out,
            cache_ptr_int32_overflow_possible,
            1,
            None,
            None,
            q_descale,
            k_descale,
            v_descale,
            max_seqlen_q,
        )
        return

    total_q, num_q_heads, head_size = query.shape
    device = query.device
    o_acc = torch.empty(
        num_q_heads, chosen, total_q, head_size,
        dtype=torch.float32, device=device,
    )
    lse_acc = torch.full(
        (num_q_heads, chosen, total_q), float("-inf"),
        dtype=torch.float32, device=device,
    )

    # The kernel writes per-split partials into o_acc/lse_acc and ignores
    # `output` when num_splits > 1. The combine below produces the final result.
    _unified_attention_fwd_kernel(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_len,
        mask_type,
        scale_s,
        scale,
        scale_k,
        scale_v,
        scale_out,
        cache_ptr_int32_overflow_possible,
        chosen,
        o_acc,
        lse_acc,
        q_descale,
        k_descale,
        v_descale,
        max_seqlen_q,
    )
    _combine_splits(output, o_acc, lse_acc)

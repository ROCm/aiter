# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import Optional

import torch

from ..jit.core import compile_ops


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
    """Pick KV-splits to oversubscribe CTAs ~2x the device's CU count.

    Cost model:
        base_ctas   = num_kv_heads * q_tiles
        target_ctas = num_cus * 2
        num_splits  = clamp(target_ctas / base_ctas, 1, min(16, kv_tile_cap))

    where kv_tile_cap = ceil(min_kv_len / page_size) prevents creating splits
    that would get zero work, and 16 is a hard ceiling to keep the combine
    cheap.

    `q_tiles` is approximated from `avg_q` only (no device sync on
    `query_start_len`) — for mixed prefill+decode batches this may slightly
    over- or under-estimate the true tile count, but the formula is robust
    to that since `clamp` absorbs the error.

    Returns 1 when splitting wouldn't help (already plenty of CTAs, or the
    min KV len is shorter than one page).
    """
    env = os.environ.get("AITER_UA_FORCE_SPLITS")
    if env is not None:
        return max(1, int(env))

    total_q       = query.shape[0]
    num_q_heads   = query.shape[1]
    page_size     = key_cache.shape[1]
    num_kv_heads  = key_cache.shape[2]
    num_seqs      = seq_lens.shape[0]
    if num_seqs <= 0:
        return 1
    num_qpkv = num_q_heads // num_kv_heads

    # Estimate q_tiles from avg_q (no device sync). Mirrors the C++
    # select_config tile-tier ladder closely enough for the heuristic.
    avg_q = total_q // num_seqs
    if num_qpkv == 1:
        # d=128 MHA: kBlockQ ∈ {128, 256}
        kBlockQ = 128 if avg_q <= 128 else 256
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

    q_tiles   = max(1, (total_q + kBlockQ - 1) // kBlockQ)
    base_ctas = num_kv_heads * q_tiles

    # One small sync: pull min KV len off the device so we never create a
    # split that gets zero pages.
    min_kv_len  = int(seq_lens.min().item())
    kv_tile_cap = max(1, (min_kv_len + page_size - 1) // page_size)

    target_ctas = _num_cus(query.device) * 2
    raw_splits  = target_ctas // max(1, base_ctas)
    return max(1, min(16, kv_tile_cap, raw_splits))


def _combine_splits(
    output: torch.Tensor,
    o_acc: torch.Tensor,
    lse_acc: torch.Tensor,
) -> None:
    """FlashDecoding-style LSE merge, written in pure torch ops.

    o_acc   : [nhead, num_splits, total_q, hdim]  fp32  (already normalized
                                                          per-split: divided by
                                                          its own l)
    lse_acc : [nhead, num_splits, total_q]        fp32  (natural-log domain)

    Writes the merged result into `output` (layout [total_q, nhead, hdim]).

    Combine math (per-split lse[s] = m[s] + log(l[s])):
        lse_max = max_s lse[s]
        w[s]    = exp(lse[s] - lse_max)
        out     = sum_s (o_acc[s] * w[s]) / sum_s w[s]

    Splits with lse == -inf (masked-out / empty) get zero weight so they
    don't contribute, and (-inf) - (-inf) is replaced with 0 to avoid NaN.
    """
    is_empty   = torch.isinf(lse_acc) & (lse_acc < 0)
    safe_lse   = torch.where(is_empty, torch.zeros_like(lse_acc), lse_acc)
    lse_max    = safe_lse.amax(dim=1, keepdim=True)
    weight     = torch.exp(safe_lse - lse_max)
    weight     = torch.where(is_empty, torch.zeros_like(weight), weight)
    weight_sum = weight.sum(dim=1, keepdim=True)
    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    w_full     = (weight / weight_sum).unsqueeze(-1)
    o_merged   = (o_acc * w_full).sum(dim=1)  # [nhead, total_q, hdim]
    output.copy_(o_merged.transpose(0, 1).to(output.dtype))


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
    )
    _combine_splits(output, o_acc, lse_acc)

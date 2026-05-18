# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import Optional

import torch
import triton

from ..jit.core import compile_ops
from .triton._triton_kernels.attention.unified_attention import (
    reduce_segments_ck_layout,
)


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
    """Pick KV-splits to oversubscribe CTAs ~2x the device's CU count.

    Cost model (pure-CPU, no device sync — safe under CUDA graph capture):
        base_ctas   = num_kv_heads * q_tiles
        target_ctas = num_cus * 2
        num_splits  = clamp(target_ctas / base_ctas, 1, 16)

    The 16 cap keeps the per-split workspace and the combine cheap. The
    Triton combine kernel handles non-power-of-2 split counts internally
    via `NUM_SPLITS_PADDED = next_pow2(num_splits)` with a runtime mask,
    so we don't pad the heuristic — picking exactly the right number of
    splits avoids wasted CTAs and workspace.

    `q_tiles` is approximated from `avg_q` to follow the C++ select_config
    tile-tier ladder; mixed prefill+decode batches may be slightly over- or
    under-estimated but the `clamp` absorbs the error.

    A split that ends up with zero KV pages (because num_splits > the seq's
    KV-page count) is harmless: the kernel writes -inf to that split's
    lse_acc, and _combine_splits drops -inf rows from the merge. We
    intentionally do not read seq_lens off the device here to keep the
    wrapper compatible with CUDA-graph capture and avoid per-call syncs.
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
    target_ctas = _num_cus(query.device) * 2
    raw_splits  = target_ctas // max(1, base_ctas)
    return max(1, min(16, raw_splits))


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
    reduce_segments_ck_layout[grid](
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

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import functools
import os

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_cu_num, get_gfx
from ..utility import dtypes


# Shape-aware AVO dispatch floor. See top_k_per_row_prefill's docstring for the
# measured crossover this comes from: many rows fill the machine at a narrower
# row and win from 32K, few rows do not and lose until 48K.
# stride0 at which `sampled` overtakes the FlyDSL one-block prefill path, which
# is what this function reaches below it. Measured per shape, k=2048, fp32, both
# ops under one @perftest on the same data (flydsl_us / sampled_us, above 1.00
# meaning `sampled` is faster):
#
#     M \ N     49152   65536  131072  262144
#     1          0.92    1.06    1.68    2.71
#     8          0.78    0.90    1.38    2.15
#     64         0.72    0.83    1.17    1.57
#     256        0.67    0.79    0.99    1.14
#     512        0.72    0.77    1.08    1.27
#     1024       0.89    0.96    1.07    1.30
#     2048       0.85    0.89    1.02    1.37
#     4096       0.83    0.92    1.04    1.32
#
# The crossover is a clean function of stride0 and not of the row count, so
# unlike the floor this replaces there is no wide/narrow split. M=256 at 131072
# is 0.99, a wash, and is left on the simple rule rather than carved out.
#
# This supersedes a 32768/49152 floor that was fitted against the mb/ob path.
# That measurement was not wrong when it was taken; upstream put FlyDSL in front
# of mb/ob afterwards, which made it a comparison against an op this function no
# longer reaches at those widths.
SAMPLED_MIN_STRIDE0 = 131072


# Raw binding: no argument validation, correction_bias must be a real tensor.
# Callers should use topk_gating() below.
@compile_ops("module_moe_topk", fc_name="topk_gating", develop=True)
def topk_gating_fwd(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,
    score_func: str = "sqrtsoftplus",
) -> None: ...


_VALID_SCORE_FUNCS = {"sqrtsoftplus", "sigmoid", "softmax"}


def _valid_bias_dtypes(gating_dtype: torch.dtype) -> tuple[torch.dtype, ...]:
    """Bias dtypes instantiated for this gating dtype; see _AITER_TOPK_GATING_SLICE.

    Checked in Python because the C++ side aborts rather than raising.
    """
    if gating_dtype is torch.float16:
        return (torch.float32,)
    return (torch.float32, torch.bfloat16)


def topk_gating(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None = None,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    score_func: str = "sqrtsoftplus",
) -> None:
    """Unified fused topk gating for MoE routing.

    Args:
        score_func: one of {"sqrtsoftplus" (DeepSeek V4-Pro default),
                            "sigmoid" (Llama4),
                            "softmax" (DeepSeek V3 / classic MoE)}.
        correction_bias: optional bias tensor, pass None for no bias. Must be
            float32, or bfloat16 when gating_output is not float16.
    """
    assert (
        score_func in _VALID_SCORE_FUNCS
    ), f"Unknown score_func '{score_func}', expected one of {_VALID_SCORE_FUNCS}"
    if correction_bias is None:
        correction_bias = torch.empty(
            0, dtype=torch.float32, device=gating_output.device
        )
    else:
        valid = _valid_bias_dtypes(gating_output.dtype)
        assert correction_bias.dtype in valid, (
            f"correction_bias dtype {correction_bias.dtype} is not supported for "
            f"{gating_output.dtype} gating_output, expected one of {valid}"
        )
    topk_gating_fwd(
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        need_renorm,
        routed_scaling_factor,
        score_func,
    )


# DEPRECATED: the kernel routes sigmoid and softmax as well, so the name is now
# topk_gating.  Kept until callers migrate.
topk_softplus = topk_gating


@compile_ops("module_moe_asm", fc_name="biased_grouped_topk", develop=True)
def biased_grouped_topk_hip(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_grp: int,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,
) -> None: ...


@compile_ops("module_moe_asm", develop=True)
def grouped_topk(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    is_softmax: bool = True,
    routed_scaling_factor: float = 1.0,
) -> None: ...


def gen_moe_fused_gate_fake_tensor(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    n_share_experts_fusion: int,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(
        topk_weights, dtype=topk_weights.dtype, device=topk_weights.device
    )

    indices = torch.empty_like(topk_ids, dtype=topk_ids.dtype, device=topk_ids.device)

    return [output, indices]


@compile_ops("module_moe_asm", fc_name="moe_fused_gate", develop=True)
def _moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    n_share_experts_fusion: int,
    routed_scaling_factor: float = 1.0,
) -> None: ...


def moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    n_share_experts_fusion: int,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # C side fills topk_weights / topk_ids in place and returns void; return the
    # (aliased) tensors to preserve the original API.
    _moe_fused_gate(
        input,
        bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        topk,
        n_share_experts_fusion,
        routed_scaling_factor,
    )
    return topk_weights, topk_ids


def biased_grouped_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
):
    token_num = gating_output.shape[0]
    num_experts = gating_output.shape[1]
    cu_num = get_cu_num()
    if token_num <= cu_num * 212 or num_experts // num_expert_group > 32:
        return biased_grouped_topk_hip(
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            need_renorm,
            routed_scaling_factor,
        )
    else:
        topk = topk_ids.shape[1]
        assert need_renorm, "Renormalization is required for moe_fused_gate."
        return moe_fused_gate(
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            topk,
            n_share_experts_fusion=0,
            routed_scaling_factor=routed_scaling_factor,
        )


# this one copied from sglang
def biased_grouped_topk_torch(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    return_score: bool = False,
):
    scores = gating_output.to(dtypes.fp32).sigmoid()
    num_token = scores.shape[0]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)

    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if return_score:
        return topk_weights.to(dtypes.fp32), topk_ids.to(dtypes.i32), scores
    else:
        return topk_weights.to(dtypes.fp32), topk_ids.to(dtypes.i32)


# this one copied from sglang
def grouped_topk_torch(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
):
    gating_output = gating_output.to(dtypes.fp32)
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Scoring function '{scoring_func}' is not supported.")

    num_token = scores.shape[0]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(dtypes.fp32), topk_ids.to(dtypes.i32)


@compile_ops("module_top_k_per_row", fc_name="top_k_per_row_prefill", develop=True)
def _top_k_per_row_prefill(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    workspace: torch.Tensor | None = None,
    stable: bool = False,
) -> None: ...


@compile_ops("module_top_k_per_row")
def topk_mb_workspace_size(
    numRows: int, stride0: int, k: int, is_decode: bool
) -> int: ...


@compile_ops("module_top_k_per_row")
def topk_ob_workspace_size(
    numRows: int, stride0: int, k: int, is_decode: bool
) -> int: ...


@compile_ops("module_top_k_per_row")
def topk_use_mulblocks(numRows: int, stride0: int) -> bool: ...


@functools.lru_cache(maxsize=1024)
def _mb_workspace_size_cached(
    numRows: int, stride0: int, k: int, is_decode: bool
) -> int:
    """topk_mb_workspace_size() memoised: 5.06 us per call through the binding.

    query_mb_workspace() is a pure function of (numRows, stride0, kTopK)
    (topk_per_row_kernels.cu:2705 -- no getenv, no device query), so the size is
    fixed for a shape and the second call onward is a dict hit.
    """
    return topk_mb_workspace_size(numRows, stride0, k, is_decode)


@functools.lru_cache(maxsize=1024)
def _ob_workspace_size_cached(
    numRows: int, stride0: int, k: int, is_decode: bool
) -> int:
    """topk_ob_workspace_size() memoised: 6.74 us per call, pure in
    (numRows, stride0, kTopK) (topk_per_row_kernels.cu:2721).

    This is the one every small shape pays, because low stride0 stays on the
    one-block path: at numRows=64 stride0=512 the binding query cost 6.74 us to
    size a workspace for a kernel that runs in 2.36 us.
    """
    return topk_ob_workspace_size(numRows, stride0, k, is_decode)


@functools.lru_cache(maxsize=1024)
def _use_mulblocks_cached(
    numRows: int, stride0: int, _force_path: str | None, _dispatch_factor: str | None
) -> bool:
    """topk_use_mulblocks() memoised: 6.34 us per call.

    should_use_mulblocks() (topk_per_row_kernels.cu:2648) compares the shape
    against thresholds chosen by CU count, and it already caches that CU count in
    a function-local static -- so the decision is device-pinned in C++ before we
    cache it here. It does re-read TOPK_FORCE_PATH and TOPK_DISPATCH_FACTOR every
    call, so both are in the key rather than assumed constant: two dict lookups
    against a 6.34 us binding call keeps those override knobs live for free.
    """
    return topk_use_mulblocks(numRows, stride0)


def _use_mulblocks(numRows: int, stride0: int) -> bool:
    return _use_mulblocks_cached(
        numRows,
        stride0,
        os.environ.get("TOPK_FORCE_PATH"),
        os.environ.get("TOPK_DISPATCH_FACTOR"),
    )


@functools.lru_cache(maxsize=16)
def _get_topk_mb_workspace_keyed(
    device: torch.device, stream_id: int, size: int
) -> torch.Tensor:
    return torch.zeros(size, dtype=torch.uint8, device=device)


def get_topk_mb_workspace(device: torch.device, size: int) -> torch.Tensor:
    """Return a per-(device, stream, bucketed-size) zero-initialized workspace
    for the multi-block radix top-k path.

    The mb kernel uses cross-block atomic counters / histograms that must start
    at zero; instead of a per-call ``hipMemset`` the kernel resets the scratch
    back to zero after each launch, so a cached zeroed buffer can be reused.
    Concurrent launches on different streams must not share the buffer, or their
    atomic counters get mixed. Do not call from paths that violate the kernel's
    self-reset invariant.

    ``size`` is data-dependent (batch / seq_len / k), so it is rounded up to the
    next power of two before keying/allocating. That bounds the number of
    distinct cached buffers to ~log2(max_size) magnitudes (and the LRU cap of 16
    bounds it further) instead of one buffer per exact shape, trading <=2x size
    per buffer for far fewer retained buffers. The C++ side lays out its scratch
    within the first ``size`` bytes, so a larger (rounded) buffer is fine.
    """
    # Round up to the next power of two (size >= 1) to bucket nearby shapes.
    alloc = 1 if size <= 1 else 1 << (int(size) - 1).bit_length()
    stream = torch.cuda.current_stream(device)
    return _get_topk_mb_workspace_keyed(device, stream.cuda_stream, alloc)


def get_topk_scratch_workspace(device: torch.device, size: int) -> torch.Tensor:
    """Return an exact-size scratch workspace for the one-block (ob) / radix
    top-k paths.

    Unlike the multi-block buffer (get_topk_mb_workspace), these kernels do their
    own internal memset on each launch, so the buffer need not be zero-initialized
    and need not be a persistent, reused buffer. This mirrors how the C++ side
    originally allocated it — a plain, exactly-sized ``torch.empty`` per call —
    only moved to the Python side so the host code never allocates device scratch
    itself. torch's caching allocator reuses freed blocks, so no explicit cache
    (or size bucketing) is needed here."""
    return torch.empty(max(1, int(size)), dtype=torch.uint8, device=device)


_FLYDSL_TOPK_PREFILL_DISABLED = os.environ.get(
    "AITER_DISABLE_FLYDSL_TOPK_PREFILL", "0"
) in ("1", "true", "True", "yes", "YES")


def top_k_per_row_prefill(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
) -> None:
    """Per-row top-k (prefill). Both the multi-block and one-block paths run on a
    caller-provided workspace allocated (and cached) on the Python side, so the
    C++ kernels never allocate device scratch. The mb path needs a zeroed,
    self-reset buffer (get_topk_mb_workspace); the ob path uses plain scratch
    (get_topk_scratch_workspace).

    When stable=True, the one-block path is forced with deterministic,
    ascending-index ordered, smallest-index tie-breaking emit so every
    tensor-parallel rank selects and orders an identical KV set; the caller sizes
    the workspace for the ob path in that case.

    When stable=False and stride1 == 1 and topk_sampled_supports() returns true and
    stride0 is at least SAMPLED_MIN_STRIDE0, dispatches to
    top_k_per_row_prefill_sampled. Set AITER_DISABLE_TOPK_SAMPLED=1 to force the original
    mb/ob path for A/B or fallback.

    The threshold is shape-aware because a flat one was wrong in both directions.
    It used to be a flat 32768, which sent every numRows <= 512 shape at
    stride0 = 32768 to a path that is SLOWER than the mb/ob one it replaced --
    0.81x to 0.94x, measured through this dispatch with both backends under one
    @perftest. No gate in the repo could see it: score_grid compares `sampled` against
    its own earlier baseline, so a cell that is correct, not regressing, and
    simply worse than the op it replaces is invisible.

    Measured crossover, aiter_us / sampled_us, below 1.00 meaning `sampled` is slower
    (log/crossover/, k=2048, fp32):

        M \\ N     32K   40K   48K   56K   64K
        1         0.87  1.05  1.15  1.31  1.22
        8         0.88  1.05  1.22  1.09  1.27
        64        0.94  0.95  1.15  1.11  1.28
        256       0.81  0.84  1.04  1.04  1.23
        512       0.91  0.92  1.03  1.02  1.19
        1024      1.25  1.26  1.40  1.45  1.52
        2048      1.14  1.17  1.28  1.30  1.38
        4096      1.02  1.04  1.19  1.24  1.34

    numRows >= 1024 wins from 32768 already, so its floor stays there; everything
    below only turns positive at 49152. Raising the threshold to 65536 for all
    shapes was the other tempting answer and it gives up the 1.15-1.22x that
    numRows <= 64 earns at 48K.

    stride1 is part of the routing condition and not of topk_sampled_supports(),
    which only takes (numRows, stride0, k) and so cannot see it. The mb/ob path
    ignores stride1 entirely, while the `sampled` entry asserts it is 1; routing a
    stride1 != 1 call here would therefore turn a working (if questionable) call
    into an abort purely because `sampled` became available. Keeping the assert for
    direct callers of top_k_per_row_prefill_sampled and routing around it here
    preserves the pre-`sampled` behaviour exactly."""
    # Ahead of the FlyDSL check, and gated at a floor measured against FlyDSL
    # rather than against mb/ob -- see SAMPLED_MIN_STRIDE0. Behind it this was
    # unreachable: FlyDSL served every shape tested, M=4096 N=65536 included.
    if (
        not stable
        and stride1 == 1
        and stride0 >= SAMPLED_MIN_STRIDE0
        and _sampled_supports_cached(numRows, stride0, k)
        and os.environ.get("AITER_DISABLE_TOPK_SAMPLED", "0") != "1"
    ):
        return top_k_per_row_prefill_sampled(
            logits,
            rowStarts,
            rowEnds,
            indices,
            values,
            numRows,
            stride0,
            stride1,
            k,
        )

    use_mulblocks = not stable and _use_mulblocks(numRows, stride0)
    # FlyDSL one-block outperforms HIP one-block on the remaining prefill cases.
    if not use_mulblocks and not _FLYDSL_TOPK_PREFILL_DISABLED:
        from .flydsl.topk.topk_per_row import (
            _is_flydsl_radix_topk_one_block_supported,
        )

        if _is_flydsl_radix_topk_one_block_supported(
            logits,
            rowStarts,
            rowEnds,
            indices,
            values,
            numRows,
            stride0,
            stride1,
            k,
        ):
            return flydsl_radix_topk_one_block_prefill(
                logits,
                rowStarts,
                rowEnds,
                indices,
                values,
                numRows,
                stride0,
                stride1,
                k,
                stable,
            )

    if use_mulblocks:
        size = _mb_workspace_size_cached(numRows, stride0, k, False)
        workspace = get_topk_mb_workspace(logits.device, size)
    else:
        size = _ob_workspace_size_cached(numRows, stride0, k, False)
        workspace = get_topk_scratch_workspace(logits.device, size)
    return _top_k_per_row_prefill(
        logits,
        rowStarts,
        rowEnds,
        indices,
        values,
        numRows,
        stride0,
        stride1,
        k,
        workspace,
        stable,
    )


@compile_ops("module_top_k_per_row", fc_name="top_k_per_row_prefill_sampled", develop=True)
def _top_k_per_row_prefill_sampled(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    workspace: torch.Tensor | None = None,
) -> None: ...


@compile_ops("module_top_k_per_row")
def topk_sampled_workspace_size(numRows: int, stride0: int, k: int) -> int: ...


@compile_ops("module_top_k_per_row")
def topk_sampled_supports(numRows: int, stride0: int, k: int) -> bool: ...


@functools.lru_cache(maxsize=1024)
def _sampled_supports_cached(numRows: int, stride0: int, k: int) -> bool:
    """topk_sampled_supports() memoised, because the binding call is not cheap.

    Measured in the correctness image: 4.86 us per call, against a kernel that
    is 43 us at numRows=64 stride0=65537. Adding one unmemoised call to the
    validation below cost +12.5% there and +7.5% on average across the twelve
    shapes in reports/odd_n_ab.tsv -- a constant ~5 us offset that did not grow
    with the work, which is what host overhead looks like.

    Safe to cache: topk_sampled_supports is a pure function of these three ints.
    It computes sampled::params_for -> derive_shape_params, which reads no device
    state (CU_COUNT is a constexpr in topk_shape.hip.hpp, and there is no
    hipGetDeviceProperties anywhere in that header).
    """
    return bool(topk_sampled_supports(numRows, stride0, k))


@functools.lru_cache(maxsize=1024)
def _sampled_workspace_size_cached(numRows: int, stride0: int, k: int) -> int:
    """topk_sampled_workspace_size() memoised, for the same reason and the same
    measured cost: 4.80 us per call through the binding.

    Together with the supports() lookup above, these two were 9.66 us of every
    call that routes to AVO, and at small M that was the whole call. Measured on
    the enqueue side, where a profiler kernel trace cannot see it:
    M=16 N=32768 spent 25.29 us on the host against 25.43 us end to end, so 99.4%
    of the call was the caller's CPU and the GPU work was entirely hidden behind
    it. Optimising the kernel there would have changed nothing.

    Safe to cache for the same reason: it is a pure function of these three ints.
    topk_sampled_workspace_size computes params_for -> derive_shape_params and then
    ws_layout, none of which reads device state (CU_COUNT is a constexpr in
    topk_shape.hip.hpp).
    """
    return int(topk_sampled_workspace_size(numRows, stride0, k))


def top_k_per_row_prefill_sampled(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    workspace: torch.Tensor | None = None,
) -> None:
    """Per-row top-k (prefill) via the topk-prefill-avo kernels.

    Same call shape as top_k_per_row_prefill, and the same workspace rule: this
    allocates on the Python side so the C++ never allocates device scratch. The
    buffer is plain scratch rather than the zeroed, self-resetting kind the mb
    path needs -- Phase A clears the counters it shares before anything reads
    them, and the small_n path uses no workspace at all.

    Ragged rows are served: rowEnds[row] is the exclusive end column and a row
    shorter than k emits min(k, row_len) indices followed by -1, the same
    padding top_k_per_row_prefill writes.

    rowStarts and rowEnds bound each row's window [rowStart, rowEnd); emitted
    indices are absolute column numbers, matching top_k_per_row_prefill
    (topk_per_row_kernels.cu:377 and :404).

    `values` is optional: pass an fp32 numRows*k tensor to also receive the
    selected scores, or None to skip the stores entirely (it is a template
    parameter on the four output kernels, not a runtime branch). Padded slots
    get -inf rather than 0, matching top_k_per_row_prefill, so a consumer that
    ranks these scores across ranks cannot have padding outrank a real
    negative logit.

    Every argument is validated here and a bad one raises ValueError. That is
    not belt-and-braces over the C++ checks, it is the only place the check can
    be survivable: AITER_CHECK calls std::abort() unless g_aiter_can_throw is
    set (csrc/include/aiter_hip_common.h), and only the aiter_safe_call ctypes
    bridge sets it, which this entry does not go through. Before this, passing
    k above the Phase C cap, or stride1 != 1, or a short workspace killed the
    caller's process with a message instead of raising. Measured with
    bench/stress_topk.py.

    None of the checks costs a device sync: every term is a scalar argument or
    a tensor attribute. rowStarts/rowEnds CONTENTS are deliberately not checked
    here -- they live in device memory, so validating them host-side would cost
    a D2H sync on every call. The kernel clamps them into [0, stride0] instead
    (RowExtents in csrc/topk_common.hip.hpp).

    `workspace` is optional: pass a buffer of at least
    topk_sampled_workspace_size(numRows, stride0, k) bytes to own it yourself, or
    leave it None to get the shared scratch buffer.

    Call topk_sampled_supports() first if you want to route around the shapes this
    declines (k above the Phase C LDS cap) rather than handle the exception."""
    if numRows <= 0:
        return  # matches the C++ entry, which returns before touching anything
    if stride1 != 1:
        raise ValueError(
            f"top_k_per_row_prefill_sampled: logits inner stride must be 1, got {stride1}"
        )
    if not _sampled_supports_cached(numRows, stride0, k):
        raise ValueError(
            f"top_k_per_row_prefill_sampled: unsupported shape (numRows={numRows} "
            f"stride0={stride0} k={k}); ask topk_sampled_supports() first"
        )
    if logits.dtype is not torch.float32:
        raise ValueError(
            f"top_k_per_row_prefill_sampled: logits must be fp32, got {logits.dtype}"
        )
    if indices.dtype is not torch.int32:
        raise ValueError(
            f"top_k_per_row_prefill_sampled: indices must be int32, got {indices.dtype}"
        )
    if indices.numel() < numRows * k:
        raise ValueError(
            f"top_k_per_row_prefill_sampled: indices holds {indices.numel()} entries, "
            f"needs numRows*k = {numRows * k}"
        )
    if values is not None:
        if values.dtype is not torch.float32:
            raise ValueError(
                f"top_k_per_row_prefill_sampled: values must be fp32, got {values.dtype}"
            )
        if values.numel() < numRows * k:
            raise ValueError(
                f"top_k_per_row_prefill_sampled: values holds {values.numel()} entries, "
                f"needs numRows*k = {numRows * k}"
            )
    if rowStarts.numel() < numRows or rowEnds.numel() < numRows:
        raise ValueError(
            f"top_k_per_row_prefill_sampled: rowStarts/rowEnds hold "
            f"{rowStarts.numel()}/{rowEnds.numel()} entries, need {numRows}"
        )
    size = _sampled_workspace_size_cached(numRows, stride0, k)
    if workspace is None:
        workspace = get_topk_scratch_workspace(logits.device, size)
    elif workspace.numel() * workspace.element_size() < size:
        raise ValueError(
            f"top_k_per_row_prefill_sampled: workspace is "
            f"{workspace.numel() * workspace.element_size()} B, needs {size} B"
        )
    return _top_k_per_row_prefill_sampled(
        logits,
        rowStarts,
        rowEnds,
        indices,
        values,
        numRows,
        stride0,
        stride1,
        k,
        workspace,
    )


@compile_ops("module_top_k_per_row", ffi_type="ctypes")
def top_k_per_row_prefill_fast(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    numRows: int,
    stride0: int,
    stride1: int,
) -> None: ...


@compile_ops("module_top_k_per_row", fc_name="top_k_per_row_decode", develop=True)
def _top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    workspace: torch.Tensor | None = None,
    stable: bool = False,
    values: torch.Tensor | None = None,
) -> None: ...


_FLYDSL_TOPK_DECODE_DISABLED = os.environ.get(
    "AITER_DISABLE_FLYDSL_TOPK_DECODE", "0"
) in ("1", "true", "True", "yes", "YES")

# Dispatch uses physical width because seq_lens is device-resident; padded rows
# with short effective lengths may therefore enter a long-row gate.
# (minimum width, maximum width, maximum rows), with inclusive bounds.
_FLYDSL_TOPK_DECODE_GATES = {
    "gfx942": {
        True: (
            (0, 20_000, 64),
            (131_072, None, 16),
        ),
        False: ((524_288, None, 32),),
    },
    "gfx950": {
        True: (
            (0, 20_000, 128),
            (32_768, 65_535, 16),
            (65_536, 131072, 32),
            (131072, 200_000, 48),
            (200_000, None, 64),
        ),
        False: ((131_072, None, 32),),
    },
}
_FLYDSL_TOPK_DECODE_KS = (512, 1024, 2048, 4096)


@functools.lru_cache(maxsize=128)
def _flydsl_topk_decode_shape_supported(
    arch: str,
    stable: bool,
    width: int,
    num_rows: int,
    k: int,
) -> bool:
    if k not in _FLYDSL_TOPK_DECODE_KS:
        return False
    return any(
        min_width <= width
        and (max_width is None or width <= max_width)
        and 0 < num_rows <= max_rows
        for min_width, max_width, max_rows in _FLYDSL_TOPK_DECODE_GATES[arch][stable]
    )


def _should_use_flydsl_topk_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    stable: bool,
    values: torch.Tensor | None = None,
) -> bool:
    if (
        _FLYDSL_TOPK_DECODE_DISABLED
        or not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
    ):
        return False

    arch = get_gfx()
    if arch not in _FLYDSL_TOPK_DECODE_GATES:
        return False

    if not _flydsl_topk_decode_shape_supported(
        arch,
        stable,
        logits.shape[1],
        num_rows,
        k,
    ):
        return False

    from .flydsl.topk.topk_per_row import is_flydsl_top_k_per_row_decode_supported

    return is_flydsl_top_k_per_row_decode_supported(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        stride0,
        stride1,
        k,
        values,
    )


def _hip_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    stable: bool,
    values: torch.Tensor | None,
) -> None:
    size = _ob_workspace_size_cached(num_rows, stride0, k, True)
    workspace = get_topk_scratch_workspace(logits.device, size)
    return _top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        stride0,
        stride1,
        k,
        workspace,
        stable,
        values,
    )


def top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    values: torch.Tensor | None = None,
) -> None:
    """Per-row top-k (decode). Always uses the one-block kernel; the scratch
    workspace is allocated + cached on the Python side and passed in, so the C++
    side never allocates device scratch.

    When stable=True, the deterministic ascending-ordered, smallest-index
    tie-break emit is used so every TP rank selects and orders an identical
    KV set.

    When `values` is given (float32, same shape as `indices`), each selected
    index's logit is written alongside it. Rows shorter than k pad the index
    with -1 and the score with -inf, so the padding sorts below every real
    candidate and a consumer that ranks these scores needs no extra mask."""
    if _should_use_flydsl_topk_decode(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k,
        stable,
        values,
    ):
        return flydsl_top_k_per_row_decode(
            logits,
            next_n,
            seqLens,
            indices,
            numRows,
            stride0,
            stride1,
            k,
            stable,
            values,
        )

    # FlyDSL one-block outperforms HIP one-block on the remaining decode cases.
    if not _FLYDSL_TOPK_DECODE_DISABLED:
        from .flydsl.topk.topk_per_row import (
            _is_flydsl_radix_topk_one_block_supported,
        )

        if _is_flydsl_radix_topk_one_block_supported(
            logits,
            None,
            seqLens,
            indices,
            values,
            numRows,
            stride0,
            stride1,
            k,
            is_decode=True,
            next_n=next_n,
        ):
            return flydsl_radix_topk_one_block_decode(
                logits,
                next_n,
                seqLens,
                indices,
                numRows,
                stride0,
                stride1,
                k,
                stable,
                values,
            )

    if values is not None:
        # The C++ side takes values.data_ptr() as a raw float* and writes k
        # entries per row through it, with no metadata of its own. A wrong
        # dtype or a short buffer is therefore silent memory corruption, not a
        # type error -- check here, where the tensor is still a torch object.
        if values.dtype != torch.float32:
            raise ValueError(f"values must be float32, got {values.dtype}")
        if values.shape != indices.shape:
            raise ValueError(
                f"values must match indices shape {tuple(indices.shape)}, "
                f"got {tuple(values.shape)}"
            )
        if not values.is_contiguous():
            raise ValueError("values must be contiguous")
        if values.device != indices.device:
            raise ValueError(
                f"values on {values.device} but indices on {indices.device}"
            )

    return _hip_top_k_per_row_decode(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k,
        stable,
        values,
    )


def flydsl_dcp_topk_merge(
    gathered_scores: torch.Tensor,
    local_idx: torch.Tensor,
    block_table: torch.Tensor,
    out_kv_indices: torch.Tensor,
    out_kv_indptr: torch.Tensor,
    owned_counts: torch.Tensor,
    staging: torch.Tensor,
    dcp_rank: int,
    world_size: int,
    topk_tokens: int,
    page_size: int,
) -> None:
    """DCP decode top-k merge: emit this rank's owned KV slots, packed.

    Allocates no device scratch, so it is safe inside a captured CUDAGraph;
    the first call for a new shape JIT-compiles, so warm up before capturing.
    """
    from .flydsl.dcp_topk_merge import (
        flydsl_dcp_topk_merge as _impl,
    )

    return _impl(
        gathered_scores,
        local_idx,
        block_table,
        out_kv_indices,
        out_kv_indptr,
        owned_counts,
        staging,
        dcp_rank,
        world_size,
        topk_tokens,
        page_size,
    )


def flydsl_radix_topk_one_block_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
) -> None:
    """Prefill wrapper with the same argument order as HIP top_k_per_row_prefill."""
    from .flydsl.topk.topk_per_row import flydsl_radix_topk_one_block

    return flydsl_radix_topk_one_block(
        logits,
        row_starts,
        row_ends,
        indices,
        values,
        num_rows,
        stride0,
        stride1,
        k,
        stable,
        is_decode=False,
        next_n=1,
    )


def flydsl_radix_topk_one_block_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    values: torch.Tensor | None = None,
) -> None:
    """Decode wrapper with the same argument order as HIP top_k_per_row_decode."""
    from .flydsl.topk.topk_per_row import flydsl_radix_topk_one_block

    return flydsl_radix_topk_one_block(
        logits,
        seq_lens,
        seq_lens,
        indices,
        values,
        num_rows,
        stride0,
        stride1,
        k,
        stable,
        is_decode=True,
        next_n=next_n,
    )


def flydsl_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    values: torch.Tensor | None = None,
) -> None:
    """FlyDSL per-row decode TopK with the same call shape as the HIP interface.

    This path is optimized for long-context decode, where its multi-CTA radix
    selection typically outperforms the HIP one-block implementation.
    """
    from .flydsl.topk.topk_per_row import (
        flydsl_top_k_per_row_decode as _flydsl_top_k_per_row_decode,
    )

    return _flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k,
        stable,
        values,
    )


@compile_ops("module_top_k_per_row", ffi_type="ctypes")
def top_k_per_row_decode_fast(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
) -> None: ...

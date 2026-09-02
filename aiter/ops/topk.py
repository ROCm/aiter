# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import functools
import os

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_cu_num, get_gfx
from ..utility import dtypes


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
    the workspace for the ob path in that case."""
    if not stable and topk_use_mulblocks(numRows, stride0):
        size = topk_mb_workspace_size(numRows, stride0, k, False)
        workspace = get_topk_mb_workspace(logits.device, size)
    else:
        size = topk_ob_workspace_size(numRows, stride0, k, False)
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
            (65_536, None, 32),
        ),
        False: ((131_072, None, 32),),
    },
}
_FLYDSL_TOPK_DECODE_KS = (512, 1024, 2048, 4096)


@functools.lru_cache(maxsize=1)
def _flydsl_topk_decode_available() -> bool:
    """Whether the optional FlyDSL package is available on this device."""
    try:
        from .flydsl.utils import is_flydsl_available

        return is_flydsl_available()
    except (ImportError, OSError, RuntimeError):
        return False


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
    if arch not in _FLYDSL_TOPK_DECODE_GATES or not _flydsl_topk_decode_available():
        return False

    if not _flydsl_topk_decode_shape_supported(
        arch,
        stable,
        logits.shape[1],
        num_rows,
        k,
    ):
        return False

    from .flydsl.topk_per_row import is_flydsl_top_k_per_row_decode_supported

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
    size = topk_ob_workspace_size(num_rows, stride0, k, True)
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
    from .flydsl.topk_per_row import (
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

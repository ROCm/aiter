# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import functools
import os
from typing import NamedTuple

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..utility import dtypes


def is_flydsl_available() -> bool:
    try:
        from .flydsl.utils import is_flydsl_available as _is_flydsl_available
    except ImportError:
        return False
    return _is_flydsl_available()


# DEPRECATED: low-level binding kept for backward compatibility only.
# Will be removed once all callers have migrated to topk_gating() below.
# New code should use topk_gating(), which:
#   - accepts an Optional[Tensor] correction_bias (None => no bias)
#   - validates score_func string
#   - exposes the same C++ kernel under a more accurate name
@compile_ops("module_moe_topk", develop=True)
def topk_softplus(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,
    score_func: str = "sqrtsoftplus",
) -> None: ...


_VALID_SCORE_FUNCS = {"sqrtsoftplus", "sigmoid", "softmax"}


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
        correction_bias: optional bias tensor, pass None for no bias.

    Note: softmax is already normalized, so renorm is forced off.
    """
    assert (
        score_func in _VALID_SCORE_FUNCS
    ), f"Unknown score_func '{score_func}', expected one of {_VALID_SCORE_FUNCS}"
    if correction_bias is None:
        # Match gating dtype/device so dispatch picks DTYPE_B == DTYPE_I,
        # avoiding extra kernel template instantiations.
        correction_bias = torch.empty(
            0, dtype=gating_output.dtype, device=gating_output.device
        )
    if score_func == "softmax":
        need_renorm = False
    topk_softplus(
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        need_renorm,
        routed_scaling_factor,
        score_func,
    )


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


@compile_ops("module_top_k_per_row", fc_name="top_k_per_row_prefill")
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
) -> None: ...


@compile_ops("module_top_k_per_row")
def topk_mb_workspace_size(
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
) -> None:
    """Per-row top-k (prefill). The multi-block path runs on a persistent,
    zero-initialized workspace (memset-free; see get_topk_mb_workspace); the
    one-block path allocates its own scratch internally."""
    workspace = None
    if topk_use_mulblocks(numRows, stride0):
        size = topk_mb_workspace_size(numRows, stride0, k, False)
        workspace = get_topk_mb_workspace(logits.device, size)
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


@compile_ops("module_top_k_per_row", fc_name="top_k_per_row_decode")
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
) -> None: ...


_TRUTHY_ENV = ("1", "true", "True", "yes", "YES")


# Per-arch gates for the FlyDSL tiered decode kernel. Each arch carries its whole
# admission window -- min padded width, max rows, the k set it has wins for, and any
# individual rows it must exclude -- because the safe window does not transfer
# between archs: the kernel's grid-trim, batch-cap, row-proportional and early-stop
# tuning is gfx950-only, so gfx942 runs a frozen configuration and keeps the lowest
# width it has evidence for (65K and up).
#
# Note what the width gate can and cannot see. Callers hand us a score buffer sized
# to the model's max context, not to the request's -- vLLM's sparse indexer builds
# logits as (batch * next_n, max_model_len) -- and the real per-row lengths live in
# seqLens, on the device, where reading them would cost a sync. So this asks "is
# this a long-context model", not "is this a long request", and it is deliberately
# the former: a padded buffer is itself what puts the HIP kernel behind, since HIP
# slows down with the buffer width while the FlyDSL kernel tracks seqLens.
#
# gfx950's window is the largest that loses no cell in the eager width-sweep, which
# is the regime vLLM pays whenever a decode batch overflows its CUDA-graph capture
# limit and falls back to eager submission; every captured cell only does better.
# Below 131072 the host submit path (~19us) sinks short-context cells the kernel
# would otherwise win; k=512 still carries losses at full width, so only k=2048
# ships; and rows==2 is the one admitted row that stays behind at width even as
# rows 1 and 4-16 pull ahead, so it is carved out by hand.
class _DecodeGate(NamedTuple):
    min_width: int
    max_rows: int
    ks: frozenset
    excluded_rows: frozenset = frozenset()


_FLYDSL_TOPK_DECODE_GATES = {
    "gfx950": _DecodeGate(131072, 16, frozenset({2048}), frozenset({2})),
    "gfx942": _DecodeGate(65536, 8, frozenset({512, 2048})),
}

# Narrows the table above, e.g. AITER_FLYDSL_TOPK_ARCHS=gfx950 leaves gfx942 on
# HIP. Listing an arch that has no row in the table does not enable it: adding an
# arch means measuring it and giving it thresholds.
_FLYDSL_TOPK_DECODE_ARCHS = frozenset(
    os.environ.get("AITER_FLYDSL_TOPK_ARCHS", " ".join(_FLYDSL_TOPK_DECODE_GATES))
    .replace(",", " ")
    .split()
)

# Escape hatch: route every shape back to HIP without a code change.
_FLYDSL_TOPK_DECODE_DISABLED = (
    os.environ.get("AITER_DISABLE_FLYDSL_TOPK_DECODE", "0") in _TRUTHY_ENV
)

# Opt-in dispatch counters so a benchmark can prove which kernel actually ran
# (a serving A/B is worthless if the FlyDSL path was never reached). Off by
# default: the hot path stays free of the two increments below.
_FLYDSL_TOPK_COUNT = os.environ.get("AITER_FLYDSL_TOPK_COUNT", "0") in _TRUTHY_ENV
topk_decode_dispatch_counts = {"flydsl": 0, "hip": 0}

if _FLYDSL_TOPK_COUNT:
    import atexit

    def _dump_topk_decode_dispatch_counts():
        # Each serving worker prints its own tally at shutdown, so a benchmark
        # log proves whether the FlyDSL path was reached (flydsl>0) or the run
        # silently stayed on HIP.
        print(
            f"[aiter] topk_decode_dispatch_counts pid={os.getpid()} "
            f"{topk_decode_dispatch_counts}",
            flush=True,
        )

    atexit.register(_dump_topk_decode_dispatch_counts)

# Sweep overrides, applied to every arch so a benchmark can walk a threshold
# without editing the table. Env is read once at import (as elsewhere in aiter,
# e.g. rotary_embedding.py) because this gate runs on every decode step; tests
# patch _FLYDSL_TOPK_DECODE_GATES directly instead of setting env late.
_FLYDSL_TOPK_MIN_WIDTH_ENV = os.environ.get("AITER_FLYDSL_TOPK_MIN_WIDTH")
_FLYDSL_TOPK_MAX_ROWS_ENV = os.environ.get("AITER_FLYDSL_TOPK_MAX_ROWS")
if _FLYDSL_TOPK_MIN_WIDTH_ENV or _FLYDSL_TOPK_MAX_ROWS_ENV:
    _FLYDSL_TOPK_DECODE_GATES = {
        _arch: _gate._replace(
            min_width=(
                int(_FLYDSL_TOPK_MIN_WIDTH_ENV)
                if _FLYDSL_TOPK_MIN_WIDTH_ENV
                else _gate.min_width
            ),
            max_rows=(
                int(_FLYDSL_TOPK_MAX_ROWS_ENV)
                if _FLYDSL_TOPK_MAX_ROWS_ENV
                else _gate.max_rows
            ),
        )
        for _arch, _gate in _FLYDSL_TOPK_DECODE_GATES.items()
    }


def _should_use_flydsl_decode(
    logits: torch.Tensor,
    next_n: int,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int,
) -> bool:
    """Whether decode top-k should take the FlyDSL tiered kernel instead of HIP.

    Never reads ``seqLens``: it lives on the device, so inspecting it would
    synchronize on every decode step and give back more than the kernel saves.
    The padded width ``logits.shape[1]`` stands in for it, which is also what the
    kernel's own tier selection keys on; rows whose real length is shorter are
    handled inside the kernel.

    Checks run cheapest first. Everything up to the stride comparisons is a host
    value test, while ``is_flydsl_available()`` can pull in the whole flydsl
    kernel package, so it runs only once an arch has already claimed the shape.
    """
    if _FLYDSL_TOPK_DECODE_DISABLED:
        return False

    arch = get_gfx()
    if arch not in _FLYDSL_TOPK_DECODE_ARCHS:
        return False
    gate = _FLYDSL_TOPK_DECODE_GATES.get(arch)
    if gate is None:
        return False

    if numRows > gate.max_rows or numRows in gate.excluded_rows:
        return False
    if k not in gate.ks:
        return False
    if logits.ndim != 2 or logits.shape[1] < gate.min_width:
        return False

    # HIP drops stride1 entirely and never validates next_n, so a call that works
    # there today can violate FlyDSL's contract, which raises rather than falling
    # back. Screen those here so the fallback stays a routing decision.
    if stride1 != 1 or next_n < 1 or stride0 != logits.stride(0):
        return False

    return is_flydsl_available()


def _is_stream_capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


# torch.cuda.current_stream() builds a Python Stream wrapper and measures ~1.9 us
# here, which is most of what this workspace lookup is allowed to cost. Inductor's
# generated launchers read the raw pointer instead (~0.07 us), and a cache key is
# all we need it for. Fall back if a torch build ever drops the private entry.
_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _stream_key(device: torch.device) -> int:
    if _current_raw_stream is not None:
        return _current_raw_stream(device.index)
    return torch.cuda.current_stream(device).cuda_stream


@functools.lru_cache(maxsize=64)
def _flydsl_topk_workspace_alloc(numRows: int, max_model_len: int) -> int:
    """Workspace element count for this shape, rounded up to a power of two.

    Memoized because the FlyDSL sizing helper rebuilds the whole kernel config on
    every call (~5.7 us measured), which dwarfs the allocation it exists to avoid.
    """
    from .flydsl.topk_per_row_decode import (
        flydsl_top_k_per_row_decode_workspace_size,
    )

    size = flydsl_top_k_per_row_decode_workspace_size(numRows, max_model_len)
    if size <= 0:
        return 0
    return 1 if size <= 1 else 1 << (int(size) - 1).bit_length()


@functools.lru_cache(maxsize=16)
def _get_flydsl_topk_workspace_keyed(
    device: torch.device, stream_id: int, size: int
) -> torch.Tensor:
    return torch.zeros(size, dtype=torch.int32, device=device)


def _get_flydsl_topk_workspace(
    device: torch.device, numRows: int, max_model_len: int
) -> torch.Tensor | None:
    """A cached int32 workspace for the FlyDSL tiered decode path.

    Keyed like get_topk_mb_workspace above -- (device, stream, size rounded up to
    a power of two) -- so concurrent streams never share one buffer's cross-block
    counters, and nearby batch sizes collapse onto a few allocations instead of
    one per exact shape. The kernel wants 24.8 KB per row and the dispatcher gates
    rows at 16, which comes to five buckets and 992 KB per stream at most.

    Handing the launcher a ready buffer saves ~2.4 us per call, so everything on
    the way to it is memoized down to lookups; the whole function measures ~1 us.

    Returns None while a graph capture is in flight. An allocation there comes
    from the graph's private memory pool, and a cache that outlives the capture
    would go on handing that tensor to eager calls; tuned_gemm.py skips its own
    workspace warming under capture for the same reason. The FlyDSL launcher then
    allocates a per-call temporary, which stays inside the capture.
    """
    if _is_stream_capturing():
        return None

    alloc = _flydsl_topk_workspace_alloc(numRows, max_model_len)
    if alloc <= 0:
        return None
    return _get_flydsl_topk_workspace_keyed(device, _stream_key(device), alloc)


def clear_flydsl_topk_decode_workspace_cache() -> None:
    """Drop the workspaces cached by _get_flydsl_topk_workspace.

    An lru_cache holds them, so torch.cuda.empty_cache() cannot reclaim the
    memory on its own; call this first when a caller needs it back.
    """
    _get_flydsl_topk_workspace_keyed.cache_clear()
    _flydsl_topk_workspace_alloc.cache_clear()


def top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    workspace: torch.Tensor | None = None,
) -> None:
    """Per-row top-k (decode), writing k indices per row.

    Takes the FlyDSL tiered kernel on the archs and shapes where it beats HIP and
    the HIP one-block kernel everywhere else; see _should_use_flydsl_decode for
    the gates and AITER_DISABLE_FLYDSL_TOPK_DECODE to force HIP. Both kernels
    return the indices as an unordered set.

    ``workspace`` lets a caller that already owns a buffer (a serving framework
    reserving device memory up front, say) hand it over instead of paying an
    allocation per call. It is an int32 tensor sized by
    ``flydsl_top_k_per_row_decode_workspace_size``, and it only reaches the FlyDSL
    path: the HIP decode kernel allocates its own scratch and ignores the
    argument, so a shape that falls back drops the buffer on the floor rather than
    misreading it.
    """
    if _should_use_flydsl_decode(logits, next_n, numRows, stride0, stride1, k):
        if _FLYDSL_TOPK_COUNT:
            topk_decode_dispatch_counts["flydsl"] += 1
        from .flydsl.topk_per_row_decode import flydsl_top_k_per_row_decode

        if workspace is None:
            workspace = _get_flydsl_topk_workspace(
                logits.device, numRows, logits.shape[1]
            )
        return flydsl_top_k_per_row_decode(
            logits,
            next_n,
            seqLens,
            indices,
            numRows,
            stride0,
            stride1,
            k,
            ordered=False,
            workspace=workspace,
        )

    if _FLYDSL_TOPK_COUNT:
        topk_decode_dispatch_counts["hip"] += 1
    # Decode always takes the ob path (see topk_per_row_kernels.cu), and the C++
    # side ignores the workspace argument there.
    # The original mb dispatch is commented out below for reference:
    #   workspace = None
    #   if topk_use_mulblocks(numRows, stride0):
    #       size = topk_mb_workspace_size(numRows, stride0, k, True)
    #       workspace = get_topk_mb_workspace(logits.device, size)
    return _top_k_per_row_decode(
        logits, next_n, seqLens, indices, numRows, stride0, stride1, k, None
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

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
) -> None: ...


_TRUTHY_ENV = ("1", "true", "True", "yes", "YES")


# Per-arch admission windows: a row cap per width band, plus the k set that wins.
# A window does not transfer between archs -- the grid-trim, batch-cap and
# early-stop tuning is gfx950-only and gfx942 runs a frozen configuration.
#
# Width asks "is this a long-context model", not "is this a long request", and it
# has to: callers size the score buffer to the model's max context (vLLM's sparse
# indexer builds logits as (batch * next_n, max_model_len)), the real per-row
# lengths live in seqLens on the device where reading them would cost a sync, and
# under CUDA-graph capture the request length does not exist yet. Width only
# bounds the request, so this kernel wins across the long half of what one width
# covers and loses on the short half; the window is a bet that a long-context
# deployment actually serves long requests.
#
# The row cap varies with width because the two interact: a wider buffer means
# more work per row, so the multi-block kernel's cooperation keeps paying for more
# concurrent rows before the batch alone fills the machine. A single cap is a
# compromise between two different crossings and loses at both ends.
#
# Every threshold is read off graph replay rather than eager, and sits below the
# measured sign change rather than on it. The SILOTIGER-699 gate investigation
# holds the per-cell numbers and the reasoning behind every one.
class _DecodeGate(NamedTuple):
    # (min padded width, max rows) bands, widest first. A call is capped by the
    # first band whose width it reaches, and one that reaches none is refused.
    row_caps: tuple
    ks: frozenset


_ALL_DECODE_KS = frozenset({256, 512, 1024, 2048})

_FLYDSL_TOPK_DECODE_GATES = {
    "gfx950": _DecodeGate(((163840, 15), (131072, 9)), _ALL_DECODE_KS),
    "gfx942": _DecodeGate(((163840, 18), (131072, 11)), _ALL_DECODE_KS),
}


def _decode_max_rows(gate: _DecodeGate, width: int) -> int:
    """Rows this gate admits at `width`, or 0 when the width is below every band."""
    for min_width, max_rows in gate.row_caps:
        if width >= min_width:
            return max_rows
    return 0


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
#
# MAX_ROWS replaces the cap in every band. MIN_WIDTH additionally flattens the
# table to one band, since a sweep that moves the width is asking for a single
# rectangular window rather than the shipped staircase.
_FLYDSL_TOPK_MIN_WIDTH_ENV = os.environ.get("AITER_FLYDSL_TOPK_MIN_WIDTH")
_FLYDSL_TOPK_MAX_ROWS_ENV = os.environ.get("AITER_FLYDSL_TOPK_MAX_ROWS")
if _FLYDSL_TOPK_MIN_WIDTH_ENV or _FLYDSL_TOPK_MAX_ROWS_ENV:

    def _override_row_caps(caps: tuple) -> tuple:
        rows = (
            int(_FLYDSL_TOPK_MAX_ROWS_ENV)
            if _FLYDSL_TOPK_MAX_ROWS_ENV
            else max(_rows for _, _rows in caps)
        )
        if _FLYDSL_TOPK_MIN_WIDTH_ENV:
            return ((int(_FLYDSL_TOPK_MIN_WIDTH_ENV), rows),)
        return tuple((_width, rows) for _width, _ in caps)

    _FLYDSL_TOPK_DECODE_GATES = {
        _arch: _gate._replace(row_caps=_override_row_caps(_gate.row_caps))
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

    if k not in gate.ks:
        return False
    if logits.ndim != 2:
        return False
    # Width first: it selects which row cap applies.
    if numRows > _decode_max_rows(gate, logits.shape[1]):
        return False

    # HIP drops stride1 and never validates next_n, so a call that works there can
    # violate FlyDSL's contract, which raises rather than falling back. Screen those
    # here to keep the fallback a routing decision. Strides are checked against the
    # tensor, not the declared value: a caller claiming stride1 == 1 for a strided
    # buffer is exactly the case HIP accepts and misreads.
    if (
        stride1 != 1
        or next_n < 1
        or stride0 != logits.stride(0)
        or logits.stride(1) != 1
    ):
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


@functools.lru_cache(maxsize=16)
def _flydsl_decode_cu_count(device: torch.device) -> int:
    """CU count of `device`, cached because it sits on the per-call decode path."""
    from .flydsl.topk_per_row_decode import decode_cu_count

    return decode_cu_count(device)


@functools.lru_cache(maxsize=256)
def _flydsl_topk_workspace_alloc(
    numRows: int, max_model_len: int, cu_count: int | None = None
) -> int:
    """Workspace element count for this shape, rounded up to a power of two.

    Memoized because the FlyDSL sizing helper rebuilds the whole kernel config on
    every call (~5.7 us measured), which dwarfs the allocation it exists to avoid.
    """
    from .flydsl.topk_per_row_decode import (
        flydsl_top_k_per_row_decode_workspace_size,
    )

    size = flydsl_top_k_per_row_decode_workspace_size(numRows, max_model_len, cu_count)
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

    alloc = _flydsl_topk_workspace_alloc(
        numRows, max_model_len, _flydsl_decode_cu_count(device)
    )
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
    stable: bool = False,
    *,
    workspace: torch.Tensor | None = None,
) -> None:
    """Per-row top-k (decode), writing k indices per row.

    Takes the FlyDSL tiered kernel on the archs and shapes where it beats HIP and
    the HIP one-block kernel everywhere else; see _should_use_flydsl_decode for
    the gates and AITER_DISABLE_FLYDSL_TOPK_DECODE to force HIP. Both kernels
    return the indices as an unordered set.

    stable=True asks for the deterministic ascending-ordered, smallest-index
    tie-break emit so every TP rank selects and orders an identical KV set. Only
    the HIP kernel emits that order -- FlyDSL returns an unordered set and
    rejects ordered output outright -- so stable disqualifies the FlyDSL path
    rather than silently handing back an unordered answer.

    ``workspace`` lets a caller that already owns a buffer (a serving framework
    reserving device memory up front, say) hand it over instead of paying an
    allocation per call. It is an int32 tensor sized by
    ``flydsl_top_k_per_row_decode_workspace_size``, and it reaches the FlyDSL path
    only. The HIP one-block kernel needs scratch of its own -- a different dtype,
    a different size, and a layout it computes itself -- so a shape that falls
    back drops the caller's buffer and gets a freshly sized one rather than
    misreading it.

    Keyword-only on purpose. ``stable`` has been the ninth positional parameter
    since before a workspace existed, and the C++ entry orders it the other way
    (k, workspace, stable), so admitting ``workspace`` positionally here would let
    ``(..., k, True)`` bind True to the buffer and silently drop the stable
    request -- on the HIP path, which ignores ``workspace`` entirely, with nothing
    to raise on.
    """
    # The gate screens every layout contract but not these two, so without them a
    # wrong dtype raises on a gated shape and is silently reinterpreted by HIP's
    # hard cast on every other one. FlyDSL re-checks; HIP checks nothing.
    if (
        logits.dtype is not torch.float32
        or seqLens.dtype is not torch.int32
        or indices.dtype is not torch.int32
    ):
        raise TypeError(
            "decode top-k needs float32 logits with int32 seqLens/indices, got "
            f"{logits.dtype}, {seqLens.dtype}, {indices.dtype}"
        )
    # Index comparison, not torch.device: building the three wrappers costs twice
    # what reading the indices does, and a CPU tensor answers -1.
    device = logits.get_device()
    if (
        not logits.is_cuda
        or seqLens.get_device() != device
        or indices.get_device() != device
    ):
        raise ValueError(
            "logits, seqLens, and indices must share one CUDA/ROCm device, got "
            f"{logits.device}, {seqLens.device}, {indices.device}"
        )

    # Neither kernel reads a column stride: FlyDSL raises on one, HIP ignores it and
    # walks the dense buffer the view sits in, returning the Top-K of the neighbouring
    # elements with nothing raised. Densify up front so the gate routes on shape alone
    # and both kernels see what the caller asked about.
    if stride1 != 1 or logits.stride(1) != 1:
        logits = logits.contiguous()
        stride0, stride1 = logits.stride()

    if not stable and _should_use_flydsl_decode(
        logits, next_n, numRows, stride0, stride1, k
    ):
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
    # Decode always takes the ob path (see topk_per_row_kernels.cu), which now
    # requires the host to hand it scratch -- the C++ side no longer allocates any.
    # Size it here rather than forwarding ``workspace``: that one is the FlyDSL
    # buffer and has neither the dtype nor the layout this kernel reads.
    ob_workspace = get_topk_scratch_workspace(
        logits.device, topk_ob_workspace_size(numRows, stride0, k, True)
    )
    return _top_k_per_row_decode(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k,
        ob_workspace,
        stable,
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

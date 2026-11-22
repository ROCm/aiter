# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A caller's precomputed MoE sort, handed to the next fused_moe call.

fused_moe, fhmoe and the comm-fused runtime all consume it, so it lives here
rather than in fused_moe.py, which the other two can only import lazily.
"""

import functools
import threading

import torch

from aiter import dtypes, logger


def aux_sort_capacity(M, topk, num_experts, block_m):
    """Rows a BM-block MoE sort must hold: every route, plus block_m - 1 padding
    rows for each expert that can be active."""
    routes = M * topk
    active = min(num_experts, routes)
    return (routes + active * (block_m - 1) + block_m - 1) // block_m * block_m


def bytes_overlap(a, b):
    return (
        a.data_ptr() < b.data_ptr() + b.nbytes
        and b.data_ptr() < a.data_ptr() + a.nbytes
    )


# Optional sort metadata from a caller's fused route+sort kernel (e.g.
# SGLang's radix-4 router), registered on the calling thread immediately before
# fused_moe. CUDA-graph replay does not re-enter Python, so a captured fused_moe
# keeps whichever sort launches it was captured with.
_PRESORTED_MOE = threading.local()
# The block size the aux-sort (output_aux) pipelines accept a presort at.
PRESORTED_MOE_BLOCK_M = 16
# The block sizes the FlyDSL two-stage pipelines that read moe_sorting's standard
# layout accept a presort at. Other tile sizes re-sort.
PRESORTED_MOE_BLOCK_MS = (16, 32)
# Reasons a registration was ignored, so each one is logged only once.
_PRESORTED_MOE_IGNORED = set()
_PRESORTED_MOE_STD_KEYS = (
    "sorted_token_ids",
    "sorted_weights",
    "sorted_expert_ids",
    "num_valid_ids",
)
# The aux-sort pipelines also read these two.
_PRESORTED_MOE_SORT_KEYS = _PRESORTED_MOE_STD_KEYS + ("m_indices", "reverse_sorted")


# Under torch.compile, dynamo defers this thread-local store until the graph has
# run, so the compiled fused_moe ops would read the previous step's registration.
# Running it eagerly (a graph break) keeps it ordered before its consumer.
@torch.compiler.disable
def register_presorted_moe(aux: dict | None) -> None:
    """Hand the next fused_moe call on this thread a precomputed sort.

    ``aux`` holds the sort outputs (sorted_token_ids, sorted_weights,
    sorted_expert_ids, num_valid_ids, m_indices, reverse_sorted) and the keys
    they were computed for (topk_ids_ptr, topk_weights_ptr, M, block_m,
    num_experts, topk). The sort outputs must use aiter's sort layout for
    block_m: contiguous tensors on topk_ids' device, fp32 sorted_weights and
    int32 everything else, sorted_expert_ids with one entry per block_m sorted
    rows, sorted_weights and m_indices at least as long as sorted_token_ids,
    reverse_sorted with M * topk entries and num_valid_ids with 2. The rows are
    what moe_sorting(block_size=block_m) writes: each expert's routes in token
    order, ``(slot << 24) | token``, padded to a multiple of block_m with
    ``(topk << 24) | M`` (or plain M, as aiter's BM=16 aux sort writes) and
    weight 0. A pipeline without the aux sort ignores m_indices and
    reverse_sorted, which may then be omitted.
    A registration that does not match the call or breaks this layout is
    ignored (logged once per reason), and fused_moe runs its own sort.

    Set moe_buf_zeroed and moe_buf_ptr when the caller already zeroed the buffer
    it will pass as ``output``; fused_moe then skips its own zeroing and launches
    only the two GEMMs. Without ``output`` fused_moe always zeroes the buffer it
    allocates, so the skipped sort launch is replaced by a fill. That
    decision is made in Python, so under CUDA graphs it is frozen at capture:
    the caller's zeroing must run on the current stream before fused_moe on
    every call and be captured in the same graph (e.g. inside the caller's
    route+sort kernel), or replays accumulate into a stale buffer.

    The next fused_moe consumes the registration on every path, whether or not
    it matches or is used, so a stale registration never applies to a later
    call. A caller that registers and then does not call fused_moe must clear it
    with ``register_presorted_moe(None)``. It is only used when the call runs
    without expert_mask or num_local_tokens, on an aux-sort pipeline at
    block_m == PRESORTED_MOE_BLOCK_M or a FlyDSL two-stage pipeline (only the
    tested stage 1 families; see presorted_moe_block_m) at a block_m in
    PRESORTED_MOE_BLOCK_MS; other calls run fused_moe's own sort.

    Under torch.compile this call runs eagerly, so it stays ordered before the
    fused_moe that consumes it; a ``fullgraph=True`` compile rejects it instead.
    """
    _PRESORTED_MOE.aux = aux


def pop_presorted_moe():
    aux = getattr(_PRESORTED_MOE, "aux", None)
    _PRESORTED_MOE.aux = None
    return aux


def presorted_moe_ignored(reason):
    # A caller that, e.g., omits a key would otherwise lose the presort on every
    # call without a trace.
    if reason not in _PRESORTED_MOE_IGNORED:
        _PRESORTED_MOE_IGNORED.add(reason)
        logger.warning(
            f"[fused_moe] registered presort ignored ({reason}); "
            "fused_moe runs its own sort"
        )


def match_presorted_moe(
    aux,
    topk_ids,
    topk_weight,
    *,
    block_m,
    num_experts,
    topk,
    aux_sort=True,
    stage_block_m=None,
    expert_mask=None,
    num_local_tokens=None,
    output=None,
):
    """Return ``aux`` if it can replace fused_moe's own sort for this call, else None.

    ``block_m`` is the selected kernels' tile size; ``stage_block_m`` is the
    block size fused_moe hands its sort and stages (a caller's block_size_M),
    which defaults to ``block_m``. ``aux_sort`` says whether the selected
    pipeline reads the aux sort (m_indices, reverse_sorted). Both block sizes
    must be equal and supported for that pipeline: PRESORTED_MOE_BLOCK_M with
    the aux sort, one of PRESORTED_MOE_BLOCK_MS without it.
    """
    if not aux:
        return None
    supported = (PRESORTED_MOE_BLOCK_M,) if aux_sort else PRESORTED_MOE_BLOCK_MS
    if int(block_m) not in supported:
        presorted_moe_ignored(
            f"block_m={block_m} on the {'aux' if aux_sort else 'standard'}-sort path"
        )
        return None
    # The stages index sorted_expert_ids by stage_block_m rows, so a caller's
    # block_size_M that overrides the kernels' tile would misread the sort.
    if stage_block_m is not None and int(stage_block_m) != int(block_m):
        presorted_moe_ignored(f"block_size_M={stage_block_m} != kernel block_m")
        return None
    # The presort has no expert mask applied and covers every row of topk_ids;
    # moe_sorting honors both, so a masked or row-limited call must re-sort.
    if expert_mask is not None or num_local_tokens is not None:
        presorted_moe_ignored("expert_mask or num_local_tokens is set")
        return None
    for key, expected in (
        ("topk_ids_ptr", topk_ids.data_ptr()),
        # sorted_weights were gathered from this tensor.
        ("topk_weights_ptr", topk_weight.data_ptr()),
        ("M", topk_ids.shape[0]),
        ("block_m", block_m),
        ("num_experts", num_experts),
        ("topk", topk),
    ):
        if key not in aux:
            presorted_moe_ignored(f"no {key}")
            return None
        if int(aux[key]) != int(expected):
            presorted_moe_ignored(f"{key} does not match the call")
            return None
    keys = _PRESORTED_MOE_SORT_KEYS if aux_sort else _PRESORTED_MOE_STD_KEYS
    if not _presorted_layout_ok(aux, keys, topk_ids, topk, num_experts, block_m):
        presorted_moe_ignored(f"sort tensors break the BM={block_m} sort layout")
        return None
    # The stages zero or write output while still reading the sort tensors.
    if output is not None and any(bytes_overlap(output, aux[k]) for k in keys):
        presorted_moe_ignored("output overlaps the sort tensors")
        return None
    return aux


def _presorted_layout_ok(aux, keys, topk_ids, topk, num_experts, block_m):
    """Whether the sort tensors in ``keys`` match the layout the stages read.

    The stages size their intermediates and grids from sorted_token_ids and
    sorted_expert_ids (max_sorted rows, one expert per block_m-row block) and
    index sorted_weights and m_indices by sorted row and reverse_sorted by
    route. num_valid_ids[0] (padded rows) is read by every stage; [1] (valid
    tokens) only by the valid-token-count GEMM2, which this path never enables,
    but every aiter sort emits both. ``keys`` names the four standard tensors,
    plus m_indices and reverse_sorted for an aux-sort pipeline.
    """
    if any(k not in aux for k in keys):
        return False
    device = topk_ids.device
    i32 = dtypes.i32
    for k in keys:
        t, dtype = aux[k], dtypes.fp32 if k == "sorted_weights" else i32
        if (
            not isinstance(t, torch.Tensor)
            # The stages take lengths from shape[0], so a reshaped buffer would
            # be sized by its first dimension only.
            or t.ndim != 1
            or t.dtype != dtype
            or t.device != device
            or not t.is_contiguous()
        ):
            return False
    max_sorted = aux["sorted_token_ids"].numel()
    # Worst case for a BM-block sort: every route plus BM - 1 padding rows per
    # active expert. Smaller cannot hold all routings.
    capacity = aux_sort_capacity(topk_ids.shape[0], topk, num_experts, block_m)
    if not (
        max_sorted >= capacity
        and aux["sorted_expert_ids"].numel() * block_m == max_sorted
        and aux["sorted_weights"].numel() >= max_sorted
        and aux["num_valid_ids"].numel() >= 2
    ):
        return False
    if "m_indices" not in keys:
        return True
    return (
        aux["m_indices"].numel() >= max_sorted
        and aux["reverse_sorted"].numel() == topk_ids.shape[0] * topk
    )


def drop_presort_on_error(fn):
    """Clear the registration if ``fn`` raises.

    An entry point that fails before _fused_moe_impl consumes the registration
    would otherwise leave it live for a later call on reused routing buffers.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BaseException:
            pop_presorted_moe()
            raise

    return wrapper


def drop_presort_after(fn):
    """Clear the registration when ``fn`` returns or raises.

    The registration is for the next fused_moe call, and this call is that
    call. A kernel plugin that serves it returns before _fused_moe_impl
    consumes the registration, which would otherwise apply to the next native
    call on reused routing buffers.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            pop_presorted_moe()

    return wrapper

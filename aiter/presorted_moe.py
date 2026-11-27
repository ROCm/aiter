# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A caller's precomputed BM=16 MoE sort, handed to the next fused_moe call.

fused_moe, fhmoe and the comm-fused runtime all consume it, so it lives here
rather than in fused_moe.py, which the other two can only import lazily.
"""

import functools
import threading
import weakref

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


# Optional BM=16 sort metadata from a caller's fused route+sort kernel (e.g.
# SGLang's radix-4 router), registered on the calling thread immediately before
# fused_moe. CUDA-graph replay does not re-enter Python, so a captured fused_moe
# keeps whichever sort launches it was captured with.
_PRESORTED_MOE = threading.local()
# The only block size a presort is accepted for; other tile sizes re-sort.
PRESORTED_MOE_BLOCK_M = 16
# Reasons a registration was ignored, so each one is logged only once.
_PRESORTED_MOE_IGNORED = set()
_PRESORTED_MOE_SORT_KEYS = (
    "sorted_token_ids",
    "sorted_weights",
    "sorted_expert_ids",
    "num_valid_ids",
    "m_indices",
    "reverse_sorted",
)


# Under torch.compile, dynamo defers this thread-local store until the graph has
# run, so the compiled fused_moe ops would read the previous step's registration.
# Running it eagerly (a graph break) keeps it ordered before its consumer.
@torch.compiler.disable
def register_presorted_moe(aux: dict | None) -> None:
    """Hand the next fused_moe call on this thread a precomputed BM=16 sort.

    ``aux`` holds the sort outputs (sorted_token_ids, sorted_weights,
    sorted_expert_ids, num_valid_ids, m_indices, reverse_sorted) and the keys
    they were computed for (topk_ids_ptr, topk_weights_ptr, M, block_m,
    num_experts, topk). The sort outputs must use aiter's BM=16 sort layout:
    contiguous tensors on topk_ids' device, fp32 sorted_weights and int32
    everything else, sorted_expert_ids with one entry per 16 sorted rows,
    sorted_weights and m_indices at least as long as sorted_token_ids,
    reverse_sorted with M * topk entries and num_valid_ids with 2. A
    registration that does not match the call or breaks this layout is ignored
    (logged once per reason), and fused_moe runs its own sort.

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
    with block_m == 16 and without expert_mask or num_local_tokens; other calls
    run fused_moe's own sort.

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
    stage_block_m=None,
    expert_mask=None,
    num_local_tokens=None,
    output=None,
):
    """Return ``aux`` if it can replace fused_moe's own sort for this call, else None.

    ``block_m`` is the selected kernels' tile size; ``stage_block_m`` is the
    block size fused_moe hands its sort and stages (a caller's block_size_M),
    which defaults to ``block_m``. Both must be 16.
    """
    if not aux:
        return None
    if is_plain_presort(aux):
        presorted_moe_ignored("a general-layout sort cannot replace the BM=16 aux sort")
        return None
    if int(block_m) != PRESORTED_MOE_BLOCK_M:
        presorted_moe_ignored(f"block_m={block_m}")
        return None
    # The stages index sorted_expert_ids by stage_block_m rows, so a caller's
    # block_size_M that overrides the kernels' tile would misread a BM=16 sort.
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
    if not _presorted_layout_ok(aux, topk_ids, topk, num_experts):
        presorted_moe_ignored("sort tensors break the BM=16 sort layout")
        return None
    # The stages zero or write output while still reading the sort tensors.
    if output is not None and any(
        bytes_overlap(output, aux[k]) for k in _PRESORTED_MOE_SORT_KEYS
    ):
        presorted_moe_ignored("output overlaps the sort tensors")
        return None
    return aux


def _presorted_layout_ok(aux, topk_ids, topk, num_experts):
    """Whether the six sort tensors match the layout the aux-sort kernels read.

    The stages size their intermediates from sorted_token_ids.numel() (max_sorted)
    and index sorted_weights and m_indices by sorted row, sorted_expert_ids by
    16-row block and reverse_sorted by route. num_valid_ids[0] (padded rows) is
    read by every stage; [1] (valid tokens) only by the valid-token-count GEMM2,
    which this path never enables, but every aiter sort emits both.
    """
    try:
        ids, weights, experts, num_valid, m_indices, reverse = (
            aux[k] for k in _PRESORTED_MOE_SORT_KEYS
        )
    except KeyError:
        return False
    device = topk_ids.device
    i32 = dtypes.i32
    for t, dtype in (
        (ids, i32),
        (weights, dtypes.fp32),
        (experts, i32),
        (num_valid, i32),
        (m_indices, i32),
        (reverse, i32),
    ):
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
    max_sorted = ids.numel()
    # Worst case for a BM=16 sort, as aiter's own sort allocates: every route
    # plus BM - 1 padding rows per active expert. Smaller cannot hold all routings.
    capacity = aux_sort_capacity(
        topk_ids.shape[0], topk, num_experts, PRESORTED_MOE_BLOCK_M
    )
    return (
        max_sorted >= capacity
        and experts.numel() * PRESORTED_MOE_BLOCK_M == max_sorted
        and weights.numel() >= max_sorted
        and m_indices.numel() >= max_sorted
        and reverse.numel() == topk_ids.shape[0] * topk
        and num_valid.numel() >= 2
    )


# A caller sort in the layout of fused_moe's general moe_sorting path (any
# block_m, optionally under an EP expert_mask), registered with
# aux["layout"] = PRESORTED_MOE_LAYOUT_PLAIN. It carries no m_indices or
# reverse_sorted, so only the plain sort path takes it.
PRESORTED_MOE_LAYOUT_PLAIN = "plain"
_PLAIN_SORT_KEYS = (
    "sorted_token_ids",
    "sorted_weights",
    "sorted_expert_ids",
    "num_valid_ids",
)
# id(mask) -> (weakref to the mask, (version, data_ptr, numel, ep_layout) when
# checked, whether it matched). An entry is dropped when its mask is freed.
_EP_MASK_CHECKED = {}
# (M, num_experts, topk) -> block_m of the last general-path sort for that shape.
_PLAIN_SORT_BLOCK_M = {}


def note_plain_sort_block_m(M, num_experts, topk, block_m):
    _PLAIN_SORT_BLOCK_M[(int(M), int(num_experts), int(topk))] = int(block_m)


def plain_sort_block_m(M, num_experts, topk):
    """block_m of fused_moe's last general-path sort for this shape, or None.

    The block size comes from fused_moe's kernel selection, which a caller sorting
    ahead of it cannot see; a caller whose layers share one MoE configuration
    can register with what the previous call used, and a stale guess only costs
    that call its presort (match_plain_presorted_moe checks block_m)."""
    return _PLAIN_SORT_BLOCK_M.get((int(M), int(num_experts), int(topk)))


def plain_sort_capacity(M, topk, num_experts, block_m):
    """Rows fused_moe's general moe_sorting allocates for this call."""
    padded = M * topk + num_experts * block_m - topk
    return (padded + block_m - 1) // block_m * block_m


def is_plain_presort(aux):
    return bool(aux) and aux.get("layout") == PRESORTED_MOE_LAYOUT_PLAIN


def forget_ep_mask(expert_mask=None):
    """Drop the cached EP-mask check of ``expert_mask``, or of every mask if None.

    Updates through torch ops on the mask (or a view of it) and a new mask tensor
    are detected without this. Call it after writing a mask's memory any other
    way: a raw-pointer kernel, a write through ``.data``, or a peer's IPC handle.
    """
    if expert_mask is None:
        _EP_MASK_CHECKED.clear()
        return
    entry = _EP_MASK_CHECKED.get(id(expert_mask))
    if entry is not None and entry[0]() is expert_mask:
        del _EP_MASK_CHECKED[id(expert_mask)]


def _forget_freed_mask(ref, key):
    entry = _EP_MASK_CHECKED.get(key)
    if entry is not None and entry[0] is ref:
        del _EP_MASK_CHECKED[key]


def _tensor_version(t):
    try:
        return t._version
    except RuntimeError:  # inference tensors have no version counter
        return None


def _ep_mask_matches(expert_mask, num_experts, ep_layout):
    """Whether expert_mask selects exactly the experts ep_layout describes.

    ep_layout = (routed_begin, num_routed, shared_begin, num_shared): the rank
    owns global experts [routed_begin, routed_begin + num_routed) and
    [shared_begin, shared_begin + num_shared), shared after routed, so local ids
    are g - routed_begin and num_routed + g - shared_begin.

    The comparison reads the mask back to the host, so its result is cached per
    mask tensor and reused while the tensor is alive and its version counter,
    data_ptr and numel are unchanged. Writes that bypass the version counter
    need forget_ep_mask(). An inference tensor has no version counter, so its
    mask is compared on every eager call. Under graph capture, where the
    read-back cannot run, a mask with no valid cached result returns None (not
    checked). A captured graph keeps the decision made at capture.
    """
    layout = tuple(int(v) for v in ep_layout)
    version = _tensor_version(expert_mask)
    state = (version, expert_mask.data_ptr(), expert_mask.numel(), layout)
    key = id(expert_mask)
    entry = _EP_MASK_CHECKED.get(key)
    if (
        version is not None
        and entry is not None
        and entry[0]() is expert_mask
        and entry[1] == state
    ):
        return entry[2]
    if torch.cuda.is_current_stream_capturing():
        return None
    routed_begin, num_routed, shared_begin, num_shared = layout
    ok = (
        expert_mask.numel() == num_experts
        and 0 <= routed_begin
        and routed_begin + num_routed <= shared_begin
        and shared_begin + num_shared <= num_experts
    )
    if ok:
        want = torch.zeros(num_experts, dtype=torch.int32, device="cpu")
        want[routed_begin : routed_begin + num_routed] = 1
        want[shared_begin : shared_begin + num_shared] = 1
        ok = torch.equal(expert_mask.detach().to("cpu", torch.int32), want)
    if version is not None:
        ref = weakref.ref(expert_mask, lambda r, k=key: _forget_freed_mask(r, k))
        _EP_MASK_CHECKED[key] = (ref, state, ok)
    return ok


def match_plain_presorted_moe(
    aux,
    topk_ids,
    topk_weight,
    *,
    block_m,
    num_experts,
    topk,
    expert_mask=None,
    num_local_tokens=None,
    output=None,
):
    """Return ``aux`` if it can replace the general moe_sorting of this call, else None.

    ``aux`` must hold what moe_sorting would return for (topk_ids, topk_weight,
    num_experts, block_m, expert_mask): sorted_token_ids, sorted_weights,
    sorted_expert_ids and num_valid_ids in its layout, sized as it allocates,
    with local expert ids under a mask. ``ep_layout`` describes the mask the
    sort applied (None for none) and is checked against ``expert_mask``.
    """
    if not aux:
        return None
    if not is_plain_presort(aux):
        presorted_moe_ignored("a BM=16 aux sort cannot replace the general sort")
        return None
    if num_local_tokens is not None:
        presorted_moe_ignored("num_local_tokens is set")
        return None
    for key, expected in (
        ("topk_ids_ptr", topk_ids.data_ptr()),
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
    ep_layout = aux.get("ep_layout")
    if (expert_mask is None) != (ep_layout is None):
        presorted_moe_ignored("expert_mask and the sort's ep_layout disagree")
        return None
    if expert_mask is not None:
        matches = _ep_mask_matches(expert_mask, num_experts, ep_layout)
        if matches is None:
            presorted_moe_ignored(
                "expert_mask was not checked before graph capture (new, updated "
                "or inference tensor)"
            )
            return None
        if not matches:
            presorted_moe_ignored("expert_mask does not match the sort's ep_layout")
            return None
    try:
        ids, weights, experts, num_valid = (aux[k] for k in _PLAIN_SORT_KEYS)
    except KeyError:
        presorted_moe_ignored("sort tensors missing")
        return None
    device = topk_ids.device
    for t, dtype in (
        (ids, dtypes.i32),
        (weights, dtypes.fp32),
        (experts, dtypes.i32),
        (num_valid, dtypes.i32),
    ):
        if (
            not isinstance(t, torch.Tensor)
            or t.ndim != 1
            or t.dtype != dtype
            or t.device != device
            or not t.is_contiguous()
        ):
            presorted_moe_ignored("sort tensors break the general sort layout")
            return None
    max_sorted = ids.numel()
    if not (
        max_sorted == plain_sort_capacity(topk_ids.shape[0], topk, num_experts, block_m)
        and experts.numel() * int(block_m) == max_sorted
        and weights.numel() >= max_sorted
        and num_valid.numel() >= 2
    ):
        presorted_moe_ignored("sort tensors are not sized as moe_sorting sizes them")
        return None
    if output is not None and any(
        bytes_overlap(output, aux[k]) for k in _PLAIN_SORT_KEYS
    ):
        presorted_moe_ignored("output overlaps the sort tensors")
        return None
    return aux


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

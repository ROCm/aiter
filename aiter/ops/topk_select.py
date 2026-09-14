# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row top-k with a DeepSelect-shaped interface, over four backends.

aiter carries four per-row selectors, each fastest in a different corner and
none able to cover the whole domain:

    small_k  one chunk per lane, so k <= wave_size, and a survivor buffer that
             grows with the row bound. Unbeatable on short rows and tiny k.
    plain    the C++/ASM radix selector. Bounded at k=2048. Scales with rows
             better than anything else, so it owns the wide-M middle.
    decode   a grid-wide radix select: histogram, reduce, gather. Owns the very
             wide rows at low row counts, and all of k=4096.
    stream   one workgroup per row, reading the row once. Owns the narrow rows
             and the large-M end of the wide ones.

`topk_select` picks between them. The parameter names, order and return shape
follow `deep_select.topk` so code written against DeepSelect ports across; where
this cannot honour DeepSelect's behaviour it raises rather than diverging
silently. The differences are listed on `topk_select` itself.

Ties: as in DeepSelect, no prefix index may be assumed by default. `plain` has
no tie order at all (measured: 40 equal scores, 8 places, a set that is neither
the lowest nor the highest); `small_k` resolves toward the larger column,
`decode` and `stream` toward the smaller. `tie=` narrows the backend set to
those that can promise a direction, which costs speed.
"""

from functools import lru_cache

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, wave_size_of
from aiter.ops.flydsl.kernels.topk_per_row_radix_stream import (
    build_topk_per_row_radix_stream_module,
    topk_per_row_radix_stream_serves,
)
from aiter.ops.flydsl.topk_per_row import flydsl_top_k_per_row_decode
from aiter.ops.flydsl.topk_per_row_small_k import (
    topk_per_row_small_k,
    topk_per_row_small_k_serves,
)
from aiter.ops.topk_plain import topk_plain, topk_plain_batches_ragged_rows

__all__ = ["topk_select", "topk_select_backend"]

_SUPPORTED_GFX = ("gfx942", "gfx950")

_PLAIN_MAX_K = 2048
# Which backends can promise a column order among equal scores. `plain` appears
# in no list: its tie order falls out of its internal geometry.
#
# `small_k` is absent from "low" for more than a flipped comparison: once more
# chunks tie at the cut than there are places, step 2 has already dropped chunks
# by lane id, and lane is `(col // 4) % wave`, not monotone in the column.
# Measured on 64 tied chunks at k=16, the flipped build returns [208, 256, 260,
# ...] against a canonical [128, 132, 136, ...]. Serving "low" needs the chunk
# tie-break made column-aware first.
_BACKENDS_BY_TIE = {
    None: ("small_k", "plain", "decode", "stream"),
    "low": ("decode", "stream"),
    "high": ("small_k",),
}
# `plain` selects a different set of tied columns from one call to the next:
# measured, 2 of 512 slots differed on a repeat, 18 for one row across a batch of
# 100. Every other backend is a pure function of the row -- the property a
# tensor-parallel caller needs, and weaker than promising a direction.
_NONDETERMINISTIC = frozenset({"plain"})
# Order to fall back in when the shape rules name nothing that is available.
# Streaming first because it takes a row length natively and scales with rows;
# `plain` last for the reasons below.
_PREFERENCE = ("stream", "decode", "small_k", "plain")
# Fitted to a 232-cell sweep -- M 1..16384, N 2048..1M, k 16..4096, to 64 GiB --
# by topk_fit_policy.py over topk_full_sweep.csv. Re-run both rather than nudging
# a number: the function is piecewise constant. Every backend was checked against
# torch in a poisoned buffer before being timed, in interleaved rounds, because
# one reading of this domain runs 1.42x off another.
#
# Fitted to a preference, not the stopwatch alone: among backends within 1.4x of
# the fastest it takes the most preferred (`_PREFERENCE`). That costs a mean
# 1.063x and a worst 1.77x against the fastest of the moment, and lands a
# deterministic backend on 200 of the 232 cells. The rules it replaces lost a
# mean 1.181x, a worst 4.18x, and missed the 1.4x tolerance on 38 cells -- they
# had no k term at all, having been sampled only at k=16.
#
# Two unsampled gaps, deliberate: N=65536, and 16384 x 1M above k=16 (the
# allocator could not return 64 GiB fast enough). Spot-measured, small_k at
# 64 x 65536, k=16 is 12.7us against plain's 28.6 -- where the rule already
# sends it. Neither gap is load-bearing; neither is measured.

# `plain` is the only one that scales WITH rows, so past enough of them on a
# middling width it wins outright -- ahead of small_k, hence tested first.
_PLAIN_MANY_ROWS = 16384
_PLAIN_MANY_ROWS_BAND = (8192, 32768)
# small_k narrows by dropping chunks below the cut, and a chunk is a lane: at k
# equal to the wave width it drops none. Survivors at 8192 columns run 18 at
# k=16, 44 at k=32, then 300 at k=64, and the time steps 1.7x-1.8x between k=63
# and k=64 alone. Below this bound it wins over the whole row range, 1 to 16384,
# so it needs no width or row term.
_SMALL_K_MAX_K = 32
# decode is grid-wide, so it needs a wide row to spread over, and wins on few
# rows AND (a very wide row OR a k past plain's reach) -- never on few rows
# alone: eight dispatches, 37us of fixed cost at one row of 65536, against a
# 0.045us streamed read.
_DECODE_MAX_M = 64
_DECODE_WIDE_N = 1048576
_DECODE_MID_N = 8192
_DECODE_MID_MIN_K = 4096
# `plain` also holds a middling-width band at very few rows, where the other
# three are still paying fixed costs.
_PLAIN_FEW_ROWS = 8
_PLAIN_FEW_ROWS_BAND = (16384, 131072)


@lru_cache(maxsize=1)
def _unsupported_arch() -> str | None:
    """The arch, if this is one the selectors are not built for. Not evaluated
    at import: aiter is imported for codegen on hosts with no GPU."""
    gfx = get_gfx()
    return None if gfx in _SUPPORTED_GFX else gfx


@lru_cache(maxsize=8)
def _full_rows(rows: int, width: int, device: torch.device) -> torch.Tensor:
    """The default `end`: every row live to the full width.

    Cached because it is a constant the kernels only read, and building it per
    call is an allocation and a fill launch -- 6us against 7-10us of device time
    for the selection.
    """
    return torch.full((rows,), width, dtype=torch.int32, device=device)


@lru_cache(maxsize=8)
def _no_range(device: torch.device) -> torch.Tensor:
    """`plain`'s "no per-row range given" sentinel."""
    return torch.empty(0, dtype=torch.int32, device=device)


@lru_cache(maxsize=256)
def _available(width: int, k: int, wave_size: int, ragged: bool) -> frozenset:
    """Backends that can serve this geometry at all.

    Each is asked through its own predicate rather than through a copy of its
    limits kept here: a second copy drifts, and drift reads as declining a shape
    that works or, worse, accepting one that does not. Asked once per geometry,
    because asking costs 12us of Python against 7-10us of device time for the
    selection -- every predicate otherwise re-derives the arch from scratch.

    The tensor half of each contract (fp32, inner stride 1, int32 indices) is
    already enforced by `_reject_unsupported`, so what is left is geometry.
    """
    out = set()
    if topk_per_row_small_k_serves(k, width, wave_size) is None:
        out.add("small_k")
    if k <= _PLAIN_MAX_K and not (
        ragged and not topk_plain_batches_ragged_rows(width, k)
    ):
        # Outside its batched regime, ranged rows put plain on a host-side loop
        # of one call per row: 294918 launches at k=16, N=32768 where the batched
        # form is one. The other three take a row length natively.
        out.add("plain")
    # decode was refused at 4 GiB when it built descriptors over the whole
    # tensor; it slices the row first now, so there is nothing left to ask.
    out.add("decode")
    if topk_per_row_radix_stream_serves(k, wave_size) is None:
        out.add("stream")
    return frozenset(out)


@lru_cache(maxsize=1024)
def _choose(
    rows: int,
    width: int,
    k: int,
    wave_size: int,
    ragged: bool,
    tie: str | None,
    deterministic: bool,
) -> str:
    """The backend for one call shape, resolved once.

    Every input is a scalar the caller varies rarely, and the whole decision --
    which backends can serve, which the promises leave, which the shape rules
    name -- is a pure function of them. Memoized as one step so the serving path
    is a dict lookup rather than a set build, an intersection and a rule chain.
    """
    allowed = frozenset(_BACKENDS_BY_TIE[tie])
    if deterministic:
        allowed -= _NONDETERMINISTIC
    available = _available(width, k, wave_size, ragged) & allowed
    if not available:
        raise RuntimeError(
            f"no backend serves rows={rows} width={width} topk={k} "
            f"tie={tie!r} deterministic={deterministic}"
        )
    return topk_select_backend(rows, width, k, available)


def topk_select_backend(
    rows: int, width: int, k: int, available: frozenset[str]
) -> str:
    """Name the backend to use for this shape among those that can serve it.

    Not simply the fastest: where two are within 1.4x, this prefers the one
    whose answer is a function of the row alone, and among those the streaming
    selector. See the threshold block above for what that preference costs.

    The shape of the answer: `plain` takes the many-row middle, where it is the
    only one that scales with rows rather than against them; the small-k selector
    takes everything its narrowing still bites on; decode takes the few-row end
    of the very wide rows and of the large k; `plain` also takes a middling band
    at very few rows; and the streaming selector takes the rest.

    The returned name is always one of `available`.
    """
    if not available:
        raise ValueError("topk_select_backend needs at least one backend")
    lo, hi = _PLAIN_MANY_ROWS_BAND
    if "plain" in available and rows >= _PLAIN_MANY_ROWS and lo <= width <= hi:
        return "plain"
    if "small_k" in available and k <= _SMALL_K_MAX_K:
        return "small_k"
    if (
        "decode" in available
        and rows <= _DECODE_MAX_M
        and (
            width >= _DECODE_WIDE_N
            or (width >= _DECODE_MID_N and k >= _DECODE_MID_MIN_K)
        )
    ):
        return "decode"
    lo, hi = _PLAIN_FEW_ROWS_BAND
    if "plain" in available and rows <= _PLAIN_FEW_ROWS and lo <= width <= hi:
        return "plain"
    if "stream" in available:
        return "stream"
    # Every rule declined: `tie` or `deterministic` narrowed the set to backends
    # the shape rules never reach. Naming one outside `available` breaks the
    # promise the narrowing was made to keep -- `tie='high'` once fell through
    # here and was served by decode, which ties the opposite way, silently.
    return next(b for b in _PREFERENCE if b in available)


def _reject_unsupported(
    *,
    input,
    indices_type,
    idx_oob_fill_value,
    abort_when_nan_found,
    begin,
    hint,
    tie,
    sorted,
    sorted_index,
):
    """Refuse what this cannot do, rather than quietly doing something else."""
    if begin is not None:
        raise NotImplementedError("`begin` is not supported (nor is it in DeepSelect)")
    if hint is not None:
        raise NotImplementedError("`hint` is not supported (nor is it in DeepSelect)")
    if input.dim() != 2 or input.dtype != torch.float32:
        raise ValueError(
            f"input must be 2-D float32; got {tuple(input.shape)} {input.dtype}. "
            "DeepSelect also takes bfloat16; no backend here does."
        )
    if input.stride(1) != 1:
        raise ValueError("input must have inner stride 1")
    if indices_type is not torch.int32:
        raise NotImplementedError(
            f"indices_type={indices_type}: the kernels emit int32. DeepSelect "
            "defaults to int64; cast the returned tensor if you need it."
        )
    if idx_oob_fill_value != -1:
        raise NotImplementedError(
            f"idx_oob_fill_value={idx_oob_fill_value}: the kernels pad short rows "
            "with -1. DeepSelect defaults to 2147483647."
        )
    if abort_when_nan_found:
        raise NotImplementedError(
            "abort_when_nan_found=True needs the NaN count read back on the host, "
            "and this path may not synchronise. NaN outranks +inf here, which is "
            "what torch.topk does on narrow rows; DeepSelect aborts instead."
        )
    if tie not in _BACKENDS_BY_TIE:
        raise ValueError(f"tie must be None, 'low' or 'high'; got {tie!r}")
    if sorted and sorted_index:
        # Descending by value and ascending by index are two different orders of
        # the same pairs; honouring both would mean returning a `values` that
        # does not line up with `idx`.
        raise ValueError(
            "sorted=True and sorted_index=True ask for two different orderings "
            "of the same (value, index) pairs; pick one"
        )


def topk_select(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin: torch.Tensor | None = None,
    end: torch.Tensor | None = None,
    indices_type: torch.dtype = torch.int32,
    sorted_index: bool = False,
    hint: torch.Tensor | None = None,
    output_idx: torch.Tensor | None = None,
    output_idx_offset: torch.Tensor | None = None,
    idx_oob_fill_value: int = -1,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = False,
    abort_when_nan_found: bool = False,
    tie: str | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Per-row top-k, dispatched across aiter's four selectors.

    Parameter names, order and return shape follow ``deep_select.topk``. Four
    defaults differ, because this cannot honour DeepSelect's and will not
    pretend to -- each raises if you pass DeepSelect's value:

    ======================  ============  =========  =============================
    parameter               DeepSelect    here       why
    ======================  ============  =========  =============================
    ``abort_when_nan_found``  ``True``    ``False``  aborting needs a device-to-host
                                                     read, which this path may not do
    ``idx_oob_fill_value``    2147483647  ``-1``     what the kernels write
    ``indices_type``          int64       int32      what the kernels emit
    ``input`` dtype           bf16/fp32   fp32       no bf16 backend
    ======================  ============  =========  =============================

    ``return_value`` also defaults the other way -- ``False`` here, ``True`` in
    DeepSelect -- and this one does not raise, since asking for the values back
    is still honoured. No backend produces the values: they are gathered from
    ``input`` afterwards, which is a separate kernel plus a compare and a select
    over ``[rows, topk]``, and it is pure overhead for the callers that only
    route on the indices. Measured at rows=1024, width=32768, topk=16: 39.8us of
    selection under 74.3us of call, so the gather and the buffers around it were
    most of the time spent. Pass ``return_value=True`` to get them.

    Args:
        input: ``[rows, width]`` float32, inner stride 1.
        topk: elements to select per row.
        sorted: sort the returned values descending. Done on the host.
        end: ``[rows]`` int32 exclusive right bound per row, DeepSelect's
            ``end``; this is each row's live length. Defaults to the full width.
        sorted_index: sort the returned indices ascending. Done on the host.
        output_idx: ``[rows, topk]`` int32 to write into; allocated if omitted.
        output_idx_offset: ``[rows]`` int32 added to every live index.
        return_value: gather the selected scores and return them. Off by
            default; see above. ``sorted`` still works without it -- the values
            are gathered to derive the order and then dropped.
        tie: ``None`` leaves which of several equal scores wins unspecified, as
            DeepSelect does, and lets every backend run. ``'low'`` and ``'high'``
            promise the smallest or largest column and restrict the backend set,
            which can cost up to 1.9x. Either implies ``deterministic``.
        deterministic: the *set* of selected columns is a function of the row
            alone -- the same input selects the same columns on every call, and a
            row selects the same columns wherever it sits in the batch. The order
            they are returned in is not promised, and does move between calls:
            the streaming selector takes its output slots from a shared counter,
            so a row with more ties at the cut than places permutes. Pass
            ``sorted=True`` or ``sorted_index=True`` if the returned tensor
            itself has to be reproducible, not just its contents.

            Weaker than ``tie`` and cheaper than it: it only excludes ``plain``,
            keeping the small-k selector that ``tie='low'`` has to give up. Costs
            up to 1.9x where ``plain`` would have won.

    Returns:
        ``(values, indices)``; ``values`` is None when ``return_value`` is False.
    """
    _reject_unsupported(
        input=input,
        indices_type=indices_type,
        idx_oob_fill_value=idx_oob_fill_value,
        abort_when_nan_found=abort_when_nan_found,
        begin=begin,
        hint=hint,
        tie=tie,
        sorted=sorted,
        sorted_index=sorted_index,
    )
    unsupported = _unsupported_arch()
    if unsupported is not None:
        raise RuntimeError(f"topk_select is not supported on {unsupported}")
    rows, width = input.shape
    if not 1 <= topk <= width:
        raise ValueError(f"topk must be in [1, {width}], got {topk}")

    row_lens = _full_rows(rows, width, input.device) if end is None else end
    if row_lens.shape != (rows,) or row_lens.dtype != torch.int32:
        raise ValueError(f"end must be int32 [{rows}], got {tuple(row_lens.shape)}")
    idx = (
        torch.empty((rows, topk), dtype=torch.int32, device=input.device)
        if output_idx is None
        else output_idx
    )
    if idx.shape != (rows, topk) or idx.dtype != torch.int32:
        raise ValueError(
            f"output_idx must be int32 [{rows}, {topk}], got {tuple(idx.shape)}"
        )

    backend = _choose(
        rows,
        width,
        topk,
        wave_size_of(input.device.index),
        end is not None,
        tie,
        deterministic,
    )
    _dispatch(backend, input, row_lens, idx, topk, rows, end is not None)

    values = None
    if return_value or sorted:
        # Gather before any offset is applied: the offset renumbers the output
        # for a sharded vocabulary, but the values still live in this tensor,
        # and gathering with shifted indices reads off the end of the row.
        #
        # `sorted` orders the pair by value, so the values are needed to derive
        # that order even when the caller does not want them back. Gathering
        # them and dropping them is the cost of asking for the order; returning
        # indices in an arbitrary order from `sorted=True` is not an option.
        gathered = input.gather(1, idx.long().clamp_min_(0))
        gathered.masked_fill_(idx < 0, value_oob_fill_value)
        if sorted:
            gathered, order = torch.sort(gathered, dim=1, descending=True)
            idx = idx.gather(1, order)
        if return_value:
            values = gathered
    if sorted_index:
        # Reorder the values with it: `values[j]` is the score at `idx[j]`, and
        # sorting one of the pair alone silently breaks that. Note padded slots
        # carry -1 and so sort to the front; DeepSelect's 2147483647 default
        # sends them to the back instead.
        idx, order = torch.sort(idx, dim=1)
        if values is not None:
            values = values.gather(1, order)
    if output_idx_offset is not None:
        # A padded slot keeps its sentinel; only live indices move. Per-row
        # constant, so it cannot disturb an ordering applied above.
        idx = idx + (idx >= 0).to(torch.int32) * output_idx_offset.view(rows, 1)
    if output_idx is not None and idx.data_ptr() != output_idx.data_ptr():
        # Sorting and offsetting build new tensors; the caller asked for its own
        # buffer to hold the answer, so put it back.
        output_idx.copy_(idx)
        idx = output_idx
    return values, idx


def _dispatch(backend, input, row_lens, idx, topk, rows, ragged):
    if backend == "small_k":
        topk_per_row_small_k(input, row_lens, idx, topk)
    elif backend == "plain":
        # plain takes a [start, end) pair, not a length, and an empty
        # `rowStarts` with a real `rowEnds` reads as "no range" -- silently over
        # the whole row. Pass the pair only when the rows really differ: uniform
        # rows through the ranged overload cost 294918 launches against 25.
        # Write-only scratch: the caller never sees these values. Left to the
        # caching allocator rather than kept, the way `get_topk_scratch_workspace`
        # argues for -- a kept buffer would be shared across streams.
        vals = torch.empty_like(idx, dtype=input.dtype)
        if ragged:
            starts = torch.zeros_like(row_lens)
            topk_plain(input, idx, vals, topk, True, starts, row_lens, -1, 1)
        else:
            empty = _no_range(input.device)
            topk_plain(input, idx, vals, topk, True, empty, empty, -1, 1)
    elif backend == "decode":
        flydsl_top_k_per_row_decode(
            input, 1, row_lens, idx, rows, input.stride(0), 1, topk, stable=True
        )
    elif backend == "stream":
        _run_compiled(
            build_topk_per_row_radix_stream_module(
                topk, wave_size_of(input.device.index)
            ),
            input,
            row_lens,
            idx,
            torch.empty(1, 1, dtype=input.dtype, device=input.device),
            1,
            rows,
            torch.cuda.current_stream(input.device),
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")

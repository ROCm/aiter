# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row top-k with a DeepSelect-shaped interface, over four backends.

aiter carries four per-row selectors, each fastest in a different corner and
none able to cover the whole domain:

    argmax   k=1 only, and a reduction rather than a selection: the row split
             across as many workgroups as it takes to fill the part. Owns k=1,
             and is the only one to take bf16/fp16 as well as fp32.
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
the lowest nor the highest); `small_k` resolves toward the larger column and
`stream` toward the smaller, whether or not they were asked to. `decode` gives
the smaller column only when asked, since that is a mode of its kernel and the
mode costs up to 20%; `tie='low'` is the request. `tie=` also narrows the
backend set to those that can promise a direction, which costs speed.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, wave_size_of
from aiter.ops.flydsl.kernels.topk.topk_per_row_radix_stream import (
    build_topk_per_row_radix_stream_module,
    topk_per_row_radix_stream_block_threads,
    topk_per_row_radix_stream_lds_plan,
    topk_per_row_radix_stream_serves,
)
from aiter.ops.flydsl.topk.topk_per_row import flydsl_top_k_per_row_decode
from aiter.ops.flydsl.topk.topk_per_row_argmax import (
    ARGMAX_DTYPES,
    topk_per_row_argmax,
    topk_per_row_argmax_serves,
)
from aiter.ops.flydsl.topk.topk_per_row_small_k import (
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
    None: ("argmax", "small_k", "plain", "decode", "stream"),
    "low": ("argmax", "decode", "stream"),
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
_PREFERENCE = ("argmax", "stream", "decode", "small_k", "plain")
# Fitted to a 565-cell sweep -- rows 1..16384, widths 2048..1M, k 16..4096, to
# 8 GiB -- by `topk_backend_fit.py` over `topk_backend_sweep.py`'s table. Re-run
# both rather than nudging a number: the function is piecewise constant, and
# three of these constants are pinned by ties rather than by a measurement (see
# below). Every backend is checked against the others' selected VALUES before
# being timed, in interleaved rounds, because one reading of this domain runs
# 1.42x off another.
#
# Fitted to the stopwatch alone. The rules this replaces were fitted to a
# preference instead -- among backends within 1.4x of the fastest they took the
# most preferred, and `_PREFERENCE` lists `stream` ahead of `decode` -- which
# was defensible while both were deterministic and close. They are not close:
# decode is the fastest backend on 257 of 565 cells and the old rules reached it
# on 65, for a mean 1.295x and a worst 5.916x against the per-cell oracle.
#
# Constants the sweep cannot separate, held at their shipped values rather than
# moved on a tie: `_SMALL_K_MAX_K` (no k=32 cell) and `_PLAIN_MANY_ROWS_BAND`'s
# floor (no width between 4096 and 8192). Widen the sweep before touching either.
#
# One rule was dropped, not retuned: `plain` used to take a middling-width band
# at <= 8 rows, and with decode's gate widened that band is now decode's on
# every cell of it. Every value of the old bound scored identically, which is
# what a dead rule looks like.

# The block widths `topk_per_row_radix_stream_block_threads` chooses between.
# `_available` has to ask about both without knowing which the row count picks.
_STREAM_BLOCK_WIDTHS = (512, 1024)

# `plain` is the only backend that scales WITH rows, so past enough of them on a
# middling width it wins outright -- ahead of small_k, hence tested first. It is
# never the fastest below k=1024: of the 64 cells it wins, 29 are k=1024 and 35
# are k=2048.
_PLAIN_MANY_ROWS = 256
_PLAIN_MANY_ROWS_BAND = (8192, 65536)
_PLAIN_MIN_K = 1024

# small_k narrows by dropping chunks below the cut, and a chunk is a lane: at k
# equal to the wave width it drops none. Survivors at 8192 columns run 18 at
# k=16, 44 at k=32, then 300 at k=64, and the time steps 1.7x-1.8x between k=63
# and k=64 alone. Below this bound it wins over the whole row range.
_SMALL_K_MAX_K = 32

# decode is grid-wide: it needs a row wide enough to spread a grid over, and few
# enough rows that the grid is not already full. BOTH bounds relax at the k that
# amortises its launch -- its time is flat at 30..36us across widths
# 32768..65536, k 16..1024 and 1..64 rows, which is fixed cost rather than work,
# while `stream` over those same cells runs 18.6..51.4us.
#
# `_DECODE_AMORTISING_K` sits in an unsampled gap (256..1024) and the narrow
# gate's width in another (4096..8192). Widen the sweep before moving either.
_DECODE_AMORTISING_K = 1024
# (narrowest row, most rows) decode will take, below and at that k.
_DECODE_GATE = (65536, 64)
_DECODE_GATE_AMORTISED = (8192, 128)
# At the largest k served, decode wins at ANY row count over a band of widths --
# wide enough to spread a grid across, narrow enough that `stream`'s split
# cannot outrun it. Measured at k=4096 over rows 1..16384, decode against
# stream: 0.28x..0.96x inside the band, 0.90x..1.26x at 8192 and 1.02x..1.73x at
# 262144, which is why it has both ends.
_DECODE_ANY_ROWS_K = 4096
_DECODE_ANY_ROWS_BAND = (16384, 131072)


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


@triton.jit
def _gather_selected_kernel(
    scores_ptr,
    idx_ptr,
    out_ptr,
    scores_stride0,
    idx_stride0,
    out_stride0,
    topk,
    fill,
    BLOCK_K: tl.constexpr,
):
    """`out[r, j] = scores[r, idx[r, j]]`, or `fill` where `idx` is negative."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    live = offs < topk
    idx = tl.load(idx_ptr + row * idx_stride0 + offs, mask=live, other=-1)
    # A padded slot holds -1, which would address backwards; the mask below is
    # what keeps `fill` in that lane, so the clamp only has to be in bounds.
    kept = live & (idx >= 0)
    val = tl.load(
        # int64 throughout: `rows * stride0` passes 2^31 at 16384 rows of a
        # 1M-wide fp32 tensor, and a 32-bit offset wraps there silently.
        scores_ptr + row * scores_stride0 + tl.where(kept, idx, 0).to(tl.int64),
        mask=kept,
        other=fill,
    )
    tl.store(out_ptr + row * out_stride0 + offs, val, mask=live)


def _gather_selected(scores, idx, fill):
    """The selected scores, padded slots filled, in one launch.

    Replaces `idx.long().clamp_min_(0)`, a `gather`, an `idx < 0` and a
    `masked_fill_` -- five launches whose cost is almost all fixed. Measured
    across `[rows, topk]` from 640 to 524288 elements, the element count grows
    800x while those five grow 17.1us to 38.1us, so what they cost is being five
    rather than what they touch. One launch over the same data measured 5.9-11.4us.
    """
    rows, topk = idx.shape
    out = torch.empty((rows, topk), dtype=scores.dtype, device=scores.device)
    _gather_selected_kernel[(rows,)](
        scores,
        idx,
        out,
        scores.stride(0),
        idx.stride(0),
        out.stride(0),
        topk,
        fill,
        BLOCK_K=triton.next_power_of_2(topk),
    )
    return out


@lru_cache(maxsize=256)
def _available(
    width: int, k: int, wave_size: int, ragged: bool, fp32: bool = True
) -> frozenset:
    """Backends that can serve this geometry at all.

    Each is asked through its own predicate rather than through a copy of its
    limits kept here: a second copy drifts, and drift reads as declining a shape
    that works or, worse, accepting one that does not. Asked once per geometry,
    because asking costs 12us of Python against 7-10us of device time for the
    selection -- every predicate otherwise re-derives the arch from scratch.

    The tensor half of each contract (inner stride 1, int32 indices) is already
    enforced by `_reject_unsupported`, so what is left is geometry -- and the
    one dtype fact the geometry predicates do not carry: only the reduction has
    a half-format build, so a non-fp32 input leaves it alone in the set. That
    shape is already unreachable via `_reject_unsupported`, which refuses a half
    format past k=1; keeping it here means the two cannot disagree about which
    backend would have been asked.
    """
    out = set()
    if topk_per_row_argmax_serves(k) is None:
        out.add("argmax")
    if not fp32:
        return frozenset(out)
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
    # Both block widths, not the one `_dispatch` will pick: this set is memoized
    # without the row count, and the width `_dispatch` chooses depends on it. So
    # admit the streaming selector only where EITHER width could be asked for,
    # which keeps `_available` from promising a build that then refuses.
    #
    # Not free by construction -- a wider block has a wider tile and so wants
    # more LDS for the same prefetch depth (k=2048 resolves to 72 KiB at 512 and
    # 120 KiB at 1024) -- but `_resolve_lds` trades depth for room, and on gfx950
    # both widths resolve over the whole k range.
    #
    # On gfx942 they do not: a 64 KiB CU holds the half-width build up to k=1024
    # and the full-width one only to k=256, so `all` withdraws the streaming
    # selector for a k the narrow block could still have served. That is the
    # conservative direction and the one this set can express -- it is memoized
    # without the row count, so it cannot know which width will be asked for --
    # and it is a decline the router can act on rather than a launch failure.
    if all(
        topk_per_row_radix_stream_serves(k, wave_size, block_threads=bt) is None
        for bt in (_STREAM_BLOCK_WIDTHS)
    ):
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
    fp32: bool,
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
    available = _available(width, k, wave_size, ragged, fp32) & allowed
    if not available:
        raise RuntimeError(
            f"no backend serves rows={rows} width={width} topk={k} "
            f"tie={tie!r} deterministic={deterministic} fp32={fp32}"
        )
    return topk_select_backend(rows, width, k, available)


def _plain_takes(rows: int, width: int, k: int) -> bool:
    """Enough rows for the row-scaling selector, on a width it is tuned for."""
    lo, hi = _PLAIN_MANY_ROWS_BAND
    return rows >= _PLAIN_MANY_ROWS and k >= _PLAIN_MIN_K and lo <= width <= hi


def _decode_takes(rows: int, width: int, k: int) -> bool:
    """Room to spread a grid across, and a grid that is not already full."""
    min_n, max_m = _DECODE_GATE_AMORTISED if k >= _DECODE_AMORTISING_K else _DECODE_GATE
    band_lo, band_hi = _DECODE_ANY_ROWS_BAND
    return (width >= min_n and rows <= max_m) or (
        k >= _DECODE_ANY_ROWS_K and band_lo <= width <= band_hi
    )


def topk_select_backend(
    rows: int, width: int, k: int, available: frozenset[str]
) -> str:
    """Name the backend to use for this shape among those that can serve it.

    The shape of the answer: `plain` takes the many-row middle, where it is the
    only one that scales with rows rather than against them; the small-k selector
    takes everything its narrowing still bites on; decode takes the rest of the
    low-row end, over any row wide enough to spread a grid across; and the
    streaming selector takes what is left, which is the many-row end outside
    plain's band.

    Fitted to a 565-cell sweep -- rows 1..16384, widths 2048..1M including the
    non-power-of-two widths a sparse indexer produces, k 16..4096 -- against the
    fastest backend measured at each cell. Re-swept twice since, because
    `_dispatch` twice changed what it hands a backend: the streaming selector's
    re-select trigger learned a row-count term, and then its `partial` split was
    retuned and went from unreachable to live, which moved `stream` by up to
    2.58x on the low-row end. A mean 1.014x and a p90 1.020x against the oracle,
    more than 1.2x off on 12 cells, and 1.009x of it by total time.

    The worst cell is 1.564x: `plain` takes 256 rows of 8192 and decode is
    faster there. Every way of trimming that band regresses somewhere else by
    0.631x..0.707x for a mean gain of at most 1.004x, so it stands.

    Two rules for changing any of this. Score candidates as an A/B against the
    rule in place, not only against the per-cell oracle -- the two disagreed
    four times over this table, and each time the oracle preferred a rule that
    made some shape markedly slower. And re-run the SWEEP, not just
    `topk_backend_fit.py`, whenever `_dispatch` changes what it hands a backend;
    twice now that has moved a boundary the fit alone would have kept.

    The returned name is always one of `available`.
    """
    if not available:
        raise ValueError("topk_select_backend needs at least one backend")
    # k=1 first and unconditionally: the others answer it by building machinery
    # the answer does not need, and lose 1.3x to 12x doing so.
    if "argmax" in available:
        return "argmax"
    if "plain" in available and _plain_takes(rows, width, k):
        return "plain"
    if "small_k" in available and k <= _SMALL_K_MAX_K:
        return "small_k"
    if "decode" in available and _decode_takes(rows, width, k):
        return "decode"
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
    topk,
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
    if input.dim() != 2 or input.dtype not in ARGMAX_DTYPES:
        raise ValueError(
            f"input must be 2-D and one of "
            f"{sorted(str(d) for d in ARGMAX_DTYPES)}; got "
            f"{tuple(input.shape)} {input.dtype}"
        )
    if input.dtype is not torch.float32 and topk != 1:
        # Named here rather than left to `_choose`, whose "no backend serves"
        # reads as a geometry problem. Only the k=1 reduction has a half-format
        # build: it folds int32 ordering keys, so it can widen each element as
        # it reads it. The selectors compare and re-read the scores themselves,
        # so for them a half format is a second set of kernels, not a load.
        raise NotImplementedError(
            f"{input.dtype} is served only at topk=1, the reduction; got "
            f"topk={topk}. Cast to float32 for a wider selection."
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

    Parameter names, order and return shape follow ``deep_select.topk``, and
    the return shape means the 2-tuple specifically: DeepSelect returns
    ``(values, indices)`` always, and its ``return_value=False`` sets the first
    element to None rather than changing the arity. So does this.

    Checked against DeepSelect v1.0.0 / main at 0f03b68 (2026-09-10): the first
    fourteen parameters match by name, order and default except where the table
    below says otherwise. Re-check before trusting it -- that repository is days
    old and moving. Two traps found while checking: its README's usage snippet
    passes ``sorted_index=True`` and ``indices_type=torch.int32`` explicitly, and
    neither is the default; and DeepWiki's generated page reports the
    ``indices_type`` default as int32, which is wrong.

    Four defaults differ, because this cannot honour DeepSelect's and will not
    pretend to -- each raises if you pass DeepSelect's value:

    ======================  ============  =========  =============================
    parameter               DeepSelect    here       why
    ======================  ============  =========  =============================
    ``abort_when_nan_found``  ``True``    ``False``  aborting needs a device-to-host
                                                     read, which this path may not do
    ``idx_oob_fill_value``    2147483647  ``-1``     what the kernels write
    ``indices_type``          int64       int32      what the kernels emit
    ``input`` dtype           bf16/fp32   see below  half formats only at ``topk=1``
    ======================  ============  =========  =============================

    fp32 is served at every k. bf16 and fp16 are served at ``topk=1`` and raise
    past it: the reduction folds int32 ordering keys, so it widens each element
    as it reads it, while the selectors compare and re-read the scores and would
    need a second set of kernels. At k=1 this is worth having rather than
    casting -- the row is half the bytes, and the cast is its own pass over it.

    That row is the one divergence that is not a narrowing. fp16 is not a
    DeepSelect dtype at all -- its ``csrc/api.cpp`` takes bfloat16 or float32 and
    rejects the rest -- so at ``topk=1`` this accepts an input DeepSelect will
    not, while above ``topk=1`` it accepts less. Do not read the table as "a
    subset of DeepSelect" in either direction.

    ``return_value`` also defaults the other way -- ``False`` here, ``True`` in
    DeepSelect -- and this one does not raise, since asking for the values back
    is still honoured. No backend produces the values: they are gathered from
    ``input`` afterwards, which is a separate kernel plus a compare and a select
    over ``[rows, topk]``, and it is pure overhead for the callers that only
    route on the indices. Measured at rows=1024, width=32768, topk=16: 39.8us of
    selection under 74.3us of call, so the gather and the buffers around it were
    most of the time spent. Pass ``return_value=True`` to get them.

    Args:
        input: ``[rows, width]``, inner stride 1. float32 at any ``topk``;
            bfloat16 and float16 at ``topk=1`` only.
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
        topk=topk,
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
        input.dtype is torch.float32,
    )
    _dispatch(
        backend, input, row_lens, idx, topk, rows, end is not None, tie, deterministic
    )

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
        gathered = _gather_selected(input, idx, value_oob_fill_value)
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


# A row is one workgroup, so a call with few rows leaves the machine empty
# however wide those rows are. The split cuts each row into G slices, selects
# each in its own workgroup, and merges the G*k survivors -- trading a second
# pass over a much smaller array for G times the parallelism.
#
# Sweep these over the domain the ROUTER reaches, not by forcing `stream`. Fitted
# the other way once, and the cells it was fitted on could not run: the chain put
# `decode` ahead of `stream` across the whole of the split's own gate, so of 565
# routing cells 36 would have split and `stream` was picked on none -- which is
# how the block target came to contradict itself, stopping G at 2 for 128 rows
# where 4 measured 1.2x faster.
#
# Over the 108 cells the router does send here (`stream_split_grid.csv`), the
# best G is 4 below 128 rows, 2 at 192..256 but 1.4x..1.6x SLOWER there at
# k >= 2048, and 1 from 512 rows up where the machine is full and the merge is
# pure cost. Hence the row bound: lifting it to 256 scores better against the
# per-cell oracle and is 0.976x as an A/B, 26 cells slower. This rule changes
# twelve cells and none of them regress.
_SPLIT_TARGET_BLOCKS = 512
_SPLIT_MAX_ROWS = 128
_SPLIT_MIN_WIDTH = 1 << 18
# Bounds the MERGE, not the fan-out: stage 2 selects from `G * k` values in one
# workgroup per row, so a G large enough makes it the original problem again.
# Never binding in the reachable domain -- `_SPLIT_TARGET_BLOCKS` holds G to
# `512 // rows`, and the router keeps 64 rows and fewer for `decode`.
_SPLIT_MAX_PARTS = 16


def _stream_split_parts(rows: int, width: int, k: int, ragged: bool) -> int:
    """How many slices to cut each row into, or 1 to select it whole.

    Ragged rows are excluded, and not only for tidiness: a slice shorter than k
    pads with -1, and the merge's labels would then repeat. Its tie select
    assumes labels are unique -- it gives untied slots the key 0, and a repeated
    -1 label maps onto that same 0 -- so the count would come out wrong. A row
    at least `parts * k` wide has no short slice and no repeated label.
    """
    if ragged or rows > _SPLIT_MAX_ROWS or width < _SPLIT_MIN_WIDTH:
        return 1
    parts = 1
    while (
        parts * 2 <= _SPLIT_MAX_PARTS
        and rows * parts * 2 <= _SPLIT_TARGET_BLOCKS
        and width >= parts * 2 * k
    ):
        parts *= 2
    return parts


def _stream_scratch(input, dtype=None):
    """A placeholder for a stream argument this mode does not use.

    The kernel takes one tensor per optional output -- the per-slice values a
    split writes, and the column labels a merge reads -- and a build that uses
    neither still has to be handed something of the right dtype for the launcher
    to specialise on.
    """
    return torch.empty(1, 1, dtype=dtype or input.dtype, device=input.device)


def _dispatch(
    backend, input, row_lens, idx, topk, rows, ragged, tie=None, deterministic=False
):
    if backend == "argmax":
        topk_per_row_argmax(input, row_lens, idx)
    elif backend == "small_k":
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
        # `stable` buys the smallest-index tie-break AND the selected set being
        # a function of the row, so both `tie='low'` and `deterministic` need
        # it. Only "no promise at all" may drop it.
        #
        # The second half of that was measured, not assumed: without `stable`
        # this backend selects a DIFFERENT set of tied columns from one call to
        # the next, which `test_invariants` catches at 16 rows of 262144, k=2048
        # as "selects the same set under {'deterministic': True}". A radix
        # select being a pure function of the row is the obvious guess and it is
        # wrong.
        #
        # Hardwired on, it cost 0-20% -- and the routing was fitted against a
        # decode that was always paying it, which is part of why the old rules
        # sent these shapes elsewhere.
        flydsl_top_k_per_row_decode(
            input,
            1,
            row_lens,
            idx,
            rows,
            input.stride(0),
            1,
            topk,
            stable=tie == "low" or deterministic,
        )
    elif backend == "stream":
        wave = wave_size_of(input.device.index)
        parts = _stream_split_parts(rows, input.shape[1], topk, ragged)
        if parts > 1:
            # Stage 1 launches `rows * parts` blocks and stage 2 launches
            # `rows`, so the two sit on opposite sides of the block-width rule
            # and each has to ask it for itself.
            part_idx = torch.empty(
                rows, parts * topk, dtype=torch.int32, device=input.device
            )
            part_val = torch.empty(
                rows, parts * topk, dtype=input.dtype, device=input.device
            )
            stream = torch.cuda.current_stream(input.device)
            _run_compiled(
                build_topk_per_row_radix_stream_module(
                    topk,
                    wave,
                    partial=True,
                    block_threads=topk_per_row_radix_stream_block_threads(
                        rows * parts, topk
                    ),
                    # A slice, not the row: this stage runs one block per
                    # slice over a slice's width, and the plan reads both.
                    lds_plan=topk_per_row_radix_stream_lds_plan(
                        rows * parts, -(-input.shape[1] // parts), topk
                    ),
                ),
                input,
                row_lens,
                part_idx,
                part_val,
                _stream_scratch(input, torch.int32),
                parts,
                rows * parts,
                stream,
            )
            # The merge orders by the ORIGINAL column, which `part_idx` carries,
            # so its answer is the unsplit answer rather than merely a valid
            # one -- and it writes those columns straight out, with no slot-to
            # -column gather to undo afterwards.
            _run_compiled(
                build_topk_per_row_radix_stream_module(
                    topk,
                    wave,
                    labelled=True,
                    block_threads=topk_per_row_radix_stream_block_threads(rows, topk),
                    # The merge's rows are `parts * topk` wide, not the input's.
                    lds_plan=topk_per_row_radix_stream_lds_plan(
                        rows, parts * topk, topk
                    ),
                ),
                part_val,
                torch.full(
                    (rows,), parts * topk, dtype=torch.int32, device=input.device
                ),
                idx,
                _stream_scratch(input),
                part_idx,
                1,
                rows,
                stream,
            )
            return
        _run_compiled(
            # The block width follows the ROW COUNT and k, not the row width --
            # see the sweep behind `topk_per_row_radix_stream_block_threads`.
            # Costs a mean 1.001x against choosing per cell, where holding one
            # width costs 1.135x and a worst 2.03x.
            build_topk_per_row_radix_stream_module(
                topk,
                wave,
                block_threads=topk_per_row_radix_stream_block_threads(rows, topk),
                # The deepest prefetch the budget allows is not the one that
                # wins, and the budget has no term for the row count.
                lds_plan=topk_per_row_radix_stream_lds_plan(rows, input.shape[1], topk),
            ),
            input,
            row_lens,
            idx,
            _stream_scratch(input),
            _stream_scratch(input, torch.int32),
            1,
            rows,
            torch.cuda.current_stream(input.device),
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")

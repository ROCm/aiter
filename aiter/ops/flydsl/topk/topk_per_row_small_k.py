# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL per-row TopK for a small k (block-sparse indexer selection)."""

from functools import lru_cache

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from ..kernels.tensor_shim import wave_size_of
from ..kernels.topk.topk_launch import run_topk_launcher
from ..kernels.topk.topk_per_row_small_k import (
    build_topk_per_row_small_k_module,
    lds_limit,
    topk_per_row_small_k_shape,
)
from ..kernels.topk.topk_per_row_small_k_wave import (
    MAX_WIDTH as _WAVE_MAX_WIDTH,
)
from ..kernels.topk.topk_per_row_small_k_wave import (
    build_topk_per_row_small_k_wave_module,
    wave_strips,
)
from .row_lens import row_len_slot

# The row bound fixes the register tile, so it is a compile-time constant.
# Bucketing to powers of two keeps a serving run to a couple of compilations.
_MIN_ROW_BUCKET = 256


def _row_bucket(width: int) -> int:
    bucket = _MIN_ROW_BUCKET
    while bucket < width:
        bucket *= 2
    return bucket


_WAVE_K = 16
# One wave a row cannot fill the part with few rows, and a wide row is its latency. Measured on gfx950, k=16,
# the one-wave build against the chunked one (graph replay, us per call, 6 rotating buffers):
#   rows     width 1024-1408   1700-3072       4096            6144           8192
#   1-1024   0.5-0.9x (wins)   1.1-1.3x slower 1.2-2.0x slower 1.6x slower    1.5-2.0x slower
#   2048     0.7-0.9x          0.8-0.9x        parity (1.0x)   parity         -
#   3072+    0.5-0.8x          0.6-0.8x        0.75-0.9x       0.87-0.9x      1.0-1.07x (parity)
# so it takes a row up to the width listed for the row count: parity at one row of 1408 columns, a win above.
_WAVE_WIDTH_BY_ROWS = ((3072, 6144), (2048, 3072), (0, 1408))


def _wave_width_limit(rows: int) -> int:
    return next(w for r, w in _WAVE_WIDTH_BY_ROWS if rows >= r)


@lru_cache(maxsize=64)
def _wave_tile(k: int, width: int, wave_size: int, width_limit: int) -> int:
    """The window tile of the one-wave-per-row build if it serves and wins this geometry, else 0.

    It serves k=16 rows of 16 to 8192 columns (the 13 index bits of its tie-break) on gfx950, where a wave is 64
    lanes, and wins up to `width_limit` columns (`_wave_width_limit` of the row count, table above). The tensor
    half of its contract (fp32, inner stride 1) is the one `_unsupported_reason` already enforces for both
    builds. A pure function of its arguments, like `topk_per_row_small_k_serves`, so the hot path asks it once
    per shape through `_plan`.
    """
    if (
        k != _WAVE_K
        or not _WAVE_K <= width <= min(width_limit, _WAVE_MAX_WIDTH)
        or wave_size != 64
        or get_gfx_runtime() != "gfx950"
    ):
        return 0
    return wave_strips(width)


@lru_cache(maxsize=64)
def topk_per_row_small_k_serves(k: int, width: int, wave_size: int) -> str | None:
    """Why this geometry cannot be built, or None if it can.

    Separate from the tensor contract below because it is what the build
    itself would hit, and because it is a pure function of three integers --
    the hot path asks it through `_plan`, once per shape.
    """
    # One chunk per lane, so k cannot outrun the wave.
    if not 1 <= k <= wave_size:
        return f"k must be in [1, {wave_size}] (one chunk per lane), got {k}"
    if k > width:
        return f"k={k} exceeds the row width {width}"
    # The survivor buffer grows with the bucketed row bound. Over budget the
    # build fails inside the compiler and leaves the HIP context unusable, so
    # this has to be a decline rather than a crash.
    bucket = _row_bucket(width)
    _, _, lds_bytes = topk_per_row_small_k_shape(k, bucket, wave_size)
    if lds_bytes > lds_limit():
        return (
            f"k={k} over a {bucket}-wide row bound needs {lds_bytes} bytes of "
            f"LDS, over the {lds_limit()} limit"
        )
    return None


def _unsupported_reason(
    scores: torch.Tensor,
    row_lens: torch.Tensor | None,
    indices: torch.Tensor,
    k: int,
) -> str | None:
    """Why this selector cannot serve these tensors, or None if it can.

    The geometry half is shared with the launch path, so the two cannot drift
    into declining a shape that works or accepting one that does not.
    """
    if scores.dim() != 2 or scores.dtype != torch.float32:
        return f"scores must be 2-D float32, got {tuple(scores.shape)} {scores.dtype}"
    if scores.stride(1) != 1:
        return "scores must have inner stride 1"
    rows, width = scores.shape
    if row_lens is not None and (
        row_lens.shape != (rows,)
        or row_lens.dtype != torch.int32
        or row_lens.stride(0) != 1
    ):
        return (
            f"row_lens must be contiguous int32 [{rows}], got {tuple(row_lens.shape)}"
        )
    if indices.shape != (rows, k) or indices.dtype != torch.int32:
        return f"indices must be int32 [{rows}, {k}], got {tuple(indices.shape)}"
    if indices.stride(1) != 1:
        return "indices must have inner stride 1"
    if not all(
        t.is_cuda and t.device == scores.device
        for t in (scores, row_lens, indices)
        if t is not None
    ):
        return "every tensor must be on the scores' GPU"
    return topk_per_row_small_k_serves(k, width, wave_size_of(scores.device.index))


def topk_per_row_small_k_supported(
    scores: torch.Tensor,
    row_lens: torch.Tensor | None,
    indices: torch.Tensor,
    k: int,
) -> bool:
    """Would :func:`topk_per_row_small_k` serve these tensors?

    Everything it tests is fixed once the buffers are allocated, so a caller
    holding a fallback asks once and keeps the answer -- asking costs 11us,
    against 7-10us of device time for the selection itself.
    """
    return _unsupported_reason(scores, row_lens, indices, k) is None


@lru_cache(maxsize=64)
def _plan(
    k: int,
    width: int,
    wave_size: int,
    forced_blocks: bool,
    full_rows: bool,
    width_limit: int,
):
    """The launcher for one (k, row-width) shape, and whether it is the one-wave build.

    `full_rows` compiles the width in as every row's length, so the kernel
    reads no length tensor. It costs no extra plan -- the width is already
    the key -- only a build per exact width rather than per bucket (per window
    tile for the one-wave build).
    """
    reason = topk_per_row_small_k_serves(k, width, wave_size)
    if reason is not None:
        raise ValueError(f"[FlyDSL topk_per_row_small_k] {reason}")
    strips = _wave_tile(k, width, wave_size, width_limit)
    if strips:
        return build_topk_per_row_small_k_wave_module(strips, full_rows), True
    return (
        build_topk_per_row_small_k_module(
            k,
            _row_bucket(width),
            wave_size,
            # Compile the forced-block pins in only for callers that use them: six
            # instructions per element, and the row read is what a wide row costs --
            # worth 1.07x at 4096 columns rising to 1.29x at 16384. Keyword so every
            # call lands on one `cache` entry.
            forced_blocks=forced_blocks,
            fixed_row_len=width if full_rows else 0,
        ),
        False,
    )


def topk_per_row_small_k(
    scores: torch.Tensor,
    row_lens: torch.Tensor | None,
    indices: torch.Tensor,
    k: int,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> None:
    """Write each row's top-k column ids, descending by score.

    Args:
        scores: ``[rows, width]`` float32, inner stride 1.
        row_lens: ``[rows]`` int32; columns at or past a row's length are
            invisible to it. None makes every row full-width, and then no
            length tensor is read at all.
        indices: ``[rows, k]`` int32, written in place. A row with fewer than
            ``k`` live columns is right-padded with -1.
        init_blocks: leading columns forced into the selection.
        local_blocks: trailing columns (relative to each row's length) forced
            into the selection, outranking ``init_blocks``.

    Ties go to the larger column. Rows of k=16 on gfx950 that the table above says it wins (up to 1408
    columns, wider as the row count grows) are served by a one-wave-per-row build that follows that rule exactly (the order key is the score's bits, NaN as -inf, -0 below +0, then the
    larger column); every other geometry takes the chunked build, which keeps the right multiset of scores but
    may keep a different tied column among equal scores at the k-th place. Only the geometry is checked here, since the
    check is what the cached plan already does; the tensor contract above is
    the caller's, and :func:`topk_per_row_small_k_supported` is how to ask.
    """
    rows, width = scores.shape
    launcher, wave = _plan(
        k,
        width,
        wave_size_of(scores.device.index),
        bool(init_blocks or local_blocks),
        row_lens is None,
        _wave_width_limit(rows),
    )
    stream = torch.cuda.current_stream(scores.device)
    if wave:
        # One wave a row, raw addresses and row pitches (in elements): the
        # flattened [heads * rows, width] rows of a prefill scorer, with the
        # head-major `row_lens` it already publishes, are this same 2-D call.
        if rows:
            run_topk_launcher(
                launcher,
                scores.data_ptr(),
                scores.stride(0),
                width,
                row_len_slot(row_lens, scores.device).data_ptr(),
                int(init_blocks),
                int(local_blocks),
                indices.data_ptr(),
                indices.stride(0),
                rows,
                stream,
            )
        return
    run_topk_launcher(
        launcher,
        scores,
        row_len_slot(row_lens, scores.device),
        indices,
        width,
        int(init_blocks),
        int(local_blocks),
        rows,
        stream,
    )

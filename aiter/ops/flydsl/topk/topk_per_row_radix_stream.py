# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL streaming per-row TopK: one workgroup per row, or per slice of one."""

import torch

from ..kernels.tensor_shim import wave_size_of
from ..kernels.topk.topk_launch import run_topk_launcher
from ..kernels.topk.topk_per_row_radix_stream import (
    build_topk_per_row_radix_stream_module,
    build_topk_per_row_radix_stream_module_for_width,
    topk_per_row_radix_stream_block_threads,
    topk_per_row_radix_stream_lds_plan,
)
from .row_lens import row_len_slot

__all__ = ["topk_per_row_radix_stream"]

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


def _split_parts(rows: int, width: int, k: int, ragged: bool) -> int:
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


def _scratch(scores, dtype=None):
    """A placeholder for a stream argument this mode does not use.

    The kernel takes one tensor per optional output -- the per-slice values a
    split writes, and the column labels a merge reads -- and a build that uses
    neither still has to be handed something of the right dtype for the launcher
    to specialise on.
    """
    return torch.empty(1, 1, dtype=dtype or scores.dtype, device=scores.device)


def topk_per_row_radix_stream(
    scores: torch.Tensor,
    row_lens: torch.Tensor | None,
    indices: torch.Tensor,
    k: int,
    lds_budgets: tuple[int, int] | None = None,
) -> None:
    """Write each row's top-k column ids, in no promised order.

    Args:
        scores: ``[rows, width]`` float32, inner stride 1.
        row_lens: ``[rows]`` int32; columns at or past a row's length are
            invisible to it. None makes every row full-width, and then no
            length tensor is read at all.
        indices: ``[rows, k]`` int32, written in place. A row with fewer than
            ``k`` live columns is right-padded with -1.
        k: elements selected per row.
        lds_budgets: the LDS the builds may claim, `_lds_budgets` of the
            input's GPU; by default the current GPU's.

    Ties go to the smaller column. Few full-width rows are cut into slices that
    are selected apart and then merged, which returns the unsplit answer.
    """
    rows, width = scores.shape
    wave = wave_size_of(scores.device.index)
    stream = torch.cuda.current_stream(scores.device)
    ragged = row_lens is not None
    len_slot = row_len_slot(row_lens, scores.device)
    parts = _split_parts(rows, width, k, ragged)
    if parts == 1:
        run_topk_launcher(
            # The block width follows the ROW COUNT and k, not the row width --
            # see the sweep behind `topk_per_row_radix_stream_block_threads`.
            # Costs a mean 1.001x against choosing per cell, where holding one
            # width costs 1.135x and a worst 2.03x.
            build_topk_per_row_radix_stream_module_for_width(
                k,
                wave,
                width,
                # The direct terminal placement is measured and
                # regression-tested at k=2048. Keep it there until the other k
                # values have equivalent coverage rather than silently
                # broadening their path.
                terminal_ok=k == 2048,
                block_threads=topk_per_row_radix_stream_block_threads(rows, k),
                lds_budgets=lds_budgets,
                # The deepest prefetch the budget allows is not the one that
                # wins, and the budget has no term for the row count.
                lds_plan=topk_per_row_radix_stream_lds_plan(rows, width, k),
                full_rows=not ragged,
            ),
            scores,
            len_slot,
            indices,
            _scratch(scores),
            _scratch(scores, torch.int32),
            width,
            1,
            rows,
            stream,
        )
        return

    # Ragged rows never split, so both stages are full-rows builds. Stage 1
    # launches `rows * parts` blocks and stage 2 launches `rows`, so the two sit
    # on opposite sides of the block-width rule and each has to ask it for itself.
    part_idx = torch.empty(rows, parts * k, dtype=torch.int32, device=scores.device)
    part_val = torch.empty(rows, parts * k, dtype=scores.dtype, device=scores.device)
    run_topk_launcher(
        build_topk_per_row_radix_stream_module(
            k,
            wave,
            partial=True,
            block_threads=topk_per_row_radix_stream_block_threads(rows * parts, k),
            lds_budgets=lds_budgets,
            # A slice, not the row: this stage runs one block per slice over a
            # slice's width, and the plan reads both.
            lds_plan=topk_per_row_radix_stream_lds_plan(
                rows * parts, -(-width // parts), k
            ),
            full_rows=True,
        ),
        scores,
        len_slot,
        part_idx,
        part_val,
        _scratch(scores, torch.int32),
        width,
        parts,
        rows * parts,
        stream,
    )
    # The merge orders by the ORIGINAL column, which `part_idx` carries, so its
    # answer is the unsplit answer rather than merely a valid one -- and it
    # writes those columns straight out, with no slot-to-column gather to undo
    # afterwards.
    run_topk_launcher(
        build_topk_per_row_radix_stream_module(
            k,
            wave,
            labelled=True,
            block_threads=topk_per_row_radix_stream_block_threads(rows, k),
            lds_budgets=lds_budgets,
            # The merge's rows are `parts * k` wide, not the input's.
            lds_plan=topk_per_row_radix_stream_lds_plan(rows, parts * k, k),
            full_rows=True,
        ),
        part_val,
        len_slot,
        indices,
        _scratch(scores),
        part_idx,
        parts * k,
        1,
        rows,
        stream,
    )

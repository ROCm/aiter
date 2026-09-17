# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row argmax: k=1, as a reduction rather than a selection.

At k=1 the other selectors do work that has no answer to produce. The small-k
one ranks chunks by their maximum to bound a survivor set -- but the top-1 chunk
is just the chunk holding the maximum, and the survivors are whatever ties with
it; the radix ones walk the key space for a threshold that is the answer itself.
Measured against `torch.argmax`, which is the reduction this should have been,
the small-k selector is 1.3x behind at 50 rows of 129280 and 12x at 1024 rows of
262144.

A row is split across `splits` workgroups, not held by one. That is the whole
point of the shape this serves: one workgroup per row leaves a 50-row call using
a fifth of the machine, and both `torch.argmax` and the selectors it beats are
stuck at the same ceiling -- 0.65-0.86 TB/s against the 5.7 the same reduction
reaches once there are enough rows to fill the GPU.

The answer is the largest score, ties going to the smallest column, which is
what `torch.argmax` returns and the order `topk_per_row_decode` and the
streaming selector already promise. NaN outranks +inf, as everywhere else here.

Each thread scans its own columns in increasing order, so a strict `>` already
keeps the earliest of a tie without a column test; only the cross-thread fold
needs one, and that is log(block) comparisons, not one per element.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    BFloat16,
    Float16,
    Float32,
    Int32,
    const_expr,
    gpu,
    range_constexpr,
)

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature, ord_signed_f32
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_atom

# A 16-byte load whatever the dtype -- four fp32, eight of either half format --
# so the element count per vector travels with it. Everything downstream counts
# vectors: the split rule, the slice bounds, the column arithmetic. All of them
# read this, and none of them may reach for a module-level default instead.
VEC_BY_ELEM = {Float32: 4, BFloat16: 8, Float16: 8}
# One ordering rule for all three: widen to fp32, then `ord_signed_f32`. fp32
# covers the range and the precision of both half formats, subnormals included,
# so the widen is exact -- injective and order-preserving -- and ties to the
# smallest column and NaN over +inf follow by construction rather than by a
# second rule someone has to keep in step with the first.
_ELEM_TAG = {Float32: "f32", BFloat16: "bf16", Float16: "f16"}
_VEC = VEC_BY_ELEM[Float32]
# Swept jointly with the split rule below -- 64/128/256/512 against every split
# count, on the same 34 cells -- because the two are coupled: a narrower block
# absorbs less of a slice, which is what the split rule decides. 256 is the best
# single width (mean 1.043x off a per-cell (block, split) oracle, against 1.046
# for 128, 1.064 for 512 and 1.080 for 64), and no block rule tried beat holding
# it flat. The whole configuration sits 1.056x off that oracle, so ~1.3% is left
# in the split rule and the rest needs a per-cell block choice nobody has found a
# form for. `argmax_joint_fit.py` re-runs the sweep.
_BLOCK_THREADS = 256
_INT32_MIN = -2147483648

# Fitted to a 34-cell sweep -- rows 1..16384 by widths 2048..1M -- against the
# best split count measured at each cell, on an idle card. Costs a mean 1.015x
# and a worst 1.087x against that oracle, over gains that run to 11x. The three
# constants are not independent; re-run `argmax_rule_fit.py` rather than nudging
# one.
#
# Workgroups worth having in flight. Past this the rows already fill the part and
# a split only adds partials to fold.
_TARGET_WORKGROUPS = 1024
# Vectors one workgroup should chew at most. A longer slice runs a longer tail
# that nothing overlaps, which is why splitting still pays at 16384 rows.
_MAX_SLICE_VECTORS = 2048
# ...and at least. Below this the fold and the second launch cost more than the
# slice saves -- at 512 vectors a row, every split measured worse than none.
_MIN_SLICE_VECTORS = 512
# A row under this is short enough that one workgroup finishes it without a tail
# worth splitting away, so only the row count decides. Without it a 1024-row
# call over a 32768-wide row -- decode against a vocabulary, the common argmax --
# splits four ways and loses 13%, where one split is already at 6.05 TB/s.
_LONG_ROW_VECTORS = 16384


def topk_per_row_argmax_splits(rows: int, width: int, vec: int = _VEC) -> int:
    """How many workgroups share one row.

    Enough of them to fill the part, and few enough that each still has a slice
    worth launching for. At 16384 rows of 2048 columns this is 1 -- the rows fill
    it on their own -- and at one row of 262144 it is 128, which is 11x.

    `vec` is the caller's element count per vector, which the dtype sets. The
    thresholds below are in vectors rather than columns, so a bf16 row -- half
    the bytes, half the vectors -- reaches them where its byte count does, not
    where its column count does. Measured per dtype; see the sweep note above.
    """
    vectors = (width + vec - 1) // vec
    by_fill = max(1, -(-_TARGET_WORKGROUPS // max(rows, 1)))
    by_work = (
        max(1, vectors // _MAX_SLICE_VECTORS) if vectors > _LONG_ROW_VECTORS else 1
    )
    return min(max(by_fill, by_work), max(1, vectors // _MIN_SLICE_VECTORS))


@cache
def build_topk_per_row_argmax_module(
    splits: int,
    elem=Float32,
    block_threads: int = _BLOCK_THREADS,
    vec: int | None = None,
):
    """Compile the two halves of a `splits`-way per-row argmax.

    Returns `(slice_launcher, fold_launcher)`; the fold is None at one split,
    where the slice kernel writes the answer itself and there is nothing to
    fold. The row width is a runtime value -- nothing here is sized by it, which
    is what lets one build serve every width.

    `elem` is the score element type. Only the slice half reads scores, so the
    fold is the same kernel for every dtype; the partials it folds are int32
    ordering keys, which is what lets that be true.
    """
    if splits < 1:
        raise ValueError(f"splits must be positive, got {splits}")
    if block_threads & (block_threads - 1):
        raise ValueError(f"block must be a power of two, got {block_threads}")
    if elem not in VEC_BY_ELEM:
        raise ValueError(f"no argmax build for {elem}; have {list(VEC_BY_ELEM)}")
    vec = VEC_BY_ELEM[elem] if vec is None else vec

    @fx.struct
    class SharedStorage:
        red_key: fx.Array[Int32, block_threads, 16]
        red_col: fx.Array[Int32, block_threads, 16]

    def make_kernel(folding: bool):
        """One kernel body for both passes.

        They differ only in where a thread's candidate comes from -- a slice of
        the row, or one partial each -- and where the block's answer goes. The
        tree fold between those two is the same, and it has to live inside the
        kernel: a runtime `if` is only rewritten into an `scf.if` within the
        traced body, so a helper defined beside it traces its guard as a Python
        bool and raises.
        """

        @flyc.kernel(
            name="topk_per_row_argmax_"
            + kernel_signature(
                fold=folding,
                sp=splits,
                blk=block_threads,
                vec=vec,
                ty=_ELEM_TAG[elem],
            ),
            known_block_size=[block_threads, 1, 1],
        )
        def argmax_kernel(
            scores: fx.Tensor,
            row_lens: fx.Tensor,
            indices: fx.Tensor,
            part_key: fx.Tensor,
            part_col: fx.Tensor,
            vectors: fx.Int32,
        ):
            tid = fx.thread_idx.x
            zero = Int32(0)
            bottom = Int32(_INT32_MIN)

            storage = fx.SharedAllocator().allocate(SharedStorage)
            red_key = storage.red_key.peek().view(fx.make_layout(block_threads, 1))
            red_col = storage.red_col.peek().view(fx.make_layout(block_threads, 1))

            if const_expr(folding):
                row = fx.block_idx.x
                row_key = fx.slice(part_key, (row, None))
                row_col = fx.slice(part_col, (row, None))
                my_key = bottom
                my_col = Int32(-1)
                # A thread folds every `block_threads`-th partial, so any split
                # count is covered. One each would drop the partials past the
                # block silently, and the split rule does reach past it: at one
                # or two rows of a 1M-wide row it asks for 512 against a
                # 256-thread block. A thread's slots ascend, so its columns do
                # too, and a strict `>` keeps the earliest of a tie; the tree
                # below has the column rule for across threads.
                for chunk in range_constexpr(
                    (splits + block_threads - 1) // block_threads
                ):
                    slot = tid + Int32(chunk * block_threads)
                    live = slot < Int32(splits)
                    safe = live.select(slot, zero)
                    peer_key = row_key[safe]
                    better = live & (peer_key > my_key)
                    my_key = better.select(peer_key, my_key)
                    my_col = better.select(row_col[safe], my_col)
            else:
                part = fx.block_idx.x
                row = fx.block_idx.y
                row_len = row_lens[row]
                # Slice the row before building the descriptor: over the whole
                # tensor `num_records` is a 32-bit byte count and wraps at 4 GiB,
                # after which every load returns zero with nothing raised.
                score_row = fx.logical_divide(
                    fx.rocdl.make_buffer_tensor(
                        fx.slice(scores, (row, None)), max_size=False
                    ),
                    fx.make_layout(vec, 1),
                )
                # Whole vectors per workgroup, rounded up, so the last slice is
                # the short one and the bounds-check covers its tail.
                slice_vecs = (vectors + Int32(splits - 1)) // Int32(splits)
                first = part * slice_vecs
                last = fx.min(first + slice_vecs, vectors)

                my_key = bottom
                my_col = Int32(-1)
                # A thread's columns ascend, so a strict `>` keeps the earliest
                # of a tie and no column test is needed here.
                for vec_idx in range(first + tid, last, Int32(block_threads)):
                    src = fx.slice(score_row, (None, vec_idx))
                    fragment = fx.make_fragment_like(src)
                    fx.copy(buf_copy_atom(vec * (elem.width // 8), elem), src, fragment)
                    loaded = fx.Vector(fx.memref_load_vec(fragment))
                    for j in range_constexpr(vec):
                        col = vec_idx * Int32(vec) + Int32(j)
                        value = loaded[j]
                        key = ord_signed_f32(
                            value if elem is Float32 else value.to(Float32)
                        )
                        better = (col < row_len) & (key > my_key)
                        my_key = better.select(key, my_key)
                        my_col = better.select(col, my_col)

            # An LDS tree over the whole block, not the usual wave `shuffle_xor`
            # fold with one cross-wave pass through LDS. The shuffle form trades
            # log2(block) barriers for six shuffles, which only pays if those
            # barriers are what the fold costs -- and they are not: a wider block
            # adds a fold step and still wins where the fold is heaviest (one row
            # of 129280, two vectors a thread, 5.77us at 512 threads against 6.60
            # at 64). Revisit with a measurement, not with the argument above.
            red_key[tid] = my_key
            red_col[tid] = my_col
            gpu.barrier()
            step = block_threads // 2
            while step >= 1:
                if tid < Int32(step):
                    peer_key = red_key[tid + Int32(step)]
                    peer_col = red_col[tid + Int32(step)]
                    mine_key = red_key[tid]
                    mine_col = red_col[tid]
                    # Ties to the smaller column, so the answer does not depend
                    # on which half of the tree an element landed in.
                    take = (peer_key > mine_key) | (
                        (peer_key == mine_key) & (peer_col < mine_col)
                    )
                    red_key[tid] = take.select(peer_key, mine_key)
                    red_col[tid] = take.select(peer_col, mine_col)
                gpu.barrier()
                step //= 2

            if tid == zero:
                if const_expr(folding or splits == 1):
                    fx.slice(indices, (row, None))[zero] = red_col[zero]
                else:
                    fx.slice(part_key, (row, None))[part] = red_key[zero]
                    fx.slice(part_col, (row, None))[part] = red_col[zero]

        return argmax_kernel

    slice_kernel = make_kernel(folding=False)
    fold_kernel = make_kernel(folding=True)

    @flyc.jit
    def launch_argmax_slice(
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        indices: fx.Tensor,
        part_key: fx.Tensor,
        part_col: fx.Tensor,
        vectors: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        slice_kernel(scores, row_lens, indices, part_key, part_col, vectors).launch(
            grid=(splits, rows, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_argmax_fold(
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        indices: fx.Tensor,
        part_key: fx.Tensor,
        part_col: fx.Tensor,
        vectors: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        fold_kernel(scores, row_lens, indices, part_key, part_col, vectors).launch(
            grid=(rows, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_argmax_slice, (launch_argmax_fold if splits > 1 else None)

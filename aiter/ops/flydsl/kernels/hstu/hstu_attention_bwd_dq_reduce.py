# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-K reduction for the fused HSTU backward kernel's dQ partials.

`hstu_attention_bwd_fused_pp.py` produces dQ as a PARTIAL per KV tile, because
a workgroup there owns a KV range and `dQ[q][d] = sum_kv dS[q][kv].K[kv][d]`
reduces over kv. This kernel sums those partials.

Split-K rather than atomics: `global_atomic_add_f32` is a hard 0.40 TB/s wall
on gfx942 while `global_store_dwordx2` runs at 3.56 TB/s -- 8.8x, measured in
`tmp/hstu_repro/atomic_probe.py`, and independent of contention, atomic width
and instruction count. The atomic design needed 337 GB of traffic, about twice
the entire backward pass.

Scratch layout, built by `partial_offsets` on the host and compact by
necessity -- the dense `[max_tiles][L][H][D]` alternative is 331 GB at the
benchmark shape:

    offset[b][t]  prefix sum over (b, t) of (S_b - t*BLOCK_M) * H * D
    index         offset[b][t] + ((q_row - t*BLOCK_M) * H + h) * D + d

kv tile `t` only serves q rows `[t*BLOCK_M, S_b)` -- under the causal mask
nothing above the diagonal contributes -- so q row `r` sums tiles
`0 ..= r // BLOCK_M`. All `ROWS_PER_BLOCK` rows of a workgroup share that count
because `BLOCK_M % ROWS_PER_BLOCK == 0`, which is what lets one runtime loop
cover the whole workgroup.

The scale is already applied by the fused kernel's flush, so this kernel only
sums and converts.

`first_pass`/`last_pass` support kv-chunking (G > 1), where the fused kernel is
run several times over disjoint KV tile groups into the same reused scratch:
every pass but the first accumulates into an fp32 carry, and only the last
converts to bf16.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.hstu.hstu_attention_common import (
    WARP_SIZE,
    _dtype_to_elem_type,
)

# 16 q rows per workgroup. Each row is H*D fp32 = 1 KB at the tuned shape, so a
# thread-per-(h, d) mapping makes every tile read one fully coalesced 1 KB run,
# and 16 rows keeps the fp32 accumulators at 16 VGPR.
ROWS_PER_BLOCK = 16


def validate_hstu_attention_bwd_dq_reduce(num_heads, head_dim, block_m, dtype_str):
    """Gate the supported shape, as the fused kernel's own validator does."""
    arch = get_rocm_arch()
    if arch != "gfx942":
        raise ValueError(f"dq_reduce is specialized for gfx942, got {arch}")
    if dtype_str != "bf16":
        raise ValueError(f"dq_reduce is specialized for bf16, got {dtype_str}")
    if block_m % ROWS_PER_BLOCK != 0:
        raise ValueError(
            f"block_m ({block_m}) must be a multiple of ROWS_PER_BLOCK "
            f"({ROWS_PER_BLOCK}): a workgroup's rows share one tile count."
        )
    if num_heads * head_dim % WARP_SIZE != 0:
        raise ValueError(
            f"num_heads*head_dim ({num_heads * head_dim}) must be a multiple of "
            f"the wave size ({WARP_SIZE})"
        )


@functools.lru_cache(maxsize=None)
def build_hstu_attention_bwd_dq_reduce(
    num_heads,
    head_dim,
    block_m,
    max_tiles,
    dtype_str,
    *,
    first_pass=True,
    last_pass=True,
):
    validate_hstu_attention_bwd_dq_reduce(num_heads, head_dim, block_m, dtype_str)
    elem_dtype = _dtype_to_elem_type(dtype_str)
    HD = num_heads * head_dim
    THREADS = HD
    ROWS = ROWS_PER_BLOCK

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def hstu_attention_bwd_dq_reduce(
        dq_part: fx.Tensor,
        part_off: fx.Tensor,
        red_map: fx.Tensor,
        seq_offsets: fx.Tensor,
        dq_acc: fx.Tensor,
        out_dq: fx.Tensor,
        tile_lo: fx.Int32,
        tile_hi: fx.Int32,
    ) -> None:
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        # (batch, row block) comes from a host table rather than
        # `bid // max_blocks`. The sequences are jagged -- mean length is about
        # half of max at the benchmark shape -- so a rectangular grid would
        # spend roughly half its workgroups re-reducing rows that do not exist.
        b = red_map[fx.Int64(bid) * fx.Int64(2)]
        blk = red_map[fx.Int64(bid) * fx.Int64(2) + fx.Int64(1)]

        seq_start = fx.Int32(seq_offsets[fx.Int32(b)])
        seq_len = fx.Int32(seq_offsets[fx.Int32(b) + fx.Int32(1)]) - seq_start
        row0 = fx.Int32(blk) * fx.Int32(ROWS)

        h = tid // fx.Int32(head_dim)
        d = tid % fx.Int32(head_dim)

        # Every row of this workgroup lies in the same kv tile, so one count
        # serves all of them: BLOCK_M is a multiple of ROWS.
        n_t = row0 // fx.Int32(block_m) + fx.Int32(1)
        n_t = (n_t < tile_hi).select(n_t, tile_hi)

        acc0 = [fx.Float32(0.0) for _ in range_constexpr(ROWS)]
        if const_expr(not first_pass):
            for r in range_constexpr(ROWS):
                # The tail rows of the last block are clamped onto the last
                # real row, so they read (and later rewrite) a valid element
                # with the value that row would get anyway. Bounding the WRITE
                # instead would need a dynamic `if` around a tensor subscript
                # store, which makes the AST rewriter demand an scf.if result
                # for the output tensor.
                _rr = row0 + fx.Int32(r)
                _rr = (_rr < seq_len).select(_rr, seq_len - fx.Int32(1))
                acc0[r] = dq_acc[
                    fx.Int64(seq_start + _rr) * fx.Int64(HD) + fx.Int64(tid)
                ]

        for _t, it in range(
            fx.Int64(tile_lo), fx.Int64(n_t), fx.Int64(1), init=acc0
        ):  # ty: ignore
            # The loop-carried state comes back as raw ArithValue, which has
            # none of the DSL numeric methods; re-wrap before using it.
            acc = [
                fx.Float32(x)
                for x in (list(it) if isinstance(it, (list, tuple)) else [it])
            ]
            t = fx.Int32(_t)
            o = part_off[fx.Int64(b) * fx.Int64(max_tiles) + fx.Int64(t)]
            base = fx.Int64(o) + fx.Int64(tid)
            row_in_slot = row0 - t * fx.Int32(block_m)
            nxt = []
            for r in range_constexpr(ROWS):
                _rr = row0 + fx.Int32(r)
                _rr = (_rr < seq_len).select(_rr, seq_len - fx.Int32(1))
                _ri = row_in_slot + (_rr - row0)
                nxt.append(
                    acc[r] + dq_part[base + fx.Int64(_ri) * fx.Int64(HD)]
                )
            acc = yield nxt

        res = [
            fx.Float32(x)
            for x in (list(acc) if isinstance(acc, (list, tuple)) else [acc])
        ]
        for r in range_constexpr(ROWS):
            _rr = row0 + fx.Int32(r)
            _rr = (_rr < seq_len).select(_rr, seq_len - fx.Int32(1))
            if const_expr(last_pass):
                out_dq[fx.Int64(seq_start + _rr), h, d] = res[r].to(elem_dtype)
            else:
                dq_acc[
                    fx.Int64(seq_start + _rr) * fx.Int64(HD) + fx.Int64(tid)
                ] = res[r]

    @flyc.jit
    def launch_hstu_attention_bwd_dq_reduce(
        n_blocks: fx.Int32,
        dq_part: fx.Tensor,
        part_off: fx.Tensor,
        red_map: fx.Tensor,
        seq_offsets: fx.Tensor,
        dq_acc: fx.Tensor,
        out_dq: fx.Tensor,
        tile_lo: fx.Int32,
        tile_hi: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        hstu_attention_bwd_dq_reduce(
            dq_part, part_off, red_map, seq_offsets, dq_acc, out_dq,
            tile_lo, tile_hi,
        ).launch(  # fmt: skip
            grid=(n_blocks,), block=(THREADS,), smem=0, stream=stream
        )

    return launch_hstu_attention_bwd_dq_reduce


def build_reduce_map(seq_offsets, device):
    """Host table: one (batch, row_block) pair per REAL row block.

    Returns a flat int32 tensor of `2 * n_blocks` and `n_blocks`.
    """
    import torch

    # Built entirely on CPU with EXPLICIT devices: callers commonly set
    # `torch.set_default_device("cuda")`, which otherwise silently mixes
    # devices here and fails inside repeat_interleave.
    cpu = torch.device("cpu")
    S = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.int64).to(cpu)
    nb = torch.div(S + ROWS_PER_BLOCK - 1, ROWS_PER_BLOCK, rounding_mode="floor")
    b = torch.repeat_interleave(
        torch.arange(len(S), dtype=torch.int64, device=cpu), nb
    )
    # per-batch block index: a ramp restarted at each sequence boundary
    starts = torch.cumsum(nb, 0) - nb
    blk = torch.arange(
        int(nb.sum()), dtype=torch.int64, device=cpu
    ) - torch.repeat_interleave(starts, nb)
    m = torch.stack([b, blk], dim=1).reshape(-1).to(torch.int32)
    return m.to(device), int(nb.sum())

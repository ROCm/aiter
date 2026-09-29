# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-stage split-K epilogue: last arriving split reduces the fp32 partials."""

import math

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Float32, Int32
from flydsl.expr.typing import Vector as Vec

from . import communication_ops_utils as comm_ops

# Split-K partials cross XCDs, which do not share L2. An agent-scope fence
# would write back the whole L2 per CTA and evict the A/B tiles everyone else
# is reading; sc0|sc1 writes through just these accesses.
CPOL_COHERENT = 0x1 | 0x10

VEC = 4  # elements per thread per access: dwordx4 loads, dwordx2 bf16 stores

_COPY_ATOM = {
    32: fx.rocdl.BufferCopy32b,
    64: fx.rocdl.BufferCopy64b,
    128: fx.rocdl.BufferCopy128b,
}


def pairwise_sum(parts):
    """Sum a Python list of Vectors as a balanced tree.

    Module level because the kernel AST rewriter would turn this ``while`` into
    an ``scf.while``; it is trace-time metaprogramming over a Python list.
    """
    while len(parts) > 1:
        nxt = []
        for lhs, rhs in zip(parts[0::2], parts[1::2]):
            nxt.append(lhs + rhs)
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    return parts[0]


def reduce_thread_split(tile_m, tile_n, nthreads):
    """Threads along (M, N) for the reduce, VEC contiguous columns per thread.

    The MMA's C layout gives each lane elements that are contiguous in M, whose
    memory stride is N, so partitioning the reduce with it would issue one dword
    per element. This layout is chosen for the memory instead: one row per
    thread where the block divides the tile rows, so the ragged-M skip below is
    exact, otherwise as many threads along N as the row splits into.
    """
    row_vecs = tile_n // VEC
    tm_thr = math.gcd(tile_m, nthreads)
    if row_vecs % (nthreads // tm_thr):
        tm_thr = nthreads // math.gcd(row_vecs, nthreads)
    if tile_n % VEC or tile_m % tm_thr or row_vecs % (nthreads // tm_thr):
        raise ValueError(f"tile ({tile_m}, {tile_n}) does not split over {nthreads}")
    return tm_thr, nthreads // tm_thr


@comm_ops.traced
def splitk_reduce_epilogue(
    workspace,
    out,
    semaphore,
    flag_ptr,
    tile_m,
    tile_n,
    nthreads,
    out_elem_cls,
    tid,
    bid_x,
    bid_y,
    i32_m,
    n,
    split_k,
):
    """Reduce ``split_k`` fp32 planes of ``workspace`` into ``out``, in-kernel.

    ``workspace`` and ``out`` are 2D ``[*, n]`` views; workspace holds the planes
    back to back. Every CTA bumps this tile's ``semaphore`` slot and the one that
    sees ``split_k - 1`` prior arrivals reduces. The caller must have stored its
    partial with ``CPOL_COHERENT`` and waited on it: that is the release, and it
    is why no agent fence (which flushes the whole L2) is needed.

    ``out_elem_cls`` is the only dtype knob -- any 16/32-bit output works.
    """
    out_bytes = out_elem_cls.width // 8
    tiles_n = n // tile_n

    if tid == fx.Int32(0):
        arrival = fx.Int32(
            comm_ops.atomic_add_agent(
                _semaphore_addr(semaphore, bid_x, bid_y, tiles_n), fx.Int32(1)
            )
        )
        is_last = (arrival == fx.Int32(split_k - 1)).select(fx.Int32(1), fx.Int32(0))
        fx.ptr_store(Vec.from_elements([is_last], Int32), flag_ptr)
    gpu.barrier()

    is_last = Vec(fx.make_view(flag_ptr, fx.make_layout(1, 1)).load())[0]
    if is_last != fx.Int32(0):
        plane = fx.Int64(i32_m) * fx.Int64(n)
        g_workspace = fx.rocdl.make_buffer_tensor(
            workspace,
            max_size=False,
            num_records_bytes=plane * fx.Int64(4 * split_k),
        )
        g_out = fx.rocdl.make_buffer_tensor(
            out, max_size=False, num_records_bytes=plane * fx.Int64(out_bytes)
        )
        tile = fx.make_tile(tile_m, tile_n)
        t_workspace = fx.flat_divide(g_workspace, tile)[None, None, bid_x, bid_y]
        t_out = fx.flat_divide(g_out, tile)[None, None, bid_x, bid_y]

        tm_thr, tn_thr = reduce_thread_split(tile_m, tile_n, nthreads)
        tiler_mn, tv = fx.make_layout_tv(
            fx.make_layout((tm_thr, tn_thr), (tn_thr, 1)),
            fx.make_layout((1, VEC), (1, 1)),
        )
        load_atom = fx.make_copy_atom(_COPY_ATOM[VEC * 32](CPOL_COHERENT), Float32)
        store_atom = fx.make_copy_atom(
            _COPY_ATOM[VEC * out_elem_cls.width](), out_elem_cls
        )
        p_workspace = (
            fx.make_tiled_copy(load_atom, tv, tiler_mn)
            .get_slice(tid)
            .partition_S(t_workspace)
        )
        p_out = (
            fx.make_tiled_copy(store_atom, tv, tiler_mn)
            .get_slice(tid)
            .partition_D(t_out)
        )

        # Both descriptors stop at row M, so a ragged-M access is dropped in
        # hardware; skipping the threads whose rows are all past M as well keeps
        # a short tile from issuing them at all.
        stride = fx.Int32(i32_m) * fx.Int32(n)
        row0 = fx.Int32(bid_x) * tile_m + fx.Int32(tid) // tn_thr
        if row0 < fx.Int32(i32_m):
            parts = []
            for s in range_constexpr(split_k):
                frag = fx.make_fragment_like(p_workspace)
                fx.copy(load_atom, p_workspace, frag, soffset=fx.Int32(s) * stride)
                parts.append(Vec(frag.load()))
            frag_out = fx.make_fragment_like(p_out)
            frag_out.store(pairwise_sum(parts).to(out_elem_cls))
            fx.copy(store_atom, frag_out, p_out)

        if tid == fx.Int32(0):
            # Undo this tile's arrivals with the same atomic instead of storing
            # 0, so an early increment from the next launch is not clobbered.
            comm_ops.atomic_add_agent(
                _semaphore_addr(semaphore, bid_x, bid_y, tiles_n), fx.Int32(-split_k)
            )


def _semaphore_addr(semaphore, bid_x, bid_y, tiles_n):
    idx = fx.Int32(bid_x) * tiles_n + fx.Int32(bid_y)
    return fx.Int64(fx.ptrtoint(fx.get_iter(semaphore))) + fx.Int64(idx) * fx.Int64(4)

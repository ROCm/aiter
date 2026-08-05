# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepGEMM-contiguous M-tile prefix sum (FlyDSL), single-block parallel scan.

Computes tile-aligned exclusive prefix sum of per-expert counts for the
contiguous grouped-GEMM scheduler. Single-block parallel scan replaces
torch.cumsum (avoids rocprim trampoline overhead for small E).

The block is ``MAX_EXPERTS_PER_BLOCK`` threads wide but E is not bounded by it:
the scan sweeps the experts in block-sized chunks and carries the running offset
between chunks in LDS. Kimi-K3 (E=896) is the first model to exceed one chunk.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

MAX_EXPERTS_PER_BLOCK = 512
REMAP_BLOCK_THREADS = 256


@fx.struct
class _PsumStorage:
    """LDS for the prefix-scan kernels.

    ``lds0``/``lds1`` are ping-pong buffers: each Hillis-Steele step reads one
    and writes the other, then the two swap. ``carry`` accumulates the total of
    the chunks already scanned, so an E wider than the block still gets one
    continuous prefix sum. The trailing 16 is the byte alignment of each array.
    """

    lds0: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds1: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    carry: fx.Array[fx.Int32, 1, 16]


@fx.struct
class _RoutePsumStorage:
    """LDS for the fused route+psum kernel.

    Adds ``cnt`` -- the per-expert counter the route phase bumps with LDS
    atomics -- to the ping-pong scan buffers of :class:`_PsumStorage`.
    """

    cnt: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds0: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds1: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]


def _lds_load(ptr, idx):
    """Scalar i32 load from an LDS pointer at element offset ``idx``."""
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_store(ptr, val, idx):
    """Scalar i32 store to an LDS pointer at element offset ``idx``."""
    fx.ptr_store(val, ptr + fx.Int64(idx))


# The chunked scan below is written out in both kernels rather than shared:
# @flyc.kernel AST-transforms only the decorated body, so a dynamic `for`/`if`
# does not survive being factored into a plain helper.


def build_moe_contiguous_psum_module():
    """JIT launcher: tile-aligned prefix sum over per-expert counts."""

    @flyc.kernel(
        name="moe_contiguous_psum",
        known_block_size=[MAX_EXPERTS_PER_BLOCK, 1, 1],
    )
    def psum_kernel(
        masked_m: fx.Pointer,  # (E,) int32 in
        starts: fx.Pointer,  # (E,) int32 out
        psum: fx.Pointer,  # (E,) int32 out
        contiguous_m: fx.Pointer,  # (1,) int32 out
        experts: Int32,
        tile_m: Int32,
    ):
        i32 = T.i32
        # Uint32: every value here is a non-negative count/index, so `<`, `>=`
        # and `//` lower to ult/uge/divui exactly like the arith.* calls they
        # replace.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_PsumStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        carry = lds.carry.ptr

        m_rsrc = ptr_rsrc(masked_m)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)
        c_rsrc = ptr_rsrc(contiguous_m)

        is_lane0 = tid == fx.Uint32(0)
        if is_lane0:
            _lds_store(carry, fx.Int32(0), 0)
        gpu.barrier()

        # One Hillis-Steele scan spans exactly one thread per lane, so a single
        # pass covers at most MAX_EXPERTS_PER_BLOCK experts -- it used to be the
        # whole kernel, which silently left starts/psum unwritten for every
        # expert past 512 (Kimi-K3 has 896: garbage offsets, then a memory fault
        # in the GEMM that indexes with them).
        #
        # So sweep E in block-sized chunks instead. Each chunk scans as before
        # and then adds ``carry``, the tile-aligned total of all chunks already
        # scanned, which is what makes the per-chunk scans one continuous prefix
        # sum. ``carry`` has to be LDS, not a register: it is produced by lane 0
        # and consumed by all of them on the next iteration.
        #
        # Lanes past ``experts`` feed 0 into the scan -- they keep the last lane
        # holding the true chunk total, and write no output.
        for base in range(0, experts, MAX_EXPERTS_PER_BLOCK):
            e = fx.Uint32(base) + tid
            in_expert = e < fx.Uint32(experts)
            m_e = fx.Uint32(0)
            if in_expert:
                m_e = fx.Uint32(
                    buffer_ops.buffer_load(m_rsrc, e, vec_width=1, dtype=i32)
                )
            _lds_store(lds0, fx.Int32((m_e + tile_minus_1) // tile_v * tile_v), tid)
            gpu.barrier()

            src = lds0
            dst = lds1
            for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
                if const_expr((offset & (offset - 1)) != 0):
                    continue
                val = _lds_load(src, tid)
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = _lds_load(src, tid - offset)
                _lds_store(dst, val + prev, tid)
                gpu.barrier()
                src, dst = dst, src

            base_off = _lds_load(carry, 0)
            if in_expert:
                is_not_first = tid != 0
                excl = fx.Int32(0)
                if is_not_first:
                    excl = _lds_load(src, tid - 1)
                start = excl + base_off
                buffer_ops.buffer_store(start, s_rsrc, e)
                buffer_ops.buffer_store(start + fx.Int32(m_e), p_rsrc, e)

            # Fold this chunk's total in before the next one overwrites lds0.
            chunk_total = _lds_load(src, MAX_EXPERTS_PER_BLOCK - 1)
            gpu.barrier()
            if is_lane0:
                _lds_store(carry, base_off + chunk_total, 0)
            gpu.barrier()

        if is_lane0:
            total = _lds_load(carry, 0)
            gt = total > fx.Int32(tile_v)
            buffer_ops.buffer_store(gt.select(total, tile_v), c_rsrc, 0)

    @flyc.jit
    def launch_psum(
        masked_m: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        experts: fx.Int32,
        tile_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        psum_kernel(masked_m, starts, psum, contiguous_m, experts, tile_m).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    launch_psum.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_psum


def build_moe_contiguous_psum_remap_module(split_remap: bool = False):
    """JIT launcher: contiguous psum + in-place masked-to-contiguous row remap."""

    @flyc.kernel(
        name=(
            "moe_contiguous_psum_scan"
            if split_remap
            else "moe_contiguous_psum_remap"
        ),
        known_block_size=[MAX_EXPERTS_PER_BLOCK, 1, 1],
    )
    def psum_remap_kernel(
        masked_m: fx.Pointer,
        topids_to_rows: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        numel: Int32,
        experts: Int32,
        route_max_m: Int32,
        tile_m: Int32,
        num_valid_routes: fx.Pointer,  # (1,) int32: only remap routes < this (EP dead-tail skip)
    ):
        i32 = T.i32
        # Uint32: every value here is a non-negative count/index, so `<`, `>=`
        # and `//` lower to ult/uge/divui exactly like the arith.* calls they
        # replace.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_PsumStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        carry = lds.carry.ptr

        m_rsrc = ptr_rsrc(masked_m)
        rows_rsrc = ptr_rsrc(topids_to_rows)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)
        c_rsrc = ptr_rsrc(contiguous_m)

        is_lane0 = tid == fx.Uint32(0)
        if is_lane0:
            _lds_store(carry, fx.Int32(0), 0)
        gpu.barrier()

        # One Hillis-Steele scan spans exactly one thread per lane, so a single
        # pass covers at most MAX_EXPERTS_PER_BLOCK experts -- it used to be the
        # whole kernel, which silently left starts/psum unwritten for every
        # expert past 512 (Kimi-K3 has 896: garbage offsets, then a memory fault
        # in the GEMM that indexes with them).
        #
        # So sweep E in block-sized chunks instead. Each chunk scans as before
        # and then adds ``carry``, the tile-aligned total of all chunks already
        # scanned, which is what makes the per-chunk scans one continuous prefix
        # sum. ``carry`` has to be LDS, not a register: it is produced by lane 0
        # and consumed by all of them on the next iteration.
        #
        # Lanes past ``experts`` feed 0 into the scan -- they keep the last lane
        # holding the true chunk total, and write no output.
        for base in range(0, experts, MAX_EXPERTS_PER_BLOCK):
            e = fx.Uint32(base) + tid
            in_expert = e < fx.Uint32(experts)
            m_e = fx.Uint32(0)
            if in_expert:
                m_e = fx.Uint32(
                    buffer_ops.buffer_load(m_rsrc, e, vec_width=1, dtype=i32)
                )
            _lds_store(lds0, fx.Int32((m_e + tile_minus_1) // tile_v * tile_v), tid)
            gpu.barrier()

            src = lds0
            dst = lds1
            for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
                if const_expr((offset & (offset - 1)) != 0):
                    continue
                val = _lds_load(src, tid)
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = _lds_load(src, tid - offset)
                _lds_store(dst, val + prev, tid)
                gpu.barrier()
                src, dst = dst, src

            base_off = _lds_load(carry, 0)
            if in_expert:
                is_not_first = tid != 0
                excl = fx.Int32(0)
                if is_not_first:
                    excl = _lds_load(src, tid - 1)
                start = excl + base_off
                buffer_ops.buffer_store(start, s_rsrc, e)
                buffer_ops.buffer_store(start + fx.Int32(m_e), p_rsrc, e)

            # Fold this chunk's total in before the next one overwrites lds0.
            chunk_total = _lds_load(src, MAX_EXPERTS_PER_BLOCK - 1)
            gpu.barrier()
            if is_lane0:
                _lds_store(carry, base_off + chunk_total, 0)
            gpu.barrier()

        if is_lane0:
            total = _lds_load(carry, 0)
            gt = total > fx.Int32(tile_v)
            buffer_ops.buffer_store(gt.select(total, tile_v), c_rsrc, 0)

        gpu.barrier()

        if const_expr(not split_remap):
            # Only remap valid routes ([0, valid_route_count)); dead-tail routes
            # hold unwritten/garbage rows from the route kernel and must NOT be used
            # as a row index (would OOB-read starts[expert]). They are never read
            # downstream. When truncation is disabled the caller passes a null pointer
            # instead of a (1,) tensor, so the load must not run unconditionally.
            num_valid_routes_is_set = fx.Int64(ptrtoint(num_valid_routes)) != 0
            valid_route_count = fx.Uint32(numel)
            if num_valid_routes_is_set:
                valid_route_count = fx.Uint32(
                    buffer_ops.buffer_load(
                        ptr_rsrc(num_valid_routes),
                        fx.Uint32(0),
                        vec_width=1,
                        dtype=i32,
                    )
                )
            for route_i32 in range(tid, valid_route_count, MAX_EXPERTS_PER_BLOCK):
                row = fx.Uint32(
                    buffer_ops.buffer_load(
                        rows_rsrc, route_i32, vec_width=1, dtype=i32
                    )
                )
                m = fx.Uint32(route_max_m)
                expert = row // m
                slot = row - expert * m
                start = fx.Uint32(
                    buffer_ops.buffer_load(s_rsrc, expert, vec_width=1, dtype=i32)
                )
                buffer_ops.buffer_store(start + slot, rows_rsrc, route_i32)

    @flyc.kernel(
        name="moe_contiguous_remap",
        known_block_size=[REMAP_BLOCK_THREADS, 1, 1],
    )
    def remap_kernel(
        topids_to_rows: fx.Pointer,
        starts: fx.Pointer,
        numel: Int32,
        route_max_m: Int32,
        num_valid_routes: fx.Pointer,
    ):
        i32 = T.i32
        route = (
            fx.Uint32(fx.block_idx.x) * REMAP_BLOCK_THREADS
            + fx.Uint32(fx.thread_idx.x)
        )
        valid_route_count = fx.Uint32(numel)
        num_valid_routes_is_set = fx.Int64(ptrtoint(num_valid_routes)) != 0
        if num_valid_routes_is_set:
            valid_route_count = fx.Uint32(
                buffer_ops.buffer_load(
                    ptr_rsrc(num_valid_routes),
                    fx.Uint32(0),
                    vec_width=1,
                    dtype=i32,
                )
            )
        if route < valid_route_count:
            rows_rsrc = ptr_rsrc(topids_to_rows)
            row = fx.Uint32(
                buffer_ops.buffer_load(rows_rsrc, route, vec_width=1, dtype=i32)
            )
            m = fx.Uint32(route_max_m)
            expert = row // m
            slot = row - expert * m
            start = fx.Uint32(
                buffer_ops.buffer_load(
                    ptr_rsrc(starts), expert, vec_width=1, dtype=i32
                )
            )
            buffer_ops.buffer_store(start + slot, rows_rsrc, route)

    @flyc.jit
    def launch_psum_remap(
        masked_m: fx.Pointer,
        topids_to_rows: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        numel: fx.Int32,
        experts: fx.Int32,
        route_max_m: fx.Int32,
        tile_m: fx.Int32,
        num_valid_routes: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        psum_remap_kernel(
            masked_m,
            topids_to_rows,
            starts,
            psum,
            contiguous_m,
            numel,
            experts,
            route_max_m,
            tile_m,
            num_valid_routes,
        ).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )
        if const_expr(split_remap):
            remap_grid = (
                fx.Uint32(numel) + REMAP_BLOCK_THREADS - 1
            ) // REMAP_BLOCK_THREADS
            remap_kernel(
                topids_to_rows,
                starts,
                numel,
                route_max_m,
                num_valid_routes,
            ).launch(
                grid=(arith.index_cast(T.index, remap_grid), 1, 1),
                block=(REMAP_BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    launch_psum_remap.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_psum_remap


def build_moe_route_psum_fused_module():
    """JIT launcher: single-workgroup fused route + atomic + psum + remap.

    For small token counts every route fits in one workgroup, so the three
    pre-GEMM launches (route-maps, contiguous-psum, remap) collapse into one
    kernel. The per-expert atomic counter lives in LDS (workgroup-scope
    atomics, no global round-trip), and the tile-aligned prefix sum + in-place
    masked->contiguous row remap reuse the single-block scan below.

    Outputs match ``topids_to_rows`` (contiguous layout) + ``masked_m`` counts
    + ``psum`` (m_tile_map) of the split-kernel path bit-for-bit.
    """

    @flyc.kernel(
        name="moe_route_psum_fused",
        known_block_size=[MAX_EXPERTS_PER_BLOCK, 1, 1],
    )
    def route_psum_fused_kernel(
        topk_ids: fx.Pointer,  # (numel,) i32 in
        topids_to_rows: fx.Pointer,  # (numel,) i32 out (contiguous rows)
        masked_m: fx.Pointer,  # (E,) i32 out (per-expert counts)
        starts: fx.Pointer,  # (E,) i32 out (contiguous row base per expert)
        psum: fx.Pointer,  # (E,) i32 out (= m_tile_map)
        numel: Int32,
        experts: Int32,
        max_m: Int32,
        tile_m: Int32,
    ):
        i32 = T.i32
        # Uint32: every value here is a non-negative count/index, so `<`, `>=`
        # and `//` lower to ult/uge/divui exactly like the arith.* calls they
        # replace.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_RoutePsumStorage).peek()
        lds_cnt = lds.cnt.ptr
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr

        topk_rsrc = ptr_rsrc(topk_ids)
        rows_rsrc = ptr_rsrc(topids_to_rows)
        m_rsrc = ptr_rsrc(masked_m)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)

        in_expert = tid < fx.Uint32(experts)

        # Phase A: zero the LDS per-expert atomic counter.
        if in_expert:
            _lds_store(lds_cnt, fx.Int32(0), tid)
        gpu.barrier()

        # Phase B: route + workgroup-scope LDS atomic -> masked-layout rows.
        # The atomic needs a raw addrspace(3) pointer, so the counter array's
        # base is taken as an integer here; SharedAllocator has already folded
        # its offset in, leaving only the per-expert element offset to add.
        cnt_base_i64 = fx.Int64(fx.ptrtoint(lds_cnt))
        numel_i32 = fx.Uint32(numel)
        for route_i32 in range(tid, numel_i32, MAX_EXPERTS_PER_BLOCK):
            e = buffer_ops.buffer_load(topk_rsrc, route_i32, vec_width=1, dtype=i32)
            off_i64 = fx.Int64(e) * 4
            ptr = buffer_ops.create_llvm_ptr(
                cnt_base_i64 + fx.Int64(off_i64), address_space=3
            )
            ptr = ptr._value if hasattr(ptr, "_value") else ptr
            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                arith.constant(1, type=i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result
            row = fx.Uint32(slot) + fx.Uint32(e) * fx.Uint32(max_m)
            buffer_ops.buffer_store(row, rows_rsrc, route_i32)
        gpu.barrier()

        # Phase C: tile-aligned inclusive scan of per-expert counts.
        if in_expert:
            m = fx.Uint32(_lds_load(lds_cnt, tid))
            _lds_store(lds0, (m + tile_minus_1) // tile_v * tile_v, tid)
            buffer_ops.buffer_store(m, m_rsrc, tid)
        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            if in_expert:
                val = _lds_load(src, tid)
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = _lds_load(src, tid - offset)
                _lds_store(dst, val + prev, tid)
            gpu.barrier()
            src, dst = dst, src

        if in_expert:
            is_not_first = tid != 0
            start = fx.Int32(0)
            if is_not_first:
                start = _lds_load(src, tid - 1)
            m_tid = _lds_load(lds_cnt, tid)
            buffer_ops.buffer_store(start, s_rsrc, tid)
            buffer_ops.buffer_store(start + m_tid, p_rsrc, tid)
        gpu.barrier()

        # Phase D: in-place masked -> contiguous row remap.
        for route_i32 in range(tid, numel_i32, MAX_EXPERTS_PER_BLOCK):
            row = fx.Uint32(
                buffer_ops.buffer_load(rows_rsrc, route_i32, vec_width=1, dtype=i32)
            )
            m = fx.Uint32(max_m)
            expert = row // m
            slot = row - expert * m
            start = fx.Uint32(
                buffer_ops.buffer_load(s_rsrc, expert, vec_width=1, dtype=i32)
            )
            buffer_ops.buffer_store(start + slot, rows_rsrc, route_i32)

    @flyc.jit
    def launch_route_psum_fused(
        topk_ids: fx.Pointer,
        topids_to_rows: fx.Pointer,
        masked_m: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        numel: fx.Int32,
        experts: fx.Int32,
        max_m: fx.Int32,
        tile_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        route_psum_fused_kernel(
            topk_ids,
            topids_to_rows,
            masked_m,
            starts,
            psum,
            numel,
            experts,
            max_m,
            tile_m,
        ).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    launch_route_psum_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_route_psum_fused

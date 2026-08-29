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
    ptr_buf_tensor,
)

MAX_EXPERTS_PER_BLOCK = 512

# The ep_rowmap remap+scatter is grid-strided across this many blocks. Each block
# re-derives the tiny per-expert prefix sum in LDS (barrier-free across blocks), so
# no cross-block sync is needed. Sized to fill the 256-CU gfx1250; the grid-stride
# loop stays correct for any token count.
EP_REMAP_NBLK = 256


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


@fx.struct
class _PsumRemapEpStorage:
    """LDS for the EP remap kernel: the scan ping-pong buffers only.

    No ``carry``: EP shards the experts across ranks, so the local count always
    fits one chunk. The buffer left free by the last scan step doubles as the
    per-block copy of the exclusive starts the scatter reads.
    """

    lds0: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds1: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]


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
        # Uint32: every value here is a non-negative count/index, so `<`, `>=`
        # and `//` lower to ult/uge/divui rather than their signed forms.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_PsumStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        carry = lds.carry.ptr

        m_p = ptr_buf_tensor(masked_m)
        s_p = ptr_buf_tensor(starts)
        p_p = ptr_buf_tensor(psum)
        c_p = ptr_buf_tensor(contiguous_m)

        is_lane0 = tid == fx.Uint32(0)
        if is_lane0:
            carry[0] = fx.Int32(0)
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
                m_e = fx.Uint32(m_p[e])
            lds0[tid] = fx.Int32((m_e + tile_minus_1) // tile_v * tile_v)
            gpu.barrier()

            src = lds0
            dst = lds1
            for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
                if const_expr((offset & (offset - 1)) != 0):
                    continue
                val = src[tid]
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = src[tid - offset]
                dst[tid] = val + prev
                gpu.barrier()
                src, dst = dst, src

            base_off = carry[0]
            if in_expert:
                is_not_first = tid != 0
                excl = fx.Int32(0)
                if is_not_first:
                    excl = src[tid - 1]
                start = excl + base_off
                s_p[e] = start
                p_p[e] = start + fx.Int32(m_e)

            # Fold this chunk's total in before the next one overwrites lds0.
            chunk_total = src[MAX_EXPERTS_PER_BLOCK - 1]
            gpu.barrier()
            if is_lane0:
                carry[0] = base_off + chunk_total
            gpu.barrier()

        if is_lane0:
            total = carry[0]
            gt = total > fx.Int32(tile_v)
            c_p[0] = gt.select(total, tile_v)

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
            grid=(1, 1, 1),
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


def build_moe_contiguous_psum_remap_module():
    """JIT launcher: contiguous psum + in-place masked-to-contiguous row remap."""

    @flyc.kernel(
        name="moe_contiguous_psum_remap",
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
        # Uint32: every value here is a non-negative count/index, so `<`, `>=`
        # and `//` lower to ult/uge/divui rather than their signed forms.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_PsumStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        carry = lds.carry.ptr

        m_p = ptr_buf_tensor(masked_m)
        rows_p = ptr_buf_tensor(topids_to_rows)
        s_p = ptr_buf_tensor(starts)
        p_p = ptr_buf_tensor(psum)
        c_p = ptr_buf_tensor(contiguous_m)

        is_lane0 = tid == fx.Uint32(0)
        if is_lane0:
            carry[0] = fx.Int32(0)
        gpu.barrier()

        # Chunked scan; see psum_kernel for why E is swept in block-sized chunks.
        for base in range(0, experts, MAX_EXPERTS_PER_BLOCK):
            e = fx.Uint32(base) + tid
            in_expert = e < fx.Uint32(experts)
            m_e = fx.Uint32(0)
            if in_expert:
                m_e = fx.Uint32(m_p[e])
            lds0[tid] = fx.Int32((m_e + tile_minus_1) // tile_v * tile_v)
            gpu.barrier()

            src = lds0
            dst = lds1
            for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
                if const_expr((offset & (offset - 1)) != 0):
                    continue
                val = src[tid]
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = src[tid - offset]
                dst[tid] = val + prev
                gpu.barrier()
                src, dst = dst, src

            base_off = carry[0]
            if in_expert:
                is_not_first = tid != 0
                excl = fx.Int32(0)
                if is_not_first:
                    excl = src[tid - 1]
                start = excl + base_off
                s_p[e] = start
                p_p[e] = start + fx.Int32(m_e)

            # Fold this chunk's total in before the next one overwrites lds0.
            chunk_total = src[MAX_EXPERTS_PER_BLOCK - 1]
            gpu.barrier()
            if is_lane0:
                carry[0] = base_off + chunk_total
            gpu.barrier()

        if is_lane0:
            total = carry[0]
            gt = total > fx.Int32(tile_v)
            c_p[0] = gt.select(total, tile_v)

        gpu.barrier()

        # Only remap valid routes ([0, valid_route_count)); dead-tail routes
        # hold unwritten/garbage rows from the route kernel and must NOT be used
        # as a row index (would OOB-read starts[expert]). They are never read
        # downstream. When truncation is disabled the caller passes a null pointer
        # instead of a (1,) tensor, so the load must not run unconditionally.
        num_valid_routes_is_set = fx.Int64(ptrtoint(num_valid_routes)) != 0
        valid_route_count = fx.Uint32(numel)
        if num_valid_routes_is_set:
            valid_route_count = fx.Uint32(
                ptr_buf_tensor(num_valid_routes)[fx.Uint32(0)]
            )
        for route_i32 in range(tid, valid_route_count, MAX_EXPERTS_PER_BLOCK):
            row_raw = rows_p[route_i32]
            # An EP route with no grouped row carries the negative
            # DROPPED_ROUTE_ROW sentinel: the row math would turn it into a wild
            # expert index (OOB starts[] read), and downstream consumers check for
            # the sentinel, so leave the slot untouched.
            row_is_mapped = fx.Int32(row_raw) >= fx.Int32(0)
            if row_is_mapped:
                row = fx.Uint32(row_raw)
                m = fx.Uint32(route_max_m)
                expert = row // m
                slot = row - expert * m
                start = fx.Uint32(s_p[expert])
                rows_p[route_i32] = start + slot

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
            grid=(1, 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    launch_psum_remap.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_psum_remap


def build_moe_contiguous_psum_remap_ep_module():
    """psum + masked->contiguous remap, fused with the gemm2 EP ep_rowmap build.

    Same prefix-sum and in-place row remap as
    ``build_moe_contiguous_psum_remap_module``, but the remap pass also scatters,
    for each kept local route, the packed dest (origin_pe*slot_stride +
    origin_lid*topk + k) plus f32 weight bits into ep_rowmap[final_row], reusing
    the row it just computed. ep_rowmap is a flat (cap_rows+1, 2) i32 tensor;
    rows no kept route claims keep the -1 sentinel the host memset wrote.
    """

    @flyc.kernel(
        name="moe_contiguous_psum_remap_ep",
        known_block_size=[MAX_EXPERTS_PER_BLOCK, 1, 1],
    )
    def psum_remap_ep_kernel(
        masked_m: fx.Pointer,
        topids_to_rows: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        experts: Int32,
        route_max_m: Int32,
        tile_m: Int32,
        num_valid_routes: fx.Pointer,
        gather_w: fx.Pointer,  # (numel,) bf16, 0 for dropped/remote
        tis: fx.Pointer,  # (recv_cap,) i32 recv_slot -> origin enc
        ep_rowmap: fx.Pointer,  # (cap_rows+1, 2) i32 flat out
        topk: Int32,
        max_tok: Int32,
        slot_stride: Int32,
    ):
        # Uint32 for the same reason as psum_remap_kernel: all counts/indices.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_PsumRemapEpStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr

        m_p = ptr_buf_tensor(masked_m)
        rows_p = ptr_buf_tensor(topids_to_rows)
        s_p = ptr_buf_tensor(starts)
        p_p = ptr_buf_tensor(psum)
        c_p = ptr_buf_tensor(contiguous_m)
        w_p = ptr_buf_tensor(gather_w, fx.BFloat16)
        tis_p = ptr_buf_tensor(tis)
        ep_p = ptr_buf_tensor(ep_rowmap)

        # Lanes past ``experts`` stay out of the scan entirely: they never write
        # lds0, and an in-range lane only ever reads indices below its own, so
        # it never picks up their uninitialised slots.
        in_expert = tid < fx.Uint32(experts)
        if in_expert:
            m_e = fx.Uint32(m_p[tid])
            lds0[tid] = fx.Int32((m_e + tile_minus_1) // tile_v * tile_v)

        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            if in_expert:
                val = src[tid]
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = src[tid - offset]
                dst[tid] = val + prev
            gpu.barrier()
            src, dst = dst, src

        bid = fx.Uint32(fx.block_idx.x)
        gtid = bid * MAX_EXPERTS_PER_BLOCK + tid
        is_blk0 = bid == fx.Uint32(0)
        # Multi-block: every block keeps its exclusive per-expert starts in LDS so the
        # grid-strided scatter reads starts from LDS (never global -> no cross-block
        # barrier). Only block 0 writes the global starts/psum/contiguous_m outputs.
        starts_lds = dst  # spare ping-pong buffer now holds the exclusive starts
        if in_expert:
            is_not_first = tid != 0
            start = fx.Int32(0)
            if is_not_first:
                start = src[tid - 1]
            starts_lds[tid] = start
            if is_blk0:
                m_tid = m_p[tid]
                s_p[tid] = start
                p_p[tid] = start + fx.Int32(m_tid)
                is_last = tid == fx.Uint32(experts) - 1
                if is_last:
                    total = src[tid]
                    gt = total > fx.Int32(tile_v)
                    c_p[0] = gt.select(total, tile_v)

        gpu.barrier()

        # ep_rowmap is pre-filled with the (-1, 0) sentinel by a host-side memset
        # ordered before this launch, so the scatter below only writes kept rows.
        nvr = fx.Uint32(ptr_buf_tensor(num_valid_routes)[fx.Uint32(0)])
        topk_v = fx.Uint32(topk)
        max_tok_v = fx.Uint32(max_tok)
        # Fused remap + ep_rowmap scatter over valid routes ([0, nvr)).
        for route in range(gtid, nvr, EP_REMAP_NBLK * MAX_EXPERTS_PER_BLOCK):
            row_raw = rows_p[route]
            # A route with no grouped row carries DROPPED_ROUTE_ROW: the
            # masked->contiguous math would turn it into a wild expert index (OOB
            # LDS read of starts) and there is nothing to scatter. The route
            # kernel gives exactly the gather_w == 0 routes this sentinel.
            row_is_mapped = fx.Int32(row_raw) >= fx.Int32(0)
            if row_is_mapped:
                row = fx.Uint32(row_raw)
                m = fx.Uint32(route_max_m)
                expert = row // m
                slot = row - expert * m
                final_row = fx.Uint32(starts_lds[expert]) + slot
                rows_p[route] = final_row
                # ep_rowmap scatter for this row: packed dest + f32 weight bits.
                w_f32 = w_p[route].to(fx.Float32)
                t = fx.Uint32(route) // topk_v
                k = fx.Uint32(route) - t * topk_v
                enc = fx.Uint32(tis_p[t])
                origin_pe = enc // max_tok_v
                origin_lid = enc - origin_pe * max_tok_v
                packed = origin_pe * fx.Uint32(slot_stride) + origin_lid * topk_v + k
                ep_base = final_row * 2
                ep_p[ep_base] = packed
                ep_p[ep_base + 1] = w_f32.bitcast(fx.Int32)

    @flyc.jit
    def launch_psum_remap_ep(
        masked_m: fx.Pointer,
        topids_to_rows: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        experts: fx.Int32,
        route_max_m: fx.Int32,
        tile_m: fx.Int32,
        num_valid_routes: fx.Pointer,
        gather_w: fx.Pointer,
        tis: fx.Pointer,
        ep_rowmap: fx.Pointer,
        topk: fx.Int32,
        max_tok: fx.Int32,
        slot_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        psum_remap_ep_kernel(
            masked_m,
            topids_to_rows,
            starts,
            psum,
            contiguous_m,
            experts,
            route_max_m,
            tile_m,
            num_valid_routes,
            gather_w,
            tis,
            ep_rowmap,
            topk,
            max_tok,
            slot_stride,
        ).launch(
            grid=(EP_REMAP_NBLK, 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    launch_psum_remap_ep.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_psum_remap_ep


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
        # Uint32: every value here is a non-negative count/index, so `<`, `>=`
        # and `//` lower to ult/uge/divui rather than their signed forms.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_RoutePsumStorage).peek()
        lds_cnt = lds.cnt.ptr
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr

        topk_p = ptr_buf_tensor(topk_ids)
        rows_p = ptr_buf_tensor(topids_to_rows)
        m_p = ptr_buf_tensor(masked_m)
        s_p = ptr_buf_tensor(starts)
        p_p = ptr_buf_tensor(psum)

        in_expert = tid < fx.Uint32(experts)

        # Phase A: zero the LDS per-expert atomic counter.
        if in_expert:
            lds_cnt[tid] = fx.Int32(0)
        gpu.barrier()

        # Phase B: route + workgroup-scope LDS atomic -> masked-layout rows.
        # The atomic needs a raw addrspace(3) pointer, so the counter array's
        # base is taken as an integer here; SharedAllocator has already folded
        # its offset in, leaving only the per-expert element offset to add.
        # (fx.to_llvm_ptr would be the current spelling, but it needs a newer
        # fly dialect than the pinned LLVM build exposes.)
        cnt_base_i64 = fx.Int64(fx.ptrtoint(lds_cnt))
        numel_i32 = fx.Uint32(numel)
        for route_i32 in range(tid, numel_i32, MAX_EXPERTS_PER_BLOCK):
            e = topk_p[route_i32]
            ptr = buffer_ops.create_llvm_ptr(
                cnt_base_i64 + fx.Int64(e) * 4, address_space=3
            )
            ptr = ptr._value if hasattr(ptr, "_value") else ptr
            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                arith.constant(1, type=T.i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result
            row = fx.Uint32(slot) + fx.Uint32(e) * fx.Uint32(max_m)
            rows_p[route_i32] = row
        gpu.barrier()

        # Phase C: tile-aligned inclusive scan of per-expert counts.
        if in_expert:
            m = fx.Uint32(lds_cnt[tid])
            lds0[tid] = (m + tile_minus_1) // tile_v * tile_v
            m_p[tid] = m
        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            if in_expert:
                val = src[tid]
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = src[tid - offset]
                dst[tid] = val + prev
            gpu.barrier()
            src, dst = dst, src

        if in_expert:
            is_not_first = tid != 0
            start = fx.Int32(0)
            if is_not_first:
                start = src[tid - 1]
            m_tid = lds_cnt[tid]
            s_p[tid] = start
            p_p[tid] = start + m_tid
        gpu.barrier()

        # Phase D: in-place masked -> contiguous row remap.
        for route_i32 in range(tid, numel_i32, MAX_EXPERTS_PER_BLOCK):
            row = fx.Uint32(rows_p[route_i32])
            m = fx.Uint32(max_m)
            expert = row // m
            slot = row - expert * m
            start = fx.Uint32(s_p[expert])
            rows_p[route_i32] = start + slot

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
            grid=(1, 1, 1),
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

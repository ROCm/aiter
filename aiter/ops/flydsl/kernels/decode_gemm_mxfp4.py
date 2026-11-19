# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 decode A4W4: resident A, register-pipelined B, ordered split-K sum.

Scales retain the asm N32/K256 shuffle; each dword contains two K128
halves and two N16 halves. Split-K writes fp32 planes for a separate reduction.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from aiter.ops.flydsl.kernels.kernels_common import ceildiv

# idx2crd wraps every mode; this extent keeps the outer stage non-wrapping.
_UNBOUNDED_STAGE_EXTENT = 2**31 - 1


@lru_cache(maxsize=128)
def compile_decode_gemm_mxfp4(
    *, M, N, K, tile_n=32, k_waves=2, num_buffers=4, split_k=1
):
    n_waves = tile_n // 16
    total_tiles = ceildiv(K, 256)
    k_tiles = ceildiv(total_tiles, split_k)
    padded_k = k_tiles * 256
    group_tiles = ceildiv(k_tiles, k_waves)
    depth = min(num_buffers, group_tiles)

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.Int32, M * padded_k // 8, 16]
        scales: fx.Array[fx.Int32, padded_k // 4, 16]
        partial: fx.Array[fx.Float32, 16 * tile_n if k_waves == 2 else 1, 16]

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, AS: fx.Tensor, BS: fx.Tensor, C: fx.Tensor):
        tid = fx.Int32(fx.gpu.thread_id("x"))
        bid = fx.Int32(fx.gpu.block_id("x"))
        split = fx.Int32(0)
        slice_start = fx.Int32(0)
        slice_tiles = fx.Int32(k_tiles)
        if const_expr(split_k > 1):
            # Give the first remainder slices one extra K256 tile. LDS and
            # the pipeline use the maximum size; shorter slices mask B.
            split = fx.Int32(fx.gpu.block_id("y"))
            extra = total_tiles % split_k
            slice_start = split * (total_tiles // split_k) + (split < extra).select(
                split, fx.Int32(extra)
            )
            slice_tiles = fx.Int32(total_tiles // split_k) + (split < extra).select(
                fx.Int32(1), fx.Int32(0)
            )
        lane = tid % 64
        wave = tid // 64
        nw = wave % n_waves
        kg = wave // n_waves
        col_group = bid * n_waves + nw
        row = lane % 16
        # Dead lanes broadcast a live row from their eight-row service group.
        fallback_row = (row // 8) * 8 if const_expr(M > 8) else fx.Int32(0)
        safe_row = (row < M).select(row, fallback_row)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # A coordinates are (dword, K32 group, row, K256 stage); each dword
        # holds eight FP4 values. Swizzling excludes stage, including for odd M.
        a_coord = fx.make_layout(
            (4, 8, M, _UNBOUNDED_STAGE_EXTENT),
            (fx.E(0), 4 * fx.E(0), fx.E(1), fx.E(2)),
        )
        a_swizzle = fx.static(fx.CoordSwizzleType.get(3, 0, [1], 2, [0]))
        a_coord = fx.make_composed_layout(a_swizzle, a_coord)
        a_storage = fx.make_composed_layout(
            fx.make_layout((32, M, k_tiles), (1, 32, M * 32)), a_coord
        )
        a_lds = lds.a.view(a_storage)
        a_linear = lds.a.view(fx.make_layout(M * padded_k // 8, 1))
        a_gather_layout = fx.make_layout((8, M, _UNBOUNDED_STAGE_EXTENT), (1, 8, M * 8))
        as_lds = lds.scales.view(fx.make_layout(padded_k // 4, 1))
        partial = lds.partial.view(fx.make_layout((16, tile_n), (tile_n, 1)))

        def buffer(t, elem, size):
            return fx.make_view(
                fx.recast_iter(
                    elem, fx.get_iter(fx.rocdl.make_buffer_tensor(t, max_size=False))
                ),
                fx.make_layout(size, 1),
            )

        ga = buffer(A, fx.Int32, M * K // 8)
        a_source_layout = fx.make_layout((M, 32, total_tiles), (K // 8, 1, 32))
        gas = fx.make_view(
            fx.add_offset(
                fx.get_iter(buffer(AS, fx.Int32, total_tiles * 64)), slice_start * 64
            ),
            fx.make_layout((1, 256, ceildiv(padded_k // 4, 256)), (1, 1, 256)),
        )
        gb = buffer(B, fx.Int32, N * K // 8)
        b_stage_coord = fx.make_layout((N // 16, K // 128), (K // 128, 1))
        bs_stage_coord = fx.make_layout((N // 32, total_tiles), (total_tiles, 1))
        # Scale mode order keeps compile-time stage terms in immediates;
        # the coalesced K128 and unpadded K64 paths need different ordering.
        if const_expr(K % 128 == 0):
            b_view = fx.make_view(
                fx.get_iter(gb),
                fx.make_layout((4, (K // 128) * (N // 16), 64), (1, 256, 4)),
            )
            gbs = fx.make_view(
                fx.get_iter(buffer(BS, fx.Int32, N * total_tiles * 2)),
                fx.make_layout((1, total_tiles * (N // 32), 64), (1, 64, 1)),
            )
            as_view = fx.make_view(
                fx.get_iter(as_lds), fx.make_layout((1, k_tiles, 64), (1, 64, 1))
            )
        else:
            # K64 tails retain their unpadded N-tile packing.
            b_tiles = fx.make_view(
                fx.get_iter(gb),
                fx.make_layout((K * 2, N // 16), (1, K * 2)),
            )
            b_view = fx.make_view(
                fx.get_iter(fx.slice(b_tiles, (None, col_group))),
                fx.make_layout((4, 64, ceildiv(K, 128)), (1, 4, 256)),
            )
            scale_tiles = fx.make_view(
                fx.get_iter(buffer(BS, fx.Int32, N * total_tiles * 2)),
                fx.make_layout((total_tiles * 64, N // 32), (1, total_tiles * 64)),
            )
            gbs = fx.make_view(
                fx.get_iter(fx.slice(scale_tiles, (None, col_group // 2))),
                fx.make_layout((1, 64, total_tiles), (1, 1, 64)),
            )
            as_view = fx.make_view(
                fx.get_iter(as_lds), fx.make_layout((1, 64, k_tiles), (1, 1, 64))
            )
        out_elem = fx.Float32 if const_expr(split_k > 1) else fx.BFloat16
        gc = fx.make_view(
            fx.get_iter(buffer(C, out_elem, split_k * M * N)),
            fx.make_layout((split_k, M, N), (M * N, N, 1)),
        )
        load128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        load32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        copy128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        copy32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        float_copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        store16 = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy32b()
                if const_expr(split_k > 1)
                else fx.rocdl.BufferCopy16b()
            ),
            out_elem,
        )

        def tile(view, offset, width):
            return fx.make_view(
                fx.add_offset(fx.get_iter(view), offset), fx.make_layout(width, 1)
            )

        # Gather A in K256/swizzled order so the LDS writes remain contiguous.
        # Keep A older than B in VMEM order so its LDS stores only wait for A.
        # Bounded buffer descriptors make the cooperative tail loads safe.
        a_regs = []
        as_regs = []
        for pos in range_constexpr(ceildiv(M * padded_k // 32, 256)):
            r = fx.make_rmem_tensor(4, fx.Int32)
            idx = (pos * 256 + tid) * 4
            gather = fx.idx2crd(pos * 256 + tid, a_gather_layout)
            a_row = fx.Int32(fx.get(gather, 1))
            a_stage = fx.Int32(fx.get(gather, 2))
            source_coord = fx.crd2idx((0, fx.get(gather, 0), a_row, a_stage), a_coord)
            a_k = fx.Int32(fx.get(source_coord, 0))
            src = fx.get_scalar(fx.crd2idx((a_row, a_k, a_stage), a_source_layout))
            if const_expr(K % 256 != 0):
                # Each load is one complete K32 group, including in the tail.
                valid = (idx < M * padded_k // 8) & (a_stage * 32 + a_k < K // 8)
                src = valid.select(src, fx.Int32(M * K // 8))
            if const_expr(split_k > 1):
                global_stage = slice_start + a_stage
                src = fx.get_scalar(
                    fx.crd2idx((a_row, a_k, global_stage), a_source_layout)
                )
                valid = (
                    (idx < M * padded_k // 8)
                    & (a_stage < slice_tiles)
                    & (global_stage * 32 + a_k < K // 8)
                )
                src = valid.select(src, fx.Int32(M * K // 8))
            if const_expr(K % 256 == 0 and split_k == 1):
                src = (idx < M * K // 8).select(src, fx.Int32(M * K // 8))
            fx.copy(load128, tile(ga, src, 4), r)
            a_regs.append(r)
        for pos in range_constexpr(ceildiv(padded_k // 4, 256)):
            r = fx.make_rmem_tensor(1, fx.Int32)
            idx = pos * 256 + tid
            fx.copy(load32, fx.slice(gas, (None, tid, pos)), r)
            as_regs.append(r)
        # Keeps the A staging loads ahead of the B-ring prefetch issued next, so
        # the scheduler cannot move either across the LDS barrier (11.42% at
        # M1/N7168/K5120).
        fx.rocdl.sched_barrier(0)

        def safe_scales(stage, scales):
            if const_expr(K % 256 != 0):
                # A dword packs both N16 halves for two K128 halves.
                for half in range_constexpr(2):
                    valid = (slice_start + stage) * 256 + half * 128 + (
                        lane // 16
                    ) * 32 < K
                    mask = fx.Int32(0xFFFF << (half * 16))
                    neutral = fx.Int32(0x7F7F << (half * 16))
                    scales = valid.select(scales, (scales & ~mask) | neutral)
            return scales

        def prefetch(local_stage, mask=True):
            global_stage = kg * group_tiles + local_stage
            safe_stage = (global_stage < slice_tiles).select(
                global_stage, slice_tiles - 1
            )
            keep = (global_stage < slice_tiles).select(fx.Int32(-1), fx.Int32(0))
            safe_stage = safe_stage + slice_start
            words = []
            # MMA-derived FP4 B partitions add lane-address arithmetic even after
            # dword recasting (+2 VGPRs for M16/N5120/K8704 split-K4).
            for half in range_constexpr(2):
                if const_expr(K % 128 == 0):
                    b_stage = fx.get_scalar(
                        fx.crd2idx((col_group, safe_stage * 2), b_stage_coord)
                    )
                    b_half = b_stage + half
                    source = fx.slice(b_view, (None, b_half, lane))
                else:
                    source = fx.slice(b_view, (None, lane, safe_stage * 2 + half))
                if const_expr(K % 256 != 0):
                    # B consists of K64 tiles, not padded K128/K256 tiles.
                    valid = safe_stage * 256 + half * 128 + (lane // 16) * 32 < K
                    source = fx.make_view(
                        fx.add_offset(
                            fx.get_iter(source),
                            valid.select(fx.Int32(0), fx.Int32(N * K // 8)),
                        ),
                        fx.make_layout(4, 1),
                    )
                r = fx.make_rmem_tensor(4, fx.Int32)
                fx.copy(load128, source, r)
                word = r.load()
                if const_expr(mask):
                    word = word & fx.Vector.filled(4, keep, fx.Int32)
                words.append(word)
            sr = fx.make_rmem_tensor(1, fx.Int32)
            if const_expr(K % 128 == 0):
                bs_stage = fx.get_scalar(
                    fx.crd2idx((col_group // 2, safe_stage), bs_stage_coord)
                )
                fx.copy(load32, fx.slice(gbs, (None, bs_stage, lane)), sr)
            else:
                fx.copy(load32, fx.slice(gbs, (None, lane, safe_stage)), sr)
            return words + [safe_scales(safe_stage - slice_start, sr.load()[0])]

        # Defer B masking until after the barrier to avoid consuming its loads.
        ring = []
        for s in range_constexpr(depth):
            ring += prefetch(fx.Int32(s), mask=False)
        for pos in range_constexpr(ceildiv(M * padded_k // 32, 256)):
            idx = (pos * 256 + tid) * 4
            if idx < M * padded_k // 8:
                fx.copy(copy128, a_regs[pos], tile(a_linear, idx, 4))
        for pos in range_constexpr(ceildiv(padded_k // 4, 256)):
            idx = pos * 256 + tid
            if idx < padded_k // 4:
                fx.copy(copy32, as_regs[pos], tile(as_lds, idx, 1))
        fx.gpu.barrier()
        for s in range_constexpr(depth):
            keep = (kg * group_tiles + s < slice_tiles).select(
                fx.Int32(-1), fx.Int32(0)
            )
            for half in range_constexpr(2):
                ring[s * 3 + half] = ring[s * 3 + half] & fx.Vector.filled(
                    4, keep, fx.Int32
                )

        def prefetch_a(local_stage):
            stage = kg * group_tiles + local_stage
            safe_stage = (stage < slice_tiles).select(stage, slice_tiles - 1)
            words = []
            for half in range_constexpr(2):
                ar = fx.make_rmem_tensor(4, fx.Int32)
                # The composed-view slice splits shift/add address generation
                # and changes the ds_read schedule (+2 VGPRs for M16 split-K4).
                # Keep the flat form of a_storage to preserve that lowering.
                chunk = (half * 4 + lane // 16) ^ (safe_row % 8)
                idx = safe_stage * (M * 32) + safe_row * 32 + chunk * 4
                fx.copy(copy128, tile(a_linear, idx, 4), ar)
                words.append(ar.load())
            sr = fx.make_rmem_tensor(1, fx.Int32)
            if const_expr(K % 128 == 0):
                fx.copy(copy32, fx.slice(as_view, (None, safe_stage, lane)), sr)
            else:
                fx.copy(copy32, fx.slice(as_view, (None, lane, safe_stage)), sr)
            return words + [safe_scales(safe_stage, sr.load()[0])]

        mma = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float4E2M1FN))
        tiled_mma = fx.make_tiled_mma(mma, fx.make_layout((1, 1, 1), (1, 1, 1)))
        a_shape = fx.make_view(
            fx.recast_iter(fx.Float4E2M1FN, fx.get_iter(a_lds)),
            fx.make_layout((16, 128), (128, 1)),
        )
        b_shape = fx.make_view(
            fx.recast_iter(fx.Float4E2M1FN, fx.get_iter(gb)),
            fx.make_layout((16, 128), (128, 1)),
        )
        c_shape = fx.make_view(
            fx.get_iter(partial),
            fx.make_layout((16, 16), (16, 1)),
        )

        def compute(acc, b0, b1, bs, a0, a1, sa):
            cr = tiled_mma.make_fragment_C(c_shape)
            cr.store(acc)
            for half in range_constexpr(2):
                av = [a0, a1][half] & fx.Vector.filled(
                    4, (row < M).select(fx.Int32(-1), fx.Int32(0)), fx.Int32
                )
                af = tiled_mma.make_fragment_A(a_shape)
                bf = tiled_mma.make_fragment_B(b_shape)
                af.store(av.bitcast(fx.Float4E2M1FN))
                bf.store([b0, b1][half].bitcast(fx.Float4E2M1FN))
                # N-half is wave-dependent, so select that byte with a shift;
                # opsel selects the compile-time K128 half of the packed dword.
                atom = fx.make_mma_atom(
                    fx.rocdl.cdna4.MFMA_Scale(
                        16, 16, 128, fx.Float4E2M1FN, opsel_a=half * 2, opsel_b=half * 2
                    )
                )
                fx.gemm(
                    atom,
                    cr,
                    af,
                    bf,
                    cr,
                    scale_a=sa,
                    scale_b=bs >> ((col_group % 2) * 8),
                )
            return cr.load()

        initial_acc = tiled_mma.make_fragment_C(c_shape)
        initial_acc.fill(0.0)
        initial = [initial_acc.load()] + prefetch_a(fx.Int32(0)) + ring
        for iv, state in range(
            fx.Int32(0), fx.Int32(group_tiles - depth), fx.Int32(1), init=initial
        ):
            next_b = prefetch(fx.Int32(iv) + depth)
            next_a = prefetch_a(fx.Int32(iv) + 1)
            # Removing this costs 5.51% at M16/N5120/K3072.
            fx.rocdl.sched_barrier(0)
            acc = compute(state[0], state[4], state[5], state[6], *state[1:4])
            results = yield [acc] + next_a + list(state[7:]) + next_b
        acc = results[0]
        current_a = list(results[1:4])
        for s in range_constexpr(depth):
            next_a = current_a
            if const_expr(s + 1 < depth):
                next_a = prefetch_a(fx.Int32(group_tiles - depth + s + 1))
                # Removing this costs 5.88% at M8/N5120/K8704.
                fx.rocdl.sched_barrier(0)
            acc = compute(
                acc,
                results[4 + s * 3],
                results[5 + s * 3],
                results[6 + s * 3],
                *current_a,
            )
            current_a = next_a

        epilogue_mma = fx.make_tiled_mma(
            mma, fx.make_layout((1, n_waves, 1), (0, 1, 0))
        )
        if const_expr(k_waves == 2):
            if kg == 1:
                c_copy = fx.make_tiled_copy_C(float_copy, epilogue_mma).get_slice(
                    (row, lane // 16, nw)
                )
                partial_c = c_copy.partition_D(partial)
                for i in range_constexpr(4):
                    r = fx.make_rmem_tensor(1, fx.Float32)
                    r.store(fx.Vector.from_elements([acc[i]], fx.Float32))
                    fx.copy(float_copy, r, fx.slice(partial_c, ((None, i), 0, 0)))
            fx.gpu.barrier()
        if kg == 0:
            c_copy = fx.make_tiled_copy_C(float_copy, epilogue_mma).get_slice(
                (row, lane // 16, nw)
            )
            # The tuple's N coordinate is global so the block and wave terms
            # stay coalesced, preserving the B-prefetch schedule and immediates.
            out_copy = fx.make_tiled_copy_C(store16, epilogue_mma).get_slice(
                (row, lane // 16, col_group)
            )
            c_rows = c_copy.partition_S(
                fx.make_view(0, fx.make_layout((16, tile_n), (1, 0)))
            )
            out_tile = fx.make_view(
                fx.get_iter(fx.slice(gc, (split, None, None))),
                fx.make_layout((16, N), (N, 1)),
            )
            out_c = out_copy.partition_D(out_tile)
            if const_expr(k_waves == 2):
                partial_c = c_copy.partition_S(partial)
            for i in range_constexpr(4):
                out_row = fx.get_scalar(c_rows[((0, i), 0, 0)])
                value = acc[i]
                if const_expr(k_waves == 2):
                    r = fx.make_rmem_tensor(1, fx.Float32)
                    fx.copy(float_copy, fx.slice(partial_c, ((None, i), 0, 0)), r)
                    value = value + r.load()[0]
                if out_row < M:
                    out_reg = fx.make_rmem_tensor(1, out_elem)
                    out_reg.store(
                        fx.Vector.from_elements([value.to(out_elem)], out_elem)
                    )
                    fx.copy(store16, out_reg, fx.slice(out_c, ((None, i), 0, 0)))

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        AS: fx.Tensor,
        BS: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(A, B, AS, BS, C).launch(
            grid=(N // tile_n, split_k), block=(256,), stream=stream
        )

    return launch

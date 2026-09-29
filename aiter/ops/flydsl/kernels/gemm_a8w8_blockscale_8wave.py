# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B008
# The FlyDSL body uses a typed Stream default for its tracing semantics.
# Ported from pyhip a3a94c5a34fc525c118649418b76221cd6d91579, fixed to raw DMA.
"""gfx950 FP8 GEMM: C = A @ B.T, FP32 accumulation and BF16 output.

The blockscale contract is ScaleA[KB, M] and ScaleB[ceil(N/128), KB], KB=K/128.
The kernel uses a 256x256 WG tile, eight waves, and ping-pong padded LDS.
Optional preshuffle_b consumes aiter.shuffle_weight(B, layout=(16, 16)) directly;
B uses packed LDS tiles in that mode, while both scale layouts stay unchanged.
Only the half-M single-FIFO pipeline is supported: each wave's 64x32
quadrant is computed as two 32x32 M slices sharing A and partial registers.
Raw DMA loads A and B directly into LDS; tiled DMA is not supported.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Float8E4M3FN, Float32, T
from flydsl.expr.typing import Vector as Vec


def div_up(x, y):
    return (x + y - 1) // y


def _buffer_resource(tensor, num_records_bytes):
    buffer = rocdl.make_buffer_tensor(tensor, num_records_bytes=num_records_bytes)
    return rocdl.get_buffer_rsrc(fx.get_iter(buffer))


def _lds_byte_ptr(ptr, byte_offset):
    return fx.to_llvm_ptr(
        fx.add_offset(fx.recast_iter(fx.Uint8, ptr), fx.make_int_tuple(byte_offset))
    )


def encode_waitcnt_950(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def compile_gemm_fp8_8wave(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    pid_swizzle=True,
    permlane_epilogue=True,
    preshuffle_b=False,
):
    """Compile the half-M single-FIFO blockscale kernel with raw A/B DMA.

    Scalar B scales, one-phase-ahead B_l reads, and four scalar FMAs per new
    MFMA retain the validated split pipeline. There are no s_setprio changes.
    preshuffle_b uses the FP8 (16, 16) weight shuffle, without gate/up interleave.
    Its coalesced raw B DMA retains the plain path's load count and scheduling.
    B scales use scalar buffer loads directly, without LDS staging.
    """
    assert (TILE_M, TILE_N, TILE_K) == (256, 256, 128)
    if preshuffle_b:
        assert N % 16 == 0 and K % 256 == 0
    BLOCK_M = TILE_M // 2
    BLOCK_N = TILE_N // 2
    BLOCK_K = TILE_K
    # BLOCK_M is split into two halves to reduce A VGPRs.
    M_SLICES = 2
    PHASE_M_REP = BLOCK_M // 32 // M_SLICES
    assert N % 8 == 0
    # Only descriptor-relative WG spans, including speculative prefetches,
    # must fit signed i32. Whole matrices may extend past 4 GiB.
    assert TILE_M * K + 2 * TILE_K < 2**31
    assert TILE_N * K + 2 * TILE_K * (16 if preshuffle_b else 1) < 2**31
    assert TILE_M * N * 2 < 2**31
    element_type = fx.Float8E4M3FN
    elements_per_128b = 16  # 128bit / fp8(8bit)
    scaleA_stride = K // 128
    scaleB_rows = TILE_N // 128
    scaleB_elems = scaleB_rows * scaleA_stride

    def _get_pids_950(pid, M, GRID_MN, NUM_XCDS, GROUP_SIZE_M):
        num_pid_m = (M + TILE_M - 1) // TILE_M
        num_pid_n = div_up(N, TILE_N)
        if const_expr(NUM_XCDS != 1):
            pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
            tall_xcds = GRID_MN % NUM_XCDS
            tall_xcds = (tall_xcds == 0).select(NUM_XCDS, tall_xcds)
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            if xcd < tall_xcds:
                pid = xcd * pids_per_xcd + local_pid
            else:
                pid = (
                    tall_xcds * pids_per_xcd
                    + (xcd - tall_xcds) * (pids_per_xcd - 1)
                    + local_pid
                )
        if const_expr(GROUP_SIZE_M == 1):
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            remaining_pid_m = num_pid_m - first_pid_m
            group_size_m = (remaining_pid_m < GROUP_SIZE_M).select(
                remaining_pid_m, GROUP_SIZE_M
            )
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    get_pids_950 = ASTRewriter.transform(_get_pids_950)

    # Pad each eight rows by 16 bytes and each sixteen rows by another 32.
    A_GROUP = 8 * BLOCK_K + 16
    a_lds_elems = (BLOCK_M // 16) * (2 * A_GROUP + 32)

    # A/B ping-pong tiles use 132 KiB; ScaleA uses another 4 KiB.
    # ScaleB is wave-uniform and stays on the scalar global-memory path.
    @fx.struct
    class LDS:
        a_t0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_l0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_l1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_r0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_r1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        # scale a ping-pong LDS
        scale_a0: fx.Array[Float32, 512, 4]
        scale_a1: fx.Array[Float32, 512, 4]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gemm_kernel(
        argA: fx.Tensor,
        argB: fx.Tensor,
        argC: fx.Tensor,
        argScaleA: fx.Tensor,
        argScaleB: fx.Tensor,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        wave_id = tid // 64
        num_pid_n = div_up(N, TILE_N)
        if const_expr(pid_swizzle):
            # Match launch grid; the restricted target prevents OCKL inlining.
            grid_mn = div_up(M, TILE_M) * num_pid_n
            bid_x, bid_y = get_pids_950(fx.block_idx.x, M, grid_mn, 8, 4)
        else:
            bid_x = fx.block_idx.x // num_pid_n
            bid_y = fx.block_idx.x % num_pid_n

        # Rebase global pointers in i64 before building buffer descriptors.
        # All DMA/store offsets below are local to this WG; physical row
        # strides and the ScaleA/ScaleB descriptors remain unchanged.
        a_iter = fx.add_offset(
            fx.recast_iter(element_type, fx.get_iter(argA)),
            fx.Int64(bid_x) * TILE_M * K,
        )
        b_iter = fx.add_offset(
            fx.recast_iter(element_type, fx.get_iter(argB)),
            fx.Int64(bid_y) * TILE_N * K,
        )
        c_iter = fx.add_offset(
            fx.get_iter(argC),
            fx.Int64(bid_x) * TILE_M * N + fx.Int64(bid_y) * TILE_N,
        )
        a_rows_left = M - bid_x * TILE_M
        b_rows_left = N - bid_y * TILE_N
        a_rows = (a_rows_left < TILE_M).select(a_rows_left, TILE_M)
        b_rows = (b_rows_left < TILE_N).select(b_rows_left, TILE_N)
        A_2d = fx.Tensor(fx.make_view(a_iter, fx.make_layout((TILE_M, K), (K, 1))))
        B_2d = fx.Tensor(fx.make_view(b_iter, fx.make_layout((TILE_N, K), (K, 1))))
        C_2d = fx.Tensor(fx.make_view(c_iter, fx.make_layout((TILE_M, TILE_N), (N, 1))))

        A = rocdl.make_buffer_tensor(A_2d, num_records_bytes=a_rows * K)
        B = rocdl.make_buffer_tensor(B_2d, num_records_bytes=b_rows * K)
        # Include row-stride gaps, but stop at the last valid row/column.
        C = rocdl.make_buffer_tensor(
            C_2d, num_records_bytes=((a_rows - 1) * N + b_rows) * 2
        )
        a_dma_rsrc = rocdl.get_buffer_rsrc(fx.get_iter(A))
        b_dma_rsrc = rocdl.get_buffer_rsrc(fx.get_iter(B))

        ### all needed copy atom
        buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
        lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)

        ### 分配LDS
        lds = fx.SharedAllocator().allocate(LDS).peek()

        # ScaleB still needs a global descriptor for the two scalar K-tile loads.
        scaleB_rsrc = _buffer_resource(
            argScaleB,
            num_records_bytes=arith._to_raw(
                fx.Int32(div_up(N, 128) * scaleA_stride * 4)
            ),
        )

        ### read LDS layout and write LDS layout. AC copy tile
        # Write groups (8,2,8) become read groups (2,8,8) without moving data.
        _wr = fx.make_layout(
            ((8, 2, BLOCK_M // 16), BLOCK_K),
            ((BLOCK_K, 8 * BLOCK_K + 16, 2 * (8 * BLOCK_K + 16) + 32), 1),
        )
        _rd = fx.make_layout(
            ((2, BLOCK_M // 16, 8), (32, BLOCK_K // 32)),
            ((8 * BLOCK_K + 16, 2 * (8 * BLOCK_K + 16) + 32, BLOCK_K), (1, 32)),
        )
        ldsA_t_wr = [fx.make_view(lds.a_t0.ptr, _wr), fx.make_view(lds.a_t1.ptr, _wr)]
        ldsA_b_wr = [fx.make_view(lds.a_b0.ptr, _wr), fx.make_view(lds.a_b1.ptr, _wr)]
        ldsA_t_rd = [fx.make_view(lds.a_t0.ptr, _rd), fx.make_view(lds.a_t1.ptr, _rd)]
        ldsA_b_rd = [fx.make_view(lds.a_b0.ptr, _rd), fx.make_view(lds.a_b1.ptr, _rd)]
        if const_expr(preshuffle_b):
            # Logical B[n,k] -> [n//16, k//16, n%16, k%16] in packed LDS.
            # k_perm still assigns each MFMA lane the same contiguous 32 K values.
            _b_rd = fx.make_layout(
                ((16, BLOCK_N // 16), (16, BLOCK_K // 16)),
                ((16, BLOCK_K * 16), (1, 256)),
            )
        else:
            _b_rd = _rd
        ldsB_l_wr = [fx.make_view(lds.b_l0.ptr, _wr), fx.make_view(lds.b_l1.ptr, _wr)]
        ldsB_r_wr = [fx.make_view(lds.b_r0.ptr, _wr), fx.make_view(lds.b_r1.ptr, _wr)]
        ldsB_l_rd = [
            fx.make_view(lds.b_l0.ptr, _b_rd),
            fx.make_view(lds.b_l1.ptr, _b_rd),
        ]
        ldsB_r_rd = [
            fx.make_view(lds.b_r0.ptr, _b_rd),
            fx.make_view(lds.b_r1.ptr, _b_rd),
        ]

        # MMA computes transposed C: B is operand A, A is operand B.
        # The (4,2) MMA wave grid therefore partitions four N and two M groups.
        mma_atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, element_type)
        )
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
        ### MFMA instruction spec with scale:k_perm is fx.make_layout(((16, 2), 4), ((1, 64), 16)), spec里面每条lane 处理32个K，32个K 分两段连续。每段16个K连续。
        ### 这里每条lane处理连续得32个K，没有follow spec,但是结果是一样， A， B做同样的permute.
        ### 每个MFMA按照spec可以理解把A分解为：   A [16m, 128k] -> [16m, [2k2, 4k1, 16k0]]
        ### 目前的permute相当于分解后做了一次permute：A.view(16, 2, 4, 16).permute(0, 2, 1, 3).contineous().view(16,128)
        ### A*B = A.view(16, 2, 4, 16).permute(0, 2, 1, 3).contineous().view(16,128) * B.view(16, 2, 4, 16).permute(0, 2, 1, 3).contineous().view(16,128)
        k_perm = fx.make_layout((32, 4), (1, 32))
        # 8waves: 4 waves on MMA 'logical M' dimension(operand A) and 2 waves  on MMA 'logical N' dimension(operand B)
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((4, 2, 1), (1, 4, 0)), (None, None, k_perm)
        )
        # 8waves: tensor A is the MFMA operand B, tensor B is the MFMA operand A.
        # physical A is operand B and physical B is operand A.
        # In physical world, 2 waves on real M(logical N) dimension and 4 waves on real N(logical M) dimension.
        copy_a = fx.make_tiled_copy_B(lds_copy_atom, tiled_mma).get_slice(tid)
        copy_b = fx.make_tiled_copy_A(lds_copy_atom, tiled_mma).get_slice(tid)
        s2r_src0_B_l = copy_b.partition_S(ldsB_l_rd[0])
        s2r_src0_B_r = copy_b.partition_S(ldsB_r_rd[0])
        s2r_src1_B_l = copy_b.partition_S(ldsB_l_rd[1])
        s2r_src1_B_r = copy_b.partition_S(ldsB_r_rd[1])

        thr_mma = tiled_mma.thr_slice(tid)

        def _a_m_slice_view(src, m_slice):
            return fx.flat_divide(src, (BLOCK_M // M_SLICES, BLOCK_K))[
                None, None, m_slice, 0
            ]

        # Both M slices reuse 16 A dwords per lane. The WG and full C
        # fragments stay 256x256.
        frag_A_t = thr_mma.make_fragment_B(_a_m_slice_view(ldsA_t_rd[0], 0))
        frag_B_l = thr_mma.make_fragment_A(ldsB_l_rd[0])
        frag_B_r = thr_mma.make_fragment_A(ldsB_r_rd[0])

        dest_frag_A_t = copy_a.retile(frag_A_t)
        dest_frag_B_l = copy_b.retile(frag_B_l)
        dest_frag_B_r = copy_b.retile(frag_B_r)

        # Four full accumulators survive both M slices; only the partial FIFO shrinks.
        bC_tl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, 0, 0]
        bC_tl = fx.composition(
            bC_tl, fx.make_ordered_layout((BLOCK_N, BLOCK_M), (1, 0))
        )

        frag_C_tl = thr_mma.make_fragment_C(bC_tl)
        frag_C_tr = thr_mma.make_fragment_C(bC_tl)
        frag_C_bl = thr_mma.make_fragment_C(bC_tl)
        frag_C_br = thr_mma.make_fragment_C(bC_tl)
        # frag_P VGPRs is half of frag_C_tl VGPRs.
        c_slice = fx.flat_divide(bC_tl, (BLOCK_N, BLOCK_M // M_SLICES))[
            None, None, 0, 0
        ]
        frag_P = thr_mma.make_fragment_C(c_slice)  # One 16-f32 FIFO.

        # ==== A/B block-scale 设置：A per-token group-128，B per-128 rows/group-128 ====
        # C[m,n] = sum_kb scaleA[m,kb] * scaleB[n//128,kb] * partial[kb]。
        # scaleA [KB, M] 以 f32 写入 ping-pong LDS，计算 phase 按当前 MFMA 行读回。
        # C fragment 布局 [val=N, n0(N_REP), m0(M_REP)]；M 行 = quadrant_m*128 + m0*32
        #   + wave_m*16 + lane%16（wave_m=wave_id//4）=> scaleA 随 m0/lane 变化，广播 val/n0。
        N_REP = BLOCK_N // 64
        ### Ascale layout: groups = K //128, [groups, M//256, 256m]
        ### 是一次kiter 256 rows 只需要256个scale(atop+abottom)就够了。
        ### 每个lane读一个dword, 512个lane读取512个元素， 512个元素前后得256指向相同得A scale, 所以只有256个A scale.
        scaleA_rsrc = _buffer_resource(
            argScaleA,
            num_records_bytes=arith._to_raw(fx.Int32(M * scaleA_stride * 4)),
        )
        lane_id = tid % 64
        wave_m = wave_id // 4
        scale_a_lds = [
            fx.make_view(lds.scale_a0.ptr, fx.make_layout(512, 1)),
            fx.make_view(lds.scale_a1.ptr, fx.make_layout(512, 1)),
        ]
        scale_lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        scale_wave_dst_offset = rocdl.readfirstlane(
            T.i32, arith._to_raw(fx.Int32(tid * 4))
        )

        def _scale_dst_ptr(root_view, byte_offset):
            return _lds_byte_ptr(fx.get_iter(root_view), byte_offset)

        scale_row = tid % TILE_M
        scale_lane_src_offset = fx.Int32(scale_row * 4)
        scale_src_tile_base = fx.Int32(bid_x * TILE_M * 4)

        def _ac_scale_a(buf, kb):
            scale_dst = _scale_dst_ptr(scale_a_lds[buf], scale_wave_dst_offset)
            rocdl.raw_ptr_buffer_load_lds(
                scaleA_rsrc,
                scale_dst,
                fx.Int32(4),
                scale_lane_src_offset,
                fx.Int32(scale_src_tile_base + kb * M * 4),
                fx.Int32(0),
                fx.Int32(0),
            )

        def load_scale_b(kb):
            addr = fx.Int32((bid_y * scaleB_elems + kb) * 4)
            result_type = ir.Type.parse("!llvm.struct<(f32, f32)>")
            result = _llvm.inline_asm(
                result_type,
                [
                    arith._to_raw(scaleB_rsrc),
                    arith._to_raw(addr),
                    arith._to_raw(addr + scaleA_stride * 4),
                ],
                "s_buffer_load_dword $0, $2, $3\n" "s_buffer_load_dword $1, $2, $4",
                "=&s,=&s,s,s,s,~{memory}",
                has_side_effects=True,
            )
            return Vec.from_elements(
                [
                    fx.Float32(_llvm.extractvalue(T.f32, result, [0])),
                    fx.Float32(_llvm.extractvalue(T.f32, result, [1])),
                ],
                fx.Float32,
            )

        def lds_rd_scale_a(buf, bottom, m_slice=0):
            half_offset = bottom * BLOCK_M
            wave_copy_offset = wave_m * TILE_M
            scales = []
            for m0 in range_constexpr(PHASE_M_REP):
                scale_offset = (
                    wave_copy_offset
                    + half_offset
                    + wave_m * 16
                    + lane_id % 16
                    + (m_slice * PHASE_M_REP + m0) * 32
                )
                scale_src = fx.make_view(
                    fx.add_offset(
                        lds.scale_a0.ptr if buf == 0 else lds.scale_a1.ptr,
                        scale_offset,
                    ),
                    fx.make_layout(1, 1),
                )
                scale_frag = fx.make_fragment_like(scale_src)
                fx.copy(scale_lds_copy_atom, scale_src, scale_frag)
                scales.append(Vec(scale_frag.load())[0])
            return Vec.from_elements(scales, fx.Float32)

        def do_gemm(
            frag_C,
            frag_B,
            frag_A,
            prev_scale_a,
            prev_scale_b,
            prev_m_slice,
        ):
            # Half-M pipeline: M_SLICES=2, PHASE_M_REP=2, N_REP=2.
            # Per-lane fragment shapes for the 256x256x128 WG:
            #   frag_A [Kval, Mrep, Krep]: (32,2,1) fp8 -> 16 dwords
            #   frag_B [Kval, Nrep, Krep]: (32,2,1) fp8 -> 16 dwords
            #   frag_C [Cval, Nrep, Mrep]: (4,2,4) f32  -> 32 dwords/quadrant
            #   frag_P [Cval, Nrep, Mrep]: (4,2,2) f32  -> 16 dwords
            # These are logical register equivalents per fragment, not
            # additive physical VGPR allocations. A storage is reused across
            # M slices; both B fragments and all four C quadrants stay live.
            # One P FIFO is reused across slices/quadrants. prev_scale_a has
            # two f32 values; prev_scale_b is one wave-uniform scalar.

            # for mm in (Mrep):
            #   dq_scale = prev_scale_a[mm] *prev_scale_b
            #   m_slice_offset = prev_m_slice * PHASE_M_REP
            #   for nn in (Nrep):
            #       frag_C[0, nn, m_slice_offset + mm] += dq_scale * frag_P[0, nn, mm]
            #       frag_C[1, nn, m_slice_offset + mm] += dq_scale * frag_P[1, nn, mm]
            #       frag_C[2, nn, m_slice_offset + mm] += dq_scale * frag_P[2, nn, mm]
            #       frag_C[3, nn, m_slice_offset + mm] += dq_scale * frag_P[3, nn, mm]
            #       frag_P[0：3, nn, mm] = mfma_16x16x128(frag_A[:, mm], frag_B[:, nn], 0)

            #
            # Compiler-only sched_barrier fences preserve 4 scalar FMAs ->
            # 1 MFMA; the launch's -packed-fp32-ops disables FP32 packing.
            # This compute block uses fx.fma and the MFMA intrinsic, not
            # inline asm/early-clobber constraints. WG sync stays in the caller.
            for m0 in range_constexpr(PHASE_M_REP):
                scale = Vec(prev_scale_a)[m0] * prev_scale_b
                rocdl.sched_barrier(0)
                for n0 in range_constexpr(N_REP):
                    cs = frag_C[None, n0, prev_m_slice * PHASE_M_REP + m0]
                    ps = frag_P[None, n0, m0]
                    partial = Vec(ps.load())
                    accum = Vec(cs.load())
                    values = []
                    for elem in range_constexpr(4):
                        values.append(fx.fma(partial[elem], scale, accum[elem]))
                    cs.store(Vec.from_elements(values, fx.Float32))
                    rocdl.sched_barrier(0)
                    ps.store(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [
                                Vec(frag_B[None, n0, 0].load()).bitcast(fx.Int32),
                                Vec(frag_A[None, m0, 0].load()).bitcast(fx.Int32),
                                Vec.filled(4, 0.0, fx.Float32),
                                0,
                                0,
                                0,
                                fx.Int32(0),
                                0,
                                fx.Int32(0),
                            ],
                        )
                    )
                    rocdl.sched_barrier(0)

        num_tiles = K // BLOCK_K
        assert num_tiles % 2 == 0

        def begin_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
            rocdl.sched_barrier(0)

        def end_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        _s2r_Bl = [s2r_src0_B_l, s2r_src1_B_l]
        _s2r_Br = [s2r_src0_B_r, s2r_src1_B_r]

        def lds_rd_At(lds_idx, m_slice=0):
            src = copy_a.partition_S(_a_m_slice_view(ldsA_t_rd[lds_idx], m_slice))
            fx.copy(lds_copy_atom, src, dest_frag_A_t)

        def lds_rd_Ab(lds_idx, m_slice=0):
            src = copy_a.partition_S(_a_m_slice_view(ldsA_b_rd[lds_idx], m_slice))
            fx.copy(lds_copy_atom, src, dest_frag_A_t)

        def lds_rd_Bl(lds_idx):
            fx.copy(lds_copy_atom, _s2r_Bl[lds_idx], dest_frag_B_l, pred=None)

        def lds_rd_Br(lds_idx):
            fx.copy(lds_copy_atom, _s2r_Br[lds_idx], dest_frag_B_r, pred=None)

        # Scalar LDS bases plus static chunk offsets avoid per-load address
        # VGPRs. A and unshuffled B share the grouped-row / dual-padding map.
        _elem_bytes = element_type.width // 8  # fp8 = 1

        def _dma_dst_ptr(root_view, byte_offset):
            # Keep a typed byte pointer until the final raw-DMA boundary,
            # so subsequent static chunk offsets stay byte-addressed.
            return fx.add_offset(
                fx.recast_iter(fx.Uint8, fx.get_iter(root_view)),
                fx.make_int_tuple(byte_offset),
            )

        # pyhip 的 g2s tv（512 线程 = 64 行 × 8 k-组，每线程 1×16 fp8）
        _g2s_tile, _g2s_tv = fx.make_layout_tv(
            fx.make_layout((8 * 8, 8), (8, 1)),
            fx.make_layout((1, elements_per_128b), (1, 1)),
        )
        _copy_g2s = fx.make_tiled_copy(buffer_copy_atom, _g2s_tv, _g2s_tile).get_slice(
            tid
        )
        _dst_stride = _copy_g2s.partition_D(ldsA_t_wr[0]).stride[1].to_py_value()

        # 每 wave 的 LDS 基址（dual-padding group），readfirstlane 一次
        _a_wave_off_elems = wave_id % 2 * (8 * BLOCK_K + 16) + wave_id // 2 * (
            2 * (8 * BLOCK_K + 16) + 32
        )
        _a_wave_off_bytes = rocdl.readfirstlane(
            T.i32, arith._to_raw(fx.Int32(_a_wave_off_elems * _elem_bytes))
        )
        _b_wave_off_bytes = _a_wave_off_bytes  # 非 preshuffle B 与 A 同布局

        dma_aT_dst = [
            _dma_dst_ptr(ldsA_t_wr[0], _a_wave_off_bytes),
            _dma_dst_ptr(ldsA_t_wr[1], _a_wave_off_bytes),
        ]
        dma_aB_dst = [
            _dma_dst_ptr(ldsA_b_wr[0], _a_wave_off_bytes),
            _dma_dst_ptr(ldsA_b_wr[1], _a_wave_off_bytes),
        ]
        dma_bL_dst = [
            _dma_dst_ptr(ldsB_l_wr[0], _b_wave_off_bytes),
            _dma_dst_ptr(ldsB_l_wr[1], _b_wave_off_bytes),
        ]
        dma_bR_dst = [
            _dma_dst_ptr(ldsB_r_wr[0], _b_wave_off_bytes),
            _dma_dst_ptr(ldsB_r_wr[1], _b_wave_off_bytes),
        ]

        # 每 thread 的 (row, k) 源映射（对标 pyhip a_lane_row = tid//8）
        _a_lane_row = tid // 8
        _a_lane_k = tid % 8 * elements_per_128b
        _a_local_row = _a_lane_row % 8 * (BLOCK_M // 8) + _a_lane_row // 8
        _lane_src_offset = fx.Int32((_a_local_row * K + _a_lane_k) * _elem_bytes)
        _aT_src_wave_base = fx.Int32(0)
        _bL_src_wave_base = fx.Int32(0)

        def _raw_g2s(rsrc, dst_base, src_wave_base, ki):
            for chunk in range_constexpr(BLOCK_M // 64):
                _dp = _lds_byte_ptr(
                    dst_base,
                    chunk * _dst_stride * _elem_bytes,
                )
                _so = src_wave_base + fx.Int32(
                    ki * BLOCK_K * _elem_bytes + chunk * 8 * K * _elem_bytes
                )
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc,
                    _dp,
                    fx.Int32(16),
                    _lane_src_offset,
                    _so,
                    fx.Int32(0),
                    fx.Int32(0),
                )

        if const_expr(preshuffle_b):
            # Each wave copies a contiguous 16x64-byte block; two instructions
            # per quadrant retain the existing pipeline's vmcnt accounting.
            # Global offset(n,k) = (n//16)*16*K + (k//16)*256
            #                       + (n%16)*16 + k%16.
            _b_ps_lane_offset = fx.Int32(
                (wave_id // 2) * 16 * K + (wave_id % 2) * 1024 + (tid % 64) * 16
            )
            _b_ps_wave_offset = rocdl.readfirstlane(
                T.i32, arith._to_raw(fx.Int32(wave_id * 1024))
            )

            def _preshuffle_g2s(root_view, quadrant, ki):
                for chunk in range_constexpr(2):
                    dst = _lds_byte_ptr(
                        fx.get_iter(root_view), _b_ps_wave_offset + chunk * 8192
                    )
                    src_base = fx.Int32(
                        (quadrant * BLOCK_N + chunk * 64) * K + ki * BLOCK_K * 16
                    )
                    rocdl.raw_ptr_buffer_load_lds(
                        b_dma_rsrc,
                        dst,
                        fx.Int32(16),
                        _b_ps_lane_offset,
                        src_base,
                        fx.Int32(0),
                        fx.Int32(0),
                    )

        def _ac_At(lds_idx, ki):
            _raw_g2s(a_dma_rsrc, dma_aT_dst[lds_idx], _aT_src_wave_base, ki)

        def _ac_Ab(lds_idx, ki):
            _raw_g2s(
                a_dma_rsrc,
                dma_aB_dst[lds_idx],
                _aT_src_wave_base + BLOCK_M * K * _elem_bytes,
                ki,
            )

        def _ac_Bl(lds_idx, ki):
            if const_expr(preshuffle_b):
                _preshuffle_g2s(ldsB_l_rd[lds_idx], 0, ki)
            else:
                _raw_g2s(b_dma_rsrc, dma_bL_dst[lds_idx], _bL_src_wave_base, ki)

        def _ac_Br(lds_idx, ki):
            if const_expr(preshuffle_b):
                _preshuffle_g2s(ldsB_r_rd[lds_idx], 1, ki)
            else:
                _raw_g2s(
                    b_dma_rsrc,
                    dma_bR_dst[lds_idx],
                    _bL_src_wave_base + BLOCK_N * K * _elem_bytes,
                    ki,
                )

        rocdl.sched_barrier(0)
        _ac_scale_a(0, fx.Int32(0))
        rocdl.sched_barrier(0)
        _ac_Bl(0, 0)
        rocdl.sched_barrier(0)
        _ac_At(0, 0)
        rocdl.sched_barrier(0)
        _ac_Br(0, 0)
        rocdl.sched_barrier(0)
        _ac_Ab(0, 0)
        rocdl.sched_barrier(0)
        # Offset the two groups of four waves by one stage. The lower group
        # closes this unmatched barrier after the final FIFO drain.
        if wave_id >= 4:
            rocdl.s_barrier()
        frag_C_tl.fill(0)
        frag_C_tr.fill(0)
        frag_C_bl.fill(0)
        frag_C_br.fill(0)

        vm_load_cnt_a = 2
        vm_load_cnt_b = 2
        vm_load_cnt_scale_a = 1

        rocdl.sched_barrier(0)
        ### todo: this part useless??? barrier needed?
        vmcnt = vm_load_cnt_a + vm_load_cnt_b
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        rocdl.sched_barrier(0)
        _ac_scale_a(1, fx.Int32(1))
        rocdl.sched_barrier(0)
        _ac_At(1, 1)
        rocdl.sched_barrier(0)
        _ac_Bl(1, 1)
        rocdl.sched_barrier(0)
        _ac_Br(1, 1)
        rocdl.sched_barrier(0)

        vmcnt = vm_load_cnt_a + vm_load_cnt_b * 2 + vm_load_cnt_scale_a
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        lds_rd_Bl(0)

        frag_P.fill(0)
        acc_init = [
            frag_C_tl.load(),
            frag_C_tr.load(),
            frag_C_bl.load(),
            frag_C_br.load(),
            frag_P.load(),
            Vec.filled(PHASE_M_REP, 0.0, fx.Float32),
            fx.Float32(0),
            frag_B_l.load(),
        ]

        for kidx, states in range(0, num_tiles, 2, init=acc_init):
            frag_C_tl.store(states[0])
            frag_C_tr.store(states[1])
            frag_C_bl.store(states[2])
            frag_C_br.store(states[3])
            frag_P.store(states[4])
            fifo_scale_a_0 = Vec.filled(PHASE_M_REP, 0.0, fx.Float32)
            fifo_scale_b_0 = fx.Float32(0)
            fifo_scale_a_1 = Vec(states[5])
            fifo_scale_b_1 = fx.Float32(states[6])
            frag_B_l.store(states[7])
            kiter = fx.Int32(kidx)

            # Each K tile: TL[s0]/TR[s0]/BL[s0]/BR[s0], then TL[s1]/TR[s1]/BL[s1]/BR[s1].
            # A and P are half-sized and reused, B_l/B_r survive both M slices.
            # P always holds the immediately preceding compute phase:
            # TL[s0] consumes BR[s1](k-1); TL[s1] consumes BR[s0](k). Other phases
            # consume the preceding quadrant of their current M slice.
            for tile in range_constexpr(2):
                tick = tile
                tock = 1 - tile
                ki = kiter + tile
                for m_slice in range_constexpr(2):
                    lds_rd_At(tick, m_slice)
                    mfma_scaleA = lds_rd_scale_a(tick, 0, m_slice)
                    if const_expr(m_slice == 0):
                        mfma_scaleB = load_scale_b(ki)
                        _ac_Ab(tock, ki + 1)
                    rocdl.sched_barrier(0)

                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    begin_compute_phase()
                    do_gemm(
                        frag_C_br,
                        frag_B_l,
                        frag_A_t,
                        fifo_scale_a_1,
                        fifo_scale_b_1,
                        1 - m_slice,
                    )
                    end_compute_phase()

                    if const_expr(m_slice == 0):
                        lds_rd_Br(tick)
                    else:
                        # A_t must survive slice0; only slice1's read closes
                        # its lifetime in both staggered wave groups.
                        _ac_At(tick, ki + 2)

                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    begin_compute_phase()
                    do_gemm(
                        frag_C_tl,
                        frag_B_r,
                        frag_A_t,
                        fifo_scale_a_0,
                        fifo_scale_b_0,
                        m_slice,
                    )
                    end_compute_phase()

                    lds_rd_Ab(tick, m_slice)
                    mfma_scaleA = lds_rd_scale_a(tick, 1, m_slice)
                    if const_expr(m_slice == 0):
                        # B survives in registers through slice1; its LDS
                        # slot is already free after slice0's reads.
                        _ac_Bl(tick, ki + 2)

                    fifo_scale_a_0, fifo_scale_b_0 = mfma_scaleA, mfma_scaleB[0]
                    begin_compute_phase()
                    do_gemm(
                        frag_C_tr,
                        frag_B_l,
                        frag_A_t,
                        fifo_scale_a_1,
                        fifo_scale_b_1,
                        m_slice,
                    )
                    end_compute_phase()

                    if const_expr(m_slice == 0):
                        _ac_Br(tick, ki + 2)
                    else:
                        # BL[s1]'s end barrier closes all current ScaleA reads.
                        _ac_scale_a(tick, ki + 2)
                        rocdl.s_waitcnt(
                            encode_waitcnt_950(
                                vmcnt=vm_load_cnt_a
                                + vm_load_cnt_b * 2
                                + vm_load_cnt_scale_a
                            )
                        )
                        # BL[s1] has consumed the current B_l registers.
                        # The next LDS slot predates A_b(k+1), which
                        # the rolling vmcnt wait above completes.
                        lds_rd_Bl(tock)

                    fifo_scale_a_1, fifo_scale_b_1 = mfma_scaleA, mfma_scaleB[1]
                    begin_compute_phase()
                    do_gemm(
                        frag_C_bl,
                        frag_B_r,
                        frag_A_t,
                        fifo_scale_a_0,
                        fifo_scale_b_0,
                        m_slice,
                    )
                    end_compute_phase()
            yield_values = [
                frag_C_tl.load(),
                frag_C_tr.load(),
                frag_C_bl.load(),
                frag_C_br.load(),
                frag_P.load(),
                fifo_scale_a_1,
                fifo_scale_b_1,
                frag_B_l.load(),
            ]
            results = yield yield_values

        frag_C_tl.store(results[0])
        frag_C_tr.store(results[1])
        frag_C_bl.store(results[2])
        frag_C_br.store(results[3])

        c_store_rsrc = rocdl.get_buffer_rsrc(fx.get_iter(C))
        bC_tr = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, 0, 1]
        bC_bl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, 1, 0]
        bC_br = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, 1, 1]
        transposed_c_layout = fx.make_ordered_layout((BLOCK_N, BLOCK_M), (1, 0))
        bC_tr = fx.composition(bC_tr, transposed_c_layout)
        bC_bl = fx.composition(bC_bl, transposed_c_layout)
        bC_br = fx.composition(bC_br, transposed_c_layout)

        frag_P.store(results[4])
        fifo_scale_a_1 = Vec(results[5])
        fifo_scale_b_1 = fx.Float32(results[6])
        for m0 in range_constexpr(PHASE_M_REP):
            for n0 in range_constexpr(N_REP):
                cs = frag_C_br[None, n0, (M_SLICES - 1) * PHASE_M_REP + m0]
                scale = Vec(fifo_scale_a_1)[m0] * fifo_scale_b_1
                scale_vec = Vec.filled(4, scale, fx.Float32)
                cs.store(fx.fma(frag_P[None, n0, m0].load(), scale_vec, cs.load()))
        if wave_id < 4:
            rocdl.s_barrier()

        # ---- epilogue store ----
        N_tail = N % TILE_N != 0
        if const_expr((permlane_epilogue or N_tail) and TILE_N % 256 == 0):
            pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
            lane_id = tid % 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4
            lane_group = lane_id // 16
            fragment_mode_0_repeat = TILE_N // 128
            fragment_mode_1_repeat = TILE_M // 64

            def store_c_quadrant(c_frag, quadrant_m, quadrant_n):
                for row_repeat in range_constexpr(fragment_mode_1_repeat):
                    for col_repeat in range_constexpr(0, fragment_mode_0_repeat, 2):
                        acc_a = Vec(c_frag[None, col_repeat, row_repeat].load())
                        acc_b = Vec(c_frag[None, col_repeat + 1, row_repeat].load())
                        d0_a = rocdl.cvt_pk_bf16_f32(acc_a[0], acc_a[1])
                        d1_a = rocdl.cvt_pk_bf16_f32(acc_a[2], acc_a[3])
                        d0_b = rocdl.cvt_pk_bf16_f32(acc_b[0], acc_b[1])
                        d1_b = rocdl.cvt_pk_bf16_f32(acc_b[2], acc_b[3])
                        swap0 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(d0_a),
                            arith._to_raw(d0_b),
                            False,
                            False,
                        )
                        swap1 = rocdl.permlane16_swap(
                            pair_type,
                            arith._to_raw(d1_a),
                            arith._to_raw(d1_b),
                            False,
                            False,
                        )
                        packed = Vec.from_elements(
                            [
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [1])),
                            ],
                            fx.Int32,
                        )
                        row = (
                            quadrant_m * (TILE_M // 2)
                            + row_repeat * 32
                            + wave_m * 16
                            + lane_id % 16
                        )
                        col = (
                            quadrant_n * (TILE_N // 2)
                            + col_repeat * 64
                            + lane_group % 2 * 64
                            + wave_n * 16
                            + lane_group // 2 * 8
                        )
                        byte_offset = fx.Int32((row * N + col) * 2)
                        if const_expr(N_tail):
                            byte_offset = (col < b_rows_left).select(
                                byte_offset, fx.Int32(0x7FFFFFFF)
                            )
                        rocdl.raw_ptr_buffer_store(
                            packed.ir_value(),
                            c_store_rsrc,
                            byte_offset.ir_value(),
                            fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(T.i32, 0),
                        )

            store_c_quadrant(frag_C_tl, 0, 0)
            store_c_quadrant(frag_C_tr, 0, 1)
            store_c_quadrant(frag_C_bl, 1, 0)
            store_c_quadrant(frag_C_br, 1, 1)
        else:
            assert (
                N % TILE_N == 0
            ), "N must be a multiple of TILE_N for permlane_epilogue=False, not supported for now"
            c_frag_bf16 = fx.make_fragment_like(frag_C_tl, dtype=fx.BFloat16)
            store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
            store_thr = fx.make_tiled_copy_C(store_atom, tiled_mma).get_slice(tid)

            def store_c_quadrant(c_frag, bC):
                c_frag_bf16.store(c_frag.load().to(fx.BFloat16))
                fx.copy(
                    store_atom, store_thr.retile(c_frag_bf16), store_thr.partition_D(bC)
                )

            store_c_quadrant(frag_C_tl, bC_tl)
            store_c_quadrant(frag_C_tr, bC_tr)
            store_c_quadrant(frag_C_bl, bC_bl)
            store_c_quadrant(frag_C_br, bC_br)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        scaleA: fx.Tensor,
        scaleB: fx.Tensor,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        value_attrs = {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}
        gemm_kernel(A, B, C, scaleA, scaleB, M, value_attrs=value_attrs).launch(
            grid=(div_up(M, TILE_M) * div_up(N, TILE_N), 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    return launch_gemm

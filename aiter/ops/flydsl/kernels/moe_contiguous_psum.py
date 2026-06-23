# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepGEMM-contiguous M-tile prefix sum (FlyDSL), single-block serial scan.

Given per-expert row counts ``masked_m`` (E,), computes the tile-aligned
exclusive prefix sum used by the contiguous grouped-GEMM scheduler:

    aligned[e]   = ceil(masked_m[e] / tile_m) * tile_m
    starts[e]    = sum(aligned[0:e])            # exclusive prefix sum
    psum[e]      = starts[e] + masked_m[e]      # actual end (NOT tile-aligned)
    contiguous_m = max(tile_m, sum(aligned))

Replaces ``torch.cumsum``, which on ROCm lowers to a rocprim scan (a
``trampoline_kernel`` plus an internal D2D temp copy) on every call. The number
of experts E is tiny, so a single-thread serial scan in one block is cheaper and
copy-free.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, vector
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf


def build_moe_contiguous_psum_module():
    """Return a JIT launcher computing the tile-aligned prefix sum in one pass.

    Launcher: ``(masked_m, starts, psum, contiguous_m, experts, tile_m, stream=)``
      masked_m     : (E,)  int32  in   per-expert row counts
      starts       : (E,)  int32  out  exclusive prefix sum of aligned counts
      psum         : (E,)  int32  out  starts[e] + masked_m[e]
      contiguous_m : (1,)  int32  out  max(tile_m, sum(aligned))
    """

    @flyc.kernel(name="moe_contiguous_psum", known_block_size=[64, 1, 1])
    def psum_kernel(
        masked_m: fx.Tensor,  # (E,) int32 in
        starts: fx.Tensor,  # (E,) int32 out
        psum: fx.Tensor,  # (E,) int32 out
        contiguous_m: fx.Tensor,  # (1,) int32 out
        experts: Int32,
        tile_m: Int32,
    ):
        i32 = T.i32
        tid = ArithValue(fx.thread_idx.x)
        is_leader = arith.cmpi(CmpIPredicate.eq, tid, arith.constant(0, type=i32))
        _if = scf.IfOp(is_leader)
        with ir.InsertionPoint(_if.then_block):
            m_rsrc = buffer_ops.create_buffer_resource(masked_m, max_size=True)
            s_rsrc = buffer_ops.create_buffer_resource(starts, max_size=True)
            p_rsrc = buffer_ops.create_buffer_resource(psum, max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(contiguous_m, max_size=True)

            tile_v = ArithValue(tile_m)
            tile_minus_1 = tile_v - arith.constant(1, type=i32)
            # -tile_m in two's complement = ~(tile_m - 1) for power-of-2 tile_m;
            # used as AND mask for fast ceil-alignment.
            neg_tile = arith.constant(0, type=i32) - tile_v

            c0_i32 = arith.constant(0, type=i32)
            c4_i32 = arith.constant(4, type=i32)
            c0_idx = arith.index(0)
            c1_idx = arith.index(1)

            # Main loop: 4 experts per iteration with vec4 load/store.
            e_total = ArithValue(experts)
            e_vec4 = arith.shrui(e_total, arith.constant(2, type=i32))
            e_vec4_idx = arith.index_cast(T.index, e_vec4)

            main_loop = scf.ForOp(
                c0_idx, e_vec4_idx, c1_idx, [c0_i32]
            )
            with ir.InsertionPoint(main_loop.body):
                iter_idx = main_loop.induction_variable
                cur = main_loop.inner_iter_args[0]
                base_i32 = arith.index_cast(i32, iter_idx) * c4_i32

                m_vec = buffer_ops.buffer_load(
                    m_rsrc, base_i32, vec_width=4, dtype=i32
                )
                m0 = ArithValue(
                    vector.extract(m_vec, static_position=[0], dynamic_position=[])
                )
                m1 = ArithValue(
                    vector.extract(m_vec, static_position=[1], dynamic_position=[])
                )
                m2 = ArithValue(
                    vector.extract(m_vec, static_position=[2], dynamic_position=[])
                )
                m3 = ArithValue(
                    vector.extract(m_vec, static_position=[3], dynamic_position=[])
                )

                # Expert 0: aligned = (m + tile_m - 1) & (-tile_m)
                aligned0 = (m0 + tile_minus_1) & neg_tile
                s0 = cur
                p0 = ArithValue(cur) + m0
                cur1 = ArithValue(cur) + aligned0

                # Expert 1
                aligned1 = (m1 + tile_minus_1) & neg_tile
                s1 = cur1
                p1 = cur1 + m1
                cur2 = cur1 + aligned1

                # Expert 2
                aligned2 = (m2 + tile_minus_1) & neg_tile
                s2 = cur2
                p2 = cur2 + m2
                cur3 = cur2 + aligned2

                # Expert 3
                aligned3 = (m3 + tile_minus_1) & neg_tile
                s3 = cur3
                p3 = cur3 + m3
                next_cur = cur3 + aligned3

                # Vec4 store starts
                s_vec = vector.from_elements(
                    T.vec(4, T.i32), [s0, s1, s2, s3]
                )
                buffer_ops.buffer_store(s_vec, s_rsrc, base_i32)

                # Vec4 store psum
                p_vec = vector.from_elements(
                    T.vec(4, T.i32), [p0, p1, p2, p3]
                )
                buffer_ops.buffer_store(p_vec, p_rsrc, base_i32)

                scf.YieldOp([next_cur])

            main_cur = main_loop.results[0]

            # Tail loop: remaining E % 4 experts (scalar).
            tail_start_i32 = e_vec4 * c4_i32
            tail_start_idx = arith.index_cast(T.index, tail_start_i32)
            e_upper = arith.index_cast(T.index, experts)
            tail_loop = scf.ForOp(
                tail_start_idx, e_upper, c1_idx, [main_cur]
            )
            with ir.InsertionPoint(tail_loop.body):
                e = tail_loop.induction_variable
                cur_t = tail_loop.inner_iter_args[0]
                e_i32_t = arith.index_cast(i32, e)
                m_t = buffer_ops.buffer_load(
                    m_rsrc, e_i32_t, vec_width=1, dtype=i32
                )
                aligned_t = (ArithValue(m_t) + tile_minus_1) & neg_tile
                buffer_ops.buffer_store(cur_t, s_rsrc, e_i32_t)
                buffer_ops.buffer_store(
                    ArithValue(cur_t) + ArithValue(m_t), p_rsrc, e_i32_t
                )
                next_cur_t = ArithValue(cur_t) + ArithValue(aligned_t)
                scf.YieldOp([next_cur_t])

            final_cur = tail_loop.results[0]
            # contiguous_m = max(tile_m, final_cur)
            gt = arith.cmpi(CmpIPredicate.sgt, final_cur, tile_v)
            cm = arith.select(gt, final_cur, tile_v)
            buffer_ops.buffer_store(cm, c_rsrc, c0_i32)
            scf.YieldOp([])

    @flyc.jit
    def launch_psum(
        masked_m: fx.Tensor,
        starts: fx.Tensor,
        psum: fx.Tensor,
        contiguous_m: fx.Tensor,
        experts: fx.Int32,
        tile_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        psum_kernel(masked_m, starts, psum, contiguous_m, experts, tile_m).launch(
            grid=(arith.index(1), 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return launch_psum


_FUSED_BLOCK = 256


def build_moe_contiguous_remap_module():
    """Return a JIT launcher that fuses prefix-sum + fill + remap into one kernel.

    Single block with barrier synchronization between phases:
      Phase 1 (thread 0): serial prefix sum → writes starts[], psum[]
      Phase 2 (all threads): fill remapped_rows_out with -1
      Phase 3 (all threads): parallel remap of routes

    Eliminates 3 separate kernel launches (psum + torch.full + remap).

    Launcher: ``(masked_m, topids_to_rows_in, rows_to_tokens_in,
                 remapped_topids_out, remapped_rows_out, psum_out,
                 experts, max_m, tile_m, numel, contiguous_m, stream=)``
    """

    @flyc.kernel(
        name="moe_contiguous_psum_remap", known_block_size=[_FUSED_BLOCK, 1, 1]
    )
    def fused_kernel(
        masked_m: fx.Tensor,  # (E,) int32 in
        topids_to_rows_in: fx.Tensor,  # (numel,) int32 in
        rows_to_tokens_in: fx.Tensor,  # (E*max_m,) int32 in
        remapped_topids_out: fx.Tensor,  # (numel,) int32 out
        remapped_rows_out: fx.Tensor,  # (contiguous_m,) int32 out
        psum_out: fx.Tensor,  # (E,) int32 out
        experts: Int32,
        max_m: Int32,
        tile_m: Int32,
        numel: Int32,
        contiguous_m: Int32,
    ):
        i32 = T.i32
        tid = ArithValue(fx.thread_idx.x)
        is_leader = arith.cmpi(CmpIPredicate.eq, tid, arith.constant(0, type=i32))

        m_rsrc = buffer_ops.create_buffer_resource(masked_m, max_size=True)
        old_routes_rsrc = buffer_ops.create_buffer_resource(
            topids_to_rows_in, max_size=True
        )
        old_rows_rsrc = buffer_ops.create_buffer_resource(
            rows_to_tokens_in, max_size=True
        )
        new_topids_rsrc = buffer_ops.create_buffer_resource(
            remapped_topids_out, max_size=True
        )
        new_rows_rsrc = buffer_ops.create_buffer_resource(
            remapped_rows_out, max_size=True
        )
        psum_rsrc = buffer_ops.create_buffer_resource(psum_out, max_size=True)

        # ---- Phase 1: thread 0 computes prefix sum ----
        _if_leader = scf.IfOp(is_leader)
        with ir.InsertionPoint(_if_leader.then_block):
            tile_v = ArithValue(tile_m)
            tile_minus_1 = tile_v - arith.constant(1, type=i32)
            neg_tile = arith.constant(0, type=i32) - tile_v

            c0_i32 = arith.constant(0, type=i32)
            c4_i32 = arith.constant(4, type=i32)
            c0_idx = arith.index(0)
            c1_idx = arith.index(1)

            # We write starts to psum_out temporarily; psum overwrites after.
            # Use a separate resource for starts (reuse psum_out buffer: first
            # write starts, then overwrite with actual psum in same loop).
            # Actually: write starts[] into a "starts" section. Since we need
            # starts[] to remain readable during phase 3, write psum[] to
            # psum_out and starts[] ... we need starts accessible in phase 3.
            # Solution: write psum_out[e] = starts[e] first, then after the loop
            # overwrite with actual psum? No — we need both.
            # Better: use psum_out for psum, and store starts in-place by
            # reusing masked_m buffer (it's read-only input but same size).
            # Cleanest: store starts directly into remapped_rows_out (first E
            # entries are overwritten in phase 2 anyway).
            # SIMPLEST: store starts into psum_out as "starts" then in phase 3
            # read starts from psum_out. After phase 3 is done, we overwrite
            # psum_out with actual psum values. But that requires another barrier.
            #
            # Final approach: allocate starts[] in the first E entries of
            # remapped_rows_out (which gets -1 filled in phase 2 anyway, so
            # overwritten). Actually that breaks phase 3 which reads starts.
            #
            # Real solution: the kernel outputs psum_out[] = actual ends (for
            # GEMM binary search). For phase 3, we need starts[e]. Compute both:
            # - Write psum_out[e] = cur + m  (actual end — final output)
            # - Derive starts[e] from psum during phase 3: starts[e] = psum[e] - m[e]
            #   → requires re-reading masked_m[e] in phase 3. But that's only
            #   1 extra load per route (cached in L1).
            #
            # Even simpler: just compute and store psum[e] = starts[e] + m[e].
            # In phase 3: starts[e] = psum[e] - masked_m[e]. This avoids needing
            # a separate starts buffer.

            e_total = ArithValue(experts)
            e_vec4 = arith.shrui(e_total, arith.constant(2, type=i32))
            e_vec4_idx = arith.index_cast(T.index, e_vec4)

            main_loop = scf.ForOp(c0_idx, e_vec4_idx, c1_idx, [c0_i32])
            with ir.InsertionPoint(main_loop.body):
                iter_idx = main_loop.induction_variable
                cur = main_loop.inner_iter_args[0]
                base_i32 = arith.index_cast(i32, iter_idx) * c4_i32

                m_vec = buffer_ops.buffer_load(
                    m_rsrc, base_i32, vec_width=4, dtype=i32
                )
                m0 = ArithValue(
                    vector.extract(m_vec, static_position=[0], dynamic_position=[])
                )
                m1 = ArithValue(
                    vector.extract(m_vec, static_position=[1], dynamic_position=[])
                )
                m2 = ArithValue(
                    vector.extract(m_vec, static_position=[2], dynamic_position=[])
                )
                m3 = ArithValue(
                    vector.extract(m_vec, static_position=[3], dynamic_position=[])
                )

                aligned0 = (m0 + tile_minus_1) & neg_tile
                p0 = ArithValue(cur) + m0
                cur1 = ArithValue(cur) + aligned0

                aligned1 = (m1 + tile_minus_1) & neg_tile
                p1 = cur1 + m1
                cur2 = cur1 + aligned1

                aligned2 = (m2 + tile_minus_1) & neg_tile
                p2 = cur2 + m2
                cur3 = cur2 + aligned2

                aligned3 = (m3 + tile_minus_1) & neg_tile
                p3 = cur3 + m3
                next_cur = cur3 + aligned3

                p_vec = vector.from_elements(
                    T.vec(4, T.i32), [p0, p1, p2, p3]
                )
                buffer_ops.buffer_store(p_vec, psum_rsrc, base_i32)

                scf.YieldOp([next_cur])

            main_cur = main_loop.results[0]

            # Tail loop for E % 4
            tail_start_i32 = e_vec4 * c4_i32
            tail_start_idx = arith.index_cast(T.index, tail_start_i32)
            e_upper = arith.index_cast(T.index, experts)
            tail_loop = scf.ForOp(tail_start_idx, e_upper, c1_idx, [main_cur])
            with ir.InsertionPoint(tail_loop.body):
                e = tail_loop.induction_variable
                cur_t = tail_loop.inner_iter_args[0]
                e_i32_t = arith.index_cast(i32, e)
                m_t = buffer_ops.buffer_load(
                    m_rsrc, e_i32_t, vec_width=1, dtype=i32
                )
                aligned_t = (ArithValue(m_t) + tile_minus_1) & neg_tile
                buffer_ops.buffer_store(
                    ArithValue(cur_t) + ArithValue(m_t), psum_rsrc, e_i32_t
                )
                next_cur_t = ArithValue(cur_t) + ArithValue(aligned_t)
                scf.YieldOp([next_cur_t])

            scf.YieldOp([])

        # ---- Barrier: wait for prefix sum to be visible ----
        gpu.barrier()

        # ---- Phase 2: all threads fill remapped_rows_out with -1 ----
        neg_one = arith.constant(-1, type=i32)
        c_block = arith.constant(_FUSED_BLOCK, type=i32)
        contiguous_m_v = ArithValue(contiguous_m)

        c0_idx2 = arith.index(0)
        c1_idx2 = arith.index(1)
        fill_iters = arith.index_cast(
            T.index,
            arith.divui(
                contiguous_m_v + c_block - arith.constant(1, type=i32), c_block
            ),
        )
        fill_loop = scf.ForOp(c0_idx2, fill_iters, c1_idx2, [])
        with ir.InsertionPoint(fill_loop.body):
            fi = fill_loop.induction_variable
            fill_idx = arith.index_cast(i32, fi) * c_block + tid
            fill_in_range = arith.cmpi(
                CmpIPredicate.ult, fill_idx, contiguous_m_v
            )
            fill_if = scf.IfOp(fill_in_range, results_=[], has_else=False)
            with ir.InsertionPoint(fill_if.then_block):
                buffer_ops.buffer_store(neg_one, new_rows_rsrc, fill_idx)
                scf.YieldOp([])
            scf.YieldOp([])

        # ---- Barrier: wait for fill to complete ----
        gpu.barrier()

        # ---- Phase 3: all threads remap routes ----
        numel_v = ArithValue(numel)
        max_m_v = ArithValue(max_m)

        remap_iters = arith.index_cast(
            T.index,
            arith.divui(
                numel_v + c_block - arith.constant(1, type=i32), c_block
            ),
        )
        remap_loop = scf.ForOp(c0_idx2, remap_iters, c1_idx2, [])
        with ir.InsertionPoint(remap_loop.body):
            ri = remap_loop.induction_variable
            route_idx = arith.index_cast(i32, ri) * c_block + tid
            route_in_range = arith.cmpi(
                CmpIPredicate.ult, route_idx, numel_v
            )
            route_if = scf.IfOp(route_in_range, results_=[], has_else=False)
            with ir.InsertionPoint(route_if.then_block):
                old_val = buffer_ops.buffer_load(
                    old_routes_rsrc, route_idx, vec_width=1, dtype=i32
                )
                expert_id = arith.divui(ArithValue(old_val), max_m_v)
                slot = ArithValue(old_val) - expert_id * max_m_v

                # starts[e] = psum[e] - masked_m[e]
                psum_e = buffer_ops.buffer_load(
                    psum_rsrc, expert_id, vec_width=1, dtype=i32
                )
                m_e = buffer_ops.buffer_load(
                    m_rsrc, expert_id, vec_width=1, dtype=i32
                )
                start_e = ArithValue(psum_e) - ArithValue(m_e)

                new_val = start_e + slot
                buffer_ops.buffer_store(new_val, new_topids_rsrc, route_idx)

                src_token = buffer_ops.buffer_load(
                    old_rows_rsrc, old_val, vec_width=1, dtype=i32
                )
                buffer_ops.buffer_store(src_token, new_rows_rsrc, new_val)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_fused_remap(
        masked_m: fx.Tensor,
        topids_to_rows_in: fx.Tensor,
        rows_to_tokens_in: fx.Tensor,
        remapped_topids_out: fx.Tensor,
        remapped_rows_out: fx.Tensor,
        psum_out: fx.Tensor,
        experts: fx.Int32,
        max_m: fx.Int32,
        tile_m: fx.Int32,
        numel: fx.Int32,
        contiguous_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        fused_kernel(
            masked_m,
            topids_to_rows_in,
            rows_to_tokens_in,
            remapped_topids_out,
            remapped_rows_out,
            psum_out,
            experts,
            max_m,
            tile_m,
            numel,
            contiguous_m,
        ).launch(
            grid=(arith.index(1), 1, 1),
            block=(_FUSED_BLOCK, 1, 1),
            stream=stream,
        )

    return launch_fused_remap

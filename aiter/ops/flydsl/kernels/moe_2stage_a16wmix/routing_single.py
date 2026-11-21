# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-wave stable routing metadata for a single token (gfx942).

R <= 16 routes use three R-element LDS arrays: 12 * R <= 192 bytes.
Valid experts are sorted by (expert, original slot); each unique expert gets
a complete BM block. Padding IDs use token sentinel 1, weights zero, and
unused expert entries -1. No cross-workgroup communication is performed.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

_BLOCK_THREADS = 64


def _build_routing_single_module(r: int, bm: int, experts: int):
    """Return a JIT launcher for the single-token routing kernel.

    ``r``, ``bm``, and ``experts`` are baked into the compiled module so that
    all inner loops over ``R`` and ``BM`` are compile-time unrolled.
    """

    @fx.struct
    class _LdsStorage:
        expert: fx.Array[fx.Int32, r]
        slot: fx.Array[fx.Int32, r]
        weight: fx.Array[fx.Float32, r]

    module_name = f"routing_single_r{r}_bm{bm}_e{experts}"

    @flyc.kernel(name=module_name, known_block_size=[_BLOCK_THREADS, 1, 1])
    def routing_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_experts: fx.Tensor,
        num_valid_ids: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        lane = tid
        lds = fx.SharedAllocator().allocate(_LdsStorage).peek()
        neg1 = fx.Int32(-1)
        zero_i32 = fx.Int32(0)
        one_i32 = fx.Int32(1)
        zero_f32 = fx.Float32(0.0)
        pad_id = fx.Int32(1)
        r_i32 = fx.Int32(r)
        bm_i32 = fx.Int32(bm)
        experts_i32 = fx.Int32(experts)
        total_out = fx.Int32(r * bm)
        slot_shift = fx.Int32(24)

        # ---- Phase 1a: initialise LDS expert[] to -1 ----
        init_per_thread = (r + _BLOCK_THREADS - 1) // _BLOCK_THREADS
        for it in range_constexpr(init_per_thread):
            idx = lane + it * _BLOCK_THREADS
            if idx < r:
                fx.ptr_store(neg1, lds.expert.ptr + fx.Int64(idx))

        # ---- Phase 1b: all threads zero global outputs ----
        ids_out = fx.rocdl.make_buffer_tensor(sorted_ids, max_size=True)
        wgts_out = fx.rocdl.make_buffer_tensor(sorted_weights, max_size=True)
        exp_out = fx.rocdl.make_buffer_tensor(sorted_experts, max_size=True)
        nv_out = fx.rocdl.make_buffer_tensor(num_valid_ids, max_size=True)

        out_per_thread = (r * bm + _BLOCK_THREADS - 1) // _BLOCK_THREADS
        for it in range_constexpr(out_per_thread):
            idx = lane + it * _BLOCK_THREADS
            if idx < total_out:
                ids_out[idx] = pad_id
                wgts_out[idx] = zero_f32

        exp_per_thread = (r + _BLOCK_THREADS - 1) // _BLOCK_THREADS
        for it in range_constexpr(exp_per_thread):
            idx = lane + it * _BLOCK_THREADS
            if idx < r_i32:
                exp_out[idx] = neg1

        if tid == 0:
            nv_out[0] = zero_i32
            nv_out[1] = zero_i32

        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        gpu.barrier()

        # ---- Phase 2: load route, compute stable-rank, scatter to LDS ----
        ids_in = fx.rocdl.make_buffer_tensor(topk_ids, max_size=True)
        wgts_in = fx.rocdl.make_buffer_tensor(topk_weights, max_size=True)

        if lane < r_i32:
            expert = ids_in[lane]
            weight = wgts_in[lane]
            valid = (expert >= zero_i32) & (expert < experts_i32)
            if valid:
                pos = zero_i32
                for j in range_constexpr(r):
                    other = ids_in[fx.Int32(j)]
                    o_valid = (other >= zero_i32) & (other < experts_i32)
                    smaller = other < expert
                    same = other == expert
                    j_before = fx.Int32(j) < lane
                    same_and_before = same & j_before
                    contrib = o_valid & (smaller | same_and_before)
                    pos = pos + contrib.select(one_i32, zero_i32)

                fx.ptr_store(expert, lds.expert.ptr + fx.Int64(pos))
                fx.ptr_store(lane, lds.slot.ptr + fx.Int64(pos))
                fx.ptr_store(weight, lds.weight.ptr + fx.Int64(pos))

        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        gpu.barrier()

        # ---- Phase 3: thread 0 walks LDS and writes metadata ----
        if tid == 0:
            num_groups = zero_i32
            num_valid = zero_i32
            cur_exp = neg1
            block_off = zero_i32
            route_in_blk = zero_i32

            for p in range_constexpr(r):
                p_i32 = fx.Int32(p)
                e_val = fx.Int32(
                    fx.ptr_load(
                        lds.expert.ptr + fx.Int64(p_i32),
                        result_type=fx.Int32,
                    )
                )
                if e_val != neg1:
                    num_valid = num_valid + one_i32

                    expert_changed = e_val != cur_exp
                    if expert_changed:
                        cur_exp = e_val
                        exp_out[num_groups] = cur_exp
                        num_groups = num_groups + one_i32
                        block_off = (num_groups - one_i32) * bm_i32
                        route_in_blk = zero_i32

                    s_val = fx.Int32(
                        fx.ptr_load(
                            lds.slot.ptr + fx.Int64(p_i32),
                            result_type=fx.Int32,
                        )
                    )
                    w_val = fx.Float32(
                        fx.ptr_load(
                            lds.weight.ptr + fx.Int64(p_i32),
                            result_type=fx.Float32,
                        )
                    )
                    out_idx = block_off + route_in_blk
                    packed_id = (s_val << slot_shift) | zero_i32
                    ids_out[out_idx] = packed_id
                    wgts_out[out_idx] = w_val
                    route_in_blk = route_in_blk + one_i32

            nv_out[0] = num_groups * bm_i32
            nv_out[1] = num_valid

    @flyc.jit
    def launch_routing(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_experts: fx.Tensor,
        num_valid_ids: fx.Tensor,
        stream: fx.Stream,
    ):
        routing_kernel(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_experts,
            num_valid_ids,
        ).launch(
            grid=(1, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_routing

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged FP4 decode with cooperative head waves and ping-pong LDS stages."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import Float4E2M1FN, T
from flydsl.expr.utils.arith import _to_raw

from aiter.ops.flydsl.kernels import buffer_ops


@lru_cache(maxsize=128)
def compile_pa_mqa_logits_fp4_decode(
    *,
    heads,
    head_dim,
    next_n,
    kv_block_size,
    kv_page_stride,
    kv_scale_page_stride,
    block_table_stride,
    block_k=128,
    num_warps=4,
    head_waves=2,
    stages=2,
    mfma_m=32,
    direct_chunks=0,
    max_seq_len=0,
):
    if heads < 16 or heads > 128 or heads % 16 or head_dim % 128:
        raise ValueError("Decode requires heads=16..128 in steps of 16 and D % 128=0.")
    if kv_block_size not in (64, 128):
        raise ValueError("Decode supports 64- and 128-token KV pages.")
    if head_waves not in (1, 2, 4) or num_warps % head_waves:
        raise ValueError("head_waves must divide num_warps.")
    token_waves = num_warps // head_waves
    if block_k % (token_waves * 64):
        raise ValueError("Each token wave must own complete 64-token fragments.")
    if stages not in (1, 2):
        raise ValueError("Supported LDS stage counts are one and two.")

    fragments = block_k // (token_waves * 64)
    if mfma_m not in (16, 32):
        raise ValueError("MFMA M must be 16 or 32.")
    mfma_k = 128 if mfma_m == 16 else 64
    acc_elements = mfma_m * mfma_m // 64
    n_tiles = 64 // mfma_m
    head_tiles = (heads + mfma_m * head_waves - 1) // (mfma_m * head_waves)
    k_tiles = head_dim // 128
    k_steps = head_dim // mfma_k
    qs_words = (heads // 16 + 3) // 4
    single_chunk = direct_chunks > 0 and direct_chunks * block_k >= max_seq_len
    storage_stages = 1 if single_chunk else stages
    kv_stage_bytes = block_k * head_dim // 2
    scales_per_fragment = k_tiles * 4 * kv_block_size
    scale_stage_bytes = num_warps * fragments * scales_per_fragment
    scales_base = storage_stages * kv_stage_bytes
    partial_base = scales_base + storage_stages * scale_stage_bytes
    lds_bytes = partial_base + (head_waves * block_k * 4 if head_waves > 1 else 0)

    @flyc.kernel
    def pa_mqa_logits_fp4_decode_kernel(
        out, q, qs, kv, kvs, bt, weights, info,
        out_stride: fx.Int32, weight_scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        lane = tid % fx.Int32(64)
        lane_m = lane % fx.Int32(mfma_m)
        lane_k = lane // fx.Int32(mfma_m)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid >> fx.Int32(6)).ir_value()))
        head_wave = wave % fx.Int32(head_waves)
        token_wave = wave // fx.Int32(head_waves)
        info_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(fx.get_iter(info)))
        )
        if const_expr(direct_chunks > 0):
            row = fx.Int32(gpu.block_idx.x) // fx.Int32(direct_chunks)
            context = fx.Int32(buffer_ops.buffer_load(
                info_rsrc, row // fx.Int32(next_n),
                vec_width=1, dtype=fx.Int32, is_scalar=True,
            ))
            split = fx.Int32(gpu.block_idx.x) % fx.Int32(direct_chunks)
            if const_expr(single_chunk):
                start = split
                count = fx.Int32(1)
            else:
                chunks = (context + fx.Int32(block_k - 1)) // fx.Int32(block_k)
                span = (chunks + fx.Int32(direct_chunks - 1)) // fx.Int32(direct_chunks)
                start = split * span
                remaining = chunks - start
                count = (remaining < span).select(remaining, span)
        else:
            record = fx.Vector(buffer_ops.buffer_load(
                info_rsrc, fx.Int32(gpu.block_idx.x) * fx.Int32(4),
                vec_width=4, dtype=fx.Int32, is_scalar=True,
            ))
            row = fx.Int32(record[0])
            start = fx.Int32(record[1])
            count = fx.Int32(record[2])
            context = fx.Int32(record[3])
        batch = row // fx.Int32(next_n)
        visible_raw = context - fx.Int32(next_n - 1) + row % fx.Int32(next_n)
        visible = (visible_raw > fx.Int32(0)).select(visible_raw, fx.Int32(0))
        shared = fx.SharedAllocator().allocate(lds_bytes, alignment=16)._ptr
        lds_addr = fx.Int32(fx.ptrtoint(shared))
        lds_i32 = fx.recast_iter(fx.Int32, shared)
        lds_f32 = fx.recast_iter(fx.Float32, shared)
        shared_ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared, 16)
        copy128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        copy32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        copy_float = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        vec_layout = fx.make_layout(4, 1)
        scalar_layout = fx.make_layout(1, 1)

        def resource(tensor, byte_offset, size):
            return buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(fx.get_iter(tensor))) + fx.Int64(byte_offset),
                num_records_bytes=size,
            )

        q_rsrc = resource(q, fx.Int64(row) * fx.Int64(heads * head_dim // 2),
                          heads * head_dim // 2)
        qs_rsrc = resource(qs, fx.Int64(row) * fx.Int64(k_tiles * 4 * 16 * qs_words * 4),
                           k_tiles * 4 * 16 * qs_words * 4)
        w_rsrc = resource(weights, fx.Int64(row) * fx.Int64(heads * 2), heads * 2)
        bt_rsrc = resource(bt, fx.Int64(batch) * fx.Int64(block_table_stride * 4),
                           block_table_stride * 4)
        out_rsrc = resource(out, fx.Int64(row) * fx.Int64(out_stride) * fx.Int64(4),
                            visible * fx.Int32(4))
        mfma = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(
            mfma_m, mfma_m, mfma_k, Float4E2M1FN, Float4E2M1FN, opsel_a=0, opsel_b=0,
        ))
        zero = fx.Vector.filled(acc_elements, 0.0, fx.Float32)

        def lds_load4(offset):
            reg = fx.make_rmem_tensor(4, fx.Int32)
            view = fx.make_view(fx.add_offset(lds_i32, offset), vec_layout)
            fx.copy(copy128, view, reg)
            return fx.memref_load_vec(reg)

        def lds_load1(offset):
            reg = fx.make_rmem_tensor(1, fx.Int32)
            view = fx.make_view(fx.add_offset(lds_i32, offset), scalar_layout)
            fx.copy(copy32, view, reg)
            return fx.memref_load_vec(reg)[0]

        def dma(rsrc, dst, src, width):
            ptr = fx.to_llvm_ptr(fx.inttoptr(shared_ptr_ty, fx.Int64(lds_addr + dst)))
            rocdl.raw_ptr_buffer_load_lds(
                rsrc, ptr, fx.Int32(width), src, fx.Int32(0), fx.Int32(0), fx.Int32(0),
            )

        def issue_stage(chunk, stage):
            for frag in range_constexpr(fragments):
                fragment = token_wave * fx.Int32(fragments) + fx.Int32(frag)
                token = (start + chunk) * fx.Int32(block_k) + fragment * fx.Int32(64)
                phys = fx.Int32(buffer_ops.buffer_load(
                    bt_rsrc, token // fx.Int32(kv_block_size),
                    vec_width=1, dtype=fx.Int32, is_scalar=True,
                ))
                valid = token < context
                kv_rsrc = resource(kv, fx.Int64(phys) * fx.Int64(kv_page_stride),
                                   valid.select(fx.Int32(kv_block_size * head_dim // 2),
                                                fx.Int32(0)))
                scale_rsrc = resource(kvs, fx.Int64(phys) * fx.Int64(kv_scale_page_stride),
                                      valid.select(fx.Int32(scales_per_fragment), fx.Int32(0)))
                for kt in range_constexpr(k_tiles):
                    for p in range_constexpr(4 // head_waves):
                        plane = fx.Int32(kt * 4 + p * head_waves) + head_wave
                        dst = (stage * fx.Int32(kv_stage_bytes)
                               + plane * fx.Int32(block_k * 16)
                               + fragment * fx.Int32(64 * 16))
                        src = (plane * fx.Int32(kv_block_size * 16)
                               + (token % fx.Int32(kv_block_size) + lane) * fx.Int32(16))
                        dma(kv_rsrc, dst, src, 16)
                    for part in range_constexpr(kv_block_size // 64):
                        dst = (fx.Int32(scales_base) + stage * fx.Int32(scale_stage_bytes)
                               + (wave * fx.Int32(fragments) + fx.Int32(frag))
                               * fx.Int32(scales_per_fragment)
                               + fx.Int32(kt * 4 * kv_block_size + part * 256))
                        src = fx.Int32(kt * 4 * kv_block_size + part * 256) + lane * fx.Int32(4)
                        dma(scale_rsrc, dst, src, 4)

        def run_active():
            q_fragments = []
            q_scales = []
            weight_fragments = []
            for mi in range_constexpr(head_tiles):
                head_tile = head_wave + fx.Int32(mi * head_waves)
                q_mi = []
                qs_mi = []
                for ks in range_constexpr(k_steps):
                    q_offset = ((head_tile * fx.Int32(mfma_m) + lane_m)
                                * fx.Int32(head_dim // 8) + fx.Int32(ks * mfma_k // 8)
                                + lane_k * fx.Int32(4))
                    q_vec = fx.Vector(buffer_ops.buffer_load(
                        q_rsrc, q_offset, vec_width=4, dtype=fx.Int32,
                    ))
                    q_reg = fx.make_rmem_tensor(4, fx.Int32)
                    q_reg.store(q_vec)
                    q_mi.append(q_reg)
                    head16 = head_tile * fx.Int32(mfma_m // 16)
                    qs_offset = ((fx.Int32(ks * mfma_k // 32) + lane_k)
                                 * fx.Int32(16 * qs_words)
                                 + (lane_m & fx.Int32(15)) * fx.Int32(qs_words)
                                 + head16 // fx.Int32(4))
                    qs_word = fx.Int32(buffer_ops.buffer_load(
                        qs_rsrc, qs_offset, vec_width=1, dtype=fx.Int32,
                    ))
                    shift = ((head16 % fx.Int32(4))
                             + (lane_m >> fx.Int32(4))) * fx.Int32(8)
                    qs_mi.append(fx.Int32(fx.Uint32(qs_word) >> fx.Uint32(shift)))
                q_fragments.append(q_mi)
                q_scales.append(qs_mi)
                w_mi = []
                for group in range_constexpr(acc_elements // 4):
                    w_offset = (head_tile * fx.Int32(mfma_m // 2) + fx.Int32(group * 4)
                                + lane_k * fx.Int32(2))
                    w_vec = fx.Vector(buffer_ops.buffer_load(
                        w_rsrc, w_offset, vec_width=2, dtype=fx.Int32,
                    )).bitcast(fx.BFloat16).to(fx.Float32)
                    w_mi.append(w_vec)
                weight_fragments.append(w_mi)

            def finish_pair(sum0, sum1):
                pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
                pair = rocdl.permlane32_swap(
                    pair_ty, _to_raw(sum0.bitcast(fx.Int32)),
                    _to_raw(sum1.bitcast(fx.Int32)), False, True,
                )
                lhs = fx.Int32(llvm.extractvalue(T.i32, pair, [0]))
                rhs = fx.Int32(llvm.extractvalue(T.i32, pair, [1]))
                return lhs.bitcast(fx.Float32) + rhs.bitcast(fx.Float32)

            def compute_stage(chunk, stage):
                for frag in range_constexpr(fragments):
                    fragment = token_wave * fx.Int32(fragments) + fx.Int32(frag)
                    token_local = fragment * fx.Int32(64)
                    token_global = (start + chunk) * fx.Int32(block_k) + token_local
                    accs = []
                    for nt in range_constexpr(n_tiles):
                        acc_m = []
                        for mi in range_constexpr(head_tiles):
                            acc = zero
                            for ks in range_constexpr(k_steps):
                                offset = (stage * fx.Int32(kv_stage_bytes // 4)
                                          + (fx.Int32(ks * mfma_k // 32) + lane_k)
                                          * fx.Int32(block_k * 4)
                                          + (token_local + fx.Int32(nt * mfma_m) + lane_m)
                                          * fx.Int32(4))
                                b_reg = fx.make_rmem_tensor(4, fx.Int32)
                                b_reg.store(lds_load4(offset))
                                page_token = ((token_global + fx.Int32(nt * mfma_m) + lane_m)
                                              % fx.Int32(kv_block_size))
                                scale_offset = (
                                    fx.Int32(scales_base // 4)
                                    + stage * fx.Int32(scale_stage_bytes // 4)
                                    + (wave * fx.Int32(fragments) + fx.Int32(frag))
                                    * fx.Int32(scales_per_fragment // 4)
                                    + (fx.Int32(ks * mfma_k // 32) + lane_k)
                                    * fx.Int32(kv_block_size // 4)
                                    + page_token % fx.Int32(kv_block_size // 4)
                                )
                                scale_word = lds_load1(scale_offset)
                                scale_shift = (page_token // fx.Int32(kv_block_size // 4)) * fx.Int32(8)
                                scale_b = fx.Int32(fx.Uint32(scale_word) >> fx.Uint32(scale_shift))
                                c_reg = fx.make_rmem_tensor(acc_elements, fx.Float32)
                                c_reg.store(acc)
                                fx.gemm(mfma, c_reg, q_fragments[mi][ks], b_reg, c_reg,
                                        scale_a=q_scales[mi][ks], scale_b=scale_b)
                                acc = c_reg.load()
                            acc_m.append(fx.Vector(acc))
                        accs.append(acc_m)
                    sums = []
                    for nt in range_constexpr(n_tiles):
                        total = fx.Float32(0.0)
                        for mi in range_constexpr(head_tiles):
                            relu = fx.maxnumf(accs[nt][mi], zero,
                                             fastmath=fx.arith.FastMathFlags.nnan)
                            for elem in range_constexpr(acc_elements):
                                w = weight_fragments[mi][elem // 4][elem % 4]
                                total = fx.fma(relu[elem], w, total)
                        sums.append(total)
                    def store_partial(total, token_offset):
                        if const_expr(head_waves > 1):
                            offset = (fx.Int32(partial_base // 4) + head_wave * fx.Int32(block_k)
                                      + token_local + token_offset)
                            view = fx.make_view(fx.add_offset(lds_f32, offset), scalar_layout)
                            reg = fx.make_rmem_tensor(1, fx.Float32)
                            reg.store(fx.Vector.from_elements([total], dtype=fx.Float32))
                            fx.copy(copy_float, reg, view)
                        else:
                            buffer_ops.buffer_store(total * weight_scale, out_rsrc,
                                                    token_global + token_offset)
                    if const_expr(mfma_m == 32):
                        store_partial(finish_pair(sums[0], sums[1]), lane)
                    else:
                        for nt in range_constexpr(n_tiles):
                            total = sums[nt]
                            for sh in range_constexpr(2):
                                peer = rocdl.ds_bpermute(
                                    T.i32, (lane ^ fx.Int32(16 << sh)) * fx.Int32(4),
                                    total.bitcast(fx.Int32),
                                )
                                total = total + fx.Int32(peer).bitcast(fx.Float32)
                            if lane < fx.Int32(16):
                                store_partial(total, fx.Int32(nt * 16) + lane)
                if const_expr(head_waves > 1):
                    rocdl.s_waitcnt(lgkmcnt=0)
                    gpu.barrier()
                    if head_wave == fx.Int32(0):
                        for frag in range_constexpr(fragments):
                            token_local = (token_wave * fx.Int32(fragments) + fx.Int32(frag)) * fx.Int32(64) + lane
                            total = fx.Float32(0.0)
                            for hw in range_constexpr(head_waves):
                                offset = fx.Int32(partial_base // 4 + hw * block_k) + token_local
                                total = total + lds_load1(offset).bitcast(fx.Float32)
                            buffer_ops.buffer_store(total * weight_scale, out_rsrc,
                                                    (start + chunk) * fx.Int32(block_k) + token_local)

            issue_stage(fx.Int32(0), fx.Int32(0))
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            for chunk in range(count - fx.Int32(1)):
                chunk = fx.Int32(chunk)
                stage = chunk % fx.Int32(stages)
                if const_expr(stages == 2):
                    issue_stage(chunk + fx.Int32(1), stage ^ fx.Int32(1))
                compute_stage(chunk, stage)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                gpu.barrier()
                if const_expr(stages == 1):
                    issue_stage(chunk + fx.Int32(1), fx.Int32(0))
                    rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
            last = count - fx.Int32(1)
            compute_stage(last, last % fx.Int32(stages))

        if (count > fx.Int32(0)) & (start * fx.Int32(block_k) < visible):
            run_active()

    @flyc.jit
    def launch(out, q, qs, kv, kvs, bt, weights, info,
               out_stride: fx.Int32, weight_scale: fx.Float32,
               grid: fx.Int32, stream: fx.Stream):
        pa_mqa_logits_fp4_decode_kernel(
            out, q, qs, kv, kvs, bt, weights, info, out_stride, weight_scale,
        ).launch(grid=(fx.Int64(grid),), block=(num_warps * 64, 1, 1), stream=stream)

    return launch, num_warps * 64

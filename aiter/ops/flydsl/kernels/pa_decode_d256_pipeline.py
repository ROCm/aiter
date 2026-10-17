# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""Experimental gfx950 D256 PA with wave-local online softmax.

Four waves own (16 query rows, 32 KV tokens) each. QL1/2 share one
query group and split PV into D128 slices, avoiding a masked second query
group. QL3/4 use two query groups and full D256 output per wave. Explicit
head_splits=2 retains the eight-wave per-tensor experiment. Token halves are merged
once in the epilogue. P and softmax statistics stay wave-local in the loop.
K/V are copied directly into an LDS ring, with non-temporal loads and
page descriptors in SGPRs. Ring slots are expanded at compile time. K for
the future tile is issued before QK, V after QK, and page IDs are prefetched
at the end of PV. Per-token scales use the same LDS ring as K/V. K scales multiply QK
logits, and V scales multiply probabilities before FP8 conversion. A running
V-scale maximum normalizes P; output is restored once in the epilogue.
This is an opt-in pipeline experiment. QK uses K128 MFMA, PV uses K32 MFMA.
The work-plan and packed-partial ABI match the general PA kernel (plan
tiles remain 256).
V must already be token-major; this experimental factory is not dispatched
by the public PA wrapper, whose V layouts differ.
"""

from hashlib import sha256
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from aiter.jit.utils.chip_info import get_gfx_runtime

from . import buffer_ops, dpp_utils
from .pa_decode.traits import LOG2E
from .tensor_shim import buf_base_i64, buf_load_scalar, ptr_rsrc
from .utils import rcp_f32

_CACHE = {}


def compile_pa_decode_d256_pipeline(
    *,
    num_buffers=3,
    waves_per_eu=2,
    tile_tokens=64,
    head_splits=1,
    k_swizzle=8,
    cache_modifier=2,
    **kwargs,
):
    """Compile the gfx950 QL1-4/GQA8 specialization with token-major V.

    K is [pages, heads, 16, 16, 16]; V is [pages, heads, 16, 256].
    K/V scales are device-resident FP32 scalars or contiguous per-token
    arrays [pages, heads, 16, 1]; K/V scale strides must match. QL1/2
    automatically share one query group across two D128 output slices. The launch ABI
    matches the general tile kernel, but the V layout does not.
    """
    if not (
        kwargs["head_dim"] == 256
        and kwargs["query_group_size"] == 8
        and (kwargs["query_length"] in (1, 2, 3, 4))
        and (kwargs["block_size"] == 16)
        and (kwargs["kv_dtype"] == "fp8")
        and (kwargs["query_dtype"] == "bf16")
        and (kwargs["sliding_window"] == 0)
    ):
        raise NotImplementedError(
            "wave-local D256 PA requires QL1-4/GQA8/page16 FP8 KV"
        )
    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError("wave-local D256 PA requires gfx950")
    if not kwargs.get("kv_buffer_u32", False):
        raise NotImplementedError("wave-local D256 PA requires KV caches below 4 GiB")
    if tile_tokens != 64:
        raise ValueError("tile_tokens must be 64")
    if not 2 <= num_buffers <= 4:
        raise ValueError("num_buffers must be in [2, 4]")
    if head_splits not in (1, 2):
        raise ValueError("head_splits must be 1 or 2")
    # Reuse all four page-owning waves for the single QL1/2 query group.
    short_query = kwargs["query_length"] <= 2 and head_splits == 1
    output_splits = 2 if short_query else head_splits
    waves = 4 if short_query else 4 * head_splits
    pv_group = 4 if short_query else 1
    block_threads = waves * 64
    d_chunks = 16 // output_splits
    if k_swizzle not in (1, 2, 4, 8):
        raise ValueError("k_swizzle must be 1, 2, 4, or 8")
    if cache_modifier not in (0, 2):
        raise ValueError("cache_modifier must be 0 (default) or 2 (non-temporal)")
    query_length = kwargs["query_length"]
    query_rows = query_length * 8
    per_token_kv = kwargs["per_token_kv"]
    if per_token_kv and head_splits != 1:
        raise NotImplementedError("per-token KV currently requires head_splits=1")
    scale = kwargs["softmax_scale"]
    if scale is None:
        scale = 256 ** (-0.5)
    cache_tag = sha256(Path(__file__).read_bytes()).hexdigest()
    key = (
        scale,
        query_length,
        num_buffers,
        waves_per_eu,
        tile_tokens,
        head_splits,
        k_swizzle,
        cache_modifier,
        per_token_kv,
        cache_tag,
    )
    if key in _CACHE:
        return _CACHE[key]
    half_tokens = tile_tokens // 2
    n_chunks = half_tokens // 16
    k_bytes = tile_tokens * 256
    scale_offset = k_bytes * 2
    slot_bytes = scale_offset + (512 if per_token_kv else 0)
    p_stride = max(64, half_tokens)
    p_wave = 16 * p_stride
    q_off = num_buffers * slot_bytes
    qs_off = q_off + (4096 if short_query else 8192)
    p_off = qs_off + (64 if short_query else 128)
    page_off = p_off + waves * p_wave
    total_bytes = max(page_off, 65536 + 512)

    @fx.struct
    class Storage:
        buf: fx.Array[fx.Int32, total_bytes // 4, 128]

    @flyc.kernel(known_block_size=(block_threads, 1, 1))
    def kernel(
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        query_ptr: fx.Pointer,
        key_ptr: fx.Pointer,
        value_ptr: fx.Pointer,
        tables_ptr: fx.Pointer,
        ks_ptr: fx.Pointer,
        vs_ptr: fx.Pointer,
        table_stride: fx.Int32,
        ks_block: fx.Int32,
        ks_head: fx.Int32,
        q_row: fx.Int32,
        q_head: fx.Int32,
        work_ptr: fx.Pointer,
    ):
        _ = (cache_tag, waves_per_eu, tile_tokens)
        tid = fx.Int32(gpu.thread_id("x"))
        wave, lane = (tid // 64, tid % 64)
        l16, rg = (lane % 16, lane // 16)
        qgroup, half = (wave // (2 * output_splits), wave % 2)
        dhalf = wave // 2 % output_splits
        task_id, kv_head = (fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y")))
        capacity, kv_heads = (fx.Int32(gpu.grid_dim.x), fx.Int32(gpu.grid_dim.y))
        task = fx.ptr_load(
            fx.recast_iter(fx.Int32, work_ptr) + task_id * 4,
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
        start, end = (
            fx.Int32(task[1]) * (256 // tile_tokens),
            fx.Int32(task[2]) * (256 // tile_tokens),
        )
        if start < end:
            seq, context = (fx.Int32(task[0]), fx.Int32(task[3]))
            end = fx.min(end, (context + tile_tokens - 1) // tile_tokens)
            pages = (context + 15) // 16
            smem = fx.SharedAllocator().allocate(Storage).peek()
            base = fx.recast_iter(fx.Uint8, smem.buf.ptr)

            def ptr(off, dtype):
                return fx.recast_iter(
                    fx.PointerType.get(
                        dtype.ir_type, fx.AddressSpace.Shared, dsl_size_of(dtype)
                    ),
                    base + off,
                )

            def load(off, dtype, n):
                return fx.Vector(
                    llvm.load(
                        T.vec(n, dtype.ir_type),
                        fx.as_ir_value(fx.to_llvm_ptr(ptr(off, dtype))),
                        alignment=dsl_size_of(dtype),
                    )
                )

            def store(off, val):
                llvm.store(
                    fx.as_ir_value(val),
                    fx.as_ir_value(fx.to_llvm_ptr(ptr(off, fx.Int32))),
                    alignment=4,
                )

            def vec(val, n=4):
                return fx.Vector.from_elements([val], dtype=fx.Float32).broadcast_to(n)

            def words(val):
                result = []
                for i in range_constexpr(val.shape[0] // 4):
                    lo = fx.rocdl.cvt_pk_fp8_f32(
                        T.i32, val[i * 4], val[i * 4 + 1], 0, False
                    )
                    result.append(
                        fx.rocdl.cvt_pk_fp8_f32(
                            T.i32, val[i * 4 + 2], val[i * 4 + 3], lo, True
                        )
                    )
                return fx.Vector.from_elements(result, dtype=fx.Int32)

            def wide_mma(a, b, acc):
                return fx.Vector(
                    fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.f32x4,
                        [
                            a,
                            b,
                            acc,
                            0,
                            0,
                            0,
                            fx.Int32(2139062143),
                            0,
                            fx.Int32(2139062143),
                        ],
                    )
                )

            table_rsrc = ptr_rsrc(tables_ptr)
            if fx.const_expr(per_token_kv):
                key_scale = fx.Float32(1.0)
                value_scale = fx.Float32(1.0)
                ks_rsrc, vs_rsrc = (ptr_rsrc(ks_ptr), ptr_rsrc(vs_ptr))
            else:
                key_scale = fx.Float32(fx.ptr_load(fx.recast_iter(fx.Float32, ks_ptr)))
                value_scale = fx.Float32(
                    fx.ptr_load(fx.recast_iter(fx.Float32, vs_ptr))
                )

            def async_dma(rsrc, idx, dest, byte_count=16):
                dst = fx.to_llvm_ptr(
                    base + fx.Int32(fx.rocdl.readfirstlane(T.i32, dest))
                )
                rocdl_ir.raw_ptr_buffer_load_async_lds(
                    rsrc,
                    dst,
                    fx.Int32(byte_count).ir_value(),
                    fx.Int32(idx).ir_value(),
                    fx.Int32(0).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(32), cache_modifier
                    ),
                )

            def load_page(tt):
                page = fx.Int32(fx.rocdl.readfirstlane(T.i32, wave % 4))
                idx = seq * table_stride + fx.min(tt * 4 + page, pages - 1)
                return fx.Int32(buf_load_scalar(table_rsrc, idx, dwords=1))

            def dma(tt, slot, physical=None, copy_k=True, copy_v=True):
                page = fx.Int32(fx.rocdl.readfirstlane(T.i32, wave % 4))
                sub = wave // 4
                page_idx = tt * 4 + page
                if physical is None:
                    physical = load_page(tt)
                page_bytes = (fx.Int64(physical) * kv_heads + kv_head) * 4096
                kpage = buffer_ops.create_buffer_resource_from_addr(
                    buf_base_i64(key_ptr) + page_bytes
                )
                live = fx.min(fx.max(context - page_idx * 16, 0), 16)
                vpage = buffer_ops.create_buffer_resource_from_addr(
                    buf_base_i64(value_ptr) + page_bytes,
                    num_records_bytes=fx.Int32(
                        fx.rocdl.readfirstlane(T.i32, live * 256)
                    ),
                )
                # Each page-owning wave copies its 16 FP32 K/V scales to LDS.
                # The scale arrays share block/head strides and token stride 1.
                if fx.const_expr(per_token_kv and copy_k):  # noqa: SIM102
                    if lane < 16:
                        scale_idx = physical * ks_block + kv_head * ks_head + lane
                        scale_dest = slot * slot_bytes + scale_offset + page * 64
                        async_dma(ks_rsrc, scale_idx * 4, scale_dest, byte_count=4)
                        async_dma(
                            vs_rsrc, scale_idx * 4, scale_dest + 256, byte_count=4
                        )
                for i in range_constexpr(16 // waves):
                    unit = sub * 64 + lane + i * (waves // 4) * 64
                    kh, kt = (unit // 16, unit % 16)
                    src_kt = kt ^ kh % k_swizzle
                    dest = page * 4096 + (sub + i * (waves // 4)) * 1024
                    if fx.const_expr(copy_k):
                        async_dma(
                            kpage, kh * 256 + src_kt * 16, slot * slot_bytes + dest
                        )
                    if fx.const_expr(copy_v):
                        token, band = (unit // 16, unit % 16)
                        src_band = band ^ token % 8 ^ page % 2 * 8
                        async_dma(
                            vpage,
                            token * 256 + src_band * 16,
                            slot * slot_bytes + k_bytes + dest,
                        )

            # Two token halves stage eight Q rows each, then both read 16 rows.
            qr = qgroup * 16 + half * 8 + rg * 2 + l16 // 8
            qchunk = l16 % 8 * 32
            qidx = (
                (seq * query_length + fx.min(qr // 8, query_length - 1)) * q_row
                + (kv_head * 8 + qr % 8) * q_head
                + qchunk
            )
            if dhalf == 0:
                qvalues = fx.ptr_load(
                    fx.recast_iter(fx.BFloat16, query_ptr) + qidx,
                    result_type=fx.Vector.make_type(32, fx.BFloat16),
                ).to(fx.Float32)
                qmax = fx.Vector(fmath.absf(qvalues)).reduce(ReductionOp.MAX)
                for sh in (4, 2, 1):
                    qmax = fx.maxnumf(qmax, dpp_utils.dpp_xor_f32(qmax, sh))
                qscale = qmax * fx.Float32(1.0 / 448.0)
                invq = fx.Float32(rcp_f32(fx.maxnumf(qscale, fx.Float32(1e-20))))
                store(
                    q_off + qr * 256 + (qchunk ^ qr % 8 * 32),
                    words(qvalues * vec(invq, 32)),
                )
                if l16 % 8 == 0:
                    store(qs_off + qr * 4, vec(qscale, 1).bitcast(fx.Int32))
            for stage in range_constexpr(num_buffers - 1):
                if start + stage < end:
                    dma(start + stage, fx.Int32(stage))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            qops = []
            for k in range_constexpr(2):
                chunks = [
                    load(
                        q_off
                        + (qgroup * 16 + l16) * 256
                        + (rg * 16 + k * 128 + j * 64 ^ l16 % 8 * 32),
                        fx.Int32,
                        4,
                    )
                    for j in range_constexpr(2)
                ]
                qops.append(
                    fx.Vector.from_elements(
                        [
                            chunks[j][r]
                            for j in range_constexpr(2)
                            for r in range_constexpr(4)
                        ],
                        dtype=fx.Int32,
                    )
                )
            qsc = fx.Float32(load(qs_off + (qgroup * 16 + l16) * 4, fx.Float32, 1)[0])
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            init = [fx.Float32(float("-inf")), fx.Float32(0.0)] + [zero] * d_chunks
            if fx.const_expr(per_token_kv):
                init.append(fx.Float32(0.0))
            init.append(load_page(start + num_buffers - 1))
            for block_tt, block_state in range(start, end, num_buffers, init=init):
                current = block_state
                for local_slot in range_constexpr(num_buffers):
                    tt = fx.Int32(block_tt) + local_slot
                    slot = fx.Int32(local_slot)
                    state = current
                    if tt < end:
                        # Keep future DMA in flight: 8 KV + 2 scale instructions
                        # per page-owning wave for the per-token four-wave path.
                        if tt + num_buffers - 2 < end:
                            llvm.InlineAsmOp(
                                res=None,
                                operands_=[],
                                asm_string=f"s_waitcnt vmcnt({(32 // waves + (2 if per_token_kv else 0)) * (num_buffers - 2)})",
                                constraints="",
                                has_side_effects=True,
                                is_align_stack=False,
                            )
                        else:
                            fx.rocdl.s_waitcnt(vmcnt=0)
                        fx.rocdl.s_waitcnt(lgkmcnt=0)
                        llvm.InlineAsmOp(
                            res=None,
                            operands_=[],
                            asm_string="s_barrier",
                            constraints="",
                            has_side_effects=True,
                            is_align_stack=False,
                        )
                        future = tt + num_buffers - 1
                        if future < end:
                            dma(
                                future,
                                fx.Int32((local_slot + num_buffers - 1) % num_buffers),
                                state[-1],
                                copy_v=False,
                            )
                        kscales, vscales = ([], [])
                        if fx.const_expr(per_token_kv):
                            for n in range_constexpr(n_chunks):
                                soff = (
                                    slot * slot_bytes
                                    + scale_offset
                                    + (half * 2 + n) * 64
                                    + rg * 16
                                )
                                kscales.append(load(soff, fx.Float32, 4))
                                vv = load(soff + 256, fx.Float32, 4)
                                scale_tokens = fx.Vector.from_elements(
                                    [
                                        half * half_tokens + n * 16 + rg * 4 + r
                                        for r in range_constexpr(4)
                                    ],
                                    dtype=fx.Int32,
                                )
                                vscales.append(
                                    (
                                        scale_tokens
                                        < fx.Vector.from_elements(
                                            [context - tt * tile_tokens], dtype=fx.Int32
                                        ).broadcast_to(4)
                                    ).select(vv, fx.Vector.filled(4, 0.0, fx.Float32))
                                )
                        score = []
                        for n in range_constexpr(n_chunks):
                            tok = half * half_tokens + n * 16 + l16
                            acc = zero
                            for k in range_constexpr(2):
                                aa = []
                                for j in range_constexpr(2):
                                    aa.append(
                                        load(
                                            slot * slot_bytes
                                            + (
                                                tok // 16 * 256
                                                + (k * 8 + rg + j * 4) * 16
                                                + (tok % 16 ^ (rg + j * 4) % k_swizzle)
                                            )
                                            * 16,
                                            fx.Int32,
                                            4,
                                        )
                                    )
                                kop = fx.Vector.from_elements(
                                    [
                                        aa[j][r]
                                        for j in range_constexpr(2)
                                        for r in range_constexpr(4)
                                    ],
                                    dtype=fx.Int32,
                                )
                                acc = wide_mma(kop, qops[k], acc)
                            st = half * half_tokens + n * 16 + rg * 4
                            tokens = fx.Vector.from_elements(
                                [st + r for r in range_constexpr(4)], dtype=fx.Int32
                            )
                            visible = (
                                context
                                - (query_length - 1)
                                + (qgroup * 16 + l16) // 8
                                - tt * tile_tokens
                            )
                            valid = (
                                tokens
                                < fx.Vector.from_elements(
                                    [visible], dtype=fx.Int32
                                ).broadcast_to(4)
                            ) & fx.Vector.from_elements(
                                [qgroup * 16 + l16 < query_rows], dtype=fx.Boolean
                            ).broadcast_to(4)
                            scaled = acc * vec(
                                key_scale * qsc * fx.Float32(scale * LOG2E)
                            )
                            if fx.const_expr(per_token_kv):
                                scaled = scaled * kscales[n]
                            score.append(
                                valid.select(
                                    scaled,
                                    fx.Vector.filled(4, float("-inf"), fx.Float32),
                                )
                            )
                        if future < end:
                            dma(
                                future,
                                fx.Int32((local_slot + num_buffers - 1) % num_buffers),
                                state[-1],
                                copy_k=False,
                            )
                        m = score[0].reduce(ReductionOp.MAX)
                        for n in range_constexpr(1, n_chunks):
                            m = fx.maxnumf(m, score[n].reduce(ReductionOp.MAX))
                        for sh in (16, 32):
                            m = fx.maxnumf(m, m.shuffle_xor(sh, 64))
                        new_m = fx.maxnumf(state[0], m)
                        safe_m = (new_m > fx.Float32(float("-inf"))).select(
                            new_m, fx.Float32(0.0)
                        )
                        corr = fx.Float32(1.0)
                        old_l = state[1]
                        old_o = fx.Vector.from_elements(
                            [
                                state[d + 2][r]
                                for d in range_constexpr(d_chunks)
                                for r in range_constexpr(4)
                            ],
                            dtype=fx.Float32,
                        )
                        need_rescale = (
                            fx.Int64(fx.rocdl.ballot(T.i64, new_m != state[0])) != 0
                        )
                        if need_rescale:
                            corr = fx.Float32(
                                fx.exp2(state[0] - safe_m, fastmath="fast")
                            )
                            old_l = old_l * corr
                            if fx.const_expr(not per_token_kv):
                                old_o = old_o * vec(corr, d_chunks * 4)
                        vnorm = fx.Float32(1.0)
                        inv_vnorm = fx.Float32(1.0)
                        if fx.const_expr(per_token_kv):
                            vmax = vscales[0].reduce(ReductionOp.MAX)
                            for n in range_constexpr(1, n_chunks):
                                vmax = fx.maxnumf(
                                    vmax, vscales[n].reduce(ReductionOp.MAX)
                                )
                            for sh in (16, 32):
                                vmax = fx.maxnumf(vmax, vmax.shuffle_xor(sh, 64))
                            # O is stored in units of the running V-scale maximum.
                            # Restore physical units once in the epilogue, instead
                            # of applying V-scale to every output fragment per tile.
                            prior_norm = state[-2]
                            new_norm = fx.maxnumf(prior_norm, vmax)
                            vnorm = (new_norm > 0).select(new_norm, fx.Float32(1.0))
                            inv_vnorm = fx.Float32(rcp_f32(vnorm))
                            old_o = old_o * vec(
                                corr * prior_norm * inv_vnorm, d_chunks * 4
                            )
                        l = fx.Float32(0.0)
                        for n in range_constexpr(n_chunks):
                            prob = fx.Vector(
                                fx.exp2(
                                    score[n] - vec(safe_m) + vec(fx.Float32(8.0)),
                                    fastmath="fast",
                                )
                            )
                            l = l + prob.reduce(ReductionOp.ADD)
                            if fx.const_expr(per_token_kv):
                                prob = prob * vscales[n] * vec(inv_vnorm)
                            store(
                                p_off
                                + wave * p_wave
                                + l16 * p_stride
                                + (n * 16 + rg * 4 ^ l16 % 8 * 8),
                                words(prob),
                            )
                        for sh in (16, 32):
                            l = l + l.shuffle_xor(sh, 64)
                        # P is wave-private; no workgroup barrier is necessary.
                        fx.rocdl.s_waitcnt(lgkmcnt=0)
                        pops = [
                            load(
                                p_off
                                + wave * p_wave
                                + l16 * p_stride
                                + (rg * 8 + pack * 32 ^ l16 % 8 * 8),
                                fx.Int64,
                                1,
                            )[0]
                            for pack in range_constexpr(half_tokens // 32)
                        ]
                        next_state = [new_m, old_l + l]
                        vt = half * 32 + rg * 8 + l16 // 2
                        # Read several V fragments before consuming them in MFMA.
                        for group in range_constexpr(d_chunks // pv_group):
                            vops = []
                            for within in range_constexpr(pv_group):
                                d = group * pv_group + within
                                band = d + dhalf * d_chunks ^ vt % 8 ^ vt // 16 % 2 * 8
                                voff = (
                                    slot * slot_bytes
                                    + k_bytes
                                    + vt * 256
                                    + band * 16
                                    + l16 % 2 * 8
                                )
                                vops.append(
                                    fx.Vector(
                                        rocdl_ir.ds_read_tr8_b64(
                                            T.vec(2, T.i32),
                                            fx.as_ir_value(
                                                fx.to_llvm_ptr(ptr(voff, fx.Int32))
                                            ),
                                        ).result
                                    ).bitcast(fx.Int64)[0]
                                )
                            for within in range_constexpr(pv_group):
                                d = group * pv_group + within
                                previous = fx.Vector.from_elements(
                                    [old_o[d * 4 + r] for r in range_constexpr(4)],
                                    dtype=fx.Float32,
                                )
                                op = previous
                                op = fx.Vector(
                                    fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                                        T.f32x4, [vops[within], pops[0], op, 0, 0, 0]
                                    )
                                )
                                next_state.append(op)
                        if fx.const_expr(per_token_kv):
                            next_state.append(new_norm)
                        next_state.append(load_page(tt + num_buffers))
                        current = next_state
                results = yield current
            # Reuse the dead K/V ring for the once-only token-half merge.
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()
            for d in range_constexpr(d_chunks):
                off = ((wave * d_chunks + d) * 64 + lane) * 16
                out_fragment = results[d + 2]
                if fx.const_expr(per_token_kv):
                    out_fragment = out_fragment * vec(results[-2])
                store(off, out_fragment.bitcast(fx.Int32))
            if rg == 0 and dhalf == 0:
                stat_wave = qgroup * 2 + half
                store(
                    65536 + (stat_wave * 16 + l16) * 4,
                    vec(results[0], 1).bitcast(fx.Int32),
                )
                store(
                    65792 + (stat_wave * 16 + l16) * 4,
                    vec(results[1], 1).bitcast(fx.Int32),
                )
            gpu.barrier()
            if half == 0:
                row = qgroup * 16 + l16
                w0 = qgroup * 2 * output_splits + dhalf * 2
                w1 = w0 + 1
                sw0, sw1 = (qgroup * 2, qgroup * 2 + 1)
                m0 = fx.Float32(load(65536 + (sw0 * 16 + l16) * 4, fx.Float32, 1)[0])
                m1 = fx.Float32(load(65536 + (sw1 * 16 + l16) * 4, fx.Float32, 1)[0])
                lm = fx.maxnumf(m0, m1)
                sm = (lm > fx.Float32(float("-inf"))).select(lm, fx.Float32(0.0))
                c0 = fx.Float32(fx.exp2(m0 - sm, fastmath="fast"))
                c1 = fx.Float32(fx.exp2(m1 - sm, fastmath="fast"))
                ls = (
                    fx.Float32(load(65792 + (sw0 * 16 + l16) * 4, fx.Float32, 1)[0])
                    * c0
                    + fx.Float32(load(65792 + (sw1 * 16 + l16) * 4, fx.Float32, 1)[0])
                    * c1
                )
                invl = fx.Float32(rcp_f32((ls > 0).select(ls, fx.Float32(1.0))))
                if row < query_rows:
                    partial = (kv_head * capacity + task_id) * query_rows + row
                    for d in range_constexpr(d_chunks):
                        h = d * 16 + rg * 4
                        o0 = load(((w0 * d_chunks + d) * 64 + lane) * 16, fx.Float32, 4)
                        o1 = load(((w1 * d_chunks + d) * 64 + lane) * 16, fx.Float32, 4)
                        out = (
                            (o0 * vec(c0) + o1 * vec(c1)) * vec(invl * value_scale)
                        ).to(fx.BFloat16)
                        fx.ptr_store(
                            out,
                            fx.recast_iter(fx.BFloat16, pout_ptr)
                            + partial * 256
                            + dhalf * (256 // output_splits)
                            + h,
                        )
                    if rg == 0 and dhalf == 0:
                        fx.ptr_store(
                            (lm - fx.Float32(8.0)) * fx.Float32(1.0 / LOG2E),
                            fx.recast_iter(fx.Float32, pmax_ptr) + partial,
                        )
                        fx.ptr_store(ls, fx.recast_iter(fx.Float32, psum_ptr) + partial)

    hints = {"fastmath": "contract"}
    if waves_per_eu is not None:
        hints["waves_per_eu"] = waves_per_eu

    @flyc.jit
    def launch(
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        tables: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        table_stride: fx.Int32,
        num_seqs: fx.Int32,
        kv_heads: fx.Int32,
        ks_block: fx.Int32,
        ks_head: fx.Int32,
        q_row: fx.Int32,
        q_head: fx.Int32,
        work: fx.Pointer,
        capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = (cache_tag, waves_per_eu, tile_tokens)
        with CompilationContext.compile_hints(hints):
            kernel(
                pmax,
                psum,
                pout,
                query,
                key,
                value,
                tables,
                ks,
                vs,
                table_stride,
                ks_block,
                ks_head,
                q_row,
                q_head,
                work,
            ).launch(
                grid=(capacity, kv_heads, 1), block=(block_threads, 1, 1), stream=stream
            )

    result = {"launch": launch, "kernel": kernel}
    _CACHE[key] = result
    return result

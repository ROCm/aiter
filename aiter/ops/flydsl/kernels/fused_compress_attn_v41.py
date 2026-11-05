# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL V4.1 CSA main compressor.

One wave owns one plan row: online-softmax pool, FP32 RMSNorm, GPT-J RoPE,
BF16 rounding, then the NVFP4 or MXFP8 packed row. NoPE pairs go through
the rotation too, with cos 1 / sin 0. NVFP4 uses groups of 16 with the E4M3
scale ``max(amax, 6 * 2^-9) / 6``. MXFP8 follows aiter's MX quant rule: the
``MxScaleRoundMode.RoundUp`` scale over a NaN-skipping amax, capped at 0xFE
for a group holding an Inf, with NaN and Inf converting to FP8 NaN. This
does not change ``flydsl_fused_compress_attn`` or the V4 ``fp4`` /
``group_fp8`` modes.

gfx1250 uses waves of 32 (16 values each), gfx950 waves of 64 (8 values
each); lane L holds 4-element chunks at 4L, 4L + 4 * wave, ... so every
memory instruction covers a contiguous span. For few rows a row is split
over waves, one chunk each (4 on gfx1250, 2 on gfx950), and the sum of
squares is combined through LDS in a fixed warp order. A negative position
or batch reads no state and writes nothing. Compressor state is left for the
caller to update.
"""

import struct
from functools import cache, lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, fastmath, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, Stream, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from aiter.ops.flydsl.kernels.kernels_common import LOG2E as _LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, _to_raw

_DIM = 512
_ROPE = 64
_FP8_MAX = 448.0
_AMAX_FLOOR = 1e-10
_NVFP4_FLOOR = 6.0 * 2.0**-9
_ROW_BYTES = {"nvfp4": 288, "mxfp8": 528}
_NEG = float("-inf")
_DPP_ROW_SHR = 0x110
_DPP_ROW_BCAST15 = 0x142
# quad_perm [1,0,3,2], quad_perm [2,3,0,1], row_half_mirror: lane swaps
# covering groups of 2, 4 and 8 lanes.
_DPP_GROUP_SWAP = (0xB1, 0x4E, 0x141)


@cache
def _arch():
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx()


def _f32_bits(value):
    return struct.unpack("<i", struct.pack("<f", float(value)))[0]


def _same_row(name, tensor):
    if (
        tensor.ndim != 2
        or tensor.shape[1] != _DIM
        or tensor.dtype not in (torch.float32, torch.bfloat16)
        or tensor.stride(1) != 1
    ):
        raise ValueError(
            f"{name} must be [T, 512] fp32 or bf16 with unit last stride, got "
            f"{tuple(tensor.shape)} {tensor.dtype} stride {tensor.stride()}"
        )


@lru_cache(maxsize=16)
def _build(
    nvfp4: bool,
    wave: int,
    ratio: int,
    kv_bf16: bool,
    cos_bf16: bool,
    latent: bool,
    rotated: bool,
    rows_per_block: int,
    waves_per_row: int,
):
    # Lane L owns elements 4L + g * stride for row chunks g, so each load or
    # store instruction covers one contiguous span of the row across the wave.
    # With waves_per_row > 1, wave w of the row holds chunks w * chunks + c.
    stride = 4 * wave
    row_chunks = _DIM // stride
    chunks = row_chunks // waves_per_row
    elems = chunks * 4
    group = 16 if nvfp4 else 32
    group_lanes = group // 4
    group_shift = group_lanes.bit_length() - 1
    scale_base = 256 if nvfp4 else _DIM
    # RoPE pairs sit in the last row chunk, from this lane on.
    rope_lane = (_DIM - _ROPE - (row_chunks - 1) * stride) // 4
    log2_wave = wave.bit_length() - 1
    kv_bytes = 2 if kv_bf16 else 4
    split = waves_per_row > 1
    assert row_chunks in (2, 4) and ratio in (1, 2) and 0 <= rope_lane < wave
    assert row_chunks % waves_per_row == 0
    # The split row combines its sum of squares through LDS behind a block
    # barrier, so a block holds exactly one row.
    assert not split or rows_per_block == 1

    name = (
        f"fused_compress_v41_{'nvfp4' if nvfp4 else 'mxfp8'}"
        f"_w{wave}_r{ratio}_{'bf16' if kv_bf16 else 'f32'}"
        f"_{'cbf' if cos_bf16 else 'cf32'}"
        f"{'_lat' if latent else ''}{'_rot' if rotated else ''}"
        f"_b{rows_per_block}_s{waves_per_row}"
    )
    block_threads = wave * rows_per_block * waves_per_row

    @fx.struct
    class SharedStorage:
        parts: fx.Array[fx.Float32, row_chunks * 2, 16]

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def kernel(
        kv_in: fx.Pointer,
        score_in: fx.Pointer,
        kv_state: fx.Pointer,
        score_state: fx.Pointer,
        plan: fx.Pointer,
        slots: fx.Pointer,
        ape: fx.Pointer,
        rms_weight: fx.Pointer,
        cos_cache: fx.Pointer,
        sin_cache: fx.Pointer,
        latent_out: fx.Pointer,
        rotated_out: fx.Pointer,
        cache: fx.Pointer,
        block_tables: fx.Pointer,
        kv_stride: Int32,
        score_stride: Int32,
        kv_slot_stride: Int32,
        kv_pos_stride: Int32,
        score_slot_stride: Int32,
        score_pos_stride: Int32,
        plan_stride: Int32,
        cos_stride: Int32,
        latent_stride: Int32,
        rotated_stride: Int32,
        bt_stride: Int32,
        page_stride: Int64,
        row_stride: Int64,
        rows_per_page: Int32,
        state_size: Int32,
        eps_bits: Int32,
        nrows: Int32,
    ):
        def uniform(v):
            return fx.Int32(fx.rocdl.readfirstlane(T.i32, _to_raw(v)))

        # One wave per plan row; a block holds rows_per_block independent
        # waves, or one row split over waves_per_row waves.
        row_wave = 0
        if const_expr(split):
            thread = fx.Int32(fx.thread_idx.x)
            tid = thread & (wave - 1)
            row_wave = uniform(thread >> log2_wave)
            pid = fx.block_idx.x
        elif const_expr(rows_per_block == 1):
            tid = fx.thread_idx.x
            pid = fx.block_idx.x
        else:
            thread = fx.Int32(fx.thread_idx.x)
            tid = thread & (wave - 1)
            pid = fx.Int32(fx.block_idx.x) * rows_per_block + uniform(
                thread >> log2_wave
            )
        c_log2e = fx.Float32(_LOG2E)
        eps = eps_bits.bitcast(fx.Float32)
        neg = fx.Float32(_NEG)
        zero = fx.Float32(0.0)
        one = fx.Float32(1.0)

        def addr(p):
            return fx.Int64(fx.ptrtoint(p))

        def rsrc(a):
            return buffer_ops.create_buffer_resource_from_addr(a)

        def load_i32(resource, index):
            return fx.Int32(
                buffer_ops.buffer_load(resource, index, vec_width=1, dtype=T.i32)
            )

        lane_off = tid * 4

        def row_chunk(c):
            return row_wave * chunks + c if const_expr(split) else c

        def chunk_off(c):
            return lane_off + row_chunk(c) * stride

        def load_row(resource, dtype):
            vals = []
            for c in range_constexpr(chunks):
                raw = fx.Vector(
                    buffer_ops.buffer_load(
                        resource, chunk_off(c), vec_width=4, dtype=dtype
                    )
                )
                if const_expr(dtype != T.f32):
                    raw = raw.to(fx.Float32)
                for t in range_constexpr(4):
                    vals.append(raw[t])
            return vals

        def load_f32(resource):
            return load_row(resource, T.f32)

        def load_kv(resource):
            return load_row(resource, T.bf16 if kv_bf16 else T.f32)

        def call_f32(intrin, *args):
            return fx.Float32(
                _llvm.call_intrinsic(T.f32, intrin, [_to_raw(a) for a in args], [], [])
            )

        # Each op below pins one form so the bits do not depend on the
        # backend's choices: llvm.exp2 (with its denormal range
        # fixup, unlike raw v_exp), v_rsq, and explicit FMAs where the
        # backend would otherwise pick its own contraction.
        def fexp(x):
            return call_f32("llvm.exp2.f32", x * c_log2e)

        def ffma(a, b, c):
            return call_f32("llvm.fma.f32", a, b, c)

        def dpp(v, ctrl, row_mask=0xF, bound=True):
            raw = v.bitcast(fx.Int32)
            out = update_dpp_i32(raw, raw, ctrl, row_mask, 0xF, bound)
            return fx.Int32(out).bitcast(fx.Float32)

        def softmax_step(k, m_lane, kv_lane, w_lane, score_lane, val_lane, is_state):
            """One online-softmax step; is_state is wave-uniform, or None for
            a step known to read the input.

            State steps fuse kv_acc * scale and drop -inf scores; input steps
            fuse val * w. At k == 0 the accumulators are (-inf, 0, 0), so
            scale is 0 and both forms reduce to fma(val, w, 0) and w.
            """
            new_m, new_kv, new_w = [], [], []
            for i in range_constexpr(elems):
                score = fx.Float32(score_lane[i])
                val = fx.Float32(val_lane[i])
                m_old = neg if const_expr(k == 0) else fx.Float32(m_lane[i])
                m_new = fx.max(m_old, score)
                wk = fexp(score - m_new)
                if const_expr(is_state is not None):
                    with fastmath(None):
                        killed = is_state & (score == neg)
                    wk = killed.select(zero, wk)
                new_m.append(m_new)
                if const_expr(k == 0):
                    new_kv.append(ffma(val, wk, zero))
                    new_w.append(wk)
                    continue
                with fastmath(None):
                    is_first = m_old == neg
                scale = is_first.select(zero, fexp(m_old - m_new))
                kv_old = fx.Float32(kv_lane[i])
                kv_input_form = ffma(val, wk, kv_old * scale)
                if const_expr(is_state is not None):
                    kv_state_form = ffma(kv_old, scale, wk * val)
                    new_kv.append(is_state.select(kv_state_form, kv_input_form))
                else:
                    new_kv.append(kv_input_form)
                new_w.append(ffma(fx.Float32(w_lane[i]), scale, wk))
            return new_m, new_kv, new_w

        def round_bf16(values):
            """BF16 round-trip in registers. The returned vectors are the stored form."""
            vecs, rounded = [], []
            for c in range_constexpr(chunks):
                vec = fx.Vector.from_elements(
                    [values[c * 4 + t] for t in range(4)], dtype=fx.Float32
                ).to(fx.BFloat16)
                vecs.append(vec)
                back = vec.to(fx.Float32)
                for t in range_constexpr(4):
                    rounded.append(back[t])
            return vecs, rounded

        def store_vecs(resource, vecs):
            for c in range_constexpr(chunks):
                buffer_ops.buffer_store(
                    vecs[c].bitcast(fx.Int32).ir_value(),
                    resource,
                    chunk_off(c) * 2,
                    offset_is_bytes=True,
                )

        def e4m3_to_f32(byte):
            exp = (byte >> 3) & 0xF
            mant = byte & 0x7
            normal = (((exp + 120) << 23) | (mant << 20)).bitcast(fx.Float32)
            sub = mant.to(fx.Float32) * fx.Float32(2.0**-9)
            return (exp == 0).select(sub, normal)

        def e2m1_thresholds(scale):
            """g[i] such that code > i  <=>  |x| >= g[i].

            |x / s| > t  <=>  |x| > s * t here: s is E4M3 and t has at most 3
            significant bits, so s * t is exact, and a BF16 |x| != s * t sits
            at least 2^-8 relative away, beyond the division's rounding. A
            strict > becomes >= against the next float up (bits + 1).
            """
            g = []
            for t, strict in (
                (0.25, 1),
                (0.75, 0),
                (1.25, 1),
                (1.75, 0),
                (2.5, 1),
                (3.5, 0),
                (5.0, 1),
            ):
                st = scale * fx.Float32(t)
                g.append(
                    (st.bitcast(fx.Int32) + strict).bitcast(fx.Float32)
                    if strict
                    else st
                )
            return g

        def e2m1(val, g):
            # Binary search over the 7 monotone thresholds; a NaN fails every
            # compare and gives magnitude code 0, as the threshold chain did.
            mag = fmath.absf(val)
            b2 = mag >= g[3]
            b1 = mag >= b2.select(g[5], g[1])
            b0 = mag >= b2.select(b1.select(g[6], g[4]), b1.select(g[2], g[0]))
            code = (
                b2.select(fx.Int32(4), fx.Int32(0))
                | b1.select(fx.Int32(2), fx.Int32(0))
                | b0.select(fx.Int32(1), fx.Int32(0))
            )
            sign = (val.bitcast(fx.Int32) >> 28) & fx.Int32(8)
            return code | sign

        # The last block's spare waves read an empty descriptor and write nothing.
        in_range = fx.Int32(pid) < nrows
        plan_rsrc = buffer_ops.create_buffer_resource_from_addr(
            addr(plan) + fx.Int64(pid) * fx.Int64(plan_stride) * 4,
            num_records_bytes=in_range.select(fx.Int32(16), fx.Int32(0)),
        )
        ragged = uniform(load_i32(plan_rsrc, 0))
        batch = uniform(load_i32(plan_rsrc, 1))
        position = uniform(load_i32(plan_rsrc, 2))
        window_len = uniform(load_i32(plan_rsrc, 3))

        def gated(base, live, nbytes):
            """Descriptor that reads nothing (OOB zeros, no traffic) unless live."""
            return buffer_ops.create_buffer_resource_from_addr(
                base, num_records_bytes=live.select(fx.Int32(nbytes), fx.Int32(0))
            )

        def input_rows(k):
            in_row = ragged - fx.Int32(ratio - 1 - k)
            kv_row = addr(kv_in) + fx.Int64(in_row) * fx.Int64(kv_stride) * kv_bytes
            sc_row = (
                addr(score_in) + fx.Int64(in_row) * fx.Int64(score_stride) * kv_bytes
            )
            return kv_row, sc_row

        def pool_inputs():
            """All k read the input rows (window_len == 0)."""
            steps = []
            for k in range_constexpr(ratio):
                kv_row, sc_row = input_rows(k)
                steps.append(
                    (
                        load_kv(rsrc(kv_row)),
                        load_kv(rsrc(sc_row)),
                        load_f32(rsrc(addr(ape) + k * _DIM * 4)),
                    )
                )
            m_lane = kv_lane = w_lane = None
            for k in range_constexpr(ratio):
                kv_a, sc_a, ape_v = steps[k]
                scores = [sc_a[i] + ape_v[i] for i in range(elems)]
                m_lane, kv_lane, w_lane = softmax_step(
                    k, m_lane, kv_lane, w_lane, scores, kv_a, None
                )
            return [
                fx.Float32(kv_lane[i]) / fx.Float32(w_lane[i]) for i in range(elems)
            ]

        def pool_mixed():
            # Every k is unrolled and both of its candidate sources (state ring
            # or input row) are issued up front; the unused one gets an empty
            # descriptor. window_len is wave-uniform, so is_state is too.
            slot = uniform(load_i32(rsrc(addr(slots)), batch))
            steps = []
            for k in range_constexpr(ratio):
                is_state = fx.Int32(k) < window_len
                s = position - fx.Int32(ratio - 1 - k)
                is_pad = s < fx.Int32(0)
                ring = fx.max(s, fx.Int32(0)) % state_size
                read_state = is_state & (s >= fx.Int32(0))
                kv_row = (
                    addr(kv_state)
                    + fx.Int64(slot) * fx.Int64(kv_slot_stride) * 4
                    + fx.Int64(ring) * fx.Int64(kv_pos_stride) * 4
                )
                sc_row = (
                    addr(score_state)
                    + fx.Int64(slot) * fx.Int64(score_slot_stride) * 4
                    + fx.Int64(ring) * fx.Int64(score_pos_stride) * 4
                )
                read_in = fx.Int32(k) >= window_len
                kv_in_row, sc_in_row = input_rows(k)
                steps.append(
                    (
                        is_state,
                        is_pad,
                        load_f32(gated(kv_row, read_state, _DIM * 4)),
                        load_f32(gated(sc_row, read_state, _DIM * 4)),
                        load_kv(gated(kv_in_row, read_in, _DIM * kv_bytes)),
                        load_kv(gated(sc_in_row, read_in, _DIM * kv_bytes)),
                        load_f32(gated(addr(ape) + k * _DIM * 4, read_in, _DIM * 4)),
                    )
                )
            m_lane = kv_lane = w_lane = None
            for k in range_constexpr(ratio):
                is_state, is_pad, kv_b, sc_b, kv_a, sc_a, ape_v = steps[k]
                vals, scores = [], []
                for i in range_constexpr(elems):
                    vals.append(is_state.select(kv_b[i], kv_a[i]))
                    state_score = is_pad.select(neg, sc_b[i])
                    scores.append(is_state.select(state_score, sc_a[i] + ape_v[i]))
                m_lane, kv_lane, w_lane = softmax_step(
                    k, m_lane, kv_lane, w_lane, scores, vals, is_state
                )
            return [
                fx.Float32(kv_lane[i]) / fx.Float32(w_lane[i]) for i in range(elems)
            ]

        def body():
            # Prefill rows mostly have window_len == 0; they skip the state
            # descriptors and the per-k source selects.
            comp = pool_inputs() if window_len == fx.Int32(0) else pool_mixed()
            weight = load_f32(rsrc(addr(rms_weight)))
            if const_expr(ratio == 2):
                comp_pos = (position >> 1) << 1
                entry = position >> 1
            else:
                comp_pos = position
                entry = position
            # Two RoPE pairs per lane in the last chunk; NoPE pairs rotate by
            # cos 1 / sin 0.
            is_rope = tid >= fx.Int32(rope_lane)
            if const_expr(split):
                is_rope = is_rope & (row_wave == fx.Int32(waves_per_row - 1))
            cs_off = (
                comp_pos * cos_stride
                + fx.max(tid - fx.Int32(rope_lane), fx.Int32(0)) * 2
            )
            cs_dtype = T.bf16 if cos_bf16 else T.f32
            cos_raw = fx.Vector(
                buffer_ops.buffer_load(
                    rsrc(addr(cos_cache)), cs_off, vec_width=2, dtype=cs_dtype
                )
            )
            sin_raw = fx.Vector(
                buffer_ops.buffer_load(
                    rsrc(addr(sin_cache)), cs_off, vec_width=2, dtype=cs_dtype
                )
            )
            if const_expr(cos_bf16):
                cos_raw = cos_raw.to(fx.Float32)
                sin_raw = sin_raw.to(fx.Float32)
            in_page = entry % rows_per_page
            # Uniform page keeps the cache descriptor in SGPRs (no waterfall loop).
            page = uniform(
                load_i32(
                    rsrc(
                        addr(block_tables) + fx.Int64(batch) * fx.Int64(bt_stride) * 4
                    ),
                    entry // rows_per_page,
                )
            )

            if const_expr(wave == 32):
                # The sum of squares runs in a layout of 4 warps x 32 threads
                # with 4 consecutive values each, so chunk c of this lane is
                # thread tid + 32c of warp c. Per thread
                # fma(x0, x0, x2*x2) + fma(x1, x1, x3*x3); DPP row_shr
                # 8/4/2/1 inside 16 threads, permlanex16 across the two rows
                # of a warp, then (w0 + w2) + (w1 + w3).
                parts = []
                for c in range_constexpr(chunks):
                    x0, x1, x2, x3 = comp[4 * c : 4 * c + 4]
                    p = ffma(x0, x0, x2 * x2) + ffma(x1, x1, x3 * x3)
                    for m in (8, 4, 2, 1, 16):
                        p = p + p.shuffle_xor(m, wave)
                    parts.append(p)
                if const_expr(split):
                    shared = fx.SharedAllocator().allocate(SharedStorage).peek().parts
                    for c in range_constexpr(chunks):
                        shared[row_chunk(c)] = parts[c]
                    fx.gpu.barrier()
                    parts = [fx.Float32(shared[g]) for g in range(row_chunks)]
                total = (parts[0] + parts[2]) + (parts[1] + parts[3])
            else:
                # The sum of squares runs in a layout of 4 warps x 64 threads
                # with 2 values each, so chunk c of lane L is threads 2L + 128c
                # and 2L + 1 + 128c, in warp L / 32 + 2c. Per thread
                # x0*x0 + x1*x1 (no FMA); DPP row_shr 8/4/2/1 (thread bits
                # 3..0, i.e. lane bits 2..0 then the pair), row_bcast 15/31
                # (lane bits 3, 4), then (w0 + w2) + (w1 + w3).
                # The partial sums only stay exact on the lane each shift
                # lands on: lane 8k + 7 per group, then lanes 31 / 63, read
                # out as the even / odd warp of the chunk.
                warps = []
                for c in range_constexpr(chunks):
                    x0, x1, x2, x3 = comp[4 * c : 4 * c + 4]
                    with fastmath(None):
                        a = x0 * x0 + x1 * x1
                        b = x2 * x2 + x3 * x3
                    for n in (4, 2, 1):
                        a = a + dpp(a, _DPP_ROW_SHR + n)
                        b = b + dpp(b, _DPP_ROW_SHR + n)
                    s = a + b
                    s = s + dpp(s, _DPP_ROW_SHR + 8)
                    s = s + dpp(s, _DPP_ROW_BCAST15, row_mask=0xA)
                    warps.append(call_f32("llvm.amdgcn.readlane.f32", s, fx.Int32(31)))
                    warps.append(call_f32("llvm.amdgcn.readlane.f32", s, fx.Int32(63)))
                if const_expr(split):
                    shared = fx.SharedAllocator().allocate(SharedStorage).peek().parts
                    shared[row_chunk(0) * 2] = warps[0]
                    shared[row_chunk(0) * 2 + 1] = warps[1]
                    fx.gpu.barrier()
                    warps = [fx.Float32(shared[w]) for w in range(row_chunks * 2)]
                total = (warps[0] + warps[2]) + (warps[1] + warps[3])
            var = total * fx.Float32(1.0 / _DIM)
            rrms = call_f32("llvm.amdgcn.rsq.f32", var + eps)
            normed = [comp[i] * rrms * weight[i] for i in range(elems)]
            if const_expr(latent):
                lat_vecs, _ = round_bf16(normed)
                store_vecs(
                    rsrc(
                        addr(latent_out) + fx.Int64(pid) * fx.Int64(latent_stride) * 2
                    ),
                    lat_vecs,
                )

            rot = []
            for p in range_constexpr(elems // 2):
                e = normed[2 * p]
                o = normed[2 * p + 1]
                if const_expr(p >= (chunks - 1) * 2):
                    cs = is_rope.select(cos_raw[p % 2], one)
                    sn = is_rope.select(sin_raw[p % 2], zero)
                else:
                    cs, sn = one, zero
                rot.append(ffma(e, cs, -(o * sn)))
                rot.append(ffma(o, cs, e * sn))
            # Quant reads the BF16 rounding, whether or not the echo is stored.
            rot_vecs, quant = round_bf16(rot)
            if const_expr(rotated):
                store_vecs(
                    rsrc(
                        addr(rotated_out) + fx.Int64(pid) * fx.Int64(rotated_stride) * 2
                    ),
                    rot_vecs,
                )

            # A quant group is group_lanes adjacent lanes of one chunk.
            # The MXFP8 amax skips NaN, as aiter's MX quant kernels do.
            amax = (lambda a, b: a.maximumf(b)) if nvfp4 else fx.maxnumf
            words, scale_bytes = [], []
            for c in range_constexpr(chunks):
                q4 = quant[4 * c : 4 * c + 4]
                am = fmath.absf(q4[0])
                for t in range_constexpr(1, 4):
                    am = amax(am, fmath.absf(q4[t]))
                for m in range_constexpr(group_shift):
                    if const_expr(wave == 64):
                        # gfx9 lowers wave64 shuffles to ds_swizzle; max is
                        # order-free, so lane swaps within the group suffice.
                        am = amax(am, dpp(am, _DPP_GROUP_SWAP[m], bound=False))
                    else:
                        am = amax(am, am.shuffle_xor(1 << m, wave))
                if const_expr(nvfp4):
                    raw = am.maximumf(fx.Float32(_NVFP4_FLOOR)) * fx.Float32(1.0 / 6.0)
                    raw = raw.minimumf(fx.Float32(_FP8_MAX))
                    word = fx.Int32(
                        fx.rocdl.cvt_pk_fp8_f32(T.i32, raw, zero, fx.Int32(0), False)
                    )
                    scale_bits = word & 0xFF
                    thr = e2m1_thresholds(e4m3_to_f32(scale_bits))
                    scale_bytes.append(scale_bits.to(fx.Int8))
                    codes = [e2m1(q4[t], thr) for t in range(4)]
                    half = (
                        codes[0] | (codes[1] << 4) | (codes[2] << 8) | (codes[3] << 12)
                    )
                    words.append(half.to(fx.Int16))
                else:
                    am = fx.maxnumf(am, fx.Float32(_AMAX_FLOOR))
                    # MxScaleRoundMode.RoundUp, ceil_pow2(amax / 448); an Inf
                    # group gets the largest finite scale, 0xFE.
                    bits = (am * fx.Float32(1.0 / _FP8_MAX)).bitcast(fx.Int32)
                    exponent = (bits >> 23) & 0xFF
                    bump = (exponent < 0xFF) & ((bits & 0x7FFFFF) != 0)
                    code = exponent + bump.select(fx.Int32(1), fx.Int32(0))
                    code = (code > 254).select(fx.Int32(254), code)
                    scale_bytes.append(code.to(fx.Int8))
                    # ldexp, not a multiply: 2^-127 is an fp32 denormal. No
                    # clamp: the scale keeps finite values within +-448, and
                    # NaN / Inf convert to FP8 NaN.
                    shift = fx.Int32(127) - code
                    qs = [
                        call_f32("llvm.ldexp.f32.i32", q4[t], shift) for t in range(4)
                    ]
                    packed = fx.rocdl.cvt_pk_fp8_f32(
                        T.i32, qs[0], qs[1], fx.Int32(0), False
                    )
                    packed = fx.rocdl.cvt_pk_fp8_f32(T.i32, qs[2], qs[3], packed, True)
                    words.append(fx.Int32(packed))

            row = (
                addr(cache)
                + fx.Int64(page) * page_stride
                + fx.Int64(in_page) * row_stride
            )
            out = rsrc(row)
            value_bytes = 2 if nvfp4 else 4
            for c in range_constexpr(chunks):
                buffer_ops.buffer_store(
                    words[c], out, chunk_off(c) // 4 * value_bytes, offset_is_bytes=True
                )
            if (tid & (group_lanes - 1)) == 0:
                for c in range_constexpr(chunks):
                    buffer_ops.buffer_store(
                        scale_bytes[c],
                        out,
                        scale_base
                        + (tid >> group_shift)
                        + row_chunk(c) * (stride // group),
                        offset_is_bytes=True,
                    )

        if in_range & (position >= 0) & (batch >= 0):
            body()

    @flyc.jit
    def launch(
        kv_in: fx.Pointer,
        score_in: fx.Pointer,
        kv_state: fx.Pointer,
        score_state: fx.Pointer,
        plan: fx.Pointer,
        slots: fx.Pointer,
        ape: fx.Pointer,
        rms_weight: fx.Pointer,
        cos_cache: fx.Pointer,
        sin_cache: fx.Pointer,
        latent_out: fx.Pointer,
        rotated_out: fx.Pointer,
        cache: fx.Pointer,
        block_tables: fx.Pointer,
        kv_stride: Int32,
        score_stride: Int32,
        kv_slot_stride: Int32,
        kv_pos_stride: Int32,
        score_slot_stride: Int32,
        score_pos_stride: Int32,
        plan_stride: Int32,
        cos_stride: Int32,
        latent_stride: Int32,
        rotated_stride: Int32,
        bt_stride: Int32,
        page_stride: Int64,
        row_stride: Int64,
        rows_per_page: Int32,
        state_size: Int32,
        eps_bits: Int32,
        nrows: Int32,
        nblocks: Int32,
        stream: fx.Stream,
    ):
        kernel(
            kv_in,
            score_in,
            kv_state,
            score_state,
            plan,
            slots,
            ape,
            rms_weight,
            cos_cache,
            sin_cache,
            latent_out,
            rotated_out,
            cache,
            block_tables,
            kv_stride,
            score_stride,
            kv_slot_stride,
            kv_pos_stride,
            score_slot_stride,
            score_pos_stride,
            plan_stride,
            cos_stride,
            latent_stride,
            rotated_stride,
            bt_stride,
            page_stride,
            row_stride,
            rows_per_page,
            state_size,
            eps_bits,
            nrows,
        ).launch(
            grid=(fx.Int64(nblocks), 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    return launch


def fused_compress_attn_v41(
    *,
    kv_in: torch.Tensor,
    score_in: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    plan: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    ratio: int,
    cache_format: str,
    latent_out: torch.Tensor | None = None,
    rotated_out: torch.Tensor | None = None,
) -> None:
    """V4.1 CSA main compressor: pool, RMSNorm, RoPE and the packed cache row."""
    row_bytes = _ROW_BYTES.get(cache_format)
    if row_bytes is None:
        raise ValueError(f"cache_format must be nvfp4 or mxfp8, got {cache_format}")
    if ratio not in (1, 2):
        raise ValueError(f"ratio must be 1 or 2, got {ratio}")
    _same_row("kv_in", kv_in)
    _same_row("score_in", score_in)
    if score_in.shape != kv_in.shape or score_in.dtype != kv_in.dtype:
        raise ValueError("score_in must match kv_in")
    align = 8 if kv_in.dtype == torch.bfloat16 else 4
    if kv_in.stride(0) % align or score_in.stride(0) % align:
        raise ValueError(
            f"kv_in and score_in row strides must be multiples of {align}, got "
            f"{kv_in.stride(0)} and {score_in.stride(0)}"
        )
    if (
        kv_state.ndim != 3
        or kv_state.shape[2] != _DIM
        or kv_state.dtype != torch.float32
        or kv_state.stride(2) != 1
        or kv_state.shape[1] < ratio
        or score_state.shape != kv_state.shape
        or score_state.dtype != kv_state.dtype
        or score_state.stride(2) != 1
    ):
        raise ValueError(
            "kv_state and score_state must be matching fp32 [slots, ring, 512] tensors"
        )
    if kv_state.stride(1) % 4 or score_state.stride(1) % 4:
        raise ValueError("state ring strides must be 4-element aligned")
    if (
        plan.ndim != 2
        or plan.shape[1] != 4
        or plan.dtype != torch.int32
        or plan.stride(1) != 1
    ):
        raise ValueError("plan must be [capacity, 4] int32 with unit last stride")
    if state_slot_mapping.ndim != 1 or state_slot_mapping.dtype != torch.int32:
        raise ValueError("state_slot_mapping must be [batch] int32")
    if (
        ape.shape != (ratio, _DIM)
        or ape.dtype != torch.float32
        or not ape.is_contiguous()
    ):
        raise ValueError(f"ape must be contiguous fp32 [{ratio}, 512]")
    if (
        rms_weight.shape != (_DIM,)
        or rms_weight.dtype != torch.float32
        or not rms_weight.is_contiguous()
    ):
        raise ValueError("rms_weight must be contiguous fp32 [512]")
    if (
        cos_cache.dtype not in (torch.bfloat16, torch.float32)
        or sin_cache.dtype != cos_cache.dtype
    ):
        raise ValueError("cos_cache and sin_cache must both be bf16 or fp32")
    if (
        cos_cache.shape[-1] != _ROPE // 2
        or cos_cache.stride(-1) != 1
        or sin_cache.shape != cos_cache.shape
        or sin_cache.stride(0) != cos_cache.stride(0)
    ):
        raise ValueError(
            "cos_cache and sin_cache must end in 32 contiguous elements and share a position stride"
        )
    if (
        block_tables.ndim != 2
        or block_tables.dtype != torch.int32
        or block_tables.stride(1) != 1
        or block_tables.shape[0] != state_slot_mapping.shape[0]
    ):
        raise ValueError(
            "block_tables must be int32 [batch, pages] with unit last stride"
        )
    if (
        cache.ndim != 3
        or cache.dtype != torch.uint8
        or cache.shape[2] != row_bytes
        or cache.stride(2) != 1
        or cache.shape[1] == 0
        or cache.stride(0) % 4
        or cache.stride(1) % 4
    ):
        raise ValueError(
            f"{cache_format} cache must be 4-byte-aligned uint8 [pages, rows, {row_bytes}]"
        )
    rows = plan.shape[0]
    for name, tensor in (("latent_out", latent_out), ("rotated_out", rotated_out)):
        if tensor is None:
            continue
        if (
            tensor.shape != (rows, _DIM)
            or tensor.dtype != torch.bfloat16
            or tensor.stride(1) != 1
            or tensor.stride(0) % 8
        ):
            raise ValueError(
                f"{name} must be [{rows}, 512] bf16 with an 8-element row stride"
            )
    if rows == 0:
        return
    arch = _arch()
    if arch not in ("gfx950", "gfx1250"):
        raise RuntimeError(
            f"fused_compress_attn_v41 supports gfx950 and gfx1250, got {arch}"
        )
    wave = 32 if arch == "gfx1250" else 64
    # Few rows leave most CUs idle, so latency follows one row's dependency
    # chain; splitting a row over waves (one chunk each) shortens it.
    if wave == 32:
        wpr = 4 if rows <= (512 if ratio == 1 else 1024) else 1
        rpb = 1 if ratio == 1 or wpr > 1 else 2
    else:
        wpr = 2 if rows <= (1024 if ratio == 1 else 4096) else 1
        rpb = 1 if ratio == 1 or wpr > 1 else 4
    launch = _build(
        cache_format == "nvfp4",
        wave,
        ratio,
        kv_in.dtype == torch.bfloat16,
        cos_cache.dtype == torch.bfloat16,
        latent_out is not None,
        rotated_out is not None,
        rpb,
        wpr,
    )
    placeholder = cos_cache

    def ptr(t):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    _run_compiled(
        launch,
        ptr(kv_in),
        ptr(score_in),
        ptr(kv_state),
        ptr(score_state),
        ptr(plan),
        ptr(state_slot_mapping),
        ptr(ape),
        ptr(rms_weight),
        ptr(cos_cache),
        ptr(sin_cache),
        ptr(latent_out if latent_out is not None else placeholder),
        ptr(rotated_out if rotated_out is not None else placeholder),
        ptr(cache),
        ptr(block_tables),
        int(kv_in.stride(0)),
        int(score_in.stride(0)),
        int(kv_state.stride(0)),
        int(kv_state.stride(1)),
        int(score_state.stride(0)),
        int(score_state.stride(1)),
        int(plan.stride(0)),
        int(cos_cache.stride(0)),
        int(latent_out.stride(0) if latent_out is not None else 0),
        int(rotated_out.stride(0) if rotated_out is not None else 0),
        int(block_tables.stride(0)),
        int(cache.stride(0)),
        int(cache.stride(1)),
        int(cache.shape[1]),
        int(kv_state.shape[1]),
        _f32_bits(rms_eps),
        int(rows),
        -(-int(rows) // rpb),
        Stream(torch.cuda.current_stream()),
    )

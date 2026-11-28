# SPDX-License-Identifier: MIT
"""Vectorized Muse per-expert RMSNorm followed by routing-weight reduction."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import LOG2E


def build_splitk_silu(I, split_k):
    """Reduce FP32 gate/up K-split partials and materialize BF16 SiLU*up."""
    threads = 256
    vec = 4
    block_n = threads * vec
    nblocks = (I + block_n - 1) // block_n

    @flyc.kernel(name="muse_moe_splitk_silu_flydsl")
    def kernel(partials: fx.Tensor, out: fx.Tensor, rows: fx.Int32):
        pid = fx.block_idx.x
        tid = fx.thread_idx.x
        row = pid // nblocks
        col = (pid % nblocks) * block_n + tid * vec
        pr = buffer_ops.create_buffer_resource(
            partials, num_records_bytes=split_k * rows * 2 * I * 4
        )
        outr = buffer_ops.create_buffer_resource(out, num_records_bytes=rows * I * 2)
        one = fx.Float32(1.0)
        gate = [fx.Float32(0.0) for _ in range(vec)]
        up = [fx.Float32(0.0) for _ in range(vec)]
        for sk in range_constexpr(split_k):
            base = sk * rows * (2 * I) + row * (2 * I) + col
            gate_raw = buffer_ops.buffer_load(
                pr, base, vec_width=vec, dtype=T.f32, mask=col < I
            )
            up_raw = buffer_ops.buffer_load(
                pr, base + I, vec_width=vec, dtype=T.f32, mask=col < I
            )
            gate_vec = fx.Vector(gate_raw, shape=vec, dtype=fx.Float32)
            up_vec = fx.Vector(up_raw, shape=vec, dtype=fx.Float32)
            gate = [gate[j] + gate_vec[j] for j in range(vec)]
            up = [up[j] + up_vec[j] for j in range(vec)]
        values = []
        for j in range_constexpr(vec):
            # SiLU with an exact divide, as in the GEMM epilogue (not act.py's rcp).
            exp = fx.Float32(fx.rocdl.exp2(T.f32, (gate[j] * -LOG2E).ir_value()))
            values.append((gate[j] * (one / (one + exp)) * up[j]).to(fx.BFloat16))
        packed = fx.Vector.from_elements(values, fx.BFloat16)
        buffer_ops.buffer_store(packed, outr, row * I + col, mask=col < I)

    @flyc.jit
    def launch(
        partials: fx.Tensor,
        out: fx.Tensor,
        rows: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(partials, out, rows).launch(
            grid=(fx.Int64(grid_x), 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


def build_finalize(H, topk, has_norm, has_scale, weight_dtype="bf16", split_k=1):
    threads = 256
    vec = 4 if H % 4 == 0 else 1
    chunks = (H + threads * vec - 1) // (threads * vec)
    elems = chunks * vec
    weight_bytes = 4 if weight_dtype == "fp32" else 2

    @fx.struct
    class Shared:
        # Give every routing slot its own four-wave reduction cells. This
        # removes the second barrier that was only needed before reusing the
        # same cells for the next slot.
        sums: fx.Array[fx.Float32, topk * 4, 16]

    def wave_sum(value):
        for shift in (32, 16, 8, 4, 2, 1):
            value = value + value.shuffle_xor(shift, 64)
        return value

    @flyc.kernel(name="muse_moe_finalize_vector_flydsl")
    def kernel(
        y: fx.Tensor,
        gamma: fx.Tensor,
        scores: fx.Tensor,
        ids: fx.Tensor,
        out: fx.Tensor,
        local_start: fx.Int32,
        num_local: fx.Int32,
        eps: fx.Float32,
        gain: fx.Float32,
        rows: fx.Int32,
    ):
        m = fx.block_idx.x
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        zero = fx.Float32(0.0)
        one = fx.Float32(1.0)
        yr = buffer_ops.create_buffer_resource(
            y, num_records_bytes=split_k * rows * topk * H * (4 if split_k > 1 else 2)
        )
        wr = buffer_ops.create_buffer_resource(
            gamma, num_records_bytes=H * weight_bytes
        )
        sr = buffer_ops.create_buffer_resource(scores, max_size=True)
        ir = buffer_ops.create_buffer_resource(ids, max_size=True)
        outr = buffer_ops.create_buffer_resource(out, num_records_bytes=rows * H * 2)
        lds = fx.SharedAllocator().allocate(Shared).peek()
        scales = []
        if fx.const_expr(has_norm and has_scale):
            wtype = (
                T.f32
                if weight_dtype == "fp32"
                else (T.f16 if weight_dtype == "fp16" else T.bf16)
            )
            numeric = (
                fx.Float32
                if weight_dtype == "fp32"
                else (fx.Float16 if weight_dtype == "fp16" else fx.BFloat16)
            )
            for chunk in range_constexpr(chunks):
                h = tid * vec + chunk * threads * vec
                raw = buffer_ops.buffer_load(
                    wr, h, vec_width=vec, dtype=wtype, mask=h < H
                )
                values = (
                    fx.Vector(raw, shape=vec, dtype=numeric)
                    if vec > 1
                    else [numeric(raw)]
                )
                for j in range_constexpr(vec):
                    # (gamma + gain), rounded to the weight dtype as the model does.
                    v = values[j].to(fx.Float32) + gain
                    scales.append(v.to(numeric).to(fx.Float32))
        init = [zero.ir_value() for _ in range(elems)]
        for slot_i, state in range(0, topk, 1, init=init):
            slot = m * topk + fx.Int32(slot_i)
            e = fx.Int32(buffer_ops.buffer_load(ir, slot, vec_width=1, dtype=T.i32))
            updated = fx.Vector.from_elements(state, fx.Float32).ir_value()
            if (e >= local_start) & (e < local_start + num_local):
                sc = fx.Float32(
                    buffer_ops.buffer_load(sr, slot, vec_width=1, dtype=T.f32)
                )
                values = []
                ss = zero
                for chunk in range_constexpr(chunks):
                    h = tid * vec + chunk * threads * vec
                    if fx.const_expr(split_k > 1):
                        partial = [zero for _ in range(vec)]
                        for sk in range_constexpr(split_k):
                            raw = buffer_ops.buffer_load(
                                yr,
                                sk * rows * topk * H + slot * H + h,
                                vec_width=vec,
                                dtype=T.f32,
                                mask=h < H,
                            )
                            vector = fx.Vector(raw, shape=vec, dtype=fx.Float32)
                            partial = [partial[j] + vector[j] for j in range(vec)]
                        vector = partial
                    else:
                        raw = buffer_ops.buffer_load(
                            yr,
                            slot * H + h,
                            vec_width=vec,
                            dtype=T.bf16,
                            mask=h < H,
                        )
                        vector = (
                            fx.Vector(raw, shape=vec, dtype=fx.BFloat16)
                            if vec > 1
                            else [fx.BFloat16(raw)]
                        )
                    for j in range_constexpr(vec):
                        if fx.const_expr(split_k > 1):
                            value = vector[j].to(fx.BFloat16).to(fx.Float32)
                        else:
                            value = vector[j].to(fx.Float32)
                        if fx.const_expr(has_norm and has_scale):
                            value = value * scales[chunk * vec + j]
                            value = value.to(fx.BFloat16).to(fx.Float32)
                        values.append(value)
                        ss = ss + value * value
                if fx.const_expr(has_norm):
                    ss = wave_sum(ss)
                    slot_base = fx.Int32(slot_i) * 4
                    if lane == 0:
                        lds.sums[slot_base + wave] = ss
                    fx.gpu.barrier()
                    total = zero
                    for w in range_constexpr(4):
                        total = total + lds.sums[slot_base + w]
                    mean = total * (1.0 / H) + eps
                    rms = fx.Float32(fx.rocdl.sqrt(T.f32, mean.ir_value()))
                    sc = sc * (one / rms)
                updated = fx.Vector.from_elements(
                    [fx.Float32(state[j]) + sc * values[j] for j in range(elems)],
                    fx.Float32,
                ).ir_value()
            next_values = [
                fx.Vector(updated, shape=elems, dtype=fx.Float32)[j].ir_value()
                for j in range(elems)
            ]
            final = yield next_values
        for chunk in range_constexpr(chunks):
            h = tid * vec + chunk * threads * vec
            packed = [
                fx.Float32(final[chunk * vec + j]).to(fx.BFloat16) for j in range(vec)
            ]
            value = (
                fx.Vector.from_elements(packed, fx.BFloat16) if vec > 1 else packed[0]
            )
            buffer_ops.buffer_store(value, outr, m * H + h, mask=h < H)

    @flyc.jit
    def launch(
        y: fx.Tensor,
        gamma: fx.Tensor,
        scores: fx.Tensor,
        ids: fx.Tensor,
        out: fx.Tensor,
        local_start: fx.Int32,
        num_local: fx.Int32,
        eps: fx.Float32,
        gain: fx.Float32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            y, gamma, scores, ids, out, local_start, num_local, eps, gain, grid_x
        ).launch(grid=(fx.Int64(grid_x), 1, 1), block=(threads, 1, 1), stream=stream)

    return launch

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 MarloweAI Contributors
"""Native-quantized exact-M4 direct split producer/consumer specialization.

Supplied native nine-route rows are consumed without sorting. The producer
writes FP32 split partials; each consumer reduces them locally and quantizes
the FP32 SiLU intermediate with the existing native E8M0 helper. This changes
the G1 reduction tree, not its quantizer or the weighted BF16 atomic contract.
"""

from functools import cache


def tiny_m4_supported(*, rows, hidden, inter, experts, topk, gfx, contiguous=True):
    return (rows, hidden, inter, experts, topk, gfx, contiguous) == (
        4,
        6144,
        256,
        257,
        9,
        "gfx950",
        True,
    )


def split6_indices(route, column):
    """FP32 scratch indices, also used by CPU bounds/reduction fixtures."""
    if not 0 <= route < 36 or not 0 <= column < 512:
        raise ValueError("M4 split-partial coordinate outside its allocated domain")
    return tuple((split * 36 + route) * 512 + column for split in range(6))


def reduce_split6(values):
    """Reference ordering: pad to eight, then combine halves/quarters/pairs."""
    if len(values) != 6:
        raise ValueError("split6 requires six partials")
    values = list(values) + [0.0, 0.0]
    for stride in (4, 2, 1):
        values = [values[i] + values[i + stride] for i in range(stride)]
    return values[0]


@cache
def compile_tiny_m4(*, down_n=128, down_waves=4):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
    from flydsl.expr.typing import T, as_ir_value

    from .mxfp4_gemm_common import (
        _activation_mul_batch,
        _e8m0_from_amax,
        _fabs_f32,
        _inline_dpp_pair_amax,
        _inline_e8m0,
        _scale_mma_atoms,
        _umax_i32,
        global_typed_ptr,
        lds_typed_ptr,
        lds_vec_load,
    )

    if (down_n, down_waves) not in ((128, 4), (256, 4), (128, 2)):
        raise ValueError("unsupported static M4 consumer tile")

    def load4(arg, index):
        # Build device IR only while FlyDSL emits a kernel in its MLIR context.
        fragment = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        v4i = fx.Vector.make_type(4, fx.Int32)
        fragment.store(
            fx.ptr_load(global_typed_ptr(arg, T.i32, 16) + index, result_type=v4i)
        )
        return fragment

    def zero_frag(dtype):
        fragment = fx.make_rmem_tensor(fx.make_layout(4, 1), dtype)
        fragment.store(fx.Vector.filled(4, 0, dtype))
        return fragment

    @fx.struct
    class InputStorage:
        raw: fx.Array[fx.Uint8, 544, 16]

    @flyc.kernel(
        name="marlowe_native_m4_split6_g1_iq7fffff_v1", known_block_size=[64, 1, 1]
    )
    def producer(
        X: fx.Int64,
        W: fx.Int64,
        WS: fx.Int64,
        IDS: fx.Int64,
        PART: fx.Int64,
        OUT: fx.Int64,
    ):
        v4i = fx.Vector.make_type(4, fx.Int32)
        atoms = _scale_mma_atoms("fp4")
        tid = fx.Int32(gpu.thread_id("x"))
        route = fx.Int32(gpu.block_id("y"))
        tile = fx.Int32(gpu.block_id("x"))
        split = fx.Int32(gpu.block_id("z"))
        token, slot = route // 9, route % 9
        expert = fx.Int32(
            rocdl.readfirstlane(T.i32, as_ir_value(global_typed_ptr(IDS, T.i32)[route]))
        )
        raw = load4(X, (token * 6144 + split * 1024 + tid * 16) // 2)
        raw_hi = load4(X, (token * 6144 + split * 1024 + tid * 16) // 2 + 4)
        weights, scales = [], []
        for unit in range_constexpr(4):
            halves = []
            for half in range_constexpr(2):
                columns = []
                for block in range_constexpr(4):
                    col = tile * 32 + (block % 2) * 16 + (256 if block >= 2 else 0)
                    columns.append(
                        load4(
                            W,
                            (
                                expert * 512 * 3072
                                + col * 3072
                                + tid * 16
                                + (split * 8 + unit * 2 + half) * 1024
                            )
                            // 4,
                        )
                    )
                halves.append(columns)
            weights.append(halves)
            columns = []
            for block in range_constexpr(4):
                col = tile * 32 + (block % 2) * 16 + (256 if block >= 2 else 0)
                off = (
                    expert * 512 * 192
                    + (col // 32) * (32 * 192)
                    + tid * 4
                    + (split * 4 + unit) * 256
                )
                columns.append(global_typed_ptr(WS, T.i32)[off // 4])
            scales.append(columns)
        shared = fx.SharedAllocator().allocate(InputStorage).peek().raw.ptr
        shared_base = fx.Int32(fx.ptrtoint(shared))
        values = [fx.Vector(raw.load())[index] for index in range_constexpr(4)] + [
            fx.Vector(raw_hi.load())[index] for index in range_constexpr(4)
        ]
        maximum = fx.Int32(0)
        for index in range_constexpr(8):
            word = values[index] & fx.Int32(0x7FFF7FFF)
            maximum = _umax_i32(maximum, word & fx.Int32(0xFFFF))
            maximum = _umax_i32(maximum, word.shrui(fx.Int32(16)))
        exponent = _inline_e8m0(_inline_dpp_pair_amax(maximum))
        scale = as_ir_value((exponent << fx.Int32(23)).bitcast(fx.Float32))
        for word in range_constexpr(2):
            packed = as_ir_value(fx.Int32(0))
            for pair in range_constexpr(4):
                source = as_ir_value(
                    fx.Vector.from_elements(
                        [values[word * 4 + pair]], fx.Int32
                    ).bitcast(fx.BFloat16)
                )
                packed = rocdl.cvt_scalef32_pk_fp4_bf16(
                    T.i32, packed, source, scale, pair
                )
            lds_typed_ptr(shared_base, T.i32)[tid * 2 + word] = fx.Int32(packed)
        if tid % 2 == 0:
            lds_typed_ptr(shared_base, T.i8, 1)[512 + tid // 2] = exponent.to(fx.Int8)
        # A single-wave producer owns all input storage. Wait before its reads.
        rocdl.s_waitcnt(lgkmcnt=0)
        accumulators = [zero_frag(fx.Float32) for _ in range_constexpr(4)]
        for unit in range_constexpr(4):
            for half in range_constexpr(2):
                a = zero_frag(fx.Int32)
                a_scale = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
                a_scale.store(fx.Vector.filled(1, 127, fx.Int32))
                if tid % 16 == 0:
                    a.store(
                        lds_vec_load(
                            shared_base,
                            tid // 16 * 16 + unit * 128 + half * 64,
                            v4i,
                            fx.Int32,
                            align=16,
                        )
                    )
                    a_scale.store(
                        fx.Vector.from_elements(
                            [
                                lds_typed_ptr(shared_base, T.i8, 1)[
                                    512 + tid // 16 + unit * 8 + half * 4
                                ]
                                .to(fx.Uint8)
                                .to(fx.Int32)
                            ],
                            fx.Int32,
                        )
                    )
                for block in range_constexpr(4):
                    shift = half * 16 + (block % 2) * 8
                    bscale = (scales[unit][block].shrui(fx.Int32(shift))) & fx.Int32(
                        255
                    )
                    # Transposed single-row MFMA: weight is A, token input B.
                    fx.gemm(
                        atoms[(0, 0)],
                        accumulators[block],
                        weights[unit][half][block],
                        a,
                        accumulators[block],
                        scale_a=bscale,
                        scale_b=a_scale.load()[0],
                    )
        if tid % 16 == 0:
            for block in range_constexpr(4):
                col = (
                    tile * 32
                    + (block % 2) * 16
                    + (256 if block >= 2 else 0)
                    + tid // 16 * 4
                )
                ptr = (
                    global_typed_ptr(PART, T.f32, 16) + (split * 36 + route) * 512 + col
                )
                fx.ptr_store(accumulators[block].load(), ptr)
        # Slot zero distributes a complete output reset across all producer CTAs.
        if slot == 0:
            global_typed_ptr(OUT, T.i32)[
                token * 3072 + (split * 8 + tile) * 64 + tid
            ] = fx.Int32(0)

    @fx.struct
    class MiddleStorage:
        raw: fx.Array[fx.Uint8, 144, 16]

    marker = f"marlowe_native_m4_split6_g2_fp32mid7fffff_n{down_n}_w{down_waves}_v1"

    @flyc.kernel(name=marker, known_block_size=[down_waves * 64, 1, 1])
    def consumer(
        PART: fx.Int64,
        W: fx.Int64,
        WS: fx.Int64,
        IDS: fx.Int64,
        RW: fx.Int64,
        OUT: fx.Int64,
    ):
        v4i = fx.Vector.make_type(4, fx.Int32)
        v2f = fx.Vector.make_type(2, fx.Float32)
        atoms = _scale_mma_atoms("fp4")
        tid = fx.Int32(gpu.thread_id("x"))
        lane, wave = tid % 64, tid // 64
        route = fx.Int32(gpu.block_id("y"))
        tile = fx.Int32(gpu.block_id("x"))
        expert = fx.Int32(
            rocdl.readfirstlane(T.i32, as_ir_value(global_typed_ptr(IDS, T.i32)[route]))
        )
        routing = global_typed_ptr(RW, T.f32)[route]
        blocks = down_n // down_waves // 16
        wave_n = wave * (down_n // down_waves)
        weights, scales = [], []
        for half in range_constexpr(2):
            ws, ss = [], []
            for block in range_constexpr(blocks):
                col = tile * down_n + wave_n + block * 16
                ws.append(
                    load4(
                        W,
                        (expert * 6144 * 128 + col * 128 + lane * 16 + half * 1024)
                        // 4,
                    )
                )
                off = (
                    expert * 6144 * 8
                    + (col // 32) * 256
                    + lane * 4
                    + (col % 32) // 16
                    + half * 2
                )
                ss.append(global_typed_ptr(WS, T.i8, 1)[off].to(fx.Uint8).to(fx.Int32))
            weights.append(ws)
            scales.append(ss)
        shared = fx.SharedAllocator().allocate(MiddleStorage).peek().raw.ptr
        shared_base = fx.Int32(fx.ptrtoint(shared))
        if wave < 2:
            col = wave * 128 + lane * 2
            gate_up = []
            for offset in range_constexpr(2):
                parts = []
                for split in range_constexpr(8):
                    if const_expr(split < 6):
                        index = (split * 36 + route) * 512 + col + offset * 256
                        parts.append(
                            fx.Vector(
                                fx.ptr_load(
                                    global_typed_ptr(PART, T.f32, 8) + index,
                                    result_type=v2f,
                                )
                            )
                        )
                    else:
                        parts.append(fx.Vector.filled(2, 0.0, fx.Float32))
                for stride in (4, 2, 1):
                    parts = [
                        fx.Vector.from_elements(
                            [
                                parts[index][value] + parts[index + stride][value]
                                for value in range_constexpr(2)
                            ],
                            fx.Float32,
                        )
                        for index in range_constexpr(stride)
                    ]
                gate_up.append(parts[0])
            values = _activation_mul_batch(
                [gate_up[0][index] for index in range_constexpr(2)],
                [gate_up[1][index] for index in range_constexpr(2)],
                swiglu_limit=float("inf"),
            )
            maximum = (
                _fabs_f32(values[0]).maximumf(_fabs_f32(values[1])).bitcast(fx.Int32)
            )
            for shift in (1, 2, 4, 8):
                other = fx.Int32(
                    rocdl.ds_bpermute(
                        T.i32,
                        as_ir_value((lane ^ fx.Int32(shift)) * 4),
                        as_ir_value(maximum),
                    )
                )
                maximum = _umax_i32(maximum, other)
            exponent, scale = _e8m0_from_amax(maximum.bitcast(fx.Float32))
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32,
                as_ir_value(fx.Int32(0)),
                as_ir_value(values[0]),
                as_ir_value(values[1]),
                as_ir_value(scale),
                0,
            )
            lds_typed_ptr(shared_base, T.i8, 1)[wave * 64 + lane] = fx.Int32(packed).to(
                fx.Int8
            )
            if lane % 16 == 0:
                lds_typed_ptr(shared_base, T.i8, 1)[128 + wave * 4 + lane // 16] = (
                    exponent.to(fx.Int8)
                )
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        accumulators = [zero_frag(fx.Float32) for _ in range_constexpr(blocks)]
        for half in range_constexpr(2):
            a = zero_frag(fx.Int32)
            a_scale = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            a_scale.store(fx.Vector.filled(1, 127, fx.Int32))
            if lane % 16 == 0:
                a.store(
                    lds_vec_load(
                        shared_base,
                        lane // 16 * 16 + half * 64,
                        v4i,
                        fx.Int32,
                        align=16,
                    )
                )
                a_scale.store(
                    fx.Vector.from_elements(
                        [
                            lds_typed_ptr(shared_base, T.i8, 1)[
                                128 + lane // 16 + half * 4
                            ]
                            .to(fx.Uint8)
                            .to(fx.Int32)
                        ],
                        fx.Int32,
                    )
                )
            for block in range_constexpr(blocks):
                fx.gemm(
                    atoms[(0, 0)],
                    accumulators[block],
                    weights[half][block],
                    a,
                    accumulators[block],
                    scale_a=scales[half][block],
                    scale_b=a_scale.load()[0],
                )
        # Gather the four output elements from each of four MFMA lane groups.
        output = fx.Tensor(
            fx.make_view(
                global_typed_ptr(OUT, T.bf16, 4), fx.make_layout((1, 1), (1, 1))
            )
        )
        output = fx.rocdl.make_buffer_tensor(output, max_size=True)
        atomic = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(fx.BFloat16), fx.BFloat16)
        for block in range_constexpr(blocks):
            values = fx.Vector(accumulators[block].load())
            weighted = (
                fx.Vector.from_elements(
                    [values[value] * routing for value in range_constexpr(4)],
                    fx.Float32,
                )
                .to(fx.BFloat16)
                .bitcast(fx.Int32)
            )
            source_lane = (lane // 2) * 16
            low = fx.Int32(
                rocdl.ds_bpermute(
                    T.i32, as_ir_value(source_lane * 4), as_ir_value(weighted[0])
                )
            )
            high = fx.Int32(
                rocdl.ds_bpermute(
                    T.i32, as_ir_value(source_lane * 4), as_ir_value(weighted[1])
                )
            )
            word = (lane % 2 == 0).select(low, high)
            if lane < 8:
                fragment = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.BFloat16)
                fragment.store(
                    fx.Vector.from_elements([word], fx.Int32).bitcast(fx.BFloat16)
                )
                off = route // 9 * 6144 + tile * down_n + wave_n + block * 16 + lane * 2
                fx.copy(atomic, fragment, output[None, off])

    @flyc.jit
    def launch(
        X: fx.Int64,
        W1: fx.Int64,
        S1: fx.Int64,
        W2: fx.Int64,
        S2: fx.Int64,
        IDS: fx.Int64,
        RW: fx.Int64,
        PART: fx.Int64,
        OUT: fx.Int64,
        stream: fx.Stream,
    ):
        producer(X, W1, S1, IDS, PART, OUT).launch(
            grid=(8, 36, 6), block=(64, 1, 1), stream=stream
        )
        consumer(PART, W2, S2, IDS, RW, OUT).launch(
            grid=(6144 // down_n, 36, 1), block=(down_waves * 64, 1, 1), stream=stream
        )

    return launch


def make_operator(*, weights, rows, down_n=128, down_waves=4):
    import torch

    from .tensor_shim import _run_compiled

    if rows != 4:
        raise ValueError("this prepared specialization is exact M4 only")
    w1, w2, s1, s2 = (weights[key] for key in ("w1", "w2", "w1_scale", "w2_scale"))
    if tuple(w1.shape) != (257, 512, 3072) or tuple(w2.shape) != (257, 6144, 128):
        raise ValueError("unsupported native packed expert layout")
    if not all(value.element_size() == 1 for value in (w1, w2, s1, s2)):
        raise ValueError("native packed expert and scale elements must be bytes")
    if not all(getattr(value, "is_shuffled", False) for value in (w1, w2)):
        raise ValueError("both expert weights must carry native is_shuffled metadata")
    if not all(
        value.is_contiguous() and value.device == w1.device
        for value in (w1, w2, s1, s2)
    ):
        raise ValueError("expert storage must be contiguous on one device")
    if s1.numel() != 257 * 512 * 192 or s2.numel() != 257 * 6144 * 8:
        raise ValueError("unsupported native E8M0 scale storage")
    gfx = torch.cuda.get_device_properties(w1.device).gcnArchName.split(":")[0]
    if gfx != "gfx950":
        raise ValueError("the direct M4 kernel requires gfx950")
    partials = torch.empty((6, 36, 512), dtype=torch.float32, device=w1.device)
    output = torch.empty((4, 6144), dtype=torch.bfloat16, device=w1.device)
    launcher = compile_tiny_m4(down_n=down_n, down_waves=down_waves)

    class Operator:
        reference_contract = {
            "input_rounding": "flydsl_roundup_0x7fffff",
            "middle_rounding": "flydsl_roundup_0x7fffff",
            "fp32_middle": True,
            "output": "bf16_expert_operands",
        }
        info = {
            "algorithm": "exact M4 direct split6 producer/consumer",
            "input": "native BF16 MXFP4 group32",
            "middle": "FP32 SiLU, native 0x7fffff",
            "g1_reduction": "six K1024 MFMA partials, padded8 halves/quarters/pairs",
            "bit_identical_native_g1": False,
            "scratch_bytes": partials.numel() * partials.element_size()
            + output.numel() * output.element_size(),
            "down_n": down_n,
            "down_waves": down_waves,
            "markers": [
                "marlowe_native_m4_split6_g1_iq7fffff_v1",
                f"marlowe_native_m4_split6_g2_fp32mid7fffff_n{down_n}_w{down_waves}_v1",
            ],
        }

        def run(self, x, ids9, route_weights):
            if (
                tuple(x.shape) != (4, 6144)
                or tuple(ids9.shape) != (4, 9)
                or tuple(route_weights.shape) != (4, 9)
            ):
                raise ValueError(
                    "prepared M4 input/route shape mismatch; use native fallback"
                )
            if (x.dtype, ids9.dtype, route_weights.dtype) != (
                torch.bfloat16,
                torch.int32,
                torch.float32,
            ):
                raise ValueError("prepared M4 input/route dtype mismatch")
            if not all(
                value.is_contiguous() and value.device == w1.device
                for value in (x, ids9, route_weights)
            ):
                raise ValueError(
                    "prepared M4 inputs must be contiguous on the weight device"
                )
            stream = torch.cuda.current_stream(w1.device)
            _run_compiled(
                launcher,
                x.data_ptr(),
                w1.data_ptr(),
                s1.data_ptr(),
                w2.data_ptr(),
                s2.data_ptr(),
                ids9.data_ptr(),
                route_weights.data_ptr(),
                partials.data_ptr(),
                output.data_ptr(),
                stream,
            )
            return output

        def state_check(self):
            pointers = [
                value.data_ptr() for value in (w1, w2, s1, s2, partials, output)
            ]
            return {
                "passed": len(set(pointers)) == len(pointers),
                "partial_shape": list(partials.shape),
                "output_shape": list(output.shape),
                "reset": "producer slot0 clears all output; every split partial overwritten",
                "workspace": "owned per prepared operator; concurrent graph instances require separate operators",
            }

    return Operator()

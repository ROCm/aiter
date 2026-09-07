# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 FP8 sparse-MLA prefill specialized for the GLM-5.2 shape."""

import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

_NUM_HEADS = 16
_V_HEAD_DIM = 512
_ROPE_HEAD_DIM = 64
_HEAD_DIM = _V_HEAD_DIM + _ROPE_HEAD_DIM
_TOPK = 2048
_BLOCK_N = 64
_NUM_WAVES = 4
_WAVES_PER_EU = 2
_NUM_THREADS = 64 * _NUM_WAVES
_NUM_QK_TILES = _BLOCK_N // 16
_QK_TILES_PER_WAVE = _NUM_QK_TILES // _NUM_WAVES
_DV_TILES_PER_WAVE = (_V_HEAD_DIM // 16) // _NUM_WAVES
_PV_K_STEPS = _BLOCK_N // 32
_KV_LDS_PITCH = _V_HEAD_DIM + 16
_FP8_MAX = 448.0
_LOG2E = 1.4426950408889634
_INT32_MAX = 2**31 - 1


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def _lds_ptr(base_i32, offset):
    """Return an LDS pointer at ``base_i32 + offset`` bytes."""
    return _llvm.inttoptr(
        ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int32(base_i32 + offset))
    )


def _lds_store_i32x4(base, offset, value):
    _llvm.store(_raw(value), _lds_ptr(base, offset))


def _transpose_read_b8(base, offset):
    return rocdl.ds_read_tr8_b64_(
        ir.VectorType.get([2], ir.IntegerType.get_signless(32)),
        _lds_ptr(base, offset),
    )


def _flat_buffer_tensor(arg, dtype, alignment, size):
    source = fx.get_iter(arg)
    typed = fx.recast_iter(
        fx.PointerType.get(dtype, source.memspace, alignment), source
    )
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(typed, fx.make_layout((size,), (1,)))
    )


def _key_slot(tile, row):
    """Map a QK output item to the byte order required by the PV operand."""
    return 32 * (tile >> 1) + 8 * (row >> 2) + 4 * (tile & 1) + (row & 3)


@functools.lru_cache(maxsize=1)
def _compile_sparse_mla_prefill():
    lds_v_size = _BLOCK_N * _KV_LDS_PITCH
    lds_p_size = 64 * 48

    @fx.struct
    class SharedStorage:
        values: fx.Array[fx.Uint8, lds_v_size, 16]
        indices: fx.Array[fx.Int32, _TOPK, 16]
        probabilities: fx.Array[fx.Uint8, lds_p_size, 16]
        row_max: fx.Array[fx.Float32, _NUM_WAVES * 16, 16]
        row_sum: fx.Array[fx.Float32, _NUM_WAVES * 16, 16]

    @flyc.kernel(
        name="flydsl_sparse_mla_prefill_fp8_gfx950",
        known_block_size=[_NUM_THREADS, 1, 1],
    )
    def kernel(
        q_nope: fx.Tensor,
        q_rope: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        out: fx.Tensor,
        q_token_stride: fx.Int32,
        q_head_stride: fx.Int32,
        q_rope_token_stride: fx.Int32,
        q_rope_head_stride: fx.Int32,
        scale_bits: fx.Int32,
    ):
        f32 = T.f32
        scale_log2e = fx.Float32(arith.ArithValue(_raw(scale_bits)).bitcast(f32))
        neg_inf = fx.Float32(arith.constant(float("-inf"), type=f32))
        zero_f = fx.Float32(arith.constant(0.0, type=f32))
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        zero4i = Vec.filled(4, 0, fx.Int32)

        qk_mma = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN)
        )
        pv_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FN))
        load_i32x4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        store_bf16x4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)

        token = fx.block_idx.x
        thread = fx.thread_idx.x
        wave = thread // 64
        lane = thread % 64
        lane_group = lane // 16
        lane_col = lane % 16

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        values_base = fx.Int32(fx.ptrtoint(shared.values.ptr))
        indices_base = fx.Int32(fx.ptrtoint(shared.indices.ptr))
        probabilities_base = fx.Int32(fx.ptrtoint(shared.probabilities.ptr))

        q_buffer = _flat_buffer_tensor(q_nope, T.i32, 4, 1 << 28)
        q_rope_buffer = _flat_buffer_tensor(q_rope, T.i32, 4, 1 << 28)
        kv_buffer = _flat_buffer_tensor(kv, T.i32, 4, 1 << 28)
        indices_buffer = _flat_buffer_tensor(indices, T.i32, 4, 1 << 28)
        out_buffer = _flat_buffer_tensor(out, T.bf16, 2, 1 << 29)
        scalar_layout = fx.make_layout(1, 1)
        q_divided = fx.logical_divide(q_buffer, scalar_layout)
        q_rope_divided = fx.logical_divide(q_rope_buffer, scalar_layout)
        kv_divided = fx.logical_divide(kv_buffer, scalar_layout)
        indices_divided = fx.logical_divide(indices_buffer, scalar_layout)
        out_divided = fx.logical_divide(out_buffer, scalar_layout)

        def load_i32_vector4(divided, element):
            fragment = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(
                load_i32x4,
                fx.slice(divided, (None, fx.Int32(element))),
                fragment,
            )
            return Vec(fx.memref_load_vec(fragment))

        i32_type = ir.IntegerType.get_signless(32)
        i32x4_type = ir.VectorType.get([4], i32_type)

        # Stage one shared Q copy, distributing its four nope chunks across waves.
        # The existing index barrier also makes this copy visible before QK begins.
        q_base = token * q_token_stride + lane_col * q_head_stride
        q_rope_base = token * q_rope_token_stride + lane_col * q_rope_head_stride
        q_stage_head = lane_col * _HEAD_DIM
        q_stage_chunk = wave
        q_stage_offset = 128 * q_stage_chunk + 16 * lane_group
        _lds_store_i32x4(
            values_base,
            q_stage_head + q_stage_offset,
            load_i32_vector4(q_divided, q_base + 32 * q_stage_chunk + 4 * lane_group),
        )
        _lds_store_i32x4(
            values_base,
            q_stage_head + q_stage_offset + 64,
            load_i32_vector4(
                q_divided, q_base + 32 * q_stage_chunk + 16 + 4 * lane_group
            ),
        )
        if wave == fx.Int32(0):
            _lds_store_i32x4(
                values_base,
                q_stage_head + _V_HEAD_DIM + 16 * lane_group,
                load_i32_vector4(q_rope_divided, q_rope_base + 4 * lane_group),
            )

        transpose_base = (8 * lane_group + (lane_col // 2)) * _KV_LDS_PITCH + 8 * (
            lane_col % 2
        )
        index_row = token * _TOPK
        key_slot_col = 8 * (lane_col // 4) + (lane_col % 4)
        key_slot_group = 8 * lane_group

        # Stage indices once to avoid repeating scattered VMEM loads in the key loop.
        for iteration in range_constexpr(_TOPK // (_NUM_THREADS * 4)):
            element = thread * 4 + iteration * (_NUM_THREADS * 4)
            _lds_store_i32x4(
                indices_base,
                fx.Int32(element) * 4,
                load_i32_vector4(indices_divided, index_row + element),
            )
        fx.barrier()

        q_fragments = []
        for chunk in range_constexpr(4):
            low = Vec(
                _llvm.load(
                    i32x4_type,
                    _lds_ptr(
                        values_base,
                        q_stage_head + 128 * chunk + 16 * lane_group,
                    ),
                )
            )
            high = Vec(
                _llvm.load(
                    i32x4_type,
                    _lds_ptr(
                        values_base,
                        q_stage_head + 128 * chunk + 64 + 16 * lane_group,
                    ),
                )
            )
            fragment = fx.make_rmem_tensor(8, fx.Int32)
            fragment.store(low.shuffle(high, list(range(8))))
            q_fragments.append(fragment)
        rope = Vec(
            _llvm.load(
                i32x4_type,
                _lds_ptr(
                    values_base,
                    q_stage_head + _V_HEAD_DIM + 16 * lane_group,
                ),
            )
        )
        fragment = fx.make_rmem_tensor(8, fx.Int32)
        fragment.store(rope.shuffle(zero4i, list(range(8))))
        q_fragments.append(fragment)

        # The key loop reuses values LDS for KV. Wait until every wave has
        # consumed the staged Q before any wave can overwrite that storage.
        fx.barrier()

        i32_pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
        init = [_raw(neg_inf), _raw(zero_f)] + [
            _raw(zero4) for _ in range_constexpr(_DV_TILES_PER_WAVE)
        ]
        loop_result = init

        for key_start_iv, state in range(0, fx.Int32(_TOPK), _BLOCK_N, init=init):
            key_start = fx.Int32(key_start_iv)
            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            accumulators = [
                Vec(state[2 + item]) for item in range_constexpr(_DV_TILES_PER_WAVE)
            ]

            # Issue all gathered KV loads before the QK MFMAs.
            kv_low = []
            kv_high = []
            for tile_index in range_constexpr(_QK_TILES_PER_WAVE):
                tile = wave * _QK_TILES_PER_WAVE + tile_index
                row = fx.Int32(
                    _llvm.load(
                        i32_type,
                        _lds_ptr(
                            indices_base,
                            (key_start + fx.Int32(_key_slot(tile, 0)) + key_slot_col)
                            * 4,
                        ),
                    )
                )
                safe_row = (row >= fx.Int32(0)).select(row, fx.Int32(0))
                kv_base = safe_row * (_HEAD_DIM // 4)
                low_chunks = []
                high_chunks = []
                for chunk in range_constexpr(4):
                    low_chunks.append(
                        load_i32_vector4(
                            kv_divided,
                            kv_base + 32 * chunk + 4 * lane_group,
                        )
                    )
                    high_chunks.append(
                        load_i32_vector4(
                            kv_divided,
                            kv_base + 32 * chunk + 16 + 4 * lane_group,
                        )
                    )
                low_chunks.append(
                    load_i32_vector4(
                        kv_divided, kv_base + (_V_HEAD_DIM // 4) + 4 * lane_group
                    )
                )
                kv_low.append(low_chunks)
                kv_high.append(high_chunks)

            scores = []
            for tile_index in range_constexpr(_QK_TILES_PER_WAVE):
                tile = wave * _QK_TILES_PER_WAVE + tile_index
                score_fragment = fx.make_rmem_tensor(4, fx.Float32)
                score_fragment.store(zero4)
                for chunk in range_constexpr(4):
                    kv_fragment = fx.make_rmem_tensor(8, fx.Int32)
                    kv_fragment.store(
                        kv_low[tile_index][chunk].shuffle(
                            kv_high[tile_index][chunk], list(range(8))
                        )
                    )
                    fx.gemm(
                        qk_mma,
                        score_fragment,
                        kv_fragment,
                        q_fragments[chunk],
                        score_fragment,
                    )
                kv_fragment = fx.make_rmem_tensor(8, fx.Int32)
                kv_fragment.store(kv_low[tile_index][4].shuffle(zero4i, list(range(8))))
                fx.gemm(
                    qk_mma,
                    score_fragment,
                    kv_fragment,
                    q_fragments[4],
                    score_fragment,
                )
                score = Vec(score_fragment.load().ir_value())

                key_location = fx.Int32(_key_slot(tile, 0)) + key_slot_col
                value_offset = key_location * _KV_LDS_PITCH
                for chunk in range_constexpr(4):
                    _lds_store_i32x4(
                        values_base,
                        value_offset + 128 * chunk + 16 * lane_group,
                        kv_low[tile_index][chunk],
                    )
                    _lds_store_i32x4(
                        values_base,
                        value_offset + 128 * chunk + 64 + 16 * lane_group,
                        kv_high[tile_index][chunk],
                    )

                validity = Vec(
                    _llvm.load(
                        i32x4_type,
                        _lds_ptr(
                            indices_base,
                            (key_start + fx.Int32(_key_slot(tile, 0)) + key_slot_group)
                            * 4,
                        ),
                    )
                )
                scores.append(
                    [
                        (validity[item] >= fx.Int32(0)).select(
                            score[item] * scale_log2e, neg_inf
                        )
                        for item in range_constexpr(4)
                    ]
                )

            def max_f32(left, right):
                return fx.Float32(_llvm.intr_maxnum(_raw(left), _raw(right)))

            def cross_lane_pair(value, xor_mask):
                bits = arith.ArithValue(_raw(value)).bitcast(T.i32)
                op = (
                    rocdl.permlane16_swap
                    if const_expr(xor_mask == 16)
                    else rocdl.permlane32_swap
                )
                pair = op(i32_pair_type, _raw(bits), _raw(bits), False, False)
                first = _llvm.extractvalue(i32_type, pair, [0])
                second = _llvm.extractvalue(i32_type, pair, [1])
                return (
                    fx.Float32(arith.ArithValue(first).bitcast(T.f32)),
                    fx.Float32(arith.ArithValue(second).bitcast(T.f32)),
                )

            tile_max = scores[0][0]
            for tile_index in range_constexpr(_QK_TILES_PER_WAVE):
                for item in range_constexpr(4):
                    if const_expr(tile_index != 0 or item != 0):
                        value = scores[tile_index][item]
                        tile_max = max_f32(value, tile_max)
            for xor_mask in (16, 32):
                first, second = cross_lane_pair(tile_max, xor_mask)
                tile_max = max_f32(first, second)
            if lane < fx.Int32(16):
                fx.ptr_store(
                    tile_max,
                    fx.add_offset(shared.row_max.ptr, wave * 16 + lane_col),
                )
            fx.barrier()

            new_max = running_max
            for other_wave in range_constexpr(_NUM_WAVES):
                value = fx.Float32(
                    fx.ptr_load(
                        fx.add_offset(
                            shared.row_max.ptr,
                            fx.Int32(other_wave * 16) + lane_col,
                        )
                    )
                )
                new_max = max_f32(value, new_max)
            safe_max = (new_max > neg_inf).select(new_max, zero_f)
            alpha = fx.Float32(rocdl.exp2(f32, _raw(running_max - safe_max)))

            probability_sum = zero_f
            for tile_index in range_constexpr(_QK_TILES_PER_WAVE):
                tile = wave * _QK_TILES_PER_WAVE + tile_index
                probabilities = []
                for item in range_constexpr(4):
                    probability = fx.Float32(
                        rocdl.exp2(f32, _raw(scores[tile_index][item] - safe_max))
                    )
                    probability_sum = probability_sum + probability
                    probabilities.append(probability * fx.Float32(_FP8_MAX))
                packed = rocdl.cvt_pk_fp8_f32(
                    res=T.i32,
                    src_a=_raw(probabilities[0]),
                    src_b=_raw(probabilities[1]),
                    old=_llvm.mlir_undef(i32_type),
                    word_sel=False,
                )
                packed = rocdl.cvt_pk_fp8_f32(
                    res=T.i32,
                    src_a=_raw(probabilities[2]),
                    src_b=_raw(probabilities[3]),
                    old=fx.Int32(packed),
                    word_sel=True,
                )
                _llvm.store(
                    _raw(fx.Int32(packed)),
                    _lds_ptr(probabilities_base, fx.Int32(lane) * 48 + 4 * tile),
                )
            for xor_mask in (16, 32):
                first, second = cross_lane_pair(probability_sum, xor_mask)
                probability_sum = second + first
            running_sum = running_sum * alpha + probability_sum
            running_max = new_max
            fx.barrier()

            packed_probabilities = Vec(
                _llvm.load(
                    ir.VectorType.get([_NUM_QK_TILES], ir.IntegerType.get_signless(32)),
                    _lds_ptr(probabilities_base, fx.Int32(lane) * 48),
                )
            )
            probability_fragments = []
            for step in range_constexpr(_PV_K_STEPS):
                probability_fragment = fx.make_rmem_tensor(2, fx.Int32)
                probability_fragment.store(
                    packed_probabilities.shuffle(
                        packed_probabilities, [2 * step, 2 * step + 1]
                    )
                )
                probability_fragments.append(probability_fragment)

            def load_transposed_values(dv_offset, step):
                value_fragment = fx.make_rmem_tensor(2, fx.Int32)
                value_fragment.store(
                    Vec(
                        _transpose_read_b8(
                            values_base,
                            transpose_base + dv_offset + 32 * step * _KV_LDS_PITCH,
                        )
                    )
                )
                return value_fragment

            next_values = [
                load_transposed_values((wave * _DV_TILES_PER_WAVE) * 16, step)
                for step in range_constexpr(_PV_K_STEPS)
            ]
            alpha4 = Vec.from_elements([alpha] * 4, dtype=fx.Float32)
            for tile in range_constexpr(_DV_TILES_PER_WAVE):
                values = next_values
                if const_expr(tile + 1 < _DV_TILES_PER_WAVE):
                    next_values = [
                        load_transposed_values(
                            (wave * _DV_TILES_PER_WAVE + tile + 1) * 16,
                            step,
                        )
                        for step in range_constexpr(_PV_K_STEPS)
                    ]
                output_fragment = fx.make_rmem_tensor(4, fx.Float32)
                output_fragment.store(accumulators[tile] * alpha4)
                for step in range_constexpr(_PV_K_STEPS):
                    fx.gemm(
                        pv_mma,
                        output_fragment,
                        values[step],
                        probability_fragments[step],
                        output_fragment,
                    )
                accumulators[tile] = Vec(output_fragment.load().ir_value())
            fx.barrier()

            loop_result = yield (
                [_raw(running_max), _raw(running_sum)]
                + [
                    _raw(accumulators[item])
                    for item in range_constexpr(_DV_TILES_PER_WAVE)
                ]
            )

        final_sum = fx.Float32(loop_result[1])
        accumulators = [
            Vec(loop_result[2 + item]) for item in range_constexpr(_DV_TILES_PER_WAVE)
        ]
        if lane < fx.Int32(16):
            fx.ptr_store(
                final_sum,
                fx.add_offset(shared.row_sum.ptr, wave * 16 + lane_col),
            )
        fx.barrier()
        total_sum = zero_f
        for other_wave in range_constexpr(_NUM_WAVES):
            total_sum = total_sum + fx.Float32(
                fx.ptr_load(
                    fx.add_offset(
                        shared.row_sum.ptr,
                        fx.Int32(other_wave * 16) + lane_col,
                    )
                )
            )
        inverse = (total_sum > zero_f).select(
            fx.Float32(1.0) / (total_sum * fx.Float32(_FP8_MAX)), zero_f
        )
        inverse4 = Vec.from_elements([inverse] * 4, dtype=fx.Float32)

        output_base = (token * _NUM_HEADS + lane_col) * _V_HEAD_DIM
        for tile in range_constexpr(_DV_TILES_PER_WAVE):
            dv_offset = (
                fx.Int32((wave * _DV_TILES_PER_WAVE + tile) * 16) + 4 * lane_group
            )
            output_fragment = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
            output_fragment.store((accumulators[tile] * inverse4).to(fx.BFloat16))
            fx.copy(
                store_bf16x4,
                output_fragment,
                fx.slice(out_divided, (None, output_base + dv_offset)),
            )

    @flyc.jit
    def launch(
        q_nope: fx.Tensor,
        q_rope: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        out: fx.Tensor,
        q_token_stride: fx.Int32,
        q_head_stride: fx.Int32,
        q_rope_token_stride: fx.Int32,
        q_rope_head_stride: fx.Int32,
        scale_bits: fx.Int32,
        num_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q_nope,
            q_rope,
            kv,
            indices,
            out,
            q_token_stride,
            q_head_stride,
            q_rope_token_stride,
            q_rope_head_stride,
            scale_bits,
            value_attrs={
                "rocdl.waves_per_eu": _WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{_NUM_THREADS},{_NUM_THREADS}",
            },
        ).launch(
            grid=(num_tokens, 1, 1),
            block=(_NUM_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def _require_tensor(tensor, *, name, shape, dtype, device, contiguous=True):
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_vector_load_layout(tensor, *, name):
    """Require unit inner stride and i32-aligned outer strides, without copying."""
    if tensor.stride(2) != 1 or tensor.stride(0) % 4 or tensor.stride(1) % 4:
        raise ValueError(
            f"{name} must have unit inner stride and 4-byte-aligned token/head "
            f"strides, got {tensor.stride()}"
        )


def _pointer_tensor(tensor):
    """A one-element byte view carrying only the original tensor's data pointer."""
    return tensor.as_strided((1,), (1,)).view(torch.int8)


def flydsl_sparse_mla_prefill(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse MLA for Q ``[T,16,512+64]`` and 2048 indices per token."""
    if get_gfx() != "gfx950":
        raise RuntimeError(
            f"flydsl_sparse_mla_prefill requires gfx950, got {get_gfx()}"
        )
    if q_nope.ndim != 3:
        raise ValueError(f"q_nope must be rank 3, got rank {q_nope.ndim}")
    num_tokens = q_nope.shape[0]
    device = q_nope.device
    fp8_dtype = torch.float8_e4m3fn
    _require_tensor(
        q_nope,
        name="q_nope",
        shape=(num_tokens, _NUM_HEADS, _V_HEAD_DIM),
        dtype=fp8_dtype,
        device=device,
        contiguous=False,
    )
    _require_vector_load_layout(q_nope, name="q_nope")
    _require_tensor(
        q_rope,
        name="q_rope",
        shape=(num_tokens, _NUM_HEADS, _ROPE_HEAD_DIM),
        dtype=fp8_dtype,
        device=device,
        contiguous=False,
    )
    _require_vector_load_layout(q_rope, name="q_rope")
    if kv.ndim == 3 and kv.shape[1] == 1:
        kv = kv.squeeze(1)
    if kv.ndim != 2 or kv.shape[1] != _HEAD_DIM:
        raise ValueError(
            f"kv must have shape [P, {_HEAD_DIM}] or [P, 1, {_HEAD_DIM}], "
            f"got {tuple(kv.shape)}"
        )
    if kv.dtype != fp8_dtype or kv.device != device or not kv.is_contiguous():
        raise ValueError(
            "kv must be contiguous float8_e4m3fn on the same device as q_nope"
        )
    # The launcher hands the kernel `kv.view(torch.int8).reshape(-1)`, and the
    # extent of that tensor is packed as a 32-bit int. Past 2 GiB the pack
    # itself raises `'i' format requires -2147483648 <= number <= 2147483647`
    # from inside the shim -- and the launch that follows faults the GPU rather
    # than failing cleanly. Decode does not share the limit: it passes kv as a
    # bare pointer. Say so here instead of letting a caller discover it as a
    # memory access fault.
    kv_bytes = kv.numel() * kv.element_size()
    if kv_bytes > _INT32_MAX:
        raise ValueError(
            f"kv is {kv_bytes / (1 << 30):.2f} GiB; sparse MLA prefill addresses "
            f"it through a 32-bit extent and supports at most "
            f"{_INT32_MAX / (1 << 30):.2f} GiB per call"
        )
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices.squeeze(1)
    elif indices.ndim != 2:
        raise ValueError(
            "indices must have shape [T, 2048] or [T, 1, 2048], "
            f"got {tuple(indices.shape)}"
        )
    _require_tensor(
        indices,
        name="indices",
        shape=(num_tokens, _TOPK),
        dtype=torch.int32,
        device=device,
    )
    if out is None:
        out = torch.empty(
            (num_tokens, _NUM_HEADS, _V_HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        _require_tensor(
            out,
            name="out",
            shape=(num_tokens, _NUM_HEADS, _V_HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )

    scale_bits = struct.unpack("<i", struct.pack("<f", float(softmax_scale) * _LOG2E))[
        0
    ]
    with torch.cuda.device(device):
        _run_compiled(
            _compile_sparse_mla_prefill(),
            _pointer_tensor(q_nope),
            _pointer_tensor(q_rope),
            kv.view(torch.int8).reshape(-1),
            indices.reshape(-1),
            out.reshape(-1),
            int(q_nope.stride(0) // 4),
            int(q_nope.stride(1) // 4),
            int(q_rope.stride(0) // 4),
            int(q_rope.stride(1) // 4),
            scale_bits,
            int(num_tokens),
            fx.Stream(torch.cuda.current_stream(device=device)),
        )
    return out


__all__ = ["flydsl_sparse_mla_prefill"]

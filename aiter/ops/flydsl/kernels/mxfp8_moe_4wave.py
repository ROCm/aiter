# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Four-wave A8W4 MoE tiles with sorted FP8 activation output."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.expr.typing import Vector as Vec

from .gemm_mxfp8_4wave import compile_mxfp8_gemm_4w
from .mxfp8_moe_8wave import _mxfp8_exponent, _pack_fp8x8, _store_factory


def _store_factory_4w_quant(swiglu_limit):
    exponent_for_amax = _mxfp8_exponent
    pack_fp8x8 = _pack_fp8x8

    def factory(C, rows, cols, idx, n_tiles_a, n_tiles_b, scratch):
        assert n_tiles_a == 2 and n_tiles_b == 4
        cols = cols // 2
        tile_n = n_tiles_b * 8
        lane = fx.thread_idx.x % 64
        wave = fx.thread_idx.x // 64
        # Each wave owns a full 32-column activation quantization group.
        base = fx.Int32(fx.ptrtoint(scratch[0]))
        for i in range_constexpr(1, 4):
            base = (wave == i).select(fx.Int32(fx.ptrtoint(scratch[i])), base)
        ptr = fx.recast_iter(fx.BFloat16, fx.inttoptr(scratch[0].type, base))
        kp = (cols + 255) // 256 * 256
        scales = fx.rocdl.make_buffer_tensor(
            C, max_size=False, num_records_bytes=fx.Int64(rows) * (kp + kp // 32)
        )
        out = fx.logical_divide(scales, fx.make_layout(16, 1))
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int8)

        def scratch_at(row, col, width):
            offset = row * tile_n + (col ^ ((row % (tile_n // 8)) * 8))
            return fx.make_view(ptr + offset, fx.make_layout(width, 1))

        def store_scale(row, kg, exponent, valid):
            scale_index = (
                (row // 32 * (kp // 256) + kg // 8) * 64 + kg % 4 * 16 + row % 16
            ) * 4
            scale_index += kg // 4 % 2 * 2 + row // 16 % 2
            scales[valid.select(rows * kp + scale_index, fx.Int32(-1))] = exponent.to(
                fx.Int8
            )

        def quant_store(base_row, base_col):
            row = lane // 2
            col = lane % 2 * 16
            values = [
                scratch_at(row, col + chunk * 8, 8).load()
                for chunk in range_constexpr(2)
            ]
            amax_bits = fx.Int32(0)
            for chunk in range_constexpr(2):
                bits = (
                    (values[chunk].bitcast(fx.Int16) & 0x7FFF)
                    .reduce(ReductionOp.MAX)
                    .to(fx.Int32)
                )
                amax_bits = fx.max(amax_bits, bits)
            amax_bits = fx.max(amax_bits, fx.gpu.shuffle_xor(amax_bits, 1, 64))
            exponent = exponent_for_amax(amax_bits)
            scale = (exponent << 23).bitcast(fx.Float32)
            words = []
            for chunk in range_constexpr(2):
                packed = pack_fp8x8(values[chunk], scale)
                words.extend([packed[0], packed[1]])
            reg = fx.make_rmem_tensor(16, fx.Int8)
            reg.store(Vec.from_elements(words, fx.Int32).bitcast(fx.Int8))
            offset = (base_row + row) * kp + base_col + col
            fx.copy(atom, reg, fx.slice(out, (None, offset >> 4)))
            store_scale(base_row + row, base_col // 32, exponent, lane % 2 == 0)
            # Write every padded K column and its scale on every live row.
            pad = (kp != cols) & (base_col >= cols - 128)
            reg.store(Vec.filled(16, 0, fx.Int8))
            pad_offset = pad.select(offset + 128, fx.Int32(-16))
            fx.copy(atom, reg, fx.slice(out, (None, pad_offset >> 4)))
            store_scale(
                base_row + row,
                (base_col + 128) // 32,
                fx.Int32(19),
                pad & (lane % 2 == 0),
            )
            rocdl.s_barrier()

        def store(c_frag, base_row, base_col):
            for ti in range_constexpr(n_tiles_a):
                row = ti * 16 + lane // 16 * 4
                for tj in range_constexpr(n_tiles_b // 2):
                    col = tj * 16 + lane % 16
                    gate_values = Vec(c_frag[idx(ti, tj * 2)])
                    up_values = Vec(c_frag[idx(ti, tj * 2 + 1)])
                    for i in range_constexpr(4):
                        gate, linear = gate_values[i], up_values[i]
                        if const_expr(swiglu_limit):
                            gate = fx.min(gate, swiglu_limit)
                            linear = fx.max(fx.min(linear, swiglu_limit), -swiglu_limit)
                        value = gate / (1.0 + fmath.exp(-gate)) * linear
                        scratch_at(row + i, col, 1).store(
                            Vec.filled(1, value.to(fx.BFloat16), fx.BFloat16)
                        )
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            quant_store(base_row, base_col // 2)

        return store

    return factory


def compile_mxfp8_moe_gemm_4w(
    *,
    K,
    stage,
    xcd_swizzle=1,
    logical_k=None,
    gather_a=False,
    tile_m=128,
    tile_n=256,
    expert_block_m=128,
    b_k=None,
    dynamic_rows=True,
    swiglu_limit=None,
    activation_type="silu",
    b_dtype="fp4",
    fuse_quant=False,
):
    """Four-wave G1 and compatible G2; G2 keeps the existing BF16 reduction ABI."""
    assert stage in (1, 2) and activation_type == "silu" and b_dtype == "fp4"
    assert dynamic_rows
    assert fuse_quant == (stage == 1)
    store = (
        _store_factory_4w_quant(0.0 if swiglu_limit is None else swiglu_limit)
        if stage == 1
        else _store_factory(transpose=True)
    )
    return compile_mxfp8_gemm_4w(
        K=K,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        expert_block_m=expert_block_m,
        b_k=b_k,
        xcd_swizzle=xcd_swizzle,
        logical_k=logical_k,
        gather_a=gather_a,
        store_factory=store,
    )

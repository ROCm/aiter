# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP32 serial prefix-scan of packed [Aᵀ,Cᵀ] GDN block maps."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from ..gdr_common import _gview, _load_vec, _store_vec


def _prefetch_a(maps, block, k_base, col, group, wave):
    cp = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    return fx.Vector.from_elements(
        [
            _load_vec(
                cp,
                fx.slice(
                    maps,
                    (
                        block,
                        k_base + kk * 4 + group,
                        panel * 64 + wave * 16 + col,
                        None,
                    ),
                ),
                1,
                fx.Float32,
            )
            for kk in range(8)
            for panel in range(2)
        ],
        dtype=fx.Float32,
    )


@flyc.jit
def _affine_step(
    maps: fx.Tensor,
    rhs: fx.Tensor,
    block: fx.Int32,
    next_block: fx.Int32,
    first_a: fx.Vector,
    c_values: fx.Vector,
    col: fx.Int32,
    group: fx.Int32,
    wave: fx.Int32,
):
    """Advance affine carry with an LDS RHS and prefetched maps."""
    lds_cp = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32, fx.Float32))
    frag_a = fx.make_rmem_tensor(1, fx.Float32)
    frag_b = fx.make_rmem_tensor(1, fx.Float32)
    frag_c = fx.make_rmem_tensor(4, fx.Float32)
    acc_count = 2

    def multiply(a_values, k_base, accumulators):
        values = list(accumulators)
        for kk in range_constexpr(8):
            sv = _load_vec(
                lds_cp,
                fx.slice(rhs, (k_base + kk * 4 + group, col, None)),
                1,
                fx.Float32,
            )
            frag_b.store(fx.Vector.from_elements([sv], dtype=fx.Float32))
            for panel in range_constexpr(2):
                frag_a.store(
                    fx.Vector.from_elements(
                        [a_values[kk * 2 + panel]], dtype=fx.Float32
                    )
                )
                frag_c.store(fx.Vector(values[panel]))
                fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)
                values[panel] = frag_c.load()
        return values

    for chunk, carried in range(
        fx.Int32(0),
        fx.Int32(3),
        fx.Int32(1),
        init=[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(acc_count)]
        + [first_a],
    ):
        k_base = fx.Int32(chunk) * 32
        next_a = _prefetch_a(maps, block, k_base + 32, col, group, wave)
        accumulators = multiply(
            fx.Vector(carried[acc_count]), k_base, carried[:acc_count]
        )
        result = yield accumulators + [next_a]

    # The last prefetch must remain in bounds.
    next_a = _prefetch_a(maps, next_block, fx.Int32(0), col, group, wave)
    accumulators = multiply(
        fx.Vector(result[acc_count]), fx.Int32(96), result[:acc_count]
    )
    output = fx.Vector.from_elements(
        [
            fx.Vector(accumulators[slot])[j] + c_values[slot * 4 + j]
            for slot in range(acc_count)
            for j in range(4)
        ],
        dtype=fx.Float32,
    )
    return output, next_a


def compile_chunk_gdn_carry(
    *, H: int, use_initial_state: bool, STATE_DTYPE_BF16: bool = False
):
    K = V = 128
    BV = 16
    THREADS = 256

    @fx.struct
    class SharedStorage:
        state: fx.Array[fx.Float32, K * BV, 16]

    @flyc.kernel
    def carry_kernel(
        maps_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        entry_tensor: fx.Tensor,
        block_prefix_tensor: fx.Tensor,
        blocks: fx.Int32,
        requests: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        tile_v = fx.Int32(gpu.block_id("x"))
        request_head = fx.Int32(gpu.block_id("y"))
        head = request_head % H
        request = request_head // H
        lane = tid % 64
        wave = tid // 64
        col = lane % 16
        group = lane // 16
        v = tile_v * BV + col
        cp_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        prefix = _gview(block_prefix_tensor, None, (requests + 1, 1), (1, 1))
        first = _load_vec(cp_i32, fx.slice(prefix, (request, None)), 1, fx.Int32)
        end = _load_vec(cp_i32, fx.slice(prefix, (request + 1, None)), 1, fx.Int32)
        cp = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        state_num = fx.BFloat16 if STATE_DTYPE_BF16 else fx.Float32
        cp_state = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if STATE_DTYPE_BF16 else fx.rocdl.BufferCopy32b(),
            state_num,
        )
        lds_cp = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        # Packed [Aᵀ,Cᵀ] requires the parent strides.
        maps = _gview(
            maps_tensor,
            head * (K + V) * K,
            (blocks, K + V, K, 1),
            (H * (K + V) * K, K, 1, 1),
        )
        entry = _gview(
            entry_tensor, head * K * V, (blocks, K, V, 1), (H * K * V, V, 1, 1)
        )
        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        state = shared.state.view(fx.make_layout((K, BV, 1), (BV, 1, 1)))
        for panel in range_constexpr(2):
            for j in range_constexpr(4):
                row = panel * 64 + wave * 16 + group * 4 + j
                seed = fx.Float32(0.0)
                if const_expr(use_initial_state):
                    # Public h0 [V,K] transposes into carry [K,V].
                    h0 = _gview(h0_tensor, request_head * V * K, (V, K, 1), (K, 1, 1))
                    loaded_seed = _load_vec(
                        cp_state, fx.slice(h0, (v, row, None)), 1, state_num
                    )
                    if const_expr(state_num == fx.BFloat16):
                        loaded_seed = loaded_seed.to(fx.Float32)
                    seed = loaded_seed
                _store_vec(
                    lds_cp, fx.slice(state, (row, col, None)), seed, 1, fx.Float32
                )
                if first < end:
                    _store_vec(
                        cp, fx.slice(entry, (first, row, v, None)), seed, 1, fx.Float32
                    )
        gpu.barrier()

        # Empty requests still prefetch one map.
        seed_block = (first < end).select(first, fx.Int32(0))
        first_a = _prefetch_a(maps, seed_block, fx.Int32(0), col, group, wave)
        for block, carried_a in range(first, end - 1, fx.Int32(1), init=[first_a]):
            b = fx.Int32(block)
            c_values = fx.Vector.from_elements(
                [
                    _load_vec(
                        cp,
                        fx.slice(
                            maps,
                            (b, K + v, panel * 64 + wave * 16 + group * 4 + j, None),
                        ),
                        1,
                        fx.Float32,
                    )
                    for panel in range(2)
                    for j in range(4)
                ],
                dtype=fx.Float32,
            )
            next_block = (b + 1 < end - 1).select(b + 1, b)
            values, next_a = _affine_step(
                maps,
                state,
                b,
                next_block,
                fx.Vector(carried_a[0]),
                c_values,
                col,
                group,
                wave,
            )
            gpu.barrier()
            for panel in range_constexpr(2):
                for j in range_constexpr(4):
                    row = panel * 64 + wave * 16 + group * 4 + j
                    next_state = values[panel * 4 + j]
                    _store_vec(
                        lds_cp,
                        fx.slice(state, (row, col, None)),
                        next_state,
                        1,
                        fx.Float32,
                    )
                    _store_vec(
                        cp,
                        fx.slice(entry, (b + 1, row, v, None)),
                        next_state,
                        1,
                        fx.Float32,
                    )
            gpu.barrier()
            _carried_result = yield [next_a]

    @flyc.jit
    def launch(
        maps_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        entry_tensor: fx.Tensor,
        block_prefix_tensor: fx.Tensor,
        blocks: fx.Int32,
        requests: fx.Int32,
        stream: fx.Stream,
    ):
        carry_kernel(
            maps_tensor, h0_tensor, entry_tensor, block_prefix_tensor, blocks, requests
        ).launch(grid=(V // BV, requests * H, 1), block=(THREADS, 1, 1), stream=stream)

    return launch

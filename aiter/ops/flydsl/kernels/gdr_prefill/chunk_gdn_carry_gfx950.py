# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP32 serial prefix-scan of packed [Aᵀ,Cᵀ] GDN block maps."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from ..gdr_common import _gview, _load_vec, _store_vec


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
        # Keep the packed parent strides; build-map stores probe-major [Aᵀ,Cᵀ].
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
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32, fx.Float32))
        frag_a = fx.make_rmem_tensor(1, fx.Float32)
        frag_b = fx.make_rmem_tensor(1, fx.Float32)
        frag_c = [fx.make_rmem_tensor(4, fx.Float32) for _ in range(2)]
        for panel in range_constexpr(2):
            for j in range_constexpr(4):
                row = panel * 64 + wave * 16 + group * 4 + j
                seed = fx.Float32(0.0)
                if const_expr(use_initial_state):
                    # Public h0 is [N,H,V,K]; maps and entries use native [K,V].
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

        for block in range(first, end - 1, fx.Int32(1)):
            b = fx.Int32(block)
            # Four waves cover 64 rows; retain both panels until all old-state
            # reads finish. FP32 inputs avoid truncating the accumulated maps.
            for kk, acc in range(
                fx.Int32(0),
                fx.Int32(K),
                fx.Int32(4),
                init=[
                    fx.Vector.filled(4, 0.0, fx.Float32),
                    fx.Vector.filled(4, 0.0, fx.Float32),
                ],
            ):
                k = fx.Int32(kk) + group
                sv = _load_vec(lds_cp, fx.slice(state, (k, col, None)), 1, fx.Float32)
                frag_b.store(fx.Vector.from_elements([sv], dtype=fx.Float32))
                for panel in range_constexpr(2):
                    row_a = panel * 64 + wave * 16 + col
                    av = _load_vec(
                        cp, fx.slice(maps, (b, k, row_a, None)), 1, fx.Float32
                    )
                    frag_a.store(fx.Vector.from_elements([av], dtype=fx.Float32))
                    frag_c[panel].store(fx.Vector(acc[panel]))
                    fx.gemm(mma, frag_c[panel], frag_a, frag_b, frag_c[panel])
                result = yield [frag_c[0].load(), frag_c[1].load()]
            gpu.barrier()
            for panel in range_constexpr(2):
                values = fx.Vector(result[panel])
                for j in range_constexpr(4):
                    row = panel * 64 + wave * 16 + group * 4 + j
                    next_state = values[j] + _load_vec(
                        cp, fx.slice(maps, (b, K + v, row, None)), 1, fx.Float32
                    )
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

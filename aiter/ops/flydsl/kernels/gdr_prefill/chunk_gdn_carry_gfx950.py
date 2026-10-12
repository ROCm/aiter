# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP32 serial prefix-scan of packed [Aᵀ,Cᵀ] GDN block maps."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from ..gdr_common import _gview, _load_vec


def _tiled_mma():
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32, fx.Float32))
    return fx.make_tiled_mma(mma, fx.make_layout((4, 1, 1), (1, 0, 0)))


def _prefetch_a(maps, block, k_base, tid):
    cp = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tmma = _tiled_mma()
    copy_a = fx.make_tiled_copy_A(cp, tmma).get_slice(tid)
    a = fx.make_view(
        fx.get_iter(fx.slice(maps, (block, k_base, 0, None))),
        fx.make_layout((128, 32), (1, 128)),
    )
    fragment = tmma.get_slice(tid).make_fragment_A(a)
    fx.copy(cp, copy_a.partition_S(a), copy_a.retile(fragment))
    return fragment.load()


def _load_c(atom, view, tid, numeric):
    copy_c = fx.make_tiled_copy_C(atom, _tiled_mma()).get_slice(tid)
    partition = copy_c.partition_S(view)
    return fx.Vector.from_elements(
        [
            _load_vec(atom, partition[(None, j), panel, 0], 1, numeric)
            for panel in range(2)
            for j in range(4)
        ],
        dtype=numeric,
    )


def _store_c(atom, view, values, tid):
    tmma = _tiled_mma()
    copy_c = fx.make_tiled_copy_C(atom, tmma).get_slice(tid)
    partition = copy_c.partition_D(view)
    fragment = tmma.get_slice(tid).make_fragment_C(view)
    fragment.store(values)
    retiled = copy_c.retile(fragment)
    for panel in range_constexpr(2):
        for j in range_constexpr(4):
            fx.copy(
                atom,
                retiled[(None, j), panel, 0],
                partition[(None, j), panel, 0],
            )


@flyc.jit
def _affine_step(
    maps: fx.Tensor,
    rhs: fx.Tensor,
    block: fx.Int32,
    next_block: fx.Int32,
    first_a: fx.Vector,
    c_values: fx.Vector,
    tid: fx.Int32,
):
    """Advance affine carry with an LDS RHS and prefetched maps."""
    lds_cp = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    tmma = _tiled_mma()
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32, fx.Float32))
    thr_mma = tmma.get_slice(tid)
    copy_b = fx.make_tiled_copy_B(lds_cp, tmma).get_slice(tid)
    b_view = fx.make_view(fx.get_iter(rhs), fx.make_layout((16, 128), (1, 16)))
    p_b = copy_b.partition_S(b_view)
    frag_b = fx.make_rmem_tensor(1, fx.Float32)
    a_view = fx.make_view(fx.get_iter(rhs), fx.make_layout((128, 32), (1, 128)))
    frag_a = fx.make_rmem_tensor(
        fx.size(thr_mma.partition_A(a_view)[None, 0, 0]), fx.Float32
    )
    c_view = fx.make_view(fx.get_iter(rhs), fx.make_layout((128, 16), (16, 1)))
    frag_c = thr_mma.make_fragment_C(c_view)
    c_panel = fx.make_rmem_tensor(
        fx.size(thr_mma.partition_C(c_view)[None, 0, None]), fx.Float32
    )

    def multiply(a_values, k_base, accumulators):
        acc = fx.Vector(accumulators)
        values = [
            fx.Vector.from_elements(
                [acc[panel * 4 + j] for j in range(4)], dtype=fx.Float32
            )
            for panel in range(2)
        ]
        for kk in range_constexpr(8):
            fx.copy(lds_cp, p_b[None, None, k_base // 4 + kk], frag_b)
            for panel in range_constexpr(2):
                frag_a.store(
                    fx.Vector.from_elements(
                        [a_values[kk * 2 + panel]], dtype=fx.Float32
                    )
                )
                c_panel.store(fx.Vector(values[panel]))
                fx.gemm(mma, c_panel, frag_a, frag_b, c_panel)
                values[panel] = c_panel.load()
        frag_c.store(
            fx.Vector.from_elements(
                [fx.Vector(values[panel])[j] for panel in range(2) for j in range(4)],
                dtype=fx.Float32,
            )
        )
        return frag_c.load()

    for chunk, carried in range(
        fx.Int32(0),
        fx.Int32(3),
        fx.Int32(1),
        init=[fx.Vector.filled(8, 0.0, fx.Float32), first_a],
    ):
        k_base = fx.Int32(chunk) * 32
        next_a = _prefetch_a(maps, block, k_base + 32, tid)
        accumulators = multiply(fx.Vector(carried[1]), k_base, carried[0])
        result = yield [accumulators, next_a]

    # The last prefetch must remain in bounds.
    next_a = _prefetch_a(maps, next_block, fx.Int32(0), tid)
    accumulators = multiply(fx.Vector(result[1]), fx.Int32(96), result[0])
    output = fx.Vector(accumulators) + c_values
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
        v_base = tile_v * BV
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
        state = shared.state.view(fx.make_layout((K, BV), (BV, 1)))
        seed = fx.Vector.filled(8, 0.0, fx.Float32)
        if const_expr(use_initial_state):
            # Public h0 [V,K] transposes into carry [K,V].
            h0 = _gview(
                h0_tensor,
                request_head * V * K + K * v_base,
                (K, BV),
                (1, K),
            )
            loaded_seed = _load_c(cp_state, h0, tid, state_num)
            if const_expr(state_num == fx.BFloat16):
                loaded_seed = loaded_seed.to(fx.Float32)
            seed = loaded_seed
        _store_c(lds_cp, state, seed, tid)
        if first < end:
            entry_first = fx.make_view(
                fx.get_iter(fx.slice(entry, (first, 0, v_base, None))),
                fx.make_layout((K, BV), (V, 1)),
            )
            _store_c(cp, entry_first, seed, tid)
        gpu.barrier()

        # Empty requests still prefetch one map.
        seed_block = (first < end).select(first, fx.Int32(0))
        first_a = _prefetch_a(maps, seed_block, fx.Int32(0), tid)
        for block, carried_a in range(first, end - 1, fx.Int32(1), init=[first_a]):
            b = fx.Int32(block)
            c_view = fx.make_view(
                fx.get_iter(fx.slice(maps, (b, K + v_base, 0, None))),
                fx.make_layout((K, BV), (1, K)),
            )
            c_values = _load_c(cp, c_view, tid, fx.Float32)
            # At the last block (b + 1 == end - 1) reuse b: one redundant affine step
            # is cheaper than branching around it.
            next_block = (b + 1 < end - 1).select(b + 1, b)
            values, next_a = _affine_step(
                maps,
                state,
                b,
                next_block,
                fx.Vector(carried_a[0]),
                c_values,
                tid,
            )
            gpu.barrier()
            _store_c(lds_cp, state, values, tid)
            entry_next = fx.make_view(
                fx.get_iter(fx.slice(entry, (b + 1, 0, v_base, None))),
                fx.make_layout((K, BV), (V, 1)),
            )
            _store_c(cp, entry_next, values, tid)
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

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Expert-grouped MXFP8 prefill GEMMs sharing the dense eight-wave pipeline.

Output rows are sorted by expert and each expert is padded to 256 rows.
Stage 1 can gather source-token A using ``row_map``; stage 2 uses sorted A. Weights use the
dense kernel's 16x64 preshuffle and scales use ``shuffle_scale_w4``. For stage 1,
interleave gate/up in groups of 16 rows before preshuffling. K is padded to 256
(in particular, MiniMax M3 TP8's down projection uses K=512 for logical K=384).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec

from .gemm import compile_mxfp8_gemm_8w


def _store_factory(
    *, activation=False, transpose=False, mask_n=False, swiglu_limit=7.0
):
    def factory(C, rows, cols, idx, n_tiles_a, n_tiles_b, scratch):
        cols = cols // 2 if activation else cols
        tile_n = n_tiles_b * (8 if activation else 16)
        tile_m = n_tiles_a * 16
        lane = fx.thread_idx.x % 64
        wave = fx.thread_idx.x // 64
        # SharedAllocator fields are independent LDS globals, not one contiguous array.
        base = fx.Int32(fx.ptrtoint(scratch[0]))
        for i in range_constexpr(1, 8):
            base = (wave == i).select(fx.Int32(fx.ptrtoint(scratch[i])), base)
        ptr = fx.recast_iter(fx.BFloat16, fx.inttoptr(scratch[0].type, base))
        out = fx.rocdl.make_buffer_tensor(
            C, max_size=False, num_records_bytes=fx.Int64(rows) * cols * 2
        )
        out = fx.logical_divide(out, fx.make_layout(8, 1))
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

        def scratch_at(row, col, width):
            offset = row * tile_n + (col ^ ((row % (tile_n // 8)) * 8))
            return fx.make_view(ptr + offset, fx.make_layout(width, 1))

        def store(c_frag, base_row, base_col):
            for ti in range_constexpr(n_tiles_a):
                row = ti * 16 + lane // 16 * 4
                for tj in range_constexpr(n_tiles_b // 2 if activation else n_tiles_b):
                    col = tj * 16 + lane % 16
                    value = Vec(c_frag[idx(ti, tj * 2 if activation else tj)])
                    if const_expr(transpose):
                        offset = row // 4 * (tile_n * 4) + col * 4
                        fx.make_view(ptr + offset, fx.make_layout(4, 1)).store(
                            value.to(fx.BFloat16)
                        )
                    else:
                        if const_expr(activation):
                            up = Vec(c_frag[idx(ti, tj * 2 + 1)])
                        for i in range_constexpr(4):
                            v = value[i]
                            if const_expr(activation):
                                gate = fx.min(v, swiglu_limit)
                                linear = fx.max(
                                    fx.min(up[i], swiglu_limit), -swiglu_limit
                                )
                                v = (
                                    gate
                                    / (1.0 + fmath.exp(-1.702 * gate))
                                    * (linear + 1.0)
                                )
                            scratch_at(row + i, col, 1).store(
                                Vec.filled(1, v.to(fx.BFloat16), fx.BFloat16)
                            )
            rocdl.s_waitcnt(lgkmcnt=0)
            if const_expr(activation):
                base_col = base_col // 2
            if const_expr(transpose):
                for step in range_constexpr(tile_m * tile_n // (64 * 32)):
                    linear = lane * 32 + step * 64 * 32
                    row, col = linear // (tile_n * 4) * 4, linear // 4 % tile_n
                    values = fx.make_view(ptr + linear, fx.make_layout(32, 1)).load()
                    for i in range_constexpr(4):
                        reg = fx.make_rmem_tensor(8, fx.BFloat16)
                        reg.store(
                            Vec.from_elements(
                                [values[i + j * 4] for j in range_constexpr(8)],
                                fx.BFloat16,
                            )
                        )
                        offset = (base_row + row + i) * cols + base_col + col
                        if const_expr(mask_n):
                            offset = (base_col + col < cols).select(offset, rows * cols)
                        fx.copy(atom, reg, fx.slice(out, (None, offset >> 3)))
            else:
                for step in range_constexpr(tile_m * tile_n // (64 * 8)):
                    linear = lane * 8 + step * 64 * 8
                    row, col = linear // tile_n, linear % tile_n
                    reg = fx.make_rmem_tensor(8, fx.BFloat16)
                    reg.store(scratch_at(row, col, 8).load())
                    offset = (base_row + row) * cols + base_col + col
                    if const_expr(mask_n):
                        offset = (base_col + col < cols).select(offset, rows * cols)
                    fx.copy(atom, reg, fx.slice(out, (None, offset >> 3)))

        return store

    return factory


def compile_mxfp8_moe_gemm_8w(
    *,
    K: int,
    stage: int,
    xcd_swizzle: int = 4,
    logical_k=None,
    gather_a=False,
    tile_m=256,
    tile_n=256,
    expert_block_m=256,
    b_k=None,
    dynamic_rows=False,
    swiglu_limit=7.0,
    activation=True,
):
    """Return a grouped launcher; stage 1 fuses MiniMax's clamped SwiGLU.

    Arguments: A, packed B, BF16 C, shuffled A/B scales, expert_ids, row_map,
    padded_rows, projection_N, stream. Both padded_rows and projection_N must
    be multiples of 256. Row maps use -1 for padding; initialize padded A scale
    rows to 127 before quantizing with ``scatter_scale_topk``.
    """
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")
    return compile_mxfp8_gemm_8w(
        K=K,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        expert_block_m=expert_block_m,
        b_k=b_k,
        dynamic_rows=dynamic_rows,
        b_preshuffled=True,
        xcd_swizzle=xcd_swizzle,
        grouped=True,
        logical_k=logical_k,
        gather_a=gather_a,
        store_factory=_store_factory(
            activation=stage == 1 and activation,
            transpose=stage == 2,
            mask_n=tile_n == 512,
            swiglu_limit=swiglu_limit,
        ),
    )


def compile_mxfp8_moe_quant(
    *, K: int, gather: bool, scatter_scale_topk: int = 0, dynamic_rows=False
):
    """Fuse BF16 gather, K padding and quantization into the GEMM scale layout.

    ``row_map`` contains source row indices and -1 for expert padding. With
    ``gather=False`` normally quantizes already-sorted stage-1 output. With
    ``scatter_scale_topk > 0``, quantize source tokens once and scatter their
    scales using the inverse map ``row_map[tokens, topk]``; initialize padded
    scale rows to 127 before launching.
    """
    assert K > 0 and K % 32 == 0
    assert not (gather and scatter_scale_topk)
    kp = (K + 255) // 256 * 256
    groups = kp // 32

    @flyc.kernel(
        name=f"mxfp8_moe_quant_k{K}_gather{int(gather)}", known_block_size=[256, 1, 1]
    )
    def kernel(
        x: fx.Tensor,
        y: fx.Tensor,
        scale: fx.Tensor,
        row_map: fx.Tensor,
        rows: fx.Int32,
        valid_rows: fx.Tensor,
    ):
        inp = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(x, max_size=False), fx.make_layout(8, 1)
        )
        out = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(y, max_size=False), fx.make_layout(16, 1)
        )
        scales = fx.rocdl.make_buffer_tensor(scale, max_size=False)
        load = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
        store = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int8)
        group = fx.block_idx.x * 256 + fx.thread_idx.x
        row, kg = group // groups, group % groups
        limit = valid_rows[0] if dynamic_rows else rows
        if row < limit:
            src_row = row_map[row] if gather else row
            valid = (src_row >= 0) & (kg < K // 32)
            values = []
            amax = fx.Float32(1e-30)
            for chunk in range_constexpr(4):
                offset = valid.select(src_row * (K // 8) + kg * 4 + chunk, fx.Int32(-1))
                reg = fx.make_rmem_tensor(8, fx.BFloat16)
                fx.copy(load, fx.slice(inp, (None, offset)), reg)
                v = reg.load().to(fx.Float32)
                amax = fx.max(amax, fmath.absf(v).reduce(ReductionOp.MAX))
                values.append(v)
            bits = (amax * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(fx.Int32)
            exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(fx.Int32)
            inv = ((254 - exponent) << 23).bitcast(fx.Float32)
            # [row/32, kg/8, kg%4, row%16, kg/4%2, row/16%2]
            for slot in range_constexpr(max(1, scatter_scale_topk)):
                sr = (
                    row_map[row * scatter_scale_topk + slot]
                    if scatter_scale_topk
                    else row
                )
                scale_index = (
                    (sr // 32 * (kp // 256) + kg // 8) * 64 + kg % 4 * 16 + sr % 16
                ) * 4
                scale_index += kg // 4 % 2 * 2 + sr // 16 % 2
                scales[(sr >= 0).select(scale_index, fx.Int32(-1))] = exponent.to(
                    fx.Uint8
                )
            for half in range_constexpr(2):
                words = []
                for word in range_constexpr(4):
                    v = values[half * 2 + word // 2]
                    start = word % 2 * 4
                    packed = rocdl.cvt_pk_fp8_f32(
                        T.i32, v[start] * inv, v[start + 1] * inv, fx.Int32(0), 0
                    )
                    packed = rocdl.cvt_pk_fp8_f32(
                        T.i32, v[start + 2] * inv, v[start + 3] * inv, packed, 1
                    )
                    words.append(packed)
                reg = fx.make_rmem_tensor(16, fx.Int8)
                reg.store(Vec.from_elements(words, fx.Int32).bitcast(fx.Int8))
                fx.copy(store, reg, fx.slice(out, (None, group * 2 + half)))

    @flyc.jit
    def launch(
        x: fx.Tensor,
        y: fx.Tensor,
        scale: fx.Tensor,
        row_map: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(x, y, scale, row_map, rows, row_map).launch(
            grid=((rows * groups + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_dynamic(
        x: fx.Tensor,
        y: fx.Tensor,
        scale: fx.Tensor,
        row_map: fx.Tensor,
        rows: fx.Int32,
        valid_rows: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(x, y, scale, row_map, rows, valid_rows).launch(
            grid=((rows * groups + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch_dynamic if dynamic_rows else launch


def compile_mxfp8_moe_reduce(*, N: int, topk: int, sorted_weights=False):
    """Gather sorted down projections and sum routing-weighted top-k in FP32."""
    assert N > 0 and N % 8 == 0 and topk > 0

    @flyc.kernel(name=f"mxfp8_moe_reduce_n{N}_topk{topk}", known_block_size=[256, 1, 1])
    def kernel(
        x: fx.Tensor,
        y: fx.Tensor,
        inverse: fx.Tensor,
        weights: fx.Tensor,
        tokens: fx.Int32,
    ):
        inp = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(x, max_size=False), fx.make_layout(8, 1)
        )
        out = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(y, max_size=False), fx.make_layout(8, 1)
        )
        wr = fx.rocdl.make_buffer_tensor(weights, max_size=False)
        atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
        linear = fx.block_idx.x * 256 + fx.thread_idx.x
        token, col = linear // (N // 8), linear % (N // 8) * 8
        if token < tokens:
            acc = Vec.filled(8, 0.0, fx.Float32)
            for slot in range_constexpr(topk):
                index = token * topk + slot
                row = inverse[index]
                weight_index = row if sorted_weights else index
                weight = wr[weight_index]
                reg = fx.make_rmem_tensor(8, fx.BFloat16)
                # Missing routes use negative offsets, beyond the descriptor even for buffers >2 GiB.
                fx.copy(atom, fx.slice(inp, (None, row * (N // 8) + col // 8)), reg)
                acc = acc + reg.load().to(fx.Float32) * weight
            reg = fx.make_rmem_tensor(8, fx.BFloat16)
            reg.store(acc.to(fx.BFloat16))
            fx.copy(atom, reg, fx.slice(out, (None, linear)))

    @flyc.jit
    def launch(
        x: fx.Tensor,
        y: fx.Tensor,
        inverse: fx.Tensor,
        weights: fx.Tensor,
        tokens: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(x, y, inverse, weights, tokens).launch(
            grid=((tokens * (N // 8) + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


def compile_mxfp8_moe_unpack_routes(*, topk: int, dynamic_rows=False):
    """Convert the existing MoE sorter's packed IDs into GEMM routing maps.

    The caller supplies the actual padded row count from the sorter. Its upper
    eight bits encode the top-k slot, and its lower 24 bits encode the token.
    In dynamic mode, ``rows`` is the grid upper bound and an additional GPU
    valid-row tensor follows ``tokens`` in the launcher arguments.
    """

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel(
        packed: fx.Tensor,
        row_map: fx.Tensor,
        inverse: fx.Tensor,
        rows: fx.Int32,
        tokens: fx.Int32,
        valid_rows: fx.Tensor,
    ):
        row = fx.block_idx.x * 256 + fx.thread_idx.x
        limit = valid_rows[0] if dynamic_rows else rows
        if row < limit:
            value = packed[row]
            token, slot = value & 0xFFFFFF, (value >> 24) & 0xFF
            valid = (token < tokens) & (slot < topk)
            row_map[row] = valid.select(token, fx.Int32(-1))
            if valid:
                inverse[token * topk + slot] = row

    @flyc.jit
    def launch(
        packed: fx.Tensor,
        row_map: fx.Tensor,
        inverse: fx.Tensor,
        rows: fx.Int32,
        tokens: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(packed, row_map, inverse, rows, tokens, packed).launch(
            grid=((rows + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_dynamic(
        packed: fx.Tensor,
        row_map: fx.Tensor,
        inverse: fx.Tensor,
        rows: fx.Int32,
        tokens: fx.Int32,
        valid_rows: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(packed, row_map, inverse, rows, tokens, valid_rows).launch(
            grid=((rows + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch_dynamic if dynamic_rows else launch

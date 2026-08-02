# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE topk-reduction kernel (FlyDSL).

Sums stage2's per-slot output over the ``topk`` dimension
(``Y[t, d] = sum_k X[t, k, d]``) and optionally fuses the EP validity gather so
only slots whose expert is live are accumulated. This is the epilogue of stage2
``mode="reduce"`` (``compile_moe_gemm2(accumulate=False)``), which trades the
atomic contention of atomic-accumulate mode for a separate reduce pass. It is
shared by every dtype's reduce path, not just a16wi4.

Extracted from ``moe_gemm_2stage.py`` so it lives independently of the int4
stage1/stage2 GEMM builders in that module.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector


@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
    use_mask: bool = False,
    num_experts: int = 0,
):
    """Compile a kernel summing X[tokens, topk, model_dim] over topk into Y[tokens, model_dim].

    When ``use_mask`` is True, fuses the EP validity gather
    ``valid[t, k] = expert_mask[topk_ids[t, k]] != 0`` and accumulates only valid
    slots; ``expert_mask`` [num_experts] i32 and ``topk_ids`` [tokens, topk] i32
    are then required (and sized from ``num_experts`` at compile time).
    """
    BLOCK_SIZE = 256
    VEC_WIDTH = 8

    if dtype_str not in ("f32", "f16", "bf16"):
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    def elem_type():
        ty = T.f32 if dtype_str == "f32" else (T.f16 if dtype_str == "f16" else T.bf16)
        return ty() if callable(ty) else ty

    module_name = (
        f"moe_reduction_kernel_{'masked' if use_mask else 'plain'}"
        f"_{dtype_str}_topk{topk}_md{model_dim}"
    )

    elem_bits = 32 if dtype_str == "f32" else 16
    elem_bytes_c = elem_bits // 8
    copy_vec_width = 128 // elem_bits  # 8 for f16/bf16, 4 for f32
    n_sub = VEC_WIDTH // copy_vec_width  # 1 for f16/bf16, 2 for f32

    @flyc.kernel(name=module_name)
    def moe_reduction_kernel(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        i32_m_tokens: fx.Int32,
    ):
        m_tokens = fx.Index(i32_m_tokens)
        c_model_dim = fx.Index(model_dim)

        def _ptr_buffer_resource_off(ptr, num_records_bytes, byte_off_i64=None):
            # Build a buffer resource from a raw pointer, folding an optional
            # per-WG i64 byte offset into the descriptor's 48-bit base.
            addr_i64 = arith.index_cast(T.i64, fx.ptrtoint(ptr))
            if byte_off_i64 is not None:
                addr_i64 = addr_i64 + byte_off_i64
            return buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=num_records_bytes
            )

        token_idx = gpu.block_id("x")
        tile_idx = gpu.block_id("y")
        tid = gpu.thread_id("x")

        # 64-bit base-offset fold: X is [m_tokens, topk, model_dim] and total bytes
        # can exceed 4 GiB (e.g. 131072*6*4096*2 = 6 GiB), overflowing the i32
        # voffset used by buffer_load. Fold the per-WG token byte offset into the
        # descriptor's 48-bit base (i64) so in-kernel voffsets address one token slab.
        x_slab_nbytes = fx.Index(topk) * c_model_dim * fx.Index(elem_bytes_c)
        y_slab_nbytes = c_model_dim * fx.Index(elem_bytes_c)
        x_rsrc = _ptr_buffer_resource_off(
            X, fx.Int64(x_slab_nbytes), fx.Int64(token_idx * x_slab_nbytes)
        )
        y_rsrc = _ptr_buffer_resource_off(
            Y, fx.Int64(y_slab_nbytes), fx.Int64(token_idx * y_slab_nbytes)
        )

        if const_expr(use_mask):
            tk_slab_nbytes = fx.Index(topk) * fx.Index(4)
            topk_ids_rsrc = _ptr_buffer_resource_off(
                topk_ids, fx.Int64(tk_slab_nbytes), fx.Int64(token_idx * tk_slab_nbytes)
            )
            # expert_mask: [num_experts] i32, sized exactly from the compile-time count.
            expert_mask_rsrc = _ptr_buffer_resource_off(
                expert_mask, fx.Int64(fx.Index(num_experts * 4)), None
            )

        c_vecw = fx.Index(VEC_WIDTH)
        col_base = tile_idx * fx.Index(BLOCK_SIZE * VEC_WIDTH) + tid * c_vecw

        # fx.Index is unsigned, so every bounds compare below lowers to ult.
        if (token_idx < m_tokens) & (col_base < c_model_dim):
            if col_base + c_vecw <= c_model_dim:
                # Fast path: full 128b vector in-bounds. buffer_load moves
                # copy_vec_width elems (8 bf16/f16, 4 f32) per op; n_sub ops
                # cover the VEC_WIDTH stride.
                vec_type_c = T.vec(copy_vec_width, T.f32)
                vec_type_e = T.vec(copy_vec_width, elem_type())
                acc_vecs = [
                    vector.broadcast(vec_type_c, fx.Float32(0.0).ir_value())
                    for _ in range(n_sub)
                ]

                for k in range_constexpr(topk):
                    # Within one token slab k indexes topk with stride model_dim.
                    k_off_elems = fx.Index(k) * c_model_dim + col_base

                    if const_expr(use_mask):
                        # topk_ids_rsrc is already shifted by token_idx*topk.
                        eid_i32 = buffer_ops.buffer_load(
                            topk_ids_rsrc,
                            fx.Int32(fx.Index(k)),
                            vec_width=1,
                            dtype=T.i32,
                        )
                        valid_i32 = buffer_ops.buffer_load(
                            expert_mask_rsrc, eid_i32, vec_width=1, dtype=T.i32
                        )
                        mv_ok = valid_i32 != fx.Int32(0)

                    for si in range_constexpr(n_sub):
                        off_i32 = fx.Int32(k_off_elems + fx.Index(si * copy_vec_width))
                        vec_e = buffer_ops.buffer_load(
                            x_rsrc, off_i32, vec_width=copy_vec_width, dtype=elem_type()
                        )
                        if const_expr(use_mask):
                            zero_e = vector.broadcast(
                                vec_type_e, arith.constant(0.0, type=elem_type())
                            )
                            vec_e = mv_ok.select(vec_e, zero_e)
                        if const_expr(elem_bits < 32):
                            vec_e = vec_e.extf(vec_type_c)
                        acc_vecs[si] = acc_vecs[si] + vec_e

                for si in range_constexpr(n_sub):
                    out_vec = acc_vecs[si]
                    if const_expr(elem_bits < 32):
                        out_vec = out_vec.truncf(vec_type_e)
                    y_off_i32 = fx.Int32(col_base + fx.Index(si * copy_vec_width))
                    buffer_ops.buffer_store(out_vec, y_rsrc, y_off_i32)
            else:
                # Tail path: scalar load/store per lane. Offsets are slab-local
                # since token_idx is folded into the base ptr.
                for lane in range_constexpr(VEC_WIDTH):
                    col = col_base + fx.Index(lane)
                    if col < c_model_dim:
                        a = arith.constant(0.0, type=T.f32)
                        for k in range_constexpr(topk):
                            x_idx_i32 = fx.Int32(fx.Index(k) * c_model_dim + col)
                            if const_expr(use_mask):
                                eid_i32 = buffer_ops.buffer_load(
                                    topk_ids_rsrc,
                                    fx.Int32(fx.Index(k)),
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                                valid_i32 = buffer_ops.buffer_load(
                                    expert_mask_rsrc, eid_i32, vec_width=1, dtype=T.i32
                                )
                                v = (valid_i32 != fx.Int32(0)).select(
                                    buffer_ops.buffer_load(
                                        x_rsrc,
                                        x_idx_i32,
                                        vec_width=1,
                                        dtype=elem_type(),
                                    ),
                                    arith.constant(0.0, type=elem_type()),
                                )
                            else:
                                v = buffer_ops.buffer_load(
                                    x_rsrc, x_idx_i32, vec_width=1, dtype=elem_type()
                                )
                            if const_expr(elem_bits < 32):
                                v = v.extf(T.f32)
                            a = a + v

                        if const_expr(elem_bits < 32):
                            a = a.truncf(elem_type())
                        buffer_ops.buffer_store(a, y_rsrc, fx.Int32(col))

    tile_size = BLOCK_SIZE * VEC_WIDTH
    gy_static = (model_dim + tile_size - 1) // tile_size

    @flyc.jit
    def launch_moe_reduction(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        moe_reduction_kernel(X, Y, expert_mask, topk_ids, i32_m_tokens).launch(
            grid=(fx.Index(i32_m_tokens), gy_static, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction

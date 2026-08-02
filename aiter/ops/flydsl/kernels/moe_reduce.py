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
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector


@contextmanager
def _if_then(if_op):
    """SCF IfOp then-region helper (compat across old/new Python APIs)."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    """SCF IfOp else-region helper (compat across old/new Python APIs)."""
    if getattr(if_op, "else_block", None) is None:
        raise RuntimeError("IfOp has no else block")
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


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

    if dtype_str == "f32":
        elem_type_tag = "f32"
    elif dtype_str == "f16":
        elem_type_tag = "f16"
    elif dtype_str == "bf16":
        elem_type_tag = "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    def compute_type():
        return T.f32

    def i32_type():
        return T.i32

    def elem_type():
        ty = (
            T.f32
            if elem_type_tag == "f32"
            else (T.f16 if elem_type_tag == "f16" else T.bf16)
        )
        return ty() if callable(ty) else ty

    module_name = (
        f"moe_reduction_kernel_{'masked' if use_mask else 'plain'}"
        f"_{dtype_str}_topk{topk}_md{model_dim}"
    )

    elem_bytes_c = (32 if dtype_str == "f32" else 16) // 8

    @flyc.kernel(name=module_name)
    def moe_reduction_kernel(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        i32_m_tokens: fx.Int32,
    ):
        m_tokens = fx.Index(i32_m_tokens)
        c_topk = fx.Index(topk)
        c_model_dim = fx.Index(model_dim)
        elem_bits = 32 if dtype_str == "f32" else 16
        copy_vec_width = 128 // elem_bits  # 8 for f16/bf16, 4 for f32
        n_sub = VEC_WIDTH // copy_vec_width  # 1 for f16/bf16, 2 for f32

        def _ptr_buffer_resource_off(ptr, num_records_bytes, byte_off_i64=None):
            # Build a buffer resource from a raw pointer, folding an optional
            # per-WG i64 byte offset into the descriptor's 48-bit base.
            addr = fx.ptrtoint(ptr)
            addr_i64 = arith.index_cast(T.i64, addr)
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
        slab_elems_x = c_topk * c_model_dim
        x_slab_nbytes = slab_elems_x * fx.Index(elem_bytes_c)
        y_slab_nbytes = c_model_dim * fx.Index(elem_bytes_c)
        x_base_off_i64 = fx.Int64(token_idx * x_slab_nbytes)
        y_base_off_i64 = fx.Int64(token_idx * c_model_dim * fx.Index(elem_bytes_c))

        x_rsrc = _ptr_buffer_resource_off(X, fx.Int64(x_slab_nbytes), x_base_off_i64)
        y_rsrc = _ptr_buffer_resource_off(Y, fx.Int64(y_slab_nbytes), y_base_off_i64)

        if const_expr(use_mask):
            tk_slab_nbytes = c_topk * fx.Index(4)
            tk_base_off_i64 = fx.Int64(token_idx * tk_slab_nbytes)
            topk_ids_rsrc = _ptr_buffer_resource_off(
                topk_ids, fx.Int64(tk_slab_nbytes), tk_base_off_i64
            )
            # expert_mask: [num_experts] i32, sized exactly from the compile-time count.
            em_nbytes = fx.Index(num_experts * 4)
            expert_mask_rsrc = _ptr_buffer_resource_off(
                expert_mask, fx.Int64(em_nbytes), None
            )

        # Index is unsigned, so every comparison below lowers to ult.
        tok_ok = token_idx < m_tokens
        _if_tok = scf.IfOp(tok_ok)
        with _if_then(_if_tok):
            tile_cols = BLOCK_SIZE * VEC_WIDTH
            c_tile_cols = fx.Index(tile_cols)
            c_vecw = fx.Index(VEC_WIDTH)

            col_base = tile_idx * c_tile_cols + tid * c_vecw

            col_ok = col_base < c_model_dim
            _if_col = scf.IfOp(col_ok)
            with _if_then(_if_col):
                end_ok = col_base + c_vecw <= c_model_dim
                _if_full = scf.IfOp(end_ok, has_else=True)
                with _if_then(_if_full):
                    # Fast path: full 128b vector in-bounds. buffer_load moves
                    # copy_vec_width elems (8 bf16/f16, 4 f32) per op; n_sub ops
                    # cover the VEC_WIDTH stride.
                    vec_type_c = T.vec(copy_vec_width, compute_type())
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
                            tk_idx_i32 = fx.Int32(fx.Index(k))
                            eid_i32 = buffer_ops.buffer_load(
                                topk_ids_rsrc, tk_idx_i32, vec_width=1, dtype=i32_type()
                            )
                            valid_i32 = buffer_ops.buffer_load(
                                expert_mask_rsrc, eid_i32, vec_width=1, dtype=i32_type()
                            )
                            mv_ok = valid_i32 != fx.Int32(0)

                        for si in range_constexpr(n_sub):
                            off_elems_i32 = fx.Int32(
                                k_off_elems + fx.Index(si * copy_vec_width)
                            )
                            vec_e = buffer_ops.buffer_load(
                                x_rsrc,
                                off_elems_i32,
                                vec_width=copy_vec_width,
                                dtype=elem_type(),
                            )

                            if const_expr(use_mask):
                                zero_e = vector.broadcast(
                                    vec_type_e, arith.constant(0.0, type=elem_type())
                                )
                                vec_e = mv_ok.select(vec_e, zero_e)

                            if const_expr(elem_bits < 32):
                                vec_c = vec_e.extf(vec_type_c)
                            else:
                                vec_c = vec_e
                            acc_vecs[si] = acc_vecs[si] + vec_c

                    for si in range_constexpr(n_sub):
                        out_vec = acc_vecs[si]
                        if const_expr(elem_bits < 32):
                            out_vec = out_vec.truncf(vec_type_e)
                        y_off_elems_i32 = fx.Int32(
                            col_base + fx.Index(si * copy_vec_width)
                        )
                        buffer_ops.buffer_store(out_vec, y_rsrc, y_off_elems_i32)

                with _if_else(_if_full):
                    # Tail path: scalar load/store per lane. Offsets are slab-local
                    # since token_idx is folded into the base ptr.
                    for lane in range_constexpr(VEC_WIDTH):
                        col = col_base + fx.Index(lane)
                        lane_ok = col < c_model_dim
                        _if_lane = scf.IfOp(lane_ok)
                        with _if_then(_if_lane):
                            a = arith.constant(0.0, type=compute_type())
                            for k in range_constexpr(topk):
                                k_idx = fx.Index(k)
                                x_idx_i32 = fx.Int32(k_idx * c_model_dim + col)
                                if const_expr(use_mask):
                                    tk_idx_i32 = fx.Int32(k_idx)
                                    eid_i32 = buffer_ops.buffer_load(
                                        topk_ids_rsrc,
                                        tk_idx_i32,
                                        vec_width=1,
                                        dtype=i32_type(),
                                    )
                                    valid_i32 = buffer_ops.buffer_load(
                                        expert_mask_rsrc,
                                        eid_i32,
                                        vec_width=1,
                                        dtype=i32_type(),
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
                                        x_rsrc,
                                        x_idx_i32,
                                        vec_width=1,
                                        dtype=elem_type(),
                                    )
                                if const_expr(dtype_str in ("f16", "bf16")):
                                    v = v.extf(compute_type())
                                a = a + v

                            out = a
                            if const_expr(dtype_str in ("f16", "bf16")):
                                out = out.truncf(elem_type())
                            y_idx_i32 = fx.Int32(col)
                            buffer_ops.buffer_store(out, y_rsrc, y_idx_i32)

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
        gx = fx.Index(i32_m_tokens)
        moe_reduction_kernel(X, Y, expert_mask, topk_ids, i32_m_tokens).launch(
            grid=(gx, gy_static, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction

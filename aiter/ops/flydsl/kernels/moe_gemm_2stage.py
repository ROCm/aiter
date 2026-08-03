# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM stage1/stage2 kernel implementations (FlyDSL MFMA FP8).

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from aiter.ops.flydsl.kernels import buffer_ops, vector

try:
    from flydsl.runtime.device import (
        bf16_global_atomics_arch_description,
        supports_bf16_global_atomics,
    )
except ImportError:
    # Backward compatibility for runtime.device versions that only expose get_rocm_arch.
    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))

    def bf16_global_atomics_arch_description() -> str:
        return "gfx94+/gfx95+/gfx12+"


from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr.typing import T


@contextmanager
def _if_then(if_op):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    """Compat helper for SCF IfOp else-region across old/new Python APIs."""
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
    """Compile a reduction kernel that sums over the topk dimension.

    Input:  X [tokens, topk, model_dim]
            expert_mask [num_experts] i32 (optional, if use_mask=True)
            topk_ids   [tokens, topk] i32 (optional, if use_mask=True)
    Output: Y [tokens, model_dim]

    This kernel performs: Y[t, d] = sum_k(X[t, k, d]) for all t, d.
    When use_mask=True, the kernel fuses the EP validity gather:
        valid[t, k] = expert_mask[topk_ids[t, k]] != 0
    and only accumulates X[t, k, :] when valid[t, k] is true.
    Used in conjunction with compile_moe_gemm2(accumulate=False) to avoid atomic contention.
    """
    get_hip_arch()
    ir.ShapedType.get_dynamic_size()

    # Kernel Config
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

    def i8_type():
        return T.i8

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

    if True:

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
                # Build a buffer resource from a raw pointer, optionally folding
                # a per-WG i64 byte offset into the descriptor's 48-bit base.
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

            # ── 64-bit base-offset folding ─────────────────────────────────
            # X is [m_tokens, topk, model_dim]; total bytes can exceed 4 GiB
            # for large batches (e.g. 131072 * 6 * 4096 * 2 = 6 GiB), which
            # overflows the i32 voffset used by buffer_load. To stay i32-safe,
            # fold the per-WG token byte offset into the descriptor's 48-bit
            # base address (computed in i64). The in-kernel voffsets then only
            # need to address one token's slab.
            slab_elems_x = c_topk * c_model_dim
            x_slab_nbytes = slab_elems_x * fx.Index(elem_bytes_c)
            y_slab_nbytes = c_model_dim * fx.Index(elem_bytes_c)
            x_base_off_i64 = fx.Int64(token_idx * x_slab_nbytes)
            y_base_off_i64 = fx.Int64(token_idx * c_model_dim * fx.Index(elem_bytes_c))

            x_rsrc = _ptr_buffer_resource_off(
                X, fx.Int64(x_slab_nbytes), x_base_off_i64
            )
            y_rsrc = _ptr_buffer_resource_off(
                Y, fx.Int64(y_slab_nbytes), y_base_off_i64
            )

            if const_expr(use_mask):
                tk_slab_nbytes = c_topk * fx.Index(4)
                tk_base_off_i64 = fx.Int64(token_idx * tk_slab_nbytes)
                topk_ids_rsrc = _ptr_buffer_resource_off(
                    topk_ids, fx.Int64(tk_slab_nbytes), tk_base_off_i64
                )
                # expert_mask: [num_experts] i32. Caller supplies num_experts
                # at compile time so we can size the descriptor exactly.
                em_nbytes = fx.Index(num_experts * 4)
                expert_mask_rsrc = _ptr_buffer_resource_off(
                    expert_mask, fx.Int64(em_nbytes), None
                )

            # Guard: token in range (Index is unsigned → auto ult)
            tok_ok = token_idx < m_tokens
            _if_tok = scf.IfOp(tok_ok)
            with _if_then(_if_tok):
                tile_cols = BLOCK_SIZE * VEC_WIDTH
                c_tile_cols = fx.Index(tile_cols)
                c_vecw = fx.Index(VEC_WIDTH)

                col_base = tile_idx * c_tile_cols + tid * c_vecw

                # Guard: any work in bounds (Index < → ult)
                col_ok = col_base < c_model_dim
                _if_col = scf.IfOp(col_ok)
                with _if_then(_if_col):
                    # Fast path: full vector in-bounds (Index <= → ule)
                    end_ok = col_base + c_vecw <= c_model_dim
                    _if_full = scf.IfOp(end_ok, has_else=True)
                    with _if_then(_if_full):
                        # ── Vector path via direct buffer_load ──
                        # Use buffer_load with vec_width=copy_vec_width
                        # (8 elems for bf16/f16 = 128b; 4 elems for f32 = 128b).
                        # n_sub iterations cover the full VEC_WIDTH stride.
                        vec_type_c = T.vec(copy_vec_width, compute_type())
                        vec_type_e = T.vec(copy_vec_width, elem_type())

                        acc_vecs = [
                            vector.broadcast(vec_type_c, fx.Float32(0.0).ir_value())
                            for _ in range(n_sub)
                        ]

                        for k in range_constexpr(topk):
                            # X slab base for this (token, k) — within one token's
                            # slab, k indexes the topk dim with stride model_dim.
                            # elem offset = k*model_dim + col_base + si*copy_vec_width
                            k_off_elems = fx.Index(k) * c_model_dim + col_base

                            if const_expr(use_mask):
                                # Fused EP gather: valid = expert_mask[topk_ids[token, k]] != 0
                                # topk_ids_rsrc is already shifted by token_idx*topk
                                tk_idx_i32 = fx.Int32(fx.Index(k))
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
                                        vec_type_e,
                                        arith.constant(0.0, type=elem_type()),
                                    )
                                    vec_e = mv_ok.select(vec_e, zero_e)

                                if const_expr(elem_bits < 32):
                                    vec_c = vec_e.extf(vec_type_c)
                                else:
                                    vec_c = vec_e
                                acc_vecs[si] = acc_vecs[si] + vec_c

                        # ── Store results ──
                        for si in range_constexpr(n_sub):
                            out_vec = acc_vecs[si]
                            if const_expr(elem_bits < 32):
                                out_vec = out_vec.truncf(vec_type_e)
                            y_off_elems_i32 = fx.Int32(
                                col_base + fx.Index(si * copy_vec_width)
                            )
                            buffer_ops.buffer_store(out_vec, y_rsrc, y_off_elems_i32)

                    with _if_else(_if_full):
                        # Tail path: scalar load/store per lane. All offsets
                        # are now slab-local (token_idx folded into base ptr).
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

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
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


# MoE GEMM2 Execution Modes

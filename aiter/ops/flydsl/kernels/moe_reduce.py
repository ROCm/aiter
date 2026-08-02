# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE topk-reduction kernel (FlyDSL, layout API).

``Y[t, d] = sum_k X[t, k, d]``, optionally gated by the EP validity mask
(``valid[t,k] = expert_mask[topk_ids[t,k]] != 0``). Epilogue of stage2
``mode="reduce"``, shared by every dtype's reduce path. Extracted from
``moe_gemm_2stage.py``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

_NUMERIC = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}


@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *, topk, model_dim, dtype_str="f16", use_mask=False, num_experts=0
):
    if dtype_str not in _NUMERIC:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    elem_cls = _NUMERIC[dtype_str]
    elem_bits = 32 if dtype_str == "f32" else 16
    elem_bytes = elem_bits // 8
    is_16b = elem_bits < 32
    BLOCK, V = 256, 128 // elem_bits  # V elems per 128b copy: 4 (f32), 8 (16b)
    TILE = BLOCK * V
    module_name = (
        f"moe_reduction_kernel_{'masked' if use_mask else 'plain'}"
        f"_{dtype_str}_topk{topk}_md{model_dim}"
    )

    @flyc.kernel(name=module_name)
    def moe_reduction_kernel(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        i32_m_tokens: fx.Int32,
    ):
        token, tile, tid = gpu.block_id("x"), gpu.block_id("y"), gpu.thread_id("x")
        vec_f32, vec_e = T.vec(V, T.f32), T.vec(V, elem_cls.ir_type)

        def _view(ptr_i64, elem, nbytes):
            pt = fx.PointerType.get(
                elem.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=elem.width // 8,
            )
            view = fx.Tensor(
                fx.make_view(fx.inttoptr(pt, ptr_i64), fx.make_layout(model_dim, 1))
            )
            return fx.rocdl.make_buffer_tensor(view, num_records_bytes=fx.Int64(nbytes))

        # Fold the per-token byte offset into each base ptr so in-kernel voffsets
        # stay i32-safe even when X exceeds 4 GiB.
        eb, md, tok64 = fx.Int64(elem_bytes), fx.Int64(model_dim), fx.Int64(token)
        xbuf = _view(
            fx.Int64(ptrtoint(X)) + tok64 * fx.Int64(topk) * md * eb,
            elem_cls,
            topk * model_dim * elem_bytes,
        )
        ybuf = _view(
            fx.Int64(ptrtoint(Y)) + tok64 * md * eb, elem_cls, model_dim * elem_bytes
        )
        if const_expr(use_mask):
            i32pt = fx.PointerType.get(
                T.i32, address_space=fx.AddressSpace.Global, alignment=4
            )
            tk_ptr = fx.inttoptr(
                i32pt, fx.Int64(ptrtoint(topk_ids)) + tok64 * fx.Int64(topk * 4)
            )
            em_ptr = fx.inttoptr(i32pt, fx.Int64(ptrtoint(expert_mask)))

        copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_cls)

        def _tile_thread(buf):  # buf[model_dim] -> this thread's V contiguous elems
            t = fx.slice(fx.logical_divide(buf, fx.make_layout(TILE, 1)), (None, tile))
            return fx.slice(fx.logical_divide(t, fx.make_layout(V, 1)), (None, tid))

        def _reduce_tile():
            # topk rows share one per-thread voffset via a uniform scalar
            # soffset=k*model_dim, so the loads issue back-to-back.
            src = _tile_thread(xbuf)
            frags = [fx.make_rmem_tensor(V, elem_cls) for _ in range_constexpr(topk)]
            for k in range_constexpr(topk):
                fx.copy(copy, src, frags[k], soffset=fx.Int32(k * model_dim))
            acc = fx.Vector.filled(V, 0.0, fx.Float32)
            for k in range_constexpr(topk):
                vk = fx.Vector(fx.memref_load_vec(frags[k]))
                if const_expr(use_mask):
                    vk = (em_ptr[tk_ptr[k]] != fx.Int32(0)).select(
                        vk, fx.Vector.filled(V, 0.0, elem_cls)
                    )
                acc = acc + (vk.extf(vec_f32) if is_16b else vk)
            ofrag = fx.make_rmem_tensor(V, elem_cls)
            fx.memref_store_vec(acc.truncf(vec_e) if is_16b else acc, ofrag)
            fx.copy_atom_call(copy, ofrag, _tile_thread(ybuf))

        # Skip threads whose column group starts past model_dim (their loads would
        # read the next row -- in-descriptor, wasted BW); only needed when TILE ∤ md.
        if const_expr(model_dim % TILE != 0):
            if fx.Int32(tile) * fx.Int32(TILE) + fx.Int32(tid) * fx.Int32(V) < fx.Int32(
                model_dim
            ):
                _reduce_tile()
        else:
            _reduce_tile()

    gy = (model_dim + TILE - 1) // TILE

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
            grid=(fx.Int64(i32_m_tokens), gy, 1), block=(BLOCK, 1, 1), stream=stream
        )

    return launch_moe_reduction

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE topk-reduction kernel (FlyDSL, layout API).

``Y[t, d] = sum_k X[t, k, d]``, optionally gated by the EP validity mask
(``valid[t,k] = expert_mask[topk_ids[t,k]] != 0``). Epilogue of stage2
``mode="reduce"``, shared by every dtype's reduce path. Extracted from
``moe_gemm_2stage.py``. Launch via ``moe_reduction`` (a ``@flyc.jit``); its
compile-time params are ``Constexpr`` so flyc specializes per shape/dtype.

``dtype_str="fp8"`` reduces MXFP8 route-out rows (a flat uint8 buffer of
``[model_dim fp8 bytes | model_dim/8 e8m0 scale bytes]`` per row): each fp8
value is scaled by its e8m0 microscale, accumulated in f32 and written to
``out_dtype_str`` (bf16/f16). The dense (f32/f16/bf16) path reduces a
contiguous ``X[tokens, topk, model_dim]`` tensor.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

_NUMERIC = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}
BLOCK = 256
FP8_VEC = 8  # fp8 values per 64b buffer load (also the store granularity)


@flyc.kernel
def moe_reduction_kernel(
    X: fx.Pointer,
    Y: fx.Pointer,
    expert_mask: fx.Pointer,
    topk_ids: fx.Pointer,
    i32_m_tokens: fx.Int32,
    topk: fx.Constexpr[int],
    model_dim: fx.Constexpr[int],
    dtype_str: fx.Constexpr[str],
    use_mask: fx.Constexpr[bool],
    num_experts: fx.Constexpr[int],
    out_dtype_str: fx.Constexpr[str],
):
    if const_expr(dtype_str == "fp8"):
        # -- MXFP8 route-out path: rows are [N fp8 bytes | N/8 e8m0] uint8. --
        out_tag = out_dtype_str or "bf16"
        out_numeric = fx.Float16 if out_tag == "f16" else fx.BFloat16
        fp8_row_bytes_in = model_dim + model_dim // 8
        elem_bytes_c = out_numeric.width // 8

        m_tokens = fx.Int64(i32_m_tokens)
        c_topk = fx.Int64(topk)
        c_model_dim = fx.Int64(model_dim)

        def _buffer_tensor(ptr, numeric, num_elems, num_bytes, byte_off_i64):
            addr_i64 = fx.Int64(fx.ptrtoint(ptr)) + byte_off_i64
            ptr_ty = fx.PointerType.get(
                numeric.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=numeric.width // 8,
            )
            base = fx.inttoptr(ptr_ty, addr_i64)
            view = fx.make_view(base, fx.make_layout(num_elems, 1))
            return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_bytes)

        def _tile(tensor, elem_offset, width):
            ptr = fx.add_offset(fx.get_iter(tensor), elem_offset)
            return fx.make_view(ptr, fx.make_layout(width, 1))

        token_idx = fx.Int64(gpu.block_id("x"))
        tile_idx = fx.Int64(gpu.block_id("y"))
        tid = fx.Int64(gpu.thread_id("x"))

        x_slab_nbytes = c_topk * fx.Int64(fp8_row_bytes_in)
        y_slab_nbytes = c_model_dim * fx.Int64(elem_bytes_c)
        x_buf = _buffer_tensor(
            X, fx.Int8, x_slab_nbytes, x_slab_nbytes, token_idx * x_slab_nbytes
        )
        y_buf = _buffer_tensor(
            Y,
            out_numeric,
            c_model_dim,
            y_slab_nbytes,
            token_idx * c_model_dim * fx.Int64(elem_bytes_c),
        )

        load_fp8x8 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int8)
        load_i8_atom = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Int8)
        load_i32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        store_out8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), out_numeric)
        store_out1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_numeric)

        def _load_i8(elem_offset):
            frag = fx.make_rmem_tensor(1, fx.Int8)
            fx.copy(load_i8_atom, _tile(x_buf, elem_offset, 1), frag)
            return fx.Vector(frag.load())[0]

        if const_expr(use_mask):
            tk_slab_nbytes = c_topk * fx.Int64(4)
            topk_ids_buf = _buffer_tensor(
                topk_ids, fx.Int32, c_topk, tk_slab_nbytes, token_idx * tk_slab_nbytes
            )
            expert_mask_buf = _buffer_tensor(
                expert_mask,
                fx.Int32,
                fx.Int64(num_experts),
                fx.Int64(num_experts * 4),
                fx.Int64(0),
            )
        else:
            topk_ids_buf = None
            expert_mask_buf = None

        def load_valid_mask(k):
            eid_frag = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy(load_i32_atom, _tile(topk_ids_buf, fx.Int32(k), 1), eid_frag)
            eid = fx.Vector(eid_frag.load())[0]
            valid_frag = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy(load_i32_atom, _tile(expert_mask_buf, eid, 1), valid_frag)
            return fx.Vector(valid_frag.load())[0] != fx.Int32(0)

        c_vecw = fx.Int64(FP8_VEC)
        col_base = tile_idx * fx.Int64(BLOCK * FP8_VEC) + tid * c_vecw

        # Guard: token in range, and this thread's column window overlaps the row.
        if token_idx < m_tokens:
            if col_base < c_model_dim:
                c_row_bytes_in = fx.Int64(fp8_row_bytes_in)
                c_scale_base = fx.Int64(model_dim)
                vec2_f32 = T.vec(2, T.f32)

                def load_scale_f32(scale_off):
                    """Decode one e8m0 byte into an f32 microscale (2^(e-127))."""
                    e8m0_i32 = fx.Uint32(fx.Uint8(_load_i8(scale_off)))
                    return (e8m0_i32 << fx.Uint32(23)).bitcast(fx.Float32)

                # Fast path: the full FP8_VEC window is in-bounds.
                if col_base + c_vecw <= c_model_dim:
                    acc = [fx.Float32(0.0) for _ in range(FP8_VEC)]
                    scale_col = col_base // c_vecw
                    for k in range_constexpr(topk):
                        k_row_base = fx.Int64(k) * c_row_bytes_in
                        mv_ok = load_valid_mask(k) if const_expr(use_mask) else None
                        # Preserve the dword-aligned i32-word address calculation
                        # before unpacking the 8 bytes into four f32 pairs.
                        val_byte_offset = (
                            (k_row_base + col_base) // fx.Int64(4)
                        ) * fx.Int64(4)
                        fp8_frag = fx.make_rmem_tensor(FP8_VEC, fx.Int8)
                        fx.copy(
                            load_fp8x8, _tile(x_buf, val_byte_offset, FP8_VEC), fp8_frag
                        )
                        w = fx.Vector(fp8_frag.load()).bitcast(fx.Int32)
                        scale_f32 = load_scale_f32(k_row_base + c_scale_base + scale_col)
                        words = (w[0], w[0], w[1], w[1])
                        for pi in range_constexpr(4):
                            pair = fx.Vector(
                                fx.rocdl.cvt_pk_f32_fp8(vec2_f32, words[pi], bool(pi & 1))
                            )
                            val0 = pair[0] * scale_f32
                            val1 = pair[1] * scale_f32
                            if const_expr(use_mask):
                                val0 = mv_ok.select(val0, fx.Float32(0.0))
                                val1 = mv_ok.select(val1, fx.Float32(0.0))
                            acc[2 * pi] = acc[2 * pi] + val0
                            acc[2 * pi + 1] = acc[2 * pi + 1] + val1
                    out_frag = fx.make_rmem_tensor(FP8_VEC, out_numeric)
                    out_frag.store(
                        fx.Vector.from_elements(
                            [acc[i].to(out_numeric) for i in range(FP8_VEC)], out_numeric
                        )
                    )
                    fx.copy(store_out8, out_frag, _tile(y_buf, col_base, FP8_VEC))
                else:
                    # Tail path: per-lane scalar accumulate for the partial window.
                    for lane in range_constexpr(FP8_VEC):
                        col = col_base + fx.Int64(lane)
                        if col < c_model_dim:
                            a = fx.Float32(0.0)
                            scale_col = col // c_vecw
                            for k in range_constexpr(topk):
                                k_row_base = fx.Int64(k) * c_row_bytes_in
                                mv_ok = (
                                    load_valid_mask(k) if const_expr(use_mask) else None
                                )
                                b_i32 = fx.Uint32(fx.Uint8(_load_i8(k_row_base + col)))
                                scale_f32 = load_scale_f32(
                                    k_row_base + c_scale_base + scale_col
                                )
                                pair = fx.Vector(
                                    fx.rocdl.cvt_pk_f32_fp8(vec2_f32, b_i32, False)
                                )
                                val = pair[0] * scale_f32
                                if const_expr(use_mask):
                                    val = mv_ok.select(val, fx.Float32(0.0))
                                a = a + val
                            out_frag = fx.make_rmem_tensor(1, out_numeric)
                            out_frag.store(
                                fx.Vector.from_elements([a.to(out_numeric)], out_numeric)
                            )
                            fx.copy(store_out1, out_frag, _tile(y_buf, col, 1))
        return

    elem_cls = _NUMERIC[dtype_str]
    elem_bits = 32 if dtype_str == "f32" else 16
    elem_bytes = elem_bits // 8
    is_16b = elem_bits < 32
    V = 128 // elem_bits  # V elems per 128b copy: 4 (f32), 8 (16b)
    TILE = BLOCK * V

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


@flyc.jit
def moe_reduction(
    X: fx.Pointer,
    Y: fx.Pointer,
    expert_mask: fx.Pointer,
    topk_ids: fx.Pointer,
    i32_m_tokens: fx.Int32,
    stream: fx.Stream,
    topk: fx.Constexpr[int],
    model_dim: fx.Constexpr[int],
    dtype_str: fx.Constexpr[str],
    use_mask: fx.Constexpr[bool],
    num_experts: fx.Constexpr[int],
    out_dtype_str: fx.Constexpr[str],
):
    V = FP8_VEC if dtype_str == "fp8" else 128 // (32 if dtype_str == "f32" else 16)
    gy = (model_dim + BLOCK * V - 1) // (BLOCK * V)
    moe_reduction_kernel(
        X,
        Y,
        expert_mask,
        topk_ids,
        i32_m_tokens,
        topk,
        model_dim,
        dtype_str,
        use_mask,
        num_experts,
        out_dtype_str,
    ).launch(grid=(fx.Int64(i32_m_tokens), gy, 1), block=(BLOCK, 1, 1), stream=stream)

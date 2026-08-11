# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""MoE topk-reduction kernel (FlyDSL, layout API).

``Y[t, d] = sum_k X[t, k, d]``, optionally gated by the EP validity mask
(``valid[t,k] = expert_mask[topk_ids[t,k]] != 0``). Epilogue of stage2
``mode="reduce"``, shared by every dtype's reduce path. Extracted from
``moe_gemm_2stage.py``. Build a per-shape launcher with ``compile_moe_reduction``
(cached); the kernel's compile-time params are ``Constexpr`` so flyc specializes
per shape/dtype.

``dtype_str="fp8"`` reduces MXFP8 route-out rows (a flat uint8 buffer of
``[model_dim fp8 bytes | model_dim/8 e8m0 scale bytes]`` per row): each fp8
value is scaled by its e8m0 microscale, accumulated in f32 and written to
``out_dtype_str`` (bf16/f16). The dense (f32/f16/bf16) path reduces a
contiguous ``X[tokens, topk, model_dim]`` tensor.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

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
    model_dim_pad: fx.Constexpr[int],
    dtype_str: fx.Constexpr[str],
    use_mask: fx.Constexpr[bool],
    num_experts: fx.Constexpr[int],
    out_dtype_str: fx.Constexpr[str],
):
    # One tiled-copy reduce for every dtype. Dense (f16/bf16/f32) loads V elems
    # and extends to f32; fp8 route-out loads 8 fp8 bytes + their e8m0 microscale
    # and decodes to f32. Both then run the same masked f32 topk-accumulate
    # (uniform soffset = k*row_stride) and truncating store. row_stride differs:
    # an fp8 row is padded with its N/8 scale bytes ([N fp8 | N/8 e8m0]).
    is_fp8 = dtype_str == "fp8"
    if const_expr(is_fp8):
        in_elem, in_bytes, V = fx.Int8, 1, FP8_VEC
        row_stride = model_dim + model_dim // 8
        out_numeric = fx.Float16 if (out_dtype_str or "bf16") == "f16" else fx.BFloat16
        load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int8)
    else:
        in_elem = (
            fx.Float32
            if dtype_str == "f32"
            else (fx.Float16 if dtype_str == "f16" else fx.BFloat16)
        )
        in_bytes = 4 if dtype_str == "f32" else 2
        row_stride, out_numeric = model_dim, in_elem
        V = 128 // (8 * in_bytes)  # 4 (f32), 8 (16b)
        load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), in_elem)
    out_bytes = out_numeric.width // 8
    is_16b = out_numeric.width < 32
    TILE = BLOCK * V
    store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), out_numeric)

    token, tile, tid = gpu.block_id("x"), gpu.block_id("y"), gpu.thread_id("x")
    tok64 = fx.Int64(token)
    vec_f32, vec_out = T.vec(V, T.f32), T.vec(V, out_numeric.ir_type)

    def _view(elem, ptr_i64, ncols, nbytes):  # 2D [1, ncols] V# buffer descriptor
        pt = fx.PointerType.get(
            elem.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=elem.width // 8,
        )
        view = fx.make_view(
            fx.inttoptr(pt, ptr_i64), fx.make_layout((1, ncols), (ncols, 1))
        )
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=fx.Int64(nbytes))

    # Fold the per-token byte offset into the base ptr: keeps voffsets i32-safe
    # for X > 4 GiB.
    x_row_bytes = topk * row_stride * in_bytes
    xbase = fx.Int64(ptrtoint(X)) + tok64 * fx.Int64(x_row_bytes)
    xbuf = _view(in_elem, xbase, model_dim, x_row_bytes)
    ybuf = _view(
        out_numeric,
        fx.Int64(ptrtoint(Y)) + tok64 * fx.Int64(model_dim * out_bytes),
        model_dim,
        model_dim * out_bytes,
    )
    if const_expr(is_fp8):
        # e8m0 scales trail the values in each row (one byte per FP8_VEC elems).
        scbuf = _view(
            fx.Int8,
            xbase + fx.Int64(model_dim),
            model_dim // 8,
            x_row_bytes - model_dim,
        )
    if const_expr(use_mask):
        i32pt = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        tk_ptr = fx.inttoptr(
            i32pt, fx.Int64(ptrtoint(topk_ids)) + tok64 * fx.Int64(topk * 4)
        )
        em_ptr = fx.inttoptr(i32pt, fx.Int64(ptrtoint(expert_mask)))

    # Tiled copy: BLOCK threads across the tile, V contiguous elems per thread.
    tile_mn, tv_layout = fx.make_layout_tv(
        fx.make_layout((1, BLOCK), (1, 1)), fx.make_layout((1, V), (1, 1))
    )
    thr_load = fx.make_tiled_copy(load_atom, tv_layout, tile_mn).get_slice(tid)
    thr_store = fx.make_tiled_copy(store_atom, tv_layout, tile_mn).get_slice(tid)
    if const_expr(is_fp8):
        sc_atom = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Int8)
        sc_tile, sc_tv = fx.make_layout_tv(
            fx.make_layout((1, BLOCK), (1, 1)), fx.make_layout((1, 1), (1, 1))
        )
        thr_sc = fx.make_tiled_copy(sc_atom, sc_tv, sc_tile).get_slice(tid)

    def _decode_fp8(vfrag, sfrag):  # 8 fp8 bytes + 1 e8m0 -> Vector(8, f32)
        w = fx.Vector(fx.memref_load_vec(vfrag)).bitcast(fx.Int32)
        e8m0 = fx.Uint32(fx.Uint8(fx.Vector(fx.memref_load_vec(sfrag))[0]))
        scale = (e8m0 << fx.Uint32(23)).bitcast(fx.Float32)
        words = (w[0], w[0], w[1], w[1])
        lanes = []
        for pi in range_constexpr(4):
            pair = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, words[pi], bool(pi & 1)))
            lanes.append(pair[0] * scale)
            lanes.append(pair[1] * scale)
        return fx.Vector.from_elements(lanes, fx.Float32)

    def _reduce_tile():
        p_src = thr_load.partition_S(
            fx.slice(fx.zipped_divide(xbuf, tile_mn), (None, (0, tile)))
        )
        p_dst = thr_store.partition_D(
            fx.slice(fx.zipped_divide(ybuf, tile_mn), (None, (0, tile)))
        )
        # topk rows share one per-thread voffset via a uniform scalar
        # soffset = k*row_stride, so the loads issue back-to-back.
        frags = [fx.make_fragment_like(p_src) for _ in range_constexpr(topk)]
        for k in range_constexpr(topk):
            fx.copy(load_atom, p_src, frags[k], soffset=fx.Int32(k * row_stride))
        if const_expr(is_fp8):
            p_sc = thr_sc.partition_S(
                fx.slice(fx.zipped_divide(scbuf, sc_tile), (None, (0, tile)))
            )
            sfrags = [fx.make_fragment_like(p_sc) for _ in range_constexpr(topk)]
            for k in range_constexpr(topk):
                fx.copy(sc_atom, p_sc, sfrags[k], soffset=fx.Int32(k * row_stride))

        acc = fx.Vector.filled(V, 0.0, fx.Float32)
        for k in range_constexpr(topk):
            if const_expr(is_fp8):
                vk = _decode_fp8(frags[k], sfrags[k])
            else:
                vk = fx.Vector(fx.memref_load_vec(frags[k]))
                vk = vk.extf(vec_f32) if is_16b else vk
            if const_expr(use_mask):
                vk = (em_ptr[tk_ptr[k]] != fx.Int32(0)).select(
                    vk, fx.Vector.filled(V, 0.0, fx.Float32)
                )
            acc = acc + vk
        if const_expr(model_dim_pad > 0):
            col0 = fx.Int32(tile) * fx.Int32(TILE) + fx.Int32(tid) * fx.Int32(V)
            valid_model_dim = fx.Int32(model_dim - model_dim_pad)
            zero = fx.Float32(0.0)
            acc = fx.Vector.from_elements(
                [
                    (col0 + fx.Int32(i) < valid_model_dim).select(acc[i], zero)
                    for i in range_constexpr(V)
                ],
                fx.Float32,
            )
        ofrag = fx.make_fragment_like(p_dst)
        fx.memref_store_vec(acc.truncf(vec_out) if is_16b else acc, ofrag)
        fx.copy(store_atom, ofrag, p_dst)

    # Skip threads whose column group starts past model_dim (their loads would
    # read the next row -- in-descriptor, wasted BW); only needed when TILE ? md.
    if const_expr(model_dim % TILE != 0):
        if fx.Int32(tile) * fx.Int32(TILE) + fx.Int32(tid) * fx.Int32(V) < fx.Int32(
            model_dim
        ):
            _reduce_tile()
    else:
        _reduce_tile()


@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    model_dim_pad: int = 0,
    dtype_str: str = "f16",
    use_mask: bool = False,
    num_experts: int = 0,
    out_dtype_str: str | None = None,
):
    """Compile the topk-reduce launcher for one Constexpr set (cached per shape).

    Returns a ``@flyc.jit`` taking ``(X, Y, expert_mask, topk_ids, i32_m_tokens,
    stream)``; dispatch it through ``moe_kernels._run_compiled``. The launcher is a
    distinct object per shape, so the shim's per-exe ``_cf`` cache stays correct.
    """
    if not 0 <= model_dim_pad < model_dim:
        raise ValueError(
            f"model_dim_pad must be in [0, {model_dim}), got {model_dim_pad}"
        )
    V = FP8_VEC if dtype_str == "fp8" else 128 // (32 if dtype_str == "f32" else 16)
    gy = (model_dim + BLOCK * V - 1) // (BLOCK * V)
    out_tag = out_dtype_str or dtype_str

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        moe_reduction_kernel(
            X,
            Y,
            expert_mask,
            topk_ids,
            i32_m_tokens,
            topk,
            model_dim,
            model_dim_pad,
            dtype_str,
            use_mask,
            num_experts,
            out_tag,
        ).launch(
            grid=(fx.Int64(i32_m_tokens), gy, 1), block=(BLOCK, 1, 1), stream=stream
        )

    return launch

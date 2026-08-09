# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE topk-reduction kernel (FlyDSL, layout API).

``Y[t, d] = sum_k X[t, k, d]``, optionally gated by the EP validity mask
(``valid[t,k] = expert_mask[topk_ids[t,k]] != 0``). Epilogue of stage2
``mode="reduce"``, shared by every dtype's reduce path. Extracted from
``moe_gemm_2stage.py``. Build a per-shape launcher with ``compile_moe_reduction``
(cached); the kernel's compile-time params are ``Constexpr`` so flyc specializes
per shape/dtype.

``dtype_str="fp8"`` reduces MXFP8 route-out rows (a flat uint8 buffer of
``[model_dim fp8 bytes | model_dim/scale_blk e8m0 scale bytes]`` per row): each fp8
value is scaled by its e8m0 microscale, accumulated in f32 and written to
``out_dtype_str`` (bf16/f16). The dense (f32/f16/bf16) path reduces a
contiguous ``X[tokens, topk, model_dim]`` tensor.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

from .mxfp4_gemm_common import FP8OUT_SCALE_BLK, fp8out_row_bytes

BLOCK = 256
MAX_BLOCK = 1024  # AMDGPU workgroup limit
WAVE = 64
FP8_VEC = 8  # fp8 values per 64b buffer load (also the store granularity)


def _reduce_block(model_dim, V):
    exact_block = model_dim // V
    if model_dim % V or exact_block % WAVE or not 0 < exact_block <= MAX_BLOCK:
        return BLOCK
    return exact_block


def _moe_reduction_kernel(
    X: fx.Pointer,
    Y: fx.Pointer,
    expert_mask: fx.Pointer,
    topk_ids: fx.Pointer,
    topk_weights: fx.Pointer,
    i32_m_tokens: fx.Int32,
    topk: fx.Constexpr[int],
    model_dim: fx.Constexpr[int],
    dtype_str: fx.Constexpr[str],
    use_mask: fx.Constexpr[bool],
    num_experts: fx.Constexpr[int],
    out_dtype_str: fx.Constexpr[str],
    use_weight: fx.Constexpr[bool],
    block_size: fx.Constexpr[int],
    scale_blk: fx.Constexpr[int],
    fp8_row_stride: fx.Constexpr[int],
):
    # One tiled-copy reduce for every dtype. Dense (f16/bf16/f32) loads V elems
    # and extends to f32; fp8 route-out loads 8 fp8 bytes + their e8m0 microscale
    # and decodes to f32. Both then run the same masked f32 topk-accumulate
    # (uniform soffset = k*row_stride) and truncating store. row_stride differs:
    # an fp8 row is padded with its scale bytes ([N fp8 | N/scale_blk e8m0]).
    is_fp8 = dtype_str == "fp8"
    if const_expr(is_fp8):
        in_elem, in_bytes, V = fx.Int8, 1, FP8_VEC
        row_stride = fp8_row_stride
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
    TILE = block_size * V
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
        # e8m0 scales trail the values in each row (one byte per scale_blk elems).
        i8pt = fx.PointerType.get(
            fx.Int8.ir_type, address_space=fx.AddressSpace.Global, alignment=1
        )
        sc_ptr = fx.inttoptr(i8pt, xbase + fx.Int64(model_dim))
    if const_expr(use_mask):
        i32pt = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        tk_ptr = fx.inttoptr(
            i32pt, fx.Int64(ptrtoint(topk_ids)) + tok64 * fx.Int64(topk * 4)
        )
        em_ptr = fx.inttoptr(i32pt, fx.Int64(ptrtoint(expert_mask)))
    if const_expr(use_weight):
        # Route weights deferred from the stage2 epilogue: scale each expert's
        # contribution here instead, so the GEMM stores unweighted rows.
        f32pt = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        tw_ptr = fx.inttoptr(
            f32pt, fx.Int64(ptrtoint(topk_weights)) + tok64 * fx.Int64(topk * 4)
        )

    # Tiled copy: block_size threads across the tile, V contiguous elems per thread.
    tile_mn, tv_layout = fx.make_layout_tv(
        fx.make_layout((1, block_size), (1, 1)), fx.make_layout((1, V), (1, 1))
    )
    thr_load = fx.make_tiled_copy(load_atom, tv_layout, tile_mn).get_slice(tid)
    thr_store = fx.make_tiled_copy(store_atom, tv_layout, tile_mn).get_slice(tid)

    def _decode_fp8(vfrag, scale):  # 8 fp8 bytes + its e8m0 scale -> Vector(8, f32)
        w = fx.Vector(fx.memref_load_vec(vfrag)).bitcast(fx.Int32)
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
            sc_col = (
                fx.Int32(tile) * fx.Int32(TILE) + fx.Int32(tid) * fx.Int32(V)
            ) // fx.Int32(scale_blk)
            scales = []
            for k in range_constexpr(topk):
                e8m0 = fx.Uint32(fx.Uint8(sc_ptr[sc_col + fx.Int32(k * row_stride)]))
                scales.append((e8m0 << fx.Uint32(23)).bitcast(fx.Float32))

        acc = fx.Vector.filled(V, 0.0, fx.Float32)
        for k in range_constexpr(topk):
            if const_expr(is_fp8):
                vk = _decode_fp8(frags[k], scales[k])
            else:
                vk = fx.Vector(fx.memref_load_vec(frags[k]))
                vk = vk.extf(vec_f32) if is_16b else vk
            if const_expr(use_weight):
                wk = tw_ptr[k]
                vk = fx.Vector.from_elements(
                    [vk[i] * wk for i in range_constexpr(V)], fx.Float32
                )
            if const_expr(use_mask):
                vk = (em_ptr[tk_ptr[k]] != fx.Int32(0)).select(
                    vk, fx.Vector.filled(V, 0.0, fx.Float32)
                )
            acc = acc + vk
        ofrag = fx.make_fragment_like(p_dst)
        fx.memref_store_vec(acc.truncf(vec_out) if is_16b else acc, ofrag)
        fx.copy(store_atom, ofrag, p_dst)

    # Skip threads whose column group starts past model_dim (their loads would
    # read the next row -- in-descriptor, wasted BW); only needed when TILE ∤ md.
    if const_expr(model_dim % TILE != 0):
        if fx.Int32(tile) * fx.Int32(TILE) + fx.Int32(tid) * fx.Int32(V) < fx.Int32(
            model_dim
        ):
            _reduce_tile()
    else:
        _reduce_tile()


@functools.cache
def _reduction_kernel(block_size: int):
    return flyc.kernel(
        _moe_reduction_kernel,
        name=f"moe_reduction_kernel_b{block_size}",
        known_block_size=[block_size, 1, 1],
    )


@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
    use_mask: bool = False,
    num_experts: int = 0,
    out_dtype_str: str | None = None,
    use_weight: bool = False,
):
    """Compile the topk-reduce launcher for one Constexpr set (cached per shape).

    Returns a ``@flyc.jit`` taking ``(X, Y, expert_mask, topk_ids, topk_weights,
    i32_m_tokens, stream)``; dispatch it through ``moe_kernels._run_compiled``. The
    launcher is a distinct object per shape, so the shim's per-exe ``_cf`` cache
    stays correct.
    """
    V = FP8_VEC if dtype_str == "fp8" else 128 // (32 if dtype_str == "f32" else 16)
    block = _reduce_block(model_dim, V)
    gy = (model_dim + block * V - 1) // (block * V)
    out_tag = out_dtype_str or dtype_str
    kern = _reduction_kernel(block)
    if dtype_str == "fp8":
        if FP8OUT_SCALE_BLK % FP8_VEC:
            raise ValueError(
                f"scale_blk {FP8OUT_SCALE_BLK} must be a multiple of {FP8_VEC}"
            )
        scale_blk = FP8OUT_SCALE_BLK
        fp8_row_stride = fp8out_row_bytes(model_dim)
    else:
        scale_blk, fp8_row_stride = FP8_VEC, model_dim

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        topk_weights: fx.Pointer,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        kern(
            X,
            Y,
            expert_mask,
            topk_ids,
            topk_weights,
            i32_m_tokens,
            topk,
            model_dim,
            dtype_str,
            use_mask,
            num_experts,
            out_tag,
            use_weight,
            block,
            scale_blk,
            fp8_row_stride,
        ).launch(
            grid=(fx.Int64(i32_m_tokens), gy, 1), block=(block, 1, 1), stream=stream
        )

    return launch

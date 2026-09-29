# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Tiled NCDHW -> NDHWC pre-transpose. Separate cache from the convolution."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr

from .conv3d_gfx950_utils import BF16_BYTES, CONV_COMPILE_HINTS, _as_stream

TR_TILE = 64
TR_VEC = 8
TR_THREADS = 256
_TR_VPL = TR_TILE // TR_VEC
_TR_ITERS = (TR_TILE * TR_TILE) // (TR_VEC * TR_THREADS)
_TR_PAD = 8
_TR_LDS_S = TR_TILE + _TR_PAD

TR_MAX_BIG_S = (0x7FFFFFFF - (TR_TILE - TR_VEC)) // (TR_TILE - 1)


@functools.lru_cache(maxsize=64)
def compile_transpose_ncdhw_ndhwc(n, c, s):
    """Transpose (N, C, S) -> (N, S, C). ``s`` is a runtime operand; requires c%8==0."""
    grid_c = (c + TR_TILE - 1) // TR_TILE
    elem_ty = fx.BFloat16
    BIG = (n * c * s) > 0x7FFFFFFF

    # 1-D element view so the flat gather/scatter offsets index elements. Both
    # descriptors stay max_size: an exact num_records would zero the whole
    # straddling tail read.
    _TR_REBASED_FLAT = 0xFFFFFFFF // BF16_BYTES

    def _flat_div(buf_ptr, elems):
        return fx.logical_divide(
            fx.Tensor(fx.make_view(buf_ptr, fx.make_layout(elems, 1))),
            fx.make_layout(1, 1),
        )

    @fx.struct
    class SharedStorage:
        tile: fx.Array[elem_ty, TR_TILE * _TR_LDS_S, 16]

    @flyc.kernel(known_block_size=[TR_THREADS, 1, 1])
    def transpose_kernel(out: fx.Tensor, inp: fx.Tensor, s: fx.Int32):
        lds = fx.SharedAllocator(static=False).allocate(SharedStorage).peek().tile

        tid = fx.Int32(gpu.thread_id("x"))
        s0 = fx.Int32(gpu.block_id("x")) * TR_TILE
        c0 = fx.Int32(gpu.block_id("y")) * TR_TILE
        nb = fx.Int32(gpu.block_id("z"))
        if const_expr(BIG):
            # Rebase onto this block's tile origin so the per-tile offsets stay in i32.
            GPtrTy = fx.PointerType.get(
                elem_ty.ir_type, fx.AddressSpace.Global, BF16_BYTES
            )

            def _rebased(tensor, base_elem):
                addr = fx.Int64(fx.ptrtoint(fx.get_iter(tensor))) + fx.Int64(
                    base_elem
                ) * fx.Int64(BF16_BYTES)
                return _flat_div(
                    fx.rocdl.make_buffer_ptr(fx.inttoptr(GPtrTy, addr)),
                    _TR_REBASED_FLAT,
                )

            in_base_elem = (
                fx.Int64(nb) * fx.Int64(c) * fx.Int64(s)
                + fx.Int64(c0) * fx.Int64(s)
                + fx.Int64(s0)
            )
            in_div = _rebased(inp, in_base_elem)
            out_base_elem = (
                fx.Int64(nb) * fx.Int64(s) * fx.Int64(c)
                + fx.Int64(s0) * fx.Int64(c)
                + fx.Int64(c0)
            )
            out_div = _rebased(out, out_base_elem)
        else:
            in_base = nb * c * s
            out_base = nb * s * c
            flat = n * c * s
            in_div = _flat_div(fx.get_iter(fx.rocdl.make_buffer_tensor(inp)), flat)
            out_div = _flat_div(fx.get_iter(fx.rocdl.make_buffer_tensor(out)), flat)
        tr_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
        tr_reg = fx.make_rmem_tensor(TR_VEC, elem_ty)

        _lds_st_ptr_ty = fx.PointerType.get(
            elem_ty.ir_type, fx.AddressSpace.Shared, TR_VEC * BF16_BYTES
        )

        def lds_store_vec8(elem_offset, value):
            # Destination is a raw LDS byte offset with no tensor form to copy
            # into, so the pointer is built by hand. _lds_st_ptr_ty carries the
            # 16-byte alignment, which is what keeps this one ds_write_b128
            # instead of eight scalar writes.
            base = fx.Int64(fx.ptrtoint(lds.ptr)) + fx.Int64(elem_offset * 2)
            fx.ptr_store(value, fx.inttoptr(_lds_st_ptr_ty, base))

        def lds_load_scalar(elem_offset):
            u8 = fx.recast_iter(fx.Uint8, lds.ptr)
            return fx.ptr_load(u8 + fx.Int32(elem_offset * 2), result_type=elem_ty)

        # Read: coalesced vec8 along contiguous S -> LDS[c_local][s_local].
        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rc = lin // _TR_VPL
            sv = (lin % _TR_VPL) * TR_VEC
            cc = c0 + rc
            ss = s0 + sv
            valid = (cc < c) & (ss < s)
            if const_expr(BIG):
                g = fx.Int32(rc * s + sv)
            else:
                g = fx.Int32(in_base + cc * s + ss)
            safe = valid.select(g, fx.Int32(0))
            fx.copy(tr_atom, fx.slice(in_div, (None, safe)), tr_reg)
            v = fx.memref_load_vec(tr_reg)
            lds_store_vec8(rc * _TR_LDS_S + sv, v)

        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.rocdl.s_barrier()

        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rs = lin // _TR_VPL
            cv = (lin % _TR_VPL) * TR_VEC
            ss = s0 + rs
            cc = c0 + cv
            scalars = [
                lds_load_scalar((cv + j) * _TR_LDS_S + rs)
                for j in range_constexpr(TR_VEC)
            ]
            vv = fx.Vector.from_elements(scalars, dtype=elem_ty)
            valid = (ss < s) & (cc < c)
            if valid:
                if const_expr(BIG):
                    go = fx.Int32(rs * c + cv)
                else:
                    go = fx.Int32(out_base + ss * c + cc)
                fx.memref_store_vec(vv, tr_reg)
                fx.copy(tr_atom, tr_reg, fx.slice(out_div, (None, go)))

    @flyc.jit
    def launch_transpose(
        out: fx.Tensor,
        inp: fx.Tensor,
        s_rt: fx.Int32,
        # flydsl's launcher signature convention; the default is the DSL's
        # null-stream sentinel, not a live handle captured at import.
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        transpose_kernel(out, inp, s_rt).launch(
            # S is the only grid axis the resolution moves, so it is sized
            # here rather than baked in.
            grid=((s_rt + (TR_TILE - 1)) // TR_TILE, grid_c, n),
            block=(TR_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch_transpose(out, inp, s, stream=_as_stream(stream))

    def _compile(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch_transpose, out, inp, s, _as_stream(stream))

    _launch.compile = _compile
    # Reachable from the object because the steady-state path calls the
    # compiled function directly; see ``conv_kernels._dispatch``.
    _launch.extra_args = (s,)
    return _launch

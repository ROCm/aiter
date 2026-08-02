# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE topk-reduction kernel (FlyDSL, layout API).

Sums stage2's per-slot output over the ``topk`` dimension
(``Y[t, d] = sum_k X[t, k, d]``) and optionally fuses the EP validity gather so
only slots whose expert is live are accumulated. This is the epilogue of stage2
``mode="reduce"`` (``compile_moe_gemm2(accumulate=False)``), which trades the
atomic contention of atomic-accumulate mode for a separate reduce pass. It is
shared by every dtype's reduce path, not just a16wi4.

Extracted from ``moe_gemm_2stage.py`` so it lives independently of the int4
stage1/stage2 GEMM builders in that module.

Data movement uses the FlyDSL layout API (``make_buffer_tensor`` +
``logical_divide``/``slice`` + ``fx.copy`` buffer-copy atoms). Two details keep it
on par with a hand-tuned ``buffer_load`` loop on this bandwidth/latency-bound
kernel:

* One WG per token; its X slab and Y row are viewed at an ``fx.inttoptr`` base with
  the per-token byte offset folded in. That keeps in-kernel voffsets i32-safe even
  when the whole X tensor exceeds 4 GiB (e.g. 131072*6*4096*2 = 6 GiB).
* The ``topk`` rows are reached with a *uniform scalar* ``soffset = k*model_dim``,
  so all loads share a single per-thread voffset and issue back-to-back (the
  scalar offsets fall on the SALU, off the load critical path).

The buffer descriptor is sized to the slab, so out-of-bounds lanes on the
``model_dim`` tail read past / drop their store automatically -- no scalar tail
path. The one runtime guard skips threads whose column group starts beyond
``model_dim`` (their loads would otherwise read the next row -- in-descriptor,
wasted bandwidth), and it is only emitted when ``TILE`` does not divide
``model_dim``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

_NUMERIC = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}


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
    if dtype_str not in _NUMERIC:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    elem_cls = _NUMERIC[dtype_str]
    elem_bits = 32 if dtype_str == "f32" else 16
    elem_bytes = elem_bits // 8

    BLOCK_SIZE = 256
    V = 128 // elem_bits  # elems per 128b copy: 4 (f32), 8 (f16/bf16)
    TILE = BLOCK_SIZE * V  # columns per workgroup tile
    is_16b = elem_bits < 32

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
        token = gpu.block_id("x")
        tile = gpu.block_id("y")
        tid = gpu.thread_id("x")
        vec_f32 = T.vec(V, T.f32)
        vec_e = T.vec(V, elem_cls.ir_type)

        def _view(ptr_i64, layout, elem, nbytes):
            pt = fx.PointerType.get(
                elem.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=elem.width // 8,
            )
            view = fx.Tensor(fx.make_view(fx.inttoptr(pt, ptr_i64), layout))
            return fx.rocdl.make_buffer_tensor(view, num_records_bytes=fx.Int64(nbytes))

        eb = fx.Int64(elem_bytes)
        md = fx.Int64(model_dim)
        tok64 = fx.Int64(token)
        x_base = fx.Int64(ptrtoint(X)) + tok64 * fx.Int64(topk) * md * eb
        y_base = fx.Int64(ptrtoint(Y)) + tok64 * md * eb
        # X slab as a 1D row view; the topk rows are reached via a uniform scalar
        # soffset (k*model_dim), so every load shares one per-thread voffset. The
        # descriptor spans the whole slab (for the OOB tail).
        xbuf = _view(
            x_base,
            fx.make_layout(model_dim, 1),
            elem_cls,
            topk * model_dim * elem_bytes,
        )
        ybuf = _view(
            y_base, fx.make_layout(model_dim, 1), elem_cls, model_dim * elem_bytes
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

        def _tile_thread(buf1d):
            # buf1d [model_dim] -> this thread's V contiguous elems in its column tile.
            t = fx.logical_divide(buf1d, fx.make_layout(TILE, 1))
            t = fx.slice(t, (None, tile))
            t = fx.logical_divide(t, fx.make_layout(V, 1))
            return fx.slice(t, (None, tid))

        def _reduce_tile():
            src = _tile_thread(xbuf)
            frags = [fx.make_rmem_tensor(V, elem_cls) for _ in range_constexpr(topk)]
            for k in range_constexpr(topk):
                fx.copy(copy, src, frags[k], soffset=fx.Int32(k * model_dim))
            acc = fx.Vector.filled(V, 0.0, fx.Float32)
            for k in range_constexpr(topk):
                vk = fx.Vector(fx.memref_load_vec(frags[k]))
                if const_expr(use_mask):
                    # valid = expert_mask[topk_ids[token, k]] != 0 (bases folded).
                    mv_ok = em_ptr[tk_ptr[k]] != fx.Int32(0)
                    vk = mv_ok.select(vk, fx.Vector.filled(V, 0.0, elem_cls))
                acc = acc + (vk.extf(vec_f32) if is_16b else vk)
            out = acc.truncf(vec_e) if is_16b else acc
            ofrag = fx.make_rmem_tensor(V, elem_cls)
            fx.memref_store_vec(out, ofrag)
            fx.copy_atom_call(copy, ofrag, _tile_thread(ybuf))

        # Only the last column tile can start past model_dim, and only when TILE
        # does not divide it. Skip those threads (their loads would read the next
        # row -- in-descriptor, wasted BW; only the store is dropped). Emit the
        # runtime guard solely when it can fire.
        if const_expr(model_dim % TILE != 0):
            col_base = fx.Int32(tile) * fx.Int32(TILE) + fx.Int32(tid) * fx.Int32(V)
            if col_base < fx.Int32(model_dim):
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
            grid=(fx.Int64(i32_m_tokens), gy, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction

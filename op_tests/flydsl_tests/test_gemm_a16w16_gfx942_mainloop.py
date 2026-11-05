# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The gfx942 GEMM main loop as a building block of a fused test kernel.

A minimal fused kernel, written only against ``gemm_a16w16_gfx942_mainloop``,
scales its A fragments in ``transform_a`` and stores the tile itself. It runs
through the standard HGEMM harness against an f32 oracle.

The harness runs it only on gfx942 hardware.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr

from aiter.ops.flydsl.kernels.gemm_a16w16_gfx942_mainloop import (
    GemmA16W16Gfx942Param,
    gfx942_a_coords,
    gfx942_ab_lds_elems,
    gfx942_k_wave_split,
    gfx942_red_lds_elems,
    make_gemm_a16w16_gfx942_param,
    make_gfx942_mainloop,
    make_gfx942_tiled_mma,
    run_gfx942_mainloop,
)

BUFFERED = {"buffered_load": True, "async_load_bytes": 16}
PINGPONG = {**BUFFERED, "block_k": 128, "stages": 1, "pingpong": True}
PIPELINES = {
    "pingpong": PINGPONG,
    "pingpong-k_waves2": {**PINGPONG, "k_waves": 2},
    "buffered-1stage": {**BUFFERED, "stages": 1},
    "buffered-2stage": {**BUFFERED, "stages": 2},
    "async-2stage": {"stages": 2},
    "async-3stage": {"stages": 3},
}


@flyc.kernel
def _scaled_a_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    ks_begin: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmA16W16Gfx942Param,
):
    """C = (A * S)[:, ks_begin:] @ B[ks_begin:] with S[r, c] = 2 where
    (r + c) % 3 == 0, else 1: one split-K slice of a fused kernel.

    Scaling by 2 is exact in bf16, so the oracle needs no rounding model.
    """
    tid = fx.thread_idx.x
    k_wave_split = gfx942_k_wave_split(param, tid)
    bid_m = fx.block_idx.x
    bid_n = fx.block_idx.y
    a_elems, b_elems = gfx942_ab_lds_elems(param)

    @fx.struct
    class SharedAB:
        a: fx.Array[fx.BFloat16, a_elems, 16]
        b: fx.Array[fx.BFloat16, b_elems, 16]

    @fx.union
    class Shared:
        ab: SharedAB
        red: fx.Array[fx.Float32, max(1, gfx942_red_lds_elems(param)), 16]

    smem = fx.SharedAllocator().allocate(Shared)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=False)
    gC = fx.flat_divide(out_buf, (param.block_m, param.block_n))[
        None, None, bid_m, bid_n
    ]
    rows, cols = gfx942_a_coords(param, tiled_mma, k_wave_split)
    row0 = bid_m * param.block_m

    def scale_a(frag, k_col, g_begin, g_end):
        for g in range_constexpr(g_begin, g_end):
            frag_g = frag[None, None, g]
            vals = frag_g.load().to(fx.Float32)
            group_elems = fx.size(frag_g.shape).unpack()
            scales = []
            for i in range_constexpr(group_elems):
                # Group g is the last mode: its elements are flat
                # indices [g * group_elems, (g + 1) * group_elems).
                e = g * group_elems + i
                row = row0 + fx.Int32(fx.get_scalar(rows[e]))
                col = k_col + fx.Int32(fx.get_scalar(cols[e]))
                on = (row + col) % 3 == 0
                scales.append(on.select(fx.Float32(2.0), fx.Float32(1.0)))
            scale = fx.Vector.from_elements(scales, dtype=fx.Float32)
            frag_g.store((vals * scale.reshape(vals.shape)).to(fx.BFloat16))

    ml = make_gfx942_mainloop(
        param,
        tiled_mma,
        tid=tid,
        k_wave_split=k_wave_split,
        a=a,
        b=b,
        m=m,
        n=n,
        k=k,
        a_leading_stride=a_leading_stride,
        b_leading_stride=b_leading_stride,
        bid_m=bid_m,
        bid_n=bid_n,
        ks_begin=ks_begin,
        k_tiles=(k - ks_begin) // param.block_k,
        c_tile=gC,
        smem_a=smem.ab.a.peek().ptr,
        smem_b=smem.ab.b.peek().ptr,
        smem_red=smem.red.peek().ptr,
        transform_a=scale_a,
    )
    run_gfx942_mainloop(ml)

    c_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    thr_copy_C = fx.make_tiled_copy_C(c_atom, tiled_mma).get_slice(ml.tid_mn)
    frag_out = fx.make_fragment_like(ml.frag_C, fx.BFloat16)
    frag_out.store(ml.frag_C.load().to(fx.BFloat16))
    c_src = thr_copy_C.retile(frag_out)
    c_dst = thr_copy_C.partition_S(gC)
    if const_expr(param.k_waves > 1):
        if ml.k_wave_idx == 0:
            fx.copy(c_atom, c_src, c_dst)
    else:
        fx.copy(c_atom, c_src, c_dst)


@flyc.jit
def _scaled_a_gemm(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    ks_begin: fx.Int32,
    param: GemmA16W16Gfx942Param,
    stream: fx.Stream = fx.Stream(None),  # noqa: B008
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    _scaled_a_kernel(
        out,
        a,
        b,
        m,
        n,
        k,
        ks_begin,
        fx.Int32(fx.get_scalar(a.stride[0])),
        fx.Int32(fx.get_scalar(b.stride[1])),
        make_gfx942_tiled_mma(param),
        param,
    ).launch(
        grid=((m + param.block_m - 1) // param.block_m, n // param.block_n, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def prepare_gfx942_mainloop_case(pipeline, ks_begin):
    """Return the launch, f32 oracle, and dimensions for one pipeline and K start."""
    m, n, k = 200, 128, 1024
    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda") * k**-0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * k**-0.5).to(torch.bfloat16)
    idx = torch.arange(m, device="cuda")[:, None] + torch.arange(k, device="cuda")
    scale = torch.where(idx % 3 == 0, 2.0, 1.0)
    ref = (x.float() * scale)[:, ks_begin:] @ w.float()[:, ks_begin:].t()
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    param = make_gemm_a16w16_gfx942_param(
        block_m=64, block_n=64, direct_store=True, **PIPELINES[pipeline]
    )

    def run():
        _scaled_a_gemm(out, x, w.t(), ks_begin, param)
        return out

    return run, ref, (m, n, k)

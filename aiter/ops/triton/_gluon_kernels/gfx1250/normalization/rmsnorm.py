# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gluon_rms_norm_kernel_repr = make_kernel_repr(
    "_gluon_rms_norm_kernel",
    ["BLOCK_SIZE_N", "NUM_BUFFERS", "num_warps"],
)


@gluon.jit(repr=_gluon_rms_norm_kernel_repr)
def _gluon_rms_norm_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    rsigma_ptr,
    eps,
    M,
    N,
    x_stride_m,
    out_stride_m,
    BLOCK_SIZE_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Persistent RMSNorm for gfx1250. Rows are streamed HBM -> LDS with TDM
    async loads, NUM_BUFFERS deep, so the next rows are in flight while the
    current row is reduced and written out.
    """
    pid = gl.program_id(0)
    num_pids = gl.num_programs(0)
    NUM_WARPS: gl.constexpr = gl.num_warps()
    WARP_SIZE: gl.constexpr = 32
    # 16B per lane, lanes interleaved: conflict-free LDS reads and coalesced
    # global stores (a contiguous chunk per lane would stride lanes apart).
    VEC: gl.constexpr = 128 // x_ptr.dtype.element_ty.primitive_bitwidth

    layout: gl.constexpr = gl.BlockedLayout([VEC], [WARP_SIZE], [NUM_WARPS], [0])
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    # Rows beyond M are zero-filled by TDM; columns beyond N likewise.
    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        x_ptr, [M, N], [x_stride_m, 1], [1, BLOCK_SIZE_N], smem_layout
    )
    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        weight_ptr, [1, N], [0, 1], [1, BLOCK_SIZE_N], smem_layout
    )

    smem_x = gl.allocate_shared_memory(
        x_ptr.dtype.element_ty, [NUM_BUFFERS, 1, BLOCK_SIZE_N], smem_layout
    )
    smem_w = gl.allocate_shared_memory(
        weight_ptr.dtype.element_ty, [1, BLOCK_SIZE_N], smem_layout
    )

    # Prologue. Every load is issued unconditionally (out-of-range rows are
    # zero-filled) so the in-order TDM counter used by async_wait stays exact.
    gl.amd.gfx1250.tdm.async_load(w_desc, [0, 0], smem_w)
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(x_desc, [pid + i * num_pids, 0], smem_x.index(i))

    gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 1)
    w = smem_w.reshape([BLOCK_SIZE_N]).load(layout).to(gl.float32)

    cols = gl.arange(0, BLOCK_SIZE_N, layout=layout)
    col_mask = cols < N

    it = 0
    for row in range(pid, M, num_pids):
        # Refill the slot consumed in the previous iteration.
        next_row = row + (NUM_BUFFERS - 1) * num_pids
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [next_row, 0], smem_x.index((it + NUM_BUFFERS - 1) % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 1)

        x = (
            smem_x.index(it % NUM_BUFFERS)
            .reshape([BLOCK_SIZE_N])
            .load(layout)
            .to(gl.float32)
        )
        sum_sq = gl.sum(x * x, axis=0)
        norm_factor = gl.rsqrt(sum_sq / N + eps)
        gl.store(rsigma_ptr + row, norm_factor)

        y = x * norm_factor * w
        gl.store(
            out_ptr + row * out_stride_m + cols,
            y.to(out_ptr.dtype.element_ty),
            mask=col_mask,
        )
        it += 1  # noqa: SIM113

    # Drain outstanding prefetches before LDS is released.
    gl.amd.gfx1250.tdm.async_wait(0)

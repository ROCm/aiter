# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon (gfx1250) port of triton _fused_clamp_silu_mul_kernel for GFX1250

"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.activation import _apply_activation_from_str
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_GLUON_REPR_KEYS = [
    "ROWS_PER_PROG",
    "BLOCK_SIZE_N",
    "QUANT_BLOCK_SIZE",
    "SCALE_FMT",
    "HAVE_WEIGHTS",
    "WEIGHT_BROADCAST",
    "HAVE_SWIGLU_CLAMP",
    "HAS_QUANT",
    "num_warps",
    "cache_modifier",
]

_fused_clamp_silu_mul_repr = make_kernel_repr(
    "_fused_clamp_silu_mul_gfx1250_kernel", _GLUON_REPR_KEYS
)


@gluon.jit(repr=_fused_clamp_silu_mul_repr)
def _fused_clamp_silu_mul_kernel(
    inp_ptr,
    out_ptr,
    scale_ptr,
    weights_ptr,
    M,
    n_half,
    inp_stride_m,
    inp_stride_n,
    out_stride_m,
    out_stride_n,
    scale_stride_m,
    scale_stride_n,
    weights_stride_m,
    weights_stride_n,
    swiglu_limit,
    ROWS_PER_PROG: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    QUANT_BLOCK_SIZE: gl.constexpr,
    SCALE_FMT: gl.constexpr,
    DTYPE_MAX: gl.constexpr,
    DTYPE_MIN: gl.constexpr,
    HAVE_WEIGHTS: gl.constexpr,
    WEIGHT_BROADCAST: gl.constexpr,
    HAVE_SWIGLU_CLAMP: gl.constexpr,
    HAS_QUANT: gl.constexpr,
    ACTIVATION: gl.constexpr,
    SHUFFLE: gl.constexpr,
    SCALE_N_PAD: gl.constexpr,
    num_warps: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    # constants
    NUM_N_Q_GROUPS: gl.constexpr = (
        BLOCK_SIZE_N // QUANT_BLOCK_SIZE
    )  # quant groups per row
    ROWS_PER_PROG_TOTAL: gl.constexpr = (
        ROWS_PER_PROG * BLOCK_SIZE_M
    )  # total rows processed

    # LDS layout for the staged input tiles; padded to break bank conflicts,
    # interval clamped to what the TDM descriptor can encode (1024B)
    PAD_INTERVAL: gl.constexpr = min(
        BLOCK_SIZE_N, 1024 // inp_ptr.dtype.element_ty.itemsize
    )

    # the padded LDS layout itself, over one [BM, BN] tile
    shared_tdm_layout_2d: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[PAD_INTERVAL, 16 // inp_ptr.dtype.element_ty.itemsize]],
        [BLOCK_SIZE_M, BLOCK_SIZE_N],
        [1, 0],
    )

    # setup, implementation scales across M and N tiles => 2 pids
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    m_start = pid_m * ROWS_PER_PROG_TOTAL
    n_start = pid_n * BLOCK_SIZE_N
    g_start = pid_n * NUM_N_Q_GROUPS

    # TDM setup -- 1 TDM, two separate loads for gate + up
    inp_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=inp_ptr,
        shape=[M, 2 * n_half],
        strides=[inp_stride_m, inp_stride_n],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=shared_tdm_layout_2d,
    )
    smem = gl.allocate_shared_memory(
        inp_desc.dtype,
        shape=[2 * ROWS_PER_PROG] + inp_desc.block_shape,
        layout=shared_tdm_layout_2d,
    )

    # async loads gate + up
    gl.amd.gfx1250.tdm.async_load(inp_desc, [m_start, n_start], smem.index(0))
    gl.amd.gfx1250.tdm.async_load(inp_desc, [m_start, n_half + n_start], smem.index(1))

    # layout for input tile
    N_PER_THREAD: gl.constexpr = max(1, min(8, BLOCK_SIZE_N // (32 * num_warps)))
    xLayout2D: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, N_PER_THREAD],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )

    # register layout for group scale tile
    if NUM_N_Q_GROUPS > 128:
        reg_group_bases: gl.constexpr = [
            [0, 32],
            [0, 64],
            [0, 128],
        ]
    elif NUM_N_Q_GROUPS > 64:
        reg_group_bases: gl.constexpr = [
            [0, 32],
            [0, 64],
        ]
    elif NUM_N_Q_GROUPS > 32:
        reg_group_bases: gl.constexpr = [
            [0, 32],
        ]
    else:
        reg_group_bases: gl.constexpr = []

    # one basis per bit of BLOCK_SIZE_M
    if BLOCK_SIZE_M > 8:
        reg_row_bases: gl.constexpr = [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ]
    elif BLOCK_SIZE_M > 4:
        reg_row_bases: gl.constexpr = [
            [1, 0],
            [2, 0],
            [4, 0],
        ]
    elif BLOCK_SIZE_M > 2:
        reg_row_bases: gl.constexpr = [
            [1, 0],
            [2, 0],
        ]
    elif BLOCK_SIZE_M > 1:
        reg_row_bases: gl.constexpr = [
            [1, 0],
        ]
    else:
        reg_row_bases: gl.constexpr = []

    # all-zero bases: every warp holds the whole scale tile
    if num_warps > 4:
        warp_bases: gl.constexpr = [
            [0, 0],
            [0, 0],
            [0, 0],
        ]
    elif num_warps > 2:
        warp_bases: gl.constexpr = [
            [0, 0],
            [0, 0],
        ]
    elif num_warps > 1:
        warp_bases: gl.constexpr = [
            [0, 0],
        ]
    else:
        warp_bases: gl.constexpr = []

    # lanes take the quant groups so the store coalesces
    scaleLayout2D: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=reg_group_bases + reg_row_bases,
        lane_bases=[
            [0, 1] if NUM_N_Q_GROUPS > 1 else [0, 0],
            [0, 2] if NUM_N_Q_GROUPS > 2 else [0, 0],
            [0, 4] if NUM_N_Q_GROUPS > 4 else [0, 0],
            [0, 8] if NUM_N_Q_GROUPS > 8 else [0, 0],
            [0, 16] if NUM_N_Q_GROUPS > 16 else [0, 0],
        ],
        warp_bases=warp_bases,
        block_bases=[],
        shape=[BLOCK_SIZE_M, NUM_N_Q_GROUPS],
    )

    # row/chunk 1D layouts, no convert_layout needed later
    row_layout: gl.constexpr = gl.SliceLayout(0, xLayout2D)
    m_layout: gl.constexpr = gl.SliceLayout(1, xLayout2D)
    row_scale_layout: gl.constexpr = gl.SliceLayout(0, scaleLayout2D)
    m_scale_layout: gl.constexpr = gl.SliceLayout(1, scaleLayout2D)

    # setup buffer load/store
    offs = gl.arange(0, BLOCK_SIZE_N, layout=row_layout)
    cols = n_start + offs
    mask = cols < n_half
    num_bs = gl.cdiv(n_half, QUANT_BLOCK_SIZE)
    g_offs = g_start + gl.arange(0, NUM_N_Q_GROUPS, layout=row_scale_layout)

    # row ids in the scale layout
    s_rows = gl.arange(0, BLOCK_SIZE_M, layout=m_scale_layout)

    # row ids in the data layout
    m_ids = gl.arange(0, BLOCK_SIZE_M, layout=m_layout)

    store_offs = (m_ids[:, None] * out_stride_m + cols[None, :] * out_stride_n).to(
        gl.int32
    )

    # main loop
    for i in range(ROWS_PER_PROG - 1):
        # prefetch the next tile's halves, overlapping this trip's work
        gl.amd.gfx1250.tdm.async_load(
            inp_desc,
            [m_start + (i + 1) * BLOCK_SIZE_M, n_start],
            smem.index(2 * (i + 1)),
        )
        gl.amd.gfx1250.tdm.async_load(
            inp_desc,
            [m_start + (i + 1) * BLOCK_SIZE_M, n_start + n_half],
            smem.index(2 * (i + 1) + 1),
        )

        # this trip's first row and its two LDS slots
        row = m_start + i * BLOCK_SIZE_M
        gate_tile = smem.index(2 * i)
        up_tile = smem.index(2 * i + 1)

        # this tile's rows
        abs_rows = row + m_ids
        row_mask = abs_rows < M

        if HAVE_WEIGHTS:
            if WEIGHT_BROADCAST:
                # 1D vector over the tile's rows; buffer_load is overkill + slower
                w = gl.load(
                    weights_ptr + abs_rows * weights_stride_m,
                    mask=row_mask,
                    other=0.0,
                )
            else:
                w = gl.amd.gfx1250.buffer_load(
                    weights_ptr + row.to(gl.int64) * weights_stride_m,
                    (
                        m_ids[:, None] * weights_stride_m
                        + cols[None, :] * weights_stride_n
                    ).to(gl.int32),
                    mask=row_mask[:, None] & mask[None, :],
                    other=0.0,
                    cache=cache_modifier,
                )

        s_abs_rows = row + s_rows
        scale_mask = (s_abs_rows < M)[:, None] & (g_offs < num_bs)[None, :]

        # preshuffled scale addressing, matching the triton shuffle layout
        if SHUFFLE:
            bs_r = s_abs_rows[:, None]
            bs_g = g_offs[None, :]
            bs_offs_0 = bs_r // 32
            bs_offs_1 = bs_r % 32
            bs_offs_2 = bs_offs_1 % 16
            bs_offs_1 = bs_offs_1 // 16
            bs_offs_3 = bs_g // 8
            bs_offs_4 = bs_g % 8
            bs_offs_5 = bs_offs_4 % 4
            bs_offs_4 = bs_offs_4 // 4
            bs_offs = (
                bs_offs_1
                + bs_offs_4 * 2
                + bs_offs_2 * 2 * 2
                + bs_offs_5 * 2 * 2 * 16
                + bs_offs_3 * 2 * 2 * 16 * 4
                + bs_offs_0 * 2 * 16 * SCALE_N_PAD
            )
        else:
            bs_offs = 0  # not needed

        # wait for this tile
        gl.amd.gfx1250.tdm.async_wait(2)

        gate = gate_tile.load(xLayout2D).to(gl.float32)
        up = up_tile.load(xLayout2D).to(gl.float32)

        # clamp
        if HAVE_SWIGLU_CLAMP:
            gate = gl.minimum(gate, swiglu_limit)
            up = gl.clamp(up, -swiglu_limit, swiglu_limit)

        # apply act(gate)*up
        out = _apply_activation_from_str(gate, ACTIVATION) * up

        # apply weights; broadcast form is one value per row of the tile
        if HAVE_WEIGHTS:
            if WEIGHT_BROADCAST:
                out = out * w.to(gl.float32)[:, None]
            else:
                out = out * w.to(gl.float32)

        # group quant and store
        if HAS_QUANT:
            if SCALE_FMT == "ue8m0":
                # mxfp8, reduce over inner QUANT_BLOCK_SIZE axis.
                out_3d = gl.reshape(
                    out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE]
                )
                abs_3d = gl.maximum(out_3d, -out_3d)
                max_val = gl.max(abs_3d, axis=2, keep_dims=True)  # [BM, NQB, 1]
                dequant_scale = max_val / DTYPE_MAX
                # ROUND_UP to a power of two via the fp32 exponent field.
                dequant_scale_exp = (
                    dequant_scale.to(gl.uint32, bitcast=True) + 0x007FFFFF
                ) & 0x7F800000
                dequant_scale_rounded = dequant_scale_exp.to(gl.float32, bitcast=True)
                quant_scale = gl.where(  # reciprocal, guard 0
                    dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
                )
                quant_tensor = out_3d * quant_scale  # scale into fp8 range
                out_q = gl.convert_layout(
                    gl.reshape(quant_tensor, [BLOCK_SIZE_M, BLOCK_SIZE_N]), xLayout2D
                )
                scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)  # [BM, NQB, 1]
                block_scales = gl.convert_layout(
                    gl.reshape(scale_exp, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), scaleLayout2D
                )
            else:
                # fp8 quant
                out_3d = gl.reshape(
                    out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE]
                )
                abs_3d = gl.maximum(out_3d, -out_3d)
                max_val = gl.maximum(gl.max(abs_3d, axis=2, keep_dims=True), 1e-10)
                scale_out = max_val / DTYPE_MAX
                quant_3d = gl.clamp(out_3d * (1.0 / scale_out), DTYPE_MIN, DTYPE_MAX)
                out_q = gl.convert_layout(
                    gl.reshape(quant_3d, [BLOCK_SIZE_M, BLOCK_SIZE_N]), xLayout2D
                )
                block_scales = gl.convert_layout(
                    gl.reshape(scale_out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), scaleLayout2D
                )

            # the quantized tile is what gets stored
            result = out_q

            # scale store
            if SHUFFLE:
                gl.amd.gfx1250.buffer_store(
                    block_scales.to(scale_ptr.dtype.element_ty),
                    scale_ptr,
                    bs_offs.to(gl.int32),  # exists
                    mask=scale_mask,
                )
            else:
                gl.amd.gfx1250.buffer_store(
                    block_scales.to(scale_ptr.dtype.element_ty),
                    scale_ptr + row.to(gl.int64) * scale_stride_m,
                    (
                        s_rows[:, None] * scale_stride_m
                        + g_offs[None, :] * scale_stride_n
                    ).to(gl.int32),
                    mask=scale_mask,
                )
        else:
            # no quant
            result = out

        # buffer store for a bit of perf uplift
        gl.amd.gfx1250.buffer_store(
            result.to(out_ptr.dtype.element_ty),
            out_ptr + row.to(gl.int64) * out_stride_m,
            store_offs,
            mask=row_mask[:, None] & mask[None, :],
        )

    # epilogue
    row = m_start + (ROWS_PER_PROG - 1) * BLOCK_SIZE_M
    gate_tile = smem.index(2 * (ROWS_PER_PROG - 1))
    up_tile = smem.index(2 * (ROWS_PER_PROG - 1) + 1)

    # this tile's absolute row ids and the rows that are in range
    abs_rows = row + m_ids
    row_mask = abs_rows < M

    if HAVE_WEIGHTS:
        if WEIGHT_BROADCAST:
            w = gl.load(
                weights_ptr + abs_rows * weights_stride_m,
                mask=row_mask,
                other=0.0,
            )
        else:
            # buffer load weight, also gives slightly better perf
            w = gl.amd.gfx1250.buffer_load(
                weights_ptr + row.to(gl.int64) * weights_stride_m,
                (
                    m_ids[:, None] * weights_stride_m + cols[None, :] * weights_stride_n
                ).to(gl.int32),
                mask=row_mask[:, None] & mask[None, :],
                other=0.0,
                cache=cache_modifier,
            )

    # scale offsets and mask
    s_abs_rows = row + s_rows
    scale_mask = (s_abs_rows < M)[:, None] & (g_offs < num_bs)[None, :]

    # preshuffled scale addressing, matching the triton shuffle layout
    if SHUFFLE:
        bs_r = s_abs_rows[:, None]
        bs_g = g_offs[None, :]
        bs_offs_0 = bs_r // 32
        bs_offs_1 = bs_r % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_g // 8
        bs_offs_4 = bs_g % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * SCALE_N_PAD
        )
    else:
        bs_offs = 0  # not needed

    # last tile, nothing left to overlap with, so drain fully
    gl.amd.gfx1250.tdm.async_wait(0)

    # wait for tile
    gate = gate_tile.load(xLayout2D).to(gl.float32)
    up = up_tile.load(xLayout2D).to(gl.float32)

    # clamp
    if HAVE_SWIGLU_CLAMP:
        up = gl.clamp(up, -swiglu_limit, swiglu_limit)
        gate = gl.minimum(gate, swiglu_limit)

    # apply act(gate)*up
    out = _apply_activation_from_str(gate, ACTIVATION) * up

    # apply weights; broadcast form is one value per row of the tile
    if HAVE_WEIGHTS:
        if WEIGHT_BROADCAST:
            out = out * w.to(gl.float32)[:, None]
        else:
            out = out * w.to(gl.float32)

    # group quant and store
    if HAS_QUANT:
        if SCALE_FMT == "ue8m0":
            out_3d = gl.reshape(out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
            abs_3d = gl.maximum(out_3d, -out_3d)
            max_val = gl.max(abs_3d, axis=2, keep_dims=True)  # [BM, NQB, 1]
            dequant_scale = max_val / DTYPE_MAX
            dequant_scale_exp = (
                dequant_scale.to(gl.uint32, bitcast=True) + 0x007FFFFF
            ) & 0x7F800000
            dequant_scale_rounded = dequant_scale_exp.to(gl.float32, bitcast=True)
            quant_scale = gl.where(
                dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
            )
            quant_tensor = out_3d * quant_scale
            out_q = gl.convert_layout(
                gl.reshape(quant_tensor, [BLOCK_SIZE_M, BLOCK_SIZE_N]), xLayout2D
            )
            scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)
            block_scales = gl.convert_layout(
                gl.reshape(scale_exp, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), scaleLayout2D
            )
        else:
            # fp8 quant
            out_3d = gl.reshape(out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
            abs_3d = gl.maximum(out_3d, -out_3d)
            max_val = gl.maximum(gl.max(abs_3d, axis=2, keep_dims=True), 1e-10)
            scale_out = max_val / DTYPE_MAX
            quant_3d = gl.clamp(out_3d * (1.0 / scale_out), DTYPE_MIN, DTYPE_MAX)
            out_q = gl.convert_layout(
                gl.reshape(quant_3d, [BLOCK_SIZE_M, BLOCK_SIZE_N]), xLayout2D
            )
            block_scales = gl.convert_layout(
                gl.reshape(scale_out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), scaleLayout2D
            )

        result = out_q

        # scale store
        if SHUFFLE:
            gl.amd.gfx1250.buffer_store(
                block_scales.to(scale_ptr.dtype.element_ty),
                scale_ptr,
                bs_offs.to(gl.int32),
                mask=scale_mask,
            )
        else:
            gl.amd.gfx1250.buffer_store(
                block_scales.to(scale_ptr.dtype.element_ty),
                scale_ptr + row.to(gl.int64) * scale_stride_m,
                (
                    s_rows[:, None] * scale_stride_m + g_offs[None, :] * scale_stride_n
                ).to(gl.int32),
                mask=scale_mask,
            )
    else:
        # no quant
        result = out

    # buffer store for a bit of perf uplift over TDM store
    gl.amd.gfx1250.buffer_store(
        result.to(out_ptr.dtype.element_ty),
        out_ptr + row.to(gl.int64) * out_stride_m,
        store_offs,
        mask=row_mask[:, None] & mask[None, :],
    )

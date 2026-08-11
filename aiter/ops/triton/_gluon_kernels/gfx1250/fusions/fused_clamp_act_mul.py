# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) port of triton _fused_clamp_silu_mul_kernel for GFX1250
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.activation import _apply_activation_from_str
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Human-readable repr for the compiled kernel: lists the constexpr keys that
# identify a unique specialization (shown in traces / cache keys).
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
    inp_ptr,             # *in*  [M, 2*n_half] gate|up input
    out_ptr,             # *out* [M, n_half] result (native dtype or FP8)
    scale_ptr,           # *out* [M, num_blocks] block scales (only if HAS_QUANT)
    weights_ptr,         # *in*  [M,1] or [M,n_half] weights (only if HAVE_WEIGHTS)
    M,                   # number of token rows
    n_half,              # N — half the input's inner dim (gate/up width)
    inp_stride_m,        # inp row stride
    inp_stride_n,        # inp col stride
    out_stride_m,        # out row stride
    out_stride_n,        # out col stride
    scale_stride_m,      # scale row stride
    scale_stride_n,      # scale col stride
    weights_stride_m,    # weights row stride
    weights_stride_n,    # weights col stride
    swiglu_limit,        # clamp bound (used only when HAVE_SWIGLU_CLAMP)
    ROWS_PER_PROG: gl.constexpr,    # ring buffers = rows per program = loop trips
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    QUANT_BLOCK_SIZE: gl.constexpr,
    SCALE_FMT: gl.constexpr,
    DTYPE_MAX: gl.constexpr,      # +max of the quant dtype (for scale calc)
    DTYPE_MIN: gl.constexpr,      # -max of the quant dtype
    HAVE_WEIGHTS: gl.constexpr,
    WEIGHT_BROADCAST: gl.constexpr,
    HAVE_SWIGLU_CLAMP: gl.constexpr,
    HAS_QUANT: gl.constexpr,
    ACTIVATION: gl.constexpr,     # "silu" | "gelu" | "gelu_tanh"
    SHUFFLE: gl.constexpr,        # write scales in preshuffled tiled layout
    SCALE_N_PAD: gl.constexpr,    # padded scale-col count (shuffle addressing)
    num_warps: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    # constants
    NUM_N_Q_GROUPS: gl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE  # quant groups per row

    PAD_INTERVAL: gl.constexpr = min(
        2 * BLOCK_SIZE_N, 1024 // inp_ptr.dtype.element_ty.itemsize
    )
    PAD_AMOUNT: gl.constexpr = 4 // inp_ptr.dtype.element_ty.itemsize
    shared_tdm_layout_2d: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[PAD_INTERVAL, PAD_AMOUNT]],
        [BLOCK_SIZE_M, 2 * BLOCK_SIZE_N],
        [1, 0],
    )
    pid = gl.program_id(0)
    m_start = pid * ROWS_PER_PROG

    # gate + up
    gate_up_smem = gl.allocate_shared_memory(
        inp_ptr.dtype.element_ty, [BLOCK_SIZE_M, 1, 2 * BLOCK_SIZE_N], shared_tdm_layout_2d
    ) # rows per prog tied to LDS, loop should be decoupled TODO

    inp_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=inp_ptr,
        shape=[M, 2 * n_half], # could be M - m_start TODO
        strides=[inp_stride_m, inp_stride_n],
        block_shape=[1, 2 * BLOCK_SIZE_N],
        layout=shared_tdm_layout_2d,
    )

    smem = gl.allocate_shared_memory(
        inp_desc.dtype, shape=[ROWS_PER_PROG] + inp_desc.block_shape, layout=shared_tdm_layout_2d
    )

    gLayout2D: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    
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
    if BLOCK_SIZE_M > 2:
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
    sLayout2D: gl.constexpr = gl.DistributedLinearLayout(
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
    row_layout: gl.constexpr = gl.SliceLayout(0, gLayout2D)         # [BLOCK_SIZE_N] row slice
    row_scale_layout: gl.constexpr = gl.SliceLayout(0, sLayout2D)   # [NUM_N_Q_GROUPS] scale slice
    m_scale_layout: gl.constexpr = gl.SliceLayout(1, sLayout2D)      # [BLOCK_SIZE_M] scale rows

    # setup + setup store
    offs = gl.arange(0, BLOCK_SIZE_N, layout=row_layout).to(gl.int64)
    mask = offs < n_half
    num_bs = gl.cdiv(n_half, QUANT_BLOCK_SIZE)
    g_offs = gl.arange(0, NUM_N_Q_GROUPS, layout=row_scale_layout)

    s_rows = gl.arange(0, BLOCK_SIZE_M, layout=m_scale_layout)        # [BLOCK_SIZE_M]

    m_ids = gl.arange(0, BLOCK_SIZE_M, layout=m_layout)              # [BLOCK_SIZE_M]
    store_offs = (
        m_ids[:, None] * out_stride_m + offs[None, :] * out_stride_n
    ).to(gl.int32)                                                  # [BM, BN]

    # prologue: first tile
    gl.amd.gfx1250.tdm.async_load(inp_desc, [m_start, 0], smem.index(0))

    for i in range(ROWS_PER_PROG - 1):
        # prefetch next tile
        gl.amd.gfx1250.tdm.async_load(
            inp_desc, [m_start + (i + 1) * BLOCK_SIZE_M, 0], smem.index(i + 1)
        )

        row = m_start + i * BLOCK_SIZE_M
        tile = smem.index(i)

        # weights load
        if HAVE_WEIGHTS:
            if WEIGHT_BROADCAST:
                w = gl.load(weights_ptr + row * weights_stride_m)  # scalar applied to all out
            else:
                # buffer load weight, also gives slightly better perf
                w = gl.amd.gfx1250.buffer_load(
                    weights_ptr + row.to(gl.int64) * weights_stride_m,
                    (offs * weights_stride_n).to(gl.int32),
                    mask=mask, other=0.0, cache=cache_modifier,
                )

        s_abs_rows = row + s_rows                             # [BM] absolute rows
        scale_mask = (s_abs_rows < M)[:, None] & (g_offs < num_bs)[None, :]

        if SHUFFLE:
            bs_r = s_abs_rows[:, None]     # [BM, 1]
            bs_g = g_offs[None, :]         # [1, NUM_N_Q_GROUPS]
            bs_offs_0 = bs_r // 32         # row-tile of 32
            bs_offs_1 = bs_r % 32          # position within the 32-row tile
            bs_offs_2 = bs_offs_1 % 16     # sub-position within 16
            bs_offs_1 = bs_offs_1 // 16    # which half of the 32 (0/1)
            bs_offs_3 = bs_g // 8          # block-col tile of 8
            bs_offs_4 = bs_g % 8           # position within the 8-col tile
            bs_offs_5 = bs_offs_4 % 4      # sub-position within 4
            bs_offs_4 = bs_offs_4 // 4     # which half of the 8 (0/1)
            bs_offs = (                    # weave the sub-indices into the tiled offset
                bs_offs_1
                + bs_offs_4 * 2
                + bs_offs_2 * 2 * 2
                + bs_offs_5 * 2 * 2 * 16
                + bs_offs_3 * 2 * 2 * 16 * 4
                + bs_offs_0 * 2 * 16 * SCALE_N_PAD
            )                              # [BM, NUM_N_Q_GROUPS]
        else:
            bs_offs = 0 # not needed

        gl.amd.gfx1250.tdm.async_wait(1)

        gate_up = tile.load(gLayout2D).to(gl.float32)   # [BLOCK_SIZE_M, 2*BLOCK_SIZE_N]
        gate, up = gl.split(
            gl.reshape(gate_up, [BLOCK_SIZE_M, 2, BLOCK_SIZE_N]).permute(0, 2, 1)
        )   # each [BLOCK_SIZE_M, BLOCK_SIZE_N]

        # clamp
        if HAVE_SWIGLU_CLAMP:
            up = gl.clamp(up, -swiglu_limit, swiglu_limit)   # clamp up to [-lim, lim]
            gate = gl.minimum(gate, swiglu_limit)            # clamp gate to <= lim

        # act(gate) * up
        out = _apply_activation_from_str(gate, ACTIVATION) * up

        out = gl.convert_layout(out, gLayout2D)

        # apply weights
        if HAVE_WEIGHTS:
            out = out * w.to(gl.float32)

        # group quant and store
        if HAS_QUANT:
            if SCALE_FMT == "ue8m0":
                # mxfp8, reduce over inner QUANT_BLOCK_SIZE axis.
                out_3d = gl.reshape(out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
                abs_3d = gl.maximum(out_3d, -out_3d)
                max_val = gl.max(abs_3d, axis=2, keep_dims=True)   # [BM, NQB, 1]
                dequant_scale = max_val / DTYPE_MAX
                # ROUND_UP to a power of two via the fp32 exponent field.
                dequant_scale_exp = (
                    dequant_scale.to(gl.uint32, bitcast=True) + 0x007FFFFF
                ) & 0x7F800000
                dequant_scale_rounded = dequant_scale_exp.to(gl.float32, bitcast=True)
                quant_scale = gl.where(                       # reciprocal, guard 0
                    dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
                )
                quant_tensor = out_3d * quant_scale  # scale into fp8 range
                out_q = gl.convert_layout(
                    gl.reshape(quant_tensor, [BLOCK_SIZE_M, BLOCK_SIZE_N]), gLayout2D
                )
                scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)   # [BM, NQB, 1]
                block_scales = gl.convert_layout(
                    gl.reshape(scale_exp, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), sLayout2D
                )
            else:
                # fp8 quant
                out_3d = gl.reshape(out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
                abs_3d = gl.maximum(out_3d, -out_3d)
                max_val = gl.maximum(
                    gl.max(abs_3d, axis=2, keep_dims=True), 1e-10
                )  # [BM, NQB, 1]
                scale_out = max_val / DTYPE_MAX  # dequant (block) scale
                quant_3d = gl.clamp(out_3d * (1.0 / scale_out), DTYPE_MIN, DTYPE_MAX)
                out_q = gl.convert_layout(
                    gl.reshape(quant_3d, [BLOCK_SIZE_M, BLOCK_SIZE_N]), gLayout2D
                )
                block_scales = gl.convert_layout(
                    gl.reshape(scale_out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), sLayout2D
                )

            result = out_q

            if SHUFFLE:
                gl.store(
                    scale_ptr + bs_offs, # exists
                    block_scales.to(scale_ptr.dtype.element_ty),
                    mask=scale_mask,
                )
            else:
                gl.store(
                    scale_ptr
                    + s_abs_rows[:, None] * scale_stride_m
                    + g_offs[None, :] * scale_stride_n,
                    block_scales.to(scale_ptr.dtype.element_ty),
                    mask=scale_mask,
                )
        else:
            # no quant
            result = out

        # buffer store for a bit of perf uplift
        gl.amd.gfx1250.buffer_store(
            result.to(out_ptr.dtype.element_ty),
            out_ptr + row.to(gl.int64) * out_stride_m,
            store_col_offs,
            mask=mask & (row < M),
        )

    row = m_start + (ROWS_PER_PROG - 1) * BLOCK_SIZE_M
    tile = smem.index(ROWS_PER_PROG - 1)

    # weights load
    if HAVE_WEIGHTS:
        if WEIGHT_BROADCAST:
            w = gl.load(weights_ptr + row * weights_stride_m)  # scalar applied to all out
        else:
            # buffer load weight, also gives slightly better perf
            w = gl.amd.gfx1250.buffer_load(
                weights_ptr + row.to(gl.int64) * weights_stride_m,
                (offs * weights_stride_n).to(gl.int32),
                mask=mask, other=0.0, cache=cache_modifier,
            )

    s_abs_rows = row + s_rows                             # [BM] absolute rows
    scale_mask = (s_abs_rows < M)[:, None] & (g_offs < num_bs)[None, :]

    if SHUFFLE:
        bs_r = s_abs_rows[:, None]     # [BM, 1]
        bs_g = g_offs[None, :]         # [1, NUM_N_Q_GROUPS]
        bs_offs_0 = bs_r // 32         # row-tile of 32
        bs_offs_1 = bs_r % 32          # position within the 32-row tile
        bs_offs_2 = bs_offs_1 % 16     # sub-position within 16
        bs_offs_1 = bs_offs_1 // 16    # which half of the 32 (0/1)
        bs_offs_3 = bs_g // 8          # block-col tile of 8
        bs_offs_4 = bs_g % 8           # position within the 8-col tile
        bs_offs_5 = bs_offs_4 % 4      # sub-position within 4
        bs_offs_4 = bs_offs_4 // 4     # which half of the 8 (0/1)
        bs_offs = (                    # weave the sub-indices into the tiled offset
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * SCALE_N_PAD
        )                              # [BM, NUM_N_Q_GROUPS]
    else:
        bs_offs = 0 # not needed

    gl.amd.gfx1250.tdm.async_wait(0)

    gate_up = tile.load(gLayout2D).to(gl.float32)   # [BLOCK_SIZE_M, 2*BLOCK_SIZE_N]
    gate, up = gl.split(
        gl.reshape(gate_up, [BLOCK_SIZE_M, 2, BLOCK_SIZE_N]).permute(0, 2, 1)
    )   # each [BLOCK_SIZE_M, BLOCK_SIZE_N]

    # clamp
    if HAVE_SWIGLU_CLAMP:
        up = gl.clamp(up, -swiglu_limit, swiglu_limit)   # clamp up to [-lim, lim]
        gate = gl.minimum(gate, swiglu_limit)            # clamp gate to <= lim

    # act(gate) * up
    out = _apply_activation_from_str(gate, ACTIVATION) * up

    out = gl.convert_layout(out, gLayout2D)

    # apply weights
    if HAVE_WEIGHTS:
        out = out * w.to(gl.float32)

    # group quant and store
    if HAS_QUANT:
        if SCALE_FMT == "ue8m0":
            # mxfp8, reduce over inner QUANT_BLOCK_SIZE axis.
            out_3d = gl.reshape(out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
            abs_3d = gl.maximum(out_3d, -out_3d)
            max_val = gl.max(abs_3d, axis=2, keep_dims=True)   # [BM, NQB, 1]
            dequant_scale = max_val / DTYPE_MAX
            # ROUND_UP to a power of two via the fp32 exponent field.
            dequant_scale_exp = (
                dequant_scale.to(gl.uint32, bitcast=True) + 0x007FFFFF
            ) & 0x7F800000
            dequant_scale_rounded = dequant_scale_exp.to(gl.float32, bitcast=True)
            quant_scale = gl.where(                       # reciprocal, guard 0
                dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
            )
            quant_tensor = out_3d * quant_scale  # scale into fp8 range
            out_q = gl.convert_layout(
                gl.reshape(quant_tensor, [BLOCK_SIZE_M, BLOCK_SIZE_N]), gLayout2D
            )
            scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)   # [BM, NQB, 1]
            block_scales = gl.convert_layout(
                gl.reshape(scale_exp, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), sLayout2D
            )
        else:
            # fp8 quant
            out_3d = gl.reshape(out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
            abs_3d = gl.maximum(out_3d, -out_3d)
            max_val = gl.maximum(
                gl.max(abs_3d, axis=2, keep_dims=True), 1e-10
            )  # [BM, NQB, 1]
            scale_out = max_val / DTYPE_MAX  # dequant (block) scale
            quant_3d = gl.clamp(out_3d * (1.0 / scale_out), DTYPE_MIN, DTYPE_MAX)
            out_q = gl.convert_layout(
                gl.reshape(quant_3d, [BLOCK_SIZE_M, BLOCK_SIZE_N]), gLayout2D
            )
            block_scales = gl.convert_layout(
                gl.reshape(scale_out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), sLayout2D
            )

        result = out_q

        if SHUFFLE:
            gl.store(
                scale_ptr + bs_offs, # exists
                block_scales.to(scale_ptr.dtype.element_ty),
                mask=scale_mask,
            )
        else:
            gl.store(
                scale_ptr
                + s_abs_rows[:, None] * scale_stride_m
                + g_offs[None, :] * scale_stride_n,
                block_scales.to(scale_ptr.dtype.element_ty),
                mask=scale_mask,
            )
    else:
        # no quant
        result = out

    # buffer store for a bit of perf uplift
    # 2D mask: row tail (this tile's rows past M) x col tail (cols past n_half)
    gl.amd.gfx1250.buffer_store(
        result.to(out_ptr.dtype.element_ty),
        out_ptr + row.to(gl.int64) * out_stride_m,
        store_offs,
        mask=((row + m_ids) < M)[:, None] & mask[None, :],
    )

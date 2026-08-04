# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) port of ``_fused_clamp_silu_mul_kernel`` — 2D [BLOCK_SIZE_M,
BLOCK_SIZE_N] tile (same structure as ``norm/fused_rmsnorm_add.py``).

Semantics mirror the Triton reference in
``aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py``: per token row
of an ``[M, 2*N]`` input (gate = first ``N`` cols, up = second ``N``): optional
SwiGLU clamp -> ``act(gate) * up`` -> optional weights -> optional
per-``QUANT_BLOCK_SIZE`` FP8 group quant, with an optional shuffled scale store.

Each program owns a ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` tile (grid = cdiv(M,
BLOCK_SIZE_M)): it 2D-TDM-loads the gate and up halves of BLOCK_SIZE_M rows into
LDS, reads them into a 2D register tensor (``gLayout2D``), computes
``act(gate)*up`` (+ weights), quantizes per-``QUANT_BLOCK_SIZE`` group via a
``[BM, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE]`` reshape reduced over ``axis=2``, and
2D-TDM-stores the result. Descriptor shapes ``[M, n_half]`` bound the row (tail)
and column (power-of-two) overhang to OOB zero-fill.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.activation import _apply_activation_from_str
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Human-readable repr for the compiled kernel: lists the constexpr keys that
# identify a unique specialization (shown in traces / cache keys).
_GLUON_REPR_KEYS = [
    "BLOCK_SIZE_M",
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
    BLOCK_SIZE_M: gl.constexpr,   # rows per tile
    BLOCK_SIZE_N: gl.constexpr,   # cols per tile (= next_pow2(n_half))
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
    NUM_N_Q_GROUPS: gl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE  # quant groups per row

    # 2D register layout for the [BM, BN] data tile (mirrors rmsnorm's gLayout2D:
    # lanes + warps spread over N, rows in the rep dimension).
    gLayout2D: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, max(1, BLOCK_SIZE_N // (num_warps * 32))],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    # 2D layout for the [BM, G] scale tile.
    sLayout2D: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, max(1, NUM_N_Q_GROUPS // (num_warps * 32))],
        threads_per_warp=[1, 32],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    rowN: gl.constexpr = gl.SliceLayout(1, gLayout2D)   # [BM] row-index vector (data tile)
    colN: gl.constexpr = gl.SliceLayout(0, gLayout2D)   # [BN] col-index vector
    rowS: gl.constexpr = gl.SliceLayout(1, sLayout2D)   # [BM] row-index vector (scale tile)
    colS: gl.constexpr = gl.SliceLayout(0, sLayout2D)   # [G]  group-index vector
    # Identity (unpadded) LDS staging — same as rmsnorm and the other elementwise
    # TDM kernels (padded layouts only lower when a matrix-core layout is on the
    # register side, which this elementwise BlockedLayout tile has none of).
    shared2D: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    pid = gl.program_id(0)
    m_start = pid * BLOCK_SIZE_M

    # --- 2D TDM descriptors over the gate half, up half, and output ([M, n_half]
    # bounds both the row tail and the power-of-two column overhang to zero-fill).
    gate_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        inp_ptr, [M, n_half], [inp_stride_m, inp_stride_n],
        [BLOCK_SIZE_M, BLOCK_SIZE_N], shared2D,
    )
    up_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        inp_ptr + n_half * inp_stride_n, [M, n_half], [inp_stride_m, inp_stride_n],
        [BLOCK_SIZE_M, BLOCK_SIZE_N], shared2D,
    )
    out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        out_ptr, [M, n_half], [out_stride_m, out_stride_n],
        [BLOCK_SIZE_M, BLOCK_SIZE_N], shared2D,
    )
    gate_smem = gl.allocate_shared_memory(
        inp_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], shared2D
    )
    up_smem = gl.allocate_shared_memory(
        inp_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], shared2D
    )
    out_smem = gl.allocate_shared_memory(
        out_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], shared2D
    )

    gl.amd.gfx1250.tdm.async_load(gate_desc, [m_start, 0], gate_smem)
    gl.amd.gfx1250.tdm.async_load(up_desc, [m_start, 0], up_smem)

    # row / col index vectors + tile mask (rows past M, cols past n_half)
    rows = m_start + gl.arange(0, BLOCK_SIZE_M, layout=rowN)   # [BM]
    cols = gl.arange(0, BLOCK_SIZE_N, layout=colN)             # [BN]
    row_ok = rows < M
    col_ok = cols < n_half
    tile_mask = row_ok[:, None] & col_ok[None, :]              # [BM, BN]

    # weights (loaded while the gate/up DMA is in flight)
    if HAVE_WEIGHTS:
        if WEIGHT_BROADCAST:
            # [M,1] broadcast: build a [BM, BN] tile where every column reads the
            # row's single weight (col 0), so `out * w` is a same-layout multiply
            # (avoids a broadcast between gLayout2D and a [BM,1] linear layout).
            wptr = (
                weights_ptr
                + rows[:, None] * weights_stride_m
                + (cols * 0)[None, :]
            )
            w = gl.load(
                wptr, mask=tile_mask, other=0.0, cache_modifier=cache_modifier
            ).to(gl.float32)                                  # [BM, BN]
        else:
            wptr = (
                weights_ptr
                + rows[:, None] * weights_stride_m
                + cols[None, :] * weights_stride_n
            )
            w = gl.load(
                wptr, mask=tile_mask, other=0.0, cache_modifier=cache_modifier
            ).to(gl.float32)                                  # [BM, BN]

    gl.amd.gfx1250.tdm.async_wait(1)
    gate = gate_smem.load(gLayout2D).to(gl.float32)           # [BM, BN]
    gl.amd.gfx1250.tdm.async_wait(0)
    up = up_smem.load(gLayout2D).to(gl.float32)               # [BM, BN]

    # clamp
    if HAVE_SWIGLU_CLAMP:
        up = gl.clamp(up, -swiglu_limit, swiglu_limit)
        gate = gl.minimum(gate, swiglu_limit)

    # act(gate) * up
    out = _apply_activation_from_str(gate, ACTIVATION) * up   # [BM, BN]

    # weights
    if HAVE_WEIGHTS:
        out = out * w

    # group quant and store
    if HAS_QUANT:
        if SCALE_FMT == "ue8m0":
            out_3d = gl.reshape(
                out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE]
            )
            abs_3d = gl.maximum(out_3d, -out_3d)
            max_val = gl.max(abs_3d, axis=2, keep_dims=True)      # [BM, G, 1]
            dequant_scale = max_val / DTYPE_MAX
            dequant_scale_exp = (
                dequant_scale.to(gl.uint32, bitcast=True) + 0x007FFFFF
            ) & 0x7F800000
            dequant_scale_rounded = dequant_scale_exp.to(gl.float32, bitcast=True)
            quant_scale = gl.where(
                dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
            )
            out_q = gl.convert_layout(
                gl.reshape(out_3d * quant_scale, [BLOCK_SIZE_M, BLOCK_SIZE_N]),
                gLayout2D,
            )
            scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)   # [BM, G, 1]
            block_scales = gl.convert_layout(
                gl.reshape(scale_exp, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), sLayout2D
            )
        else:
            out_3d = gl.reshape(
                out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE]
            )
            abs_3d = gl.maximum(out_3d, -out_3d)
            max_val = gl.maximum(
                gl.max(abs_3d, axis=2, keep_dims=True), 1e-10
            )                                                    # [BM, G, 1]
            scale_out = max_val / DTYPE_MAX
            quant_3d = gl.clamp(out_3d * (1.0 / scale_out), DTYPE_MIN, DTYPE_MAX)
            out_q = gl.convert_layout(
                gl.reshape(quant_3d, [BLOCK_SIZE_M, BLOCK_SIZE_N]), gLayout2D
            )
            block_scales = gl.convert_layout(
                gl.reshape(scale_out, [BLOCK_SIZE_M, NUM_N_Q_GROUPS]), sLayout2D
            )

        # 2D TDM store of the quantized output tile.
        out_smem.store(out_q.to(out_ptr.dtype.element_ty))
        if num_warps > 1:
            gl.barrier()
        gl.amd.gfx1250.tdm.async_store(out_desc, [m_start, 0], out_smem)

        # scale store (direct 2D gl.store, keyed on absolute row/group).
        num_bs = gl.cdiv(n_half, QUANT_BLOCK_SIZE)
        srows = m_start + gl.arange(0, BLOCK_SIZE_M, layout=rowS)   # [BM]
        g = gl.arange(0, NUM_N_Q_GROUPS, layout=colS)              # [G]
        scale_mask = (srows[:, None] < M) & (g[None, :] < num_bs)  # [BM, G]
        if SHUFFLE:
            # Preshuffled scale store: same tiled index math as the Triton
            # reference, per (row, group) — row is a [BM,1] vector, g a [1,G].
            r0 = srows[:, None]
            bs_offs_0 = r0 // 32
            bs_offs_1 = r0 % 32
            bs_offs_2 = bs_offs_1 % 16
            bs_offs_1 = bs_offs_1 // 16
            gg = g[None, :]
            bs_offs_3 = gg // 8
            bs_offs_4 = gg % 8
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
            gl.store(
                scale_ptr + bs_offs,
                block_scales.to(scale_ptr.dtype.element_ty),
                mask=scale_mask,
            )
        else:
            sptr = (
                scale_ptr
                + srows[:, None] * scale_stride_m
                + g[None, :] * scale_stride_n
            )
            gl.store(
                sptr, block_scales.to(scale_ptr.dtype.element_ty), mask=scale_mask
            )
        gl.amd.gfx1250.tdm.async_wait(0)
    else:
        # no quant: 2D TDM store of the native-dtype tile.
        out_smem.store(gl.convert_layout(out, gLayout2D).to(out_ptr.dtype.element_ty))
        if num_warps > 1:
            gl.barrier()
        gl.amd.gfx1250.tdm.async_store(out_desc, [m_start, 0], out_smem)
        gl.amd.gfx1250.tdm.async_wait(0)

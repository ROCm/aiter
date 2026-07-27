# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) port of ``_fused_clamp_silu_mul_kernel`` — software-pipelined.

Semantics mirror the Triton reference in
``aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py``: per token row
of an ``[M, 2*N]`` input (gate = first ``N`` cols, up = second ``N``): optional
SwiGLU clamp -> ``act(gate) * up`` -> optional weights -> optional
per-``QUANT_BLOCK_SIZE`` FP8 group quant, with an optional shuffled scale store.

Pipeline (Level 2): the launch is persistent — each program walks a contiguous
strip of ``ROWS_PER_PROGRAM`` rows and keeps ``NUM_BUFFERS`` TDM loads in flight,
so the gate/up DMA of a future row overlaps the compute of the current one. The
per-row ``async_wait`` count is derived at compile time from how many rows are
still prefetched ahead, so the wait no longer stalls on a cold load. The output
is written with a direct ``gl.store`` (no TDM store staging) to keep the TDM
completion FIFO tracking only the pipelined loads.

Gluon specifics: tensors carry explicit ``BlockedLayout``s (``row_layout`` for the
data vector, ``row_scale_layout`` for the scale vector); ``tl.ravel`` becomes
``gl.reshape`` back to 1D + ``gl.convert_layout`` onto the store layout;
``tl.abs`` becomes ``gl.maximum(x, -x)``.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.activation import _apply_activation_from_str
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Human-readable repr for the compiled kernel: lists the constexpr keys that
# identify a unique specialization (shown in traces / cache keys).
_GLUON_REPR_KEYS = [
    "BLOCK_SIZE_N",
    "QUANT_BLOCK_SIZE",
    "SCALE_FMT",
    "HAVE_WEIGHTS",
    "WEIGHT_BROADCAST",
    "HAVE_SWIGLU_CLAMP",
    "HAS_QUANT",
    "ROWS_PER_PROGRAM",
    "NUM_BUFFERS",
    "num_warps",
    "cache_modifier",
]

_fused_clamp_silu_mul_repr = make_kernel_repr(
    "_fused_clamp_silu_mul_gfx1250_kernel", _GLUON_REPR_KEYS
)


@gluon.jit
def _issue_row_loads(
    inp_ptr,
    row,
    inp_stride_m,
    inp_stride_n,
    n_half,
    BLOCK_SIZE_N: gl.constexpr,
    layout: gl.constexpr,
    gate_buf,
    up_buf,
):
    """Issue the two async TDM loads (gate + up) for a single row into the given
    LDS buffers. ``row`` is clamped by the caller to a valid row; the descriptor
    shape ``[n_half]`` bounds the BLOCK_SIZE_N overhang (OOB -> zero-fill)."""
    base = inp_ptr + row * inp_stride_m
    gate_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=base,
        shape=[n_half],
        strides=[inp_stride_n],
        block_shape=[BLOCK_SIZE_N],
        layout=layout,
    )
    up_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=base + n_half * inp_stride_n,
        shape=[n_half],
        strides=[inp_stride_n],
        block_shape=[BLOCK_SIZE_N],
        layout=layout,
    )
    gl.amd.gfx1250.tdm.async_load(gate_desc, [0], gate_buf)
    gl.amd.gfx1250.tdm.async_load(up_desc, [0], up_buf)


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
    ROWS_PER_PROGRAM: gl.constexpr,  # contiguous rows handled per program
    NUM_BUFFERS: gl.constexpr,       # TDM load buffers kept in flight
    num_warps: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    # constants
    NUM_N_Q_GROUPS: gl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE  # quant groups per row
    NB: gl.constexpr = NUM_BUFFERS
    # 1D layouts
    row_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[max(1, BLOCK_SIZE_N // (num_warps * 32))],
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )
    row_scale_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[max(1, NUM_N_Q_GROUPS // (num_warps * 32))],
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )
    shared_tdm_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])

    pid = gl.program_id(0)
    row_base = pid * ROWS_PER_PROGRAM

    offs = gl.arange(0, BLOCK_SIZE_N, layout=row_layout)
    mask = offs < n_half
    num_bs = gl.cdiv(n_half, QUANT_BLOCK_SIZE)

    # NUM_BUFFERS-deep LDS staging for the pipelined gate/up loads.
    gate_smem = gl.allocate_shared_memory(
        inp_ptr.dtype.element_ty, [NB, BLOCK_SIZE_N], shared_tdm_layout
    )
    up_smem = gl.allocate_shared_memory(
        inp_ptr.dtype.element_ty, [NB, BLOCK_SIZE_N], shared_tdm_layout
    )

    # Prologue: kick off the first NB-1 rows' loads. Rows are clamped to M-1 so an
    # out-of-range strip tail still issues a valid (redundant) load; those rows are
    # never stored. The clamp keeps issue/wait accounting fully compile-time.
    for i in gl.static_range(NB - 1):
        r = row_base + i
        r = gl.minimum(r, M - 1)
        _issue_row_loads(
            inp_ptr, r, inp_stride_m, inp_stride_n, n_half,
            BLOCK_SIZE_N, shared_tdm_layout,
            gate_smem.index(i % NB), up_smem.index(i % NB),
        )

    for k in gl.static_range(ROWS_PER_PROGRAM):
        # Prefetch the row NB-1 ahead (only if it's within this program's strip).
        if k + NB - 1 < ROWS_PER_PROGRAM:
            rp = gl.minimum(row_base + k + (NB - 1), M - 1)
            _issue_row_loads(
                inp_ptr, rp, inp_stride_m, inp_stride_n, n_half,
                BLOCK_SIZE_N, shared_tdm_layout,
                gate_smem.index((k + NB - 1) % NB), up_smem.index((k + NB - 1) % NB),
            )

        # Wait until row k's loads are done: leave the rows still prefetched ahead
        # of k in flight (2 TDM ops each). AHEAD is compile-time.
        AHEAD: gl.constexpr = min(k + NB - 1, ROWS_PER_PROGRAM - 1) - k
        gl.amd.gfx1250.tdm.async_wait(2 * AHEAD)

        gate = gate_smem.index(k % NB).load(row_layout).to(gl.float32)
        up = up_smem.index(k % NB).load(row_layout).to(gl.float32)

        # clamp
        if HAVE_SWIGLU_CLAMP:
            up = gl.clamp(up, -swiglu_limit, swiglu_limit)
            gate = gl.minimum(gate, swiglu_limit)

        # act(gate) * up
        out = _apply_activation_from_str(gate, ACTIVATION) * up

        row = row_base + k

        # weights
        if HAVE_WEIGHTS:
            if WEIGHT_BROADCAST:
                w = gl.load(weights_ptr + row * weights_stride_m).to(gl.float32)
                out = out * w
            else:
                w = gl.load(
                    weights_ptr + row * weights_stride_m + offs * weights_stride_n,
                    mask=mask, other=0.0, cache_modifier=cache_modifier,
                ).to(gl.float32)
                out = out * w

        # Rows past the end of the tensor (strip tail) computed redundantly above
        # but never written.
        store_row = row < M

        # group quant and store (direct gl.store; no TDM store staging)
        if HAS_QUANT:
            if SCALE_FMT == "ue8m0":
                out_2d = gl.reshape(out, [NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
                abs_2d = gl.maximum(out_2d, -out_2d)
                max_val = gl.max(abs_2d, axis=1, keep_dims=True)
                dequant_scale = max_val / DTYPE_MAX
                dequant_scale_exp = (
                    dequant_scale.to(gl.uint32, bitcast=True) + 0x007FFFFF
                ) & 0x7F800000
                dequant_scale_rounded = dequant_scale_exp.to(gl.float32, bitcast=True)
                quant_scale = gl.where(
                    dequant_scale_rounded == 0, 0.0, 1.0 / dequant_scale_rounded
                )
                quant_tensor = out_2d * quant_scale
                out_q = gl.convert_layout(
                    gl.reshape(quant_tensor, [BLOCK_SIZE_N]), row_layout
                )
                scale_exp = (dequant_scale_exp >> 23).to(gl.uint8)
                block_scales = gl.convert_layout(
                    gl.reshape(scale_exp, [NUM_N_Q_GROUPS]), row_scale_layout
                )
            else:
                out_2d = gl.reshape(out, [NUM_N_Q_GROUPS, QUANT_BLOCK_SIZE])
                abs_2d = gl.maximum(out_2d, -out_2d)
                max_val = gl.maximum(
                    gl.max(abs_2d, axis=1, keep_dims=True), 1e-10
                )
                scale_out = max_val / DTYPE_MAX
                quant_2d = gl.clamp(out_2d * (1.0 / scale_out), DTYPE_MIN, DTYPE_MAX)
                out_q = gl.convert_layout(
                    gl.reshape(quant_2d, [BLOCK_SIZE_N]), row_layout
                )
                block_scales = gl.convert_layout(
                    gl.reshape(scale_out, [NUM_N_Q_GROUPS]), row_scale_layout
                )

            gl.store(
                out_ptr + row * out_stride_m + offs * out_stride_n,
                out_q.to(out_ptr.dtype.element_ty),
                mask=mask & store_row,
            )

            g_offs = gl.arange(0, NUM_N_Q_GROUPS, layout=row_scale_layout)
            if SHUFFLE:
                # Preshuffled scale store: identical index arithmetic to the Triton
                # reference, keyed on the absolute row (not the program id).
                bs_offs_0 = row // 32
                bs_offs_1 = row % 32
                bs_offs_2 = bs_offs_1 % 16
                bs_offs_1 = bs_offs_1 // 16
                bs_offs_3 = g_offs // 8
                bs_offs_4 = g_offs % 8
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
                    mask=(g_offs < num_bs) & store_row,
                )
            else:
                gl.store(
                    scale_ptr + row * scale_stride_m + g_offs * scale_stride_n,
                    block_scales.to(scale_ptr.dtype.element_ty),
                    mask=(g_offs < num_bs) & store_row,
                )
        else:
            gl.store(
                out_ptr + row * out_stride_m + offs * out_stride_n,
                out.to(out_ptr.dtype.element_ty),
                mask=mask & store_row,
            )

    # Drain any loads still in flight from the strip tail before exit.
    gl.amd.gfx1250.tdm.async_wait(0)

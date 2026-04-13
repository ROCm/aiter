# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def rms_norm_input_quant_fp8_kernel(
    X,
    W,
    B,
    Z,
    Y_quant,
    Scales,
    stride_x_row,
    stride_z_row,
    stride_y_row,
    stride_s_row,
    stride_s_g,
    M,
    N: tl.constexpr,
    eps,
    RMS_TILE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    FP8_MIN_SCALING_FACTOR: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask_1d = rows < M

    # --- Full-row RMS: accumulate sum of squares in float32 ---
    sumsq = tl.zeros([ROWS_PER_BLOCK], dtype=tl.float32)
    off = 0
    while off < N:
        cols = tl.arange(0, RMS_TILE) + off
        col_mask = cols < N
        mask = row_mask_1d[:, None] & col_mask[None, :]
        row_offsets = rows[:, None] * stride_x_row
        col_offsets = cols[None, :]
        X_base = X + row_offsets + col_offsets
        x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            Z_base = Z + rows[:, None] * stride_z_row + col_offsets
            z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                x *= z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                x *= tl.sigmoid(z)
        xbar = tl.where(mask, x, 0.0)
        sumsq += tl.sum(xbar * xbar, axis=1)
        off += RMS_TILE

    var = sumsq / N
    rstd = tl.rsqrt(var + eps)

    # --- Per-group: normalize (when NORM_BEFORE_GATE), linear, optional gate, FP8 ---
    for g in range(NUM_GROUPS):
        col0 = g * GROUP_SIZE
        cols = tl.arange(0, BLOCK_G) + col0
        col_mask = cols < N
        mask = row_mask_1d[:, None] & col_mask[None, :]
        row_offsets = rows[:, None] * stride_x_row
        col_offsets = cols[None, :]
        X_base = X + row_offsets + col_offsets
        x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

        if HAS_Z and not NORM_BEFORE_GATE:
            Z_base = Z + rows[:, None] * stride_z_row + col_offsets
            z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                x *= z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                x *= tl.sigmoid(z)

        x_hat = x * rstd[:, None]

        w_mask = cols < N
        w = tl.load(W + cols, mask=w_mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + cols, mask=w_mask, other=0.0).to(tl.float32)
            y = x_hat * w[None, :] + b[None, :]
        else:
            y = x_hat * w[None, :]

        if HAS_Z and NORM_BEFORE_GATE:
            Z_base = Z + rows[:, None] * stride_z_row + col_offsets
            z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                y *= z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                y *= tl.sigmoid(z)

        abs_y = tl.where(mask, tl.abs(y), 0.0)
        absmax = tl.max(abs_y, axis=1)
        scales_raw = absmax / FP8_MAX
        if USE_UE8M0:
            scales_raw = tl.exp2(tl.ceil(tl.log2(scales_raw)))
        scales = tl.maximum(scales_raw, FP8_MIN_SCALING_FACTOR)

        y_scaled = y / scales[:, None]
        y_quant = tl.maximum(tl.minimum(y_scaled, FP8_MAX), FP8_MIN)

        Y_base = Y_quant + rows[:, None] * stride_y_row + col_offsets
        tl.store(Y_base, y_quant.to(Y_quant.dtype.element_ty), mask=mask)

        S_ptr = Scales + rows * stride_s_row + g * stride_s_g
        tl.store(S_ptr, scales, mask=row_mask_1d)

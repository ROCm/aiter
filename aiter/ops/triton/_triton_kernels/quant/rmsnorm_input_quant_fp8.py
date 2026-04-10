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
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Y_quant,  # pointer to the quantized output
    Scales,  # pointer to the scales
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_z_row,
    stride_y_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    FP8_MIN_SCALING_FACTOR: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * BLOCK_N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y_quant + rows[:, None] * stride_y_row + col_offsets
    S_base = Scales + rows

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Load weights and biases (broadcast across rows)
    w_offsets = cols + group * BLOCK_N
    w_mask = w_offsets < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    ## Now we got y, we next quantize y

    # Compute per-row absmax (only considering valid elements)
    abs_y = tl.where(mask, tl.abs(y), 0.0)
    absmax = tl.max(abs_y, axis=1)  # Shape: [ROWS_PER_BLOCK]

    # Compute scales
    scales_raw = absmax / FP8_MAX
    # TODO: Add USE_UE8M0 as a constexpr parameter if needed:
    if USE_UE8M0:
        scales_raw = tl.exp2(tl.ceil(tl.log2(scales_raw)))
    scales = tl.maximum(scales_raw, FP8_MIN_SCALING_FACTOR)  # Shape: [ROWS_PER_BLOCK]

    # Quantize: divide by scale (broadcast to match y shape) and clamp
    y_scaled = (
        y / scales[:, None]
    )  # Broadcast scales from [ROWS_PER_BLOCK] to [ROWS_PER_BLOCK, BLOCK_N]
    y_quant = tl.maximum(tl.minimum(y_scaled, FP8_MAX), FP8_MIN)

    # Store quantized output
    tl.store(Y_base, y_quant.to(Y_quant.dtype.element_ty), mask=mask)

    # Store scales (one per row)
    scales_row_mask = rows < M
    tl.store(S_base, scales, mask=scales_row_mask)

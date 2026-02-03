# SmoothQuant Int8 Kernels for MoE
# This module provides triton kernels for smoothquant-enabled int8 quantization

import triton
import triton.language as tl


@triton.jit
def _smoothquant_fuse_quant_kernel(
    # Input tensors
    X_ptr,  # bf16 input [M, K]
    stride_x_m,
    stride_x_k,
    SmoothScale_ptr,  # fp32 smooth scale [K] - one per column
    # Output tensors  
    Y_ptr,  # int8 output [M, K]
    stride_y_m,
    stride_y_k,
    RowScale_ptr,  # fp32 per-row scale [M]
    stride_row_scale,
    # Dimensions
    M,
    K,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused smoothquant quantization kernel.
    
    Performs:
    1. Multiply bf16 input by fp32 smooth scale (per column)
    2. Compute per-row max absolute value
    3. Calculate per-row quantization scale
    4. Quantize to int8
    
    Args:
        X_ptr: Input tensor in bf16 [M, K]
        SmoothScale_ptr: Smooth scale vector in fp32 [K]
        Y_ptr: Output tensor in int8 [M, K]
        RowScale_ptr: Per-row quantization scales in fp32 [M]
    """
    # Program ID
    pid_m = tl.program_id(0)
    
    # Row offset
    row_start = pid_m * BLOCK_M
    
    # Initialize per-row max values
    row_max = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # First pass: apply smooth scale and compute row max
    for k_start in range(0, K, BLOCK_K):
        # Compute offsets
        offs_m = row_start + tl.arange(0, BLOCK_M)
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Create masks
        mask_m = offs_m < M
        mask_k = offs_k < K
        mask = mask_m[:, None] & mask_k[None, :]
        
        # Load input values
        x_ptrs = X_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Load smooth scales for this K block
        smooth_scale = tl.load(SmoothScale_ptr + offs_k, mask=mask_k, other=1.0)
        
        # Apply smooth scale: x_smooth = x * smooth_scale (broadcast over M)
        x_fp32 = x.to(tl.float32) * smooth_scale[None, :]
        
        # Update row max with absolute values
        abs_x = tl.abs(x_fp32)
        # Max across K dimension for each row
        block_max = tl.max(abs_x, axis=1)
        row_max = tl.maximum(row_max, block_max)
    
    # Compute per-row quantization scale
    # scale = max_val / 127.0 (int8 max)
    INT8_MAX: tl.constexpr = 127.0
    eps = 1e-12  # Small epsilon to avoid division by zero
    row_scale = row_max / INT8_MAX + eps
    
    # Store row scales
    offs_m = row_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    scale_ptrs = RowScale_ptr + offs_m * stride_row_scale
    tl.store(scale_ptrs, row_scale, mask=mask_m)
    
    # Second pass: quantize values using computed scales
    for k_start in range(0, K, BLOCK_K):
        # Compute offsets
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Create masks
        mask_k = offs_k < K
        mask = mask_m[:, None] & mask_k[None, :]
        
        # Load input values again
        x_ptrs = X_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Load smooth scales
        smooth_scale = tl.load(SmoothScale_ptr + offs_k, mask=mask_k, other=1.0)
        
        # Apply smooth scale
        x_fp32 = x.to(tl.float32) * smooth_scale[None, :]
        
        # Quantize: x_int8 = round(x_fp32 / row_scale)
        x_scaled = x_fp32 / row_scale[:, None]
        
        # Clamp to int8 range and round
        x_clamped = tl.minimum(tl.maximum(x_scaled, -127.0), 127.0)
        # Use standard rounding: add 0.5 and truncate for positive, subtract 0.5 for negative
        x_rounded = tl.where(x_clamped >= 0, 
                             (x_clamped + 0.5).to(tl.int32),
                             (x_clamped - 0.5).to(tl.int32))
        x_int8 = x_rounded.to(tl.int8)
        
        # Store quantized values
        y_ptrs = Y_ptr + offs_m[:, None] * stride_y_m + offs_k[None, :] * stride_y_k
        tl.store(y_ptrs, x_int8, mask=mask)


@triton.jit
def _smoothquant_fuse_quant_kernel_single_pass(
    # Input tensors
    X_ptr,  # bf16 input [M, K]
    stride_x_m,
    stride_x_k,
    SmoothScale_ptr,  # fp32 smooth scale [K] - one per column
    # Output tensors  
    Y_ptr,  # int8 output [M, K]
    stride_y_m,
    stride_y_k,
    RowScale_ptr,  # fp32 per-row scale [M]
    stride_row_scale,
    # Dimensions
    M,
    K,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Single-pass smoothquant quantization kernel for small K dimensions.
    
    Loads entire rows into registers, applies smooth scaling, computes row max,
    and quantizes in a single pass. More efficient when K fits in shared memory.
    """
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]
    
    # Load entire block
    x_ptrs = X_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Load smooth scales
    smooth_scale = tl.load(SmoothScale_ptr + offs_k, mask=mask_k, other=1.0)
    
    # Apply smooth scale
    x_fp32 = x.to(tl.float32) * smooth_scale[None, :]
    
    # Compute per-row max
    abs_x = tl.abs(x_fp32)
    # Set masked values to 0 for max computation
    abs_x = tl.where(mask, abs_x, 0.0)
    row_max = tl.max(abs_x, axis=1)
    
    # Compute scale
    INT8_MAX: tl.constexpr = 127.0
    eps = 1e-12
    row_scale = row_max / INT8_MAX + eps
    
    # Store row scales
    scale_ptrs = RowScale_ptr + offs_m * stride_row_scale
    tl.store(scale_ptrs, row_scale, mask=mask_m)
    
    # Quantize
    x_scaled = x_fp32 / row_scale[:, None]
    x_clamped = tl.minimum(tl.maximum(x_scaled, -127.0), 127.0)
    # Use standard rounding: add 0.5 and truncate for positive, subtract 0.5 for negative
    x_rounded = tl.where(x_clamped >= 0,
                         (x_clamped + 0.5).to(tl.int32),
                         (x_clamped - 0.5).to(tl.int32))
    x_int8 = x_rounded.to(tl.int8)
    
    # Store quantized values
    y_ptrs = Y_ptr + offs_m[:, None] * stride_y_m + offs_k[None, :] * stride_y_k
    tl.store(y_ptrs, x_int8, mask=mask)


@triton.jit  
def _dequant_int8_to_fp32_kernel(
    # Input tensors
    X_ptr,  # int8 input [M, N]
    stride_x_m,
    stride_x_n,
    Scale_ptr,  # fp32 scale [M]
    stride_scale,
    # Output tensor
    Y_ptr,  # fp32 output [M, N]
    stride_y_m,
    stride_y_n,
    # Dimensions
    M,
    N,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Dequantize int8 tensor to fp32 using per-row scales.
    
    Y = X.to(fp32) * scale
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = col_start + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load int8 values
    x_ptrs = X_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n
    x = tl.load(x_ptrs, mask=mask, other=0)
    
    # Load scales
    scale = tl.load(Scale_ptr + offs_m * stride_scale, mask=mask_m, other=1.0)
    
    # Dequantize
    y = x.to(tl.float32) * scale[:, None]
    
    # Store result
    y_ptrs = Y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    tl.store(y_ptrs, y, mask=mask)

# Int8 SmoothQuant MoE Operations
# High-level API for smoothquant-enabled int8 MoE GEMM operations

import torch
import triton
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton._triton_kernels.moe.smoothquant_int8 import (
    _smoothquant_fuse_quant_kernel,
    _smoothquant_fuse_quant_kernel_single_pass,
)
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_int8_smoothquant import (
    _moe_gemm_int8_smoothquant,
    _reduce_grouped_int8,
)


# -----------------------------------------------------------------------------
#                         Utility Functions
# -----------------------------------------------------------------------------


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


# -----------------------------------------------------------------------------
#                         SmoothQuant Quantization
# -----------------------------------------------------------------------------


def smoothquant_quantize(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply smoothquant quantization to convert bf16/fp16 tensor to int8.
    
    Args:
        x: Input tensor in bf16/fp16 [M, K]
        smooth_scale: Per-column smooth scale in fp32 [K]
        
    Returns:
        x_int8: Quantized int8 tensor [M, K]
        x_scale: Per-row quantization scale in fp32 [M]
        
    The operation performs:
    1. x_smooth = x * smooth_scale (per column)
    2. row_scale = max(abs(x_smooth), dim=1) / 127
    3. x_int8 = round(x_smooth / row_scale)
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    assert smooth_scale.ndim == 1, f"Expected 1D smooth_scale, got {smooth_scale.ndim}D"
    assert x.shape[1] == smooth_scale.shape[0], (
        f"Dimension mismatch: x.shape[1]={x.shape[1]}, smooth_scale.shape[0]={smooth_scale.shape[0]}"
    )
    
    M, K = x.shape
    device = x.device
    
    # Allocate outputs
    x_int8 = torch.empty((M, K), dtype=torch.int8, device=device)
    x_scale = torch.empty((M,), dtype=torch.float32, device=device)
    
    # Ensure smooth_scale is fp32 and contiguous
    smooth_scale = smooth_scale.to(torch.float32).contiguous()
    
    # Choose kernel based on K size
    # Use single-pass for K <= 1024, otherwise use multi-pass (iterates over K)
    # Key insight: BLOCK_K should be small enough to fit in registers
    MAX_SINGLE_PASS_K = 1024  # Maximum K for single-pass kernel
    
    BLOCK_M = min(triton.next_power_of_2(M), 32)
    
    if K <= MAX_SINGLE_PASS_K:
        # Single pass: load entire row at once
        BLOCK_K = triton.next_power_of_2(K)
        grid = (triton.cdiv(M, BLOCK_M),)
        
        _smoothquant_fuse_quant_kernel_single_pass[grid](
            x, x.stride(0), x.stride(1),
            smooth_scale,
            x_int8, x_int8.stride(0), x_int8.stride(1),
            x_scale, 1,
            M, K,
            BLOCK_M, BLOCK_K,
            num_warps=4,
        )
    else:
        # Multi-pass: iterate over K dimension with smaller blocks
        # Use BLOCK_K that balances parallelism and register pressure
        BLOCK_K = 256  # Good for most GPUs
        grid = (triton.cdiv(M, BLOCK_M),)
        
        _smoothquant_fuse_quant_kernel[grid](
            x, x.stride(0), x.stride(1),
            smooth_scale,
            x_int8, x_int8.stride(0), x_int8.stride(1),
            x_scale, 1,
            M, K,
            BLOCK_M, BLOCK_K,
            num_warps=4,
        )
    
    return x_int8, x_scale


def quantize_weights_int8(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to int8 with per-output-channel scaling.
    
    Args:
        w: Weight tensor in bf16/fp16/fp32 [E, K, N] or [K, N]
        
    Returns:
        w_int8: Quantized int8 weights (contiguous)
        w_scale: Per-output-channel scale [E, N] or [N] (contiguous)
    """
    if w.ndim == 2:
        # [K, N] -> add expert dimension
        w = w.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    E, K, N = w.shape
    device = w.device
    
    # Compute per-channel max (along K dimension)
    w_fp32 = w.to(torch.float32)
    w_abs_max = w_fp32.abs().max(dim=1).values  # [E, N]
    
    # Compute scale
    INT8_MAX = 127.0
    w_scale = w_abs_max / INT8_MAX + 1e-12  # [E, N]
    
    # Quantize
    w_scaled = w_fp32 / w_scale[:, None, :]
    w_int8 = w_scaled.round().clamp(-127, 127).to(torch.int8)
    
    # Ensure contiguous memory layout for efficient access
    # Layout [E, K, N] with N contiguous is optimal for the GEMM kernel
    w_int8 = w_int8.contiguous()
    w_scale = w_scale.contiguous()
    
    if squeeze_output:
        w_int8 = w_int8.squeeze(0)
        w_scale = w_scale.squeeze(0)
    
    return w_int8, w_scale


# -----------------------------------------------------------------------------
#                         MoE GEMM Kernel Configuration
# -----------------------------------------------------------------------------


def get_kernel_config(m, n, k, routing_data):
    """
    Get optimized kernel configuration for int8 GEMM on AMD MI355.
    
    Key optimizations for int8:
    - Larger BLOCK_K (128-256) for int8 to maximize arithmetic intensity
    - Larger BLOCK_N (128-256) for better parallelism
    - More pipeline stages (3-4) for latency hiding
    - Appropriate num_warps for occupancy
    """
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    
    # Use more pipeline stages for int8 GEMM
    num_stages = 3

    split_k = 1
    if block_m == 16:
        # Small batch: use smaller blocks for better occupancy
        block_n = 128  # Increased from 64 for better parallelism
        block_k = 128  # Reduced for better pipelining with int8
        num_warps = 4

        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = triton.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        
        # Adjust block_n to ensure enough parallelism
        while block_n > 32 and grid < 256:
            block_n = block_n // 2
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
    elif block_m == 32:
        # Medium batch: balance between parallelism and efficiency
        if n <= 1024:
            block_n = 128
            block_k = 128
            num_warps = 4
        elif n <= 4096:
            block_n = 256
            block_k = 128
            num_warps = 8
        else:
            block_n = 256
            block_k = 128
            num_warps = 8
    else:
        # Large batch (block_m >= 64): maximize throughput
        block_n = 256
        block_k = 128
        num_warps = 8

    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_m": group_m,
        "xcd_swizzle": xcd_swizzle,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }


def allocate_output(
    x,
    w,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
):
    """Allocate output tensors for MoE GEMM."""
    N = w.shape[-1]
    M = x.shape[-2]
    if gather_indx is not None:
        M = gather_indx.shape[0]
    
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = scatter_indx.shape[0] // routing_data.n_expts_act
    
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=x.device, dtype=out_dtype)
    
    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=x.device, dtype=out_dtype)
    else:
        final_output = None
    
    return matmul_output, final_output


def reduce_grouped(
    x: torch.Tensor,
    indx: torch.Tensor,
    out: torch.Tensor,
    apply_swiglu=False,
    apply_silu=False,
    alpha=1.0,
    limit=1.0,
    reduction_n=1,
    out_dtype: torch.dtype = None,
):
    """In-place grouped row reduction with optional activation."""
    if indx is None and x.shape[0] == 1:
        return x.squeeze(0)
    
    if indx is not None:
        num_groups = indx.shape[0]
    else:
        num_groups = x.shape[-2]
    
    K = 1 if indx is None else indx.shape[1]
    out_dtype = x.dtype if out_dtype is None else out_dtype
    assert x.shape[-1] % reduction_n == 0
    
    BLOCK_N = 512
    num_blocks = triton.cdiv(x.shape[-1], BLOCK_N)

    # Map torch dtype to triton dtype
    if out_dtype == torch.bfloat16:
        out_dtype_tl = "bfloat16"
    else:
        out_dtype_tl = "float32"

    _reduce_grouped_int8[(num_blocks, num_groups)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        indx,
        x.shape[0],
        x.shape[-1],
        apply_swiglu,
        apply_silu,
        alpha,
        limit,
        reduction_n,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        K=K,
        OUT_DTYPE=out_dtype_tl,
        num_warps=2,
    )
    return out


# -----------------------------------------------------------------------------
#                         Main MoE GEMM API
# -----------------------------------------------------------------------------


def moe_gemm_int8_smoothquant(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor = None,
    routing_data: RoutingData = None,
    gather_indx: torch.Tensor = None,
    scatter_indx: torch.Tensor = None,
    gammas: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    apply_swiglu: bool = False,
    apply_silu: bool = False,
    alpha: float = 1.0,
    limit: float = 1.0,
):
    """
    Int8 MoE GEMM with SmoothQuant support.
    
    Performs MoE matrix multiplication with int8 quantized inputs:
    Y = (X @ W) * x_scale * w_scale
    
    Args:
        x: Int8 activation tensor [M, K]
        x_scale: Per-token fp32 scale [M]
        w: Int8 weight tensor [E, K, N]
        w_scale: Per-output-channel fp32 scale [E, N]
        bias: Optional bias tensor [E, N]
        routing_data: MoE routing metadata
        gather_indx: Token gather indices for layer 1
        scatter_indx: Token scatter indices for layer 2
        gammas: Optional gating weights
        out_dtype: Output data type (default: bf16)
        apply_swiglu: Apply SwiGLU activation
        apply_silu: Apply SiLU activation
        alpha: SwiGLU alpha parameter
        limit: SwiGLU clamp limit
        
    Returns:
        Output tensor in out_dtype
    """
    assert x.dtype == torch.int8, f"Expected int8 activations, got {x.dtype}"
    assert w.dtype == torch.int8, f"Expected int8 weights, got {w.dtype}"
    assert x_scale.dtype == torch.float32, f"Expected fp32 x_scale, got {x_scale.dtype}"
    assert w_scale.dtype == torch.float32, f"Expected fp32 w_scale, got {w_scale.dtype}"
    
    # Get shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    
    # Get kernel config
    config = get_kernel_config(M, N, K, routing_data)
    
    # Handle activation reduction
    # Initialize all variables to ensure they're defined
    apply_swiglu_matmul = False
    apply_silu_matmul = False
    reduction_n_matmul = 1
    apply_swiglu_reduction = False
    apply_silu_reduction = False
    reduction_n_reduction = 1
    
    if apply_swiglu and config["split_k"] > 1:
        # Defer SwiGLU to reduction phase when using split_k
        apply_swiglu_reduction = True
        reduction_n_reduction = 2
    elif apply_swiglu:
        # Apply SwiGLU in matmul phase (halves N dimension)
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
    elif apply_silu and config["split_k"] > 1:
        # Defer SiLU to reduction phase when using split_k
        apply_silu_reduction = True
    elif apply_silu:
        # Apply SiLU in matmul phase
        apply_silu_matmul = True
    
    # Allocate output
    # For intermediate computation use fp32, then convert to out_dtype in reduction
    intermediate_dtype = torch.float32
    y, y_final = allocate_output(
        x, w, intermediate_dtype,
        reduction_n_matmul, reduction_n_reduction,
        routing_data, gather_indx, scatter_indx,
        config["block_m"], config["split_k"],
    )
    
    stride_bias = None if bias is None else bias.stride(0)
    
    # MoE metadata
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map
    
    # Grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]
    
    # Launch kernel
    _moe_gemm_int8_smoothquant[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        x_scale,
        x_scale.stride(0) if x_scale.ndim > 0 else 0,
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scale,
        w_scale.stride(0),
        w_scale.stride(1) if w_scale.ndim > 1 else 0,
        bias,
        stride_bias,
        gammas,
        N, K,
        gather_indx,
        expt_hist,
        expt_token_offs_raw,
        expt_hist_sum,
        expt_block_pid_map,
        grid_m, grid_n,
        apply_swiglu_matmul,
        apply_silu_matmul,
        alpha, limit,
        reduction_n_matmul,
        routing_data.n_expts_act,
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["group_m"],
        XCD_SWIZZLE=config["xcd_swizzle"],
        EVEN_K=K % config["block_k"] == 0,
        MASK_K_LIMIT=K % config["block_k"],
        SPLIT_K=config["split_k"],
        W_CACHE_MODIFIER=config["w_cache_modifier"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        UPCAST_INDICES=should_upcast_indices(x, w, y),
    )
    
    # Reduction
    group_indx = (
        None if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    y_final = reduce_grouped(
        y, group_indx, y_final,
        apply_swiglu_reduction,
        apply_silu_reduction,
        alpha, limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
    )
    
    # Convert to output dtype if needed
    if y_final.dtype != out_dtype:
        y_final = y_final.to(out_dtype)
    
    return y_final


# -----------------------------------------------------------------------------
#                         Full SmoothQuant MoE MLP
# -----------------------------------------------------------------------------


def smoothquant_moe_mlp(
    x: torch.Tensor,
    fc1_smooth_scale: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    fc2_smooth_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    routing_data: RoutingData,
    gather_indx: torch.Tensor,
    scatter_indx: torch.Tensor,
    bias1: torch.Tensor = None,
    bias2: torch.Tensor = None,
    gammas: torch.Tensor = None,
    apply_swiglu: bool = True,
) -> torch.Tensor:
    """
    Complete SmoothQuant MoE MLP forward pass.
    
    Flow:
    1. bf16 tokens × FC1 smooth scale = fp32 tokens
    2. fp32 tokens → int8 X + fp32 X scale (per token)
    3. MoE1 int8 → int32 → fp32
    4. MoE1 out × X scale × W scale
    5. SwiGLU activation (halves output dimension: N → N/2)
    6. fp32 tokens × FC2 smooth scale = fp32 tokens
    7. fp32 tokens → int8 Y + fp32 Y scale (per token)
    8. MoE2 int8 → int32 → fp32
    9. MoE2 out × Y scale × W scale
    10. Downcast to bf16
    
    Args:
        x: Input tensor in bf16 [M, K]
        fc1_smooth_scale: Smooth scale for FC1 [K]
        w1: Int8 weights for FC1 [E, K, N] (N is double-width for gating)
        w1_scale: Per-channel scale for FC1 [E, N]
        fc2_smooth_scale: Smooth scale for FC2 [N // 2] (after SwiGLU halves)
        w2: Int8 weights for FC2 [E, N // 2, K]
        w2_scale: Per-channel scale for FC2 [E, K]
        routing_data: MoE routing metadata
        gather_indx: Gather indices for first layer
        scatter_indx: Scatter indices for second layer
        bias1: Optional bias for FC1
        bias2: Optional bias for FC2
        gammas: Optional gating weights
        apply_swiglu: Whether to apply SwiGLU activation (default: True)
        
    Returns:
        Output tensor in bf16
    """
    # Step 1-2: Apply FC1 smooth scale and quantize
    x_int8, x_scale = smoothquant_quantize(x, fc1_smooth_scale)
    
    # Step 3-5: First MoE GEMM with SwiGLU activation
    # Note: SwiGLU halves the output dimension (N → N/2)
    intermediate = moe_gemm_int8_smoothquant(
        x_int8,
        x_scale,
        w1,
        w1_scale,
        bias=bias1,
        routing_data=routing_data,
        gather_indx=gather_indx,
        scatter_indx=None,  # Don't scatter yet
        gammas=gammas,
        out_dtype=torch.float32,  # Keep in fp32 for next quant
        apply_swiglu=apply_swiglu,
    )
    
    # Step 6-7: Apply FC2 smooth scale and quantize
    y_int8, y_scale = smoothquant_quantize(intermediate, fc2_smooth_scale)
    
    # Step 8-10: Second MoE GEMM and downcast to bf16
    output = moe_gemm_int8_smoothquant(
        y_int8,
        y_scale,
        w2,
        w2_scale,
        bias=bias2,
        routing_data=routing_data,
        gather_indx=None,  # Already gathered
        scatter_indx=scatter_indx,
        out_dtype=torch.bfloat16,
    )
    
    return output


# -----------------------------------------------------------------------------
#                         Reference Implementation
# -----------------------------------------------------------------------------


def smoothquant_quantize_torch(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation of smoothquant quantization."""
    # Apply smooth scale
    x_smooth = x.to(torch.float32) * smooth_scale[None, :]
    
    # Compute per-row max
    row_max = x_smooth.abs().max(dim=1).values
    
    # Compute scale
    INT8_MAX = 127.0
    row_scale = row_max / INT8_MAX + 1e-12
    
    # Quantize
    x_scaled = x_smooth / row_scale[:, None]
    x_int8 = x_scaled.round().clamp(-127, 127).to(torch.int8)
    
    return x_int8, row_scale


def moe_gemm_int8_smoothquant_torch(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor = None,
    routing_data: RoutingData = None,
    gather_indx: torch.Tensor = None,
    scatter_indx: torch.Tensor = None,
    gammas: torch.Tensor = None,
    apply_swiglu: bool = False,
    apply_silu: bool = False,
    alpha: float = 1.0,
    limit: float = 1.0,
):
    """PyTorch reference implementation of int8 MoE GEMM."""
    import itertools
    
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if w.ndim == 2:
        w = w.view(1, *w.shape)
    
    n_expts_act = routing_data.n_expts_act
    
    # Memory offsets
    if routing_data.n_expts_tot > 1:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32, device=x.device)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[0]] for _ in range(w.shape[0])]
    
    # Compute
    n_rows = x.shape[0] if gather_indx is None else gather_indx.shape[0]
    n_cols = w.shape[-1] // 2 if apply_swiglu else w.shape[-1]
    y = torch.zeros((n_rows, n_cols), device=x.device, dtype=torch.float32)
    
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            idx = gather_indx[lo:hi] // n_expts_act
        
        # Int8 matmul
        x_block = x[idx, :].to(torch.int32)
        w_block = w[i].to(torch.int32)
        out = torch.matmul(x_block, w_block).to(torch.float32)
        
        # Apply scales
        out = out * x_scale[idx, None]
        out = out * w_scale[i, None, :]
        
        if bias is not None:
            out = out + bias[i, :]
        
        if apply_swiglu:
            # SwiGLU: split and apply gating
            a_gelu = out[..., ::2]
            a_linear = out[..., 1::2]
            if limit is not None:
                a_gelu = a_gelu.clamp(max=limit)
                a_linear = a_linear.clamp(min=-limit, max=limit)
            s = a_gelu * torch.sigmoid(alpha * a_gelu)
            out = s * (a_linear + 1)
        elif apply_silu:
            out = out * torch.sigmoid(out)
        
        if gammas is not None:
            out = out * gammas[lo:hi, None]
        
        y[lo:hi, :] = out
    
    if scatter_indx is None:
        return y
    
    # Accumulate from all experts
    n_rows_out = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows_out, y.shape[-1]), dtype=torch.float32, device=x.device)
    src_idx = scatter_indx.view(-1, n_expts_act)
    for i in range(n_rows_out):
        out[i, :] = y[src_idx[i], :].sum(0)
    
    return out

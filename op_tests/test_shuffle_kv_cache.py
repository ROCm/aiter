"""
Performance test for shuffle KV cache kernel.
Compare CUDA kernel vs Triton kernel.

Test config:
- seq = 140000
- q_head = 8
- kv_head = 1
- head_size = 128
- quant KV (FP8)
- shuffle layout
"""

import torch
import triton
import triton.language as tl
import time
from typing import Optional

# Try to import aiter module
try:
    import aiter
    from aiter.ops.fused_mrope_rms import shuffle_kv_cache_cuda
    HAS_AITER = True
except ImportError:
    HAS_AITER = False
    shuffle_kv_cache_cuda = None
    print("Warning: aiter module not found. CUDA kernel test will be skipped.")


# ============================================================================
# Triton Kernel (from user's code)
# ============================================================================
@triton.jit
def reshape_and_cache_shuffle_kernel(
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
    slot_mapping_ptr,  # [num_tokens]
    k_scale_ptr,
    v_scale_ptr,
    x,
    k_stride0,
    v_stride0,
    block_size,
    head_size,
    num_kv_heads,
    BLOCK_SIZE: tl.constexpr,
    QUANT: tl.constexpr,
):
    tid = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    src_offset_k = tid * k_stride0 + head_id * head_size
    src_offset_v = tid * v_stride0 + head_id * head_size
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    dst_offset = (
        block_id * num_kv_heads * head_size * block_size
        + head_id * head_size * block_size
    )
    dst_k_shuffle_offset = (
        dst_offset + offset // x * block_size * x + block_offset * x + offset % x
    )
    dst_v_shuffle_offset = (
        dst_offset
        + block_offset // x * head_size * x
        + offset * x
        + block_offset % x
    )
    k_val = tl.load(key_ptr + src_offset_k + offset)
    v_val = tl.load(value_ptr + src_offset_v + offset)
    if QUANT:
        k_dtype = key_cache_ptr.type.element_ty
        v_dtype = value_cache_ptr.type.element_ty
        # Fast path: skip division when scale=1, just type cast
        k_val = k_val.to(k_dtype)
        v_val = v_val.to(v_dtype)
    tl.store(key_cache_ptr + dst_k_shuffle_offset, k_val)
    tl.store(value_cache_ptr + dst_v_shuffle_offset, v_val)


def reshape_and_cache_shuffle_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
):
    num_tokens = slot_mapping.shape[0]
    _, num_kv_heads, head_size = key.shape
    # Get block_size from cache shape
    # key_cache shape: [num_blocks, block_size, num_kv_heads, head_size] (before view)
    # After view: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[1]  # Original shape before view
    x = 16 // key_cache.element_size()
    
    k_cache_template = torch.empty(
        [num_blocks, num_kv_heads, head_size // x, block_size, x],
        dtype=key_cache.dtype,
        device="meta",
    )
    v_cache_template = torch.empty(
        [num_blocks, num_kv_heads, block_size // x, head_size, x],
        dtype=value_cache.dtype,
        device="meta",
    )
    new_key_cache = key_cache.view_as(k_cache_template)
    new_value_cache = value_cache.view_as(v_cache_template)
    
    QUANT = kv_cache_dtype.startswith("fp8")
    
    grid = (num_tokens, num_kv_heads)
    reshape_and_cache_shuffle_kernel[grid](
        key,
        value,
        new_key_cache,
        new_value_cache,
        slot_mapping,
        k_scales,
        v_scales,
        x,
        key.stride(0),
        value.stride(0),
        block_size,
        head_size,
        num_kv_heads,
        BLOCK_SIZE=head_size,
        QUANT=QUANT,
    )


# ============================================================================
# Test Functions
# ============================================================================
def create_test_data(
    num_tokens: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype = torch.float16,
    kv_cache_dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Create test data for shuffle KV cache kernel."""
    # Calculate number of blocks needed
    num_blocks = (num_tokens + block_size - 1) // block_size
    
    # x = 16 // element_size
    x = 16 // torch.tensor([], dtype=kv_cache_dtype).element_size()
    
    # Input tensors
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    
    # Slot mapping (consecutive slots for simplicity)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    
    # Cache tensors in shuffle layout
    # K cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    # V cache: [num_blocks, num_kv_heads, block_size // x, head_size, x]
    # But stored as [num_blocks, block_size, num_kv_heads, head_size] and then viewed
    k_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size,
        dtype=kv_cache_dtype, device=device
    )
    v_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size,
        dtype=kv_cache_dtype, device=device
    )
    
    # Scales for quantization
    k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    
    return key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale, x


def benchmark_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    kv_cache_dtype: str,
    warmup: int = 10,
    runs: int = 100,
):
    """Benchmark Triton kernel."""
    # Warmup
    for _ in range(warmup):
        reshape_and_cache_shuffle_triton(
            key, value, k_cache.clone(), v_cache.clone(),
            slot_mapping, kv_cache_dtype, k_scale, v_scale
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        reshape_and_cache_shuffle_triton(
            key, value, k_cache, v_cache,
            slot_mapping, kv_cache_dtype, k_scale, v_scale
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / runs * 1000
    return avg_time_ms


def benchmark_cuda(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: float,
    v_scale: float,
    block_size: int,
    x: int,
    warmup: int = 10,
    runs: int = 100,
):
    """Benchmark CUDA kernel."""
    if not HAS_AITER:
        return None
    
    # Reshape caches to shuffle layout
    num_blocks = k_cache.shape[0]
    num_kv_heads = key.shape[1]
    head_size = key.shape[2]
    
    k_cache_shuffle = k_cache.view(num_blocks, num_kv_heads, head_size // x, block_size, x)
    v_cache_shuffle = v_cache.view(num_blocks, num_kv_heads, block_size // x, head_size, x)
    
    # Warmup
    for _ in range(warmup):
        shuffle_kv_cache_cuda(
            key, value, 
            k_cache_shuffle.clone(), v_cache_shuffle.clone(),
            slot_mapping, block_size, x, k_scale, v_scale
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        shuffle_kv_cache_cuda(
            key, value, k_cache_shuffle, v_cache_shuffle,
            slot_mapping, block_size, x, k_scale, v_scale
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / runs * 1000
    return avg_time_ms


def test_correctness(
    num_tokens: int = 1024,
    num_kv_heads: int = 1,
    head_size: int = 128,
    block_size: int = 16,
):
    """Test correctness of CUDA kernel vs Triton kernel."""
    print(f"\n{'='*60}")
    print(f"Correctness Test")
    print(f"num_tokens={num_tokens}, num_kv_heads={num_kv_heads}, head_size={head_size}, block_size={block_size}")
    print(f"{'='*60}")
    
    dtype = torch.float16
    kv_cache_dtype = torch.float16
    
    key, value, k_cache_triton, v_cache_triton, slot_mapping, k_scale, v_scale, x = \
        create_test_data(num_tokens, num_kv_heads, head_size, block_size, dtype, kv_cache_dtype)
    
    k_cache_cuda = k_cache_triton.clone()
    v_cache_cuda = v_cache_triton.clone()
    
    # Run Triton
    reshape_and_cache_shuffle_triton(
        key, value, k_cache_triton, v_cache_triton,
        slot_mapping, "auto", k_scale, v_scale
    )
    
    if HAS_AITER:
        # Reshape to shuffle layout for CUDA kernel
        num_blocks = k_cache_cuda.shape[0]
        k_cache_cuda_view = k_cache_cuda.view(num_blocks, num_kv_heads, head_size // x, block_size, x)
        v_cache_cuda_view = v_cache_cuda.view(num_blocks, num_kv_heads, block_size // x, head_size, x)
        
        # Run CUDA
        shuffle_kv_cache_cuda(
            key, value, k_cache_cuda_view, v_cache_cuda_view,
            slot_mapping, block_size, x, 1.0, 1.0
        )
        
        # Compare
        k_diff = (k_cache_triton - k_cache_cuda).abs().max().item()
        v_diff = (v_cache_triton - v_cache_cuda).abs().max().item()
        
        print(f"K cache max diff: {k_diff}")
        print(f"V cache max diff: {v_diff}")
        
        if k_diff < 1e-3 and v_diff < 1e-3:
            print("✓ Correctness test PASSED!")
        else:
            print("✗ Correctness test FAILED!")
    else:
        print("Skipping CUDA kernel correctness test (aiter not available)")


def test_performance(
    num_tokens: int = 140000,
    num_kv_heads: int = 1,
    head_size: int = 128,
    block_size: int = 16,
    use_fp8: bool = False,
):
    """Test performance of CUDA kernel vs Triton kernel."""
    print(f"\n{'='*60}")
    print(f"Performance Test")
    print(f"num_tokens={num_tokens}, num_kv_heads={num_kv_heads}, head_size={head_size}, block_size={block_size}")
    print(f"use_fp8={use_fp8}")
    print(f"{'='*60}")
    
    dtype = torch.float16
    if use_fp8:
        try:
            kv_cache_dtype = torch.float8_e4m3fnuz  # AMD
        except AttributeError:
            try:
                kv_cache_dtype = torch.float8_e4m3fn  # NVIDIA
            except AttributeError:
                print("FP8 not supported on this platform, falling back to FP16")
                kv_cache_dtype = torch.float16
                use_fp8 = False
    else:
        kv_cache_dtype = torch.float16
    
    key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale, x = \
        create_test_data(num_tokens, num_kv_heads, head_size, block_size, dtype, kv_cache_dtype)
    
    kv_cache_dtype_str = "fp8" if use_fp8 else "auto"
    
    # Calculate memory bandwidth
    # Read: key + value = 2 * num_tokens * num_kv_heads * head_size * 2 bytes (fp16)
    # Write: k_cache + v_cache = 2 * num_tokens * num_kv_heads * head_size * element_size
    element_size = torch.tensor([], dtype=kv_cache_dtype).element_size()
    read_bytes = 2 * num_tokens * num_kv_heads * head_size * 2  # fp16 input
    write_bytes = 2 * num_tokens * num_kv_heads * head_size * element_size
    total_bytes = read_bytes + write_bytes
    total_gb = total_bytes / 1e9
    
    print(f"Data size: {total_gb:.2f} GB (read: {read_bytes/1e6:.2f} MB, write: {write_bytes/1e6:.2f} MB)")
    print(f"x = {x}")
    
    # Benchmark Triton
    k_cache_triton = k_cache.clone()
    v_cache_triton = v_cache.clone()
    triton_time_ms = benchmark_triton(
        key, value, k_cache_triton, v_cache_triton,
        slot_mapping, k_scale, v_scale, kv_cache_dtype_str,
        warmup=10, runs=100
    )
    triton_bandwidth_gbps = total_gb / (triton_time_ms / 1000)
    print(f"Triton kernel: {triton_time_ms:.3f} ms, {triton_bandwidth_gbps:.2f} GB/s")
    
    # Benchmark CUDA
    if HAS_AITER:
        k_cache_cuda = k_cache.clone()
        v_cache_cuda = v_cache.clone()
        cuda_time_ms = benchmark_cuda(
            key, value, k_cache_cuda, v_cache_cuda,
            slot_mapping, 1.0, 1.0, block_size, x,
            warmup=10, runs=100
        )
        if cuda_time_ms is not None:
            cuda_bandwidth_gbps = total_gb / (cuda_time_ms / 1000)
            print(f"CUDA kernel:   {cuda_time_ms:.3f} ms, {cuda_bandwidth_gbps:.2f} GB/s")
            
            # Speedup
            speedup = triton_time_ms / cuda_time_ms
            print(f"Speedup (CUDA vs Triton): {speedup:.2f}x")
    else:
        print("CUDA kernel: N/A (aiter not available)")


def main():
    print("="*60)
    print("Shuffle KV Cache Kernel Benchmark")
    print("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"aiter available: {HAS_AITER}")
    
    # Correctness test (small data)
    test_correctness(num_tokens=1024, num_kv_heads=1, head_size=128, block_size=16)
    
    # FP8 performance test
    test_performance(
        num_tokens=14000,
        num_kv_heads=1,
        head_size=128,
        block_size=1024,
        use_fp8=True,
    )
    
    # Different block sizes
    for block_size in [16, 1024]:
        test_performance(
            num_tokens=14000,
            num_kv_heads=1,
            head_size=128,
            block_size=block_size,
            use_fp8=True,
        )
    # Different block sizes
    for block_size in [16, 1024]:
        test_performance(
            num_tokens=1,
            num_kv_heads=1,
            head_size=128,
            block_size=block_size,
            use_fp8=True,
        )

if __name__ == "__main__":
    main()


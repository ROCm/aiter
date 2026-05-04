# CK vs Triton Unified Attention - Performance Results

## Executive Summary

CK Tile unified attention optimizations show **batch-size dependent performance** vs Triton:

- **High batch (b≥64)**: CK wins 1.1–1.7× faster
- **Low batch (b≤32)**: Triton wins 1.4–6.6× faster (using 3D split-KV kernel)
- **Crossover point**: Between b=32 and b=64

## Detailed Results

### High Batch Performance (CK Advantage)

| Batch | CK (ms) | Triton (ms) | Triton Kernel | Speedup | Winner |
|-------|---------|-------------|---------------|---------|--------|
| 512   | 1.065   | 1.815       | 2D            | **1.70×** | **CK** |
| 256   | 0.395   | 0.684       | 2D            | **1.73×** | **CK** |
| 64    | 0.243   | 0.267       | 2D            | **1.10×** | **CK** |

### Low Batch Performance (Triton Advantage)

| Batch | CK (ms) | Triton (ms) | Triton Kernel | Speedup | Winner |
|-------|---------|-------------|---------------|---------|--------|
| 32    | 0.217   | 0.156       | 3D + reduce   | **0.72×** | **Triton 1.4×** |
| 16    | 0.213   | 0.097       | 3D + reduce   | **0.45×** | **Triton 2.2×** |
| 8     | 0.212   | 0.054       | 3D + reduce   | **0.25×** | **Triton 3.9×** |
| 4     | 0.213   | 0.032       | 3D + reduce   | **0.15×** | **Triton 6.6×** |

*Test configuration: sk=8192, hq=64, hk=8, d=64, decode, bf16*

## Key Insights

### Triton Kernel Selection
- **2D kernel**: Used at moderate to high batch (b≥64)
  - Single kernel launch per iteration
  - Simpler execution model

- **3D kernel**: Used at low batch (b≤32)
  - Split-KV parallelism to utilize GPU at low occupancy
  - Paired with `reduce_segments` reduction kernel (~4 μs overhead)
  - Significantly faster when batch size limits parallelism

### Why CK Wins at High Batch
CK Tile optimizations require sufficient parallelism to saturate the GPU. At batch sizes ≥64, there's enough work to fully utilize compute resources, allowing CK's optimizations to shine.

### Why Triton Wins at Low Batch
At low batch sizes (b≤32), standard 2D kernels have insufficient parallelism. Triton's 3D split-KV kernel parallelizes over the KV sequence dimension, maintaining high GPU utilization even with small batches.

## Correctness

100% pass rate across all tested configurations:
- Max absolute difference: ≤2.4e-4 (within bf16 precision)
- `torch.testing.assert_allclose` with atol=1e-2, rtol=1e-2: ✓ PASS

## Profiling Details

### Kernel Trace Analysis
ROCProfiler v3 captures:
- CK kernel: `ck_tile::kentry<2, UnifiedAttentionKernel>`
- Triton 2D: `kernel_unified_attention_2d`
- Triton 3D: `kernel_unified_attention_3d` + `reduce_segments`

The trace parser (`parse_kernel_trace.py`) automatically:
- Excludes warmup iterations from statistics
- Detects 2D vs 3D Triton variants
- Accounts for reduction kernel overhead in 3D timing

### Measurement Accuracy
All benchmarks use **CUDA graph mode** by default to eliminate kernel launch overhead and match production performance. This is critical for 3D kernels which launch two kernels (main + reduce) - eager mode adds ~18μs launch overhead per iteration.

Python-level median timing with graph mode matches ROCProfiler kernel trace timing within 1-3%, validating measurement accuracy.

## Recommendations

**For production inference** (typical batch 64-512+):
- **Use CK** - Consistently 1.1-1.7× faster at realistic batch sizes

**For low-concurrency scenarios** (batch <32):
- **Use Triton** - Up to 4× faster due to split-KV parallelism

**For mixed workloads**:
- Consider dynamic kernel selection based on batch size
- Crossover threshold: b=48 (midpoint between 32 and 64)

## Test Environment

- **Hardware**: AMD MI350 (gfx950), 256 CUs, ~295GB HBM
- **Software**: ROCm 7.0.0, rocprofv3
- **Configuration**: GQA-8 (64 query heads, 8 KV heads), head_dim=64, bf16
- **Workload**: Decode (seqlen_q=1), context length 8192-12000

## Bug Fixed During Testing

**Issue**: Triton unified attention path had `_try_ck_unified_attention()` enabled, causing Triton to incorrectly call CK implementation.

**Fix**: Commented out lines 234-238 in `/root/aiter/aiter/ops/triton/attention/unified_attention.py`

This bug caused early test runs to show similar performance at all batch sizes, masking the batch-dependent behavior.

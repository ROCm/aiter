# AMD Triton Performance Optimization Guide

Source: ROCm/triton wiki, AMD ROCm documentation, AMD architecture specs.

## AMD GPU Hardware Model (MI300X / CDNA3)

| Resource | Value |
|----------|-------|
| Compute Units (CUs) | 304 (38 active per XCD × 8 XCDs) |
| SIMDs per CU | 4 |
| Wavefront size | 64 threads |
| VGPRs per SIMD | 512 (allocated in units of 16) |
| LDS per CU | 64 KB |
| L1 cache per CU | 32 KB |
| L2 cache per XCD | 4 MB (shared across 40 CUs) |
| HBM3 bandwidth | 5.3 TB/s peak |
| HBM3 capacity | 192 GB |
| Peak FP16/BF16 | 1307.4 TFLOPS |
| Peak FP8/INT8 | 2614.9 TFLOPS |
| Peak FP32 (matrix) | 163.4 TFLOPS |

## CDNA4 / MI350 Additions
- 8 XCDs per GPU (3D stacked)
- HBM3E memory
- Native scaled MFMA for OCP microscaling formats (MXFP4, MXFP8)
- `tl.dot_scaled()` support for microscaling GEMM

## Software Pipelining (`num_stages`)

Controls overlapping of memory loads with compute. On AMD, data streams through **register buffers** (not directly to shared memory like NVIDIA).

| Kernel Type | Recommended `num_stages` | Reason |
|-------------|--------------------------|--------|
| GEMM (direct loads) | 2 | Hides global load latency |
| GEMM (indirect loads) | 3 | Extra stage for address computation |
| Fused GEMM (compute-bound, e.g. flash attention prefill) | 1 | Avoids register spills (many inputs) |
| Fused GEMM (memory-bound, e.g. paged attention) | 2-3 | Benefits from pipelining |
| Elementwise (layernorm, softmax) | 1-2 | Use `tl.range(..., num_stages=N)` |

**For elementwise kernels:** Pipelining must be specified on `tl.range()`, not globally via `triton.Config`. Only the innermost loop is pipelined.

```python
# Elementwise pipelining
for i in tl.range(0, N, BLOCK, num_stages=2):
    x = tl.load(ptr + i + tl.arange(0, BLOCK))
    ...
```

**If performance degrades with `num_stages > 1`:** Check assembly for register spills. Fall back to `num_stages=1`.

## `waves_per_eu` (Occupancy Hint)

Hints the compiler to reduce VGPR usage to achieve target occupancy.

```python
triton.Config({"BLOCK_M": 128, ...}, num_stages=2, num_warps=4, waves_per_eu=2)
```

**Only helps when:**
1. Occupancy is VGPR-limited
2. Current VGPR usage is slightly above an allocation boundary

**VGPR occupancy table (per SIMD, 512 total VGPRs):**

| VGPRs per wave | Max waves/SIMD | Allocation granularity |
|---------------|----------------|----------------------|
| 1-128 | 4 | 16 VGPRs |
| 129-170 | 2-3 | 16 VGPRs |
| 171-256 | 2 | 16 VGPRs |
| 257-512 | 1 | 16 VGPRs |

## Block Sizes (Tile Sizes)

**For GEMM kernels:**

Goal: Maximize memory-to-compute ratio while keeping enough blocks for all CUs.

Example analysis for MI300X (304 CUs), GEMM 4096×4096:
- 256×256 → 16×16 = 256 blocks → 84% utilization (< 304 CUs)
- 128×128 → 32×32 = 1024 blocks → 84% (1024/304 ≈ 3.37, ceil=4, 3.37/4=84%)
- 128×64 → 32×64 = 2048 blocks → 99% utilization
- 64×64 → 64×64 = 4096 blocks → 94% utilization

**BLOCK_K guidelines:**
- Optimal: 512 contiguous bytes per load → BLOCK_K=256 for fp16/bf16
- Practical: 64 or 128 (256 makes tiles too large)

**For elementwise kernels:**
- Memory-bound → want 16-32 KB in flight per program
- BLOCK_SIZE = 8192 or 16384 for 2-byte types
- Compute in `tl.float32` even if inputs are fp16/bf16

## MFMA Instruction Type (`matrix_instr_nonkdim`)

| Value | Instruction | Best For |
|-------|------------|----------|
| 16 | `v_mfma_f32_16x16x16_{f16,bf16}` / `v_mfma_f32_16x16x32_{fp8,bf8}` | GEMM kernels |
| 32 | `v_mfma_f32_32x32x8_{f16,bf16}` / `v_mfma_f32_32x32x16_{fp8,bf8}` | Fused GEMM kernels |

## `kpack`

Optimizes shared memory accesses. **Set to 2 for GEMMs**, leave unset for other kernels.

## Compiler Hints (Critical)

```python
# Tell compiler strides are positive → enables addressing optimizations
tl.assume(stride_am > 0)
tl.assume(stride_ak > 0)

# Tell compiler pointers are aligned → enables vectorized loads
ptr = tl.multiple_of(ptr, (16,))
```

These hints enable `global_load_dwordx4` (128-bit vectorized loads) in the assembly.

## Computing Occupancy

1. Get VGPR count: search `.vgpr_count` in ISA (`AMDGCN_ENABLE_DUMP=1`)
2. Get LDS usage: search `ttg.shared` in IR (`MLIR_ENABLE_DUMP=1`)
3. Get waves per workgroup: search `ttg.num-warps` in IR

```
occ_vgpr = max_waves_per_simd (from VGPR table above)
occ_lds = floor(65536 / LDS_bytes)
occ = min(floor(occ_vgpr * 4 / num_warps), occ_lds) * num_warps / 4
```

## IR and Assembly Analysis

### Generating assembly
```bash
AMDGCN_ENABLE_DUMP=1 python kernel.py > asm.amdgcn 2>&1
```

### Generating Triton IR
```bash
MLIR_ENABLE_DUMP=1 python kernel.py > ir.mlir 2>&1
```

### Finding autotuning winner
```bash
TRITON_PRINT_AUTOTUNING=1 python kernel.py
```

### What to check in assembly
- **Global loads:** Should be `global_load_dwordx4` (128-bit), not smaller
- **LDS access:** Should use `_b128` or `_b64`, not smaller
- **`s_waitcnt`:** `lgkmcnt(n)` = LDS waits, `vmcnt(n)` = global memory waits. Lower n = more synchronization. Check these are not overly conservative.
- **Register spills:** Look for `buffer_store`/`buffer_load` to scratch memory

## AMD-Specific Triton Optimization Passes

| Pass | Description |
|------|-------------|
| AMD GPU accelerate Matmul | Optimize dot layout for AMD matrix cores |
| AMD GPU Stream Pipeline | Pipeline global loads through registers to shared memory |
| AMD GPU Reorder Instructions | Reduce register pressure, improve instruction order |
| AMD GPU Block Pingpong | Interleave instructions from two warps on same SIMD |
| AMD GPU Canonicalize Pointers | Rewrite pointers as (basePtr, offset) for better codegen |
| AMD GPU Convert To Buffer Ops | Convert memory ops to buffer operations when possible |
| Optimize AMD LDS Usage | Minimize LDS consumption for better occupancy |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `AMDGCN_ENABLE_DUMP=1` | Dump assembly |
| `MLIR_ENABLE_DUMP=1` | Dump Triton IR |
| `TRITON_PRINT_AUTOTUNING=1` | Print autotuning winner |
| `TRITON_ALWAYS_COMPILE=1` | Force recompilation |
| `HIP_FORCE_DEV_KERNARG=1` | Put kernel args in device memory (2-3μs reduction) |

## Performance Benchmarking

```bash
# Set deterministic clocks
rocm-smi --setperfdeterminism 1900

# Disable NUMA balancing
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

# Profile with rocprof
rocprofv3 --stats python your_kernel.py
```

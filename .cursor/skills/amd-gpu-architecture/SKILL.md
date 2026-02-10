---
name: amd-gpu-architecture
description: AMD GPU architecture reference for kernel development. Use when writing GPU kernels targeting AMD Instinct MI300X (CDNA3) or MI350 (CDNA4), understanding compute units, memory hierarchy, MFMA matrix instructions, wavefronts, XCDs, LDS, or when optimizing for AMD hardware. Covers CDNA3 and CDNA4 architectures, ISA details, and hardware-aware optimization strategies.
---

# AMD GPU Architecture for Kernel Developers

## CDNA3 Architecture (MI300X)

### Chip Topology

```
MI300X OAM Package
├── 8 XCDs (Accelerator Complex Dies) - connected via Infinity Fabric
├── 4 I/O Dies
└── 8 HBM3 Stacks (192 GB total, 5.3 TB/s)

Each XCD:
├── 40 CUs (38 active + 2 disabled for yield)
├── 4 MB L2 Cache (shared across all CUs)
├── 4 ACE Compute Accelerators
└── 1 HWS Hardware Scheduler
```

**Total: 304 active CUs, 32 MB L2, 192 GB HBM3**

### Compute Unit (CU) Architecture

```
Compute Unit
├── 4 SIMD Units (each executes one wavefront)
│   └── Each SIMD: 512 VGPRs, 16 VGPR allocation granularity
├── Scalar Unit (shared across all SIMDs)
│   └── 800 SGPRs
├── 64 KB LDS (Local Data Share / Shared Memory)
├── 32 KB L1 Cache
├── Matrix Cores (MFMA units)
└── Texture/Load Store Units
```

### Execution Model

| Concept | AMD Term | NVIDIA Equivalent | Value (MI300X) |
|---------|----------|-------------------|----------------|
| Thread group | Wavefront | Warp | 64 threads |
| Thread block | Workgroup | Block | Up to 1024 threads |
| Processor | Compute Unit (CU) | SM | 304 total |
| Vector registers | VGPR | Registers | 512 per SIMD |
| Shared memory | LDS | Shared memory | 64 KB per CU |
| Register file | VGPR+SGPR | RF | 512 VGPR + 800 SGPR per CU |

### Memory Hierarchy

| Level | Size | Latency | Bandwidth | Scope |
|-------|------|---------|-----------|-------|
| Registers (VGPR) | 512 per SIMD | 1 cycle | Highest | Per-wavefront |
| LDS | 64 KB per CU | ~20 cycles | ~10 TB/s per CU | Per-workgroup |
| L1 Cache | 32 KB per CU | ~50 cycles | Per-CU | Per-CU |
| L2 Cache | 4 MB per XCD | ~100 cycles | Shared across XCD | Per-XCD (40 CUs) |
| HBM3 | 192 GB total | ~300+ cycles | 5.3 TB/s aggregate | Global |

**Key insight for XCD-aware programming:** L2 cache is NOT shared across XCDs. Programs on different XCDs cannot share L2 data. This is why XCD remapping in GEMM kernels matters - keeping cooperating tiles on the same XCD improves L2 hit rates.

### MFMA (Matrix Fused Multiply-Add) Instructions

MFMA instructions perform block matrix multiply-accumulate on Matrix Cores.

| Instruction | M×N×K | Input Types | Output | Cycles |
|-------------|-------|-------------|--------|--------|
| `v_mfma_f32_16x16x16_f16` | 16×16×16 | FP16 | FP32 | 16 |
| `v_mfma_f32_16x16x16_bf16` | 16×16×16 | BF16 | FP32 | 16 |
| `v_mfma_f32_32x32x8_f16` | 32×32×8 | FP16 | FP32 | 32 |
| `v_mfma_f32_32x32x8_bf16` | 32×32×8 | BF16 | FP32 | 32 |
| `v_mfma_f32_16x16x32_fp8` | 16×16×32 | FP8 | FP32 | 16 |
| `v_mfma_f32_32x32x16_fp8` | 32×32×16 | FP8 | FP32 | 32 |
| `v_mfma_i32_16x16x32_i8` | 16×16×32 | INT8 | INT32 | 16 |
| `v_mfma_i32_32x32x16_i8` | 32×32×16 | INT8 | INT32 | 32 |

**In Triton:** `matrix_instr_nonkdim=16` selects the 16x16 variant; `=32` selects 32x32.

### Peak Performance Table

| Data Type | FLOPS/clock/CU | Peak TFLOPS (MI300X) |
|-----------|----------------|---------------------|
| FP64 (matrix) | 256 | 163.4 |
| FP32 (matrix) | 256 | 163.4 |
| TF32 | 1024 | 653.7 |
| FP16 / BF16 | 2048 | 1,307.4 |
| FP8 / INT8 | 4096 | 2,614.9 |

### Occupancy

Occupancy = number of active wavefronts per SIMD. Higher occupancy hides memory latency.

**Limiting factors:**
1. **VGPRs:** 512 per SIMD, allocated in units of 16

| VGPRs/wave | Max waves/SIMD |
|-----------|----------------|
| ≤128 | 4 |
| 129-170 | 3 |
| 171-256 | 2 |
| 257-512 | 1 |

2. **LDS:** 64 KB per CU. `occ_lds = floor(65536 / LDS_per_workgroup)`
3. **Workgroup count:** `occ = min(floor(occ_vgpr * 4 / num_warps), occ_lds) * num_warps / 4`

## CDNA4 Architecture (MI350)

### Key Differences from CDNA3

| Feature | CDNA3 (MI300X) | CDNA4 (MI350X) |
|---------|----------------|----------------|
| XCDs | 8 | 8 (3D stacked) |
| Memory | HBM3, 192 GB | HBM3E, larger capacity |
| New instructions | - | Scaled MFMA for MXFP4/MXFP8 |
| Power | 750W | 1000W |
| Arch name | gfx942 | gfx950 |

**Scaled MFMA (CDNA4 only):** Native support for OCP microscaling formats. In Triton: `tl.dot_scaled(a, b, a_scale, b_scale, ...)`.

## Hardware-Aware Optimization Strategies

### 1. Maximize Arithmetic Intensity

Arithmetic intensity = FLOPs / bytes transferred. For a problem to be compute-bound on MI300X:
- FP16: Need > 246 FLOPs/byte (1307 TFLOPS / 5.3 TB/s)
- FP8: Need > 493 FLOPs/byte

GEMM is compute-bound when M,N,K are large. Elementwise ops are always memory-bound.

### 2. Maximize Memory Bandwidth Utilization

- Use vectorized loads (`global_load_dwordx4` = 128 bits)
- Ensure coalesced access: consecutive threads access consecutive addresses
- Use `tl.assume(stride > 0)` and `tl.multiple_of(ptr, (16,))` to help vectorization

### 3. Use LDS Effectively

- LDS bandwidth: ~10 TB/s per CU (much faster than global memory)
- Used automatically by Triton for `tl.dot` operands and transposes
- Avoid bank conflicts: 32 banks, 4-byte width. Pad shared arrays if needed
- LDS reads/writes should use `_b128` or `_b64` instructions

### 4. Manage Register Pressure

- Use `waves_per_eu=N` to hint compiler to reduce VGPR usage
- Reduce live variables: chain computations instead of storing intermediates
- Avoid large `num_stages` with fused kernels (causes spills)
- Check spills: look for `buffer_store`/`buffer_load` to scratch in assembly

### 5. XCD-Aware Grid Mapping

Programs on the same XCD share L2 cache. For GEMM:
- Remap PIDs so tiles accessing similar K-strips land on same XCD
- `NUM_XCDS = 8` for MI300X, `= 8` for MI350X
- See XCD remapping code in aiter-triton-gemm-kernels skill

## Architecture Documentation Links

- CDNA3 ISA Reference: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- CDNA4 Whitepaper: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
- MI300 Microarchitecture: https://instinct.docs.amd.com/latest/gpu-arch/mi300.html
- GPU Architecture Docs: https://instinct.docs.amd.com/latest/gpu-arch/gpu-arch.html
- Register Pressure Guide: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/

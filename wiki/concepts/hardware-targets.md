---
title: "Hardware Targets"
last_verified: 2026-04-06
source_files:
  - aiter/jit/utils/chip_info.py
  - 3rdparty/composable_kernel/README.md
tags: [hardware, gpu, gfx, rocm, mi300]
---

# Hardware Targets

## Overview
aiter and CK kernels are compiled for specific AMD GPU architectures. The target architecture affects available instructions, compute unit count, and optimal kernel configurations.

## Supported GPU Families

| GFX ID | GPU Series | Compute Units | Key Features |
|--------|-----------|---------------|-------------|
| gfx908 | MI100 | 120 | MFMA, CDNA1 |
| gfx90a | MI200 (MI210, MI250, MI250X) | 104-220 | MFMA, CDNA2 |
| gfx942 | MI300 (MI300X, MI300A) | 228-304 | MFMA, CDNA3, unified memory (MI300A) |
| gfx1100 | RX 7900 XTX | 96 | WMMA, RDNA3 |
| gfx1101 | RX 7800 XT | 60 | WMMA, RDNA3 |
| gfx1102 | RX 7600 | 32 | WMMA, RDNA3 |

## Impact on aiter

### Tuned Configs
Config CSV files in `aiter/configs/` are indexed by `cu_num`. A shape tuned on a 304-CU MI300X will have different optimal kernels than on a 120-CU MI100.

### Attention Selector
The CK-UA attention selector uses `get_num_sms()` (CU count) to determine the occupancy zone:
- MI300X (304 CUs): CK-UA activates for 152-608 sequences (with 8 KV-heads)

### CK Compilation
CK must be compiled with `GPU_TARGETS` matching the deployment GPU. Different architectures support different MFMA instructions:
- CDNA (gfx9xx): `mfma_f32_16x16x32`, `mfma_f32_32x32x16`
- RDNA3 (gfx11xx): WMMA instructions instead

## Runtime Detection
`aiter/jit/utils/chip_info.py` detects the GPU at runtime:
- `get_gfx()` returns the architecture string
- `get_cu_num()` returns the compute unit count
- Used by GEMM config lookup and attention selector

## Related Pages
- [[concepts/autotuning]] -- per-architecture tuning
- [[concepts/backend-selection]] -- hardware affects backend choice
- [[ck/architecture]] -- CK compilation for targets

## Source Files
- `aiter/jit/utils/chip_info.py` -- runtime GPU detection
- `3rdparty/composable_kernel/README.md` -- supported architectures

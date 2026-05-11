---
title: "Composable Kernel Architecture"
last_verified: 2026-04-06
source_files:
  - 3rdparty/composable_kernel/README.md
tags: [ck, composable-kernel, architecture, hip]
---

# Composable Kernel Architecture

## Overview
Composable Kernel (CK) is AMD's HIP C++ library for writing performance-critical ML kernels. It is vendored in aiter at `3rdparty/composable_kernel/` as a git submodule. CK provides the kernel implementations that aiter's Python API wraps.

## Core Concepts

### Tile-based Programming Model
CK uses a tile-based model where computations are expressed as operations on fixed-size tiles of data. Each workgroup processes one or more tiles, with the tile size being a compile-time template parameter that affects register usage, occupancy, and performance.

### Tensor Coordinate Transformation
CK's key innovation for code reuse. Instead of writing separate kernels for every memory layout and access pattern, CK uses coordinate transformations to map logical tensor indices to physical memory locations. This allows the same kernel template to handle different layouts (row-major, column-major, strided, paged) without code duplication.

## Four-Layer Architecture

### Layer 1: Templated Tile Operators
- Lowest level: individual tile operations (GEMM tiles, softmax tiles, element-wise tiles)
- Parameterized by data types, tile sizes, and memory layouts
- Located in `include/ck_tile/`

### Layer 2: Templated Kernel and Invoker
- Composes tile operators into complete kernels
- Kernel: the GPU function (e.g., FMHA forward, GEMM universal)
- Invoker: host-side code that sets up arguments and launches the kernel
- Located in `include/ck_tile/ops/` and `include/ck/tensor_operation/gpu/`

### Layer 3: Instantiated Kernel and Invoker
- Concrete instantiations of templated kernels for specific type/size combinations
- Generated at build time for supported configurations
- Located in `library/`

### Layer 4: Client API
- High-level C++ API that users call
- Handles argument validation, memory allocation, kernel dispatch
- Located in `client_example/`

## CK vs CK Tile
CK has two generations:
- **CK (classic)**: Template-based approach using tensor operations. Used for GEMM, some attention.
- **CK Tile**: Newer tile-based approach with more explicit tile programming. Used for unified attention, newer FMHA variants, RMSNorm.

In aiter, both are used. The JIT system in `aiter/jit/` compiles the appropriate CK code based on the operator.

## GPU Target Architecture
CK kernels are compiled for specific GPU architectures:
- `gfx908` -- MI100
- `gfx90a` -- MI200 series
- `gfx942` -- MI300 series
- `gfx1100`, `gfx1101`, `gfx1102` -- RDNA3

The `GPU_TARGETS` CMake variable controls which architectures to build for. In aiter, the target is determined at runtime by `aiter/jit/utils/chip_info.py`.

## MFMA Instructions
CK leverages AMD's Matrix Fused Multiply-Add (MFMA) instructions:
- `mfma_f32_16x16x32` -- used in smaller tile sizes (attention decode)
- `mfma_f32_32x32x16` -- used in larger tile sizes (attention prefill, GEMM)
- These are the GPU equivalent of tensor cores, performing small matrix multiplies in hardware.

## Related Pages
- [[ck/attention-pipeline]] -- CK's unified attention pipeline
- [[concepts/backend-selection]] -- when CK vs Triton is selected
- [[concepts/hardware-targets]] -- GPU architecture details

## Source Files
- `3rdparty/composable_kernel/README.md` -- CK project overview
- `3rdparty/composable_kernel/include/ck_tile/` -- CK Tile headers
- `3rdparty/composable_kernel/include/ck/` -- classic CK headers

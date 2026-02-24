# Dot Operation Lowering Reference

How Triton lowers `DotOp` and `DotScaledOp` to MFMA, WMMA, or FMA instructions for AMD GPUs.

Source: [`DotOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM.cpp), [`DotOpToLLVM/MFMA.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp), [`DotOpToLLVM/WMMA.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/WMMA.cpp), [`DotOpToLLVM/FMA.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp)

## Dispatch Logic

`DotOpConversion::matchAndRewrite` checks the accumulator encoding:

```
D = A * B + C
encoding = D.type.encoding

AMDMfmaEncodingAttr  → convertMFMA(op, ...)
AMDWmmaEncodingAttr  → convertWMMA(op, ...)
BlockedEncodingAttr  → convertAMDFMADot(op, ...)
```

`ScaledDotOpConversion` handles `DotScaledOp` similarly:

```
AMDMfmaEncodingAttr  → convertScaledMFMA(op, ...)
AMDWmmaEncodingAttr  → convertScaledWMMA(op, ...)
```

## MFMA Lowering (CDNA1-4)

### Overview

Converts `triton::DotOp` with `AMDMfmaEncodingAttr` accumulator to ROCDL MFMA intrinsics. The encoding specifies the MFMA tile shape (e.g., 32x32, 16x16), transposition, warps-per-tile, and kPack.

### Key Steps

1. **Extract encoding parameters**: tile M/N/K, `isTransposed`, `warpsPerTile`, `kPack`
2. **Unpack operands**: A (LHS), B (RHS), C (accumulator) into per-thread LLVM values
3. **Compute repetitions**: `repM = shapeM / (mDim * warpsPerTile[0])`, `repN = shapeN / (nDim * warpsPerTile[1])`, `repK = shapeK / kWidth`
4. **Loop over tiles**: for each (m, n, k) repetition:
   - Extract A sub-tile and B sub-tile for this (m, k) and (n, k)
   - Pack operands into MFMA input format (VGPR)
   - Emit `rocdl.mfma.*` intrinsic with accumulator in AGPR
   - Accumulate result back into C
5. **Pack results** into output LLVM struct

### MFMA Intrinsic Selection

Based on tile shape and element types:

| Tile | Element Types | Intrinsic |
|------|--------------|-----------|
| 32x32 | f16 | `rocdl.mfma.f32.32x32x8f16` |
| 32x32 | bf16 | `rocdl.mfma.f32.32x32x8bf16` |
| 32x32 | i8 | `rocdl.mfma.i32.32x32x8i8` (gfx942: 32x32x16) |
| 16x16 | f16 | `rocdl.mfma.f32.16x16x16f16` |
| 16x16 | bf16 | `rocdl.mfma.f32.16x16x16bf16` |
| 16x16 | i8 | `rocdl.mfma.i32.16x16x16i8` (gfx942: 16x16x32) |
| 4x4 | f32 | `rocdl.mfma.f32.4x4x1f32` |
| 16x16 | f64 | `rocdl.mfma.f64.16x16x4f64` |

### Operand Layout

MFMA operands follow `DotOperandEncodingAttr`:
- **LHS (idx=0)**: elements packed per-thread with `kWidth` elements in the K dimension
- **RHS (idx=1)**: same packing, transposed relative to LHS
- **Accumulator**: `AMDMfmaEncodingAttr` — elements distributed across lanes within warp, across warps via `warpsPerCTA`

### kPack

`kPack` multiplies `kWidth` to increase the number of K elements packed per MFMA call. Used to amortize instruction overhead. Chain-dot tails restrict `kPack=1`.

## Scaled MFMA Lowering (CDNA4)

`convertScaledMFMA` handles `DotScaledOp` → `v_mfma_scale_f32_*_f8f6f4`:

1. Extract scale factors for A and B operands
2. Determine format encoding via `cbsz`/`blgp` fields (fp8, bf8, fp6, bf6, fp4)
3. Emit `rocdl.mfma.scale.*` with extra `a_scale`, `b_scale` operands

Also handles SMFMAC (scaled indexed MFMA) path for structured sparsity patterns.

## WMMA Lowering (GFX1250)

### Overview

Converts `triton::DotOp` with `AMDWmmaEncodingAttr` accumulator to WMMA intrinsics. GFX1250 supports both 256-bit and 128-bit WMMA operand widths.

### Key Steps

1. **Extract encoding parameters**: version, `isTransposed`, warp bases
2. **Unpack operands** into per-thread values
3. **Compute repetitions** based on WMMA tile shape and warp distribution
4. **Loop over tiles**: emit WMMA intrinsic per tile
5. **Pack results**

### WMMA Instructions (GFX1250)

| Variant | A/B Width | Accumulator |
|---------|-----------|-------------|
| WMMA256b | 256 bits/lane | AGPR |
| WMMA128b | 128 bits/lane | AGPR |

Operands: `$vdst` (AGPR acc output), `$src0` (VGPR A), `$src1` (VGPR B), `$src2` (AGPR acc input).

### Scaled WMMA (GFX1250)

`convertScaledWMMA` handles `DotScaledOp` with fp8/fp4 inputs and per-block scaling. Scale layout determined by `get_wmma_scale_layout(dot_operand_layout, shape)`.

## FMA Fallback

`convertAMDFMADot` handles `DotOp` with `BlockedEncodingAttr` — no matrix core, uses scalar FMA:

1. **Unpack** A, B, C into per-thread elements
2. **Nested loops** over M, N, K dimensions (per-thread tile)
3. **Emit** `llvm.fma` or integer multiply-add per element
4. **Pack** results

This path is used when:
- The dot doesn't qualify for matrix core acceleration (too small, unsupported types)
- `AccelerateBlocked` pattern in the transform phase selected FMA over matrix core
- V_DOT instructions are used on supported architectures (`supportsVDot`)

## Operand Packing

### MFMA Operand Format

Per-thread operand packing for MFMA depends on element type:

| Element Type | Elements/Thread | Packed As |
|-------------|----------------|-----------|
| f16 | 4 per K step | v4f16 → 2×i32 |
| bf16 | 4 per K step | v4bf16 → 2×i32 |
| i8 | 4 per K step | i32 (4×i8 packed) |
| fp8/bf8 | 4+ per K step | i32 (packed bytes) |
| f32 | 1 per K step | f32 |
| f64 | 1 per K step | f64 |

### Accumulator Layout

MFMA accumulator elements per thread depend on tile size:
- 32x32: 16 elements/thread (across 4 blocks of 4)
- 16x16: 4 elements/thread
- 4x4: 4 elements/thread (but fewer active lanes)

WMMA accumulator: 8 elements/thread for 16x16 tile (32 lanes in a warp).

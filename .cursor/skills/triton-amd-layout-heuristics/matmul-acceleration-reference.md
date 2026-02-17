# Matmul Acceleration Reference (AMD)

## Overview

The `tritonamdgpu-accelerate-matmul` pass converts `DotOp` and `DotScaledOp` from blocked layouts to hardware-native MFMA or WMMA encodings.

Source: [`third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp)

## ISA Family Detection

```
CDNA1: gfx908  → mfmaVersion=1
CDNA2: gfx90a  → mfmaVersion=2
CDNA3: gfx942  → mfmaVersion=3
CDNA4: gfx950  → mfmaVersion=4
RDNA3: gfx11xx → wmmaVersion=1
RDNA4: gfx12xx → wmmaVersion=2  (except gfx1250)
GFX1250:       → wmmaVersion=3
```

## Pattern Priority (per ISA family)

### CDNA4

| Priority | Pattern | Handles |
|----------|---------|---------|
| 4 | `ScaledBlockedToScaledMFMAF8F6F4` | Native scaled MFMA (F8F6F4) |
| 3 | `DecomposeAMDScaledBlocked` | Decompose scaled dot to regular dot + explicit scaling |
| 2 | `BlockedToMFMA` | Regular DotOp → MFMA |
| 2 | `ScaledBlockedToMFMA` | DotScaledOp with fp16 emulation |

CDNA4's `BlockedToMFMA` also tries `V_MFMA_*_F8F6F4` scaled instructions when both operands are F8/F6/F4, falling back to regular MFMA.

### CDNA1-3

| Priority | Pattern | Handles |
|----------|---------|---------|
| 2 | `BlockedToMFMA` | Regular DotOp → MFMA |
| 2 | `ScaledBlockedToMFMA` | DotScaledOp with emulation |

### GFX1250

| Priority | Pattern | Handles |
|----------|---------|---------|
| 4 | `ScaledBlockedToScaledWMMAF8F6F4` | Native scaled WMMA 16x16x128 |
| 3 | `DecomposeAMDScaledBlocked` | Decompose scaled dot |
| 2 | `BlockedToWMMA` | Regular DotOp → WMMA |

### RDNA3/4

| Priority | Pattern | Handles |
|----------|---------|---------|
| 3 | `DecomposeScaledBlocked` | Generic decomposition |
| 2 | `BlockedToWMMA` | Regular DotOp → WMMA |

After all matrix core patterns, `AccelerateBlocked` (priority 1) handles FMA fallback.

## MFMA Tile Selection

`chooseMfmaInstruction(loc, mfmaVersion, cType, aElemType, bElemType, inputKSize, enforcedNonKDim, withScale, allowXF32)`

### Default heuristic (when enforcedNonKDim == 0):

```
minSize = min(M, N)
if minSize >= 32:
    mDim = nDim = 32      (16 for f64)
elif minSize >= 16:
    mDim = nDim = 16
elif minSize >= 4:
    if M >= 64: mDim=64, nDim=4
    elif N >= 64: mDim=4, nDim=64
```

Then calls `MfmaIntrinsic::selectFor(mfmaVersion, mDim, nDim, inputKSize, ...)` to find a matching instruction. Fails if `inputKSize % kDim != 0` (would cause data duplication).

### User override

Pass `matrix_instr_nonkdim` to force a specific tile size: `mDim = nDim = enforcedNonKDim`.

## WMMA Tile Selection

Always 16x16. `kDim` determined by `WmmaIntrinsic::selectFor(wmmaVersion, 16, 16, inputKSize, aElemType, bElemType, cElemType)`.

## Warp Distribution: `warpsPerTile`

### Case 1: Batched matmul (rank 3)

```
return {numWarps, 1, 1}
```

### Case 2: Chain-dot (Flash Attention)

Detection: `isChainDotHead(dotOp)` / `isChainDotTail(dotOp)` analyze the DFG for two dots where the result of the first feeds operand A of the second.

**Head dot** (1st dot in chain):
```
return {numWarps, 1}
```
Rationale: eliminates inter-warp reduction in softmax + avoids layout conversion to 2nd dot's operand 0.

**Tail dot** (2nd dot in chain):
```
ret[0] = min(numWarps, ceil(shape[0] / mDim))
ret[1] = numWarps / ret[0]
```
Fill M first, remainder to N. Avoids register pressure from over-distributing small M (decode kernels with large head dim).

### Case 3: Regular

Iteratively double warps in the dimension with more remaining tiles:
```
while ret[0] * ret[1] < numWarps:
    if (shape[0] / (mDim*2) / ret[0]) >= (shape[1] / nDim / ret[1]):
        ret[0] *= 2 (if room)
    else:
        ret[1] *= 2
```

If `ret[1] * nDim > shape[1]`, swap M and N.

## AMDMfmaEncodingAttr Parameters

| Parameter | Heuristic |
|-----------|-----------|
| `isTransposed` | `true` unless mDim=4 && nDim=64 |
| `tilesPerWarp` | Default `{1,1}`. CDNA4 16x16 chain-dot head feeding opA: `{2,1}`; feeding opB: `{1,2}`. Scaled MFMA: deduced by `deduceTilesPerWarpForScale` |
| `accBitwidth` | i32 for int, f64 for f64, f32 otherwise |

## kWidth / kPack

```
kBase = per-instruction elements per thread  (e.g. kDim/2 for mfma_32, kDim/4 for mfma_16)
kWidth = kBase
if not chain-dot tail:
    kWidth *= kPack    // kPack=1 or 2 from user option
if f16 && chain-dot tail:
    kWidth = 4         // forces no-op mma→dotOp conversion
```

`kWidth` controls `DotOperandEncodingAttr` and thus shared memory vectorization (`ds_read` width).

## `deduceTilesPerWarpForScale` (Scaled MFMA)

For native scaled MFMA on CDNA4, tries all `{1,2} x {1,2}` tilesPerWarp combinations:

1. For each combo, constructs the scale layout via `chooseScaledMfmaScaleLayout`
2. Back-propagates to infer the global load layout via `inferSourceLoadLayout`
3. Computes vectorization via `largestVectorisation` on the composed register→shared layout
4. Picks the combo maximizing `vecSizeA + vecSizeB`

## Operand Type Selection (WMMA)

`selectMatrixCoreOperandTypes` uses a cost model:
- No-conversion = 0 cost
- Widening cast = 1 cost
- Narrowing or int↔float = max cost (fallback to FMA)

Supported WMMA type combos: `{f16,f16,f32,f32}`, `{bf16,bf16,f32,f32}`, `{i8,i8,i32,i32}`, plus fp8 variants for WMMA v2/v3.

## FMA Fallback (`AccelerateBlocked`)

When no matrix core instruction matches:
1. Tries V_DOT acceleration (fp16×fp16→fp32 or i8×i8→i32) on supported archs
2. Falls back to FMA: casts all operands to a common type (f16 if all ≤16bit, else f32)

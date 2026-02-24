# Layout Conversion, Elementwise, Scheduling, and Packed Ops Reference

Detailed reference for AMD-specific register shuffles, elementwise type conversions, instruction scheduling control, and packed float scalarization.

Source: [`ConvertLayoutOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertLayoutOpToLLVM.cpp), [`ElementwiseOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp), [`SchedInstructions.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/SchedInstructions.cpp), [`ScalarizePackedFOps.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/ScalarizePackedFOps.cpp)

## AMD-Specific Layout Conversions

Two AMD-specific `ConvertLayoutOp` patterns avoid costly shared memory round-trips by working entirely in registers.

### Permlane Swap (`ConvertLayoutOpPermlaneSwap`)

**Availability**: CDNA4 (gfx950) — requires `targetInfo.supportsPermlaneSwap()`.

Uses `v_permlane16_swap` and `v_permlane32_swap` instructions for intra-warp data exchange. These instructions swap values between pairs of lanes that differ in bit 4 or bit 5 of the lane ID.

**When it matches**: `cvtNeedsWarpShuffle(srcTy, dstTy)` returns true, and the conversion decomposes into transpositions of the form `(r_i, l4)` or `(r_i, l5)` where `r_i` is a register index bit and `l4`/`l5` are lane index bits.

**Decomposition**: `getWarpLayoutConvertDecomposition(srcTy, dstTy, bitwidth)` returns:
- `pReg`: register-to-register permutation
- `pLane`: lane-to-lane permutation
- `mixedTranspositions`: list of register-lane transpositions
- `nPack`: number of intra-register packing bits (elements packed within a 32-bit register)

**Handling**:
1. Apply `pReg` to reorder register values
2. Pack elements into 32-bit words (handling 16-bit and sub-32-bit types)
3. Execute `permlane_swap` for each tile of registers:
   - Simple transposition `(r_i, l4)`: one `permlane16_swap` call
   - Simple transposition `(r_i, l5)`: one `permlane32_swap` call
   - Three-cycle `(r_i, l4, l5)`: factored as `(r_i, l4)(r_i, l5)`, two swap calls
4. Unpack results, restore element types
5. Handle broadcasting via `broadcastAs`

**Use cases**: MFMA-to-DotOperand conversions and epilogue store vectorization optimization in chained matmul kernels (flash attention).

### In-Thread VPerm (`ConvertLayoutOpInThreadSwap`)

**Availability**: all architectures (uses `v_perm` which is universally available).

**Constraint**: only matches 8-bit element types, register-only conversions (no lane/warp dimensions), minimum 4 values.

`v_perm` copies 4 bytes from two source i32 registers with arbitrary byte selection:

```
v_perm dst, src1, src2, selector
  byte i of dst = src_byte[selector_nibble[i]]
  selector_nibble[i] ∈ {0-3: src2 bytes, 4-7: src1 bytes}
```

**Multi-stage algorithm**:

1. **Repack inputs** into i32 registers (4 bytes each)
2. **One-way deps**: dst register uses bytes from a single src register → one `v_perm` or simple copy
3. **Two-way deps**: dst register uses bytes from exactly 2 src registers → one `v_perm`
4. **Four-way deps**: dst register uses bytes from 3-4 src registers:
   - Assemble byte pairs from each dst register's low/high halves
   - Merge compatible pairs into quads (each quad from at most 2 src registers)
   - Materialize quads as temporary registers via `v_perm`
   - Combine temporaries into final dst registers via `v_perm`

**Example** (4-way dependency):
```
src0=[0,1,2,3], src1=[4,5,6,7], src2=[8,9,10,11], src3=[12,13,14,15]
dst1 wants bytes: (src0,b0),(src1,b1),(src2,b2),(src3,b3) → [0,5,10,15]

Stage 1: tmp1 = v_perm(src1, src0, ...) → [0,5,1,4]   // bytes from src0+src1
         tmp2 = v_perm(src3, src2, ...) → [10,15,11,14] // bytes from src2+src3
Stage 2: dst1 = v_perm(tmp2, tmp1, ...) → [0,5,10,15]   // final assembly
```

## Elementwise Type Conversions

### FP8/BF8 Conversion Architecture

Two implementation paths based on ISA family:

**Software path** (CDNA1-3, RDNA): full bit-manipulation implementation
- Extract sign, exponent, mantissa
- Round-to-nearest-even with tie-breaking bias
- Subnormal handling via lookup table of halfway points
- Saturation: overflow → largest normal, preserve NaN
- FNUZ variants: additional handling for unsigned zero (`-0 → +0`)

**Hardware path** (CDNA4, GFX1250): packed ROCDL conversion intrinsics

### Conversion Functions

| Direction | SW Function | HW Function | Batch Size |
|-----------|------------|-------------|------------|
| FP16 → FP8 E4M3FN | `Fp16_to_Fp8E4M3FN_RTNE_SW` | `cvtScalePk4DowncastToFp8` | 4 or 8 |
| FP16 → BF8 E5M2 (RTNE) | `Fp16_to_Fp8E5M2_RTNE_SW` | `cvtScalePk4DowncastToFp8` | 4 or 8 |
| FP16 → BF8 E5M2 (RTZ) | `Fp16_to_Fp8E5M2_RTZ` (byte extract) | — | 4 |
| FP32 → FP8 E4M3FN | `Fp32_to_Fp8E4M3FN_RTNE_SW` | `cvtScalePk4DowncastToFp8` | 4 or 8 |
| FP32 → BF8 E5M2 | `Fp32_to_Fp8E5M2_RTNE_SW` | `cvtScalePk4DowncastToFp8` | 4 or 8 |
| FP8 E4M3FN → FP32 | — | `cvtScalePkUpcastFromFp8` | 4 or 8 |
| BF8 E5M2 → FP32 | — | `cvtScalePkUpcastFromFp8` | 4 or 8 |
| FP8 → FP32 (CDNA3) | `cvtPkF8ToFp32` | — | 4 |
| FP32 → FP8 (CDNA3) | `cvtPkFp32ToF8` | — | 4 |

### HW Conversion Intrinsics

**4-element packed** (`cvtScalePk4DowncastToFp8`):
```
pack 4 values → v2i16 via ConvertOp (dstLoHiSel=false, then true)
bitcast v2i16 → v4i8, extract individual bytes
```

**8-element packed** (`cvtScalePk8DowncastToFp8`):
```
pack 8 values → v8 input vector
ConvertOp → v2i32 result (opscale=0b1000 for bias 127)
bitcast → v8i8, extract bytes
```

**Upcast** (`cvtScalePkUpcastFromFp8`):
```
pack 4 i8 → i32 via bitcast
ConvertOp(i32, scale=1.0, srcLoHiSel=false) → v2<dstType> (low pair)
ConvertOp(i32, scale=1.0, srcLoHiSel=true)  → v2<dstType> (high pair)
extract individual elements
```

### SW Downcast Algorithm (`downcastToFp8_RTNE_oneValue`)

Template function parametric on source and destination FP types:

1. Extract sign, exponent, mantissa from source representation
2. Check for NaN via `llvm.is.fpclass`
3. Compute rounding bias for RTNE: `(mantissa_lsb >> reduced_bits) + base_bias`
4. Add bias to achieve round-to-nearest-even
5. Mask mantissa to destination precision
6. Clamp to smallest destination normal (eliminate denormals before bias adjustment)
7. Adjust exponent bias: `result -= (srcBias - dstBias) << srcMantissa`
8. Shift and truncate to 8 bits
9. Handle overflow: values > `dstMaxOfSrcType` → saturate to largest normal
10. Handle subnormals: LUT-based halfway-point comparison (4 or 8 entries)
11. Preserve NaN, restore sign
12. For FNUZ types: correct `-0 → +0`

### ISA Family Routing

`isCDNA4OrHigher()` selects HW vs SW path:

```cpp
ConverterT Fp16_to_Fp8E4M3FN_RTNE(AMD::ISAFamily isaFamily) {
  return isCDNA4OrHigher(isaFamily) ? HW_converter : SW_converter;
}
```

Where `isCDNA4OrHigher` = `CDNA4 || GFX1250`.

## Instruction Scheduling Hints

### InsertInstructionSchedHints Pass

Walks the module looking for `scf.for` loops containing chain-dot patterns (flash attention). When found, inserts `triton::amdgpu::InstructionSchedHint` at the start of the loop body.

Chain-dot detection: `isChainDotHead(dotOp)` identifies the first dot in a chain-dot pair.

### LowerInstructionSchedHints Pass

Converts `InstructionSchedHint` to ROCDL scheduling intrinsics:

**`SchedHint::attention`**:
```
Block start:  ROCDL::SchedBarrier(none)     // prevent upward instruction motion
              ... loop body ...
Block end:    ROCDL::IglpOpt(2)             // enable IGLP scheduling strategy 2
              ROCDL::SchedBarrier(none)      // prevent downward instruction motion
```

**`ROCDL::SchedBarrier(mask)`**: maps to `s_sched_barrier <mask>`. Controls which instruction types can cross the barrier:
- `none` (0): no instructions can cross (strongest barrier)
- Other masks allow specific types (VALU, VMEM, etc.)

**`ROCDL::IglpOpt(value)`**: maps to `s_iglp_opt <value>`. Activates an instruction group level parallelism strategy in the AMDGPU backend scheduler:
- `value=2`: interleave strategy suitable for attention-like compute patterns

**Important**: `SchedBarrier` and `IglpOpt` should not be mixed according to AMDGPU backend docs — the `limitSchedulingRange` mode (attention) uses barriers at block boundaries only to constrain IglpOpt's scope.

### When Scheduling Hints Are Used

Scheduling hints are only inserted for `SchedHint::attention` (controlled by a pass parameter). The pass is a no-op for `SchedHint::none`. Currently limited to a single `tt.dot` per `scf.for` block.

## Scalarize Packed Float Ops

### Problem

LLVM's `VectorCombine::foldPermuteOfBinops` optimization pass (runs during `optimize_module`) can combine scalar float operations into packed vector operations like `v_pk_mul_f32` / `v_pk_add_f32`. While packed ops use separate VALU execution units from MFMA/WMMA tensor cores, the instructions themselves **cannot be issued in the same cycle** as matrix core instructions. This creates issue hazards that degrade performance in MFMA/WMMA-heavy blocks.

### Solution

`ScalarizePackedFOps` is an **LLVM IR FunctionPass** (not MLIR) that runs after the LLVM optimization pipeline:

1. Scan each basic block for MFMA or WMMA intrinsic calls (`isMFMAorWMMA`)
2. In blocks containing matrix core ops, find vector `fmul`/`fadd`/`fsub` instructions
3. Replace each vector binary op with per-element scalar equivalents:
   ```
   Before: %r = fmul <2 x float> %a, %b        → v_pk_mul_f32
   After:  %a0 = extractelement %a, 0
           %b0 = extractelement %b, 0
           %r0 = fmul float %a0, %b0             → v_mul_f32_e32
           %a1 = extractelement %a, 1
           %b1 = extractelement %b, 1
           %r1 = fmul float %a1, %b1             → v_mul_f32_e32
           %r = <r0, r1>
   ```
4. Scalar ops (`v_mul_f32_e32`, `v_add_f32_e32`) interleave freely with MFMA/WMMA issue slots

### Invocation

Called via `mlir::triton::AMD::runScalarizePackedFOpsPass(Function &F)` on each LLVM function after the optimization pipeline.

### Scope

Only processes `FMul`, `FAdd`, `FSub` on fixed-length vector types. Only in basic blocks that also contain MFMA/WMMA calls. Does not affect blocks without matrix core operations.

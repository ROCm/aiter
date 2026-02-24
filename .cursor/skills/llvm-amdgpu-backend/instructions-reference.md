# AMDGPU Instruction Categories Reference

Detailed instruction formats and operand constraints for CDNA3 (gfx942), CDNA4 (gfx950), and GFX1250.

Source: [`llvm/lib/Target/AMDGPU/`](https://github.com/llvm/llvm-project/tree/main/llvm/lib/Target/AMDGPU)

## MFMA Instructions (CDNA3/4)

Defined in [`VOP3PInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/VOP3PInstructions.td).

### Standard MFMA Format (VOPProfileMAI)

```
V_MFMA_<outtype>_<M>x<N>x<K>_<intype> $vdst, $src0, $src1, $src2, $cbsz, $abid, $blgp
```

| Operand | Register class | Description |
|---------|---------------|-------------|
| vdst | AGPR (accumulator) | Output accumulator, can alias src2 |
| src0 | VGPR | A matrix tile |
| src1 | VGPR | B matrix tile |
| src2 | AGPR | Input accumulator |
| cbsz | 3-bit imm | Control bits |
| abid | 3-bit imm | Accumulator buffer ID (indexed MFMA) |
| blgp | 3-bit imm | Block-level group pattern |

### MFMA Instructions (gfx942 — CDNA3)

| Instruction | Output | Input | Tile |
|-------------|--------|-------|------|
| `v_mfma_f32_4x4x1_f32` | 4×f32 | f32 | 4×4 |
| `v_mfma_f32_16x16x1_f32` | 16×f32 | f32 | 16×16 |
| `v_mfma_f32_32x32x1_f32` | 32×f32 | f32 | 32×32 |
| `v_mfma_f32_4x4x2_f16` | 4×f32 | v4f16 | 4×4 |
| `v_mfma_f32_16x16x4_f16` | 16×f32 | v4f16 | 16×16 |
| `v_mfma_f32_32x32x8_f16` | 32×f32 | v4f16 | 32×32 |
| `v_mfma_f32_4x4x2_bf16` | 4×f32 | v2bf16 | 4×4 |
| `v_mfma_f32_16x16x4_bf16` | 16×f32 | v2bf16 | 16×16 |
| `v_mfma_f32_32x32x8_bf16` | 32×f32 | v2bf16 | 32×32 |
| `v_mfma_i32_4x4x4_i8` | 4×i32 | i32(4×i8) | 4×4 |
| `v_mfma_i32_16x16x16_i8` | 16×i32 | i32(4×i8) | 16×16 |
| `v_mfma_i32_32x32x8_i8` | 32×i32 | i32(4×i8) | 32×32 |
| `v_mfma_f64_4x4x4_f64` | 4×f64 | f64 | 4×4 |
| `v_mfma_f64_16x16x4_f64` | 16×f64 | f64 | 16×16 |

Also: `_bf16_1k` (1K bf16) and `_f16` (1K f16) variants for some tiles.

### Scaled Indexed MFMA — SMFMAC (gfx950 only)

Predicate: `HasGFX950Insts`. Format (VOPProfileSMFMAC):

```
V_SMFMAC_<type>_<M>x<N>x<K>_<intype> $vdst, $src0, $src1, $idx, $cbsz, $abid
```

| Instruction | Output | Input | Tile |
|-------------|--------|-------|------|
| `v_smfmac_f32_16x16x64_bf16` | 16×f32 | bf16 | 16×16 |
| `v_smfmac_f32_32x32x32_bf16` | 32×f32 | bf16 | 32×32 |
| `v_smfmac_f32_16x16x64_f16` | 16×f32 | f16 | 16×16 |
| `v_smfmac_f32_32x32x32_f16` | 32×f32 | f16 | 32×32 |
| `v_smfmac_i32_16x16x128_i8` | 16×i32 | i8 | 16×16 |
| `v_smfmac_i32_32x32x64_i8` | 32×i32 | i8 | 32×32 |
| `v_smfmac_f32_16x16x128_bf8_*` | 16×f32 | fp8/bf8 | 16×16 |
| `v_smfmac_f32_32x32x64_bf8_*` | 32×f32 | fp8/bf8 | 32×32 |

`$idx` is a VGPR index for scaled tile addressing.

### MFMA Scale — FP8/FP6/FP4 (gfx950 only)

Predicate: `HasGFX950Insts`.

```
V_MFMA_SCALE_F32_<M>x<N>x<K>_F8F6F4 $vdst, $src0, $src1, $src2, $a_scale, $b_scale, cbsz:N, blgp:N
```

Example: `v_mfma_scale_f32_16x16x128_f8f6f4` — `cbsz`/`blgp` encode input format (fp8, bf8, fp6, bf6, fp4).

Extra operands: `$a_scale` (VGPR), `$b_scale` (VGPR) for per-element scaling.

## WMMA Instructions (GFX1250)

Defined in [`VOP3PInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/VOP3PInstructions.td). Predicate: `HasGFX1250Insts`.

### Standard WMMA Format

```
V_WMMA_<outtype>_<M>x<N>x<K>_<intype> $vdst, $src0, $src1, $src2
```

| Operand | Register class | Description |
|---------|---------------|-------------|
| vdst | AGPR | Accumulator output |
| src0 | VGPR | A matrix tile |
| src1 | VGPR | B matrix tile |
| src2 | AGPR | Accumulator input |

Two categories:
- **WMMA256b** (`HasWMMA256bInsts`): A/B operands are 256 bits per lane (duplicated)
- **WMMA128b** (`HasWMMA128bInsts`): A/B operands are 128 bits per lane

### Scaled WMMA (GFX1250)

```
V_WMMA_SCALED_<type> $vdst, $src0, $a_scale, $a_format, $src1, $b_scale, $b_format, $src2
```

Supports fp8/fp4 inputs with per-block scaling. Scale layout retrieved via `get_wmma_scale_layout(dot_operand_layout, shape)`.

## Buffer Instructions (MUBUF)

Defined in [`BUFInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/BUFInstructions.td).

### Address Modes (BUFAddrKind)

| Mode | Effective address | Use case |
|------|-------------------|----------|
| Offset | `V# + soffset + offset` | Scalar-only addressing |
| OffEn | `V# + voffset + soffset + offset` | Per-lane variable offset |
| IdxEn | `V# + vindex + soffset + offset` | Indexed (structured) |
| BothEn | `V# + vindex + voffset + soffset + offset` | Index + variable offset |
| Addr64 | `V# + vaddr(64-bit) + offset` | 64-bit pointer addressing |

### Operands

| Operand | Register class | Bits | Description |
|---------|---------------|------|-------------|
| vdata | VGPR or AGPR (AVLdSt) | varies | Load destination / store source |
| vaddr | VGPR | 32 or 64 | Variable offset / index |
| srsrc | SReg_128_XNULL | 128 | Buffer resource descriptor (V#, 4×32-bit) |
| soffset | SCSrc_b32 or SReg_32 | 32 | Scalar offset |
| offset | immediate | 12 | Byte offset (0–4095) |
| cpol | — | — | Cache policy (GLC/SLC/DLC or TH/Scope) |

### Buffer V# (Resource Descriptor) Format

128-bit `v4i32` in SReg_128:
- **Dword 0**: Base address [31:0]
- **Dword 1**: Base address [47:32] + stride [63:48] (structured) or swizzle flags
- **Dword 2**: Num records (buffer size in elements or bytes)
- **Dword 3**: Format (data format, num format) + type/flags

Constructed by `buildRSRC()` in [`SIISelLowering.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIISelLowering.cpp).

### Instruction Variants

```
BUFFER_LOAD_UBYTE / SBYTE / USHORT / SSHORT     8/16-bit loads (zero/sign-extend)
BUFFER_LOAD_DWORD / DWORDX2 / DWORDX3 / DWORDX4 32–128-bit loads
BUFFER_STORE_BYTE / SHORT / DWORD / DWORDX2 / X3 / X4
BUFFER_LOAD_FORMAT_X / XY / XYZ / XYZW           formatted loads (texture-style)
BUFFER_LOAD_FORMAT_D16_*                          D16 (16-bit per component) loads
BUFFER_ATOMIC_ADD / SUB / SMIN / SMAX / UMIN / UMAX / AND / OR / XOR / INC / DEC / CMPSWAP
```

**CDNA4 (gfx950)**: DWORDX3/X4 `_LDS` variants for direct-to-LDS DMA.

**GFX1250**: `hasFormattedMUBUFInsts()` = false (no formatted buffer loads), `hasMTBUFInsts()` = false.

### TFE and LDS Variants

- `_TFE`: texture fail enable (extra dword for status)
- `_LDS` (gfx950): load data goes directly to LDS (not through VGPRs)

## Flat / Global / Scratch Instructions

Defined in [`FLATInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/FLATInstructions.td).

### Flat (address space 0)

```
FLAT_LOAD_UBYTE / SBYTE / USHORT / SSHORT / DWORD / DWORDX2 / X3 / X4
FLAT_STORE_BYTE / SHORT / DWORD / DWORDX2 / X3 / X4
FLAT_ATOMIC_SWAP / ADD / SUB / SMIN / SMAX / UMIN / UMAX / AND / OR / XOR / INC / DEC / CMPSWAP
```

| Operand | Type | Notes |
|---------|------|-------|
| vaddr | VGPROp_64 | 64-bit virtual address |
| offset | 13-bit signed | Byte offset (–4096 to +4095) |
| cpol | — | Cache policy |

### Global (address space 1)

```
GLOBAL_LOAD_UBYTE / SBYTE / USHORT / SSHORT / DWORD / DWORDX2 / X3 / X4
GLOBAL_STORE_BYTE / SHORT / DWORD / DWORDX2 / X3 / X4
GLOBAL_ATOMIC_*
```

| Addressing | Operands | Notes |
|------------|----------|-------|
| VGPR-only | `vaddr(64-bit) + offset` | |
| `_SADDR` | `saddr(SReg_64) + vaddr(32-bit) + offset` | Scalar base + per-lane offset |

GFX90A+: `_RTN_agpr` variants for AGPR atomics.

#### GFX1250 Async Global→LDS

From [`FLATInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/FLATInstructions.td):

```
global_load_lds_dword / dwordx2 / dwordx3 / dwordx4    (+ _SADDR variants)
global_store_lds_dword                                   (+ _SADDR variants)
```

Async path (`IsAsync = 1`): `vdst` holds DS address (no M0), increments ASYNC_CNT, `SchedRW = [WriteVMEM, WriteLDS]`.
Legacy path (`IsAsync = 0`): M0 holds DS address, increments LGKM_CNT.

### Scratch (address space 5)

```
SCRATCH_LOAD_UBYTE / SBYTE / USHORT / SSHORT / DWORD / DWORDX2 / X3 / X4
SCRATCH_STORE_BYTE / SHORT / DWORD / DWORDX2 / X3 / X4
```

| Variant | Addressing |
|---------|-----------|
| `_SADDR` | `saddr(SGPR) + offset` |
| `_SVS` | `saddr(SGPR) + vaddr(VGPR) + offset` |
| `_ST` | `offset` only (stack-relative) |

GFX1250: `hasScaleOffset()`, `hasSignedScratchOffsets()`.

## DS (LDS/GDS) Instructions

Defined in [`DSInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/DSInstructions.td).

### Standard DS Instructions

```
DS_READ_B32 / B64 / B128          LDS reads
DS_WRITE_B32 / B64 / B128         LDS writes
DS_READ2_B32 / B64                Two-element strided reads
DS_WRITE2_B32 / B64               Two-element strided writes
DS_ADD_U32 / F32, DS_SUB_U32      LDS atomics
DS_MIN_I32/U32, DS_MAX_I32/U32    LDS min/max
DS_CMPST_B32 / B64                LDS compare-and-swap
```

| Operand | Type | Notes |
|---------|------|-------|
| addr | VGPR_32 | Byte address (relative to LDS base) |
| data0/data1 | VGPR or AGPR (GFX90A+) | Data operands |
| offset | 16-bit imm | Byte offset |
| gds | 1-bit | 0=LDS, 1=GDS |

GFX9 (CDNA3/4): M0 must be initialized for LDS access (`glueCopyToM0LDSInit`).

`_gfx9` pseudo variants: suppress M0 read in encoding. `_agpr` variants: AGPR for data operands (GFX90A+).

### DS Transpose Reads (gfx950, Wave64)

Predicate: `HasGFX950Insts`, `isWave64`.

| Instruction | Output | Element size | Description |
|-------------|--------|-------------|-------------|
| `ds_read_b64_tr_b4` | 64-bit | 4-bit | FP4 transpose read |
| `ds_read_b64_tr_b8` | 64-bit | 8-bit | FP8 transpose read |
| `ds_read_b64_tr_b16` | 64-bit | 16-bit | FP16/BF16 transpose read |
| `ds_read_b96_tr_b6` | 96-bit | 6-bit | FP6/BF6 transpose read |

### DS Transpose Loads (gfx1250, Wave32)

Predicate: `isGFX1250Plus`, `isWave32`.

| Instruction | Output | Predicate | Description |
|-------------|--------|-----------|-------------|
| `ds_load_tr4_b64` | 64-bit | `HasTransposeLoadF4F6Insts` | FP4 transpose load |
| `ds_load_tr6_b96` | 96-bit | `HasTransposeLoadF4F6Insts` | FP6 transpose load |
| `ds_load_tr8_b64` | 64-bit | — | FP8 transpose load |
| `ds_load_tr16_b128` | 128-bit | — | FP16 transpose load |

### Async Barrier (gfx1250)

Predicate: `HasLdsBarrierArriveAtomic`.

| Instruction | Description |
|-------------|-------------|
| `ds_atomic_async_barrier_arrive_b64` | Async barrier arrive (uses ASYNC_CNT, not LGKM_CNT) |
| `ds_atomic_barrier_arrive_rtn_b64` | Barrier arrive with return value |

## VOP3P Packed Instructions

Defined in [`VOP3PInstructions.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/VOP3PInstructions.td).

### Packed Arithmetic

```
V_PK_ADD_F16 / I16 / U16      packed 2× add
V_PK_MUL_F16                  packed 2× multiply
V_PK_FMA_F16                  packed 2× FMA
V_PK_MAD_I16 / U16            packed 2× multiply-add
V_PK_MIN_F16, V_PK_MAX_F16    packed 2× min/max
V_PK_LSHLREV_B16, V_PK_ASHRREV_I16, V_PK_LSHRREV_B16
```

### Mix Instructions (F16↔F32 mixed precision)

```
V_FMA_MIX_F32             mixed-precision FMA producing F32 from F16/BF16 inputs
V_FMA_MIXLO_F16           produces low F16 from mixed inputs
V_FMA_MIXHI_F16           produces high F16 from mixed inputs
V_FMA_MIX_F32_BF16        BF16 variant
V_FMA_MIXLO_BF16          BF16 low result
V_FMA_MIXHI_BF16          BF16 high result
```

### Dot Product Instructions

```
V_DOT2_F32_F16            2× F16 dot → F32
V_DOT2_F32_BF16           2× BF16 dot → F32
V_DOT4_I32_I8             4× I8 dot → I32
V_DOT4_U32_U8             4× U8 dot → U32
V_DOT8_I32_I4             8× I4 dot → I32
V_DOT8_U32_U4             8× U4 dot → U32
```

### CDNA4 (gfx950) Packed Additions

```
V_PK_ADD_MAX_I16 / U16    packed add then max
V_PK_ADD_MIN_I16 / U16    packed add then min
V_PK_MAX3_I16             packed 3-way max
V_PK_MIN3_I16             packed 3-way min
V_DOT4_F32_FP8_FP8        4× FP8 dot → F32
V_DOT4_F32_FP8_BF8        mixed FP8/BF8 dots
V_DOT4_F32_BF8_FP8
V_DOT4_F32_BF8_BF8
```

## Scalar Instructions (SMEM)

Defined in [`SIInstrInfo.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInstrInfo.td).

```
S_LOAD_DWORD / DWORDX2 / X4 / X8 / X16     scalar loads from constant memory
S_BUFFER_LOAD_DWORD / DWORDX2 / X4 / X8     scalar buffer loads (via SGPR V#)
S_STORE_DWORD / DWORDX2 / X4                scalar stores (GFX10+)
```

| Operand | Type | Notes |
|---------|------|-------|
| sbase | SReg_64 | Base address (64-bit) |
| offset | SGPR or imm20/21 | Byte offset |

GFX1250: `hasScalarSubwordLoads()` for 8/16-bit scalar loads.

## Encoding Families

From [`GCNProcessors.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNProcessors.td):

| Family | Targets | ISA Feature |
|--------|---------|-------------|
| GFX940 | gfx940, gfx941, gfx942 | `FeatureISAVersion9_4_2` |
| GFX950 | gfx950 | `FeatureISAVersion9_5_0` |
| GFX12 | gfx1200, gfx1201, gfx1250 | `FeatureISAVersion12_50` |

## Feature Predicates (TableGen)

From [`AMDGPU.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPU.td) and [`AMDGPUFeatures.td`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUFeatures.td):

| Predicate | Meaning | Targets |
|-----------|---------|---------|
| `isGFX940Plus` | GFX940+ instruction set | gfx940/941/942/950 |
| `HasGFX950Insts` | GFX950 instructions | gfx950 only |
| `isGFX12Plus` | GFX12 generation | gfx1200/1201/1250 |
| `HasGFX1250Insts` | GFX1250 instructions | gfx1250 only |
| `HasMAIInsts` | MFMA matrix instructions | gfx908+ (CDNA) |
| `HasWMMAInsts` | WMMA matrix instructions | gfx1250 |
| `HasWMMA256bInsts` | 256-bit WMMA operands | gfx1250 |
| `HasWMMA128bInsts` | 128-bit WMMA operands | gfx1250 |
| `HasDot11Insts` | FP8 dot products | gfx950 |
| `HasPermlane16Swap` | Permlane16 swap | gfx950 |
| `HasPermlane32Swap` | Permlane32 swap | gfx950 |
| `HasLDSLoadB96_B128` | LDS 96/128-bit loads | gfx950 |
| `HasTransposeLoadF4F6Insts` | TR4/TR6 transpose loads | gfx1250 |
| `HasLdsBarrierArriveAtomic` | LDS barrier arrive | gfx1250 |
| `HasAshrPkInsts` | Packed arithmetic shift | gfx950 |
| `HasFP8ConversionScaleInsts` | FP8 conversion with scale | gfx950 |
| `HasBF8ConversionScaleInsts` | BF8 conversion with scale | gfx950 |
| `HasFP4ConversionScaleInsts` | FP4 conversion with scale | gfx950 |
| `HasFP6BF6ConversionScaleInsts` | FP6/BF6 conversion | gfx950 |
| `HasF16BF16ToFP6BF6ConversionScaleInsts` | F16/BF16→FP6/BF6 | gfx950 |
| `HasF32ToF16BF16ConversionSRInsts` | F32→F16/BF16 stochastic rounding | gfx950 |
| `HasCvtPkF16F32Inst` | CVT_PK F16↔F32 | gfx950 |
| `HasMinimum3Maximum3F32` | min3/max3 F32 | gfx950 |
| `HasMinimum3Maximum3PKF16` | packed min3/max3 F16 | gfx950 |

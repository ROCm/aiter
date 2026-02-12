# MFMA & WMMA Matrix Core Reference for AMD GPUs

## Source Files

- MFMA intrinsics: [`third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp)
- WMMA intrinsics: [`third_party/amd/lib/TritonAMDGPUTransforms/WmmaGroup.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUTransforms/WmmaGroup.cpp)
- MFMA lowering: [`third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp)
- WMMA lowering: [`third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/WMMA.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/WMMA.cpp)
- Target info: [`third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp)
- Gluon CDNA3: [`python/triton/experimental/gluon/language/amd/cdna3/__init__.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/cdna3/__init__.py)
- Gluon CDNA4: [`python/triton/experimental/gluon/language/amd/cdna4/__init__.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/cdna4/__init__.py)
- Gluon GFX1250: [`python/triton/experimental/gluon/language/amd/gfx1250/__init__.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/gfx1250/__init__.py)

## MFMA (CDNA3/CDNA4)

### Basic MFMA

```python
acc = ttgl.amd.cdna3.mfma(a, b, acc)
# Also: ttgl.amd.cdna4.mfma(a, b, acc)
```

- `a`: LHS operand with `DotOperandLayout(0, mfma_layout, k_width)`
- `b`: RHS operand with `DotOperandLayout(1, mfma_layout, k_width)`
- `acc`: accumulator with `AMDMFMALayout`

### Scaled MFMA (CDNA4 only)

```python
acc = ttgl.amd.cdna4.mfma_scaled(
    a, a_scale, a_format,
    b, b_scale, b_format,
    acc
)
```

- `a_format`, `b_format`: `"e2m1"`, `"e4m3"`, `"e5m2"` (OCP Microscaling)
- `a_scale`, `b_scale`: scale tensors (uint8), or `None` for default (0x7F = 1.0)
- Scale layout: `ttgl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape)`

### MFMA Instruction Shapes

| M x N | K (f16/bf16) | K (f32) | K (f64) | K (xf32) | K (i8) | K (fp8) |
|-------|--------------|---------|---------|----------|--------|---------|
| 32x32 | 8, 16 | 2 | - | 4 (v3+) | 8, 16 | 16 (v3), 32 (v4) |
| 16x16 | 16, 32 | 4 | 4 (v2+) | 8 (v3+) | 16, 32 | 32 (v3), 64 (v4) |
| 64x4 | 16 | - | - | - | - | - |
| 4x64 | 16 | - | - | - | - | - |

### CDNA4-Only Scaled MFMA

| M x N | K | Formats |
|-------|---|---------|
| 32x32 | 64 | f8f6f4 (e2m1, e4m3, e5m2) |
| 16x16 | 128 | f8f6f4 (e2m1, e4m3, e5m2) |

### MFMA Setup Pattern (CDNA3)

```python
# Layouts
MFMA_LAYOUT = ttgl.amd.AMDMFMALayout(3, [32, 32, 16], True, [4, 1])
OPERAND_A = ttgl.DotOperandLayout(0, MFMA_LAYOUT, 4)  # k_width=4 for f16
OPERAND_B = ttgl.DotOperandLayout(1, MFMA_LAYOUT, 4)

# Shared memory
SHARED_A = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0])
SHARED_B = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0])

# Computation
a = smem_a.load(layout=OPERAND_A)
b = smem_b.load(layout=OPERAND_B)
acc = ttgl.amd.cdna3.mfma(a, b, acc)
```

### MFMA Setup Pattern (CDNA4 Scaled)

```python
MFMA_LAYOUT = ttgl.amd.AMDMFMALayout(4, [32, 32, 64], True, [4, 1])
OPERAND_A = ttgl.DotOperandLayout(0, MFMA_LAYOUT, 16)
OPERAND_B = ttgl.DotOperandLayout(1, MFMA_LAYOUT, 16)

# Get scale layouts
scale_layout_a = ttgl.amd.cdna4.get_mfma_scale_layout(OPERAND_A, scale_shape_a)
scale_layout_b = ttgl.amd.cdna4.get_mfma_scale_layout(OPERAND_B, scale_shape_b)

acc = ttgl.amd.cdna4.mfma_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
```

## WMMA (GFX1250)

### Basic WMMA

```python
acc = ttgl.amd.gfx1250.wmma(a, b, acc)
```

- `a`: LHS with `DotOperandLayout(0, wmma_layout, k_width)`
- `b`: RHS with `DotOperandLayout(1, wmma_layout, k_width)`
- `acc`: accumulator with `AMDWMMALayout(version=3, ...)`
- All layouts must have matching `version=3`

### Scaled WMMA (GFX1250)

```python
acc = ttgl.amd.gfx1250.wmma_scaled(
    a, a_scale, a_format,
    b, b_scale, b_format,
    acc
)
```

- `a_format`, `b_format`: `"e2m1"`, `"e4m3"`, `"e5m2"`
- If `e2m1`: operand `instr_shape` must be `[16, 16, 64]`
- Accumulator `instr_shape` must be `[16, 16, 128]`
- Scale layout: `ttgl.amd.gfx1250.get_wmma_scale_layout(dot_operand_layout, shape)`

### WMMA Instruction Shapes

| M x N | K | Input Types |
|-------|---|-------------|
| 16x16 | 16 | f16, bf16, i8, i4 |
| 16x16 | 32 | f16, bf16, fp8, i8 |
| 16x16 | 64 | fp8 (e2m1 operands) |
| 16x16 | 128 | scaled (e2m1/e4m3/e5m2) |

### WMMA Hardware Intrinsics

```
llvm.amdgcn.wmma.f32.16x16x16.f16
llvm.amdgcn.wmma.f32.16x16x32.f16
llvm.amdgcn.wmma.f32.16x16x16.bf16
llvm.amdgcn.wmma.f32.16x16x32.bf16
llvm.amdgcn.wmma.i32.16x16x16.iu8
llvm.amdgcn.wmma.i32.16x16x16.iu4
llvm.amdgcn.wmma.scale.*  (scaled variants)
```

### WMMA Setup Pattern (GFX1250)

```python
import math

num_warps = 4  # wave32
warp_bases = [(0, 1)]
for i in range(int(math.log2(num_warps // 2))):
    warp_bases.append((1 << i, 0))
warp_bases = tuple(warp_bases)

# Layouts
WMMA_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 32])
OPERAND_A = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)  # k_width=8 for f16
OPERAND_B = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

# Shared memory (PaddedShared preferred for GFX1250)
SHARED_A = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0])
SHARED_B = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0])

# Computation
a = smem_a.load(layout=OPERAND_A)
b = smem_b.load(layout=OPERAND_B)
acc = ttgl.amd.gfx1250.wmma(a, b, acc)
```

### WMMA Scaled Setup (GFX1250 MXFP)

```python
WMMA_SCALED = ttgl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 128])
OPERAND_A = ttgl.DotOperandLayout(0, WMMA_SCALED, 32)
OPERAND_B = ttgl.DotOperandLayout(1, WMMA_SCALED, 32)

acc = ttgl.amd.gfx1250.wmma_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)
```

## Hardware Details

### Warp Sizes

- CDNA1/2/3/4: **64 threads** (wave64)
- GFX1250 (RDNA): **32 threads** (wave32)

### Shared Memory

- GFX1250: 320 KB total, 64 KB partition size
- CDNA3/4: varies by config

### Key Differences: MFMA vs WMMA

| Feature | MFMA (CDNA) | WMMA (GFX1250) |
|---------|-------------|-----------------|
| Warp size | 64 | 32 |
| Layout class | `AMDMFMALayout` | `AMDWMMALayout` |
| Base shapes | 32x32, 16x16, 64x4, 4x64 | 16x16 |
| Async data movement | async_copy (pointer/buffer) | TDM (descriptor-based) |
| Barrier type | commit_group/wait_group | mbarrier (init/wait/arrive) |
| Scheduling | warp_pipeline_stage | warp_specialize |

## Warp Pipeline (CDNA3/CDNA4)

Software pipelining via scheduling hints:

```python
from triton.experimental.gluon.language.amd.warp_pipeline import warp_pipeline_stage

for k in gl.range(0, K, one):
    with amd.warp_pipeline_stage("load", priority=3):
        a = ttgl.amd.cdna3.buffer_load(a_ptr, offs_a)
        b = ttgl.amd.cdna3.buffer_load(b_ptr, offs_b)

    with amd.warp_pipeline_stage("prep"):
        a_tile = smem_a.load(layout=OPERAND_A)
        b_tile = smem_b.load(layout=OPERAND_B)

    with amd.warp_pipeline_stage("compute", priority=0):
        acc = ttgl.amd.cdna3.mfma(a_tile, b_tile, acc)
```

- Priority 0-3, maps to `s_setprio` hardware instruction
- Stages form a pipeline cluster; operations within a stage execute as a unit
- Unspecified priority resets to 0 if any stage uses explicit priority

## Scheduling and Performance

- [`SchedInstructions.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/lib/TritonAMDGPUToLLVM/SchedInstructions.cpp): uses `ROCDL::SchedBarrier` and `ROCDL::IglpOpt`
- IGLP value 2 for attention-style kernels
- `SetPrioOp` placed next to first MFMA for priority control
- LDS transpose V1 (CDNA4), V2 (GFX1250) for efficient shared memory access
- `ds_read_tr4` for FP4 data (CDNA4)
- `supportsPermlaneSwap()` = true on GFX1250 (cross-lane permutations)

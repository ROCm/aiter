# Gluon Layouts Reference for AMD GPUs

## Source Files

- Base layouts: [`python/triton/experimental/gluon/language/_layouts.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/_layouts.py)
- AMD layouts: [`python/triton/experimental/gluon/language/amd/_layouts.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/_layouts.py)
- MLIR defs: [`include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)
- Linear layout conversions: [`lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp)

## Distributed Layouts

### BlockedLayout

Partitions tensor across threads, warps, and CTAs.

```python
BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cga_layout=[])
```

- `size_per_thread`: elements per thread per dimension
- `threads_per_warp`: threads per warp per dimension (product = warp_size)
- `warps_per_cta`: warps per CTA per dimension
- `order`: dimension ordering (e.g., `[1, 0]` = column-major access)
- `cga_layout`: optional CTA tiling bases for multi-CTA

**Example (GFX1250, wave32):**

```python
# 4 warps, 32 threads/warp, 8 elements/thread along N
BLOCKED = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
```

**Example (CDNA3/4, wave64):**

```python
# 4 warps, 64 threads/warp
BLOCKED = ttgl.BlockedLayout([1, 4], [16, 4], [4, 1], [1, 0])
```

### AMDMFMALayout (CDNA3/CDNA4)

```python
AMDMFMALayout(version, instr_shape, transposed, warps_per_cta,
              element_bitwidth=32, tiles_per_warp=None, cga_layout=[])
```

- `version`: 1=gfx908, 2=gfx90a, **3=gfx942 (CDNA3)**, **4=gfx950 (CDNA4)**
- `instr_shape`: `[M, N, K]` — valid M,N pairs: `[32,32]`, `[16,16]`, `[64,4]`, `[4,64]`
- `transposed`: if True, each thread holds consecutive row elements (good for chained dot + global write)
- `warps_per_cta`: warp arrangement in CTA
- `element_bitwidth`: 32 or 64
- `tiles_per_warp`: optional tile layout within a warp

**CDNA3 examples:**

```python
# 32x32 MFMA, 4 warps arranged [4,1], K=16 for f16
MFMA_LAYOUT = ttgl.amd.AMDMFMALayout(3, [32, 32, 16], True, [4, 1])

# 16x16 MFMA, 2 warps, K=32
MFMA_LAYOUT = ttgl.amd.AMDMFMALayout(3, [16, 16, 32], True, [2, 1])
```

**CDNA4 examples:**

```python
# 32x32 MFMA v4, K=32 for f16
MFMA_LAYOUT = ttgl.amd.AMDMFMALayout(4, [32, 32, 32], True, [4, 1])

# Scaled MFMA: 32x32, K=64 for f8f6f4
MFMA_LAYOUT = ttgl.amd.AMDMFMALayout(4, [32, 32, 64], True, [4, 1])
```

### AMDWMMALayout (GFX1250)

```python
AMDWMMALayout(version, transposed, warp_bases, reg_bases=[], instr_shape=[16,16,16],
              cga_layout=[], rank=2)
```

- `version`: 1=RDNA3, 2=RDNA4, **3=gfx1250**
- `transposed`: result tensor transposition
- `warp_bases`: list of basis vectors for warp-level distribution
- `reg_bases`: optional register-level bases
- `instr_shape`: `[M, N, K]` — default `[16, 16, 16]`, also `[16,16,32]`, `[16,16,64]`, `[16,16,128]`
- `cga_layout`: optional CTA tiling bases

**GFX1250 examples:**

```python
# 4 warps: warp_bases define how warps tile the output
# [(0,1)] means first warp bit maps to N-dim
# [(1,0)] means second warp bit maps to M-dim
WARP_BASES = [(0, 1), (1, 0)]
WMMA_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

# 8 warps:
WARP_BASES = [(0, 1), (1, 0), (2, 0)]
WMMA_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32])

# With CGA (multi-CTA):
cga_layout_a = make_cga_layout(ctas_per_cga, [ctas_per_cga[0], 1], [0, 1])
WMMA_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_a)
```

**Computing warp_bases from num_warps:**

```python
import math
warp_bases = [(0, 1)]
for i in range(int(math.log2(num_warps // 2))):
    warp_bases.append((1 << i, 0))
warp_bases = tuple(warp_bases)
```

### DotOperandLayout

Wraps a parent MMA layout for use as dot operand.

```python
DotOperandLayout(operand_index, parent, k_width)
```

- `operand_index`: 0 = LHS (A), 1 = RHS (B)
- `parent`: the MMA layout (`AMDMFMALayout` or `AMDWMMALayout`)
- `k_width`: elements per 32 bits (e.g., 8 for f16 with 32-bit groups)

```python
OPERAND_A = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)   # LHS
OPERAND_B = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)   # RHS
```

### SliceLayout

Removes one dimension from a parent layout.

```python
SliceLayout(dim, parent)
```

Used for 1D index tensors derived from 2D accumulator layout:

```python
offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, ACC_LAYOUT))
offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, ACC_LAYOUT))
```

### DistributedLinearLayout

Explicit linear layout with bases at register, lane, warp, and block levels.

```python
DistributedLinearLayout(reg_bases, lane_bases, warp_bases, block_bases, shape)
```

Reference: https://arxiv.org/abs/2505.23819

## Shared Memory Layouts

### PaddedSharedLayout (AMD preferred)

Padding + element reordering to avoid bank conflicts.

```python
PaddedSharedLayout(interval_padding_pairs, offset_bases, cga_layout, shape)
```

- `interval_padding_pairs`: list of `[interval, padding]`, both powers of 2
- `offset_bases`: linear mapping from 1D offset to n-D tensor elements
- `cga_layout`: bases for block-level shared memory offsets
- `shape`: n-D logical shape

**Convenience constructor:**

```python
PaddedSharedLayout.with_identity_for(interval_padding_pairs, shape, order, cga_layout=[])
```

**Examples:**

```python
# A matrix: [BLOCK_M, BLOCK_K], padding every BLOCK_K elements by 8
SHARED_A = ttgl.PaddedSharedLayout.with_identity_for(
    [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0])

# B matrix (non-transposed): [BLOCK_K, BLOCK_N], padding every BLOCK_N by 16
SHARED_B = ttgl.PaddedSharedLayout.with_identity_for(
    [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0])

# B matrix (transposed): [BLOCK_N, BLOCK_K], padding every BLOCK_K by 8
SHARED_B_T = ttgl.PaddedSharedLayout.with_identity_for(
    [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0])

# With CGA layout:
SHARED_A = ttgl.PaddedSharedLayout.with_identity_for(
    [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0], cga_layout_a)
```

### SwizzledSharedLayout

Generic swizzled shared memory.

```python
SwizzledSharedLayout(vec, per_phase, max_phase, order, cga_layout=[])
```

- `vec`: vector width for swizzling
- `per_phase`: elements per swizzle phase
- `max_phase`: max swizzle phases
- `order`: dimension ordering

Used for accumulator shared buffers and mbarriers:

```python
SHARED_ACC = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
```

### MBarrierLayout (GFX1250)

Special `SwizzledSharedLayout` subclass for mbarrier synchronization:

```python
from triton.experimental.gluon.language.amd.gfx1250.mbarrier import MBarrierLayout
layout = MBarrierLayout()  # SwizzledSharedLayout(1, 1, 1, [0])
```

## CGA (Cooperative Grid Array) Layouts

For multi-CTA kernels:

```python
from triton._C.libtriton.gluon_ir import make_cga_layout

cga_layout_a = make_cga_layout(ctas_per_cga, [ctas_per_cga[0], 1], [0, 1])
cga_layout_b = make_cga_layout(ctas_per_cga, [1, ctas_per_cga[1]], [0, 1])
cga_layout_c = make_cga_layout(ctas_per_cga, ctas_per_cga, [0, 1])
```

## Layout Debugging

```python
layout.format_tensor_view(shape)    # show element-to-thread mapping
layout.format_hardware_view(shape)  # show hardware-level mapping
```

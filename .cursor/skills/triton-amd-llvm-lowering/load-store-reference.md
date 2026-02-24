# Load/Store Lowering Reference

Detailed load/store lowering from TritonGPU dialect to LLVM for AMD GPUs.

Source: [`LoadStoreOpToLLVM.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp), [`BufferOpsEmitter.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp), [`AtomicRMWOpsEmitter.cpp`](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/AtomicRMWOpsEmitter.cpp)

## Load Conversion Patterns

| Pattern | Source Op | Target | Key Feature |
|---------|----------|--------|-------------|
| `LoadOpConversion` | `triton::LoadOp` | `llLoad` (global_load/flat_load) | Per-thread pointers in VGPRs |
| `BufferLoadOpConversion` | `triton::amdgpu::BufferLoadOp` | `rocdl.raw.ptr.buffer.load` | Scalar base + 32-bit offsets |
| `BufferLoadToLocalOpConversion` | `triton::amdgpu::BufferLoadToLocalOp` | `rocdl.raw.ptr.buffer.load.lds` | Direct-to-LDS (bypass VGPRs) |
| `GlobalLoadToLocalOpConversion` | `triton::amdgpu::GlobalLoadToLocalOp` | `global_load_lds` | Global → LDS direct path |
| `LocalStoreFromGlobalOpConversion` | `triton::amdgpu::LocalStoreFromGlobalOp` | `global_store_lds` | LDS → Global direct path |

## Global Load (`LoadOpConversion`)

```
triton::LoadOp(ptr, mask, other) → vectorized llLoad
```

1. **Unpack** per-thread pointer elements from the LLVM struct
2. **Determine vector size**: `getVectorSize(ptr, axisAnalysis)` — max contiguous elements per thread. Capped by mask alignment.
3. **Loop over vectors**: for each group of `vec` elements:
   - Build `VectorType(valueElemTy, vec)`
   - Create `falseVal` from `other` elements (or zeros)
   - Emit `llLoad(ptr, vecTy, pred, falseVal, multicastMask, cacheMod)`
4. **Extract** individual elements from loaded vectors
5. **Pack** all loaded elements into result LLVM struct

### Vectorization Rules

`getVectorSize` computes the contiguous access vector width:
- Based on `AxisInfo` contiguity analysis of the pointer tensor
- Capped at `128 bits / elemBits` (maximum 16 bytes per load)
- Further capped by mask alignment (mask bits must be uniform across the vector)

### Multicast (Multi-CTA)

When `targetInfo.supportsMultiCTALaunch()` (GFX1250), loads can multicast to multiple CTAs:
- `emitCtaMulticastMask()` computes a bitmask of which CTAs should receive the data
- Passed as an operand to `llLoad`

## Buffer Load (`BufferLoadOpConversion`)

```
triton::amdgpu::BufferLoadOp(ptr, offsets, mask, other, stride) → buffer_load
```

1. **Create resource descriptor**: `BufferEmitter::createResourceDescriptor(llPtr, llStride)` — 128-bit V# descriptor in SGPRs
2. **Determine vector size**: `getVectorSize(ptr, offset, axisAnalysis)`, also incorporates `op.getContiguity()` hint
3. **Loop over vectors**: for each group of `vec` elements:
   - Emit `bufferEmitter.emitLoad(vecTy, rsrcDesc, offset, pred, falseVal, cacheMod)`
4. **Unpack** and pack results

### Buffer Resource Descriptor (V#)

`BufferEmitter::createResourceDescriptor()` constructs a `v4i32`:
- **Dword 0-1**: base pointer (48-bit)
- **Dword 2**: `num_records` = max 32-bit unsigned (for OOB masking)
- **Dword 3**: format flags, stride info

### OOB Masking Strategy

Buffer ops support native out-of-bounds handling:
- `num_records` set to `0xFFFFFFFF` (max size)
- For masked-off threads: offset set to `0x80000001` (exceeds `num_records` → nop)
- No branch divergence needed for masking

### Buffer Load Types

`getBufferOpType()` packs element types for efficient buffer operations:
- `vector<8xf16>` → bitcast to `vector<4xi32>` (more efficient ROCDL lowering)
- Atomics: always operate on native types

## Direct-to-LDS Load (`BufferLoadToLocalOpConversion`)

```
triton::amdgpu::BufferLoadToLocalOp(ptr, offsets, dest, mask, other, stride) → buffer_load_lds
```

Bypasses VGPRs — data flows directly from global memory into LDS. Uses `LinearLayout` to compute per-thread shared memory addresses.

### Lowering Flow

1. **Check feasibility**: `canLoadDirectToLDS(targetInfo, ptrType, dstEnc, allocShape, vec)` verifies alignment and architecture support
2. **Compute LDS addresses**: build `regToShared` layout via `LinearLayout::invertAndCompose`
3. **Handle swizzling** (non-scattering architectures):
   - For GFX9 (CDNA3): no per-lane scattering, must use warp-coalesced writes
   - Compute `swizzledLaneOffsets` — difference between swizzled and flat LDS offsets
   - Shuffle source pointers/offsets across lanes to match swizzled LDS layout
4. **Emit thread predicate**: `emitRedundantThreadPredicate()` masks threads that hold duplicate data
5. **Emit buffer_load_lds**: `bufferEmitter.emitLoadToLds(type, byteWidth, rsrcDesc, offset, shmemAddr, pred, cacheMod)`
6. **Handle `other`** values: if masked-off threads need a fallback value, emit a separate LDS store for those elements

### Scattering vs Non-Scattering

| Feature | GFX9 (CDNA3) | CDNA4/GFX1250 |
|---------|-------------|---------------|
| Per-lane LDS address | No (warp base only) | Yes |
| Swizzle handling | Shuffle source ptrs | Native per-lane addressing |
| `laneId` for LDS addr | Set to 0 (warp base) | Actual lane ID |

### Swizzle Handling (Non-Scattering)

When shared memory uses swizzled encoding and the architecture doesn't support scattering:
1. Compute `emitSwizzledLaneOffsets()` — per-vector-load delta between swizzled and flat LDS offset
2. Use `shuffleIdx` to read source pointer/offset from the lane that should write to our LDS address
3. Use `shuffleMask` (ballot-based) to propagate mask bits to match

## Store Conversion Patterns

| Pattern | Source Op | Target |
|---------|----------|--------|
| `StoreOpConversion` | `triton::StoreOp` | `llStore` (global_store/flat_store) |
| `BufferStoreOpConversion` | `triton::amdgpu::BufferStoreOp` | `rocdl.raw.ptr.buffer.store` |

Store lowering mirrors load lowering: unpack elements → pack into vectors → emit vectorized stores with mask predication.

## Atomic Operations

### Buffer Atomics

`BufferAtomicRMWOpConversion` handles `triton::amdgpu::BufferAtomicRMWOp`:
1. Create buffer resource descriptor
2. Emit memory fences based on ordering (relaxed/acquire/release/acq_rel)
3. Call `bufferEmitter.emitAtomicRMW(rmwType, type, rsrcDesc, offset, data, pred, hasUsers)`
4. Emit post-atomic fence if needed

### Fence Emission

```
Relaxed:                → no fences
Release (agent):        → buffer_wbl2 sc1=1, s_waitcnt vmcnt(0)  [before atomic]
Acquire (agent):        → s_waitcnt vmcnt(0), buffer_inv sc1=1    [after atomic]
Release (workgroup):    → no fences (same L1/L2)
Acquire (workgroup):    → no fences (same L1/L2)
AcquireRelease:         → release fence before + acquire fence after
```

Implemented via `LLVM::FenceOp` with AMDHSA sync scope strings: `"agent"` (GPU), `"workgroup"` (CTA), `""` (system).

### Global Atomics

`AtomicRMWOpConversion` and `AtomicCASOpConversion` handle `triton::AtomicRMWOp` and `triton::AtomicCASOp`:
- Uses `AtomicRMWOpsEmitter` which decides between global atomics and buffer atomics based on pointer type
- Vectorized atomics: split into per-element operations when vector width > 1

## Cache Modifiers

`CacheModifier` enum maps to ROCDL cache policy:
- `NONE`: default caching
- `CA`: cache all levels
- `CG`: cache in global (L2)
- `CV`: cache volatile
- `WB`: write-back
- `WT`: write-through

Passed to `llLoad`/`llStore`/`BufferEmitter` methods.

## Key Helper Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `llLoad` | `Utility.cpp` | Emit vectorized global/flat load with mask |
| `llStore` | `Utility.cpp` | Emit vectorized global/flat store with mask |
| `getVectorSize` | `Utility.cpp` | Compute max contiguous vector elements |
| `createZeroVector` | `LoadStoreOpToLLVM.cpp` | Create zero-initialized vector for `other` |
| `packElementRangeIntoVector` | `LoadStoreOpToLLVM.cpp` | Pack scalar elements into LLVM vector |
| `getMaskElemsAndUpdateVeclen` | `LoadStoreOpToLLVM.cpp` | Unpack mask and cap vector size |
| `emitRedundantThreadPredicate` | `LoadStoreOpToLLVM.cpp` | Mask threads with duplicate data |
| `canLoadDirectToLDS` | `Utility.cpp` | Check if direct-to-LDS is feasible |
| `emitCtaMulticastMask` | `Utility.cpp` | Compute CTA multicast bitmask |

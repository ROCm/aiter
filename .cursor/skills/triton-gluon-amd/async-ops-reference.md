# Gluon Async Operations & Memory Reference for AMD GPUs

## Source Files

- GFX1250 TDM: [`python/triton/experimental/gluon/language/amd/gfx1250/tdm.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/gfx1250/tdm.py)
- GFX1250 async_copy: [`python/triton/experimental/gluon/language/amd/gfx1250/async_copy.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/gfx1250/async_copy.py)
- GFX1250 mbarrier: [`python/triton/experimental/gluon/language/amd/gfx1250/mbarrier.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/gfx1250/mbarrier.py)
- GFX1250 cluster: [`python/triton/experimental/gluon/language/amd/gfx1250/cluster.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/gfx1250/cluster.py)
- CDNA4 async_copy: [`python/triton/experimental/gluon/language/amd/cdna4/async_copy.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/cdna4/async_copy.py)
- CDNA3 buffer ops: [`python/triton/experimental/gluon/language/amd/cdna3/__init__.py`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/python/triton/experimental/gluon/language/amd/cdna3/__init__.py)
- AMD ops MLIR: [`third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td`](https://github.com/triton-lang/triton/blob/2b1031d66167eb7f0b55d464935b69aa76b3aff4/third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td)

## TDM (Tensor Data Movement) — GFX1250 Only

TDM is the AMD equivalent of NVIDIA TMA. Descriptor-based async tensor moves between global and shared memory.

### Tensor Descriptor

```python
desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
    base=ptr,           # pointer to global memory
    shape=(M, N),       # tensor shape (1-5 dimensions)
    strides=(stride_m, stride_n),  # tensor strides (i64)
    block_shape=(BLOCK_M, BLOCK_N),  # tile shape
    layout=shared_layout  # PaddedSharedLayout or SwizzledSharedLayout
)
```

Properties: `desc.dtype`, `desc.block_shape`, `desc.shape`, `desc.strides`, `desc.layout`

### Async Load (global -> shared)

```python
ttgl.amd.gfx1250.tdm.async_load(
    src=desc,              # tensor descriptor
    offsets=[off_m, off_n],  # offsets from base
    dest=smem_tile,        # shared memory descriptor
    pred=1,                # predicate (optional, default 1)
    mbarrier=bar           # barrier to signal on completion (optional)
)
```

Hardware: lowers to `tensor_load_to_lds`.

### Async Store (shared -> global)

```python
ttgl.amd.gfx1250.tdm.async_store(
    dest=desc,             # tensor descriptor
    offsets=[off_m, off_n],
    src=smem_tile,         # shared memory descriptor
    mbarrier=bar           # optional barrier
)
```

Hardware: lowers to `tensor_store_from_lds`.

### Async Gather (non-contiguous read)

```python
ttgl.amd.gfx1250.tdm.async_gather(
    desc=desc,              # source tensor descriptor (must be 2D)
    src_row_indices=indices, # 1D tensor of row indices (int16 or int32)
    src_col_offset=col_off,  # starting column offset
    dst=smem_tile,          # shared memory destination (must be 2D)
    mbarrier=bar            # optional
)
```

Index dtype determines capacity per instruction: int16 = 16 rows, int32 = 8 rows.

### Async Scatter (non-contiguous write)

```python
ttgl.amd.gfx1250.tdm.async_scatter(
    desc=desc,               # dest tensor descriptor (must be 2D)
    dst_row_indices=indices,  # 1D tensor of row indices
    dst_col_offset=col_off,
    src=smem_tile,           # shared memory source (must be 2D)
    mbarrier=bar             # optional
)
```

### Async Wait

```python
ttgl.amd.gfx1250.tdm.async_wait(num_outstanding=0)
```

Blocks until outstanding TDM ops <= `num_outstanding`.

### L2 Prefetch

```python
ttgl.amd.gfx1250.tdm.prefetch(
    src=desc,              # tensor descriptor
    offsets=[off_m, off_n],
    pred=True,             # predicate
    speculative=False      # True = skip OOB checks, use only if page is cached
)
```

## Pointer-Based Async Copy

### GFX1250

```python
# Global -> Shared (pointer-based, not descriptor-based)
ttgl.amd.gfx1250.async_copy.global_to_shared(smem, pointer, mask=None, other=None, cache_modifier="")
ttgl.amd.gfx1250.async_copy.shared_to_global(pointer, smem, mask=None, cache_modifier="")
ttgl.amd.gfx1250.async_copy.commit_group()
ttgl.amd.gfx1250.async_copy.wait_group(num_outstanding=0)
ttgl.amd.gfx1250.async_copy.mbarrier_arrive(mbarrier)  # arrive after async copies complete
```

### CDNA4

```python
# Global -> Shared via tensor pointers
ttgl.amd.cdna4.async_copy.global_load_to_shared(dest, ptr, mask=None, other=None, cache_modifier="")

# Global -> Shared via scalar base + offsets (preferred, better perf + HW OOB masking)
ttgl.amd.cdna4.async_copy.buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="")

ttgl.amd.cdna4.async_copy.commit_group()
ttgl.amd.cdna4.async_copy.wait_group(num_outstanding=0)

# Relaxed shared load (skip unnecessary waits)
ttgl.amd.cdna4.async_copy.load_shared_relaxed(smem, layout)
```

**CDNA4 async copy constraints:**
- `ptr` layout: `size_per_thread * bits_per_element` must be 128 or 32 (128 recommended)
- Writes to `dest` must be coalesced
- Swizzling only within warp boundary
- Completes in order with regular loads/stores — **avoid interleaving for performance**
- `buffer_load_to_shared` preferred over `global_load_to_shared` (lower register pressure, HW OOB masking)

## MBarrier (LDS Barriers) — GFX1250

Hardware barriers in shared memory for producer-consumer synchronization.

```python
from triton.experimental.gluon.language.amd.gfx1250.mbarrier import MBarrierLayout

# Allocate
bars = ttgl.allocate_shared_memory(ttgl.int64, [NUM_BUFFERS, 1], MBarrierLayout())

# Initialize with arrival count
ttgl.amd.gfx1250.mbarrier.init(bars.index(i), count=NUM_THREADS)

# Wait for phase to complete
ttgl.amd.gfx1250.mbarrier.wait(bars.index(i), phase)

# Signal arrival (decrements pending count)
phase = ttgl.amd.gfx1250.mbarrier.arrive(bars.index(i), count=1)
```

**Phase tracking pattern:**

```python
from triton.language.core import _aggregate as aggregate

@aggregate
class PhaseCounter:
    iteration: ttgl.tensor
    num_barriers: ttgl.constexpr

    @gluon.jit
    def phase(self):
        return (self.iteration // self.num_barriers) & 1

    @gluon.jit
    def next(self):
        return PhaseCounter(self.iteration + 1, self.num_barriers)
```

**Barrier count rules:**
- Per-thread arrive: `count = num_warps * warp_size`
- TDM arrive (per-warp): `count = num_producer_warps`

## Buffer Load/Store (CDNA3/CDNA4/GFX1250)

Scalar base pointer + tensor of offsets. Lower register pressure than tensor pointers.

```python
# Load: global -> registers
tensor = ttgl.amd.cdna3.buffer_load(ptr, offsets, mask=None, other=None, cache=None)

# Store: registers -> global
ttgl.amd.cdna3.buffer_store(value, ptr, offsets, mask=None, cache=None)
```

- `ptr`: scalar pointer (not tensor of pointers)
- `offsets`: tensor of int32/uint32 offsets
- `cache`: cache modifier string (optional)
- Also available as `ttgl.amd.gfx1250.buffer_load/store` (re-exported)

## Buffer Atomics (CDNA3/CDNA4)

```python
ttgl.amd.cdna3.buffer_atomic_max(ptr, offsets, value, mask, sem, scope)
ttgl.amd.cdna3.buffer_atomic_min(ptr, offsets, value, mask, sem, scope)
ttgl.amd.cdna3.buffer_atomic_add(ptr, offsets, value, mask, sem, scope)
ttgl.amd.cdna3.buffer_atomic_and(ptr, offsets, value, mask, sem, scope)
ttgl.amd.cdna3.buffer_atomic_or(ptr, offsets, value, mask, sem, scope)
ttgl.amd.cdna3.buffer_atomic_xor(ptr, offsets, value, mask, sem, scope)
ttgl.amd.cdna3.buffer_atomic_xchg(ptr, offsets, value, mask, sem, scope)
```

CDNA4 additionally supports `fadd` with `bf16`.

Supported types: `float16, float32, bfloat16, float64, int32, int64, uint32, uint64`

## Shared Memory Operations

```python
# Allocate shared memory
buf = ttgl.allocate_shared_memory(dtype, shape, layout)

# Multi-buffer allocation
buf = ttgl.allocate_shared_memory(dtype, [NUM_BUFFERS] + block_shape, layout)

# Index into buffer dimension
tile = buf.index(buffer_idx)

# Load: shared -> registers
tensor = tile.load(layout=distributed_layout)

# Store: registers -> shared
tile.store(tensor)

# Permute dimensions (e.g., transpose)
transposed_tile = tile.permute([1, 0])
```

## Cluster Synchronization — GFX1250

For multi-CTA kernels within a cluster:

```python
ttgl.amd.gfx1250.cluster.arrive()  # signal arrival
ttgl.amd.gfx1250.cluster.wait()    # wait for all CTAs
```

Must come in pairs. Arrive without wait or double-arrive = undefined behavior.

## Common Pipelining Patterns

### Double-Buffered TDM (GFX1250)

```python
NUM_BUFFERS = 2
a_buffer = ttgl.allocate_shared_memory(dtype, [NUM_BUFFERS, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)

# Prologue: fill first buffer
producer = 0
producer = issue_loads(producer, ...)

# Main loop
for k in range(num_k_tiles - 1):
    producer = issue_loads(producer, ...)           # load next
    ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)  # wait for current
    consumer, acc = issue_wmma(consumer, ...)        # compute current

# Epilogue
ttgl.amd.gfx1250.tdm.async_wait(0)
consumer, acc = issue_wmma(consumer, ...)
```

### Producer-Consumer with MBarrier (GFX1250 Warp Specialization)

```python
# Producer warp
for k in range(num_k_tiles):
    buf_idx = k % NUM_BUFFERS
    ttgl.amd.gfx1250.mbarrier.wait(empty_bars.index(buf_idx), phase)  # wait for consumer
    ttgl.amd.gfx1250.tdm.async_load(desc, offsets, buffer.index(buf_idx), mbarrier=ready_bars.index(buf_idx))

# Consumer warp
for k in range(num_k_tiles):
    buf_idx = k % NUM_BUFFERS
    ttgl.amd.gfx1250.mbarrier.wait(ready_bars.index(buf_idx), phase)  # wait for producer
    a = buffer.index(buf_idx).load(layout=OPERAND_LAYOUT)
    acc = ttgl.amd.gfx1250.wmma(a, b, acc)
    ttgl.amd.gfx1250.mbarrier.arrive(empty_bars.index(buf_idx))       # signal done
```

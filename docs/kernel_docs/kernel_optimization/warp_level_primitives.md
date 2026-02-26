# Warp-Level (Wavefront-Level) Primitives
[← Kernel Optimization](kernel_optimization.md)

---
## Overview

On AMD GPUs, **warp-level primitives** (known as **wavefront-level operations**) allow threads within a single wavefront to communicate and exchange data efficiently **without using shared memory**.  
In AMD terminology:
- **Wavefront** ≈ **Warp** on NVIDIA.
- **Wavefront size** = typically **64 threads**.

These primitives are vital for efficient parallel reductions, scans, and data sharing across threads.

---

## Key Concepts

### 🧠 What Is a Wavefront?
A **wavefront** is a group of threads that execute the same instruction simultaneously on an AMD GPU compute unit.  
Each wavefront typically consists of 64 threads (lanes), but the hardware executes them in a SIMD fashion.

### 🌀 Why Use Wavefront-Level Operations?
- Avoids latency of shared/global memory.
- Reduces synchronization overhead.
- Enables intra-wavefront data sharing.
- Improves performance for algorithms like reductions, prefix sums, and sorting.

---

## Available Wavefront-Level Primitives

### 1. `__builtin_amdgcn_readlane()`
Reads a value from a specific lane (thread) in the same wavefront.

```cpp
int readlane(int value, int lane_id) {
    return __builtin_amdgcn_readlane(value, lane_id);
}
```

### 2. `__builtin_amdgcn_writelane()`
Writes a value to a specific lane.

```cpp
int writelane(int value, int lane_id) {
    return __builtin_amdgcn_writelane(value, lane_id);
}
```

### 3. `__builtin_amdgcn_ds_bpermute()`
Performs a **broadcast or shuffle** between lanes.

```cpp
int shuffle(int value, int src_lane) {
    return __builtin_amdgcn_ds_bpermute(src_lane * 4, value);
}
```

### 4. `__builtin_amdgcn_ds_permute()`
Permutes values based on the source lane index, used for complex data exchange patterns.

```cpp
int permute(int value, int index) {
    return __builtin_amdgcn_ds_permute(index * 4, value);
}
```

---

## Example: Warp-Level Reduction

Here’s a simple reduction (sum across wavefront lanes):

```cpp
__device__ float wavefront_reduce_sum(float value) {
    for (int offset = 32; offset > 0; offset /= 2) {
        float other = __builtin_amdgcn_ds_bpermute(
            (__lane_id() + offset) * 4, value);
        value += other;
    }
    return value;
}
```

This example sums values across a wavefront without shared memory.

---

## Comparison: AMD vs NVIDIA

| Concept | AMD Term | NVIDIA Term | Typical Size |
|----------|-----------|--------------|---------------|
| Execution Group | Wavefront | Warp | 64 (AMD) / 32 (NVIDIA) |
| Shuffle Operation | DS_BPERMUTE | __shfl_sync | Intra-warp data exchange |
| Lane ID | `__lane_id()` | `threadIdx.x % warpSize` | - |

---

## Performance Notes

- These operations **bypass shared memory**, reducing latency.  
- Access pattern and bank conflicts can affect performance.  
- Wavefront-level operations are **synchronous** by nature — no barrier is needed.

---

[← Kernel Optimization](kernel_optimization.md)
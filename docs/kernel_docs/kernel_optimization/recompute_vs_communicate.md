# 🔁 Choosing to Recompute vs Communicate Intermediate Results  
[← Kernel Optimization](kernel_optimization.md)

---

### Optimizing HIP Kernels for AMD CDNA/RDNA Architectures

Efficient GPU programming often involves a fundamental trade-off:  
**Should you recompute a value locally, or communicate (load/store/share) it between threads, wavefronts, or kernels?**

On AMD GPUs, where **memory bandwidth** and **synchronization latency** can be limiting, making the right choice has a large performance impact.

This guide helps you reason about when to **recompute**, when to **share or communicate**, and how to structure HIP kernels for maximum throughput.

---

## ⚙️ 1. Background: Why This Trade-Off Exists

AMD GPU hardware (CDNA, RDNA) offers:
- Very high compute throughput (hundreds of TFLOPs).
- Comparatively lower memory bandwidth per compute unit.
- Shared memory (LDS) that is fast but limited (up to 128 KB per CU).
- Global memory with high latency (hundreds of cycles).

Because ALU (compute) operations are much cheaper than memory traffic,  
you can often **gain performance by recomputing values** rather than storing and reloading them.

---

## 🧮 2. Cost Model: Recompute vs Communicate

| Operation Type | Approx. Cost (cycles) | Description |
|----------------|----------------------|--------------|
| **FMA / Add / Mul** | 1–4 | Executed in vector ALUs; very cheap. |
| **LDS load/store** | ~10–20 | Fast shared memory, but limited bandwidth per wave. |
| **Global memory load/store** | 200–600 | High latency; depends on cache hit/miss. |
| **Synchronization (`__syncthreads`)** | 80–200 | Halts waves until all reach barrier. |

So:
- If recomputation is **<20 ALU ops**, it’s usually cheaper than writing + reading from global memory.  
- If recomputation is **deterministic and independent**, prefer it over communicating results.  
- If recomputation requires **expensive transcendental math** (`exp`, `log`, `sin`), reuse cached/shared results.

---

## 🧭 3. Guiding Principles

### ✅ Prefer Recompute When:
- The value is simple to derive (e.g., linear expressions or index math).
- The same computation is needed by multiple threads, but storing would require synchronization.
- Memory reuse is limited or unpredictable.

### ✅ Prefer Communication When:
- The intermediate is expensive to compute (nonlinear, transcendental).
- The value is reused many times by nearby threads or within a tile.
- Computation depends on random access or shared lookup tables.

---

## 🧩 4. HIP Example — Recompute vs Communicate

### ❌ Communication-based version (more memory traffic)
```cpp
__global__ void use_shared(float* input, float* output, int N) {
    __shared__ float tile[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        tile[threadIdx.x] = sqrtf(input[tid]);  // expensive op
        __syncthreads();
        output[tid] = tile[threadIdx.x] * 0.5f;
    }
}
```

This version **stores to LDS**, performs a sync, and reloads the same value.  
If the intermediate value isn’t reused across threads, the shared memory use is wasteful.

---

### ✅ Recompute-based version (lower latency)
```cpp
__global__ void recompute(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float val = sqrtf(input[tid]);     // recompute directly
        output[tid] = val * 0.5f;
    }
}
```

If the intermediate (`sqrtf(input[tid])`) isn’t shared, this version avoids LDS and sync overhead, reducing latency and improving occupancy.

---

## 🔄 5. Example — Reusing Expensive Results in Shared Memory

When recomputation involves **heavy math**, reuse is worth the LDS cost:

```cpp
__global__ void reuse_heavy(float* input, float* output, int N) {
    __shared__ float cached[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lid = threadIdx.x;

    if (tid < N) {
        float x = input[tid];
        if (lid < 256) cached[lid] = expf(x);  // expensive op
        __syncthreads();

        float val = cached[lid] / (1.0f + cached[lid]);
        output[tid] = val;
    }
}
```

Here, caching is beneficial because `expf()` is **tens of cycles**, far more expensive than LDS access.

---

## 🧠 6. Optimization Techniques

| Technique | When to Use | Effect |
|------------|--------------|--------|
| **Recompute simple expressions** | Indexing, affine transforms | Reduces shared/global memory usage |
| **LDS cache for transcendental ops** | Expensive math reused | Reduces ALU cost, amortizes sync |
| **Register caching** | Thread-local reuse | Fastest method; avoids sync |
| **Avoid redundant barriers** | When recomputing instead of sharing | Improves wavefront parallelism |
| **Profile both versions** | Always measure | AMD GPUs can trade ALU for bandwidth efficiently |

---

## 🧩 7. Summary

| Decision Factor | Recommendation |
|-----------------|----------------|
| Simple arithmetic, one-time use | **Recompute** |
| Complex math reused across threads | **Communicate (LDS)** |
| Memory pressure high, compute light | **Recompute** |
| Occupancy limited by LDS size | **Recompute** |
| Reuse pattern predictable and local | **Communicate** |

---

### 🧭 Key Takeaway

> **On AMD GPUs, recomputing simple intermediates is usually cheaper than communicating them.**  
> Use LDS or registers for expensive results that are reused many times,  
> but always profile both strategies — modern CDNA GPUs are built to trade computation for bandwidth efficiently.


---
[← Kernel Optimization](kernel_optimization.md)
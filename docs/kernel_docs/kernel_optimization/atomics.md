# ⚙️ AMD GPU Performance Guide: Avoiding and Using Atomic Operations in HIP

[← Kernel Optimization](kernel_optimization.md)

---

Atomic operations are essential tools for parallel programming, but they can also become a major performance bottleneck on AMD GPUs when misused.  
This guide explains **when to avoid**, **how to minimize**, and **how to use atomic operations efficiently** in HIP on AMD architectures (CDNA, RDNA).

---

## 🧠 What Are Atomics?

An *atomic operation* ensures that a read-modify-write sequence (like incrementing a counter) is executed without interference from other threads.  

Common examples:
```cpp
atomicAdd(&counter, 1);
atomicCAS(&flag, 0, 1);
atomicMin(&minValue, val);
```

These are *synchronized at the memory subsystem level* — only one thread at a time can update a given memory location safely.

---

## ⚡ Why Atomics Can Hurt Performance

On AMD GPUs, atomics are serialized when multiple threads in the same **wavefront (64 threads)** or **multiple waves** try to update the same memory address.  
This creates **contention** and **stalling** in the memory pipeline.

### Key Cost Factors
- **Contention:** More threads updating the same location ⇒ more serialization.
- **Scope:** Global memory atomics are slower than LDS (shared memory) atomics.
- **Data type & precision:** 64-bit atomics are slower than 32-bit.
- **Location:** Atomics on high-latency global memory are much slower than on low-latency LDS.

---

## 🧩 Strategies to Avoid Atomics

### 1. **Per-Thread Accumulation**
Each thread accumulates partial results locally (in registers), then combines them later in a reduction.

```cpp
__global__ void reduce_kernel(const float* input, float* result, int N) {
    __shared__ float block_sum[64];

    int tid = threadIdx.x;
    float sum = 0.0f;

    // Local accumulation
    for (int i = tid; i < N; i += blockDim.x)
        sum += input[i];

    block_sum[tid] = sum;
    __syncthreads();

    // In-block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            block_sum[tid] += block_sum[tid + stride];
        __syncthreads();
    }

    // Only one atomic per block
    if (tid == 0)
        atomicAdd(result, block_sum[0]);
}
```

✅ **Good pattern:** local reduction first, then one atomic per block (much fewer global atomics).

---

### 2. **Per-Wave Accumulation**
You can further reduce atomics by aggregating results **per wavefront** (64 threads) and using **wave-level intrinsics**.

HIP provides wave intrinsics (since ROCm 5.x+):
```cpp
float warp_sum = __builtin_amdgcn_readfirstlane(sum);
if (__lane_id() == 0)
    atomicAdd(result, warp_sum);
```

This replaces 64 atomics with just 1 per wave.

---

### 3. **Use Shared Memory Atomics**
LDS atomics are much faster than global memory atomics because they occur in low-latency shared memory and avoid DRAM traffic.

Example:
```cpp
__global__ void lds_atomic_example(int* data) {
    __shared__ int s_data[256];
    int tid = threadIdx.x;

    atomicAdd(&s_data[tid % 32], 1); // fast local atomic
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < 32; ++i)
            atomicAdd(&data[0], s_data[i]); // fewer global atomics
    }
}
```

---

### 4. **Privatization**
Each block or thread has its own private counter, updated independently, and merged at the end:
```cpp
__global__ void private_counter(int* result) {
    __shared__ int local_count[256];
    int tid = threadIdx.x;
    local_count[tid] = 0;

    // Do work
    if (tid % 2 == 0)
        local_count[tid]++;

    __syncthreads();

    // Reduce to a single atomic
    if (tid == 0) {
        int block_total = 0;
        for (int i = 0; i < blockDim.x; ++i)
            block_total += local_count[i];
        atomicAdd(result, block_total);
    }
}
```

---

## 🚀 When and How to Use Atomics Efficiently

### ✅ Use Atomics When:
- You truly need synchronization across wavefronts or blocks (e.g., global counters, locks).
- The probability of contention is **low** (each thread updates a different memory address most of the time).
- For correctness-critical operations like distributed reductions or sparse updates.

---

### 💡 Optimizing Atomic Use

| Technique | Description | Example |
|------------|-------------|----------|
| **Batch updates** | Accumulate multiple updates, then do one atomic write | Aggregate local sums before global atomic |
| **Use 32-bit atomics** | Faster than 64-bit | `atomicAdd(float)` instead of `atomicAdd(double)` |
| **Use LDS atomics first** | Aggregate locally in shared memory | Local atomic → global atomic once |
| **Minimize contention** | Spread updates across multiple cache lines | Use multiple counters, combine later |
| **Use relaxed atomics (HIP 5.6+)** | Relaxed memory ordering where possible | `atomicAdd_system(&var, val, memory_order_relaxed)` |

---

## ⚙️ Example: Optimized Histogram

```cpp
__global__ void histogram_kernel(const unsigned char* data, int N, int* global_hist) {
    __shared__ int local_hist[256];
    int tid = threadIdx.x;

    // Initialize shared histogram
    if (tid < 256) local_hist[tid] = 0;
    __syncthreads();

    // Local accumulation
    for (int i = tid; i < N; i += blockDim.x)
        atomicAdd(&local_hist[data[i]], 1);

    __syncthreads();

    // Merge shared histograms into global histogram
    if (tid < 256)
        atomicAdd(&global_hist[tid], local_hist[tid]);
}
```

✅ **Why it’s fast:**
- Most atomics are done in shared memory (LDS).
- Global atomics only occur once per bin per block.

---

## 🧠 Architectural Considerations (CDNA / RDNA)

- **Wavefront size:** 64 threads per wave; all lanes contending on one address serialize.  
- **Cache hierarchy:** L0/L1 caches can coalesce atomics when multiple lanes update separate cache lines.  
- **LDS atomics:** Implemented in hardware, very low latency (<100 ns).  
- **Global atomics:** Go through L2 → DRAM, latency ~400–800 ns depending on contention.

---

## 🧩 Summary

| Principle | Description |
|------------|--------------|
| Avoid atomics in tight loops | Batch or reduce locally first |
| Prefer LDS over global atomics | Local shared memory = faster |
| Use one atomic per block/wave | Drastically reduces contention |
| Match data precision | 32-bit faster than 64-bit |
| Use relaxed order atomics (HIP ≥ 5.6) | Skip unnecessary synchronization |

---
[← Kernel Optimization](kernel_optimization.md)
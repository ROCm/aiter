# ⚙️ How to Use `__launch_bounds__` on AMD GPUs
[← Kernel Optimization](kernel_optimization.md)

---

The `__launch_bounds__` attribute is a compiler hint that allows developers to guide kernel resource allocation, especially regarding **register usage**, **thread block size**, and **occupancy**. Proper use of `__launch_bounds__` can lead to better performance and improved control over GPU resource scheduling.

This guide explains what `__launch_bounds__` does, how it affects AMD GPU performance, and how to tune it effectively in HIP.

---

## 🧠 What `__launch_bounds__` Does

The `__launch_bounds__` attribute specifies two parameters:

```cpp
__launch_bounds__(maxThreadsPerBlock [, minBlocksPerMultiprocessor])
```

| Parameter                                   | Description                                                                                     |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **maxThreadsPerBlock**                      | Maximum number of threads per block that the compiler should assume when optimizing the kernel. |
| **minBlocksPerMultiprocessor** *(optional)* | Minimum number of concurrent blocks per compute unit (CU) that should be able to execute.       |

The compiler uses this information to limit register allocation per thread. If a kernel uses too many registers, fewer blocks can fit on a CU, reducing occupancy. By setting launch bounds, you tell the compiler to restrict register use to ensure more active blocks.

---

## 🔍 Why It Matters on AMD GPUs

AMD GPUs, like other architectures, schedule work in **wavefronts** (groups of 32 or 64 threads). Each Compute Unit (CU) can hold several wavefronts concurrently, limited by:

* Register file size
* LDS (shared memory) capacity
* Thread and wavefront count

`__launch_bounds__` helps control how these resources are balanced at compile time. It’s especially useful for performance tuning when kernels are:

* Register-heavy (complex arithmetic per thread)
* Sensitive to occupancy (e.g., memory latency hiding)
* Using varying thread block sizes across different launches

---

## 🧩 Syntax and Examples

### Example 1: Simple Bound

```cpp
__global__ __launch_bounds__(256)
void compute_kernel(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = fmaf(data[tid], 2.0f, 1.0f);
}
```

This tells the compiler that the kernel will **never** be launched with more than 256 threads per block. The compiler can optimize register allocation accordingly.

### Example 2: Controlling Occupancy

```cpp
__global__ __launch_bounds__(256, 2)
void reduce_kernel(float* out, const float* in) {
    // ... reduction code ...
}
```

Here, the compiler ensures that **at least 2 blocks** can reside on each CU by limiting the per-thread register count. This helps improve latency hiding and throughput when kernels are memory-bound.

---

## ⚙️ Practical Tuning Strategy

1. **Profile the kernel**
   Use `rocprof` or Radeon GPU Profiler (RGP) to determine occupancy, register usage, and performance.

2. **Start without bounds**
   Compile your kernel normally and check register usage (`--save-temps` or `hipcc --show-register-usage`).

3. **Add `__launch_bounds__` gradually**
   Begin with the actual block size, e.g.:

   ```cpp
   __launch_bounds__(blockSize)
   ```

4. **Tune for concurrency**
   If occupancy is too low, add the second parameter:

   ```cpp
   __launch_bounds__(blockSize, minBlocksPerCU)
   ```

   Increase `minBlocksPerCU` to force the compiler to reduce registers and allow more concurrent blocks.

5. **Measure trade-offs**

   * Too low register usage → higher occupancy but possibly more spills to memory.
   * Too high register usage → fewer resident waves, increased latency.

---

## 🧮 Relationship Between Registers, Occupancy, and Launch Bounds

| Parameter                | Effect                                                                      |
| ------------------------ | --------------------------------------------------------------------------- |
| **Registers per thread** | Inversely proportional to the number of active waves per CU.                |
| **`maxThreadsPerBlock`** | Sets compiler expectations for launch configuration.                        |
| **`minBlocksPerCU`**     | Forces compiler to reserve fewer registers per thread to allow concurrency. |
| **Occupancy**            | Determined by available registers, LDS, and hardware limits.                |

Example occupancy estimation:

```
MaxThreadsPerCU = floor(RegisterFileSize / (RegistersPerThread * ThreadsPerBlock))
```

By lowering `RegistersPerThread` through launch bounds, you can increase `MaxThreadsPerCU`.

---

## 🧠 Example: Tuning Register Pressure

```cpp
// Original kernel (high register use)
__global__ void heavy_compute(float* out, const float* in) {
    float temp[64]; // large per-thread array
    // ... compute ...
}

// Tuned version with launch bounds
__global__ __launch_bounds__(128, 4)
void heavy_compute_optimized(float* out, const float* in) {
    float temp[32]; // reduced local footprint
    // ... compute ...
}
```

Here, `__launch_bounds__(128, 4)` encourages the compiler to lower register usage so that at least 4 blocks can fit on a CU, increasing concurrency and potentially improving performance.

---

## 🧾 Best Practices

* Match the **first argument** to your actual block size (`blockDim.x * blockDim.y * blockDim.z`).
* Use the **second argument** only when occupancy is too low.
* Always **verify with profiling tools** — higher occupancy doesn’t always equal higher speed.
* Avoid overly aggressive bounds that cause excessive register spilling.
* Combine `__launch_bounds__` tuning with LDS and tile size adjustments for balanced optimization.

---

## ⚠️ Common Pitfalls

* **Mismatched launch size:** If you launch a kernel with more threads than specified in `maxThreadsPerBlock`, behavior is undefined.
* **Overly tight register limits:** May lead to performance drops due to register spilling into slower memory.
* **Ignoring architecture differences:** Optimal bounds differ between RDNA and GCN GPUs.
* **Over-tuning:** Avoid setting bounds without profiling data — it may reduce performance instead of improving it.

---

## 🧰 Tools for Launch Bounds Tuning

| Tool                          | Use                                               |
| ----------------------------- | ------------------------------------------------- |
| **`hipcc --save-temps`**      | Inspect compiler-generated register counts        |
| **rocprof**                   | Gather occupancy and register usage statistics    |

---

## 🧩 Summary

`__launch_bounds__` is a powerful optimization directive that gives developers control over GPU resource allocation. When tuned correctly, it helps achieve a balance between **register pressure** and **occupancy**, leading to more efficient GPU utilization.

Use it thoughtfully — guided by profiling data — to get the most out of AMD GPU hardware.

---
[← Kernel Optimization](kernel_optimization.md)
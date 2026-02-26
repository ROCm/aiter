# 🔁 Loop Unrolling and Instruction-Level Parallelism on AMD GPUs

[← Kernel Optimization](kernel_optimization.md)

---

This guide explains how to use **loop unrolling** to improve instruction-level parallelism (ILP) and throughput on AMD GPUs. Loop unrolling is a fundamental optimization that helps reduce control overhead and increase the number of independent operations that can be executed simultaneously by the GPU’s pipelines.

---

## 🎯 Why Loop Unrolling Matters

Modern AMD GPUs rely on **massive instruction-level parallelism (ILP)** and **latency hiding** to achieve high performance. Each compute unit (CU) can issue multiple instructions per cycle, provided there are enough independent instructions ready to execute.

Unrolling a loop exposes more independent instructions, giving the compiler and scheduler greater flexibility to:

* Overlap arithmetic and memory operations.
* Reduce branch and loop overhead.
* Improve instruction pipeline utilization.
* Maximize throughput on wide SIMD execution units.

---

## ⚙️ What Loop Unrolling Does

Loop unrolling replaces a repetitive loop structure with multiple copies of the loop body, reducing the number of branch iterations.

Example (before unrolling):

```cpp
for (int i = 0; i < 8; ++i)
    out[i] = a[i] * b[i] + c[i];
```

Example (manually unrolled by 4):

```cpp
for (int i = 0; i < 8; i += 4) {
    out[i]   = a[i]   * b[i]   + c[i];
    out[i+1] = a[i+1] * b[i+1] + c[i+1];
    out[i+2] = a[i+2] * b[i+2] + c[i+2];
    out[i+3] = a[i+3] * b[i+3] + c[i+3];
}
```

The compiler may also unroll loops automatically if optimization flags are enabled (e.g., `-O3`). Manual unrolling is useful when you know the iteration count and data dependencies.

---

## 🧠 Benefits on AMD Architectures

* **Increased ILP:** Each wavefront can issue multiple memory and arithmetic instructions in parallel.
* **Reduced Branch Overhead:** Fewer conditional checks and jumps.
* **Improved Memory Coalescing:** Multiple sequential loads may combine into vectorized instructions.
* **Better Register Reuse:** Allows the compiler to hold more intermediate values in registers.

However, excessive unrolling can:

* Increase register pressure (reducing occupancy).
* Increase instruction cache usage.

So tuning is essential.

---

## 🧩 Types of Loop Unrolling

### 1. Compiler-Directed Unrolling

Use `#pragma unroll` to control compiler behavior.

```cpp
#pragma unroll 4
for (int i = 0; i < N; ++i)
    sum += a[i] * b[i];
```

* **No argument:** The compiler decides an optimal unroll factor.
* **Explicit factor:** You force a specific unroll count.

### 2. Manual Unrolling

When iteration counts are small or known at compile time, manually unrolling gives you full control:

```cpp
for (int i = 0; i < 16; i += 4) {
    sum0 += a[i]   * b[i];
    sum1 += a[i+1] * b[i+1];
    sum2 += a[i+2] * b[i+2];
    sum3 += a[i+3] * b[i+3];
}
float sum = sum0 + sum1 + sum2 + sum3;
```

This approach also promotes ILP by maintaining multiple independent accumulators.

---

## 🔬 Leveraging Instruction-Level Parallelism (ILP)

On AMD GPUs, latency hiding occurs through **wavefront-level parallelism (WLP)** and **instruction-level parallelism (ILP)**. Loop unrolling boosts ILP by exposing independent instructions that can execute concurrently within the same wavefront.

Example: Fused multiply-add pipeline utilization

```cpp
float acc0 = 0, acc1 = 0;
for (int i = 0; i < N; i += 2) {
    acc0 += A[i]   * B[i];   // Independent instruction stream 1
    acc1 += A[i+1] * B[i+1]; // Independent instruction stream 2
}
```

Both accumulations can be executed in parallel by the compiler and scheduler.

**Tip:**
Keep multiple independent accumulators to expose ILP and help the compiler overlap instructions.

---

## ⚡ When to Use Loop Unrolling

✅ Use loop unrolling when:

* The loop has **small fixed iteration counts**.
* The loop body performs **independent arithmetic**.
* You are **memory-latency limited** and want to increase ILP.
* The kernel is **compute-bound** and needs better ALU utilization.

❌ Avoid or limit unrolling when:

* Loop count is large or data-dependent.
* The kernel already uses a large number of registers (check with compiler reports or profiling tools).
* Unrolling causes instruction cache pressure or code bloat.

---

## 📏 Choosing Unroll Factors

The optimal unroll factor depends on:

* **Register pressure:** Each unrolled iteration may add more live variables.
* **CU occupancy:** Too many registers per thread can reduce active wavefronts.
* **Instruction mix:** Arithmetic-heavy loops benefit more than memory-bound ones.

**Typical tuning approach:**

1. Start with `#pragma unroll 2` or `4`.
3. Check for register spills or occupancy drops.
4. Adjust unroll factor until performance peaks.

---

## 🧰 Example: Tuning Reduction Kernel

```cpp
__global__ void reduce_kernel(const float* in, float* out, int N) {
  float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < N; i += blockDim.x * gridDim.x * 4) {
    sum0 += in[i];
    sum1 += in[i + blockDim.x];
    sum2 += in[i + blockDim.x * 2];
    sum3 += in[i + blockDim.x * 3];
  }
  float sum = sum0 + sum1 + sum2 + sum3;
  out[gid] = sum;
}
```

This loop unrolling increases ILP by maintaining multiple accumulators, allowing the compiler to schedule memory and compute operations concurrently.

---

## 📊 Profiling and Validation

| Metric             | Tool                     | Goal                                                  |
| ------------------ | ------------------------ | ----------------------------------------------------- |
| **ILP efficiency** | RGP instruction timeline | Confirm overlapping ALU ops                           |
| **Register count** | `hipcc --save-temps`     | Ensure register usage stays below occupancy threshold |
| **Throughput**     | rocprof                  | Measure achieved FLOPs and bandwidth                  |

---

## ⚠️ Common Pitfalls

* **Excessive unrolling:** May cause register spilling to memory, harming performance.
* **Code size explosion:** Instruction cache misses may offset ILP gains.
* **Dependent computations:** ILP benefits vanish if each iteration depends on the previous.

---

## 🧾 Summary

Loop unrolling enhances ILP and helps AMD GPUs overlap arithmetic and memory operations. Apply it selectively, guided by profiling, to maximize compute utilization while managing register pressure.

**Key takeaways:**

* Use small unroll factors for regular loops.
* Combine unrolling with multiple accumulators to increase ILP.
* Validate gains through profiling tools — unrolling is architecture- and kernel-specific.

---
[← Kernel Optimization](kernel_optimization.md)


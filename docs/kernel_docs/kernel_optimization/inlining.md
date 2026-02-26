# ⚙️ Function Inlining and Its Impact on Compiler Optimization

[← Kernel Optimization](kernel_optimization.md)

---

Inlining is a key compiler optimization technique that replaces a function call with the actual function body. On AMD GPUs (and modern compilers in general), inlining can improve performance by eliminating call overhead and exposing more opportunities for compile-time optimizations such as **instruction-level parallelism (ILP)**, **register allocation**, and **loop unrolling**.

This section explains how function inlining works, when it helps, and how to use it effectively in HIP or C++ GPU kernels.

---

## 🎯 Why Function Inlining Matters

Function calls have an inherent cost — the compiler must:

1. Push arguments onto a call stack (or registers).
2. Jump to a new instruction address.
3. Return control to the caller after completion.

On GPUs, this cost can disrupt instruction pipelines and limit optimization opportunities. By inlining, the compiler eliminates these call/return sequences and merges the callee code into the caller. This allows:

* **Fewer control flow instructions** (no `call`/`return`).
* **Better ILP:** More independent instructions visible to the scheduler.
* **Improved constant propagation and dead code elimination.**
* **Easier vectorization and unrolling:** The compiler can treat formerly separate functions as one continuous block of code.

---

## 🧩 How Inlining Works

When a function is marked for inlining, the compiler copies its code directly into each call site. For example:

### Before inlining:

```cpp
__device__ float square(float x) {
    return x * x;
}

__global__ void kernel(float* data) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    data[gid] = square(data[gid]);
}
```

### After inlining (conceptually):

```cpp
__global__ void kernel(float* data) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = data[gid];
    data[gid] = x * x; // inlined version
}
```

The function `square()` no longer exists as a separate call — the multiply operation is inserted directly.

---

## 🧠 Benefits of Inlining on AMD GPUs

### 1. **Eliminates Function Call Overhead**

GPU hardware does not use a traditional call stack like CPUs. Function calls may still involve register and control flow setup, so inlining avoids this cost.

### 2. **Improves Compiler Optimization Scope**

Inlining gives the compiler visibility into how arguments are used, enabling optimizations such as:

* Constant folding (`square(2.0f)` → `4.0f`).
* Dead code elimination for unused paths.
* Fused multiply-add (FMA) transformations.

### 3. **Enhances ILP and Unrolling Opportunities**

Inlining increases the size of basic blocks, exposing more parallel operations for scheduling. The compiler can reorder, combine, or interleave arithmetic and memory instructions more effectively.

### 4. **Reduces Divergent Control Flow**

Inlining may simplify nested control logic, allowing wavefronts to stay more coherent if conditional branches can be optimized away.

---

## 🧾 When to Use Inlining

✅ **Use inlining for:**

* Small, frequently called helper functions.
* Simple mathematical or vector operations.
* Functions with constant or known arguments.
* Performance-critical loops where ILP matters.

❌ **Avoid inlining when:**

* The function is large and called from many places — can lead to code size explosion.
* The kernel is already register-limited (more code can increase register usage).
* The compiler already inlines automatically (check disassembly before forcing).

---

## ⚙️ How to Enable or Control Inlining

### 1. Compiler Hints

* Use `__forceinline__` to tell the compiler to always inline a function.
* Use `__noinline__` to prevent inlining when you want to isolate functions (e.g., debugging or reducing register pressure).

Example:

```cpp
__device__ __forceinline__ float fast_add(float a, float b) {
    return a + b;
}

__device__ __noinline__ float slow_add(float a, float b) {
    return a + b;
}
```

### 2. Compiler Flags

* `hipcc -O3` enables aggressive inlining by default.
* `hipcc -mllvm -inline-threshold=<N>` adjusts the heuristic threshold for inlining decisions.

---

## 🧮 Example: Inlining to Improve ILP

Before inlining:

```cpp
__device__ float transform(float x) { return sinf(x) * x + 1.0f; }
__device__ float process(float x)   { return transform(x) * 0.5f; }

__global__ void kernel(float* data) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i] = process(data[i]);
}
```

After inlining (`transform()` into `process()`):

```cpp
__global__ void kernel(float* data) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = data[i];
  float t = sinf(x) * x + 1.0f;
  data[i] = t * 0.5f;
}
```

Here, the compiler can overlap the multiply-add and sine pipeline stages, increasing ILP and reducing latency.

---

## 🔬 Profiling the Effects of Inlining

| Metric                   | Tool                            | Benefit                                                        |
| ------------------------ | ------------------------------- | -------------------------------------------------------------- |
| **Register count**       | `hipcc --save-temps`            | Verify inlining doesn’t increase register pressure excessively |

---

## ⚠️ Common Pitfalls

* **Code bloat:** Too much inlining can increase binary size, stressing instruction cache.
* **Increased register usage:** Each inlined function adds temporaries and can lower occupancy.
* **Debugging difficulty:** Debug info can become less clear after inlining.

---

## 🧰 Best Practices

* Inline small, hot functions (math helpers, index calculations).
* Avoid inlining large, infrequently used routines.
* Combine inlining with `#pragma unroll` and vectorization for compute loops.
* Use compiler flags and reports to monitor register pressure.

---

## 🧾 Summary

Inlining improves performance on AMD GPUs by removing function call overhead, exposing ILP, and enabling better compiler optimizations. Use it strategically — small helper functions benefit the most, while large kernels should balance inlining with code size and occupancy constraints.

---

[← Kernel Optimization](kernel_optimization.md)
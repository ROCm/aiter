# 🧠 How To: Using Intrinsics


[← Kernel Optimization](kernel_optimization.md)

---

Intrinsics are low-level built-in functions that allow developers to directly access hardware instructions and special GPU operations not normally exposed in high-level languages like C++ or Python. On AMD GPUs, using intrinsics can significantly optimize compute kernels, improve performance, and reduce instruction overhead — especially in performance-critical code.

This guide explains what intrinsics are, how to use them safely, and when they make sense in AMD GPU programming.

---

## 🧩 What Are Intrinsics?

Intrinsics are **compiler-provided functions** that map directly to GPU instructions. They allow developers to:

- Access **wavefront-level operations** such as shuffles, ballots, or reductions.
- Query **hardware properties** like thread ID, lane ID, or SIMD width.
- Perform **optimized math operations** (e.g., fast fused multiply-add, bit manipulations).
- Control **memory access** or synchronization primitives with high efficiency.
- No high level API for specific hardware features is provided.

In HIP (Heterogeneous-Compute Interface for Portability), AMD provides a rich set of intrinsics under the `__builtin_amdgcn_*` and `__hip_*` namespaces.

The list of all intrinsics might be found [here](https://github.com/llvm/llvm-project/blob/21f3875ffde0ec12bec9e719f5c3fad7adb41667/clang/include/clang/Basic/BuiltinsAMDGPU.td#L30).

Documentation for some of the intrinsics might be found [here](https://llvm.org/docs/AMDGPUUsage.html#llvm-ir-intrinsics).

---

## 🎯 Why Use Intrinsics?

Intrinsics are useful when:

- You need to **optimize hot loops or core math routines**.
- You’re writing **custom kernels** where every cycle counts.
- You want **fine control** over thread communication or memory behavior.
- The compiler cannot automatically produce optimal instructions.

Intrinsics are especially relevant in:
- **Matrix multiplication (GEMM)** kernels.
- **Reduction and scan** operations.
- **Warp (wavefront) shuffles and ballots**.
- **Tensor operations** and **sparse computations**.

---

## ℹ️ Common AMD GPU Intrinsics

Below are some commonly used AMD GPU intrinsics available in HIP and ROCm:

| Intrinsic | Description | Example |
|------------|-------------|----------|
| `__builtin_amdgcn_readfirstlane(x)` | Returns the value of `x` from the first lane in the wavefront. | `int first = __builtin_amdgcn_readfirstlane(val);` |
| `__builtin_amdgcn_readlane(x, lane)` | Reads the value of `x` from a specific lane. | `float y = __builtin_amdgcn_readlane(x, 3);` |
| `__builtin_amdgcn_writelane(x, y, lane)` | Writes value `y` to `lane` in `x`. | `__builtin_amdgcn_writelane(vec, val, 1);` |
| `__builtin_amdgcn_wave_barrier()` | Synchronizes all threads in a wavefront. | `__builtin_amdgcn_wave_barrier();` |
| `__builtin_amdgcn_s_barrier()` | Synchronizes all threads in a workgroup. | `__builtin_amdgcn_s_barrier();` |
| `__builtin_amdgcn_ds_bpermute(idx, val)` | Permutes data between threads efficiently. | `val2 = __builtin_amdgcn_ds_bpermute(idx, val);` |
| `__builtin_amdgcn_mbcnt_lo(mask, prev)` | Counts active lanes in lower bits of mask. | `int count = __builtin_amdgcn_mbcnt_lo(mask, 0);` |

---

## 🛠️ Example: Using Intrinsics in a HIP Kernel

Here’s a simple HIP kernel that uses AMD intrinsics to perform a warp-level sum reduction:

```cpp
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ float warp_reduce_sum(float val) {
    // Wavefront shuffle reduction using AMD intrinsics
    for (int offset = 32; offset > 0; offset /= 2) {
        float v = __builtin_amdgcn_ds_bpermute(__lane_id() + offset, val);
        val += v;
    }
    return val;
}

__global__ void reduce_kernel(float *data, float *out, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        sum = data[idx];

    sum = warp_reduce_sum(sum);

    if ((threadIdx.x & 63) == 0)
        out[blockIdx.x] = sum;
}

int main() {
    const int N = 1024;
    float *d_data, *d_out;
    float h_data[N], h_out[16];

    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    hipMalloc(&d_data, N * sizeof(float));
    hipMalloc(&d_out, 16 * sizeof(float));
    hipMemcpy(d_data, h_data, N * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(reduce_kernel, dim3(16), dim3(64), 0, 0, d_data, d_out, N);
    hipMemcpy(h_out, d_out, 16 * sizeof(float), hipMemcpyDeviceToHost);

    float total = 0;
    for (int i = 0; i < 16; ++i)
        total += h_out[i];

    printf("Total = %f\n", total);

    hipFree(d_data);
    hipFree(d_out);
    return 0;
}
```

---

## 💡 When *Not* to Use Intrinsics

While intrinsics are powerful, they come with caveats:

- **Portability:** Intrinsics are hardware-specific; avoid them for cross-platform code.
- **Maintainability:** Code readability suffers — only use in hot paths.
- **Safety:** Misusing synchronization or memory intrinsics can cause deadlocks or race conditions.
- **Future-proofing:** New GPU architectures might change or deprecate certain intrinsics.

Prefer compiler-optimized code (HIP, rocBLAS, rocPRIM) unless you truly need to hand-tune a kernel.

---

## ✅ Best Practices

✅ Use intrinsics **only in performance-critical paths**  
✅ **Profile first**, then optimize using intrinsics  
✅ Use **inline device functions** to encapsulate intrinsic logic  
✅ Comment every intrinsic use clearly  
✅ Test thoroughly on multiple AMD architectures  


---

**Summary:**  
Using intrinsics on AMD GPUs gives developers low-level control to optimize performance-critical code, particularly for wavefront synchronization, communication, and math operations. Used wisely, they can bridge the gap between compiler automation and hand-tuned GPU performance.

---

[← Kernel Optimization](kernel_optimization.md)
# Intra-Warp Communications on AMD Hardware

[← Kernel Optimization](kernel_optimization.md)

---

Intra-warp communication (also called **intra-wavefront communication** on AMD) is a technique for exchanging data between threads that belong to the **same wavefront**. AMD GPUs execute instructions in lockstep within a *wavefront* (64 threads on most RDNA and CDNA architectures), which allows developers to bypass global or shared memory for small, fast exchanges of data.

This guide expands on the basics, includes tips for performance, and provides references for deeper study.


## 🔑 Key Concepts

| Term             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Wavefront**    | A group of 64 threads that execute instructions in lockstep on AMD GPUs.     |
| **Lane ID**      | Index (0–63) of a thread within a wavefront.                                |
| **Intra-Warp**   | Communication among threads within a single wavefront.                      |
| **Shuffle Ops**  | Hardware-supported operations for reading values from another lane.          |
| **Ballot Ops**   | Generate a bitmask representing predicates across all lanes.                 |

---

## 🛠 Using HIP Shuffle and Ballot Functions

### Example 1: **Simple Shuffle Broadcast**

```cpp
#include <hip/hip_runtime.h>

__global__ void simple_shuffle(int *output) {
    int lane = hipThreadIdx_x & 63; // Lane ID
    int value = lane * 2;

    // Broadcast value from lane 0 to all lanes
    int broadcasted = __shfl(value, 0);

    output[hipThreadIdx_x] = broadcasted;
}

int main() {
    const int N = 64;
    int *d_out, h_out[N];
    hipMalloc(&d_out, N * sizeof(int));

    hipLaunchKernelGGL(simple_shuffle, dim3(1), dim3(N), 0, 0, d_out);
    hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        printf("Thread %d received %d\n", i, h_out[i]);

    hipFree(d_out);
    return 0;
}
```

**Explanation:**  
- Each lane computes a unique value.  
- `__shfl(value, 0)` broadcasts the value of lane 0 to all lanes.  
- No shared memory or synchronization primitives are needed within the wavefront.

---

### Example 2: **Warp Reduction (Sum)**

```cpp
#include <hip/hip_runtime.h>

__device__ int warp_reduce_sum(int val) {
    // Reduce within a single wavefront
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void warp_sum(int *input, int *output) {
    int lane = hipThreadIdx_x & 63;
    int val = input[hipThreadIdx_x];
    int sum = warp_reduce_sum(val);

    if (lane == 0) {
        output[hipBlockIdx_x] = sum;
    }
}
```

**How it works:**  
- Values are progressively shifted downward using `__shfl_down`.  
- Lanes add values received from neighbors until only lane 0 has the full sum.  
- Lane 0 writes the result without requiring shared memory or barriers.

---

### Example 3: **Ballot Masking**

```cpp
#include <hip/hip_runtime.h>

__global__ void ballot_example(int *input, unsigned int *mask_out) {
    int lane = hipThreadIdx_x & 63;
    int predicate = (input[hipThreadIdx_x] > 0);

    unsigned int mask = __ballot(predicate);

    if (lane == 0) {
        mask_out[hipBlockIdx_x] = mask; // Stores mask for the entire wavefront
    }
}
```

**Use case:** Ballot masks are useful for voting-based algorithms or finding active threads.

---

## 🧰 Advanced Details

- **Wavefront Size Differences**: AMD wavefronts are **64 lanes** by default, while NVIDIA warps are 32. If you’re porting CUDA code to HIP, check your assumptions.
- **Latency and Throughput**: Intra-wavefront shuffles and ballots have much lower latency than shared or global memory transfers.
- **Portability**: Use HIP’s `__shfl` and `__ballot` for portable code between NVIDIA and AMD.

---

## ⚡ Performance Tips

1. **Minimize Divergence:** Branch divergence within a wavefront reduces SIMD efficiency. Align control flow where possible.
2. **Avoid Shared Memory for Small Exchanges:** Shuffle operations are faster and avoid bank conflicts.
3. **Use Warp-Level Primitives for Reductions:** Warp-level intrinsics outperform block-level reductions when only one wavefront is involved.
4. **Batch Operations:** When possible, process multiple elements per lane to amortize overhead.
5. **Check ROCm Versions:** Some intrinsics may vary across ROCm releases—verify using the latest [ROCm documentation](https://rocmdocs.amd.com).

---
[← Kernel Optimization](kernel_optimization.md)

[← Warp to warp communication](warp_warp_communication.md)
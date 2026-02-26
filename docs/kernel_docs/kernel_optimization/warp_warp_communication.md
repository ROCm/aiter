# 🔀 How To: Warp-to-Warp Communication on AMD GPUs
[← Kernel Optimization](kernel_optimization.md)

---
Warp-to-warp communication refers to **exchanging data or synchronizing between multiple wavefronts (warps)** within a thread block or even across blocks. Unlike *intra-warp* primitives (like `__shfl` or `__ballot`), AMD hardware does **not** provide direct hardware instructions for wavefront-to-wavefront communication. Instead, you must use **shared memory**, **global memory**, or **cooperative group** features to coordinate work.

---

## 🛠 Strategies for Warp-to-Warp Communication

| Method                    | Scope           | Pros                         | Cons                                  |
|---------------------------|----------------|-----------------------------|--------------------------------------|
| **Shared Memory**         | Thread block    | Fast, low latency            | Requires manual barriers and indexing|
| **Global Memory**         | Across blocks   | Accessible to all blocks     | Slower, may need multiple kernels    |
| **Cooperative Groups**    | Block or grid   | Cleaner APIs for sync        | Requires ROCm cooperative groups     |

---

## 📂 Example 1: Shared Memory Buffer (Same Block)

This example shows **warp A** writing data to shared memory and **warp B** reading it after synchronization.

```cpp
#include <hip/hip_runtime.h>

__global__ void warp_to_warp_shared(int *output) {
    __shared__ int buffer[64];
    int thread_id = hipThreadIdx_x;
    int warp_id   = thread_id / 64; // Two warps of 64 threads each
    int lane_id   = thread_id % 64;

    // Warp 0 writes values
    if (warp_id == 0) {
        buffer[lane_id] = lane_id * 2;
    }

    // Synchronize all warps in the block
    __syncthreads();

    // Warp 1 reads values written by Warp 0
    if (warp_id == 1) {
        output[lane_id] = buffer[lane_id] + 1;
    }
}

int main() {
    const int N = 64;
    int *d_out, h_out[N];
    hipMalloc(&d_out, N * sizeof(int));

    hipLaunchKernelGGL(warp_to_warp_shared, dim3(1), dim3(128), 0, 0, d_out);
    hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        printf("Thread %d read %d\n", i, h_out[i]);

    hipFree(d_out);
    return 0;
}
```

**How it works:**  
- Warp 0 writes 64 values into shared memory.  
- `__syncthreads()` ensures all threads in the block have reached the barrier before Warp 1 reads the data.  

---

## 🌐 Example 2: Global Memory for Cross-Block Communication

For **warp-to-warp communication across blocks**, shared memory cannot be used. You must write to **global memory** and, if necessary, launch a second kernel or use cooperative groups.

```cpp
#include <hip/hip_runtime.h>

__global__ void write_data(int *global_buf) {
    int warp_id = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) / 64;
    if ((hipThreadIdx_x % 64) == 0) {
        // One thread per warp writes to global memory
        global_buf[warp_id] = warp_id * 100;
    }
}

__global__ void read_data(int *global_buf, int *output) {
    int warp_id = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) / 64;
    if ((hipThreadIdx_x % 64) == 0) {
        output[warp_id] = global_buf[warp_id];
    }
}
```

**Usage Pattern:**  
1. Launch `write_data` to populate global memory.  
2. Synchronize at the kernel level (implicit after launch).  
3. Launch `read_data` to read values written by other warps.  

---

## 🤝 Example 3: Cooperative Groups (Advanced)

HIP supports **cooperative groups** for finer control:

[➔ Cooperative groups](cooperative_groups.md)

```cpp
#include <hip/hip_cooperative_groups.h>
using namespace cooperative_groups;

__global__ void cooperative_warp_comm(int *buffer) {
    thread_block block = this_thread_block();
    int warp_id = block.thread_rank() / 64;

    if (warp_id == 0) buffer[warp_id] = 42;

    // Block-level sync ensures other warps see the write
    block.sync();

    if (warp_id == 1) buffer[warp_id] = buffer[0] + 10;
}
```

---

## ⚡ Optimisation Tips

1. **Minimize Barriers:** Use as few `__syncthreads()` calls as possible. Each barrier can stall all warps in the block.  
2. **Use Shared Memory First:** For intra-block warp communication, shared memory is much faster than global memory.  
3. **Batch Communication:** Group multiple values per warp to reduce synchronization overhead.  
4. **Avoid Bank Conflicts:** When using shared memory, organize data to minimize memory bank conflicts (stride of 1 is ideal).  
5. **Consider Cooperative Groups:** They provide cleaner abstractions for warp/block synchronization.  
6. **Pipeline Work:** Use multiple stages and double-buffering to overlap computation and communication where possible.

---

## 📚 References

- [AMD ROCm HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)  
---

[← Kernel Optimization](kernel_optimization.md)
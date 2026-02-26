# 🤝 How To: Cooperative Groups on AMD GPUs (ROCm HIP)

Cooperative groups allow GPU threads to **synchronize and collaborate** at different levels: within a block, within tiles (subgroups), or across the entire grid (multi-block). On AMD GPUs using ROCm HIP, this enables more flexible patterns than `__syncthreads()` alone.

---

## 🧠 What Are Cooperative Groups?

Cooperative groups provide **flexible synchronization** and **thread grouping** primitives. Unlike `__syncthreads()`, which synchronizes only within a thread block, cooperative groups let you:

- **Partition threads** into smaller groups (e.g., warps, tiles).  
- **Synchronize across the entire grid** (all blocks in a single launch) if launched cooperatively.  
- Simplify collective operations like reductions, broadcasts, and scans.

AMD’s ROCm HIP implements cooperative groups in a way that is **API-compatible with CUDA**, making porting easier.

---

## 📦 Setup

1. **Include the Header**  
   ```cpp
   #include <hip/hip_cooperative_groups.h>
   using namespace cooperative_groups;
   ```

2. **Launch the Kernel with Cooperative Support** (for grid-wide sync):  
   - Use `hipLaunchCooperativeKernel` instead of `hipLaunchKernelGGL`.  
   - Ensure your GPU and ROCm version support cooperative launches (`hipDeviceCooperativeLaunch`).  

---

## 🔑 Core Group Types

| Group Type             | Scope                                    | Use Case                              |
|------------------------|------------------------------------------|--------------------------------------|
| `thread_block`         | Threads in a single block                 | Block-wide sync, shared memory ops   |
| `coalesced_group`      | Subset of contiguous active threads       | Warp-style sync within divergence    |
| `grid_group`           | All threads in a grid (cooperative launch)| Grid-wide barriers, multi-block ops  |
| `thread_block_tile<N>` | Subdivide block into tiles of size N      | Warp-level collectives within blocks |

---

## 🛠 Examples

### 1️⃣ Block-Level Synchronization

```cpp
#include <hip/hip_cooperative_groups.h>
using namespace cooperative_groups;

__global__ void block_sync_example(int *data) {
    thread_block block = this_thread_block();
    int tid = block.thread_rank();

    // Each thread writes to shared memory (or global memory)
    data[tid] = tid;

    // Synchronize threads within the block
    block.sync();

    // Now all threads see updated values in data[]
    data[tid] += 1;
}
```

---

### 2️⃣ Tiled Partitioning

Divide a thread block into **tiles** of a specific size (e.g., 32 for warp-like behavior):

```cpp
#include <hip/hip_cooperative_groups.h>
using namespace cooperative_groups;

__global__ void tiled_example(int *data) {
    thread_block block = this_thread_block();
    auto tile32 = tiled_partition<32>(block);

    int tid = tile32.thread_rank();
    int val = tid * 2;

    // All threads in the tile synchronize
    tile32.sync();

    // Thread 0 in each tile writes a summary value
    if (tid == 0) {
        data[block.group_index().x * 2] = val;
    }
}
```

---

### 3️⃣ Grid-Wide Synchronization

For **multi-block coordination**, launch the kernel cooperatively:

```cpp
#include <hip/hip_cooperative_groups.h>
using namespace cooperative_groups;

__global__ void grid_sync_example(int *buffer) {
    grid_group grid = this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    buffer[tid] = tid;

    // Synchronize across *all* blocks
    grid.sync();

    // After sync, all threads see completed writes from every block
    buffer[tid] += 100;
}

int main() {
    const int blocks = 2, threads = 64;
    const int total = blocks * threads;

    int *d_buf;
    hipMalloc(&d_buf, total * sizeof(int));

    void* kernelArgs[] = { &d_buf };

    hipLaunchCooperativeKernel(
        (void*)grid_sync_example, dim3(blocks), dim3(threads), kernelArgs
    );

    // Copy results back and inspect...
    hipFree(d_buf);
}
```

---

## ⚡ Optimisation Tips

1. ✅ **Check Device Capabilities**  
   Use `hipDeviceGetAttribute` with `hipDeviceAttributeCooperativeLaunch` to ensure cooperative launches are supported.  

2. ✅ **Use Appropriate Group Sizes**  
   Match tile sizes (`thread_block_tile<N>`) to **wavefront size (64)** on AMD for best performance.  

3. ✅ **Minimize Global Memory Sync**  
   Use grid-level sync only when necessary—block-level sync or tiled partitions are faster.  

4. ✅ **Leverage Tiling for Reductions**  
   Break large reductions into tiles using `tiled_partition` to improve cache and memory efficiency.  

5. ✅ **Pipeline Work**  
   Combine cooperative groups with double-buffering techniques to overlap computation and data transfer.  

---

## 📚 References

- [AMD ROCm HIP Programming Guide: Cooperative Groups](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/cooperative_groups_reference.html#cooperative-kernel-launches)  

**Summary:**  
Cooperative groups let you **organize threads into flexible synchronization domains**, from sub-warp tiles up to **grid-wide collectives**. On AMD GPUs, they’re critical for advanced parallel algorithms that need **warp-to-warp** or **block-to-block** communication without expensive kernel relaunches. By combining **tiled partitions**, **grid groups**, and **block sync**, you can write scalable and portable high-performance GPU code.

---

[← Kernel Optimization](kernel_optimization.md)

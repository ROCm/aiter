# ⚙️ Persistent Kernels

[← Kernel Optimization](kernel_optimization.md)

---

Persistent kernels are an advanced optimization strategy for AMD GPU architectures (such as **CDNA** and **RDNA**) that help reduce kernel launch overhead and improve performance for workloads with dynamic or irregular execution patterns.  

---

## 🧠 What Are Persistent Kernels?

A **persistent kernel** is a GPU kernel that remains active on the GPU for the entire duration of an application or computation phase, rather than being launched repeatedly for each batch of work.

Instead of launching a new kernel for every data chunk:
1. You **launch a fixed number of threads or wavefronts** (usually equal to or slightly less than the number of available compute units).
2. These threads **fetch work dynamically** from a global queue or buffer.
3. When a thread finishes a unit of work, it **requests more** until all tasks are complete.

---

## 🚀 Why Use Persistent Kernels?

Persistent kernels are useful when:
- **Kernel launch latency** becomes significant compared to computation time.
- Workload size is **dynamic or irregular**, making static scheduling inefficient.
- You want to **improve GPU occupancy and data locality** for small or frequently changing workloads.
- You need to **pipeline computation and communication** efficiently (e.g., for real-time inference or streaming data).

### 🔧 Benefits:
- **Reduced kernel launch overhead** — fewer launches mean less time in the driver.
- **Improved GPU utilization** — wavefronts stay busy by fetching new work.
- **Better data locality** — threads can reuse cached or shared data.
- **Dynamic load balancing** — threads handle variable workloads efficiently.

---

## ⚙️ When to Use Persistent Kernels on AMD GPUs

### ✅ Good Use Cases

| Scenario | Why It Helps |
|-----------|---------------|
| **Small, repeated kernels** | Reduces overhead of frequent kernel launches. |
| **Dynamic workloads (queues, graphs)** | Threads fetch work adaptively, keeping the GPU busy. |
| **Real-time or streaming data** | Avoids latency from frequent re-launches. |
| **Pipelined computation** | Can overlap compute and memory transfers efficiently. |

### 🚫 Avoid When

| Condition | Reason |
|------------|---------|
| **Large, uniform workloads** | Regular kernel launches are simpler and equally efficient. |
| **Memory-bound kernels** | Persistent scheduling won’t improve bandwidth limits. |
| **Simple batch workloads** | Overhead of queue logic may outweigh benefits. |

---

## 🧩 Example Pattern in HIP

Below is a simplified HIP example showing how persistent threads can fetch tasks from a global queue.

```cpp
#include <hip/hip_runtime.h>

__device__ int globalIndex = 0;

__global__ void persistentKernel(float* data, int totalTasks) {
    int id;
    while (true) {
        // Atomically fetch the next work index
        id = atomicAdd(&globalIndex, 1);
        if (id >= totalTasks)
            break;

        // Perform the computation
        data[id] = data[id] * 2.0f;
    }
}

int main() {
    const int totalTasks = 1 << 20;
    float* d_data;
    hipMalloc(&d_data, totalTasks * sizeof(float));

    // Initialize data and reset global counter
    int zero = 0;
    hipMemcpyToSymbol(HIP_SYMBOL(globalIndex), &zero, sizeof(int));

    // Launch limited number of blocks (persistent threads)
    int numBlocks = 120;   // typically ~number of CUs
    int threadsPerBlock = 256;

    hipLaunchKernelGGL(persistentKernel, dim3(numBlocks), dim3(threadsPerBlock), 0, 0, d_data, totalTasks);
    hipDeviceSynchronize();

    hipFree(d_data);
}
```

**Notes:**
- The kernel runs until all work is completed.
- The number of blocks is chosen to match or slightly underfill the GPU’s compute units.
- `atomicAdd` ensures that each thread processes unique work units.

---

## ⚙️ Hardware Considerations (AMD CDNA & RDNA)

- **Wavefront-based execution:** Persistent kernels keep wavefronts active longer, improving SIMD unit utilization.
- **Cache behavior:** Repeated work access patterns improve L1/L2 cache reuse.
- **Scheduler efficiency:** Fewer kernel launches mean fewer transitions through the ROCm driver and HSA runtime.
- **Occupancy tuning:** Keep enough blocks to saturate all Compute Units (CUs) but avoid oversubscription to maintain residency.

---

## 📈 Performance Tips

1. **Balance queue size and task granularity.**  
   Too fine-grained tasks increase atomic contention; too coarse reduces dynamic balancing.

2. **Use LDS (shared memory) wisely.**  
   Persistent threads can cache task metadata or partial results in LDS for faster reuse.

3. **Profile kernel residency.**  
   Use ROCm profiling tools to measure how long kernels stay active and identify stalls.

4. **Avoid unnecessary synchronization.**  
   Use atomics and relaxed memory ordering where possible to reduce contention.

---

## 🧭 Summary

| Advantage | Description |
|------------|-------------|
| **Low latency** | Eliminates repeated kernel launch overhead. |
| **High utilization** | Keeps GPU wavefronts busy with dynamic workloads. |
| **Data reuse** | Persistent residency improves cache efficiency. |
| **Flexible scheduling** | Threads self-manage task allocation. |

Persistent kernels are a powerful optimization tool for AMD GPU workloads that are small, dynamic, or latency-sensitive. They require careful design of task queues and synchronization, but can deliver significant performance gains in the right scenarios.

---

[← Kernel Optimization](kernel_optimization.md)

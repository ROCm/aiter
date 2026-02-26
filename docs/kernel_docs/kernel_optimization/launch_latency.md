# 🚀 HIP Kernel Launch Latency — Understanding & Optimizing

[← Kernel Optimization](kernel_optimization.md)

---

Kernel launch latency is the **time between when the host enqueues a kernel** (via `hipLaunchKernelGGL` or similar) and when the GPU actually **begins executing instructions**.  
While often small compared to large compute workloads, this latency can dominate performance for **short-running kernels** or **high-frequency kernel launches** — such as in iterative solvers, small matrix operations, or micro-benchmarks.

This guide explains what affects launch latency on AMD CDNA/RDNA GPUs and how to design HIP applications to minimize it.

---

## 🧭 1. What Is Kernel Launch Latency?

When you call:

```cpp
hipLaunchKernelGGL(MyKernel, gridDim, blockDim, sharedMemBytes, stream, args...);
```

several things happen before execution starts:

1. **Host-side work:**
   - Kernel configuration and parameter marshaling.
   - Stream dependency management and synchronization checks.
   - Command submission to the GPU command queue.

2. **Driver and runtime work:**
   - Command packet creation for the GPU scheduler.
   - Placement into command buffer (ring).
   - Context validation and page table checks (if needed).

3. **GPU-side scheduling:**
   - The command processor (CP) on the AMD GPU dequeues the kernel dispatch packet.
   - The hardware scheduler assigns it to a Compute Unit (CU).

This entire process typically takes **tens of microseconds** — often around **10–50 µs** per kernel, depending on the platform, driver, and stream behavior.

---

## ⏱️ 2. Measuring Launch Latency

You can measure kernel launch latency using **HIP events**:

```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

hipEventRecord(start, 0);
hipLaunchKernelGGL(MyKernel, dim3(1), dim3(1), 0, 0); // trivial kernel
hipEventRecord(stop, 0);
hipEventSynchronize(stop);

float ms = 0.0f;
hipEventElapsedTime(&ms, start, stop);
printf("Launch latency = %.3f µs\n", ms * 1000.0f);
```

> **Note:** For small kernels, include a `hipDeviceSynchronize()` after the launch to ensure timing includes GPU start-up overhead.

> **Note:** HIP event–based timing method provides only an approximation of kernel launch latency because the hipEventRecord() timestamps are captured on the GPU’s timeline, not the host’s. This means the measured duration includes more than just the host-to-GPU command submission delay—it also captures parts of the GPU’s internal scheduling, command queueing, and minimal kernel execution time. Since hipEventRecord() is asynchronous, the start event is only recorded once the GPU processes the command queue up to that point, not the instant the host issues the launch. Similarly, the end event marks when the kernel has finished executing, not when it started. Therefore, while using an empty kernel minimizes the execution component, the result still reflects launch + scheduling + completion overhead, rather than the pure host-side launch latency.

---

## 🧩 3. Causes of High Launch Latency

| Category | Cause | Description |
|-----------|--------|-------------|
| **Host overhead** | Runtime function calls | Parameter setup, API serialization, driver validation. |
| **Stream synchronization** | Implicit synchronization | Kernel depends on previous events in same stream. |
| **Context switching** | Multiple processes or devices | Switching between contexts adds delay. |
| **Small kernels** | Underutilization | Latency dominates runtime for short kernels. |
| **Dynamic memory ops** | `hipMalloc`, `hipMemcpy` before launch | These may flush or block the command queue. |

---

## ⚙️ 4. Optimization Techniques

### ✅ a. **Batch Work — Fuse Kernels**

Reduce launch frequency by combining small operations into larger ones:

```cpp
__global__ void fused_op(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float tmp = A[idx] * 2.0f;     // Stage 1
        B[idx] = tmp + C[idx];         // Stage 2
    }
}
```

Instead of launching two kernels for multiplication and addition, fuse them.  
Each launch avoided saves tens of microseconds and improves data locality.

---

### ✅ b. **Use Streams and Overlap**

Launch independent kernels on **different HIP streams** to overlap host submission and GPU execution:

```cpp
hipStream_t s1, s2;
hipStreamCreate(&s1);
hipStreamCreate(&s2);

hipLaunchKernelGGL(KernelA, grid, block, 0, s1, argsA...);
hipLaunchKernelGGL(KernelB, grid, block, 0, s2, argsB...);
```

- Streamed kernels may run concurrently if resources allow.
- While one kernel runs, another can be enqueued — hiding launch latency.

---

### ✅ c. **Pre-create Streams and Events**

Creating/destroying streams or events repeatedly adds overhead.  
Reuse them across iterations instead of calling `hipStreamCreate()` or `hipEventCreate()` in loops.

```cpp
hipStream_t stream;
hipStreamCreate(&stream);

for (int i = 0; i < N; i++) {
    hipLaunchKernelGGL(MyKernel, grid, block, 0, stream, args...);
}
hipStreamSynchronize(stream);
```

---

### ✅ d. **Avoid Synchronizing Between Launches**

Frequent use of `hipDeviceSynchronize()` or blocking memory copies breaks pipelining:

❌ Bad:
```cpp
for (int i = 0; i < 100; ++i) {
    hipLaunchKernelGGL(MyKernel, grid, block, 0, 0);
    hipDeviceSynchronize(); // forces host to wait
}
```

✅ Good:
```cpp
for (int i = 0; i < 100; ++i) {
    hipLaunchKernelGGL(MyKernel, grid, block, 0, 0);
}
hipDeviceSynchronize(); // wait only once
```

---

### ✅ e. **Use Persistent Kernels**

For very small, frequent tasks, it can be better to **launch once** and keep the kernel alive, processing a queue of work on the GPU.

```cpp
__global__ void persistent_worker(Task* tasks, int total) {
    for (;;) {
        int task_id = atomicAdd(&tasks->head, 1);
        if (task_id >= total) break;
        process(tasks[task_id]);
    }
}
```

Launch once, then feed it tasks via global memory.  
This eliminates repeated launch overhead entirely.

---

### ✅ f. **Warm Up the GPU**

The first kernel after context creation incurs extra overhead for:
- Driver initialization.
- GPU power gating exit.
- Memory page table setup.

Mitigate this with a **dummy warm-up kernel**:

```cpp
hipLaunchKernelGGL(WarmUpKernel, dim3(1), dim3(1), 0, 0);
hipDeviceSynchronize();
```

Subsequent launches will be faster and more stable for measurement.

---

### ✅ g. **Pinned (Page-Locked) Memory for Transfers**

If your kernel depends on frequent small memory copies, use **pinned host memory** for lower DMA latency:

```cpp
float* h_data;
hipHostMalloc(&h_data, size * sizeof(float));
```

Pinned memory enables asynchronous, lower-overhead transfers — reducing the gap between copy and compute.


### ✅ h. **Kernel Fusion**

If launching multiple small kernels, consider fusing these into a single kernel.

Look for kernels that:

- Operate on the same data buffers
- Are always invoked sequentially
- Have simple control flow between them
- Are relatively small. Combining larger kernels can lead to register pressure.


---

## 📊 5. Benchmarking Tips

- Always perform **several warm-up iterations** before measuring.
- Use **HIP events** for timing, not CPU wall clock.
- When timing a short kernel, run it **in a loop (1000×)** and divide total time to average out noise.
- Avoid profiling tools during launch latency testing — they add intercept overhead.

Example:
```cpp
for (int i = 0; i < 1000; ++i)
    hipLaunchKernelGGL(EmptyKernel, dim3(1), dim3(1), 0, 0);
hipDeviceSynchronize();
```

---

## 🧠 7. Key Takeaways

| Principle | Benefit |
|------------|----------|
| Fuse small kernels | Reduces overhead and improves locality |
| Use streams to overlap | Hides host and device latency |
| Avoid frequent syncs | Maintains high throughput |
| Reuse streams and events | Saves API overhead |
| Use persistent kernels for micro-tasks | Removes launch latency entirely |
| Warm up before measuring | Ensures stable timings |

---
[← Kernel Optimization](kernel_optimization.md)
# 🚀 How to Optimize Kernels for AMD GPUs

[← Index](../index.md)

---

Optimizing GPU kernels is key to unlocking the full performance potential of AMD hardware. This guide will walk you through practical strategies to enhance your kernel’s efficiency by reducing bottlenecks, improving memory access patterns, managing register usage, and maximizing compute unit utilization.

This guide will help you:

- Understand common sources of kernel inefficiency
- Use profiling data to pinpoint hot spots
- Optimize memory hierarchy usage (global, local, and cache)
- Manage register pressure to prevent spilling
- Balance workload distribution across compute units

With these insights and techniques, you’ll be equipped to write faster, more scalable GPU kernels tailored specifically for AMD GPUs.
  


## 📊 Profiling and Diagnosis
Before optimizing, understand where performance is lost.

- [GPU Profiling](gpu_profiling/gpu_profiling.md)
- [Instrumenting kernel code (counters, etc)](instrumenting_code.md)

---

## 🚀 Launch and Execution Configuration
Control how kernels are launched and how GPU resources are scheduled.

- [Improving Occupancy](occupancy/improving_occupancy.md)
- [Launch parameter optimization](launch_parameter_optimization.md)
- ['_\_launch_bounds__'](launch_bounds.md)
- [Kernel Launch latency](launch_latency.md)
- [Persistent kernels](persistent_kernels.md)

---

## 💾 Memory and Data Management
Efficient memory use is crucial for bandwidth-bound workloads.

- [Shared Memory tuning](shared_memory_tuning.md)
- [Register spilling and register pressure](register_spilling.md)
- [Cache optimization](cache_optimization.md)
- [Memory access patterns (4-byte alignment, stride patterns, coalescing)](memory_access_patterns.md)
- [Choosing to recompute vs communicate intermediate results](recompute_vs_communicate.md)
- [Avoiding/using atomic ops](atomics.md)

---

## 🔀 Threading and Control Flow Strategy
Organize work distribution and minimize control flow penalties.

- [Threading strategy (which threads process which data elements, which threads are shared for different output data)](threading_strategy.md)
- [Thread divergence](thread_divergence.md)
- [Loop unrolling (can utilize instruction-level parallelism)](loop_unrolling.md)
- [Inlining](inlining.md)
- [Pipelining](pipelining.md)

---

## 🤝 Warp-Level and Cooperative Parallelism
Optimize communication and synchronization at the warp/wavefront level.

- [Warp level primitives — *(wavefront-level data exchange and shuffles)*](warp_level_primitives.md)
- [Wavefront specialization — *(dedicate wavefronts for specific subtasks like loading or compute)](wavefront_sepcialization)
- [Intra warp communications](intra_warp_communication.md)
- [Warp to warp communications](warp_warp_communication.md)
- [Cooperative Groups](cooperative_groups.md)
- [Tiling](tiling.md)
- [Warp synchronization optimizations](warp_synchronization.md)

---

## 🛠️ Compiler and Intrinsic-Level Optimization
Use compiler directives and low-level intrinsics to guide performance.

- [Intrinsics](intrinsics.md)
- [Compiler hints](compiler_hints.md)
  - `volatile`
  - `restrict` (to signal pointer usage restrictions)

---

## 🧱 Low-Level and Hardware-Aware Programming
Go beyond high-level HIP — leverage AMD hardware features directly.

- **Assembly programming**
  - [WMMA/MFMA](mfma.md)
  - [Memory Buffer load to LDS](lds_assembly.md)
- [Using sparsity hardware](using_sparsity.md)
- [Composable Kernels](composable_kernels.md)

---

## 🐞 Debugging and Validation
Catch logic, memory, and synchronization bugs before measuring performance.

- [Debugging HIP kernels](debugging/debugging.md)
- [Cache Coherency Issues](debugging/cache_issues.md)

---


[← Index](../index.md)

# 📈 How to Profile GPU Performance on AMD Hardware

Profiling your GPU applications is an essential step toward achieving peak performance on AMD GPUs. This guide will walk you through the process of using AMD’s profiling tools to uncover performance bottlenecks such as register pressure, cache inefficiencies, memory bandwidth constraints, and compute unit utilization.

By learning how to measure and interpret key hardware metrics, you can identify which parts of your code limit performance and apply targeted optimizations. Whether you’re debugging kernel stalls, reducing register spills, or improving cache usage, profiling provides the insights needed to make your GPU code faster and more efficient.

In the following sections, we’ll cover the tools available, how to collect profiling data, and how to analyze it to guide your optimization efforts on AMD GPUs.

## 📊 AMD Profilers

- [ROC Profiler](rocprof/rocprof.md)
- [ROC Profiler Version 3](rocprofv3/rocprofv3.md)
    -  [Identifying Hot Spots](rocprofv3/identifying_hotspots.md)

[← Kernel Optimization](../kernel_optimization.md)
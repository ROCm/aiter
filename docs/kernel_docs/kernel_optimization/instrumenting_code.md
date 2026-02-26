# 🧭 HIP Kernel Instrumentation
[← Kernel Optimization](kernel_optimization.md)

---

This section explains practical techniques to instrument HIP kernels for performance analysis on AMD GPUs (CDNA/RDNA). It covers **host- and device-side timing**, **software counters**, **lightweight sampling**, **correlating timestamps**, and **how to use vendor profilers** effectively. Where possible, examples are given in HIP/C++ device code.

> Goal: collect meaningful performance data (latency, stalls, hotspots) while minimizing perturbation from the instrumentation itself.

---

## Overview of Instrumentation Methods

1. **Host-side timers** — Use HIP events to time kernel launches and measure end-to-end latency. Low overhead but coarse (whole-kernel granularity).
2. **Device-side cycle counters** — Use `clock64()` inside kernels to measure per-thread or per-region cycles. Higher resolution but requires care.
3. **Software counters (atomics)** — Increment global or shared counters to count events (branch hits, memory accesses) inside kernels.
4. **Sampling / conditional instrumentation** — Only instrument a subset of threads/iterations to reduce overhead.
5. **Hardware counters & vendor tools** — Use AMD tools (rocprof, roctracer/rocprofiler) to capture hardware counters; insert user markers to correlate with kernel code.
6. **Tracing / range markers** — Insert API calls (tracing libraries) to mark ranges visible to profilers. Use with RGP/rocprof to locate hotspots.

---

## 1 — Host-side Timing (hipEvent_{record,synchronize,elapsed_time})

HIP provides event objects for timing on the host. This measures wall-clock time on the GPU stream between two points (ideal for overall kernel timing).

```cpp
// Host-side timing example (C++)
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

hipEventRecord(start, stream);
hipLaunchKernelGGL(MyKernel, grid, block, 0, stream, args...);
hipEventRecord(stop, stream);
hipEventSynchronize(stop);

float ms = 0.0f;
hipEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
```

**Notes**
- `hipEventElapsedTime` returns milliseconds. This measures GPU time as seen on the stream; it excludes host-side queueing if you record before launch and after synchronization properly.
- Use this for regression testing and coarse-grained measurement. It has low overhead and is reliable for whole-kernel latency.

---

## 2 — Device-side cycle counters (clock64)

`clock64()` is a device intrinsic (available with HIP) that returns a per-thread 64-bit cycle counter. Use it to measure cycles inside kernels or between regions. Be careful: reading the clock has overhead and can affect instruction scheduling.

```cpp
__global__ void kernel_with_cycles(int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Warm up or synchronize if needed
    __syncthreads();

    unsigned long long t0 = clock64();

    // region to measure
    float sum = 0.0f;
    for (int i = 0; i < 100; ++i) {
        sum += __sinf(i * 0.001f); // work
    }

    unsigned long long t1 = clock64();
    unsigned long long cycles = t1 - t0;

    // Store one sample per-block to reduce output volume (use atomic for global store)
    if (threadIdx.x == 0) out[blockIdx.x] = cycles;
}
```

**Best practices**
- Avoid calling `clock64()` too frequently inside tight loops; prefer measuring larger regions.
- Subtracting two `clock64()` readings gives cycles elapsed on the executing SM/CU for that lane — to convert to time, divide by GPU clock frequency (from `rocminfo` or vendor API).
- To reduce measurement noise, repeat the region multiple times and average cycles.

---

## 3 — Software counters (atomicAdd) for event counting

Software counters let you count occurrences (e.g., cache misses approximated by load patterns, branch taken counts). Use atomics sparingly to avoid contention overhead.

```cpp
__global__ void kernel_event_count(const int *data, int N, unsigned int *global_counters) {
    __shared__ unsigned int s_counts[32];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;

    // init shared counters by a few threads
    if (lane < 32) s_counts[lane] = 0;
    __syncthreads();

    // Count a local event (example: data[tid] negative)
    if (tid < N) {
        if (data[tid] < 0) {
            // local increment in shared memory
            atomicAdd(&s_counts[0], 1u);
        }
    }
    __syncthreads();

    // Aggregate from shared to global once per block
    if (lane == 0) {
        atomicAdd(&global_counters[0], s_counts[0]);
    }
}
```

**Techniques to reduce overhead**
- Use **per-block shared counters** and only one global atomic per block (as shown above).
- Use **per-wave counters** combined with wave intrinsics to reduce atomics further.
- For very hot counters, consider **sampling** (see next section).

---

## 4 — Sampling and conditional instrumentation

Instrumentation overhead can perturb the kernel. Sampling reduces overhead by instrumenting only a subset of threads, waves, or iterations.

**Example: sample 1% of threads (rough sketch)**

```cpp
__global__ void sampled_kernel(...) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int seed = gid; // cheap pseudo-random
    // simple LCG for sampling decision
    seed = (1103515245u * seed + 12345u);
    if ((seed & 0xFF) < 3) { // ~1.17% sampling
        unsigned long long t0 = clock64();
        // instrumented region ...
        unsigned long long t1 = clock64();
        // write sample
    }
    // normal execution continues for all threads
}
```

**Sampling tips**
- Prefer deterministic sampling (e.g., based on thread id mod N) when reproducibility is required.
- Combine sampling with aggregation to avoid large writes (store only per-block samples).

---

## 5 — Correlating Host and Device Timers

To relate device-side cycle counts to host timeline:
1. Use `hipEventRecord` for host-side markers **before** and **after** the kernel queueing.
2. Inside the kernel, read `clock64()` around the region.
3. Use known GPU clock frequency to map cycles → milliseconds. For rough correlation:
   - `device_time_ms = cycles / (gpu_clock_khz * 1000ULL)` where `gpu_clock_khz` is in kHz.
4. For precise correlation, use vendor tracing APIs (roctracer / rocprofiler) which can capture timestamps and correlate them with host-side events.

---

## 6 — Using Vendor Profilers and Hardware Counters (rocprof / roctracer)

**High-level approach**
- Use `rocprof` to collect hardware counters (cache misses, instruction counts, occupancy, etc.) for your kernel execution.
- Insert **user markers** or named ranges (if supported) so that rocprof output shows the kernel region you care about.

**Typical workflow**
1. Run a warm-up iteration to stabilize schedules and caches.
2. Run `rocprof --stats` while executing your workload.
3. Inspect counter outputs, filter by kernel name, and look for high stall reasons (e.g., VMEM stalls, LDS bank conflicts).

**User markers & ranges**
- Some tracing libraries allow adding named ranges around code regions. Use them to isolate kernel stages (load, compute, store) in profiler timelines.

> Note: exact CLI flags and APIs evolve. Consult the latest ROCm/RGP docs for the precise commands for your ROCm version.

---

## 7 — Lightweight Range Markers (roctracer / rocprofiler)

If you want timeline visibility, insert API-based range markers or annotations (through the `roctracer` or `rocprofiler` APIs) to create readable spans on the trace. This requires linking against the tracing libraries and adding small host-side calls around kernel launches.

Example (conceptual):
```cpp
// Pseudocode host-side
roctxRangePush("load_tile");
// launch kernels that perform load
hipLaunchKernelGGL(loadTile, ...);
hipDeviceSynchronize();
roctxRangePop();
```

When captured, the profiler will show "load_tile" ranges that you can inspect.

---

## 8 — Reducing Instrumentation Overhead & Best Practices

- **Aggregate in fast memory:** Use shared memory or registers for local counters, flush to global memory rarely.
- **Sample, don't instrument everything:** Instrumenting every lane and every iteration dramatically changes behavior.
- **Use wide regions:** Measure larger code regions to amortize read overhead (e.g., measure whole compute phase rather than one add).
- **Warm-up runs:** Run a few warm-up iterations to prime caches and avoid first-run anomalies.
- **Use compiler flags:** Build with relevant optimizations turned on (`-O3`) and disable debug symbols for production measurements.
- **Turn instrumentation on/off via macros:** Use compile-time flags (`#ifdef INSTRUMENT`) to enable/disable instrumentation easily.
- **Document and version your instrumentation:** Keep instrumentation code under version control and guarded to avoid accidental performance test contamination.

---

## 9 — Examples & Patterns

### A. Per-block timing and single atomic to global results
```cpp
__global__ void block_timing_kernel(unsigned long long *out_cycles) {
    __shared__ unsigned long long s_start;
    if (threadIdx.x == 0) s_start = clock64();
    __syncthreads();

    // ... work here ...

    if (threadIdx.x == 0) {
        unsigned long long end = clock64();
        out_cycles[blockIdx.x] = end - s_start;
    }
}
```

### B. Count branch taken frequency (per-block aggregation)
```cpp
__global__ void branch_count_kernel(const int *A, int N, unsigned int *global_counts) {
    __shared__ unsigned int s_cnt;
    if (threadIdx.x == 0) s_cnt = 0;
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        if (A[gid] > 0) atomicAdd(&s_cnt, 1u);
    }
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(global_counts, s_cnt);
}
```

### C. Minimal sampling to capture per-iteration latency
```cpp
__global__ void sample_iteration_kernel(unsigned long long *samples, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Only one sample per block every 1024 iterations
    if ((gid % 1024) == 0) {
        unsigned long long t0 = clock64();
        // run iteration work
        unsigned long long t1 = clock64();
        if (threadIdx.x == 0) samples[blockIdx.x] = t1 - t0;
    }
}
```

---

## 10 — Collecting & Post-processing Instrumentation Data

- **Collect small arrays from device:** Write counters/samples to an output buffer on the device, then copy back with `hipMemcpy`.
- **Aggregate on host:** Compute averages, histograms, quantiles to identify hotspots and outliers.
- **Visualize:** Use simple plotting (Python/matplotlib) or load into Excel/Sheets for visual inspection.
- **Correlate with profiler data:** Use RGP/rocprof hardware counters to explain why a region is slow (e.g., VMEM stalls, low ILP).

---

## 11 — Troubleshooting & Common Pitfalls

- **Instrumentation perturbation:** If instrumentation changes the kernel behavior, reduce sampling or move counters to shared memory/blocks.
- **Atomic contention:** Atomic writes from many threads can be as expensive as the measured event; prefer aggregation to one atomic per block/wave.
- **Clock drift / wraparound:** `clock64()` may wrap over long runs. Use unsigned arithmetic and consider wrap detection.
- **Precision:** Cycle counts are precise for measuring cycles, but mapping to wall-time requires knowing GPU clock (frequency scaling and boost modes may complicate the mapping).

---

[← Kernel Optimization](kernel_optimization.md)
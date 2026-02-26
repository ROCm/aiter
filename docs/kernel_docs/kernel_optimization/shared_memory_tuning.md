# 🧠 Shared Memory (LDS) Tuning on AMD GPUs

[← Kernel Optimization](kernel_optimization.md)

---

Optimizing shared memory (also known as Local Data Share, or **LDS**) usage is essential for achieving high performance on AMD GPUs. This section provides guidelines, strategies, and examples to help you tune LDS usage effectively for compute kernels written in HIP or other GPU programming frameworks targeting AMD hardware.

---

## 🔍 Understanding LDS

LDS is a **fast, on-chip memory** that is accessible by all threads within a work-group. It serves as a programmable cache that can drastically reduce global memory bandwidth usage when used properly.

### Key properties:

* **Low latency:** Access latency is an order of magnitude lower than global memory.
* **High bandwidth:** Shared among all threads in a work-group.
* **Limited capacity:** Typically 64 KB per compute unit (CU) on modern AMD GPUs.
* **Banked architecture:** Divided into multiple banks for parallel access (often 32 banks per CU).

### Primary use cases:

* Tiling for matrix/memory operations (e.g., GEMM, convolution).
* Intra-workgroup data exchange.
* Cache blocking for stencil or reduction patterns.

---

## ⚙️ Key Tuning Parameters

### 1. **Tile size (work-group footprint)**

* Each work-group’s LDS usage = bytes per thread × number of threads per group.
* Adjust tile dimensions (`Tx`, `Ty`) to balance:

  * LDS consumption per CU (so multiple groups can coexist).
  * Memory access coalescing (alignment and locality).

#### Rule of thumb:

Aim for 2–4 work-groups to fit concurrently per CU. If each group’s LDS footprint is too large, occupancy will drop.

### 2. **Occupancy vs LDS trade-off**

Each CU can host multiple active work-groups concurrently, but LDS usage limits that number.

Formula:

```
max_workgroups_per_CU = floor(LDS_per_CU / LDS_per_workgroup)
```

Example (64 KB LDS/CU):

* If your kernel uses 32 KB LDS/work-group → only 2 groups can run concurrently.
* If reduced to 16 KB/work-group → up to 4 groups can run.

Higher occupancy helps hide latency but doesn’t always mean higher throughput. Measure performance at both high and moderate occupancy levels.

### 3. **Memory bank conflicts**

LDS is divided into multiple banks (often 32). When multiple threads in a wave access different addresses in the *same bank*, accesses serialize.

#### Avoiding bank conflicts:

* Use 2D array indexing that spreads accesses across banks.
* Add padding to the second dimension of shared arrays:

  ```cpp
  __shared__ float tile[BLOCK_Y][BLOCK_X + 1];
  ```
* Prefer vectorized loads/stores that naturally align with bank boundaries.

### 4. **LDS prefetching and reuse**

Load frequently reused data into LDS once per work-group and reuse it multiple times.

Example (tiling GEMM):

```cpp
__shared__ float A_tile[TILE_M][TILE_K];
__shared__ float B_tile[TILE_K][TILE_N];

for (int t = 0; t < K / TILE_K; ++t) {
    A_tile[ty][tx] = A[row * K + t * TILE_K + tx];
    B_tile[ty][tx] = B[(t * TILE_K + ty) * N + col];
    __syncthreads();

    // Compute partial results using LDS data
    for (int k = 0; k < TILE_K; ++k)
        Cvalue += A_tile[ty][k] * B_tile[k][tx];

    __syncthreads();
}
```

This ensures each input element is fetched from global memory only once per tile.

---

## 📏 Practical Tuning Process

1. **Determine baseline LDS use**
   Profile the kernel (e.g., with `rocprof` or Radeon GPU Profiler) to measure LDS consumption per work-group.

2. **Sweep tile sizes and LDS footprint**
   Vary tile sizes (`8×8`, `16×16`, `32×8`, etc.) and measure throughput. Track how performance correlates with occupancy.

3. **Check bank conflicts**
   Use compiler or profiler metrics (e.g., `LDSBankConflict`) to confirm you’re not hitting serialisation.

4. **Measure wave occupancy**
   Ensure that LDS use allows at least 2–3 resident waves per CU. Use `rocprof --stats` or a similar profiler.

5. **Experiment with coarsening**
   If LDS is the bottleneck, try processing multiple output elements per thread to reduce shared memory footprint.

---

## 🧩 Example: Tuning for a Matrix Multiply

| Tile Size | LDS per WG | Work-Groups per CU | Relative Performance |
| --------- | ---------- | ------------------ | -------------------- |
| 16×16     | 16 KB      | 4                  | 1.00×                |
| 32×16     | 32 KB      | 2                  | 1.12×                |
| 32×32     | 64 KB      | 1                  | 0.90×                |

Observation: moderate LDS use (32 KB) provided the best performance due to balanced reuse and occupancy.

---

## 🧮 Tips & Best Practices

* Use LDS for **reuse-heavy** data, not transient values.
* Always profile — LDS usage may interact with register pressure and vector width.
* Use compiler pragmas (`__launch_bounds__`) to guide resource allocation.
* Minimize LDS-to-global-memory round trips.
* Combine LDS with **subgroup operations** (e.g., warp reductions) for intra-wave communication.

---

## ⚠️ Common Pitfalls

* **Overusing LDS** — large tiles that monopolize LDS reduce CU concurrency.
* **Ignoring alignment** — misaligned data leads to bank conflicts and lower bandwidth.
* **Redundant loads** — loading the same data multiple times into LDS wastes bandwidth.
* **Unbalanced tile shapes** — rectangular tiles may cause underutilized threads or poor memory coalescing.

---

## 🧰 Tools for LDS Tuning

| Tool                             | Use                                              |
| -------------------------------- | ------------------------------------------------ |
| **Radeon GPU Profiler (RGP)**    | Inspect LDS usage, bank conflicts, CU occupancy  |
| **rocprof**                      | Measure LDS bytes, LDS stalls, occupancy metrics |
| **CodeXL / ROCm Analysis Tools** | Visualize memory hierarchy utilization           |

---

## 🧠 Final Notes

Effective LDS tuning is a balancing act between **reuse**, **occupancy**, and **bank efficiency**. Start simple — with a correct, working kernel — then incrementally increase LDS tiling complexity while measuring the effect on performance.

---

### 📚 References & Further Reading

* AMD GPUOpen: [GCN Memory Hierarchy](https://gpuopen.com/learn/amd-gcn-memory-hierarchy/)
* ROCm Documentation: [HIP Memory Model](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
* RGP User Guide: [Performance Analysis and Occupancy Metrics](https://gpuopen.com/rgp/)

---
[← Kernel Optimization](kernel_optimization.md)


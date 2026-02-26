# ⚙️ Launch Parameter Optimization (AMD kernels)

[← Kernel Optimization](kernel_optimization.md)

---

This subsection describes practical techniques and heuristics for choosing launch parameters and mapping threads to data on AMD GPUs (HIP). The goal is to improve utilization, memory bandwidth, and reduce synchronization/overhead. This is *not* a one-size-fits-all recipe — use the heuristics below to build a small tuning matrix for your specific kernel and target device.

---

## 🔑 Key Concepts (Quick Reference)

* **Wavefront / Wave size** — AMD GPUs schedule work in units called *waves* (also called wavefronts). A work-group should usually be a multiple of a wave to avoid underutilized lanes inside a wave. Check your device (GCN vs RDNA may differ; if unsure, query the runtime).

* **Work-group (block) size** — number of threads in a single work-group. Choose it to balance occupancy, local (shared) memory, and register pressure.

* **Occupancy** — how many active work-groups/waves can be resident on a compute unit. Higher occupancy often hides latency but isn’t always best for throughput if it increases cache / register pressure.

* **Thread coarsening (loop unrolling across threads)** — assign more than one data element per thread to reduce launch overhead and increase arithmetic intensity.

* **Tiling & shared memory** — use local memory (LDS) to reduce global traffic. Trade-off: larger tiles increase LDS use and may lower occupancy.

* **Memory alignment & vector loads** — coalesce global loads/stores and prefer vectorized loads (e.g., `float4`) where it matches alignment.

---

## 🧭 Heuristics for Choosing Work-Group Sizes

1. **Start from the wave size**

   * Make the work-group size a small multiple of the hardware wave size (e.g., `wave` × 1, 2, 4). This avoids partially filled waves and simplifies reasoning about lane activity.

2. **Prefer square-ish 2D blocks for 2D problems**

   * For 2D kernels (images, matrices), start with work-groups like `16x16`, `32x8`, or `8x32`, then tune. Layout depends on memory access patterns: row-major data often benefits from contiguous threads along the fastest-varying dimension.

3. **Respect local memory and register limits**

   * If your kernel uses local memory (LDS), compute how many bytes per work-group and ensure it fits multiple times into the per-CU limit. If a large tile needs too much LDS, lower the work-group count or increase thread coarsening instead.

4. **Balance occupancy vs per-thread work**

   * If occupancy is low because each thread uses many registers, try thread coarsening (reduce number of threads, increase work per thread) or reduce register pressure with compiler hints or simpler code.

5. **Use an empirical sweep**

   * Test a small grid of work-group sizes (e.g., `wave × {1,2,4}` for 1D; for 2D try `{8,16,32}` along each axis) and measure throughput and bandwidth. Keep the best performing set.

---

## 🧮 Mapping Threads to Data (Which Thread Processes Which Elements)

### Direct mapping (1 thread → 1 element)

* Simple and often fastest when per-element work is large and memory accesses are independent.
* Good baseline for correctness and easy to reason about coalescing.

### Coarsened mapping (1 thread → N elements)

* Assign consecutive data elements along the innermost dimension to the same thread. Example: each thread processes 4 consecutive floats via a small loop or unrolled code.
* Advantages:

  * Higher arithmetic intensity (more compute per memory transaction).
  * Fewer threads → potentially lower register/LDS pressure per CU.
  * Better for cases with small per-element work where launch overhead dominates.
* Disadvantages:

  * May complicate memory accesses and reduce vector/striped coalescing if not aligned.

**Implementation pattern (pseudo-HIP):**

```cpp
size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
size_t stride = blockDim.x * gridDim.x;
for (size_t i = gid; i < N; i += stride) {
    // process element i
}
```

This pattern automatically scales and avoids strict 1:1 mapping; it’s good for irregular sizes.

### Striped vs Contiguous Assignment

* **Contiguous per-thread:** thread `t` processes elements `[t*NperThread .. t*NperThread + (NperThread-1)]`. Good for vector loads and limited per-thread loops.
* **Striped:** thread `t` processes `[t, t+T, t+2T,...]` where `T` is total threads. Good for balancing irregular costs but can hurt memory locality.

Choose contiguous assignment for bulk floating-point work that benefits from contiguous memory access.

---

## 🤝 Shared Work and Cooperation Between Threads

When multiple output elements depend on overlapping input data, consider these patterns:

1. **Tile-based cooperative compute**

   * Each work-group loads a tile of input into LDS, threads cooperate to compute multiple outputs inside the tile, then write results out.
   * Use `__syncthreads()` to coordinate.
   * This pattern is ideal for convolution-like stencils or matrix multiply.

2. **Producer-consumer partitioning**

   * If some threads produce intermediate results used by others, try to place producer and consumer threads inside the same work-group so they can communicate through LDS. Avoid cross-work-group communication.

---

## 🧩 Practical Examples

### Example 1 — 2D Convolution Using Tiles

* Tile size: `T_x × T_y` chosen so that `T_x*T_y*element_size` fits comfortably into LDS multiplied by a safety factor (e.g., 0.6 of per-CU LDS).
* Work-group shape: choose `W_x × W_y` as a multiple of wave size (flatten to compute work-group size). Each thread may compute multiple output pixels (coarsening) to amortize loads.

Pseudo-steps:

1. Each work-group computes a tile of output.
2. Each thread cooperatively loads overlapping input patch into LDS with coalesced loads.
3. `__syncthreads()`
4. Each thread computes its outputs using the tile in LDS.
5. `__syncthreads()` (if needed)
6. Each thread writes its outputs back to global memory.

### Example 2 — GEMM Microkernel Mapping

* Use work-groups that map to micro-tiles of the output (e.g., `64x64` output tile split among threads).
* Each thread can compute a small accumulator block (e.g., `4x4`) so one thread writes multiple output values.
* Tune the micro-tile vs LDS usage; larger micro-tiles reduce global memory traffic but increase LDS and register usage.

---

## ⚠️ Common Pitfalls

* Picking a work-group size that is not a multiple of the wave size, causing wasted lanes.
* Over-allocating LDS and reducing occupancy to the point where arithmetic units stall.
* Excessive thread coarsening that breaks vector alignment or increases register pressure.
* Using striped assignments that kill memory locality for row-major data.

---


## 🧠 Final Notes

* Always measure. The GPU microarchitecture, compiler, and your kernel’s arithmetic/memory mix dictate the best choices.
* Document the best configurations per device and keep the tuning matrix as part of your repository.

---
[← Kernel Optimization](kernel_optimization.md)
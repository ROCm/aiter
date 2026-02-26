# 🧩 Tiling on AMD GPUs — Block, Wavefront, and Thread-Level Strategies

[← Kernel Optimization](kernel_optimization.md)

---

This section explains tiling strategies at multiple levels of the execution hierarchy on AMD GPUs: **block-level (work-group)** tiling, **wavefront-level** tiling (subgroup/wave cooperation), and **thread-level** tiling (per-thread coarsening). Tiling reduces global memory traffic, increases data reuse, and improves arithmetic intensity when used with LDS and careful thread mapping.

---

## 🎯 Goals of Tiling

* Reduce global memory bandwidth by reusing data in LDS or registers.
* Improve coalescing and alignment of memory accesses.
* Expose more compute per memory transaction (increase arithmetic intensity).
* Balance occupancy and resource usage (registers, LDS).

---

## 🧭 Levels of Tiling

1. **Block-level tiling (work-group / block)** — a work-group cooperatively loads a tile of input data into LDS and computes a tile of outputs. This is common in GEMM, convolutions, and stencils.
2. **Wavefront-level tiling (subgroup / wave)** — organize work so that entire wavefronts work on contiguous regions, using subgroup intrinsics for fast local cooperation and reductions.
3. **Thread-level tiling (coarsening)** — each thread processes multiple output elements (contiguous or strided), reducing overhead and increasing register reuse.

All three can be combined. For example, a block may be composed of several waves, each wave processes a sub-tile, and each thread computes multiple elements within that sub-tile.

---

## 🏗️ Block-Level Tiling (Work-group Cooperative Tiling)

### Pattern

1. Partition the output into tiles (e.g., `T_x × T_y`).
2. Each work-group loads the corresponding input tile (and halo if necessary) into LDS using coalesced loads.
3. Synchronize (`__syncthreads()`), compute outputs using the tiled data, and write results back.

### Example (2D convolution / stencil):

```cpp
// Tile of output: TILE_Y x TILE_X
__shared__ float tile[TILE_Y + 2*R][TILE_X + 2*R]; // include halo for radius R
int local_x = threadIdx.x;
int local_y = threadIdx.y;
int global_x = blockIdx.x * TILE_X + local_x;
int global_y = blockIdx.y * TILE_Y + local_y;

// Cooperative load into LDS (including halo)
for (int y = local_y; y < TILE_Y + 2*R; y += blockDim.y)
  for (int x = local_x; x < TILE_X + 2*R; x += blockDim.x)
    tile[y][x] = input[(global_y + y - R) * W + (global_x + x - R)];

__syncthreads();

// Compute output for threads assigned inside the tile
float out = 0.0f;
for (int ky = -R; ky <= R; ++ky)
  for (int kx = -R; kx <= R; ++kx)
    out += kernel[(ky+R)*K + (kx+R)] * tile[local_y + ky + R][local_x + kx + R];

__syncthreads();
output[global_y*W + global_x] = out;
```

### Tuning tips

* Choose `T_x` and `T_y` so that `T_x*T_y*sizeof(element)` (plus halo and other LDS usage) allows at least 2–4 work-groups resident per CU.
* Align tile width to the wavefront lane layout (e.g., ensure contiguous threads in x map to contiguous memory).
* Use loop unrolling for inner k-loops.

---

## 🔁 Wavefront-Level Tiling (Subgroup / Wave Cooperation)

Wavefront-level tiling treats a full wave (e.g., 64 lanes) as the cooperating unit. Advantages:

* Subgroup intrinsics (shuffles, reductions) execute faster than generic LDS operations.
* Lower synchronization cost — operations inside a wave don't need `__syncthreads()`.

### Patterns

* Assign each wave a contiguous sub-tile (e.g., a row chunk) and use lane IDs to index into that sub-tile.
* Use subgroup shuffles for cross-lane reductions and broadcasts.

### Example (vector reduce within wave):

```cpp
unsigned lane = threadIdx.x % waveSize; // lane id inside the wave
float val = compute_partial(...);
// binary-tree like shuffle reduction (conceptual)
for (int offset = waveSize/2; offset > 0; offset /= 2)
  val += __builtin_amdgcn_shfl_xor(val, offset);
if (lane == 0) wave_result = val;
```

### Tuning tips

* Arrange work so waves read contiguous memory regions for coalescing.
* Use subgroup intrinsics for fast data exchange and reductions, falling back to LDS only for cross-wave communication.
* Keep wave-subtile sizes tuned to avoid bank conflicts when staging into LDS.

---

## 🧵 Thread-Level Tiling (Per-thread Coarsening)

Thread-level tiling assigns multiple output elements to a single thread, typically contiguous elements along the fastest varying dimension. This reduces launch overhead and can improve register reuse.

### Patterns

* **Contiguous coarsening:** thread `t` computes `N` consecutive outputs starting at index `t*N`.
* **Strided coarsening:** thread `t` processes elements `t`, `t + stride`, `t + 2*stride`, ...

### Example (1D coarsening):

```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = gid*COARSEN + 0; i < N; i += stride*COARSEN)
  for (int k = 0; k < COARSEN; ++k)
    out[i + k] = process(in[i + k]);
```

### Tuning tips

* Use small coarsening factors (2–8) initially; larger factors may increase register pressure or reduce parallelism.
* Prefer contiguous coarsening for better vectorization and memory access patterns.
* Combine with block-level tiling to process micro-tiles per thread.

## ⚠️ Common Pitfalls

* Over-sized tiles that consume too much LDS and reduce occupancy.
* Poor thread-to-data mapping resulting in uncoalesced loads.
* Bank conflicts due to unfortunate LDS strides (fix with padding).
* Excessive register use in thread-level tiling that reduces active wavefronts.

---

## 🧾 Summary

Tiling is a multi-level strategy. Use block-level tiles to share data through LDS, wavefront-level subtile mapping and subgroup intrinsics to accelerate intra-block cooperation, and thread-level coarsening to increase work per thread and register reuse. Tune tile sizes to balance reuse and occupancy and always validate with profiler measurements.

---
[← Kernel Optimization](kernel_optimization.md)
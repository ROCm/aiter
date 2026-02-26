# ⚙️ AMD GPU Cache Optimization
[← Kernel Optimization](kernel_optimization.md)

---

This section explains how to design GPU kernels that effectively utilize the cache hierarchy on AMD CDNA and RDNA architectures. It focuses on cache access patterns, cache blocking for reuse, and minimizing bank conflicts.

---

## 🧭 Overview

AMD GPUs employ a **hierarchical memory system** designed to balance latency and bandwidth. Understanding the size, structure, and latency of each cache tier enables developers to design kernels that maximize data locality and throughput.

### 🧱 Cache Hierarchy (Typical CDNA Layout)

| Level                      | Type    | Scope                 | Size       | Latency (approx.) | Description                                               |
| -------------------------- | ------- | --------------------- | ---------- | ----------------- | --------------------------------------------------------- |
| **L0 Vector Cache**        | Private | Per SIMD              | ~16–32 KB  | 20–30 cycles      | Stores recently accessed vector data; feeds ALUs.         |
| **L1 Cache**               | Shared  | Per Compute Unit (CU) | 64–128 KB  | 30–50 cycles      | Feeds all wavefronts in a CU; caches global memory reads. |
| **L2 Cache**               | Global  | Shared by all CUs     | 4–8 MB     | 150–250 cycles    | Acts as the primary on-die global memory cache.           |
| **VRAM (HBM)**             | Global  | Device-wide           | Several GB | 400+ cycles       | Main memory for all GPU data.                             |
| **LDS (Local Data Share)** | Shared  | Per CU                | 64–128 KB  | 20–30 cycles      | Software-managed scratchpad memory.                       |

---

## 🔍 1. Understanding Cache Access Patterns

Efficient kernel design starts by aligning memory accesses with cache-line and bank structure.

### Key Concepts

* **Cache Line Size:** Typically 64–128 bytes. Accessing data that spans multiple lines increases latency and bandwidth usage.
* **Spatial Locality:** Consecutive threads in a wavefront should access consecutive addresses to utilize full cache lines.
* **Temporal Locality:** Frequently reused data should remain in cache or LDS as long as possible.

### Good vs. Bad Patterns

✅ **Coalesced Access:**

```cpp
output[i] = input[i] * scale; // contiguous access
```

⚠️ **Non-Coalesced Access:**

```cpp
output[i] = input[i * stride]; // large stride causes multiple cache lines to load
```

**Rule of Thumb:** One cache line per wavefront access is optimal.

---

## 🧩 2. Cache Blocking (Tiling for Efficient Reuse)

Cache blocking reorganizes work to reuse data within the cache before eviction.

### Design Principles

1. **Choose tile sizes that fit in L1 or LDS** depending on your reuse pattern.
2. **Reuse loaded data** across as many operations as possible before moving to the next tile.
3. **Exploit spatial and temporal locality** — nearby data is likely to be reused by neighboring threads.

### Example — Matrix Multiplication

Instead of reading entire rows or columns repeatedly:

* Partition the matrix into **tiles** small enough to fit in L1 or LDS.
* Each workgroup processes one tile of the output matrix.
* Tiles of input matrices are reused multiple times while resident in fast memory.

```cpp
for (int block_k = 0; block_k < K; block_k += TILE_K) {
    load_tile(A, block_k);
    load_tile(B, block_k);
    barrier();

    compute_tile(C, A, B);
    barrier();
}
```

### Selecting Tile Sizes

* **Fit within L1/LDS:**  Use empirical profiling to confirm tiles don’t exceed cache capacity.
* **Balance reuse and occupancy:** Larger tiles increase reuse but reduce occupancy due to higher memory use.
* **LDS capacity check:** LDS per CU is typically 64–128 KB. For FP32 data, that’s 16K–32K elements.

---

## 🧮 3. Reducing Cache and Bank Conflicts

Cache and LDS both use *banked architectures* to allow concurrent access. When multiple threads access the same bank simultaneously, **bank conflicts** serialize the requests.

### Key Concepts

- **Wavefront / subgroup**: a collection of work‑items that execute in lockstep on AMD (historically 64 lanes for GCN; modern RDNA varies; code should consider subgroups/wave sizes returned by the runtime).
- **Bank**: a slice of LDS. Typical AMD hardware uses 32 banks — mapping of an address to a bank is usually `bank = (address / element_size) % num_banks`.
- **Bank conflict**: two or more lanes of the same wavefront access different addresses that fall in the same bank during a single cycle; hardware serializes these accesses.
- **Permutation / swizzle**: a deterministic remapping of logical indices → physical addresses (for reads/writes) that spreads concurrent accesses across banks.


### Why Permutation Helps

Simple patterns (e.g., threads reading consecutive addresses) map naturally to different banks and are fast. Problems arise when your access stride or indexing maps multiple simultaneous lanes into the same bank (for example, when storing transposed tiles, scattering with certain strides, or performing reductions across specific strides).

Permutation transforms the index used to store/read in LDS so that simultaneous accesses land in different banks. Common methods:

- **Padding (stride padding):** add an extra column or element per row so that adjacent rows map to different banks.
- **XOR swizzle / bitwise permutation:** use bit-level operations (XOR, rotate, or bit-interleaving) to remap indices, often avoiding conflicts with minimal extra memory.
- **Hash-based permutation:** small hash functions provide good empirical spreading for complex access patterns.


### Practical Techniques

#### 1) Padding (fast and simple)

If you have a tile that is `TROW x TCOL` and you write `tile[row][col]` into LDS, use a padded stride so physical layout is `stride = TCOL + PAD`, where `PAD` is chosen to break the stride that causes conflicts. Usually `PAD = 1` (in elements) is enough when element size and lane access pattern cause conflicts.

**When to use:** matrix transposes, small tiles, when memory cost of padding is acceptable.

**Pros:** simple, low-overhead arithmetic.  
**Cons:** uses extra LDS; may not remove all conflicts for exotic patterns.

---

#### 2) XOR-swizzle / bitwise permutation (low memory overhead)

XOR-swizzle computes a new index with bitwise operations on the lane index (or on row/col indices). Example: `swizzled_idx = idx ^ (idx >> k)` or `swizzled_idx = idx ^ (lane & mask)`. This spreads addresses across banks by flipping bits that affect bank selection.

**When to use:** when you need to avoid padding overhead and want a compact remapping.

**Pros:** no extra LDS, cheap ops.  
**Cons:** must be chosen carefully for the architecture and access pattern; some patterns require different masks/shifts.

---

#### 3) Small permutation (lookup) tables

If you have a fixed small wavefront size and tile size, precompute a small permutation table in registers or constant memory and apply it to indices.

**When to use:** irregular or problem-specific permutations that are expensive to compute at runtime.

**Pros:** flexible, deterministic.  
**Cons:** uses a few registers/constant memory and may not scale to large tables.

---

### HIP Code Examples

Below are compact HIP examples showing common patterns: padded tile transpose and a XOR-swizzle store/load for a tile.

> **Note:** snippets are intended to illustrate the permutation idea — integrate bounds checks and tuning for your real kernel.

### Example A — Padded tile transpose (LDS padding)

```cpp
#include <hip/hip_runtime.h>

// tile sizes: tune for your GPU
constexpr int TILE_DIM = 32;
constexpr int PAD = 1; // 1 element pad to avoid bank conflicts

__global__ void transpose_padded(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int width, int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM + PAD]; // padded row

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  // load
  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }
  __syncthreads();

  // write transposed
  int tx = blockIdx.y * TILE_DIM + threadIdx.x;
  int ty = blockIdx.x * TILE_DIM + threadIdx.y;
  if (tx < height && ty < width) {
    out[tx * height + ty] = tile[threadIdx.x][threadIdx.y];
  }
}
```

**Why it helps:** the `+ PAD` offsets rows so lane `i` and lane `j` in the same wavefront are less likely to map to the same LDS bank.

---

### Example B — XOR-swizzle permutation for LDS index

This example shows a simple XOR-based swizzle applied to a 1D index before storing into LDS.

```cpp
#include <hip/hip_runtime.h>

__device__ inline unsigned swizzle_xor(unsigned idx, unsigned mask) {
  return idx ^ (idx & mask);
}

__global__ void swizzle_store_load(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int N, unsigned swizzle_mask) {
  extern __shared__ float sdata[]; // size must be configured by caller

  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (unsigned)N) return;

  // Compute local position in shared memory (e.g., tile-local index)
  unsigned local_idx = threadIdx.x; // example: 1:1 mapping

  unsigned phys_idx = swizzle_xor(local_idx, swizzle_mask);
  sdata[phys_idx] = in[gid];
  __syncthreads();

  // read back using inverse mapping — for XOR the operation is self-inverse
  unsigned read_idx = swizzle_xor(local_idx, swizzle_mask);
  out[gid] = sdata[read_idx];
}
```

**Notes:** XOR is self‑inverse (`a ^ m ^ m == a`), so the same function can remap back. Choose `swizzle_mask` to flip bits that influence bank selection — e.g., masks that affect low bits for element-size-aware bank mapping.

---

### How to Choose Parameters

- **Bank count:** assume 32 banks on most modern AMD GPUs; design permutations to spread accesses across 32 possible banks.
- **Element size:** bank mapping depends on element byte-width. For 4‑byte elements, bank = `(address / 4) % 32`.
- **Wavefront size:** Use `hipWavefrontSize`/`__builtin_` intrinsics or `hipDeviceProp_t` and runtime queries when writing portable kernels.
- **Swizzle mask selection:** Test masks that flip bits below `log2(num_banks)` — often masks like `0x1F` (for 5 low bits) or shifts of that are used experimentally.

---

## 📊 4. Designing Kernels for Cache Efficiency

When designing kernels, consider where your data lives in the hierarchy and how it moves between levels.

### Step-by-Step Design Process

1. **Identify data reuse:** Determine which data elements will be accessed repeatedly.
2. **Choose memory placement:** Store frequently reused data in LDS; transient data in L1.
3. **Align computation with cache reuse:** Each wavefront or workgroup should operate on data that fits entirely in fast memory.
4. **Overlap compute and memory:** While one tile is being processed, prefetch the next into cache or LDS.
5. **Minimize global reads:** Use intermediate storage to reuse data already fetched.

### Example — Blocking Strategy by Cache Level

| Cache   | Blocked Data                   | Block Size | Lifetime | Notes                     |
| ------- | ------------------------------ | ---------- | -------- | ------------------------- |
| **L2**  | Entire matrix section          | MB range   | Long     | Feeds multiple workgroups |
| **L1**  | Tiles used by one CU           | 64–128 KB  | Medium   | Fit per-tile operands     |
| **LDS** | Sub-tiles reused per wavefront | 16–32 KB   | Short    | Managed by programmer     |

---

## ⚠️ Common Pitfalls

* **Ignoring cache sizes:** Overly large working sets evict useful data.
* **Improper blocking:** Too-small tiles increase overhead; too-large tiles thrash caches.
* **No padding in LDS:** Bank conflicts can nullify the benefits of fast memory.
* **Global stride access:** Leads to excessive cache-line fetches and poor reuse.

---
## 🔬 AMD vs. NVIDIA Cache Architecture Differences

While both AMD and NVIDIA GPUs employ multi-level cache hierarchies to reduce memory latency, their architectures differ significantly in structure and tuning priorities. AMD’s CDNA and RDNA architectures emphasize large, unified L1 caches and software-managed Local Data Share (LDS) to provide predictable, high-throughput access for wavefront-based execution (64-thread groups). In contrast, NVIDIA’s architectures (such as Ampere and Hopper) feature smaller, hardware-managed L1 caches tightly coupled with each Streaming Multiprocessor (SM), relying more heavily on automatic caching and shared-memory fusion. AMD developers therefore have more explicit control over cache and memory behavior—allowing fine-grained tuning via LDS tiling, cache blocking, and vectorized memory patterns—while NVIDIA kernels tend to depend on compiler heuristics and warp-level optimizations. As a result, cache-optimized AMD code often requires deliberate data layout and manual reuse management to match or exceed NVIDIA-equivalent performance.

---

## 🧾 Final Notes

Cache efficiency on AMD GPUs depends on shaping your kernel’s working set to match cache capacities and aligning memory accesses to minimize conflicts. Optimal kernels maximize data reuse within each cache level, coalesce global accesses, and leverage LDS for temporary reuse.

---
[← Kernel Optimization](kernel_optimization.md)
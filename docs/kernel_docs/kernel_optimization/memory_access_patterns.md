# 🧭 Memory Access Patterns on AMD GPUs — Alignment, Stride, and Coalescing

[← Kernel Optimization](kernel_optimization.md)

---
This guide explains how memory access patterns affect performance on AMD GPUs. It focuses on: alignment, stride patterns, coalescing, and the effective "word" sizes for global (external) memory and LDS (local data share). Understanding these concepts helps you maximize bandwidth, avoid serialization, and write kernels that use the memory hierarchy efficiently.

---

## 🔑 Why this matters

Memory bandwidth and latency are frequently the dominant constraints in GPU kernels. Small, unaligned, or scattered memory transactions can dramatically reduce effective bandwidth and increase latency. Efficient memory access means larger, aligned transactions that match the hardware's natural word size and avoid serialization — both to global DRAM and to on-chip LDS.

---

## 🧠 Word size & transaction granularity

### Global memory (external DRAM)

* GPUs transfer memory between DRAM and L2/L1 in fixed-size *transaction* units. On AMD GPUs, the natural transaction size is typically 128 bytes (but can vary by generation and configuration). That means contiguous accesses that fit into 128‑byte aligned blocks are fastest.
* **Consequence:** A set of accesses that fall inside the same 128‑byte region can be serviced by a single memory transaction; scattered accesses across multiple 128‑byte regions generate multiple transactions and lower effective bandwidth.

### LDS (Local Data Share)

* LDS uses a banked on-chip memory system typically with 32 banks. The effective smallest parallel access granularity is a machine word (e.g., 32 or 64 bits depending on data type), but the hardware services accesses per bank. Non-uniform addresses that map to the same bank cause bank conflicts and serialization.

---

## 📎 Alignment: what to align and why

* **Base-address alignment:** Ensure your base pointers (buffers, arrays) are aligned to the transaction size (e.g., 128‑byte). Misaligned bases can split what could be one transaction into two.
* **Access alignment:** Prefer accessing elements so that threads in a wave read contiguous, aligned vectors (e.g., `float4`, `float2`) rather than single unaligned words.
* **Structure alignment:** For structs used in global I/O, ensure `sizeof(struct)` and field offsets are aligned to their natural types (use `alignas()` / `__attribute__((aligned(N)))`), and avoid packing that causes misaligned fields.

**Example:** aligning a buffer

```cpp
// Allocate host memory aligned to 128 bytes
void* ptr = nullptr;
posix_memalign(&ptr, 128, size);
// or use aligned allocators in your framework
```

---

## 🔁 Stride patterns: contiguous vs. strided vs. random

* **Contiguous**: Each thread in a wave reads successive elements (e.g., thread 0 → A[0], thread 1 → A[1], ...). This yields the best coalescing and largest combined transactions.
* **Strided**: Each thread reads elements separated by a stride (e.g., thread t reads A[t * stride]). Small strides (1, 2, 4) can still be efficient if they fit into vector loads or if stride leads to aligned vector groupings. Large strides that scatter across 128‑byte blocks cause many transactions.
* **Random/Gather**: Reads with irregular indices usually cause one transaction per unique cache line touched (bad for bandwidth). Consider reordering, tiling, or using a gather kernel with coalescing-friendly buffers.

**Rule of thumb:** Aim for contiguous accesses along the fastest-varying dimension of your data layout.

---

## 🔗 Coalescing and vectorized loads

* **Coalescing**: Modern AMD GPUs combine loads from multiple lanes into the fewest number of memory transactions possible if the addresses are within the same aligned transaction region. Coalescing is maximized when lane addresses are contiguous and aligned.
* **Vector loads/stores**: Use `float2`, `float4`, or `uint4` loads when alignment permits — this reduces instruction count and encourages larger transactions. But avoid misaligned vector loads; they may generate extra transactions or require multiple scalar loads.

**Example (HIP style):**

```cpp
// Assuming data is 16-byte aligned:
float4* vec = reinterpret_cast<float4*>(data);
float4 v = vec[gid]; // one vector load covers 4 floats
```

---

## 🧪 Small transactions and their cost

* Small (e.g., 4‑ or 8‑byte) transactions that don’t pack into the hardware transaction size (e.g., 128 bytes) waste bandwidth: a 128‑byte transaction may be performed to return just a few useful bytes.
* Overfetching may be cheaper than many small transactions if you can restructure access to fetch contiguous blocks once and reuse data in LDS.

**Strategy:** Prefer bulk, aligned reads and then reuse via LDS or registers rather than multiple tiny global reads.

---

## 🧭 LDS-specific considerations (banks, conflicts, and padding)

* LDS has multiple banks (commonly 32). Consecutive 32-bit words typically map to consecutive banks; threads in a wave accessing consecutive addresses map to different banks (good).
* **Bank conflicts** occur when multiple threads access different addresses that map to the same bank within the same cycle — these accesses serialize.

**Mitigations:**

* Add padding to row-strided arrays stored in LDS:

  ```cpp
  __shared__ float tile[BLOCK_Y][BLOCK_X + 1]; // +1 avoids stride being multiple of bank pattern
  ```
* Use data layouts that avoid having threads access addresses separated by multiples that alias banks.

---

## 🔀 Gather/Scatter and irregular access patterns

* Gather/scatter patterns are inherently bandwidth-inefficient. Use one of the following approaches to mitigate:

  1. **Reorder data** so that threads read contiguous memory (precompute an index permutation on CPU or in a preprocessing kernel).
  2. **Buffer and coalesce**: threads cooperatively load a block of indices and then perform gathers from a packed block.
  3. **Compress indices**: where indices have structure (e.g., periodic), compute addresses on-the-fly to reduce memory indirections.

## 💡 Examples

### Contiguous vs Strided (pseudo-HIP):

```cpp
// Contiguous (good)
int gid = blockIdx.x * blockDim.x + threadIdx.x;
out[gid] = in[gid];

// Strided (bad if stride is large)
int tid = threadIdx.x + blockIdx.x * blockDim.x;
for (int i = tid; i < N; i += blockDim.x) {
    out[i] = in[i * stride];
}
```

### Cooperative gather into LDS:

```cpp
// Phase 1: threads cooperatively load a contiguous block
int base = blockIdx.x * BLOCK_SIZE;
if (threadIdx.x < BLOCK_SIZE) {
  shared[threadIdx.x] = data[base + threadIdx.x];
}
__syncthreads();

// Phase 2: threads perform local gathers from shared
int idx = indices[global_idx];
float v = shared[idx - base];
```

---

## 📏 Measuring and validating

Use the following metrics:

* **Memory transactions** (number of DRAM transactions) — lower is better.
* **Achieved bandwidth** — compare to peak sustained bandwidth.
* **L2/HIT rates** — higher cache hit rates mitigate DRAM costs.
* **SM/ALU utilization** — to ensure compute units are not starved.

---

## ⚠️ Common pitfalls

* Assuming vector loads always help — misaligned vectors can be worse than scalar aligned loads.
* Ignoring cache behavior — small random reads may still be cached; measure before optimizing blindly.
* Over-padding LDS — padding removes bank conflicts but increases LDS footprint and can reduce occupancy.

---

## 🧾 Final notes

Optimizing memory access patterns is a mixture of understanding hardware primitives (transaction size, bank count) and refactoring algorithms to present contiguous, aligned access patterns. Always measure — microarchitectural effects vary by AMD generation and driver/compiler versions.

---
[← Kernel Optimization](kernel_optimization.md)


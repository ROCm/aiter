
# ⚙️📘 AMD GPU Synchronization Guide

[← Kernel Optimization](kernel_optimization.md)

---

## ✳️ Overview
On AMD GPUs, threads execute in **wavefronts** (analogous to CUDA warps). Most GCN/CDNA generations use **64 threads** per wavefront, while **RDNA** can operate with **32 or 64** depending on configuration. 

Synchronization on AMD is typically expressed at the **workgroup (CUDA block)** scope using barriers to order memory operations and coordinate multiple wavefronts resident in a CU. Device-side **rocSHMEM** adds OpenSHMEM/PGAS-style primitives to **order** and **complete** remote operations inside kernels. See the ROCm profiler’s LDS/compute unit model and rocSHMEM API.

---

## 🧩 Terminology
- **Wavefront (warp)**: SIMD execution group (64 on GCN/CDNA; 32/64 on RDNA). 
- **Workgroup (CUDA block)**: Threads that share **LDS** and can synchronize via barriers.
- **LDS (Local Data Share)**: On‑chip scratchpad for sharing and coordination. 

---

## 🧭 Differences vs CUDA’s BAR / `__syncwarp()`
- **CUDA named barriers** (`bar.sync`, `cuda::barrier`) and **warp sync** (`__syncwarp(mask)`) provide fine‑grained subset synchronization in a block. [CUDA Programming Guide – Async Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)  
- **HIP/ROCm does _not_ expose a direct `__syncwarp()` equivalent** today; guidance is to use `__syncthreads()`, `__threadfence_block()`, or refactor via **cooperative groups** tiled partitions where viable. 
- There is **no PTX‑style named barrier** in HIP for AMD targets; synchronization is primarily **workgroup‑wide** (`__syncthreads()`), plus algorithmic masks/shuffles and LDS protocols. [HIP C++ extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)


When only memory ordering is required, you can replace a block-level __syncthreads() with __threadfence_block() for better performance. This is an effective alternative to __syncwarp() on AMD GPUs for two key reasons:

Memory fencing: __threadfence_block() enforces memory ordering at the compute unit (CU) level, similar to the guarantees __syncwarp() provides at the SM level on NVIDIA GPUs.
Thread reconvergence: AMD wavefronts execute in lockstep, so explicit reconvergence (as done by __syncwarp()) is unnecessary.

For finer-grained control, AMD provides built-in functions that allow explicit fences and wavefront barriers. A common sequence uses a release fence before the barrier, synchronizes all lanes in the wavefront, and then applies an acquire fence after the barrier:

```c++
__builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
__builtin_amdgcn_wave_barrier();
__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
```

> ✅ **Conclusion:** AMD’s HIP/ROCm doesn’t offer the same BAR/named barrier syncs that CUDA/PTX exposes. Use workgroup barriers, LDS-based protocols, cooperative groups, and intra‑wavefront operations instead. 
---

## 🔒 Core Synchronization on AMD (HIP)

### 1) Workgroup barrier — `__syncthreads()`
Ensures all threads in a workgroup reach the barrier; orders LDS/shared‑memory updates across the group. Use to synchronize **between different wavefronts** in the same workgroup. [HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)

```cpp
__global__ void lds_barrier_example(int *dst) {
  extern __shared__ int s[]; // LDS
  int tid = threadIdx.x;
  s[tid] = tid;
  __syncthreads(); // all wavefronts rendezvous
  dst[tid] = s[(tid+1) % blockDim.x];
}
```

### 2) Memory fences
Block-level memory ordering without necessarily blocking all threads:
- `__threadfence_block()` — ensures the calling thread’s writes are visible to threads in the same block. [HIP reference](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)

### 3) Intra‑wavefront cross‑lane ops
Use DS‑permute or HIP shuffles for lane‑to‑lane communication. These are lock‑step within a wavefront and **do not** synchronize **between** wavefronts. See [here](intra_warp_communication.md).

---

## 🛠️ Patterns for sync **between different wavefronts** (same workgroup)
1. **LDS + barrier**: Producers write to LDS; `__syncthreads()`; consumers read.
2. **Phase counters in LDS**: Each wavefront updates a counter; consumers poll with periodic barriers; use `volatile` and fences judiciously. 
3. **Cooperative Groups (tiled)**: Partition a block (e.g., tiles of 64) and use `warp.sync()` semantics where HIP CG support applies; note it’s not a drop‑in for CUDA `__syncwarp()`. 

---

## 🧷 rocSHMEM Synchronization: `fence` vs `quiet`
**rocSHMEM** provides device‑side OpenSHMEM semantics. Core routines: [rocSHMEM API](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/)

### `rocshmem_fence()` / `rocshmem_ctx_fence(ctx)`
- **Orders** previously issued PUT/AMO/mem‑store ops **before** subsequent ops to the **same destination PE** (per‑PE ordering); does **not** guarantee completion. [rocSHMEM memory ordering](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/api/memory_ordering.html) · [OpenSHMEM manpage](https://docs.open-mpi.org/en/v5.0.x/man-openshmem/man3/shmem_fence.3.html)

### `rocshmem_quiet()` / `rocshmem_ctx_quiet(ctx)`
- **Completes** all previously issued operations in the context (makes updates visible), across destinations as needed; blocks until completion. [rocSHMEM memory ordering](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/api/memory_ordering.html)

#### Minimal device-side example
```cpp
#include <rocshmem.hpp>

__global__ void shmem_kernel(int *buf, int *remote, int pe, size_t n) {
  rocshmem_wg_init();
  rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ROCSHMEM_CTX_DEFAULT, &ctx);

  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    rocshmem_put_nbi(ctx, &remote[i], &buf[i], 1, pe);
  }

  rocshmem_ctx_fence(ctx); // ordering only
  rocshmem_ctx_quiet(ctx); // completion/visibility

  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}
```
Semantics align with OpenSHMEM: `fence` ⇒ ordering, `quiet` ⇒ completion/visibility. [OpenSHMEM manpage](https://docs.open-mpi.org/en/v5.0.x/man-openshmem/man3/shmem_fence.3.html) 

---

## 🔗 Atomics for Producer‑Consumer Synchronization
Atomic operations provide fine‑grained coordination between threads/wavefronts without full barriers.

### Common HIP atomics
`atomicAdd`, `atomicSub`, `atomicExch`, `atomicCAS` (compare‑and‑swap), `atomicMax/Min` on **LDS** or global memory. See HIP built‑ins. [HIP reference](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)

### Why atomics?
- Enable **lock‑free** producer/consumer queues or counters.
- Avoid whole‑block stalls when only a subset participates.

### Caveats
- Atomics serialize on contended locations; shard queues to reduce contention.
- Prefer **LDS** atomics for intra‑workgroup; global atomics are slower.
- Combine atomics with `__threadfence_block()` where ordering matters. [HIP reference](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)

#### Example: LDS producer–consumer ring buffer
```cpp
__global__ void producer_consumer(int *out, int N) {
    extern __shared__ int queue[]; // LDS queue of size N
    __shared__ int head, tail;     // indices in LDS

    if (threadIdx.x == 0) { head = 0; tail = 0; }
    __syncthreads();

    int tid = threadIdx.x;

    // Producers: first half of threads
    if (tid < blockDim.x / 2) {
        for (int i = tid; i < N; i += blockDim.x / 2) {
            int pos = atomicAdd(&tail, 1);   // reserve slot
            queue[pos % N] = i;              // produce item
        }
    }

    // Optional ordering for consumers
    __threadfence_block();
    __syncthreads();

    // Consumers: second half of threads
    if (tid >= blockDim.x / 2) {
        while (true) {
            int pos = atomicAdd(&head, 1);
            if (pos >= tail) break;          // no more items
            int item = queue[pos % N];
            out[pos] = item * 2;             // consume
        }
    }
}
```
**Notes:**
- Use `atomicCAS` to implement multi‑producer/multi‑consumer locks or to protect critical sections when needed.
- For very high throughput, shard the buffer per‑wavefront (e.g., `head[warp]`, `tail[warp]`) and periodically merge. 

---

## 🧮 Wavefront size: portability tip
Don’t hardcode `32`. Query `warpSize` or design for **64** (typical on AMD) and parameterize tiles for RDNA (**32 or 64**). [HIP hardware docs](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)

---

## 🧑‍🔬 When you miss CUDA’s BARs
If you relied on `bar.sync`/named barriers or `cuda::barrier`, refactor to workgroup‑wide barriers and LDS protocols, or split kernels. There is **no HIP analogue** of CUDA’s **asynchronous named barriers** on AMD. 

For warp‑level rendezvous, use **cooperative groups** tiled partitions (where supported), or design lock‑step algorithms that avoid needing warp sync beyond intra‑wavefront lockstep.

---

## ✅ AMD‑friendly Sync Checklist
- [ ] Use `__syncthreads()` for **inter‑wavefront** coordination within a workgroup.
- [ ] Use **LDS** for data exchange; avoid global memory races.
- [ ] Use **rocSHMEM** `fence` (ordering) and `quiet` (completion) for device‑side PGAS.
- [ ] Prefer **atomics in LDS** for producer/consumer; shard to reduce contention.
- [ ] Avoid assuming warp=32; design for wavefront=64 and tile as needed.
- [ ] Replace CUDA named barriers with LDS + block barriers or multi‑kernel phases.

---

## 📚 References
- **AMD HIP hardware & warp size**: [HIP hardware implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html) 
- **HIP language & sync differences**: [HIP C++ extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)
- **CUDA barriers & `bar.sync`**: [CUDA Programming Guide – Async Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)
- **rocSHMEM fence/quiet**: [rocSHMEM memory ordering](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/api/memory_ordering.html) 

[← Kernel Optimization](kernel_optimization.md)
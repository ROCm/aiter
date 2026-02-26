# Wavefront Specialization — Dedicated Wavefronts for Subtasks (Load / Compute / Store)
[← Kernel Optimization](kernel_optimization.md)

---

## 🧠 Overview

**Wavefront specialization** is an optimization strategy in GPU kernels where individual **wavefronts** within a workgroup are assigned **dedicated roles**, such as:

- **Data loading** from global memory into LDS (shared memory)  
- **Computation** using data already in LDS  
- **Result storage** back to global memory  

This enables **pipelined parallelism** — while one wavefront is loading the next data tile, others are performing computation or writing results — resulting in **higher throughput and better utilization** of compute and memory units.

---

## ⚙️ Key AMD Concepts

| Concept | Description |
|----------|--------------|
| **Wavefront** | The SIMD execution unit of AMD GPUs (typically 64 threads) executing in lockstep. |
| **Workgroup** | A collection of wavefronts cooperating on a kernel task. |
| **LDS (Local Data Store)** | On-chip shared memory for inter-wavefront data exchange. |
| **Asynchronous copy** | In newer architectures (e.g. CDNA3, RDNA3), allows partial overlap of global → LDS transfer with compute. |

---

## 🎯 Why Use Wavefront Specialization?

- Overlap **data transfer and computation**  
- Reduce **stall cycles** due to memory latency  
- Increase **effective occupancy** and **utilization**  
- Exploit **fine-grained parallelism** across multiple wavefronts  

Example analogy:
> Think of one wavefront as a “loader” feeding data, while others are “workers” processing the previous chunk — like an assembly line.

---

## 🧩 Implementation Steps

### Step 1: Identify Your Kernel Subtasks

Break down your algorithm into phases:
1. **Load phase** — move data from global → LDS  
2. **Compute phase** — perform math using LDS data  
3. **Store phase** — write computed results to global memory  

### Step 2: Assign Wavefront Roles

Each wavefront in a workgroup can compute its ID:

```cpp
uint wavefront_id = (get_local_id(0) / 64); // assuming 64-lane wavefronts
uint lane_id = get_local_id(0) % 64;
```

Then branch based on `wavefront_id`:

```cpp
if (wavefront_id == 0) {
    // Loader wavefront
} else if (wavefront_id < 3) {
    // Compute wavefronts
} else {
    // Writer wavefront
}
```

### Step 3: Use LDS (Shared Memory) for Data Exchange

Use LDS to pass data between specialized wavefronts:

```cpp
__local float tileA[64];
__local float tileB[64];

if (wavefront_id == 0) {
    // Load global data into LDS
    tileA[lane_id] = globalA[global_offset + lane_id];
    tileB[lane_id] = globalB[global_offset + lane_id];
}

barrier(CLK_LOCAL_MEM_FENCE);  // ensure data visible to compute wavefronts

if (wavefront_id == 1) {
    // Compute using LDS data
    float sum = tileA[lane_id] * tileB[lane_id];
    partial_results[lane_id] = sum;
}

barrier(CLK_LOCAL_MEM_FENCE);
```

---

## 🚀 Step 4: Overlap Load and Compute (Double Buffering)

To achieve true overlap between loading and computing, you can **double-buffer** LDS memory:

```cpp
__local float bufferA[2][64];
__local float bufferB[2][64];

for (int tile = 0; tile < numTiles; ++tile) {
    int buf = tile % 2;

    if (wavefront_id == 0) {
        // Load next tile into LDS buffer buf
        bufferA[buf][lane_id] = A[tile * 64 + lane_id];
        bufferB[buf][lane_id] = B[tile * 64 + lane_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (wavefront_id == 1) {
        // Compute from previous buffer
        int prev_buf = (tile + 1) % 2;
        output[tile * 64 + lane_id] = bufferA[prev_buf][lane_id] * bufferB[prev_buf][lane_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
```

This structure overlaps wavefront 0’s loading with wavefront 1’s computation.

---

## ⚙️ (Optional) Step 5: Use `amdgcn` Intrinsics for Efficiency

On supported AMD architectures, you can use LLVM/ROCm intrinsics for more efficient global→LDS copies:

```cpp
// Load a dword from global into LDS at a given offset
__builtin_amdgcn_global_load_lds_f32(dst_lds_addr, src_global_ptr);
```

Or asynchronous copy if supported:
```cpp
__builtin_amdgcn_ds_bpermute(lane_id * 4, value);
```

> 🔧 Note: These low-level operations are architecture-specific (e.g., CDNA2/CDNA3) and may not be directly exposed in HIP/OpenCL yet.

---

## 🧮 Example: Matrix Multiply with Wavefront Specialization

```cpp
__kernel void wf_specialized_matmul(__global float* A,
                                    __global float* B,
                                    __global float* C, int N) {
    __local float tileA[64];
    __local float tileB[64];

    uint wavefront_id = get_local_id(0) / 64;
    uint lane = get_local_id(0) % 64;

    for (int tile = 0; tile < N / 64; ++tile) {
        if (wavefront_id == 0) {
            // Loader wavefront
            tileA[lane] = A[tile * 64 + lane];
            tileB[lane] = B[tile * 64 + lane];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (wavefront_id == 1) {
            // Compute wavefront
            float sum = 0.0f;
            for (int i = 0; i < 64; ++i)
                sum += tileA[i] * tileB[i];
            C[lane + tile * 64] = sum;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
```

---

## ⚖️ Best Practices & Tips

| Tip | Description |
|-----|--------------|
| **Keep roles balanced** | Ensure your loader wavefronts don’t bottleneck compute wavefronts. |
| **Use LDS efficiently** | LDS is limited (64 KB per CU). Avoid excessive buffering. |
| **Minimize barriers** | Only synchronize when necessary; wavefronts in the same workgroup can communicate implicitly via LDS. |
| **Profile overlap** | Use ROCm Profiler or Nsight Systems to confirm actual concurrency. |
| **Experiment with tile sizes** | Adjust per-architecture (CDNA vs RDNA). |

---

## 🧱 Hardware Notes

- AMD’s **CDNA** and **RDNA3** architectures increasingly support **asynchronous global-to-LDS transfers** and **fine-grained scheduling**, improving viability of this approach.  
- Earlier GCN architectures may rely on coarse-grained synchronization and explicit barriers.

## 📚 References

Wavefront optimizations : [HipKittens: Fast and Furious AMD Kernels](https://hazyresearch.stanford.edu/static/posts/2025-11-09-hk/hipkittens.pdf)
---
[← Kernel Optimization](kernel_optimization.md)
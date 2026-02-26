# 🧩 Threading Strategy on AMD GPUs — Mapping Threads to Data

This guide explains how to design efficient threading strategies for AMD GPU kernels. It focuses on how threads process data elements, how multiple threads can share computation for output results, and how to map work efficiently to GPU hardware to maximize throughput and memory efficiency.

[← Kernel Optimization](kernel_optimization.md)

---

## 🎯 Overview

An AMD GPU executes work in **wavefronts** (typically 64 threads). Each wavefront runs in lockstep on a Compute Unit (CU), and performance depends heavily on how threads map to data and how memory accesses are structured.

The goal of a good threading strategy is to:

* Maximize parallel efficiency by keeping all lanes active.
* Align thread-to-data mapping with memory layout.
* Minimize redundant computation and synchronization.
* Exploit data reuse through shared resources (LDS or registers).

---

## ⚙️ Understanding the Hardware Model

* **Thread**: The smallest execution entity. Threads are grouped into wavefronts.
* **Wavefront (warp)**: 64 threads that execute in SIMD fashion.
* **Work-group (block)**: A collection of wavefronts that share LDS and synchronize using barriers.
* **Compute Unit (CU)**: Hardware that runs several work-groups concurrently.

When designing threading strategies, think in terms of wavefronts and work-groups rather than individual threads.

---

## 🧠 Mapping Threads to Data Elements

### 1. One-thread-per-element (direct mapping)

The simplest and most common approach:

```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
output[gid] = inputA[gid] + inputB[gid];
```

* **Pros:** Straightforward and highly parallel.
* **Cons:** May lead to inefficient memory access if data is not contiguous.

**Best used when:** Each data element is independent and contiguous in memory.

---

### 2. One-thread-per-multiple-elements (loop mapping)

Each thread processes several elements in a loop:

```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
    output[i] = inputA[i] * inputB[i];
}
```

* **Pros:** Reduces kernel launch overhead for large data.
* **Cons:** Lower concurrency per iteration; must balance against available CU count.

**Best used when:** Data set is very large and the kernel is memory-bound.

---

### 3. Multi-thread-per-element (shared computation)

Multiple threads cooperate to produce one output element. Common in reductions, convolutions, and matrix multiplications.

Example (partial reduction in LDS):

```cpp
__shared__ float tile[BLOCK_SIZE];
int tid = threadIdx.x;
int gid = blockIdx.x * blockDim.x + tid;

// Load data to shared memory
float val = input[gid];
tile[tid] = val;
__syncthreads();

// Parallel reduction
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        tile[tid] += tile[tid + s];
    __syncthreads();
}

if (tid == 0)
    output[blockIdx.x] = tile[0];
```

* **Pros:** Enables shared computation and data reuse.
* **Cons:** Requires synchronization and may reduce occupancy.

**Best used when:** Computations have data dependencies or reuse shared data.

---

## 🔄 Sharing Threads for Multiple Outputs

In some algorithms (e.g., stencil filters, matrix transpositions, or convolution), each thread contributes to multiple output elements.

Example (stencil computation):

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

float val = 0.0f;
for (int j = -R; j <= R; ++j)
  for (int i = -R; i <= R; ++i)
    val += input[(y+j)*width + (x+i)] * filter[(j+R)*F + (i+R)];

output[y*width + x] = val;
```

Here, neighboring threads read overlapping input regions, so memory accesses can be coalesced and cached effectively.

**Tip:** Use LDS to cache shared input tiles and reduce redundant global memory loads.

---

## 🧩 Threading Strategies by Kernel Type

| Kernel Type                | Typical Threading Strategy                | Key Optimizations                          |
| -------------------------- | ----------------------------------------- | ------------------------------------------ |
| **Element-wise**           | One-thread-per-element                    | Ensure contiguous access and aligned loads |
| **Reduction / Prefix-sum** | Multi-thread-per-output                   | Use tree reduction, LDS buffering          |
| **Stencil / Convolution**  | Threads share overlapping input           | Use shared tiles and coalesced reads       |
| **Matrix Multiply (GEMM)** | Cooperative block of threads per tile     | Shared memory tiling, register blocking    |
| **Scatter / Gather**       | Thread reads scattered, writes contiguous | Reorder or batch reads for coalescing      |

---

## 📏 Choosing Work-group Size

The **work-group size** (threads per block) affects both occupancy and wavefront scheduling.

**Guidelines:**

* Use multiples of 64 (wavefront size).
* Tune for balance between occupancy and resource use (registers, LDS).
* Typical efficient sizes: 128, 256, or 512 threads per block.
* For 2D/3D kernels, ensure block dimensions match data layout (e.g., 16×16 threads for 2D tiles).

**Example (matrix tiling):**

```cpp
#define TILE_X 16
#define TILE_Y 16
__global__ void matmul(float* A, float* B, float* C, int N) {
  __shared__ float tileA[TILE_Y][TILE_X];
  __shared__ float tileB[TILE_Y][TILE_X];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE_Y + ty;
  int col = blockIdx.x * TILE_X + tx;

  float sum = 0;
  for (int t = 0; t < N / TILE_X; ++t) {
    tileA[ty][tx] = A[row * N + (t * TILE_X + tx)];
    tileB[ty][tx] = B[(t * TILE_Y + ty) * N + col];
    __syncthreads();

    for (int k = 0; k < TILE_X; ++k)
      sum += tileA[ty][k] * tileB[k][tx];
    __syncthreads();
  }
  C[row * N + col] = sum;
}
```

Here, each thread computes one `C[row, col]` value, but all threads in a block share sub-tiles of `A` and `B` through LDS.

---

## ⚡ Balancing Load and Resource Use

* **Too few threads:** Underutilizes GPU cores and hides less latency.
* **Too many threads per block:** May reduce occupancy due to high register or LDS usage.
* **Unbalanced workloads:** If data elements vary in computational cost, assign multiple elements per thread or use work-stealing techniques.

---
[← Kernel Optimization](kernel_optimization.md)


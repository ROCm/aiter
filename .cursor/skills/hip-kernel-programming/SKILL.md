---
name: hip-kernel-programming
description: HIP C++ GPU programming guide and AMD Composable Kernel (CK) library reference. Use when writing HIP/CUDA kernels targeting AMD GPUs, using the CK template library for GEMM/attention/convolution, understanding HIP API equivalents to CUDA, or optimizing C++ GPU code for ROCm. Covers HIP syntax, memory management, kernel launch, shared memory, synchronization, and CK tile programming.
---

# HIP Kernel Programming & Composable Kernel Guide

## HIP Language Basics

HIP (Heterogeneous-computing Interface for Portability) is AMD's C++ API for GPU programming. It is source-compatible with CUDA, meaning most CUDA code can be compiled with HIP with minor changes.

### CUDA to HIP Translation

| CUDA | HIP | Description |
|------|-----|-------------|
| `cudaMalloc` | `hipMalloc` | Allocate device memory |
| `cudaMemcpy` | `hipMemcpy` | Copy memory |
| `cudaFree` | `hipFree` | Free device memory |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | Sync device |
| `cudaStream_t` | `hipStream_t` | Stream type |
| `__shared__` | `__shared__` | Shared memory (same) |
| `__syncthreads()` | `__syncthreads()` | Block sync (same) |
| `__global__` | `__global__` | Kernel function (same) |
| `threadIdx.x` | `threadIdx.x` | Thread index (same) |
| `blockIdx.x` | `blockIdx.x` | Block index (same) |
| `blockDim.x` | `blockDim.x` | Block dimension (same) |
| `<<<grid, block>>>` | `<<<grid, block>>>` | Launch syntax (same) |

### Key Difference: Warp/Wavefront Size
- **CUDA:** Warp = 32 threads
- **HIP/AMD:** Wavefront = 64 threads
- Use `__AMDGCN_WAVEFRONT_SIZE` or `warpSize` (set to 64)

### HIP Kernel Template

```cpp
#include <hip/hip_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch
dim3 block(256);
dim3 grid((n + block.x - 1) / block.x);
vector_add<<<grid, block>>>(a, b, c, n);
hipDeviceSynchronize();
```

### Shared Memory (LDS)

```cpp
__global__ void tiled_matmul(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += TILE_SIZE) {
        // Cooperative load to shared memory
        As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute on shared memory
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

**Bank conflict avoidance:** Pad shared memory arrays: `float As[TILE][TILE+1]`

### Launch Bounds

```cpp
__global__ void __launch_bounds__(256, 4)  // 256 threads, min 4 blocks/CU
my_kernel() { ... }
```

## HIP Optimization Checklist

1. **Coalesce memory:** Consecutive threads access consecutive addresses
2. **Vectorize loads:** Use `float4`, `int4` types for 128-bit loads
3. **Minimize divergence:** Avoid `if/else` within a wavefront
4. **Use shared memory:** For data reused across threads
5. **Pad shared memory:** Prevent bank conflicts (+1 column)
6. **Prefer FP32 single-precision math:** `sinf()` not `sin()`
7. **Use intrinsics:** `__expf()`, `__logf()`, `__fsqrt_rn()`
8. **Align data:** 128-byte alignment for global memory
9. **Block sizes:** Multiples of 64 (wavefront size)
10. **Reduce register pressure:** Minimize live variables

## Composable Kernel (CK) Library

CK is AMD's C++ template library for building high-performance GPU kernels. It provides optimized building blocks for GEMM, convolution, attention, and reductions.

### CK Concepts

- **Device Operation:** Top-level abstraction (e.g., `DeviceGemm`, `DeviceBatchedGemm`)
- **Kernel:** GPU kernel launched by a device operation
- **Tile:** Block of data processed by a workgroup
- **Thread:** Individual computation within a tile
- **Tensor Coordinate Transformation:** Maps logical indices to physical memory

### CK GEMM Template

```cpp
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

// Template parameters define the full kernel configuration
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle_V3<
    ck::tensor_layout::gemm::RowMajor,    // A layout
    ck::tensor_layout::gemm::ColumnMajor, // B layout
    ck::tensor_layout::gemm::RowMajor,    // C layout
    ck::half_t,                            // A data type
    ck::half_t,                            // B data type
    ck::half_t,                            // C data type
    float,                                 // Accumulator type
    // Tile sizes
    256,    // MPerBlock
    256,    // NPerBlock
    128,    // KPerBlock
    16,     // AK1 (A vectorization)
    // MFMA instruction config
    32,     // MPerXdl
    32,     // NPerXdl
    4,      // MRepeat
    2,      // NRepeat
    // Thread cluster config
    S<4, 64, 1>,  // ABlockTransfer thread cluster
    S<1, 0, 2>,   // ABlockTransfer thread map
    S<1, 0, 2>,   // ABlockTransfer src access order
    2,             // ABlockTransfer src vector dim
    16,            // ABlockTransfer src scalar per vector
    // ... (similar for B)
    7,             // CShuffle MXdlPerWavePerShuffle
    2,             // CShuffle NXdlPerWavePerShuffle
    // Pipeline stages
    S<1, 32, 1, 8>,  // Block-to-CTile map
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v4
>;
```

### CK Tile (Newer API)

CK Tile is a more modular API used for FMHA (flash attention) and newer kernels:

```cpp
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

// FMHA arguments struct
struct mha_fwd_args {
    void* q_ptr;
    void* k_ptr;
    void* v_ptr;
    void* o_ptr;
    void* lse_ptr;     // Log-sum-exp for backward
    // Strides (in elements)
    ck_tile::index_t stride_q, nhead_stride_q, batch_stride_q;
    ck_tile::index_t stride_k, nhead_stride_k, batch_stride_k;
    ck_tile::index_t stride_v, nhead_stride_v, batch_stride_v;
    ck_tile::index_t stride_o, nhead_stride_o, batch_stride_o;
    // Dimensions
    ck_tile::index_t seqlen_q, seqlen_k;
    ck_tile::index_t hdim_q, hdim_v;
    ck_tile::index_t nhead_q, nhead_k;
    ck_tile::index_t batch;
    float scale_s;      // 1/sqrt(d)
};
```

### CK Kernel Selection Pattern

```cpp
// 1. Check tuned lookup table (from CSV config)
auto it = KernelMap.find({M, N, K});
if (it != KernelMap.end()) return it->second;

// 2. Heuristic dispatch by shape
if (M <= 16) return small_m_kernel;
if (M >= 256 && N >= 256) return large_kernel;
return default_kernel;

// 3. Bucket M to power-of-2 for better cache behavior
int M_padded = next_power_of_2(M);
```

### CK Instance Generation (Codegen)

CK uses Python scripts to generate C++ template instantiations:

```python
# gen_instances.py pattern
CONFIGS = [
    {"m_per_block": 256, "n_per_block": 256, "k_per_block": 128, ...},
    {"m_per_block": 128, "n_per_block": 128, "k_per_block": 64, ...},
]
for cfg in CONFIGS:
    generate_cpp_file(cfg)  # Creates .cu file with template instantiation
```

## PyBind Integration Pattern

```cpp
// In rocm_ops.hpp:
#define MY_KERNEL_PYBIND \
    m.def("my_kernel", &my_kernel_func, "Description", \
          py::arg("input"), py::arg("weight"), py::arg("output"));

// In *_pybind.cu:
#include "rocm_ops.hpp"
#include "my_kernel.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    MY_KERNEL_PYBIND;
}
```

## HIP Compilation

```bash
# Compile HIP kernel
hipcc -O3 --offload-arch=gfx942 kernel.hip -o kernel

# Check register usage
hipcc --resource-usage kernel.hip

# For aiter JIT: handled by aiter/jit/core.py using optCompilerConfig.json
```

## References

- HIP Programming Guide: https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/programming_manual.html
- HIP Performance Guide: https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html
- CK Documentation: https://rocm.docs.amd.com/projects/composable_kernel/en/develop/
- CK GitHub: https://github.com/ROCm/composable_kernel
- CK Tile Tutorial: https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.2.0/tutorial/tutorial_hello_world.html

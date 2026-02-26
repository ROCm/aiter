# ⚙️ Using Sparsity Hardware on AMD GPUs
[← Kernel Optimization](kernel_optimization.md)

---

AMD GPUs, particularly those based on the **CDNA** and **RDNA** architectures, include hardware and instruction-level support for **sparse computation** — allowing workloads that contain structured zeros in their data to execute faster and more efficiently. This guide explains how to take advantage of sparsity hardware and integrate sparse operations effectively in your HIP or ROCm-based applications.

---

## 🧠 What Is Sparsity?

**Sparsity** refers to matrices or tensors that contain a significant number of zero elements. Rather than performing unnecessary multiply–accumulate (MAC) operations involving zero values, AMD hardware can **skip** or **compress** these operations using specialized instructions and data layouts.

There are two main types:

| Type | Description | Example |
|------|--------------|----------|
| **Unstructured Sparsity** | Zeros occur at arbitrary locations. | General neural network pruning |
| **Structured Sparsity** | Zeros follow a fixed pattern (e.g., 2:4 block sparsity). | Training/inference optimization |

---

## 🧩 Hardware Support for Sparsity (CDNA/CDNA2/CDNA3)

AMD’s **CDNA** architecture (used in MI100, MI200, MI300) supports sparsity acceleration primarily through:

- **MFMA (Matrix Fused Multiply-Add) units** with **sparse matrix instruction variants**.
- **Data compression** that reduces memory bandwidth usage for sparse tensors.
- **Wave-level parallelism** that exploits structured sparsity patterns efficiently.

For structured 2:4 sparsity (two nonzero values out of every four), AMD GPUs can **double throughput** compared to dense operations — provided the data layout matches the hardware’s expectations.

---

## 🚀 Why Use Sparsity Hardware?

| Advantage | Description |
|------------|-------------|
| **Higher throughput** | Skip unnecessary zero-multiplications in dense math. |
| **Lower memory bandwidth** | Fewer non-zero elements reduce traffic between HBM and compute units. |
| **Improved energy efficiency** | Reduced operations lead to lower power consumption. |
| **Accelerated inference/training** | Particularly beneficial for pruned or quantized deep learning models. |

---

## ⚙️ How to Use Sparsity in Your HIP Code

While HIP does not yet expose direct sparse MFMA intrinsics, you can leverage ROCm libraries and data formats optimized for sparse computation.

### 🧱 1. Using rocSPARSE

AMD provides **rocSPARSE**, a high-performance library for sparse matrix operations (analogous to cuSPARSE).

Example:

```cpp
#include <rocsparse/rocsparse.h>

// Example: y = A * x  (sparse matrix-vector multiplication)
void sparse_mv_example(rocsparse_handle handle, int m, int n, int nnz,
                       const float* d_values, const int* d_rowPtr, const int* d_colInd,
                       const float* d_x, float* d_y) {
    const float alpha = 1.0f, beta = 0.0f;

    rocsparse_mat_descr descr;
    rocsparse_create_mat_descr(&descr);

    rocsparse_scsrmv(handle,
                     rocsparse_operation_none,
                     m, n, nnz,
                     &alpha,
                     descr,
                     d_values, d_rowPtr, d_colInd,
                     d_x, &beta, d_y);
}
```

✅ **rocSPARSE** handles optimized storage formats (CSR, COO, BSR) and ensures correct data alignment for AMD hardware.

---

### 🧮 2. Structured Sparsity with MFMA (Matrix Core Instructions)

If your workload uses structured sparsity (e.g., 2:4), the compiler can target specialized MFMA instructions such as `v_mfma_f32_16x16x16_bf16_1k_sparse`. These allow skipping of known zero patterns at hardware level.

Tips:
- Ensure **2:4 pattern compliance** — two nonzero values per four.
- Use **rocWMMA** or **hipTensor** for higher-level matrix abstractions.
- Apply **compiler flags** like `--amdgpu-target=gfx940` (MI300) to enable MFMA sparsity paths.

---

### ⚙️ 3. Data Layout and Memory Efficiency

| Format | Description | Use Case |
|---------|--------------|----------|
| **CSR/COO/BSR** | General sparse matrices (irregular patterns) | Scientific computing |
| **Block-sparse (2:4)** | Structured sparsity for ML workloads | Deep learning inference |
| **Compressed weight tensors** | Stored as dense + mask tensor | Transformer models |

Align sparse matrix blocks to **128-bit memory boundaries** to ensure efficient global memory access and avoid partial line fetches.

---

## 📊 Performance Considerations

| Factor | Recommendation |
|--------|----------------|
| **Sparsity Ratio** | >50% zeros generally needed for benefit |
| **Block Size** | Larger blocks improve cache reuse |
| **Thread Mapping** | Map wavefronts to blocks of non-zero data |
| **Prefetching** | Use `__builtin_amdgcn_s_memtime` to time and optimize prefetch distance |
| **Mixed Precision** | Combine sparsity with BF16 or FP16 for best throughput |

---

## 🧭 Summary

| Key Point | Description |
|------------|-------------|
| **Use rocSPARSE** | For general sparse matrix operations |
| **Exploit 2:4 sparsity** | On CDNA2+ hardware for MFMA speedups |
| **Compress and align data** | To maximize bandwidth efficiency |
| **Profile regularly** | Use `rocprof` to identify compute vs. memory limits |

By leveraging sparsity hardware, AMD GPUs can achieve significantly higher throughput for deep learning inference, scientific computing, and HPC applications where many computations can be skipped safely.

---

### 📚 Further Reading

- [AMD ROCm – rocSPARSE Documentation](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/)
- [ROCm Tensor Operations (rocWMMA)](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/)
- [AMD CDNA Architecture Whitepaper](https://www.amd.com/en/technologies/cdna.html)

---
[← Kernel Optimization](kernel_optimization.md)


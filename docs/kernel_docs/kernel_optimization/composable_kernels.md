# ⚙️ How to Use Composable Kernels on AMD GPUs

[← Kernel Optimization](kernel_optimization.md)

---
Composable Kernels (CK) is a high-performance, flexible, and portable library developed by AMD for implementing tensor operations — particularly matrix multiplication, convolution, and reduction — optimized for CDNA and RDNA architectures. It provides fine-grained control over GPU kernel tuning and composition, enabling developers to build domain-specific, high-efficiency operators.

---

## 🔍 Overview

AMD’s Composable Kernels framework (CK) sits between low-level HIP programming and high-level machine learning libraries like MIOpen or PyTorch. CK exposes parameterized building blocks that allow you to:

- Write custom GPU operators without starting from scratch.
- Auto-tune kernel configurations for a given device.
- Reuse pre-tuned blocks for different tensor operations.
- Achieve near hand-tuned performance while remaining maintainable.

---

## 🧩 When to Use Composable Kernels

Composable Kernels are most beneficial when:

- You need **custom tensor operations** not available in MIOpen or rocBLAS.
- You are targeting **specific GPU architectures** (e.g., MI200, MI300 series) and want to optimize for memory layout or precision (FP16, BF16, INT8).
- You are **benchmarking model components** (e.g., GEMM or convolution) and want full control of tile sizes and wavefront scheduling.
- You require **fine-tuned performance portability** across multiple GPU SKUs.

---

## ⚙️ Installation

CK is part of the ROCm ecosystem and can be built from source or installed through packages.

### Option 1: Install from Source

```bash
git clone https://github.com/ROCmSoftwarePlatform/composable_kernel.git
cd composable_kernel
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Option 2: Install via ROCm Environment (if available)

If included in your ROCm release, ensure it’s available:

```bash
ls /opt/rocm/lib/cmake/composable_kernel
```

If not found, use the source method above.

---

## 🧠 Key Concepts

### 1. **Tensor Specialization**
CK uses templates and compile-time parameters to define tensor layouts (e.g., `RowMajor`, `ColumnMajor`, `GNHWC` for convolutions). Choosing the right layout reduces address translation and improves memory coalescing.

### 2. **Block and Thread Tiling**
Tiling divides a problem into smaller chunks processed per wavefront or block. CK lets you configure tile sizes explicitly to balance shared memory and register pressure.

### 3. **Pipeline and Scheduling**
You can adjust load-store overlap and pipeline depth (LDS and global memory loads). This maximizes compute unit (CU) utilization.

### 4. **Kernel Fusion**
CK supports fusing multiple tensor operations (e.g., GEMM + activation + normalization) into a single kernel launch, reducing memory transfers and launch latency.

---

## 💡 Example: GEMM (Matrix Multiplication)

Here’s a minimal example of launching a CK GEMM kernel:

```bash
# Example command to run a GEMM kernel
./client_example_gemm   --num_gemm 1   --M 1024 --N 1024 --K 1024   --layout A_MK B_KN C_MN   --data_type fp16 fp16 fp16   --init_method 1   --verify 1   --time 1
```

This tests a single GEMM with FP16 precision and verifies results.  
To optimize, you can adjust block sizes and vectorization options in the source template.

---

## 🚀 Performance Optimization Tips

| Optimization Area | Description |
|--------------------|-------------|
| **Tile Size Selection** | Match block tile sizes to LDS capacity (e.g., 64 KB per CU on CDNA2). |
| **Wavefront Utilization** | Prefer full 64-thread wavefronts for compute-intensive operations. |
| **Data Layout Alignment** | Align memory to 128-bit boundaries for vectorized loads/stores. |
| **Pipeline Stages** | Tune prefetch depth to hide global memory latency. |
| **Precision Format** | Use FP16/BF16 for higher throughput on matrix cores (MFMA). |

---

## 🔬 Integration with HIP and PyTorch

Composable Kernels can be called directly from HIP or through PyTorch custom operators:

```python
import torch
import ck
from ck.gemm import run_gemm

A = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
B = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
C = run_gemm(A, B)
```

This allows you to embed tuned kernels in larger deep learning workflows.

---

## 🧩 Advanced Features

- **Auto-tuning framework**: CK includes a tuner that sweeps tile and vector parameters to find optimal configurations.
- **Support for Mixed Precision**: FP32 accumulation with FP16 inputs.
- **CDNA3 (MI300)** specific optimizations for matrix cores and improved LDS bandwidth.
- **Flexible API**: Composable operators can be instantiated via YAML or C++ templates.

---

## 🧰 Debugging and Profiling

Use ROCm tools to verify CK performance:

```bash
rocprof --stats ./client_example_gemm
rocminfo | grep gfx
```

You can also visualize wavefront occupancy and LDS usage using:

```bash
rocprof --hip-trace ./client_example_gemm
```

---


## 🔬 AMD vs. NVIDIA in Composable Kernel Optimization

AMD and NVIDIA GPUs differ in how they expose and manage resources for high-performance tensor operations. AMD’s CDNA and RDNA architectures provide flexible, software-managed LDS, large per-CU L1 caches, and wavefront-based execution (64 threads per wavefront), which CK leverages through template-based tensor layouts, configurable tiling, and explicit pipeline stages. In contrast, NVIDIA GPUs (Ampere, Hopper, etc.) rely on hardware-managed L1/L2 caches, smaller shared memory per SM, and warp-based execution (32 threads per warp). This means that on NVIDIA, many memory optimizations are handled automatically by the compiler and hardware, while AMD developers benefit more from manual control over tiling, block sizes, and memory layout. Consequently, CK’s design philosophy—parameterized kernels, operator fusion, and explicit tuning—is particularly effective on AMD GPUs, allowing near hand-optimized performance without assembly-level coding, whereas on NVIDIA, similar optimizations often rely more on compiler heuristics and cuBLAS/cuDNN tuned libraries

## ✅ Summary

Composable Kernels give developers **fine-grained control** over GPU execution, achieving nearly peak theoretical throughput without writing assembly. They sit between library-level and kernel-level programming, offering an ideal balance between flexibility and maintainability.

| Feature | Benefit |
|----------|----------|
| Template-based design | Reusable, compile-time efficient kernels |
| Architecture awareness | Tuned for CDNA/CDNA2/CDNA3 GPUs |
| Operator fusion | Reduces memory traffic and launch overhead |
| Performance portability | Retarget kernels with minor changes |

---

[← Kernel Optimization](kernel_optimization.md)
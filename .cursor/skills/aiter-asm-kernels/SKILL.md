---
name: aiter-asm-kernels
description: Write ASM (assembly) GPU kernels, tests, and build configurations for the aiter project. Use when creating or modifying hand-tuned assembly kernels for AMD GPUs, including HSA code objects, precompiled kernels for gfx942/gfx950, or ASM interfaces for GEMM, MHA, PA, MLA, MoE, or communication operations. Covers HSA codegen, KernelArgs structs, heuristic dispatch, and kernel config files.
---

# ASM (Assembly) Kernels in Aiter

## Project Layout

| Component | Path |
|-----------|------|
| ASM interfaces | `csrc/py_itfs_cu/asm_*.cu` |
| HSA code objects | `hsa/gfx942/`, `hsa/gfx950/` |
| HSA codegen script | `hsa/codegen.py` |
| Config headers | `csrc/include/asm_*_configs.hpp` |
| C++ interfaces | `csrc/cpp_itfs/` |
| Pybind | `csrc/pybind/` |
| Build config | `aiter/jit/optCompilerConfig.json` |
| Python ops | `aiter/ops/*.py` |

### ASM Kernel Types

| Type | Interface File | HSA Directory | Config Header |
|------|---------------|---------------|---------------|
| GEMM A8W8 (INT8) | `asm_gemm_a8w8.cu` | `hsa/*/i8gemm/` | `asm_i8gemm_configs.hpp` |
| GEMM A16W16 (BF16) | `asm_gemm_a16w16.cu` | `hsa/*/bf16gemm/` | `asm_bf16gemm_configs.hpp` |
| GEMM A4W4 | `asm_gemm_a4w4.cu` | `hsa/*/f4gemm/` | `asm_f4gemm_configs.hpp` |
| Blockscale GEMM | `asm_flatmm_a8w8_blockscale.cu` | `hsa/*/fp8gemm_blockscale/` | configs in header |
| MHA Forward | `asm_mha_fwd.cu` | `hsa/*/fmha_v3_fwd/` | `asm_fmha_v3_fwd_configs.hpp` |
| MHA Backward | `asm_mha_bwd.cu` | `hsa/*/fmha_v3_bwd/` | `asm_fmha_v3_bwd_configs.hpp` |
| Paged Attention | `asm_pa.cu` | `hsa/*/pa/` | `asm_pa_configs.hpp` |
| MLA | `asm_mla.cu` | `hsa/*/mla/` | `asm_mla_configs.hpp` |
| Fused MoE | `asm_fmoe.cu` | `hsa/*/fmoe/` | `asm_fmoe_configs.hpp` |
| MoE 2-stage | `asm_moe_2stage.cu` | `hsa/*/fmoe_2stages/` | configs in header |
| Top-K Softmax | `asm_topksoftmax.cu` | `hsa/*/topksoftmax/` | `asm_topksoftmax_configs.hpp` |
| LayerNorm | `asm_layernorm.cu` | - | - |
| Communication | `asm_communication.cu` | `hsa/*/all_reduce.co` | - |

## ASM Kernel Architecture

ASM kernels are **precompiled GPU assembly** loaded as HSA code objects (`.co` files). The C++ interface handles:

1. **KernelArgs struct** - Packed argument layout matching the assembly kernel's expectation
2. **Config tables** - Map (arch, dtype, shape) → kernel name
3. **Heuristic dispatch** - Select best kernel based on problem dimensions
4. **AiterAsmKernel** - Utility class for loading and launching `.co` blobs

## Writing an ASM Kernel Interface

### Step 1: Define KernelArgs Struct

The struct must match the assembly kernel's argument layout exactly:

```cpp
// csrc/py_itfs_cu/asm_your_kernel.cu
#include "aiter_asm_kernel.h"

struct KernelArgs {
    // Pointers (must match assembly kernel's argument order)
    void* ptr_c;      // Output
    void* ptr_a;      // Input A
    void* ptr_b;      // Input B
    void* ptr_sa;     // Scale A
    void* ptr_sb;     // Scale B
    void* ptr_bias;   // Bias
    
    // Dimensions
    uint32_t m;
    uint32_t n;
    uint32_t k;
    
    // Leading dimensions / strides
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    
    // Kernel-specific params
    uint32_t ks;       // splitK
} __attribute__((packed));
```

### Step 2: Define Config Table

```cpp
// In header or inline
struct KernelConfig {
    std::string kernel_name;
    int arch_id;       // e.g., 942 for gfx942, 950 for gfx950
    int tile_m;
    int tile_n;
    int tile_k;
    int splitK;
    // ... other config fields
};

// Config table (from asm_*_configs.hpp)
static const std::vector<KernelConfig> KERNEL_CONFIGS = {
    {"i8gemm_256x256x128_bf16_perTokenI8", 942, 256, 256, 128, 0},
    {"i8gemm_128x128x64_bf16_perTokenI8",  942, 128, 128, 64,  0},
    {"i8gemm_256x128x128_bf16_perTokenI8", 950, 256, 128, 128, 0},
    // ... more configs
};
```

### Step 3: Implement Heuristic Dispatch

```cpp
std::string get_heuristic_kernel(
    int arch_id,
    const std::string& dtype_str,
    int M, int N, int K,
    int splitK = 0)
{
    // Filter configs by arch and dtype
    std::vector<KernelConfig> candidates;
    for (auto& cfg : KERNEL_CONFIGS) {
        if (cfg.arch_id == arch_id && cfg.dtype_match(dtype_str))
            candidates.push_back(cfg);
    }
    
    // Score each candidate
    // Criteria: tile utilization, compute/memory ratio, splitK benefit
    int best_score = -1;
    std::string best_kernel;
    for (auto& cfg : candidates) {
        int tiles_m = (M + cfg.tile_m - 1) / cfg.tile_m;
        int tiles_n = (N + cfg.tile_n - 1) / cfg.tile_n;
        int utilization = tiles_m * tiles_n;
        // ... scoring logic ...
        if (score > best_score) {
            best_score = score;
            best_kernel = cfg.kernel_name;
        }
    }
    return best_kernel;
}
```

### Step 4: Implement the Entrypoint

```cpp
void your_kernel_asm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor A_scale,
    torch::Tensor B_scale,
    torch::Tensor out,
    const std::string& kernelName,  // Empty for heuristic
    torch::Tensor bias,
    int splitK)
{
    int M = A.size(0), K = A.size(1), N = B.size(0);
    
    // Select kernel
    std::string kernel = kernelName;
    if (kernel.empty()) {
        kernel = get_heuristic_kernel(get_arch_id(), get_dtype_str(A), M, N, K, splitK);
    }
    
    // Build KernelArgs
    KernelArgs args;
    args.ptr_c = out.data_ptr();
    args.ptr_a = A.data_ptr();
    args.ptr_b = B.data_ptr();
    args.ptr_sa = A_scale.data_ptr();
    args.ptr_sb = B_scale.data_ptr();
    args.ptr_bias = bias.numel() > 0 ? bias.data_ptr() : nullptr;
    args.m = M;
    args.n = N;
    args.k = K;
    args.lda = A.stride(0);
    args.ldb = B.stride(0);
    args.ldc = out.stride(0);
    args.ks = splitK;
    
    // Launch
    AiterAsmKernel asm_kernel(kernel);
    int grid_x = (M + tile_m - 1) / tile_m;
    int grid_y = (N + tile_n - 1) / tile_n;
    int grid_z = splitK > 0 ? splitK : 1;
    asm_kernel.launch(grid_x, grid_y, grid_z, &args, sizeof(args));
}
```

### Step 5: HSA Codegen

Register in `hsa/codegen.py`:

```python
# hsa/codegen.py
# Add a new module type
if args.module == "your_kernel":
    configs = [
        {"name": "your_kernel_256x256", "tile_m": 256, "tile_n": 256, ...},
        {"name": "your_kernel_128x128", "tile_m": 128, "tile_n": 128, ...},
    ]
    for cfg in configs:
        generate_asm(cfg, args.output_dir, args.arch)
```

### Step 6: Build Config

```json
{
    "module_your_kernel_asm": {
        "srcs": [
            "{AITER_CSRC_DIR}/pybind/your_kernel_asm_pybind.cu",
            "{AITER_CSRC_DIR}/py_itfs_cu/asm_your_kernel.cu"
        ],
        "extra_include": ["{AITER_CSRC_DIR}/include/"],
        "flags_extra_hip": ["-O3"],
        "blob_gen_cmd": "hsa/codegen.py -m your_kernel --output_dir {}"
    }
}
```

## ASM Paged Attention Pattern

PA has special argument structs for standard vs persistent-split variants:

```cpp
// Standard PA args
struct KernelArgs {
    void* ptr_o;       // Output
    void* ptr_q;       // Query
    void* ptr_k;       // Key cache
    void* ptr_v;       // Value cache
    void* block_tables; // Page table
    void* context_lens; // Sequence lengths
    void* k_qscale;    // Key quantization scale
    void* v_qscale;    // Value quantization scale
    float sclg2e;      // sm_scale * log2(e)
    uint32_t mblk;     // Max blocks per sequence
    uint32_t kv_nheads;
    uint32_t Qs, Bs, KVs; // Strides
    uint32_t GQA;      // nheads_q / nheads_k
};

// Persistent split (PS) variant adds:
struct PsKernelArgs : KernelArgs {
    void* kv_indices;
    void* work_meta_data;
    void* split_out;
    void* split_lse;
};
```

## ASM MHA (FMHA v3) Pattern

```cpp
// Uses mha_fwd_args struct (shared with CK)
mha_fwd_args get_asm_fmha_fwd_args(
    torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
    torch::Tensor& out, torch::Tensor& softmax_lse, ...);

void fmha_v3_fwd(
    torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
    torch::Tensor& out, torch::Tensor& softmax_lse,
    float p_dropout, float softmax_scale,
    bool is_causal, int window_size_left, int window_size_right,
    bool return_softmax_lse, ...);
```

The FMHA v3 ASM kernel configs map `(head_dim, dtype, causal, mask_type)` to kernel names.

## Testing ASM Kernels

ASM kernels are tested through the same Python test infrastructure:

```python
import aiter

def test_asm_gemm():
    M, N, K = 128, 1024, 512
    # ASM path is selected when tuned config has libtype=asm
    # Or called directly via aiter.gemm_a8w8_asm(...)
    result = aiter.gemm_a8w8(x, w, x_scale, w_scale, Y, bias)
    torch.testing.assert_close(ref, result, atol=0.02, rtol=1e-2)
```

## HSA Code Object Directory Structure

```
hsa/
├── codegen.py          # Master codegen script
├── gfx942/
│   ├── bf16gemm/       # BF16 GEMM .co files
│   ├── i8gemm/         # INT8 GEMM .co files
│   ├── f4gemm/         # FP4 GEMM .co files
│   ├── fp8gemm_blockscale/
│   ├── fmha_v3_fwd/    # Flash attention forward .co files
│   ├── fmha_v3_bwd/    # Flash attention backward .co files
│   ├── pa/             # Paged attention .co files
│   ├── mla/            # MLA .co files
│   ├── fmoe/           # Fused MoE .co files
│   ├── fmoe_2stages/   # 2-stage MoE .co files
│   ├── topksoftmax/
│   └── all_reduce.co
└── gfx950/
    └── ... (same structure)
```

## Prerequisites

Before writing ASM kernel interfaces, read these foundational skills:
- [HIP Kernel Programming](../hip-kernel-programming/SKILL.md) - HIP language, PyBind patterns
- [AMD GPU Architecture](../amd-gpu-architecture/SKILL.md) - CDNA3/4 ISA, MFMA instructions, gfx942/gfx950

## Key Design Notes

- **KernelArgs must be packed:** Use `__attribute__((packed))` - assembly kernels read args at fixed byte offsets
- **Arch-specific kernels:** Different `.co` files per GPU architecture (gfx942, gfx950)
- **Heuristic selection is critical:** Assembly kernels have fixed tile sizes; poor tile utilization wastes compute
- **AiterAsmKernel class:** Handles HSA runtime loading, kernel dispatch, and memory management
- **Fallback to CK/Triton:** If no ASM kernel matches, the Python layer falls back to CK or Triton implementation

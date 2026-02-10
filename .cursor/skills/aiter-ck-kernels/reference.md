# Composable Kernel (CK) Reference

## CK Template Parameters

Key template parameters for `DeviceGemmMultiD_Xdl_CShuffle_V3`:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| MPerBlock | M tile size | 64, 128, 256 |
| NPerBlock | N tile size | 64, 128, 256 |
| KPerBlock | K tile size | 32, 64, 128 |
| AK1 | A vectorization | 16, 32, 64 |
| MPerXdl | M per MFMA instruction | 16, 32 |
| NPerXdl | N per MFMA instruction | 16, 32 |
| MRepeat | M repetitions per thread | 1, 2, 4 |
| NRepeat | N repetitions per thread | 1, 2, 4 |

## Kernel Dispatch Hierarchy

1. **Tuned lookup:** CSV maps `(cu_num, M, N, K, dtype)` → kernel name + splitK
2. **Heuristic dispatch:** Shape ranges → kernel selection
3. **Padding:** M padded to power-of-2 bucket for better cache utilization

```cpp
// Heuristic example
if (M <= 16) return kernel_256x16x64x512;
if (M <= 64 && N >= 4096) return kernel_128x128x64;
if (M >= 256) return kernel_256x256x128;
return kernel_default;
```

## optCompilerConfig.json Structure

Each module entry:

```json
{
    "module_name": {
        "srcs": ["list of .cu source files"],
        "extra_include": ["include paths"],
        "flags_extra_cc": ["C++ compiler flags"],
        "flags_extra_hip": ["HIP compiler flags"],
        "blob_gen_cmd": "pre-compilation script (optional)"
    }
}
```

The `{AITER_CSRC_DIR}` and `{CK_DIR}` variables are expanded at build time.

## CK MoE 2-Stage Pattern

Two-stage MoE in CK:

```cpp
// Stage 1: gate + up projection
// csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu
// Uses fused expert-grouped GEMM with SiLU activation

// Stage 2: down projection
// Same kernel type but without activation
```

Config in `optCompilerConfig.json`:
```json
"module_moe_ck2stages": {
    "srcs": ["pybind/moe_ck_2stages_pybind.cu", "ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu"],
    "blob_gen_cmd": "ck_gemm_moe_2stages_codegen/gen_instances.py ..."
}
```

## CK Tile vs Classic CK

- **Classic CK:** `DeviceGemm*` templates, `ck::tensor_operation::device` namespace
- **CK Tile:** `ck_tile::*` namespace, more modular, newer API
  - Used for FMHA (flash attention), newer GEMM variants
  - `cktile_gemm_a8w8_bpreshuffle/` uses CK Tile

## Compile_ops Decorator

```python
@compile_ops("module_name", fc_name="function_name", gen_func=None, gen_fake=None)
def my_op(arg1, arg2, ...):
    ...  # Body replaced by compiled extension
```

Parameters:
- `module_name`: Key in `optCompilerConfig.json`
- `fc_name`: C++ function name to bind
- `gen_func`: Optional codegen function for blob_gen_cmd
- `gen_fake`: Returns fake output tensors for `torch.compile` tracing

## C++ Header Conventions

```cpp
// csrc/include/your_kernel.h
#pragma once
#include <torch/extension.h>

void your_kernel_fwd(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int param1 = 0);
```

## Pybind Macro Pattern

`csrc/include/rocm_ops.hpp`:

```cpp
#define YOUR_KERNEL_PYBIND \
    m.def("your_kernel_fwd", &your_kernel_fwd, "Your kernel forward", \
          py::arg("input"), py::arg("weight"), py::arg("output"), \
          py::arg("param1") = 0);
```

## GEMM Tuning with Gradlib

`gradlib/` provides GEMM autotuning:

```python
# gradlib/gradlib/gemm_tuner.py
# Profiles all kernel configurations for a given (M, N, K, dtype)
# Outputs best kernel to CSV config file
# Usage: python -m gradlib.gemm_tuner --M 128 --N 4096 --K 8192 --dtype int8
```

## Existing CK Module List

| Module | Kernel Type |
|--------|-------------|
| `module_gemm_a8w8` | CK INT8 GEMM |
| `module_gemm_a8w8_blockscale` | CK blockscale INT8 GEMM |
| `module_gemm_a4w4_blockscale` | CK 4-bit GEMM |
| `module_batched_gemm_bf16` | CK batched BF16 GEMM |
| `module_batched_gemm_a8w8` | CK batched INT8 GEMM |
| `module_deepgemm` | CK DeepGEMM |
| `module_mha_fwd/bwd` | CK Tile FMHA |
| `module_mha_batch_prefill` | CK batch prefill attention |
| `module_moe_ck2stages` | CK 2-stage MoE |
| `module_rmsnorm` | CK RMSNorm |
| `module_norm` | CK LayerNorm |

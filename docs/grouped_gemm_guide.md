# AITER Grouped GEMM Operators Guide

This guide documents the Grouped Matrix Multiply (GMM) and DeepGEMM operators in AITER, designed for Mixture-of-Experts (MoE) and variable-length grouped computations.

---

## Quick Reference

| Use Case | Recommended Operation | Backend | Why |
|----------|---------------------|---------|-----|
| **MoE expert GEMM (forward)** | `gmm` | Triton | Token-to-expert routing, bias support |
| **MoE expert GEMM (backward)** | `ptgmm` / `nptgmm` | Triton | Transposed GMM, bias gradient |
| **Variable-length attention GEMM** | `deepgemm` | CK-tile | Dynamic masking, FP8 quantized |

---

## 1. GMM (Grouped Matrix Multiply) — Triton

### `gmm` — Forward Pass

Each group of rows in `lhs` is multiplied with the corresponding weight plane in `rhs`:

```python
from aiter.ops.triton.gmm import gmm

out = gmm(
    lhs,            # (M, K) — input activations
    rhs,            # (G, K, N) — per-group weights
    group_sizes,    # (G,) int32 — rows per group, must sum to M
    preferred_element_type=torch.bfloat16,
    bias=bias,      # (G, N) — optional per-group bias
)
# out: (M, N)
```

**Operation:** `out[start:end, :] = lhs[start:end, :] @ rhs[g] + bias[g]` for each group `g`.

### `ptgmm` / `nptgmm` — Transposed GMM (Backward)

Computes transposed grouped GEMM for weight gradient computation:

```python
from aiter.ops.triton.gmm import ptgmm, nptgmm

out = ptgmm(
    lhs,            # (K, M) — transposed input
    rhs,            # (M, N) — gradient from downstream
    group_sizes,    # (G,) int32
    bias_grad=bg,   # (G, K) float32 — optional bias gradient output
    accumulate=False, # accumulate into existing output
)
# out: (G, K, N)
```

**Operation:** `out[g] = lhs[:, start:end] @ rhs[start:end, :]` for each group `g`.

`ptgmm` uses a persistent kernel; `nptgmm` uses a non-persistent variant. Benchmark both to choose.

### Supported Data Types

| Input | Output | Group sizes | Bias gradient |
|-------|--------|-------------|---------------|
| float16, bfloat16 | float16, bfloat16 | int32 | float32 |

### Layout Requirements

| Variant | `lhs` stride | `rhs` stride |
|---------|-------------|-------------|
| `gmm` | row-major `(K, 1)` | row-major `(K*N, N, 1)` or col-major `(K*N, 1, K)` |
| `ptgmm`/`nptgmm` | row-major `(M, 1)` or col-major `(1, K)` | row-major `(N, 1)` |

### Kernel Configuration

Tuning configs stored in `aiter/ops/triton/configs/{arch}-GMM.json`:

```json
{
  "gmm": {
    "default": {
      "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 64,
      "BLOCK_SIZE_N": 64, "GROUP_SIZE": 8, "GRID_DIM": 256
    }
  }
}
```

Auto-queried by `get_config(gmm_type, M, K, N, G)` with LRU caching.

### Performance Notes

- **Persistent kernel:** Processes multiple tiles across groups in a single program launch
- **XCD remapping:** Improves L2 cache locality on multi-XCD GPUs (MI300)
- **IEEE precision:** Uses `tl.dot(..., input_precision="ieee")` for numerical stability

---

## 2. DeepGEMM — CK-tile

Grouped GEMM with dynamic per-group masking, optimized for variable-length attention workloads.

```python
import aiter
from aiter import deepgemm
from aiter.ops.shuffle import shuffle_weight

# Weights must be pre-shuffled
weight_shuffled = shuffle_weight(weight_q, layout=(16, 16))

out = deepgemm(
    XQ,             # (num_groups, max_m, k) — input activations
    weight_shuffled, # (num_groups, n, k) — shuffled weights
    Y,              # (num_groups, max_m, n) — preallocated output
    group_layout,   # (num_groups,) int32 — active rows per group
    x_scale=None,   # per-token activation scale (for FP8)
    w_scale=None,   # per-group weight scale (for FP8)
)
```

**Operation:** `Y[j, :masked_m[j], :] = XQ[j, :masked_m[j], :] @ WQ[j].T` for each group `j`.

### Supported Data Types

| Input/Weight | Output | Scale | Notes |
|-------------|--------|-------|-------|
| float16, bfloat16 | Same as input | None | No quantization |
| float8_e4m3fnuz | float16 or bfloat16 | Per-token/per-group | FP8 quantized |

### Kernel Selection

| M Range | Kernel | Tile Size |
|---------|--------|-----------|
| M < 128 | `deepgemm_256x32x64x256` | Small M tile |
| M >= 128 | `deepgemm_256x128x128x128` | Large M tile |

### Requirements

- **Hardware:** GFX942 (MI300 series) required
- **Weight layout:** Must be pre-shuffled with `shuffle_weight(w, layout=(16, 16))`
- **Output:** Pre-allocated, modified in-place

---

## 3. GMM vs DeepGEMM

| Feature | GMM (Triton) | DeepGEMM (CK) |
|---------|:------------:|:--------------:|
| **Backend** | Triton | CK-tile (C++/HIP) |
| **Primary use** | MoE expert routing | Variable-length attention |
| **Input shape** | `(M, K) @ (G, K, N)` | `(G, max_M, K) @ (G, N, K)^T` |
| **Output shape** | `(M, N)` | `(G, max_M, N)` |
| **Quantization** | No | Yes (FP8) |
| **Bias support** | Yes | No |
| **Dynamic masking** | No (exact group_sizes) | Yes (masked_m per group) |
| **Weight layout** | Standard | Shuffled (16x16 blocks) |
| **Hardware** | All AMD GPUs via Triton | GFX942/GFX950 |

---

## 4. Source Files

| Component | Path |
|---|---|
| GMM Python API | `aiter/ops/triton/gmm.py` |
| GMM Triton kernels | `aiter/ops/triton/_triton_kernels/gmm.py` |
| GMM utilities | `aiter/ops/triton/utils/gmm_common.py` |
| GMM tuning configs | `aiter/ops/triton/configs/{arch}-GMM.json` |
| DeepGEMM Python API | `aiter/ops/deepgemm.py` |
| DeepGEMM CK kernels | `csrc/ck_deepgemm/deepgemm.cu` |
| DeepGEMM kernel configs | `csrc/ck_deepgemm/deepgemm_common.py` |
| Weight shuffle | `aiter/ops/shuffle.py` |

---

## 5. Test Files

| Test | Path |
|------|------|
| GMM unit tests | `op_tests/triton_tests/test_gmm.py` |
| GMM benchmarks | `op_tests/op_benchmarks/triton/bench_gmm.py` |
| DeepGEMM tests | `op_tests/test_deepgemm.py` |
| DeepGEMM attention benchmarks | `op_tests/op_benchmarks/triton/bench_deepgemm_attention.py` |

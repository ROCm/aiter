# AITER GEMM Variants & Tuning Guide

This guide documents all GEMM variants available in AITER, their backend support, tuning system, and how to choose the right one for your use case.

---

## Quick Reference: Which GEMM Should I Use?

| Use Case | Recommended Variant | Backend | Why |
|----------|-------------------|---------|-----|
| **FP8 inference (production)** | `gemm_a8w8` | ASM or CK | Pre-tuned for MI300X/MI350, best throughput |
| **FP8 with block-scale (high accuracy)** | `gemm_a8w8_blockscale` | CK/ASM | Per-128x128 block scales reduce quant error |
| **BF16 inference (no quantization)** | `gemm_a16w16` (via `tuned_gemm`) | ASM/HipBLASLt | Auto-selects best backend per shape |
| **FP4 inference (max compression)** | `gemm_a4w4` | ASM/CK | 4x weight compression; MI350 native FP4 |
| **Batched GEMM (MLA projections)** | `batched_gemm_a8w8` / `batched_gemm_bf16` | CK | Batched (B, M, K) x (B, N, K) |
| **Group-wise GEMM** | `deepgemm` | CK | Variable M per group |
| **FFN blocks (fused gate+up)** | `ff_a16w16_gated` | Triton | SwiGLU/GELU fused in one kernel |
| **Fused GEMM + element-wise** | `fused_gemm_*_mul_add` | Triton | GEMM + a*x+b in one launch |
| **Prototyping / custom shapes** | Triton GEMM variants | Triton | Portable, configurable, easy to tune |

---

## 1. GEMM Architecture in AITER

AITER provides GEMM operators across three backend tiers:

```
                    ┌──────────────────────────────────────────┐
  User API          │  gemm_a8w8()  gemm_a16w16()  gemm_a4w4()│
                    └──────────┬───────────────────────────────┘
                               │ auto-dispatch
                    ┌──────────▼───────────────────────────────┐
  Backend Selection │  Tuned CSV lookup → kernel + splitK      │
                    │  Shape padding → fallback selection       │
                    └──┬──────────┬──────────┬─────────────────┘
                       │          │          │
                 ┌─────▼──┐ ┌────▼───┐ ┌────▼────┐
  Backends       │  ASM   │ │   CK   │ │ Triton  │
                 │(MI300X)│ │(CKTile)│ │(portable)│
                 └────────┘ └────────┘ └─────────┘
```

### Key Concepts

- **CK (Composable Kernel)**: AMD's template-based GPU kernel library. Variants include CK (classic) and CKTile (newer tile-based).
- **ASM**: Hand-tuned assembly kernels for specific GPU architectures. Best performance but fixed configurations.
- **Triton**: Python-based JIT-compiled kernels. Portable and easy to customize.
- **SplitK**: Splits the K (reduction) dimension across multiple thread groups for better parallelism on small-M problems.
- **Bpreshuffle**: Weight pre-shuffling for optimized memory access patterns. Requires one-time weight transformation.
- **Block-scale**: Per-block quantization scales (e.g., 128x128 blocks) for higher accuracy than per-tensor scaling.

### Tensor Shape Conventions

```
Input A:    (M, K)           # activations (row-major)
Input B:    (N, K)           # weights (pre-transposed)
Output:     (M, N)           # result (row-major)
Scales:     varies by variant (per-tensor, per-row, per-block)
```

For batched variants: `(B, M, K) x (B, N, K) → (B, M, N)`

---

## 2. A8W8 — INT8/FP8 Activations x INT8/FP8 Weights

The primary quantized GEMM for inference. Supports per-tensor and per-block scaling.

### Backend Support

| Feature | CK | CKTile | ASM | Triton |
|---------|:---:|:---:|:---:|:---:|
| **Per-tensor scale** | Yes | - | Yes | Yes |
| **Block-scale (128x128)** | Yes | Yes | Yes | Yes |
| **Bpreshuffle** | Yes | Yes | Yes | - |
| **Block-scale + bpreshuffle** | Yes | - | Yes | - |
| **SplitK** | Yes | - | Yes | Yes |
| **Bias** | Yes | - | Yes | Yes |
| **FP8 (E4M3)** | Yes | Yes | Yes | Yes |
| **INT8** | Yes | - | Yes | Yes |
| **GFX942 (MI300X)** | Yes | Yes | Yes | Yes |
| **GFX950 (MI350)** | Yes | Yes | Yes | Yes |

### Key API Functions

```python
from aiter import gemm_a8w8, gemm_a8w8_blockscale

# Standard per-tensor A8W8
output = gemm_a8w8(
    XQ,             # (M, K) INT8/FP8 activations
    WQ,             # (N, K) INT8/FP8 weights
    x_scale,        # per-token or per-tensor activation scale
    w_scale,        # per-tensor weight scale
    bias=None,      # optional (N,) bias
    dtype=torch.bfloat16,  # output dtype
    splitK=None,    # auto-select
)

# Block-scale A8W8 (higher accuracy)
output = gemm_a8w8_blockscale(
    XQ,             # (M, K) INT8/FP8 activations
    WQ,             # (N, K) INT8/FP8 weights
    x_scale,        # (M, ceil(K/128)) block scales
    w_scale,        # (N, ceil(K/128)) block scales
    dtype=torch.bfloat16,
    isBpreshuffled=False,
)
```

### Pre-Shuffle Optimization

```python
from aiter import gemm_a8w8_bpreshuffle

# Requires one-time weight preparation (layout=(32,16) shuffle)
# Then use for repeated inference:
output = gemm_a8w8_bpreshuffle(
    XQ, WQ_shuffled, x_scale, w_scale,
    dtype=torch.bfloat16,
)
```

### Tuning

**Config files** (auto-loaded, override via environment):

| Config | Entries | Env Override |
|--------|---------|-------------|
| `aiter/configs/a8w8_tuned_gemm.csv` | 551 | `AITER_CONFIG_GEMM_A8W8` |
| `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` | 702 | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
| `aiter/configs/a8w8_blockscale_tuned_gemm.csv` | 79 | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` |
| `aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | 118 | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |

### When to Use

- FP8/INT8 quantized inference (majority of production LLM deployments)
- Block-scale variant when accuracy matters more than raw throughput
- Bpreshuffle when weights are static and inference is latency-critical

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/gemm_op_a8w8.py` | All A8W8 variants and tuning interfaces |
| `aiter/ops/triton/gemm/basic/gemm_a8w8.py` | Triton A8W8 |
| `aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py` | Triton block-scale |
| `aiter/ops/triton/gemm/basic/gemm_a8w8_per_token_scale.py` | Triton per-token |

---

## 3. A16W16 — BF16/FP16 Full-Precision GEMM

Full-precision GEMM with automatic backend selection. The recommended high-level entry point is `tuned_gemm.gemm_a16w16`.

### Backend Support

| Feature | ASM | HipBLASLt | CK | Triton | PyTorch |
|---------|:---:|:---:|:---:|:---:|:---:|
| **BF16** | Yes | Yes | Yes | Yes | Yes |
| **FP16** | Yes | Yes | Yes | Yes | Yes |
| **Bias** | Yes | Yes | - | Yes | Yes |
| **SplitK** | Yes | - | - | Yes | - |
| **Bpreshuffle** | Yes | Yes | - | - | - |
| **Output scaling** | - | - | - | - | Yes (`torch._scaled_mm`) |

### Key API Functions

```python
from aiter.tuned_gemm import gemm_a16w16

# High-level (auto-selects best backend)
output = gemm_a16w16(
    A,              # (M, K) BF16/FP16
    B,              # (N, K) BF16/FP16
    bias=None,      # optional (N,)
    otype=None,     # output dtype (default: same as input)
    scale_a=None,   # optional per-tensor scale
    scale_b=None,
    scale_c=None,
)
```

### Automatic Backend Selection

```
Bpreshuffle weights? → HipBLASLt
BF16 + N%64==0 + K%64==0? → ASM
M ≤ 4 + K ≤ 9216? → Skinny kernel
Else → PyTorch fallback
```

### Tuning

| Config | Env Override |
|--------|-------------|
| `aiter/configs/bf16_tuned_gemm.csv` | `AITER_CONFIG_GEMM_BF16` |

### When to Use

- Training workloads (no quantization)
- Inference when accuracy is paramount
- As fallback when quantized models aren't available

### Source Files

| File | Purpose |
|------|---------|
| `aiter/tuned_gemm.py` | High-level `gemm_a16w16` with auto-dispatch |
| `aiter/ops/gemm_op_a16w16.py` | Low-level ASM interface |
| `aiter/ops/triton/gemm/basic/gemm_a16w16.py` | Triton BF16 GEMM |

---

## 4. A4W4 — FP4 Activations x FP4 Weights

Maximum weight compression using MXFP4 (Microscaling FP4) format with E8M0 block scales.

### Backend Support

| Feature | ASM | CK |
|---------|:---:|:---:|
| **MXFP4 (E2M1 packed)** | Yes | Yes |
| **Block-scale (per 32 elements)** | Yes | Yes |
| **Bpreshuffle** | Yes | - |
| **Alpha/beta scaling** | Yes | Yes |
| **GFX942 (MI300X)** | - | - |
| **GFX950 (MI350)** | Yes | Yes |

**Note:** A4W4 is **not supported on GFX942 (MI300X)**. It requires GFX950's native FP4 hardware.

### Key API Functions

```python
from aiter import gemm_a4w4

output = gemm_a4w4(
    A,              # (M, K//2) FP4 packed activations
    B,              # (N, K//2) FP4 packed weights
    A_scale,        # (M, K//32) E8M0 block scales
    B_scale,        # (N, K//32) E8M0 block scales
    bias=None,
    dtype=torch.bfloat16,
    alpha=1.0,
    beta=0.0,
    bpreshuffle=True,
)
```

### Tuning

| Config | Entries | Env Override |
|--------|---------|-------------|
| `aiter/configs/a4w4_blockscale_tuned_gemm.csv` | 925 | `AITER_CONFIG_GEMM_A4W4` |

### When to Use

- MI350 (GFX950) deployments with native FP4 support
- When 4x weight compression is needed
- Models pre-quantized to MXFP4 format

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/gemm_op_a4w4.py` | A4W4 variants |
| `aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py` | Triton FP4xFP4 |
| `aiter/ops/triton/gemm/basic/gemm_a16wfp4.py` | Triton BF16xFP4 |
| `aiter/ops/triton/gemm/basic/gemm_a8wfp4.py` | Triton FP8xFP4 |

---

## 5. Batched GEMM

Batched matrix multiplication for operations like MLA latent projections where multiple independent GEMMs run in parallel.

### Variants

| Variant | Function | Data Types | Config |
|---------|----------|-----------|--------|
| **Batched A8W8** | `batched_gemm_a8w8_CK()` | INT8/FP8 | `a8w8_tuned_batched_gemm.csv` |
| **Batched BF16** | `batched_gemm_bf16_CK()` | BF16/FP16 | `bf16_tuned_batched_gemm.csv` |

### Key API Functions

```python
from aiter import batched_gemm_a8w8_CK, batched_gemm_bf16_CK

# Batched INT8 GEMM
output = batched_gemm_a8w8_CK(
    XQ,             # (B, M, K) INT8
    WQ,             # (B, N, K) INT8
    x_scale,        # (B, 1, 1) or broadcast
    w_scale,        # (B, 1, 1) or broadcast
    dtype=torch.bfloat16,
)

# Batched BF16 GEMM
output = batched_gemm_bf16_CK(
    X,              # (B, M, K) BF16
    W,              # (B, N, K) BF16
    dtype=torch.bfloat16,
)
```

### Triton Batched Variants

| Function | File | Data Types |
|----------|------|-----------|
| `batched_gemm_bf16` | `triton/gemm/batched/batched_gemm_bf16.py` | BF16 |
| `batched_gemm_a8w8` | `triton/gemm/batched/batched_gemm_a8w8.py` | INT8/FP8 |
| `batched_gemm_afp4wfp4` | `triton/gemm/batched/batched_gemm_afp4wfp4.py` | FP4 |
| `batched_gemm_a16wfp4` | `triton/gemm/batched/batched_gemm_a16wfp4.py` | BF16 x FP4 |

### When to Use

- MLA latent projections (B = num_heads)
- Any workload with independent parallel GEMMs sharing the same shape
- Multi-head operations that can't be reshaped into a single large GEMM

---

## 6. DeepGEMM — Group-wise GEMM

Group-wise matrix multiplication where each group can have a different effective M dimension.

### Key API Functions

```python
from aiter import deepgemm

output = deepgemm(
    XQ,             # (num_groups, max_m, K) quantized activations
    WQ,             # (num_groups, N, K) quantized weights
    Y,              # (num_groups, max_m, N) output buffer
    group_layout,   # (num_groups,) actual M per group
    x_scale=None,   # optional quantization scales
    w_scale=None,
)
```

### When to Use

- Variable-size group operations (e.g., MOE with different token counts per expert)
- GFX942 only (checked at runtime)

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/deepgemm.py` | DeepGEMM wrapper |

---

## 7. Triton Feed-Forward Fusions

Fused two-GEMM feed-forward blocks that combine the up-projection and down-projection with activation in fewer kernel launches.

### Variants

| Function | Description | Activation |
|----------|-------------|-----------|
| `ff_a16w16_nogate` | FFN: `down(act(up(x)))` | SiLU, GELU |
| `ff_a16w16_gated` | Gated FFN: `down(act(gate(x)) * up(x))` | SwiGLU |
| `ff_a16w16_fused_gated` | Fully fused gated variant | SwiGLU |
| `ff_a16w16_fused_ungated` | Fully fused ungated variant | SiLU, GELU |

### Key API Functions

```python
from aiter.ops.triton.gemm.feed_forward.ff_a16w16 import ff_a16w16_gated

# Fused gated FFN (SwiGLU style)
output = ff_a16w16_gated(
    x,              # (M, K) input
    w_gate,         # (inter_dim, K) gate weights
    w_up,           # (inter_dim, K) up-projection weights
    w_down,         # (K, inter_dim) down-projection weights
    dtype=torch.bfloat16,
    activation="silu",
)
```

### When to Use

- Transformer FFN blocks (MLP layers)
- When kernel launch overhead matters (small batch sizes)
- Gated architectures (LLaMA, Mistral, DeepSeek)

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/triton/gemm/feed_forward/ff_a16w16.py` | Gated/ungated FFN |
| `aiter/ops/triton/gemm/feed_forward/ff_a16w16_fused_gated.py` | Fully fused gated |
| `aiter/ops/triton/gemm/feed_forward/ff_a16w16_fused_ungated.py` | Fully fused ungated |

---

## 8. Triton Fused GEMM Operations

Combine GEMM with subsequent element-wise operations in a single kernel.

### Variants

| Function | Operation | Data Types |
|----------|-----------|-----------|
| `fused_gemm_a8w8_blockscale_mul_add` | `Y = a * GEMM + b` or `Y = a * b + GEMM` | INT8 block-scale |
| `fused_gemm_afp4wfp4_mul_add` | `Y = a * GEMM + b` | FP4 |
| `fused_gemm_a8w8_blockscale_split_cat` | GEMM with split + concatenation | INT8 block-scale |
| `fused_gemm_afp4wfp4_split_cat` | GEMM with split + concatenation | FP4 |
| `fused_gemm_a8w8_blockscale_a16w16` | Chained: A8W8 GEMM → A16W16 GEMM | INT8 → BF16 |
| `fused_gemm_afp4wfp4_a16w16` | Chained: FP4 GEMM → A16W16 GEMM | FP4 → BF16 |

### When to Use

- Residual connections after GEMM (`a * gemm_out + bias`)
- Split-K with concatenation for parallel heads
- Chained projections (e.g., QKV projection → output projection)

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_mul_add.py` | INT8 GEMM + mul_add |
| `aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_mul_add.py` | FP4 GEMM + mul_add |
| `aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_split_cat.py` | INT8 GEMM + split_cat |
| `aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_split_cat.py` | FP4 GEMM + split_cat |
| `aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_a16w16.py` | Chained INT8 + BF16 |
| `aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_a16w16.py` | Chained FP4 + BF16 |

---

## 9. Tuning System

AITER ships with pre-tuned kernel configurations for common model shapes. The tuning system selects the optimal kernel and SplitK factor for each (M, N, K) shape.

### How Tuning Works

```
1. User calls gemm_a8w8(XQ, WQ, ...)
2. System extracts (M, N, K) from tensor shapes
3. Looks up (cu_num, M, N, K) in tuned CSV
4. If found → uses specified kernelName + splitK
5. If not → pads M to next tile boundary, retries
6. If still not → falls back to default backend
```

### CSV Format

All tuning CSVs share a common schema:

```csv
cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio
304,1,3584,512,0,0,4.5,ck_gemm_f8_f8_f16_0,15.2,890.1,0.001
```

| Column | Description |
|--------|-------------|
| `cu_num` | GPU compute unit count (304 for MI300X) |
| `M,N,K` | Problem dimensions |
| `kernelId` | CK kernel variant index |
| `splitK` | K-split factor (0 = no split) |
| `us` | Measured execution time (microseconds) |
| `kernelName` | CK/ASM kernel identifier |
| `tflops` | Achieved TFLOPS |
| `bw` | Memory bandwidth (GB/s) |
| `errRatio` | Numerical error ratio vs reference |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AITER_TUNE_GEMM` | 0 | Set to 1 to collect untuned shapes for offline tuning |
| `AITER_LOG_TUNED_CONFIG` | 0 | Log when tuned configs are loaded |
| `AITER_CONFIG_GEMM_A8W8` | built-in | Override A8W8 tuning CSV path |
| `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` | built-in | Override A8W8 pre-shuffle CSV |
| `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` | built-in | Override A8W8 block-scale CSV |
| `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` | built-in | Override combined CSV |
| `AITER_CONFIG_GEMM_A4W4` | built-in | Override A4W4 CSV |
| `AITER_CONFIG_GEMM_BF16` | built-in | Override BF16 CSV |
| `AITER_CONFIG_A8W8_BATCHED_GEMM` | built-in | Override batched A8W8 CSV |
| `AITER_CONFIG_BF16_BATCHED_GEMM` | built-in | Override batched BF16 CSV |

### Model-Specific Configs

Pre-tuned configs for popular models in `aiter/configs/model_configs/`:

| Config | Model |
|--------|-------|
| `dsv3_bf16_tuned_gemm.csv` | DeepSeek-V3 |
| `a8w8_blockscale_tuned_gemm_qwen3_235b.csv` | Qwen3-235B |
| `a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3_235b.csv` | Qwen3-235B (preshuffle) |

### SplitK Tuning

SplitK divides the K dimension across multiple thread groups for better GPU utilization on small-M (decode) problems:

```python
# Auto-computed based on CU count and tile sizes:
# splitK = max(1, cu_count // (ceil(M/tile_m) * ceil(N/tile_n)))
# Capped at K // tile_k
```

Override: pass `splitK=N` explicitly to any GEMM function.

---

## 10. Backend Comparison: How to Choose

### Performance Characteristics

| Backend | Strengths | Weaknesses |
|---------|-----------|------------|
| **ASM** | Best peak throughput, hand-tuned per arch | Fixed shapes, arch-locked |
| **CK/CKTile** | Good performance, broadest variant coverage | Requires ROCm compilation |
| **HipBLASLt** | Good for BF16, auto-tuned | Limited to standard GEMM |
| **Triton** | Portable, fused ops, configurable | Slightly lower peak perf |
| **PyTorch** | Universal fallback | Lowest performance |

### Decision Tree

```
Need quantized GEMM?
├── FP8/INT8?
│   ├── Per-tensor scale → gemm_a8w8 (ASM/CK)
│   ├── Block-scale → gemm_a8w8_blockscale (CK/ASM)
│   └── Pre-shuffled weights → gemm_a8w8_bpreshuffle (CK/CKTile)
├── FP4?
│   └── gemm_a4w4 (MI350 only)
└── No quantization?
    ├── Standard GEMM → gemm_a16w16 (auto-dispatch)
    ├── FFN block → ff_a16w16_gated / ff_a16w16_nogate (Triton)
    └── Batched → batched_gemm_bf16_CK
```

---

## 11. GPU Architecture Support

| Variant | MI300X (GFX942) | MI350 (GFX950) | Other GPUs |
|---------|:---:|:---:|:---:|
| A8W8 (per-tensor) | ASM + CK + Triton | CK + Triton | Triton |
| A8W8 (block-scale) | ASM + CK + Triton | ASM + CK + Triton | Triton |
| A16W16 | ASM + HipBLASLt + Triton | CK + Triton | Triton + PyTorch |
| A4W4 | - | ASM + CK | Triton |
| Batched A8W8 | CK + Triton | CK + Triton | Triton |
| Batched BF16 | CK + Triton | CK + Triton | Triton |
| DeepGEMM | CK | - | - |
| Feed-forward fusions | Triton | Triton | Triton |
| Fused GEMM ops | Triton | Triton | Triton |

---

## 12. Test Files Reference

| Test File | Covers |
|-----------|--------|
| `op_tests/test_gemm_a8w8.py` | A8W8: per-tensor, block-scale, bpreshuffle |
| `op_tests/test_gemm_a16w16.py` | A16W16: BF16/FP16, all backends |
| `op_tests/test_gemm_a4w4.py` | A4W4: FP4 on MI350 |
| `op_tests/test_batched_gemm_a8w8.py` | Batched INT8 GEMM |
| `op_tests/test_batched_gemm_bf16.py` | Batched BF16 GEMM |
| `op_tests/test_deepgemm.py` | Group-wise GEMM |
| `op_tests/triton_tests/gemm/` | All Triton GEMM variants (23+ test files) |

---

## 13. Source Files Reference

### Main API

| File | Purpose |
|------|---------|
| `aiter/ops/gemm_op_a8w8.py` | All A8W8 variants, tuning interfaces |
| `aiter/ops/gemm_op_a16w16.py` | Low-level A16W16 ASM interface |
| `aiter/tuned_gemm.py` | High-level A16W16 with auto-dispatch |
| `aiter/ops/gemm_op_a4w4.py` | All A4W4 variants |
| `aiter/ops/batched_gemm_op_a8w8.py` | Batched A8W8 |
| `aiter/ops/batched_gemm_op_bf16.py` | Batched BF16 |
| `aiter/ops/deepgemm.py` | Group-wise GEMM |

### Triton Kernels

| Directory | Contents |
|-----------|----------|
| `aiter/ops/triton/gemm/basic/` | 11 basic GEMM variants (a8w8, a16w16, afp4wfp4, etc.) |
| `aiter/ops/triton/gemm/batched/` | 5 batched variants |
| `aiter/ops/triton/gemm/feed_forward/` | 3 FFN fusions |
| `aiter/ops/triton/gemm/fused/` | 6 fused GEMM+element-wise ops |

### Configuration

| File | Purpose |
|------|---------|
| `aiter/configs/a8w8_tuned_gemm.csv` | A8W8 per-tensor tuning (551 entries) |
| `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` | A8W8 pre-shuffle tuning (702 entries) |
| `aiter/configs/a8w8_blockscale_tuned_gemm.csv` | A8W8 block-scale tuning (79 entries) |
| `aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | Combined tuning (118 entries) |
| `aiter/configs/a4w4_blockscale_tuned_gemm.csv` | A4W4 tuning (925 entries) |
| `aiter/configs/bf16_tuned_gemm.csv` | BF16 tuning |
| `aiter/configs/a8w8_tuned_batched_gemm.csv` | Batched A8W8 (27 entries) |
| `aiter/configs/bf16_tuned_batched_gemm.csv` | Batched BF16 (26 entries) |

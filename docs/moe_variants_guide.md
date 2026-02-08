# AITER MOE (Mixture of Experts) Variants & Backend Guide

This guide documents all MOE variants available in AITER, their backend support, and how to choose the right configuration for your use case.

---

## Quick Reference: Which MOE Configuration Should I Use?

| Use Case | Recommended Config | Backend | Why |
|----------|-------------------|---------|-----|
| **BF16 inference (best accuracy)** | Standard FusedMOE, `QuantType.No` | ASM or CK | No quantization loss |
| **FP8 inference (balanced)** | FP8 W8A8 per-token | ASM | Good accuracy/perf tradeoff |
| **FP8 inference (high accuracy)** | FP8 block-scale (128x128) | CK or ASM | Per-block scales reduce error |
| **Maximum compression** | MXFP4 W4A16 | Triton (MXFP4) or CK | 4x weight reduction |
| **DeepSeek-V3 style** | 256 experts, TopK=8 | ASM | Optimized topk_softmax for (256,8) |
| **Distributed (multi-GPU)** | EP MOE with expert_mask | ASM or CK | Experts split across GPUs |
| **Shared experts (DP)** | DP Shared Expert MOE | ASM or CK | All experts replicated |
| **Prototyping / new models** | Triton FusedMOE | Triton | Easy to modify, broadest dtype |

---

## 1. MOE Architecture in AITER

AITER implements a two-stage Fused MOE pipeline:

```
Input Tokens                    Output Tokens
     │                               ▲
     ▼                               │
┌─────────┐                    ┌─────────┐
│ Routing  │ (TopK Softmax     │ Stage 2 │ GEMM: intermediate → output
│ + Sort   │  or Sigmoid)      │  (w2)   │ + routing weight application
└────┬────┘                    └────┬────┘
     │                               ▲
     ▼                               │
┌─────────┐                    ┌──────────┐
│ Stage 1 │ GEMM: input →     │Activation│ SiLU, GELU, or Swiglu
│  (w1)   │ intermediate      │  Fusion  │ (fused in Stage 1 kernel)
└─────────┘                    └──────────┘
```

### Key Concepts

- **G1U1 mode**: Gate and Up projections are combined in a single weight matrix. Stage 1 produces `gate * up` in one GEMM. This is the standard for most MOE models.
- **G1U0 mode**: Gate-only projection. Stage 1 produces just the gated output.
- **Routing weights**: TopK expert selection weights, applied either at Stage 1 (`doweight_stage1=True`) or Stage 2.
- **Token sorting**: Tokens are reordered by expert assignment before GEMM to improve cache locality.

---

## 2. MOE Variants

### 2.1 Standard Fused MOE

The default and most commonly used variant. Supports the full range of quantization types and backends.

**Backends:** ASM, CK (C++), Triton

```python
from aiter.fused_moe import fused_moe, fused_topk
from aiter import ActivationType, QuantType

# Step 1: Route tokens to experts
topk_weights, topk_ids = fused_topk(
    hidden_states, gating_output,
    topk=2, renormalize=True
)

# Step 2: Execute MOE
output = fused_moe(
    hidden_states, w1, w2,
    topk_weights, topk_ids,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
)
```

---

### 2.2 Expert Parallel (EP) MOE

Distributes experts across multiple GPUs. Each GPU holds a subset of experts and only processes tokens routed to its local experts.

**Backends:** ASM, CK (C++)

```python
# Define which experts are local to this GPU
expert_mask = torch.zeros(global_num_experts, dtype=torch.bool, device=device)
expert_mask[local_start:local_end] = True

output = fused_moe(
    hidden_states, w1, w2,
    topk_weights, topk_ids,
    expert_mask=expert_mask,
)
```

**When to use:** When the total expert weights exceed single-GPU memory, or when you want to scale to more experts than one GPU can handle.

---

### 2.3 Data Parallel (DP) MOE with Shared Experts

All experts are replicated across GPUs. Each GPU processes a subset of tokens (split by DP rank) and accumulates results via atomic operations.

**Backends:** ASM, CK (C++), Triton

```python
from aiter.fused_moe_dp_shared_expert import fused_moe_dp_share_expert

output = fused_moe_dp_share_expert(
    hidden_states, w1, w2,
    quant_type=QuantType.per_Token,
    dp_size=world_size,
    dp_rank=rank,
    moe_buf=output_buffer,  # optional pre-allocated buffer for accumulation
)
```

**When to use:** When expert weights fit on each GPU and you want simpler communication patterns than EP.

---

### 2.4 Block-Scale FP8 MOE

Uses per-block quantization (typically 128x128 blocks) for FP8, reducing quantization error compared to per-tensor FP8.

**Backends:** CK (preferred), ASM

```python
output = fused_moe(
    hidden_states, w1_fp8, w2_fp8,
    topk_weights, topk_ids,
    w1_scale=fc1_block_scale,  # shape: [E, inter_dim/128, model_dim/128]
    w2_scale=fc2_block_scale,  # shape: [E, model_dim/128, inter_dim/128]
    quant_type=QuantType.per_1x128,
    activation=ActivationType.Silu,
)
```

**When to use:** When you want FP8 performance with better accuracy than per-tensor quantization.

---

### 2.5 MXFP4 MOE

Microscaling FP4 quantization with per-32-element scaling. Provides 4x weight compression with E8M0 scale format.

**Backends:** Triton (primary), CK (gfx950 preferred)

```python
output = fused_moe(
    hidden_states, w1_mxfp4, w2_mxfp4,
    topk_weights, topk_ids,
    w1_scale=fc1_e8m0_scale,  # E8M0 format scales
    w2_scale=fc2_e8m0_scale,
    quant_type=QuantType.per_1x32,
    activation=ActivationType.Silu,
)
```

**When to use:** When maximum weight compression is needed (4x savings), especially on MI350 (gfx950) which has native FP4 support.

---

### 2.6 End-to-End (E2E) MOE

Combines routing, GEMM, activation, and output permutation into a single kernel launch, reducing kernel launch overhead.

**Backends:** Triton (experimental)

```python
from aiter.ops.triton.moe.moe_op_e2e import e2e_moe

output = e2e_moe(
    hidden_states, w1, w2,
    topk_weights, topk_ids,
    ...
)
```

**When to use:** Experimental; may help with small batch sizes where kernel launch overhead is significant.

---

## 3. Routing / Expert Selection

### TopK Softmax

Standard routing: apply softmax to gating logits, then select top-K experts.

**Backends:** ASM (optimized), HIP/CUDA (fallback)

```python
from aiter.ops.moe_op import topk_softmax

topk_weights, topk_ids = topk_softmax(
    gating_output,  # [num_tokens, num_experts]
    topk=2,
    renormalize=True,
)
```

**ASM-optimized configurations:**
| (num_experts, topk) | ASM Optimized |
|---------------------|:---:|
| (128, 4) | Yes |
| (128, 6) | Yes |
| (128, 8) | Yes |
| (256, 6) | Yes |
| (256, 8) | Yes |
| Other | Falls back to HIP/CUDA |

### TopK Sigmoid

Alternative routing using sigmoid activation. Used in some model architectures for binary-like expert selection.

**Backends:** Triton, HIP/CUDA

```python
from aiter.ops.moe_op import topk_sigmoid

topk_weights, topk_ids = topk_sigmoid(
    gating_output,
    topk=1,
)
```

---

## 4. Data Type & Quantization Support Matrix

### By Backend

| Quantization | ASM (GFX942) | CK (C++) | Triton | Description |
|-------------|:---:|:---:|:---:|-------------|
| **No (BF16/FP16)** | Yes | Yes | Yes | Full precision, no quantization |
| **FP8 W8A8 per-token** | Yes | Yes | Yes | Per-token FP8 quantization |
| **FP8 block-scale (per 128x128)** | Yes | Yes | Yes | Per-block FP8 quantization |
| **INT8 W8A8** | Yes | Yes | Yes | Per-token INT8, both weights and activations |
| **INT8 W8A16** | Yes | Yes | - | INT8 weights, BF16 activations |
| **MXFP4 W4A16** | - | - | Yes | MXFP4 weights, BF16 activations |
| **MXFP4 W4A4** | - | - | Yes | MXFP4 weights and activations |
| **MXFP4 (per 32-element)** | Yes | Yes | Yes | Microscaling FP4 with E8M0 scales |

### Choosing the Right Quantization

```
Need maximum accuracy?
├── Yes → QuantType.No (BF16)
└── No
    ├── Need good accuracy with 2x compression?
    │   ├── FP8 block-scale (QuantType.per_1x128) — best accuracy/compression
    │   └── FP8 per-token (QuantType.per_Token) — simpler, slightly less accurate
    ├── Need 4x compression?
    │   ├── MXFP4 (QuantType.per_1x32) — best for MI350
    │   └── MXFP4 W4A16 — Triton only, good accuracy
    └── Need INT8 compatibility?
        ├── INT8 W8A8 — both quantized
        └── INT8 W8A16 — keep activations in BF16
```

---

## 5. Activation Function Support

| Activation | ASM | CK | Triton | Typical Use |
|-----------|:---:|:---:|:---:|-------------|
| **SiLU** (Swish + Gate) | Yes | Yes | Yes | LLaMA, Mistral, DeepSeek, most modern LLMs |
| **GELU** | Yes | Yes | Yes | GPT, OPT, BLOOM |
| **Swiglu** | - | Yes (FP4 preshuffle) | Limited | Some specialized architectures |

```python
from aiter import ActivationType

# Select activation
activation = ActivationType.Silu   # Most common
activation = ActivationType.Gelu   # For GPT-style models
```

---

## 6. Backend Comparison

### Performance Characteristics

| Backend | Strengths | Best For |
|---------|-----------|----------|
| **ASM** | Highest throughput, hand-tuned for MI300X | Production inference on MI300X |
| **CK (C++)** | Good performance, broad dtype, 2-stage pipeline | Production on MI300X/MI350 |
| **Triton** | Most flexible, broadest quantization support | Prototyping, MXFP4, new GPUs |

### Automatic Backend Selection

AITER automatically selects the best backend based on your configuration:

1. **Tuned configs**: AITER ships with pre-tuned kernel configurations for common model shapes in `aiter/configs/tuned_fmoe.csv`. These specify the optimal kernel for each (token_count, model_dim, inter_dim, expert_count, topk, dtype) combination.

2. **Fallback**: If no tuned config matches, AITER uses heuristics to pick a reasonable default.

3. **Override**: You can control backend selection via environment variables:
   ```bash
   AITER_BYPASS_TUNE_CONFIG=1   # Skip tuned configs, use defaults
   AITER_ONLINE_TUNE=1          # Auto-tune at runtime for untuned configs
   ```

### Kernel Naming Convention (CK 2-Stage)

CK kernel names encode their configuration:
```
moe_ck2stages_gemm1_256x64x128x64_1x4_TypeCast_v3_silu_B16_B16_B16
         │      │        │         │     │        │    │    │    │
         │      │   tile sizes     │  variant  activation │  dtype_c
         │    stage            grouping             dtype_a  dtype_b
       backend
```

---

## 7. GPU Architecture Support

### MI300X (GFX942)

- **Compute Units:** 304
- **Memory:** 192 GB HBM3, ~5.3 TB/s bandwidth
- **Preferred backends:** ASM (hand-tuned), CK
- **Optimal block sizes:** M=64, N=128, K=256
- **Best for:** High-throughput inference, all quantization types

### MI350 (GFX950)

- **Compute Units:** 256+
- **Memory:** HBM3E
- **Preferred backends:** CK, Triton (MXFP4)
- **Native FP4 support:** Yes (hardware acceleration for MXFP4)
- **Best for:** MXFP4 quantized models, next-gen deployments

### Support Matrix

| Feature | MI300X (GFX942) | MI350 (GFX950) | Other GPUs |
|---------|:---:|:---:|:---:|
| ASM kernels | Yes (full) | Yes (full) | No |
| CK kernels | Yes | Yes | No |
| Triton kernels | Yes | Yes | Yes (portable) |
| TopK Softmax ASM | Yes | Yes | No |
| MXFP4 native | Software | Hardware | Triton only |

---

## 8. Common Model Configurations

AITER includes pre-tuned configurations for popular MOE models:

| Model | Experts | TopK | Hidden | Intermediate | Recommended |
|-------|---------|------|--------|-------------|-------------|
| Mixtral 8x7B | 8 | 2 | 4096 | 14336 | BF16 or FP8 W8A8 |
| Mixtral 8x22B | 8 | 2 | 6144 | 16384 | BF16 or FP8 W8A8 |
| DeepSeek-V2 | 160 | 6 | 5120 | 1536 | FP8 block-scale |
| DeepSeek-V3 | 256 | 8 | 7168 | 2048 | FP8 block-scale |
| Qwen3-235B | 128 | 8 | 4096 | 12288 | FP8 or MXFP4 |

---

## 9. Performance Tuning

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AITER_CONFIG_FMOE=<path>` | Override tuned config CSV | Built-in |
| `AITER_USE_NT=1` | Enable non-temporal memory loads | 0 |
| `AITER_KSPLIT=<value>` | Override K-split factor | Auto |
| `AITER_ONLINE_TUNE=1` | Auto-tune untuned configurations | 0 |
| `AITER_BYPASS_TUNE_CONFIG=1` | Skip tuned configs | 0 |

### Key Tuning Parameters

- **block_size_M**: Token batch per kernel block (16, 32, 64, 128). Larger = better throughput, but more padding waste for small batches.
- **ksplit**: K-dimension splitting (0-8). Higher values improve parallelism for compute-bound cases.
- **doweight_stage1**: Apply routing weights in Stage 1 instead of Stage 2. Reduces memory bandwidth at the cost of slightly different numerics.
- **use_non_temporal_load**: Skip L2 cache for weight loads. Helps when weights are large and won't be reused.

---

## 10. Test Files Reference

| Test File | Covers |
|-----------|--------|
| `op_tests/test_moe.py` | Standard FusedMOE, all backends |
| `op_tests/test_moe_2stage.py` | 2-stage pipeline |
| `op_tests/test_moe_ep.py` | Expert Parallel |
| `op_tests/test_moe_dp_share_expert.py` | DP Shared Expert |
| `op_tests/test_moe_blockscale.py` | FP8 block-scale |
| `op_tests/test_moeTopkSoftmax.py` | TopK Softmax routing |
| `op_tests/test_moe_topk_sigmoid.py` | TopK Sigmoid routing |
| `op_tests/test_moe_sorting.py` | Token sorting/permutation |
| `op_tests/test_moe_sorting_mxfp4.py` | MXFP4 sorting |
| `op_tests/test_moe_tkw1.py` | Weight-at-Stage-1 variant |
| `op_tests/triton_tests/moe/test_moe.py` | Triton MOE (all dtypes) |
| `op_tests/triton_tests/moe/test_moe_gemm_a8w8.py` | Triton FP8 W8A8 |
| `op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py` | Triton FP8 block-scale |
| `op_tests/triton_tests/moe/test_moe_gemm_a4w4.py` | Triton MXFP4 |

---

## 11. Source Files Reference

### Main API

| File | Purpose |
|------|---------|
| `aiter/fused_moe.py` | Primary FusedMOE API, backend dispatch |
| `aiter/fused_moe_bf16_asm.py` | ASM backend wrapper |
| `aiter/fused_moe_dp_shared_expert.py` | DP Shared Expert variant |
| `aiter/ops/moe_op.py` | TopK routing and MOE utilities |
| `aiter/ops/moe_sorting.py` | Token sorting operations |

### Triton Kernels

| File | Purpose |
|------|---------|
| `aiter/ops/triton/moe/moe_op.py` | Standard fused GEMM |
| `aiter/ops/triton/moe/moe_op_silu_fused.py` | Fused SiLU activation |
| `aiter/ops/triton/moe/moe_op_gelu.py` | Fused GELU activation |
| `aiter/ops/triton/moe/moe_op_e2e.py` | End-to-End MOE |
| `aiter/ops/triton/moe/moe_op_mxfp4.py` | MXFP4 quantized MOE |
| `aiter/ops/triton/moe/moe_op_mxfp4_silu_fused.py` | MXFP4 + SiLU fused |
| `aiter/ops/triton/moe/moe_routing_sigmoid_top1_fused.py` | TopK Sigmoid routing |

### Configuration

| File | Purpose |
|------|---------|
| `aiter/configs/tuned_fmoe.csv` | Pre-tuned kernel configurations |
| `aiter/configs/untuned_fmoe.csv` | Fallback configurations |
| `hsa/gfx942/fmoe/` | MI300X single-stage ASM configs |
| `hsa/gfx942/fmoe_2stages/` | MI300X 2-stage ASM configs |
| `hsa/gfx950/fmoe/` | MI350 single-stage ASM configs |
| `hsa/gfx950/fmoe_2stages/` | MI350 2-stage ASM configs |

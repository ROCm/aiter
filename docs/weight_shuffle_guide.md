# AITER Weight Shuffle & Preshuffle Guide

This guide documents the weight shuffle operators in AITER that transform weight tensor layouts for optimized GEMM execution on AMD GPUs. Preshuffling is a one-time cost at weight loading time that enables faster runtime GEMM.

---

## Quick Reference

| Use Case | Function | Layout | Backend |
|----------|---------|--------|---------|
| **CK/CKTile GEMM** | `shuffle_weight(w, (16, 16))` | 16×16 blocks | CK |
| **ASM GEMM** | `shuffle_weight(w, (32, 16))` | 32×16 blocks | ASM |
| **DeepGEMM** | `shuffle_weight(w, (16, 16))` | 16×16 blocks | CK-tile |
| **MoE FP4 weights** | `shuffle_weight_a16w4(w, NLane, gate_up)` | FP4-optimized | MoE |
| **MoE FP4 scales** | `shuffle_scale_a16w4(scales, experts, gate_up)` | MXFP4 scales | MoE |

---

## 1. Core Shuffle Functions

### `shuffle_weight`

Rearranges weight tensor from standard `(N, K)` layout to blocked layout matching kernel access patterns:

```python
from aiter.ops.shuffle import shuffle_weight

# For CK/CKTile backends
weight_ck = shuffle_weight(weight_q, layout=(16, 16))

# For ASM backend
weight_asm = shuffle_weight(weight_q, layout=(32, 16))

# For INT4
weight_int4 = shuffle_weight(weight_q, layout=(16, 16), use_int4=True)
```

**Parameters:**
- `x`: Weight tensor `(N, K)`
- `layout`: `(IN, IK)` block dimensions — `(16, 16)` for CK, `(32, 16)` for ASM
- `use_int4`: Enable int4 mode (packed `uint8`)

**Supported dtypes:** `int8`, `float8_e4m3fnuz`, `float4_e2m1fn_x2`, `uint8` (packed int4)

**What it does:** Permutes dimensions `(N, K) → (N//BN, K//BK, BK//K_elem, BN, K_elem) → permute → (N, K)` with optimized memory layout.

### `shuffle_weight_NK`

Alternative shuffle using instruction tile dimensions:

```python
from aiter.ops.shuffle import shuffle_weight_NK

weight_shuffled = shuffle_weight_NK(weight, inst_N=16, inst_K=16)
```

### `shuffle_weight_a16w4` (MoE FP4)

Shuffle for A16W4 MoE operations with gate-up fusion:

```python
from aiter.ops.shuffle import shuffle_weight_a16w4

# weight: [experts_cnt, N, K_pk] where K_pk = K // 2 (packed FP4)
weight_shuffled = shuffle_weight_a16w4(weight, NLane=16, gate_up=True)
```

### `shuffle_scale_a16w4` (MoE FP4 Scales)

Shuffle MXFP4 scale tensors for MoE operations:

```python
from aiter.ops.shuffle import shuffle_scale_a16w4

scale_shuffled = shuffle_scale_a16w4(scales, experts_cnt=64, gate_up=True)
```

---

## 2. GEMM Integration

### A8W8 FP8 with Preshuffle

```python
import aiter
from aiter.ops.shuffle import shuffle_weight

# Quantize
x_q, x_scale = aiter.pertoken_quant(x, quant_dtype=torch.float8_e4m3fnuz)
w_q, w_scale = aiter.pertoken_quant(weight, quant_dtype=torch.float8_e4m3fnuz)

# Shuffle (one-time cost)
w_shuffled = shuffle_weight(w_q, layout=(16, 16))

# Run optimized GEMM
out = aiter.gemm_a8w8_bpreshuffle(x_q, w_shuffled, x_scale, w_scale, dtype=torch.bfloat16)
```

### Backend Dispatch

The tuned GEMM system automatically selects the backend based on CSV config lookup:

| Backend | Shuffle Layout | Scale Type | Notes |
|---------|---------------|------------|-------|
| CK | `(16, 16)` | Per-tensor | Primary FP8 backend |
| CKTile | `(16, 16)` | Per-tensor | Alternative CK backend |
| ASM | `(32, 16)` | Per-tensor | Hand-tuned assembly |
| ASM Blockscale | `(32, 16)` | Block (128×128) | GFX950 only |
| Triton AFP4WFP4 | `(16, 16)` | Block (32×32) | FP4 Triton kernels |

### Blockscale Preshuffle

For block-wise quantized GEMM (128×128 blocks):

```python
# CK blockscale
w_ck = shuffle_weight(weight_q, layout=(16, 16))
out = aiter.gemm_a8w8_blockscale_bpreshuffle(x_q, w_ck, x_scale, w_scale, dtype)

# ASM blockscale (GFX950)
w_asm = shuffle_weight(weight_q, layout=(32, 16))
out = aiter.gemm_a8w8_blockscale_bpreshuffle_asm(x_q, w_asm, out, x_scale, w_scale)
```

---

## 3. Why Preshuffle?

Standard weight layouts cause strided memory accesses in GEMM kernels. Preshuffling rearranges data to match the kernel's tile access pattern:

```
Standard layout: N×K row-major
    → Strided loads for K-dimension tiles
    → Bank conflicts, poor cache utilization

Preshuffled layout: (N//BN, K//BK, BN, BK) blocked
    → Coalesced loads within each tile
    → Optimal L2 cache reuse
```

**Trade-off:** One-time shuffle cost at weight loading vs. faster GEMM at every inference step.

---

## 4. Tuning Configuration

Preshuffle-specific config files in `aiter/configs/`:

| File | Description |
|------|-------------|
| `a8w8_bpreshuffle_tuned_gemm.csv` | FP8 per-tensor preshuffle configs |
| `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | FP8 blockscale preshuffle configs |

**Tuning:**
```bash
# A8W8 preshuffle tuning
python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py \
    -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv \
    -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv \
    --libtype ck,cktile -k
```

---

## 5. Source Files

| Component | Path |
|---|---|
| Shuffle Python API | `aiter/ops/shuffle.py` |
| A8W8 GEMM dispatch | `aiter/ops/gemm_op_a8w8.py` |
| CK preshuffle kernels | `csrc/ck_gemm_a8w8_bpreshuffle/` |
| CKTile preshuffle kernels | `csrc/cktile_gemm_a8w8_bpreshuffle/` |
| Blockscale preshuffle kernels | `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/` |
| Triton FP4 preshuffle | `aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py` |
| Preshuffle tuning configs | `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` |

---

## 6. Test Files

| Test | Path |
|------|------|
| A8W8 GEMM (with preshuffle) | `op_tests/test_gemm_a8w8.py` |
| A8W8 blockscale (with preshuffle) | `op_tests/test_gemm_a8w8_blockscale.py` |
| AFP4WFP4 (with preshuffle) | `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py` |

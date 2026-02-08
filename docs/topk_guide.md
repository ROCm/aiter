# AITER Top-K Operators Guide

This guide documents all top-k selection operators available in AITER, including MOE routing variants, general-purpose top-k, and fused top-k + softmax operations.

---

## Quick Reference

| Use Case | Recommended Operation | Backend | Why |
|----------|---------------------|---------|-----|
| **Standard MOE routing** | `topk_softmax` / `topk_softmax_asm` | HIP/ASM | Fused softmax + top-k |
| **Grouped MOE routing (DeepSeek)** | `grouped_topk` | HIP/ASM | Group-aware expert selection |
| **Biased grouped routing (DeepSeek-V3)** | `biased_grouped_topk` | HIP/ASM | Correction bias for load balance |
| **General-purpose top-k** | `topk_plain` | HIP | Adaptive 3-strategy selection |
| **Prefill top-k (variable rows)** | `top_k_per_row_prefill` | HIP | Radix-sort for variable-length rows |
| **Decode top-k (speculative)** | `top_k_per_row_decode` | HIP | Speculative decoding support |
| **Triton top-k** | `aiter.ops.triton.topk.topk` | Triton | Portable, bitonic sort |

---

## 1. MOE Routing Operators

### Fused Softmax + Top-K

Standard MOE routing — computes softmax over expert logits and selects top-k experts in a single kernel:

```python
import aiter

aiter.topk_softmax(
    topk_weights,           # [num_tokens, topk] — output weights
    topk_indices,           # [num_tokens, topk] — output expert indices
    token_expert_indices,   # [num_tokens, topk] — token-expert mapping
    gating_output,          # [num_tokens, num_experts] — router logits
    need_renorm=True,       # Renormalize weights to sum to 1
)

# ASM-optimized variant
aiter.topk_softmax_asm(topk_weights, topk_indices, token_expert_indices,
                        gating_output, need_renorm)
```

Specialized for power-of-2 expert counts (1–512). Falls back to two-kernel approach for other counts.

### Grouped Top-K (DeepSeek-style)

Selects experts in two stages: first pick top groups, then pick top experts within those groups:

```python
aiter.grouped_topk(
    gating_output,          # [num_tokens, num_experts]
    topk_weights,           # [num_tokens, topk] — output
    topk_ids,               # [num_tokens, topk] — output
    num_expert_group=4,     # Number of expert groups
    topk_group=2,           # Groups to select per token
    need_renorm=True,
    is_softmax=True,        # True=softmax, False=sigmoid scoring
    routed_scaling_factor=1.0,
)
```

### Biased Grouped Top-K (DeepSeek-V3)

Adds a correction bias to expert selection scores (but NOT to routing weights):

```python
aiter.biased_grouped_topk(
    gating_output,          # [num_tokens, num_experts]
    correction_bias,        # [num_experts] — per-expert bias
    topk_weights,           # [num_tokens, topk] — output
    topk_ids,               # [num_tokens, topk] — output
    num_expert_group=4,
    topk_group=2,
    need_renorm=True,
    routed_scaling_factor=1.0,
)
```

**How the bias works:**
1. Scores are computed with sigmoid: `scores = sigmoid(gating_output)`
2. Selection uses biased scores: `scores_for_choice = scores + correction_bias`
3. But routing weights use **unbiased** scores: `weights = scores.gather(topk_ids)`

This compensates for expert load imbalances without distorting actual routing weights.

**Auto-dispatch:** Selects between `biased_grouped_topk_hip` (small batches) and `moe_fused_gate` (large batches) based on `token_count vs cu_num * 212`.

---

## 2. General-Purpose Top-K

### `topk_plain`

Adaptive top-k selection with three strategies, automatically chosen based on input size:

```python
aiter.topk_plain(
    x,              # [batch, hidden_size] — input
    topk_ids,       # [batch, topk] — output indices (int32)
    topk_out,       # [batch, topk] — output values
    topk=10,
    largest=True,   # True for largest, False for smallest
)
```

| Strategy | When Used | Method |
|----------|-----------|--------|
| **BlockTopkFilter** | Large rows | Ballot-based filtering with `__ballot()` |
| **BlockTopkSort** | Medium rows | Bitonic sort in registers |
| **BlockTopkMerge** | Multi-block reduction | Merge pre-sorted k-sized chunks |

Selection heuristic: radix sort wins for large K; bitonic wins for small K and large rows.

**Supported dtypes:** float16, bfloat16, float32, int32

### Per-Row Top-K (Variable-Length)

For attention/speculative decoding with variable-length rows:

```python
# Prefill (variable row lengths via rowStarts/rowEnds)
aiter.top_k_per_row_prefill(
    logits, rowStarts, rowEnds, indices, values,
    numRows, stride0, stride1,
)

# Decode (speculative decoding with next_n candidates)
aiter.top_k_per_row_decode(
    logits, next_n, seqLens, indices,
    numRows, stride0, stride1,
)

# Fast variants (gfx942/MI300 only)
aiter.top_k_per_row_prefill_fast(...)
aiter.top_k_per_row_decode_fast(...)
```

---

## 3. Triton Top-K

```python
from aiter.ops.triton.topk import topk

values, indices = topk(x, k=10)  # x: [B, M], returns [B, k] each
```

Uses 1-stage bitonic sort for rows ≤ 1024, 2-stage for larger rows.

### Triton MOE Routing Top-K

Returns a bitmatrix for efficient MOE dispatch:

```python
from aiter.ops.triton.moe.moe_routing.topk import topk as moe_topk

values, indices, bitmatrix = moe_topk(
    gating_logits, k=8, apply_softmax=True, return_bitmatrix=True
)
```

---

## 4. Integration with MOE Pipeline

```
gating_output [num_tokens, num_experts]
    │
    ▼
topk_softmax / grouped_topk / biased_grouped_topk
    │
    ├── topk_weights [num_tokens, topk]
    └── topk_ids     [num_tokens, topk]
            │
            ▼
        moe_sorting_fwd  →  sorted token/expert IDs
            │
            ▼
        fmoe / ck_moe_stage1+2  →  expert GEMM execution
```

---

## 5. Backend Support

| Operator | HIP/ASM | Triton |
|----------|:-------:|:------:|
| `topk_softmax` | Yes | — |
| `topk_softmax_asm` | Yes (ASM) | — |
| `grouped_topk` | Yes (ASM) | — |
| `biased_grouped_topk` | Yes (ASM) | — |
| `topk_plain` | Yes (DPP, wave intrinsics) | — |
| `top_k_per_row_*` | Yes (radix sort) | — |
| Generic topk | — | Yes (bitonic sort) |
| MOE routing topk | — | Yes (streaming + bitmatrix) |

---

## 6. Decision Tree

```
Need top-k selection?
├── MOE expert routing?
│   ├── Standard softmax routing → topk_softmax() / topk_softmax_asm()
│   ├── Grouped routing (DeepSeek) → grouped_topk()
│   ├── Biased grouped (DeepSeek-V3) → biased_grouped_topk()
│   └── Triton MOE with bitmatrix → moe_routing.topk()
├── General-purpose top-k?
│   ├── Fixed row lengths → topk_plain()
│   └── Variable row lengths → top_k_per_row_prefill()
├── Speculative decoding?
│   └── top_k_per_row_decode()
└── Portable/prototyping?
    └── aiter.ops.triton.topk.topk()
```

---

## 7. Source Files

| Component | Path |
|---|---|
| Grouped/biased topk API | `aiter/ops/topk.py` |
| Plain topk API | `aiter/ops/topk_plain.py` |
| MOE topk API | `aiter/ops/moe_op.py` |
| Radix sort kernels | `csrc/kernels/topk_per_row_kernels.cu` |
| Adaptive topk kernels | `csrc/kernels/topk_plain_kernels.cu` |
| Fused softmax+topk kernels | `csrc/kernels/topk_softmax_kernels.cu` |
| Triton topk | `aiter/ops/triton/topk.py` |
| Triton MOE routing topk | `aiter/ops/triton/moe/moe_routing/topk.py` |

---

## 8. Test Files

| Test | Path |
|------|------|
| Per-row topk (prefill + decode) | `op_tests/test_topk_per_row.py` |
| Plain topk | `op_tests/test_topk_plain.py` |
| Prefill fast vs standard | `op_tests/test_topk_row_prefill.py` |
| Triton topk | `op_tests/triton_tests/test_topk.py` |

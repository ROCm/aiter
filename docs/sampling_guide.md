# AITER Sampling Operators Guide

This guide documents all sampling operators available in AITER for LLM token generation, including greedy, random, mixed, top-k, and top-p sampling strategies.

---

## Quick Reference

| Use Case | Recommended Operation | Backend | Why |
|----------|---------------------|---------|-----|
| **Greedy decoding** | `aiter.greedy_sample` | HIP | Fused argmax |
| **Random sampling** | `aiter.random_sample` | HIP | Fused temperature + softmax + Gumbel-max |
| **Batch with mixed strategies** | `aiter.mixed_sample` | HIP | Per-row greedy/random dispatch |
| **Top-k filtering** | `top_k_renorm_probs` | HIP | Renormalize to top-k tokens |
| **Top-p (nucleus) sampling** | `top_p_sampling_from_probs` | HIP | Sample from cumulative probability threshold |
| **Joint top-k + top-p** | `top_k_top_p_sampling_from_probs` | HIP | Both constraints simultaneously |
| **Generate exponential RNG** | `aiter.exponential` | HIP | Pre-generate for outer-exponential variants |

---

## 1. Two Sampling Families

AITER provides two families of sampling operators targeting different pipeline stages:

### Family A: Logit-Level (Fused)

Operates on raw logits. Fuses temperature scaling, softmax, randomness, and token selection into a single kernel.

```python
import aiter

# Greedy: argmax over logits
out = torch.empty(batch_size, dtype=torch.int32, device="cuda")
aiter.greedy_sample(out, logits)  # logits: [M, vocab_size]

# Random: temperature-scaled stochastic sampling
aiter.random_sample(out, logits, temperatures)  # temperatures: [M], float32

# Mixed: greedy where temperature==0, random elsewhere
aiter.mixed_sample(out, logits, temperatures)
```

### Family B: Probability-Level

Operates on pre-computed probabilities (after softmax). Implements top-k/top-p filtering.

```python
import torch

# Top-k renormalization
renormed = torch.ops.aiter.top_k_renorm_probs(probs, None, top_k=50)

# Top-p sampling
samples = torch.ops.aiter.top_p_sampling_from_probs(probs, None, None, top_p=0.9)

# Joint top-k + top-p
samples = torch.ops.aiter.top_k_top_p_sampling_from_probs(
    probs, None, None, 50, None, 0.9
)
```

---

## 2. Family A: Logit-Level Operators

### `greedy_sample`

```python
aiter.greedy_sample(
    out: Tensor,    # [M], int32 — sampled token indices
    input: Tensor,  # [M, N] — logits (float32/float16/bfloat16)
) -> None
```

Pure argmax — selects the highest-logit token per row.

### `random_sample`

```python
aiter.random_sample(
    out: Tensor,                        # [M], int32
    input: Tensor,                      # [M, N] logits
    temperatures: Tensor,               # [M], float32 — per-row temperature
    lambd: float = 1.0,                 # Exponential distribution rate
    generator: Optional[Generator] = None,  # For reproducible RNG
    eps: float = 1e-10,                 # Numerical stability
) -> None
```

Uses the Gumbel-max trick: `argmax(softmax(logit/T) / exp_sample)`. Fuses temperature scaling, online softmax, exponential RNG, and argmax in one kernel.

### `mixed_sample`

```python
aiter.mixed_sample(
    out: Tensor,           # [M], int32
    input: Tensor,         # [M, N] logits
    temperature: Tensor,   # [M], float32
    lambd: float = 1.0,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None
```

Per-row dispatch: `temperature == 0` → greedy, otherwise → random. Ideal for batched inference where some requests use greedy and others use sampling.

### Outer-Exponential Variants

For deterministic control over randomness:

```python
# Pre-generate exponential samples
exponentials = torch.empty_like(logits, dtype=torch.float32)
aiter.exponential(exponentials, lambd=1.0)

# Use pre-generated samples
aiter.random_sample_outer_exponential(out, logits, exponentials, temperatures)
aiter.mixed_sample_outer_exponential(out, logits, exponentials, temperatures)
```

---

## 3. Family B: Probability-Level Operators

### `top_k_renorm_probs`

```python
torch.ops.aiter.top_k_renorm_probs(
    probs: Tensor,                          # [M, vocab_size], float32
    maybe_top_k_arr: Optional[Tensor],      # [M], int32 — per-row k (or None)
    top_k_val: int,                         # Scalar fallback k
) -> Tensor                                  # [M, vocab_size], renormalized
```

Zeros out all probabilities outside the top-k and renormalizes. Uses binary search over probability thresholds.

### `top_p_sampling_from_probs`

```python
torch.ops.aiter.top_p_sampling_from_probs(
    probs: Tensor,                          # [M, vocab_size], float32
    indices: Optional[Tensor],              # Index mapping (or None)
    maybe_top_p_arr: Optional[Tensor],      # [M], float32 — per-row p (or None)
    top_p_val: float,                       # Scalar fallback p
    deterministic: bool = False,            # Bitwise reproducible scan
) -> Tensor                                  # [M], int32 — sampled indices
```

Nucleus sampling — samples from the minimal set of tokens whose cumulative probability exceeds `p`.

### `top_k_top_p_sampling_from_probs`

```python
torch.ops.aiter.top_k_top_p_sampling_from_probs(
    probs: Tensor,                          # [M, vocab_size], float32
    indices: Optional[Tensor],
    maybe_top_k_arr: Optional[Tensor],      # [M], int32
    top_k_val: int,
    maybe_top_p_arr: Optional[Tensor],      # [M], float32
    top_p_val: float,
    deterministic: bool = False,
) -> Tensor                                  # [M], int32
```

Joint filtering: accepts tokens only if **both** top-k count and top-p cumulative probability constraints are satisfied.

---

## 4. How Temperature, Top-K, and Top-P Interact

### Family A Pipeline (fused)

```
logits → [temperature + softmax + RNG + argmax] → token_id
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         Single fused kernel (no top-k/top-p)
```

### Family B Pipeline (composable)

```
logits
  → logits / temperature              # Caller applies
  → softmax(logits)                   # Caller applies
  → top_k_renorm_probs(probs, k)     # Zero out non-top-k
  → top_p_sampling_from_probs(probs, p)  # Sample with nucleus
```

Or use the joint variant:
```
  → top_k_top_p_sampling_from_probs(probs, k, p)  # Both at once
```

---

## 5. Supported Data Types

### Family A (logit-level)

| Input | Temperature | Output |
|-------|-------------|--------|
| float32, float16, bfloat16 | float32 | int32 |

### Family B (probability-level)

| Input | Top-k | Top-p | Output |
|-------|-------|-------|--------|
| float32 (auto-cast) | int32 | float32 | int32 |

---

## 6. Performance Notes

- **Family A fusion**: Avoids materializing the full softmax output — uses online softmax with running max. Single kernel = single global memory round-trip.
- **Block size**: All kernels use 1024 threads (max for AMD GPUs).
- **Vectorized loads**: 4–16 elements per thread per iteration.
- **Inner vs outer exponential**: Inner generates RNG in-kernel (saves a launch); outer takes pre-generated samples (useful for reproducibility).
- **Deterministic mode**: Family B supports `deterministic=True` for bitwise reproducibility using a custom Belloch-style scan (at slight performance cost).

---

## 7. Decision Tree

```
Need token sampling?
├── Single fused kernel (logits in, tokens out)?
│   ├── All greedy → aiter.greedy_sample()
│   ├── All random → aiter.random_sample()
│   ├── Mixed batch → aiter.mixed_sample()
│   └── Need reproducibility → *_outer_exponential() variants
├── Composable pipeline (probabilities)?
│   ├── Top-k only → top_k_renorm_probs() + multinomial
│   ├── Top-p only → top_p_sampling_from_probs()
│   └── Both → top_k_top_p_sampling_from_probs()
└── Pre-generate randomness?
    └── aiter.exponential()
```

---

## 8. Source Files

| Component | Path |
|---|---|
| Logit-level Python API | `aiter/ops/sample.py` |
| Probability-level Python API | `aiter/ops/sampling.py` |
| HIP sampling kernels | `csrc/kernels/sample_kernels.cu` |
| C++ sampling interfaces | `csrc/cpp_itfs/sampling/` |
| Sampling CUDA header | `csrc/cpp_itfs/sampling/sampling.cuh` |
| Pybind registration | `csrc/pybind/sample_pybind.cu` |

---

## 9. Test Files

| Test | Path |
|------|------|
| Logit-level sampling | `op_tests/test_sample.py` |
| Probability-level sampling | `op_tests/test_sampling.py` |

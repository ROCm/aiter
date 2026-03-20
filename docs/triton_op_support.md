# Triton Attention Operator Support Matrix

> References: [flash-attention](https://github.com/Dao-AILab/flash-attention) · [ROCm GPU specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) · [PyTorch FP8](https://pytorch.org/docs/stable/torch.html#torch.float8_e4m3fn) · [GQA paper](https://arxiv.org/abs/2305.13245) · [HSTU paper](https://arxiv.org/abs/2402.17152)

## Overview

| Operator | File | dtype | GPU Arch | Causal | GQA | FP8 | KV Cache | Dropout | Backward |
|----------|------|-------|----------|:------:|:---:|:---:|:--------:|:-------:|:--------:|
| **MHA** | [`mha.py`](../aiter/ops/triton/attention/mha.py) | fp16 · bf16 · fp32 · fp8 | gfx942 / gfx950 / other ROCm | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **MHA v3** | [`mha_v3.py`](../aiter/ops/triton/attention/mha_v3.py) | fp16 · bf16 · fp32 · fp8 | gfx942 / gfx950 / other ROCm | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **HSTU** | [`hstu_attention.py`](../aiter/ops/triton/attention/hstu_attention.py) | bf16 · fp16 | all ROCm⁷ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## MHA

**File**: [`aiter/ops/triton/attention/mha.py`](../aiter/ops/triton/attention/mha.py)

| Attribute | Description |
|-----------|-------------|
| **API** | `flash_attn_func` · `flash_attn_varlen_func` |
| **Input format** | Standard: `[B, S, H, D]`; varlen: `[T, H, D]`¹ |
| **dtype** | fp16 · bf16 · fp32; fp8² (raw tensor pass-through) |
| **GPU arch** | gfx942 / gfx950 (incl. FP8); other ROCm (fp16/bf16) |
| **Shape** | B/S: any positive integer; H % Hk == 0 (GQA/MQA); D ≥ 8 and power-of-2 aligned |
| **Causal masking** | ✅ |
| **GQA / MQA** | ✅ |
| **FP8** | ✅ raw tensor |
| **Paged attention** | ✅ |
| **Dropout** | ✅ |
| **ALiBi** | ✅ |
| **Attention sink** | ✅ |
| **PE split (NOPE+PE)** | ✅³ |
| **Return LSE / attn probs** | ✅ |
| **Backward** | ✅ standard + fused |
| **KV cache decode** | ❌ |
| **Rotary embedding** | ❌ |
| **Sliding window / Softcap** | ❌ |
| **Accuracy (fwd atol/rtol)** | fp16/bf16: 1e-2 / 1e-2; varlen: 1e-1 / 1e-1; fp8: 0.30 / 0.25 |
| **Accuracy (bwd atol/rtol)** | 1e-2 / 1e-2 |
| **Known issues on gfx942** | PE + Causal/Dropout ⛔; non-fused bwd (D=128, dropout=0.2, short seqlen) ⛔⁴; Sink bwd (batch>1, seqlen≥1024) ⚠️ |

---

## MHA v3

**File**: [`aiter/ops/triton/attention/mha_v3.py`](../aiter/ops/triton/attention/mha_v3.py)

| Attribute | Description |
|-----------|-------------|
| **API** | `flash_attn_func` · `flash_attn_varlen_func` · `flash_attn_fp8_func` · `flash_attn_varlen_fp8_func` · `flash_attn_with_kvcache` |
| **Input format** | Standard: `[B, S, H, D]`; varlen: `[T, H, D]`¹; fp8_func output is FP32 |
| **dtype** | fp16 · bf16 · fp32; fp8² (internally auto-quantized) |
| **GPU arch** | gfx942 / gfx950 (incl. FP8); other ROCm (fp16/bf16) |
| **Shape** | B/S: any positive integer; H % Hk == 0 (GQA/MQA); D ≥ 8 and power-of-2 aligned |
| **Causal masking** | ✅ |
| **GQA / MQA** | ✅ |
| **FP8** | ✅ auto-quantized API |
| **KV cache decode** | ✅ |
| **Paged attention** | ✅ |
| **Rotary embedding** | ✅ (kvcache path) |
| **Backward** | ✅ standard |
| **Dropout** | ❌ |
| **ALiBi** | ❌ |
| **Attention sink** | ❌ |
| **PE split (NOPE+PE)** | ❌ |
| **Return LSE / attn probs** | ❌ |
| **Sliding window / Softcap** | ❌ |
| **Accuracy (fwd atol/rtol)** | fp16/bf16: 1e-2 / 1e-2; varlen: 1e-1 / 1e-1; fp8: 0.30 / 0.25 |
| **Accuracy (bwd atol/rtol)** | 1e-2 / 1e-2 |

---

## HSTU Attention

**File**: [`aiter/ops/triton/attention/hstu_attention.py`](../aiter/ops/triton/attention/hstu_attention.py)

| Attribute | Description |
|-----------|-------------|
| **API** | `_AttentionFunction.apply` · `triton_hstu_attention_fwd` · `triton_hstu_attention_bwd` |
| **Input format** | Jagged only: Q/K `[T, H, D_attn]`, V `[T, H, D_v]`, `seq_offsets [B+1]`⁵ |
| **Activation** | SiLU: `Y = silu(alpha · Q @ Kᵀ) @ V`⁶ |
| **dtype** | bf16 (recommended) · fp16 |
| **GPU arch** | All ROCm (kernel has no arch restriction); gfx942 / gfx950 have pre-tuned configs⁷ |
| **Shape** | T = Σ seq_len; D_attn / D_v: any positive integer |
| **Causal masking** | ✅ |
| **Non-causal (bidirectional)** | ✅ |
| **Multiple targets** | ✅ (`num_targets` per seq — target token masking for recommendation systems) |
| **Max attention length** | ✅ (`max_attn_len` — limits maximum attention span) |
| **Contextual prefix** | ✅ (`contextual_seq_len` — fixed context prefix length) |
| **Sort by length** | ✅ (sort variable-length sequences for better GPU utilization) |
| **Backward** | ✅ standard + sequence parallel |
| **GQA / MQA** | ❌ (Q/K/V share the same H) |
| **FP8** | ❌ |
| **KV cache decode** | ❌ |
| **Dropout** | ❌ |
| **ALiBi / Softcap / Sliding window** | ❌ |
| **Accuracy (fwd atol/rtol)** | bf16: 1e-3 / 0 |

---

> ¹ varlen mode: T = total tokens, `[T,H,D]`; standard mode: `[B,S,H,D]`, B=batch, S=seqlen, H=num_heads, D=head_dim
> ² gfx942: `float8_e4m3fnuz` / `float8_e5m2fnuz`; gfx950: `float8_e4m3fn` / `float8_e5m2`
> ³ PE mode requires both NOPE and PE head_dim to be powers of 2, e.g. (128,64), (192,128)
> ⁴ ROCm 7.1+ regression; was working on ROCm 7.0
> ⁵ HSTU supports jagged (packed) format only; T = Σ(seq_len_i), seq_offsets holds cumulative offsets
> ⁶ alpha is typically set to `1/D_attn × 10000`
> ⁷ The Triton kernel itself has no arch restriction. The current limitation is missing pre-tuned JSON config files — only gfx942/gfx950 configs exist under `aiter/ops/triton/configs/hstu_attn/`. These configs contain ROCm-specific Triton parameters (`matrix_instr_nonkdim`, `kpack`). Other ROCm architectures can be supported by adding the corresponding config file.

# Triton Operator Support Matrix

## Overview

| Kernel | DSL | dtype | GPU Arch |
|--------|:---:|-------|----------|
| [`mha.py`](../aiter/ops/triton/attention/mha.py) `:: flash_attn_func / flash_attn_varlen_func` | Triton | fp16 · bf16 · fp32 · fp8 | all ROCm |
| [`mha.py`](../aiter/ops/triton/attention/mha.py) `:: flash_attn_with_kvcache` | Triton | fp16 · bf16 | all ROCm |
| [`mha_v3.py`](../aiter/ops/triton/attention/mha_v3.py) `:: flash_attn_func / flash_attn_varlen_func` | Triton | fp16 · bf16 · fp32 · fp8 | all ROCm |
| [`mha_v3.py`](../aiter/ops/triton/attention/mha_v3.py) `:: flash_attn_fp8_func / flash_attn_varlen_fp8_func` | Triton | bf16 · fp32 → fp8 (auto-quant) | gfx942 / gfx950 |
| [`mha_v3.py`](../aiter/ops/triton/attention/mha_v3.py) `:: flash_attn_with_kvcache` | Triton | fp16 · bf16 · fp8 | gfx942 / gfx950 |
| [`hstu_attention.py`](../aiter/ops/triton/attention/hstu_attention.py) `:: triton_hstu_attention_fwd / triton_hstu_attention_bwd` | Triton | bf16 · fp16 | all ROCm⁷ |
| [`pa_decode.py`](../aiter/ops/triton/attention/pa_decode.py) `:: paged_attention_decode` | Triton | fp16 · bf16 · fp8 · int8 | gfx942 / gfx950 |
| [`pa_prefill.py`](../aiter/ops/triton/attention/pa_prefill.py) `:: context_attention_fwd` | Triton | fp16 · fp8 | gfx942 / gfx950 |
| [`pa_decode_gluon.py`](../aiter/ops/triton/gluon/pa_decode_gluon.py) `:: pa_decode_gluon` | Gluon | fp8 · bf16 | gfx942 / gfx950 |
| [`fp8_mqa_logits.py`](../aiter/ops/triton/attention/fp8_mqa_logits.py) `:: fp8_mqa_logits` | Triton | fp8 | gfx942 / gfx950 |
| [`pa_mqa_logits.py`](../aiter/ops/triton/attention/pa_mqa_logits.py) `:: deepgemm_fp8_paged_mqa_logits` | Gluon | fp8 | gfx942 / gfx950 |

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

## PA Decode

**File**: [`aiter/ops/triton/attention/pa_decode.py`](../aiter/ops/triton/attention/pa_decode.py)

| Attribute | Description |
|-----------|-------------|
| **API** | `paged_attention_decode` |
| **Input format** | Q: `[B, H_Q, D]`, k_cache/v_cache: `[N, H_KV, BLK_SZ, D]` (paged layout) |
| **dtype** | fp16 · bf16 · fp8(e4m3fnuz) · int8 · bf16×fp8 · bf16×int8 |
| **GPU arch** | gfx942 / gfx950 |
| **Shape** | B: any; H_Q % H_KV == 0 (GQA/MQA); D: any; kv_block_size: any; D=128 + BLK=512 ⛔ (shared memory overflow) |
| **GQA / MQA** | ✅ |
| **FP8 KV Cache** | ✅ |
| **INT8 KV Cache** | ✅ |
| **Per-Token quant** | ✅ (FP16 query → FP8 KV) |
| **ALiBi** | ✅ (reserved) |
| **Causal masking** | ❌ |
| **Sliding Window** | ❌ |
| **Backward** | ❌ |
| **JIT** | ✅ |
| **AOT** | ❌ |
| **Accuracy (fwd)** | standard: atol/rtol 1e-2; per-token FP8: atol/rtol 2.5e-1 |
| **Known issues** | D=128 + BLK=512 ⛔ shared memory; B≥16 + SEQ≥8192 ⛔ skip (slow); GQA + per-token quant ⚠️ untested; per-token BF16×BF16 ⚠️ commented out |

---

## PA Prefill

**File**: [`aiter/ops/triton/attention/pa_prefill.py`](../aiter/ops/triton/attention/pa_prefill.py)

| Attribute | Description |
|-----------|-------------|
| **API** | `context_attention_fwd` |
| **Input format** | Q: `[T, H, D]`, K/V: `[T, H_KV, D]`, k_cache/v_cache: paged layout |
| **dtype** | fp16 query; KV cache: auto · fp8e4m3 · fp8e5m2 |
| **GPU arch** | gfx942 / gfx950 |
| **Shape** | H: any; H % H_KV == 0 (GQA/MQA); head_size: any; sliding_window ≥ 0 |
| **GQA / MQA** | ✅ |
| **FP8 KV Cache** | ✅ |
| **Causal masking** | ✅ |
| **Sliding Window** | ✅ |
| **ALiBi** | ✅ |
| **INT8 KV Cache** | ❌ |
| **Per-Token quant** | ❌ |
| **Backward** | ❌ |
| **JIT** | ✅ |
| **AOT** | ❌ |
| **Accuracy (fwd)** | atol/rtol 1e-2 |
| **Known issues** | BF16/INT8 query dtype ⚠️ untested (only fp16 tested) |

---

## PA Decode Gluon

**File**: [`aiter/ops/triton/gluon/pa_decode_gluon.py`](../aiter/ops/triton/gluon/pa_decode_gluon.py)

| Attribute | Description |
|-----------|-------------|
| **API** | `pa_decode_gluon` · `pa_decode_gluon_aot` |
| **Input format** | Q: `[B, H_Q, D]` (supports FP8), k_cache/v_cache: paged layout |
| **dtype** | FP8(e4m3fn) Q+KV · BF16 Q · BF16+FP8 KV (sliding window only) |
| **Quant mode** | per_token · per_tensor |
| **GPU arch** | gfx942 / gfx950 |
| **Shape** | D: any; H_Q % H_KV == 0; query_length: any; context_length: any; block_size: any |
| **GQA / MQA** | ✅ |
| **FP8 KV Cache** | ✅ |
| **Per-Token quant** | ✅ |
| **Per-Tensor quant** | ✅ |
| **Causal masking** | ✅ |
| **Sliding Window** | ✅ |
| **Attention Sinks** | ✅ |
| **KV Varlen** | ✅ |
| **JIT** | ✅ |
| **AOT** | ✅ |
| **Backward** | ❌ |
| **ALiBi** | ❌ |
| **Accuracy (fwd)** | custom `err_gluon` threshold |
| **Known issues** | Trans V layout ⚠️ only perf tested, accuracy not enabled |

---

## FP8 MQA Logits

**File**: [`aiter/ops/triton/attention/fp8_mqa_logits.py`](../aiter/ops/triton/attention/fp8_mqa_logits.py)

| Attribute | Description |
|-----------|-------------|
| **API** | `fp8_mqa_logits` |
| **Input format** | Q: `[S, H, D]` (FP8), KV: `[S_kv, D]` (FP8), kv_scales: `[S_kv]`, weights: `[S, H]` (FP32) |
| **Output format** | `[S, S_kv]` (FP32) — logits for sparse attention topk selection |
| **dtype** | fp8(e4m3fnuz) Q+KV, KV with per-token scale |
| **GPU arch** | gfx942 / gfx950 |
| **Shape** | S ≤ S_kv; num_heads: power of 2; head_dim: power of 2 |
| **MQA** | ✅ (all Q heads share single KV) |
| **FP8** | ✅ |
| **Per-Token quant** | ✅ |
| **Causal masking** | ✅ |
| **Context Parallel** | ✅ |
| **Sparse attention Logits** | ✅ |
| **JIT** | ✅ |
| **Backward** | ❌ |
| **Accuracy (fwd)** | cosine diff < 1e-3 |
| **Known issues** | s_q > s_k ⛔ skip; num_heads/head_dim must be power of 2 (assert) |

---

## PA MQA Logits (Paged)

**File**: [`aiter/ops/triton/attention/pa_mqa_logits.py`](../aiter/ops/triton/attention/pa_mqa_logits.py)

| Attribute | Description |
|-----------|-------------|
| **API** | `deepgemm_fp8_paged_mqa_logits` · `deepgemm_fp8_paged_mqa_logits_schedule` |
| **Description** | Paged version of FP8 MQA logits with Gluon JIT/AOT support |
| **dtype** | fp8(e4m3) Q+KV |
| **GPU arch** | gfx942 / gfx950 |
| **JIT** | ✅ |
| **AOT** | ✅ (via `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`) |

---

> ¹ varlen mode: T = total tokens, `[T,H,D]`; standard mode: `[B,S,H,D]`, B=batch, S=seqlen, H=num_heads, D=head_dim
> ² gfx942: `float8_e4m3fnuz` / `float8_e5m2fnuz`; gfx950: `float8_e4m3fn` / `float8_e5m2`
> ³ PE mode requires both NOPE and PE head_dim to be powers of 2, e.g. (128,64), (192,128)
> ⁴ ROCm 7.1+ regression; was working on ROCm 7.0
> ⁵ HSTU supports jagged (packed) format only; T = Σ(seq_len_i), seq_offsets holds cumulative offsets
> ⁶ alpha is typically set to `1/D_attn × 10000`
> ⁷ The Triton kernel itself has no arch restriction. The current limitation is missing pre-tuned JSON config files — only gfx942/gfx950 configs exist under `aiter/ops/triton/configs/hstu_attn/`. These configs contain ROCm-specific Triton parameters (`matrix_instr_nonkdim`, `kpack`). Other ROCm architectures can be supported by adding the corresponding config file.

# Triton Attention 算子支持矩阵

> 参考：[flash-attention](https://github.com/Dao-AILab/flash-attention) · [ROCm GPU 规格](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) · [PyTorch FP8](https://pytorch.org/docs/stable/torch.html#torch.float8_e4m3fn) · [GQA 论文](https://arxiv.org/abs/2305.13245) · [HSTU 论文](https://arxiv.org/abs/2402.17152)

## 概览

| 算子 | 文件 | dtype | GPU 架构 | Causal | GQA | FP8 | KV Cache | Dropout | 反向传播 |
|------|------|-------|---------|:------:|:---:|:---:|:--------:|:-------:|:-------:|
| **MHA** | [`mha.py`](../aiter/ops/triton/attention/mha.py) | fp16 · bf16 · fp32 · fp8 | gfx942 / gfx950 / 其他 ROCm | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **MHA v3** | [`mha_v3.py`](../aiter/ops/triton/attention/mha_v3.py) | fp16 · bf16 · fp32 · fp8 | gfx942 / gfx950 / 其他 ROCm | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **HSTU** | [`hstu_attention.py`](../aiter/ops/triton/attention/hstu_attention.py) | bf16 · fp16 | gfx942 / gfx950 | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## MHA

**文件**：[`aiter/ops/triton/attention/mha.py`](../aiter/ops/triton/attention/mha.py)

| 属性 | 说明 |
|------|------|
| **API** | `flash_attn_func` · `flash_attn_varlen_func` |
| **输入格式** | 标准：`[B, S, H, D]`；varlen：`[T, H, D]`¹ |
| **dtype** | fp16 · bf16 · fp32；fp8²（raw tensor 直接传入） |
| **GPU 架构** | gfx942 / gfx950（含 FP8）；其他 ROCm（fp16/bf16） |
| **Shape** | B/S 任意正整数；H % Hk == 0（GQA/MQA）；D ≥ 8 且对齐至 2 的幂 |
| **Causal masking** | ✅ |
| **GQA / MQA** | ✅ |
| **FP8** | ✅ raw tensor |
| **Paged attention** | ✅ |
| **Dropout** | ✅ |
| **ALiBi** | ✅ |
| **Attention sink** | ✅ |
| **PE 分离（NOPE+PE）** | ✅³ |
| **Return LSE / attn probs** | ✅ |
| **反向传播** | ✅ 标准 + fused |
| **KV cache decode** | ❌ |
| **Rotary embedding** | ❌ |
| **Sliding window / Softcap** | ❌ |
| **精度（fwd atol/rtol）** | fp16/bf16：1e-2 / 1e-2；varlen：1e-1 / 1e-1；fp8：0.30 / 0.25 |
| **精度（bwd atol/rtol）** | 1e-2 / 1e-2 |
| **gfx942 已知限制** | PE + Causal/Dropout ⛔；非 fused bwd（D=128，dropout=0.2，短序列）⛔⁴；Sink bwd（batch>1，seqlen≥1024）⚠️ |

---

## MHA v3

**文件**：[`aiter/ops/triton/attention/mha_v3.py`](../aiter/ops/triton/attention/mha_v3.py)

| 属性 | 说明 |
|------|------|
| **API** | `flash_attn_func` · `flash_attn_varlen_func` · `flash_attn_fp8_func` · `flash_attn_varlen_fp8_func` · `flash_attn_with_kvcache` |
| **输入格式** | 标准：`[B, S, H, D]`；varlen：`[T, H, D]`¹；fp8_func 输出为 FP32 |
| **dtype** | fp16 · bf16 · fp32；fp8²（内部自动量化） |
| **GPU 架构** | gfx942 / gfx950（含 FP8）；其他 ROCm（fp16/bf16） |
| **Shape** | B/S 任意正整数；H % Hk == 0（GQA/MQA）；D ≥ 8 且对齐至 2 的幂 |
| **Causal masking** | ✅ |
| **GQA / MQA** | ✅ |
| **FP8** | ✅ 自动量化 API |
| **KV cache decode** | ✅ |
| **Paged attention** | ✅ |
| **Rotary embedding** | ✅（kvcache 路径） |
| **反向传播** | ✅ 标准 |
| **Dropout** | ❌ |
| **ALiBi** | ❌ |
| **Attention sink** | ❌ |
| **PE 分离（NOPE+PE）** | ❌ |
| **Return LSE / attn probs** | ❌ |
| **Sliding window / Softcap** | ❌ |
| **精度（fwd atol/rtol）** | fp16/bf16：1e-2 / 1e-2；varlen：1e-1 / 1e-1；fp8：0.30 / 0.25 |
| **精度（bwd atol/rtol）** | 1e-2 / 1e-2 |

---

## HSTU Attention

**文件**：[`aiter/ops/triton/attention/hstu_attention.py`](../aiter/ops/triton/attention/hstu_attention.py)

| 属性 | 说明 |
|------|------|
| **API** | `_AttentionFunction.apply` · `triton_hstu_attention_fwd` · `triton_hstu_attention_bwd` |
| **输入格式** | 仅 jagged：Q/K `[T, H, D_attn]`，V `[T, H, D_v]`，`seq_offsets [B+1]`⁵ |
| **激活函数** | SiLU：`Y = silu(alpha · Q @ Kᵀ) @ V`⁶ |
| **dtype** | bf16（推荐）· fp16 |
| **GPU 架构** | gfx942 / gfx950（含预调参 config） |
| **Shape** | T = Σ seq_len；D_attn / D_v 任意正整数 |
| **Causal masking** | ✅ |
| **Non-causal（双向）** | ✅ |
| **Multiple targets** | ✅（`num_targets` per seq，用于推荐系统目标 token 掩码） |
| **Max attention length** | ✅（`max_attn_len`，限制最大注意力跨度） |
| **Contextual prefix** | ✅（`contextual_seq_len`，上下文前缀长度） |
| **Sort by length** | ✅（变长序列按长度排序以提升 GPU 利用率） |
| **反向传播** | ✅ 标准 + sequence parallel |
| **GQA / MQA** | ❌（Q/K/V 共享相同 H） |
| **FP8** | ❌ |
| **KV cache decode** | ❌ |
| **Dropout** | ❌ |
| **ALiBi / Softcap / Sliding window** | ❌ |
| **精度（fwd atol/rtol）** | bf16：1e-3 / 0 |

---

> ¹ varlen 模式：T = total tokens，`[T,H,D]`；标准模式：`[B,S,H,D]`，B=batch，S=seqlen，H=num_heads，D=head_dim
> ² gfx942：`float8_e4m3fnuz` / `float8_e5m2fnuz`；gfx950：`float8_e4m3fn` / `float8_e5m2`
> ³ PE 模式要求 NOPE 和 PE 的 head_dim 分别为 2 的幂，如 (128,64)、(192,128)
> ⁴ ROCm 7.1+ 回归，ROCm 7.0 正常
> ⁵ HSTU 仅支持 jagged（packed）格式；T = Σ(seq_len_i)，seq_offsets 为累积偏移量
> ⁶ alpha 通常设为 `1/D_attn × 10000`

# Triton Attention 算子支持矩阵

> 参考：[flash-attention](https://github.com/Dao-AILab/flash-attention) · [ROCm GPU 规格](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) · [PyTorch FP8](https://pytorch.org/docs/stable/torch.html#torch.float8_e4m3fn) · [GQA 论文](https://arxiv.org/abs/2305.13245) · [HSTU 论文](https://arxiv.org/abs/2402.17152)

| 属性 | **MHA**<br>[`mha.py`](../aiter/ops/triton/attention/mha.py) | **MHA v3**<br>[`mha_v3.py`](../aiter/ops/triton/attention/mha_v3.py) | **HSTU Attention**<br>[`hstu_attention.py`](../aiter/ops/triton/attention/hstu_attention.py) |
|------|------|------|------|
| **API** | `flash_attn_func`<br>`flash_attn_varlen_func` | `flash_attn_func`<br>`flash_attn_varlen_func`<br>`flash_attn_fp8_func`<br>`flash_attn_varlen_fp8_func`<br>`flash_attn_with_kvcache` | `_AttentionFunction.apply`<br>`triton_hstu_attention_fwd`<br>`triton_hstu_attention_bwd` |
| **输入格式** | `[B,S,H,D]` 或 `[T,H,D]`¹ | 同左；fp8_func 输出为 FP32 | 仅 jagged：Q/K `[T,H,D_attn]`，V `[T,H,D_v]`，`seq_offsets [B+1]`⁵ |
| **dtype** | fp16 · bf16 · fp32<br>fp8²（raw tensor） | fp16 · bf16 · fp32<br>fp8²（内部自动量化） | bf16（推荐）· fp16 |
| **GPU 架构** | gfx942 / gfx950（含 FP8）<br>其他 ROCm（fp16/bf16） | 同左 | gfx942 / gfx950（预调参） |
| **激活函数** | Softmax | Softmax | SiLU⁶ |
| **Shape** | B / S：任意正整数<br>H % Hk == 0（GQA/MQA）<br>D ≥ 8，对齐至 2 的幂 | 同左 | T = Σ seq_len；D_attn / D_v 任意正整数 |
| **Causal masking** | ✅ | ✅ | ✅ |
| **Non-causal（双向）** | ✅ | ✅ | ✅ |
| **GQA / MQA** | ✅ | ✅ | ❌（Q/K/V 共享 H） |
| **FP8** | ✅ raw tensor | ✅ 自动量化 API | ❌ |
| **KV cache decode** | ❌ | ✅ | ❌ |
| **Paged attention** | ✅ | ✅ | ❌ |
| **Rotary embedding** | ❌ | ✅（kvcache 路径） | ❌ |
| **Dropout** | ✅ | ❌ | ❌ |
| **ALiBi** | ✅ | ❌ | ❌ |
| **Attention sink** | ✅ | ❌ | ❌ |
| **PE 分离（NOPE+PE）** | ✅³ | ❌ | ❌ |
| **Return LSE / attn probs** | ✅ | ❌ | ❌ |
| **反向传播** | ✅ 标准 + fused | ✅ 标准 | ✅ 标准 + sequence parallel |
| **Multiple targets** | ❌ | ❌ | ✅（`num_targets` per seq） |
| **Max attn length** | ❌ | ❌ | ✅（`max_attn_len`） |
| **Contextual prefix** | ❌ | ❌ | ✅（`contextual_seq_len`） |
| **Sort by length** | ❌ | ❌ | ✅（变长序列效率优化） |
| **Sliding window / Softcap** | ❌ | ❌ | ❌ |
| **精度（atol / rtol）** | fp16/bf16 fwd：1e-2 / 1e-2<br>varlen fwd：1e-1 / 1e-1<br>fp8 fwd：0.30 / 0.25<br>bwd：1e-2 / 1e-2 | 同左 | bf16 fwd：1e-3 / 0 |
| **gfx942 已知限制** | PE + Causal/Dropout ⛔<br>非 fused bwd（D=128，dropout=0.2，短序列）⛔⁴<br>Sink bwd（batch>1，seqlen≥1024）⚠️ | — | — |

> ¹ varlen 模式：T = total tokens，`[T,H,D]`；标准模式：`[B,S,H,D]`，B=batch，S=seqlen，H=num_heads，D=head_dim
> ² gfx942：`float8_e4m3fnuz` / `float8_e5m2fnuz`；gfx950：`float8_e4m3fn` / `float8_e5m2`
> ³ PE 模式要求 NOPE 和 PE 的 head_dim 分别为 2 的幂，如 (128,64)、(192,128)
> ⁴ ROCm 7.1+ 回归，ROCm 7.0 正常
> ⁵ HSTU 仅支持 jagged（packed）格式；T = Σ(seq_len_i)，seq_offsets 为累积偏移量
> ⁶ HSTU 激活：`Y = silu(alpha · Q @ Kᵀ) @ V`，alpha 通常设为 `1/D_attn × 10000`

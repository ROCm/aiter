# MHA 算子支持说明

基于 [flash-attention](https://github.com/Dao-AILab/flash-attention) 风格实现的 Triton MHA 算子，针对 AMD CDNA GPU 优化。

## 实现文件

| 文件 | 功能 |
|------|------|
| [`mha.py`](../aiter/ops/triton/attention/mha.py) | 标准/变长 attention，支持 dropout、ALiBi、sink、PE |
| [`mha_v3.py`](../aiter/ops/triton/attention/mha_v3.py) | FP8 高精度 API、KV cache decode、rotary embedding |

## 公开 API

| 函数 | 来源 | 输入格式 | 输出格式 |
|------|------|---------|---------|
| `flash_attn_func` | mha / mha_v3 | q/k/v: `[B, S, H, D]` | `[B, S, H, D]` |
| `flash_attn_varlen_func` | mha / mha_v3 | q: `[total_q, H, D]`，k/v: `[total_k, Hk, D]` | `[total_q, H, D]` |
| `flash_attn_fp8_func` | mha_v3 | q/k/v: `[B, S, H, D]`（FP16/BF16/FP32，内部自动量化至 FP8） | `[B, S, H, D]`（FP32） |
| `flash_attn_varlen_fp8_func` | mha_v3 | q: `[total_q, H, D]`，k/v: `[total_k, Hk, D]`（FP16/BF16/FP32） | `[total_q, H, D]`（FP32） |
| `flash_attn_with_kvcache` | mha_v3 | q + k_cache/v_cache，支持 rotary、paged KV | `[B, S, H, D]` |

> tensor 维度含义：B=batch，S=seqlen，H=num_heads，D=head_dim。GQA/MQA 时 k/v 的 H 记为 Hk（需满足 H % Hk == 0），参考 [GQA 论文](https://arxiv.org/abs/2305.13245)。

## 支持的 dtype

| dtype | mha.py | mha_v3.py | 备注 |
|-------|:------:|:---------:|------|
| `float16` | ✅ | ✅ | mha.py 默认，BLOCK_M=128/BLOCK_N=64 |
| `bfloat16` | ✅ | ✅ | |
| `float32` | ✅ | ✅ | mha.py 自动降为 BLOCK_M=32/BLOCK_N=32 |
| `float8_e4m3fnuz` | ✅ | ✅（内部） | 仅 [gfx942](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)（MI300X） |
| `float8_e4m3fn` | ✅ | ✅（内部） | 仅 gfx950（MI350X） |
| `float8_e5m2fnuz` / `e5m2` | ✅ | — | 需直接传入 FP8 tensor（mha.py raw 模式） |

> FP8 格式说明参考 [PyTorch FP8 docs](https://pytorch.org/docs/stable/torch.html#torch.float8_e4m3fn)。mha_v3 的 FP8 API 接受高精度输入并在内部自动完成量化/反量化。

## GPU 架构与 FP8 支持

| 架构 | gfx | FP8 | FP16/BF16 |
|------|-----|:---:|:---------:|
| MI300X (CDNA3) | gfx942 | ✅ | ✅ |
| MI350X (CDNA4) | gfx950 | ✅ | ✅ |
| 其他 ROCm 架构 | — | ❌ | ✅ |

ROCm GPU 规格详见 [GPU hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)。

**gfx942 已知限制：**

- Positional Encoding + Causal 或 Dropout：⛔ 暂不支持
- 反向传播（非 fused，HEAD_SZ=128，DROPOUT=0.2，短序列）：⛔ ROCm 7.1+ 回归（ROCm 7.0 正常）
- Sink 反向，batch > 1 且 seqlen ≥ 1024：⚠️ 精度容忍度放宽

## Shape 约束

| 参数 | 约束 |
|------|------|
| `batch_size` (B) | 任意正整数 |
| `seqlen_q / seqlen_k` | 任意正整数，测试覆盖 1~4096 |
| `num_q_heads` (H) | H % Hk == 0（[GQA/MQA](https://arxiv.org/abs/2305.13245)） |
| `head_dim` (D) | ≥ 8，内部对齐至 2 的幂（min 16） |
| PE 模式 `head_dim` | NOPE 和 PE 部分均须为 2 的幂，如 (128,64)、(192,128) |

## 功能矩阵

| 功能 | mha.py | mha_v3.py |
|------|:------:|:---------:|
| Causal masking | ✅ | ✅ |
| GQA / MQA | ✅ | ✅ |
| FP8（自动量化） | ❌ | ✅ |
| FP8（raw tensor） | ✅ | ✅（descale 参数） |
| KV cache decode | ❌ | ✅ |
| Paged attention | ✅ | ✅ |
| Rotary embedding | ❌ | ✅（kvcache 路径） |
| Dropout | ✅ | ❌ |
| ALiBi slopes | ✅ | ❌ |
| Attention sink | ✅ | ❌ |
| 位置编码分离（NOPE+PE） | ✅ | ❌ |
| Return LSE / attn probs | ✅ | ❌ |
| 反向传播 | ✅（标准 + fused） | ✅（标准） |
| Sliding window | ❌ | ❌ |
| Softcap / num_splits / pack_gqa | ❌ | ❌ |

## 精度容忍度

| 模式 | atol | rtol |
|------|------|------|
| FP16/BF16 前向 | 1e-2 | 1e-2 |
| FP16 varlen 前向 | 1e-1 | 1e-1 |
| [FP8](https://pytorch.org/docs/stable/torch.html#torch.float8_e4m3fn) 前向 | 0.30 | 0.25 |
| FP16/BF16 反向 | 1e-2 | 1e-2 |

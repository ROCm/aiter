# MHA 算子支持说明

## 源文件

| 文件 | 说明 |
|------|------|
| `aiter/ops/triton/attention/mha.py` | 主实现：标准/变长 attention，支持 dropout、ALiBi、sink、PE |
| `aiter/ops/triton/attention/mha_v3.py` | V3 实现：FP8 高精度 API、KV cache decode、rotary embedding |

---

## 公开 API

### `mha.py`

| 函数 | 输入 tensor 格式 | 输出格式 |
|------|-----------------|---------|
| `flash_attn_func` | q/k/v: `[B, S, H, D]` | `[B, S, H, D]` |
| `flash_attn_varlen_func` | q: `[total_q, H, D]`，k/v: `[total_k, Hk, D]` | `[total_q, H, D]` |

### `mha_v3.py`

| 函数 | 输入 tensor 格式 | 输出格式 |
|------|-----------------|---------|
| `flash_attn_func` | q/k/v: `[B, S, H, D]` | `[B, S, H, D]` |
| `flash_attn_varlen_func` | q: `[total_q, H, D]`，k/v: `[total_k, Hk, D]` | `[total_q, H, D]` |
| `flash_attn_fp8_func` | q/k/v: `[B, S, H, D]`（FP16/BF16/FP32 输入，内部自动量化） | `[B, S, H, D]`（FP32） |
| `flash_attn_varlen_fp8_func` | q: `[total_q, H, D]`，k/v: `[total_k, Hk, D]`（FP16/BF16/FP32） | `[total_q, H, D]`（FP32） |
| `flash_attn_with_kvcache` | q + k_cache/v_cache，支持 rotary | `[B, S, H, D]` |

---

## 支持的 dtype

### mha.py（直接传入原始 tensor）

| dtype | 支持 | 备注 |
|-------|------|------|
| `torch.float16` | ✅ | 默认，BLOCK_M=128/BLOCK_N=64 |
| `torch.bfloat16` | ✅ | |
| `torch.float32` | ✅ | 自动降为 BLOCK_M=32/BLOCK_N=32 |
| `torch.float8_e4m3fnuz` | ✅ | 仅 gfx942，需直接传 FP8 tensor |
| `torch.float8_e4m3fn` | ✅ | 仅 gfx950 |
| `torch.float8_e5m2fnuz` | ✅ | 仅 gfx942 |
| `torch.float8_e5m2` | ✅ | 仅 gfx950 |

### mha_v3.py FP8 高精度 API（`flash_attn_fp8_func` / `flash_attn_varlen_fp8_func`）

| 输入 dtype | 内部计算 dtype | 输出 dtype |
|-----------|-------------|----------|
| `float16` / `bfloat16` / `float32` | `float8_e4m3fnuz`（gfx942）或 `float8_e4m3fn`（gfx950） | `float32` |

---

## GPU 架构支持

| 架构 | FP8 可用 | FP8 dtype | 备注 |
|------|---------|-----------|------|
| **gfx942**（MI300X） | ✅ | `e4m3fnuz` / `e5m2fnuz` | kernel 已为该架构调优；部分已知问题见下方 |
| **gfx950**（MI350X） | ✅ | `e4m3fn` / `e5m2` | |
| 其他 ROCm 架构 | ❌ | — | FP16/BF16 可用，FP8 不可用 |

### gfx942 已知限制

| 场景 | 状态 |
|------|------|
| Positional Encoding + Causal | ⛔ skip（暂不支持） |
| Positional Encoding + Dropout | ⛔ skip（暂不支持） |
| 反向传播，非 fused，HEAD_SZ=128，DROPOUT=0.2，短序列 | ⛔ skip（ROCm 7.1+ 回归，ROCm 7.0 正常） |
| Sink + 反向，大 batch（>1）且长序列（≥1024） | ⚠️ 放宽精度容忍度 |

---

## Shape 约束

| 参数 | 约束 | 测试范围 |
|------|------|---------|
| `batch_size` (B) | 任意正整数 | 1, 3, 4, 57, 128 |
| `seqlen_q` | 任意正整数 | 1~4096 |
| `seqlen_k` | 任意正整数 | 1~4096 |
| `num_q_heads` (H) | 必须整除 `num_kv_heads`（GQA/MQA） | 1, 2, 4, 8, 16, 48, 64, 128 |
| `num_kv_heads` (Hk) | ≤ H，H 可被 Hk 整除 | 1, 4, 8, 16 |
| `head_dim` (D) | ≥ 8，内部向上对齐到 2 的幂（min 16） | 8, 32, 64, 96, 128, 192 |
| PE 模式 head_dim | NOPE 和 PE 部分均须为 2 的幂（无需 padding） | (128,64), (192,128), (96,64) |

---

## 功能矩阵

| 功能 | mha.py | mha_v3.py |
|------|:------:|:---------:|
| Causal masking | ✅ | ✅ |
| GQA / MQA | ✅ | ✅ |
| Dropout | ✅ | ❌ |
| Return LSE | ✅ | ❌ |
| Return attn probs (softmax mask) | ✅ | ❌ |
| ALiBi slopes | ✅ | ❌ |
| Attention sink | ✅ | ❌（反向不支持） |
| 位置编码分离（NOPE + PE head） | ✅ | ❌ |
| FP8 自动量化 API | ❌ | ✅ |
| FP8 原始 tensor 输入 | ✅ | ✅（通过 descale 参数） |
| KV cache decode | ❌ | ✅ |
| Paged attention（block_table） | ✅ | ✅ |
| Rotary embedding | ❌ | ✅（仅 kvcache 路径） |
| 反向传播（标准） | ✅ | ✅ |
| 反向传播（fused kernel） | ✅ | — |
| Sliding window | ❌（已预留但未实现） | ❌ |
| Softcap | ❌ | ❌ |
| num_splits > 1 | ❌ | ❌ |
| pack_gqa | ❌ | ❌ |
| sm_margin != 0 | ❌ | ❌ |

---

## 精度容忍度（测试基准）

| 模式 | atol | rtol |
|------|------|------|
| FP16/BF16 前向 | 1e-2 | 1e-2 |
| FP16 varlen 前向 | 1e-1 | 1e-1 |
| FP8 前向 | 0.30 | 0.25 |
| FP16/BF16 反向 | 1e-2 | 1e-2 |

# MLA v4 nm — stage1 多 split partial 污染问题 排查报告

- 内核：`aiter.mla.mla_decode_fwd_v4_nm`（gfx1250 v4 nm，asm kernel `mla_a8w8_qh64_1tg_16mx4_64nx1_sparse.co`）
- 变体：`qh64-q1-16mx4-64nx1-np`（`gqa_ratio=64`, `q_seq_logical=1`, `num_kv_heads=1`）
- 测试脚本：`aiter/op_tests/test_mla_v4_kargpreld.py`
- 探针脚本：`aiter/op_tests/_stage1_probe.py`（临时诊断工具）
- 运行环境：容器 `ff_mla`，`ENABLE_CK=0`

---

## 1. 复现配置

`test_mla_v4_kargpreld.py` 中的扫描参数：

```python
_gfx1250_CTX_LENS       = [271]   # kv_seq_lens
_gfx1250_BATCH_SIZES    = [60]    # 触发；[30] 不触发
_gfx1250_SPLIT_PER_BATCH = [8]    # num_kv_splits
```

运行命令：

```bash
ENABLE_CK=0 python op_tests/test_mla_v4_kargpreld.py
```

精度判据：`checkAllclose(atol=0.03, rtol=0.03, tol_err_ratio=0.02)`，即不匹配元素比例 > 2% 判 FAILED。

---

## 2. 现象总览

| 现象 | 结论 |
|---|---|
| 稳定复现 | 是，跨进程稳定复现，但**幅度非确定性** |
| 参考侧 | `golden_bf16 vs fp8_ref` 恒定干净（约 14 元素、delta 0.055）—— torch 参考无辜 |
| 污染位置 | stage1 输出的 **有效 split 1/2/3**（中间 split） |
| 污染值 | 恒为 **≈ `2.94e+35`** 的非法巨值（非 NaN/Inf，而是溢出/未初始化量级） |
| 头尾 split | split 0、split 4 在**所有 batch 上恒正确** |
| 无效 split | split 5/6/7（`valid_split_count=5`）—— 开关开时残留旧数据，开关关时被清 0 |
| batch 依赖 | **batch=30 不触发；batch=60 触发** |
| 分布特征 | 稀疏、随机（部分 batch + 部分中间 split + 少量 cell） |
| 根因推断 | interior split partial 写回缺同步的**竞争条件（race）** |

---

## 3. 现象一：稳定复现、幅度非确定性（10 次进程级测试）

- `golden_bf16 vs fp8_ref`（参考自检）：每次都稳定干净 → 参考侧没问题。
- `v4_nm [fp8_dequant_ref vs asm]`（ASM 输出对比）：每次都不匹配，且幅度随机。

典型 10 次结果（batch=60）：

| 指标 | 观测范围 |
|---|---|
| 不匹配比例 | 0.2% ~ 8%，间歇性越过 2% 阈值 → 有时 warning，有时 **FAILED** |
| max abs delta | 常见 **1e30 ~ 1e35 量级巨值**（正常输出应 ~1） |

即：错误**稳定复现**，但每次污染幅度/比例都不同，是典型的非确定性数值污染。

---

## 4. 现象二：孤立单次调用干净，负载下才污染

stage1 探针（预填哨兵值 + 逐 split 分类）：

- **孤立单次调用**：stage1 完全干净——split 0~4 写入正常值（~±3），无 nan/inf/巨值；split 5~7 未写入。
- **`run_perftest` / 重复调用路径下**：stage1 出现污染。

说明：污染只在多次背靠背 launch / 真实显存压力下暴露，单次隔离调用不出现 → 状态相关，指向竞争 / 未初始化读。

---

## 5. 现象三：污染精确定位 —— "头尾对、中间错"

逐 split 的 `absmax`（run_perftest 路径，典型值）：

```
split :   0        1         2         3        4      | 5   6   7
absmax:  2.2    2.94e+35  2.94e+35  2.94e+35  3.0     | 0 / 0 / 0
        ─┬─    ─────────── 中间三个坏 ───────────    ─┬─
        首端                                          尾端
```

- **两端**（第一个有效 split 0、最后一个有效 split 4）：数值正确、稳定。
- **中间**（split 1/2/3）：被 `≈2.94e+35` 的非法巨值污染。
- 污染值 `2.94e+35` 在多次运行、多个 split 间高度一致（非随机内存噪声）。

诊断价值：很多 kernel 对首段/尾段做特殊边界处理、中间段走统一循环。"头尾对、中间错"把嫌疑缩小到 **stage1 处理 interior split 的写回/累加代码**。

---

## 6. 现象四：无效 split 5~7 与 `valid_split_count`

### KV-split 机制

271 个 KV token 被切成最多 8 个 split，每个 split 由不同 workgroup 算一份 partial（stage1，写入 `logits[total_q, 8, num_heads, v_head_dim]`），再由 stage2 合并。

### valid_split_count = 5

受 kernel 粒度限制，271 token 实际只填满 **5 个 split**，于是 `valid_split_count=5`，stage2 只合并前 5 个（0~4），跳过 5/6/7。

### split 5~7 的两种状态

`logits` 由 `torch.empty` 分配（不清零）。split 5~7 是否被写取决于 `use_valid_split_count_reduce` 开关：

| 开关状态（`aiter/mla.py` v4_nm 路径 line ~1315） | split 5~7 状态 |
|---|---|
| ON：`use_valid_split_count_reduce = int(num_kv_splits>1)` | **未写入**，保留 `torch.empty` 残留（值为 0 或 `7.16e-43`） |
| OFF：`use_valid_split_count_reduce = 0` | **被 kernel 主动清 0**（每个 split 全部 cell = 0） |

### `7.16e-43` 的来历

不是随机噪声，而是**小整数被按 float32 比特重新解释**：整数 `512`（0x00000200）当 float32 读 = `512 × 2⁻¹⁴⁹ ≈ 7.17e-43`（denormal）。说明那块显存里之前躺着一个 ≈512 的整数（元数据残留）。坐实"未初始化显存旧数据"。

---

## 7. 现象五：关闭 `valid_split_count` 开关的影响

在 `aiter/mla.py` v4_nm 路径强制 `use_valid_split_count_reduce = 0` 后（注意：V3 路径的 line 326 测试用不到，真正生效的是 v4_nm 路径 line ~1315）：

- **split 5~7 被完整清 0**（预填哨兵值全被覆盖为 0，`zeros = 每split全部cell`）。
- **但 interior split 1/2/3 的污染依旧存在**（run_perftest 路径下 huge > 0，absmax 仍 `2.94e+35`）。

结论：关开关只改变"无效 split 是否清 0"，**不修复** interior split 污染 —— 根因与该开关无关。

---

## 8. 现象六：batch 规模依赖

同为 kv=271、split=8、开关关闭：

| batch | 完整测试结果 | stage1 split 1/2/3 |
|---|---|---|
| **30** | 5/5 **passed** | 干净（huge=0，含 run_perftest 路径） |
| **60** | 间歇 **FAILED** | 被 `2.94e+35` 污染 |

结论：污染**与 batch 规模相关**，batch=30 不触发、batch=60 触发。指向 workgroup 数量 / CU 占用 / launch geometry（`gdx = ceil(gqa*max_seqlen_q/64) × batch × split` 的 workgroup 总数）越过某临界规模后才暴露的竞争或越界。

---

## 9. 现象七：逐 batch × split 的污染分布（batch=60）

对 60 个 batch 行（`total_q = batch × q_seq_logical = 60`）逐一统计每个 split 的 huge cell 数：

### 头尾恒对
- **split 0 列：60/60 全 0**
- **split 4 列：60/60 全 0**
- split 5/6/7：全 0（无效，已清 0）

### 中间随机、部分污染
| split | 被污染 batch 数 | 干净 batch 举例 |
|---|---|---|
| split1 | **51 / 60** | row 2,3,4,9,10,15,23,45,59 |
| split2 | **47 / 60** | row 2,3,4,5,8,9 … |
| split3 | **55 / 60** | row 11,15,21,23,53 |

进一步：
- 几乎每个 batch（60/60 行）**至少有一个**中间 split 被污染，但**不是三个中间 split 全坏**（如 row 2 只有 split3 坏；row 11 只有 split1 坏；row 23 只有 split2 坏）。
- 污染**稀疏**：单个 (batch, split) 内被污染 cell 数从 **2 到 ~700** 不等，最多仅占该 split 的 ~2%（每 split 每 batch 共 64×512=32768 cell）。

---

## 10. 根因推断

综合以上：

1. 污染只打中 **interior split（1/2/3）**，头尾（0/4）恒对 → 边界代码正确，问题在中间段统一写回逻辑。
2. 污染**稀疏、随机**（每次坏的 batch/split/cell 都不同、只有一小撮）→ **不是确定性索引/写覆盖 bug**（那会每次固定坏同一批位置且整段坏）。
3. **状态相关**：单次隔离干净，重复 launch / 大 batch 才触发。
4. 与 `valid_split_count` 开关无关。

→ **强烈指向竞争条件（race）**：多个 workgroup 对 interior split partial 的写回 / 累加缺少同步（缺 barrier，或对同一 LDS / 寄存器 / 全局区域存在跨 wave 的读写竞争），在 batch 规模大（workgroup 多）时被触发，随机污染少量 cell，残留出溢出量级的 `2.94e+35`。

---

## 11. 复现与探针工具

### 完整测试（10 次看稳定性）
```bash
docker exec ff_mla bash -lc 'cd /home/amd/feifei/aiter && \
  for i in $(seq 1 10); do \
    ENABLE_CK=0 python -u op_tests/test_mla_v4_kargpreld.py 2>&1 \
      | grep -E "fp8_dequant_ref vs asm|max abs delta.*elements"; \
  done'
```

### stage1 探针（`aiter/op_tests/_stage1_probe.py`）
- `probe`：预填哨兵 + 逐 split 分类 nan/inf/巨值/未写入
- `leak`：给无效 split 灌毒（1e35/NaN/Inf），验证 stage2 是否泄漏（结论：不泄漏）
- `perf`：复刻 `run_perftest` 路径并对齐 fp8 参考
- `sync`：对比 per-iter sync / 背靠背

```bash
ENABLE_CK=0 python -u op_tests/_stage1_probe.py perf 8
```

---

## 12. 建议后续

1. 扫临界 batch（如 31 / 40 / 48 / 56），卡出触发阈值，佐证 launch geometry / 资源规模相关。
2. 进入 `csrc/py_itfs_cu/asm_mla_v4.cu` 与 `.s`/`.co`，检查 interior split partial 写回是否缺 barrier、是否存在跨 wave 共享写 / LDS 未初始化读。
3. 对照 `num_kv_splits=1`（单 split 直出）是否恒定正确，进一步锁死到多 split 写回路径。

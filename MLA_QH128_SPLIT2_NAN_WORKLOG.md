# MLA qh128 decode `num_kv_splits=2` NaN 排查报告

## 1. 背景 / 问题现象

- 模型：DeepSeek-R1（fp8 MLA decode，nhead=128），ATOM 起服务、lm-eval 跑 gsm8k（3-shot）。
- 现象：端到端精度随 `num_kv_splits` 显著变化：
  - `num_kv_splits=1`：gsm8k ≈ **0.944**
  - `num_kv_splits=2`：gsm8k ≈ **0.879**（下降约 6.5 个点）
- 调用入口：`app/ATOM/atom/model_ops/attention_mla.py` 的非 persistent 路径，硬编码
  `mla_decode_fwd(..., num_kv_splits=2, ...)`（约 1221 行）。

根因结论（先给）：**aiter 的 qh128 decode kernel 在 `num_kv_splits>1`（多 pass）路径上，
对部分真实序列会产出 NaN/Inf**，污染对应序列的注意力输出，导致端到端精度回退。
这不是空 split 问题，也不是 .co 过期问题，而是多 pass「跨页 strided 分配 + 尾页处理」路径的缺陷。

## 2. 复现方法（离线单测）

### 2.1 采集真实 decode 输入

ATOM 服务侧已有 dump 钩子：

- `ATOM_DEBUG_MLA_SEG=1` → `_dump_seg_decode_failure`，文件名 `seg_decode_OK_layer000_step*.pt`
  （连续步、bs=64 真实批量，本次用的就是这批）。
- 或 `ATOM_DUMP_MLA_DECODE=1` + `ATOM_MLA_DUMP_MIN_PAGES=33`（见 `atom/utils/mla_decode_dump.py`，
  新增的长上下文门控）→ 文件名 `mla_decode_layer*_step*_rank*.pt`。

dump 默认输出目录 `~/mla_decode_dump`（可用 `MLA_DUMP_DIR` 覆盖）。

### 2.2 回放对比脚本

`app/aiter/op_tests/replay_mla_split_compare.py`：对每个 dump，用**完全相同的输入**跑
`aiter.mla.mla_decode_fwd` 两次，分别强制 `num_kv_splits=1` 与 `=2`
（通过显式传 `num_kv_splits_indptr` 绕过 `get_meta_param` 的 clamp），比较两者输出，
并报告 finite 与逐 batch 的 NaN 诊断（兼容 `seg_decode_*` 与 `mla_decode_*` 两种 schema）。

运行（容器内）：

```bash
cd /app/aiter && python op_tests/replay_mla_split_compare.py ~/mla_decode_dump
```

#### 2.2.1 stage1-partial NaN 探针（新增）

为把 NaN **决定性地**归因到 stage1（asm）还是 stage2（triton reduce），脚本新增
`_stage1_partials()`：对每个 dump 在 `num_kv_splits=2` 下**只跑 stage1**
（`aiter.mla_decode_stage1_asm_fwd`，参数顺序与 `mla.py` 完全一致），返回每个 split 的
`logits`/`attn_lse` partial 缓冲。缓冲先用「空 split 安全哨兵」预置（`logits=0`、
`attn_lse=-1e20`），因此 stage1 跑完后**任何非有限值都只可能是 stage1 主动写入的**
（既排除未初始化读取，也排除 stage2）。

输出新增：
- 每条 dump 若 stage1 partial 含 NaN，打印 `logits_bad_splits` / `lse_bad_splits`（哪个 split 槽出错）；
- 汇总行 `stage1-partial probe: N/M dumps have NaN in stage1 ... BEFORE stage2`，
  以及「输出为 NaN 的 dump 中有多少在 stage1 partial 已是 NaN」。
- 预期：**stage1 partial 已 NaN ⇒ stage2 reduce 与 seg 布局被排除，根因坐实在 asm stage1 多 pass 路径**。

### 2.3 观测到的结果

```
bs=64, pg(最大页数)=15~16
fin1=True（split=1 全部有限）
fin2=False（split=2 全部出现 NaN），non-finite split2 outputs: 40/40
逐 batch 诊断：nan_batches=16/64
  nan_tok=699..872   nan_last=[3,4,5,8,9,20,26,30,31,40,41,42,59,60]
  ok_tok =715..924   ok_last =[1,4,5,6,7,...,64]
```

关键观察：
- **同一份输入下，split=1 有限、split=2 NaN** → 是 split 路径本身的问题。
- NaN 是**固定的 16/64 条序列**（随解码步推进，`nan_tok`/`nan_last` 整体 +1，是同一批序列在累积）。
- **非单调**：更长的序列(872 tok)反而 NaN，更短的(715 tok)反而 OK
  → 排除「第二个 split 为空」的经典空 split 解释，指向尾页/边界处理。

## 3. 关键事实核对

### 3.1 不是部署了旧 .co

```
deployed aiter qh128 (/app/aiter/hsa/gfx1250/mla/...co):  e6234ec0d15a5f06fb8d2023c99576ea
poc_kl 最新构建 qh128.co:                                  e6234ec0d15a5f06fb8d2023c99576ea  # 完全一致
```

部署的就是最新（含「空 split 修复」的 .s）编出来的，**重编/重部署不会改变结果**。

### 3.2 qh128 与 qh64 是同一份代码

`diff mla_a8w8_qh128_1tg_16mx4_64nx1_np.s mla_a8w8_qh64_1tg_16mx4_64nx1_np.s` 仅 12 行差异，
全部是符号名 `qh128`↔`qh64`（含 mangling 长度前缀 `32`↔`31`），**GPU 机器码逐字节相同**。
qh64 的 `.sp3`（`mla_a8w8_qh64_1tg_16mx4_64nx1_np.sp3`）即两者共同源。
因此「空 split 修复」（worklog scope 写的 qh16/qh64）实际也在 qh128 中。

### 3.3 dtype / 格式匹配正常

非 persistent 路径（ATOM 用的这条）：
- `num_kv_splits==1` 且 fp8：`logits` 别名输出 `o`（bf16），stage1 直接写最终结果，不经 reduce。
- `num_kv_splits>1`：`logits`/`attn_lse` 为独立 **fp32** 缓冲（每 split 的 partial + LSE），
  stage1 写 fp32 partial → stage2 `_fwd_kernel_stage2_asm` 做 online-softmax 合并 → 写回 `o`（bf16）。

fp32 中间量是设计如此（多 split 累加需 fp32），**无格式/类型不匹配**。NaN 是 kernel 计算出来的值。

### 3.4 主机侧预初始化（C-a）无效

在 `aiter/mla.py` 中，`num_kv_splits>1` 时 stage1 前预置 `logits.fill_(0)`、`attn_lse.fill_(-1e20)`，
结果**完全不变**（仍 40/40 NaN，nan 序列/数值逐字节一致）。
→ 说明 stage1 **主动往第 2 个 split 槽写入了 NaN**（不是读未初始化内存），主机 buffer 兜底会被覆盖。
（注：该 C-a 改动目前仍保留在 `app/aiter/aiter/mla.py`，验证纯 kernel 修复时建议先回退它。）

### 3.5 stage2 reduce 与 ATOM 接口排查（确认无关）

复核了 reduce（stage2）以及 ATOM 主机侧接口，二者**不是根因，无需为 seg 布局改动**：

- **stage2 kernel 对 seg 布局无感**（`aiter/mla.py` 行 333–359，`_fwd_kernel_stage2_asm`）：
  它只消费 stage1 产出的 `logits`/`attn_lse`（标准 `[total_s, n_splits, nhead, v_head_dim]`
  / `[..., 1]` 布局）做 online-softmax 合并，**完全不直接读 seg KV cache**。
  入参里只有 `kv_indptr`/`kv_last_page_lens`/`num_kv_splits_indptr` 与 `page_size`、
  `KV_INDPTR_IS_PAGE_LEVEL=page_size>1`，没有任何 nope/pe 段偏移逻辑——
  seg 与非 seg 在 stage2 看来是同一份中间张量。
- **stage1-partial 探针（§2.2.1）已证实**：split=2 输出的 NaN 在 **stage2 之前**就存在于
  partial 缓冲里，stage2 只是把已是 NaN 的 partial 合并下去 → **stage2 不产生、也无法修复 NaN**。
- **ATOM 接口结构正确**：`attention_mla.py` 用 `_seg_kv_cache_view` 正确 reshape seg KV，
  以正确 `page_size`/`kv_indices` 调用 `mla_decode_fwd`；`num_kv_splits=2` + `indptr=None`
  时由 `get_meta_param` 在运行时按序列长度/批大小 clamp，接口语义无误。

结论：**stage2 与 ATOM 接口均无需为 seg 布局或 split=2 做改动**；唯一需修复的是 §4 的
asm stage1 多 pass 路径。

## 4. sp3 问题定位（`mla_a8w8_qh64_1tg_16mx4_64nx1_np.sp3`）

多 pass 切分机制（关键行）：

- `var SUB_KV = 64`（行 26）—— 每个 pass 的页粒度（=1 页）。
- `_s_passes` = split 数；**每个 split z 处理 strided 页集合** `z, z+passes, z+2*passes, …`
  （行 6300 `int_div_ss(.., div1, passes)`；行 6318–6319 `LTD_buf_offset=tg_idz*4`、`LTD_buf_inc=passes*4`）。
- `total_ps = full_pages*SUB_KV + tail_len`（行 6286–6288）。
- 空 split 早退：`tg_idz*SUB_KV >= total_ps` 才跳 `kvsplit_empty_tg_exit`（行 6290–6292）。
  对 700+ token 的序列、split1 的 `64 < 700`，**该早退永不触发** → 修复的空 split 路径与本问题无关。
- **尾页（不满 64 的最后一页）只归属一个 split**：`tail_owner = full_pages % passes`
  （行 6303–6316），其余 split 的 `tail_len` 清 0。

定位结论：bug 在 **`num_kv_splits>1` 时「strided 跨页累加 + 尾页归属/尾页 masking」** 这段：
- split=1（passes=1）尾页处理正确（`fin1=True`）；
- passes=2 的 strided + 尾页路径对部分序列（与尾页长度 / 尾页归属奇偶相关）产出 NaN。

> 待确认项：NaN 是否集中在「尾页归属到非 0 号 split」或某些 `tail_len`。
> 可在 replay 中对每条 NaN/OK 序列额外打印 `full_pages、tail_len、tail_owner=full_pages%passes`
> 做相关性分析，以精确锁定 sp3 中尾页 masking 的具体指令段。

## 5. 修复建议

### 方案 A（根治，kernel）
修复 `mla_a8w8_qh64_1tg_16mx4_64nx1_np.sp3` 中 `passes>1` 的尾页/边界处理，使
「两个 split 都有数据」时尾页 masking 与 split=1 等价；改完后重新汇编 `.co` 并
**覆盖部署到 `/app/aiter/hsa/gfx1250/mla/`**（qh64/qh128 同源，注意两者都要更新），
再跑 §2 单测，期望 `fin2=True`、`cos2_1≈1.0`、`relerr2_1≈0`、`non-finite split2 = 0`。

### 方案 B（主机兜底，快速可用，保留 split=2）
在 `aiter/mla.py` 的 `mla_decode_fwd` 中，reduce 之后做 `torch.isfinite(o).all()` 检查，
若失败则用 `num_kv_splits=1`（uniform indptr）**回退重算一次**再返回。
绝大多数走 split=2 快路径，仅命中坏边界的批回退到已验证正确的 split=1。
代价：每次 decode 多一次 device→host 同步（可用环境变量门控）。

### 方案 C（不推荐）
绕过 `get_meta_param` 强行全局固定 split=2：会**取消现有保护**（`get_meta_param` 对
fp8 在平均 < 33 页时把 2 clamp 回 1），使更多短批进入坏路径，精度会**进一步恶化**。

### 临时规避
在 kernel 修好前，将 ATOM 的 `num_kv_splits` 改回 1（`attention_mla.py` ~1221 行）。
split=1 既正确、精度又更高（0.944）。

## 6. 相关文件

- 调用入口：`app/ATOM/atom/model_ops/attention_mla.py`（~1211–1231，非 persistent decode）
- host：`app/aiter/aiter/mla.py`（`mla_decode_fwd`、`get_meta_param`；C-a 预初始化）
- kernel 源：`poc_kl/mi400/mla/shaders/mla_a8w8_qh64_1tg_16mx4_64nx1_np.sp3`
  （qh128 与之同源，仅符号名不同）
- 部署 .co：`app/aiter/hsa/gfx1250/mla/mla_a8w8_qh128_1tg_16mx4_64nx1_np.co`
- 空 split 修复说明：`poc_kl/mi400/mla/shaders/empty_split_worklog.md`
- 复现脚本：`app/aiter/op_tests/replay_mla_split_compare.py`
  （含 `_stage1_partials` stage1-partial NaN 探针，见 §2.2.1）
- dump 工具：`app/ATOM/atom/utils/mla_decode_dump.py`（新增 `ATOM_MLA_DUMP_MIN_PAGES`）

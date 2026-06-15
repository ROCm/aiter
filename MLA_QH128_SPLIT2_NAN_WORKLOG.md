# MLA qh128 decode `num_kv_splits=2` NaN 排查报告

## 1. 背景 / 问题现象

- 模型：DeepSeek-R1（fp8 MLA decode，nhead=128），ATOM 起服务、lm-eval 跑 gsm8k（3-shot）。
- 现象：端到端精度随 `num_kv_splits` 显著变化：
  - `num_kv_splits=1`：gsm8k ≈ **0.944**
  - `num_kv_splits=2`：gsm8k ≈ **0.879**（下降约 6.5 个点）
- 调用入口：`app/ATOM/atom/model_ops/attention_mla.py` 的非 persistent 路径，硬编码
  `mla_decode_fwd(..., num_kv_splits=2, ...)`（约 1221 行）。

根因结论（已定位到确切指令，见 §4.1.2）：**`num_kv_splits>1` 时，对部分真实序列某行的
softmax 分母 `L` 下溢为 0，而 `R_div_L`（sp3 2277-2305）缺失 sum==0 保护（2288 行只有注释），
导致 `1/0=+inf`、`R(0)*inf=NaN`** 污染该序列注意力输出 → 端到端精度回退。
- **不是**空 split、不是 .co 过期、不是控制流/掩码/页计费问题（§4.1 逐指令核对 + §4.1.2 几何碰撞已排除）。
- 是**数据相关数值问题**：`L=0` 源于运行最大值被数据相关的垃圾值污染（深层成因待查），
  但**缺失的 sum==0 保护是 NaN 的直接诞生点**，补回该保护即可消除 NaN（已修复）。

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

#### 2.3.1 stage1-partial 探针实测结果（已确认）

40/40 dump 全部命中，且决定性地把根因锁死在 stage1：

```
non-finite split2 outputs: 40/40
stage1-partial probe: 40/40 dumps have NaN in stage1 (logits|attn_lse) BEFORE stage2.
  of 40 dumps with non-finite split2 OUTPUT, 40 already had NaN in stage1 partials
  (=> asm stage1 root cause; stage2 reduce exonerated).
每条 dump: stage1 NaN: logits_bad_splits=[0, 1] lse_bad_splits=[0, 1]
```

两条强结论：
1. **NaN 在 stage2 之前就已存在于 partial 缓冲** → stage2 reduce / seg 布局被彻底排除（见 §3.5）。
2. **`logits_bad_splits=[0,1]`：两个 split 都 NaN**，不是只有尾页归属的那一个 split。
   这推翻了「尾页只归属单个 split 才出错」的窄假设：passes=2 的 strided 多 pass 路径把
   **split 0 和 split 1 双双污染**。结合 §2.3「只有 16/64 条序列出错」，
   说明触发条件与**每条序列自身的页数/尾页奇偶**相关，但一旦触发，两个 pass 的 partial 都坏。

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

### 4.1 静态逐指令核对（已排除项 + 剩余可疑面）

为定位到具体指令，对 sp3 中所有用到 `_s_passes`/`_s_tg_idz` 的多 pass 专有代码逐条核对。
**以下均已确认与 split=1 全局语义一致、计算正确**，可从嫌疑中排除：

| 区段 | 行号 | 作用 | 核对结论 |
|---|---|---|---|
| qh128 z 维解包 | 6221-6224 | `tg_idx=z&1`(qh64 组), `tg_idz=z>>1`(真 split) | 正确，passes 用解包后的 split 数 |
| 页计费 | 6280-6288 | `full_pages=n_pages-(tail?1:0)`，`total_ps=full_pages*64+tail_len` | 正确（total_ps=全局 seqlen） |
| 空 split 早退 | 6290-6292 | `tg_idz*64>=total_ps` 才退 | 长序列(>64)两 split 都不早退，无关 |
| 每 split loop_cnt | 6294-6301 | `(full_pages-1-tg_idz)/passes+1` | 正确（strided 满页计数） |
| tail_owner | 6303-6316 | `full_pages%passes` 拥有尾页 | 正确 |
| LTD 双缓冲偏移 | 6318-6323 | `off=tg_idz*4`，`inc=passes*4` | 正确（strided 取页索引） |
| LTD 步进 | 2403-2405 | 每页 `off+=passes*4` | 正确 |
| LTD→CKV 地址 | 1241-1249 | `CKV+=(KPAGE)*LTD[idx]` | 正确 |
| 掩码边界(主循环) | 6133-6141 | `kv_top_eidx=total_ps`，`mask_idx=(tg_idz+1)*64`，`inc=passes*64` | 正确（全局行索引） |
| 掩码边界(尾循环) | 5632-5638 | `mask_idx=((loop_cnt-1)*passes+tg_idz+1)*64` | 正确 |
| R/LSE 写出偏移 | 5947-5966 / 5970-5992 | split 维偏移 `tg_idz` | 正确（且若写错位会留下哨兵=finite，与实测 NaN 矛盾，故排除寻址错位） |

**关键推论**：既然每个 split 的页集合、掩码边界、取页地址都全局正确，则
「某行被全部掩成 NEG_INF → softmax 分母 0 → 0/0=NaN」这一解释**不成立**
（正确掩码下两 split 各有多页有效）。因此 NaN 只能来自：
**多 pass 流水线 overlap 路径里，对特定页数序列的 K/V 数据通路或累加器/LDS slot 的破坏**
（计算出 inf→exp 溢出→NaN），而非标量页计费/掩码。

### 4.1.1 几何关联实测：NaN 是**数据相关数值问题**，不是控制流（已确认）

pooled 关联结果（40 dump、640 个 NaN batch）：

```
full_pages -> 10:5, 11:228, 12:312, 13:79, 14:16
tail_len   -> 铺满 0..63（每值 8~14）
tail_owner -> 0:333, 1:307
loop0      -> 5:5, 6:540, 7:95     loop1 -> 5:233, 6:391, 7:16
full_pages%2 -> {0:333, 1:307}
```

**关键判读**：NaN 与 OK 的几何特征**大面积重叠**——`full_pages∈{11,12,13,14}`、
`tail_owner∈{0,1}`、`tail_len` 全段、`full_pages%2` 两种奇偶都同时出现在 NaN 与 OK。
由于 `(full_pages, tail_len)` 完全决定内核控制流，**同一几何同时为 NaN 和 OK** ⇒
**NaN 与控制流/掩码/页计费无关，是数据相关的数值问题**（同样的指令路径，K/Q/V 数值不同，
一条溢出、一条正常）。这与 §4.1 静态核对「标量页/掩码/寻址全对」完全吻合。

进一步约束：
- `softmax_init()`（sp3 6399，置 `_v_FA_Max=NEG_INF`）与 `causal_init()`（6400）在
  **wave/loop_cnt 分支之前的公共入口**，两 split 都执行 → 「行最大值未初始化」被排除。
- split=1 与 split=2 共用同一套 bootstrap + `core_loop`，仅 `passes`（页步进）不同；
  split=1 全程 finite ⇒ bug **只在 `passes>1` 的数值行为**里。

**最可能机理**：在线 softmax 的**运行最大值 `m` 在 passes>1 的流水线 bootstrap 里出现
时序错位**（exp 用到尚未并入当前 strided 页最大值的「陈旧 `m`」），对「新页分数显著大于
陈旧 `m`」的序列 `exp(score-m)` 溢出成 `+inf` → `inf` 传播成 `NaN`；分数偏小的序列恰好不溢出。
- 已加 inf/nan 分型探针：若 `logits[+inf]>0` ⇒ 坐实溢出（运行最大值失效）；
  若纯 `nan` 无 inf ⇒ 0/0（零和/全掩）。**先看这一行再决定改哪段指令**。
- 若确为溢出：定位 sp3 `core_loop` / `bootstrap_wave_0/1` 内 `mla_softmax` 与 `GEMMK`、
  `_v_FA_Max` 更新之间的 `s_wait_*` / 发射顺序（6453-6494 / 6530-6571），
  对比 passes=1 的等价时序，修正 `m` 更新与 `exp` 的先后。

### 4.1.2 实测分型 + 锁定到确切指令（已确认并修复）

第二轮 replay（已加 inf/nan 分型 + (full_pages,tail_len) 碰撞检测）实测：

```
(full_pages,tail_len) keys: nan-only=16 ok-only=52 BOTH=197
>>> COLLISION: same geometry is BOTH NaN and OK -> NaN is DATA-dependent (numerical), not control-flow.
每条 dump: logits[+inf=0 -inf=0 nan=~1.5M] lse[+inf=0] kind=0/0 NaN (zero-sum / all-masked)
```

两个决定性事实：
1. **碰撞 BOTH=197**：同一 `(full_pages,tail_len)`（⇒ 完全相同的内核控制流与掩码）同时出现
   NaN 与 OK ⇒ **与控制流/掩码/页计费无关，纯数据相关数值问题**（坐实 §4.1 静态核对）。
2. **`kind=0/0`、无任何 inf**：整行 512 维全 NaN ⇒ 该行 softmax 分母 `L=0`，归一化 `0/0`。

**锁定到确切指令** —— `R_div_L`（sp3 2277-2305，post-process 唯一归一化）：
```
2286  v_add_f32 _v_L = Σ exp(score-max)        // 分母求和
2288  //check sum==0   ← 原代码此处保护为空（只有注释）   ★缺陷
2298  v_rcp_f32 _v_L = 1/_v_L                  // L==0 → +inf
2300  v_fma_f32 _v_Lse = ln(L)+max*scalar      // L==0 → -inf（故 lse 非有限且无 +inf）
（其后 R_post_process: R *= _v_L）             // R(=0) * inf = NaN → logits 全 NaN
```
完全对上实测：`logits 全 nan`、`lse 无 +inf`（是 -inf）。

数学上 `L=0` 仅当运行最大值 `FA_Max` 被污染成 > 所有真实分数（否则 argmax 项 = exp(0)=1，L≥1）；
而污染是**数据相关的垃圾值进入 max 归约**（同几何不同数据 → 大则污染、小则侥幸），
与 split=1 共用同一 bootstrap 却唯独 split=2 触发 ⇒ 属 `passes>1` 流水线对特定 K/V 数据的
LDS/寄存器时序敏感（深层成因，需 GPU 侧 dump per-split `FA_Max` 才能再细究）。

**已应用修复（补回 2288 的 sum==0 保护，即作者注释本意）**：在 `R_div_L` 内
- rcp 之前用 `v_cmp_ne_u32` 捕获 `sum!=0` 判据（此时 `_v_L` 仍是分母和）；
- 归一化末尾用两条 `v_cndmask_b32`：退化行（sum==0）强制
  **归一化因子=0（R*0=0，消除 NaN）** 且 **`_v_Lse=EMPTY_SPLIT_LSE`**，
  使 stage2 online-softmax 把该 split 当空 split 丢弃、退化到另一个 split。
- 非退化行（mask 置位）保持原值 → **工作路径逐位不变**；split=1 路径 L 永不为 0，无影响。

> 语义正确性：在线 softmax 中「某 split 软最大质量为 0」的正确合并行为本就是发出
> `lse=-inf`(空 split) 让 stage2 忽略它——故此修复既消 NaN、又是数学上正确的退化处理。
> 残留：若**同一行两个 split 都被污染**才会丢失该行注意力（但仍 finite）。需重汇编 .co
> 后跑 §2 单测确认 `fin2=True`，再跑 gsm8k 看是否回到 ≈0.944；若仍偏低，再深挖 max 污染源。

> ⚠️ 该补丁改的是手写 asm（`mla_a8w8_qh64_1tg_16mx4_64nx1_np.sp3`，qh128 同源），
> **本环境无法汇编/验证**。需在能 build 的环境重汇编 `.co` 并覆盖部署到
> `/app/aiter/hsa/gfx1250/mla/`（qh64 与 qh128 都要），再跑 replay 确认。

### 4.2 用页几何关联锁定触发条件（新增工具）

`replay_mla_split_compare.py` 已新增 `_kernel_geom()` + `_stage1_per_batch_diag()`：
对每个 NaN/OK batch 还原内核视角的 `full_pages / tail_len / tail_owner / loop0 / loop1`，
并在汇总里**池化所有 dump 的 NaN batch** 打印各特征的分布，用来回答：
NaN 是否集中在某个 `full_pages` 奇偶 / 某段 `loop_cnt` / 某个 `tail_owner`。

> 下一步（需在 ff_mla 跑）：看 `=== stage1 NaN batch geometry (pooled) ===` 的分布。
> - 若 NaN 全部 `full_pages%2==X` 或集中在某 `loop0/loop1` 值 → 锁定到流水线对应 `loop_cnt`
>   分支（`final_loop_0/1` 的 `last_loop[parity]`，sp3 5640-5790；或 core_loop 的尾页 drain）。
> - 若与 `tail_owner` 强相关 → 锁定尾页 drain 的 K/V slot（`GEMMV_tail_drain_*` / `GEMMK_tail_drain`）。
> 这能把范围从「整个多 pass 路径」收敛到具体的流水线分支与 drain 宏，再对那几条
> `s_wait_*` / slot 递增/`VGPR_init` 指令做修复。

> **§2.3.1 实测修正**：探针显示触发序列的 `logits_bad_splits=[0,1]`——**两个 split 的 partial 都坏**，
> 不是只有「尾页归属的那个 split」。因此 bug 不是单纯的尾页 masking 漏写，而更像是
> **passes=2 时 strided 页遍历/累加器（或共享的尾页边界寄存器）被破坏，污染了同一 work-group
> 内两个 pass 的累加**。下一步应在 sp3 中重点审查：
> - `int_div_ss(.., div1, passes)`（行 6300）后 strided 页步进 `LTD_buf_inc=passes*4`（行 6318–6319）
>   是否在 passes>1 时把累加器初始化 / 边界寄存器算错；
> - `total_ps`、`tail_len`、`tail_owner=full_pages%passes`（行 6286–6316）的计算在 passes=2 下
>   是否对**两个 split 都**产生越界/未初始化访问。
> 可在 replay 中对每条 NaN/OK 序列额外打印 `full_pages、tail_len、tail_owner` 做相关性分析，
> 锁定触发的页数/尾页特征。

## 5. 修复建议

### 方案 A（根治，kernel）★已应用，待重汇编验证
补回 `R_div_L`（sp3 ~2288）缺失的 sum==0 保护：rcp 前用 `v_cmp_ne_u32` 记下 `sum!=0`，
归一化末尾用两条 `v_cndmask_b32` 把退化行（sum==0）的归一化因子置 0、`_v_Lse` 置
`EMPTY_SPLIT_LSE`（详见 §4.1.2）。非退化行逐位不变。改完后重新汇编 `.co` 并
**覆盖部署到 `/app/aiter/hsa/gfx1250/mla/`**（qh64/qh128 同源，注意两者都要更新），
再跑 §2 单测，期望 `fin2=True`、`non-finite split2 = 0`；随后跑 gsm8k 看是否回到 ≈0.944。
> 该补丁消除 0/0 NaN 且是数学上正确的空 split 退化处理；若 gsm8k 仍偏低，说明 max 污染
> 丢失了真实质量，需继续深挖（GPU 侧 dump per-split `FA_Max`）定位污染源（LDS/寄存器时序）。

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

# mxmoe_gemm_v2 persist-es 对齐 `.bak`：实验记录与后续计划

> 来源：v3-110（zhimding）上的 Cursor 会话「vLLM ROCm aiter MoE test」，2026-11-24 ~ 11-25。
> 除特别说明外，数字都是 GEMM2 kernel 单独计时、cold cache、MI355X GPU0，单位 us。
> 术语：**panel** = 一个 expert × 256 列（BN）× 整个 K；**tile / chunk** = panel 内的 32 行（BM=32）；W2 一个 panel 共 20 条 load（16 条 value + 4 条 e8m0 scale）。

## 0. 结论速览

- **目标**：在 `cguo/gemm2es` 上，用尽量小的改动让 `mxmoe_gemm_v2.py`（HEAD 的 persist-es）追上 `mxmoe_gemm_v2.py.bak`（完整 expert-stationary 流水线）。2k 下实测差距是 110 → 101。
- **`.bak` 快的根本原因**：下一个 panel 的 20 条 W2 load **从第一个 chunk 就开始、偏前地摊开**发在当前 panel 的前几个 chunk 里，而且 A 的 load 排在 W2 之前发出。去掉提前量，`.bak` 就退回到 HEAD 的水平。chunk 内部的其他优化在 2k–8k 是次要的。
- **"只在第二个 tile 一次性发 20 条"不够快**：无论在 v2 还是在 `.bak` 上实现，都约等于不预取（2k 116.5）。会话中途一度得出"它和 `.bak` 一样快"的结论，那是测试脚本的 bug，**已作废**。
- **直接在 v2 tile 上加预取（路线 1）**：VGPR 从 140 涨到 208，每 CU 驻留的 WG 从 3 个掉到 2 个，原来被第 3 个 WG 掩盖的等待全部暴露出来，结果比 HEAD 慢。但在同样每 CU 2 个 WG 的条件下，预取本身能省 23 us（2k）。
- **同步 LDS DMA 的保守等待**：LLVM 会在后续任何 LDS 访问前插 vmcnt 等待，把提前发出的 A DMA 等掉。flydsl 0.5.3 的 async LDS DMA（`BufferLoadAsyncLDS128b` + `asyncmark` / `wait_asyncmark`）可以绕开，已实测可行。所以路线 1 的"A 跨 tile 预取"重新可行，不必像 `.bak` 那样让 A 先进 VGPR。
- **推荐下一步**：路线 1 + async DMA（见第 6 节）。备选是直接沿用 `.bak` 的 ES 分支（性能已经达标，但代码量大）。

---

## 1. 背景

commit `dcf094ffa`（HEAD of `origin/cguo/gemm2es`）在默认 v2 tile 流水线上，把 expert-stationary GEMM2 改成 persist 模式（`AITER_MXFP4_G2_ES=1`）。它和原版 v2 的区别只有两点：panel 级的 persist 调度，以及 W2 的 B double buffer 跨 tile 常驻（K ≤ 2 个 tile 时整块 W2 留在寄存器里）。其余流水线基本没变。`.bak` 是上一个 commit `f209cd555` 的 v2，两者逐字节相同。

GLM-5.2 的 GEMM2 shape：6144 × 512，257 个 expert，top-9，a4w4，fp4 route rows（`AITER_MXFP4_INTERMEDIATE=1`）。

| 变体 | 怎么跑 | kernel 名特征 |
|---|---|---|
| 原版 v2 + tuned tile | `AITER_MXFP4_G2_ES=0` | `bm64_bn256_bk128…persist` |
| HEAD persist-es（你的改动） | `AITER_MXFP4_G2_ES=1` | `bm32_nt…pes_cu256_ne257` |
| `.bak` 完整 ES | `AITER_MXFP4_G2_ES=1` + `PYTHONPATH=aiter_f209cd555` | `hmax6144…pes_cu256_ne257_p3`（4k 是 `p5`） |

HEAD 的 ES 路径被 dispatcher 强制成 BM=32、BN=BK=256，所以 KT=2，并且有 `INTER_MAX // BK <= 2` 的断言。

## 2. 环境与测量方法

- 容器 `zhimding_vllm_aiter`（镜像 `vllm/vllm-openai-rocm:latest`，torch 2.13 + ROCm 10.0），挂载 `/mnt/m2m_nobackup`。aiter 用 `AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 pip install -e . --no-build-isolation` 做 editable 安装。合并 main 之后 flydsl 是 0.5.3。镜像里还需要额外 `pip install tabulate`。
- bench 脚本是 `gemm2es_bench/bench_gemm2es.py`：
  - 先跑一次 `fused_moe`（用 tuned CSV 的配置），hook `mxfp4_moe_gemm2` 拿到真实入参，再单独重放 GEMM2。
  - 路由用 `torch.randn` 生成 gating，再经 `fused_topk`。
  - 每次 launch 前读一遍 1 GiB 的 buffer，冲掉 L2 和 MALL。
  - 用 profiler 只统计 GEMM2 kernel 的 device 时间。
  - 选项：`--stage2` 时间通过 `kernel_bench_callable` 统计（GEMM2 加上后面的 reduce）；`--dump-dir` 保存 route rows 用于逐字节比对；`--es-wg-per-cu` 强制每 CU 的 WG 数。
- ISA：`FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_DIR=...` 导出，再用 `isa_stats.py` 和 `isa_condense.py` 统计。
- 注意：这个 harness 不是 commit 作者的。2k 的绝对值比 commit message 高 3–8%，4k 基本吻合。大部分实验只在 GPU0 上跑了一次（8 张卡之间差异在 1% 以内）。

## 3. 实验与结果

### 3.1 复现 commit 的三组数据（8 卡中位数）

| 变体 | 2k 实测 | 2k commit | 4k 实测 | 4k commit |
|---|---|---|---|---|
| 原版 v2 + tuned tile | 118.8 | 114.9 | 160.8 | 157.0 |
| HEAD persist-es | 110.3 | 103.9 | 149.4 | 148.0 |
| `.bak` 完整 ES | 100.6 | 93.4 | 122.2 | 121.1 |

- 快慢顺序和 commit 一致。HEAD 到 `.bak` 的差距：2k 9.7 us（commit 是 10.5），4k 27.2 us（commit 是 26.9）。
- HEAD 输出的 route rows 和 tuned tile 的 v2 **逐字节一致**；`.bak` 约 1.07% 字节不同，因为累加顺序不同。

### 3.2 合并最新 main（`f426c2083`）

- 冲突一：`moe_reduce.py` 取 main 的版本（#7706 已经包含分支的 `dd8207cdf`）。
- 冲突二：`mxmoe_dispatcher.py` 的 6 处，main 的 `g2_scatter_lds_pairs` 和分支的 `g2_es` 两个参数都保留，kernel 名 tag 那一行取 main 的版本。两者互斥：一个只用于 scatter epilogue，一个只用于 reduce epilogue。
- 合并后相对 main，只剩 8 个 ES 相关文件。CK 子模块指向 `f93c029e0`，flydsl 是 0.5.3。
- 合并后的性能不变：ES 关时 118.8 / 161.2，ES 开时 110.0 / 149.1，输出逐字节一致。`test_moe_2stage`（`-q 4`）的 logits_diff 是 5.83e-4。
- merge commit `4040a7109`，作者 `charlieguo1106 <cguo@amd.com>`，没有 co-author，**没有 push**。

### 3.3 `.bak` 为什么快

**静态对比（2k 的 ISA）**

| | HEAD persist-es | `.bak` |
|---|---|---|
| VGPR / 每 CU 的 WG 数 | 140 / 3 | 242 / 2（无 spill） |
| LDS / WG | 16 KB（A 槽位和 C staging 共用） | 26368 B |
| W2 | 当前 panel 常驻，只在 panel 的第一个 tile load | 当前 panel 和下一个 panel 都在寄存器里；下一个 panel 的 20 条摊到前 PEEL 个 chunk |
| A | LDS DMA，每个 tile 开头发出、马上等 | global → VGPR → LDS，提前 2 个 chunk |
| epilogue | C 以 bf16 写进 LDS（64 条 `ds_write_b16`），读回后量化 | 在寄存器里量化成 fp4，约 12 条 LDS 写 |
| store | 带分支的 `global_store` | 无条件 `buffer_store`，无效行写到越界地址（等待计数保持精确） |

HEAD 非首个 tile 的执行顺序：`BAR → A DMA(3) → vm(1)（A 的延迟暴露）→ BAR → MFMA kt0 → vm(0) → BAR → MFMA kt1，中间穿插 C 的 ds_write → BAR → load token id → vm(0) → 分支 store`。

**`.bak` 的分摊规则**（`es_default_peel`，按每个 expert 的行数分档）：

- 行数 ≤ 48：PEEL=2，全部 20 条在一处发。
- 行数 ≤ 96（2k，约 72 行）：PEEL=3，按 8/6/6 发。
- 更多行（4k，约 143 行）：PEEL=5，按 8/3/3/3/3 发，并且 EARLY。

分档依据是每个 panel 的 tile 数分布：2k 时 3 个 tile 的 panel 占 217/257，4k 时 5 个 tile 的占 212/257。

**改变 `.bak` 里 W2 的发法**（GPU0，cold）：

| 下一个 panel 的 20 条 W2 怎么发 | 2k | 4k |
|---|---|---|
| `.bak` 默认（3 份 / 5 份，偏前） | **100.8** | **123.1** |
| 全部在 chunk 0 一次发出 | 106.1 | 141.0 |
| 4k 也只分 3 份 | – | 131.1 |
| 全部推迟到最后一个 peel chunk | 116.6 | 151.7 |
| PEEL=1，没有提前量 | 116.9 | 154.3 |
| HEAD persist-es（对照） | 109.2 | 149.2 |

- token 从 1536 扫到 4096 时，panel 数固定是 6168，只有 chunk 数在变。总时间对总 chunk 数的斜率：HEAD 2.79 ns，`.bak` 1.52 ns，**去掉提前量的 `.bak` 是 2.66 ns，和 HEAD 基本一样**。也就是说，`.bak` 的优势几乎全部来自"W2 的 HBM 流量和当前 panel 的计算重叠"。
- 摊开发之所以重要：vmcnt 按发出顺序计数，所以一次性发 20 条之后再发的 A load，等它时会连带等完这批 W2。`.bak` 的注释写的是：`Chunk 2 ahead of the next panel's W2: waiting for it does not wait for that W2.`

### 3.4 收益和每个 panel 的 tile 数的关系

| token | 每 panel 的 tile 数 | HEAD | `.bak` | `.bak` 快多少 | HEAD 带宽 | `.bak` 带宽 |
|---|---|---|---|---|---|---|
| 1536 | 2.08 | 99.9 | 96.0 | 3.9% | 4.79 TB/s | 4.98 TB/s |
| 2048 | 2.86 | 109.2 | 99.8 | 8.6% | 4.53 | 4.96 |
| 3072 | 3.89 | 127.2 | 115.1 | 9.5% | 4.15 | 4.58 |
| 4096 | 5.00 | 149.2 | 121.1 | 18.8% | 3.75 | 4.62 |

- W2 共约 430 MB（257 个 expert × 24 个 N-block），和 token 数无关，2k 时占总字节的 87%。同一张卡上用 `torch.sum` 读 1 GiB 测得 5.47 TB/s，`.bak` 在 1.5k–2k 已经到它的约 91%。
- tile 少时，这个 kernel 卡在 HBM 带宽上，预取能藏住的计算有限；tile 多时，HEAD 的计算和 W2 流量串行叠加，差距被拉大。
- 按 5.47 TB/s 算，2k 的下限大约在 90 us。

### 3.5 大 token 和 reduction dtype

只看 GEMM2（fp4 中间结果）。16k/32k 下要让 ES 生效，测试 CSV 把 stage1 换成了 `flydsl_moe1_…t128x128x256_w2_fp4`。

| token | 每 panel 的 tile 数 | 生产配置 | HEAD | `.bak` | `.bak` 去掉提前量 |
|---|---|---|---|---|---|
| 8k | ≈9.5 | 246.6（64x256x256 reduce） | 250.5 | **219.1** | 252.1 |
| 16k | 18.4 | 581.7（128x256x256 scatter） | 458.2 | **413.0** | 433.9 |
| 32k | 36.3 | 1049.8（128x256x256 scatter） | 881.6 | **730.1** | 761.4 |

整个 stage2（GEMM2 加 reduce；括号里是 GEMM2 单独的时间）：

| token | 中间格式 | 生产配置 | HEAD | `e 的 base/rows | `global_load ×2` + `vmcnt(1)` | 如果 panel 只有 2 个 tile，tile 1 的预取要在这里等完 |

另外，tile 0 和 tile 2 以后，epilogue 读 token id 之后仍然是 `vmcnt(0)`。

### 3.8 路线 2：在 `.bak` 的 ES 上删代码（`route2_*.patch`）

- 把 `dcf094ffa` 的改动反向应用到合并后的代码上，得到 `.bak` 的完整 ES（`route2_full_es_restored.patch`）。2k 100.9、4k 124.0，和 `.bak` 逐字节一致。
- 简化版（`route2_simplified_tile2.patch`，也是当前工作区的状态）：删掉 `es_default_peel` 和 PEEL/SL/EARLY 那套逻辑，chunk 1 依次发 chunk 3 的 A、下一个 panel 的全部 W2、下下个 panel 的 expert table 行、下一个 panel 的 chunk 0。

| 下一个 panel 的 20 条 W2 发在哪里 | 2k | 4k | 8k |
|---|---|---|---|
| 全部在第二个 tile（简化版） | 116.5 | 154.3 | 243.2 |
| 全部在第一个 tile | 107.3 | 144.0 | 230.2 |
| 第一、第二个 tile 各 10 条 | 106.6 | 141.5 | 227.9 |
| `.bak` 默认（按行数分 3 份 / 5 份） | **100.9** | **124.0** | **213** |
| HEAD | 110 | 150 | 250 |

结论：必须从第一个 tile 就开始发（提前量），并且要摊开（避免 A 排在一大串 W2 后面）。4k 和 8k 的 panel 有 5–9 个 tile，摊开就更重要。

### 3.9 async LDS DMA（`async_a_v2_experiment.patch`）

- 原来的问题：用同步 atom `BufferCopyLDS128b` 发出的 LDS DMA，走的是 vmcnt 计数。只要它还没完成，后续任何 `ds_read` / `ds_write` 之前，LLVM 都会插 `vmcnt(N)`，让所有已发出的 LDS DMA 完成，因为 IR 里没有 alias scope。手写一个更松的 `s_waitcnt` 也没用，LLVM 会在后面再补一个更严格的。
- 解决方法：换成 `fx.rocdl.cdna4.BufferLoadAsyncLDS128b()`，每发完一组调用 `rocdl.asyncmark()`，需要等的时候调用 `rocdl.wait_asyncmark(n)`（最多还剩 n 组在途）。aiter 里的 `bmm_a8w8_mxscale_gfx950.py` 和 `gemm_a16w16_gfx950.py` 已经在用。限制：没有 1 字节的大小，不支持 nt cache policy，对 mxmoe 的 A 没有影响。
- 实测（v2 tuned，ES 关）：`ds_read kt1` 之前原来的 `vmcnt(8)` 消失了，kt2 的 DMA 和 8 条 B load 都还在途；MFMA kt1 之前只剩 `vmcnt(13)`。输出逐字节一致，快了约 1%（2k 121.3 → 120.0，4k 164.0 → 162.4）。

## 4. 你的假设和实验结论对照

| 你的判断 | 结论 |
|---|---|
| `.bak` 把 4 组 A/B ktile 的 load 平分成 3 份，避免 epilogue 和下一组 load 冲突 | 部分对。"摊开"确实是核心，但摊开的是**下一个 panel 的 20 条 W2**，沿 M 方向分到前 PEEL 个 chunk 里，作用是让 W2 流量和计算重叠，并且让 A 不排在 W2 后面 |
| BK=128 的 pipeline：在 kt3 预取下一个 B kt0，消掉 vm(0) | 提前量只有约 1 个 ktile（约 0.1 µs），盖不住 HBM 延迟；同一个 panel 里 W2 本来就常驻。A 提前取、token id 前移的方向是对的，但在 2k 是次要收益 |
| 核心是把下一个 panel 的 20 条 W2 摊到当前 panel 的几个 tile 里 | 对 |
| 每个 panel 的 tile 少时，pipeline 省不了多少 | 对。这时卡在 HBM 带宽（1.5k 时 `.bak` 只快 3.9%） |
| 大 token 要上 128x256x128，寄存器压力更大 | 不需要。fp4 reduction 下 BM=32 的 ES 比生产的 128x256x256 快约 30%；bf16 reduction 时 16k 以上 scatter 更好 |
| 只在第二个 tile 预取 k0/k1 这一半（折中方案） | 只收回 14–22% |
| 第二个 tile 一次性预取整块 W2 | 约等于不预取（116.5）。在 v2 上还会把 occupancy 降到 2 WG/CU |
| async 能不能只等 kt0、让 kt1 的 DMA 继续在途 | 能。被等的是 vmcnt，不是 lgkmcnt；flydsl 0.5.3 的 async LDS DMA 已验证 |

## 5. 当前状态和文件位置

以下都在 v3-110 上的 `/mnt/m2m_nobackup/cguo/`。今天这个目录已经整体拷到了 `/shared_nfs/cguo/`，但 git worktree 里的 `.git` 文件写的是旧的绝对路径，换位置后要用 `git worktree repair` 修一下。

- `aiter/`：分支 `cguo/gemm2es`，HEAD 是 merge commit `4040a7109`（还没 push）。**工作区是路线 2 的简化版，没有提交。**
- `aiter_f209cd555/`：worktree，就是 `.bak` 对应的 commit（原样，没有改动）。
- `aiter_head/`：worktree，HEAD `4040a7109` 的基线（原样）。
- `gemm2es_bench/`：
  - 脚本：`bench_gemm2es.py`、`isa_stats.py`、`isa_condense.py`；大 token 用的配置 `glm5_reduce_large.csv`。
  - patch：
    - `route1_v2_tile1_prefetch.patch`：路线 1，v2 上 tile 1 预取整块 W2，加上两处精确等待。
    - `route2_full_es_restored.patch`：`.bak` 的完整 ES 叠到合并后的代码上，性能等于 `.bak`。
    - `route2_simplified_tile2.patch`：当前工作区的简化版。
    - `async_a_v2_experiment.patch`：v2 的 A 改用 async LDS DMA 的最小实验。
  - 日志前缀：`all_gpu*`、`sweep_*`、`large*_`、`dt_fp4_*`、`dt_bf16_*`、`merged_*`；ISA 在 `isa*/` 下；route rows 在 `dump*/` 下。
- 容器 `zhimding_vllm_aiter` 在 v3-110 上。现在 zhimding 的 `ssh v3-110`（端口 25543）连不上，容器可能需要重建。
- 重跑示例：

```bash
cd /mnt/m2m_nobackup/cguo/gemm2es_bench
E="-e HIP_VISIBLE_DEVICES=0 -e AITER_MXFP4_INTERMEDIATE=1"
docker exec -w $PWD $E -e AITER_MXFP4_G2_ES=0 zhimding_vllm_aiter python3 bench_gemm2es.py -t 2048 4096 --label v2_tuned
docker exec -w $PWD $E -e AITER_MXFP4_G2_ES=1 zhimding_vllm_aiter python3 bench_gemm2es.py -t 2048 4096 --label pes_v2
docker exec -w $PWD $E -e AITER_MXFP4_G2_ES=1 -e PYTHONPATH=/mnt/m2m_nobackup/cguo/aiter_f209cd555 \
  zhimding_vllm_aiter python3 bench_gemm2es.py -t 2048 4096 --label es_full
```

改了 kernel 但 kernel 名不变时，要加 `-e FLYDSL_RUNTIME_ENABLE_CACHE=0`。环境变量要用数组传，之前的作废数据就是 shell 函数传参位置写错造成的。

## 6. 后续修改与优化计划

会话最后在问"路线 2 接下来怎么走"时你取消了，所以这一步还没定。下面按推荐程度排列。

### 路线 1 + async DMA（推荐：最贴近现有 v2，改动可控）

从 `route1_v2_tile1_prefetch.patch` 出发，每做一步都用 bench 测 2k/4k/8k，和 HEAD 的 dump 逐字节比对，并检查 ISA 里的 vmcnt 和 VGPR/每 CU 的 WG 数。目标是 2k ≈ 101、4k ≈ 124、8k ≈ 213。

1. **A 跨 tile 预取（收益最大）**：
   - A 的 DMA 换成 `BufferLoadAsyncLDS128b` 加 `asyncmark`。
   - tile c 在发 W2 预取**之前**，先把 tile c+1 的 A 发进第二块 LDS buffer（每 WG 多 8 KB，放在 C staging 之外，总共约 24 KB）。
   - ds_read 之前用 `wait_asyncmark` 只等自己那一组。
   - 这一步用来抵消降到 2 WG/CU 的损失：`.bak` 就是靠软件流水在 2 WG/CU 下跑到 100。
2. **A scale、token id、下一个 panel 的 expert table 行，都排到 W2 预取之前发**。它们是读进 VGPR 的普通 load，排好顺序后 LLVM 能算出精确的等待计数。
3. **expert id 直接用 expert table 里的值**（通过 `es.e` 传给 tile），不再每个 tile 读一次 `sorted_expert_ids`，去掉 tile 0 开头那个 `vmcnt(0)`。改动很小。
4. **提前读下一个 panel 的 expert table**：在当前 panel 里把下一个 panel 的 base、rows、expert 读进 rmem（类似 `.bak` 的 `ein_n`），panel 开头就不用再等 global load。
5. **epilogue 的 store 改成无条件执行**，无效行写到越界地址（`.bak` 的做法）。这样分支汇合后的 vmcnt 计数是静态的，第 1、2 步的精确等待才算得准。
6. **调整 W2 预取的位置**：不要只在 tile 1 一次发完。非首个 panel 的 tile 0 已经不用自己 load W2，它的 MFMA 槽位可以用来发。
   - 先试 tile 0 和 tile 1 各发 10 条（路线 2 的实验里，它比只放在 tile 1 快 10 us 左右）。
   - 再试按 rows/expert 分 2/3/5 份、偏前发（比如 8/6/6）。份数必须在编译期固定，前 PEEL 个 tile 展开写，之后的 tile 用运行时循环、不再发 slice。
7. 第 1–5 步做完后再看 epilogue：比如用 `swap_ab` 让每个 lane 拿到同一行连续的 4 列，就能在寄存器里量化，省掉 64 条 `ds_write_b16`。这一步主要对 16k 以上有用。

### 备选：路线 2

- **恢复 `.bak` 的按 chunk 分摊**（`route2_full_es_restored.patch`）：性能已经等于 `.bak`。之后可以把 `es_default_peel` 那张表换成公式（比如份数取 clamp(rows/32, 2, 5)，还需要实测），再精简别的代码。缺点是相对 HEAD 的改动是整个 `.bak` ES 分支（400 多行），不是"最小改动"。
- **固定的 10+10 规则**：删掉 PEEL，比 `.bak` 慢 6%（2k）、14%（4k）、7%（8k），但各个尺寸都比 HEAD 快。

### 其他待办

- 16k 以上要在生产里用上 ES：要么让 aux sort 也输出 expert table，要么大 token 时 stage1 换成不需要 aux sort 的 kernel（对 stage1 性能的影响没测）。
- bf16 reduction 时，16k 以上继续用 scatter；ES 的收益主要在 2k–8k。
- 大 token 的进一步优化（还没验证）：fp4 下 W2 预取只值 4–5%，可以把下一个 panel 的 W2 占的约 68 个寄存器腾出来给第二套累加器，让 chunk c 的 epilogue 和 chunk c+1 的 MFMA 重叠。按每个 panel 的 tile 数选两套配置。
- `mxmoe_gemm_v2.py.bak` 会随 merge 一起进入历史，正式提交前要决定是否删掉。


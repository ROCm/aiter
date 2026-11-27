# mxmoe_gemm_v2 persist-es 对齐 `.bak`：实验记录与后续计划

> 来源：v3-110（zhimding）上的 Cursor 会话「vLLM ROCm aiter MoE test」，2026-11-24 ~ 11-25；11-26 在 v3-169 上复现了 3.1，提交了路线 1b（2/2/0/0），并记下了路线 1c 的设计（见 3.10、3.11 和第 6 节）；11-26 下午在 v3-058 上实测了路线 1b，并实现、实测了路线 1c（见 3.12、3.13）。
> 除特别说明外，数字都是 GEMM2 kernel 单独计时、cold cache、MI355X GPU0，单位 us。
> 术语：**panel** = 一个 expert × 256 列（BN）× 整个 K；**tile / chunk** = panel 内的 32 行（BM=32）；W2 一个 panel 共 20 条 load（16 条 value + 4 条 e8m0 scale）。

## 0. 结论速览

- **目标**：在 `cguo/gemm2es` 上，用尽量小的改动让 `mxmoe_gemm_v2.py`（HEAD 的 persist-es）追上 `mxmoe_gemm_v2.py.bak`（完整 expert-stationary 流水线）。2k 下实测差距是 110 → 101。
- **`.bak` 快的根本原因**：下一个 panel 的 20 条 W2 load **从第一个 chunk 就开始、偏前地摊开**发在当前 panel 的前几个 chunk 里，而且 A 的 load 排在 W2 之前发出。去掉提前量，`.bak` 就退回到 HEAD 的水平。chunk 内部的其他优化在 2k–8k 是次要的。
- **"只在第二个 tile 一次性发 20 条"不够快**：无论在 v2 还是在 `.bak` 上实现，都约等于不预取（2k 116.5）。会话中途一度得出"它和 `.bak` 一样快"的结论，那是测试脚本的 bug，**已作废**。
- **直接在 v2 tile 上加预取（路线 1）**：VGPR 从 140 涨到 208，每 CU 驻留的 WG 从 3 个掉到 2 个，原来被第 3 个 WG 掩盖的等待全部暴露出来，结果比 HEAD 慢。但在同样每 CU 2 个 WG 的条件下，预取本身能省 23 us（2k）。
- **同步 LDS DMA 的保守等待**：LLVM 会在后续任何 LDS 访问前插 vmcnt 等待，把提前发出的 A DMA 等掉。flydsl 0.5.3 的 async LDS DMA（`BufferLoadAsyncLDS128b` + `asyncmark` / `wait_asyncmark`）可以绕开，已实测可行。所以路线 1 的"A 跨 tile 预取"重新可行，不必像 `.bak` 那样让 A 先进 VGPR。
- **推荐下一步**：路线 1 + async DMA（见第 6 节）。备选是直接沿用 `.bak` 的 ES 分支（性能已经达标，但代码量大）。
- **v3-169 复现（11-26）**：3.1 那张表在 v3-169 上重跑，三个变体在 2k/4k 下都和原值相差 1 us 以内（3.10）。
- **路线 1b：2/2/0/0（11-26，已提交，已实测）**：只复用 v2 tile 里原本 load W2 的 K 循环槽位，A 不变。panel 的 chunk 0、chunk 1 各发下一个 panel 的一个 K-tile（各 10 条），chunk 2 以后不发。commit `0286007b4`，分支 `cguo/gemm2es_pf22`（第 6 节）；代码现在在 `mxmoe_dispatcher_route1b.py` / `mxmoe_gemm_v2_route1b.py`（第 5 节）。因为 A 仍然每个 tile 自己读，每份 W2' 大约只能在途 1 个 chunk（3.11）。实测（v3-058 GPU1）2k 107.3、4k 157.8，HEAD 是 106.4 / 146.6；降到 2 WG/CU 之后，2k 和 HEAD 打平，4k 慢 11 us（3.13）。
- **路线 1c（11-26，已实现、已实测，代码在 `mxmoe_dispatcher_route1c.py` / `mxmoe_gemm_v2_route1c.py`）**：开 ES 时也用 v2 的 BK=128 流水线，每个 tile 有 4 个 B 槽位。W2 分成前后两个半块（各 6+4 条）。每个 tile 的 kt1、kt2 槽位按顺序发一个半块：先补当前 panel 缺的后半块，再发下一个 panel 的前半块、后半块。pre-loop、kt0 只在当前 panel 的前半块不在寄存器里时照 v2 发（第 6 节）。实测 2k 126.8、4k 194.7，比 HEAD 和 1b 都慢，甚至比强制 2 WG/CU、完全不预取的 HEAD（123.9 / 177.8）还慢（3.13）。
- **1b / 1c 的教训（3.13）**：
  - W2' 能不能藏住，取决于等待精不精确，不取决于放在哪个槽位。1c 保留 v2 的 `vmcnt(0)`，W2' 在发出它的那个 tile 里就被等完了；1b 用精确的 `vmcnt(10)`，同样 2 WG/CU 下比不预取的 HEAD 快 17–20 us。
  - 不要在 load 外面包"目的寄存器不同"的运行时分支。即使两边条数相同，CFG 结构化之后汇合点也会让一条路径去等另一条路径的 load。要选的是来源地址，不是分支。

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
  - 选项：`--stage2` 时间通过 `kernel_bench_callable` 统计（GEMM2 加上后面的 reduce）；`--dump-dir` 保存 route rows 用于逐字节比对；`--es-wg-per-cu` 强制每 CU 的 WG 数；`--dispatcher <模块名>` 用 `aiter.ops.flydsl.kernels.<模块名>` 的 `mxfp4_moe_gemm2` 替换 `mxmoe_dispatcher` 的（测路线 1b、1c 用）。
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

### 3.10 在 v3-169 上复现 3.1（11-26，8 卡）

| 变体 | 2k v3-169 | 2k 3.1 | 4k v3-169 | 4k 3.1 |
|---|---|---|---|---|
| 原版 v2 + tuned tile | 118.8 | 118.8 | 160.1 | 160.8 |
| HEAD persist-es | 109.4 | 110.3 | 148.4 | 149.4 |
| `.bak` 完整 ES | 100.2 | 100.6 | 121.4 | 122.2 |

- 统计方法和 3.1 相同：每张卡 `--iters 100`，取 cold 的 **mean**，再对 8 张卡取中位数。用同一段汇总代码重算 3.1 的原始日志 `all_gpu*`，结果正好是 3.1 的数字。
- 环境：容器 `zhimding_cguo_repro`，镜像是本机的 `vllm/vllm-openai-rocm:latest`（`3cb773d23f06`，torch 2.13 + ROCm 10.0，自带 flydsl 0.4.2）。对 `aiter_head`（`4040a7109`）做 editable 安装，顺带装上 flydsl 0.5.3。`.bak` 用 `PYTHONPATH=aiter_f209cd555`。
- 和 3.1 的区别：3.1 是在合并 main 之前测的（flydsl 0.5.0），这次 `.bak` 是在 0.5.3 下跑的，结果不变。三个变体的 kernel 名和 3.1 的日志完全一致。

### 3.11 为什么 W2' 会被"等完"：vmcnt 按发射顺序计数

- `s_waitcnt vmcnt(N)` 的意思是"等到这个 wave 在途的 vector memory 操作不超过 N 条"。这些操作按发射顺序完成，gfx950 上 global load、LDS DMA（包括 async）和 store 共用这一个计数器。
- 硬件没有"只等某一条 load"的指令。要等 X，只能用 `vmcnt(X 之后发了几条)`，所以 X 之前发的所有 load 都要先完成。
- 语义上，下一个 panel 的 W2（下面记作 W2'）只需要在下一个 panel 的 MFMA 之前回来。但只要在它之后发了某条 load，并且在这之前就要等这条 load（每个 tile 自己的 A DMA、A scale、token id），等这条 load 时 W2' 就会被一起等完。
- tile 1 开头那一刻，A 在 W2' 后面发和在 W2' 前面发的区别：

```
A issued per tile (after G):           A issued one tile ahead (before G):

oldest -------------------> newest     oldest ---------------------> newest
[G0 x5][G1 x5][A1k0][A1k1]             [A1k0 A1k1 A1k2 A1k3][G0 x5][G1 x5]
               ^ want this                    ^ want these

s_waitcnt vmcnt(1)                     s_waitcnt vmcnt(10)
-> only A1k1 may stay in flight        -> G0 G1 stay in flight
-> G0 G1 must finish first
```

- 推论：一份 W2' 能在途多久，取决于从它发出到"第一条比它晚发、又要被等的 load"被等的那一刻，而不是它在循环里怎么穿插。
  - A 每个 tile 自己读（HEAD、路线 1b）：在途时间就是发出它的那个 tile 剩下的部分，不到 1 个 tile。
  - tile c 开头先发 A(c+1)，再发 W2'（路线 1 的第 1 步，以及 `.bak`）：大约 2 个 tile。这时如果 W2' 只在前两个 tile 发（2/2/0/0），最后一个 tile 期间这个 WG 就没有 W2' 在途了。3.3 里 4k "只分 3 份"（131.1）比分 5 份（123.1）慢，就是这种情况。

### 3.12 在 v3-058 上复现 3.1（11-26 下午，GPU1）

| 变体 | 2k v3-058 | 2k 3.1 | 4k v3-058 | 4k 3.1 |
|---|---|---|---|---|
| 原版 v2 + tuned tile | 118.2 | 118.8 | 157.2 | 160.8 |
| HEAD persist-es | 106.5 | 110.3 | 147.0 | 149.4 |
| `.bak` 完整 ES | 94.9 | 100.6 | 121.0 | 122.2 |

- 只用 GPU1，3 轮交替跑，取 cold mean 的平均。三个变体的 kernel 名和 3.1 的日志完全一致。
- 环境：容器 `cguo_gemm2es`，用本机的 `vllm/vllm-openai-rocm:latest` 镜像（torch 2.13 + ROCm 10.0）。HEAD 和 v2 用 `aiter_head` 加 flydsl 0.5.3（装在 `/opt/flydsl053`，通过 `PYTHONPATH` 引入）；`.bak` 用 `aiter_f209cd555` 加 flydsl 0.5.0。没有做 editable 安装。
- **这台机器 2k 的 ES kernel 卡间差异大**：`.bak` 2k 在 GPU0、GPU4 上稳定是约 100，在 GPU2/3/5/6/7 上是 95–97；GPU1 前三轮约 100、后两轮约 95，状态会变。warm 也有同样的差距，各卡 HBM 带宽（5.0–5.3 TB/s）、功耗上限和频率范围都一样，原因没查到。4k 和原版 v2 不受影响。所以 2k 的两个 ES 变体比 3.1 快了 4–6 us，而 v2 没变。3.13 里 HEAD 的 2k 也是 106 左右，说明 GPU1 当时同样处在"快"的状态；各变体在同一轮里交替跑，彼此可比。

### 3.13 路线 1b、1c 实测（11-26 下午，v3-058 GPU1）

| 变体 | 每 CU 的 WG 数 | 2k | 4k |
|---|---|---|---|
| HEAD persist-es | 3 | 106.4 | 146.6 |
| HEAD，强制 2 WG/CU（`--es-wg-per-cu 2`） | 2 | 123.9 | 177.8 |
| 路线 1b（`0286007b4`） | 2 | 107.3 | 157.8 |
| 路线 1c（最终版） | 2 | 126.8 | 194.7 |

- 3 轮交替跑，取 cold mean 的平均，轮间差异在 1 us 以内。1b、1c 的 route rows 在 2k、4k 下都和 HEAD **逐字节一致**。
- **降到 2 WG/CU 本身的代价**：HEAD 从 3 WG/CU 强制降到 2，2k 慢 17.5 us，4k 慢 31 us。1b 和 1c 都多带一套 W2 寄存器，都只能放 2 个 WG。
- **1b**：和同样 2 WG/CU 的 HEAD 比，2k 快 16.6 us、4k 快 20 us，预取是有效的。2k 基本抵消了 occupancy 的损失；4k 每个 panel 约 5 个 tile，W2' 只在前两个 chunk 发、每份在途不到 1 个 chunk，抵消不了。
- **1c**：比同样 2 WG/CU、完全不预取的 HEAD 还慢 3 us（2k）和 17 us（4k）。VGPR 234，无 spill，4 份 tile 代码（128 条 MFMA）。

**1c 为什么慢（看最终版的 ISA）**

- 非 lag 的 tile 0，各槽位已经和设计图一致：kt1 发 6 条 B 加 1 条 A scale，`ds_read` 之前是 `vmcnt(7)`；kt2 发 4 条，`ds_read` 之前是 `vmcnt(4)`。问题在后面：
  - kt3 在 C 复用 A 的 LDS 之前是 `vmcnt(0)`，把这个 tile 发的 10 条 W2' 全部等完，W2' 只和 MFMA k1、k2（共 16 条 MFMA）重叠；
  - epilogue 读 token id 之后是 `vmcnt(0)`；
  - 下一个 tile 开头读 `sorted_expert_ids` 之后也是 `vmcnt(0)`。
- 所以 W2' 的在途时间不到一个 tile。稳态下 HEAD 每个 panel 只在 tile 0 暴露一次（20 条并行发出）W2 延迟；1c 在 tile 0 和 tile 1 各暴露一次（各 10 条），延迟反而多一次。设计里"代价和 v2 在 kt1、kt2 load 自己的 k2、k3 一样"说的就是这个代价，但它在 HBM 延迟面前并不小。
- BK=128 在 BM=32 下，每个 kt 只有 8 条 MFMA。同步 LDS DMA 的保守等待（3.9）让 kt1 的 `ds_read` 等 kt0 才发的 A k2，kt2 等 kt1 才发的 A k3，A 的延迟一个 tile 暴露 3 次（HEAD 的 BK=256 是 1 次）。临时叠上 `async_a_v2_experiment.patch`（`AITER_V2_ASYNC_A=1`）只到 2k 128.0、4k 190.7，基本没变，所以这不是主因。

**实现过程中对设计的两处更正**

- **设计里"运行时分支最坏相当于 v2 的等待"不成立。** 第一版按设计在 kt1、kt2 用 `if/else` 选"当前 panel 的 h1 / W2' h0"等，两边条数相同：2k 134.8、4k 198.1（VGPR 229）。ISA 里 W2' 那条路径发完 6 条 load 之后，紧跟着 `vmcnt(9)`、`(8)`、`(7)`、`(6)`，kt2 发完 4 条之后是 `vmcnt(7)` 到 `(4)`，刚发的 load 立刻被等完。原因：
  - 两个分支往不同的寄存器里 load，汇合点要做 phi 拷贝；
  - AMDGPU 的 CFG 结构化把 if/else 变成"then 块 → 汇合块 → else 块"，else 块继承了 then 块的在途 load；
  - 所以 else 块里的拷贝要按 then 块的 load 计算等待（WAW）。
  另外，`SIInsertWaitcnts` 在汇合处取各条路径里更紧的等待，"一条路径发 load、另一条不发"时，发 load 的那条路径会被等到和不发的一样紧。
- **第二版**：load 总是进固定的暂存寄存器，来源地址在运行时选；当前 panel 缺后半块时，先把 h1 load 进暂存区，再在 MFMA k2、k3 之前用一个只做拷贝的 `if lag` 搬进 `bq[2..3]`。结果 2k 129.2、4k 196.2（VGPR 216）。LLVM 把这个拷贝部分改成了 `v_cndmask`，并把刚 load 的暂存寄存器提前拷贝，kt2 的 `ds_read` 之后多出一个 `vmcnt(0)`。
- **最终版**：在 panel 层面按 lag 分成两份 tile 0 代码，tile 内就不需要搬运了（见第 6 节 1c 的"实现"）。
- 规则：不要在 load 外面包"目的寄存器不同"的运行时分支，也不要在刚 load 的寄存器上做条件拷贝；要选的是来源地址，目的寄存器保持固定。分支汇合前要保证在途 load 已经被等完（比如放在 tile 之间）。

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
| tile 开头 pre-loop 和 kt0 连着发的 8 条 B value load 是瓶颈（路线 1c 的出发点） | 没有单独验证（设计里的对照变体没做）。1c 稳态下 pre-loop、kt0 已经没有 B load，但整体更慢：主因是 W2' 在发出它的 tile 内就被 `vmcnt(0)` 等完，再加上 2 WG/CU（3.13） |
| 路线 1c 里任何槽位的 B load 不多于 v2，所以最坏也是 v2 的等待 | 不对。目的寄存器不同的运行时分支会让一条路径去等另一条路径的 load；就算没有分支，保留 v2 的 `vmcnt(0)` 也会让 W2' 每个 tile 暴露一次延迟（3.13） |

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

### v3-169（11-26）

`/shared_nfs/cguo/` 只有 cguo 能写，所以拷了一份到 zhimding 可写的 `/mnt/m2m_nobackup/zhimding/cguo_repro/`，worktree 已经 repair 过：

- `aiter/`、`aiter_head/`、`aiter_f209cd555/`：内容同上。`aiter/` 的工作区仍然是路线 2 简化版，没有提交。两个 worktree 补上了各自 commit 对应的 CK 子模块（`f93c029e0` / `96afae1d5`）。
- `aiter_pf22/`：worktree，分支 `cguo/gemm2es_pf22`，路线 1b 的 commit `0286007b4`（基于 `4040a7109`，作者 `charlieguo1106 <cguo@amd.com>`，没有 co-author，**没有 push**）。CK 和 `aiter_head` 编好的 JIT `.so` 已经放进去了，可以直接测。
- `gemm2es_bench/`：脚本、patch 和日志（不含 `dump*/`、`isa*/`），另加 `pf22_w2n_0286007b4.patch`（路线 1b，可以直接 `git am` 到 `/shared_nfs/cguo/aiter`）。
- `logs/`：3.10 的日志 `all_gpu{0..7}_{v2_tuned,pes_v2,es_full}.log`。
- 容器 `zhimding_cguo_repro`：editable 安装的是 `aiter_head`，`/mnt/m2m_nobackup` 读写挂载，`/shared_nfs/cguo` 只读挂载。

```bash
D=/mnt/m2m_nobackup/zhimding/cguo_repro; cd $D/gemm2es_bench
E=(-e HIP_VISIBLE_DEVICES=0 -e AITER_MXFP4_INTERMEDIATE=1 -e AITER_MXFP4_G2_ES=1)
docker exec -w $PWD "${E[@]}" zhimding_cguo_repro python3 bench_gemm2es.py -t 2048 4096 --iters 100 --label pes_v2
docker exec -w $PWD "${E[@]}" -e PYTHONPATH=$D/aiter_f209cd555 zhimding_cguo_repro python3 bench_gemm2es.py -t 2048 4096 --iters 100 --label es_full
docker exec -w $PWD "${E[@]}" -e PYTHONPATH=$D/aiter_pf22 zhimding_cguo_repro python3 bench_gemm2es.py -t 2048 4096 --iters 100 --label pes_w2n
```

### v3-058（11-26 下午）

直接用 `/shared_nfs/cguo/`（cguo 可写），worktree 已经 repair 过：

- `aiter_pf22/`：worktree，分支 `cguo/gemm2es_pf22`。JIT `.so` 从 `aiter_head` 拷了过来。11-26 下午的提交（作者 `charlieguo1106 <cguo@amd.com>`，没有 co-author，**没有 push**）：
  - 路线 1b、1c 各自是一对新文件：`mxmoe_dispatcher_route1b.py` + `mxmoe_gemm_v2_route1b.py`（`0286007b4` 的代码），`mxmoe_dispatcher_route1c.py` + `mxmoe_gemm_v2_route1c.py`（1c 最终版）。新 dispatcher 只 import 自己的 gemm 文件，`fused_moe` 不会调用它们。
  - `mxmoe_dispatcher.py`、`mxmoe_gemm_v2.py` 恢复成 `4040a7109`，所以这个分支上 `AITER_MXFP4_G2_ES=1` 跑的是 HEAD persist-es。
  - 这份 test.md 的 3.12、3.13 以及相关更新。
  - 用 bench 的 `--dispatcher mxmoe_dispatcher_route1b`（或 `_route1c`）测，kernel 名分别带 `_w2n`、`_w2h`；提交前三条路径都重新编译测过，route rows 和 HEAD 逐字节一致。
- `gemm2es_bench/`：
  - `route1c_w2_halves_on_pf22.patch`：拆成新文件之前的 1c，直接改 `b1d9301d3` 的原文件（会去掉 1b 的改动）；`route1c_w2_halves_vs_4040a7109.patch`：同一份代码相对 `4040a7109` 的 diff。
  - `repro_v3058/`：3.12、3.13 的日志（`gpu1_r*_*.log`、`final_*.log`、`gpu1_pf22_1c*.log`），8 卡 `.bak` 的日志（`all_gpu*`、`r*_gpu*`、`seq_gpu*`），ISA（`isa/`），route rows（`dump/`），以及 1c 的一键脚本 `run1c.sh`（GPU1：测速、和 HEAD 的 dump 逐字节比对、ISA 统计）。
- 容器 `cguo_gemm2es`：镜像是本机的 `vllm/vllm-openai-rocm:latest`，site-packages 里装的是 flydsl 0.5.0（给 `.bak`），0.5.3 装在 `/opt/flydsl053`。容器以 root 写出的文件要在容器里 `chown` 回来，否则宿主机上删不掉。

```bash
cd /shared_nfs/cguo/gemm2es_bench
E=(-e HIP_VISIBLE_DEVICES=1 -e AITER_MXFP4_INTERMEDIATE=1 -e AITER_MXFP4_G2_ES=1 -e FLYDSL_RUNTIME_CACHE_DIR=/root/.flydsl/cache053)
docker exec -w $PWD "${E[@]}" -e PYTHONPATH=/opt/flydsl053:/shared_nfs/cguo/aiter_head cguo_gemm2es python3 bench_gemm2es.py -t 2048 4096 --iters 100 --label head
docker exec -w $PWD "${E[@]}" -e PYTHONPATH=/opt/flydsl053:/shared_nfs/cguo/aiter_pf22 cguo_gemm2es python3 bench_gemm2es.py -t 2048 4096 --iters 100 --label 1c --dispatcher mxmoe_dispatcher_route1c
docker exec -w $PWD -e HIP_VISIBLE_DEVICES=1 -e AITER_MXFP4_INTERMEDIATE=1 -e AITER_MXFP4_G2_ES=1 -e PYTHONPATH=/shared_nfs/cguo/aiter_f209cd555 cguo_gemm2es python3 bench_gemm2es.py -t 2048 4096 --iters 100 --label es_full
```

## 6. 后续修改与优化计划

会话最后在问"路线 2 接下来怎么走"时你取消了，所以这一步还没定。下面按推荐程度排列。11-26 先按你的思路实现了路线 1b；下午实测了 1b，并实现、实测了路线 1c，1c 结果不好，暂时搁置（3.13）。

### 路线 1b：2/2/0/0（11-26 已实现，已实测，见 3.13）

出发点是代码简单：只复用 v2 tile 里原本 load W2 的 K 循环槽位，A 的发射位置不变。commit `0286007b4`（分支 `cguo/gemm2es_pf22`），只改了 `mxmoe_dispatcher.py` 和 `mxmoe_gemm_v2.py`，+140/−57 行。之后这份代码挪到了 `mxmoe_dispatcher_route1b.py` / `mxmoe_gemm_v2_route1b.py`，原文件恢复成 `4040a7109`。

- **寄存器**：WG 多带一套 W2 寄存器（`bq_n` / `bs_n`，约 68 个 VGPR）。
- **发射位置**：panel 的 chunk 0 在 v2 预取 W2 的那个槽位（kt0 开头，`bhoist` 的位置），发下一个 panel 的 K-tile 0（8 条 value + 2 条 e8m0，即 G0 G1）。chunk 1 在同一个位置发 K-tile 1（G2 G3）。chunk 2 以后不发。下一个 panel 开头把 `bq_n` 拷到 `bq`，只有 WG 的第一个 panel 的 chunk 0 自己 load W2。
- **精确等待**：本 chunk 的 token id 从 epilogue 提前到 W2' 之前读。kt1 处 C 复用 A 的 LDS 之前那个 `vmcnt(0)`，以及 epilogue 等 token id 的 `vmcnt(0)`，都改成 `vmcnt(10)`，所以 W2' 在本 chunk 内不会被等完。
- **其他**：
  - chunk 1 总是执行。只有一个 chunk 的 panel 也会跑 chunk 1，它的行落在 padding 里，不会 store，因此 ES 的开启条件加了 `SBM >= 64`。
  - kernel 名多了 `_w2n`，不会命中旧的编译缓存。
  - WG 的最后一个 panel 会重读一遍自己的 W2，结果不受影响。
- **预期**（都没测过）：
  - 每份 W2' 大约在途 1 个 chunk。下一个 chunk 开头 ds_read A 之前，LLVM 会为 A 的 LDS DMA 插一个 vmcnt，把上一个 chunk 发的 W2' 一起等完（3.11）。
  - VGPR 应该和路线 1 差不多（208），也就是 2 WG/CU。
  - 和路线 1 的区别：路线 1 把整块 W2' 都放在 tile 1 发；1b 把 tile 0 的槽位也用上了，tile 0、tile 1 各发一个 K-tile。
  - 参考 3.8：在 `.bak` 上"第一、第二个 tile 各 10 条"是 106.6（2k），那时 A 已经提前 2 个 chunk。1b 的提前量更短，又是 2 WG/CU，预计不会比 106.6 好；它和 HEAD（110）谁快要实测。
- **测试步骤**：
  1. 用 `--dump-dir` 把 route rows 和 HEAD 的逐字节比对。W2 的值和累加顺序都没变，应该逐字节一致。
  2. 8 卡跑 2k/4k（8k 要用 `glm5_reduce_large.csv`）。
  3. 看 ISA：VGPR 和每 CU 的 WG 数；chunk 0、1 的 kt1 处和 epilogue 是不是 `vmcnt(10)`；下一个 chunk 第一个 ds_read 之前的 vmcnt 是多少。
- **实测**（v3-058 GPU1，3.13）：route rows 和 HEAD 逐字节一致；2 WG/CU；2k 107.3、4k 157.8（HEAD 106.4 / 146.6，强制 2 WG/CU 的 HEAD 123.9 / 177.8）。比预期的好：同样 2 WG/CU 下预取省了 17–20 us，2k 和 3 WG/CU 的 HEAD 打平。ISA 和 8k 还没看。

### 路线 1c：ES 也用 v2 的 BK=128 流水线，W2 按半块放进 kt1、kt2 的槽位（11-26 设计；已实现、已实测，见 3.13；代码在 `mxmoe_*_route1c.py`，暂时搁置）

出发点：

- ES 关时，v2 保持原样，kernel 不变。
- 开 ES 时，tile 也保持和 v2 tuned 一样的 BK=128（KT=4）流水线，每个 tile 有 4 个 B 槽位：pre-loop 的 k0，以及 kt0、kt1、kt2 分别预取的 k1、k2、k3。
- 看重的是**槽位里的 load 变少**，而不是尽量多预取：下一个 panel 的 W2（下面记作 W2'）只能替换 v2 原有的 B load，不往已经有 load 的槽位里加。所以任何槽位里的 B load 都只会和 v2 一样或者更少。每个 expert 只有一个 tile 时，完全退回 v2。
- 你的判断：tile 开头 pre-loop 和 kt0 连着发的 8 条 B value load（k0、k1 各 4 条）是瓶颈。
- 一个 panel 的 20 条 W2 分成两个半块，每个半块正好占一个 tile 的 kt1、kt2 两个槽位（6 + 4 条）：
  - **前半块 h0**：kt1 发 k0 的 4 条 value 和 scale c0 的 2 条，kt2 发 k1 的 4 条。
  - **后半块 h1**：kt1 发 k2 的 4 条 value 和 scale c1 的 2 条，kt2 发 k3 的 4 条。

v2 BK=128 一个 tile 的发射顺序（基线，`bhoist` 打开）：

```
tile start  A k0, A k1                     (dispatcher, LDS DMA, kStages = 2)
pre-loop    B k0, scale c0 (B + A)
kt0         B k1                   | BAR | ds_read A k0 | A k2 | MFMA k0
kt1         B k2, scale c1 (B + A) | BAR | ds_read A k1 | A k3 | MFMA k1
kt2         B k3                   | BAR | ds_read A k2 |      | MFMA k2
kt3                                | BAR | ds_read A k3 | vmcnt(0) lgkmcnt(0), BAR | MFMA k3 + C ds_write
```

规则：

- **pre-loop、kt0**：当前 panel 的 h0 不在寄存器里时，照 v2 发它（k0、k1）。否则这两个槽位里只剩 A scale c0。
- **kt1、kt2**：每个 tile 按顺序发下面第一个还没发的半块，最多发一个：
  1. 当前 panel 的 h1（不在寄存器里时）。这就是 v2 在 kt1、kt2 load 自己 k2、k3 的原样。
  2. W2' 的 h0。
  3. W2' 的 h1。

  三项都发过就不发。
- 槽位里 B value、B scale、A scale 的顺序保持 v2 原样，A scale 不调整。等 A scale 时如果顺带把 B 也等完了，那就是 v2 现在的情况，可以接受。
- 下一个 panel 开始时，已经发出的 W2' 半块就是它已经在寄存器里的半块。

所以，一个 W2 的后半块总是在"发它前半块的那个 tile"的下一个 tile 的 kt1、kt2 发出：

- 如果这个 tile 还属于当前 panel，就是预取。
- 如果当前 panel 只有一个 tile，这个 tile 就已经是下一个 panel 的 tile 0。它本来就要在 kt1、kt2 load 自己的 k2、k3，和 v2 一样。

两种情况下逻辑都成立。代价是再下一个 panel 的 W2 预取会往后推一个 tile，或者干脆不预取，那个 panel 就退回 v2。

稳态下 panel 的 tile 0（当前 panel 的 W2 已经全部在寄存器里）：

```
tile start  A k0, A k1                          (dispatcher, LDS DMA, kStages = 2)
pre-loop    A scale c0                          (B k0, B scale c0 not needed)
kt0         -                                    | BAR | ds_read A k0 | A k2 | MFMA k0   (B k1 not needed)
kt1         W2' B k0, W2' B scale c0, A scale c1 | BAR | ds_read A k1 | A k3 | MFMA k1
kt2         W2' B k1                             | BAR | ds_read A k2 |      | MFMA k2
kt3                                              | BAR | ds_read A k3 | vmcnt(0) lgkmcnt(0), BAR | MFMA k3 + C ds_write
```

tile 1 相同，只是 kt1 发 W2' 的 B k2 和 B scale c1，kt2 发 W2' 的 B k3。tile 2 以后没有 B load。

几种情况下每个 tile 的槽位内容（`own` 是当前 panel 自己的 W2）：

```
case                           tile  pre-loop / kt0   kt1 / kt2
-----------------------------  ----  ---------------  ---------------------------
all of W2 prefetched           0     -                W2' h0
  (steady state)               1     -                W2' h1
                               2..   -                -

only h0 prefetched             0     -                own h1 (v2: k2, k3)
  (prev panel issued only h0)  1     -                W2' h0
                               2     -                W2' h1

nothing prefetched             0     own h0 (v2)      own h1 (v2)   -> same as v2
  (WG's first panel, or        1     -                W2' h0
   prev panel issued no W2')   2     -                W2' h1
```

panel 的 tile 不够时，后面的行就不存在，没发出去的半块就不预取。如果每个 expert 都只有一个 tile，每个 panel 的 tile 0 都要先补自己的 h1，kt1、kt2 没有空位留给 W2'，几个 panel 之内就会落到 "nothing prefetched"，也就是 v2。

要改的地方：

1. **dispatcher**：ES 现在强制 `BM, BN, BK = 32, 256, 256`，改成 BK=128（BM 仍是 32），并放开 `INTER_MAX // BK <= 2` 的断言（KT=4）。
2. **当前 panel 的 W2 常驻**：现在靠 2 个 K-tile 的 B double buffer（cur 和 nxt 互换）。KT=4 时要改成 4 块 `bq[kt]`，寄存器总量不变（4×16 = 64 VGPR），scale 仍是 2 个 slot（`n_slots` 本来就是 2）。ES 时 K 循环按 `bq[kt]` 取，不再互换。
3. **W2' 的寄存器**：一套 `bq_n[4]` / `bs_n[2]`（约 68 VGPR），下一个 panel 开始时拷到 `bq` / `bs`。
4. **运行时状态**：每个 WG 记两个数：当前 panel 已经在寄存器里的半块数（0、1、2），以及本 panel 的 kt1、kt2 已经发到上面规则里的第几项。每个 tile 据此在运行时决定槽位里发什么，所以这些 load 外面要包一层分支。只有 tile 0–2 可能有 B load，tile 3 以后不用判断。
5. **A 和等待都不变**：A 按 v2 原样流水（BM=32、BK=128 时 `aStages = 3 < KT = 4`：prologue 发 k0、k1，kt0 发 k2，kt1 发 k3）。kt3 和 epilogue 的 `vmcnt(0)` 保持原样。
6. **不需要空跑 chunk 1**：后半块自然落在下一个 tile 里，所以也不需要 `0286007b4` 里的 `SBM >= 64`。

和 `0286007b4`（路线 1b）的区别：1b 是 BK=256，每个 tile 只有 kt0 一个槽位，panel 的 chunk 0、1 一定各发一个 K-tile，所以要空跑 chunk 1。1c 用 v2 的 4 个槽位，W2 按半块滚动，tile 不够时退回 v2。

预期（没测过）：

- 稳态下 panel 开头不用再等自己的 W2。HEAD 是在 chunk 0 开头发 20 条，紧接着就要等。
- 每个槽位的 B load 不多于 v2。稳态下 pre-loop、kt0 只剩 A scale，tile 2 以后没有 B load。
- 等待保持 v2 原样时，W2' 会在发出它的那个 tile 的 kt3（`vmcnt(0)`）被等完，只和 MFMA k1、k2 重叠。代价和 v2 在 kt1、kt2 load 自己的 k2、k3 一样。
- 运行时分支会让 LLVM 在分支汇合后用更保守的 vmcnt。~~因为槽位里的 load 不多于 v2，最坏也相当于这些 load 都发了，也就是 v2 的等待。~~ **这一条是错的**：汇合处取的是各条路径里更紧的等待，目的寄存器不同时还会让一条路径去等另一条路径的 load，刚发出的 W2' 会被立刻等完（3.13）。最终实现里 tile 0、tile 1 的 load 外面没有分支。
- VGPR 多出约 68 个，预计和路线 1、1b 一样降到 2 WG/CU。

待验证：

- **"开头 8 条 B load 是瓶颈"这个假设**：做一个对照变体，把 W2' 的半块放在 pre-loop、kt0 槽位，其余不变，直接比。HEAD 的 ES（BK=256）在非首个 chunk 里本来就不发 B load，也可以拿来对照。
- **VGPR 和每 CU 的 WG 数**；tile 0、1 的槽位内容和 vmcnt 是否和上面的图一致。
- **正确性**：BK=128 和 BK=256 对每个累加器都是按 k0 → k3 的顺序做 128-K 的 MFMA，e8m0 word 也相同，所以 route rows 应该和 HEAD 逐字节一致，需要用 `--dump-dir` 确认。（已确认：2k、4k 逐字节一致。）

**实现（最终版，`mxmoe_dispatcher_route1c.py` / `mxmoe_gemm_v2_route1c.py`）**

- **dispatcher**：
  - ES 改成 BM=32、BN=256、BK=128（`use_nt` 不变）；断言改成 BK=128 且 INTER_MAX=512；去掉 1b 的 `SBM >= 64`；
  - kernel 名里是 `_bn256_bk128` 和 `_pes_cu256_ne257_w2h`。
- **寄存器**：
  - 当前 panel 的 W2：`bq[4]`（每个 K tile 一块）和 `bs[2]`（每个半块一组 scale）；
  - W2' 一套：`bq_n` / `bs_n`，分成两个暂存半块，A = (`bq_n[0..1]`, `bs_n[0]`)，B = (`bq_n[2..3]`, `bs_n[1]`)。
- **panel 循环的携带变量**：
  - `res`：当前 panel 已在寄存器里的半块数；
  - `res_lag`：上一个 panel 是否 lag，决定 A、B 里两个半块的顺序。
  - panel 开头按 `res_lag` 把 A、B 拷进 `bq` / `bs`；`lag = res < 2`。
- **chunk 0**：在 panel 层面分成两份代码，汇合点在 tile 之间，那时 kt3 和 epilogue 已经 `vmcnt(0)`。
  - lag 时就是 v2 的 tile：kt1、kt2 直接把自己的 h1 load 进 `bq[2..3]`；`res == 0` 时 pre-loop、kt0 还要 load h0（这里有个 `if`，只在 WG 的第一个 panel 或者跟在单 tile panel 后面时走到）。
  - 不 lag 时：kt1、kt2 把 W2' h0 load 进 A，没有分支。
- **chunk 1**（`n_chunks > 1` 时）：把 W2' 的半块 1 − lag load 进 B。半块号是运行时值，只影响地址，没有分支。
- **chunk 2 以后**（运行时循环）：只有 `c − lag == 1`（也就是 lag 且 c = 2）时，才把 W2' h1 load 进 A（A 里自己的 h1 那时已经用完）。这是唯一一处包在 load 外面的分支，发 load 的是少见路径。
- **panel 结束**：`res = clamp(n_chunks − lag, 0, 2)`，`res_lag = lag`。
- **WG 的最后一个 panel**：W2' 的 buffer 描述符 `num_records = 0`，load 照常发出（等待计数不变）但不访存，不再像 1b 那样重读自己的 W2。
- **A 和等待**：保持 v2 原样，去掉了 1b 的 `vmcnt(10)` 和提前读 token id。

**结论**：2k 126.8、4k 194.7，比不预取的 HEAD（同样 2 WG/CU）还慢。这个方向先搁置。要让 W2' 真正藏住，它必须跨过三个 `vmcnt(0)`：kt3 的、epilogue 读 token id 的、下一个 tile 开头读 `sorted_expert_ids` 的。对应的办法：
- token id 在 W2' 之前读，kt3 和 epilogue 用精确的 vmcnt（1b 已经做了）；
- expert id 直接用 expert table 里的值（下面路线 1 + async DMA 的第 3 步）；
- 下一个 tile 的 A 在 W2' 之前发（同一节的第 1 步）。

这些做完之前，调整 W2' 放在哪个槽位意义不大。

### 路线 1 + async DMA（推荐：最贴近现有 v2，改动可控）

从 `route1_v2_tile1_prefetch.patch` 出发，每做一步都用 bench 测 2k/4k/8k，和 HEAD 的 dump 逐字节比对，并检查 ISA 里的 vmcnt 和 VGPR/每 CU 的 WG 数。目标是 2k ≈ 101、4k ≈ 124、8k ≈ 213。

1. **A 跨 tile 预取（收益最大）**：
   - A 的 DMA 换成 `BufferLoadAsyncLDS128b` 加 `asyncmark`。
   - tile c 在发 W2 预取**之前**，先把 tile c+1 的 A 发进另一块 LDS buffer。两块 8 KB 的 A buffer 都要放在 16 KB 的 bf16 C staging 之外，每 WG 共约 32 KB。（原来写的"多 8 KB、约 24 KB"不够：只要有一块 A buffer 和 C staging 重叠，tile c+1 的 epilogue 写 C 时就会覆盖正在落地的 A(c+2)。2 WG/CU 时一共 64 KB，LDS 不是瓶颈。）
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

## 7. 路线 1d 的做法里，哪些可以用到非 ES 的 v2 layout（11-27，只记录，v2 代码没改、没测）

### 7.1 背景：路线 1d（11-27，v3-111 单卡）

路线 1d 是在 1c 上把等 W2' 的 `vmcnt(0)` 都去掉。代码在 `mxmoe_dispatcher_route1d.py` / `mxmoe_gemm_v2_route1d.py`（kernel 名带 `_w2hd`，用 bench 的 `--dispatcher mxmoe_dispatcher_route1d` 测），**还没提交**。

环境：v3-111 的 Spur 作业只分到一张卡（PCI `0002:00:03.0`）；容器 `cguo_gemm2es_v033`，镜像 `vllm/vllm-openai-rocm:v0.33.0`（torch 2.13 + ROCm 10.0，自带 flydsl 0.4.0），只透传这张卡，容器里 `HIP_VISIBLE_DEVICES=0`。flydsl 0.5.3 装在 `/opt/flydsl053`（HEAD、1b、1c、1d 用），0.5.0 装在 `/opt/flydsl050`（`.bak` 用），都通过 `PYTHONPATH` 引入。日志、ISA、`run1d.sh`（测速 + 和 HEAD 的 dump 逐字节比对 + ISA 统计）在 `gemm2es_bench/repro_v3111/`。

3 轮交替跑，取 cold mean 的平均（轮间差异不超过 ±0.7 us）：

| 变体 | 每 CU 的 WG 数 | 2k | 4k |
|---|---|---|---|
| HEAD persist-es | 3 | 106.2 | 146.9 |
| HEAD，强制 2 WG/CU | 2 | 124.0 | 179.2 |
| 路线 1b | 2 | 106.7 | 158.6 |
| 路线 1c | 2 | 127.6 | 196.4 |
| 路线 1d | 2 | **98.2** | **131.1** |
| `.bak` 完整 ES | 2 | 94.8 | 119.9 |

逐步的改动（单次测量，2k / 4k，从 1c 的 127.5 / 196.7 出发；下文用"1d 第 N 步"引用）。route rows 每一步都和 HEAD 逐字节一致：

1. kt1 先发 A k3 的 DMA 和 A scale，再发 W2'；token id 在 W2' 之前读；kt3、epilogue 的 vmcnt 用编译期算出的条数：111.5 / 170.6。
2. expert id 用 expert table 里的值，去掉 lag tile 开头读 `sorted_expert_ids` 之后的 `vmcnt(0)`：111.4 / 170.1。
3. 下一个 chunk 的 A（4 个 K-tile）、A scale、token id，在当前 chunk 开头、W2' 之前，用 async LDS DMA 发到另一块 LDS buffer；C staging 有自己的 16 KB，kt3 不再需要 vmcnt 和 barrier：101.7 / 147.0。
4. 去掉 lag：chunk 0、1 总是执行（同 1b，要求 `SBM >= 64`），chunk 0、1、2 之间没有分支；WG 第一个 panel 自己的 W2 在 chunk 0 开头、预取之前 load：99.7 / 144.4。原因是分支汇合时 LLVM 要拷贝在途的 W2' 寄存器，拷贝前插 `vmcnt(0)`（ISA 里是 `v_accvgpr_write` 前的 `vmcnt(0)`）。试过按"不 lag 且 ≥ 3 个 chunk"另开一条无分支路径：tile 代码变成 9 份，VGPR 256、spill 84 个，188 / 236，放弃。
5. chunk 循环的两个代码生成问题：从循环变量算出的地址在 VGPR 里，A scale 的 buffer 描述符被编成 waterfall 循环（`v_cmp_eq_u64` + `s_cbranch_execnz`，加 `readfirstlane` 解决）；循环的回边从 epilogue 的分歧 store 分支里出来，整个循环成了分歧循环（`s_andn2_b64 exec`）：99.2 / 138.7。
6. A scale 和 token id 也走 LDS DMA，和 A 同一组：不再有跨迭代携带的在途寄存器（它们在回边上被拷贝，拷贝前 `vmcnt(0)`，连 store 一起等）：97.5 / 133.2。
7. route row 的 store 改成无条件的 buffer store，无效行的地址指到 route rows 末尾之后、被丢弃：各路径的 vm 指令条数一样，汇合后的等待不再多等 W2' 的前 4 条：97.1 / 131.5。
8. expert table 的行提前一个 panel 读，在循环体末尾 `readfirstlane`：panel 开头不再等刚发出的 global load（那里会连带等完上一个 chunk 的 store）：约 97.5 / 130.9。

最终 ISA：VGPR 214（1c 是 234）、LDS 34 KB，没有 spill，没有 exec 分支；只有 panel 循环之前还有一个 `vmcnt(0)`。chunk 1、2 开头是 `vmcnt(19)`（10 条 W2'、4 条 store、5 条预取都在途），W2' h0 在途到 chunk 2 开头，h1 在途到 chunk 3 开头或 panel 结束。和 `.bak` 的差距主要在 4k（11 us）：4k 每个 panel 约 5 个 chunk，W2' 只在 chunk 0、1 里发；`.bak` 是 8/3/3/3/3 分到 5 个 chunk。下一步可以按每个 expert 的行数在编译期选份数（比如 2k 分 2 份、4k 分 4 份）。

### 7.2 非 ES v2 tuned 的现状（v3-111，`AITER_MXFP4_G2_ES=0`，`aiter_head`）

- kernel：`…bm64_bn256_bk128_reduce_tk9_sbm128_persist_cu256…`（4k 没有 `_sbm128`）。2k 117.8、4k 156.2（单次）。ISA 在 `repro_v3111/isa/v2_tuned*`。
- VGPR 138、LDS 32 KB（C staging 32 KB，A 的 3 个槽位和它共用），每 CU 3 个 WG（3 WG 的 VGPR 上限是 168；LDS 每 CU 160 KB）。没有 spill，没有 waterfall 循环；persist 循环是 uniform 的，4 个 exec 分支都是 epilogue 的 store。
- persist-m：网格 256 × 24 = 6144 个 WG，同时驻留 768 个。每个 WG 固定一个 N-block，按 m 步长 256 处理 m-tile；按 sort 的填充估算，2k 每个 WG 约 2 个 m-tile，4k 约 3 个。
- 每个 tile 的执行顺序和等待（2k、4k 的 ISA 结构相同）：

```
tile start  BAR | A k0, A k1 (LDS DMA) | if m_block < total | sorted_expert_ids (global_load), A scale
            vmcnt(1)                  <- waits the expert id and both A DMAs
            readfirstlane e | B scales, A scale, B k0 (x4), B k1 (x4)
kt0         BAR | ds_read A k0 | A k2 | vmcnt(8..5) MFMA k0   <- B k0 latency, after the expert id's
kt1         B k2, scales c1 | vmcnt(8) BAR | ds_read A k1 | A k3 | MFMA k1
kt2         B k3 | vmcnt(4) BAR | ds_read A k2 | MFMA k2
kt3         BAR | ds_read A k3 | vmcnt(0) lgkmcnt(0) BAR | MFMA k3 + C ds_write
epilogue    BAR | token ids (buffer_load x4) | vmcnt(0) | 4 x (if valid: global_store x2)
```

- 每个 tile 暴露三段访存延迟：tile 开头 expert id 和 B k0 串行两段（A 和 expert id 并行），epilogue 的 token id 一段。只靠同一 CU 上另外 2 个 WG 掩盖。

### 7.3 可以尝试的改动

**A. 不改流水线结构，不增加 LDS**

1. **expert id 提前一个 tile 读**（对应 1d 第 2、8 步）。persist-m 下一个 tile 的 m_block 已知（当前 + 256，越界时 clamp）。在当前 tile 里发它的 `sorted_expert_ids` load，到循环体末尾再 `readfirstlane`，作为 SGPR 带进下一轮。下一个 tile 开头 B scale、B k0、B k1 可以和 A 同时发，关键路径少一段 expert id 的延迟。
   - `readfirstlane` 要放在循环体末尾。放在循环头的话，循环头汇合时按 prologue 那条路径算（load 刚发出），每一轮都会变成 `vmcnt(0)`（1d 第 8 步就是这么处理的）。
   - 最小版本：只把 expert id 的读挪到 A DMA 之前（现在 A DMA 由 dispatcher 先发，body 里才读 expert id）。等 expert id 时不再连带等 A 的 DMA，但 expert id 到 B k0 还是串行的。
   - 也可以看能不能让 LLVM 把它编成 scalar load（走 lgkmcnt，完全不碰 vmcnt）。现在循环里是 `global_load`，因为编译器不能证明它和 store 不别名；flydsl 有没有办法强制，没查。
2. **token id 提前读**（1b、1d 第 1 步）。把 epilogue 的 4 条 token id load 挪到 tile 前段（比如 kt0 发完 B 之后），epilogue 只剩 LDS 读和 store。多 4 个 VGPR（138 → 约 142），仍是 3 WG/CU。发在 B k(t) 之后的话，会在 MFMA k(t+1) 处被顺带等完；它和 B 同时在途，不额外暴露延迟。
3. **去掉 persist 循环里多余的 `if m_block < total_m_blocks`**：`n_iters` 已经保证 m_block 在范围内。现在无害（tile 之间没有在途的 load），但只要做了下面任何一种跨 tile 预取，这个包住整个 tile 的分支就会在汇合点让 LLVM 取更紧的等待，并可能拷贝在途寄存器（3.13、1d 第 4 步）。
4. **store 改成无条件的 buffer store**（1d 第 7 步）：无效行的地址指到 route rows 末尾之后，靠 buffer 的 num_records 丢弃。v2 的 fp4 route row 和 ES 一样是每行 8 字节值加 1 字节 e8m0，可以直接照搬 `mxmoe_gemm_v2_route1d.py` 里 `atomic_bf16_epilog` 的 `uncond_stores`。单独做也能去掉 epilogue 的 4 个分歧分支（每个 tile 4 次 exec 切换，BM=64 时 M_REPS=4）；做了跨 tile 预取以后，它还保证各路径的 vm 指令条数一致，汇合后的等待不会多等在途的 load。

**B. 跨 tile 预取**

5. **下一个 tile 的 B k0（加 B scale c0、A scale c0）放到当前 tile 的 kt3 槽位**（1b 的思路用在 v2 上）。v2 在 kt3 没有 B load（`kt + 1 < KT` 不成立），这时 `nxt_bqf`、`bsf_slots[0]`、`saf_slots[0]` 都已经用完，不增加 VGPR。前提是 1（下一个 tile 的 expert id 已知）。要让它跨过 tile 边界在途：
   - kt3 的 `vmcnt(0)` 改成精确计数：它只是为了等 A 的 DMA 落地，B k0' 在 A k3 之后发。
   - token id 已经提前读（2），epilogue 不再 `vmcnt(0)`。
   - B 的 cur/nxt 双缓冲在 tile 之间的角色要固定（kt3 发进 nxt，下一个 tile 的 kt0 从它读），类似 ES 的 `es.bq`。
   - 单独做 5 时，下一个 tile 开头等 A k0/k1 的那个等待会把 B k0' 一起等完（A 排在它后面发），所以 B k0' 只在途"MFMA k3 + epilogue"这一段。ES 上类似的 1d 第 1 步把 127.5 降到 111.5。要它在途更久，需要 6。
6. **下一个 tile 的 A（整块 4 个 K-tile）、A scale、token id 用 async LDS DMA 提前一个 tile 发**（1d 第 3、6 步）。tile 开头只用 `wait_asyncmark(1)` 等本 tile 那一组（上一个 tile 发的），K 循环里不再有 A 的 DMA 和逐 kt 的 barrier。
   - LDS：BM=64 时一块 A 是 16 KB，C staging 是 32 KB。照搬 1d（C 独立加两块 A）要约 66 KB，每 CU 只能放 2 个 WG，丢掉第 3 个 WG 很可能不划算（3.13 里 HEAD 从 3 降到 2 WG/CU 慢了 17–31 us）。
   - 可以让 C 复用当前 tile 的 A buffer：三块 16 KB 的 R0、R1、R2。偶数 tile 的 A 在 R0，C 写 R0+R1，下一个 tile 的 A 发到 R2；奇数 tile 的 A 在 R2，C 写 R1+R2，下一个发到 R0。A scale、token id 的尾部（BM=64 时 1 KB + 256 B）放在这三块之外，两份。总共约 51 KB，3 个 WG 约 152 KB，160 KB 放得下。kt3 仍要 barrier（C 覆盖本 tile 的 A），但不需要 `vmcnt(0)`。
   - 收益受每个 WG 的 tile 数限制：2k 约 2 个、4k 约 3 个，只有第 2 个 tile 起才受益。可以同时试更大的 m 步长（每个 WG 处理更多 m-tile、网格更小），这是另一个调度旋钮，要实测。
   - 做了 6 之后，5 的 B k0' 可以一直在途到下一个 tile 的 MFMA k0；再往后，B k1' 也可以放进 epilogue 之前的槽位（相当于 v2 版的"下一个 panel 的 W2 一半"）。

**C. 做了跨 tile 预取之后要检查的 LLVM 行为**（1d 的教训；每一步都看 ISA 的 vmcnt 列表，确认想留在途的 load 没在某处被等完）

7. **分支汇合点**：有 load 在途时，汇合处可能拷贝它的目的寄存器（拷贝前插等待），而且等待取各条路径里更紧的那个（3.13、1d 第 4 步）。稳态的 tile 代码里不要有包住 load 的运行时分支；必须有的（比如 WG 的第一个 tile），就把分支里的 load 放在其他 load 之前发。
8. **循环的回边不要从分歧分支里出来**（1d 第 5 步）：否则整个循环成了分歧循环，循环携带的寄存器在回边上拷贝、`vmcnt(0)`。ISA 里出现 `s_andn2_b64 exec` 就是这个问题。做了 4 之后 epilogue 没有分歧分支，这个问题也就不会出现。
9. **从循环变量算出的地址、buffer 描述符要 `readfirstlane`**（1d 第 5 步）：否则 buffer load 会被编成 waterfall 循环。v2 现在没有，但加了"下一个 tile"的地址计算之后可能出现。
10. **跨迭代携带的在途寄存器**（下一个 tile 的 A scale、token id、expert id）会在回边上被拷贝并等完：要么放进 LDS（和 A 同一组 DMA，1d 第 6 步），要么在循环体末尾就 `readfirstlane` 或用掉。

**D. 已经测过的**

11. tile 内的 A 改用 async LDS DMA（3.9）：v2 tuned 快约 1%（2k 121.3 → 120.0，4k 164.0 → 162.4）。6 需要它。

### 7.4 只适用于 ES、v2 用不上的

- W2 在 panel 的各个 chunk 之间常驻、预取下一个 panel 的 W2：v2 的连续 m-tile 一般是不同的 expert，B 不能复用。
- 按 panel 顺序的调度、`place_half`、"chunk 0、1 总是执行"（`SBM >= 64`）。
- expert table：v2 的 sort 不输出 `aiter_expert_info`（bench 里 `expert_info=False`），对应的做法是 1，提前读 `sorted_expert_ids`。

### 7.5 建议顺序和测法

- 顺序：1、2、3、4 互相独立，不改流水线、不增加 LDS，风险最小，先做；然后 5；最后 6（需要 3、4 和 C 里的检查）。
- 改 `mxmoe_gemm_v2.py` 会同时影响 v2 和 HEAD 的 ES 路径（共用 tile 代码）；kernel 名不变时要加 `-e FLYDSL_RUNTIME_ENABLE_CACHE=0`。改之前先用原代码 dump 一份 v2 的 route rows 做基准（这次没有存 v2 的 dump）：

```bash
cd /shared_nfs/cguo/gemm2es_bench
E=(-e HIP_VISIBLE_DEVICES=0 -e AITER_MXFP4_INTERMEDIATE=1 -e AITER_MXFP4_G2_ES=0 -e FLYDSL_RUNTIME_CACHE_DIR=/root/.flydsl/cache053)
docker exec -w $PWD "${E[@]}" -e PYTHONPATH=/opt/flydsl053:/shared_nfs/cguo/aiter_pf22 cguo_gemm2es_v033 \
  python3 bench_gemm2es.py -t 2048 4096 --iters 100 --label v2_ref --dump-dir repro_v3111/dump
```

- ISA 用 `FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_DIR=...` 导出，看 VGPR、每 CU 的 WG 数（`es_wg_per_cu` 只对 ES 有效；v2 看 VGPR ≤ 168、LDS ≤ 53 KB 才能 3 WG/CU）和 vmcnt 列表。`repro_v3111/run1d.sh` 可以改一下 dispatcher 和 `G2_ES` 直接用。容器以 root 写出的文件（包括 `__pycache__`）要在容器里 `chown` 回来。


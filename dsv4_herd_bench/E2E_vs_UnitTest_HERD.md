# DSV4-Flash HERD：e2e 与单测速度不一致的分析

对象：DeepSeek-V4-Flash，TP1，gfx1250（单卡），decode 阶段 64 并发（MoE 的 M=64），开启 HERD。
比较 FlyDSL+HERD（`ATOM_USE_TRITON_MOE=0 AITER_FLYDSL_USE_HERD=1`）和 Triton+HERD（`ATOM_USE_TRITON_MOE=1 ATOM_USE_TRITON_MOE_DECODE=1 AITER_TRITON_USE_HERD=1`）。

## 1. 结论

1. **单测和 e2e 不一致，最主要的原因是路由分布不同。**
   - 单测用的是均匀随机的 Top-7 候选，开 HERD 后每层激活约 166 个专家。
   - e2e 稳定 decode 阶段每层只激活约 105 个专家。
   - 头几个 decode step 的路由更集中，约 48 个专家，不能拿来代表稳定状态。
2. **shared expert 的 dual-stream 并发是第二个原因，单测里已经复现。**
   - FlyDSL 的路由比 Triton 短约 3 µs，GEMM1 因此提前开始，和 alt stream 上还没跑完的 shared expert 重叠，被拖慢约 3 µs。
   - Triton 的路由更长，shared expert 基本在 GEMM1 开始前就跑完了，不受影响。
   - 加上 shared expert 后，单测里每层的胜负发生反转：不加时 FlyDSL 快 3.85 µs，加了之后慢 2.39 µs。
3. **e2e 里 FlyDSL 的 GEMM 单独运行也更慢，这一点单测没有复现。**
   - e2e：GEMM1 慢 2.3–3.5 µs/层，GEMM2 慢 2.4–3.7 µs/层。
   - 单测在同样的路由分布下：GEMM1 快 0.9 µs，GEMM2 只慢 0.5 µs。
   - 原因还没找到。
4. **能解释多少：**
   - 单测复现出来的部分是 +2.39 µs/层 × 40 个 HERD 层 ≈ +96 µs/step，约占 e2e 实测差距（+320 µs/step）的 30%。
   - 其余部分来自第 3 点没复现的 GEMM 单独运行差距。关掉 dual-stream 后还出现了一部分非 kernel 的空闲时间（见 5.5）。

## 2. 环境与数据来源

| 项目 | 内容 |
|---|---|
| 容器 | `gcy_atom`，GPU 为 1 × gfx1250 |
| ATOM | `r1_1012_rebased`（`f19778df4`）+ `atom_flydsl_herd.patch` |
| aiter | `/home/cguo/aiter`，`cguo/gfx1250_herd`（`38ab715d4`） |
| e2e bench | `bench_dsv4.sh`：ISL=1024、OSL=1024、64 并发、256 条请求，缓存已热 |
| e2e trace | torch profiler，在 bs=64 的稳定 decode 阶段抓 0.5 s（约 45–50 个 step） |
| 单测 | 你的 `herd_triton_flydsl.py`，以及在它基础上扩展的 `herd_e2e_repro.py` |

GPU 是共用的：用户 sijieli2 从 06:18:58 UTC 开始用 `docker run --rm` 起临时容器跑 Qwen3.8 GDN 的 benchmark。本文所有测量都在这之前完成，时间上没有重叠。

## 3. 现象

### 3.1 e2e bench（64 并发）

| 配置 | 输出吞吐 (tok/s) | Mean TPOT (ms) |
|---|---|---|
| Triton+HERD | 6798.26 | 8.57 |
| FlyDSL+HERD（run2） | 6619.96 | 8.89 |
| FlyDSL+HERD（run3） | 6469.90 | 9.11 |
| FlyDSL（不开 HERD） | 6041.32 | 9.82 |
| Triton+HERD，关 dual-stream | 6333.13 | 9.17 |
| FlyDSL+HERD，关 dual-stream | 6068.33 | 9.80 |

吞吐上 FlyDSL+HERD 比 Triton+HERD 慢 2.7%（6620 对 6798），TPOT 每个 step 多约 320 µs。

### 3.2 单测（`herd_triton_flydsl.py`，M=64，5 个 seed，重复 3 次）

| | FlyDSL | Triton | FlyDSL − Triton |
|---|---|---|---|
| 开 HERD，graph E2E (µs) | 155.93 | 151.53 | +4.40 |
| 不开 HERD，graph E2E (µs) | 173.18 | 175.69 | −2.51 |

开 HERD 时 seed 0 的逐 kernel 时间（µs）：

| | GEMM1 | GEMM2 | 量化 | gather/reduce | 选专家 + sort |
|---|---|---|---|---|---|
| FlyDSL | 89.03 | 47.62 | 3.02 | 2.60 | 8.60 |
| Triton | 89.40 | 44.61 | 1.82 | 3.15 | 11.77 |

按单测推算，每层 MoE 只慢 3–4 µs，乘以 40 个 HERD 层约 110–180 µs/step，比 e2e 的 320 µs 少很多。而且单测里两边 GEMM1 几乎一样。

## 4. e2e 里到底慢在哪

### 4.1 每个 step 的 kernel 时间拆分（开 dual-stream）

每层 GEMM 时间取每次 launch 的平均值，每个 step 的差值按 43 层累加。trace 是在 classic graph path 下抓的，原因见 6.2。

| 类别 | Triton+HERD | FlyDSL+HERD | 差值 (µs/step) |
|---|---|---|---|
| 选专家 | — | — | −58.8 |
| sort / route | — | — | −73.3 |
| GEMM1 | 53.36 µs/层 | 60.30 µs/层 | +286.4 |
| GEMM2 | 27.96 µs/层 | 31.66 µs/层 | +152.7 |
| 非 MoE 部分 | — | — | +107.2（其中 shared expert down GEMM 占 +90.3） |
| 合计 | 9958.0 | 10371.5 | +413.5 |

### 4.2 按"是否和别的 stream 重叠"拆分 GEMM1

| | 单独运行（中位数） | 被重叠时（中位数） | 被重叠的 launch 比例 |
|---|---|---|---|
| Triton GEMM1 | 53.04 µs | 54.38 µs | 24%，且每次只重叠了 0.36% 的时长 |
| FlyDSL GEMM1 | 55.32 µs | 61.97 µs | 74%，平均重叠 8.2% 的时长 |
| Triton GEMM2 | 27.96 µs | — | 0% |
| FlyDSL GEMM2 | 31.61 µs | — | 0% |

shared expert 的 down GEMM（`gemm_a8w8_mx128_t32x32x512_mw2_nw2_nb3`）：在 Triton 模式下平均 5.65 µs，在 FlyDSL 模式下平均 7.75 µs（被重叠时 8.65 µs，单独运行 5.31 µs）。

### 4.3 时间线（截取其中一层，以 GEMM1 开始时刻为 0）

- **alt stream 上的 shared expert：** 量化 → gate_up GEMM → `act_and_mul` → 量化 → down GEMM，合计约 22 µs。
- **Triton 主 stream：** gate GEMM → Fill → `_topk` → `_keepk_sort0` → `_combined_routing` → `_downcast_to_mxfp`，从 −19.3 µs 跑到 −0.8 µs。shared expert 的 down GEMM 在 −0.2 µs 结束，基本不和 GEMM1 重叠。
- **FlyDSL 主 stream：** gate GEMM → `topk_gating` → `moe_route_psum_herd_k6` → `moe_token_multidest_quant`，从 −15.3 µs 跑到 −1.2 µs。shared expert 的 down GEMM 从 −3.0 µs 跑到 +2.8 µs，和 GEMM1 开头重叠约 3 µs，GEMM1 被拉长到 61.7 µs。

### 4.4 关掉 dual-stream 后

| | Triton+HERD | FlyDSL+HERD | 差值 |
|---|---|---|---|
| GEMM1 单独运行（中位数） | 52.20 µs | 55.72 µs | +3.52（+6.7%） |
| GEMM2 单独运行（中位数） | 27.48 µs | 29.84 µs | +2.36 |
| 每个 step 的 kernel 时间 | 9805.7 µs | 10100.2 µs | +294.4（其中选专家 + sort −132，非 MoE +2.6） |
| 不 profile 时的 TPOT | 9.17 ms | 9.80 ms | +630 µs |

- 不重叠时 FlyDSL 的 GEMM 仍然更慢，说明这部分差距是 kernel 自身的。
- TPOT 的差距（+630 µs）比 kernel 时间的差距（+294 µs）大出约 340 µs，这部分不在 kernel 里，见 5.5。

## 5. 原因逐条分析

### 5.1 路由分布（主要原因）

从 ATOM 的 decode（bs=64，eager 模式）导出每层的 router logits，每隔 64 个 decode step 存一次（`router_dump/`）：

| decode step | 每层激活专家数（Top-6，中位数） | 开 HERD 后（中位数，范围） |
|---|---|---|
| 0 | 62 | 48.5（26–74） |
| 64 | 129 | 105（84–128） |
| 192 | 137.5 | 110.5（84–125） |
| 384 | 128.5 | 103（81–118） |
| 704 | 122.5 | 97.5（76–119） |

- 单测的均匀随机候选大约对应 166 个专家，比 e2e 多约 60%，所以单测的 GEMM1 是约 89 µs，e2e 只有约 53–55 µs。
- 在不同的专家数下，两个实现谁快谁慢不一样：
  - 均匀分布（约 165 个专家）时，FlyDSL 的 GEMM2 明显更慢（+3–5 µs）；
  - 真实稳定分布（约 105 个专家）时，两边的 GEMM 基本持平；
  - 头几个 step（约 45 个专家）时，FlyDSL 的 GEMM1 更快。
- 最早只导出了头 8 个 decode step，复现出来的 GEMM1 只有约 30 µs，和 e2e 对不上。改成用稳定阶段的数据后，才得到约 58 µs，和 e2e 一致。

### 5.2 冷权重（次要原因）

- 单测每次 replay 都用同一份权重，有一部分会命中 cache；e2e 每个 step 依次走 43 层，每层的权重都是冷的。
- 复现脚本改成 8 层或 40 层轮转，每层一份独立的权重。真实路由（头 8 步）下，GEMM1 从单层的 29.56 / 30.78 µs（FlyDSL / Triton）变成 8 层的 32.37 / 34.34 µs，大约慢了 10%。
- 均匀路由下，单层时 Triton 的 GEMM1 占了便宜（87.71 对 FlyDSL 92.88 µs），8 层后两边持平（90.63 对 91.28 µs）。

### 5.3 单测里的 Triton 路径和 ATOM 不一样（影响小）

| | 单测（`Triton-unit`） | ATOM（`Triton-ATOM`） |
|---|---|---|
| 激活量化 | `_static_per_tensor_quant_fp8`（scale 预先算好） | `_downcast_to_mxfp`（MXFP8 动态） |
| GEMM1 | `..._HAS_MX_OUT_0` | `..._HAS_MX_OUT_1`（融合 MX requant） |
| 权重 | plain 布局 | FlyDSL 权重上的零拷贝视图（`moe_weight_decode_view`） |

复现脚本里两者每层的 step 时间相差约 1 µs（例如均匀路由、8 层：159.29 对 160.73 µs）。FlyDSL 一侧在单测和 e2e 里是同一套 kernel。

### 5.4 shared expert 的 dual-stream 并发（已在单测复现）

`herd_e2e_repro.py --shared on` 复刻了 ATOM 的 `dual_stream_moe_forward`：
- 主 stream：router gate GEMM（`tgemm.mm`）→ 路由 → MoE；
- alt stream：shared expert，调用 `per_group_quant_hip` → `gemm_a8w8_blockscale_bpreshuffle` → `silu_and_mul(limit=10)` → 量化 → GEMM；
- 两边最后在主 stream 上汇合相加，全部在一个 CUDA graph 里。

40 层、稳定阶段真实路由（激活专家中位数 105），每层时间：

| shared expert | 实现 | GEMM1 | GEMM2 | 选专家 + sort | shared expert 的 kernel | 每层 step 时间（默认 scheduling） |
|---|---|---|---|---|---|---|
| 无 | FlyDSL | 57.88 | 29.01 | 7.98 | — | 108.88 |
| 无 | Triton-ATOM | 58.75 | 28.47 | 11.20 | — | 112.74 |
| 有 | FlyDSL | **60.94** | 29.42 | 8.67 | 21.38 | 117.18 |
| 有 | Triton-ATOM | 57.79 | 30.12 | 11.54 | 18.41 | 114.79 |

- FlyDSL 的 GEMM1 在加上 shared expert 后多了 3.06 µs，Triton 的没有变慢。
- shared expert 自己的 kernel 在 FlyDSL 旁边也多了约 3 µs。
- 每层 step 时间从"FlyDSL 快 3.85 µs"变成"FlyDSL 慢 2.39 µs"。

### 5.5 尚未解释的部分

1. **GEMM 单独运行的差距：** e2e 里 FlyDSL GEMM1 慢 2.3–3.5 µs、GEMM2 慢 2.4–3.7 µs；同样路由分布的单测里，GEMM1 快 0.9 µs、GEMM2 只慢 0.5 µs。还没排除的因素：
   - 真实权重和激活的数值（随机数据会影响功耗和频率）；
   - 持续满载时的频率；
   - 每层 MoE 前后的 attention 等 kernel 留下的 cache 和 TLB 状态；
   - 服务环境变量里 `HSA_XNACK=1`、`HSA_USE_SVM=1` 的影响（复现时已经用了相同的环境变量，但没有单独对比过有无的差别）。
2. **关掉 dual-stream 后多出来的非 kernel 时间：** TPOT 差 +630 µs，kernel 时间只差 +294 µs。classic path 会改变 launch 间隔，在它下面量出的墙钟和空闲时间噪声很大（同一模式下每个 step 的空闲从 1.0 ms 到 2.8 ms 不等），不能用来定位这部分。
3. **e2e 的样本量：** 每种 e2e 配置大多只跑了一次。两次 FlyDSL+HERD 之间差约 2.3%，3% 量级的差距还需要重复测量确认。

## 6. 方法上的注意事项

1. **首次运行有 JIT 编译。** 第一轮 bench 期间还在编译 aiter 的 `module_top_k_per_row` 和 FlyDSL kernel，TTFT 和 P99 ITL 都被拉高，那一轮的数据作废。
2. **profiler 在多 stream 的 graph 里会记错 kernel 时长。** 在默认的 segment scheduling 下，多 stream graph 的每个 kernel 时长会被摊平或记到别的 kernel 上（例如 GEMM1 显示 5 µs、reduce 显示 37 µs）。逐 kernel 时间必须加 `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0 DEBUG_HIP_GRAPH_CLASSIC_PATH=1`。classic path 会改变 launch 间隔，所以它下面的墙钟和空闲时间不可信，step 时间要用默认 scheduling 测。单 stream 的单测在默认 scheduling 下的 kernel 时间是可靠的。
3. **默认 scheduling 下 profile 多 stream graph 会崩溃**（`malloc(): unaligned tcache chunk detected`）。默认模式下只测 step 时间（`--no-profile`）。
4. **导出路由的时机。** 头几个 decode step 的路由明显比稳定阶段集中，要用第 64 步之后的数据。
5. **GPU 是共用的。** 测之前先确认没有别人在跑，可以看 `/sys/class/kfd/kfd/proc` 和 `docker events`。

## 7. 下一步

1. 在 GPU 空闲时，把每种 e2e 配置各重复 3 次，确认 2.7% 的差距和关 dual-stream 后的 +630 µs。
2. 并发争抢：在单测里调整 shared expert 的启动时机，例如让 down GEMM 等 GEMM1 开始后再跑、只和 GEMM2 重叠；或者在 FlyDSL 路由结束和 GEMM1 之间加一个依赖。看 FlyDSL 能否拿回这约 6 µs/层。
3. GEMM 单独运行的差距：单测改用一层的真实权重和激活；在 e2e 和单测里同时记录频率；在层与层之间插入类似 attention 的 kernel，模拟 cache 被冲刷的情况。
4. 非 kernel 空闲：在不使用 classic path 的前提下，用 rocprofv3 测真实 graph 执行时 kernel 之间的间隙。

## 8. 文件与复现方法

| 文件 | 用途 |
|---|---|
| `herd_e2e_repro.py` | 扩展的单测：真实或均匀路由、多层冷权重、shared expert 的 dual-stream、ATOM 的 Triton 路径。直接 import `herd_triton_flydsl.py`，没有改原文件 |
| `/data/cguo/router_dump/` | 从 ATOM decode 导出的 router logits，每隔 64 步一次（不在仓库里） |
| `dump_router.sh`、`atom_router_dump.patch` | 导出路由用的 eager 服务和临时 hook。运行结束后会把 `moe.py` 恢复成运行前的样子 |
| `profile_mode.sh`、`analyze_trace.py`、`gemm_overlap.py`、`timeline_trace.py`、`idle_trace.py` | e2e 的 trace 采集与分析 |
| `/home/cguo/dsv4_herd_bench/profiles/`、`/home/cguo/dsv4_herd_bench/logs/` | 本文引用的 trace 和所有日志（不在仓库里）。新跑的结果默认写到本目录下的 `profiles/`、`logs/` |
| `server_env.sh` | 和服务相同的环境变量 |

复现命令（在容器内）：

```bash
cd /home/cguo/aiter && source dsv4_herd_bench/server_env.sh
# 逐 kernel 时间
DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0 DEBUG_HIP_GRAPH_CLASSIC_PATH=1 \
  python dsv4_herd_bench/herd_e2e_repro.py --routing real --layers 40 --rounds 3 --stacks FlyDSL Triton-ATOM
# step 时间（默认 scheduling）
python dsv4_herd_bench/herd_e2e_repro.py --routing real --layers 40 --rounds 3 --stacks FlyDSL Triton-ATOM --no-profile
# e2e trace（写到 dsv4_herd_bench/profiles/）
DECODE_WAIT_S=2 PROFILE_S=0.5 bash dsv4_herd_bench/profile_mode.sh triton_herd
DECODE_WAIT_S=2 PROFILE_S=0.5 bash dsv4_herd_bench/profile_mode.sh flydsl_herd
python3 dsv4_herd_bench/analyze_trace.py dsv4_herd_bench/profiles/{triton_herd,flydsl_herd}/trace/rank_0/*.json.gz
```

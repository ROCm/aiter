# Comm-fused MoE：M=512～32768 性能提升计划

> 计划基线：AITer `26485f954`，gfx950/MI355X，单机 TP8。
> 本文只定义性能优化路线和晋级门槛；临时 probe、trace 和失败候选不进入生产路径。

## 1. 目标和范围

本计划只优化已经完成精度和生产接入的 DeepSeek-V4 TP8 Stage2 通算融合路径：

```text
H=7168
I_per_TP=384
E=384
TOPK=6
TP=8
activation=FP8
weight=FP4
output=BF16
```

目标不是让所有 M 强制使用同一种 kernel，而是允许每个 M 选择自己的最快正确实现：

```text
full-width
windowed
persistent-window
后续经过验证的新 family
```

最终目标：M=512～32768 的每个 production bucket，在同轮、同输入、同测量方式下，相对
该档优化开始时冻结的**当前融合实现**至少再快 5%，同时不降低精度、稳定性和整网性能。
ordinary `Stage2 + shared partial + TP all-reduce` 只作为精度参考和最终总收益对照，不是
5% 优化目标的分母。

M=256 已单独完成 atomic 路径验证并接入 production；本轮“相对当前融合实现再优化 5%”的
范围仍是 M=512～32768。decode 小 bucket 不在本轮范围内。

### 1.1 一页执行摘要

| 顺序 | M | 冻结的当前融合基线 | 5% 目标上限 | 首选结构优化 | 备选方向 |
|---:|---:|---:|---:|---|---|
| 1 | 512 | 同轮默认 Opus atomic + RS128/AG98 | 同轮 baseline × 0.95，以正式同轮实测为准 | 固定 Opus，优化 MXFP8 后处理和通信同步 | 因果 producer/service overlap |
| 2 | 1024 | 181.13～183.76 us | 172.07～174.57 us | 复用并重调 segmented overlap | full-width 两阶段路径 |
| 3 | 2048 | 243.19～249.34 us | 231.03～236.87 us | full-width self-publish | persistent collective service |
| 4 | 4096 | 371.84～384.53 us | 353.25～365.30 us | 优化 full 对比正确 W1792/W3584 | local/collective 小范围调参 |
| 5 | 8192 | 613.00～625.49 us | 582.35～594.22 us | window 内部发布 completion，删除 barrier launch | 正确宽窗口 |
| 6 | 16384 | 1042.12～1074.36 us | 990.01～1020.64 us | 优化 window 对比独立调优 persistent | service grid / gate/drain |
| 7 | 32768 | 1966.58～1976.25 us | 1868.25～1877.44 us | persistent RS/AG 跨 phase 流水 | producer persistent / collective 预取 |

整体策略不是一次开发一个“大一统 kernel”，而是按 M 从小到大逐档建立瓶颈证据、验证
一个结构性候选、保留一个正确 winner。公共改动必须回归全部已接入 M。

## 2. 当前基线和 5% 目标

### 2.1 当前融合 baseline 是唯一晋级口径

本文记录的微秒数只是当前 HEAD、当前节点和当前测量口径下的性能快照，不是永久不变的
目标。正式判断始终使用：

```text
candidate_time / same_round_fused_baseline_time < 0.95
```

其中 `fused_baseline` 是该 M 开始优化时冻结的当前融合算法。正式 A/B 必须在同一个进程、
同一组输入中交替运行这条冻结路径和累计候选路径。阶段内获得的有效小优化可以累加，但不会
把每个新 winner 重新设成分母；否则 5% 目标会不断移动。

只要改动 baseline 与 candidate 共用的 GEMM、reduce、Opus 或其他公共 primitive，就必须让
两条路径使用同一版公共 primitive 并重新做同场 A/B。共同获得的加速不能算成候选相对当前
融合实现的收益。ordinary 仍需同轮记录，但只用于：

- 验证最终输出精度；
- 观察融合实现相对普通路径的总收益；
- 防止通过拖慢 ordinary 人为放大比例。

每个 M 只在开始该档优化时冻结一次算法 baseline；公共 primitive 更新时重建同算法 baseline，
但不改变算法差异。必须测 uniform/skew、使用相同 Graph replay 口径，并记录 commit、节点和
完整参数。

下面是 2026-08-21 在同一 MI355X/gfx950 节点测得的动态对照快照：

```text
AITer commit: 26485f954878a16e906a8adcc046557d52dca4b5
TP: 8
测量: 5 rounds × 50 CUDA Graph replay
统计: 每轮 8 rank 最大值的 median
输入: uniform / skew，seed=20260819
```

它替代 `docs/comm_fused_moe_design.md` 中的历史数字，作为本轮开始时的性能快照。

| M | family | uniform ordinary / fused / ratio | skew ordinary / fused / ratio |
|---:|---|---:|---:|
| 512 | atomic | 158.688 / 143.328 / 0.9032 | 162.392 / 150.524 / 0.9269 |
| 1024 | atomic | 209.915 / 177.882 / 0.8474 | 209.353 / 182.190 / 0.8703 |
| 2048 | full | 294.768 / 244.585 / 0.8298 | 300.930 / 251.355 / 0.8353 |
| 4096 | full | 463.890 / 375.321 / 0.8091 | 473.398 / 387.001 / 0.8175 |
| 8192 | window | 908.232 / 619.676 / 0.6823 | 926.840 / 626.760 / 0.6762 |
| 16384 | window | 1800.168 / 1037.718 / 0.5765 | 1832.655 / 1078.473 / 0.5885 |
| 32768 | persistent | 3585.732 / 1958.824 / 0.5463 | 3589.532 / 1967.662 / 0.5482 |

上表中的 ordinary 数据是附加参考，不是晋级分母。正式晋级要求每个 M、每种 route 的
`candidate / same-round frozen fused baseline` 都严格小于 0.95，并尽量保留 0.5%～1% 的余量。
不能用 uniform 的收益掩盖 skew 回退，也不能用某轮偶然最小值代替交替 A/B 的 rank-max
median。

本计划冻结基线时的 production 参数为：

| M | family | compute | local/collective 参数 |
|---:|---|---|---|
| 256 | atomic | Opus atomic Stage2 + MXFP8 quant | RS92, AG91 |
| 512 | atomic | Opus atomic Stage2 + MXFP8 quant | RS128, AG98 |
| 1024 | atomic | Opus atomic Stage2 + MXFP8 quant | RS128, AG126 |
| 2048 | full | TM64/TN256/TK128, SBM64 | RS128, AG126 |
| 4096 | full | TM64/TN256/TK128, SBM64 | RS128, AG126 |
| 8192 | window | TM64/TN256/TK128, SBM64, W1024 | LW2048, RS92, AG91 |
| 16384 | window | TM64/TN256/TK128, SBM64, W1024 | LW2048, RS92, AG91 |
| 32768 | persistent | TM64/TN256/TK128, SBM64, W1024 | LW2048, SG77 |

## 3. 为什么不能只继续扫参数

当前 tile、worker 和 collective grid 已经经过较大范围 sweep。历史结果说明：

- `TN=128` 或 `TK=256` 的部分表观快结果存在漏算，不能作为优化候选；
- 增大 local worker 或 RS grid 可以让单个 primitive 更快，但会与 GEMM 争抢 CU，使完整
  pipeline 变慢；
- MORI SDMA 和 CU push 没有超过当前 compressed direct-pull 完整 pipeline；
- W3584 的旧结果中 local reduce 只覆盖 2048 列，属于错误执行，不能用于性能判断；
- 2-slot/3-slot persistent ring 曾出现 2.6～3.3 ms 和驻留死锁风险，不能直接恢复；
- host Python 代码缩行、对象封装和普通参数整理不会让 Graph GPU 时间下降 5%。

因此新的 5% 目标仍应优先来自结构性收益：

1. 减少 device kernel/barrier launch；
2. 缩短完整流水关键路径，而不是只缩短某个 primitive；
3. 减少 local reduce 的无效 CTA、尾列浪费和中间数据落地；
4. 让 RS、AG 和不同窗口的 GEMM 真正重叠；
5. 根据 M 选择通信算法，而不是所有 M 固定 RS+AG。

## 4. 当前流水的主要固定成本

### 4.1 Full-width：M=512～4096

当前完整顺序为：

```text
GEMM2
→ local route/shared reduce
→ partial epoch barrier
→ TP reduce-scatter + owner BF16/MXFP8 publication
→ reduced epoch barrier
→ TP all-gather + BF16 decode
```

共 6 次 device launch。CUDA Graph 消除了大部分 Python 调度成本，但不会消除 GPU command
processor 的 launch 前端成本、kernel 尾部、两次全 TP barrier 和 kernel 间资源重新调度。

M 越小，GEMM 和通信有效工作越少，这些固定成本占比越高。M=512 想减少约 7.4～7.8 us，单靠
微调 tile 或 grid 很难稳定实现。

Full-width local reduce 当前每个线程处理 8 个 hidden 元素：

```text
BLOCK=256
VECTOR_WIDTH=8
每 CTA 覆盖 2048 列
H=7168 → 每 token 需要 4 CTA
```

第四个 CTA 只有 1024 列有效，约一半线程被屏蔽。M=512/1024 时，这部分 CTA 和 launch
尾部尤其值得优化。

### 4.2 Windowed：M=8192/16384

W1024 将 H=7168 切成 7 phase。当前 host 流水包含：

```text
1 × G0
6 × cycle
3 × drain
7 × partial barrier
7 × reduced barrier
```

总计约 24 次 device launch。cycle 已经把不同窗口的 G/L/RS/AG 放入同一 kernel，但
completion 仍依赖独立 host barrier kernel，因此同步 launch 数量很大。

这说明 windowed 的第一优化目标不应是再加窗口，而应是让 cycle/drain 自己发布完成状态，
删除独立 barrier launch。

### 4.3 Persistent-window：M=32768

当前 persistent 路径约 10 次 device launch：

```text
1 × G0
6 × phase producer
1 × drain
1 × final publish
1 × persistent collective service
```

service 内每个 phase 仍严格执行：

```text
wait partial[p]
→ RS[p]
→ wait reduced[p]
→ AG[p]
→ phase p+1
```

它消除了每个 phase 的 host collective/barrier launch，但没有做到 `RS[p+1] + AG[p]`，
producer 也不是单 kernel persistent。M=32768 想再减少约 195 us，必须缩短这条串行
critical path。

## 5. 总体实施原则

### 5.1 从小 M 向大 M 逐档推进

执行顺序固定为：

```text
M=512
→ M=1024
→ M=2048
→ M=4096
→ M=8192
→ M=16384
→ M=32768
```

每完成一个 M：

1. 只保留该 M 的最快正确 winner；
2. 删除生产代码中的 loser 和实验开关；
3. 回归已经完成的所有更小 M；
4. 公共 collectives 有修改时，回归全部 M；
5. winner 稳定后才进入下一个 M。

不同 M 可以使用不同 family。没有必要为了统一代码让某个 M 使用慢 3%～10% 的实现。

### 5.2 先定位关键路径，再写结构性 kernel

每个 M 先生成一份 trace，至少记录：

- 每个 kernel 的 GPU duration；
- kernel 间 gap 和实际 overlap；
- local reduce、RS、AG 的独立时间；
- producer/service 是否同时驻留；
- CU occupancy、VGPR/LDS 限制；
- HBM 和 xGMI 带宽；
- 最后一个 CTA/phase 的尾部长度。

优化只针对完整 graph critical path。单 primitive 快 5%，完整 graph 没变或回退，则立即
淘汰。

### 5.3 生产路径只保存 winner

- 候选枚举、trace、临时统计全部放仓外或 tuner；
- production kernel 不增加环境变量和实验布尔开关；
- 新算法只有成为至少一个 M 的 winner 才保留；
- CSV 只记录最终参数；
- host 只按 family/config dispatch，不在线搜索。

## 6. 第一阶段：M=512

### 6.0 当前关键路径和已淘汰候选

同场 component probe 显示当前完整路径约为 149.90 us：

| 阶段 | 耗时 |
|---|---:|
| GEMM2 producer | 110.01 us |
| local reduce | 6.36 us |
| partial epoch barrier | 6.48 us |
| TP reduce-scatter | 13.80 us |
| reduced epoch barrier | 6.35 us |
| TP all-gather | 12.96 us |

GEMM2 占约 73%，因此只优化 local reduce 很难达到整段 5%。当前已经验证并淘汰：

- 历史 124.27～124.88 us 的 GLRF 结果使用预先准备并发布的跨卡 payload，只是
  resource co-residency roof，不是 causal production pipeline；
- 把 local reduce、两次 barrier、RS 和 AG 全部压进 128 CTA persistent post kernel 后为
  177.66 us。它把原来的 2048 个 local CTA 压成 128 个常驻 worker，软件全局同步和阶段
  串行反而拉长关键路径；
- 修复 TK256 的 K tail 后精度恢复到 `max_abs=0.75, rel_l2=0.031317`，但完整 Graph 为
  152.05 us，慢于当前 TK128 winner。历史约 115.99 us 的 TK256 数据少算 128 个 K，
  不可信；
- 将 epoch barrier 吸收到 RS/AG kernel 后完整 Graph 为 164.00 us。多 CTA 的 atomic/gate
  成本和过早驻留抵消了两次 launch，已淘汰；
- 一阶段 compressed TP full-output reduce 精度正确且略好，但完整 Graph 为 211.67 us。
  每个 rank 读取全部 TP partial，跨卡流量相对 RS+AG 放大约 4 倍，已淘汰；
- 修复覆盖范围后的 W3584/TK128 两窗口单流为 173.85 us，双 stream 因果 overlap 为
  171.06 us。窗口化增加的 launch、同步和 CU 竞争远大于获得的 overlap，已淘汰；
- local reduce VECTOR16 在 gfx950 上必须拆成两次 V8 BF16 load，完整 Graph 为 152.74 us，
  慢于 VECTOR8 winner，已淘汰。
- RS 直接向全部 rank 推送 BF16、从而删除 requant 和 AG 的候选，uniform/skew 分别为
  171.98/168.00 us；额外 BF16 xGMI 写流量超过省下的阶段成本，已淘汰；
- 让 RS CTA 自发布、AG CTA 自行等待 peer epoch 的候选，uniform/skew 分别为
  156.94/155.81 us；多 CTA atomic/poll 成本超过独立 reduced barrier 的 6.35 us，已淘汰；
- 仅把 GEMM 改为两段 N-major 调度、不启动通信 overlap 时为 155.42 us；N-major 调度本身
  损失约 5～6 us 的 expert/N 局部性；
- 单次 phase-major GEMM 内由每个 CTA atomic 计数并发布两阶段 ready 的因果 overlap 候选，
  精度正确，但 uniform/skew 分别为 736.58/742.16 us。数千个 CTA 的全局 atomic 完成计数
  严重串行化，后续不再采用 per-CTA completion counter。
- 普通 Opus atomic Stage2 接 BF16+shared→MXFP8 quantize，再复用 compressed RS+AG，
  `max_abs=0.75, rel_l2≈0.03002`。同轮 ordinary baseline 为 149.85 us，候选为
  141.91 us，即相对 ordinary 为 `0.9471`；这个数字只说明总路径收益，不能用于本轮 5%
  晋级。它必须与冻结的 full-width 融合 baseline 同场比较。
  当前分解为 compute+zero 107.01 us、quantize 6.42 us、partial barrier 6.73 us、RS
  14.01 us、reduced barrier 6.67 us、AG 13.23 us；它是下一轮组合优化的起点，不进入
  production。
- 在上述 atomic 路径后，把 quantize、两次软件全局/跨卡同步、RS 和 AG 压进 128 CTA
  persistent post kernel，修复精度后完整 Graph 为 199.57 us，显著慢于同轮 149.85 us
  ordinary baseline。少 launch 无法弥补 quantize 并行度下降和常驻 CTA 内串行同步，已从
  仓外 probe 删除。
- 把 Opus Stage2 改成固定 16/24/32/48 个常驻 route CTA，最好结果仍为原 producer 的
  `1.0710x`；按连续 route block 尝试复用同一 expert W2 后最好也只有 `1.1838x`。producer
  本身已经回退，无法依靠后续通信 overlap 补回，已淘汰并清除全部 Opus/csrc 实验改动。
- 把全 peer 轮询 barrier 改成 rank0 集中计数并向各 rank 发布 release，精度通过，但相对
  当前 atomic 路径 uniform/skew 分别为 `1.0108x/1.0068x`。集中 atomic 和 release push
  没有降低同步临界路径，已淘汰。
- 删除 Opus atomic epilogue 的 LDS 中转，改用 `ds_bpermute` 从相邻 lane 直接拼 BF16x2
  atomic，精度通过，但相对同轮 baseline 的 uniform/skew 分别为 `1.0728x/1.0902x`。
- 将上述 lane 交换改为 DPP `quad_perm 0xb1` 后仍无改善，uniform/skew 分别为
  `1.0736x/1.0876x`。说明 LDS 在这里不只是冗余中转；它提供了更适合连续 BF16x2 atomic
  的输出重排。两版寄存器直出实现均已从生产工作树删除。

这些结果说明，M=512 不能靠减少一个 barrier launch、改一个向量宽度或把 full-width 简单切
窗口达到 5%。当前累计候选已经采用普通 Opus atomic Stage2 输出接 compressed RS+AG：用
atomic Stage2 直接把 routed expert 结果累加进 shared partial，再用 V16 quantize 生成通信
partial，并删除 FP8 route buffer 和六路 route local reduce。冻结参数为 RS128/AG98；它仍须
通过严格 0.95 门禁。后续候选必须是因果完整流水，不能使用预先
准备的跨卡 payload。

### 6.1 已淘汰：一阶段 TP full-output reduce

小 M 优先验证不走 RS+AG 的一阶段算法：

```text
GEMM2 route output
→ local route/shared reduce + publish compressed partial
→ 每个 rank 直接 pull TP8 partial
→ FP32 reduce
→ 写完整 BF16 output
```

它会增加跨卡读取量，但可以删除：

- owner shard MXFP8 requant；
- reduced-ready barrier；
- 独立 all-gather；
- owner/reduced payload workspace。

M=512 数据量较小，固定 launch/barrier 成本可能高于额外通信量。一阶段候选必须使用当前
MXFP8 partial 和 FP32 accumulate，不能退回普通 BF16 all-reduce。

第一版先保持 GEMM2 独立，只将 postprocess 收缩为：

```text
local reduce
→ one-stage TP reduce
```

如果仍达不到目标，再尝试一个 persistent postprocess kernel：local workers 写 partial，
collective workers 在同一 kernel 内等待 TP epoch 并生成完整输出。

实测完整 Graph 为 211.67 us，显著慢于 150.08 us baseline，不再继续。

### 6.2 已淘汰：local VECTOR16

测试以下正确候选：

```text
BLOCK256 × VECTOR8   当前，4 CTA/token
BLOCK256 × VECTOR16  2 CTA/token
BLOCK128 × VECTOR16  4 CTA/token，但线程更少
3 个 full CTA + 1 个轻量 tail CTA
```

重点观察：

- VECTOR16 是否因 FP32 accumulator 增加而降低 occupancy；
- CTA 数减半是否缩短 local kernel 尾部；
- tail 专用映射是否比统一 kernel 多一次 launch 更差；
- route、scale、shared BF16 load 是否仍连续合并。

正确实现保持累加和量化协议，但在 gfx950 上 V16 BF16 buffer load 不能直接生成，需要拆成
两个 V8 load；最终完整 Graph 为 152.74 us，未成为新 baseline。

### 6.3 下一候选：因果 segmented producer/service overlap

目标结构：

```text
GEMM producer 完成 cohort 0 → publish ready[0]
GEMM producer 继续 cohort 1   || service 处理 cohort 0
GEMM producer 继续 cohort 2   || service 处理 cohort 1
...
尾部 drain
```

cohort 必须与 GEMM 实际写回粒度一致，ready 只能由负责该 cohort 最后一份输出的 CTA 发布。
service 只读取已发布 cohort，并保持当前 MXFP8 partial、FP32 accumulate、压缩 RS+AG 语义。
第一版只允许一个小 service grid，避免复现 128 CTA persistent post 对 producer 的 CU 抢占。

实现前必须审计历史 segmented/finalize probe：

- `atomic_publish_only` 约 121.29 us，只是无通信 roof；
- 完整 `flydsl_atomic_ar` 约 150.89 us，与当前 production 接近；
- `atomic_ready_registered_ar` 约 171.64 us；
- 历史 wave cohort overlap 约 204～226 us。

新候选必须明确消除旧版本的过细 cohort、每 cohort 全局 barrier、过多 service CTA、资源争抢
或非因果 prepared payload；如果结构相同则不重复实现。

### 6.4 M=512 晋级门槛

唯一性能门槛是同一轮、同一输入分布、同一 Graph replay 口径下：

```text
candidate_time / frozen_atomic_baseline_time < 0.95
max_abs <= 1.0
rel_l2 <= 0.05
```

M=512 的冻结 baseline 是算法语义固定、并在每次正式 A/B 中同轮重测的 atomic 融合配置：

```text
Opus atomic Stage2 + V16 MXFP8 quantize + compressed RS/AG
RS128 / AG98
```

早期记录的 `148.28～150.08 us` 是公共 primitive 优化前的历史快照，不能再作为当前候选的
分母。2026-08-22 的同轮测量中，这条冻结算法通常约为 `133.2～135.0 us`；实际门槛始终按
当轮 baseline 乘以 `0.95` 计算，而不是固定微秒数。

uniform 和 skew 必须分别满足，且正式门禁至少重复两轮；任一有效轮次
`candidate / frozen baseline >= 0.95` 即不算稳定通过。
任何 GEMM、reduce 或 Opus 公共 primitive 修改都必须同时作用到 baseline/candidate 后重新 A/B；
ordinary 同轮记录但不参与 5% 判定。M=512 未通过这个冻结门禁前不进入 M=1024。

## 7. 第二阶段：M=1024

M=1024 复用 M=512 的候选，但重新寻找算法分界点，不能直接继承 winner。

2026-08-27 已完成这一阶段并将 winner 固化到 production CSV：

```text
family          persistent
tile            32 x 256 x 128
sort_block_m    32
window          1792
local_workers   512
service_grid    126
service order   first
```

正式同场 A/B 使用 5 轮、每轮 50 次 CUDA Graph replay，并取八个 rank 中最慢
rank 的跨轮中位数：

| route | atomic baseline | persistent winner | speedup |
| --- | ---: | ---: | ---: |
| uniform | 188.68 us | 164.08 us | 1.1499x |
| skew | 188.49 us | 167.29 us | 1.1268x |

正确性为 `max_abs=0.75`、`rel_l2≈0.0313`。下面保留当时的实验顺序和门禁，作为
该 winner 的设计依据。

实验顺序：

1. 当前 full RS+AG 基线；
2. M=512 一阶段 winner；
3. 一阶段 + local vector winner；
4. 两阶段 RS+AG + local vector winner；
5. 若固定同步仍占主导，测试 full-width self-publish。

Full-width self-publish 的含义是：

- local kernel 最后完成的 CTA 发布 partial epoch；
- RS kernel 自己等待 partial epoch，不再 launch partial barrier；
- RS 最后完成的 CTA 发布 reduced epoch；
- AG kernel 自己等待 reduced epoch，不再 launch reduced barrier。

完整 launch 数从 6 降到 4：

```text
GEMM2 → local → RS → AG
```

等待发生在真正消费数据的 kernel 内，不增加 host fallback。若 RS/AG CTA 因过早驻留抢占
GEMM/local CU，需通过同 stream 顺序或小 service grid 避免。

目标：

```text
uniform < 172.07 us
skew    < 174.57 us
```

预计收益：local 几何 2%～5%，删除 barrier 4%～8%，一阶段算法 5%～12%。

## 8. 第三阶段：M=2048

M=2048 当前 full-width 已经较高效，一阶段 TP full-output reduce 的额外流量可能开始超过
固定成本收益。因此优先级调整为：

1. self-publish full-width，删除两次 barrier launch；
2. local VECTOR16/tail mapping；
3. 重新扫 RS/AG grid，但只围绕当前 128/126 的小邻域；
4. 最后才测试一阶段算法。

如果 self-publish 仍不足 5%，增加一个单 persistent collective service：

```text
GEMM2
→ local reduce + partial publication
→ persistent service: wait partial → RS → rank-local gate → AG
```

目标是把两个 collective 和同步合并为一次 service launch，而不是让 service 与 GEMM
长期竞争 CU。M=2048 的 service grid 必须独立调优，不能复用 M=32768 的 77。

目标：

```text
uniform < 231.03 us
skew    < 236.87 us
```

预计收益：删除 barrier 3%～6%，local mapping 2%～5%，合并 collective service 3%～8%。

## 9. 第四阶段：M=4096

M=4096 是 full 与 window 的边界。这里同时测试两条路线。

### 9.1 路线 A：优化后的 full-width

复用 M=2048 的：

- self-publish；
- local reduce winner；
- collective grid winner。

### 9.2 路线 B：正确的宽窗口

旧 W3584 数据存在 local reduce 漏算，必须从正确实现重新开始：

```text
W3584 → 2 phase，每 phase 14 个 N tile
W1792 → 4 phase，每 phase 7 个 N tile
W1024 → 7 phase，每 phase 4 个 N tile
```

宽窗口 local reduce 必须显式覆盖整个 window：

- W3584 使用两个 column tile 或等价向量映射；
- W1792 可以由一个 2048-column worker 覆盖；
- scale group 数必须使用 `window / 32`；
- route row bytes、partial stride、payload stride 全部从 window 推导。

W3584 的价值是大幅减少 phase/launch；W1792 在 launch 数和 overlap 之间更平衡。两者都
必须用 TK128 完整计算 I=384。

目标：

```text
uniform < 353.25 us
skew    < 365.30 us
```

预计收益：优化 full 5%～10%；正确的 2/4-phase window 6%～14%。最终只保留更快 family。

## 10. 第五阶段：M=8192

M=8192 当前 W1024 window 的约 24 次 device launch 是最明显的结构问题。

### 10.1 第一优先级：self-publishing window

cycle/drain 内增加 slot-local completion counter：

```text
local group 最后 CTA
→ system release
→ publish partial_ready[slot]

RS group 等待 partial_ready[slot]
→ 完成 RS
→ 最后 CTA publish reduced_ready[slot]

AG group 等待 reduced_ready[slot]
→ 完成 AG
```

这样 host 不再 launch 14 个独立 barrier kernel，保留：

```text
1 × G0
6 × cycle
3 × drain
```

即约 10 次 launch。state 只需要双 slot epoch/counter，不引入通用 protocol 或运行时开关。

### 10.2 第二优先级：窗口宽度

在 self-publish 后比较：

```text
W1024 / W1792 / W3584
```

先固定 compute tile 和 collective grid，只扫 window；选出前两名后再扫 local worker 和
RS/AG grid。不能同时大范围组合所有参数。

目标：

```text
uniform < 582.35 us
skew    < 594.22 us
```

预计收益：删除 barrier launch 5%～12%，窗口宽度 2%～7%。

## 11. 第六阶段：M=16384

M=16384 同时比较优化后的 window 和 persistent，不预设 family winner。

### 11.1 Window 路线

复用 M=8192 的 self-publish 和正确宽窗口，然后单独调：

```text
LOCAL_WORKERS
RS_GRID
AG_GRID
```

### 11.2 Persistent 路线

先做低风险 service-grid sweep：

```text
SERVICE_GRID=32/48/64/77
```

77 是 M=32768 的 winner，不代表适合 M=16384。更小 grid 即使让单独 RS/AG 变慢，也可能
释放更多 CU 给 producer，使完整 graph 更快。

随后测试：

- 合并 drain/final-publish；
- 减少 worker0 gate 串行等待；
- 每 phase completion counter 合并；
- 4-phase W1792 persistent service。

目标：

```text
uniform < 990.01 us
skew    < 1020.64 us
```

预计收益：service grid 1%～4%，self-publish/wide-window 5%～12%，gate/drain 1%～4%。

## 12. 第七阶段：M=32768

M=32768 的目标是相对同轮冻结的融合 baseline 至少降低 5%，具体微秒上限按
`baseline × 0.95` 计算。参数微调可能不足以稳定通过，优先优化 persistent critical path。

### 12.1 第一优先级：RS/AG 跨 phase 流水

当前：

```text
RS[0] → AG[0] → RS[1] → AG[1] → ...
```

目标：

```text
RS[0]
RS[1] + AG[0]
RS[2] + AG[1]
...
AG[last]
```

将 service CTA 分为 RS 和 AG 两组，并使用 phase-private 或双缓冲 state。需要独立调
`RS_SERVICE_GRID` 和 `AG_SERVICE_GRID`，不能继续强制两者共用 SG77。

必须验证 RS/AG 同时运行时是否争抢同一 xGMI/HBM 带宽。如果 overlap 后两个 primitive
都显著变慢且 critical path 没缩短，立即淘汰。

预计完整收益：3%～8%。

### 12.2 第二优先级：producer persistent 化

当前 producer 有 9 次 launch。目标不是一次性写一个巨大万能 kernel，而是先合并最安全的
尾部：

1. drain 内可靠发布最后 phase；
2. 删除 final-publish launch；
3. 再评估将 6 个 phase producer 合成一个 persistent producer。

真正 persistent producer 需要明确解决：

- sorted expert work-item 的跨 phase 调度；
- GEMM CTA 与 local worker 的共驻留；
- phase 完成和 route slot 复用；
- producer/service 的 CU 配额；
- graph 多轮 epoch 复用。

预计收益：2%～6%。

### 12.3 第三优先级：collective 内核

只有 trace 显示 remote-load latency 仍在 critical path 时，才测试：

- source 双缓冲预取；
- wave 内 source-parallel TP8 reduce；
- 降低每线程 FP32 accumulator 数；
- 更连续的 payload/scale load；
- RS/AG 不同 grid 和 vector width。

source-parallel 会改变 FP32 加法顺序，必须重新跑完整精度门禁。不能因为单独 RS 快几微秒
就接受 rel_l2 或 max_abs 回退。

预计收益：1%～5%。

### 12.4 M=32768 达标组合

单项很可能不足 5%，预期组合为：

```text
RS/AG cross-phase overlap    3%～8%
producer launch reduction    2%～6%
gate/collective tail         1%～4%
```

目标：

```text
uniform < 1877.44 us
skew    < 1868.25 us
```

完成后必须重新跑 DeepSeek-V4-Pro TP8 整网 prefill/decode。当前整网 standard→comm-fused
TTFT 已降低约 8.9%；新 kernel 至少不能回退，并期望再降低 3%～5% 的 prefill TTFT。

## 13. 公共优化候选的优先级

| 优先级 | 候选 | 主要 M | 预期完整收益 | 风险 |
|---:|---|---|---:|---|
| 1 | 因果 segmented producer/service overlap | 512/1024 | 目标 >5% | cohort 完成协议、CU 竞争 |
| 2 | full/window completion 协议重构 | 2048～16384 | 4%～12% | epoch/counter 正确性 |
| 3 | 正确 W1792/W3584 | 4096～32768 | 2%～14% | 宽窗口 local mapping |
| 4 | M-specific service grid | 16384/32768 | 1%～4% | CU 与通信平衡 |
| 5 | persistent RS/AG 跨 phase overlap | 32768 | 3%～8% | xGMI/HBM 竞争、死锁 |
| 6 | producer persistent 化 | 32768 | 2%～6% | 调度和稳定性复杂 |
| 7 | remote-load/source-parallel reduce | 8192～32768 | 1%～5% | 精度顺序和 VGPR |

## 14. 明确暂不做的方向

- 不重新接入 MORI SDMA；已有 probe 没有完整 pipeline 收益；
- 不推送 BF16，继续保留 MXFP8 payload + E8M0 scale；
- 不恢复 TK256 或错误 W3584 数据；
- 不恢复已实测更慢的一阶段 TP reduce、barrier-in-kernel、VECTOR16 或两窗口双 stream；
- 不通过增加大量 CTA 追求单 primitive 最小值；
- 不做在线 tuner；
- 不在 production 保留 candidate family、环境变量或 fallback；
- 不为减少文件行数改变 kernel/host 边界；
- 不在没有 trace 证据时直接开发全 persistent 单 kernel。

## 15. 文件修改边界

预计只涉及：

```text
aiter/ops/flydsl/kernels/comm_fused_moe/full_width.py
aiter/ops/flydsl/kernels/comm_fused_moe/windowed.py
aiter/ops/flydsl/kernels/comm_fused_moe/persistent_window.py
aiter/ops/flydsl/kernels/comm_fused_moe/collectives.py
aiter/ops/flydsl/comm_fused_moe_host.py
aiter/configs/comm_fused_moe.csv
op_tests/multigpu_tests/tune_comm_fused_moe.py
```

边界规则：

- kernel 文件实现算法和 compile-time 参数；
- host 只在新 family 成为 winner 时增加最小 launch 流程；
- tuner 保存候选和完整 pipeline 测量；
- CSV 只更新最终 winner；
- ATOM 不参与单算子实验，只在 winner 确定后做整网回归。

## 16. 每个候选的验证流程

每个 M 都按同一顺序执行，不能先凭单 kernel 数字修改 production：

```text
冻结该档当前融合算法 baseline
→ trace/PMC 定位 rank-max critical path
→ 仓外最小 probe
→ 完整 Graph 快速筛选
→ 交替正式 A/B
→ 精度与长稳
→ 全 M 回归
→ 删除 loser 后晋级 winner
```

### 16.1 快速筛选

```text
5 rounds
10 graph replays/round
TP8 rank-max
uniform + skew
每次 Graph replay 后单独同步
```

小于 2% 的结果先视为噪声，不进入 production。只保留每个方向最快的 1～2 个候选。

### 16.2 正式 A/B

```text
frozen-fused-baseline/candidate 交替运行
两轮独立复测
每轮 11 rounds
每个 round 30 graph replays
每次 Graph replay 后单独同步
报告 rank-max median、min、max、p90
```

正式脚本必须同时输出 ordinary，但 pass/fail 只使用：

```text
candidate / frozen-fused-baseline < 0.95
```

不得把 ordinary 当作 5% 门禁分母，也不得在阶段内把一个中间 winner 滚动替换成新分母。

### 16.3 精度

```text
max_abs <= 1.0
rel_l2  <= 0.05
```

同时记录与当前 winner 的误差变化。若加法顺序改变，必须覆盖 uniform、skew 和至少一组
真实模型 routing。

### 16.4 稳定性

- 普通候选：至少 1000 次连续 Graph replay；
- epoch/persistent 候选：uniform、skew 各 10000 次；
- 检查 hang、memory fault、旧 epoch 串轮、输出漂移；
- 仓根生成的 gpucore 直接清理，不进入仓库。

### 16.5 全 M 回归

公共 kernel/helper 修改后必须重跑：

```text
M=512/1024/2048/4096/8192/16384/32768
uniform/skew
eager/Graph
精度
```

任何未目标 M 回退超过 1%，都不能直接提交公共修改。若优化只适合单个 M，应通过新的静态
winner/family 隔离，而不是改变所有 M 的默认参数。

### 16.6 每档必须记录的数据

文档中的每个 M 最终都要补齐一行结果，避免只留下“更快”结论：

| 字段 | 内容 |
|---|---|
| frozen fused baseline | commit、family、完整 config、uniform/skew rank-max median |
| ordinary reference | 同轮普通 Stage2 + shared + TP all-reduce 耗时 |
| candidate | family 和全部 compile-time 参数 |
| 分解 | GEMM、local、barrier、RS、AG、launch gap、critical tail |
| 完整性能 | eager/Graph 的 median、min、max、p90 |
| 精度 | max_abs、rel_l2、真实 routing 结果 |
| 稳定性 | replay 次数、是否 hang/fault/跨轮污染 |
| 回归 | 其他全部 M 的变化和整网 TTFT/吞吐变化 |
| 决策 | winner 或淘汰原因；失败代码是否已删除 |

## 17. 提交顺序

建议每个可独立验证的 winner 单独提交：

1. M=512/1024 小 M 算法 winner；
2. M=2048 full-width winner；
3. M=4096 full/wide-window winner；
4. M=8192 self-publish window winner；
5. M=16384 window/persistent winner；
6. M=32768 persistent pipeline winner；
7. 全 M tuner 数据和最终文档整理。

每个提交前必须：

- 删除 loser；
- 删除仓内临时 probe；
- 更新对应 CSV winner；
- 附完整 A/B 与精度数据；
- 不 amend 下一阶段未 review 的实验。

## 18. 第一轮实际执行清单

从 M=512 开始时，第一轮只做以下工作：

1. 在当前 TP8 节点重跑 512～32768 基线，冻结同场数据；
2. 对 M=512 full-width 做 kernel trace；
3. 审计历史 segmented/cohort probe 和日志，明确 171～225 us 版本的瓶颈；
4. 设计最粗且因果正确的 producer completion cohort；
5. 实现仓外最小 service overlap probe；
6. 达到 5% 后清理并形成 M=512 winner；
7. 回归全部 M，再进入 M=1024。

不在第一轮同时修改 window/persistent。按 M 逐次推进可以更清楚地判断收益来自哪里，也能
避免最后留下多套互相影响的实验实现。

## 19. M=512 已淘汰方向

当前尚无候选同时通过 uniform 和 skew 的严格 5% 门禁。最好但仍未达标的稳定结果约为：

```text
KID2035 + QV32 hybrid-native formal pass 1: uniform 0.952216, skew 0.946960
KID2035 + QV32 hybrid-native formal pass 2: uniform 0.953398, skew 0.947688
```

以下方向已经完成完整 Stage2 + shared + TP communication A/B，不再继续组合：

- RS 最后一个 CTA 发布 reduced-ready：uniform/skew 均无稳定收益，部分组合回退；
- exact quant、native scaled encode 与 RS-last 的组合：不能叠加到 5%；
- Opus Block-N 扫描：BN64 的 KID2094 仅 uniform 接近门槛，skew 为 0.976183；
- 将 barrier 提前排到独立 stream：跨流 event/调度开销抵消潜在 overlap；
- quant-last CTA 自行发布 partial-ready：grid 384/512/768/1024 在 uniform 上分别为
  `1.0118/1.0290/1.0783/1.1205`，在 skew 上分别为
  `1.0165/1.0445/1.0753/1.1279`，全部慢于同轮 baseline；
- short-K 串行加载两个 B half：精度通过，uniform/skew quick ratio 为
  `0.9669/0.9725`；单独有效但未达到 5%，只继续检查与 A2 延迟加载的叠加效果；
- 串行 B half 与“整块 tile0 后发 A2”叠加：uniform/skew quick ratio 为
  `0.9669/0.9589`，skew 继续改善但仍未过门禁；下一候选将 A2 提前到 tile0 的前半 N 之后；
- 将 A2 再提前到 tile0 的前半 N 之后没有收益，uniform/skew quick ratio 为
  `0.9718/0.9632`，已恢复整块 tile0 后发布的版本；
- 将当前最佳 short-K 候选与历史最佳量化/collective 参数组合后，四组结果仍只有
  `0.9588～0.9734`，没有形成双分布 `<0.95` 的配置；
- short-K priority=0 的一次 quick 结果为 `0.9472/0.9622`，只通过 uniform；priority=1
  为 `0.9685/0.9674`，也未通过。只提高最后 K tile priority 的版本为
  `0.9724/0.9675`，已删除；
- K=3 专用 scale 预取为 `0.9673/0.9617`，单 B fragment 串行版本为
  `1.0116/1.0241`；两者均已删除。“两个 B half 串行、整块 tile0 后发布 A2、priority=2”
  的 quick ratio 为 `0.9669/0.9589`，最终也未晋级并已回退；
- 在该保留版本上继续组合 `quant cache=2/3` 与 `RS113/AG112`：`cache=2` 为
  `0.9754/0.9691`，`cache=3` 为 `0.9791/0.9582`，仍未形成双分布 `<0.95`；停止继续
  扫描 quant cache、collective grid 和 priority 的排列组合；
- A-input cache policy 1/2/3：`max_abs` 约 9.1～9.8、`rel_l2` 约 0.48，精度协议错误；
- BN256/BT256：完整融合路径明显回退。
- stock route-output + 独立 local-reduce：uniform/skew 约 `1.10x`，精度正确但明显回退；
- 将当前 KID2035 调度改成 MXFP8 route-output 后 uniform 为 `1.057399x`，仍慢于 direct
  atomic，不再实现复杂的 last-CTA readiness/service 协议；
- KID2034～2037 与 QV32 hybrid-native 交叉 quick：uniform 最好为 KID2037 的
  `0.944251`，skew 最好也只有 KID2037 的 `0.957209`。occupancy 调节改变了两种分布的
  平衡，但没有形成双分布 `<0.95` 的 winner；
- native-exact、native-exact cache1、BF16-hybrid cache1 三个最终量化候选均未通过双分布
  门禁，停止继续排列 quant cache/block/native/exact 参数。

因此下一轮停止修改两条路径共用的 Opus producer，也不再包装 barrier 或排列简单参数组合。
M=512 后续只考察 quant、同步、RS/AG 和因果 overlap 的结构性改动。所有新候选继续先在仓外
quick screen；双分布都满足 `candidate / baseline < 0.95` 后，才进入两轮正式验证。

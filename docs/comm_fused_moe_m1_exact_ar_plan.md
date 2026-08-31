# Comm-fused MoE：M=1/2 exact-M AllReduce 与 megakernel 实施计划

## 1. 目标与范围

本文只讨论以下固定场景：

```text
gfx950 / MI355X
TP=8
M=1 或 M=2
H=7168
I=384
E=384
topk=6
Stage2 A=FP8, W=FP4, output=BF16
```

目标是在保持 exact-M、完整 BF16 输出以及当前数值语义的前提下，降低：

```text
Stage2 GEMM2 + shared expert merge + TP AllReduce
```

的 CUDA Graph replay 延迟。

本计划不把 M=1/2 填充到 8，不引入 MXFP8 Reduce-Scatter/AllGather，不复用面向
M=32768 的 persistent-window 调度。大 M 的瓶颈是通信吞吐；M=1/2 的瓶颈主要是
dispatch、全局完成检测和跨卡同步延迟，两者需要不同算法。

## 2. 当前事实和基线

### 2.1 当前 exact-M 结果

独占 MI355X TP8 节点，CUDA Graph 连续 replay，3 轮 × 30 次，取每轮 rank-max 后的
中位数：

| M | ordinary exact | ordinary pad8 | atomic pad8 | persistent pad8 |
|---:|---:|---:|---:|---:|
| 1 | 18.08 us | 29.27 us | 33.20 us | 33.42 us |
| 2 | 18.90 us | 27.75 us | 31.71 us | 33.44 us |

独立 GEMM tuner 中，M=1 的 Stage2 GEMM2 约为 `7.33 us`。这个数据与 TP8
rank-max 测试不是完全相同的测量口径，只用于估计下限，不能直接与 18.08 us 做严格减法。

### 2.2 ordinary exact 当前调用链

当前 benchmark 的 ordinary 路径是：

```text
partial.zero_()
→ ordinary atomic GEMM2 写 BF16 partial
→ partial.add_(shared_partial)
→ get_tp_group().all_reduce(partial, ca_fp8_quant=False)
```

对于 M=1/2，TP8 BF16 payload 分别只有 14 KiB 和 28 KiB。当前没有设置
`AITER_QUICK_REDUCE_QUANTIZATION`，因此 communicator 会跳过 Quick AllReduce，进入
custom AllReduce。由于 payload 小于 TP8 的 80 KiB 阈值，custom AllReduce 使用单个
`cross_device_reduce_1stage` kernel。

因此当前路径通常包含四个 device operation：zero、GEMM2、add 和 one-stage AR。
CUDA Graph 下输入可以直接注册，custom AR 不需要额外的 staging copy。

### 2.3 当前 RS/AG 不支持 exact M=1/2 的原因

Atomic/persistent RS/AG 使用：

```text
shard_rows = M // TP
global_token = rank * shard_rows + local_token
```

M 小于 8 时 `shard_rows == 0`。这是当前 row-shard 布局的实现限制，不是 collective
语义要求。可以改成不均匀 row shard 或 column shard，但 M=1/2 仍会承担两轮跨卡同步，
不是本计划的首选。

## 3. 性能模型与目标

### 3.1 M=32768 与 M=1 的差异

M=32768 的 BF16 partial 每卡约 448 MiB，压缩、RS/AG 和多 phase overlap 能显著减少
通信关键路径。M=1 只有 14 KiB，压缩节省的字节很少，quantize/decode 和第二轮同步的
固定成本反而更大。

M=1 的优化重点是：

1. 删除 zero 和 standalone shared add；
2. 保留已有 one-stage custom AR 的高效数据通路；
3. 在有明确收益后，删除 GEMM2 到 AR 之间的 kernel 边界；
4. 最后才尝试按 N tile 提前发布结果，实现数微秒级 overlap。

### 3.2 分级目标

| 级别 | M=1 目标 | 说明 |
|---|---:|---|
| 当前基线 | 18.08 us | ordinary exact |
| 两-kernel 目标 | 14--16 us | GEMM2 直接累加 shared + 原 custom AR |
| megakernel 目标 | 11--14 us | 复用 one-stage AR 算法并删除 launch 边界 |
| 已知计算下界 | 约 7.33 us | 仅 GEMM2，不含跨卡同步和归约 |

这些是工程目标，不是预先证明的硬件极限。必须先获得同一 TP8 进程内的 zero、GEMM2、
add、custom AR 单项时间，再收紧 megakernel 的 go/no-go 门槛。

## 4. 阶段 A：最小 shared-accumulator probe

### 4.1 要验证的问题

M=1/2 当前选择的是 atomic Stage2 kernel。它要求输出预先清零，因为 routed expert
结果通过 atomic add 累加。代数上，初值不必是零；可以直接是本 rank 的 shared expert
结果：

```text
output = shared_partial
for routed expert contribution:
    atomic_add(output, contribution)
```

这样最终得到：

```text
output = shared_partial + sum(routed expert contributions)
```

与原来的 `zero → GEMM2 → add(shared)` 等价。

### 4.2 Probe 对照

只增加测试代码，不修改 production dispatch：

```text
baseline-local:
    output.zero_()
    GEMM2(output)
    output.add_(shared)

candidate-local:
    output.copy_(shared)
    GEMM2(output)

baseline-e2e:
    baseline-local
    custom AR

candidate-e2e:
    candidate-local
    custom AR
```

`copy_(shared)` 是测试环境对“shared expert 直接写最终 partial”的保守模拟。真实模型接入
应让 shared expert GEMM 直接产生该 buffer，从而不保留 copy kernel。

### 4.3 正确性检查

必须检查：

- M=1 和 M=2；
- TP8 每个 rank 使用不同 shared_partial；
- local candidate 对 baseline-local；
- candidate-e2e 对 baseline-e2e；
- 所有输出 finite；
- `max_abs` 和 `rel_l2` 不超过原 ordinary 数值误差；
- 至少 1000 次 graph replay 后没有跨轮累加，证明每轮 seed 正确覆盖旧结果。

### 4.4 性能门禁

短筛：3 轮 × 30 replay，rank-max median。

```text
candidate-local < baseline-local
candidate-e2e <= ordinary-exact - 0.8 us
```

正式门禁：7 轮 × 100 replay，uniform/skew，各自满足：

```text
candidate / same-round baseline < 0.95
```

若 copy-seeded 版本已经没有收益，不直接修改生产 shared expert 输出；先用 profiler 判断
copy、zero、add 是否走 shader kernel，以及 custom AR 是否占据大部分余量。

## 5. 阶段 B：生产两-kernel路径

阶段 A 通过后，修改生产数据流，而不是把 `copy_` 固化到生产路径。

### 5.1 目标调用链

从 shared expert 已经完成的时刻计：

```text
K1: ordinary atomic GEMM2
    input/output partial 初值已经是 shared expert result
    routed experts atomic-add 到相同 BF16 buffer

K2: existing TP8 cross_device_reduce_1stage
    input 为 exact [M, H] registered partial
    output 为 exact [M, H] BF16
```

若把 shared expert GEMM 本身也计入，总共是 shared GEMM、routed GEMM2、AR 三个 kernel；
这里的“两 kernel”专指 shared partial 已产生后的 Stage2+TP 路径。

### 5.2 Buffer contract

必须满足：

- partial 是 BF16、连续、exact `[M, H]`；
- partial 地址在 CUDA Graph capture 期间稳定；
- custom AR 能把它注册为 IPC/VMM 输入；
- shared producer 与 routed GEMM2 在同一 stream 上，或存在明确 event 依赖；
- routed GEMM2 只能 atomic-add，不能覆盖 shared 初值；
- 每次 replay 都由 shared producer 完整覆盖 partial，不能依赖 host 清零；
- shared_partial 不再被其他异步消费者读取，允许就地作为 routed 输出。

### 5.3 Kernel 选择限制

第一版只允许 `stage2_uses_route_reduce(metadata.stage2) == False` 的 atomic Stage2 kernel。
如果 tuner 选择 route-output/local-reduce kernel，必须回退普通路径，不能假设它也支持
预置 accumulator。

M=1 和 M=2 分别建立配置，不因为实现方便而 pad 到同一个 bucket。

## 6. 阶段 C：拆解和调优 one-stage custom AR

两-kernel版本稳定后，对 custom AR 做 gfx950 M=1/2 专项扫描。只改 standalone AR，不先做
mega，避免把 AR 回退误判成融合问题。

扫描项：

- `threads/block`: 256、512；
- blocks：围绕 M=1 的 14 和 M=2 的 28 小范围扫描；
- 每线程 vector pack 数；
- LDS double-buffer 是否对 14/28 KiB 仍有收益；
- system/agent fence 范围；
- registered output write mode；
- signal cache-line 和 block/rank 映射。

门禁：candidate AR 必须在独立 AR 和两-kernel E2E 两个口径均获益，且固定 rank 0→7
累加顺序、bitwise repeatability 和 graph replay 均通过。

## 7. 阶段 D：M=1 专用 GEMM2+AR megakernel

只有阶段 B/C 证明仍有至少约 2 us 可回收空间时才进入。

### 7.1 CTA 布局

当前 M=1 为 6 个有效 routed expert block、28 个 N tile，共 168 个 GEMM CTA。候选布局：

```text
CTA [0, 168):      GEMM2 producer
CTA [168, 182/196): one-stage AR service
```

service CTA 数由移植后的 256-thread AR microbenchmark 决定，首测 14 和 28。总 CTA 数低于
256，但仍必须通过 occupancy API/编译资源报告确认所有 service CTA 与 producer CTA 可以
同时驻留；不能仅依据 CU 数证明 residency。

M=2 有 336 个 GEMM CTA，首轮不做同样假设。M=1 megakernel 成功后，再为 M=2 选择：

- 保留两-kernel路径；或
- bounded persistent producer grid；或
- 少量 service CTA 与 producer waves 共存。

### 7.2 同步协议

第一版只做一次全局 readiness，不做 28 套跨卡 barrier：

```text
每个 producer CTA 完成输出写入
→ 发布自己的 local epoch 槽

一个 coordinator CTA 扫描全部 producer epoch
→ system release fence
→ 发布 rank_ready[epoch]

所有 service CTA 等待本 rank gate
→ 进入移植的 cross_device_reduce_1stage
→ 复用其 per-block/per-rank signal 协议
→ 直接写最终 BF16 output
```

使用每 CTA 独立 epoch 槽，避免 168 个 CTA 争抢一个全局 atomic counter。coordinator 在
等待期间使用低开销 sleep。所有状态带 invocation epoch，支持 CUDA Graph 重放，不在每轮
清零整个状态区。

### 7.3 AR 数据通路

不能沿用早期 probe 的朴素 remote-load 循环。需要逐项移植主线
`cross_device_reduce_1stage`：

- 固定 rank 顺序，保证各 rank 结果一致；
- vector pack；
- warp/rank 分工；
- LDS 双缓冲和下一轮预取；
- 当前 start-sync 协议；
- registered input/output pointer table；
- 小消息 block 数计算。

由于 GEMM2 使用 256 threads，而主线 AR 使用 512 threads，mega 第一版必须提供
256-thread AR specialization。不能改变同一个 launch 内不同 CTA 的 block size。

### 7.4 Megakernel 门禁

```text
M=1 mega / same-round two-kernel < 0.95
```

并要求：

- uniform/skew 都通过；
- 10000 次 graph replay 无 hang、串轮或 stale epoch；
- 八个 rank 输出一致；
- 不降低普通 M=2 和其他 production bucket；
- 编译资源没有使 GEMM2 本体回退超过 0.5 us。

若 mega 不能稳定低于两-kernel版本，则保留两-kernel winner，删除 mega 实验实现。

## 8. 阶段 E：可选的 tile-ready overlap

只有全局 gate mega 已经接近或超过两-kernel版本时再尝试。

M=1 每个 N tile 有 6 个 routed expert CTA。为每个 N tile 维护 local completion epoch：

```text
6 个 producer 完成 tile n
→ tile_ready[n]
→ 对应 service CTA 等待所有 rank 的 tile_ready[n]
→ 归约该 256-column tile
```

这样前面 tile 的 AR 可以与后面 tile 的 GEMM 尾部重叠。但旧实验使用 28 套自定义跨卡
ready/consumed，M=1 达到 44.19 us，已经证明同步设计不当会完全吞掉收益。新实现必须复用
custom AR 的 per-block signal，不增加第二套 consumed barrier。

预期 overlap 上限只有 GEMM2 的约 7.33 us；如果 tile readiness 集中在 kernel 尾部，实际
收益可能只有 1--2 us。

## 9. 测试矩阵

### 9.1 正确性

```text
M:       1, 2
route:   uniform, skew
mode:    eager, CUDA Graph
replay:  1, 1000, 10000
```

检查：finite、max_abs、rel_l2、rank 间一致性、epoch wrap/reuse、不同输入连续 replay。

### 9.2 性能

每项在独占 TP8 节点同进程交替执行：

```text
ordinary exact
shared-seeded two-kernel
two-kernel + tuned AR
M=1 megakernel
```

统计 7 轮 × 100 replay 的 rank-max median，同时保留 min/max，避免只报告单 rank 或单轮
最小值。

### 9.3 Profiler

对最终候选采集一次 kernel trace，确认：

- 实际 dispatch 数；
- custom AR 是否为 `cross_device_reduce_1stage`；
- graph 模式没有隐藏的 D2D copy；
- mega 中 producer/service 的开始和结束位置；
- 没有 ROCclr blit/fill kernel 意外落入关键路径。

## 10. 实施顺序与退出条件

1. 增加 shared-accumulator probe，不改 production dispatch。
2. 若正确且 local/E2E 有收益，将 exact-M 两-kernel路径接入 M=1/2。
3. 单独拆解并扫描 custom AR 小消息参数。
4. 只为 M=1 实现复用 one-stage AR 的 megakernel。
5. 全局 gate mega 通过后，才尝试 tile-ready overlap。
6. 每阶段失败即删除失败实现，只保留结果和结论，避免实验分支持续膨胀。

明确停止条件：

- shared accumulator 无法保持当前数值误差；
- 两-kernel版本不能稳定优于 18.08/18.90 us；
- AR 调优只改善 standalone、却不改善 E2E；
- mega 的同步/资源代价抵消 launch 收益；
- M=1 优化导致 M=2 或其他 production bucket 回退。

最终 production dispatch 应按 exact token bucket 选择算法：

```text
M=1:     two-kernel 或专用 mega winner
M=2:     two-kernel winner，mega 需独立通过门禁
small M: ordinary/atomic winner
large M: full/window/persistent winner
```

不存在一个实现同时适合全部 M；每个 bucket 只保留实测 winner。

## 11. 阶段 A 首轮结果（2026-08-27）

最小 probe 已在独占 `crsuse2-m2m-v2-024` 上完成。口径为 TP8、uniform/skew route、
CUDA Graph、3 轮 × 30 replay、每轮取 rank-max。

Probe：

`op_tests/multigpu_tests/probe_flydsl_moe2_shared_accum_tp8.py`

结果日志：

`/home/yifehuan/data/box_comm_fused_moe_sdma/out/1012_shared_accum_probe_20260827.log`

`/home/yifehuan/data/box_comm_fused_moe_sdma/out/1013_shared_accum_skew_probe_20260827.log`

| M | route | 项目 | baseline | shared-seeded candidate | 改善 |
|---:|---|---|---:|---:|---:|
| 1 | uniform | local producer | 11.13 us | 8.65 us | 2.48 us / 22.3% |
| 1 | uniform | producer + custom AR | 18.20 us | 16.07 us | 2.12 us / 11.7% |
| 2 | uniform | local producer | 10.86 us | 9.38 us | 1.48 us / 13.6% |
| 2 | uniform | producer + custom AR | 19.00 us | 17.03 us | 1.97 us / 10.4% |
| 1 | skew | local producer | 11.12 us | 9.21 us | 1.90 us / 17.1% |
| 1 | skew | producer + custom AR | 18.23 us | 15.98 us | 2.25 us / 12.3% |
| 2 | skew | local producer | 10.68 us | 9.09 us | 1.59 us / 14.9% |
| 2 | skew | producer + custom AR | 18.31 us | 16.38 us | 1.92 us / 10.5% |

数值结果：

| M | route | local max_abs / rel_l2 | E2E max_abs / rel_l2 |
|---:|---|---:|---:|
| 1 | uniform | 0.0234375 / 0.004815 | 0.03125 / 0.003370 |
| 2 | uniform | 0.0156250 / 0.004513 | 0.03125 / 0.003237 |
| 1 | skew | 0.0234375 / 0.004728 | 0.03125 / 0.003357 |
| 2 | skew | 0.0156250 / 0.004431 | 0.03125 / 0.003248 |

首轮结论：

1. 当前 M=1/2 atomic Stage2 可以正确累加到非零 BF16 初值；不需要修改 GEMM
   数学主体即可吸收 shared add。
2. 即使 probe 仍用一次 `copy_(shared)` 模拟 shared producer，端到端也已稳定改善约
   2 us，超过阶段 A 的短筛门槛。
3. 真实生产路径若让 shared expert GEMM 直接写入 registered partial，可进一步删除
   probe 中的 copy；下一步应先实现这个 buffer contract，而不是立即开始 megakernel。
4. uniform 和 skew 的短筛均通过；进入生产修改前还需补较长 replay 和实际模型调用链的
   buffer 生命周期验证。

## 12. 阶段 C 首轮结果（2026-08-27）

已增加 standalone FlyDSL TP8 one-stage BF16 AllReduce：

`aiter/ops/flydsl/kernels/comm_fused_moe/small_m_allreduce.py`

实现复用了主线 custom AR 的关键结构：16-byte BF16 pack、8 个 logical subgroup 分别
读取 8 个 rank、LDS staging、固定 rank 顺序 FP32 累加，以及每个 workgroup 独立的
start/end epoch。同步必须使用主线同类的 32-bit system store 与 agent-scope polling；
早期通用 64-bit system polling 会让 M=2 回退约 2 us。

重构后的默认 512-thread 路径在独占 `crsuse2-m2m-v2-024` 上重新回归，口径为 TP8、
CUDA Graph、3 轮 × 50 replay、每轮取 rank-max：

| M | blocks | C++ custom AR | FlyDSL AR | FlyDSL 相对变化 |
|---:|---:|---:|---:|---:|
| 1 | 14 | 8.20 us | 7.30 us | 快 11.1% |
| 2 | 28 | 8.23 us | 7.86 us | 快 4.5% |

两条路径均为 bitwise exact，`max_abs=0`、`rel_l2=0`。回归日志：

`/home/yifehuan/data/box_comm_fused_moe_sdma/out/1023_flydsl_small_m_ar_t512_regression_20260827.log`

在完整 `shared init + atomic GEMM2 + AR` probe 中，FlyDSL AR 也能转化为端到端收益：

| M | 原 ordinary 路径 | shared-accum + custom AR | shared-accum + FlyDSL AR |
|---:|---:|---:|---:|
| 1 | 18.25 us | 15.96 us | 15.17 us |
| 2 | 18.86 us | 16.91 us | 16.11 us |

日志：

`/home/yifehuan/data/box_comm_fused_moe_sdma/out/1020_shared_accum_flydsl_ar_e2e_20260827.log`

为后续单-launch 准备的 256-thread 兼容形态也已验证。M=1 使用 14 blocks、每 block
两轮 double-buffer iteration 时为 8.99 us，同轮 custom AR 为 8.82 us，仅慢 2.0%，
且结果 bitwise exact。日志：

`/home/yifehuan/data/box_comm_fused_moe_sdma/out/1022_flydsl_small_m_ar_t256_b14_20260827.log`

当前结论是：shared-accumulator 与 standalone FlyDSL AR 都已产生稳定收益，适合先作为
独立 checkpoint 固化；尚未完成的是把 256-thread AR service body 嵌入 GEMM2 的真正
单-launch kernel。

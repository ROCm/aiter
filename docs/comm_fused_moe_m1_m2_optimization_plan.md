# Comm-fused MoE M=1/2 极限性能优化计划

## 1. 目标与范围

本文只优化以下固定生产形状：

```text
GPU:        gfx950 / MI355X
TP:         8
M:          1 或 2，exact-M，不 pad 到 8
H:          7168
I:          384 / TP rank
experts:    384
topk:       6
Stage2:     FP8 activation × FP4 weight
output:     BF16
execution:  CUDA Graph replay
```

目标是继续降低单-launch：

```text
direct-small-M GEMM2
→ per-N-tile local route reduction
→ TP8 one-stage BF16 AllReduce
```

的 rank-max 延迟。本文不把 M=4/8/16 的调度问题混入 M=1/2，也不重新引入
RS/AG、SDMA、cooperative launch、GWS full-grid barrier 或常驻 polling service CTA。

首轮目标：

```text
M=1: stable rank-max < 9.0 us
M=2: stable rank-max < 12.0 us
```

长期工程目标而非预先承诺的硬件极限：

```text
M=1: 7.0--8.5 us
M=2: 9.0--11.0 us
```

## 2. 为什么 M=1/2 是同一个性能区间

当前使用 `tile_m=16, tile_n=512, tile_k=128`。uniform routing 下，每个 token
最多产生 6 个非空 routed expert，每个 expert 有 14 个 N tile：

| M | compute CTA 上界 | 256 CU 调度轮数 | 每 tile BF16 元素 | 256-thread service 轮数 |
|---:|---:|---:|---:|---:|
| 1 | 84 | 1 | 512 | 1 |
| 2 | 168 | 1 | 1024 | 1 |
| 4 | 336 | 2 | 2048 | 1 |
| 8 | 672 | 3 | 4096 | 2 |
| 16 | 1344 | 6 | 8192 | 4 |

M=1/2 的所有 producer CTA 都能在一轮 CU 调度容量内运行，且每个 tile 的 local reduce
和 TP reduce 都只需一轮 service。因此该区间的关键问题是同步延迟、rank skew、最后 tile
的 local/remote reduce 尾巴，而不是多轮 producer 调度或持续通信带宽。

M=1 和 M=2 共用同一个源码 generator，但必须保留 exact-M 编译特化，使编译器静态删除：

- 不存在的输出行；
- 不需要的 service iteration；
- 多余地址计算和 active mask；
- 不属于当前 M 的 LDS 空间。

这意味着维护一套公共源码、两个 binary，而不是复制两套算法。

## 3. 当前权威基线

原节点 M=1 正式结果：

| 路径 | rank-max median |
|---|---:|
| route Stage2 only | 5.3445 us |
| atomic Stage2 + partial init | 9.6757 us |
| two-kernel Stage2 + AR | 17.3294 us |
| single-launch tile-ready megakernel | 9.6249 us |
| production ordinary | 17.5038 us |
| production `_SmallMRunner` | 9.6809 us |

10,000 graph replay：

| route | M=1 fused rank-max | max_abs | rel_l2 |
|---|---:|---:|---:|
| uniform | 9.4853 us | 0.03125 | 0.002823 |
| skew | 9.4570 us | 0.03125 | 0.002791 |

当前新节点极短 `1×10` 样本：

| M | route | ordinary | small megakernel |
|---:|---|---:|---:|
| 1 | uniform | 21.284 us | 10.936 us |
| 1 | skew | 21.324 us | 11.716 us |
| 2 | uniform | 23.832 us | 14.028 us |
| 2 | skew | 25.868 us | 12.556 us |

不同节点的绝对值不能直接混合。所有优化必须在同一轮、同一节点、同一 JIT 环境中与默认
explicit-release 版本 A/B。

## 4. 当前 kernel 的关键路径

每个 N tile 有 6 或 12 个 producer CTA。流程为：

```text
producer GEMM direct epilogue 写 route slot
→ s_waitcnt + workgroup barrier
→ agent-release
→ per-tile completion atomic

最后到达的 producer CTA：
→ agent-acquire
→ 读取 shared + 6 路 route
→ balanced FP32 local reduction
→ BF16 partial 写 symmetric buffer，同时保留本 rank LDS 副本
→ s_waitcnt
→ explicit system-release
→ system-scope peer-ready store
→ 等 8 rank ready
→ 7 路 remote BF16 load + 1 路 LDS local load
→ balanced FP32 TP reduction
→ BF16 output store
```

M=1 每 tile 从 7 个 peer 读取约 7 KiB，M=2 约 14 KiB。该规模主要受远端访问延迟、
可见性协议和 rank skew 影响。MXFP8 只能节省约 3.5/7 KiB 每 tile，却增加 group max、
scale、pack 和 decode，不列入 M=1/2 候选。

## 5. 性能下限

M=1 已测 route Stage2-only 为 5.3445 us，因此严格下限为：

```text
T_M1 >= 5.3445 us
```

真正可实现下限还要包含最后一个 tile 无法隐藏的：

```text
local reduce
+ payload publication
+ cross-rank ready skew
+ final remote reduce/store
```

因此 M=1 的现实硬件区间预计为 7.0--8.5 us。低于约 6 us 除非改变数学、数据类型或
collective 语义，否则不可信。

M=2 还缺少同一正式口径的 route Stage2-only 时间。由于 168 个 CTA 仍小于 256 CU，其
compute critical path 不应简单按 M 翻倍，但更高的并发会增加权重/L2 压力，tile service
payload 也翻倍。第一轮必须补齐：

```text
M=2 route Stage2-only
M=2 local-reduce-only
M=2 ready + remote-reduce tail
```

在得到分项前，M=2 的 9--11 us 只作为工程目标，不宣称为已证明硬件下限。

## 6. 实验 1：删除 explicit system-release

### 6.1 要回答的问题

当前 hot path 在 symmetric partial 写完后执行：

```text
s_waitcnt vmcnt(0)
workgroup barrier
thread0 system-release fence
workgroup barrier
system-scope ready store
```

实验候选仅删除 `system-release fence`，保持以下内容完全不变：

- partial store；
- `s_waitcnt vmcnt(0)`；
- 两侧 workgroup barrier，首轮不顺带删除；
- system-scope ready store；
- peer polling；
- remote payload load cache policy；
- 所有 GEMM/local-reduce/TP-reduce 数学顺序。

候选必须使用独立 kernel symbol/JIT cache key，避免误复用 baseline binary。产品默认仍为
explicit release。

### 6.2 为什么旧 18.60 us 证据不能直接复用

旧 coarse fast path 的 payload 写发生在 completion release 之前，最终 ISA 包含：

```text
s_waitcnt vmcnt(0)
buffer_wbl2 sc1
completion atomic
buffer_inv sc1
system-coherent ready store
```

当前 tile-ready winner 不同：最后 producer 在 completion acquire 之后重新执行 local
reduce，再写一个新的 BF16 partial。删除此处 system-release 后，新 partial store 和 ready
store 之间未必还有 `buffer_wbl2`。

因此旧路径的 50k/20k generation `0 mismatch` 只证明旧 ISA 链，不证明当前路径删 fence
仍然安全。

### 6.3 静态 ISA 门禁

baseline 和 candidate 都要反汇编，定位：

```text
最后一批 partial buffer_store
→ payload waitcnt
→ optional buffer_wbl2/system fence
→ ready global_store sc0 sc1
→ ready poll
→ remote buffer_load
```

分类：

1. candidate 仍生成等价 `buffer_wbl2`：删除源码 fence 没有真正改变机器级同步，继续测
   性能但不能把它归因于删 writeback；
2. candidate 删除了 payload writeback：这是实质性弱化，只能作为高风险 probe；
3. candidate 重排 ready store 到 payload completion 之前：立即 no-go，不运行压力测试。

### 6.4 短测顺序

所有测试先使用：

```text
COMM_FUSED_PERF_ROUNDS=1
COMM_FUSED_PERF_ITERS=10
```

依次执行：

1. M=1 uniform baseline/candidate correctness + Graph rank-max；
2. M=1 skew；
3. M=2 uniform；
4. M=2 skew。

每次都检查所有 rank：

- finite；
- `max_abs`、`rel_l2`；
- 输出与 ordinary reference 一致到现有阈值；
- 无 hang、stale epoch 或 graph replay 污染。

### 6.5 晋级门槛

候选只有同时满足以下条件才进入 generation-changing 压测：

```text
四个 M/route case 均正确
候选 rank-max 至少改善 0.20 us 或 2%
没有某个 route 回退超过 1%
ISA 顺序能够被完整解释
```

若性能差异在噪声内，保留 explicit release，删除实验开关。

### 6.6 generation-changing 压测

短测晋级后，不直接跑百万次，先执行：

```text
M=1: 10,000 generations
M=2: 10,000 generations
uniform + skew
每代输入同时依赖 generation、rank 和 element index
```

必须避免固定输入导致读取上一代数据仍然假通过。任意 mismatch、hang 或 rank divergence 都
立即判定 no-go。

即使 10k 全部通过，删除 fence 仍不能仅凭经验成为无条件产品路径。产品化还需满足至少一个：

- 源码/ISA 内存顺序得到明确平台保证；或
- 对 gfx950 + 已验证 ROCm compiler/runtime 范围做严格 guard，并保留 fallback；或
- 找到性能等价且语义明确的更轻 release primitive。

## 7. 后续优化顺序

### 7.1 M=2 独立分项时间线

system-release 实验完成后，优先补齐 M=2 的：

- route Stage2-only；
- 最早/最晚 tile completion；
- local reduce 完成；
- ready 收齐；
- remote reduce 完成。

只凭总时间不能判断是 168 producer CTA 的 compute/L2 压力，还是 2-row collective tail。

### 7.2 M=2 service 映射

当前 16-byte pack 下，M=2 每 tile 有 128 个 pack，即两个 wave 执行数据工作、四个 wave
参与 barrier。仅测试两个候选：

```text
PACK=8: 128 threads，16-byte load，当前基线
PACK=4: 256 threads， 8-byte load，四 wave 全参与
```

候选必须保持固定 rank 0→7 的 FP32 累加顺序。若窄 load 增加 transaction 或 VGPR 压力，
立即停止，不继续扩展 pack sweep。

### 7.3 M=2 tile-ready 粒度

M=1 的 grouped/coalesced ready 已经实测回退，不重复。M=2 只有在时间线证明 14 套 ready
占据明显关键路径时，才比较一次：

```text
14 × 1 tile-ready
7 × 2-tile-ready
```

若没有至少 2% 收益，保留逐 tile overlap。

### 7.4 M=1 只做证据驱动的尾部优化

M=1 已淘汰以下方向，不再重复：

- rank-major 或 64/128 B padded ready；
- remote-poll、global shared gate、grouped/coalesced ready；
- N-major CTA 重排；
- peer-pair + shuffle reduction；
- wave-only service 或删除必要 block barrier；
- `tile_n=1024`；
- system-coherent partial store；
- resident service CTA、software full-grid barrier、cooperative/GWS coarse path。

除 system-release A/B 外，新的 M=1 修改必须先由 ISA 或时间线指出具体未隐藏尾巴，不能重新
扫已失败参数。

## 8. 测试纪律

每个候选遵循：

```text
静态检查
→ 编译/ISA
→ 1×10 uniform
→ 1×10 skew
→ 同轮 baseline/candidate rank-max
→ 只有 winner 才做 7×100 和 10k generation
```

运行要求：

```bash
ulimit -c 0
export HSA_COREDUMP_PATTERN=/dev/null
```

- 使用显式 timeout；
- 失败或取消时清理整个 torchrun 子进程树；
- 报告 rank-max，不报告 rank0 或单次最小值；
- baseline 与 candidate 使用独立 kernel symbol，但相同输入、route、JIT/toolchain；
- 不因某一个 route 获益而掩盖另一个 route 回退；
- 失败 probe 在结论写入本文后删除；
- 未完成正式门禁前，不改变 production 默认语义。

## 9. 交付状态表

| 项目 | M=1 | M=2 | 状态 |
|---|---:|---:|---|
| explicit-release baseline | 已有正式结果 | 已有短测 | 保留默认 |
| no-system-release ISA A/B | 已完成 | 不再运行 | candidate 少一个 `buffer_wbl2 sc0 sc1` |
| no-system-release correctness | 失败 | 因 M=1 失败而停止 | no-go |
| no-system-release generation litmus | 不运行 | 不运行 | correctness 首测已失败 |
| route Stage2-only | 5.3445 us | 6.680 us | 已完成分项定位 |
| PACK=4 four-wave service | 不适用 | 已淘汰 | uniform 持平、skew 回退 9.1% |
| two-service-CTA | 不适用 | 已淘汰 | correctness 通过，fused 回退到 18.512 us |
| per-producer generation flags | 未进入 | 已淘汰 | 14.216 us，对照 atomic 13.888 us，慢 2.4% |
| per-tile 128B state padding | 回退 1.0% | 仅 M=2 特化有收益 | 禁止按 M 分支，因此不产品化 |
| pair-ready | 已淘汰 | 条件触发 | 非默认 |

最终只保留通过正确性、同轮性能和长 replay 三重门禁的 variant。即使 no-release 更快，若
无法建立可依赖的内存顺序保证，也只能保留为实验结论，不能替换默认产品路径。

## 10. 实验 1 结果：no-system-release 直接 no-go

2026-08-28 在 `mi355-gpu-41`、gfx950 TP8 上完成第一轮最小实验。baseline 和 candidate
使用独立 FlyDSL cache 与独立 kernel symbol；candidate 只删除 explicit
`fence_system_release()`，保留两侧 workgroup barrier、system-scope ready store 和其余
数据通路。

baseline M=1 uniform 单次 correctness smoke：

```text
max_abs = 0.031250
rel_l2  = 0.002773
```

no-system-release candidate 在第一次相同 smoke 即失败：

```text
max_abs = 7.345703
rel_l2  = 0.952354
```

错误同时出现在多个 rank，幅度接近读取未发布或陈旧 partial，而不是普通 BF16 舍入误差。
因此没有继续运行性能、M=2 或 generation 压测。

提取两个 JIT artifact 的 gfx950 ISA 后，关键计数为：

| 指令 | baseline | no-system-release |
|---|---:|---:|
| `buffer_wbl2` | 2 | 1 |
| `buffer_inv` | 1 | 1 |
| `global_store_dword ... sc0 sc1` ready store | 1 | 1 |

baseline 在新 partial 写完、ready store 之前额外包含：

```text
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_barrier
buffer_wbl2 sc0 sc1
s_waitcnt vmcnt(0)
s_barrier
global_store_dword ... sc0 sc1
```

candidate 删除的正是第二个、负责发布 local-reduce partial 的
`buffer_wbl2 sc0 sc1`；ready store 本身仍为 system-coherent，但没有把此前由整个 CTA
写入的 payload 正确发布给 peer。首次运行即出现大幅错误，直接证明当前 tile-ready
kernel 不能依赖 ready store 隐式发布 payload。

最终结论：

```text
explicit system-release 是当前算法的正确性必要条件；
不进入性能门禁；
不保留实验环境开关；
M=1/2 production 继续无条件执行 fence_system_release()。
```

## 11. 实验 2 结果：M=2 PACK=4 four-wave no-go

候选把 service vector 从 8 个 BF16 缩为 4 个 BF16，使 M=2 每 tile 从 128 个 active
threads 扩展到完整 256 threads。GEMM、system-release、ready 和 reduction 顺序均不变。

gfx950 TP8、Graph rank-max、`1×10`：

| route | PACK=8 baseline | PACK=4 | 变化 |
|---|---:|---:|---:|
| uniform | 13.788 us | 13.780 us | -0.008 us，噪声内 |
| skew | 12.372 us | 13.492 us | +1.120 us，慢 9.1% |

两条路径 correctness 均为 `max_abs=0.03125`，`rel_l2≈0.00263--0.00268`。

结论：增加 active wave 没有提高 uniform 的远端吞吐，反而因 8-byte load、更多 wave 工作
和同步压力使 skew 明显回退。保留 16-byte `PACK=8`，删除实验开关，不做 PACK=2 扩展。

## 12. M=2 完整 kernel 分项定位

在 gfx950 TP8、uniform、Graph rank-max、`3×20` 下，对同一个 direct-small-M GEMM2
emitter 增加受限截断点，得到：

| 截断位置 | rank-max median |
|---|---:|
| route Stage2-only | 6.680 us |
| local-reduce 完成 | 8.826 us |
| cross-rank ready 完成 | 10.800 us |
| 完整 fused | 11.730 us |

截断 kernel 会改变 codegen，因此这些数字不能直接相减当成严格阶段耗时。为避免这个问题，
又在完整 fused kernel 的最后一个 N tile 内加入 `s_memrealtime` 时间戳，八个 rank 观测到：

```text
local reduce      0.07--0.11 us
system release    0.04--0.06 us
cross-rank ready  0.08--1.14 us
remote reduce     0.09--0.11 us
total tail        0.30--1.36 us
```

HSA system timestamp frequency为 1000 MHz。由此可知 local/remote payload 本体都只有约
0.1 us；当前剩余成本主要来自所有 producer CTA 共同承担的 completion 协议，以及把同步、
local reduce 和 remote reduce 融入 GEMM 后带来的控制流和寄存器膨胀。

FlyDSL artifact metadata 也支持这一判断：

| variant | VGPR | SGPR | LDS |
|---|---:|---:|---:|
| M=2 route-only | 148 | 46 | 4224 B |
| local-only | 148 | 54 | 4224 B |
| ready/full | 150 | 54 | 4224 B |

因此下一轮不再优先改 AR 向量宽度或拆更多 service CTA，而应降低 producer completion 的
每-CTA 成本和完整 kernel 的控制流/SGPR 压力。

## 13. M=2 双 service CTA no-go

候选让两个 CTA 各处理一行。correctness 通过：

```text
max_abs = 0.031250
rel_l2  = 0.002680
```

但 fused Graph rank-max 回退到 `18.512 us`，远慢于约 `11.7--12.1 us` 的单 service CTA
路径。M=2 的 payload 只需一个 256-thread CTA 的一次 service iteration；额外 CTA 引入的
同步和协调成本远大于并行收益。候选代码已删除。

## 14. Per-producer generation flags no-go

候选将每个 N tile 的 12-way contended completion atomic 替换为：每个 producer 写独立
generation flag，由固定最后一个 expert CTA 的 12 个线程并行等待。首次普通 store 版本会
挂住；改成 agent-release store 后 correctness 通过：

```text
max_abs = 0.031250
rel_l2  = 0.002676
```

同节点、同配置、M=2 uniform、Graph rank-max、`1×10`：

| completion | fused |
|---|---:|
| per-tile atomic | 13.888 us |
| generation flags | 14.216 us |

flags 慢 `0.328 us`，约 `2.4%`。说明当前 12-way atomic 的竞争还没有重到足以抵消 12 个
release store、固定 service CTA 扫描 flag 和额外地址/控制流。该候选没有进入 skew 或长测，
代码已删除，production 保持 per-tile atomic。

## 15. Per-tile 128B state padding：性能有效但策略 no-go

候选只改变 M=2 的同步状态布局：每个 N tile 的 `done`、`epoch` 和 `ready[8]` 各自占用
独立 128B 区域。GEMM、local reduce、system-release、ready 协议和 TP reduce 均不变；
M=1 保持原紧凑布局。

同时对 M=1 full-pad 做了 `3×20` 门禁：baseline `10.730 us`，pad128 `10.838 us`，
回退约 `1.0%`。由于 uniform 已回退，不继续运行 skew。其原因与预期一致：M=1 每 tile
只有 6 个 completion atomic，cache-line 竞争较弱，padding 增加的状态访问跨度无法被收益
抵消。

gfx950 TP8、Graph rank-max、同节点 `3×20`：

| route | compact baseline | M=2 pad128 | 变化 |
|---|---:|---:|---:|
| uniform | 12.718 us | 11.314 us | -1.404 us，快 11.0% |
| skew | 11.892 us | 11.726 us | -0.166 us，快 1.4% |

两种 route 均通过 correctness，`max_abs=0.031250`，`rel_l2≈0.00267--0.00270`。
结果说明 M=2 的 168 个 producer CTA 同时更新 14 个紧邻 counter 时确实存在明显 cache-line
争用；uniform 的 tile 并发更高，因此收益远大于 skew。但项目约束明确禁止使用
`config.m == 2` 选择不同状态布局或微架构；统一 pad128 又会让 M=1 回退。因此该候选只作为
硬件诊断结论保留，不进入产品代码。

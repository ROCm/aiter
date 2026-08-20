# Comm-Fused MoE Persistent Kernel 实现计划

## 1. 一句话结论

当前 M=32768 `windowed.py` 已经达到约 2.2 ms，它是可用的 production
baseline，不应被直接重写。下一个候选方案是新增独立的
**persistent communication KID**：G/L 仍由窗口 producer kernel 计算，一个
persistent service kernel 在一次 Stage2 内持续处理七个窗口的 RS/AG，用
device-side epoch 代替每个窗口的 host-launched barrier kernel。

它只有在同场 A/B 中稳定快于当前 `windowed.py`、整网也无回退时，才会
替换 M=32768 的 shape 映射。

## 2. 为什么现在值得做

最新 TP8 Graph rank-max median：

| shape / route | 当前 production KID |
| --- | ---: |
| M=2048 uniform | 261.511 us |
| M=2048 skew | 266.215 us |
| M=32768 uniform | 2215.401 us |
| M=32768 skew | 2221.662 us |

M=32768 已经达到原定约 2200 us 的目标，所以 persistent 不是功能修复，而是
降低调度和同步成本的下一个性能 KID。

当前 M=32768 有七个 1024-column 窗口，host 调度是：

```text
1 x G0
6 x (G[n] + L[n-1] + optional RS/AG)
3 x drain
7 x partial epoch barrier
7 x reduced payload epoch barrier
```

一共是：

```text
10 个工作 kernel + 14 个 barrier kernel = 24 次 device launch / Stage2
```

CUDA Graph 可以避免 Python 逐次提交，但不会消除 24 个 device graph node 的
dispatch、14 次 system-scope epoch 同步和窗口间 CTA 重调度。

M=2048 的粗粒度窗口实验也说明，仅增加 overlap 但不减少 launch/barrier
不足以提速。最佳两窗口候选仍为 uniform `273.418 us`、skew
`275.915 us`，慢于当前 full-width KID。

## 3. 第一版的准确边界

### 要做

- 只针对当前已验证的 TP8、M=32768、H=7168、I/TP=384、E=384、topk=6。
- 一次 Stage2 启动一个 persistent RS/AG service kernel，在 kernel 内处理七个窗口。
- G/L 保留现有 FlyDSL GEMM emitter 和窗口 compact-route 布局。
- 用单调 device epoch 发布 partial/reduced readiness，不再为每个窗口 launch
  `compile_epoch_barrier()`。
- persistent service 和 G/L producer 使用两条固定 GPU stream 并发，runner
  初始化时创建所有 stream/event/workspace。
- 通过仓外 benchmark 做 KID A/B，production 代码最终只保留 winner。

### 第一版不做

- 不把 G/L/RS/AG 全部塞进一个大而全的 persistent kernel。
- 不做跨多次 Stage2 常驻、无限循环的 daemon kernel。persistent 只在一次
  Stage2 内跨七个窗口存活。
- 不增加 runtime planner、Protocol、registry、在线 tuner、fallback 或 transport
  开关。
- 不改 M=2048 `full_width.py`、普通 MoE、ATOM adapter 和 Stage2 runtime seam。
- 不同时引入 SDMA。SDMA 仍是另一个独立 KID，不与 persistent 实验混在一起。
- 不为“未来可能会有”的 shape 提前加入动态分支。

## 4. 目标架构

```text
current/model stream                         persistent service stream

record start event ------------------------> wait start event
G0                                          launch persistent RS/AG once
G1 + L0 -- publish partial_ready[0] -------> wait all ranks partial[0]
G2 + L1 -- publish partial_ready[1]          RS0 -> publish reduced_ready[0]
G3 + L2 -- publish partial_ready[2]          wait all reduced[0] -> AG0
G4 + L3 -- publish partial_ready[3]          RS1 -> AG1
G5 + L4 -- publish partial_ready[4]          RS2 -> AG2
G6 + L5 -- publish partial_ready[5]          ...
     L6 -- publish partial_ready[6]          RS6 -> AG6 -> record done event

main stream <------------------------------- record done event
wait done event, then return BF16 output
```

在保留现有 G/L 分窗口计算的前提下，目标 launch 数是：

```text
1 x G0
6 x (G[n] + L[n-1])
1 x L6
1 x persistent RS/AG
= 9 个 kernel launch / Stage2
```

与当前 24 次相比，去掉全部 14 个 epoch barrier kernel，工作 kernel 从 10 个
减为 9 个。多 stream event 会成为 Graph 依赖节点，但不会变成每窗口的 GPU
barrier kernel。

## 5. Device-side 同步协议

### 5.1 计数器

第一版只使用固定数量的 `uint64` 单调计数器：

| counter | 作用 | 可见范围 |
| --- | --- | --- |
| `local_done[7]` | 统计本 rank 每个 L 的 producer CTA 完成数 | agent |
| `partial_ready[7]` | 宣布本 rank 的 partial window 可被 peer 读取 | system |
| `service_arrive` | persistent CTA 在 rank 内做 phase barrier | agent |
| `reduced_ready[7]` | 宣布本 rank reduced payload/scale 可被 peer all-gather | system |
| `service_epoch` | 完成一整次 Stage2，为下一轮提供 expected epoch | agent |

计数器不在每轮 host `zero_()`。它们只单调增长，因此同一 CUDA/HIP Graph
可重复 replay，也不依赖 CPU 下发当前 epoch。

### 5.2 L 完成后发布 partial

每个参与 L(window) 的 CTA 完成写入后：

1. 执行 agent release。
2. 原子增加 `local_done[window]`。
3. 本轮最后一个 producer CTA 执行 system release。
4. 将 `partial_ready[window]` 增加到当前 invocation epoch。

persistent service 在 RS(window) 前由一个控制 CTA 等待所有 TP rank 的
`partial_ready[window] >= expected`，然后执行 system acquire，再放行本 rank 其他
service CTA。

### 5.3 RS 完成后发布 reduced shard

1. service CTA 并行执行 RS(window)。
2. 使用 `service_arrive` 完成 rank 内 grid phase barrier。
3. 控制 CTA 执行 system release，发布 `reduced_ready[window]`。
4. 控制 CTA 等待所有 peer 的 reduced epoch，执行 system acquire。
5. 放行 all-gather CTA 读取 peer reduced payload/scale 并写入完整 BF16 output。

### 5.4 不能依赖的假设

- 不能假设 G/L 一定比 RS/AG 慢。
- 不能依赖 HIP stream 的偶然调度顺序。
- 不能让 persistent CTA 占满所有 CU，否则 producer 无法发布 partial，会死锁。
- 不能在缺少 release/acquire 的情况下，只因 counter 数值正确就读取 payload。
- 不能让 Graph replay 依赖 host 重置 counter。

## 6. Workspace 策略

### 6.1 功能原型：每个逻辑窗口一份 workspace

第一个仓外原型使用七份 partial/reduced-payload/reduced-scale，不复用
2-slot ring。这样可以先
独立验证：

- persistent grid 是否能与 G/L 稳定并发；
- device epoch 和跨 rank 可见性是否正确；
- 减少 launch/barrier 后是否存在足够性能收益。

这会比当前 2-slot ring 多约 186 MiB/GPU 的 temporary workspace，所以只是原型，
不是 production 接入形态。原型脚本和 runner 放仓外。

### 6.2 production 候选：恢复 2-slot ring

只有七 workspace 原型已经证明稳定快于当前 KID，才增加两个 slot 的
backpressure：

```text
producer 写 slot s 前，等待 partial_consumed[s]
service 完成 RS(window) 后，发布 partial_consumed[s]
service 完成 AG(window) 后，reduced slot s 才可复用
```

不允许用“实测上 service 跟得上”代替 backpressure。如果 2-slot 的等待使 producer
占据剩余 CU 并影响性能，则比较 3-slot 和七窗口 workspace，最终仅固化一个
winner，不在 production 中保留 slot 参数。

## 7. Persistent grid 规则

persistent service 的 CTA 会在等待 partial 时保持 resident，因此 grid 不能直接照搬
当前短命 RS/AG kernel 的 `REDUCE_SCATTER_GRID=92` 并当作必然 winner。

约束：

- 先根据编译后的 VGPR/SGPR/LDS 确认每 CU residency。
- persistent grid 必须可以一次全部 resident，不允许 grid barrier 等待尚未调度的 CTA。
- 必须为 G/L producer 保留足够 CU，否则 persistent 只是用调度节点减少换取
  GEMM 降速。
- service CTA 使用 grid-stride loop 覆盖全部 RS/AG work，因此可以少于当前
  92/91 个 CTA。

仓外只测少量有意义的固定候选，例如 32、64、92 CTA。这些是离线 KID
候选，不是最终代码中的实验参数或运行时分支。

## 8. 文件边界

### 实验阶段

| 位置 | 作用 |
| --- | --- |
| 仓外 `comm_fused_moe_tests/` | candidate runner、A/B 脚本、grid/workspace 候选 |
| `kernels/comm_fused_moe/persistent.py` | 仅放 producer publication 和 persistent RS/AG KID |

实验阶段不修改 M=32768 的 production shape 映射。

### 候选胜出后

| 文件 | 最小改动 |
| --- | --- |
| `aiter/ops/flydsl/kernels/comm_fused_moe/persistent.py` | 保留唯一 winner 的固定常量和 kernel |
| `aiter/ops/flydsl/comm_fused_moe_host.py` | 新增/替换 M=32768 runner，初始化固定 stream/event/workspace |
| `aiter/ops/flydsl/kernels/comm_fused_moe/windowed.py` | 保留为 A/B baseline；不在其中增加 persistent 开关 |
| `aiter/ops/flydsl/kernels/comm_fused_moe/sync.py` | 原则上不改；persistent-only 同步逻辑留在新 KID 内 |

如果 persistent 最终替换 production M=32768，在完成整网验证前不删除
`windowed.py`。验证完成后再决定是否将旧 KID 移到备份分支，避免 production
同时维护两套 M=32768 路径。

### 明确不改

```text
aiter/fused_moe.py
aiter/ops/comm_fused_moe_runtime.py
aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py
ATOM model adapter
MORI repository
custom_all_reduce.py
```

如果实现中发现必须改上述文件，先停止并单独 review 原因，不把改动顺手
混入 persistent KID。

## 9. 实现顺序

### Step 0：冻结 baseline

- 完成当前 kernel 结构清理的 review 和 A/B。
- 记录 commit、容器、节点、JIT cache、uniform/skew 精度与七轮 Graph 数据。
- 保存当前 `windowed.py` 作为同进程 A/B baseline。

### Step 1：只做 persistent 同步骨架

- 启动固定 service grid，在 kernel 内循环七个窗口。
- 暂不执行 RS/AG，只测 device epoch、rank 内 grid phase barrier 和多 stream Graph capture。
- 连续 eager 1000 轮和 Graph replay 1000 轮，不允许 hang、counter 错位或前一轮泄漏。

### Step 2：让 G/L producer 直接发布 partial

- 生成独立 candidate producer compose，不给 production `windowed.py` 加 optional 开关。
- 保持现有 GEMM tile、compact route 和 local reducer 不变。
- 仅在 L 完成后增加 last-CTA publication。
- 比较生成 IR/HSACO，确认 GEMM 主体没有意外改变。

### Step 3：接入 RS/AG

- 直接复用 `collectives.py` 的 `emit_tp_reduce_scatter()` 和
  `emit_tp_all_gather()`。
- 先使用七份 logical-window workspace 验证正确性与性能上限。
- 确认 reduced-shard rank 与 peer rank 仍从同一份 MXFP8 reduced payload 解码为 BF16。

### Step 4：离线选择唯一 service grid

- 同一节点、同一进程交替运行 current windowed 和 persistent candidate。
- 只测少量可完全 resident 的 grid，记录 R、F、G/L 和完整 Graph。
- 删除 loser 配置，文件中只留一组编译期常量。

### Step 5：恢复可接入的 workspace

- 实现 2-slot backpressure，或用实测证明更多 slot 对性能/内存更合理。
- 重跑精度、长时间 Graph replay 和内存占用。
- 只保留最终 slot 布局，不把原型 workspace 分支留在 production。

### Step 6：host 最小接入

- runner 初始化时一次性创建 workspace、MORI external windows、service stream
  以及 start/done event。
- `__call__()` 只做固定参数组装、9 次 kernel launch/event 依赖和返回 output。
- 不为 current/persistent 加运行时布尔分支；仓外 A/B 用显式 runner 完成。

### Step 7：回归和整网

- M=32768 uniform/skew，eager/Graph，output/shared alias。
- M=2048 full-width 原路径。
- 普通 Opus/FlyDSL Stage2、MegaMoE 和其他已接入 shape。
- ATOM 整网 prefill/decode，记录端到端 latency/throughput 和显存。

## 10. 验收门槛

### 正确性

- uniform/skew 都通过。
- eager 和 Graph 输出一致。
- M=32768 精度与当前 KID 一致：`max_abs` 约 `0.8125`、`rel_l2`
  约 `0.03013`。
- local reduced shard 与 remote gathered shard 输出一致，shared-output alias 通过。
- eager 和 Graph 各连续 1000 轮无 hang、无偶发 stale payload、无 epoch 错位。

### 性能

- 在同一 TP8 进程中交替测量，使用 rank-max 七轮 median。
- M=32768 uniform 和 skew 都必须至少稳定快约 1%，即以当前数据为参考
  需达到约 `<= 2190 us`，否则不值得增加 production 复杂度。
- M=2048 保持当前 full-width 性能，不接受稳定回退。
- 不能用 RS/AG 局部变快掩盖 G/L 因 CU 被占用而变慢，最终只看完整 Stage2
  Graph 和整网。
- 整网没有 latency、throughput 或显存的不可接受回退。

### 代码

- production 中不保留 loser grid、workspace 方案、debug counter 和实验分支。
- 新 kernel 文件只包含 persistent candidate 必需的 producer publication、phase
  synchronization 和 RS/AG loop。
- host 不变成 planner，shape 到 runner 仍是一次字典查找。
- 所有仓内修改都在 review 和 A/B 后再单独 commit，不直接 amend。

## 11. 应立即停止的情况

出现下列任一情况，不继续向 production 堆叠保护逻辑：

- 多 stream HIP Graph 不能稳定 capture/replay。
- 服务 CTA 保留导致 G/L 回退大于 launch/barrier 节省。
- 只有占用过多 workspace 才能达到小于 1% 的不稳定收益。
- 需要修改公共 GEMM emitter、MORI 或 runtime seam 才能勉强运行。
- 长时间 replay 出现任何偶发 hang 或 stale data。

这时保留当前约 2.2 ms 的 `windowed.py`，删除 persistent 候选即可，不影响
已经完成的 production 接入。

## 12. 后续方向，不属于第一版

只有 persistent communication 已经稳定胜出后，才依次评估：

1. 将多个 G/L 窗口收缩成更少 producer launch。
2. 让 persistent RS/AG 与 MORI SDMA all-gather 作为新的完整 KID 单独 A/B。
3. 为新 model shape 离线生成固定 KID 记录，而不是把当前 kernel 改成动态万能
   kernel。
4. 只在 Opus/ASM 真正出现可替换 winner 时，增加显式 backend/KID 映射。

全 persistent G/L/RS/AG 仍是更后面的实验，因为它会让 service CTA 承担 GEMM 的
LDS/寄存器配额，还需要将当前编译期 N-tile window 改成 kernel 内工作队列。
这不符合第一版的轻量化和低回归风险目标。

# Comm-fused MoE M=1 单-kernel 10 us Goal

> 状态（2026-08-27）：性能与正式接入目标均已通过。production runner 在 7 轮 ×
> 100 次 CUDA Graph replay 中得到 `9.6809 us` rank-max median，范围
> `9.6737–9.7413 us`，7 轮全部低于 10 us。已验证 compose 已接入严格 shape-guarded
> small-M dispatch，不再继续扩展已失败的同步实验。

## 1. 最终目标

固定目标场景：

```text
GPU:        gfx950 / MI355X
TP:         8
M:          1
H:          7168
I:          384
experts:    384
topk:       6
Stage2:     FP8 activation × FP4 weight → BF16 atomic partial
collective: TP8 BF16 one-stage AllReduce
execution:  CUDA Graph replay
```

最终需要交付一个产品可接入的单-launch `GEMM2 + TP AllReduce` megakernel，并满足：

```text
rank-max median < 10.00 us
```

如果最终无法进入 10 us，不能只以“同步开销太大”或“硬件不适合”为结论。必须给出可重复的
时间线、吞吐、occupancy 和同步下界，明确哪一段硬件关键路径使 10 us 不可达。

本目标不以减少代码量为优先级，也不因为 ROCm 版本兼容性尚未完全确定而暂停 overlap。
ROCm 兼容性验证与 overlap 优化是两条并行证据链：前者决定 fast path 的 guard 范围，后者
决定最终性能。

## 2. 当前基线与已确认事实

### 2.1 性能基线

当前同一 TP8 测试口径下的关键结果：

| 路径 | M=1 rank-max median |
|---|---:|
| Stage2 only | 8.509 us |
| shared-seeded Stage2 + standalone FlyDSL AR | 约 15.17 us |
| 同轮 two-kernel 对照 | 16.121 us |
| 旧 tail-14 coarse megakernel | 18.600 us |
| 显式 system bridge coarse megakernel | 约 24.08 us |
| GWS coarse megakernel | 23.87 us |

`18.60 us` 版本仍然是后续 overlap 的母体。GWS 和重型 system bridge 结果用于解释同步成本，
不作为新实现起点。

### 2.2 旧版同步 ISA

旧版源码中的抽象顺序是：

```text
partial atomic writes
→ agent release
→ agent monotonic completion atomic
→ coordinator agent poll/acquire
→ system monotonic remote-ready store
→ peer agent monotonic poll
→ non-temporal remote partial loads
```

gfx950 最终 ISA 为：

```asm
buffer_atomic_pk_add_bf16 ...
s_waitcnt vmcnt(0)
buffer_wbl2 sc1
s_waitcnt vmcnt(0)
global_atomic_add ... sc0

global_load_dword ... sc1
s_waitcnt vmcnt(0)
buffer_inv sc1

global_store_dword ... sc0 sc1
global_load_dword ... sc1
buffer_load_dwordx4 ... nt
```

这说明当前 binary 中确实存在 L2 writeback、completion 顺序和 system-coherent ready store。
它不是一个完全没有硬件屏障的偶然成功路径。

### 2.3 generation-changing 验证

2026-08-27 在 `crsuse2-m2m-v2-024` 完成：

| generations | 每 rank payload | 有向 peer 路径 | 远端校验流量 | mismatch |
|---:|---:|---:|---:|---:|
| 50,000 | 14,336 B | 56 | 40.14 GB | 0 |
| 20,000 | 114,688 B | 56 | 128.45 GB | 0 |

每一代 payload 都依赖 generation、source rank 和 element index，不能用上一代陈旧值假通过。
这足以把旧同步路径定性为当前 gfx950/ROCm 环境下经过压力验证的 fast path，但还不能宣称为
跨架构、跨 ROCm 版本的通用内存模型保证。

## 3. 工作流 A：ROCm 版本兼容性矩阵

### 3.1 目标

回答两个不同问题：

1. 不同 ROCm LLVM 是否仍把同一 IR lowering 成相同的关键 ISA；
2. 不同 ROCm runtime/compiler 组合运行时是否仍通过 generation-changing litmus。

不能把“离线 ISA 相同”和“运行时完整兼容”混为同一结论。

### 3.2 工具链矩阵

当前容器：

```text
ROCm SDK: 7.14.0
HIP:      7.14.60850
LLVM:     23.0.0git
Torch:    2.13.0+rocm7.14.0
FlyDSL:   0.3.0
```

对照下载到仓库外缓存，不修改系统 ROCm：

| ROCm | LLVM | 用途 |
|---|---:|---|
| 7.0.2 | 20.0 | 较早 gfx950 release 工具链 |
| 7.1.0 | 20.0 | 中间版本 |
| 7.2.0 | 22.0 | 当前公开 release 工具链 |
| 7.14.0 | 23.0git | 当前容器基线 |

缓存目录：

```text
/home/yifehuan/data/rocm_toolchains/<version>/
```

只下载并解包官方 `rocm-llvm`，不通过 apt 安装、不覆盖 `/opt/rocm`，避免影响当前节点。

### 3.3 离线 ISA 对照

输入固定为旧版 exact kernel 的同一份 LLVM IR。每个版本使用自己的 `llc` 或 `clang -x ir`
生成 gfx950 ISA，随后检查：

```text
partial write 后是否仍有完整 wait
agent release 是否仍为 buffer_wbl2 sc1
writeback 后是否仍有 s_waitcnt vmcnt(0)
completion atomic 是否仍晚于 writeback
coordinator poll 是否仍为 sc1 load
agent acquire 是否仍生成 buffer_inv sc1
ready store 是否仍为 sc0 sc1
ready poll 与 nt payload load 是否仍保持控制流顺序
```

除关键指令窗口外，还需要记录：

- VGPR/SGPR/LDS；
- code size；
- wave occupancy；
- 分支和 waitcnt 数量；
- 是否出现额外 `buffer_wbl2 sc0 sc1` 或 `buffer_inv sc0 sc1`。

### 3.4 运行时验证

若旧版编译器生成的 code object 能被当前驱动/runtime 接受，则直接运行相同 TP8 litmus：

```text
50,000 generations × 14 KiB
20,000 generations × 112 KiB
```

若 code object ABI、FlyDSL MLIR ABI 或 runtime 不兼容，应明确记录为“组合无法运行”，不能
误报为同步失败。必要时使用对应 ROCm PyTorch 容器，但不替换当前工作容器。

版本矩阵的结论分为：

- `ISA-equivalent`：关键指令和顺序一致；
- `runtime-pass`：generation litmus 通过；
- `unsupported-combination`：工具链/driver/code-object ABI 不兼容；
- `semantic-regression`：能够运行但出现 stale generation 或关键 ISA 退化。

无论矩阵最终属于哪一种，工作流 B 都继续推进。

## 4. 工作流 B：以 18.60 us 为母体做真实 overlap

### 4.1 为什么 coarse megakernel 不可能达到目标

旧版时间线是：

```text
168 GEMM CTA 全部结束
→ tail-14 completion
→ 14 CTA one-stage AR
```

它只有单 launch，没有 GEMM/AR overlap。因此即使软件 barrier 免费，关键路径仍近似：

```text
Stage2 8.5 us + AR 7～9 us + rank skew/sync
```

单纯继续优化 counter、GWS 或 dispatch 不能把 18.6 us 压到 10 us。进入 10 us 必须让前部
输出通信与后部 GEMM 并行。

### 4.2 第一候选：N-major 四 band、tail CTA 接管 AR

H=7168、tile_n=256，共 28 个 N tile。第一版划为四个 band：

```text
band 0: tile  0.. 6
band 1: tile  7..13
band 2: tile 14..20
band 3: tile 21..27
```

每个 tile 有 6 个 routed expert CTA，因此每 band 有 42 个 producer CTA。

物理 CTA 顺序改成 N-major，但 GEMM 数学主体不改：

```text
physical_id = block_y * 28 + block_x
n_tile      = physical_id // 6
expert      = physical_id % 6
logical_id  = expert * 28 + n_tile

emit_gemm2(..., block_id=logical_id)
```

`mixed_moe_gemm_2stage_common.py` 已支持可选 `block_id`，所以第一版不需要修改 common 文件。
所有重排和同步逻辑放在独立 compose/probe 中。

每个 band 的流程：

```text
42 个 CTA 执行各自 GEMM tile
→ s_waitcnt + CTA barrier
→ agent release
→ band_counter[band].fetch_add(1)

最后 S 个到达的 CTA：
→ 等 band_counter == 42
→ worker 0 发布 rank_band_ready[band]
→ 收齐 8 rank ready
→ S 个 CTA 直接读取该 band 的 8-rank BF16 partial
→ 固定 rank 0..7 顺序 FP32 sum
→ 写最终 BF16 output
```

首测 `S=1`，因为一个 256-thread CTA 使用每线程 8 个 BF16 元素即可覆盖一个 1792-column
band。若单 CTA 的 xGMI 并发不足，再只测试 `S=2` 和 `S=4`，不做大范围参数扫。

该结构没有从 kernel 开头常驻轮询的 service CTA。service worker 是已经完成 GEMM 的尾部
producer CTA，因此不会复现早期 14 个 polling CTA 让 GEMM 增加约 3 us 的问题。

### 4.3 跨 replay 生命周期

band ready 只负责发布本 band 的 partial。四个 band 都完成后，最后完成的 band coordinator
执行一次全局 consumed barrier：

```text
band_service_done[band]
→ all_bands_done.fetch_add(1)
→ 最后一个 band coordinator 发布 rank_consumed
→ 等待 8 rank consumed
→ reset 四个 band counter / local gate
→ epoch++

不为每个 band 增加 consumed barrier。下一次 graph replay 只能在所有 rank 完成上一代 remote
read 后开始覆盖 partial。
```

### 4.4 第二候选：七 band 或 tile-ready

四 band 若 overlap 窗口太粗，再按单一变量推进：

1. `7 bands × 4 tiles`；
2. `14 bands × 2 tiles`；
3. 最后才是 `28 × 1 tile`。

每增加 band 数，都必须同时报告：

- 第一个 band ready 时间；
- 最后一个 GEMM tile 完成时间；
- 每 band rank skew；
- ready store/poll 成本；
- remote reduction 持续时间；
- 总 kernel 延迟。

如果 4→7 band 没有降低关键路径，不进入 14/28 band，避免再次得到几十微秒的同步爆炸版本。

## 5. 10 us 性能模型

目标不是假设 AR 完全免费，而是满足：

```text
T_total = max(T_gemm_tail, T_first_ready + T_pipelined_AR)
          + T_final_consumed
        < 10 us
```

以 Stage2 `8.509 us` 为基准，允许的非重叠尾部只有约 `1.49 us`。因此至少需要：

- 第一 band 在约 2～3 us 内 ready；
- 前三个 band 的通信大部分隐藏在 GEMM 尾部；
- 最后一个 band 的归约加最终 consumed 控制在约 1～1.5 us；
- N-major 重排本身不能让 GEMM 增加超过约 0.3 us。

如果上述任一条件明显不成立，10 us 就不可能通过微调达到。

## 6. 每轮实验必须回答的问题

### 6.1 正确性

- uniform、skew；
- generation-changing 输入；
- 所有 8 rank 结果一致；
- 10,000 次 graph replay 无 stale、串轮、hang；
- 与 two-kernel 固定 rank 顺序结果满足现有误差阈值。

### 6.2 时间线

用 `s_memrealtime` 或等价低扰动 timestamp 记录：

```text
kernel start
每个 band 的首个/最后一个 GEMM completion
每个 band 的 local-ready
每个 band 的 all-rank-ready
每个 band 的 AR start/end
最后 consumed start/end
kernel end
```

时间戳 probe 与正式性能版本分离；正式版本不能因为 instrumentation 回退而被误判。

### 6.3 硬件计数器

只对最终两个候选采集一次 PMC/ATT，至少回答：

- GEMM 的 MFMA active 是否因 direct remote load 下降；
- service 期间 VMEM/XGMI latency；
- L2 hit/miss、remote read 带宽；
- VGPR/LDS 是否降低 occupancy；
- 是否存在 service CTA 占住 CU、后续 producer 无法调度；
- 最后一 band 是计算尾巴、rank skew、同步还是 xGMI 吞吐受限。

## 7. 失败时必须给出的硬件理由

只有满足以下证据要求，才能结论为“10 us 在当前硬件上不可达”：

1. N-major Stage2 本身已调到不高于原 Stage2 `+0.3 us`；
2. 至少测试 4-band `S=1`，必要时单独测试 `S=2/4`；
3. 时间戳证明剩余尾部来自哪一段；
4. standalone 对应 band AR 给出最小耗时；
5. PMC/ATT 区分以下三类瓶颈：
   - xGMI/remote-load latency；
   - GEMM 与通信争抢 L2/HBM；
   - CU occupancy/scheduler interference；
6. 给出测得的下界：

```text
lower_bound = max(measured GEMM critical path,
                  earliest-ready + unavoidable final-band AR)
              + minimum final cross-rank completion
```

如果该下界仍小于 10 us，则不能停止，必须继续优化实现。

## 8. 实验纪律与代码落点

- ROCm 工具链和原始日志放在 `/home/yifehuan/data/`，不提交二进制或大量 dump。
- overlap 候选先作为仓外 probe，避免再次扩大 production diff。
- `mixed_moe_gemm_2stage_common.py` 第一版不修改；使用已有 `block_id` 参数完成 N-major remap。
- 每个确认有收益的阶段单独提交，便于 A/B 和回退。
- 失败候选在记录结果后删除，只保留结论、日志路径和最小复现。
- 未经明确要求不 push。

## 9. 执行顺序

1. 下载并解包 ROCm 7.0.2、7.1、7.2 的 `rocm-llvm`。
2. 用四个 LLVM 版本生成旧 kernel ISA，完成关键指令矩阵。
3. 在兼容组合上复跑 generation litmus；不兼容组合记录 ABI 原因。
4. 复现 `18.60 us` 母体，确认当前节点同轮基线。
5. 实现 N-major 但不 overlap 的控制组，隔离重排成本。
6. 实现 4-band、`S=1` tail-takeover overlap。
7. 只有证据表明 xGMI 并发不足时测试 `S=2/4`。
8. 通过后做 7-band；未通过则先分析时间线，不盲目增加 phase。
9. 最终版本执行正确性长测、正式 A/B 和一次 PMC/ATT。
10. 达到 `<10 us` 后接入 production；否则提交硬件下界报告。

## 10. 最终实测结果

### 10.1 通过门槛的固定实现

最终实现没有采用常驻 service CTA、全 grid barrier 或 GWS。它使用每个 N tile 的最后一个
GEMM CTA 直接接管 collective：

```text
84 个 compute CTA = 14 N tiles × 6 routed experts

每个 tile：
6 个 GEMM CTA 写 direct-row route
→ 最后到达 CTA 在 LDS/寄存器中做 local reduce
→ 写本 rank ping-pong BF16 partial
→ system-release fence
→ TP8 per-tile ready
→ 7 路 xGMI load + 1 路 LDS local partial
→ balanced FP32 tree sum
→ 写最终 BF16 output
```

固定参数：

| 项目 | 最终值 |
|---|---|
| GEMM tile | `M16 × N512 × K128` |
| grid 顺序 | expert-major |
| Stage2 epilogue | M=1 direct-row |
| compute CTA | 84 |
| collective 粒度 | 14 个 512-column tile |
| partial | 双缓冲 BF16 |
| local reduction | balanced FP32 tree |
| remote reduction | balanced FP32 tree |
| 本 rank partial | LDS 复用，不回读 global |
| local load cache modifier | 0 |
| remote xGMI load cache modifier | 1 |
| 可见性 | explicit system-release + system-scope ready store |

低层固定 probe 的权威稳定日志：

```text
/home/yifehuan/data/box_comm_fused_moe_sdma/out/
1209_m1_sub10_minimal_stable_20260827.log
```

7 轮 × 100 replay 的 rank-max 结果：

| 路径 | median | min | max |
|---|---:|---:|---:|
| route Stage2 only | 5.3445 us | 5.1561 us | 5.8269 us |
| atomic Stage2 + partial init | 9.6757 us | 9.6609 us | 9.7637 us |
| two-kernel Stage2 + AR | 17.3294 us | 17.2274 us | 17.4658 us |
| single-launch tile-ready megakernel | **9.6249 us** | **9.5925 us** | **9.6773 us** |

精度：

```text
max_abs = 0.03125
rel_l2  = 0.00313977
```

因此 exit criterion 已经通过：

```text
9.6249 us < 10.00 us
```

相对同轮 two-kernel 路径减少 `7.7045 us`，约 `44.5%`；同时单 launch 已略快于
同轮 atomic Stage2 + partial 初始化口径。

正式 production runner 的权威日志：

```text
/home/yifehuan/data/box_comm_fused_moe_sdma/out/
1212_m1_production_symmetric_routes_20260827.log
```

| production 路径 | median | min | max |
|---|---:|---:|---:|
| ordinary Stage2 + TP AllReduce | 17.5038 us | 17.4690 us | 17.8406 us |
| `_SmallMRunner` single launch | **9.6809 us** | **9.6737 us** | **9.7413 us** |

production speedup 为 `1.808×`，并且 7 个 rank-max 样本全部低于 10 us。精度为
`max_abs=0.03125`、`rel_l2=0.002816`。

最终 graph replay 长稳日志：

```text
/home/yifehuan/data/box_comm_fused_moe_sdma/out/
1213_m1_production_10k_uniform_skew_20260827.log
```

| route | graph replays | fused rank-max | max_abs | rel_l2 |
|---|---:|---:|---:|---:|
| uniform | 10,000 | 9.4853 us | 0.03125 | 0.002823 |
| skew | 10,000 | 9.4570 us | 0.03125 | 0.002791 |

两种 route 均正常退出，没有 hang、stale generation 或跨 replay epoch 污染。

### 10.2 从 18.60 us 到 9.62 us 的关键变化

1. 删除 kernel 起点常驻轮询的 service CTA，避免其让真实 GEMM 增加约 3 us。
2. 从 coarse 全局完成改为 per-N-tile 最后 CTA 接管，使早期 tile 的 AR 与后续 GEMM 真正重叠。
3. `tile_m=16` + single-row direct epilogue：M=1 只写 row0，跳过 16 行 C-shuffle/LDS
   materialization。
4. `tile_n=512`：CTA 从 168 减到 84，collective handoff 从 28 次减到 14 次；1024
   因单 CTA 过宽/VGPR 压力回退。
5. local 和 remote 均改为 balanced FP32 reduction tree，缩短 dependency chain。
6. local BF16 partial 同时留在 LDS，TP8 reduce 时只发起七路 remote global load。
7. cache policy 固定为 local `0`、remote `1`。remote cache modifier 短测为：

| remote modifier | rank-max median |
|---:|---:|
| 0 | 10.040 us |
| 1 | **9.972 us** |
| 2 | 10.841 us |
| 3 | 10.780 us |

稳定长测在相同 `0/1` 策略下进一步得到 9.6249 us。

### 10.3 ROCm 工具链证据

ROCm 7.0.2、7.1.0、7.2.0 的 `rocm-llvm` 已下载并在仓库外解包：

```text
/home/yifehuan/data/rocm_toolchains/7.0.2/
/home/yifehuan/data/rocm_toolchains/7.1/
/home/yifehuan/data/rocm_toolchains/7.2/
```

三版均成功从同一兼容 LLVM IR 生成 gfx950 ISA，关键同步窗口一致包含：

```text
s_waitcnt vmcnt(0)
buffer_wbl2 sc1
global_atomic_add ... sc0
buffer_inv sc1
global_store_dword ... sc0 sc1
buffer_load_dwordx4 ... nt
```

对应汇编保存在各版本的 `old_tail14/kernel.s`。这证明旧 fast path 不是当前 LLVM 23
单版本偶然 lowering。运行时没有强行混装旧 ROCm runtime/PyTorch；跨版本结论限定为
`ISA-equivalent`，而当前 ROCm 7.14 的 runtime 正确性由 50k/20k generation litmus 证明。

### 10.4 已淘汰方向

以下方向已经实测回退，不再重复：

- rank-major 或 64/128 B padded ready；
- remote-poll ready、global shared ready gate、grouped/coalesced ready；
- N-major CTA 重排；
- peer-pair + shuffle reduction；
- wave-only service、删除必要 block barrier；
- `tile_n=1024`、不满足映射约束的 448/896；
- K-split 两轮 collective；
- system-coherent partial store；
- 常驻 service CTA、软件 full-grid barrier、cooperative/GWS coarse megakernel。

失败与重复 probe 已从工作树删除；最终可复现入口使用正式 benchmark：

```text
COMM_FUSED_M=1 COMM_FUSED_ROUTE=uniform \
COMM_FUSED_PERF_ROUNDS=7 COMM_FUSED_PERF_ITERS=100 \
torchrun --standalone --nproc-per-node=8 \
op_tests/multigpu_tests/test_flydsl_comm_fused_full_tp8_perf.py
```

## 11. 正式接入门槛

正式 dispatch 已通过 `family=small` 的 M=1 配置接入，并保持以下不变量：

- 仅匹配 `gfx950 / TP8 / M=1 / H7168 / I384 / E384 / topk6`；
- symmetric partial/state 的注册与生命周期覆盖 CUDA Graph replay；
- 当前 runner 与其他 comm-fused runner 一样要求调用串行化；若上层未来并发使用多个
  stream/context，必须为每个并发实例提供独立 runner/state，不能共享 epoch/counter；
- 保留 explicit system-release，不使用只在当前 ISA 上观察到可工作的 legacy ready；
- 未匹配形状继续回退到现有 atomic + one-stage AR；
- 正式入口的 7×100 A/B 必须再次满足 `<10 us` 且精度阈值不回退。

## 12. ROCm 7.2.4 新环境的 Graph replay 下限

2026-08-28 在同一台 MI355X 节点上完成了 code object、MORI、PyTorch 与
HIP/HSA runtime 的交叉实验。结果表明，新镜像中单算子 Graph rank-max 从约
`9.4 us` 回退到约 `13.3 us`，主要不是 kernel codegen 回退，而是
`hipGraphLaunch` 提交吞吐下降。

关键证据如下：

| 对照 | M=1 fused Graph rank-max |
|---|---:|
| LLVM23 code object + ROCm 7.14 runtime | 约 9.42 us |
| 同一 LLVM23 code object + ROCm 7.2.4 runtime | 约 13.25 us |
| ROCm 7.2.4 runtime + 旧 MORI 20260810 | 约 16.98 us |
| ROCm 7.14 runtime + 新 MORI 20260826 | 约 9.47 us |

因此 MORI 不是根因；新 MORI 在 7.2.4 上反而比旧 MORI 更好。单 GPU 的最小
Graph replay 测试也给出了相同结论：

| 用户态 runtime | 单节点 Graph replay 提交时间 |
|---|---:|
| ROCm 7.2.4 | 约 9.3--10.0 us |
| ROCm 7.14 | 约 4.6--5.6 us |

PyTorch 2.10 保持不变、仅交叉加载 ROCm 7.14 HIP/HSA 后，最小 Graph replay
也恢复到约 `5.5 us`，因此差异位于 HIP/HSA 用户态 runtime，而不是 PyTorch
`CUDAGraph.replay()` 的 Python 封装。

设备 trace 进一步显示：在 8 个 rank 同步起跑的样本中，ROCm 7.2.4 下同一
fused kernel 的设备执行时间仍约为 `8.7--10.2 us`。较长样本表现为先启动的
rank 在 kernel 内等待晚启动的 peer，而不是 GEMM/AllReduce 指令本身变慢。

为消除每次只 replay 一个极短 graph 时的 host/runtime 吞吐上限，在同一 graph
中捕获 16 次调用并均摊后，当前 LLVM24/JIT kernel 得到：

| M | fused | ordinary |
|---:|---:|---:|
| 1 | **8.72 us** | 14.44 us |
| 2 | **10.58 us** | 16.50 us |

这说明当前 kernel 本体仍优于原来的 `M=1 9.6 us / M=2 12.7 us` 门槛。整网
CUDA Graph 一次 replay 包含完整模型，`hipGraphLaunch` 固定成本只支付一次，
不应把单节点 graph 的提交下限重复计入每个 MoE kernel。

已排除的本地绕过方式：

- `enable-post-misched=false`、expert scheduling、LSR、delay-ALU、kernarg preload
  等编译开关没有改善 ISA 或性能；
- `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0` 虽降低单 GPU 极小 graph 的提交时间，
  但真实 TP8 M=1 回退到约 `19.5 us`，不可使用；
- `AMD_DIRECT_DISPATCH=0`、force async queue、graph batch size、kernarg copy 等
  runtime 开关均无有效收益；
- ROCm 7.14 HIP 与 ROCm 7.2.4 HSA 不能混装，缺少
  `hsa_amd_vmem_export_fabric_handle`；整套旧 HIP/HSA 与当前 PyTorch 的
  symmetric-memory allocator 也不兼容，不能作为产品方案。

ROCm 7.2.4 的 CLR 分支尚未包含后续 graph fast-dispatch 优化，包括 flat AQL
packet buffer (`bdbc555`)、instantiate-time dependency 预计算 (`b203915`) 和
Ext dispatch sync-plan 优化 (`8750dbd`)。
若产品要求恢复单 graph replay 的绝对延迟，应升级匹配的 ROCm/PyTorch 用户态栈，
或在发行版 runtime 中回移对应 CLR 修复；继续修改 small-M kernel 编译参数不能解决
这个 runtime floor。

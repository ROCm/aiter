# Comm-fused MoE persistent kernel：最终方案与门禁

## 1. 目标

为 DeepSeek-V4-Pro 的 TP8 Stage2 提供轻量、可维护的通算融合路径：

- M=2048 保留已验证的 full-width kernel。
- M=32768 使用 windowed producer + persistent collective service。
- 未支持的 token bucket 在模型层选择普通 MoE，不在 AITer runtime 内回退。
- M=512～32768 的既有路径不因本实现回退。
- 只保留 production winner，不保留实验开关、备用 runner 或失败分支。

当前固定模型 shape：

```text
H=7168, I=384, E=384, TOPK=6, TP=8
```

shape 与 token bucket 是否支持，由 runtime 的 runner 表决定；这不是“任意 shape
都能正确且高效”的假泛化。新 shape 应增加独立配置并通过完整门禁后再映射。

## 2. 最终路径

### 2.1 M=2048：full-width

执行顺序：

```text
Stage2 GEMM
→ local top-k/shared reduce
→ TP epoch barrier
→ TP reduce-scatter + owner MXFP8 publication
→ TP epoch barrier
→ TP all-gather + BF16 output
```

这条路径已经适合小 M。persistent 方案不替换它，避免额外 stream、event、状态轮询
和大 workspace 成本。

### 2.2 M=32768：persistent window pipeline

H=7168 按 1024 列切成 7 个 phase。producer stream 负责窗口 GEMM 和 local reduce；
service stream 上常驻一组 CTA，依次完成每个 phase 的 TP reduce-scatter 和
all-gather。

逻辑流水：

```text
producer: G0 → (G1+L0) → ... → (G6+L5) → L6 → publish6
service:             RS0+AG0 → RS1+AG1 → ... → RS6+AG6
```

device kernel launch 共 10 次：

```text
1 × G0
6 × producer cycle
1 × L6 drain
1 × final 1-CTA publication
1 × persistent service
```

service 在 phase 0 producer 完成后启动，随后通过 device-side epoch 协议等待后续
phase，不再为每个窗口发起 host barrier、RS 和 AG kernel。

## 3. 为什么采用 phase-private workspace

每个 phase 使用独立的 partial、reduced payload 和 scale 区域：

```text
7 × partial
7 × reduced MXFP8 payload
7 × reduced E8M0 scale
```

相对 2-slot ring 约多占 186 MiB/GPU，但这是当前实测 winner：

- producer 不需要等待 slot 回收，不会让持有 GEMM LDS 的 CTA 自旋。
- service CTA 可以稳定驻留，不形成 producer/service 互相等待。
- 同步关系只沿 phase 单向推进，容易验证和复用。

2-slot、3-slot ring 已实测约 2655～3313 us，并出现驻留死锁风险，已淘汰。生产
代码中不保留 ring 参数或分支。

## 4. Device-side 同步协议

状态区是 symmetric memory，并注册为 MORI external window，使所有 rank 可以用固定
flat-VA 直接访问 peer 状态和通信 payload。

每次调用由递增 epoch 区分，不需要每轮清零大 workspace。核心状态为：

- `SERVICE_EPOCH`：本次 service invocation。
- `WORKER_EPOCH[service_cta]`：每个 service CTA 的调用进度。
- `PARTIAL_READY[phase]`：本 rank 的 local partial 已发布。
- `PHASE_DONE[phase]`：本 rank 完成 RS 的 service CTA 数。
- `REDUCED_READY[phase]`：本 rank 的 reduced shard 已发布。
- `PHASE_GATE[phase]`：rank 内所有 service CTA 的 phase gate。

每个 phase 的顺序：

```text
producer release + PARTIAL_READY
→ rank 0 service CTA 等待所有 TP rank
→ rank 内 partial gate
→ 所有 service CTA 执行 TP reduce-scatter
→ 最后完成的 CTA release + REDUCED_READY
→ rank 0 service CTA 等待所有 TP rank
→ rank 内 reduced gate
→ 所有 service CTA 执行 TP all-gather
```

轮询使用 `s_sleep 1`，避免等待 wave 持续占满指令发射。system scope 用于跨 GPU
可见性，agent scope 用于同一 GPU 的 service CTA 协调。

## 5. 文件职责

### AITer

- `aiter/ops/comm_fused_moe_runtime.py`
  - 持有 token bucket → runner 映射。
  - `supports(tokens)` 只查询能力，不执行 fallback。

- `aiter/ops/flydsl/comm_fused_moe_host.py`
  - 创建 symmetric workspace、MORI window、stream 和 event。
  - M=256～4096 映射 `_FullWidthRunner`。
  - M=8192/16384 映射 `_WindowedRunner`。
  - M=32768 映射 `_PersistentWindowRunner`。
  - runner 首次命中时才分配对应 workspace，避免一次常驻全部 bucket。
  - 只负责编排已编译 kernel，不实现实验调度策略。

- `aiter/ops/flydsl/kernels/comm_fused_moe/full_width.py`
  - full-width producer、local reduce、TP reduce-scatter/all-gather。
  - 只保存 M=256～4096 的准确静态配置。

- `aiter/ops/flydsl/kernels/comm_fused_moe/windowed.py`
  - 纯 1024-column window pipeline。
  - M=8192/16384 的 G/L/RS/AG overlap 和 drain。

- `aiter/ops/flydsl/kernels/comm_fused_moe/persistent_window.py`
  - M=32768 的 window producer、local reduce 和 phase publication。
  - 单 persistent TP reduce-scatter/all-gather service。
  - 跨 rank 和 rank 内 epoch 协议。

- `aiter/ops/flydsl/kernels/comm_fused_moe/collectives.py`
  - 公共 MXFP8 TP reduce-scatter/all-gather primitive。
  - 本轮不引入 persistent 特例。

### ATOM

- `atom/model_ops/fused_moe/comm_fused_moe.py`
  - 暴露 `supports_comm_fused(tokens)` 和 fused forward。

- `atom/models/deepseek_v4.py`
  - 模型调用层按 token bucket 选择 comm-fused 或普通 MoE。
  - prefill M=32768 走 comm-fused，decode M=1 走普通路径。

- `atom/plugin/vllm/moe.py`
  - lazy wrapper 必须保留子类实例，避免 `CommFusedMoe` 被构造成基类。

## 6. 已修复的接入问题

### 6.1 Windowed GEMM grid 越界

full-width 的 `sort_block_m=64, tile_m=32`，每个 sorted expert block 对应 2 个
GEMM M tile；windowed 的 `tile_m=64`，只对应 1 个。host 现在统一计算：

```python
sorted_expert_blocks * SORT_BLOCK_M // TILE_M
```

不能再写死 `* 2`。旧写法会让 M=32768 多 launch 一倍 block，并在整网中产生越界。

### 6.2 Decode 不应进入 fused runtime

AITer runtime 不做普通 MoE fallback。ATOM 在模型层调用 `supports_comm_fused`，未映射
bucket 直接走原普通 MoE。这保证 decode M=1 不会触发 `KeyError`，也不会把普通路径
逻辑重新塞进 fused host。

### 6.3 Fused 方法不能覆盖普通 forward

comm-fused 三参数实现使用 `forward_comm_fused_impl`；普通两参数 `forward_impl` 继续
继承基类，保证同一个模块可以在 prefill/decode 间选择路径。

## 7. 已验证结果

### 7.1 最终单算子 TP8 gate

清理 loser 后的交替 graph A/B：

| M | route | ordinary | comm-fused | speedup | max_abs | rel_l2 |
|---:|:---|---:|---:|---:|---:|---:|
| 2048 | uniform | 286.090 us | 256.953 us | 1.1134x | 0.7500 | 0.029969 |
| 2048 | skew | 299.097 us | 262.957 us | 1.1374x | 0.7500 | 0.029988 |
| 32768 | uniform | 3569.349 us | 1953.705 us | 1.8270x | 0.8125 | 0.030132 |
| 32768 | skew | 3580.094 us | 1950.825 us | 1.8352x | 0.8125 | 0.030131 |

M=512/1024/4096/8192/16384 没有 runner 映射，模型层继续走原普通 MoE，因此
persistent 接入不会让这些 bucket 回退。旧离线 probe 也证明 M=512 强行走 full-width
graph 会慢约 2.6%～4.3%，所以不应为了“覆盖更多 shape”错误放开。

同一 persistent epoch 协议已完成 uniform、skew 各 10000 次 graph replay，无 hang、
fault 或串轮，结果保持 bitwise 一致。

### 7.2 DeepSeek-V4-Pro 整网

TP8、M=32768 prefill、output=1、concurrency=1、6 requests + 2 warmups：

```text
standard median TTFT     1162.724 ms
comm_fused median TTFT   1059.366 ms
median speedup           1.097566x
TTFT reduction           8.889%
completed                6/6
```

结果：

```text
/home/yifehuan/data/comm_fused_moe_full_model_m32768_ab_20260820
```

该测试同时覆盖 M=32768 persistent prefill、M=1 普通 decode、无 memory fault、无
unsupported bucket 错误。

## 8. 最终门禁状态

- Python AST、`git diff --check`、旧 windowed 符号搜索：通过。
- ATOM adapter：7/7 通过。
- AITer runtime：3/3 通过。
- M=2048 uniform/skew：通过。
- M=32768 uniform/skew：通过。
- persistent graph replay 10000 × uniform + 10000 × skew：通过。
- DeepSeek-V4-Pro 整网 prefill/decode A/B：通过。
- 仓内无临时测试、gpucore 或 production loser。

精度门槛：

```text
M=32768: max_abs <= 1.0, rel_l2 <= 0.05
其他 shape: 不差于各自清理前 production baseline
```

性能判断使用同节点、同容器、同输入、交替 A/B。单个 primitive 变快但完整 Stage2
变慢时，按完整 Stage2 结果淘汰。

## 9. 新 shape 的接入流程

1. 从模型真实调用采集 `(M bucket, H, I, E, TOPK, TP)`。
2. 先复用 collectives 和 host 生命周期，不复制 runtime/fallback。
3. 为该 shape 选择 full-width 或 windowed producer，并离线搜索少量结构参数：
   `WINDOW`、tile、local worker 数、service grid。
4. 用完整 Stage2 而非单 primitive 选 winner。
5. 通过 uniform、skew、spectrum、graph replay、整网门禁。
6. 只把 winner 加入 runner 映射；删除 probe 和 loser。

后续 Opus 或 ASM kernel 也遵循同一边界：模型层做能力选择，runtime 只查表，host 管
资源与调度，kernel 文件实现 producer/service。不要为了预留后端重新引入 Protocol、
多重开关或 runtime fallback。

## 10. Persistent 后续优化路线

本节记录当前实现完成后的后续优化方向。它们都是离线实验计划，不改变现有
M=32768 production winner；只有通过完整精度、uniform/skew A/B 和整网门禁的最终
winner 才能写入 runner 映射。

### 10.1 当前性能边界

M=16384 的 persistent 不是明显慢于 legacy window，而是两者处于小于 1% 的等价区：

| route | legacy window | persistent | 差异 |
|:---|---:|---:|:---|
| uniform | 1076.18 us | 1085.21 us | window 快 0.84% |
| skew | 1110.39 us | 1104.13 us | persistent 快 0.56% |

M=32768 时 persistent 才形成稳定优势：

| route | legacy window | persistent | persistent 降时 |
|:---|---:|---:|---:|
| uniform | 2068.22 us | 1995.00 us | 3.54% |
| skew | 2067.86 us | 2006.74 us | 2.96% |

M=16384 的短 sweep window 结果与上述正式 persistent A/B 不属于同一轮测试，不能
直接拿单个最小值宣称 window 稳定胜出。

### 10.2 为什么当前参数偏向 M=32768

当前固定参数为：

```text
WINDOW=1024
PHASES=7
LOCAL_WORKERS=2048
SERVICE_GRID=77
```

M 从 32768 减半到 16384 时，GEMM、local reduce、RS 和 AG 的有效工作量减半，但以下
固定成本不变：

- 7 个 phase；
- 每个 phase 的 partial/reduced gate；
- service block barrier 和 completion atomic；
- producer launch 数；
- service stream/event；
- 77 个常驻 service CTA。

当前 MI355X 有 256 CU。77 个 service CTA 在整段流水中保持存活，可能长期占用约 30%
的 CTA/CU 驻留位置；实际共驻留取决于 producer 的寄存器和 LDS。M=16384 每个 phase
更短，service 等待和同步占比更高，常驻 CTA 与 GEMM 的资源竞争也更难摊薄。

另外，当前只有 RS/AG service 是 persistent。producer 仍是：

```text
1 x G0
6 x phase-specific producer cycle
1 x drain
1 x final publish
```

因此这不是单一的全 persistent Stage2 kernel。

### 10.3 优化优先级

| 优先级 | 优化方向 | 目的 | 改动风险 |
|---:|---|---|---|
| 1 | M=16384 单独扫 `SERVICE_GRID=32/48/64/77` | 用更少常驻 CTA 换回 GEMM CU，寻找完整 graph 平衡点 | 低 |
| 2 | RS/AG 跨 phase 流水 | 让 `AG(p)` 与 `RS(p+1)` 重叠，减少 service 串行尾部 | 中高 |
| 3 | 减少 worker0 gate 串行化 | 避免所有 service CTA 都由 worker0 单点发布 phase gate | 中 |
| 4 | 减少每 phase barrier/atomic | 降低中等 M 的固定同步占比 | 中 |
| 5 | 合并 drain/final-publish | 去掉一个 producer kernel launch | 中；需要可靠的全 grid 完成协议 |
| 6 | producer 也改为真正 persistent | 将 7 个 phase-specific producer 合成常驻 producer | 高 |

### 10.4 第一优先级：按 M 调 service grid

已有 grid 数据只覆盖 M=32768，并证明 77 优于更大的 105、119、126；它没有证明 77
也是 M=16384 的 winner。第一步只做仓外短 sweep：

```text
M=16384
WINDOW=1024
TM/TN/TK=64/256/128
SERVICE_GRID=32/48/64/77
route=uniform/skew
```

判断必须使用完整 Stage2 graph，不能用单独 RS/AG primitive。若更小 grid 只让通信
primitive 变慢，但完整 graph 因 GEMM 获得更多 CU 而变快，应选择完整 graph winner。

筛选后只对前两名执行 9 轮交替 A/B。差距小于 1% 时视为性能等价，优先保留现有
实现，不为噪声增加新的 production 配置。

### 10.5 第二优先级：RS/AG 跨 phase 流水

当前 persistent service 对每个 phase 严格串行执行：

```text
wait partial[p]
→ RS[p]
→ wait reduced[p]
→ AG[p]
→ phase p+1
```

后续候选使用双缓冲，将 service CTA 分成 RS 和 AG 两类角色：

```text
RS[0]
RS[1] + AG[0]
RS[2] + AG[1]
...
AG[6]
```

该方案只有在 trace 证明 service 串行尾部是主要瓶颈时才实施。必须同时检查：

- RS/AG 对 xGMI 和 HBM 的竞争是否抵消 overlap；
- CTA 分组后单阶段带宽是否下降；
- phase-private payload/scale 的可见性与复用；
- output 不同列窗口的并发写入是否保持无冲突；
- uniform/skew 10000 次 graph replay 是否无 hang、串轮和旧 epoch 泄漏。

### 10.6 Gate、barrier 和 producer 优化

当前每个 phase 由 worker0 等待八个 peer 状态，再向其余 service CTA 发布 rank 内
gate。若 trace 显示 worker0 gate 是明显气泡，可以让前八个 service worker 分别等待
一个 peer，并通过 rank 内 completion counter 发布 gate，避免单 CTA 串行等待。

drain 与 final-publish 不能只靠删除一次 launch 合并。final publish 必须发生在全部
local worker 完成后；若放进 drain，需要 last-block counter 或 cooperative grid 完成
协议。增加 2048 次 atomic 后未必比当前 1-CTA launch 更快，必须完整 A/B。

全 persistent producer 是最后一层实验。它需要解决 GEMM work-item 调度、phase 间
全 grid 完成、local publication 和 service 共驻留，代码和死锁风险都显著高于当前
方案。CUDA Graph 已降低 phase launch 的 host 成本，因此只有 trace 证明 producer
launch/尾部是剩余主瓶颈时才值得实施。

### 10.7 保留和淘汰规则

- 只测试精度已通过的 `TN=256, TK=128` 路径。
- TK256 当前少算 `I=384` 最后 128 个 K，其耗时不是优化候选。
- W3584 当前 local reduce 只覆盖 2048 列，其耗时也不是优化候选。
- 所有实验参数留在仓外 harness；生产代码不保留环境变量、候选分支或在线搜索。
- M=16384 如果没有稳定超过 window，继续走 window 即为合理静态映射。
- M=32768 的新方案必须同时超过当前约 1.95～2.00 ms persistent baseline，且整网 TTFT
  不回退，才能替换 production winner。

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
  - M=2048 映射 `_FullWidthRunner`。
  - M=32768 映射 `_PersistentRunner`。
  - 只负责编排已编译 kernel，不实现实验调度策略。

- `aiter/ops/flydsl/kernels/comm_fused_moe/windowed.py`
  - 1024-column Stage2 GEMM producer。
  - local top-k/shared reduce。
  - producer phase publication。

- `aiter/ops/flydsl/kernels/comm_fused_moe/persistent.py`
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

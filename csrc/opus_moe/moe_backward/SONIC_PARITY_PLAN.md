# gfx950 SonicMoE backward 对齐与 1000 TFLOP/s 优化计划

## 1. 目标、判定口径和结论

目标算例固定为：

```text
T = 32768, H = 2048, I = 1024, E = 64, topk = 8
M = T * topk = 262144 routes
dtype = BF16
activation = standard SwiGLU: silu(gate) * up
external W1/dW1 layout = concat [all gate rows; all up rows]
saved P/dP layout = packed pairs [g0,u0,g1,u1,...]
top-k scores = FP32
bias = disabled for the primary performance target
```

本文的最终目标是让 gfx950 上的**完整 MoE backward**超过 1000 TFLOP/s，且功能范围
与 SonicMoE `benchmarks/moe-cute.py` 的训练路径一致：包含 expert backward、top-k
score backward、`router_w` backward 和两条 `dx` 分支的合并。

SwiGLU expert backward 的计数口径为：

```text
F_bwd = 12 * T * H * I * topk
      = 6,597,069,766,656 FLOP
      = 6.597069766656 TFLOP
```

因此严格超过 1000 TFLOP/s 要求：

```text
full backward time < 6.597070 ms
```

Sonic B300 的 `6.604 ms, 999.0 TFLOP/s` 与这条线只有约 0.1% 的差别，可以视为
参考目标，但 gfx950 的最终验收仍按 `< 6.597070 ms`，不能把 6.604 ms 四舍五入成
“已经超过 1000”。

当前公平基线为：

| backend | expert-only backward | full backward | full 有效 TFLOP/s |
|---|---:|---:|---:|
| Opus gfx950 | 15.203 ms | 15.327 ms | 430.4 |
| Triton gfx950 | 14.234 ms | 14.752 ms | 447.2 |

从 Opus full `15.327 ms` 到门槛至少需要 `2.323x` 加速。按 gfx950 约 2.3 PFLOP/s
的 BF16 matrix 峰值估算，1000 TFLOP/s 约为峰值的 43%。这是激进目标，但不是只靠
理论峰值就能否定的目标。真正难点是：四个 grouped GEMM 要在不物化巨型 gather
中间量的前提下长期保持约 1.15 PFLOP/s 聚合吞吐，并把所有非 GEMM 开销控制在
约 0.5 ms。

实现原则只有一句话：**移植 Sonic 的算法与 kernel 边界，但根据 gfx950 的 MFMA、
LDS、cache 和调度机制重新实现，不能照搬 B300 的 TMA、TMEM、2CTA MMA 或 CLC。**

---

## 2. 研究范围、源码依据和已知边界

本计划基于以下本地 SonicMoE 版本：

```text
repository: ../sonic-moe
commit:     0349404acd7952592f73d180ff0c1510f6d112c2
```

同时逐段核对了本地 QuACK：

```text
repository: ../quack
commit:     60d88082272a256fa9b3b2ab631c82cfa78337c6
tag base:   v0.6.4 + 3 commits
```

逐段核对过的主要文件：

- [`benchmarks/moe-cute.py`](../../../../sonic-moe/benchmarks/moe-cute.py)：参数、
  正确性参考、500-repeat 性能口径和 backward 计算方式。
- [`sonicmoe/functional/__init__.py`](../../../../sonic-moe/sonicmoe/functional/__init__.py)：
  autograd call graph、保存张量和 forward/backward kernel 边界。
- [`sonicmoe/functional/backward.py`](../../../../sonic-moe/sonicmoe/functional/backward.py)：
  `gemm_dgated`、两个 varlen-K wgrad、gather-sum 和 router-score backward。
- [`sonicmoe/functional/forward.py`](../../../../sonic-moe/sonicmoe/functional/forward.py)：
  top-k 和 expert aggregation。
- [`sonicmoe/functional/reduction_over_k_gather.py`](../../../../sonic-moe/sonicmoe/functional/reduction_over_k_gather.py)：
  token gather-and-sum 实现。
- [`sonicmoe/functional/triton_kernels`](../../../../sonic-moe/sonicmoe/functional/triton_kernels)：
  GPU routing metadata 的 histogram、prefix-sum、tile-local sort 和 route maps。
- [`assets/2026-04-22-sonicmoe-blackwell.md`](../../../../sonic-moe/assets/2026-04-22-sonicmoe-blackwell.md)：
  官方算法、IO 分析、persistent scheduler、gather fusion 和 dH epilogue 说明。
- [`quack/epilogue/library.py`](../../../../quack/quack/epilogue/library.py)：
  `dgated_mod` 的 scale、dSwiGLU、postact 和 colvec reduce 精确公式。
- [`quack/varlen_utils.py`](../../../../quack/quack/varlen_utils.py)：
  varlen-M/varlen-K 的 ragged tensor、batch offset 和 `A_idx` gather 语义。
- [`quack/tile_scheduler.py`](../../../../quack/quack/tile_scheduler.py)：
  static、global-atomic dynamic、CLC persistent scheduler 和 varlen-M swizzle。
- [`quack/gemm_sm100.py`](../../../../quack/quack/gemm_sm100.py)：
  TMA/cp.async gather、2CTA MMA、warp specialization、register reallocation 和 pipeline。
- [`quack/gemm_config.py`](../../../../quack/quack/gemm_config.py)：
  SM100 tile/cluster/autotune 配置和 dynamic-persistent 开关。
- [`tests/test_linear_varlen_m.py`](../../../../quack/tests/test_linear_varlen_m.py) 与
  [`tests/test_linear_varlen_k.py`](../../../../quack/tests/test_linear_varlen_k.py)：
  两种 ragged GEMM、两种 gather 维度及 reference 语义。

SonicMoE 的依赖只写了 `quack-kernels>=0.4.0`，没有 lock 到生成 B300 参考数字的唯一
QuACK commit；本地 QuACK 又比 Sonic checkout 更新。因此算法和当前实现机制已经有源码
依据，但如果要逐指令复现官方 6.604 ms，仍需从原 benchmark 环境的 wheel metadata 或
autotune cache 追溯实际 QuACK revision。

QuACK 源码确认了几个之前只能从 API 推断的关键点：

1. `dgated_mod` 的 accumulator 是未缩放 `Z=dO@W2`；`dP` 使用 `score*Z`，
   `postact_out` 明确写 `score*A`，colvec reduce 明确累加 `(A,Z)`，即
   `dS=<A,Z>`。因此本文 §3.3 的 `A'=p*A` 和 dS 公式不是推测。
2. 同一个 `A_idx` iterator 在 `cu_seqlens_m` 下 gather M 行，在 `cu_seqlens_k` 下
   gather K 列。Sonic 的 dW1/dW2 确实是在 varlen-K mainloop 内 gather compact X/dO。
3. QuACK varlen-M scheduler 支持 static、global-atomic dynamic 和 SM100 CLC，并带
   L2-oriented group/swizzle/serpentine 顺序。autotuned `GemmConfig.is_dynamic_persistent`
   会与调用参数做 OR，所以 Sonic Python 传入 `dynamic_scheduler=False` 不会禁止一个
   autotune winner 自己启用 CLC。
4. SM100 gather 路径在 cp.async 与 TMA gather 间 autotune；cp.async 路径使用多个
   load warps，heavy DGated epilogue还会调整 warp-group registers 来避免 spill。
5. 当前 QuACK 明确禁止 `gemm_dact/gemm_dgated` split-K；普通 GEMM/wgrad 的 split-K
   才有 serial、parallel、separate reduction 模式。

---

## 3. 精确数学语义与布局

### 3.1 张量和 route 映射

目标原生 API 使用当前 AITER 更自然的布局：

| tensor | shape | dtype | 含义 |
|---|---|---|---|
| `X` | `[T, H]` | BF16 | compact token input |
| `router_w` | `[E, H]` | BF16 | router linear weight |
| `W1` | `[E, 2I, H]` | BF16 | concat `[gate; up]` |
| `W2` | `[E, H, I]` | BF16 | down-projection weight |
| `scores` | `[T, topk]` | FP32 | softmax-over-selected-top-k |
| `topk_ids` | `[T, topk]` | INT32 | selected experts |
| `expert_offsets` | `[E+1]` | INT32 | expert-packed route offsets |
| `route_to_token` | `[M]` | INT32 | grouped route -> compact token |
| `grouped_to_flat` | `[M]` | INT32 | grouped route -> original `(t,k)` |
| `flat_to_grouped` | `[M]` | INT32 | original `(t,k)` -> grouped route |

在数学描述中，`r` 表示按 expert 排序后的 route，`t(r)` 是该 route 的 token，
`e(r)` 是 expert，`p_r` 是对应的 FP32 routing score。每个 expert 的 routes 位于
`[expert_offsets[e], expert_offsets[e+1])`。

Sonic 对外使用 `W1=[2I,H,E]`、`W2=[H,I,E]` 的 view，并要求特定 stride order。
这里的 concat 只描述 W1/dW1 的外部权重顺序。QuACK `concat_layout=("B",)` 在
GEMM 内把两半逻辑重排为 adjacent pairs，所以保存的 `P` 是
`[g0,u0,g1,u1,...]`，`gemm_dgated` 可以直接把每两个 BF16 lane 打包读取；dW1 的
`concat_layout=("out",)` 再把 paired dP 结果写回 concat 权重顺序。第一阶段提供零拷贝
或仅 view 的 compatibility adapter；核心 kernel 不因兼容 API 在计时区间内转置权重。

### 3.2 Forward

对 token `t`：

```text
logits_t = X_t @ router_w^T
(topk_ids_t, p_t) = softmax(topk(logits_t))  # p_t stays FP32

P_r = X_t(r) @ W1_e(r)^T                     # logical concat W1, stored paired
G_r = P_r[0::2]
U_r = P_r[1::2]
A_r = silu(G_r) * U_r                        # [I]
Y_r = A_r @ W2_e(r)^T                        # [H]
O_t = sum_{r:t(r)=t} p_r * Y_r
```

训练 forward 只应长期保存：

- compact `X`；
- pre-SwiGLU `P=[M,2I]`；
- FP32 scores、top-k ids 和 compact route metadata；
- router backward 需要的 logits 或等价状态。

`A` 可以是 forward 两个 GEMM 之间的临时量，但不能作为 backward cache。不得缓存
gathered `X[M,H]`、`Y[M,H]` 或 scattered `Y`。

### 3.3 Backward 的 Sonic 重排

令 `dO=[T,H]`。先计算未乘 routing score 的 down-projection activation gradient：

```text
Z_r = dO_t(r) @ W2_e(r)                      # [I]
```

由结合律：

```text
dscore_r = dot(Z_r, A_r)
A'_r     = p_r * A_r
dW2_e    = sum_r dO_t(r)^T @ A'_r
dP_r     = dSwiGLU(p_r * Z_r, P_r)           # [2I]
```

其中 `A_r` 在 dH epilogue 中由保存的 `P_r` 重算，`A'_r` 是紧接着 dW2 使用的
`[M,I]` 临时量。然后：

```text
dXroute_r = dP_r @ W1_e(r)                   # [H]
dW1_e     = sum_r dP_r^T @ X_t(r)
dXexpert_t = sum_{r:t(r)=t} dXroute_r
```

router tail：

```text
dot_t       = sum_k dscore_tk * p_tk
dtopk_tk    = p_tk * (dscore_tk - dot_t)
dlogits     = scatter(dtopk, topk_ids)        # [T,E]
drouter_w   = dlogits^T @ X
dXrouter    = dlogits @ router_w
dX          = dXexpert + dXrouter
```

这个重排的重要结果是：backward 不需要保存或物化 `Y`、`dY`、gathered `dO`，也不需要
保存 forward `A`。

### 3.4 精确 FLOP 分解

| component | formula | FLOP | TFLOP |
|---|---:|---:|---:|
| stage2 dgrad / fused dH | `2*M*H*I` | 1,099,511,627,776 | 1.099512 |
| dW2 | `2*M*H*I` | 1,099,511,627,776 | 1.099512 |
| stage1 dgrad | `2*M*(2I)*H` | 2,199,023,255,552 | 2.199023 |
| dW1 | `2*M*(2I)*H` | 2,199,023,255,552 | 2.199023 |
| **expert backward total** | `12*T*H*I*topk` | 6,597,069,766,656 | 6.597070 |

两个 router GEMM 各只有 `2*T*E*H = 0.008590 TFLOP`，所以 Sonic 的 TFLOP/s
分母不计 router 是合理的；但是 full backward 的时间必须计入它们。

### 3.5 关键张量大小

| tensor class | size |
|---|---:|
| compact `X` 或 `dO`, `[T,H]` BF16 | 128 MiB |
| route-expanded `[M,H]` BF16 | 1024 MiB |
| preactivation `P` 或 `dP`, `[M,2I]` BF16 | 1024 MiB |
| `A` 或 `A'`, `[M,I]` BF16 | 512 MiB |
| `W1` / `dW1` BF16 | 512 MiB |
| `W2` / `dW2` BF16 | 256 MiB |
| one `[M]` FP32 score or INT32 map | 1 MiB |
| router logits `[T,E]` BF16 | 4 MiB |

只看每个张量至少读写一次的 unique-byte envelope，不等同于实际 HBM traffic：

| component | unique tensors | envelope |
|---|---|---:|
| fused stage2 dH/dS/A' | compact dO + W2 + P + scores + dP + A' + dS | 2946 MiB |
| dW2 gather-fused | compact dO + A' + dW2 | 896 MiB |
| stage1 dgrad | dP + W1 + dXroute | 2560 MiB |
| dW1 gather-fused | dP + compact X + dW1 | 1664 MiB |
| dx gather-sum | dXroute + compact dX | 1152 MiB |

实际 traffic 会受 tile 复读、L2 命中、cache eviction 和 routing skew 影响。后续不得用
上述 envelope 代替 profiler 的 HBM/L2 计数。

---

## 4. SonicMoE 的实际 kernel 边界

### 4.1 Training forward

```text
X
 ├─ router GEMM: F.linear(X, router_w) -> logits
 ├─ fused top-k + softmax              -> FP32 scores, INT32 ids
 ├─ GPU metadata kernels               -> offsets + three route maps
 └─ varlen-M gemm_gated
      gather X in mainloop
      W1 GEMM
      SwiGLU in epilogue
      store P for backward and transient A
        └─ varlen-M W2 GEMM -> expert-packed Y
             └─ token gather-and-sum(scores) -> O
```

Sonic 仍然让 W2 GEMM 连续写 expert-packed `Y`，随后单独 gather-and-sum。官方消融显示，
这比在 GEMM epilogue scatter store 或 atomic accumulation 更快。gfx950 也必须实测三种
策略，不能因为“少一个 kernel”就默认 scatter-fused 更快。

### 4.2 Backward

```text
dO + P + W2 + scores + route maps
 └─ varlen-M gemm_dgated
      gather compact dO in mainloop
      W2 dgrad GEMM -> Z accumulator
      epilogue: recompute A, dSwiGLU, score scaling, dS reduction
      outputs: dP, dS, A'
        ├─ varlen-K dW2: gather compact dO on K dimension, consume A'
        └─ varlen-M W1 dgrad: dP @ W1 -> dXroute
             ├─ varlen-K dW1: gather compact X on K dimension
             └─ token gather-and-sum -> dXexpert

dS -> top-k softmax JVP -> dlogits
dlogits + X + router_w -> drouter_w + dXrouter
dXexpert + dXrouter -> dX
```

四个 backward GEMM 中，stage2 dgrad、stage1 dgrad 是 varlen-M；dW2、dW1 是
varlen-K。Sonic 全部基于同一个 producer-consumer mainloop 和可定制 epilogue 抽象，
差异只在 operand gather 和 epilogue。

### 4.3 Sonic 的关键性能手段

1. 在 GEMM global-to-shared load 中 gather compact `X`/`dO`，不预生成 K 倍大的
   gathered buffer；compact source 更容易保留在 L2。
2. varlen-M 和 varlen-K 使用 persistent tile scheduler，缓解 expert 长度不均导致的
   tail 和 SM 空闲。
3. SwiGLU、dSwiGLU、routing scale、dS reduction 和 A' 生成放进 epilogue。
4. B300 用多级异步 load/MMA pipeline、TMA/cp.async gather、TMEM ping-pong、
   2CTA MMA 和 CLC dynamic scheduler。
5. dH heavy epilogue 与 MMA 重叠。官方较大 Qwen shape 上，普通 GEMM 约
   1213 TFLOP/s，融合 dH 约 1078 TFLOP/s；HBM traffic 增加 24%，吞吐只下降 11%。
6. gather-and-sum 达到 6.5 TB/s 以上，是必须单独优化的带宽 kernel。

第 4 项是 NVIDIA-specific mechanism；第 1、2、3、5、6 项是应当在 gfx950 上实现的
算法或调度目标。

---

## 5. 当前 gfx950 实现与结构差距

当前实现入口：

- [`aiter/ops/opus/moe_bwd.py`](../../../aiter/ops/opus/moe_bwd.py)
- [`aiter/ops/triton/moe_bwd_ref.py`](../../../aiter/ops/triton/moe_bwd_ref.py)
- [`include/gfx950/opus_moe_dgrad_mfma_gfx950.cuh`](include/gfx950/opus_moe_dgrad_mfma_gfx950.cuh)
- [`include/gfx950/opus_moe_wgrad_tn_gfx950.cuh`](include/gfx950/opus_moe_wgrad_tn_gfx950.cuh)
- [`include/gfx950/opus_moe_wgrad_mfma_gfx950.cuh`](include/gfx950/opus_moe_wgrad_mfma_gfx950.cuh)
- [`include/opus_moe_bwd_host_impl.cuh`](include/opus_moe_bwd_host_impl.cuh)

当前 Opus 路径：

```text
Torch topk / argsort / bincount
 -> materialize gathered X
 -> Opus W1 dgrad primitive used as forward GEMM
 -> separate Torch activation
 -> Opus W2 forward GEMM
 -> materialize and save expert-packed Y
 -> Torch index_add combine

backward:
 saved Y + gathered dO -> separate combine kernel -> materialized dy + dp
 -> stage2 dgrad
 -> separate activation backward -> materialized dP
 -> direct-global-load TN dW2
 -> stage1 dgrad -> materialized dXroute
 -> direct-global-load TN dW1 using saved gathered X
 -> gather-sum
 -> router-score kernel + Torch reorder
 -> PyTorch router linear backward
```

差距表：

| area | current gfx950 | Sonic target | consequence |
|---|---|---|---|
| routing metadata | Torch sort/bincount，部分路径有 CPU sync | GPU histogram/prefix/sort | launch、同步、不可 graph capture |
| forward X | 物化 `[M,H]` gathered X | GEMM mainloop gather compact X | 多 1 GiB cache/workspace |
| forward Y | 保存 Y 给 `dp=<dO,Y>` | backward 用 `<Z,A>` | 多保存和读取 1 GiB |
| stage2 input | 先物化 `dy=p*dO` | gather dO + scale in fused dH | 多 1 GiB write/read |
| dSwiGLU | separate kernel | dH epilogue | 多读写 dH/P |
| gate/up layout | P/dP 按两半 concat 处理 | 权重 concat、P/dP adjacent pairs | 无法直接复用 packed-pair epilogue |
| dW2 input | expanded dy | compact dO gather + A' | L2 working set 更大 |
| dW1 input | saved gathered X | compact X gather | L2 working set更大 |
| wgrad | 128x128 direct global load，无 LDS | LDS/multistage varlen-K | 当前最大瓶颈 |
| scheduler | 每 expert 静态 3D grid | persistent work queue | skew/tail load imbalance |
| dgrad | 固定 128x256x32，plain mapping | 多配置、swizzle、persistent | 未覆盖 shape optimum |
| epilogue overlap | 无 | gfx950-specific overlap | fused dH 可能压低 MFMA |
| workspace | forward 中转置权重和动态分配 | 预打包权重、稳定 workspace | 额外流量和 allocator 开销 |

当前 direct TN wgrad 每个 wave 计算 64x64 register tile，确实在 wave 内复用了加载值；
但不同 output tiles 仍反复读取 route slices，kernel 没有 LDS staging、多级 load/MFMA
pipeline 或 persistent reuse。已有 LDS-pipelined wgrad 文件要求先 transpose+pad 输入，
并不是当前端到端 fast path；它证明了 Opus MFMA pipeline 可复用，但不能直接视为问题已解决。

旧 component profile（在最近清理之前）显示方向性瓶颈：

| component | time | effective TFLOP/s |
|---|---:|---:|
| dW1 | 6.11 ms | 360 |
| stage1 dgrad | 2.94 ms | 748 |
| dW2 | 2.84 ms | 387 |
| stage2 dgrad | 2.29 ms | 479 |
| four GEMMs | 14.2 ms | 465 aggregate |

最近清理把 expert total 降到约 15.2 ms，但没有改变 wgrad 是第一瓶颈的结论。后续所有
优化必须按新的统一 benchmark 重测，不能把上表直接与新总时间相加。

---

## 6. 公平 benchmark 规范

### 6.1 必须同时报告的三组结果

1. **Sonic parity derived backward**：完全复刻 `moe-cute.py`。
   - 单独测 training forward；
   - 单独测 training forward + backward；
   - `bwd = fwd_bwd - fwd`；
   - warmup 5，repeat 500；
   - 这是和 `6.604 ms` 比较的唯一主指标。
2. **direct expert-only backward**：复用一次 forward 产生的 context，计时从 `dO`
   到 `dXexpert/dW1/dW2/dlogits`，不含 router linear backward。
3. **direct full backward**：在 2 的基础上加入 `drouter_w`、router `dX` 和最终合并。

direct 指标用于定位，但不能拿 direct expert time 和 Sonic derived full 6.604 ms 混比。

### 6.2 计时约束

- 使用 GPU event；每项同样 5 warmup、500 timed repeats。
- JIT compile、autotune、权重预打包和首次 workspace allocation 全部在计时前完成。
- training parity 路径不默认使用 graph，因为 Sonic training 参考也没有 graph；另行报告
  graph-captured direct backward，名称必须带 `graph`。
- 每个结果运行至少 5 轮，每轮 500 repeats，报告 median、min 和 p95；主判定看 median，
  p95 用于发现 clock/邻居任务干扰。
- 固定 seed、输入分布、权重、`dO` 和 routing mode。记录每个 expert 的 count、
  `max/mean`、标准差和空 expert 数。
- 同时跑两套 routing：
  - 与真实 router/top-k 相同的自然分布，用于端到端主结果；
  - 人工 balanced `M/E=4096`，用于 kernel roofline 和回归。
- 使用独占 GPU、固定可复现的 power/clock 策略；记录 GPU 型号、ROCm、PyTorch、Triton、
  AITER commit、容器、温度和是否有其他进程。
- QuACK 的实测经验表明，共享节点的 clock/co-tenant drift 可达数倍，短 benchmark 还会
  处于 boost clock。无法独占时，必须在单 launch 粒度交错 backend/config，轮换顺序，
  加 contention canary，并让预热时长和输入分布完全一致。
- 对低于约 100 us 的 router/metadata 小 kernel，单独 event pair 可能主要测到 host enqueue
  gap；component microbenchmark 使用 GPU-side backlog/burner，端到端 6 ms 结果仍按原始
  Sonic event 口径。
- CPU pinning 作为 launch-overhead 实验的固定条件记录；不能一边 pinned、一边 unpinned
  比较。随机数据、small-int 和 zero 输入可能导致不同 settled clock，主结果只用相同数据。
- 梯度输出 dtype 必须与 Sonic 一致：BF16 parameter 产生 BF16 `dW`，MFMA 内部 FP32
  accumulate。FP32 dW 是另一种产品模式，必须单独报告时间。

### 6.3 FLOP 报告

三种 backward 结果都使用固定 expert denominator `6.597069766656 TFLOP`，与 Sonic
一致。router FLOP 不加入分母，但 full 时间包含 router。报告中同时列 raw time，避免只看
TFLOP/s 隐藏范围差异。

### 6.4 目标 benchmark 驱动

计划新增一个长期保留的 benchmark，而不是一次性 `op_tests`：

```text
benchmarks/opus_moe/bench_sonic_parity_bwd.py
```

建议参数：

```text
--shape 32768,2048,1024,64,8
--dtype bf16
--activation swiglu
--concat-layout
--routing softmax-over-topk
--warmup 5 --repeat 500 --trials 5
--scope derived,expert,full
--backend opus,triton
--routing-distribution natural,balanced
--output-json <path>
```

JSON 必须保存每个 component event、总时间、route histogram、dispatch config、workspace
bytes 和 profiler 版本，避免只留下终端中的单个最好数字。

---

## 7. 分项最终时间预算

最终预算不能只把四个 GEMM 各写成“1 ms”。stage1 dgrad 和 dW1 各有 2.199 TFLOP，
约为 stage2 对应 GEMM 的两倍。以下 nominal budget 的 full backward 是 6.15 ms，
相当于 1073 effective TFLOP/s，给 6.597 ms 门槛保留约 0.45 ms 风险余量。

| component | FLOP | target time | component TFLOP/s |
|---|---:|---:|---:|
| fused stage2 dH+dS+dSwiGLU+A' | 1.099512 T | 1.10 ms | 1000 |
| dW2 varlen-K gather-fused | 1.099512 T | 0.95 ms | 1157 |
| stage1 dgrad | 2.199023 T | 1.78 ms | 1235 |
| dW1 varlen-K gather-fused | 2.199023 T | 1.82 ms | 1208 |
| **four GEMMs / fused GEMM path** | 6.597070 T | **5.65 ms** | **1168 aggregate** |
| dx gather-sum | — | 0.18 ms | about 6.7 TB/s logical bytes |
| top-k JVP + two router GEMMs + dx merge | 0.017180 T GEMM | 0.22 ms | — |
| launch/residual/workspace overhead | — | 0.10 ms | — |
| **full target** | denominator 6.597070 T | **6.15 ms** | **1073 effective** |

阶段性门槛：

- correctness complete：不设性能门槛，但 exact-shape 不 OOM、无 CPU sync；
- fused dH：`<=1.15 ms`；
- dW2：先过 `<=1.10 ms`，最终 `<=1.00 ms`；
- stage1 dgrad：先过 `<=2.05 ms`，最终 `<=1.85 ms`；
- dW1：先过 `<=2.10 ms`，最终 `<=1.90 ms`；
- expert-only direct：最终 `<=5.95 ms`；
- full direct 与 derived parity：先过 `<=6.45 ms`，最终稳定 `<6.597070 ms`。

如果四个核心路径无法达到约 5.8 ms，继续优化 router 或 Python launch 不可能补足差距；
这会作为明确的 go/no-go 判断。

---

## 8. 分阶段实施路线

### Phase 0：冻结语义、源码和测量基线

交付：

1. 固定 Sonic `0349404a` 和本地 QuACK `60d88082` 源码；另行追溯官方 B300 运行实际
   使用的 QuACK wheel/commit，避免把更新后的 v0.6.4 行为误认为原 benchmark 二进制。
2. 落地 §6 benchmark，重跑 Opus/Triton baseline 和 component breakdown。
3. 明确 benchmark context 中哪些张量属于 forward cache，哪些是 backward workspace。
4. 为每个 component 添加独立 GPU event 和 profiler label。
5. 建立 JSON 历史表；任何 kernel 变更都能关联 config、正确性和端到端结果。

验收：同一环境 5 轮的 median 波动小于 2%；Opus full 基线可复现到历史结果的 ±3%。
超出则先排查 GPU sharing、clock、autotune cache 或 timing scope，不进入 kernel 优化。

### Phase 1：先跑通 Sonic 等价数据流

目标不是马上融合成一个 kernel，而是先让数学顺序、保存张量和对外功能完整一致。

新增 production-style autograd API：

```text
opus_moe_training(X, router_w, W1, W2, topk=8,
                  activation="swiglu", concat_layout=True,
                  scores_dtype=fp32)
```

功能要求：

1. 标准 SwiGLU，不能使用带 clamp、`alpha=1.702` 或 `up+1` 的 GPT-OSS 变体。
2. FP32 top-k scores 贯穿 combine 和 router JVP。
3. 返回或 autograd 生成 `dX/dW1/dW2/drouter_w`；expert-only 调试入口另返回
   `dlogits`。
4. 首版允许把 fused dH 语义拆成 `Z -> dS/A'/dP` 两个 kernel，但不得依赖保存 `Y`
   或生成 `dY`。
5. 首版 varlen-K 可用正确性 kernel，但 API 从一开始接收 compact `X/dO` 和 gather
   map，不把 expanded operand 规定成长期 ABI。
6. 外部 W1/dW1 主路径是 concat；kernel accumulator、保存 P 和 dP 使用 adjacent-pair
   布局。外部 interleaved W1 只作为兼容正确性模式，不进入首轮调优。
7. no-bias 是性能主线；bias gradient 在 correctness complete 后补齐。

正确性顺序：

1. tiny FP32/BF16 PyTorch oracle；
2. small BF16 Triton reference；
3. balanced、skewed、包含 zero-token expert 的 metadata 和 grouped GEMM；
4. exact shape smoke，逐个 gradient 与 Triton 比较抽样及全局 norm；
5. router 小 shape finite-difference；top-k ids 和 maps 必须 bitwise exact。

退出条件：exact shape 完整 forward+backward 可重复运行，不保存 `Y/dY`，所有四个参数/输入
梯度齐全。性能暂时允许不优于基线，但总 workspace 和保存张量必须打印出来。

### Phase 2：全 GPU metadata、稳定 workspace 和预打包权重

1. 先把 Sonic bitmatrix 三阶段 metadata 移植成 Triton/AMD 可运行版本：
   - tile histogram；
   - per-expert/tile exclusive prefix sum；
   - tile-local expert sort + segmented scan，生成三张 route map。
2. 性能稳定后再决定是否改写成 Opus HIP；不要为了统一语言提前重写。
3. 删除热路径中的 `.cpu().tolist()`、Python `int(tensor)`、Torch argsort/bincount 和
   动态 shape-dependent allocation。
4. `expert_offsets`、problem descriptors、persistent work queue 和 reverse token routes
   都写入预分配 workspace。
5. W1/W2 的 kernel-native layout 在 optimizer step 后预打包或由训练权重直接维护；
   backward 计时区间内禁止 `.transpose(...).contiguous()` 复制 256–512 MiB 权重。
6. workspace 地址和 launch shape 保持 graph stable。

验收：metadata 与 Torch reference bitwise 一致；自然 routing 无 host sync；重复调用零 allocator
事件；metadata 时间单独可见且不污染 direct backward。

### Phase 3：production varlen-M dgrad primitive

先构建一个不带 heavy epilogue 的稳定 varlen-M MFMA 基类，供 forward W1/W2、stage2
dgrad 和 stage1 dgrad 共享。

核心设计：

- compact route rows；每个 work item 是 `(expert, m_tile, n_tile)`；
- operand A 支持 contiguous 或 `route_to_token` gather 两种 address iterator；
- stage1 dgrad 的 B iterator 把外部 concat W1 逻辑映射为 paired contraction 顺序，
  与 packed dP 对齐，不生成一份 interleaved W1；
- global buffer load -> 多 stage LDS -> MFMA -> register epilogue；
- ragged M 只在边界 mask，不给每个 expert 物理 pad 大 tensor；
- static grouped order、XCD swizzle 和 global atomic persistent queue 三种 scheduler 都可选；
- exact shape 首先 autotune，随后扩展通用 shape。

第一轮搜索：

| dimension | candidates |
|---|---|
| `BM` | 64, 128, 256（256 仅资源允许时） |
| `BN` | 64, 128, 256 |
| `BK` | 16, 32, 64 |
| waves/CTA | 4, 8 |
| pipeline stages | 2, 3 |
| scheduler | static-linear, expert-major swizzle, dynamic persistent |
| cache policy | default, streaming A, cache/reuse B；以 profiler 为准 |

每个 config 记录：VGPR/SGPR、LDS bytes、waves/CU、spill、MFMA issue/utilization、L2 hit、
HBM bytes 和 tail-wave 比例。资源导致 occupancy 下降到无法覆盖访存延迟，或出现 scratch
spill，直接淘汰，不因单次最好时间保留。

验收：stage1 plain dgrad 先到 `<=2.05 ms`，最终调到 `<=1.85 ms`；balanced 与 natural
routing 都必须过，natural 不得比 balanced 退化超过 10%，否则继续调 scheduler。

### Phase 4：融合 stage2 dH+dS+dSwiGLU+A'

在 Phase 3 mainloop 上增加专用 epilogue：

1. mainloop gather compact `dO[t(r)]`，与 W2 计算 FP32 `Z` accumulator；
2. epilogue 以 packed BF16 pair 读取 `P=[g0,u0,g1,u1,...]`；
3. register 中重算 `A=silu(G)*U`；
4. 做 route 内 reduction `dS=dot(Z,A)`，以 FP32 写回原 `(t,k)` 顺序；
5. 写 `A'=p*A` BF16，供 dW2；
6. 写 `dP=dSwiGLU(p*Z,P)` BF16，供 stage1；
7. 不生成 `Y`、`dY`、gathered dO 或独立 `dy`。

QuACK 的 `ColVecReduce` 会先产生每个 N tile 的 dS partial，再由 finalize 沿 partial
维归约。gfx950 版可以选择 partial-buffer + 小归约 kernel、跨 CTA atomic 或让一个
persistent owner 完成多个 N tiles；三者都要按总时间比较。上面的 1.10–1.15 ms 指标包含
最终 dS，不允许只计 MFMA 主 kernel。

gfx950 没有 B300 TMEM ping-pong。候选实现是：

- 把 accumulator 分成较小 subtiles，交错 MFMA、VALU epilogue 和 vector store；
- 对比 4-wave 全协作与 producer/consumer wave specialization；
- 双缓冲 LDS operand，但不假设 register accumulator 能像 TMEM 一样跨 warp 转移；
- 将 dS reduction 映射到 wave shuffle/LDS 的最小同步路径；
- 如果 full fusion 的 VGPR 或 VALU 压力使 MFMA 吞吐下降过大，比较“GEMM + 一个融合
  vector epilogue kernel”的总时间，保留总时间更快者。

回退判据：full fusion 虽减少 HBM bytes，但相对 plain dgrad 的 MFMA 吞吐下降超过 20%，
且 component 总时间没有改善至少 5%，则撤回该 config，不为追求 kernel 数量强行融合。

验收：component `<=1.15 ms`；`dP/dS/A'` 同时正确；profiler 中没有 scratch spill，
相对 plain GEMM 的 throughput loss 有明确解释。

### Phase 5：重写 varlen-K wgrad——最高优先级

这是从当前约 360–387 TFLOP/s 提升到整层目标的决定性阶段。

统一 primitive：

```text
dW[e,P,Q] = Left[e,:,routes] @ Right[e,routes,:]
```

- dW2：Left 从 compact `dO[T,H]` 按 route gather，Right 是 expert-packed `A'[M,I]`；
- dW1：Left 是 expert-packed `dP[M,2I]`，Right 从 compact `X[T,H]` 按 route gather；
- dW1 store 把 paired accumulator row 直接映射回 external concat dW1，禁止再 launch
  一个 512 MiB layout-conversion pass；
- contraction K 是每个 expert 的 route count，平均 4096，但允许 ragged/zero expert；
- FP32 accumulator、BF16 store 是公平主路径。

必须解决当前 direct-global-load TN kernel 的问题：

1. route slice 先协作载入 LDS，让多个 waves 和多个 MFMA subtiles 共享；
2. 至少双缓冲 global->LDS 与 LDS->register/MFMA；
3. 采用更大 output tile 降低相同 route panel 被不同 blocks 读取的次数；
4. address iterator 直接 gather compact X/dO，不做 transpose+pad 大中间量；
5. expert-major tile order、XCD swizzle 和 cache policy 提升同一 compact token 的 L2 reuse；
6. 用 persistent queue 平衡 ragged K 和不同 output tile 的完成时间。

首轮搜索矩阵：

| parameter | dW2 candidates | dW1 candidates |
|---|---|---|
| `BM x BN` | 64x128, 128x128, 128x256, 256x128 | 128x128, 128x256, 256x128 |
| `BK` routes | 32, 64, 128 | 32, 64, 128 |
| waves/CTA | 4, 8 | 4, 8 |
| LDS stages | 2, 3 | 2, 3 |
| MFMA subtiles/wave | 2x1, 1x2, 2x2 | 2x1, 1x2, 2x2 |
| scheduler | static expert-major, swizzled, dynamic persistent | same |
| gather side | left compact dO | right compact X |

每个 candidate 在编译期检查 LDS 容量和 accumulator VGPR。示例：128x128x32 的一份
BF16 A+B tile 是 16 KiB，双缓冲约 32 KiB（不含 padding/额外 scratch）；
128x256x32 双缓冲约 48 KiB。最终以实际编译资源报告为准。

split-K 不是 exact shape 的默认方案：这里 output tiles 已足以填满 GPU，split-K 会增加
partial buffer、归约和确定性成本。只有以下情况才启用实验：

- profiler 显示单 CTA K-loop latency 导致明显 tail；
- 极端 routing skew 让可并行 work items 不足；
- `split=2/4` 加归约后的端到端时间仍改善至少 5%。

验收分两级：

- Level 1：dW2 `<=1.10 ms`，dW1 `<=2.10 ms`；
- Level 2：dW2 `<=1.00 ms`，dW1 `<=1.90 ms`；
- 最终 stretch：nominal 0.95/1.82 ms。

任何结果都必须包含 gather 和所有必要的 metadata/workspace 成本。只报告“预转置、预 padding
输入上的 kernel-only 速度”不能算通过。

### Phase 6：gather-and-sum、router tail 和 store 策略

#### dx gather-and-sum

- fixed topk=8 编译期展开；
- token-major，H 维 vectorized，FP32 累加、BF16 store；
- 搜索每 token 1/2/4 waves 和 `BLOCK_H=256/512/1024`；
- 比较普通 global load、cache hint 和 route-order swizzle；
- 目标 `<=0.20 ms`，stretch 0.18 ms。

逻辑上该 kernel 读约 1 GiB、写 128 MiB，0.18 ms 相当于约 6.7 TB/s，必须接近 gfx950
实际 HBM 上限；若 profiler 显示 L2 命中贡献，需要同时报告 HBM 和 L2 bytes。

#### Router tail

1. dS 直接按 flat `(t,k)` 写，删除 Torch unsort。
2. top-k softmax JVP + scatter dlogits 融成一个小 kernel。
3. `drouter_w=dlogits^T@X` 和 `dXrouter=dlogits@router_w` 调用成熟 gfx950 GEMM，
   不为 0.017 TFLOP 重造低效 kernel。
4. 让 router dx GEMM 用 `beta=1` 写入 `dXexpert`，或让最后执行的 gather-sum 加载
   `dXrouter`，删除独立 128 MiB dx-add pass。

#### Expert output store

同时测：

- expert-packed store + gather-sum；
- scatter store + contiguous sum；
- atomic add epilogue。

默认保留 Sonic 的第一种。只有后两种端到端稳定快至少 3%，且不破坏确定性/正确性时才切换。

### Phase 7：persistent scheduler、autotune 和 launch 收尾

1. 构造 compact work descriptor，把所有 experts 的 tiles 放入同一 work space。
2. 比较：
   - static expert-major；
   - XCD-aware swizzle；
   - software dynamic persistent queue（global atomic chunk fetch）。
3. dynamic queue 一次领取 2/4/8 tiles，摊薄 atomic；队列开销必须从 profiler 可见。
4. autotune key 至少包含 `(arch,dtype,H,I,E,topk,route histogram class,kernel kind)`；
   不能只按 H/E 缓存而忽略 routing skew。
5. exact shape 固化生产 config，通用 shape 保留有限候选，避免运行时巨大搜索。
6. 合并可合并的小 kernel，预分配 outputs，移除 Python/Torch 隐式 casts。
7. direct backward 支持 graph capture；derived parity 仍按 Sonic 非 graph 口径报告。

验收：full nominal `<=6.15 ms`；连续多轮 median `<6.597070 ms`；自然 routing 和
balanced routing 都通过；没有把 required preparation 移出 scope 伪造结果。

---

## 9. gfx950 调优与 profiler 记录模板

每个 GEMM config 必须保存以下字段：

```text
kernel kind
BM, BN, BK
waves/CTA
MFMA instruction shape
pipeline stages
LDS bytes/CTA
VGPR, SGPR, scratch bytes/thread
max resident waves/CU
scheduler and work-chunk size
XCD swizzle
cache policy for each operand
split-K and reduction mode
balanced/natural route histogram
time, TFLOP/s, p95
```

Profiler 至少回答：

1. MFMA 是否饱和，VALU/SALU/branch 是否抢占 issue；
2. wave occupancy 是被 VGPR、LDS 还是 block 数限制；
3. global read/write bytes、HBM bandwidth、L2 hit/miss；
4. LDS read/write bytes、bank conflict 和 barrier stall；
5. memory dependency、waitcnt 和 long scoreboard stall；
6. 各 XCD/CU 的 active time 是否均衡，尾部有多少空闲；
7. fused epilogue 相对 GEMM-only 增加的 bytes 和吞吐损失。

实际 counter 名称随 ROCm/rocprofiler 版本变化。每个环境先用
`rocprofv3 --list-avail` 固定可用 counters，再把映射写进 benchmark JSON；不要在脚本里
硬编码另一版本的 NVIDIA 或 AMD counter 名。

优化决策顺序：

1. 先消灭 scratch spill 和明显低 occupancy；
2. 再看 MFMA issue 与 LDS/barrier；
3. 再看 HBM/L2 和 tile reuse；
4. 最后才处理 sub-0.1 ms launch/router 细节。

---

## 10. 正确性、性能 gate 和回退规则

### 10.1 Correctness matrix

至少覆盖：

| axis | cases |
|---|---|
| shape | tiny、非方阵、`H==2I` 方阵、目标大 shape |
| routing | balanced、random natural、highly skewed、zero-token expert |
| topk | 1, 2, 8；目标为 8 |
| layout | concat 主路径、interleaved compatibility |
| activation | standard SwiGLU 主路径 |
| scope | expert-only、full router_w |
| output | dX, dW1, dW2, dlogits, drouter_w |

小 shape 使用 FP32 PyTorch autograd golden；中/大 shape 使用 Triton reference 和分项
invariant。top-k ids、offsets 和 route-map permutation 必须 exact。FP32 scores/router JVP
按 FP32 tolerance；BF16 gradients 用 scale-aware absolute/relative error、cosine 和 norm ratio
共同判定，阈值由当前 Triton-vs-PyTorch 基线冻结，新的 kernel 不得放宽阈值掩盖错误。

额外 invariant：

- `flat_to_grouped[grouped_to_flat] == arange(M)`；
- `expert_offsets[-1] == M`；
- 每个 expert 的 dW 与独立 reference contraction 一致；
- `sum(topk JVP, dim=-1)` 接近 0；
- 无 NaN/Inf；
- 重复运行 deterministic 模式结果 bitwise stable。

### 10.2 Performance gate

- kernel-only 改善小于 3% 视为噪声，不进入默认 config；
- kernel-only 改善但 required prep 后 component 退化，立即回退；
- component 改善但 full backward 退化超过 1%，默认回退并分析 cache/并发影响；
- 为某个 balanced case 优化后 natural routing 退化超过 10%，不能作为 production default；
- autotune winner 必须在至少 5 轮中多数获胜，不取单次最低值；
- 任何超过预算的 phase 必须先解释 gap 是 compute、memory、scheduler 还是 launch，
  再进入下一 phase。

### 10.3 每阶段保留可比较 fallback

- unfused dH correctness path；
- static scheduler；
- current direct TN wgrad；
- expert-packed store + gather-sum；
- Triton full reference。

fallback 通过 dispatch/config 保留到最终目标稳定达成，避免一次大改后无法二分回归。

---

## 11. 风险和应对

| risk | impact | mitigation / decision point |
|---|---|---|
| 1000 TFLOP/s 约为 2.3 PF 峰值的 43% | 目标本身激进 | 四 GEMM 5.65 ms 预算；阶段 gate 及早判定 |
| wgrad 需要约 1.15–1.2 PFLOP/s | 当前仅 360–387 TFLOP/s | LDS staging、大 tile、gather fusion、persistent 是主线 |
| fused dH heavy epilogue 压 VGPR/VALU | MFMA 吞吐下降 | subtile、wave specialization；保留 two-kernel fallback |
| gfx950 无 TMA/TMEM/CLC | 不能复制 B300 实现 | buffer load + LDS + waitcnt + software scheduler |
| route imbalance | static grid tail 严重 | natural histogram gate、dynamic queue、chunk fetch |
| compact gather 地址离散 | HBM/L2 miss | expert/tile ordering、XCD swizzle、cache-policy profiling |
| forward 仍保存 expanded X/Y | backward 快但结构/内存不对齐 | Phase 1 同时改 forward cache contract |
| BF16 dW 与 FP32 dW混比 | 结果不公平 | BF16 主结果，FP32 单独模式 |
| derived backward 是两个 benchmark 相减 | 噪声放大 | 同时报告 direct；5 trials + median/p95 |
| Sonic 未锁定官方 benchmark 的 QuACK revision | 逐指令对照可能版本错位 | 已读本地 `60d88082`；Phase 0 追溯原 wheel/autotune metadata |
| GPU 被其他任务占用 | 性能不可复现 | 独占、记录 clocks/processes、波动 gate |

如果在完成 production varlen-M、fused dH 和 gather-fused LDS wgrad 后，四 GEMM 仍稳定
高于 6.0 ms，说明 1000 full TFLOP/s 在当前实现/硬件状态下风险极高。此时应输出 roofline
和 counter 证据，决定是继续做更深的 assembly/scheduler 工作，还是把阶段目标调整为
900/950 TFLOP/s；不能用缩小 benchmark scope 宣称完成。

---

## 12. 推荐的实际执行顺序

```text
P0  benchmark + pin Sonic/local QuACK + trace official wheel revision
 |
P1  Sonic-equivalent math/dataflow, all gradients correct
 |
P2  GPU metadata + stable workspace + native weight layout
 |
P3  shared production varlen-M primitive
 |\
 | P4 fused dH to <=1.15 ms
 |
P5  gather-fused LDS varlen-K wgrad to <=3.0 ms combined
 |
P6  dx gather-sum + router tail to <=0.4 ms combined
 |
P7  persistent scheduling/autotune/launch cleanup
 |
full < 6.597070 ms and >1000 effective TFLOP/s
```

优先级上，P5 wgrad 是最大收益项；依赖关系上，不能跳过 P1/P2/P3。否则很容易得到一个
在预转置、预 gather、balanced-only 输入上很快，但无法组成完整 Sonic-style backward 的
kernel-only 数字。

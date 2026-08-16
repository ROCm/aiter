# Dispatch + GEMM1 融合设计 (aiter gfx1250)

- 日期: 2026-07-29
- 目标硬件: gfx1250 (AMD, FlyDSL / WMMA a8w4 路径)
- 相关代码:
  - `aiter/aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py`
  - `aiter/aiter/ops/flydsl/dispatch_combine_v2/intranode_kernels.py` (`make_dispatch`)
  - `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py` (`_grouped_a8w4_tdm_moe`)
  - `aiter/aiter/ops/flydsl/moe_kernels.py` (`flydsl_moe_topids_to_rows`, `flydsl_moe_fused_quant_preshuffle`)
  - `aiter/op_tests/multigpu_tests/test_mega_moe.py` (端到端测试入口)

## 1. 背景与动机

当前 EP MoE 一层的前半段（TDM EP 默认路径, `_grouped_a8w4_tdm_moe`）是多个独立 kernel：

```
dispatch(P2P bf16 → arena disp_out[arrival order])
  → flydsl_moe_topids_to_rows(+g2l_lut, gather_w)      # 路由/分桶行映射
  → contiguous_psum_remap                              # 前缀和 + 行重映射
  → flydsl_moe_fused_quant_preshuffle(读 disp_out →    # 读一遍全 hidden
        grouped fp8 payload + preshuffle e8m0 scale)
  → gemm1 (TDM batched a8w4)
```

现有的 `scatter_fused`（combine_mode）已经把 **gemm2 + combine** 融合：gemm2 的 epilogue 直接把加权结果 P2P 写进 peer 的 `comb_inp`，combine 退化为 barrier + 求和。

本设计做它的**镜像**：把 dispatch **之后** 的 "scatter 到 grouped 布局 + fp8 量化 + e8m0 scale + preshuffle" 折进 dispatch kernel 的接收路径，即 **dispatch prologue 融合**。

**关键约束（为何 dispatch→gemm1 比 gemm2→combine 难融）**: gemm2→combine 是"输出即散射"，纯 epilogue 行为，`ep_rowmap` 在计算前已知。而 gemm1 的 tile 网格形状依赖 **每个本地 expert 收到多少 token**，这是所有 rank P2P 到齐后的跨 rank 归约结果。真正的单 kernel（等 count 到齐再开算）需要 persistent kernel + kernel 内 grid sync + 动态 scheduler —— 这是方案 A（Phase 2）。方案 C（Phase 1）**不改变 kernel 边界**，只把输入准备工作前移进 dispatch，不试图隐藏通信延迟。

## 2. 现状事实（设计前提）

1. **接收布局是到达顺序**: dispatch 接收端用 per-rank 全局计数器 `off_tok_off` 原子分配 `dest_tok_id ∈ [0, max_recv)`，token 字节写入 `disp_out[dest_tok_id]`，**不按本地 expert 分桶**。分桶发生在后续 `topids_to_rows` + `fused_quant_preshuffle`。
2. **per-dest-PE 去重**: 一个源 token 若命中同一 dest rank 上的多个本地 expert，dispatch 只 P2P 发送**一份物理拷贝**（`make_dispatch` Phase 1 的 dedup 逻辑）。但 gemm1 的 grouped 布局需要它在**每个 expert 桶各占一行**。当前这份 "一份物理拷贝 → 多个 grouped 行" 的复制由 `topids_to_rows` 间接索引在 quant+preshuffle scatter 阶段完成。

## 3. 方案 C (Phase 1): dispatch prologue 融合

### 3.1 目标

消除独立的 `flydsl_moe_fused_quant_preshuffle` kernel 及其对 `disp_out` 的一次全 hidden 宽度读回；将 "分桶 scatter + 量化 + preshuffle" 合并进 dispatch 接收路径。

### 3.2 核心机制：接收端 "边收边分桶量化"

改造 `make_dispatch` 接收/写入路径。每个 recv token 落地时：

- 用 **per-(本地 expert) 计数器**（替代或并存于现有 per-rank `off_tok_off`）拿到该 token 在其 expert 桶内的 slot。
- 对该 token 做 fp8 量化 + 每 32 元素 e8m0 scale。
- 按 gemm1 需要的 WMMA preshuffle 布局，写入 `grouped_a1[expert, slot]` 与 `grouped_a1_scale`。
- **intra-rank 复制**: token 命中本地多个 expert 时，接收端 warp 遍历该 token 的 topk，对每个"本地且未被丢弃"的 expert 各写一份 grouped 行（量化只算一次、写多次），以此取代当前 `topids_to_rows` 的间接复制。

### 3.3 布局选择

- **采用 masked `[E, max_m]` 布局**喂 gemm1（dispatch 天然按 expert 分桶，最自然；gemm1 已支持 masked 路径）。
- **不采用** contiguous 布局作为 Phase 1 首选：contiguous 需要全局 psum（仍需一发轻量 psum kernel），留待后续优化。

### 3.4 改动点

- `EpDispatchCombineConfig` / `EpDispatchCombineOp`:
  - 新增 arena region `grouped_a1`（`[E, max_m, hidden]` fp8 容量）+ `grouped_a1_scale`（preshuffle e8m0），或复用现有 gemm1 输入 buffer。
  - 新增 per-(本地 expert) 计数器区（替代/并存 `off_tok_off`）。
  - 新增开关属性 `fuse_dispatch_gemm1`（由 env `AITER_EP_FUSE_DISPATCH_GEMM1` 驱动，默认关，用于 A/B 对照，风格对齐现有 `AITER_EP_SCATTER_TDM`）。
- `make_dispatch(..., fuse_gemm1_prep=True, off_grouped_a1, off_grouped_a1_scale, wmma_rep, quant_mode, experts_per_rank, ...)`:
  - 接收 epilogue 增加：量化 → e8m0 → preshuffle → 分桶写 → per-expert 计数累加。
  - preshuffle 索引复刻 `_grouped_a8w4_preshuffle_e8m0_scale` 的 WMMA 4×32 转置布局。
- `_grouped_a8w4_tdm_moe`:
  - 当 dispatch 已产出 grouped 输入（`fuse_dispatch_gemm1` 开）时，**跳过** `flydsl_moe_fused_quant_preshuffle`，直接把 `grouped_a1 / grouped_a1_scale / masked_m / (masked 布局无需 contiguous psum)` 喂给 gemm1。
  - `topids_to_rows` 的 scatter 部分并入 dispatch；仍需保留 gather_w（route weight）用于 gemm1/后续，或一并在 dispatch 接收端写出。

### 3.5 触发与回退

- `AITER_EP_FUSE_DISPATCH_GEMM1=1` 开启；默认 `0` 走现有分离路径。
- 与现有 `scatter_fused`（gemm2+combine 融合）正交，可组合。

### 3.6 收益与非目标

- **收益**: 省 `fused_quant_preshuffle`（及部分 route scatter）kernel launch/层 + 一次全 hidden 宽度中间读写。61 层累积可观。属于与 gemm2+combine 融合同类的 kernel/带宽收益。
- **非目标**: 不隐藏 dispatch 通信延迟（属方案 A）。
- **CUDA graph**: 天然兼容（仍是固定 kernel 序列）。

### 3.7 主要风险

- **preshuffle 在 kernel 内的正确性**: e8m0 scale 的 WMMA 4×32 转置布局要在 FlyDSL dispatch kernel 内精确复刻，是主要工程量与正确性风险点。
- **per-expert 计数与 slot 分配**: 从 per-rank 计数改为 per-expert 计数，需保证原子分配与 `max_m` 容量边界（丢弃/溢出行为与现路径一致）。
- **intra-rank 复制**: 去重语义与 grouped 复制的交互需仔细处理，避免漏写或重复写。

## 4. 方案 A (Phase 2): dispatch+gemm1 单 persistent kernel（细化设计）

### 4.1 目标与边界

单一 persistent kernel 完成 **dispatch(P2P 拉 token) + gemm1(a8w4 grouped GEMM)**，用 gemm1 的计算**隐藏 dispatch 的跨 rank 通信延迟**：已经到齐的 expert 立即开算，同时后续 expert 的 token 还在传输。

边界（本设计只做 dispatch+gemm1）:
- SwiGLU、gemm2、combine **不**并入本 kernel（保持现状，可与 `scatter_fused` 组合）。这与 DeepGEMM 的"整层 mega"不同 —— 有意收窄范围，降低风险。
- 输出: gemm1 的结果（bf16 中间态或已 SwiGLU+fp8 量化的 gemm2 输入），落回现有 buffer 供后续 kernel 消费。

### 4.2 kernel 结构：warp 专精 + 环形池

参考 DeepGEMM `sm100_fp8_fp4_mega_moe` 的三类 warp 角色，但砍掉 combine 相关角色：

- **Dispatch warps**: 复用方案 C 的接收路径 —— P2P 拉 token 并"边收边量化+preshuffle+分桶写"进环形池 grouped 布局，累加 per-(本地 expert) 计数（计数带"已到齐"高位标志，见 4.4）。
- **Scheduler warp（1 warp）**: 自旋等某 expert 的 recv count 被标记"所有 SM×rank 到齐"后，按 `BLOCK_M` 把该 expert 切成 M-block 任务，通过 LDS 上的 full/empty barrier 队列发给 gemm warps。动态调度以吃掉 expert 间 token 数不均的尾部。
- **GEMM warps**: 从任务队列取 M-block，做 a8w4 WMMA GEMM1，写输出。gemm1 的 A（激活）直接读环形池里已量化+preshuffle 的 grouped fp8，prologue 无需再量化（复用 C 的产物）。

### 4.3 对称 arena 追加

- **grouped 环形池**: `grouped_a1`（fp8 payload）+ `grouped_a1_scale`（preshuffle e8m0），容量按"同时存活的 pool block 上限"开（参考 DeepGEMM `get_num_max_live_pool_blocks`，避免按最坏全量），作为 dispatch warp（生产者）→ gemm warp（消费者）的队列。
- **per-expert recv count（带到齐标志）**: 每个本地 expert 一个 64-bit 计数，低 32 位为已收 token 数、高 32 位为"已上报的 (SM×rank) 数"。scheduler 自旋等高位 == `num_sms * num_ranks`（等价 DeepGEMM `fetch_expert_recv_count` 的 `>>32 == kNumSMs*kNumRanks`）。
- **任务队列 barrier**: LDS 内 `full_barriers` / `empty_barriers`（生产-消费）+ scheduler↔gemm 的 task_info 槽。

### 4.4 跨 rank 同步

- 复用 cco `Window.lsa_ptr` 做 P2P；dispatch warp 完成本 SM 的发送后，用 **kernel 内 grid sync**（所有 SM 到一个全局 barrier）+ 向各 peer 的 per-expert count 高位做 `atomic_add_sys`，使 scheduler 能判定"全局到齐"。
- 现有 `make_dispatch` 已有 Phase 2/3 的 grid barrier（`addr_disp_bar` + `spin_until_eq`）与 per-source count 归约（`off_recv_num`/`total_recv`）—— 方案 A 在此之上把"到齐信号"细化到 per-expert 粒度即可，不用从零造。

### 4.5 FlyDSL 需补齐的基建（prerequisite）

- kernel 内 persistent 调度（固定 grid = num_sms，kernel 不退出、循环取任务）。
- kernel 内 producer-consumer barrier（LDS mbarrier 语义）与动态 M-tile 网格（tile 数运行时决定）。
- 现状: FlyDSL 的 grouped gemm 是"一发算完整个 grouped 布局"的静态网格，没有 kernel 内 scheduler / 动态 tile / 生产-消费队列。这是方案 A 的**主要前置工程**。

### 4.6 风险与工作量

- 工作量: 高（周级~月级）。核心难点是 FlyDSL 侧 persistent + 动态调度 + 生产消费队列基建，以及防死锁的 L1 warmup（gemm 不能在其 M-block 的 token 未到齐前开算）。
- 收益: 在通信/计算比高的场景可隐藏大部分 dispatch 延迟，超出方案 C 的"省 kernel/带宽"。
- 复用关系: 方案 C 的"接收即量化+preshuffle"逻辑直接作为方案 A 里 dispatch warp 的产出 + gemm warp 的 prologue —— **C 是 A 的组件，非一次性投入**。
- 建议门槛: 仅当 Phase 1(C) 落地后 profile 显示瓶颈仍在 dispatch 通信本身（而非小 kernel/带宽）时才启动 A。

## 5. 测试策略

- 复用 `test_mega_moe.py`:
  - 正确性: `AITER_EP_FUSE_DISPATCH_GEMM1=1` vs `=0`，跑 `--acc_verify 1`，对齐 fp32 参考（`MEGA-CHECK PASS`）。
  - 性能: `--profile_table 1`，确认 `fused_quant_preshuffle` kernel 消失、per-layer 时间下降。
  - 与 `--combine scatter_fused` 组合验证正交性。
- 命令示例（非 `/app` 目录下运行，避免 `/app/triton` 遮蔽）:

```bash
cd /tmp && AITER_EP_FUSE_DISPATCH_GEMM1=1 \
ENABLE_CK=0 AITER_FORCE_A8W4=1 AITER_USE_GROUPED_GEMM=1 AITER_BF16_FP8_MOE_BOUND=0 \
torchrun --standalone --nproc_per_node=4 \
  /app/aiter/op_tests/multigpu_tests/test_mega_moe.py \
  -q a8w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 \
  --combine scatter_fused --layers 2 --acc_verify 1 --profile_table 1
```

## 6. 交付范围（Phase 1 / 方案 C）

- [ ] `make_dispatch` 接收 epilogue 融合量化+preshuffle+分桶写 + per-expert 计数
- [ ] arena region / config / env 开关
- [ ] `_grouped_a8w4_tdm_moe` 在开关下跳过独立 quant+preshuffle，喂 masked 布局给 gemm1
- [ ] 正确性对齐 + 性能对照

## 7. Phase 3 (方案 A+): 整层 dispatch+L1+SwiGLU+L2+combine 单巨核

对标 DeepGEMM `sm100_fp8_fp4_mega_moe`（Blackwell 整层 mega）。把 EP MoE 一整层——dispatch 拉 token → gemm1(L1) → SwiGLU+fp8 量化 → gemm2(L2) → combine 加权散射回源——全部塞进**一个 persistent kernel**，层内**不落任何中间张量到 HBM**，并用计算掩盖 dispatch 与 combine 两端的跨 rank 通信。

### 7.1 与前序阶段的关系

Phase 3 = **Phase 2（dispatch+L1 单核）** 向后延伸吞掉 **现有 `scatter_fused`（L2+combine 融合）**，中间用 SwiGLU+量化桥接：

```
Phase 1(C):        [dispatch 接收即量化+preshuffle] ──独立 kernel──> L1 ... L2 ... combine
Phase 2(A):        [dispatch + L1]  单核             ──> SwiGLU ──> L2 ──> combine(scatter_fused)
Phase 3(A+):       [dispatch + L1 + SwiGLU + L2 + combine]  单一 persistent 巨核
```

- 入口沿用 Phase 1 的"接收即量化+preshuffle"作为 dispatch warp 产出 + L1 prologue。
- 出口沿用现有 `scatter_fused`：L2 epilogue 把加权结果 P2P 写进 peer 的 `comb_inp`，combine 退化为 barrier+求和——只是把这段搬进同一个 kernel 的 epilogue warp。
- 因此 Phase 3 不是全新造轮子，而是把已验证的两端融合逻辑 + Phase 2 的持久调度**缝合**成一层一核。

### 7.2 kernel 结构：全 warp 专精 + 双级 tile 环形池

参考 DeepGEMM：一个 warpgroup 做调度（non-epilogue），其余 warp 做 epilogue 流水（L1→SwiGLU→L2→combine）。gfx1250 上映射为：

- **Dispatch warps**：P2P 拉 token，边收边量化+preshuffle 写进 L1 输入环形池（grouped fp8），累加 per-expert recv count（带到齐高位）。
- **Scheduler warp(s)**（non-epilogue）：`MegaMoEScheduler` 等价物。自旋等 per-expert recv count 到齐后，为该 expert 生成 **L1 tile 任务**；L1 tile 完成后再派生对应 **L2 tile 任务**；用两套 full/empty barrier 管 L1↔L2 的生产-消费与 combine 就绪。动态调度吃 expert 间不均。
- **GEMM/epilogue warps**：从任务池取任务：
  1. **L1 GEMM**（a8w4 WMMA）：A=环形池里的 grouped fp8，W=expert 的 gate/up 权重（MXFP4 preshuffle）。
  2. **SwiGLU + fp8 量化**：在寄存器/LDS 内对 L1 输出做 `silu(gate)*up`，直接 fp8 量化 + e8m0，写进 **L2 输入环形池**（不落 HBM）。
  3. **L2 GEMM**（a8w4 WMMA）：产出该 token 的层输出。
  4. **combine 散射 epilogue**：按 topk 权重乘好，P2P 写进 peer 的 `comb_inp` 槽（复刻现 `scatter_fused` 的 `ep_scatter_params` 目标寻址）。

### 7.3 环形池与防死锁容量

- 两级环形池：`ring_a1`（L1 输入，dispatch→L1）与 `ring_a2`（L2 输入，SwiGLU→L2）。容量按"同时存活 pool block 上限"开，需**跨 L1/L2 两种 BLOCK_M 组合**求最坏存活块数（等价 DeepGEMM `get_num_max_live_pool_blocks`），否则 L1 占满池、L2 拿不到输入 → 死锁。
- L1 warmup 约束：某 M-block 的 token 未到齐前，其 L1 tile 不得开算；scheduler 用到齐高位门控。

### 7.4 跨 rank 同步（两处 barrier）

- **入口**：dispatch 完成后 kernel 内 grid sync + per-expert count 高位 `atomic_add_sys` 上报（同 Phase 2 §4.4）。
- **出口**：combine 散射前后需跨 rank barrier，确保所有 peer 的 `comb_inp` 写完再做 barrier+求和。复用现 `scatter_fused` 的 `cross_device_barrier` + `_combine_fused()` 语义，但在同一 kernel 内完成（kernel 内 NVLink barrier）。
- 全程只有这两个跨 rank 同步点，层内其余数据流靠 kernel 内 LDS barrier。

### 7.5 权重驻留与带宽

- L1(gate/up) 与 L2(down) 权重都需可访问；按 tile 从 HBM 流式读入（grouped 已 preshuffle 的 MXFP4 payload + e8m0）。整层一核使权重复用局部性变差（同一 SM 可能连续处理不同 expert），需权衡 tile 顺序（按 expert 聚簇调度以复用权重）。
- 寄存器/LDS 压力是主要 occupancy 杀手：L1 累加器 + SwiGLU 中间 + L2 累加器 + 双环形池 barrier 同时占用。

### 7.6 CUDA graph 与调用形态

- 目标形态同 DeepGEMM：一层一次 kernel launch，61 层在一张 CUDA graph 内捕获（`test_mega_moe.py` 的 fused 路径即此形态）。
- host 侧退化为"准备权重/路由指针 + launch 巨核"，无层内动态分配。

### 7.7 FlyDSL 需补齐的基建（在 Phase 2 之上追加）

- kernel 内 **两级 GEMM 串联**：L1 输出不落 HBM，经 SwiGLU+量化直喂 L2（寄存器/LDS 中转）。
- **SwiGLU+fp8 量化 epilogue** 作为 L1→L2 之间的 in-kernel 融合算子。
- kernel 内 **combine 散射 + 跨 rank barrier**（把 `scatter_fused` 的 epilogue 与 `_combine_fused` 语义内联进巨核）。
- **双级环形池的死锁安全容量推导**（跨 L1/L2 block 尺寸）。
- 前三项之外，Phase 2 的 persistent 调度 / 动态 tile / 生产-消费 barrier 均为前提。

### 7.8 风险与工作量

- 工作量：最高（月级+），是三阶段里最大的框架级投入；正确性与死锁调试成本显著。
- 主要风险：
  - **死锁**：L1/L2 双环形池容量推导错误、跨 rank barrier 顺序错误。
  - **occupancy/寄存器压力**：双 GEMM + SwiGLU + 双池共存，可能压到极低并发，反噬吞吐。
  - **权重带宽**：整层一核削弱权重复用，需按 expert 聚簇调度补偿。
  - **数值**：in-kernel SwiGLU+二次量化的舍入需对齐分离路径参考。
- 收益：层内零中间 HBM 往返 + 两端通信被计算掩盖，理论上是三阶段的性能上限；但仅在 Phase 2 已证明 dispatch 掩盖有效、且中间张量带宽确为瓶颈时才值得。

### 7.9 落地策略（增量、可回退）

1. 先做 Phase 1（方案 C）拿 kernel/带宽收益。
2. 再做 Phase 2（dispatch+L1 单核）验证"kernel 内持久调度 + 跨 rank 到齐掩盖"这套基建。
3. Phase 3 在 Phase 2 的 dispatch+L1 巨核基础上，逐段吞并 SwiGLU→L2→combine：
   - 先把 L2+combine（已有 `scatter_fused`）内联为 epilogue，验证出口 barrier。
   - 再打通 L1→SwiGLU→L2 的 in-kernel 中转，去掉中间 HBM。
4. 每步都保留 env 开关与分离路径回退，`--acc_verify` 逐段对齐 fp32 参考。

### 7.10 建议门槛

仅当 Phase 1+2 落地、profile 表明**中间张量 HBM 往返 + 两端通信**仍是层级瓶颈时才启动 Phase 3。否则停在 Phase 2 已能吃到"dispatch 掩盖 + 出口 scatter_fused"的大部分收益，风险/收益比更优。

## 8. 设计复核（执行前代码核对，2026-07-29）

进入实现前对现网代码逐行核对，发现原设计（§3）几处与实际不符，需在动代码前重估。

### 8.1 事实订正

1. **config 字段**: `EpDispatchCombineConfig` 直接有 `num_experts_per_rank`（不是 `num_experts // world_size`）；`is_fp8/is_fp4` 指**传输 dtype**。a8w4 路径 dispatch 传 **bf16**，量化目标固定 `quant_mode="fp8"`（激活 8-bit），与传输 dtype 无关。
2. **两条 GEMM 路径**（`grouped_moe_gfx1250.py`）：
   - **TDM 路径** `_grouped_a8w4_tdm_moe`（默认，`AITER_GROUPED_A8W4_TDM=1` / `_TDM_EP=1` / `AITER_EP_SCATTER_TDM=1` 全默认开）：**仅 contiguous 布局**。`ep_rowmap` 折进 `contiguous_psum_remap`（Opportunity A）。
   - **grouped-masked 路径**（fallback）：支持 **masked**（用独立 `build_ep_rowmap` kernel 产 ep_rowmap）**或** contiguous（折叠）。
   - ⇒ **masked 布局并不会破坏 gemm2+combine 融合**（有 standalone `build_ep_rowmap`），但 masked = 切到**非默认、未按 TDM 调优**的 GEMM 路径。
3. **实际只剩一处独立 quant+preshuffle 可融**：TDM 路径上 gemm1→gemm2 的 a2 量化+preshuffle **已折进 gemm1 epilogue**（`_fuse_quant`, line 569）。唯一还独立的是 **a1** 那次（line 555）`flydsl_moe_fused_quant_preshuffle(hidden_states, ..., topids_to_rows=..., source_topk=topk)` —— 它从本 rank 的 dispatch 输出 `hidden_states`（dense `[token_num, model_dim]`）按 `topids_to_rows` gather 进 **contiguous grouped 行**。**Phase 1 的精确目标 = 干掉这一处 a1 调用**，让 dispatch 直接产出 a1_payload/a1_scale。

### 8.2 核心张力：布局 vs 流式可写性 vs 默认路径

- **contiguous（默认 TDM 需要）**: grouped 行号 = `row_starts[e] + slot`，`row_starts` 是**跨 expert 的全局前缀和**，只有所有 token 收齐后才知道。⇒ dispatch **无法在接收流中直接写最终 contiguous 行**；需 "接收+量化 → grid-sync → 前缀和 → 二次 remap 落位"。二次落位只搬 fp8 payload（hidden/1B）+ e8m0（远小于 bf16 全宽），仍省掉对 `hidden_states` 的一次 **bf16 全宽**读回 + 独立 kernel launch。
- **masked（§3 原选）**: 行号 = per-expert 原子 slot，**无需全局前缀和**，dispatch 可**接收即流式写**，最省事。但代价是切到非默认 grouped-masked GEMM 路径 + standalone `build_ep_rowmap`，且 `[E,max_m]` 有容量浪费；能否达到 TDM 默认路径的性能未知。

### 8.3 修订方案选项

- **R1（推荐）—— contiguous，两阶段折进 dispatch kernel**: 保留 `topids_to_rows`/`contiguous_psum_remap`（⇒ `ep_rowmap`、gemm2+combine 融合零改动、留在默认 TDM 路径）。dispatch 接收阶段做 fp8 量化+preshuffle 写进 **arrival-order per-expert scratch**；同 kernel grid-sync 后按 per-expert 前缀和把 payload/scale remap 到 contiguous 行。净收益 = 省掉 line-555 独立 kernel + 对 `hidden_states` 的 bf16 全宽读回。改动集中在 dispatch kernel，GEMM/combine 侧几乎不动。
- **R2 —— masked，切 grouped-masked 路径**: dispatch 流式写最简单，但改变默认 GEMM 路径 + 换 ep_rowmap 产法，性能回归风险高，需先证明 masked 路径不比 TDM 慢。
- **R3 —— 收窄目标**: 若 profile 显示 line-555 占比很小，可能不值得融；先量它的实际耗时占比再决定。

### 8.4 对 plan 的影响

若选 R1：plan **Task 1** 的 `experts_per_rank` 属性删除（用现有 `num_experts_per_rank`）、`quant_mode` 固定 `"fp8"`、grouped region 改 **contiguous 容量** `[contiguous_m, ...]`（非 `[E,max_m]`）；**Task 2** dispatch epilogue 改为"两阶段（接收量化 scratch → grid-sync 前缀和 remap）"，保留 `topids_to_rows`/`contiguous_psum_remap`；**Task 3** 改为"只把 line-555 的 a1 quant_preshuffle 换成读 dispatch 产出的 contiguous a1_payload/a1_scale"，不动 psum/ep_rowmap。

---

## 8.5 1a 实测结论（2026-07-29，4×gfx1250，e384/k6/hd7168/id3072/scatter_fused）

R1-1a（到达序 per-token 量化折进 dispatch）已完整落地并逐层验证正确（单 rank 数值门禁、fp8-gather 与 bf16 路径逐字节一致、2/4-rank `MEGA-CHECK PASS` 且 logits 与基线相同）。但**性能是负优化**，A/B（self device time，avg/4 ranks）：

| kernel | 融合 OFF | 融合 ON | Δ |
|---|---|---|---|
| `ep_dispatch_0` | 754.0us | **1326.2us** | **+572** |
| a1 prep（`quant_preshuffle` bf16 → `fp8_gather` fp8） | 75.2us | 78.6us | +3 |
| gemm1 `...K7168` | 1176.8 | 1177.3 | ~0 |
| gemm2 / combine | 802/226 | 809/257 | +小 |
| **TOTAL device / layer** | **570.3us** | **673.4us** | **+103 (+18%)** |

**根因**：1a 在**照常 P2P 跨 rank 写 bf16 token（14KB）之上，又额外 P2P 写 fp8（7KB）+ e8m0**，跨 rank 写流量 +50%，而这条 P2P 写正是 dispatch 瓶颈 → `ep_dispatch` +572us；而 a1 端本来读的是**本地** bf16、量化用廉价原生 pk8，换成本地 fp8 只省了本地半带宽，**几乎无收益**。

**结论**：只要 dispatch 仍发 bf16，"bf16+fp8 并存"结构上就赢不了。1a **废弃**（代码保留 `AITER_EP_FUSE_DISPATCH_GEMM1` 开关，默认关，不影响主线；作为负结果实证）。

## 8.6 DeepGEMM 对照（`/app/DeepGEMM/.../sm100_fp8_fp4_mega_moe.cuh`）

逐行核对 DeepGEMM 的整层 mega MoE，其做法与 1a 正相反：

1. **激活 FP8(e4m3)、权重 FP4(e2m1)**（L143-144）；mega kernel 的输入 `input_token_buffer` **本就是 fp8** + `input_sf_buffer`（scale）。⇒ **量化发生在 dispatch 之前**（上游/上一层输出即产 fp8），每源 token 只量化一次。
2. **传输即 fp8**：dispatch warps（L414-575）用 TMA 把远端 **fp8 token 字节 + SF** 拉进本地 L1 ring —— **纯字节搬运，pull 路径零量化**；网络上跑 fp8（bf16 的一半），**不传 bf16**。
3. **唯一的"计算中量化"是 L1→L2 requant**（GEMM1 输出→fp8 喂 GEMM2），在 epilogue warps 用片上 amax 归约完成，数据不出芯片。
4. **形态**是 warp-specialized 持久巨核：dispatch warps 拉 fp8 ring / MMA warps 算 GEMM1 / epilogue warps 做 SwiGLU+requant+GEMM2+combine（对应本 spec §7 Phase 3）。

DeepGEMM 的赢点：**fp8 取代 bf16 上线**（传输减半）+ **dispatch 是纯 fp8 mover（零量化开销）** + **GEMM1 直接吃 fp8（免 re-quant）**。1a 之所以回退，正因为它做成了"并存"而非"取代"。

## 8.7 方向修正 → fp8 transport 打底 + 真·dispatch+GEMM1 融合（1a-v2）

采用 DeepGEMM 式路线；aiter dispatch **已具备 fp8 transport**（`EpDispatchCombineConfig.is_fp8` / `data_type=fp8` → token 按 fp8 字节搬运 `token_nbytes=hidden×1`，并把 per-token scale 转发进 `out_scales`；见 `dispatch_combine_op.py` `is_fp8`、`out_scales` region、`dispatch(scale=...)`）。

**基础（fp8 transport，可测且收益为正）**：
- 在 dispatch **之前**把层输入（RMSNorm 输出）量化成 fp8+e8m0（每源 token 一次，理想融进 rmsnorm epilogue）。
- dispatch 以 `data_type=fp8` + `scale_dim=hidden//32` 传输：**只发 fp8+e8m0（减半）**，不发 bf16；`dest` 得到 fp8 `recv_x` + `out_scales`（到达序）。
- GEMM1 的 a1 prep 直接消费 fp8 `recv_x` + `out_scales`，复用**已落地并验证**的 `build_moe_fp8_gather_preshuffle_route_ksplit_module` / `flydsl_moe_fused_quant_preshuffle(in_fp8_payload=, in_fp8_scale=)`（原 Task 3 产物，只需把输入源从 `disp_out_q` 换成 fp8-transport 的 `recv_x`/`out_scales`）。
- 净收益：跨 rank 传输减半 + 免 bf16 读回 + 免 re-quant + dispatch 不做量化（避开 1a 的 +572us）。

**用户目标（看得到收益的 dispatch+GEMM1 融合）**：在 fp8 transport 之上，把 a1 的 gather+preshuffle **折进 GEMM1 的 A-load prologue**——GEMM1 按 `topids_to_rows`/`psum` 从 fp8 `recv_x` 直接 gather 源行 + 片上 preshuffle 进 LDS/寄存器，**消除独立 a1 kernel launch + grouped a1 的一整趟 HBM 往返**（写 `contiguous_m×hidden` fp8 + 再读回）。这是本阶段"dispatch→GEMM1 融合"的可见收益点，改动集中在 `batched_gemm_mxfp4` 的 TDM a8w4 A-loader。

**终极**：Phase 3 整层巨核（§7），dispatch warps 把 fp8 ring 直喂 GEMM1，连 dispatch→L1 的 HBM ring 也省掉。

**弃用**：1a 的 in-dispatch per-token 量化 + `disp_out_q`/`disp_out_qscale` 双写（`AITER_EP_FUSE_DISPATCH_GEMM1`）。

## 8.8 T-B（GEMM1 A-load gather）e2e 实测结论（2026-07-29，4×gfx1250，e384/k6/hd7168/id3072/scatter_fused/layers=2）

T-B 把 a1 的 gather 折进 GEMM1 的 A-load prologue（`AITER_EP_FP8_TRANSPORT=1 AITER_EP_A_GATHER=1`）已端到端跑通并数值正确（`MEGA-CHECK PASS`，logits_diff=0.002158，与纯 fp8 transport 逐位一致）。但**性能是严重负优化**，A/B（per_call，avg/4 ranks）：

| kernel (per_call) | baseline | T-A (fp8 transport) | **T-B (a_gather)** | Δ vs T-A |
|---|---|---|---|---|
| gemm1 `...K7168` | 176.5 | 177.6 | **639.0** | **+461 (+3.6×)** |
| gemm2 `...K3072` | 129.1 | 107.9 | 104.9 | ~0 |
| `ep_dispatch_0` | 108.1 | 91.8 | 97.8 | +6 |
| combine | 68.9 (fused) | 134.8 | 112.1 | -23 |
| **per_layer wall** | **518.5** | **538.0** | **1098.5** | **+560 (+2.0×)** |
| device / layer | 555.6 | 547.5 | **1053.7** | +506 |

**根因**：正是 T-B.1 feasibility verdict 预言的架构冲突落地：
1. **wave-spec 流水线被关**：`mxfp4_preshuffle_gfx1250_tdm.py:288` `WAVE_SPEC = ... and not ep_a_gather` —— 一开 gather，4-stream wave-specialized 流水线整个关闭。
2. **coalescing 崩塌**：A-load 从"单块连续 TDM 2D 矩形（`[tile_m, packed_tile_k]`）"退化为逐组 gather（`tensor_load_gather`，8 行/32b-index 条），失去大块连续访存。
- 两者叠加 → GEMM1 的 A-load 从"一发大块 + 流水线隐藏"退化成"零散 gather + 无流水线"，gemm1 +3.6×。而这还是在 a1_payload 仍照常物化（scale-only 未拆）的前提下——即使拆了 scale-only 省掉一趟 `contiguous_m×hidden` fp8 写，也远补不回 gemm1 的 +461us/call。

**结论**：T-B（把 gather 折进 GEMM1 A-load）在当前"单块 TDM + wave-specialized"架构下**走不通**，与 1a 同属"机制正确、数值 byte-exact、性能负优化"类结果。**T-B（A-load gather）废弃**（代码保留 `AITER_EP_A_GATHER` 开关，默认关，作负结果实证；机制/rowmap 单测 T-B.3/B.4 仍绿，留作 Phase 2/3 dispatch warp→gemm ring 的组件参考）。

**后续方向（ROI 排序）**：
1. **修 combine 非对称 dtype 限制**（恢复 fp8-transport 下的 `scatter_fused` 融合 combine）：当前 fp8 transport 强制 combine 从 `ep_combine_fused`(68.9) 退回 `ep_combine`(134.8)，是 +66us/call 的实打实回退，收益面 > T-B。
2. T-B fallback：dispatch 现有 grid 内第二遍把到达序 fp8 拷成 grouped 连续 a1（省 kernel launch 不省 HBM），低风险低收益。
3. Phase 3 整层巨核（§7）：dispatch warps 把 fp8 ring 直喂 GEMM1，从根上绕开"gather 破坏单块 TDM"的矛盾（gather 发生在 ring 生产侧而非 gemm A-load 侧）。

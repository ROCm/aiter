# comm_fused_moe Stage2 + Reduce-Scatter (DSV4 DPA TP-MoE) 适配计划

## 1. 背景与目标

DSV4 DPA（Data Parallel Attention）场景下，MoE 走 **TP**（权重按 TP 切分），
attention 走 DP。每个 decoder layer 的 MoE 通信模式为：

```
attn(DP, 本地 token 分片) → AG(收集全量 token hidden+router)
→ MoE stage1(TP 权重) → act → MoE stage2(TP 权重)
→ RS(跨 TP 归约 partial，按 token 维度散回各 rank 的 DP 分片)
→ residual → 下一层 attn ...
```

- 输入侧 AG 后，**全部 TP rank 持有相同的全量 token 集合**（行序按源 rank 排列：
  行 `[r*M/tp, (r+1)*M/tp)` 属于 rank r）。
- 每个 rank 用自己的 TP 权重分片对全部 M 个 token 计算 stage2，得到 **partial 和**。
- stage2 之后需要 **reduce-scatter**：跨 TP 求和，rank r 只保留自己拥有的
  token 行段，输出形状 `[M/tp, H]`（紧凑 shard，非全量）。

原有 comm_fused_moe 支持 Stage2 + AllReduce，但不匹配 DPA 输出布局：

| 模式 | 语义 | 输出 | 适用 |
| --- | --- | --- | --- |
| `direct`/`rsag`/`rs_broadcast` (AR) | stage2 + AllReduce | 每 rank 全量 `[M, H]` | TP-MoE，token 不复制的场景 |
| **本次新增 `rs`** | **stage2 + Reduce-Scatter** | **每 rank `[M/tp, H]` 自有 token 分片** | **DSV4 DPA TP-MoE** |

目标：在 gfx950 a8w4 megakernel 中新增 `collective="rs"`，单 kernel 完成
stage2 + RS，性能不低于「未融合 stage2 + NCCL reduce_scatter」基线。

## 2. 关键发现：rsag 的 RS 阶段即为所需归约

现有 `rsag` collective 用「reduce-scatter + all-gather」两阶段在 kernel 内实现
AllReduce 语义：

1. **RS 阶段**（`emit_reduce_scatter_items`）：rank r 通过 `rank_ready` 握手确认
   所有 peer 的 MXFP8 partial 就绪后，**拉取全部 peer 对自己拥有的 token 行段
   `token = rank*shard_tokens + shard_token` 的 partial**，f32 归约，然后
   - bf16 写入全量输出中属于自己的行段；
   - MXFP8 量化写入 `reduced_resource`（供 AG 阶段交换）。
2. **AG 阶段**（`emit_reduced_exchange` + gather）：从 peer 的 reduced shard
   拉取其余行段，拼出全量 `[M, H]`。

因此 **DPA 的 stage2+RS = rsag 截断到 RS 阶段**，差异仅三点：

- bf16 结果写 **紧凑 shard 布局** `[M/tp, H]`（`shard_token*H + ...`），
  而非全量布局中的全局行（`token*H + ...`）；
- 跳过 MXFP8 `reduced_resource` 写入（无 AG 阶段消费）；
- 跳过 `emit_reduced_exchange` 握手与整个 gather 阶段，改为发布
  `gather_done` ack（表示「我已读完所有 peer 的 partial」）后直接收尾。

### AR 变体评估（RS + 下一层 AG 合并）

数学上 RS + AG = AR。现有 `rsag` 本身就是「一次 kernel 内完成 RS+AG」的 AR 变体：
若模型布局允许消费**复制全量输出**，用 `rsag` 可在 MoE 出口直接得到全量结果，
省掉后续一次独立 AG。

但在 DSV4 DPA 中，MoE 的 RS 与下一层 MoE 的 AG 之间隔着 residual + **DP
attention**（每层序列：MoE-RS → attn(本地分片) → AG → 下一层 MoE），下一层 AG
收集的是 attention 之后的新 hidden，**两者数据不同，无法跨层合并**；除非把
attention 也改为 TP（放弃 DPA），代价过大。结论：DPA 应使用纯 `rs`；同时实测
`rs` vs `rsag` vs 未融合，量化 AR 变体在「布局可消费全量输出」时的潜在收益，
作为数据记录。

## 3. 同步与 WAR 安全性分析（复用 rsag 机制）

- **partial 就绪（RAW）**：复用既有 `rank_ready` 握手（service_groups==1 的公共
  路径，与 direct/rsag 相同），保证拉取前所有 peer partial 已发布。
- **partial 覆写（WAR）**：复用 `uses_rsag` 入口节流——每个 rank 在覆盖
  slot 复用的 partial 前，等待所有 peer 的 `gather_done` ack ≥ expected-SLOTS
  （SLOTS=2 双缓冲）。`rs` 在完成 RS 读取后（`s_waitcnt`+barrier 之后）调用
  `emit_gather_ack()` 发布 ack，语义与 rsag/rs_broadcast 一致；routes 模式下
  本地不等待 ack（等待推迟到下一次 launch 入口，留出 2 个 epoch slack）。
- **输出区**：`rs` 的输出 shard 只被本 rank 写、kernel 结束后由 host 读，
  无跨 rank 竞争，不需要 AG 模式的 epoch 入口节流。
- **epoch/状态复位**：service_groups==1 时走末尾 `reset_tile_state_values`
  （producer 计数清零 + epoch 自增），与 direct 路径相同。

## 4. 改动清单

### M1 config + megakernel（本仓核心）

`aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py`
- `collective` 允许 `"rs"`；校验：`rs` 要求 `service_groups==1`（grouped 后续支持）、
  `producer_mode="routes"`；`m % tp == 0` 与 vector 均分校验经由
  `uses_rsag` 自动生效。
- `uses_rsag` 属性纳入 `"rs"`（复用 slotted partial、入口节流、`gather_done`
  区域分配）。
- `output_region_bytes`：`rs` 时为 `payload_bytes // tp`（紧凑 shard）。
- `reduced_payload_bytes`：`rs` 时返回 0（不分配 AG 交换用的 reduced 区域，
  workspace 布局自动前移，kernel 侧 `reduced_resource` 仅在 `collective=="rsag"`
  时创建，互不干扰）。

`aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/megakernel.py`
- `emit_reduce_scatter_items`：`rs` 分支 bf16 写 `shard_token*H + n_tile*tile_n +
  vector_lane*vw` 的紧凑偏移；跳过 MXFP8 reduced 写。
- `emit_rsag_reduce`：`rs` 在 RS 项完成后调用 `publish_gather_completion()`
  （sg==1 → `emit_gather_ack`）并 `return`，跳过 exchange/gather。
- 分发处 `uses_rsag` 已覆盖 `rs`，sg==1 时无需 `wait_for_collective`。

### M2 host / runtime / 测试

`aiter/ops/flydsl/comm_fused_moe_host.py`
- kernel 名 tag：`"rs"` ↔ `collective="rs"` 双向（注意与数值 tag `rs<cache_modifier>`
  的解析顺序：精确匹配优先，无冲突）。
- `ShapeKey.comm` 增加 `"rs"`；CSV 的同一 ordinary shape 行通过一个
  `comm_fused_configs` JSON 字段同时保存 `ar` 与 `rs` 配置。
- `create_flydsl_comm_fused_runners(comm="rs")`；runner 输出视图 `[m//tp, H]`。

`aiter/ops/comm_fused_moe_runtime.py`
- `comm="rs"` 路径：输出为本 rank shard 视图；`add_shared` 的 padding 行写本 rank
  shard（与 AG 的 shared_partial 处理对齐）。

`op_tests/multigpu_tests/test_comm_fused_moe.py`
- `test_comm_fused_stage2_reducescatter`：全 rank 相同 token 输入（模拟 AG 后的
  复制布局），reference = 未融合 stage2 全量输出 → `dist.reduce_scatter_tensor`
  等效切片，逐 rank 比较自有 shard。
- M=8/16 的 eager/graph RS case 均覆盖 uniform/skew，并对 graph workspace 做
  poisoned replay；DPA 的 runtime 接线由 ATOM 定向单测和整网 smoke 覆盖。
- 性能：graph 模式下对比 unfused（stage2 + NCCL reduce_scatter）、`rs`、
  以及 AR 变体 `rsag`（数据记录用）。

### M3 tuner / CSV
- tuner 增加 `--comm-mode rs`；`write_winner` 根据 mode 更新同一 ordinary 行的
  `comm_fused_configs` 项，不再复制 ordinary 前缀生成并列行。

### M4 ATOM 集成
- ATOM DSV4 DPA 路径在 `dp_gather_hidden_and_router`（AG）之后，以 DP collective
  group 创建 `comm="rs", add_shared=False` 的 fused runner；它只替换 routed
  GEMM2 和尾部独立 RS。GEMM1、ordinary GEMM2 以及 rank-local shared expert
  路径保持不变。

## 5. 验证计划

- 正确性：MI350 TP8、DSV4 形状（H=7168, I=384, E=384, topk=6 的 stage2 侧），
  M ∈ {8, 16, 32, 64, ...}（含 TP 整数倍与 runtime padding 场景），
  err 阈值与既有 AR/AG 用例一致。
- 性能：graph capture 模式，unfused（gemm2 + NCCL reduce_scatter_tensor /
  ca_comm 等价物）vs fused `rs`；附 `rsag`（AR 变体）参考数据。
  目标：fused `rs` 不劣于 unfused，小 M 预期 1.3x+。
- 回归：既有 AR 用例与新增 RS 用例全绿。

## 6. 里程碑

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M1 | config + megakernel `rs` 路径 | 已完成 |
| M2 | host/runtime/测试 + GPU 正确性 | 已完成（MI350 TP8，M=8/16 全绿） |
| M3 | 性能对比 + RS tuner/独立 RS winner | 已完成（M8 cg32，M16 cg96） |
| M4 | ATOM DSV4 DPA 接入与整网验证 | 已完成首轮接入与 C2 稳态验证 |

## 7. 初步实测（MI350 gfx950, TP8, DSV4 形状 H=7168/I=384/E=384/topk=6）

graph 模式（生产路径），RS 配置由 direct AR winner 派生（未专属调优）：

| 用例 | 未融合 stage2+NCCL RS | 融合 RS | 加速比 | 融合 AR(direct, 参考) |
| --- | --- | --- | --- | --- |
| M=8 uniform | 51.9us | 35.8us | 1.45x | 35.1us |
| M=8 skew | 41.1us | 29.6us | 1.39x | 23.9us |
| M=16 uniform | 60.9us | 47.4us | 1.28x | 52.8us |
| M=16 skew | 47.2us | 30.8us | 1.53x | 39.0us |

结论：
- 融合 RS 全部快于未融合基线（1.28–1.53x），满足「不劣于融合前」。
- 融合 RS 约等于融合 AR（direct winner 同配置互换 collective）：这些 M 下
  kernel 以 GEMM 为主，RS 少做 (tp-1)/tp 的远端读取尚未体现优势；AR 变体
 （全量输出 + 本地切片）在 DPA 中性能与 RS 相当，可作为布局允许时的替代。
- 后续专属调优确认 M=8 应从 cg48 降到 cg32；M=16 上调 compute groups 只有
  约 0.3%–1% uniform 收益，却明显伤害 skew，因此保留 cg96。

## 8. 当前验证状态（2026-09-14）

RS 已写入 ordinary 行的 `comm_fused_configs["rs"]` 项，不再依赖运行时临时
改写 AR winner，也不会给 ordinary selector 制造重复 shape：

- M=8：`t32x256x128_cg32_v8_bnt2_rs`
- M=16：`t32x256x128_v8_bnt2_rs`（默认 cg96）

M=8 的长稳态同进程 A/B 中，cg32 相对原 cg48：

| 路由 | cg48 | cg32 | cg32 收益 |
| --- | ---: | ---: | ---: |
| uniform | 23.7029 us | 22.2125 us | 6.3% |
| skew | 15.3294 us | 14.5610 us | 5.0% |

cg36 虽在一次 uniform 窄搜中达到 22.0423 us，但 skew 恶化到 19.0030 us；
整网 C2 也只有 91.71 tok/s，低于 cg32 的两轮稳定中位 92.50 tok/s，因此未采用。

最新 8 卡 graph correctness/performance 回归通过：

```text
COMM_FUSED_UT_OK stage2_cases=0 stage2_rs_cases=8 runtime_cases=0
```

| 用例 | ordinary + custom RS | fused RS | 加速比 |
| --- | ---: | ---: | ---: |
| M=8 uniform | 38.6404 us | 31.0805 us | 1.24x |
| M=8 skew | 32.9044 us | 28.0363 us | 1.17x |
| M=16 uniform | 48.7086 us | 42.7965 us | 1.14x |
| M=16 skew | 34.1365 us | 26.6525 us | 1.28x |

ATOM DSV4 DPA C2（固定输入 128、输出 64、80 请求）的 cg32 两轮稳定中位：

| 指标 | ordinary | fused RS cg32 | 收益 |
| --- | ---: | ---: | ---: |
| output throughput | 88.9346 tok/s | 92.5046 tok/s | +4.01% |
| mean TPOT | 19.7351 ms | 18.8991 ms | -4.24% |
| median TPOT | 19.5917 ms | 18.8893 ms | -3.59% |
| mean E2EL | 1438.9364 ms | 1383.2359 ms | -3.87% |

该 C2 对比必须保持 `ATOM_PREFILL_DECODE_INTERVAL=10`。若遗漏，默认 prefill
coalescer 会在低并发下把请求近似串行化，吞吐会降到约 59 tok/s；这是服务调度
配置差异，不是 fused kernel 性能回退。

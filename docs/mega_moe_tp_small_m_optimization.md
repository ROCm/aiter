# Mega TP 小 M 优化：实施计划与验证

本文记录基于最新 main `c53e9b3f42` 和 benchmark 提交 `80ffd42b02` 的小 M 优化。采用在 Mega 上修改尾部的方案，保留 GEMM、LB 路由、reduce-scatter 和尾部 FP8 all-gather；不移植 comm_fused_moe 的 GEMM2 producer，也不引入 CSV 分支选择。

## 1. 实施范围与计划

默认在 **全局 M≤32、LB、融合 tail** 时选择新路径。选择依据是实际 M，而不是 engine 的 `max_local_tokens`；同一 engine 可以同时服务小 M 和大 M。无 tail 的 forward、AR 模式、非 LB 调度及 M>32 继续使用原算法。

| 项目 | 改动 | 验证方法 |
| --- | --- | --- |
| 整行最终归约与 norm/quant 融合 | 每行一个 CTA 直接读取部分和，完成 FP32 归约、BF16 舍入、residual add、RMSNorm 和 per-token FP8 quant | 同一输入与原路径逐位比较 y、residual、FP8、scale 和可选 BF16 输出 |
| 并行 LB 收尾 | CTA0 的 8 个线程分别等待 8 个 XCD 的 LBR 计数，再由线程 0 更新 LBSEQ | 变化输入及空路由下连续 graph replay，检查 watchdog 和结果 |
| 小 M 轮询与访存 | 通信轮询休眠从 16 降到 1；整行路径提前读取 residual 和 norm 权重 | 按开关消融，比较同进程 GPU 图计时及完整 E2E |
| 大 M 回归 | M>32 不启用上述尾部变化 | M=64、256 等正确性检查和 E2E 对照 |

诊断开关为 `AITER_MEGAMOE_TP_SMALL_M_OPT`，在 engine 创建时读取，默认值为 3：

| 位值 | 功能 |
| ---: | --- |
| 1 | 整行最终归约与 norm/quant 融合，包含 residual/权重提前读取 |
| 2 | 并行 LB 收尾 |
| 4 | 缩短小 M 通信轮询休眠 |
| 0 | 使用原路径 |
| 3 | 默认：整行融合和并行收尾 |
| 7 | 三项全部启用 |

每种配置具有不同的 kernel 名称和 launcher cache key，支持独立比较。开关只作用于融合 tail 的小输入 variant，不改变独立 GEMM kernel 的配置。

## 2. 尾部数据流的变化

原路径：

```text
各列块部分和 RDY
  → 多个 CTA 分列完成最终归约
  → 写 BF16 y
  → CTRL_TNC 行计数汇合
  → 最后到达的 CTA 读回整行 y
  → residual add、RMSNorm、per-token FP8 quant
  → FP8 all-gather
```

新整行路径：

```text
推送工作继续由原通信 helpers 完成
  → 每行的 CTA 等待所有列块、所有 rank 的 RDY
  → 直接读取部分和，在寄存器中完成最终归约
  → 舍入到 BF16，然后 residual add、RMSNorm、per-token FP8 quant
  → FP8 all-gather
```

新路径将通信 helpers 的最终归约任务标记为不需要，由 `tn_small_rows` 负责整行归约，避免重复计算。只有行所属 CTA 轮询远端 RDY，其他 CTA 完成原推送工作后进入原输出完成协议。

`forward()` 仍需返回 y，因此新路径保留 y 的写入，但 norm 直接使用寄存器中的 BF16 舍入结果，**省掉 y 的读回和 CTRL_TNC 汇合**。没有省掉尾部 FP8 all-gather，因此仍存在其固定延迟。

## 3. 正确性约束

- 共用 `part_sum` 保持原来 p=0..TP−1 的 FP32 求和顺序，以及 MXFP8/BF16 通信部分和的解码方式。
- 归约结果先转换为 BF16，再与 residual 在 FP32 中相加；norm 使用该 FP32 相加结果，保留原有 norm 输出到 BF16 后再做 FP8 quant 的舍入顺序。
- 每个 row CTA 先等待全部列块和 rank 的 RDY，再用系统一致性读取部分和。RDY 的发布规则和推送完成计数保持原样。
- 每行只更新所属卡的 residual；FP8、scale 和可选 BF16 norm 输出仍通过原 all-gather 写给各卡。
- 并行收尾仍等待全部 CTA 的 LBR 到达，再修改 LBSEQ。没有删除这一先行发生约束，也没有修改双 bank 清零规则。
- 不引入 kernel 级全局屏障，其他 CTA 的 GEMM/推送工作可以继续推进。

代码位置：

- [按实际 M 选择配置](../aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp.py)
- [编译期配置与名称](../aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py)
- [最终部分和读取、helper 工作分配及轮询](../aiter/ops/flydsl/kernels/mega_moe_tp/communication.py)
- [整行归约与 norm/quant](../aiter/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_kernel.py)
- [并行 LB 收尾](../aiter/ops/flydsl/kernels/mega_moe_tp/schedule.py)

## 4. 正确性验证

新增 [test_mega_moe_tp_small_m.py](../op_tests/multigpu_tests/test_mega_moe_tp_small_m.py)，在同一 engine、相同输入上比较原路径和优化 variant：

- Eager 调用与交替 replay 的 CUDA Graph。
- 每次 replay 改变输入、残差、expert IDs 和权重，包含所有 expert ID 被屏蔽的空路由。
- GemmaRMSNorm＋FP8 输出，以及 RMSNorm＋FP8/BF16 双输出。
- 逐位比较 y、residual、FP8、scale 和可选 BF16 输出，并检查各 rank watchdog。
- 多个 M 共用较大容量 engine，验证小 M 选择与容量无关，以及 M>32 自动回到原路径。

对照测试在捕获每个 variant 的 graph 前调用现有 `prepare()`，完成跨卡预热和 watchdog 检查。实施中曾有一次未使用 `prepare()` 的冷启动，在原路径 mask=0、第一轮 eager 就触发 watchdog；补齐预热流程后的默认配置对照通过。该失败没有作为优化性能数据使用。

已通过的运行范围：

| 范围 | 结果 |
| --- | --- |
| M3 TP4、MT=1、MXFP8 通信、M=8/16/32/64/256、开关 1/2/4/7 | 逐位一致，watchdog=0 |
| M3 TP4、MT=3、BF16 通信、M=16/32、开关 7 | 逐位一致，watchdog=0 |
| DSV3 TP2、H=7168、MT=3、MXFP8 通信、M=16/32、开关 7 | 逐位一致，watchdog=0 |
| M3 TP4、MT=3、MXFP8 通信、M=16/32、开关 3/7 | 逐位一致，watchdog=0 |
| 最终默认 3、M3 TP4、MT=1、M=8/16/32/64/256，调用 prepare 后 | 逐位一致，watchdog=0 |

复现示例：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 HSA_ENABLE_IPC_MODE_LEGACY=1 \
  AITER_USE_SYSTEM_TRITON=1 \
  torchrun --standalone --nproc_per_node=4 \
  op_tests/multigpu_tests/test_mega_moe_tp_small_m.py \
  --tokens 8 16 32 64 256 --variants 1 2 4 7
```

## 5. 性能验证

以下为 2026-10-12 在 GPU 0,1,2,3 上的新测结果。测试在代码未提交的工作区完成；运行中的 `source_commit` 字段记录基线 HEAD `80ffd42b02`，不能单独用于区分原路径和优化路径，需结合这里列出的开关、配置和结果目录。

### 5.1 同进程消融

M3 TP4、MT=1，在同一 engine 中对各配置的 graph replay 计时；使用同一组变化输入，图中包含 residual 重置和 Mega forward，**不包含 E2E 前段的 SpRsNorm/router**。每项预热 10 次，5 轮各 100 次调用，报告最快轮次的最慢 rank，单位 µs。

| M | 原路径 0 | 整行融合 1 | 并行收尾 2 | 轮询 4 | 全开 7 | 0→7 差值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 128.782 | 126.072 | 126.424 | 128.651 | 123.118 | −5.664 |
| 32 | 167.190 | 164.151 | 165.053 | 167.019 | 161.322 | −5.868 |

整行融合和并行收尾各有收益；轮询单独收益较小，组合收益需与调度相互作用一起看。日志：`/tmp/mega_small_m_ablation.log`。脚本加 `--benchmark` 可输出上述类型的计时。

MT=3 下补充比较默认候选 3 与全开 7：

| M | 原路径 0 | 整行融合＋并行收尾 3 | 全开 7 |
| ---: | ---: | ---: | ---: |
| 16 | 130.221 | 125.041 | 125.024 |
| 32 | 175.448 | 169.938 | 170.756 |

M=32 时缩短轮询略慢，因此最终默认值采用 **3**；4 号位保留为诊断选项，没有默认启用。日志：`/tmp/mega_small_m_mt3_ablation.log`。

最终测试在使用 `prepare()` 后，对 MT=1 的默认配置再做同输入逐位对照和同进程计时：

| M | 原路径 0 | 默认 3 | 差值 |
| ---: | ---: | ---: | ---: |
| 16 | 128.431 | 123.010 | −5.421 |
| 32 | 168.799 | 162.588 | −6.211 |

变化输入和交替 graph replay 的 y/residual/FP8/scale/BF16 逐位一致，watchdog=0。日志：`/tmp/mega_small_m_default3_equivalence_prepared.log`。

### 5.2 完整 E2E 与后段

M3 TP4、均衡 routed experts、shared expert 保留、lb_min=M、lb_q=3、npp=2。不同开关在相邻轮次运行，存在测量波动；E2E、trace 与 stamp 本身也来自不同轮次，不能把分段差值精确相加当作 E2E 收益。

最终默认配置 **3** 的完整四阶段复核结果如下，MT=1，单位 µs：

| M | Mega E2E 原路径 | Mega E2E 默认 3 | 变化 | Mega 后段原路径 | Mega 后段默认 3 | 同轮 Split E2E / 后段 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 141.9 | 137.9 | −4.0 | 18.3 | 13.0 | 120.9 / 10.9 |
| 32 | 207.6 | 198.4 | −9.2 | 20.7 | 15.3 | 210.9 / 11.9 |

默认配置的 q/residual rel_l2 为 M=16 的 0.1184/0.1154、M=32 的 0.1171/0.1142，watchdog=0。结果来自 `/tmp/mega_small_m_default3/M{16,32}`。原路径来自下述 MT=1 基线目录，运行轮次不同。

实施过程中全开配置 **7** 的对照结果也保留如下：

| M | MT | Mega E2E 原路径 | Mega E2E 全开 | 变化 | Mega 后段原路径 | Mega 后段全开 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 141.9 | 137.8 | −4.1 | 18.3 | 13.4 |
| 32 | 1 | 207.6 | 200.5 | −7.1 | 20.7 | 18.3 |
| 16 | 3 | 149.9 | 144.1 | −5.8 | 未分段 | 未分段 |
| 32 | 3 | 214.9 | 211.8 | −3.1 | 未分段 | 未分段 |

MT=1 的原路径分别来自 `/tmp/mega_small_m_baseline_plan/M16` 和 `/tmp/m3_tp4_balanced_M32_mt1_gpu0123`；优化结果来自 `/tmp/mega_small_m_opt7/M16` 和 `/tmp/mega_small_m_opt7/M32`。MT=3 的结果来自 `/tmp/mega_small_m_mt3/M{16,32}_opt{0,7}`。

所有 E2E 运行均通过现有 0.15 门限，q/residual rel_l2 大约为 0.117–0.119 / 0.114–0.115，watchdog=0。相邻 E2E 轮次的微小误差差异不能替代同输入逐位对照；第 4 节已另行验证逐位一致。

### 5.3 大 M 对照

M=256、MT=1，开关设为 0/7 时实际配置均为 `small_m_opt=0`，Mega E2E 为 243.7 / 242.2 µs，watchdog=0；没有观察到回归。结果在 `/tmp/mega_small_m_regression/M256_opt{0,7}`。

新路径并未让所有小 M 的完整 E2E 都快于 Split：例如 M=16 仍有前段和 GEMM/调度差距，M=32 的后段仍高于 Split 的约 12–14 µs。本次验证支持减少固定尾部开销，不能据此声称已经消除了全部小 M 性能差距。

## 6. 原路径与优化路径的对照命令

```bash
# 原路径，单独保留结果目录。
AITER_MEGAMOE_TP_SMALL_M_OPT=0 AITER_MEGAMOE_TP_LB_MT_SMALL=1 \
  M3_TP4_OUTPUT_DIR=/tmp/m3_tp4_M32_original \
  bash op_tests/op_benchmarks/flydsl/run_m3_tp4_balanced_e2e.sh 32 0,1,2,3 all

# 默认优化路径（3），可以省略 SMALL_M_OPT。
AITER_MEGAMOE_TP_SMALL_M_OPT=3 AITER_MEGAMOE_TP_LB_MT_SMALL=1 \
  M3_TP4_OUTPUT_DIR=/tmp/m3_tp4_M32_small_tail \
  bash op_tests/op_benchmarks/flydsl/run_m3_tp4_balanced_e2e.sh 32 0,1,2,3 all
```

后续若继续降低后段，应先细分仍保留的推送握手和 FP8 all-gather，再决定是否启用 LB 的低延迟 RS 包或调整完整 residual 的传播。它们没有包含在本次改动中。

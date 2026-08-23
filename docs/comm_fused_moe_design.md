# 通算融合 MoE：轻量生产架构与离线 Tuner

## 1. 目标

普通 MoE 已经有成熟的 routing、sort、Stage1、量化和权重布局。通算融合只替换：

```text
Stage2 GEMM + shared partial + TP communication
```

第一版必须同时满足：

- 现有 M=256～32768 的精度和性能不回退；
- forward 中不调优、不 fallback、不查复杂配置；
- 新增 M 只增加 winner 数据，不增加一套新代码；
- 新模型、新 shape、Opus 或 ASM 后端可以沿用同一接入边界。

## 2. 三层边界

```text
ATOM
  standard backend 下按模型契约自动选择 CommFusedMoe 容器
  每次 forward 按 exact M winner 选择融合路径，否则走普通 MoE
        ↓
AITer 公共层
  复用普通 MoE 到 Stage1，通过 _stage2_override 接管 Stage2
        ↓
后端 runner
  FlyDSL atomic-compressed / full-width / windowed / persistent-window
  以后可以增加 Opus / ASM
```

ATOM 只在 `moe_backend=standard`、非 online quant、FP4 权重且 exact 模型 shape
受支持时创建 `CommFusedMoe`，不需要额外启动参数。已有 exact M winner 时传
`_stage2_override`；没有 winner 时在模型层继续走普通 Stage2 + AllReduce，不进入
融合 runtime。M=1 decode 同时保留普通 MoE 的 shared/routed dual-stream。

## 3. 当前文件职责

| 文件 | 唯一职责 |
| --- | --- |
| `aiter/ops/flydsl/comm_fused_moe_host.py` | CSV exact lookup、通信资源、四种真实 pipeline 和 lazy runner cache |
| `op_tests/multigpu_tests/tune_comm_fused_moe.py` | 离线候选、完整 pipeline 测量、精度 gate 和 CSV 输出 |
| `aiter/configs/comm_fused_moe.csv` | 所有 production winner |
| `aiter/ops/flydsl/kernels/comm_fused_moe/*.py` | 四种 GPU 算法，不保存 production bucket 表 |

生产包只保留 host、winner CSV 和 kernel。离线 tuner 位于 `op_tests`，不会被生产路径导入。
没有独立 runners package，也不再为 spec、factory、common 和每个 runner 建文件。GPU 算法
仍按 atomic/full/window/persistent 分成四个 kernel 文件。

这里没有为了压低 `host.py` 行数把代码搬到别处。buffer 注册、launch 顺序、barrier 和四种
pipeline 都是真实 host runtime，因此保留在一个文件中；新增 shape/M 不会继续增加这些代码。

## 4. Production winner 数据

生产路径只做两层 exact 映射：

```text
ShapeKey
  gfx
  H
  I_per_tp
  experts
  topk
  TP
       ↓
padded M → winner row → kernel Config
```

量化契约不重复写入 CSV：ATOM 在初始化时已限定非 online quant 的 FP4 权重，
runner/kernel 内部则是固定的 MXFP8 通信协议。未来如果同一 shape 需要支持多种量化
契约，再把 dtype/quant 加入 key，第一版不预留无用列。

winner 直接保存在一个 CSV 中，每行对应一个 exact ShapeKey/M：

```csv
gfx,...,m,kid,family,tile_m,tile_n,tile_k,sort_block_m,window,...
gfx950,...,256,atomic_...,atomic,,,,,,...
gfx950,...,8192,window_tm64_...,window,64,256,128,64,1024,...
gfx950,...,32768,persistent_tm64_...,persistent,64,256,128,64,1024,...
```

KID 只保存独立调优参数：

```text
family
tile_m / tile_n / tile_k
sort_block_m
window
local_workers
reduce-scatter grid
all-gather grid
service grid
```

下面这些值必须由代码推导，不能写进 winner：

```text
shard_rows
tiles_per_window
phases
workspace bytes
payload / scale stride
epoch / gate / ready offset
```

初始化时使用标准库 `csv` 读取一次并缓存。CSV 行直接构造对应 kernel 文件定义的
`Config`；KID 只是可读、可追踪的参数标识，不再额外维护一套 `Spec`。forward 只访问已经
构造好的 runner，不解析 CSV。新增 M 只增加一行 winner。

## 5. 为什么不用现有重型 tuner

普通 FMoE 的 `TunerCommon` 解决 CSV 合并、pandas、多进程、在线补调、fallback 和多算子
兼容，远大于当前需要。MegaMoE 的大量 shape 启发式也不适合作为 exact production
winner。

本设计只借鉴三点：

- Opus：torch-free、结构化 KID metadata；
- FlyDSL：显式 `kernel name -> compile params` 候选；
- Triton：winner 数据和算法代码分离，启动时缓存。

不恢复旧 `moe_tp_stage2_tuner.py`，生产路径也不 import tuner。

## 6. 轻量离线 tuner

`op_tests/multigpu_tests/tune_comm_fused_moe.py` 提供五类能力：

1. `atomic_compressed_candidates(...)`
2. `full_width_candidates(...)`
3. `windowed_candidates(...)`
4. `persistent_window_candidates(...)`
5. `benchmark(...) / select_winner(...) / winner_row(...) / write_winner(...)`

候选生成器只做独立参数的笛卡尔积，不把派生 layout 值暴露给调参脚本。tuner 和生产端
直接复用 `full_width.Config`、`windowed.Config`、`persistent_window.Config`，不存在重复的
tuner spec 数据模型。

单个 candidate 的测量流程固定为：

```text
用 production create_runner 创建 runner
  → 运行完整 Stage2 + shared partial + TP communication
  → 对比普通 Stage2 + shared + BF16 AllReduce reference
  → max_abs / rel_l2 精度 gate
  → 用 TP graph_capture 捕获完整 pipeline
  → 多轮计时并取 TP8 rank-max median
  → 输出 TuningResult
```

`winner_row()` 使用和 production 完全相同的 CSV schema，`write_winner()` 按
ShapeKey/M 更新目标 CSV。production CSV 只包含：

```text
M
KID
family
独立 kernel Config 参数
```

`latency_us / max_abs / rel_l2` 保留在 `TuningResult` 和仓外 sweep 报告，不写入生产 CSV。
promotion 时把 tuner 生成的一行 winner 合入 production CSV，不再人工改 Python
KIDS/BUCKETS。

为了避免非法 candidate 造成 GPU context poisoning，正式大 sweep 推荐由仓外 driver
以“一个 candidate 一个 TP8 进程组”运行。这个故障隔离不进入 production，也不需要在
AITer 内再建设一套多进程 tuner framework。

## 7. 新 shape 的接入流程

1. 定义 exact `ShapeKey`。
2. 列出需要覆盖的 padded M。
3. 为每个 M 生成 atomic/full/window/persistent 候选。
4. 使用相同输入建立普通路径 reference。
5. 每个 candidate 跑完整 pipeline 精度和 Graph rank-max。
6. 选择无精度问题的最快结果。
7. tuner 生成或更新一行 winner CSV。
8. 初始化时 exact lookup 该 ShapeKey/M。
9. 跑全 bucket、uniform/skew、eager/Graph 和整网验证。

没有 winner 的 shape 在初始化时保持普通 MoE；没有 winner 的 M 在模型 forward 处
走普通 Stage2 + AllReduce。融合 backend 内不做最近邻、向上取 bucket、自动 family
切换或内部 fallback。

## 8. 当前验证结果

MI355X/gfx950、TP8，同一节点完成结构重构后的全量验证：

| M | uniform Graph us | skew Graph us | max_abs | rel_l2 约值 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 136.63 | 102.96 | 0.8125 | 0.0313 |
| 512 | 151.17 | 148.75 | 0.8125 | 0.0313 |
| 1024 | 181.43 | 184.76 | 0.7500 | 0.0313 |
| 2048 | 245.45 | 251.79 | 0.7500 | 0.0300 |
| 4096 | 373.69 | 384.46 | 0.7500 | 0.0300 |
| 8192 | 615.26 | 626.23 | 0.8125 | 0.0301 |
| 16384 | 1044.12 | 1087.08 | 0.8750 | 0.0301 |
| 32768 | 1953.16 | 1985.58 | 0.8125 | 0.0301 |

16 组 uniform/skew、eager/Graph 和精度检查全部通过。tuner API 也用 M=256 production
winner 做了 smoke，Graph rank-max 约 `138.91 us`，精度 `0.7500 / 0.031323`。另外，
256/512/8192/16384/32768 在同一 TP8 进程组内依次 lazy 创建 runner 的验证也已通过。
uniform M=8192/32768 另用 7 轮、每轮 50 次 Graph replay 复测，分别为
`615.26 / 1953.16 us`，确认首轮不到 1% 的波动是测量噪声。

MORI SDMA、CU push 等实验未超过当前 compressed direct-pull 完整 pipeline，因此不进入
production。MORI 只在初始化时注册 external window，热路径没有 MORI host 调用。

### 最终模型接入验证

- ATOM 已在 `moe_backend=standard` 下自动选择，不再需要
  `--moe-backend comm_fused`。
- M=256/512/1024 走 atomic-compressed，M=2048/4096 走 full-width，
  M=8192/16384 走 windowed，M=32768 走 persistent-window。
- CUDA Graph capture smoke 已通过；kernel trace 已确认 full/window/persistent 使用
  `flydsl_fused_moe_full_*`、`flydsl_fused_moe_win_*`、`flydsl_fused_moe_pwin_*`
  的逻辑名称，不再把某个 M 写入 kernel name。
- 不支持的 M（包括 M=1 decode）保持普通 Stage2 + AllReduce；M=1 的
  shared/routed dual-stream 也已恢复。targeted A/B 中 M=1 TPOT 仅波动 `+0.200%`。
- 全 bucket 整网验证覆盖 M=256～32768：融合支路 TTFT 在 M=512～32768
  改善 `0.55%～12.58%`，M=256 的 `-1.81%` 在 2% 测量波动内；各 bucket
  TPOT 变化在约 `-0.54%～+1.12%` 内。
- 最终 InferenceX-style 同场 A/B（ISL=8192、OSL=1024、concurrency=64、
  640 requests）两侧均 640/640 成功，无 traceback、memory fault、HTTP 5xx 或 hang。

| 整网指标 | baseline | candidate | candidate 变化 |
| --- | ---: | ---: | ---: |
| benchmark duration | 1245.61 s | 1227.73 s | -1.44% |
| total token throughput | 4279.42 tok/s | 4341.72 tok/s | +1.46% |
| median TTFT | 442.57 ms | 407.68 ms | -7.88% |
| P90 / P99 TTFT | 1248.81 / 14095.80 ms | 1081.11 / 13277.86 ms | -13.43% / -5.80% |
| median TPOT | 127.10 ms | 126.06 ms | -0.82% |
| P90 / P99 TPOT | 131.41 / 139.32 ms | 129.77 / 137.61 ms | -1.25% / -1.23% |
| median E2EL | 118530.98 ms | 117585.57 ms | -0.80% |
| P99 E2EL | 144853.61 ms | 142822.52 ms | -1.40% |

median ITL 有 `+0.99%` 的单项波动，P90 基本持平（`+0.12%`），P99 改善
`4.25%`；结合 TPOT、E2EL 和总吞吐的同向改善，该变化归为调度测量波动。

完整结果保存在仓外：

```text
/home/yifehuan/data/comm_fused_moe_final_inferencex_ab_20260823
```

## 9. 扩展 Opus / ASM

新 backend 只需要：

1. 定义自己的 kernel `Config`；
2. 实现相同 runner 调用契约；
3. 在 host 的显式 `Config type -> runner` 映射中注册；
4. 在离线 tuner 中增加候选生成器和 CSV family；
5. promotion 最终 winner 行。

模型层、`fused_moe.py` seam、公共 runtime 和已有 FlyDSL runner 不需要随之修改。

只有真实接入 Opus/ASM 时才增加对应 Config 和 runner 条目；第一版不提前建设通用 plugin
registry。

## 10. 明确不做

- 不在线调优；
- 不在 forward 解析 CSV；
- 不保留实验分支或运行时参数开关；
- 不做模糊 shape 匹配和 fallback；
- 不把 workspace stride、epoch offset 等派生值写入 winner；
- 不让 kernel 文件维护 production bucket；
- 不为尚未接入的 backend 增加抽象。

判断标准：代码必须直接服务当前正确性、当前性能、离线 winner 生成，或下一个已经确定
要接入的 backend。

## 11. 下一步

1. 用 tuner 跑一组小范围已知参数 sweep，持续校验 winner 选择和输出格式。
2. 将仓外历史 sweep driver 改为直接构造 kernel Config，不再 monkey-patch kernel 全局常量。
3. 优先尝试 persistent-window 的 RS/AG 跨 phase overlap，但只 promotion 全 M 和
   整网都通过门禁的 winner。
4. 需要新模型或新 shape 时，按第 7 节流程新增 CSV winner，不扩张 host 热路径。

Persistent 的结构性优化记录见 `docs/comm_fused_moe_persistent_kernel_plan.md`。

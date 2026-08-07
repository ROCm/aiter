# MXFP4 GEMM1 Tuning Best Practices

本文面向 [`mxfp4_gemm1.py`](../aiter/ops/flydsl/kernels/mxfp4_gemm1.py)
当前公开的性能参数：

- `BM`
- `BN`
- `use_nt`
- `xcd_swizzle`
- `inline_quant`
- `interleave`

`BK` 当前固定为 256，不纳入 tuning。

## 1. Tuning 单位

每组配置应按以下 workload 独立 tuning：

```text
GPU 架构、CU 数、tokens、D_HIDDEN、D_INTER、NE、TOPK、
activation、a_dtype、out_dtype、权重布局
```

应使用接近生产环境的专家路由分布。相同 token 数下，均匀路由和热点专家
路由可能选出不同的 `BM` 或 `xcd_swizzle`。

## 2. 合法搜索空间

### 2.1 A4W4

| 模式 | 合法的 `(BM, use_nt, inline_quant)` |
|---|---|
| 预量化 | `(32, True, False)`、`(32, False, False)`、`(64, True, False)`、`(64, False, False)`、`(128, False, False)` |
| 融合量化 | `(16, True, True)` |

对每个合法组合完整搜索：

```text
BN          = {128, 256}
xcd_swizzle = {0, 2, 4}
```

当 separated 与 interleaved 两种 gate/up 预排布都可用时，`BN=128` 必须同时
搜索 `interleave={False,True}`。

当前推荐使用 `{0, 2, 4}` 作为 `xcd_swizzle` 搜索空间。除非实际 workload
仍存在明显的跨 XCD 负载不均，否则不建议盲目扩大搜索范围。

### 2.2 A8W4

当前生产路径建议使用以下搜索空间：

```text
BM=32:       use_nt = {False, True}
BM=64/128:   use_nt = {False}
BN:          256
interleave:  True
xcd_swizzle: {0, 2, 4}
```

## 3. 参数选择原则

| 参数 | 较小值更适合 | 较大值更适合 |
|---|---|---|
| `BM` | 每个专家 token 少、padding 比例高、需要更多 blocks | 专家负载高、希望提高计算密度 |
| `BN` | workload 小、GPU 并行度不足、VGPR 压力较高 | workload 大、希望提高 A 数据复用 |
| `xcd_swizzle` | grid 小或专家负载均匀时使用 0 | 专家负载不均或跨 XCD 调度不均时尝试 2/4 |

### 3.1 `use_nt`

`use_nt=True` 为 B load 使用 non-temporal cache 策略，适合权重复用低的流式
访问；当同一专家权重被较多 M blocks 重复访问时，`use_nt=False` 通常更合适。

small-token workload 不能仅根据 token 数机械地选择 `use_nt=True`。重复访问
同一份 routing 时，少量 active experts 的权重工作集可能进入 L2，
`use_nt=False` 会得到流式访问中不存在的额外收益。因此 small token 应同时
保留 cacheable 和 non-temporal 候选，并在目标 routing/cache 状态下比较。

对于 `BM=32`，host wrapper 会应用以下自动切换：

```text
ceil(tokens * TOPK / 32) >= NE
    => effective use_nt = False
```

因此满足该条件时，`use_nt=True` 和 `use_nt=False` 会落到相同的实际 kernel，
应在生成候选时去重。

### 3.2 `inline_quant`

`inline_quant` 改变的是端到端流水线，而不仅是 GEMM 微配置。比较时必须计入：

- 被消除或新增的量化 kernel；
- BM16 输出 scale 的清零；
- 额外的 kernel launch 和内存访问。

因此，跨 `inline_quant` 模式应按端到端耗时选择，不能只比较 GEMM1 kernel
时间。

### 3.3 `interleave`

`interleave` 改变 gate/up 权重和 scale 的物理排布。只有同时生成了匹配的两种
预排布权重时才可以比较；不能将同一份权重直接传给两个模式。

`BN=128` 和 `BN=256` 均支持 separated 与 interleaved gate/up 布局。

### 3.4 small-token 参数耦合

small token 下有效计算比例很低，固定开销和离散调度效应会被放大。例如
`token=1, TOPK=9, BM=32` 时，约 9 个有效 route 可能占用约
`9 * 32 = 288` 个 padded rows，有效行比例仅约 3%。此时：

- `BN=128` 相比 `BN=256` 产生两倍 N blocks，可能通过增加并行度获益，也可能
  因额外 workgroup 固定开销变慢；
- `xcd_swizzle=2/4` 的分组粒度可能接近整个 grid 大小，少量 workgroup 的放置
  差异会造成明显的跨 XCD 不均；
- 1–4 µs 的绝对差异在 10–20 µs kernel 上会显示为 10%–40% 的相对差异；
- 单一 routing seed 只覆盖很少的 experts，kernel 排名容易随 expert 组合变化。

因此 small token 的 `BN`、`use_nt` 和 `xcd_swizzle` 必须联合比较，不能逐项
套用静态 heuristic。

## 4. `BM` 的上下游耦合

`BM` 同时影响：

- MoE sorting 和 padding；
- GEMM1 grid；
- 中间量化值及 scale 的布局；
- GEMM2 的 block M。

因此：

1. 只优化已有 tuned row 时，应锁定 `BM` 和 GEMM2，仅搜索 GEMM1 的
   `BN`、`use_nt` 和 `xcd_swizzle`。
2. 如果要改变 `BM`，必须重新执行 sorting，并将 GEMM1 与 GEMM2 作为整体
   tuning。
3. 不应使用单独 GEMM1 kernel 时间决定不同 `BM` 之间的胜负。

## 5. 搜索策略

当前公开搜索空间较小，并且参数间交互明显，建议对所有合法组合做穷举，而
不是逐参数 greedy tuning。

推荐顺序：

1. 固定 dtype、activation、量化模式和权重布局。
2. 如果上下游允许，枚举合法 `BM`；否则锁定现有 `BM`。
3. 枚举 `BN`。
4. 枚举有效的 `use_nt`，去除 host wrapper 会映射成相同 kernel 的候选。
5. 枚举 `xcd_swizzle={0,2,4}`。
6. 将 `inline_quant` 和 `interleave` 作为独立的端到端方案比较。

可使用以下配置作为搜索起点，但不能替代实际 benchmark：

| Workload 特征 | 起始配置 |
|---|---|
| 每个专家 token 很少 | `BM=32, BN=128` |
| 中等专家负载 | `BM=64, BN=256` |
| 专家负载较高 | `BM=128, BN=256, use_nt=False` |
| 跨 XCD 负载明显不均 | 在基线基础上尝试 `xcd_swizzle=2/4` |

### 5.1 small-token 搜索空间

对于 `token <= 32`，或平均 expert rows
`tokens * TOPK / NE <= 1` 的 workload，不应只保留基于静态 heuristic 的单一
参数方向。合法时应覆盖：

1. `BN={128,256}`；
2. `use_nt={False,True}`；
3. `xcd_swizzle={0,2,4}`；
4. 至少一个 `use_nt=False, xcd_swizzle=2/4` 组合。

固定一种 `interleave` 布局时，small token 的完整预量化搜索空间最多为
`BN(2) * use_nt(2) * xcd_swizzle(3) = 12` 个候选。该空间较小且交互明显，
应优先完整枚举。若必须缩小空间，不能同时删除 cacheable 候选和
`xcd_swizzle=2/4`。

## 6. 候选比较规则

比较任意一个参数时，必须固定所有非目标参数及 workload 条件：

- 只改变 `BN/use_nt/xcd_swizzle` 时，固定 `BM`、GEMM2、权重布局和 routing；
- 改变 `BM` 时，同时更新 sorting 和 GEMM2 的匹配配置；
- 改变 `inline_quant` 时，计入被消除或新增的量化工作；
- 改变 `interleave` 时，使用与该模式匹配的预排布权重和 scale；
- 比较 `use_nt` 时，使用相同且能代表生产行为的 cache 状态；
- 比较 `xcd_swizzle` 时，覆盖均匀、热点和 sparse expert routing。

候选排序遵循：

1. 锁定 `BM` 和 GEMM2、只优化 GEMM1 时，按 `us1` 排序。
2. 改变 `BM`、`inline_quant`、`interleave` 或 GEMM2 时，按完整路径
   `e2e_us` 排序。
3. 相对差异小于 3% 时视为平局。
4. 对低于 50 us 的 small-token kernel，还应要求至少 2 us 的绝对差异。
5. 平局时优先选择跨 routing seed 和相邻 token bucket 更稳定的参数组合。

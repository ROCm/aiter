# 通算融合 MoE：轻量设计与实现计划

## 1. 一句话目标

保留普通 MoE 已经成熟的 routing、sort、Stage1、量化和权重布局，只替换：

```text
Stage2 GEMM + shared partial + TP AllReduce
```

模型初始化时直接选择普通 `FusedMoE` 或 `CommFusedMoe`。普通路径不经过新
selector，也不承担额外开销。

## 2. 第一版只保留三层

```text
模型层（ATOM）
  选择 FusedMoE / CommFusedMoe
          │
AITer 公共层
  复用普通 MoE 到 Stage1，调用一个 _stage2_override
          │
后端层
  FlyDSL runner；以后可增加 Opus / ASM runner
```

公共层不理解 FlyDSL tile、Opus kernel id 或 ASM 参数。后端只需准备按 token
bucket 索引的 runner，并返回完整 `[M, H]` 结果。

## 3. 核心接口

`aiter/fused_moe.py` 只增加一个内部参数：

```python
_stage2_override: Callable | None = None
```

- `None`：完全执行原 Stage2，返回原 `moe_out`。
- 非 `None`：把 Stage2 参数交给 override；override 直接返回完成通信后的
  `[M, H]`。

不再保留 `_stage2_transform`、`_stage2_returns_output` 两个开关。

公共 `CommFusedMoeRuntime` 只做两件事：

1. 按 token bucket 找已经准备好的 runner。
2. 必要时补齐输入并执行融合 Stage2。

选择 `comm_fused` 后不再静默回退。缺少 runner 说明该模型或 shape 尚未接入，
直接报错。

没有 Protocol、全局 backend registry、在线 tuner、通用 manifest 解析或复杂
运行时校验。

## 4. 为什么普通 MoE 仍需一个小改动

通算融合只替换 Stage2，前面的 routing、sort、Stage1 和中间量化仍属于普通
MoE。复制这些逻辑会产生两套实现，精度和性能配置容易漂移。

因此模型层负责“选哪条路”，`fused_moe.py` 的单一 seam 负责“在哪一行接管
Stage2”。普通模型没有传 `_stage2_override`，执行路径与原来一致。

## 5. 文件边界

### AITer

| 文件 | 作用 |
| --- | --- |
| `aiter/fused_moe.py` | 单一 `_stage2_override` seam |
| `aiter/ops/comm_fused_moe_runtime.py` | 轻量 runner 选择、padding 和调用 |
| `aiter/ops/flydsl/comm_fused_moe.py` | 后续重新实现 FlyDSL runner 准备逻辑 |
| `aiter/configs/model_configs/...` | 只保存离线确认过的 winner |

### ATOM

| 文件 | 作用 |
| --- | --- |
| `atom/model_ops/fused_moe/comm_fused_moe.py` | 模型 adapter，复用 routing 和权重 |
| `atom/models/deepseek_v4.py` | 初始化时选类；融合 shared partial；跳过外层 AR |
| `atom/config.py`、`arg_utils.py` | 暴露 `moe_backend=comm_fused` |

## 6. 新模型或新 shape 如何接入

不要修改公共运行时。只做下面四步：

1. 记录模型契约：`H、I/TP、E、topk、dtype、quant、TP、activation`。
2. 为需要的 token buckets 离线测试候选 kernel。
3. 只把精度和性能通过的 winner 做成 prepared runner。
4. 模型初始化时构造 `CommFusedMoe`，把这些 runners 交给公共层。

只有 runner 覆盖完整服务 token buckets 后才启用 `comm_fused`。缺少 winner 的
模型或 shape 继续在模型配置层选择普通 `FusedMoE`，不会在融合路径里静默回退。

## 7. Tuner 流程

Tuner 不进入线上推理，也不决定模型走哪条路。

```text
生成候选参数
  → 与普通 Stage2 + AR 做精度比较
  → 测完整输出耗时
  → 选择稳定胜出的 winner
  → 写入后端配置
  → 启动时准备 runner
```

第一版 tuner 只需要：shape 输入、候选枚举、正确性比较、计时和 winner 输出。
promotion report、lower-bound 报告、复杂 schema、在线搜索等以后确有需要再加。

FlyDSL、Opus、ASM 各自维护候选参数和 runner 构造；公共层只接收最终 runner。

## 8. Shared expert

DeepSeek-V4 的 shared expert 保持独立计算：

```text
shared W2 partial ─┐
                   ├─ 融合 Stage2 + TP 通信 → 完整输出
routed Stage2 ─────┘
```

shared W2 先生成普通 partial。融合 runner 提供固定 buffer，AITer 公共层在
Stage2 前完成 padding 和复制，再由 runner 完成融合计算与通信。

## 9. 实现步骤

### 第一步：公共 seam 和模型选路

- AITer 分支统一为 `yifehuan/comm_fused_moe`。
- 385 行旧公共文件改为轻量 `comm_fused_moe_runtime.py`。
- 双开关收敛为 `_stage2_override`。
- ATOM 在 DeepSeek-V4 初始化时直接选择 `FusedMoE` 或 `CommFusedMoe`。
- shared expert 保持原 Linear 调用，不修改通用 LinearBase。

状态：已完成公共 seam 和严格模型选路；安装融合 runner 前不会启用
`comm_fused` 执行。

### 第二步：重新实现 FlyDSL backend

- 只读参考备份分支中的 kernel 行为、shape 和已验证参数。
- 不整文件恢复旧 `moe_tp_stage2.py`、旧 tuner 或旧低层文件。
- 新建小型 FlyDSL runner builder，将已准备 runner 交给公共层。
- 首先恢复 DeepSeek-V4 已验证 shape 和 token buckets 的性能。

### 第三步：轻量离线 tuner

- 从一个明确 shape 开始枚举。
- 比较完整 `Stage2 + shared + AR` 输出和耗时。
- 配置文件只保存 winner，不保存运行时不需要的报告字段。

### 第四步：整网验证

- 普通 backend 精度与性能回归。
- `comm_fused` eager、CUDA Graph、随机 M、hash routing 和 shared expert 精度。
- 各 TP 下逐 bucket 对比备份版本的性能，确认没有回退。

### 第五步：扩展 Opus / ASM

- 新后端实现相同 runner 调用契约。
- 模型层和 `fused_moe.py` 不再修改。
- 同一 shape 可离线比较多个后端，只部署最终 winner。

## 10. 第一版明确不做

- 不在线调优。
- 不建设通用插件框架。
- 不恢复旧的重型 tuner 和生产报告体系。
- 不让普通 MoE 经过新 selector。
- 不为尚未出现的后端提前增加抽象层。

判断标准很简单：一行代码如果既不服务当前正确性、当前性能，也不直接支持
下一个 backend runner，就暂时不进入第一版。

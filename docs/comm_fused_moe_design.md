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
| `aiter/ops/flydsl/comm_fused_moe_host.py` | FlyDSL workspace、MORI 资源和 runner launch |
| `aiter/ops/flydsl/kernels/comm_fused_moe/full_width.py` | 当前 full-width winner 的 G/L/R/F kernel |
| `aiter/configs/model_configs/...` | 只保存离线确认过的 winner |

当前 FlyDSL runner 只依赖两类已有基础设施：

- `torch.symm_mem` 提供 cached VMM workspace，MORI CCO external window 提供 peer 地址；
- FlyDSL 增加 system-scope epoch wait，`buffer_ops` 适配当前 ROCDL cache operand。

MORI 和 custom AR 都不需要为融合算子新增接口。FlyDSL 改动由真实 GPU 失败或性能结果
证明是必需项，不属于线上 tuner 或通用框架。

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
- 新建轻量 FlyDSL runner，将已准备 runner 交给公共层。
- 第一版只接入 DeepSeek-V4 TP8 的 M=2048 winner：

```text
G：完整 H 的 Stage2 GEMM
  → L：本 rank route reduce + shared partial + MXFP8
  → epoch
  → R：TP peer reduce + owner BF16/MXFP8
  → epoch
  → F：owner 结果复制到各 TP rank
```

- 固定契约：`M=2048, H=7168, I/TP=384, E=384, topk=6, TP=8`。
- 固定已测调度：`32x256x128, PEER_GRID=128, FANOUT_GRID=126,
  LOCAL_WORKERS=640`。
- cached `torch.symm_mem` tensor 通过已有 MORI CCO external window 映射到 flat VA；kernel
  直接按固定 rank stride 计算 peer 地址，不使用 MORI symmetric heap，也不扩展 custom AR。
- 没有动态 window、service kind、persistent/XCD 分支、在线 fallback、planner 或 tuner。

状态：第一版 production bucket 已完成。runner 只保留 workspace、peer 映射和固定
launch；kernel 的主体是 FP8/MXFP8 解码、归约、量化和 fanout 的实际 GPU 算法，不是
通用框架代码。

GPU 结果（MI355X TP8）：

```text
精度：max_abs=0.75, rel_l2=0.029969
eager：310.360 us → 271.384 us，1.1436x
CUDA Graph：293.277 us → 269.107 us，1.0898x
```

普通 Opus、普通 FlyDSL A8W4 Stage2 和 MegaMoE TP8 回归均通过。更多 token bucket
不在运行时猜参数，按第三步流程逐个离线加入。

### 第三步：轻量离线 tuner

- 从一个明确 shape 开始枚举。
- 比较完整 `Stage2 + shared + AR` 输出和耗时。
- 配置文件只保存 winner，不保存运行时不需要的报告字段。
- 每增加一个 bucket，只新增经过验证的 workspace 尺寸、kernel 编译参数和 runner
  映射；不修改公共 runtime，也不让未覆盖 bucket 静默走普通路径。

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

## 11. M=32768 独立 KID 实现计划

### 11.1 结论与性能基线

M=32768 不在当前 M=2048 单窗口 kernel 内增加运行时分支，而是新增一个独立的
pipeline KID。KID 代表完整的 `G/L/R/F` winner，不只代表 Stage2 GEMM 的 tile 参数。

现有数据：

| 路径 | ordinary Graph | fused Graph | speedup |
| --- | ---: | ---: | ---: |
| 清理前 7-window winner | 3581.112 us | 2272.528 us | 1.5758x |
| 当前 full-H 单窗口 | 3590.766 us | 2688.808 us | 1.3354x |

baseline 基本一致，当前约 18.3% 的回退来自窗口间 overlap 消失。因此不能只修改
`grid`、worker 数量或在现有 kernel 外连续 launch 七次；必须恢复同一次 launch 内的
GEMM worker 与 L/R/F service worker 并发。

### 11.2 KID 选择

第一版使用普通 Python 静态映射，不引入 CSV parser、插件 registry 或在线 tuner：

```python
_KID_BY_SHAPE = {
    (2048, 7168, 384, 384, 6, 8):
        "flydsl_comm_tp8_m2048_full_v1",
    (32768, 7168, 384, 384, 6, 8):
        "flydsl_comm_tp8_m32768_win7_v1",
}

_BUILDERS = {
    "flydsl_comm_tp8_m2048_full_v1": build_m2048_runner,
    "flydsl_comm_tp8_m32768_win7_v1": build_m32768_runner,
}
```

shape key 固定为：

```text
(padded_M, H, I_per_tp, experts, topk, TP)
```

初始化阶段完成 `shape -> KID -> prepared runner`。运行时继续只做现有的
`self.runners[bucket]` 查找，GPU kernel 内不判断 M，也不在线选择算法。

以后增加 Opus 或 ASM pipeline 时，只增加新的 KID 和 builder，例如：

```text
opus_comm_tp8_m32768_v1
asm_comm_tp8_m32768_v1
```

通算融合 KID 不复用普通 MoE 的 `kernelName2` 字段，因为它描述的是完整 pipeline，
而不是一个 Stage2 计算 kernel。

kernel 文件按实现方式命名，不按 token bucket 命名：

```text
kernels/comm_fused_moe/full_width.py
kernels/comm_fused_moe/windowed.py
```

KID 仍然包含 shape，因为同一种实现方式在不同 M 上可以使用不同编译期参数。当前
M=2048 使用 full-width KID；历史 production winner 曾使用两个 hidden windows，约
`268.713 us`，当前 full-width direct-LSA 约 `266.319 us`，因此暂不回退到旧 window
配置。`windowed.py` 完成后仍需为 M=2048 离线比较 2-window 和必要时 7-window 候选；
只有完整 Graph 至少稳定快约 1% 才替换现有 full-width KID，不增加运行时算法分支。

### 11.3 固定的大 M winner

第一版只固化清理前已经通过生产 gate 的配置：

```text
M=32768
H=7168
I/TP=384
E=384
topk=6
TP=8

window_dim=1024
hidden_windows=7
tile_m=64
tile_n=256
tile_k=128

local_workers=2048
local_vector_width=32
peer_grid=128
fanout_grid=126
service_stride=3
```

固定流水：

```text
G0
G1 + L0
G2 + L1 + R0
G3 + L2 + R1 + F0
...
drain L / R / F
```

不保留 `service_kind`、dynamic window、persistent、XCD、early/after-compute、动态
worker 数量或其他实验开关。已验证 winner 中的参数直接成为编译期常量。

### 11.4 文件改动边界

| 文件 | 计划改动 |
| --- | --- |
| `aiter/ops/flydsl/comm_fused_moe_host.py` | 增加 shape/KID/builder 静态映射，准备 M=32768 runner |
| `aiter/ops/flydsl/kernels/comm_fused_moe/full_width.py` | 保持当前 full-width production KID，不加入 window 分支 |
| `aiter/ops/flydsl/kernels/comm_fused_moe/windowed.py` | 新增固定 TP8、7-window 的组合 kernel |
| `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py` | 只增加组合 kernel 必需的三个私有编译期 hook |

下面这些文件原则上不再修改：

```text
aiter/fused_moe.py
aiter/ops/comm_fused_moe_runtime.py
ATOM 模型 adapter
```

这样大 M 接入不会改变公共 seam、模型选路和已经验证的 M=2048 热路径。

### 11.5 GEMM2 common 的最小 hook

只增加以下私有能力：

```python
_n_tile_range=None
_compact_route=False
_compose_entry=None
```

三项能力分别用于：

1. 只计算当前 1024 列窗口对应的四个 N tiles。
2. 将 route 输出写成紧凑的 `[M, topk, 1024 + 1024/8]` 布局。
3. 把现有 GEMM2 work-item emitter 嵌入 G/L/R/F 组合 kernel。

内部 emitter 还需要接受外部传入的 `block_x`、`block_y` 和 active mask，但不增加
普通 launch 的运行时参数。

普通调用保持默认值时必须满足：

- 原 kernel 名不变；
- 原 launch ABI 不变；
- 原 cache key 不变；
- 原 grid 计算不变；
- 生成的普通 MoE IR/HSACO 不变。

不恢复旧实现中的 `Gemm2OutputPolicy` Enum、`kernel_namespace`、transport 参数、
service 配置和通用校验。drain 阶段使用新文件中的固定 L/R/F kernel，不要求 common
为 service-only 路径预留额外抽象。

### 11.6 Workspace 布局

M=32768 runner 固定持有：

- 两个 compact route buffer，供相邻 G/L 窗口交替使用；
- 七个 partial window workspace 和 readiness epoch；
- 七个 owner payload workspace、owner scale 和 readiness epoch；
- 一个完整 BF16 输出；
- 一个完整 shared partial buffer。

partial、owner payload 和 owner scale 使用 cached `torch.symm_mem`，通过已有 MORI CCO
external window 映射到 flat VA，kernel 直接计算 peer 地址。无需新增 IPC manager、MORI
接口、custom AR 接口或另一套通信 runtime。

不能用两个 full-H route buffer 替代 compact route：M=32768 时两个 full-H route
约占 3.16 GB，而两个 1024-window compact route 约占 452 MB；历史 winner 也使用
compact window 布局。

### 11.7 实施顺序与 review 点

1. 在当前分支基础上创建短期大 M 开发分支，保留现有备份分支不动。
2. 增加 GEMM2 common 的最小 hook，暂不接入通算融合调用。
3. 对普通 Opus/FlyDSL Stage2 比较修改前后的 IR、HSACO 和单算子性能；不一致则先停。
4. 新增固定 `kernels/comm_fused_moe/windowed.py`，只实现已验证的 G/L/R/F worker 和静态调度。
5. 新增 M=32768 workspace 与 host 侧七窗口 launch/drain 顺序。
6. 将新 builder 注册为 `flydsl_comm_tp8_m32768_win7_v1`。
7. 先跑单算子 uniform/skew correctness，再跑 CUDA Graph 性能。
8. 回归 M=2048、普通 Opus、普通 FlyDSL A8W4 Stage2 和 MegaMoE。
9. 单算子全部通过后再进行 ATOM 整网验证。

每个阶段保留独立 diff 供 review，不 amend 到已有提交；临时测试和候选脚本继续放在
仓外。

### 11.8 准入门槛

M=32768 使用与历史 gate 相同的 TP8 节点、输入和 rank-max 统计方式：

- uniform routing 与 skew routing 都通过；
- eager 与 CUDA Graph 输出一致；
- `max_abs=0.8125`、`rel_l2` 约 `0.030132`；
- 七轮 CUDA Graph rank-max median 目标不高于 `2272.528 us`；
- 考虑机器噪声，promotion 判断最多允许约 1% 波动，不接受稳定回退；
- M=2048 的现有约 `266.319 us` Graph 性能不能回退；
- 普通 backend 的生成代码、精度和性能不能回退。

只有满足这些门槛，M=32768 KID 才进入正式 shape 映射。

### 11.9 MORI SDMA 后续实验

SDMA 不是 M=32768 第一版恢复项。先完成并验证固定七窗口 direct-LSA KID，再把 SDMA
作为独立候选做 A/B。原因是 MORI host 注册只发生在初始化阶段，替换 host 接口不会减少
每轮 Stage2 时间；只有 MORI device-side SDMA 搬运可能改变热路径。

第一优先级实验只替换 owner fanout：

```text
R 生成一个压缩 owner MXFP8 chunk
  → MORI SDMA put 到各 peer 的本地接收槽
  → 与下一个 R chunk 重叠
  → peer 从本地槽解码并写 BF16 output
```

暂不优先用 SDMA 搬运 partial reduce 输入。R 阶段仍要执行 MXFP8 解码、跨 rank 求和和
再量化；先把 partial 搬到本地会增加一次 HBM 写入和读取，未必优于当前直接 remote load。

实现约束：

- 只使用 MORI 已有的 `create_dev_comm` 和 FlyDSL `Sdma.put/commit/quiet`，不先修改
  MORI 仓或增加新接口；
- SDMA 候选使用独立 KID，例如 `flydsl_comm_tp8_m32768_win7_sdma_v1`，不在 direct-LSA
  kernel 中增加运行时分支；
- M=2048 不创建 DevComm、SDMA queue、signal 或接收 workspace，继续使用当前 direct
  flat-VA 路径；
- 所有 queue、signal 和接收槽在 runner 初始化时一次性准备，每轮不得出现 MORI host
  调用或动态分配；
- 不用 `Window.lsa_ptr()` 替换当前固定 flat-VA 地址计算。该接口更通用，但不会减少
  数据搬运，并可能增加 device-side window 字段读取；
- 不增加在线 tuner、fallback 或 transport 开关，仓外 benchmark 只输出 direct 或 SDMA
  两个完整 pipeline KID 中的 winner。

SDMA 会增加本地接收 workspace、queue/signal 同步以及一次额外 HBM 落地，因此只在大块
数据和 copy/compute 能稳定重叠时可能获益。验收时必须在同一次 TP8 运行中比较
七窗口 direct-LSA 与 SDMA KID，并同时记录 R、F 和完整 Graph：

- uniform、skew、eager 和 CUDA Graph 精度与 direct-LSA 一致；
- 完整 Graph rank-max median 至少稳定快约 1%，否则删除 SDMA 候选；
- 最终性能仍以不高于历史 `2272.528 us` 为目标，不能用 SDMA 的局部 F 提升掩盖完整
  pipeline 回退；
- M=2048 保持当前约 `266.319 us` Graph 性能，普通 backend 和其他 shape 不受影响。

### 11.10 第一版不增加 production tuner

第一版使用仓外 benchmark 脚本复现历史 winner并验证固定实现，不恢复
`moe_tp_stage2_tuner.py`。后续接入新模型或新 shape 时，离线枚举候选并输出一个 KID；
生产代码只保存通过精度与性能 gate 的 winner。

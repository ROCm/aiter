# 通算融合 MoE：FlyDSL Host 重构计划

## 1. 一句话目标

FlyDSL host 只做：

```text
持有通信资源 → 准备固定 ABI → 按固定顺序 launch kernel → 返回输出
```

它不是第二套 MoE runtime。routing、sorting、Stage1、后端选择和 tuner 都不进入 host。

第一版继续保留两个明确实现：

- M=2048：full-width runner；
- M=32768：windowed runner。

不把两者合并成万能 runner，不增加 Protocol、基类、在线 tuner、fallback、动态 planner
或插件 registry。当前 host 为 386 行，重构目标约 250～300 行；不靠压行或把 host 逻辑
藏进 kernel 文件达标。

## 2. 仓内 FlyDSL 实现给出的启发

| 实现 | 应该借鉴 | 不应照搬 |
| --- | --- | --- |
| 普通 MoE | compile 参数、launch ABI、`_run_compiled` 三层分开 | 3000 行公共兼容逻辑 |
| MXFP4 GEMM1/2 | cached compile → 固定参数 → 直接 launch | 通用 API 的大量输入修复 |
| FHMoE | 复用普通 MoE 主流程 | `_compile_kernel` 等注入机制 |
| MegaMoE | production winner 编译期固定 | 实验 knob、手工 cache 和调参参数 |
| Dispatch/Combine | 长生命周期对象持有通信 buffer 和 peer pointer | 大 Config、tuning table、多模式分支 |

最符合 FlyDSL 习惯的边界是：

```text
kernel 文件：GPU 算法，compile_* 返回 cached @flyc.jit launcher
host 文件：tensor/MORI 生命周期、ABI 和固定 launch 顺序
runtime：只按 bucket 取得 runner，不理解 FlyDSL 实现
```

普通 `moe_kernels.py` 是通用入口，不是专用融合 host 的体量模板。

## 3. Host 的职责边界

Host 只负责五件事：

1. 初始化时分配长期复用的 tensor 和 symmetric workspace。
2. 注册 MORI external window，保存 flat-VA base 和 window 生命周期。
3. 把普通 Stage2 参数转成固定 FlyDSL ABI。
4. 按 production winner 的顺序 launch kernel 和 epoch barrier。
5. 返回完整 `[M, H]`，并提供 padding 所需的复用 buffer。

Host 不负责：

- routing、sorting、Stage1、权重预处理；
- 在线选择 tile/window/worker；
- 读取 tuner CSV/JSON；
- 普通 MoE fallback；
- Opus/ASM 选择；
- 每次 forward 重复 shape、dtype、TP 校验；
- 为尚未接入的模型预留参数。

模型布局和量化格式由 ATOM adapter 初始化时确认。host factory 只保留一次 exact shape
检查，避免固定 kernel 被错误模型调用。

## 4. 目标文件结构

```text
aiter/ops/flydsl/comm_fused_moe_host.py
  ├─ 共享资源 helper
  ├─ _FullWidthRunner
  ├─ _WindowedRunner
  └─ create_flydsl_comm_fused_runners

aiter/ops/flydsl/kernels/comm_fused_moe/sync.py
  ├─ FLAT_VA_RANK_STRIDE
  └─ compile_epoch_barrier

aiter/ops/flydsl/kernels/comm_fused_moe/full_width.py
  └─ full-width G/L/RS/AG kernel

aiter/ops/flydsl/kernels/comm_fused_moe/windowed.py
  └─ windowed compute/cycle/drain kernel
```

`sync.py` 只是移动 full-width/windowed 真正共用的通信原语，不新增框架。barrier kernel
名称和实现保持不变。

以下文件不应因本次 host 重构继续变化：

```text
aiter/fused_moe.py
aiter/ops/comm_fused_moe_runtime.py
aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py
ATOM 模型 adapter
```

## 5. 目标 Host 设计

### 5.1 最多五个共享 helper

```text
_align         对齐 workspace 字节数
_workspace     分配 symmetric payload + epoch，只初始化 epoch
_register      注册 MORI window，返回 handle 和 flat base
_barrier       launch 固定 epoch barrier
_stage2_args   生成两个 runner 共用的 Stage2 GEMM ABI
```

只有两个 runner 都使用的逻辑才进入 helper。不增加 base class、mixin、Protocol 或
`RunnerConfig`。

### 5.2 两个具体 runner

```python
class _FullWidthRunner:
    output: torch.Tensor
    def __call__(self, *, stage2_args, stage2_kwargs, shared_partial): ...


class _WindowedRunner:
    output: torch.Tensor
    def __call__(self, *, stage2_args, stage2_kwargs, shared_partial): ...
```

Python 实际调用契约已经足够，不再定义正式接口类。Padding 时 runtime
直接复用 `runner.output` 存放 shared partial，不为返回同一字段定义空转发方法。

`__call__` 热路径只允许：

1. 取得当前 CUDA stream；
2. 解包必要 tensor；
3. 构造 pointer/scalar ABI；
4. 执行固定 launch；
5. 返回 output。

热路径中禁止分配、清零、注册通信资源、读取环境变量、查询 tuner、选择 kernel、logging
或 fallback。首次 JIT 仍由统一 `_run_compiled` 完成，host 不维护第二套 compiled cache。

### 5.3 Factory

当前只有一个 production shape，继续使用最直接的写法：

```python
if shape != SUPPORTED_SHAPE:
    raise KeyError(...)

if runners is None:
    runners = {
        2048: _FullWidthRunner(tp_group),
        32768: _WindowedRunner(tp_group),
    }
return runners
```

真正接入第二个 shape 时，再替换成局部 exact 映射：

```text
(H, I_per_tp, E, topk, TP) -> {bucket: runner_builder}
```

它只是 production winner 表，不做模糊匹配或 fallback。

## 6. Workspace 收缩

| 当前冗余 | 重构方式 | 预期收益 |
| --- | --- | --- |
| 独立 `shared` tensor | padding 时直接复用 output | 节省 28 MiB + 448 MiB/GPU |
| `reduced_payload(s)` view | 直接传 reduced workspace 首地址 | 删除无意义对象和代码 |
| 专用 empty-bias tensor | disabled bias ABI 使用已有合法 pointer | 删除零长度占位 tensor |
| 整块 symmetric buffer 清零 | 只清零 8-byte epoch；payload 由 kernel 覆盖 | 减少初始化写流量 |
| 七份 window scratch | partial/reduced-payload/reduced-scale 改为 2-slot ring | 约节省 186 MiB/GPU |

output 复用 shared 的时序依据：

- full-width：L 完整读取 shared 后，RS/AG 才写 output；
- windowed：L 读取 N 时，只写更早的 N-1/N-2 输出窗口。

window scratch 使用：

```python
slot = window & 1
```

复用 slot 前，必须由 stream 顺序和 epoch barrier 保证所有 rank 已完成旧数据的 RS/AG 读取。
该结论需要通过固定流水时序检查和 GPU A/B，不只做代码推断。

收缩完成后，两个 bucket 的 tensor payload 预计从约 1.714 GiB/GPU 降到约
1.07 GiB/GPU。

## 7. Window 流水如何表达

保留短循环表达 steady state：

```text
G0
for local in 0..5:
    G(local+1) + L(local) + optional R(local-1) + optional F(local-2)
    required barriers
drain: L6/R5/F4 -> R6/F5 -> F6
```

不手工展开七轮。`reduce_scatter/all_gather` 是否存在只选择三种已编译
launcher，不是在线算法选择。

drain 的固定 ABI 对关闭阶段仍需要合法 dummy pointer。统一使用 slot 0 作为 dummy，删除
多层 `None/anchor` fallback；三次 drain 的真实逻辑窗口在调用点明确写出。

## 8. 行数预算

| 部分 | 目标行数 |
| --- | ---: |
| imports、常量、缓存 | 25～35 |
| 共享 helper | 35～50 |
| full-width runner | 65～80 |
| windowed runner | 105～125 |
| factory | 10～20 |
| 合计 | 250～300 |

如果一个抽象只节省十几行，却引入配置对象、回调或分支，不采用。少量清晰重复优于通用
框架。

## 9. 后续扩展方式

### 新 token bucket

1. 离线确认 full-width/windowed 是否适用。
2. 固化 kernel 常量和 workspace 尺寸。
3. 新增 runner builder。
4. bucket 映射增加一行。
5. 精度、性能、整网通过后启用。

### 新模型 shape

调度不同就新建按实现方式命名的 kernel 文件，不在旧 kernel 内加入 shape 分支；host
新增一个具体 runner/builder 和一条 exact shape 映射，公共 runtime 不变。

### Opus / ASM

各自实现自己的 host/runner，遵守同一调用契约。FlyDSL host 不导入 Opus/ASM，也不负责
跨后端选 winner；选择发生在模型初始化或离线配置层。

## 10. 实施步骤

### Step 0：冻结基线

- 保存当前 commit、diff、M=2048/M=32768 uniform/skew eager/Graph 数据。
- 记录每卡 workspace 显存。
- 不修改 ATOM 和普通 MoE。

### Step 1：纯结构收缩

- 移动公共 barrier/stride 到 `sync.py`。
- 合并五个共享 helper。
- 类改名为 `_FullWidthRunner`、`_WindowedRunner`。
- 删除重复 communicator、flat-base、Stage2 ABI 构造。

本步不改变 buffer 数量、地址布局和 launch 顺序。验证无回退后再继续。

### Step 2：删除冗余资源

- output 复用 padding shared；
- reduced workspace 直接作为 payload pointer；
- 删除 empty-bias tensor；
- symmetric workspace 只初始化 epoch。

独立保留 diff，完成单算子 A/B 后 review，不直接 amend。

### Step 3：window 改为 2-slot ring

- partial/reduced-payload/reduced-scale 从七份改为两份；
- scratch 索引统一使用 `window & 1`；
- 七个 logical output view 保持不变；
- 三次 tail 显式写出，删除 `None/anchor` 泛化。

重点验证 slot 复用前所有 rank 的 RS/AG 已完成。

### Step 4：factory 收口并整网验证

- 只保留当前 exact shape 和已通过准入的 bucket。
- 不增加空配置、占位 builder 或 fallback。
- 单算子全部通过后再跑 ATOM 整网。

## 11. 准入门槛

- M=2048 uniform/skew 精度不变，Graph 性能无稳定回退；
- M=32768 uniform/skew 保持约 2200 us，最多接受约 1% 噪声；
- eager 和 CUDA Graph 均通过；
- 普通 FlyDSL A8W4 Stage2、Opus、MegaMoE 回归通过；
- GPU 算法未改时，kernel 名和 launch ABI 不变；
- workspace 显存达到预期下降；
- ATOM 整网精度和性能不回退；
- 仓内不留下临时测试、core dump、tuner 结果或实验参数。

## 12. 逐行判断标准

最终 host 的每一行只能属于：

1. 当前 production shape 的资源布局；
2. 当前 kernel ABI；
3. 当前 G/L/RS/AG 顺序和同步；
4. 下一个已验证 shape 可直接复用的极小 helper。

不能归入这四类的代码不进入第一版。

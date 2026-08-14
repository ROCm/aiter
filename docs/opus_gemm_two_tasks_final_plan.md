# OPUS GEMM 两项任务最终计划（任务一冻结版 / 任务二执行版）

> 更新时间：2026-08-11。
>
> 本文已经按当前工作树整体重写，不再保留早期方案正文。当前唯一施工入口是第 6 节
> Step B0；任务一不从 A0 重做。
>
> 本轮只更新计划文档，不修改 OPUS 源码、不运行 GPU 测试、不提交 commit。

## 0. 文档定位、结论和范围

### 0.1 当前结论

两项任务的状态已经分开：

1. 任务一按照当前 dirty 工作树中的确定实现冻结。它已经完成 workspace 所有权迁移和
   gfx1250 fused family 迁入，但仍有明确列出的硬件、性能和 mono-tile FP32 归因开放项。
2. 任务二从这个冻结端点继续，只重构 OPUS GEMM 接口、运行时职责和生成表命名，不回退
   任务一，也不借接口重构修改 kernel 算法。

任务二最终要得到下面的边界：

~~~text
Python
  a16w16: explicit -> tuned CSV -> Python heuristic -> framework fallback
  family adapter: shape/dtype/layout检查，输出和workspace准备
      |
      v
C++ family-specific raw launch
  arch/family/kid严格复核
  generated kid dispatch
  kargs构造和kernel launch
~~~

C++ 最终不再包含：

- generic optional-parameter mega entry；
- 运行时 <code>(M,N,K) -&gt; kid</code> shape lookup；
- C++ a16w16 heuristic；
- framework fallback policy；
- workspace allocator、registry、handle、mirror或prewarm。

### 0.2 权威顺序

发生冲突时按下面的顺序判断：

1. 当前源码和当前 canonical registry；
2. <code>docs/task1_checkpoint.md</code> 的“任务一冻结检查点”；
3. <code>docs/task1_detail.md</code> 的当前主线章节；
4. <code>docs/opus_gemm_splitk_workspace_torch_current_flow_changes.md</code>；
5. 本文中的任务二实施顺序。

早期文档里“保留 WorkspacePlan”“gfx1250 fused 不在范围”“gfx1250 workspace 全为 FP32”
等描述都不是当前设计。当前源码中不存在 WorkspacePlan，两个独立 workspace Python 文件
也已经删除。

### 0.3 本次范围

任务二包含：

- 保留 a16w16 高层 shape-driven API；
- 把覆盖 gfx942、gfx950、gfx1250 的 a16w16 显式 kid raw entry 从 tune 命名改成
  launch 命名；
- 从 generic C++ entry 提取 gfx950 a8w8 no-scale 和 blockscale（plain WQ）两条能力；
- 把 gfx942 a8w8 blockscale bpreshuffle 显式入口改成 launch 命名；
- 现在就保留一个覆盖 gfx942、gfx950、gfx1250 的架构中立
  <code>opus_gemm_a8w8_blockscale_bpreshuffle_launch</code> ABI：当前只有gfx942表非空，
  gfx950/gfx1250生成合法空表并返回明确的no-registered-kernel错误；
- 为四种物理合同建立独立 C++、pybind 和 Python family entry；
- 对 gfx942、gfx950、gfx1250 建立明确的 family capability matrix、跨架构拒绝和逐架构验收；
- 在 <code>gemm_a8w8_blockscale_bpreshuffle</code> 高层路由中只迁移
  <code>libtype == "opus"</code> 分支，保持 gfx950 的 CK/CKTile/ASM/Triton 和 gfx1250 的
  FlyDSL/Triton/Gluon 路径不变；
- 删除 C++ runtime shape lookup 和 C++ heuristic；
- 保留 build-time subset compile 的 CSV、sidecar、arch filter 和 heuristic-default kid 集合；
- 更新所有仓内调用方、测试和 README。

任务二不包含：

- 修改任何 GEMM kernel 数学算法、tile、prefetch、workspace layout或reduce顺序；
- 修复任务一遗留的 gfx950 mono-tile FP32 数值问题；
- 完成任务一尚未做的 gfx942/gfx1250 实机和性能验收；
- 修改 OPUS MoE a8w4 接口；
- 修改 CK、CKTile、FlyDSL 或其他 GEMM 的公共接口；
- 为当前 OPUS registry 中没有实现的 gfx942/gfx1250 A8W8 family 新写 kernel；
- 新增 a4w4 family；
- 恢复任何 C++ raw HIP workspace 所有权路径。

## 1. 当前仓库和任务切换基线

| 项目 | 当前值 |
|---|---|
| 仓库 | <code>/root/workspace/0810/aiter</code> |
| 分支 | <code>splitk_to_torch_2</code> |
| 当前 HEAD | <code>2352c46c784d6ba3a0c71ff89b4bdb4c2fefa59f</code> |
| HEAD 标题 | <code>[OPUS] Finalize workspace migration audit</code> |
| 任务一原始比较基线 | <code>ca68b4f3501762c15c550cb920a5516e9710cf89</code> |
| 工作树 | 有任务一未提交修改和未跟踪文件，必须原地保留 |
| 当前动作 | 保存任务一，开始任务二文档和后续接口重构 |

禁止用 reset、checkout 或整树覆盖清理任务一工作。任务二每一步只修改列出的接口相关文件，
并在修改前后保存 <code>git status --short</code> 和目标文件 diff。

## 2. 任务一冻结版本：当前确定实现

本节只描述当前确定版本，不再描述任务一的历史中间方案。

### 2.1 唯一生产调用流程

~~~text
gemm_a16w16_opus(A, B, ...)
  -> validate/reshape XQ, WQ, Y
  -> select_launch_config(...)
       explicit kid
       -> tuned CSV
       -> per-arch Python heuristic
       -> framework fallback
  -> resolve requested_kid / actual_kid / split-K
  -> framework fallback ? Torch path : OPUS path
  -> _launch_a16w16_with_torch_workspace(...)
  -> _init_a16w16_workspace(config, XQ, Y, optional workspace)
       exact actual_kid -> canonical instance
       exact actual_kid -> workspace capability/dtype/tile/layout
       caller Tensor or per-call torch.empty
  -> _opus_gemm_a16w16_tune_raw(...)
  -> generated 5参数 non-workspace 或 6参数 workspace launcher
  -> C++物理合同复核
  -> launch
~~~

固定选择顺序是：

~~~text
explicit -> tuned CSV -> Python heuristic -> framework fallback
~~~

关键语义：

- <code>requested_kid</code> 是 explicit、CSV 或 heuristic 最初提出的 kid；
- <code>actual_kid</code> 是 redirect、shape、dtype、bias 和 split-K 合法性解析后真正 launch
  的 kid；
- workspace capability、dtype、tile、shape和launcher ABI只读取
  <code>actual_kid</code>；
- tuned row 无效时，kid 和 split-K 必须作为一个整体丢弃，不能把旧 split-K 带到
  heuristic；
- framework fallback 是 selector 的终点，不进入 OPUS raw launch。

### 2.2 唯一 workspace 初始化点

当前唯一 Python 分配入口是：

~~~text
aiter/ops/opus/gemm_op_a16w16.py::_init_a16w16_workspace()
~~~

不存在独立 planner 文件、通用 plan 对象或全局 Tensor cache。调用方未提供 workspace 时，
该函数只在 actual kid 确实需要 external workspace 时执行一次：

~~~python
torch.empty(shape, dtype=exact_kid_dtype, device=XQ.device)
~~~

调用方提供 workspace 时，Python 将同一个 Tensor 交给 raw binding；generated C++ launcher
负责最终 device、dtype、contiguous、alignment、overflow 和 capacity 复核。non-workspace
kid 必须传 <code>workspace=None</code>。

### 2.3 当前 workspace 物理合同

令：

~~~text
padded_M = ceil_div(M, B_M) * B_M
padded_N = ceil_div(N, B_N) * B_N
~~~

其中 <code>B_M/B_N/B_K</code> 都来自 exact actual kid。

| 架构 / family | workspace shape | split-K来源 | batch |
|---|---|---|---|
| gfx950 two-stage | <code>[allocation_split_k,batch,padded_M,padded_N]</code> | resolved runtime split-K | 多 batch |
| gfx942 two-stage | <code>[allocation_split_k,batch,padded_M,padded_N]</code> | resolved runtime split-K | 多 batch |
| gfx1250 two-stage | <code>[allocation_split_k,padded_M,padded_N]</code> | resolved runtime split-K | 必须为1 |
| gfx1250 fused | <code>[tiles_m,tiles_n,fuse_split_k-1,B_M,B_N]</code> | exact kid compile-time值 | 必须为1 |

gfx1250 fused 的 tile-major workspace 不能套用 two-stage 的 split-major 公式；runtime/CSV
里的 splitK 也不能改变 fused kid 的容量。

当前 canonical registry 快照：

| 架构 / family | workspace kids | BF16 workspace | FP32 workspace | non-workspace a16w16 |
|---|---:|---:|---:|---:|
| gfx950 FlatMM two-stage | 48 | 0 | 48 | 92 |
| gfx942 two-stage | 8 | 3 | 5 | 14 |
| gfx1250 two-stage | 496 | 496 | 0 | 0 |
| gfx1250 fused | 1378 | 780 | 598 | 0 |
| gfx1250 合计 | 1874 | 1276 | 598 | 0 |

<code>OpusGemmInstance.splitk_workspace_dtype</code> 是 exact-kid 单一事实源。所有 external
workspace kid 必须显式声明 BF16 或 FP32；non-workspace kid 保持未设置。

### 2.4 gfx942 requested/actual kid 规则

gfx942 BF16 workspace exact-N 集合是：

~~~python
{64, 128, 256, 384, 512, 1024, 2048}
~~~

N 不在集合时：

| requested kid | actual kid |
|---:|---:|
| 10210 | 10200 |
| 10213 | 10203 |
| 10216 | 直接拒绝 |

redirect 必须在 workspace 分配前完成。generated launcher不再先接收错误 dtype 的 buffer
再静默跳转；raw C++仍保留 exact-N 物理防线。

### 2.5 当前 C++ launcher ABI

任务一已经把生成表按函数指针类型拆开：

~~~cpp
using OpusA16W16Kernel = void (*)(
    XQ, WQ, Y, optional_bias, split_k);

using OpusA16W16WorkspaceKernel = void (*)(
    XQ, WQ, Y, workspace, optional_bias, split_k);
~~~

workspace 和 non-workspace 指针不能 type-pun、强转或混装。arch router先查 workspace
membership，再进入对应的 strict kid table。C++只验证并launch，不分配或保留 workspace。

### 2.6 已删除或停用的旧路径

当前确定版本已经满足：

- 旧 C++ allocator、per-stream registry、owner、handle、host/device mirror全部删除；
- 旧 prewarm、capture stream猜测和 warmed-set全部删除；
- generic C++ <code>opus_gemm()</code> 的 BF16 a16w16分支硬拒绝；
- Python <code>opus_gemm_workspace_init()</code> 仅为 deprecated no-op；
- <code>aiter/ops/opus/_workspace.py</code> 已删除；
- <code>aiter/ops/opus/_workspace_a16w16.py</code> 已删除；
- gfx1250 #4246 fused family已经进入 registry、codegen、workspace和dispatch路径。

这里停用的是“generic C++ BF16入口自行选择kernel”的旧路径，不是停用 BF16 OPUS GEMM
能力。任务一已经用下面的生产链替代它：

~~~text
gemm_a16w16_opus(...)
  -> Python select_launch_config(...)
       explicit -> tuned CSV -> per-arch Python heuristic -> framework fallback
  -> resolved requested_kid / actual_kid / split-K
  -> exact actual_kid Torch workspace（需要时）
  -> _opus_gemm_a16w16_tune_raw
  -> C++ opus_gemm_a16w16_tune
  -> per-arch strict kid dispatch
  -> generated launcher
~~~

因此 generic BF16 C++分支硬拒绝后有完整替代，而且替代链覆盖 gfx942、gfx950、gfx1250。
任务二只把最后两层的 raw/C++生产名字改成
<code>_opus_gemm_a16w16_launch_raw</code> / <code>opus_gemm_a16w16_launch</code>；
Python selector、actual-kid解析和Torch workspace所有权保持不变。framework fallback在进入raw
launch之前完成，不由C++接管。

任务二不得恢复上述任何路径。

### 2.7 任务一验证事实和开放项

已确认：

- CPU过滤回归：<code>149 passed, 18 deselected, 0 failed</code>；
- gfx950 focused suite：<code>162 passed, 14 skipped, 0 failed</code>；
- gfx950 140-kid sweep 中 48/48 workspace kid全部通过；
- 140-kid整体结果：<code>130 passed, 10 failed</code>；
- 10项失败全部是 non-workspace mono-tile FP32：
  <code>1400--1404</code>、<code>6400--6404</code>；
- Python compile、diff检查、full fused codegen和代表性 gfx1250 syntax检查已完成。

仍开放：

- 10个 mono-tile FP32失败在原始基线上的归因；
- 原始/当前性能 A/B；
- gfx942 实机数值、graph、并发和性能；
- gfx1250 two-stage/fused 实机数值、graph、并发和性能。

这些开放项不是任务二可以改写成“已通过”的项目。任务二验收采用“相对任务一零新增回归”：
已知10项可以继续列为已知问题，但不能增加新的失败。

## 3. 任务二开始时的真实接口和问题

### 3.1 当前 C++ / pybind 顶层符号

| 当前符号 | 当前职责 | 问题 | 任务二处置 |
|---|---|---|---|
| <code>opus_gemm</code> | gfx950 FP8 no-scale或blockscale（源码称scale）；BF16分支只报错 | optional mega entry、group_layout无效、两种scale合同混合 | 提取两条family entry后删除 |
| <code>opus_gemm_a16w16_tune</code> | 三架构显式kid，接收optional workspace | tune命名与生产launch职责不符 | 改为 <code>opus_gemm_a16w16_launch</code> |
| <code>opus_gemm_a8w8_blockscale_bpreshuffle_tune</code> | gfx942 kid 11000 strict dispatch | tune命名、scale仍用optional C++类型 | 改为family launch，scale改必选引用 |

当前 pybind 宏和注册点位于：

~~~text
csrc/include/rocm_ops.hpp
csrc/pybind/opus_gemm_pybind.cu
~~~

当前 generic <code>opus_gemm</code> 的具体问题：

- <code>group_layout</code> 参数未被任何 OPUS GEMM 路径消费；
- 只有 x_scale 和 w_scale 同时存在才进scale分支；
- 只传一个scale时会落到no-scale分支，属于危险的静默合同变化；
- gfx950 kid 1/2是硬编码launcher，不是strict generated kid table；
- Python没有对应的独立公开 family wrapper；
- BF16 binding仍作为私有 ABI 残留存在，虽然生产路径不会调用。

### 3.2 当前 Python API

| Python名字 | 当前角色 | 任务二目标 |
|---|---|---|
| <code>gemm_a16w16_opus</code> | a16w16高层shape-driven API | 原名和行为保留 |
| <code>opus_gemm_a16w16_tune</code> | 显式kid兼容wrapper | 保留一个发布周期的deprecated wrapper |
| <code>_opus_gemm_a16w16_tune_raw</code> | 当前pybind raw binding | 改为私有 <code>_launch_raw</code> |
| <code>_opus_gemm_bf16_dispatch</code> | generic符号私有ABI残留 | 删除 |
| <code>opus_gemm_a8w8_blockscale_bpreshuffle_tune</code> | gfx942显式kid wrapper | 旧名deprecated，新名launch |
| gfx950 a8w8 Python wrapper | 当前不存在 | 新增no-scale和blockscale两个入口 |

### 3.3 当前仓内调用方

a16w16旧Python名字的生产/调优调用方：

~~~text
aiter/tuned_gemm.py
csrc/opus_gemm/opus_gemm_tune.py
csrc/gemm_a16w16/gemm_a16w16_tune.py
aiter/ops/deepgemm.py
~~~

gfx942 a8w8旧名字的调用方：

~~~text
csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py
aiter/ops/gemm_op_a8w8.py
~~~

直接依赖当前私有raw名字的测试：

~~~text
op_tests/test_opus_dispatch.py
op_tests/test_opus_workspace.py
op_tests/test_opus_graph.py
op_tests/test_opus_gfx950_exhaustive.py
~~~

任务二必须逐一修改，不能只改 pybind 后等待运行时才发现旧名字。

### 3.4 当前 codegen 的两类 lookup

当前生成物含义不同：

| 生成物 | key | 是否属于runtime policy | 最终处置 |
|---|---|---:|---|
| <code>opus_gemm_lookup.h</code> | <code>(M,N,K) -&gt; kid</code> | 是 | 停止生成并删除依赖 |
| <code>opus_gemm_a16w16_tune_lookup.h</code> | <code>kid -&gt; typed launcher</code> | 否 | 保留能力并改名为kid dispatch |
| <code>opus_gemm_a8w8_tune_lookup.h</code> | gfx942 <code>kid -&gt; launcher</code> | 否 | 扩展为三种a8 family的独立typed表并改名 |

三个 arch header仍包含旧 C++ shape结构、二分查找、heuristic函数或heuristic include：

~~~text
csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh
csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh
~~~

对应旧heuristic header：

~~~text
csrc/opus_gemm/include/gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh
csrc/opus_gemm/include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_heuristic_dispatch_gfx1250.cuh
~~~

Python生产路径已经不使用它们，任务二在 parity golden 固化后删除。

### 3.5 两层三架构 capability matrix

必须分开看“用户可调用的AITER高层功能”和“当前OPUS backend registry”。前者会根据CSV的
<code>libtype</code> 路由到 CK、CKTile、ASM、Triton、Gluon、FlyDSL 或 OPUS；不能因为某个
架构没有OPUS kid，就写成该架构没有这项功能。

高层功能的实际覆盖是：

| 高层功能 | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| <code>gemm_a16w16_opus</code> | OPUS | OPUS | OPUS |
| <code>gemm_a8w8_blockscale_bpreshuffle</code> | 有：CK/ASM/Triton；部分tuned row为OPUS kid 11000 | 有：CK/CKTile/ASM/Triton | 有：FP8-E8M0 128-block走FlyDSL；FP32-scale路径走Triton/Gluon |

所以 gfx950 确实有 blockscale bpreshuffle，当前基础tuned CSV就含gfx950的CK/ASM行；gfx1250
也确实有同名高层能力，并且model配置中已经存在FlyDSL tuned row。它们不是“将来才有”。

再看当前 canonical OPUS kernel和任务二完成后的family槽位：这里的“无”或“0 kernel”只表示
<code>csrc/opus_gemm</code> 当前没有对应的OPUS kernel/kid，不表示AITER高层功能不存在。

| logical family | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| a16w16 | 有：non-workspace + two-stage workspace | 有：non-workspace + FlatMM two-stage workspace | 有：two-stage + fused workspace |
| a8w8 no-scale | 无 | 有：kid 2 | 无 |
| a8w8 blockscale，plain WQ | 无 | 有：kid 1；当前内部kernel tag为 <code>a8w8_scale</code> | 无 |
| a8w8 blockscale，bpreshuffle WQ | 接口槽位非空：kid 11000 | 接口/dispatch槽位预留：当前0 kernel | 接口/dispatch槽位预留：当前0 kernel |

所以任务二不是“只做gfx950”：a16w16的接口迁移、strict dispatch、codegen集合等价和GPU验收
必须完整覆盖 gfx942、gfx950、gfx1250。a8w8 raw接口迁移当前OPUS registry里真实存在的
三个物理family；同时必须做三架构高层路由回归，确保 gfx950/gfx1250 的非OPUS
blockscale-bpreshuffle实现不被改坏或错误导向OPUS raw entry。

任务二不把现有 CK/FlyDSL/Triton kernel 改名为OPUS kernel，但现在就把稳定的OPUS
blockscale-bpreshuffle family接口和三架构dispatch槽位开出来。当前状态是：gfx942槽位含
kid 11000，gfx950/gfx1250槽位为空；在后两者调用同一接口时，符号必须存在并明确报告该
family在当前架构没有已注册kernel，而不是AttributeError或误入gfx942表。

以后把gfx950或gfx1250 blockscale bpreshuffle kernel接入OPUS时，不再改C++声明、pybind、
Python公共签名或高层 <code>libtype == "opus"</code> 分支，只需增加：

1. canonical registry实例和该arch的logical-family到kernel-tag映射；
2. 对应arch codegen emitter和物理合同检查；
3. generated per-arch typed dispatch表内容；
4. mandatory kid、sidecar或 <code>libtype == "opus"</code> tuned CSV记录；
5. 该架构的数值、graph、并发和性能验收。

kernel本身及上述接入/验收仍是后续工作；本任务先保证接口ABI不需要再改。

### 3.6 当前 OPUS 三种 a8w8 物理合同必须分开

| family | arch / kid | 输入输出 | scale合同 | 当前限制 |
|---|---|---|---|---|
| a8w8 no-scale | gfx950 / 2 | FP8 XQ/WQ，FP32 Y | 无scale | 3D batch；generated prefetch/K约束 |
| a8w8 blockscale，plain WQ | gfx950 / 1 | FP8 XQ/WQ，FP32 Y | group为1x128x128，x/w scale都必需 | 3D batch；scale shape/dtype/contiguous需显式化 |
| a8w8 blockscale bpreshuffle | gfx942 / 11000 | FP8 XQ/WQ，BF16 Y | FP32 block scale，128x128 | batch=1，N/K exact tile，B为bpreshuffle布局 |

gfx950 blockscale的逻辑scale布局固定为：

~~~text
x_scale: [batch, M, K/128]
w_scale: [batch, N/128, K/128]
~~~

batch=1时可由Python adapter接受省略leading batch的2D形式，再以同一物理连续布局传给raw
entry。任务二要把 dtype、shape、device和contiguous检查写清楚，不能继续靠
<code>optional.value()</code> 和隐式扁平地址假设。

gfx942 kid 11000 当前合同：

~~~text
x_scale: fp32 [M, K/128]
w_scale: fp32 [N/128, K/128]
Y:       bf16 [M, N] 或 [1, M, N]
batch:   1
N % 128 == 0
K % 128 == 0
~~~

前两条正是“no-scale”与“blockscale”；第三条是在blockscale数学语义上要求bpreshuffle WQ。
但当前OPUS gfx950 plain-WQ blockscale 与 gfx942 OPUS blockscale bpreshuffle 不能做成一个
<code>bpreshuffle=true/false</code> 开关：除了WQ布局，它们的arch、Y dtype、batch能力和scale
物理读取合同也不同。三条路径不能合并成一个带 optional scale、optional layout和dtype路由
的入口。

## 4. 任务二最终接口

### 4.1 命名原则

- <code>gemm_*</code> 表示面向普通调用方的高层shape-driven API；
- <code>opus_gemm_*_launch</code> 的C++/私有raw表示显式family和resolved kid；Python wrapper
  可以接受 <code>kid=None</code>，但必须先在Python解析成exact kid再调用raw；
- <code>*_tune</code> 只作为旧Python兼容名，不能继续作为C++生产ABI；
- C++参数使用 <code>kid</code> 和 <code>split_k</code>，新Python入口也采用相同命名；
- raw <code>compile_ops</code> binding保持私有，普通调用方使用Python wrapper。

### 4.2 最终 C++ raw API

~~~cpp
void opus_gemm_a16w16_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k);

void opus_gemm_a8w8_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    int kid);

void opus_gemm_a8w8_blockscale_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    int kid);

void opus_gemm_a8w8_blockscale_bpreshuffle_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    aiter_tensor_t& Y,
    int kid);
~~~

约束：

- 每个raw entry只接受自己的物理合同；
- 两个a8 blockscale入口的scale参数是必选引用，不是optional；
- unknown kid、错误arch或错误family直接失败；
- raw entry不选择默认kid、不查CSV、不运行heuristic、不分配Tensor；
- a16 workspace参数继续是optional，因为同一a16 family中同时存在5参数和6参数launcher；
- 参数顺序由pybind keyword固定，gfx942 blockscale保留现有Python调用习惯中的
  <code>XQ,WQ,x_scale,w_scale,Y</code> 顺序。

新OPUS raw架构路由固定为：<code>opus_gemm_a16w16_launch</code> 覆盖
gfx942、gfx950、gfx1250；no-scale和plain-WQ blockscale当前只接受gfx950；
blockscale-bpreshuffle的公共符号在三个架构上都存在，并按runtime arch进入各自独立的
generated typed table。当前gfx942表含kid 11000，gfx950/gfx1250表为空；空表必须报告
“family接口已存在，但当前arch没有已注册OPUS kernel”，不能落入其他架构的kid表。
这不影响第3.5节所列的gfx950/gfx1250高层非OPUS后端。

blockscale-bpreshuffle raw入口只做所有实现共有的检查：五个Tensor存在、同device、显式kid
和family membership。Y dtype、scale dtype/shape、batch、tile以及WQ bpreshuffle物理布局属于
<code>(arch,family,kid)</code> 合同，由per-arch generated launcher复核；不能在公共router中
硬编码gfx942的BF16/FP32-scale规则，否则将来增加gfx950/gfx1250 kernel仍要修改公共ABI实现。

### 4.3 最终 Python API

高层API保持：

~~~python
gemm_a16w16_opus(
    A,
    B,
    bias=None,
    dtype=torch.bfloat16,
    *,
    kernelId=None,
    splitK=None,
    out=None,
) -> Tensor
~~~

新增 canonical family API（传入kid时为explicit；bpreshuffle的Python wrapper允许
<code>kid=None</code>，raw仍为exact kid）：

~~~python
opus_gemm_a16w16_launch(
    XQ,
    WQ,
    Y,
    bias=None,
    *,
    kid: int,
    split_k: int = 0,
    workspace: Tensor | None = None,
) -> Tensor

opus_gemm_a8w8_launch(
    XQ,
    WQ,
    Y,
    *,
    kid: int = 2,
) -> Tensor

opus_gemm_a8w8_blockscale_launch(
    XQ,
    WQ,
    Y,
    x_scale,
    w_scale,
    *,
    kid: int = 1,
) -> Tensor

opus_gemm_a8w8_blockscale_bpreshuffle_launch(
    XQ,
    WQ,
    x_scale,
    w_scale,
    Y,
    *,
    kid: int | None = None,
) -> Tensor
~~~

canonical Python blockscale-bpreshuffle接口不能默认写死gfx942的11000。它允许
<code>kid=None</code>，并只在Python按下面顺序解析：

~~~text
explicit kid
  -> 当前shape/arch的 tuned row，且 libtype == "opus"
  -> Python OPUS_DEFAULT_A8W8_BPRESHUFFLE_KID_BY_ARCH
  -> 没有default/registered kernel时报no-registered-kernel
~~~

当前default映射固定为 <code>{gfx942: 11000, gfx950: None, gfx1250: None}</code>。default非空
时必须同时满足：属于canonical registry、属于正确family/arch，并位于mandatory compile set；
测试在生成前验证这个不变量。未来某架构有多个kernel时，tuned CSV决定shape winner；default只
负责没有tuned row时的明确保底，不在runtime猜“第一个kid”。

解析完成后，私有 <code>_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw</code> 和C++ raw仍
必须接收确定的整数kid。这样runtime policy仍在Python，C++不恢复shape lookup或heuristic。
旧 <code>opus_gemm_a8w8_blockscale_bpreshuffle_tune</code> compatibility wrapper可在兼容期继续
保留 <code>kernelId=11000</code>，但它只表达旧gfx942调用合同。

这里分三个时间点：接口ABI现在建立；未来kernel加入源码/registry后由build或首次JIT把它
编入模块；实际调用时Python才根据当前GPU和shape选择kid。runtime可以延后“选择哪个已编译
kid”，但不能让一个已经加载、完全不含新kernel符号的旧二进制凭kid数字凭空获得未来kernel；
kernel落地后至少需要重新build/JIT。

四个wrapper都返回传入的 Y。新的launch API不偷偷分配输出；只有高层
<code>gemm_a16w16_opus</code> 和旧 gfx942 tune兼容wrapper可以按其原有合同分配 Y。

对应私有binding命名：

~~~text
_opus_gemm_a16w16_launch_raw
_opus_gemm_a8w8_launch_raw
_opus_gemm_a8w8_blockscale_launch_raw
_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw
~~~

### 4.4 兼容策略

| 旧接口 | 兼容期行为 | 最终raw状态 |
|---|---|---|
| Python <code>opus_gemm_a16w16_tune</code> | 至少一个发布周期发出DeprecationWarning，适配kernelId/splitK和旧位置参数 | 旧C++/pybind符号删除 |
| Python <code>opus_gemm_a8w8_blockscale_bpreshuffle_tune</code> | 至少一个发布周期发出DeprecationWarning，可保留Y=None自动分配 | 旧C++/pybind符号删除 |
| <code>aiter.ops.deepgemm.opus_gemm_a16w16_tune</code> | 继续作为旧路径shim，内部直接调用canonical launch，避免双重warning | 无独立C++符号 |
| Python <code>opus_gemm_workspace_init</code> | 继续deprecated no-op；移除周期另行决定 | C++/pybind中不存在 |
| C++ generic <code>opus_gemm</code> | 不保留别名 | 删除 |

本仓没有直接调用旧 C++ raw符号的源码。任务二不为未声明的外部C++ ABI复活旧policy；如果
发布方确认必须维持外部二进制ABI，应单独立项一个只转发、不含selector的compatibility TU，
不能把兼容逻辑塞回新raw entry。

### 4.5 最终职责矩阵

| 能力 | Python | C++ |
|---|---:|---:|
| runtime tuned CSV lookup | 唯一实现 | 无 |
| a16w16 heuristic | 唯一实现 | 无 |
| explicit/tuned/heuristic优先级 | 是 | 无 |
| framework fallback | 是 | 无 |
| output分配 | 高层API | 无 |
| workspace分配 | a16 family adapter | 无 |
| friendly shape/dtype/layout检查 | 是 | 最终安全复核 |
| arch/family/kid复核 | 是 | 是 |
| kid到launcher | 否 | generated strict dispatch |
| kargs和kernel launch | 否 | 是 |

## 5. 任务二不可破坏的不变量

### 5.1 任务一不变量

1. selector顺序不变；
2. actual kid继续是workspace capability、dtype、tile和layout的唯一事实；
3. <code>_init_a16w16_workspace()</code> 继续是唯一分配点；
4. gfx942 redirect继续在分配前完成；
5. gfx1250 two-stage继续为496个BF16 workspace kids；
6. gfx1250 fused继续为1378个kids，保留780 BF16 / 598 FP32和tile-major layout；
7. a16 generated 5参数/6参数函数指针继续严格分表；
8. 不增加全局Tensor cache、raw allocator或prewarm。

### 5.2 subset compile 不变量

删除runtime shape table时必须保留：

~~~python
S = (csv_kids | sidecar_kids | HEURISTIC_DEFAULT_KIDS) & valid_kids
~~~

还必须保留：

- <code>GPU_ARCHS</code> arch filter；
- per-arch <code>OPUS_MANDATORY_A8_KIDS</code>：当前为
  <code>{gfx950: {1,2}, gfx942: {11000}, gfx1250: set()}</code>；
- CSV路径展开和 <code>libtype == "opus"</code> kid提取；
- compiled-kids sidecar读取和写回；
- <code>heuristic_kids_for_arch()</code> 完整性断言；
- kernel_tag开发过滤后重新加入heuristic默认kids；
- gfx1250 fused kid生成和workspace dispatch；
- <code>opus_build_archs.h</code> 的per-arch宏。

<code>OPUS_MANDATORY_A8_KIDS</code> 是build/JIT-time compile-set保底，不是Python调用方
必须填写的参数。当前11000必须被编入gfx942模块，才能让 <code>kid=None</code> 在没有tuned row
时解析到per-arch default。gfx950/gfx1250现在保持空集合；以后新增kernel后，通过mandatory、
compiled-kids sidecar或 <code>libtype == "opus"</code> CSV中的任一来源进入编译集。

删除的是“把CSV shape行烘焙进C++ runtime表”，不是“CSV决定哪些kid需要编译”。

### 5.3 typed dispatch 不变量

最终生成表至少分成五种函数指针合同：

1. a16w16 non-workspace；
2. a16w16 workspace；
3. gfx950 a8w8 no-scale；
4. gfx950 a8w8 blockscale（plain WQ）；
5. a8w8 blockscale bpreshuffle统一函数指针合同，并在gfx942、gfx950、gfx1250下各有独立表。

第5类当前只有gfx942表包含kid 11000；gfx950/gfx1250必须生成
<code>std::array&lt;Entry, 0&gt;</code> 等标准C++合法空表，不能生成零长度C数组，也不能完全省略
符号。即使两个a8 launcher表面参数个数相同，也不能因scale layout不同而混成一个family。

统一bpreshuffle函数指针类型固定为：

~~~cpp
using OpusA8W8BlockscaleBpreshuffleKernel = void (*)(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    aiter_tensor_t& Y);
~~~

未来arch-specific launcher可以有不同dtype、shape和tile约束，但不能改变这个family ABI；差异
留在generated launcher的物理检查与kargs构造中。

### 5.4 dtype、layout和bias不变量

a16w16继续允许：

~~~text
XQ/WQ: innermost K stride == 1，row pitch >= K
batch==1，或batch stride == rows * row pitch
Y:     contiguous [batch,M,N]
~~~

不能统一收紧成所有输入完全contiguous，否则会误拒绝合法的padded leading dimension。

bias继续使用输出feature维：

~~~text
[N] 或 [batch,N]
~~~

并保留现有per-kid规则：

- gfx950/gfx942只有bias-aware kid可接收bias；
- gfx1250 reduce继续允许 FP32 bias 配 BF16 Y；
- 不把gfx1250能力错误推广到其他架构；
- unsupported bias不能被静默丢弃。

### 5.5 a8安全不变量

- XQ、WQ、Y和scales必须在同一device；
- XQ/WQ必须都是FP8；
- gfx950 Y必须FP32；
- gfx942 blockscale Y必须BF16；
- only-one-scale必须直接报错，不允许退到no-scale；
- scale shape按对应group合同验证；
- unknown kid必须失败，kid 1不能进入no-scale，kid 2不能进入blockscale，kid 11000不能跨arch；
- generated launcher保留prefetch、tile和物理stride最后防线。

## 6. 任务二详细施工顺序

每个Step结束都必须是可生成、可import、测试可运行的状态。B1至B5可以连续开发，但不能把
raw符号已删除、Python调用方尚未迁移的中间态提交。

### Step B0：冻结Task1和接口golden

目标：在删旧符号和旧C++ policy前保存可比较证据。

只读记录：

~~~text
git status --short
git diff --stat
git diff --name-status
当前三个raw/pybind签名
全部旧名字调用点
当前generated header清单
当前canonical registry计数
~~~

测试golden：

1. 固化 <code>op_tests/test_opus_dispatch.py</code> 的selector优先级、gfx942 redirect和
   framework fallback用例；
2. 固化任务一workspace shape/dtype/capacity测试；
3. 为当前a16 public wrapper和raw参数顺序增加signature测试；
4. 为 gfx950 kid 1/2 和 gfx942 kid 11000 增加family-contract测试，并固定当前
   gfx950/gfx1250 OPUS bpreshuffle capability为空的golden；
5. 有对应空闲GPU时保存三条OPUS a8 raw路径的数值golden；没有GPU时只保存CPU合同测试，不把skip
   写成通过；
6. 保存当前generated dispatch中每架构kid集合，供改名后做集合等价比较。

B0只建立比较基线，不修改selector、workspace或kernel。

### Step B1：增量加入canonical a16w16 launch接口

修改：

~~~text
csrc/opus_gemm/include/opus_gemm.h
csrc/opus_gemm/opus_gemm.cu
csrc/include/rocm_ops.hpp
csrc/pybind/opus_gemm_pybind.cu
aiter/ops/opus/gemm_op_a16w16.py
aiter/ops/opus/__init__.py
op_tests/test_opus_interfaces.py
~~~

动作：

1. 把当前 a16 raw实现主体抽成一个内部helper，保持所有检查和dispatch不变；
2. 新增 C++ <code>opus_gemm_a16w16_launch</code> 调用这个helper；
3. 新增对应pybind宏，参数名使用
   <code>XQ,WQ,Y,bias,workspace,kid,split_k</code>；
4. 新增 <code>_opus_gemm_a16w16_launch_raw</code>；
5. 新增 public <code>opus_gemm_a16w16_launch</code>，继续复用
   <code>select_launch_config</code>、layout检查、
   <code>_init_a16w16_workspace</code> 和统一launch helper；
6. 保持 <code>gemm_a16w16_opus</code> 的签名、选择和fallback完全不变；
7. 暂时保留旧 C++ raw符号，以便B1形成additive、可比较检查点。

B1验收：

- 新旧a16 Python入口对同一显式kid解析出相同actual kid和split-K；
- 自动分配和caller workspace传入完全相同；
- raw缺workspace、错dtype、短一个元素等错误不变；
- 新接口可以被 <code>torch.compile</code> fake注册正确识别。

### Step B2：迁移a16调用方并移除旧raw tune符号

修改：

~~~text
aiter/tuned_gemm.py
csrc/opus_gemm/opus_gemm_tune.py
csrc/gemm_a16w16/gemm_a16w16_tune.py
aiter/ops/deepgemm.py
aiter/ops/opus/gemm_op_a16w16.py
aiter/ops/opus/__init__.py
csrc/opus_gemm/include/opus_gemm.h
csrc/opus_gemm/opus_gemm.cu
csrc/include/rocm_ops.hpp
csrc/pybind/opus_gemm_pybind.cu
op_tests/test_opus_dispatch.py
op_tests/test_opus_workspace.py
op_tests/test_opus_graph.py
op_tests/test_opus_gfx950_exhaustive.py
~~~

动作：

1. 四个仓内生产/调优调用方全部改调
   <code>opus_gemm_a16w16_launch</code>；
2. tuner变量和日志中的 tune可以保留“调优流程”语义，但raw函数名必须改成launch；
3. 旧Python <code>opus_gemm_a16w16_tune</code> 改为纯compat wrapper：
   解析旧位置参数和 <code>kernelId/splitK</code>，发一次warning，然后调canonical launch；
4. deepgemm shim直接调canonical launch，避免先在deepgemm warning、再在旧wrapper warning；
5. 测试monkeypatch目标统一改为 <code>_opus_gemm_a16w16_launch_raw</code>；
6. 删除 C++ <code>opus_gemm_a16w16_tune</code> 声明、实现和旧pybind注册；
7. C++错误信息和注释统一改为launch/kid命名；
8. 不改变 <code>LaunchConfig</code>、workspace shape或generated launcher签名。

B2结束时，旧名字只允许存在于Python兼容函数、兼容测试和迁移文档中。

### Step B3：拆出三条a8w8 family launch

修改：

~~~text
csrc/opus_gemm/opus_gemm_common.py
csrc/opus_gemm/gen_instances.py
csrc/opus_gemm/codegen/gen_instances_gfx950.py
csrc/opus_gemm/codegen/gen_instances_gfx942.py
csrc/opus_gemm/codegen/gen_instances_gfx1250.py
csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh
csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh
csrc/opus_gemm/include/opus_gemm.h
csrc/opus_gemm/opus_gemm.cu
csrc/include/rocm_ops.hpp
csrc/pybind/opus_gemm_pybind.cu
aiter/ops/opus/gemm_op_a8w8.py
aiter/ops/opus/__init__.py
csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py
aiter/ops/gemm_op_a8w8.py
op_tests/test_opus_interfaces.py
op_tests/test_opus_workspace.py
~~~

registry动作：

1. 扩展 <code>get_kernel_instance(arch,family,kid,output_dtype=None)</code> 的窄查询，使其识别
   <code>a8w8</code>、<code>a8w8_blockscale</code> 和
   <code>a8w8_blockscale_bpreshuffle</code>；
2. logical family必须映射到明确的kernel_tag集合：gfx950 logical
   <code>a8w8_blockscale</code> 对应当前内部 <code>a8w8_scale</code> tag，gfx942 logical
   blockscale bpreshuffle family对应当前
   <code>a8w8_blockscale_bpreshuffle_singlebuf</code> tag；
3. 为blockscale bpreshuffle显式登记三个arch槽位；gfx942 tag集合当前非空，gfx950/gfx1250
   tag集合当前为空，空集合是合法capability状态；
4. 仍以 <code>(arch,family,kid,Y.dtype)</code> 校验，不按裸kid区间或名字substring猜family；
5. 不为a8新增workspace能力，也不建立第二份metadata投影。

生成器动作：

1. 把 <code>A8W8_SCALE_HOST_EXTRA</code> 重命名为
   <code>A8W8_BLOCKSCALE_HOST_EXTRA</code>，并从两个optional tensor改为两个必选
   <code>aiter_tensor_t&amp;</code>；
2. 同步修改gfx950 blockscale和gfx942 blockscale generated launcher、manifest和显式host
   instantiation；gfx942 bpreshuffle launcher参数顺序统一为
   <code>XQ,WQ,x_scale,w_scale,Y</code>，与family函数指针完全一致；
3. 新增/改造 <code>gen_a8w8_kid_dispatch()</code>；
4. 生成独立typed表：gfx950 no-scale、gfx950 blockscale，以及按
   <code>(arch,Y.dtype)</code> 分开的blockscale-bpreshuffle表；后者在gfx942/gfx950/gfx1250
   都生成table和size，当前只有gfx942 BF16表含kid 11000；
5. kid 1、2、11000都来自canonical registry，不能在
   <code>opus_gemm.cu</code> 再硬编码具体kernel symbol；
6. table按kid排序，unknown kid严格失败；
7. mandatory compile-set当前强制加入gfx950 kid 1/2和gfx942 kid 11000；
8. 空family用 <code>std::array&lt;Entry,0&gt;</code> 和size=0表示，host TU不得引用不存在的
   kernel symbol；单arch和multiarch构建都必须通过。

C++动作：

1. 新增 <code>opus_gemm_a8w8_launch</code>，只接受gfx950 no-scale kid 2；
2. 新增 <code>opus_gemm_a8w8_blockscale_launch</code>，只接受gfx950 blockscale
   plain-WQ kid 1；
3. 新增稳定的 <code>opus_gemm_a8w8_blockscale_bpreshuffle_launch</code>，runtime先按arch和
   Y dtype选择per-arch表，再按exact kid查找；当前gfx942 kid 11000成功，gfx950/gfx1250
   返回no-registered-kernel；
4. 公共family入口只做device、scale存在性、arch/family/dtype/kid复核；
5. generated launcher做各arch自己的dtype、shape、batch、tile、layout和stride物理复核；
6. 区分三类错误：module未编入当前arch、当前arch family表为空、非空表中unknown kid；
7. 暂时保留generic <code>opus_gemm</code>，用相同输入做新旧数值对照；
8. 删除旧gfx942 C++ tune符号，保留旧Python wrapper。

Python动作：

1. 在 <code>gemm_op_a8w8.py</code> 增加三个私有raw binding和三个canonical wrapper；
2. gfx950入口要求调用方显式传Y，不增加第二套auto-selector；
3. canonical bpreshuffle wrapper接受 <code>kid=None</code>，复用现有A8 config读取并按
   explicit -&gt; OPUS tuned row -&gt; per-arch Python default解析；私有raw只接收resolved int kid；
4. gfx942旧tune wrapper保留原有Y=None和 <code>kernelId=11000</code> 行为，内部适配到
   canonical launch；
5. <code>__init__.py</code> 在gfx942/gfx950/gfx1250都导出同一新名字；只有三者之外的arch
   使用unsupported-arch stub，空family由运行时capability错误表达；
6. gfx942 tuner和高层路由的 <code>libtype == "opus"</code> 分支改用canonical launch，并写成
   arch-neutral逻辑，以便以后gfx950/gfx1250 OPUS row无需再改分支；
7. 高层路由其余 <code>ck/cktile/asm/triton/gluon/flydsl</code> 分支保持原样，不能把gfx950或
   gfx1250 blockscale bpreshuffle错误导向OPUS raw；
8. 单scale、错scale shape、错dtype、错kid和跨arch全部增加负例；
9. 增加三架构路由单测：gfx942 OPUS row调用新raw，gfx950 CK/ASM row和gfx1250
   FlyDSL/Triton row不调用OPUS raw。

### Step B4：删除generic opus_gemm mega entry

前置条件：gfx950 kid 1/2通过新family entry完成数值对照，且仓内没有generic OPUS调用方。

修改：

~~~text
csrc/opus_gemm/include/opus_gemm.h
csrc/opus_gemm/opus_gemm.cu
csrc/include/rocm_ops.hpp
csrc/pybind/opus_gemm_pybind.cu
aiter/ops/opus/gemm_op_a16w16.py
op_tests/test_opus_dispatch.py
op_tests/test_opus_interfaces.py
~~~

删除：

- C++ <code>opus_gemm()</code> 声明和实现；
- <code>OpusScaleKernel</code>、<code>OpusNoscaleKernel</code> 及两个hardcoded dispatch
  helper；
- <code>OPUS_GEMM_PYBIND</code>；
- pybind模块中的generic注册；
- <code>_gen_opus_gemm_bf16_dispatch_fake_tensors</code>；
- <code>_opus_gemm_bf16_dispatch</code>；
- <code>group_layout</code> 这项未使用OPUS参数；
- 所有“generic BF16暂时拒绝”的过渡注释。

本步不删除 <code>deepgemm</code> CK entry，也不修改其他模块中恰好同名的Python调度函数。

### Step B5：删除C++ runtime shape lookup和heuristic

修改：

~~~text
csrc/opus_gemm/gen_instances.py
csrc/opus_gemm/opus_gemm_common.py
csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh
csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh
op_tests/test_opus_workspace.py
op_tests/test_opus_dispatch.py
~~~

删除：

~~~text
csrc/opus_gemm/include/gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh
csrc/opus_gemm/include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh
csrc/opus_gemm/include/gfx1250/opus_gemm_heuristic_dispatch_gfx1250.cuh
~~~

<code>gen_instances.py</code> 的精确动作：

1. 删除 <code>gen_lookup_dict()</code>；
2. 删除 <code>get_tune_dict()</code>；
3. 删除 <code>_combined_opus_tuned.csv</code> 临时拼接；
4. 删除“CSV shape行 -&gt; opus_gemm_lookup.h”的烘焙块；
5. 删除 <code>gen_instances()</code> 中runtime lookup生成调用；
6. 停止生成 <code>opus_gemm_lookup.h</code>；
7. 删除只服务上述路径的 <code>default_kernels_dict</code> import和定义；
8. 保留CSV kid提取、sidecar、arch filter、mandatory a8 kids和heuristic-default invariant；
9. 更新日志，不再打印“baked shape entries”，改成只报告compile-set来源；
10. 保持两次生成字节稳定。

arch header的精确动作：

1. 删除 <code>OpusA16W16Shape</code> 和runtime shape entry；
2. 删除 <code>find_shape_kid()</code>；
3. 删除 <code>opus_select_a16w16_kid_gfx*</code>；
4. 删除gfx942 header内嵌的legacy heuristic函数；
5. 删除gfx950/gfx1250 heuristic include和4GiB policy probe；
6. 保留strict kid查找、workspace membership和typed launcher dispatch；
7. 4GiB物理安全检查继续留在Python筛选和generated launcher，不能随policy probe一起删掉。

生成表重命名：

~~~text
opus_gemm_a16w16_tune_lookup.h
  -> opus_gemm_a16w16_kid_dispatch.h

opus_gemm_a8w8_tune_lookup.h
  -> opus_gemm_a8w8_kid_dispatch.h
~~~

最终宏命名固定为：

~~~text
GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_<ARCH>_<DTYPE>
GENERATE_A16W16_WORKSPACE_KID_DISPATCH_<ARCH>
GENERATE_A8W8_NOSCALE_KID_DISPATCH_GFX950
GENERATE_A8W8_BLOCKSCALE_KID_DISPATCH_GFX950
GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_<ARCH>_<DTYPE>
GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_<ARCH>_<DTYPE>_SIZE
~~~

其中 <code>ARCH</code> 至少生成 GFX942/GFX950/GFX1250，dtype集合由family支持的公开dtype
集合固定；当前只有GFX942_BF16 size非零。宏改名必须同步三个arch header和codegen测试；
不能通过兼容宏永久保留 tune/lookup混合命名。

### Step B6：统一family校验并完成调用方、文档收尾

本步不新增通用大对象或新的policy层。校验继续放在：

~~~text
aiter/ops/opus/gemm_op_a16w16.py
aiter/ops/opus/gemm_op_a8w8.py
generated host launcher
csrc/opus_gemm/opus_gemm.cu
~~~

a16检查：

- 保留当前2D/3D normalize和padded leading stride能力；
- explicit/tuned/heuristic继续共用同一个selector合法性规则；
- canonical explicit launch必须经过actual-kid解析，不能直接绕过gfx942 redirect；
- error message统一使用launch/kid/split_k术语；
- workspace和bias规则完全沿用任务一。

gfx950 no-scale检查：

- XQ/WQ/Y都是3D且batch、M、N、K相互匹配；
- XQ/WQ为FP8，Y为FP32；
- A/B K-contiguous，Y contiguous；
- 保留generated launcher现有
  <code>ceil_div(K,B_K) &gt;= 2</code>、even-loop和K-even约束；
- kid只能属于gfx950 a8w8 no-scale family。

gfx950 blockscale检查：

- 在no-scale基础上要求两个FP32 scale同时存在；
- scale与XQ同device且contiguous；
- K和N满足128分组合同；
- x_scale和w_scale形状匹配第3.6节；
- 对prefetch所需最小/奇偶K-tile做host检查，避免kernel内负tile或越界预取；
- kid只能属于gfx950 a8w8 blockscale family；
- 不接受bias和group_layout。

blockscale bpreshuffle公共检查：

- Python <code>kid=None</code>只做runtime OPUS winner解析，不把shape表重新烘焙进C++；
- resolved kid必须属于当前runtime arch的logical family和Y dtype；
- 五个Tensor必须在同一device，两个scale都必须存在；
- C++先选择当前arch/dtype表，空表报告no-registered-kernel，非空表unknown kid报告
  unknown-kid；绝不尝试其他arch表；
- 公共router不硬编码gfx942的BF16、FP32 scale、batch=1或128 tile规则。

当前gfx942 blockscale bpreshuffle generated launcher检查：

- 保留2D或3D输入，但batch必须为1；
- XQ/WQ为FP8，Y为BF16，scales为FP32；
- <code>N % 128 == 0</code>、<code>K % 128 == 0</code>；
- scale shape、WQ shape/stride和contiguous明确验证；
- bpreshuffle是WQ内容语义，不能从Tensor元数据证明；Python API命名和文档必须明确要求，
  数值测试使用真实shuffle后的weight验证，不能假装shape检查已经验证内容布局；
- kid只能属于gfx942 blockscale bpreshuffle family。

当前gfx950/gfx1250 blockscale bpreshuffle检查：

- Python属性、pybind和C++公共符号必须存在；
- <code>kid=None</code> 且没有OPUS tuned/default kid时报告当前arch无已注册OPUS kernel；
- 显式任意kid也不能越过空family table或借用gfx942的11000；
- 不影响同名AITER高层接口继续选择CK/ASM/FlyDSL/Triton/Gluon。

README更新：

~~~text
aiter/ops/opus/README.md
csrc/opus_gemm/README.md
~~~

必须同步：

- 当前任务一实际workspace布局和dtype；
- gfx1250 fused已存在；
- 四个canonical launch API；
- 旧Python名字的兼容周期；
- runtime policy只在Python；
- build-time CSV仍参与subset compile；
- 不再提不存在的generic BF16入口和旧生成表。

### Step B7：静态、codegen、CPU、GPU、ABI和性能验收

#### B7.1 静态检查

~~~bash
git diff --check
python3 -m py_compile \
  aiter/ops/opus/gemm_op_a16w16.py \
  aiter/ops/opus/gemm_op_a8w8.py \
  aiter/ops/opus/__init__.py \
  csrc/opus_gemm/gen_instances.py
~~~

目标扫描：

~~~bash
rg -n "OPUS_GEMM_PYBIND|_opus_gemm_bf16_dispatch" \
  csrc/opus_gemm csrc/pybind csrc/include aiter/ops/opus

rg -n "opus_gemm_lookup\.h|opus_select_a16w16_kid|find_shape_kid" \
  csrc/opus_gemm

rg -n "opus_gemm_a16w16_tune|opus_gemm_a8w8_blockscale_bpreshuffle_tune" \
  csrc/opus_gemm csrc/pybind csrc/include
~~~

第一组在全部列出的范围内必须为空；后两组在生产 C++/pybind 范围内必须为空。由于后两组
原样命令也会读取 Python 生成器，允许它们仅命中 `gen_instances.py` 的陈旧生成文件清理白名单：
`opus_gemm_lookup.h`、`opus_gemm_a16w16_tune_lookup.h` 和
`opus_gemm_a8w8_tune_lookup.h`。这些字符串只用于复用 blob 目录时删除旧产物，不是生产引用；
除该白名单外必须为空，也不得通过拆分字符串来规避扫描。旧 Python 兼容函数名允许存在。

#### B7.2 codegen检查

分别对 gfx942、gfx950、gfx1250：

1. 生成到全新临时目录；
2. 再生成到第二个临时目录并比较字节稳定性；
3. 核对 <code>opus_build_archs.h</code> 只包含目标arch；
4. 核对a16 workspace/non-workspace表集合与B0完全一致；
5. 核对gfx950 kid 1/2和gfx942 kid 11000位于正确typed表；
6. 核对bpreshuffle的gfx950/gfx1250各dtype表以标准C++ size=0形式存在且不引用kernel符号；
7. 用codegen fixture向gfx950/gfx1250各注入一个合成registry实例，证明只增加
   registry/emitter即可让对应表从0变1，公共header/pybind/Python签名无需修改；
8. 核对不再生成 <code>opus_gemm_lookup.h</code>；
9. 编译全部generated host TU、device TU和reduce TU；
10. gfx1250 1378 fused kids和496 two-stage kids数量、dtype分布不变。

#### B7.3 CPU测试

~~~bash
python3 -m pytest -q \
  op_tests/test_opus_interfaces.py \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_gemm_codegen.py
~~~

覆盖：

- 新接口signature、return Y、三架构导出和三者之外的unsupported-arch stub；
- bpreshuffle <code>kid=None</code> 的explicit/tuned/per-arch-default/no-kernel解析；
- gfx950/gfx1250空OPUS family是capability错误而不是缺少Python/pybind符号；
- 旧Python wrapper warning只发一次；
- old keyword/position参数适配；
- exact family/kid拒绝；
- selector顺序和tuned-row原子回退；
- workspace shape/dtype/capacity不变；
- generated表命名、类型和kid集合；
- subset compile公式及mandatory kid；
- generic/private legacy binding不存在。

#### B7.4 GPU测试

gfx950：

- a16 focused suite保持任务一结果；
- 48/48 workspace kids继续通过；
- full 140-kid相对任务一不得新增失败；
- 已知mono FP32 10项继续单列，不伪装成任务二失败或通过；
- kid 2 no-scale数值、shape和unknown-kid负例；
- kid 1 blockscale数值、两个scale shape/dtype/device和single-scale负例；
- 高层 <code>gemm_a8w8_blockscale_bpreshuffle</code> 的CK/CKTile/ASM/Triton代表路由保持，
  并确认不会误调OPUS bpreshuffle raw；
- 直接调用预留OPUS bpreshuffle接口时，当前明确报告gfx950无registered kernel；
- graph capture/replay和双stream至少覆盖一个workspace kid和两个a8 family。

gfx942：

- a16 BF16/FP32 workspace代表kid；
- 10210/10213 redirect和10216拒绝；
- kid 11000 blockscale bpreshuffle数值；
- direct canonical wrapper的 <code>kid=None</code> 在无tuned row时解析gfx942 default 11000；
- 至少一条 <code>libtype == "opus"</code> tuned row经高层接口到达新的canonical raw；
- 2D/3D batch=1、scale shape和exact tile负例；
- graph和双stream。

gfx1250：

- two-stage BF16 workspace代表kid；
- fused BF16和FP32 workspace代表kid；
- batch大于1拒绝；
- compile-time fused split-K不受runtime splitK改变；
- 高层blockscale bpreshuffle分别覆盖可用的FP8-E8M0 128-block FlyDSL路径和FP32-scale
  Triton/Gluon路径，并确认不会误调OPUS bpreshuffle raw；
- 直接调用预留OPUS bpreshuffle接口时，当前明确报告gfx1250无registered kernel；
- graph、双stream和caller workspace复用。

没有对应架构硬件时必须记录“未执行”，不能用compile或skip替代实机通过。

#### B7.5 ABI和符号检查

构建 <code>module_deepgemm_opus</code> 后核对：

- pybind只暴露四个新的raw C++名字；
- blockscale-bpreshuffle raw/public名字在gfx942、gfx950、gfx1250构建中一致存在；
- generic <code>opus_gemm</code> raw属性不存在；
- 旧 C++ tune名字不存在；
- public Python仍导出约定中的deprecated wrapper；
- <code>aiter</code> 顶层star import不会因unsupported arch中断后续op import；
- fake tensor schema与实际参数顺序一致。

#### B7.6 性能检查

接口重构不应改变kernel。对每个可用arch使用相同输入、相同actual kid、相同split-K：

1. warmup后至少记录多轮median；
2. 比较任务一端点与任务二端点；
3. a16 high-level、a16 explicit和三条a8 family分别记录；
4. 超过正常噪声带的退化必须定位到Python adapter、C++检查或生成器变化；
5. 不以改回generic入口解决性能问题。

## 7. 逐文件修改清单

### 7.1 C++入口和pybind

| 文件 | 修改 |
|---|---|
| <code>csrc/opus_gemm/include/opus_gemm.h</code> | 声明四个family launch；删除generic和旧raw tune声明 |
| <code>csrc/opus_gemm/opus_gemm.cu</code> | family router、arch/dtype安全检查、strict kid dispatch；删除generic policy |
| <code>csrc/include/rocm_ops.hpp</code> | 新增四个launch pybind宏；删除generic和旧tune宏 |
| <code>csrc/pybind/opus_gemm_pybind.cu</code> | 只注册四个新raw entry |

### 7.2 Python入口和调用方

| 文件 | 修改 |
|---|---|
| <code>aiter/ops/opus/gemm_op_a16w16.py</code> | 新raw和public launch；保留高层API；旧tune变compat；删除generic私有binding |
| <code>aiter/ops/opus/gemm_op_a8w8.py</code> | 新增gfx950两入口；新增三架构稳定bpreshuffle wrapper和 <code>kid=None</code> Python解析；旧gfx942 tune变compat |
| <code>aiter/ops/opus/__init__.py</code> | 三个OPUS arch统一导出新名字；导出compat名字和真正unsupported-arch stubs |
| <code>aiter/tuned_gemm.py</code> | 改调canonical a16 launch |
| <code>csrc/opus_gemm/opus_gemm_tune.py</code> | tuner改调canonical a16 launch |
| <code>csrc/gemm_a16w16/gemm_a16w16_tune.py</code> | 汇总tuner改调canonical a16 launch |
| <code>aiter/ops/deepgemm.py</code> | 旧路径shim直接适配canonical launch |
| <code>csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py</code> | gfx942 tuner改调canonical a8 launch |
| <code>aiter/ops/gemm_op_a8w8.py</code> | tuned row的OPUS分支改调canonical a8 launch |

### 7.3 registry、codegen和arch dispatch

| 文件 | 修改 |
|---|---|
| <code>csrc/opus_gemm/opus_gemm_common.py</code> | 保持canonical registry；删除只服务runtime shape表的default mapping |
| <code>csrc/opus_gemm/gen_instances.py</code> | 删除shape lookup生成；保留compile set；重命名a16/a8 kid dispatch；生成三架构bpreshuffle表和合法空表 |
| <code>csrc/opus_gemm/codegen/gen_instances_gfx950.py</code> | gfx950 blockscale改必选scale引用；保留kernel/kargs算法和现有内部tag映射 |
| <code>csrc/opus_gemm/codegen/gen_instances_gfx942.py</code> | gfx942 blockscale scale参数改必选引用；保留kid11000物理合同 |
| <code>csrc/opus_gemm/codegen/gen_instances_gfx1250.py</code> | 当前不新增a8 kernel；接入空family codegen槽位，未来只需注册emitter |
| 三个 <code>opus_gemm_arch_gfx*.cuh</code> | 只保留strict typed kid dispatch；三者均声明bpreshuffle per-dtype表查询 |
| 三个 <code>opus_gemm_heuristic_dispatch_gfx*.cuh</code> | 删除 |

### 7.4 测试和文档

| 文件 | 修改 |
|---|---|
| <code>op_tests/test_opus_interfaces.py</code> | 新增接口、兼容、family合同和符号测试 |
| <code>op_tests/test_opus_dispatch.py</code> | 新raw名字；删除generic probe；保留selector golden |
| <code>op_tests/test_opus_workspace.py</code> | 新生成表名；workspace不变量 |
| <code>op_tests/test_opus_graph.py</code> | 新raw名字和family graph用例 |
| <code>op_tests/test_opus_gfx950_exhaustive.py</code> | 新public launch名；保留canonical registry枚举 |
| <code>aiter/ops/opus/README.md</code> | 最终Python API和实际workspace合同 |
| <code>csrc/opus_gemm/README.md</code> | 最终C++/codegen边界 |

## 8. 风险和防护

| 风险 | 结果 | 防护 |
|---|---|---|
| 把runtime lookup和kid dispatch一起删 | raw无法按kid launch | 先做生成集合golden，只删shape表 |
| 删除CSV读取过多 | tuned或heuristic kid未编译 | 固定subset公式并做集合测试 |
| a16 rename时绕过selector | gfx942错误dtype workspace | public launch仍走LaunchConfig和actual kid |
| a8 blockscale参数继续optional | 单scale静默进错family | C++、pybind、generated launcher全部改必选引用 |
| 三种a8物理合同混表 | 函数指针或layout错配 | family-specific typed table和strict membership |
| 把“没有OPUS kid”误写成“没有高层能力” | gfx950/gfx1250现有bpreshuffle路由被删或误拒绝 | 两层capability matrix；只迁移 <code>libtype == "opus"</code> 分支 |
| canonical接口默认写死11000 | 未来gfx950/gfx1250误用gfx942 kid | Python默认 <code>kid=None</code>，runtime按arch解析后再进raw |
| 空family完全不生成table | 未来加kernel仍需修改router/ABI | 三架构均生成size-aware typed table，空表使用标准 <code>std::array&lt;Entry,0&gt;</code> |
| runtime选到未编译kid | exact dispatch找不到launcher | CSV/sidecar/mandatory compile-set校验，错误区分no-kernel与unknown-kid |
| 统一要求全部contiguous | 拒绝合法padded lda | 保留a16现有stride合同 |
| 统一bias规则 | gfx1250能力丢失或其他arch错误放宽 | per-kid规则和generated guard |
| 旧Pythonshim形成第二套policy | 新旧行为漂移 | shim只做参数改名并调用canonical launch |
| C++旧符号长期别名 | 旧命名和ABI永久存在 | 只保留Python兼容层 |
| 任务二顺手修mono FP32 | 无法区分接口回归和kernel修复 | mono问题单独任务，不在本计划修改 |
| dirty工作树被清理 | 任务一成果丢失 | 逐文件patch，禁止reset/checkout |

## 9. Definition of Done

### 9.1 任务一冻结条件

任务一在本文中的“确定版本”意味着：

- Python先解析actual kid，再准备exact-kid typed workspace；
- Torch拥有调用级workspace，C++不分配也不保留；
- gfx942/gfx950/gfx1250 two-stage和gfx1250 fused都进入统一actual-kid流程；
- 当前registry数量和dtype分布保持；
- gfx950 48/48 workspace kid的已有证据保持有效；
- 任务一开放项继续明确列出。

它不意味着gfx950全140 kid零失败，也不意味着gfx942/gfx1250实机和性能已经通过。

### 9.2 任务二完成条件

- <code>gemm_a16w16_opus</code> 行为不变；
- 四个family-specific canonical launch在C++、pybind和Python对齐；
- a16w16在gfx942/gfx950/gfx1250的现有family全部保留，三架构codegen kid集合与任务一等价；
- a8w8按第3.5节能力矩阵支持或明确拒绝，不虚构gfx1250 OPUS A8W8实现；
- <code>opus_gemm_a8w8_blockscale_bpreshuffle_launch</code> 的C++/pybind/Python名字在三个
  OPUS arch上稳定存在，Python默认 <code>kid=None</code>，C++ raw只接收resolved int kid；
- 当前gfx942表含11000，gfx950/gfx1250表为可编译空表；未来表从0变非0不要求修改公共ABI；
- gfx942/gfx950/gfx1250现有高层blockscale-bpreshuffle能力和backend选择不变，只有
  <code>libtype == "opus"</code> 分支完成新raw名字迁移；
- generic C++ <code>opus_gemm</code> 消失；
- 旧C++/pybind tune符号消失；
- 旧Python tune名字只作为有时限的deprecated adapter；
- C++不再包含runtime shape lookup、heuristic或framework fallback；
- Python仍是a16w16唯一runtime policy层；
- generated strict kid dispatch继续存在并按函数类型分表；
- subset compile公式、arch filter、mandatory a8 kids、sidecar和heuristic默认集合全部保留；
- task1 workspace shape、dtype、fused family和graph ownership不变；
- 三种a8物理合同独立，两个blockscale入口的scale不再optional；
- 所有仓内调用方完成迁移；
- static、codegen、CPU、对应架构GPU、ABI和性能结果按第6节记录；
- 相对任务一端点没有新增数值、graph、并发或性能回归。

## 10. 实际开工点

后续实现从 Step B0 开始：

~~~text
B0 冻结Task1和接口golden
 -> B1 增量加入canonical a16 launch
 -> B2 迁移a16调用方并删除旧raw tune
 -> B3 拆出三条a8 family launch
 -> B4 删除generic opus_gemm
 -> B5 删除C++ runtime shape lookup/heuristic
 -> B6 统一family校验并收尾调用方/README
 -> B7 全面验收
~~~

不要回到任务一 A0，不要恢复已删除的workspace抽象，也不要把任务一开放项写成任务二的已通过
结果。

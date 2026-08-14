# OPUS split-K Torch 化续接检查点

更新时间：`2026-08-13 06:14 UTC`

这是一份短上下文恢复文件。下一次继续工作时先读本文件；需要逐文件历史时再读
`docs/task1_detail.md`；需要直接比较最初
`ca68b4f...` C++ 隐式 workspace流程与当前 Torch 管理流程、查看完整45个代码文件端点清单时，
读 `docs/opus_gemm_splitk_workspace_torch_current_flow_changes.md`。不要重新从对话推导 Step 1
至 Step 6，也不要把中间 `WorkspacePlan` 版本当成该对比文档的原始基线。

## 当前最新状态：PR #4320 MXFP8 BMM 已接入统一 A8W8 路径（2026-08-13 06:14 UTC）

- 没有新增`aiter/ops/opus/bmm_op.py`。raw binding、exact-kid launcher和Torch workspace计划
  位于现有`aiter/ops/opus/gemm_op_a8w8.py`；tuned调用方位于现有
  `aiter/ops/batched_gemm_op_a8w8.py`；唯一public仍为`aiter.ops.opus.opus_gemm`；
- 45个上游BMM id以`global_kid = 8000 + upstream_kid`加入统一`kernels_list`，总registry为
  2084项。runtime直接查list并进入private `a8w8_mxscale_bmm` family，不fallback或redirect；
- BMM split-K workspace由Torch FP32 Tensor持有。two-stage和fused launcher、reduce kernel都接收
  direct pointer；BMM生产路径不存在workspace handle、内部allocator或隐藏Tensor cache；
- fresh gfx950 JIT为`/tmp/aiter-pr4320-bmm-fresh2.r7Buc2`，包含140个A16 kid和45个BMM kid；
  A16重点回归`126 passed, 17 skipped`，A16 exhaustive `140 passed`，BMM的45-kid guard、
  auto/caller workspace和graph replay全部通过；
- 接口性能artifact为`/tmp/aiter-pr4320-final-20260813/interfaces_*.log`。相对冻结Task2模块，
  当前A16 direct/graph分别变化`-0.216%`/`-0.177%`；BMM 18个代表shape的tuned路径相对固定
  kid 8000累计耗时下降`60.51%`（token-major）和`60.93%`（batch-major）；
- `aiter/jit/core.py`内容hash仍为`43231ab3cd9ea24caaa6e8535b71455386dbe0f5`。

## 最新接口纠正：直接使用基线已有registry（2026-08-13 04:52 UTC）

- gfx950 `<10000`、gfx942 `10000..19999`、gfx1250 `>=20000`的kid编号段，以及合并后的
  `kernels_list`，在原始基线`ca68b4f...`中已经存在；不能把它们写成Task2新增能力；
- 唯一public `opus_gemm`现在直接执行`kernels_list.get(kid)`。后加的`get_kernel_route()`和
  反向tag map已删除；A16/A8的dtype/layout规则留在现有family文件；
- 没有新增生产Python文件，`aiter/jit/core.py`仍与`ca68b4f...`零diff；
- CPU/接口/workspace为`98 passed, 4 skipped`，物理GPU 4 focused为
  `124 passed, 17 skipped`，gfx950 exhaustive为`140 passed`；
- 改前两轮与改后三轮逐case median对比：public eager
  `1496.886513 -> 1497.065125 us`（`+0.011932%`，`48快/48慢`），graph
  `1113.556335 -> 1114.603673 us`（`+0.094053%`）。判定无性能回退；artifact为
  `/tmp/aiter-opus-thin-router.ILVqXb`。完整解释见性能计划第13节。

## 当前最终边界：`core.py`零改动、无新增生产interface文件（2026-08-13 04:09 UTC）

这是当前最高优先级状态。Task1仍然是“把OPUS A16W16的Split-K workspace交给Torch
管理”；caller-resolved exact-kid只是Task1完成后的接口收敛，不是Task1的新定义。

- `aiter/jit/core.py`不承载任何OPUS特例。当前文件与原始Task1基线`ca68b4f...`的blob均为
  `43231ab3cd9ea24caaa6e8535b71455386dbe0f5`，
  `git diff --quiet ca68b4f... -- aiter/jit/core.py`退出码为0；
- 最终没有新增生产Python interface文件。实现只使用已有的
  `aiter/ops/opus/__init__.py`、`_arch.py`、`gemm_op_a16w16.py`和
  `gemm_op_a8w8.py`；早期新增的`_selector_a16w16.py`、`common.py`和
  `heuristics/`下四个文件全部删除。基线等价的gfx942/gfx950/gfx1250 private heuristic已
  收敛到现有`gemm_op_a16w16.py`，不形成新的public/interface文件；
- A16上层调用保持`explicit -> tuned CSV -> per-arch heuristic -> PyTorch fallback`；tuned或
  heuristic得到最终kid后才调用唯一public，public/C++ exact路径不重新选核；
- mixed pybind/C ABI所需的最小CDLL装载、固定A16参数签名、Tensor descriptor转换和TLS错误
  读取均局限在已有`gemm_op_a16w16.py`。缓存只保存CDLL/function/ABI helper，不保存Tensor、
  data pointer、workspace或stream；
- 首次调用继续走原有private pybind raw，由它完成正常lazy JIT build/rebuild和架构检查；随后
  从同一个`module_deepgemm_opus.so`装载checked C ABI，后续eager调用走局部C ABI。

已在物理GPU 4用空目录`/tmp/aiter-task1-corefree-fresh.jrkldN`验证完整fresh路径：

- 从零生成并编译41个gfx950 subset kid，成功产出并加载
  `module_deepgemm_opus.so`；
- 第一次kid 200、split-K 2调用经pybind构建并通过数值校验，OPUS模块构建/首调约
  `9.838 s`；
- 首调后把Python中的pybind raw替换为“调用即失败”的哨兵，第二次调用仍成功并通过数值
  校验，证明没有回到pybind而是进入局部C ABI；第二次同步wall time约`0.228 ms`；
- C ABI loader cache为`CacheInfo(hits=1, misses=1, maxsize=1, currsize=1)`。

复用full-kid JIT的最终回归仍为private C ABI `13 passed, 2 skipped`、focused
`124 passed, 17 skipped`、gfx950 exhaustive `140 passed`。移出`core.py`后的96项性能防回退
结果为raw eager `1288.246629 us`、public eager `1472.746886 us`、public graph
`1114.347694 us`；相对移出前同口径分别为`-3.092815%`、`-4.555991%`、`+0.002226%`。
完整边界、命令口径和性能解释见
`docs/opus_gemm_next_performance_optimization_plan.md`第12节。

## Caller-resolved exact-kid统一入口已完成（2026-08-13 03:11 UTC）

这是上一阶段的接口检查点；最终文件边界以上一节为准。它取代下方“Task1 public Host性能
最终闭环”作为接口结论；下方scalar-cache数据继续作为统一入口前的权威历史快照。

- Python公开面只保留`opus_gemm(XQ,WQ,Y,*,kid,...)`，kid必传且Y由调用方持有；
- public直接读取基线已有的`kernels_list[kid]`；dtype/layout/scale只用于进入和校验对应family，
  不选核；
- 已删除selector、common tuned lookup、三架构heuristic、旧family公开wrapper和compat shim；
- A16 C ABI、C++ family raw、Torch-owned workspace、scalar launch-plan cache和generated checked
  validator保留；新增public contract cache也只保存registry/dtype/layout/option标量，不保存Tensor；
- focused在物理GPU 4为`124 passed, 17 skipped`；gfx950 full-140 exhaustive为`140 passed`；
- 最终同场性能相对最初C++内部workspace：raw eager `-9.307176%`、public eager
  `-3.237657%`、public graph `-10.557885%`；汇总三项均提升；
- exact-kid route相对统一前family边界使public eager增加`3.752722%`，因此统一前scalar-cache
  版本的public收益更大。完整每版本表和artifact SHA见
  `docs/opus_gemm_next_performance_optimization_plan.md`第11节。

最新性能artifact：

```text
/tmp/aiter-opus-exactkid-final3-original-source-20260813
/tmp/aiter-opus-exactkid-final-public-20260813
/tmp/aiter-opus-exactkid-adjacent-20260813
```

后续若继续优化，应以统一public为唯一生产口径，针对约`0.621 us/项`的route Host成本建立新的
相邻ABBA；不得恢复Tensor/pointer/stream/workspace cache，也不得把私有family入口重新公开。

## Task1 public Host性能最终闭环（2026-08-13 02:11 UTC）

这是caller-resolved exact-kid统一前的历史性能检查点。完整实现、逐项性能和日志哈希见
`docs/opus_gemm_next_performance_optimization_plan.md`第10节；该节取代其中第9.4/9.5节缓存
实施前的“public eager仍有回退”结论。

- 当前Python路径保留device只读信息缓存和等价layout直写，并新增
  `@lru_cache(maxsize=256)`纯标量explicit launch-plan缓存；缓存只保存resolved kid、launch
  split-K和workspace shape/dtype，不保存Tensor、地址、workspace、stream或allocation；
- 每次调用仍检查live XQ/WQ/Y layout，逐次分配或接收workspace，并进入C++ checked
  validator；没有恢复prepared launcher、per-stream workspace或Tensor cache；
- 同源码cache-off/cache-on相邻ABBA表明缓存自身把public eager从
  `1839.589044`降到`1487.834022 us`（`-19.121391%`），`96/96`项更快；graph仅变化
  `-0.025667%`且方向混合，证明收益只来自eager Host路径；
- 物理GPU 4、CPU 68上的最终`A1 -> B1 -> B2 -> A2`覆盖96项。相对最初internal-workspace
  版本，public eager为`1597.634663 -> 1480.381021 us`（`-7.339202%`），graph replay为
  `1245.734637 -> 1113.622186 us`（`-10.605184%`），两者均`96/96`项更快；
- 最终代码在物理GPU 4--7的focused回归为`239 passed, 24 skipped, 0 failed`；gfx950
  canonical 140-kid exhaustive为`140 passed, 0 failed`；
- 权威artifact目录为`/tmp/aiter-task1-scalarcache-resume.OYrIey`、
  `/tmp/aiter-task1-scalarcache-adjacent.ac8xj1`、
  `/tmp/aiter-task1-scalarcache-regression-resume.D88m6W`和
  `/tmp/aiter-task1-scalarcache-exhaustive-resume.6KfDGy`。

Task1当前没有待补性能轮次或正确性shard。Phase 2A/2B没有启动；若继续优化，必须建立新的
相邻版本目标和独立ABBA。工作树修改尚未提交，续接时先审阅diff，不要reset/checkout。

## workspace launch 性能优化第 2/3 项（实验已回退；2026-08-11 13:22 UTC）

当前权威状态：用户确认更深的 pointer/prepared ABI 不依赖这两项局部实验后，已回退第2/3项
实现。下面的性能数据和实验设计继续保留作决策依据，但其中的 prepared/prevalidated 路径不再
存在于当前源码。回退没有撤销 Torch-owned workspace 迁移、gfx950 mono-tile FP32修复、
gfx1250 fused工作或任务二的其他修改。

用户曾要求先做两项局部优化：首调用完整校验、重复 launch 走 prepared 路径；以及把
`has_workspace + workspace_dispatch` 两次表查询合为一次。两项都已实现并完成 gfx950
正确性、三架构生成/语法和性能 A/B，随后因端到端无可测收益而回退。实验没有改变公开
Python API，也没有恢复 C++ allocator、全局 Tensor cache、prewarm 或 workspace 所有权。

### 已归档的实验实现（当前源码已不存在）

- gfx950 每个 workspace launcher保留原 checked wrapper，并生成同 ABI 的
  `<kernel>_prevalidated`；公共内部实现为 `<kernel>_impl<Validate, D_C>`；
- 第一次合法调用始终进入 checked wrapper。随后每线程只缓存 POD 标量、workspace原地址和
  function pointer，不保存或分配 Torch Tensor；
- prepared命中要求 kid、split-K、M/N/K/batch、输入/输出基本 dtype/dim以及 workspace
  地址、容量、dtype、device、contiguous合同仍有效；XQ/WQ/Y和bias数据地址可以变化，launcher
  始终使用本次参数，bias仍逐次完整校验；
- short、错误 dtype、non-contiguous、misaligned或不同 workspace都会退出快路径并重新走
  checked wrapper；
- workspace generated row现在一次返回 `{checked, prevalidated}`。gfx950填两者，gfx942和
  gfx1250的 `prevalidated` 明确为 `nullptr`，因此后两架构行为不变；
- 实现集中在6个文件：`opus_gemm.cu`、两份generator和三个arch dispatch header；另在
  `test_opus_interfaces.py`、`test_opus_workspace.py`补回归。不是公开接口或workspace模型重构。

### 验证

- 最终全量 gfx950 JIT：`/tmp/aiter-gfx950-prepared2.TYzqMO`；142-entry sidecar SHA-256仍为
  `b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7`；
- focused suite（dispatch/workspace/graph/a16w16/interfaces）：
  `217 passed, 15 skipped, 0 failed`；skip仍是缺少gfx942/gfx1250硬件的条件项；
- 新GPU回归覆盖：连续两次合法workspace、随后同地址short/dtype/non-contiguous、misaligned
  回退拒绝，以及不同XQ/WQ/Y地址和BF16/FP32 Y均匹配golden；
- fresh multiarch codegen：`/tmp/aiter-prepared-multiarch.qyHiYr`，32-kid默认subset；gfx942、
  gfx950、gfx1250三个generated host TU和同时包含三架构router的`opus_gemm.cu`均通过目标
  `hipcc -fsyntax-only`；
- `git diff --check`和相关Python `py_compile`通过。

### 性能结论

先隔离已经预转换好的 `aiter_tensor_t`，只计pybind/C++ launch。代表kid 200从
`5.514 us`降到`5.145 us`，prepared C++路径改善`6.694%`，证明快路径确实命中并跳过了重复
校验/查询。

但是正常Torch raw入口还会在每次调用中经过`torch.ops`，并把XQ/WQ/Y/workspace分别转换成
pybind `aiter_tensor_t`。以修改前`mono-final` JIT为A、最终prepared JIT为B，独占GPU 4按
`A1 -> B1 -> B2 -> A2`对全部96项复测：

| 口径 | 修改前配对总和 | prepared配对总和 | 变化 | 逐项方向 |
|---|---:|---:|---:|---:|
| eager/raw | `1600.927 us` | `1601.621 us` | `+0.043%` | 55慢 / 41快 |
| graph replay | `1111.840 us` | `1111.310 us` | `-0.048%` | 45慢 / 51快 |

两项都属于测量噪声，结论是第2/3项对正常Torch raw端到端**无可测收益，也无可测回退**。
日志在`/tmp/aiter-gfx950-prepared-perf.ydkarB/`：

```text
3eac53b64047b1f7973a4914458b7208095d815496db30e0cea5e951ca04c233  perf_before_P0A1.log
20af62fea6baf5d18e8032f9909eb7b8692c5f9fa84b8d9ef4fe6c88e532822d  perf_compact_P2B1.log
5dde9081a82a92436a5980969a6e1e16010ba31daf7084923a94d5fefecc6412  perf_compact_P2B2.log
290af9ec09156c102b8d67e6bdf4b7fd12153dcf71a245a747c8ef54066577d3  perf_before_P0A2.log
```

最终prepared与最初内部workspace baseline另做同样ABBA，eager仍为`+10.311%`，graph为
`-12.856%`。因此当前eager慢的主因不是workspace dtype、device kernel或剩余C++ validator；
而是Torch/custom-op/pybind边界，尤其新增workspace Tensor的逐调用转换。若要继续消除约
1.5--2 us host差距，下一步必须进入更有侵入性的pointer/prepared ABI或复用已转换metadata；
这超出本轮第2/3项，不能仅继续删C++ `AITER_CHECK`解决。

### 回退范围与回退后验证

只撤销第2/3项本身：删除 gfx950 `<kernel>_prevalidated`、`<kernel>_impl<Validate, D_C>`和
thread-local prepared合同；generated workspace row恢复为`{kid, func}`；三架构 entry恢复为
单个`OpusA16W16WorkspaceKernel`；runtime恢复独立的`has_workspace(kid)`与
`workspace_dispatch(kid)`；删除只验证 prepared/prevalidated 路径的测试。gfx950 workspace
launcher再次在每次调用执行完整 checked validator。

回退后使用全新目录完成验证：

```text
gfx950 JIT: /tmp/aiter-gfx950-rollback23.4aK1Lp
multiarch:  /tmp/aiter-multiarch-rollback23.CQJaRY
```

- `prevalidated|PreparedWorkspace|OpusA16W16WorkspaceDispatch|<bool Validate|workspace_try_dispatch`
  在`csrc/opus_gemm`和`op_tests`中无命中；
- fresh gfx950生成、完整编译、链接和加载成功；接口/workspace首轮为
  `88 passed, 13 skipped`；
- dispatch/workspace/graph/a16w16/interfaces focused suite为
  `218 passed, 23 skipped, 0 failed`；gfx950 kid 200的BF16/FP32数值，以及short、错误dtype、
  non-contiguous、alignment拒绝均在该集合内通过；
- fresh三架构默认subset共32个kid；gfx942/gfx950/gfx1250三个generated host TU和同时包含
  三架构router的`opus_gemm.cu`均通过对应HIP语法编译；
- `python -m py_compile`和`git diff --check`通过。

因此后续更深的 pointer/prepared ABI可以从当前 checked-only 基线独立设计；它可以在新ABI
内部复用“预验证launcher”思想，但不依赖、也不应恢复本次端到端无收益的thread-local实现。

## 任务一最终续测更新（2026-08-11 11:43 UTC）

本节是当前任务一的最高优先级状态，取代下面 08:30 UTC 的“性能等待”和 06:31 UTC 的
`130 passed / 10 failed` 冻结状态。当前 dirty 工作树仍未 reset/checkout。

### 结论

- gfx950 原始/当前 workspace 性能 A/B 已在独占 MI355X 上按 `A1 -> B1 -> B2 -> A2`
  完整重跑，四轮均为 `96/96` 数值通过；
- 10 个 mono-tile FP32 失败已定位并修复，定向测试为 `10 passed`；
- 最终 fresh JIT 的 gfx950 140-kid sweep 为 `140 passed, 0 failed`，四个 shard各
  `35 passed`；
- focused gfx950 suite 为 `166 passed, 14 skipped, 0 failed`；CPU过滤回归为
  `151 passed, 18 deselected, 0 failed`；
- gfx950 任务一已闭环。gfx942/gfx1250实机数值、graph、并发和性能仍属于对应硬件补验，
  本轮结果没有扩大到这些架构。

### 有效性能 A/B

有效日志目录：

```text
/tmp/aiter-gfx950-perf-valid.HMflXg
```

覆盖全部48个 workspace kid（`200--223`、`1200--1223`）乘 BF16/FP32 Y，共96项；每项
20次 warmup、9轮乘100次 launch，同时测 raw eager binding与 CUDA/HIP graph replay。
baseline graph capture前先在 capture stream初始化其内部 workspace。四轮之间检查到8张卡均
为0%利用率、0%显存，且没有KFD进程；本轮没有混用此前受外部八卡任务污染的无效日志。

每个 `(kid, dtype)` 分别取两轮 baseline median的平均和两轮 current median的平均，再做
配对总和：

| 口径 | 全部96项 | BF16 48项 | FP32 48项 | 逐项方向 |
|---|---:|---:|---:|---:|
| eager/raw current 对 baseline | `+9.375%` | `+9.471%` | `+9.280%` | 88慢 / 8快 |
| graph replay current 对 baseline | `-15.479%` | `-15.457%` | `-15.500%` | 96快 / 0慢 |

eager/raw 与 graph replay测量边界不同，必须分别报告，不能合并成一个“总体提升/回退”。原始
日志 SHA-256：

```text
68009ecf5a67be8e95993d207fa7b1431375658d93d7a2cfa1bf2eaae3907888  perf_baseline_A1.log
97f91ee42c9fdb960ac950513f3f2f75fb8d961b8ddfc72f5d3bdf3ee1f37c1b  perf_baseline_A2.log
f83d8cda3e4c7597af0c490e4c0ed1bb94e7ea77053e0e21088337841a85452d  perf_current_B1.log
283fb22c7825c81ca92cfbae2be11cc2a378d7a757f9cde37f191e449504e14f  perf_current_B2.log
```

新增 benchmark 为 `op_tests/bench_opus_gfx950_workspace_ab.py`，195行，SHA-256
`0a96d5d917cf2a970ee9c9e8d2c70f632f59cfdbf81e8161742cc00d091ce87d`。

### 10个 mono FP32失败的根因和修复

失败集合仍为：

```text
1400, 1401, 1402, 1403, 1404
6400, 6401, 6402, 6403, 6404
```

根因不是 external workspace；这些 kid全部是 non-workspace。mono pipeline固定逻辑
`VEC_C=8`：BF16 的8元素为16字节，而 FP32 的8元素为32字节。
`csrc/include/opus/opus.hpp` 的 `gmem::_store` 只实现到16字节，没有32字节分支，因此旧
FP32 kernel实际没有发出Y写回。把Y预填 `12345.0` 后，旧 kid 1400/6400均为
`changed=0/49152`；此前约50%或99.6%的 mismatch只是 `torch.empty` 地址复用后读到的旧显存
形态，不是部分正确计算。

同时，上游 BF16-only epilogue把每个8元素 chunk的 lane-half交换硬编码为两对 `u32`；FP32
需要四对。最终修复同步完成：

- lane-half交换按 `sizeof(D_C)`推导，BF16为两对、FP32为四对；
- 保持逻辑 `VEC_C=8` layout；BF16仍发一笔 `store<8>`，FP32对每个逻辑 issue显式发
  `offset+0`和 `offset+4` 两笔 `store<4>`，每笔均16字节；
- 不把 vec=8 cached layout直接当成 vec=4 bulk layout。验证实验表明那会产生重叠/错误
  offsets；最终实现先按 vec=8取得逻辑 issue基址，再显式加4元素；
- 普通 mono与4G-safe镜像使用相同修复，并删除“kernel body byte-for-byte identical to
  upstream”的过时注释。

修改文件：

```text
csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh
csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_mono_tile_4g_safe_gfx950.cuh
csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh
op_tests/test_opus_a16w16_gemm.py
```

最后一个文件新增普通 mono 1400与4G-safe 6400的FP32回归：先把输出预填为
`12345.0`，再要求 public API原地复用该Tensor、全部 `49152/49152` 元素被覆盖并匹配
FP32 golden。两项定向运行结果为 `2 passed, 9 deselected`。

代表 kid 1400 的最终 FP32 code object中可见成对
`buffer_store_dwordx4 ... offset:0/16`；同一 tile每lane BF16为12笔 b128 store、FP32为24笔，
符合物理字节数翻倍。

### 最终验收和可恢复 artifact

最终隔离 JIT：

```text
/tmp/aiter-gfx950-mono-final.xAJYPc
```

sidecar为142项（140个 canonical a16w16加2个必需 a8w8），SHA-256仍为
`b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7`。fresh完整构建、链接、
加载成功。sentinel smoke覆盖普通1400和4G-safe 6400的BF16/FP32，四项均
`changed=49152/49152`；FP32最大绝对误差均为 `7.6293945e-06`。

最终四卡结果：

| shard | 结果 | log SHA-256 |
|---|---:|---|
| 0 | `35 passed` | `659e756a8b2a0faaccefadf06e8445d70c8ea8d2f78c514db6678d50e1006e19` |
| 1 | `35 passed` | `bf180a49dee30933ea2ec3e094e023d8c6bed37d3da50047cd7855007769fe79` |
| 2 | `35 passed` | `3bed0587228cab5e61bce6d1c3fcd855b8f4cc5dc3c14d63da617465b66887e1` |
| 3 | `35 passed` | `db35884c985bcca8238a966769e892b699b7eef9f34d7dd591d094a1bc31cd1c` |

日志与JUnit XML位于
`/tmp/aiter-gfx950-mono-final.xAJYPc/results/shard{0,1,2,3}.{log,xml}`。此外：

- 原10项：`10 passed, 130 deselected`；
- focused GPU：`166 passed, 14 skipped`；14项只因需要gfx942/gfx1250硬件；
- CPU过滤集：`151 passed, 18 deselected`；18项是明确排除的GPU raw/graph/two-stream项；
- `python -m py_compile`（benchmark、exhaustive与focused regression test）和
  `git diff --check`：通过。

## 任务一续测更新（2026-08-11 08:30 UTC）

本节晚于下面 06:31 UTC 的冻结检查点，是恢复任务一后的最新权威测试状态。当前 dirty
工作树没有被 reset/checkout；本轮只新增测试记录，没有覆盖任务一代码修改。

### 已完成：`ca68b4f...` 基线构建和 mono FP32 归因

- 08:17 UTC 检查时 physical GPU 0--7 均为空闲 MI355X/gfx950；本轮先在 physical GPU 0
  完成基线构建和数值复现；
- 从 `ca68b4f3501762c15c550cb920a5516e9710cf89` 用 `git archive` 创建全新隔离源码
  `/tmp/aiter-gfx950-baseline-src3.EXHfY5`，对应 JIT 为
  `/tmp/aiter-gfx950-baseline-jit3.DLB5t1`；没有复用旧的半成品 JIT或锁文件；
- 构建显式传入
  `CK_DIR=/root/workspace/0810/aiter/3rdparty/composable_kernel`、
  `GPU_ARCHS=gfx950`，并使用与首轮全量 sweep 相同的 142-entry sidecar，SHA-256仍为
  `b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7`；
- `module_deepgemm_opus` 全量构建成功，耗时约13.6秒；kid 1400 BF16烟测通过，最大绝对误差
  `0.1248779296875`，日志出现明确的 `BASELINE_BUILD_AND_SMOKE_OK`；
- 在基线和当前端点分别用相同 seed、shape、kid以及 BF16→FP32顺序运行
  `1400--1404`、`6400--6404`。两个端点均为 BF16 `10/10`通过、FP32 `0/10`通过；
- 本轮定向 harness在每个 dtype launch前分配对应 Y，FP32超差元素约
  `49.49%--50.53%`，最大绝对误差约 `80.38--102.26`。首轮 exhaustive test是先同时分配
  BF16/FP32两个 Y，其失败日志为约 `99.6%--99.8%`超差、最大绝对误差约 `42.93--77.67`。
  两种分配时序都稳定失败，说明既有 mono FP32缺陷的具体错误形态对地址/allocator布局敏感；
- 基线与当前的20条定向结果在去掉日志前缀后逐字相同；两端生成的20个 mono-tile device TU
  也逐字相同。由此可以闭环归因：这10项是最初 `ca68b4f...` 已有的 non-workspace mono
  FP32缺陷，不是 Torch-owned split-K workspace迁移引入的回归。

本轮原始记录位于：

```text
/tmp/aiter-gfx950-continuation-results.aVTGDs/baseline_build.log
/tmp/aiter-gfx950-continuation-results.aVTGDs/baseline_numeric.log
/tmp/aiter-gfx950-continuation-results.aVTGDs/current_numeric.log
```

对应 SHA-256依次为：

```text
917a1210b029dcf11016e3529bef3c9ffc6ea21a9af0bb7ca1d18607598a1e7b
a18c1d1cabfd8b02f233e882b67a261f8bb0e215fee94efe84c78cc24099d966
20dfcff981def137c5ab9e437df57e26981477135722758f71a97506e06da3a3
```

### 性能 A/B 已启动但数据作废，等待重新获得独占 GPU

计划的性能口径为全部48个 gfx950 workspace kid、BF16/FP32 Y共96项，基线使用预热后的
内部 workspace，当前使用复用的 caller Tensor；每项20次 warmup、9轮乘100次 launch取
median，并按 A--B--B--A顺序运行。

完成 A1并进入 B1/B2时，宿主突然出现外部 PID `2109525--2109532`；8张卡随后各被占用约
`55.8--56.7 GB`（约18%--19% VRAM），KFD还记录了 queue eviction。它们不在当前容器中，
本轮没有终止或干扰这些进程。由于不再满足“单卡独占”，已有 A1/B1/B2日志全部标为**无效
性能数据**，不能用于声称通过、回退或改进；A2和graph-replay隔离计时没有继续执行。

恢复时不需要重建基线或重跑已闭环的数值归因；等至少一张卡的外部进程和显存占用消失后，
从性能 A1重新开始并完成 A--B--B--A及graph-replay对照。当前仍开放的 gfx950项只剩有效的
原始/当前性能 A/B；gfx942/gfx1250硬件验收边界保持不变。08:32:59 UTC再次采样时，该外部
任务已升至每卡72%--75%显存和17%--66% GFX activity，确认不能在本轮继续 benchmark。

## 任务一冻结检查点（切换任务二前，2026-08-11 06:31 UTC）

用户已明确要求先保存任务一当前进度，再切换到任务二。本节是切换时点的最高优先级记录；
它表示**任务一冻结**，不表示剩余实机与性能验收已经全部完成。任务二必须从当前 dirty
工作树继续，不能 reset/checkout，也不能恢复已经删除的 C++ workspace 所有权路径、两个
`_workspace*.py` 文件或 `WorkspacePlan`。

### gfx950 已经闭环的部分

- 两张 MI355X/gfx950 上的 focused suite 已通过：
  `162 passed, 14 skipped, 2 warnings, 0 failed`；
- 已验证代表性 split-K 数值、BF16/FP32 Y、bias、raw workspace 正反合同、跨 device 拒绝、
  无旧 prewarm 的 graph capture/replay、双 stream 和调用级 Tensor 生命周期；
- 新增可复现的 opt-in 全量测试
  `op_tests/test_opus_gfx950_exhaustive.py`，直接枚举最终 canonical registry，不维护手写 kid
  副本；
- 隔离 JIT 已成功构建全部 140 个 canonical gfx950 a16w16 kid。最终集合为 48 个
  external-workspace kid和 92 个 non-workspace kid；生成表核对为 BF16 direct 92、FP32
  direct 92、workspace 48，二进制只包含 gfx950；
- 48 个 workspace kid（`200--223`、`1200--1223`）全部通过：BF16 Y、FP32 Y、caller
  workspace 复用、Torch 自动分配、同步后的数值和弱引用生命周期均通过。该结果覆盖了任务一
  在 gfx950 上新增的 Torch-owned split-K workspace 主路径，而不只覆盖代表性 kid 200。

### 140-kid 全量 sweep 的精确结果

全量测试按 physical GPU 4--7 分成四个 shard，每个 shard 35 个 canonical kid。首轮结果：

| shard | 结果 | 失败 kid |
|---|---:|---|
| 0 | `32 passed, 3 failed` | 1400、1404、6401 |
| 1 | `33 passed, 2 failed` | 1401、6402 |
| 2 | `33 passed, 2 failed` | 1402、6403 |
| 3 | `32 passed, 3 failed` | 1403、6400、6404 |
| 合计 | `130 passed, 10 failed` | 1400--1404、6400--6404 |

10 个失败全部集中在 non-workspace `a16w16_mono_tile` 及其 4G-safe 镜像。每个失败 case
先运行 BF16 Y，再运行 FP32 Y；BF16 断言已经通过，随后 FP32 Y 出现大范围数值不匹配。
因此当前不能写成“140 个 kid 全部通过”，也不能把这 10 项归因于 Torch workspace：它们都
是 non-workspace kid，并且不会调用 `torch.empty` 分配 external workspace。

原始日志仍位于：

```text
/tmp/aiter-gfx950-current.NtJydE/results/shard0.log
/tmp/aiter-gfx950-current.NtJydE/results/shard1.log
/tmp/aiter-gfx950-current.NtJydE/results/shard2.log
/tmp/aiter-gfx950-current.NtJydE/results/shard3.log
```

这些 `/tmp` 路径不是长期存储；四个 shard 的结果、失败集合和分类已经完整抄入本文件及
implementation log，后续即使临时目录被清理也不会丢失关键结论。

### 中断位置与尚未完成的归因

全量 sweep 已经完成，真正中断的是下一步“在最初基线 `ca68b4f...` 复现 10 个 mono-tile
FP32 失败”：

1. 第一次基线隔离构建因未传 `CK_DIR`，host TU 找不到 `ck_tile/core.hpp`；这是构建环境配置
   错误，不是产品代码或 GPU 数值失败；
2. 第二次已补上
   `CK_DIR=/root/workspace/0810/aiter/3rdparty/composable_kernel`，于 06:02:03 启动；
3. 旧会话在 06:02:16 等待构建返回时停止，build log只写到 06:02:22，尚无
   `BASELINE_BUILD_AND_SMOKE_OK`；
4. 06:04:20 内核记录旧 Docker veth被移除，同时旧 AMD GPU queues被 evict。没有 OOM、GPU
   reset、page fault或 RAS 记录，说明是容器整体被终止/回收，而不是 pytest正常失败；
5. 因基线复现未完成，目前不能判定这 10 个 mono FP32问题是原始 `ca68b4f...` 已有缺陷，
   还是当前端点的独立 non-workspace 回归。

### 切换时 GPU 与进程快照

2026-08-11 06:31 UTC 最后检查时，physical GPU 0--7 均为 100% GFX activity，每张约占
`285.6--287.0 GB` 显存；宿主可见 PID 为 `802980--802987`，名称为 `N/A`，在当前容器内
不可见。当前容器没有残留 pytest/JIT测试进程。不要终止这些未知进程；GPU 释放前不继续任务一
实机测试。

### 任务一冻结结论与恢复顺序

可以确认：gfx950 的 48/48 Torch-owned workspace kid全量通过，任务一的新 workspace 路径在
gfx950 上已得到强于 focused suite 的实机证据。尚不能确认：gfx950 全 140 kid零回归、
mono-tile FP32归因、基线/当前性能 A/B，以及 gfx942/gfx1250（尤其 gfx1250 fused）的对应
硬件验收。

以后恢复任务一时按以下顺序继续，不重跑已经闭环的 48 个 workspace kid：

1. 等至少一张 gfx950空闲后，用已修正 `CK_DIR` 的独立源码/JIT目录完成 `ca68b4f...` 构建；
2. 只复现 `1400--1404`、`6400--6404` 的 FP32 Y，并同时保留 BF16 对照；
3. 若基线也失败，记录为既有 mono FP32缺陷；若基线通过，则定位当前 non-workspace回归；
4. 在一张独占 gfx950上串行完成原始/当前同 shape、同 kid、同输入的性能 A/B；
5. 资源具备时补 gfx942/gfx1250数值、graph、并发和 gfx1250 fused验收。

任务二可以按用户当前决定开始，但不得把上述任务一开放项改写为“已通过”，也不得删除
`op_tests/test_opus_gfx950_exhaustive.py` 或覆盖当前任务一代码修改。

## 最新权威摘要（2026-08-11）

本节和后面的“2026-08-11 当前主线”是当前工作树的权威版本。后文 Step 1--Step 6 中保留的
`WorkspacePlan`、独立 workspace Python 模块以及“1378 个 fused kid尚未合入”等说法只用于
历史审计，不代表当前实现。

### 最直接的整体框架

```text
gemm_a16w16_opus(A, B, ...)
  -> kernel 选择：explicit -> tuned CSV -> Python heuristic -> framework fallback
  -> 完成 redirect / legality / split-K 解析，得到最终 actual_kid
  -> exact-kid registry 决定：是否需要 workspace、dtype、tile、layout、执行 family
       non-workspace
         -> 不分配 external workspace，直接 launch
       two-stage
         -> 分配 split-major workspace
         -> main kernel 写 partial
         -> 独立 reduce kernel 累加并写 Y
       fused
         -> 分配 tile-major workspace
         -> 前 SplitK-1 个 WG 写 partial
         -> 最后一个 WG 在同一 clustered kernel 内 reduce 并写 Y
  -> generated C++ 校验 device / dtype / contiguous / alignment / capacity
  -> launch
```

唯一核心原则是：**resolved `actual_kid` 是 workspace capability、dtype、shape 和物理执行合同的
共同真值**。不能按 architecture 统一写死 BF16/FP32，也不能按 kid 数值范围或 tag 字符串猜测。

### 当前 registry 快照

| 架构 / family | external workspace | BF16 | FP32 | non-workspace a16w16 |
|---|---:|---:|---:|---:|
| gfx950 FlatMM two-stage | 48 | 0 | 48 | 92 |
| gfx942 | 8 | 3 | 5 | 14 |
| gfx1250 two-stage | 496 | 496 | 0 | 0 |
| gfx1250 fused | 1378 | 780 | 598 | 0 |
| gfx1250 合计 | 1874 | 1276 | 598 | 0 |

gfx950 的其余 split-barrier、persistent、wave/cooperative 和 atomic-accumulate 路径没有
caller-owned external Torch workspace，因此没有需要“保持为 BF16 或 FP32”的 workspace。

### 当前验证状态

- CPU过滤回归：`151 passed, 18 deselected, 0 failed`；18项是明确排除、未执行的GPU
  raw、graph capture/replay和双-stream用例；
- gfx950 focused suite：`166 passed, 14 skipped, 0 failed`；
- gfx950 140-kid sweep：`140 passed, 0 failed`；原10个 mono-tile FP32失败已修复；
- gfx950 workspace性能 A/B 已按独占 `A1 -> B1 -> B2 -> A2`完成：eager/raw总和
  `+9.375%`，graph replay总和 `-15.479%`，两种口径分别报告；
- Python `py_compile`、`git diff --check`、全量 fused codegen和代表/边界 gfx1250 HIP syntax通过；
- 尚未运行 gfx1250 fused GPU 数值、graph、双 stream和性能测试；
- 未终止或干扰未知 GPU/KFD 占用进程。

## 当前 Git 状态

- 仓库：`/root/workspace/0810/aiter`
- 分支：`splitk_to_torch_2`
- 任务原始基线：`ca68b4f3501762c15c550cb920a5516e9710cf89`
- 当前 HEAD：`2352c46c784d6ba3a0c71ff89b4bdb4c2fefa59f`
- HEAD 标题：`[OPUS] Finalize workspace migration audit`
- 2026-08-11 结构精简基线：当前 HEAD `2352c46c`
- tracked 工作区：有一组**未提交**的结构精简修改；不要 reset、checkout 或按旧
  `WorkspacePlan` 架构重做
- 当前 tracked diff：17 个文件，`+1366/-731`；另有 1 个未跟踪的新 fused pipeline
- 删除：
  - `aiter/ops/opus/_workspace.py`
  - `aiter/ops/opus/_workspace_a16w16.py`
- 修改：
  - `aiter/ops/opus/_selector_a16w16.py`
  - `aiter/ops/opus/gemm_op_a16w16.py`
  - `csrc/opus_gemm/opus_gemm_common.py`
  - `csrc/opus_gemm/opus_gemm_tune.py`
  - `csrc/opus_gemm/codegen/common.py`
  - `csrc/opus_gemm/codegen/gen_instances_gfx942.py`
  - `csrc/opus_gemm/codegen/gen_instances_gfx950.py`
  - `csrc/opus_gemm/codegen/gen_instances_gfx1250.py`
  - `csrc/opus_gemm/gen_instances.py`
  - `csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh`
  - `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh`
  - `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh`
  - `csrc/opus_gemm/include/gfx1250/splitk_reduce_gfx1250.cuh`
  - `op_tests/test_opus_workspace.py`
  - `op_tests/test_opus_graph.py`
- 未跟踪代码：
  - `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_fuse_gfx1250.cuh`
  - `op_tests/test_opus_gfx950_exhaustive.py`
- 当前 docs 均为用户/本任务的未跟踪文件，不要删除、覆盖或误提交

## 2026-08-11 当前主线：去掉 WorkspacePlan，按 actual kid 直接初始化 workspace

用户在 Step 1 至 Step 6 完成后明确收紧了设计：不要为了 workspace 增加专门的 Python
模块和通用 `WorkspacePlan` 抽象；workspace shape、dtype 和分配应在现有 a16w16 入口中，
根据 selector 已经解析出的最终 `actual_kid` 直接完成。当前未提交修改已经按此方向落地。

本轮又在该轻量流程上完成了 #4246 fused family 的实际迁入，不再是“待合入”计划：1378 个
fused kid、两种物理 workspace dtype、独立 tile-major layout 和 compile-time SplitK 都已进入
当前代码路径。

### 当前唯一有效的运行流程

```text
gemm_a16w16_opus(A, B, ...)
  -> validate/reshape XQ, WQ, Y
  -> select_launch_config(...)
       explicit kid
         -> tuned CSV
         -> architecture heuristic
         -> framework fallback
       requested_kid
         -> gfx942 redirect / legality / split-K resolve
       actual_kid + allocation_split_k + launch_split_k
  -> framework fallback ? torch path : OPUS path
  -> _launch_a16w16_with_torch_workspace(..., resolved config)
  -> _init_a16w16_workspace(config, XQ, Y, optional workspace)
       get canonical instance by config.actual_kid
       determine workspace capability by actual kid
       read B_M/B_N/B_K from that instance
       two-stage: compute split-major padded shape and runtime allocation split-K
       fused: compute tile-major shape from exact-kid compile-time SplitK
       read splitk_workspace_dtype from that exact kid
       reuse caller Tensor or torch.empty(...)
  -> raw binding(..., workspace, config.actual_kid, config.launch_split_k)
  -> generated 5-arg non-workspace or 6-arg workspace launcher
  -> C++ physical-contract validation
  -> direct workspace pointer main + standalone reduce（two-stage）
     or single clustered main/reduce kernel（fused）
```

### heuristic、requested kid 与 actual kid 的边界

- heuristic 仍然是 selector 的第三优先级，只在没有 explicit kid、tuned CSV 也没有有效
  结果时运行；它不在 workspace init 内重跑。
- `requested_kid` 是 explicit/CSV/heuristic 最初提出的 kid。
- `actual_kid` 是 redirect、shape/dtype/bias 合法性和 split-K 解析后真正 launch 的 kid。
- `_init_a16w16_workspace()` 只接受已经完整解析的 `LaunchConfig`，只读取
  `config.actual_kid`；它不查询 CSV、不运行 heuristic，也不重新选择 kernel。
- 关键例子：gfx942 非 exact-N 的 `requested_kid=10210` 会解析为
  `actual_kid=10200`。workspace 因此使用 10200 的 tile 和 fp32 dtype，而不是 10210 的
  bf16 dtype。

### Python workspace 层已经收缩

- 删除 `WorkspacePlan`、`checked_numel()`、`allocate_workspace()` 和 Python
  `validate_workspace()` 抽象；
- 删除 `_workspace.py` 与 `_workspace_a16w16.py`；
- `gemm_op_a16w16.py` 内只保留一个私有 `_init_a16w16_workspace()`；
- non-workspace actual kid 返回 `None`，且拒绝 caller 多传 workspace；
- workspace actual kid 使用 instance 的 `B_M/B_N/B_K` 计算 padding 和 K-tile 上限；
- gfx942/gfx950 shape 为
  `[allocation_split_k, batch, padded_M, padded_N]`；
- gfx1250 two-stage shape 为 `[allocation_split_k, padded_M, padded_N]`；
- gfx1250 fused shape 为
  `[num_tiles_m, num_tiles_n, fuse_split_k - 1, B_M, B_N]`；
- gfx1250 两类路径都继续要求 batch=1；fused 的 `fuse_split_k` 来自 exact kid，runtime/CSV
  `splitK` 不改变执行或容量；
- Python 仍在分配前检查 split-K、shape extent/字节乘法溢出和 gfx1250 batch；
- caller 提供 workspace 时 Python 原样传给 raw binding，不重复实现物理合同验证；
- caller 未提供且 actual kid 需要 workspace 时，唯一分配点为该 init 内的
  `torch.empty(shape, dtype=..., device=XQ.device)`；
- 没有全局 Tensor cache，每次自动调用得到独立 Tensor。

### workspace dtype 现在由 exact kid metadata 决定

- `OpusGemmInstance.splitk_workspace_dtype` 默认值从隐式 `"fp32_t"` 改为 `None`；
- 所有 `SPLITK_KIDS` 都必须显式声明 `"bf16_t"` 或 `"fp32_t"`，模块加载时有全量
  invariant；
- non-workspace kid 可以保持 `None`；
- Python init、gfx942/gfx950/gfx1250 codegen 和 generated launcher 的 expected dtype
  都消费同一项 exact-kid metadata；
- `codegen/common.py::splitk_workspace_type()` 统一映射 C++ storage type、pointer type 和
  Aiter dtype token；
- gfx942 继续支持由 kid 区分 bf16/fp32 workspace；
- gfx950 的 A/B 是 bf16，Y 支持 bf16/fp32；但当前登记的 48 个 two-stage split-K kid 的
  partial workspace 均为 fp32，main/reduce 也按 fp32 writer/reader 实现；
- 当前未提交修改已把 gfx1250 登记的 496 个 two-stage split-K kid 恢复为 PR #4246 的
  bf16 workspace 合同：plain 28 个、clusterlaunch 468 个都显式声明 `"bf16_t"`；
- gfx1250 的 two-stage traits、main store、reduce、validator 和 codegen 现在都从 exact-kid
  metadata 取得 `D_WS`，底层路径可生成 bf16 或 fp32 workspace，不再有架构级
  FP32-only guard；
- #4246 的 1378 个 fused kid 已实际加入 registry：`21000--22377`，其中 780 个为 BF16
  workspace、598 个为 FP32 workspace；
- gfx1250 当前因此共有 1874 个 external-workspace kid：496 个 two-stage BF16，加上 fused
  的 780 个 BF16 / 598 个 FP32；按物理 dtype 汇总为 BF16 1276、FP32 598；
- fused 复用统一 `splitk_workspace_dtype`，没有恢复旧 `fuse_ws_dtype` 第二真值；
- 权威来源是 kid metadata，但 metadata 必须与该 kid 的真实 writer/reader 合同一致，不能
  只改字符串。

### BF16 能力复核：I/O dtype 不等于 partial-workspace dtype

用户指出 gfx950/gfx1250 存在 bf16 能力后，重新核对了当前树、原始 gfx950 实现和
PR #4246 历史。此前“gfx950/gfx1250 都只支持 fp32”的说法过度概括，正确边界是：

| 路径 | A/B | Y | split-K partial workspace |
|---|---|---|---|
| gfx950 当前 two-stage FlatMM | bf16 | bf16/fp32 | fp32；当前无已登记 bf16-workspace kid |
| gfx1250 当前未提交修改中的 496 个 two-stage kid | bf16 | bf16/fp32 | bf16；已对齐 PR #4246 |
| gfx1250 exact-kid two-stage code path | bf16 | bf16/fp32 | 可按 metadata 生成 bf16/fp32 |
| gfx1250 当前 #4246 fused family（1378 个） | bf16 | bf16/fp32 | 780 个 bf16 / 598 个 fp32 |

精确证据与影响：

- gfx950 从原始提交 `29810587` 起就由 main 写 fp32 partial，
  `splitk_reduce_gfx950.cuh` 固定从 float workspace 读取。它“有 bf16”指 bf16 A/B 以及
  bf16 Y，不等价于已有 bf16 partial-workspace kernel；直接把 kid 200--223/1200--1223 的
  metadata 改成 bf16 会让 Tensor dtype 与 writer/reader 不一致。
- gfx1250 的 PR #4246 feature 提交 `b32785d0` 把 two-stage main 的
  `OPUS_WS_BF16` 默认设为 1，reduce 使用匹配的 `D_WS=__bf16`。plain 28 个 kid
  （20000--20027）和 clusterlaunch 468 个 kid（20100--20567）因此在该分支的物理合同中
  都是 bf16 workspace；当前修改已把这 496 个 kid 的 metadata、main store、reduce、
  validator 和测试同步到该合同。
- 同一 feature 分支的 1378 个 fused kid（21000--22377）现在也已迁入：780 个使用 bf16、
  598 个使用 fp32 partial storage。当前实现不仅补了 `SPLITK_KIDS`，还同步加入独立
  tile-major capacity 公式、compile-time SplitK、generated launcher/validator、tuner 与
  subset codegen，避免把 fused 错套到 two-stage 的 `[S, padded_M, padded_N]` 合同。
- PR 分支的 two-stage 代码虽然实际读写 bf16，旧 `splitk_workspace_dtype` 仍保留默认 fp32，
  本身存在 metadata/物理合同不一致。当前修改没有照抄该旧 metadata，而是同步修正 main
  store、reduce `D_WS`、exact-kid metadata、codegen validator、容量字节数和测试。

因此，当前轻量流程“由 resolved actual kid 决定 dtype”仍然成立，而且 gfx1250 已不再按
architecture 硬编码 FP32。496 个 two-stage kid 是 BF16；1378 个 fused kid按 exact kid
登记为 780 个 BF16、598 个 FP32，并已使用独立 tile-major shape/capacity 分支。

#### gfx1250 two-stage 与 fused 的准确含义

- **two-stage**：第一阶段 main GEMM 的每个 split 都把 partial 写到
  `[split_k, padded_M, padded_N]` workspace；第二阶段启动独立 reduce kernel，按 split 轴
  重新以 FP32 累加、加 bias 并写 Y。当前树中的 496 个 two-stage kid属于这一类。
- **fused**：`SplitK` 个 workgroup在同一个 clustered kernel 内协作；前
  `SplitK-1` 个 split把 partial 发布到外部 workspace，最后一个 split经 cluster barrier
  读取这些 partial、在 kernel 内完成 reduce 并直接写 Y，因此没有第二次 reduce-kernel
  launch，但仍然需要外部 workspace。
- fused 的物理索引是 `tile -> (SplitK-1) partial -> B_M x B_N`，不是 two-stage 的
  split-major padded matrix。二者可以共享 exact-kid dtype metadata，不能共享 shape 公式。

#### gfx950 其他 split-K 路径是否有 workspace

- gfx950 只有 FlatMM two-stage 的 48 个 kid（200--223、1200--1223）使用 caller-owned
  external Torch workspace，物理类型固定 FP32；
- 其他 split-barrier、persistent、wave/cooperative 和 atomic-accumulate 路径都直接写 Y 或
  使用 kernel 内同步/累加机制，不发布 external partial workspace；
- 因而这些“普通 split-K”路径没有“保持 BF16 还是 FP32 Torch workspace”的问题，不能给它们
  分配本任务的 external workspace；
- gfx950 汇总仍为 external workspace 48（全部 FP32）、non-workspace a16w16 92。

#### fused bias 的当前安全边界

#4246 round-1 fused launcher 只接受 contiguous BF16 `[N]`，而公共 API/tuned CSV 的 bias 合同
还允许 FP32 和 `[batch, N]`，CSV key 又只有布尔值。为避免用 BF16 `[N]` 调出的 tuned row被
FP32 bias 重放，当前 selector 和 tuner 对任何 `bias=True` 都排除 fused kid；无 bias 时 fused
正常参与调优。two-stage gfx1250 仍支持公共的 BF16/FP32、`[N]`/`[batch, N]` bias 合同。

fused 候选的 compile-time SplitK 必须按每个 `N-cluster x workspace dtype` 分别从 exact-kid
registry 选择。BF16 当前覆盖 SplitK 2--15，FP32 覆盖 2--8；不能先用两者共用的 SplitK
top-N 再查 dtype，否则大 K、小 grid形状会先选出 13--15 并把仍然合法的 FP32 2--8
候选意外筛空。当前 tuner 对每个 family独立按 occupancy取最多 3 个 SplitK，并继续由 registry
自然施加 LDS 与 `SplitK * n_cluster <= 16` 约束。

### Python 与 C++ 的验证责任

Python init 只负责选择完成后的结构计算和分配前检查。generated C++ launcher 继续是
caller-provided Tensor 物理合同的最终防线，保留：

- workspace 必须存在或必须为 `None` 的路由检查；
- device 与输入一致；
- exact kid 要求的 dtype；
- contiguous；
- 16-byte alignment；
- clamp 后 required capacity；
- checked extent 以及 kernel stride 上限。

raw ABI、5/6 参数 dispatch、direct pointer kernel ABI、graph 所有权模型和 deprecated
`opus_gemm_workspace_init()` no-op 均未回退。

### 当前验证结果与未完成边界

已完成：

- 相关 Python 文件 `py_compile`：通过；
- `git diff --check`：通过；
- deleted workspace module / `WorkspacePlan` / planner 的 Python 引用扫描：无残留；
- 只选择 CPU 用例运行 dispatch、workspace 和 graph 测试：
  `149 passed, 18 deselected, 2 warnings in 3.69s`；18 项均为明确排除的 GPU raw、graph
  replay 或双 stream 用例，没有执行且不是失败；本次选择集为 `0 failed`；
- 在两张空闲 MI355X/gfx950（物理 GPU 4、5映射为进程内 device 0、1）运行未过滤的 focused
  suite，包括 `test_opus_a16w16_gemm.py`：`162 passed, 14 skipped, 2 warnings in 4.39s`，
  `0 failed`；14项均因需要 gfx942/gfx1250而跳过；gfx950真实执行覆盖 split-K BF16/FP32 Y、
  bias数值、raw workspace正反合同、跨device拒绝、无预热 graph capture/replay和双stream；
  测试退出后物理 GPU 4、5均恢复0%利用率、0%显存占用且无测试残留KFD进程；
- CPU 覆盖包括 heuristic/CSV/explicit 优先级、gfx942 redirect、actual-kid dtype/tile、
  two-stage/fused 两套 shape、compile-time SplitK、每个 workspace kid 显式 dtype、独立
  Tensor、无 Python cache、bias 安全门、BF16/FP32 fused 候选独立 SplitK top-N 和 raw
  boundary 参数传递；
- canonical registry 审计为：gfx950 external 48（FP32）；gfx942 external 8（BF16 3 / FP32
  5）；gfx1250 external 1874（BF16 1276 / FP32 598）；
- gfx1250 two-stage：plain 28 + clusterlaunch 468，496 个全部 BF16；
- gfx1250 fused：1378 个、kid `21000--22377`、BF16 780 / FP32 598，全部进入显式
  external-workspace registry 和 workspace dispatch；
- 全量 fused codegen 生成 1378 个 impl header、2756 个 device TU、1378 个 lookup row，且
  standalone reduce TU 数量为 0；
- fresh 组合 subset `/tmp/opus-fused-combined.yOZtal` 覆盖 two-stage、clusterlaunch、BF16/
  FP32 fused、BF16/FP32 Y、128x128 tile 以及 `SplitK * n_cluster == 16` 边界；其 host TU、
  `opus_gemm.cu`、pybind TU 和 14 个 device/reduce TU 全部通过 gfx1250
  `hipcc -fsyntax-only`；
- fused pipeline 已从 #4246 旧 `opus::tdm/make_tdm` API迁移到当前树的
  `opus::tdm_window` API；静态扫描确认没有恢复旧 `fuse_ws_dtype`。

尚未完成：

- gfx1250 fused 实机数值、graph replay、并发 stream 和性能测试；
- 全量 2756 个 fused device TU 的逐文件 HIP syntax（已完成全量生成和代表/边界编译，不把
  代表编译写成全量编译）；
- gfx942/gfx1250 外部硬件验收，以及 gfx942 迁移前后真实性能对比；
- 最终提交。

本轮没有终止或干扰任何未知 GPU 进程。宿主机物理 GPU 0--3仍由既有进程占用；本轮仅使用
明确空闲的 GPU 4、5完成 gfx950验收并在退出后释放。gfx942/gfx1250仍缺对应硬件，因此不宣称
gfx1250 fused数值、graph、并发或性能已经实测。

> 以下 Step 1 至 Step 6 和收尾审计保留为 `2352c46c` 之前的已提交历史。凡是其中提到
> `_workspace.py`、`_workspace_a16w16.py`、`WorkspacePlan`、Python allocator/validator 或
> planner 是“当前架构”的表述，均已被本节的 2026-08-11 未提交精简流程取代；不要据此
> 恢复已删除模块。其他 selector、raw ABI、generated dispatch、direct pointer 和历史
> 验证数据仍然有效。

## 已完成：Step 1

目标“让 Python 在分配前知道最终 kid 和 split-K”已经实现、测试并单独提交。

完成内容：

- selector 顺序固定为
  `explicit -> tuned CSV -> Python heuristic -> framework fallback`；
- Python 在 launch 前解析 requested/actual kid；
- tuned row 无效时原子丢弃 `(kid, splitK)`；
- 三架构 a16w16 heuristic 已移植到 Python；
- gfx942 launcher symbol 通过 canonical instance `.name` 反查 kid；
- gfx942 auto split-K、effective split-K、even-loop/down-clamp 已移植；
- gfx942 非 exact-N：`10210 -> 10200`、`10213 -> 10203`、`10216` 拒绝；
- CSV miss 生产路径不再调用 generic C++ bf16 selector；
- 未修改 workspace ABI；
- 未删除 C++ allocator、registry、lookup 或三个 heuristic golden。

Step 1 实际涉及 10 个代码文件，完整的“每个文件改了哪里、增加/替换/删除了什么”在：

```text
docs/task1_detail.md
```

不要重做 Step 1，也不要把后续修改 amend 到 `83ce59db`；后续 Step 应独立提交。

## Step 1 验证状态

- `pytest -q op_tests/test_opus_dispatch.py`：`56 passed`；
- Python `compileall`：通过；
- `git diff --check`：通过；
- gfx942 codegen import / exact-N 共享断言：通过；
- gfx950 CSV-miss 数值 smoke：kid `208`、`1206`、`1300` 通过；
- smoke 明确确认 legacy generic C++ binding 未被生产路径调用；
- gfx942/gfx1250 完成 CPU branch parity，没有对应硬件数值测试。

## 历史提交：Step 2（WorkspacePlan 方案已被 2026-08-11 精简替代）

目标“增加动态 WorkspacePlan 和集中 Torch 分配准备路径”已经实现并测试；raw C++ ABI
仍保持旧签名，因此生产路径尚未启用 Torch workspace。

完成内容：

- 新增 family-neutral `_workspace.py`：
  - `WorkspacePlan(shape, dtype, required_numel, alignment)`；
  - checked extent/bytes；
  - `allocate_workspace()`；
  - dtype/device/contiguous/alignment/capacity 共享验证；
- 新增 `_workspace_a16w16.py`：
  - 只接受 canonical actual `OpusGemmInstance`；
  - capability 来自 `(arch, family, kid)` 窄查询；
  - gfx950/gfx942：`[S, batch, padded_M, padded_N]`；
  - gfx1250：`[S, padded_M, padded_N]` 且 batch 必须为 1；
  - gfx942 workspace dtype 从 actual kid 的 instance 读取；
  - 在分配前拒绝 split-K 超过 actual instance 的 K-tile 上限；
- `gemm_op_a16w16.py` 新增 `_prepare_a16w16_workspace()` 和注入 fake/raw callable 的
  `_launch_a16w16_with_torch_workspace()`；
- 显式 workspace 只走共享 validator，不分配；
- 没有 Tensor cache；
- 当前 legacy raw binding、生产调用点、C++ allocator/registry 和
  `opus_gemm_workspace_init()` 全部保留；
- implementation log 已追加 4 个实际代码文件的逐文件记录。

Step 2 提交：

```text
b2c99c8494ac1193c97ee1be23452a6e43af48d6
[OPUS] Add typed a16w16 workspace plans
```

Step 2 已作为独立提交落在 Step 1 的 `83ce59db` 之后；后续不要 amend 这两个提交。

## Step 2 验证状态

- `pytest -q op_tests/test_opus_workspace.py`：`24 passed`；
- `pytest -q op_tests/test_opus_dispatch.py op_tests/test_opus_workspace.py`：
  `80 passed`；
- Python `compileall`：通过；
- `git diff --check`：通过；
- 测试确认 exact capacity 成功、少 1 element 失败；
- 测试确认错 dtype/device/contiguous/alignment 均失败；
- 测试确认显式 workspace 不调用 allocator；
- 测试确认 Step 2 生产路径未提前启用 Torch workspace；
- registry 遍历 552 个 canonical a16w16 workspace instance 均成功生成自洽 plan。

## CK 状态

- `3rdparty/composable_kernel` 已初始化；
- 当前 commit：`f33252cebe5a52362ec1ee12c124dde7800dda3a`；
- 与仓库记录的 pinned commit 一致；
- `3rdparty/composable_kernel/include/ck_tile/core.hpp` 已存在；
- `git submodule status --recursive` 显示 clean pinned 状态。

## 已完成并提交：Step 3

目标“把 codegen 拆成 non-workspace / workspace 两套 dispatch”已经实现、验证并单独
提交。

完成内容：

- 三架构 generated reduce ABI 改为 direct `const void* ws_ptr`；
- manifest 将 non-workspace 保持为 5 参数，workspace launcher 改为 6 参数；
- tune table 按 arch 拆成 5 参数 non-workspace 和 6 参数 workspace 两类；
- `(M,N,K)` runtime table 只保存 integer kid；
- 三架构 split-K launcher 接收 caller-owned typed Tensor，checked 计算 clamp 后
  `required_numel` 并复用共享 validator；
- 删除 generated launcher 内的 capture、registry、grow、raw allocation、handle mirror
  和 sync；
- gfx942 删除 10210/10213 host redirect，全部 bf16ws launcher 统一保留 exact-N raw
  guard；
- gfx1250 C++ launcher 增加 `batch == 1` 最后防线；
- arch header 以 generated workspace table membership 判断 kid 类型，没有数值区间副本；
- subset CSV/sidecar/`HEURISTIC_DEFAULT_KIDS` 保留。

Step 3 提交：

```text
0a9bd8101c1d8ac84d6734deaf2ec385a45c0e54
[OPUS] Split a16w16 workspace dispatch tables
```

实际涉及 8 个代码文件，`+856/-689`。逐文件新增、替换、删除内容已完整追加到
implementation log；不要 amend Step 1、Step 2 或 Step 3 提交。

## Step 3 验证状态

- 四个 Python generator `py_compile`：通过；
- `pytest -q op_tests/test_opus_dispatch.py op_tests/test_opus_workspace.py`：
  `80 passed`；
- fresh 三架构 representative codegen：通过；
- canonical dispatch 分类：gfx950 workspace 48、gfx942 workspace 8、gfx1250
  workspace 496，共 552，与 `SPLITK_KIDS` 完全一致；
- generated manifest 5/6 参数分类、runtime integer-kid table、allocator token scan：通过；
- 三架构真实 subset CLI：CSV/sidecar/default-kid invariant 通过；
- 三个 arch header 分别及组合 HIP `-fsyntax-only`：通过；
- gfx942 pure-kid heuristic 与 Python port 对拍 8712 cases：通过；
- `git diff --check`：通过；
- 没有运行 GPU 数值/graph/performance：Step 5 raw binding 和 Python 生产入口尚未接通。

## 已完成并提交：Step 4

目标“把三架构 kernel 改成 direct pointer”已经实现、验证并单独提交。

完成内容：

- 三架构 traits 删除 `opus_splitk_ws_handle` 和 guard，split-K kargs 统一改为
  `void* ptr_ws`；
- workspace 物理类型改由 `D_WS/DataWS` 表达；
- gfx950/gfx1250 main pipeline 直接 cast caller-owned pointer；
- gfx942 新增 `opus_gfx942_uniform_ws_ptr()`，删除旧 handle helper，同时保留 64 位地址
  high/low split 和两次 `readfirstlane`；
- gfx942 common workspace epilogue 和 quad common store path 分别以 `D_WS/D_STORE`
  表达 store 类型；
- 三架构 reduce definition 统一为 `const void* ws_ptr`；
- gfx942 baseline、bf16ws、exact-N/HAS_OOB 和全部 forwarding layer 均已覆盖；
- main/reduce 的 workspace layout、offset、clamp 和数值路径不变；
- 没有恢复 handle shim、device mirror 或 kernel-side allocator。

Step 4 提交：

```text
34b70a8430273e5458862724fc781102d6fe5afe
[OPUS] Use direct split-K workspace pointers
```

实际涉及 16 个代码文件，`+95/-113`。除计划中的 traits/pipeline/reduce 外，还修改：

- `gfx942/a16w16/opus_gemm_helpers_a16w16.cuh`：workspace epilogue 的类型名收口为
  `D_WS`；
- `opus_gemm_common.cuh`：删除失效的 shared-handle 注释。

逐文件细节已追加到 implementation log；不要 amend Step 1 至 Step 4 的任何提交。

## Step 4 验证状态

- `pytest -q op_tests/test_opus_dispatch.py op_tests/test_opus_workspace.py`：
  `80 passed, 2 warnings`；
- `git diff --check`：通过；
- fresh representative codegen：通过，覆盖 gfx950 200；gfx942 non-workspace
  10000/10001/10003/10006、workspace 10200/10204/10210/10213/10216；gfx1250
  20000/20100；
- 三架构 generated host TU、13 个 main device TU、3 个 reduce TU 的对应 arch
  `hipcc -fsyntax-only`：通过；
- 三 target 的 common/traits/reduce 组合 header 检查：通过；
- `csrc/opus_gemm/include` 内 handle、旧 helper、`kargs.ws_handle` 全部清零；
- gfx942 main/reduce 汇编均确认 direct pointer 仍生成成对
  `v_readfirstlane_b32`；
- 全仓旧 handle 命中只剩 `opus_gemm.cu` registry 和 README，属于 Step 5/6 范围；
- 没有运行 GPU 数值、graph 或性能测试，因为 Step 5 raw entry 尚未接通。

## Step 5 已完成

Step 5 独立提交：

```text
4e8ce216eed77a94fbb504cc26585ce6afef8b5f
[OPUS] Route split-K through Torch workspaces
```

实际涉及 7 个代码文件，`+134/-409`：

```text
csrc/opus_gemm/include/opus_gemm.h
csrc/include/rocm_ops.hpp
csrc/pybind/opus_gemm_pybind.cu
csrc/opus_gemm/opus_gemm.cu
aiter/ops/opus/gemm_op_a16w16.py
aiter/ops/opus/__init__.py
aiter/tuned_gemm.py
```

完成状态：

- raw tune ABI 增加无默认值 optional workspace；
- 当前架构 generated workspace table 命中时要求 Tensor 并调用六参数 launcher；未命中时
  要求 `workspace=None` 并调用原五参数 launcher；
- 删除所有 workspace kid 数值区间分流副本；
- Step 2 的 `resolve -> plan -> validate/torch.empty -> raw launch` 已进入两个 Python 生产
  入口；显式 Tensor 复用共享 validator，缺失时按调用分配，不缓存 Tensor；
- generic `opus_gemm()` bf16 分支已停用，a8w8 分支未改；
- `SplitkWsRegistry`、三个 `opus_splitk_ws_*` 函数、host/device handle、raw allocation、
  mirror/sync/capture query 与 `<mutex>/<unordered_map>` 已删除；
- workspace-init C++ 声明/实现/pybind 已删除，Python 名字保留 deprecated no-op；
- `aiter/tuned_gemm.py` 的 prewarm set、capture-stream 猜测、同步和调用点已删除；
- tuner/deepgemm 三个调用方保持调用公共 wrapper，没有复制 workspace 逻辑。

详细逐文件说明和验证结果已追加到 implementation log。

## Step 5 验证状态

- Python `py_compile` / `compileall` 与 `git diff --check`：通过；
- 现有 CPU 测试：`78 passed, 2 deselected, 2 warnings`；两项 deselect 是明确断言
  “Step 2 尚未启用 workspace”或 monkeypatch 旧高层调用边界的测试，Step 6 必须改写；
- fresh 多架构 codegen：`/tmp/opus-step5-codegen.YgHY5P`，同时启用
  gfx942/gfx950/gfx1250，workspace table size 为 7/6/6；
- fresh headers 下 `opus_gemm.cu` 和 pybind TU 的 HIP `-fsyntax-only`：通过；
- gfx950 `module_deepgemm_opus` 完整 JIT 重建/链接/加载：通过；
- pybind raw 签名确认 7 参数，公共 Python 签名保留旧顺序并新增 keyword-only
  `workspace=None`；
- `.so` 不再导出 C++ workspace init；missing workspace、non-workspace 多传 Tensor、错误
  dtype 的 C++ 前置拒绝已确认；
- 旧 registry/handle/prewarm 符号在代码路径中清零；两份 README 的旧说明留给 Step 6。

一次 raw capacity 探针因忽略 launcher clamp，把实际足够的 workspace 当成短容量，并用
CPU Tensor 指针误入 GPU kernel；产生的 151 MiB GPU core 已删除，结果未计入验证。Step 6
容量负例必须用真实 device Tensor，并以 clamp 后 effective split-K 构造 exact/short-one。

## 已完成并提交：Step 6

提交：

```text
b72e4cc414d843f592a4115a8dbd0da949dedada
[OPUS] Cover Torch workspace lifecycle
```

提交包含计划指定的 6 个文件：4 个测试文件和 2 个 README；新增
`op_tests/test_opus_graph.py`。详细逐文件记录、numstat、ISA 数据和测试矩阵已追加到
implementation log。

完成内容：

- 改写两个 pre-Step-5 断言，生产路径现在验证 raw 7 参数、actual kid 和自动 workspace；
- 三架构 selector parity 与 gfx942 split resolver 独立对照；
- gfx942 redirect 覆盖 complete exact-N set、多个非 exact-N 和独立 N=384；
- Python validator 与真实 raw C++ exact/short-one、missing、dtype、device、contiguous、
  alignment 防线；
- 三架构条件数值、workspace dtype、batch 和 bias 测试；
- graph capture/replay、双 stream 独立 workspace、weakref 生命周期和无全局 Tensor cache；
- split-K 上限在 allocator 前失败；
- gfx942 direct-pointer uniform helper 静态测试与迁移前后 ISA/register 对照；
- a8w8/a8w4 API 与 generated-output scope isolation；
- 两份 README 删除旧 registry/handle/raw allocator/prewarm 说明，改为当前 Torch 所有权
  模型。

## Step 6 验证状态

当前机器：8 张 AMD Instinct MI355X，`gfx950:sramecc+:xnack-`，256 CU；Torch
`2.11.0+rocm7.14.0`，HIP `7.14.60850`。

```text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py

159 passed, 14 skipped, 2 warnings in 4.30s
```

实际 gfx950 通过：bf16/fp32 数值、bias 规则、raw exact 8192 / short-one 8191、全部
workspace 负例（含 cuda:0/cuda:1 device mismatch）、graph 三次 replay、双 stream 和
生命周期。14 个 skip 均为缺少 gfx942/gfx1250 硬件的条件 case。

fresh `/tmp/opus-step6-codegen.nFnIx1` 的三架构 40 个 generated host/device/reduce TU
全部通过目标架构 `hipcc -fsyntax-only`。

gfx942 交叉编译 ISA：代表 main 的 VGPR/SGPR 维持 169/96，readfirstlane 维持 5，handle
dereference load 从 1 降为 0；134 个 reduce kernel 的 readfirstlane 总数维持 276，按
四-VGPR block 取整 124 个不变、10 个改善、0 个回退。当前无 gfx942，性能和 gfx942/
gfx1250 实机数值不能宣称通过；条件测试已保留供对应硬件执行。

a8w8 generated launcher/device/lookup 与任务基线逐字一致，a8w4/a8w8 family 文件从基线
无 tracked diff。机械扫描和 `git diff --check` 均通过，CK 保持 pinned commit。

## 历史：最终完成定义收尾审计（HEAD `2352c46c`）

最终收尾提交：

```text
2352c46c784d6ba3a0c71ff89b4bdb4c2fefa59f
[OPUS] Finalize workspace migration audit
```

收尾时发现 `csrc/opus_gemm/README.md` 抄录了 forbidden-symbol 扫描命令，导致扫描整棵
`csrc/opus_gemm` 时命中文档自身。已删除该自命中并补充 family-adapter 边界。现在用户指定
的三条 `rg` 均为 exit 1 且空输出，`git diff --check` 为 exit 0 且空输出。

枚举 canonical a16w16 registry 并逐一调用 planner 的结果为：gfx950 workspace 48 /
non-workspace 92，gfx942 workspace 8 / non-workspace 14，gfx1250 workspace 496；分类、
canonical instance 和 plan/None 一致性错误均为 0。共享 `_workspace.py` 中没有 arch、kid、
split-K 或 launcher 选择逻辑；这些规则只存在于 a16w16 selector/planner/dispatch adapter。

收尾 focused suite 再次通过：`159 passed, 14 skipped, 2 warnings in 4.46s`。gfx942 ISA
artifact 仍确认 main readfirstlane `5 -> 5`、VGPR/SGPR `169/96 -> 169/96`，reduce
readfirstlane `276 -> 276`。本机仍无 gfx942；真实性能没有执行，不能把 ISA 结果表述成
gfx942 性能实测通过。

## 当前结论

Step 1 至 Step 6 和 `2352c46c` 收尾审计是已提交、可恢复的历史基线；2026-08-11 又按用户
确认的新意图完成了一轮未提交的结构精简。当前代码主线不再包含 `WorkspacePlan` 和两个
workspace Python 模块，而是由 resolved `actual_kid` 直接驱动单一
`_init_a16w16_workspace()`，并让 workspace dtype 对三架构统一来自 exact-kid metadata。

当前 gfx1250 workspace 合同已经同时覆盖：496 个 two-stage BF16 kid，以及 1378 个 fused
kid（780 BF16 / 598 FP32）。fused 的 tile-major layout、compile-time SplitK、显式 registry、
Python 分配、generated validator/dispatch、tuner candidate 和 subset codegen均已接通；它不
启动第二个 reduce kernel。

代码、CPU 回归、全量 fused 生成和代表/边界 HIP syntax 已闭环。仍不能宣称 gfx1250 fused
实机完成，因为本轮没有运行 GPU 数值、graph 或性能测试；交叉编译不能替代这些验收。

## 当前未跟踪 docs

以下文件属于用户或当前任务，全部保留：

```text
docs/opus_gemm_refactor_architecture_brief.md
docs/task1_checkpoint.html
docs/task1_checkpoint.md
docs/opus_gemm_splitk_workspace_torch_current_flow_changes.html
docs/opus_gemm_splitk_workspace_torch_current_flow_changes.md
docs/task1_detail.html
docs/task1_detail.md
docs/task1_plan.html
docs/task1_paln.md
docs/task1_arh.html
docs/task1_arh.md
docs/opus_gemm_two_tasks_final_plan.md
```

## 恢复工作时的最短流程

```bash
cd /root/workspace/0810/aiter
git status --short --branch
git log -2 --oneline --decorate
git submodule status --recursive
git diff --stat
git diff --check
```

确认 HEAD 仍为 `2352c46c`，并确认上面列出的 17 个 tracked 修改和新 fused pipeline仍存在。
工作区 dirty 是当前预期状态，不要把它当成异常清理。随后：

1. 先读本文件的“2026-08-11 当前主线”和 implementation log 最后一节；
2. 不要恢复 `_workspace.py`、`_workspace_a16w16.py` 或 `WorkspacePlan`；
3. 不要把 fused workspace 改回 two-stage shape，也不要恢复旧 `fuse_ws_dtype`；
4. GPU 验证前运行 `rocm-smi --showuse --showmemuse --showpids`，只有资源可用且进程归属
   明确时才运行完整 focused suite；
5. 实机验收或提交当前工作树时，不要 amend Step 1 至 Step 6 的历史提交。

若要继续外部硬件补验，再读 implementation log 的 Step 6 “实际运行环境”、
“ISA/register 对照”和未执行边界，不要重做已提交步骤。

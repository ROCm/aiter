# OPUS GEMM 下一步性能优化计划

记录时间：`2026-08-11 UTC`；实施与最终验收：`2026-08-12--13 UTC`

状态：**Phase 1和caller-resolved exact-kid接口已经实施、验收并采用。** 正式a16w16 public/high-level路径已经从pybind raw
切换到C ABI/ctypes raw；原pybind raw继续保留为兼容和A/B端点。2026-08-12又完成并采用了
device只读信息缓存和重复layout检查消除两项public Host热路径优化。2026-08-13继续完成并采用
有界纯标量launch-plan缓存；统一exact-kid入口前的public eager相对最初版本改善`7.339%`，
统一入口后的当前结果为`3.238%`。
当前最高优先级结论见第11节；第9.4/9.5节保留的是缓存实施前的历史快照，第10节保留的是
统一exact-kid入口实施前的最终Task1快照。Phase 2A/2B没有启动。

## 0. 最终结果摘要

本轮没有修改device kernel、workspace布局、kid集合或generated traits。C ABI仍调用原有
`opus_gemm_a16w16_launch` checked launcher，不复制dispatch或validator。最终gfx950结果为：

| 同边界比较 | pybind A | ctypes B | 变化 | 平均每项变化 |
|---|---:|---:|---:|---:|
| raw eager，96项配对总和 | `1644.093507 us` | `1372.621492 us` | `-16.511957%` | `-2.827833 us` |
| raw graph replay，96项配对总和 | `1109.366086 us` | `1107.945324 us` | `-0.128070%` | `-0.014800 us` |
| public eager，96项配对总和 | `2487.414976 us` | `2205.882312 us` | `-11.318283%` | `-2.932632 us` |
| public graph replay，96项配对总和 | `1109.186787 us` | `1111.668675 us` | `+0.223757%` | `+0.025853 us` |

raw eager有`94/96`项更快；public eager有`95/96`项更快。graph测试只计相同captured kernel的
replay，不会在每次replay重新执行Python或C ABI，因此上述`-0.128%/+0.224%`属于轮次噪声，
没有可归因的graph回退。Phase 1超过“至少追回约`1 us`”的采用阈值，因此不进入Phase 2。

采用前的双设备补充测试发现：只设置thread-local stream而不切换HIP current device时，
“当前device 0、Tensor在device 1”的隔离进程会发生memory fault。最终实现已改为成对切换并
恢复HIP device和thread-local stream，并在共同checked launcher中先验证XQ/WQ/Y、optional
bias和workspace的device合同。修复后的device 1数值、current-device恢复和mixed-device
launch前拒绝均通过。

## 1. 当前基线

已经回退的实验内容：

- gfx950 `_prevalidated` launcher；
- `<kernel>_impl<Validate, D_C>`双wrapper；
- thread-local prepared workspace合同；
- 合并`has_workspace(kid)`和`workspace_dispatch(kid)`的单次查询。

当前行为：

- generated workspace row为`{kid, func}`；
- 三架构workspace entry都只有一个function pointer；
- runtime先调用`has_workspace(kid)`，命中后再调用`workspace_dispatch(kid)`；
- gfx950 workspace launcher每次执行完整checked validator。

以下内容继续保留，不属于待回退范围：Torch-owned workspace、gfx950 mono-tile FP32修复、
graph/stream/lifetime合同、gfx1250 fused family以及任务二的接口/dispatch重构。

回退后fresh gfx950 focused suite结果为：

```text
218 passed, 23 skipped, 0 failed
```

## 2. 已知性能结论

此前第2/3项实验的分层结果：

| 测量边界 | 修改前 | prepared实验 | 变化 |
|---|---:|---:|---:|
| 隔离pybind/C++，kid 200 | `5.514345 us` | `5.145200 us` | `-6.694%` |
| 正常Torch raw eager，96项配对总和 | `1600.927 us` | `1601.621 us` | `+0.043%` |
| graph replay，96项配对总和 | `1111.840 us` | `1111.310 us` | `-0.048%` |

结论：跳过C++检查和合并一次查表虽然能在隔离C++边界节省约`0.369 us`，但正常Torch
端到端没有可测收益。下一轮不应恢复第2/3项，而应优化Torch/custom-op/pybind边界。

### 2026-08-11只读转换微基准

代表输入为kid 200常用的四个device Tensor：XQ、WQ、Y和workspace。每种方式warmup后运行
`10000`次、重复`7`轮，报告每次转换四个Tensor的median：

| 转换方式 | median | 相对当前pybind |
|---|---:|---:|
| `torch_to_aiter_pybind` | `6.107 us` | 基线 |
| `torch_to_aiter`（ctypes struct） | `3.347 us` | 约省`2.760 us` |
| 只读取四个`data_ptr()` | `0.283 us` | 仅作下界参考 |

这个微基准只测转换，不等价于完整launch；但ctypes的潜在节省已经大于当前待追回的约
`1.5--2 us` eager host差距，因此应先做低侵入的C ABI/ctypes端到端原型。

## 3. 推荐实施顺序

### Phase 1：并行C ABI/ctypes实验入口（优先）

当前路径：

```text
Torch Tensor
  -> torch.ops
  -> Python读取每个Tensor的metadata
  -> 创建pybind aiter_tensor_t
  -> C++ checked launcher
```

目标路径：

```text
Torch Tensor
  -> torch.ops
  -> ctypes aiter_tensor_t
  -> C ABI（显式传当前HIP stream）
  -> 原有C++ checked launcher
```

设计要求：

1. 保持公开Python API和workspace所有权不变。
2. 先增加一个并行实验symbol，不立即删除现有pybind入口，便于同源码A/B。
3. C ABI接收XQ/WQ/Y、optional bias、optional workspace对应的`aiter_tensor_t*`，以及
   `kid`、`split_k`和当前`hipStream_t`。
4. C ABI入口设置当前thread-local HIP stream，然后调用现有
   `opus_gemm_a16w16_launch`；不复制kernel dispatch或workspace validator。
5. 使用现有ctypes异常桥，把C++异常转换为状态码和thread-local错误字符串，不能让异常穿过
   C ABI。
6. Torch custom-op schema继续保留所有Tensor参数。不能把公开/custom-op边界改成只有整数
   pointer，否则Torch无法正确追踪Tensor生命周期、读写、alias和graph依赖。
7. 不增加全局Tensor cache，不保存Tensor对象或历史data pointer。
8. pybind和ctypes共用同一个模块时，必须先验证JIT首次构建、`.so`加载和现有a8接口不会因
   `torch_exclude`/Python-module构建模式发生冲突；必要时使用独立的薄C ABI TU或明确的混合
   build配置。

Phase 1的目标是验证“去掉pybind对象构造”能否在正常Torch raw端到端稳定追回差距，而不是
提前引入prepared状态。

### Phase 2A：native Torch C++ Tensor边界（ctypes不足时首选）

如果Phase 1仍不能满足目标，增加一个薄的native Torch C++ operator：

```text
torch.ops
  -> C++ at::Tensor参数
  -> C++栈上构造轻量aiter_tensor_t/raw launch view
  -> 现有checked launcher
```

该TU可以单独包含Torch/ATen header，kernel、generated host TU和公共launcher继续保持
torch-free。优点是Tensor metadata转换全部在C++完成，同时Torch dispatcher天然持有Tensor
引用和alias信息。需要单独验证编译时间、首次JIT大小、torch.compile fake/meta注册以及当前
HIP stream获取方式。

### Phase 2B：prepared descriptor/pointer ABI（最后才做）

只有Phase 1/2A profile仍证明metadata或validator是主要剩余开销时，才引入prepared
descriptor。

prepare阶段只保存不可变合同：

```text
kid / checked或专用function pointer
batch / M / N / K / effective split-K
workspace dtype / required numel / alignment
必要的tile和stride标量
```

每次launch仍接收当前XQ/WQ/Y/bias/workspace Tensor，并读取当前data pointer和stream。
descriptor不得保存Tensor、storage ownership或历史XQ/WQ/Y/workspace data pointer。compact
guard失败时必须回到checked路径或明确报错，不能静默继续。

新的ABI内部可以重新使用“预验证launcher”的概念，但不依赖、也不应直接恢复已回退的
thread-local prepared cache。

## 4. 明确不采用的方法

- 不直接恢复性能实验第2/3项；
- 不通过继续删除`AITER_CHECK`解决Torch边界开销；
- 不把workspace改回C++内部allocator；
- 不建立全局Torch Tensor或pybind对象cache；
- 不以更换workspace物理dtype作为host优化；
- 不在Torch custom-op schema中只传整数pointer；
- 不以牺牲CUDA/HIP graph、当前stream或多stream并发语义换取微小收益。

## 5. Phase 1历史修改边界（`core.py`部分现已撤销）

- `aiter/jit/core.py`
  - Phase 1当时曾给ctypes loader增加`force_torch_exclude`并给
    `compile_ops()`增加`ctypes_force_torch_exclude`；
  - 2026-08-13按最终文件边界要求已完整撤销这些修改；当前文件与Task1原始基线
    `ca68b4f...`零diff，不承担任何OPUS mixed-module特例。
- `aiter/ops/opus/gemm_op_a16w16.py`
  - 增加私有`_opus_gemm_a16w16_launch_ctypes_raw`；最终版本把最小CDLL装载和
    `aiter_tensor_t`转换局限在本文件；
  - 第一次调用借现有pybind wrapper完成正常lazy JIT build/rebuild，后续复用同一`.so`的C ABI；
  - public explicit、deprecated adapter和shape-driven high-level最终都使用该raw；
  - 原`_opus_gemm_a16w16_launch_raw` pybind入口继续存在，只用于兼容、测试和A/B。
- `csrc/opus_gemm/include/opus_gemm.h`
  - 声明`opus_gemm_a16w16_launch_cabi(...)`；optional Tensor用空指针，kid/split_k为
    `int64_t`，stream显式传入。
- `csrc/opus_gemm/opus_gemm.cu`
  - 接入现有TLS异常桥，C++异常不会越过C ABI；
  - C ABI检查整数范围后转调原canonical checked launcher；
  - `OpusCabiDeviceStreamGuard`切换/恢复HIP device和thread-local stream；
  - canonical launcher补齐输入、bias和workspace同设备检查。
- `op_tests/test_opus_ctypes.py`
  - 覆盖ABI形状、fake/`torch.compile`、BF16/FP32 parity、五种workspace错误、非默认
    stream、graph、双stream、跨current-device成功与mixed-device安全拒绝。
- `op_tests/test_opus_interfaces.py`、`test_opus_dispatch.py`、`test_opus_workspace.py`、
  `test_opus_graph.py`
  - production mock/hook改为跟随最终ctypes后端，既有合同继续回归。
- `op_tests/bench_opus_gfx950_workspace_ab.py`
  - 增加ctypes、public-pybind和最终public端点，可分别做raw与public同边界ABBA。
- `op_tests/bench_opus_task1_task2_interfaces.py`
  - 当前端点跟随ctypes；冻结Task1模块仍可注入旧pybind raw做历史对照。

没有修改`aiter/jit/optCompilerConfig.json`，也没有修改任何gfx950/gfx942/gfx1250 device
kernel、workspace物理布局、registry、codegen kid集合或generated traits。

## 6. 最终验证

### 6.1 fresh构建与ABI

最终full-142-kid gfx950目录：

```text
/tmp/aiter-opus-ctypes-final.MINoRH
```

compiled-kids sidecar SHA-256：

```text
b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7
```

fresh JIT完成生成、编译、链接和public数值启动。最终`.so`导出：

```text
aiter_ctypes_abi_version
aiter_get_last_error
aiter_clear_last_error
opus_gemm_a16w16_launch_cabi
```

### 6.2 正确性、graph、stream和lifetime

- ctypes定向组：`15 passed`；
- focused组（dispatch/workspace/graph/a16w16/interfaces/ctypes）：
  `254 passed, 22 skipped`；
- gfx950 canonical全量public sweep：`140 passed`，覆盖92个non-workspace和48个workspace
  kid，各自BF16/FP32输出，以及workspace复用/自动分配合同；
- 22个skip仍是gfx942/gfx1250硬件条件项，不是通过结论；
- gfx942/gfx950/gfx1250 fresh默认32-kid subset生成在
  `/tmp/aiter-opus-ctypes-multiarch.kRKLyS`；三个`all_instances_host_<arch>.cu`分别通过目标
  `hipcc -fsyntax-only`，包含三架构router和最终device/stream guard的`opus_gemm.cu`也通过；
- `py_compile`与定向`git diff --check`通过。

gfx942/gfx1250仍只有fresh codegen和host syntax结论；实机数值、graph、并发和性能继续按
`docs/gfx942_gfx1250_validation_runbook.md`执行，不能写成已通过。

### 6.3 最终性能方法

最终日志目录：

```text
/tmp/aiter-opus-ctypes-final-perf.nFPg1D
```

设备为gfx950 `AMD Instinct MI355X`。raw和public各自执行
`A1 -> B1 -> B2 -> A2`，覆盖48个workspace kid乘BF16/FP32共96项；每项
`20 warmup + 9 rounds x 100 launches`。每轮前GPU占用记录为`0--1%`，每个case先做数值
断言。A/B值均为同一case两轮median的平均后再求和。

raw边界：

| 项目 | pybind A | ctypes B | 变化 |
|---|---:|---:|---:|
| eager全部 | `1644.093507 us` | `1372.621492 us` | `-16.511957%` |
| eager BF16 | `821.327029 us` | `686.094224 us` | `-16.465159%` |
| eager FP32 | `822.766478 us` | `686.527268 us` | `-16.558673%` |
| graph全部 | `1109.366086 us` | `1107.945324 us` | `-0.128070%` |

public边界使用完全相同的selector/workspace helper，只把末端raw在pybind和ctypes之间切换：

| 项目 | public-pybind A | final public ctypes B | 变化 |
|---|---:|---:|---:|
| eager全部 | `2487.414976 us` | `2205.882312 us` | `-11.318283%` |
| eager BF16 | `1242.759873 us` | `1106.718597 us` | `-10.946706%` |
| eager FP32 | `1244.655104 us` | `1099.163715 us` | `-11.689294%` |
| graph全部 | `1109.186787 us` | `1111.668675 us` | `+0.223757%` |

最终日志SHA-256：

```text
d08e6439b2316523602e511ad89af8f7383329d5075f26e4f8a6f89fd72d5649  perf_raw_A1.log
0f6bc2bfdde0bc780c95687dce96b375b37d487ae51513062eb13463ba1fba19  perf_raw_A2.log
b957bcd61def8dca8a1dee1a853c887680a6bfacb3f6f07b65f699bcbabd47e8  perf_raw_B1.log
c19ddb43e1be11c298facaca566825472ecab771c5127b49f9346e684d9e2af1  perf_raw_B2.log
600002f661efe5e83d9d662aa00ed1c8ebba7925c95de1e073d2c9f67adf74b7  perf_public_A1.log
b66a5837948c92a72b3b5646edd349a340dfc40a99b4d6be3a5619ed04660f79  perf_public_A2.log
fe9fc20da48a3f8d8f75bb46825c888797257779599c30c7d5dc46befb0141b9  perf_public_B1.log
051c930f7aa43f111db20dc642d8d18243fc95e594a32fc097b18978e117b0b7  perf_public_B2.log
```

## 7. 最终采用状态

1. Phase 1的eager收益稳定且超过阈值，正式public/high-level路径采用ctypes。
2. pybind raw不删除，保留兼容、故障隔离和可重复A/B能力。
3. Phase 2A native Torch C++ Tensor边界不启动；当前没有继续扩大实现面的收益理由。
4. Phase 2B prepared descriptor/pointer ABI不启动；此前prepared/prevalidated实验仍保持回退。
5. 不缓存Tensor、data pointer、stream、workspace或launcher，不删除generated安全检查。
6. 若未来继续优化，应先profile public路径剩余的Python selector/workspace规划成本，建立新的
   同边界ABBA；不得把本轮ctypes收益当作恢复prepared状态的依据。

## 8. 一句话结论

Phase 1已经安全落地：最终public eager在gfx950全96项同边界比较中改善`11.318%`、平均每项
追回`2.933 us`，graph无可归因回退，因此保留checked validator并停止在Phase 1。

## 9. 2026-08-12 Task1 public Host热路径续优化

### 9.1 修改边界

本轮只修改Python Host路径和对应接口测试：

- `_device_arch_and_cu()`按解析后的显式device缓存架构和CU数；`cuda`无index时先解析为当前
  device，因此多卡切换不会复用错误设备的信息；
- `_explicit_a16w16_launch()`完成layout检查后向下层传递`_layout_checked=True`，避免
  `_launch_a16w16_with_torch_workspace()`重复执行同一检查；
- 测试覆盖同一device重复读取、无index device随current device切换，以及每个explicit调用
  只执行一次layout检查。

没有修改device kernel、C ABI、workspace shape/dtype/所有权、kid/split-K选择、registry、
generated launcher或graph内容；也没有缓存Tensor、data pointer、workspace或stream。

### 9.2 相对提交`4af16d5f`的public增量ABBA

A端为未含上述两项优化的`4af16d5f`冻结源码，B端为当前源码；两端共用已经验收的
`/tmp/aiter-opus-ctypes-final.MINoRH`，固定使用物理GPU 4。每轮覆盖48个workspace kid乘
BF16/FP32共96项，参数仍为`20 warmup + 9 rounds x 100 launches`，顺序为
`A1 -> B1 -> B2 -> A2`。GPU 4开跑前利用率和显存占用均为`0%`。

| 项目 | `4af16d5f` A | Host优化 B | 变化 |
|---|---:|---:|---:|
| public eager全部 | `2316.989467 us` | `2059.729041 us` | `-11.103219%` |
| public eager BF16 | `1157.079209 us` | `1040.103015 us` | `-10.109610%` |
| public eager FP32 | `1159.910258 us` | `1019.626026 us` | `-12.094404%` |
| public graph全部 | `1110.808749 us` | `1113.318296 us` | `+0.225921%` |

public eager有`85/96`项更快，逐项配对变化的median为`-2.655395 us`。graph replay不执行
Python Host路径，`+0.226%`与前述ctypes实验的graph波动量级相同，不能归因于本轮修改。

日志目录和SHA-256：

```text
/tmp/aiter-task1-hostopt-perf
d08312ebb0cf315095545a34d104b2a7f2400ea6765e11384efa932ee04db4da  perf_public_A1.log
ddfcdd6150398e2c678f0c63c9c5183c9955215c442733f62f351c7fd3de5ff6  perf_public_B1.log
4ff162610095b2ec3218b304feda51714de78b098ec4ef9ffc1f5fa7b9098b67  perf_public_B2.log
3a88da731d5dd8064b752edb7c2a86ffad668762b04810e31ceefdc74cc3b496  perf_public_A2.log
```

### 9.3 相对原始内部workspace的最终三端同场对照

为避免继续混用“原始基线”和“相邻优化阶段”，另按`A1 -> B1 -> C1 -> C2 -> B2 -> A2`
直接比较三个raw端点：

- A：修改前C++内部workspace；
- B：Task1 caller-owned Torch workspace加pybind raw；
- C：Task1 caller-owned Torch workspace加最终ctypes raw。

三端使用各自冻结源码和JIT、相同物理GPU 4、相同96项及相同计时参数。表中每个值都是同一
case两轮median取平均后再求和，变化统一相对A端：

| raw端点 | eager配对总和 | 相对A | graph配对总和 | 相对A |
|---|---:|---:|---:|---:|
| A：原始内部workspace | `1461.907939 us` | 基线 | `1249.560189 us` | 基线 |
| B：Torch workspace + pybind | `1665.108534 us` | `+13.899685%` | `1122.270175 us` | `-10.186785%` |
| C：Torch workspace + ctypes | `1367.221318 us` | `-6.476921%` | `1117.767402 us` | `-10.547134%` |

C端相对原始A端的eager有`95/96`项更快，graph为`96/96`项更快。该同场结果证明最终ctypes
raw既消除了pybind端点的eager回退，也保留了direct Torch workspace device路径的graph收益。
第9.2节再证明本轮两项Python优化进一步降低public Host开销而没有改变graph。

日志目录和SHA-256：

```text
/tmp/aiter-task1-final3-perf
127518e7afe1562c9e5444900cef7631afbc15a77d349f7bbd9fe2c9db13f3ad  perf_A1_baseline.log
ac403e77f1a5042c40c04f8673825675e06ece976a46efa7398f3f7a6bb0c71b  perf_B1_pybind.log
c7289ffb6b5890fee3ac88ce0083995cb55c434cdccfa134308a5de426d61517  perf_C1_ctypes.log
ca03292703054033f7f4e01e6f37e9156a30148537d03a3bd129d72091fb244e  perf_C2_ctypes.log
6b986bbb9963b24079d52115acd13501844d4ab9d9719c556e5c3ddc2620113b  perf_B2_pybind.log
1dbfa96d220666329dd1e08188b6c4bd4752b0f7694cd766b112d8c5b50993ba  perf_A2_baseline.log
```

### 9.4 最初版本与当前版本的直接对照

最终只保留两个端点重新测量：A为最初C++内部workspace版本，B为当前最新版本。测试固定物理
GPU 4和同NUMA节点CPU 68；开跑前CPU平均利用率约`4%`、GPU 0--4计算利用率均为`0%`。
raw使用`A1 -> B1 -> B2 -> A2`逐项配对；public执行两组相同顺序，并对每个case的四轮结果
取中间两值平均，以排除单轮调度异常。每个端点仍覆盖96项。

| 测量项 | 最初C++ workspace | 当前最新版本 | 变化 |
|---|---:|---:|---:|
| raw eager配对总和 | `1417.524772 us` | `1305.275458 us` | `-7.918684%` |
| public eager配对总和 | `1563.338715 us` | `1930.050585 us` | `+23.456968%` |
| public graph replay配对总和 | `1249.037862 us` | `1114.485576 us` | `-10.772475%` |

换算为每项平均：raw eager为`14.765883 -> 13.596619 us`，public eager为
`16.284778 -> 20.104694 us`，graph replay为`13.010811 -> 11.609225 us`。当前版本raw eager和
graph均为`96/96`项更快，public eager为`0/96`项更快。因此最终准确结论是：raw eager和graph
已相对最初版本提升，但public eager仍有明确回退。

最终两端日志目录：

```text
/tmp/aiter-task1-current-vs-internal-final
```

### 9.5 回归与采用结论

- 接口定向组：`66 passed, 8 skipped`；
- gfx950 focused组使用隔离JIT：`251 passed, 26 skipped`；
- gfx950 140-kid canonical全量数值回归：`140 passed`；
- `git diff --check`通过。

两项Host优化相对`4af16d5f`确有收益，因此保留；但它们尚未消除当前public入口相对最初版本
的全部Host开销。不得再把raw eager的提升表述成public eager也已提升。若目标要求public eager
同样不回退，需要继续单独优化public selector/workspace规划路径；本轮不恢复per-stream
workspace、prepared pointer cache或双pybind/ctypes生产路径，也不擅自启动Phase 2A/2B。

## 10. 2026-08-13 Task1 public标量计划缓存闭环

本节是当前Task1性能状态，取代第9.4/9.5节的“public eager仍有回退”结论。第9节原数据继续
保留，用于说明优化过程和相邻阶段差异。

### 10.1 实现边界

本轮继续只修改Python Host路径和对应测试，没有修改C++/HIP、C ABI、device kernel、workspace
布局或graph内容：

- 把workspace合同的纯标量计算拆为`_plan_a16w16_workspace()`，返回不可变的
  `(shape, dtype) | None`；原`_init_a16w16_workspace()`仍负责逐调用使用live Tensor device
  分配workspace，兼容调用路径保持不变；
- 新增`@lru_cache(maxsize=256)`的`_cached_explicit_a16w16_plan()`。key只含selector函数、
  arch、M/N/K/batch、CU数、是否有bias、输入/输出dtype、kid和split-K；value只含resolved
  kid、launch split-K以及workspace shape/dtype计划；
- 每次public调用仍重新检查本次XQ/WQ/Y的shape和stride，并把本次bias、workspace和Tensor
  传给raw launcher；C++ checked validator仍逐次验证dtype、device、容量、contiguous、alignment
  等物理合同；
- layout校验改为等价的固定顺序直写，避免临时tuple/循环；缓存命中调用使用位置参数构造key，
  避免关键字key和已是整数的shape重复转换；
- 缓存不保存Tensor、data pointer、workspace、stream、device allocation或launcher状态。
  生命周期测试明确验证两轮调用的XQ/WQ/Y/workspace删除后均可回收，缓存最多保留256个纯标量
  计划。

CPU fake-raw微基准中，缓存前完整public Python包装约`5.58 us/次`；最终缓存命中路径约
`1.7--1.8 us/次`。该结果只用于定位Host开销，正式采用结论以下面的GPU端到端ABBA为准。

### 10.2 同源码cache-off/cache-on相邻ABBA

为单独归因标量缓存，而不是只比较Task1首尾版本，另使用完全相同的当前Python源码、JIT、
Tensor和C++ launcher做相邻ABBA。A端在独立进程启动时把
`_cached_explicit_a16w16_plan`临时替换为lru wrapper的`__wrapped__`，因此每次重新执行
selector和workspace纯标量规划；B端保留正常cache-on路径。除此之外两端代码完全相同，
工作树没有为测试增加开关或修改生产接口。

测试仍固定物理GPU 4和CPU 68，顺序为`A1 -> B1 -> B2 -> A2`，每轮仍是相同96项和
`20 warmup + 9 rounds x 100 launches`：

```text
cache-off A1  eager 1841.093757 us, graph 1113.901254 us
cache-on  B1  eager 1488.020091 us, graph 1113.312107 us
cache-on  B2  eager 1487.647953 us, graph 1114.090724 us
cache-off A2  eager 1838.084332 us, graph 1114.073439 us
```

cache-off两轮eager漂移`-0.163%`，cache-on两轮漂移`-0.025%`；测试期间sibling CPU 196
平均和最大占用均为`0%`，最大load average为`1.33`。逐case配对结果为：

| 测量项 | cache-off A | cache-on B | 变化 | 逐项方向 |
|---|---:|---:|---:|---:|
| public eager全部96项 | `1839.589044 us` | `1487.834022 us` | `-19.121391%` | `96快 / 0慢` |
| public eager BF16 48项 | `919.605818 us` | `743.420114 us` | `-19.158829%` | `48快 / 0慢` |
| public eager FP32 48项 | `919.983227 us` | `744.413908 us` | `-19.083970%` | `48快 / 0慢` |
| public graph replay全部96项 | `1113.987346 us` | `1113.701416 us` | `-0.025667%` | `57快 / 39慢` |

缓存自身平均每项减少`3.664115 us`，逐项变化median为
`-3.730632 us / -19.449894%`。graph只变化`-0.026%`且方向混合，证明开关没有改变captured
device工作；收益来自每次eager调用不再重复执行selector和workspace纯标量规划。

相邻ABBA日志和SHA-256：

```text
/tmp/aiter-task1-scalarcache-adjacent.ac8xj1
89983088ec6e40b83effc4055d7be37cbef715e58b6e0466d9a6fbfb2efcfe4a  perf_cacheoff_A1.log
c05e7a92a9a75c3b7a34a45ddc48a95ce776925839b8c487556c4a6853a32711  perf_cacheon_B1.log
4f7de20364a01b3b5411731cdfc1ddc66a25540103fdd27260418a694a619027  perf_cacheon_B2.log
e1aea64dcd11624e5b398c8043b563e184b4c3033299d22f01a0ce5c73dbb1d7  perf_cacheoff_A2.log
536dbb964aef370fb4d0f9268a33d579975da2054e07afdc78440971feb9f972  cpu_monitor.log
```

### 10.3 相对最初internal-workspace版本的最终public ABBA

A端为最初C++内部workspace冻结源码和JIT；B端为当前Torch-owned workspace + ctypes raw +
最终Host优化。测试固定物理GPU 4和同NUMA节点CPU 68，顺序为
`A1 -> B1 -> B2 -> A2`。每轮覆盖48个workspace kid乘BF16/FP32共96项，每项仍为
`20 warmup + 9 rounds x 100 launches`。开跑前GPU 4--7均为0%且无KFD进程，CPU 68/196
连续采样均为0%；测试期间sibling CPU 196平均占用`0.03%`、最大`3.8%`，最大load average为
`1.50`。

四轮96项总和：

```text
A1  eager 1603.396186 us, graph 1242.186871 us
B1  eager 1477.286097 us, graph 1114.085675 us
B2  eager 1483.475946 us, graph 1113.158697 us
A2  eager 1591.873140 us, graph 1249.282404 us
```

A1到A2的eager总和漂移`-0.719%`，B1到B2漂移`+0.419%`，没有此前宿主高负载造成的失真。
每个case分别取两轮median的平均后求和：

| 测量项 | 最初internal workspace A | 当前最终版本 B | 变化 | 逐项方向 |
|---|---:|---:|---:|---:|
| public eager全部96项 | `1597.634663 us` | `1480.381021 us` | `-7.339202%` | `96快 / 0慢` |
| public eager BF16 48项 | `797.875032 us` | `739.331684 us` | `-7.337408%` | `48快 / 0慢` |
| public eager FP32 48项 | `799.759631 us` | `741.049338 us` | `-7.340992%` | `48快 / 0慢` |
| public graph replay全部96项 | `1245.734637 us` | `1113.622186 us` | `-10.605184%` | `96快 / 0慢` |

public eager换算为每项平均`16.642028 -> 15.420636 us`，平均追回`1.221392 us`；逐项变化
的median为`-1.223313 us / -7.410257%`。BF16和FP32改善幅度一致，说明收益来自共同Host路径，
不是某一workspace dtype或kernel子集。

graph replay不执行Python标量缓存；其`-10.605%`是完整Task1 direct Torch workspace相对原始
内部workspace端点的既有device/graph收益，不能归因于本轮缓存。相反，eager和graph同时
`96/96`更快证明最终端点已经同时满足两种口径，不再存在第9.4节记录的public eager回退。

权威日志目录和SHA-256：

```text
/tmp/aiter-task1-scalarcache-resume.OYrIey
d4187cddb61ba9fb67166812e85825483c5ffba8cb255ce7541544680e45d31a  perf_public_A1.log
fb061da989cde67eb3b183c8d6cede082f557dabf878fad655c4d5ee745804a0  perf_public_B1.log
4679f6061c80dba903b43f903c0929fcb0684dd5becb4c342088f8208307e18a  perf_public_B2.log
cd945cd62a13a07c98e0065f79af7d0fcb457ee8309460f556539c42d68960e4  perf_public_A2.log
4725596f3b11ce414334c53ecacf395a3a2a44d8b3ce48d6e31277691b759b56  cpu_monitor.log
```

### 10.4 最终代码回归

所有测试都在最后一次位置参数/cache-key细化之后运行，并只使用物理GPU 4--7：

- focused：接口`65 passed, 9 skipped`，workspace`39 passed, 5 skipped`，graph/lifetime
  `9 passed, 4 skipped`，dispatch+GEMM`126 passed, 6 skipped`；合计
  `239 passed, 24 skipped, 0 failed`；
- gfx950 canonical 140-kid exhaustive：四个shard各`35 passed`，合计
  `140 passed, 0 failed`；
- 本轮只有Python与测试修改，因此复用已经验收的
  `/tmp/aiter-opus-ctypes-final.MINoRH`，没有伪造“fresh C++ build”结论。

回归日志目录：

```text
/tmp/aiter-task1-scalarcache-regression-resume.D88m6W
/tmp/aiter-task1-scalarcache-exhaustive-resume.6KfDGy
```

### 10.5 采用结论

保留有界纯标量launch-plan缓存和等价layout直写。同源码相邻ABBA证明缓存自身使public eager
改善`19.121%`且`96/96`项更快；最终public eager相对最初版本改善`7.339%`、graph replay
改善`10.605%`，两者也均为`96/96`项更快。Task1此前唯一剩余的public Host性能回退已经闭环。

本轮没有恢复per-stream workspace、Tensor cache、prepared pointer/launcher状态或跳过C++
checked validator，也没有启动Phase 2A/2B。若继续优化，应建立新的相邻版本目标和独立ABBA，
不能继续把第9.4节缓存前的`+23.457%`回退当作当前状态。

## 11. 2026-08-13 caller-resolved exact-kid统一入口

### 11.1 接口与实现边界

Python公开面收敛为一个入口：

```python
opus_gemm(
    XQ, WQ, Y,
    *,
    kid,
    layout="plain",
    x_scale=None,
    w_scale=None,
    bias=None,
    split_k=0,
    workspace=None,
)
```

调用方必须传最终kid并持有Y。入口从最终合并后的`kernels_list[kid]`获得唯一
`(arch,family,instance)`，dtype/layout/scale只验证物理合同，不参与shape-driven选核。删除
selector、tuned lookup、architecture heuristic、default kid、framework fallback以及旧公开family/
compat wrapper。C++ family raw ABI、A16 C ABI、Torch-owned workspace和generated checked launcher
保持不变。

为控制统一route的Host开销，保留两个有界纯标量缓存：

- `_cached_public_contract(maxsize=4096)`：key只含kid、三个dtype、layout、option presence和
  split-K，value只含canonical registry metadata；
- `_cached_explicit_a16w16_plan(maxsize=256)`：key/value只含arch/shape/kid/split和workspace
  shape/dtype计划。

两者都不保存Tensor、data pointer、workspace、stream、device allocation或launcher。每次调用
仍检查live A16 layout/A8 same-device合同，并进入C++ exact-instance validator。

### 11.2 统一入口相邻A/B

A端为统一前的A16 family public边界（当前私有`_launch_a16w16`，行为等价于删除前的
`opus_gemm_a16w16_launch`）；B端为唯一public `opus_gemm`。两端使用相同当前源码、JIT、kid、
split-K、Tensor和workspace，固定物理GPU 4、CPU 68，顺序为
`baseline-public A1 -> family F1 -> public P1 -> public P2 -> family F2 -> baseline-public A2`。
每端96项，每项`20 warmup + 9 rounds x 100 launches`。

| 测量项 | 旧family边界 F | 统一public P | P相对F | 逐项方向 |
|---|---:|---:|---:|---:|
| eager全部96项 | `1487.236173 us` | `1543.048008 us` | `+3.752722%` | `0快 / 96慢` |
| graph replay全部96项 | `1114.902036 us` | `1114.322884 us` | `-0.051946%` | `60快 / 36慢` |

eager每项增加的median为`0.621305 us`。graph不重放Python route，`-0.052%`且方向混合，属于
轮次噪声。这是统一arch/family/dtype/layout合同所付出的明确Host成本，不能把旧family边界的
`-6.738%`收益直接写成新public结果。

### 11.3 相对最初C++内部workspace的最终三口径

baseline端同时使用冻结的最初Python源码和冻结JIT；current端同时使用当前源码和已验收的
full-142-kid JIT。这样避免“旧`.so`配当前Python binding”把baseline raw人为放慢。raw和public
分别使用匹配边界的两轮逐case median平均：

| 口径 | 最初版本 | exact-kid最终版本 | 变化 | 逐项方向 |
|---|---:|---:|---:|---:|
| raw eager | `1465.784459 us` | `1329.361318 us` | `-9.307176%` | `96快 / 0慢` |
| public eager | `1594.678216 us` | `1543.048008 us` | `-3.237657%` | `95快 / 1慢` |
| public graph replay | `1245.859275 us` | `1114.322884 us` | `-10.557885%` | `96快 / 0慢` |

最终三个汇总口径都优于最初版本。public eager的改善从统一前第10节的约`-7.339%`缩小到
本轮同场`-3.238%`，原因是新增统一route；raw和graph没有执行该Python Host逻辑，收益保持。
不同ABBA轮次的绝对总和会受CPU/GPU状态影响，因此版本判断以各自相邻/首尾配对百分比为准。

### 11.4 版本性能总表

下表把历史上不同版本和不同实验边界分开列出；“增量”只与该行明确的A端相比：

| 版本/实验 | raw eager | public eager | graph | 比较A端 |
|---|---:|---:|---:|---|
| 最初C++内部workspace | `0%` | `0%` | `0%` | 基线 |
| Torch workspace + pybind raw | `+13.899685%` | 未测同边界 | `-10.186785%` | 最初版本，三端同场 |
| Torch workspace + ctypes raw | `-6.476921%` | 未测同边界 | `-10.547134%` | 最初版本，三端同场 |
| C ABI替换pybind（Phase 1增量） | `-16.511957%` | `-11.318283%` | `-0.128% raw / +0.224% public` | 同源码pybind边界 |
| device cache + layout去重增量 | 未变 | `-11.103219%` | `+0.225921%` | `4af16d5f` public |
| scalar launch-plan cache增量 | 未变 | `-19.121391%` | `-0.025667%` | 同源码cache-off |
| scalar-cache最终版对最初 | `-7.918684%` | `-7.339202%` | `-10.605184%` | 最初版本，历史权威ABBA |
| exact-kid统一route增量 | 未变 | `+3.752722%` | `-0.051946%` | scalar-cache family边界 |
| **exact-kid当前最终版对最初** | **`-9.307176%`** | **`-3.237657%`** | **`-10.557885%`** | 冻结源码+JIT匹配复测 |

其中Phase 1、device cache和scalar cache三行是相邻版本归因实验，不能直接把百分比相加；
“对最初”两行才是首尾口径。最新raw复测与历史`-7.919%`的差异是独立轮次漂移，两次方向和
96/96逐项结论一致。

### 11.5 正确性和artifact

最终统一接口在物理GPU 4（gfx950）完成：

- focused：`124 passed, 17 skipped, 0 failed`；
- gfx950 canonical exhaustive：`140 passed, 0 failed`；
- GPU 4--7均为gfx950，因此gfx942/gfx1250只报告CPU registry/codegen/合同测试，未伪称实机
  数值通过。

性能artifact：

```text
/tmp/aiter-opus-exactkid-final3-original-source-20260813
/tmp/aiter-opus-exactkid-final-public-20260813
/tmp/aiter-opus-exactkid-adjacent-20260813
```

最终public六轮SHA-256：

```text
88903f83f93e651be36254d0810d66dea0b8335432e9963f303e089c817eea84  perf_baseline-public_A1.log
1072b6572887408a93342a1cc7b2e74afdec2a24a33418b605f7a09c38cc4b70  perf_baseline-public_A2.log
b11579e393ac969114646ba4b149d3f09fafa94dce1c90f6c5343206cefe67d4  perf_family_F1.log
310c79a45d9c244f207df02224552729b4d3d20aac02bbe14a164ea068120bb3  perf_family_F2.log
07188a3b5d346de2127b82aa5397e3706e0e69d4927ae04b403ec549d55cf199  perf_public_P1.log
b240b81720f435cca7e8d50f5acfe5815c58f7abfecb13f8da9a8b6fd6896099  perf_public_P2.log
```

## 12. 2026-08-13 移除通用JIT改动、保留局部C ABI

### 12.1 最终文件边界

- `aiter/jit/core.py`已恢复为原始Task1基线内容；验证命令
  `git diff --exit-code ca68b4f... -- aiter/jit/core.py`通过；
- 不新增生产Python interface文件。早期selector和三架构heuristic文件最终全部删除；基线等价
  heuristic policy收敛在现有A16文件中，由上层caller在public exact调用前执行；
- mixed pybind/C ABI所需的最小CDLL装载、固定A16参数签名、Tensor descriptor转换和TLS错误读取
  位于已有`aiter/ops/opus/gemm_op_a16w16.py`；
- 首次调用仍由原pybind wrapper完成正常lazy JIT build/rebuild和架构检查，成功后从同一个
  `module_deepgemm_opus.so`加载C ABI；后续eager调用走局部C ABI；
- 局部缓存只保存CDLL/function和ABI helper，不保存Tensor、data pointer、workspace或stream。

这使OPUS特例不再扩散到通用编译核心，同时保留Task1的Torch-owned workspace、checked C ABI、
graph和live-stream语义。

### 12.2 正确性与fresh JIT

物理GPU 4、空目录`/tmp/aiter-task1-corefree-fresh.jrkldN`：

- fresh生成并编译41个gfx950 subset kid，成功产出并加载
  `module_deepgemm_opus.so`；
- 第一次kid 200、split-K 2调用经原pybind raw完成lazy build并通过数值校验；
- 首调后把Python pybind raw替换为“调用即失败”的哨兵，第二次调用仍成功并通过数值校验，
  直接证明第二次进入局部C ABI而没有回退到pybind；
- OPUS模块构建/首调约`9.838 s`，第二次同步wall time约`0.228 ms`，loader cache为
  `CacheInfo(hits=1, misses=1, maxsize=1, currsize=1)`。

物理GPU 4、复用未修改的full-kid JIT继续完成完整回归：

- private C ABI：`13 passed, 2 skipped`；
- focused：`124 passed, 17 skipped`；
- gfx950 canonical exhaustive：`140 passed`；
- `py_compile`和`git diff --check`通过。

### 12.3 性能防回退

固定物理GPU 4、CPU 68、96项、每项
`20 warmup + 9 rounds x 100 launches`，顺序为
`ctypes C2 -> public P1 -> public P2 -> ctypes C3`。两轮平均：

| 口径 | 局部C ABI版本 | 第11节此前当前值 | 独立轮次变化 |
|---|---:|---:|---:|
| raw eager | `1288.246629 us` | `1329.361318 us` | `-3.092815%` |
| public eager | `1472.746886 us` | `1543.048008 us` | `-4.555991%` |
| public graph replay | `1114.347694 us` | `1114.322884 us` | `+0.002226%` |

这轮用于证明移出`core.py`没有性能回退。它没有同时重跑冻结的最初版本，因此不能取代第11.3
节的首尾权威ABBA；与最初绝对值直接换算只作观察，不升级为新的首尾结论。

## 13. 2026-08-13 直接复用基线`kernels_list`

### 13.1 范围纠正

不同arch的kid编号段、各arch子表和合并后的`kernels_list`在原始Task1基线
`ca68b4f...`中已经存在，不是Task2新增。最终public入口已去掉后加的
`get_kernel_route()`包装，改为直接执行`kernels_list.get(kid)`；instance tag只负责进入现有
A16或A8文件，dtype/layout/scale规则也留在对应family文件。没有新增生产Python文件，
`aiter/jit/core.py`与`ca68b4f...`的blob仍同为
`43231ab3cd9ea24caaa6e8535b71455386dbe0f5`。

### 13.2 正确性

物理GPU 4（gfx950）最终结果：

- CPU/接口/workspace：`98 passed, 4 skipped`；
- focused：`124 passed, 17 skipped`；
- gfx950 canonical exhaustive：`140 passed`；
- `py_compile`和`git diff --check`通过。

### 13.3 相邻性能防回退

固定物理GPU 4、CPU 68、相同full-kid JIT、96项、每项
`20 warmup + 9 rounds x 100 launches`。A端是改动前两轮，B端是直接dict查询后三轮。
每个case先分别取A的两轮median和B的三轮median，再求96项总和：

| 口径 | 改前route包装 A | 直接`kernels_list.get` B | 变化 | 逐项方向 |
|---|---:|---:|---:|---:|
| public eager | `1496.886513 us` | `1497.065125 us` | `+0.011932%` | `48快 / 48慢` |
| public graph replay | `1113.556335 us` | `1114.603673 us` | `+0.094053%` | `29快 / 67慢` |

eager变化约万分之一且方向正好对半，判定无Host性能回退。graph replay不执行Python查询，
其约`0.09%`变化只能作为本轮device测量噪声，不能归因于dict查询。raw/C++ ABI未改，本轮不把
未重测的raw写成新结果。

完整日志：

```text
/tmp/aiter-opus-thin-router.ILVqXb
5cf3694fc0e5ad31eafa28aeaaae992950686f0c86d6223c73d3d2b539b7d31f  before-1.log
f9bed9d8f11621c207dfe00f0bf25e8564a29bea396f0bab672329f108212114  before-2.log
cbc0b9aaaac03b8e4d2225337b88371a41589430f58b50d2978224cc49281732  after-direct-list-1.log
b826f24d93d51b52f19d7f7c52a391e48812de67d793652f9f4cb244f81f771d  after-direct-list-2.log
3f85f7186069ef8aba33c9ec599472b69061a615a96f7642e7aec3fa1e605da9  after-direct-list-3.log
```

## 14. 2026-08-13 PR #4320 MXFP8 BMM 接入后复测

### 14.1 当前结构

45个gfx950 BMM id按`global_kid = 8000 + upstream_kid`加入统一`kernels_list`。Python raw、
exact launcher和Torch workspace计划位于已有`gemm_op_a8w8.py`，tuned调用方位于已有
`batched_gemm_op_a8w8.py`；public仍只有`opus_gemm`。fresh JIT为
`/tmp/aiter-pr4320-bmm-fresh2.r7Buc2`，生成表大小为45，生产与生成路径均无workspace handle
或内部workspace allocator。

### 14.2 正确性

- focused OPUS：`126 passed, 17 skipped`；
- gfx950 A16 canonical：`140 passed`；
- BMM M-alignment与exact launcher：45/45 kid；
- token-major、batch-major、tile-N、automatic/caller workspace和Graph replay均通过；
- tuned config collision：`13 passed`；
- registry：2084项，其中BMM 45项，无重复global key。

### 14.3 Task1 / Task2 / 当前接口性能

物理GPU 4（MI355X），A16 kid 200，BF16/FP32分别测量后取均值；每个endpoint两轮，
每轮`20 warmup + 9 rounds x 100 launches`，顺序为
`Task1 -> Task2 -> current -> current -> Task2 -> Task1`：

| 层级 | 冻结Task1 | 冻结Task2 | 当前BMM模块 | 当前相对Task2 |
|---|---:|---:|---:|---:|
| public/high-level eager | `23.1645 us` | `14.6772 us` | `15.4390 us` | `+5.190%` |
| compile_ops raw eager | `18.1955 us` | `13.1684 us` | `13.3347 us` | `+1.263%` |
| direct pybind/C++ | `8.4800 us` | `8.4417 us` | `8.4235 us` | `-0.216%` |
| graph replay | `9.4873 us` | `9.5034 us` | `9.4866 us` | `-0.177%` |

current public相对Task1快`33.351%`。相对Task2的约5% public差异仍是统一Python route；
direct和graph持平，说明BMM并入同一个`.so`没有改变A16 device路径。

### 14.4 BMM性能

BF16 Y，`G={2,8,16}`、`M={1,16,128,512,2048,8192}`、`N=1024`、`K=4096`共18项：

| 路径 | 固定global kid 8000累计 | tuned统一路径累计 | 变化 | 逐shape speedup |
|---|---:|---:|---:|---:|
| token-major high-level | `3640.135 us` | `1437.482 us` | `-60.51%` | `1.23x--2.86x` |
| batch-major view public | `3608.364 us` | `1409.615 us` | `-60.93%` | `1.15x--2.90x` |

artifact目录为`/tmp/aiter-pr4320-final-20260813`，包含六份接口日志和一份BMM日志。

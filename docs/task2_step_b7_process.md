# OPUS GEMM 任务二 Step B7 验收记录

> 状态：B7.1--B7.6 已执行完成；性能验收记录一项已定位的 gfx950 no-scale public
> adapter 开销。更新时间：`2026-08-11 14:48 UTC`。
>
> 本轮以当前 checked-only workspace 为基线；没有恢复 prepared/prevalidated 实验，没有
> reset、checkout、clean、commit 或清理既有 dirty 工作树。

## 1. B7.1 基线

开始前执行了 `git status --short`、`git diff --check`、checked-only 禁止项扫描和进程检查。

- dirty 文件集合与 `docs/task2_checkpoint.md` 记录的 Task1 + Task2 B0--B6 累计成果一致；
- `git diff --check` 返回 0 且无输出；
- `prevalidated|PreparedWorkspace|OpusA16W16WorkspaceDispatch|workspace_try_dispatch` 在
  `csrc/opus_gemm` 和 `op_tests` 中零命中；
- 没有运行中的 pytest、codegen、CMake、Ninja 或 HIP 编译进程；系统存在其他 Codex 进程，
  但本轮前后未观察到源码状态漂移，因此没有终止或覆盖它们。

## 2. B7.1 静态检查结果

按总计划执行后的结果如下。`rg` 返回 1 且无输出表示零命中。

| 检查 | 退出码 | 结果 |
|---|---:|---|
| `git diff --check` | 0 | 通过，无输出 |
| 四个目标文件 `python3 -m py_compile` | 0 | 通过，无输出 |
| `OPUS_GEMM_PYBIND|_opus_gemm_bf16_dispatch` | 1 | 全部指定范围零命中 |
| `opus_gemm_lookup\.h|opus_select_a16w16_kid|find_shape_kid` | 0 | 原样扫描仅命中下述清理白名单一项 |
| 两个旧 C++ tune 名 | 0 | 原样扫描仅命中下述清理白名单一项 |

原样扫描的两个输出是：

~~~text
csrc/opus_gemm/gen_instances.py:926:            "opus_gemm_lookup.h",
csrc/opus_gemm/gen_instances.py:927:            "opus_gemm_a16w16_tune_lookup.h",
~~~

它们与同一 tuple 中的 `opus_gemm_a8w8_tune_lookup.h` 一起，仅供
`Path(...).unlink(missing_ok=True)` 删除复用 blob 目录里的陈旧生成文件。该保留项已经由
`op_tests/test_opus_interfaces.py` 的重复生成测试覆盖，并在 checkpoint 第 4 节明确允许。

为区分清理 inventory 与生产引用，保持字符串原样并复扫生产范围：

~~~bash
rg -n --glob '!gen_instances.py' \
  'opus_gemm_lookup\.h|opus_select_a16w16_kid|find_shape_kid' csrc/opus_gemm

rg -n --glob '!gen_instances.py' \
  'opus_gemm_a16w16_tune|opus_gemm_a8w8_blockscale_bpreshuffle_tune' \
  csrc/opus_gemm csrc/pybind csrc/include
~~~

两条复扫都返回 1 且无输出。由此 B7.1 的生产 C++/pybind 旧符号条件通过；没有删除清理机制，
也没有拆分字符串来制造扫描假阴性。总计划 B7.1 已同步写明这一白名单口径。

## 3. B7.1 变更

B7.1 没有修改生产源码或测试，只新增本验收记录并更新计划/checkpoint。检查前后的
`git status --short` 中原有 tracked/untracked 源码集合不变；`py_compile` 产物受 ignore 规则
管理，没有形成新的可见工作树条目。

## 4. B7.2 完整 fresh codegen 与字节稳定性

所有生成物位于新的临时根目录：

~~~text
/tmp/aiter-b7.2.km7yQp
~~~

每个架构分别建立 `run1`、`run2`，用包含该架构全部 canonical kids 的独立 sidecar 驱动真实
`gen_instances.py` CLI。没有使用默认 32-kid subset，也没有写仓内 JIT blob：

| arch | sidecar/canonical kids | impl | regular device TU | host TU | reduce TU |
|---|---:|---:|---:|---:|---:|
| gfx942 | 23 | 23 | 24 | 1 | 1 |
| gfx950 | 142 | 142 | 234 | 1 | 1 |
| gfx1250 | 1874 | 1874 | 3252 | 1 | 1 |

三组 `diff -qr run1 run2` 均返回 0。将相对文件名和每个文件 SHA-256 再聚合后的目录 digest
如下，两次运行逐架构完全相同：

| arch | run1/run2 aggregate SHA-256 |
|---|---|
| gfx942 | `edd8255dfb58a3465c574c1c5287171af51548260d65d78ceaf83e8d3baef925` |
| gfx950 | `6a6268a870f39d9167afa2c0656f627603e78a8d680fc4881b2cb4ffb1cfa5d0` |
| gfx1250 | `b960551d502c5f17eb78fd1b37c567a028af3a36efea27fe437066340f9ba916` |

每份 `opus_build_archs.h` 仅有对应的一个 `OPUS_BUILD_HAS_GFX*` 定义。六个 fresh 目录均未
生成 `opus_gemm_lookup.h`、`opus_gemm_a16w16_tune_lookup.h` 或
`opus_gemm_a8w8_tune_lookup.h`。

## 5. B7.2 B0 集合、typed table 与数量分布

对 `run1` 的 generated header 解析实际 entry、检查声明 size、重复 kid 和完整集合 digest。
A16 结果与 B0 golden 完全一致：

| arch/table | count | kid-set SHA-256 |
|---|---:|---|
| gfx942 BF16 non-workspace | 14 | `62ba8933000e2392d38f368f555882a26361369e8966ca46d1e43e7633638dab` |
| gfx942 FP32 non-workspace | 1 | `39e5b4830d4d9c14db7368a95b65d5463ea3d09520373723430c03a5a453b5df` |
| gfx942 workspace | 8 | `3d34ca7ffe881e360cf767711d6ecefaaa8d7838d84c840e9f56a9ba0d4f6f3e` |
| gfx950 BF16 non-workspace | 92 | `f9743fd6634ab6d010798208d4fc0526f941ccb50be1eb064d2912719f1e8994` |
| gfx950 FP32 non-workspace | 92 | `f9743fd6634ab6d010798208d4fc0526f941ccb50be1eb064d2912719f1e8994` |
| gfx950 workspace | 48 | `64dc006db4356f018ec42fddec710ce8be6ef5c5e4b7c47356abeb46eb93a6a6` |
| gfx1250 BF16 non-workspace | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| gfx1250 FP32 non-workspace | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| gfx1250 workspace | 1874 | `c63348e05fcabc4bac52228dbae4ad46c851ef19ad23d79cbf1a80ef3e22639a` |

A8 typed table 检查结果：

- gfx950 no-scale FP32 表仅含 kid 2；plain-WQ blockscale FP32 表仅含 kid 1；
- gfx942 bpreshuffle BF16 表仅含 kid 11000，FP32 表为空；
- gfx950/gfx1250 bpreshuffle 的 BF16、FP32 表均以 `SIZE 0` + 空 initializer macro 生成；
- 两个 arch header 均用 `std::array<Entry, ..._SIZE>` 消费空表；empty macro body、manifest 和
  生成 impl 均不引用 bpreshuffle kernel symbol。

gfx1250 registry 和 generated impl 名集合逐项一致。数量/dtype 分布保持：

~~~text
two-stage plain:          28,  workspace dtype BF16 28
two-stage clusterlaunch: 468,  workspace dtype BF16 468
two-stage total:         496,  workspace dtype BF16 496
fused total:            1378,  workspace dtype BF16 780 / FP32 598
~~~

## 6. B7.2 synthetic registry/emitter 0 -> 1 fixture

`op_tests/test_opus_interfaces.py` 新增参数化结构 fixture：

~~~text
gfx950:  synthetic kid 900001, BF16 table 0 -> 1
gfx1250: synthetic kid 900002, FP32 table 0 -> 1
~~~

每个 case 先在当前 canonical registry 上生成并确认目标表 `SIZE 0`，再仅临时增加：

1. canonical `kernels_list` instance；
2. 对应 arch logical-family 的 tag membership；
3. `(arch, kernel_tag)` emitter registration。

随后完整调用 `opus_gemm_codegen.gen_instances()`，确认 emitter 被实际调用、host launcher产生、
目标 typed table 为 `SIZE 1`、另一 dtype 仍为 `SIZE 0`。fixture 是结构验证，不把合成项声明成
真实 kernel capability。

fixture 对公共 `opus_gemm.h`、目标 arch header、`rocm_ops.hpp`、pybind TU、Python A8 wrapper和
`__init__.py` 做前后逐字节比较，全部不变。两份 synthetic host TU 和两份 size-1 arch typed table
也分别通过目标架构 HIP 编译。定向 fixture 结果为 `2 passed`。

## 7. B7.2 全部 generated TU 编译

先用 `-fsyntax-only` 同时执行 HIP host/device pass，再用 `-O3 -c` 真正生成对象，避免把前端
语法检查写成完整编译。两种检查都覆盖 run1 的全部 host、per-kid device 和 reduce TU：

| arch | total TU | syntax | `-O3 -c` objects | zero-size | source/object mapping |
|---|---:|---:|---:|---:|---:|
| gfx942 | 26 | 26/26 | 26/26 | 0 | exact |
| gfx950 | 236 | 236/236 | 236/236 | 0 | exact |
| gfx1250 | 3254 | 3254/3254 | 3254/3254 | 0 | exact |
| total | 3516 | 3516/3516 | 3516/3516 | 0 | exact |

对象目录位于 `/tmp/aiter-b7.2.km7yQp/objects`，大小约为 gfx942 1.5 MiB、gfx950 7.5 MiB、
gfx1250 135 MiB。代表 device object 均含 `.hip_fatbin`，不是空的 host-only占位文件。

## 8. B7.2 定向回归、并发保护与下一步

registry、capability、B0 dispatch digest、synthetic fixture 和 gfx1250 dtype golden 的定向组：

~~~text
9 passed, 2 warnings in 2.78s
~~~

warnings 仍是 Cython `dep_util` deprecation。B7.2 没有修改生产源码；仅给
`op_tests/test_opus_interfaces.py` 增加上述可重复 fixture，并更新 B7 文档/checkpoint。没有恢复
prepared、没有清理 dirty 工作树。

执行期间 `docs/opus_gemm_next_performance_optimization_plan.md` 在 `13:41 UTC` 作为新的无关
未跟踪文件出现。它不在 B7.2 初始状态中；按并发保护将其视为外部改动，未读取、覆盖或删除。
没有观察到 OPUS 生产源码漂移，也没有终止其他 Codex 进程。

B7.2 完成时的下一步是 B7.3；其执行结果见下节。B7.2 的 compile 结果没有被当作 CPU 或
GPU 测试通过。

## 9. B7.3 CPU 测试结果

先原样执行总计划命令，现有测试得到：

~~~text
219 passed, 12 skipped, 2 warnings in 8.15s
~~~

随后逐条审计 B7.3 的覆盖清单。审计发现原集合虽已有当前环境的公共导出、mandatory 集合和
heuristic invariant 检查，但没有直接逐一执行 gfx942/gfx950/gfx1250 与 unsupported arch 的包
初始化，也没有用真实 OPUS CLI 验证完整 subset compile 公式。为避免把间接覆盖写成通过，在
`op_tests/test_opus_interfaces.py` 补充了以下 CPU fixture：

- 用隔离的 source loader 分别以 `GPU_ARCHS=gfx942/gfx950/gfx1250/gfx999` 执行真实
  `aiter.ops.opus.__init__`；三个支持架构导出完全相同且不是 stub，gfx999 的七个 arch-gated
  公共入口均为带 detected-arch 信息的调用时 `RuntimeError` stub；
- bpreshuffle fixture 增加成功 explicit kid 分支，并令 tuned lookup 在该分支被调用就失败；原有
  tuned、per-arch default、空 capability 和 foreign kid 分支保持；
- 构造临时 tuned CSV、compiled-kids sidecar 和生成目录，以 `GPU_ARCHS=gfx950` 执行真实
  `gen_instances.py` CLI；输出 sidecar 严格等于
  `(csv_opus_kids | sidecar_kids | HEURISTIC_DEFAULT_KIDS) & valid_kids` 经 arch filter 后再加入
  per-arch mandatory A8 kids 的集合；同时证明非 OPUS CSV row、off-arch kid 和 invalid kid 被排除；
- generated table golden 除宏名、size 和 kid-set digest 外，增加三架构五种函数指针签名、A16
  workspace/non-workspace entry 类型、A8 family entry 类型及 arch header 消费对应宏的断言。

补齐后原样命令最终结果：

~~~text
224 passed, 12 skipped, 2 warnings in 4.31s
~~~

同一集合用 `-rs` 复跑为 `224 passed, 12 skipped, 2 warnings in 4.50s`。两个 warning 都是
Cython `dep_util` deprecation，不是 OPUS 合同 warning。

## 10. B7.3 覆盖映射

| B7.3 合同 | 直接覆盖节点 |
|---|---|
| 新 signature、return Y、三架构导出、unsupported stub | `test_canonical_a16_python_cpp_and_pybind_signatures`、`test_python_a8_canonical_and_legacy_signatures`、`test_compat_and_canonical_explicit_launch_contracts_match`、`test_opus_package_exports_supported_arches_and_unsupported_stubs` |
| bpreshuffle explicit/tuned/default/no-kernel | `test_bpreshuffle_python_resolution_and_empty_capabilities` |
| gfx950/gfx1250 空 family 是 capability error，符号仍存在 | 上述 bpreshuffle fixture、逐架构 export fixture、`test_cpp_pybind_signatures_and_removed_legacy_raw_symbols` 和 A8 fake-schema fixture |
| deprecated warning 一次及旧参数适配 | `test_a16_tune_compat_warns_once_and_calls_canonical`、`test_a8_legacy_wrapper_warns_once_and_forwards_to_canonical`、`test_deepgemm_compat_warns_once_and_calls_canonical` |
| exact family/kid 拒绝 | `test_existing_a8_family_contracts_are_kid_and_arch_scoped`、`test_common_queries_are_arch_and_family_scoped`、`test_explicit_unknown_or_wrong_arch_kid_fails_strictly` |
| selector 顺序和 tuned-row 原子回退 | `test_explicit_selection_precedes_tuned_lookup`、`test_tuned_selection_precedes_heuristic`、两个 tuned-row fallback fixture |
| workspace shape/dtype/capacity | `test_a16w16_workspace_init_uses_actual_kid_tile_and_dtype`、fused layout fixture、raw one-element-short/invalid-contract fixture；实机项按下节单列 |
| generated 表命名、类型和 kid 集合 | `test_generated_dispatch_kid_sets_match_b0_golden` 及 synthetic 0 -> 1 fixture |
| subset compile 公式及 mandatory kid | `test_subset_compile_formula_arch_filter_and_mandatory_kids`、`test_a8_capability_slots_and_mandatory_compile_set_are_explicit`、`test_heuristic_kid_must_be_in_force_compiled_set` |
| generic/private legacy binding 不存在 | `test_cpp_pybind_signatures_and_removed_legacy_raw_symbols`、`test_python_a8_canonical_and_legacy_signatures` |

## 11. B7.3 skip 边界、变更和下一步

12 个 skip 全部明确要求当前不可用硬件：

- gfx942 A8 bpreshuffle 数值 1 个、负例参数化 7 个；
- gfx942 raw typed-workspace 2 个；
- gfx1250 raw typed-workspace 1 个、batch raw 1 个。

这些只记录为“未执行硬件项”，不是通过；B7.3 也不替代 B7.4。gfx950 对应可执行项在本集合中
没有 skip。

B7.3 没有修改生产源码，只增强 `op_tests/test_opus_interfaces.py` 并更新验收文档。最终
`git diff --check`、四个目标测试文件 `py_compile` 均通过；checked-only 禁止项扫描返回 1 且
无输出。没有恢复 prepared 实验，没有清理 dirty 工作树，也没有读取、覆盖或删除并发新增的
`docs/opus_gemm_next_performance_optimization_plan.md`。

下一步从 B7.4 开始，按当前可用硬件执行 GPU 数值、负例、graph 和双 stream 验收；gfx942/
gfx1250 无对应实机时继续明确写“未执行”，不能用 B7.2 cross-compile 或本节 pytest skip 代替。

## 12. B7.4 硬件边界与 fresh gfx950 模块

实测前重新检查了设备和占用。当前节点有 8 张 `AMD Instinct MI355X`，HIP 架构均为
`gfx950:sramecc+:xnack-`、每卡 256 CU；检查时显存占用为 0、利用率约 0--1%，没有其他 KFD
进程。节点上没有 gfx942 或 gfx1250，因此本节只有 gfx950 可以形成实机结论。

本节没有复用 prepared 实验模块，而是从当前 checked-only workspace 在新的目录构建：

~~~text
/tmp/aiter-b7.4-gfx950.U4cvOE
~~~

构建环境为 `GPU_ARCHS=gfx950`。输入 sidecar 是 canonical 140 个 gfx950 A16 kids；生成器再按
subset 公式加入 mandatory A8 kid 1/2，最终 sidecar 为 142 项，SHA-256 为：

~~~text
b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7
~~~

生成报告为 `|S|=142`、`CSV=35`、`sidecar=140`、`heuristic-default=8`、`mandatory-a8=2`。
`module_aiter_core` 和 `module_deepgemm_opus` fresh 构建分别约 10.3 s、6.0 s；最终 OPUS `.so`
大小 6,409,880 bytes。generated `opus_build_archs.h` 只定义
`OPUS_BUILD_HAS_GFX950 1`。随后所有 gfx950 测试均复用这一 `.so`，没有并发重建。

## 13. B7.4 gfx950 A8、路由、graph 和双 stream

逐条审计 B7.4 清单后，只增强测试 fixture，没有修改生产源码：

- plain-WQ blockscale 负例新增 `w_scale` shape/dtype，并让 `x_scale`、`w_scale` 分别位于第二张
  可见 gfx950，验证两个跨 device 拒绝；同时实际调用 public wrapper 的单 scale和 `None` scale
  负例；
- no-scale kid 2 增加 tensor shape拒绝；既有 K-contiguous和 unknown kid拒绝保持；
- gfx950 bpreshuffle 同时检查 public `kid=None` 的空 capability错误，以及 private exact raw 对
  gfx942 kid 11000 的 generated typed-table拒绝；两条错误都明确含
  `no registered kernel ... a8w8_blockscale_bpreshuffle ... gfx950`；
- 高层 `gemm_a8w8_blockscale_bpreshuffle` 的代表分支从 CK/ASM 扩成
  CK/CKTile/ASM/Triton。该 fixture 用受控 backend替身验证分支选择和参数传递，并把 canonical
  OPUS bpreshuffle入口设为 must-not-run；它是路由保持测试，不冒充四个 backend kernel的数值测试；
- 新增 gfx950 kid 2 no-scale和 kid 1 blockscale各自的真实 graph capture/replay及双 stream
  数值测试。两条 family均用 public canonical wrapper、caller提供的 `Y` 和显式 exact kid；
  scale=1 时与 FP32 Torch matmul逐元素一致；
- 既有 A16 graph/replay和双 stream继续实际覆盖 external-workspace kid 200，因此 graph和双
  stream现在均覆盖一个 workspace family以及两个 A8 family。

定向集合在物理 GPU 4、5（进程内 device 0、1）执行：

~~~text
23 passed, 2 warnings in 7.71s
~~~

其中 kid 2 no-scale和 kid 1 plain blockscale数值 golden均通过；新增/既有 A8 shape、dtype、
device、single-scale、prefetch、layout、exact-family/kid和空 capability负例全部按预期拒绝。
两个 warning 仍只是 Cython `dep_util` deprecation。

## 14. B7.4 gfx950 focused、48 workspace 和 140-kid 全量

复用同一 fresh 模块，在物理 GPU 4、5执行总 focused 集合：

~~~text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py \
  op_tests/test_opus_interfaces.py

236 passed, 22 skipped, 2 warnings in 4.44s
~~~

用 `-rs` 再执行一次为 `236 passed, 22 skipped, 2 warnings in 4.56s`。22 个 skip逐项来自
gfx942/gfx1250硬件条件：workspace 4、A16 graph/stream 4、A16数值/redirect/batch 6、gfx942
A8数值/负例 8；gfx950 项无 skip、无失败。通过数较 Task1 checked-only快照增加，是 B7.3/B7.4
新增接口和 A8覆盖，不表示原 Task1用例集合发生回退；Task1已有 focused行为保持为 0 failed。

完整 140-kid sweep 使用同一 `.so`，物理 GPU 4--7各跑一个稳定 ordinal shard，四个进程均未设
`AITER_REBUILD`：

| shard | 结果 |
|---:|---|
| 0 | `35 passed, 0 failed` |
| 1 | `35 passed, 0 failed` |
| 2 | `35 passed, 0 failed` |
| 3 | `35 passed, 0 failed` |
| 合计 | `140 passed, 0 failed` |

每个 non-workspace kid实际运行 BF16/FP32 Y并证明没有 workspace分配；每个 workspace kid实际
运行 BF16/FP32 Y、caller workspace复用和 auto workspace生命周期。为把 workspace结论从总数中
显式分离，又在单卡重跑完整 workspace test node：

~~~text
48 passed, 2 warnings in 9.63s
~~~

因此 gfx950 external-workspace集合是实机 `48/48 passed`，完整 canonical A16集合是
`140/140 passed`，相对 Task1没有新增失败。

历史 mono FP32 十项另行选择并执行：

~~~text
1400, 1401, 1402, 1403, 1404
6400, 6401, 6402, 6403, 6404

10 passed, 130 deselected, 2 warnings in 3.52s
~~~

它们全部是 Task1已定位并修复的 mono/4G-safe non-workspace项；本节只记录保持性通过，不把它们
包装为 Task2新发现、新失败或新修复。

## 15. B7.4 gfx942/gfx1250 明确未执行

当前节点没有对应架构硬件，以下项目全部是 **未执行**，不是通过：

| arch | 未执行的实机项目 |
|---|---|
| gfx942 | A16 BF16/FP32 workspace代表 kid；10210/10213 redirect与10216拒绝；kid 11000 bpreshuffle数值；canonical `kid=None` default 11000；真实 tuned OPUS row经高层到 canonical raw；2D/3D batch=1、scale shape和exact tile负例；graph和双 stream |
| gfx1250 | two-stage BF16 workspace代表 kid；fused BF16/FP32 workspace代表 kid；batch>1拒绝；compile-time fused split-K对 runtime splitK不变；FP8-E8M0 FlyDSL和FP32-scale Triton/Gluon高层实机路径；预留OPUS bpreshuffle空 capability实机错误；graph、双 stream和caller workspace复用 |

B7.2 的 gfx942/gfx1250 generated TU交叉编译、B7.3 CPU fixture以及本次 focused输出中的 skip均
没有被用来替代上述实机结果。相关 route/capability CPU fixture仍可作为接口结构证据，但不改变
“未执行”的硬件结论。

## 16. B7.4 变更、保护与下一步

B7.4 没有修改生产源码；只增强 `op_tests/test_opus_interfaces.py` 和
`op_tests/test_opus_graph.py` 的验收覆盖并更新文档。没有恢复 prepared/prevalidated路径，没有
reset、checkout、clean或清理 dirty 工作树；无关未跟踪文件
`docs/opus_gemm_next_performance_optimization_plan.md` 保持未触碰。

B7.4完成后的下一步是 B7.5 ABI和符号检查。B7.5应复用已构建模块或按目标架构构建独立模块，
核对四个新 raw名字、三架构一致的 bpreshuffle符号、generic/旧 C++ tune缺失、Python compat
wrapper保留、unsupported arch import连续性以及 fake schema参数顺序。

## 17. B7.5 三架构独立模块构建

在新的临时根目录为每个架构建立独立 JIT目录，并用当前 checked-only源码实际生成、编译、链接
`module_deepgemm_opus.so`：

~~~text
/tmp/aiter-b7.5-abi.oaplCv/
  gfx942/
  gfx950/
  gfx1250/
~~~

三个构建均设置单一 `GPU_ARCHS` 和 `AITER_REBUILD=1`；它们是 ABI subset build，不承担 B7.4
中缺失硬件的数值结论。结果如下：

| arch | subset kids | module build | `.so` bytes | `.so` SHA-256 |
|---|---:|---:|---:|---|
| gfx942 | 21 | 4.6 s | 1,464,400 | `786cd2e87dbbed9b22a3f10357cb547ec9a0501baf9449c380837a8d358c03f1` |
| gfx950 | 41 | 5.1 s | 2,099,880 | `2d584d204b7db1122e2c20d42b75bb0e414ff8b6c722eb458b845dd3472d23a8` |
| gfx1250 | 6 | 4.7 s | 641,192 | `80dae7506f3e8da1e4242c960cdef5910c6feaad6d625e8f99a6f89e44fc2409` |

各目录同时 fresh构建了 host辅助 `module_aiter_core`。三份 `opus_build_archs.h` 分别且仅定义
`OPUS_BUILD_HAS_GFX942`、`OPUS_BUILD_HAS_GFX950`、`OPUS_BUILD_HAS_GFX1250`；从最终 ELF
offload bundle提取的 arch集合也分别严格为 `{gfx942}`、`{gfx950}`、`{gfx1250}`。

## 18. B7.5 动态 pybind、ELF 与 Python暴露面

每份 `.so` 在独立 Python进程中直接加载；off-arch构建只做 host ABI/属性检查，没有启动 kernel。
三架构得到相同的业务 pybind属性集合：

~~~text
opus_gemm_a16w16_launch
opus_gemm_a8w8_launch
opus_gemm_a8w8_blockscale_launch
opus_gemm_a8w8_blockscale_bpreshuffle_launch
~~~

模块另有所有 AITER扩展共用的 `_set_current_hip_stream` 辅助属性；除它之外，以 `opus_` 开头的
业务属性恰好是上述四项。三份模块均满足：

- C++ module attribute和 `nm -C --defined-only` 均包含四个 canonical launch；
- `extension.opus_gemm` 不存在，ELF中也没有精确 `opus_gemm(...)`；
- `opus_gemm_a16w16_tune` 和
  `opus_gemm_a8w8_blockscale_bpreshuffle_tune` 均不属于 C++/pybind/ELF；
- Python private raw `_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw` 与 package public
  `opus_gemm_a8w8_blockscale_bpreshuffle_launch` 在三个目标架构环境中均存在；空 capability
  没有通过删除符号实现；
- generic `_opus_gemm_bf16_dispatch`、旧 A16 private tune raw和旧 bpreshuffle private tune raw
  均不存在；
- public Python仍在 `__all__` 和实现模块中导出
  `opus_gemm_a16w16_tune` 与
  `opus_gemm_a8w8_blockscale_bpreshuffle_tune` 两条 deprecated wrapper。定向 warning fixture
  继续验证每次兼容调用只产生一条 `DeprecationWarning`。

动态审计脚本对三构建全部 exit 0。gfx942/gfx950/gfx1250各自报告
`bpreshuffle_raw=true`、`bpreshuffle_public=true`，且 `pybind_business_attrs` 集合逐字一致。

## 19. B7.5 top-level import 与 fake schema顺序

为避免把完全未知架构的其他 AITER helper错误混入结论，unsupported OPUS检查使用 AITER认识但
OPUS不支持的 `GPU_ARCHS=gfx90a`，在 fresh subprocess执行真实 `import aiter` 和
`from aiter import *`。结果：

- OPUS package检测到 `gfx90a`，导出调用时 `RuntimeError` stub而不是 import-time异常；
- stub错误明确包含 `detected 'gfx90a'`；
- 位于 top-level OPUS star import之后的 `rmsnorm2d_fwd_with_add`、`topk_plain`、
  `fused_split_gdr_update` 和最终 `mla` 均存在；
- 显式 `from aiter import *` 同样完成并包含上述后续 op。

fake schema审计将四层参数顺序逐项比较：generated pybind doc、Python raw源码定义、fake generator
签名和 `torch.ops.aiter` schema。四条最终顺序为：

~~~text
a16w16:                 XQ, WQ, Y, bias, workspace, kid, split_k
a8w8:                   XQ, WQ, Y, kid
a8w8 blockscale:        XQ, WQ, Y, x_scale, w_scale, kid
a8w8 bpreshuffle:       XQ, WQ, x_scale, w_scale, Y, kid
~~~

A8 schema中的既有内部 `dummy` Tensor由 `compile_ops` 注册层添加，不属于 Python/C++ ABI，比较时
明确排除。decorated raw callable运行时显示通用 `(*args, **kwargs)`，因此 raw合同取 AST中的实际
函数定义，而不是把 wrapper反射结果误当 ABI。三架构动态审计和增强后的持久 fixture均通过。

## 20. B7.5 回归、变更与下一步

新增/增强的持久测试位于 `op_tests/test_opus_interfaces.py`：

- 新增 `gfx90a` subprocess，证明 unsupported OPUS不会截断 `aiter` 顶层后续 import；
- 将 A16/A8 fake-registration检查从仅查找 `kid` 字样增强为四条 schema的完整有序参数合同，
  并同步比较 raw源码和 fake generator签名。

定向 ABI/signature/compat集合结果：

~~~text
12 passed, 2 warnings in 3.14s
~~~

新增 top-level和两条 schema fixture定向结果为 `3 passed, 2 warnings in 6.43s`；完整 interfaces
为 `63 passed, 8 skipped, 2 warnings in 7.66s`。随后原 B7 CPU四文件综合回归在当前测试集合上
为：

~~~text
231 passed, 12 skipped, 2 warnings in 7.70s
~~~

12个skip仍全部是 gfx942/gfx1250实机项；两个warning仍只是 Cython `dep_util` deprecation。
B7.5没有修改生产源码，只增强验收测试和文档。没有恢复 prepared/prevalidated实验，没有清理
dirty工作树，也没有触碰无关未跟踪性能计划。

B7.5完成后的下一步是 B7.6性能检查。当前只有 gfx950硬件可执行性能对比；gfx942/gfx1250若仍
无对应硬件，必须继续记录未执行，不能用本节的 cross-arch ABI build替代性能实测。

## 21. B7.6 端点、输入和测量方法

### 21.1 独占硬件与保留端点

正式测量前、ABBA四轮之间和全部退出后均检查到8张 MI355X/gfx950为0%利用率、0% VRAM，
没有KFD PID。所有数据来自物理 GPU 0上的串行独立进程，没有与其他测试或编译并发。

Task1使用B0冻结ABI和Task1有效workspace性能轮次使用过的 caller-owned-workspace模块：

~~~text
/tmp/aiter-gfx950-current.NtJydE/module_deepgemm_opus.so
SHA-256: 6de932079275d0d6cfde7ad889725864e182fae0eb65532a451725c1d1b6651a
exports: opus_gemm, opus_gemm_a16w16_tune,
         opus_gemm_a8w8_blockscale_bpreshuffle_tune
~~~

Task2使用B7.4 full 140-kid验收模块：

~~~text
/tmp/aiter-b7.4-gfx950.U4cvOE/module_deepgemm_opus.so
SHA-256: 457b26cfb7aa1da2518fa5752f990858aa0302cdb3c098965b702635ab325d34
exports: 四个 canonical family launch
~~~

两端 `compiled_kids_opus.json` 均为相同142-kid全集，SHA-256均为
`b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7`。Task1后续mono FP32
修复只修改 non-workspace kid 1400--1404、6400--6404，不涉及本轮使用的 kid 200、1、2；
因此没有把mono差异混入接口比较。两端均直接复用保留 `.so`，未设置 `AITER_REBUILD`，没有
恢复prepared/prevalidated源码或模块。

新增可复现基准：

~~~text
op_tests/bench_opus_task1_task2_interfaces.py
695 lines
SHA-256: 22c44e87dfe7c3453e7da5eeeb7c2dcdda96628451cc8c3274a0dca43a34fb4b
~~~

Task1高层口径把B0旧raw注入未改变的shape selector/workspace链；B2的执行记录证明生产迁移只把
该raw目标从旧tune名换为canonical launch名。Task1 explicit口径按B0有效参数路径执行
`selector -> actual kid -> Torch workspace -> old raw`，不调用当前deprecated shim，也不恢复
任何旧生产入口。A8 Task1口径调用B0真实generic pybind；Task2口径调用新的family public/raw。

固定case如下：

| family | 输入/输出 | actual kid | split-K | workspace/scale |
|---|---|---:|---:|---|
| A16 high-level/explicit | BF16 `1x64x2048`，BF16与FP32 `Y=1x64x64` | 200 | 2 | FP32 `2x1x64x64` |
| A8 no-scale | native FP8 `1x256x256`，FP32 Y | 2 | N/A | 无scale |
| A8 plain blockscale | 同上 | 1 | N/A | FP32 x/w scale，128 block |
| A8 bpreshuffle | gfx950无registered kernel | — | N/A | 不生成伪数值 |

输入由固定seed或固定整数周期生成；同一case两端的shape、数值、actual kid和split-K完全一致。
每项先warmup 20次，再记录9轮、每轮100次launch的event时间，取每轮单次微秒数的median；每个
端点运行两遍，最终值为两遍median的平均。顺序固定为：

~~~text
Task1 A1 -> Task2 B1 -> Task2 B2 -> Task1 A2
~~~

除要求的high-level/explicit/family public口径外，还逐项记录：

- `compile_ops` raw：含Torch到`aiter_tensor_t`转换；
- direct pybind/C++：预先转换Tensor handle，隔离C++检查和launch；
- graph replay：capture后只重放device工作，用于核对kernel/codegen是否变化。

四轮各输出18条 `PERF_CASE`，全部数值断言通过，并各自明确输出gfx950 bpreshuffle不可执行记录。

### 21.2 日志

完整日志保留在：

~~~text
/tmp/aiter-b7.6-gfx950.qcNVOS
~~~

SHA-256：

~~~text
555d438785ee1676bbad380a8b94c9fb034f3ee4cd1e24e2294604133a488b21  perf_task1_A1.log
94f8906c56801dcd1d56dbc99bc32d256130746d7777d79d80088cebb4adb05d  perf_task1_A2.log
a1fd630d993cd958045116415ae53ad9085a0add5b23a31e989fa49f20d585e8  perf_task2_B1.log
c4383d3f7516b3ef778e91a71131fa3734947412392c713cebce15e93da13229  perf_task2_B2.log
4a59da3a088f9d3de80663f53e96fa126252023c6797bdb13b95d067e5f46171  adapter_task2.log
~~~

## 22. B7.6 gfx950结果与分层归因

### 22.1 要求的public/high-level口径

下表正值表示Task2更慢，负值表示Task2更快：

| case | Task1 | Task2 | Task2变化 | 结论 |
|---|---:|---:|---:|---|
| A16 high-level BF16 | `27.988 us` | `26.845 us` | `-4.083%` | 无回退 |
| A16 high-level FP32 | `27.961 us` | `26.336 us` | `-5.813%` | 无回退 |
| A16 explicit BF16 | `25.617 us` | `25.437 us` | `-0.703%` | 噪声内 |
| A16 explicit FP32 | `25.931 us` | `25.322 us` | `-2.350%` | 无回退；端点内漂移更大 |
| A8 no-scale public，kid 2 | `13.272 us` | `16.005 us` | `+20.591%` | 超出噪声，见22.3 |
| A8 plain blockscale public，kid 1 | `33.977 us` | `34.014 us` | `+0.107%` | 噪声内 |
| A8 blockscale-bpreshuffle | 未执行 | 未执行 | — | gfx950两端均无registered kernel |

A16两种dtype的端点内两遍漂移为high-level `0.477%--2.468%`、explicit
`1.733%--4.895%`；所有Task2变化均非正向回退。A8 blockscale两端漂移不超过`0.332%`。
no-scale Task1/Task2两遍漂移分别为`2.342%`和`1.712%`，因此`+20.591%`不是轮间噪声。

### 22.2 direct与graph证明kernel未变

| case | direct pybind/C++变化 | graph replay变化 |
|---|---:|---:|
| A16 kid 200 BF16 | `-0.205%` | `+0.078%` |
| A16 kid 200 FP32 | `-0.673%` | `+0.726%` |
| A8 no-scale kid 2 | `+0.114%` | `-0.087%` |
| A8 blockscale kid 1 | `+0.032%` | `-0.011%` |

direct/graph全部变化落在`-0.673%--+0.726%`，两端重复漂移最大`1.151%`。因此本轮约
`±1.2%`可作为device/C++层的实测噪声带；没有kernel或生成器性能回退证据。A8两条raw变化也
分别只有`-0.580%`和`-0.043%`。A16 raw显示Task2约`-8%`，但Task2两轮raw自身漂移达到
`5.779%--6.512%`，且两保留端点的`module_aiter_core.so`不是同一ELF，因此只将它记录为
“无回退”，不把它宣传成稳定优化。

### 22.3 no-scale回退精确位于Python adapter

Task2 no-scale同一模块内：

~~~text
public family wrapper: 16.005 us
compile_ops raw:        13.445 us
差值:                   2.560 us
~~~

而Task1/Task2 raw只差`-0.580%`，direct只差`+0.114%`，graph只差`-0.087%`。因此
`+2.733 us / +20.591%`的public端点回退不在C++检查、generated launcher或device kernel，
而在新public wrapper的Python安全解析。

对该wrapper的三个纯Python组成项另做20次warmup、9轮、每轮10000次的wall-clock median：

| Python检查 | median |
|---|---:|
| `_check_same_device(XQ,WQ,Y)` | `0.484 us` |
| `_device_arch(XQ.device)` | `1.403 us` |
| exact registry/kid/dtype查询 | `0.564 us` |
| 三项组合 | `2.524 us` |

组合值与public/raw实测差`2.560 us`一致。plain blockscale的device工作约34 us，Python检查可与
GPU队列重叠而没有形成可测回退；no-scale约13 us，新增host policy使stream出现约2.6 us空档。
这是一项已精确定位的Python adapter开销，不是kernel变化。B7.6没有为掩盖该结果恢复generic
生产入口，也没有删除family/kid/device安全检查或擅自引入arch cache；是否优化该adapter留给
后续明确决策。

## 23. B7.6硬件边界与B7总总结

- gfx942：本机没有gfx942，A16与kid 11000 bpreshuffle性能全部**未执行**；B7.2交叉编译和
  B7.3 skip不替代实机median。
- gfx1250：本机没有gfx1250，two-stage/fused A16性能全部**未执行**；该架构当前三条OPUS A8
  family也没有registered kernel，未伪造数值。
- gfx950 bpreshuffle：Task1旧tune只支持gfx942，Task2 typed table为空；两端都记录
  “no registered gfx950 kernel”，不把异常计时或其他backend计时冒充OPUS family性能。

B7.1--B7.6的要求均已实际执行或按缺失硬件/空capability明确记录未执行。静态、codegen、CPU、
gfx950 GPU、ABI和device-kernel性能层没有新增失败；gfx942/gfx1250实机边界保持开放。性能层
唯一超出噪声的结果是gfx950 no-scale public wrapper `+20.591%`（绝对`+2.733 us`），已由
raw/direct/graph和纯Python微基准共同定位到约`2.524 us`的Python adapter，不属于kernel、C++
检查或生成器回退。

B7.6只新增上述可复现benchmark并更新验收文档，没有修改生产源码，没有恢复prepared实验，
没有引入generic入口，没有reset/checkout/clean或清理dirty工作树，也未触碰无关文件
`docs/opus_gemm_next_performance_optimization_plan.md`。本日到此停止；后续若要消除该2.6 us
adapter开销，应从family public policy本身优化，不能回退到generic入口。

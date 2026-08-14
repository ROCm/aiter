# OPUS GEMM 任务二 Step B7及后续性能检查点

更新时间：`2026-08-12 02:59 UTC`

这是一份短上下文恢复文件。新会话继续后先完整读取本文件，再读取
`docs/opus_gemm_two_tasks_final_plan.md` 的“Step B7：静态、codegen、CPU、GPU、ABI和性能验收”。
不要重新执行已完成的 B7，不要从对话重推 B0--B6，也不要清理当前 dirty 工作树。

## 0. 2026-08-12 gfx950 A8 no-scale public adapter优化

本节是当前最高优先级状态，取代第1节中B7.6的“`+20.591% / +2.733 us`唯一public回退”作为
当前性能结论；第1节及第12节保留为2026-08-11历史基线。

### 0.1 最终保留改动

只修改A8 Python family adapter及其测试/文档，没有修改C++、generated launcher或device kernel：

- gfx arch按显式`torch.device`缓存；无index的`cuda`别名先按当前device归一化，不能把多GPU
  主机假设成同构单卡；
- 成功的exact `(arch, family, kid, Y.dtype)` capability查询使用`lru_cache`；异常不被缓存，
  因此失败能力不会冻结成陈旧negative；
- `_check_same_device`仍在每次public调用执行完整Tensor类型和同device检查，但合法热路径不再
  构造list/set，只有错误路径才构造友好诊断；
- Tensor、data pointer、stream和raw launcher均不缓存，C++/generated最终安全复核保持不变。

没有恢复Task1已回退的prepared/prevalidated workspace实验。另曾临时把device->arch与
arch->capability合成单一cache：纯Python微基准约省`0.275 us`，但`D -> E -> E -> D`端到端
对照中合并版public平均反而慢约`0.081 us`，已完整回退，最终源码中不存在该实验helper。

### 0.2 修改前/最终同端点对照

所有数据使用物理GPU 0（MI355X/gfx950）、相同Task2 B7.4 `.so`、相同输入和
`20 warmup + 9 rounds x 100 launches`。修改前两轮通过真实旧实现/显式关闭新增缓存得到，最终
两轮来自最终源码：

| case | 修改前 | 最终 | 变化 |
|---|---:|---:|---:|
| no-scale public | `15.776530 us` | `13.945425 us` | `-1.831105 us / -11.607%` |
| no-scale raw | `13.444915 us` | `13.197610 us` | `-1.839%`，设备/轮次漂移 |
| no-scale direct | `12.878705 us` | `12.891005 us` | `+0.096%` |
| no-scale graph | `13.745115 us` | `13.724220 us` | `-0.152%` |
| blockscale public | `33.939685 us` | `33.834131 us` | `-0.311%` |

用各自raw扣除底层漂移后，no-scale public/raw差从`2.331614 us`降至`0.747815 us`，追回
`1.583800 us / 67.927%`的adapter空档。direct/graph未变化，仍无kernel回退。

### 0.3 fresh Task1/最终Task2 ABBA

随后用原B7保留的Task1模块和同一Task2模块重新执行：

~~~text
Task1 T1 -> Task2 G1 -> Task2 G2 -> Task1 T2
~~~

| case | Task1 | 最终Task2 | Task2变化 |
|---|---:|---:|---:|
| no-scale public | `13.328800 us` | `13.973215 us` | `+0.644415 us / +4.835%` |
| no-scale raw | `13.594405 us` | `13.409010 us` | `-1.364%` |
| no-scale direct | `12.885595 us` | `12.879800 us` | `-0.045%` |
| no-scale graph | `13.717010 us` | `13.723810 us` | `+0.050%` |
| blockscale public | `33.806120 us` | `33.837121 us` | `+0.092%` |

原B7的`+2.733 us / +20.591%`已缩小为`+0.644 us / +4.835%`，但没有将剩余差异写成
“完全消失”：它仍略高于两端约1%的重复漂移，来自每次调用保留的动态Tensor安全检查和热cache
查询。若后续必须继续消除，需单独决策pre-resolved public合同或native边界，不能恢复无端到端
收益的A16 prepared workspace实验。

较长合法shape `[1,1024,1024,512]`的最终Task2实测为public `16.202641 us`、raw
`16.211430 us`，数值逐元素通过；该shape中host policy已被device工作完全覆盖。

### 0.4 验证和日志

最终验证：

~~~text
py_compile: passed
git diff --check: passed
cache/adapter定向: 4 passed, 2 warnings
interfaces完整: 65 passed, 8 skipped, 2 warnings
dispatch/workspace/graph/a16/interfaces: 239 passed, 22 skipped, 2 warnings
~~~

skip仍是缺失gfx942/gfx1250硬件项；两个warning仍是Cython `dep_util` deprecation。性能日志目录：

~~~text
/tmp/aiter-a8-arch-cache.99rNOu
~~~

正式日志SHA-256：

~~~text
2308eb20f6d9fb182048a9df6a9563fb62fae19f4faa7535a15106554090b992  arch_A1.log
71b199bde7a085eacca0253a061ab149079b2e26a003fa0041e122c832fd99f1  arch_A2.log
73e28b7fcbf7671a38d1715e21f7b75a913463d2fed1ccf7ab2b573da1a7454b  final_F1.log
39b66c9cab5087466be2909a76dbc6883558d6381ee39a5e1da0cb38103d2bba  final_F2.log
d9116fcee426906a4a63bdf8b7888e195093d802c47a68a87e72bc1c3f31ccf8  fresh_T1.log
38dfbcd67641e8729306c86e1bd8475e95b68168bdd9391adf522689a287c9ff  fresh_G1.log
ec217cc6e37cbd2decfce88427769496334bc14a07dbb543eb391ec5fcab0f0f  fresh_G2.log
1b47d6bbaf499e31ecbd8904c46f75e19bb6442ab0970d9d2b3ae1138c715bd7  fresh_T2.log
~~~

## 1. B7收尾时结论（2026-08-11历史基线）

- Task2 的 Step B0--B6 已实现，B7.1--B7.6 已执行完成。本日不再操作；明日只按用户新的
  明确指令继续，不自动重跑或修改生产代码。
- B7.6 的device kernel、direct C++和除一个case外的public性能均无回退。唯一超出噪声的是
  gfx950 no-scale kid 2 public wrapper：Task2相对Task1为`+20.591%`、绝对`+2.733 us`；已定位为
  `_check_same_device + _device_arch + registry`约`2.524 us`的Python adapter开销，不是kernel、
  C++检查或生成器变化。
- 当前没有 commit。工作树同时包含 Task1 和 Task2 B0--B7 的累计成果及未跟踪文档/测试；B7.6
  新增可复现benchmark `op_tests/bench_opus_task1_task2_interfaces.py`。
- 禁止 `git reset --hard`、`git checkout --`、`git clean` 或覆盖用户既有改动。
- 当前机器可实测 gfx950；gfx942/gfx1250 已完成生成和 HIP 交叉语法验证，但没有实机数值或
  性能结论。目标节点执行入口为 `docs/gfx942_gfx1250_validation_runbook.md`。
- B7 完整验收条目以总计划为准；无对应硬件时必须写“未执行”，不能把 compile 或 pytest skip
  写成实机通过。

## 2. 最重要的状态勘误：workspace 是 checked-only

当前源码与 `docs/task1_checkpoint.md` 在 `2026-08-11 13:23 UTC` 记录的最终回退一致：

~~~text
generated workspace row: { kid, func }
runtime: has_workspace(kid) + workspace_dispatch(kid)
launcher: 每次执行完整 checked validator
~~~

当前生产代码中不存在：

~~~text
prevalidated
PreparedWorkspace
OpusA16W16WorkspaceDispatch
workspace_try_dispatch
template <bool Validate>
~~~

Task1 曾实验 `{kid, checked, prevalidated}`、POD-only thread-local matcher和单次表查询；该实验
证明 C++ 层局部可提速，但正常 Torch raw/graph 端到端无可测收益，随后按用户决定回退。
`docs/task2_step_b6_process.md` 和 `docs/task2_step_b6_detail.md` 中描述“恢复 prepared”的正文是
执行历史，不是当前源码合同；两文件顶部已经添加勘误。

**不要在 B7 为了匹配旧 B6 文字恢复 prepared 路径。** B7 应验收当前 checked-only 基线。

## 3. B0--B6 已落地的最终架构

四个 canonical launch API 已在 Python、C++、pybind 对齐：

~~~text
opus_gemm_a16w16_launch
opus_gemm_a8w8_launch
opus_gemm_a8w8_blockscale_launch
opus_gemm_a8w8_blockscale_bpreshuffle_launch
~~~

核心合同：

- A16 explicit/tuned/heuristic 共用 Python selector，并在 launch 前消费 gfx942 redirect 后的
  `actual_kid`；保留 2D/3D normalize、padded leading stride、bias 和 Task1 workspace规则。
- runtime shape policy 和 heuristic 只在 Python；C++ 只按当前 device、logical family、Y dtype和
  exact kid进入 generated typed table。
- build-time CSV/sidecar仍参与 subset compile，但不再生成或驱动 C++ runtime shape lookup。
- gfx950 no-scale kid 2、gfx950 plain-WQ blockscale kid 1、gfx942 blockscale-bpreshuffle
  kid 11000均来自 canonical registry和独立 typed table。
- gfx950/gfx1250 blockscale-bpreshuffle公共 Python/pybind/C++符号存在，但当前表为空；显式 kid
  不能借用 gfx942 的 11000。
- generated launcher拥有架构/实例特有的 dtype、shape、stride、layout、tile和prefetch校验；
  公共 bpreshuffle router不硬编码 gfx942 的 BF16、batch=1或128分组规则。
- gfx950/gfx1250同名高层 AITER API继续选择 CK/CKTile/ASM/FlyDSL/Triton/Gluon，不会因空
  OPUS family被错误导向 OPUS raw。
- bpreshuffle 是 WQ 内容语义；API和README要求真实 shuffle后的weight，shape检查不证明内容布局。

当前 capability：

| Logical family | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| A16W16 | registry kids | registry kids | registry kids，含 fused |
| A8W8 no-scale | 无 | kid 2，Y FP32 | 无 |
| A8W8 blockscale plain-WQ | 无 | kid 1，Y FP32 | 无 |
| A8W8 blockscale bpreshuffle | kid 11000，Y BF16 | 空表 | 空表 |

## 4. 已删除和允许保留的旧名字

生产 C++/pybind中已经删除：

- generic `opus_gemm()`、`OPUS_GEMM_PYBIND`和 `_opus_gemm_bf16_dispatch`；
- C++ `opus_gemm_a16w16_tune`及旧 gfx942 A8 tune符号；
- `opus_gemm_lookup.h` runtime shape lookup、三个 heuristic header和shape selector；
- 旧 tune/lookup generated表和兼容宏。

允许保留：

- Python `opus_gemm_a16w16_tune`和
  `opus_gemm_a8w8_blockscale_bpreshuffle_tune` deprecated wrapper；
- 兼容测试、迁移文档和 tuner CSV字段中的 `kernelId`/`splitK`/tuned语义；
- `gen_instances.py` 中三个旧 generated文件名字符串，仅用于删除复用 blob里的陈旧产物；
- `op_tests/bench_opus_gfx950_workspace_ab.py` baseline分支对旧
  `_opus_gemm_a16w16_tune_raw` 的调用。它是 Task1 旧/新端点 A/B 工具，不是生产调用方。

## 5. B6 最终验证快照

最终四文件回归（当前 checked-only源码）：

~~~text
pytest -q \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_interfaces.py

215 passed, 16 skipped, 2 warnings
~~~

gfx950 fresh JIT定向实机：

~~~text
finish build [module_deepgemm_opus], cost 8.3s
10 passed, 48 deselected, 2 warnings
~~~

该组覆盖 plain-WQ blockscale kid 1数值和9类 A8负例。Task1 更完整的 gfx950 140-kid、
48 workspace kid及性能证据见 `docs/task1_checkpoint.md` 和 `docs/task1_detail.md`。

其他已通过项：

- `git diff --check`；
- B6目标 Python文件 `py_compile`；
- 三架构 fresh codegen到 `/tmp/aiter-b6-multiarch.p6dipk`；
- gfx942/gfx950/gfx1250三个 generated host TU、multiarch `opus_gemm.cu`和
  `opus_gemm_pybind.cu` 的 HIP syntax；
- 旧 C++ tune/raw、generic entry、runtime lookup和heuristic残留扫描。

上述结果是 B6快照，不替代 B7要求的全套重新验收。

## 6. B7.1 最终验证快照

`2026-08-11 13:31 UTC` 以当前 checked-only 源码重新执行：

~~~text
git diff --check: passed
四个目标 Python 文件 py_compile: passed
OPUS_GEMM_PYBIND / _opus_gemm_bf16_dispatch: zero matches
生产 C++/pybind runtime lookup / selector / old tune names: zero matches
checked-only guard: zero matches
~~~

总计划中的后两条原样 `rg` 会分别命中 `gen_instances.py` 的
`opus_gemm_lookup.h` 和 `opus_gemm_a16w16_tune_lookup.h`。它们属于第 4 节已允许的三项陈旧
生成文件清理白名单；限定生产范围复扫后均为空。没有删除清理逻辑或拆分字符串来规避扫描。
完整命令、退出码和分类见 `docs/task2_step_b7_process.md`。

B7.1 没有修改生产源码或测试，没有恢复 prepared 实验，也没有清理 dirty 工作树。

## 7. B7.2 最终验证快照

完整生成和对象位于 `/tmp/aiter-b7.2.km7yQp`。对 gfx942、gfx950、gfx1250 各用全 canonical
kid sidecar执行两次 fresh CLI codegen：

~~~text
byte stability:       3/3 arch passed
single-arch header:   3/3 arch passed
B0 A16 set/digest:    9/9 tables passed
A8 typed placement:   kid 1 / 2 / 11000 passed
empty bpreshuffle:    gfx950/gfx1250 BF16+FP32 passed
legacy lookup absent: 6/6 fresh dirs passed
generated TU syntax:  3516/3516 passed
generated TU -O3 -c:  3516/3516 objects passed
~~~

gfx1250 保持 496 two-stage 和 1378 fused；two-stage workspace 全 BF16，fused 为 BF16 780、
FP32 598。新增 synthetic fixture 分别证明 gfx950 BF16、gfx1250 FP32 表可只通过临时
registry/tag/emitter数据从0变1，公共 C++/pybind/Python 文件不变；该项 `2 passed`，合并定向
codegen golden 为 `9 passed, 2 warnings`。fixture 不是实机 kernel capability 声明。

B7.2 没有修改生产源码，只新增可重复 fixture 和验收文档。执行期间外部新增的未跟踪文件
`docs/opus_gemm_next_performance_optimization_plan.md` 已保留且未触碰。完整命令、digest、数量和
对象证据见 `docs/task2_step_b7_process.md`。

## 8. Step B7 完成状态

总计划中的 B7 六部分均已执行并逐项记录：

1. **静态检查（已完成）**：diff、py_compile和三组旧符号扫描均通过；
2. **codegen检查（已完成）**：三架构双 fresh生成、字节稳定性、arch header、typed表、空表、
   synthetic registry注入、全部生成 TU编译和 gfx1250数量分布均通过；
3. **CPU测试（已完成）**：interfaces、dispatch、workspace、gemm_codegen最终为
   `224 passed, 12 skipped, 2 warnings`；12个skip均为gfx942/gfx1250硬件项；
4. **GPU测试（已完成可用硬件范围）**：gfx950数值、负例、graph、双stream、48 workspace和
   140-kid全量已通过；gfx942/gfx1250因无对应硬件明确未执行；
5. **ABI/符号检查（已完成）**：三架构独立模块的四个raw、bpreshuffle一致导出、旧符号缺失、
   compat wrapper、unsupported top-level import和fake schema顺序均通过；
6. **性能检查（已完成）**：相同actual kid/split-K的Task1/Task2 ABBA已覆盖A16 high-level、
   explicit和gfx950三条A8 family状态；raw/direct/graph证明kernel不变，唯一no-scale public回退
   已精确定位到Python adapter，未恢复generic入口。

明日若继续任何后续工作，仍先做：

~~~bash
cd /root/workspace/0810/aiter
git status --short
git diff --check
rg -n 'prevalidated|PreparedWorkspace|OpusA16W16WorkspaceDispatch|workspace_try_dispatch' \
  csrc/opus_gemm op_tests
~~~

第三条在当前源码应为空。B7已经执行完毕，不应把这些guard解释为需要自动重跑B7。若文件时间
突然变化或测试合同与本 checkpoint不一致，先检查是否仍有
另一个 Codex/pytest/codegen进程在写同一工作树；不要直接覆盖、kill、reset或checkout。

## 9. B7.3 最终验证快照

计划指定的四文件命令最终结果：

~~~text
224 passed, 12 skipped, 2 warnings in 4.31s
~~~

覆盖审计后新增/增强了四架构包导出与unsupported stub、bpreshuffle explicit优先、真实CLI
subset compile公式，以及三架构generated table类型断言。12个skip是gfx942 A8数值/负例、
gfx942 typed workspace和gfx1250 typed workspace/batch raw实机项；均未写成通过。两个warning是
Cython `dep_util` deprecation。

B7.3 没有修改生产源码。`git diff --check`、四测试文件`py_compile`和checked-only guard均通过；
完整覆盖映射和skip分类见 `docs/task2_step_b7_process.md` 第9--11节。

## 10. B7.4 最终验证快照

当前节点为 8 张 MI355X/gfx950，没有 gfx942/gfx1250。以当前 checked-only源码 fresh构建：

~~~text
/tmp/aiter-b7.4-gfx950.U4cvOE
final sidecar: 140 A16 + mandatory A8 kid 1/2 = 142
sidecar SHA-256: b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7
~~~

gfx950实机结果：

~~~text
A8定向数值/负例/路由/graph/stream: 23 passed
focused dispatch/workspace/graph/a16/interfaces: 236 passed, 22 skipped, 0 failed
external-workspace全集合: 48/48 passed
canonical A16全集合: 140/140 passed（四 shard各35）
历史 mono FP32 1400--1404、6400--6404: 10/10 passed，单独记录
~~~

新增测试覆盖 kid 1/2 shape/dtype/device/single-scale/unknown-kid合同，CK/CKTile/ASM/Triton
高层路由不进入OPUS bpreshuffle，public `kid=None`和private raw均报告gfx950空 capability，以及
workspace + 两个A8 family的真实 graph replay和双 stream。生产源码未修改。

gfx942和gfx1250的总计划 B7.4项目因无对应架构硬件全部明确为“未执行”；B7.2 compile、B7.3
CPU fixture和 pytest skip都没有被写成实机通过。完整命令、分类和边界见
`docs/task2_step_b7_process.md` 第12--16节。

## 11. B7.5 最终验证快照

三架构 fresh ABI subset模块位于：

~~~text
/tmp/aiter-b7.5-abi.oaplCv/{gfx942,gfx950,gfx1250}/module_deepgemm_opus.so
~~~

每个 ELF只包含对应 offload arch。三份动态模块除通用 stream辅助属性外，以 `opus_` 开头的业务
pybind属性都严格等于四个 canonical launch；bpreshuffle C++、private raw和public名字三架构一致
存在。generic `opus_gemm`、`_opus_gemm_bf16_dispatch`以及两个旧 C++ tune均不存在；两条 public
deprecated Python wrapper仍导出。

四条 raw的 pybind doc、Python源码定义、fake generator和Torch schema参数顺序逐项一致；A8
内部 `dummy`不属于ABI。`GPU_ARCHS=gfx90a`的真实 top-level subprocess证明unsupported OPUS
只安装调用时stub，`aiter`中位于其后的 rmsnorm/topk/fused update/mla以及
`from aiter import *`均继续成功。

回归结果：

~~~text
ABI/signature/compat定向: 12 passed
interfaces完整:          63 passed, 8 skipped
B7四文件综合:            231 passed, 12 skipped
~~~

B7.5只增强 `op_tests/test_opus_interfaces.py` 和验收文档，没有修改生产源码。完整构建信息、ELF
SHA-256、动态属性集合和schema顺序见 `docs/task2_step_b7_process.md` 第17--20节。

## 12. B7.6 最终验证快照

正式性能日志目录：

~~~text
/tmp/aiter-b7.6-gfx950.qcNVOS
~~~

使用Task1 B0 ABI模块 `/tmp/aiter-gfx950-current.NtJydE` 和Task2 B7.4模块
`/tmp/aiter-b7.4-gfx950.U4cvOE`；两端sidecar均为相同142 kids、SHA-256均为
`b43395710e4d99e2e4ed5807dc495a6312e435b056d5f475d088496ff830bdf7`。没有重编译端点。
正式顺序为 `Task1 A1 -> Task2 B1 -> Task2 B2 -> Task1 A2`，每case执行20次warmup、9轮、
每轮100次；四轮各18条case全部数值通过。

要求的public/high-level结果：

| case | Task1 | Task2 | 变化 |
|---|---:|---:|---:|
| A16 high-level BF16 / FP32 | `27.988 / 27.961 us` | `26.845 / 26.336 us` | `-4.083% / -5.813%` |
| A16 explicit BF16 / FP32 | `25.617 / 25.931 us` | `25.437 / 25.322 us` | `-0.703% / -2.350%` |
| A8 no-scale kid 2 public | `13.272 us` | `16.005 us` | `+20.591%` |
| A8 blockscale kid 1 public | `33.977 us` | `34.014 us` | `+0.107%` |
| gfx950 bpreshuffle | 未执行 | 未执行 | 两端均无registered kernel |

A16与两条A8 family的direct/graph变化全部在`-0.673%--+0.726%`，实测device/C++噪声带约
`±1.2%`，因此kernel没有回退。no-scale Task2 public/raw差`2.560 us`；纯Python组合微基准为
`2.524 us`，完成adapter归因。没有为消除该结果改回generic入口或修改生产源码。

gfx942和gfx1250无对应硬件，性能全部明确为“未执行”。完整方法、逐case median、两遍漂移、
日志SHA-256和归因见 `docs/task2_step_b7_process.md` 第21--23节。

B7.6后8张卡均为0%利用率、0%显存且无KFD PID；没有残留benchmark/pytest/JIT进程。新增benchmark
SHA-256为`22c44e87dfe7c3453e7da5eeeb7c2dcdda96628451cc8c3274a0dca43a34fb4b`。本日到此停止。

## 13. 续接文件索引

- 当前短检查点：`docs/task2_checkpoint.md`
- B7完整验收清单：`docs/opus_gemm_two_tasks_final_plan.md` 的 Step B7
- B7增量验收记录：`docs/task2_step_b7_process.md`
- B6执行记录：`docs/task2_step_b6_process.md`
- B6最终合同：`docs/task2_step_b6_detail.md`
- Task1当前权威状态：`docs/task1_checkpoint.md`
- Task1逐文件与GPU/性能证据：`docs/task1_detail.md`
- Python用户文档：`aiter/ops/opus/README.md`
- C++/codegen文档：`csrc/opus_gemm/README.md`

新会话可直接使用以下提示：

~~~text
读取 /root/workspace/0810/aiter/docs/task2_checkpoint.md 和
/root/workspace/0810/aiter/docs/opus_gemm_two_tasks_final_plan.md 的 Step B7，
B7.1--B7.6已经执行完毕；以当前 checked-only workspace等待并执行用户下一条明确指令。
不要恢复 prepared实验，不要清理dirty工作树；若后续优化no-scale public adapter，不得改回
generic入口。
~~~

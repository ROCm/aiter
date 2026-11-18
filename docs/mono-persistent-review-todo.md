# A03／A09：MI400 复核补测与性能修复 TODO

给 MI400 机器上的 agent：请执行本清单，补齐正确性与真实 JIT 证据，定位 standalone P256 回退，再验证最小修改。不要只返回计划。当前任务状态为**待执行**；下列候选优化均未实施、未测收益。

**当前优先只排查 A03 的改造是否引入回退：先执行§2.3.1的原 native 源码 P256 对照。A03 的 C1／C2、跨迭代预取和其他 P 扫描暂不先行实施；取得该对照后再决定。** 原 native 每 TG 只做一行，没有下一迭代需要预取，不能把缺少跨迭代预取直接定为回退原因。A09清单保留，当前先处理A03。

## 0. 版本、范围与启动

| 项目 | 固定值 |
| --- | --- |
| 仓库／分支 | `https://github.com/ROCm/aiter`，`asm_mon_dev`；以公开上游为唯一执行与回传入口 |
| 本次 review 的结果版本 | `a8a913883d7be9837c2f46f2e92de8fc701bac26` |
| 上轮实际 GPU 测试代码 | `d5930aff35d67fa0a2e74ba425909747f3ccef9f`；相对 `efa58c2` 只修复 Triton-only import |
| 原验收指南 | [mono-persistent-validation.md](mono-persistent-validation.md) |
| 上轮报告／原始数据 | [2026-11-18-gfx1250](mono-results/2026-11-18-gfx1250/README.md) |
| 上轮完整产物 | `/home/weizhu15/workspace/mono-acceptance-20261118`，同级 `mono-acceptance-20261118-results.tar.gz` |
| 上轮归档 SHA256 | `ec38c863944cc59bef4980f5409b3ba05aac8c33defde8bc89b0e9b9dd40808a` |

- [ ] 读取适用的 `AGENTS.md`、`aiter/ops/triton/README.md`、`op_tests/triton_tests/README.md`、`.github/instructions/aiter-ops-triton.instructions.md` 和原指南。该版本没有根目录 `AGENTS.md`；适用入口为 `aiter/ops/triton/AGENTS.md`。
- [ ] 使用独立 checkout 和既有可用 ROCm 环境，保留其他任务、框架版本、时钟和功耗设置。记录 GPU UUID／ordinal、gfx1250／wave32、镜像、驱动、PyTorch、Triton、代码 SHA、dirty diff、环境变量及采样时间。
- [ ] 拉取最新 `asm_mon_dev`，将本 TODO 保存在 checkout 外；基线使用 `a8a913883`，其中已含 import 修复。**本轮不要机械执行旧指南中的 `git checkout efa58c2`**。从该基线建立本轮修复分支；待发布时再与最新远端合并，不 reset 或 force-push。

```bash
set -euo pipefail
git clone --branch asm_mon_dev git@github.com:ROCm/aiter.git aiter-mono-followup
cd aiter-mono-followup
if test -f docs/mono-persistent-review-todo.md; then
  cp docs/mono-persistent-review-todo.md ../mono-persistent-review-todo.md
fi
git switch -c mono-p256-followup a8a913883d7be9837c2f46f2e92de8fc701bac26
export AITER_TRITON_ONLY=1
export TRITON_DEFAULT_FP_FUSION=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MONO_RESULTS="$PWD/../mono-followup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$MONO_RESULTS"
git rev-parse HEAD > "$MONO_RESULTS/base-commit.txt"
```

Python 使用机器上匹配 ROCm 的既有环境；如需隔离，按原指南创建继承 system packages 的 venv。以上命令不安装或升级框架。

若本 TODO 尚未进入上游，先将收到的 MD 保存到 checkout 外，再按正文执行；收到的文档补丁可在发布分支用 `git am <补丁路径>` 应用。基线测试仍固定上述代码，不依赖文档提交是否已合入。

范围固定为 standalone P256，正式 PS 仍为 16×16 TG、每 TG 128 threads。冻结 native、两种 scale 公式、零容差及原测试均保留。完整 mono 的 DEV／HT、s104/s105、跨算子 join、设备函数 ABI 和模型接入不在本轮实施。

## 1. 正确性／功能是否都 OK：已有结论与补测

**已确认：原 36 项 UT 全通过，无 skip；目标 rows=96 的两个算子通过。还不能声称所有性能 shape 均有完整数值验收。** A09 `[513,2048]` 有 1026 个 task，最忙 worker 执行 5 轮；原 shape UT 最大相关配置为 `[513,1024]`，Graph 配置为 `[257,2048]`，两者最多 3 轮。Benchmark 本身不校验输出。

### 1.1 保留原门槛，补齐精确 shape

- [ ] 在基线先运行原 36 项 UT 和冻结 native hash／12 项 compile 检查，保存 XML、日志及真实 JIT。原 36 项必须全通过且无 skip；不能只重新收集测试。
- [ ] 修改 `op_tests/triton_tests/quant/test_mono_persistent.py::test_output_quant_native_persistent` 的 shape 参数，增加 `(257,2048)`、`(513,2048)`。保留原 case；每个新增 case 同时检查 frozen native、非 PS port、PS、独立 PyTorch FP8／E8M0／QAT oracle、输入未写、输出 guard 和 XOR `0xFF` 预填充。
- [ ] 给 `test_graph_replay_changes_inputs` 增加 A09 `[513,2048]`，保留现有 A03 和 A09 case；至少 20 次改变输入再 replay，逐次比较全部输出。验证新值确实改变参考输出，避免只重复同一输入。
- [ ] 第 2 节边界性能测试之前，对所有新增 shape 先做同样的 native／PS／oracle／guard 数值检查。可参数化上述测试；不要把只跑过的 shape 写成校验过。
- [ ] 保存 `baseline-36.xml` 和 `expanded.xml` 两份结果。扩展后的总数可大于 36；报告原 36 项均保留，以及新增 node ID、数量和结果，不再用 `len(cases)==36` 拒绝合法扩展。
- [ ] 任一失败先保存 node ID、seed、shape／stride、输入／权重、native／PS 输出、代码 SHA／diff 和完整日志，再修复候选；修复后重跑失败 case、原 36 项及全部新增项。禁止改冻结 native、降低数值标准或删除失败 case。

原指南的 compile／pytest 命令可直接沿用；基线、每个候选分别使用独立 `TRITON_CACHE_DIR`／`TRITON_DUMP_DIR`，防止混淆产物。

### 1.2 功能结论的边界

- [ ] 保留 A03 同一 `[96,1792]` BF16 storage 的 Q1280／KV512 view、row stride=1792；eps_q／eps_k、floor 与输出布局保持。
- [ ] A03 Q 以冻结 native 逐字节比较，KV 额外以原 FP64 oracle 和既定 BF16 容差检查；A09 独立 oracle 与 QAT 均保留。不要将两者表述成相同 oracle 覆盖。
- [ ] 单独记录 `tests/test_import_without_gpu.py`：上轮 ROCm 对 `ROCR_VISIBLE_DEVICES=-1` 在 `torch.cuda.is_available()` 阶段 abort，空 visibility 的 import smoke 通过。这不构成算子数值失败，也不能写成全仓库测试通过；如修测试，仅单独提交兼容修改并保留禁 GPU 的意图。
- [ ] 若当前环境有兼容 GPU memory checker，运行目标／尾 task／5 轮 case 并保存日志；没有则写明 sanitizer 未验证，不能以 guard 通过替代。不要为此更换机器的工作运行时。

## 2. 为什么下降、什么条件下降：取证与区分实验

### 2.1 当前已确认到哪一步

下表为上轮 final matrix 的 median；耗时单位 µs，变化为 `PS/native - 1`。rows=96 四项按原判据均为 inconclusive；rows=257／513 八项均回退，三次完整矩阵方向一致。

| Op | rows | hot native → PS | hot 变化 | rotating native → PS | rotating 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A03 | 96 | 1.579 → 1.564 | −0.93% | 1.912 → 1.898 | −0.75% |
| A03 | 257 | 1.692 → 2.065 | +22.06% | 2.054 → 2.478 | +20.65% |
| A03 | 513 | 1.838 → 2.690 | +46.33% | 2.274 → 3.121 | +37.24% |
| A09 | 96 | 1.412 → 1.452 | +2.81% | 1.714 → 1.713 | −0.05% |
| A09 | 257 | 1.512 → 2.064 | +36.49% | 2.038 → 2.445 | +19.99% |
| A09 | 513 | 1.668 → 3.078 | +84.48% | 2.754 → 3.648 | +32.47% |

源码确定的分配是 `task = worker + k*256`。A03 每行一个 task；A09 每 1024 元素一个 task，K=2048 时每行两个 task。以 A09 rows=513 为例：1026 task，254 个 worker 各做 4 个，2 个 worker 各做 5 个；native 发出 1026 个独立 TG。PS 将后续 task 放进同一 worker 的循环，改变了独立任务的供给和可重叠程度；具体 load／wait 重叠程度仍需真实 ISA 与测量。

**先验证“同样的 task 切分，为何不能保持性能”，不要将循环轮数直接当作回退原因。** A03 的整个 `tl.range` 循环 AST 与冻结 native 相同，原 wrapper 发 M 个 TG、步长 M，每 TG 实际只做一行；PS 改为 256 TG、步长 256。A09 原来无 task loop，每 TG 做一个 `[32 groups,32 elements]` tile，现在把相同逻辑 tile 放进循环。两者 native／PS 每 TG 均为 128 threads，未增加跨 TG barrier／flag／poll，也未把一个 task 拆小。

| M=513 时的映射 | Native | PS |
| --- | --- | --- |
| A03 | TG r 同时负责第 r 行的 Q1280 和 KV512 | worker 0 做 row 0／256／512；worker 1 做 1／257 |
| A09，K=2048 | TG 2r 做 row r 的前1024元素，TG 2r+1 做后1024元素 | worker 0 做 row 0／128／256／384／512 的前半；worker 1 做相同行的后半 |

96 行时，两者所有有效 worker 都只有一个 task，不能用多轮串行解释该配置。task 内访问地址与逻辑布局不变；A03 worker 下一行相差 896 KiB、A09 下一 task 相差 512 KiB，是跨 task 步长，不能据此称为 task 内访存不合并。A03 原本每 TG 都读权重，PS 每轮读权重不自动构成新增总读取量。

如果 native 也只能以等效 256 个独立 task 同时推进，则1026个task同样需要至少5批；在每 task 指令、等待重叠和调度供给相同的理想条件下，PS 没有必然变慢的理由。需核实实际差异是否为：native 有更多可同时推进的 wave／请求；PS 下一 task load 未重叠；loop／寄存器／grid 形状改变编译或调度。不能仅按 TG 总数推断实际驻留数，也不能把上一 task 发出 store 等同于等待其全局写回后才开始下一 task。

| 条件 | A03 | A09，K=2048 |
| --- | --- | --- |
| task 数 N | M | 2M |
| 首次出现第二轮 | M=257 | M=129 |
| rows=96 活跃／空 worker | 96／160 | 192／64 |
| rows=257 最长 task 链 | 2 | 3 |
| rows=513 最长 task 链 | 3 | 5 |

一般最长链为 `ceil(N/256)`；N 除以 256 余数很小时，最后一轮只有少数 worker 工作。这个轮数是当前不拆 task 的下界，简单换 worker 编号或动态抢 task 不能消除。**跨过 256 task 是调度发生变化的边界，不是已实测证明的通用回退阈值。** N≤256 也可能有循环、空 TG 或编译开销；rows=96 的 A09 hot 三轮 median 均小幅偏慢，需要进一步判定。

真实 JIT 汇总显示 A03 native／PS 均 83 VGPR、2 个条件分支；A09 为 47／51 VGPR、0／2 个条件分支，全部 scratch=0。不能把 A03 回退归因于新增分支，也不能把 A09 的全部回退归因于 4 个寄存器或 spill。汇总不是完整 ISA；目前没有逐项损失占比证据。

### 2.2 先把真实产物带回可 review 的位置

- [ ] 校验上轮归档 SHA256，依据 `target-resources.json` 的 JIT hash 提取两算子、rows=96／257／513、native／PS 的 `.source/.ttir/.ttgir/.llir/.amdgcn/.hsaco/.json`。相同产物可去重，但保留 12 个 launch 到产物的映射、shape／stride／编译参数和 SHA256。
- [ ] 将原始 `target-launches.csv`、提取脚本、精简产物包和 manifest 提交到本轮结果目录。必须让无 SSH 的 reviewer 能通过分支获得真实 ISA、IR 和 metadata；code object 也保留。若文件超出仓库限制，压缩去重后提供同账号可读取的下载地址及 hash，不能只给主机绝对路径。排除凭据和无关机器数据。
- [ ] 输出 native／PS 的带行号 ISA 对照：入口检查、当前／下一 task 的 load、首次依赖 wait、reduce／量化、store、循环回边。说明每次循环实际执行的指令；静态分支条数不等于动态执行次数。
- [ ] A03：确认已有 `tl.range(..., num_stages=2)` 是否实际形成跨行预取；检查 Q／KV activation 与 weight load 位于循环内外、等待位置和 live range。没有重叠时说明是何种依赖／编译结果，不能只建议再加 `num_stages=2`。
- [ ] A09：核对 native、`persistent=False` port、PS 的实际 body、指针更新、bounds check 和循环；将数值 body 差异与循环适配差异分开。记录 code-object VGPR／SGPR／scratch／LDS，不直接混用 profiler 的 `VGPR_Count` 数字；如解释差异，给工具字段单位依据。
- [ ] Profiler 检查正式 PS 实际 16×16 TG、128 threads/TG。grid 正确不能证明全部 256 TG 同时驻留；有可用 occupancy／wave／memory stall counters 时单独采样，写清计数器含义和窗口。不支持则保留未验证状态，不伪造周期归因。

### 2.3 边界矩阵和单变量对照

#### 2.3.1 A03：先用未修改的 native kernel 直接做 P256

A03 冻结 kernel 已有 `tl.range(row_start, M, NUM_PRGMS, num_stages=2)`，无需添加新循环就能执行 P256。先只增加独立的验收 wrapper，调用冻结的 `_fused_dual_rmsnorm_mxfp8_quant_kernel`；测试／benchmark经该wrapper调用。**不修改 `mono_native.py`，不复制或改写计算体。** 下表中的 M 始终是实际行数，不能为凑 grid 改成256。

| 对照 | 使用的kernel源码 | grid | NUM_PRGMS／row_start |
| --- | --- | --- | --- |
| A：原native | 冻结native | `(M,)` | M／pid0 |
| B：原native直接P256 | 与A相同的冻结native | `(256,)` | 256／pid0 |
| C：当前PS | 当前候选，`persistent=True` | `(16,16)` | 256／pid0+16*pid1 |

- [ ] 三者固定相同 Q／KV view、权重、eps、floor、dtype、stride、4 wave32、输入pool，以及在capture前分配并保留的同一套输出pool。逐项记录实际kernel参数及JIT specialization；`NUM_PRGMS`数值／整除属性可能改变生成代码，不能仅因源码相同就宣称binary相同。
- [ ] 先校验 rows=96／256／257／512／513，native逐字节比较、guard／输入未写均通过；再用同一Graph方法，hot／rotating各15轮，按轮轮换A／B／C顺序。保留原正式benchmark结果，三者的诊断时间单列，不混用不同输出pool口径。
- [ ] 若B与A相当、C较慢，回退位于当前适配路径；对照B／C真实ISA、JIT参数及1D／2D grid。进一步用当前候选的 `PERSISTENT=False`、1D grid=256／步长256作为D，区分源码迁入／编译与二维ID适配。先修该差异，不用预取掩盖。
- [ ] 若B和C都比A慢，说明保留原源码、只限为P256也可复现；先比较A／B真实代码及运行时参数，再区分TG调度供给和loop执行成本，不能直接写“原native也一样，所以无问题”或“PS必然更慢”。
- [ ] 若三者在统一buffer的诊断中相当，检查原benchmark的输出分配／地址／Graph节点和JIT差异，复现原回退后再归因。当前Python接口检查发生在capture阶段，不能把它直接加到Graph replay耗时。

完成上述对照后，按结果选择以下边界与优化任务；A03尚未完成本对照时，不以新增跨迭代预取替代排查。

- [ ] 正确性先通过，再用原 benchmark 测以下共同 rows，hot／rotating 各 15 轮：`96 127 128 129 255 256 257 511 512 513`。它们覆盖 A09 第一轮边界、两算子的整轮／少量尾 task；A03 在 127～129 作为未跨轮对照。

```bash
python -m op_tests.op_benchmarks.triton.bench_mono_persistent \
  --rows 96 127 128 129 255 256 257 511 512 513 \
  --cache both --trials 15 --output "$MONO_RESULTS/boundary-baseline.json" \
  2>&1 | tee "$MONO_RESULTS/boundary-baseline.log"
```

- [ ] 另加诊断对照 provider：A09 `quantize_fp8_group32(..., persistent=False)`，与冻结 native 及 PS 使用同 pool／seed／Graph。先校验再计时；三 provider 按轮轮换顺序，并保留原两 provider 的正式结果。用于判断迁入 kernel 的通用循环／编译成本与 P256 多轮成本，不能仅凭 native-vs-PS 一组归因。
- [ ] rows=96 优先比较同一数值 body 的三条路径：native 原 grid、P256 一次执行加有效 task guard、P256 当前循环。一次执行路径只用于 N<=256。这样分别检查空 TG／二维 grid 成本和循环成本；真实 TTGIR 核对 lane layout，不能由源码 tile 相同推断机器布局必然相同。
- [ ] 在独立诊断分支对 rows=513 比较同一循环 body 的 P=256／512／1024／2048；仍为4 wave32，grid=`(16,P/16)`、worker=`x+16*y`、task stride=P，保持完整工作量。诊断 wrapper 与结果明确标 P，先检查每 task 恰好一次、byte equality／guard，再测量；另以 P256 的 `(256,)` grid 对照 `(16,16)`，前者 worker 直接取 pid0。正式 `persistent=True` 仍固定 P256，其他 P 的时间不能混入正式验收。若增加 P 降低耗时，只能证明 worker 数／迭代链组合影响性能，仍需 ISA／counter 区分供给与循环流水。
- [ ] 必须保留每轮样本及执行顺序，报告 median、P10、P90、绝对差值和比值；记录 case 起止、GPU 负载及采样间隔、时钟。计时不启 profiler；hot／rotating 只是输入复用方式，不能直接称为已测 cache 命中／未命中。
- [ ] 比较 A03 256→257、512→513 与 A09 128→129、256→257、512→513 的 native／PS 各自变化。如果整轮配置也回退，则尾轮只能解释部分损失；如果 port 已有相同小幅成本，区分其与多轮 PS 的额外成本。A09 rotating 的回退比例较小也要列绝对 µs，不能据此认定问题消失。
- [ ] 对决定优化取舍且接近波动的项再做两次完整相关矩阵；不从多次结果拼最好值。未稳定分开的项继续标 inconclusive。

## 3. 有什么方案解决：按证据实施有限候选

当前优先级：A03原native源码P256对照（§2.3.1）→定位改造／编译／计时差异并修复→必要时再选C1或C2。其余正确性缺口保留；不得把预取作为恢复原性能的先决条件。每个候选独立 commit／diff，只改候选代码，保留基线可重跑；目标是解决 standalone 成本，不进入融合。收益全部以实测为准。

| 候选 | 具体修改对象与方法 | 作用与代价 | 验收 |
| --- | --- | --- | --- |
| C1：单 task 路径 | A09 `aiter/ops/triton/quant/fp8_group32.py` wrapper 将 `N<=256` 作为 constexpr 传入 `_triton_kernels/quant/fp8_group32.py`；该 specialization 用 uniform `task<N` guard 后执行一次原 body，不进入 task loop。A03 在对应 wrapper／kernel 对 M<=256 做同类 specialization，保留原 Q／KV 数学顺序。仅在 ISA 仍有可去掉的循环成本时保留修改 | 针对 rows=96 的循环状态与回边；仍发 256 TG，无效 worker 不 load/store。不解决 N>256 的多轮链；多一个 JIT specialization | rows=96 与边界测试、零尺寸／尾 task、Graph；核对实际分支与资源是否减少，并以 standalone 时间决定是否保留 |
| C2：重叠相邻迭代的 activation load | 在两个 `_triton_kernels/quant/` 文件中保留每 task 的计算／存储 body：prologue load task t；循环中先对 t+256 发 masked load，再计算／存储 t，然后轮换两个寄存器 buffer；epilogue 消费最后一个有效 task。A03 权重仅在 ISA 证实循环内重复 load 时另作独立 hoist 对照，不与预取混成一个变量 | 增加同 worker 可同时在途的请求，尝试隐藏多轮 load 等待；增加寄存器 live range，可能适得其反。既有 `num_stages=2` 不代表已达到该效果 | 下一 task 越界时不得访问／写出；RMS reduction、scale、舍入顺序不变。真实 ISA 必须看到预期 load 前移；scratch 保持 0，资源增加须报告；全部正确性及大 shape 时间通过 |

C2 可以先在 A09 验证，因其每个 group32 独立且现有回退最大；A03 按实际 ISA 决定是否适用。若无法保持原 task body 的数学与布局，则不将它混入最小修复，写清编译限制和所需 body 变更。

若 C1／C2 都不能消除大 shape 回退，交付失败证据及下列方案设计，避免无限调参：

- **A09 两个 1024 task 合批**：同 TG 同时持有 t 与 t+256 的输入，沿 task 维并行计算，reduce 仅在每个原 group 的 32 元素内进行，两个 task 分别写回原地址。stride 仍服务同一 P256 worker；报告 layout／寄存器变化。它改变计算体布局，须作为后续独立候选，不能只删循环后宣称等价优化。
- **大 shape standalone native fallback**：可在外层 dispatch 根据已测 shape 选择 native，避免独立服务使用慢路径；它不满足该 case 的 P256 实现目标。不要在 `persistent=True` 下静默改为 native，或将 native 时间记作 PS；本轮给出明确适用 shape 和接口方案即可。
- 动态 task counter、改变 worker 编号、直接增 P 不能作为本轮正式修复；前两者不能消除不拆 task 时的 `ceil(N/256)` 下界，增 P 则改变验收条件。第2节的 P 对照仅用于诊断，不能据其通过而宣称 P256 已修复。

## 4. 最终验收与回传

- [ ] 每个保留候选：冻结 hash／compile、原 36 项、新 shape／Graph／guard／oracle 全通过，无 skip；失败候选保留失败证据，不进入正式实现。
- [ ] 正式报告同时给出 frozen native、原 PS、修复 PS 在 rows=96／257／513、hot／rotating 的全部 12 项；边界矩阵另表。同 GPU、同输入 pool、同 Graph 方法、每 provider 15 轮，性能不带 profiler。
- [ ] 沿用上轮判据：PS median 增量超过 native P90−P10 且 PS P10 高于 native P90，判 regression；小差异／区间重叠标 inconclusive，不自动写 pass。若修复仅改善部分 case，逐行保留回退。不能用预期融合收益替代 standalone 验收。
- [ ] 每个原因写成“观测 → 对应源码／ISA → 区分实验结果 → 结论／剩余证据”，分别讨论串行迭代、尾 task、循环／port、资源与访存。没有 counters 或区分实验时，不分配虚构损失百分比。
- [ ] 报告用中文，按“1. 为什么慢／何时慢／怎么改；2. 正确性和功能边界”组织，问题与答案就地对应。报告、CSV、每轮 JSON、XML、日志、真实 JIT 产物／manifest、修改 commit／diff 一并提供。
- [ ] 将本轮结果写入新的 `docs/mono-results/<日期>-gfx1250-followup/`，保留上轮数据；主结果报告开头汇总基线及各候选。更新本 TODO 的复选框和状态，填写证据路径；未执行不能勾选。
- [ ] 代码、测试、报告分别提交，发布到 `ROCm/aiter:asm_mon_dev`，核对远端 SHA。推送前 fetch 并保留他人的新提交，不 force-push。交付“已验证／失败／未验证”三类清单及可读取的完整产物位置。

若阻塞，保存已经完成的结果，说明具体错误、已尝试处理、所缺条件及解除方法；GPU 负载无法满足计时条件、工具缺少计数器或没有兼容 sanitizer 应分别记录，不能笼统称为全部验收完成。

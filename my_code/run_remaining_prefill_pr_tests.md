# 从 host 执行剩余 prefill PR 测试

在 a07-3 的 host 上，从 `at_merge` 仓库根目录执行：

```bash
cd /data/yanguahe/code/wk_sp1/at_merge
python3 -u my_code/run_remaining_prefill_pr_tests.py
```

默认执行会先运行下面三个新增的 **Non-balanced** workload，再继续原有完整测试。只想运行这三个规模时使用：

```bash
python3 -u my_code/run_remaining_prefill_pr_tests.py --non-balanced-only
```

| workload | experts | tokens | topk | model_dim | inter_dim | routing |
|---|---:|---:|---:|---:|---:|---|
| e96_t16384 | 96 | 16384 | 6 | 7168 | 3072 | Non-balanced |
| e1_t16384 | 1 | 16384 | 1 | 16384 | 8192 | Non-balanced |
| e96_t512 | 96 | 512 | 6 | 7168 | 3072 | Non-balanced |

新增部分的全部配置都设置 `AITER_MOE_EXPERT_BALANCE=false`、`AITER_GROUPED_A4W4_A_PRESHUFFLE=1`，在 `hyg_fyd1` 中相邻执行 main pin0、ad9882df pin0/pin1、发布版 pin0/pin1。新增部分固定使用已修复 E96/T512 GEMM2 的完整发布 commit `9bebc86b4e8ac4c3c6b1db8021318b45b580fc6f`，其源码和容器状态仍在 host 侧核验。

每个 workload 的每个配置先做原 numerical reference 验证，再串行跑三次性能取中值；配置顺序正反交替。三个 workload 共 15 个 correctness worker 和 45 个 performance worker。复用这一组同期样本报告同 pin 的 ad9882df 兼容性及 main pin0 对发布版最佳 pin 的变化，不额外重复计时。

新增部分有 18 项性能检查：12 项同 pin 的 ad9882df 对照、6 项 main pin0 对发布版逐 GEMM 最佳 pin 对照，耗时最多回退 0.5%。默认完整运行共检查原有 42 项和新增 18 项，即 60 项；`--non-balanced-only` 只检查新增的 18 项，不运行原九 T、balanced 旧表、候选筛选或 CPU ABI 全表，也不读取或复用它们的历史性能归档。

单独运行的日志目录带 `_non_balanced_` 标记。`result.json`、`summary.md`、`samples.csv` 和 `pr_before_after_comparison.json` 都保存新增结果；CSV 增加 `balanced` 列，三个新 workload 为 `false`，可与原有同名 balanced workload 区分。`--summarize <run>` 和 `--resume <run>` 自动读取日志中的运行范围，不把两个范围混用。单独模式通过仅表示新增的 18 项通过。

仅检查这三个规模的计划，不执行测试：

```bash
python3 my_code/run_remaining_prefill_pr_tests.py --non-balanced-only --plan-only
```

由用户保证本次测试期间机器性能有效。脚本没有 GPU 空闲、利用率或 GPU/KFD 所有权检查，也不调用旧的自动等待 matrix。所有 compute worker 串行执行，通过 `docker exec` 进入 `hyg_fyd1`。PR 修改前的完整 main 只测 pin=0；PR 修改后保留 pin=0 和 pin=1。

每个 worker 使用原始 numerical reference、精度门槛和 `torch.profiler` timer，固定 `AITER_LOG_MORE=0`、`--warmup 5 --iters 101 --seed 0 --data-init uniform --scale-init auto`。单项运行使用 `timeout --signal=TERM --kill-after=2s 115s`，host 外层等待上限为 119 秒。程序失败或超时会保存日志并停止后续测试。

运行时间较长时，可以先进入固定 tmux session 再执行上述命令：

```bash
tmux new-session -A -s remaining-prefill-pr
```

## 测试范围与顺序

1. 核对完整源码 manifest、Git commit、源码 SHA256、`hyg_fyd1` container 和 FlyDSL/PyTorch 版本。
2. 对两份已经准备的修复候选执行原精度验证，保存实际 ELF、`.text` SHA256 和 ISA。分别针对 T256 GEMM2 的普通短行路径、T4096 GEMM1 的 paired scale 调度路径进行三次同期筛选。
3. 目标阶段耗时减少至少 0.8%、另一阶段最多回退 0.5% 时，执行正反向各三组 control；通过后再跑三次归档。paired scale 候选的归档覆盖 T4096/5120/6144/16384。T256 归档对照完整 main；大 T 归档同时检查原固定通用扩展归档。
4. 根据上述验收结果，选用完整发布版、单项候选或两项修改组合的完整候选，执行完整验收。源码仍在隔离 worktree 中，不修改正式 branch 或工作区。
5. 九 T 的 PR 修改前后 GEMM1/GEMM2 比较：before 为当前 PR 的完整 main 基线 `d0fd39399b46a60d8b497ec622d531e743bd27ea`，after 为选定的完整发布版或修复候选，每版三次取中值。T256/512/1024/2048/3072 使用 APRE0/pin0；大 T 的 before 使用 APRE1/pin0，after 使用 APRE1/pin1。汇总独立列出九个 T 的 PR 前后中值与两阶段耗时变化百分比。另列规定门槛：小 T 对同期完整 main 最多回退 0.5%；大 T 对原固定通用扩展归档最多回退 0.5%。
6. 旧表四 workload，APRE1：pin0 组相邻执行完整 main pin0、完整 `ad9882df24b734748974afa161eb66ed76108fcf` pin0、完整 after pin0；pin1 组只执行 `ad9882df` pin1 和 after pin1。每个配置各跑三次，均在 `hyg_fyd1`。固定 pin0 的 PR 前后表以 main 为分母；兼容性验收仍以同期同 pin 的 `ad9882df` 为分母，保留 16 项检查。
7. 逐 workload 的 PR 前后比较：before 只使用 `hyg_fyd1` 的完整 main pin0；after 的 pin0/pin1 都在 `hyg_fyd1`。三个配置相邻执行，每个配置各跑三次；逐 GEMM 独立选择 after 更快的中值。after 的最佳 pin 相对 main pin0 最多回退 0.5%，这八项检查列入验收结果。不再测试 main pin1，也不依赖 `dsr1_ep4`。

旧表 workload 为：

| workload | routing |
|---|---|
| E64/T1536/topk8/H7168/I2048 | non-balanced |
| E96/T16384/topk6/H7168/I3072 | balanced |
| E1/T16384/topk1/H16384/I8192 | balanced |
| E96/T512/topk6/H7168/I3072 | balanced |

只有 boot、before/after commit、完整源码与已完成证据均未改变，才允许复用完整历史配对。第一次九 T 的 before 为 `3ba5ff7b`，与当前 PR 基线 `d0fd3939` 不同，因此不能用于本轮 PR 前后比较；原始中值和五项失败继续保存为历史证据。重启后不跨 boot 复用性能样本。选择新候选后，九 T 和旧表完整重测，不把原发布版样本混入候选中值。

## 原任务对应关系

九 T 使用 H5120/I2304、global E384/topk6/EP4 的单 GPU per-rank compute 等效输入：local E96、每卡有效 T、random non-balanced routing。GEMM1 的输入准备按 APRE0/APRE1 配置执行；APRE1 下输入已 preshuffle，kernel 内 activation、requant 和输出处理保留在计时范围。

| 要求 | 参考源码或数据 | 输出与检查 |
|---|---|---|
| 九个 T 的 PR 修改前后性能变化 | 完整 main 对完整 after | 九个 T 的两阶段 μs、TFLOPS、BW，及 PR 前后耗时变化百分比 |
| T256/512/1024/2048/3072 不回退 | 同期 main，双方 APRE0/pin0 | 10 项 kernel 检查，耗时最多增加 0.5% |
| T4096/5120/6144/16384 保持通用扩展收益 | 原 v152 通用扩展的固定归档，SHA256 锁定 | 8 项 kernel 检查，耗时最多增加 0.5% |
| 旧表四 workload 的 pin0/pin1 兼容性 | 同配置、同容器的完整 ad9882df | 16 项 kernel/pin 检查，耗时最多增加 0.5% |
| 旧表 PR 前后最佳 pin 比较 | before: main pin0/hyg；after: pin0 / pin1/hyg | 每阶段独立选 after 最佳 pin，记录变化并检查 8 项 0.5% 门槛 |
| 最终代码发布与 PR body | 通过验收的源码及完整原始记录 | agent 后续 commit/push、实际 PR HEAD 核验、性能及复现命令写入 PR body |

原有部分共 42 项性能门槛；默认执行还包含新增 Non-balanced 的 18 项，共 60 项。原有完整测试的 43 项 CPU launch ABI 与原 numerical reference 也必须通过。大 T 的 PR 前后比较以 main 为分母，规定验收以固定通用扩展归档为分母，两张表分别保存。固定归档不是重新测量 main 得到的替代分母。

每组性能配置相邻执行，样本顺序正反交替；routing hashes 必须一致。完整源码使用隔离 worktree，并校验实际加载的 Python 文件、CSV、commit 与 SHA256。脚本只运行本轮已准备的候选和完整验收；性能不达标时保存全部数据并返回失败状态，由 agent 根据代码差异继续修复、准备下一轮。

当前 PR 为 [ROCm/aiter#7431](https://github.com/ROCm/aiter/pull/7431)，脚本不自动 commit、push、修改 PR body 或合并 PR。最终发布需要确认实际 PR HEAD 与通过测试的源码一致；若后续发生影响本次路径的 main 合入或源码修改，需要重新验收。生产代码的通用性、条件分支和修改行数由后续代码审查负责，性能脚本不能单独证明这些代码质量要求。

## 日志与汇总

每次运行创建唯一目录：

```text
my_code/remaining_prefill_pr_test_logs/YYYYMMDD_HHMMSS_<id>/
```

其中包含：

| 文件或目录 | 内容 |
|---|---|
| `controller.log` | 脚本进度和停止原因 |
| `commands.sh` | 实际执行的 worker 命令 |
| `logs/` | 每项测试及 ISA 的原始 stdout/stderr |
| `results/` | worker JSON、runtime manifest、CPU ABI XML |
| `artifacts/` | 实际 ELF 和 MLIR |
| `inputs/` | 使用的源码 manifest、固定归档和首次发布验收记录 |
| `result.json` | 实时状态、全部样本、候选评估和门槛结果 |
| `samples.csv` | 样本的 GEMM1/GEMM2 μs、TFLOPS、BW(GB/s) |
| `pr_before_after_comparison.json` | 九 T 与旧表 pin0 的 PR 前后完整中值、两阶段耗时变化百分比和源码 commit |
| `assessment.json` | 已完成、通过、未通过、未完成的性能门槛与 CPU ABI 状态；中途失败时仍生成 |
| `summary.md` | 各配置三次中值、九 T 与旧表的 PR 前后变化、独立的 0.5% 门槛和 main pin0 对 after 最佳 pin 比较 |

最新运行位置写在 `my_code/remaining_prefill_pr_test_logs/latest_run.json`。用户跑完后，agent 从该位置读取日志，分析回退并继续后续代码与 PR 收尾。

退出码 `0` 表示本脚本全部性能门槛通过；`2` 表示数据完整但仍有回退；`1` 表示运行、正确性或配置检查失败；`130` 表示用户中断。原始日志始终保留。

只查看计划：

```bash
python3 my_code/run_remaining_prefill_pr_tests.py --plan-only
```

从已有结果重新生成报告，不执行测试：

```bash
python3 my_code/run_remaining_prefill_pr_tests.py --summarize my_code/remaining_prefill_pr_test_logs/<run>
```

`--resume <run>` 只允许同 boot、同源码和同脚本下继续尚未开始的项目；已有失败或超时 worker 时应先分析日志，脚本不会自动重试。

## 只补跑本次中断的最后对比

你已完成的 `20261106_050242_4e432184` 运行，九 T、旧表与 CPU ABI 结果继续保留。按新的 main pin0 要求，只补最后四个 workload 的三个配置：main pin0、after pin0、after pin1，每项三次，共 36 个性能 worker：

```bash
cd /data/yanguahe/code/wk_sp1/at_merge
python3 -u my_code/run_remaining_prefill_pr_tests.py --finish-best-pin my_code/remaining_prefill_pr_test_logs/20261106_050242_4e432184
```

这个入口新建日志目录，保留原 `result.json` 与失败记录，不重新筛选两份已经失败的候选，也不重测已完成的九 T 和 ad9882df 验收。只有 boot、完整源码、冻结输入和 `hyg_fyd1` 版本与原运行一致，才能复用。只允许处理被撤销的 main pin1/dsr 接口失败；其他失败、已经完成的最佳 pin 对比或跨 boot 复用会被拒绝。

原来的八项性能回退继续列为未通过；补完最后对比不会将这些失败改成通过。补跑结果中同时保存继承数据的原路径和 SHA256。若机器已重启，需要建立新 cohort；脚本会在启动性能 worker 前拒绝复用旧数据。

这些数据是单 GPU 的 per-rank compute 性能和原 numerical reference 验证；真实四卡 EP/P2P runtime 尚未验收。脚本完成不等于已发布最终修改，commit/push 和实际 PR HEAD 复核仍由 agent 后续执行。

## 2026-11-06 原任务复核与同步（历史记录）

新版脚本已同步到本地与 a07-3 的上述路径，SHA256 为 `6b01eb4cea9d13ec03015d4dfa4ceeaa881f67827713c73b38b21925de38312c`。完整 main before 为 `d0fd3939`，正式 branch 保持 `hyg_opt_moe_a4w4_prefill_dsv41 / bdb170a1`。远端旧脚本和文档均已备份；两个既有容器已在重启后启动。

本轮只完成离线及远端静态检查，没有启动 GPU 测试或产生新性能数据。用户按本文入口启动后，再按实际日志完成 42 项门槛、最终代码发布与 PR body 收尾。历史 `3ba5ff7b` 对照不能用于本轮 main 前后性能比较。

## 2026-11-06 更新：PR 前只测 main pin0

本次按用户最新指示撤销 main pin1/dsr 对照。活动测试计划只使用 `hyg_fyd1`；main_dsr manifest 和 dsr 容器预检查已删除。九 T 的 APRE/pin 配置、43 项 CPU ABI、16 项同 pin 的 ad9882df 回退门槛和原 timer/reference/精度均保留。PR 后最佳 pin 的八项检查现在以 main pin0 为分母，总计仍为 42 项性能门槛。

脚本增加中断结果的部分验收汇总：缺失项目明确标记未完成，不能取尚未测完配置的最小值作为最佳 pin。重新生成报告只写衍生文件，保留原 `result.json`。本次脚本和说明同步不运行 GPU、不修改生产 kernel、Git branch 或 PR。


本次活动脚本 SHA256：`bd6b5ec83d4dcf9a44accb17f828992dd3205c69a02affadab9299a82b9aae8d`。离线验证与全部配置生成检查通过；计时worker未改变。

## 2026-11-07 更新：默认包含三个新增 Non-balanced workload

默认入口先运行新增三组，再执行原完整测试；`--non-balanced-only` 只运行新增三组。新模式使用当前修复发布 commit `9bebc86b4e8ac4c3c6b1db8021318b45b580fc6f`，旧表中的 balanced 定义和历史原始记录保持原样。

活动脚本 SHA256：`f2b77c9fa147b3aa82f4b528013f3d913098ee719420c669b3178503578cc473`。89 项离线检查通过，覆盖精确参数、全部 false routing 配置、正反顺序、三次中值、45 条完整性能样本、缺失 pin 拒绝判为通过、同 pin 的 ad9882df 回退不被最佳 pin 隐藏、历史 42 项检查与默认 60 项检查、无需旧 balanced 归档的汇总，以及原 `result.json` 不被汇总改写。

这些检查使用明确标记的离线合成 fixture，不作为真实性能数据；没有启动 GPU worker。原 CLI 参数、timer、reference、timeout 和 worker 的执行、清理、采样方法保持原样。源码和文档均为 UTF-8 无 BOM、LF；修改前的脚本和说明已保存到 `.codex_tmp/gfx1250_prefill_pr_20261104/manual_non_balanced_mode_20261107/`，离线检查记录为其中的 `offline_validation.json`。

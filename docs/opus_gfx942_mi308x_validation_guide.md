# OPUS gfx942 / MI308X 验证与离线 Tune 指南

本文用于在 AMD Instinct MI308X（运行时架构应为 `gfx942`）上验证当前分支的
OPUS A16W16/A8W8 生产调用、Torch workspace、Graph/stream、离线 tune 与生产
配置回放。

本文的目标是提供一套可以在目标机器上直接执行、保留原始证据的流程。它不把
其他架构上的 skip 当作 gfx942 覆盖，也不把 gfx950-only family 当作 gfx942
能力。

## 1. 当前 gfx942 能力边界

当前 canonical registry 在 gfx942 上包含：

| Family | gfx942 能力 | 公共入口 | 生产高层入口 |
|---|---|---|---|
| A16W16 | 22 个 exact kid：14 direct + 8 workspace | `opus_gemm()` / `opus_bmm()` | GEMM 使用 `aiter.tuned_gemm.gemm_a16w16()`；目前没有高层 OPUS BF16 BMM wrapper |
| A8W8 block-scale B-preshuffle | kid `11000`，FP8 输入、BF16 输出、FP32 scale | `opus_gemm(..., layout="bpreshuffle")` | `gemm_a8w8_blockscale_bpreshuffle()`，只有 tuned row 为 `libtype=opus` 才进入 OPUS |
| A8W8 no-scale | 不支持；kid `2` 是 gfx950-only | 不应执行 | 不应作为 gfx942 OPUS 覆盖 |
| A8W8 plain block-scale | 不支持；kid `1` 是 gfx950-only | 不应执行 | 不应作为 gfx942 OPUS 覆盖 |
| MXFP8 BMM | 不支持；当前是 gfx950-only | 不应执行 | 不应作为 gfx942 OPUS 覆盖 |

gfx942 A16 workspace kids：

- FP32 workspace：`10200, 10201, 10203, 10204, 10205`。
- BF16 workspace：`10210, 10213, 10216`。
- BF16-workspace kids 只接受 BF16 输出，并且要求 N 属于其 exact-N 合同；不能像
  gfx950 workspace kids 一样统一测试 BF16 和 FP32 输出。

MI308X 与其他产品可能都报告 `gfx942`，生产配置还以 `cu_num` 为 key。必须在
目标 MI308X 上获取实际 CU 数并重新 tune，不能直接复制另一种 gfx942 卡的 tuned
row。

## 2. 代码与文档入口

以这些文件为准：

- `aiter/ops/opus/README.md`：公共 exact API、family 能力、workspace、Graph 和
  subset compile 合同。
- `csrc/opus_gemm/opus_gemm_common.py`：canonical registry、kid 和实例元数据。
- `aiter/ops/opus/launch_plan.py`：gfx942 split-K、workspace shape/dtype 和 ABI
  split 规划。
- `aiter/ops/opus/policy.py`：A16 tuned/heuristic candidate 解析。
- `aiter/ops/opus/gemm_op_a16w16.py`：A16 Torch workspace 与 C ABI 执行路径。
- `aiter/ops/gemm_op_a8w8.py`：A8 B-preshuffle tuned backend 生产路由。
- `csrc/gemm_a16w16/README.md`：A16 多 backend tune 参数与 CSV 格式。
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/README.md`：A8 B-preshuffle CSV 与
  tune 基础流程。
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`：A8 tuner 当前
  实现。其 `--libtype` 已支持 `opus`；如果 README 的 backend 列表不同，以代码
  为准。

本文已经包含 GPU 空闲门、fresh module 和进程级 A1→B1→B2→A2 方法；不需要
依赖某份本地 gfx950 HTML 报告。gfx950 的 kid、family、shape 和结论也不能直接
用于 gfx942。

## 3. 准备隔离环境和证据目录

在仓库根目录执行。将 `<PHYSICAL_GPU>` 替换为 MI308X 的物理 GPU 编号；
`--idle-gpu` 和 `rocm-smi -d` 使用的也是这个物理编号。设置
`HIP_VISIBLE_DEVICES` 后，PyTorch 进程内目标卡会映射为 `cuda:0`。

```bash
cd /path/to/aiter

export MI308_PHYSICAL_GPU=<PHYSICAL_GPU>
export HIP_VISIBLE_DEVICES="$MI308_PHYSICAL_GPU"
export GPU_ARCHS=gfx942
export PYTHONPATH="$PWD"

export MI308_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export MI308_RUN_ROOT="/tmp/opus-mi308x-${MI308_RUN_ID}"
mkdir -p "$MI308_RUN_ROOT/logs" "$MI308_RUN_ROOT/csv" \
  "$MI308_RUN_ROOT/jit-current"

export AITER_JIT_DIR="$MI308_RUN_ROOT/jit-current"
```

每轮使用新的 `MI308_RUN_ROOT`，不要复用 gfx950、其他提交或中断编译留下的 JIT
目录。

记录源码和软件环境：

```bash
git status --short | tee "$MI308_RUN_ROOT/logs/git_status.log"
git rev-parse HEAD | tee "$MI308_RUN_ROOT/logs/git_head.log"
git branch --show-current | tee "$MI308_RUN_ROOT/logs/git_branch.log"

python - <<'PY' | tee "$MI308_RUN_ROOT/logs/torch_device.log"
import torch

assert torch.cuda.is_available(), "ROCm device is unavailable"
p = torch.cuda.get_device_properties(0)
arch = str(getattr(p, "gcnArchName", "")).split(":", 1)[0].lower()
print("torch:", torch.__version__)
print("device:", p.name)
print("arch:", p.gcnArchName)
print("cu_num:", p.multi_processor_count)
assert arch == "gfx942", f"expected gfx942, got {arch!r}"
PY

rocm-smi -d "$MI308_PHYSICAL_GPU" --showproductname --showuse --showmemuse \
  | tee "$MI308_RUN_ROOT/logs/rocm_smi_start.log"
```

验收前应确认：

- PyTorch 输出的 arch 是 `gfx942`。
- 记录了实际 `cu_num`。
- 目标 GPU 无其他 KFD 进程，利用率稳定接近 0%，显存没有外部任务占用。
- 不终止、不抢占外部任务；GPU 忙时等待空闲窗口。

## 4. CPU/静态合同测试

这些用例验证 registry、公共接口、tuned row、split-K 规划和离线 tune runner，
但它们不能代替 gfx942 真机 kernel 测试：

```bash
set -o pipefail
python -m pytest -q -s -rs \
  op_tests/test_opus_interfaces.py \
  op_tests/test_opus_dispatch.py \
  op_tests/tuning_tests/test_opus_gemm_tune_reference.py \
  2>&1 | tee "$MI308_RUN_ROOT/logs/cpu_contracts.log"
```

要求：命令退出码为 0，没有 failed/error。

## 5. gfx942 A16 真机正确性、workspace 与 Graph

第一次使用 fresh JIT 时允许构建模块。该阶段只暴露一张 MI308X：

```bash
set -o pipefail
AITER_REBUILD=1 python -m pytest -q -s -rs \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py \
  2>&1 | tee "$MI308_RUN_ROOT/logs/gfx942_a16_hardware.log"
```

fresh build 完成后，后续阶段使用：

```bash
export AITER_REBUILD=0
```

这组测试在 gfx942 上应实际执行：

- kid `10200` 的 FP32 workspace 路线。
- kid `10210` 的 BF16 workspace 路线。
- automatic workspace 与 caller workspace。
- workspace dtype、容量、contiguous、16-byte alignment 和 device 错误合同。
- Graph capture/replay、两个 stream 的独立 call-scoped workspace。
- exact-kid 数值结果对 Torch FP32 golden。
- gfx942 split-K kid 的 bias 拒绝，不允许静默 framework fallback。

测试文件还包含 gfx950/gfx1250 或双卡参数，目标条件不满足时发生 skip 是正常的；
必须通过 `-rs` 输出确认 gfx942 对应节点实际运行，不能仅以整份测试“无失败”替代
目标覆盖审计。

## 6. gfx942 当前接口、C ABI、A8 kid 11000 与 Graph 代表测试

`op_tests/bench_opus_task1_task2_interfaces.py` 的 `current/gfx942` 路线是现有最完整
的代表性真机测试。它固定 kernel/shape，并对每一层执行 Torch golden 检查：

- A16 kid `10200`：FP32 workspace/output。
- A16 kid `10210`：BF16 workspace/output。
- A8 B-preshuffle kid `11000`：BF16 output。
- public exact API、private family、raw binding、direct C++ 和 Graph replay。

先运行两次 current，观察结果和重复漂移：

```bash
set -o pipefail
python op_tests/bench_opus_task1_task2_interfaces.py \
  --arch gfx942 \
  --endpoint current \
  --pass-id current-B1 \
  --idle-gpu "$MI308_PHYSICAL_GPU" \
  --idle-timeout 600 \
  --warmup 20 \
  --rounds 9 \
  --iters 100 \
  2>&1 | tee "$MI308_RUN_ROOT/logs/interface_current_B1.log"

python op_tests/bench_opus_task1_task2_interfaces.py \
  --arch gfx942 \
  --endpoint current \
  --pass-id current-B2 \
  --idle-gpu "$MI308_PHYSICAL_GPU" \
  --idle-timeout 600 \
  --warmup 20 \
  --rounds 9 \
  --iters 100 \
  2>&1 | tee "$MI308_RUN_ROOT/logs/interface_current_B2.log"
```

如果机器的 `rocm-smi` 不接受该物理索引，先单独解决索引映射；不要删除
`--idle-gpu` 后在繁忙 GPU 上采正式性能。

要求：

- 两个进程均退出 0。
- 每份日志恰有一个 `PERF_COMPLETE`。
- `PERF_COMPLETE` 中 `all_correct` 为 `true`。
- 日志中没有 `nan/inf`、数值 mismatch、uncompiled-id 或 JIT 截断。
- 当前 gfx942 代表集预期产生 15 个 `PERF_CASE`：两个 A16 case 和一个 A8 case，
  每个 case 各五个 timing layer。
- Graph replay 主要作为 device-work control；eager 与 Graph 的趋势需要结合两轮
  重复漂移判断。

该 benchmark 验证的是 current exact public route，不等同于读取 tuned CSV 的
生产高层路由；生产高层必须继续执行下面的 `--run_config`。

## 7. A16W16 OPUS 离线 tune 与生产回放

### 7.1 输入 CSV

A16 shape CSV 至少包含：

```csv
M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle
128,128,512,False,torch.bfloat16,torch.bfloat16,False,False
128,128,512,False,torch.bfloat16,torch.float32,False,False
```

正式测试应替换为目标模型实际 shape。A16 OPUS 输入必须为 BF16，且本路线不使用
`scaleAB` 或 B-preshuffle。

```bash
export MI308_A16_INPUT=/path/to/mi308x_a16_shapes.csv
export MI308_A16_TUNED="$MI308_RUN_ROOT/csv/a16w16_opus_tuned.csv"
export MI308_A16_PROFILE="$MI308_RUN_ROOT/csv/a16w16_opus_profile_all.csv"
```

### 7.2 OPUS-only tune

先使用单卡确认完整流程：

```bash
set -o pipefail
python csrc/gemm_a16w16/gemm_a16w16_tune.py \
  -i "$MI308_A16_INPUT" \
  -o "$MI308_A16_TUNED" \
  -o2 "$MI308_A16_PROFILE" \
  --libtype opus \
  --mp 1 \
  --shape_grouped \
  --warmup 5 \
  --iters 101 \
  2>&1 | tee "$MI308_RUN_ROOT/logs/a16w16_opus_tune.log"
```

检查输出 CSV：

- `gfx` 必须是 `gfx942`。
- `cu_num` 必须等于本机检测结果。
- winner row 的 `libtype` 必须是 `opus`。
- `solidx`/`splitK` 必须是最终 exact pair。
- `err_ratio` 必须在 tuner 门限内。
- 不得把没有 OPUS 可执行 candidate 的 shape 伪装成 PASS。

如果要比较 OPUS 与 gfx942 的其他生产 backend，使用另一份输出文件运行：

```bash
python csrc/gemm_a16w16/gemm_a16w16_tune.py \
  -i "$MI308_A16_INPUT" \
  -o "$MI308_RUN_ROOT/csv/a16w16_asm_opus_tuned.csv" \
  --libtype asm,opus \
  --mp 1 \
  --shape_grouped
```

### 7.3 生产高层回放

`--run_config <CSV>` 会临时把该 CSV 注入生产配置，然后通过
`aiter.tuned_gemm.gemm_a16w16()` 执行；它不是直接调用 tuner raw binding：

```bash
set -o pipefail
python csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --run_config "$MI308_A16_TUNED" \
  --warmup 20 \
  --iters 100 \
  2>&1 | tee "$MI308_RUN_ROOT/logs/a16w16_production_replay.log"
```

要求每个 shape 的 status 为 `ok`，并确认日志中生产 config 实际选择
`libtype=opus`。这一步验证：tuned CSV → final kid/split pair → public
`opus_gemm()` → A16 planner/workspace → C ABI/C++ exact launcher。

## 8. A8W8 B-preshuffle OPUS 离线 tune 与生产回放

### 8.1 输入 CSV

A8 B-preshuffle shape CSV 使用 `M,N,K`：

```csv
M,N,K
128,256,256
256,256,256
```

正式 shape 必须满足 kid `11000` 的 N/K 128-wide contract。tuner 会生成正确的
FP8 输入、B-preshuffled WQ 和 scale storage。

```bash
export MI308_A8_INPUT=/path/to/mi308x_a8_bpreshuffle_shapes.csv
export MI308_A8_TUNED="$MI308_RUN_ROOT/csv/a8w8_bpreshuffle_opus_tuned.csv"
export MI308_A8_PROFILE="$MI308_RUN_ROOT/csv/a8w8_bpreshuffle_opus_profile_all.csv"
```

### 8.2 OPUS-only tune

```bash
set -o pipefail
python csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
  --preshuffle \
  -i "$MI308_A8_INPUT" \
  -o "$MI308_A8_TUNED" \
  -o2 "$MI308_A8_PROFILE" \
  --libtype opus \
  --mp 1 \
  --shape_grouped \
  --warmup 5 \
  --iters 101 \
  2>&1 | tee "$MI308_RUN_ROOT/logs/a8w8_bpreshuffle_opus_tune.log"
```

检查 winner row：

- `gfx=gfx942` 且 `cu_num` 与本机一致。
- `libtype=opus`。
- `kernelId=11000`、`splitK=0`。
- 数值误差通过门限。

若要比较 CK/CKTile/ASM/OPUS，使用另一份输出文件和 `--libtype all`，不要覆盖
OPUS-only 证据：

```bash
python csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
  --preshuffle \
  -i "$MI308_A8_INPUT" \
  -o "$MI308_RUN_ROOT/csv/a8w8_bpreshuffle_all_tuned.csv" \
  --libtype all \
  --mp 1 \
  --shape_grouped
```

### 8.3 生产高层回放

```bash
set -o pipefail
python csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
  --preshuffle \
  --run_config "$MI308_A8_TUNED" \
  --warmup 20 \
  --iters 100 \
  2>&1 | tee "$MI308_RUN_ROOT/logs/a8w8_bpreshuffle_production_replay.log"
```

要求每个 shape 的 status 为 `ok`。必须确认 tuned row 是 `libtype=opus`；否则高层
会合法地进入 CK/CKTile/ASM 等 backend，但那不能算作 OPUS 生产路线覆盖。

## 9. 当前分支与 origin/main 性能比较方法

正式性能比较遵循：

1. baseline 与 current 使用独立 source worktree 和独立 fresh JIT 目录。
2. 固定同一张物理 GPU、同一 CPU core、相同 shape/kid/split/output dtype。
3. 每个 endpoint 使用新进程，顺序为 baseline A1 → current B1 → current B2 →
   baseline A2。
4. 每次 timed case 前等待 GPU 空闲；记录开始/结束 `rocm-smi`。
5. 两端先通过数值检查，再比较 eager 和 Graph。
6. Graph replay 是 kernel/device-work control；若 Graph 稳定而 eager 变化，重点分析
   Python/dispatcher/descriptor/stream host overhead。
7. 回退判定必须高于 A/A、B/B 自身重复漂移；单次小幅变化不能直接判回退。

当前仓库还没有可以不修改就完成 gfx942 全量 paired ABBA 的脚本：

- `op_tests/bench_opus_task1_workspace_paired.py` 硬编码 gfx950 workspace kid
  `200..223/1200..1223`。
- `op_tests/bench_opus_task2_gemm_bmm_paired.py` 硬编码 gfx950 family。
- `op_tests/bench_opus_task1_task2_interfaces.py --endpoint task1` 需要与其预期 ABI
  一致的历史 preserved module，不能直接假设任意 `origin/main` module 都匹配。

因此，在 paired 脚本完成 gfx942 参数化前：

- 可以用第 6 节的 `current-B1/current-B2` 建立 current 稳定性和各层基准。
- 不应把不同脚本、不同 kid 或不同 JIT build 的数值拼成 baseline/current 结论。
- 不应直接宣称已经完成“origin/main internal workspace vs current Torch
  workspace”的 gfx942 性能验收。

若扩展 paired 脚本，必须从 baseline/current 两端 registry 的交集解析 exact kid，
并分别处理 FP32/BF16 workspace 与 BF16-workspace kid 的输出限制。

## 10. 距离 gfx950 同等级全量验收尚缺的测试

以下现有文件是 gfx950-only，不能把它们在 MI308X 上的 skip 当成 PASS：

- `op_tests/test_opus_gfx950_exhaustive.py`。
- `op_tests/test_opus_ctypes.py` 的真机部分（`_gfx950_case`）。
- `op_tests/bench_opus_task1_workspace_paired.py`。
- `op_tests/bench_opus_task2_gemm_bmm_paired.py`。
- `op_tests/test_gemm_a8w8_opus_highlevel.py` 的真机部分。
- `op_tests/test_opus_a8w8_bmm.py`（MXFP8 BMM 是 gfx950-only）。

要达到 gfx950 报告中的全量同等级，需要新增或参数化：

1. gfx942 A16 exhaustive：覆盖全部 22 个 A16 exact kid，其中 8 workspace、14
   direct；BF16-workspace kids 只测允许的 BF16 output。
2. gfx942 ctypes：首次 pybind priming、缓存 C ABI、非默认 stream、Graph、错误桥和
   device restore。
3. gfx942 paired workspace ABBA：baseline/current 同 kernel、同 shape、同 split，
   覆盖 eager 与 Graph。
4. gfx942 public/private router paired ABBA。
5. A8 B-preshuffle 高层 tuned-row 真机回归；第 8 节的 tune + `--run_config` 在新增
   pytest 前承担这一生产验收。

在这些补齐前，合适的结论措辞是：

> gfx942/MI308X 的代表性 A16 workspace、Graph、exact API、A8 B-preshuffle kid
> 11000、离线 tune 和生产回放已经验证；尚未完成全部 22 个 A16 kid、gfx942
> ctypes 与 baseline/current paired ABBA 的全量验收。

## 11. 最终验收清单

一次可追溯的 MI308X 验证至少应保留：

- `git_head.log`、`git_status.log`、`torch_device.log`。
- 开始和结束时的 `rocm-smi`。
- CPU/static contract 测试日志。
- gfx942 A16 hardware/workspace/Graph pytest 日志。
- 两份 current interface benchmark 日志。
- A16 OPUS-only tuned CSV、全 candidate profile、production replay 日志。
- A8 B-preshuffle OPUS-only tuned CSV、全 candidate profile、production replay
  日志。
- 所有文件的 SHA256、命令退出码和 UTC 时间。

最终 PASS 必须同时满足：

- 所有目标命令退出 0。
- 真机数值检查无 mismatch、NaN 或 Inf。
- workspace/Graph/stream 合同通过。
- A16 production replay 确实选中 `libtype=opus`。
- A8 production replay 确实选中 `libtype=opus,kernelId=11000`。
- 没有把 gfx950/gfx1250 skip、CK/ASM fallback 或空 candidate 当作 OPUS PASS。
- 性能结论附带重复漂移和 Graph control，不只引用单次 eager 数字。

完成全部阶段后记录结束状态并生成校验和：

```bash
rocm-smi -d "$MI308_PHYSICAL_GPU" --showuse --showmemuse \
  | tee "$MI308_RUN_ROOT/logs/rocm_smi_end.log"

find "$MI308_RUN_ROOT" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$MI308_RUN_ROOT/SHA256SUMS"

echo "$MI308_RUN_ROOT"
```

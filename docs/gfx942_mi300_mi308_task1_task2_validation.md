# OPUS Task1 / Task2 gfx942（MI308、MI300）全量正确性与性能验收

更新时间：`2026-08-13 UTC`

状态：**执行方案已定义；MI308 和 MI300 的实机结果尚未产生。**

本文定义如何在 MI308 和 MI300 两个 gfx942 节点上，按 gfx950/MI355X 已采用的口径，分别完成
Task1 与 Task2 的全量正确性、Graph、并发和性能验收。两个节点必须独立执行、独立保存结果，
不能因为二者都报告 gfx942 就复用 JIT、日志或通过结论。

本文只定义最终执行端点，不保存中间实现版本。执行中不得把 codegen、交叉编译、CPU fixture、
目标架构 skip 或其他 GPU 的结果写成 gfx942 实机通过。

## 1. 验收目标与端点

### 1.1 Task1

Task1 的目标是 `OPUS A16W16 Split-K workspace To Torch`。验收必须比较：

| 端点 | 内容 | 用途 |
|---|---|---|
| 原始基线 | commit `ca68b4f3501762c15c550cb920a5516e9710cf89`；C++ 按 stream 管理 workspace | Task1 改造前基准 |
| Task1 最终端点 | Torch 创建或复用 workspace，并把 Tensor pointer 传给 A16 启动路径 | 验证 workspace 改造本身 |
| 当前最终端点 | Task1 workspace 语义加上 Task2 的统一 public 与 Python 选核 | 验证累计版本没有破坏 Task1 |

Task1 正确性至少覆盖全部 8 个 gfx942 workspace kid 的合法输出 dtype、自动 workspace、调用方
workspace、Graph、双 stream 和错误 Tensor。5 个 FP32-workspace kid 同时覆盖 BF16/FP32 Y；
3 个 BF16-workspace kid 覆盖 BF16 Y，并验证 FP32 Y 被明确拒绝。性能必须用原始基线与
Task1/current 的相同输入做 `A1 → B1 → B2 → A2` 配对。

### 1.2 Task2

Task2 的目标是：Python 负责 tuned CSV、heuristic、fallback、dtype/layout 路由和输出准备；C++
只保留必要检查、`kid → launcher` 查询及 kernel launch。验收必须证明：

1. gfx942 tuned miss 在 Python heuristic 中得到最终 kid；
2. 有效 tuned row 优先于 heuristic；无效的整条 `(kid, split-K)` 被丢弃；
3. heuristic 无法提供合法 OPUS kid 时进入 PyTorch fallback；
4. 统一 `aiter.ops.opus.opus_gemm()` 按最终 kid 进入 A16 或 A8 私有启动函数；
5. C++ 不再根据 shape、bias 或 tuned row 改选另一个 kid；
6. gfx942 A8W8 blockscale bpreshuffle kid 11000 通过 unified public、真实 tuned 高层路径、Graph 和双
   stream；
7. Task2 public 路由的性能成本与 Task1 私有 A16 启动分开报告。

### 1.3 gfx942 全量 kernel 范围

当前 `kernels_list` 中 gfx942 的实机范围固定为：

| 类型 | 数量 | kid |
|---|---:|---|
| A16W16 非 workspace | 14 | `10000, 10001, 10003, 10006, 10300, 10301, 10302, 10303, 10305, 10310, 10311, 10312, 10313, 10314` |
| A16W16 workspace | 8 | `10200, 10201, 10203, 10204, 10205, 10210, 10213, 10216` |
| A8W8 blockscale bpreshuffle | 1 | `11000` |

因此当前最终端点的全量正确性必须实际执行 `22` 个 A16 kid 和 `1` 个 A8 kid。gfx950 的 A8
no-scale、plain blockscale 和 MXFP8 BMM 不属于 gfx942 实机范围；它们在 gfx942 上只能做跨架构
拒绝测试，不能计入通过数。

## 2. 两台机器必须分别形成结论

每台机器使用一行独立状态：

| 机器 | runtime arch | 当前状态 | 最终要求 |
|---|---|---|---|
| MI308 | 必须为 `gfx942` | NOT RUN | 全量正确性、Task1 ABBA、Task2 ABBA 全部完成 |
| MI300 | 必须为 `gfx942` | NOT RUN | 全量正确性、Task1 ABBA、Task2 ABBA 全部完成 |

两台机器即使使用相同源码，也必须分别记录 GPU 名称、`gcnArchName`、CU 数量、ROCm、PyTorch、
CPU、module SHA-256 和性能重复漂移。Python heuristic 使用实际 CU 数量，不能在测试中硬编码
`304`。

## 3. 执行前必须准备的测试能力

当前仓库的 `test_opus_gfx950_exhaustive.py`、`bench_opus_gfx950_workspace_ab.py` 和
`bench_opus_task1_task2_interfaces.py` 含 gfx950 固定 kid/shape/arch guard，不能直接在 gfx942 上
运行，也不能只删除 guard 后沿用 kid 200、kid 1 或 kid 2。

正式执行前只增加或参数化测试/benchmark，不新增生产 Python interface：

1. 新增 `op_tests/test_opus_gfx942_exhaustive.py`，从当前 `kernels_list` 动态得到 22 个 A16 kid；
2. 为 `op_tests/bench_opus_gfx950_workspace_ab.py` 增加 `--arch gfx942`，gfx942 时动态使用 8 个
   workspace kid 和每个 kid 声明的 workspace dtype；
3. 为 `op_tests/bench_opus_task1_task2_interfaces.py` 增加 `--arch gfx942`，覆盖 A16 kid 10200、
   10210，以及 A8 kid 11000；
4. 不修改生产接口来方便测试，不恢复旧 generic C++ runtime lookup。

这三项完成前，本文的“全量正确性”和“性能”状态必须保持 `BLOCKED`，不能用现有 focused suite
代替。

### 3.1 gfx942 exhaustive 的硬性内容

`test_opus_gfx942_exhaustive.py` 必须：

- 只在 `OPUS_GFX942_EXHAUSTIVE=1` 时运行；
- 断言 runtime arch 为 gfx942，否则失败或明确 skip，且该 skip 不能计为通过；
- 从 `kernels_list` 与 `get_kernel_instance("gfx942", "a16w16", kid)` 得到 22 个 A16 kid；
- 断言其中 workspace 为 8、non-workspace 为 14；
- 对 5 个 FP32-workspace kid 同时执行 BF16、FP32 输出；对 3 个 BF16-workspace kid 执行
  BF16 输出并验证 FP32 输出被拒绝；同时检查每个 kid 声明的 workspace dtype；
- 对 non-workspace kid 执行其声明支持的输出 dtype，并证明没有创建 workspace；
- 每个 case 使用固定 seed、Torch FP32 golden、`torch.cuda.synchronize()` 和有限值检查；
- workspace case 同时覆盖 caller Tensor 复用与正常自动创建；
- 输出每个 kid、kernel 类型、shape、输出 dtype、workspace shape/dtype、split-K 和最大误差；
- 任何 kid 未编译、未启动、数值失败或被目标架构 skip，都使全量结果失败。

建议使用每个条目的 `B_M/B_N`，并令 `K = 32 * B_K`。workspace kid 使用 `split_k=2`；
wave-K-coop 等 non-workspace 路径仍须满足生成 launcher 的 K 整除条件。

### 3.2 gfx942 性能 benchmark 的硬性内容

Task1 benchmark 在 gfx942 必须使用：

```text
WORKSPACE_KIDS = 10200, 10201, 10203, 10204,
                 10205, 10210, 10213, 10216
```

每个 kid 使用 `M=B_M, N=B_N, K=32*B_K, batch=1, split-K=2`。5 个 FP32-workspace kid
分别测 BF16 与 FP32 Y，3 个 BF16-workspace kid 只测 BF16 Y，共 `5 × 2 + 3 × 1 = 13`
个数值 case；另执行 3 个 FP32 Y 预期拒绝 case。当前端点的 workspace 不能统一硬编码为
FP32，必须按该 kid 的 `splitk_workspace_dtype` 创建。

Task2 分层 benchmark 至少包含：

| case | 目的 |
|---|---|
| A16 kid 10200、FP32 workspace | 覆盖 gfx942 FP32 workspace 路径 |
| A16 kid 10210、BF16 workspace | 覆盖 gfx942 BF16 workspace 路径 |
| A8 kid 11000、BF16 Y | 覆盖 bpreshuffle 的旧 Task1 tune 与当前 unified public |

每个 case 分别报告高层/统一 public、私有启动函数、raw binding、direct pybind/C++ 和 Graph
replay；不允许把 gfx950 kid 或空 capability 计时写进 gfx942 表。

## 4. 每台机器的目录与环境

MI308、MI300 各开一个全新 shell。下面先以 MI308 为例；MI300 只替换 `MACHINE_LABEL`。所有环境
变量必须在第一次 import Torch/AITER 前设置。

```bash
set -euo pipefail

export REPO_ROOT=/root/workspace/0810/aiter
export MACHINE_LABEL=mi308
export PHYSICAL_GPU=0
export CPU_CORE=0
export VALIDATION_ROOT
VALIDATION_ROOT="$(mktemp -d "/tmp/aiter-gfx942-${MACHINE_LABEL}.XXXXXX")"

export CURRENT_SRC="$REPO_ROOT"
export BASELINE_SRC=/absolute/path/to/ca68b4f-source
export TASK1_FROZEN_SRC=/absolute/path/to/task1-final-source

export CURRENT_JIT="$VALIDATION_ROOT/jit-current"
export BASELINE_JIT="$VALIDATION_ROOT/jit-baseline"
export TASK1_JIT="$VALIDATION_ROOT/jit-task1"
export LOG_DIR="$VALIDATION_ROOT/logs"

mkdir -p \
  "$CURRENT_JIT/build" \
  "$BASELINE_JIT/build" \
  "$TASK1_JIT/build" \
  "$LOG_DIR"

export HIP_VISIBLE_DEVICES="$PHYSICAL_GPU"
export GPU_ARCHS=gfx942
export AITER_REBUILD=0

printf 'VALIDATION_ROOT=%s\n' "$VALIDATION_ROOT"
```

三个源码端点必须是独立目录。禁止在当前 dirty 工作树内 checkout 原始基线或 Task1 冻结版本。
gfx950 的 `.so` 不能带到 gfx942 使用；三个端点都必须在当前目标节点 fresh build。

若暂时没有 `TASK1_FROZEN_SRC`，可以先完成原始基线/当前 Task1 workspace A/B 和当前 Task2
private/public 路由 A/B，但“历史 Task1 → Task2 完整性能”必须标为 `BLOCKED`，不能用当前源码猜测
冻结端点。

## 5. 冻结源码、软件和硬件证据

### 5.1 当前源码

```bash
cd "$CURRENT_SRC"
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "$LOG_DIR/start_utc.txt"
git rev-parse HEAD | tee "$LOG_DIR/current_git_head.txt"
git status --short | tee "$LOG_DIR/current_git_status.txt"
git diff --check | tee "$LOG_DIR/current_git_diff_check.txt"
git diff --binary > "$LOG_DIR/current_tracked_changes.patch"
git ls-files --others --exclude-standard -z \
  | sort -z \
  | xargs -0 -r sha256sum \
  > "$LOG_DIR/current_untracked_files.sha256"
sha256sum "$LOG_DIR/current_tracked_changes.patch" \
  | tee "$LOG_DIR/current_tracked_changes.sha256"
```

### 5.2 原始基线与 Task1 冻结端点

```bash
test "$(git -C "$BASELINE_SRC" rev-parse HEAD)" \
  = "ca68b4f3501762c15c550cb920a5516e9710cf89"
git -C "$BASELINE_SRC" status --short \
  | tee "$LOG_DIR/baseline_git_status.txt"

git -C "$TASK1_FROZEN_SRC" rev-parse HEAD \
  | tee "$LOG_DIR/task1_git_head.txt"
git -C "$TASK1_FROZEN_SRC" status --short \
  | tee "$LOG_DIR/task1_git_status.txt"
```

若 Task1 冻结端点不是 git checkout，必须保存源码 archive 的 SHA-256、文件清单，以及它与 gfx950
Task1 性能端点相同的证明。

### 5.3 软件与 GPU

```bash
python3 --version 2>&1 | tee "$LOG_DIR/python_version.txt"
/opt/rocm/bin/hipcc --version 2>&1 | tee "$LOG_DIR/hipcc_version.txt"
/opt/rocm/bin/rocm-smi --showproductname --showuse --showmemuse --showpids \
  2>&1 | tee "$LOG_DIR/gpu_before.txt"

python3 - <<'PY' | tee "$LOG_DIR/runtime_arch.txt"
import os
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()

print(f"machine={os.environ['MACHINE_LABEL']}")
print(f"device_name={props.name}")
print(f"gcnArchName={getattr(props, 'gcnArchName', None)}")
print(f"multi_processor_count={props.multi_processor_count}")
print(f"torch={torch.__version__}")
print(f"torch.version.hip={torch.version.hip}")
print(f"GPU_ARCHS={os.environ['GPU_ARCHS']}")

assert arch == "gfx942", arch
assert os.environ["GPU_ARCHS"] == "gfx942"
PY
```

GPU 有其他计算进程、显存占用持续变化或利用率不为空闲时，不开始性能测试。不得终止不属于本次
验收的进程。

## 6. 生成统一的全量 kid sidecar

当前端点从 `kernels_list` 动态生成，不维护第二份手写路由表：

```bash
cd "$CURRENT_SRC"
CURRENT_JIT="$CURRENT_JIT" python3 - <<'PY' \
  | tee "$LOG_DIR/current_sidecar.txt"
import json
import os
from pathlib import Path

from csrc.opus_gemm.opus_gemm_common import get_kernel_instance, kernels_list

a16 = {
    int(kid)
    for kid, instance in kernels_list.items()
    if get_kernel_instance("gfx942", "a16w16", kid) is instance
}
a8 = {
    int(kid)
    for kid, instance in kernels_list.items()
    if get_kernel_instance(
        "gfx942", "a8w8_blockscale_bpreshuffle", kid
    ) is instance
}

assert len(a16) == 22, sorted(a16)
assert len(a8) == 1 and a8 == {11000}, sorted(a8)
kids = sorted(a16 | a8)

path = Path(os.environ["CURRENT_JIT"]) / "build/compiled_kids_opus.json"
path.write_text(json.dumps(kids) + "\n")
print(f"path={path}")
print(f"a16_count={len(a16)}")
print(f"a8_count={len(a8)}")
print(f"kids={kids}")
PY

cp "$CURRENT_JIT/build/compiled_kids_opus.json" \
  "$BASELINE_JIT/build/compiled_kids_opus.json"
cp "$CURRENT_JIT/build/compiled_kids_opus.json" \
  "$TASK1_JIT/build/compiled_kids_opus.json"

sha256sum \
  "$CURRENT_JIT/build/compiled_kids_opus.json" \
  "$BASELINE_JIT/build/compiled_kids_opus.json" \
  "$TASK1_JIT/build/compiled_kids_opus.json" \
  | tee "$LOG_DIR/sidecar.sha256"
```

若旧端点生成器不识别 sidecar，必须在旧端点的独立副本中把同一 23-kid 集合作为 build subset
输入，并在构建后审计生成表。不得因此改动当前生产代码。

## 7. 三个端点 fresh build

### 7.1 当前最终端点

```bash
cd "$CURRENT_SRC"
AITER_JIT_DIR="$CURRENT_JIT" AITER_REBUILD=1 \
python3 - <<'PY' 2>&1 | tee "$LOG_DIR/current_fresh_build.log"
import torch
from aiter.ops.opus import opus_gemm

torch.manual_seed(94201)
XQ = torch.randn((1, 128, 4096), device="cuda", dtype=torch.bfloat16)
WQ = torch.randn((1, 128, 4096), device="cuda", dtype=torch.bfloat16)
Y = torch.empty((1, 128, 128), device="cuda", dtype=torch.float32)
opus_gemm(XQ, WQ, Y, kid=10200, split_k=2)
torch.cuda.synchronize()
torch.testing.assert_close(
    Y, torch.bmm(XQ.float(), WQ.float().transpose(1, 2)),
    rtol=1e-3, atol=0.05,
)
print("current_fresh_build=PASS")
PY
```

### 7.2 原始基线

```bash
cd "$BASELINE_SRC"
PYTHONPATH="$BASELINE_SRC" AITER_JIT_DIR="$BASELINE_JIT" AITER_REBUILD=1 \
python3 - <<'PY' 2>&1 | tee "$LOG_DIR/baseline_fresh_build.log"
import torch
from aiter.ops.opus import opus_gemm_a16w16_tune

torch.manual_seed(94201)
XQ = torch.randn((1, 128, 4096), device="cuda", dtype=torch.bfloat16)
WQ = torch.randn((1, 128, 4096), device="cuda", dtype=torch.bfloat16)
Y = torch.empty((1, 128, 128), device="cuda", dtype=torch.float32)
opus_gemm_a16w16_tune(
    XQ, WQ, Y, bias=None, kernelId=10200, splitK=2
)
torch.cuda.synchronize()
torch.testing.assert_close(
    Y, torch.bmm(XQ.float(), WQ.float().transpose(1, 2)),
    rtol=1e-3, atol=0.05,
)
print("baseline_fresh_build=PASS")
PY
```

### 7.3 Task1 冻结端点

Task1 冻结端点用其当时的 public/raw 名称触发一次 kid 10200，并使用 Torch workspace。调用签名必须
来自冻结源码本身，不允许用当前源码包装一个不匹配的旧 `.so`。日志保存为：

```text
$LOG_DIR/task1_fresh_build.log
```

三次 build 后统一设置：

```bash
export AITER_REBUILD=0
```

## 8. module、目标架构与 ABI 审计

对三个 `.so` 分别执行：

```bash
for item in \
  "baseline:$BASELINE_JIT" \
  "task1:$TASK1_JIT" \
  "current:$CURRENT_JIT"
do
  name="${item%%:*}"
  dir="${item#*:}"
  so="$dir/module_deepgemm_opus.so"
  test -f "$so"
  sha256sum "$so" | tee "$LOG_DIR/${name}_module.sha256"
  /opt/rocm/llvm/bin/llvm-objdump --offloading "$so" \
    > "$LOG_DIR/${name}_offload.txt" 2>&1
  test "$(rg -o 'gfx[[:alnum:]]+' "$LOG_DIR/${name}_offload.txt" | sort -u)" \
    = "gfx942"
done
```

当前端点还必须证明：

- pybind 只暴露当前 A16/A8/BMM typed launch 名称；
- generic `opus_gemm` C++ binding、旧 A16 tune C++ binding、旧 bpreshuffle tune C++ binding 不存在；
- Python public 只有 `aiter.ops.opus.opus_gemm()`；
- C++ gfx942 路径只按最终 kid 查询 launcher，不包含 runtime tuned/shape heuristic。

原始基线保留旧符号是端点定义的一部分，不得拿当前 ABI 规则判它失败。

## 9. 当前最终端点 focused 正确性

```bash
cd "$CURRENT_SRC"
AITER_JIT_DIR="$CURRENT_JIT" AITER_REBUILD=0 \
python3 -m pytest -q -rs \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py \
  op_tests/test_opus_interfaces.py \
  op_tests/test_opus_ctypes.py \
  op_tests/test_gemm_codegen.py \
  2>&1 | tee "$LOG_DIR/current_focused.log"
```

通过要求：

- pytest exit code 为 0；
- `failed = 0`；
- 不得出现“requires gfx942 hardware”的 skip；
- gfx950、gfx1250 或第二张可见 GPU 专属项可以 skip，但必须逐项分类；
- warning 必须保存并判断是否为新问题。

目标架构 skip 审计：

```bash
if rg -n 'SKIPPED .*requires (idle )?gfx942|requires gfx942 hardware' \
  "$LOG_DIR/current_focused.log"
then
  printf 'gfx942 test was skipped; acceptance failed\n' >&2
  exit 1
fi
```

## 10. gfx942 全量正确性

### 10.1 当前 22 个 A16 kid

```bash
cd "$CURRENT_SRC"
OPUS_GFX942_EXHAUSTIVE=1 \
AITER_JIT_DIR="$CURRENT_JIT" AITER_REBUILD=0 \
python3 -m pytest -q -s -rs \
  op_tests/test_opus_gfx942_exhaustive.py \
  2>&1 | tee "$LOG_DIR/current_gfx942_exhaustive.log"
```

最终报告必须明确：

```text
A16 canonical kids: 22/22 passed
workspace kids:      8/8 passed
non-workspace kids: 14/14 passed
target-arch skips:   0
```

对于 5 个 FP32-workspace kid，BF16/FP32 两种 Y 都必须通过；对于 3 个 BF16-workspace kid，
BF16 Y 必须通过且 FP32 Y 必须被明确拒绝。最终报告为 `13/13` 数值通过和 `3/3` 合同拒绝。

### 10.2 Task1 原始基线 workspace 数值

用同一 exhaustive runner 的 baseline 模式，或基线专用兼容 runner，对 8 个 workspace kid × 两种
Y dtype 执行 16 个数值 case：

```bash
cd "$CURRENT_SRC"
OPUS_GFX942_EXHAUSTIVE=1 \
OPUS_GFX942_ENDPOINT=baseline \
PYTHONPATH="$CURRENT_SRC" \
AITER_JIT_DIR="$BASELINE_JIT" AITER_REBUILD=0 \
python3 -m pytest -q -s -rs \
  op_tests/test_opus_gfx942_exhaustive.py \
  2>&1 | tee "$LOG_DIR/baseline_gfx942_workspace_numeric.log"
```

baseline 模式只需执行 Task1 相关的 8 个 workspace kid，但必须是 `13/13` 合法数值 case 通过，
并且 3 个 BF16-workspace kid 的 FP32 Y 为 `3/3` 明确拒绝。runner 未实现该模式时，本项保持
`BLOCKED`。该 runner 应像现有性能 benchmark 一样在当前测试源码中局部声明旧
`opus_gemm_a16w16_tune` binding，再加载 `$BASELINE_JIT`；不能让 pytest 因 cwd/PYTHONPATH 顺序
意外导入错误端点。

### 10.3 gfx942 A8 kid 11000

必须在真实 GPU 上覆盖：

1. 2D 与 batch=1 的 3D raw 数值；
2. unified public：`layout="bpreshuffle"`、FP32 scale、BF16 Y；
3. 临时 tuned CSV 的真实 `libtype=opus, kernelId=11000` 高层路径；
4. Graph capture/replay；
5. 两条 stream 的独立输入与输出；
6. batch、scale dtype/shape、N/K tile、contiguous layout、wrong kid 负例；
7. 跨架构 gfx950/gfx1250 kid 在 gfx942 上正确拒绝。

权重必须由 `aiter.ops.shuffle.shuffle_weight(WQ, layout=(16, 16))` 得到；不能拿未重排 WQ 冒充
bpreshuffle 数值测试。

### 10.4 Task2 tuned / heuristic / fallback

使用临时 CSV，不修改仓内正式 tuned 文件。至少执行：

| 场景 | 预期 |
|---|---|
| 显式最终 kid | 直接调用 unified public，不查 tuned/heuristic |
| 有效 gfx942 tuned row | 使用整条 `(kid, split-K)`，不运行 heuristic |
| 失效 tuned row | 丢弃整条配置，再运行 Python heuristic |
| tuned miss | Python gfx942 heuristic 给出最终 kid，再进入 unified public |
| heuristic 不适配 bias/dtype/shape | PyTorch fallback |
| 10210/10213 非支持 N | 在上层候选解析中转为 10200/10203 |
| 10216 非支持 N | 上层候选失效；最终 public 本身不改 kid |

日志必须打印 requested kid、最终 kid、split-K、选核来源和实际启动结果。

## 11. Graph、双 stream 与跨 device

当前端点至少覆盖：

| 路径 | Graph | 双 stream | caller workspace |
|---|---|---|---|
| A16 kid 10200 FP32 workspace | 必须 | 必须 | 必须 |
| A16 kid 10210 BF16 workspace | 必须 | 必须 | 必须 |
| A8 kid 11000 | 必须 | 必须 | 不适用 |

正确性测试同时覆盖自动 workspace；正式性能统一预先创建并传入 workspace，避免把 Python allocation
混入 kernel/route A/B。

若节点有两张空闲 gfx942，可增加跨 device 错误 Tensor：

```bash
HIP_VISIBLE_DEVICES=0,1 GPU_ARCHS=gfx942 \
AITER_JIT_DIR="$CURRENT_JIT" AITER_REBUILD=0 \
python3 -m pytest -q -rs \
  op_tests/test_opus_workspace.py::test_raw_cpp_rejects_workspace_on_another_device \
  2>&1 | tee "$LOG_DIR/cross_device_workspace.log"
```

单卡时记录 `NOT RUN`，不影响同卡 Graph/双 stream，但不能写成跨 device 已通过。

## 12. Task1 性能：原始基线与 Torch workspace

### 12.1 测量口径

- 5 个 FP32-workspace kid × BF16/FP32 Y，加 3 个 BF16-workspace kid × BF16 Y，
  共 13 个合法数值 case；
- 每个 case：warmup 20、9 rounds、每 round 100 次；
- Eager 与 Graph 分开；
- 固定同一 CPU core；
- 两端输入值、shape、stride、kid、split-K、输出 dtype 完全相同；
- 原始基线在 capture 前调用旧 `opus_gemm_workspace_init()` 并完成所需扩容；
- 当前端点预先创建并传入正确 shape/dtype 的 Torch workspace；
- 正式顺序固定为 `baseline A1 → current B1 → current B2 → baseline A2`。

### 12.2 raw/C ABI 对比

```bash
export TASK1_BENCH=op_tests/bench_opus_gfx950_workspace_ab.py

taskset -c "$CPU_CORE" env \
  OPUS_BENCH_SOURCE_ROOT="$BASELINE_SRC" \
  PYTHONPATH="$BASELINE_SRC" \
  AITER_JIT_DIR="$BASELINE_JIT" AITER_REBUILD=0 \
  python3 "$CURRENT_SRC/$TASK1_BENCH" \
    --arch gfx942 --endpoint baseline --pass-id A1 \
    --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$LOG_DIR/task1_raw_baseline_A1.log"

taskset -c "$CPU_CORE" env \
  OPUS_BENCH_SOURCE_ROOT="$CURRENT_SRC" \
  PYTHONPATH="$CURRENT_SRC" \
  AITER_JIT_DIR="$CURRENT_JIT" AITER_REBUILD=0 \
  python3 "$CURRENT_SRC/$TASK1_BENCH" \
    --arch gfx942 --endpoint ctypes --pass-id B1 \
    --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$LOG_DIR/task1_raw_current_B1.log"

taskset -c "$CPU_CORE" env \
  OPUS_BENCH_SOURCE_ROOT="$CURRENT_SRC" \
  PYTHONPATH="$CURRENT_SRC" \
  AITER_JIT_DIR="$CURRENT_JIT" AITER_REBUILD=0 \
  python3 "$CURRENT_SRC/$TASK1_BENCH" \
    --arch gfx942 --endpoint ctypes --pass-id B2 \
    --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$LOG_DIR/task1_raw_current_B2.log"

taskset -c "$CPU_CORE" env \
  OPUS_BENCH_SOURCE_ROOT="$BASELINE_SRC" \
  PYTHONPATH="$BASELINE_SRC" \
  AITER_JIT_DIR="$BASELINE_JIT" AITER_REBUILD=0 \
  python3 "$CURRENT_SRC/$TASK1_BENCH" \
    --arch gfx942 --endpoint baseline --pass-id A2 \
    --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$LOG_DIR/task1_raw_baseline_A2.log"
```

### 12.3 当前生产 A16 私有启动对比

再独立执行一轮 `baseline A1 → family B1 → family B2 → baseline A2`，其中当前端点使用
`--endpoint family`。不要把 raw/C ABI 和包含 Python kid/shape/workspace 计划的私有启动结果混成一
张表。

### 12.4 Task1 输出表

每台机器分别填写：

| 路径 | Eager 13-case 总和 | Graph 13-case 总和 | 相对原始基线 |
|---|---:|---:|---:|
| 原始基线：C++ 内部 workspace | | | 基准 |
| Task1 raw：Torch workspace pointer → A16 C 接口 | | | |
| Task1 当前私有 A16 启动 | | | |

同时保存 13 个逐 case 结果和 3 个 FP32 Y 拒绝结果。总和提升不能掩盖单个大幅退化；逐 case
的快/慢数量、最大退化和对应
kid/dtype 必须报告。

## 13. Task2 性能：私有启动与统一 public

### 13.1 A16 全 workspace 路由成本

在同一个当前 module 上执行：

```text
private A1 → public B1 → public B2 → private A2
```

private 使用 `_launch_a16w16()`；public 使用 `aiter.ops.opus.opus_gemm()`。两端都传同一个 caller
workspace，不运行 tuned/heuristic。这样只测统一 public 的 Python kid → kernel 类型路由成本。

仍使用 13 个合法 A16 数值 case，输出：

| 路径 | Eager 13-case 总和 | Graph 13-case 总和 | 相对私有 A16 启动 |
|---|---:|---:|---:|
| Task1：私有 A16 启动 | | | 基准 |
| Task2：统一 public `opus_gemm()` | | | |

### 13.2 A16/A8 分层性能

参数化后的 `bench_opus_task1_task2_interfaces.py` 按下列顺序运行：

```bash
export TASK2_BENCH=op_tests/bench_opus_task1_task2_interfaces.py

# 冻结 Task1 A1
taskset -c "$CPU_CORE" env \
  PYTHONPATH="$CURRENT_SRC" \
  AITER_JIT_DIR="$TASK1_JIT" AITER_REBUILD=0 \
  python3 "$CURRENT_SRC/$TASK2_BENCH" \
    --arch gfx942 --endpoint task1 --pass-id A1 \
    --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$LOG_DIR/task2_task1_A1.log"

# 当前 Task2 B1、B2
for pass_id in B1 B2; do
  taskset -c "$CPU_CORE" env \
    PYTHONPATH="$CURRENT_SRC" \
    AITER_JIT_DIR="$CURRENT_JIT" AITER_REBUILD=0 \
    python3 "$CURRENT_SRC/$TASK2_BENCH" \
      --arch gfx942 --endpoint current --pass-id "$pass_id" \
      --warmup 20 --rounds 9 --iters 100 \
    2>&1 | tee "$LOG_DIR/task2_current_${pass_id}.log"
done

# 冻结 Task1 A2
taskset -c "$CPU_CORE" env \
  PYTHONPATH="$CURRENT_SRC" \
  AITER_JIT_DIR="$TASK1_JIT" AITER_REBUILD=0 \
  python3 "$CURRENT_SRC/$TASK2_BENCH" \
    --arch gfx942 --endpoint task1 --pass-id A2 \
    --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$LOG_DIR/task2_task1_A2.log"
```

每个端点必须对 A16 kid 10200、10210 和 A8 kid 11000 做数值断言后才输出性能。A8 必须使用真实
shuffle 后的 WQ 和相同 scale；不能只计一个空函数或 capability error。

该分层 benchmark 始终使用当前测试源码，并通过不同的 `AITER_JIT_DIR` 加载冻结 Task1 或当前
module；历史 binding 已在 benchmark 内局部声明。`TASK1_FROZEN_SRC` 只用于在目标节点构建并证明
`$TASK1_JIT` 的来源，不在计时进程中与当前 Python 文件混合导入。

### 13.3 性能判定

MI308、MI300 不互相作为性能基线。每台机器分别用 A1/A2 与 B1/B2 的重复漂移建立本机噪声范围：

1. direct C++、raw 和 Graph 若超过本机重复漂移，必须定位；
2. public/private 差异只归因于 Python route，不能混入不同 workspace、kid 或 kernel；
3. 数值断言失败的计时全部作废；
4. 不能直接套用 gfx950 的噪声阈值或性能百分比；
5. 不通过删除检查、恢复 generic C++ lookup 或改 kid 来“修复”性能结果。

## 14. MI308 与 MI300 结果模板

每台机器复制一份填写。

### 14.1 环境

| 字段 | 值 |
|---|---|
| 机器 | MI308 / MI300 |
| UTC 时间 | |
| GPU 名称 / `gcnArchName` / CU | |
| 物理 GPU / 进程内 GPU | |
| ROCm / hipcc | |
| Python / PyTorch / HIP | |
| CPU core | |
| 当前 HEAD / patch SHA-256 | |
| 原始基线 SHA-256 | |
| Task1 冻结端点 SHA-256 | |
| current `.so` SHA-256 | |
| baseline `.so` SHA-256 | |
| Task1 `.so` SHA-256 | |
| sidecar SHA-256 | |

### 14.2 正确性

| 项目 | MI308 | MI300 | 日志 |
|---|---|---|---|
| runtime/build/offload 都为 gfx942 | | | |
| 当前 focused，0 failed、0 gfx942 skip | | | |
| 当前 A16 22/22 | | | |
| workspace kid 8/8，合法数值 13/13、拒绝 3/3 | | | |
| baseline workspace 数值 13/13、拒绝 3/3 | | | |
| A8 kid 11000 raw/public/tuned | | | |
| Python tuned → heuristic → fallback | | | |
| A16 Graph / 双 stream | | | |
| A8 Graph / 双 stream | | | |
| 跨 device 负例 | | | 单卡可写 NOT RUN |

### 14.3 Task1 性能

| 机器 | 路径 | Eager 13-case 总和 | Graph 13-case 总和 | 相对基线 |
|---|---|---:|---:|---:|
| MI308 | 原始基线 | | | 基准 |
| MI308 | Task1 raw | | | |
| MI308 | Task1 私有 A16 启动 | | | |
| MI300 | 原始基线 | | | 基准 |
| MI300 | Task1 raw | | | |
| MI300 | Task1 私有 A16 启动 | | | |

### 14.4 Task2 性能

| 机器 | 路径 | Eager 13-case 总和 | Graph 13-case 总和 | 相对私有启动 |
|---|---|---:|---:|---:|
| MI308 | Task1 私有 A16 启动 | | | 基准 |
| MI308 | Task2 unified public | | | |
| MI300 | Task1 私有 A16 启动 | | | 基准 |
| MI300 | Task2 unified public | | | |

A8 kid 11000 的 high-level、raw、direct、Graph 另列逐层表，不与 A16 13-case 总和相加。

## 15. 日志封存

每台机器完成后执行：

```bash
/opt/rocm/bin/rocm-smi --showproductname --showuse --showmemuse --showpids \
  2>&1 | tee "$LOG_DIR/gpu_after.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "$LOG_DIR/end_utc.txt"

find "$LOG_DIR" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$LOG_DIR/SHA256SUMS"
```

保留整个 `$VALIDATION_ROOT`，直到两台机器结果都已复核并写入 Task1/Task2 checkpoint。失败日志也不
删除；修复后使用新的 JIT 与结果目录重跑，不覆盖原失败证据。

## 16. Definition of Done

只有以下项目全部满足，才能写“Task1 和 Task2 已在 MI308/MI300 的 gfx942 上完成验证”：

- [ ] MI308 runtime arch、build arch、offload bundle 均为 gfx942；
- [ ] MI300 runtime arch、build arch、offload bundle 均为 gfx942；
- [ ] 两台机器各自 fresh build 原始基线、Task1 冻结和当前最终端点；
- [ ] 当前端点 focused suite 为 0 failed，且没有 gfx942 测试被 skip；
- [ ] 两台机器当前 A16 均为 22/22，workspace/non-workspace 数量不减少；
- [ ] 两台机器原始基线的 8 个 workspace kid 均为合法数值 13/13、FP32 Y 拒绝 3/3；
- [ ] kid 11000 的 raw、unified public、真实 tuned 高层、Graph、双 stream均通过；
- [ ] Python tuned、invalid tuned、heuristic、fallback 和 gfx942 redirect policy 均在实机通过；
- [ ] Task1 的 raw 与私有 A16 两组 ABBA 均完成，数值正确且无未解释退化；
- [ ] Task2 的 private/public ABBA 和 A16/A8 分层性能均完成；
- [ ] MI308、MI300 的结果、漂移和结论分别报告，不互相替代；
- [ ] 源码、module、sidecar、日志 SHA-256 与所有 skip 分类完整；
- [ ] `task1_checkpoint.md`、`task2_checkpoint.md` 及最终 HTML 按实测结果更新。

任何一个节点只完成 codegen、只跑代表 kid、缺少原始/Task1 端点、目标测试被 skip，或性能没有
ABBA，都不能签署完整通过。

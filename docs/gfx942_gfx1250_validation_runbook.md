# OPUS GEMM Task1/Task2 gfx942、gfx1250 实机验证 Runbook

更新时间：`2026-08-12 UTC`

状态：**可作为目标节点执行手册；gfx942/gfx1250 实机项目目前仍是“未执行”**。

本文用于把当前 Task1 + Task2 累计工作树带到 gfx942 或 gfx1250 节点，完成实机正确性、
workspace、graph、并发、ABI 和性能验收。本文不会把 codegen、交叉编译、CPU fixture 或 pytest
skip 当成目标架构实机通过。

权威验收口径来自：

- [opus_gemm_two_tasks_final_plan.md](opus_gemm_two_tasks_final_plan.md) 的 B7.4、B7.5、B7.6 和
  Definition of Done；
- [task2_step_b7_process.md](task2_step_b7_process.md) 的 gfx950 已执行证据以及
  gfx942/gfx1250 “未执行”矩阵；
- [task1_checkpoint.md](task1_checkpoint.md) 和 [task2_checkpoint.md](task2_checkpoint.md) 的
  当前 checked-only workspace 与性能边界。

## 1. 使用结论和当前边界

本 runbook 分为三个门：

1. **现有自动化门**：当前仓库已有测试，可以直接在目标 GPU 上运行；
2. **补充实机门**：总计划要求、但当前仓库还没有完整 GPU test node 的项目；
3. **性能门**：必须具备同架构 Task1 冻结端点和 Task2 当前端点后执行 ABBA。

只有三个门全部通过，才能把对应架构从“未执行”改成“通过”。仅运行第一个门，结论只能写成
“现有自动化通过，完整验收未完成”。

当前状态如下：

| 架构 | 已有非实机证据 | 已有实机自动化节点 | 当前缺口 |
|---|---|---|---|
| gfx942 | full codegen、对象编译、ABI、registry/typed table | A16 10200/10210 数值与 typed workspace、workspace 负例、A16 graph/双 stream、bias、A8 kid 11000 raw 数值与 7 类负例 | public `kid=None` 真实 default、真实 tuned OPUS 高层路由、10210/10213 redirect 与 10216 拒绝的实机保持性、A8 graph/双 stream、性能 |
| gfx1250 | full codegen、对象编译、ABI、496 two-stage + 1378 fused 数量/dtype | two-stage kid 20000 BF16/FP32 数值、typed workspace、batch 拒绝、bias、two-stage graph/双 stream | fused BF16/FP32 数值、caller workspace、fused graph/双 stream、compile-time split-K 实机保持性、高层 FlyDSL/Triton/Gluon 路由、OPUS 空 capability 实机错误、性能 |

代表性 kid 固定为：

| 架构/family | kid | 物理合同 |
|---|---:|---|
| gfx942 A16 FP32 workspace | 10200 | two-stage，FP32 workspace |
| gfx942 A16 BF16 workspace | 10210 | exact-N 时 BF16 workspace；非 exact-N redirect 到 10200 |
| gfx942 A16 redirect pair | 10213 -> 10203 | 非 exact-N 在分配 workspace 前 redirect |
| gfx942 A16 exact-N only | 10216 | 非 exact-N 必须拒绝 |
| gfx942 A8 bpreshuffle | 11000 | FP8 XQ/WQ、FP32 scales、BF16 Y、128x128 block |
| gfx1250 A16 two-stage | 20000 | BF16 workspace，batch=1 |
| gfx1250 A16 fused BF16 workspace | 21000 | tile-major workspace，compile-time split-K=2 |
| gfx1250 A16 fused FP32 workspace | 21030 | tile-major workspace，compile-time split-K=2 |

## 2. 节点和安全前置条件

### 2.1 必需条件

- ROCm、PyTorch 和编译器版本与待比较端点兼容；
- 至少一张空闲的目标架构 GPU；
- 仓库位于 `/root/workspace/0810/aiter`，或把下文 `REPO_ROOT` 改成实际绝对路径；
- 有足够的 `/tmp` 空间用于独立 JIT build 和日志；
- 性能门还需要目标架构的 Task1 冻结 `.so`/源码端点，详见第 10 节。

一张卡足以运行核心数值、graph capture/replay 和双 stream。第二张可见的同架构卡只用于
workspace/scale 跨 device 负例；没有第二张卡时必须记录该负例为 `NOT RUN`，不能影响其他核心项。

### 2.2 禁止事项

- 不在当前 dirty 工作树执行 `git reset --hard`、`git checkout --`、`git clean`；
- 不复用 gfx950、gfx942、gfx1250 之间的 `AITER_JIT_DIR`；
- 不让 Task1 和 Task2 性能端点共用一个 JIT 目录；
- 不在第一次 build 后继续使用 `AITER_REBUILD=1` 跑回归或性能；
- 不终止不属于本次验证的 GPU/KFD、pytest、编译或其他用户进程；
- 不恢复已经回退的 prepared/prevalidated launcher 或 workspace cache；
- 不把目标架构不匹配、pytest skip 或 backend 依赖缺失写成通过。

如果目标 GPU 正在被占用，停止本轮并换卡或等待；不要通过 kill 未知进程获得空闲卡。

## 3. 每个架构使用一个全新 shell

gfx942 和 gfx1250 必须分别从全新 shell 执行本节，且所有环境变量必须在第一次 Python import
之前设置。

下面以 gfx942 为例。验证 gfx1250 时只把 `VALIDATION_ARCH` 改为 `gfx1250`：

```bash
set -euo pipefail

export REPO_ROOT=/root/workspace/0810/aiter
export VALIDATION_ARCH=gfx942
export PHYSICAL_GPU_SLOT=0

cd "$REPO_ROOT"

case "$VALIDATION_ARCH" in
  gfx942|gfx1250) ;;
  *) printf 'unsupported VALIDATION_ARCH=%s\n' "$VALIDATION_ARCH" >&2; exit 2 ;;
esac

export HIP_VISIBLE_DEVICES="$PHYSICAL_GPU_SLOT"
export GPU_ARCHS="$VALIDATION_ARCH"
export AITER_JIT_DIR
AITER_JIT_DIR="$(mktemp -d "/tmp/aiter-${VALIDATION_ARCH}-task12-validation.XXXXXX")"
export VALIDATION_LOG_DIR="$AITER_JIT_DIR/logs"

mkdir -p "$AITER_JIT_DIR/build" "$VALIDATION_LOG_DIR"
printf 'AITER_JIT_DIR=%s\n' "$AITER_JIT_DIR"
```

不要在同一个 Python 进程中途修改 `HIP_VISIBLE_DEVICES`、`GPU_ARCHS`、`AITER_JIT_DIR` 或
`AITER_REBUILD`；AITER 的部分 JIT 状态在 import 时确定。

## 4. 冻结源码、软件和硬件证据

先记录本轮实际验证的源状态。当前工作树含未提交和未跟踪文件，因此只记录 commit SHA 不够：

```bash
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "$VALIDATION_LOG_DIR/start_utc.txt"
git rev-parse HEAD | tee "$VALIDATION_LOG_DIR/git_head.txt"
git status --short | tee "$VALIDATION_LOG_DIR/git_status.txt"
git diff --check | tee "$VALIDATION_LOG_DIR/git_diff_check.txt"
git diff --binary > "$VALIDATION_LOG_DIR/tracked_changes.patch"
git ls-files --others --exclude-standard -z \
  | sort -z \
  | xargs -0 -r sha256sum \
  > "$VALIDATION_LOG_DIR/untracked_files.sha256"
sha256sum "$VALIDATION_LOG_DIR/tracked_changes.patch" \
  | tee "$VALIDATION_LOG_DIR/tracked_changes.sha256"

python3 --version 2>&1 | tee "$VALIDATION_LOG_DIR/python_version.txt"
/opt/rocm/bin/hipcc --version 2>&1 | tee "$VALIDATION_LOG_DIR/hipcc_version.txt"
/opt/rocm/bin/rocm-smi --showproductname --showuse --showmemuse --showpids \
  2>&1 | tee "$VALIDATION_LOG_DIR/gpu_before.txt"
```

随后从 PyTorch 读取**真实运行架构**，不能只相信 `GPU_ARCHS`：

```bash
python3 - <<'PY' | tee "$VALIDATION_LOG_DIR/runtime_arch.txt"
import os
import torch

assert torch.cuda.is_available(), "ROCm GPU is not visible"
assert torch.cuda.device_count() >= 1, "no visible ROCm device"

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
runtime_arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()
expected_arch = os.environ["VALIDATION_ARCH"].lower()

print(f"torch={torch.__version__}")
print(f"torch.version.hip={torch.version.hip}")
print(f"visible_device_count={torch.cuda.device_count()}")
print(f"visible_device={device}")
print(f"device_name={props.name}")
print(f"gcnArchName={getattr(props, 'gcnArchName', None)}")
print(f"multi_processor_count={props.multi_processor_count}")
print(f"expected_arch={expected_arch}")

assert runtime_arch == expected_arch, (runtime_arch, expected_arch)
assert os.environ["GPU_ARCHS"].lower() == expected_arch
PY
```

任何断言失败都应停止，不允许在 gfx950 或错误目标模块上继续形成 gfx942/gfx1250 结论。

## 5. 准备 subset sidecar 并 fresh build

### 5.1 写入本轮必需 kid

sidecar 位于 `$AITER_JIT_DIR/build/compiled_kids_opus.json`。下列集合是本 runbook 的最小实机集合；
生成器还会按现有公式并入 tuned CSV、Python heuristic defaults 和 mandatory A8 kids：

```bash
case "$VALIDATION_ARCH" in
  gfx942)
    VALIDATION_KIDS='[10200, 10203, 10210, 10213, 10216, 11000]'
    ;;
  gfx1250)
    VALIDATION_KIDS='[20000, 21000, 21030]'
    ;;
esac

printf '%s\n' "$VALIDATION_KIDS" \
  > "$AITER_JIT_DIR/build/compiled_kids_opus.json"
sha256sum "$AITER_JIT_DIR/build/compiled_kids_opus.json" \
  | tee "$VALIDATION_LOG_DIR/sidecar_input.sha256"
```

### 5.2 用一次真实 canonical launch 触发 fresh build

只在这一进程设置 `AITER_REBUILD=1`。该 smoke 同时验证构建产物能够在真实目标卡启动：

```bash
export AITER_REBUILD=1

python3 - <<'PY' 2>&1 | tee "$VALIDATION_LOG_DIR/fresh_build_smoke.log"
import importlib
import os
import pathlib

import torch

from aiter.ops.opus.gemm_op_a16w16 import gemm_a16w16_opus

arch = os.environ["VALIDATION_ARCH"]
torch.manual_seed(0xA17E2)

if arch == "gfx942":
    kid, m, n, k, split_k, out_dtype = 10200, 128, 128, 512, 2, torch.float32
    rtol, atol = 1e-3, 0.05
elif arch == "gfx1250":
    kid, m, n, k, split_k, out_dtype = 20000, 16, 32, 512, 2, torch.bfloat16
    rtol, atol = 0.03, 0.5
else:
    raise AssertionError(arch)

a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
y = gemm_a16w16_opus(
    a,
    b,
    dtype=out_dtype,
    kernelId=kid,
    splitK=split_k,
)
torch.cuda.synchronize()
torch.testing.assert_close(y.float(), a.float() @ b.float().T, rtol=rtol, atol=atol)

module = importlib.import_module("module_deepgemm_opus")
print(f"arch={arch}")
print(f"kid={kid}")
print(f"module={pathlib.Path(module.__file__).resolve()}")
print("fresh_build_smoke=PASS")
PY

export AITER_REBUILD=0
```

从此以后，本 runbook 的所有命令都必须复用同一 `.so`。如果 fresh build 失败，保留目录和日志，
修复根因后使用另一个全新 `AITER_JIT_DIR` 重来；不要清理失败证据后原地猜测。

### 5.3 审计生成集合

```bash
python3 - <<'PY' | tee "$VALIDATION_LOG_DIR/sidecar_final.txt"
import json
import os
import pathlib

path = pathlib.Path(os.environ["AITER_JIT_DIR"]) / "build" / "compiled_kids_opus.json"
actual = {int(value) for value in json.loads(path.read_text())}
required = {
    "gfx942": {10200, 10203, 10210, 10213, 10216, 11000},
    "gfx1250": {20000, 21000, 21030},
}[os.environ["VALIDATION_ARCH"]]

print(f"path={path}")
print(f"count={len(actual)}")
print(f"kids={sorted(actual)}")
print(f"required={sorted(required)}")
assert required <= actual, sorted(required - actual)
PY

sha256sum "$AITER_JIT_DIR/build/compiled_kids_opus.json" \
  | tee "$VALIDATION_LOG_DIR/sidecar_final.sha256"
```

## 6. `.so`、offload arch 和 ABI 审计

### 6.1 文件与目标架构

```bash
export OPUS_SO="$AITER_JIT_DIR/module_deepgemm_opus.so"
test -f "$OPUS_SO"
sha256sum "$OPUS_SO" | tee "$VALIDATION_LOG_DIR/module.sha256"
stat "$OPUS_SO" | tee "$VALIDATION_LOG_DIR/module.stat.txt"

OFFLOAD_ARCHES="$(
  /opt/rocm/llvm/bin/llvm-objdump --offloading "$OPUS_SO" 2>&1 \
    | rg -o 'gfx[[:alnum:]]+' \
    | sort -u
)"
printf '%s\n' "$OFFLOAD_ARCHES" | tee "$VALIDATION_LOG_DIR/offload_arches.txt"
test "$OFFLOAD_ARCHES" = "$VALIDATION_ARCH"

ARCH_HEADER="$(
  find "$AITER_JIT_DIR/build/module_deepgemm_opus" \
    -type f -name opus_build_archs.h -print -quit
)"
test -n "$ARCH_HEADER"
rg '^#define OPUS_BUILD_HAS_' "$ARCH_HEADER" \
  | tee "$VALIDATION_LOG_DIR/opus_build_archs.txt"
test "$(rg '^#define OPUS_BUILD_HAS_' "$ARCH_HEADER")" \
  = "#define OPUS_BUILD_HAS_${VALIDATION_ARCH^^} 1"
```

offload arch 多于一个、为空或不是目标架构都属于失败，不能继续跑性能。

### 6.2 动态 pybind/Python ABI

```bash
python3 - <<'PY' | tee "$VALIDATION_LOG_DIR/abi_dynamic.txt"
import importlib

import aiter  # 将 AITER_JIT_DIR 加入当前进程的 module search path

module = importlib.import_module("module_deepgemm_opus")
expected = {
    "opus_gemm_a16w16_launch",
    "opus_gemm_a8w8_launch",
    "opus_gemm_a8w8_blockscale_launch",
    "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
}
business = {name for name in dir(module) if name.startswith("opus_")}
removed = {
    "opus_gemm",
    "opus_gemm_a16w16_tune",
    "opus_gemm_a8w8_blockscale_bpreshuffle_tune",
}

print(f"module={module.__file__}")
print(f"pybind_business_attrs={sorted(business)}")
assert business == expected, (business, expected)
assert not (business & removed), business & removed

opus = importlib.import_module("aiter.ops.opus")
for name in expected:
    assert hasattr(opus, name), name

# 旧名只允许作为 Python deprecated adapter；不能回到 C++/pybind。
assert hasattr(opus, "opus_gemm_a16w16_tune")
assert hasattr(opus, "opus_gemm_a8w8_blockscale_bpreshuffle_tune")

implementation = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
assert hasattr(
    implementation,
    "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
)
assert not hasattr(
    implementation,
    "_opus_gemm_a8w8_blockscale_bpreshuffle_tune_raw",
)
print("abi_dynamic=PASS")
PY

nm -C --defined-only "$OPUS_SO" > "$VALIDATION_LOG_DIR/module.nm.txt"
if rg -n 'opus_gemm_a16w16_tune|opus_gemm_a8w8_blockscale_bpreshuffle_tune' \
  "$VALIDATION_LOG_DIR/module.nm.txt"; then
  printf 'removed C++ tune symbol is still present\n' >&2
  exit 1
fi
```

## 7. 运行当前已有的完整 focused suite

先确认 `AITER_REBUILD=0`，再执行：

```bash
test "$AITER_REBUILD" = 0

python3 -m pytest -q -rs \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_a16w16_gemm.py \
  op_tests/test_opus_interfaces.py \
  2>&1 | tee "$VALIDATION_LOG_DIR/focused.log"

python3 -m pytest -q -rs op_tests/test_gemm_codegen.py \
  2>&1 | tee "$VALIDATION_LOG_DIR/gemm_codegen.log"
```

通过要求：

- 两条命令退出码均为 0；
- `failed`、unexpected `error` 为 0；
- 日志中不得有“requires 当前目标架构”的 skip；
- 另一架构、gfx950-only 和第二张 GPU 的条件性 skip 可以存在，但必须逐项记录原因；
- warning 必须分类，不能仅凭“测试通过”忽略新的 runtime/编译 warning。

可用下面的命令快速检查目标架构是否仍被跳过；有输出即停止并查明 arch 检测或模块复用错误：

```bash
if rg -n "SKIPPED .*requires (idle )?${VALIDATION_ARCH}" \
  "$VALIDATION_LOG_DIR/focused.log"; then
  printf 'target-architecture tests were skipped\n' >&2
  exit 1
fi
```

### 7.1 gfx942 已有自动化实际覆盖

focused suite 在 gfx942 上应真实执行：

- A16 kid 10200 FP32、10210 BF16 对 Torch golden；
- raw typed workspace：10200 为 FP32、10210 为 BF16；
- workspace 缺失、错 dtype、短容量、non-contiguous、alignment；
- A16 kid 10200 graph capture/replay 和两个独立 stream/call-scoped workspace；
- bias 的 framework fallback 与 explicit split-K rejection；
- A8 kid 11000 raw 2D/3D 数值；
- A8 batch、scale dtype/shape、N/K exact tile、layout、wrong-kid 七类负例。

### 7.2 gfx1250 已有自动化实际覆盖

focused suite 在 gfx1250 上应真实执行：

- two-stage kid 20000 的 BF16/FP32 Y 数值；
- raw BF16 typed workspace；
- public/selector 与 raw 的 batch>1 拒绝；
- BF16 Y + FP32 bias 数值；
- two-stage kid 20000 graph capture/replay 和两个独立 stream/call-scoped workspace。

registry、fused workspace shape/dtype 和 selector 的 pytest 通过仍是 CPU 合同证据，不等于 fused
kernel 已经在 gfx1250 上运行。

## 8. 完整签字前必须补齐的 GPU test node

这一节是**硬门**。当前仓库还没有覆盖以下全部实机行为的节点，所以仅完成第 7 节不能签署
gfx942/gfx1250 完整通过。

建议把节点集中新增到：

```text
op_tests/test_opus_gfx942_gfx1250_validation.py
```

生产代码不应为了测试而增加分支。测试必须使用当前 canonical public/raw 接口、固定 seed、Torch
golden，并在调用后 `torch.cuda.synchronize()`。

### 8.1 gfx942 必需新增节点

| 建议 node | 必须证明的内容 |
|---|---|
| `test_gfx942_a16_redirects_and_10216_rejection` | public explicit 10210 在非 exact-N 实际 launch 10200；10213 实际 launch 10203；workspace 按 actual kid 准备；10216 在同类非 exact-N 输入上 launch 前拒绝；成功 case 对 Torch golden |
| `test_gfx942_bpreshuffle_public_none_and_real_tuned_route` | canonical public `kid=None` 在无匹配 tuned row时解析 11000 并得到正确数值；再用临时 tuned CSV 的真实 `libtype=opus,kernelId=11000` 行，从高层 `gemm_a8w8_blockscale_bpreshuffle` 到 canonical raw，不能用纯 fake backend 代替 |
| `test_gfx942_bpreshuffle_graph_replay_and_two_streams` | kid 11000 public canonical wrapper 可 capture/replay；两条 stream 使用独立输入/输出并都与 golden 一致；不能只复用 A16 graph 结论 |

临时 tuned CSV 必须写到 `tmp_path`，其 `gfx`、`cu_num`、M/N/K 与真实节点匹配，并通过
`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` 传入；不得修改仓内正式 CSV。测试日志应打印解析到的
`libtype=opus` 和 `kernelId=11000`。

### 8.2 gfx1250 必需新增节点

| 建议 node | 必须证明的内容 |
|---|---|
| `test_gfx1250_fused_bf16_fp32_numerics_and_workspace_reuse` | kid 21000 与 21030 分别使用 BF16/FP32 tile-major workspace；BF16/FP32 Y 至少各一例；caller workspace 地址被实际复用；结果对 Torch golden |
| `test_gfx1250_fused_exact_split_graph_and_two_streams` | 同一 exact fused kid 对不同 runtime `splitK` 请求仍使用 kid 内 compile-time split-K；graph capture/replay 和双 stream 均执行 fused kernel并数值正确；workspace 不是 two-stage layout |
| `test_gfx1250_high_level_bpreshuffle_backends_and_empty_opus_capability` | 可用的 FP8-E8M0 128-block FlyDSL 高层路径，以及 FP32-scale Triton/Gluon 高层路径实际运行且不进入 OPUS raw；直接调用预留 OPUS bpreshuffle public `kid=None` 明确报告 gfx1250 no registered kernel |

如果目标镜像缺少 FlyDSL、Triton 或 Gluon 的必要依赖，该高层能力项应记录 `BLOCKED/NOT RUN`，不能
用现有 monkeypatch 路由 fixture 替代实机 backend。可以先完成 OPUS A16/fused 核心验收，但不能把
整架构写成完整通过。

### 8.3 补齐后执行门

上述文件和节点进入工作树后，先确认 pytest 能收集全部 6 个节点，再执行：

```bash
python3 -m pytest --collect-only -q \
  op_tests/test_opus_gfx942_gfx1250_validation.py \
  2>&1 | tee "$VALIDATION_LOG_DIR/supplemental_collect.log"

python3 -m pytest -q -rs \
  op_tests/test_opus_gfx942_gfx1250_validation.py \
  2>&1 | tee "$VALIDATION_LOG_DIR/supplemental_gpu.log"
```

当前该文件尚不存在；这是有意保留的显式完成门，不是可忽略的命令错误。

## 9. 可选的双卡跨 device 负例

若节点有两张可见且相同架构的 GPU，在一个新 shell 中把两张卡都暴露，再复用同一架构的 `.so`：

```bash
export HIP_VISIBLE_DEVICES=0,1
export GPU_ARCHS="$VALIDATION_ARCH"
export AITER_REBUILD=0

python3 -m pytest -q -rs \
  op_tests/test_opus_workspace.py::test_raw_cpp_rejects_workspace_on_another_device \
  2>&1 | tee "$VALIDATION_LOG_DIR/cross_device_workspace.log"
```

gfx942 blockscale scale 跨 device 负例若随第 8 节补充节点实现，也应在此执行。两个可见设备的
`gcnArchName` 必须先逐卡断言与 `VALIDATION_ARCH` 相同。

## 10. 性能门：先准备两个可重建端点

### 10.1 当前不能直接运行现有 benchmark

现有 [bench_opus_task1_task2_interfaces.py](../op_tests/bench_opus_task1_task2_interfaces.py) 目前：

- 在运行入口明确要求 `arch == "gfx950"`；
- case 固定为 gfx950 A16 kid 200、A8 kid 1/2；
- 不包含 gfx942 kid 11000 或 gfx1250 fused family。

因此在 gfx942/gfx1250 上直接运行它会报错，不能通过删除 arch guard 后继续使用 gfx950 case。
验收前必须参数化该脚本，或新增：

```text
op_tests/bench_opus_task1_task2_multiarch.py
```

当前累计工作树也只能构建 Task2 最终 ABI，不能单独重建 Task1 的旧 C++ ABI。性能门开始前必须
提供以下两个互相隔离的端点：

| 端点 | 要求 |
|---|---|
| Task1 | Task1 最终、Task2 B1 之前的源码快照或对应目标架构 `.so`；包含旧 `opus_gemm_a16w16_tune`，gfx942 还包含旧 bpreshuffle tune；记录源码/模块 SHA-256 |
| Task2 | 本 runbook 第 5 节 fresh build 的当前模块；只含四条 canonical family launch |

不要在当前 dirty 工作树 checkout 到旧版本。若需要重建 Task1，应使用独立 clone/worktree/容器，
并先明确 Task1 source endpoint；没有可证明的 Task1 端点时，性能状态保持 `NOT RUN`。

### 10.2 多架构 benchmark 的必需 case

| 架构 | 必测 family/case |
|---|---|
| gfx942 | A16 kid 10200 FP32 workspace；A16 kid 10210 BF16 workspace；A8 bpreshuffle kid 11000 |
| gfx1250 | two-stage kid 20000；fused BF16-workspace kid 21000；fused FP32-workspace kid 21030 |

每个可执行 case 必须保持两端相同：

- 输入 tensor 数值、shape、stride 和 dtype；
- actual kid，而不是只记录 requested kid；
- resolved/compile-time split-K；
- caller workspace shape、dtype、alignment；
- compiler、ROCm、GPU、时钟/功耗策略；
- sidecar 中对应 kid 均已编译。

分别记录：

1. high-level/public Python；
2. explicit canonical/旧 Task1 等价接口；
3. `compile_ops` raw；
4. direct pybind/C++（预转换 tensor handle）；
5. graph replay。

gfx942/gfx1250 没有注册的 OPUS A8 family必须记录 `unavailable`，不得伪造 kernel。gfx1250 的
FlyDSL/Triton/Gluon 是高层 backend 保持性项目，应单独计时并确认两端选择同一 backend。

### 10.3 ABBA 正式顺序

多架构 benchmark 实现并校验后，设置两个绝对路径：

```bash
export TASK1_JIT_DIR=/absolute/path/to/frozen-task1-jit
export TASK2_JIT_DIR="$AITER_JIT_DIR"
export MULTIARCH_BENCH=op_tests/bench_opus_task1_task2_multiarch.py

test -f "$TASK1_JIT_DIR/module_deepgemm_opus.so"
test -f "$TASK2_JIT_DIR/module_deepgemm_opus.so"
test -f "$MULTIARCH_BENCH"
```

正式顺序固定为 `Task1 A1 -> Task2 B1 -> Task2 B2 -> Task1 A2`：

```bash
AITER_JIT_DIR="$TASK1_JIT_DIR" AITER_REBUILD=0 \
python3 "$MULTIARCH_BENCH" \
  --arch "$VALIDATION_ARCH" --endpoint task1 --pass-id A1 \
  --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$VALIDATION_LOG_DIR/perf_task1_A1.log"

AITER_JIT_DIR="$TASK2_JIT_DIR" AITER_REBUILD=0 \
python3 "$MULTIARCH_BENCH" \
  --arch "$VALIDATION_ARCH" --endpoint task2 --pass-id B1 \
  --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$VALIDATION_LOG_DIR/perf_task2_B1.log"

AITER_JIT_DIR="$TASK2_JIT_DIR" AITER_REBUILD=0 \
python3 "$MULTIARCH_BENCH" \
  --arch "$VALIDATION_ARCH" --endpoint task2 --pass-id B2 \
  --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$VALIDATION_LOG_DIR/perf_task2_B2.log"

AITER_JIT_DIR="$TASK1_JIT_DIR" AITER_REBUILD=0 \
python3 "$MULTIARCH_BENCH" \
  --arch "$VALIDATION_ARCH" --endpoint task1 --pass-id A2 \
  --warmup 20 --rounds 9 --iters 100 \
  2>&1 | tee "$VALIDATION_LOG_DIR/perf_task1_A2.log"
```

每个 case 取 9 轮单次 launch 时间的 median，再比较两遍 endpoint median。不能直接沿用 gfx950
测得的 `±1.2%` device/C++ 噪声带；应由本目标节点 A1/A2、B1/B2 的重复漂移建立本机噪声带。

性能通过要求：

- 每轮数值断言通过；
- raw/direct/graph 没有超出本机重复漂移的未解释退化；
- public/high-level 超出噪声时，必须分层定位到 Python adapter、C++ 检查或 device/kernel；
- 不能通过恢复 generic C++ 入口或删除安全检查来“修复”结果；
- Task1 endpoint、Task2 endpoint、sidecar 和每份日志均保存 SHA-256。

## 11. 结果记录模板

每个架构单独复制并填写一份结果，`PASS`、`FAIL`、`NOT RUN`、`BLOCKED` 四种状态不可混用。

### 11.1 环境

| 字段 | 值 |
|---|---|
| UTC 时间 | |
| `VALIDATION_ARCH` | gfx942 / gfx1250 |
| GPU 型号、`gcnArchName`、CU | |
| 物理卡号 / 进程内卡号 | |
| ROCm / hipcc | |
| Python / PyTorch / `torch.version.hip` | |
| git HEAD | |
| tracked patch SHA-256 | |
| untracked manifest SHA-256 | |
| Task2 `.so` SHA-256 | |
| final sidecar SHA-256 / kid 列表 | |
| Task1 `.so` SHA-256 | 性能门需要 |

### 11.2 功能与性能矩阵

| 项目 | 状态 | 日志/说明 |
|---|---|---|
| runtime arch 与 build arch 一致 | | |
| fresh canonical smoke | | |
| single-arch offload bundle | | |
| 四条 canonical pybind ABI / 旧 C++ ABI 缺失 | | |
| focused suite | | passed/skipped/failed 逐类填写 |
| 目标架构 skip 审计 | | |
| gfx942 补充节点 | | 不适用时写 N/A，仅 gfx1250 |
| gfx1250 补充节点 | | 不适用时写 N/A，仅 gfx942 |
| graph capture/replay | | family/kid 列明 |
| 双 stream | | family/kid 列明 |
| 跨 device 负例 | | 单卡时写 NOT RUN |
| Task1/Task2 ABBA | | |
| 未解释性能退化 | | 必须为“无”才能通过 |

### 11.3 日志哈希

执行结束后记录 GPU 空闲状态并生成日志清单：

```bash
/opt/rocm/bin/rocm-smi --showuse --showmemuse --showpids \
  2>&1 | tee "$VALIDATION_LOG_DIR/gpu_after.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' | tee "$VALIDATION_LOG_DIR/end_utc.txt"

find "$VALIDATION_LOG_DIR" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$VALIDATION_LOG_DIR/SHA256SUMS"
```

在结果被抄入仓内 checkpoint 前不要删除 `$AITER_JIT_DIR`、Task1 endpoint 或性能日志。

## 12. Definition of Done

一个目标架构只有同时满足以下条件才算完成：

- [ ] 真实 `gcnArchName` 与 `GPU_ARCHS`、single-arch offload bundle 一致；
- [ ] fresh sidecar 包含本架构必需 kid，且 fresh canonical smoke 数值通过；
- [ ] focused suite 0 failed，且没有当前目标架构测试被 skip；
- [ ] ABI 恰好暴露四条 canonical family raw，generic/旧 C++ tune 不存在，Python compat 仍存在；
- [ ] 第 8 节对应架构的补充实机节点全部通过；
- [ ] A16 two-stage、gfx1250 fused 或 gfx942 A8 的 workspace dtype/shape/caller ownership符合 exact kid；
- [ ] graph capture/replay 和双 stream 覆盖总计划要求的 family；
- [ ] 高层 backend 路由真实执行且没有误入 OPUS 空 family；
- [ ] Task1/Task2 同架构 ABBA 完成，数值正确，且没有未解释的性能退化；
- [ ] 环境、源码、`.so`、sidecar、日志 SHA-256 和 skip 分类完整保存；
- [ ] `task1_checkpoint.md`、`task2_checkpoint.md` 和 B7 未执行矩阵按实测结果更新。

以下情况不能签字：

- 只有 codegen/交叉编译通过；
- pytest 因没有目标硬件而 skip；
- 只运行了 two-stage，却把 gfx1250 fused 写成通过；
- 只运行 raw，却把 gfx942 `kid=None` 或真实 tuned 高层路由写成通过；
- 没有 Task1 冻结端点，却把当前端点的单边性能写成 Task1/Task2 无回退；
- backend 依赖缺失，却把 gfx1250 高层 FlyDSL/Triton/Gluon 能力写成通过。

## 13. 失败处理

| 失败层 | 处理 |
|---|---|
| arch/模块不匹配 | 停止，换全新 shell 和全新 `AITER_JIT_DIR`；不复用错误 `.so` |
| codegen/build | 保留 sidecar、build log 和 generated header；先分类 missing kid、arch filter、编译或链接错误 |
| 数值 | 记录 family、requested/actual kid、split-K、dtype、shape、workspace；缩小到单 case，不立即跑全量 |
| workspace/graph/并发 | 保留失败 tensor 合同和 stream/capture 条件；不能通过全局 cache 或跳过安全检查规避 |
| high-level route | 同时记录 tuned row、`libtype`、backend、canonical raw 是否被调用 |
| 性能 | 先看 direct/graph，再看 raw，最后看 public；用 ABBA 重复漂移判断，不能单次定性 |
| GPU hang/driver reset | 停止该卡上的后续实验，保存最后一个 case 和系统日志；不要自动 kill 或 reset 其他用户任务 |

完成 gfx942 后，应在另一个全新 shell 从第 3 节开始验证 gfx1250；两者的模块、sidecar、日志和最终
结论必须保持独立。

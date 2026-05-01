# Step 6 - Arch Wiring And JIT Build Attempt

## 目的

补齐 aiter 对 `gfx1250` 的 build target 和 device name 识别，使 `GPU_ARCHS=gfx1250` 不再因为未知 arch 在 JIT build target 解析阶段失败，并尝试构建 `module_mla_asm`。

## 具体操作

1. 修改 `aiter/aiter/jit/utils/build_targets.py`：
   - 在 `GFX_CU_NUM_MAP` 中新增 `"gfx1250": 64`。
   - 注释标明这是 MI400 placeholder，真实目标 SKU 应通过 `CU_NUM` 显式覆盖。
2. 修改 `aiter/aiter/jit/utils/chip_info.py`：
   - 在 `get_device_name()` 中新增 `gfx == "gfx1250" -> "MI400"`。
3. 使用 `ReadLints` 检查两个 Python 文件，无 linter 报错。
4. 尝试 JIT 构建：
   - 命令：`PYTHONPATH="/tmp/aiter_codegen_deps:$PWD" GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 python3 -c "from aiter.jit.core import build_module; build_module('module_mla_asm')"`
   - 当前机器的 `/usr/bin/python3` 缺少 `torch`，导入 `aiter` 时失败。

## 结果

- `gfx1250` arch wiring 已完成。
- JIT build 未能在当前 Python 环境运行，阻塞点是 `ModuleNotFoundError: No module named 'torch'`。
- 这不是代码路径本身的编译错误；需要在带 PyTorch/ROCm Python 环境中重新执行 `module_mla_asm` 构建。
- 后续仍可继续做 `.co` 符号/加载路径静态校验和 smoke test 文件编写。

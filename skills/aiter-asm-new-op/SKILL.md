---
name: aiter-asm-new-op
description: Integrates user-provided HSA .co assembly kernels into aiter as a brand-new op. Use when the user supplies a folder whose name is the op name, containing .cpp host reference code and to_aiter/<arch>/*.co where <arch> is each immediate subdirectory of to_aiter (e.g. to_aiter/gfx942/*.co), and wants JIT-built asm modules, a customer-facing fused_*.py API (see fused_moe.py), op_tests for correctness and performance, plus C++/pybind and compile_ops wiring (not extending an existing MOE/GEMM path).
---

# Aiter：从用户目录接入全新汇编 Op（.co）

本 skill 描述**第一步：仅新增 op**（不接在已有 `fmoe_2stages` 等路径上扩展）。

## 用户输入约定

用户会提供一个**文件夹路径**。约定如下：

| 含义 | 规则 |
|------|------|
| **Op 名** | 取该文件夹的**目录名**（basename），记为 `{op_name}`。须与 `hsa/{arch}/{op_name}/` 的目录名一致，建议小写、`[a-z0-9_]+`。 |
| **Host 参考** | 文件夹内（**不含** `to_aiter/`）的 `*.cpp` / `*.cxx` / `*.cc`：作为 host 侧逻辑参考（launch 参数、grid/block、`hipModuleGetFunction` 符号名、`KernelArgs` 布局等）。 |
| **待接入的 .co** | 路径 **`to_aiter/<arch>/*.co`**。这里的 **`<arch>` 就是 `to_aiter/` 下面的一级子目录名**（GPU 目标架构目录），与 aiter 里 `hsa/<arch>/` 的目录名一致，例如 `gfx942`、`gfx950`。**`.co` 文件直接放在该子目录下**，例如 `to_aiter/gfx942/k_vadd.co`，而不是放在 `to_aiter/` 根目录或其它嵌套路径（除非用户明确说明另有布局）。每个要支持的架构各建一个 `to_aiter/<arch>/` 目录，并在其中放入对应架构编出来的 `*.co`。 |

**识别步骤（Agent）**：先列出 **`to_aiter/*/`**（或 `glob("to_aiter/*")` 且筛出目录），每个子目录名即一个 **`<arch>`**；再对每个 arch 执行 **`to_aiter/<arch>/*.co`** 收集待拷贝的 `.co`。**不要**仅凭 `glob("to_aiter/**/*.co")` 就推断「缺少 arch 子目录」——若用户已按约定放置，`.co` 就在 **`to_aiter/gfx942/`** 这类路径下。

**不要**把 `to_aiter/` 里的内容当作 host 参考；host 参考仅来自上层目录的 C++ 源文件。

## Aiter 侧目标结构（新增 op）

在 aiter 仓库根目录下：

1. **二进制与 CSV**（每个需支持的架构各一份，常见 `gfx942` / `gfx950`）  
   `hsa/{arch}/{op_name}/`  
   - 从用户目录 **`to_aiter/<arch>/*.co`**（`<arch>` 为 `to_aiter/` 下的一级目录名）拷贝到 **`hsa/<arch>/{op_name}/`**（与 CSV 同目录）；每个架构各拷一份。  
   - 新建 **`{op_name}.csv`**（或按变体拆多表，见 `hsa/codegen.py`）。**必须列**：`tile_m`, `tile_n`, `knl_name`, `co_name` —— `knl_name` 为 .co 中导出符号（即 host 侧 `hipModuleGetFunction` 使用的 **kernel 名字符串**，常为 C++ mangled 名）；`co_name` 为对应 `.co` 文件名。其余列（如 `tg`, `pf`, `dtype` 等）按 `get_heuristic_*` 与业务按需添加；列类型须与 `hsa/codegen.py` 一致（数值列生成 `int`，否则 `std::string`）。**注意**：`hsa/codegen.py` 固定要求列名为 `knl_name`，不可用 `kernel_name` 作为列名。

2. **C++ 实现**（新建，命名建议一致）  
   - `csrc/py_itfs_cu/asm_{op_name}.cu`  
   - `#include "asm_{op_name}_configs.hpp"`  
   - 定义与用户 kernel ABI 一致的 `struct __attribute__((packed)) KernelArgs { ... }`（顺序与 padding 须与汇编一致；**以用户提供的 host .cpp 为准**）。  
   - 实现：从 `CFG` 选 kernel（`get_heuristic_*` 或固定 key）、`AiterAsmKernel` / `load_asm_kernel`、`launch_kernel`（参考 `csrc/py_itfs_cu/asm_topksoftmax.cu` 或 `asm_pa.cu`）。  
   - `knl_name` 必须与 .co 中导出符号一致（可从用户 cpp 或 `llvm-objdump` / 链接脚本确认）。

3. **声明与绑定**  
   - 在 `csrc/include/` 中合适头文件声明对外 C++ API（可参考 `moe_op.h` / 各 asm 头文件组织方式）。  
   - 新建或扩展现有 **pybind** `.cu`，`m.def("...", &your_fn, ...)`。  
   - 在 `csrc/include/rocm_ops.hpp` 的对应 `TORCH_LIBRARY` 块中注册 Python 可见名字（搜索现有 `*_asm` 的 `m.def` 作为模板）。

4. **JIT 模块**（仅此路径生成配置头，不单独手跑 codegen）  
   在 `aiter/jit/optCompilerConfig.json` 增加一项，例如 `module_{op_name}_asm`：  
   - `srcs`：上述 `asm_{op_name}.cu` + pybind 源。  
   - `blob_gen_cmd`：`f'{AITER_META_DIR}/hsa/codegen.py -m {op_name} --output_dir {{}}'`（JIT `build_module` 会在编译前执行，生成 `asm_{op_name}_configs.hpp` 到 module 的 blob 目录，供 `#include`；需保证构建环境已设置 **`AITER_GPU_ARCHS`**，与 `hsa/{arch}/` 一致）。  
   - 其余字段对齐 `module_mla_asm` / `module_attention_asm`（空 flags、`extra_ldflags` 等按需补）。

5. **Python 入口（ops 层）**  
   在 `aiter/ops/` 中新增 `compile_ops` 桩函数（如 `aiter/ops/{op_name}_asm.py`），通过 `get_module("module_{op_name}_asm")` 触发 JIT（参考 `aiter/ops/moe_op.py`）。

6. **客户接口 Python 模块（aiter 根包，必须）**  
   在 **`aiter/` 目录下**新增 **`fused_{op_name}.py`**（或与业务一致的命名，如 `fused_{op_name}.py`），作为**对外推荐**的调用面：  
   - 风格参考 **`aiter/fused_moe.py`**：模块级 docstring、对参数的说明与约束、默认输出缓冲区策略、在函数体内调用 `aiter/ops` 已暴露的底层符号（通过 `from .ops.xxx import ...` 避免与包根循环 import）。  
   - 在 **`aiter/__init__.py`** 中导出公开 API（如 `from .fused_{op_name} import fused_xxx`），便于 `import aiter; aiter.fused_xxx(...)`。

7. **单测（op_tests，必须）**  
   在 **`op_tests/`** 下新增 **`test_{op_name}_asm.py`**（或 `test_{op_name}.py`），至少包含：  
   - **正确性**：与 PyTorch / 参考实现对比（如 `checkAllclose` / `torch.allclose`），覆盖客户接口与 ops 底层各一条路径（若两者都存在）。  
   - **性能**：使用 `aiter.test_common` 中的 **`run_perftest`** / **`@benchmark`** 等（参考 `op_tests/test_activation.py`、`test_moeTopkSoftmax.py`），提供可复现的耗时或带宽指标；可加 `if __name__ == "__main__"` 与 `--perf` 便于本地扫 shape。  
   - 无 GPU 时使用 **`pytest.mark.skipif(not torch.cuda.is_available(), ...)`** 跳过。

8. **运行时加载 .co**  
   - 未嵌入二进制时：设置环境变量 **`AITER_ASM_DIR`** 指向 **`hsa` 所在目录**（使得 `{AITER_ASM_DIR}/{arch}/{relpath}/{co_name}` 可加载），见 `csrc/include/aiter_hip_common.h` 中 `load_asm_kernel`。  
   - 仓库内通常：`export AITER_ASM_DIR=<repo>/aiter/hsa`（路径以实际布局为准）。

## 执行流程（Agent  checklist）

1. [ ] 解析 `{op_name}` = 用户给定文件夹的 basename；列出 **`to_aiter/` 下每个子目录名作为 `<arch>`**，再列出各 **`to_aiter/<arch>/*.co`**，以及用户目录下（不含 `to_aiter/`）的 host `*.cpp`。  
2. [ ] 阅读 host `.cpp`，提取：kernel 符号名、参数块布局、launch 维度和流。  
3. [ ] 将每个 **`to_aiter/<arch>/*.co`** 拷贝到 **`hsa/<arch>/{op_name}/`**（与 aiter 支持的架构一一对应），并编写 CSV 行：必填 `tile_m`, `tile_n`, `knl_name`, `co_name`，其余列按需。  
4. [ ] 实现 `asm_{op_name}.cu` + pybind + `rocm_ops.hpp` + `optCompilerConfig.json`（含 `blob_gen_cmd`）。  
5. [ ] 在 `aiter/ops/` 增加 `compile_ops` 入口并可在 Python 中调用。  
6. [ ] 新增 **`aiter/fused_{op_name}.py`**（客户接口），并在 `aiter/__init__.py` 导出。  
7. [ ] 新增 **`op_tests/test_{op_name}_asm.py`**：正确性 + 性能（`run_perftest` / benchmark 模式）。  
8. [ ] 触发 JIT 编译/导入 `module_{op_name}_asm`；设置 `AITER_ASM_DIR` 验证加载 `.co`。

## 参考文件（仓库内）

| 用途 | 路径 |
|------|------|
| Codegen | `hsa/codegen.py` |
| 较简单的 asm + CFG 示例 | `csrc/py_itfs_cu/asm_topksoftmax.cu` |
| Kernel 加载 | `csrc/include/aiter_hip_common.h` |
| JIT 配置示例 | `aiter/jit/optCompilerConfig.json` → `module_mla_asm`, `module_attention_asm` |
| 客户层 Python 接口风格 | `aiter/fused_moe.py` |
| 单测（正确性 + 性能） | `op_tests/test_activation.py`、`op_tests/test_moeTopkSoftmax.py` |
| vadd 接入示例 | `aiter/fused_vadd.py`、`op_tests/test_vadd_asm.py` |
| 已有 MOE 变体扩展（非本 skill 范围） | `skills/aiter-moe-asm-kernel/SKILL.md` |

## 注意事项

- **模块名** `-m {op_name}` 必须与 `hsa/{arch}/{op_name}/` 目录名一致。  
- CSV **必须**含 `tile_m`, `tile_n`, `knl_name`, `co_name`（`knl_name` = kernel 符号列名，与 codegen 一致），否则无法满足本 skill 的 heuristic 与 codegen 要求。  
- 若用户文件夹名含非法字符，与用户确认后改为合法 `{op_name}` 再建目录。

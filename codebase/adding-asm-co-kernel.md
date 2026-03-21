# 在 Aiter 中新增 .co 汇编 Kernel（含全新 Op）

本文档概括：**如何加入 .co 二进制**、**`hsa/codegen.py` 的用法**，以及当 **工程中尚不存在对应 Op** 时的完整接入路径。更细的操作清单见仓库内 **Agent Skill**：`skills/aiter-asm-new-op/SKILL.md`（新 Op）、`skills/aiter-moe-asm-kernel/SKILL.md`（在现有 MOE 路径上扩展 kernel）。

---

## 一、工程里 .co 与 codegen 分别做什么

| 对象 | 作用 |
|------|------|
| **`.co` 文件** | HSA 代码对象，运行时由 `hipModuleLoad` / `hipModuleGetData` 加载（见 `csrc/include/aiter_hip_common.h` 中 `load_asm_kernel`）。 |
| **`hsa/codegen.py`** | **不生成 .co**。它根据各架构下的 **CSV** 合并生成 **C++ 配置头** `asm_{module}_configs.hpp`，内含 kernel 符号名、`.co` 相对路径、`tile_*` 等元数据，供 C++ 侧选择 kernel 并加载。 |

---

## 二、Codegen 用法

### 2.1 前置条件

- 环境变量 **`AITER_GPU_ARCHS`**：分号分隔，与 `hsa/` 下架构目录一致，例如：  
  `export AITER_GPU_ARCHS=gfx942;gfx950`
- CSV 路径模式：`hsa/{arch}/{module}/**/*.csv`
- 每个 CSV **必须**包含列：**`knl_name`**、**`co_name`**（脚本会校验）。

### 2.2 命令

在 **aiter 仓库根目录**执行：

```bash
python hsa/codegen.py -m <module> -o <输出目录>
```

- **`-m <module>`**：与 `hsa/{arch}/{module}/` 目录名一致（例如 `fmoe_2stages`、`mla`、`vadd`）。
- **`-o`**：生成 `asm_<module>_configs.hpp` 的输出目录。

### 2.3 与 JIT 编译的关系

多数 asm 算子在 **`aiter/jit/optCompilerConfig.json`** 里通过 **`blob_gen_cmd`** 在 **编译该 Python 扩展模块时自动执行** codegen，输出写到对应 module 的 **blob** 目录，无需每次手跑：

```text
f'{AITER_META_DIR}/hsa/codegen.py -m <module> --output_dir {{}}'
```

本地调试也可手跑 codegen，将 `-o` 指到 `aiter/jit/build` 等目录。

### 2.4 生成内容说明

- 为每个 **CSV 文件名（不含扩展名）** 生成一个 `static CFG cfg_<名字>`。
- 表项中包含：`knl_name`、`co_name`、`arch`，以及 CSV 中其余列（数值列映射为 `int`，否则 `std::string`）。
- `.co` 在配置里的路径一般为 **`{relpath}/{co_name}`**（`relpath` 相对 `hsa/{arch}/`）。

---

## 三、添加 .co 二进制文件的步骤（概览）

### 3.1 仅在「已有 Op / 已有 module」上增加新 kernel 变体

1. 将 **`.co`** 放到对应目录，例如：  
   `hsa/{arch}/{module}/your_kernel.co`
2. 编辑或新增 **CSV 行**：填写 **`knl_name`**（与 .co 中导出符号一致）、**`co_name`**，以及启发式需要的列（如 `tile_m`、`tile_n` 等）。
3. **重新生成配置头**：手跑 `hsa/codegen.py` 或重新 JIT 编译依赖该 module 的扩展（触发 `blob_gen_cmd`）。
4. 若 kernel **参数或类型有变**：在对应 **`asm_*.cu`** 中更新 **`get_cfg` / `get_heuristic_*`** 与 **`KernelArgs`**（`packed` 布局须与汇编 ABI 一致）。
5. **重新编译**扩展；运行时若未嵌入 HSA，设置 **`AITER_ASM_DIR`** 指向 **`hsa` 所在目录**（使得 `{AITER_ASM_DIR}/{arch}/.../*.co` 能打开）。

### 3.2 工程中「尚不存在」该 Op：新 Op 全流程

1. **目录与二进制**  
   - `hsa/{arch}/{op_name}/`：放入各架构 **`.co`** 与 **CSV**（至少 `knl_name`、`co_name`；按 skill 还可要求 `tile_m`、`tile_n` 等）。
2. **Codegen**  
   - `-m` 与目录名 `{op_name}` 一致；通过 JIT 的 `blob_gen_cmd` 或手跑生成 **`asm_{op_name}_configs.hpp`**。
3. **C++**  
   - 新建 `csrc/py_itfs_cu/asm_{op_name}.cu`：`#include` 配置头，实现 **`KernelArgs`**、`CFG` 查找、`AiterAsmKernel` 与 `launch_kernel`。
4. **绑定**  
   - 声明头文件、**pybind**、`csrc/include/rocm_ops.hpp` 中宏（如 `*_PYBIND`）。
5. **JIT**  
   - 在 **`optCompilerConfig.json`** 注册 **`module_{op_name}_asm`**，配置 **`blob_gen_cmd`**。
6. **Python**  
   - `aiter/ops/` 下 **`compile_ops`** 桩函数；**`aiter/fused_{op_name}.py`** 客户接口；**`op_tests`** 正确性与性能测试。  
   - 详见 **`skills/aiter-asm-new-op/SKILL.md`**。

### 3.3 运行时加载 .co

- 设置 **`AITER_ASM_DIR`** 为包含 **`gfx942`** / **`gfx950`** 等子目录的 **`hsa` 根路径**（与 `load_asm_kernel` 中拼接方式一致）。  
- 若使用嵌入的 HSA map（`AITER_EMBEDDED_HSA_*`），则按构建系统约定，无需文件路径。

---

## 四、与其他「codegen」脚本的区别

仓库内还有 **Composable Kernel 实例生成**等脚本（如 `csrc/ck_gemm_a8w8/gen_instances.py`、`csrc/ck_gemm_moe_2stages_codegen/`），用于 **CK/Triton 侧实例化**，与 **`hsa/codegen.py`（汇编 .co 元数据头文件）** 不是同一条流水线。根据你要改的是 **HSA CSV** 还是 **CK gemm 实例**，选择对应脚本。

---

## 五、相关路径速查

| 说明 | 路径 |
|------|------|
| 汇编配置生成脚本 | `hsa/codegen.py` |
| HSA 说明（简要） | `hsa/readme.md` |
| Kernel 加载 | `csrc/include/aiter_hip_common.h` |
| JIT 与 `blob_gen_cmd` | `aiter/jit/core.py`、`aiter/jit/optCompilerConfig.json` |
| 新 Op Skill | `skills/aiter-asm-new-op/SKILL.md` |
| MOE 扩展 .co Skill | `skills/aiter-moe-asm-kernel/SKILL.md` |

---

*本文档由对话总结整理，随工程演进请以脚本与 Skill 为准。*

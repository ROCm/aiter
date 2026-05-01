# port mi400 to aiter skill

> 分析 poc_kl/mi400 与 aiter 的工程结构，制定 mi400 汇编 kernel 移植到 aiter 的标准工作流，并生成名为 `port-mi400-op-to-aiter` 的项目级 skill 供以后复用。

## 关键决策（已与用户对齐）

- **集成策略 = hybrid**：mi400 中已有同名 aiter op 的（`fmha_fwd_f16` / `fmha_bwd_f16` / `f4gemm` / `mla`）走"扩展现有 op"路径；其它（`fmha_fwd_mxfp8` / `fmha_decode_mxfp8` / `mxfp8fp4gemm`）走"新建独立 op"路径。
- **资产策略 = co_only（新策略，需要显式校验）**：只把构建好的 `.co + .csv` 提交到 `aiter/hsa/gfx1250/<op>/`；`.sp3 / .s` 仍留在 `poc_kl/mi400/<op>/shaders/` 作为 source of truth。当前 `aiter/hsa` 主要只有 CSV，新增 `.co` 前必须确认 CI、wheel/package data、embedded HSA 资源路径会包含这些 `.co`。
- **skill 位置 = project skill**：如果 Cursor workspace root 是 `/home/feifei/repo`，skill 应放在 `/home/feifei/repo/.cursor/skills/port-mi400-op-to-aiter/`；只有在单独打开 `aiter/` 作为 workspace 时才放 `aiter/.cursor/skills/...`。

## 目标

为以后所有 mi400 单 op 移植到 aiter 创建一份可复用 skill，覆盖"扩展已有 op"与"新建独立 op"两条路径，并以 co_only 的资产策略提交 `.co + .csv` 到 `aiter/hsa/gfx1250/`。skill 需要能指导 agent 先发现真实工程结构，再选择 pybind/Tensor 入口或 ctypes 入口，避免把所有 op 强行套进同一个样板。

## 工程结构对应关系（核心映射）

- mi400 `<op>.cpp` 内的 `struct *KernelArgsBase`（`p2`/`p3` 槽位）→ aiter packed args struct（保持槽位，槽位定义见 [aiter/csrc/include/aiter_hip_common.h](aiter/csrc/include/aiter_hip_common.h)）
- mi400 `fill_*_kernel_args` lambda → aiter `init_<op>_args(args, ...)`
- mi400 `run.sh` / `test.sh` / `CMakeLists.txt` / op-specific scripts（如 `new_run.sh`、patch 脚本）的变体矩阵 → `aiter/hsa/gfx1250/<op>/<op>.csv`（调度字段 + `knl_name` + `co_name`）
- mi400 `<base>_KERNEL_FUNC` 符号 → CSV `knl_name` 列（无 C++ mangle，直接用符号串）
- mi400 `do_compute` 中的 grid/block 公式 → aiter dispatcher 的 `get_grid_dim()` + `bdx/bdy/bdz`
- mi400 `*.co`（由 [poc_kl/mi400/fmha_fwd_f16/run.sh](poc_kl/mi400/fmha_fwd_f16/run.sh) 中 `compile()` 用 `amdclang++ --offload-arch=gfx1250 *.s -o *.co` 产出）→ commit 到 `aiter/hsa/gfx1250/<op>/*.co`
- aiter `hsa/codegen.py` 生成的 config map key 是 `arch + knl_name`，不是任意调度键；dispatcher 通常要遍历 `cfg_*` 匹配字段，或自己建立 heuristic/cache。

## aiter 链路关键文件（参考样板：`fmha_v3_fwd`）

- 配置生成：[aiter/hsa/codegen.py](aiter/hsa/codegen.py)（CSV → `asm_<module>_configs.hpp`，map 名 `cfg_<csv_basename>`）
- 运行时加载器：`AiterAsmKernel` 在 [aiter/csrc/include/aiter_hip_common.h](aiter/csrc/include/aiter_hip_common.h) 第 240 行附近，按 `AITER_ASM_DIR/<arch>/<hsaco_path>` 读 `.co`
- 扩展样板：[aiter/csrc/cpp_itfs/mha_fwd.cu](aiter/csrc/cpp_itfs/mha_fwd.cu) 中 `fmha_fwd_v3()`（dispatcher）+ [aiter/csrc/include/mha_fwd.h](aiter/csrc/include/mha_fwd.h) 的 `fmha_fwd_v3_args`（packed struct）
- 张量入口样板：[aiter/csrc/py_itfs_cu/asm_mha_fwd.cu](aiter/csrc/py_itfs_cu/asm_mha_fwd.cu)
- pybind 注册：[aiter/csrc/include/rocm_ops.hpp](aiter/csrc/include/rocm_ops.hpp) 的 `MHA_FWD_ASM_PYBIND` 宏
- Python stub：[aiter/aiter/ops/mha.py](aiter/aiter/ops/mha.py) 的 `@compile_ops("module_fmha_v3_fwd", fc_name="fmha_v3_fwd")`
- 模块条目：[aiter/aiter/jit/optCompilerConfig.json](aiter/aiter/jit/optCompilerConfig.json) 的 `module_fmha_v3_fwd`（带 `blob_gen_cmd` 调 codegen.py）
- ctypes 样板：[aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu](aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu) 的 `AITER_CTYPES_DEFINE_ENTRYPOINT_*` + [aiter/aiter/ops/gemm_op_a4w4.py](aiter/aiter/ops/gemm_op_a4w4.py) 的 `ffi_type="ctypes"`。

## mi400 各 op 的路径分叉（hybrid）

- 扩展已有：
  - `fmha_fwd_f16` → `fmha_v3_fwd`
  - `fmha_bwd_f16` → `fmha_v3_bwd`（注意 mi400 的 `bwd_convert_dq` 与 `bwd_odo` 是 pre/post helper kernel，需作为附属 .co 一起注册）
  - `f4gemm` → 现有 `module_gemm_a4w4_asm` / `hsa/<arch>/f4gemm/*.csv` / `asm_gemm_a4w4.cu` / `aiter/ops/gemm_op_a4w4.py` 链路
  - `mla` → `module_mla_asm`
- 新建独立：
  - `fmha_fwd_mxfp8`、`fmha_decode_mxfp8`、`mxfp8fp4gemm`（独立 module name + 独立 pybind 宏 + 独立 `aiter.<op>` 入口）

## skill 产物（要创建的文件）

位置：优先使用仓库根目录的 `.cursor/skills/port-mi400-op-to-aiter/`。若实际工作区根目录是 `aiter/`，再使用 `aiter/.cursor/skills/port-mi400-op-to-aiter/`。

- `SKILL.md`（约 250 行）：discovery 清单 + 8 步工作流 + extend/new 两分支模板 + 验证清单
- `reference.md`：完整字段映射表 + CSV/codegen 约定 + `AiterAsmKernel` 加载机制 + grid/block 推导 + HSA `.args:` 元数据规则 + arch 白名单、`get_gpu_arch`、`GPU_ARCHS`/`AITER_GPU_ARCHS` 行为
- `templates/`（agent 复制后填空）
  - `args_struct.h.tmpl`
  - `dispatcher.cu.tmpl`（含 `cfg_*` 查表、`init_*_args`、`get_grid_dim`、`AiterAsmKernel` 调用）
  - `asm_op_torch.cu.tmpl`（`at::Tensor` → args 的 stride/shape 折算）
  - `ctypes_entrypoint.cu.tmpl`（`aiter_tensor_t` + `AITER_CTYPES_DEFINE_ENTRYPOINT_*` 路径）
  - `pybind_macro.hpp.tmpl`
  - `ops_py.py.tmpl`（`@compile_ops` 装饰 + fake tensor gen）
  - `csv_header.tmpl`（必含 `knl_name,co_name`）
  - `optCompilerConfig_entry.json.tmpl`
- `checklists/port_checklist.md`：每个 op 必勾选的项

## 8 步工作流（skill 主流程）

1. **Discovery**：读 `poc_kl/mi400/<op>/<op>.cpp`、`run.sh`、`test.sh`、`CMakeLists.txt` 和 op-specific scripts，定位 `*KernelArgsBase`、`fill_*_kernel_args`、`do_compute` 的 grid/block 公式、真实 build/test 命令和变体矩阵。
2. **Build .co**：按 discovery 得到的真实命令在 poc_kl 侧构建 `.co`。不要默认所有 op 都是 `bash run.sh convert && bash run.sh compile`；遇到 `new_run.sh`、patch 脚本或 CMake 路径要按本 op 实际流程执行。
3. **Place assets**：`mkdir -p aiter/hsa/gfx1250/<op>` 并复制需要提交的 `.co`；写 `<op>.csv`，列名包含调度字段和固定的 `knl_name,co_name`。运行 `hsa/codegen.py -m <module>` 后检查生成的 `cfg_*` 中 `co_name` 最终能定位到真实 `.co`。
4. **Args struct + ABI 校验**：把 `*KernelArgsBase` 字段逐一翻成 aiter packed struct（保持 `p2`/`p3` 槽位），并对照 `.s/.sp3` 的 `.args` 元数据核对字段顺序、16B slot、padding、`sizeof(args)` 和指针/标量类型。
5. **Dispatcher .cu**：仿 `fmha_fwd_v3()` 或 ctypes asm 样板写 dispatcher；arch 白名单、Python wrapper 限制和 build target 解析均需覆盖 `gfx1250`；通过 `cfg_<op>` 匹配字段拿 `knl_name/co_name`，用 `SynchronizedCache` 缓存 `AiterAsmKernel`，最后 `launch_kernel`。
6. **路径分叉**：
   - 扩展现有：在该 op 的 `mha_fwd.cu` / `mla.cu` 等里加 `if(arch_id == "gfx1250") {...}` 分支；CSV 合并到现有 csv（加 `arch=gfx1250` 行）；现有 `mha_fwd_args` 结构如能复用就直接复用，否则在 dispatcher 内做 mi400 专属变量映射。
   - 新建独立：先判断入口风格。Tensor/pybind 风格新增 `csrc/py_itfs_cu/asm_<op>.cu` + `csrc/pybind/<op>_pybind.cu` + `rocm_ops.hpp` 宏；ctypes 风格在 `asm_<op>.cu` 内使用 `AITER_CTYPES_DEFINE_ENTRYPOINT_*` 并在 Python stub 里设置 `ffi_type="ctypes"`。两种路径都要在 `aiter/aiter/ops/<op>.py` 和 `optCompilerConfig.json` 加 module 条目，必要时补包导出/测试/benchmark 入口。
7. **符号与加载验证**：用 `.co` 的符号表或等价工具确认 CSV `knl_name` 是真实可加载符号；确认 `AITER_ASM_DIR/<runtime_arch>/<cfg.co_name>` 能打开 `.co`。
8. **JIT 构建 + correctness + 回归**：先跑 poc_kl 原始测试，再跑 aiter wrapper；每个 CSV 变体至少覆盖一个 shape；fmha/bwd 类覆盖 causal/non-causal、group/batch、边界 seqlen、helper kernel 顺序和 workspace；最后回归 mi300/mi350，确认未破坏现有 arch 的同名 op。

## 关键约束（skill 强调项）

- `gfx1250` 目录目前不存在，新增 op 是 greenfield；走 `aiter/hsa/codegen.py` 自动注入 `arch=gfx1250` 列。
- `aiter/jit/core.py` 已允许 `gfx1250`，但还要检查 `aiter/jit/utils/build_targets.py` 的 `GFX_CU_NUM_MAP`、op 级 `arch_id` 白名单、Python wrapper 和测试环境 `GPU_ARCHS=gfx1250`。
- `kernel_name` 在 csv 里优先用 sp3/汇编产出的 `<base>_KERNEL_FUNC` 字面符号，不要 C++ mangle；但必须用 `.co` 符号表实际确认。
- `AiterAsmKernel::load_hsaco_file` 路径相对 `AITER_ASM_DIR/<arch>/`，所以 csv 里的 `co_name` 路径是相对该目录。
- `hsa/codegen.py` 会按 `hsa/<arch>/<module>/**/*.csv` 自动发现 CSV，并把所在子目录拼到 `co_name` 前；提交前必须核对最终路径。
- packed args struct 必须保留 `p2`/`p3` 槽位；任何字段错位都会导致 HSA 拒绝启动。
- mi400 的 host 用裸 HIP，没有 ck_tile；新 op 的 dispatcher 不要 `#include "ck_tile/core.hpp"`（定义 `ENABLE_CK=0`），扩展现有 op 时按对应 op 既有约定。
- 扩展现有 op 路径要小心：mi400 kernel arg 顺序与 gfx950 可能不同，`init_*_args` 必须分支或独立函数。
- `.co` 是新增可提交资产类型，skill 里要提醒检查 `.gitignore`、package data、CI artifact、wheel 安装后 `AITER_ASM_DIR` 或 embedded HSA 是否能找到它。

## 实施 todos

- [ ] **skill_dir** — 创建目录 `.cursor/skills/port-mi400-op-to-aiter/{templates,checklists}/`（若 workspace root 是 `aiter/`，则使用 `aiter/.cursor/skills/...`）
- [ ] **skill_md** — 写 `SKILL.md`：description（含触发词 mi400 / poc_kl / asm kernel 移植）、8 步工作流、extend vs new 分支、验证清单
- [ ] **ref_md** — 写 `reference.md`：字段映射表、CSV/codegen 约定、`AiterAsmKernel` 加载机制、grid/block 推导、HSA `.args:` 元数据、arch 白名单、`.co` 打包/加载检查
- [ ] **tmpl_args** — 写 `templates/args_struct.h.tmpl`（含 `p2`/`p3` 槽位说明）
- [ ] **tmpl_disp** — 写 `templates/dispatcher.cu.tmpl`（cfg 查表 + `init_args` + `get_grid_dim` + `AiterAsmKernel`）
- [ ] **tmpl_torch** — 写 `templates/asm_op_torch.cu.tmpl`（`at::Tensor` 入口与 stride 折算）
- [ ] **tmpl_ctypes** — 写 `templates/ctypes_entrypoint.cu.tmpl`（`aiter_tensor_t` 入口与 `ffi_type="ctypes"` 配套）
- [ ] **tmpl_py** — 写 `templates/{pybind_macro.hpp.tmpl, ops_py.py.tmpl, csv_header.tmpl, optCompilerConfig_entry.json.tmpl}`
- [ ] **checklist** — 写 `checklists/port_checklist.md`（逐 op 勾选清单，覆盖 build、ABI、符号、路径、correctness、arch 回归）

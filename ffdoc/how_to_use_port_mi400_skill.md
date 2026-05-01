# 如何使用 `port-mi400-op-to-aiter` skill

本文档说明如何在 Cursor 中触发并使用上一步生成的 skill，把 `poc_kl/mi400/<op>/` 里的某个汇编 kernel 移植到 `aiter`。

## 1. 这个 skill 在哪、什么时候会被加载

- **物理位置**：`/home/feifei/repo/.cursor/skills/port-mi400-op-to-aiter/`（工作区根级 `.cursor/skills/`，poc_kl 与 aiter 子目录都能复用）。
- **启用方式**：Cursor 在每次会话启动时会扫描 `<workspace>/.cursor/skills/**/SKILL.md`。无需任何注册步骤；只要 workspace 是 `/home/feifei/repo`，agent 就能在系统提示里看到这条 skill 的 description。
- **加载条件**：只在当前对话里出现 description 中的触发词时，agent 才会主动读 `SKILL.md` 全文。否则 agent 看到的只是一行简介。

### 触发词（出现其一即可让 agent 主动加载 skill）

`mi400` / `poc_kl` / `asm kernel 移植` / `sp3` / `hsaco` / `.co` / `fmha_fwd_f16` / `fmha_fwd_mxfp8` / `fmha_decode_mxfp8` / `fmha_bwd_f16` / `mxfp8fp4gemm` / `f4gemm` / `mla`（仅当上下文是移植）/ `vadd` / `AiterAsmKernel` / `hsa/codegen.py` / `aiter/hsa/gfx1250`

### 不会被自动加载的场景

- 单纯讨论 aiter 已有 op 的功能、性能调优、bug 修复。
- 单纯讨论 mi300/mi350 的 kernel，不涉及 mi400 / gfx1250。
- 修改 aiter 与 asm kernel 无关的 Python 代码。

## 2. 何时使用、何时不用

| 场景 | 是否使用 skill |
|---|---|
| 把一个新 mi400 op 第一次接入 aiter | 用 |
| 给一个已经移植过的 op 加一个新变体（新 `.s` / `.co`） | 用（按 Step 3 仅追加 CSV 行） |
| 在 aiter 里给一个已存在 op（mi300/mi350 已支持）加 gfx1250 分支 | 用（走 extend 路径） |
| 修一个已经移植过的 op 的 numerics bug | 不用，直接定位修复即可 |
| 调整 aiter 已有 op 的 Python 接口 | 不用 |
| 仅在 poc_kl 调试 sp3/.s 而不接入 aiter | 不用 |

## 3. 调用 skill 的标准提示词（推荐）

直接把以下任一句作为 prompt 发给 agent，即可完整触发 skill：

- "请按 `port-mi400-op-to-aiter` skill，把 `poc_kl/mi400/fmha_fwd_mxfp8/` 移植到 aiter。"
- "用 mi400 移植 skill 处理 `fmha_decode_mxfp8`。"
- "把 `poc_kl/mi400/<op>/` 里的 kernel 接入 aiter（gfx1250），按 skill 走完所有步骤。"

如果你不指定 op，agent 会先列出 `poc_kl/mi400/` 下所有子目录并请你选一个；这等同于跳过 skill 第 1 步的 discovery 阶段。

### 调用前最好显式说清楚

| 信息 | 是否必填 | 默认值 |
|---|---|---|
| 要移植的 op 名（`poc_kl/mi400/<op>/`） | 必填 | 无；不填会被反问 |
| 是 extend 还是 new | 推荐填 | skill 自动按 op-to-module 映射表判断 |
| 入口风格：pybind / ctypes | 选填 | extend 时跟随已有 op；new 时按"FMHA 用 pybind，GEMM 用 ctypes"启发式 |
| 目标 CU 数（用于 `GFX_CU_NUM_MAP`） | 选填 | skill 会反问，无默认 |
| 是否要写 op_test 与跑 correctness | 选填 | 默认要写，不跑（无 GPU 时）；有 GPU 可让 agent 跑 |

## 4. agent 走完 skill 后会做的事

按 SKILL.md 的 12 步工作流：

1. **Discovery** — 读 `poc_kl/mi400/<op>/` 全部相关文件（`<op>.cpp`、`run.sh`、`test.sh`、`new_run.sh`、`CMakeLists.txt`、`shaders/`、helper scripts），输出一份结构化报告。
2. **Build .co** — 用真实命令构建 `.co`（不假设 `run.sh convert/compile`）。
3. **Place assets** — `mkdir aiter/hsa/gfx1250/<op>/` + 写 `<op>.csv`。
4. **KernelArgs 翻译** — 把 `*KernelArgsBase` 翻成 packed struct，按 `.s` 的 `.args:` 校验 ABI。
5. **Path 选择** — 按 `reference.md §10` 的 op→module 映射表选 extend 或 new。
6. **Dispatcher** — 套 `templates/dispatcher.cu.tmpl`。
7. **Entry 接入** — pybind 或 ctypes 二选一，配相应模板。
8. **module 注册** — 在 `optCompilerConfig.json` 加条目。
9. **Arch wiring** — 给 `GFX_CU_NUM_MAP` 加 gfx1250、补任何 arch whitelist。
10. **符号与加载验证** — `llvm-readelf -s`、`AITER_ASM_DIR` smoke test。
11. **Correctness** — `op_tests/test_<op>.py`，PyTorch eager 当 reference。
12. **Regression** — 回归 mi300/mi350 同名 op。

每一步对应 `checklists/port_checklist.md` 的一组 todo，agent 会全部勾选或显式标 N/A。

## 5. 你需要在每个阶段做的配合

| 阶段 | 你的动作 |
|---|---|
| Discovery 后 | 检查报告里的 KernelArgs 字段顺序、变体矩阵是否齐全；如果 agent 漏读了某个 `.cpp`/脚本，及时补 |
| Build .co 时 | 如果 docker / sp3 工具链路径有特殊要求，告诉 agent；这一步 agent 通常需要在你的环境里手跑 |
| Place assets 后 | 看一眼 `<op>.csv` 的列名是否和已有同模块 csv 一致（extend 路径必须严格一致） |
| Path 与 entry 风格 | 如果你不同意 skill 的默认推断，明确告知"用 ctypes 而非 pybind"或反之 |
| Arch wiring | 提供目标 SKU 的 CU 数（gfx1250 默认值未知，必须给） |
| 验证阶段 | 提供有 gfx1250 的机器跑 correctness；mi300/mi350 回归也需要相应硬件 |

## 6. 典型对话示例

### 示例 A：移植一个新 op（new 路径，pybind）

> 用户："请按 port-mi400-op-to-aiter skill 把 `poc_kl/mi400/fmha_fwd_mxfp8/` 接入 aiter，gfx1250 的 CU 数是 192。"

agent 应当依次：

1. 读 SKILL.md + reference.md，加载到上下文。
2. discovery：读 `fmha_fwd_mxfp8.cpp`、`run.sh`、`test.sh`、`shaders/MXFP8_FMHA_FWD_D128_1TG_4W_64mx4_128nx1.s`，列出 `*KernelArgsBase` 字段。
3. 让你确认（或自动决定）该 op 走 new + pybind。
4. 给出 build 命令；如果可在本机执行，跑 `bash run.sh convert && bash run.sh compile`。
5. 创建 `aiter/hsa/gfx1250/fmha_fwd_mxfp8/`，写 `.co` + `fmha_fwd_mxfp8.csv`。
6. 用 `templates/args_struct.h.tmpl` 在 `aiter/csrc/include/fmha_fwd_mxfp8.h` 里建 packed struct。
7. 用 `templates/dispatcher.cu.tmpl` 写 `aiter/csrc/cpp_itfs/fmha_fwd_mxfp8.cu`。
8. 用 `templates/asm_op_torch.cu.tmpl` 写 `aiter/csrc/py_itfs_cu/asm_fmha_fwd_mxfp8.cu`。
9. 用 `templates/pybind_macro.hpp.tmpl` 在 `aiter/csrc/include/rocm_ops.hpp` 添加宏，写 `aiter/csrc/pybind/fmha_fwd_mxfp8_pybind.cu`。
10. 用 `templates/ops_py.py.tmpl` Flavor A 写 `aiter/aiter/ops/fmha_fwd_mxfp8.py`，并在 `aiter/__init__.py` 导出。
11. 在 `optCompilerConfig.json` 加 `module_fmha_fwd_mxfp8` 条目。
12. 在 `aiter/aiter/jit/utils/build_targets.py::GFX_CU_NUM_MAP` 加 `"gfx1250": 192`。
13. 写 `aiter/op_tests/test_fmha_fwd_mxfp8.py`。
14. 给出 build 与运行命令、回归脚本。
15. 用 `port_checklist.md` 走完所有勾选项。

### 示例 B：扩展已有 op（extend 路径，pybind）

> 用户："给 aiter 现有的 fmha_v3_fwd 加上 gfx1250 支持（数据来自 `poc_kl/mi400/fmha_fwd_f16/`），按 mi400 移植 skill 走。"

agent 应当：

1. 走 discovery，但 Step 5 直接选 extend → `module_fmha_v3_fwd`。
2. **追加**新行到 `aiter/hsa/gfx950/fmha_v3_fwd/fmha_fwd.csv` 旁边的 `aiter/hsa/gfx1250/fmha_v3_fwd/fmha_fwd.csv`（不修改 gfx950 行）。
3. 不创建新 `.cu` 文件；改 `aiter/csrc/cpp_itfs/mha_fwd.cu::fmha_fwd_v3()`，在 arch 白名单里加 `gfx1250`，必要时加 `init_fmha_fwd_mi400_args` 分支。
4. 不动 `aiter/aiter/ops/mha.py` 的对外函数签名，只确保 `module_fmha_v3_fwd` 配置不变。
5. 跑 mi300/mi350 回归确认无破坏。

### 示例 C：扩展已有 op（extend 路径，ctypes）

> 用户："把 `poc_kl/mi400/f4gemm/` 接入 aiter，复用 gemm_a4w4_asm。"

agent 应当：

1. 沿用现有 ctypes 路径（`asm_gemm_a4w4.cu` + `gemm_op_a4w4.py`）。
2. 给 `aiter/hsa/gfx1250/f4gemm/` 提供 `.co` 与 csv 行。
3. 在 `asm_gemm_a4w4.cu` 内修改 `get_cfg` / heuristic 函数，使其在 `arch_id == "gfx1250"` 时返回 mi400 cfg。
4. 不增加 pybind 宏（ctypes 路径不需要）。

## 7. 验证 agent 是否真的按 skill 在做

在 agent 的回答里查这些迹象：

- 它显式提到自己读了 `SKILL.md` 与 `reference.md`，或在第一条消息就引用了 skill 里的术语（如 "co_only 资产策略"、"`p2`/`p3` 槽位"、"`AITER_CTYPES_DEFINE_ENTRYPOINT_VOID`"）。
- 在 discovery 阶段它读了**多个**文件，至少包括 `<op>.cpp`、`run.sh`、`shaders/` 里的样本 `.s`，而不是只看 `<op>.cpp`。
- 它复用了 `port_checklist.md` 里的勾选项，而不是临时编一个 todo。
- 它在 Step 8 主动检查 `GFX_CU_NUM_MAP` 是否含 `gfx1250`。
- 它在 Step 9 提示要 `llvm-readelf -s` 校验 `.co` 符号。

如果它直接跳到写代码、不读 poc_kl 源码，或者忘了改 `GFX_CU_NUM_MAP`，那就是没在按 skill 走，请提醒它"先读 SKILL.md"。

## 8. 维护 skill 的入口

- **修触发词或 description**：编辑 `SKILL.md` 的 frontmatter（注意 description ≤ 1024 字符、第三人称）。
- **加新 op 的 mapping**：编辑 `reference.md` §10 的表格。
- **加新约束 / 反模式**：编辑 `SKILL.md` 的 Anti-patterns 节。
- **改某个模板**：直接改 `templates/*.tmpl`；改完后让 agent 在下个 op 移植时验证模板还能正确套用。
- **整理文件位置约定变化**（例如 aiter 改了 `csrc/cpp_itfs` 路径）：同步改 `SKILL.md` Step 5 的映射表与 `reference.md` §6。

## 9. 一次性快速调用：复制粘贴模板

下面这段可以直接发给 agent，让它按 skill 完整走完一个 op 的移植。把 `<op>` 与 CU 数填上即可：

```
请按 port-mi400-op-to-aiter skill 移植 poc_kl/mi400/<op>/ 到 aiter。

约束：
- 目标 arch: gfx1250，CU 数: <CU_NUM>
- 资产策略: co_only（.s/.sp3 留在 poc_kl）
- 路径与入口风格：按 skill 的 op→module 映射表自动选
- 必须做：补 GFX_CU_NUM_MAP、写 op_tests/test_<op>.py、走完 port_checklist.md
- 暂不做：实际跑 GPU 测试（我会另外在有 gfx1250 的机器上跑）

请先输出 discovery 报告再开工，待我确认后再继续后面的步骤。
```

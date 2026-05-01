# Step 7 - Symbol And Load Verification

## 目的

确认复制到 aiter HSA 目录的目标 `.co` 仍包含正确的 kernel 符号和 `.kd` 描述符，并记录当前环境下 runtime load 的可执行状态。

## 具体操作

1. 对 aiter 侧 `.co` 执行符号检查：
   - 文件：`aiter/hsa/gfx1250/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
   - 命令核心：`llvm-readelf -s ...`
2. 已确认符号表中存在：
   - `mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC`
   - `mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC.kd`
3. 检查 aiter 侧 HSA 目录：
   - `mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
   - `mla_asm.csv`
4. Runtime load 验证未在当前环境实际执行：
   - Step 6 已确认当前 `/usr/bin/python3` 缺少 `torch`，无法导入 `aiter` 并构建/调用 `module_mla_asm`。
   - 真实 `LoadKernel: ... hsaco: .../aiter/hsa/gfx1250/mla/...co` 日志需要在带 PyTorch/ROCm/gfx1250 的环境中通过 smoke test 验证。

## 结果

- aiter 侧 `.co` 符号校验通过。
- codegen dry-run 在 Step 3 已确认 `co_name` 会解析为 `mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`，runtime loader 将在设置 `AITER_ASM_DIR=$(pwd)/hsa` 后查找 `hsa/gfx1250/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`。
- Runtime load 受当前 Python/GPU 环境阻塞，留待 Step 8 的 smoke test 在目标环境执行。

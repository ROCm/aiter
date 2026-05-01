# Step 8 - Smoke Test And Regression Status

## 目的

新增一个最小化 smoke test，用于在真实 gfx1250 + PyTorch/ROCm 环境中验证目标 mi400 MLA `.co` 能通过 aiter loader 装载并完成 kernel launch，同时记录当前环境下无法实际运行的阻塞原因。

## 具体操作

1. 新增测试文件：`aiter/op_tests/test_mla_mi400.py`。
2. 测试约束：
   - 仅在 `get_gfx() == "gfx1250"` 时运行，否则 skip。
   - 设置 `AITER_ASM_DIR` 指向仓库内 `aiter/hsa`。
   - 固定 `batch=1`、`q_seq_len=4`、`kv_seq_len=578`、`page_size=64`、`num_kv_splits=1`。
   - 固定 `nhead=16`、`nhead_kv=1`、`qk_head_dim=576`、`v_head_dim=512`。
   - `q` 和 `kv_buffer` 使用 fp8。
   - `q_scale/kv_scale` 使用长度为 `batch` 的 fp32 tensor。
   - 显式传入 `num_kv_splits=1` 和 `num_kv_splits_indptr=[0, 1]`，避免 `get_meta_param()` 根据 CU 数自动选择 split。
3. 测试 docstring 明确说明：
   - 目标 `.co` 是 baked masked/`causal=1` 语义。
   - 当前 smoke 只验证 load/launch 和 shape，不验证 non-causal API 行为或数值正确性。
4. 使用 `ReadLints` 检查 `test_mla_mi400.py`，无 linter 报错。
5. 尝试运行：
   - 命令：`GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/repo/aiter/hsa python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q`
   - 当前环境在 pytest collection 阶段失败：`ModuleNotFoundError: No module named 'torch'`。

## 结果

- smoke test 文件已添加。
- 当前机器无法实际运行 smoke，阻塞点为 Python 环境缺少 `torch`。
- gfx950/gfx942 回归测试也未在当前环境运行，原因相同。
- 在目标环境中下一步应执行：
  - `GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/repo/aiter/hsa python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q`
  - 在 gfx950/gfx942 环境中回归现有 `aiter/op_tests/test_mla.py`。

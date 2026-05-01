# Step 3 - Place Assets And Codegen Dry-Run

## 目的

把 Step 2 生成的 `.co` 按 co-only 策略放入 aiter 的 `gfx1250` HSA 目录，创建调度 CSV，并确认 `aiter/hsa/codegen.py` 能发现该 CSV 并生成 `cfg_mla_asm` 的 gfx1250 行。

## 具体操作

1. 创建目标目录并复制 `.co`：
   - 目标目录：`aiter/hsa/gfx1250/mla/`
   - 复制产物：`mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
2. 新增 CSV：`aiter/hsa/gfx1250/mla/mla_asm.csv`
   - header 与 `gfx950/mla/mla_asm.csv` 对齐：
     `qType,kvType,Gqa,ps,qSeqLen,prefill,causal,lse,knl_name,co_name`
   - 新增 1 行：
     `fp8,fp8,16,0,4,0,1,0,mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC,mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
   - `causal=1` 用来反映目标 shader 的 baked masked 语义，不伪装为 non-causal。
3. 运行 codegen dry-run：
   - 系统 `/usr/bin/python3` 缺少 `numpy/pandas`，先用 `pip --target /tmp/aiter_codegen_deps` 临时安装依赖，不修改仓库依赖文件。
   - 命令使用：`PYTHONPATH=/tmp/aiter_codegen_deps AITER_GPU_ARCHS='gfx1250;gfx950;gfx942' python3 hsa/codegen.py -m mla --output_dir /tmp/aiter_mla_codegen`
4. 检查 `/tmp/aiter_mla_codegen/asm_mla_configs.hpp` 中的 gfx1250 行：
   - 已看到 `ADD_CFG("fp8", "fp8", 16, 0, 4, 0, 1, 0, "gfx1250", "mla/", "mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC", "mla_a8w8_qh16_1tg_16mx4_64nx1_np.co")`

## 结果

- 新增资产目录：`aiter/hsa/gfx1250/mla/`
- 新增 `.co`：`aiter/hsa/gfx1250/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
- 新增 CSV：`aiter/hsa/gfx1250/mla/mla_asm.csv`
- codegen dry-run 成功，`cfg_mla_asm` 能包含 gfx1250 的目标行。
- 后续 Step 4 可以开始修改 `asm_mla.cu`，新增 `MlaMi400KernelArgs`。

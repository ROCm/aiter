# Step 5 - Add GFX1250 Dispatcher Branch

## 目的

在 `mla_decode_stage1_asm_fwd` 中为 gfx1250 增加独立 mi400 分支，使用 Step 4 新增的 288B packed args，并通过 `cfg_mla_asm` 查找 Step 3 新增的 `.co/.csv` 行来启动目标 kernel。

## 具体操作

1. 修改 `aiter/csrc/py_itfs_cu/asm_mla.cu`。
2. 新增静态函数 `mla_decode_mi400_dispatch(...)`，参数与 `mla_decode_stage1_asm_fwd` 中 stage1 入口保持一致。
3. 在 `mla_decode_stage1_asm_fwd` 中取得 `arch_id = get_gpu_arch()` 后，若 `arch_id == "gfx1250"`，立即早返回到 `mla_decode_mi400_dispatch(...)`。
4. 在 mi400 dispatcher 内增加 guard：
   - 仅允许 `gfx1250`。
   - 仅允许 non-persistent decode：`work_meta_data/work_indptr/work_info_set == nullptr` 且 `num_kv_splits_indptr != nullptr`。
   - 暂不支持 `lse` 输出。
   - 仅允许 `fp8/fp8`。
   - 仅允许 `nhead_kv=1`、`gqa_ratio=16`、`max_seqlen_q=4`、`page_size=64`、`num_kv_splits=1`。
   - 仅允许 `Q` head dim 576、`output` head dim 512。
   - `q_scale/kv_scale` 必须非空、为 fp32，且长度至少覆盖 `batch`。
5. 通过 `get_heuristic_kernel_mla("fp8", "fp8", 16, 0, 0, 1, 4, "gfx1250", &cfg_mla_asm, 0)` 固定匹配 baked masked/`causal=1` 的 CSV 行。
6. 使用独立 `SynchronizedCache<std::string_view, AiterAsmKernel>` 缓存 mi400 kernel，避免与现有 MLA cache 混用。
7. 初始化 `MlaMi400KernelArgs`：
   - pointer 字段来自对应 tensor 的 `data_ptr()`。
   - `q_seq_lens = max_seqlen_q * gqa_ratio`。
   - `passes = splitData->size(1)`。
   - `stride_Q = Q->stride(0) * Q->element_size() * max_seqlen_q`，按 v3 host 语义填充。
   - `stride_page = KV->stride(0) * KV->element_size()`。
   - `ptr_QROPE/ptr_KVROPE` 按 v3 语义填 `q_scale/kv_scale`。
8. 启动参数：
   - `bdx=128, bdy=1, bdz=1`
   - `gdx=ceil(max_seqlen_q * gqa_ratio / 64)`
   - `gdy=batch`
   - `gdz=splitData->size(1)`
9. 使用 `ReadLints` 检查 `asm_mla.cu`，无 linter 报错。

## 结果

- `gfx1250` 运行时会进入新的 mi400 dispatcher。
- 现有 gfx942/gfx950 路径仍在早分支之后走原逻辑，未改原有 dispatcher 的核心分支。
- mi400 dispatcher 当前只覆盖 minimal smoke 的单形状、单 split、baked masked kernel。
- 后续 Step 6 需要补 `gfx1250` 的 build target/device name，并尝试 JIT 构建 `module_mla_asm`。

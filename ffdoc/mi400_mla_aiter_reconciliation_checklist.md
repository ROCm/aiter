# MI400 MLA 与 Aiter gfx1250 MLA 对账清单

日期：2026-05-13

## 目的

本文对账 `poc_kl/mi400/mla` 与当前 aiter gfx1250 MLA 接入状态，区分以下几类状态：

- 已复制到 aiter 的资产。
- 已注册到 aiter 的 CSV 元数据。
- 已真正接入 runtime 的路径。
- 已注册但主动阻断或尚未完成的路径。

## poc_kl 源端

源目录：

```text
/home/feifei/repo/poc_kl/mi400/mla
```

shader 源：

```text
/home/feifei/repo/poc_kl/mi400/mla/shaders/*.s
```

构建脚本：

```text
/home/feifei/repo/poc_kl/mi400/mla/run.sh
```

关键 host 文件：

- `mla.cpp`：根据 `model_version` 选择 v3 或 v4。
- `mla_execute_v3_hip.inl`：v3 HIP module launch 和 v3 kernarg layout。
- `mla_execute_v4_hip.inl`：v4 HIP module launch 和 v4 kernarg layout。
- `mla_helper.h`、`mla_v4.h`：v4 维度、QROPE/KVROPE 数据准备。

## poc_kl 中 v3/v4 的分流

`mla.cpp` 中的分流：

- `model_version=3` -> `RunV3Model` -> `do_compute_v3` -> `execute_v3_kernel`
- `model_version=4` -> `RunV4Model` -> `do_compute_v4` -> `execute_v4_kernel`

v3：

- 源端 host 支持 BF16 和 FP8。
- 当前 aiter gfx1250 runtime 只接了 fp8/fp8 变体。

v4：

- 源端 host 支持 FP8。
- v4 需要单独的 QROPE/KVROPE buffer 和 ABI。

## Kernarg ABI 对账

共同点：

- v3 和 v4 都是 18 个 slot。
- 总大小都是 288B。
- pointer slot 为 8B 指针加 padding。
- scalar slot 为 4B scalar 加 padding。

v3 layout：

- slot 0：R
- slot 1：LSE
- slot 2：Q
- slot 3：KV
- slot 4：kv_indptr
- slot 5：kv_page_indices
- slot 6：kv_last_page_lens
- slot 7：scalar
- slot 8：q_seq_lens
- slot 9：passes
- slot 10：stride_Q
- slot 11：stride_Page
- slot 12：log2_page
- slot 13：q_indptr
- slot 14：split_indptr
- slot 15：out_16_nosplit
- slot 16：descale_q
- slot 17：descale_k

v4 layout：

- slot 0：R
- slot 1：LSE
- slot 2：Q
- slot 3：KV
- slot 4：kv_indptr
- slot 5：kv_page_indices
- slot 6：kv_last_page_lens
- slot 7：scalar
- slot 8：q_seq_lens
- slot 9：passes
- slot 10：total_kv
- slot 11：stride_Page
- slot 12：log2_page
- slot 13：q_indptr
- slot 14：split_indptr
- slot 15：out_16_nosplit
- slot 16：qrope
- slot 17：kvrope

当前 aiter packed struct：

- `MlaMi400KernelArgs` 是 288B。
- 当前填参按 v3 语义实现。
- slot 10 填 `stride_Q`。
- slot 16/17 填 `q_scale` / `kv_scale`。

结论：

- v3 ABI 已接入。
- v4 ABI 尚未接入。
- v4 不能复用当前 v3 struct 或当前 `q_scale/kv_scale` 字段。

## 变体逐项对账

Variant 1：

- 源端：`mla_a8w8_qh16_1tg_16mx4_64nx1_np.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=3`，`variantId=1`。
- Runtime：默认变体，已接入。
- 状态：有可运行路径，需在 gfx1250 机器验证。

Variant 2：

- 源端：`mla_a8w8_qh16_1tg_16mx4_64nx1.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=3`，`variantId=2`。
- Runtime：`AITER_MLA_MI400_VARIANT_ID=2` 可选。
- 状态：v3 路径已接入。

Variant 3：

- 源端：`mla_a8w8_qh16_1tg_16mx4_64nx1_np_3p.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=3`，`variantId=3`。
- Runtime：`AITER_MLA_MI400_VARIANT_ID=3` 可选。
- 状态：v3 路径已接入。

Variant 4：

- 源端：`mla_a8w8_qh16_1tg_16mx4_64nx1_np_1tdm.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=3`，`variantId=4`。
- Runtime：`AITER_MLA_MI400_VARIANT_ID=4` 可选。
- 状态：v3 路径已接入。

Variant 5：

- 源端：`mla_a8w8_qh16_1tg_16mx4_128nx1_np_3p.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=3`，`variantId=5`，`passSize=128`。
- Runtime：`AITER_MLA_MI400_VARIANT_ID=5` 可选。
- 状态：v3 路径已接入。

Variant 6：

- 源端：`mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=3`，`variantId=6`，`qSeqLen=1`，`passSize=128`。
- Runtime：`AITER_MLA_MI400_VARIANT_ID=6` 可选。
- 状态：v3 路径已接入。

Variant 7：

- 源端：`mla_a8w8_qh16_1tg_16mx2_32nx4_np_3p.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=3`，`variantId=7`，`qSeqLen=2`，`passSize=128`。
- Runtime：`AITER_MLA_MI400_VARIANT_ID=7` 可选。
- 状态：v3 路径已接入。

Variant 8：

- 源端：`mla_a8w8_qh32_1tg_32mx4_64nx1.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=3`，`variantId=8`，`Gqa=32`。
- Runtime：`AITER_MLA_MI400_VARIANT_ID=8` 可选。
- 状态：v3 路径已接入。

Variant 9：

- 源端：`mla_a8w8_qh16_1tg_16mx4_64nx1_np_nm.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=4`，`variantId=9`。
- Runtime：主动阻断。
- 缺口：v4 packed args 和 QROPE/KVROPE 接口。

Variant 10：

- 源端：`mla_a8w8_qh16_1tg_16mx4_64nx1_sparse.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=4`，`variantId=10`。
- Runtime：主动阻断。
- 缺口：v4 packed args 和 QROPE/KVROPE 接口。

Variant 11：

- 源端：`mla_a8w8_qh16_1tg_16mx4_64nx1_sparse_msb_cycling.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=4`，`variantId=11`。
- Runtime：主动阻断。
- 缺口：v4 packed args 和 QROPE/KVROPE 接口。

Variant 12：

- 源端：`mla_a8w8_qh16_1tg_16mx4_64nx1_sparse_msb_cycling_pure_issue.s`
- aiter `.co`：已存在。
- CSV：`modelVersion=4`，`variantId=12`。
- Runtime：主动阻断。
- 缺口：v4 packed args 和 QROPE/KVROPE 接口。

额外注意：

- `run.sh` 中还有 `test_kl_mla_mtp0_sparse_test`，使用 `mla_a8w8_qh16_1tg_16mx1_32nx4_sparse.sp3`。
- 当前 `poc_kl/mi400/mla/shaders` 本次移植清单中没有对应 `.s`，因此 aiter 中未登记该项。

## Aiter 路径隔离对账

已完成：

- gfx1250 有独立 HSA 目录：`aiter/hsa/gfx1250/mla`。
- gfx1250 CSV 使用 mi400 符号名，即 `*_KERNEL_FUNC`。
- `mla_decode_stage1_asm_fwd` 会检查 `get_gpu_arch()`。
- `arch_id == "gfx1250"` 时提前进入 `mla_decode_mi400_dispatch`。
- 原 gfx9 逻辑位于 early return 之后。
- gfx1250 dispatcher 内再次检查 `arch_id == "gfx1250"`。
- gfx1250 dispatcher 使用独立的 `SynchronizedCache`。
- gfx1250 lookup 传入 `modelVersion` 和 `variantId`。
- v4 行不会被 v3 ABI 误启动，而是主动报错。

未完成：

- 没有 `MlaMi400V4KernelArgs`。
- 没有 `mla_decode_mi400_v4_dispatch`。
- 没有 QROPE/KVROPE 的 Python/C API。
- 没有 v4 smoke 或 correctness test。
- 没有 gfx1250 persistent MLA 路径。
- 没有 gfx1250 prefill MLA 路径。

## 验证对账

已完成：

- 使用 `amdclang++ --offload-arch=gfx1250` 编译 12 个 `.s`。
- 已复制 12 个 `.co` 到 `aiter/hsa/gfx1250/mla`。
- 已向 `mla_asm.csv` 加入 12 条 gfx1250 行。
- codegen dry-run 成功。
- codegen 输出包含 12 条 gfx1250 行。
- 已校验每条 CSV 都指向存在的 `.co`。
- 已校验每条 CSV 的 `knl_name` 都出现在对应 `.co` 的符号表中。

当前环境阻塞：

- `import torch` 失败：`ModuleNotFoundError: No module named 'torch'`。
- 因此未在本环境运行 JIT build 和 pytest smoke。

需要在 gfx1250 机器继续验证：

- 构建 `module_mla_asm`。
- 运行 `op_tests/test_mla_mi400.py`。
- 分别设置 `AITER_MLA_MI400_VARIANT_ID=1..8` 跑 smoke。
- v4 ABI 实现后再对 `variantId=9..12` 增加测试。

## 关闭 v4 缺口所需工作

1. 新增匹配 `poc_kl/mi400/mla/mla_execute_v4_hip.inl` 的 `MlaMi400V4KernelArgs`。
2. 新增 v4 dispatcher，填充：
   - `total_kv`
   - `stride_Page`
   - `ptr_qrope`
   - `ptr_kvrope`
3. 设计 Python API：
   - 显式传入 QROPE/KVROPE tensor，或
   - 定义可推导 QROPE/KVROPE 指针的 packed tensor layout。
4. 保持 v3/v4 在 `gfx1250` 分支内部继续分流，避免共用错误 ABI。
5. 为 `variantId=9..12` 增加 v4 smoke。
6. 重新执行 codegen、符号校验、JIT build 和 gfx1250 pytest。

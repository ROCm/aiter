# Step 1 - Discovery

## 目的

复核 `poc_kl/mi400/mla` 的真实 host/脚本/ABI 信息，并确认本次 minimal smoke 的 aiter 接入点，避免后续按错误的 kernel arg layout 或错误的 build 路径实施。

## 具体操作

- 复核 `poc_kl/mi400/mla/mla_execute_v3_hip.inl`，确认目标 `model_version=3` 的 kernarg 是 18 个 16B slot，总大小 288B。
- 复核 v3 slot 语义：offset 160 虽然 patched `.args` 名称显示为 `total_kv`，但 v3 host 实际填的是 `stride_Q`；offset 256/272 实际是 `descale_q/descale_k`。
- 复核 `poc_kl/mi400/mla/run.sh`，确认 `convert` 会调用 `sp3cvt` 后再运行 `patch_mla_kernargs.py`，`compile` 会用 `amdclang++ --offload-arch=gfx1250` 生成 `.co`。
- 复核 `poc_kl/mi400/mla/patch_mla_kernargs.py`，确认 patched `.args` 固定写入 18 个 slot 和 `KERNARG=288`。
- 复核 `aiter/csrc/py_itfs_cu/asm_mla.cu` 与 `aiter/csrc/include/aiter_tensor.h`，确认 `mla_decode_stage1_asm_fwd` 是本次 extend 的 C 入口，`aiter_tensor_t` 可通过 `size/stride/data_ptr/dtype/element_size` 初始化 packed args。

## 结果

- 本次移植继续选择 `mla_a8w8_qh16_1tg_16mx4_64nx1_np` 单变体。
- 必须在 aiter 中新增独立 `MlaMi400KernelArgs`，不能复用现有 `KernelArgs`。
- Dispatcher 必须为 gfx1250 单独分支，且只允许本次 non-persistent decode smoke 的形状和语义。
- 后续 Step 2 可以进入 `.co` 构建与符号/metadata 校验。

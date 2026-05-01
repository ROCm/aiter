# Step 4 - Add MI400 Packed Args Struct

## 目的

在 aiter 的 MLA asm 入口中新增 gfx1250/mi400 专用 packed args struct，保持与目标 `.s` 的 288B kernarg ABI 对齐，并避免复用现有 gfx942/gfx950 的 320B `KernelArgs`。

## 具体操作

1. 修改 `aiter/csrc/py_itfs_cu/asm_mla.cu`。
2. 在现有 `KernelArgs` 后新增 `MlaMi400KernelArgs`：
   - 18 个 16B slot。
   - pointer slot 使用 `void* + p2`。
   - scalar slot 使用 `unsigned int/float + p3`。
   - offset 160 的字段命名为 `stride_Q`，并添加注释说明 patched `.args` 名为 `total_kv`，但本次 v3 kernel 实际语义是 `stride_Q`。
   - offset 256/272 字段保留为 `ptr_QROPE/ptr_KVROPE`，后续 init 按 v3 的 `descale_q/descale_k` 填。
3. 添加 `static_assert(sizeof(MlaMi400KernelArgs) == 288, ...)`。
4. 对照目标 `.s` 的 `.args`：
   - `.kernarg_segment_size: 288`
   - offset 160: `.name: total_kv`
   - offset 256: `.name: QROPE`
   - offset 272: `.name: KVROPE`
5. 使用 `ReadLints` 检查 `asm_mla.cu`，无 linter 报错。

## 结果

- `aiter/csrc/py_itfs_cu/asm_mla.cu` 已包含 mi400 专用 `MlaMi400KernelArgs`。
- ABI 大小通过 `static_assert` 固化为 288B。
- 当前修改尚未接入 dispatcher，不影响现有 gfx942/gfx950 执行路径。
- 后续 Step 5 可以实现 `mla_decode_mi400_dispatch`，并在 `arch_id == "gfx1250"` 时早分流。

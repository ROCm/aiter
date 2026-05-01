# Step 2 - Build `.co` in `poc_kl`

## 目的

在 `poc_kl/mi400/mla` 侧生成本次 minimal smoke 需要提交到 aiter 的目标 code object，并确认 `.s` 元数据和 `.co` 符号满足 aiter loader 的要求。

## 具体操作

1. 执行 `cd /home/feifei/repo/poc_kl/mi400/mla && bash ./run.sh convert`。
   - 结果：脚本运行成功，但当前 `.s` 文件没有配套 `.text` 文件，`patch_mla_kernargs.py` 对所有现有 `.s` 输出 `skip (no .text)`，没有重新 patch。
   - 影响：继续使用仓库中已有的目标 `.s`，并单独校验其 metadata。
2. 校验 `shaders/mla_a8w8_qh16_1tg_16mx4_64nx1_np.s` metadata。
   - `.amdhsa_code_object_version 6`
   - `.amdhsa_kernarg_size 288`
   - `.kernarg_segment_size: 288`
   - `.amdhsa_group_segment_fixed_size 327680`
   - `.group_segment_fixed_size: 327680`
   - `.args` block 存在，18 个 slot 与计划一致。
3. 执行单文件编译：
   - `amdclang++ -ggdb -g -x assembler -target amdgcn--amdhsa --offload-arch=gfx1250 shaders/mla_a8w8_qh16_1tg_16mx4_64nx1_np.s -o mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
   - 编译成功，使用的 `amdclang++` 路径为 `/usr/bin/amdclang++`。
4. 执行符号校验：
   - `llvm-readelf -s mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
   - 已确认存在 `mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC`
   - 已确认存在 `mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC.kd`

## 结果

- 生成产物：`poc_kl/mi400/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
- 目标 `.s` 的 code object、kernarg、LDS metadata 通过检查。
- 目标 `.co` 的 kernel 符号与 `.kd` 描述符通过检查。
- 后续 Step 3 可以复制 `.co` 到 `aiter/hsa/gfx1250/mla/` 并创建 `mla_asm.csv`。

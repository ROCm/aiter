# 在 poc_kl 中编译并测试 mi400 MLA Kernel

日期：2026-05-13

## 目标 Kernel

当前要测试的 mi400 MLA kernel：

```text
mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC
```

对应文件：

```text
/home/carhuang/feifei/poc_kl/mi400/mla/shaders/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.s
/home/carhuang/feifei/poc_kl/mi400/mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co
```

对应 poc_kl 测试：

```text
kl_mla_mtp0_np_3p_k128_test
```

## 推荐方式：使用 run.sh 编译并测试

进入目录：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla
```

如果 ROCm 工具不在 PATH，先设置：

```bash
export PATH=/opt/rocm/bin:$PATH
```

编译 `.co` 和 `mla.out`：

```bash
bash run.sh compile
```

运行当前 kernel 对应测试：

```bash
bash run.sh test-one kl_mla_mtp0_np_3p_k128_test
```

期望看到：

```text
>>> PASS: kl_mla_mtp0_np_3p_k128_test
```

## 只编译当前单个 Kernel

如果不想编译全部 shader，可以只编译当前 `.s`：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla
export PATH=/opt/rocm/bin:$PATH

amdclang++ -ggdb -g -x assembler \
  -target amdgcn--amdhsa \
  --offload-arch=gfx1250 \
  shaders/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.s \
  -o mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co
```

然后编译 host 测试程序：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla
export PATH=/opt/rocm/bin:$PATH

hipcc -std=c++17 -ggdb -g mla.cpp -o mla.out \
  -I. -I../../common \
  -DFMHA_MI300=1
```

最后运行对应测试：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla
bash run.sh test-one kl_mla_mtp0_np_3p_k128_test
```

## 展开后的等价测试命令

`kl_mla_mtp0_np_3p_k128_test` 在 `run.sh` 中等价于：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla

./mla.out \
  lds_size=256 \
  vgpr_cnt=1024 \
  dbg_trace=1 \
  rope_split=1 \
  sub_Q=16 \
  batch=2 \
  num_kv_heads=1 \
  gqa_ratio=16 \
  kv_seq_lens=580 \
  dim=512 \
  block_size=1 \
  code_pfth=0 \
  init_pattern=0 \
  wv_tg=4 \
  atm_f32=0 \
  data_type=2 \
  passes=1 \
  pass_size=64 \
  seed=0 \
  out_16_nosplit=0 \
  cs0=mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.sp3 \
  sub_Q=16 \
  gqa_ratio=16 \
  kv_seq_lens=578 \
  new_split=0 \
  q_seq_lens=1 \
  mask=0 \
  out_16_nosplit=0 \
  rope_split=2 \
  block_size=64 \
  pattern_q=0 \
  pattern_k=0 \
  pass_size=128 \
  check_mode=1
```

注意：同一个参数在命令中出现多次时，后面的值覆盖前面的默认值。这与 `run.sh` 中的 `BASE_ARGS` 加测试特定参数的方式一致。

该测试进入 `parse_runtime_args()` 后会执行 `q_seq_lens *= gqa_ratio`，因此 kernel 实际看到：

```text
q_seq_lens = 1 * 16 = 16
stride_Q = 1 * 16 * 576 * sizeof(fp8) = 9216
stride_Page = 64 * 576 * sizeof(fp8) = 36864
log2_page = 6
grid = (ceil(16 / 16), batch, passes) = (1, 2, 1)
block = (128, 1, 1)
```

## 符号与产物检查

确认 `.co` 存在：

```bash
ls -l /home/carhuang/feifei/poc_kl/mi400/mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co
```

确认 kernel 符号存在：

```bash
llvm-readelf -s /home/carhuang/feifei/poc_kl/mi400/mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co | \
  /usr/bin/python3 -c 'import sys; s=sys.stdin.read(); print("OK" if "mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC" in s else s)'
```

确认 host 程序存在：

```bash
ls -l /home/carhuang/feifei/poc_kl/mi400/mla/mla.out
```

## 与 aiter 测试的对应关系

poc_kl：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla
bash run.sh test-one kl_mla_mtp0_np_3p_k128_test
```

aiter：

```bash
cd /home/carhuang/feifei/aiter
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -q -s
```

如果 aiter 失败而 poc_kl 通过，优先检查 aiter 侧：

- `.co` 是否来自同一个 `poc_kl` 编译产物。
- CSV 中 `knl_name` 是否是 `mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC`。
- CSV 中 `co_name` 是否是 `mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co`。
- `AITER_ASM_DIR` 是否指向 `/home/carhuang/feifei/aiter/hsa`。
- aiter 的 packed args 是否与 poc_kl v3 ABI 一致。

# gfx1250 MLA 失败时如何对照 poc_kl Debug

日期：2026-05-13

## 背景

已知 golden 源端：

```text
/home/carhuang/feifei/poc_kl/mi400/mla
```

该目录中的 `mi400/mla` 已验证通过，可以作为 aiter 中 `op_tests/test_mla_mi400.py` 失败时的对照基准。

当前单 kernel 对应关系：

- aiter 测试：`op_tests/test_mla_mi400.py`
- poc_kl 测试：`bash run.sh test-one kl_mla_tp_np_test`
- kernel：`mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC`
- code object：`mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
- poc_kl shader：`poc_kl/mi400/mla/shaders/mla_a8w8_qh16_1tg_16mx4_64nx1_np.s`

## 对照原则

先判断失败发生在哪一层：

1. aiter Python/JIT/build/import 失败。
2. aiter loader 找不到 `.co` 或符号。
3. kernel launch 失败或 GPU fault。
4. launch 成功但输出 NaN/Inf。
5. shape 通过但 numerics 不过。

poc_kl 已通过时，优先怀疑 aiter 侧：

- `.co` 是否和 poc_kl 编译产物一致。
- CSV 中 `knl_name/co_name` 是否正确。
- `AITER_ASM_DIR` 是否指向正确 HSA 目录。
- `MlaMi400KernelArgs` 是否与 poc_kl v3 ABI 完全一致。
- aiter 填参公式是否与 `execute_v3_kernel` 一致。
- grid/block 是否与 poc_kl launch geometry 一致。
- Python 输入 layout 是否与 poc_kl host 准备的数据语义一致。
- stage2 reduction 是否错误处理了 split output。

## Step 1：先确认 poc_kl golden 仍然通过

在源端构建：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla
bash run.sh compile
```

运行与 aiter 单 kernel 对应的 poc_kl 测试：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla
bash run.sh test-one kl_mla_tp_np_test
```

该测试在 `run.sh` 中的参数为：

```text
cs0=mla_a8w8_qh16_1tg_16mx4_64nx1_np.sp3
sub_Q=64
gqa_ratio=16
kv_seq_lens=578
new_split=0
q_seq_lens=4
mask=1
out_16_nosplit=0
rope_split=2
block_size=64
pattern_q=0
pattern_k=0
check_mode=1
```

如果 poc_kl 失败，先不要排 aiter；需要先恢复 golden。

如果当前要测试 MTP0 3p k128 路径，则使用：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla
bash run.sh test-one kl_mla_mtp0_np_3p_k128_test
```

该测试在 `run.sh` 中的覆盖参数为：

```text
cs0=mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.sp3
sub_Q=16
gqa_ratio=16
kv_seq_lens=578
new_split=0
q_seq_lens=1
mask=0
out_16_nosplit=0
rope_split=2
block_size=64
pattern_q=0
pattern_k=0
pass_size=128
check_mode=1
```

poc_kl host 会在 `parse_runtime_args()` 中执行 `q_seq_lens *= gqa_ratio`，所以该测试 kernel 实际看到：

```text
q_seq_lens = 16
stride_Q = 1 * 16 * 576 * sizeof(fp8) = 9216
stride_Page = 64 * 576 * sizeof(fp8) = 36864
log2_page = 6
grid = (1, 2, 1)
block = (128, 1, 1)
kernel = mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC
co = mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co
```

## Step 2：运行 aiter 测试并保留完整日志

基础运行：

```bash
cd /home/feifei/Desktop/repo/aiter
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/Desktop/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -q -s
```

如果出现异步 GPU 错误，打开同步调试：

```bash
cd /home/feifei/Desktop/repo/aiter
HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3 \
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/Desktop/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -q -s
```

只跑 smoke：

```bash
cd /home/feifei/Desktop/repo/aiter
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/Desktop/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k minimal_smoke -q -s
```

只跑数值：

```bash
cd /home/feifei/Desktop/repo/aiter
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/Desktop/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k numerics -q -s
```

## Step 3：确认 aiter 加载的是正确 code object

检查 CSV：

```bash
cd /home/feifei/Desktop/repo/aiter
python3 - <<'PY'
import csv
from pathlib import Path
p = Path("hsa/gfx1250/mla/mla_asm.csv")
rows = list(csv.DictReader(p.open()))
for r in rows:
    print(r)
PY
```

单 kernel 版本应至少有类似行：

```text
knl_name=mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC
co_name=mla_a8w8_qh16_1tg_16mx4_64nx1_np.co
```

检查 `.co` 是否存在：

```bash
ls -l /home/feifei/Desktop/repo/aiter/hsa/gfx1250/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co
```

检查符号：

```bash
llvm-readelf -s /home/feifei/Desktop/repo/aiter/hsa/gfx1250/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co | \
  /usr/bin/python3 -c 'import sys; s=sys.stdin.read(); print("OK" if "mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC" in s else s)'
```

对照 poc_kl 产物：

```bash
cd /home/feifei/repo/poc_kl/mi400/mla
llvm-readelf -s mla_a8w8_qh16_1tg_16mx4_64nx1_np.co | \
  /usr/bin/python3 -c 'import sys; s=sys.stdin.read(); print("OK" if "mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC" in s else s)'
```

如果 aiter 的 `.co` 可疑，重新从 poc_kl 编译并复制：

```bash
cd /home/feifei/repo/poc_kl/mi400/mla
amdclang++ -ggdb -g -x assembler -target amdgcn--amdhsa --offload-arch=gfx1250 \
  shaders/mla_a8w8_qh16_1tg_16mx4_64nx1_np.s \
  -o mla_a8w8_qh16_1tg_16mx4_64nx1_np.co

cp mla_a8w8_qh16_1tg_16mx4_64nx1_np.co \
  /home/feifei/Desktop/repo/aiter/hsa/gfx1250/mla/
```

## Step 4：确认 codegen 生成了正确配置

运行 codegen dry-run：

```bash
cd /home/feifei/Desktop/repo/aiter
mkdir -p /tmp/aiter_mla_codegen_debug
AITER_GPU_ARCHS=gfx1250 python3 hsa/codegen.py -m mla --output_dir /tmp/aiter_mla_codegen_debug
```

如果缺少 `numpy/pandas`，用临时依赖目录：

```bash
python3 -m pip install --target /tmp/aiter_codegen_deps numpy pandas
cd /home/feifei/Desktop/repo/aiter
PYTHONPATH=/tmp/aiter_codegen_deps AITER_GPU_ARCHS=gfx1250 \
  python3 hsa/codegen.py -m mla --output_dir /tmp/aiter_mla_codegen_debug
```

检查生成头：

```bash
/usr/bin/python3 - <<'PY'
from pathlib import Path
p = Path("/tmp/aiter_mla_codegen_debug/asm_mla_configs.hpp")
text = p.read_text()
for needle in [
    "gfx1250",
    "mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC",
    "mla_a8w8_qh16_1tg_16mx4_64nx1_np.co",
]:
    print(needle, "OK" if needle in text else "MISSING")
PY
```

若生成头缺少行，问题在 CSV、`AITER_GPU_ARCHS` 或 `hsa/codegen.py` 输入目录。

## Step 5：对照 v3 kernarg ABI

poc_kl v3 ABI 在：

```text
/home/feifei/repo/poc_kl/mi400/mla/mla_execute_v3_hip.inl
```

关键布局：

```text
slot 0  ptr_r
slot 1  ptr_lse
slot 2  ptr_q
slot 3  ptr_kv
slot 4  ptr_ltp              kv_indptr
slot 5  ptr_ltd              kv_page_indices
slot 6  ptr_ltl              kv_last_page_lens
slot 7  scalar
slot 8  q_seq_lens
slot 9  passes
slot 10 stride_Q
slot 11 stride_Page
slot 12 log2_page
slot 13 ptr_qtp              qo_indptr
slot 14 ptr_stp              split_indptr / num_kv_splits_indptr
slot 15 out_16_nosplit
slot 16 ptr_descale_q
slot 17 ptr_descale_k
```

aiter 必须满足：

```text
sizeof(MlaMi400KernelArgs) == 288
q_seq_lens = max_seqlen_q * gqa_ratio
passes = num_kv_splits
stride_Q = nhead_kv * max_seqlen_q * gqa_ratio * qk_head_dim * sizeof(fp8)
stride_page = KV.stride(0) * KV.element_size()
log2_page = log2(page_size)
out_16_nosplit = 0
ptr_QROPE = q_scale.data_ptr()  # v3 descale_q
ptr_KVROPE = kv_scale.data_ptr() # v3 descale_k
```

如果 smoke 发生 GPU fault，优先检查：

- `splitData` 是否是 fp32，不可复用 bf16 `out.view(...)`。
- `splitLse` 是否是 fp32。
- `q_seq_lens` 是否错误填成 `max_seqlen_q * gqa_ratio`。
- `stride_Q` 是否错误乘了 `num_heads`。
- `KV` 的 `stride(0)` 是否符合 page layout。
- `kv_indices` 是否至少覆盖 `num_pages`。
- `kv_last_page_lens` 是否为 `kv_seq_len % page_size`，如果整除时需要确认是否应为 page_size。

建议临时在 aiter `mla_decode_mi400_dispatch` 中打印这些字段，与 poc_kl `execute_v3_kernel` 中打印的 `stride_Q/stride_Page/log2_page` 对齐。

## Step 6：对照 shape 与测试参数

`test_mla_mi400.py` 当前构造的关键参数应对齐 `kl_mla_tp_np_test`：

```text
batch = 1
q_seq_len = 4
kv_seq_len = 578
page_size = 64
num_kv_splits = 1
nhead = 16
nhead_kv = 1
gqa_ratio = 16
qk_head_dim = 576
v_head_dim = 512
```

poc_kl 对照参数：

```text
sub_Q=64
gqa_ratio=16
kv_seq_lens=578
q_seq_lens=4
mask=1
rope_split=2
block_size=64
passes=1
pass_size=64
out_16_nosplit=0
data_type=2
```

注意：poc_kl 测试自己生成输入数据，aiter 测试用 PyTorch 随机数据。两边默认不是同一份输入，所以不能直接逐元素比较 poc_kl output 与 aiter output，除非额外做输入 dump/import。

## Step 7：区分 kernel stage1 失败还是 stage2 失败

`test_mla_mi400.py` 调用：

```text
aiter.mla.mla_decode_fwd
  -> aiter.mla_decode_stage1_asm_fwd
  -> Triton stage2 reduction
```

如果 `attn_logits` / `attn_lse` 已经 NaN/Inf：

- 问题多半在 stage1 kernel、kernarg、输入 layout 或 `.co`。

如果 `attn_logits` / `attn_lse` 正常，但 `out` 数值不对：

- 问题可能在 stage2 reduction、mask 语义、lse 处理或 reference。

可以临时在测试中增加：

```python
print("out finite", torch.isfinite(case["out"].float()).all())
print("attn_logits finite", torch.isfinite(attn_logits.float()).all())
print("attn_lse finite", torch.isfinite(attn_lse.float()).all())
print("out max/min", case["out"].float().max(), case["out"].float().min())
print("logits max/min", attn_logits.float().max(), attn_logits.float().min())
print("lse max/min", attn_lse.float().max(), attn_lse.float().min())
```

## Step 8：做精确对照时，导出相同输入

如果需要让 aiter 与 poc_kl 做逐元素对照，需要让两边使用同一份输入。建议流程：

1. 在 aiter 测试中 dump 以下 tensor：
   - Q fp8
   - KV fp8 page buffer
   - q_scale
   - kv_scale
   - qo_indptr
   - kv_indptr
   - kv_indices
   - kv_last_page_lens
   - num_kv_splits_indptr
2. 在 poc_kl 添加一个临时 import 路径，跳过 `prepare_v3_host_inputs` 的随机生成，直接读取这些 dump。
3. 让 poc_kl `execute_v3_kernel` 跑同一份输入。
4. 对比：
   - stage1 R / splitData
   - stage1 LSE / splitLse
   - final output

没有同输入前，poc_kl 的主要作用是校验：

- kernel 本身可执行。
- v3 ABI 和 launch geometry 的源端公式。
- 目标 case 的 mask/shape/pass 参数。

## 常见失败与判断

### 找不到模块或 JIT build 失败

检查：

```bash
cd /home/feifei/Desktop/repo/aiter
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  python3 -c "from aiter.jit.core import build_module; build_module('module_mla_asm')"
```

如果 codegen 缺行，回到 Step 4。

### 找不到 hsaco 或 kernel symbol

检查：

```bash
echo "$AITER_ASM_DIR"
ls -l /home/feifei/Desktop/repo/aiter/hsa/gfx1250/mla/
llvm-readelf -s /home/feifei/Desktop/repo/aiter/hsa/gfx1250/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co
```

### GPU page fault

优先检查：

- `splitData` 是否 fp32。
- `stride_Q` 是否等于 `1 * 4 * 16 * 576 * 1 = 36864`。
- `stride_page` 是否等于 `64 * 1 * 576 * 1 = 36864`，具体以 KV 实际 layout 的 `stride(0) * element_size()` 为准。
- `q_seq_lens` 是否为 `4 * 16 = 64`。
- grid 是否为 `gdx=1, gdy=batch, gdz=1`。
- block 是否为 `bdx=128, bdy=1, bdz=1`。

### 输出全 NaN 或 cosine 不过

优先检查：

- causal mask 是否与 poc_kl `mask=1` 对齐。
- reference 是否使用同样的 fp8 scale 语义。
- `q_scale/kv_scale` 是否是 v3 descale 指针，长度至少覆盖 batch。
- `kv_last_page_lens` 在余数为 0 时是否需要特殊处理。
- stage2 reduction 是否在 `num_kv_splits=1` 时错误复用了 output buffer。

## 最小恢复策略

如果怀疑 aiter 当前状态被改坏，先恢复到最小单 kernel 可调试状态：

1. 只保留 CSV 中 `mla_a8w8_qh16_1tg_16mx4_64nx1_np` 这一行。
2. 确认 `.co` 来自 poc_kl 最新编译产物。
3. 确认 `mla_decode_stage1_asm_fwd` 中 `gfx1250` 早返回到 mi400 dispatcher。
4. 确认 mi400 dispatcher 使用 v3 ABI，不要混入 v4 的 `total_kv/qrope/kvrope`。
5. 先跑 smoke，再跑 numerics。

## Debug 结束条件

可认为单 kernel 路径恢复的条件：

- poc_kl `bash run.sh test-one kl_mla_tp_np_test` 通过。
- aiter codegen 能生成目标 gfx1250 行。
- aiter loader 能找到 `.co` 和 `*_KERNEL_FUNC`。
- `test_mla_mi400_minimal_smoke` 通过。
- `test_mla_mi400_numerics` 通过或数值误差在文档化阈值内。

# MLA mi400 (gfx1250) 测试指引

本文档记录当前 `op_tests/test_mla_mi400.py` 的测试范围、参数组合和常用运行命令。

当前测试覆盖的 kernel 是：

```text
mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC
```

它来自 poc_kl 的 `kl_mla_mtp0_np_3p_k128_test`，对应 mi400/gfx1250 MLA v3 decode 路径：

```text
q_seq_len = 1
nhead = 16
nhead_kv = 1
gqa_ratio = 16
qk_head_dim = 576
v_head_dim = 512
page_size = 64
num_kv_splits = 1
dtype = fp8 / fp8 input, bf16 output
non-causal
```

注意：当前 `aiter/mla.py` 的 gfx1250 stage2 仍是临时 PyTorch debug 实现，日志会打印：

```text
[aiter][mi400][debug] _fwd_kernel_stage2_asm replaced by temporary PyTorch stage2
```

因此本文档的测试结果用于验证 stage1 kernel + 临时 stage2 reduction 的数值正确性，不代表原 Triton stage2 已验证通过。

## 测试项目

`test_mla_mi400.py` 现在包含两个 pytest 项目，二者都对同一组参数矩阵做参数化。

```text
test_mla_mi400_minimal_smoke
  运行 mla_decode_fwd，检查 output / attn_logits shape，
  并确认不请求 return_lse 时 attn_lse 为 None。

test_mla_mi400_numerics
  运行 mla_decode_fwd(return_lse=True)，检查 output / attn_logits / attn_lse finite，
  再用 PyTorch reference 对 case["out"] 做 cosine diff 校验。
```

每个 case 会执行一次 smoke 和一次 numerics。默认自生成输入模式下共 5 个 case，因此完整测试数量是：

```text
5 cases * 2 tests = 10 tests
```

## 固定参数

当前 mi400 dispatcher 和该 `.co` 只覆盖以下固定参数：

```text
q_seq_len = 1
page_size = 64
num_kv_splits = 1
nhead = 16
nhead_kv = 1
gqa_ratio = 16
qk_head_dim = 576
v_head_dim = 512
Q dtype = fp8
KV dtype = fp8
q_scale dtype = fp32
kv_scale dtype = fp32
splitData dtype = fp32
splitLse dtype = fp32
output dtype = bf16
```

不要在当前测试里随意扩展 `q_seq_len`、`nhead`、`nhead_kv`、`page_size`、`num_kv_splits` 或 head dim。C++ dispatcher 对这些字段有 strict assert，超出范围不是当前 kernel 支持的有效测试。

## 参数化 Case 列表

当前 `_MI400_CASES` 包含以下 5 个 case：

```text
1. poc-kl-dump-shape
   batch = 2
   kv_seq_len = 578
   shuffle_pages = True
   page_indices_oob = 4
   use_non_unit_scales = False

2. single-batch-short-kv
   batch = 1
   kv_seq_len = 65
   shuffle_pages = True
   page_indices_oob = 4
   use_non_unit_scales = False

3. two-batch-two-pages-contiguous
   batch = 2
   kv_seq_len = 128
   shuffle_pages = False
   page_indices_oob = 4
   use_non_unit_scales = False

4. two-batch-long-kv-scaled
   batch = 2
   kv_seq_len = 578
   shuffle_pages = True
   page_indices_oob = 4
   use_non_unit_scales = True

5. three-batch-partial-page
   batch = 3
   kv_seq_len = 257
   shuffle_pages = True
   page_indices_oob = 4
   use_non_unit_scales = False
```

覆盖点：

```text
poc-kl-dump-shape:
  保持与 poc_kl dump replay 的 canonical shape 一致。

single-batch-short-kv:
  覆盖 batch=1、短 KV、partial last page。

two-batch-two-pages-contiguous:
  覆盖整页 KV，kv_last_page_lens 应为 page_size=64，而不是 0；
  同时覆盖不 shuffle 的连续 page table。

two-batch-long-kv-scaled:
  覆盖长 KV 和非 1.0 q_scale/kv_scale。

three-batch-partial-page:
  覆盖 batch=3、grid.y=3、多 batch partial page。
```

## 输入 Layout

测试里明确区分 logical reference layout 和 kernel packed layout。

```text
case["q_ref"], case["kv_buffer_ref"]:
  logical layout，传给 _ref_mla_mi400() PyTorch reference。

case["q"], case["kv_buffer"]:
  rope_split=2 packed layout，传给 aiter kernel。
```

`rope_split=2` packing 规则：

```text
Q:
  对每个 batch、q token，把 16 个 GQA head 的 nope(512) 连续放前面，
  再把 16 个 GQA head 的 rope(64) 连续放后面。

KV:
  对每个 physical page，把 page_size=64 个 token 的 nope(512) 连续放前面，
  再把 64 个 token 的 rope(64) 连续放后面。
```

这点很重要：PyTorch reference 应该使用 logical tensor，kernel 应该使用 packed tensor。不要把 packed tensor 直接拿去当数学 reference。

## 默认测试命令

默认模式不读取 poc_kl dump，所有输入由 `test_mla_mi400.py` 自己生成。

```bash
docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
  env -u AITER_MLA_LOAD_POC_DUMP_DIR -u AITER_MLA_DEBUG_DUMP_DIR \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  timeout 300s python3 -m pytest op_tests/test_mla_mi400.py -q -s'
```

最新结果：

```text
10 passed in 2.92s
```

## 只跑 Smoke

```bash
docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
  env -u AITER_MLA_LOAD_POC_DUMP_DIR -u AITER_MLA_DEBUG_DUMP_DIR \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  timeout 180s python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q -s'
```

预期：

```text
5 passed
```

## 只跑 Numerics

```bash
docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
  env -u AITER_MLA_LOAD_POC_DUMP_DIR -u AITER_MLA_DEBUG_DUMP_DIR \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  timeout 180s python3 -m pytest op_tests/test_mla_mi400.py -k numerics -q -s'
```

预期：

```text
5 passed
```

## poc_kl Dump Replay 测试命令

如果已经在 `/tmp/poc_kl_mla_debug_dump` 准备好 poc_kl dump，可以运行 replay 模式：

```bash
docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
  AITER_MLA_LOAD_POC_DUMP_DIR=/tmp/poc_kl_mla_debug_dump \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  timeout 180s python3 -m pytest op_tests/test_mla_mi400.py -q -s'
```

最新结果：

```text
2 passed, 8 skipped in 2.83s
```

这里 `8 skipped` 是预期行为。poc_kl dump replay 只支持 canonical `poc-kl-dump-shape`，其余 4 个参数化 case 在 smoke 和 numerics 中都会跳过。

replay 模式读取：

```text
q.bin
kv_buffer.bin
q_scale.bin
kv_scale.bin
```

numerics expected 使用 poc_kl 的 `splitData.bin` raw prefix。

## 必需环境变量

```text
ROCM_HOME=/opt/rocm
ENABLE_CK=0
GPU_ARCHS=gfx1250
AITER_GPU_ARCHS=gfx1250
AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa
```

说明：

```text
ENABLE_CK=0:
  避免 gfx1250 下 CK 编译路径触发不支持的 arch assert。

ROCM_HOME=/opt/rocm:
  确保 JIT link 时能找到 ROCm runtime。

GPU_ARCHS / AITER_GPU_ARCHS:
  固定编译和 runtime dispatch arch 为 gfx1250。

AITER_ASM_DIR:
  指向当前仓库里的 hsa 目录，确保 loader 使用本地 mi400 .co。
```

## JIT 重编译

如果修改了 `csrc/py_itfs_cu/asm_mla.cu` 或相关 JIT C++ 代码，测试前清理：

```bash
docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so'
```

然后重新运行 pytest。否则可能复用旧的 dispatcher 或 debug instrumentation。

## 当前注意事项

```text
1. stage2 仍是临时 PyTorch debug 实现。
2. C++ dispatcher 仍保留大量 [aiter][mi400][debug] 打印。
3. GPU finite 检查在测试中使用 CPU copy，避免 torch.isfinite GPU reduction 挂起。
4. dump replay 通过只说明 canonical poc_kl dump shape 下与 poc_kl 对齐；
   默认随机输入通过说明当前 PyTorch reference 与 test input packing 自洽。
5. 后续恢复原 Triton stage2 后，需要重新运行本文档中的默认测试和 replay 测试。
```

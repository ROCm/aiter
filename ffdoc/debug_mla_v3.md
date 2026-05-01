# Debug MLA v3 kernel 参数打印

日期：2026-05-13

## 目的

记录当前为了对比 aiter 与 poc_kl 的 mi400 MLA v3 kernel 参数而加入的临时 debug 改动，以及两边分别如何运行来打印参数。

当前重点测试：

```text
kl_mla_mtp0_np_3p_k128_test
```

对应 poc_kl kernel：

```text
mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC
```

## 当前结论

截至本次对比，aiter 与 poc_kl 的 mi400 MLA v3 host 传参已经对齐，未发现 kernarg 或 launch 参数 mismatch。
进一步让 aiter 直接读取 poc_kl dump 出来的 `q/kv_buffer/q_scale/kv_scale` 后，aiter stage1 的 `splitData/splitLse` 与 poc_kl 的 raw prefix 完全一致，说明当前 stage1 kernel 输出已与 poc_kl 对齐。
再将 `mla.py` 中 gfx1250 的 stage2 临时替换为 PyTorch 实现后，完整 `out` numerics 也可在共享 poc_kl 输入/expected 的模式下通过，说明当前失败点已经不在 stage1；下一步应回到原 Triton stage2 对比其输入/输出和 reduction 逻辑。
最新一次清理 `module_mla_asm` JIT 缓存并重新编译后，`op_tests/test_mla_mi400.py` 中 smoke 和 numerics 两个测试都已执行并通过：`2 passed in 10.23s`。
2026-05-15 复测：保持 `AITER_MLA_LOAD_POC_DUMP_DIR=/tmp/poc_kl_mla_debug_dump`，即继续使用 poc_kl dump replay 输入和 poc_kl `splitData.bin` raw prefix expected；临时 PyTorch stage2 仍启用。完整运行 `op_tests/test_mla_mi400.py` 再次通过：`2 passed in 2.72s`。
随后继续对齐 `_ref_mla_mi400()` 与 poc_kl 的 CPU 验证语义，定位到原始随机输入未按 `rope_split=2` 做 kernel layout packing：poc_kl 在执行 kernel 前会把每个 Q group / KV page 的 `nope(512)` 连续放在前面，再把 `rope(64)` 连续放在后面；PyTorch reference 应使用未 pack 的 logical tensor，kernel 输入应使用 pack 后 tensor。修正 Q 和 KV page 的 `rope_split=2` packing 后，不设置 `AITER_MLA_LOAD_POC_DUMP_DIR` 的原始随机输入模式也通过：`2 passed in 2.84s`。poc dump replay 模式回归也通过：`2 passed in 2.74s`。
2026-05-15 最新复测：显式取消 `AITER_MLA_LOAD_POC_DUMP_DIR` 和 `AITER_MLA_DEBUG_DUMP_DIR`，确认测试不读取 poc_kl dump 输入，而是由 `test_mla_mi400.py`/aiter 自己生成随机 Q/KV 输入；完整运行 smoke 与 numerics 通过：`2 passed in 2.98s`。当前日志仍显示 `_fwd_kernel_stage2_asm replaced by temporary PyTorch stage2`，即 stage2 仍是临时 PyTorch debug 实现，不是原 Triton stage2。

已确认一致的关键项：

```text
kernel symbol = mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC
batch = 2
num_heads = 16
nhead_kv = 1
gqa_ratio = 16
q_seq_lens = 16
passes = 1
scalar = 0.0416667
stride_Q = 9216
stride_page = 36864
log2_page = 6
out_16_nosplit = 0
arg offsets = 0, 16, ..., 272
launch grid = (1, 2, 1)
launch block = (128, 1, 1)
```

唯一需要注意的是 KV shape 的打印口径不同：

```text
aiter:  KV=(20,64,1,576)              # 两个 batch 的总 pages
poc_kl: KV=(10,64,1,576)              # 每 batch 的 max_num_blocks
poc_kl raw: K_buf_shuffle=(2,1,640,576)
```

由于两边 `stride_page=36864`、`grid=(1,2,1)`、kernarg offset/value 都一致，这个 KV shape 打印差异目前不认为是直接错误。如果后续打开 kernel launch 后数值仍不一致，下一步优先 dump 并对比：

```text
kv_indptr
kv_page_indices
kv_last_page_lens
Q/KV raw input
splitData / splitLse
```

## Buffer dump 调试

两边都已加入环境变量控制的 raw buffer dump，默认不启用。

### aiter dump

实现位置：

```text
/home/carhuang/feifei/aiter/aiter/mla.py
```

环境变量：

```text
AITER_MLA_DEBUG_DUMP_DIR=/tmp/aiter_mla_debug_dump
```

运行命令：

```bash
docker exec ff_sp3 bash -c "rm -rf /tmp/aiter_mla_debug_dump && \
  mkdir -p /tmp/aiter_mla_debug_dump && \
  cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  AITER_MLA_DEBUG_DUMP_DIR=/tmp/aiter_mla_debug_dump \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q -s"
```

aiter 会在 `mla_decode_fwd()` 中 stage1 返回后 dump：

```text
q.bin / q.meta.txt
kv_buffer.bin / kv_buffer.meta.txt
qo_indptr.bin / qo_indptr.meta.txt
kv_indptr.bin / kv_indptr.meta.txt
kv_page_indices.bin / kv_page_indices.meta.txt
kv_last_page_lens.bin / kv_last_page_lens.meta.txt
num_kv_splits_indptr.bin / num_kv_splits_indptr.meta.txt
q_scale.bin / q_scale.meta.txt
kv_scale.bin / kv_scale.meta.txt
splitData.bin / splitData.meta.txt
splitLse.bin / splitLse.meta.txt
output.bin / output.meta.txt
final_lse.bin / final_lse.meta.txt  # 仅 return_lse=True 时存在
```

每个 `.bin` 是 contiguous raw tensor bytes；`.meta.txt` 记录 dtype、shape、stride、element_size、numel、nbytes。

### poc_kl dump

实现位置：

```text
/home/carhuang/feifei/poc_kl/mi400/mla/mla_execute_v3_hip.inl
```

环境变量：

```text
POC_KL_MLA_DEBUG_DUMP_DIR=/tmp/poc_kl_mla_debug_dump
```

运行命令：

```bash
docker exec ff_sp3 bash -c "rm -rf /tmp/poc_kl_mla_debug_dump && \
  mkdir -p /tmp/poc_kl_mla_debug_dump && \
  cd /home/carhuang/feifei/poc_kl/mi400/mla && \
  export PATH=/opt/rocm/bin:\$PATH && \
  bash run.sh compile >/tmp/poc_kl_compile_debug.log && \
  POC_KL_MLA_DEBUG_DUMP_DIR=/tmp/poc_kl_mla_debug_dump \
  bash run.sh test-one kl_mla_mtp0_np_3p_k128_test"
```

poc_kl 会在 v3 kernel 执行并把 `d_r/d_lse` copy 回 host 后 dump：

```text
q.bin / q.meta.txt                    # buffers.Q_buf
kv_buffer.bin / kv_buffer.meta.txt    # buffers.K_buf_shuffle
qo_indptr.bin / qo_indptr.meta.txt    # buffers.q_indptr
kv_indptr.bin / kv_indptr.meta.txt
kv_page_indices.bin / kv_page_indices.meta.txt
kv_last_page_lens.bin / kv_last_page_lens.meta.txt
num_kv_splits_indptr.bin / num_kv_splits_indptr.meta.txt  # buffers.split_indptr
q_scale.bin / q_scale.meta.txt        # descale_q
kv_scale.bin / kv_scale.meta.txt      # descale_k
splitData.bin / splitData.meta.txt    # gpu_split_Data after kernel copy-back
splitLse.bin / splitLse.meta.txt      # gpu_split_Lse after kernel copy-back
```

### 已验证 dump 文件

已在 `ff_sp3` 中验证两边均能生成 dump 文件。

aiter dump 目录：

```text
/tmp/aiter_mla_debug_dump
```

poc_kl dump 目录：

```text
/tmp/poc_kl_mla_debug_dump
```

两边共同生成的核心文件：

```text
q.bin
kv_buffer.bin
qo_indptr.bin
kv_indptr.bin
kv_page_indices.bin
kv_last_page_lens.bin
num_kv_splits_indptr.bin
q_scale.bin
kv_scale.bin
splitData.bin
splitLse.bin
```

注意：

- 当前 aiter 侧 kernel launch 被注释掉，`splitData/splitLse/output` 是未执行 kernel 后的 buffer 内容，只适合检查 shape/布局/dump 机制，不适合作数值对比。
- poc_kl 侧 kernel 仍会真实执行，`splitData/splitLse` 是 kernel copy-back 后的结果。
- 对比输入时，优先看 `q/kv_buffer/*indptr/*indices/*scale`。
- 打开 aiter kernel launch 后，再对比两边 `splitData/splitLse`。

### 当前 dump 对比结果

运行两边 dump 后，用 SHA256 和 raw bytes 做逐文件比较，当前结果如下：

```text
qo_indptr.bin              MATCH
kv_last_page_lens.bin      MATCH
num_kv_splits_indptr.bin   MATCH

q.bin                      DIFF
kv_buffer.bin              DIFF
kv_indptr.bin              DIFF
kv_page_indices.bin        DIFF
q_scale.bin                DIFF
kv_scale.bin               DIFF
splitData.bin              DIFF
splitLse.bin               DIFF
```

`output.bin` 仅 aiter 侧存在；poc_kl 当前未 dump final output raw buffer。

关键 int/float buffer 解码结果：

```text
qo_indptr:
  aiter  = [0, 1, 2]
  poc_kl = [0, 1, 2]

kv_indptr:
  aiter  = [0, 10, 20]
  poc_kl = [0, 10, 20]

kv_indptr_tokens:
  aiter  = [0, 578, 1156]
  poc_kl = <not dumped>

kv_page_indices:
  aiter  = [8, 18, 13, 2, 10, 6, 16, 1, 9, 4, 11, 17, 12, 14,
            19, 5, 15, 7, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]
  poc_kl = [8, 18, 13, 2, 10, 6, 16, 1, 9, 4, 11, 17, 12, 14,
            19, 5, 15, 7, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]

kv_last_page_lens:
  aiter  = [2, 2]
  poc_kl = [2, 2]

num_kv_splits_indptr:
  aiter  = [0, 1, 2]
  poc_kl = [0, 1, 2]

q_scale:
  aiter  = [1.0, 1.0]
  poc_kl = [1.1122715473, 0.3453236818]

kv_scale:
  aiter  = [1.0, 1.0]
  poc_kl = [0.5996563435, 0.9239861369]
```

当前判断：

- `q/kv_buffer/q_scale/kv_scale` 不一致是预期的：aiter 使用 PyTorch 随机输入且 scale 固定为 1，poc_kl 使用自己的 `fmha_batch_init()` 随机/初始化逻辑。
- `kv_indptr` 已修复：aiter 现在传给 gfx1250 stage1 kernel 的是 page/block 级 indptr `[0,10,20]`，与 poc_kl 一致；原 token 级 indptr 仅额外 dump 为 `kv_indptr_tokens.bin` 便于追踪来源。
- `kv_page_indices` 已修复：aiter 测试层现在按 poc_kl 当前 `mla_shuffle()` 输出生成 compact shuffled page table，并将逻辑 KV pages scatter 到对应 physical page id。当前 raw dump 与 poc_kl 完全一致。
- `splitData/splitLse` 当前不能作为数值对比依据：aiter kernel launch 仍被注释掉，poc_kl kernel 真实执行；此外 poc_kl host 分配 split buffer 使用 `kMaxSplit=8`，raw dump 大小大于 aiter 的 `num_kv_splits=1` buffer。

下一步建议：

1. 如果要逐元素对比 `q/kv_buffer/q_scale/kv_scale`，需要让其中一边读取另一边的 raw input，而不是各自随机生成。
2. 打开 aiter kernel launch 后，再重新对比 `splitData/splitLse`。

最新 index buffer 对齐结果：

```text
qo_indptr              MATCH [0, 1, 2]
kv_indptr              MATCH [0, 10, 20]
kv_page_indices        MATCH [8,18,13,2,10,6,16,1,9,4,11,17,12,14,19,5,15,7,0,3,0,0,0,0,0,0,0,0]
kv_last_page_lens      MATCH [2, 2]
num_kv_splits_indptr   MATCH [0, 1, 2]
```

### 共享 poc_kl dump 输入后的 stage1 对齐结果

目标：

```text
先由 poc_kl 跑 kl_mla_mtp0_np_3p_k128_test 并 dump 输入/输出；
再让 aiter 直接读取 poc_kl dump 的 q/kv_buffer/q_scale/kv_scale；
aiter 继续跳过 stage2，直接验证 stage1 fp32 splitData。
```

aiter 新增环境变量：

```text
AITER_MLA_LOAD_POC_DUMP_DIR=/tmp/poc_kl_mla_debug_dump
```

启用后，`op_tests/test_mla_mi400.py` 会从该目录读取：

```text
q.bin
kv_buffer.bin
q_scale.bin
kv_scale.bin
```

并在 numerics 中使用 `poc_kl` 的 `splitData.bin` raw prefix 作为 expected，而不是使用 PyTorch reference。原因是当前 debug 目标是验证 aiter stage1 是否与 poc_kl stage1 kernel 输出一致；另外 `poc_kl` 的 split buffer 实际按 `kMaxSplit=8` 分配，raw 文件大小是 aiter `num_kv_splits=1` 输出的 8 倍，有效对齐区域是 raw prefix。

运行步骤：

1. 重新生成 poc_kl dump：

```bash
docker exec ff_sp3 bash -lc 'rm -rf /tmp/poc_kl_mla_debug_dump /tmp/aiter_mla_debug_dump && \
  mkdir -p /tmp/poc_kl_mla_debug_dump /tmp/aiter_mla_debug_dump && \
  cd /home/carhuang/feifei/poc_kl/mi400/mla && \
  export PATH=/opt/rocm/bin:$PATH && \
  bash run.sh compile >/tmp/poc_kl_compile_debug.log && \
  POC_KL_MLA_DEBUG_DUMP_DIR=/tmp/poc_kl_mla_debug_dump \
  timeout 180s bash run.sh test-one kl_mla_mtp0_np_3p_k128_test'
```

2. 让 aiter 读取 poc_kl dump 并跑 numerics：

```bash
docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
  AITER_MLA_LOAD_POC_DUMP_DIR=/tmp/poc_kl_mla_debug_dump \
  AITER_MLA_DEBUG_DUMP_DIR=/tmp/aiter_mla_debug_dump \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  timeout 120s python3 -m pytest op_tests/test_mla_mi400.py -k numerics -q -s'
```

本次验证结果：

```text
q                        MATCH
kv_buffer                MATCH
qo_indptr                MATCH
kv_indptr                MATCH
kv_page_indices          MATCH
kv_last_page_lens        MATCH
num_kv_splits_indptr     MATCH
q_scale                  MATCH
kv_scale                 MATCH

splitData raw prefix     EXACT MATCH
splitLse raw prefix      EXACT MATCH
pytest result            1 passed, 1 deselected
```

关键观察：

```text
aiter splitData.bin 与 poc_kl splitData.bin 的 raw prefix 完全一致。
aiter splitLse.bin 与 poc_kl splitLse.bin 的 raw prefix 完全一致。
poc_kl splitData.bin / splitLse.bin raw 文件仍包含 kMaxSplit=8 的额外空间，
其中有效对齐区域是开头的 aiter 输出大小。
```

因此，当前可认为：

```text
在相同输入、相同 index/page table、相同 scale、相同 kernarg/launch 参数下，
aiter mi400 MLA stage1 fp32 输出已经与 poc_kl stage1 输出对齐。
```

补充：本次为了定位卡点，`asm_mla.cu` 的 stage1 launch 后仍保留了临时 debug 同步和打印：

```text
hipGetLastError()
hipDeviceSynchronize()
```

日志显示：

```text
after launch enqueue: hipGetLastError=no error (0)
after hipDeviceSynchronize: status=no error (0)
```

如果后续要恢复正常性能测试，应移除这段临时同步探针。

### 临时 PyTorch stage2 验证完整输出

目标：

```text
stage1 已证明与 poc_kl 对齐后，临时绕过 Triton stage2，
用 PyTorch 实现同等 reduction，并在 test_mla_mi400.py 中比较完整 out。
```

实现位置：

```text
/home/carhuang/feifei/aiter/aiter/mla.py
```

临时行为：

```text
gfx1250 路径下不调用 _fwd_kernel_stage2_asm；
改为用 PyTorch 写入完整输出 o 和 final_lse。
```

`num_kv_splits == 1` 时，stage2 退化为：

```python
o.copy_(logits[:, 0].to(o.dtype))
final_lse.copy_(attn_lse[:, 0, :, 0])
```

多 split 时，使用标准 log-sum-exp 合并：

```python
cur_final_lse = torch.logsumexp(cur_lse, dim=1)
weights = torch.exp(cur_lse - cur_final_lse[:, None, :])
o.copy_((cur_logits * weights[..., None]).sum(dim=1).to(o.dtype))
```

测试侧改动：

```text
/home/carhuang/feifei/aiter/op_tests/test_mla_mi400.py
```

`test_mla_mi400_numerics` 不再比较 `stage1_out = attn_logits[:, 0]`，而是比较完整输出：

```text
case["out"]
```

在 `AITER_MLA_LOAD_POC_DUMP_DIR` 模式下，expected 仍来自 poc_kl `splitData.bin` 的 raw prefix。对于当前 `num_kv_splits=1` case，这等价于 poc_kl stage1 的完整输出；如果后续测试多 split，需要改为 poc_kl final output dump 或先确认 poc_kl stage2/final output buffer。

运行命令：

```bash
docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
  AITER_MLA_LOAD_POC_DUMP_DIR=/tmp/poc_kl_mla_debug_dump \
  AITER_MLA_DEBUG_DUMP_DIR=/tmp/aiter_mla_debug_dump \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  timeout 120s python3 -m pytest op_tests/test_mla_mi400.py -k numerics -q -s'
```

结果：

```text
[aiter][mi400][debug] _fwd_kernel_stage2_asm replaced by temporary PyTorch stage2
1 passed, 1 deselected in 2.57s
```

结论：

```text
在共享 poc_kl 输入和 expected 的条件下：
1. stage1 splitData/splitLse 与 poc_kl raw prefix exact match；
2. PyTorch stage2 写出的完整 out 通过 numerics；
3. 因此当前应重点排查原 Triton _fwd_kernel_stage2_asm，
   包括 shape/stride、FINAL_OUT 分支、lse reduction 和 final output 写入。
```

注意：

```text
PyTorch stage2 只是临时 debug 实现，不应用于性能测试或最终提交。
恢复正式路径时需要重新启用 _fwd_kernel_stage2_asm。
```

### 对齐 PyTorch reference 与 poc_kl CPU 验证

问题：

```text
读取 poc_kl dump 的 replay 模式下，aiter 与 poc_kl 输出 exact match；
但去掉 AITER_MLA_LOAD_POC_DUMP_DIR，使用 test_mla_mi400.py 原始随机输入时，
aiter 与 _ref_mla_mi400() 一开始不一致。
```

排查过程：

1. 对比 `test_mla_mi400.py` 的 `_ref_mla_mi400()` 和 `poc_kl/mi400/mla` 的 CPU 验证入口。
2. 确认当前目标 case `kl_mla_mtp0_np_3p_k128_test` 是 non-causal：`mask=0`，所以缺少 causal mask 不是主因。
3. 尝试 PyTorch reference 变体：
   - softmax probability 转 fp8 后再乘 V；
   - 调整 `q_scale/kv_scale` 应用位置；
   - 调整 KV page 读取顺序；
   - 复刻 poc_kl FP8 decode。
4. 以上变体均不能完全解释差异。
5. 最终定位到 `rope_split=2` 的输入 layout 差异。

poc_kl 的关键路径：

```text
prepare_v3_reference_data() 先用 logical Q/K 计算 CPU reference；
execute kernel 前，do_compute_v3() 会调用 process_rope_split()；
process_rope_split() 会把 Q/KV 拆成 nope 和 rope 两段，再 concat 成 kernel layout。
```

对应代码语义：

```text
copy_head_dim_partial_with_offset(Q_buf, QNOPE_buf, ..., nope_dim=512, offset=0)
copy_head_dim_partial_with_offset(Q_buf, QROPE_buf, ..., rope_dim=64, offset=512)
concatBuffersFast(QNOPE_buf, QROPE_buf, Q_buf,
                  batch,
                  q_seq_lens / gqa_ratio,
                  gqa_ratio,
                  nope_dim,
                  rope_dim)

copy_head_dim_partial_with_offset(K_buf_shuffle, KVNOPE_buf, ..., nope_dim=512, offset=0)
copy_head_dim_partial_with_offset(K_buf_shuffle, KVROPE_buf, ..., rope_dim=64, offset=512)
concatBuffersFast(KVNOPE_buf, KVROPE_buf, K_buf_shuffle,
                  batch,
                  kv_seq_lens_ajst / block_size,
                  block_size,
                  nope_dim,
                  rope_dim)
```

也就是说：

```text
Q kernel layout:
  对每个 batch、logical q token，把 16 个 GQA head 的 nope(512) 连续放前面，
  再把 16 个 GQA head 的 rope(64) 连续放后面。

KV kernel layout:
  对每个 physical page，把 64 个 token 的 nope(512) 连续放前面，
  再把 64 个 token 的 rope(64) 连续放后面。
```

`test_mla_mi400.py` 的修正：

```text
新增 _pack_rope_split2_pages()：
  用于 Q，按 q_seq_len/head group 维度 pack rope_split=2 layout。

新增 _pack_rope_split2_kv_pages()：
  用于 KV page，按 page_size=64 维度 pack rope_split=2 layout。

case["q"] / case["kv_buffer"]:
  传给 aiter kernel 的 packed tensor。

case["q_ref"] / case["kv_buffer_ref"]:
  传给 _ref_mla_mi400() 的 logical tensor，不做 packing。
```

修正后的验证结果：

```text
原始随机输入模式（不设置 AITER_MLA_LOAD_POC_DUMP_DIR）：
  docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
    env -u AITER_MLA_LOAD_POC_DUMP_DIR \
    ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
    AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
    timeout 240s python3 -m pytest op_tests/test_mla_mi400.py -q -s'

  结果：2 passed in 2.84s

原始随机输入模式最新复测（同时取消 debug dump 输出）：
  docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
    env -u AITER_MLA_LOAD_POC_DUMP_DIR -u AITER_MLA_DEBUG_DUMP_DIR \
    ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
    AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
    timeout 240s python3 -m pytest op_tests/test_mla_mi400.py -q -s'

  关键日志：
    _fwd_kernel_stage2_asm replaced by temporary PyTorch stage2

  结果：2 passed in 2.98s

poc_kl dump replay 模式：
  docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
    AITER_MLA_LOAD_POC_DUMP_DIR=/tmp/poc_kl_mla_debug_dump \
    ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
    AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
    timeout 120s python3 -m pytest op_tests/test_mla_mi400.py -q -s'

  结果：2 passed in 2.74s
```

结论：

```text
_ref_mla_mi400() 的数学公式可以与当前 mi400/poc_kl case 对齐；
关键是区分 logical reference layout 与 kernel rope_split=2 packed layout。
原始随机输入和 poc_kl dump replay 两种模式都已经通过。
```

### 最新完整测试命令和结果

目的：

```text
强制重新编译 aiter module_mla_asm，然后不使用 -k 过滤，
完整执行 op_tests/test_mla_mi400.py 中的 smoke 与 numerics 两个测试。
```

完整命令：

```bash
docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  AITER_MLA_LOAD_POC_DUMP_DIR=/tmp/poc_kl_mla_debug_dump \
  AITER_MLA_DEBUG_DUMP_DIR=/tmp/aiter_mla_debug_dump \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  timeout 180s python3 -m pytest op_tests/test_mla_mi400.py -q -s'
```

关键输出：

```text
[aiter] start build [module_mla_asm] under /home/carhuang/feifei/aiter/aiter/jit/build/module_mla_asm
[aiter] finish build [module_mla_asm], cost 7.6s
[aiter][mi400][debug] after launch enqueue: hipGetLastError=no error (0)
[aiter][mi400][debug] after hipDeviceSynchronize: status=no error (0)
[aiter][mi400][debug] _fwd_kernel_stage2_asm replaced by temporary PyTorch stage2
2 passed in 10.23s
```

本次运行环境含义：

```text
AITER_MLA_LOAD_POC_DUMP_DIR=/tmp/poc_kl_mla_debug_dump
  让 aiter test 读取 poc_kl dump 的 q/kv_buffer/q_scale/kv_scale，
  并使用 poc_kl splitData.bin raw prefix 作为 numerics expected。

AITER_MLA_DEBUG_DUMP_DIR=/tmp/aiter_mla_debug_dump
  让 aiter dump 当前输入、index、scale、stage1 splitData/splitLse、
  PyTorch stage2 后的 output/final_lse。

rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so
  强制 asm_mla.cu 重新 JIT 编译，避免复用旧 debug instrumentation。
```

本次完整测试覆盖：

```text
test_mla_mi400_minimal_smoke  PASS
test_mla_mi400_numerics       PASS
```

### 当前遗留的 debug 代码

当前代码仍包含多处临时 debug instrumentation。后续如果要恢复正式路径，需要逐项清理或改成更明确的环境变量开关。

1. `aiter/csrc/py_itfs_cu/asm_mla.cu`

遗留内容：

```text
mla_decode_mi400_dispatch() 中大量 [aiter][mi400][debug] printf：
- kernel/config 名称
- input 参数
- tensor shape/stride
- device pointers
- kernarg offsets/values
- launch grid/block
```

当前额外保留了 launch 后同步探针：

```cpp
hipError_t launch_status = hipGetLastError();
std::printf("[aiter][mi400][debug] after launch enqueue: hipGetLastError=%s (%d)\n", ...);
std::printf("[aiter][mi400][debug] before hipDeviceSynchronize after stage1 launch.\n");
hipError_t sync_status = hipDeviceSynchronize();
std::printf("[aiter][mi400][debug] after hipDeviceSynchronize: status=%s (%d)\n", ...);
```

作用：

```text
确认 stage1 kernel enqueue 成功，并且 hipDeviceSynchronize() 返回 no error。
```

风险/清理建议：

```text
hipDeviceSynchronize() 会强制全设备同步，严重影响性能，也会改变异步执行行为。
正式性能测试或提交前必须移除，或者至少改成环境变量控制。
大量 printf 也会污染测试日志，应在完成 debug 后移除或加开关。
```

2. `aiter/aiter/mla.py`

遗留内容 A：`AITER_MLA_DEBUG_DUMP_DIR` dump helper 和调用。

```text
_dump_mla_debug_tensor()
AITER_MLA_DEBUG_DUMP_DIR=/tmp/aiter_mla_debug_dump
```

当前会 dump：

```text
q, kv_buffer, qo_indptr, kv_indptr, kv_indptr_tokens,
kv_page_indices, kv_last_page_lens, num_kv_splits_indptr,
q_scale, kv_scale, splitData, splitLse, output, final_lse
```

作用：

```text
用于和 poc_kl raw dump 做 byte-level 对比。
```

风险/清理建议：

```text
该逻辑由环境变量控制，默认不启用，可以暂时保留；
但如果进入正式代码，应确认命名、路径、安全性和 CPU copy 开销符合项目规范。
```

遗留内容 B：gfx1250 路径临时 PyTorch stage2 替代原 Triton stage2。

```text
if _is_gfx1250:
    if num_kv_splits == 1:
        o.copy_(logits[:, 0].to(o.dtype))
        final_lse.copy_(attn_lse[:, 0, :, 0])
    else:
        torch.logsumexp + torch.exp 权重合并
    return logits, final_lse if return_lse else None
```

作用：

```text
绕过 _fwd_kernel_stage2_asm，验证在 stage1 已对齐时完整 out 是否可通过。
```

风险/清理建议：

```text
这是临时 debug fallback，不是最终实现。
它使用 PyTorch op，会引入额外 kernel、同步/调度行为和性能开销。
正式路径应恢复 _fwd_kernel_stage2_asm，并继续定位原 Triton stage2 的问题。
```

3. `aiter/op_tests/test_mla_mi400.py`

遗留内容 A：读取 poc_kl dump 的 helper 和环境变量。

```text
_load_raw_fp8()
_load_raw_float32()
_load_raw_float32_prefix()
AITER_MLA_LOAD_POC_DUMP_DIR=/tmp/poc_kl_mla_debug_dump
```

作用：

```text
让 aiter 使用 poc_kl dump 的 q/kv_buffer/q_scale/kv_scale，
并在 numerics 中用 poc_kl splitData.bin raw prefix 作为 expected。
```

风险/清理建议：

```text
这是 debug/replay 测试模式，不是普通 numerics 测试。
可以保留为显式 debug 模式，但不要让普通 CI 依赖 /tmp/poc_kl_mla_debug_dump。
如果要进入长期测试，应把 dump 文件变成受控 fixture，或拆成单独 debug test。
```

遗留内容 B：numerics 有意避开 GPU 端 `torch.isfinite(...).all()` reduction。

```text
torch.isfinite(tensor.detach().float().cpu()).all()
```

原因：

```text
当前 ff_sp3 环境中，最小复现显示 GPU 端 torch.isfinite(...).all()
在 bool 同步时会卡住；CPU copy 后检查可以正常完成。
```

风险/清理建议：

```text
这会增加 CPU copy 开销，但用于 debug numerics 可接受。
后续如果 GPU reduction 问题解决，可恢复普通 GPU 检查或保留 CPU 检查作为稳定路径。
```

4. `poc_kl/mi400/mla/mla_execute_v3_hip.inl`

遗留内容：

```text
POC_KL_MLA_DEBUG_DUMP_DIR=/tmp/poc_kl_mla_debug_dump
kernel args / shape / stride / pointer / launch debug printf
raw buffer dump helper
```

作用：

```text
生成 reference dump，供 aiter replay 和 raw output 对比。
```

风险/清理建议：

```text
dump 由环境变量控制，默认不启用，可以作为 debug 工具保留。
大量 printf 如无开关，可能影响普通 poc_kl 测试日志，应按需收敛。
```

## aiter 侧改动

文件：

```text
/home/carhuang/feifei/aiter/csrc/py_itfs_cu/asm_mla.cu
```

改动位置：`mla_decode_mi400_dispatch()` 中 mi400/gfx1250 分支。

当前临时行为：

1. 打印 kernel/config 名称。
2. 打印 Python/C++ 传入的 shape、stride、dtype 相关参数。
3. 打印 packed kernarg 中的指针和值。
4. 打印 launch grid/block。
5. 注释掉实际 kernel launch：

```cpp
// impl_ptr->launch_kernel({&args, &arg_size, gdx, gdy, gdz, bdx, bdy, bdz, stream});
```

输出前缀：

```text
[aiter][mi400][debug]
```

主要打印项：

```text
kernelName
knl_name / co_name
inputs: arch, batch, num_heads, nhead_kv, gqa_ratio, max_seqlen_q, page_size,
        qk_head_dim, q_elem_size, kv_split, softmax_scale
tensor shapes: Q, KV, splitData, splitLse, output
tensor strides: Q, KV, splitData, splitLse, output
ptrs: R, LSE, Q, KV, LTP, LTD, LTL, QTP, STP, QROPE, KVROPE, output, final_lse, stream
kernargs: arg_size, scalar, q_seq_lens, passes, stride_Q, stride_page, log2_page, out_16_nosplit
launch: grid, block
```

文件：

```text
/home/carhuang/feifei/aiter/aiter/mla.py
```

改动位置：`mla_decode_fwd()` 非 persistent 路径中，stage1 调用之后、`_fwd_kernel_stage2_asm` 之前。

当前临时行为：

```python
if _is_gfx1250:
    print(
        "[aiter][mi400][debug] _fwd_kernel_stage2_asm skipped by debug instrumentation",
        flush=True,
    )
    return logits, final_lse
```

也就是说：gfx1250 下 stage2 Triton reduction 暂时不会执行，方便只观察 stage1 launch 参数。

## 运行 aiter 打印参数

注意：`asm_mla.cu` 是 JIT 编译路径，改动后必须清 JIT 缓存。

```bash
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  ROCM_HOME=/opt/rocm \
  ENABLE_CK=0 \
  GPU_ARCHS=gfx1250 \
  AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q -s"
```

如果要跑 numerics 路径但仍只打印参数：

```bash
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  ROCM_HOME=/opt/rocm \
  ENABLE_CK=0 \
  GPU_ARCHS=gfx1250 \
  AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k numerics -q -s"
```

因为 kernel launch 和 stage2 都被跳过，上述命令不会验证数值，只用于打印参数。

## poc_kl 侧改动

文件：

```text
/home/carhuang/feifei/poc_kl/mi400/mla/mla_execute_v3_hip.inl
```

改动位置：`execute_v3_kernel()` 中 `MlaV3HipKernelArgs` 填完、`gdx/gdy/gdz` 算完之后，真正 `hipModuleLaunchKernel()` 之前。

当前行为：打印 poc_kl v3 host 实际传给 HIP kernel 的参数，但不跳过 kernel launch。

输出前缀：

```text
[poc_kl][mi400][debug]
```

主要打印项：

```text
kernelName
knl_name / co_name
inputs: batch, num_heads, nhead_kv, gqa_ratio, max_seqlen_q, page_size,
        qk_head_dim, q_elem_size, kv_split, softmax_scale
tensor shapes: Q, KV, splitData, splitLse, output
raw buffer shapes: Q_buf, K_buf_shuffle, gpu_split_Data, gpu_split_Lse
tensor strides: Q, KV, splitData, splitLse, output
ptrs: R, LSE, Q, KV, LTP, LTD, LTL, QTP, STP, QROPE, KVROPE, output, final_lse, stream
kernargs: arg_size, scalar, q_seq_lens, passes, stride_Q, stride_page, log2_page, out_16_nosplit
launch: grid, block
```

## 运行 poc_kl 打印参数

重新编译 host：

```bash
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/poc_kl/mi400/mla && \
  export PATH=/opt/rocm/bin:\$PATH && \
  bash run.sh compile"
```

运行当前目标测试：

```bash
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/poc_kl/mi400/mla && \
  export PATH=/opt/rocm/bin:\$PATH && \
  bash run.sh test-one kl_mla_mtp0_np_3p_k128_test"
```

也可以直接展开运行：

```bash
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/poc_kl/mi400/mla && \
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
    check_mode=1"
```

## 当前目标参数速查

`kl_mla_mtp0_np_3p_k128_test` 在 poc_kl 中覆盖参数：

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

poc_kl host 会在 `parse_runtime_args()` 中执行：

```cpp
args.q_seq_lens *= args.gqa_ratio;
```

所以 kernel 实际看到：

```text
q_seq_lens = 1 * 16 = 16
stride_Q = 1 * 16 * 576 * sizeof(fp8) = 9216
stride_Page = 64 * 576 * sizeof(fp8) = 36864
log2_page = 6
grid = (1, 2, 1)
block = (128, 1, 1)
```

## 当前实际运行结果

### aiter

运行命令：

```bash
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q -s"
```

结果：

```text
1 passed, 1 deselected
```

关键参数：

```text
kernelName=gfx1250mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC
knl_name=mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC
co_name=mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co
inputs: batch=2 num_heads=16 nhead_kv=1 gqa_ratio=16 max_seqlen_q=1 page_size=64 qk_head_dim=576 q_elem_size=1 kv_split=1 softmax_scale=0.0416667
tensor shapes: Q=(2,16,576) KV=(20,64,1,576) splitData=(2,1,16,512) splitLse=(2,1,16,1) output=(2,16,512)
tensor strides: Q=(9216,576,1) KV=(36864,576,576,1) splitData=(8192,8192,512,1) splitLse=(16,16,1,1) output=(8192,512,1)
kernargs: arg_size=288 scalar=0.0416667 q_seq_lens=16 passes=1 stride_Q=9216 stride_page=36864 log2_page=6 out_16_nosplit=0
launch: grid=(1,2,1) block=(128,1,1)
```

逐参数 offset/value：

```text
arg00 offset=0   name=ptr_R           value=<device ptr>
arg01 offset=16  name=ptr_LSE         value=<device ptr>
arg02 offset=32  name=ptr_Q           value=<device ptr>
arg03 offset=48  name=ptr_KV          value=<device ptr>
arg04 offset=64  name=ptr_LTP         value=<device ptr>
arg05 offset=80  name=ptr_LTD         value=<device ptr>
arg06 offset=96  name=ptr_LTL         value=<device ptr>
arg07 offset=112 name=scalar          value=0.0416667
arg08 offset=128 name=q_seq_lens      value=16
arg09 offset=144 name=passes          value=1
arg10 offset=160 name=stride_Q        value=9216
arg11 offset=176 name=stride_page     value=36864
arg12 offset=192 name=log2_page       value=6
arg13 offset=208 name=ptr_QTP         value=<device ptr>
arg14 offset=224 name=ptr_STP         value=<device ptr>
arg15 offset=240 name=out_16_nosplit  value=0
arg16 offset=256 name=ptr_QROPE       value=<device ptr>
arg17 offset=272 name=ptr_KVROPE      value=<device ptr>
```

### poc_kl

运行命令：

```bash
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/poc_kl/mi400/mla && \
  export PATH=/opt/rocm/bin:\$PATH && \
  bash run.sh compile >/tmp/poc_kl_compile_debug.log && \
  bash run.sh test-one kl_mla_mtp0_np_3p_k128_test"
```

结果：

```text
>>> PASS: kl_mla_mtp0_np_3p_k128_test
```

关键参数：

```text
kernelName=mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC
knl_name=mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC
co_name=mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co
inputs: batch=2 num_heads=16 nhead_kv=1 gqa_ratio=16 max_seqlen_q=1 page_size=64 qk_head_dim=576 q_elem_size=1 kv_split=1 softmax_scale=0.0416667
tensor shapes: Q=(2,16,576) KV=(10,64,1,576) splitData=(2,1,16,512) splitLse=(2,1,16,1) output=(2,16,512)
raw buffer shapes: Q_buf=(2,1,16,576) K_buf_shuffle=(2,1,640,576) gpu_split_Data=(2,1,16,512) gpu_split_Lse=(2,1,16,1)
tensor strides: Q=(9216,576,1) KV=(36864,576,576,1) splitData=(8192,8192,512,1) splitLse=(16,16,1,1) output=(8192,512,1)
kernargs: arg_size=288 scalar=0.0416667 q_seq_lens=16 passes=1 stride_Q=9216 stride_page=36864 log2_page=6 out_16_nosplit=0
launch: grid=(1,2,1) block=(128,1,1)
```

逐参数 offset/value：

```text
arg00 offset=0   name=ptr_r            value=<device ptr>
arg01 offset=16  name=ptr_lse          value=<device ptr>
arg02 offset=32  name=ptr_q            value=<device ptr>
arg03 offset=48  name=ptr_kv           value=<device ptr>
arg04 offset=64  name=ptr_ltp          value=<device ptr>
arg05 offset=80  name=ptr_ltd          value=<device ptr>
arg06 offset=96  name=ptr_ltl          value=<device ptr>
arg07 offset=112 name=scalar_f         value=0.0416667
arg08 offset=128 name=q_seq_lens_a     value=16
arg09 offset=144 name=passes_a         value=1
arg10 offset=160 name=stride_q_a       value=9216
arg11 offset=176 name=stride_page_a    value=36864
arg12 offset=192 name=log2_page_a      value=6
arg13 offset=208 name=ptr_qtp          value=<device ptr>
arg14 offset=224 name=ptr_stp          value=<device ptr>
arg15 offset=240 name=out_16_nosplit_a value=0
arg16 offset=256 name=ptr_descale_q    value=<device ptr>
arg17 offset=272 name=ptr_descale_k    value=<device ptr>
```

## 当前逐项对比结论

| 项目 | aiter | poc_kl | 结论 |
|---|---|---|---|
| kernel symbol | `mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC` | `mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_KERNEL_FUNC` | 一致。aiter 的 `kernelName` 配置 key 带 `gfx1250` 前缀，但实际 `knl_name` 一致。 |
| code object | `mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co` | `mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co` | 文件名一致；路径前缀来自各自 loader。 |
| batch / heads | `2 / 16 / 1 / gqa=16` | `2 / 16 / 1 / gqa=16` | 一致。 |
| q seq | `max_seqlen_q=1`, kernel `q_seq_lens=16` | `max_seqlen_q=1`, kernel `q_seq_lens=16` | 一致。 |
| page / dim | `page_size=64`, `qk_head_dim=576`, `v=512` | `page_size=64`, `qk_head_dim=576`, `v=512` | 一致。 |
| scale | `0.0416667` | `0.0416667` | 一致。 |
| Q shape/stride | `(2,16,576)`, `(9216,576,1)` | `(2,16,576)`, `(9216,576,1)` | 一致。 |
| KV shape/stride | `(20,64,1,576)`, `(36864,576,576,1)` | logical `(10,64,1,576)`, raw `(2,1,640,576)`, stride `(36864,576,576,1)` | stride 和实际 batch raw buffer 等价；shape 打印口径不同。后续若数值不一致，应进一步 dump/对比 `kv_page_indices` 内容。 |
| splitData | `(2,1,16,512)`, `(8192,8192,512,1)` | `(2,1,16,512)`, `(8192,8192,512,1)` | 一致。 |
| splitLse | `(2,1,16,1)`, `(16,16,1,1)` | `(2,1,16,1)`, `(16,16,1,1)` | 一致。 |
| output | `(2,16,512)`, `(8192,512,1)` | `(2,16,512)`, `(8192,512,1)` | 一致。 |
| arg size | `288` | `288` | 一致。 |
| scalar | `0.0416667` | `0.0416667` | 一致。 |
| passes | `1` | `1` | 一致。 |
| stride_Q | `9216` | `9216` | 一致。 |
| stride_page | `36864` | `36864` | 一致。 |
| log2_page | `6` | `6` | 一致。 |
| out_16_nosplit | `0` | `0` | 一致。 |
| arg offsets | `0,16,...,272` | `0,16,...,272` | 18 个参数 offset 完全一致。 |
| launch | `grid=(1,2,1) block=(128,1,1)` | `grid=(1,2,1) block=(128,1,1)` | 一致。 |

当前从 host 参数角度看，没有发现会导致 kernel 行为差异的 kernarg/launch mismatch。

唯一需要继续留意的是 KV buffer 的表达方式：

- aiter 直接以 page 维度暴露为 `KV=(20,64,1,576)`，即两个 batch 共 20 页。
- poc_kl debug 的 logical KV 行打印的是每 batch `max_num_blocks=10`，同时 raw buffer 行 `K_buf_shuffle=(2,1,640,576)` 表达了两个 batch。
- 两边 `stride_page=36864`、`grid=(1,2,1)` 一致，因此这不是直接错误；若后续打开 kernel launch 后结果不一致，下一步应优先 dump 并对比 `kv_indptr`、`kv_page_indices`、`kv_last_page_lens` 的实际内容。

## 对比建议

先分别保存两边日志：

```bash
# aiter
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q -s" \
  2>&1 | tee /tmp/aiter_mla_v3_args.log

# poc_kl
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/poc_kl/mi400/mla && \
  export PATH=/opt/rocm/bin:\$PATH && \
  bash run.sh test-one kl_mla_mtp0_np_3p_k128_test" \
  2>&1 | tee /tmp/poc_kl_mla_v3_args.log
```

重点对比：

```text
kernelName / co_name
max_seqlen_q / q_seq_lens
mask / causal 对应关系
stride_Q
stride_page
log2_page
passes
out_16_nosplit
grid / block
Q/KV/splitData/splitLse shape 与 stride
```

注意：指针地址每次运行都会不同，只需要确认非空和语义对应，不应逐字比较。

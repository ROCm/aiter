# mi400 MLA numerics debug handoff

日期：2026-05-13

## 背景

目标是让 aiter 中的 `op_tests/test_mla_mi400.py -k numerics` 跑通。

已知 golden 源端是当前 `ff_sp3` 容器中的：

```text
/home/carhuang/feifei/poc_kl/mi400/mla
```

对应 aiter 仓库是：

```text
/home/carhuang/feifei/aiter
```

目标 kernel：

```text
mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC
```

## 已确认通过的 poc_kl 测试

在 `ff_sp3` 中运行：

```bash
cd /home/carhuang/feifei/poc_kl/mi400/mla
export PATH=/opt/rocm/bin:$PATH
bash run.sh compile
bash run.sh test-one kl_mla_tp_np_test
```

结果：`>>> PASS: kl_mla_tp_np_test`。

关键 poc_kl 输出：

```text
stride_Q:36864 stride_Page:36864 log2_page:6
Start Kernel (HIP v3) local=(128,1,1) grid=(1,2,1)
#TEST PASSED
>>> PASS: kl_mla_tp_np_test
```

## 已修复的问题

### 1. aiter smoke test 导入问题

当前容器里 `import aiter` 会因为可选依赖 `flydsl._mlir` 缺失而提前跳过 bulk op exports，导致：

```text
module 'aiter' has no attribute 'mla'
module 'aiter' has no attribute 'mla_decode_stage1_asm_fwd'
```

已在 `op_tests/test_mla_mi400.py` 中显式导入：

```python
import aiter.mla as mla
from aiter.ops.attention import mla_decode_stage1_asm_fwd
aiter.mla_decode_stage1_asm_fwd = mla_decode_stage1_asm_fwd
```

### 2. smoke test shape 断言问题

`mla_decode_fwd()` 默认返回的 `attn_logits` 实际 shape 是：

```text
[q_seq_len, num_kv_splits, nhead, v_head_dim]
```

即当前 case 中是：

```text
[4, 1, 16, 512]
```

已修正 smoke 断言。

另外默认 `return_lse=False` 时 `attn_lse` 是 `None`，smoke 已改为断言 `attn_lse is None`。

### 3. mi400 kernarg 与 poc_kl v3 host 对齐

通过阅读：

```text
/home/carhuang/feifei/poc_kl/mi400/mla/mla_helper.h
/home/carhuang/feifei/poc_kl/mi400/mla/mla_execute_v3_hip.inl
```

确认 poc_kl 在 `parse_runtime_args()` 中会先执行：

```cpp
args.q_seq_lens *= args.gqa_ratio;
```

因此 `kl_mla_tp_np_test` 虽然 CLI 传 `q_seq_lens=4`，但 kernel 实际看到的是：

```text
q_seq_lens = 4 * 16 = 64
stride_Q = 1 * 64 * 576 * 1 = 36864
```

已在 `csrc/py_itfs_cu/asm_mla.cu` 的 `mla_decode_mi400_dispatch()` 中对齐：

```cpp
const int q_seq_lens_kernel = max_seqlen_q * gqa_ratio;
args.q_seq_lens = static_cast<unsigned int>(q_seq_lens_kernel);
args.stride_Q = static_cast<unsigned int>(
    nhead_kv * q_seq_lens_kernel * qk_head_dim * q_elem_size);
```

同时移除了 gfx1250 分支中对 `lse != nullptr` 的 hard assert。`return_lse=True` 时 stage1 忽略 final LSE 指针，stage2 负责写 final LSE。

### 4. aiter `.co` 与 poc_kl `.co` 不一致

曾发现 aiter 当前加载的 `.co` 和 poc_kl 刚通过测试的 `.co` SHA 不一致：

```text
aiter old: 8b73df6d41561f329191a42fff702fe33b5375b8a28284bce4c85addcbaf331b
poc_kl:    d5aa8f2f9ec8dfffd79b257a6178f70e04aac00c599711df60e588f487622150
```

已将 poc_kl 通过测试的 `.co` 复制到 aiter：

```bash
cp /home/carhuang/feifei/poc_kl/mi400/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co \
   /home/carhuang/feifei/aiter/hsa/gfx1250/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co
```

当前复核两边 SHA 已一致：

```text
d5aa8f2f9ec8dfffd79b257a6178f70e04aac00c599711df60e588f487622150
```

### 5. 文档同步

已更新：

```text
ffdoc/mla_mi400_test_guide.md
ffdoc/debug_mla_mi400_against_poc_kl.md
ffdoc/poc_kl_compile_test_single_mla_kernel.md
```

重点修正：

```text
q_seq_lens = max_seqlen_q * gqa_ratio
stride_Q = nhead_kv * max_seqlen_q * gqa_ratio * qk_head_dim * sizeof(fp8)
```

## 已验证结果

清理 JIT 后 smoke 曾经通过：

```bash
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q"
```

结果：

```text
1 passed, 1 deselected
```

## 当前阻塞

`numerics` 运行时挂住。进一步检查发现不是单纯 pytest 断言问题，而是当前 `ff_sp3` / GPU runtime 已处于异常状态。

症状：

```bash
timeout 15s docker exec ff_sp3 bash -c "ROCM_HOME=/opt/rocm ENABLE_CK=0 python3 -u - <<'PY'
import torch
print('before available', flush=True)
print(torch.cuda.is_available(), flush=True)
print('after available', flush=True)
PY"
```

只输出：

```text
before available
```

然后超时。

另一个最小 CUDA 同步脚本也会卡住：

```text
import torch ok
available True
init...
alloc...
sync...
```

然后在 `torch.cuda.synchronize()` 挂住。

容器中出现过 `D` 状态 Python 进程，即使 `kill -9` 也无法清掉：

```text
python3 -u -
```

这通常表示进程卡在不可中断的内核/驱动调用中，需要恢复 GPU runtime。

## 建议下一步

先恢复 GPU 环境。可选方式取决于机器使用情况：

1. 如果允许，重启 `ff_sp3` 容器并确认没有残留 KFD 进程。
2. 如果容器重启无效，需要管理员重置 amdgpu 驱动或重启机器。
3. 在多人共享机器上，不要直接卸载/重载 amdgpu，除非确认不会影响其他人的 GPU 任务。

GPU 恢复后先跑最小 sanity：

```bash
docker exec ff_sp3 bash -c "ROCM_HOME=/opt/rocm ENABLE_CK=0 python3 -u - <<'PY'
import torch
print('available', torch.cuda.is_available(), flush=True)
torch.cuda.init()
x = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
print('cuda sanity OK', x.item(), flush=True)
PY"
```

如果 sanity 通过，再跑 numerics：

```bash
docker exec ff_sp3 bash -c "cd /home/carhuang/feifei/aiter && \
  rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \
  ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k numerics -q -s"
```

如果再次挂住，建议先绕过 `mla_decode_fwd()` 的 Triton stage2，直接调用 `aiter.mla_decode_stage1_asm_fwd()` 并在 stage1 launch 后立刻 `torch.cuda.synchronize()`，用来区分：

```text
stage1 kernel / kernarg / input layout 问题
vs
Triton stage2 reduction 问题
```

## 需要重点关注的文件

```text
/home/carhuang/feifei/aiter/op_tests/test_mla_mi400.py
/home/carhuang/feifei/aiter/aiter/mla.py
/home/carhuang/feifei/aiter/csrc/py_itfs_cu/asm_mla.cu
/home/carhuang/feifei/aiter/hsa/gfx1250/mla/mla_asm.csv
/home/carhuang/feifei/aiter/hsa/gfx1250/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co
/home/carhuang/feifei/poc_kl/mi400/mla/mla_execute_v3_hip.inl
/home/carhuang/feifei/poc_kl/mi400/mla/mla_helper.h
```

## 当前判断

代码侧已完成三项高概率必要修复：

1. aiter 测试导入路径修复。
2. mi400 v3 kernarg 与 poc_kl host 对齐。
3. aiter 使用的 `.co` 与 poc_kl 通过测试的 `.co` 对齐。

剩余无法确认的是：GPU 恢复后 `test_mla_mi400_numerics` 是否数值通过。如果不通过，下一步应优先比较 stage1 的 `splitData` / `splitLse` 是否 finite，以及 stage2 reduction 与 PyTorch reference 的 mask/layout 语义是否一致。

# Aiter gfx1250 MLA 完整路径与接口

日期：2026-05-13

## 目的

本文记录当前 aiter 中 gfx1250 MLA 的完整调用路径、接口、dispatcher、HSA 元数据和 loader 关系。

## 总体调用路径

```text
aiter.mla.mla_decode_fwd(...)
  -> aiter.mla_decode_stage1_asm_fwd(...)
     -> csrc/py_itfs_cu/asm_mla.cu::mla_decode_stage1_asm_fwd(...)
        -> get_gpu_arch() == "gfx1250" 时进入 mla_decode_mi400_dispatch(...)
           -> 查 cfg_mla_asm
           -> AiterAsmKernel(knl_name, co_name)
           -> launch_kernel(...)
        -> 其他架构继续走原 gfx942/gfx950 路径
  -> 必要时执行 aiter/mla.py 中的 Triton stage2 reduction
```

## Python 接口

用户侧主要接口：

```python
aiter.mla.mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    page_size=1,
    nhead_kv=1,
    sm_scale=None,
    logit_cap=0.0,
    num_kv_splits=None,
    num_kv_splits_indptr=None,
    work_meta_data=None,
    work_indptr=None,
    work_info_set=None,
    reduce_indptr=None,
    reduce_final_map=None,
    reduce_partial_map=None,
    q_scale=None,
    kv_scale=None,
    intra_batch_mode=False,
    return_logits=False,
    return_lse=False,
)
```

gfx1250 特殊逻辑：

- `get_gfx() == "gfx1250"` 时禁用 gfx9 的 `o.view(...)` alias 优化。
- gfx1250 会分配独立 fp32 `logits`，因为 mi400 shader 写 fp32 partial output。
- 非 persistent decode 调用 `aiter.mla_decode_stage1_asm_fwd(...)` 后再执行 Triton stage2 reduction。
- 当前 gfx1250 mi400 dispatcher 不支持 persistent mode。

## C 接口

文件：

```text
aiter/csrc/py_itfs_cu/asm_mla.cu
```

入口：

```cpp
AITER_C_ITFS
void mla_decode_stage1_asm_fwd(
    aiter_tensor_t* Q,
    aiter_tensor_t* KV,
    aiter_tensor_t* qo_indptr,
    aiter_tensor_t* kv_indptr,
    aiter_tensor_t* kv_page_indices,
    aiter_tensor_t* kv_last_page_lens,
    aiter_tensor_t* num_kv_splits_indptr,
    aiter_tensor_t* work_meta_data,
    aiter_tensor_t* work_indptr,
    aiter_tensor_t* work_info_set,
    int max_seqlen_q,
    int page_size,
    int nhead_kv,
    float softmax_scale,
    aiter_tensor_t* splitData,
    aiter_tensor_t* splitLse,
    aiter_tensor_t* output,
    aiter_tensor_t* lse,
    aiter_tensor_t* q_scale,
    aiter_tensor_t* kv_scale,
    hipStream_t stream)
```

gfx1250 与 gfx9 的分流点：

- 函数内读取 `arch_id = get_gpu_arch()`。
- `arch_id == "gfx1250"` 时立即 `return mla_decode_mi400_dispatch(...)`。
- 原 `gfx942/gfx950` 逻辑位于该 early return 之后。

## gfx1250 Dispatcher

函数：

```cpp
static void mla_decode_mi400_dispatch(...)
```

运行时 guard：

- 只允许 `arch_id == "gfx1250"`。
- 只允许非 persistent decode：
  - `work_meta_data == nullptr`
  - `work_indptr == nullptr`
  - `work_info_set == nullptr`
  - `num_kv_splits_indptr != nullptr`
- 不支持 `lse` 输出。
- 要求 `Q` 和 `KV` dtype 为 fp8。
- 要求 `nhead_kv == 1`。
- 要求 `page_size == 64`。
- 要求 `kv_split == 1`。
- 要求 Q head dim 为 576。
- 要求 output head dim 为 512。
- 要求 `q_scale`、`kv_scale` 是 fp32 tensor。
- 要求 `splitData`、`splitLse` 是 fp32。

变体选择：

- 默认 `AITER_MLA_MI400_VARIANT_ID=1`。
- 可通过环境变量覆盖：

```bash
export AITER_MLA_MI400_VARIANT_ID=<id>
```

当前可运行路径对应的变体：

- `1`：v3，qSeqLen 4，GQA 16，passSize 64，`np`
- `2`：v3，qSeqLen 4，GQA 16，passSize 64
- `3`：v3，qSeqLen 4，GQA 16，passSize 64，`np_3p`
- `4`：v3，qSeqLen 4，GQA 16，passSize 64，`np_1tdm`
- `5`：v3，qSeqLen 4，GQA 16，passSize 128
- `6`：v3，qSeqLen 1，GQA 16，passSize 128
- `7`：v3，qSeqLen 2，GQA 16，passSize 128
- `8`：v3，qSeqLen 4，GQA 32，passSize 64，`out_16_nosplit=1`

当前被阻断的变体：

- `9..12`：v4/sparse，资产存在，但 runtime 会主动报错，因为还没有 QROPE/KVROPE ABI。

## Packed Args

当前 gfx1250 runtime 使用：

```cpp
struct __attribute__((packed)) MlaMi400KernelArgs
```

特点：

- 总大小 288B。
- 对应 18 个 16B slot。
- 当前填参逻辑按 v3 ABI 实现。

当前 v3 语义：

- slot 10 填 `stride_Q`。
- slot 16 填 `q_scale->data_ptr()`，作为 v3 `descale_q`。
- slot 17 填 `kv_scale->data_ptr()`，作为 v3 `descale_k`。

为什么不能直接跑 v4：

- v4 slot 10 是 `total_kv`，不是 `stride_Q`。
- v4 slot 16 是 `ptr_qrope`，不是 `descale_q`。
- v4 slot 17 是 `ptr_kvrope`，不是 `descale_k`。

## HSA 元数据路径

CSV：

```text
aiter/hsa/gfx1250/mla/mla_asm.csv
```

CSV 字段：

```text
qType,kvType,Gqa,ps,qSeqLen,prefill,causal,lse,passSize,modelVersion,variantId,knl_name,co_name
```

codegen 命令：

```bash
cd /home/feifei/repo/aiter
PYTHONPATH=/tmp/aiter_codegen_deps AITER_GPU_ARCHS='gfx1250;gfx950;gfx942' \
  python3 hsa/codegen.py -m mla --output_dir /tmp/aiter_mla_codegen_all
```

生成头：

```text
/tmp/aiter_mla_codegen_all/asm_mla_configs.hpp
```

正常 JIT 构建时，`asm_mla_configs.hpp` 会生成到模块构建目录，并由 `asm_mla.cu` include。

## Loader 路径

dispatcher 从 `cfg_mla_asm` 获取：

- `knl_name`
- `co_name`

然后创建或复用：

```cpp
AiterAsmKernel(name, co_name)
```

源码树运行时通常需要设置：

```bash
export AITER_ASM_DIR=/home/feifei/repo/aiter/hsa
```

## 构建与运行命令

构建模块：

```bash
cd /home/feifei/repo/aiter
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  python3 -c "from aiter.jit.core import build_module; build_module('module_mla_asm')"
```

运行 smoke：

```bash
cd /home/feifei/repo/aiter
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q
```

运行指定 v3 变体：

```bash
cd /home/feifei/repo/aiter
AITER_MLA_MI400_VARIANT_ID=5 \
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q
```

## 尚未形成完整接口的部分

当前没有 aiter Python/C 接口能把独立的 v4 QROPE/KVROPE tensor 传入 `mla_decode_stage1_asm_fwd`。

补齐 v4 需要：

- 新增匹配 `MlaV4HipKernelArgs` 的 packed struct。
- 新增 v4 dispatcher，例如 `mla_decode_mi400_v4_dispatch`。
- 修改 Python/C API，显式传入 QROPE/KVROPE，或定义能推导出二者指针的 packed tensor layout。
- 增加 v4 输入构造和 smoke/correctness 测试。

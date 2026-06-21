# DeepSeek Sparse Attention (DSA) Training Kernels

Triton kernels for DeepSeek-V3 style sparse MLA (Multi-head Latent Attention) training on AMD MI300X GPUs.

## What is DSA?

In DeepSeek-V3's MLA architecture, each query token attends to only a **TopK subset** of KV tokens (e.g., 1024 out of 128K), rather than the full sequence:
- MQA: 128 query heads share 1 KV head
- KV is compressed: `kv_lora_rank=512` + `rope_rank=64` = `d_qk=576`

```
Q:    [total_tokens, num_heads=128, d_qk=576]    bf16
KV:   [total_tokens, 1,             d_qk=576]    bf16
TopK: [total_tokens, topk=1024]                  int32  (absolute KV token indices)
```

## Quick Start

```python
from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_fwd, sparse_mla_bwd, sparse_mla_train,
)

# Forward
o, lse = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank=512)

# Backward (explicit)
dq, dkv = sparse_mla_bwd(q, kv, o, do, topk_indices, lse,
                          kv_lora_rank=512, method="chunked_gather")

# Autograd-integrated
o, lse = sparse_mla_train(q, kv, topk_indices, kv_lora_rank=512,
                           bwd_method="chunked_gather")
loss = o.sum()
loss.backward()  # populates q.grad and kv.grad
```

## Backward Strategies

All numbers measured on MI300X (gfx942), B=1, S=4096, H=128, D=576, TOPK=1024.

| Method | Time (ms) | TFLOPS | Speedup | Extra Memory | Bottleneck |
|--------|----------:|-------:|--------:|-------------:|------------|
| `"fused"` | 61.1 | 39.4 | 1.00x | 0 | atomicAdd + S/P recompute |
| `"recompute"` | 52.2 | 46.1 | 1.17x | 0 | atomicAdd (Infinity Cache) |
| `"split_intermediate"` | 37.9 | 63.5 | 1.61x | ~2 GiB | atomicAdd (Infinity Cache) |
| `"privatized"` | 37.9 | 63.4 | 1.61x | ~2.1 GiB | atomicAdd — no improvement over split_intermediate |
| `"xcd_privatized"` | 37.9 | 63.4 | 1.61x | ~2.1 GiB | atomicAdd — no improvement (dKV > 4 MB L2, misses to Infinity Cache) |
| `"gather"` | 18.0 | 133.7 | **3.40x** | ~6.5 GiB | HBM bandwidth |
| `"chunked_gather"` | 23.1 | 103.9 | **2.64x** | ~1.65 GiB | HBM bandwidth |
| `"persistent"` | — | — | — | ~25 MB | blocked: Triton/LLVM hangs at D_V=512 |

### Recommended

| Scenario | Method |
|---|---|
| Best speed, memory not a constraint | `"gather"` |
| Best speed/memory tradeoff | `"chunked_gather"` |
| Zero extra memory | `"recompute"` |
| Small head count (H≤32) | `"fused"` |

## Why atomicAdd Is the Bottleneck

The dKV scatter is a scatter-reduce: each of T×TOPK=4M query-rank pairs writes a D=576-element gradient to a shared KV token via `atomic_add_f32`. On MI300X:

- The dKV buffer (T×D×4 = 9.44 MB fp32) exceeds the 4 MB L2 per XCD → atomics always miss L2 and serialize at the **shared Infinity Cache** (~0.12 TFLOPS empirical, vs 1,307 TFLOPS bf16 matrix compute)
- XCD privatization (`"privatized"`, `"xcd_privatized"`) does not help: the private copy still exceeds L2, so atomics still route to the Infinity Cache atomic unit
- rocprof `TCC_EA0` counters show **4.5× write amplification** vs non-atomic stores (~603M dirty writebacks vs ~67M)

`"gather"` and `"chunked_gather"` eliminate atomics entirely by writing each contribution to a uniquely-owned slot in an intermediate buffer (one writer per slot → plain store, no serialization). The gather phase accumulates per KV token with plain loads.

## The `persistent` Method (Blocked)

The persistent kernel was designed to fix the L2-miss problem by chunking the KV token range so each XCD's private dkv_chunk (3.14 MB) fits in its 4 MB L2, making atomics L2-local. The kernel logic is correct (passes correctness tests at small D_V), but Triton/LLVM hangs during compilation at production config (D_V=512) due to register pressure (~164 VGPRs/thread with BLOCK_H=64). See `dsa_dev/docs/persistent_kernel_postmortem.md` for the full analysis and recommended paths forward (reduce BLOCK_H, split dQ/dKV passes, or HIP implementation).

## File Structure

```
aiter/ops/triton/_triton_kernels/attention/
  deepseek_sparse_attention.py       # Forward kernel + backward dispatch + autograd wrapper
  _dsa_bwd_preprocess.py             # Delta precompute kernel (shared by all methods)
  _dsa_bwd_fused.py                  # "fused": single fused bwd kernel
  _dsa_bwd_recompute.py              # "recompute": split dQ+dKV, recompute S/P/dS
  _dsa_bwd_split_intermediate.py     # "split_intermediate": split dQ+dKV, store dS/P
  _dsa_bwd_privatized.py             # "privatized", "xcd_privatized": private dKV copies
  _dsa_bwd_gather.py                 # "gather", "chunked_gather": no-atomic methods
  _dsa_bwd_persistent.py             # "persistent": 304-CTA L2-local kernel (blocked)

op_tests/triton_tests/attention/
  README_DSA.md                      # This file
  bench_dsa_methods.py               # Correctness + benchmark for all 7 working methods
  test_sparse_mla_fwd_train.py       # Forward correctness test
  test_sparse_mla_bwd_train.py       # Backward correctness test (vs PyTorch reference)
```

## Running Tests and Benchmarks

```bash
# Correctness (all 7 methods)
python op_tests/triton_tests/attention/bench_dsa_methods.py --test-only

# Benchmark (all 7 methods, 3 configs)
python op_tests/triton_tests/attention/bench_dsa_methods.py --bench-only

# Forward unit test
python op_tests/triton_tests/attention/test_sparse_mla_fwd_train.py

# Backward unit test
python op_tests/triton_tests/attention/test_sparse_mla_bwd_train.py
```

Expected benchmark output (MI300X, B=1, S=4096, H=128, TOPK=1024):

```
  Method                  Time (ms)   TFLOPS  Speedup   Extra mem
  ----------------------------------------------------------------
  fused                       61.11     39.4     1.00x          0
  recompute                   52.16     46.1     1.17x          0
  split_intermediate          37.89     63.5     1.61x    2.0 GiB
  privatized                  37.94     63.4     1.61x    2.1 GiB
  xcd_privatized              37.93     63.4     1.61x    2.1 GiB
  gather                      17.99    133.7     3.40x    6.5 GiB
  chunked_gather              23.14    103.9     2.64x   1.63 GiB
```

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V3.2 Training](https://arxiv.org/abs/2512.02556)
- [AITER](https://github.com/ROCm/aiter)

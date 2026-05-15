# Sparse MLA Backward Benchmark on H100

Benchmark the Triton sparse MLA backward kernel on H100 to compare against MI300X results.

## MI300X reference numbers (Triton backward kernel)

| Config | ms | TFLOPS | Autotune config |
|--------|---:|-------:|-----------------|
| B1 S4096 H128 topk1024 | 60.8 | 39.6 | BH=64 TK=16 w=4 s=2 |
| B1 S4096 H128 topk2048 | 118.7 | 40.5 | BH=64 TK=16 w=4 s=2 |
| B1 S8192 H128 topk1024 | 121.5 | 39.6 | BH=64 TK=16 w=4 s=2 |
| B1 S8192 H128 topk2048 | 237.1 | 40.6 | BH=64 TK=16 w=4 s=2 |
| B1 S4096 H32 topk1024 | 14.4 | 22.7 | BH=32 TK=16 w=8 s=2 |
| B1 S4096 H16 topk1024 | 27.5 | 10.9 | BH=16 TK=16 w=8 s=1 |

## 1. Clone and install aiter

```bash
git clone https://github.com/ROCm/aiter.git
cd aiter
git checkout 7890e4be7  # same commit as MI300X benchmarks

# Install (CUDA/Triton only — no ROCm needed for the Triton kernels)
pip install -e .
# If the full install fails due to ROCm dependencies, the Triton kernels
# can still be imported directly — see step 2b below.
```

## 2a. Run the built-in benchmark (if aiter installs)

```bash
python op_tests/triton_tests/attention/test_sparse_mla_bwd_train.py --bench-only
```

This runs:
- Forward pass to get O and LSE
- Backward kernel with autotune across all configs
- Reports latency (ms), TFLOPS, and selected autotune config

## 2b. Standalone benchmark (if aiter doesn't install)

If `pip install -e .` fails on H100/CUDA, copy just the kernel files and run a standalone benchmark. Create this directory structure anywhere:

```
sparse_mla_bench/
  sparse_mla_fwd_train.py   # copy from aiter/ops/triton/_triton_kernels/attention/
  sparse_mla_bwd_train.py   # copy from aiter/ops/triton/_triton_kernels/attention/
  __init__.py                # empty file
  bench_bwd.py               # see below
```

Copy the kernel files:
```bash
mkdir -p sparse_mla_bench
cp aiter/ops/triton/_triton_kernels/attention/sparse_mla_fwd_train.py sparse_mla_bench/
cp aiter/ops/triton/_triton_kernels/attention/sparse_mla_bwd_train.py sparse_mla_bench/
touch sparse_mla_bench/__init__.py
```

Fix the import in `sparse_mla_bwd_train.py` — change line 22:
```python
# FROM:
from .sparse_mla_fwd_train import _get_lds_limit
# TO:
from sparse_mla_fwd_train import _get_lds_limit
```

Then create `sparse_mla_bench/bench_bwd.py`:

```python
#!/usr/bin/env python3
"""Standalone sparse MLA backward benchmark for H100."""
import torch
import triton
from sparse_mla_fwd_train import sparse_mla_fwd_train
from sparse_mla_bwd_train import sparse_mla_bwd_train, _sparse_mla_bwd_kernel

CONFIGS = [
    # (batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk)
    (1, 4096, 128, 512, 64, 1024),
    (1, 4096, 128, 512, 64, 2048),
    (1, 8192, 128, 512, 64, 1024),
    (1, 8192, 128, 512, 64, 2048),
    (1, 4096,  32, 256, 64, 1024),
    (1, 4096,  16, 512, 64, 1024),
]

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Triton: {triton.__version__}")
print(f"PyTorch: {torch.__version__}")
print()
print(f"{'Config':<35s} {'ms':>8s} {'TFLOPS':>8s} {'Autotune':>35s}")
print("-" * 90)

for batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk in CONFIGS:
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device="cuda")
    topk_indices = torch.randint(0, total_tokens, (total_tokens, topk),
                                 dtype=torch.int32, device="cuda")

    # Forward
    o, lse = sparse_mla_fwd_train(q, kv, topk_indices, kv_lora_rank, scale)
    do = torch.randn_like(o)

    # Warmup + autotune
    for _ in range(5):
        sparse_mla_bwd_train(q, kv, o, do, topk_indices, lse, kv_lora_rank, scale)
    torch.cuda.synchronize()

    # Benchmark
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    reps = 100
    for _ in range(reps):
        sparse_mla_bwd_train(q, kv, o, do, topk_indices, lse, kv_lora_rank, scale)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / reps

    flops_bwd = total_tokens * num_heads * topk * (
        2 * d_qk + 2 * kv_lora_rank + 2 * d_qk + 2 * d_qk
    )
    tflops = flops_bwd / (ms * 1e-3) / 1e12

    best = _sparse_mla_bwd_kernel.best_config
    cfg = f"BH={best.kwargs['BLOCK_H']} TK={best.kwargs['TILE_K']} w={best.num_warps} s={best.num_stages}"

    label = f"B{batch}_S{seq_len}_H{num_heads}_topk{topk}"
    print(f"{label:<35s} {ms:8.3f} {tflops:8.1f} {cfg:>35s}")
```

Run:
```bash
cd sparse_mla_bench
python bench_bwd.py
```

## 3. What to report

For each config, please report:
- Latency (ms)
- TFLOPS
- Selected autotune config (BLOCK_H, TILE_K, num_warps, num_stages)
- H100 GPU model (SXM5 80GB vs PCIe)
- Triton version

## 4. Notes

- The kernel is pure Triton (no HIP/ROCm-specific code) — it should compile and run on NVIDIA GPUs without modification.
- The `_get_lds_limit()` function returns 65536 bytes (default) on non-AMD GPUs. H100 has 228KB shared memory per SM, so all configs should fit.
- Autotune will explore all configs including those pruned on MI300X. The AGPR WAR hazard is AMD-specific — H100 should be fine with all configs.
- The backward kernel uses `tl.atomic_add` for dKV accumulation. Performance may vary depending on how Triton/CUDA handles atomics vs MI300X/HIP.

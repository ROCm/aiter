# TileLang DSA Forward Benchmark on H100

## 1. Install TileLang (pip, CUDA)

```bash
pip install tilelang
# Verify
python -c "import tilelang; print(tilelang.__version__)"
```

If pip install has issues, build from source:

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
pip install -e .
```

## 2. Run the built-in test (correctness + perf)

TileLang ships with a DSA sparse MLA forward example:

```bash
cd $(python -c "import tilelang; import os; print(os.path.dirname(tilelang.__file__))")/../examples/dsa_sparse_finetune
# Or if installed from source:
# cd tilelang/examples/dsa_sparse_finetune

python sparse_mla_fwd.py
```

Default config: B=1, S=4096, H=128, topk=1024, DQK=576, DV=512, block_I=64, num_stages=2.

## 3. Run extended benchmarks

Create and run this script in the same `dsa_sparse_finetune` directory (it imports `index.py` and `utils.py` from there):

```python
#!/usr/bin/env python3
"""Benchmark TileLang sparse MLA fwd on H100 across DeepSeek V3 configs."""
import torch
from sparse_mla_fwd import sparse_mla_fwd_interface
from tilelang.profiler import do_bench

CONFIGS = [
    # (S, H, topk, block_I, num_stages, threads)
    (4096,  128, 1024, 64, 2, 256),
    (4096,  128, 2048, 64, 2, 256),
    (8192,  128, 1024, 64, 2, 256),
    (8192,  128, 2048, 64, 2, 256),
    (4096,   32, 1024, 64, 2, 256),
    (4096,   16, 1024, 32, 2, 128),
]

DQK, DV, HKV = 576, 512, 1

print(f"{'Config':<35s} {'ms':>8s} {'TFLOPS':>8s}")
print("-" * 55)

for S, H, topk, block_I, num_stages, threads in CONFIGS:
    torch.manual_seed(0)
    q  = torch.randn(S, H, DQK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(S, HKV, DQK, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor([0, S], dtype=torch.int32, device="cuda")

    indices = torch.full((S, HKV, topk), S, dtype=torch.int32, device="cuda")
    for t in range(S):
        n = min(topk, max(1, t))
        idx = torch.randperm(max(1, t), device="cuda")[:n]
        indices[t, 0, :n] = idx

    # Warmup (first call triggers JIT compilation)
    sparse_mla_fwd_interface(q, kv, indices, offsets,
                             block_I=block_I, num_stages=num_stages, threads=threads)

    def fn():
        return sparse_mla_fwd_interface(q, kv, indices, offsets,
                                        block_I=block_I, num_stages=num_stages, threads=threads)

    ms = do_bench(fn, rep=100, warmup=50)
    flops = S * (DQK + DV) * topk * 2 * H
    tflops = flops / (ms * 1e-3) / 1e12

    label = f"S{S}_H{H}_topk{topk}"
    print(f"{label:<35s} {ms:8.3f} {tflops:8.1f}")
```

## 4. Expected H100 tuning parameters

| H | block_I | num_stages | threads | Notes |
|---|---------|------------|---------|-------|
| 128 | 64 | 2 | 256 | Default, best for DeepSeek V3 |
| 32 | 64 | 2 | 256 | Should work (padded_H=32) |
| 16 | 32 | 2 | 128 | padded_H=16, smaller tiles |

H100 has 228KB shared memory per SM, so all these configs should fit.

## 5. What to report

Please report for each config:
- Latency (ms)
- TFLOPS
- H100 GPU model (SXM5 80GB vs PCIe)

These numbers will be compared against our Triton kernel and AITER on MI300X.

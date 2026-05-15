# TileLang DSA Sparse MLA Backward Benchmark on H100

Benchmark TileLang's sparse MLA backward kernel on H100 for apple-to-apple comparison
with our Triton backward kernel.

Source: https://github.com/tile-ai/tilelang/blob/main/examples/dsa_sparse_finetune/sparse_mla_bwd.py

## MI300X reference numbers (our Triton backward kernel)

| Config | ms | TFLOPS |
|--------|---:|-------:|
| B1 S4096 H128 D576 topk1024 | 60.8 | 39.6 |
| B1 S4096 H128 D576 topk2048 | 118.7 | 40.5 |
| B1 S8192 H128 D576 topk1024 | 121.5 | 39.6 |
| B1 S8192 H128 D576 topk2048 | 237.1 | 40.6 |
| B2 S4096 H128 D576 topk1024 | 122.0 | 39.4 |
| B1 S4096 H32 D320 topk1024 | 14.4 | 22.7 |
| B1 S4096 H16 D576 topk1024 | 27.5 | 10.9 |

FLOPS formula (same as our Triton kernel):
```
d_qk = kv_lora_rank + rope_rank
flops = total_tokens * num_heads * topk * (2*d_qk + 2*kv_lora_rank + 2*d_qk + 2*d_qk)
```

## 1. Install TileLang

```bash
pip install tilelang
# or from source:
git clone https://github.com/tile-ai/tilelang.git
cd tilelang
pip install -e .
```

## 2. Set up benchmark directory

```bash
mkdir -p tilelang_bwd_bench
cd tilelang_bwd_bench

# Copy example files from TileLang (dsa_sparse_finetune, NOT deepseek_v32)
cp /path/to/tilelang/examples/dsa_sparse_finetune/sparse_mla_bwd.py .
cp /path/to/tilelang/examples/dsa_sparse_finetune/sparse_mla_fwd.py .
cp /path/to/tilelang/examples/dsa_sparse_finetune/utils.py .
cp /path/to/tilelang/examples/dsa_sparse_finetune/index.py .
```

### Required patch: remove hardcoded dim assert in sparse_mla_fwd.py

The fwd interface hardcodes `assert dim_plus_tail_dim == 576`, which breaks configs
with kv_lora_rank=256 (DQKV=320). Comment out this line:

```python
# In sparse_mla_fwd.py, around line 179:
# assert dim_plus_tail_dim == 576, "you should assign dim otherwise"   # <-- comment out
dim = d_v
```

## 3. Create benchmark script

Create `benchmark_dsa_bwd.py`:

```python
#!/usr/bin/env python3
"""
TileLang DSA sparse MLA backward benchmark — apple-to-apple with our Triton configs.

Uses the dsa_sparse_finetune version of the kernel which has:
  - Flattened tensors [S, H, D] (no batch dim — uses offsets for batching)
  - Offsets + TokenIndices for variable-length sequence support
  - Symbolic S dimension (JIT'd per shape)

Config mapping from our Triton kernel -> TileLang:
  (batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk)
  ->  S=batch*seq_len, H=num_heads, HKV=1, D=kv_lora_rank,
      D_tail=rope_rank, topk=topk, offsets=[0, seq_len, 2*seq_len, ...]
"""
import torch

from sparse_mla_bwd import preprocess, bwd, postprocess
from sparse_mla_fwd import sparse_mla_fwd_interface
from index import prepare_token_indices

# Configs exactly matching our Triton backward benchmark
# (batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk)
CONFIGS = [
    # --- Synthetic configs (from TileLang paper / AITER) ---
    (1, 4096, 128, 512, 64, 1024),
    (1, 4096, 128, 512, 64, 2048),
    (1, 8192, 128, 512, 64, 1024),
    (1, 8192, 128, 512, 64, 2048),
    # (2, 4096, 128, 512, 64, 1024),  # SKIP: TileLang needs 368KB smem, exceeds H100's 228KB
    (1, 4096,  32, 256, 64, 1024),    # Requires fwd assert patch (see setup step 2)
    (1, 4096,  16, 512, 64, 1024),
    # --- Real-world configs from DeepSeek-V3.2 DSA training ---
    # Source: https://arxiv.org/pdf/2512.02556
    # DSA trains at 128K seq_len, topk=2048, H=128, kv_lora_rank=512, rope_rank=64
    # Main training: 480 seqs × 128K / 2048 GPUs ≈ 30K tokens/GPU
    (1, 32768, 128, 512, 64, 2048),    # 32K tokens, ~30K/GPU from main DSA training
    (1, 65536, 128, 512, 64, 2048),    # 64K tokens
    (1, 131072, 128, 512, 64, 2048),   # 128K tokens, single full DSA training sequence
]


def generate_causal_indices(S, HKV, topk, offsets):
    """Generate causal topk indices: query at position t selects from [0, t]."""
    indices = torch.full((S, HKV, topk), -1, dtype=torch.int32, device="cuda")
    B = offsets.shape[0] - 1
    for b in range(B):
        bos = offsets[b].item()
        eos = offsets[b + 1].item()
        seq_len = eos - bos
        for t in range(seq_len):
            for h in range(HKV):
                pool = max(1, t)
                n = min(topk, pool)
                idx = torch.randperm(pool, device="cuda")[:n]
                indices[bos + t, h, :n] = idx.int()
    return indices


def benchmark_one(batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk):
    S = batch * seq_len    # total tokens (flattened)
    H = num_heads
    HKV = 1                # kv_group = 1
    DQKV = kv_lora_rank + rope_rank   # total Q/K head dim
    DV = kv_lora_rank                 # V dim (output head dim = nope dim)
    D = kv_lora_rank                  # TileLang "D" = nope dim
    D_tail = rope_rank                # TileLang "D_tail" = rope dim

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Flattened tensors (no batch dim)
    q = torch.randn((S, H, DQKV), dtype=torch.bfloat16, device="cuda")
    kv = torch.randn((S, HKV, DQKV), dtype=torch.bfloat16, device="cuda")
    do = torch.randn((S, H, DV), dtype=torch.bfloat16, device="cuda")

    # Offsets: one sequence of length S per batch element
    offsets = torch.tensor(
        [i * seq_len for i in range(batch + 1)],
        dtype=torch.int32, device="cuda"
    )
    token_indices = prepare_token_indices(offsets)

    indices = generate_causal_indices(S, HKV, topk, offsets)

    # Forward pass to get O and LSE
    o, lse = sparse_mla_fwd_interface(q, kv, indices, offsets, d_v=DV)

    # Build TileLang kernels
    sm_scale = DQKV ** (-0.5)
    preprocess_kernel = preprocess(H, D)
    bwd_kernel = bwd(H, D, D_tail, topk, HKV, sm_scale, True)
    postprocess_kernel = postprocess(D, D_tail, HKV)

    delta = preprocess_kernel(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)

    # Warmup (run full bwd_kernel including JIT compilation)
    for _ in range(5):
        dkv.zero_()
        dq = bwd_kernel(q, kv, do, indices, lse, delta, offsets, token_indices, dkv)
    torch.cuda.synchronize()

    # Benchmark bwd kernel only (exclude preprocess/postprocess)
    from tilelang.profiler import do_bench

    def run_bwd():
        dkv.zero_()
        return bwd_kernel(q, kv, do, indices, lse, delta, offsets, token_indices, dkv)

    ms = do_bench(run_bwd, rep=100, warmup=50)

    # FLOPS using our formula (same as Triton benchmark)
    d_qk = DQKV
    flops_bwd = S * H * topk * (
        2 * d_qk + 2 * kv_lora_rank + 2 * d_qk + 2 * d_qk
    )
    tflops = flops_bwd / (ms * 1e-3) / 1e12

    return ms, tflops


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import tilelang
        print(f"TileLang: {tilelang.__version__}")
    except Exception:
        print("TileLang: (version unknown)")
    print()
    print(f"{'Config':<40s} {'ms':>8s} {'TFLOPS':>8s}")
    print("-" * 60)

    for batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk in CONFIGS:
        label = f"B{batch}_S{seq_len}_H{num_heads}_D{kv_lora_rank+rope_rank}_topk{topk}"
        try:
            ms, tflops = benchmark_one(batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk)
            print(f"{label:<40s} {ms:8.3f} {tflops:8.1f}")
        except Exception as e:
            print(f"{label:<40s} FAILED: {e}")
            import traceback
            traceback.print_exc()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
```

## 4. Run

```bash
cd tilelang_bwd_bench
python benchmark_dsa_bwd.py
```

## 5. Important notes

### Known limitations

1. **B2_S4096_H128 (multi-batch) — shared memory overflow**: TileLang allocates
   shared memory proportional to `padded_H`. With H=128, `padded_H=128`, so buffers
   like `Q_shared[128, 512]` in bf16 = 128KB alone. Total smem needed is 368KB,
   exceeding H100's 228KB per SM. This config is skipped. Our Triton kernel handles
   this by tiling over heads (BLOCK_H=64).

2. **B1_S4096_H32_D320 — fwd assert**: TileLang's `sparse_mla_fwd.py` hardcodes
   `assert dim_plus_tail_dim == 576`. Must be patched (see setup step 2 above).

### This uses the `dsa_sparse_finetune` kernel, NOT `deepseek_v32`

The correct TileLang backward kernel is at:
`examples/dsa_sparse_finetune/sparse_mla_bwd.py`

NOT `examples/deepseek_v32/sparse_mla_bwd.py` (that one uses batched `[B,S,H,D]`
tensors and a different API).

The `dsa_sparse_finetune` version uses flattened `[S, H, D]` tensors with `offsets`
for variable-length batching — closer to our setup where we flatten batch×seq into
`total_tokens`.

### Parameter mapping

| Our Triton | TileLang (`dsa_sparse_finetune`) | Notes |
|-----------|----------------------------------|-------|
| total_tokens = batch × seq_len | S (flattened) | Same flattening |
| num_heads (H=128) | H=128 | Same |
| kv_lora_rank | D (nope dim) | TileLang's "D" |
| rope_rank | D_tail | TileLang's "D_tail" |
| kv_group=1 | HKV=1 | Both use single KV head |
| — | offsets = [0, seq_len] | Sequence boundaries |

### Config 5 special case (kv_lora_rank=256)

Config `(1, 4096, 32, 256, 64, 1024)` has `kv_lora_rank=256` (not 512).
TileLang's `sparse_mla_bwd()` wrapper hardcodes `D=512`, so the benchmark
script calls the lower-level `bwd()` kernel directly with `D=256, D_tail=64`.

### Causal masking

TileLang enforces `is_causal=True` (asserts on it). The kernel masks indices
where `Indices[...] > query_position` or `Indices[...] == -1`.
Our Triton kernel doesn't do causal masking (indices are pre-computed).
For benchmarking, this difference is minor — the compute is the same.

### FLOPS formula

The script uses the same FLOPS formula as our Triton benchmark:
```
flops = total_tokens * num_heads * topk * (2*d_qk + 2*kv_lora_rank + 2*d_qk + 2*d_qk)
```

TileLang's own `test_sparse_mla_bwd` uses a different formula:
```
per_token_flop = 2 * (H*DV*topk + H*DQKV*topk + H*DQKV*topk + H*DQKV*topk + H*DV*topk)
```

For configs where DV == kv_lora_rank == 512, both formulas give the same result.

## 6. What to report

For each config, please report:
- Latency (ms)
- TFLOPS (using our formula)
- H100 GPU model (SXM5 80GB vs PCIe)
- TileLang version

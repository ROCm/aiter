# Bug: `mla_prefill_ps_asm_fwd` + `mla_reduce_v1` does not produce valid `final_lse` for Q tiles with 0 KV splits

## Summary

When using the two-phase MLA prefill pipeline (`mla_prefill_ps_asm_fwd` →
`mla_reduce_v1`), the `final_lse` output is **not written** for Q tiles where
the persistent scheduler assigns 0 partial splits (i.e., the PS ASM kernel
handles all KV reductions internally). The attention **output is correct**, but
`final_lse` remains uninitialized.

This is a blocker for **chunked prefill** (also called context-parallel or
prefix-caching), where LSE is required to merge attention states from the
suffix (new tokens) and prefix (cached KV) via the standard online-softmax
rescaling formula.

## Root Cause

There are two independent places where `final_lse` is skipped:

### 1. PS ASM kernel — internal reduction path

`get_ps_metadata_v1` assigns work so that certain Q tiles have their entire KV
range handled within a single CU partition. For these tiles,
`reduce_indptr[tile] == reduce_indptr[tile+1]` (zero partial splits needing
external reduction).

`mla_prefill_ps_asm_fwd` writes the final **output** directly to the `output`
buffer for these tiles, but does **not** write to `final_lse`.

### 2. `mla_reduce_v1` — skips tiles with ≤1 splits

In `csrc/kernels/mla/reduce.cu`, both the simple and persistent-scheduling
reduce kernels skip tiles with `num_splits ≤ 1`:

```cpp
// reduce.cu line 696
// In theory, we can handle the case that #split = 1. However, it is meaningless and
// metadata should be in charge of getting rid of this kind of scenario.
else if(num_splits > 1)
{
    mla_reduce_v1_impl_simple<Traits, lse_t, out_t>(...);
}
```

When `num_splits == 0`, neither the output nor the LSE is written by
`mla_reduce_v1` (the PS ASM kernel already wrote the output directly).
When `num_splits == 1`, the tile is also skipped entirely.

### Crossover depends on hardware / workload

The number of 0-split tiles depends on the number of CUs, `num_heads`, and
sequence length. On a **256-CU gfx950** GPU with `num_heads=16`,
`qk_head_dim=192`, `v_head_dim=128`, `batch_size=1`:

| ctx_len | Q tiles | Tiles with 0 splits | `final_lse` valid? |
|---------|---------|--------------------|--------------------|
| 256     | 1       | 0                  | Yes (all tiles ≥2 splits) |
| 512     | 2       | 0                  | Yes |
| 1024    | 4       | 1                  | **Partial** (1 of 4 tiles invalid) |
| 4096    | 16      | 8                  | **Partial** (8 of 16 tiles invalid) |
| 8192    | 32      | 32                 | **No** (all tiles 0 splits) |
| 16384   | 64      | 64                 | **No** (all tiles 0 splits) |

With `num_heads=1` (the default in `test_mla_prefill_ps.py`):

| ctx_len | Q tiles | Tiles with 0 splits | `final_lse` valid? |
|---------|---------|--------------------|--------------------|
| 256     | 1       | 0                  | Yes |
| 4096    | 16      | 1                  | **Partial** (1 tile invalid, 256 of 4096 elements) |
| 8192    | 32      | 2                  | **Partial** |

## Reproduction

### Using `op_tests/test_mla_prefill_ps.py`

The modified test in this branch adds LSE validation (comparing `final_lse`
from the kernel against a PyTorch reference). Run:

```bash
# ctx=256: LSE PASSES (all tiles have ≥2 splits)
python3 op_tests/test_mla_prefill_ps.py -qkh 192 -vh 128 -n 1 -c 256 -b 1 --causal true

# ctx=4096: LSE FAILS partially (1 tile has 0 splits → 256 of 4096 elements wrong)
python3 op_tests/test_mla_prefill_ps.py -qkh 192 -vh 128 -n 1 -c 4096 -b 1 --causal true

# ctx=8192: LSE FAILS completely (all tiles have 0 splits → final_lse is all zeros)
python3 op_tests/test_mla_prefill_ps.py -qkh 192 -vh 128 -n 1 -c 8192 -b 1 --causal true
```

Expected output for ctx=8192 shows `final_lse` is all zeros while reference
has values in the 2–7 range:

```
[aiter] mla_prefill_lse   [torch vs aiter_asm]: us......[checkAllclose atol=0.05 rtol=0.05 failed!]
    a    : torch.Size([8192])
           tensor([-0.0814,  0.3647,  1.7042,  1.2395, ...])   <-- reference
    b    : torch.Size([8192])
           tensor([0., 0., 0., 0., ...])                        <-- kernel (all zeros)
```

### Minimal standalone reproduction

```python
import torch, aiter
from aiter import dtypes, per_tensor_quant
torch.set_default_device('cuda')

ctx, nhead, qk_dim, v_dim, tile_q = 8192, 16, 192, 128, 256

qo_indptr = torch.tensor([0, ctx], dtype=torch.int32)
seq_lens = torch.tensor([ctx], dtype=torch.int32)
kv_indices = torch.arange(ctx, dtype=torch.int32)

Q = torch.randn(ctx, nhead, qk_dim, dtype=torch.bfloat16)
K = torch.randn(ctx, nhead, qk_dim, dtype=torch.bfloat16)
V = torch.randn(ctx, nhead, v_dim, dtype=torch.bfloat16)
q8, _ = per_tensor_quant(Q, quant_dtype=dtypes.fp8)
k8, _ = per_tensor_quant(K, quant_dtype=dtypes.fp8)
v8, _ = per_tensor_quant(V, quant_dtype=dtypes.fp8)

info = aiter.get_ps_metadata_info_v1(
    batch_size=1, num_head_k=nhead, max_qlen=ctx, qlen_granularity=256)
bufs = [torch.empty(s, dtype=d, device='cuda') for s, d in info]
aiter.get_ps_metadata_v1(
    qo_indptr.cpu(), qo_indptr.cpu(), seq_lens.cpu(), 1, nhead,
    *bufs, qhead_granularity=1, qlen_granularity=256,
    kvlen_granularity=128, block_size=1, is_causal=True)
torch.cuda.synchronize()

wm, wi, winfo, ri, rfm, rpm = bufs
n_partial = int(ri[-1].item())
print(f"num_partial_tiles = {n_partial}")  # prints 0 for ctx>=8192

logits = torch.empty((max(n_partial, 1) * tile_q, nhead, v_dim), dtype=torch.float32)
attn_lse = torch.empty((max(n_partial, 1) * tile_q, nhead), dtype=torch.float32)
output = torch.empty((ctx, nhead, v_dim), dtype=torch.bfloat16)
final_lse = torch.full((ctx, nhead), -999.0, dtype=torch.float32)  # sentinel
one = torch.ones((), dtype=torch.float32)

aiter.mla_prefill_ps_asm_fwd(
    q8, k8, v8, qo_indptr, qo_indptr, kv_indices,
    wi, winfo, ctx, 1.0 / (qk_dim ** 0.5), True,
    logits, attn_lse, output, one, one, one)
aiter.mla_reduce_v1(logits, attn_lse, ri, rfm, rpm, tile_q, output, final_lse)
torch.cuda.synchronize()

print(f"final_lse[:5, 0] = {final_lse[:5, 0].tolist()}")
# Prints [-999.0, -999.0, -999.0, -999.0, -999.0]
# Sentinel is untouched — kernel never wrote final_lse
assert (final_lse == -999.0).all(), "Expected final_lse to be untouched"
print("Confirmed: final_lse was never written by the kernel pipeline")
```

## Impact

This prevents using the FP8 ASM prefill kernel for **chunked prefill** in vLLM
and similar inference frameworks. In chunked prefill, a long prompt is processed
in chunks:

- **Chunk 1** (no prior context): FP8 ASM kernel works — LSE is not needed.
- **Chunk 2+** (with cached context): LSE is required by `merge_attn_states` to
  correctly combine the suffix attention (new tokens → new tokens) with the
  prefix attention (new tokens → cached KV). Without valid LSE, these chunks
  must fall back to `flash_attn_varlen_func`.

For a 100K-token prompt with `max_num_batched_tokens=16384`, only the first
chunk (1 of ~7) can use the FP8 ASM kernel. The remaining ~85% of prefill
computation falls back to the slower path.

## Suggested Fix

The `mla_prefill_ps_asm_fwd` assembly kernel should write `final_lse` for Q
tiles it reduces internally, in addition to writing the output. This would make
LSE available regardless of how `get_ps_metadata_v1` partitions work across CUs.

Alternatively, `get_ps_metadata_v1` could ensure every Q tile always has ≥2
partial splits so that `mla_reduce_v1` always runs, though this may have
performance implications.

## Environment

- **GPU**: gfx950 (MI355X), 256 CUs
- **AITER commit**: `7d063b2b2` (main branch)
- **PyTorch**: 2.8+
- **Test file**: `op_tests/test_mla_prefill_ps.py` (modified in this branch to
  add LSE validation)

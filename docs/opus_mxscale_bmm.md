# OPUS MXScale BMM: scale groups and tuned CSV compatibility

On gfx950, the raw-weight `batched_gemm_a8w8_mxscale` entry accepts
`w_scale_block="32x32"` as well as its existing default, `"128x128"`:

```python
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale

y = batched_gemm_a8w8_mxscale(
    x, weight, x_scale, w_scale, dtype=torch.bfloat16, w_scale_block="32x32"
)
```

The block is named N-by-K, so `"32x32"` means `n_block = k_block = 32`.

Inputs are FP8 `x[M,G,K]` and `weight[G,N,K]`. E8M0 scale tensors have shapes
`x_scale[M,G,ceil(K/k_block)]` and
`w_scale[G,ceil(N/n_block),ceil(K/k_block)]`. The output is `[M,G,N]`.
The caller must pass the block used to quantize its tensors; changing this
argument does not convert scales between formats. The weight-preshuffled
entry retains its 128x128 contract.

The block is included in tuning lookup, launch-plan caching, and heuristic
selection. Both the direct and split-K workspace paths use the selected kid.
The lower-level `opus_bmm(..., kid=..., layout="mxscale_bmm")` API still uses
the explicit kid to determine the scale blocks; registered 32x32 twins use
9xxx kids and 128x128 uses 8xxx kids.

## Existing tuned CSVs

The raw-weight MXScale table key is now `(gfx,b,m,n,k,w_scale_block)`. Tables
without `w_scale_block`, including files selected through
`AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE`, load with a warning and
are interpreted as 128x128. Existing callers and legacy local kernel IDs remain
supported. No manual CSV migration is required for 128x128 users.

When adding 32x32 rows, add a `w_scale_block` column and use `32x32` or
`128x128` on each row. Rows for the same shape and different blocks coexist.
OPUS rows whose kid does not support the declared block are skipped with an
invalid-row warning; the resolver then uses the heuristic for the requested
block. Generic config merging also includes the block in its duplicate key and
normalizes legacy source files in memory before merging.

## Performance evidence and limits

The model CSV contains tuning results, not a controlled base-versus-head
benchmark. Historical 128x128 timings must not be used as a same-condition
baseline for 32x32 or for the full PR.

The 128x128 path is unchanged by this work, established two ways. The compiled
ISA for the 128x128 kernels is identical across the change -- same instruction
mix, same VGPR and LDS footprint, no spills either side -- so there is no code
for a regression to come from. An interleaved A/B on an otherwise idle card
then agreed: alternating the tree order round by round and taking paired
within-round medians, small M came out at +0.49%, +0.39% and +1.19%, with a
+0.07% median across the shape set.

A 5% regression reported at small M before that did not survive this method.
The card drifts about 12% over a few minutes under co-tenancy, and a harness
that measures one tree to completion and then the other attributes that drift
to whichever tree held the slower window. Measurements here must interleave.

32x32 costs 3-8% against 128x128 at M >= 8192, which is consistent with its
four-times-larger scale traffic: at B=8, M=32768 the tuned rows are 2013 vs
2176 TFLOPS, and only at B=16 does 32x32 come out marginally ahead. These are
tuning-table timings on one card, not an end-to-end serving guarantee.

Reproduction and the current correctness results are recorded in
`opus_mxscale_bmm_validation.md`.

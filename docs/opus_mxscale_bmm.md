# OPUS MXScale BMM: scale groups and tuned CSV compatibility

On gfx950, the raw-weight `batched_gemm_a8w8_mxscale` entry accepts
`group_size=32` as well as its existing default, `group_size=128`:

```python
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale

y = batched_gemm_a8w8_mxscale(
    x, weight, x_scale, w_scale, dtype=torch.bfloat16, group_size=32
)
```

Inputs are FP8 `x[M,G,K]` and `weight[G,N,K]`. E8M0 scale tensors have shapes
`x_scale[M,G,ceil(K/group_size)]` and
`w_scale[G,ceil(N/group_size),ceil(K/group_size)]`. The output is `[M,G,N]`.
The caller must pass the group used to quantize its tensors; changing this
argument does not convert scales between formats. The weight-preshuffled
entry retains its GS128 contract.

The group is included in tuning lookup, launch-plan caching, and heuristic
selection. Both the direct and split-K workspace paths use the selected kid.
The lower-level `opus_bmm(..., kid=..., layout="mxscale_bmm")` API still uses
the explicit kid to determine the scale blocks; registered GS32 twins use
9xxx kids and GS128 uses 8xxx kids.

## Existing tuned CSVs

The raw-weight MXScale table key is now `(gfx,b,m,n,k,groupSize)`. Tables
without `groupSize`, including files selected through
`AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE`, load with a warning and
are interpreted as GS128. Existing callers and legacy local kernel IDs remain
supported. No manual CSV migration is required for GS128 users.

When adding GS32 rows, add a `groupSize` column and use 32 or 128 on each row.
Rows for the same shape and different groups coexist. OPUS rows whose kid
does not support the declared group are skipped with an invalid-row warning;
the resolver then uses the heuristic for the requested group. Generic config
merging also includes the group in its duplicate key and normalizes legacy
source files in memory before merging.

## Performance evidence and limits

The model CSV contains tuning results, not a controlled base-versus-head
benchmark. Historical GS128 timings must not be used as a same-condition
baseline for GS32 or for the full PR.

**The claim that all GS128 paths have no performance regression against the
exact PR base is unverified.** A full exact-base/head timing run and downstream
serving validation remain required before making that claim. The scoped
batch-pair experiments described below measure the traversal change only;
their baseline binary is not established as the PR base.

The earlier same-card, same-input-pool experiment used six rounds, alternating
module order, B=16, N=1024, K=4096, BF16. With GS128, M=16384 changed from
1211.2573 to 1018.5182 us and M=32768 from 2413.6035 to 1998.5998 us. These are
limited local observations, not an end-to-end serving throughput guarantee.
The proposed explanation involving memory-channel or cache traffic is a
hypothesis; hardware counters have not established the mechanism.

Reproduction and the current correctness results are recorded in
`opus_mxscale_bmm_validation.md`.

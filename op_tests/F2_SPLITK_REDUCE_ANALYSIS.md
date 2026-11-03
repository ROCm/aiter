# F2 gfx1250 PTPC split-K reduction analysis

## Decision

Do not enable an aiter-only fused reduction in the production path from this
branch. Both implemented one-kernel candidates were numerically correct, but
the shared-GPU timing did not establish a benefit and showed material
regressions at several real shapes. Shipping either candidate by default would
trade a proven two-kernel selection for an unproven, often slower one.

Fusing into QK norm/RoPE or SwiGLU remains the plausible next design, but that
requires changing the ATOM call sites together with the aiter consumer API.
Adding an unused aiter fused op here would be dead code, so it is intentionally
not included.

## Does tuning include the reduction?

Yes.

The gfx1250 branch of
`GemmA8W8BpreShuffleTuner._get_flydsl_tune_task_gfx1250` benchmarks
`run_gemm_flydsl_gfx1250`. That runner calls
`run_preshuffle_gemm_a8_gfx1250`, which launches the PTPC GEMM and then
`compile_gemm_a8w8_splitk_reduce` whenever `split_k > 1`. `mp_tuner` measures
the complete callable, so the saved model-specific `us` winner already
includes GEMM plus reduction. The production benchmark path also calls
`gemm_a8w8_bpreshuffle`, including both launches.

Consequently, the existing split-K rows are not artifacts of timing only the
main GEMM.

## Real shapes

The validation script covers MiniMax-M3 TP4 decode M values
`4, 8, 16, 32, 48, 64` for:

- sparse QKV: `(M, N, K) = (M, 2560, 6144)`;
- dense QKV: `(M, 2304, 6144)`;
- dense gate/up: `(M, 6144, 6144)`.

These widths follow the model configuration and the ATOM projection packing:
64 Q heads, 4 KV heads, hidden size 6144, and dense intermediate size 12288.

## Is split-K still required?

Generally yes. On the shared GPU, HIP-graph replay of the configured
split-K GEMM plus reduction was compared with the same tile/pipeline at
`split_k=1`.

- Sparse QKV: split-1 was slower in all six cases; median slowdown `18.5%`.
- Dense QKV: split-1 was slower in all six cases; median slowdown `23.4%`.
- Dense gate/up: median split-1 slowdown `18.0%`; M=32 was `1.4%` faster and
  M=64 was within `0.6%`, which is not sufficient evidence on a shared card to
  rewrite those tuned rows.

The full native tuner had also selected split-K while measuring the reduction.
There is therefore no justified global “retune to split-1” shortcut.

## One-kernel experiments

Two production-call-chain implementations were built and exercised before
being removed:

1. A last-arriving workgroup published FP32 partials, used an agent-scope
   arrival counter, reduced all splits in split order, and reset the counter.
   It was bitwise equal to the existing GEMM plus reducer and produced exactly
   one profiled kernel, but became substantially slower as M increased.
2. A gfx1250 hardware-cluster epilogue made all K-split workgroups cooperate:
   each workgroup kept its local row stripe, published peer rows by TDM, and
   reduced one stripe after a cluster barrier. It was also bitwise equal and
   one kernel. The best shared-card version was mixed:
   - sparse QKV deltas over M=4..64: `-0.3%, 0.0%, +7.3%, +2.6%, +18.7%, +2.5%`;
   - dense QKV: `-4.2%, -3.5%, +1.9%, -0.9%, +3.5%, +4.3%`;
   - dense gate/up: `+6.2%, 0.0%, -0.3%, +1.2%, +47.1%, +4.4%`.

A BF16-partial variant was also rejected: it was slower in the sampled case
and `8.24%` of values differed from the existing path under the tuner
`rtol=atol=1e-2` criterion.

These timings are diagnostic only because the GPU was shared. They are still
strong enough not to make a regressing implementation the default. An
exclusive-card run is pending; no performance acceptance claim is made.

## Reproducible validation

`op_tests/test_gemm_a8w8_ptpc_splitk_f2.py` checks all 18 real shapes against
torch and the explicit unchanged path, repeats each production output five
times, profiles kernel names/counts, and optionally compares graph replay with
the same tile at split-1.

Observed baseline:

- production versus explicit configured path: bitwise equal for all 18 shapes;
- five repeated runs: bitwise deterministic for all 18 shapes;
- maximum absolute error versus torch: `2.0`;
- maximum mismatch ratio at `rtol=atol=1e-2`: `0.922309%`;
- maximum relative L2 error: `0.108015%`;
- configured split-K path: 2 kernels (PTPC GEMM plus
  `gemm_a8w8_splitk_reduce`);
- same-tile split-1 control: 1 kernel.

The test was run in
`rocm/fw-bringup:gfx1250-minimax-m3-20261102` with
`HIP_VISIBLE_DEVICES=0`; `aiter.__path__` resolved to this worktree.

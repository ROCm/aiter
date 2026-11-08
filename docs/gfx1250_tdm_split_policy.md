# Tunable split-K policy for the unified gfx1250 TDM GEMM

The explicit TDM GEMM now supports two reduction strategies in the **same
kernel definition**, with the same unroll2 main loop and common TDM helpers.
Existing GEMM dispatch/tuning tables are not changed.

| split_k_policy | split_k > 1 | Precision / memory |
|---|---|---|
| `atomic` | rank0 zero + cluster barrier, packed BF16/FP16 atomic adds | rounded partials and low-precision additions, nondeterministic, no partial workspace |
| `workspace` | FP32 LDS C-shuffle -> FP32 scratch -> cluster barrier -> rank0 fixed-order FP32 sum | no half rounding until final sum; scratch proportional to padded M*N*split_k |

Explicit calls keep the prior `split_k_policy="atomic"` default for compatibility.
**The offline tuner defaults to workspace only**, always including split_k=1.
Atomic tuning requires `--allow-atomic`; loading atomic-capable contracts
requires `allow_atomic=True` as well. A measured error bound is not a proof of
accuracy on arbitrary data: cancellation/overflow remain limitations of atomic.

## Direct control

    from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm as gemm

    # B is logical [K,N]; weights are physical [N,K].
    y = gemm(a, weight.t(), split_k=4, split_k_policy="workspace")
    y = gemm(a, weight.t(), split_k=4, split_k_policy="atomic")

Workspace mode supports odd N and deterministic-algorithm mode. It may still
differ from split_k=1 due to regrouping FP32 products. Both modes add optional
bias once, after final output rounding. Scratch lifetime is recorded on the
specified stream. Different output buffers can execute concurrently without
shared counters or a persistent workspace.

## Offline tuning with AITER infrastructure

This is a `GemmCommonTuner` subclass: ranking, tuned/profile CSV generation and
merge use AITER's base tuner. Measurements use AITER `run_perftest` on an
eight-call CUDA Graph replay, divided by eight. They include all producer,
cluster synchronization, scratch and output work, but not host allocation cost.

    python op_tests/tuners/tune_gemm_a16w16_tdm.py \
      --shapes 256,256,8192 1024,1024,8192 8192,8192,8192 \
      --dtype bf16 --splits 1,2,4,8 --workspace-mib 256 \
      --rounds 5 --iters 24 -o /tmp/tdm_bf16.csv

    # Input CSV alternatively contains M,N,K columns:
    python op_tests/tuners/tune_gemm_a16w16_tdm.py \
      -i shapes.csv --dtype fp16 -o /tmp/tdm_fp16.csv

    # Explicit low-precision opt-in; does NOT guarantee atomic will win.
    python op_tests/tuners/tune_gemm_a16w16_tdm.py \
      --shapes 256,256,8192 --dtype bf16 \
      --allow-atomic --errRatio .008 -o /tmp/tdm_atomic.csv

Search covers a finite set of tiles/wave shapes, K tiles 64/128/256, stages2/3,
group_m0/4, split factors and allowed reduction modes. The default split list
is 1/2/4/8; `--splits` accepts factors1..9, subject to K divisibility.
Split1 is always retained even if omitted in that list.

Pruning checks full kernel legality, wave/WMMA geometry, LDS bytes (including
FP32 C-shuffle for workspace), K pipeline depth, odd-N atomic restrictions and
workspace cap. A grid heuristic excludes split>1 when output_tiles*split>2*CUs.
`--exhaustive` disables that grid heuristic and one deep-small-K heuristic;
it does not remove numerical, legal-shape or memory limits.

Every candidate is tested against an independent FP32 matmul reference rounded
to the output dtype, with optional bias. TF32 is disabled. Eager output and
changed-input graph replay must be finite and within the relative-L2 threshold
and a maximum-error check. GPU/runtime errors abort the run; invalid/numerically
rejected candidates are recorded. The fastest candidates and a representative
of each allowed reduction family are remeasured in shuffled order. Final ranking
uses the median of confirmation rounds. No universal peak or exhaustive optimum
is claimed. Input distributions here are two reproducible random seeds, not a
replacement for application-specific numerical qualification.

`--errRatio` means relative L2 for this tuner (unlike some other AITER tuners).
Effective limits are capped by dtype and atomic permission:
BF16 .004 / FP16 .0006 without atomic; BF16 .02 / FP16 .003 with atomic.
Default requested limit is .004; using `.008` in the example above is explicit.
The limit is part of the lookup identity, not merely a selection annotation.

Output files:
- `-o`: winners, merged by exact contract, preserving other measured rows;
- `<output>.audit.json`: all candidate rejections, screen times and confirmation
  samples for the current invocation;
- optional `-o2`: AITER profile of the confirmed shortlist.
Re-tuning replaces the same contract instead of creating duplicate rows.
Use separate output files for concurrent tuner processes; concurrent writes to
one table are not supported. This tuner requires an exclusive GPU and `--mp 1`.
It intentionally rejects `--compare`, `--run_config` and `--all` rather than
pretending to implement the production dispatch of a different GEMM family.

## Reuse measured policy, no hidden runtime tuning

Load the file once before graph capture. Shapes, dtype, architecture/CUs,
layout, actual leading strides, bias presence, atomic permission, workspace
budget and effective numerical limit form the exact key.

    from aiter.ops.flydsl import TdmTunedPolicy

    policy = TdmTunedPolicy("/tmp/tdm_bf16.csv")
    config = policy.select(a, weight.t())  # inspect the selected strategy
    y = policy(a, weight.t())              # ordinary explicit GEMM wrapper
    # Warm once, then capture policy(a, weight.t(), out=y) in a CUDA Graph.

    atomic_policy = TdmTunedPolicy("/tmp/tdm_atomic.csv")
    y = atomic_policy(
        a, weight.t(), allow_atomic=True, relative_l2_limit=.008,
    )

No matching contract raises `KeyError`, not a guessed/rounded shape fallback.
For an 8-MiB tuned budget, also pass `max_workspace_bytes=8 << 20` at runtime.
Tables with invalid keys/configs, duplicate contracts, nonpositive/nonfinite
timings, unsupported dtype/architecture, invalid strides, over-budget scratch
or out-of-contract measured errors are rejected. This is validation of metadata,
not authentication of a CSV: only load tuning tables you trust.

Deterministic algorithms reject a selected atomic split, even with opt-in.
No default AITER environment configuration is overwritten. Kernel/target or
compiler upgrades can change performance; re-tune after such changes.

## Validation and local measurements

Final combined suite: **169 passed** on C5 gfx1250.
Ruff check/format and `git diff --check` also passed.

Tests cover:
- existing packed path plus workspace split2..9, both dtypes and schedules;
- FP32 cancellation/overflow cases, odd N, guarded strides, bias;
- changed-input graph replay, deterministic workspace and simultaneous streams;
- candidate pruning/config serialization, precision-key isolation, invalid rows;
- real tuner subprocess -> CSV -> retune/dedup -> load -> graph replay;
- mismatched shape/precision/atomic/budget lookup rejection.

Run:

    python -m pytest -q op_tests/test_gemm_a16w16_tdm.py \
      op_tests/test_gemm_a16w16_tdm_workspace.py \
      op_tests/tuning_tests/test_tdm_split_policy.py \
      op_tests/tuning_tests/test_tdm_split_tuner_pipeline.py

C5 gfx1250, FlyDSL0.5.0; measured before publication with five confirmation rounds:
- BF16 M=N256,K8192, safe mode: tile64x64x128, stages3, 2x2 waves,
  workspace split4, **12.28 us**. Explicit atomic opt-in selected the same tile
  with atomic split4 at **9.10 us**, measured relative-L2 vs rounded reference
  about .00353 rather than .000047.
- FP16 same shape, safe mode: workspace split4, **12.36 us**.
- BF16/FP16 M=N1024,K8192: tile64x64x128 stages3, **split1**,
  approximately **19.90/20.98 us**.
- BF16/FP16 8192 cubed: tile256x256x128 stages2, interleaved 2x2,
  **split1**, approximately **461.02/488.42 us**.
- FP16 NN+bias M65,N79,K1024 also tuned/loaded with odd N and FP32 workspace.

Separate CSV-reload measurements verify selected configs execute and match
direct calls. A/A differences can be several percent; small timing differences
should not be interpreted as universally stable wins.
These are whole-GEMM GPU times, not serving TTFT/TPOT results. The default grid
pruner intentionally declines unhelpful split candidates on 8192 cubed.

Raw winner configurations, confirmation samples, and separate policy reload timings
are recorded in `gfx1250_tdm_split_policy_results.json`.

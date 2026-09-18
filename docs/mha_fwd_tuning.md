# MHA forward tuning contract

This document defines the Aiter contract for packed-varlen MHA forward tuning.
Candidate enumeration, correctness, timing, winner serialization, and runtime
dispatch are owned by Aiter.

## Hardware identity

MHA winner rows begin with three hardware fields:

1. `gfx`: instruction-set architecture, for example `gfx942`.
2. `gpu_model`: normalized product model, for example `mi300x` or `mi325x`.
3. `cu_num`: visible compute-unit count, including partitioned or binned devices.

All three fields participate in exact lookup. Architecture and CU count alone
do not distinguish MI300X from MI325X when both expose gfx942 and 304 CUs.
`gpu_model` uses a stable lowercase token rather than the complete marketing
string returned by the runtime. `AITER_GPU_MODEL` provides an explicit override
for controlled replay environments.

Rows without `gpu_model` are not silently assigned to the current product.

## Three distinct artifacts

The MHA workflow does not use one CSV for three jobs:

1. The workload catalogue is the `-i` CSV. It contains only normalized problem
   fields. Hardware identity is attached by the tuner.
2. The measurement record is the `-o2` CSV plus an append-only JSONL journal
   and an atomic evidence manifest. It contains every candidate, timing,
   correctness result, and failure.
3. The runtime artifact is the `-o` CSV. It contains only the exact hardware
   and problem key plus `backend,num_splits,backend_config`.

The runtime CSV is a cross-backend dispatch table. It does not store latency,
error, or status. `backend_config` is a JSON object string for Triton and Gluon
tile launches, and empty for every other backend. A runtime reader rejects
measurement columns rather than silently treating evidence as deployable policy.

CK recipe search is not expanded here: the tuner measures the current default
CK launch. A faster correctness-gated CK, Triton, Gluon, FlyDSL, OPUS, or ASM
candidate is the winner and is launched as-is.

## Split-KV and backend launch

- `backend=asm_v3` requires `num_splits` in `[1, 8]` and launches
  `_fmha_v3_varlen_splitkv_fwd` so a tuned unsplit row is not re-auto-selected.
- `num_splits=1` selects the non-split ASM implementation through that operator.
- `num_splits=2..8` selects the corresponding split-KV implementation and
  combine stage.
- `ck` launches `mha_varlen_fwd` / `FlashAttnVarlenFunc` with
  `selected_backend="ck"`.
- `triton` and `gluon` launch the existing varlen entry with the stored
  `backend_config`. A direct call to
  `aiter.ops.triton.attention.mha.flash_attn_varlen_func` (or `flash_attn_func`)
  with `config=None` performs the same exact CSV lookup and uses those tiles
  when the winning backend matches; otherwise it keeps the DEFAULT.json /
  mha.json feature-bucket fallback.
- `flydsl` and `opus` launch the same functions the tuner already measures.
- An explicit public API `num_splits` argument on the dense D64 path remains a
  caller override and is left unchanged.

The tuner compares every legal candidate. Runtime code validates a loaded plan
against the same backend, architecture, split, and config vocabulary before
launch. Unknown backends, illegal splits, and schema mismatches fail closed.
Lookup never broadens an exact row to a nearby shape.

## Search and result records

`aiter.ops.mha_fwd_policy` owns immutable records and exhaustive enumeration for:

- `MhaFwdProblem`: normalized exact problem and hardware identity;
- `MhaFwdCandidate`: one legal backend, split, and launch configuration;
- `MhaFwdPlan`: one runtime dispatch payload for any legal backend; and
- `MhaFwdResult`: typed status, error value, and timing samples.

Candidate evidence records `ok`, `unsupported`, `mismatch`, `oom_preflight`,
`oom_runtime`, `timeout`, or `crash`. Separately, the whole run advances through
`failed`, `partial`, `measured`, `verified`, and `review-ready`. Status is never
stored in the runtime artifact.

Current enumeration is: ASM splits 1–8, one CK default, the Triton grid, Gluon
and OPUS on gfx950, and FlyDSL on gfx1250. Tuner-only paths that production
cannot launch stay invalid.

## Lifecycle

A complete MHA tuning run:

1. Normalize an explicit problem catalogue.
2. Enumerate every legal candidate.
3. Correctness-gate and measure candidates. ASM measurement uses
   `_fmha_v3_varlen_splitkv_fwd(..., num_splits)`.
4. Append each candidate result to a deterministic-ID journal so a killed or
   faulted shape group can resume only missing phases.
5. Re-measure finalists in fresh worker rounds and select by median latency.
6. Write the measurement record, cross-backend runtime CSV, and evidence
   manifest atomically. The fastest correctness-gated row is the winner.
7. Start a fresh process with `AITER_CONFIG_MHA_FWD` pointing to that file.
8. Run the public MHA operator, repeat correctness, and prove the expected
   backend, split count, and backend config were actually dispatched.

Direct candidate timing and fresh public-path timing are separate evidence
domains. A candidate is not a deployed winner until public replay succeeds.

## Fallback

An absent exact row preserves Aiter's existing production fallback: FlyDSL
when it claims the call, otherwise `fmha_v3_varlen_fwd` with C++ `num_splits=0`
auto-select. Malformed rows, duplicate exact keys, unknown backends, bad
splits, measurement columns, and incompatible hardware fail closed. Runtime
lookup does not broaden an exact row to another GPU model, architecture, CU
count, or problem shape.

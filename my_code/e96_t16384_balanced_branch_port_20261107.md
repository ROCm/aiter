# E96/T16384 balanced GEMM branch port

## Scope and source revisions

Port useful GEMM1/GEMM2 scheduling changes from
`at_merge_sched_vgpr_regression_20261103:hyg_moe_a4w4_prefill_pr`
into `at_merge:hyg_opt_moe_a4w4_prefill_dsv41`, while preserving the
destination branch's supported inputs, numerical accuracy, and existing
prefill and short-row optimizations.

| Role | Commit |
|---|---|
| Source branch | `f2aec9adfc1d565ccc0f277742b4d43c7ee79000` |
| Source parent | `ad9882df24b734748974afa161eb66ed76108fcf` |
| Destination baseline | `9bebc86b4e8ac4c3c6b1db8021318b45b580fc6f` |

Target: E96/T16384/topk6/H7168/I3072, balanced routing, random uniform
inputs and automatic scales. Test only on a07-3 in `hyg_fyd1` with
FlyDSL `0.4.1.dev1005`, `AITER_LOG_MORE=0`, `--warmup 5 --iters 101`.

Both branches select `t256x256x256/w2x2/b4` and `CM4/CN4` for both GEMMs.
Test APRE1 with pin0 and pin1; record the actual launch parameters as well
as the requested policy.

## Code differences to validate

The source's newest kernel commit adds early initialization of resident
accumulators, M288 register/scale/store support, and updated work metrics.
The target tuning row still uses M256, so M288 support itself does not
explain a speed difference for this row. Its TFLOPS/BW accounting changes
also do not establish a kernel latency improvement.

The first candidate ports the early initialization: explicitly schedule
the initial C zero in its resident VGPRs after the prologue TDM issues and
before the first input fence. This places loop-carried initialization
before that wait instead of leaving its constant materialization at the
loop entrance. It retains peeled initialization and existing register
placement for all other paths.

Unlike a wholesale file replacement, this candidate preserves the
destination's generic native compute schedule, EP scatter handling,
quantized output layouts, `k_waves`, pointer/Tensor ABI handling, and the
verified E96/T512 short-row allocator fix.

Short K pipelines keep their original initialization: the new gate
requires `K_TILES > num_buffers`, because the early-zero operation is
inserted in the mid-compute prefetch branch. Legacy FlyDSL without
`rocdl.sched_vgpr` also keeps the original path.

The initial C remains positive FP32 zero; all WMMA operands, accumulation
order, activation, quantization and valid-row output computations remain
the same. This is a scheduling change. Acceptance still requires actual
reference comparisons, not this argument alone.

## Measurement policy

Use complete, hash-verified source manifests in isolated worktrees. Git
runs on the host, while GPU workers run inside `hyg_fyd1`. Reuse a native
extension only after verifying that its complete native build-input trees
are identical. Keep the original op test and `run_perftest` timer.

Collect three valid samples and medians, with adjacent before/after runs
and alternating order. A candidate needs at least 0.8% improvement before
the additional forward/reverse control and independent three-sample
archive. Keep failed and interrupted attempts in the evidence; do not
select samples based on their latency.

## Test coordination

The user's manual Non-balanced test controller became active during setup.
This task's initial correctness controller briefly overlapped it and was
terminated using its own controller PID and verified worker run tokens.
No other process was terminated. The affected interval is recorded in
`e96_balanced_branch_port_20261107/initial_correctness_paused.json` and
the initial controller's records. Performance measurements overlapping
that interval must not be accepted.

Subsequent automatic comparisons wait for
`run_remaining_prefill_pr_tests.py` to exit, and then use the usual
whole-machine GPU/KFD idle checks. A transient idle gap between the user's
workers does not authorize this task to start its own measurement.

## Target-workload results

The candidate is `67757d4f3fa993ad534cf9bdc90838c5ec732d34`, based on
`9bebc86b4e8ac4c3c6b1db8021318b45b580fc6f`. Its only changed source file is
`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`, SHA256
`4f8fb5d906a6cec51bd866f8aae9fd51adc5692e0c6c953be9218d2340921551`.

Initial adjacent source/destination comparison, three accepted samples per
configuration:

| Requested pin | Source GEMM1 us | Destination GEMM1 us | Source GEMM2 us | Destination GEMM2 us |
|---|---:|---:|---:|---:|
| 0 | 623.778313 | 627.038020 | 390.283459 | 390.344418 |
| 1 | 563.734000 | 578.061899 | 361.629293 | 368.181622 |

The source's updated GEMM1 bandwidth accounting differs from the
destination test's accounting. Do not interpret that BW difference as a
change in memory traffic or performance. The original timer implementation
has the same SHA256 in both source trees:
`ab6fca8ff0c159869ac12016caf852efab8eba458f9f5381b9f2ad3aa5e58032`.

Candidate screening, three accepted samples per configuration:

| Requested pin | GEMM1 us | GEMM2 us | GEMM1 delta vs initial destination | GEMM2 delta vs initial destination |
|---|---:|---:|---:|---:|
| 0 | 627.879653 | 389.912959 | +0.134224% | -0.110533% |
| 1 | 571.639720 | 361.330302 | -1.110985% | -1.860853% |

The pin1 improvements exceeded the 0.8% screening threshold. The next
phase alternated adjacent destination/candidate and candidate/destination
runs and retained three complete pairs per configuration:

| Requested pin | Baseline GEMM1 us | Candidate GEMM1 us | GEMM1 delta | Baseline GEMM2 us | Candidate GEMM2 us | GEMM2 delta |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 628.466030 | 625.666384 | -0.445473% | 390.908162 | 390.992859 | +0.021667% |
| 1 | 578.702140 | 569.554753 | -1.580673% | 368.317020 | 361.695929 | -1.797661% |

Pin1 passed the improvement gate for both GEMMs. Pin0 stayed within the
0.5% maximum-regression gate. These same-run comparisons establish the
benefit; a later archive should not be mixed with an earlier baseline to
claim a different improvement percentage.

Independent three-sample archive after the control passed:

| Requested pin | GEMM1 us | GEMM1 TFLOPS | GEMM1 effective BW (GB/s) | GEMM2 us | GEMM2 TFLOPS | GEMM2 effective BW (GB/s) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 628.020152 | 13787.223305 | 5134.184297 | 390.030602 | 11099.967571 | 6903.927933 |
| 1 | 570.304240 | 15182.517438 | 5653.773852 | 361.727216 | 11968.485745 | 7444.126527 |

TFLOPS and BW are the original destination test's work-accounting
metrics, not hardware-counter measurements of physical traffic. Every
accepted run passed its unchanged numerical reference test.

The source comparison and control each retained 12 of 15 raw records;
whole pairs were excluded after invalid ownership/post-timing checks.
Screening retained 6 of 8 raw records. The archive retained all 6 records.
Raw attempts and exclusion reasons remain in their JSON evidence. No
attempt was selected or excluded based on its measured latency.

The candidate's GEMM2 reaches the source branch's observed performance
range. GEMM1 remains about 1% slower than the initial source result; the
remaining difference has not been causally isolated. This port does not
claim that all source/destination differences have been transplanted.

## Source integration

Target-workload screening, forward/reverse control, independent archiving,
existing prefill compatibility, and eleven complete MoE output comparisons
have passed. The candidate kernel has replaced the destination working-tree
source locally and in a07-3's `/data/yanguahe/code/wk_sp1/at_merge`.
The SHA256 matches at both locations. The accepted candidate was based
on branch HEAD `9bebc86b`. Its kernel content replaced the working-tree
source in `hyg_opt_moe_a4w4_prefill_dsv41` at both locations before
publication.

The tracked source diff is one file, 25 insertions and 4 deletions. It adds
no function, changes no import, and adds no expert-count, token-count, or
model-name dispatch condition. Its new condition is evaluated at compile
time from existing pipeline state. The source branch's M288 extension and
test-accounting changes are separate features and were not part of this
M256 scheduling port.

Review the production change with:

```bash
git diff -- aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
```

## Numerical and boundary validation

The original `--scenario verify` test, original reference, and production
`logits_diff < 0.01` gate are unchanged. Each verification hashes the
actual returned MoE `out`, including all output bytes. It does not hash
the reference as a substitute. The two source trees' valid outputs are
identical for all eleven cases:

| Case | Baseline logits_diff | Candidate logits_diff | Complete output bytes |
|---|---:|---:|---|
| E96/T16384 balanced, seed0 | 3.541113117e-06 | 3.541113117e-06 | Identical |
| E96/T16384 balanced, seed1 | 3.543033467e-06 | 3.543033467e-06 | Identical |
| E96/T16384 non-balanced, seed0 | 3.604417552e-06 | 3.604417552e-06 | Identical |
| E96/T16384 non-balanced, seed1 | 3.605155652e-06 | 3.605155652e-06 | Identical |
| E64/T1536/topk8 non-balanced | 3.605092039e-06 | 3.605092039e-06 | Identical |
| E96/T512/topk6 balanced | 3.539569267e-06 | 3.539569267e-06 | Identical |
| E96/T512/topk6 non-balanced | 3.590504145e-06 | 3.590504145e-06 | Identical |
| E1/T16384/topk1 non-balanced | 0 | 0 | Identical |
| E4/T384/topk2/H1024/I512 | 4.006129761e-06 | 4.006129761e-06 | Identical |
| E4/T384/topk2/H2048/I1152 | 4.080747823e-06 | 4.080747823e-06 | Identical |
| E4/T384/topk2/H2048/I1280 | 4.095541916e-06 | 4.095541916e-06 | Identical |

All hash checks use APRE1/pin1, random uniform input, automatic scales,
Silu, and no bias. E64 uses H7168/I2048; E96 uses H7168/I3072;
E1 uses H16384/I8192. The last three cases use explicit verification-only
CSV rows for the common M256/N256/K256, w2x2/b4, CM4/CN4 geometry.
They are local test fixtures, not changes to production tuning tables.

The H1024 case checks GEMM1's `K_TILES == num_buffers == 4` initialization
boundary. Its GEMM2 K512 automatically reduces the requested four buffers
to two, also checking the established short-pipeline path. The H2048
cases exercise the new early initialization in GEMM1 and GEMM2. GEMM2
K1152 has five tiles with a final K128 tail; K1280 has five full K256 tiles.
T384 provides 192 live rows per expert, so these checks also include
partial M256 tiles.

The initially attempted H1152/I512 fixture failed on the baseline before
any numerical comparison: N1152 cannot use GEMM2 tile_n2=256. A fitted
tile_n2=128 exposed the additional established GEMM1 routed-input APRE
requirement that logical K be divisible by 1024. These two failed baseline
attempts remain in `output_hash_verification/` and
`output_hash_verification_short_k/`. They were not counted as passes.
Using H2048 with GEMM2 K1152/K1280 tests the intended K boundary while
respecting the unchanged input packer's support range.

The value-level argument agrees with the tests: both paths start every C
fragment from positive FP32 zero, and the returned `sched_vgpr` SSA value
is consumed by the store. The existing resident C range and size check
are preserved. The port changes placement of initialization relative to
the input wait; WMMA operands, accumulation order, K-loop extent,
activation, scale computation, quantization, and valid-row output are
unchanged. Pin0, peeled initialization, short pipelines, and legacy API
paths retain their prior initialization code. Legacy API behavior is a
code-preservation check; GPU execution in this task used FlyDSL
`0.4.1.dev1005`.

Hash evidence: `e96_balanced_branch_port_20261107/output_hash_verification_combined.json`
contains the eleven comparison hashes and the two underlying result
paths. It records 22 accepted verification workers and preserves the
invalid-fixture history separately. These eager checks do not call the
performance timer.

## Existing prefill compatibility

Adjacent baseline/candidate runs also cover the existing H5120/I2304,
E96/topk6, non-balanced per-rank compute workloads. Both versions use
pin1 and preshuffled GEMM1 input. The worker performs the existing EP4
production config lookup for `4*T` incoming-capacity tokens, while the
actual valid input is `T`. This is compute evidence, not an actual
four-rank EP dispatch/combine benchmark.

| Valid T per rank | Version | GEMM1 us | GEMM1 TFLOPS | GEMM1 effective BW (GB/s) | GEMM2 us | GEMM2 TFLOPS | GEMM2 effective BW (GB/s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | Baseline | 148.302313 | 7819.440880 | 9327.797111 | 88.979222 | 6516.359331 | 9927.708087 |
| 4096 | Candidate | 147.936857 | 7838.757645 | 9350.840046 | 88.835030 | 6526.936311 | 9943.822172 |
| 5120 | Baseline | 170.553140 | 8499.119174 | 8374.850912 | 120.379768 | 6020.743728 | 7923.213165 |
| 5120 | Candidate | 171.287470 | 8462.682427 | 8338.946918 | 120.238111 | 6027.836969 | 7932.547769 |
| 6144 | Baseline | 174.666360 | 9958.767990 | 8435.398505 | 122.830592 | 7080.735055 | 8338.552804 |
| 6144 | Candidate | 172.098809 | 10107.343363 | 8561.246650 | 123.103636 | 7065.029947 | 8320.057849 |
| 16384 | Baseline | 373.444152 | 12421.039807 | 5151.004938 | 256.377010 | 9046.374076 | 6742.326599 |
| 16384 | Candidate | 374.167773 | 12397.018161 | 5141.043162 | 256.214667 | 9052.106072 | 6746.598696 |

| Valid T per rank | GEMM1 latency delta | GEMM2 latency delta | Maximum 0.5% regression gate |
|---:|---:|---:|---|
| 4096 | -0.246426% | -0.162051% | Pass |
| 5120 | +0.430558% | -0.117675% | Pass |
| 6144 | -1.469975% | +0.222294% | Pass |
| 16384 | +0.193770% | -0.063322% | Pass |

All 24 performance records form 12 accepted complete pairs, with three
samples per version and valid T. Eight untimed preflight checks also
passed the original numerical reference. None of these records needed
exclusion or remeasurement.

The captured GEMM1 configuration uses CM4/CN2 with peeled initialization;
GEMM2 uses CM1/CN4. Neither configuration enables the newly introduced
`early_zero_acc` branch. The numerical and timing comparisons validate
that these established paths were preserved; small measured changes here
are not claimed as additional optimization gains.

## Reproduction

Run from the tested repository root inside `hyg_fyd1`. Repeat the command
three times for the archive and take each GEMM's latency median. Set
`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0` to reproduce the pin0 policy.

```bash
PYTHONPATH="$PWD" \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=0 \
AITER_FORCE_GFX1250=1 \
GPU_ARCHS=gfx1250 \
FLYDSL_GPU_ARCH=gfx1250 \
CU_NUM=256 \
AITER_FORCE_A8W4=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_GROUPED_A4W4_A_PRESHUFFLE=1 \
AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1 \
AITER_CONFIG_GROUPED_FMOE="$PWD/aiter/configs/tuned_grouped_fmoe.csv" \
python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
  --scenario kernel --data-format a4w4 \
  --experts 96 --tokens 16384 --topk 6 \
  --model-dim 7168 --inter-dim 3072 \
  --act silu --no-bias --no-check-aot-cache \
  --warmup 5 --iters 101 --seed 0 \
  --data-init uniform --scale-init auto
```

The host-side adjacent before/after control used full frozen revisions:

```bash
python3 -u .codex_tmp/gfx1250_prefill_pr_20261104/run_tightly_paired_source_matrix.py \
  --label e96_balanced_early_init_control_20261107 \
  --revision-directory .codex_tmp/gfx1250_prefill_pr_20261104/e96_balanced_branch_port_20261107/vs_previous \
  --cases e96_t16384 --pins 0,1 --count 3
```

The compatibility comparison used the existing EP4 config lookup:

```bash
python3 -u .codex_tmp/gfx1250_prefill_pr_20261104/run_tightly_paired_prefill_matrix.py \
  --label e96_balanced_early_init_prefill_compat_20261107 \
  --revision-directory .codex_tmp/gfx1250_prefill_pr_20261104/e96_balanced_branch_port_20261107/vs_previous \
  --tokens 4096,5120,6144,16384 --pin 1 --a-preshuffle 1 --count 3
```

These labels identify existing evidence directories. Use a new label
when reproducing a measurement; the controllers deliberately refuse to
overwrite an existing run.

Evidence paths are relative to `.codex_tmp/gfx1250_prefill_pr_20261104`:

- Source comparison: `e96_balanced_source_vs_release_20261107/result.json`.
- Screening: `e96_balanced_early_init_screen_20261107/result.json`.
- Adjacent control: `e96_balanced_early_init_control_20261107/result.json`.
- Independent archive: `e96_balanced_early_init_archive_20261107/result.json`.
- Existing prefill: `e96_balanced_early_init_prefill_compat_20261107/result.json`.
- Candidate preparation, manifests, patch, acceptance, and output hashes:
  `e96_balanced_branch_port_20261107/`.

## Live working-tree verification

After promotion, one untimed `--scenario verify` run used the actual
`/data/yanguahe/code/wk_sp1/at_merge` working tree, rather than the isolated
candidate directory. It first verified every file in the complete
candidate manifest against the live repository. All 174 loaded Python
modules in the `aiter` namespace came from that live repository and
matched the manifest hashes.

The original test passed with `logits_diff = 3.5411131171589562e-06`.
The complete returned MoE output SHA256 was:

```text
956d5a5d6d6fd0386883fdafe9b6ef4a63f4429b143c0c0631ac862d98824abf
```

This is identical to both the previously verified destination baseline
and candidate. The performance timer was not called. The host-controlled
worker finished in 4.27 seconds, with the existing 115-second worker and
119-second outer timeouts. Git checks ran on the host; no Git command
ran inside the container.

Evidence in `e96_balanced_branch_port_20261107/`:

- `live_port_host_verification.json`: host command, pre/post checks,
  return code, and complete verification result.
- `live_port_verification.json`: actual output hash and loaded module paths.
- `live_port_verification.log`: original test output.
- `finalize_live_port_verification.py`: host-side sequencing and timeout guard.

## Repair of the previous manual performance run

The earlier correctness overlap affected the user's completed run
`my_code/remaining_prefill_pr_test_logs/20261106_182143_8ae0eaf6/`.
Eleven performance records directly overlapped. To preserve adjacent
comparisons, the repair withdrew three complete five-configuration sample
groups, including four records that did not overlap directly:

| Workload, all non-balanced | Sample | Replaced complete-group records |
|---|---:|---:|
| E96/T16384/topk6/H7168/I3072 | 2 | 5 |
| E1/T16384/topk1/H16384/I8192 | 0 | 5 |
| E1/T16384/topk1/H16384/I8192 | 1 | 5 |

The first repair controller incorrectly required its first post-exit GPU
utilization read to be zero. It rejected repeated samples even when KFD
owners were empty before and after exit, VRAM was unchanged at 176582656
bytes, and only the recently completed worker's utilization counter still
showed 5--8%. That controller was stopped; its attempts and interruption
remain in `overlap_repair_completed/`. None of those rejected attempts
was silently reused or chosen for its latency.

The corrected controller uses the existing held-context ownership audit:
stable whole-machine idle before launch, the unchanged original timer and
reference, synchronization of all devices after timing, exact measured
worker identity, no foreign KFD owner, explained VRAM, and no measured
worker KFD ownership after exit. It retains the first post telemetry
snapshot and refreshes a residual utilization counter after six seconds.
These checks run before launch or after timing, never concurrently with
the measured kernel.

Each of the three replacement groups passed on its first corrected
attempt. All 15 workers passed source, numerical, ownership, exit and
adjacency checks; the longest worker was 7.43 seconds. Routing hashes
match within each group. The original result was not modified; its SHA256
remains:

```text
2b0ae2c0901b23ce10577af296deb5ef6bc83aea1eb8fef3b6ae14b4fbdd72e6
```

The repaired derived report is:

```text
.codex_tmp/gfx1250_prefill_pr_20261104/e96_balanced_branch_port_20261107/overlap_repair_postcheck_fixed/derived_previous_run/summary.md
```

Its `result.json` preserves the withdrawn records, original result hash,
audit path and replacement worker identifiers. Other records retain
their original evidence. This restores the previous frozen
`9bebc86b` Non-balanced measurements; it is not a timing comparison of
the newly ported kernel.

That prior-run report remains `complete_with_regressions`: 60 checks
completed, 50 passed, 10 failed, with no pending checks and CPU ABI passed.
Those results belong to the frozen versions in that report. They do not
replace this port's target-workload control, compatibility measurements,
or output verification. The complete PR's outstanding acceptance and
publication work are not claimed complete by this single port.

## Test driver and publication state

`my_code/run_remaining_prefill_pr_tests.py` defaults to the three added
Non-balanced workloads before the original full test sequence.
`--non-balanced-only` runs only those three workloads. Their after
manifest remains the complete frozen `9bebc86b` revision, intentionally
preserving the manual run and repair source identities. To measure this
new live working-tree port, use the explicit target command in
Reproduction above; rerunning the frozen driver is not a substitute.

The local and remote production kernel sources match the accepted
candidate SHA256 and use LF with no UTF-8 BOM. The publication unit is
the kernel port, host test driver, and both associated documents, on
`hyg_opt_moe_a4w4_prefill_dsv41`. The accepted kernel SHA256 above remains
the tested source identity across Git publication. Remote synchronization
uses a host-side fast-forward pull after saving the already copied task
files in a dedicated stash. Complete PR acceptance remains separate from
publishing this port.

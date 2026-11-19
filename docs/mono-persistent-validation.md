# A03/A09 mono validation

## Current acceptance contract: 2026-11-19

This section replaces the earlier standalone-only scope and performance gate.
The sections below retain historical results. They do not certify this contract.
The user's table is the authority for the mono rules below. SH and section
references are retained as supplied. A target main-kernel call interface has
not been connected to these standalone wrappers.

### Mono requirements

| Item | Required behavior | Required evidence |
| --- | --- | --- |
| Scheduling | Exactly 256 TGs. The selected grid is `(1,256)`, with `tg = program_id(1)`. Each op processes `task = tg + j*256`. A TG without work returns from the op and still reaches the next join. | Actual main-kernel launch, task coverage, and a join test with idle TGs. For this wrapper, check a `(1,256,1)` TG grid and `(128,1,1)` workgroup. |
| Residency | All 256 TGs must be able to reside together for the join. The flag array has 256 four-byte entries, or 1 KB. | Resource and residency checks for the final fused launch, plus a join progress test. TG count alone is not residency evidence. |
| TG shape | Four wave32 waves, or 128 threads, for every op in the launch. | Real dispatch metadata. The join assigns 64 flags to each wave. |
| SGPRs | Only ASM kernels must avoid reading or writing `s104` or `s105`. HIP, FlyDSL and Triton kernels have no such requirement. A03/A09 use Triton/Gluon and do not impose this register restriction. | Apply reserved-register operand and preservation checks to ASM kernels only. Record compiler resource usage for A03/A09 without an s104/s105 acceptance gate. |
| Cluster | Every op uses the same physical cluster geometry, currently 1 x 1. | The final launch descriptor and code-object metadata; mono section 4.6.4. |
| Output stores | DEV scope and HT for every payload, scale and metadata store handed to the next op. This includes ordinary and TDM stores. | Actual loaded ISA and producer/consumer tests; SH section 4.2.1 and Table 78. HT is a retention hint, not a cache-hit or speed guarantee. |
| Activation loads | DEV scope. TH may be omitted, with default RT. Loads and prefetches start only after the required join. | Actual loaded ISA and ordering in the main-kernel path. Load TH need not match store TH. |
| Flags and join | The main kernel stores flags with DEV + HT and loads them with DEV + default RT. Only wave0 lane0 publishes per TG, after all writers complete. The TG converges after successful reads. The all-DEV path adds no WB/INV. | The pinned join implementation, writer-completion ordering and replay tests, including idle TGs; mono section 0.4. |

Keep the original task unit explicit: one row for A03 and one 1024-element
tile for A09. A09 task count is `ceil(numel/1024)`, not the row count.
Unrolling or issuing several loads may group tasks from the required sequence
`tg + (j+k)*256`. A contiguous batch mapping must not silently redefine that
sequence. The previous adaptive two-task and four-task mappings need review
against this rule.

An op return is not a kernel exit. Idle TGs must retain a path to the main
kernel's join. Standalone Graph replay does not test this condition.

### Correctness requirements

For every shape in the source operator's supported input domain, preserve its
outputs byte for byte. Do not restrict correctness to model performance shapes.
Keep A03 rounded-amax and A09 ceil-scale algorithms separate. Do not change the
frozen native, relax comparisons or remove supported shapes to pass a test.
Inputs, weights and guards must remain unchanged. Check partial tasks, unusual
legal widths and strides, empty inputs, scale boundaries and Graph replay.

Keep all existing 89 UT cases. Add coverage for changed scheduling, all selected
model shapes and the actual mono call/join path. A finite passing suite records
tested coverage; it does not prove all possible shapes.

### Performance priorities

1. DSV4.1 c16 is the primary gate: no performance regression against the
   original/native implementation under the same model configuration.
2. DSV4.1 c64, c128 and c256 are the secondary performance targets. Report
   each target separately. Gains there do not compensate for c16 regression.
3. Other shapes remain correctness cases. Their performance is not an
   acceptance gate or an optimization target. In particular, rows 257/513
   are no longer mandatory performance points unless a target model trace
   contains them.

Pin model revision, TP/DP/EP mode, DSpark settings, phase, rank, Graph bucket,
shape, stride and dtype before assigning a concurrency label to a benchmark.
Concurrency is a request count, not a tensor row count. The local historical
recipe uses TP4 for c16/c64 and DP attention with EP4 for c128/c256. The user
confirmed this recipe, DSpark5 and the original/native baseline. With
DSpark5, a full c16 verify batch can have 96 rows. This is a source-derived
candidate, not a recorded current model trace. Draft, verify, partial batches,
rank imbalance and Graph padding can select other shapes.

Use the same GPU, input pool and Graph method for native and PS. Alternate
providers, keep hot/rotating cases and 15 trials, wait for the authorized
exclusive window, and disable profiling during timing. Report median, P10,
P90, PS/native and the noise assessment. Do not require an improvement on
every old synthetic case. Do not turn an inconclusive result into proof of
non-regression or invent an allowed slowdown. Repeat unresolved c16 evidence
as needed. Model-level non-regression requires the model path to be exercised;
an isolated op benchmark must be labeled as such.

### Current measured result: 2026-11-19 (1 x 256)

The user limits the s104/s105 rule to ASM kernels. A03/A09 use Triton/Gluon,
so the wrappers and offline compiler no longer set `amdgpu-num-sgpr=104`.
Their ISA may use these registers; this is not an acceptance failure.
DEV activation loads and DEV + HT output stores remain required.
Both PS wrappers use `(1,256)` and `tg = program_id(1)`, with 128 threads/TG.
Task ownership remains `tg + j*256`; task batching and native math are unchanged.

Tested commit: `ce7b69057f6da66c3291ec89361cdc202253034d`. All 108 UTs pass with zero skips, including
byte output checks, input/weight integrity, guards and Graph replay. Frozen
native hashes and 20 offline variants pass. Real Graph profiling records
336 PS dispatches with work-item grid `(128,256,1)` and workgroup `(128,1,1)`.
That is exactly `(1,256,1)` TGs. All 22 recorded real PS JIT variants pass
DEV activation load and DEV + HT output store checks. Actual JIT metadata
has no explicit SGPR allocation cap. Register references are recorded only;
they do not gate Triton/Gluon acceptance.

The four target profiles have 22 unique timing cases, 15 alternating trials
each, with shared native/PS input pools and the same Graph method. Timing
runs without profiling, under the shared GPU lock after the idle gate.
No foreign GPU PID was seen in any case's 100 ms samples. c256 A03 reuses
c64's identical shape. These are source-derived full/balanced verify shapes,
not four-GPU model results.

[Raw performance and launch evidence](mono-persistent-results/2026-11-19-grid1x256-unrestricted-sgpr) includes all samples, JUnit,
actual profiler records, and JIT IR/ISA/code objects/metadata in the archive.
Its manifest records file hashes and the tested revision. The previous
1 x 256 capped candidate had no GPU results; the older 16 x 16 results below
apply only to that revision. Separate windows do not isolate a geometry-only
or SGPR-cap-only speed change.

Values below are median / P10 / P90 in us. PS/native greater than one means
slower. The existing noise rule distinguishes regressions, improvements and
inconclusive cases. Inconclusive does not prove equivalence.

| Profile | Op | Input storage | Cache | Native median / P10 / P90 (us) | PS median / P10 / P90 (us) | PS/native | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| c128-verify-proxy | a03 | [192, 1792] | hot | 1.642247 / 1.628110 / 1.651762 | 1.820719 / 1.807771 / 1.834877 | 1.108675 | regression |
| c128-verify-proxy | a03 | [192, 1792] | rotating | 1.992524 / 1.983766 / 2.005778 | 2.116570 / 2.094131 / 2.128387 | 1.062256 | regression |
| c128-verify-proxy | a09 | [192, 8192] | hot | 1.795180 / 1.791494 / 1.835548 | 2.379880 / 2.351662 / 2.406297 | 1.325705 | regression |
| c128-verify-proxy | a09 | [192, 8192] | rotating | 2.787266 / 2.523658 / 2.798663 | 3.651453 / 3.619377 / 3.898790 | 1.310048 | regression |
| c128-verify-proxy | a09-qat | [192, 8192] | hot | 1.522592 / 1.518187 / 1.523627 | 1.866671 / 1.857305 / 1.873475 | 1.225982 | regression |
| c128-verify-proxy | a09-qat | [192, 8192] | rotating | 1.936981 / 1.935680 / 1.938701 | 2.561208 / 2.559520 / 2.563050 | 1.322268 | regression |
| c16-verify-proxy | a03 | [96, 1792] | hot | 1.591535 / 1.578307 / 1.601127 | 1.698395 / 1.678168 / 1.703538 | 1.067143 | regression |
| c16-verify-proxy | a03 | [96, 1792] | rotating | 1.915124 / 1.911720 / 1.925315 | 1.977866 / 1.968349 / 1.998499 | 1.032761 | regression |
| c16-verify-proxy | a09 | [96, 2048] | hot | 1.425262 / 1.398810 / 1.435245 | 1.434224 / 1.428618 / 1.450415 | 1.006288 | inconclusive |
| c16-verify-proxy | a09 | [96, 2048] | rotating | 1.673170 / 1.667351 / 1.687780 | 1.688970 / 1.674123 / 1.693980 | 1.009443 | inconclusive |
| c16-verify-proxy | a09-qat | [96, 2048] | hot | 1.248566 / 1.246202 / 1.254113 | 1.251943 / 1.250060 / 1.256892 | 1.002704 | inconclusive |
| c16-verify-proxy | a09-qat | [96, 2048] | rotating | 1.612727 / 1.611016 / 1.614337 | 1.622654 / 1.620817 / 1.624282 | 1.006155 | regression |
| c256-verify-proxy | a09 | [384, 8192] | hot | 2.767693 / 2.689468 / 2.925294 | 3.283733 / 3.232896 / 3.430986 | 1.186452 | regression |
| c256-verify-proxy | a09 | [384, 8192] | rotating | 3.643368 / 3.302288 / 3.995756 | 5.523183 / 5.348033 / 5.537386 | 1.515955 | regression |
| c256-verify-proxy | a09-qat | [384, 8192] | hot | 2.042780 / 2.040198 / 2.043853 | 2.621875 / 2.620315 / 2.624765 | 1.283484 | regression |
| c256-verify-proxy | a09-qat | [384, 8192] | rotating | 2.683343 / 2.681049 / 2.685822 | 3.519535 / 3.517631 / 3.523570 | 1.311623 | regression |
| c64-verify-proxy | a03 | [384, 1792] | hot | 1.764094 / 1.749916 / 1.786133 | 2.593680 / 2.588503 / 2.603332 | 1.470262 | regression |
| c64-verify-proxy | a03 | [384, 1792] | rotating | 2.128011 / 2.122766 / 2.129647 | 3.026929 / 3.023472 / 3.277277 | 1.422422 | regression |
| c64-verify-proxy | a09 | [384, 2048] | hot | 1.603946 / 1.597546 / 1.607983 | 1.695803 / 1.692431 / 1.698723 | 1.057270 | regression |
| c64-verify-proxy | a09 | [384, 2048] | rotating | 2.424262 / 2.396061 / 2.430109 | 2.709286 / 2.699976 / 2.721768 | 1.117571 | regression |
| c64-verify-proxy | a09-qat | [384, 2048] | hot | 1.376505 / 1.373316 / 1.378113 | 1.404523 / 1.401549 / 1.406672 | 1.020355 | regression |
| c64-verify-proxy | a09-qat | [384, 2048] | rotating | 1.726001 / 1.725239 / 1.727264 | 1.795072 / 1.792825 / 1.796607 | 1.040018 | regression |


Decisions: {'regression': 19, 'inconclusive': 3}. Primary c16: a03 hot regression (+6.71%); a03 rotating regression (+3.28%); a09 hot inconclusive (+0.63%); a09 rotating inconclusive (+0.94%); a09-qat hot inconclusive (+0.27%); a09-qat rotating regression (+0.62%).
The full mono call/join path, simultaneous residency, common fused cluster
and model path still require runtime evidence. The s104/s105 preservation
gate applies only to ASM kernels.

### Previous grid-only candidate: 2026-11-19

The user clarified that the invariant is 256 TGs, not a 16 x 16 geometry.
The A03 and A09 PS wrappers now use `(1,256)`. The kernel takes the TG id
from `program_id(1)`. Task ownership remains `task = tg + j*256`.
The TG still has four wave32 waves (128 threads). Batch selection, native
math, DEV activation loads and DEV + HT output stores are unchanged.
The prior 16 x 16 measurements below remain historical. GPU validation of
the new geometry is pending; do not relabel the old trace as a 1 x 256 run.

Candidate commit: `2dc6f7c3516d3a99c04a4be2cd33870775d0e3b4`.
Black and Ruff pass. All 12 frozen native symbol hashes and 20 offline builds
pass. Static checks of 12 PS variants pass for DEV loads, DEV + HT stores and
no s104/s105 operands. These are offline checks, not GPU correctness or launch
evidence. See the [candidate manifest](mono-persistent-results/2026-11-19-grid1x256/manifest.json).

The GPU attempt waited for the shared lock. Another job held it and four
waiters preceded this candidate. The attempt was stopped before lock acquisition;
no UT, real-launch trace or performance run started. The resumable runner is
in `/home/weizhu15/workspace/mono-grid1x256-20261119`. Resume when the shared
lock and five-second exclusive idle gate are available. Keep the earlier
16 x 16 evidence attached to its original revision.

### Previous 16 x 16 result: 2026-11-19

Published evidence is in [mono-persistent-results/2026-11-19](mono-persistent-results/2026-11-19).
The [manifest](mono-persistent-results/2026-11-19/manifest.json) records the tested
commit, result counts and file hashes. The four performance JSON files contain
all 15 trial samples. JUnit, the real launch gate and the process audit are
included. `validation-evidence.tar.gz` contains actual wrapper JIT IR, ISA,
code objects and metadata, full profiler traces, final test logs, and the two
earlier performance rounds. `jit-evidence.json` maps each JIT hash to archive
paths. The archive preserves original runtime paths in the raw launch maps;
use the manifest and JIT evidence map to locate the portable copies.


Tested code: `9a294c8c1661c5c69e8b75ea8c8ce7fdb7d977c8`.
The full acceptance item remains open. Correctness and standalone launch
checks pass. The c16 A03 performance gate fails.

Keep the best measured candidate under the current contract. Both ops use
one task per TG iteration when the task count is at most 256. A03 retains
one task through 767 rows and four above that bound. A09 retains one/two/four
tasks at the 256/512 bounds. These choices do not guarantee no regression.
Do not replace a measured path with an untested fixed batch size.

The wrappers use physical `(16,16)` TG grids. Each batched task retains the
original `tg + j*256` owner. Activation loads use DEV; all output payload and
scale stores use DEV + HT. PS wrappers limit SGPR allocation with
`amdgpu-num-sgpr=104`. Actual instruction operands are checked as well.
Native and non-PS A03 sources are unchanged.

All 108 GPU tests passed with zero skips. This retains the prior 89 tests
and adds target rows and the DP-attention width. Byte comparisons, input and
weight integrity, guards, scale cases and Graph replay pass. All 12 frozen
native symbol hashes and 20 main offline builds pass. Three wide-stride
fallback variants also compile; those fallback paths do not yet have GPU
coverage. The earlier unsupported `gl.broadcast_to` failure and the SGPR
violations before the allocation limit remain in the evidence directory.

Real Graph profiling records 336 PS dispatches across eight kernel names.
Every work-item grid is `(2048,16,1)` and every workgroup is `(128,1,1)`:
16 x 16 TGs, 128 threads per TG, wave32. The full UT and Graph run saves
22 actual PS JIT variants. None references s104/s105. All inspected activation
loads use DEV, and all output stores use DEV + HT. Actual metadata has
`num_ctas=1`. This does not verify the common cluster of a future fused launch.
All selected PS paths have zero scratch spill. Resource counts do not prove
simultaneous residency.

A two-line weight-load trial (`e055dedfb`) changed A03 weights to ordinary
loads. It passed 108 UTs but was slower. At c16 its hot median changed from
1.569852 us native to 1.939973 us PS; rotating changed from 1.913189 to
2.327549 us. Actual A03 B1 VGPR count rose from 61 to 74, and static
instruction count rose from 342 to 361. Commit `9a294c8c1` reverts that trial.
Its tree is identical to the earlier measured `892f22b4b`. The final tree was
then tested again in full. No data from the slower trial is hidden.

The user confirmed c16/c64 TP4 + DSpark5 and c128/c256 EP4 + DP attention,
with native as the baseline. Source-derived full verify rows are 96/384/192/384;
A09 widths are 2048/2048/8192/8192. DP rows assume a balanced batch. A03 uses
shared storage with Q1280/KV512 views and row stride 1792. The c256 A03 case
reuses the c64 measurement. These are isolated op measurements, not actual
four-GPU model traces. See `model-shape-provenance.json` for source locations.

The final exclusive window was 2026-11-19 03:05:22 through 03:06:09 UTC.
All 22 unique cases have 15 alternating native/PS trials, one shared input
pool per case and the same Graph method. Profiling was off during timing.
The shared GPU lock and a five-second idle gate were used. No foreign GPU
PID was seen in the 100 ms samples for any final timing case. Sampling cannot
exclude a shorter task between samples.

Final measurements are below. Values are median / P10 / P90 in us.
PS/native greater than one means slower. Use the stated noise rule; an
inconclusive row is not proof of equivalence.

| Profile | Op | Input storage | Cache | Native median / P10 / P90 (us) | PS median / P10 / P90 (us) | PS/native | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| c128-verify-proxy | a03 | [192, 1792] | hot | 1.630421 / 1.609686 / 1.672318 | 1.813660 / 1.804882 / 1.832563 | 1.112387 | regression |
| c128-verify-proxy | a03 | [192, 1792] | rotating | 2.000277 / 1.992518 / 2.010591 | 2.108283 / 2.090597 / 2.122904 | 1.053996 | regression |
| c128-verify-proxy | a09 | [192, 8192] | hot | 1.771914 / 1.765367 / 1.806087 | 2.450433 / 2.435325 / 2.469004 | 1.382930 | regression |
| c128-verify-proxy | a09 | [192, 8192] | rotating | 2.543625 / 2.539411 / 2.553766 | 3.653580 / 3.650046 / 3.657656 | 1.436367 | regression |
| c128-verify-proxy | a09-qat | [192, 8192] | hot | 1.525492 / 1.524571 / 1.536172 | 1.877733 / 1.868006 / 1.883597 | 1.230903 | regression |
| c128-verify-proxy | a09-qat | [192, 8192] | rotating | 1.960629 / 1.959303 / 1.962539 | 2.558295 / 2.557087 / 2.560527 | 1.304834 | regression |
| c16-verify-proxy | a03 | [96, 1792] | hot | 1.566726 / 1.557397 / 1.579537 | 1.679187 / 1.668888 / 1.693446 | 1.071781 | regression |
| c16-verify-proxy | a03 | [96, 1792] | rotating | 1.913230 / 1.909461 / 1.918965 | 1.967319 / 1.957823 / 1.980313 | 1.028271 | regression |
| c16-verify-proxy | a09 | [96, 2048] | hot | 1.398305 / 1.373869 / 1.444611 | 1.422293 / 1.406815 / 1.479777 | 1.017155 | inconclusive |
| c16-verify-proxy | a09 | [96, 2048] | rotating | 1.701877 / 1.694562 / 1.724261 | 1.694866 / 1.672302 / 1.723398 | 0.995881 | inconclusive |
| c16-verify-proxy | a09-qat | [96, 2048] | hot | 1.247919 / 1.243472 / 1.255294 | 1.226547 / 1.221013 / 1.241018 | 0.982874 | improvement |
| c16-verify-proxy | a09-qat | [96, 2048] | rotating | 1.606013 / 1.605191 / 1.607560 | 1.596490 / 1.590440 / 1.598033 | 0.994071 | improvement |
| c256-verify-proxy | a09 | [384, 8192] | hot | 2.778085 / 2.697536 / 2.894113 | 3.291468 / 3.237629 / 3.356923 | 1.184797 | regression |
| c256-verify-proxy | a09 | [384, 8192] | rotating | 3.645972 / 3.632150 / 3.697716 | 5.542000 / 5.355763 / 5.572408 | 1.520033 | regression |
| c256-verify-proxy | a09-qat | [384, 8192] | hot | 2.083562 / 2.063756 / 2.084862 | 2.630134 / 2.614614 / 2.637116 | 1.262326 | regression |
| c256-verify-proxy | a09-qat | [384, 8192] | rotating | 2.708276 / 2.705739 / 2.709637 | 3.516237 / 3.514233 / 3.517834 | 1.298331 | regression |
| c64-verify-proxy | a03 | [384, 1792] | hot | 1.784493 / 1.777990 / 1.819571 | 2.605750 / 2.589721 / 2.634008 | 1.460219 | regression |
| c64-verify-proxy | a03 | [384, 1792] | rotating | 2.138024 / 2.120752 / 2.146905 | 3.030929 / 3.027107 / 3.033556 | 1.417631 | regression |
| c64-verify-proxy | a09 | [384, 2048] | hot | 1.594527 / 1.590420 / 1.600424 | 1.657133 / 1.651747 / 1.672075 | 1.039263 | regression |
| c64-verify-proxy | a09 | [384, 2048] | rotating | 2.400931 / 2.040275 / 2.448104 | 2.682670 / 2.242688 / 2.733188 | 1.117346 | inconclusive |
| c64-verify-proxy | a09-qat | [384, 2048] | hot | 1.374938 / 1.373812 / 1.377627 | 1.362607 / 1.360615 / 1.366206 | 0.991032 | improvement |
| c64-verify-proxy | a09-qat | [384, 2048] | rotating | 1.735767 / 1.733683 / 1.736753 | 1.774638 / 1.773583 / 1.775772 | 1.022394 | regression |

There are 16 regressions, three inconclusive results and three improvements.
c16 A03 regresses by 7.18% hot and 2.83% rotating. c16 A09 FP8 is inconclusive;
c16 QAT improves. Secondary gains do not offset the primary A03 failure.

At c16, A03 has 96 original tasks and A09 has 192. These are B1 paths; the
slowdown cannot be explained as reducing thousands of active TGs to 256.
Larger shapes add serial task batches and masked tail work. Cyclic A09 batches
also place consecutive per-TG slots 512 KiB apart in BF16 input memory.
DEV/HT, address work, batch size and register liveness can all affect latency.
No cache or wave-residency counter measurement assigns a precise contribution
to each factor. Do not state a measured TG-per-WGP occupancy from launch count.

Evidence: `/home/weizhu15/workspace/mono-criteria-20261119`.
See `REPORT.zh.md`, `final/performance-table.md`,
`final/timing-process-audit.json`, `final/all-real-jit-audit.json`,
`final/launch-current-trace/`, `final/logs/` and `jit-cache/`.
The first run is at the directory root. The rejected weight trial is in
`round2/`. All raw trials, JIT artifacts, failure logs and diffs are retained.

Still unverified: the actual main-kernel call/return and register preservation,
idle-TG joins, flag ordering, cross-TG producer/consumer visibility, common
fused-launch cluster, simultaneous residency, and the TP4/EP4 model path.
Standalone Graph replay is not join validation. The machine has one visible
GPU. The op timing is valid evidence, but it does not certify the model path.

## Adaptive PS result: 2026-11-18

Tested code: `96826c2c50f69d50484d8e64a250d7793268ce9d`. This result supersedes the fixed-four-task
experiment below. It does not claim a universal performance win.

A03 counts rows. It selects one task through 767 rows and four tasks above
that bound. The two-row main batch regressed in measured cases and is not
selected. A09 counts `ceil(numel/1024)`: one task through 256, two tasks
through 512, and four tasks above 512. Thus A09 width-2048 rows 96/257/513
select 1/4/4. The two-task path balances one/two-task workers. The four-task
path keeps the direct cyclic loop; splitting its tail into smaller batches
did not give a general win. Configs use the existing JSON config loaders.

All 89 GPU tests passed with zero skips. All original 52 tests remain, with
37 new threshold, tail and Graph cases. All 12 frozen native hashes and
20 offline compile variants passed. Native source and byte assertions did
not change. The final Graph trace recorded 336 PS dispatches across eight
kernel names. All had grid `(32768,1,1)` and workgroup `(128,1,1)`.
Actual four-task ISA instruction sequences match the prior PS4 kernels.
All selected PS paths have zero scratch spill.

Against old single-task PS, the 108 four-provider cases have 77 improvements,
31 inconclusive results and no regressions under the stated rule. Inconclusive
does not prove equivalence. Four cases still regress against fixed PS4.

The original paired matrix has 18 cases. A separate four-provider matrix has
108 cases, including threshold boundaries and larger rows. Each selected case
has 15 trials and no foreign GPU PID observed by 100 ms sampling. Timing uses
the shared GPU lock, common input pools and graph timing without profiling.
Values below are median / P10 / P90 in microseconds.

| Op | Rows | Cache | Native median / P10 / P90 (us) | Adaptive median / P10 / P90 (us) | Adaptive/native | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| a03 | 96 | hot | 1.577154 / 1.565078 / 1.594461 | 1.598902 / 1.566974 / 1.611840 | 1.013790 | inconclusive |
| a03 | 96 | rotating | 1.919952 / 1.915746 / 1.925917 | 1.915368 / 1.913960 / 1.923626 | 0.997613 | inconclusive |
| a03 | 257 | hot | 1.692482 / 1.673074 / 1.709228 | 2.177104 / 1.895471 / 2.196804 | 1.286338 | regression |
| a03 | 257 | rotating | 2.034134 / 2.023184 / 2.046160 | 2.299303 / 2.244865 / 2.446154 | 1.130360 | regression |
| a03 | 513 | hot | 1.817402 / 1.810269 / 1.835534 | 2.592026 / 2.585223 / 2.606715 | 1.426226 | regression |
| a03 | 513 | rotating | 2.242839 / 2.235572 / 2.246908 | 3.085473 / 3.058017 / 3.127181 | 1.375700 | regression |
| a09 | 96 | hot | 1.352658 / 1.327280 / 1.358627 | 1.369643 / 1.367575 / 1.371885 | 1.012557 | inconclusive |
| a09 | 96 | rotating | 1.686201 / 1.675495 / 1.697029 | 1.670057 / 1.655876 / 1.683546 | 0.990426 | inconclusive |
| a09 | 257 | hot | 1.498188 / 1.490096 / 1.509130 | 1.327011 / 1.323278 / 1.328867 | 0.885744 | improvement |
| a09 | 257 | rotating | 2.042094 / 2.034590 / 2.092444 | 1.837972 / 1.825438 / 1.844247 | 0.900043 | improvement |
| a09 | 513 | hot | 1.636515 / 1.630898 / 1.643407 | 1.612555 / 1.604425 / 1.620328 | 0.985359 | improvement |
| a09 | 513 | rotating | 2.737604 / 2.733202 / 2.743499 | 2.007082 / 1.996708 / 2.020106 | 0.733153 | improvement |
| a09-qat | 96 | hot | 1.250873 / 1.245912 / 1.252150 | 1.227627 / 1.226595 / 1.228271 | 0.981416 | improvement |
| a09-qat | 96 | rotating | 1.600152 / 1.598866 / 1.604137 | 1.579173 / 1.577714 / 1.580304 | 0.986889 | improvement |
| a09-qat | 257 | hot | 1.307610 / 1.305376 / 1.308629 | 1.332011 / 1.329461 / 1.334294 | 1.018660 | regression |
| a09-qat | 257 | rotating | 1.691141 / 1.689636 / 1.692849 | 1.751627 / 1.750559 / 1.754123 | 1.035767 | regression |
| a09-qat | 513 | hot | 1.419213 / 1.418548 / 1.424220 | 1.721550 / 1.686713 / 1.747807 | 1.213031 | regression |
| a09-qat | 513 | rotating | 1.794043 / 1.792317 / 1.795352 | 1.856036 / 1.849132 / 1.859300 | 1.034555 | regression |

Some cases still regress against native, so standalone performance acceptance
does not pass. Task-count dispatch alone cannot promise zero regression for
every valid shape. Other widths have correctness coverage, not a broad
performance guarantee. Full mono ABI, DEV/HT, reserved-register, joins and
model integration remain outside this change.

Raw evidence: `/home/weizhu15/workspace/mono-adaptive-20261118`. See `REPORT.zh.md`,
`performance-four-way-table.md`, `performance-accepted.json`,
`performance-selections.json`, `launch-final-gate.json`, `jit-final-review.json`
and `logs/`. Earlier strategies and their rejected results remain in that
directory; they are not mixed into the final timing matrix.

## Four-task PS experiment: 2026-11-18

This section records the current user-requested four-task experiment. The
sections below record the earlier one-task implementation and its handoff.
The older static resource limits describe that earlier selection. The new
experiment deliberately increases per-wave work and does not meet those old
VGPR/code-size limits. It is not a successful performance acceptance.

Tested code: `25c1092044557ab3e0c5f90374cb2e4b06cc6142`. Baseline PS1: `6f85fc6d8dbc4c01cf4d40a6e48f1990c7952056`.
A03 handles four adjacent rows per TG iteration. A09 handles four adjacent
1024-element tasks (two rows at width 2048). Both keep 256 TGs and 128 threads
per TG. The actual grid is one-dimensional, with work-item grid `(32768,1,1)`.

All 52 GPU tests passed with zero skips. These retain all original 45 cases
and add seven A03 cases at the new 1024-row loop boundary. Frozen native hashes
and all 12 compile variants passed. The first A03 attempt failed six byte
checks because its automatic reduction layout changed. Explicit Gluon layouts
fixed these cases; no native source or tolerance changed. Non-PS A03 is unchanged.

Actual Graph tracing recorded 48 dispatches for each of A03 PS, A09 FP8 PS and
A09 QAT PS. All used 256 TGs x 128 threads. Actual ISA issues four A09 loads
before the first load-result wait. A03 issues four rows of Q/K loads before
that wait. PS4 VGPR/shared/scratch are A03 168/64/0, FP8 76/0/0 and QAT 60/0/0.
These resource fields do not measure residency.

Timing used the shared GPU lock and idle checks, without profiling. Each
selected case has 15 trials and no foreign GPU PID in its 100 ms process
samples. Two contaminated paired cases were discarded and repeated in full.
Native/PS4 use one input pool and alternating order. Values are microseconds.

| Op | Rows | Cache | Native median / P10 / P90 (us) | PS4 median / P10 / P90 (us) | PS4/native | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| a03 | 96 | hot | 1.556748 / 1.553161 / 1.562145 | 2.892640 / 2.878984 / 2.896413 | 1.858130 | regression |
| a03 | 96 | rotating | 1.918209 / 1.915257 / 1.920165 | 3.276872 / 3.269959 / 3.365023 | 1.708298 | regression |
| a03 | 257 | hot | 1.685472 / 1.661242 / 1.691207 | 2.963542 / 2.950575 / 2.971481 | 1.758286 | regression |
| a03 | 257 | rotating | 2.036438 / 2.026278 / 2.053536 | 3.508891 / 3.504066 / 3.515869 | 1.723053 | regression |
| a03 | 513 | hot | 1.840280 / 1.787212 / 1.846371 | 3.001283 / 2.998181 / 3.013520 | 1.630883 | regression |
| a03 | 513 | rotating | 2.242006 / 2.234880 / 2.245511 | 3.675697 / 3.672838 / 3.678854 | 1.639468 | regression |
| a09 | 96 | hot | 1.399276 / 1.392892 / 1.408441 | 1.444631 / 1.323942 / 1.454359 | 1.032414 | inconclusive |
| a09 | 96 | rotating | 1.732855 / 1.725823 / 1.741467 | 1.719751 / 1.717479 / 1.722751 | 0.992438 | inconclusive |
| a09 | 257 | hot | 1.538324 / 1.492725 / 1.540745 | 1.507490 / 1.500089 / 1.512898 | 0.979956 | inconclusive |
| a09 | 257 | rotating | 2.002499 / 1.984849 / 2.009294 | 1.739041 / 1.737788 / 1.740148 | 0.868435 | improvement |
| a09 | 513 | hot | 1.637156 / 1.631263 / 1.647279 | 1.452247 / 1.431756 / 1.456366 | 0.887055 | improvement |
| a09 | 513 | rotating | 2.707783 / 2.699110 / 2.715536 | 2.026919 / 2.015178 / 2.044917 | 0.748553 | improvement |
| a09-qat | 96 | hot | 1.257856 / 1.255068 / 1.258948 | 1.302819 / 1.301787 / 1.307620 | 1.035746 | regression |
| a09-qat | 96 | rotating | 1.614831 / 1.613323 / 1.616599 | 1.679771 / 1.678194 / 1.681111 | 1.040215 | regression |
| a09-qat | 257 | hot | 1.314621 / 1.313751 / 1.316103 | 1.341153 / 1.339088 / 1.343135 | 1.020182 | regression |
| a09-qat | 257 | rotating | 1.692847 / 1.691781 / 1.693656 | 1.754295 / 1.753258 / 1.756638 | 1.036299 | regression |
| a09-qat | 513 | hot | 1.427337 / 1.425034 / 1.431802 | 1.728108 / 1.712330 / 1.747755 | 1.210722 | regression |
| a09-qat | 513 | rotating | 1.808997 / 1.805118 / 1.811625 | 2.019332 / 1.999071 / 2.059459 | 1.116272 | regression |

A03 regresses. A09 FP8 improves some cases but does not pass all six cases.
A09 QAT remains slower than native. The separate three-provider comparison
shows that four-task PS helps A09 at 257/513 rows relative to old PS. That gain
does not override a native regression. A03 gains no benefit from this mapping.
For A03, useful TG counts at 96/257/513 rows fall to 24/65/129; registers and
per-TG work increase. A09 issues more independent loads and reduces loop trips.
PS4 ATT, per-WGP residency and exact residual cycle attribution remain untested.

Raw evidence and full before/after statistics: `/home/weizhu15/workspace/mono-ps4-20261118`.
See `REPORT.zh.md`, `performance-accepted.json`, `performance-selections.json`,
`performance-load-audit.json`, `trace-jit.jsonl`, `jit-review.json`, and `logs/`.
No full mono ABI, DEV/HT, reserved-register, join or model integration was added.

Use this guide only after the external runner's offline gate and Claude's
independent local review. Neither certifies GPU correctness or performance.
The [review TODO](mono-persistent-review-todo.md) records the selected exact
power-of-two reciprocal change, its static cost and the unresolved request cost.

## Follow-up: A03 timing and A09 diagnosis

The user requested A03 retesting and further A09 analysis after the first
acceptance. All GPU work used the same fixed delivery `5446016c` and image.
Production kernel, wrapper, benchmark, tests and native source did not change.
The diagnostic wrappers and raw evidence stay outside the checkout.

A03 passed all 20 selected tests again, with zero skips. Its Graph trace passed;
all 24 PS dispatches used grid `(32768,1,1)` and workgroup `(128,1,1)`.
The original benchmark then ran all six A03 cases with 15 alternating trials.
The 101 process samples during the case windows saw only this benchmark.

| Rows | Cache | Native median / P10 / P90 (us) | PS median / P10 / P90 (us) | PS/native | Decision |
| ---: | --- | ---: | ---: | ---: | --- |
| 96 | hot | 1.564855 / 1.554226 / 1.589715 | 1.585209 / 1.567499 / 1.604769 | 1.013007 | inconclusive |
| 96 | rotating | 1.919946 / 1.917698 / 1.925404 | 1.916001 / 1.912472 / 1.919784 | 0.997945 | inconclusive |
| 257 | hot | 1.667741 / 1.656719 / 1.701918 | 1.920399 / 1.917161 / 1.967340 | 1.151497 | regression |
| 257 | rotating | 2.043015 / 2.025434 / 2.046702 | 2.349874 / 2.278817 / 2.467875 | 1.150199 | regression |
| 513 | hot | 1.821483 / 1.816528 / 1.851258 | 2.701228 / 2.663911 / 2.783161 | 1.482983 | regression |
| 513 | rotating | 2.252620 / 2.244447 / 2.263607 | 3.190436 / 3.169208 / 3.216171 | 1.416322 | regression |

A03 performance fails: four regressions and two inconclusive rows. These are
current-delivery measurements, not the historical two-dimensional-grid data.

A09 diagnostics kept the task math fixed. The external P256 replica has
instruction-identical ISA to selected P256. P512, P1024 and P1026 retain the
same instruction sequence and resources; only four stride immediates change.
At `[513,2048]`, all cases below had no observed concurrent GPU process.
These values are median microseconds from 15 trials without profiling.

| Diagnostic | FP8 hot | QAT hot | FP8 rotating | QAT rotating |
| --- | ---: | ---: | ---: | ---: |
| Frozen native | 1.666988 | 1.420895 | 2.766726 | 1.778635 |
| Selected non-PS | 1.628382 | 1.351147 | 2.778071 | 1.765415 |
| P256 replica | 3.046068 | 2.559087 | 3.503421 | 3.186402 |
| Fixed descriptor, P256 | 2.918942 | 2.560750 | 3.363139 | 3.181355 |
| Same loop body, P512 | 2.028084 | 1.828533 | 3.052226 | 2.261622 |
| Same loop body, P1024 | 1.674411 | 1.528544 | 2.588580 | 1.791406 |
| Same loop body, P1026 | 1.649970 | 1.359507 | 2.799002 | 1.777249 |

Raw JSON has full precision and P10/P90. Removing descriptor reuse/XACK waits
at fixed P256 barely changes QAT and only partly helps FP8. Thus those waits
are not the main cause. The fixed-descriptor diagnostic also changes address
arithmetic and is limited to inputs below 2 GiB; it is not a production fix.

An actual QAT ATT trace at SE0, target CU/WGP 1, SIMD0 shows four overlapping
waves for native, one wave for P256, and four for P1026. In this sample, native
and P1026 issue four loads across four wave contexts. P256 issues four loads
serially in one context. Its wave lasts 3655 GFX cycles, with 1682 cycles at
load-result waits and 233 at XACK waits. This is local trace evidence, not a
whole-GPU occupancy percentage or a substitute for benchmark timing.
Both the timed controls and trace identify the main mechanism: the current
P256 loop exposes fewer independent waves and does not pipeline the next tile's
load ahead of the current tile's work. Hardware workgroup scheduling can overlap
independent wave contexts; this software loop cannot provide the same overlap.
Equal task counts do not imply equal latency hiding. Exact cache-level and
whole-kernel cycle attribution remain open.

Tail work cannot explain the clean boundary regression. At exactly 512 rows, every
P256 worker has four tasks, yet QAT hot is 2.392025 us versus native 1.419148 us.
There is already a large regression without a fifth-iteration tail.

The external diagnostic wrappers passed 64 strict inherited assertion cases,
plus four exact-boundary cases, including special BF16 values and Graph replay.
One of 12 main diagnostic cases and four of eight boundary cases overlapped
other GPU jobs. They are marked unusable; no favorable samples were selected
from them. SQ_WAVES matched 1024/4104 for P256/native at 513 rows. Four other
SQ instruction/wait/cycle counters returned zero on all variants and are unusable.
The final ATT sample had no other GPU process in adjacent 100 ms observations.

Evidence root: `/home/weizhu15/workspace/mono-followup-5446016c-20261118/`.
See `REPORT.zh.md`, `a03-performance.json`, `diagnostic-performance.json`,
`per-case-load-audit.json`, `review/worker-count-isa.diff`, and
`diagnostic-att-clean-summary.json`. `att-wave-timeline.svg` plots the actual
wave trace. No production fix or upstream push was made. The original A09
acceptance result remains failed.

## MI400 result: 2026-11-18

Delivery `5446016c426910d96e2b13996586dfe875863da0` was tested in a clean,
independent checkout on `heliosr-1b114-a07-3`. The machine had recovered from
the earlier driver block. No kernel, wrapper, test, benchmark or frozen native
source changed. All 12 frozen symbol hashes and 12 offline builds passed.

The numerical suite passed once without profiling: **45 passed, zero skips**.
This includes 20 A03 cases and 25 A09 cases. Native/PS bytes, input/weight
preservation, guards, special BF16 fixtures and all Graph checks passed.
A03 includes shared `[96,1792]` storage with Q1280/KV512 views and row stride
1792. A09 includes `[96,2048]`, `[513,2048]` and `[513,2080]`.

Both A09 Graph nodes then passed under rocprofv3. All 96 PS dispatches
(48 FP8, 48 QAT) used grid `(32768,1,1)` and workgroup `(128,1,1)`:
**256 TGs x 128 threads**. Actual wrapper JIT IR, ISA, code objects and metadata
were saved. Actual PS FP8 is 808 B / 21 VGPR; QAT is 844 B / 20 VGPR.
Both use 68 SGPR, shared=0 and scratch=0. These differ from the earlier local
binder numbers below, but pass the fixed resource gates. Vector widths, masks,
loop dependencies and FP modes passed review. No general division, LDS,
scratch or memory-mask EXEC branch was found. Non-PS has no task loop.
The raw profiler VGPR/SGPR fields are retained separately from compiler counts.

The one required timing matrix ran without a profiler at 12:01:37--12:01:52 UTC.
Each row has 15 alternating native/PS trials on the same input pool. Each trial
uses 10 replays of a 1000-kernel Graph. Hot pool length is 1; rotating pool
lengths are 523, 196 and 98 for 96, 257 and 513 rows. Each sample is mean
latency per launch. Decisions use the rules in section 3, before rounding.

| Path | Rows | Input reuse | Native median / P10 / P90 (us) | PS median / P10 / P90 (us) | PS/native | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| FP8 | 96 | hot | 1.386007 / 1.374745 / 1.414506 | 1.403738 / 1.387628 / 1.428432 | 1.012792 | inconclusive |
| FP8 | 96 | rotating | 1.717423 / 1.692109 / 1.730208 | 1.688139 / 1.679708 / 1.709433 | 0.982949 | inconclusive |
| FP8 | 257 | hot | 1.480265 / 1.474389 / 1.548339 | 2.034931 / 1.946374 / 2.066170 | 1.374707 | regression |
| FP8 | 257 | rotating | 2.087570 / 2.077553 / 2.207312 | 2.492392 / 2.398856 / 2.518862 | 1.193920 | regression |
| FP8 | 513 | hot | 1.663527 / 1.658817 / 1.695893 | 3.053400 / 2.986177 / 3.085353 | 1.835498 | regression |
| FP8 | 513 | rotating | 2.738817 / 2.608372 / 2.744959 | 3.523660 / 3.503407 / 3.546427 | 1.286563 | regression |
| QAT | 96 | hot | 1.252347 / 1.237009 / 1.256956 | 1.227174 / 1.225645 / 1.228566 | 0.979899 | improvement |
| QAT | 96 | rotating | 1.603641 / 1.601850 / 1.604787 | 1.581236 / 1.578911 / 1.582523 | 0.986028 | improvement |
| QAT | 257 | hot | 1.315974 / 1.312891 / 1.571942 | 1.714587 / 1.689177 / 1.753119 | 1.302903 | regression |
| QAT | 257 | rotating | 1.692194 / 1.689980 / 1.694664 | 2.343633 / 2.328332 / 2.364742 | 1.384967 | regression |
| QAT | 513 | hot | 1.421809 / 1.418243 / 1.425017 | 2.465512 / 2.430558 / 2.489546 | 1.734067 | regression |
| QAT | 513 | rotating | 1.793473 / 1.792476 / 1.795698 | 3.187413 / 3.182001 / 3.196832 | 1.777229 | regression |

**FP8 fails:** four regressions and two inconclusive rows.
**QAT fails:** four regressions and two improvements. Neither path passes all
six rows. No automatic rerun, tuning or ablation matrix followed that run.
The later user-requested follow-up above adds current A03 timing and A09
diagnostics. Historical A03 timings do not certify either run. No upstream push
was made because the requested success gate failed.

The GPU was ordinal 0, gfx1250/wave32, with 256 CUs. The image digest was
`sha256:6a40615d965ec6088217e5682a866875e2c218365c7f135ecb83c4907d6bf2dd`.
It used PyTorch `2.14.0+rocm10.3.0a20261103-bkc.20261105`, HIP `7.18.26435`,
Triton `3.9.0+gitb7196d97` and amdgpu `7.1.0.31300009`.
Packages, clocks and power settings were preserved. Performance mode was AUTO.
The 100 ms process log observed only the benchmark GPU process during timing.
Seven metric samples at about 2.177 s intervals recorded GFX 2213--2240 MHz,
HBM 1900 MHz and socket power 1038--1177 W. No sampled thermal or power
throttle was active. These samples cannot exclude shorter unobserved events.

The measured regression grows when P256 workers must repeat tasks. There are
192, 514 and 1026 tiles at the tested row counts. PS workers execute up to
one, three and five serial iterations; native dispatches one TG per tile.
This is a concrete scheduling and loop-cost difference. The timing pattern is
consistent with that cost, but does not isolate its cycle contribution.
Cache traffic, scale-store request combining, residency and runtime retry state
remain unmeasured. Smaller code and fewer VGPRs do not establish a speedup.

Full logs, JSON/XML, maps, JIT artifacts and private review are on the runner at
`/home/weizhu15/workspace/mono-acceptance-5446016c-resumed/`.
Start with `REPORT.zh.md`, `performance.json`, `performance-decisions.json`,
`jit-launch-gate.json`, `load-audit.json` and `logs/`. Memory-sanitizer coverage
and full mono ABI/DEV/HT/reserved-register/join/model work remain unverified.

## Source and local status

Selected code commit is `0d54cbf3e7b16ac7563c34df6e5d744945f6a495`.
This is the exact kernel and test source approved by Claude Opus 5.5 in local
run `20261118-155450-63f32955`. The fixed outer acceptance passed, exit 0.
The published delivery adds only guide updates after that code commit. Use
`Wanzizhu/aiter`, branch `mono-dev`, and the exact delivery SHA supplied in the
handoff. Read this guide from that checkout and keep the delivery checkout for
all tests; do not reset to an older test commit or apply historical patches.
Record HEAD, status, dirty diff and source hashes. Check the two selected-source
hashes below and the frozen native manifest before execution.

If using the offline patch bundle instead, apply its single patch to a clean
independent checkout at public base
`a8a913883d7be9837c2f46f2e92de8fc701bac26`. Verify that bundle's manifest;
HEAD remains at the public base after patch application. The offline bundle and
the published delivery have identical reviewed kernel/test bytes; later guide
updates can differ. Preserve framework packages and unrelated tasks.
The isolated review clone omitted one unrelated gitlink for runner support;
both delivery methods preserve production submodule entries and declarations.

| Reference | Fixed source |
| --- | --- |
| Public comparison base | ROCm/aiter `a8a913883d7be9837c2f46f2e92de8fc701bac26` |
| A09 original native | ATOM `10b99a085ba46e5b3a125f0114cec7c71e401552` |
| A03, unchanged | `7108dbec079595026f76b7b1b8e082fe1669796d` |
| Frozen excerpts / hashes | `op_tests/triton_tests/utils/mono_native.py` and `mono_native_manifest.json` |

Selected code SHA256:

```text
0393295fc04b699224174e77ed5ede74f28d46ea588afed986f48cd87c2a2849  aiter/ops/triton/_gluon_kernels/gfx1250/quant/fp8_group32.py
6e64b2b37adc63e55a9d5306ae5c5d9a6ca9d5b18efa436374490a5908c71e60  op_tests/triton_tests/quant/test_mono_persistent.py
```

The worker completed 12 offline native/candidate builds and checked all 12
frozen symbol hashes. The collection-only CPU import harness found the same
45 node IDs. All original assertions remain. Black checks on individual files,
Ruff and diff whitespace checks passed. The worker also built 22 real-binder
variants: 12 selected, six native and four previous-code comparisons. These
cover FP8/QAT, PS/non-PS and `[1,32]`, `[513,2048]`, `[513,2080]` tails. QAT
uses the wrapper's actual `S=None` specialization.

The unchanged external CPU checker was copied into private workspace artifacts
and run there. It checked 65,536 BF16 bit patterns against 165 scale codes:
10,813,440 pairs, with zero post-clamp FP8 or QAT byte mismatches. Pre-clamp
bits match except that NaNs are compared as both NaN. This checks CPU algebra
and reference semantics; it is not GPU numerical proof. At the local handoff,
the new GPU fixtures had not run. They passed in the hardware result above.
Full worker logs, IR/ISA, code objects, source manifest, patch and dependency
review are in private `build/a09-local/`.

The outer runner's fixed command remains:

```text
/home/weizhu15/workspace/aiter-mono-tools/bin/python /home/weizhu15/workspace/mono-a09-finalize-20261118/acceptance.py
```

It checks 12 builds, frozen hashes, Black/Ruff, the exact 45 node IDs, four real
binder variants, exhaustive CPU arithmetic and ISA/resource selection gates.
The outer runner executed this command successfully. Claude independently
reviewed the recorded result and the exact candidate, and returned `approve`
with no blockers. Its two advisory items are handled in this guide: special-value
failure triage below and line wrapping. No reviewed kernel or test bytes changed
after approval. This approval covers local readiness only. The separate
hardware result above records GPU correctness, actual JIT/launch and the
failed performance gate.
Never use the CPU collection harness to run numerical tests or bypass the GPU
gate below.

The [historical report](mono-results/2026-11-18-gfx1250/README.md) covers
`d5930aff35d67fa0a2e74ba425909747f3ccef9f`, before this implementation. Its
36 passes, eight regressions and four inconclusive rows are historical only.

## 1. Prepare the supplied checkout

Use the machine's existing compatible gfx1250 ROCm environment. Set
MONO_RESULTS to the authorized artifact directory before these commands.
Preserve its packages, other tasks, clocks and power settings. Stop on a failed
command and retain the full error. Do not repair the environment by installing
frameworks as part of this handoff.

```bash
set -euo pipefail
: "${MONO_RESULTS:?Set the authorized artifact directory}"
mkdir -p "$MONO_RESULTS"
export AITER_TRITON_ONLY=1
export TRITON_DEFAULT_FP_FUSION=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD > "$MONO_RESULTS/commit.txt"
git status --porcelain > "$MONO_RESULTS/git-status.txt"
git diff --binary > "$MONO_RESULTS/source.diff"
sha256sum aiter/ops/triton/_gluon_kernels/gfx1250/quant/fp8_group32.py \
  aiter/ops/triton/quant/fp8_group32.py \
  op_tests/triton_tests/quant/test_mono_persistent.py \
  op_tests/op_benchmarks/triton/bench_mono_persistent.py \
  op_tests/triton_tests/utils/mono_native.py \
  op_tests/triton_tests/utils/mono_native_manifest.json \
  > "$MONO_RESULTS/source-sha256.txt"
python - <<'PY' | tee "$MONO_RESULTS/environment.txt"
import platform
import torch
import triton
from triton.runtime import driver
assert torch.version.hip is not None and torch.cuda.is_available()
target = driver.active.get_current_target()
assert target.arch in ("gfx1250", "gfx1250-strict") and target.warp_size == 32
print(platform.platform())
print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
print("target", target)
print(torch.cuda.get_device_properties(torch.cuda.current_device()))
PY
```

Check the selected hashes above and the frozen native manifest. For a patch
bundle, also check its delivery manifest. Record image digest,
host, selected GPU UUID/ordinal, driver, clock/power settings and other GPU
workloads. Retain load/clock observations with timestamps and sampling interval.
Runtime retry state is unknown unless the machine provides evidence for it.
Do not infer it from MODE.REPLAY_MODE in ISA.

## 2. Correctness and actual launch evidence first

The offline compiler checks frozen symbols before compiling. Its metadata and
ISA do not replace the real GPU wrapper specializations saved by pytest.
Run all 45 numerical tests once without a profiler. Then trace only the two
A09 Graph nodes. A profiler failure must not discard the numerical result.
Profiler durations are not timing results.

```bash
python -m op_tests.compile_mono_persistent --output "$MONO_RESULTS/compile" \
  2>&1 | tee "$MONO_RESULTS/compile.log"
export TRITON_CACHE_DIR="$MONO_RESULTS/jit-cache"
export TRITON_DUMP_DIR="$MONO_RESULTS/jit-dump"
export TRITON_KERNEL_DUMP=1
python -m pytest \
  op_tests/triton_tests/quant/test_mono_persistent.py -ra -v \
  --junitxml="$MONO_RESULTS/correctness.xml" \
  2>&1 | tee "$MONO_RESULTS/correctness.log"
python - "$MONO_RESULTS/correctness.xml" <<'PYXML'
import sys
import xml.etree.ElementTree as ET
cases = list(ET.parse(sys.argv[1]).iter("testcase"))
assert len(cases) == 45, f"Expected 45 tests, found {len(cases)}"
assert all(c.find("skipped") is None and c.find("failure") is None
           and c.find("error") is None for c in cases), "Correctness gate failed"
PYXML
```

After the numerical gate passes, collect launch evidence. The selected nodes
cover FP8 and QAT at 257 and 513 rows. Reuse the JIT cache from the full run.

```bash
if command -v rocprofv3 >/dev/null 2>&1; then
  if rocprofv3 --kernel-trace --output-format csv \
    --output-directory "$MONO_RESULTS/launch-trace" -- \
    python -m pytest \
    'op_tests/triton_tests/quant/test_mono_persistent.py::test_graph_replay_changes_inputs[a09]' \
    'op_tests/triton_tests/quant/test_mono_persistent.py::test_graph_replay_changes_inputs[a09-five-iterations]' \
    -ra -v --junitxml="$MONO_RESULTS/launch-correctness.xml" \
    > "$MONO_RESULTS/launch.log" 2>&1; then
    printf 'Trace command completed; inspect grid and workgroup before timing.\n'
  else
    printf 'Launch gate pending; keep correctness.xml and inspect launch.log.\n'
    exit 1
  fi
else
  printf 'Launch gate pending: rocprofv3 is unavailable.\n'
  exit 1
fi
```

Coverage retains all existing nodes and assertions. A09 retains native and
independent finite-value FP8/E8M0/QAT oracles, XOR-prefilled outputs, guards and
unchanged inputs. The existing scale-boundary node also builds integer bits
on GPU for `[65536,32]` (all BF16 patterns, 4 MiB) and `[8193,32]` (mixed groups,
512 KiB plus 64 B). The fixtures cover normal/subnormal values, signed zeros,
finite extremes, Inf and signed quiet/signalling NaNs. Mixed groups rotate
values through all lane slices and waves; finite groups force underflow with
large maxima. Both fixtures compare PS and non-PS FP8/E8M0/QAT bytes against
frozen native and check input/output guards. Nonfinite data never enters the
finite-only oracle. There is no relaxed NaN or byte tolerance. Shapes also
include `[513,2048]` with five iterations and `[513,2080]` with one valid group
in the last task. Scale ownership spans all four waves.
The two A09 Graph nodes check both FP8 and QAT after each of 20 input changes.
A03 tests and its existing FP64 K tolerance remain unchanged.

Pass requires 45 passes and zero skips, byte/guard mismatches or GPU errors.
For a failure, retain node ID, seed, input, outputs, source manifest and complete
log. Stop before timing. Do not loosen numerical checks or change frozen native.
GPU memory-sanitizer coverage remains unverified unless separately evidenced.

For the new special-value fixtures, record fixture name, row/group, raw BF16
bits, native/candidate E8M0 codes and FP8/QAT output bytes. If scales differ,
inspect the existing amax reduction, input masks and scale ownership/storage
before attributing the mismatch to the new multiply. If scales match, inspect
the arithmetic, clamp/conversion and payload storage paths. An sNaN-only failure
would motivate a reduction-order check; it is not proof of that cause. No such
GPU failure has been observed locally. Retain strict assertions and do not run
an automatic variant sweep.

Map A09 `PERSISTENT_1` dispatches to actual JIT artifacts and host calls.
Every nonempty PS launch must have 256 TGs x 128 threads, or work-item grid
`(32768,1,1)` and workgroup `(128,1,1)`. This does not prove concurrent residency.
For each changed FP8/QAT path, save actual IR, ISA, code object and metadata.
Check buffer access widths, masks, dynamic metadata.shared=0, scratch=0 and
loop dependencies. A zero fixed LDS field alone is insufficient. Non-PS has
no task loop. If launch tracing is unavailable, report the P256 runtime gate
as pending and stop before performance acceptance.

The selected real-binder FP8/QAT PS uses 21/18 VGPR and 892/876 code bytes.
Frozen native uses 49/44 VGPR and 1388/1264 bytes. The previous PS used 53/47
VGPR and 1436/1420 bytes. All use 68 SGPR. Selected PS compiler VGPRBlocks
are 1/1, versus native 3/2; these are allocation fields, not measured occupancy.
The reduction removes eight general division sequences per lane in both paths.
Scale derivation, clamp/conversion order and stored E8M0 bytes remain fixed.
The [review TODO](mono-persistent-review-todo.md) gives the arithmetic proof
and the non-PS comparison.

Expected accesses remain buffer_load_b128, FP8 buffer_store_b64 plus
buffer_store_b8, and QAT buffer_store_b128. Local IR/ISA has no general divide,
LDS, scratch or memory-mask EXEC branch. It retains a single scale writer,
global tail masks, zero-filled masked loads and safe descriptor/source reuse.
FP32 denorm mode remains 3, round mode 0, with `allow_flush_denorm=false`.
Review actual JIT for these properties before timing. A new spill, LDS copy,
mask branch, lost vector width, changed FP mode or unsafe dependency requires
review. Static selection requires FP8 code < 1388 B and VGPR <= 49, QAT code
< 1264 B and VGPR <= 44, with shared=0 and scratch=0 on both paths. These are
local readiness gates, not a runtime speed guarantee.

## 3. Time only frozen native and final PS

After correctness and launch checks pass, run this one matrix without a
profiler. QAT is required because the shared arithmetic change affects both
paths.
The benchmark now defaults to A09 FP8 and QAT; it does not include A03 unless
`--ops a03` is supplied. This handoff selects both A09 paths explicitly.

```bash
python -m op_tests.op_benchmarks.triton.bench_mono_persistent \
  --ops a09 a09-qat --rows 96 257 513 --cache both --trials 15 \
  --output "$MONO_RESULTS/performance.json" \
  2>&1 | tee "$MONO_RESULTS/performance.log"
```

This retains RotatingBufferBench, the common input pool, seeded shuffle,
alternating native/PS order and 15 trials. Each sample is mean latency per
launch over Graph replays. Compile, allocate, capture and warm up before timed
replays. Wrappers allocate outputs during capture; the providers do not share
a preallocated output pool. Hot/rotating labels input reuse, not measured cache
hits. These are standalone timings, not model TPOT.

JSON records each round's order, provider times, PS-minus-native microseconds,
start/end timestamps, all samples, median/P10/P90, pool and graph lengths,
median difference and PS/native ratio. Report each of the 12 op/rows/cache
rows separately. Use these conservative decision rules:

```text
delta = PS median - native median
noise = native P90 - native P10
regression = delta > noise and PS P10 > native P90
improvement = -delta > noise and PS P90 < native P10
otherwise = inconclusive
```

An improvement passes that row. A regression fails that row. Overlap,
equal medians or insufficient separation remain inconclusive; they do not
establish equivalence. No positive slowdown allowance is assumed. A path
passes only if all six required rows are improvements, and its correctness
and P256 gates also pass. Report FP8 and QAT status separately. A gain against
historical PS cannot offset a native regression. Return failed or inconclusive
rows with the evidence; do not automatically rerun or start another tuning or ablation
matrix.

## 4. Return the evidence and exact status

Return the recorded SHA/source hashes, dirty diff, environment, correctness
XML/log, dispatch mapping, actual JIT artifacts, offline artifacts and timing
JSON/log. Full review explanations belong in private runner artifacts. Update
these two existing guides with verified status; add no tracked result report.
Separate implementation, code/algebra review, CPU exhaustive checks, offline
ISA/resource/collection, Claude local review, GPU numerical/Graph/JIT/launch
and performance status. Lower-level scale request combining, cache traffic,
retry state and cycle attribution remain unknown without evidence. Full mono
ABI, DEV/HT, reserved registers, joins and model integration are outside this
standalone acceptance.

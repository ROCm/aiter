# GLM-5.3 IQ2R source snapshot — 2026-09-25

This snapshot records the current TP4/TP8 work in the paired draft PRs
[ATOM #2335](https://github.com/ROCm/ATOM/pull/2335) and
[AITER #5728](https://github.com/ROCm/aiter/pull/5728).
**The performance and answer-quality goals remain unmet.**

- [MXFP4 versus IQ2R results](BENCHMARK_RESULTS.md): both official workloads,
  completed bookends, and explicit missing coverage.
- [Optimization inventory](OPTIMIZATION_INVENTORY.md): attempted mechanisms,
  selected and rejected outcomes, and remaining qualifications through E196.
- [Benchmark recipe](BENCHMARK_RECIPE.json): candidate flags and matched protocol.
- [Source identity](SOURCE_SNAPSHOT.json): files and native module hashes.

## Included implementation

AITER contains TP sharding helpers and shared-expert checkpoint support,
quantize-once routing, quad-packed gates, wait3/register scheduling, direct gate
epilogues, specialized down kernels, static ballot routing, bounded task
ordering, and the TP8 dense activation/weight lookahead. The native sources
consolidate E167/E172/E181 from the measured experimental runtime. The E190
top-k candidate-count guard is also included; it was **not** in E180 r8.
Experimental dispatch remains controlled by its existing environment flags.
The unselected E194–E196 TP4 compact fragment scout is documented but is not
enabled or added to the production kernel path in this snapshot.

ATOM contains packed TP4/TP8 loading, fused shared-expert routing, prepacked
quad weights, EP placement support, shared-output integration, dispatch audits,
and the existing optional rocprofiler controls. The TP4 quad-layout acceptance
check is synchronized with the measured runtime. The fixed diagnostic capture
selection used in some experiment copies is not a serving requirement.
The official `atom.benchmarks.benchmark_serving` source is unchanged.

## Validation

- The consolidated IQ2R native module rebuilt successfully for gfx950 with the
  existing ROCm 10 SDK. Its SHA-256 is
  `b8089077684bd4c887be84b68cf62a8d427490e9354e03a8a84394c35ebdf2ef`.
- 131 focused native GPU tests passed: TP4/TP8 scheduling and exact graph
  outputs, ballot ordering, changing and invalid padding, repeated large32
  codebook reuse, and the grouped-top-k NaN regression. These use synthetic
  weights/inputs and do not replace full-model accuracy validation.
- 124 CPU tests passed, covering checkpoint metadata, exact packed slicing,
  shared overlays, GLM model style, quantization configuration, ATOM routing,
  and both TP4/TP8 quad-layout preparation paths.
- Python formatting is checked with Black and lint with Ruff on changed files.
  Formatting preserves the Python AST. Generated modules and caches are not
  committed.

The native tests rebuilt IQ2R from this tree in a private JIT directory. The
router regression used the previously rebuilt E190 module whose router source
is byte-identical to this tree. E190 separately passed 72 finite-input exact
comparisons, 900 poisoned-LDS cases, and 32 regression cases.

## Qualification still required

E180 r8 completed all 18 TP8 IQ2R official-script points with zero failed
requests and a passing native dispatch audit. Both MXFP4 arms failed at 1k/1k
C16, so the full bookend comparison is incomplete. TP4 official results and
all MXFP4 8k/1k points are still unavailable at this snapshot.

E183 normal-EOS custom answer checks completed 352 requests per TP width with
zero request failures, but only 340/352 TP4 and 339/352 TP8 checks passed.
The E191 router-guard qualification and fresh official sweeps remain separate
acceptance work. A completed benchmark request is not evidence of a correct
answer. See the result document for the partial performance comparison.

The [single-GPU optimization report](SINGLE_GPU_ANALYSIS.md) tracks E199–E326 local MoE experiments. Those microsecond timings are separate from the official serving results; dense candidate kernels remain isolated; E225 restores a previously selected frontend.

E225 restores the previously selected E167 C4/C8 static ballot frontend omitted
from the consolidated AITER source. New tests inspect actual GPU kernel names:
the old source failed both enabled C4/C8 dispatch cases; the corrected source
passes all 54 dispatch, routing, and TP8/TP4 complete-MoE checks. Dense candidate
kernels remain isolated. Existing E199 small-token local controls used the generic
frontend despite setting STATIC_BALLOT. Fresh TP8 comparison and TP4 IQ2R-only checks are complete; small-token TP4 MXFP4 comparison is withheld after fixed-tolerance atomic-output failures.

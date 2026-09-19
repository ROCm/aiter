# Standard MXFP8 GEMM

Public entry: `aiter.gemm_a8w8_mxfp8(A, B, ScaleA, ScaleB, ...)`.
Both operands use E4M3 data with one independent E8M0 scale per **32 K elements**.
No configurable scale block or block128 broadcasting is provided by this backend.

## Separate architecture contracts

| | gfx950 | gfx1250 |
|---|---|---|
| Backend | FlyDSL, tuned CSV + AOT | Existing ASM heuristic/kernelName |
| A data | Row-major, unshuffled | Existing A-preshuffle layout by default |
| B data | `(16,16)` preshuffle by default; plain B optional | Existing preshuffled B |
| Scales | Unshuffled `[M,K/32]`, `[N,K/32]` E8M0/uint8 | Existing shuffled ASM scale layout |
| Output | BF16/FP32, optional bias/out | Existing BF16-returning API |

`a_preshuffle=None` selects the native default: False on gfx950, True on gfx1250.
Explicit `a_preshuffle=True` is rejected on gfx950 rather than silently reading
shuffled A as row-major. The gfx1250 branch does not import or query the gfx950
FlyDSL tuned table, convert its scale layout, or accept gfx950 kernel names.
The original positional arguments and default gfx1250 execution are preserved.
gfx950-only additions (`out`, `bias` and plain B) are rejected
on gfx1250. The underlying scale buffers are **not interchangeable across arches**.

gfx950 example:

    import aiter
    from aiter.ops.shuffle import shuffle_weight

    y = aiter.gemm_a8w8_mxfp8(a, shuffle_weight(w), sa, sw, out=out)
    y_plain = aiter.gemm_a8w8_mxfp8(a, w, sa, sw, bpreshuffle=False)

`tgemm.mm(..., scale_a=sa, scale_b=sw, bpreshuffle=True)` also forwards native
MXFP8 through this entry. Under torch.compile, specify the B layout explicitly.

On gfx950, kernel selection and the dynamic split count both come from the
tuned CSV row. Callers do not specify `splitK` or override `kernelName`.
`get_mxfp8_config` is cached only by shape, output dtype, bias and B layout;
`kernelName` and `splitK` are configuration values, not lookup inputs.
The public `kernelName` argument exists only for gfx1250 ASM compatibility and
is rejected on gfx950. The low-level `flydsl_mxfp8_gemm(config=...)` remains
available to the tuner, while production calls use the table.

## gfx950 kernel and limits

FT/HTI pipelines, local slice-K and dynamic split-K are supported. Full-tile
16x16x128 MFMA can either read preshuffled B directly into registers or stage
it through LDS. The internal `direct_b` policy is selected by the tuned table,
not by the caller. `_bd1` / `_bd0` in kernelName and the constexpr ABI distinguish
these two binaries. HTI, plain B and MMA32 only support LDS (`direct_b=False`);
unsupported direct combinations are rejected.

Split-K reduction is also a tuning axis, `use_split_k_semaphore`:

- `False` (default): write FP32 partial slabs, then reduce once, applying bias
  and output conversion only at the end.
- `True`: use the HGEMM semaphore protocol to initialize output (including bias)
  once, atomically accumulate K partitions, and reset the synchronization state.
  There is no separate reduction kernel or FP32 partial workspace. This mode
  supports FT/HTI, including valid slice-K and direct-B combinations.

Names retain `ks1`/`ksd` for the default mode and encode `ksd_sem1` for the
semaphore mode; the actual split count remains dynamic in CSV `splitK`.
Old names/CSV rows without `_sem1` keep the partials/reduce behavior.
`split_k=1` only enumerates `use_split_k_semaphore=False`; explicitly requesting
semaphores without splitting is rejected. Production callers still select both
the reduction mode and split count through the tuned row, not API overrides.

BF16 atomics round each partition and their accumulation order can vary, so
this mode is not numerically equivalent to FP32 partials. The tuner retains
the existing `rtol=.03, atol=.1` checks (default zero error fraction), screens
the two modes separately, and rechecks successful split candidates over 16
dirty-output invocations before selection. Inaccurate candidates are rejected,
not accepted by relaxing tolerances. FP32 output uses FP32 atomics.

Positive dimensions and vector alignment are required. There is no K-tail:
tile, stage and split partitions must pass validation. HTI uses two stages,
two M waves, no slice-K and even K-tile counts per partition. Partial workspace
is limited to 4 GiB. Semaphore mode instead requires at most 256 output tiles
and whole-workgroup vector coverage for output initialization; invalid
candidates are removed before compilation. Its two 256-element int32 buffers
are stream-local and reused after reset (2 KiB total per cached stream/device).
CPU-only AOT uses tiny CPU buffers with the same dynamic pointer/layout ABI.
AITER's unrelated blockscale operators are unchanged.

## MiniMax-M3 tuning only

The source shape list is
`aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_minimax_m3.csv`.
Only its **100 unique M/N/K shapes** are reused; the original FP8 kernels,
scales and timings are not copied.

The standard MXFP8 files are:

- `model_configs/mxfp8_untuned_gemm_minimax_m3.csv`
- `model_configs/mxfp8_tuned_gemm_minimax_m3.csv`

No DSv4 or example performance rows remain. Both generic
`mxfp8_tuned_gemm.csv` and `mxfp8_untuned_gemm.csv` are removed.
Runtime/AOT default directly to the MiniMax-M3 tuned table; the tuner uses its
model-specific untuned table. Explicit multi-table merging also takes its
deduplication keys from this model shape table, without placeholder CSVs.
All new measurements use independent per-32 E8M0 scales, BF16 output, no bias
and B preshuffle; this is an operator shape sweep, not full-model inference.

    python csrc/gemm_mxfp8/gemm_mxfp8_tune.py \
      --screen-topk 4 --mp 1 --shape_grouped --warmup 3 --iters 31

The tuner defaults to these MiniMax-M3 input/output files. Override `-i/-o`
for custom experiments. `AITER_CONFIG_GEMM_MXFP8` overrides runtime/AOT lookup.
HGEMM-style tile/stage/wave axes include K tiles 128/256/512, slice-K 1/2/4,
and legal split divisors through 32. Optional graph screening retains finalists
per split/reduction/slice/B-loading regime; `mp_tuner` profiles them with three
rotating tensor sets. Both legal semaphore and partials variants are enumerated
for split counts above one; unsplit candidates are not duplicated.
`--screen-topk 0` skips screening and profiles the full valid space.

### Pruning and the B-loading axis

Yes, the search is pruned before timing:
1. Hard legality checks: MMA divisibility, vector alignment, stages, wave count,
   actual LDS usage (different for direct/LDS B), split partitions, output shape
   and workspace bounds.
2. HGEMM-derived heuristics: tile IOU relative to the best candidate, tile-grid
   size <= four times the larger of CU count and the minimum tile grid, split
   count <= ceil(2*CU/base_grid) (no split when base_grid already fills the CUs),
   and group_m=4 only for sufficiently large grids divisible by eight.
3. Non-HTI policies also limit M/N MMA repeats per wave to four.
   MXFP8 disables HGEMM's slice-K occupancy deduplication (`prune_slice_k=False`).

`direct_b=False/True` is independently enumerated for preshuffled full-tile
MMA16. A retained direct policy's matching LDS policy is kept whenever it is
also resource-legal; neither pruning nor screening merges them. Screening
reserves separate finalist slots for each B path, reduction mode and
split/slice regime. A direct-only policy is not evidence of timing preference:
the corresponding LDS policy exceeds resource limits.

The existing MiniMax-M3 table was migrated to explicit `_bd0/_bd1` names while
preserving its previous B path, selected tiles and measured timings (75 direct,
25 HTI/LDS). This is **not** a full retune of the expanded B-loading space.
The table has not been retuned with the semaphore axis either: existing rows
still select partials/reduce, and their old timings have not been relabeled as
semaphore measurements. A real tuner smoke with `--screen-topk 1` profiled both
B paths; integration
tests cover their distinct cache signatures, CPU-only AOT/run-only execution,
public table-driven calls and preservation through pruning.


## AOT and test closure

    AITER_AOT_IMPORT=1 GPU_ARCHS=gfx950 HIP_VISIBLE_DEVICES='' \
      FLYDSL_RUNTIME_CACHE_DIR=/tmp/mxfp8_cache \
      python -m aiter.aot.flydsl.gemm \
      --csv aiter/configs/model_configs/mxfp8_tuned_gemm_minimax_m3.csv

    FLYDSL_RUNTIME_CACHE_DIR=/tmp/mxfp8_cache FLYDSL_RUNTIME_RUN_ONLY=1 \
      python op_tests/test_flydsl_mxfp8.py

    pytest -q op_tests/flydsl_tests/test_mxfp8_integration.py

The standard `@benchmark` op test invokes the public `gemm_a8w8_mxfp8` entry
with dirty preallocated outputs, reports us/TFLOPS/TB/s/err, and defaults to
the new MiniMax-M3 tuned shapes. CPU-only AOT shares the layout-dynamic ABI
with runtime; tests cover new-process run-only, dynamic split reuse, actual
MXFP8 quantization, out/graph/inductor, rejected coarse scales and architecture
dispatch isolation. gfx1250 dispatch tests stub the ASM call on gfx950; they
do not claim GPU execution coverage on gfx1250.

Local semaphore validation covered old-name compatibility, separate cache
identities, hard legality bounds, tuning/screening mode preservation,
precision-failure filtering, exact BF16/FP32 bias and dirty-output checks,
two-stream graph replay/reset, and CPU AOT followed by public run-only dispatch
with multiple dynamic split counts. The standalone semaphore test file is not
included. Updating this ABI requires rebuilding the FlyDSL AOT cache before
running with `FLYDSL_RUNTIME_RUN_ONLY=1`.


## MiniMax-M3 verification

All **100/100** source shapes were independently tuned in standard MXFP8
format (A/B scale group 32), with zero observed tuning errors. The table
contains 57 distinct policies: 21 rows use split-K and 57 use slice-K
(27 with two K waves, 30 with four). The original FP8 timings were not reused.

The 181643 shape/config candidates were shape-sharded over six idle
gfx950/256-CU GPUs, keeping each shape's comparison on one GPU. All 6748
unique candidate configurations were successfully AOT-compiled. Finalists
used 31 profiler iterations and three rotating tensor sets.

A fresh cache and a single gfx950 then validated the public-entry closure:
100 selected AOT jobs succeeded; all 100 shapes passed run-only default
`aiter.gemm_a8w8_mxfp8` with CSV-selected kernel/splitK, repeated NaN-poisoned
preallocated output, and `tgemm.mm` calls. Standard op-test profiling used
101 iterations and three rotating sets; every error column is zero.
This is GEMM-only timing (including split-K reduce), not model throughput.
gfx1250 is covered by dispatch/ABI tests here, not gfx1250 GPU execution.

Representative M=32 raw op-test results (other source shape groups do not
all contain M=32):

|   m |    n |    k | dtype          | layout     | gfx    | tile      |   split_k |   k_waves |   flydsl us |   flydsl TFLOPS |   flydsl TB/s |   flydsl err |
|----:|-----:|-----:|:---------------|:-----------|:-------|:----------|----------:|----------:|------------:|----------------:|--------------:|-------------:|
|  32 | 2304 | 6144 | torch.bfloat16 | preshuffle | gfx950 | 16x32x512 |         4 |         1 |    10.6292  |         85.2339 |       1.40634 |            0 |
|  32 | 2560 | 6144 | torch.bfloat16 | preshuffle | gfx950 | 16x64x512 |         3 |         4 |    10.5819  |         95.1275 |       1.56746 |            0 |
|  32 | 6144 | 2048 | torch.bfloat16 | preshuffle | gfx950 | 16x32x512 |         1 |         4 |     6.00749 |        134.05   |       2.2367  |            0 |
|  32 | 6144 | 3072 | torch.bfloat16 | preshuffle | gfx950 | 16x32x512 |         1 |         4 |     7.39418 |        163.366  |       2.69926 |            0 |
|  32 | 6144 | 6144 | torch.bfloat16 | preshuffle | gfx950 | 16x32x512 |         1 |         2 |    11.6024  |        208.226  |       3.40657 |            0 |


## Same-device comparison with the BF16 model configuration

The MiniMax-M3 MXFP8 table overlaps
`minimax_m3_eagle_bf16_tuned_gemm.csv` on 61 gfx950/256-CU shapes
(BF16 input/output, no bias/scaling/preshuffle for BF16).
The selected BF16 backends were 47 FlyDSL, 11 Torch, 2 Triton and 1 ASM;
each production lookup was checked against the model row. Shapes without a
matching BF16 model row are excluded rather than benchmarked with fallback.

Both operands' MXFP8 values were dequantized to exactly representable BF16
values before benchmarking so both GEMMs compute the same mathematical input.
Preparation/quantization/preshuffle/compilation are outside timing. Both model
entries allocate output inside the call. Results are the median of three
alternating-order runs on one gfx950, with five warmups, 51 profiler iterations
and three rotating tensor sets; times sum GPU kernel durations, including
split-K reduction, and are not model end-to-end latency.

MXFP8 was faster on 50/61 shapes; geometric-mean BF16/MXFP8 speedup was 1.341x.
Small M (<=32) is mixed: 9/19 wins, geometric mean 1.018x.
At M>=1024, MXFP8 won all 28 shapes with geometric mean 1.668x.
These numbers describe the existing tuned configurations, not the theoretical
best possible performance of either format.

MXFP8 observed error ratios were zero at rtol=.03/atol=.1.
Some BF16 split-K configurations use BF16 atomic accumulation: their maximum
observed element mismatch fraction was 0.00449219; these were explicitly reported
and checked against a 0.05 bound, not hidden or claimed bitwise-equivalent.
This comparison does not evaluate quantization error relative to arbitrary
original BF16 model activations.

Representative raw latency rows:

|    m |    n |    k | bf16_backend   |   bf16 us |   mxfp8 us |
|-----:|-----:|-----:|:---------------|----------:|-----------:|
|    1 | 2304 | 6144 | flydsl         |   7.54116 |    8.96068 |
|   32 | 2304 | 6144 | flydsl         |   8.90196 |   10.1673  |
|  256 | 2304 | 6144 | flydsl         |  21.0678  |   16.2492  |
| 4096 | 2304 | 6144 | flydsl         | 161.531   |   81.2592  |
|    1 | 2560 | 6144 | flydsl         |   8.1291  |    7.48308 |
|   32 | 2560 | 6144 | flydsl         |   9.43416 |    9.88455 |
|  256 | 2560 | 6144 | flydsl         |  19.1412  |   15.84    |
| 4096 | 2560 | 6144 | flydsl         | 202.925   |   81.1208  |
|    1 | 6144 | 2048 | flydsl         |   7.12587 |    5.65744 |
|   32 | 6144 | 2048 | triton         |   7.88388 |    6.09128 |
|  256 | 6144 | 2048 | flydsl         |  14.5691  |   10.7288  |
| 4096 | 6144 | 2048 | flydsl         | 103.42    |   63.4581  |
| 4096 | 6144 | 3072 | flydsl         | 166.952   |   88.1024  |
| 4096 | 6144 | 6144 | torch          | 253.171   |  177.396   |


## Pre-PR review

The final functional review passed 140 tests plus 37 subtests. Coverage includes:
plain/preshuffled B, both direct/LDS paths, BF16/FP32, bias, split/slice/HTI,
minimal pipeline and scale-chunk boundaries, K=64 fallback, non-contiguous
inputs/scales, storage offsets and output guards, batched tgemm, streams/graphs,
inductor, real per-32 quantization, CSV lookup/mismatch rejection, architecture
dispatch isolation, and tuner -> CSV -> CPU-only AOT -> fresh-process run-only.
The checked-in 100-row MiniMax-M3 table passed a fresh-cache AOT build and
public-entry/tgemm/dirty-output verification with all error columns zero.
Existing HGEMM split-K and BF16/ordinary FP8 eager/inductor smoke tests passed.

Both main GEMM kernels now receive the dynamic `split_k` count for semaphore
completion/reset. The default partials path still uses it in the launcher and
standalone reducer. Only split presence and the semaphore/partials choice are
constexpr, so counts above one share the corresponding AOT binary.
Host input validation rejects invalid data/device/output/bias before launch.
AOT checks that dtype/bias/layout/target CSV fields agree with kernelName
rather than producing an unusable cache entry.

Limits of this review:
- gfx1250 hardware was not exercised; its original ASM function body and ABI
  are preserved and dispatch forwarding is tested with a stub on gfx950.
- The expanded direct/LDS B tuning axis is implemented and smoke-tested, but
  the shipped table preserves the previously tuned path choices. It has NOT
  been exhaustively retuned across both B paths.
- No full-model inference, memory-sanitizer run, or exhaustive shape-space
  proof is claimed. GEMM measurements exclude quantization/weight preparation.

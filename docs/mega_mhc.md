# HIP Mega-MHC stream kernels

H=5120/gfx950 uses one shared HIP translation unit with compile-time tuning.
The public API, weight packing, workspace management, shape policy, and JIT
launcher are all in `aiter/ops/mega_mhc.py`.

## Source organization

- `csrc/kernels/mega_mhc_stream.cu`: shared tuned implementation. The eleven
  small/mid-token variants instantiate `distributed_tiles<Config>`; the large
  path uses `large_tiles`; explicit splits use `generic_tiles`. The T=4160
  entry calls the large and generic tile functions in one kernel. Large and
  generic paths share the last-arrival reduction and normalization finisher.
- `csrc/include/mega_mhc_stream_config.h`: the fourteen compile-time parameter
  rows (split, chunks, wave groups, LDS padding, token specialization, reduction
  groups, prefetch, and segment layout). IDs match `_STREAM_VARIANTS` in Python.
- `aiter/jit/optCompilerConfig.json`: one `module_mega_mhc_stream` build recipe.
  The Python launcher supplies `MHC_STREAM_VARIANT` and a distinct module name;
  each configuration is compiled lazily from the same source.
- `csrc/kernels/mega_mhc.cu`: original generic implementation for other hidden
  sizes and gfx942, with its original packed-weight format.

There are no generated per-shape `.cu` copies and no external experiment-file
runtime dependencies. All tuning branches are resolved at compile time.

## Public entry point

Use `aiter.prepare_mega_mhc(fn, tokens)` once on the intended HIP stream, warm up
`aiter.mega_mhc(..., workspace=workspace)` before capture, and retain the workspace
for the graph's lifetime. The call returns `(residual, post, comb, norm, next_pre)`.
Prepare a separate workspace for each stream. The weights must remain unchanged;
prepare again after updating them. Inference tensors have no version counter, so
their immutability is the caller's responsibility.

With a workspace, each call launches one HIP kernel. The convenience call without
a workspace also packs weights and initializes scratch. Published kernel timings
exclude that preparation, allocations, compilation, and Python execution.

The public signatures are unchanged. H=5120/gfx950 selects this dispatch by
default; other H and gfx942 use `csrc/kernels/mega_mhc.cu`. Explicit `split_k` uses `generic`
with the new weight layout. Sinkhorn repeats other than 20 also use `generic`,
sharing the packed weights and backing partial storage. A separate counter array
lets one workspace alternate between generic and generation-tagged kernels,
including in a captured graph. A CU mask or fewer than 256 CUs selects the generic
path for shapes that otherwise use distributed normalization.

## Dispatch

| Tokens | Variant | Split K | Waves per workgroup |
|---|---|---:|---:|
| 0 | generic | 1 | 4 |
| 1–3 | decode_small | 32 | 4 |
| 4 | decode4 | 32 | 4 |
| 5–8 | decode8 | 32 | 4 |
| 9–128, post enabled | small (16 and 128 have dedicated variants) | 20 | 8 |
| 9–128, no post | small (16 and 128 have dedicated variants) | 32 | 4 |
| 129–256 | mid256 | 16 | 8 |
| 257–384 | mid384 | 10 | 8 |
| 385–512 | mid512 | 8 | 8 |
| 513–1024 | mid1024 | 4 | 8 |
| 1025–2048 | mid2048 | 2 | 8 |
| 4160 | tail | body 1, tail 32 | 4 |
| Other >2048 | large | 1 | 4 |

The distributed and generic paths assign one wave to each of four residual
streams. Eight-wave distributed kernels use two groups of four stream-owning
waves, each processing half the split's columns.
Small post tiles use 128-byte contiguous segments, with each lane still reading
16 bytes. Their workspace reserves capacity for 32 splits, and post uses the
first 20. This lets post and no-post share a prepared workspace, including in
one captured graph. Explicit `split_k` requests retain the generic path.

The large path normally uses two streams per wave when post-mix is enabled. Waves 0/1
cover columns 0–2559, and waves 2/3 cover columns 2560–5119. Within each pair,
the first wave produces streams 0/1 and the second produces streams 2/3. Each
wave reuses its four old residual values and sublayer value for two output
streams. Without post-mix, the large path retains one stream per wave. Both
choices instantiate the same `large_tiles` function and preserve 128-byte
segments. For grids fitting in one CU round (at most 4096 tokens), post keeps all four
streams in each wave, which owns one quarter of H. Post-mix and collapse then
reuse registers, and only the MFMA layout conversion needs wave-local LDS.
T=4160 uses four streams per wave for its body; its generic tail retains one.

Large tiles now choose the M dimension and collapse staging independently of
the split count. No-post uses M=32 for T>=8192 when the last round of 256
workgroups is at least 15/16 full (or exactly full); both M16 MFMA fragments
reuse each loaded fn operand. Other cases keep M=16. Token count remains a
runtime value, and all large tiles retain 16-byte lane loads and split K=1.

M=16 tiles can keep part of the BF16 collapse in LDS: 3584 columns per token
with post, or 4096 without post. The two-stream post layout stages the first
1792 columns of each column half; the four-stream layout stages the first 896
columns of each quarter. Normalization reads these values directly from LDS and
only reloads the remaining columns from global memory. It preserves the
main loop's 128-byte segments and processes four independent vectors per
batch. This uses 154 KiB of LDS, so staging is enabled only for at most one
CU round or a last round that is at least 15/16 full. Sparse final rounds keep
the original M16/global-staging path to avoid an occupancy-related tail.
For T=4160 with post, blocks 0–255 process the body and blocks 256–511 cover
the generic tail (128 of the tail blocks have valid rows). The body stages
2560 columns, keeping its shared storage plus the generic storage below the
LDS limit. Tail blocks run as CUs become available, without a second launch.
The no-post variant retains its two-phase M16 body and generic tail.
These choices add template instantiations inside the existing large JIT
module; the fourteen module names and the three tile implementations remain.

## Cache policy

On gfx950, buffer auxiliary value `2` means WAVE_NT (`nt`), while `GROUP_NT`
from `aiter_opus_plus.h` is `3` (`sc0 nt`), also used by `mhc_kernels.cu`.
The large path already uses WAVE_NT for residual/sublayer loads and residual
output stores. Switching those accesses to GROUP_NT did not improve timings.

Final normalized output stores use GROUP_NT in the two-split distributed
variant (1025–2048 tokens) and in large tiles with LDS collapse staging.
These paths benefit with both repeated and rotating input/output addresses.
Other variants retain their previous output policy. Intermediate collapse
stores, projection/norm weights, and split publication accesses retain their
existing policies; the kernel still needs their temporal reuse or visibility.
The two-split variant's residual/sublayer reads remain temporal: applying NT
to those reads substantially slows the 2048-token case.

Earlier fixed-address graph timings had a cache-assisted peak around 2048
tokens. With post-mix and H=5120, the dominant tensor footprint is about 212 MB
at 2048 tokens and 421 MB at 4096, spanning the MI355's 256 MB MALL capacity.
An address-rotation experiment with the same values, kernels and graph length
reduced the original 2048-token result from about 5.0 to 3.8 TB/s. Those numbers
came from a benchmark that explicitly disabled argument rotation; they do not
describe the default rotating-buffer behavior of `test_mhc.py` or the corrected
Mega-MHC benchmark below. The dispatch also changes from two splits and eight
waves to one split and four waves at this boundary. All reported TB/s values
are operator-level effective bandwidth, not physical HBM measurements.

## What changed

- **Projection:** the BF16 hi/lo weight pack fits the 24 projection rows into
  three MFMA operands instead of four. The last eight rows combine hi/lo in the
  two halves of the third operand. Projection and square-sum accumulation stay FP32.
- **Main loop:** contiguous elementwise accesses, LDS transpose/swizzle,
  multiple 32-column chunks per barrier, and shape-specific chunk/unroll settings
  reduce memory and synchronization costs. Large-token kernels also use
  128-byte residual segments and non-temporal accesses. BF16 dot2 accumulates
  residual/collapse square sums in FP32.
- **Large-token post-mix:** two streams per wave halve the old-residual LDS
  reads and duplicate sublayer loads. Packed FP32 multiply/FMA instructions
  process two elements together, retaining each element's accumulation order.
  Each stream keeps its own projection accumulator until the final reduction.
  Four streams per wave remove the old-residual LDS exchange for medium grids.
  This increases register use, so larger grids retain the two-stream mapping.
  Optional collapse staging uses additional LDS.
- **Large-token M reuse and epilog:** no-post M32 tiles share fn loads across
  two M16 fragments. Staged M16 tiles avoid the collapse write/read round trip
  for 70% of columns with post or 80% without post. A runtime workgroup-fill
  check avoids these choices when their scheduling tail would outweigh the
  saved loads. Sinkhorn still runs all 20 iterations for each token.
- **Reduction and gates:** compact partials, parallel/prefetched reduction,
  reciprocal multiplication, and DPP lane exchange shorten the serial finish.
  RMS means multiply by compile-time reciprocals of H and 4H, avoiding full
  floating-point division sequences for these constant divisors.
  Sinkhorn lane permutation reduces each column exchange to one DPP mirror
  instruction. Its DPP intrinsic has no old-value passthrough operand, avoiding
  unnecessary register copies while retaining compiler-scheduled dependency
  waits. Four/eight-lane elementwise square sums also use DPP: first a quad
  sum, then a half-row mirror to combine adjacent quad sums when needed.
  The projection fold and non-elementwise square sums retain their cross-lane
  `xor 16/32` shuffles. The final reduction computes and publishes rstd directly,
  eliminating two workgroup barriers in distributed variants.
- **Normalization:** each split retains its collapse tile in LDS and normalizes
  its own columns after statistics become available. Generation-tagged arrivals,
  L1-bypassing publication, and bounded waits with atomic handoff coordinate
  splits. The finisher handles columns handed off by delayed participants.
- **Dispatch:** token-specific split counts balance main-loop work against
  reduction overhead. T=4160 post uses separate body and tail workgroups within
  one launch. No workgroup must finish both phases in sequence.

These gains come from the combination. Reducing VGPR count alone did not help
when occupancy was unchanged, and forced higher occupancy with register spills
was slower. Stream ownership is one, two, or four streams per wave depending on the
path and mode; resource usage differs from the earlier 112-VGPR stream baseline.

## Validation

Run the correctness and performance comparison through the installed AITER
interfaces, in the same style as `op_tests/test_mhc.py`. All HIP Mega-MHC
checks live in the single standalone script `op_tests/test_mega_mhc.py`:

```bash
HIP_VISIBLE_DEVICES=6 GPU_ARCHS=gfx950 OMP_NUM_THREADS=1 MAX_JOBS=2 \
  python op_tests/test_mega_mhc.py --tokens 4 128 384 1024 4096 16384 \
  --hidden 5120 --mode post no_post identity_pre no_post_identity
```

The default run compares HIP, FlyDSL and Triton for 20 token counts and all four
modes, using the implementations in this checkout. Install the repository's
declared dependencies (including `flydsl==0.3.4.1`). `--backends hip triton` omits
FlyDSL; `--backends hip` runs HIP alone. FlyDSL Mega-MHC requires gfx950, while
the HIP and Triton comparison also runs on gfx942.

The script prints a plain text table with a latency (`*_us`) and effective
bandwidth (`*_TB/s`) column for each backend; it does not emit JSON or speedup
ratios. `--seed` selects the shared input seed, and `--iters` sets the number of
timed calls (default 101). T=0 checks the empty outputs and displays `-` for
latency and bandwidth.

All backends use the same dominant-tensor traffic model as the FlyDSL Mega-MHC
benchmark: `20*T*H + 384*H` bytes with post-mix, or `10*T*H + 384*H` without it.
This counts the BF16 residual input and normalized output, the sublayer input and
new residual output when present, and one read of the FP32 projection weight
(equivalently its BF16 hi/lo pack). Identity-pre modes use the corresponding
post/no-post formula. Effective bandwidth is `bytes / latency_us / 1e6` TB/s
(decimal). Scratch, repeated loads and small gate/norm tensors are excluded;
this metric is not a measurement of physical HBM traffic.

Each backend first passes the same FP32 reference checks with the existing
tolerances. Like `test_mhc.py`, timing passes actual tensor arguments to
`run_perftest`, keeping its default `num_rotate_args=0` and `testGraph=False`.
The helper selects a rotation count from its existing memory/cache sizing
logic, deep-copies the arguments, and profiles calls with those copies.
Residual/sublayer inputs, weights, mixing tensors and the preallocated residual
output all rotate. The log reports each backend's actual buffer-set count and
the test verifies that every prepared residual address was used.

The helper's optional `prepare_func` callback prepares the original arguments
before memory sizing and every copied set before profiling. HIP gives each
weight copy its own packed weights and workspace; FlyDSL warms its weight
cache and launcher. Thus compilation, weight packing and workspace setup are
outside the returned kernel timing. The callback does not change the helper's
default rotation or timing policies. Public-wrapper GPU work remains included,
such as Triton's identity-pre initialization. The reported time sums GPU
kernel durations, excluding Python and host allocation time. Outputs are
checked after timing; separate graph replays check changed and restored input.

Run the full correctness matrix without benchmarking using the same script:

```bash
HIP_VISIBLE_DEVICES=6 GPU_ARCHS=gfx950 OMP_NUM_THREADS=1 MAX_JOBS=2 \
  python op_tests/test_mega_mhc.py --check-only
```

`--check-only` runs ordinary Python checks and prints a pass count; no test
runner is needed. It uses the full regression matrix rather than the benchmark
shape/backend selection flags. The checks cover the original supported shapes,
all tuned dispatches and their
boundaries, explicit split reuse, output reference tolerances, inference weights,
concurrent graphs, generation wrap, changed graph inputs, CU-mask fallback, and
alternating Sinkhorn repeat counts on one workspace. Reference tolerances and
the default 20 Sinkhorn iterations are unchanged.

The device bodies are guarded for gfx950 so multi-architecture builds can also
include gfx942 without compiling unsupported large-LDS specializations there.
The original kernel provides the gfx942 implementation.

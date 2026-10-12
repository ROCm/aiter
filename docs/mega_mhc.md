# HIP Mega-MHC stream kernels

All HIP Mega-MHC kernels use one shared translation unit. H=4096/5120/6144/7168
on gfx950 selects tuned tiles; gfx942 and other hidden sizes use portable generic
tiles with the same weight pack and workspace layout.
The public API, weight packing, workspace management, shape policy, and JIT
launcher are all in `aiter/ops/mega_mhc.py`.

## Source organization

- `csrc/kernels/mega_mhc.cu`: shared implementation. The eleven
  small/mid-token variants instantiate `distributed_tiles`; the large
  path uses `large_tiles`; gfx942, other H and explicit splits use `generic_tiles`.
  The T=4160 entry calls the large and generic tile functions in one kernel. Large and
  generic paths share the last-arrival reduction and normalization finisher.
  The same file contains the fourteen compile-time parameter rows (split,
  chunks, wave groups, LDS padding, token specialization, reduction groups,
  prefetch, and segment layout) and the shared group32 MXFP8 normalization
  epilogue. Configuration IDs match `_STREAM_VARIANTS` in Python. Quantization's
  boolean template parameter removes that work from BF16 kernels.
- `csrc/include/mega_mhc.h`: the shared native interface declaration used by
  the kernel and `csrc/pybind/mega_mhc_pybind.cu`. This is the only Mega-MHC header;
  implementation-only configuration and quantization helpers live in the CU.
- `aiter/jit/optCompilerConfig.json`: one `module_mega_mhc_stream` build recipe.
  The Python launcher supplies `MHC_HIDDEN` and `MHC_ARCH_GFX950`. Each module is
  named `module_mega_mhc_stream_h{H}_{arch}` and contains all configurations
  supported for that hidden size and architecture. C++ selects the variant and
  BF16/MXFP8 template instantiation at launch. The module names stay unchanged
  after the source rename; both output types live in the same gfx950 module.

There are no generated per-shape `.cu` copies and no external experiment-file
runtime dependencies. All tuning branches are resolved at compile time.

JIT compilation is lazy per hidden size and architecture. On gfx950, using all
four tuned hidden sizes produces four `.so` files, rather than a separate file
for every token configuration and output type. First use of one H compiles its
complete set of kernels; subsequent token/output switches reuse the same module.
Each kernel still has independent register and LDS allocations. Gfx942 modules
contain only BF16 generic kernels. Other supported H values also get one module
each, containing generic kernels and, on gfx950, both output types.

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

H=4096/5120/6144/7168 on gfx950 selects this dispatch by
default; other H and gfx942 select `generic` in the same CU. Explicit `split_k`
and Sinkhorn repeats other than 20 also use `generic`,
sharing the packed weights and backing partial storage. A separate counter array
lets one workspace alternate between generic and generation-tagged kernels,
including in a captured graph. A CU mask or fewer than 256 CUs selects the generic
path for shapes that otherwise use distributed normalization.

The portable generic kernel uses four waves, two 32-column chunks, and at most
26,628 bytes of LDS, below gfx942's 64 KiB limit. It uses FP32 FMA square sums
instead of gfx950-only BF16 dot2 instructions. The same arithmetic is used by
generic on gfx950, allowing that fallback to be exercised on either architecture.
Split-K publication uses device-scope fences and an acquire/release arrival
counter; it does not assume split workgroups share an XCD or are simultaneously
resident. The default split count follows the original generic policy: the largest
divisor of H/32 up to 20 whose token grid fits the visible CU count, or one split
for a larger grid. Gfx942 never selects the large-LDS distributed/large variants;
both Python dispatch and the native entry enforce this. MXFP8 remains gfx950-only.

### Fused MXFP8 output

On gfx950, pass `out_dtype="mxfp8"` to return
`(residual, post, comb, (norm, q, e8m0), next_pre)`. `norm` remains BF16 `(T,H)`;
`q` is contiguous `torch.float8_e4m3fn` `(T,H)` and `e8m0` is contiguous uint8
`(T,H/32)`. Reinterpret the exponent bytes with `.view(torch.float8_e8m0fnu)`
when required by a downstream GEMM. The default `out_dtype="bf16"` retains
the original return structure. Both modes reuse the same prepared workspace.

Quantization consumes the **rounded BF16 norm**. Each lane owns eight values;
two DPP permutations reduce amax across four adjacent lanes, covering 32 columns.
The scale is `ceil_pow2(max(amax, 1e-10) / 448)` with biased exponent clamped to
1..254, matching FlyDSL Mega-MHC's MXFP8 output. Four
`v_cvt_scalef32_pk_fp8_bf16` instructions convert the eight elements. Each lane
stores eight FP8 bytes, and one lane per group writes the exponent. The quantized
outputs use temporal stores for the following consumer. No extra kernel,
global reduction, or change to split K is needed.

This policy follows FlyDSL Mega-MHC; ATOM's native attention quantizer currently
uses a different amax floor (`1e-4`). ATOM integration requires aligning that
policy and forwarding quantized inputs alongside BF16 inputs for other consumers.

Run the fused comparison with:

```bash
python op_tests/test_mega_mhc.py --out-dtype mxfp8 --backends hip flydsl
```

Use `--out-dtype bf16 mxfp8 --backends hip flydsl` to show both output modes.
The default BF16 table retains Triton. Triton Mega-MHC has no fused MXFP8 output.
All timings use `run_perftest` with actual tensor arguments, automatic buffer
rotation, and 101 calls by default. Effective bandwidth includes the extra
`T*H + T*H/32` output bytes for MXFP8. With `--out-dtype mxfp8`, both normal
runs and `--check-only` verify quantized bytes and scales against the reference.

## Dispatch

The table below is the original H=5120 policy, retained unchanged. Other
supported hidden sizes share the tile implementations and use the adjustments
listed after the table.

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
| 2049–4096, post enabled | large | 1 | 8 |
| 4160 | tail | body 1, tail 32 | 4 |
| Other >2048 | large | 1 | 4 |

For H=4096/6144/7168, small post uses split 16 instead of 20, and the
257–384-token range uses split 8 instead of 10, preserving the required K
divisibility. Decode and the 129–256-token range use H/1024 chunks, with LDS
padding to an even number of chunks. H=7168 uses four waves for small post
and the 129–256-token range, and two chunks for the 257–384-token range.
These choices preserve complete K tiles; a compile-time assertion checks the
divisibility instead of introducing partial K iterations into distributed staging.
At H=6144/7168, the 1025–2048-token range uses
four waves and one stream per wave: its larger collapse tile leaves insufficient
LDS for the H=5120 eight-wave exchange. H=4096 no-post large tiles stage 3072
columns instead of 4096. Hidden size stays a compile-time constant throughout
weight addressing, reductions, normalization and the T=4160 tail offsets.

Tuned gfx950 variants at all four widths keep each four-instruction BF16
square-sum sequence in one assembly block with an early-clobber accumulator and
explicit dependency waits.
This protects the result when the compiler copies the accumulator to another
VGPR in a different unroll schedule. Development regression checks also covered
H=5120, where the original sequence could lose a square-sum term.

To compare all supported stream widths, use:

```bash
python op_tests/test_mega_mhc.py --hidden 4096 5120 6144 7168
python op_tests/test_mega_mhc.py --hidden 4096 5120 6144 7168 --out-dtype mxfp8 --backends hip flydsl
```

Only H=5120 is selected by default. `--check-only` uses the same M/H/mode/
output/backend selection as the performance run, checking eager output without
timing.

The following layout sizes describe H=5120; the other widths use the adjustments
above and scale column stripes with H.

The generic path and most distributed variants assign one wave to each of four
residual streams. Their eight-wave kernels use two groups of four stream-owning
waves, each processing half the split's columns.
Small post tiles use 128-byte contiguous segments, with each lane still reading
16 bytes. Their workspace reserves capacity for 32 splits, and post uses the
first 20. This lets post and no-post share a prepared workspace, including in
one captured graph. Explicit `split_k` requests retain the generic path.

For 1025–2048 tokens with post enabled, each of the eight waves owns all four
streams and a distinct 320-column stripe of its split. It reuses residuals for
post-mix and collapse in registers, using wave-local LDS only for the MFMA
transpose. Five 64-column steps replace the original twenty 64-column steps
per wave. Each lane still reads 16 bytes, and eight adjacent lanes cover one
128-byte segment. Split K remains 2, and the no-post mapping is unchanged.

This path issues the first fn chunk before loading the current residual tile;
later fn chunks are loaded when their MFMAs are reached. Holding the next
residual tile at the same time caused register spills, so this configuration
does not prefetch that tile. Scalar post arithmetic with learned pre-mix and
packed arithmetic with identity pre-mix fit without spills. Compact 26-column
projection/square-sum scratch shares storage with the subsequent split reduction,
keeping total LDS at 162832 bytes. FP32 accumulations, BF16 rounding seams, and
all 20 Sinkhorn iterations are retained.

The large path normally uses two streams per wave when post-mix is enabled. Waves 0/1
cover columns 0–2559, and waves 2/3 cover columns 2560–5119. Within each pair,
the first wave produces streams 0/1 and the second produces streams 2/3. Each
wave reuses its four old residual values and sublayer value for two output
streams. Without post-mix, the large path retains one stream per wave. Both
choices instantiate the same `large_tiles` function and preserve 128-byte
segments. For grids fitting in one CU round (at most 4096 tokens), post keeps all four
streams in each of eight waves, which owns a 640-column stripe of H. Post-mix
and collapse reuse registers, and only the MFMA layout conversion needs
wave-local LDS. The two 32-column MFMA chunks reuse one 32-column transpose
buffer: each half of the row's eight lanes publishes its chunk, then a wave
barrier protects the read and subsequent reuse. This saves 32 KiB of LDS
compared with an eight-wave, 64-column transpose and keeps 3584 collapse
columns staged. Its reduction scratch is shared between phases; total LDS is
162816 bytes. The first fn chunk overlaps the current residual load and the
second is loaded at its MFMA. Scalar post arithmetic and packed identity-pre
arithmetic fit in 256 and 240 VGPRs respectively, without spills.
T=4160 retains four waves with four streams per wave for its body; its generic
tail retains one stream per wave.

Large tiles now choose the M dimension and collapse staging independently of
the split count. No-post uses M=32 for T>=8192 when the last round of 256
workgroups is at least 15/16 full (or exactly full); both M16 MFMA fragments
reuse each loaded fn operand. Other cases keep M=16. Token count remains a
runtime value, and all large tiles retain 16-byte lane loads and split K=1.

M=16 tiles can keep part of the BF16 collapse in LDS: 3584 columns per token
with post, or 4096 without post. The two-stream post layout stages the first
1792 columns of each column half; the eight-wave post layout stages the first
448 columns of each 640-column stripe. Normalization reads these values
directly from LDS and only reloads the remaining columns from global memory.
Eight-wave normalization keeps the main loop's 128-byte segments, handles
rows r and r+8 together, and reuses one weight vector for both rows. Each wave
chooses LDS or global loads uniformly. The other staged layouts retain their
four-vector batches. Staging uses 154–159 KiB of LDS, so it is enabled only
for at most one CU round or a last round that is at least 15/16 full. Sparse
final rounds keep the original M16/global-staging path to avoid an occupancy-related tail.
For T=4160 with post, blocks 0–255 process the body and blocks 256–511 cover
the generic tail (128 of the tail blocks have valid rows). The body stages
2560 columns, keeping its shared storage plus the generic storage below the
LDS limit. Tail blocks run as CUs become available, without a second launch.
The no-post variant retains its two-phase M16 body and generic tail.
These choices add template instantiations inside each hidden-size module;
the fourteen tuning configurations and three tile implementations remain.
The output mode selects an instantiation within the same module.

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
The two-split post variant uses WAVE_NT residual/sublayer reads and residual
stores with its four-stream layout. Its no-post reads remain temporal. Applying
NT to the earlier one-stream post layout was slower; the policy is selected
together with the stream mapping and load schedule.

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
  128-byte residual segments and non-temporal accesses. Tuned gfx950 variants use
  BF16 dot2 to accumulate residual/collapse square sums in FP32; portable generic
  variants use FP32 FMA.
- **Mid-token post-mix:** the two-split path keeps all streams in one wave,
  removing workgroup-wide exchanges from each post/collapse step. Deferred fn
  loads and current-tile residual loads bound register use without spills.
  The same distributed template still handles the other shapes and no-post.
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

`op_tests/test_mega_mhc.py` is a standalone correctness and performance test,
following the reference → backend → `run_perftest` → table flow in `test_mhc.py`.
One explicit parameter matrix is used for both normal runs and `--check-only`:

| Option | Meaning | Default |
|---|---|---|
| `-m`, `--tokens` | M: number of tokens | 1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 |
| `-n`, `--hidden` | H: hidden size per stream | 5120 |
| `--mode` | With or without post-mix | post, no_post |
| `--out-dtype` | Normalized BF16, optionally with fused MXFP8 | bf16 |
| `--backends` | Implementations to check and time | HIP/FlyDSL/Triton on gfx950; HIP/Triton on gfx942 |

The default run has 26 cases (13 M × 1 H × 2 modes). Both default modes use
learned delayed pre-mix. Select `identity_pre` or `no_post_identity` explicitly
to use stream 0 as the delayed pre-mix input. Use `--out-dtype bf16 mxfp8` to
include both outputs; Triton has no fused MXFP8 and its cells show `-` for those
rows. FlyDSL and fused MXFP8 require gfx950. Install the selected backend's
repository dependencies (including `flydsl==0.3.4.1` for FlyDSL).
Triton's shipped launch configs are tuned for H=5120 and require
`H % (NUM_KSPLIT * TILE_K) == 0`. Cases that violate this constraint show SKIP
in the correctness table or `-` in timing columns; the final count distinguishes
completed checks from skipped comparisons. Actual backend errors still fail the run.

```bash
# Default M/H matrix: reference checks followed by backend timings.
python op_tests/test_mega_mhc.py

# Focus on the two performance cases, at H=5120.
python op_tests/test_mega_mhc.py -m 2048 4096 -n 5120 --mode post

# The same selected cases and backends, without timing.
python op_tests/test_mega_mhc.py -m 2048 4096 -n 5120 --mode post --check-only

# Fused quantization comparison.
python op_tests/test_mega_mhc.py -m 1 16 2048 4096 -n 5120 \
  --out-dtype mxfp8 --backends hip flydsl
```

Each selected case checks all outputs against the FP32 reference with the
existing BF16 seam rounding and tolerances. MXFP8 codes and E8M0 scales are
checked byte-for-byte against quantization of that backend's rounded BF16 norm.
Preparation, correctness checks and timings use the caller's current stream;
the script does not create a separate stream or capture/replay graphs. HIP
workspace counters are checked after calls. `--check-only` prints PASS for each
selected backend and never invokes the timing helper. There is no separate
hard-coded regression matrix, pytest dependency, JSON output, or speedup column.

Performance runs print a plain text table with `*_us` and `*_TB/s` columns.
`--seed` selects shared input data; `--iters` sets timed calls (default 101).
M=0 may be selected to check empty outputs; no latency or bandwidth is reported.

Timing passes actual tensor arguments to `run_perftest`, using its default
`num_rotate_args=0` and `testGraph=False`. Inputs, weights, mixing tensors and
the preallocated residual output rotate automatically. The `prepare_func`
callback packs each HIP weight copy and prepares its workspace, or warms the
FlyDSL weight cache, before profiling. The test checks that all prepared buffer
sets were used. GPU timings exclude compilation, packing and workspace setup;
public-wrapper GPU work, such as Triton's identity-pre initialization, remains
included. Outputs are checked again after timing.

Effective bandwidth uses the same dominant-tensor model for every backend:
`20*M*H + 384*H` bytes with post-mix, or `10*M*H + 384*H` without it. MXFP8 adds
`M*H + M*H/32` bytes. This counts residual/norm traffic, sublayer input and new
residual when present, and one projection-weight read. Bandwidth is
`bytes / latency_us / 1e6` TB/s. Scratch, repeated loads and small gate/norm
tensors are excluded; this is not a measurement of physical HBM traffic.

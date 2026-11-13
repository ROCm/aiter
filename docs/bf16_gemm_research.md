# Source-driven BF16 GEMM research and bank-layout optimization

Research date:2026-11-13. Baseline:`f215a62ebb7c806e88b7c379c2f5d26fd5e58c63`.

## Research scope and evidence discipline

The user requested a broad literature/source investigation rather than more
unguided parameter sweeps. This pass indexed **14 repositories**, saved
**51 unique source/document files at pinned revisions**, and successfully
retrieved **9 external documents/pages**, including5 PDFs. The PDFs total
608/21/41/13/75 pages for ISA/whitepaper/HipKittens/Stream-K/Volkov. Text was
extracted for targeted reading; this is **not** a claim that every page,
indexed file, repository or public source was read exhaustively.

[Retrieval manifest](research/bf16_gemm_sources_20261113.json) records URLs,
revisions, content hashes, dates, and failed retrievals. The rocm-libraries
root tree was truncated; its hipblaslt and rocwmma subtrees were separately
retrieved without truncation. Anonymous GitHub API throttling was resolved
using the existing CLI; Google redirect pages and a hipBLASLt404 were not
treated as useful search results. No downloaded build scripts were executed.

Read in depth:HipKittens sections3.2–3.4 and appendicesD.1–D.4; CDNA4 ISA
LDS/register allocation sections; corresponding layout/GEMM source; AMD
GEMM/decode/occupancy articles; Tensile parameter validators and persistent
components; CK pipeline scheduling sections; Stream-K section5.2; NVIDIA
CUTLASS/Tilus persistent scheduling. Other items are reference/index coverage,
not a claim of exhaustive code review.

### Primary sources and what they establish

| Source | Relevant evidence | Applicability / limits |
|:---|:---|:---|
| [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf), sections2.2.1,3.6.4,3.6.5,11 | 160KiB LDS,64 banks, register allocation and memory instruction semantics | Primary hardware reference; not a complete per-lane LDS phase specification |
| [CDNA4 whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) |8 XCDs,32 active CUs/XCD,4MiB L2/XCD | Overrides conflicting informal hardware tables |
| [HipKittens paper](https://arxiv.org/abs/2511.08083),3.2/3.3/D.1/D.2 | Non-contiguous LDS phases; why no single swizzle works for all tile shapes; AMD WS tradeoffs | Phase table is authors' measured model; corroborated by our PMC, not assumed an ISA guarantee |
| [HipKittens BF16 GEMM](https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp) and [shared tile shapes](https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna4/types/shared/st_shape.cuh) | Optimized CDNA4 tile layouts,8-wave ping-pong | Square-GEMM author results are not our rectangular-shape results |
| [AMD CDNA4 GEMM](https://github.com/ROCm/rocm-blogs/blob/7d12cba060c82bf4d17529786ee6e26032d9f610/blogs/software-tools-optimization/cdna4-gemm-kernels/README.md) | Direct LDS DMA, bank conflicts, ping-pong | FP8 examples, not permission to change BF16 precision |
| [AMD low-latency GEMM](https://github.com/ROCm/rocm-blogs/blob/7d12cba060c82bf4d17529786ee6e26032d9f610/blogs/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.md) | Hierarchical K parallelism and B-to-register alternative | Parts already attempted here; single-launch reduction must retain correct precision/lifetime |
| [AMD occupancy analysis](https://github.com/ROCm/rocm-blogs/blob/7d12cba060c82bf4d17529786ee6e26032d9f610/blogs/software-tools-optimization/occupancy-math-mi355x/README.md) / [Volkov](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf) | Resource allocation vs ILP, low occupancy can still saturate compute | Microbenchmarks/other data types do not establish GEMM improvement |
| [CK v5](https://github.com/ROCm/composable_kernel/blob/3f0199f548332bdf0b67f8140618def6859289ac/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v5.hpp) / [CK-Tile v4](https://github.com/ROCm/composable_kernel/blob/3f0199f548332bdf0b67f8140618def6859289ac/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp) | Separate global/local prefetch, register buffers and instruction-count-based scheduler | Not equivalent to globally removing sched_barrier |
| [Current Tensile options](https://github.com/ROCm/rocm-libraries/blob/eb9b307c4d56d5df63bd3bc60603d33aaa3abbc5/projects/hipblaslt/tensilelite/tensilelite/Common/ValidParameters.py) / [validators](https://github.com/ROCm/rocm-libraries/blob/eb9b307c4d56d5df63bd3bc60603d33aaa3abbc5/projects/hipblaslt/tensilelite/tensilelite/SolutionStructs/Solution.py) | DirectToVgpr, persistent strategies, prefetch constraints | Check validators, not just parameter names |
| [PersistentLoop](https://github.com/ROCm/rocm-libraries/blob/eb9b307c4d56d5df63bd3bc60603d33aaa3abbc5/projects/hipblaslt/tensilelite/tensilelite/Components/PersistentLoop.py) | Prefetch next output tile vs retaining A across tiles are distinct | Some combinations/TDM-only paths are unavailable on gfx950 |
| [Stream-K paper](https://arxiv.org/abs/2301.03598),5.2 / [ROCm implementation](https://github.com/ROCm/triton/blob/0ec280cf80dd91e9a86887981a670f2d4541a32b/python/perf-kernels/streamk/streamk_kernel.py) | Hybrid data-parallel + limited Stream-K reduces cache skew | Existing all-segment Stream-K here is not necessarily that hybrid |
| [hipBLASLt current docs](https://github.com/ROCm/rocm-libraries/blob/eb9b307c4d56d5df63bd3bc60603d33aaa3abbc5/projects/hipblaslt/docs/how-to/how-to-use-streamk.rst) | Persistent grid/work assignment and heuristic controls | Source revision may be newer than installed runtime; no unseen feature is claimed locally available |
| [CUTLASS ping-pong](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp) / [Tilus persistent](https://github.com/NVIDIA/tilus/blob/d8e6cc69f88de0dfe333032b2f7d03ae122080ba/examples/hopper_matmul/matmul_v6.py) | Cross-output-tile pipeline and compute/epilogue overlap | TMA/WGMMA/TMEM/setmaxnreg cannot be copied literally to gfx950 |
| [Matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator/blob/94b1f16bf3a2c7089f0f3ec0b3eb303396a18776/README.md) | MFMA operand/output lane maps | Used as layout reference, not a performance oracle |

### Corrections obtained by following sources to implementations

1. Our HTI already implements AMD8-wave ping-pong. Renaming it or adding
   loader waves is not a new optimization. HipKittens paper Table2 also
   reports dedicated producer/consumer BF16 variants slower than its larger
   non-specialized tile; this supports our failed WS experiments, not a
   universal claim against WS.
2. HipKittens' BF16 directory README contains an inconsistent chiplet table
   (32 XCDs/8 CUs) and cache/register descriptions. Whitepaper and the paper
   specify8 XCDs/32 active CUs,4MiB L2 per XCD. We do not repeat that README
   table as fact.
3. The [new HipKittens GEMM ladder](https://github.com/ROCm/rocm-blogs/blob/7d12cba060c82bf4d17529786ee6e26032d9f610/blogs/software-tools-optimization/hipkittens-gemm-ladder/README.md) targets
   CDNA5/Helios, wave32, TDM/partitioned LDS/clusters. It is a conceptual
   scheduling reference, not a gfx950 instruction recipe.
4. hipBLASLt and rocWMMA standalone repositories are retired. Current source
   inspection uses `ROCm/rocm-libraries`, not only the deprecated branches.
5. Per-tensor `PrefetchGlobalReadA/B` looks immediately useful, but current
   [DecouplePGR](https://github.com/ROCm/rocm-libraries/blob/eb9b307c4d56d5df63bd3bc60603d33aaa3abbc5/projects/hipblaslt/tensilelite/tensilelite/Components/DecouplePGR.py) validators require TDMInst=3 and reject
   DirectToLds/DirectToVgpr. It is **not a gfx950 switch** we can just enable.
   `ReuseAcrossPersistent` currently also requires wave-separated TDM; the
   general operand-residency idea needs a separate gfx950 implementation.
6. `nvjet_*` denotes a cuBLAS/cuBLASLt kernel family. Public TensorRT-LLM
   references identify it, but no complete public implementation was found.
   CUTLASS/Tilus mechanisms are not asserted to be nvJET internal algorithms.

## Mechanism-to-experiment ledger

| Mechanism | Status against earlier work | Outcome / next action |
|:---|:---|:---|
| BK-dependent LDS phase-bank layout | Earlier generic XOR sweeps did not derive BK128 phase ownership | **Implemented and measured in this pass; selected below** |
| Finite XCD chunk C + cache window W | Earlier grouping treated the whole grid as one chunk | Implemented16 permutations on each of2 large shapes; one small screening gain, no production promotion |
| B direct-to-VGPR | Already attempted in portions of the earlier small-M work | Revisit only with a new register-prefetch schedule and measured reuse; not claimed new merely by renaming |
| Cross-output-tile prefetch | Existing persistent loops refill/drain per segment | Still a distinct next experiment; separate LDS C/input lifetimes required |
| Full-K operand residency across tiles | Not equivalent to prefetch | Hardware/current implementation constraints above; capacity-model first |
| Hybrid DP + bounded Stream-K remainder | Existing Stream-K does not establish this exact hybrid | Mainly large down:896 CTAs=3.5 waves at256CUs; full large up has1792=7 waves, less quantization benefit |
| CK-derived fine-grained4-wave scheduling | Prior blanket sched masks/AGPR variants mostly failed | Requires emitted instruction-count model, not another blind mask sweep |
| Mismatched precision/FP8/TDM/WGMMA shortcuts | Outside BF16 gfx950 contract | Excluded |

## Finding: BK128's generic LDS layout aliases banks within a read phase

For MFMA16x16x32 BF16, a representative operand address is
`row=lane%16`, `k=8*(lane//16)` plus tile offsets. HipKittens Table5 reports
`ds_read_b128` phases:
-0–3,12–15,20–27;
-4–11,16–19,28–31;
-and the same sets+32.

We enumerated each lane's four32-bit words, using64 banks and treating
same-address reads separately from distinct-address bank contention.

At BK128, the old element-offset permutation is:

    x = row * 128 + col
    old = x ^ (((x >> 6) & 7) << 3)   # Swizzle(3,3,3)

It gives two distinct words per bank in the modeled read phase. A compatible
permutation is:

    new = x ^ (((x >> 7) & 7) << 4)   # Swizzle(3,4,3)

This XORs K bits6:4 with row bits2:0. It stays inside the same128-element row,
is self-inverse, and preserves every contiguous8-BF16 (16-byte) DMA packet.
Enumeration covers all64x128 addresses and all K subchunks. The new phase
mapping gives one distinct word per bank. **The old BK64 default does not
have this modeled input-read conflict.** We fix the blocked BK128 candidate,
then compare that candidate against the old BK64 default.

### Hardware counters confirm the same-geometry difference

Integrated-code rocprofv3 PMC, second matching dispatch, same input and
64x64x128/stage3/waves2x4 geometry. Counters are aggregated per-SE as reported
by the tool, not wall-clock latency percentages:

| Variant | SQ_INSTS_LDS | SQ_LDS_BANK_CONFLICT | SQ_LDS_IDX_ACTIVE |
|:---|---:|---:|---:|
| Original BK64 default |618,240|28,672|2,472,960|
| BK128, old Swizzle(3,3,3) |618,240|2,437,120|4,881,408|
| BK128, new Swizzle(3,4,3) |618,240|28,672|2,472,960|

The same-geometry LDS instruction count is unchanged while conflict stall
count drops98.82%. This is not a98.82% GEMM speedup. Residual counter activity
may include epilogue operations; we do not call the whole kernel conflict-free.

## Deployed scope and compatibility

Exactly one model CSV key changes: gfx950,256CU,M128,N7168,K3584,no bias,
BF16 input/output,no scales,no preshuffle.

- Before:64x64x64/stage7/waves4x2x1/split1/FT.
- After:64x64x128/stage3/waves2x4x1/split1/FT with qualified `_sw343` layout.
- Still ordinary FT, not wave-specialized; no scratch, extra launch or dtype
  change. FP32 accumulation and single BF16 output conversion remain.
- The internal `bank_swizzle_343` flag is derived only for that exact shape
  and config. Caller overrides cannot turn it on for other shapes.
- New binary suffix invalidates stale disk-cache entries. Stable CSV name
  specifies tile/stages/waves; FlyDSL dispatch resolves it by name.
- When padded-M lookup reaches the retuned row for M!=128, it returns that
  row's original BK64 config. It does not silently fall through to ASM or
  use the new tile on neighbors. Public strided inputs retain normalization;
  direct kernel calls with valid aligned row padding are tested.
- The other three task rows are unchanged, including f215a62's large-up gain.
  Independent cold compilations confirm all their `.text` hashes unchanged;
  the original BK64 kernel is still emitted for padded-M consumers.
- New kernel metadata:56 VGPR,0 AGPR,96KiB LDS,zero private scratch.
  PMC resource columns are not substituted for code-object metadata.

## Actual before/after CSV and public-call measurements

Before:f215a62 source + its model CSV. After:integrated repository source +
retuned model CSV. Both go through `gemm_a16w16_out`; implementation/config
switching and lookup-cache invalidation are outside timing. No per-call mock
or cached result. Two initially idle MI355X/256CU cards: A=0000:05:00.0/HIP1,
B=0000:15:00.0/HIP3. Initial idle does not guarantee full-run exclusivity;
no clocks or other users' workloads were changed.

Six fresh processes,16 AB/BA rounds per mode:
-rotation: standard run_perftest profiler rotation,10 warmups/101 iterations.
  It reports device-time aggregation, not Python wall time.
-rotate_graph8:eight distinct A/B/output sets,5 warmups/30 event replays,
  divided by8. This is a different reuse distance from rotation.
-graph4:four complete calls at fixed addresses, same event protocol,
  divided by4.
-Table cells are medians of three process medians on each card.
  Delta=after-before. No rounds/processes are removed.

| Card suffix | Mode | Before us | After us | Delta us |
|:---|:---|---:|---:|---:|
| 05:00.0 | rotation | 19.993 | 19.750 | -0.243 |
| 05:00.0 | rotate_graph8 | 21.981 | 21.886 | -0.095 |
| 05:00.0 | graph4 | 19.512 | 17.811 | -1.701 |
| 15:00.0 | rotation | 20.380 | 20.010 | -0.371 |
| 15:00.0 | rotate_graph8 | 22.005 | 21.865 | -0.140 |
| 15:00.0 | graph4 | 19.594 | 18.002 | -1.592 |

All18 target process/mode comparisons improve:
rotation0.341–1.819%, rotate_graph8 0.434–6.579%, graph4 8.126–11.279%.
Seed3562 encountered a much higher-latency regime on cardB; its results remain
below, but its larger speedup is not advertised as a stable isolated-hardware
gain. CardA's graph4 reductions are8.57–8.82%.

CSV `us=19.7504`, `tflops=332.99`, `bw=2740.85` GB/s use cardA's three-process
**rotation** median. The historical CSV19.5768us is not the measured before
value and is not used to claim a regression or a win; same-batch before on
that card is19.993us. graph4 is reported separately.

### Every final target process

| Seed | Mode | Before us | After us | Improvement |
|---:|:---|---:|---:|---:|
| 3551 | rotation | 19.993 | 19.750 | 1.214% |
| 3551 | rotate_graph8 | 22.047 | 21.921 | 0.572% |
| 3551 | graph4 | 19.627 | 17.896 | 8.820% |
| 3552 | rotation | 20.135 | 19.997 | 0.683% |
| 3552 | rotate_graph8 | 21.487 | 21.289 | 0.924% |
| 3552 | graph4 | 19.268 | 17.617 | 8.569% |
| 3553 | rotation | 19.787 | 19.719 | 0.341% |
| 3553 | rotate_graph8 | 21.981 | 21.886 | 0.434% |
| 3553 | graph4 | 19.512 | 17.811 | 8.717% |
| 3561 | rotation | 20.492 | 20.277 | 1.049% |
| 3561 | rotate_graph8 | 21.863 | 21.758 | 0.478% |
| 3561 | graph4 | 19.257 | 17.642 | 8.388% |
| 3562 | rotation | 20.380 | 20.010 | 1.819% |
| 3562 | rotate_graph8 | 33.112 | 30.933 | 6.579% |
| 3562 | graph4 | 31.982 | 28.375 | 11.279% |
| 3563 | rotation | 19.813 | 19.632 | 0.916% |
| 3563 | rotate_graph8 | 22.005 | 21.865 | 0.636% |
| 3563 | graph4 | 19.594 | 18.002 | 8.126% |

### Unchanged controls, including unfavorable fluctuations

| Card | M,N,K | Mode | Before us | After us | Implementation |
|:---|:---|:---|---:|---:|:---|
| a | 128,3584,7168 | rotation | 21.126 | 21.085 | unchanged device .text |
| a | 128,3584,7168 | graph4 | 19.077 | 19.309 | unchanged device .text |
| a | 128,7168,3584 | rotation | 19.993 | 19.750 | retuned + bank layout |
| a | 128,7168,3584 | graph4 | 19.512 | 17.811 | retuned + bank layout |
| a | 16384,3584,7168 | rotation | 596.533 | 595.738 | unchanged device .text |
| a | 16384,3584,7168 | graph4 | 586.798 | 587.066 | unchanged device .text |
| a | 16384,7168,3584 | rotation | 571.701 | 571.711 | unchanged device .text |
| a | 16384,7168,3584 | graph4 | 556.172 | 556.514 | unchanged device .text |
| b | 128,3584,7168 | rotation | 20.883 | 20.793 | unchanged device .text |
| b | 128,3584,7168 | graph4 | 19.035 | 19.055 | unchanged device .text |
| b | 128,7168,3584 | rotation | 20.380 | 20.010 | retuned + bank layout |
| b | 128,7168,3584 | graph4 | 19.594 | 18.002 | retuned + bank layout |
| b | 16384,3584,7168 | rotation | 588.413 | 587.315 | unchanged device .text |
| b | 16384,3584,7168 | graph4 | 578.777 | 578.159 | unchanged device .text |
| b | 16384,7168,3584 | rotation | 560.139 | 560.792 | unchanged device .text |
| b | 16384,7168,3584 | graph4 | 546.136 | 545.770 | unchanged device .text |

Some unchanged controls fluctuate upward even though independently compiled
machine code is identical. These are not gains, are not hidden, and cannot be
used to claim blanket whole-application non-regression. Only the changed
small-up key is selected on the evidence above.

## Correctness and final checks

-Strict FP32 reference at rtol=atol=.01,zero violating elements; bitwise
  same-geometry and before/after non-split outputs.
-Aligned offsets, padding0/16, input/output canaries, random/scaled/dyadic
  changed-input replays, concurrent streams, explicit non-current stream and
  allocation pressure.
-112 single-active-K exact probes cover all28 K tiles and offsets0/31/64/127.
  Cold capture, poisoned output, public non-unit-K/row-padding normalization.
-Public lookup M65/M127 selects original BK64; M128 selects BK128.
-CPU60 tests/4605 subtests pass, including permutation/DMA/phase-bank model,
  exact shape/config guard, caller-override rejection and padded lookup.
-Existing latent suite seed3573 passes; final merged-CSV/public check seed3579.
  The unchanged down-M128 atomic path keeps its documented strict
  random-reference limitation, not repaired or used as a false strict oracle.
-Separate-cache before/after binary identity and integrated PMC capture.
-Black/Ruff/diff checks. Current-head remote GPU CI not claimed complete.

No claim of exhaustive public-source coverage, all-shape optimality,
full-model speedup or universal hardware performance. The deliverable is
a pinned research ledger plus a narrowly validated source-derived optimization,
not a collection of unchecked links or benchmark cherry-picks.

# GLM-5.3 IQ2R optimization inventory

Updated 2026-09-26. This is a concise inventory of the documented optimization
attempts from the initial GLM integration through E391. Related scouts,
integration steps, and qualification runs are grouped together. Rejected
approaches remain listed so they are not mistaken for unexplored ideas.

**The overall goal is still unmet:** IQ2R must beat default ATOM/AITER MXFP4 on
matched same-node TP8 and TP4 serving sweeps, with TP4's relative gain at least
TP8's, correct model behavior, native dispatch, zero failed requests, and fresh
MXFP4 bookends. The official workloads are 1024/1024 and 8192/1024 at
C1/2/4/8/16/32/64/128/256. The final agentic workload uses real MTP acceptance.

The latest measurements and limitations are posted on
[ATOM #2335](https://github.com/ROCm/ATOM/pull/2335) and
[AITER #5728](https://github.com/ROCm/aiter/pull/5728).
The execution plan: `GLM53_IQ2R_DATAFLOW_EXECUTION_PLAN.md` owns the remaining
acceptance work. This source snapshot consolidates the measured E167/E172/E181 native implementation and the E190 router guard; the latter was not in the E180 r8 benchmark. Later E194–E196 scouts remain unselected.

The published [benchmark comparison](BENCHMARK_RESULTS.md) and [source/validation notes](README.md) accompany this inventory. Artifact paths in the tables identify the retained optimization workspace; their hashes are in [EVIDENCE_INDEX.json](EVIDENCE_INDEX.json). Large raw traces, generated binaries, and model data are retained outside these source repositories.

## Isolated single-GPU optimization, E199–E391

The user paused serving sweeps and requested one-rank synthetic MoE profiling.
The [single-GPU report](SINGLE_GPU_ANALYSIS.md) lists all attempts.
E261 at TP8 and E262 at TP4 combine adjacent9-bit index packing, exact
codebook-sign normalization and larger-token column batches. Stored payload
sizes and decoded FP8 weights are unchanged. Both improve their prior isolated
controls and remain behind MXFP4. E225's static-frontend restoration remains
the only production change from this isolated phase.

E255 independent M32 gate tasks, E256 alternate LDS codebook layouts, E257
per-token completion counters and E263 direct global codebook reads were
unselected or rejected. E258 established a useful two-batch down/reduction
order at4096 tokens. E259/E260 reduce sign/index instructions; E261 confirms
signed-book exactness and shows that deferring reductions loses the batching
gain. E264 N-atom lookahead is exact but has small mixed gains; no broad selection.
E265 completed exactly without a broad gain. E266 verifies repeated MFMA16/32
primitive equivalence. E267 MFMA32 and E268 larger activation batches were exact but did not win.
E269/E270 gate grids and E271 TP4 MFMA32 down show small/mixed gains.
E272/E273 changed-order diagnostics have no broad speedup. E274 forced residency
spills regress. E275/E276 extend exact packing to small-token kernels; E277
scale-byte/hint variants do not fix the cliff. E278/E280 restore small-gate
residency and improve timing through bounded sign lifetimes and read batching.
E279/E281 down extensions and E282 dense scheduling are unselected. E283
down scheduling has TP8/TP4 correctness failures and is ineligible. E284
passes30 E280 task-boundary/shape cases with exact intermediate values/scales. Earlier attempts and numerical-reference limitations remain in the
single-GPU report. Serving sweeps remain paused.


| Experiment | Brief explanation and outcome |
|:---|:---|
| E285 | Qualify unchanged E280 on16 actual captures and six real layer/rank weight slices:32 cases/256 changes, exact final/intermediate values. |
| E286 | Assign down columns to independent waves, removing cross-wave partial sums. Exact, useful at TP8 M16 spread; TP4/hot mixed. |
| E287 | Instantiate static ballot routing for M16. Exact; saves about0.3–0.4µs. Invalid-route/stable-order tests pass under E291. |
| E288 | TP4 four/eight-token adaptive fused down. Exact but spread regressions outweigh hot gains; unselected. |
| E289 | Batch nine independent reduction reads. Exact; combined TP8 M16 spread48.88µs beats fresh MXFP449.86µs. TP4 comparison fails unchanged-baseline tolerance. |
| E290 | Widen route9 output tiles to96/192 columns. Exact, slower at TP8 M4; rejected. |
| E291 | Unchanged E289 combination passes30 boundary/shape cases,240 changing steps and45 frontend tests; captured qualification remains. |
| E292 | Map48 fused-down output columns to48 reducing lanes. Exact and modestly faster at TP8/TP4 M4; TP8 hot gap remains. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E293 | Partition small input quantization across two/four column groups. Exact; original timing drift prevents qualification. E302 supplies stable combined results. |
| E294 | Reuse M4 down weights across compact expert tasks. Exact N16/N48 variants regress on hot routes; unselected, scout drift retained. |
| E295 | Batch independent reads in dense chunk reduction. Exact, no broad improvement over E261; unselected. |
| E296 | Expand active gate weights to temporary FP8 each call. All preparation counted; exact but slower with or without prefetch; rejected. |
| E297 | Combined policy passes 30 TP8/TP4 cases, 240 changing steps, 720 exact checks and 45 frontend tests. |
| E298 | Read M4 codebooks directly from global memory. Exact but down time nearly doubles; rejected. TP4 compiled, not measured. |
| E299 | Combined policy passes 32 real-capture/TP cases and 768 exact checks using six actual weight slices. TP4 reuses TP8 inputs; M16 unqualified on captures. |
| E300 | Retain dense hot-route down weights in registers and fuse ordered route reduction. Dynamic guards/fallbacks timed; exact but much slower. |
| E301 | Store each persistent expert result once; map routes at reduction. Exact; recovers part of E300 regression but remains unselected. |
| E302 | One-second graph warmup yields stable TP8 combined comparisons. Five of six small-token cases beat fresh MXFP4; M4 hot remains +8.5%. |
| E303 | Swizzle persistent-down LDS output columns. Bank conflicts fall 96.5%, time does not improve; exact, zero scratch, unselected. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E304 | Stage route metadata once per persistent M32 tile. Exact; vector reads fall 62%, but total time still trails E261 badly. Unselected. |
| E305 | Retire persistent down accumulators one N16 result at a time. Exact, fewer registers, no meaningful gain; unselected. |
| E306 | Disable M16-half unrolling and request two-block compiler residency. Both variants use 112 registers and about 27% measured occupancy; no gain. |
| E307 | Exact N64 down records and N256/N512 tiles at TP8. N512 gives mixed small gains; no broad selection. First compile failure preserved. |
| E308 | TP4 quad down with N512/batch4. Exact; 0.7–3.5% better than E262 in all four dense cases, still 9–34% behind MXFP4. |
| E309 | Use N64 quads in M4 token-owned down. Exact but slower at spread/hot routes; unselected. |
| E310 | Load only a compact N16 quad record, reuse it across M4, and share a 384-workgroup grid with the N64 fallback. Exact; no gain over the selected path. |
| E311 | Qualify E308 and E262 at 39 TP4 dense shape/routing cases, 312 changes and 936 exact checks. Correct the raw-row checker using verified route permutations; no tolerance change. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E312 | Fuse route preparation and BF16 quantization into gate. Exact, but repeats work and regresses; unselected. |
| E313 | Quantize unique inputs once before route-fused gate. Exact; partial recovery, still slower than separate frontend. |
| E314 | Replace each gate workgroup histogram/scans with one-wave stable sorting. Exact but fusion still slower; excessive drift retained. |
| E315 | Use one-wave stable sorting in standalone frontend. Exact; useful M4 gain at two column partitions. |
| E316 | Cooperative 32-lane scale-group quantization. Exact, slower; drift makes timings provisional. |
| E317 | Pair native FP8 conversions and pack bytes directly. Exact, small additional gain; early input loads regress. |
| E318 | Qualify E315/E317 input/intermediate/final outputs at M4 TP8/TP4, plus 240 CPU-route/native-quant frontend checks. |
| E319 | Complete each route wave’s private codebook without the initial workgroup barrier; test direct VMEM-to-LDS copy. Exact, modest gains; TP8 timing provisional, TP4 stable. |
| E320 | Qualify E317 frontend with 16 actual captures and six real TP8/TP4 weight slices; exact intermediate/final results. |
| E321 | GPU-guarded two/four-token weight reuse with separate reducing waves. Two-token candidate reaches M4 TP8 hot parity and beats spread MXFP4. |
| E322 | Qualify wave-private codebook and token reuse/fallback on changing guards, reordered slots and invalid IDs; exact. Correct reporting-only invalid bincount without changing runtime/tolerances. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E323 | Profile original M32/M64/M128/M256 MoE. Down dominates M64/M128 spread despite lower DRAM bytes; padding and decode instruction volume are material. Excessive drift remains visible. |
| E324 | Qualify E321 TP8 M4 reuse and E319 TP4 M4 codebook completion on real slices/captures. Exact, 768 checks; no new MXFP4/serving comparison. |
| E325 | Use existing independent-output-wave down kernels at M64/M128. N384 halves sparse MFMA work and nearly matches MXFP4 down; correct intermediate checks for atomic-sort permutations. |
| E326 | Overlap route sorting and identity input quantization in one launch; pair FP8 conversions. Combined policy beats MXFP4 at M64 spread/hot and M128 spread; M128 hot still behind. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E327 | Normalize codebook signs and repack adjacent 9-bit down indices; compare final-reduction variants. E289b9 is best among these arms, but M128 hot remains behind. Excessive bookend drift is retained. |
| E328 | Batch independent packed gate codebook reads at M32. Batch4 helps spread modestly; register weight-record prefetch regresses. All numerical checks pass; unstable timings remain provisional. |
| E329 | Direct gate epilogue removes the shared gate/up transfer and barrier while retaining BF16 rounding. LDS drops by 4 KiB; the hot gap remains. |
| E330 | Split M32 gate tasks into M16 subtiles, testing workgroup counts and ordering. M16 improves hot routes but badly regresses spread; no route-specific CPU selector is selected. |
| E331 | Use E326 frontend plus TP4 N256/N512 quad down tiles. Four ordered K128 sums remain exact. N256 at M64 and N512 at M128 beats spread MXFP4 but leaves hot 3–5% behind. |
| E332 | Transpose static sign bits into four planes so shift plus AND/OR replaces nibble extraction and multiplication. Same bytes/decoded values, 16–19% fewer gate VALU instructions. Gate-only gains; down-only unselected. |
| E333 | Adapt packed/direct/sign-plane gate to TP4. E333 sign-plane gate with fixed E331 down policy beats spread MXFP4 by 14–15%; hot remains about 4% behind. |
| E334 | Ablate component-major gate reduction storage and all-eight-wave M32 epilogues. Component storage reduces LDS instructions without lowering total bank conflicts. The combined version helps; plain parallel reduction regresses. M64 hot timing drift is retained. |
| E335 | Emit compact M16 gate tasks while retaining M32 down tasks. Stable fixed grid2 beats M64/M128 spread and M64 hot MXFP4; M128 hot remains +4.0%. All 504 checks and independent task-list checks pass. The second task-prefix scan costs about 1.7 µs. |
| E336 | TP4 compact M16 gate tasks with unchanged N256/N512 down shape policy. All 504 exact/task-list checks pass. Stable spread gains of 16–19%; hot gaps remain 3.3% at M64 and 1.5% at M128. |
| E337 | Extend medium TP8 candidates to M32/M256 with explicit short-K/scheduled-large entry wrappers so packed tensors reach only their matching decoder. Pending GPU qualification. |
| E338 | Combine component-major compact M16 gate reduction and one packed task-prefix scan. Six of eight stable medium-token cases beat fresh MXFP4; TP8 M128 hot +1.4%, TP4 M64 hot +0.3%. TP4 r1 hot drift is retained; frozen selected-arm r2 qualifies with max 0.61% drift. |
| E339 | Frozen E338 passes 1,152 exact arm checks across TP8/TP4 with a new weight/input seed, zeros, 16x inputs, reordered slots, and spread/hot/skew/mixed routes. No performance or actual-weight claim. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E337 / E344 | The M256 extension exposed changed down rounding: two independent K128 partial sums differed from the original sequential accumulator. Preserve the failure; E344 restores the ordered chain and passes 504 checks without changing tolerances. Spread cases improve; hot gaps remain. |
| E340 | Independent route-reduction wave grouping remains unselected; all attempted results are retained. |
| E341 | Aggregate periodic-route histogram/scatter atomics. Exact at TP8 but slower than E338 in all four M64/M128 cases; TP4 remains compiled and unmeasured. |
| E342 | Parallel task-record emission improves TP8 medium cases modestly. TP4 r1 drift and the r2 dispatcher-edit mistake are preserved. Correctly asserted r3 qualifies all rows: both spread cases and M128 hot beat MXFP4; M64 hot remains 2.0% slower. |
| E343 | Transpose sign bits into static planes in the dense TP4 gate and group four independent codebook reads. Exact and modestly faster; gate VALU counts fall, while total all-wait cycles do not improve against E308. |
| E345 | Apply sign planes and grouped codebook reads to TP4 dense down while fixing the E343 gate. Original-IQ2R timing drift invalidates different r1/r2 rows. Prospective r3 retains original IQ2R for exact correctness/native checks and times MXFP4/candidate only: every row qualifies, but dense gaps remain 6.1–28.0%. |
| E346 | Fuse shared-expert down into final route reduction, retaining BF16 rounding and route order. All 504 poisoned-scratch checks pass. Scattered output ownership nearly doubles physical traffic at M4096; much slower, rejected. |
| E347 | Combine compact component-major M16 gate, E342 frontend and exact M256 down. R1 timing drift is preserved. Frozen r2 qualifies all rows: M32 spread/hot and M256 spread beat MXFP4; M256 hot remains7.4% slower. |
| E348 | Transpose shared results locally and give reduction lanes contiguous eight-column reads. All 588 checks and timing rows qualify; traffic falls sharply versus E346, but every new fusion arm still loses to E345. Rejected. |
| E349 | Frozen E347 passes576 extended checks with new seed, eight changes, zeros, large inputs, slot permutations and four routing patterns. Native gate/frontend/ordered-down and zero scratch verified; no performance claim. |
| E350 | Explicit four-read codebook groups and one-atom lookahead in TP4 dense gate. All504 checks and timing rows pass, but all candidates lose. At M4096 hot, four-read lookahead lowers LDS waits yet measured occupancy roughly halves and gate time rises270→358µs. Unroll1 stays above256 registers and loses further. |
| E351 | Two-read dense gate lookahead holds static allocation at256 registers. All420 checks and all timing rows pass. Improves E345 control by0.3–1.2%, with occupancy retained and DRAM essentially unchanged; still4.5–27.4% behind MXFP4. |
| E352 | Compact TP8 codebook schedule ablation: all504 checks and timing rows pass. Four-read lookahead is fastest across all four cases, saving0.7–2.5% versus control. Spread wins; M128/M256 hot remain0.6%/4.5% slower than MXFP4. LDS waits rise slightly, so this is not a fewer-LDS-waits claim. |
| E353 | Replace TP4 down shared activation cache with direct register loads, M32/M64 and optional one-K128 activation lookahead. All588 checks and timing rows pass, but every variant loses. Global-read instruction count rises sharply while physical DRAM bytes change modestly; rejected. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E354 | Widen batched route reduction to vector8/batch3. Exact; reduces metadata shuffles, global reads and VALU instructions. Retained, with small TP8 full-block gains; hot parity remains open. |
| E355 | Cache activations cooperatively and reuse each K128 fragment across four output atoms. M64 without extra lookahead improves dense TP4; M64 lookahead crosses 256 registers and loses. All 588 exact checks pass. |
| E356 | Apply four-read codebook lookahead to compact TP4. All 504 checks pass and all four M64/M128 spread/hot rows beat fresh MXFP4. |
| E357 | Frozen E354 TP8 expanded correctness/native qualification:1152 exact checks across M32/M64/M128/M256, four routing patterns, new seed and eight input/route changes. |
| E358 | Reuse the dense activation/weight pipeline at M32 TP8. Exact, but no broad gain; both hot timing rows exceed 3% drift. Lower traffic and waits do not compensate for low useful workgroup occupancy. |
| E359 | MFMA32 with bounded current/next K128 records passes 672 exact checks and stable timing, but every variant loses to E355. More VALU and global-read instructions offset wider matrix instructions. |
| E360 | Frozen E356 TP4 expanded correctness/native qualification:576 exact checks at M64/M128 with spread/hot/skew/mixed routes and eight input changes. |
| E361 | Qualify frozen E354/E356 on three real layer/rank weight slices at TP8 and TP4: 6480 exact checks. capture_tiled repeats saved M4/M8 rows; remaining patterns use generated inputs. Operator correctness only. |
| E362 | Enable XCD remapping at M256 with grid multiplier2. Both patterns improve, all 336 exact checks pass; hot falls to 64.71us versus fresh MXFP4 62.76us (3.1% gap). Actual grid geometry and zero scratch verified. |
| E363 | Combine E351 two-read gate and E355 M64 down: 504 exact checks, all rows stable, 0.6–3.6% faster than E345. Dense MXFP4 gaps remain 3.3–23.5%. Incorrect r1 tensor selection is preserved; corrected fresh r2 qualifies. |
| E364 | Pair MFMA32 records and reuse activations: 588 exact checks and all stable rows. M32 improves one hot case but loses on the other three; M64 lookahead crosses 256 registers. No broad selection; initial compilation error retained. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E365 | One-ahead register weight records and gate grid3. R1 hot drift 3.56% is provisional; narrowed r2 passes 168 exact checks and stable timing. Register/grid2 improves hot versus its matched control; grid3 is unselected. |
| E366 | Pair partial reads and emit four FP8 output bytes in one dword store, preserving all rounding. 168 exact checks and stable timing; independently useful. Four byte stores become one dword and data-permutation instructions fall. |
| E367 | Combine register-record loading with vector gate output. 252 exact checks; qualified spread loses to both components and hot drift is 4.46%. No additive gain established; unselected. |
| E368 | Extend compact TP4 to M256 with a matching quad decoder preserving four sequential K128 accumulations. 168 exact checks and stable rows. Remapping helps both patterns; spread beats MXFP4 but hot remains 29.7% behind. |
| E369 | Adapt ordered quad down to TP8 K256 with N256/N512 outputs and matching tile-major reduction. 210 exact checks and stable rows. N512 improves both patterns against the matched triplet control; hot remains 1.7% behind that run's MXFP4. |
| E370 | Reuse gate weights over M32 rows, separately testing split-K read4 and full activation/weight pipelines. 252 exact checks. All new variants lose qualified spread; hot drift 3.19% prevents qualification. |
| E371 | Hold N512 down arithmetic/layout fixed and compare persistent grid8/4/2. 210 exact checks, both rows stable. Grid4 improves hot 1.9% for a 0.24% spread cost; its matched MXFP4 hot gap is 2.3%. |
| E372 | Use four K waves with two separate K768 accumulators each, retaining the original eight-part ordered sum. 252 exact checks, zero scratch. Both M16/M32 versions lose qualified spread; MXFP4 hot drift 3.61% is retained. |
| E373 | Adapt vector8/batch3, vector8/batch9 and vector16/batch3 reduction to N512 tile-major output. R1 hot drift 3.47% is retained. Narrowed r2 passes 168 exact checks with stable rows; vector8/b3 improves hot 1.4% for a 0.28% spread cost. Native E308 already overlaps nine payload loads; no missing-load-overlap claim. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E374 | Move first codebook lookups before activation completion. 210 exact checks and stable rows, but slower. The unchanged Pipeline4 arm independently qualifies a TP4 M256 tradeoff: hot improves 9.7% versus compact control while spread loses 10.6%. |
| E375 | Pass actual active M16 fragments to the dense M64 gate. 336 exact checks and stable rows. Spread MFMA counts fall 36.6%/13.1%, but only M1024 spread improves (1.2%); other rows regress. Keep dense control. |
| E376 | Choose compact/Pipeline4 work per actual M32 task count in two disjoint launches. 315 exact checks include 16/17-row boundaries. Qualified spread loses 19.9%; hot/boundary drift remains provisional. Reject extra sparse pipeline launch. |
| E377 | Halve dense N atoms per wave and load only the matching packed-record half. 336 exact checks and stable rows. Registers fall 256 to 182 with identical MFMA counts, but global-read instructions rise about 62% and all cases regress 11.7–24.1%. |
| E378 | Visit route slots across adjacent tokens; separately test guarded uniform-wave atomics. Corrected r2 passes 315 exact checks and all rows stabilize. Plain token-major visits improve their control 0.8–1.7% and are retained for qualification; hot remains 4.3% behind fresh MXFP4. Setup/preparation failures are preserved. |
| E379 | Widen dense down to N1024 by reusing each M64 activation cache across four sequential N groups. 420 exact checks and stable rows; unchanged 220 registers and identical MFMA counts. Retain grid8 at M4096 only (0.2–0.9% gains); M1024 stays N512. Dense MXFP4 gaps remain about 23%. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E380 | Carry four next-K codebook reads across the compact gate loop without deeper weight prefetch. 252 exact checks, max drift 2.63%; improves E378 by 2.9–4.0%. Retained at TP8 M256; hot remains behind MXFP4. |
| E381 | Keep the shared codebook while a persistent workgroup stays on the same expert. 252 exact checks and stable timing. Global-read/LDS instruction savings match the task model exactly; 0.5–2.6% gains, with small spread/mixed differences. |
| E382 | Move TP4 compact gate records from LDS to registers and add cross-K lookup scheduling. 378 exact checks, max drift 2.91%; cross-K improves E368 by 2.7–4.5%. Register-only loses mixed; codebook reuse adds no consistent benefit. TP4 M256 hot remains 23.4% behind MXFP4. |
| E383 | Remove the pre-partial-store barrier inherited from the unused LDS weight cache in the register-only gate. 252 r1 checks pass but hot drifts 3.31%. Independent r2 passes 72 new checks and stable bookends, improving E381 by 0.7–2.2%; TP8 M256 hot remains 1.5% behind. |
| E384 | Qualify E381/E383 TP8 M256, E382 TP4 M256 and E363/E379 TP4 M1024/M4096 on three real layer/rank slices each. All 4,320 exact changing graph/eager checks, input/intermediate scales, native dispatch and zero scratch pass. Small captures are tiled to larger shapes; no serving or timing claim. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E385 | Three-slot activation ring with actual selective VM retirement passes 420 checks and stable bookends, but both variants lose. LDS waits fall while VALU instructions rise about 12%. Compiler-drain r2/r3 and scratch-descriptor r4 attempts are preserved without GPU timing. Rejected. |
| E386 | Direct codebook byte addressing removes one VALU instruction per atom. All 420 checks and timing rows pass. Retain at TP4 M4096 only for 0.3–0.6% gains; M1024 is mixed and paired 64-bit sign shifts are unselected. Dense gaps remain about 22%. |
| E387 | Transfer direct byte addresses to selected compact TP8 M256. All 252 checks pass; VALU counts fall about 2.8%, but qualified spread regresses. Hot drift 3.2529% remains provisional at the unchanged 3% ceiling. Unselected. |


| Experiment | Brief explanation and outcome |
|:---|:---|
| E388 | Modern sign-plane/byte-address MFMA32 gate passes 336 exact checks and stable bookends, but loses 5.4–10.5%. Wider matrix instructions reduce counts without reducing arithmetic; waits rise. Failed padded fixture retained. |
| E389 | Pair exact K64 records into three K128 loads. All 420 checks and stable bookends pass; VMEM instruction savings recover 1.3–2.9% versus E388, but remain slower than the selected kernel. Unselected. |
| E390 | Double gate M reuse to 128 rows and retain N64 per wave. All 336 checks pass, with zero scratch and stable bookends. Register allocation rises to 446, occupancy roughly halves, padding grows on spread routes, and whole-MoE regresses 12.2–59.3%. Rejected; build failures preserved. |
| E391 | Qualify frozen E386 at TP4 M4096 on three real layer/rank slices, five route patterns and eight input changes. All 1,080 exact checks pass, including input/intermediate/final results, native dispatch and zero scratch. Operator qualification only; no timing or whole-model-quality claim. |


## How to read the outcomes

- **Selected** means used by the current experimental candidate within its
  dispatch bounds. It does not mean the complete serving goal has passed.
- **Historical retained** means a change was retained at that stage; later
  formats, kernels, or dispatch policies may have superseded it.
- **Rejected / unselected** means measurements or correctness did not justify
  enabling it. Exact outputs alone are not a performance qualification.
- **Pending** means the required evidence is incomplete.
- Kernel and whole-MoE gains below compare with the named earlier IQ2R path,
  unless explicitly stated otherwise. They are not MXFP4 serving speedups,
  and gains from successive experiments must not be added together.
- Early logs sometimes use `M` for routed rows; later scouts often use it for
  input tokens or tile height. This inventory spells out the relevant meaning.
  C always denotes serving concurrency. A wave has 64 lanes; wave counts and
  workgroup sizes vary between these kernels.

## What the current experimental candidate uses

| Change | Purpose and current scope | Main experiments |
| --- | --- | --- |
| Fused 257-expert representation | Encode the shared expert as expert 256 and append its always-on ninth route, removing the separate shared-MLP/add boundary. | E039–E041 |
| Quantize-once and indexed inputs | Avoid quantizing/copying the same token independently for every selected expert; use the bounded grouped or indexed frontend appropriate to the shape. | E048–E051, E058, E063–E064 |
| Four-atom gate packing and sparse row handling | Align gate/up work with the 32-value activation-scale group and omit unused row slabs. Repack existing records without requantizing weights. | E069, E071, E076, E079–E081 |
| Wait3 / register-load gate scheduling | Overlap independent loads and shorten decoded-weight lifetimes; choose the qualified small-token schedule by shape. | E089–E096, E106 |
| Specialized down kernels and ordered route reduction | Reduce inactive-wave work, redundant transfers, and serialized route phases while preserving the established rounding and reduction order. | E094, E098–E111 |
| Direct gate epilogue | After reducing partial sums, perform SwiGLU and FP8 quantization without the intermediate gate/up LDS round trip. | E116, E118 |
| Ballot-based frontend, with static C4/C8 bounds | Replace serial predecessor scans with wave ballots; omit chunks that cannot contain predecessors at these fixed token counts. | E139–E140, E167 |
| Bounded C128 task/XCD ordering | Change the order in which compute dies encounter expert tasks to improve locality. | E150–E151 |
| TP8 prefill activation/weight lookahead | Hold the next compressed weight record, activation fragments, and scales in registers while computing the current K block. Enabled only on qualified prefill shapes. | E122, E127, E152 |
| Compact C8 down scheduling | Assign up to 12 expert tasks to simultaneous waves, reducing repeated phase work for low-population TP8 C8 routing. | E172, E180 |
| Native invalid-route protection | Bound expert addressing and initialize invalid-route results in the small-token down path. This is a safety fix, with full-model qualification still required. | E181 |

The E190 top-k guard is **queued for full-model validation** and is installed in
the planned fresh comparison runtimes. It is not part of the already-measured
E180 r8 runtime. General codebook batching, deeper prefetch rings, unconditional
grouped-token down, TP4 dense lookahead, and the MTP M6/M12 extension are not selected.

## Initial architecture and topology work

The source for these early, partly unnumbered trials is the dated
analysis log: `GLM53_IQ2R_ANALYSIS.md`.

| Attempt | Brief explanation | Outcome |
| --- | --- | --- |
| Cooperative/persistent GLM GEMMs | Replace generic launches sized for maximum task capacity with cooperative workgroups that process actual expert work. This avoids large numbers of empty workgroups at decode. | Retained foundation; later kernels specialize the same GLM shapes further. |
| Fused EP expert-offset mapping | Apply global-to-local expert-ID offsets inside existing routing kernels, removing a separate chain of subtraction, comparison, selection, and casts. | Retained for the EP-capable path; the current benchmark target uses no EP. |
| Replicated full experts on every TP rank | Test avoiding EP while keeping complete expert weights on each rank. | Rejected: correct execution but higher memory use and slower serving at every probe. |
| True TP weight sharding | Give every rank all logical experts but only its gate/up N slice and down K slice. TP8 uses gate K6144/N512 and down K256/N6144. | Retained basis of the TP8/no-EP path. |
| Specialized 256-expert sorter | Replace a serial prefix/task-construction bottleneck with block scans while preserving sorting semantics. | Retained historically; extended to the fused 257-expert representation. |
| Static EP8 placement | Reassign experts using a route-count objective to balance ranks. | Rejected: fewer projected routes did not translate to lower full-model latency. |
| Disable shared/routed stream overlap | Test executing the old, separate shared and routed experts on one stream. | Rejected for that old architecture; the later fused-257 design removes the separate shared branch entirely. |
| MXFP4 shared expert in the old mixed-format path | Quantize the separately executed shared MLP to MXFP4 while keeping routed experts in IQ2R. | Historical retained improvement; superseded by the calibrated IQ2R shared expert in E039. |
| Early fused biased-sigmoid router | Fuse plain-GLM routing for the 256-expert/top-8 contract. | Rejected: native checks passed, but serving regressed about 15–31% in that EP8 experiment. |
| Extend one-task-per-route construction to larger batches | Avoid sorting by assigning individual routes directly to GEMM tasks. | Rejected: concentrated routes lost expert reuse and produced many more tasks. |

## Routing, input quantization, and reduction indexing

| Experiments | Brief explanation | Outcome / evidence |
| --- | --- | --- |
| E001 | Quantize each token once and broadcast FP8 values/scales to its eight routed destinations. | Early global policy rejected: isolated savings did not produce consistent serving gains. Later shape-bounded versions were revisited. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E003 | Write activation scales directly in the tile16 layout expected by GEMMs. | Rejected as a global default: exact and faster in isolation, but inconsistent serving performance, including C4–C32 regressions. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E007 | Specialize indexed route reduction for GLM's fixed output width; use simpler tile indexing and broadcast route metadata within each wave. | Historical retained specialization. Early prose in the log is stale; its completed record identifies the retained result. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E040 | Build specialized grouped tasks for the new 257-expert/top-9 boundary. | Historical retained improvement; E040 serving beat E029 at the tested points. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E045 | Fuse low-token routing/preparation using direct route tasks. | Rejected: frontend improved about 52%, but whole-MoE performance regressed about 16% because expert grouping was lost. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E048–E050 | Quantize once and broadcast to nine sorted routes, combined with the qualified high-prefill task policy. | Retained for bounded prefill shapes; exact-route checks and historical serving transfer passed. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E051 | Fuse route permutation with quantize-once broadcast while preserving grouped 16-row expert tasks. | Retained foundation of the small-token frontend; exact whole-MoE capture gains around 4–6%. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E058–E060 | Extend exact quantize-once scatter/broadcast to the C256 decode boundary and combine it with the selected down kernel. | Historical retained composition; standalone frontend gains were checked again at the complete boundary. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E063–E064 | Keep quantized activations token-major and let the gate gather the needed token through indices, avoiding route-expanded activation materialization. | Bounded integration retained after fixing the inherited down synchronization defect. E063: `experiments/e063/README.md` |
| E074 | Fuse router/task/input preparation more aggressively. | Rejected: exact results, but all ten whole-MoE screens regressed roughly 22–33%. E074: `experiments/e074/README.md` |
| E139–E140 | Distribute predecessor/rank queries across waves and use ballots/popcounts instead of serial route scans. Preserve compact tasks, stable order, and quantization. | Selected at qualified C4/C8/C16 shapes; all 48 captured cases improved. E140: `experiments/e140/README.md` |
| E163 | Emit fixed route-task slots and remove histogram/prefix compaction work. | Rejected: all eight TP8 route-pattern scouts regressed despite fewer frontend scans. E163: `experiments/e163/README.md` |
| E167 | Compile C4/C8-specific ballot bounds, checking only one/two 64-lane predecessor chunks instead of three. | Selected after native and 32 captured-input checks; whole-MoE gains about 1–2% depending on TP and shape. Plan: `GLM53_IQ2R_DATAFLOW_EXECUTION_PLAN.md` |

## Gate tiling, packing, and epilogues

| Experiments | Brief explanation | Outcome / evidence |
| --- | --- | --- |
| E002 | Try existing AITER/Redline fused gate-up, SiLU, and activation quantization at GLM TP8 geometry. | Rejected: neither available implementation beat the separate path, and one changed the accepted numerical contract. Later E066/E069/E116 redesigned the implementation. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E008 | Force gate accumulators into AGPRs to reduce general vector-register pressure. | Rejected: correct but slower. Completed record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E009 | Decode and multiply one weight atom at a time so decoded fragments die earlier. | Historical retained scheduling variant; later quad kernels have their own schedules. Completed record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E012 | Reuse decoded weights across more gate rows using existing large32/large64 families. | Rejected for the tested GLM application: slower and not universally bit-exact. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E013 | Reduce large-row gate workgroups to two waves to expose more workgroups. | Rejected: greater launch parallelism did not offset the longer work per wave. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E014, E017 | Reuse each decoded gate weight across two 16-row slabs and add a bounded automatic crossover. | Historical retained high-row family; smaller shapes needed different policies. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E018 | Reduce that cooperative gate from eight waves to four. | Rejected: higher theoretical residency did not deliver robust latency gains. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E041 | Extend cooperative two-row gate/down specialization to the fused-257 decode geometry. | Historical retained integration; C256 serving improved versus E040. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E044 | Try a broader fused gate/SwiGLU/quantization kernel at the fused M2304 geometry. | Rejected: the full path regressed about 20%. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E066 | Align four raw gate/up atoms with one 32-value activated FP8 scale group; compare 16- and 32-row output tiles. | Exact, but the initial broad implementation was slower. It motivated the more efficient E069 weight packing. E066: `experiments/e066/README.md` |
| E067 | Skip the second 16-row slab when an indexed expert task has no rows there. | Standalone version unselected: synthetic gains were not consistent across real captures. Revisited successfully inside the quad kernel. E067: `experiments/e067/README.md` |
| E068 | Compute one output atom per workgroup to expose more independent gate work. | Broad small-token policy rejected; spread C4 and larger cases regressed. E068: `experiments/e068/README.md` |
| E069 | Repack the existing compressed records into exact four-atom groups, avoiding redundant reads caused by fitting four atoms into two three-atom records. | Retained foundation of the quad gate. No weight requantization. E069: `experiments/e069/README.md` |
| E070 | Store single-atom records in compact planes so narrow-output workgroups do not fetch whole triplets. | Unconditional C4 policy rejected: early captures improved but late captures regressed; M1 remained an isolated lead. E070: `experiments/e070/README.md` |
| E071, E076 | Apply device-side live-row guards inside the better-aligned quad gate, including activation loads, MFMA, and epilogue work. | Selected after exact captures and ownership/integration checks. E071: `experiments/e071/README.md` |
| E073, E080 | Spread the existing K partitions across separate workgroups, then combine their partial sums. | Split4 qualified on real C4 captures; retained as an alternative, while the stronger one-kernel E079 path was selected for serving. E080: `experiments/e080/README.md` |
| E079, E081 | Use the quad fused gate directly for 1–16 input tokens with route-major FP8 input and no empty second slab. | Selected foundation of later small-token scheduling work. E079: `experiments/e079/README.md` |
| E104 | Bound epilogue stores and scratch to live rows and adjust launch limits. | Rejected: no consistent improvement; a register-allocation threshold could halve residency. E104: `experiments/e104/README.md` |
| E116, E118 | Reduce the K partials, keep reduced gate/up values in registers, exchange neighboring gate/up lanes, apply SwiGLU, and write FP8 values/scales directly. Removes an extra LDS round trip and barrier; the cross-wave partial reduction remains. | Selected after 17 native tests and 40 real captures. Whole-MoE gains about 1.2–2.5% over E111 by small-token shape. Implementation: `experiments/e116/generate.py` |
| E119, E124–E125 | Extend gate packing, N tiles, down K tiles, and bounded dispatch to TP4 geometry. Diagnose differences caused by changing the accumulation order. | Native TP4 support retained. Exact cooperative-reference tests and FP64 diagnostics do not replace model-quality qualification. E124: `experiments/e124/README.md` |
| E134 | Read four partial-result components together and remap epilogue lanes to reduce repeated LDS reads. | Rejected: additional shuffles and register pressure outweighed the lower read count. E134: `experiments/e134/README.md` |
| E138 | Split one activated output tile across two workgroups; compare a last-arrival epilogue with a separate quantization kernel. | Both variants rejected: signaling or extra-kernel costs caused material spread-route regressions. E138: `experiments/e138/README.md` |
| E148 | Extend the direct epilogue to sparse M32/M64 expert tasks. | Unselected: exact TP8/TP4 tests passed, but whole-MoE changes were small and inconsistent. E148: `experiments/e148/README.md` |

## Load scheduling, codebooks, and the fuller pipeline

| Experiments | Brief explanation | Outcome / evidence |
| --- | --- | --- |
| E010 | Read codebook entries through the global/read-only cache instead of staging the codebook in LDS. | Inconclusive historical scout; not promoted as the current default. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E011 | Propose activation lookahead for the earlier three-atom gate. | The original entry remains not-run; later GLM-specific lookahead was actually tested in E062 and E122/E127. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E062 | Test an exact earlier gate lookahead configuration on a C256 capture. | Rejected: complete time increased from 95.18 to 134.00 microseconds in the inspected case. Audit: `GLM53_IQ2R_REPOSITORY_AUDIT_20260924.md` |
| E089–E090 | Specialize down K/N, load compressed records with uniform register addressing, and defer scale replication. | Selected through E096; actual whole-MoE gain about 2.09% over E085 for the best combination. Naive register addressing was much slower and was discarded. E090: `experiments/e090/README.md` |
| E091, E093, E095–E096, E106 | Replace a full VMEM drain with a dependency-correct `vmcnt(3)` schedule, or use uniform register loads and six-iteration unrolling. Compare complete gate/down combinations. | Selected shape-dependent scheduling. Wait3 allows three younger activation requests to remain outstanding while older weights are ready; it is not a universal wait constant. E091: `experiments/e091/README.md` |
| E107 | Keep two or three future compressed records in a deeper register prefetch ring, including deferred-scale variants. | Rejected: spread-route latency increased; several variants reduced residency from two workgroups per CU to one. E107: `experiments/e107/README.md` |
| E117 metadata-first reads | Issue metadata first and reconstruct indices/scales while another weight-record read finishes. | Rejected for the tested quad layout: no useful gain; the combined variant also increased register pressure. E117: `experiments/e117/README.md` |
| E117, E120 codebook batching | Issue independent codebook reads together before sign application; test four-, eight-, and sixteen-read batches and compiler scheduling barriers. | Exact native/capture checks passed, but the global policy failed its selection rule. Whole-MoE gains were below 0.5% by shape, with a small C2 regression. Remains disabled. E117: `experiments/e117/README.md` |
| E122 register-weight pipeline | Keep one pending compressed weight record in registers instead of round-tripping it through the LDS weight cache. | Useful bounded dense-kernel candidate, integrated and qualified in E127. E122: `experiments/e122/README.md` |
| E122 full activation/weight lookahead | While decoding and multiplying K block `i`, issue loads for block `i+1`'s compressed weights/metadata, FP8 activations, and scales. Keep the pending fragments in registers. | M32 broadly rejected because register pressure reduced residency; M64 required actual-prefill qualification. This is one-block lookahead, not an arbitrarily deep ring. Generator: `experiments/e122/generate.py` |
| E127, E152 | Integrate the full pipeline and restrict it to populated TP8 prefill tasks. | Selected at supported M1536 and M2048–4096 input shapes. Fifteen real prefill captures passed; per-shape whole-MoE gains were 5.51–9.46%. TP4 dense lookahead remains disabled. E152: `experiments/e152/README.md` |

## Down projection, route fusion, and work assignment

| Experiments | Brief explanation | Outcome / evidence |
| --- | --- | --- |
| E015 | Reuse wide down-projection weights across two row slabs in one workgroup. | Rejected: extra live accumulators caused scratch allocation and slower execution. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E016 | Narrow the two-row down tile to 48 columns to preserve weight reuse with fewer live accumulators. | Historical retained no-scratch family for bounded larger-row tasks. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E027 | Start the old MXFP4 shared expert on an alternate stream and fuse its final add into indexed IQ2R route reduction. | Historical retained intermediate architecture; superseded by fused-257 IQ2R. E028 only packaged its benchmark. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E039 | Calibrate/encode the shared expert into IQ2R and execute its ninth route inside the common MoE boundary. | Retained architecture; removed the separately exposed shared MLP and add. This changed the shared-expert representation and required its own numerical validation. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E053, E056, E059–E060 | Select a `large32/grid4` down family on the exact C256 route population, retaining task size 32 and the existing gate choice. | Historical retained improvement, later refined by sparse-slab and short-K scheduling. Task size 64 was rejected for that decode shape. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E065 atomic large-row fusion | Eliminate the route-output buffer by accumulating weighted results into an FP32 token buffer. | Rejected: clear/cast costs and atomic accumulation made the complete boundary slower; summation order also changed. E065: `experiments/e065/README.md` |
| E065 exact small-token fusion, E084 | Let token-owned workgroups compute routes and reduce them in the original order; compare 16/48-column tiles with gate alternatives. | Mixed early/late capture behavior prevented unconditional promotion. Later E098–E111 improved this family. E084: `experiments/e084/README.md` |
| E085, E087 | Omit unused second-row-slab MFMAs and stores in the C256 down kernel, using the task's device-side population. | Selected after real captures; every qualified case improved. Profiling supported reducing wasted work. E085: `experiments/e085/README.md` |
| E094–E096 | Stop inactive K waves from prefetching weights; use register loads and a single six-atom output phase. | Selected after actual whole-MoE qualification. Down-only scouts improved about 20–28%; those numbers are not serving gains. E094: `experiments/e094/README.md` |
| E098–E099, E108 | Store only live partial components, load compressed weights into registers, and vectorize codebook copies in fused down/reduction. | Qualified and integrated as the basis for E110/E111. E098: `experiments/e098/README.md` |
| E110–E111 | Give each of the nine routes its own wave and compute both TP8 K128 tiles in that wave. Preserve the two partial sums and ordered route reduction. Compare one-, three-, and nine-route assignments. | Nine-route variant selected for C1/C2/C4; higher occupancy alone did not make the narrower alternatives faster. E110: `experiments/e110/README.md` |
| E115, E133 | Extend token-owned route9 down to C8/C16, including TP4. | Rejected: duplicated work loses shared-expert reuse on concentrated routes; TP4 regressions were particularly large. E133: `experiments/e133/README.md` |
| E141–E142 | Share decoded expert weights across two or four tokens, then reduce their nine routes in order. | Unconditional policy unselected: positive mean capture gains still contained early varied-route regressions beyond the allowed limit. E142: `experiments/e142/README.md` |
| E145 | Port grouped-token down to TP4 K512 with the original four-term reduction order. | Rejected: exact tests passed, but the intended C8/C16 policies regressed. E145: `experiments/e145/README.md` |
| E161–E162 | Choose four-token groups for at most 16 tasks and two-token groups otherwise, directly from GPU metadata. | Rejected by the per-capture rule: +10.38% mean, but the early varied C8 case regressed 4.234%. E162: `experiments/e162/README.md` |
| E166 | Skip entire empty phases in the grouped down kernel using phase masks and ballots. | Unselected: exact tests passed, but the early varied weakness remained. E166: `experiments/e166/README.md` |
| E169 | Use one-token groups for 10–16 tasks, four-token groups for exactly nine, and two-token groups otherwise. | Rejected: the problematic early varied case regressed 20.68%. E169: `experiments/e169/README.md` |
| E172, E180 | Compact at most 12 expert tasks into simultaneously active waves at TP8 C8, avoiding multiple serialized group phases. | Selected after actual C8 captures and combined native qualification. The final E180 combination improves the prior IQ2R whole-MoE capture baseline by 10.64% geometrically; this is not MXFP4 serving uplift. Plan: `GLM53_IQ2R_DATAFLOW_EXECUTION_PLAN.md` |
| E173 | Extend compact C8 down to TP4's four K tiles. | Rejected: 17 native tests passed, but eight route-pattern scouts regressed 7.75% geometrically. Decision: `experiments/e173/decision-r1.json` |
| E174 | Shorten TP4 accumulator lifetimes by completing one K-tile result at a time, retaining explicit ordered additions. | Rejected: VGPRs fell 104→95, but residency stayed at one workgroup/CU and performance regressed 10.55%. E174: `experiments/e174/README.md` |
| E177 | Request a different launch-bound occupancy setting on the TP4 compact kernel. | Unselected. The installed HIP macro's second argument controls waves/SIMD; value 2 did not request two workgroups/CU. Correction: `experiments/e177/README.md` |
| E186 | Force two-token groups so the compiler can remove upper-token accumulators, trading extra workgroups for less live state. | Rejected: no spills, but no residency increase and all eight scouts regressed; mean −14.79%. E186: `experiments/e186/README.md` |
| E187 | Correct the occupancy request to six waves/SIMD for two 12-wave TP4 workgroups/CU. | Rejected: residency did increase, but scratch appeared and all eight scouts regressed; mean −4.35%. E187: `experiments/e187/README.md` |
| E194–E196 | Decode and multiply one TP4 down weight atom at a time, reusing fragment and temporary-accumulator registers while preserving K-partial summation order. Integrate invalid-route protection and the current runtime separately. | E194 removes spills and retains two workgroups/CU. E195 passes 48 native/padding tests; eight scouts give mean +0.63%, worst −0.90%. E196 is queued for captured-value qualification on Fleet; unselected. E194: `experiments/e194/README.md`, E195: `experiments/e195/README.md`, E196: `experiments/e196/README.md` |

## Launch grids, task size, locality, and dispatch

| Experiments | Brief explanation | Outcome / evidence |
| --- | --- | --- |
| Early task-row sweeps | Compare 16/32/64/128 rows per expert task under the original TP8 route distribution. | Larger tasks were not universally faster; retained shape-specific policies rather than one global size. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E029 | Start the TP8 down 4×CU grid at 16 routed rows instead of the old higher threshold. | Historical retained change with a measured small serving uplift over the then-current IQ2R baseline. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E030 | Broaden high-row gate/down family and grid policies. | Rejected: inconsistent cross-layer results. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E031–E034 | Compare gate grid 2/4/6 at exactly 2048 routed rows; also test stream priority and lower-shape expansion. | Grid 4 won isolation but was not promoted to automatic dispatch without a valid serving transfer. Stream-priority and broader-shape changes were not robust. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E032 | Move the shared-expert launch to later points in the old two-stream pipeline. | Rejected: starting it before routing was best among the tested placements. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E042–E043, E046 | Sweep fused-M2304 gate/down grids and inspect stage costs. | Gate grid 6 remained an isolated candidate; the existing down grid was supported. No broad policy promotion. Record: `GLM53_IQ2R_EXPERIMENTS.md` |
| E047, E049–E050 | Use 64-row expert tasks and tuned cooperative gate/down grids for actual large prefill batches. | Historical retained bounded prefill policy; measured complete-boundary gains transferred to a smaller C256 serving gain. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E052 | Tune the actual M86 decode tail across layers/ranks. | Rejected as a common rule: the winning combination changed with the sample. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E054 | Extend the qualified prefill policy specifically to 1536 input tokens. | Historical retained exact-shape extension; the neighboring M96 shape did not justify extension. Analysis: `GLM53_IQ2R_ANALYSIS.md` |
| E061 | Recheck C64 component costs, grids, scatter preparation, and group32 activation handling. | No new robust policy; group32 activation handling lost about 9% at the full boundary. Audit: `GLM53_IQ2R_REPOSITORY_AUDIT_20260924.md` |
| E084, E093, E095 | Compare gate and down choices together on identical full-MoE captures, including split-K, quad, wait3, register, and fused reductions. | Produced the E096 shape-specific policy; isolated substage winners were not automatically promoted. E096: `experiments/e096/README.md` |
| E102, E108 | Extend token-major indexed sparse gate and short-K down to the exact C32 boundary, with neighboring-shape dispatch checks. | Qualified and selected; whole-MoE capture improvement about 10.5% over the older C32 path. E102: `experiments/e102/README.md` |
| E136, E144 | Test task-major versus N-major gate order and Redline-style eight-XCD permutation for small-token kernels. | Small, mixed gains; not selected for those shapes. E144: `experiments/e144/README.md` |
| E150–E151 | Apply task-major/eight-XCD ordering to sparse gate tasks and qualify C128. | Selected at C128: all real captured cases improved, geometric means +2.09% TP8 and +2.80% TP4. TP4 used TP8 captured inputs with TP4 shards. E151: `experiments/e151/README.md` |
| E155–E156 | Extend XCD ordering to C64 and sweep sparse-gate grid sizes at C64/C128/C256. | No additional policy selected; C128's existing ordering remained supported. E156: `experiments/e156/README.md` |
| E146 | Revisit TP4 wait3/register gates, codebook batching, and fused versus separate reduction at C1/C2/C4. | No robust general improvement. The initial mislabeled register-policy run used fallback and is excluded. E146: `experiments/e146/README.md` |
| E168, E170 | Admit six/twelve-row verification shapes used by five-token MTP into existing scheduled kernels; diagnose the TP4 reference mismatch. | Native checks pass with the correct cooperative reference. Synthetic gains remain unselected pending real MTP captures and serving evidence. E168: `experiments/e168/README.md` |

## Correctness fixes encountered during optimization

| Experiments | Fix or diagnosis | Status |
| --- | --- | --- |
| Initial encoder/device-cache work | Correct multi-device encoding state so compiled expert data belongs to the intended device. | Integration correctness work; see the dated analysis log: `GLM53_IQ2R_ANALYSIS.md`. |
| E063 | Add required codebook-reuse/tail synchronization to the inherited large32 down kernel. | A changing-graph regression reproduced the old race; corrected path retained. E063: `experiments/e063/README.md` |
| E072, E075 | Keep repacked expert weights owned by their model layer instead of reusing scratch that gets overwritten by later layers. | Ownership defect reproduced and corrected. E072's invalid serving result remains invalid; later quality failures are not erased. E075: `experiments/e075/README.md` |
| E119, E124–E125, E168, E170 | Distinguish legitimate reduction-order differences from comparisons accidentally using the wrong TP4 reference. | Native cooperative-reference checks pass; independent numerical diagnostics and full-model quality remain distinct evidence. E124: `experiments/e124/README.md` |
| E165, E175, E179 | Capture raw expert IDs and weights around shrinking graph batches, separating active rows from padding. Diagnostic clamping protects downstream addressing only to collect evidence. | Diagnostic runs do not qualify normal serving and their clamping is not a production fix. Current state: `experiments/CURRENT_STATUS.md` |
| E181, E183 | Add native bounds protection and initialized invalid-route results, then queue normal model checks without diagnostic clamping. | Native fault-injection tests pass; full-model qualification pending. E181: `experiments/e181/README.md` |
| E184 | Restore/build the missing causal/noncausal attention variants with and without LSE that official serving reaches during chunked prefill. | Both runtime copies pass all four independent numerical checks. Dependency repair, not an IQ2R speedup. E184: `experiments/e184/README.md` |
| E188 | Replay the exact MXFP4 sort binary with valid IDs, reused buffers, graph shapes, and interleaved prefills. | 552 graph cases plus 72 eager prefills pass; this does not clear the full-model fault. E188: `experiments/e188/README.md` |
| E189–E190 | Prevent register top-k from reading unwritten LDS result slots when NaN scores leave too few candidates. | Confirmed defect; full AITER module passes 72 finite comparisons, 900 poisoned-LDS cases, and 32 regression tests. Connection to the C16 crash and full-model fix qualification remain pending. E190: `experiments/e190/README.md` |
| E191 | Run normal TP4/TP8 quality with the actual E190 guard in private runtime copies. | Queued; semantic failures are retained separately from transport failures. E191: `experiments/e191/README.md` |

## Validation, profiling, and benchmark work supporting the search

These entries are included for completeness; they are not additional kernel
speedups or separate optimization wins.

| Experiments | Purpose / result |
| --- | --- |
| E004, E019–E021 | Establish matched whole-model attribution and detect profiler bias. Historical evidence identified the exposed shared-expert/add boundary and showed communication was already close to parity. |
| E028, E035 | Package/check historical serving comparisons. E028 has no independent optimization result. |
| E055–E056 | Capture actual prefill/decode routes across all MoE layers and TP ranks; replace misleading synthetic/sliced fixtures with production-derived inputs. |
| E057 | Hold an intermediate combined candidate while further qualified changes were being assembled; no standalone serving win. |
| E060, E064, E081, E088, E097 | Successive combined-runtime qualification and historical serving comparisons. Their different workloads and runtime versions cannot be merged into today's official-script tables. |
| E077, E092 | Shadow real model calls against prior native gate/down implementations to test exact values over many layers and steps. Instrumented timings do not qualify serving performance. |
| E078, E082–E083 | Validate graph-safe timing events and capture missing token shapes, prompt types, and early/late decode steps. |
| E086–E087, E101, E105 | Collect joined hardware-counter/ISA evidence for occupancy, instructions, waits, traffic, and actual dispatched kernels. Static wait counts are not dynamic stall cycles. |
| E103, E109, E112, E121, E126, E131, E147 | Historical nine-concurrency serving/quality sweeps and their revisions. E109 was superseded before its intended serving probe. E147 completed both TP widths, but its custom workload does not satisfy the requested ATOM benchmark script. |
| E113, E135, E158–E159, E171, E176 | Build and run diverse/normal-EOS answer checks. Retain incorrect, malformed, incomplete, and failed-request outcomes separately. These are not a substitute for a standard comparative accuracy evaluation. |
| E114 | Roofline and useful-work analysis: distinguish real token/expert work from padded MFMA rows, and measured bandwidth from nominal compressed-weight savings. |
| E123, E137, E143 | Identify the InferenceX run settings and prepare the offline AIPerf client, dataset, and real-acceptance recipe. Exact GLM-5.2 MXFP4/image Fleet assets remain unavailable at the latest check. |
| E149 | Attribute historical varied-prompt gaps using request logs; mainly steady decode, with a prefill/first-token component at C256. |
| E157 | Isolated eager sort reproduction at the originally suspected prefill shape. Passing that case did not reproduce or clear the shrinking-batch serving fault. |
| E180, E185 | Run unchanged official ATOM benchmark clients at both requested lengths and all nine concurrencies, with matched MXFP4 A/IQ2R/MXFP4 B arms. TP8 is partial and TP4 is queued; no complete performance qualification yet. |
| E100, E128–E129, E132, E153–E154, E160, E164, E178, E193 | Preserve/restore immutable sources, binaries, overlays, weights, and results. Some earlier preservation attempts failed or expired; only completed, hash-verified sets are restore sources. |
| E192 | Let the already-running C256 client finish under a longer, bounded external supervisor; future sweeps record the longer limit directly. Official workload and server commands remain unchanged. Details: `experiments/e192/README.md` |

## Remaining acceptance work

1. Establish the actual cause of the full-model C16 fault and validate the
   correction during normal TP4/TP8 serving.
2. Complete both official ATOM workloads at all concurrencies with fresh MXFP4
   bookends, native dispatch evidence, and zero failed requests.
3. Close every remaining performance deficit and demonstrate that TP4's relative
   advantage is at least TP8's on the matched workload.
4. Qualify model behavior against the appropriate reference; do not treat
   successful HTTP requests or native exactness tests as answer-quality proof.
5. Run the final InferenceX agentic workload with real MTP acceptance and retain
   actual acceptance counters, output quality, and failure counts.

## Source and coverage notes

The main sources are the experiment log: `GLM53_IQ2R_EXPERIMENTS.md`,
the dated analysis: `GLM53_IQ2R_ANALYSIS.md`,
the execution plan: `GLM53_IQ2R_DATAFLOW_EXECUTION_PLAN.md`, and the linked
experiment READMEs, source files, and result summaries. Several historical log
entries retain their initial "active" prose after a later result was recorded;
the inventory uses the later explicit result and notes unresolved qualifications.

The coverage includes all documented optimization families found through E196,
including early unnumbered trials. E005–E006, E022–E026, E036–E038, and E130 have
no distinct completed experiment record identified in these sources. E182 is an
empty local staging directory. No optimization result is invented for those IDs.
Revisions that only repair commands, packaging, or observation are grouped with
their experiment rather than counted as new optimization ideas.

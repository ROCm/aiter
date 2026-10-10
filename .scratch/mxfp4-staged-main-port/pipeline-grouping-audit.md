# fMoE execution and staged Cartesian-group audit

Status: ready-for-human

Audited on 2026-10-10 against target main `882ea79dd7cc7091f859a5bedac914523ac77bbd` and the source staged implementation `f68e08528cb85f7f3f7211d3f8a1664660bb7f50`. Main locations below are from `git show` of the fixed target, not from the old working-tree files. Three parallel investigations covered dispatch, sorting/GEMM1/measurement, and GEMM2/output processing. No GPU timing, numerical execution or migration implementation was performed.

This document describes the code and design at the audit time. The later [reviewer resolution](reviewer-comments.md) supersedes the previous skip-on-reference/all-stage-screening-failure policy with group-local exhaustive normal validation. Refer to the current spec for required behavior; the source-code behavior reported here remains historical evidence.

## Judgment

The planned `(BM, accumulation mode)` groups are reasonable **pipeline compatibility groups** for the fixed A4W4 candidate universe. The code does not reveal a GEMM1 tuning parameter that makes only some GEMM2 configurations compatible within one such group. Reduce and scatter do not need separate groups merely because they use different scratch layouts or final kernels, when GEMM2 screening includes their complete path to final BF16 output.

These groups are not proven groups of performance separability. Independent G1/G2 screening cannot currently be claimed to completely determine fMoE latency or the best valid pair. In-pipeline execution conditions differ from isolated measurement; correctness also depends on the actual produced intermediate and GEMM2's rounding/quantization. Splitting by more kernel parameters does not establish exact independence.

Keep the current groups for this port and retain end-to-end correctness/timing as authority. Atomic/non-atomic merging is a possible alternative measurement/search design, not a confirmed redundant split. The more immediate risk is measuring a different effective production path, particularly input-dtype resolution and stage2 FP8 arguments.

## Public fMoE dispatch map

The scope here is the actual public `aiter.fused_moe.fused_moe` call and its execution alternatives, then the complete path relevant to the staged tuner. Separate MegaMoE, communication and other standalone APIs are not presumed to execute inside this call.

| Execution family | Actual path / distinguishing work | Why the current staged grouping does not cover it |
| --- | --- | --- |
| Shared-expert FHMoE | Shared tensors/expert id divert the public call to `_fhmoe` before ordinary dispatch (`aiter/fused_moe.py`:864–901). | Different request and combined expert work; not the current two-stage candidate source. |
| gfx1250 grouped MoE | Runtime checks gate layout and attempts grouped A8W4 execution; a supported grouped result returns before normal metadata/sorting (`fused_moe.py`:1162, 1209–1243). | Different architecture, weight/input layout, routing and execution implementation. |
| Whole-graph FlyDSL gfx942 | A configured `full_impl` executes a complete request and returns before common sorting (`fused_moe.py`:1376–1426, 3300–3359). | No exposed independently interchangeable G1/G2 pair at this seam. |
| ASM one-stage, including FLAT | Selected metadata calls fused one-stage execution after sorting or raw-route preparation; FLAT has a distinct raw topk-input contract (`fused_moe.py`:118, 1516–1550, 3490–3529). | Two GEMMs/activation are fused; ordinary G1×G2 enumeration does not describe it. |
| Generic CK / CK-Tile / ASM two-stage | Sorting → input quantization/conversion → G1 → optional intermediate quantization/scale handling → G2 (`fused_moe.py`:4054–4519). K-split, dtype, gate/up, padding and bias may change intermediate handling. | BM alone does not identify layout, quantization, split-K or intermediate contract. |
| Generic FlyDSL / Opus two-stage | Metadata selects stage families, may set G1 output dtype from layout-v2 G2's A dtype, and chooses `fuse_quant`, `skip_inter_quant`, auxiliary sorting and route metadata (`fused_moe.py`:3531–3635). | Some paths can share a proven intermediate contract, but cannot be merged merely by final dtype/BM; not in this tuner's enumeration. |
| Native MXMOE GEMM2 | `_mxfp4_a4w4_stage2_fw` has a separate native branch with native alignment/output-mode handling (`fused_moe.py`:2274–2411, 2600–2630). | Deliberately excluded from main's dedicated candidate source; old native alias rules do not apply to layout-v2. |
| Dedicated A4W4 MXMOE G1 + layout-v2 G2 | Auxiliary routing/sort, fixed BM-derived input quantization, fused G1 activation/output FP4 quantization, then layout-v2 G2 and final handling. | This is the port's universe. Its exact fixed and derived contracts justify the proposed groups. |

Metadata is resolved before sorting. `q_dtype_a` comes from `resolve_activation_dtype`, unless explicitly overridden (`fused_moe.py`:1188–1202); `get_2stage_cfgs` keys include gfx/CU, shape, activation, output dtype, effective input/weight quant dtype and other shape contracts (`:3088–3103`). A matching kernel-name string alone does not establish that the same execution family or quantization path is selected.

## Dedicated A4W4 pipeline

Fixed scope: gfx950, BF16 caller input/output, FP4 A/W quantization policy, per-1x32, G1U1, separated gate layout, no G1 routing weight, no bias/EP/padding extension, fixed shape/routes/effective environment per comparison.

```text
BF16 hidden + shuffled FP4 weights + topk ids/weights
        |
        v
sort/pad routes at BM; emit ids, weights, experts, cumsum,
m_indices and reverse_sorted; initialize final output for atomic
        |
        +-- BM16: raw BF16 consumed by G1, inline input quantization
        |
        +-- BM32/64/128: fused input FP4 quantization + sorted E8M0 scales
        |
        v
G1: matrix multiply -> activation/limit -> FP4 payload + E8M0 scales
    in the BM-specific sorted layout
        |
        v
G2: matrix multiply -> one of the output paths below -> BF16 [M,H]
```

`_make_mxfp4_metadata` binds runtime gate layout to G1 and binds the G1/G2 kernel names separately, with FP4 `fuse_quant` and auxiliary sort (`fused_moe.py`:2855–2889). `_fused_moe_impl` selects accumulation from G2 before sorting (`:1456–1469`). BM also controls the sort's padding, expert tiles, G1 inline/prequant policy and intermediate scale layout, not just launch size.

BM16 chooses adaptive versus Opus auxiliary sort by routed-row/expert crossover and backend settings (`fused_moe.py`:78–103, 432–450). Atomic changes output initialization in that sort. Other BMs normally use Opus for the selected auxiliary policy. Production forces the non-inline A4W4 input-prequant policy for BM≠16 (`:4148–4161`); the selected effective input contract must still satisfy the actual branch conditions discussed below.

G1 produces its packed payload and scale in the kernel, so this universe has **no additional G1-to-G2 quantization kernel**. Production recognizes the returned tuple and directly passes payload/scales (`fused_moe.py`:4388–4393). This differs from generic two-stage paths with external intermediate quantization (`:4411–4474`).

### G2 output paths

| Path | G2 writes | Routing weight and final work | Initialization / effect on grouping |
| --- | --- | --- | --- |
| Atomic BF16 | Contribution to caller `[M,H]` | Multiply route weight, BF16 rounding, atomic BF16 accumulation. | Caller output must be zero before G2; production does this during sorting. Distinct accumulation contract. |
| BF16 reduce | Weighted BF16 `[M,topk,H]` | Weight is applied before the route scratch store; final reduction accumulates in FP32, stores BF16. | Non-EP final output is overwritten; no previous output contents required. |
| BF16 scatter | Unweighted BF16 `[max_sorted,H]` | `reverse_sorted` gathers route scratch, applies sorted weight during FP32 FMA, stores BF16. | Same caller-visible overwrite contract as reduce; different internal rounding and scratch. |
| FP8 reduce | Unweighted packed FP8 routes + E8M0 scales | Dequantize and apply `topk_weights` during final FP32 reduction, store BF16. | Same final overwrite contract; extra quantization, scale decoding and weighting placement. |
| FP4 scatter | Unweighted packed FP4 `[max_sorted,H/2]` + scales `[max_sorted,H/32]` | Dequantize and apply sorted weights during final FP32 scatter reduction, store BF16. | Same final overwrite contract; additional lossy quantization. Enabled only for eligible no-bias scatter. |

Evidence: `_flydsl_v2_stage2_wrapper` allocations/branches and final operations (`fused_moe.py`:2705–2851); atomic/reduce weighted BF16 stores (`aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py`:1012–1021, 1255–1257); unweighted BF16/FP4 scatter (`:1344–1426`); final reduction (`kernels/moe_reduce.py`:164–182); scatter decoding/weight/FMA/overwrite (`csrc/kernels/mxfp4_moe/moe_aux/moe_scatter_reduce.cuh`:35–81, 122–152).

Neither identical final BF16 dtype nor common compatibility implies bitwise-identical numerical results. The weight/rounding positions differ, atomic addition order differs, and FP4/FP8 paths quantize before the final sum. Full end-to-end correctness must remain pair-specific.

### Execution fields and their classification

| Field | Role in the present universe | Grouping judgment |
| --- | --- | --- |
| Shape, actual routes, input dtype/scales, activation/limit, gate, device, execution flags | Fixed context for one measurement/search. Flags can change actual output handling without changing names. | Validate/fix them before grouping; no scores/cache reuse across contexts. They need not create extra groups inside one fixed context. |
| G1 BM / G2 BM / SBM | Sort padding, input quantization policy and intermediate payload/scale interpretation. | Keep. Main enumerates only equal BM/SBM; merging different BM products would admit unsupported cross-pairs. |
| G1 A/output dtype, intermediate scale layout | G2 operand and layout contract. Current universe fixes FP4; BM16 scale layout is derived. | Validate explicitly. Do not merge generic BF16/FP8 intermediates with this universe. |
| Accumulation mode | Final output prezero/atomic versus overwrite, with production prologue differences. | Keep in current contract. It also reserves separate screening shortlists. |
| G2 reduce versus scatter | Scratch layout, output operation and numerical order; both complete callables overwrite final BF16. | No compatibility need to split. Splitting to reserve each family a shortlist quota would be a search-policy change. |
| G1 BN, waves, k_wave, XCD, NT, prefetch | Kernel tuning axes; not consumed as G2 configuration restrictions. | Preserve full identity. Do not add to compatibility keys without an actual contract difference. May affect numerical output or cache state. |
| G2 BN/BK, persistence, load policy, epilog implementation | Launch/input-tile/output performance axes and complete execution identity. | Preserve identity, not additional compatibility keys if the prepared-input/final-output callable contract is unchanged. |
| Bias / EP / caller prequantization / alternate backend | Can change operands, zeroing, route masks, output or dispatch. | Outside current universe or requires separately proven context; generic G2 capability does not expand the dedicated tuner. |

Scatter's effective persistence is forced in the dispatcher even when its name lacks `_persist` (`mxmoe_dispatcher.py`:817). Current registry provides no duplicate scatter persist alias to collapse. G2 name NT controls B loads, while scatter postprocessing NT is derived from BM (`mxmoe_gemm_v2.py`:437; `moe_mxfp4_aux.cu`:293, 324); do not conflate them.

## Cartesian completeness: what was verified

For fixed context and accumulation mode, main's record set is:

`P(BM, mode) = G1(BM, shape, pruning) × G2(BM, mode, shape)`.

The enumerator calls `_g1_variants`, then enumerates the same G2 registry using only dtype, BM/H/I, filtering `tile_m == BM` (`tuner gemm_moe_tune.py`:6658–6683). G2 does not inspect G1 BN/waves/k_wave/XCD. M_est pruning depends only on G1 fields and shape (`:6517–6531, 6588–6591`). The G2 kernel interprets A/scales by BM/SBM, A/B dtype and logical dimensions (`mxmoe_gemm_v2.py`:266–310, 393–435).

A CPU-only check extracted the actual main enumeration/static-support definitions with AST, avoiding full GPU/FlyDSL imports, and verified each group equals the product of its projected unique G1/G2 sets. At H3072/M_est64:

| I | BM | Atomic product | Non-atomic product |
| --- | --- | --- | --- |
| 384 | 32 | 9×8 =72 | 9×8 =72 |
| 384 | 64 | 12×8 =96 | 12×8 =96 |
| 384 | 128 | 6×8 =48 | 6×12 =72 |
| 512 | 32 | 9×16 =144 | 9×16 =144 |
| 512 | 64 | 12×16 =192 | 12×16 =192 |
| 512 | 128 | 6×16 =96 | 6×24 =144 |

All six groups were complete products (456 / 912 total records). This checks candidate-record generation and the static contract, not that every pair compiles, executes or passes the numerical gate. The passing-pair relation need not itself remain Cartesian.

## Concrete mismatch risks before grouping changes

### 1. Effective production input dtype / prequantization

Production resolves actual input quant dtype from activation/token/layout. Default Situv2 is A16W4 unless `AITER_SITUV2_A4W4=1` or an explicit override selects FP4; separated Swiglu defaults to BF16 below its token bound (default 256) (`fused_moe.py`:785–801). Config lookup includes this effective dtype (`:3088–3103`). The dedicated tuner instead fixes FP4 A/W (`tuner`:6686–6700).

Production's BF16 Swiglu/Situv2 input branch can keep raw BF16 and no scales (`fused_moe.py`:4165–4179), then a non-inline MXMOE G1 would internally execute `mxfp4_moe_quant` plus `mxfp4_moe_sort_scales` (`:2170–2194`). The tuner's BM≠16 path always performs fused prequant first (`tuner`:6752–6762). Thus even reusing a stage wrapper does not establish the same whole pipeline. An A4W4 tuned row may also not be selected under default A16W4 dispatch.

For this port, preserve the agreed A4W4 scope by explicitly fixing the effective FP4 input contract in production replay and checking both selected metadata and actual operation chain. Do not silently expand to A16W4 or treat a fallback `OK` as the selected pair's acceptance. Ordinary A16W4 production behavior remains a distinct path.

### 2. Stage2 FP8 arguments and activation limits

Main's `_port_e2e` lacks `topk_weights` (`tuner`:6787–6802), while production supplies it for FP8 reduction (`fused_moe.py`:4345–4355). The wrapper disables FP8 output without it (`:2707–2708`). This is an actual operation-chain mismatch, already in the parameter/AOT repair scope.

The source G1 direct screening omits explicit normalized limit (`source tuner`:6642–6672); main wrapper normalizes SiLU to infinity and Swiglu to 7 or the configured limit (`fused_moe.py`:2444; `moe_kernels.py`:821–831). Screening, bootstrap, reference, E2E and deployment must all use that same effective value. This is not fixed by adding BM/epilog groups.

### 3. Isolated measurement is not in-pipeline measurement

| Measurement | Actual repeated work/state |
| --- | --- |
| Source G1 screen | Setup creates data, non-atomic sort, fused prequant and output buffers once. Repeated callable launches only G1 into fixed buffers. There is no G2 between repetitions. (`source tuner`:6567–6689.) |
| Source G2 screen | A validated reference G1 produces one intermediate during setup. Repetitions read that fixed intermediate; atomic explicitly zeroes output immediately before G2. (`source tuner`:6690–6747.) |
| Normal pair timing | Each repetition performs sort/atomic initialization, BM-specific prequant, actual G1 writes its intermediate, then actual G2 and final output work. (`main tuner`:6720–6802, 6858–6865.) |

`run_perftest` rotates only explicitly supplied args/kwargs (`aiter/test_common.py`:76–79). These callables have zero explicit tensor arguments; tensors are captured in partial/lambda closures. Rotation therefore does not rotate their tensor state. A pure-Python reproduction of this deepcopy mechanism confirmed bound/captured object identity stays fixed. The helper's input-size estimate also only counts positional tensor args (`:184–194`). This is not evidence of cache isolation.

BN/XCD/NT/prefetch/waves can change memory access/production order and the hardware state inherited by G2. G2 repetition may leave W2 and intermediate hot rather than enter immediately after a candidate-specific G1. Atomic output initialization in production occurs before G1 and can share a sorting launch, whereas isolated screening places a standalone zero immediately before G2 (`fused_moe.py`:172–182, 1469; source tuner:6705). Code establishes these differing conditions, not the magnitude of their timing effects.

### 4. Correctness and numerical coupling

G1 k_wave changes K-slice accumulation and sum order (`kernels/mxfp4_gemm1.py`:1080–1095, 1127–1131). The kernel then applies activation/limit, computes E8M0 scales and packs FP4 (`:1254–1275, 1323`). The common layout does not establish bitwise-identical intermediates for different G1 configurations. G2's output/weight/quantization variants also differ numerically. Consequently, bootstrap correctness for one reference producer does not certify all pair outcomes.

## What independent timings can and cannot decide

Exact additive determination would require:

`T(i,j | fixed context) = C(group) + t1(i) + t2(j)`.

Here C must cover all shared sorting/input-quantization/initialization work in a non-overlapping, production-equivalent measurement definition. Isolated t1/t2 must equal their conditional durations in the complete pipeline, without pair-dependent cache/data/address/resource effects. The valid-pair set must also be known. None of these performance/validity conditions is certified by the present static grouping.

For ranking within one group, a constant missing C alone is harmless. Across BMs or accumulation modes, C differs, so independent G1/G2 sums cannot choose the shape winner without additional preparation costs. Within a fixed group, possible interaction remains; current repeated-call conditions provide no guarantee that it is constant or too small to change the winner.

The main default `run_perftest` returns averaged **device trace kernel-duration sum** (`aiter/test_common.py`:364–446), not invocation wall time including all Python work/launch gaps. CUDA-event/graph modes have different boundaries. Python metadata and `torch.empty` CPU work must not be asserted to be included in that device sum. They can still affect memory state, and tuning wall time is a separate quantity. Any decomposition experiment must use one explicitly chosen metric consistently.

### Merge/split judgment

- **Keep BM separate.** Cross-BM pairing changes actual sorting and scale interpretation and is outside the selected source. Re-tile support elsewhere does not make these products interchangeable.
- **Keep reduce/scatter together for the agreed full-G2 score.** Within the fixed non-EP context, both consume the same prepared intermediate and complete final-output work is measured. Their internal scratch/rounding difference is not a pair-specific compatibility restriction. Reserving K per epilog may reduce approximation risk, but increases search cost and is a new strategy choice.
- **Atomic/non-atomic can be merged as a mathematical candidate-set partition, but are not proven redundant execution/screening groups.** They share the same G1 set: `(A × B_atomic) union (A × B_nonatomic) = A × (B_atomic union B_nonatomic)`. Thus such a record-set merge creates no new G1/G2 pairing; M_est64 could have three BM products instead of six mode-specific products. Their prologue/initialization contracts still differ. A redesigned complete-output G2 score could allow common ranking, but needs production-equivalent zero-cost treatment and loss-of-family-quota evaluation. With K=2, one shared G2 shortlist would keep two candidates total per BM rather than two per mode; initial pair count could fall from 24 to 12 at the cost of eliminating entire output families. Current separate shortlists preserve alternatives at a known cost. Neither design is a static proof of independence.
- **Do not split by every G1 tuning axis.** It preserves increasingly many combinations but cannot certify away shared cache state or pair correctness. At the extreme it merely restores exhaustive pair search.
- **Do not mix other fMoE families under BM×mode.** Only identical effective intermediate, routing, quantization and output-preparation contracts can justify sharing a context. The present helper's fixed-contract validation is essential.

## Evidence to resolve the remaining performance question

Use the already agreed calibration shapes and targeted remeasurement, not a new full cross-configuration matrix. Record the exhaustive-prune winner's G1/G2 ranks under screening and whether the pair survived the shortlist. Large regret accompanied by a missing winner diagnoses shortlist loss; an invalid reference/shortlist is a different failure.

On selected same-group candidates, run a small crossed set with fixed data/routes and alternating order, retaining production prelude and postlude. Vary representative NT/XCD/BN/k_wave or G2 persistence/epilog axes when investigating a concrete discrepancy. Measure individual kernel segments in the full pipeline as well as the full device metric; compare them to isolated-screening durations. Distinguish input/scale numerical differences from hardware-state effects.

A necessary test of additivity is the rectangle contrast:

`D(i,j) = T(i,j) - T(i,j0) - T(i0,j) + T(i0,j0)`.

It is zero for an exactly additive pair-time matrix. Repeat to establish whether nonzero contrasts or ranking reversals exceed the measurement noise. Agreement on a finite sample only supports an approximation for that sample/context; it cannot establish universal exact independence. Do not introduce a hard performance threshold that the user declined in Q14.

If sampled interactions are small and winners are retained, the existing staged heuristic has supporting evidence. If rankings flip or the fastest valid pair is pruned, revise the search policy using those measurements. No group refinement alone justifies replacing the end-to-end correctness/time gate with summed independent scores.

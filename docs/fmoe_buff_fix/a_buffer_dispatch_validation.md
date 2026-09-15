# FMoE GEMM2 A addressing: implementation and validation

## Result

The implementation selects the original buffer A loads below 4 GiB and the existing 64-bit global A loads at or above 4 GiB. Public runtime APIs and the GPU launch ABI are unchanged. Runtime and AOT explicitly select the Boolean variant; compiler callers without A metadata retain the global default. Cache identities and kernel symbols distinguish the two variants.

**Complete spec acceptance remains partial.** The selected large FP8 full-FMoE configurations produce intermittent nonfinite results on both the frozen global baseline and the candidate. The root cause is unresolved. Full-size standalone GEMM2 passes, but that does not turn failed full-FMoE runs into successes. No production defect introduced by this addressing change was identified in the Standards/Spec review.

## Source identity and protocol

- The original buffer baseline is `a7c3b96e6009c54d8d6e48adeb30feb763829114`.
- The previously uncommitted global baseline has shared-builder SHA-256 `3500cfd10c59602278ce6e46d8ac776a2aafa6a7e15dbc90ad66bbb1a9b20190`.
- The candidate production implementation is commit `b7cbdc9ede80444ae88cab3925ed422b5f9c18b3`; its shared-builder SHA-256 is `413639df097988bc4b4bc16d0a12874666d75eb78ae68721f218c2ffa7561202`. The later review fix changes only unsupported test-sweep filtering.
- The final per-file hashes and dependency versions are recorded in [final_sources.json](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/final_sources.json).

Every comparison uses one locked serial queue per GPU. The six physical BDFs are `0000:05:00.0`, `0000:65:00.0`, `0000:75:00.0`, `0000:85:00.0`, `0000:e5:00.0`, and `0000:f5:00.0`. HIP ordinals differ from rocm-smi indices; each worker verifies its BDF. Initial comparisons use six alternating pairs and five warmup/five measured launches per block. GEMM2 uses only its kernel events; full FMoE uses the existing test entry’s returned `us`. Atomic reset and subsequent reduction are excluded from GEMM2 time.

The original source rows and parameters were frozen, including SiTUv2 beta=4, linear_beta=25 and DSV4 swiglu_limit=10. Raw events, compiler parameters, device identity, timestamps and block snapshots are retained. Continuous power/process telemetry covers the original batch; later tests retain block snapshots, so external interference cannot be excluded completely. No clock settings were changed.

## Correctness and compatibility

| Check | Observed result |
| --- | --- |
| FP8 / packed FP4 A boundaries | Below, exact FP4 4 GiB, and above-boundary cases passed; FP8 nearest representable neighbors passed. |
| FMoE / FHMoE stage2 | Sync/async and atomic/per-slot reduce passed with high routed rows and invalid tail padding. |
| FP8 weights | Small and both FP8 A boundary neighbors passed for sync/async and both accumulation modes. |
| Padding and contiguous views | Declared K padding stayed in size selection; small views starting beyond 4 GiB in backing storage selected buffer loads and passed. |
| Process caches | Small→large→small and large→small→large runs passed with matching symbols. |
| Stage2 Graph | Warmup/capture/repeated replay and updated input/routing passed for both variants. |
| Full fused_moe Graph | Small sync, registered async, and large F8-1 checks passed, including a joint permutation of input/routing vs the permuted independent Torch reference. Replay made zero Python stage2 calls. |
| Existing FHMoE tests | `pytest -q op_tests/test_fhmoe.py`: 41 passed. |
| Existing auxiliary tests | The complete `test_flydsl_moe_aux.py` eager/Graph sweep passed. |
| Public signatures / GPU ABI | Public signatures were unchanged; device signatures remain ten pointers and five i32 values. |

The sparse boundary tests check complete output and preserve their original assertions; their timings include reset and are not used as full-workload performance evidence. New comparisons also require finite output and `logits_diff <= 0.01`. The original inventory includes four FP4 BM16/BK128 sync configurations exercising 4-byte loads and eighteen configurations exercising 8-byte loads. The known FP4 BM16/BK128 async combination was not added to the passing set.

The large Graph run used the fallback GEMM1 selected by the initial constructed harness, as recorded in `graph_F8-1.log`. It validates the warmup/capture contract for that path; it does not validate the failing matched tuned-GEMM1 configuration.

The [A-load IR/ISA note](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/a_load_evidence.md) traces A itself, not weight/scale buffer instructions, in four compiled variants. It confirms descriptor construction only for buffer variants, 64-bit global offsets, the different async LDS address semantics, and no GPU size-selection branch.

## Original performance inventory

The new run attempted 372 configurations, produced 365 valid paired comparisons, and retained 7 excluded configurations. It triggered 13 initial >3% retests. All 13 completed 20 resident-process pairs and a further 20 independent-process pairs on their originally assigned physical GPU.

The resident-process retests met the agreed confirmation rule for 6 cases. The independent-process retests met it for 0 cases. These results are sensitive to process/allocation state; the former observations are retained, not erased. The independent-process result does not establish that every configuration is regression-free.

| Aggregate of initial valid pairs | GEMM2 | Full FMoE |
| --- | ---: | ---: |
| Configuration-weighted geometric mean | -0.572% | -1.384% |
| Model-equal geometric mean | -0.731% | -1.154% |

The 26 historically confirmed regressions were all rechecked; 0 triggered >3% in this run. The seven registered async configurations had an initial GEMM2 geometric-mean change of -0.153% against the original buffer baseline. These are current measurements, not historical pass counts.

| Triggered case | Initial GEMM2 | Resident retest | Independent retest | Independent delta | Independent bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv3_fp4_tuned_fmoe:row72` | +4.995% | +5.121% | -0.313% | -0.280 µs | [-0.823%, -0.134%] |
| `dsv4_fp8fp4_tuned_fmoe:row133` | +3.343% | +1.077% | +0.400% | +0.799 µs | [-0.089%, +0.772%] |
| `dsv4_fp8fp4_tuned_fmoe:row34` | +3.006% | +4.915% | -0.021% | +0.001 µs | [-3.710%, +2.247%] |
| `dsv4_fp8fp4_tuned_fmoe:row41` | +4.139% | +2.856% | -0.088% | -0.300 µs | [-0.537%, +0.398%] |
| `dsv4_fp8fp4_tuned_fmoe:row68` | +3.586% | +6.650% | +0.061% | +0.140 µs | [-0.217%, +0.480%] |
| `dsv4_fp8fp4_tuned_fmoe:row84` | +4.618% | +7.596% | +0.101% | +0.480 µs | [-0.632%, +0.695%] |
| `dsv4_fp8fp4_tuned_fmoe:row94` | +9.305% | -13.547% | -0.231% | -0.200 µs | [-1.453%, +0.993%] |
| `dsv4_fp8fp4_tuned_fmoe:row96` | +4.485% | +3.495% | +0.279% | +0.280 µs | [-0.395%, +0.562%] |
| `glm5_mxfp8_tuned_fmoe:row11` | +3.422% | +4.326% | +0.241% | +0.379 µs | [-0.051%, +0.498%] |
| `glm5_mxfp8_tuned_fmoe:row28` | +10.553% | -0.513% | -0.368% | -0.400 µs | [-1.209%, +0.588%] |
| `q3vl_fp4_tuned_fmoe:row22` | +3.294% | +2.946% | -0.415% | -0.060 µs | [-3.080%, +2.387%] |
| `q3vl_fp4_tuned_fmoe:row26` | +8.652% | -4.192% | +2.783% | +0.460 µs | [-1.885%, +6.607%] |
| `q3vl_fp4_tuned_fmoe:row48` | +3.578% | -0.122% | +0.101% | +0.159 µs | [-0.364%, +0.630%] |

The confirmation rule is retest median >3% and CI lower bound >0. The complete [case table](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/case_results.csv), [summary with paired percentages and absolute differences](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/final_summary.json), [initial/resident raw records](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/performance/raw_measurements.jsonl), and [independent raw records](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/fresh_retest/raw.jsonl) preserve every result.

The excluded configurations were:

- `minimax_m3_fp4_tuned_fmoe:row3` retained status `incorrect` because correctness failed on both versions.
- `kimik3_a8w4_tuned_fmoe:row36` retained status `incorrect` because correctness failed on both versions.
- `kimik3_a8w4_tuned_fmoe:row4` retained status `failed` because runtime selected the separate port instead of the modified shared GEMM2.
- `minimax_m3_fp4_tuned_fmoe:row19` retained status `incorrect` because correctness failed on both versions.
- `minimax_m3_fp4_tuned_fmoe:row20` retained status `incorrect` because correctness failed on both versions.
- `minimax_m3_fp4_tuned_fmoe:row2` retained status `incorrect` because correctness failed on both versions.
- `minimax_m3_fp4_tuned_fmoe:row18` retained status `incorrect` because correctness failed on both versions.

## Constructed full-workload group

All nine GEMM2 comparisons below have six valid pairs. The FP8 comparisons use the public stage2 entry with the full independently quantized Torch-reference intermediate and complete routing. The FP4 comparisons extract GEMM2 events from matched full FMoE. Buffers and kernel workloads were never reduced to sparse correctness surrogates. At/above 4 GiB the baseline is the correct global implementation; below the boundary it is the original buffer implementation.

| Case | Actual A bytes | GEMM2 source | Base µs | Candidate µs | Paired change | Full-FMoE change |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| F8-C | 4294950912 | standalone stage2 | 31286.019 | 30998.117 | -0.927% | not valid: nonfinite full FMoE |
| F8-1 | 4294969344 | standalone stage2 | 30945.415 | 31158.417 | +0.762% | not valid: nonfinite full FMoE |
| F8-2 | 4831838208 | standalone stage2 | 34891.732 | 34803.831 | -0.384% | not valid: nonfinite full FMoE |
| F8-3 | 4831838208 | standalone stage2 | 36345.163 | 36109.862 | -0.702% | not valid: nonfinite full FMoE |
| F4-C | 4294959104 | GEMM2 event in full FMoE | 9635.396 | 9747.184 | +1.252% | +2.463% |
| F4-E | 4294967296 | GEMM2 event in full FMoE | 9167.792 | 9309.717 | +1.452% | +0.334% |
| F4-1 | 4294975488 | GEMM2 event in full FMoE | 9123.120 | 9167.056 | +0.772% | +0.502% |
| F4-2 | 4831838208 | GEMM2 event in full FMoE | 10084.166 | 10059.473 | -0.476% | -1.140% |
| F4-3 | 4831838208 | GEMM2 event in full FMoE | 9377.583 | 9495.178 | +0.839% | +0.228% |

The separate F4-3 reduction kernel took 1904.336 µs on base and 2071.678 µs on candidate (median of recorded reduction events). These times are not in the GEMM2 metric.

### Remaining full-FMoE failure

The matched F8-C/F8-1/F8-2/F8-3 full-FMoE runs contain nonfinite results on both versions, including the below-boundary control. The captured A values themselves were finite in the diagnostic runs; this does not locate the failing component. No full-FMoE latency ratio is accepted for those configurations.

| Full-FMoE case / attempt | Base incorrect blocks | Candidate incorrect blocks |
| --- | ---: | ---: |
| F8-C / 1 | 1/6 | 1/6 |
| F8-C / 2 | 1/6 | 1/6 |
| F8-1 / 1 | 5/6 | 5/6 |
| F8-1 / 2 | 5/6 | 5/6 |
| F8-2 / 1 | 3/6 | 3/6 |
| F8-2 / 2 | 3/6 | 3/6 |
| F8-3 / 1 | 3/6 | 3/6 |
| F8-3 / 2 | 3/6 | 3/6 |

The [normalized attempt table](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/large_full_fmoe_attempts.csv) separates repeated round keys; attempts are not combined into extra independent pairs. Earlier textual case logs were overwritten by their retry, but raw numeric/event records remain. The [standalone stage2 records](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/large_stage2_normalized.json) rename the worker’s misleading `fmoe_us` field to `stage2_with_reset_us`; it is never presented as full-FMoE time.

Initial constructed probes that missed the intended GEMM1 or GEMM2 overrides are retained under `large/` and `large_dispatch_probe/`, but excluded from acceptance. Matched runs use CSV lookup token 131072 and preserve the larger actual token during input preparation; requested/effective stage1 and stage2 compiler parameters are checked before timing. A reference-only FP4 quantization wrapper flattens leading dimensions and processes bounded row chunks to avoid a Python loop over every token.

## AOT, host and build costs

AOT uses the existing CSV CLI and FakeTensor path. Small and large artifacts load in separate run-only processes and pass Graph replay. The small-only cache rejects a crossing input with the existing `no usable AOT cache` error and adds zero artifacts. Ordinary JIT recovery adds exactly the needed global artifact; a following independent run-only process succeeds without adding artifacts. Supported FHMoE FP8 small/large cases also pass. FHMoE AOT rejects FP4 activation by its existing support restriction; those attempted failures are preserved.

The real grid-adjustment case has 4,294,967,296-byte A and output tensors. Runtime and AOT agree on global A, `persist_m=3`, and `accumulate=False`; eager/Graph output checks pass. FakeTensor compile checks for small, large and grid cases record zero peak Torch GPU allocation. The common two-artifact CSV result reflects the existing bias=False/True jobs, not dual A variants; FHMoE emits one, and the grid case adds the existing reduction artifact.

Host measurements use a real small public stage2 call, including the required atomic output reset. Synchronization and input preparation stay outside timed CPU sections. They are not full-FMoE CPU timings. Each phase records 1,000 warmed calls; metadata uses 21 blocks of 100,000 evaluations. Cold/load values below are one first-call measurement per fresh process, not statistical build-cost guarantees.

| Variant / process phase | First call ms | Metadata ns | Warmed eager CPU µs (p10 / p50 / p90) | Graph replay CPU µs (p10 / p50 / p90) | Artifact bytes |
| --- | ---: | ---: | --- | --- | ---: |
| base / cold | 1465.329 | 70.813 | 21.030 / 21.649 / 24.379 | 5.100 / 6.260 / 7.900 | 577804 |
| base / load | 269.804 | 68.043 | 20.989 / 21.579 / 22.950 | 7.110 / 9.490 / 14.490 | 577804 |
| head / cold | 1500.940 | 70.431 | 21.449 / 21.930 / 22.779 | 5.890 / 7.050 / 8.689 | 578547 |
| head / load | 245.200 | 70.416 | 21.729 / 22.239 / 24.389 | 5.690 / 6.750 / 8.670 | 578547 |

All four cost runs observed zero Python stage2 invocations during Graph replay. Raw samples are in `base_cost_{cold,load}.log` and `head_cost_{cold,load}.log`; build/load/recovery commands and outcomes are in [AOT results](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/aot/results.json), [additional AOT results](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/aot/more_results.json), [FH run-only/cost outcomes](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/after_large.json), and [FakeTensor allocation evidence](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/aot/memory.log). No host-cost threshold was invented.

## Checks and review

`git diff --check`, scoped Ruff, Python compilation, and mypy for the added test pass. Mypy on the six production files retains 25 baseline diagnostics; Ruff on the shared builder retains 15 baseline E402 diagnostics. They were not suppressed or represented as a clean repository-wide check. The source/test change did not require full-repository CI; the relevant test suites and original inventory were executed.

### Standards

No remaining documented production standards violation was found. Review found that the test Cartesian product admitted unsupported FHMoE/FP8-weight combinations and aborted before its table. The test now warns and skips that combination before launch; the mixed-axis sweep passed. No material heuristic smell findings remain. The [Standards review record](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/standards_review.md) retains the resolved finding.

### Spec

No concrete production implementation mismatch or scope expansion was identified. Addressing, cache/default behavior, unchanged APIs, AOT, Graph and compilation-evidence requirements were checked. One P2 acceptance finding remains: selected large FP8 full-FMoE configurations have intermittent nonfinite output on both versions. Standalone GEMM2 success does not close it. Complete spec acceptance is therefore partial. The [Spec review record](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/spec_review.md) retains the P2 finding. Standards has zero remaining findings; Spec has one remaining P2 finding.

## Reproduction and artifacts

All validation artifacts remain in the workspace under `.scratch/fmoe-gemm2-a-buffer-dispatch/validation/`; large raw artifacts are not added to the code commit. The standalone regression is committed as `op_tests/test_flydsl_moe_large_buffer.py`. Representative commands are:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python op_tests/test_flydsl_moe_large_buffer.py --tokens 64 233016 233017 --a-dtype fp8 --modes atomic reduce --families moe fhmoe
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python op_tests/test_flydsl_moe_large_buffer.py --tokens 64 524287 524288 524289 --a-dtype fp4 --modes atomic reduce --families moe fhmoe
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest -q op_tests/test_fhmoe.py
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python op_tests/test_flydsl_moe_aux.py
```

The actual orchestration scripts record cache isolation, locks and commands: [original comparisons](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/run_perf.py), [independent retests](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/retest_fresh.py), [matched constructed FMoE](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/run_large_matched.py), [full-size FP8 stage2](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/run_large_stage2.py), [Graph checks](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/run_graphs.py), [AOT checks](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/aot/run_aot_checks.py), [IR generation](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/compile_evidence.py), and [cost checks](../../.scratch/fmoe-gemm2-a-buffer-dispatch/validation/run_after_large.py). The earlier red test, harness/import failures, discarded probes and retries remain separately labeled in that directory.

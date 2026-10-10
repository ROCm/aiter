# MXFP4 staged port: recovered facts

Read-only code investigations completed on 2026-10-10. No migration code or GPU tests were run in these investigations; the subsequent crossed-pair experiment is linked from the published [spec.md](spec.md). Source locations and planning-state statements below describe their historical investigation point, not necessarily the current working tree or final acceptance decisions.

## Baselines

| Purpose | Commit |
| --- | --- |
| Current target main | `882ea79dd7cc7091f859a5bedac914523ac77bbd` |
| Earlier local main used for the feasibility test | `4faf87137cfb5e4d5538e010bb4fefa6c8c5df8d` |
| Remote main discovered by the interrupted agent | `9ec2a5b1f19e655079db0b840db7f749569303b5` |
| Source base | `3f3d5f63034ef524746b2271820a9e9ac53357f4` |
| Source tip | `f68e08528cb85f7f3f7211d3f8a1664660bb7f50` |
| Source/main merge base | `10b192f5b5bda90f2af33ceae7a6c2f416bfc674` |

The source tip is not an ancestor of current main. Differences between them are parallel history, not evidence that main later reverted the staged implementation. Source commits `3d8c82283` and `f68e08528` add and revert GEMM1 pruning; the resulting source tree equals `961b5eb79`.

Between `4faf87137c` and the current target, the relevant MXFP4 tuner, candidate registry, name parser, G1/G2 implementation and dispatcher, fused-MoE runtime, AOT, auxiliary kernels, and tuning tests have no source changes. The changed FlyDSL files are under `kernels/mega_moe/`. Between `9ec2a5b` and current main there is one MegaMoE tuning/CI commit, `882ea79dd` (#6259). This limits the new compatibility questions, but does not turn the earlier temporary merge's test results into current acceptance evidence.

## Current main behavior

All locations in this section refer to `882ea79dd7`. `tuner` means `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`.

| Area | Verified behavior | Evidence |
| --- | --- | --- |
| CLI | `--mxfp4-search-mode prune|full` controls GEMM1 selection and defaults to prune. Full removes M_est pruning. Both currently validate every pipeline pair in the selected range. | tuner:508, 525, 6588, 6911; introduced by `fe4814c64b1` (#4526). |
| GEMM1 | Registry variants and tuning axes are filtered through the kernel's `_assert_supported`. BK is fixed at 256; M_est is `ceil(token * routing_topk / expert)`. | tuner:6511, 6537, 6556, 6590. |
| GEMM2 source | `_candidate_rows(row, full_search=False)` proposes only layout-v2 GEMM2, shape-filters the registry, and requires `tile_m == GEMM1 BM`. Native GEMM2 and re-tile combinations are excluded. | tuner:6658–6683; `fe4814c64b1`. |
| GEMM2 output handling | Atomic and reduce are available; BM128 additionally has scatter. Reduce and scatter use different scratch layouts and final-output operations. | `aiter/ops/flydsl/moe_kernels.py`:398–413; `aiter/fused_moe.py`:2719, 2739, 2824, 2836; scatter added by `7be37e87a56` (#5812). |
| Effective scatter execution | Scatter names are non-persist in the registry, while the dispatcher forces persistent execution. With `AITER_MXFP4_INTERMEDIATE=1`, eligible layout-v2 scatter uses FP4 payload/scales and quantized scatter. | `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py`:817; `aiter/fused_moe.py`:2712, 2811. |
| Parser information | `parse_g2_kname_any` exposes atomic versus non-atomic but loses the reduce/scatter distinction. `parse_flydsl_v2_gemm2_kernel` retains the epilog and launch fields. | `aiter/ops/flydsl/mxfp4_kname.py`:56, 156, 176–187. |
| Gate layout | Production uses runtime `gate_mode`; one G1 name can compile both layouts. Current MXFP4 tuner data preparation remains separated and does not pass interleave to G1. | `aiter/ops/flydsl/mxfp4_kname.py`:100; `aiter/fused_moe.py`:2878; tuner:6701, 6763. |
| Activation | Silu, Swiglu and Situv2 are recognized; others raise. Default Situv2 betas and `AITER_MXFP4_TUNE_SWIGLU_LIMIT` are used. Production has additional limit validation that must be respected when proving deployment. | tuner:6614–6622, 6781, 6846; `aiter/fused_moe.py`:2444, 3252; `aiter/ops/flydsl/moe_kernels.py`:821. |
| Production validation seam | `_port_e2e` already calls production stage1/stage2 functions, using the same BM, Opus prequant, accumulation setup, reverse sorting, default Situv2 betas and swiglu_limit. | tuner:6720–6802. |
| Batch lifecycle | Shared `run` tunes each batch, post-processes and writes/compares. A batch exception stops the loop and skips the trailing sort; the finally block still attempts summary. | `aiter/utility/base_tuner.py`:1451–1517. |
| Input-only metadata | Ordinary input can retain extra columns. `--all` projects onto shape keys and loses gate_mode. MXFP4 results copy only keys, while shared summary selects every input column. | tuner:5913, 6637; `aiter/utility/base_tuner.py`:368, 384, 555. |
| Final output | MXFP4 writes valid winners and retains invalid results in `self.failed`. Its `post_process` does not write `profile_file`, despite broad shared README wording. | tuner:6982–7010; tuner README:99. |
| Process isolation | Main executes single-GPU full search directly and uses its Queue-based fresh-shape supervisor for multi-GPU execution. The source staged Pipe supervisor is absent. | tuner:6957, 6967, 7045. |

The supported scaled-MFMA/auxiliary/AOT path remains gfx950. The dedicated tuner lacks the source branch's preflight support check. Production support for other MoE paths or layouts does not establish this tuner's support.

## Source behavior to reconsider or preserve

Locations in this section refer to `f68e08528`.

| Behavior | Source evidence | Port implication |
| --- | --- | --- |
| Shared pipeline-pair source | tuner:6345; `mxfp4_staged_search.py`:219, 283. | Reuse the selected main enumeration with its explicit `full_search` value; do not rebuild from another registry. |
| Stable screening shortlist | `mxfp4_staged_search.py`:80; tuner:6986. | Positive finite scores, original-order tie breaking and bounded screening top-K are existing strategy requirements. |
| Correctness authority | tuner:6968–7033. | Screening ranks only. Normal end-to-end validation supplies final correctness and timing. |
| Exhaustive group fallback | tuner:6994–7033. | Expand only after no valid shortlist pair, evaluate all remaining pairs, reuse every returned validation outcome. All-screening-failed groups currently skip without fallback. |
| Reference bootstrap | `mxfp4_staged_search.py`:91, 110; tuner:6755–6898. | Deterministic producer/consumer selection, untimed correctness checks and fallback reference selection are existing requirements; recheck against the new candidate graph. |
| Timing | tuner:6594, 6675, 6702, 6726, 6535. | G1 kernel-only screening; candidate-local G2 intermediate; include final-output handling in G2 screening. Final `us1=us`, `us2=0`. |
| Native rules | `mxfp4_staged_search.py`:230, 366. | Old native alignment filters and bare/f4out/cshuffle alias rules have no candidates in main's current enumeration. The new layout-v2 environment branches require separate analysis. |
| Metadata repairs | tuner:6130, 6174, 7196; `aiter/utility/base_tuner.py`:552. | Preserve validation metadata through selection, retain unsupported rows for invalid reporting, and prevent summary from requiring input-only fields. Commits `ae3c6a08b`, `de1bde76d`, `05bf83339`. |
| Staged supervision | tuner:7128, 7263–7356. | Fresh worker per staged shape, private result Pipe, result/exit distinction, GPU release after exit, bounded Process handles, cleanup and original exception propagation. Commits `86b3e07f9`, `bf0b9dcf7`. |
| Timeout | tuner:6088, 6798. | Per-operation best-effort SIGALRM; a still-live permanent hang is not automatically recovered. No parent hard deadline. Heartbeat/watchdog was explicitly superseded. |
| Failed shapes | tuner:7187–7195, 7225. | INVALID_TIME enters in-memory failure/summary handling; failure reason remains console-only. Final CSV does not persist failed-shape records. |

For the current non-EP tuner and a fixed environment, layout-v2 reduce and scatter both overwrite caller-visible `[M,H]` BF16 output, without requiring initial contents. The tuner sends both through `moe_sorting(accumulate=False)` (main tuner:6723, 6743); runtime sorting documents their empty-buffer contract (`aiter/fused_moe.py`:462–475). Reduction starts its accumulator at zero and stores to Y (`aiter/ops/flydsl/kernels/moe_reduce.py`:164, 181–182); scatter likewise starts at zero and stores the final result (`csrc/kernels/mxfp4_moe/moe_aux/moe_scatter_reduce.cuh`:35, 73, 81). Their different scratch layouts and rounding do not establish bitwise equality or require different final-output initialization. This evidence supports discussing a shared non-atomic screening context; the search policy decision is recorded separately in the spec.

## Open source-branch review findings

The [2026-09-08 review](../../docs/mxmoe_gemm1_retune/review_comment/2026-09-08-staged-search-full-range-review.md) identifies issues still present in the source tree:

- Canonical CSV projection drops `_tag` before preserving its filtering semantics; tagged fallback rows can become ordinary entries. Source tuner:7173–7197; review:30–107.
- Shared sorting can label an entire legacy CSV with the current gfx. Missing numeric fields such as xbf16/flat are filled with empty strings, which production cannot reliably convert. Source `base_tuner.py`:455; tuner:7181; review:58–107.
- `get_gfx_runtime()` returns the first rocminfo gfx match and does not validate every scheduled logical GPU or its visible-device mapping. Source tuner:6122; `aiter/jit/utils/chip_info.py`:22–34; review:109–141.

These are existing review findings supported by source inspection, not newly reproduced runtime results in this recovery. The port must establish how to fix them within its compatibility scope.

## Verification boundaries

The old calibration used only `(token=4, model_dim=3072, inter_dim=512, expert=128, routing_topk=4)` with `AITER_MXFP4_INTERMEDIATE=0`. Full ran 2,928 validation pairs and staged 32; both selected the same pair. Its reported −3.48% regret was independent measurement variation, not evidence of a better staged winner. See [validation-04](../mxfp4-staged-tuner/validation-04/validation.md).

The old smoke verifies the exact selected production dispatch and numerical correctness. The broader old suite had seven failures; no all-green claim is supported. Those results must be replaced by acceptance checks on the port.

The old spec sets no numerical performance-regret limit, fixed calibration matrix, or total measurement budget. Its `errRatio=0.1` calibration setting is a correctness threshold, not a performance target. Current main's `minimax_m25_fp4_untuned_fmoe.csv` includes `(H=3072,I=384,E=256,routing_topk=8)` with tokens 64, 128, 512, 1024, 2048, 4096 and 8192, giving M_est 2, 4, 16, 32, 64, 128 and 256. This supplies an existing-model candidate matrix for the new calibration; it has not yet been approved or measured. I384 is legal with G2 K128, and main's auxiliary support rounds I384 to the supported I512 key (`aiter/ops/moe_mxfp4_aux.py`:51, 68). Exact adjacent M_est boundaries can be tested with policy fixtures without adding expensive GPU calibration rows.

Useful current-main seams include `op_tests/tuning_tests/test_tuner_infra.py`, `test_compare_logic.py`, `op_tests/test_moe_glm5_fp4_ep_dispatch.py` for named-kernel dispatch fixtures, `op_tests/tuning_tests/test_run_config.py` for production replay, and `op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py` for layout-v2 aligned-shape behavior. Main has no dedicated staged policy suite or staged pipeline smoke. The existing generic multiprocessing suite does not cover the dedicated supervisor.

The local package metadata differs from target requirements and lacks FlyDSL/einops. No packages were installed during planning. Test implementation must separate policy checks from runtime/toolchain validation and establish the required environment before claiming GPU acceptance.

Calling the production stage functions does not by itself establish identical effective execution. In `AITER_FLYDSL_STAGE2_FP8=1`, production passes `topk_weights` (`aiter/fused_moe.py`:4345–4355), while `_port_e2e` does not (tuner:6787–6802); the wrapper therefore disables its FP8-reduce branch during tuning (`aiter/fused_moe.py`:2707–2708). Enabled FP4 scatter also has an AOT gap: `aiter/aot/flydsl/mxfp4_moe.py`:204–207 chooses FP8-reduce or BF16 output and :371–374 prepares BF16 scatter buffers, rather than FP4 payload/scales. Non-default swiglu_limit enters the G1 compile key, while AOT :189–190 uses fixed 7/infinity. Any claim covering these options needs additional tuning/production/AOT agreement. Production configuration-cache handling also requires explicit verification.

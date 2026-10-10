# Configured paths GPU acceptance

All22 prescribed pairs completed normal pair, exact public production, matching FlyDSL AOT and exact run-only replay in88 sequential independent tasks on physical GPU7 (UUID `GPU-5d8a72a332aa124c`, PCI `0000:f5:00.0`, gfx950/256 CU). Each task retained its identity/idle samples, process start/exit, environment and commit. Every task exited0.

Numerics use the unchanged Torch reference and finite cosine_diff <=0.1; the deliberately nonuniform route weights are1..topk normalized to sum1. Exact metadata and actual A4W4 paths are asserted. Swiglu limit5 uses active clamp data with hidden multiplier16 and matching input quantization. Two additional independent production invocations with matching limits5/7 also passed: fixed input/routes have identical hashes, actual output hashes differ, and both ordinary Torch references pass (.00755436/.00911617). The limit5 call uses exact run-only AOT; the limit7 control compiles its matching normal entry. Their commands, output hashes/scalars and comparison live under `clamp-control/`.

| Fixture | Pair error | Kernel sum μs | Pair wall s | Production error | Run-only error | AOT jobs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| minimax128-bm16-atomic | 0.01128391 | 88.1325 | 6.674 | 0.01128354 | 0.01128365 | 4 |
| minimax128-bm32-reduce | 0.01127915 | 104.4056 | 26.675 | 0.01127915 | 0.01127915 | 4 |
| minimax2048-bm64-atomic | 0.01117674 | 201.9535 | 7.158 | 0.01117664 | 0.01117671 | 4 |
| minimax2048-bm128-reduce | 0.01117271 | 189.8507 | 9.224 | 0.01117167 | 0.01117167 | 4 |
| minimax2048-bm128-scatter | 0.01117413 | 234.4953 | 5.220 | 0.01117190 | 0.01117552 | 4 |
| gpt128-fp8-bm16-k128 | 0.01174294 | 63.7405 | 6.504 | 0.01174294 | 0.01174294 | 4 |
| gpt128-fp8-bm32-k256 | 0.01174294 | 66.8635 | 5.599 | 0.01174294 | 0.01174294 | 4 |
| gpt2048-fp8-bm64-k128 | 0.01180350 | 116.3891 | 6.879 | 0.01180350 | 0.01180350 | 4 |
| gpt2048-fp8-bm128-k256 | 0.01180493 | 111.9463 | 8.867 | 0.01181705 | 0.01180350 | 4 |
| gpt2048-fp4-n128 | 0.01814416 | 138.0449 | 9.759 | 0.01814639 | 0.01814750 | 4 |
| gpt2048-fp4-n256 | 0.01814495 | 123.6319 | 7.142 | 0.01814416 | 0.01814856 | 4 |
| gpt2048-limit5-reduce-clamp | 0.00755293 | 129.2256 | 9.520 | 0.00756395 | 0.00754774 | 4 |
| gpt2048-combined7-reduce | 0.01180537 | 116.7624 | 9.089 | 0.01180464 | 0.01181451 | 4 |
| gpt2048-combined7-scatter | 0.01815661 | 139.8082 | 6.053 | 0.01814416 | 0.01814452 | 4 |
| gpt2048-combined5-reduce-clamp | 0.00793011 | 118.4313 | 9.412 | 0.00790670 | 0.00791230 | 4 |
| gpt2048-combined5-scatter-clamp | 0.01437949 | 123.7924 | 7.178 | 0.01436604 | 0.01436047 | 4 |
| gpt2048-situv2-reduce | 0.01110149 | 129.7578 | 9.809 | 0.01110692 | 0.01109902 | 6 |
| minimax128-bm16-reduce | 0.01127915 | 91.6373 | 4.484 | 0.01127915 | 0.01127915 | 4 |
| minimax128-bm32-atomic | 0.01128349 | 96.4334 | 4.491 | 0.01128318 | 0.01128394 | 4 |
| minimax2048-bm64-reduce | 0.01117167 | 195.0431 | 5.067 | 0.01117167 | 0.01117167 | 4 |
| minimax2048-bm128-atomic | 0.01117747 | 205.9212 | 5.514 | 0.01118539 | 0.01117666 | 4 |
| gpt2048-combined5-atomic | 0.01147297 | 135.7924 | 5.429 | 0.01145764 | 0.01145757 | 4 |

Timing labels remain distinct. The CSV stores the normal continuous `run_perftest` kernel-duration sum in `us1` and `us`, with `us2=0`. Synchronized diagnostic kernel sums, real-kernel trace spans and elapsed medians come from matched diagnostic calls and remain separate fields. Pair wall time includes preparation, correctness, warmup, profiling and compilation; it supplies actual attempt-cost evidence for calibration planning.

FP8 reduce checks captured uint8 route payloads with embedded E8M0 scales and final reduction receiving the original nonuniform weights once. FP4 scatter checks captured uint8 `[max_sorted,H/2]` payload and `[max_sorted,H/32]` scales for N128/N256, followed by complete quantized scatter. Both-flags/default7 and custom5 reduce/scatter paths pass; combined atomic remains BF16. AOT preserves shared BF16/bias variants and persistent scatter CU keys.

Situv2 evidence here explicitly supplies the tuner reference defaults beta4/linear_beta25 with A4W4 input. AOT retains those and API1/1 variants. Omitted public API parameters1/1 are a separate final acceptance check; these records do not claim that omitted call was already executed.

Full result/configuration/job details are in `acceptance-summary.json`, per-fixture results and the retained full traces. Current-port evidence is independent of historical experiment results. The88tasks process wall sum is715.600435s, with first start15:10:09.225931Z and last exit15:35:35.952397Z. The wider interval includes physical-card idle waits.

Reproduce from the retained integration checkout with `run_acceptance.py --root /path/to/checkout --python /path/to/toolchain/python --runtime /path/to/external/cache`. Existing completed task records are reused only as recorded acceptance results; a fresh acceptance should use a fresh copied result directory/cache. The physical device controller is loaded from the selected root.

The final task sequence includes both clamp controls:90 completed processes, all exit0, no overlap,731.86s summed process wall time. The88 prescribed pair/deployment actions are retained separately from these two numerical controls. Fixture CSV rows are validated specified-pair measurements; they are not search winners and do not substitute for the27 calibration searches.

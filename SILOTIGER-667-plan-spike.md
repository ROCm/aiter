**Gather gate/up + native down**, same preshuffled weights. Default stays as HEAD until the numbers say otherwise.

**Why:** native `gate_up` is `B×TOPK×INTER/16` waves (Qwen B=1 ≈ 1.25 waves/CU, ~2.5× CK). Native `down` already has enough waves. Old-grid `gate_up` (`B×TOPK×INTER`, strided 16 B loads) was never timed together with native `down`. The combined API uses one `weight_layout` for both stages, so this mix cannot be run today.

**Change (small):**
- On `weight_layout=preshuffled`, let `gate_up` choose `native` (HEAD) or `gather` (the `50bd0e972` path: `_kpack_load_i32_words`, fat grid).
- `down` stays native on legal tiles.
- Combined wrapper: two map args, one shared `preshuffled` layout. Default both `native`.

**Correctness:** op_test, cache-off, GPU 6. Gather `gate_up` vs current native vs k-contiguous, cos ≥ 0.999.

**Perf (GPU 6, cold, loaded SCLK ≳ 2 GHz, same timer):**
1. G9 `--flydsl-weight-layout preshuffled --shapes qwen3next,deepseek-v3 --batches 1,2`
   native `gate_up` vs gather `gate_up`; `down` native both times. FP8 and FP4.
2. Combined `tickets/667/bench/bench_moe_warp_decode.py`, preshuffled, cold:
   HEAD (native+native) vs mix (gather `gate_up` + native `down`).
   100 iters to screen, 1000 to claim.

**Stop / ship:**
- If gather `gate_up` is not faster on Qwen B=1 FP8 (G9), stop. WontFix.
- Ship as default only if **combined** time wins (or noise) on Qwen **and** DeepSeek, B=1 **and** B=2.
- If only Qwen wins: auto by occupancy (gather `gate_up` when native waves/CU < 2), not a global default.
- Do not put `down` back on gather.

**Do not do in this experiment:** split-K, 32-row, unpacking weights, FP4 `dot2_acc=2`.

## Result (2026-09-15): WontFix

Implemented the selectable preshuffled `gate_up_map={native,gather}` while
keeping legal-tile `down` native. Cache-off correctness passed against
k-contiguous for FP8 and FP4, including the combined wrapper.

GPU 6 cold screen: `timing=device`, 100 iterations, 20 warmup, one repeat,
disjoint-expert router rotation. Median loaded SCLK was 2265 MHz for native
and 2339 MHz for gather.

| shape | B | gate_up | dtype/act | native us | gather us | gather/native |
|---|---:|---|---|---:|---:|---:|
| Qwen3Next | 1 | gate_up | FP8/BF16 | 18.0762 | 24.8417 | 1.374 |
| Qwen3Next | 1 | gate_up | FP8/FP8 | 18.6213 | 23.7278 | 1.274 |
| Qwen3Next | 1 | gate_up | FP4/BF16 | 16.5757 | 14.8969 | 0.899 |
| Qwen3Next | 2 | gate_up | FP8/BF16 | 19.3814 | 48.8147 | 2.519 |
| DeepSeek-V3 | 1 | gate_up | FP8/BF16 | 71.6219 | 264.0457 | 3.687 |
| DeepSeek-V3 | 2 | gate_up | FP8/BF16 | 85.9533 | 530.9237 | 6.177 |

The G9 canary failed: gather was not faster on Qwen B=1 FP8.

### Combined screen (also tried): WontFix

Ran anyway so the mix is not an unmeasured hole. GPU 6 cold,
`tickets/667/bench/bench_moe_warp_decode.py`, 100 iters, 20 warmup,
`run_perftest` device time. HEAD = native `gate_up` + native `down`. Mix =
gather `gate_up` + native `down`. Ship metric = wrapper `core_us`. The mix
does not ship:

| shape | B | path | native core_us | mix core_us | mix/native |
|---|---:|---|---:|---:|---:|
| Qwen3Next | 1 | flydsl_fp8 | 35.7 | 41.7 | 1.17 |
| Qwen3Next | 2 | flydsl_fp8 | 37.6 | 63.9 | 1.70 |
| DeepSeek-V3 | 1 | flydsl_fp8 | 105.4 | 290.2 | 2.75 |
| DeepSeek-V3 | 2 | flydsl_fp8 | 144.8 | 580.1 | 4.01 |
| Qwen3Next | 1 | flydsl_fp4 | 28.3 | 26.3 | 0.93 |
| Qwen3Next | 2 | flydsl_fp4 | 30.2 | 39.3 | 1.30 |
| DeepSeek-V3 | 1 | flydsl_fp4 | 75.8 | 155.1 | 2.05 |
| DeepSeek-V3 | 2 | flydsl_fp4 | 112.1 | 301.8 | 2.69 |

Staged `down_us` stayed native (unchanged within noise). Qwen B=1 FP4 is the
only mix win; it is not enough to ship (needs Qwen and DeepSeek, B=1 and B=2).

### Combined 1000-iter claim: still WontFix

Same bench, GPU 6, cold, 1000 iters, 20 warmup. Same verdict as the 100-iter
screen. Mix still loses every FP8 cell and every DeepSeek cell.

| shape | B | path | native core_us | mix core_us | mix/native |
|---|---:|---|---:|---:|---:|
| Qwen3Next | 1 | flydsl_fp8 | 34.6 | 38.7 | 1.12 |
| Qwen3Next | 2 | flydsl_fp8 | 36.4 | 63.8 | 1.75 |
| DeepSeek-V3 | 1 | flydsl_fp8 | 106.2 | 374.5 | 3.53 |
| DeepSeek-V3 | 2 | flydsl_fp8 | 145.3 | 579.9 | 3.99 |
| Qwen3Next | 1 | flydsl_fp4 | 28.5 | 25.2 | 0.88 |
| Qwen3Next | 2 | flydsl_fp4 | 29.7 | 38.1 | 1.28 |
| DeepSeek-V3 | 1 | flydsl_fp4 | 75.4 | 155.4 | 2.06 |
| DeepSeek-V3 | 2 | flydsl_fp4 | 111.2 | 302.0 | 2.72 |

Qwen B=1 FP4 still the only mix win (slightly larger than at 100 iters). Keep
native as the default; do not add occupancy auto-dispatch and do not change
down.
# C8 AITER-guided region tuning — 2026-10-21

Selected isolated single-launch regions. Four GPUs used. All µs are second-round medians per eight-token decode batch. Native is installed ATOM/AITER dispatch.

| Group | Native µs (fresh) | Old fused µs | New fused µs | New vs native |
|---|---:|---:|---:|---:|
| SWA attention + reduce, 128 KV | 7.57 | 39.78 | 8.19 | +8.2% |
| HCA attention + reduce, 136 KV | 8.04 | 38.98 | 8.27 | +2.8% |
| HCA attention + reduce, 144 KV | 7.95 | 39.35 | 8.41 | +5.8% |
| CSA attention + reduce, 384 KV | 9.73 | 36.46 | 9.48 | -2.6% |
| CSA attention + reduce, 512 KV | 10.90 | 35.01 | 10.12 | -7.2% |
| CSA attention + reduce, 640 KV | 9.71 | 34.70 | 10.68 | +10.0% |
| mHC post + FFN pre/RMSNorm | 4.99 | 11.65 | 5.28 | +5.6% |
| Top6 routing + metadata | 7.60 | 12.05 | 6.41 | -15.6% |
| Routed MoE including input quant | 49.66 | 76.79 | 57.22 | +15.2% |
| HCA compressor non-boundary | 7.17 | 6.69 | 6.87 | -4.2% |
| HCA compressor boundary | 10.16 | 19.90 | 14.03 | +38.1% |
| Indexer compressor non-boundary | 5.82 | 6.55 | 5.86 | +0.7% |
| Indexer compressor boundary | 7.28 | 8.77 | 8.03 | +10.3% |

## Selected changes

- Attention: eight splits; row/head/split task interleaving across the resident grid; no empty-split stores; direct paged-cache TDM gathers and direct current-KV substitution. Bounded per-task TDM L2 prefetch retained. The separate 6–17 µs compact-cache copy is absent from this new region.
- mHC: inline AITER gfx1250 fused post/projection/squared-sum producer and original warp-specialized reduce/apply kernel in one launch. Dedicated producer subgroup barrier. Exact outputs in all16 tested fixtures.
- Routing: native-style parallel per-row topk; histogram/offset computation distributed across row CTAs rather than a serial CTA0 loop. 64 checks cover batches1/2/4/8, bias, and equal-score ties.
- MoE: match native K512/two-buffer pipeline; widen N128 to N256 for the fixed256 resident grid; use device-coherent load polling and actual active-expert task bounds. N512 and two/four persistent-N-tile variants were tested and rejected.
- HCA compressor: native-style K-parallel pooling, distributed across four CTAs per token; merge partial softmax statistics before norm/RoPE/cache write. Dedicated producer subgroup barriers.
- Indexer compressor: limit producer barrier to the32 producing CTAs; preserve sequential pooling arithmetic after parallel pooling regressed.

## Remaining work

Full-layer embedding and end-to-end validation are separate outstanding steps. Selected region ABIs/scratch need wiring; the serving kernel was not silently changed. The remaining >10% gaps are routed MoE, HCA boundary compression, and indexer boundary compression (CSA640 is approximately10%). Do not sum these ablations as full-layer latency.

## Validation and limitations

- Before column is previous fixed-fixture ablation at corresponding shapes; newly paired native/after columns are the current controlled comparison. Attention new cache is scattered/paged, old cache was compact.
- 8 splits changes the attention reduction grouping versus native default32. Ragged stress max relative RMS 0.001486 versus default32, <=0.000009887 versus AITER same8 splits. No modelwide accuracy claim.
- Native mHC warp-specialized reducer is retained; full-layer embedding can change thread/register/LDS requirements.
- HCA compressor subgroup uses a dedicated counter and extra 8x4x3x512 FP32 pooling scratch, only STATE128/R128 boundary alignment tested. Do not share this counter with a 256-CTA barrier.
- Routing is learned top6 with optional bias, not hash routing; 64 metadata comparisons passed.
- MoE selected BN256/BK512/two TDM buffers, 256 resident CTAs, 297 VGPR/157152B LDS/0 spills; all8 random/collision fixtures bitwise match native.
- No new end-to-end TTFT/TPOT/tokens/s run in this task. Existing serving path remains the previous saved version.

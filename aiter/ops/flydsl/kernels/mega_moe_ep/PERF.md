# MegaMoE EP performance vs PR 6881 (MegaMoEV2 small_m / large_m)

Baseline: [ROCm/aiter#6881](https://github.com/ROCm/aiter/pull/6881), the two-stage
`MegaMoEV2` with its small-m and large-m paths enabled (`AITER_MEGA_SMALL_M=1
AITER_MEGA_LARGE_M=1`). Lower is better everywhere except throughput; the percentage
is EP relative to the PR.

## Conditions

- 8x MI355X (gfx950), one rank per GPU; torch 2.10 (ROCm 7.2.4), flydsl 0.4.1.
- DeepSeek-V4-Pro MoE shape: hidden 7168, inter 3072, 384 experts (48 per rank), top-6,
  SiLU with swiglu_limit 10, a8w4 (MXFP8 activations, MXFP4 weights).
- Single layer: `op_tests/multigpu_tests/bench_mega_moe_ep.py` (`--engine ep`, and
  `--engine v2` with the PR tree on `PYTHONPATH`), both engines at the same
  `max_tok_per_rank` (mtpr). Mean of 20 CUDA-graph replays after 5 warm-up replays,
  wall clock of the worst rank, averaged over 2 rounds. "bs" is tokens per rank;
  differences below ~1% are noise.

## Routings

- **uniform**: top-6 of softmax(randn) logits; every rank and expert evenly loaded.
- **zipf:S**: the expert of popularity rank r (a fixed random permutation of the 384
  experts) gets a logit bias of -S * log(1 + r). The hot experts land on different
  ranks, so the load is uneven *between* ranks: mean / max rows per rank is 0.54 /
  0.37 / 0.31 for S = 0.5 / 1 / 2, and the hottest expert gets 31 / 59 / 64 x the
  mean expert load.
- **spike** (`spike:0.8,0.5,0.35,0.26,0.2,0.17`): the shape the full model shows after
  expert-load rebalancing (EPLB). Hot expert j lives on rank j and is picked by a
  fraction p_j of all tokens (80 / 50 / 35 / 26 / 20 / 17 %); the remaining routes go
  to the other experts with rank weight max(6/8 - p_r, 0.1), i.e. EPLB fills the hot
  ranks less. Ranks stay fairly balanced (mean / max 0.83) but a hot rank's rows mostly
  belong to *one* expert, so the imbalance is *within* a rank (hottest expert 51 x
  the mean). zipf cannot produce this shape.
- **rb122 / pre61**: top-k ids dumped from the full-model run (below) at bs 8192 per
  rank, replayed layer by layer (sum of the per-layer times, ms).
  - rb122: 2 prefill steps x 61 layers right after the first EPLB rebalance. Split
    into the 65 "hot" layers (max / mean expert load >= 50) and the 57 "cold" ones.
  - pre61: 1 prefill step x 61 layers before any rebalance (mean / max rows per rank
    about 0.55, inter-rank imbalance like zipf).

## Single layer, mtpr 8192 (ms)

| route | bs1024 PR | bs1024 EP | bs2048 PR | bs2048 EP | bs4096 PR | bs4096 EP | bs8192 PR | bs8192 EP |
|---|---|---|---|---|---|---|---|---|
| uniform | 0.933 | 0.736 (-21%) | 1.391 | 1.118 (-20%) | 2.345 | 1.982 (-15%) | 4.205 | 3.849 (-8%) |
| zipf:0.5 | 1.641 | 1.074 (-35%) | 2.417 | 1.808 (-25%) | 3.995 | 3.468 (-13%) | 7.647 | 6.756 (-12%) |
| zipf:1 | 1.664 | 1.301 (-22%) | 2.840 | 2.452 (-14%) | 5.443 | 4.860 (-11%) | 10.816 | 9.673 (-11%) |
| zipf:2 | 1.891 | 1.462 (-23%) | 3.361 | 2.782 (-17%) | 6.394 | 5.693 (-11%) | 12.794 | 11.329 (-11%) |
| spike | 1.329 | 0.917 (-31%) | 2.150 | 1.446 (-33%) | 2.726 | 2.513 (-8%) | 5.159 | 4.905 (-5%) |

Full-model routings, bs 8192 (ms, sum over layers):

| routing | PR | EP |
|---|---|---|
| rb122, 65 hot layers | 362.1 | 350.2 (-3.3%) |
| rb122, 57 cold layers | 285.3 | 255.2 (-10.6%) |
| rb122, all 122 layers | 647.4 | 605.4 (-6.5%) |
| pre61 | 479.7 | 439.5 (-8.4%) |

## Single layer, mtpr 32768 (ms)

| route | bs8192 PR | bs8192 EP | bs16384 EP | bs32768 PR | bs32768 EP |
|---|---|---|---|---|---|
| uniform | 4.548 | 3.982 (-12%) | 8.001 | 16.834 | 16.316 (-3%) |
| zipf:0.5 | 8.270 | 6.984 (-16%) | 14.041 | 38.564 | 27.819 (-28%) |
| zipf:1 | 11.024 | 10.017 (-9%) | 19.676 | 43.864 | 39.951 (-9%) |
| zipf:2 | 12.795 | 11.673 (-9%) | 23.779 | 53.903 | 48.451 (-10%) |
| spike | 5.418 | 5.051 (-7%) | 10.136 | 24.003 | 20.386 (-15%) |

## Full model

ATOM serving DeepSeek-V4-Pro a8w4 on the same 8 GPUs, EP 8 with EPLB, CUDA graph
FULL, prompts 8192 in / 1024 out, no profiler (except the decode-call row); the same
server configuration for both (concurrency 512: dual-stream MoE off), only the MoE
layer swapped. These runs predate the token-bucket dispatch (the single-layer
gains above at bs >= 4096 and on rb122 / pre61 are not in them yet).

| run | metric | PR | EP |
|---|---|---|---|
| concurrency 4096, 8192 prompts | output throughput (tok/s) | 79,183 | 78,700 (-0.6%) |
| | TTFT mean (s) | 194.5 | 200.0 (+2.8%) |
| | TPOT mean (ms) | 275.1 | 272.4 (-1.0%) |
| concurrency 512, 2048 prompts | output throughput (tok/s) | 55,181 | 55,648 (+0.8%) |
| | TTFT mean (s) | 25.8 | 26.6 (+3.2%) |
| | TPOT mean (ms) | 58.3 | 56.8 (-2.6%) |
| concurrency 512, decode MoE call (profiler) | us per call | 293.2 (small_m) | 262.6 (-10.4%) |

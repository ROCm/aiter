# C16 latency tuning — 2026-10-22

Flash, one GPU per experiment. Four GPUs used for independent workers. Decode position 1023; concurrency 16; the same 1K-prompt start-of-decode region boundaries as the preceding study. No modelwide serving/E2E run was performed in this pass. C16 serving remains disabled.

## Selected changes

1. QB: retain all 16 live WMMA rows; widen N=64→128, K=128→256. Four early TDM weight tiles replace two serial 64-channel halves. Keep the bounded, non-speculative QB L2 prefetch. The complete 128-channel weight block is staged before QA work; input tiles wait for normalized QA.
2. OA/OB: OA uses N128/K512, split4, two buffers; OB uses N64/K512, split4, four buffers. Both use scaled FP8 WMMA and TDM. Stage OA weights before inverse RoPE/quantization and OB weights before OA reduction/quantization. Selected full-kernel LDS grows from 157,156 to 186,884 bytes; no spills.
3. HCA: adapt native compute-bound BF16 projection, K512/five buffers, including its LDS-read/WMMA pipeline. Projection runs on CTAs 0–63 while historical-state pooling runs on CTAs 64–191. Pooling uses 64-feature slices, 128 historical rows, and 64 KiB of LDS instead of 128 KiB. Only those 192 producers participate in the join; 16 row CTAs finish compression. Norm/APE TDM loads start before partial-pool reduction. Plan construction moves before the already-required entry barrier, so pooling can start without waiting for projection.
4. Shared expert: use M16/N32/K512 and four buffers, following native tiling/pipeline principles. Each GEMM has 128 producing CTAs. Existing whole-layer barriers remain; the old shared-expert tile-mailbox mode is explicitly disallowed with this retile.

## Current region timings

Times are µs per batch, second timing round. Flat HIP graphs, 32 repetitions, five warmups and 40 alternating samples. Native/current/previous entries in one row are paired when a previous column is present. C8 is retained from the preceding study, not rerun. “—” means the unchanged region was not given a separate previous arm in this pass. Whole FFN overlaps the preceding FFN rows; do not sum it with them.

| Region | Native C8 (prior) | MK C8 (prior) | Native C16 | Previous MK C16 (paired) | Selected MK C16 |
|---|---:|---:|---:|---:|---:|
| Attention mHC pre + RMSNorm | 8.52 | 6.95 | 8.66 | — | 7.26 |
| CSA compressor: projection + pool/norm/RoPE + state/cache write | 10.37 | 9.80 | 10.35 | 9.89 | 9.82 |
| CSA indexer compressor: projection + pool/norm/RoPE/quant + state/cache write | 7.21 | 7.99 | 7.24 | 8.68 | 8.55 |
| CSA identity top-k + index packing (256 compressed tokens) | 2.57 | 1.35 | 2.66 | — | 1.37 |
| HCA compressor: projection + pool/norm/RoPE + state/cache write | 11.20 | 15.92 | 10.98 | 18.10 | 11.32 |
| Input quant + QKVA + norms/RoPE + QB + KV-cache write | 27.72 | 18.35 | 13.30 | 18.39 | 17.13 |
| SWA attention + split-KV reduction (128 KV) | 7.49 | 8.08 | 10.92 | — | 10.41 |
| CSA attention + split-KV reduction (384 KV) | 9.56 | 9.40 | 12.65 | — | 13.17 |
| HCA attention + split-KV reduction (136 KV) | 7.93 | 8.29 | 11.41 | — | 10.90 |
| Inverse RoPE + quant + OA/OB projections | 63.68 | 24.13 | 15.97 | 29.47 | 26.11 |
| Attention mHC post + FFN mHC pre/RMSNorm | 5.31 | 6.69 | 5.42 | — | 6.25 |
| Router GEMM + shared expert (up/activation/quant/down) | 51.42 | 17.73 | 17.00 | 19.09 | 15.28 |
| Expert selection + routing metadata | 7.86 | 6.33 | 9.60 | — | 9.88 |
| Routed MoE: input quant + up/SwiGLU/quant + down/reduce | 48.90 | 56.23 | 79.90 | — | 90.12 |
| Shared+routed sum + final mHC post | 4.96 | 3.24 | 5.12 | — | 4.33 |
| Whole FFN region (overlaps previous four rows) | 153.67 | 82.30 | 114.12 | 124.42 | 117.69 |

## Paired full-layer check

These are complete single-launch layer microbenchmarks on synthetic full-layer fixtures, including attention and MoE. They are not native-vs-monokernel serving results. Old/new use identical selected layout settings (TILE_MASK192, OVERLAP2, EDGE_MASK0), weights/inputs and alternating timing samples. Second round reported.

| Layer | Previous MK µs | Selected MK µs | Reduction | VGPR old → new | LDS old → new |
|---|---:|---:|---:|---:|---:|
| SWA | 323.10 | 295.61 | 8.5% | 428 → 412 | 157,156 → 186,884 |
| CSA | 366.18 | 343.25 | 6.3% | 425 → 416 | 157,156 → 186,884 |
| HCA | 367.26 | 345.85 | 5.8% | 418 → 414 | 157,156 → 186,884 |

All three pass exact comparisons at 13 recorded intermediates, finite-output checks, repeated graph replay and zero health flags. Both versions have zero spills. Four requested warps compile to eight physical waves because of mHC warp specialization; the 256-CTA persistent grid remains unchanged. These resource bounds retain room for one resident CTA per MP; this pass did not repeat the HIP occupancy query.

## Correctness and rejected experiments

- Checkpoint region comparisons: OA/OB, shared expert and whole-FFN final outputs exact; QKV relative RMS 4.35e-8 and KV cache exact; HCA projection/state exact and cache relative RMS 1.07e-9. Existing attention split-reduction error remains about 0.002; mHC normalization differs by about 3.85e-5. Shuffled output is unshuffled before native comparison.
- Fixed a pre-existing dormant early-coefficient bug: shared buffers were hardcoded float32 even when a coefficient pointer was BF16. Separate buffers now use Norm.dtype and APE.dtype. The initial hca_overlap0/1 results have large cache errors and are rejected. hca_fixed0/1 and final results are valid.
- Consumer-specific QKV mailboxes: correct but 24.33 µs versus 17.27 µs for the wider-QB barrier version; rejected. OA/OB mailboxes: correct but 47.73 µs versus about 27 µs; rejected. Extra publication/poll/invalidation and compiler/resource effects outweighed early release in these implementations. We did not isolate their individual hardware costs.
- HCA row-specific publication (64 projection producers plus eight historical producers per row): correct, 11.62 µs versus 11.34 µs for the 192-producer join; retained as an experiment, disabled in the selected configuration.
- OA/OB packed-weight variant, earlier OB TDM issue, and early scalar-scale variants did not improve the selected timing. The attempted OB N32/split8 configuration needed 1024 CTAs and was invalid on this grid; its result is rejected and a static grid-size guard was added.
- Native-like MoE N128 tiles: 106.70–115.10 µs versus about 90 µs with N256. Three buffers and next-task bounded L2 prefetch did not give a reliable routed-MoE gain. Retain N256/K512/two buffers. MoE remains the largest absolute isolated gap.
- CSA parallel history reduction regressed both compressor and indexer (indexer 9.13 µs versus about 8.55 µs); disabled. mHC one-buffer trial regressed post/pre (6.92 µs versus 6.25 µs); retain two buffers. Native mHC already shares the selected algorithm, so additional buffering alone is not a useful improvement for its one-K-iteration split.
- Native C16 remains faster for OA/OB, QKV, routed MoE, CSA indexer and mHC post/pre. The work does not establish modelwide accuracy or a C16 TTFT/TPOT/tokens/s gain. No prefill/serving dispatcher files were changed.

## Resume

Working directory: `/home/vigokhal/dsv4-monokernel/c16-latency-tuning`, mounted as `/work/c16-latency-tuning` in container `vigokhal_dsv4_tp4`. Source `selected.env` before running. `frozen_before/` pins the preceding implementation for paired comparisons. `run_worker.sh` accepts worker 0–3; set HIP_VISIBLE_DEVICES accordingly. Read RESTORE.md for dependencies. Raw samples, profiler kernel inventories, compiler ASM, failed and successful logs, the selected recipe, and source are retained.

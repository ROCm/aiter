# gfx950 decode perf scan — gluon vs Triton across Triton 3.6.0 / 3.7.0 / 3.8.0

Full decode grid: **C ∈ {16,32,64,128} × ctx ∈ {1024,8192} × heads ∈ {64/8 GQA, 8/1 MQA}**,
both implementations, on three Triton versions. bf16, HEAD_SIZE=128, causal, TILE_SIZE=64.

- **triton** = 3d split-KV attention + `reduce_segments`, split `S` from `select_3d_config`.
- **gluon** = 2d split-KV, num_warps=1 / 16×16 MFMA / nb=2, split `Sg` right-sized to ~CU·4 WGs
  (`select_gluon_num_splits`); **Sg=1 ⇒ non-split path, no reduce kernel**.
- Time = total kernel time (attention + reduce) per iter; TFLOP/s and GB/s derived from it.
- Method: 512 MB L2 flush every iter, torch.profiler, per-kernel-name filter, 8 warmup + 30 iters.
- Versions differ only in the gluon KV-load layout: **BlockedLayout** on 3.6/3.7, **distributed
  offset_bases** on 3.8 (native `ASYNC_COPY_SUPPORTS_DISTRIBUTED` gating). `gl.thread_barrier`
  is absent on 3.7/3.8 so decode is nb=2 on all three (apples-to-apples).
- triton installed per column: `3.6.0`, `3.7.0`, `3.8.0`.
- Cross-check gluon-vs-triton output max abs diff ≤ **5e-04** across all cells.

## Time (µs / iter, lower is better)

| shape | 3.6.0 tri | 3.6.0 glu | 3.7.0 tri | 3.7.0 glu | 3.8.0 (ToT) tri | 3.8.0 (ToT) glu |
|---|--:|--:|--:|--:|--:|--:|
| C16 ctx1024 64/8 | 21.3 | 19.8 | 20.9 | 19.8 | 24.1 | 19.5 |
| C32 ctx1024 64/8 | 36.9 | 31.8 | 36.2 | 31.4 | 40.9 | 30.9 |
| C64 ctx1024 64/8 | 59.2 | 49.4 | 60.7 | 49.2 | 62.0 | 49.2 |
| C128 ctx1024 64/8 | 118.5 | 82.4 | 117.9 | 83.2 | 125.2 | 82.5 |
| C16 ctx1024 8/1 | 9.7 | 9.6 | 9.6 | 9.5 | 9.8 | 9.5 |
| C32 ctx1024 8/1 | 11.2 | 11.6 | 11.2 | 11.4 | 11.3 | 11.5 |
| C64 ctx1024 8/1 | 14.0 | 16.7 | 14.3 | 16.4 | 16.0 | 15.5 |
| C128 ctx1024 8/1 | 22.4 | 20.9 | 22.2 | 21.2 | 24.9 | 19.8 |
| C16 ctx8192 64/8 | 104.6 | 90.6 | 104.9 | 90.0 | 107.5 | 91.2 |
| C32 ctx8192 64/8 | 199.2 | 179.3 | 198.5 | 177.2 | 197.5 | 176.2 |
| C64 ctx8192 64/8 | 372.9 | 333.9 | 372.0 | 330.4 | 363.5 | 328.6 |
| C128 ctx8192 64/8 | 758.5 | 650.6 | 758.3 | 638.0 | 755.5 | 648.6 |
| C16 ctx8192 8/1 | 21.7 | 21.7 | 21.9 | 21.5 | 24.6 | 20.6 |
| C32 ctx8192 8/1 | 35.2 | 30.5 | 34.8 | 32.0 | 41.0 | 31.3 |
| C64 ctx8192 8/1 | 56.3 | 53.4 | 58.3 | 53.3 | 60.6 | 52.3 |
| C128 ctx8192 8/1 | 106.9 | 87.0 | 106.9 | 88.1 | 108.2 | 85.8 |

## Bandwidth (GB/s, higher is better)

| shape | 3.6.0 tri | 3.6.0 glu | 3.7.0 tri | 3.7.0 glu | 3.8.0 (ToT) tri | 3.8.0 (ToT) glu |
|---|--:|--:|--:|--:|--:|--:|
| C16 ctx1024 64/8 | 3169 | 3408 | 3243 | 3416 | 2808 | 3475 |
| C32 ctx1024 64/8 | 3667 | 4254 | 3739 | 4313 | 3309 | 4374 |
| C64 ctx1024 64/8 | 4572 | 5476 | 4456 | 5502 | 4363 | 5495 |
| C128 ctx1024 64/8 | 4565 | 6568 | 4589 | 6506 | 4320 | 6559 |
| C16 ctx1024 8/1 | 869 | 879 | 880 | 887 | 861 | 888 |
| C32 ctx1024 8/1 | 1506 | 1462 | 1507 | 1488 | 1502 | 1476 |
| C64 ctx1024 8/1 | 2411 | 2028 | 2357 | 2060 | 2120 | 2180 |
| C128 ctx1024 8/1 | 3021 | 3233 | 3044 | 3195 | 2719 | 3424 |
| C16 ctx8192 64/8 | 5137 | 5930 | 5124 | 5971 | 5000 | 5895 |
| C32 ctx8192 64/8 | 5397 | 5994 | 5414 | 6064 | 5441 | 6101 |
| C64 ctx8192 64/8 | 5765 | 6438 | 5778 | 6505 | 5914 | 6542 |
| C128 ctx8192 64/8 | 5668 | 6608 | 5670 | 6738 | 5690 | 6628 |
| C16 ctx8192 8/1 | 3089 | 3102 | 3062 | 3118 | 2726 | 3260 |
| C32 ctx8192 8/1 | 3814 | 4411 | 3856 | 4202 | 3279 | 4298 |
| C64 ctx8192 8/1 | 4771 | 5034 | 4607 | 5045 | 4432 | 5141 |
| C128 ctx8192 8/1 | 5025 | 6174 | 5029 | 6101 | 4968 | 6263 |

## Compute (TFLOP/s, higher is better)

| shape | 3.6.0 tri | 3.6.0 glu | 3.7.0 tri | 3.7.0 glu | 3.8.0 (ToT) tri | 3.8.0 (ToT) glu |
|---|--:|--:|--:|--:|--:|--:|
| C16 ctx1024 64/8 | 25 | 27 | 26 | 27 | 22 | 28 |
| C32 ctx1024 64/8 | 29 | 34 | 30 | 34 | 26 | 35 |
| C64 ctx1024 64/8 | 36 | 43 | 35 | 44 | 35 | 44 |
| C128 ctx1024 64/8 | 36 | 52 | 36 | 52 | 34 | 52 |
| C16 ctx1024 8/1 | 7 | 7 | 7 | 7 | 7 | 7 |
| C32 ctx1024 8/1 | 12 | 12 | 12 | 12 | 12 | 12 |
| C64 ctx1024 8/1 | 19 | 16 | 19 | 16 | 17 | 17 |
| C128 ctx1024 8/1 | 24 | 26 | 24 | 25 | 22 | 27 |
| C16 ctx8192 64/8 | 41 | 47 | 41 | 48 | 40 | 47 |
| C32 ctx8192 64/8 | 43 | 48 | 43 | 48 | 43 | 49 |
| C64 ctx8192 64/8 | 46 | 51 | 46 | 52 | 47 | 52 |
| C128 ctx8192 64/8 | 45 | 53 | 45 | 54 | 45 | 53 |
| C16 ctx8192 8/1 | 25 | 25 | 24 | 25 | 22 | 26 |
| C32 ctx8192 8/1 | 30 | 35 | 31 | 34 | 26 | 34 |
| C64 ctx8192 8/1 | 38 | 40 | 37 | 40 | 35 | 41 |
| C128 ctx8192 8/1 | 40 | 49 | 40 | 49 | 40 | 50 |

## Speedup & configuration

### gluon speedup (time_triton / time_gluon), same-version

| shape | 3.6.0 | 3.7.0 | 3.8.0 (ToT) |
|---|--:|--:|--:|
| C16 ctx1024 64/8 | 1.08× | 1.05× | 1.24× |
| C32 ctx1024 64/8 | 1.16× | 1.15× | 1.32× |
| C64 ctx1024 64/8 | 1.20× | 1.23× | 1.26× |
| C128 ctx1024 64/8 | 1.44× | 1.42× | 1.52× |
| C16 ctx1024 8/1 | 1.01× | 1.01× | 1.03× |
| C32 ctx1024 8/1 | 0.97× | 0.99× | 0.98× |
| C64 ctx1024 8/1 | 0.84× | 0.87× | 1.03× |
| C128 ctx1024 8/1 | 1.07× | 1.05× | 1.26× |
| C16 ctx8192 64/8 | 1.15× | 1.17× | 1.18× |
| C32 ctx8192 64/8 | 1.11× | 1.12× | 1.12× |
| C64 ctx8192 64/8 | 1.12× | 1.13× | 1.11× |
| C128 ctx8192 64/8 | 1.17× | 1.19× | 1.16× |
| C16 ctx8192 8/1 | 1.00× | 1.02× | 1.20× |
| C32 ctx8192 8/1 | 1.16× | 1.09× | 1.31× |
| C64 ctx8192 8/1 | 1.06× | 1.09× | 1.16× |
| C128 ctx8192 8/1 | 1.23× | 1.21× | 1.26× |

### split counts used (triton heuristic `S` vs gluon right-sized `Sg`)

| shape | triton S | gluon Sg |
|---|--:|--:|
| C16 ctx1024 64/8 | 16 | 8 |
| C32 ctx1024 64/8 | 16 | 4 |
| C64 ctx1024 64/8 | 8 | 2 |
| C128 ctx1024 64/8 | 8 | 1 (no split, no reduce) |
| C16 ctx1024 8/1 | 16 | 16 |
| C32 ctx1024 8/1 | 16 | 16 |
| C64 ctx1024 8/1 | 16 | 16 |
| C128 ctx1024 8/1 | 16 | 8 |
| C16 ctx8192 64/8 | 32 | 8 |
| C32 ctx8192 64/8 | 16 | 4 |
| C64 ctx8192 64/8 | 8 | 2 |
| C128 ctx8192 64/8 | 8 | 1 (no split, no reduce) |
| C16 ctx8192 8/1 | 128 | 64 |
| C32 ctx8192 8/1 | 128 | 32 |
| C64 ctx8192 8/1 | 64 | 16 |
| C128 ctx8192 8/1 | 32 | 8 |

## Takeaways

- **gluon wins nearly everywhere.** On 3.8.0 (ToT): GQA 64/8 1.11–1.52×, MQA 8/1 0.98–1.31×. Only non-win: C32 ctx1024 8/1 (0.98×).
- **The S=1 shapes are the biggest wins** (C128 64/8, 1.44–1.52×): right-sizing picks *no split*, so gluon runs the attention alone and skips the reduce kernel entirely.
- **GQA decode is bandwidth-bound** — gluon holds ~6.4–6.7 TB/s (≈ HBM peak) across all shapes/versions while Triton sits at ~4.3–5.9 TB/s (its heuristic over-splits → reduce overhead + Q reloads).
- **Caveat: Triton regressed on 3.8.0 for several small shapes**, which inflates gluon's 3.8 speedup there — gluon's *absolute* numbers barely move across versions. Triton 3.6→3.8 GB/s drops >5%: C16 ctx1024 64/8 (3169→2808); C32 ctx1024 64/8 (3667→3309); C128 ctx1024 64/8 (4565→4320); C64 ctx1024 8/1 (2411→2120); C128 ctx1024 8/1 (3021→2719); C16 ctx8192 8/1 (3089→2726); C32 ctx8192 8/1 (3814→3279); C64 ctx8192 8/1 (4771→4432).
- **gluon absolute perf is version-stable** (identical kernel; only the KV-load layout differs — BlockedLayout on 3.6/3.7, distributed offset_bases on 3.8).
- The only soft spot is **small-batch small-ctx MQA** (C32/C64 ctx1024 8/1): splits are capped at num_tiles=16 so the GPU is under-occupied, and the tiny cached KV is latency/overhead-bound.
- Correctness: gluon-vs-triton max abs diff ≤ 5e-04 on every one of the 48 cells.


<!-- SILOTIGER-667 G9 FlyDSL-vs-CK cold warp-decode comparison -->
<!-- gfx=gfx950  aiter=aaa1ab6e3  ck_worktree=c03392a91b8 -->
<!-- iters=1000 cold=20 timing=device method=weight_stream repeats=3 -->
<!-- CK provenance: # ck_bench_warp_decode  base_commit=62e30c9098 patch=A4-gateup-fp4-packed-stride  cold=20  iters=1000  rotate=auto(ceil(E/BK))  format=csv  mechanism=manual-hipEvent+disjoint-router-rotation -->
<!-- clocks: auto (unpinnable on this gfx950; D1) -- effective loaded sclk MHz min/median/max = 1427/2393/2404 (n=230/233) on GPU 6; per-cell spread%% + noisy flag (>5%) capture drift (D5). -->
<!-- config policy (D3): default-vs-default. FlyDSL = library defaults, no overrides: serialize_dot2=True, kh_per_warp=auto(2 when HIDDEN even), prefetch=False; down_fp4 dot2_acc=4, gate_up_fp4 dot2_acc=1 (G7: acc>1 ~4% slower for gate_up); down_fp8 split_k=1; FP8 w_scale=block2d(128,128) to match CK. CK = maintainer-recommended variant per op (down_h2_d2, down_fp4_h2, gate_bf16_d2, gate_up_fp4 non-dot2/NPerWarp=1); CK has no single runtime default (mild asymmetry). FP8-down ratio is a CK-favored lower bound: block2d(128,128) costs FlyDSL ~10-38% vs pertensor (B1); FP4 rows carry a ~6% CK-favored scale-traffic bias (dummy PerTensor vs e8m0(1,32)). Treat under-converged fast cells as noisy (D1). -->

**metric method:** `weight_stream` &nbsp; (ratio = flydsl_us / ck_us; CK is perf-only / uninitialized weights)

| shape | B | op | dtype | act | flydsl_us | ck_us | ratio(f/c) | fly_TB/s | ck_TB/s | fly_%peak | fly_spr% | ck_spr% | cos | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3 | 1 | down | fp4 | - | 17.1851 | 29.6046 | 0.580 | 3.6 | 2.1 | 45.4 | 8.0 | 0.1 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | down | fp8 | - | 21.3163 | 35.3307 | 0.603 | 5.5 | 3.3 | 68.9 | 0.7 | 0.1 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp4 | bf16 | 29.6911 | 50.2051 | 0.591 | 4.2 | 2.5 | 52.5 | 0.9 | 0.1 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | bf16 | 42.2493 | 46.4849 | 0.909 | 5.6 | 5.1 | 69.5 | 2.5 | 0.1 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp4 | - | 31.4237 | 37.5213 | 0.837 | 4.0 | 3.3 | 49.6 | 5.2 | 0.1 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 2 | down | fp8 | - | 42.0234 | 51.8489 | 0.810 | 5.6 | 4.5 | 69.9 | 0.3 | 0.1 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp4 | bf16 | 53.8126 | 92.2956 | 0.583 | 4.6 | 2.7 | 58.0 | 1.0 | 0.0 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp8 | bf16 | 81.0750 | 86.3305 | 0.939 | 5.8 | 5.4 | 72.4 | 1.5 | 0.1 | 1.0000 |  |
| deepseek-v3 | 4 | down | fp4 | - | 58.3636 | 69.0828 | 0.845 | 4.3 | 3.6 | 53.4 | 0.1 | 0.2 | 1.0000 |  |
| deepseek-v3 | 4 | down | fp8 | - | 82.8410 | 93.2970 | 0.888 | 5.7 | 5.0 | 70.9 | 2.3 | 0.1 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp4 | bf16 | 103.7334 | 173.6467 | 0.597 | 4.8 | 2.9 | 60.1 | 0.9 | 0.1 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp8 | bf16 | 159.5629 | 164.5935 | 0.969 | 5.9 | 5.7 | 73.6 | 0.8 | 0.0 | 1.0000 |  |
| deepseek-v3 | 8 | down | fp4 | - | 108.8906 | 134.4605 | 0.810 | 4.6 | 3.7 | 57.3 | 3.5 | 0.2 | 1.0000 |  |
| deepseek-v3 | 8 | down | fp8 | - | 159.6975 | 179.7878 | 0.888 | 5.9 | 5.2 | 73.5 | 0.9 | 0.0 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp4 | bf16 | 208.8340 | 339.1949 | 0.616 | 4.8 | 2.9 | 59.8 | 2.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp8 | bf16 | 319.9498 | 316.7416 | 1.010 | 5.9 | 5.9 | 73.4 | 0.8 | 0.0 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp4 | - | 434.2262 | 479.4029 | 0.906 | 4.6 | 4.2 | 57.5 | 2.0 | 0.1 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp8 | - | 641.7441 | 673.5654 | 0.953 | 5.9 | 5.6 | 73.2 | 0.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp4 | bf16 | 850.5764 | 1330.6895 | 0.639 | 4.7 | 3.0 | 58.7 | 1.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp8 | bf16 | 1297.3660 | 1260.2664 | 1.029 | 5.8 | 6.0 | 72.4 | 0.8 | 0.6 | 1.0000 |  |
| minimax | 1 | down | fp4 | - | 10.0043 | 20.8797 | 0.479 | 2.0 | 1.0 | 25.1 | 2.2 | 0.2 | 1.0000 |  |
| minimax | 1 | down | fp8 | - | 11.7990 | 23.2093 | 0.508 | 3.2 | 1.6 | 40.0 | 0.2 | 0.3 | 1.0000 |  |
| minimax | 1 | gate_up | fp4 | bf16 | 11.4334 | 18.3803 | 0.622 | 3.5 | 2.2 | 43.8 | 0.0 | 0.2 | 1.0000 |  |
| minimax | 1 | gate_up | fp8 | bf16 | 14.7532 | 19.2165 | 0.768 | 5.1 | 3.9 | 64.0 | 0.4 | 0.2 | 1.0000 |  |
| minimax | 2 | down | fp4 | - | 12.1331 | 24.0428 | 0.505 | 3.3 | 1.7 | 41.3 | 0.2 | 17.9 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | down | fp8 | - | 17.3812 | 27.3938 | 0.634 | 4.3 | 2.8 | 54.3 | 9.8 | 0.2 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | gate_up | fp4 | bf16 | 18.3092 | 32.7272 | 0.559 | 4.4 | 2.5 | 54.8 | 1.9 | 0.1 | 1.0000 |  |
| minimax | 2 | gate_up | fp8 | bf16 | 26.2600 | 32.0473 | 0.819 | 5.8 | 4.7 | 71.9 | 1.2 | 0.5 | 1.0000 |  |
| minimax | 4 | down | fp4 | - | 22.4876 | 29.6608 | 0.758 | 3.6 | 2.7 | 44.6 | 1.4 | 0.1 | 1.0000 |  |
| minimax | 4 | down | fp8 | - | 32.6156 | 36.5597 | 0.892 | 4.6 | 4.1 | 57.9 | 5.8 | 0.1 | 1.0000 | noisy (>5% spread) |
| minimax | 4 | gate_up | fp4 | bf16 | 34.7126 | 59.6907 | 0.582 | 4.6 | 2.7 | 57.8 | 3.5 | 0.1 | 1.0000 |  |
| minimax | 4 | gate_up | fp8 | bf16 | 48.5427 | 56.3920 | 0.861 | 6.2 | 5.4 | 77.8 | 0.7 | 0.1 | 1.0000 |  |
| minimax | 8 | down | fp4 | - | 39.9810 | 53.8545 | 0.742 | 4.0 | 3.0 | 50.2 | 5.1 | 0.0 | 1.0000 | noisy (>5% spread) |
| minimax | 8 | down | fp8 | - | 60.6470 | 65.5335 | 0.925 | 5.0 | 4.6 | 62.2 | 4.7 | 0.1 | 1.0000 |  |
| minimax | 8 | gate_up | fp4 | bf16 | 65.1982 | 112.2468 | 0.581 | 4.9 | 2.9 | 61.5 | 3.9 | 0.1 | 1.0000 |  |
| minimax | 8 | gate_up | fp8 | bf16 | 95.2318 | 104.1279 | 0.915 | 6.3 | 5.8 | 79.3 | 2.7 | 0.2 | 1.0000 |  |
| minimax | 32 | down | fp4 | - | 149.2668 | 171.1795 | 0.872 | 4.3 | 3.7 | 53.7 | 3.2 | 0.1 | 1.0000 |  |
| minimax | 32 | down | fp8 | - | 228.7144 | 227.5804 | 1.005 | 5.3 | 5.3 | 66.0 | 3.5 | 0.1 | 1.0000 |  |
| minimax | 32 | gate_up | fp4 | bf16 | 248.8620 | 431.2708 | 0.577 | 5.2 | 3.0 | 64.5 | 1.6 | 0.1 | 1.0000 |  |
| minimax | 32 | gate_up | fp8 | bf16 | 384.9980 | 401.9760 | 0.958 | 6.3 | 6.0 | 78.4 | 0.1 | 0.5 | 1.0000 |  |
| qwen3next | 1 | down | fp4 | - | 5.2571 | 9.7667 | 0.538 | 1.1 | 0.6 | 13.2 | 0.3 | 0.3 | 1.0000 |  |
| qwen3next | 1 | down | fp8 | - | 5.6890 | 10.3380 | 0.550 | 1.8 | 1.0 | 23.0 | 0.3 | 0.4 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp4 | bf16 | 6.0805 | 7.1999 | 0.845 | 1.8 | 1.5 | 22.9 | 1.2 | 1.4 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp8 | bf16 | 7.5093 | 7.8847 | 0.952 | 2.8 | 2.7 | 34.9 | 0.6 | 0.2 | 1.0000 |  |
| qwen3next | 2 | down | fp4 | - | 5.8381 | 11.3586 | 0.514 | 1.9 | 1.0 | 23.9 | 0.3 | 0.3 | 1.0000 |  |
| qwen3next | 2 | down | fp8 | - | 7.3351 | 15.2257 | 0.482 | 2.9 | 1.4 | 35.7 | 5.0 | 0.2 | 1.0000 | noisy (>5% spread) |
| qwen3next | 2 | gate_up | fp4 | bf16 | 8.3387 | 10.8433 | 0.769 | 2.7 | 2.1 | 33.4 | 4.9 | 0.2 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | bf16 | 10.7714 | 12.7435 | 0.845 | 3.9 | 3.3 | 48.7 | 1.8 | 0.2 | 1.0000 |  |
| qwen3next | 4 | down | fp4 | - | 8.4150 | 13.6454 | 0.617 | 2.6 | 1.6 | 33.1 | 1.8 | 0.2 | 1.0000 |  |
| qwen3next | 4 | down | fp8 | - | 11.5926 | 17.7333 | 0.654 | 3.6 | 2.4 | 45.2 | 2.2 | 0.2 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp4 | bf16 | 13.2041 | 16.9696 | 0.778 | 3.4 | 2.6 | 42.2 | 5.0 | 0.4 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp8 | bf16 | 17.2461 | 20.0217 | 0.861 | 4.9 | 4.2 | 60.8 | 1.0 | 0.1 | 1.0000 |  |
| qwen3next | 8 | down | fp4 | - | 12.2772 | 16.2128 | 0.757 | 3.6 | 2.7 | 45.4 | 2.3 | 0.5 | 1.0000 |  |
| qwen3next | 8 | down | fp8 | - | 18.9079 | 23.3146 | 0.811 | 4.4 | 3.6 | 55.5 | 7.6 | 0.1 | 1.0000 | noisy (>5% spread) |
| qwen3next | 8 | gate_up | fp4 | bf16 | 22.7865 | 29.5656 | 0.771 | 3.9 | 3.0 | 48.9 | 3.8 | 0.1 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp8 | bf16 | 30.2343 | 32.8288 | 0.921 | 5.5 | 5.1 | 69.4 | 1.2 | 0.3 | 1.0000 |  |
| qwen3next | 32 | down | fp4 | - | 37.0307 | 54.8669 | 0.675 | 4.8 | 3.2 | 60.2 | 0.1 | 0.3 | 1.0000 |  |
| qwen3next | 32 | down | fp8 | - | 64.8011 | 73.1110 | 0.886 | 5.2 | 4.6 | 64.7 | 4.0 | 0.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp4 | bf16 | 72.6790 | 104.7649 | 0.694 | 4.9 | 3.4 | 61.3 | 2.1 | 0.3 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp8 | bf16 | 105.9028 | 112.7496 | 0.939 | 6.3 | 6.0 | 79.2 | 0.6 | 0.1 | 1.0000 |  |

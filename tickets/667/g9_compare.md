<!-- SILOTIGER-667 G9 FlyDSL-vs-CK cold warp-decode comparison -->
<!-- gfx=gfx950  aiter=f9524f50e  ck_worktree=c03392a91b8 -->
<!-- iters=1000 cold=20 timing=device method=weight_stream repeats=3 -->
<!-- CK provenance: # ck_bench_warp_decode  base_commit=62e30c9098 patch=A4-gateup-fp4-packed-stride  cold=20  iters=1000  rotate=auto(ceil(E/BK))  format=csv  mechanism=manual-hipEvent+disjoint-router-rotation -->
<!-- clocks: auto (unpinnable on this gfx950; D1) -- effective loaded sclk MHz min/median/max = 1385/2392/2407 (n=277/280) on GPU 6; per-cell spread%% + noisy flag (>5%) capture drift (D5). -->
<!-- config policy (D3): default-vs-default. FlyDSL = library defaults, no overrides: serialize_dot2=True, kh_per_warp=auto(2 when HIDDEN even), prefetch=False; down_fp4 dot2_acc=4, gate_up_fp4 dot2_acc=1 (G7: acc>1 ~4% slower for gate_up); down_fp8 split_k=1; FP8 w_scale=block2d(128,128) to match CK. CK = maintainer-recommended variant per op (down_h2_d2, down_fp4_h2, gate_bf16_d2, gate_up_fp4 non-dot2/NPerWarp=1); CK has no single runtime default (mild asymmetry). FP8-down ratio is a CK-favored lower bound: block2d(128,128) costs FlyDSL ~10-38% vs pertensor (B1); FP4 rows carry a ~6% CK-favored scale-traffic bias (dummy PerTensor vs e8m0(1,32)). Treat under-converged fast cells as noisy (D1). -->

**metric method:** `weight_stream` &nbsp; (ratio = flydsl_us / ck_us; CK is perf-only / uninitialized weights)

| shape | B | op | dtype | act | flydsl_us | ck_us | ratio(f/c) | fly_TB/s | ck_TB/s | fly_%peak | fly_spr% | ck_spr% | cos | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3 | 1 | down | fp4 | - | 18.0360 | 29.6081 | 0.609 | 3.5 | 2.1 | 43.2 | 6.4 | 0.4 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | down | fp8 | - | 21.3414 | 35.3600 | 0.604 | 5.5 | 3.3 | 68.8 | 5.8 | 0.2 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | gate_up | fp4 | bf16 | 28.6880 | 50.1763 | 0.572 | 4.3 | 2.5 | 54.4 | 4.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | bf16 | 42.0316 | 46.4457 | 0.905 | 5.6 | 5.1 | 69.9 | 2.0 | 0.1 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | fp8 | 40.3266 | 46.5792 | 0.866 | 5.8 | 5.0 | 72.8 | 0.1 | 0.2 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp4 | - | 33.0580 | 38.0847 | 0.868 | 3.8 | 3.3 | 47.2 | 4.9 | 1.5 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp8 | - | 42.1353 | 51.7724 | 0.814 | 5.6 | 4.5 | 69.7 | 1.9 | 0.3 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp4 | bf16 | 54.8238 | 92.2947 | 0.594 | 4.6 | 2.7 | 56.9 | 3.8 | 0.0 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp8 | bf16 | 80.6782 | 86.3320 | 0.935 | 5.8 | 5.4 | 72.8 | 2.3 | 0.1 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp8 | fp8 | 76.9043 | 87.5684 | 0.878 | 6.1 | 5.4 | 76.4 | 1.0 | 0.2 | 1.0000 |  |
| deepseek-v3 | 4 | down | fp4 | - | 58.3354 | 69.0524 | 0.845 | 4.3 | 3.6 | 53.5 | 6.1 | 0.1 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 4 | down | fp8 | - | 82.0239 | 93.2583 | 0.880 | 5.7 | 5.0 | 71.6 | 1.9 | 0.2 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp4 | bf16 | 109.4223 | 173.9199 | 0.629 | 4.6 | 2.9 | 57.0 | 4.2 | 0.2 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp8 | bf16 | 161.0585 | 164.9965 | 0.976 | 5.8 | 5.7 | 72.9 | 2.5 | 0.3 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp8 | fp8 | 150.2301 | 163.0545 | 0.921 | 6.3 | 5.8 | 78.2 | 0.5 | 0.4 | 1.0000 |  |
| deepseek-v3 | 8 | down | fp4 | - | 110.2144 | 134.4499 | 0.820 | 4.5 | 3.7 | 56.6 | 5.5 | 0.1 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 8 | down | fp8 | - | 158.5850 | 179.7359 | 0.882 | 5.9 | 5.2 | 74.1 | 2.5 | 0.1 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp4 | bf16 | 209.6745 | 338.9502 | 0.619 | 4.8 | 2.9 | 59.5 | 2.8 | 0.1 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp8 | bf16 | 314.8819 | 316.6423 | 0.994 | 6.0 | 5.9 | 74.6 | 1.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp8 | fp8 | 297.0171 | 314.0712 | 0.946 | 6.3 | 6.0 | 79.1 | 0.5 | 0.2 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp4 | - | 426.6925 | 479.5294 | 0.890 | 4.7 | 4.2 | 58.5 | 2.9 | 0.2 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp8 | - | 641.5298 | 673.1402 | 0.953 | 5.9 | 5.6 | 73.2 | 0.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp4 | bf16 | 858.2542 | 1329.8825 | 0.645 | 4.7 | 3.0 | 58.2 | 3.2 | 0.0 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp8 | bf16 | 1289.2404 | 1261.0589 | 1.022 | 5.8 | 6.0 | 72.9 | 0.3 | 0.2 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp8 | fp8 | 1189.8235 | 1247.1691 | 0.954 | 6.3 | 6.0 | 79.0 | 0.7 | 0.1 | 1.0000 |  |
| minimax | 1 | down | fp4 | - | 10.4061 | 20.8723 | 0.499 | 1.9 | 1.0 | 24.1 | 7.1 | 0.2 | 1.0000 | noisy (>5% spread) |
| minimax | 1 | down | fp8 | - | 11.7398 | 23.2285 | 0.505 | 3.2 | 1.6 | 40.2 | 10.3 | 0.2 | 1.0000 | noisy (>5% spread) |
| minimax | 1 | gate_up | fp4 | bf16 | 11.4636 | 18.4050 | 0.623 | 3.5 | 2.2 | 43.7 | 4.6 | 0.1 | 1.0000 |  |
| minimax | 1 | gate_up | fp8 | bf16 | 15.1928 | 19.2182 | 0.791 | 5.0 | 3.9 | 62.1 | 2.8 | 0.1 | 1.0000 |  |
| minimax | 1 | gate_up | fp8 | fp8 | 15.1193 | 20.2879 | 0.745 | 5.0 | 3.7 | 62.4 | 1.7 | 0.1 | 1.0000 |  |
| minimax | 2 | down | fp4 | - | 12.5951 | 24.0261 | 0.524 | 3.2 | 1.7 | 39.8 | 9.2 | 14.5 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | down | fp8 | - | 16.7262 | 31.6930 | 0.528 | 4.5 | 2.4 | 56.4 | 17.6 | 24.9 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | gate_up | fp4 | bf16 | 18.5901 | 32.5697 | 0.571 | 4.3 | 2.5 | 53.9 | 5.5 | 0.7 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | gate_up | fp8 | bf16 | 25.9635 | 32.0763 | 0.809 | 5.8 | 4.7 | 72.7 | 0.6 | 0.5 | 1.0000 |  |
| minimax | 2 | gate_up | fp8 | fp8 | 26.0871 | 33.2554 | 0.784 | 5.8 | 4.5 | 72.4 | 1.4 | 0.6 | 1.0000 |  |
| minimax | 4 | down | fp4 | - | 22.2927 | 29.6483 | 0.752 | 3.6 | 2.7 | 45.0 | 7.8 | 0.1 | 1.0000 | noisy (>5% spread) |
| minimax | 4 | down | fp8 | - | 32.5855 | 36.5248 | 0.892 | 4.6 | 4.1 | 57.9 | 5.9 | 0.8 | 1.0000 | noisy (>5% spread) |
| minimax | 4 | gate_up | fp4 | bf16 | 34.6453 | 59.7322 | 0.580 | 4.6 | 2.7 | 57.9 | 1.0 | 0.2 | 1.0000 |  |
| minimax | 4 | gate_up | fp8 | bf16 | 48.7861 | 56.3797 | 0.865 | 6.2 | 5.4 | 77.4 | 2.3 | 0.3 | 1.0000 |  |
| minimax | 4 | gate_up | fp8 | fp8 | 49.7813 | 58.6881 | 0.848 | 6.1 | 5.1 | 75.8 | 1.9 | 0.3 | 1.0000 |  |
| minimax | 8 | down | fp4 | - | 39.5722 | 53.8662 | 0.735 | 4.1 | 3.0 | 50.7 | 5.0 | 0.1 | 1.0000 | noisy (>5% spread) |
| minimax | 8 | down | fp8 | - | 57.9277 | 65.5180 | 0.884 | 5.2 | 4.6 | 65.2 | 4.8 | 0.1 | 1.0000 |  |
| minimax | 8 | gate_up | fp4 | bf16 | 63.6531 | 112.3417 | 0.567 | 5.0 | 2.9 | 63.0 | 3.8 | 0.2 | 1.0000 |  |
| minimax | 8 | gate_up | fp8 | bf16 | 97.1255 | 104.2087 | 0.932 | 6.2 | 5.8 | 77.7 | 2.9 | 0.1 | 1.0000 |  |
| minimax | 8 | gate_up | fp8 | fp8 | 97.5687 | 108.8491 | 0.896 | 6.2 | 5.5 | 77.4 | 1.6 | 0.2 | 1.0000 |  |
| minimax | 32 | down | fp4 | - | 146.0072 | 170.9048 | 0.854 | 4.4 | 3.8 | 54.9 | 2.5 | 0.1 | 1.0000 |  |
| minimax | 32 | down | fp8 | - | 228.3026 | 227.6341 | 1.003 | 5.3 | 5.3 | 66.1 | 1.9 | 0.1 | 1.0000 |  |
| minimax | 32 | gate_up | fp4 | bf16 | 251.7162 | 431.7791 | 0.583 | 5.1 | 3.0 | 63.7 | 0.8 | 0.2 | 1.0000 |  |
| minimax | 32 | gate_up | fp8 | bf16 | 387.7989 | 400.7219 | 0.968 | 6.2 | 6.0 | 77.9 | 1.7 | 0.2 | 1.0000 |  |
| minimax | 32 | gate_up | fp8 | fp8 | 383.4931 | 417.6989 | 0.918 | 6.3 | 5.8 | 78.7 | 0.3 | 0.4 | 1.0000 |  |
| qwen3next | 1 | down | fp4 | - | 5.7889 | 9.7557 | 0.593 | 1.0 | 0.6 | 12.0 | 9.6 | 0.1 | 1.0000 | noisy (>5% spread) |
| qwen3next | 1 | down | fp8 | - | 6.2422 | 10.3693 | 0.602 | 1.7 | 1.0 | 21.0 | 9.1 | 0.3 | 1.0000 | noisy (>5% spread) |
| qwen3next | 1 | gate_up | fp4 | bf16 | 6.4575 | 7.2069 | 0.896 | 1.7 | 1.5 | 21.6 | 8.9 | 0.5 | 1.0000 | noisy (>5% spread) |
| qwen3next | 1 | gate_up | fp8 | bf16 | 7.6475 | 7.8902 | 0.969 | 2.7 | 2.7 | 34.3 | 2.7 | 0.1 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp8 | fp8 | 7.5861 | 8.5989 | 0.882 | 2.8 | 2.4 | 34.6 | 6.6 | 0.4 | 1.0000 | noisy (>5% spread) |
| qwen3next | 2 | down | fp4 | - | 6.3919 | 11.3648 | 0.562 | 1.7 | 1.0 | 21.8 | 0.3 | 0.1 | 1.0000 |  |
| qwen3next | 2 | down | fp8 | - | 7.6772 | 15.2326 | 0.504 | 2.7 | 1.4 | 34.1 | 5.3 | 0.4 | 1.0000 | noisy (>5% spread) |
| qwen3next | 2 | gate_up | fp4 | bf16 | 8.6752 | 10.8627 | 0.799 | 2.6 | 2.1 | 32.1 | 0.3 | 0.5 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | bf16 | 10.9074 | 12.7155 | 0.858 | 3.8 | 3.3 | 48.1 | 1.2 | 0.1 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | fp8 | 10.6484 | 13.2660 | 0.803 | 3.9 | 3.2 | 49.2 | 1.0 | 0.2 | 1.0000 |  |
| qwen3next | 4 | down | fp4 | - | 8.8162 | 13.6372 | 0.646 | 2.5 | 1.6 | 31.6 | 1.0 | 0.2 | 1.0000 |  |
| qwen3next | 4 | down | fp8 | - | 11.2384 | 17.7065 | 0.635 | 3.7 | 2.4 | 46.7 | 0.2 | 0.5 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp4 | bf16 | 12.9029 | 17.0149 | 0.758 | 3.5 | 2.6 | 43.2 | 2.1 | 0.5 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp8 | bf16 | 17.2533 | 20.0134 | 0.862 | 4.9 | 4.2 | 60.8 | 0.6 | 0.1 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp8 | fp8 | 16.9024 | 21.4454 | 0.788 | 5.0 | 3.9 | 62.0 | 0.1 | 0.2 | 1.0000 |  |
| qwen3next | 8 | down | fp4 | - | 12.7557 | 16.1901 | 0.788 | 3.5 | 2.8 | 43.7 | 2.2 | 0.1 | 1.0000 |  |
| qwen3next | 8 | down | fp8 | - | 19.5162 | 23.3389 | 0.836 | 4.3 | 3.6 | 53.7 | 7.3 | 0.2 | 1.0000 | noisy (>5% spread) |
| qwen3next | 8 | gate_up | fp4 | bf16 | 22.8401 | 29.5673 | 0.772 | 3.9 | 3.0 | 48.8 | 5.5 | 0.2 | 1.0000 | noisy (>5% spread) |
| qwen3next | 8 | gate_up | fp8 | bf16 | 30.0273 | 32.7518 | 0.917 | 5.6 | 5.1 | 69.8 | 1.7 | 0.2 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp8 | fp8 | 29.1151 | 35.0338 | 0.831 | 5.8 | 4.8 | 72.0 | 1.4 | 0.2 | 1.0000 |  |
| qwen3next | 32 | down | fp4 | - | 37.0877 | 55.1159 | 0.673 | 4.8 | 3.2 | 60.1 | 0.2 | 0.5 | 1.0000 |  |
| qwen3next | 32 | down | fp8 | - | 62.0862 | 73.1286 | 0.849 | 5.4 | 4.6 | 67.6 | 0.2 | 0.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp4 | bf16 | 73.5359 | 104.8616 | 0.701 | 4.8 | 3.4 | 60.6 | 1.8 | 0.2 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp8 | bf16 | 105.7111 | 112.4396 | 0.940 | 6.3 | 6.0 | 79.4 | 1.0 | 0.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp8 | fp8 | 105.1152 | 123.9296 | 0.848 | 6.4 | 5.4 | 79.8 | 0.3 | 0.0 | 1.0000 |  |

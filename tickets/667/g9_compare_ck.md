<!-- SILOTIGER-667 G9 FlyDSL-vs-ck cold warp-decode comparison -->
<!-- gfx=gfx950  aiter=3a09d9fc8  backend=ck  ck_worktree=c03392a91b8 -->
<!-- iters=100 cold=20 timing=device method=weight_stream repeats=3 -->
<!-- ck provenance: # ck_bench_warp_decode  base_commit=62e30c9098 patch=A4-gateup-fp4-packed-stride  cold=20  iters=100  rotate=auto(ceil(E/BK))  format=csv  mechanism=manual-hipEvent+disjoint-router-rotation -->
<!-- clocks: auto (unpinnable on this gfx950; D1) -- effective loaded sclk MHz min/median/max = 2383/2398/2402 (n=144/183) on GPU 1; per-cell spread%% + noisy flag (>5%) capture drift (D5). -->
<!-- config policy (D3): default-vs-default. FlyDSL = library defaults, no overrides: serialize_dot2=True, kh_per_warp=auto(2 when HIDDEN even), prefetch=False; down_fp4 dot2_acc=4, gate_up_fp4 dot2_acc=1 (G7: acc>1 ~4% slower for gate_up); down_fp8 split_k=1; FP8 w_scale=block2d(128,128) to match CK. CK = maintainer-recommended variant per op (down_h2_d2, down_fp4_h2, gate_bf16_d2, gate_up_fp4 non-dot2/NPerWarp=1); CK has no single runtime default (mild asymmetry). FP8-down ratio is a CK-favored lower bound: block2d(128,128) costs FlyDSL ~10-38% vs pertensor (B1); FP4 rows carry a ~6% CK-favored scale-traffic bias (dummy PerTensor vs e8m0(1,32)). Treat under-converged fast cells as noisy (D1). -->

**metric method:** `weight_stream` &nbsp; (ratio = flydsl_us / ck_us; CK is perf-only / uninitialized weights)

| shape | B | op | dtype | act | flydsl_us | ck_us | ratio(f/c) | fly_TB/s | ck_TB/s | fly_%peak | fly_spr% | ck_spr% | cos | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3 | 1 | down | fp4 | - | 22.8804 | 28.3130 | 0.808 | 2.7 | 2.2 | 34.1 | 2.8 | 0.3 | 1.0000 |  |
| deepseek-v3 | 1 | down | fp8 | - | 22.6487 | 42.4223 | 0.534 | 5.2 | 2.8 | 64.8 | 4.9 | 0.5 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp4 | bf16 | 28.8063 | 49.7952 | 0.578 | 4.3 | 2.5 | 54.1 | 1.5 | 1.5 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | bf16 | 41.9839 | 48.9128 | 0.858 | 5.6 | 4.8 | 69.9 | 0.7 | 4.4 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | fp8 | 40.5603 | 48.6136 | 0.834 | 5.8 | 4.8 | 72.4 | 1.2 | 4.9 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp4 | - | 40.7148 | 33.6243 | 1.211 | 3.1 | 3.7 | 38.3 | 0.6 | 8.1 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 2 | down | fp8 | - | 42.6506 | 61.7693 | 0.690 | 5.5 | 3.8 | 68.8 | 0.3 | 1.8 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp4 | bf16 | 54.3257 | 91.1847 | 0.596 | 4.6 | 2.7 | 57.4 | 0.3 | 1.4 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp8 | bf16 | 80.9250 | 88.1099 | 0.918 | 5.8 | 5.3 | 72.6 | 0.6 | 2.6 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp8 | fp8 | 77.3866 | 89.0515 | 0.869 | 6.1 | 5.3 | 75.9 | 1.5 | 2.7 | 1.0000 |  |
| deepseek-v3 | 4 | down | fp4 | - | 68.3697 | 66.4513 | 1.029 | 3.7 | 3.8 | 45.6 | 0.7 | 0.8 | 1.0000 |  |
| deepseek-v3 | 4 | down | fp8 | - | 82.7327 | 113.1742 | 0.731 | 5.7 | 4.2 | 71.0 | 0.3 | 4.1 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp4 | bf16 | 107.2129 | 174.9341 | 0.613 | 4.7 | 2.9 | 58.2 | 3.8 | 1.3 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp8 | bf16 | 159.0114 | 163.3588 | 0.973 | 5.9 | 5.8 | 73.9 | 0.2 | 1.6 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp8 | fp8 | 151.2424 | 163.5680 | 0.925 | 6.2 | 5.7 | 77.7 | 0.7 | 2.4 | 1.0000 |  |
| deepseek-v3 | 8 | down | fp4 | - | 130.3262 | 118.9926 | 1.095 | 3.8 | 4.2 | 47.9 | 0.7 | 2.8 | 1.0000 |  |
| deepseek-v3 | 8 | down | fp8 | - | 162.1019 | 217.5911 | 0.745 | 5.8 | 4.3 | 72.4 | 3.4 | 2.3 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp4 | bf16 | 209.0731 | 341.5277 | 0.612 | 4.8 | 2.9 | 59.7 | 1.3 | 1.2 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp8 | bf16 | 314.2942 | 319.2012 | 0.985 | 6.0 | 5.9 | 74.7 | 2.7 | 3.3 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp8 | fp8 | 298.9526 | 317.2220 | 0.942 | 6.3 | 5.9 | 78.6 | 1.5 | 2.1 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp4 | - | 477.2293 | 465.1773 | 1.026 | 4.2 | 4.3 | 52.3 | 0.7 | 0.2 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp8 | - | 630.9248 | 806.2938 | 0.782 | 6.0 | 4.7 | 74.5 | 0.4 | 3.9 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp4 | bf16 | 818.9281 | 1337.6523 | 0.612 | 4.9 | 3.0 | 60.9 | 0.3 | 1.2 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp8 | bf16 | 1257.6871 | 1264.1766 | 0.995 | 6.0 | 5.9 | 74.7 | 2.1 | 1.7 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp8 | fp8 | 1193.3665 | 1261.0922 | 0.946 | 6.3 | 6.0 | 78.7 | 0.9 | 2.1 | 1.0000 |  |
| minimax | 1 | down | fp4 | - | 9.7320 | 20.2522 | 0.481 | 2.1 | 1.0 | 25.8 | 1.6 | 0.2 | 1.0000 |  |
| minimax | 1 | down | fp8 | - | 14.6498 | 24.3174 | 0.602 | 2.6 | 1.6 | 32.2 | 0.2 | 0.4 | 1.0000 |  |
| minimax | 1 | gate_up | fp4 | bf16 | 12.5806 | 18.5258 | 0.679 | 3.2 | 2.2 | 39.9 | 4.1 | 0.4 | 1.0000 |  |
| minimax | 1 | gate_up | fp8 | bf16 | 15.2319 | 18.9634 | 0.803 | 5.0 | 4.0 | 62.0 | 0.6 | 1.8 | 1.0000 |  |
| minimax | 1 | gate_up | fp8 | fp8 | 14.8659 | 19.8134 | 0.750 | 5.1 | 3.8 | 63.5 | 1.5 | 1.8 | 1.0000 |  |
| minimax | 2 | down | fp4 | - | 12.1716 | 22.9514 | 0.530 | 3.3 | 1.7 | 41.2 | 6.6 | 0.2 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | down | fp8 | - | 19.6265 | 31.2915 | 0.627 | 3.8 | 2.4 | 48.1 | 2.3 | 0.5 | 1.0000 |  |
| minimax | 2 | gate_up | fp4 | bf16 | 20.5149 | 32.6855 | 0.628 | 3.9 | 2.5 | 48.9 | 5.4 | 2.9 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | gate_up | fp8 | bf16 | 26.3280 | 31.9791 | 0.823 | 5.7 | 4.7 | 71.7 | 0.6 | 1.8 | 1.0000 |  |
| minimax | 2 | gate_up | fp8 | fp8 | 25.8666 | 32.9219 | 0.786 | 5.8 | 4.6 | 73.0 | 1.0 | 2.2 | 1.0000 |  |
| minimax | 4 | down | fp4 | - | 22.1102 | 27.3226 | 0.809 | 3.6 | 2.9 | 45.4 | 3.0 | 3.1 | 1.0000 |  |
| minimax | 4 | down | fp8 | - | 31.8650 | 45.3584 | 0.703 | 4.7 | 3.3 | 59.2 | 1.7 | 4.2 | 1.0000 |  |
| minimax | 4 | gate_up | fp4 | bf16 | 38.7521 | 59.4329 | 0.652 | 4.1 | 2.7 | 51.7 | 1.4 | 0.1 | 1.0000 |  |
| minimax | 4 | gate_up | fp8 | bf16 | 49.9203 | 56.7957 | 0.879 | 6.0 | 5.3 | 75.6 | 1.0 | 2.7 | 1.0000 |  |
| minimax | 4 | gate_up | fp8 | fp8 | 50.0624 | 59.1341 | 0.847 | 6.0 | 5.1 | 75.4 | 0.5 | 3.1 | 1.0000 |  |
| minimax | 8 | down | fp4 | - | 41.8095 | 49.4869 | 0.845 | 3.8 | 3.2 | 48.0 | 0.5 | 2.9 | 1.0000 |  |
| minimax | 8 | down | fp8 | - | 61.6534 | 80.1055 | 0.770 | 4.9 | 3.8 | 61.2 | 3.1 | 5.1 | 1.0000 | noisy (>5% spread) |
| minimax | 8 | gate_up | fp4 | bf16 | 75.3556 | 112.6698 | 0.669 | 4.3 | 2.8 | 53.2 | 5.3 | 1.1 | 1.0000 | noisy (>5% spread) |
| minimax | 8 | gate_up | fp8 | bf16 | 98.3914 | 104.3189 | 0.943 | 6.1 | 5.8 | 76.7 | 1.8 | 2.0 | 1.0000 |  |
| minimax | 8 | gate_up | fp8 | fp8 | 97.9958 | 108.0370 | 0.907 | 6.2 | 5.6 | 77.0 | 1.4 | 0.8 | 1.0000 |  |
| minimax | 32 | down | fp4 | - | 146.3079 | 168.8051 | 0.867 | 4.4 | 3.8 | 54.8 | 0.6 | 0.3 | 1.0000 |  |
| minimax | 32 | down | fp8 | - | 223.6144 | 279.3721 | 0.800 | 5.4 | 4.3 | 67.5 | 2.1 | 6.0 | 1.0000 | noisy (>5% spread) |
| minimax | 32 | gate_up | fp4 | bf16 | 264.8557 | 431.8723 | 0.613 | 4.8 | 3.0 | 60.6 | 1.3 | 0.7 | 1.0000 |  |
| minimax | 32 | gate_up | fp8 | bf16 | 384.7362 | 406.0465 | 0.948 | 6.3 | 5.9 | 78.5 | 0.6 | 1.4 | 1.0000 |  |
| minimax | 32 | gate_up | fp8 | fp8 | 381.9464 | 417.1182 | 0.916 | 6.3 | 5.8 | 79.1 | 0.9 | 2.0 | 1.0000 |  |
| qwen3next | 1 | down | fp4 | - | 5.1490 | 10.7153 | 0.481 | 1.1 | 0.5 | 13.5 | 0.9 | 0.2 | 1.0000 |  |
| qwen3next | 1 | down | fp8 | - | 6.3670 | 11.0525 | 0.576 | 1.6 | 0.9 | 20.6 | 0.8 | 1.2 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp4 | bf16 | 5.9621 | 7.4485 | 0.800 | 1.9 | 1.5 | 23.4 | 0.3 | 3.0 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp8 | bf16 | 7.4699 | 7.8197 | 0.955 | 2.8 | 2.7 | 35.1 | 2.1 | 6.7 | 1.0000 | noisy (>5% spread) |
| qwen3next | 1 | gate_up | fp8 | fp8 | 7.1915 | 8.5205 | 0.844 | 2.9 | 2.5 | 36.5 | 0.4 | 6.0 | 1.0000 | noisy (>5% spread) |
| qwen3next | 2 | down | fp4 | - | 5.7904 | 10.9577 | 0.528 | 1.9 | 1.0 | 24.1 | 0.3 | 0.3 | 1.0000 |  |
| qwen3next | 2 | down | fp8 | - | 7.8053 | 14.9701 | 0.521 | 2.7 | 1.4 | 33.6 | 0.7 | 0.4 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp4 | bf16 | 8.5670 | 11.0829 | 0.773 | 2.6 | 2.0 | 32.5 | 3.2 | 0.7 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | bf16 | 10.8338 | 12.6257 | 0.858 | 3.9 | 3.3 | 48.4 | 0.2 | 0.4 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | fp8 | 10.4130 | 13.0205 | 0.800 | 4.0 | 3.2 | 50.3 | 0.8 | 0.9 | 1.0000 |  |
| qwen3next | 4 | down | fp4 | - | 8.2019 | 12.7417 | 0.644 | 2.7 | 1.7 | 34.0 | 0.4 | 0.2 | 1.0000 |  |
| qwen3next | 4 | down | fp8 | - | 12.1218 | 20.2262 | 0.599 | 3.5 | 2.1 | 43.3 | 0.1 | 0.5 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp4 | bf16 | 12.6622 | 17.3406 | 0.730 | 3.5 | 2.6 | 44.0 | 5.1 | 0.7 | 1.0000 | noisy (>5% spread) |
| qwen3next | 4 | gate_up | fp8 | bf16 | 17.4278 | 19.7362 | 0.883 | 4.8 | 4.3 | 60.2 | 0.3 | 4.4 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp8 | fp8 | 16.8860 | 21.0582 | 0.802 | 5.0 | 4.0 | 62.1 | 1.1 | 3.9 | 1.0000 |  |
| qwen3next | 8 | down | fp4 | - | 12.9576 | 15.1585 | 0.855 | 3.4 | 2.9 | 43.0 | 0.0 | 0.6 | 1.0000 |  |
| qwen3next | 8 | down | fp8 | - | 20.7077 | 27.1714 | 0.762 | 4.1 | 3.1 | 50.6 | 7.7 | 4.3 | 1.0000 | noisy (>5% spread) |
| qwen3next | 8 | gate_up | fp4 | bf16 | 21.1857 | 30.1943 | 0.702 | 4.2 | 3.0 | 52.6 | 7.7 | 0.3 | 1.0000 | noisy (>5% spread) |
| qwen3next | 8 | gate_up | fp8 | bf16 | 30.0859 | 32.9279 | 0.914 | 5.6 | 5.1 | 69.7 | 2.4 | 3.8 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp8 | fp8 | 29.0298 | 34.6127 | 0.839 | 5.8 | 4.8 | 72.2 | 0.7 | 4.7 | 1.0000 |  |
| qwen3next | 32 | down | fp4 | - | 40.7298 | 52.4081 | 0.777 | 4.4 | 3.4 | 54.7 | 1.6 | 0.9 | 1.0000 |  |
| qwen3next | 32 | down | fp8 | - | 65.6137 | 88.4076 | 0.742 | 5.1 | 3.8 | 63.9 | 3.7 | 2.8 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp4 | bf16 | 74.9261 | 105.8914 | 0.708 | 4.8 | 3.4 | 59.5 | 1.1 | 0.4 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp8 | bf16 | 107.4067 | 114.4290 | 0.939 | 6.2 | 5.9 | 78.1 | 0.3 | 1.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp8 | fp8 | 104.2960 | 125.8188 | 0.829 | 6.4 | 5.3 | 80.4 | 0.4 | 1.1 | 1.0000 |  |

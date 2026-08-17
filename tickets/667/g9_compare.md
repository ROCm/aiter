<!-- SILOTIGER-667 G9 FlyDSL-vs-CK cold warp-decode comparison -->
<!-- gfx=gfx950  aiter=256d1001b  ck_worktree=62e30c90989 -->
<!-- iters=1000 cold=20 timing=device method=weight_stream repeats=3 -->
<!-- CK provenance: # ck_bench_warp_decode  commit=62e30c9098  cold=20  iters=1000  rotate=auto(ceil(E/BK))  format=csv  mechanism=manual-hipEvent+disjoint-router-rotation -->
<!-- clocks: auto (unpinnable on this gfx950; D1) -- effective loaded sclk MHz min/median/max = 1507/2394/2404 (n=194/197) on GPU 6; per-cell spread%% + noisy flag (>5%) capture drift (D5). -->
<!-- config policy: default-vs-default (D3); treat under-converged fast cells as noisy (D1). -->

**metric method:** `weight_stream` &nbsp; (ratio = flydsl_us / ck_us; CK is perf-only / uninitialized weights)

| shape | B | op | dtype | act | flydsl_us | ck_us | ratio(f/c) | fly_TB/s | ck_TB/s | fly_%peak | fly_spr% | ck_spr% | cos | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3 | 1 | down | fp4 | - | 18.1595 | 29.6787 | 0.612 | 3.4 | 2.1 | 42.9 | 6.4 | 0.2 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | down | fp8 | - | n/a | 35.4379 | n/a | n/a | 3.3 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 1 | gate_up | fp4 | bf16 | 29.1259 | 50.2479 | 0.580 | 4.3 | 2.5 | 53.6 | 2.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | bf16 | n/a | 46.5295 | n/a | n/a | 5.0 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 2 | down | fp4 | - | 31.3694 | 38.1140 | 0.823 | 4.0 | 3.3 | 49.7 | 0.4 | 1.6 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp8 | - | n/a | 51.9136 | n/a | n/a | 4.5 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 2 | gate_up | fp4 | bf16 | 53.4316 | 92.3103 | 0.579 | 4.7 | 2.7 | 58.4 | 0.3 | 0.0 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp8 | bf16 | n/a | 86.4602 | n/a | n/a | 5.4 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 4 | down | fp4 | - | 54.9192 | 69.1565 | 0.794 | 4.5 | 3.6 | 56.8 | 0.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 4 | down | fp8 | - | n/a | 93.3772 | n/a | n/a | 5.0 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 4 | gate_up | fp4 | bf16 | 104.6358 | 173.8979 | 0.602 | 4.8 | 2.9 | 59.6 | 0.4 | 0.2 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp8 | bf16 | n/a | 165.3396 | n/a | n/a | 5.7 | n/a | n/a | 0.4 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 8 | down | fp4 | - | 107.8973 | 134.5611 | 0.802 | 4.6 | 3.7 | 57.8 | 2.2 | 0.1 | 1.0000 |  |
| deepseek-v3 | 8 | down | fp8 | - | n/a | 179.9425 | n/a | n/a | 5.2 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 8 | gate_up | fp4 | bf16 | 208.2891 | 339.0975 | 0.614 | 4.8 | 2.9 | 59.9 | 0.2 | 0.1 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp8 | bf16 | n/a | 317.1326 | n/a | n/a | 5.9 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 32 | down | fp4 | - | 416.2370 | 480.3306 | 0.867 | 4.8 | 4.2 | 60.0 | 0.1 | 0.2 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp8 | - | n/a | 673.9190 | n/a | n/a | 5.6 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 32 | gate_up | fp4 | bf16 | 840.9866 | 1330.3783 | 0.632 | 4.7 | 3.0 | 59.3 | 0.2 | 0.1 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp8 | bf16 | n/a | 1261.7803 | n/a | n/a | 6.0 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| minimax | 1 | down | fp4 | - | 9.8269 | 20.8752 | 0.471 | 2.0 | 1.0 | 25.5 | 1.3 | 0.1 | 1.0000 |  |
| minimax | 1 | down | fp8 | - | 11.7430 | 23.2273 | 0.506 | 3.2 | 1.6 | 40.2 | 7.5 | 0.1 | 1.0000 | noisy (>5% spread) |
| minimax | 1 | gate_up | fp4 | bf16 | 11.4687 | 18.4187 | 0.623 | 3.5 | 2.2 | 43.7 | 0.1 | 0.5 | 1.0000 |  |
| minimax | 1 | gate_up | fp8 | bf16 | 14.7509 | 19.2084 | 0.768 | 5.1 | 3.9 | 64.0 | 0.2 | 0.1 | 1.0000 |  |
| minimax | 2 | down | fp4 | - | 13.0909 | 24.0578 | 0.544 | 3.1 | 1.7 | 38.3 | 7.5 | 0.1 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | down | fp8 | - | 18.5441 | 31.7458 | 0.584 | 4.1 | 2.4 | 50.9 | 7.3 | 2.8 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | gate_up | fp4 | bf16 | 18.2783 | 32.5577 | 0.561 | 4.4 | 2.5 | 54.9 | 1.3 | 0.7 | 1.0000 |  |
| minimax | 2 | gate_up | fp8 | bf16 | 26.0768 | 32.1445 | 0.811 | 5.8 | 4.7 | 72.4 | 0.8 | 0.1 | 1.0000 |  |
| minimax | 4 | down | fp4 | - | 22.2039 | 29.6480 | 0.749 | 3.6 | 2.7 | 45.2 | 1.1 | 0.2 | 1.0000 |  |
| minimax | 4 | down | fp8 | - | 32.6574 | 36.5466 | 0.894 | 4.6 | 4.1 | 57.8 | 2.5 | 0.1 | 1.0000 |  |
| minimax | 4 | gate_up | fp4 | bf16 | 34.8996 | 59.7388 | 0.584 | 4.6 | 2.7 | 57.5 | 0.2 | 0.1 | 1.0000 |  |
| minimax | 4 | gate_up | fp8 | bf16 | 49.6131 | 56.3480 | 0.880 | 6.1 | 5.4 | 76.1 | 1.9 | 0.1 | 1.0000 |  |
| minimax | 8 | down | fp4 | - | 40.0078 | 53.8557 | 0.743 | 4.0 | 3.0 | 50.1 | 4.8 | 0.1 | 1.0000 |  |
| minimax | 8 | down | fp8 | - | 60.7133 | 65.5697 | 0.926 | 5.0 | 4.6 | 62.2 | 0.3 | 0.1 | 1.0000 |  |
| minimax | 8 | gate_up | fp4 | bf16 | 64.6681 | 112.2778 | 0.576 | 5.0 | 2.9 | 62.0 | 3.8 | 0.1 | 1.0000 |  |
| minimax | 8 | gate_up | fp8 | bf16 | 97.1357 | 104.2554 | 0.932 | 6.2 | 5.8 | 77.7 | 0.2 | 0.2 | 1.0000 |  |
| minimax | 32 | down | fp4 | - | 149.4718 | 171.4806 | 0.872 | 4.3 | 3.7 | 53.7 | 1.9 | 1.1 | 1.0000 |  |
| minimax | 32 | down | fp8 | - | 228.5380 | 227.8319 | 1.003 | 5.3 | 5.3 | 66.1 | 0.7 | 0.1 | 1.0000 |  |
| minimax | 32 | gate_up | fp4 | bf16 | 252.0858 | 431.5867 | 0.584 | 5.1 | 3.0 | 63.6 | 0.3 | 0.2 | 1.0000 |  |
| minimax | 32 | gate_up | fp8 | bf16 | 388.7390 | 401.6740 | 0.968 | 6.2 | 6.0 | 77.7 | 0.3 | 0.3 | 1.0000 |  |
| qwen3next | 1 | down | fp4 | - | 5.2364 | 9.7527 | 0.537 | 1.1 | 0.6 | 13.3 | 0.4 | 0.2 | 1.0000 |  |
| qwen3next | 1 | down | fp8 | - | 5.6364 | 10.3585 | 0.544 | 1.9 | 1.0 | 23.3 | 1.6 | 0.5 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp4 | bf16 | 6.0312 | 7.2173 | 0.836 | 1.8 | 1.5 | 23.1 | 0.7 | 0.8 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp8 | bf16 | 7.0149 | 7.8413 | 0.895 | 3.0 | 2.7 | 37.4 | 2.4 | 0.3 | 1.0000 |  |
| qwen3next | 2 | down | fp4 | - | 5.8380 | 11.3555 | 0.514 | 1.9 | 1.0 | 23.9 | 0.6 | 0.1 | 1.0000 |  |
| qwen3next | 2 | down | fp8 | - | 7.1058 | 15.2204 | 0.467 | 3.0 | 1.4 | 36.9 | 0.3 | 0.2 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp4 | bf16 | 8.7025 | 10.8631 | 0.801 | 2.6 | 2.1 | 32.0 | 2.3 | 1.6 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | bf16 | 9.9849 | 12.7389 | 0.784 | 4.2 | 3.3 | 52.5 | 2.2 | 6.1 | 1.0000 | noisy (>5% spread) |
| qwen3next | 4 | down | fp4 | - | 8.3447 | 13.6203 | 0.613 | 2.7 | 1.6 | 33.4 | 1.0 | 0.1 | 1.0000 |  |
| qwen3next | 4 | down | fp8 | - | 10.7736 | 17.7597 | 0.607 | 3.9 | 2.4 | 48.7 | 0.8 | 0.2 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp4 | bf16 | 13.1998 | 17.0583 | 0.774 | 3.4 | 2.6 | 42.2 | 4.9 | 0.7 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp8 | bf16 | 16.2464 | 20.0436 | 0.811 | 5.2 | 4.2 | 64.5 | 0.0 | 0.2 | 1.0000 |  |
| qwen3next | 8 | down | fp4 | - | 12.2752 | 16.1667 | 0.759 | 3.6 | 2.8 | 45.4 | 1.0 | 0.2 | 1.0000 |  |
| qwen3next | 8 | down | fp8 | - | 18.8520 | 23.3640 | 0.807 | 4.4 | 3.6 | 55.6 | 1.3 | 0.1 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp4 | bf16 | 22.6294 | 29.5806 | 0.765 | 3.9 | 3.0 | 49.2 | 0.4 | 0.3 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp8 | bf16 | 28.9910 | 32.9025 | 0.881 | 5.8 | 5.1 | 72.3 | 0.2 | 0.4 | 1.0000 |  |
| qwen3next | 32 | down | fp4 | - | 37.1478 | 55.0559 | 0.675 | 4.8 | 3.2 | 60.0 | 0.4 | 0.2 | 1.0000 |  |
| qwen3next | 32 | down | fp8 | - | 62.1377 | 73.1556 | 0.849 | 5.4 | 4.6 | 67.5 | 0.1 | 0.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp4 | bf16 | 73.2585 | 104.8636 | 0.699 | 4.9 | 3.4 | 60.8 | 3.8 | 0.2 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp8 | bf16 | 103.6841 | 112.7071 | 0.920 | 6.5 | 6.0 | 80.9 | 0.2 | 0.6 | 1.0000 |  |

<!-- SILOTIGER-667 G9 FlyDSL-vs-CK cold warp-decode comparison -->
<!-- gfx=gfx950  aiter=eab5bcbe3  ck_worktree=62e30c90989 -->
<!-- iters=1000 cold=20 timing=device method=weight_stream repeats=3 -->
<!-- CK provenance: # ck_bench_warp_decode  commit=62e30c9098  cold=20  iters=1000  rotate=auto(ceil(E/BK))  format=csv  mechanism=manual-hipEvent+disjoint-router-rotation -->
<!-- clocks: auto (unpinnable on this gfx950; D1) -- effective loaded sclk MHz min/median/max = 1478/2391/2403 (n=168/171) on GPU 6; per-cell spread%% + noisy flag (>5%) capture drift (D5). -->
<!-- config policy: default-vs-default (D3); treat under-converged fast cells as noisy (D1). -->

**metric method:** `weight_stream` &nbsp; (ratio = flydsl_us / ck_us; CK is perf-only / uninitialized weights)

| shape | B | op | dtype | act | flydsl_us | ck_us | ratio(f/c) | fly_TB/s | ck_TB/s | fly_%peak | fly_spr% | ck_spr% | cos | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3 | 1 | down | fp4 | - | 18.1061 | 29.5318 | 0.613 | 3.4 | 2.1 | 43.1 | 7.9 | 0.2 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | down | fp8 | - | n/a | 34.4005 | n/a | n/a | 3.4 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 1 | gate_up | fp4 | bf16 | 29.0998 | n/a | n/a | 4.3 | n/a | 53.6 | 2.7 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| deepseek-v3 | 1 | gate_up | fp8 | bf16 | n/a | 46.5189 | n/a | n/a | 5.0 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 2 | down | fp4 | - | 31.2764 | 38.1792 | 0.819 | 4.0 | 3.3 | 49.9 | 0.3 | 0.1 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp8 | - | n/a | 51.8739 | n/a | n/a | 4.5 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 2 | gate_up | fp4 | bf16 | 53.4554 | n/a | n/a | 4.7 | n/a | 58.4 | 0.2 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| deepseek-v3 | 2 | gate_up | fp8 | bf16 | n/a | 87.0369 | n/a | n/a | 5.4 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 4 | down | fp4 | - | 54.9228 | 69.1817 | 0.794 | 4.5 | 3.6 | 56.8 | 0.2 | 0.1 | 1.0000 |  |
| deepseek-v3 | 4 | down | fp8 | - | n/a | 90.9481 | n/a | n/a | 5.2 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 4 | gate_up | fp4 | bf16 | 104.5487 | n/a | n/a | 4.8 | n/a | 59.7 | 0.1 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| deepseek-v3 | 4 | gate_up | fp8 | bf16 | n/a | 166.3183 | n/a | n/a | 5.6 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 8 | down | fp4 | - | 107.9163 | 134.4822 | 0.802 | 4.6 | 3.7 | 57.8 | 2.0 | 0.1 | 1.0000 |  |
| deepseek-v3 | 8 | down | fp8 | - | n/a | 175.3445 | n/a | n/a | 5.4 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 8 | gate_up | fp4 | bf16 | 208.2781 | n/a | n/a | 4.8 | n/a | 59.9 | 0.1 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| deepseek-v3 | 8 | gate_up | fp8 | bf16 | n/a | 312.8437 | n/a | n/a | 6.0 | n/a | n/a | 1.4 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 32 | down | fp4 | - | 416.2006 | 482.6565 | 0.862 | 4.8 | 4.1 | 60.0 | 0.0 | 0.2 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp8 | - | n/a | 677.6060 | n/a | n/a | 5.5 | n/a | n/a | 0.2 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 32 | gate_up | fp4 | bf16 | 840.7088 | n/a | n/a | 4.7 | n/a | 59.4 | 0.2 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| deepseek-v3 | 32 | gate_up | fp8 | bf16 | n/a | 1265.1093 | n/a | n/a | 5.9 | n/a | n/a | 0.4 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| minimax | 1 | down | fp4 | - | 9.9138 | 20.8614 | 0.475 | 2.0 | 1.0 | 25.3 | 2.0 | 0.1 | 1.0000 |  |
| minimax | 1 | down | fp8 | - | 11.7191 | 23.2474 | 0.504 | 3.2 | 1.6 | 40.3 | 3.8 | 0.1 | 1.0000 |  |
| minimax | 1 | gate_up | fp4 | bf16 | 11.4566 | n/a | n/a | 3.5 | n/a | 43.8 | 0.1 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| minimax | 1 | gate_up | fp8 | bf16 | 14.7508 | 19.2281 | 0.767 | 5.1 | 3.9 | 64.0 | 0.0 | 0.2 | 1.0000 |  |
| minimax | 2 | down | fp4 | - | 13.0746 | 24.0615 | 0.543 | 3.1 | 1.7 | 38.3 | 7.8 | 0.2 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | down | fp8 | - | 18.5336 | 27.4445 | 0.675 | 4.1 | 2.8 | 50.9 | 10.4 | 0.2 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | gate_up | fp4 | bf16 | 18.3826 | n/a | n/a | 4.4 | n/a | 54.5 | 1.9 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| minimax | 2 | gate_up | fp8 | bf16 | 26.1033 | 32.1790 | 0.811 | 5.8 | 4.7 | 72.3 | 0.9 | 0.1 | 1.0000 |  |
| minimax | 4 | down | fp4 | - | 22.2156 | 29.6927 | 0.748 | 3.6 | 2.7 | 45.1 | 0.5 | 0.1 | 1.0000 |  |
| minimax | 4 | down | fp8 | - | 32.6155 | 34.6355 | 0.942 | 4.6 | 4.4 | 57.9 | 0.1 | 0.5 | 1.0000 |  |
| minimax | 4 | gate_up | fp4 | bf16 | 34.9160 | n/a | n/a | 4.6 | n/a | 57.4 | 0.2 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| minimax | 4 | gate_up | fp8 | bf16 | 49.6079 | 56.7208 | 0.875 | 6.1 | 5.3 | 76.1 | 1.8 | 0.3 | 1.0000 |  |
| minimax | 8 | down | fp4 | - | 39.9637 | 53.8762 | 0.742 | 4.0 | 3.0 | 50.2 | 2.0 | 0.0 | 1.0000 |  |
| minimax | 8 | down | fp8 | - | 60.7598 | 63.8912 | 0.951 | 5.0 | 4.7 | 62.1 | 0.1 | 0.1 | 1.0000 |  |
| minimax | 8 | gate_up | fp4 | bf16 | 65.2031 | n/a | n/a | 4.9 | n/a | 61.5 | 3.8 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| minimax | 8 | gate_up | fp8 | bf16 | 97.1088 | 104.0784 | 0.933 | 6.2 | 5.8 | 77.7 | 0.3 | 0.1 | 1.0000 |  |
| minimax | 32 | down | fp4 | - | 149.2579 | 171.1138 | 0.872 | 4.3 | 3.8 | 53.7 | 1.9 | 0.2 | 1.0000 |  |
| minimax | 32 | down | fp8 | - | 226.1061 | 230.9302 | 0.979 | 5.3 | 5.2 | 66.8 | 1.3 | 0.5 | 1.0000 |  |
| minimax | 32 | gate_up | fp4 | bf16 | 251.5618 | n/a | n/a | 5.1 | n/a | 63.8 | 1.9 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| minimax | 32 | gate_up | fp8 | bf16 | 388.8475 | 400.7598 | 0.970 | 6.2 | 6.0 | 77.7 | 0.0 | 0.3 | 1.0000 |  |
| qwen3next | 1 | down | fp4 | - | 5.2235 | 9.7732 | 0.534 | 1.1 | 0.6 | 13.3 | 0.8 | 0.4 | 1.0000 |  |
| qwen3next | 1 | down | fp8 | - | 5.6963 | 10.4018 | 0.548 | 1.8 | 1.0 | 23.0 | 0.5 | 0.9 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp4 | bf16 | 6.0645 | n/a | n/a | 1.8 | n/a | 23.0 | 0.4 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| qwen3next | 1 | gate_up | fp8 | bf16 | 7.0200 | 7.8682 | 0.892 | 3.0 | 2.7 | 37.3 | 2.9 | 0.0 | 1.0000 |  |
| qwen3next | 2 | down | fp4 | - | 5.8459 | 11.3659 | 0.514 | 1.9 | 1.0 | 23.8 | 0.2 | 0.1 | 1.0000 |  |
| qwen3next | 2 | down | fp8 | - | 7.1304 | 15.2976 | 0.466 | 2.9 | 1.4 | 36.8 | 0.6 | 0.4 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp4 | bf16 | 8.6780 | n/a | n/a | 2.6 | n/a | 32.1 | 1.9 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| qwen3next | 2 | gate_up | fp8 | bf16 | 9.8272 | 12.7585 | 0.770 | 4.3 | 3.3 | 53.4 | 0.3 | 0.1 | 1.0000 |  |
| qwen3next | 4 | down | fp4 | - | 8.3986 | 13.6344 | 0.616 | 2.7 | 1.6 | 33.2 | 0.6 | 0.2 | 1.0000 |  |
| qwen3next | 4 | down | fp8 | - | 10.7023 | 17.7308 | 0.604 | 3.9 | 2.4 | 49.0 | 0.5 | 0.5 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp4 | bf16 | 13.0743 | n/a | n/a | 3.4 | n/a | 42.6 | 4.1 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| qwen3next | 4 | gate_up | fp8 | bf16 | 16.1441 | 20.0033 | 0.807 | 5.2 | 4.2 | 65.0 | 0.7 | 0.2 | 1.0000 |  |
| qwen3next | 8 | down | fp4 | - | 12.4538 | 16.1789 | 0.770 | 3.6 | 2.8 | 44.7 | 0.1 | 0.3 | 1.0000 |  |
| qwen3next | 8 | down | fp8 | - | 19.2306 | 22.5729 | 0.852 | 4.4 | 3.7 | 54.5 | 3.1 | 0.3 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp4 | bf16 | 22.7173 | n/a | n/a | 3.9 | n/a | 49.0 | 0.9 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| qwen3next | 8 | gate_up | fp8 | bf16 | 28.9468 | 32.8102 | 0.882 | 5.8 | 5.1 | 72.4 | 0.3 | 0.1 | 1.0000 |  |
| qwen3next | 32 | down | fp4 | - | 37.3789 | 54.9180 | 0.681 | 4.8 | 3.2 | 59.6 | 0.4 | 0.2 | 1.0000 |  |
| qwen3next | 32 | down | fp8 | - | 62.0689 | 72.5373 | 0.856 | 5.4 | 4.6 | 67.6 | 0.1 | 0.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp4 | bf16 | 73.4788 | n/a | n/a | 4.9 | n/a | 60.6 | 3.4 | n/a | 1.0000 | CK n/a (gate_up FP4 -> A4) |
| qwen3next | 32 | gate_up | fp8 | bf16 | 103.8476 | 112.8212 | 0.920 | 6.5 | 5.9 | 80.8 | 0.1 | 0.1 | 1.0000 |  |

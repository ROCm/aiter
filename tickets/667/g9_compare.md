<!-- SILOTIGER-667 G9 FlyDSL-vs-CK cold warp-decode comparison -->
<!-- gfx=gfx950  aiter=7d325bc40  ck_worktree=c03392a91b8 -->
<!-- iters=1000 cold=20 timing=device method=weight_stream repeats=3 -->
<!-- CK provenance: # ck_bench_warp_decode  base_commit=62e30c9098 patch=A4-gateup-fp4-packed-stride  cold=20  iters=1000  rotate=auto(ceil(E/BK))  format=csv  mechanism=manual-hipEvent+disjoint-router-rotation -->
<!-- clocks: auto (unpinnable on this gfx950; D1) -- effective loaded sclk MHz min/median/max = 1698/2394/2403 (n=193/196) on GPU 6; per-cell spread%% + noisy flag (>5%) capture drift (D5). -->
<!-- config policy (D3): default-vs-default. FlyDSL = library defaults, no overrides: serialize_dot2=True, kh_per_warp=auto(2 when HIDDEN even), prefetch=False; down_fp4 dot2_acc=4, gate_up_fp4 dot2_acc=1 (G7: acc>1 ~4% slower for gate_up); down_fp8 split_k=1; FP8 w_scale=block2d(128,128) to match CK. CK = maintainer-recommended variant per op (down_h2_d2, down_fp4_h2, gate_bf16_d2, gate_up_fp4 non-dot2/NPerWarp=1); CK has no single runtime default (mild asymmetry). FP8-down ratio is a CK-favored lower bound: block2d(128,128) costs FlyDSL ~10-38% vs pertensor (B1); FP4 rows carry a ~6% CK-favored scale-traffic bias (dummy PerTensor vs e8m0(1,32)). Treat under-converged fast cells as noisy (D1). -->

**metric method:** `weight_stream` &nbsp; (ratio = flydsl_us / ck_us; CK is perf-only / uninitialized weights)

| shape | B | op | dtype | act | flydsl_us | ck_us | ratio(f/c) | fly_TB/s | ck_TB/s | fly_%peak | fly_spr% | ck_spr% | cos | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3 | 1 | down | fp4 | - | 18.1453 | 29.6461 | 0.612 | 3.4 | 2.1 | 43.0 | 5.8 | 0.3 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | down | fp8 | - | n/a | 35.3736 | n/a | n/a | 3.3 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 1 | gate_up | fp4 | bf16 | 29.0553 | 50.2104 | 0.579 | 4.3 | 2.5 | 53.7 | 2.5 | 0.1 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | bf16 | n/a | 46.5244 | n/a | n/a | 5.0 | n/a | n/a | 0.2 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 2 | down | fp4 | - | 31.2949 | 37.4887 | 0.835 | 4.0 | 3.3 | 49.8 | 0.1 | 0.2 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp8 | - | n/a | 51.8767 | n/a | n/a | 4.5 | n/a | n/a | 0.2 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 2 | gate_up | fp4 | bf16 | 53.3716 | 92.3138 | 0.578 | 4.7 | 2.7 | 58.4 | 0.4 | 0.0 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp8 | bf16 | n/a | 86.4009 | n/a | n/a | 5.4 | n/a | n/a | 0.2 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 4 | down | fp4 | - | 54.8550 | 69.0738 | 0.794 | 4.5 | 3.6 | 56.9 | 0.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 4 | down | fp8 | - | n/a | 93.4108 | n/a | n/a | 5.0 | n/a | n/a | 0.2 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 4 | gate_up | fp4 | bf16 | 104.4302 | 173.7681 | 0.601 | 4.8 | 2.9 | 59.7 | 0.1 | 0.1 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp8 | bf16 | n/a | 164.6944 | n/a | n/a | 5.7 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 8 | down | fp4 | - | 107.8992 | 134.5758 | 0.802 | 4.6 | 3.7 | 57.8 | 2.3 | 0.1 | 1.0000 |  |
| deepseek-v3 | 8 | down | fp8 | - | n/a | 179.9731 | n/a | n/a | 5.2 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 8 | gate_up | fp4 | bf16 | 208.1574 | 339.2121 | 0.614 | 4.8 | 2.9 | 59.9 | 0.2 | 0.0 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp8 | bf16 | n/a | 317.0799 | n/a | n/a | 5.9 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 32 | down | fp4 | - | 416.0498 | 480.9226 | 0.865 | 4.8 | 4.2 | 60.0 | 0.2 | 0.2 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp8 | - | n/a | 673.8299 | n/a | n/a | 5.6 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 32 | gate_up | fp4 | bf16 | 839.0408 | 1330.6024 | 0.631 | 4.8 | 3.0 | 59.5 | 0.3 | 0.1 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp8 | bf16 | n/a | 1261.3939 | n/a | n/a | 6.0 | n/a | n/a | 0.2 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| minimax | 1 | down | fp4 | - | 9.9348 | 20.8878 | 0.476 | 2.0 | 1.0 | 25.2 | 1.5 | 0.2 | 1.0000 |  |
| minimax | 1 | down | fp8 | - | 12.0506 | 23.2594 | 0.518 | 3.1 | 1.6 | 39.2 | 5.8 | 0.1 | 1.0000 | noisy (>5% spread) |
| minimax | 1 | gate_up | fp4 | bf16 | 11.5039 | 18.3964 | 0.625 | 3.5 | 2.2 | 43.6 | 0.4 | 0.6 | 1.0000 |  |
| minimax | 1 | gate_up | fp8 | bf16 | 14.9456 | 19.2601 | 0.776 | 5.1 | 3.9 | 63.1 | 0.2 | 0.5 | 1.0000 |  |
| minimax | 2 | down | fp4 | - | 13.3267 | 27.3360 | 0.488 | 3.0 | 1.5 | 37.6 | 7.5 | 12.3 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | down | fp8 | - | 18.6686 | 27.4500 | 0.680 | 4.0 | 2.8 | 50.6 | 10.0 | 0.2 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | gate_up | fp4 | bf16 | 18.3276 | 32.7425 | 0.560 | 4.4 | 2.4 | 54.7 | 1.4 | 0.1 | 1.0000 |  |
| minimax | 2 | gate_up | fp8 | bf16 | 26.0758 | 32.1434 | 0.811 | 5.8 | 4.7 | 72.4 | 0.8 | 0.1 | 1.0000 |  |
| minimax | 4 | down | fp4 | - | 22.3383 | 29.6531 | 0.753 | 3.6 | 2.7 | 44.9 | 0.8 | 0.0 | 1.0000 |  |
| minimax | 4 | down | fp8 | - | 32.5741 | 36.5965 | 0.890 | 4.6 | 4.1 | 57.9 | 2.3 | 0.1 | 1.0000 |  |
| minimax | 4 | gate_up | fp4 | bf16 | 34.8894 | 59.7405 | 0.584 | 4.6 | 2.7 | 57.5 | 0.1 | 0.1 | 1.0000 |  |
| minimax | 4 | gate_up | fp8 | bf16 | 49.6355 | 56.3683 | 0.881 | 6.1 | 5.4 | 76.1 | 1.8 | 0.2 | 1.0000 |  |
| minimax | 8 | down | fp4 | - | 39.9162 | 53.8744 | 0.741 | 4.0 | 3.0 | 50.2 | 2.0 | 0.1 | 1.0000 |  |
| minimax | 8 | down | fp8 | - | 60.6538 | 65.5963 | 0.925 | 5.0 | 4.6 | 62.2 | 1.1 | 0.1 | 1.0000 |  |
| minimax | 8 | gate_up | fp4 | bf16 | 64.2237 | 112.3167 | 0.572 | 5.0 | 2.9 | 62.5 | 3.6 | 0.2 | 1.0000 |  |
| minimax | 8 | gate_up | fp8 | bf16 | 97.1102 | 104.3384 | 0.931 | 6.2 | 5.8 | 77.7 | 0.5 | 0.1 | 1.0000 |  |
| minimax | 32 | down | fp4 | - | 149.3293 | 171.2609 | 0.872 | 4.3 | 3.7 | 53.7 | 0.0 | 0.3 | 1.0000 |  |
| minimax | 32 | down | fp8 | - | 226.6455 | 227.8645 | 0.995 | 5.3 | 5.3 | 66.6 | 1.4 | 0.7 | 1.0000 |  |
| minimax | 32 | gate_up | fp4 | bf16 | 251.2445 | 431.5957 | 0.582 | 5.1 | 3.0 | 63.9 | 1.9 | 0.2 | 1.0000 |  |
| minimax | 32 | gate_up | fp8 | bf16 | 388.6799 | 401.8832 | 0.967 | 6.2 | 6.0 | 77.7 | 0.2 | 0.2 | 1.0000 |  |
| qwen3next | 1 | down | fp4 | - | 5.2268 | 9.7728 | 0.535 | 1.1 | 0.6 | 13.3 | 0.1 | 0.2 | 1.0000 |  |
| qwen3next | 1 | down | fp8 | - | 5.7116 | 10.3765 | 0.550 | 1.8 | 1.0 | 22.9 | 0.3 | 0.1 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp4 | bf16 | 6.0655 | 7.2126 | 0.841 | 1.8 | 1.5 | 23.0 | 0.3 | 0.4 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp8 | bf16 | 7.0350 | 7.8502 | 0.896 | 3.0 | 2.7 | 37.3 | 5.4 | 0.5 | 1.0000 | noisy (>5% spread) |
| qwen3next | 2 | down | fp4 | - | 5.8393 | 11.3508 | 0.514 | 1.9 | 1.0 | 23.8 | 0.2 | 0.2 | 1.0000 |  |
| qwen3next | 2 | down | fp8 | - | 7.1388 | 15.2024 | 0.470 | 2.9 | 1.4 | 36.7 | 0.3 | 0.7 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp4 | bf16 | 8.6882 | 10.9344 | 0.795 | 2.6 | 2.0 | 32.1 | 1.8 | 0.4 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | bf16 | 9.8442 | 12.7384 | 0.773 | 4.3 | 3.3 | 53.3 | 5.8 | 0.2 | 1.0000 | noisy (>5% spread) |
| qwen3next | 4 | down | fp4 | - | 8.3848 | 13.6263 | 0.615 | 2.7 | 1.6 | 33.2 | 1.1 | 0.2 | 1.0000 |  |
| qwen3next | 4 | down | fp8 | - | 10.7613 | 17.7337 | 0.607 | 3.9 | 2.4 | 48.7 | 0.1 | 0.3 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp4 | bf16 | 13.2059 | 17.0219 | 0.776 | 3.4 | 2.6 | 42.2 | 5.2 | 0.2 | 1.0000 | noisy (>5% spread) |
| qwen3next | 4 | gate_up | fp8 | bf16 | 16.1635 | 20.0079 | 0.808 | 5.2 | 4.2 | 64.9 | 4.3 | 0.2 | 1.0000 |  |
| qwen3next | 8 | down | fp4 | - | 12.5754 | 16.1413 | 0.779 | 3.5 | 2.8 | 44.3 | 0.6 | 0.1 | 1.0000 |  |
| qwen3next | 8 | down | fp8 | - | 18.7549 | 23.3650 | 0.803 | 4.5 | 3.6 | 55.9 | 2.9 | 0.1 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp4 | bf16 | 22.6951 | 29.6694 | 0.765 | 3.9 | 3.0 | 49.1 | 1.4 | 0.0 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp8 | bf16 | 29.0067 | 32.8654 | 0.883 | 5.8 | 5.1 | 72.3 | 2.4 | 0.2 | 1.0000 |  |
| qwen3next | 32 | down | fp4 | - | 37.2433 | 54.6628 | 0.681 | 4.8 | 3.3 | 59.8 | 0.4 | 0.4 | 1.0000 |  |
| qwen3next | 32 | down | fp8 | - | 62.0467 | 73.1378 | 0.848 | 5.4 | 4.6 | 67.6 | 0.1 | 0.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp4 | bf16 | 73.3704 | 104.9539 | 0.699 | 4.9 | 3.4 | 60.7 | 3.6 | 0.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp8 | bf16 | 103.7448 | 112.6343 | 0.921 | 6.5 | 6.0 | 80.9 | 0.5 | 0.0 | 1.0000 |  |

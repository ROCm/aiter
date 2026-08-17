<!-- SILOTIGER-667 G9 FlyDSL-vs-CK cold warp-decode comparison -->
<!-- gfx=gfx950  aiter=6bf56cbc4  ck_worktree=c03392a91b8 -->
<!-- iters=1000 cold=20 timing=device method=weight_stream repeats=3 -->
<!-- CK provenance: # ck_bench_warp_decode  commit=62e30c9098  cold=20  iters=1000  rotate=auto(ceil(E/BK))  format=csv  mechanism=manual-hipEvent+disjoint-router-rotation -->
<!-- clocks: auto (unpinnable on this gfx950; D1) -- effective loaded sclk MHz min/median/max = 1470/2395/2403 (n=193/196) on GPU 6; per-cell spread%% + noisy flag (>5%) capture drift (D5). -->
<!-- config policy (D3): default-vs-default. FlyDSL = library defaults, no overrides: serialize_dot2=True, kh_per_warp=auto(2 when HIDDEN even), prefetch=False; down_fp4 dot2_acc=4, gate_up_fp4 dot2_acc=1 (G7: acc>1 ~4% slower for gate_up); down_fp8 split_k=1; FP8 w_scale=block2d(128,128) to match CK. CK = maintainer-recommended variant per op (down_h2_d2, down_fp4_h2, gate_bf16_d2, gate_up_fp4 non-dot2/NPerWarp=1); CK has no single runtime default (mild asymmetry). FP8-down ratio is a CK-favored lower bound: block2d(128,128) costs FlyDSL ~10-38% vs pertensor (B1); FP4 rows carry a ~6% CK-favored scale-traffic bias (dummy PerTensor vs e8m0(1,32)). Treat under-converged fast cells as noisy (D1). -->

**metric method:** `weight_stream` &nbsp; (ratio = flydsl_us / ck_us; CK is perf-only / uninitialized weights)

| shape | B | op | dtype | act | flydsl_us | ck_us | ratio(f/c) | fly_TB/s | ck_TB/s | fly_%peak | fly_spr% | ck_spr% | cos | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v3 | 1 | down | fp4 | - | 18.1698 | 29.6545 | 0.613 | 3.4 | 2.1 | 42.9 | 7.4 | 0.2 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | down | fp8 | - | n/a | 35.4162 | n/a | n/a | 3.3 | n/a | n/a | 0.2 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 1 | gate_up | fp4 | bf16 | 29.1355 | 50.2091 | 0.580 | 4.3 | 2.5 | 53.5 | 2.6 | 0.1 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | bf16 | n/a | 46.5161 | n/a | n/a | 5.0 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 2 | down | fp4 | - | 31.2976 | 37.5535 | 0.833 | 4.0 | 3.3 | 49.8 | 0.1 | 1.6 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp8 | - | n/a | 51.8869 | n/a | n/a | 4.5 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 2 | gate_up | fp4 | bf16 | 53.5045 | 92.3475 | 0.579 | 4.7 | 2.7 | 58.3 | 0.0 | 0.0 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp8 | bf16 | n/a | 86.4003 | n/a | n/a | 5.4 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 4 | down | fp4 | - | 54.8451 | 69.1005 | 0.794 | 4.6 | 3.6 | 56.9 | 0.0 | 0.1 | 1.0000 |  |
| deepseek-v3 | 4 | down | fp8 | - | n/a | 93.3732 | n/a | n/a | 5.0 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 4 | gate_up | fp4 | bf16 | 104.5552 | 173.7550 | 0.602 | 4.8 | 2.9 | 59.7 | 0.2 | 0.2 | 1.0000 |  |
| deepseek-v3 | 4 | gate_up | fp8 | bf16 | n/a | 164.7113 | n/a | n/a | 5.7 | n/a | n/a | 0.4 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 8 | down | fp4 | - | 107.8880 | 134.5709 | 0.802 | 4.6 | 3.7 | 57.8 | 2.1 | 0.2 | 1.0000 |  |
| deepseek-v3 | 8 | down | fp8 | - | n/a | 179.9111 | n/a | n/a | 5.2 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 8 | gate_up | fp4 | bf16 | 208.5363 | 339.1987 | 0.615 | 4.8 | 2.9 | 59.8 | 0.3 | 0.0 | 1.0000 |  |
| deepseek-v3 | 8 | gate_up | fp8 | bf16 | n/a | 317.0283 | n/a | n/a | 5.9 | n/a | n/a | 0.1 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 32 | down | fp4 | - | 415.9976 | 480.7447 | 0.865 | 4.8 | 4.2 | 60.0 | 0.1 | 0.2 | 1.0000 |  |
| deepseek-v3 | 32 | down | fp8 | - | n/a | 674.0883 | n/a | n/a | 5.6 | n/a | n/a | 0.0 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| deepseek-v3 | 32 | gate_up | fp4 | bf16 | 841.7881 | 1330.5324 | 0.633 | 4.7 | 3.0 | 59.3 | 0.1 | 0.2 | 1.0000 |  |
| deepseek-v3 | 32 | gate_up | fp8 | bf16 | n/a | 1262.2932 | n/a | n/a | 6.0 | n/a | n/a | 0.2 | n/a | FlyDSL n/a (DeepSeek FP8 -> K3 Tier-2, B5) |
| minimax | 1 | down | fp4 | - | 9.8436 | 20.8817 | 0.471 | 2.0 | 1.0 | 25.5 | 1.9 | 0.3 | 1.0000 |  |
| minimax | 1 | down | fp8 | - | 11.8029 | 23.1946 | 0.509 | 3.2 | 1.6 | 40.0 | 6.8 | 0.3 | 1.0000 | noisy (>5% spread) |
| minimax | 1 | gate_up | fp4 | bf16 | 11.6131 | 18.4125 | 0.631 | 3.5 | 2.2 | 43.2 | 0.5 | 4.2 | 1.0000 |  |
| minimax | 1 | gate_up | fp8 | bf16 | 14.7479 | 19.2240 | 0.767 | 5.1 | 3.9 | 64.0 | 0.4 | 0.3 | 1.0000 |  |
| minimax | 2 | down | fp4 | - | 13.1382 | 27.5199 | 0.477 | 3.1 | 1.5 | 38.2 | 7.7 | 15.6 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | down | fp8 | - | 18.7638 | 27.4814 | 0.683 | 4.0 | 2.7 | 50.3 | 9.8 | 0.3 | 1.0000 | noisy (>5% spread) |
| minimax | 2 | gate_up | fp4 | bf16 | 18.3902 | 32.7301 | 0.562 | 4.4 | 2.5 | 54.5 | 1.2 | 0.6 | 1.0000 |  |
| minimax | 2 | gate_up | fp8 | bf16 | 26.2205 | 32.1405 | 0.816 | 5.8 | 4.7 | 72.0 | 0.9 | 4.9 | 1.0000 |  |
| minimax | 4 | down | fp4 | - | 22.3758 | 29.6807 | 0.754 | 3.6 | 2.7 | 44.8 | 1.2 | 0.1 | 1.0000 |  |
| minimax | 4 | down | fp8 | - | 32.5952 | 36.5630 | 0.891 | 4.6 | 4.1 | 57.9 | 0.4 | 0.1 | 1.0000 |  |
| minimax | 4 | gate_up | fp4 | bf16 | 34.9438 | 59.7533 | 0.585 | 4.6 | 2.7 | 57.4 | 0.1 | 0.1 | 1.0000 |  |
| minimax | 4 | gate_up | fp8 | bf16 | 49.6118 | 56.3563 | 0.880 | 6.1 | 5.4 | 76.1 | 1.6 | 0.1 | 1.0000 |  |
| minimax | 8 | down | fp4 | - | 39.9441 | 53.8612 | 0.742 | 4.0 | 3.0 | 50.2 | 4.7 | 0.1 | 1.0000 |  |
| minimax | 8 | down | fp8 | - | 60.7079 | 65.5763 | 0.926 | 5.0 | 4.6 | 62.2 | 0.7 | 0.1 | 1.0000 |  |
| minimax | 8 | gate_up | fp4 | bf16 | 63.5943 | 112.3993 | 0.566 | 5.0 | 2.9 | 63.1 | 3.7 | 0.2 | 1.0000 |  |
| minimax | 8 | gate_up | fp8 | bf16 | 96.9018 | 104.2940 | 0.929 | 6.2 | 5.8 | 77.9 | 0.4 | 1.6 | 1.0000 |  |
| minimax | 32 | down | fp4 | - | 149.4354 | 171.1072 | 0.873 | 4.3 | 3.8 | 53.7 | 0.3 | 0.3 | 1.0000 |  |
| minimax | 32 | down | fp8 | - | 228.1005 | 227.8084 | 1.001 | 5.3 | 5.3 | 66.2 | 1.3 | 0.1 | 1.0000 |  |
| minimax | 32 | gate_up | fp4 | bf16 | 252.1838 | 431.9675 | 0.584 | 5.1 | 3.0 | 63.6 | 0.2 | 0.2 | 1.0000 |  |
| minimax | 32 | gate_up | fp8 | bf16 | 388.4063 | 401.9118 | 0.966 | 6.2 | 6.0 | 77.8 | 0.1 | 0.2 | 1.0000 |  |
| qwen3next | 1 | down | fp4 | - | 5.2493 | 9.7675 | 0.537 | 1.1 | 0.6 | 13.3 | 0.4 | 0.2 | 1.0000 |  |
| qwen3next | 1 | down | fp8 | - | 5.7134 | 10.3767 | 0.551 | 1.8 | 1.0 | 22.9 | 0.5 | 0.2 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp4 | bf16 | 6.0894 | 7.2447 | 0.841 | 1.8 | 1.5 | 22.9 | 0.3 | 0.4 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp8 | bf16 | 7.0219 | 7.8736 | 0.892 | 3.0 | 2.7 | 37.3 | 2.4 | 0.3 | 1.0000 |  |
| qwen3next | 2 | down | fp4 | - | 5.8677 | 11.3534 | 0.517 | 1.9 | 1.0 | 23.7 | 0.5 | 0.0 | 1.0000 |  |
| qwen3next | 2 | down | fp8 | - | 7.1412 | 15.2473 | 0.468 | 2.9 | 1.4 | 36.7 | 4.1 | 0.5 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp4 | bf16 | 8.6099 | 10.9550 | 0.786 | 2.6 | 2.0 | 32.3 | 1.5 | 0.4 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | bf16 | 9.8466 | 12.7422 | 0.773 | 4.3 | 3.3 | 53.2 | 0.1 | 0.3 | 1.0000 |  |
| qwen3next | 4 | down | fp4 | - | 8.4418 | 13.6421 | 0.619 | 2.6 | 1.6 | 33.0 | 1.1 | 0.1 | 1.0000 |  |
| qwen3next | 4 | down | fp8 | - | 10.7434 | 17.7622 | 0.605 | 3.9 | 2.4 | 48.8 | 0.3 | 0.1 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp4 | bf16 | 13.0669 | 17.0222 | 0.768 | 3.4 | 2.6 | 42.6 | 4.3 | 0.6 | 1.0000 |  |
| qwen3next | 4 | gate_up | fp8 | bf16 | 16.1235 | 19.9945 | 0.806 | 5.2 | 4.2 | 65.0 | 0.5 | 0.2 | 1.0000 |  |
| qwen3next | 8 | down | fp4 | - | 12.6836 | 16.1738 | 0.784 | 3.5 | 2.8 | 43.9 | 0.7 | 0.2 | 1.0000 |  |
| qwen3next | 8 | down | fp8 | - | 19.1244 | 23.3570 | 0.819 | 4.4 | 3.6 | 54.8 | 2.7 | 0.2 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp4 | bf16 | 22.7205 | 29.6560 | 0.766 | 3.9 | 3.0 | 49.0 | 1.2 | 0.3 | 1.0000 |  |
| qwen3next | 8 | gate_up | fp8 | bf16 | 28.9758 | 32.8894 | 0.881 | 5.8 | 5.1 | 72.4 | 0.2 | 0.1 | 1.0000 |  |
| qwen3next | 32 | down | fp4 | - | 37.3538 | 55.0783 | 0.678 | 4.8 | 3.2 | 59.7 | 0.6 | 0.3 | 1.0000 |  |
| qwen3next | 32 | down | fp8 | - | 62.1308 | 73.1537 | 0.849 | 5.4 | 4.6 | 67.5 | 0.1 | 0.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp4 | bf16 | 73.4864 | 104.8119 | 0.701 | 4.9 | 3.4 | 60.6 | 3.1 | 0.1 | 1.0000 |  |
| qwen3next | 32 | gate_up | fp8 | bf16 | 103.7575 | 112.8524 | 0.919 | 6.5 | 5.9 | 80.8 | 0.0 | 1.4 | 1.0000 |  |

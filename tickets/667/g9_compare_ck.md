<!-- SILOTIGER-667 G9 FlyDSL-vs-ck cold warp-decode comparison -->
<!-- gfx=gfx950  aiter=832c55702  backend=ck  ck_worktree=c03392a91b8 -->
<!-- iters=1000 cold=20 timing=device method=weight_stream repeats=3 flydsl_weight_layout=preshuffled ck_weight_layout=k_contiguous -->
<!-- ck provenance: # ck_bench_warp_decode  base_commit=62e30c9098 patch=A4-gateup-fp4-packed-stride  cold=20  iters=1000  rotate=auto(ceil(E/BK))  format=csv  mechanism=manual-hipEvent+disjoint-router-rotation -->
<!-- clocks: auto (unpinnable on this gfx950; D1) -- effective loaded sclk MHz min/median/max = 2059/2390/2400 (n=61/63) on GPU 6; per-cell spread%% + noisy flag (>5%) capture drift (D5). -->
<!-- config policy (D3): default-vs-default except the explicit FlyDSL weight layout. FlyDSL weight_layout=preshuffled; other FlyDSL knobs use library defaults: serialize_dot2=True, kh_per_warp=auto(2 when HIDDEN even), prefetch=False; down_fp4 dot2_acc=4, gate_up_fp4 dot2_acc=1 (G7: acc>1 ~4% slower for gate_up); down_fp8 split_k=1; FP8 w_scale=block2d(128,128) to match CK. CK = maintainer-recommended variant per op (down_h2_d2, down_fp4_h2, gate_bf16_d2, gate_up_fp4 non-dot2/NPerWarp=1); CK has no single runtime default (mild asymmetry). FP8-down ratio is a CK-favored lower bound: block2d(128,128) costs FlyDSL ~10-38% vs pertensor (B1); FP4 rows carry a ~6% CK-favored scale-traffic bias (dummy PerTensor vs e8m0(1,32)). Treat under-converged fast cells as noisy (D1). -->

**metric method:** `weight_stream` &nbsp; (ratio = flydsl_us / ck_us; CK is perf-only / uninitialized weights)

| shape | B | op | dtype | act | flydsl_us | ck_us | ratio(f/c) | fly_TB/s | ck_TB/s | fly_%peak | fly_spr% | ck_spr% | cos | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3next | 1 | down | fp4 | - | 9.4737 | 9.5693 | 0.990 | 0.6 | 0.6 | 7.4 | 1.0 | 0.2 | 1.0000 |  |
| qwen3next | 1 | down | fp8 | - | 15.3855 | 11.1027 | 1.386 | 0.7 | 0.9 | 8.5 | 3.7 | 0.3 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp4 | bf16 | 16.1139 | 7.3062 | 2.206 | 0.7 | 1.5 | 8.6 | 0.1 | 0.7 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp8 | bf16 | 22.0691 | 7.7709 | 2.840 | 1.0 | 2.7 | 11.9 | 1.2 | 1.3 | 1.0000 |  |
| qwen3next | 1 | gate_up | fp8 | fp8 | 22.2304 | 8.4727 | 2.624 | 0.9 | 2.5 | 11.8 | 1.2 | 0.9 | 1.0000 |  |
| qwen3next | 2 | down | fp4 | - | 12.0303 | 10.8077 | 1.113 | 0.9 | 1.0 | 11.6 | 3.6 | 0.3 | 1.0000 |  |
| qwen3next | 2 | down | fp8 | - | 15.9282 | 15.0054 | 1.062 | 1.3 | 1.4 | 16.5 | 3.0 | 0.5 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp4 | bf16 | 18.0685 | 10.8219 | 1.670 | 1.2 | 2.1 | 15.4 | 0.3 | 0.1 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | bf16 | 24.8791 | 12.6618 | 1.965 | 1.7 | 3.3 | 21.1 | 1.0 | 0.3 | 1.0000 |  |
| qwen3next | 2 | gate_up | fp8 | fp8 | 23.8774 | 13.1107 | 1.821 | 1.8 | 3.2 | 22.0 | 6.3 | 0.2 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | down | fp4 | - | 26.3271 | 28.2408 | 0.932 | 2.4 | 2.2 | 29.6 | 5.1 | 0.2 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | down | fp8 | - | 37.2139 | 42.5135 | 0.875 | 3.2 | 2.8 | 39.4 | 6.5 | 0.3 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 1 | gate_up | fp4 | bf16 | 62.0699 | 50.1277 | 1.238 | 2.0 | 2.5 | 25.1 | 0.4 | 1.1 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | bf16 | 83.5156 | 48.1331 | 1.735 | 2.8 | 4.9 | 35.2 | 1.3 | 3.5 | 1.0000 |  |
| deepseek-v3 | 1 | gate_up | fp8 | fp8 | 83.2491 | 46.5851 | 1.787 | 2.8 | 5.0 | 35.3 | 0.5 | 0.1 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp4 | - | 46.6954 | 36.2379 | 1.289 | 2.7 | 3.4 | 33.4 | 0.4 | 1.9 | 1.0000 |  |
| deepseek-v3 | 2 | down | fp8 | - | 59.3041 | 62.6325 | 0.947 | 4.0 | 3.8 | 49.5 | 0.8 | 2.2 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp4 | bf16 | 79.8376 | 91.2783 | 0.875 | 3.1 | 2.7 | 39.1 | 2.9 | 0.8 | 1.0000 |  |
| deepseek-v3 | 2 | gate_up | fp8 | bf16 | 106.7472 | 88.1698 | 1.211 | 4.4 | 5.3 | 55.0 | 7.3 | 1.2 | 1.0000 | noisy (>5% spread) |
| deepseek-v3 | 2 | gate_up | fp8 | fp8 | 100.4695 | 89.2016 | 1.126 | 4.7 | 5.3 | 58.4 | 8.5 | 0.4 | 1.0000 | noisy (>5% spread) |

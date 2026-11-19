# Persistent256 with TDM gather scatter — 2026-11-19

The selected GEMM2 configuration keeps TDM gather scatter and enables 256
persistent workers. In this session's two interleaved comparisons, GEMM2
latency decreases by **7.23%**, and native E2E per layer by **3.57%**.
The buffer-store experiment remains slower and opt-in.

Workload: EP4, **16384 tokens/rank = 65536 global**, E384/topk6/H5120/I2304,
a4w4_mxfp4, balanced routing, eight layers, five warmups, 32 iterations.
Both modes keep original gather/preshuffled S1F, P96/C160, affinity3:5,
legacy instruction prefetch and the same DSV4.1 Flash CSV. GEMM2 is
M256/N256/K256, waves2x2, nb4, WPT1, cluster1x4. The only configuration
change between the fresh controls is `AITER_FLYDSL_PERSISTENT_WORKERS2=256`;
`AITER_MOE_EP_BUFFER_STORE=0` in both.

Each stage value averages the four GPUs' HIP graph-event medians. E2E uses a
separate uninstrumented graph measurement divided by eight layers. These are
different timing scopes; summing stage medians does not reconstruct E2E.
The rows below are in actual run order. Source snapshots and exact environment
are retained in each ignored run directory, based on commit `5e22c9bb4` plus
its recorded working diff. Buffer-only changes in that diff are disabled here.

| Configuration / run | GEMM1 us | GEMM2 us | Plan us | Ready us | E2E/layer us |
|---|---:|---:|---:|---:|---:|
| TDM persistent256 / 02 | 543.79 | 542.99 | 37.37 | 11.37 | 1525.12 |
| TDM default grid / 02 | 547.78 | 594.42 | 39.33 | 12.22 | 1595.54 |
| TDM persistent256 / 03 | 545.55 | 554.29 | 36.14 | 12.16 | 1541.87 |
| TDM default grid / 03 | 543.23 | 588.39 | 39.17 | 13.45 | 1585.00 |

| Two-run average | GEMM1 us | GEMM2 us | Plan us | Ready us | E2E/layer us |
|---|---:|---:|---:|---:|---:|
| TDM default grid | 545.50 | 591.41 | 39.25 | 12.84 | 1590.27 |
| TDM persistent256 | 544.67 | 548.64 | 36.75 | 11.77 | 1533.50 |

Latency reduction is `100 * (default - persistent) / default`. GEMM1 is
essentially unchanged (0.15%). Do not attribute plan/ready variation to a
planner optimization: no planner code or configuration changed. Persistent
E2E spans 1525.12–1541.87 us; control spans 1585.00–1595.54 us. Both fresh
pairs favor persistent256. Earlier noninterleaved run01 values are preserved
in RESULTS_EP_BUFFER_20261119.md and excluded from this aggregate; they did
not show an E2E gain. This supports the selected shape in this session, not
all token counts or random routing.

## Why persistent scheduling helps

The observed GEMM2 grid is 256 TGs instead of 32640, with 128 threads/TG and
327680 bytes LDS in both modes. Default scheduling uses receive capacity.
Persistent workers read `tile_map[n_experts - 1]` (live psum extent) each graph
replay, and iterate only its M/N tasks. For balanced routing, 98304 valid rows
produce 384 M256 tiles, each with 20 N256 tiles: 7680 live tasks.
Allocation capacity, scale-buffer strides and routing are unchanged. This
avoids launching capacity-sized empty work; task reuse can also affect caching.
These observations explain a plausible mechanism, not a measured cycle-level
attribution. Worker stride256 remains a multiple of cluster_n4.

Actual selected symbol:

```
mxfp4_tdm_fp4_apre_t256x256x256_w2x2_b4_K2304_e96_cluster1x4_prefetch_wpt1_epscatter_mg2_dsgroup3_fencewait_regpart_planarlds_interleavelds_pw256_et1_wmma2x4_order_prebarrier4_barrier4_ksl0tail_4_ksl0prefix2_ab2_earlyacc
```

## Correctness and reproduction

`ipflegacy_p96_check_src_ep_tdm_persist_flash_01` exited0. All four ranks
passed six changing-input/routing graph replays at16384/rank, including
random/hot experts, empty/disjoint ownership and unequal tails, with
`rtol=atol=0.01`. Dispatch bytes/maps/scales are also checked. The numerical
reference is existing push with persistent TDM, not an independent fp32 oracle;
this does not prove absence of a bug shared by both persistent implementations.
No timeout or boot change occurred. Three prelaunch reviews are in REVIEW.md.

```bash
# Native E2E with the selected original-gather S1F and GEMM2 configuration.
AITER_MOE_EP_BUFFER_STORE=0 AITER_FLYDSL_PERSISTENT_WORKERS2=256 AITER_MOE_S1F_AFFINITY=1 bash run_ds41_flash_best.sh
# Stage events plus separate native E2E; use new labels each time.
BENCH_EP_BUFFER_STORE=0 BENCH_GEMM2_WORKERS=256 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_ep_tdm_persist_NEW
BENCH_EP_BUFFER_STORE=0 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_ep_original_NEW
# Changing-routing correctness.
CHECK_TPR=16384 CHECK_EP_TDM=1 BENCH_EP_BUFFER_STORE=0 BENCH_GEMM2_WORKERS=256 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity check_src_ep_tdm_persist_NEW
```

The script preserves explicit switches. For exact reproduction use a clean
shell without optional source packing/unique pull/linear scatter overrides.
Its idle checks poll every second and use the shared GPU lock when available.
The library defaults and CSV are unchanged; persistent256 is explicit.

Per-GPU medians and symbols: [results_ep_tdm_persist_20261119.json](results_ep_tdm_persist_20261119.json).

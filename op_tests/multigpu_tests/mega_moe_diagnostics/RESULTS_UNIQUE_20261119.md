# Unique pull / row-major scatter results, 2026-11-19

EP4, 16384 tokens/rank = 65536 global; E384/topk6/H5120/I2304. P96/C160,
affinity3:5, external-ready, M64 flags. Both GEMMs use256x256x256,2x2 waves,
4 buffers; GEMM2 retains cluster1x4. Same CSV and native/HIP-event harness.
The baseline is the existing pull on this branch, not origin/main or push.

One successful process run/configuration;32 replays after5 warmups,8 layers.
Table cells average the four GPU medians, each pooling layers and replays.
GEMM1 includes the fused dispatch producer and consumer. Stage timing uses
explicit HIP graph event nodes (with boundary overhead). Native E2E is measured
separately without event instrumentation and divided by8. Do not sum stage medians.

| Routing | Configuration | GEMM1 us | GEMM2 us | Plan us | Ready us | Native E2E/layer us |
|---|---|---:|---:|---:|---:|---:|
| Balanced | Original pull, preshuffled | 538.56 | 579.77 | 37.49 | 12.53 | 1554.78 |
| Balanced | Unique pull, preshuffled generic scatter (superseded) | 1025.95 | 580.23 | 38.02 | 11.82 | 2055.34 |
| Balanced | Unique pull, row-major generic scatter (superseded) | 1038.20 | 586.72 | 36.75 | 11.55 | 2071.80 |
| Balanced | Unique pull, row-major TDM scatter | 894.38 | 580.57 | 36.47 | 11.94 | 1914.92 |
| Random | Original pull, preshuffled | 595.03 | 585.23 | 37.70 | 12.10 | 1625.65 |
| Random | Unique pull, row-major TDM scatter | 1503.27 | 578.20 | 36.50 | 11.32 | 2541.17 |

## Interpretation

- Row-major is implemented end to end: GEMM1 input/output and GEMM2 input all
  omit `_apre` in runtime symbols. This removes the preshuffle constraint on
  local fan-out; it does not by itself guarantee a faster producer.
- Replacing row-major generic payload stores with indexed TDM stores reduces
  balanced GEMM1 from1038.20 to894.38us (13.85% lower), but the original pull
  remains faster at538.56us. The feature remains opt-in.
- Random routing has4.5003 remote routes/token and2.4730 remote ranks/token:
  deduplication reduces requested peer payload/scale bytes by45.05%, compared
  with83.33% for the balanced pattern (4.5 routes vs0.75 ranks). These are
  logical traffic counts from routing, not measured link bandwidth.
- The hybrid additionally scans source route maps, scatters every local route,
  and uses per-route completion RMWs to join independent writers. Source-token
  scheduling also changes when whole expert tiles become ready. Balanced IDs
  repeat an expert every64 source tokens; completing one expert M256 can require
  contributions spread across almost an entire16384-token source shard. The
  original producer explicitly works on destination tiles instead.
- The relative costs of metadata, completion RMWs and consumer readiness were
  not measured separately here. A claim that one of them is the sole cause
  would need new cycle probes. Row-major/TDM fixes local transfer mechanics,
  but this implementation has not demonstrated an overall GEMM1 speedup.

## Correctness and reproducibility

Final TDM-scatter path passed the existing EP4 byte/map/counter/output checks
at257 and16384 tokens/rank, with six changing-input/routing replays each.
Includes random routes, hot/empty experts, disjoint rank ownership, and unequal
non-M16 tails. Both final runs exited0 with unchanged boot IDs:
- `ipflegacy_p96_check_src_unique_tdm_small_02`
- `ipflegacy_p96_check_src_unique_tdm_flash_01`

The first TDM-scatter attempt failed during compilation due to a missing typed
Int32 conversion on a descriptor index; it is excluded from measurements.
No GPU hang occurred during the successful validation/timing runs.

Each run directory holds command/environment/CSV/source snapshots, boot IDs,
idle checks and raw logs; the sibling JSON retains per-GPU medians and symbols.
See `README_dispatch_pull.md` for reproduction commands. GPU runners check for
no KFD PIDs immediately before launch, poll every second and enforce a timeout.

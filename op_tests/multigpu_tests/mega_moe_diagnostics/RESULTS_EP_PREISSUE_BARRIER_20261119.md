# Persistent GEMM2 barrier relocation — 2026-11-19

Move the previous task's closing TG barrier to the next task's first TDM
issue boundary. Keep per-wave output `tensor_wait(0)` at the old location.
The retained implementation uses raw dscnt + signal/wait to avoid draining
next-task metadata loads before input TDM issue.

Same Flash workload and environment as RESULTS_EP_TDM_PERSIST_20261119.md:
EP4,16384 tokens/rank (65536 global),E384,K6,H5120,I2304,balanced,8 layers,
5 warmups,32 iterations. GEMM2 M256/N256/K256,waves2x2,nb4,cluster1x4,
persistent256,TDM gather scatter. Original preshuffled gather S1F,P96/C160,
affinity3:5. GEMM1,CSV,planner,ready and combine code/settings are unchanged.
Control is commit9196b21b1; candidate differs only by this barrier relocation.

## Fresh interleaved results

Stage values are means of four per-GPU HIP-event medians. Native E2E is a
separate uninstrumented eight-layer graph measurement divided by8. The table
is in execution order; two runs each for the control and retained raw variant,
one exploratory regular-barrier run.

| Run | GEMM1 us | GEMM2 us | Plan us | Ready us | E2E/layer us |
|---|---:|---:|---:|---:|---:|
| preissue_flash_01 | 545.39 | 554.94 | 36.39 | 13.25 | 1546.80 |
| endbarrier_control_01 | 546.40 | 566.78 | 38.88 | 13.45 | 1578.11 |
| preissueraw_flash_01 | 542.86 | 546.17 | 37.80 | 12.62 | 1526.91 |
| endbarrier_control_02 | 542.84 | 556.55 | 40.34 | 13.41 | 1577.68 |
| preissueraw_flash_02 | 543.34 | 544.70 | 37.44 | 11.52 | 1525.17 |

| Average | GEMM1 us | GEMM2 us | Plan us | Ready us | E2E/layer us |
|---|---:|---:|---:|---:|---:|
| Original task-end barrier, two runs |544.62|561.66|39.61|13.43|1577.90|
| Moved regular barrier, one run |545.39|554.94|36.39|13.25|1546.80|
| Moved raw barrier, two runs |543.10|545.44|37.62|12.07|1526.04|

Against the fresh control, raw-barrier GEMM2 latency decreases2.89% and E2E
3.29%, using100*(control-candidate)/control. Raw E2E is1526.91/1525.17us;
control is1578.11/1577.68us. Both pairs favor the candidate. Individual GPU
GEMM2 medians vary more than these E2E results; preserve per-GPU JSON when
assessing small effects. GEMM1 changes0.28%; no GEMM1 improvement is claimed.
Planner/ready timings also vary without code changes. The52us E2E delta is
not a direct measurement of time saved by one barrier: stages are instrumented
in a separate pass, and cache/system state can affect other phases.

Prior-session persistent256 E2E averaged1533.50us; it is not pooled here or
used as the fresh control. This is a same-workspace, selected-shape comparison,
not a new origin/main or random-routing comparison.

## Ordering and actual ISA

```
Task i: C -> LDS -> barrier -> TDM gather stores -> tensor_wait(0)
Task i+1: metadata/address setup -> dscnt0 + TG signal/wait -> first input TDM
```

The barrier also runs on the first task. Each wave completes its final TDM
store before kernel exit, so removing the final TG rendezvous does not remove
output completion. `has_work` is TG-uniform; skipped tasks do not touch LDS.
The new barrier precedes optional A-scale TDM as well as A/B prologue issue.
No LDS allocation, strides, task mapping or multicast protocol change.

The regular moved barrier's actual ISA had:

```asm
global_load_b32 v62, v[36:37], off offset:4
s_wait_loadcnt_dscnt 0x0
s_barrier_signal -1
s_barrier_wait -1
; Descriptor setup
tensor_load_to_lds s[100:103], s[56:63]
```

The retained raw variant changes that wait to `s_wait_dscnt 0x0`. LLVM emits
`s_wait_loadcnt 0x0` later, following `s_wait_tensorcnt 0x3` in the GEMM
pipeline. Thus metadata loads overlap initial TDM rather than draining before
issue. Aside from runtime address constants and the symbol name, these are
the changes between the regular/raw candidate ISA dumps. The previous task
still ends with `tensor_store_from_lds` then `s_wait_tensorcnt 0x0`, followed
by loop bookkeeping and next-task metadata, without a task-end TG barrier.
Both variants still have implicit XCNT drains at branches/barriers; no claim
is made that all memory waits were removed.

Target source: `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`.
Final symbol includes `_pw256_` and `_preissuerawbarrier_`. The change applies
only to persistent EP TDM scatter; buffer-store and GEMM1/S1F paths retain
their existing barriers. No new environment toggle is required.

## Validation and reproduction

Both moved variants passed the existing four-GPU16384/rank test, six changing
routing/input graph replays, including empty experts and unequal tails.
Tolerance is rtol=atol0.01 against the existing push/persistent-TDM reference;
this is differential coverage, not an independent fp32 oracle. Dispatch
payload/map checks passed. No timeout, reset or boot change occurred.

Three prelaunch reviews are in REVIEW.md. Full dumps remain in the ignored
`ipflegacy_p96_check_src_ep_tdm_preissue_flash_01/asm` and
`ipflegacy_p96_check_src_ep_tdm_preissueraw_flash_01/asm` directories. Logs
record the source diff and launch environment. Small committed evidence:
[per-GPU results](results_ep_preissue_barrier_20261119.json).

```bash
# Native E2E, final retained source.
AITER_MOE_EP_BUFFER_STORE=0 AITER_FLYDSL_PERSISTENT_WORKERS2=256 AITER_MOE_S1F_AFFINITY=1 bash run_ds41_flash_best.sh
# Explicit HIP stage events plus separately measured E2E.
BENCH_EP_BUFFER_STORE=0 BENCH_GEMM2_WORKERS=256 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity events_ep_tdm_preissueraw_NEW
# Correctness and emitted assembly; use a fresh label.
CHECK_TPR=16384 CHECK_EP_TDM=1 BENCH_EP_BUFFER_STORE=0 BENCH_GEMM2_WORKERS=256 BENCH_DUMP_ASM=1 BENCH_IPF=legacy bash op_tests/multigpu_tests/mega_moe_diagnostics/run.sh affinity check_src_ep_tdm_preissueraw_NEW
```

The old barrier control requires source at9196b21b1. Keep benchmarks serial
and avoid changing source during JIT jobs. Scripts poll GPU idleness every1s.

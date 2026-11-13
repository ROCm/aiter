# WS CSV promotion: measured before/after microseconds

## Scope

At the user's request, promote the graph-latency improvement rather than leave
all WS code as an explicit-only experiment. Exactly one row in
`aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv` changes:

- Key: gfx950,256 CU,M128,N7168,K3584,no bias,BF16 input/output,no scaling,
  no preshuffle.
- Before:64x64x64,7 stages,4x2 compute waves,legacy FT,split-K1.
- After:64x64x128,3 stages,2 loader +2x2 compute waves,prefetch,cumulative
  counter handshake,split-K1.
- CSV kernel name:
  `flydsl_hgemm_ws_abf16_wbf16_bf16_t64x64x128x3_ks1_p2_w2x2_pf1_ch1_gfx950`.
  `solidx=0` is not a legacy tuner ordinal; this FlyDSL row resolves by its
  fully specified kernelName. SplitK remains1.
- Only contiguous, DMA-aligned inputs at the exact shape use WS. Non-exact
  padded-M consumers return the original row, not a different fallback row.
  Non-contiguous/unaligned layouts use the original HGEMM's normalization.
- The other three task rows, all other model CSV rows, and device kernels
  are unchanged in this continuation.

**This is a graph-latency tradeoff, not an across-all-modes improvement.**
The new row improves graph1/graph4 on the initially idle card, but regresses
rotation. CSV `us` is still the measured rotation value, not the lower graph
value; TFLOPS/GB/s are recomputed from that same value. We do not substitute
historical CSV `us` for the measured before baseline.

## Same-device public-API measurements

Before is parent bf6094e's model CSV; after is the new model CSV. Both use
the same integrated dispatcher, with CSV source and lookup caches switched
**outside** the timed region. Neither GEMM implementation nor timed calls are
mocked. Calls are actual `aiter.tuned_gemm.gemm_a16w16_out`; captured graphs
retain their own selected kernels. Target lookup and modified-input numerical
results are checked before timing.

Hardware: MI355X gfx950,256 CU,BDF0000:85:00.0 (HIP5), observed0% graphics
activity and282828 MB free VRAM at selection. No clock changes, forced power
settings or termination of other users. This initial idle observation is not
a guarantee of exclusive ownership throughout the run.

Three fresh processes, seeds2921/2922/2923,10 alternating AB/BA rounds:
- rotation:default `run_perftest` rotating-input profiler mode,10 warmups,
  101 iterations,complete public operation;
- graph1 / graph4:CUDA events,5 warmups/30 replays,one/four complete public
  calls per capture. Graph4 is divided by4. These include graph replay overhead;
  graph1 is not interchangeable with a pure device-kernel timestamp.
- Each table entry is the median of3 per-process medians in microseconds.
  Delta is after minus before:negative means faster.
- Unchanged controls show timing variation only, **not** new optimizations.

| M,N,K | Mode | Before us | After us | Delta us | CSV action |
|:---|:---|---:|---:|---:|:---|
| 128,3584,7168 | rotation | 20.645 | 20.697 | +0.053 | unchanged control |
| 128,7168,3584 | rotation | 19.592 | 20.493 | +0.901 | WS row updated |
| 16384,3584,7168 | rotation | 597.781 | 598.033 | +0.252 | unchanged control |
| 16384,7168,3584 | rotation | 582.041 | 581.533 | -0.507 | unchanged control |
| 128,3584,7168 | graph1 | 33.686 | 33.526 | -0.160 | unchanged control |
| 128,7168,3584 | graph1 | 34.402 | 30.320 | -4.082 | WS row updated |
| 16384,3584,7168 | graph1 | 600.261 | 599.867 | -0.394 | unchanged control |
| 16384,7168,3584 | graph1 | 584.400 | 585.315 | +0.915 | unchanged control |
| 128,3584,7168 | graph4 | 19.218 | 19.297 | +0.079 | unchanged control |
| 128,7168,3584 | graph4 | 19.506 | 18.378 | -1.128 | WS row updated |
| 16384,3584,7168 | graph4 | 586.646 | 586.496 | -0.150 | unchanged control |
| 16384,7168,3584 | graph4 | 567.765 | 568.166 | +0.401 | unchanged control |

The target's graph4 improves in all three initially-idle-card processes:
5.46–6.14%; graph1:11.84–13.02%. Rotation is3.50–4.77% slower. The selected
CSV `us` is20.4931 (rotation), with graph4 latency18.37775us reported separately.

An earlier batch on busy BDF0000:15:00.0 (seeds2901..2903) is retained:
graph4 target before/after48.161/50.327,51.172/48.061,51.838/48.839us;
graph1 regressed in all three. This is why no universal or co-tenancy-independent
speedup is claimed. The very large apparent rotation changes in that batch
are not used as qualification gains. Host evidence: `flydsl_only/ws_csv/`
contains both batches, per-round samples, CSV snapshots and summary.json.

## Validation

- Real CSV lookup must select the finite WS name at exact M128.
- Public API cold graph capture and changed-input replays call WS; M127,
  row-padded A and strided-K A call the original64x64x64x7 policy.
- Existing functional/out, concurrent graph, input/output guards, large-M
  and default down-M128 ownership regression suite seed2913:PASS.
- Strict numerical checks for the new WS row:zero violations at rtol=atol=.01.
  The unchanged default down-M128 BF16 atomic path retains its documented
  strict random-reference limitation; no accuracy repair is claimed for it.
- CPU name roundtrip/rejection, exact-key/padded-key isolation and layout
  fallback tests added. The older all-legacy-name test excludes this finite
  WS catalog entry and the new tests cover it separately.
- CSV diff is exactly one row; unchanged controls have identical kernel names.

Final gates on the committed source set:59 CPU tests/4604 subtests passed,
Black/Ruff/diff checks passed. A fresh normal merged-config load (without
forcing the model CSV path) resolves the new name and `us=20.4931`. Additional
public dispatch/functional/out/graph validation seed2931 passed.
Remote GPU CI completion is not claimed.

# All-WMMA Flash C1 prototype

**Status: implemented and measured; experimental, not accepted for serving.** All learned GEMMs and attention QK/PV now use BF16 WMMA with FP32 accumulation in one layer kernel. No changes were made to the ATOM branch, server, or published image. This is one uncompressed/hash-routed Flash layer with controlled 128-row cache, not the entire model or Pro TP4.

Coverage:
- mHC attention and FFN learned 24×16384 projections: WMMA with FP32 weights split into BF16 high/low components.
- Q-A, KV, Q-B: previously validated WMMA implementation with eight-way split-K for Q-A/KV.
- Grouped O-A and O-B, router, shared expert gate/up/down: WMMA.
- All six routed FP4 experts, gate/up/down: WMMA; selected-expert tiles distributed across CTAs.
- Attention QK and probability×V: WMMA. Eight online-softmax value contributions are batched into matrix rows and combined in the original online order; probabilities use three BF16 components.
- Norm, quantization, routing selection, softmax, residual/mHC stream mixing and reductions remain ordinary vector operations. No learned vector matvec calls remain in the compiled layer.

Packed checkpoint FP8 and FP4 weights stay packed in global memory and are converted to BF16 operands at load time. This does not use native FP4/FP8 WMMA arithmetic; BF16 was selected after the preceding QKV accuracy experiments.

Second-round timings (µs):

| Region | Unfused vector control, summed kernel time | Fused vector | Fused WMMA |
|---|---:|---:|---:|
| Attention input mHC + norm + quant | 14.436 | 16.544 | 18.227 |
| Q/KV projections + norm + RoPE | 27.152 | 28.682 | 23.174 |
| Cache update + attention QK/PV | 11.015 | 11.376 | 35.733 |
| Attention output projections | 52.073 | 53.919 | 61.050 |
| Attention residual + FFN input mHC | 18.904 | 20.009 | 22.032 |
| Routing + shared expert + routed MoE | 155.626 | 160.940 | 230.123 |
| Expert sum + final residual | 2.482 | 2.843 | 3.084 |
| Complete layer | 357.343 | 379.026 | 509.081 |

The all-WMMA full layer is 34.3% slower than the fused vector control. Only QKV improves materially. The first serial-expert implementation was 1989.295 µs; parallel expert tiles and batched attention reduced it to 509.081 µs.

Measurement: same process/GPU, second round only, median of 30 captured-graph replays. Unfused times sum kernel durations per replay. Inputs restored before each arm, outside capture. No serving client, TPOT/TTFT, or native AITER performance claim. Sum of regional times is not the full-layer latency because fusion changes resource allocation and execution/cache context.

Validation and remaining failure:
- Initial token 42 final-output relative RMS versus independent reference: 0.036467%; passes unchanged 0.1% gate.
- All individual region checks have finite outputs; complete-layer and seven-region executions are bitwise identical for recorded buffers.
- 1000 changing-input graph replays, signed/unsigned counter wrap and health checks pass. Trace confirms one kernel per full-layer replay.

| Additional token | Vector output relative RMS (%) | WMMA output relative RMS (%) | WMMA 0.1% gate |
|---|---:|---:|---|
| 0 | 0.072402 | 0.000000 | Pass |
| 1 | 0.000002 | 0.000000 | Pass |
| 7 | 0.000000 | 0.356783 | FAIL |
| 43 | 0.689537 | 0.000018 | Pass |

Token 7 fails for WMMA (0.3568%); token 43 also exposes a pre-existing vector/reference discrepancy (0.6895%). These are independent-reference comparisons, not native ATOM equivalence. We did not relax the threshold. The token-7 difference appears at mHC/residual rounding and is amplified by subsequent quantization; adding a third BF16 weight component to mHC did not fix it. No GSM8K was run.

Resources and tuning:
- Full layer: 349 VGPRs, 32768 bytes compiler-reported shared memory, 227 SGPR spills into VGPR lanes, zero VGPR spills and zero private scratch. HIP reports 2 resident blocks/CU; launch uses 256 CTAs, one per CU, 128 threads each.
- Routed-expert N-tile sweep (second round): BN16 263.053 µs; BN32 233.469 µs; BN64 394.770 µs; BN128 350.484 µs. Retain BN32. Separate sweep timings differ slightly from full benchmark due to context.
- Global-softmax attention was rejected for initial full-layer numerical failure. Large attention tiles were rejected for scratch spills. The online-order batched variant is retained, with the remaining cross-input failure explicitly recorded.
- A third-component mHC precision trial is preserved as rejected; it did not change token 7 output error.

Next work: resolve native-model numerical equivalence at mHC/residual boundaries; use native gfx1250 AITER preshuffled WMMA staging as the performance reference for FP4 experts; reduce unpack/scale/layout-conversion overhead and SGPR pressure; improve attention tile scheduling. The existing QKV-only WMMA result remains a separate validated regional experiment.

Reproduce after confirming GPU0 is idle:
```sh
docker start vigokhal_dsv4_tp4
docker exec -e HIP_VISIBLE_DEVICES=0 -e TRITON_CACHE_DIR=/work/full-layer-c1/triton-cache vigokhal_dsv4_tp4 bash -lc "ulimit -c 0; cd /work/full-layer-c1/all-wmma; python test.py --grid 256"
```
The main test persists diagnostics/timings even when an additional-embedding reference gate fails; acceptance.json is the explicit acceptance record. Copy result files first; fixed filenames are overwritten. The container was stopped after this task.

Source notes: region_kernel.py and full_kernel.py contain the final parallel expert mapping and RBN knob. build.py is the initial serial-expert generator and is retained as history; do not run it over final sources. v3-serial-experts/ and v4-parallel-experts/ preserve complete prior experiment states. setup.py loads checkpoint weights under /data; models are not included in the archive.

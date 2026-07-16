# FP8 FMHA Prefill — Roofline Analysis (gfx942 / MI308)

Kernel: `fmha_fwd_hd128_fp8_causal_qkptph_vph_ps64`
Config: causal=1, varlen, hd=128, fp8, p_scale, q_heads=8, kv_heads=1
GPU: gfx942 (MI308), 80 CUs, 4 SIMD/CU

---

## TL;DR

- **Measured fp8 matrix peak on MI308 = ~458 TFLOPS.**
- **Measured kernel throughput: 269.7 TFLOPS @ 32k, 228.5 @ 8k** (full sweep below).
- **269.7 / 458 = 59%**, matching the hardware counter **`MfmaUtil = 59.5%`**: throughput is **limited by how busy the matrix unit is.**
- The matrix unit is **idle ~40% of the time**, and that idle time is the **softmax phase** — while softmax runs on the vector (VALU) unit, the matrix pipe issues nothing.
- On MI308, **MFMA and VALU cannot overlap** — matrix/vector co-execution is not allowed on a SIMD, so the GEMMs (MFMA) and softmax (VALU) run strictly serially. The MFMA cycles are fixed by the FLOP count, so the only lever to raise throughput is **reducing softmax VALU cycles**.
- **Not HBM-bandwidth-bound:** measured HBM traffic uses **<2% of the 5.2 TB/s peak** (40–98 GB/s) and the fraction *falls* with seqlen. This rules out DRAM-*throughput* saturation. It does **not** by itself rule out memory-*latency* stalls (bytes moved != time waiting on loads); the stronger argument that latency is largely hidden is the cycle accounting in §3/§5, where `MFMA_busy + VALU_busy` closes the wall-clock budget without a separate memory-stall term.

---

## 1. Measured peak (pure MFMA microbenchmark)

A kernel that issues *only* back-to-back `v_mfma_f32_32x32x16_fp8_fp8` (saturating at
4 waves/CU) establishes the dense fp8 ceiling:

- **Dense fp8 32x32x16 peak on MI308 = ~458 TFLOPS.**

---

## 2. Measured busy split of the real FMHA kernel (HW counters)

From `rocprofv3` on the ps64 kernel @ 32k (267 TFLOPS):

| quantity | counter | value |
|---|---|---|
| MFMA cadence | `SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA` | 32.0 cyc |
| MFMA busy cycles | `SQ_VALU_MFMA_BUSY_CYCLES` | 2.16e9 |
| VALU busy cycles | `SQ_INSTS_VALU * 4` | ~1.64e9 |
| **MFMA unit utilization** | `MfmaUtil` | **59.5%** |
| **VALU issue utilization** | `VALUBusy` | **49.3%** |
| SALU | `SALUBusy` | 2.2% |

Interpretation:

- MFMA busy (2.16e9) vs VALU busy (~1.64e9)  =>  **MFMA is ~57% of the compute cycles.**
- **The matrix pipe is idle ~40% of the time**, during softmax.
- `MfmaUtil` and `VALUBusy` are normalized differently (MFMA busy-cycles vs fraction of
  cycles a VALU instruction is issued, which can be stalled/in-flight), so they are not
  additive and their sum carries no overlap information.

---

## 3. The roofline model

The two GEMMs (QK and PV) produce all the useful FLOPs on the matrix pipe.
Softmax runs on the VALU. Matrix/vector co-execution is not allowed on a SIMD, so MFMA and
VALU **cannot overlap** — the matrix pipe is idle while softmax issues VALU, and
the two run strictly serially.

Basic identity (plain text):

```
TFLOPS = peak * MFMA_busy / wall_time_in_MFMA_equivalent_cycles
```

Serial execution (MFMA then softmax, back to back) is therefore the operative model:

```
wall = MFMA_busy + VALU_busy = 2.16e9 + 1.64e9 = 3.80e9
ceiling = 458 * 2.16e9 / 3.80e9 = 458 * 0.568 = ~260 TFLOPS
```

Equivalent, counter-based way to see it:
```
achieved / peak  =  267 / 458  =  0.583   ~=   MfmaUtil (0.595)
```
Throughput = peak x (fraction of time the matrix pipe is busy). Raising MfmaUtil
toward 100% approaches 458. Every +1% MfmaUtil is roughly +4.6 TFLOPS.

---

## 4. Measured sweep and why throughput scales with seqlen

Full perf sweep (causal=1, varlen, hq=8, hk=1, p_scale):

| seqlen | latency | TFLOPS | workgroups / CUs used | HBM BW (% of 5.2 TB/s) |
|---|---|---|---|---|
| 512 | 0.0398 ms | 13.5 | 16 / 80 (20%) | 55.8 GB/s (1.07%) |
| 1024 | 0.0588 ms | 36.6 | 32 / 80 (40%) | 70.4 GB/s (1.35%) |
| 2048 | 0.0969 ms | 88.7 | 64 / 80 (80%) | 87.0 GB/s (1.67%) |
| 4096 | 0.1753 ms | 196.1 | 80 / 80 (100%) | 98.6 GB/s (1.90%) |
| 8192 | 0.6016 ms | 228.5 | 80 / 80 (100%) | 60.5 GB/s (1.16%) |
| 16384 | 2.1414 ms | 256.7 | 80 / 80 (100%) | 49.2 GB/s (0.95%) |
| 27507 | 5.8467 ms | 265.0 | 80 / 80 (100%) | 42.2 GB/s (0.81%) |
| 32768 | 8.1541 ms | 269.7 | 80 / 80 (100%) | 40.3 GB/s (0.78%) |

HBM traffic (`FETCH_SIZE`+`WRITE_SIZE` counters) never exceeds **~2% of the 5.2 TB/s peak** and
*falls* with seqlen (compute is O(seqlen^2), traffic ~O(seqlen)).

Two distinct regimes drive the ramp:

- **seqlen <= 2048 — GPU underutilization.** The kernel launches one workgroup per CU and
  the workgroup count scales with the work, capping at 80 (the CU count). Below 4096 fewer
  than 80 CUs run at all (16 / 32 / 64), so most of the low TFLOPS is simply idle CUs, not
  in-kernel inefficiency.
- **seqlen >= 4096 — all 80 CUs busy.** From here the remaining ramp (196 -> 270) is
  in-kernel overhead amortizing: fixed global loads of Q/K/V, tile-boundary barriers, and
  the heavier exact-softmax **diagonal** tiles (non-diagonal tiles use the cheaper
  **Schraudolph 2^x approximation**) spread over more interior approximate tiles.

Suggestion for the underutilized regime (seqlen <= 2048): use a **smaller Q tile**. The
workgroup count is `(seqlen / Q_tile) * num_q_heads`, capped at 80. With the current
Q tile of 256 that gives 16 / 32 / 64 workgroups at 512 / 1024 / 2048. Halving the Q tile
(256 -> 128) roughly doubles the workgroup count and CUs used (512: 16 -> 32, 1024: 32 ->
64, 2048: 64 -> full 80), turning idle CUs into work. Only worthwhile where CUs are
otherwise idle, since it trades per-tile efficiency: K/V is re-read ~2x more (more Q tiles
each streaming K/V), fixed per-tile overhead (Q load, barriers, softmax bookkeeping,
epilogue) amortizes over half the rows, and less ILP per tile means lower `MfmaUtil` per CU.

---

## 5. Throughput = peak x MfmaUtil (HW-counter reconciliation)

Applying the §2 method (`MfmaUtil` from HW counters) across seqlens shows throughput is set
directly by matrix-pipe occupancy. Cadence is 32.0 cyc/MFMA (HW) at both points.

| seqlen | MfmaUtil | 458 x MfmaUtil | measured |
|---|---|---|---|
| 8k | 50.0% | 229 | 228.5 |
| 32k | 59.5% | 272 | 269.7 |

throughput ~= peak x (fraction of time the matrix pipe is busy). The gap to 458 is the
~40-50% of time the matrix pipe is idle while softmax runs on the VALU.

---

## 6. How to reproduce

All testing scripts and programs referenced here live on branch **`AITERKER-112-ASM`**.

Pure MFMA peak (`mfma_peak.cpp` lives in `/workspace/aiter`):
```
cd /workspace/aiter
hipcc -O3 --offload-arch=gfx942 mfma_peak.cpp -o mfma_peak && ./mfma_peak
```

FMHA HW counters (ps64 kernel):
```
cd /workspace/aiter
ASM_PAGE=64 AITER_ASM_PERF=1 rocprofv3 \
  --pmc SQ_INSTS_MFMA SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_VALU MfmaUtil VALUBusy \
  --kernel-include-regex fmha --output-format csv -d /tmp/rocprof_ps64 \
  -- python -m pytest "op_tests/test_batch_prefill.py::test_batch_prefill_asm_perf[32768]" -s
```

Key derived checks:
```
cadence          = SQ_VALU_MFMA_BUSY_CYCLES / SQ_INSTS_MFMA        (= 32.0)
achieved / peak  = 267 / 458                                        (= 0.58 ~= MfmaUtil)
```

HBM bandwidth (FETCH_SIZE / WRITE_SIZE are in KB; collect one per pass):
```
cd /workspace/aiter
for C in FETCH_SIZE WRITE_SIZE; do
  ASM_PAGE=64 AITER_ASM_PERF=1 rocprofv3 --pmc $C \
    --kernel-include-regex fmha --kernel-iteration-range "1-1" --output-format csv \
    -d /tmp/bw_$C -- python -m pytest \
    "op_tests/test_batch_prefill.py::test_batch_prefill_asm_perf[32768]" -s
done
# BW = (FETCH_SIZE + WRITE_SIZE) * 1024 / (End_Timestamp - Start_Timestamp)
```

---

## Glossary

- **MFMA_busy** — cycles the matrix pipe is busy. `SQ_INSTS_MFMA * 32`.
- **VALU_busy** — cycles the vector unit is busy. `SQ_INSTS_VALU * 4` (wave64 rate).
- **MfmaUtil** — % of active time the matrix pipe is busy (rocprof, normalized).
- **VALUBusy** — % of active time the VALU issues (rocprof, normalized).
- **peak (458)** — measured max fp8 32x32x16 throughput on MI308.
- **cadence (32)** — cycles one 32x32x16 fp8 MFMA occupies the matrix pipe.
- **FETCH_SIZE / WRITE_SIZE** — HBM bytes read / written by a dispatch (rocprof derived, in KB).
- **HBM peak (5.2 TB/s)** — MI308 peak DRAM bandwidth.

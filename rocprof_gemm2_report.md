# ROCProf Compute Analysis Report

## Run Metadata

- commit: (working tree; flydsl stage2 edits — bf16 trunc + reduce default)
- gpu/soc: gfx942 (MI300X), 80 CU
- workload command: `python /tmp/prof_min.py` (token=32768, model_dim=4096, inter_dim=192, expert=193, topk=9, fp8/per_tensor)
- profile path: `/tmp/rpc_gemm2b`
- rocprofv3 counter/ATT path: `/tmp/rpc_gemm2b/out/pmc_1/`
- target kernel: `moe_gemm2_0` (flydsl stage2, `flydsl_moe2_afp8_wfp8_bf16_t32x256x64_reduce_bnt0`)
- kernel regex/filter: `-k moe_gemm2_0`
- date: 2026-06-12

## Hotspot Summary (Top 5, full fused_moe e2e ≈ 9919us)

| kernel | pct | sum_ns | mean_ns | count | notes |
|---|---:|---:|---:|---:|---|
| moe_gemm2_0 (flydsl stage2) | 47% | — | 4688000 | — | target; reduce mode streaming write |
| ck moe_gemm (stage1) | 31% | — | 3115000 | — | CK gemm1 |
| _topk_sum_kernel | 7% | — | 709000 | — | reduce-mode topk collapse |
| data_to_scale (per_tensor) | 6% | — | 562000 | — | global-max scale (per_tensor only) |
| scaled_quant | 5% | — | 498000 | — | apply fp8 scale |

## Target Kernel Diagnosis

### Kernel
- name: `moe_gemm2_0` (reduce t32x256)
- classification: **memory-bound (data-feed / memory-overlap limited)**
- confidence: high

### Key Metrics (rocprof-compute SOL, panel 2)
- MFMA PoP (F8): **21.4%** of peak
- VALU PoP: 4.5% (FLOP) / 8.1% (IOP)
- MFMA Utilization: **21.9%**
- VALU Utilization: 52.6%
- VMEM Utilization: **7.3%**
- IPC: **0.92 / 5.0 (18%)**
- CU Utilization: 100% (Active CUs 80/80)
- Wavefront Occupancy: **2494 / 2560 = 97.4%** (NOT occupancy-bound)
- VGPR / SGPR / LDS bytes: 52 / 112 / 16896
- L2 Hit Rate: 79.2%
- L1/TCP (vL1D) Hit Rate: 58.6%
- HBM BW vs Peak: **~514 GB/s write ≈ 10% of 5.3 TB/s**
- LDS Bank Conflicts/Access: 0.26 (0.8% — negligible)
- waitcnt/barrier signal: inst-wait 35.3% of wave-cycles; ALU busy only 15.9%
- AI(HBM): low (down-proj K=192, low arithmetic intensity)

### Evidence
- Compute-path: MFMA util 21.9%, IPC 0.92 — ALUs mostly idle; not compute-bound.
- Parallelism: occupancy 97.4%, VGPR=52 (4 waves/SIMD headroom) — NOT occupancy/VGPR-bound.
- Memory-path: 90% of HBM traffic is **output write** (2.4GB/dispatch streaming partial = reduce target `[tokens*topk*model_dim]`); reads hit L2 79%. Achieved HBM BW only ~10% of peak.
- LDS/ISA: LDS bank conflicts 0.26/access (negligible — earlier "LDS conflict" hypothesis ruled out). VALU:MFMA ≈ 9.6 but hidden under stalls.

### Pipeline And LDS Triage

| signal | value | interpretation | follow-up |
|---|---:|---|---|
| MFMA efficiency | 21.9% | <60% → data-feed bound | inspect VMEM overlap, not math tiling |
| `SQ_LDS_BANK_CONFLICT`/access | 0.26 | negligible | do NOT touch LDS layout |
| dominant `vmcnt` producer | output write + W2 read | write-dominated (90%) | reduce write traffic / improve overlap |
| occupancy | 97.4% | not the limiter | do NOT chase occupancy |
| achieved HBM BW | ~10% peak | overlap/MLP insufficient | needs asm-depth pipeline (JIT can't) |

## Action Plan (A/B)

### Action 1 (DONE)
- change: bf16 epilogue RNE→truncation (`_cvt_out`, env `FLYDSL_MOE_STAGE2_BF16_TRUNC`)
- expected metric movement: fewer epilogue VALU
- runtime impact: gemm2 5781→5416 (-6.3%) — verified
- validation: `AITER_CONFIG_FMOE=... python /tmp/one.py`

### Action 2 (DONE)
- change: switch default to reduce t32x256 (streaming write, no bf16-atomic CAS)
- expected metric movement: L2 hit 55.6%→79.2%, avoid atomic RMW amplification
- runtime impact: gemm2 5416→4684 (-13.4%); e2e tied (+672us reduce kernel)
- validation: tuned CSV row token=32768 → verified gemm2 4684, cos 1.0

### Action 3 (NOT VIABLE — evidence-closed)
- change: deeper B prefetch / higher occupancy / f32-atomic / async-copy
- expected: none — occupancy 97.4% (not the limit); deep-B raised wait 21.8→26.8%; f32-atomic doubles write bytes; async-copy gfx942 ISA-unsupported (size≤4B)
- conclusion: remaining gap to asm (gemm2 3162) is JIT achieving ~10% vs asm ~14% HBM BW (deeper register pipeline) — not reachable via stage2 codegen knobs.

## Before/After Comparison

| metric | before (atomic t64x128) | after (reduce t32x256) | delta | delta% |
|---|---:|---:|---:|---:|
| e2e(us) | 9915 | 9878 | -37 | -0.4% |
| target kernel mean(us) | 5413 | 4684 | -729 | -13.5% |
| MFMA Utilization | ~ | 21.9% | — | — |
| Occupancy | (VGPR92→2w/SIMD) | 97.4% | up | — |
| L2 Hit Rate | 55.6% | 79.2% | +23.6pp | — |
| HBM write share | 90% | 90% | — | — |
| LDS conflicts/access | low | 0.26 | — | — |
| VGPR count | 92 | 52 | -40 | — |

## Final Recommendation
- keep: reduce t32x256 (+ bf16 truncation). Locked into `hy3_fp8_pertensor_tuned_fmoe.csv` (token=32768).
- reason: best gemm2 (4684) and marginally best e2e; memory-overlap is the true limiter and reduce minimizes write amplification.
- next priority kernel: stage1 ck gemm (3115us, 31%) and `data_to_scale` (562us, per_tensor-specific — per_token would remove it).
- risk notes: closing to asm (3162) requires asm-depth register pipelining for higher achieved HBM BW; not achievable within flydsl stage2 codegen. For guaranteed asm perf, dispatch large-M to asm 1-stage (e2e 8602 vs 9878).

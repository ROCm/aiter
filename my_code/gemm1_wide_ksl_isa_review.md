# GEMM1 hot loop ISA vs. the wide-KSL plan

`gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1`,
built with `AITER_TDM_WIDE_KSL=1`, gfx1250, `waves-per-eu=1,1`.

- ISA: `21_final_isa.s`, steady-state K-tile loop = `.LBB0_27` (L828) .. `s_cbranch_scc1 .LBB0_27` (L1092), 266 lines.
- `vgpr_count: 346`, `vgpr_spill_count: 0`, `sgpr_count: 90`.
- Registers `v256+` are reached through VGPR-MSB indexing
  (MI400 Shader Programming #65, §3.3.2.3 :2249). In the listing a operand prints
  as `v[66:73] /*v[322:329]*/` — **the comment holds the real register number**,
  the bare `v[66:73]` is just the 8-bit encoding. All analysis below uses the
  real numbers.

## Operand roles

`v_wmma_scale_f32_16x16x128_f8f6f4 dst, srcA, srcB, acc, sA, sB` where for this
kernel `srcA` = 8 dwords of **preshuffled FP4 weight (B matrix)** and `srcB` =
16 dwords of **FP8 activation (A matrix)**. Naming below follows the plan
(A = activation, B = weight), not the ISA operand order.

## Loop skeleton as emitted

```
 828  .LBB0_27:                                  <- loop head
 829    s_wait_dscnt 0xc                         <- plan step 5   (12)
 831    ds_load_b128 x16  -> v64..v127           <- plan step 6   (A1, 16 ops)
 849    s_wait_dscnt 0x13
 850    v_wmma  x1
 851    s_wait_dscnt 0x0                         <- plan step 8
 853    v_wmma  x31                              <- plan steps 7+9 (32 total)
 893    s_cbranch_scc0 .LBB0_44
 896    s_wait_tensorcnt 0x2
 900    s_barrier_signal -1
 921    s_barrier_wait   -1
 924    ds_load_b128      x16 -> v0..v63         <- next tile's A0 (plan step 2)
 941    ds_load_b128      x16 -> v256..v319      <- next tile's B0+B1 (steps 1,3)
 950    ds_load_2addr_b32 x8  -> v136..v151      <- next tile's scales (steps 1,3)
 998    tensor_load_to_lds x5 (predicated)       <- plan step 4
1092    s_cbranch_scc1 .LBB0_27
```

The loop is software-pipelined: the loads that the plan lists as steps 1–4 sit at
the **end** of the body (L924–L1089) and feed the *next* iteration, while the
body opens at the partial wait that consumes them.

## Step-by-step check

| plan step | expected | in ISA | OK |
|---|---|---|---|
| 1. `load_b_and_scales(ksl=0)` ≈12 phys DS | 8×b128 + 4×2addr_b32 | L941–948 (8×b128) + part of L950–976 | yes |
| 2. all A0, 16×b128 | 16 | L924–939, → v0..v63 | yes |
| 3. `load_b_and_scales(ksl=1)` ≈12 | 8×b128 + 4×2addr_b32 | L954–961 + rest of L963–976 | yes |
| 4. next K-tile TDM | TENSORcnt, no DScnt | L998/1022/1041/1062/1089 `tensor_load_to_lds` | yes |
| 5. `s_wait_dscnt(12)` | 12 | **L829 `s_wait_dscnt 0xc`** | yes |
| 6. all A1, 16×b128, DScnt 12→28 | 16 | L831–847, → v64..v127 | yes |
| 7. KSL0: 16 WMMA | 16 | L850–868 (16 WMMA, activation = v0..v63) | yes |
| 8. `s_wait_dscnt(0)` | 0 | **L851** | see note |
| 9. KSL1: 16 WMMA | 16 | L869–886 (16 WMMA, activation = v64..v127) | yes |

Physical DS counts match the plan exactly: 12 per `load_b_and_scales`
(8 `ds_load_b128` + 4 `ds_load_2addr_b32`, the backend pairs the 8 logical scale
b32 loads), 16 per A slice, 56 per K256 tile.

Activation split confirms steps 7 and 9 are the two K-slices:

```
WMMA  1..16  activation v0..v63     (A0, resident at loop entry)
WMMA 17..32  activation v64..v127   (A1, loaded at L831 this iteration)
```

## Where the ISA deviates from the plan

### 1. `s_wait_dscnt(0)` lands after the *first* WMMA, not after all 16

Plan: 16 WMMA (step 7) → `s_wait_dscnt(0)` (step 8) → 16 WMMA (step 9).

Emitted:

```
849  s_wait_dscnt 0x13     <- 19, not in the plan
850  v_wmma  #1
851  s_wait_dscnt 0x0      <- step 8 arrives here, after ONE WMMA
853  v_wmma  #2 .. #32
```

So only one WMMA overlaps the outstanding A1 loads; the other 15 of step 7 run
*after* everything has already been waited on. The latency-hiding window the plan
asks for (16 WMMA covering B1/SB1/SA1 + A1) is not what executes — it is 1 WMMA
wide. LLVM inserted the extra `s_wait_dscnt 0x13` and hoisted `0x0` on its own;
the frontend only emits `s_wait_dscnt(12)` and `s_wait_dscnt(0)`.

### 2. TDM issue is predicated and split across five blocks

Plan step 4 is one logical "issue next K-tile TDM". The ISA has five
`tensor_load_to_lds`, each guarded by its own branch (L986–L1089,
`.LBB0_30/31/33/35/37/38/40`), because `issue()` emits one copy per job and the
wave-specialised path predicates each on `wave == j.wave`. Functionally the same
work, but it puts ~100 lines of scalar branching between the compute block and
the loop back-edge.

## Correctness status

This build **fails accuracy**: `rel_l2 = nan`, the run aborts on
`logits_diff gate 0.01`. Same tiles with `AITER_TDM_WIDE_KSL=0` give
`rel_l2 = 2.8725e-03`.

Static checks done on this ISA that came back **clean**, i.e. they do *not*
explain the NaN:

- no VGPR/SGPR spill (`vgpr_spill_count: 0`);
- 346 VGPRs, inside the 1024 wave32 compute limit (:2250);
- the 16 A1 destinations (v64..v127) are read by WMMA 17..32 — not dead, not
  clobbered;
- no WMMA reads a register that a later load in the same iteration overwrites;
- accumulators (v128..v255, v322..v337) do not overlap any load destination;
- `s_wait_dscnt` immediates are 12 and 0 as the plan requires.

The root cause is still unidentified. Deviation 1 changes *scheduling*, not
semantics, so it explains why the wide schedule is not faster but not why it is
wrong.

## Measured (b8-2, `g2_m64_nb3` tiles, alternating runs)

| | gemm1 us | gemm2 us | e2e us | rel_l2 |
|---|---:|---:|---:|---|
| `WIDE_KSL=0` | 204.3 / 203.1 | 190.4 / 190.7 | 993.7 / 996.3 | 2.8725e-03 |
| `WIDE_KSL=1` | 208.9 / 208.7 | 207.1 / 209.2 | 1020.5 / 1024.8 | **nan** |

The WIDE timings are from runs that produce wrong results, so they only bound
the schedule's cost, not its value.

## Reproduce

```bash
# ISA for both schedules, compile only, no kernel launch
for w in 1 0; do
  rm -rf ~/.flydsl/cache/*
  COMPILE_ONLY=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa_cmp/w$w \
  ENABLE_CK=0 AITER_FORCE_GFX1250=1 AITER_TDM_WIDE_KSL=$w \
  AITER_TDM_TILE_M=64 AITER_TDM_TILE_N=256 AITER_TDM_TILE_K=256 AITER_TDM_NUM_BUFFERS=3 \
  python my_code/compile_only_timing.py --child 64x256x256x3
done

# accuracy + timing
AITER_TDM_WIDE_KSL=1 bash my_code/sweep_tdm.sh g2_m64_nb3
```

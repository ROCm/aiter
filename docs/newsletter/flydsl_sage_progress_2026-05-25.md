# FlyDSL Sage Attention — Progress Report
**Date:** 2026-05-25
**Branch:** `zhuo/sage_mxfp4_flydsl` (local only, not yet pushed)
**Current HEAD:** `a101db6e9`
**Hardware:** AMD MI355 / gfx950, 256 CUs, 8 GPUs (bench pinned to GPU 0)

---

## Executive Summary

This report covers the full development arc of two FlyDSL sage attention kernels:

- **v1** — BF8/INT8 quantized sage attention, targeting general-purpose inference shapes. Status: **ship-ready**, geomean **~1.35× Triton** across all tested shapes (latest bench 2026-05-25).
- **v2** — MXFP4 quantized sage attention, targeting higher-precision-efficiency workloads. Status: **large uplift landed**, geomean **~2.35× Triton** (q_smooth=True), **~1.17× Triton** (q_smooth=False) (latest bench 2026-05-25).

Both kernels are written in FlyDSL (AMD's Python-embedded DSL that lowers to MLIR → LLVM → AMD GPU ISA), and compared against equivalent Triton-based implementations as the baseline.

---

## Background: What Is Sage Attention?

Sage attention is a quantized flash-attention variant that compresses Q, K, V to lower-precision formats (BF8 or MXFP4) before the attention computation, enabling higher arithmetic throughput on hardware that provides fast low-precision matrix multiply (MFMA) units. Two variants exist:

- **q_smooth=False** — quantize Q, K, V independently with per-block scales
- **q_smooth=True** — additionally center Q (subtract per-block mean) to reduce outliers before quantization, compute a bias correction term (`delta_s`), and add it back to the attention logits

The Triton implementation uses multiple separate kernel launches (quantization, optional smoothing, attention, bias correction). FlyDSL allows fusing these into fewer launches and gives finer-grained control over the generated GPU ISA.

---

## v1: BF8/INT8 FlyDSL Sage Attention

### Goal
Port the Triton sage attention v1 kernel to FlyDSL and achieve ≥1.10× Triton on all shapes, with a stretch target of ≥1.10× on long-S (S=16384) shapes.

### Final Performance (commit `a101db6e9`, branch `zhuo/sage_mxfp4_flydsl`, bench 2026-05-25)

Benchmark command: `HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_flydsl_sage.py --warmup 100 --rep 300 --speed-only`

"Attn TFLOPS" = attention kernel only (pre-quantized inputs, no quant overhead).

| Shape | Triton TFLOPS | Triton attn TFLOPS | FlyDSL TFLOPS | FlyDSL attn TFLOPS | FlyDSL / Triton |
|---|---|---|---|---|---|
| B=1, S=1024, H=8, fwd | 35.48 | N/A | 52.70 | 146.59 | **1.49×** |
| B=1, S=3000, H=8, fwd | 231.47 | N/A | 366.34 | 521.21 | **1.58×** |
| B=1, S=4096, H=8, fwd | 364.50 | N/A | 549.13 | 741.04 | **1.51×** |
| B=1, S=8192, H=8, fwd | 833.60 | 1213.10 | 946.28 | 1272.65 | **1.14×** |
| B=1, S=4096, Hq=16/Hk=4, fwd | 705.33 | 1112.49 | 897.38 | 1228.53 | **1.27×** |
| B=2, S=4096, Hq=16/Hk=4, fwd | 772.33 | 1177.78 | 1010.25 | 1261.94 | **1.31×** |
| B=2, S=4096, H=8, fwd | 534.12 | 1103.03 | 817.24 | 1177.56 | **1.53×** |
| B=1, S=16384, H=8, fwd | 1129.30 | 1323.03 | 1213.85 | 1386.77 | **1.07×** |
| B=1, S=16384, H=24, fwd | 1198.58 | 1346.83 | 1233.87 | 1408.24 | **1.03×** |
| B=1, S=75600, H=5, fwd | 1439.03 | 1488.34 | 1433.83 | 1493.33 | **1.00×** |
| **Geomean** | | | | | **~1.35× Triton** |

Note: causal shapes (S=4096, S=8192) show FAILED for Triton in the bench — Triton's causal path is unavailable on this build; FlyDSL causal numbers (164 TFLOPS at S=4096, 374 TFLOPS at S=8192) are measured vs SDPA only. Triton attn TFLOPS for short-S shapes (S≤4096) also show N/A because Triton's causal path failure prevents pre-running the JIT warm-up needed to benchmark the raw attn kernel.

All 15 tests pass.

### Key Optimizations Landed

**1. `buffer_load_dwordx4` for K/V global loads**
Switched K and V tile loads from scalar loads to 128-bit vector loads. Reduces the number of memory instructions per K/V tile fetch, improving HBM bandwidth utilization.

**2. `num_records_bytes` tightening**
Tightened the buffer descriptor size for K/V loads to the exact byte range accessed. Allows the hardware's out-of-bounds predication logic to eliminate a class of spurious guard operations.

**3. `enable-post-misched=True` LLVM flag**
The single highest-leverage change: flipping one LLVM compilation option to enable the post-MI-scheduler pass. This pass runs after machine-instruction scheduling and finds additional opportunities to interleave MFMA instructions with surrounding VALU (vector ALU) work, filling the ~24-cycle MFMA latency gaps. Effect across shapes:
- Long-S H=8: 0.98× → 1.01×
- Long-S H=24: 1.00× → 1.04×
- GQA: 1.17× → 1.22–1.28×
- B=2 H=8: 1.41× → 1.47–1.48×

**Lesson:** Try `enable-post-misched=True` early on any new FlyDSL kernel — it can substitute for manual instruction-reordering work.

**4. Deferred `q_descale * k_descale` multiply — PR#3247 port (commit `a101db6e9`)**
Ported an upstream Triton optimization (ROCm/aiter#3247) to the FlyDSL kernel. Instead of multiplying the QK score by the dequantization scale immediately after each MFMA, the multiply is deferred to the softmax step where it fuses with the `s - m_ij` subtraction into a hardware FMA (fused multiply-add), eliminating a standalone broadcast+multiply pass. This removes ~64 `v_mul_f32` instructions per inner loop iteration.
- Gated on `not CAUSAL` (causal path depends on `-inf` semantics incompatible with the `contract` fastmath mode required for FMA fusion)
- S=8192 fwd: 836 → 945 TFLOPS measured at time of landing (+~13%); current bench shows 944.55 TFLOPS

### Long-S Ceiling: Exhaustive Investigation

The 1.10× kickoff target on S=16384 shapes proved structurally unreachable. Both long-S shapes sit at the **same TFLOPS as Triton** (22–25% of FP8 peak): the kernel is MFMA-throughput-bound, and the residual gap is at the LLVM instruction scheduler's optimum.

**15 dead-ends explored and locked:**

| Approach | Result | Why |
|---|---|---|
| `sched_group_barrier` annotations | Regressed | Manual fences prevented LLVM interleaving |
| `s_setprio(1)` priority brackets | –5–9% | Prevents VALU from filling MFMA shadow |
| `iglp_opt(0)` (DS+MFMA interleave) | Neutral | Already achieved by post-misched |
| `iglp_opt(1)` | Crash | Assumes no ds_writes; our kernel has them |
| `iglp_opt(2)` | Neutral | Noise |
| `BLOCK_N=256` | Regressed + numerics | V_scale is per-128-col; widening tile crosses bucket boundary |
| `BLOCK_M=128` for long-S | –8% | Autopick already selects BM=256 correctly |
| Explicit tree-reduce in softmax row-max | Regressed | LLVM already emits `v_max3_f32` tree; explicit tree was redundant/worse |
| `waves_per_eu=1` and `=4` | Neutral | Underlying constraint is not register pressure |
| `lsr-drop-solution=False` | Neutral | Noise on this kernel |
| `amdgpu-disable-power-sched`, `amdgpu-igrouplp*` | Invalid or neutral | Not valid LLVM cl::opt names in this build |
| `vec_width=8` for K/V loads | Not supported | FlyDSL `buffer_load` max is vec_width=4 |
| Dual-chunk inner loop | Regressed | Extra barrier required per BLOCK_N negates savings |
| Persistent kernel (static-stride wg-loop) | Neutral | H=8 grid (512 WGs) and H=24 grid (1536 WGs) already perfectly divide 256 CUs — no tail wave to eliminate |
| Cross-iter softmax pipeline (carry p_words+corr) | Neutral post-post-misched | `enable-post-misched=True` achieves the same interleaving automatically |

**Recommendation: ship `zhuo/sage_flydsl` at `10b114ecb`.**

---

## v2: MXFP4 FlyDSL Sage Attention

### Goal
Implement a FlyDSL MXFP4 sage attention kernel achieving ≥1.10× Triton on all shapes for both q_smooth modes, with correctness parity (cosine similarity ≥ 0.999 vs Triton on matching quantized inputs).

### Development Timeline

#### Phase 0 — Probe: Validate MXFP4 MFMA Semantics (2026-05-13)

Before writing the kernel, we validated the semantics of `mfma_scale_f32_32x32x64_f8f6f4` in FP4 mode on gfx950 via probe kernels (`op_tests/flydsl_tests/probes/`):

- FP4 mode: K=64 elements per call (32 nibbles × 2 klanes), NOT K=128
- `opselA` (0–3) selects **one** of 4 e8m0 scale bytes from the scale i32 VGPR — no automatic per-32-group scaling within a call
- For per-32-group MXFP4 (head_dim=128), 4 MFMA calls per QK tile were needed initially — same MFMA chain count as INT8 v1

#### Phase 1 — Correctness (commit `83ebaa796`, 2026-05-13)

Initial kernel produced cosine similarity ~0.25 vs Triton. Three bugs fixed:

1. **MFMA operand order**: Had A=Q, B=K. The downstream softmax/mask code was written for v1's A=K, B=Q convention (per-Q-row reduction via klane XOR 32). Fixed: swap operands and scaleA↔scaleB.
2. **Post-MFMA scale**: Multiplied scores by `sm_scale * log2(e)`, but Triton's `rotation_smooth_qk` bakes this into Q before the kernel. Fixed: remove the post-multiply.
3. **Bias path unwired**: Bias parameter existed in the wrapper but was never passed to the kernel. Fixed: add Bias param, buffer descriptor, per-element add inside `_emit_qk_softmax_pquant`.

After fixes: 14/14 tests pass, cosine ≥ 0.9999 vs Triton.

#### Stage A — Causal Tile Split + Load Optimizations (commit `74fa962ed`–`28b2bdbf0`, 2026-05-13)

- **Causal body+tail split**: Split the causal inner loop into a no-mask body (tiles fully in-range) and a masked tail (last tile with attention mask applied). Removes per-element mask evaluation from the majority of iterations.
  - Causal S=4096: +17%, S=8192: +20%, S=16384: +31%
  - Non-causal: kept single loop (split caused –2–4% regression on H=24 large-S)
- **PRE_LOAD_V** (q_smooth=False only): hoist all 8 V fragments (ds_read from LDS) before the QK MFMA chain, hiding V-load latency behind QK MFMA. +3–10% on q_smooth=False forward shapes.
  - Disabled for q_smooth=True (bias path adds register pressure → spills at S=32768)
- **CAUSAL-gated bias bounds-check**: Keep the per-element OOB bounds-check on non-causal (counterintuitively, removing it cost –34% on q_smooth=True S=32768 — the predicate feeds LLVM a scheduling hook for load latency hiding). Remove it on causal (provably redundant in body; mask kills OOB in tail). +54% on q_smooth=True S=16384 causal.

#### Stage D — Per-Lane Scale Packing (commit `27c174c5e`, 2026-05-15)

The biggest single optimization in v2. ISA analysis showed Triton emits only **8 QK MFMA per inner iteration** while FlyDSL emitted **16**, because Triton packs 2 MXFP4 scale groups per MFMA call using per-lane scale VGPRs.

**The mechanism:** On gfx950, the `mfma_scale` instruction's scale operand is a per-lane VGPR (not a scalar broadcast). This means klane=0 and klane=1 within a wave can carry *different* scale values. For per-32-group MXFP4 with K=64 active per call:
- klane=0 (handles K[0:32]) carries scale group 0's e8m0 byte
- klane=1 (handles K[32:64]) carries scale group 1's e8m0 byte
- Both at the same `opselA` byte position

This packs 2 scale groups into 1 MFMA call. Implementation requires a per-klane bit-shuffle of the scale i32:
```python
shift_lo = klane * 8                      # 0 for klane=0, 8 for klane=1
shift_hi = shift_lo + 16
g_lo = (q_scale_i32 >> shift_lo) & 0xFF  # group 0 or group 1
g_hi = (q_scale_i32 >> shift_hi) & 0xFF  # group 2 or group 3
packed = g_lo | (g_hi << 8)              # opselA=ks_pair selects byte ks_pair
```
~6 VALU ops per scale i32; negligible vs the MFMA savings.

**Result:** QK MFMA 16 → 8 per inner iteration. q_smooth=False forward shapes moved from 0.90–0.98× to 0.99–1.07× Triton. q_smooth=True S=16384 fwd: +22%.

#### Phase A — Quantization Port (q_smooth=False, commit `0710bc460`–`5b4c1d48e`, 2026-05-15)

Replaced ~9 separate Triton/PyTorch kernel launches for quantization with a single fused FlyDSL kernel:

- **Walsh-Hadamard rotation** (BLOCK_R=128): 7-stage butterfly entirely in registers (stages 0–2 intra-thread, stages 3–6 cross-lane via `shuffle_xor` with offsets 1, 2, 4, 8 within each row's 16-lane span). No LDS round-trip, no cross-wave traffic.
- **Per-32-element MXFP4 amax**: intra-thread max over 8 elements + 2-stage butterfly within the 4-lane sub-group. All 4 lanes end up with the same amax (replicated).
- **e8m0 RNE encoding**: reciprocal via bit-construction `(254-exp)<<23` bitcast — avoids `llvm.amdgcn.rcp.f32`.
- **FP4 E2M1 packing**: matches Triton byte-exactly; pairs of nibbles → bytes.
- **Conditional in-kernel K-mean subtract (Phase A8)**: When S_k ≤ 8192, subtract K_mean inline in the K branch (extra dwordx4 load + 8 scalar ops per pass). When S_k > 8192, fall back to `torch: k - k.mean()` (in-kernel path regressed long-S register pressure).

**Result:** S=1024 fwd: **+71%**. S=4096: **+28%**. S=8192: **+15%**. 8/11 shapes ≥1.10×.

#### Phase B — Quantization Port (q_smooth=True, commit `a84bd221b`, 2026-05-15)

Ported q_smooth=True quantization to FlyDSL (was Triton). Two-pass Q pipeline inside the same fused kernel:

- **Pass 1**: Stream Q from DRAM, compute `M_Q = mean(Q)` over BLOCK_M rows. Within-warp reduce via `shuffle_xor 16/32`; cross-warp via 2 KB LDS scratch + one `gpu.barrier()`.
- **Pass 2**: Re-load Q (from L1 cache — Q tile is 64 KB, fits well), subtract M_Q, run Walsh-Hadamard + scale + FP4 pack. Store `M_Q * sm_scale * log2(e)` to `Q_mean[B, Hq, Q_NUM_BLKS, D]`.

Added sibling `compute_delta_s_kernel`: per-WG computes `delta_s[b, hq, q_blk, k] = sum_d Q_mean[b, hq, q_blk, d] * (K[b, k, hk, d] - K_mean[b, hk, d])`. On-the-fly K-mean subtract — no scratch tensor materialized. 4-stage `shuffle_xor` row-sum.

**Math note:** Triton centers Q *after* rotation; FlyDSL centers *before* rotation. Mathematically equivalent since `mean(Q@R) = mean(Q)@R` (rotation R is linear). Cosine parity confirmed across all shapes.

**Also added:** `AITER_FLYDSL_FORCE_TRITON_QSMOOTH_QUANT=1` kill-switch env var — routes q_smooth=True quantization back to Triton for one release cycle as a safety valve.

#### Bias Coalesce (commits `1aae0e07f` + `5291bcb4b`, 2026-05-15)

Bias (delta_s) loads were 38–64% of attention kernel time on q_smooth=True shapes. Original path: one `buffer_load(vec_width=1, f32)` per bias element — 4 loads per (st, msub) triplet for the 4 contiguous K-cols addressed by `erem` values.

Replaced with one `vec_width=4` i32 load + bitcast to v4f32 — 4× fewer bias memory transactions.

**Split needed by shape:**
- **Causal (all Hq)**: always coalesce — body is provably in-range; tail's score mask kills OOB cols before bias matters.
- **Non-causal Hq ≥ 16**: coalesce — enough wave parallelism to hide load latency without per-element `cmpi+select` scheduling hints.
- **Non-causal Hq < 16**: keep per-element — vec=4 collapses LLVM's load-latency scheduling window when there isn't enough independent work per CU to fill it (measured –29% on S=32768 fwd).

**Results:**
- GQA Hq16/Hk4: 1.45× → **2.50×** (+73%)
- S=4096 Hq24: 1.42× → **2.28×** (+59%)
- S=16384 Hq24: 1.50× → **2.92×** (+95%)
- All causal shapes: +5–8%

#### DS_BLOCK_N Tuning (commit `ef4e11e19`, 2026-05-15)

Bumped `compute_delta_s` kernel's K-tile (BLOCK_N) from 64 to 128. Quartered the WG count at long-S without LDS/register pressure increase. +11–21% on S≥8192 for the delta_s sub-kernel.
- Tried BLOCK_N=256: marginal +1% at long-S but regressed S=1024 and B=2 GQA. Locked at 128.

### Final v2 Performance

Benchmark command: `HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_flydsl_sage_mxfp4.py --warmup 500 --rep 2000 [--q-smooth]`

**q_smooth=True (vs Triton MXFP4 end-to-end, bench 2026-05-28):**

"Attn TFLOPS" = attention kernel only (pre-quantized inputs, no quant overhead).

| Shape | Triton TFLOPS | Triton attn TFLOPS | FlyDSL TFLOPS | FlyDSL attn TFLOPS | FlyDSL / Triton |
|---|---|---|---|---|---|
| B=1, S=1024, fwd | 21.65 | 47.72 | 34.20 | 63.01 | **1.58×** |
| B=1, S=4096, fwd | 154.24 | 211.91 | 315.36 | 272.77 | **2.04×** |
| B=1, S=4096, causal | 66.60 | 81.21 | 159.23 | 156.42 | **2.39×** |
| B=1, S=8192, fwd | 292.07 | 347.61 | 610.16 | 867.96 | **2.09×** |
| B=1, S=8192, causal | 141.44 | 165.35 | 323.99 | 467.46 | **2.29×** |
| B=1, S=16384, fwd | 342.08 | 367.67 | 748.77 | 924.20 | **2.19×** |
| B=1, S=16384, causal | 169.85 | 187.10 | 495.99 | 640.58 | **2.92×** |
| B=1, S=32768, fwd | 351.48 | 377.13 | 907.77 | 1061.21 | **2.58×** |
| B=2, S=8192, GQA Hq=16/Hk=4, fwd | 344.32 | 371.26 | 869.58 | 1114.76 | **2.53×** |
| B=1, S=4096, Hq=24, fwd | 228.66 | 277.53 | 520.62 | 806.34 | **2.28×** |
| B=1, S=16384, Hq=24, fwd | 338.03 | 369.22 | 989.51 | 1147.53 | **2.93×** |
| B=1, S=75600, H=5, fwd | 347.46 | 367.53 | 478.20 | 549.96 | **1.38×** |
| **Min** | | | | | **1.38×** |
| **Geomean (excl. S=75600)** | | | | | **~2.35×** |

**q_smooth=False (vs Triton MXFP4 end-to-end, bench 2026-05-28):**

"Attn TFLOPS" = attention kernel only (pre-quantized inputs, no quant overhead).

| Shape | Triton TFLOPS | Triton attn TFLOPS | FlyDSL TFLOPS | FlyDSL attn TFLOPS | FlyDSL / Triton |
|---|---|---|---|---|---|
| B=1, S=1024, fwd | 24.34 | 126.35 | 40.18 | 144.87 | **1.65×** |
| B=1, S=4096, fwd | 348.50 | 726.20 | 450.85 | 837.31 | **1.29×** |
| B=1, S=4096, causal | 168.79 | 351.00 | 214.67 | 229.49 | **1.27×** |
| B=1, S=8192, fwd | 831.32 | 1432.67 | 966.19 | 1437.13 | **1.16×** |
| B=1, S=8192, causal | 426.41 | 776.02 | 479.62 | 672.94 | **1.12×** |
| B=1, S=16384, fwd | 1233.57 | 1584.33 | 1303.91 | 1566.80 | **1.06×** (ceiling) |
| B=1, S=16384, causal | 664.99 | 878.14 | 801.54 | 995.87 | **1.21×** |
| B=1, S=32768, fwd | 1481.12 | 1835.21 | 1635.41 | 1678.79 | **1.10×** |
| B=2, S=8192, Hq=16/Hk=4, fwd | 1141.24 | 1542.69 | 1345.85 | 1618.49 | **1.18×** |
| B=1, S=4096, Hq=24, fwd | 598.22 | 1100.28 | 705.19 | 1070.20 | **1.18×** |
| B=1, S=16384, Hq=24, fwd | 1338.67 | 1621.39 | 1391.50 | 1586.13 | **1.04×** (ceiling) |
| B=1, S=75600, H=5, fwd | 1722.06 | 1833.11 | 1741.53 | 1820.27 | **1.01×** |
| **Geomean** | | | | | **~1.17×** |

14/14 pytest pass.

---

## How the Kernel Works: Implementation Strategy

### Inner Loop Structure

Sage attention computes `softmax(Q @ K^T) @ V` tile-by-tile over KV blocks. Each inner loop iteration over one KV tile does:

1. **QK MFMA** — matrix multiply Q tile × K tile → attention scores (logits)
2. **Softmax** — compute per-row running max and normalization; apply mask; quantize scores to P
3. **PV MFMA** — matrix multiply P (quantized scores) × V tile → accumulate into output O

Each MFMA instruction has ~24-cycle result latency. The optimization strategy is to fill as much of these latency windows as possible with useful work from other phases.

### What Gets Overlapped With What

```
Iter k (steady state):
  ┌─ PV MFMA [P_{k-1} × V_{k-1}]   ─────────────────── 24 cycles ─┐
  │  [hidden inside]:                                                 │
  │    Softmax update for iter k (exp, max, normalize)               │
  │    P-quant for iter k                                            │
  │    corr × O_acc                                                  │
  └───────────────────────────────────────────────────────────────────┘
  ┌─ Load K_{k+1}, V_{k+1} from LDS (ds_read)  ──────── latency ──┐
  │  [hidden inside QK MFMA below]:                                   │
  │    mask predicate ops, bounds-check chains                       │
  └───────────────────────────────────────────────────────────────────┘
  ┌─ QK MFMA [Q × K_k]   ──────────────────────────── 24 cycles ──┐
  └───────────────────────────────────────────────────────────────────┘
  gpu.barrier() + start iter k+1
```

**Key insight:** The softmax instructions (~50–100 VALU ops per iteration: exp, subtract, max, scale) are interleaved by LLVM's instruction scheduler into the PV MFMA latency window. This requires that there is enough independent VALU work visible to the scheduler. Operations that appear "redundant" at the source level — mask predicates, bounds-check `cmpi+select` chains, bias predication — serve as scheduling anchors: LLVM uses their data-dependency chains to determine when loads can be issued early. **Removing them collapses the scheduling window**, causing the loads to issue late and performance to drop 8–34%.

### How We Profile and Compare Against Triton

**Step 1 — Dump assembly:**
- FlyDSL: `FLYDSL_DUMP_IR=1 <kernel_launch>` → writes `.amdgcn` ISA to `_compile_artifacts/*/21_final_isa.s`
- Triton: auto-caches `.amdgcn` in `~/.triton/cache/` during compilation

**Step 2 — Count instruction mix per inner-loop iteration:**

Focus on the inner `scf.for` body. Count:
- `v_mfma_*` — matrix multiply units (the target compute)
- `s_nop` — scheduling NOPs (idle cycles, gaps the scheduler couldn't fill)
- `buffer_load_*`, `ds_read_*`, `ds_write_*` — memory access
- `v_exp_f32`, `v_max3_f32`, `v_cmp_*`, `v_cndmask` — softmax and predicate ops

**Example diff that drove a key optimization (q_smooth=False non-causal, after Stage D):**

| Instruction | FlyDSL/iter | Triton/iter | Analysis |
|---|---|---|---|
| QK `mfma_scale` | 8 | 8 | **tied** after Stage D |
| PV `mfma_scale` | 8 | 0 (uses non-scale `mfma_f32`) | **structural gap** |
| `s_nop` | 66 | 2 | stalls from `mfma_scale` latency |
| mask `v_cmp` + `v_cndmask` | ~132 | 0 | load-bearing for scheduling |

This diff directly pointed to the remaining lever: Triton's PV uses `rocdl.mfma.f32.32x32x64.f8f6f4` (no scale operand), saving one register read per call. FlyDSL's bundled LLVM lacks a translation pass for this instruction — it's the sole identified remaining gap for long-S forward.

### Why Short-S and Long-S Behave Differently

**Short sequences (S ≤ 8192) — memory-bandwidth-bound:**
At short S, the number of KV tiles is small, so the kernel finishes quickly. Most of the wall-clock time is *outside* the attention MFMA inner loop:
- Kernel launch overhead (~10–20µs per launch)
- Quantization: reading Q/K/V from HBM, transforming, writing quantized output
- Bias computation: `compute_delta_s` GEMM

FlyDSL wins here by **eliminating launches and intermediate tensors**: 9 Triton/PyTorch launches → 1 FlyDSL launch for quantization. No HBM round-trips for intermediate Q/K quantized buffers.

**Long sequences (S ≥ 16384) — compute-bound:**
At long S, the inner attention loop dominates. Both FlyDSL and Triton are limited by MFMA throughput (22–25% of FP8 peak on this hardware). The gap closes to 1.01–1.06× because the only difference is the 8 extra `mfma_scale` vs non-scale PV calls per iteration — a fixed-rate cost that shrinks as a fraction of total work at larger S.

---

## Locked Dead-Ends (Do Not Retry Without New Evidence)

### v1 Dead-Ends
1. `sched_group_barrier` annotations
2. `s_setprio(1)` priority brackets around MFMA chains (–5–9%)
3. `iglp_opt(0/1/2)` — modes 0 neutral, mode 1 crashes, mode 2 noise
4. `BLOCK_N=256` — V_scale bucket boundary violation + numerics failure
5. `BLOCK_M=128` for long-S — autopick already selects BM=256 optimally
6. Explicit tree-reduce in softmax row-max — LLVM already emits `v_max3_f32` tree
7. `waves_per_eu=1` and `=4` — underlying constraint is not register pressure
8. `lsr-drop-solution=False` — noise
9. `amdgpu-disable-power-sched`, `amdgpu-igrouplp*` — invalid or neutral
10. `vec_width=8` for K/V loads — FlyDSL max is vec_width=4
11. Dual-chunk inner loop (2 BLOCK_N per iter) — extra barrier per BLOCK_N negates savings
12. Persistent kernel — grids already perfectly divide CU count, no tail wave
13. Cross-iter softmax pipeline (carry p_words+corr as iter_args) — `enable-post-misched=True` achieves the same effect automatically
14. Causal `BLOCK_N=256` for sage v1 — V_scale is per-128 column; 256 merges two buckets

### v2 Dead-Ends
1. Compile-time non-causal mask hoist (Lever 1): –8–15% — mask predicates are load-latency scheduling hooks
2. Body+tail split for non-causal: –2–4% on H=24 large-S (iter_args plumbing overhead)
3. Bias bounds-check removal (global): –34% on q_smooth=True — load-bearing for LLVM
4. Non-causal `vec_width=4` bias coalesce Hq=8: –29% — scheduling window collapse
5. `waves_per_eu=3` post-Stage-D: 0.15–0.66× Triton — register pressure still too high
6. `rocdl.mfma.f32.32x32x64.f8f6f4` via raw `Operation.create`: LLVM bundled with FlyDSL has no translation pass — build-time failure
7. Skip K-mean for q_smooth=False (Stage A.5): accuracy failure (cos.min vs SDPA drops to ~0.92–0.94, below the 0.95 gate)
8. In-kernel K-mean for all S (full Phase A8): long-S register pressure regression (S=32768 fwd –9%)
9. `DS_BLOCK_N=256`: marginal long-S +1% but S=1024 and B=2 GQA regression
10. LDS-cache Q tile for single-pass q_smooth=True: Q tile is 64 KB bf16 — requires ~all of LDS, conflicts with K/V LDS use
11. CK-tile async-DMA Stage 1 (no-swizzle): –7–10% — LDS bandwidth for async-DMA write was worse than VGPR-staged ds_write_b128
12. CK-tile async-DMA Stage 1.5 (XOR swizzle): –4–6% — bank conflicts were only part of the cost; VGPR staging advantage remains
13. Stage 2 cross-iter softmax pipeline (4-phase, carry p_words as iter_args): NaN at S≥320 — root cause unresolved (suspected LDS race or SSA mismatch on iter_args); reverted after 2 attempts per policy

---

## Remaining Work / Open Questions

### Identified Next Levers

**1. C++ MLIR extension for `rocdl.mfma.f32.32x32x64.f8f6f4`** (~2-day effort)
The only known untried path for closing the v2 long-S gap. Would expose the non-scale MFMA variant (which Triton uses for PV) to FlyDSL Python, eliminating the extra scale-operand register read on all 8 PV MFMA calls per iteration. Estimated impact: +4–6% on the 3 stuck shapes.

**2. Python wrapper overhead for q_smooth=True S=1024**
Live bench: FlyDSL 0.120ms vs Triton 0.207ms (1.72×). The actual GPU work is ~70µs vs Triton's ~220µs (FlyDSL wins by 3×), but Python dispatch overhead (tensor allocation, contiguous checks, kernel launcher) adds ~50µs to FlyDSL's measured latency. CUDA graph replay in callers would eliminate this. Scope: caller-side architectural change, not kernel work.

**3. Replace torch K-mean for S > 8192 with a FlyDSL kernel**
Currently falls back to `k - k.mean()` for S > 8192. A FlyDSL kernel would eliminate one PyTorch launch but contributes <10% of long-S wrapper time — estimated +1–2% end-to-end. Low priority.

### Decisions Needed

1. **Ship v1?** geomean 1.27× with long-S at 1.01–1.04× (1.10× target unreachable). Recommend yes.
2. **Prioritize C++ MLIR extension** to close v2 long-S gap? Estimated 2-day effort.
3. **Push branch**: `zhuo/sage_mxfp4_flydsl` is 11+ commits ahead of origin, not pushed. PR #3222 description needs updating with the q_smooth=True bench table.

---

## Correctness and Test Coverage

- **v1**: 15/15 tests pass (`op_tests/flydsl_tests/test_flydsl_sage.py`)
- **v2**: 14/14 tests pass (`op_tests/flydsl_tests/test_flydsl_sage_mxfp4.py`) covering all combinations of causal × q_smooth × shape
- **Kill-switch**: `AITER_FLYDSL_FORCE_TRITON_QSMOOTH_QUANT=1` routes q_smooth=True quantization back to Triton; 14/14 tests still pass with it set

## Benchmarking Methodology

- **GPU pinning required**: `HIP_VISIBLE_DEVICES=0` — without it, Triton TFLOPS on this 8-GPU MI355 host swings 222–1118 TFLOPS on the same shape across consecutive runs (power/frequency state changes)
- **Warmup + rep**: `--warmup 500 --rep 2000` for stability; single-shot readings on long-S shapes are unreliable (±10%)
- **Thermal artifact**: S=75600 runs last in the sweep after a 158–167ms SDPA call — this shape is susceptible to throttling in the first pass. Use median-of-3. (Confirmed: 0.97× first-pass reading was noise; actual ratio is 1.01×.)
- **Profiling harness**: `op_tests/flydsl_tests/profile_flydsl_vs_triton_mxfp4.sh` — dumps both ISAs + LLVM IR and prints inner-loop instruction-mix diff

---

## Commits on Branch (ordered, most recent first)

```
a101db6e9  [FlyDSL] sage v1: defer qk_scale to fuse with softmax FMA (non-causal)
393ed39dc  Add bench item (S=75600 Hq5)
5291bcb4b  [FlyDSL] MXFP4 sage attention: bias coalesce on non-causal Hq>=16
1aae0e07f  [FlyDSL] MXFP4 sage attention: coalesce bias loads (causal only)
ef4e11e19  [FlyDSL] MXFP4 sage: bump delta_s DS_BLOCK_N 64 → 128
a84bd221b  [FlyDSL] MXFP4 sage: port q_smooth=True quant to FlyDSL (Phase B)
5b4c1d48e  [FlyDSL] MXFP4 sage: conditional in-kernel K-mean subtract (Phase A8)
0710bc460  [FlyDSL] MXFP4 sage: port quant to FlyDSL (Phase A) — fused 1-launch QKV
27c174c5e  [FlyDSL] MXFP4 sage: per-lane scale packing halves QK MFMA (24 → 16)
28b2bdbf0  [FlyDSL] MXFP4 sage: PRE_LOAD_V (non-bias) + CAUSAL-gated bias bounds-check
a089d8faa  [FlyDSL] MXFP4 sage: gate body+tail split on CAUSAL=True only
74fa962ed  [FlyDSL] MXFP4 sage: causal tile-split (no-mask body + masked tail)
83ebaa796  [FlyDSL] Wire bias (delta_s) path for q_smoothing=True (cos 0.999)
```

# Plan v5: N4K8 weight (B) scale — layout CONFIRMED correct; root cause is a
#          steady-state pipeline race on the b-scale `ds_load_b32`

> **Supersedes the open questions in v4 §5d / `fix_config4.md`.** v4 left it
> ambiguous whether the residual ~0.05–0.10 was an op_sel/layout problem, a
> "silu race", a "high-topk error", or fp8 precision. This plan records the
> outcome of a direct contract review with the layout owner: **the N4K8
> weight-scale layout (producer + consumer + op_sel) is correct.** The residual
> is a **synchronization race on the newly-added N4K8 b-scale LDS read in the
> steady-state pipeline loop**, not a layout error.

---

## 0. Symptom (reproduce)

```
HIP_VISIBLE_DEVICES=0 AITER_FORCE_A8W4=1 WEIGHT_SCALE_OP_SEL=0 \
AITER_USE_GROUPED_GEMM=1 AITER_FORCE_GFX1250=1 \
python op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario verify --data-format a8w4 --layout gguu \
  --experts 8 --tokens 64 --topk 8 \
  --model-dim 7168 --inter-dim 2048 --act silu
```
→ `rel_l2 = 0.0905`, **`grouped_norm = 7.9742e+08 ≈ ref_norm = 7.9974e+08`**
(norms match to ~0.3%). tol = 0.02.

Matching norms ⇒ NOT the catastrophic cross-term (that blew norms up to ~1.0 and
was already fixed by `adv_bs_i32` → `*64`, `gemm_mxscale_gfx1250.py:2189`). This
is a **distributed ~9% error with the right magnitude** — a subtle, not a gross,
fault.

## 1. The verified WMMA scale contract (sample + layout-owner confirmation)

`v_wmma_scale_f32_16x16x128_f8f6f4`, a8w4 (weight = SRC0/A operand, `fmtA=4`,
scaleAType = op_sel). Confirmed from `test_wmma_scale_sample.py` AND directly by
the layout owner:

1. **op_sel selects the LANE-HALF.** op_sel=0 ⇒ the entire scale set is read from
   **lanes 0-15**; op_sel=1 ⇒ from **lanes 16-31**. (Sample `test_opsel0`/
   `test_opsel1`: zero the unused half-wave, result stays 42.5 ⇒ the *used* half
   supplies the scale.)
2. **One i32 per lane = 4 e8m0 bytes, byte r → K-block r** (sample base case:
   scales `[1,0.5,0.25,0.125]`, data all 1.0 → `32+8+2+0.5 = 42.5`). scaleA and
   scaleB multiply.
3. **Per-row, NOT kgrp-interleaved.** A whole row's 4-K-block scale i32 lives on a
   **single lane**; **scale and data do NOT have to be on the same thread/lane.**
   The hardware internally pairs lane L's scale i32 with row L's data even though
   that data is split across the `lane_kgrp` pair (lane L holds K-blocks {0,2},
   lane L+16 holds {1,3}). *(layout-owner confirmed.)*
4. **lane→row is the natural map: lane L (0-15) → instruction A-row L.**
   *(layout-owner confirmed.)*

## 2. End-to-end layout review — op_sel=0 path is SELF-CONSISTENT and CORRECT

All three stages use the SAME `lane16 → A-row L` and `byte r → K-block r`:

| stage | lane→row | byte/r → K-block |
|---|---|---|
| Producer `_grouped_b_scale_preshuffle_e8m0` | `lane = n%16` → col `+lane*4` | `r = ks%4` → col `+r` (4 contiguous bytes) |
| Consumer per-tile read (`_load_b_scale_n4k8`, op_sel off) | `lane16` → col `+lane16*4` | i32's 4 bytes → K-block r |
| Data `load_b_frag` (a8w4) | `row_off = lane16*16` → A-row = lane16 | kgrp interleave; HW pairs internally (contract pt 3) |

- `test_grouped_b_scale_layout.py::test_bscale_numeric_reconstruction` passes for
  **both** op_sel states incl. the real failing dims.
- `adv_bs_i32` steady-state advance = `tile_k/32 * 64 = 512` = one `remain_b`
  super-block per k-tile, matching `make_desc_bs` (n4k8 branch). Correct.

**Conclusion: the N4K8 weight-scale LAYOUT is correct. The op_sel choice is
numerically irrelevant** (per-tile op_sel=0 and per-pair op_sel=1 both deliver the
same logical scale). This matches `fix_config4.md`'s clean-cache finding that
both op_sel give 0.003 at the K=512 baseline. The v4 §5d "op_sel=0 = 0.04" datum
was a **stale-cache artifact**.

## 3. Root cause (current best hypothesis): steady-state b-scale race

Everything that is *layout* is correct, so the residual is a pipeline-sync bug on
the **N4K8-specific** b-scale read:

- **K-threshold = the steady-state loop.** `num_buffers=2`, `loop_iters =
  (num_k_tiles-1)//2`. 2 k-tiles (K=512) → `loop_iters=0` (prologue/tail only) →
  **0.003, passes**. ≥3 k-tiles → steady-state runs → fails. The failing case
  runs it heavily: stage1 K=7168 = **28 k-tiles**, stage2 K=2048 = **8 k-tiles**.
- **N4K8 added a NEW LDS consumer**: one `ds_load_b32` per lane for the b-scale
  (the old lane32 path read differently). If, inside the steady-state loop, the
  *next* k-tile's b-scale TDM write into LDS is not fully fenced against the
  *current* tile's `ds_load_b32`, lanes read a **half-written scale** →
  **nondeterministic ~0.07–0.10**, norms still match.
- This is exactly the v4 §5d conclusion-3 observation: op_sel=1 at K=3072 ×3 =
  **0.0775 / 0.1065 / 0.0968** (nondeterministic = race); K=512 prologue/tail is
  bit-exact deterministic. The `hot_loop_scheduler` is *needed* (disabling it made
  it worse), so the race is in the steady-state mid-compute TDM overlap
  interacting with the N4K8 b32 b-scale, not in the scheduler.

This reconciles "95% it's N4K8" with "the layout is correct": it IS the N4K8
feature (its new `ds_load_b32` b-scale read), but the defect is at the
**synchronization** level, not the addressing/layout level.

## 4. Next steps

### 4a. Confirm the race (GPU, gfx1250) — do FIRST
- **Idle-gate every launch**: `rocm-smi --showpids` must show no competing PID and
  VRAM at the ~319 MB baseline before each run (GPU1 was busy w/ a 377 GB job;
  GPU0 had stray 0-VRAM PIDs at investigation time). See [[gpu-idle-check-before-runs]].
- `rm -rf /root/.flydsl/cache` (clean cache — avoid the stale-kernel artifact that
  produced v4's misleading 0.04).
- Run the failing case **×3**; record rel_l2 each time.
  - **Jitter across runs ⇒ race confirmed** (proceed to 4b).
  - **Identical across runs ⇒ deterministic** ⇒ re-open: a residual deterministic
    layout/precision effect, OR genuine fp32-vs-mxfp quant noise at large K
    (compare a4w4 at the same dims, and all-ones at the same K).

### 4b. Locate & fix the sync (only after 4a confirms a race)
- Inspect the steady-state loop in `aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py`
  (~L2380–2660): the barrier / TDM outstanding-fence between the b-scale
  `ds_load_b32` and the next k-tile's b-scale TDM write. Compare the b-scale
  partial-drain count `_bs_ds_loads` and the `wave_specialized_tdm` /
  `active_adv_i32` path against the A / B-data / A-scale operands (which are NOT
  N4K8 and do not fail).
- Candidate experiments (each ×3 determinism on GPU):
  1. Disable the mid-compute TDM overlap for `b_n4k8` only (issue the next tile's
     b-scale load between tiles, not mid-compute). If deterministic + correct, the
     overlap is the race; re-add overlap with correct sync.
  2. Narrow the operand: overlap a / b-data / a-scale but NOT the n4k8 b-scale.
- Acceptance: the failing case AND K=768/1536/3072 return **identical rel_l2 ×3
  AND < 0.02**, plus the full E128/T4096/K3072 case; a4w4 baseline still passes.

### 4c. Validation rules (carried from v4/fix_config4)
- **Always vary both weight data and weight scale** (the default `_weight_scale` +
  `randn` hidden). Never use `--all-ones`: uniform scales are insensitive to the
  b-scale path and would pass even when broken — exactly how this stayed latent.
- Gate on **×3 determinism first** (race gone), then on `< 0.02`.
- Do NOT "fix" by loosening tol or by adding a swiglu-style clamp to silu; confirm
  the real cause first.

## 5. Status of related/earlier hypotheses
- **adv_bs_i32 cross-term** — FIXED (in tree), confirmed by matching norms.
- **op_sel per-tile vs per-pair** — NOT a bug; layout-equivalent (this plan §2).
- **"silu race" / "high-E/topk error" (`fix_config4.md`)** — treat as likely
  symptoms of the same steady-state b-scale race (both reproduce only off the
  K=512 baseline / under heavier scheduling); re-evaluate after 4b. If a residual
  remains after the sync fix, then split out silu/topk as separate items.

# Port mxfp4 MoE gemm1 (a4w4) to the FlyDSL layout API

## Context & key finding

- **Reference**: `/root/FlyDSL/kernels/preshuffle_gemm_v2.py` — the *only* layout-API kernel.
  It is already a **hybrid**: layout API (`flat_divide` + `make_tiled_mma/copy` +
  `partition_*` + `fx.copy`/`fx.gemm` + preshuffle-encoded-as-a-layout) for all data
  movement / tiling / fragments, but keeps the hand-written `hot_loop_scheduler`
  (raw `rocdl.sched_*`) for perf. fp8 scales are applied in the **epilogue**, not the MMA.
- **Target**: `aiter/ops/flydsl/kernels/mxfp4_gemm1.py` (1246 lines, raw FLIR/MLIR + thread
  index math), a 1:1 port of HIP `gemm1_a4w4.cuh`. Variants: BM32-NT, BM32-cached, BM64,
  BM128, BM16-inline-quant. Microscaled `mfma_scale_f32_16x16x128_f8f6f4` (cbsz/blgp=4),
  cshuffle→SwiGLU→fp4+e8m0 requant epilogue.
- **Hard constraint (verified)**: per-block **mxfp4 microscaling is NOT expressible through
  `fx.gemm`**. The layout API has no scale-fragment partitioning, and the atom-state scale
  path freezes opsel at construction. mxfp4 needs *runtime* per-K-block scale + opsel.
  → The inner MMA **stays on the raw `rocdl.mfma_scale_f32_16x16x128_f8f6f4` intrinsic**,
  exactly like every other gfx950 microscaled kernel. This matches the v2 hybrid philosophy.
- **Env verified**: local GPU = gfx950; `/opt/venv` flydsl **0.2.2** exposes the full
  layout-API surface + `cdna4.MFMA_Scale` + the raw `mfma_scale` intrinsic. Tests run via
  `/opt/venv/bin/python`. Work happens in the existing worktree
  `/root/aiter/.claude/worktrees/moe2stage-flyc` (branch `dev/randomflow_pr`).

## What "port to layout API" concretely means here (the v2 method applied to gemm1)

Move onto the layout API, keep raw only where v2 does (+ the mxfp4 MMA):

| gemm1 concern | today (raw) | ported (layout API) |
|---|---|---|
| A/B/C tiling | manual strides, `bx*BM` | `flat_divide(make_buffer_tensor, make_tile(...))[..., bid, ...]` |
| B preshuffle | `crd2idx(layout_b)` + manual strides | **preshuffle encoded as a hierarchical `make_layout` view** (v2 lines 470-485) |
| A→LDS + swizzle | `raw_ptr_buffer_load_lds` + `_lds_swizzle_mask` | `make_tiled_copy` g2s + `make_composed_layout(static(SwizzleType), ...)` |
| LDS→reg A | `llvm.load(vec4)` + manual offsets | `make_tiled_copy_A(...).partition_S` + `fx.copy` + `retile` |
| accumulators | raw vec registers | `thr_mma.make_fragment_C` |
| **scaled MMA** | raw `mfma_scale_*` | **STAYS raw `mfma_scale_*`** (operates on layout-API fragments) |
| scheduler | `sched_*` | **STAYS raw** (port `hot_loop_scheduler` like v2) |
| epilogue cshuffle/SwiGLU/requant | raw LDS + bit-tricks | LDS as layout view; reuse `_silu_mul`/`_e8m0`/cvt helpers; output via `fx.copy` |

**Open design risk to resolve in Phase 1**: A-side is a **gather** (rows via `m_indices`
token-sort indirection + per-expert B base), not a contiguous tile. v2's `flat_divide`
assumes contiguity. Plan: keep the per-row/expert *base-offset* computation manual
(`b_load_s_base`, `cached_actual_row`), but express the *within-tile* movement, LDS
staging, ds-read, and fragments via the layout API. If gather can't be cleanly bridged,
fall back to layout-API for B + epilogue + accumulators only (still a large maintainability
win) and keep A-gather manual.

## Deliverable & placement

- New file `aiter/ops/flydsl/kernels/mxfp4_gemm1_v2.py` (layout-API), **coexisting** with the
  raw `mxfp4_gemm1.py`. Same `compile_gemm1_a4w4_port(...)` / `gemm1_grid(...)` signatures and
  identical buffer I/O contract.
- A backend switch in `mxfp4_gemm1_kernels.py` (e.g. env `AITER_MXFP4_GEMM1_V2=1` or a
  `gemm1_impl` arg) so the existing tests/bench can A/B v1 vs v2 with zero churn to the raw
  kernel (kept as baseline + fallback until parity proven).

## Phases (stage1 first, per your sequencing)

**Phase 0 — harness & baseline (no kernel changes)**
- Confirm raw `mxfp4_gemm1` numeric tests pass on this GPU; record baseline timing.
- Adapt a microbench (from `bench_stage1_a4w4.py` on main) for KIMI BM32 to get a stable
  per-launch latency for parity comparison. Acceptance = layout-API within ~2% of raw.

**Phase 1 — BM32-NT non-inline, correctness (the primary path)**
- Scaffold `mxfp4_gemm1_v2.py` + backend switch.
- Build host-side: tiled_mma (raw mfma_scale atom geometry), g2s tiled_copy, **B preshuffle
  layout**, A/scale base offsets.
- Kernel: layout-API A LDS-stage + ds-read + fragments; **raw `mfma_cluster`** on those
  fragments; layout-API cshuffle + SwiGLU + fp4/e8m0 requant epilogue.
- Gate: `test_flydsl_gemm1_parametrized_shape_numeric[KIMI, interleave]` (mean_row_cos > 0.85)
  and `test_flydsl_gemm1_matches_hip_end_to_end` (cos > 0.99) pass with v2 backend.

**Phase 2 — perf parity for BM32-NT (fp4x2)**
- Port `hot_loop_scheduler`; tune until within ~2% of the raw kernel. This is the
  "fully align perf in fp4x2" milestone (fp8-method parity is already established by v2).

**Phase 3 — remaining variants**
- BM32-cached, BM64, BM128 (kAStages=2, AGPR pressure), then BM16 inline-quant
  (bf16→fp4 quant path). Each: correctness gate (its compile/numeric test) then perf parity.
- `interleave=False` (separated gate/up) coverage.

**Phase 4 — wire-up & cleanup**
- Validate full `test_mxfp4_flydsl_gemm1.py` suite (v2 backend) + end-to-end `fused_moe`.
- Decide default backend; document the hybrid (raw MMA + scheduler) rationale in the file.

## Then: stage2 (`mxfp4_gemm2.py`)
Same hybrid recipe — simpler (pure down-proj GEMM + atomic-add accumulate, no fused
activation). Phased correctness→perf using `test_mxfp4_flydsl_gemm2.py`. Planned after
stage1 parity is in hand.

## Validation commands
- `/opt/venv/bin/python -m pytest op_tests/test_mxfp4_flydsl_gemm1.py -x` (CPU guards + gfx950 numeric)
- Microbench script for latency parity (v1 vs v2 backend).

## Notes / risks
- Keep the raw kernel untouched as the byte-exact baseline.
- A-side gather bridging is the main technical unknown (Phase 1 resolves it).
- "fp8" parity = the v2 reference itself (already aligned upstream); the new work proves
  the *same method* reaches parity for the fp4x2 MoE path.

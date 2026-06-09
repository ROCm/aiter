# Compare Harness: Timing-Path Module Breakdown

This document maps out, module by module, what each of the two implementations
actually executes inside the timed region of
`op_tests/test_flydsl_grouped_gemm_gfx1250_compare.py` — and which source each
module comes from. Use it to reason about what the `compare` latency numbers
include and where the two paths legitimately differ.

## Key clarification: neither path "calls ATOM"

**ATOM is not an invoked library** — it is the *reference template* we followed
for the gluon two-stage MoE structure (gemm1 + swiglu → re-quant → gemm2 +
scatter). In `compare`, the gluon path is `_run_gluon_stages` (in
`test_moe_gemm_a8w4_e2e.py`), which is **written to mirror ATOM's structure but
lives inside the aiter repo** and does not import ATOM. So the "source" column
below is always an aiter-internal file, never ATOM.

---

## A. FlyDSL path (timed region = `_flydsl_thunk` → `_invoke_grouped_fused_moe`)

Entry: `_invoke_grouped_fused_moe` (`test_flydsl_grouped_gemm_gfx1250.py`)
→ `aiter.fused_moe.fused_moe` → `fused_moe_` (dispatch at `fused_moe.py:649`)
→ `_maybe_grouped_gfx1250_a8w4_moe` (`grouped_moe_gfx1250.py`).
Almost entirely FlyDSL's own kernels:

| Module | Implementation source | File:line |
|---|---|---|
| route-map build | FlyDSL kernel `build_route_maps` | `grouped_moe_gfx1250.py:486` → `aiter/ops/flydsl/moe_kernels.py` |
| a1 activation quant (mxfp8) | **triton** `dynamic_mxfp8_quant` | `grouped_moe_gfx1250.py:526-528` → `aiter/ops/triton/quant.py` |
| route-gather (scatter-copy) | FlyDSL kernel | `grouped_moe_gfx1250.py:590` `flydsl_moe_scatter_copy_token` |
| m_tile prefix map | FlyDSL | `grouped_moe_gfx1250.py:523` `_make_m_tile_prefix_map` |
| stage1 GEMM + swiglu | FlyDSL kernel | `grouped_moe_gfx1250.py:675` `stage1(...)`, compiled by `compile_moe_grouped_gemm1_a8w4_masked` (`kernels/moe_grouped_gemm_mxscale_gfx1250.py`) |
| a2 intermediate quant (mxfp8) | **triton** `dynamic_mxfp8_quant` | `grouped_moe_gfx1250.py:756` |
| stage2 GEMM | FlyDSL kernel | `grouped_moe_gfx1250.py:807` `stage2(...)`, `compile_moe_grouped_gemm2_a8w4_masked` |
| gather-reduce (write back to token order) | FlyDSL kernel | `grouped_moe_gfx1250.py:859` `flydsl_moe_gather_reduce` |

> Note: FlyDSL's activation quant uses **triton's `dynamic_mxfp8_quant`** (the
> post-merge version), which is *not* the same function the gluon path uses for
> quant (`downcast_to_mxfp`). See section C/D.

---

## B. Gluon path (timed region = `_gluon_thunk`)

Entry: `_gluon_thunk` (`test_flydsl_grouped_gemm_gfx1250_compare.py:257`)
→ `_run_gluon_stages` (`test_moe_gemm_a8w4_e2e.py:262`).
Structure mirrors ATOM; every kernel is aiter triton/gluon:

| Module | Implementation source | File:line |
|---|---|---|
| routing (official) | **triton** `routing()` | `_compare.py:258` → `aiter/ops/triton/moe/moe_routing/routing.py:255` |
| a1 activation quant (mxfp8) | **triton** `downcast_to_mxfp` | `test_moe_gemm_a8w4_e2e.py:269` → `aiter/ops/triton/moe/quant_moe.py:64` |
| stage1 GEMM + swiglu | **gluon kernel** | `test_moe_gemm_a8w4_e2e.py:272` `moe_gemm_a8w4` → `moe_op_gemm_a8w4.py:436/488` `_moe_gemm_a8w4_decode/prefill_gluon` (`_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py`) |
| a2 intermediate quant (mxfp8) | **triton** `downcast_to_mxfp` | `test_moe_gemm_a8w4_e2e.py:293` |
| stage2 GEMM | **gluon kernel** | `test_moe_gemm_a8w4_e2e.py:296` `moe_gemm_a8w4` (same gluon kernel) |
| scatter-reduce (write back + gammas weighting) | **triton** `reduce_grouped` | `moe_op_gemm_a8w4.py:18,612` `reduce_grouped` (`aiter/ops/triton/moe/reduce.py`) |

---

## C. Side-by-side (same function, different implementation)

| Function | FlyDSL path | Gluon path | Same impl? |
|---|---|---|---|
| routing / topk | `routing()` official (in thunk, `_compare.py:258`) + FlyDSL `build_route_maps` (internal) | `routing()` official (in thunk, same call) | topk same source; route-map build differs (FlyDSL vs gluon gather) |
| activation quant a1/a2 | triton `dynamic_mxfp8_quant` | triton `downcast_to_mxfp` | ❌ **different functions** (both mxfp8 e4m3fn / 448, but different impl) |
| stage1/stage2 GEMM | FlyDSL grouped kernel | gluon kernel (`_gluon_kernels/gfx1250`) | ❌ each its own kernel (this is exactly what we want to compare) |
| swiglu | fused inside FlyDSL kernel | fused inside gluon kernel | ❌ each its own kernel |
| output gather/reduce | FlyDSL `flydsl_moe_gather_reduce` | triton `reduce_grouped` | ❌ different |

---

## D. Two asymmetries worth noting

1. **Activation-quant function differs.** FlyDSL uses `dynamic_mxfp8_quant`;
   gluon uses `downcast_to_mxfp`. Both target mxfp8 / e4m3fn / 448, but the
   kernel implementations (and their performance) may differ. This step is
   timed on both sides, so strictly speaking it is not "the same quant." If you
   want quant to be perfectly fair too, both sides must use the same function.

2. **routing is shared, but route-map build diverges.** The `routing()` (topk)
   call is shared by both paths, but afterwards FlyDSL uses its own
   `build_route_maps` / scatter-copy, while gluon uses the gather/scatter from
   `routing()` plus the final triton `reduce_grouped`. This is an inherent
   difference between the two MoE frameworks and cannot be eliminated — it is
   part of what the end-to-end performance comparison is measuring.

---

## Optional next step

We can **unify the activation quant** so both sides use the same function
(either both `dynamic_mxfp8_quant` or both `downcast_to_mxfp`), making the a1/a2
quant stage perfectly aligned. After that, the remaining difference is purely
the GEMM kernels + the route framework.

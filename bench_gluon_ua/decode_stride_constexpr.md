# gfx950 gluon decode — KV/block-table strides: runtime args vs constexpr (triton 3.8 ToT)

vLLM (and friends) allocate the paged KV cache **once**, so its strides and the block-table stride
are fixed for the life of the server. Declaring them `gl.constexpr` lets the compiler bake the values
in (constant-fold the address arithmetic) instead of carrying them in registers. A/B on triton 3.8.0
(ToT), single-buffer decode (nb=1, 34 KB LDS), right-sized split, 512 MB L2 flush, 30 iters.

## How the constexpr is routed (matches gfx1250)

The 9 strides (`stride_k_cache_{0..3}`, `stride_v_cache_{0..3}`, `block_table_stride`) are `gl.constexpr`
at the kernel entry and are **routed through the `AttentionConfig` aggregate** (wrapped
`self.stride_* = gl.constexpr(stride_*)` in its `__init__`), exactly like
`_gluon_kernels/gfx1250/attention/unified_attention.py`. The KV loaders no longer carry their own
stride fields — they read `self.cfg.stride_*`. (Wrapping is required: a bare `int` fails the
`@aggregate` field type-check, `Expected constexpr … got int`.)

**int64 / large KV (>2 GB cache, `USE_LOAD_BUFFER_OP=False`, global-load path).** With runtime
strides the kernel cast the *strides* to int64. Constexpr strides can't be `.to()`-cast, so instead the
one offset term that can exceed int32 — the block-base `block_id * stride_k_cache_0` — is widened by
casting the *index* (`block_id.to(gl.int64)`) inside the loader; the per-tile offset stays small (int32).
This is the gfx1250 "cast the offset, not the stride" approach.

### Verified on the large-KV path (the inference case)

| test | cache | check | result |
|---|--:|---|--:|
| moderate, global-load path | 1.07 GB | gluon (int64 path) vs Triton | **xc 1.2e-4** ✓ |
| large, int32 block-base **overflows** (2.68e9 > 2³¹) | 5.37 GB | gluon int64 global-load vs **torch** (last seq) | **xc 6.2e-5** ✓ |
| same large cache, buffer-load | 5.37 GB | gluon buffer-load vs torch | 1.7e-2 ✗ (expected) |

The buffer-load path is wrong for >2 GB caches (buffer descriptors can't address them) — but the host
wrapper already selects `USE_LOAD_BUFFER_OP = kv_size <= 2 GB`, so large caches take the int64
global-load path, which is **correct with constexpr strides**.

## Result — perf-neutral, small VGPR saving

Decode GB/s, both with the same right-sized split; `regs` = VGPRs (0 spills either way):

| shape | runtime GB/s | constexpr GB/s | Δ | VGPR runtime→constexpr |
|---|--:|--:|--:|:--:|
| C64 ctx1024 8/1  | 2289 | 2407 | **+5.1%** | 164→156 |
| C16 ctx8192 8/1  | 3400 | 3498 | +2.9% | 164→156 |
| C32 ctx8192 8/1  | 4234 | 4333 | +2.4% | 164→156 |
| C16 ctx1024 8/1  | 857  | 875  | +2.1% | 164→156 |
| C16 ctx1024 64/8 | 3456 | 3490 | +1.0% | 164→156 |
| C64 ctx8192 8/1  | 5462 | 5512 | +0.9% | →156 |
| C128 ctx8192 8/1 | 6115 | 6135 | +0.3% | →156 |
| C128 ctx8192 64/8| 7026 | 7023 | −0.0% | 164→160 |
| C64 ctx8192 64/8 | 6267 | 6202 | −1.0% | →156 |
| C16 ctx8192 64/8 | 5766 | 5685 | −1.4% | →156 |
| C128 ctx1024 64/8| 6625 | 6500 | −1.9% | 164→160 |

**Mean Δ = +0.8%** across the 16-shape grid (range −1.9% … +5.1%).

## Takeaways

- **Perf-neutral** (mean +0.8%, inside run-to-run noise). Decode is **bandwidth-bound**, and the
  occupancy limiter at nb=1 is **LDS (34 KB → 2 wg/CU), not VGPRs** — so freeing registers doesn't buy
  throughput.
- **Consistent ~8-VGPR saving** (164→156; 164→160 for the S=1 non-split path). Real but not on the
  critical resource here. It would matter if a variant were ever VGPR-occupancy-bound.
- The only shapes that nudge up (+2–5%) are **small-batch MQA** (C16/C32/C64 8/1), where per-tile
  address arithmetic is a larger share of a short kernel — constant-folded strides trim it slightly.
- **Compile note:** the runtime-stride version of the *current* kernel does **not** compile on 3.8-ToT
  (`builtin.unrealized_conversion_cast` fails to lower), so a same-snapshot A/B wasn't possible; the
  runtime column above is the nb=1 single-buffer baseline from the pre-constexpr kernel. Net: constexpr
  strides are effectively **free** and here also sidestep that lowering failure.

_Verdict: keep as a clean/robustness win (and compile workaround), not a perf lever. Implemented
gfx1250-style with the constexpr strides on `AttentionConfig` (loaders read `self.cfg.stride_*`);
config-routing is perf-identical to the loader-wrap (regs 156/160, GB/s within noise) and the large-KV
int64 path is verified above._

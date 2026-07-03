# DSA V4 sparse-MLA forward — optimization wrap-up (2026-06-27)

Consolidated findings from the gluon fwd optimization + the Leon (aiter PR3456/3833) comparison.
All numbers measured on **MI355X (gfx950), `yewang_dsa_triton`, Triton 3.7**, same-session, parity-verified.

## Where we landed

- **nat16** — committed best for the **V3.2 form** (D_QK=576, separate rope). 16x16x32 + topk-LDS + krope-b128.
- **nat23_TK32** — best for the **V4 form** (512, rope baked/zero-pad = the inference + V4 training shape):
  rope-skip + exp2-fold + deferred-rescale + Leon's reorder/partial-wait + `load_shared_relaxed` streaming.
  **−4% vs nat16 on V4 inputs; competitive with Leon (beats on small shapes, ~7% behind on the largest).**
- **nat17** — nat16 + exp2-fold (~2%, marginal). Folded into nat23.

## Head-to-head vs Leon's prefill (V4, 512, same inputs)

| config | nat16(576) | nat23_TK32(512) | Leon prefill(512) |
|---|---|---|---|
| H128 tk2048 | 3.527 ms | 3.398 (−4%) | **3.179** |
| H64 tk512 | 0.572 ms | **0.563** (−2%, **beats Leon −8%**) | 0.612 |

**Leon is faster on large shapes (~7%); we beat him on small.** His edge = **BK64 (bigger K tile)
amortizing per-token VALU/SALU** (ATT: his VALU 18.7/tok vs our 38.5, SALU 4 vs 16.8), **enabled
by a deep-pipelined `load_shared_relaxed` stream that doesn't hold big register tiles.**

## The BK64 wall — 4 attempts, non-transferable

We tried to capture his BK64 amortization; **all four regressed at TILE_K=64:**

| attempt | approach | TK64 result |
|---|---|---|
| nat20 | 1-ahead, `.load()` | +26% (WAIT 1672) |
| nat21 | register-staged | +967% (spill 601) |
| nat22 | JIT `load_shared_relaxed` | +20% (WAIT 2792, spill 0) |
| nat23 | Leon's exact skeleton | +47% (spill 57) |

**BK64 is bound to Leon's *specific* kernel** (its buffer layout / per-tile-rescale / unified-mma),
not portable via the skeleton. See the spill analysis below.

## What each side does better

| axis | ours (nat23) | Leon |
|---|---|---|
| exp2 | **folded into scale** (no per-elem mul) | per-element `*LOG2E` |
| rescale | **deferred** (edge on long TOPK) | per-tile |
| operand feed | streaming `load_shared_relaxed` | streaming `load_shared_relaxed` (both) |
| tile | TK32 (TK64 regresses) | **BK64** (amortizes) |
| bpermute | 3 (topk in klora+krope+mma) → 0-ish in V4 (1 gather layout) | 0 |
| gather-addr VALU | present | present (**both need scalar-base gather**) |

## Recommendation

1. **Training fwd:** ship **nat23_TK32** for the V4 form; keep **nat16** for V3.2 (real rope).
2. **Inference prefill:** use **Leon's kernel** (faster on large shapes, already in aiter).
3. **Stop chasing BK64** — 4 measured failures; it's not transferable to our structure.
4. **Contribute the scalar-base gather primitive** (`GLUON_GATHER_PRIMITIVE_ASK.md`) — helps both
   kernels; even Leon's pays the `[D,K]` offset-tensor VALU (`v_mul_lo_u32 ×49`).

## Companion docs
- `LEON_GLUON_ANALYSIS.md` — Leon PR3456/3833 tricks + corrected comparison.
- `GLUON_GATHER_PRIMITIVE_ASK.md` — scalar-base gather request for Leon.
- `SOFTMAX_OVERLAP_DESIGN.md`, `NAT17_DESIGN.md` — earlier design explorations.
- Traces: `../profiling/att_fwd_earlygather/att_{tv2,leon,nat16_mi355,n22}/`.

## Meta-lesson
~6 "from reading the code" claims this thread were corrected by measurement (rope-is-zero,
bpermute-avoided, load_shared_relaxed-as-WAIT-fix, "many-ahead", BK64-via-skeleton). **Measure
before asserting** — every "did you run it?" turned a wrong picture right.

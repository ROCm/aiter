# dsa_fwd_v4_gluon_nat23 — V4-form training fwd (gfx950 / MI355)

nat23 = the V4 (rope baked / zero-pad, D_V=512) training forward. Run at **TILE_K=32,
BLOCK_H=64** (TK64 regresses — see WRAPUP_FWD_OPT.md). Correct (1.95e-3 vs nat16), spill=0.

## What it is (on top of nat16)
- **rope-skip** (single 512 latent; the V4 rope block is a provable zero) — the ~5% vs nat16.
- **exp2-fold** (log2e folded into scale; nat17).
- **deferred rescale** (kept from nat4/nat16; our edge on long TOPK vs Leon's per-tile).
- **Q -> registers** (direct dot-operand buffer_load, frees LDS; nat19).
- **Leon-style streaming**: `load_shared_relaxed` JIT K/V reads; issue KV[t+1] gather AFTER the
  QK mfma (hides under softmax+PV); topk deep-prefetch; **partial `wait_group(1)`, never (0)** in
  the steady loop; prologue primes [topk[2], KV[0]] as two groups.

## Measured (MI355X, V4 512 inputs, parity-verified)
- H128 tk2048: nat16(576) 3.527 | **nat23_TK32 3.398 (-4%)** | Leon 3.179 (nat23 +7% vs Leon)
- H64  tk512 : nat16(576) 0.572 | **nat23_TK32 0.563 (-2%, beats Leon -8%)** | Leon 0.612

**Competitive with Leon's prefill** (faster on small shapes, ~7% behind on largest). The
large-shape gap is Leon's BK64 amortization, which does NOT transfer here (4 attempts, all
regress at TK64). Use Leon's kernel for inference prefill; nat23 for V4 training fwd; nat16 for
V3.2 (real rope). See WRAPUP_FWD_OPT.md + LEON_GLUON_ANALYSIS.md.

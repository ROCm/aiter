# Kernel Dev Journal — `jagged_dense_bmm_broadcast_add` (jdbba)

_A running log of what we've learned developing the AMD (CKTile / FlyDSL) implementation of
Meta's HSTU `jagged_dense_bmm_broadcast_add` operator. Forward-only, BF16, MI300X first._

Sources: Slack threads (`tiger-customer-meta-aigc`, `tiger-kernel-meta`), Sami's design doc
(`jagged_dense_bmm_broadcast_add_sami_plan.md`), Meta `generative_recommenders` Triton kernel.

---

## 1. Why we're doing this

- Meta's HSTU recsys stack uses this op for feature engineering (training + inference).
- On MI300X, Meta's **Triton** kernel is ~20% slower than on H100.
- Meta's internal **CUTLASS** kernel is ~2x faster than Triton — but it is **internal / not
  public**. So we can't copy it; we must build our own and aim to close the Triton→CUTLASS gap.
- Decision: implement in **FlyDSL / CKTile** on AMD. yashagar's first FlyDSL attempt matches
  Triton speed but only hits **20–30% HBM utilization**; Sami estimates **60–80%** is achievable
  (suggested a cktile-V3-style pipeline — note: not yet documented in our KB).

## 2. What the operator is

For each group `b`, over its slice of rows:
```
Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
```
- It's a **grouped GEMM with variable M per group**. Groups share `N` and `K`; each group has
  its own weight `Dense[b]` and its own per-group `Bias[b]` (broadcast across the group's rows).
- **The one fact that rules out reusing any stock kernel:** group row boundaries come from a
  **device-resident `seq_offsets`** array (a `B+1` prefix sum). The host does **not** know each
  `M_b` at launch — so the group→row mapping must be resolved **on the GPU**.

| Tensor | Shape | Layout | Dtype |
|---|---|---|---|
| Jagged (A) | `(L, K)`, `L = Σ M_b` | row-major | BF16 |
| Dense (B)  | `(B, K, N)` | row-major | BF16 |
| Bias       | `(B, N)` broadcast | row-major | BF16/FP32 |
| Out        | `(L, N)` | row-major | BF16 |
| seq_offsets| `(B+1,)` | device | int32/int64 |

## 3. The shared 5-step recipe (CKTile and FlyDSL both follow it)

Mirrors what the Triton kernel does:
1. **Group on a grid axis** — put `b` on a dedicated grid dimension; size the M-axis by
   `max_seq_len`, N-axis by `N`.
2. **Resolve the group on device** — read `seq_offsets[b]` / `[b+1]`, compute `M_b`,
   **early-exit** tiles past the real length (`if i_m >= M_b: return`).
3. **Offset jagged pointers by row base** — `A_ptr += s*stride_A`, `Out_ptr += s*stride_E`.
4. **Select per-group weight + bias** — `B_ptr += b*K*N`, `Bias_ptr += b*N` (bias stride 0 on M).
5. **Accumulate FP32 → add bias → cast BF16.**

Only steps 2–4 are new vs a normal GEMM; all are cheap per-block scalar/pointer ops.
Subtlety: read `seq_offsets` as int32 but do the **row-base multiply in int64** (`L` can exceed
2^31), matching Triton's `.to(tl.int64)`.

## 4. Two implementation avenues

### CKTile (C++, production/tuned path)
- Reuse `UniversalGemmKernel::RunGemm` mainloop + **CShuffle multi-D epilogue** for bias
  (one D-tensor with **`stride_Ds[0]=0`** = broadcast over rows).
- **Approach A (max reuse):** reuse the **persistent** `GroupedGemmKernel` + a ~20-line **device
  prep kernel** that reads `seq_offsets` and fills each group's `group_karg`. The persistent path
  resolves tile→group on device, so no host-side `M_b` needed. Good for the skewed deployment
  distribution (grid-stride loop absorbs imbalance).
- **Approach B (simplest):** a small new `BatchedGemmKernel`-shaped `JaggedDenseBmmKernel` that
  reads `seq_offsets` directly and sizes the M-axis by `max_seq_len`. **Zero tail waste on the
  uniform benchmark.** Start here.
- **Do NOT** use the *non-persistent* `GroupedGemmKernel` — it needs host-side `M_b`.

### FlyDSL (Python, fast prototype)
- Fork the **generic plain BF16 `hgemm`** (`splitk_hgemm.py::compile_hgemm_kernel`, `TILE_M=128`,
  `SPLIT_K=1`) — **no B preshuffle needed**. `small_m_hgemm` is for `M≤16`, which does **not**
  apply (our `M_i≈7800`).
- Add: group axis on grid, device prologue reading `seq_offsets` (copy the `chunk_gated_delta_h`
  varlen pattern), per-group pointer/bias offsets, extend `HAS_BIAS` to per-group bias.
- **B-layout wrinkle:** plain HGEMM stores B as `(N,K)` and computes `A @ B^T`; our `Dense` is
  `(K,N)`. Fix = **pre-transpose weights once** to `(B, N, K)` (cheap, weights are static), or
  fork the B-load indexing. Prefer the pre-transpose.

**Order of work:** prototype in **FlyDSL first** (fastest to correct, easy A/B vs Triton), then
build the **CKTile** kernel for production. Both share the §3 recipe, so FlyDSL de-risks CKTile.

## 5. Shape facts that drive every decision

Bench shapes (uniform `M_i`, `max_seq_len=16384`; for headline numbers set `M_i` to a tile
multiple near the deployment mean `L/B≈7800`, e.g. `7680`):

| Shape | B | D (=reduction K) | K (=output N) | regime |
|---|---|---|---|---|
| B1024_D256_K256_N16384 | 1024 | 256 | 256 | train, small |
| B1024_D512_K512_N16384 | 1024 | 512 | 512 | train, large |
| B120_D256_K256_N16384  | 120  | 256 | 256 | inference, small |
| B120_D512_K512_N16384  | 120  | 512 | 512 | inference, large |

- **Many tiles** (`M_i≈7800` × many groups → 14k–125k tiles) → no occupancy problem → **split-K
  unnecessary, keep `k_batch=1`**.
- **Tiny reduction `D=256/512`** → K-loop is only **4–8 steps** → per-block setup/epilogue is a
  *large* fraction of runtime → **amortize fixed costs** (see §6).
- `Dense[b]` is 128KB–512KB, reused across all ~61 M-tiles of its group → **L2-resident**, not an
  HBM bottleneck. Kernel is **memory-bound on A/Out streaming**.
- ⚠️ **Naming clash:** bench `(B, D, K, N)` ≠ standard GEMM. Bench **D** = reduction K, bench
  **K** = output N, bench **N** = max_seq_len. Sami's doc uses *standard* GEMM meaning.

## 6. How to make it fast (amortize fixed costs — the short-K regime)

The premise (overhead-bound, not flop-bound) is correct. Levers:
1. **Bigger output tile / cover whole N** — `NPerBlock = N`, larger `MPerBlock`. Fewer, fatter
   tiles → fewer fills + epilogues.
2. **B-stationary (the big one)** — load `Dense[b]` once, run several M-tiles back-to-back so the
   pipeline never drains. (FlyDSL: port `small_m_hgemm`'s `PERSISTENT_N_TILES` to the **M** axis +
   `B_TO_LDS=True`. CKTile: persistent block + a contiguous run of M-tiles within a group.)
3. **Fuse epilogue + resident bias** — fold any adjacent elementwise op into the CShuffle
   round-trip; keep `(B,N)` bias resident across tiles. For small N, try a direct/register
   epilogue (skip LDS shuffle).
4. **Share A across weights (highest leverage IF it applies)** — if the graph projects the same
   jagged input by several weights (Q/K/V, gate/up), load A once, run vs B₁,B₂,… **But the model
   graph does NOT expose this for jdbba** (see §7) — so likely not available here.

What does **not** help: lengthening reduction (D fixed), split-K (opposite of amortization),
huge BLOCK_K (kills the steady state that hides fill latency).

⚠️ **Honest ceiling:** these recover the overhead fraction (~20–40% with a 4–8-step loop), **not**
a multiplier on peak. **Profile before investing** (rocprofv3: MFMA-active vs total cycles).

## 7. Open questions — and their current answers from Slack

Sami's doc §10 lists open questions; most are now **answered** in the Slack thread:

| Question | Answer (from Mikael/Sami/Nico) |
|---|---|
| Bias modes — elementwise needed for v1? | **No.** Broadcast `(B,N)` only. Elementwise is a different HSTU op. |
| Backward in scope? | **Fwd first**, bwd later. Bwd = 3 GEMMs (`d_jagged`, `d_dense`, `d_bias`); Triton parity kernels exist upstream. Nico: Meta may not even backprop through it. |
| `seq_offsets` dtype? | **int64** in their path (from `fbgemm.asynchronous_complete_cumsum`); int32 also fine — kernel doesn't constrain. |
| Weight layout — can Dense be `(B,N,K)` pretransposed? | **Kernel team's choice.** Upstream produces a flat tensor they reshape, so either order costs the same. Pretransposed may coalesce the K-reduction better. |
| Target arch? | **gfx942 / MI300X primary** (deployment HW: chi-mi300x-017). gfx950/MI325X secondary. |
| Multi-weight A-sharing (§8.5)? | **Not available.** In `ContextualizedMLP`, the Dense weight is freshly computed per step (not static), and the jagged input is NOT multiplied by multiple weights here. No post-bmm activation either. So the §8.5 fusion is **not motivated** for this op. |

Still genuinely open: which avenue ships to production (FlyDSL→CKTile recommended).

## 8. Validation plan

- **Reference:** PyTorch eager `for b: Out[s:e] = Jagged[s:e] @ Dense[b] + Bias[b]`, plus the
  Triton kernel on identical inputs. BF16 w/ FP32 accum. Check **mean signed error / cosine**,
  not just `allclose`, to catch systematic bias error.
- **Shapes:** vary B/N/K and the `M_b` distribution — include **empty groups** (`M_b=0`),
  **skewed** (one long + many short), `M_b` not a multiple of tile size, `max_seq_len ≫ mean`.
- **Bias modes:** broadcast primary, elementwise secondary.
- **seq_offsets dtype:** test int32 and int64; verify int64 row-base math for large L.
- **Build:** CKTile via `dev-gfx950`/`dev-gfx942` Ninja; aiter/FlyDSL via JIT (`AITER_REBUILD=1`
  after C++ changes). Run inside the docker container (torch/triton/aiter not on bare host).

## 9. References & tracking

- **Triton kernel:** `generative_recommenders/ops/triton/triton_jagged.py` —
  `jagged_dense_bmm_broadcast_add_kernel` (line ~281); wrapper at `ops/jagged_tensors.py`.
- **Baselines:** Meta repo `meta-recsys/generative-recommenders`; NVIDIA `NVIDIA/recsys-examples`
  (has bwd + bench recipe). Mikael's bench work: internal `AMD-AGI/mvonstra-amd` under
  `recsys-kernels`.
- **Jira:** `SILOTIGER-546` (HSTU attention via dispatcher) + a ticket for jdbba.
- **Related effort:** HSTU Attention also being benchmarked (Mohsen) — CK currently behind Triton.
- **CKTile building blocks:** `rocm-libraries/.../ck_tile/ops/gemm/kernel/`
  (`universal_gemm_kernel.hpp`, `batched_gemm_kernel.hpp`, `grouped_gemm_kernel.hpp`),
  `ops/epilogue/cshuffle_epilogue.hpp`. Example: `example/ck_tile/17_grouped_gemm/`.
- **FlyDSL building blocks:** `FlyDSL/kernels/hgemm_splitk.py`, aiter
  `ops/flydsl/kernels/{splitk_hgemm,small_m_hgemm}.py`, varlen pattern in `chunk_gated_delta_h.py`.

---

## Worklog

- **2026-06-05** — Journal created. Reviewed Slack threads + Sami's design doc. Status: design
  phase complete, no implementation yet. Next: FlyDSL prototype forking generic `hgemm`.
- **2026-06-10** — Established the **official gfx942 (MI300X) baseline** and made the dispatch
  JSON **arch-keyed**. The shipped `jagged_dense_bmm_dispatch_v2.json` was gfx950-only and forced
  `use_mfma_k32=true`; on gfx942 that core-dumps (`Cannot select intrinsic
  llvm.amdgcn.mfma.f32.16x16x32.bf16` — no 16x16x32 bf16 atom on CDNA3). Restructured the JSON to
  `arch-keyed-v1` (`by_arch.gfx942` / `by_arch.gfx950`); the loader now auto-selects the section
  matching `flydsl.runtime.device.get_rocm_arch()` (overridable via `FLYDSL_JAGGED_DENSE_BMM_ARCH`),
  and still accepts a flat legacy JSON via the `..._DISPATCH_V2_JSON` env override. The committed
  `by_arch.gfx942` section is the baseline: empty `winners` → D-bucketed heuristic with
  `use_mfma_k32=false`, `xcd_c/xcd_w` null. Out-of-box bench passes correctness with no core-dump;
  the 2× target is the remaining gap vs Triton (record measured numbers in a results table, not
  here). Also hit a stale-build blocker: `import aiter` failing on
  `MxScaleRoundMode` needs a full `module_aiter_core` rebuild (`rm -rf` the build dir; `AITER_REBUILD=1`
  reuses ccache and does not fix it). Run env is the `jdbba-flydsl` container, aiter at
  `/workspaces/meta/aiter`. Next: re-tune per-shape gfx942 winners (BLOCK_K, tiles, XCD remap, warps)
  via the `/jdbba-autoresearch` loop.
- **2026-06-10 (loop session)** — Ran the autoresearch loop on gfx942. **Bound check:** achieved
  HBM BW is only 25-32% of MI300X peak (~5.3 TB/s) → kernel is **overhead/latency-bound**, not
  HBM-bound, so amortization + occupancy levers have the headroom, not bandwidth tricks.
  Levers tried (cold-L2 do_bench, both regimes, cos=1.0 gate):
  - **#6 XCD remap (tier-A live-knob sweep, `tune_jdbba_xcd.py`)** → KEPT. Swept xcd_c×xcd_w;
    per-shape winners beat the kernel auto-default by 0.9-2.6% (uniform only; skew has remap off).
    Promoted to `by_arch.gfx942.winners`. Marginal size confirms the kernel is *not* L2/chiplet-bound.
  - **BLOCK_K=64 for all K (was 128 for K≤256)** → KEPT, the session's big win. The K≤256→BLOCK_K=128
    special case made A-staging LDS = 64KB, saturating gfx942's 64KB ceiling and halving occupancy.
    BLOCK_K=64 → 32KB → doubles occupancy: **+11-19% on D256** (uniform B120 1.11×, B1024 1.14×;
    skew B120 1.19×), cos=1.0. The old "128 wins ~4%" note was a **gfx950** result (bigger LDS hides
    the occupancy cost). **This is the central gfx942 lesson: occupancy is the binding constraint.**
  - **Tier-B grow-levers — all DISCARDED** (fan-out, 4 isolated worktrees): BLOCK_M=256 (crashes D256
    at 128KB LDS, D512 −39% occupancy-killed at the 64KB ceiling), BLOCK_N=256 (cos=1.0 but −8-15%
    uniform / up to −7.7× skew, 64KB C-tile collapses occupancy), STAGES_A=3 (crashes D256 on old
    BLOCK_K=128; **re-tested clean on the new BLOCK_K=64 baseline → still −24-29%**, deeper pipeline
    costs occupancy with no latency to hide on the 4-8-step K-loop), waves_per_eu={2,3,4} (parity at
    best; wpe=3 spills 2.5× on B1024_D256). Every LDS-*growing* change loses; the only tile/pipeline
    win was *shrinking* LDS (BLOCK_K=64). gfx942's 64KB LDS (2.5× smaller than gfx950) is the wall.
  - **Standing vs Triton (uniform, post-levers):** D256 now ~1.28-1.29×, D512 ~1.28-1.31×. Skew
    1.13-1.21×. The 2× target needs a *structural* amortization lever (#12 B-stationary multi-M-tile:
    load Dense[b] once, run several M-tiles to keep the pipeline warm) — the remaining big idea, since
    every knob-level lever is now exhausted or occupancy-capped.
- **2026-06-10 (loop session, batch 2 + profiling + structural)** — Continued the loop.
  - **Profiled B1024_D512 (rocprofv3 PMC):** OccupancyPercent=**23.5%**, MeanOccupancyPerActiveCU=1.93
    waves, VALUBusy=17%, MFMA low, MemUnitStalled=**4%**, LDSBankConflict=2.3%. Verdict: the kernel
    is **occupancy-limited with all execution units idle** — overhead/latency-bound, occupancy capped
    by VGPR pressure (the C accumulator is BLOCK_M·BLOCK_N/THREADS = 64 fp32/thread). Device p10
    (12.06ms) ≈ do_bench (12.24ms) → host launch overhead is negligible.
  - **Skew regime gates re-derived for gfx942 (KEPT, +20%):** the persist-vs-static and skew-remap
    gates were gfx950 decisions. Re-measured (`_regime_gates.py`, do_bench): the **persistent kernel
    LOSES on every gfx942 skew shape** (B1024_D256: persist 1.045 > static-off 0.904 > remap-ON 0.857).
    Gated persist to gfx95x only; routed B1024_D256 skew → static remap-ON (xcd_c=32). **B1024_D256
    skew 1.051→0.873 (+20%), now 1.17× faster than Triton (was a tie).** Dispatch test passes.
  - **Batch-2 fan-out (4 LDS-neutral levers) — all DISCARDED:** warp (2,2,1) (compile-fails — C-shuffle
    epilogue hardcodes the (1,4,1) all-N split), warp (4,1,1) (cos=1.0 but −46-70%, all-M serializes
    N-MFMA + kills store coalescing), A-staging global→reg (cos=1.0 but −50-78%: freeing 32KB LDS
    doesn't help because per-thread A fragments then blow VGPR — VGPR is the real wall, the LDS A
    buffer was load-balancing register pressure), THREADS=512 w/ (1,8,1) (cos=1.0 but +1-5% slower).
  - **#12 B-stationary multi-M-tile (structural) — DISCARDED:** grid-shrink + inner unrolled M-tile loop,
    correct everywhere (cos=1.0 both regimes incl. skew empty/partial), but −1-7% uniform / marginal
    skew. The reframe it gave: **on uniform every M-tile is already occupied, so fewer/bigger blocks
    LOWER pipeline fill instead of amortizing it — the kernel wants MORE concurrent blocks, not fewer.**
    Only over-subscribed large-B skew showed a small win (B1024_D256 +6% @P=4), not worth a regime default.
  - **BLOCK_N=64 (more blocks + less VGPR) — DISCARDED:** halves C-accumulator VGPR (64→32 fp32/thr)
    AND doubles N-blocks, both the directions profiling + #12 pointed to, but −7-15% uniform: 2× more
    N-blocks each pay a full LDS C-shuffle epilogue, and that fixed cost multiplies with block count.
  - **Standing:** the C-shuffle epilogue fixed cost is the floor — it can't be amortized by *fewer*
    blocks (#12: fill drops) nor by *more smaller* blocks (BLOCK_N=64: epilogue multiplies). The
    genuine remaining structural idea is a **cheaper epilogue** (direct register→global store at small
    N, skipping the LDS C-shuffle) so the per-block fixed cost itself shrinks — the next lever to try.
- **2026-06-10 (loop session, batch 3 — the B-prefetch BREAKTHROUGH).** Profiling said latency-bound,
  units idle → the win was hiding global-load latency, NOT amortizing fixed cost.
  - **B-fragment prefetch decoupled to 2-ahead / 3-stage (KEPT, +5-8% ALL 8 shapes).** B (dense weight)
    is register-staged (not LDS), so deepening its prefetch costs VGPR, not LDS — sidestepping the 64KB
    wall entirely. A stays LDS 1-ahead/2-deep (mod 2); B now uses 3 register slots and prefetches 2
    K-tiles ahead (read slot i%3, prefetch tile i+2→(i+2)%3). The crux: naive `stages=3` alone is a
    no-op (the `^1` toggle never touches slot 2 → wasted VGPR, ~2% slower); the win needs the explicit
    mod-3 rotation + 2-ahead prefetch + slot-1 prologue prime. Measured do_bench cold-L2, cos=1.0, stable
    2 runs: uniform 0.509→0.473 / 1.468→1.352 / 4.236→4.042 / 12.24→11.33; skew similar. **This is the
    latency-hiding the profile predicted.** B_STAGES=4 is within noise of 3 (D512 8 K-tiles: both ~11.15);
    B_STAGES=5 regresses (VGPR pressure overtakes). B_STAGES=3 is the committed peak.
  - **Batch-3 fan-out (4 occupancy/VGPR levers), 3 DISCARDED + the B-prefetch win:** A global→LDS direct
    (blocked — gfx942 buffer_load→LDS is DWORD-only, the 32b atom breaks the 128b LDS swizzle the MFMA
    s2r needs → cos 0.01), STAGES_A=1 (null — the 32KB epilogue C-tile is the binding LDS term, shrinking
    the 16KB A buffer frees nothing), waves_per_eu=2 re-tested on the new baseline (flat/regress).
  - **Also discarded:** epilogue barrier-removal (one barrier provably redundant, cos=1.0 ×3 skew runs,
    but flat — the epilogue cost is the HBM C round-trip, not barrier latency; a direct coalesced store
    is impossible because the MFMA fragment is M-major per lane), BLOCK_K=32 (compile-fail — the (4,4,2)
    K-permute fragment layout assumes BLOCK_K=64).
  - **Final standing vs Triton:** uniform D256 ~1.35×, D512 **~1.40-1.42×**; skew up to 1.32×. The
    C-shuffle epilogue HBM round-trip is now the floor — not amortizable by fewer blocks (#12 lowers fill)
    nor more smaller blocks (BLOCK_N=64 multiplies epilogues), and no direct coalesced store exists. 2×
    would require a fundamentally cheaper epilogue or a different output data layout.
- **2026-06-11 (loop session, batch 4 — the AGPR-occupancy axis).** Premise reset: the prior "epilogue
  HBM round-trip is the floor" conclusion went STALE after the B-prefetch win moved the bottleneck.
  Attribution (env-gated `JDBBA_EPI_NOSTORE`/`JDBBA_EPI_NOEPI`, all 4 shapes) proved dropping the ENTIRE
  epilogue changes runtime 0.0% — it is fully hidden behind MFMA. Fresh PMC: the kernel is now
  **AGPR-occupancy bound** (128×128 fp32 C accum = 64 fp32/thread → 128 AGPR full → ~2 waves/CU). So this
  batch attacks per-thread accumulator AGPR.
  - **threads=512 (8-warp (1,8,1)) — KEPT, WIN on D256 uniform ONLY.** Doubling THREADS halves the
    accumulator to 32 fp32/thread (64 AGPR) → higher occupancy. **B120_D256 −13.5% (0.563→0.487),
    B1024_D256 −7.4% (4.45→4.12), cos=1.0.** Wins where the K-loop is SHORTEST (D256 = 4 K-tiles, most
    occupancy-sensitive). LOSES D512 (+6-7%: its 8-tile steady-state K-loop suffers MFMA issue-port
    contention from 8 warps) and ALL skew → gated to D256-uniform via the per-shape dispatch. `threads`
    is now a per-call dispatch knob (default 256); the warp N-count derives THREADS//64, splitting across
    M when n_warps would exceed BLOCK_N/16. Commits 3a62cc2 + 09ace8e.
  - **KEY MECHANISM: relieve AGPR via MORE WARPS, not SMALLER TILES.** Both halve the accumulator, but
    more warps KEEPS tile size (no extra block-grid → WIN) while smaller tiles DOUBLE the block-grid
    (BLOCK_N=64 +20%, BLOCK_M=64 +1.6% — re-pay 2× pipeline fills/B-reloads → LOSE). The old "BLOCK_N=64
    loses to 2× epilogues" reason was wrong (epilogue=0.0%); the verdict stands for a different reason
    (2× pipeline fills). The kernel wants FATTER tiles + MORE warps.
  - **threads=1024 — DISCARDED** (0.554 > 0.489 on B120_D256): past 512 occupancy is no longer the
    bottleneck and extra warp-scheduling (the (2,8,1) M-split) dominates. 512 is the sweet spot.
  - **XCD re-swept at threads=512** (tune_jdbba_xcd_t512.py, commit 866dc0f): B1024_D256 shifted
    c=32/w=16→w=8 (occupancy doubling moved the L2 sweet spot, ~within run-to-run noise). B120_D256 stays
    c=32/w=16. B_STAGES re-swept: 3 remains optimal (unchanged by warp count).
  - **waves_per_eu — INERT** (wpe=1, wpe=2 on D512 both flat/noise): at AGPR=128 the compiler already
    packs the max wave budget; the hint has no slack. Don't re-try.
  - **D512 next lever (open):** PMC (B1024_D512) = MfmaUtil 8%, VALUBusy 44%, AGPR=128, occ ~2 waves/CU.
    Also AGPR-bound, but threads=512 regresses it. Busiest unit is VALU (addr-gen/index math). A 4N×2M
    warp split at threads=512 (to cut N-direction issue contention while keeping doubled occupancy) needs
    the g2s/s2g copy partitions made m_warps-aware — a kernel-surface project, not a quick lever; deferred.
  - **Final standing vs Triton:** uniform D256 **~1.30-1.34×** (was 1.16-1.22×), D512 ~1.41-1.43×; skew
    up to 1.31×. The D256 shapes gained the most this batch.

## 2026-06-11 (loop session, batch 5 — fresh PMC closes the AGPR axis)

Resumed the autoresearch loop. Re-confirmed the baseline (do_bench cold-L2, all cos=1.0): uniform
B120_D256 0.493 / B120_D512 1.357 / B1024_D256 4.239 / B1024_D512 11.363 ms; flydsl leads Triton
1.29-1.44× uniform, 1.19-1.32× skew. This batch took the diagnosed-but-unattempted D512 lever and the
remaining occupancy probes to a definitive verdict via fresh profiling. **Net: no new commit — the batch
proves the kernel is at its gfx942 ceiling for these shapes and corrects two stale diagnoses.**

- **D512 warp-split at threads=512 — REFUTED (the deferred batch-4 lever).** The batch-4 note hypothesized
  D512 regresses at threads=512 because 8 N-warps cause MFMA issue-port contention, and a 4N×2M split
  might recover it. Tested cleanly via a `JDBBA_NWARP_CAP` env knob capping n_warps (cap=4 → (2,4,1),
  cap=2 → (4,2,1), default (1,8,1)). threads=512 regressed D512 ~24% in **all three** layouts
  (B120 1.34→1.67, B1024 11.2→13.9, cos=1.0). The contention hypothesis is wrong — threads=512 hurts
  D512's longer 8-tile K-loop regardless of warp arrangement; the warp-scheduling overhead dominates and
  halved-AGPR occupancy doesn't pay on the longer loop. D512 wants threads=256. Probe hook reverted.
- **Fresh D512 PMC corrects the stale VALU note.** rocprofv3 (B1024_D512, threads=256): **MfmaUtil 44%,
  VALUBusy ~20%, MemUnitStalled 0.08%, LDSBankConflict 2.8%, VGPR=56, AGPR=128.** The batch-4
  "VALUBusy 44%" was wrong — VALU is only ~20% and idle; **MFMA at 44% is the busy unit**. There is no
  VALU lever to pull (and the inner-loop soffset multiplies are already constant-folded by the unrolled
  K-loop anyway).
- **D256@threads=512 PMC.** (B1024_D256): MfmaUtil 31%, VALUBusy 22%, MemUnitStalled 0.13%,
  LDSBankConflict 3.9%, **VGPR=112, AGPR=0** — threads=512 pushed the entire accumulator into VGPR; the
  occupancy win is fully banked.
- **waves_per_eu re-tested at threads=512 — still INERT.** wpe∈{0..4} on both D256 shapes, all within
  noise (B120 best wpe=0; B1024 wpe=1 0.5% = noise). Even at AGPR=64/0 the hint has no slack. Discarded.
- **Memory-amortization lever class REFUTED by PMC.** Both shapes show MemUnitStalled ≈ 0.1% with all
  non-MFMA units idle → there is NO memory stall to hide. So B-stationary multi-M-tile (#12), deeper
  A-staging / STAGES_A (#5/#9), and async-copy changes cannot help — they amortize memory/fill latency
  that the profile shows isn't there. The kernel is **MFMA-throughput + occupancy bound** on both shapes.
- **Conclusion — the AGPR-occupancy axis is exhausted; the kernel is at its gfx942 ceiling.** The only
  remaining ceiling-raisers are (a) the 32-K MFMA atom (gfx942 hardware lacks it → use_mfma_k32 must stay
  False) or (b) lower C-accumulator register footprint without doubling the block grid — and threads=512
  already did (b) for D256 (1024 lost, smaller tiles lose to 2× block-grid). No quick lever remains.
  **Standing vs Triton unchanged from batch 4:** uniform D256 ~1.30-1.34× / D512 ~1.41-1.43×; skew up to
  ~1.31×. The committed dispatch (threads=512 D256-uniform, per-shape XCD, B-prefetch 3-stage, BLOCK_K=64)
  is the gfx942-optimal static config found by this loop.

## 2026-06-11 (loop session, batch 6 — the bottleneck reclassified: MFMA-issue latency, not occupancy)

Resumed the loop to attack the 2× gap directly. The decisive experiment reclassifies the bottleneck and
explains why every occupancy lever (batches 4-5) has plateaued. **Net: no new commit — batch 6 proves the
kernel is at its gfx942 ceiling on the current surface, and identifies the one path to 2× (a warp-
specialized rewrite) as out of scope for an autoresearch lever.**

- **Smoking gun:** threads=512 on D512 **does** double occupancy (MeanOccupancyPerActiveCU 1.92→3.82
  waves/CU, OccupancyPercent 23%→46%, AGPR 128→0) — **yet MfmaUtil stays FLAT at ~44% and runtime
  regresses** (B1024_D512 11.26→12.0). More waves changed nothing → MFMA is not idle for lack of waves;
  it idles ~56% waiting on its INPUTS (A-staging LDS round-trip + per-K barrier in the dependency chain).
  **The kernel is MFMA-issue/dependency-latency bound, not occupancy bound** — this supersedes the
  batch 4-5 "AGPR-occupancy bound" framing (occupancy was a symptom). MfmaFlopsBF16 = 4.12e12 = exactly
  2·L·K·N (zero wasted MFMA work), ~28% of bf16 MFMA peak.
- **Refuted against the new diagnosis (all measured, cos=1.0):** smaller tiles on D512 (64×128/128×64/
  64×64 all lose — occupancy gain irrelevant, block-grid overhead dominates); fatter tiles (256×* lose
  on D256, 256×256 crashes the 64KB LDS ceiling); BLOCK_K=128 on D512 (13.96/13.11 vs 11.26 — fewer
  barriers don't beat the longer K-tile); STAGES_A=3 on D512 (13.79/14.13 vs 11.26 — deeper A pipeline
  doesn't recover the MFMA gap even though D512 ignores occupancy); B_STAGES=2/4 at threads=512 (3 still
  optimal). All probe hooks (JDBBA_NWARP_CAP/BSTAGES/BLOCKK/STAGESA) reverted, tree clean.
- **D256 @ threads=512 is now VGPR-capped** (3.83 waves/CU, VGPR=112, AGPR=0, MfmaUtil 31%) — the
  threads=512 win is fully banked, no further headroom on the occupancy axis.
- **The 2× verdict.** MFMA work is already minimal (zero waste), so raising the ~28% MFMA-peak
  utilization requires either (a) the 32-K MFMA atom — **gfx942 hardware lacks it** (hard ceiling); or
  (b) a **warp-specialized producer/consumer pipeline** (dedicated copy warps feed the MFMA warps via a
  multi-buffer LDS queue so MFMA never stalls on the round-trip/barrier — the CUTLASS ping-pong surface).
  (b) is a ground-up kernel rewrite (new warp-role partitioning, async LDS queue, cross-warp barriers),
  not a one-lever autoresearch change — multi-day effort with real correctness risk. Within the current
  surface (shared-memory A-staging, single-role warps, barrier-synced K-loop) the kernel is at its gfx942
  ceiling: **uniform D256 ~1.30-1.34×, D512 ~1.41-1.43× vs Triton; skew up to ~1.31×.** The committed
  config remains the gfx942-optimal static kernel this loop found.

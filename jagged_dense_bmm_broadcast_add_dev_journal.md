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

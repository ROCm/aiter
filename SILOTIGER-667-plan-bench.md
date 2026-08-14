# SILOTIGER-667 — FlyDSL-vs-CK benchmark comparison plan (G9)

**Scope:** this document tracks **only** the per-stage **FlyDSL-vs-CK** warp-decode
benchmark comparison (gap **G9** in the main plan) — generating both sides across shapes
and dtypes and reporting an apples-to-apples table. The overall convergence plan remains
[`SILOTIGER-667-plan-10082026.md`](./SILOTIGER-667-plan-10082026.md); the Split-K work is
in [`SILOTIGER-667-plan-Split-K.md`](./SILOTIGER-667-plan-Split-K.md). A **complementary
full-MoE / AITER-default track** (comparing against `aiter.fused_moe`) is intentionally
**out of scope here** and is tracked separately in `SILOTIGER-667-bench-TODO.txt`.

**Inherited constraints — the main plan's §2 "Locked decisions" still apply in full.**
In particular:
- **Test env:** `flydsl_venv` (triton 3.6), e.g.
  `HIP_VISIBLE_DEVICES=6 ./flydsl_venv/bin/python -m pytest -q op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`.
- **GPU 6** (`HIP_VISIBLE_DEVICES=6`) for clean cold-HBM numbers.
- **Methodology:** perf only via `run_perftest` (IQR-trimmed device time); cold-HBM
  rotation; `cos ≥ 0.999` + `checkAllclose`; markdown tables.
- **Kernel/harness locations:** FlyDSL entry points in `aiter/ops/flydsl/warp_decode_moe.py`
  and the op_test/bench in `op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`; the CK
  bench is `tickets/667/harness/ck_bench_warp_decode.cpp` (+ `build_ck_bench.sh`), built
  against CK commit `62e30c9098` → `/workspaces/rocm-libraries-wdec/bench_ck_warp_decode`.

---

## 1. Goal & design decisions

Produce a `compare.py` that generates FlyDSL and CK warp-decode benchmarks over the same
shapes/dtypes/batches and emits a joined **FlyDSL/CK ratio table**.

**Locked design decisions for the comparison:**
- **Compare on TIME (µs)** — the one identical physical quantity. The ratio
  `flydsl_us / ck_us` is unit-free and definition-free (primary metric).
- **Derived metrics via a shared pluggable helper.** CK's `gu_bytes/dn_bytes/gu_flops`
  count different bytes/FLOPs than FlyDSL's weight-stream accounting, so TB/s / TFLOP/s are
  **not** directly comparable. A single `compute_metrics(..., method=)` helper (methods
  **`weight_stream`** = current FlyDSL default, **`total_traffic`** = CK-style) is the
  single source of truth, applied identically to both sides' raw times.
- **Default-vs-default config policy.** Headline compares each family at its default/
  recommended config (avoids the tuning-asymmetry bias, since we can deeply tune FlyDSL but
  not CK). Tuned FlyDSL upside (e.g. `prefetch=True` for B≤2 FP4 down) is a footnote row.
- **Both sides cold.** CK's `cold_niters_` is only *warmup*; it must be made truly cold to
  match FlyDSL's rotate-over-disjoint-experts sweep before any ratio is meaningful.
- **When a change is needed, prefer changing the CK side over FlyDSL** (keeps FlyDSL's
  recorded numbers stable).

---

## 2. Steps (tracked)

### Group A — CK harness parity (CK-side changes)

#### A1 — flip CK to cold  [x] DONE (2026-08-14)
**Mechanism change vs the original recipe:** the `stream_config` `flush_cache_` /
`rotating_count_` flags are **no-ops on this launcher**. `launch_warp_decode_*` →
`launch_kernel()` only ever calls `timing_loop_impl()` (it never reads those fields — they
are consumed by `launch_kernel_time_mask_flush_cache` and the gemm-universal profiler), and
ck_tile's `flush_cache()` is `s_icache_inv` (flushes the **instruction** cache, not the
L2/MALL **data** cache holding the weights). So flipping the flags would compile, run, and
change nothing. Instead A1 implements the cold mechanism **in the bench itself**, mirroring
the FlyDSL cold harness.
- [x] Replaced `make_cfg` + `launch_warp_decode_*` with a manual hipEvent timing loop
      (`bench_cold<Kern>`) that rotates `a.p_router_ids` over precomputed disjoint expert
      groups tiling the full E pool, using a **continuous** launch counter across
      warmup+timed so every launch reads a fresh expert group (any reuse is a full-pool
      sweep apart → cold). Env `CK_WD_ROTATE` (`<=0` = auto `ceil(E/BK)`, `1` = warm
      baseline, `>1` = that many groups).
- [x] Memory: rotation only grows the tiny router-id buffer (`rotate*B*K` int32, KB-scale);
      the GB weight pool is unchanged, so **no OOM** (verified DeepSeek B=1 and B=32).
- [x] Rebuilt via `build_ck_bench.sh`; smoke test passes.
- [x] **Verified cold works:** cold numbers drop vs warm (`CK_WD_ROTATE=1`), most on the
      previously cache-resident small Qwen shapes — e.g. B=1 `down_h2_d2` 4977→2999 GB/s
      (0.60×), `down_fp4_h2` 4299→2339 GB/s (0.54×), `gate_bf16_d2` 7830→5560 GB/s (0.71×).
- [x] Emits a provenance line to stderr (commit, cold/iters/rotate, mechanism), keeping
      stdout clean for the A2 parser.
- **Residual mechanism caveat (document in the compare table):** CK cold = disjoint-expert
  router rotation over the full pool (now *structurally the same* as FlyDSL's), not a scratch
  cache-flush; note this next to the numbers.
- Context: CK previously measured **warm** (fixed seed-42 router ⇒ the same TOPK experts
  stay MALL-resident where they fit).

#### A2 — machine-readable, FlyDSL-compatible CK output  [x] DONE (2026-08-14)
- [x] Added `CK_WD_FORMAT=csv` mode (global `g_csv`): stable stdout schema
      `shape,H,I,K,E,B,kernel,us`. Pretty table remains the default.
- [x] Emits **microseconds** (`ms*1e3`, `setprecision(4)`) to match FlyDSL units. Only raw
      `us` is emitted (no TFLOP/s or GB/s) since derived metrics are recomputed downstream by
      the shared `compute_metrics` helper (B3) applied identically to both harnesses.
- [x] Emits shape dims `H,I,K,E` as columns so the join key is dimension-based
      `(H,I,E,K,B,op)`, not coupled to the `"deepseek-v3"` name strings.
- [x] Provenance line goes to **stderr** (commit `62e30c9098`, cold/iters/rotate, `format`,
      mechanism), keeping stdout a clean parseable CSV. Smoke-tested both modes.

#### A3 — expand the CK B-set to cover FlyDSL's  [x] DONE (2026-08-14)
- [x] `CK_WD_BATCHES="1,2,4,8,32"` (env-only, no cpp change) — B=32 now has a peer.
- [x] Confirmed **every** `shape × B × kernel` cell prints for B∈{1,2,4,8,32}: 75/75 rows
      (15 per B) in both the warm and cold sweeps *and* in CSV mode, so no
      `IsSupportedArgument` skips. B=32 is OOM-free with the A1 rotation (rotation only adds
      a KB router buffer; verified DeepSeek B=32 earlier).
- [ ] (C1) Ensure `compare.py` joins on the full B set so every FlyDSL row has a CK peer.

#### A4 — extend the CK cpp with a gate_up FP4 bench  [ ]
- [ ] Add `GUProbFP4 = WarpDecodeGateUpProblem<bf16_t, pk_fp4_t, ...>` + kernel alias and a
      bench block mirroring `down_fp4_h2` (env-filterable row `gate_up_fp4`).
- [ ] Respect the **`NPerWarp=1`** constraint (the kernel static-asserts NPerWarp=2 rejects
      packed FP4).
- [ ] Pick a scale layout matching FlyDSL's e8m0 block scale; allocate packed FP4 gate/up
      pools (`E*I*H/2` bytes each). Rebuild; add to `compare.py`'s join.
- Context: **not** a CK capability gap — `WarpDecodeGateUpKernel` supports `pk_fp4_t`
  (`unpack_fp4_nibble`, `is_packed_w`); only the bench cpp never instantiates the alias.
  Pairs with FlyDSL's `flydsl_warp_decode_gate_up_fp4` (the 1.44–1.65× cold-A/B win).

### Group B — FlyDSL-side changes

#### B1 — match scale layout for the comparison  [ ]
- [ ] Run the FlyDSL FP8 legs (`bench_gate_up_cold` / `bench_down_cold`) with
      `w_scale_mode="block2d", scale_block=(128,128)` to mirror CK's `Block2D<128,128>`
      weight scales (so weight-scale byte traffic is apples-to-apples).
- [ ] FP4 down: CK uses a dummy PerTensor scale=1.0 while FlyDSL uses real e8m0 block scale
      (1,32); keep FlyDSL's e8m0 and document as perf-negligible (scale bytes ≪ FP4 stream).

#### B2 — add TOPK to the FlyDSL cold-bench return dicts  [x] DONE (2026-08-14)
- [x] Added `"TOPK": TOPK` (after `"E"`) to both `bench_down_cold` and `bench_gate_up_cold`
      return dicts, so `compare.py` has the full `(H,I,E,K=TOPK,B)` join key. `_fmt_table`
      is DataFrame-based, so it just gains a TOPK column; py_compile clean, no new lints.

#### B3 — pluggable `compute_metrics(...)` helper  [ ]
- [ ] Add `compute_metrics(B,H,I,K,dtype,us,method="weight_stream") -> {TFLOPS, TB/s, %peak}`;
      keep raw µs + dims + dtype as source of truth (do **not** bake a mode into timing).
- [ ] Two methods only: **`weight_stream`** (default; weight bytes streamed + core MACs) and
      **`total_traffic`** (mirrors CK `62e30c9098`: all operands for bytes, full epilogue for
      FLOPs — `gu_bytes=B*K*I*(H*xe+2*H*we+2)`, `dn_bytes=B*H*(K*I*2+K*I*we+2*K*4+2)`,
      `gu_flops=B*K*I*(4H+5)`, `dn_flops=3*B*H*K*I`).
- [ ] Route the existing bench columns through it with the default ⇒ recorded numbers
      unchanged. Comment-pin `total_traffic` to CK commit `62e30c9098`.

#### B4 — FlyDSL counterpart to CK's `gate_fp8_d2` (FP8-activation gate_up)  [ ]
- [ ] Add an FP8-activation gate_up entry: quantize `x` BF16→FP8 (activation scale contract),
      feed FP8 `x` + its `(1,128)` block scale through the cvt/dot2 path, fold the activation
      scale after dot2 (exponent-only convert, main plan §2).
- [ ] Match CK's layout: `XScaleFP8 = Block2D<1,128>` activation scale, FP8 weight
      `Block2D<128,128>`, dot2 on.
- [ ] Correctness (cos vs BF16-act reference) + a cold-bench field so `compare.py` can pair
      it with CK `gate_fp8_d2`. Distinct from MXFP8 block-scaled activation (separate follow-on).

#### B5 — fix the DeepSeek-E256 FP8 hole (K3 Tier-2 addressing)  [ ]
- [ ] Add per-expert **i64 base** addressing to the FP8 down + gate_up builders (i64 expert
      row base, i32 in-expert offset); mirror the reference per-row i64 pattern.
- [ ] Guard so it's only paid when `E*H*I >= 2^31` (don't regress the i32-safe path; keep FP4
      on the Tier-1 restructure).
- [ ] Correctness: DeepSeek E=256 FP8 down + gate_up cos vs torch (repro
      `/tmp/repro_k3_addr.py` should go ~0.02 → ~1.0).
- [ ] Once fixed: drop the FP8 auto-skip in the cold harness for DeepSeek E256, fill the G9
      FP8 DeepSeek rows, update the main plan's G1/K3 status + §8.2 matrix. Also unblocks the
      Kimi-K3 follow-on (E=896).
- Context: FP8 `w_row*INTER` (~3.76e9 > 2³¹) overflows the i32 byte offset at DeepSeek
  E=256, so the FP8 legs report n/a there and CK (per-row i64) has no FlyDSL peer.

### Group C — the comparison script

#### C1 — write `compare.py` (generate both, then join/compare)  [ ]
- [ ] Drive CK: subprocess the binary (path via `CK_BENCH`, exported by `build_ck_bench.sh`)
      with `CK_WD_SHAPES/BATCHES/ITERS/COLD/FORMAT`; parse CSV into records keyed by
      `(H,I,E,K,B,kernel)`.
- [ ] Drive FlyDSL: import the op_test module (under `flydsl_venv`, GPU 6), call
      `bench_down_cold` / `bench_gate_up_cold` over the same shapes/B/iters; **melt** the
      two-op rows (`fp4_us` + `fp8_us` in one row) into per-op records.
- [ ] Canonical op mapping: CK `gate_bf16_d2 → gate_up FP8`; `down_h2_d2 → down FP8`;
      `down_fp4_h2 → down FP4`; `gate_up_fp4 → gate_up FP4`; `gate_fp8_d2 → gate_up FP8-act`.
- [ ] Join on `(H,I,E,K,B,op)`; emit `ck_us`, `flydsl_us`, `ratio`, and TB/s recomputed via
      the shared `compute_metrics` (default `weight_stream`; `total_traffic` optional) applied
      to both sides. Carry FlyDSL `cos` as a sanity column; mark CK perf-only/unverified
      (uninitialized weights). Handle n/a cells (DeepSeek FP8 until B5).
- [ ] Output: markdown ratio table (for the plan) + optional CSV; print the provenance header.
- **Depends on:** A1 (cold), A2 (machine-readable), B2 (TOPK), B3 (compute_metrics).
  Fairness/completeness also depends on B1 (scale), A3 (B=32), A4/B4 (extra peers), B5 (Tier-2).

### Group D — methodology, rigor, reproducibility

#### D1 — align timing methodology  [ ]
- [ ] **Use a flat `iters=1000` for the of-record numbers on both sides (measured 2026-08-14).**
      Small-shape × low-B × FP4 cells are so fast (~10 µs) that fixed per-launch/event
      overhead dominates and a flat `iters=30–100` gives noisy, *pessimistic* numbers, e.g.
      Qwen `down_fp4_h2` B=1 read 1917 GB/s (original) / 2334 (iters=100) but converges to
      ~2731 GB/s by iters≥1000; the big cells are already flat at iters=100. A per-cell
      auto-scaling scheme is unnecessary at this scale: a full flat-1000 sweep (75 cells,
      B∈{1,2,4,8,32}, cold=20) is only **~11 s** (a warm+cold pair ~23 s), because the few
      slow cells (DeepSeek B=32 gate ~1.3 ms) dominate and 1000 iters on the fast cells costs
      almost nothing. `iters=1000` clears the fastest cell we have (Qwen B=1 FP4 = 2696 GB/s
      @1000, within ~1.3% of the 3000-iter asymptote). Keep warmup `cold≥15` (e.g. 20).
      - Keep a smaller default (e.g. `CK_WD_ITERS=100`) for quick smoke/dev runs; pin
        `iters=1000` in the one-command driver (D6) for the recorded table. Same on FlyDSL.
      - Corollary: a low-iter run can invert fast pairs (it spuriously showed Qwen B=1 FP4
        down *slower* than FP8; converged, FP4 is marginally faster in time). Do not trust
        ratios from under-converged fast cells — treat unconverged cells as noisy in D5.
- [ ] Document the residual stat difference (FlyDSL IQR-trimmed median vs CK mean); pick one
      to report or report both.
- [ ] **Lock GPU clocks** (`rocm-smi` fixed SCLK, disable boost) for both runs; record the
      locked clock in provenance. Run both under `HIP_VISIBLE_DEVICES=6`.

#### D2 — define the G9 comparison coverage matrix (the completion gate)  [ ]
- [ ] Enumerate `op × dtype × shape × B` (see §3); mark each cell covered / n/a (+reason).
- [ ] **G9 closes only when every non-n/a cell has a FlyDSL+CK pair in the table.**

#### D3 — decide + document the FlyDSL config policy (default-vs-default)  [ ]
- [ ] Pin the exact config per side next to the table: FlyDSL default (prefetch off, `kh`
      auto=2, serialize on); CK = the maintainer-intended recommended variant (`down_h2_d2`,
      `gate_bf16_d2`), noting CK has no single runtime "default" (disclose the mild asymmetry).
- [ ] Keep tuned upside as a footnote row where a knob is the documented recommendation
      (e.g. `prefetch=True` for B≤2 FP4 down, ~5%); headline stays default.

#### D4 — confirm functional equivalence between the harnesses  [ ]
- [ ] Verify (with code refs) both compute the same work: SiLU on both; same `silu(gate)·up`
      (gate_up) and `Σ rw·(inter·w)` (down); same scale semantics (block2d weight/activation
      scale) and output dtype (bf16). Flag any epilogue/scale difference; if the math differs,
      the pair is not comparable and is marked n/a.

#### D5 — capture environment/provenance + run-to-run variance  [ ]
- [ ] `compare.py` header records GPU model, ROCm version, arch, locked clocks, CK commit
      (`62e30c9098`), and the aiter commit.
- [ ] Run each config N times; report spread (CV or min/median/max) and flag noisy rows.

#### D6 — one-command reproducible driver + checked-in artifact  [ ]
- [ ] Top-level entry (script / Make target) that optionally rebuilds CK, locks clocks, runs
      `compare.py` under `flydsl_venv` on GPU 6, and writes CSV + markdown into `tickets/667/`.
- [ ] Reference the generated artifact from the plan doc.

#### D7 — (optional/stretch) numerical cross-check FlyDSL-vs-CK outputs  [ ]
- [ ] Add a CK mode that accepts/initializes real inputs and dumps the output tensor; feed
      identical inputs to both and compare (cos / `checkAllclose`). Heavier — explicitly
      optional / stretch (both harnesses are currently perf-only).

---

## 3. G9 comparison coverage matrix (completion gate)

Legend: ✅ has a FlyDSL+CK pair · ⏳ pending the named step · ⛔ n/a (+reason).
Shapes: DeepSeek-V3 (H7168/I2048/E256/K8), MiniMax (H3072/I1536/E256/K8),
Qwen3Next-TP1 (H2048/I512/E512/K10). Batches B∈{1,2,4,8,32}.

| Op | dtype | DeepSeek-V3 | MiniMax | Qwen3Next-TP1 |
|---|---|---|---|---|
| gate_up | BF16-act × FP8-w | ⛔ FP8 hole (⏳ B5) | ⏳ | ⏳ |
| gate_up | FP8-act × FP8-w | ⛔ FP8 hole (⏳ B5) | ⏳ B4 | ⏳ B4 |
| gate_up | FP4-w | ⏳ A4 | ⏳ A4 | ⏳ A4 |
| down | FP8-w | ⛔ FP8 hole (⏳ B5) | ⏳ | ⏳ |
| down | FP4-w | ⏳ | ⏳ | ⏳ |

(DeepSeek FP8 cells stay ⛔ until B5 lands the Tier-2 i64 base addressing.)

---

## 4. Dependencies (build order)
- **C1 (`compare.py`)** needs **A1, A2, B2, B3**.
- **Full FP8 coverage** (DeepSeek) needs **B5**.
- **gate_up FP4 pair** needs **A4** (FlyDSL side already exists).
- **gate_up FP8-act pair** needs **B4** (FlyDSL side) — CK already has `gate_fp8_d2`.
- **Trustworthy ratios** need **A1 (cold), B1 (scale), D1 (timing), D3 (config), D4 (equiv)**.

## 5. Risks / watch-items
- **Cold flush unproven:** A1 must be *verified* (numbers drop), not assumed.
- **Rotating-buffer OOM:** GB-scale weight pools × `rotating_count_` can OOM at B=32/DeepSeek.
- **`total_traffic` drift:** pinned to CK commit `62e30c9098`; revisit if CK's byte/FLOP
  formulas change.
- **Env reconciliation** (shared with the full-MoE track): one env must import FlyDSL +
  aiter; the CK side is a standalone binary so it's driven via subprocess.
- **Regime honesty:** ratios are only "apples-to-apples" once cold + scale + timing + config
  are aligned; report the config/regime next to every table.

## 6. Status log
- 2026-08-14 — plan drafted from `SILOTIGER-667-bench-TODO.txt`; per-stage G9 items folded
  here. Full-MoE / AITER-default track kept separate in the TODO file (deferred).
- 2026-08-14 — **A1 done.** Discovered the `stream_config` flush/rotate flags are no-ops on
  the warp-decode launcher (and ck_tile's flush is icache-only); implemented cold in the
  bench via a manual hipEvent loop + disjoint-expert router rotation (`CK_WD_ROTATE`).
  Verified cold numbers drop vs warm on the small Qwen shapes; no OOM on DeepSeek B=1/32.
  `tickets/667/harness/ck_bench_warp_decode.cpp` updated + rebuilt.
- 2026-08-14 — **A2 done.** Added `CK_WD_FORMAT=csv` (schema `shape,H,I,K,E,B,kernel,us`,
  µs, provenance on stderr) alongside the default pretty table; smoke-tested both.
- 2026-08-14 — **D1 refined.** Adopted flat `iters=1000` (cold≥15) for of-record numbers
  (full sweep ~11 s; clears the fastest cell within ~1.3%), smaller default for dev runs.
- 2026-08-14 — **A3 done.** Confirmed B∈{1,2,4,8,32} fully covered (75/75 cells, table +
  CSV), env-only, no `IsSupportedArgument` skips, no OOM at B=32. Remaining sub-item is the
  `compare.py` full-B join (belongs to C1).
- 2026-08-14 — **B2 done.** Added `"TOPK"` to both FlyDSL cold-bench return dicts (completes
  the `(H,I,E,K,B)` join key); py_compile clean.

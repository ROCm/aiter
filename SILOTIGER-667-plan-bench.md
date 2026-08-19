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
  rotation; `checkAllclose`; markdown tables. (Note: the *cold comparison* benches gate at
  `cos ≥ 0.99` on the first `_COS_CHK_TOKENS=4` tokens — per-token work is uniform and the
  full-pool fp32 dequant would OOM at DeepSeek B=32; the stricter `cos ≥ 0.999` still applies
  to the primitive/warm unit tests.)
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
  not CK). Tuned FlyDSL upside (e.g. `prefetch=True` for B≤2 FP4 down) is disclosed as a
  footnote/caveat, not folded into the headline — the of-record table itself carries only the
  default config (no tuned row is materialized today).
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
- [x] (C1) `compare.py` joins on the full B set so every FlyDSL row has a CK peer — verified
      in every of-record regen (B∈{1,2,4,8,32} all paired).

#### A4 — extend the CK cpp with a gate_up FP4 bench  [x]
- [x] Added `GUProbFP4 = WarpDecodeGateUpProblem<bf16_t, pk_fp4_t, …, XScaleBF16, WScalePT,
      Silu, kVec4>` + `GUKernFP4` alias and a `gate_up_fp4` bench block in the cpp, mirroring
      `down_fp4_h2` (packed pools `E*I*H/2` bytes each, PerTensor dummy scale, `stride=H/2`).
- [x] Uses **`NPerWarp=1`** + the non-dot2 scalar path (dot2 / NPerWarp=2 reject packed FP4).
- [x] **CK kernel patch required (correction to the old "not a capability gap" note).** The
      gate_up kernel's `IsSupportedArgument` rejected the packed stride `H/2` (`stride_w_gate <
      hidden`) — the down kernel already had the `pk_fp4_t → hidden/2` exception but gate_up
      did not. Patched `warp_decode_gate_up_kernel.hpp` to mirror down (`min_w_stride =
      pk_fp4_t ? hidden/2 : hidden`). **This means the CK worktree source no longer matches
      pinned `62e30c9098` byte-for-byte** (one-line validator fix; kernel math unchanged). See
      the provenance note in §5.
- [x] Rebuilt; `compare.py` join now fills the gate_up FP4 CK column. Ratios (f/c) 0.56–0.84
      (FlyDSL ~1.2–1.7× faster, matching the expected 1.44–1.65× cold win), all `%peak≤100`,
      `cos=1.0000`. Pairs with FlyDSL's `flydsl_warp_decode_gate_up_fp4`.

### Group B — FlyDSL-side changes

#### B1 — match scale layout for the comparison  [x] DONE (2026-08-14)
- [x] Run the FlyDSL FP8 legs (`bench_gate_up_cold` / `bench_down_cold`) with
      `w_scale_mode="block2d", scale_block=(128,128)` to mirror CK's `Block2D<128,128>`
      weight scales. Both FP8 pool generators now emit a flat Block2D scale
      (`_FP8_SCALE_BLOCK=(128,128)`), the refs apply it via `_block2d_scale_matrix`, and the
      timed kernel now does the same **per-block scale load+multiply** as CK (not just a byte
      match — the fairness win is in the timed work). Verified `fp8_cos=1.0000` for down &
      gate_up on the FP8-running shapes (minimax, qwen3next; DeepSeek FP8 still gated on B5).
      Note the Block2D<128,128> scale-byte traffic is genuinely tiny for FP8 (≪ FP4 e8m0
      (1,32)), so `compute_metrics(weight_stream)` FP8 byte accounting is left unchanged
      (preserves the B3 byte-for-byte invariant); only the FP4 e8m0 term is carried there.
- [x] **Measured block2d-vs-pertensor cost (A/B, same weights, iters=1000, 2026-08-17):**
      the per-block scale work lands entirely on **`down`**, not `gate_up`:
      | shape | B | down Δ | gate_up Δ |
      |---|---|---|---|
      | minimax | 1 | **+38.5%** | −1.6% |
      | minimax | 32 | **+14.0%** | −1.1% |
      | qwen3next | 1 | **+10.5%** | −0.8% |
      | qwen3next | 32 | **+11.1%** | +2.2% |
      `gate_up` amortizes the scale load across its much larger grid (neutral, within noise);
      the `down` inner loop pays a real **~10–38%** penalty for the per-(128,128)-block scale
      reload+multiply vs a single pertensor broadcast (worst at small B). This is the correct
      fairness adjustment (CK's recommended `down_h2_d2` already runs Block2D<128,128>, so
      pre-B1 FlyDSL down FP8 under-counted scale work); it moved the down-FP8 `flydsl/ck`
      ratios modestly toward CK. Cross-checks compare.py (minimax B=1 down FP8 = 11.98 A/B vs
      11.16 in-sweep). **Optimization lead:** FlyDSL's block2d scale handling in the `down`
      kernel is a candidate to reclaim that 10–38%.
- [x] **Resolved as a documented storage-byte caveat via D7 (2026-08-18).** D7 showed CK's generic
      `load_block2d_scale` handles a `Block2D<1,32>` scale with **no kernel work** and matches torch
      at cos=0.999999, so the FP4 scale *values* are not a gap; the only residual is scale
      **storage-byte accounting** (CK float 4B vs FlyDSL e8m0 1B per `(1,32)` block) — a
      traffic-metric modeling choice, disclosed as the ~6% CK-favored caveat in the artifact header
      (B1 lines below / D3 / D4 / §5). Not actionable kernel work. Original note (kept for context):
      CK uses a dummy PerTensor scale=1.0 while FlyDSL uses real e8m0 block scale
      (1,32). **Not perf-negligible (corrected 2026-08-14):** FlyDSL's e8m0 stream is
      `TOPK·H·(I/32)` ≈ 0.33 MB vs the `TOPK·H·I/2` ≈ 5.24 MB FP4 weight stream at Qwen
      down B=1 — **~6%** of the weight bytes. CK's PerTensor streams ~0 scale bytes, so CK
      does ~6% less real traffic ⇒ its time is slightly lower and the FlyDSL/CK ratio is
      modestly **CK-favored** on FP4 cells. (Also note `compute_metrics(weight_stream)`
      includes the e8m0 term for *both* sides, so it over-attributes bytes to CK — only the
      **time** ratio is ground truth there.) Keep FlyDSL's e8m0 and treat this as a
      documented ~6% CK-favored caveat next to the FP4 rows.
- [x] **Corrected 2026-08-18 (via D7):** the CK down kernel's `Block2D` scale path is *generic*
      (`load_block2d_scale` indexes any `Block_N/Block_K`), so a `Block2D<1,32>` MXFP4 scale needs
      **no kernel work** — D7 instantiates `down_fp4_h2` with a `(1,32)` float scale and it matches
      torch at cos=0.999999. The scale *values* are therefore not the blocker; the only residual
      B1 gap is scale **storage bytes** (CK float 4B vs FlyDSL e8m0 1B per `(1,32)` block), i.e. a
      traffic-metric modeling choice. Numerically-exact FP4 parity is thus already available; the
      ~6% weight-byte caveat above is purely about how the two sides *count* scale traffic.
- **See also D7** — CK outputs are now numerically validated against torch across FP8/BF16/FP4.

#### B2 — add TOPK to the FlyDSL cold-bench return dicts  [x] DONE (2026-08-14)
- [x] Added `"TOPK": TOPK` (after `"E"`) to both `bench_down_cold` and `bench_gate_up_cold`
      return dicts, so `compare.py` has the full `(H,I,E,K=TOPK,B)` join key. `_fmt_table`
      is DataFrame-based, so it just gains a TOPK column; py_compile clean, no new lints.

#### B3 — pluggable `compute_metrics(...)` helper  [x] DONE (2026-08-14)
- [x] Added `compute_metrics(op, B, HIDDEN, INTER, TOPK, w_dtype, us, method="weight_stream",
      act_dtype="bf16") -> {TFLOPS, TB/s, %peak}` in the op_test module. Raw µs + dims + dtype
      stay the source of truth (no mode baked into timing). `op∈{down,gate_up}`,
      `w_dtype/act_dtype∈{fp4,fp8,bf16}`; `%peak` = TB/s vs `_HBM_PEAK_TBS`; NaN-guarded for
      us≤0/NaN. `act_dtype` param is ready for the B4 FP8-activation (`gate_fp8_d2`) peer.
- [x] Two methods only: **`weight_stream`** (default) and **`total_traffic`** (CK-style, all
      operands + full epilogue — the `gu_bytes/dn_bytes/gu_flops/dn_flops` formulas).
- [x] Routed both cold benches (`bench_down_cold`, `bench_gate_up_cold`) through it. Verified
      by a standalone numeric check that `weight_stream` **reproduces the old wbytes/TB-s and
      TFLOPS exactly** across sampled DeepSeek/MiniMax/Qwen shapes (recorded numbers
      unchanged); `total_traffic` ≈2× bytes as expected.
- [x] `total_traffic` docstring **pinned to CK commit `62e30c9098`** with a "revisit if CK's
      byte/FLOP formulas change" note.
- [x] **Follow-up (B3, 2026-08-18):** migrated the *warm* benches `bench_gate_up` /
      `bench_down` / `bench_down_fp4` to `compute_metrics` (`weight_stream`) so the byte/FLOP
      model has a single source of truth on the warm path too — all inline `outputs`/`flops`/
      `wbytes`/`tbs` accounting removed; each now returns `m["TFLOPS"]`/`m["TB/s"]`/`m["%peak"]`.
      Exact-equivalence re-verified by a standalone numeric check (60k sampled DeepSeek/MiniMax/
      Qwen shape-cells, all with even & `%32`-divisible `INTER`): **0 mismatches**, so recorded
      TFLOPS/TB-s/%peak are byte-for-byte unchanged. The only behavioral delta is the `us<=0`
      edge (old inline returned `0.0`, `compute_metrics` returns `NaN`) — never hit in real runs.

#### B4 — FlyDSL counterpart to CK's `gate_fp8_d2` (FP8-activation gate_up)  [x] DONE (2026-08-18)
- [x] Added `build_gate_up_fp8_act_module` + public `flydsl_warp_decode_gate_up_fp8act`
      (exported in `aiter/ops/flydsl/__init__.py`): FP8 e4m3 `x` + `Block2D<1,128>` activation
      scale, both operands scaled-converted to BF16 (scale=1) and fed through dot2; per K-block
      the lane partial gets `dot * (x_scale_block * w_scale_block)` before the single wavefront
      reduce (exponent-only fold, main plan §2). Reuses the K3 Tier-2 i64 base (B5) so DeepSeek
      E=256 works.
- [x] Matches CK's layout: activation scale `Block2D<1,128>` (`x_scale_bk=128` = CK `kBXK`),
      FP8 weight `Block2D<128,128>`, dot2 on. `x` and weight share the 4-fp8/word dword layout.
- [x] Correctness: cos vs a dequant-both torch reference (`_ref_gate_up_fp8act_pool`) =
      **1.000000** at small / qwen3next E=512 / DeepSeek E=256. Added an `fp8act_*` field to
      `bench_gate_up_cold` and a new `("gate_up","fp8","fp8")` cell in `compare.py` that pairs
      with CK `gate_fp8_d2`. Regenerated the of-record G9 artifact: **15 new fp8-act cells**
      fill (3 models × 5 B) at cos=1.0000, ratios 0.75→0.95, ~35–79 %peak; **0 n/a**. Matrix now
      fully paired for every op×dtype×act CK exposes. Distinct from MXFP8 block-scaled
      activation (separate follow-on).

#### B5 — fix the DeepSeek-E256 FP8 hole (K3 Tier-2 addressing)  [x] DONE (2026-08-17)
- [x] Added per-expert **i64 base** addressing to the FP8 `down` + `gate_up` builders: fold the
      per-expert byte base (`e * H*I`, fp8 = 1 B/elem) into the buffer descriptor as i64
      (`_ptr_rsrc_off`), and address weights by the i32-safe **in-expert** dword offset
      (`neuron_j*(H/4)` for gate_up, `(out_j+h)*(I/4)` for down). Scale offsets stay i32
      (they never overflow). `kernels/warp_decode_moe.py`.
- [x] Guarded by a build-time `num_experts` param: Tier-2 only fires when
      `E*I*H >= 2^31` (`use_i64_base`); the i32-safe fast path and all FP4 shapes are byte-for-
      byte unchanged. Threaded `E` through `_get_gate_up`/`_get_down_reduce` + both public wrappers.
- [x] Correctness — `/tmp/repro_k3_addr.py` gate_up @ **E=896** (dword idx ~2.46e9 > 2^31):
      cos ~0.02 → **0.999999**. DeepSeek **E=256** FP8 (`/tmp/repro_k3_deepseek_fp8.py`):
      `down` cos=**1.000000**, `gate_up` cos=**1.000000** (was ~0.02).
- [x] Dropped the cold-harness FP8 auto-skip: `run_fp8` now gates on the *in-expert* size
      (`H*I < 2^31`) instead of `E*H*I`, so DeepSeek E=256 runs. Regenerated the of-record G9
      artifact — **0 n/a cells** (was 10); the 10 DeepSeek FP8 cells fill at cos=1.0000, ratios
      0.60→1.03, ~69–74 %peak (repeats=3, loaded sclk median ~2393 MHz). Matrix now 15/15
      op×dtype×shape (×5 B = 75 paired points). Also unblocks the Kimi-K3 follow-on (E=896).
- Context (resolved): FP8 whole-pool byte offset `w_row*INTER*4` (~1.5e10 for the last
  DeepSeek expert; dword idx ~3.76e9) wrapped the i32 voffset, so the FP8 legs read garbage
  (cos ~0.02) and reported n/a; the Tier-2 i64 base keeps the in-expert offset i32-safe
  (an expert spans H*I ≤ ~1.5e7 B).

### Group C — the comparison script

#### C1 — write `compare.py` (generate both, then join/compare)  [x] DONE (2026-08-14)
`tickets/667/harness/compare.py`; smoke-tested end-to-end on GPU 6 (qwen3next B=1).
- [x] Drives CK via subprocess (path from `CK_BENCH`/`--ck-bench`) with
      `CK_WD_SHAPES/BATCHES/ITERS/COLD` + `CK_WD_FORMAT=csv`; parses the CSV into records
      keyed by dims, taking the **recommended** kernel per cell (stderr provenance captured).
- [x] Drives FlyDSL by importing the op_test module and calling `bench_down_cold` /
      `bench_gate_up_cold` over the same shapes/B/iters; **melts** the merged fp4+fp8 rows
      into per-`(op,dtype,act)` records (with `cos`). Added `fp8_cos` to both cold dicts so
      the FP8 sanity column populates.
- [x] Canonical map `CK_MAP` (kernel → op/w_dtype/act_dtype/recommended): `gate_bf16_d2 →
      gate_up fp8/bf16-act`; `down_h2_d2 → down fp8`; `down_fp4_h2 → down fp4`; `gate_up_fp4
      → gate_up fp4` (CK ✅ A4); `gate_fp8_d2 → gate_up fp8/fp8-act` (FlyDSL pending B4).
- [x] Joins on `(H,I,E,K,B,op,w_dtype,act_dtype)`; emits `flydsl_us`, `ck_us`, `ratio=f/c`,
      and TB/s + %peak recomputed via the shared `compute_metrics` (`--method
      weight_stream|total_traffic`) applied to both sides. Carries FlyDSL `cos`; CK marked
      perf-only/uninitialized. n/a cells noted (DeepSeek FP8 → B5; gate_up FP4 CK now ✅ A4).
- [x] Output: markdown ratio table (HTML-comment provenance header: gfx, aiter+CK commits,
      iters/cold/timing/method, CK provenance line, config-policy note) + optional `--csv-out`.
- **Depends on:** A1 (cold) ✅, A2 (CSV) ✅, B2 (TOPK) ✅, B3 (compute_metrics) ✅, B1 (scale)
  ✅, A4 (gate_up FP4 peer) ✅, B4 (FP8-act peer) ✅, B5 (DeepSeek Tier-2) ✅ — all cells now
  joined (0 n/a).

### Group D — methodology, rigor, reproducibility

#### D1 — align timing methodology  [x]
- [x] **Flat `iters=1000` for the of-record numbers on both sides (measured 2026-08-14; now
      pinned in the D6 driver).** Small-shape × low-B × FP4 cells are so fast (~10 µs) that
      fixed per-launch/event overhead dominates and a flat `iters=30–100` gives noisy,
      *pessimistic* numbers, e.g. Qwen `down_fp4_h2` B=1 read 1917 GB/s (original) / 2334
      (iters=100) but converges to ~2731 GB/s by iters≥1000; the big cells are already flat at
      iters=100. A per-cell auto-scaling scheme is unnecessary at this scale. `iters=1000`
      clears the fastest cell we have (Qwen B=1 FP4 = 2696 GB/s @1000, within ~1.3% of the
      3000-iter asymptote). Keep warmup `cold≥15` (e.g. 20).
      - Keep a smaller default (`CK_WD_ITERS=100`) for quick smoke/dev runs; `iters=1000` is
        pinned in the D6 driver for the recorded table. Same on FlyDSL.
      - Cost (post distinct-per-token routing): a full single-pass FlyDSL+CK compare is
        **~22–25 s**; the D6 of-record run at `--repeats 3` is **~66–75 s**. (The old "~11 s"
        note referred to the CK-only bench alone.)
      - Corollary: a low-iter run can invert fast pairs (it spuriously showed Qwen B=1 FP4
        down *slower* than FP8; converged, FP4 is marginally faster in time). Do not trust
        ratios from under-converged fast cells — D5 flags them as noisy.
- [x] **Residual statistic difference documented (report both, no reconciliation needed).**
      FlyDSL `device` timing = torch-profiler device time, **IQR-trimmed** when iters>30
      (`_timing_kwargs`); CK = **arithmetic mean** (`ms/iters` over the hipEvent loop). At the
      flat `iters=1000` with the measured <1% per-cell spread (D5), trimmed-median and mean
      differ negligibly, so both sides' raw µs are reported as-is and the ratio is unaffected;
      the small definitional difference is disclosed here rather than forced into one statistic.
- [x] **GPU clocks: lock NOT available on this gfx950 — use variance control instead
      (decided 2026-08-17).** The driver exposes only two discrete SCLK DPM levels
      (`0:500`, `1:2400` MHz; plus `S:94` idle) — **no mid ~1700 level** — so
      `--setperfdeterminism 1700` silently no-ops and `--showsupportedclocks` is empty.
      MCLK is fixed at **2000 MHz** (single level), and firmware DVFS holds a stable
      **~1789 MHz** SCLK under sustained load. Forcing `high` would request 2400 (above the
      sustainable ~1789) and get thermally clamped mid-run, so it is not reliably more
      deterministic than `auto`. Since MCLK (the dominant factor for these memory-bound
      kernels) is already pinned and SCLK is empirically stable under load, we **keep `auto`
      clocks and control run-to-run variance in D5** (warm to steady state, sample the
      effective SCLK during the sweep into provenance, N-repeat spread, flag noisy cells)
      rather than pinning. Run both sides under `HIP_VISIBLE_DEVICES=6`.
- [x] **Cold routing verified + made fair (2026-08-17).** `run_perftest(num_rotate_args=1)`
      re-invokes the rotation closure every timed iter (rotation is real). `_router_group_list`
      now routes **B*TOPK distinct experts per launch** (`bk=B*TOPK`, `rotate=ceil(E/bk)`),
      matching CK (`rids[i]=i%E`); this both fixed the `%peak>100%` artifact and closed a
      B>1 weight-traffic fairness gap vs CK (see status log).

#### D2 — define the G9 comparison coverage matrix (the completion gate)  [x]
- [x] Enumerated `op × dtype × shape` (×B∈{1,2,4,8,32}) in §3, reconciled against the
      of-record `tickets/667/g9_compare.{md,csv}`. **9/15 cells (45 paired points) covered**:
      `down fp4` ×3, `down fp8` ×{MiniMax,Qwen}, `gate_up fp4` ×3, `gate_up fp8 bf16-act`
      ×{MiniMax,Qwen}. Each marked covered / ⏳(step) / ⛔(reason).
- [x] **G9 completion gate:** every cell now has a FlyDSL+CK pair — the last two holes
      (DeepSeek FP8 → **B5**, gate_up FP8-act → **B4**) closed 2026-08-17/18; see §3 (0 n/a).

#### D3 — decide + document the FlyDSL config policy (default-vs-default)  [x]
- [x] **Pinned the exact config per side in the artifact provenance header** (verified from
      `aiter/ops/flydsl/warp_decode_moe.py`; the benches pass no overrides, so these are the
      library defaults):
      - **FlyDSL:** `serialize_dot2=True`, `kh_per_warp=auto` (→2 when HIDDEN even),
        `prefetch=False`, `kvector=auto`. `down_fp4` `dot2_acc=4`; `gate_up_fp4` `dot2_acc=1`
        (G7: acc>1 ~4% slower for gate_up); `down_fp8` `split_k=1`. FP8 legs use
        `w_scale_mode=block2d(128,128)` to match CK.
      - **CK:** the maintainer-recommended variant per op (`down_h2_d2`, `down_fp4_h2`,
        `gate_bf16_d2`, and the new `gate_up_fp4` non-dot2/NPerWarp=1). CK exposes no single
        runtime "default" (variant is chosen at instantiation), so the pairing is a **mild
        asymmetry** — disclosed in-header rather than hidden.
- [x] Tuned upside stays a footnote, not the headline: e.g. `prefetch=True` is a documented
      A/B lever for B≤2 FP4 `down` (~5%) but the of-record table uses `prefetch=False`.
- [x] **FP8 scale-granularity caveat (B1):** the headline pins FlyDSL FP8 to `block2d(128,128)`
      to match CK, which costs `down` **~10–38%** vs pertensor (measured; gate_up neutral).
      So the down-FP8 `flydsl/ck` ratio is a **conservative (CK-favored) lower bound** on
      FlyDSL's advantage — a model tolerating a coarser scale could reclaim that 10–38%.
      Also noted: FP4 rows carry a ~6% CK-favored scale-traffic bias (CK dummy PerTensor vs
      FlyDSL e8m0 `(1,32)`). Both caveats are in the artifact header (D3 line).

#### D4 — confirm functional equivalence between the harnesses  [x]
Audited both sides by code inspection (2026-08-17). **Conclusion: the compared cells compute
the same math** — no cell needs to be marked n/a on equivalence grounds. Details:
- [x] **gate_up epilogue — identical.** CK computes `result = silu(gate_acc) * up_acc` and
      stores bf16 intermediate (`warp_decode_gate_up_kernel.hpp` ~824–830); FlyDSL ref is
      `(gate * sigmoid(gate)) * up` → bf16 (`_ref_gate_up_fp4_pool`/`_fp8_pool`, test ~1717/
      1773). CK's `element_wise::Silu` is `x*(1/(1+exp(-x)))` = `x*sigmoid(x)`
      (`unary_element_wise_operation.hpp` ~1364) — same function, SiLU on the **gate** only.
- [x] **down reduction — identical.** CK: `y[b,out_j] = Σ_k w_k · ds_k · Σ_i inter·w_down`,
      bf16 store (`warp_decode_down_reduce_kernel.hpp` ~756–872); FlyDSL ref:
      `y[b] += rwt[b,k] · (inter[b,k] @ w_down_eᵀ)` → bf16 (`_ref_down_fp4_pool`, test ~1461–
      1474). Same `Σ router_wt · (inter·w)` structure, same bf16 output.
- [x] **FP8 weight scale — matched (B1).** Both apply a `Block2D<128,128>` weight scale
      (`load_block2d_scale` on CK vs the `_block2d_scale_matrix` dequant on FlyDSL). Output/
      intermediate dtype bf16 on both.
- [x] **FP4 weight scale — the one documented difference (not a formula difference).** FlyDSL
      applies a real e8m0 `(1,32)` block scale; CK uses a dummy PerTensor scale=1.0. This is a
      **scale-granularity / traffic** difference (the ~6% CK-favored caveat, B1/D3/§5), not a
      difference in the compute graph, so the cells remain comparable on *time* — just flagged.
- [x] **Router weights are perf-irrelevant.** CK's harness fills `rwts=1/K` (uniform) and runs
      **uninitialized weights** (perf-only, data-independent timing); FlyDSL runs real
      quantized weights and validates `cos≥0.99`. So equivalence of the *math* is established
      here by inspection; a numerical output cross-check on identical inputs is the separate
      **D7** (still optional/stretch).
- [x] **FP8-activation (`gate_fp8_d2`) — now joined (B4).** FlyDSL
      `flydsl_warp_decode_gate_up_fp8act` computes the same `silu(gate)·up` with the activation
      block scale folded after dot2 (same graph as the BF16-act peer, just an extra scaled
      convert on `x`); cos=1.0 vs the dequant-both reference, so it's comparable, not n/a.

#### D5 — capture environment/provenance + run-to-run variance  [x]
- [x] `compare.py` header records arch (`get_gfx()`), aiter commit, CK worktree commit
      (`62e30c9098`), CK provenance line, iters/cold/timing/method/**repeats**, and the
      effective loaded-SCLK min/median/max sampled during the run. (Full ROCm-version string
      still available via `rocm-smi` if wanted; the header covers the perf-relevant provenance.)
- [x] **N-repeat variance + effective-SCLK sampling (primary defense against clock drift,
      since clocks can't be pinned on this gfx950 — see D1).** `--repeats N` (default 3) reruns
      each full sweep; the headline `flydsl_us`/`ck_us` is the per-cell **median** and new
      `fly_spr%`/`ck_spr%` columns report `100*(max-min)/median`. A cell is flagged **noisy**
      (`--noise-pct`, default 5%) when either side's spread exceeds the threshold. A background
      `ClockSampler` polls `rocm-smi --showgpuclocks` SCLK every 0.25s across the whole run and
      records loaded (`≥400 MHz`) min/median/max into the provenance header.
- [x] **Observed (2026-08-17, repeats=3, GPU 6):** under sustained load the GPU boosts to
      **loaded SCLK median ~2391 MHz** (max 2404) — i.e. near the top {2400} DPM level, *higher*
      than the ~1789 seen under a pure-matmul probe; the decode sweep keeps it pinned high.
      Per-cell spread is **<1% for almost all cells**; CK is extremely stable (mostly <0.5%).
      The only noisy (>5%) cells are the small/fast `down` cells (e.g. deepseek down fp4 B=1
      ~7%, minimax down fp8 B=1/B=2 ~6–7%) — exactly the under-converged fast-cell regime D1
      warned about. Artifact: `tickets/667/g9_compare.{md,csv}` (D6 of-record).
- [x] `%peak>100%` resolved (2026-08-17): was the shared-expert routing bug (FlyDSL read
      TOPK vs CK's B*TOPK per launch); after the distinct-per-token fix all cells are
      `%peak≤100%` (post-route table). `_HBM_PEAK_TBS=8.0` confirmed reasonable for gfx950.

#### D6 — one-command reproducible driver + checked-in artifact  [x]
- [x] `tickets/667/harness/run_g9_compare.sh` — one-command driver that optionally
      rebuilds CK (`--build-ck`), then runs `compare.py` under `flydsl_venv` on GPU 6 with
      the of-record flags (`--iters 1000 --cold 20 --repeats 3`) and writes CSV + markdown to
      `tickets/667/g9_compare.{md,csv}`. Flags: `--gpu/--repeats/--iters/--cold/--out-prefix`
      plus `-- <compare.py args>` passthrough. **No clock-locking step** — clocks can't be
      pinned on this gfx950 (D1); D5's N-repeat spread + effective-SCLK sampling record the
      clock regime in the artifact header instead.
- [x] Checked-in artifact: `tickets/667/g9_compare.{md,csv}` (of-record, repeats=3). Reproduce
      with `bash tickets/667/harness/run_g9_compare.sh`.

#### D7 — numerical cross-check CK-vs-torch outputs  [x] DONE (2026-08-18)
The perf harnesses run uninitialized weights + dummy scales (timing is data-independent, so this
doesn't affect perf numbers). D7 = "convert CK to real weights + real per-block scales and validate
the outputs numerically." Done in two increments (D7-lite, then full D7) — both behind a
`CK_WD_VALIDATE=1` flag that returns before the timing loop, so perf runs are untouched.

**How it works.** `ck_bench_warp_decode.cpp` host-quantizes real inputs
(`type_convert<fp8_t>` = OCP e4m3 = torch `float8_e4m3fn`; FP4 nibble-packed 2/byte, low nibble =
even element, matching FlyDSL), runs each kernel **once** functionally, and dumps the *exact* input
bytes (weights, activations, per-block Block2D scales, router ids/wts) + GPU output to
`ck_validate_dump/` + a manifest. The Python validator (`tickets/667/harness/validate.py`) reloads
the identical bytes, rebuilds the torch reference (same math + Block2D scale indexing as the FlyDSL
op-test refs — the CK `load_block2d_scale` index formula `(row/BN)*(cols/BK)+(col/BK)` is
byte-identical to FlyDSL's `_block2d_scale_matrix`), and compares (cos / max Δ). Wired as
`run_g9_compare.sh --validate` (build-if-needed → dump → compare, non-zero exit on divergence).
Dumps are git-ignored (regenerated on demand).

**Result** on a small `B4/H512/I512/K2/E8` shape (honors the kernels' `%512` / Block2D `%128` /
FP4 `%512` gates), with **real non-uniform per-block scales** (weight 128×128, x 1×128, FP4 1×32):

| kernel | dtype (act × w) | scale | cos | max Δ |
|---|---|---|---|---|
| `gate_bf16_d2` | bf16 × fp8 | 128×128 | 0.999999 | 0.19% |
| `gate_fp8_d2` | fp8 × fp8 | 128×128 + 1×128 x | 0.999999 | 0.15% |
| `down_h2_d2` | bf16 × fp8 | 128×128 | 0.999999 | 0.31% |
| `down_fp4_h2` | bf16 × **fp4** | **1×32 MXFP4** | 0.999999 | 0.25% |

Deltas are pure bf16 output rounding. This validates each kernel's matmul + fp8/fp4 dequant + silu
+ router-weight math **and** the Block2D scale-index layout.

**Key finding (corrects an earlier assumption + B1):** the down FP4 path's MXFP4 `(1,32)` scale
needs **no kernel work** — CK's generic `load_block2d_scale<Block2D<1,32>, float>` handles any
`Block_N/Block_K`, and feeding it power-of-two float scales (matching FlyDSL's e8m0 range) makes
`down_fp4_h2` numerically exact. So B1's "exact FP4 scale match needs an e8m0 (1,32) kernel" is
overstated: the *values* already match with a `(1,32)` float scale; the only residual B1 gap is
scale **storage bytes** (float 4B vs e8m0 1B), which is a traffic-metric modeling choice, not a
kernel limitation.

**One informational (non-gated) leg — `gate_up_fp4`:** CK's gate_up FP4 is the slow packed
*scalar* path (dot2/NPerWarp=2 reject packed FP4) and loads weights through a tiled
`make_naive_tensor_view` whose in-memory pk_fp4 layout is swizzled (unlike `down_fp4_h2`'s raw
memcpy loads). A positional compare fails (`positional_cos≈0`) while the **value-set** matches
(`value_set_cos≈0.99`) — right arithmetic, permuted output positions. Reproducing that tile packing
is low-value (it's CK's secondary/slow FP4 path, and FP4 math is already validated exactly by
`down_fp4_h2`), so the validator reports it as INFO and excludes it from the PASS gate.

---

## 3. G9 comparison coverage matrix (completion gate)

Legend: ✅ FlyDSL+CK pair present in the of-record artifact for every B ·
⏳ pending the named step · ⛔ blocked (+reason).
Shapes: DeepSeek-V3 (H7168/I2048/E256/K8), MiniMax (H3072/I1536/E256/K8),
Qwen3Next-TP1 (H2048/I512/E512/K10). Each ✅ covers all **B∈{1,2,4,8,32}**.
Reconciled against the of-record `tickets/667/g9_compare.{md,csv}` (repeats=3); the exact
aiter/CK commits are recorded in that artifact's provenance header, not pinned here (they
change on every regen).

| Op | dtype (act × w) | DeepSeek-V3 | MiniMax | Qwen3Next-TP1 |
|---|---|---|---|---|
| gate_up | BF16-act × FP8-w | ✅ (B5) | ✅ | ✅ |
| gate_up | FP8-act × FP8-w | ✅ (B4) | ✅ (B4) | ✅ (B4) |
| gate_up | BF16-act × FP4-w | ✅ (A4) | ✅ (A4) | ✅ (A4) |
| down | FP8-w | ✅ (B5) | ✅ | ✅ |
| down | FP4-w | ✅ | ✅ | ✅ |

**Covered now: 15 of 15** (op×dtype×shape) cells are complete FlyDSL+CK pairs across all
5 batches (= **75 paired data points**). The last two holes closed 2026-08-17/18:
DeepSeek FP8 (`down`+`gate_up bf16-act`) via **B5** (E256 Tier-2 i64 base), and
`gate_up FP8-act × FP8-w` (all shapes) via **B4** (`gate_fp8_d2` peer). All cos=1.0000.

**G9 coverage is closed** — every op×dtype×act CK exposes now has a joined FlyDSL peer.

---

## 4. Dependencies (build order)
- **C1 (`compare.py`)** needs **A1, A2, B2, B3**.
- **Full FP8 coverage** (DeepSeek) — ✅ done (B5; K3 Tier-2 i64 per-expert base).
- **gate_up FP4 pair** — ✅ done (A4; needed a one-line CK gate_up kernel packed-stride fix).
- **gate_up FP8-act pair** — ✅ done (B4; FlyDSL `flydsl_warp_decode_gate_up_fp8act` peer of CK `gate_fp8_d2`).
- **Trustworthy ratios** need A1 (cold) ✅, B1 (scale) ✅, D3 (config) ✅, D1 (timing) ✅,
  and D4 (equiv) ✅ — **all prerequisites now met**; ratios are defensible for review.

## 5. Risks / watch-items
- **Cold flush** — *resolved (A1):* verified cold numbers drop vs warm on the cache-resident
  small Qwen shapes (e.g. `down_h2_d2` B=1 4977→2999 GB/s); no longer an open assumption.
- **Rotating-buffer OOM** — *obsolete (A1):* the `stream_config` `rotating_count_` path is a
  no-op on this launcher and is not used. The cold mechanism only grows a **KB-scale router-id
  buffer** (`rotate*B*K` int32), leaving the GB weight pool untouched — verified OOM-free at
  DeepSeek B=32. (The original concern was about `rotating_count_` deep-copying the weight
  pool, which this launcher never does.)
- **`total_traffic` drift:** pinned to CK commit `62e30c9098`; revisit if CK's byte/FLOP
  formulas change.
- **CK worktree carries a local patch (A4):** `warp_decode_gate_up_kernel.hpp` has a one-line
  `IsSupportedArgument` fix so packed-FP4 gate/up rows (`stride=H/2`) are accepted (mirrors the
  down kernel). The kernel *math* is unchanged. The fix is **committed on top of the pinned
  base**: worktree HEAD `c03392a91b8` = base `62e30c9098` + the A4 patch. Provenance now names
  both accurately — `compare.py`'s header prints `ck_worktree=c03392a91b8` (git) and the cpp's
  stderr line prints `base_commit=62e30c9098 patch=A4-gateup-fp4-packed-stride`.
  **Reproducibility:** `build_ck_bench.sh` pins a fresh worktree to `CK_COMMIT=c03392a91b`
  (= base + A4), so a clean rebuild checks out the patched commit and includes the `gate_up_fp4`
  row. The only residual caveat is that a pristine/upstream CK checkout wouldn't carry the fix
  until it is upstreamed.
- **Env reconciliation** (shared with the full-MoE track): one env must import FlyDSL +
  aiter; the CK side is a standalone binary so it's driven via subprocess.
- **Regime honesty:** ratios are only "apples-to-apples" once cold + scale + timing + config
  are aligned; report the config/regime next to every table.
- **FP4 scale-traffic bias (~6%, CK-favored):** CK FP4 uses a dummy PerTensor scale while
  FlyDSL streams a real e8m0 `(1,32)` scale (~6% of the FP4 weight bytes); CK reads ~0 scale
  bytes, so FP4-cell ratios are modestly CK-favored. Exact match needs e8m0 `(1,32)` kernel
  support (B1/D7); until then, document the caveat and trust the time ratio.

## 6. Status log
- 2026-08-18 — **D7 done (full) — CK outputs numerically cross-checked across FP8/BF16/FP4 with
  real non-uniform Block2D scales.** Extended the `CK_WD_VALIDATE` dump + `validate.py` to
  non-uniform per-block scales (weight 128×128, x 1×128, FP4 **1×32 MXFP4**) and added the FP4
  legs. **4 gated kernels cos=0.999999** (`gate_bf16_d2`, `gate_fp8_d2`, `down_h2_d2`,
  `down_fp4_h2`; max Δ ≤0.31% = bf16 rounding). Found the FP4 `(1,32)` scale needs **no kernel
  work** (generic `load_block2d_scale` + power-of-two float scales), which also downgrades B1's
  "needs e8m0 kernel" caveat to a pure storage-byte metric choice. `gate_up_fp4` (CK's slow packed
  scalar path, swizzled tiled weight layout) is reported INFO-only: value-set matches
  (`value_set_cos≈0.99`), positions permuted; FP4 math already validated by `down_fp4_h2`.
- 2026-08-18 — **D7-lite done — CK outputs numerically cross-checked (FP8 + BF16).** Added a
  `CK_WD_VALIDATE=1` dump path to `ck_bench_warp_decode.cpp` (host-quantized real inputs, one
  functional launch, dumps exact input bytes + GPU output for `gate_bf16_d2`/`gate_fp8_d2`/
  `down_h2_d2`) + a `validate.py` torch cross-check. All three cos=0.999999 at B4/H512/I512/K2/E8
  with uniform scales (superseded by the full-D7 non-uniform run above).
- 2026-08-18 — **B3 follow-up done — warm benches migrated to `compute_metrics`.** Dropped the
  inline `outputs`/`flops`/`wbytes`/`tbs` accounting from `bench_gate_up`, `bench_down`, and
  `bench_down_fp4`; each now derives TFLOPS/TB-s/%peak from `compute_metrics(..., weight_stream)`,
  so the byte/FLOP model is single-sourced across cold + warm + `compare.py`. Re-ran the
  exact-equivalence numeric check (60k sampled shape-cells, even & `%32`-divisible `INTER`):
  **0 mismatches** — recorded numbers byte-for-byte unchanged (only `us<=0` now yields `NaN`
  instead of `0.0`, never hit in practice).
- 2026-08-18 — **B4 done — FP8-activation gate_up (CK `gate_fp8_d2` peer).** Added
  `build_gate_up_fp8_act_module` + `flydsl_warp_decode_gate_up_fp8act` (FP8 e4m3 `x` +
  `Block2D<1,128>` activation scale, FP8 weight `Block2D<128,128>`, dot2; both scales fold
  after dot2 per K-block; reuses B5's Tier-2 i64 base). Correctness cos=1.0 at small/qwen/
  DeepSeek. Wired an `fp8act` field into `bench_gate_up_cold` and a `("gate_up","fp8","fp8")`
  cell into `compare.py`, pairing with CK `gate_fp8_d2`. Regenerated the of-record artifact:
  15 new fp8-act cells (3 models × 5 B) at cos=1.0000, ratios 0.75→0.95, ~35–79 %peak, **0
  n/a** — the comparison is now fully paired for every op×dtype×act CK exposes.
- 2026-08-17 — **B5 done — DeepSeek-E256 FP8 hole closed (K3 Tier-2 addressing).** Added a
  per-expert i64 base (`_ptr_rsrc_off`) to the FP8 `down`/`gate_up` builders, guarded by a
  build-time `num_experts` so it only fires at `E*I*H >= 2^31` (i32-safe path + all FP4 shapes
  untouched). Fixes the i32 voffset wrap: gate_up @ E=896 cos 0.02→0.999999; DeepSeek E=256
  `down`+`gate_up` cos→1.000000. Dropped the cold-harness FP8 skip (now gates on in-expert
  `H*I < 2^31`) and regenerated the of-record G9 artifact — **0 n/a cells** (was 10); the 10
  DeepSeek FP8 cells fill at cos=1.0000, ratios 0.60→1.03, ~69–74 %peak. Matrix now 15/15
  op×dtype×shape (×5 B = 75 points). Unblocks the Kimi-K3 E=896 follow-on.
- 2026-08-17 — **D4 done — functional-equivalence audit.** Cross-checked both harnesses by
  code inspection: gate_up epilogue (`silu(gate)·up`, SiLU on gate only, bf16) and down
  reduction (`Σ router_wt · (inter·w)`, bf16) are identical; CK `element_wise::Silu` =
  `x·sigmoid(x)` matches FlyDSL. FP8 uses matched `Block2D<128,128>` weight scale both sides.
  The only difference is FP4 weight-scale *granularity* (FlyDSL e8m0 `(1,32)` vs CK dummy
  PerTensor) — a documented traffic/granularity caveat (~6%, B1/§5), not a compute-graph
  difference, so cells stay comparable on time. No cell marked n/a. CK's math equivalence is
  by inspection (perf harness runs uninitialized weights / uniform rwts); a numerical output
  cross-check remains the optional **D7**. **This closes the D-track rigor/methodology prereqs
  (A1/B1/D1/D3/D4 ✅)** — time ratios are now defensible for review. Docs only.
- 2026-08-17 — **Plan hygiene pass (review items #2–#11).** Reconciled stale checkboxes and
  dependency lines: A3's C1-join sub-item ✅, C1 depends-on now shows B1/A4 done (only B4/B5
  left), §4 trustworthy-ratios shows only D4 open. Marked **D1 done** — flat `iters=1000`
  pinned in the D6 driver, and documented the FlyDSL(IQR-trimmed device)-vs-CK(mean) statistic
  difference (negligible at iters=1000; report both, ratio unaffected); corrected the stale
  "~11 s" sweep estimate to the measured ~22–25 s single-pass / ~66–75 s at repeats=3. Reworded
  §5 risks: cold-flush → *resolved (A1)*, rotating-buffer OOM → *obsolete (A1)* (the
  `rotating_count_` path is unused; only a KB router buffer grows). Nits: dropped the churny
  pinned aiter SHA from §3 (defer to the artifact header), reconciled the `cos≥0.99` cold-gate
  vs the `≥0.999` primitive-test constraint, and clarified the tuned-upside "footnote" wording
  (no tuned row is materialized). Docs only — no code/artifact change.
- 2026-08-17 — **CK provenance corrected (#1).** The cpp's stderr provenance previously
  hardcoded a bare commit that didn't reflect the A4 patch. Set it to
  `base_commit=62e30c9098 patch=A4-gateup-fp4-packed-stride`, rebuilt, and regenerated the
  artifact so both provenance fields agree: `ck_worktree=c03392a91b8` (git = base + A4 commit)
  and the explicit base+patch stderr line. Reframed the §5 risk note (patch is committed at
  `c03392a91b8`; `build_ck_bench.sh` pins `CK_COMMIT=c03392a91b` so a clean rebuild checks out
  the patched commit and includes `gate_up_fp4` — only a pristine/upstream CK lacks the fix).
- 2026-08-17 — **D3 config policy DONE.** Verified the FlyDSL defaults the benches actually
  use (no overrides in `bench_*_cold`): `serialize_dot2=True`, `kh_per_warp=auto(2)`,
  `prefetch=False`, `down_fp4 dot2_acc=4`, `gate_up_fp4 dot2_acc=1`, `down_fp8 split_k=1`, FP8
  `block2d(128,128)`. Expanded `compare.py`'s provenance header to enumerate both sides'
  config (FlyDSL defaults vs CK recommended variant, with the CK no-single-default asymmetry)
  and both fairness caveats (FP8-down CK-favored lower bound ~10-38%; FP4 ~6% scale-traffic
  bias). Regenerated `tickets/667/g9_compare.{md,csv}` so the of-record artifact is
  self-describing. ruff/py_compile clean.
- 2026-08-17 — **D2 coverage matrix DONE.** Reconciled §3 against the of-record artifact:
  **9/15 op×dtype×shape cells complete** (×5 batches = 45 paired points) — all FP4 legs, all
  MiniMax/Qwen FP8 legs. Two holes gate G9 completion: DeepSeek FP8 (down + gate_up bf16-act)
  → **B5**, and gate_up FP8-act×FP8-w (all shapes) → **B4**. Matrix now shows accurate
  ✅/⏳(step) marks with per-cell reasons and a completion-gate statement.
- 2026-08-17 — **A4 gate_up FP4 CK bench DONE.** Added `GUProbFP4`/`GUKernFP4` +
  `gate_up_fp4` bench block (packed `E*I*H/2` pools, PerTensor dummy scale, `stride=H/2`,
  NPerWarp=1 scalar path). Discovered the gate_up kernel's `IsSupportedArgument` rejected the
  packed stride (the down kernel had the `pk_fp4→hidden/2` exception, gate_up didn't); patched
  `warp_decode_gate_up_kernel.hpp` with the same one-line fix (kernel math unchanged, but the
  CK worktree now diverges from pinned `62e30c9098` — provenance note added in §5). Rebuilt;
  the join now fills the gate_up FP4 CK column: ratios 0.56–0.84 (FlyDSL ~1.2–1.7× faster,
  matching the expected 1.44–1.65× cold win), all `%peak≤100`, `cos=1.0000`. Regenerated
  `tickets/667/g9_compare.{md,csv}`.
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
- 2026-08-14 — **B3 done.** Added the shared `compute_metrics(method=weight_stream|
  total_traffic)` helper; routed both cold benches through it and verified `weight_stream`
  reproduces the prior TB/s & TFLOPS exactly. `total_traffic` pinned to CK `62e30c9098`.
- 2026-08-14 — **C1 done.** `tickets/667/harness/compare.py` drives CK (CSV) + FlyDSL cold
  benches, joins on dims, emits a ratio table (markdown + CSV) via the shared metrics helper;
  smoke-tested on GPU 6. Added `fp8_cos` to the cold dicts for the sanity column. n/a cells
  (gate_up FP4 CK → A4; DeepSeek FP8 → B5) render correctly.
- 2026-08-14 — **B1 done.** Switched the FlyDSL FP8 cold legs (down + gate_up) to
  `w_scale_mode="block2d", scale_block=(128,128)` to mirror CK's `Block2D<128,128>` (pools +
  refs + kernel calls). The timed kernel now does the same per-block scale work as CK;
  `fp8_cos=1.0000` on minimax & qwen3next (DeepSeek FP8 still gated on B5). ruff/py_compile
  clean. `compute_metrics` FP8 byte accounting unchanged (Block2D<128,128> scale bytes are
  negligible; keeps the B3 invariant).
- 2026-08-17 — **B1 A/B measured.** PerTensor-vs-Block2D<128,128> FP8 (same weights,
  iters=1000): the scale cost is entirely on `down` (+10–38%, worst minimax B=1); `gate_up`
  neutral (−1.6%..+2.2%). Recorded in B1 + a D3 conservative-lower-bound caveat. Also
  regenerated the full post-B1 compare table: all `cos=1.0000`, FP8 cells populated for
  minimax/qwen; flagged several `%peak>100%` on small FP8/large-B cells (later root-caused
  below to a routing asymmetry; that post-B1 artifact is superseded by the post-route one).
- 2026-08-17 — **Routing fairness fix (major; D1/D5).** Investigating the `%peak>100%` cells
  found the real cause was a **fairness bug**, not just a metric artifact: FlyDSL's cold
  `_router_group_list` had all B tokens share one TOPK expert set (`expand(B,TOPK)`), so a
  launch read only TOPK experts and reused them across B, while **CK reads B*TOPK distinct
  experts per launch** (`rids[i]=i%E` over `bk=B*TOPK`). So at B>1 FlyDSL read up to B× less
  weight than CK — the time ratio (not just `%peak`) was unfair, flattering FlyDSL. Rewrote
  `_router_group_list` to distinct-experts-per-token (`bk=B*TOPK`, `rotate=ceil(E/bk)`),
  byte-for-byte matching CK; dropped `n_route`; bounded the correctness gate to the first
  `_COS_CHK_TOKENS=4` tokens (per-token work is uniform) so the fp32 reference doesn't
  dequant the whole pool (would OOM at DeepSeek B=32). Verified `cos=1.0000` (FP4+FP8) and no
  OOM through DeepSeek B=32. Regenerated the compare table (then `g9_compare_postroute`,
  since superseded by the D6 of-record `g9_compare.{md,csv}`):
  **all `%peak≤100%`** (max 80.9%) and B>1 ratios corrected sharply toward parity, e.g.
  minimax down fp8 B=32 0.563→0.995, qwen gate_up fp8 B=32 0.430→0.922, deepseek down fp4
  B=32 0.596→0.863. New story: FlyDSL's edge is largest at B=1 (latency regime) and converges
  to parity as B grows (both bandwidth-bound on the same bytes). B=1 ratios ~unchanged
  (distinct==shared at B=1). `compute_metrics` unchanged — its B-scaled weight bytes were
  always right; the routing now matches them. ruff/py_compile clean.
- 2026-08-17 — **Clock lock not feasible → variance control (D1/D5).** gfx950 here exposes only
  discrete SCLK levels {500, 2400} MHz (no mid ~1700; `--setperfdeterminism 1700` silently
  no-ops, `--showsupportedclocks` empty), MCLK fixed at 2000, DVFS stable ~1789 under load.
  Forcing `high` (2400) would thermally clamp mid-run, so it's not more deterministic than
  `auto`. Decided to keep `auto` and control variance in D5 (warm-to-steady + effective-SCLK
  sampling + N-repeat spread). Post-route numbers stand as-is (auto clocks); of-record run
  just needs the D5 variance capture, not a clock lock.
- 2026-08-17 — **D5 variance capture DONE.** `compare.py` gained `--repeats N` (default 3;
  headline us = per-cell median, new `fly_spr%`/`ck_spr%` = 100·(max−min)/median), a
  `--noise-pct` (default 5%) noisy-cell flag, and a background `ClockSampler` that polls
  `rocm-smi --showgpuclocks` SCLK @0.25s and folds loaded (≥400 MHz) min/median/max +
  repeats into the provenance header. Regenerated the of-record table (repeats=3):
  effective **loaded SCLK median ~2391 MHz** under sustained load (near the top {2400} DPM
  level — the decode sweep pins it higher than the ~1789 matmul probe). Spread is **<1% for
  almost every cell**; CK is rock-steady (<0.5%). Only the small/fast `down` cells trip the
  5% flag (deepseek down fp4 B=1 ~7%, minimax down fp8 B=1/B=2 ~6–7%), matching D1's
  under-converged fast-cell caveat. ruff/py_compile clean.
- 2026-08-17 — **D6 reproducible driver + checked-in artifact DONE.** Added
  `tickets/667/harness/run_g9_compare.sh` (one command: optional `--build-ck`, then the
  of-record `compare.py` sweep on GPU 6 with `--iters 1000 --cold 20 --repeats 3`, writing
  `tickets/667/g9_compare.{md,csv}`; flags `--gpu/--repeats/--iters/--cold/--out-prefix` + a
  `-- <args>` passthrough). No clock-lock step (D1). Generated the canonical of-record
  `g9_compare.{md,csv}` and retired the transitional `g9_compare_postroute.{md,csv}`.
- 2026-08-14 — **B1/D7 refined.** Corrected B1's FP4 scale note from "perf-negligible" to a
  measured **~6% CK-favored** bias (CK PerTensor vs FlyDSL e8m0 `(1,32)`); added the risk
  entry and the exact-match-needs-kernel-work caveat. Documented the CK-real-weights steps in
  D7 (init/quant/FP4-pack+e8m0/dump-compare, ~2–3 d) and cross-linked B1↔D7.

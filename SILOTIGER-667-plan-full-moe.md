# SILOTIGER-667 — Full-MoE / AITER-default Comparison Plan (Living Document)

**Ticket:** [SILOTIGER-667] MoE decode warp-decode kernels (small-M): FP8 + MXFP4 gate_up/down
**Goal of this doc:** Track the work to measure the FlyDSL warp-decode MoE kernels against
**whatever AITER dispatches by default**, at the **full fused-MoE level** (routing + gate_up +
activation + down) — not the per-stage cold FlyDSL-vs-CK comparison (which is already of record in
`tickets/667/g9_compare.md`). This is a *living document*: update the status boxes and notes as work
progresses. Ticket description is in `SILOTIGER-667.md`.

---

## 1. Interpretation & scope (agreed)

- **The question this answers:** "how does FlyDSL compare vs whatever AITER dispatches by default"
  for a real fused MoE, end-to-end. This is complementary to — and distinct from — the per-stage
  cold FlyDSL-vs-CK G9 comparison already published in `tickets/667/g9_compare.md`.
- **Deliverable:** a reproducible full-MoE benchmark that emits a **3-(or-4-)way table** — AITER
  default vs `warp_decode_ext` (reference) vs FlyDSL (+ CK if a peer is later merged) — with the
  same default-vs-default config policy, provenance, and variance capture used for G9.
- **Anchor (do NOT build a harness):** extend the working copy at
  `tickets/667/bench/bench_moe_warp_decode.py` (copied from
  `/workspaces/rocm-libraries-wdec/bench/bench_moe_warp_decode.py` on
  `origin/users/samremes/ck/warp-decode`; **not** on `develop`). The original stays
  in the CK worktree; FlyDSL is added only on the aiter copy. Est **~1–2 days**
  (wrappers + scale/signature reconcile + env) vs ~4–6 to build from scratch.
- **Target HW:** gfx950 (CDNA4, wave64). Hardware available for run/bench.
- **Out of scope:** kernel-level optimization of the FlyDSL path (covered by the other SILOTIGER-667
  tracks); building a new CK full-MoE peer (only wired in *if* one is merged later).

## 2. Locked decisions

- **Test environment:** run all tests/benches in **`flydsl_venv`** **GPU 1**
  (`HIP_VISIBLE_DEVICES=1`) for clean cold-HBM numbers.
- **Perf methodology (production-representative):** perf numbers come **only** from `run_perftest`
  (IQR-trimmed torch-profiler **device** time — pure kernel), never ad-hoc `time.perf_counter`
  loops. Cold-HBM reads via **rotation over disjoint expert groups** so each timed iter streams
  weights cold from HBM (the representative decode number).
- **Code locations (FlyDSL side):** FlyDSL kernels live in
  `aiter/ops/flydsl/kernels/warp_decode_moe.py`; the registered entry points are in
  `aiter/ops/flydsl/warp_decode_moe.py`; the combined correctness+perf op_test is
  `op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`.
- **E8M0 convert rule (relied on when reconciling scales):** `cvt_scalef32_pk_bf16_{fp8,fp4}` applies
  only the **EXPONENT** of its f32 scale operand — pass `scale=1` for arbitrary per-tensor/per-token
  scales and fold the real scale into the f32 accumulator **after** dot2; feed only power-of-two
  (e8m0) block scales through the convert's scale operand.
- **Config policy = default-vs-default.** Every path (AITER default, `warp_decode_ext`, FlyDSL) runs
  its own default configuration — no per-path tuning asymmetry — matching the G9 comparison's policy,
  and the config actually used is recorded in the output.
- **Do not fork the anchor script's plumbing.** Reuse its weight build, prepack, routing, torch
  reference, correctness (cos/err), timing, GB/s, ratio, and CSV machinery as-is; FlyDSL is added as
  an additional path column, not a parallel harness.

## 3. What already exists in the anchor script (inventory)

`tickets/667/bench/bench_moe_warp_decode.py` already:

- builds shared weights and prepacks them into **BOTH** layouts:
  - the **AITER fused layout** — `shuffle_weight` (16,16), `w1`/`w2` concat, 128×128 block quant;
  - the **warp-decode flat scale layout**;
- runs **`fused_topk` + `aiter.fused_moe`** — the DEFAULT dispatch (`QuantType.per_1x128`);
- runs FlyDSL warp-decode paths (`flydsl_bf16` / `flydsl_fp8` / `flydsl_fp4`);
- CKTile `warp_decode_ext` (`wd_*`) was removed from this benchmarker;
- runs a **`torch_moe_blockscale`** reference + correctness (cos/err);
- emits **per-stage timings, GB/s, ratio, and CSV**.

So the remaining work is: add FlyDSL as another path, reconcile signatures/scales, and reconcile the
environment.

## 4. Phased plan & status

Status legend: [ ] todo · [~] in progress · [x] done

### Phase A — Environment reconcile (MAIN RISK)  [x]
- [x] Stand up ONE environment where `aiter` + `fused_moe`, the locally-built `warp_decode_ext`, and
      `flydsl` are all importable together. **Resolved in `flydsl_venv` on GPU 1:** one interpreter
      loads `aiter`/`fused_moe`, JIT `warp_decode_ext` from `tickets/667/bench/`, `flydsl` 0.3.2, and
      the registered `aiter.ops.flydsl` warp-decode entries (`gate_up`, `down_reduce`, `_fp8act`,
      `_fp4`). Two-process fallback is not needed.
- **Risk note:** this was the single largest unknown. Closed — unified env is achievable.

### Phase B — FlyDSL path wrappers  [x]
- [x] Add FlyDSL path wrappers mirroring the script's `wd_bf16_moe_block` / down, calling the
      registered `aiter.ops.flydsl` entries. Op mapping:
      - `wd_bf16` gate_up (BF16 act × FP8 w) ↔ `flydsl_warp_decode_gate_up` (block2d). **Wired as `flydsl_bf16`.**
      - down ↔ `flydsl_warp_decode_down_reduce` (block2d). **Wired.**
      - FP4 gate_up/down (`flydsl_..._fp4`) → **NEW** columns the script lacks. **Deferred to Phase D** (needs
        MXFP4 data-gen + E8M0 scale layout, not just a wrapper).
      - `wd_fp8` gate_up is FP8-ACTIVATION × FP8-w → **Wired as `flydsl_fp8`** via
        `flydsl_warp_decode_gate_up_fp8act` (block-scaled FP8 act).
- Smoke (flydsl_venv, GPU 1, deepseek B=1): `flydsl_bf16` err/cos `0.000/1.0000` vs torch ref;
  `flydsl_fp8` matches `wd_fp8` (`0.562/0.9994`) because both quantize activations.

### Phase C — Signature & scale-layout reconcile  [x]
- [x] Reconcile signatures + scale layout: FlyDSL uses `w_scale_mode="block2d"`,
      `scale_block=(128,128)`, `out=`. The script's `w_*_scale_wd` (`[E*N/128, K/128]`) **is**
      FlyDSL block2d: kernel `sidx = (e*N+j)//128 * (K/128) + k//128` equals that 2D view
      flattened (verified on deepseek gate/up/down). Adapters flatten to 1D f32 before the
      call. `router_ids` i32 / `router_wts` f32 asserted in the adapters. hip per-1x128
      `x_scale` `[B, HIDDEN/128]` flattens to FlyDSL `[B * HIDDEN/128]` (token, K-block).
- [x] E8M0 rule (§2) on the **wired FP8 paths**: kernels already pass `scale=1` into
      `cvt_scalef32_pk_bf16_fp8` and fold the bench's arbitrary f32 `pertoken_quant` block
      scales **after** dot2. Adapters must not recode those tensors as E8M0. FP4 (convert
      scale operand = E8M0 byte) is still unwired (Phase D).

### Phase D — Emit the comparison table  [x]
- [x] Produce the **3-way** table: AITER default vs `warp_decode_ext` vs FlyDSL (CK full-MoE peer
      not merged). Artifacts: `tickets/667/full_moe_compare.md` + `.csv`. Provenance header records
      gfx, venv, iters, and per-path default configs. Regime labeled **WARM-ish**. G9-style
      repeat median/spread not wired (single pass).
- [x] Add the FP4 FlyDSL column (`flydsl_fp4`): MXFP4 pack of the same bf16 weights via
      `per_1x32_f4_quant` (`shuffle=False`), E8M0 `scale_block=(1,32)` into the convert. Cos vs the
      FP8 torch ref is skipped (different quant).

### Phase E — Cold-read semantics  [x]
- [x] Fix the **warm-ish** cold semantics of `bench_moe_warp_decode.py`. Headline `core_us` (and
      stage times) now rotate **disjoint** expert groups: `rotate = ceil(E/(B*TOPK))`,
      `rids = (g*bk + arange(bk)) % E`, same as the FlyDSL/CK cold harness. Timed callables
      capture weights in a closure (`num_rotate_args=1`) so they are not deep-copied.
      `fused_topk` stays a separate `topk_us` column; cold **core** feeds rotating `topk_ids`
      into `fused_moe` / warp-decode / FlyDSL. `--regime warm` restores the Phase D path.
- [x] Same rotation on every headline path (aiter, wd_*, flydsl_*). Citeable table:
      `tickets/667/full_moe_compare.md` + `.csv` labeled **COLD**. Phase D WARM-ish checkpoint
      kept as `tickets/667/full_moe_compare_warm.md` + `_warm.csv`.

## 5. Design notes

### 5.1 Op mapping (FlyDSL ↔ script paths)

| Script path | FlyDSL entry | Notes |
|---|---|---|
| `wd_bf16` gate_up (BF16 act × FP8 w) | `flydsl_warp_decode_gate_up` (block2d) | direct peer |
| down | `flydsl_warp_decode_down_reduce` (block2d) | direct peer |
| FP4 gate_up/down | `flydsl_warp_decode_gate_up_fp4` / `_down_reduce_fp4` | wired as `flydsl_fp4`; E8M0 (1,32) |
| `wd_fp8` gate_up (FP8 act × FP8 w) | `flydsl_warp_decode_gate_up_fp8act` | block-scaled FP8 act; peer now exists |

### 5.2 Scale-layout reconcile

FlyDSL expects `w_scale_mode="block2d"`, `scale_block=(128,128)`, an explicit `out=`.
**Confirmed:** `w_*_scale_wd` `[E*N/128, K/128]` is that layout (row-major row-block × K-block).
Adapters flatten to 1D. Router tensors: `router_ids` i32, `router_wts` f32. FP8 weight/act
scales stay f32 and are folded post-dot2; convert always sees `1.0`.

### 5.3 Output

A single table per shape × B with columns for AITER default, `warp_decode_ext` (reference), and
FlyDSL (plus `flydsl_fp4`; CK omitted until a full-MoE peer merges), each with its own default
config recorded in provenance, plus cos/err against `torch_moe_blockscale` (FP8 paths), timing,
GB/s, and ratio — CSV + markdown. **Citeable table is COLD** (`full_moe_compare.md`); Phase D
WARM-ish checkpoint is `full_moe_compare_warm.md`.

## 6. Open questions / risks

- [closed] **Unified environment** (Phase A): `aiter`+`fused_moe`, `warp_decode_ext`, and `flydsl`
  coexist in `flydsl_venv` (Python 3.14.4, gfx950, `flydsl` 0.3.2). No two-process fallback.
- [closed] **Cold vs warm regime** (Phase E): default `--regime cold` uses disjoint-expert
  router rotation (FlyDSL/CK pattern). `--regime warm` keeps the labeled Phase D path.
- [closed] **FP4 columns:** `flydsl_fp4` wired; MXFP4 + E8M0 `(1,32)` from the same bf16 weights.
  No AITER/CK full-MoE FP4 peer in this table.
- [open] **CK peer:** the 4th column (CK) is included only if/when a CK full-MoE peer is merged; not a
  prerequisite for the AITER-vs-FlyDSL result.

## 7. Changelog

- 2026-09-09 — Removed CK `warp_decode_ext` from `tickets/667/bench/` (the
  samaario/warp-decode-moe JIT extension and `wd_*` paths). The full-MoE
  benchmarker is AITER `fused_moe` vs FlyDSL only; no CKTile include / hipify
  build. G9 per-stage FlyDSL-vs-CK harness under `tickets/667/harness/` is unchanged.
- 2026-09-09 — Phase E: disjoint-expert router rotation on the tickets/667 bench
  (`--regime cold` default). Citeable artifacts `tickets/667/full_moe_compare.md` + `.csv`
  (COLD, GPU 1, flydsl_venv). WARM-ish Phase D table kept as `full_moe_compare_warm.*`.
  deepseek B=8 `flydsl_fp8` hitch from D is gone under cold. G9-style repeats still not wired.
- 2026-09-09 — Phase D: headline table `tickets/667/full_moe_compare.md` (WARM-ish, GPU 1,
  flydsl_venv). Wired `flydsl_fp4` (E8M0 1x32). CK column omitted. deepseek B=8 flydsl_fp8
  total/core looks like a hitch (stage times don't add); do not cite that cell.
- 2026-09-09 — Phase C: `w_*_scale_wd` matches FlyDSL block2d indexing; adapters flatten
  to 1D f32 and assert router i32/f32. FP8 E8M0 rule already in the kernels (convert scale=1,
  fold f32 after dot2); do not recode bench scales as E8M0.
- 2026-09-09 — Phase B: wired `flydsl_bf16` / `flydsl_fp8` paths on the tickets/667 bench copy
  (block2d adapters around registered `aiter.ops.flydsl` entries). FP4 columns deferred to Phase D.
- 2026-09-09 — Phase A closed: unified `flydsl_venv` + GPU 1 loads aiter/`fused_moe`,
  `warp_decode_ext`, and FlyDSL warp-decode entries together. Two-process fallback not needed.
- 2026-09-09 — copied `bench_moe_warp_decode.py` + `warp_decode_ext/` to
  `tickets/667/bench/`; CK include points at `/workspaces/rocm-libraries-wdec`
  (override `WARP_DECODE_CK_INCLUDE`). Run from `flydsl_venv` on GPU 1. No FlyDSL
  path yet.
- 2026-09-09 — noted that `bench/bench_moe_warp_decode.py` lives on
  `origin/users/samremes/ck/warp-decode` (not `develop`).
- _init_ — plan spun out of the SILOTIGER-667 TODO "Full-MoE / AITER" track. Scope, locked decisions
  (env + perf methodology + code locations + E8M0 rule + default-vs-default policy), anchor-script
  inventory, phased plan (A env → B wrappers → C reconcile → D table → E cold-read semantics), design
  notes (op mapping, scale reconcile, output), and open risks recorded. No code changes yet.

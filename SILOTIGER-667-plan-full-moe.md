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
- **Anchor (do NOT build a harness):** extend the existing reference script
  `/workspaces/rocm-libraries/bench/bench_moe_warp_decode.py`. It already does the expensive full-MoE
  plumbing; we ADD FlyDSL as another path. Est **~1–2 days** (wrappers + scale/signature reconcile +
  env) vs ~4–6 to build from scratch.
- **Target HW:** gfx950 (CDNA4, wave64). Hardware available for run/bench.
- **Out of scope:** kernel-level optimization of the FlyDSL path (covered by the other SILOTIGER-667
  tracks); building a new CK full-MoE peer (only wired in *if* one is merged later).

## 2. Locked decisions

- **Test environment:** run all tests/benches in **`flydsl_venv`** (triton 3.6) on **GPU 6**
  (`HIP_VISIBLE_DEVICES=6`) for clean cold-HBM numbers. The default env's older triton blocks
  `import aiter` on the FlyDSL path.
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

`/workspaces/rocm-libraries/bench/bench_moe_warp_decode.py` already:

- builds shared weights and prepacks them into **BOTH** layouts:
  - the **AITER fused layout** — `shuffle_weight` (16,16), `w1`/`w2` concat, 128×128 block quant;
  - the **warp-decode flat scale layout**;
- runs **`fused_topk` + `aiter.fused_moe`** — the DEFAULT dispatch (`QuantType.per_1x128`);
- runs a **`warp_decode_ext`** (bartified reference) FP8/BF16 path;
- runs a **`torch_moe_blockscale`** reference + correctness (cos/err);
- emits **per-stage timings, GB/s, ratio, and CSV**.

So the remaining work is: add FlyDSL as another path, reconcile signatures/scales, and reconcile the
environment.

## 4. Phased plan & status

Status legend: [ ] todo · [~] in progress · [x] done

### Phase A — Environment reconcile (MAIN RISK)  [ ]
- [ ] Stand up ONE environment where `aiter` + `fused_moe`, the locally-built `warp_decode_ext`, and
      `flydsl` are all importable together. Today the script imports a locally-built `warp_decode_ext`
      in the default env, while FlyDSL needs `flydsl_venv` (triton 3.6). Resolve this first — it gates
      every later phase.
- **Risk note:** this is the single largest unknown. If a unified env is not achievable, fall back to
  a two-process split (FlyDSL numbers produced in `flydsl_venv`, joined into the script's CSV), and
  document the split in the output.

### Phase B — FlyDSL path wrappers  [ ]
- [ ] Add FlyDSL path wrappers mirroring the script's `wd_bf16_moe_block` / down, calling the
      registered `aiter.ops.flydsl` entries. Op mapping:
      - `wd_bf16` gate_up (BF16 act × FP8 w) ↔ `flydsl_warp_decode_gate_up` (block2d).
      - down ↔ `flydsl_warp_decode_down_reduce` (block2d).
      - FP4 gate_up/down (`flydsl_..._fp4`) → **NEW** columns the script lacks.
      - `wd_fp8` gate_up is FP8-ACTIVATION × FP8-w → the FlyDSL peer now **EXISTS**
        (`flydsl_warp_decode_gate_up_fp8act`, block-scaled FP8 act); wire it in.

### Phase C — Signature & scale-layout reconcile  [ ]
- [ ] Reconcile signatures + scale layout: FlyDSL uses `w_scale_mode="block2d"`,
      `scale_block=(128,128)`, `out=`. Verify the script's `w_*_scale_wd` (`[E*N/128, K/128]`) matches
      FlyDSL's block2d expectation; confirm `router_ids` i32 / `router_wts` f32.
- [ ] Apply the E8M0 rule (§2) wherever a scale is fed to a convert vs folded post-dot2.

### Phase D — Emit the comparison table  [ ]
- [ ] Produce the **3-(or-4-)way** table: AITER default vs `warp_decode_ext` (ref) vs FlyDSL
      (+ CK if later merged), reusing the default-vs-default config policy + provenance/variance
      capture from G9.
- [ ] Add the FP4 gate_up/down FlyDSL columns the script currently lacks.

### Phase E — Cold-read semantics  [ ]
- [ ] Fix the **warm-ish** cold semantics of `bench_moe_warp_decode.py`. Today it builds weights once
      per shape and reuses them across B/iters with a **fixed** router (`fused_topk` on random gating)
      and `run_perftest`'s default rotation → the selected experts stay MALL-resident where they fit,
      i.e. **WARM-ish** (the same caveat CK's default had before it was flipped to cold). For decode
      realism the weight reads should be **COLD**.
- [ ] Apply the same treatment as the FlyDSL cold harness (oversized pool + router rotation over
      **disjoint** expert groups, or a cache flush) so the full-MoE numbers are cold too.

## 5. Design notes

### 5.1 Op mapping (FlyDSL ↔ script paths)

| Script path | FlyDSL entry | Notes |
|---|---|---|
| `wd_bf16` gate_up (BF16 act × FP8 w) | `flydsl_warp_decode_gate_up` (block2d) | direct peer |
| down | `flydsl_warp_decode_down_reduce` (block2d) | direct peer |
| FP4 gate_up/down | `flydsl_..._fp4` | **new columns** the script lacks |
| `wd_fp8` gate_up (FP8 act × FP8 w) | `flydsl_warp_decode_gate_up_fp8act` | block-scaled FP8 act; peer now exists |

### 5.2 Scale-layout reconcile

FlyDSL expects `w_scale_mode="block2d"`, `scale_block=(128,128)`, an explicit `out=`. The script's
warp-decode scale tensors are `w_*_scale_wd` shaped `[E*N/128, K/128]` — confirm this matches
FlyDSL's block2d indexing before trusting any number. Router tensors: `router_ids` i32,
`router_wts` f32.

### 5.3 Output

A single table per shape × B with columns for AITER default, `warp_decode_ext` (reference), and
FlyDSL (plus CK if merged later), each with its own default config recorded, plus cos/err against
the `torch_moe_blockscale` reference, timing, GB/s, and ratio — written to the script's CSV. Label
the cold/warm regime explicitly (see Phase E).

## 6. Open questions / risks

- [open, HIGH] **Unified environment** (Phase A): can `aiter`+`fused_moe`, `warp_decode_ext`, and
  `flydsl` coexist in one interpreter? If not, define the two-process fallback + CSV join.
- [open] **Cold vs warm regime** (Phase E): the anchor script is warm-ish by construction; decide
  whether to make it cold via disjoint-expert rotation / cache flush, or to ship warm-labeled numbers
  first and add cold as a follow-up. Either way the regime must be labeled.
- [open] **FP4 columns:** the script has no FP4 path today; adding the FlyDSL FP4 gate_up/down columns
  is net-new plumbing (data-gen + scale layout), not just a wrapper.
- [open] **CK peer:** the 4th column (CK) is included only if/when a CK full-MoE peer is merged; not a
  prerequisite for the AITER-vs-FlyDSL result.

## 7. Changelog

- _init_ — plan spun out of the SILOTIGER-667 TODO "Full-MoE / AITER" track. Scope, locked decisions
  (env + perf methodology + code locations + E8M0 rule + default-vs-default policy), anchor-script
  inventory, phased plan (A env → B wrappers → C reconcile → D table → E cold-read semantics), design
  notes (op mapping, scale reconcile, output), and open risks recorded. No code changes yet.

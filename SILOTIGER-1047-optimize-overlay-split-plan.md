# SILOTIGER-1047 — Optimize K2 overlay split

Close family A FlyDSL sparse GQA **overlay split** vs **live AMD**. Overlay
is the retained K2 variant: one MMA scratch (K then V), transposed
`K @ Q^T` so QK C is already PV-A, softmax m/l in registers, no
C/P/metadata LDS.

Parent: [SILOTIGER-1040](https://amd.atlassian.net/browse/SILOTIGER-1040).
Ticket: [SILOTIGER-1047](https://amd.atlassian.net/browse/SILOTIGER-1047).
Ticket-wide plan: `SILOTIGER-1047-plan.md` (K2 keep-gate and do-not-retry
live there). Prior campaign:
`SILOTIGER-1047-reduce-bank-conflicts-and-wait-barrier-plan.md` (C/P/live
LDS path; do not continue its unchecked phases here).

**Kernel name.** Living name is **K2 overlay**. Source is still
`aiter/ops/flydsl/kernels/qsa/k2_family_a.py`. The HSACO symbol is still
`qsa_k2_family_a_port_split_ps16_bn{16,64}_…` until a rename lands. Prefer
`qsa_k2_family_a_overlay_split` if the symbol is renamed. Traces live under
`tickets/1047/tmp/k2_overlay_att/`.

**Bar:** width-2051 `L=512` family A GQA on GPU 6 / gfx950 vs live
`qsa_sparse_paged_attention`. Headline is overlay-split wall clock
(`M=1` / `M=8` / `M=512`). Retained mixed-gate medians (2026-09-22):
FlyDSL **16.47 / 20.29 / 318.13** µs vs AMD **9.93 / 13.30 / 202.80** µs
(~1.66× / ~1.53× / **~1.57×**). Merge is out of scope (already ahead on
the old port). ATT CU0 wave span on overlay is 1.57× decode / 1.61×
prefill vs AMD at matched 8 waves/CU.

Work the phases **in order**. Later items assume earlier ones have landed.
Leave checkboxes unchecked until that item is done; paste tables and notes
under the relevant phase as evidence. One ISA line per experiment; ≥3%
keep-gate or revert; stage then one-line commit when that is the ticket
convention.

## Progress

- [ ] 0. Overlay PMC (MFMA/conflict/VMEM per wave vs AMD)
- [ ] 1. Prefill V overlay store map
- [ ] 2. Prefill packed K32 PV (`cvt_pk`); decode MFMA unchanged
- [ ] 3. Reuse gather `live` for softmax (optional; after a keep)
- [ ] 4. Drop leftover `v_perm` on the pack path (optional; after 1)
- [ ] 5. Stop: keep-gate dry **or** prefill store/MFMA in AMD’s band

## Locked decisions

These locks apply to **this campaign** unless a later note explicitly
supersedes them. Ticket-wide locks in `SILOTIGER-1047-plan.md` still
apply. Conflict/wait do-not-retry in that parent file and in
`SILOTIGER-1047-reduce-bank-conflicts-and-wait-barrier-plan.md` still
apply; this campaign does not retry those.

### Skills

- **Skills (read, do not recall).** Before writing or reviewing FlyDSL
  for this ticket, Read
  `.claude/skills/flydsl-kernel-authoring/SKILL.md` and follow it.
  For `op_tests/test_flydsl_qsa.py`, also Read
  `.claude/skills/aiter-op-test/SKILL.md`.
  Do not start kernel code from memory of those skills.
  Cleanup/modernization of existing kernels uses
  `flydsl-kernel-code-cleanup`, not authoring, and is not a phase
  of this plan unless a later lock says otherwise.

### Test environment

- **Test environment:** run all tests/benches in **`flydsl_venv`** **GPU 6**
  (`HIP_VISIBLE_DEVICES=6`).
- **Compile cache.** After kernel-source edits, run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or clear `~/.flydsl/cache`) so a stale
  HSACO cannot mask a bad rewrite.
- **`CACHE=0` needs `ROCM_PATH=/root/.flydsl/toolkit`.** FlyDSL runs
  `gpu-module-to-binary` with an empty `toolkit=`, so MLIR resolves
  `ld.lld` from `ROCM_PATH`, which is unset in this container. Every
  *real* compile then dies with `lld invocation failed` — the ROCm wheel
  puts the linker at `_rocm_sdk_core/lib/llvm/bin`, not the `llvm/bin`
  MLIR appends. `CACHE=1` hides this by hitting the disk cache. Always
  run the HEAD control with `CACHE=0` before blaming a kernel edit.
- **Gate (after a kernel exists).** Both layers:
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    ROCM_PATH=/root/.flydsl/toolkit \
    python3 -m pytest op_tests/test_flydsl_qsa.py -q
  HIP_VISIBLE_DEVICES=6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    ROCM_PATH=/root/.flydsl/toolkit \
    python3 op_tests/test_flydsl_qsa.py
  ```
  Paths may move; keep this snippet in sync. Overlay keep-gate benches
  use `tickets/1047/tmp/profile_k2_gqa.py` (or the parent width-2051
  row) vs live AMD, not the full family-B sweep.

### Scope and bar

- **Overlay split only.** Do not spend keep-gate budget on merge, indexer
  K1, family B, occupancy of unused CU slots, or undoing overlay back to
  C/P/metadata LDS.
- **Family A GQA, live AMD is the gate.** Width-2051 `L=512`
  `M∈{1,8,512}` vs vendored Triton `_qsa_sparse_paged_gqa_splitk`.
  `err=0` vs the oracle (`checkAllclose` `1e-2` as in the parent plan).
- **Keep gate is 3% wall-clock** on that width-2051 row, with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` after kernel-source edits. Flat or a
  loss reverts; paste the row and restore HEAD.
- **One path per experiment** unless the change is shared (V map used by
  both QK-K and PV-B). Decode stays `BLOCK_N=16` / 256 threads / 32
  splits. Prefill stays `BLOCK_N=64` / 128 threads / `splits=1`.
- **Do not reopen occupancy forks**, extra decode splits, 256-thread
  BN64 prefill, or “more K32” QK unroll.
- **Do not chase extra MFMA.** Overlay prefill is already **64** static
  MFMA vs AMD **48** (32 K32 QK + 32 K16 PV vs 48 K32). Reducing PV to
  K32 (`cvt_pk`) to **match** AMD is in scope (phase 2). Adding QK K32
  or extra tiles is not.
- **Inherit K2 do-not-retry** from `SILOTIGER-1047-plan.md` and the
  conflict/wait campaign. In particular do **not** retry: hoist V
  `buffer_load` before K-publish `lgkmcnt(0)` / `s_barrier`; split that
  barrier into waitcnt + raw `s_barrier`; next-tile K/V in the
  `range(..., init=)` SSA; decode `ds_write_b64` on `[BLOCK_N, D]`; XOR
  as a replacement for K `D+8`; live/phys LDS rows; token-major ↔
  row-major V swaps; NT gathers.

### Overlay contract

Keep this shape unless a later note explicitly supersedes it. A miss on
phases 1–4 is not a reason to restore C/P LDS.

- Compute **`K @ Q^T`**. QK C is token-major in-lane and feeds PV-A
  without P-LDS or a bpermute transpose of P.
- **One MMA scratch.** K and V overlay the same LDS; barrier before
  overwrite. gfx950 emitted sizes at retain: decode **8192 B**, prefill
  **32768 B** (rocprof dispatch padded to 8960 / 33280).
- Softmax **m/l stay in registers**. Four `shuffle_idx` only for
  alpha/epilogue to PV-C’s four-head map.
- **No C/P/metadata LDS.** Do not reintroduce those rows to chase a
  decode-only win.

### Profiling

- **ATT GPU index is physical.** With `HIP_VISIBLE_DEVICES=6` pass
  `--att-gpu-index 6`. Index `0` traces idle GPU 0 and yields empty
  decode.
- **Decode ATT uses `M=8`.** Same decode kernel as `M=1`
  (`bn16_blk256_ns32`). Prefill ATT uses `M=512` (`bn64_blk128_ns1`).
- **PMC pass that fits hardware:** `SQ_WAVES SQ_BUSY_CYCLES
  SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT SQ_INSTS_VALU SQ_INSTS_MFMA
  SQ_INSTS_VMEM SQ_INSTS_SALU`. Adding `SQ_INSTS_LDS` to that group is
  error 38. Report **per wave**. Overlay prefill MFMA/wave is unknown
  until phase 0; do not assume the old port’s 1.00× AMD.
- **rocprof under JIT:** fill FlyDSL cache then
  `FLYDSL_RUNTIME_ENABLE_CACHE=1` for ATT/PMC; keep-gate benches stay
  `CACHE=0`.
- **Stall unit `k` is thousand ATT `Stall`**, for ranking PCs, not
  microseconds.

### After each experiment

- Pytest `op_tests/test_flydsl_qsa.py` as in the gate snippet (focused
  decode+prefill is enough when that is the parent convention).
- Width-2051 `L=512` `M=1/8/512` vs live AMD.
- On a keep: PMC per-wave conflict / wait LDS / MFMA / VMEM. On a keep
  that claims a store-map or waitcnt win: ATT stall histogram on GPU
  index 6 (`tickets/1047/tmp/k2_overlay_att/` or a dated sibling).
- Evidence goes under that phase. Reverts go to do-not-retry here
  **and** in `SILOTIGER-1047-plan.md`.

## Subtasks

### 0. Overlay PMC (MFMA/conflict/VMEM per wave vs AMD)

ATT (2026-09-22, GPU 6, CU 0, `CACHE=1` after fill) is already in
`tickets/1047/tmp/k2_overlay_att/`. Occupancy matches AMD (8 waves/CU).
Dispatch: decode LDS/VGPR **8960 / 104** vs AMD **8960 / 112**; prefill
**33280 / 232** vs **33280 / 248**. Decode MFMA hits match (**12 / 384**).
Prefill static MFMA is **64 vs 48**. Prefill `ds_write_b128` is 14.6% of
latency (AMD stores 9.4%). Prefill buffer-load hits **8512 vs 25376**.
No overlay PMC yet; the port-era “MFMA/wave = 1.00×” table does not
apply.

- [ ] PMC per-wave FlyDSL vs AMD at `M=1` and `M=512` (same counter
      group as the campaign lock).
- [ ] **Done when:** a table of busy / conflict / wait-LDS / MFMA / VALU
      / VMEM per wave is pasted here and is the comparison point for
      later phases.

### 1. Prefill V overlay store map

Hottest prefill instructions are `ds_write_b128` on one addr at **+32 B**
offsets (96…448). K on gfx950 already uses `SwizzleType.get(3, 3, 3)` on
stride `D`. V overlay is still `(BLOCK_N, span):(D, 1)` with
`UniversalCopy128b`.

**ISA line:** V store **and** PV-B read use the same composed map as K,
or an AMD-like strided `write2st64` view. Not a second physical tile.
Not decode `ds_write_b64`. Prefill-only unless the map is shared.

- [ ] Confirm ISA: V `ds_write` offsets stop being a 32 B ladder on one
      VGPR.
- [ ] Focused pytest; keep-gate `M=1/8/512`. `M=512` must move ≥3%;
      decode must not regress.
- [ ] On keep: ATT `ds_write*` latency share + PMC conflict/wave.
- [ ] **Done when:** kept with evidence, or reverted and listed under
      do-not-retry.

### 2. Prefill packed K32 PV (`cvt_pk`); decode MFMA unchanged

Decode already matches AMD MFMA. Prefill runs **32 K16 PV**; AMD packs P
with `v_cvt_pk_bf16_f32` into K32 PV (**48** K32 total).

**ISA line:** pack each 4-token P with `cvt_pk` (or equivalent) and run
PV as K32 so static MFMA goes **64 → 48**. Softmax stays in the
transposed register C map. No P-LDS. Do not change decode (12 MFMA).

- [ ] ISA: prefill K16 PV count → 0; MFMA/wave ≈ AMD at `M=512`.
- [ ] Focused pytest; keep-gate. Decode must stay flat.
- [ ] **Done when:** kept with evidence, or reverted and listed under
      do-not-retry.

### 3. Reuse gather `live` for softmax (optional; after a keep)

Prefill has **359× `v_cndmask_e32`** vs AMD **48**. Softmax re-walks
`indices` / `page_table` per `n` even though gather already has `live`
for that column.

**ISA line:** `score_live` is the gather `live` for `n` (shuffle if the
owner differs). No new LDS. Do not retry per-thread translate.

Skip unless phase 1 or 2 kept. Likely small.

- [ ] Skip unless a prior overlay keep landed.
- [ ] Focused pytest; keep-gate.
- [ ] **Done when:** kept, reverted and DNR, or explicitly skipped.

### 4. Drop leftover `v_perm` on the pack path (optional; after 1)

Prefill **168× `v_perm_b32`** (3.7% latency) packs 8-wide gathers into
LDS. Phase 1 may remove it. If it remains after a phase-1 keep, one
isolated change: store the gather vector without that shuffle, same
logical V map.

Do not start here.

- [ ] Skip unless phase 1 kept and ATT still ranks `v_perm`.
- [ ] **Done when:** kept, reverted and DNR, or explicitly skipped.

### 5. Stop: keep-gate dry **or** prefill store/MFMA in AMD’s band

- [ ] Stop when no remaining item can pass 3% without reopening a DNR,
      **or** overlay ATT shows prefill `ds_write*` latency near AMD’s
      ~9% band **and** prefill MFMA/wave ~1× AMD, with decode join share
      still ~8%.
- [ ] Do not spend the next experiment on VMEM-across-K-publish,
      occupancy, or more QK math.

## Do not retry (this campaign)

Paste misses here **and** in `SILOTIGER-1047-plan.md`. Inherited DNR is
not repeated unless an overlay retry is proposed.

- *(empty — no overlay-era miss yet)*

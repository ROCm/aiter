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

- [x] 0. Overlay PMC (MFMA/conflict/VMEM per wave vs AMD)
- [x] 1. Prefill V overlay store map
- [x] 2. Prefill packed K32 PV (`cvt_pk`); decode MFMA unchanged (miss; DNR)
- [x] 3. Reuse gather `live` for softmax (optional; after a keep)
- [x] 4. Drop leftover `v_perm` on the pack path (optional; after 1)
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
  error 38. Report **per wave**. Overlay prefill MFMA/wave is **1.33×**
  AMD (2112 vs 1584 = 64/48 static). Decode MFMA/wave is 2.00× only
  because AMD launches 2× waves; totals match.
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

- [x] PMC per-wave FlyDSL vs AMD at `M=1` and `M=512` (same counter
      group as the campaign lock).
- [x] **Done when:** a table of busy / conflict / wait-LDS / MFMA / VALU
      / VMEM per wave is pasted here and is the comparison point for
      later phases.

Measured 2026-09-23 on GPU 6 / gfx950. Last split dispatch. Traces:
`tickets/1047/tmp/k2_overlay_pmc/`. rocprof LDS on the overlay kernel is
**8192 / 32768** (decode / prefill). AMD CSV `LDS_Block_Size` is 0
(Triton under-report; ATT still 8960 / 33280).

| | decode `M=1` FlyDSL | decode `M=1` AMD | ratio | prefill `M=512` FlyDSL | prefill `M=512` AMD | ratio |
|--|--:|--:|--:|--:|--:|--:|
| grid / waves | 16384 / 256 | 32768 / 512 | 0.50× waves | 131072 / 2048 | 131072 / 2048 | 1.00× |
| WG / LDS / VGPR | 256 / 8192 / 100 | 256 / 0 / 108 | | 128 / 32768 / 100 | 128 / 0 / 120 | |
| busy / wave | 3877 | 869 | **4.46×** | 11995 | 7522 | **1.59×** |
| conflict / wave | 1258 | 81 | **15.6×** | 69696 | 10560 | **6.60×** |
| wait-LDS / wave | 109 | 50 | **2.19×** | 24544 | 9264 | **2.65×** |
| MFMA / wave | 48.4 | 24.2 | 2.00× | 2112 | 1584 | **1.33×** |
| VALU / wave | 1203 | 641 | 1.88× | 33343 | 30554 | 1.09× |
| VMEM / wave | 81 | 39 | 2.07× | 2218 | 3180 | **0.70×** |
| SALU / wave | 212 | 109 | 1.94× | 2598 | 3646 | 0.71× |

Decode **total** MFMA is identical (**12384**). The 2.00× MFMA/wave (and
~2× VALU/VMEM/SALU) is AMD’s extra-split grid (DNR), not extra math per
token. Decode busy **total** is still **2.23×** AMD (993k vs 445k), so
the per-wave 4.46× is half occupancy artifact and half longer waves.
Conflict/wave **15.6×** is the leftover decode bank cost on the overlay
tile.

Prefill grids match. Busy **1.59×** matches the ~1.57× wall / ATT span.
MFMA/wave **1.33×** is exactly **64 / 48** static K32 (phase 2). Conflict
**6.60×** and wait-LDS **2.65×** are the phase-1 store-map target. VMEM
**0.70×** is fewer outstanding loads (DNR to hoist before K-publish).

**This table is the comparison point for later phases.** Do not treat
decode MFMA 2.00×/wave as a reason to add splits or drop overlay QK.

### 1. Prefill V overlay store map

Hottest prefill instructions are `ds_write_b128` on one addr at **+32 B**
offsets (96…448). K on gfx950 already uses `SwizzleType.get(3, 3, 3)` on
stride `D`. V overlay is still `(BLOCK_N, span):(D, 1)` with
`UniversalCopy128b`.

**ISA line:** V store **and** PV-B read use the same composed map as K,
or an AMD-like strided `write2st64` view. Not a second physical tile.
Not decode `ds_write_b64`. Prefill-only unless the map is shared.

- [x] Confirm ISA: V `ds_write` offsets stop being a 32 B ladder on one
      VGPR.
- [x] Focused pytest; keep-gate `M=1/8/512`. `M=512` must move ≥3%;
      decode must not regress.
- [x] On keep: PMC conflict/wave (ATT stall histogram not re-taken).
- [x] **Done when:** kept with evidence, or reverted and listed under
      do-not-retry.

Kept **prefill-only** XOR (2026-09-23). Shared XOR (decode+prefill
`make_k_lds_view` on V) passed `checkAllclose` 1e-2 but PV-B must be
composed XOR on `(16,16):(1,D)`, not `(D,1)` (that layout was
`err≈0.985`). Decode wall-clock vs overlay HEAD **16.47** looked
+22% on a noisy GPU (live AMD `M=1` **36.94** vs retained **9.93**);
decode PMC vs overlay HEAD is identical, so the keep is prefill-only
to match the ISA line.

**Code.** Prefill `use_k32` V store uses `make_k_lds_view` +
`UniversalCopy128b`. Prefill PV-B is `SwizzleType.get(3,3,3)` on
`(16,16):(1,D)` from `k_arr.ptr`. Decode V stays unswizzled
`(BLOCK_N, span):(D, 1)` + `UniversalCopy64b`; decode PV-B stays
`(16,16):(1,D)` on `v_lds`.

**ISA** (`tickets/1047/tmp/k2_phase1_prefillxor_isa/`). Prefill
`v_bitop3` **69**, `ds_write_b128` **32** on **v117–v131** (one leftover
`v117` `offset:32`, not the old single-VGPR +32 B ladder). Decode
`v_bitop3` **9**, `ds_write_b128` **4**, no extra V XOR.

**Correctness.** `test_k2_family_a_decode_matches_oracle` and
`test_k2_family_a_prefill_matches_oracle` **2 passed**.

**Keep-gate** vs overlay HEAD **16.47 / 20.29 / 318.13** µs (`CACHE=0`,
GPU 6). FlyDSL medians **20.14 / 20.18 / 261.54**. `M=512` **−17.8%**.
`M=8` flat. `M=1` wrapper is in the same noisy band as AMD decode
(**36.94** vs **9.93**). Same-session AMD prefill **201.21** matches
retained **202.80**.

**PMC** (`tickets/1047/tmp/k2_phase1_prefillxor_pmc/`) vs overlay HEAD
(`k2_overlay_pmc/`), last split dispatch, per wave:

| | decode overlay | decode phase 1 | prefill overlay | prefill phase 1 | prefill Δ |
|--|--:|--:|--:|--:|--:|
| VGPR | 100 | 100 | 100 | 124 | |
| busy | 3877 | 3916 | 11995 | 9808 | **−18.2%** |
| conflict | 1258 | 1258 | 69696 | 44352 | **−36.4%** |
| wait-LDS | 109 | 110 | 24544 | 9781 | **−60.1%** |
| MFMA | 48.4 | 48.4 | 2112 | 2112 | 0 |
| VALU | 1203 | 1203 | 33343 | 33510 | +0.5% |
| VMEM | 81 | 81 | 2218 | 2218 | 0 |

Decode instruction counts match overlay HEAD. Prefill conflict **6.60×
→ 4.20×** AMD (10560/wave). Wait-LDS **2.65× → 1.06×** AMD (9264/wave).
MFMA still **1.33×** (phase 2). ATT stall histogram not re-taken this
pass; PMC wait-LDS is the store-map keep signal.

### 2. Prefill packed K32 PV (`cvt_pk`); decode MFMA unchanged

Decode already matches AMD MFMA. Prefill runs **32 K16 PV**; AMD packs P
with `v_cvt_pk_bf16_f32` into K32 PV (**48** K32 total).

**ISA line:** pack each 4-token P with `cvt_pk` (or equivalent) and run
PV as K32 so static MFMA goes **64 → 48**. Softmax stays in the
transposed register C map. No P-LDS. Do not change decode (12 MFMA).

- [x] ISA: prefill K16 PV count → 0; MFMA/wave ≈ AMD at `M=512`.
- [x] Focused pytest; keep-gate. Decode must stay flat.
- [x] **Done when:** kept with evidence, or reverted and listed under
      do-not-retry.

Missed the 3% keep-gate (2026-09-23). Reverted to phase-1 HEAD
`a4ec4c938`. Oracle **2 passed**. Prefill ISA
(`tickets/1047/tmp/k2_phase2_k32pv_isa/`): **0** `v_mfma_f32_16x16x16`,
**48** `v_mfma_f32_16x16x32`, **40** `v_cvt_pk_bf16_f32`. Decode ISA
stayed **8** K32 QK + **4** K16 PV. PMC `M=512`
(`tickets/1047/tmp/k2_phase2_k32pv_pmc/`): MFMA/wave **2112 → 1584**
(1.00× AMD), conflict unchanged **44352**, wait-LDS **9781 → 9280**,
busy **9808 → 9865**, VGPR **124**. Keep-gate vs phase 1
**20.14 / 20.18 / 261.54**: FlyDSL **20.32 / 20.41 / 259.26** (`M=512`
**−0.9%**). Prefill is still LDS-bound; matching AMD’s MFMA count does
not move wall clock 3%. Keep register `.to(BFloat16)` P and K16 PV
gemm.

### 3. Reuse gather `live` for softmax (optional; after a keep)

Prefill has **359× `v_cndmask_e32`** vs AMD **48**. Softmax re-walks
`indices` / `page_table` per `n` even though gather already has `live`
for that column.

**ISA line:** `score_live` is the gather `live` for `n` (shuffle if the
owner differs). No new LDS. Do not retry per-thread translate.

Skip unless phase 1 or 2 kept. Likely small.

- [x] Skip unless a prior overlay keep landed.
- [x] Focused pytest; keep-gate.
- [x] **Done when:** kept, reverted and DNR, or explicitly skipped.

Kept (2026-09-23). Softmax `score_live` is `shuffle_idx` of gather
`live` for column `n` (lane `n` owns `tid % BLOCK_N`). No
`indices`/`page_table` re-walk. Oracle **2 passed**. Prefill
`v_cndmask_b32` **295 → 264**; `ds_bpermute` **8 → 24**.

**Keep-gate** vs phase 4 **20.38 / 20.29 / 250.57** µs: FlyDSL
**20.47 / 20.30 / 236.93**. `M=512` **−5.4%**. Decode median flat
(one noisy 30.87 µs run).

**PMC** (`tickets/1047/tmp/k2_phase3_live_pmc/`) vs phase 4, per wave:

| | decode p4 | decode p3 | prefill p4 | prefill p3 |
|--|--:|--:|--:|--:|
| VGPR | 128 | 108 | 124 | 124 |
| busy | 3458 | 3425 | 9568 | 8889 |
| conflict | 1258 | 1258 | 44352 | 44352 |
| wait-LDS | 330 | 346 | 16127 | 27726 |
| MFMA | 48.4 | 48.4 | 2112 | 2112 |
| VALU | 1009 | 815 | 20805 | 15012 |
| VMEM | 81 | 49 | 2218 | 1162 |

Prefill VALU **−28%**, VMEM **−48%** (dropped softmax table walks).
Conflict unchanged. Wait-LDS rose again; wall clock still keeps.

### 4. Drop leftover `v_perm` on the pack path (optional; after 1)

Prefill **168× `v_perm_b32`** (3.7% latency) packs 8-wide gathers into
LDS. Phase 1 may remove it. If it remains after a phase-1 keep, one
isolated change: store the gather vector without that shuffle, same
logical V map.

Do not start here.

- [x] Skip unless phase 1 kept and ATT still ranks `v_perm`.
- [x] **Done when:** kept, reverted and DNR, or explicitly skipped.

Kept (2026-09-23). `live.select` on the 8-wide gather vector replaces
per-element `to(f32)` / `from_elements` pack on K and V LDS stores
(decode 64-bit V and gfx942 V included). XOR V map unchanged. Oracle
**2 passed**. Prefill ISA `v_perm_b32` **168 → 40**; decode **50 → 34**.

**Keep-gate** vs phase 1 **20.14 / 20.18 / 261.54** µs: FlyDSL
**20.38 / 20.29 / 250.57**. `M=512` **−4.2%**. Decode wrapper flat
within noise.

**PMC** (`tickets/1047/tmp/k2_phase4_noperm_pmc/`) vs phase 1, per wave:

| | decode p1 | decode p4 | prefill p1 | prefill p4 |
|--|--:|--:|--:|--:|
| VGPR | 100 | 128 | 124 | 124 |
| busy | 3916 | 3458 | 9808 | 9568 |
| conflict | 1258 | 1258 | 44352 | 44352 |
| wait-LDS | 110 | 330 | 9781 | 16127 |
| MFMA | 48.4 | 48.4 | 2112 | 2112 |
| VALU | 1203 | 1009 | 33510 | 20805 |

Prefill VALU **−38%**. Conflict unchanged; wait-LDS rose (decode VGPR
128). Wall clock still keeps on `M=512`.

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

- Prefill packed K32 PV (`v_cvt_pk_bf16_f32` + `MFMA 16×16×32` so
  static MFMA 64→48). ISA and MFMA/wave matched AMD; `M=512` only
  **−0.9%** vs phase-1 XOR. Kernel restored.
- The first AMD-like pair-store attempt was incomplete, not a DNR.
  Rank-matched destination layout compiled to `ds_write2_b64`, and its
  nested tiled-MMA `partition_S` was not the inverse of the store
  partition (`err≈0.985`). Scalar inverse reads proved the permutation
  but regressed `M=512` to **374.88 µs**.

  **Retained replacement (2026-09-23).** The complete prefill V path
  directly emits **16× `ds_write2st64_b64 offset1:16`** and directly
  invokes `LDSReadTrans16_64b` with its per-lane source address
  (**32× `ds_read_b64_tr_b16`**). Its specialized blocked permutation is
  `p(n,d) = (d&3) + 4*(n&31) + 128*(d>>3) + 4096*((d>>2)&1)
  + 8192*(n>>5)`: low token bits spread stores over LDS banks, while
  dimension bit 2 selects the 8192-byte `st64` pair region.

  Decode and prefill oracle tests passed. Same-session `M=512` median
  **247.92 → 229.50 µs** (**−7.4%**); the earlier campaign baseline
  **236.93 → 229.50 µs** is **−3.1%**, meeting the keep-gate. PMC per
  wave vs phase-3 XOR: conflict **44352 → 31680** (**−28.6%**),
  wait-LDS **27725 → 16543** (**−40.3%**), busy **8889 → 8405**
  (**−5.4%**); MFMA and VMEM are unchanged. Conflict is still **3.0×**
  AMD (10560), but this reproduces the material Triton lowering:
  bank-spread addressing, `write2st64`, and transposed LDS reads.
- Prefill overlap of `ds_read_b64_tr_b16` with the current PV MFMA by
  issuing `load_pv_b(ng+1)` before `pv_mfma(..., v_cur)` on a **single**
  B fragment (no ping-pong dest VGPRs). Decode/prefill oracle passed.
  Prefill ISA (`tickets/1047/tmp/k2_tr16_overlap_isa/`) still has
  `s_waitcnt lgkmcnt(0)` before **every** PV MFMA because the next
  `ds_read_b64_tr_b16` reuses `v[66:67]`. Keep-gate vs same-session
  write2st64 HEAD **30.79 / 27.02 / 230.93** µs: FlyDSL
  **25.66 / 26.01 / 229.34**; AMD same session **42.84 / 42.53 / 211.96**.
  `M=512` **−0.7%** (need ≤ **224.00**). Decode did not regress. Kernel
  restored. Do not retry this single-buffer interleave. Ping-pong B
  dests are a different experiment.

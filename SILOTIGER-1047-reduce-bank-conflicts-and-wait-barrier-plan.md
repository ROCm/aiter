# SILOTIGER-1047 — Reduce K2 split LDS bank conflicts and wait/barrier

Close family A FlyDSL sparse GQA **split** vs **live AMD** by cutting
per-wave LDS bank conflicts and `lgkmcnt` wait/barrier, not by adding
MFMA or workgroups. MMA per wave already matches AMD. Merge is already
ahead. Parent ticket plan: `SILOTIGER-1047-plan.md` (K2 keep-gate and
do-not-retry live there). This file tracks only the conflict/wait
campaign.

Parent: [SILOTIGER-1040](https://amd.atlassian.net/browse/SILOTIGER-1040).
Ticket: [SILOTIGER-1047](https://amd.atlassian.net/browse/SILOTIGER-1047).

**Bar:** width-2051 `L=512` family A GQA on GPU 6 / gfx950 vs live
`qsa_sparse_paged_attention`. Headline is split-kernel wall clock
(`M=1` / `M=8` / `M=512`). Kernel-trace baseline (45 launches):
FlyDSL split 15.06 µs vs AMD 6.94 µs at `M=1`; 516.94 vs 209.76 µs at
`M=512`. Merge is FlyDSL 3.54 vs AMD 4.13 µs — out of scope here.

Work the phases **in order**. Later items assume earlier ones have
landed. Leave checkboxes unchecked until that item is done; paste
tables and notes under the relevant phase as evidence. One ISA line
per experiment; ≥3% keep-gate or revert; stage then one-line commit
when that is the ticket convention.

## Progress

- [x] 0. Baseline PMC + ATT (physical GPU 6)
- [ ] 1. Lock launch paths; decode ATT that actually hits a CU
- [ ] 2. Pack live / phys / page_off LDS
- [ ] 3. Overlap K/V `buffer_load` with LDS wait/barrier
- [ ] 4. Bank-conflict-free K (and C) stores; V layout unchanged
- [ ] 5. Token-major V: pack stores only
- [ ] 6. Optional: drop C-LDS QK reduce (last, likely small)
- [ ] 7. Stop: within ~2× AMD per-wave conflict/wait **or** keep-gate dry

## Locked decisions

These locks apply to **this campaign** unless a later note explicitly
supersedes them. Ticket-wide locks in `SILOTIGER-1047-plan.md` still
apply.

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

### Scope and bar

- **Split kernel only.** Do not spend keep-gate budget on merge,
  indexer K1, or occupancy of unused CU slots. `M=1` is ~0.25–0.5 WG/CU.
- **Family A GQA, live AMD is the gate.** Width-2051 `L=512`
  `M∈{1,8,512}` vs vendored Triton `_qsa_sparse_paged_gqa_splitk`.
  `err=0` vs the oracle (`checkAllclose` `1e-2` as in the parent plan).
- **Keep gate is 3% wall-clock** on that width-2051 row, with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` after kernel-source edits. Flat or
  a loss reverts; paste the row and restore HEAD.
- **One path per experiment** unless the change is shared (live/phys
  LDS, HBM overlap). Decode stays `BLOCK_N=16` / 256 threads /
  32 splits / token-major V. Prefill stays `BLOCK_N=64` / 128 threads
  / `splits=1` / row-major V.
- **Do not chase MFMA count.** PMC: `SQ_INSTS_MFMA / SQ_WAVES` is
  **1.00×** AMD at `M=1` (24.2) and `M=512` (1584). Static 8 vs 2 K32
  is unroll, not extra useful math.
- **Do not reopen occupancy forks**, extra decode splits, 256-thread
  BN64 prefill, or “more K32.”
- **Inherit K2 do-not-retry** from `SILOTIGER-1047-plan.md` (token-major
  V → row-major / transpose PV, NT gathers, merge widen, 64 decode
  splits, in-register full-D QK `ld.lld` miss, and the rest). This
  campaign does not retry those.

### Profiling

- **ATT GPU index is physical.** With `HIP_VISIBLE_DEVICES=6` pass
  `--att-gpu-index 6`. Index `0` traces idle GPU 0 and yields empty
  decode (`Number of services generating output: 0`).
- **Decode ATT can miss CU 0 / SE 0.** Default `--att-target-cu 0`
  and shader-engine mask `0x1` often miss the `M=1` split grid. Use
  `M=8`, a wide `--att-shader-engine-mask`, or `M=512` (BN64 path).
  Do not treat an empty `stats_*.csv` as “decoder broken.”
- **PMC pass that fits hardware:** `SQ_WAVES SQ_BUSY_CYCLES
  SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT SQ_INSTS_VALU SQ_INSTS_MFMA
  SQ_INSTS_VMEM SQ_INSTS_SALU`. Adding `SQ_INSTS_LDS` to that group
  is error 38 (request exceeds hardware). Report **per wave**.
- **rocprof under JIT:** fill FlyDSL cache then
  `FLYDSL_RUNTIME_ENABLE_CACHE=1` for ATT/PMC; keep-gate benches stay
  `CACHE=0`.
- **Packaging holes (container, not git):** unversioned
  `librocprof-trace-decoder.so` and
  `lib/rocprofiler-sdk/librocprofv3-list-avail.so` may need symlinks
  to the `.so.*` the wheel actually ships.

### After each experiment

- Pytest `op_tests/test_flydsl_qsa.py` as in the parent gate snippet.
- Width-2051 `L=512` `M=1/8/512` vs live AMD.
- On a keep: PMC per-wave conflict / wait LDS / MFMA (MFMA must stay
  ~1× AMD). On a keep that claims a waitcnt win: ATT stall histogram
  on GPU index 6.
- Evidence goes under that phase. Reverts go to do-not-retry here
  **and** in `SILOTIGER-1047-plan.md`.

## Subtasks

### 0. Baseline PMC + ATT (physical GPU 6)

Already measured 2026-09-21. Traces:
`tickets/1047/tmp/k2_pmc_att/`, canvas `k2-split-pmc-att`.

- [x] Kernel-trace named split/merge at `M=1` and `M=512`.
- [x] PMC per-wave FlyDSL vs AMD at `M=1` and `M=512`.
- [x] ATT opcode + barrier map at `M=512` (BN64). Decode `M=1` ATT
      empty on CU 0.

| | M=1 busy/wave | M=1 conflict/wave | M=1 wait LDS/wave | MFMA/wave | M=512 busy | M=512 conflict | M=512 wait LDS |
|--:|--:|--:|--:|--:|--:|--:|--:|
| FlyDSL / AMD | 5.05× | 12.8× | 2.97× | **1.00×** | 2.60× | 8.90× | 4.73× |

ATT `M=512` FlyDSL latency: `ds_write` 24.5%, `s_waitcnt` 24%
(mostly `lgkmcnt`), `s_barrier` 13%, MFMA 1.2%. AMD: VALU 39%, HBM
load 14%, `vmcnt` waits, packed `ds_write_b128`. Hottest FlyDSL
barriers: tile sync ~449k stall, live/phys `ds_write_b32` ~122k,
C-LDS QK publish ~11k.

**Done when:** this table is the comparison point for later phases.

### 1. Lock launch paths; decode ATT that actually hits a CU

- [ ] Confirm HEAD still decode BN16 / 256-thread / token-major V and
      prefill BN64 / 128-thread / row-major V.
- [ ] Capture non-empty decode ATT (`M=8` and/or wide SE mask,
      `--att-gpu-index 6`, kernel-include split). Rank `s_barrier`
      vs the BN64 map.
- [ ] **Done when:** a `stats_*.csv` names
      `qsa_k2_family_a_port_split_ps16_bn16_*` with hitcount > 0.
      If the 449k / 122k ranking differs from BN64, reorder phases 2–6
      in this file before implementing.

### 2. Pack live / phys / page_off LDS

Shared decode+prefill. ATT’s second-hottest FlyDSL barrier is
`ds_write_b32` then `lgkmcnt(0)` then barrier.

- [ ] Vector / conflict-free stores for live, phys, page_off.
- [ ] Fold or delay that barrier if the next reader still orders.
- [ ] Keep-gate `M=1` first, then `M=8` / `M=512`.
- [ ] **Done when:** kept (≥3%) or reverted + do-not-retry note.

### 3. Overlap K/V `buffer_load` with LDS wait/barrier

AMD issues `buffer_load_dwordx4` then waits `vmcnt` across a barrier.
FlyDSL waits LDS, barriers, then loads.

- [ ] Issue gathers before `lgkmcnt(0)` / the tile barrier; do not add
      barriers.
- [ ] Keep-gate; ATT should move stall from `lgkmcnt` toward `vmcnt`.
- [ ] **Done when:** kept or reverted + note.

### 4. Bank-conflict-free K (and C) stores; V layout unchanged

Prefill ATT: expensive `ds_write_b128` + `lgkmcnt(0)` before KV-publish
and C-publish. Swizzle/pad/phase the **existing** tiled K/C maps.

- [ ] Decode-only K/C if the maps differ; else one shared change.
- [ ] **Do not** switch decode V to `[BLOCK_N, D]`.
- [ ] Keep-gate; PMC conflict/wave must move.
- [ ] **Done when:** kept or reverted + note.

### 5. Token-major V: pack stores only

Keep `v_lds[d, col]` for PV. Replace scalar `ds_write_b16` with wider
stores that still land in that layout.

- [ ] Decode-only. If PV `lgkmcnt` / waitcnt rises, revert — same
      failure class as the row-major V keep-gate misses.
- [ ] **Done when:** kept or reverted + note.

### 6. Optional: drop C-LDS QK reduce

Last. ATT ~11k stall vs 449k tile barrier. Parent plan already
**do-not-retry** in-register full-D QK (`ld.lld` abort). Only reopen
with a compile that links, and only if phases 2–5 left C-LDS as the
limiter.

- [ ] Skip unless phase-1 decode ATT plus later PMC say C-LDS wait
      dominates.
- [ ] **Done when:** skipped with a note, or kept/reverted.

### 7. Stop

- [ ] Per-wave `SQ_LDS_BANK_CONFLICT` and `SQ_WAIT_INST_LDS` within
      ~2× of AMD, **or** keep-gate dry on this list.
- [ ] Paste final width-2051 `M=1/8/512` vs live AMD and per-wave PMC.

## Do not retry (this campaign)

Copied here so this file stands alone for the conflict/wait loop.
The parent plan remains the full K2 list.

- Occupancy / extra WGs / 64 decode splits / 256-thread BN64 prefill.
- Adding MFMA or unrolling QK to “match AMD’s 8 K32.”
- Decode V layout change to row-major `[BLOCK_N, D]` or transpose PV-B
  (parent keep-gate losses).
- In-register full-D QK as a first experiment (parent `ld.lld` miss).
- PMC groups that include `SQ_INSTS_LDS` with the eight-counter pass.
- ATT `--att-gpu-index 0` under `HIP_VISIBLE_DEVICES=6`.

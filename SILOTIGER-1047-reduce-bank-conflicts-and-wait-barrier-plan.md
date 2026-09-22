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
- [x] 1. Lock launch paths; decode ATT that actually hits a CU
- [x] 2. Pack live / phys / page_off LDS
- [x] 3. Overlap K/V `buffer_load` with LDS wait/barrier
- [x] 4. Bank-conflict-free K (and C) stores; V layout unchanged
      (K pad **kept**; C 66-lane pad **reverted**)
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
- **`CACHE=0` needs `ROCM_PATH=/root/.flydsl/toolkit`.** FlyDSL runs
  `gpu-module-to-binary` with an empty `toolkit=`, so MLIR resolves
  `ld.lld` from `ROCM_PATH`, which is unset in this container. Every
  *real* compile then dies with `lld invocation failed` — the ROCm
  wheel puts the linker at `_rocm_sdk_core/lib/llvm/bin`, not the
  `llvm/bin` MLIR appends. `CACHE=1` hides this by hitting the disk
  cache, so it looks like "the edit broke the compile" when in fact
  any source edit (new cache key) fails and HEAD fails too. Always
  run the HEAD control with `CACHE=0` before blaming a kernel edit.
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

- [x] Confirm HEAD still decode BN16 / 256-thread / token-major V and
      prefill BN64 / 128-thread / row-major V.
- [x] Capture non-empty decode ATT (`M=8` and/or wide SE mask,
      `--att-gpu-index 6`, kernel-include split). Rank `s_barrier`
      vs the BN64 map.
- [x] **Done when:** a `stats_*.csv` names
      `qsa_k2_family_a_port_split_ps16_bn16_*` with hitcount > 0.
      If the 449k / 122k ranking differs from BN64, reorder phases 2–6
      in this file before implementing.

Measured 2026-09-22 on GPU 6 / gfx950. `_launch_config` (`Hk=2`,
`n_sel=2051`, `use_k32` ⇒ `token_major_v = (block_n==16)`):

| M | `base_programs` | `BLOCK_N` | threads | splits | V layout |
|--:|--:|--:|--:|--:|--|
| 1 | 2 | 16 | 256 | 32 | token-major |
| 8 | 16 | 16 | 256 | 32 | token-major |
| 512 | 1024 | 64 | 128 | 1 | row-major |

`M=8` is the same decode kernel as `M=1` (512 WGs vs 64). Trace:
`tickets/1047/tmp/k2_pmc_att/att_flydsl_m8/stats_ui_output_agent_43127_dispatch_138.csv`.
First instruction comment:
`qsa_k2_family_a_port_split_ps16_bn16_blk256_ns32_qkk32`. Hitcounts
on `s_barrier` are 16 (prologue) and 64 (loop). `--att-gpu-index 6`
`--att-target-cu 0` was enough; no wide SE mask.

Opcode latency (decode `M=8` vs prefill `M=512` FlyDSL):

| | decode M=8 | prefill M=512 |
|--|--:|--:|
| `s_waitcnt` | 37.9% | ~24% |
| `s_barrier` | 37.8% | ~13% |
| `ds_write_b128` | 3.9% | (hottest ds) |
| `ds_write_b16` (token-major V) | 3.2% | n/a |
| MFMA | 0.42% | 1.2% |

Waitcnt split on decode: `lgkmcnt(0)` 44.8k stall, `vmcnt(1)` 39.8k,
`vmcnt(0)` 10.5k. Decode is already ~half `vmcnt`.

`s_barrier` map (decode `M=8`, stall; source in `k2_family_a.py`):

| stall | vaddr | after | next | source |
|--:|--:|--|--|--|
| 67520 | 8804 | `lgkmcnt(0)` | `ds_read` P/V for PV | loop `gpu.barrier()` ~468 |
| 25664 | 9172 | `ds_write_b32` page_off / phys / live + `lgkmcnt(0)` | `ds_read_b32` those rows | ~314 |
| 7660 | 9588 | `ds_write_b128` K + `lgkmcnt(0)` | `buffer_load_dwordx4` V | ~344 |
| 3916 | 9960 | `ds_write_b16` token-major V + `lgkmcnt(0)` | C-LDS reads | ~398 |
| 1360 | 8976 | QK MFMA | column-0 gather | loop entry ~293 |
| 80 | 8032 | init m/l | | ~285 |

Prefill hottest pair was tile-join then live/phys (`~449k` / `~122k`).
Decode hottest join is the **softmax / P-LDS publish** (~468), not
loop-entry (~293, only 1.4k). Second is still live/phys (~314).
Phases **2–6 stay in order**: pack live/phys first (shared
decode+prefill, still #2), then overlap gathers (K barrier already
sits in front of `buffer_load`), then K/C maps, then pack V stores,
C-LDS last (4th, small). Do not add a P-LDS rewrite ahead of phase 2;
~468 is a full-WG join after P stores, same class as the BN64 tile
barrier, and packing live/phys does not depend on it.

**Done.** Next: phase 2.

### 2. Pack live / phys / page_off LDS

Shared decode+prefill. ATT’s second-hottest FlyDSL barrier is
`ds_write_b32` then `lgkmcnt(0)` then barrier.

- [x] Vector / conflict-free stores for live, phys, page_off.
      **Reverted 2026-09-22:** replaced the three Int32 LDS rows with
      one `[BLOCK_N, 4]` row (`phys`, `page_off`, `live`, pad) and one
      128-bit vector store per translating column. Readers kept
      the same logical values and the translate barrier stayed.
- [x] Fold or delay that barrier if the next reader still orders.
      Skipped: the parent plan already keep-gate-lost per-thread
      redundant page translation. The earlier untested replay was
      only an environment failure, not evidence for reopening it.
- [x] Keep-gate `M=1` first, then `M=8` / `M=512`.
- [x] **Done when:** packed stores kept (≥3%) or reverted + note.

Correctness: focused GPU-6 pytest passed (`2 passed`, decode and
prefill, `CACHE=0`, `ROCM_PATH=/root/.flydsl/toolkit`). ISA:
three metadata `ds_write_b32` instructions became one
`ds_write_b128` (offset 20992).

| width-2051 `L=512` wrapper | HEAD median (3 runs) | packed median (3 runs) | speedup |
|--|--:|--:|--:|
| M=1 | 20.18 µs | 20.19 µs | -0.05% |
| M=8 | 20.86 µs | 20.69 µs | +0.81% |
| M=512 | 515.98 µs | 518.41 µs | -0.47% |

No row reaches the 3% keep gate, and M=1/prefill regress. Kernel
restored to HEAD; no PMC/ATT follow-up for a reverted change.
**Done. Next: phase 3.**

### 3. Overlap K/V `buffer_load` with LDS wait/barrier

AMD issues `buffer_load_dwordx4` then waits `vmcnt` across a barrier.
FlyDSL waits LDS, barriers, then loads.

- [x] Issue gathers before `lgkmcnt(0)` / the tile barrier; do not add
      barriers.
      **Reverted 2026-09-22:** gfx950 `use_k32` issued V `g_copy` after
      K LDS stores and before the K-publish barrier; V LDS stores and
      QK ran after it. No extra barrier. Distinct from the parent
      next-tile-K-with-PV miss.
- [x] Keep-gate; ATT should move stall from `lgkmcnt` toward `vmcnt`.
      Skipped ATT: wall-clock missed the gate.
- [x] **Done when:** kept or reverted + note.

Correctness: focused GPU-6 pytest passed (`2 passed`, decode and
prefill, `CACHE=0`, `ROCM_PATH=/root/.flydsl/toolkit`). Decode ISA:
V `buffer_load_dwordx4` sits with K loads after the live/phys barrier
and before K `ds_write_b128` + `lgkmcnt(0)` + `s_barrier`; QK then
waits `vmcnt(1)`.

| width-2051 `L=512` wrapper | HEAD median (3 runs) | hoist median (3 runs) | speedup |
|--|--:|--:|--:|
| M=1 | 20.18 µs | 20.13 µs | +0.25% |
| M=8 | 20.86 µs | 20.96 µs | -0.48% |
| M=512 | 515.98 µs | 518.30 µs | -0.45% |

No row reaches the 3% keep gate. Kernel restored to HEAD.
**Done. Next: phase 4.**

### 4. Bank-conflict-free K (and C) stores; V layout unchanged

Prefill ATT: expensive `ds_write_b128` + `lgkmcnt(0)` before KV-publish
and C-publish. Swizzle/pad/phase the **existing** tiled K/C maps.

- [x] Shared K-row pad `_K_STRIDE = D+8` (gfx950 `k` and gfx942 aliased
      `kv`). Decode token-major V stays `[D, BLOCK_N]`. gfx950 row-major
      V stays unpadded stride `D`. C LDS not in this ISA line.
- [x] **Do not** switch decode V to `[BLOCK_N, D]`.
- [x] Keep-gate on K pad; PMC conflict/wave moved.
- [x] C 64→66 lane-axis pad: pytest 2/2, miss keep-gate, reverted.
- [x] **Done.** K kept; do not retry C 66-pad. Phase 5 next.

Measured 2026-09-22 GPU 6 / gfx950, `CACHE=0`, `ROCM_PATH=/root/.flydsl/toolkit`.
Focused pytest `test_k2_family_a_*` 2/2. Width-2051 `L=512` median of
three `warmup=20`/`iters=100` vs HEAD `20.18 / 20.86 / 515.98`:

| M | HEAD | K pad | Δ |
|--:|--:|--:|--:|
| 1 | 20.18 | 20.09 | −0.45% |
| 8 | 20.86 | 20.52 | −1.63% |
| 512 | 515.98 | 424.09 | **−17.8%** |

Keep: prefill clears 3%; decode does not regress. New wall-clock
baseline is **20.09 / 20.52 / 424.09**.

PMC (rocprofv3 8-counter pass, last split dispatch per-wave; this
sqlite path reports fewer `SQ_WAVES` than phase-0 CSV, so compare
/wave not totals):

| | M=1 conflict/wave | M=1 wait LDS/wave | M=1 MFMA/wave | M=512 conflict | M=512 wait LDS | M=512 MFMA |
|--|--:|--:|--:|--:|--:|--:|
| HEAD (phase 0 CSV) | 1032 | 146 | 24.2 | 93984 | 44095 | 1584 |
| K pad | 384 | 70 | 24.0 | 51744 | 27279 | 1584 |

LDS `group_segment_size` 21632 (decode) / 77760 (prefill) vs
21504 / 76800. MFMA/wave unchanged. Leftover decode conflicts are
still expected (col vs col+8 on the pad). C-LDS `(n_subtiles,
num_waves, 64, 4)` is the remaining store map in this phase.

**K pad kept.** Next in phase 4: C stores only.

C-LDS pad `_C_LANE_STRIDE=66` (logical 64, wave-row stride 264 so
`wave·264 % 32 = 8·wave`). Indexing `c_lds[ng, wave, lane, i]`
unchanged. GPU 6 / gfx950, `CACHE=0`. Pytest 2/2. Median vs K-pad
baseline `20.09 / 20.52 / 424.09`:

| M | K pad | C 66-pad | Δ |
|--:|--:|--:|--:|
| 1 | 20.09 | 20.31 | +1.10% |
| 8 | 20.52 | 19.98 | −2.63% |
| 512 | 424.09 | 431.45 | +1.74% |

M=8 under 3%; M=1 and prefill regress. Kernel C layout restored.
**Phase 4 done.** Next: phase 5 (pack token-major V stores).

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
- Reading `lld invocation failed` as a kernel-source bug. It is an
  environment fault in this container (see the `ROCM_PATH` lock).
- Packing `phys/page_off/live/pad` into one `[BLOCK_N,4]` Int32 LDS
  row with one 128-bit vector store. Correct and emitted
  `ds_write_b128`, but M=1 was flat, M=8 improved only 0.81%, and
  M=512 regressed 0.47%; all miss the 3% gate.
- Hoisting gfx950 V `buffer_load` before the K-publish `lgkmcnt(0)` /
  `s_barrier`. ISA moved the loads; wall-clock missed 3% (M=1 +0.25%,
  M=8 −0.48%, M=512 −0.45%). Keep V gather after that barrier.
- Padding C-LDS's 64-lane axis to 66. Correct; M=8 −2.63%, M=1 +1.10%,
  M=512 +1.74% vs K-pad baseline. Keep unpadded `(n_subtiles,
  num_waves, 64, 4)`.

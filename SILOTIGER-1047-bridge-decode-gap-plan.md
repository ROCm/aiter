# SILOTIGER-1047 — Bridge K2 overlay decode split vs Live AMD

Close family A FlyDSL overlay **decode split** vs **live AMD** by
incrementally matching Live AMD’s decode lowering. Prefill already
beats AMD on the st64-imm keeper. Decode still trails because the
BN16 / 4-wave split body is not the same physical lowering as AMD’s
decode HSACO.

Parent: [SILOTIGER-1040](https://amd.atlassian.net/browse/SILOTIGER-1040).
Ticket: [SILOTIGER-1047](https://amd.atlassian.net/browse/SILOTIGER-1047).
Ticket-wide plan: `SILOTIGER-1047-plan.md` (ticket locks, K2 keep-gate,
and do-not-retry live there). Prefill overlay campaign:
`SILOTIGER-1047-optimize-overlay-split-plan.md` (leave that path frozen).

**Kernel name.** Living name is **K2 overlay decode**. Source is still
`aiter/ops/flydsl/kernels/qsa/k2_family_a.py`. Decode HSACO is
`qsa_k2_family_a_port_split_ps16_bn16_blk256_ns{32,64}_qkk32`. Prefill
`bn64_blk128_ns1` is out of scope except as a no-regress sanity row.

**Bar:** width-2051 `L=512` family A GQA on GPU 6 / gfx950 vs live
`qsa_sparse_paged_attention`. Headline is **decode split wall** at
`M=1` and `M=8` (wrapper is acceptable when kernel-trace is not
available; prefer split vs split when rocprof is run). Retained
mixed-gate medians after st64-imm prefill (2026-09-23): FlyDSL decode
**20.02 / 19.99** µs vs AMD **9.93 / 13.30** µs (~2.02× / ~1.50×).
Prefill FlyDSL **159.41** µs vs AMD **202.80 / 211.55** µs is already
ahead and is **not** this campaign’s bar.

The campaign is a **success** when FlyDSL K2 **decode split** matches
or exceeds Live AMD decode split on that row at **both** `M=1` and
`M=8`. Intermediate steps **record** wall clock; they **do not**
keep-gate on it.

Work the phases **in order**. Later items assume earlier ones have
landed. Leave checkboxes unchecked until that item is done; paste
tables, ISA lines, and notes under the relevant phase as evidence.
Stage then one-line commit when that is the ticket convention.

## Progress

- [x] 0. Decode baseline: ISA / PMC / ATT vs Live AMD M=1 and M=8
- [x] 1. AMD-matched decode K LDS map + inverse QK reads
- [x] 2. AMD-matched decode V publication (`ds_write_b64` ×4) + inverse PV reads
- [x] 3. Cut leftover permute / bpermute / extra waits that AMD decode does not emit
- [x] 4. Waitcnt / barrier schedule toward AMD’s decode mix
- [x] 5. Split-count specializations (`ns64` at M=1, `ns32` at M=8)
- [ ] 6. Stop: decode split matches or exceeds Live AMD at M=1 and M=8

## Locked decisions

These locks apply to **this campaign** unless a later note explicitly
supersedes them. Ticket-wide locks in `SILOTIGER-1047-plan.md` still
apply. Prefill overlay DNR in that parent file and in
`SILOTIGER-1047-optimize-overlay-split-plan.md` still apply; this
campaign does not retry those on decode.

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
- **Correctness after a kernel exists:**
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    ROCM_PATH=/root/.flydsl/toolkit \
    python3 -m pytest op_tests/test_flydsl_qsa.py -q
  ```
  Focused decode oracle (`test_k2_family_a_decode_matches_oracle` and
  the prefill counterpart as a no-regress check) is enough between
  steps. Full `__main__` family-B sweep is not required per step.
- **Perf measurement (not a keep-gate).** After every phase, record
  width-2051 `L=512` `M∈{1,8}` (and `M=512` as a sanity row) vs live
  AMD with `tickets/1047/tmp/profile_k2_gqa.py` or the parent bench,
  `CACHE=0`. Prefer five-run medians. Dump ISA
  (`FLYDSL_DUMP_IR=1`) and, when the ISA line moved, PMC/ATT. Paste
  the row under the phase. **Do not revert solely because wall clock
  is flat or slightly worse.** Revert if oracle fails, if the change
  moves the decode ISA **away** from Live AMD, or if it touches the
  frozen prefill branch.

### Scope and bar

- **Decode overlay split only.** Do not spend this campaign on merge,
  indexer K1, family B, prefill store maps, or undoing overlay back to
  C/P/metadata LDS. Historical kernel-trace had FlyDSL merge already
  at or faster than AMD (`~3.5` vs `~4.1` µs at M=1); the gap is the
  split kernel.
- **Family A GQA, live AMD is the success bar.** Width-2051 `L=512`
  `M∈{1,8}` vs vendored Triton `_qsa_sparse_paged_gqa_splitk`.
  `err=0` vs the oracle (`checkAllclose` `1e-2` as in the parent plan).
- **No 3% keep-gate on wall clock.** This campaign measures between
  steps so the log shows whether lowering is paying off. A step that
  is oracle-correct **and** closer to AMD’s decode opcode/resource
  mix is kept even if wrapper µs is flat. Success is **only** declared
  when decode split matches or exceeds Live AMD at both M=1 and M=8.
- **Prefill is frozen.** All experiments stay behind
  `const_expr(decode_tr_pv)` (`use_k32 and block_n == 16`). Do not
  edit the BN64 `amd_k_elem` / `write2st64` / tr16-immediate lattice
  unless a decode change would otherwise share that code path; then
  `const_expr` it. Record `M=512` so a leak is visible.
- **One logical decode body.** M=1 and M=8 share the BN16 / 256-thread
  / 4-wave tile. Live AMD already does this: same material split
  lowering, different `NUM_SPLITS` constexpr (64 vs 32) and merge
  `BLOCK_SPLITS`. Do not invent a second decode algorithm for M=8.
- **Two launch specializations are in scope.** After the tile body is
  AMD-like, compile `ns64` for tiny M (`base_programs ≤ 4`) and keep
  `ns32` for M=8 (`4 < base_programs < 32`), matching
  `qsa_vllm_amd.py`. Until then, leave dispatch at the current 32
  splits. Parent DNR “do not restore 64 decode splits” applied to the
  **old heavier body** where merge traffic dominated; it is **not** a
  lock against AMD-matched `ns64` in phase 5.
- **Do not start from scratch.** Keep the overlay contract below. Fork
  the current decode branch; do not rewrite math, and do not seed
  decode from the winning prefill physical V map.
- **Do not chase extra MFMA.** Decode static MFMA already matches AMD
  (8× K32 QK + 4× K16 PV = 12). Packed K32 PV that would make decode
  8+8 or 12 K32 is out of scope.

### Overlay contract (decode)

Keep this shape unless a later note explicitly supersedes it.

- Compute **`K @ Q^T`**. QK C is token-major in-lane and feeds PV-A
  without P-LDS or a bpermute transpose of P.
- **One MMA scratch.** K and V overlay the same LDS; barrier before
  overwrite. gfx950 emitted decode size at retain: **8192 B** (rocprof
  dispatch padded to 8960). AMD decode is the same class (8192
  logical).
- Softmax **m/l stay in registers**.
- **No C/P/metadata LDS.** Do not reintroduce those rows to chase a
  decode-only win.

### Target Live AMD decode lowering

Same Triton source as prefill, **different compile**. Fresh GPU-6
captures (2026-09-23) under `/tmp/k2_amd_lowering_compare/`:

| | AMD M=1 | AMD M=8 | AMD M=512 | FlyDSL decode HEAD |
|--|--:|--:|--:|--:|
| BLOCK_N / threads / splits | 16 / 256 / **64** | 16 / 256 / **32** | 64 / 128 / 1 | 16 / 256 / **32** |
| Split WGs | **128** | **512** | 1024 | **128 (M=1 ns64) / 512 (M=8 ns32)** |
| LDS B | 8192 | 8192 | 32768 | 8192 |
| VGPR | 106 | 106 | 245 | 108 |
| Static inst | 1002 | 1017 | 2374 | 1070 |
| `buffer_load_dwordx4` | 6 | 6 | 36 | 12 |
| `ds_write_b128` / `b64` / `write2st64` | **4 / 4 / 0** | 4 / 4 / 0 | 20 / 0 / 16 | **4 / 0 / 0** |
| `ds_read_b128` / `tr16` | **16 / 4** | 16 / 4 | 40 / 32 | 16 / 4 |
| K32 / K16 MFMA | **8 / 4** | 8 / 4 | 48 / 0 | 8 / 4 |
| `s_waitcnt` / `s_barrier` | **33 / 5** | 34 / 5 | 102 / 5 | **32 / 4 split** (whole-file ~108 includes merge) |
| `v_perm` / `ds_bpermute` | **0 / 0** | 0 / 0 | 0 / 0 | **34 / 12** |

M=1 vs M=8 opcode-sequence similarity is **~0.97**; decode vs prefill
is **~0.50**. FlyDSL decode vs AMD decode opcode similarity is
**~0.06**. Match the **decode** column, not the prefill column.

AMD K/V publication on decode is **4× `ds_write_b128` + 4×
`ds_write_b64`**, then **16× `ds_read_b128` + 4× `ds_read_b64_tr_b16`**.
Current FlyDSL V stores lower to **`ds_write_b32`**, not `b64`. That
is the first material opcode miss.

### Profiling

- **ATT GPU index is physical.** With `HIP_VISIBLE_DEVICES=6` pass
  `--att-gpu-index 6`. Index `0` traces idle GPU 0 and yields empty
  decode.
- **Decode ATT uses `M=8` as the body proxy** (same BN16 HSACO as
  M=1). Also dump M=1 when split count changes (phase 5).
- **PMC pass that fits hardware:** `SQ_WAVES SQ_BUSY_CYCLES
  SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT SQ_INSTS_VALU SQ_INSTS_MFMA
  SQ_INSTS_VMEM SQ_INSTS_SALU`. Adding `SQ_INSTS_LDS` to that group is
  error 38. Report **per wave** and **totals**. Overlay decode
  MFMA/wave was 2.00× AMD only because AMD launches 2× waves at M=1;
  totals matched (12384).
- **Primary decode counters to watch:** LDS bank conflict/wave (HEAD
  ~1258 vs AMD ~81, **15.6×**), `s_waitcnt` static count (**split**
  32 vs AMD 33–34; whole-file ~108 is merge’s `vmcnt(31)` ladder),
  wait-LDS/wave (~109 vs ~50). MFMA totals should stay matched.

### Do not reopen (decode)

Inherited from `SILOTIGER-1047-plan.md` unless a later lock here
explicitly reopens it for an AMD-matched body:

- Hoist V `buffer_load` before K-publish `lgkmcnt(0)` / `s_barrier`.
- Split that barrier into waitcnt + raw `s_barrier` **as a schedule
  tweak on the old map** (the +97–107% miss). A waitcnt+barrier that
  is the inverse of a **new** AMD-matched K/V map is phase 4, not a
  retry of that DNR.
- `BLOCK_N=64` / 8 splits for M=8; four-wave BN64 prefill.
- Prefill `write2st64` / blocked `p(n,d)` V map on decode.
- Prefill packed K32 PV on decode.
- Token-major ↔ row-major V swaps that were already DNR on the old
  C/P or overlay maps. Phase 2 is matching AMD’s **decode** V
  publication (4× `ds_write_b64`), not retrying those failed maps.
- NT gathers; extra prefill splits; occupancy forks; merge unroll /
  256-thread merge / BF16 partials.
- Restoring 64 splits **on the current heavier body** (parent: M=1
  18.65 → 23.26 µs). Phase 5 reopens this **only** after phases 1–4
  have an AMD-like split ISA.

## Subtasks

### 0. Decode baseline: ISA / PMC / ATT vs Live AMD M=1 and M=8

Snapshot HEAD before changing the decode branch. Do not edit the
kernel in this phase.

- [x] Dump FlyDSL decode ISA (`FLYDSL_DUMP_IR=1`, `CACHE=0`) for
      `bn16_blk256_ns32`. Compare opcode mix to
      `/tmp/k2_amd_lowering_compare/{m1,m8}/kernel.s` (or a fresh
      Live AMD capture on GPU 6).
- [x] Record five-run wrapper medians `M=1/8/512` vs live AMD.
      Optional: rocprof split vs merge so wrapper is not the only
      number.
- [x] PMC M=1 (and M=8 if cheap) per-wave + totals vs
      `tickets/1047/tmp/k2_overlay_pmc/`.
- [x] ATT M=8 vs `tickets/1047/tmp/k2_overlay_att/att_amd_m8/` and
      `att_flydsl_m8/`. Confirm stall mix is still waitcnt-dominated.
- [x] **Done when:** the table in **Target Live AMD decode lowering**
      is pasted as a dated HEAD row under this phase, with paths to
      dumps. No kernel diff.

Measured 2026-09-24 on GPU 6 / gfx950, `flydsl_venv`, `CACHE=0` for
ISA + wrapper, `CACHE=1` after fill for PMC/ktrace. No kernel diff.
Dumps: `tickets/1047/tmp/k2_decode_baseline/`
(`isa_flydsl_m1/launch/22_final_isa.s`, `amd_m{1,8}_split.s`,
`pmc_{flydsl,amd}_m{1,8}/`, `ktrace_{flydsl,amd}_m{1,8}/`).
ATT reused from overlay (`tickets/1047/tmp/k2_overlay_att/att_*_m8/`);
the decode HSACO name is still
`qsa_k2_family_a_port_split_ps16_bn16_blk256_ns32_qkk32`.

**HEAD opcode mix** (static ISA; AMD LDS/VGPR from this session’s
rocprof dispatch when `.amdhsa_*` is missing):

| | AMD M=1 | AMD M=8 | FlyDSL decode HEAD |
|--|--:|--:|--:|
| BLOCK_N / threads / splits | 16 / 256 / **64** | 16 / 256 / **32** | 16 / 256 / **32** |
| Split WGs | **128** | **512** | **64** / **512** |
| LDS B | 8192 | 8192 | 8192 |
| VGPR / SGPR | 108 / 80 | 108 / 80 | 108 / 54 (ISA) / 64 (rocprof) |
| Static inst | 1002 | 1017 | 1070 |
| `buffer_load_dwordx4` / `dword` | 6 / 12 | 6 / 12 | **12 / 0** |
| `ds_write_b128` / `b64` / `b32` | **4 / 4 / 0** | 4 / 4 / 0 | **4 / 0 / 2** |
| `ds_read_b128` / `tr16` | 16 / 4 | 16 / 4 | 16 / 4 |
| K32 / K16 MFMA | 8 / 4 | 8 / 4 | 8 / 4 |
| `s_waitcnt` / `s_barrier` | **33 / 5** | 34 / 5 | **106 / 5** |
| `v_perm` / `ds_bpermute` | **0 / 0** | 0 / 0 | **34 / 12** |

**Wrapper** (CUDA-event five-run median, warmup 10 then 5×200 iters,
interleaved). Prefill is the sanity row. AMD **decode wrapper** this
session is inflated vs the retained 9.93 / 13.30 µs; **do not use it
as the split bar**. Kernel-trace below is the decode split number.

| M | FlyDSL µs | AMD µs |
|--:|--:|--:|
| 1 | 19.78 | 37.27 |
| 8 | 20.01 | 37.53 |
| 512 | 155.81 | 199.68 |

**Kernel-trace split vs merge** (`rocprofv3 --kernel-trace --stats`,
warmup 5 + 40 timed; mean of timed dispatches):

| M | Kernel | FlyDSL µs | AMD µs | ratio |
|--:|--|--:|--:|--:|
| 1 | split | **11.674** | **6.406** | **1.82×** |
| 1 | merge | 3.633 | 3.550 | 1.02× |
| 1 | split+merge | 15.307 | 9.956 | 1.54× |
| 8 | split | **14.959** | **10.525** | **1.42×** |
| 8 | merge | 3.623 | 3.027 | 1.20× |
| 8 | split+merge | 18.582 | 13.552 | 1.37× |

AMD split times match the older rocprof keep-gate (~6.8 / ~11.2 µs).
Merge is already near parity. The decode gap is the split kernel.

**PMC last split dispatch** (sum of SQ instances; per wave in
parentheses):

| | FlyDSL M=1 | AMD M=1 | FlyDSL M=8 | AMD M=8 |
|--|--:|--:|--:|--:|
| waves | 256 | **512** | 2048 | 2048 |
| WGs | 64 | 128 | 512 | 512 |
| busy | 2903.5 | 769.6 | 452.6 | 306.2 |
| conflict | **1257.8** | **80.6** | **1257.8** | 161.2 |
| wait-LDS | 345.2 | 49.8 | 812.6 | 149.7 |
| MFMA | 48.4 (tot **12384**) | 24.2 (tot **12384**) | 48.4 (tot 99072) | 48.4 (tot 99072) |
| VALU | 814.9 | 640.7 | 814.9 | 983.3 |
| VMEM | 49.2 | 39.2 | 49.2 | 71.5 |

M=1 MFMA **totals match**; FlyDSL’s 2× MFMA/wave is the 32-vs-64
split grid. Conflict/wave is still **15.6×** AMD at M=1 and **7.8×**
at M=8. That is the first physical miss for phases 1–2.

**ATT M=8 stall mix** (overlay traces; decode body unchanged):

| | AMD | FlyDSL |
|--|--:|--:|
| total stall / latency | 99,704 / 146,168 | 182,776 / 240,456 (1.65× lat) |
| stall `s_waitcnt` | 64.7% | **73.0%** |
| stall `s_barrier` | 5.8% | 8.5% |
| top waits | `vmcnt(3/5/1)` | `lgkmcnt(0)`, `vmcnt(0/1/7)` |
| dynamic `v_perm` / `bpermute` hits | 0 / 0 | 832 / 160 |
| dynamic `ds_write_b64` hits | 128 | 0 |

Waitcnt still dominates FlyDSL decode. Phase 1 starts from this
opcode/conflict gap, not from wrapper µs.

### 1. AMD-matched decode K LDS map + inverse QK reads

Live AMD decode K is 4× `ds_write_b128` into an 8 KiB XOR map, then
16× `ds_read_b128` for QK A. Prefill already has `amd_k_elem` for
BN64 / 2-wave; decode still uses generic `make_k_lds_view` XOR.
Port the **decode-sized** inverse (BN16, 4 waves, one 16-token
subtile), behind `decode_tr_pv`. Do not copy prefill’s 16-store
ladder or V `write2st64` map.

- [x] Emit 4× `ds_write_b128` whose addresses match AMD decode K
      (byte-level, not “also XOR”).
- [x] QK A reads inverse that map with 16× `ds_read_b128` and the
      existing 8× K32 gemm. No extra `v_perm` on the K path if AMD
      has none.
- [x] Oracle decode+prefill. Record wall / ISA / PMC. Keep if the
      K opcode line moved toward AMD and `err=0`.
- [x] **Done when:** decode K store/read mix is AMD’s 4 write-b128 /
      16 read-b128, prefill ISA unchanged.

Kept (2026-09-24). Decode K stores now use Live AMD’s 256-thread
byte map, behind `decode_tr_pv` only:

`(tid*16) ^ ((tid & 0xe0)>>1)` and `(that ^ 0x80) + 4096`.

QK A is the inverse via `amd_decode_k_elem` / `load_amd_k8`. Prefill
stays on `amd_k_elem`. Oracle decode+prefill: **2 passed**. Prefill
ISA sha matches the st64-imm keeper (`64ee586155fc6a63`).

**ISA** (`tickets/1047/tmp/k2_decode_kmap_isa/launch/22_final_isa.s`)
vs phase-0 HEAD:

| | HEAD | phase 1 | AMD decode |
|--|--:|--:|--:|
| `ds_write_b128` | 4 | 4 (K pair now `offset:4096`) | 4 (`offset:4096`) |
| `ds_read_b128` | 16 | 16 | 16 |
| K32 / K16 | 8 / 4 | 8 / 4 | 8 / 4 |
| `s_waitcnt` | 106 | **102** | 33 |
| `v_bitop3` | 9 | **5** | 6 |
| `v_perm` / `bpermute` | 34 / 12 | 34 / 12 | 0 / 0 |

K publication is the AMD `offset:4096` pair. Overlay QK still reads
8 D-chunks as 4+4 `offset:4096` (AMD’s extra 8 QK reads use
`offset:256` because its C map is `[M,N]`, not `K@Q^T`). V overlay
is unchanged (`ds_write_b32`, plus the pre-existing `offset:256`
b128 pair).

**Wall** (five-run median, interleaved; not a keep-gate) vs phase 0:

| M | phase 0 FlyDSL | phase 1 FlyDSL | AMD wrapper |
|--:|--:|--:|--:|
| 1 | 19.78 | **19.22** | 36.58 |
| 8 | 20.01 | **19.35** | 36.69 |
| 512 | 155.81 | 156.30 | 199.82 |

**Kernel-trace M=1 split:** **10.459 µs** vs phase 0 **11.674 µs**
vs AMD **6.406 µs**. Merge still ~3.65 µs.

**PMC M=1** (`tickets/1047/tmp/k2_decode_kmap/pmc_flydsl_m1/`) per
wave vs phase 0:

| | phase 0 | phase 1 | AMD |
|--|--:|--:|--:|
| conflict | 1257.8 | **677.2** | 80.6 |
| wait-LDS | 345.2 | **206.5** | 49.8 |
| busy | 2903.5 | 2715.6 | 769.6 |
| MFMA tot | 12384 | 12384 | 12384 |

Conflict/wave almost halved. Still 8.4× AMD; phase 2 is V
publication.

### 2. AMD-matched decode V publication + inverse PV reads

Live AMD decode V is 4× `ds_write_b64` (not b32, not write2st64),
then 4× `ds_read_b64_tr_b16`. FlyDSL HEAD currently lowers V overlay
stores to `ds_write_b32`. Match AMD’s decode V geometry and the
inverse tr16 sources. Do **not** transplant prefill’s 16×
`write2st64` lattice; that is a BN64 map and was DNR on decode.

- [x] Four `ds_write_b64` (or an equivalent pair sequence AMD
      actually emits on decode) into the overlaid 8 KiB scratch
      after the K-read barrier.
- [x] PV B from that map via 4× `ds_read_b64_tr_b16`, still 4× K16
      PV. Drop decode `UniversalCopy64b` / `ds_write_b32` if they
      remain.
- [x] Oracle decode+prefill. Record wall / ISA / PMC (conflict/wave
      is the metric that should move first: 1258 → toward 81).
- [x] **Done when:** decode V opcode mix is AMD’s 4 write-b64 + 4
      tr16, MFMA still 8+4, prefill untouched.

Kept (2026-09-24). Decode V stores use Live AMD’s 256-thread 8 KiB
b64 map, behind `decode_tr_pv` only. Each 8×bf16 gather splits into
two 4×bf16 `ds_write_b64`:

`A = (tid*16) ^ ((tid & 0xe0)>>2)`, then `A`, `A^8`,
`(A^64)+4096`, `(A^0x48)+4096`.

PV B is the inverse 4-pack (`amd_decode_v_pack`) + existing
`LDSReadTrans16_64b` / 4× K16. Not prefill `write2st64`. Oracle
decode+prefill: **2 passed**. Prefill ISA sha still
`64ee586155fc6a63`.

**ISA** (`tickets/1047/tmp/k2_decode_vmap_isa/launch/22_final_isa.s`)
vs phase 1:

| | phase 1 | phase 2 | AMD decode |
|--|--:|--:|--:|
| `ds_write_b128` / `b64` / `b32` | 4 / 0 / 2 | **2 / 4 / 2** | **4 / 4 / 0** |
| `ds_read_b128` / tr16 | 16 / 4 | 16 / **4** | 16 / 4 |
| K32 / K16 | 8 / 4 | 8 / 4 | 8 / 4 |
| `s_waitcnt` | 102 | 103 | 33 |
| `v_bitop3` | 5 | 12 | 6 |
| `v_perm` / `bpermute` | 34 / 12 | 34 / 12 | 0 / 0 |

V publication is AMD’s `offset:4096` b64 pair. Overlay K is still
two `ds_write_b128` (AMD’s extra 2 b128 are its `[M,N]` C map). The
leftover 2 `ds_write_b32 offset:256` are not V overlay (epilogue).
tr16 is 4 ops but 4 addr VGPRs, not AMD’s one addr + 128/256/384
immediates (phase 4).

**Wall** (five-run median, interleaved; not a keep-gate) vs phase 1:

| M | phase 1 FlyDSL | phase 2 FlyDSL | AMD wrapper |
|--:|--:|--:|--:|
| 1 | 19.22 | 20.38 | 37.50 |
| 8 | 19.35 | 20.56 | 37.90 |
| 512 | 156.30 | 156.18 | 199.74 |

**Kernel-trace M=1 split:** **10.503 µs** vs phase 1 **10.459 µs**
vs AMD **6.406 µs**. Merge ~3.77 µs.

**PMC M=1** (`tickets/1047/tmp/k2_decode_vmap/pmc_flydsl_m1/`) per
wave vs phase 1:

| | phase 1 | phase 2 | AMD |
|--|--:|--:|--:|
| conflict | 677.2 | **96.8** | 80.6 |
| wait-LDS | 206.5 | **103.0** | 49.8 |
| busy | 2715.6 | 2560.4 | 769.6 |
| MFMA tot | 12384 | 12384 | 12384 |
| VGPR | 108 | 124 | 108 |

Conflict/wave is now **1.2×** AMD (was 8.4×). Phase 3 is leftover
`v_perm` / `bpermute`.

### 3. Cut leftover permute / bpermute / extra waits

HEAD decode still has **34 `v_perm_b32`** and **12 `ds_bpermute_b32`**
versus AMD **0 / 0**, and **106 `s_waitcnt`** versus **33–34**. After
the maps are right, remove packing that exists only because the old
generic TV copies needed it.

- [x] Identify each `v_perm` / `ds_bpermute` cluster in the decode
      ISA and delete the FlyDSL that produces it if AMD’s fragment
      map does not need it (softmax shuffles may remain).
- [x] Do not add permute to “fix” a wrong map; go back to phase 1/2.
- [x] Oracle + ISA line. Keep if permute/bpermute/wait counts fall
      toward AMD even if wrapper µs is flat.
- [x] **Done when:** decode ISA has no `ds_bpermute` and `v_perm` is
      near AMD (ideally 0), or a dated note explains the leftover
      (e.g. softmax pack that AMD does in SALU).

Kept (2026-09-24). Decode-only: Q OOB heads are a packed `cndmask`
(`q_live.select(q_vec, 0)`), not a per-element f32 round-trip, and
token-live for softmax is `ballot` + bit test, not `shuffle_idx`.
Dropped the unused `m_final` `shuffle_idx` pack (already DCE on
prefill). Prefill ISA sha still `64ee586155fc6a63`. Oracle
decode+prefill: **2 passed**.

**ISA** (`tickets/1047/tmp/k2_decode_perm_isa/launch/22_final_isa.s`)
vs phase 2:

| | phase 2 | phase 3 | AMD decode |
|--|--:|--:|--:|
| `v_perm` | 34 | **2** | 0 |
| `ds_bpermute` | 12 | **8** | 0 |
| `s_waitcnt` | 103 | 108 | 33 |
| K32 / K16 | 8 / 4 | 8 / 4 | 8 / 4 |
| `ds_write_b64` / tr16 | 4 / 4 | 4 / 4 | 4 / 4 |

The 32 Q-load `v_perm` and 4 live-mask `ds_bpermute` are gone.
Leftover **2 `v_perm`** pack four softmax P bf16 into PV-A; leftover
**8 `ds_bpermute`** are `shuffle_idx(alpha)` (4) and epilogue
`l_final` dens (4). Overlay QK C is token-major in-lane while PV C /
store is 4 heads, so those gathers stay; AMD’s `Q@K` C is already
head-major and uses SALU/`permlane` instead. Not a map bug.

**Wall** (five-run median, interleaved; not a keep-gate) vs phase 2
rerun 19.22 / 19.40 / 155.98:

| M | phase 3 FlyDSL | AMD wrapper |
|--:|--:|--:|
| 1 | 20.53 | 37.69 |
| 8 | 21.05 | 38.10 |
| 512 | 156.21 | 199.72 |

**Kernel-trace M=1 split:** **10.381 µs** vs phase 2 **9.893 / 10.503**
vs AMD **6.391**. Waitcnt did not fall (phase 4).

### 4. Waitcnt / barrier schedule toward AMD’s decode mix

AMD decode: **33–34 waitcnt**, **5 barriers**, wait-stall is VM-heavy
(`vmcnt(3/5/1)`), not a deep prefill-style `vmcnt(31)` ladder. FlyDSL
HEAD is waitcnt-dominated (~73% of M=8 ATT stalls).

- [x] Place `lgkmcnt` / `vmcnt` next to the AMD decode sites: after
      K publish, after V overlay, before QK/PV uses. Do not hoist V
      loads across K-publish (inherited DNR).
- [x] Keep 5 barriers if that is still AMD’s count; do not add a
      sixth to paper over a map bug.
- [x] Oracle + ATT/PMC. Conflict and wait-LDS should fall with the
      maps; this phase is for remaining schedule.
- [x] **Done when:** static waitcnt is in AMD’s band (tens, not
      ~100) or further cuts require compiler dest-pinning that is
      documented as blocked.

Kept as **measurement + dest-pinning blocked** (2026-09-24). No
kernel keep: extra `vmcnt(0)` and bursting QK A / PV B into named
dests were tried and reverted. Prefill ISA sha still
`64ee586155fc6a63`. Oracle decode+prefill: **2 passed**.

**Count split only.** Phase 0–3 `s_waitcnt ~106–108` mixed the merge
kernel’s `vmcnt(31)` ladder (from `qsa_k2_family_a_port_merge`,
`k2_decode_perm_isa/launch/22_final_isa.s` ~line 735+). Split body
(through `.LBB0_6`) is **32 waitcnt / 4 barriers** vs AMD split
**33 / 5**. Whole-file FlyDSL stays ~108 because merge is unchanged.

**AMD sites already present on overlay decode**
(`k2_decode_perm_isa/launch/22_final_isa.s` tile loop):

| Site | FlyDSL split | AMD split |
|--|--|--|
| K VMEM vs K LDS | `vmcnt(3)` then `vmcnt(2)` around the two `ds_write_b128` | `vmcnt(2)` / `vmcnt(1)` |
| K publish | `lgkmcnt(0)` + `s_barrier` | same |
| QK A | first pair `lgkmcnt(1)`, then **`lgkmcnt(0)` on reused `v[96:99]`** | `lgkmcnt(1)` + `offset:256` second dest |
| K-read vs V overlay | `s_barrier` (compiler may sit it before the last K32) + `vmcnt(0)` | `s_barrier` then V `ds_write_b64` |
| V publish | `lgkmcnt(0)` + `s_barrier` | same |
| PV tr16 | `lgkmcnt(0)` on reused `v[98:99]`, 4 addr VGPRs | `lgkmcnt(3..0)`, one addr + 128/256/384 |

V `buffer_load_dwordx4` already issues with K (same cluster as AMD)
and stays in flight across K-publish (`vmcnt(3/2)` only waits K).
That is not the inherited hoist-V-before-K-publish DNR (that DNR was
waiting VMEM behind the K LDS barrier). Do not add a 5th split
barrier: AMD’s extra one is Q LDS publish; overlay Q lives in
registers.

**Dest-pinning blocked.** Bursting eight QK `load_amd_k8` and four
tr16 into distinct Python dests still lowered to `v[96:99]` /
`v[98:99]` (`tickets/1047/tmp/k2_decode_waitcnt_isa/`). An authored
`s_waitcnt vmcnt(0)` before V overlay duplicated the compiler’s
wait (32→33 split waitcnt). Inline-asm dest constraints remain the
parent prefill DNR and are not reopened here.

**Wall** (five-run median, interleaved `CACHE=0`; not a keep-gate):

| M | FlyDSL | AMD wrapper |
|--:|--:|--:|
| 1 | 19.57 | 37.62 |
| 8 | 19.98 | 37.42 |
| 512 | 158.88 | 201.71 |

Kernel-trace / PMC from phase 3 still apply (no ISA keep): M=1 split
**10.381 µs** vs AMD **6.391 µs**. Remaining split gap is occupancy
(phase 5 `ns64` at M=1) plus overlay QK C still token-major (no
AMD `offset:256` dual dest), not a missing waitcnt.

Leftover mix vs AMD: **20 `lgkmcnt(0)` vs 9**, **4 barriers vs 5**.
Static split waitcnt is in AMD’s tens. Phase 5.

### 5. Split-count specializations (`ns64` at M=1, `ns32` at M=8)

Live AMD already uses **one tile body** and **two split counts**:
`base_programs ≤ 4` → 64 splits / 128 WGs; `4 < base_programs < 32`
→ 32 splits / 512 WGs. FlyDSL currently caps both at 32, so M=1
launches **half** AMD’s waves. Parent DNR against 64 splits was on
the old body. Re-measure only after phases 1–4.

- [x] Add an `ns64` compile of the **same** decode tile; do not fork
      math. Dispatch: 64 when `M * Hk ≤ 4`, 32 otherwise (decode
      BN16 band).
- [x] Measure M=1 `ns32` vs `ns64` and M=8 `ns32` vs `ns64` (split,
      merge, total). Expect M=1 to need 64 to match AMD’s grid;
      M=8 should stay 32 (AMD does; extra splits add merge).
- [x] Merge cost is a check, not a rewrite. If `ns64` wins the
      split and loses the wrapper only because merge got worse,
      record it; merge work is a follow-on, not phases 1–4.
- [x] **Done when:** launch policy matches AMD’s decode band, or a
      dated measurement shows `ns32` already matches AMD split at
      both M (then leave dispatch).

Kept (2026-09-24). `_launch_config` now matches live AMD: `ns64`
when `rows * Hk ≤ 4`, `ns32` for the rest of the BN16 band. Same
tile body; `_plan(..., n_splits)` already compiles a distinct
HSACO. Prefill `M=512` stays `ns1`. Oracle decode+prefill: **2
passed**. Prefill ISA sha still `64ee586155fc6a63`.

**Kernel-trace** (GPU 6, `CACHE=1` after fill, 40 timed iters;
dumps `tickets/1047/tmp/k2_decode_ns/`):

| | split µs | merge µs | split+merge | grid (threads x y z) |
|--|--:|--:|--:|--|
| FlyDSL M=1 ns32 | 11.040 | 3.640 | 14.680 | 256×2×**32** (64 WGs) |
| FlyDSL M=1 ns64 | **7.800** | 12.360 | 20.160 | 256×2×**64** (128 WGs) |
| AMD M=1 | **6.440** | 3.500 | 9.940 | 256×2×64 (128 WGs) |
| FlyDSL M=8 ns32 | **12.820** | 3.720 | 16.540 | 2048×2×**32** (512 WGs) |
| FlyDSL M=8 ns64 | 13.860 | 11.120 | 24.980 | 2048×2×**64** (1024 WGs) |
| AMD M=8 | **10.440** | 2.880 | 13.320 | 2048×2×32 (512 WGs) |

M=1 `ns64` cuts split **11.04 → 7.80 µs** (1.71× AMD → **1.21×**).
Merge goes **3.64 → 12.36 µs** (64 live lanes, loop of 64). M=8
`ns64` loses both split and merge; leave M=8 at 32.

**Wall** (five-run median, interleaved `CACHE=0`; policy ns64/ns32;
not a keep-gate) vs phase 4 ns32/ns32 19.57 / 19.98 / 158.88:

| M | FlyDSL | AMD wrapper |
|--:|--:|--:|
| 1 | 20.43 | 37.18 |
| 8 | 20.46 | 37.51 |
| 512 | 159.31 | 202.11 |

Wrapper M=1 is slightly slower than ns32 because merge ate the
split win. Split is the campaign bar; merge is a follow-on, not a
revert of `ns64`. `ns32` does **not** match AMD split at M=1, so
dispatch stays AMD-matched.

**Compile specialization.** Merge folds `n_splits` into the lane
mask (`lane < n_splits`) and the loop-carried split scan; split
tile bounds also fold. Two HSACOs (`ns32` / `ns64`), not one
runtime split count. Phase 6.

### 6. Stop: decode split matches or exceeds Live AMD

- [ ] Same-session five-run medians, `CACHE=0`, GPU 6: FlyDSL decode
      split ≤ Live AMD split at **M=1 and M=8**. Wrapper may still
      include merge; if split is ahead and wrapper is not, say so
      and open a merge follow-on rather than widening this plan.
- [ ] Prefill `M=512` still at or better than the st64-imm keeper
      band (no silent decode-branch leak).
- [ ] ISA cheat-sheet updated: opcode mix vs AMD decode, PMC
      conflict/wait, ATT stall mix.
- [ ] **Done when:** the success bar in the intro is met. Do not
      keep iterating permute/wait folklore after that.

## Do not retry (this campaign)

Paste misses here **and** in `SILOTIGER-1047-plan.md`. Inherited DNR
is not repeated unless a decode retry is proposed.

- Overlay decode QK/PV **named-dest burst** (eight `load_amd_k8` or
  four tr16 live at once) or an extra `s_waitcnt vmcnt(0)` before V
  overlay, hoping for AMD `lgkmcnt(1)` / fewer waits. GPU 6 / gfx950
  stayed exact; ISA still reused `v[96:99]` / `v[98:99]` and the
  extra vmcnt duplicated the compiler. Kernel restored. Inline-asm
  dest pins are parent DNR.

## Non-goals (do not pull into this plan)

- Changing GPU from the locked `HIP_VISIBLE_DEVICES=6`.
- Prefill store/MFMA work (`write2st64`, packed K32 PV, VMEM depth).
- Merge-kernel ISA matching unless phase 5/6 proves merge is the
  leftover after split is at or ahead of AMD.
- Restoring C/P/metadata LDS or starting a new decode from a blank
  kernel.
- Family B, gfx942, indexer K1, vLLM opt-in.
- Occupancy of unused CU slots as a goal of its own.
- A 3% wall-clock keep-gate (that was the prefill overlay campaign).

## Open questions (resolve into locks; do not guess in code)

Leave these open until the named phase produces evidence. When
resolved, move the answer into **Locked decisions** and check the
item.

- [x] Whether AMD decode K XOR is the same byte formula as prefill
      `amd_k_elem` scaled to 16 tokens / 4 waves, or a distinct
      decode map (phase 1). **Distinct.** Prefill uses
      `(tid*16)^((tid&0x60)>>1)` plus `group/quarter` * 2048/8192
      into 32 KiB. Decode uses `(tid*16)^((tid&0xe0)>>1)` and
      `(that^0x80)+4096` into 8 KiB. Do not reuse `amd_k_elem` on
      decode.
- [x] Whether AMD’s 4 `ds_write_b64` are one 8×bf16 vector split in
      the compiler or an authored 64-bit TV copy (phase 2).
      **Compiler split of two 8×bf16 gathers.** AMD ISA writes
      `v[78:79]`/`v[80:81]` then `v[86:87]`/`v[88:89]` from two
      `buffer_load_dwordx4`. FlyDSL authors the same: lo/hi 4 of
      each gather with `UniversalCopy64b`. Not a 64-bit TV tile and
      not `write2st64`.
- [x] Whether leftover FlyDSL `v_perm` is softmax/epilogue or K/V
      packing (phase 3). **Q packing, not K/V maps.** 32 of 34
      `v_perm` were the Q f32 round-trip; those are gone. The last 2
      pack P for PV-A. 4 of 12 `ds_bpermute` were token-live
      `shuffle_idx` (now `ballot`); the remaining 8 are alpha +
      epilogue `l_final` because overlay C is token-major and the
      store is 4 heads.
- [x] Whether M=1 still needs `ns64` after the tile matches AMD, or
      the current 32-split grid becomes enough (phase 5). **Needs
      ns64.** M=1 split 11.04 → 7.80 µs vs AMD 6.44; ns32 is still
      1.71×. Merge 3.64 → 12.36 is a follow-on, not a reason to
      keep half AMD’s waves.
- [x] Whether one `ns*` HSACO with a runtime split count is enough,
      or Triton-style constexpr `NUM_SPLITS` must remain a compile
      specialization because merge `BLOCK_SPLITS` and split tile
      bounds fold (phase 5). **Compile specialization.** Merge
      folds `n_splits` into `lane < n_splits` and the split scan;
      `_plan` already keys the HSACO on `n_splits`.

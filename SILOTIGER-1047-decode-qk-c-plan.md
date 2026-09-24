# SILOTIGER-1047 — Decode QK-C / softmax dataflow vs Live AMD

Close the **remaining** family A FlyDSL overlay decode gap after the
K/V-map campaign. Physical decode K/V publication is already AMD-class
(conflict and wait-LDS now **better** than Live AMD). Split still trails
**~21–23%** because overlay QK C is token-major (`K @ Q^T`) and LDS
reads serialize on reused destinations. Merge at `ns64` is a second,
wrapper-visible hole.

Parent: [SILOTIGER-1040](https://amd.atlassian.net/browse/SILOTIGER-1040).
Ticket: [SILOTIGER-1047](https://amd.atlassian.net/browse/SILOTIGER-1047).
Ticket-wide plan: `SILOTIGER-1047-plan.md`. Prefill overlay:
`SILOTIGER-1047-optimize-overlay-split-plan.md` (frozen). Map campaign
that landed K/V + `ns64`/`ns32` and then **stopped short of the split
bar**: `SILOTIGER-1047-bridge-decode-gap-plan.md`. Do not reopen that
campaign’s phases; this file is the follow-on.

**Kernel name.** **K2 overlay decode**. Source:
`aiter/ops/flydsl/kernels/qsa/k2_family_a.py`. Split HSACO:
`qsa_k2_family_a_port_split_ps16_bn16_blk256_ns{32,64}_qkk32`. Merge:
`qsa_k2_family_a_port_merge_ns{32,64}_…`. Prefill `bn64_blk128_ns1` is
a no-regress row only.

**Bar:** width-2051 `L=512` family A GQA on GPU 6 / gfx950 vs live
`qsa_sparse_paged_attention`. Headline is **decode split** at `M=1`
and `M=8` (kernel-trace, interleaved, same process). Wrapper is
recorded; at `M=1` it is currently merge-dominated and is **not** the
split keep signal.

Same-session five-dispatch kernel trace, 2026-09-24, GPU 6, `CACHE=0`
(`tickets/1047/tmp/k2_decode_final/ktrace_same_session/`):

| M | FlyDSL split | AMD split | ratio | FlyDSL merge | AMD merge |
|--:|--:|--:|--:|--:|--:|
| 1 (`ns64`) | **8.840 µs** | **7.280 µs** | **1.21×** | 12.280 | 6.040 |
| 8 (`ns32`) | **14.240 µs** | **11.600 µs** | **1.23×** | 3.680 | 5.480 |

The campaign is a **success** when FlyDSL decode **split** matches or
exceeds Live AMD at **both** `M=1` and `M=8`. If split is at or ahead
and wrapper is not, finish the merge phase here rather than claiming
a serving win. Intermediate steps **record** wall clock; they **do
not** 3% keep-gate on it.

Work the phases **in order**. Later items assume earlier ones have
landed. Leave checkboxes unchecked until that item is done; paste
tables, ISA lines, and notes under the relevant phase as evidence.
Stage then one-line commit when that is the ticket convention.

## Progress

- [ ] 0. Pin remaining split sites from `k2_decode_final` (no kernel edit)
- [ ] 1. Head-major decode QK C (decode-only; keep overlay LDS)
- [ ] 2. Drop leftover `v_perm` / `ds_bpermute` that exist only for token-major C
- [ ] 3. Overlap QK/PV LDS reads (compiler dests, not named-dest burst)
- [ ] 4. `ns64` merge: vectorized `BLOCK_SPLITS`-style scan
- [ ] 5. Stop: decode split ≤ Live AMD at M=1 and M=8

## Why the map campaign is not the leftover

Final PMC (one timed dispatch; SE instances summed;
`tickets/1047/tmp/k2_decode_final/pmc_{flydsl,amd}_m{1,8}/`):

| M / backend | waves | busy/w | wait-LDS/w | conflict/w | MFMA total |
|--|--:|--:|--:|--:|--:|
| 1 FlyDSL ns64 | 512 | **965.5** | 36.5 | 48.4 | 12,384 |
| 1 AMD ns64 | 512 | 760.9 | 48.3 | 80.6 | 12,384 |
| 8 FlyDSL ns32 | 2,048 | **399.9** | 129.6 | 96.8 | 99,072 |
| 8 AMD ns32 | 2,048 | 304.1 | 150.0 | 161.2 | 99,072 |

K/V maps won: FlyDSL conflict and wait-LDS are **below** AMD while
MFMA totals match. The split gap tracks **~1.27–1.32× busy cycles**.

Final ATT M=8 (`--att-gpu-index 6`;
`tickets/1047/tmp/k2_decode_final/att_{flydsl,amd}_m8/`):

| | FlyDSL | AMD |
|--|--:|--:|
| total stall | **176,436** | 96,520 (~1.83×) |
| `s_waitcnt` share | 75.9% | 63.8% |
| `vmcnt` share of all stall | 47.1% | 50.6% |
| `lgkmcnt` share of all stall | **28.8%** | **13.2%** |
| `s_barrier` | 8.1% | 6.8% |

Hot FlyDSL stalls vs AMD (same CU-0 wave sample):

| FlyDSL | stall | AMD analogue |
|--|--:|--|
| `s_waitcnt vmcnt(0)` (two sites, hit 32) | 27,880 + 14,104 | AMD `vmcnt(0)` is **0.4%** of stall |
| `s_waitcnt vmcnt(3)` | 22,000 | AMD top: `vmcnt(3)` 16,192 |
| `s_waitcnt vmcnt(7)` | 14,272 | AMD pipelines `vmcnt(5/1)`, not a deep 7 |
| `s_waitcnt lgkmcnt(0)` (prologue + tile) | **27.4%** of all stall | AMD `lgkmcnt(0)` **7.2%**; extra overlap is `lgkmcnt(1)` 5.7% |
| `ds_bpermute_b32` | 968 (0.5%) | 0 |

ISA leftover vs AMD decode (`tickets/1047/tmp/k2_decode_final/isa_m{1,8}/`):

| | FlyDSL split | AMD decode |
|--|--:|--:|
| `ds_write_b128` / `b64` | 2 / 4 | 4 / 4 (extra 2 b128 are Q LDS) |
| `ds_read_b128` / tr16 | 8 / 4 | 16 / 4 (extra 8 b128 are Q) |
| K32 / K16 | 8 / 4 | 8 / 4 |
| split `s_waitcnt` / `s_barrier` | 32 / 4 | 33–34 / 5 |
| `v_perm` / `ds_bpermute` | **2 / 8** | **0 / 0** |
| VGPR | 113 | 106 |
| QK C layout | token-major (`K @ Q^T`) | head-major (`Q @ K`) |
| QK/PV LDS dests | reused `v[96:99]` / `v[98:99]` + `lgkmcnt(0)` | distinct dests + `offset:256` / `lgkmcnt(1)` |

The 8 `ds_bpermute` are `shuffle_idx(alpha)` ×4 and epilogue
`shuffle_idx(l_final)` ×4: overlay C lives on **token** lanes, PV
accumulators and the store live on **head** lanes. The 2 `v_perm`
pack four softmax P bf16 into PV-A. AMD’s C is already head-major, so
those gathers do not exist. Direct permute/wait folklore was tried in
the map campaign (phases 3–4) and did not close the split gap.

## Locked decisions

These locks apply to **this campaign** unless a later note explicitly
supersedes them. Ticket-wide locks in `SILOTIGER-1047-plan.md` still
apply. Prefill DNR and map-campaign DNR still apply.

### Skills

- **Skills (read, do not recall).** Before writing or reviewing FlyDSL
  for this ticket, Read
  `.claude/skills/flydsl-kernel-authoring/SKILL.md` and follow it.
  For `op_tests/test_flydsl_qsa.py`, also Read
  `.claude/skills/aiter-op-test/SKILL.md`.
  Do not start kernel code from memory of those skills.

### Test environment

- **Test environment:** run all tests/benches in **`flydsl_venv`** **GPU 6**
  (`HIP_VISIBLE_DEVICES=6`).
- **Compile cache.** After kernel-source edits, run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or clear `~/.flydsl/cache`).
- **`CACHE=0` needs `ROCM_PATH=/root/.flydsl/toolkit`.**
- **Correctness after a kernel exists:**
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    ROCM_PATH=/root/.flydsl/toolkit \
    python3 -m pytest op_tests/test_flydsl_qsa.py -q
  ```
  Focused decode oracle (`test_k2_family_a_decode_matches_oracle` and
  the prefill counterpart) is enough between steps.
- **Perf measurement (not a keep-gate).** After every phase, record
  width-2051 `L=512` `M∈{1,8}` (and `M=512` as a sanity row) vs live
  AMD. Prefer interleaved kernel-trace split **and** merge. Dump ISA
  (`FLYDSL_DUMP_IR=1`) when the QK-C / softmax / merge line moved.
  **Do not revert solely because wall clock is flat.** Revert if
  oracle fails, if the change moves decode **away** from AMD’s C
  layout / dest overlap, if it touches frozen prefill, or if it
  retries a DNR below.
- **ATT GPU index is physical.** `--att-gpu-index 6`. Decode ATT uses
  `M=8` as the body proxy (same BN16 HSACO as M=1) plus M=1 when merge
  `ns64` changes.
- **Count split waitcnt on the split body only.** Whole-file ~108 is
  merge’s `vmcnt(31)` ladder.

### Scope and bar

- **Decode dataflow, then merge.** Do not spend this campaign on K/V
  XOR maps, extra MFMA, indexer K1, family B, or undoing overlay back
  to C/P/metadata LDS.
- **Family A GQA, live AMD is the success bar.** `err=0` vs the
  oracle (`checkAllclose` `1e-2`).
- **No 3% keep-gate on wall clock.** A step that is oracle-correct
  **and** moves QK C toward AMD’s head-major fragment (or merge toward
  AMD’s `BLOCK_SPLITS` load) is kept even if wrapper µs is flat.
- **Prefill is frozen.** All experiments stay behind
  `const_expr(decode_tr_pv)` (`use_k32 and block_n == 16`). Do not
  edit BN64 `amd_k_elem` / `write2st64` / tr16-immediate lattice.
  Record `M=512` so a leak is visible.
- **One logical decode body.** M=1 and M=8 share the BN16 / 256-thread
  / 4-wave tile. Keep AMD-matched dispatch: `ns64` when
  `rows * Hk ≤ 4`, `ns32` otherwise.
- **Do not chase extra MFMA.** Stay 8× K32 QK + 4× K16 PV.
- **Overlay LDS contract stays.** One MMA scratch (K then V), 8 KiB,
  softmax m/l in registers, **no C/P/metadata LDS**. What this
  campaign **does** reopen for decode is the **register** QK C map:
  `K @ Q^T` was kept so C could feed PV-A without a P-LDS. Phase 6 of
  the map campaign showed that bargain is the leftover split cost.
  A decode-only `Q @ K` (or equivalent head-major C) is in scope **if
  it does not bring C/P LDS back**.

### Target Live AMD decode dataflow

Live AMD (`qsa_vllm_amd.py` `_qsa_sparse_paged_gqa_splitk`) compiles
`Q @ K` so QK C is `[M, N]` head-major. Softmax and PV-A are already
in that layout; Q stages through LDS (the extra 2 `ds_write_b128` /
8 `ds_read_b128`). Overlay keeps Q in registers. Matching AMD does
**not** require Q LDS; it requires C’s **lane ownership** to match
PV-A / store (4 heads), so `shuffle_idx(alpha)` and `l_final` dens
go away.

Merge: AMD `BLOCK_SPLITS = next_power_of_2(NUM_SPLITS)` and a masked
vector load of split LSEs / partials (`qsa_vllm_amd.py` ~426–433).
FlyDSL merge is 128 threads, `lane < n_splits`, then a loop-carried
`range(0, n_splits)` over `partial_out[s, …]` — at `ns64` that is 64
serial VMEM hits (the `vmcnt(31)` ladder). M=8 `ns32` merge is
already **ahead** of AMD (3.68 vs 5.48 µs); do not “fix” ns32 merge
by copying ns64 pain.

## Profiling

Reuse `tickets/1047/tmp/profile_k2_gqa.py`. Prefer
`tickets/1047/tmp/k2_decode_qk/` (or similar) for this campaign’s
dumps; leave `k2_decode_final/` as the map-campaign snapshot.

**Primary counters to watch now:** busy/wave, ATT `lgkmcnt(0)` share,
ATT `vmcnt(0)` share, static `v_perm` / `ds_bpermute`, split µs.
Conflict/wave is a **guard** (must not return to hundreds); it is
not the optimization target.

## Do not reopen (this campaign)

Inherited unless a later lock here explicitly reopens it:

- Decode K/V XOR maps (`amd_decode_k_elem`, 4× `ds_write_b64`, inverse
  tr16). Already AMD-class on conflict.
- Overlay decode QK/PV **named-dest burst** (eight `load_amd_k8` or
  four tr16 live at once) or an extra `s_waitcnt vmcnt(0)` before V
  overlay. ISA still reused `v[96:99]` / `v[98:99]`. Kernel restored.
- Inline-asm dest pins / `=&v` constraints (parent prefill DNR).
- Prefill `write2st64` / packed K32 PV / `amd_k_elem` on decode.
- Hoist V `buffer_load` before K-publish `lgkmcnt(0)` / `s_barrier`.
- Extra split barrier to paper over Q-in-registers (AMD’s 5th is Q
  LDS).
- Restoring C/P/metadata LDS or starting decode from a blank kernel.
- `BLOCK_N=64` / 8 splits for M=8; four-wave BN64 prefill.
- Unrolling the 32-split LSE merge, 256-thread merge, BF16 partials
  (parent DNR). Phase 4 is a **vectorized** `BLOCK_SPLITS` load, not
  those retries.
- Occupancy of unused CU slots as a goal of its own.
- A 3% wall-clock keep-gate.

## Subtasks

### 0. Pin remaining split sites from `k2_decode_final`

No kernel edit. Annotate the decode ISA so phase 1 edits the right
FlyDSL, not wait folklore.

- [ ] On `isa_m8/launch/22_final_isa.s` (split body through the tile
      loop, not merge): mark (a) 8 QK `ds_read_b128` into reused
      `v[96:99]`, (b) 4 tr16 into reused `v[98:99]`, (c) 4+4
      `ds_bpermute` around softmax/epilogue, (d) 2 `v_perm` P-pack,
      (e) the hot `vmcnt(0)` that ATT attributed 27,880 stall.
- [ ] Cross-check AMD decode ISA
      (`tickets/1047/tmp/k2_decode_baseline/amd_m8_split.s`): QK reads
      use a second dest + `offset:256` and `lgkmcnt(1)`; softmax has
      `v_permlane*` and **no** `ds_bpermute`.
- [ ] Write the mapping “FlyDSL construct → ISA cluster” under this
      phase (Q as MMA B, `shuffle_idx(alpha)`, `Vector.from_elements`
      P pack, `load_amd_k8` dest reuse).
- [ ] **Done when:** that map is pasted here. No kernel diff.

### 1. Head-major decode QK C (decode-only)

Reopen overlay’s decode-only register contract: C should be owned by
the same lanes that own PV-A and the 4-head store, like AMD `Q @ K`.

- [ ] Behind `const_expr(decode_tr_pv)` only, form Q as the **A**
      operand and K-from-LDS as **B** (or an equivalent tiled MMA
      that yields head-major C). Keep 8× K32. Do not add Q LDS.
- [ ] Softmax m/l stay in registers. Tile max / `p_sum` reductions
      must run across **tokens**, not require `shuffle_idx` to move
      `alpha` onto head lanes.
- [ ] PV-A must consume that C (or a cheap in-register pack that is
      not 2× `v_perm` + 8× `bpermute`). No P-LDS.
- [ ] Oracle decode+prefill. Record ktrace split/merge, ISA C
      comments, `v_perm` / `bpermute` counts, busy/wave. Keep if C
      ownership moved toward AMD even if split µs is flat.
- [ ] **Done when:** decode QK C is head-major in-lane (documented
      against AMD `[M,N]`), prefill ISA sha still the st64-imm
      keeper (`64ee586155fc6a63`), **or** a dated miss shows that
      `Q @ K` cannot feed PV-A without P-LDS — then stop and do not
      silently restore C/P LDS.

If phase 1 is a miss, do **not** proceed to pretend-permute cleanup.
Record DNR and skip to phase 4 only if split is still ~1.2× and merge
is the serving-visible hole; otherwise stop the campaign.

### 2. Drop leftover `v_perm` / `ds_bpermute`

Only after phase 1’s C map is real.

- [ ] Delete decode `shuffle_idx(alpha)` / `shuffle_idx(l_final)` if
      head-major C made them dead. Delete the 2 `v_perm` P pack if
      PV-A already holds packed bf16.
- [ ] Do not add permute to “fix” a wrong C map; go back to phase 1.
- [ ] Oracle + ISA. Keep if `v_perm`/`ds_bpermute` fall toward AMD
      0/0 even if wrapper µs is flat.
- [ ] **Done when:** decode ISA has no `ds_bpermute` and `v_perm` is
      0, or a dated note explains a leftover that AMD also emits
      (`v_permlane*` for the softmax tree is allowed; `ds_bpermute`
      is not).

### 3. Overlap QK/PV LDS reads (compiler dests)

ATT leftover after maps: FlyDSL `lgkmcnt(0)` is 27.4% of stall vs
AMD 7.2%; AMD spends 5.7% on `lgkmcnt(1)` because the next K32/tr16
already has a distinct dest. Kernel-side bursting is DNR.

- [ ] Change how FlyDSL **allocates** LDS-read destinations so
      successive `load_amd_k8` / tr16 are not coalesced into
      `v[96:99]` / `v[98:99]`. That is a compiler / lowering change
      (or a supported dest-coloring API), **not** eight Python
      temps and **not** inline-asm pins.
- [ ] Success signal in ISA: QK pair waits `lgkmcnt(1)` on a live
      next dest; PV tr16 uses one addr + 128/256/384 (or four dests
      that are actually distinct). ATT `lgkmcnt(0)` share should
      fall toward AMD’s ~7%.
- [ ] If the compiler cannot color those dests in this ticket,
      document blocked and skip. Do not retry phase-4 named-dest
      burst.
- [ ] **Done when:** decode QK/PV dests overlap like AMD, or
      blocked-with-evidence.

### 4. `ns64` merge: vectorized `BLOCK_SPLITS`-style scan

Split can win at M=1 and still lose the wrapper: phase-5/6 ktrace
had FlyDSL merge **12.28 µs** vs AMD **6.04 µs** at `ns64`, while
`ns32` merge is already faster than AMD.

- [ ] Keep 128-thread merge. Specialize `ns64` so split LSE/partial
      loads match AMD’s masked `arange(0, BLOCK_SPLITS)` gather, not
      a 64-trip `range` over `partial_out[s]`.
- [ ] Do not unroll ns32 (parent DNR). Do not 256-thread merge. Do
      not BF16 partials.
- [ ] Measure M=1 ns64 **and** M=8 ns32 (ns32 must not regress).
- [ ] **Done when:** M=1 merge is in AMD’s band (~6 µs on this
      machine’s same-session traces) or a dated miss shows the
      ladder is compiler VMEM scheduling, not the loop shape.

### 5. Stop: decode split ≤ Live AMD at M=1 and M=8

- [ ] Same-session five-run medians, `CACHE=0`, GPU 6: FlyDSL decode
      split ≤ Live AMD split at **M=1 and M=8**.
- [ ] If split is ahead and wrapper is not, say so; phase 4 must be
      done or recorded as a follow-on with numbers.
- [ ] Prefill `M=512` still in the st64-imm keeper band (~159 µs).
- [ ] ISA cheat-sheet: C layout, dest overlap, `v_perm`/`bpermute`,
      PMC busy/conflict, ATT stall mix.
- [ ] **Done when:** the success bar in the intro is met. Do not
      keep iterating waitcnt folklore after that.

## Non-goals (do not pull into this plan)

- Changing GPU from the locked `HIP_VISIBLE_DEVICES=6`.
- Prefill store/MFMA work.
- Family B, gfx942, indexer K1, vLLM opt-in.
- Restoring Q LDS just to copy AMD’s extra 2+8 b128.
- Occupancy of unused CU slots as a goal of its own.

## Open questions (resolve into locks; do not guess in code)

- [ ] Whether decode `Q @ K` can feed PV-A without P-LDS or a
      bpermute transpose of P (phase 1). If no, this campaign’s
      split lever is gone; do not smuggle C/P LDS back.
- [ ] Whether FlyDSL can color successive LDS-read dests without
      inline-asm (phase 3). If no, document compiler work as out of
      ticket and live with `lgkmcnt(0)`.
- [ ] Whether the merge `vmcnt(31)` ladder is the 64-trip `range` or
      LLVM serializing 64-bit partial loads even after a vectorized
      gather (phase 4).
- [ ] Whether leftover FlyDSL `vmcnt(0)` (25% of ATT stall) is V
      overlay waiting the in-flight K/V gathers, or softmax/epilogue
      VMEM. Phase 0 must name the ISA site before anyone authors
      another wait.

# Persistent Push-group Dispatch→GEMM1 Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On gfx1250, overlap the P2P FP8 payload transfer of later push-group GEMM1 M-tiles with the persistent GEMM1 computation of earlier fully arrived M-tiles, without changing GEMM2 P2P scatter/combine behavior.

**Architecture:** Implement one fixed-grid FlyDSL stage-1 mega kernel with three CTA roles: a small set of dispatch producers, one planner CTA, and the remaining CTAs as persistent GEMM1 consumers. The planner first completes a device-only per-source/per-local-expert count-and-prefix protocol, allowing every producer to write a contiguous span in the Q1 fixed-slot layout. Producers publish only completed tile fragments into a per-tile arrival counter. A persistent consumer waits until the counter equals that tile's planned valid-row count, then reuses the existing contiguous wave-specialized TDM GEMM1 path. This matches DeepGEMM's “metadata first, payload block arrival then compute” model; it deliberately does not claim to overlap the metadata/count phase itself with GEMM.

**Tech Stack:** Python, PyTorch CUDA Graph, FlyDSL, ROCm gfx1250, cco symmetric arena (`Window.lsa_ptr`), `flydsl_prims` system-scope release/acquire primitives, TDM a8w4 WMMA, pytest, torchrun.

## Global Constraints

- This plan builds on Q1 `AITER_EP_PUSH_GROUP=1`; the existing atomic fixed-slot path remains the fallback.
- Apply only to `a8w4` with `AITER_EP_FP8_TRANSPORT=1`; dispatch moves FP8 payload plus E8M0 scale, never BF16 plus FP8 in parallel.
- New gate: `AITER_EP_PUSH_GROUP_OVERLAP=1`, default `0`. Gate off must retain the current dispatch → finalize → preshuffle → GEMM1 sequence and output.
- Use one mega kernel, not two concurrently launched streams. A persistent consumer grid can otherwise occupy all CUs and starve dispatch CTAs.
- Reserve `0 < dispatch_ctas < num_cu`, one planner CTA, and at least one consumer CTA. Reject invalid environment settings before launch.
- Cross-CTA/device publication uses only `flydsl_prims.py`: `atomic_add_global`, `store_i32_system`, `fence_system_release`, `fence_system_acquire`, and volatile `spin_until_eq_i32`.
- The tile-arrival counter is additive, not a boolean. A consumer may compute tile `(expert, tile_idx)` only after `arrived_rows == expected_rows`.
- All counters use a two-epoch generation or are reset in the captured graph before every launch; no host `.item()`, dynamic launch, or D2H synchronization is allowed.
- Preserve the fixed-slot address `row = local_expert * CAP + local_row`; preserve existing GEMM2 `pg_rowmap` scatter.
- The initial scope ends after GEMM1. GEMM2 and combine remain existing kernels, launched after the stage-1 mega kernel completes.

---

## Why this supersedes the cross-CTA ring plan

The earlier global-ring proposal launches producer and consumer as separate kernels on the same stream. That does not overlap GPU kernels: stream ordering serializes them. The current Q1 persistent GEMM1 also cannot overlap dispatch because `launch_push_group_finalize()` creates `num_valid`, `tile_row_base`, `expert_ids`, and `tile_valid` only after dispatch completes.

The reference `origin/megamoe_v2` branch has the required CTA partition/work-pool shape but its fixed-slot path waits for every source before `emit_direct_fixed_slot_finalize()` and only then starts GEMM work. DeepGEMM goes one step further: it first finalizes expert counts, then marks individual payload blocks arrived so tensor-core work can begin while later blocks are still being pulled. This plan ports that second property to the gfx1250 push path.

## File Structure

| File | Responsibility |
|---|---|
| `aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py` | Gate parsing, overlap arena regions, pointer accessors, graph-safe reset. |
| `aiter/ops/flydsl/kernels/push_group_overlap_stage1_gfx1250.py` | New single-kernel count/plan/payload/consumer state machine. |
| `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py` | Extract the existing contiguous GEMM1 tile body into a reusable FlyDSL builder. |
| `aiter/ops/flydsl/batched_gemm_mxfp4.py` | Expose the reusable TDM GEMM1 builder/ABI without changing the default caller. |
| `aiter/ops/flydsl/grouped_moe_gfx1250.py` | Select the overlap stage-1 path, allocate GEMM1 outputs, and retain GEMM2/combine. |
| `op_tests/flydsl_tests/test_push_group_overlap_plan.py` | CPU-only arena/layout and launch-configuration tests. |
| `op_tests/flydsl_tests/test_push_group_overlap_stage1.py` | Single-GPU synchronization, arrival, and GEMM1 parity tests. |
| `op_tests/multigpu_tests/test_mega_moe.py` | 2/4-rank correctness and profile-table A/B validation. |

---

### Task 1: Add overlap configuration and symmetric arena state

**Files:**
- Modify: `aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py`
- Test: `op_tests/flydsl_tests/test_push_group_overlap_plan.py`

**Interfaces:**
- Produces `EpDispatchCombineConfig.push_group_overlap: bool`.
- Produces `EpDispatchCombineConfig.push_group_overlap_dispatch_ctas: int`.
- Produces `EpDispatchCombineOp.push_group_overlap_ptrs() -> dict[str, int]`.
- Produces `EpDispatchCombineOp.reset_push_group_overlap() -> None`.

- [ ] **Step 1: Write the failing region-layout tests**

```python
def test_overlap_regions_are_absent_when_disabled(monkeypatch):
    monkeypatch.setenv("AITER_EP_PUSH_GROUP_OVERLAP", "0")
    assert _push_group_overlap_regions(_cfg(), cap=128, tile_m=64) == []


def test_overlap_regions_cover_counts_and_tiles(monkeypatch):
    monkeypatch.setenv("AITER_EP_PUSH_GROUP_OVERLAP", "1")
    regions = dict(_push_group_overlap_regions(_cfg(world_size=4, epr=8), cap=128, tile_m=64))
    assert regions["pg_overlap_rank_count"] == 4 * 8 * 4
    assert regions["pg_overlap_rank_base"] == 4 * 8 * 4
    assert regions["pg_overlap_tile_arrived"] == 8 * 2 * 4
    assert regions["pg_overlap_tile_expected"] == 8 * 2 * 4
    assert regions["pg_overlap_epoch"] == 4
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_push_group_overlap_plan.py -q
```

Expected: `ImportError` for `_push_group_overlap_regions`.

- [ ] **Step 3: Add config, regions, accessors, and reset**

Use these names and shapes:

```python
@property
def push_group_overlap(self) -> bool:
    return os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP", "0").lower() in _TRUTHY_ENV


def _push_group_overlap_regions(cfg, *, cap: int, tile_m: int):
    if not cfg.push_group_overlap:
        return []
    tiles_per_expert = cap // tile_m
    return [
        ("pg_overlap_rank_count", cfg.world_size * cfg.num_experts_per_rank * 4),
        ("pg_overlap_rank_base", cfg.world_size * cfg.num_experts_per_rank * 4),
        ("pg_overlap_tile_arrived", cfg.num_experts_per_rank * tiles_per_expert * 4),
        ("pg_overlap_tile_expected", cfg.num_experts_per_rank * tiles_per_expert * 4),
        ("pg_overlap_expert_done", cfg.num_experts_per_rank * 4),
        ("pg_overlap_work_head", 4),
        ("pg_overlap_epoch", 4),
    ]
```

`reset_push_group_overlap()` must zero all mutable counters before the stage-1 launch. `pg_overlap_tile_expected` is repopulated by the planner every epoch, so reset it too.

- [ ] **Step 4: Verify the tests pass**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /app/aiter
git add aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py \
  op_tests/flydsl_tests/test_push_group_overlap_plan.py
git commit -m "feat(ep): add push-group overlap workspace"
```

---

### Task 2: Extract the contiguous GEMM1 tile body for mega-kernel reuse

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
- Modify: `aiter/ops/flydsl/batched_gemm_mxfp4.py`
- Test: `op_tests/flydsl_tests/test_push_group_gemm1_parity.py`

**Interfaces:**
- Produces `build_gemm_a8w4_tdm_tile_runner(...) -> Callable[[fx.Int32, fx.Int32, fx.Int32], None]`.
- The returned callable consumes `(row_base, expert_id, valid_rows)` and performs one existing contiguous TDM GEMM1 tile.
- Existing `launch_gemm_a8w4_tdm()` keeps its current public ABI and calls the same builder.

- [ ] **Step 1: Extend parity coverage before refactoring**

Add a direct fixed-slot test that runs two valid tiles with different `row_base`, `expert_id`, and a partial `valid_rows`, then compares the result to the current static wrapper with `atol=0, rtol=0`.

- [ ] **Step 2: Verify the new test passes on the pre-refactor implementation**

Run:

```bash
cd /tmp && HIP_VISIBLE_DEVICES=0 python -m pytest \
  /app/aiter/op_tests/flydsl_tests/test_push_group_gemm1_parity.py -q -x
```

Expected: PASS. This establishes the refactor baseline.

- [ ] **Step 3: Move only the existing tile body**

Extract the code currently in `_run_tile()` after its swizzle calculation into a builder that receives the fixed-slot row base, expert id, valid row count, and N-tile id. Keep:

```python
WAVE_SPEC = num_waves >= 4 and tile_m >= 64 and tile_n >= 64
```

unchanged. The overlap path must use the contiguous `add_tdm_loads()` A-load path, never `tensor_load_gather`.

- [ ] **Step 4: Re-run parity**

Run the command from Step 2. Expected: PASS with byte-identical output.

- [ ] **Step 5: Commit**

```bash
cd /app/aiter
git add aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py \
  aiter/ops/flydsl/batched_gemm_mxfp4.py \
  op_tests/flydsl_tests/test_push_group_gemm1_parity.py
git commit -m "refactor(ep): expose contiguous push GEMM1 tile runner"
```

---

### Task 3: Implement count-and-prefix planning inside the mega stage-1 kernel

**Files:**
- Create: `aiter/ops/flydsl/kernels/push_group_overlap_stage1_gfx1250.py`
- Test: `op_tests/flydsl_tests/test_push_group_overlap_stage1.py`

**Interfaces:**
- Produces `launch_push_group_overlap_stage1(..., overlap_ptrs: dict[str, int], dispatch_ctas: int, consumer_ctas: int)`.
- The planner publishes `rank_base[src_rank, local_expert]`, `tile_expected[local_expert, tile]`, and a release-store epoch.
- Producers must not write payload until the planner epoch is observed with an acquire load.

- [ ] **Step 1: Write a count-plan test**

For `world_size=2`, `E_local=2`, `tile_m=4`, use source counts:

```python
rank_count = [[3, 5], [2, 1]]
```

Assert planner output:

```python
rank_base = [[0, 0], [3, 5]]
tile_expected = [[4, 1], [4, 2]]
```

The expected rows are per destination-local expert; a final partial tile is legal.

- [ ] **Step 2: Verify the test fails**

Run:

```bash
cd /tmp && HIP_VISIBLE_DEVICES=0 python -m pytest \
  /app/aiter/op_tests/flydsl_tests/test_push_group_overlap_stage1.py -q -x
```

Expected: import failure for `launch_push_group_overlap_stage1`.

- [ ] **Step 3: Implement Phase A — count then plan**

Each dispatch CTA counts its assigned `(token, topk)` routes by destination-local expert. It publishes only counts first:

```text
source rank → destination `pg_overlap_rank_count[source_rank, local_expert]`
```

The planner waits until every source count is published, computes:

```text
rank_base[source, expert] = sum(rank_count[0:source, expert])
expected_rows[expert, tile] =
    min(tile_m, total_count[expert] - tile * tile_m)
```

and release-publishes the epoch. Count arrays must be reset only after all consumers exit the epoch.

- [ ] **Step 4: Verify the plan test passes**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /app/aiter
git add aiter/ops/flydsl/kernels/push_group_overlap_stage1_gfx1250.py \
  op_tests/flydsl_tests/test_push_group_overlap_stage1.py
git commit -m "feat(ep): plan per-source push spans for overlap"
```

---

### Task 4: Publish payload arrival by tile fragment

**Files:**
- Modify: `aiter/ops/flydsl/kernels/push_group_overlap_stage1_gfx1250.py`
- Test: `op_tests/flydsl_tests/test_push_group_overlap_stage1.py`

**Interfaces:**
- Producers write contiguous rows beginning at `rank_base[source_rank, local_expert]`.
- Producers add one contiguous fragment length to `pg_overlap_tile_arrived[expert, tile]` only after payload, scale, route weight, and rowmap stores finish.
- Consumers use `spin_until_eq_i32(tile_arrived_addr, expected_rows)`.

- [ ] **Step 1: Add the visibility test**

Construct two producer fragments for one tile: first writes rows `[0, 2)`, second writes `[2, 4)`. Start the consumer before the second producer. Assert it cannot observe/compute from the tile after the first fragment and observes all four expected payload rows after the second fragment.

- [ ] **Step 2: Verify it fails**

Run the command from Task 3, Step 2. Expected: the arrival counter is absent or the consumer reads incomplete data.

- [ ] **Step 3: Implement Phase B — contiguous payload push**

For each source/expert span, split at tile boundaries. For every fragment:

```python
P.fence_system_release()
P.atomic_add_global(tile_arrived_addr, fragment_rows)
```

The source must write FP8 payload, E8M0 scale, weight, and source map before this fence. Do not use `pg_running` for row allocation in overlap mode; the planned `rank_base` determines the unique contiguous destination range.

- [ ] **Step 4: Add the consumer condition**

Before the reused GEMM tile runner reads A:

```python
P.spin_until_eq_i32(tile_arrived_addr, expected_rows)
P.fence_system_acquire()
```

For the planner-created tail tile, `expected_rows < tile_m`; pass that value as `valid_rows` so existing out-of-bounds masking remains active.

- [ ] **Step 5: Verify the synchronization and parity tests pass**

Run the command from Task 3, Step 2. Expected: PASS with no timeout and byte-identical GEMM1 output.

- [ ] **Step 6: Commit**

```bash
cd /app/aiter
git add aiter/ops/flydsl/kernels/push_group_overlap_stage1_gfx1250.py \
  op_tests/flydsl_tests/test_push_group_overlap_stage1.py
git commit -m "feat(ep): publish completed push tiles to GEMM1 workers"
```

---

### Task 5: Add CTA roles and an online persistent work scheduler

**Files:**
- Modify: `aiter/ops/flydsl/kernels/push_group_overlap_stage1_gfx1250.py`
- Modify: `aiter/ops/flydsl/grouped_moe_gfx1250.py`
- Test: `op_tests/flydsl_tests/test_push_group_overlap_stage1.py`

**Interfaces:**
- CTA IDs `[0, dispatch_ctas)` are producers.
- CTA ID `dispatch_ctas` is the planner.
- CTA IDs `[dispatch_ctas + 1, dispatch_ctas + 1 + consumer_ctas)` are GEMM1 consumers.
- `AITER_EP_PUSH_GROUP_OVERLAP_DISPATCH_CTAS` defaults to `max(1, num_cu // 8)`.

- [ ] **Step 1: Add launch-configuration validation tests**

```python
with pytest.raises(ValueError, match="dispatch CTAs"):
    _validate_overlap_ctas(num_cu=8, dispatch_ctas=0)
with pytest.raises(ValueError, match="consumer CTA"):
    _validate_overlap_ctas(num_cu=8, dispatch_ctas=7)
assert _validate_overlap_ctas(num_cu=8, dispatch_ctas=2) == (2, 1, 5)
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_push_group_overlap_plan.py -q
```

Expected: import failure for `_validate_overlap_ctas`.

- [ ] **Step 3: Implement the fixed-grid state machine**

Launch exactly:

```text
grid.x = dispatch_ctas + 1 + consumer_ctas
```

Consumers acquire work with a local atomic work head, map work to `(expert, tile_idx, n_tile)`, wait for the tile arrival count, and invoke the extracted contiguous tile runner. The planner may join the consumer loop only after it publishes the plan; producer CTAs may join only after completing all payload stores and the source-done epoch.

`grouped_a8w4_tdm_moe_push_scatter()` selects this path only when all are true:

```python
cfg.push_group and cfg.push_group_overlap and _fuse_quant
```

Otherwise it retains the current `launch_push_group_finalize()` path.

- [ ] **Step 4: Verify unit tests pass**

Run both:

```bash
cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_push_group_overlap_plan.py -q
cd /tmp && HIP_VISIBLE_DEVICES=0 python -m pytest \
  /app/aiter/op_tests/flydsl_tests/test_push_group_overlap_stage1.py -q -x
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /app/aiter
git add aiter/ops/flydsl/kernels/push_group_overlap_stage1_gfx1250.py \
  aiter/ops/flydsl/grouped_moe_gfx1250.py \
  op_tests/flydsl_tests/test_push_group_overlap_plan.py \
  op_tests/flydsl_tests/test_push_group_overlap_stage1.py
git commit -m "feat(ep): overlap push dispatch with persistent GEMM1"
```

---

### Task 6: End-to-end correctness, graph replay, and overlap measurement

**Files:**
- Modify: `op_tests/multigpu_tests/test_mega_moe.py` only if it lacks a profile range that captures the new mega stage-1 kernel.
- Test: `op_tests/multigpu_tests/test_mega_moe.py`

- [ ] **Step 1: Verify overlap-off regression**

```bash
cd /tmp && AITER_EP_PUSH_GROUP=1 AITER_EP_PUSH_GROUP_OVERLAP=0 \
ENABLE_CK=0 AITER_FORCE_A8W4=1 AITER_USE_GROUPED_GEMM=1 \
torchrun --standalone --nproc_per_node=2 \
  /app/aiter/op_tests/multigpu_tests/test_mega_moe.py \
  -q a8w4_mxfp4 -bs 64 -hd 7168 -id 3072 -e 384 -k 6 \
  --layers 2 --combine scatter_fused --acc_verify 1
```

Expected: `MEGA-CHECK PASS`.

- [ ] **Step 2: Verify overlap-on correctness**

```bash
cd /tmp && AITER_EP_PUSH_GROUP=1 AITER_EP_PUSH_GROUP_OVERLAP=1 \
AITER_EP_PUSH_GROUP_OVERLAP_DISPATCH_CTAS=16 \
ENABLE_CK=0 AITER_FORCE_A8W4=1 AITER_USE_GROUPED_GEMM=1 \
torchrun --standalone --nproc_per_node=2 \
  /app/aiter/op_tests/multigpu_tests/test_mega_moe.py \
  -q a8w4_mxfp4 -bs 64 -hd 7168 -id 3072 -e 384 -k 6 \
  --layers 2 --combine scatter_fused --acc_verify 1
```

Expected: `MEGA-CHECK PASS`. Repeat at `bs=8, 128, 512, 2048`, then repeat the `bs=64` case with `--nproc_per_node=4`.

- [ ] **Step 3: Capture performance evidence**

Run overlap on/off with the same fixed shape and:

```bash
--profile_table 1 --layers 61
```

Accept the feature only if the trace shows the stage-1 mega kernel has both active dispatch producers and GEMM1 consumers before all payload fragments are published, and the median per-layer device time is no worse than `AITER_EP_PUSH_GROUP_OVERLAP=0`.

- [ ] **Step 4: Record results and commit**

Append the measured 2/4-rank correctness and on/off median table to this plan, including `dispatch_ctas`, `consumer_ctas`, batch size, and whether the trace shows overlapped intervals.

```bash
cd /app/aiter
git add docs/superpowers/plans/2026-08-06-persistent-push-group-overlap-gfx1250.md \
  op_tests/multigpu_tests/test_mega_moe.py
git commit -m "test(ep): validate persistent push overlap"
```

## Fallback

If the count-and-plan phase, per-tile arrival atomics, or reduced consumer occupancy erase the overlap benefit, retain the implementation behind `AITER_EP_PUSH_GROUP_OVERLAP=0` by default. The already-shipped Q1 fixed-slot dispatch plus current persistent GEMM1 remains the performance baseline. Do not revert to the same-stream producer/consumer ring design because it cannot overlap kernels.

## Self-Review

- **Coverage:** Task 1 creates graph-safe storage; Task 2 preserves the optimized contiguous GEMM body; Task 3 supplies metadata before payload; Task 4 establishes visibility and per-tile readiness; Task 5 creates the persistent producer/consumer CTA schedule; Task 6 verifies correctness and measurable overlap.
- **Scope:** GEMM2/combine, CTA-internal named barriers, and an all-layer DeepGEMM-equivalent kernel are excluded.
- **Consistency:** All later tasks use `pg_overlap_rank_count`, `pg_overlap_rank_base`, `pg_overlap_tile_arrived`, `pg_overlap_tile_expected`, and `AITER_EP_PUSH_GROUP_OVERLAP` exactly as introduced in Task 1.

# Push-group Persistent GEMM1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make push-group GEMM1 execute exactly the device-reported valid M tiles through a CUDA-Graph-safe persistent grid-stride scheduler.

**Architecture:** Reuse finalize's tile-aligned `num_valid` device scalar. A fixed number of GEMM1 worker CTAs reads that scalar and grid-strides over the swizzled `(m_tile, n_tile)` work stream, while still addressing fixed-slot A/output rows through existing compact `tile_row_base`, `expert_ids`, and `tile_valid` metadata. GEMM2 remains on the current static compact grid.

**Tech Stack:** Python, PyTorch CUDA Graph, FlyDSL, MLIR `scf.ForOp`, ROCm gfx1250 TDM WMMA kernels, pytest, torchrun.

## Global Constraints

- Persistent scheduling applies only to push-group GEMM1; GEMM2 P2P scatter remains unchanged.
- No host `.item()`, D2H synchronization, indirect launch, global work-queue atomic, or megakernel.
- `num_valid` is a `(1,) int32` device scalar and is tile_m aligned.
- Default push GEMM1 policy is hybrid: enable persistent only when `AITER_EP_PUSH_GROUP_PERSISTENT_GEMM1!=0` and static `grid_m < cm`; `AITER_EP_PUSH_GROUP_PERSISTENT_GEMM1=0` restores the static compact scheduler.
- All routes run under CUDA Graph capture and must remain graph-safe.
- Preserve pull behavior and fixed-slot `[E_local * CAP]` output allocation/layout.
- Validate with 2-rank DeepSeek shape `hd=7168,id=3072,E=384,k=6`, bs `8/64/128/256/512/2048`.

---

### Task 1: Add a persistent GEMM1 launch ABI and kernel scheduler

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py:51-109,187-245,888-902`
- Modify: `aiter/ops/flydsl/batched_gemm_mxfp4.py:27-142`
- Test: `op_tests/flydsl_tests/test_push_group_gemm1_parity.py`

**Interfaces:**
- Produces `launch_gemm_a8w4_tdm(..., ep_persistent_gemm1: Constexpr[int], persistent_workers: Constexpr[int], arg_num_valid_rows: fx.Pointer)`.
- Produces `flydsl_grouped_gemm_a8w4_masked(..., ep_persistent_gemm1=False, num_valid_rows=None, persistent_workers=None)`.
- `num_valid_rows` is an on-device `(1,) torch.int32`; its value is a multiple of `tile_m`.

- [ ] **Step 1: Add a parity test for persistent GEMM1**

Extend the existing fixed-slot test to invoke GEMM1 twice from identical `a1_payload/a1_scale/trb/eids/tvd`:

```python
y_static = torch.zeros_like(y_push)
flydsl_grouped_gemm_a8w4_masked(
    y_static, a1_payload, w1_u8, a1_scale, w1s_i32, psum_dummy,
    n_experts=E, contiguous_m=grid_m, N=two_inter, K=model_dim,
    tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    push_group=1, tile_row_base=trb, expert_ids=eids, tile_valid=tvd,
)

y_persistent = torch.zeros_like(y_push)
flydsl_grouped_gemm_a8w4_masked(
    y_persistent, a1_payload, w1_u8, a1_scale, w1s_i32, psum_dummy,
    n_experts=E, contiguous_m=grid_m, N=two_inter, K=model_dim,
    tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    push_group=1, tile_row_base=trb, expert_ids=eids, tile_valid=tvd,
    ep_persistent_gemm1=1, num_valid_rows=num_valid, persistent_workers=256,
)
torch.cuda.synchronize()
assert torch.equal(y_static, y_persistent)
```

Also add an all-empty metadata case (`num_valid.zero_()`, every `expert_ids=E`, `tile_valid=0`) and assert persistent output remains zero.

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
cd /tmp && HIP_VISIBLE_DEVICES=0 python -m pytest \
  /app/aiter/op_tests/flydsl_tests/test_push_group_gemm1_parity.py -q -x
```

Expected: failure because `ep_persistent_gemm1` is not yet an accepted wrapper argument.

- [ ] **Step 3: Extend the host wrapper ABI**

In `flydsl_grouped_gemm_a8w4_masked`, add keyword-only arguments and a valid JIT dummy pointer:

```python
    ep_persistent_gemm1=0,
    num_valid_rows=None,
    persistent_workers=None,
):
    if persistent_workers is None:
        persistent_workers = 1
    num_valid_rows_tensor = (
        num_valid_rows if num_valid_rows is not None else out
    )
```

Pass the following to `launch_gemm_a8w4_tdm`:

```python
        ep_persistent_gemm1=int(ep_persistent_gemm1),
        persistent_workers=int(persistent_workers),
        arg_num_valid_rows=ptr_arg(num_valid_rows_tensor),
```

The dummy is legal because the kernel reads it only under `const_expr(ep_persistent_gemm1)`.

- [ ] **Step 4: Refactor TDM tile execution**

In `launch_gemm_a8w4_tdm`:

1. Import `gpu` from `flydsl.expr`.
2. Add constexpr/runtime parameters:

```python
    ep_persistent_gemm1: Constexpr[int] = 0,
    persistent_workers: Constexpr[int] = 1,
    arg_num_valid_rows: fx.Pointer = None,
```

3. Add persistent flag and worker count to `cache_tag`.
4. Move all code beginning at the current swizzle calculation through the epilogue into:

```python
        def _run_tile(work_id, total_m_tiles):
            # Existing swizzle and body.
            # Replace bid_x with work_id.
            # Replace `(i32_m + tile_m - 1) // tile_m` with total_m_tiles.
```

5. Preserve the existing fixed launch semantics:

```python
        if const_expr(not ep_persistent_gemm1):
            _run_tile(bid_x, (i32_m + (tile_m - 1)) // tile_m)
```

6. Add a device-side persistent branch using `scf.ForOp`; do not use Python `range` with computed FlyDSL values:

```python
        else:
            i32_ptr = fx.PointerType.get(
                elem_ty=fx.Int32.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=4,
            )
            valid_ptr = fx.recast_iter(i32_ptr, arg_num_valid_rows)
            valid_rows = valid_ptr[0]
            valid_m_tiles = valid_rows // tile_m
            total_work = valid_m_tiles * total_n_tiles

            start = arith.index_cast(T.index, bid_x)
            stop = arith.index_cast(T.index, total_work)
            step = arith.index(persistent_workers)
            loop = scf.ForOp(start, stop, step)
            with ir.InsertionPoint(loop.body):
                work_id = arith.index_cast(T.i32(), loop.induction_variable)
                workgroup_barrier()
                _run_tile(work_id, valid_m_tiles)
                scf.YieldOp([])
```

`total_work=0` creates an empty loop, so metadata is never read.

7. Choose the launch grid by constexpr branch:

```python
    if const_expr(ep_persistent_gemm1):
        launch_grid = (persistent_workers, 1, 1)
    else:
        launch_grid = (m_tiles * n_tiles, 1, 1)
```

Use `launch_grid` in the kernel launch.

- [ ] **Step 5: Run parity and lint**

Run:

```bash
cd /tmp && HIP_VISIBLE_DEVICES=0 python -m pytest \
  /app/aiter/op_tests/flydsl_tests/test_push_group_gemm1_parity.py -q -x
```

Expected: persistent and static outputs are byte-identical; all-empty case is zero.

Then:

```bash
python -m ruff check \
  /app/aiter/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py \
  /app/aiter/aiter/ops/flydsl/batched_gemm_mxfp4.py \
  /app/aiter/op_tests/flydsl_tests/test_push_group_gemm1_parity.py
```

- [ ] **Step 6: Inspect ISA before proceeding**

Compile both modes with the existing ISA dump procedure. Record final ISA VGPR count, SGPR count, VGPR spill count, and SGPR spill count for GEMM1. Do not enable persistent by default if compilation fails or the persistent binary crosses an occupancy boundary.

- [ ] **Step 7: Commit the isolated scheduler**

```bash
git add aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py \
        aiter/ops/flydsl/batched_gemm_mxfp4.py \
        op_tests/flydsl_tests/test_push_group_gemm1_parity.py
git commit -m "perf(ep): add persistent push GEMM1 scheduler"
```

### Task 2: Wire persistent GEMM1 into push-group MoE

**Files:**
- Modify: `aiter/ops/flydsl/grouped_moe_gfx1250.py:673-850`
- Modify: `op_tests/flydsl_tests/test_push_group_finalize.py`
- Test: `op_tests/multigpu_tests/test_mega_moe.py`

**Interfaces:**
- Consumes `num_valid` emitted by `launch_push_group_finalize`.
- Consumes Task 1 wrapper keywords.
- Produces default-on GEMM1 persistent launch with `AITER_EP_PUSH_GROUP_PERSISTENT_GEMM1=0` fallback.

- [ ] **Step 1: Add finalize edge-case coverage**

In `test_push_group_finalize.py`, add parametrized count vectors:

```python
@pytest.mark.parametrize("running_list", [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 1],
    [256, 256, 256, 256],
])
def test_finalize_compact_metadata_edge_cases(running_list):
    ...
```

For each vector, verify:

```python
assert int(num_valid.item()) == sum(
    ((min(c, cap) + tile_m - 1) // tile_m) * tile_m
    for c in running_list
)
```

and verify the compact metadata prefix contains exactly the expected `(row_base, expert, valid_rows)` triples.

- [ ] **Step 2: Run the finalize tests**

Run:

```bash
cd /tmp && HIP_VISIBLE_DEVICES=0 python -m pytest \
  /app/aiter/op_tests/flydsl_tests/test_push_group_finalize.py -q -x
```

Expected: PASS before integration, establishing that `num_valid` is tile-aligned and complete.

- [ ] **Step 3: Add default-on policy in push helper**

At the top of `grouped_a8w4_tdm_moe_push_scatter`, import `get_cu_num` locally:

```python
from aiter.jit.utils.chip_info import get_cu_num
```

After `grid_m` is computed and before GEMM1 launch, add:

```python
    persistent_gemm1_requested = os.environ.get(
        "AITER_EP_PUSH_GROUP_PERSISTENT_GEMM1", "1"
    ).lower() in ("1", "true", "yes", "on")
    persistent_gemm1 = persistent_gemm1_requested and grid_m < cm
    workers_env = os.environ.get("AITER_EP_PUSH_GROUP_PERSISTENT_WORKERS")
    persistent_workers = (
        int(workers_env) if workers_env else int(get_cu_num())
    )
    if persistent_workers < 1:
        raise ValueError(
            "AITER_EP_PUSH_GROUP_PERSISTENT_WORKERS must be >= 1"
        )
```

Pass these only to both GEMM1 call sites:

```python
            ep_persistent_gemm1=int(persistent_gemm1),
            num_valid_rows=num_valid,
            persistent_workers=persistent_workers,
```

Do not add those arguments to the GEMM2 call.

- [ ] **Step 4: Run push-off and static fallback regressions**

Run:

```bash
cd /app/aiter
AITER_EP_PUSH_GROUP=0 timeout 900 torchrun --standalone --nproc_per_node=2 \
  op_tests/multigpu_tests/test_mega_moe.py -q a8w4_mxfp4 \
  -bs 8 -hd 7168 -id 3072 -e 384 -k 6 --layers 8 \
  --combine scatter_fused --acc_verify 1

AITER_EP_PUSH_GROUP=1 AITER_EP_PUSH_GROUP_PERSISTENT_GEMM1=0 \
  timeout 900 torchrun --standalone --nproc_per_node=2 \
  op_tests/multigpu_tests/test_mega_moe.py -q a8w4_mxfp4 \
  -bs 8 -hd 7168 -id 3072 -e 384 -k 6 --layers 8 \
  --combine scatter_fused --acc_verify 1
```

Expected: both print `MEGA-CHECK ... PASS`; the latter is the pre-persistent static compact behavior.

- [ ] **Step 5: Run default persistent graph e2e**

Run:

```bash
cd /app/aiter
AITER_EP_PUSH_GROUP=1 timeout 900 torchrun --standalone --nproc_per_node=2 \
  op_tests/multigpu_tests/test_mega_moe.py -q a8w4_mxfp4 \
  -bs 8 -hd 7168 -id 3072 -e 384 -k 6 --layers 8 \
  --combine scatter_fused --acc_verify 1 --profile_table 1
```

Expected: graph capture succeeds, `MEGA-CHECK ... PASS`, and GEMM1 is no longer reported with the static bound-tile grid behavior.

- [ ] **Step 6: Commit integration**

```bash
git add aiter/ops/flydsl/grouped_moe_gfx1250.py \
        op_tests/flydsl_tests/test_push_group_finalize.py
git commit -m "perf(ep): enable persistent scheduler for push GEMM1"
```

### Task 3: Establish persistent performance policy

**Files:**
- Modify: `op_tests/multigpu_tests/test_mega_moe.py` only if a repeat/median CLI is absent
- Test: `op_tests/multigpu_tests/test_mega_moe.py`

**Interfaces:**
- Consumes persistent default-on and static fallback environment variables.
- Produces a reproducible graph-replay p50 comparison before retaining default-on policy.

- [ ] **Step 1: Add a graph replay sample-count CLI only if needed**

If `test_mega_moe.py` has no repeat mechanism, add:

```python
p.add_argument(
    "--bench_samples", type=int, default=11,
    help="independent CUDA-graph replay timing samples; report median",
)
```

Replace the single `time.perf_counter()` sample with:

```python
samples_us = []
for _ in range(args.bench_samples):
    torch.cuda.synchronize()
    self.comm.barrier()
    t0 = time.perf_counter()
    self.graph.replay()
    torch.cuda.synchronize()
    samples_us.append((time.perf_counter() - t0) * 1e6)
total_us = statistics.median(samples_us)
```

Keep profiler collection after timing and outside this median loop.

- [ ] **Step 2: Verify graph benchmark still checks correctness**

Run:

```bash
cd /app/aiter
AITER_EP_PUSH_GROUP=1 timeout 900 torchrun --standalone --nproc_per_node=2 \
  op_tests/multigpu_tests/test_mega_moe.py -q a8w4_mxfp4 \
  -bs 8 -hd 7168 -id 3072 -e 384 -k 6 --layers 8 \
  --combine scatter_fused --acc_verify 1 --bench_samples 11
```

Expected: output reports median graph-replay per-layer latency and `MEGA-CHECK ... PASS`.

- [ ] **Step 3: Run paired static/persistent/pull sweep**

For every `BS in 8 64 128 256 512 2048`, run:

```bash
BASE="timeout 900 torchrun --standalone --nproc_per_node=2 \
  op_tests/multigpu_tests/test_mega_moe.py -q a8w4_mxfp4 \
  -bs $BS -hd 7168 -id 3072 -e 384 -k 6 --layers 8 \
  --combine scatter_fused --acc_verify 1 --bench_samples 11"

AITER_EP_PUSH_GROUP=0 $BASE
AITER_EP_PUSH_GROUP=1 AITER_EP_PUSH_GROUP_PERSISTENT_GEMM1=0 $BASE
AITER_EP_PUSH_GROUP=1 AITER_EP_PUSH_GROUP_PERSISTENT_GEMM1=1 $BASE
```

Record p50 per-layer latency, static→persistent GEMM1 delta, and every `MEGA-CHECK` result.

- [ ] **Step 4: Apply the policy gate**

Keep persistent default-on only when:

```text
bs=8 persistent p50 <= static-push p50
and
for every bs >= 64:
    persistent p50 <= static-push p50 * 1.02
```

If the second condition fails, change the default environment fallback to `0` and add an explicit small-batch enable condition based on `max_land < E_local * CAP`; do not use `num_valid.item()` or any host-visible dynamic decision.

- [ ] **Step 5: Commit benchmark policy and documentation**

```bash
git add op_tests/multigpu_tests/test_mega_moe.py \
        docs/superpowers/specs/2026-08-06-push-group-persistent-gemm1-gfx1250-design.md \
        docs/superpowers/plans/2026-08-06-push-group-persistent-gemm1-gfx1250.md
git commit -m "test(ep): measure persistent push GEMM1 graph latency"
```

## Plan self-review

- Spec coverage: Tasks 1–3 cover ABI/scheduler, push-only integration, parity, edge cases, ISA, graph e2e, p50 sweep, and fallback policy.
- Placeholder scan: no TBD/TODO markers or unnamed interfaces.
- Type consistency: `num_valid_rows` is consistently an on-device `(1,) int32`; persistent control names are consistent across wrapper, kernel, helper, and environment.

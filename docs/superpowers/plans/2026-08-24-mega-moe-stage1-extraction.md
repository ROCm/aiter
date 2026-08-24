# Standalone MegaMoE Stage1 Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, CUDA-Graph-safe `MegaMoEStage1` operator that preserves current compact EP dispatch semantics and emits independently owned tensors matching the ordinary FP8 FlyDSL v2 FMoE GEMM2 ABI.

**Architecture:** Add a Stage1-only Python owner and a dedicated compact fused FlyDSL kernel. Reuse the existing quantizer, group-major SHMEM owner, compact dispatch primitives, GEMM1 builder, and SwiGLU/FP8 epilogue; rewrite only the top-level scheduler glue and direct FMoE metadata publication. Keep `MegaMoEV2` on its existing implementation path.

**Tech Stack:** Python, PyTorch/ROCm, FlyDSL, Mori SHMEM, pytest, torchrun, CUDA Graphs.

**Approved design:** `docs/superpowers/specs/2026-08-24-mega-moe-stage1-extraction-design.md`

---

## Implementation constraints

- Read and follow `AGENTS.md` plus the repository-required skills before implementation. In particular, use `karpathy-guidelines`, `superpowers:test-driven-development`, `aiter-op-test`, and `flydsl-kernel-authoring` when their trigger conditions apply.
- Do not cherry-pick, copy, or use implementation code from `dev/tp_fuse_gemm1_v0`.
- Do not change `MegaMoEV2` to call the new operator.
- Treat existing public implementations and kernel behavior as compatibility-frozen:
  - `aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py`
  - `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py`
  - `aiter/ops/flydsl/kernels/mega_moe/dispatch.py`
  - `aiter/ops/flydsl/kernels/mega_moe/gemm1.py`
  - `aiter/ops/flydsl/kernels/mega_moe/gemm_util.py`
  - `aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py`
  - `aiter/fused_moe.py`
  - `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py`
- Reuse existing callable building blocks by importing and calling them. Purely
  additive definitions are allowed in existing modules or classes when needed:
  new methods, new helper functions, new classes, and new export entries.
- Do not edit any pre-existing callable body, signature, default, branch, data
  layout, synchronization rule, or runtime behavior.
- Reimplement the compact top-level scheduler in the new kernel file because
  sharing it would require rewriting the existing kernel implementation.
- In `mega_moe/__init__.py`, add the two new lazy-export mappings without
  changing any existing mapping or the `__getattr__`/`__dir__` implementation.
- Do not add fixed-slot support.
- Do not add a post-Stage1 metadata conversion kernel.
- The prequantized entry must launch exactly one fused Stage1 kernel after its inputs are available.
- Do not add performance thresholds or tuning work.
- Preserve unrelated dirty-worktree files.

## File map

- Create `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py`
  - Public result dataclass, capacity math, validation, Stage1-only resources, BF16/prequant entry points.
- Create `aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py`
  - Dedicated compact EP Stage1 compiler and launcher.
- Modify `aiter/ops/flydsl/kernels/mega_moe/__init__.py`
  - Add only lazy exports for `MegaMoEStage1` and `MegaMoEStage1Output`.
- Create `op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py`
  - CPU/import-level contract and capacity tests.
- Create `op_tests/multigpu_tests/test_mega_moe_stage1.py`
  - EP8 semantic correctness, lifetime, CUDA Graph, launch-count, and downstream ABI smoke tests.
- Read-only regression target: `op_tests/multigpu_tests/test_mega_moe_v2.py`

## Reference implementation points

- Existing host ownership and dispatch-table construction:
  - `aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py:25-258`
- Existing fused Stage1 scheduling:
  - `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:71-502`
- Reusable compact dispatch primitives:
  - `aiter/ops/flydsl/kernels/mega_moe/dispatch.py:416-926`
- Reusable GEMM1 builder:
  - `aiter/ops/flydsl/kernels/mega_moe/gemm1.py`
- Reusable fused SwiGLU/FP8 epilogue:
  - `aiter/ops/flydsl/kernels/mega_moe/gemm_util.py:550`
- FMoE v2 result/consumer ABI:
  - `aiter/fused_moe.py:1409-1485`
  - `aiter/ops/flydsl/moe_kernels.py:1423-1881`
  - `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:578-715`
- Reference config:
  - `docs/fp8_retune_config/dsv4_fp8fp4_tp8_k384_flydslv2_tuned_20260726_144002.csv:3`

### Task 1: Define the result contract and capacity math

**Files:**
- Create: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py`
- Create: `op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py`

- [ ] **Step 1: Write failing tests for the FMoE v2 output layout**

Add tests that require a pure layout helper and the frozen result type:

```python
from dataclasses import FrozenInstanceError

import pytest
import torch

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1_op import (
    MegaMoEStage1Output,
    _stage1_output_layout,
)


def test_stage1_output_layout_matches_fmoe_capacity():
    layout = _stage1_output_layout(
        run_tokens=2,
        world_size=8,
        experts_per_rank=48,
        topk=6,
        inter_dim=3072,
        sort_block_m=32,
    )
    assert layout.logical_tokens == 16
    assert layout.route_capacity == 16 * 6 + 48 * 32 - 6
    assert layout.max_sorted == 1632
    assert layout.scale_rows == 1792
    assert layout.scale_cols == 96
    assert layout.expert_blocks == 51


def test_stage1_output_is_frozen():
    output = MegaMoEStage1Output(
        inter_sorted_quant=torch.empty(0),
        inter_sorted_shuffled_scale=torch.empty(0),
        sorted_token_ids=torch.empty(0, dtype=torch.int32),
        sorted_weights=torch.empty(0),
        sorted_expert_ids=torch.empty(0, dtype=torch.int32),
        num_valid_ids=torch.empty(2, dtype=torch.int32),
        logical_tokens=16,
        max_sorted=1632,
        num_local_experts=48,
        model_dim=7168,
        inter_dim=3072,
        topk=6,
        sort_block_m=32,
        _keepalive=(),
    )
    with pytest.raises(FrozenInstanceError):
        output.logical_tokens = 8
```

- [ ] **Step 2: Run the contract tests and verify the import fails**

Run:

```bash
pytest -q op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py
```

Expected: FAIL because `mega_moe_stage1_op` does not exist.

- [ ] **Step 3: Implement the immutable result and layout helper**

Create the module with these definitions:

```python
from dataclasses import dataclass, field

import torch


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


@dataclass(frozen=True, slots=True)
class _Stage1OutputLayout:
    logical_tokens: int
    route_capacity: int
    max_sorted: int
    scale_rows: int
    scale_cols: int
    expert_blocks: int


def _stage1_output_layout(
    *,
    run_tokens: int,
    world_size: int,
    experts_per_rank: int,
    topk: int,
    inter_dim: int,
    sort_block_m: int,
) -> _Stage1OutputLayout:
    logical_tokens = int(world_size) * int(run_tokens)
    route_capacity = (
        logical_tokens * int(topk)
        + int(experts_per_rank) * int(sort_block_m)
        - int(topk)
    )
    max_sorted = _round_up(route_capacity, sort_block_m)
    scale_rows = _round_up(max_sorted, 256)
    scale_cols = _round_up(inter_dim // 32, 8)
    return _Stage1OutputLayout(
        logical_tokens=logical_tokens,
        route_capacity=route_capacity,
        max_sorted=max_sorted,
        scale_rows=scale_rows,
        scale_cols=scale_cols,
        expert_blocks=max_sorted // int(sort_block_m),
    )


@dataclass(frozen=True, slots=True, eq=False)
class MegaMoEStage1Output:
    inter_sorted_quant: torch.Tensor
    inter_sorted_shuffled_scale: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    logical_tokens: int
    max_sorted: int
    num_local_experts: int
    model_dim: int
    inter_dim: int
    topk: int
    sort_block_m: int
    _keepalive: tuple[torch.Tensor, ...] = field(repr=False, compare=False)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
pytest -q op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py
```

Expected: PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py \
        op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py
git commit -m "feat(mega_moe): define standalone stage1 result contract"
```

### Task 2: Add Stage1-only resource ownership and validation

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py`
- Modify: `op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py`

- [ ] **Step 1: Add failing pure validation tests**

Add parameterized tests for constructor invariants without allocating a GPU object:

```python
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quant": "bf16"}, "quant='a8w4'"),
        ({"world_size": 9}, "world_size must be in"),
        ({"experts": 383}, "must be divisible"),
        ({"max_tok_per_rank": 300}, "power of two"),
        ({"topk": 257}, "topk must fit"),
        ({"swiglu_limit": -1.0}, "non-negative"),
    ],
)
def test_validate_constructor_rejects_invalid_values(kwargs, match):
    values = dict(
        rank=0,
        world_size=8,
        model_dim=7168,
        inter_dim=3072,
        experts=384,
        topk=6,
        quant="a8w4",
        max_tok_per_rank=256,
        swiglu_limit=0.0,
    )
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        _validate_constructor_args(**values)
```

- [ ] **Step 2: Run the validation tests and verify they fail**

Run:

```bash
pytest -q op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py
```

Expected: FAIL because `_validate_constructor_args` is missing.

- [ ] **Step 3: Implement explicit constructor validation**

Add `_validate_constructor_args()` and call it before any allocation. The checks must be explicit:

```python
def _validate_constructor_args(
    *, rank, world_size, model_dim, inter_dim, experts, topk,
    quant, max_tok_per_rank, swiglu_limit,
):
    if quant != "a8w4":
        raise ValueError("MegaMoEStage1 currently supports quant='a8w4' only")
    if not 1 <= world_size <= 8:
        raise ValueError(f"world_size must be in [1, 8], got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} must be in [0, {world_size})")
    if experts <= 0 or experts % world_size:
        raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
    if max_tok_per_rank <= 0 or max_tok_per_rank & (max_tok_per_rank - 1):
        raise ValueError("max_tok_per_rank must be a positive power of two")
    if world_size * max_tok_per_rank > 1 << 24:
        raise ValueError("source-token encoding exceeds 24 bits")
    if not 0 < topk <= 256:
        raise ValueError("topk must fit the packed high 8-bit field")
    if model_dim <= 0 or model_dim % 256:
        raise ValueError("model_dim must be positive and divisible by 256")
    if inter_dim <= 0 or inter_dim % 32:
        raise ValueError("inter_dim must be positive and divisible by 32")
    if swiglu_limit < 0:
        raise ValueError("swiglu_limit must be non-negative")
```

- [ ] **Step 4: Implement the Stage1-only owner**

Add `MegaMoEStage1.__init__` with the same keyword-only model/topology arguments as the approved API. It must:

```python
self.rank = int(rank)
self.world_size = int(world_size)
self.model_dim = int(model_dim)
self.inter_dim = int(inter_dim)
self.experts = int(experts)
self.experts_per_rank = self.experts // self.world_size
self.topk = int(topk)
self.max_tok_per_rank = int(max_tok_per_rank)
self.swiglu_limit = float(swiglu_limit)
self.device = torch.device("cuda", self.rank)
self.scale_dim = self.model_dim // 32
self._w1 = w1.contiguous().view(torch.uint8)
self._w1_scale = w1_scale.contiguous().view(torch.uint8)
self._epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.device)
self._epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.device)
self._num_cu = torch.cuda.get_device_properties(self.device).multi_processor_count
```

Construct only `FlyDSLDispatchGroupMajorOp`, not `FlyDSLDispatchCombineIntraNodeOp`:

```python
self._group = FlyDSLDispatchGroupMajorOp(
    rank=self.rank,
    world_size=self.world_size,
    hidden_dim=self.model_dim,
    max_tok_per_rank=self.max_tok_per_rank,
    experts_per_rank=self.experts_per_rank,
    topk=self.topk,
    data_type=torch.float8_e4m3fn,
    unit_size=128,
    scale_dim=self.scale_dim,
    scale_type_size=1,
    compact=True,
)
```

Validate W1 storage after converting to byte views:

```python
expected_w1_bytes = (
    self.experts_per_rank * 2 * self.inter_dim * self.model_dim // 2
)
expected_scale_bytes = (
    self.experts_per_rank * 2 * self.inter_dim * (self.model_dim // 32)
)
if self._w1.numel() != expected_w1_bytes:
    raise ValueError(f"W1 must contain {expected_w1_bytes} packed MXFP4 bytes")
if self._w1_scale.numel() != expected_scale_bytes:
    raise ValueError(f"W1 scale must contain {expected_scale_bytes} E8M0 bytes")
```

- [ ] **Step 5: Allocate the persistent compact dispatch workspace**

Create `_allocate_dispatch_workspace()` and `_build_dispatch_table()` in the new module. Copy the shapes and `DispatchSlot` mapping from `MegaMoEV2._allocate_dispatch_workspace` and `MegaMoEV2._build_v2_disp_table`, with these exact differences:

```text
- Keep all compact count/group/payload/epoch/work-pool slots.
- Keep persistent symmetric rx_em, scale_em, wts_em, and srcmap_em in self._group.
- Do not allocate _s1_out or _s1_osd here; those are per-call outputs.
- Do not allocate any W2, Stage2, combine-input, combine-output, or xdev-barrier resource.
- Keep the dispatch table instance-owned and stable across CUDA Graph replay.
```

Populate every `DispatchSlot` used by `emit_dispatch_plan`, `emit_dispatch_group`, `emit_dispatch_payload`, the epoch handshake, and the GEMM work pool. Use the existing mapping in `mega_moe_v2.py:148-199` verbatim for those slots so the shared primitives see the same ABI.

- [ ] **Step 6: Add compact-only config selection**

```python
def _select_stage1_config(self, run_tokens: int) -> Stage1Config:
    config_mtpr = max(self.max_tok_per_rank, FIXED_SLOT_MAX_MTPR + 1)
    return select_mega_moe_config(
        run_tokens,
        config_mtpr,
        experts_per_rank=self.experts_per_rank,
        model_dim=self.model_dim,
        inter_dim=self.inter_dim,
    ).stage1
```

Do not pass `fixed_slot_dispatch=True` from the new class, regardless of the actual MTPR.

- [ ] **Step 7: Run static checks**

```bash
pytest -q op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py
python -m py_compile aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py
```

Expected: PASS.

- [ ] **Step 8: Commit the resource owner**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py \
        op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py
git commit -m "feat(mega_moe): add standalone stage1 resources"
```

### Task 3: Add the dedicated compact fused Stage1 kernel

**Files:**
- Create: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py`
- Modify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py`
- Create: `op_tests/multigpu_tests/test_mega_moe_stage1.py`

- [ ] **Step 1: Add a failing multi-rank prequantized smoke test**

Create an 8-rank test harness using the same distributed/Mori initialization and W1 preparation pattern as `test_mega_moe_v2.py`. Its first case must construct DSV4 EP8 resources and call:

```python
op = MegaMoEStage1(
    rank=rank,
    world_size=world,
    model_dim=7168,
    inter_dim=3072,
    experts=384,
    topk=6,
    quant="a8w4",
    w1=w1_kernel,
    w1_scale=w1_scale_kernel,
    max_tok_per_rank=256,
    swiglu_limit=10.0,
)

x_q, x_scale = per_1x32_mx_quant(x_bf16, quant_mode="fp8")
result = op.forward_prequant(x_q, x_scale, route_weights, topk_ids)
torch.cuda.synchronize()

assert result.logical_tokens == world * local_tokens
assert result.num_local_experts == 384 // world
assert result.inter_sorted_quant.dtype == torch.float8_e4m3fn
assert result.inter_sorted_quant.shape == (result.max_sorted, 3072)
assert result.inter_sorted_shuffled_scale.dtype == dtypes.fp8_e8m0
assert result.inter_sorted_shuffled_scale.shape == (
    ((result.max_sorted + 255) // 256) * 256,
    96,
)
assert result.sorted_token_ids.dtype == torch.int32
assert result.sorted_token_ids.shape == (result.max_sorted,)
assert result.sorted_weights.dtype == torch.float32
assert result.sorted_weights.shape == (result.max_sorted,)
assert result.sorted_expert_ids.dtype == torch.int32
assert result.sorted_expert_ids.shape == (
    result.max_sorted // result.sort_block_m,
)
assert result.num_valid_ids.dtype == torch.int32
assert result.num_valid_ids.shape == (2,)
```

Add this CLI selection so individual checks can be run without executing the
whole multi-rank suite:

```python
parser.add_argument(
    "--mode",
    default="all",
    help="comma-separated checks: smoke,metadata,numeric,entrypoints,lifetime,graph,launch-count,abi,all",
)
parser.add_argument("--local-tokens", type=int, default=2)
parser.add_argument("--max-tok-per-rank", type=int, default=256)
requested_modes = {item.strip() for item in args.mode.split(",") if item.strip()}
if "all" in requested_modes:
    requested_modes = {
        "smoke",
        "metadata",
        "numeric",
        "entrypoints",
        "lifetime",
        "graph",
        "launch-count",
        "abi",
    }
```

- [ ] **Step 2: Run the smoke test and verify it fails**

Run:

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode smoke --local-tokens 2 --max-tok-per-rank 256
```

Expected: FAIL because the dedicated launcher is not implemented.

- [ ] **Step 3: Create the dedicated compile/run entry**

Use `compile_mega_moe_stage1` and `run_mega_moe_stage1` from
`mega_moe_stage1.py:71-502` as the behavioral reference. Implement the compact
subset independently in `mega_moe_fmoe_stage1.py` under the names
`compile_mega_moe_fmoe_stage1` and `run_mega_moe_fmoe_stage1`, with the following
exact runtime argument order:

```python
def run_mega_moe_fmoe_stage1(
    out,
    rx,
    w1,
    rx_scale,
    w1_scale,
    internal_tile_row_base,
    internal_expert_ids,
    internal_num_valid,
    out_scale,
    result_sorted_token_ids,
    result_sorted_weights,
    result_sorted_expert_ids,
    result_num_valid_ids,
    row_capacity,
    addr_disp,
    run_tokens,
    addr_in_tok,
    addr_in_idx,
    addr_in_wts,
    addr_in_scale,
    addr_parity,
    addr_expected,
    stream,
    **compile_kwargs,
):
    launch = compile_mega_moe_fmoe_stage1(**compile_kwargs)
    _run_compiled(
        launch,
        out,
        rx,
        w1,
        rx_scale,
        w1_scale,
        internal_tile_row_base,
        internal_expert_ids,
        internal_num_valid,
        out_scale,
        result_sorted_token_ids,
        result_sorted_weights,
        result_sorted_expert_ids,
        result_num_valid_ids,
        row_capacity,
        addr_disp,
        run_tokens,
        addr_in_tok,
        addr_in_idx,
        addr_in_wts,
        addr_in_scale,
        addr_parity,
        addr_expected,
        stream,
    )
```

Use `mega_moe_stage1.py:71-472` as the scheduling reference, but make the dedicated kernel compact-only:

```text
- Import only emit_dispatch_plan, emit_dispatch_group, and emit_dispatch_payload.
- Omit emit_direct_fixed_slot_payload and emit_direct_fixed_slot_finalize.
- Set direct_fixed_slot=False as a compile-time fact.
- Preserve the owner, producer, and shared GEMM work-pool ticket roles.
- Preserve launch_grid_x, grid_epoch_slot, WORK_SHARDS, and the existing epoch protocol.
- Preserve external grouping/counting and payload chunk/tile-ready branches.
- Preserve the PLAN_READY and PAYLOAD_READY/TILE_READY waits.
- Reuse build_fused_gemm1 unchanged.
```

The dedicated kernel must have a unique name prefix such as:

```python
kernel_name = (
    f"megamoe_fmoe_stage1_compact_t{sort_block_m}x{tile_n}x{tile_k}"
    f"_w{num_waves}_gm{grid_mult}_dcu{num_dispatch_cu}"
)
```

- [ ] **Step 4: Preserve the original synchronization protocol exactly**

Copy the owner initialization and non-owner wait logic from `mega_moe_stage1.py:210-282`, including:

```python
ticket64 = comm_ops.atomic_add_agent(entry_count_slot, fx.Int64(1))
generation = ticket64 // fx.Int64(launch_grid_x)
ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
gate_epoch = fx.Int32(generation + fx.Int64(1))
```

Retain system release/acquire around cross-rank `LAUNCH_READY`, `COUNT_DONE`, plan, and payload publication. Do not replace these operations with `fx.barrier()`, which is CTA-local.

- [ ] **Step 5: Reuse the compact dispatch calls with their current arguments**

The new kernel must call the existing helpers with the same internal MTPR-strided source encoding:

```python
emit_dispatch_plan(
    num_waves=NUM_WAVES,
    fz_npes=fz_npes,
    fz_epr=fz_epr,
    fz_k=fz_k,
    fz_mtpr=fz_mtpr,
    fz_rank=fz_rank,
    fz_tile_m=fz_tile_m,
    fz_total_experts=fz_total_experts,
    addr_disp=addr_disp,
    i32_cur_tok=i32_cur_tok,
    addr_in_idx=addr_in_idx,
    parity=payload_parity,
    expected=payload_expected,
    external_grouping=external_grouping,
    external_counting=external_counting,
    dispatch_blocks=dispatch_blocks,
    payload_chunk_rows=payload_chunk_rows,
    payload_tile_ready=payload_tile_ready,
)
```

Use the corresponding existing signatures for `emit_dispatch_group` and `emit_dispatch_payload`; do not duplicate those functions in the new file.

- [ ] **Step 6: Build and run GEMM1 using the existing builder**

Call `build_fused_gemm1` with the same receive buffers and weight interpretation as current MegaMoE:

```python
expert_of_flat, do_scheduled_tile = build_fused_gemm1(
    x_tensor=rx,
    w_rsrc=w_rsrc,
    sw_rsrc=sw_rsrc,
    sx_rsrc=sx_rsrc,
    out_rsrc=out_rsrc,
    os_rsrc=os_rsrc,
    trb_rsrc=tile_row_base_rsrc,
    expert_rsrc=expert_rsrc,
    out_tensor=out,
    a_buf=a_buf,
    a_scale_lds=a_scale_lds,
    c_tile=c_tile,
    model_dim=model_dim,
    inter_dim=inter_dim,
    sort_block_m=sort_block_m,
    tile_n=tile_n,
    num_waves=NUM_WAVES,
    n_per_wave=n_per_wave,
    wave_id=wave_id,
    m_repeat=M_REPEAT,
    num_acc_n=NUM_ACC_N,
    a_k_step_bytes=A_K_STEP_BYTES,
    total_threads=TOTAL_THREADS,
    k_iters=K_ITERS,
    a_lds_i32=a_lds_i32,
    n_tiles=N_TILES,
    expert_offset=rank * experts_per_rank,
    b_cache_modifier=b_cache_modifier,
    swizzle_a=swizzle_a,
    pipe_weights=pipe_weights,
    mfma_amajor=mfma_amajor,
    async_a_copy=async_a_copy,
    use_tile_resource=use_tile_resource,
    swiglu_limit=swiglu_limit,
)
```

The `out` and `out_scale` arguments are the per-call output tensors, not instance-owned tensors.

- [ ] **Step 7: Implement per-call output allocation and `forward_prequant`**

Add this allocation method to `MegaMoEStage1`:

```python
def _allocate_call_outputs(self, run_tokens: int, sort_block_m: int):
    layout = _stage1_output_layout(
        run_tokens=run_tokens,
        world_size=self.world_size,
        experts_per_rank=self.experts_per_rank,
        topk=self.topk,
        inter_dim=self.inter_dim,
        sort_block_m=sort_block_m,
    )
    outputs = (
        torch.empty(
            (layout.max_sorted, self.inter_dim),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        ),
        torch.empty(
            (layout.scale_rows, layout.scale_cols),
            dtype=dtypes.fp8_e8m0,
            device=self.device,
        ),
        torch.empty(layout.max_sorted, dtype=torch.int32, device=self.device),
        torch.empty(layout.max_sorted, dtype=torch.float32, device=self.device),
        torch.empty(layout.expert_blocks, dtype=torch.int32, device=self.device),
        torch.empty(2, dtype=torch.int32, device=self.device),
    )
    return layout, outputs
```

Add `_validate_runtime_inputs()` with the following checks:

```python
def _validate_runtime_inputs(self, x_fp8, x_scale, route_weights, topk_ids):
    run_tokens = int(x_fp8.shape[0])
    if not 0 < run_tokens <= self.max_tok_per_rank:
        raise ValueError(
            f"run_tokens={run_tokens} must be in [1, {self.max_tok_per_rank}]"
        )
    if x_fp8.dtype != torch.float8_e4m3fn or not x_fp8.is_contiguous():
        raise ValueError("x_fp8 must be contiguous float8_e4m3fn")
    if tuple(x_fp8.shape) != (run_tokens, self.model_dim):
        raise ValueError(
            f"x_fp8 must have shape ({run_tokens}, {self.model_dim})"
        )
    if x_scale.element_size() != 1 or not x_scale.is_contiguous():
        raise ValueError("x_scale must be contiguous one-byte E8M0 storage")
    if tuple(x_scale.shape) != (run_tokens, self.scale_dim):
        raise ValueError(
            f"x_scale must have shape ({run_tokens}, {self.scale_dim})"
        )
    if route_weights.dtype != torch.float32 or not route_weights.is_contiguous():
        raise ValueError("route_weights must be contiguous float32")
    if tuple(route_weights.shape) != (run_tokens, self.topk):
        raise ValueError(
            f"route_weights must have shape ({run_tokens}, {self.topk})"
        )
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous int32")
    if tuple(topk_ids.shape) != (run_tokens, self.topk):
        raise ValueError(
            f"topk_ids must have shape ({run_tokens}, {self.topk})"
        )
    for name, tensor in (
        ("x_fp8", x_fp8),
        ("x_scale", x_scale),
        ("route_weights", route_weights),
        ("topk_ids", topk_ids),
    ):
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")
    return run_tokens
```

Implement the private helper with this signature:

```python
def _forward_prequant_impl(
    self,
    x_fp8,
    x_scale,
    route_weights,
    topk_ids,
    *,
    stream,
    extra_keepalive: tuple[torch.Tensor, ...] = (),
):
```

It selects the config, allocates the outputs, launches
`run_mega_moe_fmoe_stage1` with this mapping:

```python
run_tokens = self._validate_runtime_inputs(
    x_fp8, x_scale, route_weights, topk_ids
)
config = self._select_stage1_config(run_tokens)
layout, outputs = self._allocate_call_outputs(
    run_tokens, config.sort_block_m
)
(
    out,
    out_scale,
    sorted_token_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
) = outputs
group = self._group
views = group._ll_views()
torch_stream = (
    torch.cuda.current_stream(self.device) if stream is None else stream
)

run_mega_moe_fmoe_stage1(
    out,
    views["rx_em"],
    self._w1,
    views["scale_em_i32"],
    self._w1_scale,
    group.tile_row_base,
    group.sorted_expert_ids,
    group.num_valid,
    out_scale,
    sorted_token_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    fx.Int32(layout.max_sorted),
    fx.Int64(self._dispatch_table.data_ptr()),
    fx.Int32(run_tokens),
    fx.Int64(x_fp8.data_ptr()),
    fx.Int64(topk_ids.data_ptr()),
    fx.Int64(route_weights.data_ptr()),
    fx.Int64(x_scale.data_ptr()),
    fx.Int64(self._epoch_parity.data_ptr()),
    fx.Int64(self._epoch_expected.data_ptr()),
    fx.Stream(torch_stream),
    model_dim=self.model_dim,
    inter_dim=self.inter_dim,
    rank=self.rank,
    experts_per_rank=self.experts_per_rank,
    fuse_npes=self.world_size,
    fuse_topk=self.topk,
    fuse_cap=group.ll_cap,
    fuse_mtpr=self.max_tok_per_rank,
    fuse_scale_dim=self.scale_dim,
    num_cu=self._num_cu,
    sort_block_m=config.sort_block_m,
    tile_n=config.tile_n,
    tile_k=config.tile_k,
    num_waves=config.num_waves,
    grid_mult=config.grid_mult,
    pipe_weights=config.pipe_weights,
    mfma_amajor=config.mfma_amajor,
    swizzle_a=config.swizzle_a,
    async_a_copy=config.async_a_copy,
    num_dispatch_cu=config.num_dispatch_cu,
    use_tile_resource=config.use_tile_resource,
    waves_per_eu_hint=config.waves_per_eu_hint,
    b_nt=config.b_nt,
    work_shards=config.work_shards,
    external_grouping=config.external_grouping,
    external_counting=config.external_counting,
    payload_chunk_rows=config.payload_chunk_rows,
    payload_tile_ready=config.payload_tile_ready,
    swiglu_limit=self.swiglu_limit,
)
```

It then returns:

```python
return MegaMoEStage1Output(
    inter_sorted_quant=out,
    inter_sorted_shuffled_scale=out_scale,
    sorted_token_ids=sorted_token_ids,
    sorted_weights=sorted_weights,
    sorted_expert_ids=sorted_expert_ids,
    num_valid_ids=num_valid_ids,
    logical_tokens=layout.logical_tokens,
    max_sorted=layout.max_sorted,
    num_local_experts=self.experts_per_rank,
    model_dim=self.model_dim,
    inter_dim=self.inter_dim,
    topk=self.topk,
    sort_block_m=config.sort_block_m,
    _keepalive=(x_fp8, x_scale, route_weights, topk_ids) + extra_keepalive,
)
```

The public `forward_prequant()` calls `_forward_prequant_impl` with an empty
`extra_keepalive` tuple. Do not read `num_valid_ids` on the host.

- [ ] **Step 8: Run the prequantized smoke test**

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode smoke --local-tokens 2 --max-tok-per-rank 256
```

Expected: PASS. The metadata tensors have the correct allocated ABI; their
semantic contents are tested in Task 4.

- [ ] **Step 9: Run syntax and import checks**

```bash
python -m py_compile \
  aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py \
  aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py
```

Expected: PASS.

- [ ] **Step 10: Commit the dedicated kernel core and prequantized entry**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py \
        aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py \
        op_tests/multigpu_tests/test_mega_moe_stage1.py
git commit -m "feat(mega_moe): add standalone compact stage1 kernel"
```

### Task 4: Publish FMoE v2 metadata inside the fused kernel

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py`
- Modify: `op_tests/multigpu_tests/test_mega_moe_stage1.py`

- [ ] **Step 1: Add failing route-contract checks**

After synchronizing in the test process, inspect the first `num_valid_ids[0]` rows and require:

```python
packed = result.sorted_token_ids[:valid_rows]
dense_token = packed & 0x00FFFFFF
slot = packed >> 24

valid = dense_token < result.logical_tokens
assert torch.all(slot[valid] < result.topk)
assert torch.all(result.sorted_expert_ids[: valid_rows // result.sort_block_m] >= 0)
assert torch.all(
    result.sorted_expert_ids[: valid_rows // result.sort_block_m]
    < result.num_local_experts
)
```

Build the expected route multiset from all-gathered `topk_ids` and `route_weights`, retaining only routes whose global Expert owner is the current rank. Compare keys `(dense_token, slot, local_expert)` and their FP32 weights; do not compare row order within an Expert.

- [ ] **Step 2: Run the route-contract case and verify it fails**

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode metadata --local-tokens 2 --max-tok-per-rank 256
```

Expected: FAIL because the per-call metadata outputs are not populated.

- [ ] **Step 3: Add a dedicated metadata-export helper inside the new kernel file**

Implement a local FlyDSL helper with this semantic body:

```python
@flyc.jit
def export_fmoe_tile_metadata(
    *,
    flat,
    n_tiles,
    sort_block_m,
    total_threads,
    rank,
    experts_per_rank,
    npes,
    mtpr,
    run_tokens,
    tile_row_base_rsrc,
    internal_expert_rsrc,
    internal_srcmap_rsrc,
    internal_weight_rsrc,
    result_token_rsrc,
    result_weight_rsrc,
    result_expert_rsrc,
):
    m_tile = flat // fx.Int32(n_tiles)
    n_tile = flat - m_tile * fx.Int32(n_tiles)
    if n_tile == fx.Int32(0):
        row_base = _buffer_load(tile_row_base_rsrc, m_tile, fx.Int32)
        global_expert = _buffer_load(internal_expert_rsrc, m_tile, fx.Int32)
        if fx.thread_idx.x == fx.Int32(0):
            _buffer_store(
                result_expert_rsrc,
                m_tile,
                global_expert - fx.Int32(rank * experts_per_rank),
                fx.Int32,
            )
        for row_in_tile in range(fx.thread_idx.x, fx.Int32(sort_block_m), fx.Int32(total_threads)):
            input_row = row_base + row_in_tile
            output_row = m_tile * fx.Int32(sort_block_m) + row_in_tile
            packed = _buffer_load(internal_srcmap_rsrc, input_row, fx.Int32)
            low24 = packed & fx.Int32(0x00FFFFFF)
            slot = packed >> fx.Int32(24)
            source_rank = low24 >> fx.Int32((mtpr.bit_length() - 1))
            local_token = low24 & fx.Int32(mtpr - 1)
            valid = (source_rank < fx.Int32(npes)) & (local_token < run_tokens)
            dense_token = source_rank * run_tokens + local_token
            public_id = valid.select(
                dense_token | (slot << fx.Int32(24)),
                fx.Int32(npes) * run_tokens,
            )
            weight = _buffer_load(internal_weight_rsrc, input_row, fx.Float32)
            weight = valid.select(weight, fx.Float32(0.0))
            _buffer_store(result_token_rsrc, output_row, public_id, fx.Int32)
            _buffer_store(result_weight_rsrc, output_row, weight, fx.Float32)
```

Use the existing file's `fx.Int32`, `_buffer_load`, and `_buffer_store` helpers
for this body. The indexing, validity predicate, sentinel, and single-writer
rules shown above are normative.

- [ ] **Step 4: Invoke metadata export exactly once per M tile**

In the shared work loop, after the existing payload-ready wait and system acquire and before or after `do_scheduled_tile(work)`, call the helper for the work item's `n_tile == 0` branch.

Do not let every N tile write the same metadata. The single-writer rule is required for deterministic ownership and to avoid unnecessary traffic.

- [ ] **Step 5: Publish `num_valid_ids` without a host read**

After the compact planner has completed, have the owner CTA copy the internal device values to the per-call result:

```python
if tid == fx.Int32(0):
    valid_rows = _buffer_load(internal_num_valid_rsrc, fx.Int32(0), fx.Int32)
    _buffer_store(result_num_valid_rsrc, fx.Int32(0), valid_rows, fx.Int32)
    _buffer_store(result_num_valid_rsrc, fx.Int32(1), fx.Int32(0), fx.Int32)
```

This must remain inside the fused kernel. Do not call `.item()` and do not launch a copy kernel.

- [ ] **Step 6: Run the route-contract test**

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode metadata --local-tokens 2 --max-tok-per-rank 256
```

Expected: PASS on all ranks.

- [ ] **Step 7: Commit metadata publication**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py \
        op_tests/multigpu_tests/test_mega_moe_stage1.py
git commit -m "feat(mega_moe): emit fmoe stage1 metadata"
```

### Task 5: Add numerical correctness, BF16 input, and output lifetime

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py`
- Modify: `op_tests/multigpu_tests/test_mega_moe_stage1.py`

- [ ] **Step 1: Add a failing order-independent Stage1 reference test**

The test must all-gather source inputs/routes for reference only, then build
expected routes targeting the current rank:

```python
expected = {}
for source_rank in range(world):
    for local_token in range(local_tokens):
        dense_token = source_rank * local_tokens + local_token
        for slot in range(topk):
            global_expert = int(all_ids[source_rank, local_token, slot])
            if global_expert // experts_per_rank != rank:
                continue
            local_expert = global_expert - rank * experts_per_rank
            expected[(dense_token, slot, local_expert)] = float(
                all_weights[source_rank, local_token, slot]
            )
```

For each valid emitted row, decode the packed token/slot, obtain the tile's local
Expert ID, and compare the dequantized Stage1 row against:

```python
gate_up = x_dequant[dense_token] @ w1_dequant[local_expert].T
gate, up = gate_up.chunk(2)
if swiglu_limit > 0:
    gate = gate.clamp(max=swiglu_limit)
    up = up.clamp(-swiglu_limit, swiglu_limit)
reference = F.silu(gate) * up
```

Use the current MegaMoE rule that `swiglu_limit == 0` disables both clamps.
Require the emitted route-key set to equal the expected set, exact FP32 route
weights, and mean Stage1 cosine greater than `0.99`.

- [ ] **Step 2: Add failing BF16/prequant equivalence and lifetime tests**

Require both entries to match semantically after canonicalizing by
`(dense_token, slot, local_expert)`. Also require independent storage:

```python
first = op.forward_prequant(x1_q, s1, weights1, ids1)
torch.cuda.synchronize()
snapshot = tuple(t.clone() for t in public_tensors(first))
second = op.forward_prequant(x2_q, s2, weights2, ids2)
torch.cuda.synchronize()

assert all(
    a.data_ptr() != b.data_ptr()
    for a, b in zip(public_tensors(first), public_tensors(second))
)
for before, after in zip(snapshot, public_tensors(first)):
    assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
```

- [ ] **Step 3: Run the new cases and verify the BF16 entry fails**

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode numeric,entrypoints,lifetime --local-tokens 2 --max-tok-per-rank 256
```

Expected: FAIL because the BF16 `forward` entry is not implemented. The
prequantized numerical and lifetime assertions remain active and must not regress.

- [ ] **Step 4: Implement the BF16 entry on the requested stream**

```python
def quantize(self, x_bf16):
    return per_1x32_mx_quant(x_bf16, quant_mode="fp8")


def forward(self, x_bf16, route_weights, topk_ids, *, stream=None):
    if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
        raise ValueError("x_bf16 must be contiguous bfloat16")
    torch_stream = (
        torch.cuda.current_stream(self.device) if stream is None else stream
    )
    with torch.cuda.stream(torch_stream):
        x_fp8, x_scale = self.quantize(x_bf16)
        return self._forward_prequant_impl(
            x_fp8,
            x_scale,
            route_weights,
            topk_ids,
            stream=torch_stream,
            extra_keepalive=(x_bf16, x_fp8, x_scale),
        )


forward_bf16 = forward
__call__ = forward
```

- [ ] **Step 5: Confirm the output scale is the v2 layout**

Use the same byte-index formula as the existing FlyDSL Stage1/FMoE v2 path when dequantizing in the test:

```text
(row >> 5) * (padded_scale_cols * 32)
+ (scale_col >> 3) * 256
+ (scale_col & 3) * 64
+ (row & 15) * 4
+ ((scale_col >> 2) & 1) * 2
+ ((row >> 4) & 1)
```

Do not reinterpret the returned scale as ordinary row-major `[row, inter_dim / 32]` storage.

- [ ] **Step 6: Compare BF16 and prequantized entry points semantically**

Because route order within an Expert may differ, canonicalize both results by `(dense_token, slot, local_expert)` and compare weights plus dequantized activation values.

- [ ] **Step 7: Run numerical, entry-point, and lifetime tests**

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode numeric,entrypoints,lifetime --local-tokens 2 --max-tok-per-rank 256
```

Expected: PASS.

- [ ] **Step 8: Commit numerical correctness, BF16 support, and lifetime coverage**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py \
        op_tests/multigpu_tests/test_mega_moe_stage1.py
git commit -m "test(mega_moe): validate standalone stage1 numerics"
```

### Task 6: Add CUDA Graph and single-kernel verification

**Files:**
- Modify: `op_tests/multigpu_tests/test_mega_moe_stage1.py`
- Modify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py`

- [ ] **Step 1: Add a failing CUDA Graph capture/replay case**

Warm up the selected specialization, then capture only the prequantized call:

```python
op.forward_prequant(x_q, x_scale, route_weights, topk_ids)
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
capture_stream = torch.cuda.Stream()
with torch.cuda.graph(graph, stream=capture_stream):
    captured = op.forward_prequant(x_q, x_scale, route_weights, topk_ids)

for _ in range(3):
    graph.replay()
torch.cuda.synchronize()
validate_result_semantics(captured, reference)
```

- [ ] **Step 2: Run the graph case and verify it fails before stream fixes**

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode graph --local-tokens 2 --max-tok-per-rank 256
```

Expected: FAIL if any `.item()`, host synchronization, wrong-stream allocation/launch, or capture-unsafe temporary remains.

- [ ] **Step 3: Make stream handling capture-safe**

Centralize stream normalization:

```python
def _torch_stream(self, stream):
    return torch.cuda.current_stream(self.device) if stream is None else stream

def _fx_stream(self, stream):
    return fx.Stream(self._torch_stream(stream))
```

Allocate outputs and launch quantization/Stage1 under the selected torch stream. Remove every `.item()`, `torch.cuda.synchronize()`, and host read from the forward path.

- [ ] **Step 4: Verify one fused kernel for the prequantized path**

Add a profiler mode that warms compilation first, profiles one `forward_prequant` call, and checks that the operator emits exactly one application kernel whose name starts with `megamoe_fmoe_stage1_compact_`. Allocator/runtime events are ignored; no metadata-copy kernel may appear.

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    op.forward_prequant(x_q, x_scale, route_weights, topk_ids)
    torch.cuda.synchronize()

stage1_events = [
    event for event in prof.key_averages()
    if "megamoe_fmoe_stage1_compact_" in event.key
]
assert sum(event.count for event in stage1_events) == 1
```

- [ ] **Step 5: Run graph and launch-count modes**

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode graph,launch-count --local-tokens 2 --max-tok-per-rank 256
```

Expected: PASS.

- [ ] **Step 6: Commit graph support**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py \
        op_tests/multigpu_tests/test_mega_moe_stage1.py
git commit -m "test(mega_moe): cover stage1 graph capture"
```

### Task 7: Export the API and verify the target FMoE v2 ABI

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/__init__.py`
- Modify: `op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py`
- Modify: `op_tests/multigpu_tests/test_mega_moe_stage1.py`

- [ ] **Step 1: Add a failing public-import test**

```python
def test_public_stage1_exports():
    from aiter.ops.flydsl.kernels.mega_moe import (
        MegaMoEStage1,
        MegaMoEStage1Output,
    )

    assert MegaMoEStage1.__name__ == "MegaMoEStage1"
    assert MegaMoEStage1Output.__name__ == "MegaMoEStage1Output"
```

- [ ] **Step 2: Add lazy exports**

Extend `_LAZY` in `aiter/ops/flydsl/kernels/mega_moe/__init__.py`:

```python
_LAZY = {
    "MegaMoEConfig": "mega_moe_config",
    "MegaMoEStage1": "mega_moe_stage1_op",
    "MegaMoEStage1Output": "mega_moe_stage1_op",
    "MegaMoEV2": "mega_moe_v2",
    "Stage1Config": "mega_moe_config",
    "Stage2Config": "mega_moe_config",
    "compile_gemm1": "gemm1",
    "gemm1_kernel": "gemm1",
    "select_mega_moe_config": "mega_moe_config",
}
```

- [ ] **Step 3: Add a target-ABI smoke launch**

Using the result tensors, invoke the ordinary v2 Stage2 API with an EP-local shape and matching SBM:

```python
mxfp4_moe_gemm2(
    inter_sorted_quant=result.inter_sorted_quant,
    inter_sorted_shuffled_scale=result.inter_sorted_shuffled_scale,
    w2_u8=w2_kernel.view(torch.uint8),
    w2_scale_u8=w2_scale_kernel.view(torch.uint8),
    sorted_expert_ids=result.sorted_expert_ids,
    cumsum_tensor=result.num_valid_ids,
    sorted_token_ids=result.sorted_token_ids,
    sorted_weights=result.sorted_weights,
    out=local_partial,
    M_logical=result.logical_tokens,
    max_sorted=result.max_sorted,
    NE=result.num_local_experts,
    D_HIDDEN=result.model_dim,
    D_INTER=result.inter_dim,
    topk=result.topk,
    BM=32,
    BN=128,
    BK=128,
    use_nt=False,
    a_dtype="fp8",
    b_dtype="fp4",
    epilog="atomic",
    SBM=result.sort_block_m,
)
```

For the first-phase DSV4 EP shape, `D_INTER=3072` is divisible by the reference
kernel's `BK=128`, so the smoke test uses the exact Stage2 geometry from CSV line
3: `BM=32`, `BN=128`, `BK=128`, atomic epilogue, `SBM=32`. The purpose of this
check is ABI acceptance, not Stage2/all-reduce numerical validation.

- [ ] **Step 4: Run public-import and ABI smoke tests**

```bash
pytest -q op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py

torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode abi --local-tokens 2 --max-tok-per-rank 256
```

Expected: PASS.

- [ ] **Step 5: Commit public exports and ABI coverage**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/__init__.py \
        op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py \
        op_tests/multigpu_tests/test_mega_moe_stage1.py
git commit -m "feat(mega_moe): export standalone stage1 operator"
```

### Task 8: Run regression and final verification

**Files:**
- Verify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py`
- Verify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py`
- Verify: `aiter/ops/flydsl/kernels/mega_moe/__init__.py`
- Verify: `op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py`
- Verify: `op_tests/multigpu_tests/test_mega_moe_stage1.py`
- Verify unchanged behavior: `op_tests/multigpu_tests/test_mega_moe_v2.py`

- [ ] **Step 1: Run formatting and static checks**

```bash
python -m py_compile \
  aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py \
  aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py \
  op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py \
  op_tests/multigpu_tests/test_mega_moe_stage1.py

ruff check \
  aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py \
  aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py \
  aiter/ops/flydsl/kernels/mega_moe/__init__.py \
  op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py \
  op_tests/multigpu_tests/test_mega_moe_stage1.py

git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 2: Run the complete new Stage1 suite**

```bash
pytest -q op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py

torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_stage1.py \
  --mode all --local-tokens 2 --max-tok-per-rank 256
```

Expected: all contract, metadata, numerical, entry-point, lifetime, graph, launch-count, and ABI-smoke checks pass.

- [ ] **Step 3: Run existing MegaMoEV2 fixed-slot regression**

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_v2.py \
  --bs-list 2,128 --iters 1 --accuracy-max-bs 128 \
  --max-tok-per-rank 128
```

Expected: both sizes pass the existing relative-L2 threshold.

- [ ] **Step 4: Run existing MegaMoEV2 compact regression**

```bash
torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_mega_moe_v2.py \
  --bs-list 2,512 --iters 1 --accuracy-max-bs 512 \
  --max-tok-per-rank 1024
```

Expected: both sizes pass the existing relative-L2 threshold.

- [ ] **Step 5: Confirm scope and changed files**

```bash
git status --short
git diff --stat HEAD~1..HEAD
git diff --name-only $(git merge-base HEAD origin/main)..HEAD
```

Confirm that no TP AllGather, TP all-reduce, fixed-slot support, performance gate, high-level dispatcher integration, or unrelated cleanup was introduced.

Inspect additive changes to pre-existing files against the design-only baseline:

```bash
git diff --unified=0 b84c97b73 -- \
  aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py \
  aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py \
  aiter/ops/flydsl/kernels/mega_moe/dispatch.py \
  aiter/ops/flydsl/kernels/mega_moe/gemm1.py \
  aiter/ops/flydsl/kernels/mega_moe/gemm_util.py \
  aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py \
  aiter/fused_moe.py \
  aiter/ops/flydsl/kernels/mxmoe_dispatcher.py
```

Expected: either no output, or additive hunks containing only new definitions or
new export entries. There must be no removed line (`-`, excluding the diff
header) from a pre-existing callable. If an existing callable would need an edit,
move the new behavior into a new helper/class/kernel instead.

- [ ] **Step 6: Commit verification corrections only when files changed**

If verification required a code or test correction, commit only those directly related files:

```bash
git add aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py \
        aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py \
        aiter/ops/flydsl/kernels/mega_moe/__init__.py \
        op_tests/flydsl_tests/test_mega_moe_stage1_contracts.py \
        op_tests/multigpu_tests/test_mega_moe_stage1.py
git commit -m "test(mega_moe): finalize standalone stage1 coverage"
```

If no files changed, do not create an empty commit.

## Completion checklist

- [ ] The new public API imports without importing Stage2/combine code.
- [ ] `forward_prequant` launches one dedicated fused Stage1 kernel.
- [ ] `forward` launches input quantization followed by that Stage1 kernel.
- [ ] All six FMoE v2 tensor outputs use independent per-call storage.
- [ ] Output token IDs are dense rank-major IDs.
- [ ] Output Expert IDs are local W2 indices.
- [ ] Route order is treated as unspecified within an Expert.
- [ ] No device scalar is read on the host path.
- [ ] CUDA Graph capture/replay passes.
- [ ] The target v2 GEMM2 API accepts the returned output.
- [ ] Existing MegaMoEV2 fixed and compact correctness regressions pass.
- [ ] Existing public API implementations have no removed or rewritten lines from commit `b84c97b73`; any existing-file changes are purely additive.
- [ ] No performance requirement is added.

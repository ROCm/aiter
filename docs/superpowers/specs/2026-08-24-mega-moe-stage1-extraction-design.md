# MegaMoE Stage1 Extraction Design

**Date:** 2026-08-24  
**Status:** Approved design  
**Scope:** Extract the current MegaMoE input-through-GEMM1 path into a standalone Stage1 operator whose outputs match the ordinary FlyDSL v2 FMoE GEMM2 ABI.

## 1. Context

`MegaMoEV2` currently owns the complete pipeline:

```text
BF16 input quantization
  -> EP dispatch/count/plan/group/payload
  -> GEMM1
  -> SwiGLU
  -> intermediate FP8/E8M0 quantization
  -> MegaMoE GEMM2/P2P return
  -> combine
```

The first project phase extracts everything through the end of Stage1 into a new standalone operator. The new operator remains an EP operator in this phase. It is intentionally designed so that a later project can replace the EP dispatch front-end with TP4/TP8 AllGather while retaining the GEMM1 core and output contract.

The downstream ABI reference is the FlyDSL v2 FMoE GEMM2 family. The concrete reference row is line 3 of:

```text
docs/fp8_retune_config/dsv4_fp8fp4_tp8_k384_flydslv2_tuned_20260726_144002.csv
```

That row selects:

```text
Stage1: flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8
Stage2: flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_sbm32
```

The exact row has `q_dtype_a=FP8`, `q_dtype_w=MXFP4`, BF16 output, Stage2 `BM=32`, `BN=128`, `BK=128`, atomic epilogue, and `SBM=32`.

The first phase does **not** adopt that row's TP8 weight shape. It keeps current MegaMoE EP semantics and shapes. For the DSV4 EP8 case, each rank owns 48 Experts and uses local `inter_dim=3072`. The CSV row is the output-layout and consumer-ABI reference. A later TP8 phase changes each rank to 384 Experts and `inter_dim=384`; TP4 analogously uses its per-rank intermediate shard.

The existing experimental branch `dev/tp_fuse_gemm1_v0` is not an implementation source for this work.

## 2. Goals

1. Add a public, stateful `MegaMoEStage1` operator.
2. Support both BF16 input and prequantized FP8/E8M0 input.
3. Preserve current EP ownership and cross-rank compact dispatch semantics.
4. Run dispatch, GEMM1, SwiGLU, intermediate FP8 quantization, and FMoE metadata publication in one fused Stage1 kernel for the prequantized entry.
5. Return all tensors and host metadata required by ordinary FlyDSL v2 FMoE GEMM2.
6. Allocate independent output tensors for every call so a later call cannot overwrite an earlier result.
7. Support CUDA Graph capture without device-to-host scalar reads or host synchronization.
8. Keep the existing `MegaMoEV2` public API, resource ownership, and execution path unchanged.
9. Reuse existing quantization, compact dispatch, GEMM1, SwiGLU, and output-quantization building blocks wherever their semantics match.

## 3. Non-goals

- No fixed-slot dispatch in the new operator.
- No TP AllGather, TP-sharded GEMM1, or TP all-reduce in this phase.
- No support for unequal runtime token counts across ranks.
- No requirement that route order within an Expert exactly match `moe_sorting()`.
- No end-to-end Stage2 plus cross-rank reduction correctness test.
- No performance target, tuning work, or performance-regression gate.
- No integration into the high-level FMoE dispatcher.
- No conversion of the existing `MegaMoEV2` implementation to use the new class.

## 4. Selected Architecture

Use a dedicated fused Stage1 kernel rather than adding a public-output mode to the existing `compile_mega_moe_stage1` kernel.

Files:

- `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1_op.py`
  - Public `MegaMoEStage1` class.
  - Public `MegaMoEStage1Output` result type.
  - Input validation, output allocation, resource ownership, configuration selection, and launch wrappers.
- `aiter/ops/flydsl/kernels/mega_moe/mega_moe_fmoe_stage1.py`
  - Dedicated compile/run entry for the compact EP Stage1 kernel.
  - Ticket/epoch scheduling glue and direct FMoE metadata publication.
- `aiter/ops/flydsl/kernels/mega_moe/__init__.py`
  - Lazy exports for the new public types.
- `op_tests/multigpu_tests/test_mega_moe_stage1.py`
  - Multi-rank correctness, ABI, lifetime, and CUDA Graph coverage.

### 4.1 Reused implementation

The new path reuses:

- `per_1x32_mx_quant` for the BF16 entry.
- `FlyDSLDispatchGroupMajorOp` for compact receive buffers and symmetric P2P storage.
- The compact helpers in `dispatch.py`:
  - `emit_dispatch_plan`
  - `emit_dispatch_group`
  - `emit_dispatch_payload`
- `build_fused_gemm1` and its existing FP8 x MXFP4 loaders/MFMA path.
- The existing SwiGLU and per-1x32 FP8/E8M0 output epilogue.
- `Stage1Config` and the existing MegaMoE Stage1 tuning table. The selector uses
  `config_mtpr = max(max_tok_per_rank, 256)` so the new operator never selects
  the fixed-slot class.
- Existing `DispatchSlot` definitions and synchronization primitives.

### 4.2 New implementation

The new path implements:

- A Stage1-only resource owner that does not allocate W2, Stage2, or combine resources.
- A compact-only configuration selection path.
- A dedicated top-level fused kernel and launch wrapper.
- Per-call independent output allocation.
- Dense global token-ID conversion.
- Global-to-local Expert-ID conversion.
- Direct, output-row-aligned publication of FMoE route metadata.
- A stable result object describing all tensors and host-side dimensions needed by FMoE v2 GEMM2.

The existing `MegaMoEV2` continues to call its existing Stage1 kernel. It is not
changed to compose or delegate to `MegaMoEStage1`; the new kernel imports the
shared lower-level helpers listed above directly.

## 5. Public API

The public class is stateful because Stage1 requires symmetric receive buffers, P2P pointer tables, dispatch workspaces, and persistent epoch state.

```python
class MegaMoEStage1:
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        quant: str,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        max_tok_per_rank: int,
        swiglu_limit: float = 0.0,
    ): ...

    def quantize(self, x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        x_bf16: torch.Tensor,
        route_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        stream=None,
    ) -> MegaMoEStage1Output: ...

    def forward_prequant(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        route_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        stream=None,
    ) -> MegaMoEStage1Output: ...
```

`forward_bf16` may alias `forward`, and `__call__` may alias `forward`, matching `MegaMoEV2` conventions.

W1 and W1 scale are constructor-owned references. Their logical formats remain the current MegaMoE formats:

```text
W1 logical values: [experts_per_rank, 2 * inter_dim, model_dim]
W1 packed MXFP4:   two values per stored byte
W1 scale:          per-1x32 E8M0, preshuffled for gate/up-interleaved W1
```

## 6. Result Contract

```python
@dataclass(frozen=True)
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

    _keepalive: tuple[torch.Tensor, ...]
```

The private `_keepalive` field retains temporary input-quantization tensors and any per-call backing storage until all asynchronously enqueued work that produces the result has completed.

### 6.1 Tensor fields

| Field | Dtype and shape | Contract |
|---|---|---|
| `inter_sorted_quant` | FP8 E4M3 `[max_sorted, inter_dim]` | GEMM1 + SwiGLU output in sorted/padded row order. |
| `inter_sorted_shuffled_scale` | E8M0 `[align_up(max_sorted, 256), align_up(inter_dim / 32, 8)]` | Native FlyDSL v2 GEMM2 scale layout, not a conventional row-major scale matrix. |
| `sorted_token_ids` | INT32 `[max_sorted]` | High 8 bits are Top-K slot; low 24 bits are dense global token ID. |
| `sorted_weights` | FP32 `[max_sorted]` | Route weight aligned with the same output row. Stage1 does not apply it. |
| `sorted_expert_ids` | INT32 `[max_sorted / sort_block_m]` | One local Expert ID per Stage1 sort tile. |
| `num_valid_ids` | INT32 `[2]` | Element 0 is the device-resident padded valid-row count. Both elements are initialized before launch. |

All tensor fields own independent per-call storage. A later call on the same operator cannot overwrite a previously returned result.

### 6.2 Host metadata and capacity

For equal `run_tokens` on every rank:

```text
logical_tokens = world_size * run_tokens
route_capacity = logical_tokens * topk
               + experts_per_rank * sort_block_m
               - topk
max_sorted = align_up(route_capacity, sort_block_m)
```

`max_sorted` is the Stage2 `max_sorted` argument and the row capacity of `inter_sorted_quant`.

The result maps directly to ordinary FlyDSL v2 GEMM2:

```python
mxfp4_moe_gemm2(
    inter_sorted_quant=result.inter_sorted_quant,
    inter_sorted_shuffled_scale=result.inter_sorted_shuffled_scale,
    sorted_expert_ids=result.sorted_expert_ids,
    cumsum_tensor=result.num_valid_ids,
    sorted_token_ids=result.sorted_token_ids,
    sorted_weights=result.sorted_weights,
    M_logical=result.logical_tokens,
    max_sorted=result.max_sorted,
    NE=result.num_local_experts,
    D_HIDDEN=result.model_dim,
    D_INTER=result.inter_dim,
    topk=result.topk,
    SBM=result.sort_block_m,
    # W2, output, BM/BN/BK and epilogue are selected by the downstream caller.
)
```

## 7. ID and Ordering Semantics

### 7.1 Input IDs

Input `topk_ids` use global Expert IDs in `[0, experts)`.

The compact dispatch path uses them to compute the destination rank and local Expert. Invalid Expert IDs retain the current MegaMoE behavior and are skipped.

### 7.2 Output Expert IDs

`sorted_expert_ids` use local Expert IDs:

```text
local_expert = global_expert - rank * experts_per_rank
```

This lets ordinary FMoE GEMM2 index the rank-local W2 tensor directly.

### 7.3 Output token IDs

Internal MegaMoE dispatch may continue to use its MTPR-strided source encoding:

```text
internal_low24 = source_rank * max_tok_per_rank + local_token
```

Before publishing the public result, the dedicated kernel converts it to a dense runtime token index:

```text
source_rank = internal_low24 >> log2(max_tok_per_rank)
local_token = internal_low24 & (max_tok_per_rank - 1)
dense_token = source_rank * run_tokens + local_token
public_id = dense_token | (topk_slot << 24)
```

Padding rows use `logical_tokens` as the invalid low-24-bit sentinel and a zero route weight.

The first phase requires all ranks to use the same `run_tokens`. It does not add a count exchange to verify or support unequal counts.

### 7.4 Ordering

Rows remain grouped by local Expert and padded to `sort_block_m`. `sorted_token_ids`, `sorted_weights`, and `inter_sorted_quant` must be aligned row by row.

The ordering of routes within one Expert is not required to match `moe_sorting()` exactly. Tests compare the semantic route multiset and numerical values rather than demanding byte-identical ordering.

## 8. Dedicated Kernel Data Flow

The prequantized entry performs one fused Stage1 launch:

```text
local FP8 input / E8M0 scale / routes
  -> epoch and LAUNCH_READY handshake
  -> compact count
  -> destination-owned plan
  -> route grouping
  -> P2P payload into persistent symmetric receive buffers
  -> PLAN/PAYLOAD/TILE ready protocol
  -> local W1 GEMM1
  -> SwiGLU
  -> per-1x32 FP8/E8M0 output quantization
  -> direct FMoE metadata publication
  -> independent MegaMoEStage1Output
```

The BF16 entry first launches the existing BF16-to-FP8/E8M0 quantization operation, then launches the fused Stage1 kernel.

### 8.1 Internal and public storage

Remote dispatch producers continue to write activation, input scale, source map, and route weight into persistent symmetric SHMEM owned by the Stage1 instance. Per-call outputs are ordinary local device tensors.

The dedicated kernel does not require per-call symmetric output allocation. After the existing payload-ready wait and system acquire, the GEMM consumer already has safe local access to the internal route metadata.

### 8.2 Direct metadata publication

For each M tile, the work item whose N-tile index is zero performs the metadata publication exactly once:

1. Load internal `srcmap` and route weight from `tile_row_base + row_in_tile`.
2. Convert the MTPR-strided source ID to the dense runtime token ID.
3. Write `sorted_token_ids[output_row]` and `sorted_weights[output_row]`.
4. Write the local Expert ID for the tile to `sorted_expert_ids[m_tile]`.
5. Write invalid-token sentinel and zero weight for padding rows.

The output row is the same sorted row used by the v2 GEMM1 epilogue:

```text
output_row = m_tile * sort_block_m + row_in_tile
```

The planner also publishes the padded valid-row count into the per-call `num_valid_ids` result. No postprocessing kernel is launched.

### 8.3 Reused synchronization

The dedicated kernel retains the current compact Stage1 protocol:

- ticket/generation role assignment
- epoch/parity state
- `LAUNCH_READY`
- `COUNT_DONE`
- `PLAN_READY`
- `PAIR_READY` and `PAIR_ORDER_READY`
- `PAYLOAD_READY` or `TILE_READY`
- agent-scope release/acquire for local workspace
- system-scope release/acquire for P2P-visible data

Kernel completion provides visibility to the next operation on the same stream.

## 9. Configuration Selection

The new operator always selects a compact Stage1 configuration.

- Runtime `run_tokens` continues to choose the token bucket.
- Instance `max_tok_per_rank` continues to size internal receive and dispatch capacity.
- When `max_tok_per_rank <= 255`, configuration lookup uses the bounded compact class rather than selecting fixed-slot.
- Existing Stage1 tuning fields remain available: SBM, tile N/K, wave count, dispatch CU count, work shards, external grouping/counting, payload chunking, tile-ready mode, and other current GEMM1 options.
- The result records the selected `sort_block_m`; the downstream v2 GEMM2 must use a compatible `SBM`.

The new operator does not own or select a Stage2 configuration.

## 10. Stream, Lifetime, and CUDA Graph Contract

- Every call allocates independent result tensors before launching the kernel.
- Result tensors remain valid after later calls and are not overwritten by the instance.
- Persistent dispatch/SHMEM workspace and epoch state remain single-buffered.
- Back-to-back calls on the same stream are supported because stream order serializes reuse of internal workspace.
- Concurrent calls on different streams using the same instance are unsupported.
- A downstream kernel may consume the result immediately on the same stream.
- Cross-stream consumption requires the caller to establish an event dependency.
- No device scalar is read on the host.
- No unconditional `torch.cuda.synchronize()` is performed in the call path.
- BF16 input quantization and Stage1 launch both execute on the caller-selected stream.
- Output sizes are computed from static tensor shapes and host configuration, keeping capture and replay stable.

## 11. Validation and Error Handling

### 11.1 Construction-time validation

- `quant == "a8w4"`.
- Target architecture is gfx95x when compilation occurs.
- `world_size <= 8`.
- `experts` is divisible by `world_size`.
- `max_tok_per_rank` is a positive power of two.
- Source-token low-24-bit and Top-K high-8-bit fields cannot overflow.
- `swiglu_limit >= 0`.
- W1 and W1 scale are contiguous, on the expected device, and have the expected packed byte counts/layout.
- Selected Stage1 tile parameters satisfy the existing kernel constraints.

### 11.2 Call-time validation

BF16 entry:

```text
x_bf16:       contiguous BF16 [run_tokens, model_dim]
route_weights: contiguous FP32 [run_tokens, topk]
topk_ids:      contiguous INT32 [run_tokens, topk]
```

Prequantized entry:

```text
x_fp8:         contiguous FP8 E4M3 [run_tokens, model_dim]
x_scale:       contiguous E8M0/UINT8 [run_tokens, model_dim / 32]
route_weights: contiguous FP32 [run_tokens, topk]
topk_ids:      contiguous INT32 [run_tokens, topk]
```

`run_tokens` must be positive and no greater than `max_tok_per_rank`. All tensors must be on the operator device.

The equal-token-count and collective-order requirements are documented preconditions rather than dynamically checked through extra communication.

## 12. Test Strategy

The implementation is complete when the following pass.

### 12.1 Contract tests

- Public import and class construction.
- Output dtype, shape, capacity, and device checks.
- FP8 scale layout shape and storage checks.
- Dense packed token-ID encoding and padding sentinel checks.
- Local Expert-ID range checks.
- `num_valid_ids` remains device-resident.
- Invalid dtype, shape, MTPR, topology, and capacity inputs fail clearly.

### 12.2 Multi-rank semantic correctness

For each valid output row:

1. Decode dense token ID and Top-K slot.
2. Resolve the corresponding source input and global route.
3. Verify the route targets the current rank and the emitted local Expert ID is correct.
4. Verify the emitted route weight matches the source route weight.
5. Dequantize the Stage1 FP8 row using the shuffled E8M0 scale layout.
6. Compare against a Torch W1 + SwiGLU reference for that route.

The comparison is order-independent within each Expert.

Coverage includes multiple runtime token counts, multiple supported Stage1 SBMs, and both inline and external grouping/counting configurations selected by current tuning rules where available.

### 12.3 Entry and lifetime tests

- BF16 and prequantized entry points produce semantically equivalent results.
- Two calls queued on the same stream without host synchronization return different output addresses.
- The second call does not alter the first call's result.
- CUDA Graph capture and replay succeed with stable tensor addresses inside the captured graph.

### 12.4 Compatibility tests

- Launch one matching `mxfp4_moe_gemm2` configuration using the returned fields to verify ABI compatibility.
- This is a launch/contract smoke test only; Stage2 plus cross-rank reduction numerical correctness is outside this phase.
- Run the existing MegaMoEV2 correctness test to ensure reused helpers have not changed the old path.

Performance measurement and performance thresholds are outside this phase.

## 13. Alternatives Considered

### 13.1 Add an output mode to the existing MegaMoE Stage1 kernel

This minimizes duplicated scheduler code, but expands the existing kernel ABI and makes a new experimental output contract part of the production MegaMoE kernel. It was rejected in favor of isolating the new path.

### 13.2 Dedicated fused Stage1 kernel — selected

This isolates the new contract and future TP changes. Existing lower-level primitives are reused; only the top-level scheduling glue and FMoE output publication are independently implemented.

### 13.3 Wrap `MegaMoEV2` and convert its outputs

This would allocate unrelated Stage2/combine resources, depend on private instance state, and normally require a postprocessing kernel. It conflicts with the requested single-kernel Stage1 boundary and was rejected.

## 14. Future TP4/TP8 Evolution

The follow-up TP project replaces the EP input/data-movement side while keeping the result ABI stable:

```text
Phase 1:
local EP input -> compact EP P2P dispatch -> shared GEMM1 core -> FMoE v2 output

Future TP:
local DP shard -> TP AllGather -> local grouping -> TP-sharded GEMM1
               -> the same FMoE v2 output
```

The TP phase will separately decide:

- TP process-group ownership and rank mapping.
- Equal-count versus variable-count AllGather.
- AllGather chunking and communication/compute overlap.
- All-Expert replication and W1/W2 intermediate-dimension sharding.
- Stage2 partial-output reduction.

Those choices do not enter the first-phase API beyond preserving the six-tensor FMoE v2 result contract.

## 15. Acceptance Criteria

The first phase is accepted when:

1. `MegaMoEStage1` is publicly importable and supports BF16 and prequantized inputs.
2. The new prequantized path uses exactly one fused Stage1 kernel after inputs are available.
3. The operator always uses compact EP dispatch and produces dense rank-major token IDs plus local Expert IDs.
4. Every call returns independently owned FMoE v2-compatible output tensors.
5. Stage1 output values and route metadata pass the order-independent multi-rank reference checks.
6. The target FP8 FlyDSL v2 GEMM2 accepts the returned ABI in a smoke launch.
7. CUDA Graph capture/replay passes without host synchronization or device scalar reads.
8. Existing MegaMoEV2 correctness tests remain passing.
9. No performance threshold is imposed in this phase.

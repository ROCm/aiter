# Architecture and hybrid metadata

## Recover the actual model computation

Read model code, library dispatch, and reference kernel together. Record:

- Every input/output shape and stride; logical versus physical/padded batch.
- Weight and activation quantization, scale group size and shuffled layout.
- Accumulation precision, rounding boundaries, activation constants, norm epsilon.
- Router score transform, correction bias, top-k tie order, normalization, and
  whether correction bias affects selection only or also mixture weights.
- Shared/routed experts, residual order, TP shards, EP ownership and collectives.
- Attention-core boundary, output projection, KV-cache updates, prefix/block state.

Do not assume two models with an MoE have the same expert activation or routing.
Do not fuse through an externally visible side effect without preserving it.
Keep the first implementation close enough to the reference to isolate errors.

Build a dependency table before assigning CTAs. For each consumer, state exactly
which producer tiles must be ready. Tile-local readiness can overlap stages;
a grid-wide barrier often destroys that overlap and needs a stronger residency
proof. A stage number in source does not imply that all CTAs finish it together.

## Resident schedules

A persistent grid assigns multiple operation roles to the same CTA IDs. The
compiler generally allocates resources for the whole compiled function, so one
register-heavy branch can reduce residency for all roles. Record registers,
spills, LDS bytes, warp count, and occupancy for every shape specialization.

For a schedule that requires all CTAs resident:

    grid_ctas <= active_ctas_per_cu(compiled_kernel) * available_cus

This is a necessary capacity check, not a complete deadlock proof. Also reason
about scheduling guarantees, resource contention, barrier participation, and
producer dependencies. Cooperative launch requirements differ by runtime;
ordinary launches need their own justified schedule. No-spill and occupancy
checks must be repeated after compiler or tuning changes.

Use free CTA ranges for independent work only after proving they are free in
that phase. Overlap output zeroing with dense projections when the output is not
read/written until a subsequent same-stream kernel. Do not race zeroing with
atomic accumulation in the same unordered phase.

## When a hybrid is the correct architecture

A resident grid can save launches but serialize too much expert weight traffic.
Batch growth increases routed rows, distinct experts and padded work while the
resident CTA count stays fixed. A baseline expert implementation can launch a
much larger grid and use different resources/quantization. Keeping its two GEMMs
can be faster than fusing them into the resident kernel.

A useful hybrid boundary is:

    fused dense/front + routing + metadata + output zero
        -> existing MoE1 (including input quant/activation as supported)
        -> existing MoE2 (including output accumulation as supported)
        -> fused reduction/norm/projections/residual tail

The front/tail may use different grids if their dependencies and buffer layouts
allow it; do not force a shared grid just for symmetry. Same-stream boundaries
order front metadata/zeroing before MoE and MoE output before tail. Cross-rank
reductions inside the tail still need IPC synchronization.

Consume the exact tensor returned by MoE2. A copy into a preallocated scratch
buffer is often removable by passing its pointer to the tail. Preserve ownership
through graph replay and allocator reuse. A pointer valid for the first eager
launch is insufficient. Do not add `.contiguous()`, `.to()`, memset, or a
standalone quantizer without accounting for its device launch and bytes.

## Library ABI is part of the kernel

Read the selected library implementation, not only its public function signature.
Pin or reject incompatible tuning rows. Specify:

| ABI field | Questions to resolve |
|---|---|
| Token IDs | Packed sample/route bits? Padding sentinel? Signedness? |
| Expert IDs | Per-row or per-block? Sorted? Local or global IDs? |
| Counts | Valid rows, padded rows, live tokens, or multiple counters? |
| Routing weights | Sorted or original order? Applied by GEMM1 or GEMM2? |
| Auxiliary maps | Row-to-token indices, route-to-sorted reverse map? |
| Scales | Packed E8M0 format, tile order, scale stride and group size? |
| Output | Atomic accumulation requiring zero, or direct writes? |
| Intermediate | Sorted/padded rows or token/top-k order? Who allocates? |

An internal `fused_moe_2stages` call can still choose prequantization or conversion
kernels. Verify exact GEMM names, block size, inline quantization and stage count;
revalidate on library updates. Diagnostic code should fail clearly when an ABI
assumption changes rather than silently claiming the same launch count.

## Small-batch metadata without another launch

For B tokens and K distinct selected experts per token, flatten R=B*K routes.
Let e[i] be expert ID. The K3 B8 specialization computes:

    row[i] = count(j < i with e[j] == e[i])
    count[i] = count(j with e[j] == e[i])
    leader[i] = (row[i] == 0)
    block[i] = count(j with leader[j] and e[j] < e[i])
    destination[i] = block[i] * BM + row[i]

Pairwise comparisons cost O(R^2), acceptable for the measured R=128 case but not
a universal sorting algorithm. Distinctness within each token implies at most B
rows per expert. If B <= BM, each active expert needs only one padded block.
Each leader writes padding positions `count[i]..BM-1`; each real route writes
only its own valid position. The two sets are disjoint, so no clear-then-scatter
barrier is needed. Leaders write expert IDs and total padded-row count.

For K3: B=8, K=16, BM=16, 896 experts. IDs pack sample in low 24 bits and route
slot in high bits; padding token is B; padding weight is zero. `SortM` stores
sample IDs; `SortReverse` maps each original route to its sorted position.
Counts are `[num_active_experts * 16, 8]`. Allocate for the worst supported
routing pattern, not a typical expert count. Only the counted region is valid.

This construction must change for repeated experts within a token, expert masks,
EP remapping, more than BM rows/expert, token dropping, capacity limits, different
packing or padding conventions. For multiple blocks/expert compute padded counts
and an exclusive prefix of block counts, then map rows accordingly. Large R may
favor histogram/prefix sort or a retained library sorting kernel.

## Integration and source-only completion

Prepare weights, IPC handles, metadata buffers, workspace and JIT variants before
capture. Separate immutable compile configuration from per-invocation values.
Keep a small launch wrapper shared by serving and the standalone benchmark, and
validate shape/grid/ABI assumptions before dispatch where feasible.

Without GPUs, check syntax, lint, host launch ordering, option propagation, pointer
identity, metadata bounds mathematically, and rejected unsupported configurations.
Mocks must not import the GPU runtime. They cannot prove compiler lowering,
occupancy, numerical accuracy, graph safety or actual device launch count.
Preserve previously validated variants until the candidate gets that evidence.

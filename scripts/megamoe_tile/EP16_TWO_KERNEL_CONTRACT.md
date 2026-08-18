# EP16 A4W4 MegaMoE: strict two-kernel contract

## Locked semantic boundary

The candidate is the EP16 equivalent of MegaMoEV2. It is not the earlier CCO
record-fanout cascade.

```text
input on each EP rank
  x_bf16       [128, 7168]
  route_weight [128, 16] float32
  topk_id      [128, 16] int32

Stage1 -- exactly one GPU launch
  BF16 -> A4/E8M0 quant
  + MORI InterNodeV1-style hierarchical top-k dispatch
  + receive-side scoreboard direct-to-expert-tile placement
  + A4W4 GMM1
  + SiLU(gate) * up
  + A4/E8M0 requant

Stage2 -- exactly one GPU launch
  A4W4 GMM2
  + route-weight multiply
  + direct LSA FP32 atomic into source-aligned node accumulator
  + combine back to the original source rank/token

output on each source rank
  y_bf16       [128, 7168]
```

The fixed K3 shape is:

```text
tokens/rank = 128
H           = 7168
I           = 3072
E           = 896
EP          = 16 (2 nodes x 8 GPUs)
experts/rank= 56
topk        = 16
activation  = SiLU
quant       = A4W4 for input, W1, Stage1 output, and W2 input
default route = rank-balanced-hot from bench_mega_moe_v2.py
```

`E=384` is DSV4 and is rejected by this test.

## Dispatch meaning: MORI InterNodeV1 hierarchical deduplication

One input token initially exists only on its source EP rank. Stage1 follows
the MORI InterNodeV1 ownership semantics and deduplicates the token payload at
both rank and node boundaries. It does **not** send one full hidden payload per
top-k route:

```text
source token + all top-k metadata
  |
  |-- selected ranks on the source node
  |     one payload copy / selected destination rank
  |     (multiple routes on that rank share the payload)
  |
  `-- selected remote node
        one inter-node payload / token / destination node
          |
          `-- aligned proxy on the remote node
                one payload copy / selected destination rank
                target rank expands only its locally owned routes
```

Thus the cross-node payload is aggregated once per `(source token,
destination node)`.  The 4096-byte record carries the complete top-k IDs and
weights plus one u16 top-k-slot bitmap per EP rank; its popcount is that rank's
route multiplicity and its set bits recover the corresponding ID/weight pairs.
If several selected experts reside on the same rank, the target receives one
hidden payload plus several local route descriptors, not several hidden
payload copies.

The default benchmark fixture has `topk == EP == 16` and chooses exactly one
expert on every rank.  Additional correctness fixtures place two routes on
each of eight ranks, vary the number of remote tokens from 0 through 128, and
place the full 32768-route capacity on one rank.  A node with no selected
expert receives no inter-node payload.

The balanced fixture gives each destination rank:

```text
16 source ranks x 128 routes/source = 2048 route rows
```

Arbitrary top-k16 supports up to 32768 route rows on one destination rank
(1080 physical BM32 tiles including conservative per-expert tail capacity).
The receiver writes the 3584-byte activation once into a fixed 2048-row
source-indexed rank-local inbox.  Every matching route independently reserves
its final expert-tile row and stores scale, source/top-k identity and weight;
GMM1 gathers the shared activation through `tile_row_input`.

Stage2 likewise does not materialize one `rank_partial` slab per expert rank
and scan eight slabs. Its weighted GMM2 epilogue performs a direct LSA FP32
atomic into the source-aligned node accumulator. Completion counters make the
node contribution visible once all selected routes have arrived. At most one
completed node contribution per `(source token, remote node)` crosses back to
the source node, where the final BF16 output is produced.

## Required public API

The constructor names and `forward` boundary match `MegaMoEV2`; the only
accepted quant mode is `a4w4`:

```python
op = Candidate(
    rank=global_rank,
    world_size=16,
    model_dim=7168,
    inter_dim=3072,
    experts=896,
    topk=16,
    quant="a4w4",
    w1=w1, w1_scale=w1_scale,
    w2=w2, w2_scale=w2_scale,
    max_tok_per_rank=128,
    mega_scheme="hierarchical",
    swiglu_limit=0.0,
    stage1_transport="chunked",  # or explicit opt-in "sparse_wqe"
)
y = op.forward(x_bf16, route_weights, topk_ids)
```

`y` must be a local contiguous-or-viewable BF16 tensor of shape
`[run_tokens, 7168]`. The implementation must use the current local CUDA
device; `rank` is a global EP rank and can be 8--15 on the second node.
The public default remains `chunked + non-split`; selecting `sparse_wqe`
switches only Stage1 to the 256-CTA split fanout while Stage2 remains 160 CTA.

Calling `forward_prequant` is not part of the test. BF16-to-A4 conversion must
be compiled into Stage1. A standalone quant kernel fails the launch audit.

## What may happen before timing

The following are construction/prime work and do not count toward the two hot
launches:

- weight quantization and native weight-layout conversion;
- symmetric/registered arena allocation and pointer-table construction;
- CCO/QP/team creation;
- JIT and code-object load;
- zeroing a newly allocated workspace;
- one or more explicit warmup forwards.

Every subsequent `forward` must reuse preallocated double-buffered state.
Independent quant, memset, pack, unpack, route-sort, wait/progress, credit,
copy, or finalization kernels inside a hot iteration are forbidden. Those
roles must be CTAs/waves in Stage1 or Stage2.

## Suggested internal organization

The physical launch count does not imply that all CTAs do the same job.

```text
Stage1 persistent grid
  communication/planner CTAs
    quantize local BF16 tiles
    group routes by destination node and destination rank
    deduplicate hidden payload once per selected local rank
    aggregate one remote payload per token/node over multiple QPs / one flush
    reserve exactly eight receive communication roles
      role 0: one aligned cross-node RAIL stream
      role 1..7: the seven other intra-node ranks
      local-self routes bypass a communication role
    aligned proxy deduplicates to selected destination ranks
    reserve final expert rows with alloc_count
    write payload/scale/source/weight directly into expert tiles
    release-increment tile_arrived only after the complete row is visible
    after all eight roles publish EOS, seal and enqueue every partial tail tile
  compute CTAs
    claim a full tile when tile_arrived == BM
    claim a tail tile only after EOS publishes its final expected row count
    GMM1 -> SiLU -> A4 requant

Stage2 persistent grid
  compute CTAs
    claim Stage1 output tiles
    GMM2 and apply route weights
    epilogue LSA-atomic-add FP32 directly to the source-aligned node accumulator
    publish a route/tile completion only after the atomic payload is visible
  return/communication CTAs
    wait for the source token's expected direct-atomic contribution count
    aggregate one remote partial return per token/node
    publish exact source-token completion
    write local BF16 output
```

The production `sparse_wqe` implementation realizes the Stage1 organization
above as one 256-CTA kernel. Full tiles reserve contiguous 24-job batches in an
eight-shard ready queue from their unique row-32 last-arriver; the first queue
slot is the release publication for the batch. CTAs join the queue as soon as
their own quant/transport/fanout obligation is complete, so GMM1 can run before
global communication EOS. The EOS finisher publishes only padded partial tails
and then seals the final queue tail. The `mori64x2` diagnostic retains the
post-EOS static scheduler as the comparison baseline.

The two Stage1 scoreboard counters have different meanings and may never be
aliased:

```text
alloc_count   number of rows reserved in an expert/tile
tile_arrived  number of fully written, release-published rows
```

Allocation alone must not make a tile runnable. For a full tile, readiness is
`tile_arrived == BM`. For the last partial tile of an expert, readiness is
published only after all `1 cross + 7 intra` receive roles reach EOS and the
owner freezes the final `alloc_count` as the tail expectation. This EOS rule
also covers experts receiving zero or fewer than BM rows without a host-side
finalization kernel.

The epoch protocol must own payload and ticket buffers until every consumer
has acknowledged EOS. Two parity buffers are the minimum needed for a future
continuous loop. Bring-up timing still aligns ranks before every iteration.

Stage1 and Stage2 use one physical CCO registered window with two aligned
logical layouts:

```text
window base
  + 0                                      Stage1ArenaLayout
  + align(Stage1.total_bytes, 4096)        Stage2ArenaLayout
```

Stage1 writes source contribution expectations and completion generations
directly into the Stage2 region. Stage2 owns the FP32 source-aligned node
accumulator and its expected/done/ready scoreboards. The Stage2 offset is
compiled into both kernels; no host copy, rank-partial translation, pointer
table conversion, node scan, or bridge kernel exists between launches.

## Correctness oracle

The headline oracle is an unfused MORI `InterNodeV1LL` path with identical
inputs, routes, local packed weights, A4 rounding points, SiLU, and weighted
combine:

```text
BF16->A4
MORI dispatch
local route selection / expert sort
A4W4 GMM1 + SiLU + A4 requant
weighted A4W4 GMM2
MORI combine
```

This is compared only after timed loops. Required checks are:

- finite BF16 output with exact shape `[128, 7168]` on all 16 ranks;
- rank-balanced and duplicate-rank fixtures preserve every expected
  `(source, top-k slot, local expert, weight)` route;
- candidate prime versus candidate final sample (epoch reuse check);
- candidate versus MORI relative L2, reduced with MAX over ranks, `< 1e-2`;
- logits diff, max absolute error, norm ratio, and per-token relative-L2 are
  recorded for diagnosis.

Before any candidate timing, the driver also rejects an implementation unless
its two compiled launchers publish the following architecture manifest:

```text
Stage1
  dispatch                     scoreboard_direct_to_expert_tile
  receive_comm_roles           8
  cross_node_comm_roles        1
  intra_node_comm_roles        7
  allocation_counter           alloc_count
  arrival_counter              tile_arrived
  eos_tail                     true
  uses_rank_inbox              false
  uses_source_activation_inbox true
  uses_group_sort              false

Stage2
  epilogue                     direct_lsa_atomic_source_aligned_node_accumulator
  node_accumulator_dtype       fp32
  uses_rank_partial            false
  uses_node_scan               false
  uses_external_reduce_kernel  false
```

`alloc_count` and `tile_arrived` must be distinct. The manifest is a cheap
preflight guard; the multi-epoch correctness run, device error counters and
unfiltered trace remain the behavioral proof.

For debug/CI builds the operator may expose an untimed
`debug_direct_tile_snapshot()` method. After the prime epoch the harness
validates:

```text
len(comm_role_eos) == 8 and every role reached this generation
sum(alloc_count) == the fixture's expected route count (0..32768)
for each active tile: tile_arrived == alloc_count and tile_ready is published
for each partial tile: tail_tile and tail_sealed are true
inactive capacity slots have no arrived/ready/tail state
for all 128 source tokens: node_atomic_done == node_atomic_expected (0..16)
node_atomic_ready is published for every source token
protocol_error_count == 0
```

The snapshot copy/read occurs after device synchronization and outside all
warmup/timed regions. Production builds need not expose it; their in-kernel
protocol error counter and output comparison remain mandatory.

The baseline includes its standalone BF16-to-A4 cost because candidate Stage1
includes that work. A prequantized MORI number may be reported separately but
must not be used as the same-boundary headline comparison.

## Timing and rank alignment

For prime, warmup, and every measured round:

```text
CUDA synchronize
Gloo barrier                  # outside timing
HIP event start
forward = Stage1 + Stage2
HIP event end
CUDA synchronize
```

After all local samples are captured, each iteration is reduced using the
maximum of all 16 ranks. The report takes mean, p50, and p95 over the final
`--tail-iters` rank-max samples. It does not average per-rank means, and no
Gloo numerical collective runs inside a timed iteration.

The safe bring-up mode intentionally measures independent-round latency. A
barrier-once throughput mode is valid only after device-side epoch ownership,
EOS, and double-buffer reuse pass a continuous stress test.

## Automated launch audit

Use an **unfiltered** rank-0 rocprofv3 kernel trace. `KERNEL_RE` must remain
`.*`; filtering for Stage1/Stage2 would hide exactly the helper kernels the
audit is meant to catch.

Example worker arguments:

```bash
PROFILE_MODE=trace \
KERNEL_RE='.*' \
MEGAMOE_TILE_PROFILE_REGIONS=1 \
PROFILE_ROOT=/home/hzm/profiles/megamoe_ep16_two_kernel \
scripts/megamoe_tile/profile_rank0_worker.sh \
op_tests/multigpu_tests/bench_megamoe_tile_ep16_two_kernel.py \
  --paths candidate \
  --operator-factory aiter.ops.flydsl.kernels.megamoe_tile:HierarchicalMegaMoEV2 \
  --warmup 3 --iters 6 --tail-iters 5
```

Then audit all six profiled steady pairs:

```bash
python3 scripts/megamoe_tile/assert_ep16_two_kernel_trace.py \
  --trace /home/hzm/profiles/megamoe_ep16_two_kernel \
  --stage1-regex '.*megamoe_tile_ep16_stage1.*' \
  --stage2-regex '.*megamoe_tile_ep16_stage2.*' \
  --iterations 6 \
  --require-no-trailing-kernels
```

The profile-region flag pauses rocprof through construction, JIT, warmup and
post-loop correctness work. The resulting kernel trace therefore contains
only the six measured forwards, so the no-trailing check covers the final
iteration as well as every boundary between iterations. Without profiler
region control, audit a tail of at least two pairs and leave the trailing
option off; post-timing output checks legitimately launch additional kernels.

The checker requires each regex to resolve to one unique symbol and then
requires the unfiltered launch stream to be exactly:

```text
Stage1, Stage2, Stage1, Stage2, ...
```

Any extra quant, memset, rank-inbox grouping/sort, pack, send/progress, unpack,
rank-partial reduction, node scan, copy, or reduce kernel between pairs is a
hard failure. The candidate kernels therefore need stable symbol tags
containing `megamoe_tile_ep16_stage1` and `megamoe_tile_ep16_stage2`.

## Files

- Public operator:
  `aiter/ops/flydsl/kernels/megamoe_tile/mega_moe_tile_a4w4.py`
- Composite ABI: `aiter/ops/flydsl/kernels/megamoe_tile/stage1_abi.py`
  and `aiter/ops/flydsl/kernels/megamoe_tile/stage2_abi.py`
- Driver: `op_tests/multigpu_tests/bench_megamoe_tile_ep16_two_kernel.py`
- Trace assertion: `scripts/megamoe_tile/assert_ep16_two_kernel_trace.py`
- CPU trace-auditor test: `op_tests/test_megamoe_tile_two_kernel_trace.py`
- Direct-tile/atomic manifest and snapshot tests:
  `op_tests/test_megamoe_tile_direct_tile_contract.py`

No MORI source is modified by this harness.

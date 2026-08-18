# megamoeTile development status

Target: Kimi-K3 A4W4, EP16 on two 8-GPU MI355X servers. SiLU remains the
default; the external `activation` field also selects SwiGLU or SiTUv2.

The current one-host harness splits EP8 into two logical 4-GPU nodes.  The
cross-logical-node path uses the exact MORI SHMEM put+signal ABI; MORI selects
its P2P backend on this machine, while a real remote PE will select RDMA.

## Implemented milestone

- EP16 and EP8/2x4 topology, rank-aligned proxy and explicit EP-rank -> SHMEM-PE map.
- Accepted route plan, token/node de-duplication and per-node expected route count.
- Fixed symmetric dispatch/combine ring contract and copy-only workspace.
- FlyDSL copy+release-signal kernel.
- FlyDSL MORI put+signal, QP quiet and non-zero EOS/control kernels.
- Existing AITER A4W4 baseline:
  `moe_sorting -> GMM1+activation+A4 requant -> GMM2 local partial`.
- Fused H1 prototype: low-ticket copy/publish plus the real A4W4
  GMM1+activation+A4 body in one code object, specialized at compile time.
- Persistent H1 prototype: device entry ticket, epoch gate, 64-byte-separated
  sharded work heads, per-flat-tile ticket/strided schedulers, and checked
  fixed-grid workspace reuse without a host memset.
- Persistent A4W4 GMM1 core port: fixed worker grid with per-tile grid-stride
  execution, kept separate from communication control for clean MFMA codegen.
- Fused H2 prototype: low-ticket partial copy/return plus the real A4W4 GMM2
  body in one code object.
- FP32/A4 references and ROCtx ranges.

## Verified

```text
CPU/topology/reference tests                       6 passed
FlyDSL copy stub                                  payload + generation exact
MORI symmetric self put/signal/quiet/EOS           1 passed
EP8 2x4 logical-node MORI ring                     payload/ready/credit exact
local A4W4 SiLU baseline                           passed A4 reference
local A4W4 SiLU/SwiGLU/SiTUv2 baseline             passed A4 references
EP8 all-activation local-expert partial combine    logits_diff <= 1.393e-6
fused H1 all activation payload/scale              bitwise vs matching port
persistent H1 ticket/strided, all activations      bitwise vs matching port
persistent workspace consecutive epochs             no host memset
fused H2 local combine                             matches unfused GMM2
```

## Persistent H1 result

The new persistent scheduler follows MegaMoE Stage1's control structure:

```text
entry_count → owner/comm ticket → device epoch gate
            → 8 sharded work heads → one flat GEMM tile per claim
            → communication ticket rejoins the common compute loop
```

The public checked launcher binds `PersistentH1Workspace` to one worker-grid,
work-shard count and stream. The current copy stub is restricted to one
communication ticket, drains global stores with `s_waitcnt(0)`, publishes a
system-release signal and rejoins after a CTA barrier.

Target-shape warm medians (`H3584/I384/E56/topk16`, tokens=8, 162 tiles,
192 workers) are:

| Mode | Median |
|---|---:|
| Standalone compute | 13.989 us |
| Serial copy + compute | 50.064 us |
| Old static rejoin H1 | 67.256 us |
| Dynamic persistent ticket H1 | 76.858 us |
| Dynamic persistent strided H1 | 77.026 us |
| Persistent core compute | 14.106 us |
| Sidecar | 50.511 us |
| Persistent-core sidecar | 50.804 us |

Therefore the single-kernel dynamic persistent variant does **not** provide a
benefit for this decode shape: it is 53.5% slower than serial. ATT shows the
control path raises the kernel to 168 VGPR / 81 SGPR and introduces five VGPR
spills plus 24 bytes of private memory. By contrast, the isolated persistent
GMM core is spill-free and retains the original ~14 us compute time. Ticket and
strided results are almost identical, so the work-head atomic is not the main
problem; the combined control-flow live range/code generation is.

The current recommended implementation is therefore the persistent GMM core
plus a communication sidecar while the A4 body is ported into MegaMoE's native
work-loop implementation. Full measurements are recorded in
`gfx950_persistent_h1_validation.json`.

This result still uses an independent copy buffer. It is not yet the real
`internode_v1` data path: MORI NBI TX/RX progress, credit/EOS, plan-ready and
per-M-tile ready/expected wait-acquire must be connected before claiming real
cross-server communication/compute overlap.

## Activation contract

Set `activation` on `HierMegaMoETileConfig`, or pass it directly to
`run_local_ep_a4w4` and `compile_hier_stage1_a4w4`:

```python
cfg = HierMegaMoETileConfig.production_ep16(
    rank=rank,
    activation="situv2",
    situ_beta=4.0,
    situ_linear_beta=25.0,
)
```

Canonical values are `silu`, `swiglu`, and `situv2`; matching is
case-insensitive and `situ` is accepted as the K3 model-config alias.
`siluv2` is intentionally rejected. Their exact epilogues are:

```text
silu:    silu(gate) * up
swiglu:  clamp_gate * sigmoid(1.702 * clamp_gate) * (clamp_up + 1)
          default clamp limit = 7
situv2:  [beta*tanh(gate/beta)*sigmoid(gate)]
          * [linear_beta*tanh(up/linear_beta)]
          K3 defaults = beta 4, linear_beta 25
```

Activation and its effective parameters are included in the kernel symbol and
JIT cache key. There is therefore no runtime activation branch around MFMA.
The old `run_local_ep_a4w4_silu` and `compile_hier_stage1_a4w4_silu` entry
points remain strict SiLU compatibility wrappers.

This support is wired through the explicit `megamoeTile` baseline/H1 paths.
It does not claim that every tuned top-level `fused_moe`/CSV route now carries
the new activation parameters. In particular, the legacy `compute.a4w4_local`
guard rejects SwiGLU because top-level decode dispatch may select A16W4; use
`run_local_ep_a4w4` to retain the explicit A4W4 contract.

The complete accuracy, ATT resource and target-shape copy-stub measurements
are recorded in `gfx950_activation_validation.json`. In the ATT smoke shape,
all three variants use 130 VGPR, 46 SGPR and 32 KiB LDS with no scratch or
spill. At the target local geometry (`H=3584, I=384, E=56, topk=16`), all
three use exactly 132 VGPR, 49 SGPR and 32 KiB LDS with no scratch or spill,
matching the old SiLU baseline. The current rejoin H1 still takes about 66--67
us versus 50--52 us for separate communication plus compute, so its measured
hidden fraction is zero for all three activations. The two-stream sidecar hides
roughly 45--51% of the shorter phase. This confirms activation selection did
not create the overlap problem; the remaining issue is the dynamic
communication role/scheduler structure.

## Current resource profile (gfx950, token=8 geometry)

| Role/kernel | VGPR | SGPR | LDS | Spill/scratch |
|---|---:|---:|---:|---:|
| copy put+signal | 8 | 22 | 0 | 0 |
| A4W4 GMM1+SiLU+A4 requant | 134 | 54 | 41,088 B | 0 |
| A4W4 GMM2 atomic | 54 | 46 | 8,192 B | 0 |
| fused H1 copy + A4W4 GMM1/SiLU/A4 | 132 | 49 | 32,768 B | 0 |
| fused H2 copy + A4W4 GMM2 | 52 | 54 | 8,192 B | 0 |

With 160 KiB LDS/CU and four waves/workgroup, the fused H1 improves the LDS
limit from three to five workgroups/CU; wave slots permit up to eight H2
workgroups/CU.  Resource
counts show co-residency is possible, but they are not an overlap measurement.

The standalone 64 KiB copy body is about 4 us in the rocprof kernel trace, while
one-copy end-to-end launch observation is about 40 us.  The fixed launch gap is
the reason the copy/RDMA progress role must be inlined into persistent H1/H2,
not emitted as one kernel per chunk.

Initial overlap experiment at token=8/local-E=56:

```text
comm-only                  43.970 us
GMM1-only                  14.232 us
separate-kernel serial     52.997 us
naive dynamic-role H1      68.613 us   (regression; do not use)
two-stream sidecar         49.554 us
sidecar hidden fraction    60.77%
sidecar speedup vs serial   6.95%
```

The dynamic role branch harms AMDGPU scheduling even when its comm body is
compiled empty.  Production H1 should therefore reuse MegaMoE's existing
ticket/work-queue structure (comm roles later join a common GMM work loop), or
use a persistent sidecar until that integration is complete.  Do not ship the
naive outer `if comm else GEMM` prototype.

## Next device-kernel step

1. Port the proven H1/H2 compute bodies into MegaMoE's ticket/work-queue
   scaffold so role selection does not wrap the MFMA body in a dynamic branch.
2. Replace the raw-copy role with the already compiled MORI
   put+signal adapter while preserving the shared-resource profile.
3. Emit local expert IDs from dispatch and publish per-tile ready after payload,
   scale, source map and routing weight are visible.
4. Change the current H2 atomic local-combine epilogue into route-ready enqueue,
   form one BF16 node partial/token, then return it through the same ring.
5. Re-run code-object statistics.  H1 must remain spill-free and preserve at
   least one communication plus one compute workgroup worth of residency.
6. Measure comm-only, compute-only, serial and overlap with identical traffic;
   report actual hidden fraction, not a value inferred from register counts.

The K3 CSV used for initial tile geometry is tagged SiTUv2.  This operator uses
the user's requested SiLU semantics; a dedicated EP16/local-E=56 SiLU tuning
table is required before treating current geometry as performance-optimal.

## Private CCO transport contract

`aiter.ops.flydsl.kernels.megamoe_tile.cco` is an AITER-local scalar FlyDSL bridge over MORI's
public `ccoGda` device API. It does not require a MORI source change. Data,
ready and credit words must be subregions of a CCO-allocated and registered
window; pass `RegisteredWindow.handle` to device ops, never a tensor
`data_ptr()` or `RegisteredWindow.local_ptr` in place of the handle.

The collective host initialization order is:

```text
Communicator.init
  -> alloc_mem/register_window for every arena (same order and size on all ranks)
  -> CCODevCommRequirements
  -> create_dev_comm
  -> barrier
  -> kernels
```

MegaMoETile stores ready/credit generations in its registered arena, so its
DevComm explicitly requests no unused resource pools:

```python
reqs.gda_signal_count = 0
reqs.gda_counter_count = 0
reqs.lsa_barrier_count = 0
reqs.rail_gda_barrier_count = 0
reqs.barrier_count = 0
```

`TEAM_WORLD` takes a world rank. `TEAM_RAIL` uses `CCO_TEAM_GDA` and therefore
takes a node index; it is valid only with `GDA_CONNECTION_RAIL`. Each QP has one
owner wave. Payload puts and the trailing absolute-generation `put_value` use
the same QP and `AggregateRequests`, followed by one `flush_async`; the remote
side polls the generation with the bridge's system-scope atomic-acquire helper.
Credits must be observed before an arena slot is reused.

The two-node smoke is
`op_tests/multigpu_tests/test_megamoe_tile_cco_transport.py`. It runs both
directions for two epochs and validates payload, ready and credit generations.

### Stage-1 CCO sidecar skeleton

`cco/stage1_sidecar.py` is a separate communication code object; no CCO branch
is present in the MFMA kernel. A ring slot is a bounded set of fixed segments.
For the legacy H3584 dispatch wire format, the full-chunk geometry is:

```text
segment_bytes = 2048
num_qp = 4
batch_per_qp = 8
total = 4 * 8 * 2048 = 64 KiB ≈ one BM32 M-tile
```

The generic H7168/topk16 format has a 3952-byte raw record and a 4096-byte
aligned stride.  The fused EP16 Stage1 ABI extends its otherwise-unused tail:
bytes 3952..3983 hold sixteen packed u16 rank-slot masks, making its raw extent
3984 bytes without changing the 4096-byte transport stride.  Their popcounts
encode per-rank route multiplicity while set bits select the existing top-k
ID/weight entries.  The full-chunk geometry remains
`4 QP * 4 records * 4096 B`.

Each QP has one owner wave. It appends its payload writes and one trailing ready
generation with `AggregateRequests`, calls `flush_async`, stores the opaque
request in `dispatch_request[slot, qp]`, and returns without waiting. Ring-slot
reclaim first waits remote credit and then waits the retained request, so source
storage remains live for the complete NIC DMA lifetime.

The default H1 readiness path separates plan and payload publication:

```text
publish_plan_expected(active_m_tiles, expected_per_tile)
mark_chunk_ready(slot, first_m_tile, tile_count, delta)
```

The 64 KiB chunk therefore increments only the M-tile range it actually makes
visible. The experimental spill-avoidance path writes flat GMM tile IDs into
`h1_ready_queue` and release-publishes `h1_queue_header.tail`; its header is
`[epoch, total_work, tail, done_generation]`. Queue publication remains an A/B
option because the WPE2 direct-ready H1 is currently spill-free.

The sidecar currently assumes the dispatch ring has already been packed and
the final expert-major activation/scales plus metadata have already been
materialized. Token packing, destination count/plan, expert sort, proxy unpack
and intra-node fan-out are still missing and must precede `mark_chunk_ready` or
queue append. Credit must not be returned until that unpack/fan-out step has
stopped reading the ring slot.

### Stage-2 CCO return sidecar skeleton

`cco/stage2_sidecar.py` applies the same deferred-request lifecycle to the
`partial_tx/partial_rx` ring. Its default BF16 node-partial wire record is
7424 bytes:

```text
4 QP * 2 records/QP * 7424 B = 59,392 B <= 64 KiB
```

Each QP appends its bounded records and a trailing `partial_ready` generation,
then stores the `flush_async` token in `partial_request[slot, qp]`. The source
waits that request only after `partial_credit` permits slot reuse. The receive
publisher may mark a bounded contiguous source-token range in
`node_partial_ready`, but assumes records are already consumer-visible. GMM2,
record construction/unpack and final source combine remain outside this
sidecar and are intentionally not duplicated in the communication code object.

`kernels/partial_record.py` supplies the independent pack/unpack seam. A record
is `BF16[H] + u32 source token + zero padding` aligned to 256 bytes. H=1024 uses
2304 bytes; H=3584 uses 7424 bytes. Pack selects arbitrary node-partial rows
through an int32 source-id array. Unpack returns contiguous BF16 rows and source
IDs, validates source range and padding, and zeroes an invalid row instead of
indexing out of bounds. Both kernels use raw CCO-arena pointers and
system-release their writes. Transport ready/acquire must precede unpack, and
unpack must finish before `node_partial_ready` publication.

### Node-local CCO-LSA dispatch fan-out

`kernels/dispatch_fanout_lsa.py` copies aligned records from a proxy's
local dispatch buffer into arbitrary local ranks through
`cco.lsa_ptr(registered_window.handle, dest_lsa_rank, offset)`. The host plan
preassigns `dest_lsa_rank`, `dest_slot` and `valid` for every input record. The
kernel performs no device allocation: after the peer record stores drain, it
release-publishes `fanout_ready[parity, dest_slot] = generation` and increments
the peer's `fanout_count[parity]`.

The bounded inbox capacity is `max_fanout_records` (defaulting to
`max_source_tokens`). One plan entry maps one input record; routing one token
to multiple destination ranks requires
the upstream plan to expand/duplicate entries. Dynamic slot allocation, compact
multi-destination expansion and EOS finalization remain future work. The
rank-aligned path bypasses this fan-out kernel entirely.

The compute-side expert-route reduction is now a separate
`compile_node_partial_reduce` kernel. Its input slab is
`[node_ep_ranks, source_capacity, hidden]` BF16. Missing rank slots must be
zero-filled; `rank_route_expected/ready[source]` is an arrival count, not a
rank mask. One CTA owns one source token, waits for that count, accumulates up
to eight rank slots in FP32, writes either BF16 or FP32 node output, and then
release-publishes `node_partial_ready[source] = generation`. Keeping one CTA
per source avoids another done-counter array between hidden-dimension tiles.

`compile_node_partial_reduce_lsa` is the zero-copy form of the same contract.
It takes a CCO `RegisteredWindow.handle` plus the common partial byte offset
and resolves all peer VAs with `cco.lsa_ptr`/`ccoGetLsaPeerPtr`. It does not
copy into a temporary rank slab and does not duplicate MORI's private LSA
stride calculation. Every EP rank must register the same-size window in the
same collective order and place its BF16 source rows at the agreed offset.

The batched rank-partial bring-up path uses
`compile_rank_partial_epoch_gate_lsa` immediately after its same-stream
rank-partial scatter. The one-CTA gate release-publishes the local absolute
generation through the parity-selected `partial_eos` word before one lane per
LSA rank waits for every peer generation. The following same-stream LSA
reducer therefore needs no host device synchronize or Gloo barrier. This is an
epoch gate, not yet per-source overlap; finer-grained readiness remains a
separate optimization.

This reducer covers only node-local EP partials. TP reduction is deliberately
not performed here: a TP communicator must consume the published node partial
in a later pipeline stage before final output publication.

After the local node partial and the returned remote-node partial are both
available, `compile_final_combine` waits on their independent absolute-ready
generations, converts either BF16 or FP32 inputs to FP32, adds them, and stores
BF16 or FP32 output. One CTA owns each source row and release-publishes
`final_output_ready[source]` only after the complete hidden row is visible.
This is still an EP combine result: when TP is enabled, the consumer must run a
separate TP all-reduce before treating it as the model-layer output.

## Strict EP16 two-kernel operator

`MegaMoETileA4W4` in `mega_moe_tile_a4w4.py` is the replacement path for the
K3 `E896/EP16/H7168/I3072/topk16/tokens128` target. Its public constructor and
`forward(x_bf16, route_weights, topk_ids)` match MegaMoEV2, with
`quant="a4w4"` only. Construction collectively creates one CCO registered
window containing aligned Stage1 and Stage2 logical arenas, creates the RAIL
device communicator, clears workspace, and compiles both kernels.

The production FlyDSL implementations live in the standard AITER kernel tree:

```text
aiter/ops/flydsl/kernels/megamoe_tile/stage1.py
aiter/ops/flydsl/kernels/megamoe_tile/stage2.py
aiter/ops/flydsl/kernels/megamoe_tile/gemm1.py
aiter/ops/flydsl/kernels/megamoe_tile/gemm2.py
aiter/ops/flydsl/kernels/megamoe_tile/gemm_common.py
aiter/ops/flydsl/kernels/megamoe_tile/comm_ops.py
```

The operator is exposed only through this canonical package; the former
legacy tree and temporary flat Stage1/Stage2 modules have been removed.
Node-local payload and metadata traffic uses MORI's official
`Window.lsa_ptr` followed by FlyDSL `buffer_load`/`buffer_store`.  RAIL
payload/terminal/credit traffic still uses the minimal package-local async
bridge because the current public MORI FlyDSL binding does not expose the
required RAIL-team runtime-QP,
`AggregateRequests`-without-doorbell, or `flushAsync` request/wait semantics.

MegaMoE Tile owns private Stage1/Stage2 GMM implementations in this package.
It does not modify the shared `mxfp4_gemm1.py`, `mxfp4_gemm2.py`, or
`mxfp4_gemm_common.py` used by other AITER operators.  Its additional
system-acquire polling and acquire-release last-arriver primitives likewise
live in package-local `comm_ops.py`; shared `communication_ops_utils.py`
remains unchanged.

The public default remains `stage1_transport="chunked"` with the non-split
legacy fanout.  `stage1_transport="sparse_wqe"` is an explicit opt-in that
selects the 256-CTA split fanout, per-rank slot masks, multi-CTA aggregate WQE
posting, one terminal doorbell per QP, and the tile-ready GMM1 pipeline;
Stage2 remains at 160 CTAs.  The CCO wave handles QPs serially, but each QP is
scanned, terminal-published and flushed immediately instead of waiting to scan
all four first.  An inter-node fanout CTA acquires only the terminal for its
own QP and can therefore place remote routes while later QPs are still being
prepared.  The last of 32 route writers release-publishes all 24 GMM1 N-block
jobs for a full expert tile.  Completed intra-node fanout CTAs consume those
early jobs; inter-node fanout CTAs wait for the four-QP batch gate before
joining the queue, which prevents early compute from starving transport.
Partial expert tails are padded and published only after all eight
communication EOS signals have been acquired.

The hot forward makes exactly two launcher calls. Stage1 includes BF16-to-A4,
InterNodeV1 node/rank-deduplicated transport, receive-side scoreboard
direct-to-expert-tile placement, GMM1, SiLU and A4 requant. Stage2 uses a
weighted GMM2 epilogue to LSA-atomic-add FP32 directly into the source-aligned
node accumulator, followed by one partial return per token/node and final
combine. Stage1 has no full-record rank inbox or group-sort pass; it only keeps
one source-indexed quantized-activation row per `(source token, destination
rank)` so duplicate expert routes can share the payload. Stage2 has no
rank-partial slab or eight-rank scan. It never falls back to the multi-kernel
pack/fan-out/sidecar cascade described above. See
`scripts/megamoe_tile/EP16_TWO_KERNEL_CONTRACT.md` for the trace and benchmark
acceptance contract.

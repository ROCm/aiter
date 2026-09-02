# Fused Stage2 tile-overlap WIP handoff (2026-09-01)

## Resume identity

- Codex session: `01a041cb-107c-7612-a6df-08eb5323c56a`
- Repository: `/home/zihuang/work/aiter-mega-tile-pr`
- Branch: `dev/stage2_compute_pipeline_v1`
- Base commit: `a69a16420`
- Remote containers: `mi355-gpu-{46,50}:hzm_work:/home/hzm/aiter`
- Fixed case: TPR128/512, TopK16, E896, H7168, I3072, EP16, A4W4.
- Do not apply experimental stashes wholesale. Preserve the dirty worktree and
  `STAGE2_CU_OPT_CHECKPOINT_20260901.md`.

## Objective and ordering

The requested optimization is finer-grained overlap inside the single fused
Stage2 kernel:

```text
N-major GMM scheduling
-> rank-local tile/group completion
-> source-proxy peer-pull tile reduction
-> tile-mask RAIL batches (4/8/12 tiles)
-> tile-level final add/unpermute
-> optional node-reduced FP8 blockwise RAIL payload
```

Quantization must remain a compile-time ops option:

```text
stage2_rail_quant_type = none | fp8_blockwise
```

The unquantized tile pipeline must be validated and benchmarked first. Kernel
branches must use `const_expr`; `none` remains the default.

## Completed foundations

### Token/capacity generalization

- `max_tokens` is layout-derived for Stage1/Stage2, range 1..4096.
- Stage1 uses a bounded 128-CTA producer pool and a runtime strided token loop.
- Source encode/decode no longer assumes `>>7` / `&127`.
- Large Stage1 CCO and Stage2 return loops were changed from capacity-sized
  constexpr expansion to runtime loops.
- `max_routes_per_token_per_rank` is an explicit capacity contract; cap2 is
  required by the current paired fixture.
- TPR512/cap2 combined arena is 461,037,568 B and registers successfully.

### Baseline correctness fix

`rank_pending` initialization previously used `tx/THREADS` outside a bx0
guard, so every CTA repeated every route increment. It now uses the correct
global-grid partition plus clear-before-increment and increment-before-use
grid barriers. Paired and arbitrary four-generation validation passed.

### P1 N-major swizzle

Compile-time options:

```text
gmm_work_swizzle = token_major | n_major_window
window_n_groups  = 1 | 2 | 4 | 7 | 14
```

TPR512 `gmm2_only` selected W2. Full fused performance:

```text
token-major rank-max mean  4.303 ms
N-major W2 rank-max mean   3.916 ms  (-8.98%)
token-major all-rank mean  3.950 ms
N-major W2 all-rank mean   3.769 ms  (-4.58%)
```

W2 is opt-in and is the scheduling base for P2.

## P2 implementation currently in the worktree

Compile-time option:

```text
ready_granularity = token | tile
```

Tile mode has append-only ABI regions:

- `rank_tile_pending`, `rank_tile_ready`
- `node_tile_arrived`, `node_tile_ready`, `node_ready_mask`
- `tile_reduce_queue`, `tile_reduce_queue_ready`
- `tile_reduce_queue_head`, `tile_reduce_queue_tail`

The first implementation used one queue task per BN256 tile (3584 tasks at
TPR128) and was functionally correct but slow. It measured about 3.1 ms at
TPR128 because metadata atomics, queue claims and per-tile peer setup dominated.

Formal rejected measurements for the per-tile-task family are:

```text
TPR128 original per-tile task:        3150.698 us rank-max mean
TPR128 wave-independent per-tile:     3091.406 us rank-max mean
TPR512 wave-independent per-tile:    10092.620 us rank-max mean
```

Wave independence did not recover the metadata/queue/peer-setup overhead, and
the TPR512 scaling is unacceptable. Do not restore these variants as a
performance candidate.

The active implementation is the lower-overhead token-owner design:

1. Tile mode initializes `rank_pending[source]` as route count (one increment
   per route), not route_count times 14/28.
2. `rank_tile_pending[source,n_group]` is used as an arrival counter. The last
   route for a group publishes its two BN256 `rank_tile_ready` generations.
3. Only group 0 performs node arrival and enqueues one token owner task.
4. A reducer CTA claims the token once. Its four waves retain the efficient
   full-row reducer loop, but each wave waits for the relevant peer
   `rank_tile_ready[source,tile]` before pulling that tile.
5. BF16 token-level compact RAIL is still used after the reducer finishes the
   row. P3 tile RAIL/final has not been implemented.

The obsolete per-tile wave reducer helper remains in `stage2.py` but is no
longer selected by the active tile branch; remove it only after the token-owner
path is fully validated.

## Verified P2 checkpoints

Earlier per-tile-task implementation:

```text
producer-only: tile_pending_nonzero=0
producer-only: tile_reduce_queue_tail=3584
producer-only: tile_node_arrived_nonzero=3584
reducer-only:  node_ready_mask_full_count=128
reducer-only:  partial_ready planes=[64,64]
reducer-only:  rank_return_counts=[64,64,128]
errors=0
```

Latest token-owner implementation:

- TPR128 full smoke: both nodes exit 0.
- Latest exact cold compile: 14.146 s, LDS 28,992 B, below the 180 s gate.
- The most recent performance run `tilep2_groupcounter_tpr128_w2_perf` was the
  token-owner/group-counter revision before final cleanup and measured:

```text
rank-max mean 3.106 ms
P50           3.071 ms
P95           3.572 ms
all-rank mean 3.006 ms
```

This is still a regression versus the token baseline. Do not promote P2 yet.
The latest source after this measurement includes the token-owner reducer
selection and has passed smoke/cold compile, but needs a fresh performance run
after environment recovery.

The current group-arrival/token-owner diagnostic semantics are:

- `rank_pending[source]` equals the route count;
- `rank_tile_pending[source,n_group]` is an arrival counter and must equal the
  route count when a group completes;
- last group arrival publishes generation readiness for its two tiles;
- only group 0 performs node arrival and enqueues one token-owner task;
- the queue contains 128 token tasks at TPR128, not 3584 tile tasks.

Review/compile guards now require tile mode to use `n_tile_group == 2`, require
`hidden_tiles % 4 == 0`, and report explicit errors on arrival-counter overflow.
Node50 contract tests pass (`113 passed`).

## P3 design, not implemented

- Ready/mask granularity matches one GMM N-group work, not one BN tile:
  `hidden_tiles=ceil(H/BN)`, `ready_group_tiles=n_tile_group`, and
  `ready_group_count=ceil(hidden_tiles/ready_group_tiles)`.
- For H7168/BN256/n_tile_group2 this is 28 tiles but only 14 uint64 mask bits;
  one bit covers two consecutive tiles / 512 hidden elements.
- Reducer publishes each completed group bit after both valid tile payload
  stores, waitcnt and system release.
- RAIL CTA claims `ready_mask & ~sent_mask`, appends one PUT per contiguous run
  and uses one flush/doorbell for batches of 4/8/12 groups (8/16/24 tiles in
  the current n_tile_group2 configuration).
- A tail batch must flush when all producers complete even if fewer than the
  configured batch size remain.
- Receiver publishes `(token,tile_mask)` only after all payload PUTs complete.
- Final CTAs consume tile tasks and write each BN256 output range without
  waiting for the whole token.
- Node-local tiles bypass RAIL and use the same local-ready mask.

For a tail group, only `min(ready_group_tiles, hidden_tiles - group_start)`
tiles are valid. Producers/reducers/RAIL/final must mask the missing tile and
the final partial tile's elements when H is not divisible by BN. Bits at or
above `ready_group_count` must remain zero and cannot satisfy completion.

## P4 blockwise FP8 design, plumbing only

Host/ABI supports `none|fp8_blockwise`; kernel codec is not implemented.
FP8 mode has independent append-only payload/scale regions, not overlapping
the BF16 TX/RX buffers:

```text
rail_fp8_tx_payload / rail_fp8_rx_payload: uint8 [P,T,H]
rail_fp8_tx_scale   / rail_fp8_rx_scale:   fp32 [P,T,H/128]
```

Use E4M3, block size 128, FP32 scale. Quantize only after node reduction;
dequantize in final. `const_expr(rail_quant_type == "fp8_blockwise")` must
compile it as a separate artifact. Do not enable until BF16 P3 wins.

## Performance references

TPR512:

```text
fused token-major rank-max mean       4.303 ms
fused N-major W2 rank-max mean        3.916 ms
MORI BF16 GMM2+combine                5.607 ms
MORI FP8 direct-cast GMM2+combine     3.000 ms
```

MORI FP8 direct-cast is only a performance reference; its full numerical
error comparison has not been signed off.

## Resume steps

1. Confirm node46/50 are idle and `podman exec hzm_work` works. Node46 may need
   `/tmp/crun-no-new-keyring -> /usr/bin/crun` restored after `/tmp` cleanup.
2. Sync the full tracked diff to both nodes and compare checksums. The most
   recent local diagnostic additions may be newer than remote copies.
3. Run local `py_compile`, `bash -n`, `git diff --check`, then container
   contracts.
4. Re-run TPR128 tile token-owner full smoke and formal perf. If still above
   the token baseline, do not proceed to P3; profile/measure metadata and
   per-tile wait overhead or reduce granularity to grouped tiles.
5. Only after P2 is non-regressing implement P3 and test overall fused Stage2
   at TPR64/128/256/512. Do not substitute isolated GMM timings for the final
   requested result.

## Latest compile gate

```text
TPR128 tile-ready full cold compile: 14.146 s
LDS: 28,992 B
status: pass
```

## 2026-09-02 environment blocker

The latest exact group-arrival/token-owner formal performance run is waiting
for node46. Node46 is occupied by an external V4-Pro TP8 server plus serving
benchmark; these processes are unrelated to this work and must not be killed.
Node50 contracts have passed, but do not run a one-node substitute for the
required EP16 measurement. Resume the two-node smoke/perf only after node46 is
released and both worktrees' hashes match.

## Exact P2 formal gate: rejected

The current group-arrival + token-owner source was measured against a matched
W2 token-ready baseline at TPR128:

| Path | Rank-max mean | P50 | P95 | All-rank mean |
|---|---:|---:|---:|---:|
| P2 group-arrival/token-owner | 2842.748 us | 2795.720 us | 3117.879 us | 2758.828 us |
| matched W2 token baseline | 1666.154 us | 1645.591 us | 1781.581 us | 1566.334 us |

P2 regresses rank-max mean by 70.6% and fails its promotion gate. The remaining
dominant structural overhead is the token-owner reducer performing 28 tile
iterations per token, with per-tile peer-ready acquire/polling across peers.
Reducing queue tasks from 3584 to 128 was insufficient because the wait/setup
cost remained proportional to token x tile x peer.

Keep this P2 path opt-in for diagnostics and correctness reference, but do not
make it default and do not implement P3 mask RAIL on top of it.

## Next direction: rank-group watermark

Exploit N-major W2 ordering at rank/group granularity:

1. For each local rank and N-group, count completion of all M work owned by
   that rank. Publish exactly one generation-tagged group watermark when the
   complete group is available.
2. A reducer claims an N-group, waits once for the required eight peer-rank
   group watermarks, then batch-reduces that group's two BN256 tiles across all
   active tokens.
3. Remove per-route tile arrival, per-token tile queueing, and
   token x tile x peer polling from the active path. Preserve group overlap:
   group g reduction may proceed while GMM computes later N-groups.
4. Keep the existing token-ready P2 as an opt-in reference until the watermark
   path independently passes producer/reducer diagnostics.

Required invariants:

- the rank/group completion count equals the exact shape-derived number of M
  work items; one owner publishes one watermark per generation;
- N-major tail windows/groups publish correctly when work is not divisible by
  the window size;
- a reducer never reads any token/tile payload before every participating peer
  rank's group release watermark;
- active-token membership is stable/available to the batch reducer and empty
  routes do not block watermark completion;
- generation/parity reuse cannot accept a stale group watermark;
- reducer group claims are unique and batch output covers every active token
  exactly once for both tiles;
- compile <=180 s, TPR128 numerical correctness, and same-run full A/B versus
  W2 token baseline are required before revisiting tile-mask RAIL/final.

## Alternative reducer experiment: peer2_stage_last

Evaluate an opt-in node-reducer work schedule:

```text
node_reduce_work_schedule = peer2_stage_last
16 reducer CTAs = 8 peer ranks x 2 CTAs/peer
```

Each peer owns an independent task head. The two CTAs assigned to a peer split
that peer's token/tile work; four waves per CTA may claim/copy independent
tasks. Remote payload is not atomically accumulated into the final node
partial. Instead each peer copy writes a unique proxy-local BF16 staging row:

```text
peer_stage[peer, parity/generation, token, tile, BN256]
```

After a peer copy completes it release-increments/publishes an arrival counter.
The wave observing the last required peer arrival acquires all preceding
staging writes, loads the eight peer-local rows in fixed peer order, performs
the FP32 reduction locally, writes the node partial, then publishes the tile
ready mask. Fixed peer reduction order preserves the numerical ordering of the
current reducer.

Required protocol checks:

- each peer/token/tile staging row has exactly one writer per generation;
- peer task heads cannot duplicate or skip work when two CTAs/four waves claim;
- payload store and waitcnt complete before release arrival publication;
- exactly one last-arrival winner performs the eight-row reduction and mask
  publication;
- generation/parity reuse cannot expose stale staging rows or counters;
- absent peers are represented in the expected mask/count and never waited on;
- local/remote peer identity and staging offsets are shape-derived;
- reducer completion, node partial, and ready mask cover every active tile once.

Performance cost model must remain explicit: every contribution adds one local
BF16 staging write and the winner performs eight local staging reads. The
existing `load_first` reducer can already overlap eight LSA peer loads, so this
scheme only wins if distributing remote pulls across peer-owned CTAs and hiding
peer-ready skew outweighs extra local traffic, counters, and last-arrival
coordination. Do not assume a win.

Implementation sequence:

1. Add the compile-time schedule name without changing the default.
2. Implement a minimal producer/copy/staging diagnostic and inspect exact
   arrivals, unique writers, and final local reduction.
3. Enforce <=180-second cold compile and TPR128 paired/arbitrary numerical
   correctness.
4. Same-run full A/B against both rank-group watermark and the W2 token-ready
   baseline. Reject if it does not provide a stable end-to-end improvement.

## Authoritative ready-flag granularity correction

All P2/P3 readiness, arrival, queue, and mask references should be interpreted
at GMM N-group granularity:

```text
hidden_tiles       = ceil(H / BN)
ready_group_tiles  = n_tile_group
ready_group_count  = ceil(hidden_tiles / ready_group_tiles)
```

The current shape gives `28 / 2 = 14` ready groups. Never allocate, publish,
poll, or transmit 28 independent flags for this configuration. A group bit is
generation-tagged and becomes ready only when both valid constituent tiles are
complete. Tail groups may contain fewer than two valid tiles, and the final
tile may contain fewer than BN valid elements; absent tiles/elements require
bounds masks and do not create extra arrivals. RAIL batch sizes are group
counts unless a result explicitly labels a different unit.

## Latest local group-flag implementation boundary

The group-size correction is implemented locally. The kernel derives
`ready_groups=ceil(hidden_tiles/n_tile_group)` (14 for H7168/BN256/group2).
Initialization, publication, and reducer wait indexing all use the group index;
one completed group publishes one generation flag, not two tile flags. ABI,
host plumbing, and tests were updated. Node50 contracts pass (`61 passed`).

Authoritative local hashes:

```text
stage2.py      c0ed2cf9bdb3871301e0ef9d5e33694e5fa619e855762fb8b81ea0c9c37d835d
stage2_abi.py  5c0d077f3fe6b6b44b8086c44bd0b155d3ee915640270b172fcc0319c47e5375
tracked diff   9137bd8158c63c8387f0eccb5250b11ba9fcfb42a78ee8c32d9c4f283a042777
```

Node46 remains occupied by an external service, so this exact group-flag
source has not completed two-node smoke/performance. After release, synchronize
the complete tracked diff, verify hashes, cold-compile the exact artifact
(<=180 s), then run narrow producer/reducer diagnostics and full smoke before
formal performance. Earlier compile/smoke results predate this exact source and
cannot substitute for its gates.

# Memory ordering, coherence, peer memory and in-kernel synchronization

## Contents
1. Coherence scopes and cache-modifier bits
2. Memory types (normal, uncached IPC peer heap)
3. Wait counters and in-order completion
4. Polling: how to spin correctly
5. Mailboxes: tagged payloads, epochs, never-reset slots
6. Cross-GPU exchange and all-reduce design
7. Timeouts and hang diagnosis
8. In-kernel timelines

## 1. Coherence scopes and cache-modifier bits (gfx942 / gfx950)

| Source-level hint (Triton/Gluon) | ISA bits | Use |
|---|---|---|
| `.cv` load | `sc0 sc1` (system scope, bypass) | polls of peer / cross-XCD data |
| `.wt` store | `sc0 sc1` (write-through to system scope) | flags and small coherent payloads |
| `.cg` / `.cs` load | `nt` (streaming, non-temporal) | one-pass weight streams |
| `sc1` only | device scope | cross-CTA on the same GPU (aiter's choice; verify for your case) |

- `.cv` is invalid on stores. `.cv` on a `gl.load` scalarizes in Gluon (see `codegen.md`); use
  `gl.amd.cdna4.buffer_load(..., cache=".cv")` on int32/int64 elements.
- A `[1]`-shaped poll lowers to a scalar (K$) load, which the `.cv` bit does not make coherent; it hung. Poll a
  64-wide replicated vector instead.
- The L2 is per XCD and not coherent inside a kernel (see `architecture.md`). Kernel start invalidates the L2, so
  bulk data published before a flag can be read with plain cached loads after the flag is seen, if the reader's
  lines were not cached earlier in this launch.
- Plain stores are not enough for a cross-CTA protocol that re-reads the slot: an external kernel failed its exact
  equality check until its stores were made coherent.

## 2. Memory types

- Normal device memory (hipMalloc / torch): cached in the writer's L2 and MALL; fine for kernel-boundary handoff.
- Uncached IPC heap (e.g. `hipExtMallocWithFlags(..., hipDeviceMallocUncached)` exported with `hipIpcGetMemHandle`,
  wrapped in AITER's `UncachedIpcHeap` and vLLM's `PeerBuffer`): never cached; every access is a memory or XGMI
  transaction. Allocations below 2 MiB must be whole 4 KiB pages for the IPC handle.
- Measured on uncached peer memory (MI355X):
  - Write-through (`.wt`) stores: ~30 GB/s per rank; plain stores: 110-300 GB/s. Use plain stores for data, `.wt`
    only for flags. A 3.4 MB all-reduce went 484 -> 183 us from this change alone.
  - A library kernel's output written into the uncached heap ran at the same speed as into normal memory
    (MoE expert GEMM with atomic accumulate: 61.4 vs 62.5 us) — the cost is in write-through, not in "uncached".
  - Remote reads want 128-byte contiguous runs per row segment; 64-byte runs were ~2x slower.

## 3. Wait counters and in-order completion

- `vmcnt` counts outstanding vector-memory loads and stores together (gfx9 family). `s_waitcnt vmcnt(0)` in a poll
  loop drains every in-flight prefetch, and a load issued after remote stores waits for their acknowledgements.
- Vector memory operations complete in order. A poll issued after a large weight burst returns only after the
  burst; a publish store queued behind it is delayed.
  - Order code as: critical-path polls and publishes first, then bulk prefetch; or move prefetch into LDS with
    async copies (`buffer_load_to_shared`), which retire through their own group counters.
- Peer fan-out: loading each peer's base address inside the store loop turned 8 remote stores into 8 round trips
  (5.7 us). Load all peer bases up front (or keep the table in LDS), then issue one store over a
  [peers x elements] tensor: 1.8-3.1 us. For 4 KB to 8 ranks: 4.3 -> 0.68 us.
- Replicated lanes storing the same `.wt` value serialize; de-replicate through a blocked layout first.
- `s_barrier` waits on LDS (`lgkmcnt`) but not on `vmcnt`: a barrier does not make other waves' global loads or
  stores complete.

## 4. Polling

- Spin in wave-local inline asm: n x `global_load_dwordx2 ... sc1` (or sc0 sc1 for peer memory),
  `s_waitcnt vmcnt(0)`, compare tags, OR results into an SGPR, `s_sleep` and retry; use `s_and_saveexec` for the
  active lanes. Outputs of inactive lanes are undefined; mask them.
- The asm is elementwise: if a thread holds k elements of the polled tensor you get k sequential round trips. Poll
  in a one-element-per-thread layout, then convert or stage through LDS.
- The compiler will hoist ordinary data loads above a spin loop unless the poll is an asm with
  `~{memory}` clobber, or the data address carries an opaque dependency on the poll result (`s_mov_b32 $0, 0` with
  `"=s,r"`, added to the pointer; keep `gl.multiple_of` on it or the loads devectorize). Check the LLVM IR for the
  clobber. Symptom: zeros read on long waits, typically first launch on one rank.
- A flag wait that guards other loads by other warps needs a CTA barrier after it: warps with no lanes on the flag
  skip the wait. Payload polls, where each lane consumes what it polled, need no barrier.
- Many pollers on peer memory amplify uncached traffic (240 CTAs polling with a 64-cycle sleep keep a steady
  stream of uncached loads). Have a few CTAs poll and republish a small tagged copy locally, or poll with backoff.
  Measured backoff changes were neutral in one all-reduce; test, do not assume.
- CTA-wide "all ok" reductions per iteration (LDS + barrier) are slower than wave-local asm polls.

## 5. Mailboxes: tagged payloads, epochs, never-reset slots

- Put the tag in the payload word: int64 slot = (tag << 32) | payload (or tag in a lane-parallel side word).
  Consumers spin until the tag matches. This removes the data -> fence -> flag sequence, which costs an extra hop
  (~2-3 us); used for per-stage outputs, expert partials, routing selections, reduced activations.
- Tag = (epoch mod 2^22) * 256 + slot + 1, slot unique per (kernel, layer, stage). The epoch is bumped once per
  forward pass by a tiny op captured in the CUDA graph, so replays never see stale tags and nothing is ever zeroed.
  Double-buffer peer regions by epoch parity so a fast rank cannot overwrite data a slow rank is still reading.
  When one step is split into two launches, give the second its own epoch range (e.g. offset by 2^21).
- Use separate flags (+ bulk untagged data) only for large payloads where tagging doubles traffic: at 240 rows a
  tagged-pair mailbox format would roughly double XGMI bytes, so the all-reduce moved to untagged bf16 with per-tile
  flags.
- Size every region from the maximum row count. A region sized for 8 rows overflowed into the next region at 16
  rows; the stray writes carried valid tags, so tag censuses and stage-mask bisects pointed the wrong way.

## 6. Cross-GPU exchange and all-reduce design

Final design of an exact bf16 all-reduce of [240, 7168] (3.4 MB) across 8 GPUs, 33 us (from 484 us):
- Producers (the GEMM itself, via `out=`) write their partial directly into their own uncached peer buffer
  ("direct input"), then raise a ready flag on every rank (`.wt`, one CTA).
- 224 owner units, one per CTA, each owning [16 rows x 64 cols] (or [8 x 64] for narrower tensors) so each remote
  read is a 128-byte row segment. Each owner polls the 8 ready flags, pulls its tile from all 8 ranks in a fixed
  rank order, sums in fp32 (deterministic, identical on all ranks), writes bf16 into its own buffer and raises a
  per-tile all-gather flag on every rank.
- Consumers poll the tile flags they need and pull the reduced tiles (all-gather by pull, not push: one CTA pushing
  32 KB of fan-out was ~10x slower).
- Fuse the consumer's math (residual update, residual mix, norm) into the same kernel, and compute anything that
  does not depend on the all-reduce before waiting on it.
Lessons: 112 bigger units were slower than 224 smaller ones; [32x32] units (64 B segments) were worse (54.7 us);
destination rotation and polling backoff changed nothing; WAR safety comes from the protocol (an owner raises its
all-gather flag only after reading every rank's input, and each rank waits on all flags before finishing).

## 7. Timeouts and hang diagnosis

- Bound every wait with a wall-clock timeout (`s_memrealtime`, 100 ticks per us; e.g. 10 s), not an iteration cap:
  a counter changes codegen enough to hide a race. On timeout the first waiter writes a health record (region,
  slot, expected tag, value seen, rank, CTA), raises an alarm flag and lets every CTA exit, turning a hang into a
  report.
- Hang triage from the host: launch on a non-blocking stream, wait, then copy scratch and peer regions on a side
  stream into pinned host memory and print a per-rank census (stages finished, regions with stale tags, missing
  stamps). A copy on the default stream blocks behind the hung kernel.
- Never run two persistent spin-waiting grids on the same GPUs at once (yours and someone else's benchmark): they
  can deadlock each other; killing them once wedged the driver (KFD) for ~20 minutes.

## 8. In-kernel timelines

- Stamp `s_memrealtime` at stage boundaries per CTA into a scratch array (8 slots per task: start, sub-steps,
  end); make the stamp data-dependent on the value it times (`"=s,v"`) so it is not reordered. Reduce to
  min / median / max over CTAs per stage; compare with the reference kernel's stamps stage by stage.
- Device clocks are not synchronized across GPUs; stamp relative to each rank's own layer start. To split a
  cross-rank wait into "transfer" and "straggler", add a stamp right after the push: a hypothesized straggler problem
  turned out to be a slow push (peer data was present within 0.2 us).

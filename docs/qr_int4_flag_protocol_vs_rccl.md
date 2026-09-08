# The QRInt4 flag protocol, and how RCCL solves the same problem

**Status: working document.** Written against the ATT capture in
`.wavescope/runs/run-2026-09-07T13-09-30-825Z-91a36202650c968a/ui_output_agent_55393_dispatch_705/`
(TP4, MI350P/PCIe, `qr_int4_kernel_ablate.py`, `super_tile=8`, 56 workgroups,
448 tiles = 14 MiB bf16). RCCL code read at
[`ab6aee04d`](https://github.com/ROCm/rccl/tree/ab6aee04d095f9a63c2911ecd5cc246b7eb39973)
(`develop`, 2026-08-29) — a pinned commit, so the line links below stay valid.

Every number quoted from the trace is marked **[trace]**. Every number quoted
from RCCL source or from a compiler experiment is marked **[src]** or
**[compiler]**. Anything else is inference, and says so.

---

## 0. Why this document exists

The ATT annotation
[`rendezvous-is-the-kernel`](../.wavescope/runs/run-2026-09-07T13-09-30-825Z-91a36202650c968a/ui_output_agent_55393_dispatch_705/annotations.json)
puts **97.6% of all attributed stall in the inter-rank rendezvous** — 70.4% in
the `while current != color` spin of `_wait_flag`, and a further 27.2% in the
`gpu.barrier()` that exists only to hold the workgroup behind that spin. The
INT4 codec, the LDS traffic and every load and store together are **0.405%**.
**[trace]**

So the interesting question is not "is the codec fast" but "what is the
rendezvous, and what does a production collectives library do instead". This
note answers both, in that order, at the same level of detail as
`gpu-collectives/10-algorithms/03-rccl-ring-all-reduce.md` in the knowledge
base.

Two things this note does **not** claim:

- It does not claim the 18M-cycle reduce-scatter wait is comm cost. The
  annotation's leading hypothesis is host-side rank start skew, and §6.4 argues
  RCCL would eat that skew too. That hypothesis has to be settled by
  measurement, not by reading code.
- It does not claim RCCL is faster here. It claims RCCL's rendezvous has four
  specific properties QRInt4's does not, and §8 sizes each one.

---

## 1. What the protocol has to guarantee

Both kernels are solving the same problem, and it is worth writing the contract
down before looking at either implementation.

Rank `A` wants rank `B` to consume some bytes `A` produced. `A` writes them
straight into memory `B` owns (a HIP IPC mapping — see
[IPC and peer memory](../../claude-knowledge-base/gpu-collectives/00-foundations/02-ipc-and-peer-memory.md)).
There is no interrupt, no doorbell, no queue. `B` finds out by **reading a
second location that `A` writes afterwards**. That second write is the *flag*,
and the whole protocol is three obligations:

| obligation | who | why it is not free |
|---|---|---|
| **release** — payload must be *visible* before the flag is | writer | a store that has retired (`vmcnt(0)`) may still be dirty in the writer's L2 |
| **acquire** — the reader must not answer the flag load, or the payload loads that follow it, from a stale cached line | reader | `nt` is a *hint*; only a bypass or an invalidate is a guarantee |
| **liveness** — the reader must poll, and polling costs the very fabric the writer is using | reader | this is the part the trace is stuck on |

Everything below is a different way of discharging those three.

---

## 2. QRInt4's inbox: the address space

Read this section with
[`qr_int4_kernel.py`](../aiter/ops/flydsl/kernels/qr_int4_kernel.py) open.

### 2.1 Constants

| symbol | value | meaning |
|---|---:|---|
| `BLOCK` | 256 | threads per workgroup (4 waves of 64) |
| `ATOMS` | 8 | 16 B atoms per thread per tile |
| `TILE_BYTES` | 32768 | one tile = 256 × 8 × 16 B of bf16 payload |
| `RANK_TILE_BYTES` | 1152 | one rank's INT4 share of one tile: 1024 B nibbles + 128 B E4M3 scales |
| `N_SECTORS` | 18 | 1152 / 64 — the rank-tile in 64 B fabric sectors |
| `PHASES` | 2 | reduce-scatter, all-gather |
| `super_tile` (ST) | 1 or 8 | tiles batched behind one publish |

Derived, for the traced configuration (`world_size=4`, `ST=8`):

```
rank_atoms        = ATOMS / world_size          = 2
rank_payload_i32  = rank_atoms * 288            = 576 i32  = 2304 B   (one rank-tile pair)
release_i32_off   = ST * rank_payload_i32       = 4608 i32 = 18432 B  (flag sits here)
wire_tile_i32     = release_i32_off + 16        = 4624 i32 = 18496 B  (slot stride)
```

The `+ 16` is the flag: **16 i32 = one 64 B sector**, appended after the ST
rank-tiles of every slot.

### 2.2 The slot layout

[`wire_slot_layout`, qr_int4_kernel.py:572-580](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L572-L580)
is a 4-D layout indexed `(phase, block, src_rank, sub_tile)`:

```
inbox
├── [0 .. flags_i32)                 reserved header, never written by the kernel *
└── wire slots, phase-major:
    phase 0 (reduce-scatter)
      block 0
        src 0 : [ sub 0 | sub 1 | ... | sub 7 ][ FLAG 64 B ]   18496 B
        src 1 : [ sub 0 | sub 1 | ... | sub 7 ][ FLAG 64 B ]
        src 2 : ...
        src 3 : ...
      block 1
        ...
    phase 1 (all-gather)
      ... same shape ...
```

\* `flags_i32 = PHASES * grid * world_size` dwords are allocated
([`qr_int4.py:264`](../aiter/ops/flydsl/kernels/qr_int4.py#L264) sums
`flags_bytes + data_bytes`) and every wire address is offset past them by
[`_sub_tile_i32`](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L618), but
nothing in the kernel reads or writes that region. It is a vestigial header.
Minor, but worth knowing when reading the address arithmetic.

**The slot is indexed by the *source* rank, not the destination.** Rank `A`
writing to rank `B` writes into `B`'s inbox at `src = A`. So each rank's inbox
is `PHASES × grid × world_size` private mailboxes; no two writers ever touch the
same bytes. That is what makes the whole thing lock-free: there is no
arbitration because there is no sharing.

### 2.3 Where the flag lives, exactly

Within slot `(phase, bid, src)`, the flag is the 64 B at byte offset
`release_i32_off * 4` = 18432, i.e. the **last** 64 B. Its 16 dwords all hold
the same value, the *colour*.

The reader only ever reads dword 0 of that sector
([`_wait_release`, :775-786](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L775-L786)).
The other 15 dwords exist so the *writer* can emit the flag as a full 64 B
sector — four lanes × `global_store_dwordx4` — matching the fabric's transfer
granule rather than issuing a sub-sector partial write.

---

## 3. Publish: the release side, instruction by instruction

[`_publish`, :732-767](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L732-L767).
Under the `default` (coarse-grained) and `finegrained` inbox policies it emits:

```
  s_waitcnt vmcnt(0)        ; this wave's payload stores have retired
  s_barrier                 ; ... and so have the other three waves'
  buffer_wbl2 sc1           ; push this XCD's dirty L2 lines out
  s_waitcnt vmcnt(0)        ; wait for the writeback to land
  global_store_dwordx4 <peer0 flag>, <colour x4>, off sc0 sc1   ; write-through
  global_store_dwordx4 <peer1 flag>, <colour x4>, off sc0 sc1
  ...                                                            (one quad, one lane per peer)
```

Four things are load-bearing and all four are commented in the source:

1. **`vmcnt(0)` is per-wave.** The workgroup barrier joins all four waves, so a
   flag issued by wave 0 cannot precede payload still in flight from wave 3.
2. **`buffer_wbl2 sc1`** is the actual release on a *cacheable* inbox. `vmcnt(0)`
   only says the store left the wave; on coarse-grained pages the line can sit
   dirty in the XCD's L2 indefinitely. L2 is per-XCD, so every workgroup must
   issue its own — one workgroup's writeback says nothing about a workgroup on
   another die.
3. **`sc0 sc1` on the flag store** makes it write-through, so the peer's spin
   sees it as soon as it lands rather than after an eviction.
4. The payload stores themselves are **plain** (no `nt`) precisely so they *can*
   linger in L2 and be write-combined — the 64 B destination-interleaved fanout
   is worth ~1.5× when the L2 gets to coalesce it
   ([policy table, :92-131](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L92-L131)).

**This release sequence is byte-for-byte what RCCL emits.** §6.5 shows the
compiler output. The release side is not where the two differ.

---

## 4. Wait: the acquire side, instruction by instruction

[`_wait_flag` / `_wait_release`, :769-786](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L769-L786):

```python
def _wait_flag(flag_rsrc, color):
    current = _load_i32_uncached(flag_rsrc)
    while current != color:
        current = _load_i32_uncached(flag_rsrc)
        _invalidate_l1()

def _wait_release(phase, color):
    if tid < world_size:
        elem = _sub_tile_i32(phase, tid, 0) + release_i32_off
        _wait_flag(create_buffer_resource_from_addr(peer_vec[rank] + elem*4), color)
    gpu.barrier()
```

Three structural facts:

- **`tid < world_size` — thread `t` polls the flag written by source rank `t`.**
  With TP4 that is lanes 0–3 of wave 0. The other 252 threads go straight to the
  barrier.
- **The address is `peer_vec[rank]` — this rank's *own* inbox.** The poll is a
  read of local memory, not a read across the fabric. (RCCL's is too; §6.2.)
- **The barrier after it is a correctness fence**, not just a join: nothing that
  reads the inbox may be hoisted above it.

### 4.1 What actually gets emitted

Per the trace disassembly **[trace]**, one loop iteration is:

```
  buffer_load_dword v3, off, s[40:43], 0 nt     ; idx 1831
  s_waitcnt vmcnt(0)
  buffer_inv sc1                                 ; idx 1839
  <compare, backward branch>                     ; idx 1835
```

plus, ahead of the load, a **descriptor waterfall** — four
`v_readfirstlane_b32`, two `v_cmp_eq_u64`, `s_and_saveexec_b64` — because
`create_buffer_resource_from_addr` is handed a *lane-varying* address, so the
buffer descriptor itself is divergent (annotation
`flag-poll-descriptor-waterfall`; 1,243,588 cy total, 0.159% of stall).

### 4.2 The `nt` is a bug

The source asks for `cache_modifier=_CM_SC1`
([`_load_i32_uncached`, :450-455](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L450-L455)),
with `_CM_SC1 = 2` from the constants at
[:59-61](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L59-L61).
`buffer_ops.buffer_load` passes that integer straight through as the intrinsic's
`aux` operand, which LLVM interprets as its `CPol` encoding
([`SIDefines.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIDefines.h)) **[src]**:

```c
enum CPol { GLC = 1, SLC = 2, DLC = 4, SCC = 16,
            SC0 = GLC, SC1 = SCC, NT = SLC, ... };
```

So on gfx942/gfx950 the correct constants are **`SC0 = 1`, `NT = 2`, `SC1 = 16`**,
and the kernel's table has `_CM_SC1 = 2` (which is NT) and `_CM_NT = 4` (which is
DLC, dropped on CDNA). The emitted `nt` in §4.1 is the direct consequence, and it
is *measured*, not inferred — the annotation read it off the listing.

`nt` is a reuse hint. It does not bypass anything. So the loop's entire
correctness rests on `buffer_inv sc1`, and the **pre-loop load has no invalidate
before it at all**.

The same constants are in
[`qr_int4_kernel.py:59-61`](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L59-L61),
[`qr_int4_kernel_ablate.py:145-147`](../op_tests/multigpu_tests/qr_ablation/qr_int4_kernel_ablate.py#L145-L147),
and the same `_CM_*` names are imported by the ring and 1-stage kernels.

### 4.3 `buffer_inv sc1` is an agent-scope acquire fence

Verified on this host's toolchain (HIP 7.15.26333, `--offload-arch=gfx950`) **[compiler]**:

| source | emitted |
|---|---|
| `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")` | `buffer_inv sc1` |
| `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "")` (system) | `buffer_inv sc0 sc1` |
| `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "")` | `buffer_wbl2 sc0 sc1` ; `s_waitcnt vmcnt(0)` |
| `__atomic_load_n(u64*, __ATOMIC_RELAXED)` | `global_load_dwordx2 ... sc0 sc1` |

That last row is the one to remember: **a plain relaxed system-scope atomic load
already bypasses both caches, with no invalidate**. The spin loop is doing with a
device-wide cache invalidate what one CPol bit on the load would do for free.

The cost of getting it wrong scales with how many things are doing it. In this
capture **[trace]**: 56 workgroups, ~1,157 polls per polling wave, **≈64k
invalidate + uncached-load round trips per rank**, each invalidate discarding the
CU's vector L1 and the XCD L2's non-locally-owned lines — while the peers'
incoming payload writes are trying to land in the same L2s.

### 4.4 What a poll costs, measured

From `att range` over idx 1835 **[trace]**:

| | executions | total cy | mean | median | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| reduce-scatter poll (idx 1835) | 33,520 | 520,246,056 | 15,520 | 17,028 | 17,232 | 4,188 | 57,600 |
| all-gather poll (idx 2477) | 2,464 | — | 11,640 | — | — | — | — |

**15.5k cycles for one dword read of local memory.** At the ~2.18 GHz implied by
`realtime.json` that is ~7 µs per poll; even the fastest observed poll is 4,188
cy ≈ 1.9 µs.

Two consequences, and it is important not to mix them up:

- **The loop is *already* rate-limited**, at roughly one poll per 15.5k cycles.
  So "add a backoff" is not, by itself, going to reclaim the 18M cycles — the
  annotation says this plainly, and it is right. Detection overshoot is at most
  one poll period, 0.09% of the wait.
- **But the reason it is rate-limited is that each poll is expensive**, and that
  expense is at least partly self-inflicted (uncached load through a divergent
  descriptor, plus a device-scope invalidate, issued by 56 concurrently polling
  waves — one per workgroup, `world_size` lanes wide). Whether
  the remainder is fabric queueing behind the peers' incoming writes is
  **unmeasured** — this run has no PMC pass (annotation
  `unmeasured-memory-side`, item (d)).

---

## 5. Worked example: one workgroup, TP4, 14 MiB

Concrete numbers for the traced shape. `world_size=4`, `M=1024`, `hidden=7168`,
bf16 → 14.68 MB → **448 tiles** of 32 KiB, **56 workgroups**, `ST=8`, so each
workgroup owns exactly 8 tiles = **one super-tile group** = one pass through the
loop body at
[:858-901](../aiter/ops/flydsl/kernels/qr_int4_kernel.py#L858-L901).

Follow workgroup `bid = 17` on rank 1, with colour `c`.

**Phase 0 — reduce-scatter**

| step | what happens | bytes on the wire |
|---|---|---:|
| 1 | for `s = 0..7`: load tile `17 + s*56` from HBM, quantize each destination's 2 atoms into LDS, `_fanout_nt` 2×1152 B per peer | 2304 B × 3 remote peers × 8 sub-tiles = **55,296 B** |
| 2 | `_publish(0, rank=1, c)`: `vmcnt(0)`, barrier, `buffer_wbl2 sc1`, `vmcnt(0)`, then one 64 B colour sector to each of 4 peers | 192 B remote |
| 3 | `_wait_release(0, c)`: lanes 0–3 spin on `own_inbox[phase 0][block 17][src 0..3] + 18432` until each reads `c` | — |
| 4 | `gpu.barrier()` — the other 3 waves have been parked here since step 2 | — |

At step 3, rank 1 block 17 is waiting for **block 17 of every other rank** to
have finished its own step 2. The rendezvous is per-`(phase, block)` and
**N-way**: it completes at the *maximum* over 4 arrival times.

**Phase 1 — all-gather**: identical shape, `phase = 1`, dequantize-accumulate
first, then quantize the reduced slice and replicate it to all peers.

Then `color = color + 1` (skipping 0, which is the unset sentinel) and the block
exits.

**Totals per rank for the whole collective:**

| quantity | value |
|---|---:|
| remote payload egress | 56 blocks × 8 subs × 2304 B × 3 peers × 2 phases = **6.19 MB** |
| — vs bf16 two-shot (`2(N-1)/N × S`) | 22.0 MB → **3.55× compression** |
| publishes (each = one `buffer_wbl2` + 4 flag sectors) | 56 × 2 = **112** |
| flag bytes | 112 × 256 B = 28.7 KB (0.5% of payload) |
| rendezvous events | **112** |
| polling lanes live at once, per phase | 56 × 4 = **224** |
| poll round trips (load + `buffer_inv sc1`) | **≈64,000** **[trace]** |

The flag *volume* is negligible. The flag *protocol* is 97.6% of the stall. That
is the whole story in two lines.

### 5.1 What the trace says about the two waits

| | mean per leader wave | min | max | stdev |
|---|---:|---:|---:|---:|
| reduce-scatter spin `[1821,1842)` | 17,988,490 cy | 17,794,060 | 18,268,208 | 144,388 (**0.80%**) |
| all-gather spin `[2462,2484)` | 994,709 cy | 749,856 | 1,200,688 | 97,767 (**9.83%**) |

**[trace]** Ratio 18.1×. The 0.80% spread is the discriminating number: 29
workgroups spread over 8 shader engines were all released within ~0.5M cycles of
each other. Bandwidth-limited arrivals stagger (the all-gather wait, 9.8%,
*is* staggered). A long flat wait ending in a near-simultaneous release is the
signature of **one global event arriving late** — most plausibly the peer ranks
starting the dispatch ~8.2 ms after this one. By the second rendezvous the ranks
are in lockstep and the wait falls 18×.

That hypothesis is not settled by this trace and §8 does not assume it.

---

## 6. RCCL: the same three obligations, four different answers

RCCL's ring is `2(N-1)` sequential hops instead of 2, and each hop has exactly
the same release/acquire/liveness problem. The algorithm itself is covered in
[the KB note](../../claude-knowledge-base/gpu-collectives/10-algorithms/03-rccl-ring-all-reduce.md);
this section is only about the handshake.

### 6.1 Who waits: roles

RCCL does **not** have every thread, or every lane of a wave, participate in the
rendezvous. The `Primitives` constructor assigns *roles*
([`prims_simple.h:822-827`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/prims_simple.h#L822-L827)) **[src]**:

```c
if      (tid < nrecv)                 { flags |= RoleWaitRecv; index = tid; }
else if (tid < nrecv+nsend)           { flags |= RoleWaitSend; index = tid-nrecv; }
else if (nthreads-nsend <= tid)       { flags |= RolePostSend; index = ...; }
else if (nthreads-nrecv-nsend <= tid) { flags |= RolePostRecv; index = ...; }
```

A ring is instantiated with `FanSymmetric<1>`
([`all_reduce.h:74`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/all_reduce.h#L74)),
so `nrecv = nsend = 1`. **Exactly one thread per block polls the incoming
counter** and one polls the outgoing credit. Everything else is a worker.

QRInt4 puts `world_size` lanes on the poll, and the count of *pollers per rank*
is `blocks × world_size` — 224 in the traced run, against RCCL's
`nChannels × 1` (typically 8–32).

### 6.2 What is polled: credit counters, in local memory

There is no per-message flag. Each connection has two monotonic 64-bit counters
([`device.h:193-199`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/include/device.h#L193-L199)) **[src]**:

```c
uint64_t *tail;   // Local for recv, remote for send
uint64_t *head;   // Local for send, remote for recv
```

and `p2pRecvConnect` wires them up
([`p2p.cc:569-570`, `575-577`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/transport/p2p.cc#L569-L577)) **[src]**:

```c
struct ncclRecvMem* devMem = resources->recvDevMem;   // MY buffer
recv->conn.tail = &devMem->tail;                      // local
recv->conn.head = &remDevMem->head;                   // peer's
```

So the receiver's spin, like QRInt4's, reads **local** memory that a peer writes
into. Same allocation type, too: the shareable buffer comes from
`hipDeviceMallocUncached` or `hipDeviceMallocFinegrained` depending on the
`HIP_UNCACHED_MEMORY` build macro
([`p2p.cc:249-253`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/transport/p2p.cc#L249-L253)).

**The memory is the same. What differs is the loop.**

### 6.3 The spin loop

[`waitPeer`, `prims_simple.h:144-157`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/prims_simple.h#L144-L157) **[src]**:

```c
int spins = 0;
while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
  __builtin_amdgcn_s_sleep(1);
  connStepCache = loadStepValue(connStepPtr);
  if (checkAbort(flags, Aborted, spins)) break;
}
__asm__ __volatile__("s_wakeup");
```

with
([`:115-130`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/prims_simple.h#L115-L130)):

```c
inline __device__ uint64_t loadStepValue(uint64_t* ptr) {
  // volatile is faster than acquire but not as correct. Make sure reduceCopy
  // loads data using volatile so it doesn't see stale data in L1.
  return __atomic_load_n(ptr, __ATOMIC_RELAXED);
}
```

Four differences from `_wait_flag`, in decreasing order of how much they matter:

**(a) `connStepCache` is a register, and the counter is monotonic.**
This is the big one. The loop exits as soon as the *cached* value is large
enough. A receiver that has fallen behind the sender by `k` slices therefore
runs its next `k` `waitPeer` calls with **zero memory traffic** — the register
already proves the data is there. In the steady state, when the sender is
comfortably ahead, `waitPeer` is a register compare and a fall-through.

QRInt4 has no equivalent. Every `_wait_release` is a fresh N-way read of memory,
because the colour changes every iteration and there is nothing to cache.

**(b) The sender has a credit window of `NCCL_STEPS = 8`.**
The `isSendNotRecv ? NCCL_STEPS : 0` term is the flow control: the sender may
run up to 8 FIFO steps ahead of the receiver's `head` before it must block. With
`NCCL_STEPS = 8` and
([`collectives.h:22-24`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/include/collectives.h#L22-L24))
`ALLREDUCE_CHUNKSTEPS = 4`, `ALLREDUCE_SLICESTEPS = 2`,
`ALLREDUCE_SLICESTEPS_SINGLE_NODE = 4`, the ring/Simple all-reduce is built with
([`all_reduce.h:574-581`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/all_reduce.h#L574-L581)):

| | `SlicePerChunk` | `StepPerSlice` | credit window |
|---|---:|---:|---:|
| `work->rcclUseOneSlice` | 1 | 4 | 8/4 = **2 slices** |
| otherwise | 2 | 2 | 8/2 = **4 slices** |

Combined with (a), that is a pipeline: a producer that is transiently faster
never blocks, and a consumer that is transiently faster never polls.

QRInt4's pipeline depth is **1**. Every super-tile group is a full N-way barrier
and nobody may run ahead.

**(c) `s_sleep(1)` between attempts, `s_wakeup` on exit.**
`s_sleep N` parks the wave for ≈`64·N` clocks. It costs nothing when the flag is
already there (the loop body never runs) and it stops a long wait from
re-issuing at full rate. RCCL uses the same construct inside its *workgroup*
barrier too
([`primitives.h:19-47`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/primitives.h#L19-L47)).
`_wait_flag` has no backoff at all; `qr_1stage_kernel.py` already added one
([:369-379](../aiter/ops/flydsl/kernels/qr_1stage_kernel.py#L369-L379)) and
`qr_int4` has not picked it up.

**(d) Freshness comes from the load, not from an invalidate.**
`__atomic_load_n(ptr, __ATOMIC_RELAXED)` on a `uint64_t*` compiles to
`global_load_dwordx2 ... sc0 sc1` **[compiler]** — one instruction, bypassing L1
and L2. There is **no `buffer_inv` anywhere in the spin**. QRInt4 issues a
device-scope acquire fence per iteration instead, because its load asked for
`sc1` and got `nt` (§4.2).

### 6.4 Rank start skew: RCCL does not solve it either

Worth stating, because it is the annotation's leading hypothesis for the 18M
cycles and it would be easy to read §6.3 as a cure.

In `runRing`, operation `j = 1` is `prims.directSend` — a pure send. Its
`waitPeer` is on `RoleWaitSend`, i.e. on the *outgoing credit*, and the FIFO
starts empty, so it passes immediately. The first thing that can actually block
is the `directRecvReduceDirectSend` at `j = 2`, which waits on the predecessor's
payload. If the predecessor's kernel has not launched yet, RCCL spins there for
exactly as long as QRInt4 spins in `_wait_release`.

The differences are second-order but real: RCCL waits on **one** predecessor
(max over 1 arrival) rather than **N** (max over N); it burns ~1 poller per
channel rather than 224; and each poll is one bypassing load plus a 64-clock
sleep rather than an uncached load plus a device-wide invalidate. **Skew is a
host problem in both cases.** Settling it is annotation suggestion (a): a
`dist.barrier()` + `torch.cuda.synchronize()` on every rank immediately before
the timed launch, then re-measure.

### 6.5 The release side is identical

`postPeer`
([`prims_simple.h:216-237`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/prims_simple.h#L216-L237)) **[src]**:

```c
else if ((flags & RolePostSend) && dataStored) {
#ifdef __GFX9__
  __threadfence();
#else
  __threadfence_system();
#endif
}
if ((flags & Send*RolePostSend) && next_hdp_reg) STORE(next_hdp_reg, 0x1);
if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
  step += StepPerSlice;
  STORE(connStepPtr, step);
}
```

`STORE` on GFX9 is `__atomic_store_n(..., __ATOMIC_RELAXED)`
([`common.h:21-23`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/common.h#L21-L23)).
Compiling `*q = 1; __threadfence(); __atomic_store_n(p, 2ull, __ATOMIC_RELAXED);`
for gfx950 gives **[compiler]**:

```
global_store_dword v0, v1, s[2:3]
buffer_wbl2 sc1
s_waitcnt vmcnt(0)
buffer_inv sc1
global_store_dword v0, v2, s[0:1] sc0 sc1
```

Compare §3. **`buffer_wbl2 sc1` → `s_waitcnt vmcnt(0)` → write-through flag
store is exactly QRInt4's `_publish`.** The two kernels discharge the release
obligation the same way, down to the instruction.

### 6.6 The HDP flush is not a lever on gfx950

`next_hdp_reg` above is the peer GPU's *HDP memory-flush control* register — the
AMD mechanism for pushing writes that crossed PCIe out of the destination's host
data path. It looks like exactly the PCIe visibility knob QRInt4 is missing.

It is not, on this hardware. `p2pSendSetup` only fetches it when the link is
**not** xGMI **and** the arch is **not** gfx90a/gfx942/gfx950
([`p2p.cc:393-402`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/transport/p2p.cc#L393-L402)) **[src]**:

```c
if (!isXGMI && !IsArchMatch(..., "gfx90a") && !IsArchMatch(..., "gfx942") && !IsArchMatch(..., "gfx950")) {
  CUDACHECK(hipDeviceGetAttribute((int*)&resources->next_hdp_reg, hipDeviceAttributeHdpMemFlushCntl, peerInfo->cudaDev));
}
```

On MI350P over PCIe the register is never read and `next_hdp_reg` stays 0. RCCL
relies on `buffer_wbl2 sc1` + the write-through counter store, same as QRInt4.
Closing this off explicitly because it is an attractive red herring.

### 6.7 Push, not pull — on AMD, always

`ncclTopoCheckP2p` sets `*read = 0` unconditionally and only raises it inside a
`#if !defined(__HIP_PLATFORM_AMD__)` block
([`paths.cc:274-280`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/graph/paths.cc#L274-L280)) **[src]**.
`NCCL_P2P_READ_ENABLE` can override it. So by default RCCL, like QRInt4,
**pushes**: the sender stores into the receiver's buffer. The comparison is
apples to apples.

### 6.8 LL and LL128: delete the flag by putting it in the data

This is the one place where RCCL does something QRInt4 has no analogue for, and
it is the most directly transplantable idea in this document.

`ncclLLFifoLine`
([`device.h:99`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/include/device.h#L99)) **[src]**:

```c
union ncclLLFifoLine {
  struct { uint32_t data1; uint32_t flag1; uint32_t data2; uint32_t flag2; };
  uint64_t v[2];
  int4 i4;
};
```

16 B carrying 8 B of payload. Each `(data, flag)` pair is one naturally-aligned
8 B write, and 8 B writes do not tear, so **observing the flag implies observing
the data**. The receive is
([`prims_ll.h:150-180`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/prims_ll.h#L150-L180)):

```c
do {
  *((u64_gptr)i4.v)     = __builtin_nontemporal_load((u64_gptr)src->v);
  *((u64_gptr)i4.v + 1) = __builtin_nontemporal_load((u64_gptr)src->v + 1);
  if (checkAbort(abort, 1, spins)) break;
} while ((i4.flag1 != flag) || (i4.flag2 != flag));
```

Consequences, all of them the point:

- **No release fence.** No `buffer_wbl2`, no `vmcnt(0)` join, no workgroup
  barrier before publishing.
- **No separate poll.** The spin *is* the payload load. A failed poll is not
  wasted traffic — it fetched real data, it just was not there yet.
- **No acquire fence.** Freshness is per-line, from the flag word travelling in
  the same line as the data it guards.
- **The flag is the step counter**, `NCCL_LL_FLAG(step+1)`
  ([`prims_ll.h:67`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/prims_ll.h#L67))
  — monotonic, exactly like QRInt4's colour, for exactly the same reason: a
  boolean that gets reset races the next call.

The price is **2× the bytes**. LL128 buys most of it back with one flag per
*line* instead of per 8 B — and RCCL redefines the line to 64 B, not NCCL's 128
([`device.h:162`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/include/device.h#L162)):

| | line | data | flag | wire efficiency |
|---|---:|---:|---:|---:|
| NCCL LL128 | 128 B | 120 B | 8 B | 93.75% |
| **RCCL LL128** | **64 B** | **56 B** | **8 B** | **87.5%** |
| RCCL LL | 16 B | 8 B | 8 B | 50% |

64 B is xGMI's transfer granule — and it is exactly the sector QRInt4's fanout
already writes.

At the traced size RCCL would pick LL128: the gfx950 `llProtoRanges` AllReduce
row declares LL over `[0, 256 KiB)` and LL128 over `[256 KiB, ~67.4 MB)`
([`tuning.cc:406-419`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/graph/tuning.cc#L406-L419)) **[src]**.

### 6.9 Worked example: the same 14 MiB through the ring

TP4, 14.68 MB, Ring/Simple (chosen for mechanism clarity; LL128 is what the
tuner would actually pick — see above). Say 16 channels.

| quantity | value |
|---|---:|
| hops on the critical path | `2(N-1)` = **6** |
| per-rank wire volume | `2(N-1)/N × S` = 1.5 × 14.68 = **22.0 MB** |
| — vs QRInt4 | 6.19 MB → RCCL moves **3.55×** more (that is the INT4 codec, not the protocol) |
| per-channel buffer | 14.68 MB / 16 = 917 KB |
| FIFO slot size | `buffSizes[SIMPLE]/NCCL_STEPS` = 4 MiB / 8 = **512 KiB** |
| slice size | `StepPerSlice × stepSize` = 1 or 2 MiB, then clamped to what is left of `nelem` ([`prims_simple.h:249-250`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/prims_simple.h#L249-L250)) |
| threads polling the recv counter, per rank | **16** (one per channel) |
| threads polling the send credit, per rank | 16 |
| release fences per rank | one `__threadfence()` per `postPeer` with data |

And the shape of the schedule: rank `p` sends chunk `p-1` at `j=1` without ever
waiting; from `j=2` on it does *receive → reduce → forward*, with the FIFO
letting the producer be two slices ahead. Nothing in the ring is a barrier
across all `N` ranks. The chain is long (6 hops) but every link is a
**depth-2 pipelined point-to-point handshake**, not a rendezvous.

That is the trade in one sentence: **QRInt4 buys a 2-hop critical path by making
every hop an N-way lockstep barrier; RCCL buys a pipelined, skew-tolerant
handshake by making the path `2(N-1)` hops long.**

---

## 7. Side by side

| | QRInt4 two-shot | RCCL ring, Simple | RCCL ring, LL/LL128 |
|---|---|---|---|
| **critical path** | 2 hops | `2(N-1)` hops | `2(N-1)` hops |
| **wire volume / rank** | `2(N-1)/N × S / 3.55` (INT4) | `2(N-1)/N × S` | `2(N-1)/N × S × 1.14` (LL128) |
| **what the reader polls** | per-(phase,block,src) colour sector, own inbox | monotonic `tail` counter, own memory | the payload line itself |
| **poll instruction** | `buffer_load_dword … nt` + `s_waitcnt` + `buffer_inv sc1` | `global_load_dwordx2 … sc0 sc1` | `global_load_dwordx2 … nt` ×2 |
| **freshness mechanism** | device-scope acquire fence, every iteration | CPol bits on the load | flag rides in the line |
| **backoff** | none | `s_sleep(1)` + `s_wakeup` | none (but the poll is useful work) |
| **amortization across waits** | none — new colour every time | `connStepCache` register; a satisfied cache means zero traffic | none needed |
| **pipeline depth** | **1** (full barrier per super-tile group) | `NCCL_STEPS/StepPerSlice` = **2** slices | 2 slices |
| **fan-in of one wait** | **N** peers (max over N arrivals) | **1** predecessor | 1 predecessor |
| **pollers per rank** | `blocks × world_size` = **224** here | `nChannels × 1` ≈ 16 | ≈ 16 |
| **release** | `vmcnt(0)`, barrier, `buffer_wbl2 sc1`, `vmcnt(0)`, `sc0 sc1` store | *identical* | **none** |
| **reduction precision** | fp16 accumulate, one rounding | operand dtype, `N-1` roundings | same |

---

## 8. What is actually different, and what each difference is worth

Ranked by expected value, with the sizing evidence attached. **Nothing here
should be acted on before the skew hypothesis in §5.1 is settled** — if the 18M
cycles is the harness, all of these are optimizations of a 1M-cycle wait, not an
18M-cycle one.

### 8.1 The CPol constants are wrong — fix regardless

`_CM_SC1 = 2` is NT and `_CM_NT = 4` is DLC (§4.2). Correct: `SC0 = 1`,
`NT = 2`, `SC1 = 16`.

- **Correctness risk today:** on the `default` (coarse-grained) inbox arm,
  `recv: _CM_SC0 | _CM_SC1` = 3 emits `sc0 nt`, so the *payload reader does not
  bypass L2* — precisely the stale-line hazard the policy comment warns about.
  Any coarse-grained ablation number taken before the fix is measuring something
  other than what it claims.
- **Perf value:** with the flag load actually `sc1`, `buffer_inv sc1` can come
  out of the loop body entirely and move to a single post-barrier acquire (which
  the ring and 1-stage kernels already do:
  [`qr_int4_ring_kernel.py:486-489`](../aiter/ops/flydsl/kernels/qr_int4_ring_kernel.py#L486-L489)).
  That removes ~64k device-scope L2 invalidates per rank.
- Sites: `qr_int4_kernel.py:59-61`, `qr_int4_kernel_ablate.py:145-147`, and every
  importer of those names.
- **Verify by disassembly**, not by reasoning: rebuild with
  `inbox_memory="default"` and check the receive loads come out `sc0 sc1`.

### 8.2 Adopt `s_sleep` backoff — cheap, already written

`qr_1stage_kernel.py` has the pattern
([:369-379](../aiter/ops/flydsl/kernels/qr_1stage_kernel.py#L369-L379)),
parameterized by `spin_sleep`. RCCL uses `s_sleep(1)` (≈64 clocks) flat.

Honest sizing: the annotation is right that this does *not* buy back the 8 ms —
the loop is already rate-limited at 15.5k cy/poll (§4.4), and detection
overshoot is 0.09% of the wait. Its value is (a) it reduces contention between
224 polling lanes across 56 waves and the incoming payload writes they are waiting on, which is the
one memory-side question worth a counter pass, and (b) it is the right shape
once the wait is short. **Measure it on the all-gather wait (idx 2477, ~1M cy,
skew-free), not the reduce-scatter one.**

### 8.3 Hoist the buffer descriptor out of the spin

`create_buffer_resource_from_addr(peer_vec[rank] + elem*4)` with a lane-varying
`elem` makes the *descriptor* divergent, forcing a readfirstlane waterfall per
poll (§4.1). Build one uniform resource over the local inbox outside the wait
and address the per-rank slot with `offen`.

Sized honestly: 1,243,588 cy, **0.159%** of stall **[trace]**. A latency-floor
and readability fix, not a throughput lever. Only matters in combination with
§8.1 and §8.2, once the wait itself is short.

### 8.4 Give the wait something to amortize against (the `connStepCache` idea)

This is RCCL's biggest structural advantage (§6.3a) and the hardest to port,
because QRInt4's colour is *not* monotonic across ranks in a way a register can
exploit: each `(phase, block, src)` mailbox gets a fresh colour and there is no
"the peer is already 3 ahead" state to cache.

The transplantable version is to **make the flag a monotonic per-(phase, src)
counter rather than a per-iteration colour**, so that a block which has fallen
behind can satisfy several iterations from one read. That only pays when a block
processes many tile groups — i.e. at `ST=1`, or at large `num_tiles/grid`. In
the traced configuration each block does exactly one group, so it buys nothing
there. Note this as structure, not as a change to make today.

### 8.5 Flag-in-data (the LL idea) — the one with real upside

QRInt4's rank-tile is 18 × 64 B sectors: 16 INT4 + 2 E4M3 scale. An LL128-style
scheme would put an 8 B step tag in each 64 B sector and let `_recv_quantized`'s
own loads be the poll.

What it would delete, per publish:
- the `s_waitcnt vmcnt(0)` + `gpu.barrier()` join,
- the `buffer_wbl2 sc1` + second `vmcnt(0)`,
- the flag store itself,
- **the entire `_wait_release` and its trailing `gpu.barrier()`** — which is
  27.2% of stall on its own **[trace]**.

What it would cost:
- 8 B per 64 B = **12.5% more wire volume**. QRInt4 has room: it is already
  3.55× under RCCL's bf16 volume (§5).
- The 56 B/64 B split does not fit the current sector layout (1024 B nibbles +
  128 B scales, at 4 B/thread) without re-cutting it. This is a redesign, not a
  patch.
- **It rests on an assumption that must be measured on PCIe**: that a 64 B peer
  write is not torn or reordered such that a reader can see the tag without the
  data. RCCL relies on this on AMD at an 8 B granule (LL) and 64 B (LL128), but
  RCCL's LL128 is gated on `comm->topo->ll128Enabled`, and this note has not
  traced what that gate does over PCIe. **Verify before building.**

### 8.6 The structural one: N-way fan-in on a fabric with one uplink

Two-shot's wait is `max` over `N` arrivals; the ring's is over 1. On a fully
meshed xGMI node that costs little — all `N-1` destinations have private links
and arrive together. On PCIe, `N-1` destinations share one uplink, so the
arrivals *serialize*, and the max is over a serialized set. This is the same
structural argument the MI350P case study makes about bandwidth, applied to
latency.

That is a hypothesis consistent with "works on xGMI, does not on PCIe", and it
is exactly what the `qr_int4_ring_kernel.py` variant exists to test — its
`_wait` has one source and one polling thread
([:467-472](../aiter/ops/flydsl/kernels/qr_int4_ring_kernel.py#L467-L472)). If
the ring variant's *wait* time on PCIe is materially below two-shot's at the same
size, the fan-in is the mechanism. That comparison is cheap and does not need a
counter pass.

---

## 9. Open questions this document cannot answer

Carried forward from the annotation, unchanged, because none of them are
settled by reading source:

1. **Is the 18M-cycle reduce-scatter wait rank start skew?** Settle first, by
   either (a) a device-synchronized rank barrier immediately before the timed
   launch, or (b) capturing the same dispatch on all four ranks and comparing
   `realtime.json` wave-begin clocks. Everything in §8 is sized against the
   wrong denominator until this is known.
2. **Why does one uncached dword read of *local* memory cost 4.2k–17k cycles?**
   ATT times the stall; it does not say whether the cost is the bypass path, the
   `buffer_inv`, fabric queueing behind the peers' incoming writes, or
   contention among the 224 polling lanes. This run has **no PMC pass**, so none of it was
   measured. This is the one counter question worth the GPU-exclusive re-run.
3. **Does a 64 B peer write over PCIe give the ordering LL128 assumes?** Gates
   §8.5 entirely.
4. **What does `ll128Enabled` resolve to on a PCIe-attached MI350P?** If RCCL
   itself falls back to Simple there, the LL128 comparison in §7 is theoretical.

---

## Where this comes from

| Concept | Source |
|---|---|
| Stall attribution, poll cost, waterfall cost, barrier accounting | `annotations.json` in the run directory above — **our own capture** |
| QRInt4 inbox layout, publish, wait | [`aiter/ops/flydsl/kernels/qr_int4_kernel.py`](../aiter/ops/flydsl/kernels/qr_int4_kernel.py) |
| Ablation build used for the trace | [`op_tests/multigpu_tests/qr_ablation/qr_int4_kernel_ablate.py`](../op_tests/multigpu_tests/qr_ablation/qr_int4_kernel_ablate.py) |
| `s_sleep` backoff precedent in our own kernels | [`qr_1stage_kernel.py:369-379`](../aiter/ops/flydsl/kernels/qr_1stage_kernel.py#L369-L379), [`qr_int4_ring_kernel.py:467-489`](../aiter/ops/flydsl/kernels/qr_int4_ring_kernel.py#L467-L489) |
| PCIe peer-write measurements (fine-grained vs uncached, 64 B interleave vs contiguous) | [`op_tests/dump_data/docs/mi350p_peer_write_primitive_2026-09-07.md`](../op_tests/dump_data/docs/mi350p_peer_write_primitive_2026-09-07.md) |
| RCCL `waitPeer` / `postPeer` / roles / `loadStepValue` | [`prims_simple.h:115-237, 822-827`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/prims_simple.h#L115-L237) |
| `barrier_generic`, `NCCL_SPINS_BEFORE_CHECK_ABORT` | [`primitives.h:18-47`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/primitives.h#L18-L47) |
| LL line layout, `readLL`, `recvFlag` | [`prims_ll.h:67, 150-180`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/device/prims_ll.h#L150-L180); [`device.h:99, 152-162`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/include/device.h#L99) |
| Counters are local to the receiver; buffer allocation mode | [`p2p.cc:249-253, 569-577`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/transport/p2p.cc#L249-L253) |
| HDP flush skipped on gfx90a/942/950 | [`p2p.cc:393-402`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/transport/p2p.cc#L393-L402) |
| Push-not-pull on AMD | [`paths.cc:274-280`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/graph/paths.cc#L274-L280) |
| `NCCL_STEPS`, `DEFAULT_BUFFSIZE`, `ALLREDUCE_*STEPS`, `llProtoRanges` | [`device.h:53`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/include/device.h#L53); [`init.cc:1146`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/init.cc#L1146); [`collectives.h:22-24`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/include/collectives.h#L22-L24); [`tuning.cc:406-419`](https://github.com/ROCm/rccl/blob/ab6aee04d095f9a63c2911ecd5cc246b7eb39973/src/graph/tuning.cc#L406-L419) |
| CPol bit encoding (`GLC=1, SLC=2, DLC=4, SCC=16`) | [LLVM `SIDefines.h`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIDefines.h) |
| gfx950 fence/atomic lowering (`buffer_inv sc1`, `buffer_wbl2 sc1`, `sc0 sc1` loads) | **our own compiler experiment**, `hipcc --offload-arch=gfx950`, HIP 7.15.26333 |
| The ring algorithm itself, cost model, protocol crossover | [KB: RCCL ring all-reduce](../../claude-knowledge-base/gpu-collectives/10-algorithms/03-rccl-ring-all-reduce.md) |

---
[QRInt4 kernel](../aiter/ops/flydsl/kernels/qr_int4_kernel.py) ·
[ring variant](../aiter/ops/flydsl/kernels/qr_int4_ring_kernel.py) ·
[one-stage variant](../aiter/ops/flydsl/kernels/qr_1stage_kernel.py) ·
[MI350P peer-write primitive](../op_tests/dump_data/docs/mi350p_peer_write_primitive_2026-09-07.md)

# Synchronization, collectives, and failure diagnosis

## First distinguish the dependency types

| Edge | Ordering mechanism | Main failure mode |
|---|---|---|
| Within a wave | Instruction dependencies and correct active mask | Inactive lanes entering a collective |
| Across waves in a CTA | Appropriate memory completion plus CTA barrier | Only some waves reaching the barrier |
| Across CTAs in one launch | Explicit global publication/poll protocol | Stale cache lines, unscheduled producer, wrong epoch |
| Across ranks in one launch | IPC peer-memory protocol with system visibility | Unopened pointer, stale tag, remote producer never launched |
| Across kernels on one stream | Stream launch ordering | Hidden stream changes or freeing/reusing buffers too early |
| Across different streams | Explicit event dependency or another proved protocol | Assuming launch order across streams |

A CTA barrier is not a grid barrier. A coherent load is not by itself a release
or acquire operation. GPU-scope visibility is not automatically system scope.

## Two protocols that must not be confused

**Self-contained tagged payload:** K3 packs two BF16 values into the low 32 bits
of an aligned 64-bit word and an invocation tag into the high 32 bits. A producer
publishes that complete word; the consumer reloads the complete word until the
expected tag appears, then uses the payload from that same observed word. There
is no separately ordered payload in this message. The implementation uses
system-scope relaxed atomic stores for IPC messages and coherent 64-bit loads.
The absence of a separate payload is why this protocol can avoid a separate
release/acquire fence. Alignment, non-tearing access, and platform coherence
remain essential; do not generalize it to arbitrary devices or message sizes.

**Separate payload and ready flag:** A producer writes tensors, completes those
writes and any participating-wave coordination, then publishes readiness. A
consumer waits for readiness, then reads the tensor using a visibility mechanism
that observes the new data. Use correct release/acquire ordering or an equivalent
explicit target-specific protocol with a demonstrated ISA/memory-order proof.
Replacing acquire/release with relaxed atomics alone is incorrect. K3 uses
explicit store completion and coherent consumer loads; inspect `finish_stores`,
`publish`, `wait_ready`, and each producer, not just the poll function.

## Porting FlyDSL instructions to Gluon

Translate instruction semantics and register ownership, not syntax. For each
primitive record address width, access width/alignment, cache modifiers,
ordering/scope, wait counters, active lanes, and clobbers. Gluon inline assembly
can express instructions absent from its high-level API; compiler constraints
and layout conversion still control how many instructions execute.

The gfx950 K3 poll uses `global_load_dwordx2 ... sc1`, `s_waitcnt vmcnt(0)`, and a
comparison against the high tag bits. It saves/restores EXEC, handles inactive
lanes, bounds retry work, and marks assembly side-effecting (`is_pure=False`) with
memory, VCC, and SCC clobbers. Unique inline labels prevent collisions. A pure
expression or missing memory clobber can allow hoisting, elimination, or reuse.
Inspect generated ISA before trusting the source-level intent.

Consumer tensor loads in the frozen coherent variants use `.cv`. That is a
backend lowering choice, not a portable cache guarantee. Verify its emitted ISA
on the exact compiler and architecture. Ordinary cached loads remain useful for
immutable weights; indiscriminately bypassing caches can lose performance.
Repeated ordinary loads of semaphore memory are not a valid polling design.

K3's compiler build exposed Gluon atomic_load/atomic_store; the stock experiment
image did not. Check the installed API and compile target before adopting it.
Do not add a compatibility try/except that silently changes memory semantics.

## IPC and all-reduce

Allocate/export/open HIP IPC memory once per owning rank, exchange opaque
handles over a CPU group, and keep the allocations and imported mappings alive
through every graph replay. Peers need mapped device addresses, not another
process's raw virtual address. Check peer topology and access support when
hardware work is authorized. Destroy graphs/users before closing handles.

The K3 small-message primitive broadcasts each rank's tagged contribution into
peer destination slots. Each destination polls the rank slots and sums them in a
fixed rank order. This is direct peer exchange/reduction; do not describe it as a
reduce-scatter followed by all-gather without inspecting the code. It trades
extra traffic for low latency and fusion at small tensor sizes. At larger sizes,
benchmark against a staged collective or the existing communication library.
A better large-message collective need not be the best fused decode primitive.

Start with rank-distinguishable inputs and compare every output against a CPU
sum or a trusted collective. Exercise actual offsets, alignment, masks, peer
count, repeated invocations, and alternating payloads. Measure the slowest rank.
A single successful transfer does not prove graph-replay safety.

## Epochs, buffering, and overwrite hazards

Increment the invocation epoch exactly once per logical fused region. K3's
hybrid front advances it; the tail reuses it (`BEGIN=False`). Give independent
layers/batch specializations independent workspaces unless reuse has a proof.

A matching tag distinguishes old data but does not prevent a fast producer from
overwriting data before a slow consumer reads it. Document why reuse is safe:
acknowledgment, dependency ordering, enough bounded buffering, or a proven
lockstep schedule. Double buffering alone is insufficient if arbitrary rank
skew is possible. Define initialization and tag wraparound behavior; do not reset
live flags from the host while kernels or captured graphs can still use them.

Keep input-dependent values dynamic through capture. Repeat identical inputs to
find epoch problems, then change inputs and route patterns to expose stale reads.
Churn the allocator before replay to expose accidental tensor lifetime reliance.

## Diagnose timeouts by separating causes

1. Identify the exact waiting rank/CTA, stage, expected epoch, observed tag,
   and producer. Preserve a bounded error record; a timeout is not a diagnosis.
2. Check whether all ranks launched. Lazy JIT or model preparation can leave
   one rank polling while a peer is still compiling. Compile/warm every variant
   before timing, synchronize streams, and align ranks with a CPU barrier.
   Also align the first real eager invocation: profiling may prepare a variant
   long before attention metadata makes it executable.
3. Check residency for the compiled variant. Spinning consumers can occupy all
   CUs while producers remain queued. Register/LDS changes can invalidate a grid
   that fitted before; source-level thread counts are not occupancy evidence.
4. Check pointer mapping, peer rank indexing, allocation lifetime, epoch reuse,
   buffer offsets, and masks. Recheck all TP-specific formulas after changing TP.
5. Check stale polling and stale payload reads separately. Correct flag visibility
   does not prove correct tensor visibility. Inspect the actual load instructions.
6. Check missing writes, failed producer branches, and duplicate writers. A
   watchdog branch must avoid using uninitialized metadata or reporting success.

Use a measured device clock conversion for time-based watchdogs. Iteration limits
have architecture- and load-dependent durations. Increasing a watchdog may help
collect evidence but is not a fix for deadlock, stale data, or missing launches.

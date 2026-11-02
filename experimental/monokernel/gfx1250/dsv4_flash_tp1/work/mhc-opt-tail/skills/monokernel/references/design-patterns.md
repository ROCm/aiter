# Monokernel design patterns

Generic patterns for persistent multi-GPU decode kernels. Examples cite the Kimi-K3 implementation in vLLM
(`vllm/models/common/mono/` for framework plumbing, `vllm/models/kimi_k3/amd/ops/mono/` for the kernel,
`vllm/models/kimi_k3/amd/glue/` for the hybrid glue) where it helps; file names are illustrative.

## Contents
1. Persistent grid and stage layout
2. Stage walk vs task table
3. Work decomposition and load balance
4. Prefetch before poll; streaming depth
5. Scratch and peer regions
6. Mailboxes: tagged pairs vs payload + flag
7. Cross-GPU collectives inside the kernel
8. Routing and expert scheduling (MoE)
9. Row-count specializations
10. Fail-stop and watchdogs
11. Build keys, ABI and tracing hygiene
12. What did not work (so you do not repeat it)

## 1. Persistent grid and stage layout

- Grid = number of CUs, one CTA per CU (e.g. 256 CTAs x 512 threads, `num_warps=8`, `waves_per_eu=1`, all of a
  CU's LDS per CTA). Refuse to run when the device reports a different CU count (partitioned GPUs): a spin wait
  progresses only if its producer is resident.
- Stages are CTA ranges inside one body. Place stages that can start early (dense GEMVs whose inputs are ready) on
  CTAs that the critical-path stage does not use, so they overlap it: e.g. all-reduce reducers on CTAs 0-55, the
  o_proj on 56-255, the router and latent GEMVs on other ranges that start as soon as the reduced activations
  appear.
- The heaviest branch of the body sets the register and LDS budget of every role. Move cold code behind an
  `unlikely()` branch and keep kernel-lifetime values out of hot registers (an opaque `v_mov` "fresh()" helper
  stops LLVM from hoisting / CSE-ing values into long-lived registers that then spill).

## 2. Stage walk vs task table

- Stage walk (reference style, recommended): every CTA walks every stage in program order and does its bid-indexed
  share; dependencies are tag/flag waits on exactly the data it needs. Phases inline into one function.
- Task table: a static list of (kind, args) tasks dealt round-robin to CTAs with `noinline` phases. Pitfalls seen:
  must be topologically ordered or a CTA waits on itself; position aliasing mod grid size decides which earlier
  task gates a CTA; each task hop costs ~3 us (callee-saved register spill/restore, `s_waitcnt` at return,
  task-entry loads). Moving to a stage walk took a kernel from 168.8 to ~91 us at S=1 after tuning.
- The stage walk shares one register budget across all stages; at larger row counts it spilled (438 spills, then a
  memory fault). Keep per-row-count entries or `noinline` wrappers where inlining does not fit.
- If your custom scheduler has no measured advantage over the reference design, adopt the reference's.

## 3. Work decomposition and load balance

- Count dependent memory hops on the critical path first (each ~1-3 us); remove hops before tuning compute.
- Size work units to the grid at the realistic distribution. Example: 12 tasks per active expert x 24 experts = 288
  tasks on 256 CTAs gave 32 CTAs two tasks and 224 one: ~1.8x critical path and ~3 us measured. Pick unit counts
  that are multiples of the grid, split remainders (K-split with cross-CTA partials), or use finer columns (64-column
  expert tasks with K halves split across wave halves).
- Dynamic task counts: reversed placement (`task = BLOCKS - 1 - bid`) puts leftovers on CTAs that finish their early
  work first; issue the next task's loads before computing the current one.
- Rendezvous points wait for the slowest producer: spread tiles evenly so producers finish together (a 6 us spread
  in expert-up completion became 5.5 us of waiting in the down stage).
- Move serial work off the critical path: one otherwise idle CTA computes the top-k and publishes it as tagged
  (expert, weight) pairs, instead of every expert CTA recomputing it (saved ~2-5 us).
- Fold small stages into neighbors: e.g. down-projection tasks publish sums of squares and the final reducer applies
  the norm, removing a single-CTA norm stage.

## 4. Prefetch before poll; streaming depth

- The main lever in decode: issue most of a task's weight bytes before spinning on its activation input. Per-CU
  streaming has ~4 us fixed latency and ~28 GB/s per CU, so a 458 KB task needs ~300+ KB in flight to approach the
  floor.
- Register prefetch is capped by the VGPR budget (~256 KB per CTA at 248 VGPRs); async global->LDS copies add depth
  (to 320 KB here: 105.3 -> 97.6 us).
- Vector memory completes in order: a poll issued behind a weight burst waits for the burst; publish stores queue
  behind it too. So: publish, then prefetch; keep polling CTAs free of in-flight bulk loads; or keep the bulk in LDS.
- Prefetch only on CTAs that use the data: an unconditional prefetch made every CTA load the up-proj weights (23 MB
  of useless traffic).
- Bytes matter less than keeping loads in flight: a reference stage with half the bytes (MXFP8) but no prefetch was
  slower than bf16 with deep prefetch.
- Hold-back delays (delaying a weight burst so it does not starve a concurrent stage) are row-count specific: 0 at
  S=1, 7 us at S>1.

## 5. Scratch and peer regions

- Per rank: one device scratch buffer (sized to the max over all builds, shared by all launches, distinguished by
  tags), one uncached IPC peer buffer, an epoch counter, health words. Allocate at attach time, outside graph capture.
- Compute the region layout once on the host (8-byte pairs, constexpr offsets in the kernel). Derive every size and
  stride from the maximum rows; assert non-overlap. A region sized for 8 rows overflowed at 16 rows into the next
  one and wrote valid tags there.
- Peer regions as `[source rank][token][column]` so each writer owns disjoint slots; double-buffer by epoch parity
  (`base = (epoch % 2) * half`) so a rank one step ahead writes the other half.
- Reuse rule per region, written down: a rank may rewrite a peer's region for step N+1 only after that peer has
  finished reading step N (proved by a later message from that peer).

## 6. Mailboxes

- Tagged pair: (value, tag) in one 8-byte store; the reader re-polls until all tags match. One round trip, no fence,
  no flag. Tag = (epoch % 2^22) * 256 + slot + 1 with a slot per (kernel, layer, stage); the epoch is a device
  counter bumped once per forward inside the captured graph. Nothing is ever zeroed.
- Payload + flag: store payloads, drain (`s_waitcnt vmcnt(0)`), CTA barrier, then one thread stores the flag. A
  workgroup-scope LLVM release does not drain stores outside tgsplit mode, so do the drain explicitly. Use for bulk
  data where tags would double the bytes (at 240 rows tags cost ~100 us of extra XGMI traffic).
- Memory scope by space: device scratch at device scope (sc1), peer memory at system scope (sc0 sc1). Polls are
  side-effecting asm with memory clobbers (see `amd-gpu-kernel-perf` memory-and-sync).
- Integer payloads must not round-trip through f32 (NaN canonicalization changes bits).

## 7. Cross-GPU collectives inside the kernel

Choose by size:
- One-shot push (small, <= ~16 rows): each producer tile stores its bf16 partial into every rank's `[src]` slot
  (one vectorized store over [peers x elements] with peer bases preloaded); each owner polls 8 slots and sums in
  fixed rank order in fp32, so results are bit-identical on all ranks.
- Two-shot (wide): reduce-scatter to column owners, owners sum and raise per-tile flags, readers pull (all-gather by
  pull; one CTA pushing 32 KB was ~10x slower).
- Direct pull (hybrid): a library GEMM writes its partial straight into this rank's peer buffer via `out=`; one CTA
  raises ready flags on all ranks; owner units pull [rows x 64 cols] tiles from all ranks (128 B segments, 224 small
  units beat 112 large).
- Why the protocol changes with rows: a one-shot broadcast sends each rank's partial to all 7 peers. At 16 rows
  (~230 KB per rank) that is cheap and saves a hop; at 240 rows (3.4 MB per rank) it would put ~24 MB on the fabric
  per rank, ~48 MB with tags. Reduce-scatter + all-gather moves ~2 x 7/8 x 3.4 MB per rank instead, with untagged
  bf16 data and one epoch-tagged flag per tile.
- Remote atomics instead of the reduce-scatter (each rank atomically adds its slice into the owner's buffer): same
  bytes as a push reduce-scatter, saves the owner's read round trip, but needs a zeroed destination every step, a
  per-tile arrival count (a flag again), and either bf16 accumulation in arbitrary order (precision loss,
  run-to-run nondeterminism) or fp32 atomics (2x bytes plus a conversion pass); the all-gather is still required.
  Remote atomic throughput over XGMI is unmeasured here, and local fp32 atomics for a split-K reduction measured
  187 us, so treat it as an experiment, not a default. Local atomics are fine where a library already uses them
  (MoE stage 2 accumulates with bf16 atomics into a zeroed buffer).
- Sharding tricks that remove collectives: compute router logits 1/TP per rank and all-gather the small fp32 logits
  so every rank computes the same top-k; split an up-projection along K by the owner of the latent slice so the
  latent all-gather disappears and the partial joins the final reduce-scatter; replicate small GEMVs on every rank
  when the extra bytes are cheaper than a hop (evaluate; one external design replicated ~86 MB per rank).

## 8. Routing and expert scheduling (MoE)

- Every rank must compute the identical top-k: same logits (all-gathered), same tie rule (lower expert id), same
  ordering (by score descending, matching the library's slot order), same renormalization order. NaN logits rank
  last.
- In-kernel top-k: radix select by a 32-step ballot/popcount search works well for hundreds of experts.
- Expert union: flag array + exclusive scan (or a 1024-bit bitmap with popcount prefixes) gives slot_of[expert],
  expert_of[slot], and per-slot token lists; dedupe experts across tokens (an external design that ran one task per
  (token, expert) wasted ~15/16 of its MMA).
- Down stages poll each pick's intermediate at its point of use so finished slots overlap later ones; accumulate in
  each token's top-k order with the reference rounding.
- For hybrids, the routing kernel emits exactly the library's ABI (e.g. ids int32 [S, k], weights fp32 [S, k]).

## 9. Row-count specializations

- Widths follow speculative decoding: whole requests only, widths = multiples of (k + 1).
- Design per bucket. When a backend is capped at R rows, either run R-row groups with separate scratch copies inside
  one launch, or two launches split on whole requests with a second epoch range (offset 2^21) so tags never alias.
- Re-check spills, region sizes and load balance for each width; delays and prefetch depths are width-specific.

## 10. Fail-stop and watchdogs

- Bound every wait by wall-clock time (`s_memrealtime`, 100 MHz; e.g. 10 s). On the first give-up: write an 8-word
  record (region id, index, expected tag, seen tag, CTA, clock), store an alarm word to rank 0, drain, set a sticky
  FAILED state. Every later wait exits immediately when FAILED, so the step finishes instead of hanging.
- Host: copy health words next to the sampled token ids each step and raise before tokens are returned; count
  incidents by rank. Provide fault-injection fields (rank, epoch, layer, delay) and a test.
- Do not use iteration caps as the timeout mechanism; they change codegen and can hide races. Keep them debug-only.

## 11. Build keys, ABI and tracing hygiene

- The JIT keys only what it sees; make build parameters a frozen scalar dataclass whose key includes a digest of
  all kernel sources (a missing key component once made every rank run rank 3's kernel).
- Check the kernel's and the launcher's signatures against a declared ABI and pack arguments by name.
- Type mailbox addresses (region, memory space) and keep a host-side trace of (stage, region, put/poll) so raw
  address accesses and region mix-ups are caught before running.

## 12. What did not work (examples)

Expert-up placed on the CTAs that free up first; pipelining down by expert group; a 3-chunk prefetch that starved a
concurrent stage; K-split latent-down doubling CTAs; replicated route tables; chunk rotation; epilogue overlap with
down prefetch; dword scale loads; a 64-column M-stacked path (correct, no faster); dot2 inline asm next to async
copies (wrong results). Record yours the same way with numbers; most "obvious" micro-optimizations were neutral.

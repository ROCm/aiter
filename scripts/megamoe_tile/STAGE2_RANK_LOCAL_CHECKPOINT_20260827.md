# Stage2 rank-local optimization checkpoint (2026-08-27)

## Session and environment

- Recovered main session: `01a03dbb-8a0b-79a1-b93e-520dc5dda021`.
- Local tree: `/home/zihuang/work/aiter-mega-tile-pr`, branch `dev/mega_tile`.
- Remote containers: `mi355-gpu-{46,50}:hzm_work:/home/hzm/aiter`.
- The worktree was already dirty before this continuation; do not reset it.
- Node 50 has an unrelated idle SGLang TP8 service retaining about 91% VRAM.
  Node 46 also became externally occupied later in this continuation. Do not
  terminate either service.
- `run_stage2_breakdown_ep16.sh` now preserves an externally supplied
  `MORI_SHMEM_HEAP_SIZE`; its default remains 40G. An 8G override is enough for
  the memory-light MORI comparison while SGLang is resident.

## Recovered baseline and retained uniform reducer

The pre-uniform active-queue best was:

```text
grid208 / reducer16 / final4 / compact16 / vec8 / interleaved
rank-max mean 1595.705 us
all-rank mean 1510.012 us
```

The retained uniform-address changes are:

- explicit wave-uniform peer bases and queue metadata;
- scalar row offsets for peer reducer loads;
- `load_first` reducer scheduling support;
- the packed-BF16 atomic scalar-soffset experiment was rejected and reverted.

Initial same-run screens at grid208:

| Reducer | rank-max mean | all-rank mean |
|---|---:|---:|
| uniform interleaved | 1540.307 us | 1427.281 us |
| uniform load-first | 1521.344 us | 1416.153 us |

The reverse long run also favored load-first in the mean, but both paths had
large co-tenant outliers. Keep `load_first` as the experimental baseline, not a
public default, until an idle-node rerun.

Correctness after the uniform change:

- 122 Stage2 contracts passed at that point.
- paired 4-generation validation: protocol errors 0, max rel-L2 0.004944.
- arbitrary TopK 4-generation validation: protocol errors 0, max rel-L2
  0.002507.
- active-only queue stress passed 18 generations with local route counts 0..16
  and poisoned parity reuse.

## Grid and MORI results

At grid224, reducer16/final4 measured 1547.805 us rank-max and 1443.129 us
all-rank, versus 1615.125/1509.040 us for the adjacent grid208 run. Reducer8
at grid224 measured 1572.338/1478.415 us and did not win.

Grid240 hung in the kernel-wide software barrier and was terminated. Later,
even grid224 and grid208 could hang while unrelated co-tenant kernels were
resident. Use grid176 while the nodes are shared; only retry grid208/224 when
both nodes are genuinely idle. Never use grid240/256 for this kernel without a
new residency proof or removal of the full-grid barrier.

With memory-light packed weights and an 8G MORI heap under the shared-node
conditions:

```text
fused rank-local load-first  1705.261 / 1585.337 us
MORI GMM2 + combine          1718.738 / 1630.420 us
```

This same-run fused win is provisional because the co-tenant environment is
noisy. The clean historical MORI target remains 1382.874/1272.023 us.

## Dynamic reducer head and capped GMM rejoin

Implemented as experimental, default-off controls:

```text
node_reduce_work_schedule = static_strided | dynamic_head
node_reduce_rejoin_blocks = 0 | 8 | 16 | 32
```

The ABI appends a parity-local, cache-line-padded
`rank_reduce_queue_head`; all previous offsets remain unchanged. Fixed dynamic
reducers and capped post-GMM helpers share this head. Helpers first wait for
`queue_tail >= active_count`, then still acquire each generation-ready slot.
There is no new full-grid barrier and final CTAs never rejoin.

Validation:

- full Stage2 contract suite: 128 passed before vec16, 135 passed after vec16.
- static, dynamic0, and rejoin8 all lower at
  `183 VGPR / 100 SGPR / 28,992 B LDS / 64 B private`.
- paired dynamic/rejoin8: protocol errors 0, max rel-L2 0.005051.
- arbitrary dynamic/rejoin8: protocol errors 0, max rel-L2 0.004083.
- 18-generation route-count 0..16 + arbitrary + poison stress passed,
  including exact dynamic queue-head accounting.

Performance at grid224/final4/compact16/vec8/load-first:

| Variant | rank-max mean | all-rank mean |
|---|---:|---:|
| static A | 1528.686 us | 1399.358 us |
| dynamic head, rejoin0 | 1511.956 us | 1392.728 us |
| dynamic head, rejoin8, r16 | 1761.855 us | 1600.597 us |
| dynamic head, rejoin16, r16 | 1589.282 us | 1481.916 us |
| dynamic head, rejoin8, r8 | 1533.083 us | 1420.521 us |
| static B | 1507.225 us | 1404.873 us |

Conclusion: dynamic-head alone is approximately neutral; GMM rejoin is a
regression. Keep both as experimental controls and retain static/rejoin0 as the
default.

## Resource attribution

The complete uniform resource matrix is under:

```text
/home/hzm/profiles/stage2_ranklocal_resource_matrix_uniform_20260827/
```

| Diagnostic mode | VGPR | SGPR | LDS | private |
|---|---:|---:|---:|---:|
| init_only | 81 | 96 | 28,992 B | 0 |
| atomic_only (zero GMM + epilogue + reducer) | 81 | 96 | 28,992 B | 0 |
| gmm2_only | 162 | 96 | 28,992 B | 0 |
| route_store_only (GMM + rank atomic/publish) | 182 | 96 | 28,992 B | 0 |
| gmm2_atomic_only | 183 | 100 | 28,992 B | 0 |
| return_only | 81 | 96 | 28,992 B | 64 B |
| full | 183 | 100 | 28,992 B | 64 B |

Therefore the extra VGPRs come primarily from the rank-local GMM epilogue,
not cross-role reducer/RAIL liveness. The 64B private segment and 15 scratch
loads/4 stores come entirely from MORI CCO `quietUntil`'s device-assert call
chain in the return role.

Latest measured decomposition at grid224 (shared/noisy nodes):

| Mode | rank-max mean | all-rank mean |
|---|---:|---:|
| init_only | 599.335 us | 423.188 us |
| gmm2_only | 813.341 us | 616.314 us |
| GMM + rank atomic/publish | 1147.260 us | 925.238 us |
| zero-GMM atomic + reducer | 750.629 us | 644.170 us |
| GMM + atomic + reducer | 1425.382 us | 1192.972 us |
| return + final | 927.877 us | 725.104 us |
| full | 1595.437 us | 1444.697 us |

## Vec16 reducer experiment

Implemented as an opt-in rank-local-only value of
`node_reduce_vec_bytes=16`. It uses 32 active lanes x 8 BF16 with safe masked
addresses. Defaults remain unchanged.

- Compile succeeds for interleaved and load-first.
- Resources remain `183/100/28,992/private64`.
- 56 `buffer_load_dwordx2` become 56 `buffer_load_dwordx4`; the static memory
  instruction count is not reduced and ISA size grows 15,042 -> 15,387 lines.
- Paired real-GMM validation passes: protocol errors 0, max rel-L2 0.005063.
- One noisy grid176 sandwich gave:

| Variant | rank-max mean | all-rank mean |
|---|---:|---:|
| vec8/load-first A | 1672.680 us | 1569.193 us |
| vec16/interleaved | 1599.178 us | 1510.080 us |
| vec16/load-first | 1644.244 us | 1496.987 us |
| vec8/load-first B | 1568.417 us | 1461.788 us |

The signal is too small/noisy to promote vec16. A reverse-order run was blocked
by co-tenant-induced resident-grid hangs. Keep vec16 experimental and default
vec8.

## Next implementation plan

Use the configured agent roles after restarting Codex:

- planner: `gpt-5.6-sol`, effort `ultra`;
- worker: `gpt-5.6-luna`, effort `high`.

The next isolated code experiment is rank-local epilogue dynamic LDS addressing:

1. Replace eight long-lived dynamic C-shuffle row addresses with one dynamic
   base plus Python compile-time element offsets.
2. Require route-store-only VGPR to fall by at least four (target <=176), with
   32 `ds_write_b16` and 64 packed atomic sites unchanged.
3. Require full resources no worse than 183/100/28,992/private64 and no new
   scratch.
4. Only then run paired/arbitrary/stress and sandwich performance.

Second, independently test a rank-local full-wave epilogue (64 lanes x 4 BF16,
two packed atomics/lane) to halve static atomic instruction sites from 64 to
32 per two-N-tile group. Do not combine it with dynamic-base until each is
individually attributed.

If neither yields a stable >=1% improvement, implement the designed m-block
completion tree, reducing completion atomics from about 28,672/rank to roughly
2,944-3,616/rank while keeping the existing rank-pending and ready protocol.

## Resume checklist

1. Confirm the new Codex session loaded project agent roles successfully.
2. Confirm no `bench_megamoe_tile`/`torch.distributed` process remains on 46/50.
3. Check GPU memory and utilization; use grid176 while co-tenant services are
   present.
4. Start with the dynamic-base implementation and ISA gate above.
5. Keep all new scheduler/vector modes opt-in until clean-node long runs pass.

## Dynamic-base follow-up (worker role, 2026-08-27)

The configured worker implemented an opt-in
`rank_epilogue_lds_addressing=expanded|dynamic_base` switch. The default is
`expanded`; dynamic-base is restricted to rank-local, vec8, load-first,
static-strided, rejoin0. It changes only the weighted C-shuffle write loop:
`(row_base+v)*BN+col` becomes one dynamic `row_base*BN+col` plus constexpr
`v*BN` offsets. ABI, LDS layout, barriers, reads, atomics, and ready protocol
are unchanged.

Node50 gfx950 compile gate:

| Diagnostic | expanded | dynamic-base |
|---|---:|---:|
| route_store-only VGPR | 182 | **174** |
| full VGPR | 183 | **177** |
| SGPR | 96/100 | 96/100 |
| LDS | 28,992 B | 28,992 B |
| private | 0/64 B | 0/64 B |
| `ds_write_b16` | 64 | 64 |
| `buffer_atomic_pk_add_bf16` | 64 | 64 |

The dynamic path passes:

- 135 Stage2 contract tests;
- paired 4-generation real-GMM validation, protocol errors 0, max rel-L2
  0.005313;
- arbitrary TopK validation, protocol errors 0, max rel-L2 0.005672;
- 18-generation local-route-count 0..16 + arbitrary + poisoned-parity stress,
  status pass on every rank.

Shared-node grid176 performance was not sign-off quality because the two
expanded sandwich endpoints drifted from 1.659 ms to 2.387 ms rank-max while
dynamic-base measured 1.645 ms. Keep dynamic-base opt-in and rerun an
expanded/dynamic/expanded sandwich when both nodes are idle. Do not promote it
from this noisy run alone.

The current project config is now valid for the installed Codex CLI and maps
`planner.toml` to Sol Ultra and `worker.toml` to Luna High. A new session is
required to reload those role layers.

## Staged-reduce groundwork (2026-08-27)

Planner evaluated the proposal to replace per-element GMM packed BF16 atomics
with per-route staging plus a small completion counter. CTA/work-index staging
was rejected because persistent queue reuse would require roughly 469 MB per
parity and has unsafe lifetimes. The safe first layout is keyed by
`(source, topk_slot, n_group)` and is about 448 MiB per parity (roughly 896 MiB
for both parity buffers), plus counters. The earlier 32 MiB estimate omitted
the 14 n-group dimension.

Worker added the optional ABI skeleton (`rank_stage_values`, slot generation,
group/tile counters) and threaded `rank_accumulation_mode=atomic|staged_reduce`
through the compiler, factories, benchmark, validation, stress, and contracts.
The runtime staged path is intentionally rejected at compile time until the
last-contributor FP32 reduction and system-release publication are implemented:

```text
ValueError: staged_reduce is ABI-only and not yet enabled; use atomic
```

This keeps the default atomic path unchanged. The default atomic path lowered
successfully on gfx950 with `183 VGPR / 100 SGPR / 28,992 B LDS / 64 B private`,
and the complete Stage2 contract suite remains `135 passed`. Do not enable
`staged_reduce` in performance or correctness runs yet.

The next implementation step is the last-contributing GMM CTA prototype:

1. stage each weighted route tile by `(source, topk_slot, group)`;
2. decrement the source/group pending counter after staging release;
3. when the last route arrives, scan valid Top-K slots and FP32-reduce into
   `rank_accumulator`;
4. after all groups complete, reuse the existing `rank_ready`,
   `node_partial_done`, and reduce-queue publication chain.

It must preserve duplicate-source handling, reject duplicate `(source,slot)`
overwrites, use system release before peer-visible `rank_ready`, and be tested
separately on unique-route and duplicate-heavy fixtures. Keep atomic as the
control and do not combine staged-reduce with dynamic-base, vec16,
dynamic-head, or rejoin in the first A/B.

## Dedicated CTA reduce/send design (paused for next session)

The user proposed moving the reduce/send work into a dedicated CTA. Planner
compared two designs:

- **A: local stage-reducer CTA.** GMM CTAs stage route tiles and enqueue completed
  `(source, group)` tasks. A dedicated local CTA reduces valid Top-K staging
  slots into `rank_accumulator`; the existing source-proxy peer reducer and
  RAIL send path remain unchanged. This is the safest correctness oracle, but
  needs a tile queue (worst case 28,672 tasks/parity, about 115 KiB), producer
  completion/EOS handling, and an additional resident role.
- **B: direct stage pull.** Keep the current `rank_pending → rank_ready →
  partial_done → reduce queue` protocol and change `reduce_rank_queue_slot` to
  load each peer's route staging directly, reducing into node output/return
  buffers. This avoids a new CTA/queue but can require up to 16 LSA loads per
  tile and is likely worse for dense duplicate routes.

Recommended resume order is A0 (last-GMM CTA reduce as the correctness oracle),
then A1 (dedicated local reducer CTA only if GMM tail/register pressure is
unacceptable), with B0 as a separate direct-pull control. Do not remove the
existing `rank_accumulator` ABI in the first experiment.

Do not move CCO `put/flush/credit` into every reducer CTA: role 0 remains the
single QP/RAIL owner. Also note that the existing rank-reduce queue is emitted
only after all expected ranks' local completion; a dedicated local stage
reducer therefore needs its own stage-tile queue/EOS, while direct-pull B can
reuse the existing source-token queue.

Current staged runtime status when paused:

- ABI/staging regions, pointer lowering, tile-counter shape (28 tiles/source),
  and system-release ordering have been fixed locally.
- Staged route/full lowering artifacts were generated before the final shape and
  release fixes; recompile after resuming is required.
- Default atomic path is unaffected and remains validated by 135 contract tests.
- Staged runtime has not yet passed paired/arbitrary or poison correctness;
  do not benchmark or promote it until those pass.

Session to resume: `01a041cb-107c-7612-a6df-08eb5323c56a`.

## Staged runtime lowering / current blocker

After the pointer, tile-shape, and system-release fixes, staged route/full
lowering was generated on node50. The staged full artifact still uses
`183 VGPR / 100 SGPR / 28,992 B LDS / 64 B private` and is a very large scalar
correctness prototype. The default atomic path remains `135 passed` and lowers
normally.

The first staged paired validation did not enter the kernel: MORI
`ccoMemAlloc` failed while allocating the arena. Retrying with
`MORI_SHMEM_HEAP_SIZE=8G` also failed at arena allocation while node46 was at
84% VRAM occupied by the unrelated service. This is an external resource
blocker, not a staged numerical/protocol failure. Do not interpret it as a
correctness result; rerun staged paired/arbitrary only when at least one
full two-node allocation window is genuinely available.

The dedicated-CTA recommendation remains:

1. A0: last GMM contributor reduces staged source/group tiles, existing peer
   reducer and role-0 RAIL remain unchanged;
2. A1: dedicated local stage-reducer CTA with its own tile queue/EOS if A0
   creates unacceptable GMM tail/register pressure;
3. B0: source-proxy reducer directly pulls peer staging as a separate control.

Do not move CCO put/flush/credit into every reducer CTA, and do not benchmark
staged until arena allocation and correctness both pass.

The latest v3 handshake retry reached direct reference successfully, then
staged rank failed in `ccoMemAlloc` before any kernel launch. This is still an
external arena-allocation blocker, not a staged correctness result. The
watchdog and all remote staged processes are now stopped.

The arena-size investigation showed the deeper issue: the current staged
`rank_stage_values` shape is `(parity=2, source=2048, topk=16, groups=28,
BF16[512])`, about 939,524,096 B (896 MiB) before the other regions, making the
full Stage2 layout about 1.184 GB. It still fails `hipMemCreate` on completely
idle nodes, so reducing `MORI_SHMEM_HEAP_SIZE` is not a sufficient fix; the
single symmetric VMM allocation shape is the blocker.

The next implementation should replace full-lifetime staging with a bounded
ring and a dedicated local stage-reducer CTA:

- route tile payload is 512 BF16 values (1 KiB) per item;
- ring capacity is bounded by resident GMM CTAs (roughly 16 MiB/parity,
  32 MiB for both parity buffers);
- producer uses sequence/head-tail backpressure and an EOS/expected-task
  counter;
- stage reducer consumes items, accumulates source/group tiles, writes the
  existing `rank_accumulator` with system release, and only then publishes
  rank-ready/partial-done/queue;
- existing cross-rank reducer and role-0 RAIL remain unchanged.

Do not retry the current 1.184 GB staged arena. Implement the ring/CTA variant
as a new opt-in mode, keeping atomic as the control.

## Staged correctness watchdog

`scripts/megamoe_tile/watch_stage2_staged_correctness_ep16.sh` is deployed as a
one-shot watchdog on both nodes. The v1 attempt had a single-sided rendezvous
timeout; v3 added a two-way TCP handshake and synchronized both nodes, but the
staged arena still failed at CCO allocation. The watchdog now defaults to an
8G MORI heap, requires all eight local GPUs to report no KFD PIDs and <=5% VRAM,
and never kills or modifies external services. It should be relaunched only
after the arena allocation issue is resolved.

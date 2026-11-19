# Fine S1F probes: three pre-launch reviews

1. Address/ownership: one record per M64 producer wave; one per (M,N,wave)
   consumer task. Trace arrays sized from compact capacity. No shared log head,
   no new atomics, no LDS allocation. Only lane0 writes its disjoint records.
   Unused pull inp_scale argument carries trace pointer; runtime args checked.
2. Synchronization/semantics: no new wait or barrier. Clock is wave-local
   s_get_shader_cycles_u64 with compiler memory clobber. Never subtract clocks
   from different waves/TGs. Producer sample is the first M16 of each M64;
   its timestamps are written only after actual readiness publication. Earlier
   producer trace stores can affect later store waits; report perturbation.
   Consumer probes add timestamp stores inside hooks; overhead measured versus
   original with identical E2E harness. Fine intervals are issue spans unless
   bounded by an existing wait. No claim that scale/payload stores have retired.
3. Launch: reuse EP4 changing-input correctness first, then one full balanced
   tpr16384 run, all same CSV/roles. External-ready, TILE_READY0, P96/C160,
   no experimental scheduler/clusters. rocm-smi idle gate before every launch,
   1s polling, 240s process timeout; use shared gpu-lock if present. No foreign
   process termination. Source snapshots and checksums retained per run.

Source-shuffle launch review after reboot: the first committed test exposed a
null metadata pointer in _plan_args under external-ready. Fixed it to pass
allocated tile_storage when src_rank_shuffle is set; a host-side test assertion
checks the argument before any plan launch. Arena/epoch/row bounds re-reviewed;
existing padding ownership and remote readiness ordering are unchanged. Processes
exited after the fault, boot ID unchanged and no KFD PIDs remained.

Wave-load three-pass review: (1) one ordinary2D TDM covers four consecutive
M16 source frames per wave/K segment; source grouping guarantees one owner;
frame pitch43520B, payload width16384B for1024B K payload, last width10752B
includes scales. All source and LDS offsets are bounded,128B aligned. (2) one
wait follows each K-segment load before any LDS read; there are no M16 waits.
Payload/scale stores retain existing per-wave store wait and A-ready publication.
For4-buffer planar GEMM, producer leaves a64KiB hole [128KiB,192KiB), and control
uses only the intersection with the GEMM's existing unused gap. (3) run small
then full changing-routing correctness, compare native E2E and HIP events with
baseline, keep GEMM2 unchanged. K can be reduced for smaller LDS constraints.

Linear GEMM2 three-pass prelaunch review:
1. Source-home slot bound = ceil((max_routes + global_experts*255)/256)*256.
   Each source expert segment owns a disjoint padded extent. Plan publishes a
   delta for each expert, a remote M256 landing base, and a local route-to-slot
   map. Original rowmap/weights are unchanged. All byte addresses widen to i64.
2. The existing plan completion and external-ready protocol covers the new map
   stores. GEMM2 retains its LDS barrier and per-wave tensor wait; each wave
   writes M64 including owned padding. Descriptor tensor width excludes the
   padded LDS columns, so adjacent N tiles cannot overwrite one another.
   Combine retains the cross-rank barrier and original top-k summation order.
3. Feature defaults off, requires source shuffle and bf16 fused combine.
   Run small then full changing-input correctness before timings. Push reference
   explicitly disables linear store. Idle gate and boot-ID timeout remain.

Unique pull three-pass review (2026-11-19):
1. One wave owns one (source rank, M16 source-token interval); ownership is
   disjoint across producer slots. The symmetric route map has max_routes+1
   words, with a current-token-count header. Invalid tail lanes never issue a
   payload read; all route destinations come from the existing bounded plan.
   LDS has four M16 slices at the existing 64*n+4 DWORD row pitch. One loaded
   payload/scale is reused for every matching local expert, with 64-bit offsets.
2. Existing external-ready waits for the complete source plan/quant, including
   the route-map writes. The ready kernel resets local per-M64 row counters.
   Each wave retires all fan-out stores before AcqRel agent RMWs. The last row
   acquires all prior writer releases and publishes the existing SYS A flag.
   Consumers, combine barrier and producer-to-consumer transition are retained.
3. Default off; restrict to external-ready, unpacked S1F, M64 flags and group1.
   First small changing-input test, then full EP4. Compare outputs, exact bytes,
   maps and completion counts. Native/HIP event timing only after correctness.
   Idle gate, one-second polling and boot/timeout guards remain unchanged.

Unique row-major follow-up, three prelaunch review passes:
1. Address/layout: one wave owns M16 unique source tokens; for each token,
   adjacent lanes load/store adjacent 16B K vectors. Global payload offset is
   destination_row * payload_bytes + vector * 16. Scale rows remain contiguous.
   Destination IDs are broadcast from that token's lane before fan-out. Tail
   tokens and dropped routes remain masked; no new source-token permutation.
2. Synchronization: both gathers complete before LDS reads. All M16 fan-out
   stores retire before the original per-route completion RMW/release chain.
   Counters still count rows, not vectors; the last writer publishes M64 ready.
3. Consumers: ExpertGrouping and fused GEMM1 both select a_preshuffled=0.
   The existing grouped wrapper propagates that layout to the quantized GEMM1
   output and GEMM2 input. Tests decode dispatched bytes as row-major. Run small
   and full changing-input checks, then matching baseline/hybrid HIP timings.

Row-major payload TDM scatter follow-up, three review passes:
1. The existing indexed-store API writes two groups of8 LDS rows per top-k
   route. Global row stride/payload width are2560B; LDS row width remains2832B
   for Flash. Columns beyond2560B are OOB stores, so scales/padding cannot reach
   payload output. Dropped/nonlocal routes use index=rows, the descriptor bound.
2. TDM gather finishes before scatter reads LDS. Payload TDM stores overlap
   generic scale stores, both only read the wave-owned LDS slice. tensor_wait(0)
   retires all payload stores before counter publication or LDS reuse; ordinary
   scale stores retain store wait and the existing AcqRel completion chain.
3. Producer storage, counter ownership, consumer layout and scheduler unchanged.
   Repeat small/full correctness before timings. No GPU jobs active at edit time;
   runner retains immediate idle check, three-review snapshot and timeout guard.

Persistent GEMM2 buffer scatter, three prelaunch review passes:
1. Direct compute-wave output: multiply routing weight in fp32 before bf16
   conversion, exactly as TDM scatter. Each WMMA lane owns its existing disjoint
   (M row, eight N values); one typed BufferCopy128b emits their four dwords.
   Per-peer SGPR descriptors use64-bit bases; per-peer offsets are bounded below
   4GiB and row/column tails are masked. No new LDS output staging is needed.
2. All task barriers and pipeline fences select dscnt plus TG signal/wait for
   this mode. Output uses VMEM buffer stores, not tensor stores, and the task
   emits no storecnt wait. A single final storecnt wait occurs after the outer
   persistent loop, before the stream-ordered combine kernel. Existing combine
   inter-rank barrier remains. ASM must verify no compiler-inserted store waits
   appear on the next-task backedge; source review alone is insufficient.
3. Opt-in AITER_MOE_EP_BUFFER_STORE, persistent256, bf16 combine only. Baseline
   keeps original gather S1F (unique/pack/source-shuffle off). Small and full
   checks compare with a push/TDM reference forced off at invocation, including
   six graph replays with changing routes/weights/input and empty experts. Keep
   all existing idle/timeout/boot checks; dump ASM separately from timing.

ASM review follow-up: small and full direct-buffer paths passed correctness.
The first ASM had exactly one storecnt wait at exit, but opaque CCO window
loads left resource descriptors in VGPRs: LLVM inserted descriptor waterfall
loops with wait_xcnt. Window base/stride are wave-uniform by the CCO contract;
readfirstlane of both32-bit address halves now establishes SGPR descriptors.
This only changes representation, not addresses or ownership. Re-review all
bounds and exit ordering unchanged; repeat small/full checks and inspect both
storecnt and xcnt on the actual loop backedge before timing.

SGPR-base ASM removed descriptor waterfalls, but every divergent peer branch
still emitted wait_xcnt before restoring EXEC. Replace the branch with buffer
OOB predication: non-target/invalid lanes use the descriptor's exact byte bound
as their offset. The16-byte access is dropped completely. Valid offsets, fp32
weighting, final store retirement and LDS barriers are unchanged. Three-pass
recheck: (1) bounds include all legal slots, (2) no mask restoration between
stores and next task, (3) re-run both correctness sizes and CFG/ISA audit.

Counter semantics correction: LLVM SIInsertWaitcnts.cpp associates X_CNT with
SMEM_GROUP/VMEM_GROUP and protects instruction source operands until address
translations finish under XNACK. STORE_CNT separately tracks VMEM_WRITE_ACCESS.
OOB predication reduces xcnts but remaining translation/source-register hazards
must be retained. The audit reports xcnt counts, and rejects storecnt in any CFG
cycle; it no longer incorrectly requires zero xcnt instructions.
Reference: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp

Coalesced compute-wave row writeback, three review passes:
1. Restore original weighted-bf16 LDS staging. Each wave reads only its own
   (wave_m M rows, wave_n N span);32 lanes each transfer four adjacent bf16
   values, giving256B contiguous writes for the Flash128-column wave tile.
   Row routes are broadcast from the preloaded compute-wave metadata. Scalar
   row descriptors name one peer/slot; masks cover M/N tails and invalid routes.
2. Each wave waits dscnt before reading its own C values. No TG rendezvous is
   required for wave-private C ownership. The final raw TG barrier retires all
   LDS reads before next-task TDM reuse. Buffer stores remain in flight; only
   the final loop-exit storecnt waits for remote completion.
3. This replaces the validated but slow fragmented register-store experiment.
   Retain P256, original S1F/preshuffle/CSV, and all idle/timeout guards. Re-run
   small/full correctness, inspect the _epbuffer_rows_ symbol and CFG, then time.

Final review after row-writeback validation:
1. Small257 and full16384 correctness passed six changing-input graph replays
   against push/TDM. M tails, hot/empty experts and disjoint ownership passed.
2. Full b4/cluster1x4/pw256 ISA has128 buffer stores, zero tensor stores and
   one storecnt in the noncyclic exit block. Raw barrier on the task backedge
   waits only LDS. Existing combine synchronization remains unchanged.
3. Coalesced-buffer GEMM2/E2E729.77/1685.21us loses to default578.35/1553.67us;
   keep opt-in disabled. Formatting/comments and unused-import cleanup leave
   the tested kernel AST identical. Commit lightweight metrics and ISA audit,
   not compilation caches. All successful runs retained their boot ID.

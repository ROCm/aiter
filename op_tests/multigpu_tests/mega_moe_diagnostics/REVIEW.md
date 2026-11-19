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

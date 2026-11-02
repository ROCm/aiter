# Resumed C4/C8 diagnosis

GPU work resumed after user said Proceed. Saved measurements remain separate from these diagnostics.

Initial matrix (C8,16requests, 1K/64 then1K/1K):
- Native control through opaque QKV op: both complete.
- Original fused graph: short test illegal memory access.
- Native norm/RoPE/cache tail: both complete this time (earlier runs had faults; do not assume conclusive fix).
- Fused eager: both complete; long test213.74seconds. Diagnostic only, not a performance comparison.

Ablations:
- Clone all persistent custom-op outputs: short completes, long illegal memory access. Not a sufficient fix.
- Retain fused Q/KV outputs, use native cache write: both complete. Native qk-norm computes redundant values only for isolation.
- Destination upper-bound guard + diagnostic records: fault persists; no recorded invalid-destination exception before fault. Bounds alone are insufficient. CacheRows changed from constexpr to runtime argument to avoid43layer-specific kernel compilations.
- Reuse normalized KV registers to write cache directly (instead of late TDM reload): short completes, long faults. Removing that TDM reload alone is insufficient.
- Added native cache-write position>=0 predicate: in progress. Native FlyDSL implementation also checks destination lower/upper bounds and position>=0. These missing predicates are real contract differences, but are not yet proven to explain the serving fault.

Generated LLVM/ASM checked: cache row offset is widened to64bits before multiply by1024bytes. Saved in asm-review/. No evidence yet of the old32-bit cache-address overflow.

New switches are diagnostic/opt-in: CLONE_OUTPUT, NATIVE_SCATTER, GUARD, DIRECT_KV, all prefixedATOM_DSV4_BATCH_QKV_. Current source now always bounds cache writes to CacheRows and skips negative positions; no correctness/performance claim until validated.

Further results (2026-10-21 resumed session):
- Position predicate passes short and 16-request long diagnostics, but fails larger C8 client after 25/64.
- Explicit preallocation before capture also fails C8 round1 (after49/64 reported); no valid second-run result.
- Isolated cache contract tests pass M1/2/4/8 and graph replay; a204.8GB cache test with100replays/case also passes.
- Audit found no active negative positions or invalid destinations before fault.
- ROCprof-instrumented 16-request diagnostics both complete without reproduction; trying to flush profiler output.
- Native-led shadow comparison in progress. Native outputs authoritative; candidate writes no cache. Metrics cover QA, QR, QB, Qnorm, KVnorm. Fixed diagnostic FP32 stats allocation and FP8 masked-load fill datatype; helper tested standalone before server restart.
- New native benchmark files resumed-native-c{2,4,8}-round2.json completed. C2 overlapped profiler activity and is not a clean final performance result.

Latest isolation:
- Native-led shadow also faults, with candidate Cache=None and native outputs authoritative. Saved pre-fault snapshot shadow-region-results.json has zero nonfinite values/health flags; QA/QR/QB maxdiff0; Qnorm maxdiff0.03125; KVnorm maxdiff0.0078125 across43layers. Snapshot precedes fault, so not proof all subsequent values are valid.
- Native-cache-scatter long stress faults; cache write is not the sole cause.
- Delayed first QB TDM load also faults; early QB TDM alone is not the cause.
- AMD_SERIALIZE_KERNEL=3 completes short+long16-request diagnostics. This changes scheduling and is not a performance result.
- Native-only opaque-op64-request stress and classic HIP graph scheduling64-request stress are in progress. Default early TDM/L2 prefetch retained.
- ROCprof trace flushed after terminating its compiler child; kernel trace is~1.2GB. No fault reproduced in that profiled run.

Progressive end-to-end isolation (latest):
- Native-only opaque wrapper: both64-request C8 clients PASS. Second TTFT295.603676ms, TPOT13.755246ms,total1126.821367tok/s.
- QA-only fused region: FAIL.
- Quant separated from projection+norm: FAIL.
- Quant / projection / reduction+norm separate launches: short+long16-request PASS (not64-request tested).
- Input quant only, native everything after it: short+long16-request PASS;64-request pair ongoing.
- Input quant + standalone original global_prefetch_b8 kernel, native everything after it: FAIL during short C8 diagnostic. This isolates the prefetch instruction sequence as a fault trigger without WMMA or cross-CTA barriers in the experimental region.
- Full fused QKV with only L2 prefetch disabled, early TDM retained: BOTH64-request clients PASS. Second TTFT299.158299ms,TPOT12.987541ms,total1186.546444tok/s. GSM32 check ongoing.
- Atomic polling and classic graph scheduling both FAIL larger stress.
- New descriptor-based TDM prefetch candidate (AITER moe_op_gemm_a4w4.py pattern) is in progress. Do not claim instruction/hardware root cause yet: we have identified a triggering sequence, not explained its failure.

Source switches introduced for diagnosis remain opt-in. Current source default still original global prefetch; do NOT deploy until the replacement is validated.

User clarified non-speculative prefetch must be masked like any potentially out-of-bounds load.
- Checked all43 checkpoint QB tensors:FP8[32768,1024]=33554432bytes. Saved qb-weight-shapes.json.
- Added explicit per-lane EXEC masking for raw global_prefetch_b8, with runtime allocation length. A standalone grid spanning8MiB against a1KiB tensor passes. ASM saved in asm-review/masked-prefetch.amdgcn.
- Added full-matrix bounded TDM descriptor variant, non-speculative; compiler emits global_prefetch_b8 with scalar base/vector offsets and bounds handling.
- Both variants currently in64-request two-client C8 tests. Do not yet claim root cause or fix.
- Original instruction sequence's intended offset formula is0..33554176 inclusive (256byte spacing), inside32MiB; lack of explicit mask is fixed in new variants, but why original serving run faults is still under investigation.
- No-prefetch GSM8K32=30/32 versus latest native31/32; limited check, not numerical parity proof.

Masked-prefetch validation completed:
- Explicitly masked inline global prefetch: both64-request C8 clients PASS. Second TTFT292.386149ms, TPOT12.862003ms,total1218.449509tok/s.
- Bounded non-speculative TDM prefetch: both64-request C8 clients PASS. Second TTFT295.437046ms, TPOT12.864193ms,total1215.605177tok/s.
- Input-only control also passes both64-request C8 clients.
- Default batched QA/QKV PREFETCH_STYLE is now2 (explicit per-lane mask). Style1 bounded TDM remains available; style0 unmasked is diagnostic-only and has reproduced serving faults.
- Early TDM and L2 prefetch both retained. Current scope QKV-only hybrid; not a C2/C4/C8 full-layer monokernel. C1 full-layer source has not been changed by this fix and its other prefetch sites still need audit.
- Precise underlying cause of the old unmasked sequence is NOT established. Intended offsets were inside logical allocation; masking changes generated execution/address handling. Do not label it a hardware/driver bug or assert a specific OOB address without evidence.
- Second-client records saved in prefetch-second-run-results.json. No image/branch push performed this resumed session; prior image/archive do not contain these changes.

Final checks for masked-prefetch checkpoint:
- Default masked kernel passes isolated M1/2/4/8: QA/QR/QB bitexact vs fixture native outputs, Q/KV relative RMS0, cache bounds/negative position/inactive entries pass,health0.
- Both masked and bounded-TDM serving candidates GSM8K32=30/32, same as no-prefetch candidate, versus latest native31/32. Numerical parity is not yet established.
- No C2/C4 performance reruns for the newly masked candidate yet.

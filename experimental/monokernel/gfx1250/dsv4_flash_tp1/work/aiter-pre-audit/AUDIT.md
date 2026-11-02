# Full-layer monokernel audit — 2026-10-21

Scope: current integrated-c8 Flash full layer, default optimized flags, C8/1k-input/1k-output representative shapes. C1/C2/C4 exceptions noted. This is a source-level dependency audit, not a measured attribution of barrier or memory latency. Imported AITER operator internals were read; table records missing overlap across region boundaries, not claims that their inner GEMM loops lack pipelining. No GPU work changed other than attention mHC pre.

## Adopted mHC pre

New mhc_pre_aiter.py specializes AITER's gfx1250 post/pre producer design for pre-only work: 64 producers, each four residual streams ×64 channels, all rows in a16-row WMMA tile; packed BF16 hi/lo weights; FP32 projection and sum-square partials; LDS reduction of head partials. No previous-post calculation or residual rewrite. One AITER reduce/apply CTA per token with a separate Sinkhorn worker warp. Producers synchronize on a dedicated64 counter; all256 synchronize only at the region exit, before downstream consumers. The existing MHC scratch and counter are reused sequentially, with the region-exit barrier separating lifetimes. Partial buffers are [64,M,32] and [64,M]; coefficient outputs [M,4] and [M,4,4]. Rows beyond M are descriptor-zero-filled; stores masked. Optional FP32 input statistics remain supported.

Runtime selects with ATOM_DSV4_AITER_PRE=1 (default); =0 restores previous pre. Full-kernel API OPT_PRE defaults false unless runtime/harness explicitly passes packed PRE_W and existing MHC scratch/layouts. Decode-only dispatch is unchanged. No new serving accuracy or E2E throughput claim is made from region tests.

## Barriers where fewer CTAs do the preceding work

All counts are participating CTAs, not warps. A producer count less than256 does NOT prove that only those producers need to synchronize. A producer-only readiness semaphore can let all consumers wait without all consumers atomically arriving; simply changing sync(...,256) to a smaller number while256 CTAs call it is incorrect. Each independently advancing subset needs its own counter/epoch and a buffer-lifetime proof.

| Boundary in execution order | Producing CTAs at C8 | Current synchronization | Finding / required change |
|---|---:|---|---|
| Residual unshuffle → attention mHC pre |256 (64C below C4)|All256|Not oversized at C8; at C1/C2 only64/128 produce. Pre producer can eventually read native shuffled residual directly and remove materialization.|
| Old attention-pre projection → finishing |256|All256|Replaced by64-producer dedicated barrier with new design.|
| New attention-pre projection → finishing |64 →8 finishers|Dedicated64|Already scoped.|
| Attention-pre finishing → compressor/QKV |8 new;16 old|All256|Small producer set, broad consumers. Candidate for8-arrival publish/readiness, not8-CTA-only execution.|
| HCA compressor projection → historical pooling |64|All256|Oversized rendezvous; pooling uses up to32 CTAs. Previous subgroup attempt failed in integration; do not remove without revalidation.|
| HCA historical pooling → final compressor |Up to32, only boundary tokens active|All256|Only up to8 final compressors consume partials. Dedicated pool readiness candidate.|
| Compressor/indexer/cache/CSA index publication → QKV |HCA up to8; CSA up to32 distinct CTAs|All256|QKV math itself is independent of these publications. Readiness is needed before attention/cache consumers, not necessarily before QKV. Historical-read/state-write ordering must remain intact.|
| RoPE table staging → QKV |8|All256|Replicate tiny coefficient loads in actual consumers, or stage earlier and publish per row.|
| QA/KV split projection → RMS finishing (B2) |192|All256|16 finishers consume. Dedicated192 producer readiness possible; QB prefetch currently runs before this wait.|
| QA/KV finishing → QB (B3) |16 total; only8 QA finishers feed QB|All256|QB needs QA normalization/quantization only. Separate QA readiness from KV readiness, which attention needs later.|
| HCA attention partials → split reduction |160 for136/144 KV entries|All256|96 inactive slots still arrive. Ragged lengths make active count dynamic; use per-row/group epochs or an active-count protocol. SWA/CSA at these C8 shapes use all256.|
| Attention output projection final reduction → mHC post/pre |256 (min(256,32C) at smaller C)|All256|Not oversized at C8; fewer producers at C1/C2/C4.|
| Attention post/FFN-pre projection → finishing |64|Dedicated64|Already scoped.|
| FFN-pre finishing → quant/router |8|All256|Broad consumers; producer-only readiness possible.|
| Router and two input quantizations → routing selection |Router16; quantization covers all256|All256|Total producer count is256, but top-k needs only router logits. Router readiness can decouple selection from quantization.|
| Routing selection → histogram/map construction |8|All256|Only8 metadata CTAs consume selected rows. Dedicated8 barrier candidate.|
| Shared gate/up plus routing metadata → activation |64 GEMM producers;8 metadata producers within that set|All256|Shared activation uses128 CTAs; MoE also needs routing metadata later. Separate readiness streams possible.|
| Shared activation/quant → shared down + MoE |128|All256|Shared down has128 CTAs. Routed MoE input is already quantized and could start independently; do not block it on shared activation.|
| Routed MoE gate/up → down |min(256,16×active_experts)|All256|May be full at C8, underfilled for concentrated routing/C1. Dynamic expert scheduling requires per-expert/phase readiness, not a fixed smaller count.|
| Routed MoE down → routed output reduction |min(256,16×active_experts)|All256|Same dynamic-producer issue; output reduction has min(256,32C) CTAs.|
| Shared+routed output → add → final post → reshuffle |256 at C8;32C at smaller C|All256 after each stage|Not oversized at C8. At smaller C, narrower readiness or same-owner chaining can remove arrivals.|

Other audited all256 boundaries: QB projection→Q normalization, Q normalization→attention, attention reduction→output projection, inverse-RoPE/quant→OA, OA→reduce/quant, reduce/quant→OB, OB→final reduction. At C8 their producers span all256; these are not fewer-producer offenders, although dependency-local alternatives may still help.

## Missing early loading / prefetch opportunities

“Missing” means no explicit load/prefetch scheduled in the preceding region for the listed data. Warm L2 may hide some cost. TDM in a GEMM prologue or double-buffering within it does not constitute overlap across the previous-region boundary. Prefetch only valid mapped ranges; dynamic indices must be validated. Adding every possible prefetch indiscriminately could evict useful data or increase LDS/register residency.

| Next region / data | Current first access | Earlier overlap window | Constraints / status |
|---|---|---|---|
| Attention-pre packed projection weights |Inside new pre producer, after unshuffle barrier|During unshuffle|Missing cross-boundary overlap. New producer issues weight TDM before residual TDM, but both are after unshuffle.|
| Attention-pre scale/base/norm weights |Inside reducer after64-producer wait|During producer WMMA / before wait|Missing. Old custom pre had early residual/norm-weight TDM; adopting AITER does not preserve that schedule. Residual is already available for early loading too.|
| Compressor main and CSA indexer projection weights |Inside GEMM after mHC-pre exit barrier|During mHC-pre finishing|Missing for both MW/IW.|
| Compressor APE/norm/RoPE coefficients |APE after historical pooling; norm after pooled output; RoPE near epilogue|While projection or historical pooling runs|Missing, including internal late row() loads. Position-dependent addresses known earlier; mask inactive/padded tokens.|
| Main Q/KV RoPE tables |8 CTAs stage them only after compressor completion|During mHC or compressor work|Missing. Position-dependent but independent of activations; could replicate consumer loads and remove staging barrier.|
| QA/KV projection weight scales |Loaded during projection loop|During mHC/weight prefetch|Missing early scales. Weight values already receive L2 prefetch during mHC, but first LDS weight tiles still start only at QKV entry.|
| QA/KV RMSNorm weights |Inside finish(), after B2|While split-K projection runs / before B2|Missing.|
| QB weight scales |Loaded in QB WMMA loop after B3|During QA/KV projection or RMS finishing|Missing for scales. QB values already have early TDM for first64-row half and L2 prefetch before B2.|
| QB second64-row weight half |TDM starts only after first-half result is stored|While first-half WMMA runs|Partial overlap gap: L2 prefetch covers it, but no overlapping LDS fill; current buffer is reused. Needs ping-pong LDS/residency check.|
| Attention sink coefficients |Inside attention/reduction functions after QKV/partial barriers|During QKV or attention partial work|Missing cross-boundary preload; tiny data, benefit unmeasured.|
| Attention historical paged KV/indices |First explicit prefetch starts in attention after QKV boundary|During QB projection/norm|Missing boundary overlap for historical data. Dynamic indices/lengths known earlier; exclude unready current/compressed rows and validate slots.|
| OA weights and scales |split_project prologue after inverse-RoPE/quant barrier|During attention reduction or inverse-RoPE/quant|Missing. Inner GEMM double buffering already exists.|
| OB weights and scales |split_project prologue after OA reduce/quant barrier|During OA GEMM or its reduction|Missing.|
| Next FFN mHC-pre packed weights |post_pre producer prologue after output projection barrier|During OB GEMM/reduction|Missing. Dynamic post/comb coefficients are already available from attention pre, so those can also be loaded early.|
| FFN mHC-pre scale/base/norm weights |Reducer after dedicated64 barrier|During post/pre producer|Missing. Native reducer overlaps some of its own TDM and arithmetic, but starts after producer completion.|
| Router weights |router GEMM prologue after FFN-pre exit and quant work|During FFN-pre finishing / input quantization|Missing.|
| Routing correction bias |Inside top-k selection after FFN initial barrier|During router GEMM|Missing when HAS_BIAS.|
| Shared gate/up weights and scales |Inside shared GEMM after routing selection barrier|During router/top-k work or FFN-pre|Missing. Shared weights do not depend on routing.|
| Shared down weights and scales |Inside shared GEMM after activation barrier|During gate/up GEMM or activation|Missing.|
| Routed expert gate/up weights/scales |Inside MoE stage after shared down work in the same CTA|After expert map publication, while shared branch executes|Missing once expert IDs are known. Cannot safely select expert addresses before routing.|
| Routed expert down weights/scales |Inside down stage after gate/up phase barrier|During routed gate/up / before phase wait|Missing; expert IDs are already known. Internal stage pipelines exist but do not preload across this phase boundary.|
| Final mHC-post mixing coefficients |Inside post_body after add barrier|During MoE/output add|Missing explicit early loads of already-produced coefficients; they are dynamic but available long before this consumer.|

Already present and not listed as wholly missing: QA weight L2 prefetch before mHC; QB first-half early TDM before QA; whole-QB bounded L2 prefetch before B2; pipelined OA/OB and routed-MoE inner loops; paged attention slot/TDM pipelining. These do not cover the separate scales/second-half/boundary gaps listed above.

## Sequencing recommendation

First test router/shared-branch decoupling and early OA/OB/shared weights, then replace the8-producer RoPE staging barrier with duplicated loads. Next separate QA readiness from KV readiness and replace routing's8-producer/all256 barrier. HCA producer/pool counters need an isolated protocol test because a previous subset implementation failed after integration. Do not change all counters in one patch. Measure isolated and full-layer regressions; validate each before serving rollout.

## Synchronization/preload interaction

The current sync helper executes s_wait_loadcnt0 and s_wait_storecnt0 before its CTA barrier and atomic arrival. Hoisting ordinary vector loads just above sync may therefore make publication wait for those loads. For tiled static data, use dedicated LDS TDM staging and an explicit wait at consumption; verify that the arrival helper does not inadvertently drain or alias that staging. Across a phase change, do not overwrite a producer's shared-memory workspace with consumer prefetches. Source-order prefetch is not proof of hardware overlap; inspect assembly and benchmark each change.

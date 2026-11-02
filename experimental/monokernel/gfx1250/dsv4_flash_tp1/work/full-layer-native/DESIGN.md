# Full uncompressed Flash layer, C1

Extend the existing full-attention prototype with FFN pre/RMS, both input quantizations, router projection, C1 stock top-6 routing, shared expert, routed FP4 experts, combine and FFN post. One 256-CTA resident launch. Reuse WMMA/TDM implementations and keep attention L2 prefetch. No extra device-side launches.

Stage plan: existing attention (all 256 CTAs with subgroup work); attention post (32 CTAs); FFN mHC projection (256) and norm/mix (2); shared quant (32) and MX quant (128); router projection (16); top-k/metadata (1), shared gate/up (64); activation+quant (16); shared down (128); expert gate/up (192 tasks), down (192 tasks), combine (32); add (32); post (32).

Cross-stage payloads are global scratch in coherent L2. Existing device-scope store drain, atomic arrival and load-poll barriers order all consumers; prototype MoE has two separately counted barriers. Buffers have fixed addresses and counters advance across graph replay. No concurrent launches may reuse this state. Health flags must stay zero. Compile and inspect VGPR/LDS/scratch before treating the full grid as resident. Target no spills, at most one CTA per CU.

Important routing correction: AITER_TRITON_USE_HERD=1 does not activate HERD at C1; default HERD minimum M is 16. C1 uses stock top-6. Preserve native BF16 score rounding, tie preference and expert-sorted output. No hash-table substitute.

Known inherited limitation: full-attention currently differs numerically from ATOM. Validate full outputs and intermediates, do not label timing-only success as accuracy acceptance. Initial full-layer scope is the uncompressed sliding-window layer, not compressed attention layers. Serving integration must be opt-in, narrowly guarded and preserve HCState/cache behavior and native fallback.

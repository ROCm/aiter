# C16 experimental full-layer monokernel

Fixed 256 resident CTAs; 16 live WMMA rows for projections and shared MoE. C8 uses the same arithmetic tile with eight masked rows. The persistent launch is not expanded because synchronization requires all CTAs resident.

- Native shuffled residual enters 64-CTA AITER-style mHC projection; 16 row reducers, producer subgroup synchronization.
- QKVA: 192 producers (128 QA, 64 KV), split K16; producer-local quantization. QA/KV row reductions; QB 256 CTAs. Bounded L2 weight prefetch and early TDM retained.
- Attention: four KV splits, four head groups, 16 rows =256 tasks over256 CTAs, selected after an A/B against eight splits. C8 retains eight splits. Split reduction has1024 tasks over256 CTAs.
- OA/OB retain early TDM first-weight tiles and split K4/K8.
- Attention post/FFN pre use64 producer CTAs and16 reducers with AITER warp specialization.
- Shared-expert quantization scales use stride16 at C16 (stride8 at C8); scale buffers resized together.
- Routed MoE uses existing AITER/HERD-derived FP4 stage with up to96 token-expert routes; tasks loop over resident CTAs.
- Residual sum and final post now iterate512 tasks rather than dropping rows8–15. Final post writes native shuffled output directly.
- Use TILE_MASK192, OVERLAP2, EDGE_MASK0. Unvalidated fine-grained tile mailboxes remain disabled; their C8 offsets overlap at C16 and are guarded explicitly.

Only microbenchmarks are requested. Serving dispatch and the working C8 installation are unchanged.

C16 follows native HERD routing (top7, batch popularity, drop least-popular then lowest-score then lowest-index, renormalize kept6). A dedicated tail in Chosen holds temporary drop decisions; it never aliases the gather map. Coherent fresh reads are used after the kept-set publication.

```text
C16, one resident grid: 256 CTAs (projection tile M=16)

native shuffled residual
  | 64 mHC projection producers -> 64-producer barrier -> 16 row reducers
  | [layer boundary publication]
  +-- CSA: 128 projection CTAs + 32 index projection CTAs
  |        16 main + 16 index compressors; 64 identity-pack tasks
  +-- HCA: 64 projection CTAs -> 64 pooling tasks -> 16 merge/compress CTAs
  |
  | QKVA: 192 CTAs = 128 QA +64 KV; split-K16, local quant
  | -> producer completion -> 16 QA +16 KV norm/reduce CTAs
  | -> QB:256 CTAs; early TDM weights + bounded L2 prefetch
  | -> Q norm/RoPE:1024 row/head tasks over256 CTAs
  |
  | attention:4 splits x4 head groups x16 rows =256 tasks/CTAs
  | -> completion ->1024 head-reduction tasks over256 CTAs
  |
  | inverse RoPE/quant -> OA:256 CTAs, split-K4
  | -> reduce/quant -> OB:256 CTAs, split-K8 -> reduce
  | early TDM first weight tiles overlap input work
  |
  | attention post + FFN pre:64 producers -> subgroup barrier ->16 reducers
  |
  | shared/routed input quant + router GEMM
  | -> HERD:16 row CTAs; top7 publication, then TWO16-CTA barriers
  |    temporary drop decisions use dedicated scratch
  | shared expert:64 up CTAs ->256 activation/quant tasks ->128 down CTAs
  | routed MoE:active-expert tiles over256 CTAs; stage1 -> stage2 ->6-route sum
  |
  | residual sum + final post:512 tasks over256 CTAs
  | final post directly stores native shuffled output
  | KV cache persistence:16 CTAs (attention already consumed current KV directly)
```

The other existing layer-wide completion barriers remain256 participants in this selected recipe. Experimental per-edge/tile handshakes are not enabled. HERD's extra internal barriers have only16 participants and their own reset counter; other CTAs may proceed with independent shared-expert work. Full-layer correctness and isolated comparisons include this final version.

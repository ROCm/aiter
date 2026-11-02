# Execution flow: recommended two-launch C1 Q/KV preparation

Input: one BF16 activation row4096. Q-A maps4096->1024. KV maps4096->512. QB maps1024->32768 (64heads x512). This is preparation for attention, not the attention operation.

| Step | Who executes it | Work and handoff |
|---|---|---|
| Input quantization | All192 first-launch CTAs | Each CTA quantizes its K256 slice as two128-element blocks, in registers. Shared input blocks are recomputed across12output tiles. No initial global FP8 buffer/launch. |
| QA projection |128CTAs =8Ntiles x16Ksplits | Each computes128output partials over K256 using BF16 WMMA/FP32 accumulation. |
| KV projection |64CTAs =4Ntiles x16Ksplits | Same128output/K256 decomposition. Q/KV branches are independent. |
| Split-K reduction | All192producer CTAs | Every producer FP32-atomic-adds its128partials to shared Acc. Each final channel receives16adds. No dedicated reduction kernel/CTA. |
| Completion election | All128Q or64KV producers | Drain atomic writes, CTA barrier, increment separate padded group counter. Tickets127/63 identify final Q/KV finisher. Identity is dynamic. |
| QA weighted RMSNorm | The last Q-A CTA (one of128) | Read complete1024FP32 sums; round BF16; compute FP32 sumsquares/mean/rsqrt(eps1e-6); multiply gamma. |
| Latent Q quantization | The same last Q-A CTA | Quantize normalized1024 values in8groups of128, publish1024FP8 values +8E8M0 scales. Also stores FP32 norm diagnostic output. |
| KV weighted RMSNorm + RoPE | The last KV CTA (one of64) | Read complete512sums; BF16round; weighted RMSNorm; rotate last64channels in interleaved pairs; write512BF16. |
| QA/KV state reset | Respective finishing CTAs | Zero their accumulator portions and completion counters. Included in timing. |
| Launch boundary | GPU stream ordering | Qlatent and KV output published before next launch. No polling/global grid barrier in recommended path. |
| QB projection |256CTAs =64heads x4Ntiles | Updated N128tile, full K1024, no split-K. Eight K128blocks processed with four-way unrolling. WMMA result rounds BF16; stored in FP32 scratch. Previous version used512N64CTAs. |
| Head completion election |4producer CTAs/head | Drain stores, CTA barrier, increment head counter. Last ticket3 wins. Previous version used8CTAs/head, ticket7. |
| Q head RMSNorm + RoPE | One winning CTA/head,64winners total | Read full512completed channels, unweighted RMSNorm, rotate last64channels, write BF16[512]. |
| QB state reset |64head finishers | Reset each head counter. Scratch overwritten by next invocation. |

```text
BF16 input[4096]
        |
        +-------------------------+-------------------------+
        |                         |                         |
        |  LAUNCH 1:192CTAs, each4waves                      |
        |                         |                         |
        v                         v                         |
 Q-A:128CTAs                KV:64CTAs                       |
 8Ntiles x16Ksplits          4Ntiles x16Ksplits              |
 local input quant          local input quant              |
 WMMA partials              WMMA partials                  |
        |                         |                         |
 16atomic adds/channel      16atomic adds/channel           |
        |                         |                         |
 AccQ[1024]                 AccKV[512]                      |
        |                         |                         |
 last Q ticket127           last KV ticket63               |
 ONE Q producer CTA         ONE KV producer CTA            |
        |                         |                         |
 BF16round                  BF16round                      |
 weighted RMSNorm           weighted RMSNorm               |
 same CTA:FP8quant          same CTA:RoPE                   |
        |                         |                         |
 Qlatent[1024]+8scales       KVout[512] BF16                 |
        |                         |                         |
        +--- launch/stream boundary -----------------------+
        |
        v
 LAUNCH 2:256CTAs, each4waves
 QB WMMA:64heads x4tiles/head,128columns/tile
 Full K1024 perCTA; NO split-K; K-loop unroll4
        |
 BF16round -> FP32 scratch Q[64,512]
        |
 4completion tickets/head
        |
 Last producer CTA perhead (64winners)
        |
 Read512values -> RMSNorm -> RoPE -> BF16 Qout[64,512]
```

The last QA/KV CTA does not serially sum16scratch vectors: atomics have already performed split-K reduction. Do not square each Kpartial to form RMS statistics; squared final sums contain cross terms absent from sums of squared partials.

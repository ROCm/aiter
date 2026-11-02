# mHC through attention reduction, Flash C1

Scope: layer-0 checkpoint weights; four residual streams (4x4096), controlled
BF16 KV cache, 64 heads x512, split-KV 4. One resident 256-CTA grid for the fused
prefix. This is an isolated experiment, not ATOM integration or serving metrics.

The mHC projection uses the prior all-WMMA algorithm: 24x16384 FP32 weights
split into BF16 high/low components, BF16 WMMA, 128 K partitions. TDM now loads
both its input and weight tiles. All 256 CTAs produce mHC partials; 24 CTAs
reduce them and apply input RMS/learned scale/bias. CTA0 does Sinkhorn and
pre/post coefficient transforms. Sixteen CTAs mix four residual streams;
CTA0 applies weighted input RMSNorm. These stage boundaries use the existing
bounded device-scope grid barrier. Normalized input is BF16, as before.

The current QKV+attention implementation follows: input quant inside QA/KV
WMMA producers, fixed-order split-K reduction and RMSNorm, latent quant or KV
RoPE; QB WMMA with L2 prefetch and double-buffered TDM; per-head RMSNorm/RoPE;
Q/KV publication barrier; direct new-KV consumption and concurrent single-writer
cache update; split-KV attention WMMA and last-arrival reduction. No cache-update
barrier. The attention reduction is part of the same launch.

Matched unfused control: same arithmetic, shapes, tiling, and weight format,
separated at the operation boundaries (17 launches). It uses TDM and WMMA, not
the historical vector GEMVs. QB and attention retain L2 prefetch in both arms.
The control is not native AITER/ATOM's complete launch sequence.

- mHC: projection; partial reduction + input RMS + scale/bias; Sinkhorn;
  residual stream mix; weighted input RMSNorm (5 launches).
- QKV: input quant; QA/KV projection; split-K reduce; QA/KV RMSNorm; QA quant;
  QB projection; Q RMSNorm; Q RoPE; KV RoPE (9 launches).
- Attention: cache store; split attention partials; split reduction (3 launches).

An additional three-region arm provides the closer launch-fusion comparison:
one mHC kernel, one current QKV kernel, one current attention kernel.
All arms preserve mHC coefficients for the later residual stage, even though
that stage is outside this prefix. Kernel completion publishes cache updates.
No overlapping calls may share workspaces or counters. Requires an otherwise
idle gfx1250 with 256 CUs available for the grid barriers.

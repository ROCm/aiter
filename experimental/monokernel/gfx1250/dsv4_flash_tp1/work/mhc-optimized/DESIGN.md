# Accepted mHC dataflow

Inputs: R BF16[4,4096], HAF FP32[24,16384], HAS FP32[3], HAB FP32[24],
AN BF16[4096]. Outputs: Mix FP32[24], Raw BF16[4096], X BF16[4096].
Original FP32 projection partial workspace Temp[128,24] retained.

```
                         256 resident CTAs
         +------------------------------------------------+
         | Optional early QA/KV L2 + TDM (CTAs 0..191)    |  opt_both
         | Early full R + AN TDM (CTAs 0 and 1 only)      |
         | mHC projection: 2 N tiles x 128 K partitions  |
         | WMMA, original FP32-weight hi/lo decomposition |
         +-----------------------+------------------------+
                                 |
                  one device-scope grid barrier
                                 |
              +------------------+------------------+
              | CTA0                                | CTA1
              | First4 logit reduction + exact RMS   | Logit reductions + exact RMS
              | Sigmoid pre coefficients            | Pre/post coefficients
              | Four-stream mix -> BF16 Raw         | 4x4 Sinkhorn
              | Exact BF16 Raw RMSNorm -> BF16 X     | Persist all24 Mix values
              +------------------+------------------+
                                 |
       Standalone: kernel completion publishes all outputs.
       Prefix: existing mHC-to-QA grid barrier publishes X/Mix.
                                 |
              Current QA/KV -> QB -> attention -> reduce
```

CTAs2..255 have no tail work. Duplicate first-four-logit/RMS computation in two
CTAs avoids a cross-CTA scalar/coefficients handoff. Only CTA0 writes Raw/X;
only CTA1 writes Mix. Projection partial ownership is unchanged and disjoint.

The grid barrier drains ordinary loads/stores, executes a CTA barrier, atomically
increments the shared epoch counter, and polls with pure device-scope loads.
No polling atomic-RMW is required. Bounded timeout sets health; nonzero health
invalidates output. GPU availability must permit all256 CTAs to be resident.
No overlapping calls may share workspaces/counters.

RMS arithmetic: four independent 1024-element sums per residual stream, with
exactly the original per-chunk tree; sixteen scalar sums are then added in the
original sequence. Reshape supplies the chunks without cross-lane gathers.
The naive four-by4096 reduction was faster but changed rounding and failed the
full-prefix gate after quantization; it is not the selected implementation.

The persistent Mix coefficients have tiny reciprocal/reduction FP32 differences
(up to1.47e-7 relative RMS), while tested Raw/X/Q/KV/attention outputs are bitwise
identical to the original control. Later residual/MoE consumers of Mix are beyond
this prefix's validation scope.

Selected options are explicit, based on tested endpoints: T128 `opt_both` and
T1024 `opt`. All keep mHC early residual/norm TDM and existing QB/cache prefetch.
Additional early QA staging stores two128x128 FP8 tiles per producer in LDS;
QA reuses them, without issuing another tile fetch. This increased early live
state did not increase the final compiled LDS allocation or introduce spills.

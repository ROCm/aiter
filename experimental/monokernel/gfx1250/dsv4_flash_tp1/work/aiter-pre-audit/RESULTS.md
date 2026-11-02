# AITER-style attention mHC pre results

Checkpoint-derived layer0 mHC weights and embedding-based residual fixtures. C1/2/4/8 region microbenchmarks; reported timing is second graph benchmark round. Both old/new wrappers include their exit barrier. These are not serving client measurements.

|C|Previous pre us|AITER-style pre us|Reduction|
|---|---:|---:|---:|
|1|7.274|5.972|17.9%|
|2|7.038|5.873|16.6%|
|4|8.621|5.927|31.3%|
|8|8.522|6.765|20.6%|

C8 full SWA layer synthetic fixture, graph replay, second round: previous255.201us, new243.795us (4.5% reduction). This is a full-layer microbenchmark, not E2E throughput.

Numerics: checkpoint-derived normalized outputs exact versus native for C1/C2; C4/C8 maximum absolute difference0.000244140625, relative RMS6.37e-5/4.50e-5. Post/comb maximum absolute differences <=1.79e-7. C8 full-layer old/new final relative RMS: SWA3.17e-5, CSA6.73e-5, HCA3.41e-5. Q was identical in these three synthetic full-layer comparisons. No claim of bitwise equivalence or full-model accuracy qualification.

All tested full variants: zero spills,157156B LDS; VGPRs reduced from434/436/432 to424/426/424 for SWA/CSA/HCA. Region LDS44004B,96–98VGPR. Barrier health clean, including repeated SWA graph replay.

Implementation: integrated-c8/mhc_pre_aiter.py; runtime default ATOM_DSV4_AITER_PRE=1; set0 to A/B previous pre. No modifications to prefill or decode eligibility. No additional audit optimizations applied. See AUDIT.md for all barrier and preloading findings.

No new E2E serving or GSM8K run in this task. Base environment/model remain external to the source bundle.

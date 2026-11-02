# mHC early TDM experiment, GPU3

This is an experimental branch, not the accepted full-prefix implementation.

Uses the split pre/mix/norm and post/Sinkhorn tail from `mhc-opt-tail/optimized_rcp.py`. Two CTAs duplicate residual RMS and first-four logit reductions so normalized input can be prepared independently from Sinkhorn. Residual 4x4096 BF16 and norm4096 BF16 are loaded using TDM before mHC projection. One projection-publication grid barrier replaces four old mHC internal barriers. Projection remains WMMA/TDM. Register reciprocal Sinkhorn preserves coefficients to standalone tolerance.

Second-round matched 128-replay graph event results on GPU3: unfused12.888us; old fused14.768us; new direct-load tail6.124us; earlyTDM before projection5.996us; TDM after projection before barrier6.078us. Thus earlyTDM was a modest gain beyond the major synchronization reduction. EarlyTDM has245VGPR,56496LDS,zero spills.

Standalone64input checks and10240 changing-input graph invocations with poisoned outputs passed0.001relativeRMS, maximum stress error5.172e-8. HOWEVER the parent full-prefix checks found a quantization boundary amplification caused by changing residual RMS summation order (four4096 reductions rather than sixteen ordered1024 reductions). Parent is fixing exact ordering in the integration copy. Do not present this branch as numerically accepted for full prefix until the parent's fix and full validation are incorporated.

Source `candidate.py`; API `candidate_body` same positional arguments as original `mhc_body`, with EARLY=True. EARLY=2 issues TDM after projection, before barrier. EARLY=False direct residual loads for controlled ablation. `candidate_initial.py` is preserved before timing-placement extension. Candidate benchmark and stress scripts self-contained with inherited fixture/dependencies. Older copied REPORT/STATUS files describe the baseline, not this experiment; use this report and candidate-* outputs.

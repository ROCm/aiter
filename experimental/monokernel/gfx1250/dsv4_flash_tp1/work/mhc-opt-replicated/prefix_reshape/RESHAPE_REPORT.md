# Exact-order RMS optimization

Accepted candidate source is `tdm_tail.py`, copied from the exact-order integration and changed only to replace sixteen 4096-to1024 gathers with reshape reductions. Each residual stream is reshaped to [4,1024] with all128 threads along the1024 dimension. Four chunk sums are computed in parallel, then each scalar is extracted and accumulated in the original sixteen-chunk sequential FP32 order. This preserves the old numerical boundary without expensive gathers.

GPU3 second-round matched graph timing: unfused12.953023us, original fused14.905953us, optimized6.219641us. Optimized267VGPR,56496LDS,zero spills. WMMA projection, early residual+norm TDM, split independent pre/Sinkhorn branches, reciprocal Sinkhorn preserved.

Full-prefix validation521cases (9 checkpoint embeddings +512 random tensors, seeds81100..81611, varying magnitude) passed. Raw, normalized X, normalized QA/KV Y, QB QP, finalQ, finalKV, attention output are bitwise identical to original prefix. Mix coefficient maximum relativeRMS1.09302583e-7. Final group timings, generatedISA, all validation rows, maxima and scripts saved here. Independent changing-input graph stress delegated to the tail agent onGPU1; final parent integration owns those results.

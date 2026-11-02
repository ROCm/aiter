# Numerical investigation, 2026-10-21

All error values below are relative RMS against the native output. The shadow
validator returns the native result to the next layer, so its inputs do not
accumulate candidate error. It is a localization test, not an end-to-end quality
qualification.

## Proven differences and fixes

1. QA/QB/OA/OB: replacing BF16-upcast FP8 dot operands with gfx1250 scaled-FP8
   WMMA reduced query error substantially. In 172 real-layer observations,
   same-input FFN (including routed and shared experts) was exactly native.
2. Attention: native AITER scales Q in FP32 and rounds Q to BF16 before the
   QK WMMA. The prior adaptation instead scaled the FP32 dot result. Native
   uses a single BF16 probability operand; the prior adaptation added a second
   residual operand. Native C1 uses BK16 and one tile per active KV split;
   the prototype used BK32 and four splits.
3. `attention_matched.py` / `aiter_matched.py` restore those boundaries. With
   64 splits and BK16, all 14 isolated lengths (1 through 640) match actual
   ATOM output exactly. In 172 layerwise real-model observations, attention
   matched exactly in 161 cases and output projection in 165. Worst full-layer
   relative RMS dropped to 0.0177903, but free-running generation still degraded.
4. AITER's fused mHC post/pre computes its residual RMS statistic from the FP32
   post-mix values, before storing the BF16 residual. `mhc_precise.py`,
   `post_precise.py`, and `model_layer_precise.py` preserve that value in
   64-KiB per-layer FP32 buffers. This variant must be checked with confirmed
   decode engagement; earlier shape-gated controls did not engage it.

## Serving controls

Native eager and graph serving produce the same five deterministic smoke
completions. Candidate eager and graph both showed degradation. A Python shape
check around the layer hook did not reliably engage decode in the compiled
model. The new `state_op` carries all four HCState tensor fields across its
opaque boundary: prefill uses native pending post/pre state; decode uses a
full fused layer and identity pending state. A native-only `state_op` control
isolates adapter behavior from fused arithmetic.

The same cache-directory key appeared in multiple dispatch experiments, but
those directories were empty when inspected. This is NOT proof of stale-cache
reuse and must not be reported as the established cause.

No GSM8K result or general model accuracy qualification has been obtained in
this model-wide extension. See REPORT.md for the final state and measurements.

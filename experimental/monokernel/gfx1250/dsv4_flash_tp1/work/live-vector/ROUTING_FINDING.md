# Image-specific Flash hash-routing finding

Image: `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, local image ID `sha256:d1434db7030a8355a332b2a9237598fadb07a8bf7bb6dbbd081567a13d54686a`.
ATOM image checkout HEAD: `f19778df4054e94d64d6f6bf6dd68f5681bd5945`, with pre-existing working-tree changes. This finding describes this image, not current upstream.

In `Mxfp4MoEMethod.apply`, the Triton TP decode branch recognizes a non-null `custom_routing_function` but calls `routing(router_logits, ...)` without invoking that callback. Flash's first hash layers register `MoE._hash_topk`, which selects experts from checkpoint `tid2eid[input_ids]`. Thus the two paths select different experts.

The isolated copy now calls AITER `routing_from_hash` when the callback owner is a hash layer, preserving the existing token-per-expert block-size heuristic. Non-hash routing is unchanged. Original image files are untouched. Both the corrected native baseline and vector experiment use this patch.

Evidence from the first live decode token of the France smoke prompt, layer 0:
- Vector and the native hash callback selected `[44, 252, 233, 249, 29, 209]`, with matching routing weights.
- Before correcting the actual native decode dispatch, identical-input FFN relative RMS disagreement was about 22%.
- After correcting dispatch, identical-input FFN disagreement was 0.2703%; routed-output best-fit scale 1.00041, routed residual disagreement 0.7431%.
- Full-layer disagreement remains 1.984%; the independent reference also fails the 0.1% acceptance gate at 1.009%. This is NOT a correctness-qualified serving implementation.

The initially investigated BF16 MoE branch is not the active gfx1250 GUGU branch. Active decode internally quantizes activations to MXFP8; the experiment retains FP8 quantization accordingly.

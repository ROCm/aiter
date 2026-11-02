# Actual ATOM Flash BF16 attention microbenchmark

GPU3. Native function is `atom.model_ops.v4_kernels.paged_decode.sparse_attn_v4_paged_decode`, which invokes installed AITER `pa_decode_sparse` with no tuning overrides, `has_invalid=False`. C1 Q=[1,64,512], BF16 cache rows of512, FP32 checkpoint sink, scale512^-0.5. Synthetic Q/cache values; exact installed native dispatch, not a serving trace duration. Parent live-server capture independently confirms both names below.

Native baseline max_num_seqs128 retains the entire16384-entry SWA index buffer. Although only128 indices are active, heuristic chooses128 KV splits. Main kernel `_pa_decode_sparse_BLOCK_H_16_BLOCK_D_512_BLOCK_K_16_H_64_D_512_KV_SPLITS_128`, grid(1,4,128),32threads. Reduce `_pa_decode_sparse_reduce_BLOCK_H_1_BLOCK_D_512_BLOCK_K_16_H_64_D_512_KV_SPLITS_128`, grid(1,64),128threads. Larger CSA/HCA envelopes produce same128-split cap.

Two timing rounds,128 calls/captured graph,5 warmup replays,40 alternating samples, median GPUevent us/call; report second round only. Traces identify kernels; single-call profiler durations are not the reported timings.

| Visible KV rows | Native ATOM attention us | Fused attention+cache write us | Fused splits |
|---|---:|---:|---:|
|128|8.393|9.746|4|
|136|8.369|10.722|2|
|140|8.421|10.376|2|
|143|8.394|10.194|2|
|384|8.667|12.610|4|
|512|8.783|13.953|4|
|639|8.994|15.658|4|

128 is SWA at every decode step for1024input/1024output.136/140/143 represent HCA context lengths1024/1536/2047;384/512/639 represent CSA at those contexts. These are attention-only comparisons; compressor/indexer costs are excluded. The fused HCA rows use split2 because current split4 requires each split own a tile and rejects136..143; they are diagnostics, not the accepted full-prefix path. Native cache write is already fused upstream into Q/KVnorm+RoPE; our attention launch includes a cache write, so row boundaries differ slightly. For whole-prefix comparison, count native cachewrite once with QKV.

Numerics: native-vs-fused relativeRMS is~0.0021, outside the previous0.001 equivalence gate. Both differ slightly from FP32 reference due arithmetic/rounding; fused is closer on these random inputs, but that does not establish native-equivalent full-model outputs. Do not claim validated ATOM replacement from timing. Independent BF16-rounded-reference diagnostics are in envelope-results.json.

Envelope diagnostic (not actual maxseq128 baseline): changing only indices capacity to128 selects8 splits and native attention 5.298us. This demonstrates why capacity is part of exact dispatch reproduction. No native source was modified.

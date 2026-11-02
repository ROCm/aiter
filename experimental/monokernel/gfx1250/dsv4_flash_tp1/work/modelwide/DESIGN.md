# Model-wide Flash C1 extension

The existing full-layer prototype supports layer 0 only and is not accuracy qualified.
This directory implements the missing operations behind an opt-in experimental
serving path; the baseline remains the default. Target: 43 layers (2 uncompressed, 21 ratio-4, 20 ratio-128), 1024 input/1024 output, concurrency one. The config ratio list has 44 entries; its trailing zero is unused.

## Compressor stage

Use AITER gfx1250 a16w16 WMMA/TDM for the BF16 merged KV/gate projection.
After projection publication, one CTA owns each compressor's entire pool, norm,
RoPE, cache scatter and state update. Main and indexer compressors have disjoint
state. Pool reads old state first; the same CTA updates state afterward, avoiding
an extra global barrier. Input/output state slots remain distinct, as in ATOM.
No state update occurs for a sentinel write plan; no scatter for a sentinel
compression plan. All pool/state addresses use 64-bit slot offsets.

| Edge | Producer | Consumer | Publication/reuse |
|---|---|---|---|
| MHC normalized input | MHC grid | projection CTAs | existing device barrier |
| merged BF16 projection | WMMA CTAs | compressor owner CTA | device barrier |
| compressed KV | compressor owner | attention/indexer CTAs | device barrier before cache reads |
| updated ring state | compressor owner | next token | kernel completion, no concurrent request reuse |

Pool arithmetic follows native sequential online softmax and FP32 RMS/RoPE;
BF16 rounding occurs at projection output and final main-cache store, not before RMS.
Bulk row reads use TDM. Metadata scalars use ordinary loads. Initial standalone
validation covers main ratio 4, main ratio 128, FP8 indexer ratio 4, and ring-slot
forks/sentinel behavior on four GPUs. These building blocks are now included in the model-wide serving experiment.

## Implemented full-layer sequence

256 persistent CTAs, 4 waves/CTA, 492 VGPRs, 133088 LDS bytes, zero spills
for the four tested combinations (ratio 0 / 4 / 128, biased ratio 4).

1. Unshuffle residual; grid barrier.
2. QA L2 prefetch; attention mHC; grid barrier.
3. Ratio 4/128: main merged BF16 compressor projection (128/64 CTAs),
   ratio 4 indexer compressor projection (32 CTAs); grid barrier.
4. Main compressor owner CTA 0; indexer compressor owner CTA 1. Each pools
   old state, normalizes, rotates, scatters, then updates its state ring.
   Ratio 4 identity top-k/page translation on CTAs 0..3; grid barrier.
5. Stage native compressed-first plus SWA cache order using TDM and 64-bit
   physical addresses. Load position-specific RoPE tables; grid barrier.
6. Existing QA/KV/QB fused region with WMMA, TDM, QB prefetch; grid barrier.
7. Split attention and reduction, current-token direct overlay; grid barrier.
8. Inverse RoPE, quant, OA, quant, OB, attention mHC post; grid barrier.
9. FFN mHC pre/norm, quant, router (including correction bias), shared expert,
   routed FP4 experts, combine and mHC post, all inside the same launch.
10. Publish current KV to the native SWA destination.

Important gfx1250 difference: native FlyDSL overrides FP8 preshuffle to linear
rows. Generic ATOM Triton fallback uses the old 16x16 preshuffle. The candidate
matches gfx1250's serving-default linear layout.

For context <=2048, index_topk=512 contains every compressed row. AITER C++
`topk_per_row_kernels.cu` branches on row_len<=k and writes identity indices
without consuming logits. The candidate eliminates query projection/scoring
for that guarded context range, but retains indexer compressor state updates.
This is not a general long-context indexer implementation.

The old adapted attention assumed nonempty KV splits; HCA length 136 yields
an empty split and could hang. `attention_extended.py` now publishes neutral
M=-inf/L=0/acc=0 for those splits and still participates in reduction/barriers.
Validated lengths: 1,31,32,33,65,127,128,129,136,144,257,384,513,640.

## Current numerical status

Real checkpoint compressor projection+state tests (layers 2,3,40 and layer-2
indexer) matched projection, cache, and state outputs exactly in the sampled
cases at positions 1023,1024,1151,1535,2047. These use synthetic activations and
historical states. Standalone routing matches native IDs/weights in 200 cases.
The full-layer smoke tests are NOT an accuracy qualification: they use layer-0
attention/FFN weights with synthetic compressor weights. Existing full-layer
native discrepancies can alter expert selection; this remains unresolved.

## Rejected experiment

Parallel softmax reduction inside each TDM tile was slower: HCA compressor
~71us vs ~15.36us sequential tiled, native ~7.44us (round-two graph replay).
Keep it as PARALLEL_POOL ablation, disabled by default. HCA is still slower on
compression boundaries; those occur once per 128 decode tokens.

## Arithmetic-aligned variant (model_layer_matched.py)

The original adapted attention changed two native rounding boundaries: Q was
scaled after the QK dot instead of scaling/rounding Q to BF16 before WMMA, and
probabilities used a high+low BF16 expansion instead of one native BF16 operand.
It also used BK32 and four splits, while the native C1 path uses BK16 and enough
splits for one tile per active split. These are not numerically interchangeable
once attention output enters FP8 quantization and expert routing.

The aligned variant uses scaled FP8 WMMA for QA/QB/OA/OB, native pre-dot Q scaling,
BF16 probabilities, BK16, and 64 KV splits. At most 40 splits are active for the
640-key context cap. All 256 CTAs publish split results, then a device barrier
precedes 64 one-head reduction CTAs. Empty splits publish neutral partials.
This keeps TDM tiled transfers and direct current-token overlay.

Isolated attention matches actual ATOM output bit-for-bit at all 14 tested key
lengths from 1 through 640. BK16 with only 16 splits stops matching beyond 256
keys, confirming the online-softmax partition matters. Full-layer synthetic
smoke: 470 VGPR, 133088 LDS bytes, zero spills, finite output and clean counters.
The model-wide accuracy and serving results for this variant are recorded in
REPORT.md; isolated parity alone is not a model qualification.

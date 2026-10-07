# DeepSeek-V4.1-Flash mono decode layer (blueprint)

One fused decoder layer for vLLM's decode steps on MI355X (gfx950): the
attention seam, the attention, the attention all-reduce, the FFN seam, the MoE
and the MoE all-reduce, in two persistent launches a layer. ATOM's V4.1 mono
(`atom/models/deepseek_v41/mono`) is the reference design; the attention
stages are `dsv41_mega_attn`'s, the rest follows ATOM's stage tables with
vLLM's numerics.

## Scope

- Layers: every backbone layer with the standard seam and no compressor /
  indexer: 3-7, 9-13, 15-19, 21-23, 25-27, 29-31, 33-35, 37-39 (30 of 40).
  Layer 0 (embedding-broadcast seam), layer 1 (Engram at the seam) and the
  compressor / indexer layers 2, 8, 14, 20, 24, 28, 32, 36 keep vLLM's path;
  the two paths interleave freely (same tensors at the layer boundary).
- Steps: decode only (FULL graph or eager), M <= 48 rows (8 requests x 6
  DSpark tokens), TP2 / TP4, fp8_ds_mla KV records (ROCm).
- Contract: a drop-in for `DeepseekV4DecoderLayer.forward`:
  in  `x` (the previous FFN output, reduced, bf16 [M, 5120]), `residual`
      (bf16 [M, 4, 5120]), `post_mix` [M, 4, 1], `res_mix` [M, 4, 4],
      `pre_mix` [M, 4] (f32, the previous seam's)
  out the MoE output (reduced), the residual after the FFN seam, and that
      seam's post / comb / pre mixes.

## Launches

    K1  seam_attn: slice (160 x 32 columns): fold x into the residual
        (post x + sum comb r), the collapsed input (pre_in . R'), partial mixes
        (fn . R') and sums of squares -> gate (S): 24 mixes -> pre / post /
        comb (Sinkhorn 20) -> norm (S): attn_norm -> vLLM MXFP8 -> X8
        attention front (dsv41_mega_attn K1): wqkv, q/kv norms + KV insert, wq_b
    K2  attention back (dsv41_mega_attn K2): split, combine, wo_a, wo_b
        AR1: the bf16 wo_b partial pushed to every rank (system scope), summed
        in rank order 0 .. TP-1 in fp32 -> bf16 (the attention output)
        seam_ffn: slice (fold the attention output), gate, norm -> MoE input
        MoE: xq, router (bf16 GEMV), route (sqrtsoftplus + bias, top-6,
        renormalized x 1.5), shared expert (MXFP8 GEMVs, clamped SiLU), ug
        (MXFP4 x MXFP8 per expert of the step's union), down (+ routing weight,
        top-k order, bf16 adds; + shared) -> AR2 (push, rank-order sum) -> out

## Hand-offs and peer memory

- In a launch: tagged pairs / plain data + flags at device scope (as
  dsv41_mega_attn); the tag is the launch pair's epoch, which the last CTA of
  K2 moves on (no host step hook, graph-safe).
- Across ranks: one symmetric uncached buffer a rank (aiter
  `UncachedIpcHeap`), every rank holding every peer's address; pushes are
  system-scope stores of (value, tag) pairs. The epochs agree across ranks (all
  run the same launches); regions alternate by epoch parity, so a rank one
  layer ahead never overwrites data a peer has not read.

## Numerics (the vLLM path each stage reproduces)

- Seams: aiter `mhc_fused_post_pre_delayed_rmsnorm` (post fold order, fn as
  bf16 hi + lo, products summed in order, Triton sigmoid / exp / rcp).
- Attention: dsv41_mega_attn (vLLM bit-exact projections and KV records).
- MoE: vLLM's AITER a8w4 (`aiter.fused_moe`, interleaved gate/up, swiglu
  limit 10, weights in aiter's (16, 16) shuffle and `shuffle_scale` order, TP4
  intermediate padded to 640); shared expert as vLLM's MXFP8 block-32 linears.
- All-reduce: exact, fp32 in rank order (vLLM: custom all-reduce, or INT4
  quick reduce above its size threshold).

## Validation

- Per stage against vLLM ops; per layer against vLLM's modules in a TP
  process group (seams, attention chain, MoE module, all-reduces).
- Bench: layers of one type back to back in one HIP graph, cold weights,
  us per layer, M = 1..48, TP2 / TP4; then e2e (AgentX, PR 3690 recipe).

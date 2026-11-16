# MLA V4.1 H64 fused A8W4

`mla_sparse_prefill_v41(..., backend="asm")` supports H=64, D=512 on gfx1250 B0+.
Q/SWA use FP8 with E8M0 scales, CSA uses NVFP4 with E4M3 scales, and output is BF16.

- Inputs must be contiguous. CSA data `[P,256]` and scales `[P,32]` are separate
  arrays. Convert interleaved 288-byte rows to these arrays before calling the API.
- `softmax_scale` must equal the FP32 value of `1/sqrt(512)`.
  Sink values must be finite or negative infinity. `attn_sink=None` allocates
  and fills a negative-infinity tensor; prepare the tensor before kernel timing.
- `num_kv_splits` accepts 1..16. With `None`, N > 64 uses 1 split. For N <= 64,
  an average of 512 CSA prefix rows uses 1 split; 1024 uses 4 splits at N <= 32
  and 2 otherwise. Other prefix lengths use the ASM fallback heuristic. SWA
  rows do not count toward the prefix average.
- Grid `(S,N,1)`, block `(128,1,1)`; S>1 uses cluster `(S,1,1)` and fused reduction.
  The API allocates FP32 workspaces `[N,S,65,512]` and `[N,S,64]` for S>1.
- Code object `fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41.co`.
- Symbol `fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41`; 120-byte arguments, 327680-byte static LDS.
- CO SHA256: `ef2f135b06915be3487958d2f21c06da804cf7e5199772e2d4419ce6801b632e`.

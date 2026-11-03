# MLA V4.1 H64 fused A8W4

`mla_sparse_prefill_v41(..., backend="asm")` supports H=64, D=512 on gfx1250 B0+.
Q/SWA use FP8 with E8M0 scales, CSA uses NVFP4 with E4M3 scales, and output is BF16.

- Inputs must be contiguous. CSA data `[P,256]` and scales `[P,32]` are separate
  arrays. Convert interleaved 288-byte rows to these arrays before calling the API.
- `softmax_scale` must equal the FP32 value of `1/sqrt(512)`.
  Sink values must be finite or negative infinity. `attn_sink=None` allocates
  and fills a negative-infinity tensor; prepare the tensor before kernel timing.
- `num_kv_splits` accepts 1..16. `None` uses an initial shape-based heuristic
  that is not performance tuned.
- Grid `(S,N,1)`, block `(128,1,1)`; S>1 uses cluster `(S,1,1)` and fused reduction.
  The API allocates FP32 workspaces `[N,S,65,512]` and `[N,S,64]` for S>1.
- Symbol `mla_v41_fused_cs_v1`; 120-byte arguments, 327680-byte static LDS.
- CO SHA256: `d601c98168d65a92187efc739a3ba4a21b054da986163e4ddffc7b67168d79ca`.

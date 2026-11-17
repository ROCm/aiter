# FlyDSL MXFP6 grouped MoE (gfx950)

Routed-expert span of one SwiGLU MoE layer with OCP MX weights and activations:

    y[t] = sum_j w[t, j] * W2[e_tj] @ (silu(W1g[e_tj] @ x[t]) * (W1u[e_tj] @ x[t]))

| | |
|---|---|
| Weights | MXFP6 E2M3 (default) or E3M2 (`MX6_WFMT=e3m2`), packed 6-bit, E8M0 scale per 32 along K |
| Activations | MXFP6 E2M3 / E3M2 (W6A6, `MX6_AFMT`), or MXFP8 E4M3 (W6A8, `MX6_AFMT=e4m3`); the SwiGLU output is quantized to the same format |
| Shapes | `hidden` and `inter` any multiple of 128; any expert count, top-k and token count; weight tensors > 4 GiB |
| Determinism | bitwise run to run (no atomics on outputs; fixed-order combine) |
| Entry point | `aiter.ops.flydsl.mxfp6_moe.run` (prepare: `run(None, {"w1", "w2"})`; forward: `run({"x", "ids", "w"}, state)`) |

Files: `mxfp6_moe.py` (host API), `mxfp6_moe_quant.py` (torch quantization / packing / weight preshuffle),
`kernels/mxfp6_moe_gemm.py` (grouped GEMM), `kernels/mxfp6_moe_aux.py` (row quant, routers, combine,
shared MX quantizer). Tests and benchmark: `python3 op_tests/test_flydsl_mxfp6_moe.py` (summary tables per test function).

## Pipeline (all FlyDSL, no host syncs)

1. **route** — counting sort by expert: one 1024-thread block (<= 32k entries) or histogram / scan /
   scatter over many blocks. Produces the gather index, the tile -> expert map, the number of used tiles,
   and `pos[t, j]`. Slot order inside an expert comes from LDS atomics and is irrelevant to the result bits:
   every output row depends only on its own input row.
2. **quant_rows** — x to MX rows (MFMA K order) + E8M0 scales packed 4 K-steps per dword.
3. **stage 1** (`epi="swiglu_q"`) — gathered x @ W1^T with gate/up fragments interleaved, SwiGLU in the
   epilogue, and the MX quantization of h in the epilogue (bitwise equal to running `quant_rows` on bf16 h).
4. **stage 2** (`epi="store"`) — h @ W2^T to bf16 rows.
5. **combine** — `y[t] = sum_j w[t, j] * o[pos[t, j]]` in fixed j order, fp32.

## GEMM kernel (`compile_gg6e`)

- `V_MFMA_SCALE_F32_16X16X128_F8F6F4` through `fx.rocdl.cdna4.MFMA_Scale` atoms (one per E8M0
  scale-byte select), FP6 B, FP6/FP8 A. Measured operand contract (see `test_mfma_contract` in the op test):
  FP6/FP4 lanes hold 32 contiguous K values; **FP8 lanes hold two K=64 halves** (bytes 0-15 = K[16g,+16),
  16-31 = K[64+16g,+16)); scale of (row r, K-block b) in lane 16b + r.
- 4 waves; wave tile 128x128 (8x8 fragments) -> block 256x256. 256x128 when N % 256 != 0 or for FP8
  activations (LDS budget); 128-row tiles when experts average <= 160 rows.
- Weights preshuffled once into 1536-B MFMA fragments (64 lanes x 24 B), N-block-major
  (`[E][N/BN][K/128][BN/16][1536]`), so one block's K-slab is contiguous and each wave needs a single
  base offset; activation rows gathered by DMA (row-major in LDS, coalesced 16-B pieces).
- Activation scales stored `[rows][K/512][4]` (four 128-K steps per 16 B), so a wave DMAs 64 rows' scales
  with one 16-B load per quad. Fewer live address registers cut the accumulator AGPR<->VGPR moves in the
  K loop (268 -> 234 reads per K step) and gained 4-5% on every layer.
- 3-stage LDS ring (48 KB/stage at 256x256) filled by **async** buffer->LDS DMA in 16-B pieces
  (`raw_ptr_buffer_load_async_lds` + `asyncmark` / `wait_asyncmark`; the synchronous variant made LLVM put
  `vmcnt(0)` before every LDS read). Scales are DMA'd per 4 K-steps into two LDS slots (LDS total 160 KB).
- The DMA issue is **interleaved across the 8 MFMA groups** of each K step (ATT: back-to-back DMA issue
  stalled the single wave per SIMD; +33% on stage 1).
- `sched_barrier(0)` around every `s_barrier` and a full drain before the epilogue (raw barriers have no
  memory semantics; without these some tile configs raced).
- XCD-contiguous, grouped tile order over the used tiles only (the router writes the used-tile count).
- Epilogue staged through LDS (padded rows) with 16-B coalesced stores.
- All per-expert / per-tile bases are 64-bit (rebased buffer descriptors), so weights > 4 GiB work.

## Results (MI355X, 1 GPU, exclusive; AITER main + FlyDSL 0.5.0)

E=64, balanced top-k routing, 20 timed iterations of the whole op (routing, quantization, both GEMMs,
combine); TFLOP/s counts the useful 6*T*k*H*I flops:

| Layer (H, I, E, k) | 4k tokens | 32k tokens |
|---|---|---|
| W6A6 s (6144, 2048, 64, 4) | 0.742 ms / 1666 TF/s | 4.349 ms / 2275 TF/s |
| W6A6 m (8192, 4608, 64, 4) | 1.808 ms / 2053 TF/s | 12.201 ms / 2433 TF/s |
| W6A6 l (10496, 3072, 64, 4) | 1.616 ms / 1961 TF/s | 10.241 ms / 2476 TF/s |
| W6A8 s | 0.868 ms / 1425 TF/s | 6.205 ms / 1595 TF/s |
| W6A8 m | 2.171 ms / 1710 TF/s | 15.243 ms / 1948 TF/s |
| W6A8 l | 1.907 ms / 1662 TF/s | 13.203 ms / 1921 TF/s |

The GPU runs at its power cap here (~1350 W, ~2040-2140 MHz); about half of the K-loop VALU issue is
accumulator AGPR<->VGPR traffic forced by the 256 fp32 accumulators per lane of the 128x128 wave tile.

rel L2 vs bf16: W6A6 E2M3 0.074, W6A8 0.068 (E3M2: 0.107-0.130, format-inherent). Against a torch reference
with identical MX numerics the plugin agrees to 1.6e-5 (E2M3) .. 1.6e-3 (with E4M3/E3M2), see
`test_mxfp6_moe` (0 elements outside rtol/atol 2e-2).

Also verified correct and deterministic: hidden 6272 (a multiple of 128 but not 256), E=160 / E=192 at
hidden 8192 x inter 4096 (both weight tensors > 4 GiB packed), hidden 10496 at 256 / 1024 tokens.

## Known limits / next steps

- The global->LDS DMA stays a raw `rocdl.raw_ptr_buffer_load_async_lds`: `BufferLoadAsyncLDS128b/32b` copy
  atoms gave identical results but grew the K loop from ~2.8k to ~5.1k instructions, spilled 12 VGPRs
  (scratch reloads wait on vmcnt and drain the async DMAs) and cost 8-12% on the s/m layers (FlyDSL 0.5.0).
- Register pressure: the MFMA AGPR-destination form (`-amdgpu-mfma-vgpr-form=false`), 8 waves, and a
  piece-major A layout all spilled or coalesced worse (15-23 ms vs 12.2 ms on m 32k).
- Stage 2 (short K) carries ~5 us fixed cost per tile (pipeline fill + epilogue). Tried and not adopted:
  a persistent variant overlapping the next tile's prologue with the epilogue (correct, but spills; 1.67
  vs 2.12 PF/s on the stage-2 shape) and smaller 128x128 / 128x256 tiles for two blocks per CU (12-22% /
  2-8% slower than 256x256).
- FP8 activations use 256x128 tiles (256x256 does not fit three LDS stages).
- The combine re-reads the stage-2 rows (near HBM bandwidth); fusing it needs an ordered cross-tile reduction.

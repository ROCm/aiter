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

- `V_MFMA_SCALE_F32_16X16X128_F8F6F4`, FP6 B, FP6/FP8 A. Measured operand contract (see `test_mfma_contract` in the op test):
  FP6/FP4 lanes hold 32 contiguous K values; **FP8 lanes hold two K=64 halves** (bytes 0-15 = K[16g,+16),
  16-31 = K[64+16g,+16)); scale of (row r, K-block b) in lane 16b + r.
- 4 waves; wave tile 128x128 (8x8 fragments) -> block 256x256. 256x128 when N % 256 != 0 or for FP8
  activations (LDS budget); 128-row tiles when experts average <= 160 rows.
- Weights preshuffled once into 1536-B MFMA fragments (64 lanes x 24 B); activation rows gathered by DMA.
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
| W6A6 s (6144, 2048, 64, 4) | 0.777 ms / 1592 TF/s | 4.560 ms / 2170 TF/s |
| W6A6 m (8192, 4608, 64, 4) | 1.896 ms / 1957 TF/s | 12.810 ms / 2317 TF/s |
| W6A6 l (10496, 3072, 64, 4) | 1.704 ms / 1860 TF/s | 10.753 ms / 2358 TF/s |

On the previous FlyDSL (0.3.4.1) the same op measured top-8 m 24.4 ms at 32k (2433 TF/s) and W6A8
1.40-1.88 PF/s; the migration to FlyDSL 0.5.0 / tensor_shim buffer views is within 0-2% of it.

rel L2 vs bf16: W6A6 E2M3 0.074, W6A8 0.068 (E3M2: 0.107-0.130, format-inherent). Against a torch reference
with identical MX numerics the plugin agrees to 1.6e-5 (E2M3) .. 1.6e-3 (with E4M3/E3M2), see
`test_mxfp6_moe` (0 elements outside rtol/atol 2e-2).

Also verified correct and deterministic: hidden 6272 (a multiple of 128 but not 256), E=160 / E=192 at
hidden 8192 x inter 4096 (both weight tensors > 4 GiB packed), hidden 10496 at 256 / 1024 tokens.

## Known limits / next steps

- Stage 2 (short K) still carries ~5 us fixed cost per tile (pipeline fill + epilogue). A persistent variant
  that overlaps the next tile's prologue with the epilogue was prototyped but spilled registers (K-loop
  scratch reloads wait on `vmcnt`, which drains the async DMAs); not included.
- FP8 activations use 256x128 tiles (256x256 does not fit three LDS stages).
- The combine re-reads the stage-2 rows (near HBM bandwidth); fusing it needs an ordered cross-tile reduction.

# HipKittens block-FP8 MoE

`aiter.hk_fp8_moe` computes routing, gate/up projection, SwiGLU, down projection, and weighted combine
on gfx950. The grouped GEMMs use four waves per workgroup, persistent scheduling, and HipKittens ART
tiles. FP8 scales are applied with `V_MFMA_SCALE`.

## Supported inputs

| Parameter | Requirement |
|---|---|
| Shapes | Hidden dimension 4096, intermediate dimension 1024, top-k 8 or 9, 1 to 1024 experts, at most 524287 tokens |
| Weights | FP8 e4m3: `w13` [E, 2048, 4096] with gate rows followed by up rows, and `w2` [E, 4096, 1024], both in `shuffle_weight(w, (16, 16))` layout |
| Weight scales | FP32 powers of two for 128x128 blocks: `w13_scale` [E, 16, 32] and `w2_scale` [E, 32, 8]. Only the UE8M0 exponent is used. |
| Activations | BF16 input and output; internal FP8 e4m3 quantization with a power-of-two scale per (token, 128 columns) |
| Routing | Int32/int64 expert IDs and floating-point weights of shape [T, top-k]; out-of-range expert IDs contribute nothing |
| SwiGLU | `silu(min(gate, L)) * clamp(up, -L, L)` when `swiglu_limit` L > 0; otherwise `silu(gate) * up` |

Weights with arbitrary block scales must be requantized before calling the operator. The Python
wrapper allocates the workspace through PyTorch's caching allocator. All work runs on the current
stream and supports graph capture after JIT warmup. All tensors must be on the same gfx950 device.
Weights and an explicit output buffer must have 16-byte-aligned base pointers.

## Pipeline

The forward pass clears the expert counts and launches seven kernels:

1. `count_and_quant`: count selections per expert and quantize the input activations.
2. `moe_scan`: prefix-sum the counts into expert row ranges padded to 256 rows, assign expert IDs
   to row tiles, and mark padding rows.
3. `moe_scatter`: place token IDs and routing weights into their expert row ranges.
4. `build_gather_tables`: prepare input gather offsets and packed activation scales for stage 1.
5. `moe_gemm_kernel<2048, 4096, true>`: compute gate/up projections, apply SwiGLU, and quantize
   the result to FP8 with packed scales for stage 2.
6. `moe_gemm_kernel<4096, 1024, false>`: compute the down projection for each sorted row.
7. `moe_reduce_w`: combine each token's selected expert outputs using its routing weights.

Routing processes 4096 selections per 1024-thread block. For up to 304 experts, the histogram and
row offsets are striped by lane to avoid LDS bank conflicts. Two 16-bit counts share each histogram
word; the block's selection bound prevents carries between them. Larger expert sets use global atomics.

Each GEMM workgroup processes 256x256 output tiles. Its two-step K pipeline continues across tile
boundaries, and the down GEMM overlaps output stores with the next tile's MFMAs. A small-batch variant
skips MFMAs for tile halves containing only padding rows.

The kernels, launch sequence, workspace query, and AITER ctypes entrypoints are in `hk_moe.cu`.

## Build and test

AITER JIT-builds `module_hk_fp8_moe` on first use. HipKittens headers come from `HIP_KITTENS_DIR`
when set; otherwise AITER fetches the pinned revision.

Run from the repository root on gfx950:

```bash
python3 op_tests/test_hk_fp8_moe.py
python3 op_tests/test_hk_fp8_moe.py --routing-only
```

The default suite checks an FP32 reference and reports timings for 1 to 16384 tokens, top-k 8 and 9,
and SwiGLU with and without clamping. Top-k 9 includes a shared expert selected by every token.
Graph replay checks change inputs and exercise random, concentrated, bank-aliasing, partially invalid,
and all-invalid routing around block boundaries. Use `--tokens` to select token counts for the main
sweep; `--routing-only` runs routing replay and unsupported-shape checks.

## Pinned registers

The GEMM reserves v128-v255 for operand tiles and all 256 AGPRs for accumulators. With this AGPR
allocation, `amdgpu_num_vgpr(128)` does not keep compiler-generated instructions out of v128-v255.
Changes that increase register pressure can therefore corrupt operand tiles.

After kernel or compiler changes, run the correctness suite with a fresh JIT cache, then check its
compiler configuration for VGPR spills and instructions that touch pinned registers. Run from the
repository root:

```bash
export AITER_JIT_DIR="$(mktemp -d /tmp/aiter-hk-moe.XXXXXX)"
GPU_ARCHS=gfx950 python3 op_tests/test_hk_fp8_moe.py
python3 op_tests/test_hk_fp8_moe_codegen.py \
    --build-dir "$AITER_JIT_DIR/build/module_hk_fp8_moe/build" \
    --output /tmp/hk_moe.s
```

The code-generation test obtains the `hk_moe.cu` compile command from `ninja -t compdb`, preserving
the JIT flags and changing only the output to device assembly. All four grouped GEMM variants must
avoid compiler instructions that touch pinned registers, and all VGPR spill counts must be zero.
Inline assembly implementing HipKittens tiles is excluded from the register check.

To check saved assembly without PyTorch, a GPU, or the HIP compiler:

```bash
python3 op_tests/test_hk_fp8_moe_codegen.py --assembly /tmp/hk_moe.s
```

## Compare with current AITER

```bash
python op_tests/op_benchmarks/bench_hk_fp8_moe.py --output /tmp/hk-moe-benchmark.json
```

The benchmark compares complete MoE calls with AITER's checked-in GLM-5.3-Flash tuning and its
default heuristic. It uses identical inputs and weights, checks an FP32 reference, and records dispatch
metadata and 21 alternating timing samples for each implementation. Timings use warm HIP graphs
with ten calls per replay; routing, quantization, both GEMMs and weighted combination are included.
Weight preparation, router top-k and compilation are excluded.

It also compares AITER's native FlyDSL MXFP8 path, its fused shared-expert path for top-k 9,
and their stage-two reduction variants. The original FP8 weight payloads are unchanged; each
power-of-two 128x128 scale is repeated losslessly into per-32 E8M0 scales before weight shuffling.
This changes activation quantization to per-32. One-time weight repacking is excluded, while
runtime activation quantization is included. Every candidate must pass the same FP32-reference
check. Failed candidates are retained in the JSON rather than counted as valid baselines.

The native candidate sweep also retargets the checked-in GLM-5 and MiniMax-M3 MXFP8 recipes
to this operator's dimensions and expert counts. This includes the newer stage-two layouts and
fused FP8 intermediates. Each recipe must be selected by the dispatcher and pass the same
numerical check before it is ranked; the original recipes were tuned for different shapes.

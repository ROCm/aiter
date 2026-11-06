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
wrapper allocates the workspace through PyTorch's caching allocator. Padding is bounded by
`min(E, T * topk)` nonempty experts, so a one-token top-8 call uses about 18 MiB of workspace
even with 1024 experts. All work runs on the current
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
when set; otherwise AITER fetches the pinned revision and updates an older clean managed checkout.
A modified managed checkout fails with an actionable error; an explicit `HIP_KITTENS_DIR` is never
rewritten. Bundled headers without Git metadata are preserved; AITER fetches a verifiable checkout
in the JIT cache instead. Managed Git commands trust only the exact checkout path for that command,
and reject tracked or untracked changes even at the expected pin. A checkout-specific lock is shared
across JIT directories and symlink aliases; waiting builders revalidate the checkout after acquiring it.
The module is experimental and excluded from aggregate prebuilds unless
`AITER_ENABLE_EXPERIMENTAL=1`. Device implementations are guarded for gfx950; the host entrypoint
rejects unsupported devices before launching kernels, caching architecture properties per device.

Run from the repository root on gfx950:

```bash
python3 op_tests/test_hk_fp8_moe.py
python3 op_tests/test_hk_fp8_moe.py --routing-only
python3 op_tests/test_hk_fp8_moe_codegen.py
```

The default suite reports timings and compares against a quantization-aware FP32 reference for
1 to 16384 tokens, top-k 8 and 9, and SwiGLU limits 0, 1, and 10. The reference independently models
both power-of-two 1x128 E4M3 activation rounds and BF16 stores. The arithmetic residual must stay
below 0.01 relative L2; quantization displacement and total error against the original ideal FP32
reference are reported separately. The limit-1 cases require a clamp effect above 0.12 relative L2,
so bypassing the clamp cannot pass. Top-k 9 includes a
shared expert selected by every token. Graph replay uses an active limit of 1 and checks changed
inputs with random, concentrated, bank-aliasing, partially invalid,
and all-invalid routing around block boundaries. Both top-k values also cover 1, 304, 305, and 1024
experts, including the routing implementation boundary. Use `--tokens` to select token counts for the
main sweep; `--routing-only` runs routing replay and unsupported-shape checks.

## Pinned registers

The GEMM reserves v128-v255 for operand tiles and all 256 AGPRs for accumulators. With this AGPR
allocation, `amdgpu_num_vgpr(128)` does not keep compiler-generated instructions out of v128-v255.
Changes that increase register pressure can therefore corrupt operand tiles. Every JIT build containing
`hk_moe.cu` for gfx950 emits device assembly using the actual Ninja compile command and audits it
before publishing the shared library. This also applies to experimental aggregate prebuilds. An audit
failure prevents installation; compiler upgrades cannot silently bypass the check.

After kernel or compiler changes, run the correctness suite with a fresh JIT cache. The standalone
audit can also retain assembly for inspection. Run from the
repository root:

```bash
export AITER_JIT_DIR="$(mktemp -d /tmp/aiter-hk-moe.XXXXXX)"
GPU_ARCHS=gfx950 python3 op_tests/test_hk_fp8_moe.py
python3 op_tests/test_hk_fp8_moe_codegen.py \
    --build-dir "$AITER_JIT_DIR/build/module_hk_fp8_moe/build" \
    --output /tmp/hk_moe.s
```

With no arguments, the code-generation test builds and audits on gfx950, skipping other devices.
Explicit `--build-dir` or `--assembly` checks also work without a GPU. The test obtains the
`hk_moe.cu` compile command from `ninja -t compdb`, preserving
the gfx950 JIT flags and changing the output to device assembly (other targets are removed
when auditing a multiarchitecture build). All four grouped GEMM variants must
avoid instructions that touch pinned registers except for the four explicit HipKittens tile forms:
LDS operand reads, scaled FP8 MFMA, accumulator reads, and accumulator writes. Their
compiler-selected address, scale, and copy operands must still be below v128. Other inline assembly
is checked too, including BF16 conversion. Decimal and hexadecimal register notation are recognized.
Every GEMM must report `.agpr_count: 256` and `.vgpr_count: 512` in its kernel descriptor;
missing or reduced allocations fail the build. It must also have zero VGPR spills,
zero private-segment bytes, no dynamic stack, and no scratch
instructions. SGPR-to-VGPR lane spills are allowed only within the checked temporary registers.

CPU regression coverage for the audit and dependency upgrades runs with:

```bash
python3 -S op_tests/test_hk_fp8_moe_build.py
python3 -S op_tests/test_hk_fp8_moe_checkout.py
```

The independent reference also has CPU-only regressions requiring PyTorch:

```bash
python3 -m unittest op_tests.test_hk_fp8_moe_reference
```

To check saved assembly without PyTorch, a GPU, or the HIP compiler:

```bash
python3 op_tests/test_hk_fp8_moe_codegen.py --assembly /tmp/hk_moe.s
```

## Compare with current AITER

```bash
python op_tests/op_benchmarks/bench_hk_fp8_moe.py --output /tmp/hk-moe-benchmark.json
```

The benchmark compares complete MoE calls with AITER's checked-in GLM-5.3-Flash tuning and its
default heuristic. It retains the original common ideal FP32 reference across differing quantization
schemes. It uses identical inputs and weights and records dispatch
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

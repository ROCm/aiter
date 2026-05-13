# CK W4A16 b_scale GEMM (gfx1151 / RDNA 3.5)

CK WMMA W4A16 b_scale GEMM with both symmetric (`uint4b8`) and asymmetric (AWQ
per-group zero-point) variants. Tuned for gfx1151 (Strix Halo / RDNA 3.5),
covering the four prefill linear columns of Qwen3-4B-AWQ:

| Layer        | N      | K     |
|--------------|-------:|------:|
| `gate_up_proj` | 19456  | 2560  |
| `qkv_proj`     | 6144   | 2560  |
| `o_proj`       | 2560   | 4096  |
| `down_proj`    | 2560   | 9728  |

Same kernel binary handles all four shapes; M is dynamic. K must be a multiple
of `KPerBlock=32` and `Scale_Block_K=128`. Group size is fixed at 128. Weight
dtype is `pk_i4`; activation / scale / output dtype is fp16 or bf16 (selected
by the output tensor's dtype at dispatch). Asymmetric variant requires the CK PR adding the
per-group zero-point machinery to `wmma_cshuffle_v3_b_scale` (see
ROCm/rocm-libraries `[CK] Add per-group zero-point support to
wmma_cshuffle_v3_b_scale (asymmetric int4)` — branch
`users/marcusr/ck_asym_support`); symmetric variant builds against vanilla
upstream CK.

## Status

First-cut drop of the kernel + dispatcher, structured to match
`csrc/ck_gemm_a4w4_blockscale/`. Provides:

- `gemm_w4a16.cu` — single torch op `gemm_w4a16(in_a, in_b, in_s, Y, group_size, scaled_zp=None)`
  covering both symmetric (uint4b8) and asymmetric (AWQ) variants. Pass
  `scaled_zp=None` for symmetric, pass `(zp - 8) * scale` precomputed at weight
  load for asymmetric. Output `Y` is caller-allocated (matches aiter
  convention). Dispatches between fp16 and bf16 instantiations on the output
  dtype, mirroring the F16/B16 split in `csrc/ck_gemm_a4w4_blockscale/`.
- `include/gemm_w4a16_common.cuh` — type aliases, constants, and the
  gfx1151-tuned `DeviceGemmInstance<T>` template.

Still missing for full aiter integration (matching `ck_gemm_a4w4_blockscale/`):

- `gemm_w4a16_manifest.h` — auto-generated kernel manifest
- `gemm_w4a16_lookup.h` — auto-generated `(gfx, cu_num, M, N, K)` lookup table
- `gen_instances.py` — generates per-shape kernel instances
- `gemm_w4a16_tune.{cu,py}` — tuning harness following aiter's CSV-driven model
- `aiter/configs/w4a16_untuned_gemm.csv` + tuned-CSV under `model_configs/`
- pybind registration in `csrc/pybind/`
- Python op surface under `aiter/ops/` for `gemm_w4a16`
- `op_tests/test_gemm_w4a16.py` — correctness + perf smoke
- `setup.py` integration to compile against the bundled CK submodule with
  `GPU_ARCHS=gfx1151` (must use a CK SHA that has the asymmetric machinery PR
  applied for the asymmetric variant)

The single-config kernel + single-function dispatcher is intentionally simpler
than the full aiter pattern — there's only one tuned tile config so the
manifest/lookup scaffolding is overkill until we add bf16 or per-shape variants.

## Performance reference (Strix Halo, gfx1151, fp16)

Standalone CK at the gate_up_proj column (N=19456, K=2560), the dominant
prefill GEMM by FLOPs in Qwen3-4B prefill (~2× the next-largest):

| M    | TFLOPS | % of 38.6 hipBLASLt fp16 roofline |
|-----:|-------:|----------------------------------:|
| 256  | 30.1   | 78% |
| 512  | 31.4   | 81% |
| 1024 | 31.4   | 81% |
| 1920 | 31.1   | 81% |
| 2048 | 30.9   | 80% |
| 3968 | 30.0   | 78% |
| 4096 | 30.1   | 78% |
| 8192 | 30.0   | 78% |

Triton W4A16 baseline at the same shape: 15.6 TFLOPS (M=3968) → CK is 1.92×.

## E2E reference (Qwen/Qwen3-4B-AWQ, num_prompts=10, fp16, default chunk=2k)

Median TTFT, with all four columns dispatched to CK (asymmetric variant)
through the vllm-side dispatcher in `HybridW4A16LinearKernel`:

| Config                | TTFT median |
|-----------------------|------------:|
| Triton baseline       | 1772 ms     |
| **CK ON (all 4)**     | **1571 ms (−11.4%)** |

Decode unchanged (TPOT 16.3 ms — CK only fires on prefill).

## Consumer

The vllm-side dispatcher and weight-repack helpers live at:

- `vllm/model_executor/kernels/linear/mixed_precision/hybrid_w4a16.py` — branch
  `marcusr/aiesw-32176-w4a16-ck-wmma` on ROCm/vllm.
- The dispatcher uses a per-layer min-M threshold; CK fires when the layer's
  `(N, K, group, dtype)` matches a registered shape and the runtime M ≥ 256.

## JIRA

AIESW-32176.

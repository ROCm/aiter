# Delayed mHC Seam (gfx942)

`mhc_fused_post_pre_delayed` is the delayed ("shifted") mHC seam of the DeepSeek-V4 family on gfx942
(MI300X, MI325X): the post-mix of the previous sub-layer, the collapse of the residual streams with
the carried pre gate, and the next seam's gates (pre, post, comb with Sinkhorn) from one projection
of the new residual. It runs asm kernels only; unsupported configurations raise instead of falling
back.

## Scope

- gfx942, 4 residual streams, hidden size 5120 (`mhc_fused_post_pre_delayed_asm_supported`).
  Other architectures, stream counts and hidden sizes raise `NotImplementedError`.
- bf16 residual and sub-layer output; fp32 gates and gate projection `fn` [24, 4·H].
- Any token count; launches are split every 65,536 tokens so every buffer stays under 4 GiB.
- `pre_mix=None` is the identity pre gate (the collapse is stream 0).
- The RMSNorm that follows the seam is not folded in.

Other configurations: `mhc_fused_post_pre_delayed_rmsnorm` (Triton, folds the RMSNorm) or
`mhc_post` + `mhc_pre`.

## API

```python
residual_out, post_mix, comb_mix, layer_input, next_pre_mix = mhc_fused_post_pre_delayed(
    residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
    hc_post_mult_value, sinkhorn_repeat, pre_mix, sublayer_out, post_layer_mix, comb_res_mix,
)
```

The arguments and results are those of `mhc_fused_post_pre_delayed_rmsnorm` without the norm.

## Kernels

Two launches at every token count:

| Tokens | Seam kernel | Then |
|---|---|---|
| T < 60 | `mhc_seam_v4s`: one wave per 128-column slice and token; post, collapse and the pre-GEMM partials | `mhc_seam_gates_b40` |
| 60 ≤ T < 8192 | `mhc_seam_v3_b2`: persistent; post, collapse, MFMA pre GEMM and RMS sum in one pass over the residual | `mhc_seam_gates_b40` |
| T ≥ 8192 | `mhc_seam_v3_b2_stnt`: as above with nontemporal stores | `mhc_seam_gates_b40` |

`mhc_seam_gates_b40` reduces the split-K partials and computes the gates; Sinkhorn runs on DPP
lanes. The switch at 60 tokens is where the two seam kernels cross on 304-CU parts.

## Exactness Contract

- `residual_out` equals `mhc_post` bit for bit (same fp32 operation order and bf16 truncation).
- `layer_input` equals a sequential fp32 collapse rounded to bf16, bit for bit.
- The gates differ from `mhc_pre`'s only in the split-K summation order; their error against fp32
  gates stays at or below `mhc_pre`'s own.

## Compile And ABI Rules

1. The op is registered through `torch_compile_guard` with a fake that returns the output shapes;
   it works under `torch.compile(fullgraph=True)` and in CUDA graphs.
2. Kernel arguments follow AITER's asm layout: one 16-byte slot per argument (pointer + `p2`,
   uint32 + `p3`); the kernels build their buffer descriptors from the pointers and sizes
   (176 bytes for the seam kernels, 224 for the gates).
3. Loader entry points use the ctypes error wrapper, so failed checks raise `RuntimeError`.
4. Keep `Optional[T]` in public, fake and custom-op declarations.

## Validation

Run `python op_tests/test_mhc_seam.py`: T = 1 to 32,768 with carried and identity pre gate,
bit-exact checks of `residual_out` and `layer_input`, gate bounds, the fused kernel's split-K
partials against fp64, guard checks for unsupported configurations and loader errors, and timing
against `mhc_post` + `mhc_pre` and `mhc_fused_post_pre_delayed_rmsnorm`. Kernel changes
additionally require validation of every code object with negative controls and graph-timed runs
across the switch point.

Key implementation locations:

- Python op: `aiter/ops/mhc.py`
- Host launcher: `csrc/py_itfs_cu/asm_mhc_seam.cu`
- Registry and binaries: `hsa/gfx942/mhc_seam/` (`mhc_seam.csv`)

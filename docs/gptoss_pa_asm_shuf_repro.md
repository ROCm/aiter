# GPT-OSS gfx1250 PA ASM Repro Notes

This branch is the AITER side paired with the ATOM branch:

```text
yhl/gptoss-pa-asm-shuf-repro-20260611
```

It contains the gfx1250 PA decode BF16 ASM support and runtime fallbacks needed
by the ATOM repro branch. Use it from ATOM through:

```bash
PYTHONPATH=/app/aiter:/app/aiter/aiter/jit/utils:/app/ATOM:/app/triton/python
```

## Expected Runtime Build Artifacts

The first run may JIT-build metadata and PA decode bindings:

```text
module_ps_metadata
module_pa_decode_bf16_asm
```

After build/import, the PA ASM kernel load should look like:

```text
LoadKernel: _ZN5aiter31pa_decode_bf16_d64_page256_gqa8E hsaco: /app/aiter/hsa//gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8.co
```

## ATOM Launch Contract

The paired ATOM branch expects:

- `ATOM_GPTOSS_USE_PA_DECODE_BF16_ASM=1`
- `ATOM_USE_UNIFIED_ATTN=1`
- `--kv_cache_dtype fp8`
- `--block-size 256`
- GPT-OSS head dimension 64 and GQA ratio 8 for eligible full-attention decode
  layers.

Prefill and SWA layers are intentionally handled by Triton unified attention;
the PA ASM kernel is used only for eligible decode layers.

## Validated Smoke

The paired ATOM branch was launched on `HIP_VISIBLE_DEVICES=1`, port `8013`,
with model path:

```text
/data/models/gpt-oss-120b
```

Ten short prompts returned HTTP 200. The final answers after the model's
`assistantfinal` marker were correct.

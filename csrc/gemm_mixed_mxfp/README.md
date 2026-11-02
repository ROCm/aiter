# Mixed MXFP GEMM tuning

The A6W4 and A4W6 operators select prebuilt gfx950 assembly kernels by shape.
Both operands are already transformed into kernel-native tiled layouts by
`quant_mxfp6_gemm` or `quant_mxfp4_gemm`; the manifest's `bpreshuffle=0`
means that the legacy A4W4 optional B-preshuffle ABI is not used.
`pack_layout` identifies the mixed kernel's actual packed layout.

Kernel names encode scheduling choices: `s0` is natural workgroup order,
`s3` is grouped-M swizzling, `aN` is the MFMA32 lookahead distance, and `nt`
uses non-temporal output stores.

Tune the canonical shape lists with:

```bash
python3 csrc/gemm_a6w4/gemm_a6w4_tune.py
python3 csrc/gemm_a4w6/gemm_a4w6_tune.py
```

Use `-i` and `-o` to provide another untuned or tuned CSV. Each input row uses
logical `M,N,K`; tuned rows are keyed by `gfx,cu_num,M,N,K`. Runtime lookup
tries an exact shape before its physical 256x256x128 padded shape.

The tuner enumerates every compatible row in the corresponding HSA manifest,
rejects unsupported swizzle bounds and split-K, validates outputs against the
safe default kernel, and applies the same paired minimum-gain guard as A6W6.

Useful options include:

- `--shape_grouped` to reuse packed operands across candidates on each worker.
- `--min-gain-pct` to change the default 0.5% retention threshold.
- `--disable-gain-guard` for diagnostic sweeps only.
- `--run_config [CSV]` to validate production dispatch.
- `--compare --update_improved` to update only shapes that improve.

Override runtime configuration paths with `AITER_CONFIG_GEMM_A6W4_ASM` or
`AITER_CONFIG_GEMM_A4W6_ASM`.

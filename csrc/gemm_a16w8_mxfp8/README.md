# A16W8 MXFP8 GEMM tuning (gfx942)

`gemm_a16w8_mxfp8_asm` selects a prebuilt gfx942 assembly kernel from
`hsa/gfx942/a16w8gemm/a16w8gemm_mxfp8.csv` by shape. The tuned table is keyed
by `gfx,cu_num,M,N,K`; at run time a call with M rows uses the entry with the
smallest tuned M at or above M for its `(gfx, cu_num, N, K)`. Shapes without an
entry run a default kernel and `is_gemm_a16w8_mxfp8_tuned()` returns False, so
engines keep BF16 there.

Kernel names encode the tile: `tr` 16-row tiles (M <= 16 * tr), `tn` 16-column
tiles per workgroup (N % (16 * tn) == 0), `nw` waves splitting K inside the
workgroup; `sk` kernels also split K over `splitK` workgroups (1 to 8), `xl`
stages the activations through LDS and `wd2` / `wd3` prefetch the weights two or
three K steps ahead.

## Files

- `aiter/configs/a16w8_mxfp8_asm_{tuned,untuned}_gemm.csv`: the family's
  default files (header only).
- `aiter/configs/model_configs/dsv41_a16w8_mxfp8_asm_{tuned,untuned}_gemm.csv`:
  DeepSeek-V4.1-Flash, tensor parallel 4, per-rank projections, M = 1, 12, 24,
  36, 48, 64, tuned on MI325X (gfx942, 304 CUs).

At run time `AITER_CONFIGS.AITER_CONFIG_GEMM_A16W8_MXFP8_ASM_FILE` merges the
default file with every `model_configs/*a16w8_mxfp8_asm_tuned_gemm*.csv`.
Setting `AITER_CONFIG_GEMM_A16W8_MXFP8_ASM` (one path, or several joined with
`:`) replaces that set.

## Re-tuning

Re-tune the DeepSeek-V4.1-Flash rows on one gfx942 GPU with:

```bash
python3 csrc/gemm_a16w8_mxfp8/gemm_a16w8_mxfp8_tune.py \
    -i aiter/configs/model_configs/dsv41_a16w8_mxfp8_asm_untuned_gemm.csv \
    -o aiter/configs/model_configs/dsv41_a16w8_mxfp8_asm_tuned_gemm.csv \
    --all --shape_grouped
```

Without `-i` / `-o` the tuner reads and writes the default files. For every
shape it:

1. times each registered kernel that admits it (the split-K kernels at splits
   2, 3, 4, 6 and 8, within the op's split-K workspace) with the common tuner
   (`mp_tuner`, kernel time) and checks the outputs against a float64
   reference;
2. times the 8 fastest again in CUDA graphs, together with BF16 `F.linear` on
   the dequantized weights (the GEMM the kernel replaces), interleaved over 5
   rounds, with the weights rotated over more than 1 GiB of copies so the
   Infinity Cache does not serve them. Decode runs in CUDA graphs, and per-launch
   costs that the kernel time leaves out differ between kernels, so the lowest
   median graph time picks the kernel;
3. writes a row only when that kernel is faster than BF16; shapes that BF16 wins
   have no row (and lose an old one when re-tuned).

`us` is the kernel's median graph time per call.

Kernels of a few microseconds vary between runs by a few percent, so close
candidates (for example two split counts of one kernel) can swap between runs.
Expect comparable results, not identical ones.

Useful options:

- `--splitk-values` to change the split counts tried for the split-K kernels.
- `--graph-topk`, `--graph-iters`, `--graph-reps` for the graph timing.
- `--min-gain-pct` to require a margin over BF16 (default 0: strictly faster).
- `--disable-bf16-guard` for diagnostic sweeps only (fastest kernel time, every
  shape written).
- `-o2 PROFILE.csv` to keep the kernel time of every candidate.
- `--run_config [CSV]` to validate production dispatch.

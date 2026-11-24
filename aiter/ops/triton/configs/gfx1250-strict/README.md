# gfx1250-strict configs: NOT TUNED

gfx1250-strict reads only this tree. The tables here are copied from gfx1250 so
the kernels run; they are not tuned. Replace a family's tables with measured
values when it is tuned.

## How the tables were made

Every gfx1250 family has a `DEFAULT.json` here. Shape-specific files
(`<CONFIG_NAME>-N=...-K=....json`) are not copied, except for
fused_fp8_bmm_rope_cat_and_cache_mla.

- Bucketed tables keep only the `"any"` entry (the `bm*_any` entries for
  Gluon MoE), except:
  - fused_sigmoid_mul also keeps `"strided"`.
  - afp4wfp4 preshuffled also keeps the small-M entries, needed for M < 32.
  - Gluon quant also keeps one bucket key, because the loader reads the axis
    order from the bucket keys.
- Triton MoE a8w4 is `{}`: the kernel uses its built-in defaults.
- Tables keyed by kernel variant (attention, KDA, grouped GEMM,
  block_cat_fused) are exact copies, because the kernels look up every key by
  name.
- These are exact copies, because `"any"` alone breaks them:
  - unified_attention (Triton and Gluon): some shapes give wrong results.
  - gemm_afp4wfp4 (Triton): small M gets 256x256x256, and the fused split_cat
    kernel that reads this table hangs in LLVM register allocation.
  - fused_fp8_bmm_rope_cat_and_cache_mla, with its shape-specific files: small M
    gets a single tile, which the kernel fails to compile.

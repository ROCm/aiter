# gfx1250-strict configs: NOT TUNED

gfx1250-strict reads only this tree. Nothing here is tuned. The tables let the
kernels run; they say nothing about performance. Replace a family's tables with
measured values when it is tuned.

Every family that gfx1250 has, has a table here, made from the gfx1250 table:

- Tables with buckets keep only the generic entries: the "any" entry (GEMM,
  normalization and the other M-bucketed families), one "any" entry for each
  unified_attention section, and the `bm*_any` entries for Gluon MoE.
  fused_sigmoid_mul also keeps the "strided" entry, which its kernel needs, and
  the afp4wfp4 preshuffled tables keep the small-M entries, which their kernels
  need for M < 32.
- The Gluon quant tables keep one bucket key next to "any", with the "any"
  config, because the loader reads the axis order from the bucket keys.
- Triton MoE a8w4 is empty: the kernel uses its built-in defaults.
- Tables keyed by kernel variant (attention, KDA, grouped GEMM, block_cat_fused)
  are byte-identical copies, because the kernels ask for every key by name.

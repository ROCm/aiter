# gfx1250 Mega-MHC

`aiter.ops.mega_mhc.mega_mhc()` runs these kernels by default for the delayed
post/pre/RMSNorm seam with H=5120. Every other call keeps the HIP kernels of
`csrc/kernels/mega_mhc_gfx1250.cu`.

## Selection

A call runs an asm kernel only if all of these hold; otherwise it runs HIP.

- gfx1250 B0+ (`is_gfx1250_asm_supported()`), 256 CUs, no `HSA_CU_MASK` or
  `ROC_GLOBAL_CU_MASK`, and `AITER_MEGA_MHC_ASM` is not `0`.
- H=5120, `sinkhorn_repeat=20`, plain residual layout (`res_preshuffle=False`),
  a workspace prepared under the same conditions, and a code object in
  `mega_mhc.csv` for the call's tile, output and (post, pre) specialization.
- Token tile: 16 for T<=256, 32 for T<=512, 64 (persistent) above. T=1025..1280
  and, without post-mix, T>3072 keep HIP.

## Kernels

36 code objects: three tiles x three outputs x `HAS_POST` x `IDENTITY_PRE`,
named `mhc_{ns16|ns32|ps}_{bf16|mxfp8|mxfp8m32k4}_hp{0,1}_ip{0,1}_cs2`.
`hp1` applies the post-mix (`sublayer_out` given); `ip1` takes stream 0 as the
pre-mix (`pre_mix=None`). `mxfp8` writes row-major `[T, H/32]` E8M0 scales,
`mxfp8m32k4` the gfx1250 MXFP8 GEMM A-scale layout with rows padded to 32.
The CSV header describes its columns; `codegen.py -m mega_mhc` turns it into the
config table of `csrc/py_itfs_cu/asm_mega_mhc.cu`.

## Launch and ABI

- Block `(128,1,1)` (four wave32), cluster `(16,1,1)`: the 16 workgroups of a
  cluster each own 320 columns of H and meet through the partials.
- `ns16`/`ns32`: `ceil(T/tile)` clusters. `ps`: `min(16, ceil(T/64))` clusters;
  each round advances 16 tiles (1024 tokens), so T>1024 runs several rounds on 16.
- Static LDS and VGPRs are what each kernel uses (`manifest.json`): `ns16`
  295680 B / 912, `ns32` 306176 B / 880, `ps` 318976 B / 928 (BF16 912); launch
  with zero dynamic LDS; 128 SGPRs. One workgroup still fills a CU, and the
  remainder lets small kernels (one wave, a few VGPRs, one 2 KiB LDS granule)
  stay resident beside it.
- 192-byte arguments: the argument record of the gfx1250 HIP Mega-MHC kernel
  (`MegaMhcKernelArgs` in `asm_mega_mhc.cu`), with `split_arg=16`,
  `hidden_arg=5120` and `counters` unused.
- Workspace: FP32 partials `[16][T][32]` followed by `[16][T]` sum-of-squares
  records, sized by `prepare_mega_mhc`; no counters.

## Numerics

The collapse reads the updated residual rounded to BF16 and the RMS sum of
squares the unrounded FP32 values, so the norm and gates are not bitwise equal
to the HIP kernels. With post-mix the updated residual can differ from HIP by
one BF16 ulp (accumulation order). MXFP8 output quantizes the BF16 norm with the
same group32 E4M3fn codes and power-of-two E8M0 scales as HIP.

## Code objects

Each instruction entry is aligned to 16 KiB. The entry address is
`kernel_object + kernel_code_entry_byte_offset`; descriptor alignment alone does
not establish instruction entry alignment. The launch instruction prefetch,
`inst_pref_size` x 128 bytes from the entry, stays within the code.
`manifest.json` records each code object's symbol, entry offset, code size,
`inst_pref_size`, resource usage and SHA-256.

The ASM source and its offline build tooling are maintained separately and are
not distributed with AITER.

## Tests

```
python op_tests/test_mega_mhc.py --check-only --backends hip asm --out-dtype bf16 mxfp8 \
    --mode post no_post identity_pre no_post_identity [--scale-row-major]
python op_tests/test_mega_mhc.py --asm-suite
```

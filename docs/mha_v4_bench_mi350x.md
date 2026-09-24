# MHA v4 kernel benchmark on MI350X (gfx950)

Reproduces the forward FMHA throughput numbers for `mxfp8`, `mxfp6` and `f4f4`
at `h5 s65536`.

## What this branch contains

`mha_v4_bench_mi350x` is branched off aiter `4b6184fef` and changes only two
prebuilt kernel binaries:

| file | why |
|---|---|
| `hsa/gfx950/fmha_v4_fwd/fwd_hd128_f4f4.co` | the committed binary **memory-faults** on any `seqlen_k` that is not a multiple of 128 |
| `hsa/gfx950/fmha_v4_fwd/fwd_hd128_mxfp6.co` | rebuilt from the same kernel-source commit so all three kernels are consistent |

`fwd_hd128_mxfp8.co` is used **unchanged** — see "Known issues" for why it
cannot currently be rebuilt.

No Python, no manifest and no host code is modified. The dense manifest rows
for all three kernels are already present and distinct upstream, so no
rewiring is needed:

```
3,3,3,0,2,5,5,1,0,128,128,0,0,256,128,...mxfp8...,fwd_hd128_mxfp8.co
7,7,7,1,2,5,5,5,0,128,128,0,0,256,128,...mxfp6...,fwd_hd128_mxfp6.co
9,9,9,1,2,5,5,5,0,128,128,0,0,256,128,...f4f4...,fwd_hd128_f4f4.co
```

## Environment

Validated on:

| component | version |
|---|---|
| GPU | AMD Instinct MI355X (`gfx950`) |
| ROCm | 7.14.0 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+gitff65f5b (HIP 7.14.60850) |
| aiter | `4b6184fef` + this branch |

MI350X and MI355X are both `gfx950` and use the same binaries. Clocks differ,
so absolute TFLOPS will not match exactly between the two parts.

## Container setup

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size 16G \
  -v "$PWD":/workspace -w /workspace \
  rocm/pytorch:latest
```

Any ROCm 7.x PyTorch image with `hipcc` available works; aiter JIT-compiles its
host ops on first use, so the image must have the compiler, not just the runtime.

Inside the container:

```bash
git clone https://github.com/ROCm/aiter.git /app/external/aiter
cd /app/external/aiter
git checkout mha_v4_bench_mi350x
git submodule update --init --recursive

export PYTHONPATH=/app/external/aiter:$PYTHONPATH
export AITER_USE_SYSTEM_TRITON=1
```

`AITER_USE_SYSTEM_TRITON=1` avoids pulling aiter's pinned Triton, which is not
needed for these kernels.

The first benchmark invocation JIT-builds `module_mha_v4_quant` and
`module_fmha_v4_fwd` (roughly 10-30 s each). Subsequent runs reuse them.

## Running

One kernel:

```bash
HIP_VISIBLE_DEVICES=0 python3 op_tests/op_benchmarks/triton/bench_sage.py \
  --kernel mha4_f4f4 \
  --b 1 --hq 5 --hk 5 --sq 65536 --sk 65536 --d 128 --dv 128 \
  --input-distribution transformer --seed 0
```

All three:

```bash
for k in mha4_mxfp8 mha4_mxfp6 mha4_f4f4; do
  printf '%-12s ' "$k"
  HIP_VISIBLE_DEVICES=0 python3 op_tests/op_benchmarks/triton/bench_sage.py \
    --kernel "$k" --b 1 --hq 5 --hk 5 --sq 65536 --sk 65536 \
    --d 128 --dv 128 --input-distribution transformer --seed 0 \
    2>/dev/null | tail -1 | awk '{print $NF}'
done
```

The last line of output is the results table; the final column is TFLOPS.

Add `--compare-to-ref --ref torch` for a cosine check against a PyTorch
reference. Note that the bench only prints `Cosine Similarity` when
`--sq` equals `--sk`, and a full-precision reference at `s65536` will not fit
in memory, so run correctness checks at a smaller length.

## Reference numbers

`b1 hq5 hk5 sq65536 sk65536 d128 dv128`, transformer distribution, seed 0,
three consecutive runs on one MI355X:

| kernel | TFLOPS (3 runs) | median |
|---|---|---|
| `mha4_mxfp8` | 3029.8 / 3027.9 / 3028.5 | **3028.5** |
| `mha4_mxfp6` | 3803.3 / 3803.6 / 3819.5 | **3803.6** |
| `mha4_f4f4` | 4568.3 / 4589.0 / 4567.2 | **4568.3** |

Run-to-run spread is well under 1 %. Expect a few percent difference on MI350X.

## Known issues

**Stale JIT modules.** If aiter's Python sources are updated without clearing
the JIT cache, calls fail with a signature mismatch such as:

```
TypeError: rotate_activation_mxfp4_quant_k() missing 1 required positional argument: 'mean'
```

The compiled `.so` was built from an older signature. Clear and let it rebuild:

```bash
rm -f  aiter/jit/module_mha_v4_quant.so
rm -rf aiter/jit/build/module_mha_v4_quant
```

The same applies to `module_fmha_v4_fwd`, which additionally caches the dense
manifest — clear it after editing `fmha_v4_fwd.csv` or the change is ignored.

**`mxfp8` and `mxfp4` generators do not build.** Against current PyISA
(`e8e87b3`) both fail with:

```
TypeError: v_cvt_scalef32_pk_fp8_bf16() got an unexpected keyword argument 'op_sel'
```

The PyISA instruction signature changed and the generators have not been
updated. Their committed `.co` files still load and run, so the benchmark is
unaffected, but those two kernels cannot be rebuilt from source right now.

**`f4f4` on non-tile-aligned `seqlen_k`.** Fixed in the binary shipped here.
`s65536` is a multiple of 128 so the benchmark result is unaffected either way,
but the previously committed binary faults at, for example, `sk=32`.

## Rebuilding the kernels (optional)

Only needed to regenerate the `.co` files. Requires the kernel generator repo
and PyISA:

```bash
export PYTHONPATH=/app/external/aiter:/app/external/PyISA
ASM=/app/external/diffusion-models-inference-private/asm/fmha_sage_fwd

for k in f4f4 mxfp6; do
  AITER_DIR=/app/external/aiter $ASM/tools/build_deploy.sh \
    $ASM/gfx950/mi350_fmha_hd128_$k.py /tmp/$k.co \
    --slot hsa/gfx950/fmha_v4_fwd/fwd_hd128_$k.co
done
```

Kernel source for the binaries in this branch:
`diffusion-models-inference-private` at `f29276b`
(`mha_v4_paper`, merged from `main`), PyISA `e8e87b3`.

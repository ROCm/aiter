# gfx942 FP8 FMHA probability scaling

The adjacent patch records the ISA change to `hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_fp8.co`. It applies to the standalone assembly extracted from the original code object with SHA-256 `7bffc09e0fea2165b609e3661c896839bff3d0418e4511e99ef62bc2101c8b04`. Other FMHA code objects are unchanged.

The kernel rounds small softmax exponentials to zero when it converts them directly to FP8 FNUZ for the PV product, while retaining their mass in the FP32 denominator. Each of the six packing blocks now multiplies its 32 FP32 probabilities by 128 after denominator accumulation and before conversion. The final reciprocal is multiplied by 1/128 after the existing V descale and before broadcasting it to both packed output lanes. The denominator, online maximum updates, accumulator rescaling, masks, waits and LSE calculation are unchanged.

The patch adds 193 vector instructions without adding registers or LDS. Scaling expands the representable probability range; it does not eliminate FP8 rounding or guarantee exact normalization for every input. The original denominator remains the reference rather than the sum of rounded probabilities.

From the repository root, with the original code object saved as `original.co`:

```bash
tools=docs/examples/isa_optimization
bash "$tools/roundtrip.sh" original.co --mcpu gfx942 --keep /tmp/fmha-roundtrip
patch -o /tmp/fmha-scaled.s /tmp/fmha-roundtrip/kernel.s < "$tools/fmha_fp8_probability_scaling.patch"
/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 \
    -o /tmp/fmha-scaled.co /tmp/fmha-scaled.s
sha256sum /tmp/fmha-scaled.co
```

The measured build used ROCm 7.2.3 LLVM and produced SHA-256 `e5575cb14fe020289f31dd0653d213f792724b71a76a993219eae215fda435dc`. Toolchain differences may change the ELF container, so also compare the extracted instructions, kernel descriptor and metadata. The unmodified roundtrip matched the original `.text`, descriptor, metadata and target flags. Re-extraction of the patched object recovered exactly the intended instruction sequence and unchanged resource declarations.

Run `HIP_VISIBLE_DEVICES=0 ENABLE_CK=1 python -m pytest op_tests/test_mha_fp8_probability.py -q` on a free gfx942 GPU. The test covers uniform scores, long probability tails, a later maximum, masked sequence tails and nonconstant values through the public FP8 API. Confirm the `LoadKernel` log selects the intended file; the v3 and v4 objects share an FP8 symbol name. The generator for this object is not present in the public repository, so the same arithmetic should be applied there when the object is regenerated.

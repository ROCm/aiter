# Native-compatible mHC post body

`post.py` exposes `post_body(R, X, M, Out, B=128, SHUFFLED=False, FMA=2)` and standalone `post` with identical arguments. R is plain BF16[4,4096], X is the attention output BF16[4096], M is FP32[24] from preceding mHC: post=M[4:8], comb=M[8:24]. Caller must publish R/X/M before entering the body. No internal grid barrier. With grid256,32CTAs participate at B128; each reads R[4,128] and X[1,128] via TDM and writes all4residual streams. Other224CTAs do nothing. B256 uses16CTAs. Out is plain BF16[4,4096] or native KS64shuffled `(k//64)*256+h*64+k%64`.

The exact arithmetic is `r0*comb[0,h]`, then FP32FMA(x,post[h],acc), then FP32FMA(r1..r3,comb[1..3,h],acc), followed BF16 rounding. This matches the actual installed HIP `aiter.mhc_post` compiler reassociation, inspected in `native-post.amdgcn`. An opaque FP32 move before BF16 conversion is essential: otherwise Triton folds the final FMA+cast into `v_fma_mixlo_bf16`, removing intermediate FP32 rounding and causing10BF16differences in1,048,576tested values. FMA2 plus the move gives exact output. FMA0/1 remain diagnostic options only; default2 is accepted.

Validation:64plain randomcases exact.512additional randomcases each at B128/B256,16,777,216BF16values, bitwise native `aiter.mhc_post` plus `aiter.mhc_res_shuffle` agreement. Both input and output tensors have fixed allocations during final flatgraph microbench. Two rounds,128invocations pergraph,40alternating samples, report second median.

Final GPU3 times: native standalone plain1.3952us; native standalone+shuffle3.0565us; candidate B128plain1.5248us; B128shuffled1.5235us. These are endpoint ablations: live ATOM normally calls `mhc_fused_post_pre`, fusing this attention residual update with the next FFN input mHC. Do not identify standalone nativepost as a separate live server launch or add it atop live fusedpostpre timings.

Resources B128:27VGPR,1328bytesLDS,0spills. B256:30VGPR,2688bytesLDS,0spills. Source frozen after opaque FP32 move correction. `bench.py` reproduces validation and timings on one GPU; use HIP_VISIBLE_DEVICES=3. `validation.json`, `timings.json`, `resources.json`, candidate/native ISA and trace retained.

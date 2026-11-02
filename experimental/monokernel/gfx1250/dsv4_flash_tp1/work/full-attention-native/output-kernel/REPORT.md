# Attention output region: inverse RoPE through OB

Implemented `output_body` for inclusion inside the existing resident256CTA full-attention kernel. This adds inverseRoPE+FP8 quant, groupedOA, OA BF16 rounding andFP8quant, OB andBF16output. It does not implement mHCpost; caller owns it. Input-ready synchronization must precede body; Y-ready synchronization must follow before mHCpost. Body itself uses4 existing coherentL2 device-scope atomic-count/pure-load grid barriers.

| Step | CTAs | Work |
|---|---:|---|
| InverseRoPE+quant |256|128values each, per128E8M0scale, no intermediateBF16round afterRoPE|
| OA |256|64Ntiles128each ×4Ksplits1024each; grouped8 ×1024outputs ×4096K|
| OA reduce+quant |64|Fixed-order4split FP32sum, BF16round, per128FP8quant|
| OB |256|32Ntiles128each ×8Ksplits1024each;4096outputs ×8192K|
| OB reduce |32|Fixed-order8split FP32sum, BF16round|

Every learned GEMM uses WMMA BF16 operands from exact FP8+power-of-two scale upcast and FP32accumulation. Both tiled weights128×128 and activation1×128 use two-slot TDM double buffering. Fourwaves32, launch256CTAs. Smallscales, inverseRoPE vectors, and reductions use device-scope scalar/vector loads. Existing parent-prefix L2prefetch remains; this output body adds no new L2prefetch.

ABI: `output_body(O,Cos,Sin,OA,OAS,OB,OBS,XQ,XS,AP,AY,AQ,AS,BP,Y,Counter,Health,SA=4,SB=8)`.

- O BF16[64,512],Cos/Sin32values matching native table atposition1024.
- OA FP8[8192,4096],OAS uint8[64,32];OB FP8[4096,8192],OBSuint8[32,64], original unshuffled checkpoint layout.
- XQFP8[32768],XSuint8[256];APFP32[4,8192],AYBF16[8192];AQFP8[8192],ASuint8[64];BPFP32[8,4096],YBF16[4096].
- Counter/Healthint32 from existing resident-grid protocol.
- OptionalSA/SB allow no-split precision experiment; defaults remain4/8. Buffersallocatedformaxsplits workfornosplit.

GPU1 isolated outputregion, secondround medians, stableaddresses flat128copy graph,40samples alternatingorder,5warmups:

| Actual ATOM | Fused body | Custom five-stage control |
|---:|---:|---:|
|66.16608us|17.84034us|14.50581us|

Round1:66.15809/17.77791/14.50706us. Native uses actual installed `_attn_post`, groupedFP8OA andFP8OB dispatch; originalweightpacking is setup outside timing. NativeOB selects N1024tile/SPLITK1, which has only4CTAs at this shape. These timings are **not an approved replacement speedup** because validation below fails the existing0.001gate. Do not add isolated times to predict combined-kernel latency.

Standalone fused resources:246VGPR,37,648bytesLDS,0scratch. GeneratedISA/LLVM and resourcejson retained. Fused vs same-math five-stage control output is bitwise identical. Every barrier health check0.

Native numerical comparison:64 deterministic random BF16attention inputs, seeds991000..991063. OA relative RMS maximum0.000952843; OBmaximum0.004665904, fails0.001gate. Native inverseRoPE+quant produces exactly equal FP8bytes/E8M0scales on8 tested inputs. No-splitOA/OB experiment gives effectively identical maximum errors, so removing splitK does not fix the difference. Native FP8WMMA GEMM accumulation/rounding differs from this BF16-upcastWMMA implementation and smallOA differences amplify through subsequentFP8quantization. Precise causal instruction difference not yet established. Preserve this limitation in final report; no relaxed numerical gate used.

Files: `check.py` compiles and comparesfive-stage/fused; `native_check.py`64nativecases; `native_check_nosplit.py`ablation; `quant_check.py`8bytecomparisons; `bench.py`matched native/fused/control; `output_region.py`frozen implementation; rawlogs/results/ASM/LLVM. Nativefixture andAPI in sibling `output-native`; imports sharedsync/devicehelpers from `/work/mhc-optimized` and flatgraphhelper from `/work/atom-actual-microbench/attention`.

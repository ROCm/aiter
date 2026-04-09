# AITER v0.1.12 Release Notes

**Version:** v0.1.12
**Date:** 2026-04-09
**Base:** `48b0fb59` (334 commits since v0.1.11.post1)
**GPU Architectures:** gfx942 (MI300X), gfx950 (MI355X)
**Python:** 3.10, 3.12

---

## Highlights

1. **gfx1250 (MI450) Initial Support** — Enable gfx1250 architecture support across OPUS, kernels, and tests (#2450, #2449, #2394, #2599)
2. **OPUS API Migration** — Systematic migration from ck_tile to OPUS header-only library across activation, allreduce, type conversion, and MOE kernels (15 PRs)
3. **FlyDSL Integration** — Add FlyDSL as install dependency with a4w4 MoE support, split-k GEMM, a8w8 bpreshuffle (#2430, #2390, #2113, #2581, #2546)
4. **MLA Enhancements** — HipKittens nhead=128 kernel, persistent mode, nhead fold (64->32), LSE output, MTP=4 support, Qwen3.5 head_num=40 (18 PRs)
5. **ctypes C-ABI Migration** — Migrate MoE ASM, paged attention, topksoftmax to ctypes binding with error bridging to prevent worker crashes (#2255, #2341, #2395, #2498)
6. **CK Dependency Removal** — Remove CK dependency from FMHA fwd/bwd, HIP kernels; remove torch from csrc; ENABLE_CK build option for CK-free builds (#2353, #2250, #2545, #2501, #2074)
7. **New Model Tuned Configs** — Kimi K2.5/K2, GLM-5, Qwen3.5, Qwen3-next, DSv3-MXFP4, DeepSeek-R1 FP4, GPT-OSS-120B + 493 new FP4 GEMM shapes (#2619, #2518, #2324, #2616, #2092)
8. **Critical Overflow Fixes** — 7 fixes for 32-bit overflow in >4GB KV cache, FMHA backward, iGEMM, and MoE pointer offsets
9. **allreduce Prefill + Fusion** — Refactored allreduce for prefill case + allreduce+rmsnorm+quant 3-op fusion (#2453, #1990, #2586)
10. **Triton 3.6.0 Compatibility** — Fix MFMA instr_shape API changes, split-K correctness for M < BLOCK_SIZE_M (#2575, #2434)

---

## New Features (83)

### gfx1250 (MI450) Support
- Enable gfx1250 support (#2450)
- [OPUS] gfx1250 support for opus wmma scale and moe_sorting kernel (#2449)
- [OPUS] enable gfx1250 support for OPUS tests (#2394)
- Update HIP kernels to support different warp_size for topksoftmax/grouptopk, cache, sample (#2599)

### gfx1150/RDNA Support
- Adding gfx1150/51 to RDNA arch (#2014)
- [TRITON] Improve config selection for RDNA gpus (#2402)
- [TRITON] Improve RDNA config selection for FA (#2397)

### MLA (Multi-Latent Attention)
- Introduce HipKittens based nhead=128 MLA Kernel (#2039)
- Add gfx950 mla a8w8 qh32 kernel (#1912)
- Add LSE output support for MLA decode qseqlen=1 persistent kernel (#2440)
- Add LSE-aware kernel dispatch for MLA (#2378)
- MLA PS mode fp8 -kvd fp8 -n 128,4 support return lse (#2144)
- MLA PS mode add metadata split reference code (#2177)
- MI350 MLA PS mode fold nhead64,2 to nhead32,4 kernel (#2570)
- MI350 MLA PS mode support nhead=8 mtp=4 feature (#2461)
- Add decode_update_mla_metadata_v1 for quickly update mla_metadata in decoding (#2215)
- Upload mla_a8w8_qh64_qseqlen4_gqaratio16 co in MI300 (#2042)
- Add head_num=40 for mla fp8 reduce kernel for qwen3.5 (#2481)

### FMHA (Flash MHA)
- Add FP8 hdim=256 tile for batch prefill kernel (#2549)
- CK mha bwd: add sink attention score gradient support (#2321)
- [CK_TILE] FMHA BWD Use Persistent Kernels in Deterministic Mode (#2216)
- MHA fwd v3 hdim128 support per tensor fp8 for MI300/MI308 (#2105)
- Hipgraph support for fav3 kvcache (#2096)
- Add FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON env var support (#2000)
- Support per_block for Pa PS (#2053)

### Sage Attention
- [TRITON] Sage attention v2: Q*K in mxfp4 (#2066)
- [Triton] Add sliding window support for sink attention (#2505)
- [TRITON] Sagev2 patch (#2240)

### GEMM Kernels
- feat(ck_tile): add a8w8 blockscale gemm with preshuffleB support (#1954)
- [TRITON] Add support for splitk to A8W8 kernel (#2180)
- [FP8_BS_GEMM] add fp8 blockscale asm kernel (#2142)
- [MI308] add fp8 blockscale asm kernel (#2118)
- Add igemm kernel for mi325 (#1968)
- [MI325] Support gfx942 i8gemm tilesize 112x256 (#2006)
- Introduce asm 64x256 kernels for mi300 (#2404)
- 64x256 fmoe (#2279)
- Add 32x128 and 64x128 asm kernels to support qwen3-next TP4 (#2285)
- Add f32 MFMA support (#2070)
- Enable hd192_128 br kernel in python (#2009)
- Add support to dpsk-fp4 tp2/tp4(head=64/32) cases (#2031)

### MoE Kernels
- [TRITON] Add smoothquant int8 MoE kernel (#2049)
- Add moe_smooth_per_token_scaled_quant_v1&v2 (#2295)
- Group_topk: moe_fused_gate support num expert is not power of 2 (192/96) (#2604)
- Enable topk_softmax with share expert scoring function (#2356)
- Add asm topsoftmax support 384x8 (#2130)
- Support strided gating_score for topk_softmax (#2124)
- Add fused_dynamic_mxfp4_quant_moe_sort_hip (#2620)

### FlyDSL
- [FlyDSL] import flydsl implement of a4w4 moe (#2113)
- [FlyDSL] Add FlyDSL MoE a4w4 support and update kernels (#2390)
- [setup][flydsl] Add flydsl into aiter install requires (#2430)
- Add flydsl splitk gemm and update kimi-2 bf16 tunned config (#2536)
- So/a8w8 bpreshuffle flydsl (#2546)

### Fusion Kernels
- Add `allreduce+rmsnorm+quant` fusion pass (#1990)
- Fuse rms rope blk quant kernel (#2027)
- Add fused_qk_norm_group_quant kernel (#2527)
- Add fused_qknorm hip kernel (#2442)
- feat(custom_all_reduce): support GPT-OSS-120B hidden_size=2880 in fused allreduce rmsnorm (#2329)
- Adding double buffer option to cross_device_reduce_1stage (#2064)
- Add mhc_pre hip kernel (mhc_pre_gemm_sqrsum, mhc_pre_big_fuse) (#2136)
- mhc: add mhc_post hip kernel (#2479)

### RoPE / KV Cache
- [feat](rope): support shuffle value cache layout (#2568)
- Support naive mrope in get_rope (#2292)
- Enhance fused_qk_norm_rope_cache_quant functions by adding rotary_dim (#2199)
- feat(fused_kv_cache): support value_cache 5D shuffle layout and gpt-oss-120b precision tests (#2217)
- Feat gen kv prefix preshuffle1 (#2288)
- Add gather_kv_b_proj triton kernel (#1411)

### Other Features
- pa_decode_gluon_aot C++ api (#2085)
- feat: add fast gelu (#1220)
- [HIP] causal conv1d hip decode (#2084)
- Support FP8/MXFP4-quantized activation dtype (#2188)
- Add ENABLE_CK build option for CK-free builds (#2074)
- Identify device name by chip id (#2325)
- support_int64_ctypes (#2486)
- Support comments in csv (#2422)

### OPUS Framework
- [OPUS] Add gfx950 smem transpose load (#2480)
- [OPUS] Add finfo class for float-valued type properties (#2330)
- [OPUS] Enhance opus.hpp, add moe_sorting_opus, workgroup_barrier, and device tests (#2077)
- opus: tiled scaled MFMA + fix mfma_adaptor_swap_ab (#2384)

---

## Performance (53)

### Model-Specific Tuned Configs
- [Perf] Add Kimi-K2.5 tuned configs for MI355X (#2619)
- Retune kimi k2 moe configs (#2625)
- Add tuned csv files for Gemm and MoE to accelerate Kimi-K2 (#2290)
- [Perf] Add DSv3-MXFP4 tuned configs for MI355X (#2616)
- Add GLM-5 tuned configs (#2518)
- Add tuned config for qwen3.5 fp8, a8w8 blockscale gemm (#2324)
- Tuned qwen3.5 gemm (#2485)
- Gemm & moe tunning for DeepSeek-R1 in InferenceX FP4 case (#2261)
- Update dsv3 ptpc a8w8 gemm config (#2253)
- Add MI355X (gfx950) tuned GEMM configs for FP4 and FP8 shapes (#2037)
- Add new GEMM configuration files for various matrix sizes and parameters (#2024)
- tune: add 493 new FP4 GEMM shapes for LLM inference (#2092)
- Tune triton gemm kernel for MI355 DSV3 DP+EP configuration (#2016)
- Add more configs (#2506)
- Tune i8gemm in MI308 (#2590)
- [MI325][TUNE] igemm asm (#2125)
- Replace ck moe config in tp4 configs (#2626)
- Update flydsl bf16 gemm implementation and tuned config (#2634)

### Kernel Optimizations
- Top-K Top-P Sampling Kernel Optimization (#2034)
- Optimize the flash attention (#2265)
- Optimize prefill a4w4 moe (#2233)
- opt fuse_qk_norm_rope_blkquant kernel (#2206)
- perf: optimize _moe_mxfp4_sort_kernel to reduce Triton recompilation (#2414)
- [HIP] Optimized fused split GDR decode (#2326)
- [FlyDSL] Optimize a4w4 MOE kernels (#2581)
- [TRITON] fav3 sage optmization (#2045)
- Replace qseqlen fold with native qh64 kernel for nhead=64 decode on gfx950 (#2636)
- Fold qh128 to qh16 in gfx950 (#2204)
- Update config of MHA PE forward kernel on gfx950 (#2260)
- Add blockPerCu support for CKTile GEMMs and CKTile MOE tuning (#2313)
- enable_hipblaslt_fp8_tune (#2212)
- Use unreg path for custom all-reduce during CUDA graph capture (#2075)
- Edit batched_gemm_a8w8 and gemm_a16_w16 kernel args for no recompile (#2427)
- Reduce wasted get_module overhead for module with custom module name (#2455)
- Update decode_update_mla_metadata_v1 for atom dp attention (#2392)
- Defer expensive build operations to build_ext.run() (#1973)
- fix(gemm): add EVEN_MN heuristic to restore vectorized store in gemm (#2482)

---

## Bug Fixes (88)

### Overflow/OOB Fixes (Critical)
- [CK][FMHA] Fix 32-bit overflow in batch prefill kernel for >4GB KV cache (#2183)
- [FIX] address overflow fix on fmha_bwd of gfx942/gfx950 (#2189)
- [FIX] address overflow fix on gfx942 for hd128 fmha bwd case (#2151)
- Fix mha fwd_v3 _s_buff_Q/K/V/D address overflow (#1957)
- Fix moe ptr int32 offset overflow (#2196)
- [FIX] fix igemm 4GB oob bug (#2373)
- Fix OOB GU scales access in 64x128 kernels (#2328)
- Fix smoothquant hip kernel exceed int32's maximum (#2104)

### Use-After-Free Fixes
- Fix use after free issue in moe_sorting_opus_fwd (#2500)
- Fix use after free issue in moe_sorting_fwd (#2381)
- Fix use-after-free in cktile blockscale GEMM x_scale handling (#2358)

### Accuracy/Correctness Fixes
- Fix numerical accuracy in allreduce_fusion_kernel_1stage (#2586)
- Fix accuracy issues in top-p sampling kernels (#2035)
- Fix residual_out accuracy of hip rmsnorm fused add (#2011)
- Fix MHA bwd numeric issue (#2379)
- [MXFP4] Patch fp4_utils.py rounding logic (#2249)
- Fix resolve eightwarp functional failure in fp8 blockscale (#2207)
- Fix FlyDSL split-k HGEMM correctness and precision issues (#2567)
- [FIX] fix a8w8 asm kernel ks>1 mismatch (#2526)

### MLA/NaN Fixes
- Fix mla ps fp8 the kv_seq tail len < 4 nan error (#2106)
- Fix mla_a8w8_qh64_qseqlen4_gqaratio16_ps kv_len < 4 nan error (#2128)
- mla nps fp8 mode fix kv_tail_len < max_seqlen_q and fix nhead=128 reduce (#2319)

### MoE Fixes
- Fix a4w4 moe decode regression (#2218)
- Fixes around MoE kernel selection (#2152)
- Fix fmoe_int8_g1u0_a16 (#2322)
- Fix data overwrite problem in asm fmoe 1stage kernels for mi350 (#2507)
- interdim not divisible by 128 — force 1stage ASM kernels (#2193)

### FMHA/Attention Fixes
- [CK_TILE][FMHA] fix fmha_fwd_args alignment with CK struct update (#2259)
- Fix batch_prefill kernel dispatch failure for sliding window attention (#2170)
- [FIX] fix a16 causal mha bwd case for python api (#2029)
- fix(attention): restrict ASM paged attention to head_size=128 (#2273)
- [Triton] Fav3 sage attention mask fix (#2158)
- [TRITON] Sage v2 stride fix (#2117)
- [CK_TILE] Temporarily remove KV cache offset overflow assert checks (#2641)

### GEMM Fixes
- Fix CKTile blockscale GEMM to read strides from tensor metadata (#2466)
- Fix GEMM test failures and retune with latest triton (#2434)
- Fix LRU cache pollution causing BLOCK_SIZE_S3 KeyError in gemm_afp4wfp4 (#2169)
- fix(gemm_a8w8_bpreshuffle): pass splitK/KBatch to CK kernels (#2335)
- [TRITON] MXFP4 GEMM fixes (#2078)
- Fix splitk tmp_out undersized buffer avoid double-zeroing (#2551)

### Triton Compatibility Fixes
- Fix pa_mqa logits compile failure on triton 3.6.0 caused by MFMA instr_shape API (#2575)
- Fix triton3.5.1 vllm error in pa_mqa (#2108)
- [Triton] Fix triton tests fail due to API changes from Triton upstream (#2122)
- [TRITON] fix TILE_SIZE pow2 error if block_size is not pow2 (#2393)

### Other Fixes
- fix(car): write mode dispatch (#2607)
- fix(car): shfl and ag dispatch (#2346)
- fix(hip): launch FMHA Philox, sampling, and MM kernels on current stream (#2564)
- Fix correct duplicate knl_name in mla_asm.csv causing PP8 failure (#2030)
- fix STR_DTYPE_TO_TORCH_DTYPE import issue (#2593)
- Fix stride error check in fused_qk_norm_group_quant (#2637)
- Fix group topk dispatch for glm5 (#2611)
- Fix error checking in aiter_hip_common.h (#2225)
- Fix nondeterministic RNG in test_fused_mxfp4_quant (#2562)
- Fix crash on import if git is missing (#2226)
- Update topk.py to support non-power-of-2 experts (Kimi-K2) for long contexts (#2359)
- fix: split asm_topksoftmax into separate module to fix ctypes JIT build (#2603)
- Add ctypes C-ABI error bridging to prevent worker crashes (#2498)

---

## Refactoring (55)

### ck_tile -> OPUS API Migration
- Replace ck_tile api by opus in activation (#2589)
- Replace ck_tile api with opus api in some hip kernels (#2533)
- Replace ck_tile type convert by opus cast (#2331)
- Replace ck_tile by opus in allreduce (#2107)
- Fmha fwd remove ck dependency (#2353)
- Fmha bwd v3 remove ck dependency (#2250)

### OPUS Framework Enhancements
- [OPUS] enhance cast(), add numeric_limits, add missing test files (#2110)
- [OPUS] enhance opus UT by adding more tests (#2040)
- [OPUS] add opus UT (#2017)
- [OPUS] opus device test speed up (#2127)

### ctypes Migration
- refactor: use ctypes binding (#2255)
- Add MoE ASM ctypes migration (#2341)
- refactor(pa): use ctypes binding for pa_fwd and pa_ps_fwd (#2395)

### Torch Dependency Removal
- Remove torch dependency in MHA shared lib build (#2501)
- Refactor hip kernel -- remove torch from csrc (#2545)
- Exclude torch.h in a8w8 cu files (#2382)

### Code Refactoring
- Refactor RoPE Operators (#2534)
- Refactor allreduce for supporting prefill case (#2453)
- Refactor kernel bind way (#2377)
- Refactor topk softmax asm bind (#2327)
- Refactor hip kl (#2624)
- Assembly kernel cleanup work (#2439)
- Add WARP_SIZE define in aiter_hip_common.h && rm hip_compat.h (#2525)
- Remove unused keys (#2629)
- Remove asm mask type (#2026)
- del asm layernorm (#2571)

### FlyDSL Dependency Management
- Upgrade flydsl dependency to 0.1.2 (#2635)
- Update flydsl ver (#2618)
- rm flydsl find links (#2653)

---

## CI & Infrastructure (61)

### Test Sharding & Balancing
- CI: Split Aiter tests and triton into multiple shards (#1970)
- CI: Rebalance triton test shards with actual CI durations (#2281)
- CI: Build Triton wheel once and share across test shards (#2380)
- CI: auto-update split test FILE_TIMES (#2459, #2458, #2623)

### Monitoring & Analytics
- CI: add AMD CI job monitor workflow (#2550)
- CI: add runner label queue time analytics (#2606)
- CI: improve AMD CI monitor runner fleet summary (#2633)

### Runner & Environment
- CI: replace MI355 runner labels with MI35X (#2467)
- CI: Add opt-in MI355 Triton runner via ci:triton-355 label (#2347)
- CI: Test Aiter tests on rocm 7.2 (#2272)
- CI: use pip editable install and safe.directory in runtime CI (#2474)

### Release & Build
- CI: add docker username input for aiter release workflow (#2535)
- CI: add selectable Docker password secret for release workflow (#2528)
- CI: Upload wheel to S3 in CI test workflow (#2239)
- Fix CI prebuild: use build_ext so kernels are actually compiled (#2100)

### Downstream Integration Tests
- CI: Replace simple inference with accuracy tests in ATOM test workflow (#2266)
- CI: Add steps to monitor system health before ATOM tests (#2097)
- CI: Fix SGlang dependencies issues (#2007)
- CI: Update vllm_benchmark.yaml to use latest nightly image (#2165)

### Flash Attention CI
- [CI] Flash Attention Integration CI (#1974)
- [CI] Flash Attention use submodules (#2208)

### Documentation
- [Docs] Add Sphinx documentation website with GitHub Actions deployment (#2167)

---

## Known Issues

- **Issue #2656**: DeepSeek-R1-MXFP4 accuracy regression from Triton GEMM config retune (PR #2434). Partial fix applied on release branch (reverted `gfx950-GEMM-AFP4WFP4-N=7168-K=2304.json`). Full bisect of other config changes pending.

## Compatibility

- **GPU Architectures:** gfx942 (MI300X/MI308/MI325), gfx950 (MI355X/MI350)
- **Python:** 3.10, 3.12
- **ROCm:** 7.0+
- **Triton:** 3.6.0
- **FlyDSL:** 0.1.2

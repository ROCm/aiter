# AITER v0.1.12 Release Notes

**Release Date:** 2026-04-10  
**Previous Release:** v0.1.11.post1 (2026-03-05)  
**Commits:** 334 (excluding release branch maintenance)  
**Supported GPU Architectures:** gfx942 (MI300X/MI325X), gfx950 (MI355X)

## Highlights

**OPUS Migration -- Replacing CK Tile Primitives.**
A major effort migrated internal kernel code from CK Tile APIs to the new OPUS (Operator Utility for Shader) abstraction layer. This includes replacing CK Tile in activation kernels (#2589), HIP kernels (#2533), allreduce (#2107), and type conversion primitives (#2331). OPUS also gained tiled scaled MFMA (#2384), finfo class (#2330), cast/numeric_limits enhancements (#2110), moe_sorting_opus (#2077), gfx950 smem transpose load (#2480), and comprehensive unit tests (#2017, #2040, #2127). This migration decouples AITER from CK internals and establishes OPUS as the portable device-code foundation.

**FlyDSL Integration for MoE Kernels.**
FlyDSL, AMD's high-performance domain-specific language, is now a first-class AITER dependency. Initial A4W4 MoE kernel support was imported (#2113) and enhanced (#2390), split-k GEMM was added (#2536), A4W4 MoE kernels were optimized (#2581), correctness and precision issues in split-k HGEMM were fixed (#2567), and FlyDSL was added to install requirements (#2430). The dependency was upgraded to v0.1.2 (#2635).

**MLA (Multi-head Latent Attention) Enhancements.**
MLA received extensive feature additions: HipKittens-based nhead=128 kernel (#2039), gfx950 A8W8 qh32 kernel (#1912), MLA persistent kernel LSE output (#2440), LSE-aware dispatch (#2378), FP8 return-LSE support (#2144), metadata split reference code (#2177), fast metadata update for decoding (#2215), and MI350-specific PS mode improvements including nhead=8/mtp=4 (#2461) and nhead64-to-nhead32 folding (#2570). Multiple NaN and accuracy bugs were also fixed (#2106, #2128, #2319).

**ctypes Kernel Binding Refactoring.**
Kernel dispatch was systematically migrated from pybind11 to ctypes, reducing build complexity and improving JIT build reliability. This includes the foundational ctypes binding refactor (#2255), paged attention ctypes migration (#2395), MoE ASM ctypes migration (#2341), int64 ctypes support (#2486), and a fix for ctypes JIT build issues with asm_topksoftmax (#2603).

**CK Dependency Removal for FMHA.**
Flash MHA forward (#2353) and backward v3 (#2250) kernels had their Composable Kernel dependencies removed, and a build-time `ENABLE_CK` option was added (#2074) enabling fully CK-free builds. The torch dependency was also removed from the MHA shared library build (#2501). These changes reduce build times and external dependency complexity.

**Warp Size Generalization.**
HIP kernels were updated to support variable warp sizes rather than hardcoding warp_size=64. A `WARP_SIZE` macro was added to the common header for both host and device use (#2525), and topksoftmax, grouptopk, cache, and sample kernels were updated (#2599). This is essential for cross-architecture portability between CDNA (warp_size=64) and future targets.

**Allreduce Refactoring and Fusion.**
The custom allreduce path was refactored to support prefill-phase collective operations (#2453), an `allreduce+rmsnorm+quant` fusion pass was added (#1990), GPT-OSS-120B hidden_size=2880 support was enabled in fused allreduce rmsnorm (#2329), numerical accuracy was improved (#2586), a double-buffer option was added for cross_device_reduce_1stage (#2064), and CUDA graph capture compatibility was fixed (#2075).

**Sage Attention v2 and Flash Attention Improvements.**
Triton-based Sage Attention v2 received multiple updates: MXFP4 Q*K support (#2066), optimizations (#2045), stride fixes (#2117), mask fix (#2158), and a consolidated patch (#2240). Flash Attention v3 gained hipgraph support for KV cache (#2096), configurable Triton configs via environment variable (#2000), Windows build support (#2433), and integration CI (#1974).

**RDNA Architecture Support.**
AITER expanded beyond data center GPUs with gfx1150/1151 RDNA arch registration (#2014), improved RDNA config selection for Flash Attention (#2397) and general kernels (#2402), and RDNA CI infrastructure (#2222).

## New Features

### Attention & MLA
- Introduce HipKittens-based nhead=128 MLA kernel (#2039)
- Add gfx950 MLA A8W8 qh32 kernel (#1912)
- Add LSE output support for MLA decode qseqlen=1 persistent kernel (#2440)
- Add LSE-aware kernel dispatch for MLA (#2378)
- MLA PS mode FP8 with return LSE for nhead=128,4 (#2144)
- MLA PS mode add metadata split reference code (#2177)
- Add decode_update_mla_metadata_v1 for fast metadata update in decoding (#2215)
- MI350 MLA PS mode support nhead=8, mtp=4 (#2461)
- MI350 MLA PS mode fold nhead64,2 to nhead32,4 kernel (#2570)
- Add head_num=40 for MLA FP8 reduce kernel for Qwen3.5 (#2481)
- Upload mla_a8w8_qh64_qseqlen4_gqaratio16 config for MI300 (#2042)
- Add FP8 hdim=256 tile for batch prefill kernel (#2549)
- Support per_block for PA PS kernels (#2053)
- Add sliding window support for Triton sink attention (#2505)
- CK MHA backward: add sink attention score gradient support (#2321)
- MHA forward v3 hdim128 support per-tensor FP8 for MI300/MI308 (#2105)
- CK Tile FMHA backward use persistent kernels in deterministic mode (#2216)
- Optimize flash attention forward (#2265)
- Sage Attention v2: Q*K in MXFP4 (#2066)
- Sage Attention v2 patch (#2240)
- Hipgraph support for fav3 KV cache (#2096)
- Add FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON env var support (#2000)
- Flash Attention Triton Windows build support (#2433)

### MoE Kernels
- FlyDSL A4W4 MoE kernel import (#2113)
- FlyDSL A4W4 MoE support and kernel update (#2390)
- FlyDSL A4W4 MoE kernel optimization (#2581)
- Add FlyDSL split-k GEMM (#2536)
- Triton smoothquant int8 MoE kernel (#2049)
- Introduce ASM 64x256 kernels for MI300 (#2404)
- Introduce 64x256 fmoe kernels (#2279)
- Support topk_softmax with shared expert scoring function (#2356)
- Group_topk: moe_fused_gate support non-power-of-2 experts (192/96) (#2604)
- Update topk.py to support non-power-of-2 experts (Kimi-K2) for long contexts (#2359)
- Add moe_smooth_per_token_scaled_quant v1 and v2 (#2295)
- Add ASM topsoftmax support 384x8 (#2130)
- Support strided gating_score for topk_softmax (#2124)

### GEMM
- Add FP8 blockscale ASM kernel (#2142)
- CK Tile A8W8 blockscale GEMM with preshuffleB support (#1954)
- Add Triton A8W8 split-k support (#2180)
- Add compiler configurations for bpreshuffle CK Tile modules (#2069)
- Add 32x128 and 64x128 ASM kernels for Qwen3-next TP4 (#2285)
- Add precision fix and gelu kernels for 64x256 (#2471)
- MI325 support gfx942 i8gemm tilesize 112x256 (#2006)
- Add igemm kernel for MI325 (#1968)
- Enable hipblaslt FP8 tuning (#2212)
- Add f32 MFMA support for 32x32x2f32 and 16x16x4f32 (#2070)
- Add fast gelu activation (#2220)

### Fused Kernels
- Add fused_qk_norm_group_quant kernel (#2527)
- Add fused_qknorm HIP kernel (#2442)
- Fuse RMS + RoPE + block quantization kernel (#2027)
- Optimize fused_qk_norm_rope_blkquant kernel (#2206)
- Add `allreduce+rmsnorm+quant` fusion pass (#1990)
- Support GPT-OSS-120B hidden_size=2880 in fused allreduce rmsnorm (#2329)
- Add mhc_post HIP kernel (#2479)
- Add mhc_pre HIP kernel (mhc_pre_gemm_sqrsum, mhc_pre_big_fuse) (#2136)
- Add fused_qk_norm_rope_cache_quant rotary_dim parameter (#2199)
- Top-K Top-P sampling kernel optimization (#2034)

### RDNA Support
- Adding gfx1150/51 to RDNA arch (#2014)
- Improve RDNA config selection for Flash Attention (#2397)
- Improve config selection for RDNA GPUs (#2402)

### Other Features
- HIP causal conv1d decode kernel (#2084)
- PA decode gluon AOT C++ API (#2085)
- Support naive mrope in get_rope (#2292)
- Support FP8/MXFP4-quantized activation dtype (#2188)
- Support value_cache 5D shuffle layout with GPT-OSS-120B precision tests (#2217)
- Generate KV prefix preshuffle (#2288)
- Support dim(-1) allgather (#2162)
- Add ep, pp, dp group interface (#2137)
- Respect AITER_LOG_LEVEL for C++ stdout prints (#2086)
- Identify device name by chip ID (#2325)
- Support comments in tuned config CSV files (#2422)
- Defer expensive build operations to build_ext.run() (#1973)
- Hipgraph support: correct arg_size type from int to size_t (#2163)
- Add double-buffer option for cross_device_reduce_1stage (#2064)
- Use unreg path for custom all-reduce during CUDA graph capture (#2075)

## Performance

### Tuned Configs
- Add Kimi-K2.5 tuned configs for MI355X (#2619)
- Add DSv3-MXFP4 tuned configs for MI355X (#2616)
- Retune Kimi K2 MoE configs (#2625)
- Replace CK MoE config in TP4 configs (#2626)
- Add GLM-5 tuned configs (#2518)
- Add Qwen3.5 FP8 and A8W8 blockscale GEMM tuned configs (#2324)
- Tuned Qwen3.5 GEMM (#2485)
- Add tuned CSV files for GEMM and MoE to accelerate Kimi-K2 (#2290)
- Add MI355X (gfx950) tuned GEMM configs for FP4 and FP8 shapes (#2037)
- Tune 493 new FP4 GEMM shapes for LLM inference (#2092)
- Add new GEMM configuration files for various matrix sizes (#2024)
- GEMM and MoE tuning for DeepSeek-R1 InferenceX FP4 (#2261)
- Tune Triton GEMM kernel for MI355 DSV3 DP+EP configuration (#2016)
- MI325 igemm ASM tuning (#2125)
- Add blockPerCu support for CK Tile GEMMs and CK Tile MoE tuning (#2313)
- Update dsv3 ptpc A8W8 GEMM config (#2253)
- Add GEMM-A16W16-ATOMIC-N=256-K=6144 Triton GEMM tune config (#2213)
- Update gfx950 PA PS kernels and wire stride_scale_blk in asm_pa (#2569)
- Update gfx942 PA PS kernels and wire stride_scale_blk in asm_pa (#2522)
- Add more MoE/GEMM configs (#2506)
- Fix MoE stage2 tune config (#2438)
- Fix MoE GEMM tuned config (#2463)
- Remove duplicate tuned configs (#2219)
- Add FlyDSL split-k GEMM with Kimi-2 BF16 tuned config (#2536)
- Fix GEMM test failures and retune with latest Triton (#2434)

### Kernel Optimizations
- Optimize prefill A4W4 MoE (#2233)
- Optimized fused split GDR decode (#2326)
- Optimize _moe_mxfp4_sort_kernel to reduce Triton recompilation (#2414)
- Triton fav3 sage optimization (#2045)
- Fold qh128 to qh16 in gfx950 (#2204)
- Support dpsk-fp4 TP2/TP4 (head=64/32) cases (#2031)
- Enable hd192_128 BR kernel in Python (#2009)
- Optimize moe_smooth_per_token_scaled_quant dispatch; v2 supports block_m multiple of 16 (#2333)
- Add EVEN_MN heuristic to restore vectorized store in GEMM (#2482)
- Reduce wasted get_module overhead for modules with custom names (#2455)
- Update config of MHA PE forward kernel on gfx950 (#2260)
- Update decode_update_mla_metadata_v1 for ATOM DP attention (#2392)

## Bug Fixes

### Overflow / Out-of-Bounds (Critical)
- Fix 32-bit overflow in batch prefill kernel for >4GB KV cache (#2183)
- Fix overflow in FMHA backward on gfx942/gfx950 (#2189)
- Fix hd128 FMHA backward overflow on gfx942 (#2151)
- Fix FMHA forward buffer address overflow (#1957)
- Fix MoE pointer int32 offset overflow (#2196)
- Fix igemm 4GB OOB bug (#2373)
- Fix OOB GU scales access in 64x128 kernels (#2328)
- Fix smoothquant HIP kernel exceeding int32 maximum (#2104)

### Use-After-Free
- Fix use-after-free in moe_sorting_opus_fwd (#2500)
- Fix use-after-free in moe_sorting_fwd (#2381)
- Fix use-after-free in CK Tile blockscale GEMM x_scale handling (#2358)

### Attention / MLA Fixes
- Fix MLA PS FP8 NaN error when kv_seq tail len < 4 (#2106)
- Fix mla_a8w8_qh64_qseqlen4_gqaratio16_ps NaN error when kv_len < 4 (#2128)
- Fix MLA NPS FP8 mode: avoid kv_tail_len < max_seqlen_q and fix nhead=128 reduce (#2319)
- Fix duplicate knl_name in mla_asm.csv causing PP8 failure (#2030)
- Fix MTP mock (#2164)
- Fix batch prefill kernel dispatch failure for sliding window attention (#2170)
- Fix MHA backward numeric issue (#2379)
- Fix a16 causal MHA backward for Python API (#2029)
- Restrict ASM paged attention to head_size=128 (#2273)
- Fix FMHA forward args alignment with CK struct update (#2259)
- Temporarily remove KV cache offset overflow assert checks in batch prefill (#2641)
- Fix Triton3.5.1 vLLM error in pa_mqa (#2108)
- Fix pa_mqa logits CDNA version (#2323)
- Fav3 sage attention mask fix (#2158)
- Sage v2 stride fix (#2117)
- Remove FMHA backward assert and compat CK API change (#1966)

### MoE Fixes
- Fix data overwrite in ASM fmoe 1stage kernels for MI350 (#2507)
- Fix A4W4 MoE decode regression (#2218)
- Fix fmoe_int8_g1u0_a16 (#2322)
- Fixes around MoE kernel selection (#2152)
- Fix group topk dispatch for GLM5 (#2611)
- Update HIP MoE smoothquant to support expert_num <= 1024 (#2231)
- Fix A8W8 ASM kernel ks>1 mismatch (#2526)
- Interdim not divisible by 128: force 1stage ASM kernels (#2193)

### GEMM Fixes
- Fix LRU cache pollution causing BLOCK_SIZE_S3 KeyError in gemm_afp4wfp4 (#2169)
- Fix CK Tile blockscale GEMM to read strides from tensor metadata (#2466)
- Fix gemm_a8w8_bpreshuffle: pass splitK/KBatch to CK kernels (#2335)
- Fix splitk tmp_out undersized buffer, avoid double-zeroing (#2551)
- Fix resolve eightwarp functional failure in FP8 blockscale (#2207)
- Fix FlyDSL split-k HGEMM correctness and precision issues (#2567)
- Triton MXFP4 GEMM fixes (#2078)
- Revert Triton GEMM kernel config due to core dump (#2065)

### Sampling & Quantization Fixes
- Fix accuracy issues in top-p sampling kernels (#2035)
- Fix nondeterministic RNG in test_fused_mxfp4_quant (#2562)
- Patch fp4_utils.py rounding logic (#2249)
- Copy config before mutate (#2173)

### Triton Compatibility
- Fix pa_mqa_logits compile failure on Triton 3.6.0 caused by MFMA instr_shape API change (#2575)
- Fix TILE_SIZE pow2 error if block_size is not pow2 (#2393)
- Fix Triton tests due to API changes from Triton upstream (#2122)
- Default enabling of TRITON_HIP_USE_ASYNC_COPY caused runtime errors (#1932)

### Other Fixes
- Fix numerical accuracy in allreduce_fusion_kernel_1stage (#2586)
- Fix CAR write mode dispatch (#2607)
- Fix CAR shfl and ag dispatch (#2346)
- Fix FMHA Philox, sampling, and MM kernels launch on current stream (#2564)
- Fix error checking in aiter_hip_common.h (#2225)
- Fix residual_out accuracy of HIP rmsnorm fused add (#2011)
- Fix crash on import if git is missing (#2226)
- Fix racing problem when read/write merged file at same time by atomic write (#1593)
- Fix STR_DTYPE_TO_TORCH_DTYPE import issue (#2593)
- Fix flatten get_block_m() to avoid unconditional early return
- Fix block_m heuristic placement before 1stage/2stage branch, fix fake tensor shape
- Fix mhc build (#2168)

## Architecture / Refactoring

### OPUS Migration
- Replace CK Tile API with OPUS in activation kernels (#2589)
- Replace CK Tile API with OPUS in HIP kernels (#2533)
- Replace CK Tile by OPUS in allreduce (#2107)
- Replace CK Tile type convert with OPUS cast (#2331)
- OPUS tiled scaled MFMA + fix mfma_adaptor_swap_ab (#2384)
- Add OPUS finfo class for float-valued type properties (#2330)
- Add OPUS gfx950 smem transpose load (#2480)
- Enhance OPUS cast(), add numeric_limits, add missing test files (#2110)
- Enhance opus.hpp, add moe_sorting_opus, workgroup_barrier, and device tests (#2077)

### CK Dependency Removal
- FMHA forward: remove CK dependency (#2353)
- FMHA backward v3: remove CK dependency (#2250)
- Add ENABLE_CK build option for CK-free builds (#2074)
- Remove torch dependency in MHA shared lib build (#2501)

### ctypes Refactoring
- Foundational ctypes binding refactor (#2255)
- Paged attention ctypes binding for pa_fwd and pa_ps_fwd (#2395)
- MoE ASM ctypes migration (#2341)
- Support int64 ctypes (#2486)
- Split asm_topksoftmax into separate module to fix ctypes JIT build (#2603)

### Warp Size Generalization
- Add WARP_SIZE define in aiter_hip_common.h for host and device; remove hip_compat.h (#2525)
- Update topksoftmax, grouptopk, cache, and sample kernels for different warp_size (#2599)

### Allreduce Refactoring
- Refactor allreduce for supporting prefill case (#2453)

### Other Refactoring
- Refactor RoPE operators (#2534)
- Refactor topk softmax ASM bind (#2327)
- Refactor kernel bind way (#2377)
- Refactor HIP kernel: remove torch from csrc (#2545)
- Exclude torch.h in A8W8 CU files (#2382)
- Remove ASM layernorm (#2571)
- Remove gemm_common bind (#2425)
- Assembly kernel cleanup (#2439)
- Remove ASM mask type (#2026)
- Remove unused keys (#2629)
- Edit batched_gemm_a8w8 and gemm_a16_w16 kernel args for no recompile (#2427)
- Silence certain warnings stemming from CK (#2055)
- Discard check_LLVM_MAIN_REVISION object file (#2345)
- Use regex to extract arch from rocminfo string (#2082)
- Set logging propagate=False to prevent duplicate log output (#2436)
- Add hipblaslt error log (#2252)
- Fix tensor_address_log (#2523)
- mdf_asm_kl_bind (#2412)
- mdf_setup_py (#2195)
- Update CK submodule (#2462)

## CI / Infrastructure

### Test Sharding & Performance
- Split Aiter tests and Triton into multiple shards (#1970)
- Build Triton wheel once and share across test shards (#2380)
- Rebalance Triton test shards with actual CI durations (#2281)
- Auto-update split test FILE_TIMES (#2459, #2458, #2623)
- Look up FILE_TIMES by full path so shard estimates use real times (#2146)

### CI Monitoring
- Add AMD CI job monitor workflow (#2550)
- Improve AMD CI monitor runner fleet summary (#2633)
- Add runner label queue time analytics (#2606)
- Move monitor scripts under .github/scripts (#2572)

### Workflow Improvements
- Skip CI for draft PR or docs changes (#2103)
- Make extended tests label-triggered on PRs (#2192)
- Add opt-in MI355 Triton runner via ci:triton-355 label (#2347)
- Replace MI355 runner labels with MI35X (#2467)
- Test Aiter tests on ROCm 7.2 (#2272)
- Replace simple inference with accuracy tests in ATOM test workflow (#2266)
- Fix OOM issues in ATOM tests (#2123)
- Increase timeout to 60 minutes in ATOM tests (#2109)
- Add timeout-minutes to test steps (#2237)
- Add 60-min timeout to vLLM benchmark job (#2230)
- Update vllm_benchmark.yaml to use latest nightly image (#2165)
- Pin Triton to known-good commit to fix Shard 4/6 failures (#2186)
- Fix Triton commit to c147f098 (#2083)
- Use pip editable install and safe.directory in runtime CI (#2474)
- Fix dubious ownership for sglang checkout (#2477)
- Fix sglang test failures for non-standard fork names (#2145)
- Fix SGLang dependency issues (#2007)
- Fix multi-GPU tests only running one shard (#2243)
- Skip test_fused_ar_rms.py in multi-GPU tests (#2280)
- Add CK submodule sync step for non-fork PRs (#2310)
- Fix CI prebuild: use build_ext so kernels are actually compiled (#2100)
- Upload wheel to S3 in CI test workflow (#2239)
- Increase standard test timeout and MAX_JOBS for fork PR builds (#2289)
- Pin linter versions in CONTRIBUTE.md and update pre-commit hook (#2448)
- Revert pre-checks Ruff step to baseline behavior (#2574)
- Use pinned CK on main branch and add nightly schedule for latest CK (#2291)
- Fix build CK pipeline (#2399)
- Add steps to monitor system health before ATOM tests (#2097)
- Add pull-requests write permission for welcome comment (#2202)
- Fix pull_request_target for PR welcome comment on forked PRs (#2444)
- Update PR welcome comment (#2342)
- Add runner-config.yml for runner-to-GPU mapping (#2126)

### Release Infrastructure
- Add auto-release workflow, smoke test, and release process documentation
- Add docker username input for aiter release workflow (#2535)
- Add selectable Docker password secret for release workflow (#2528)
- Update runner name in Aiter release package pipeline (#2532)

### Flash Attention CI
- Flash Attention integration CI (#1974)
- Flash Attention use submodules (#2208)

### RDNA CI
- Add RDNA CI (#2222)

## Documentation & Testing

### Documentation
- Add Sphinx documentation website with GitHub Actions deployment (#2167)

### Benchmarking Tools
- Add model benchmarking tool (bench_models.py) (#2050)
- Add attention support to bench_models benchmarking script (#2274)
- Add kernel filter to bench_models.py (#2490)
- Create script to benchmark attention kernels with LLM model shapes (#2111)
- Triton GEMM tuning script (#1833)
- Fix benchmark scripts to generate output CSVs (#2555)
- Fix bench_mha (#2317)
- Fix wrong path to tune script (#2023)
- Avoid AttributeError in bench_moe_align_block_size.py (#2557)

### Test Improvements
- Improve batch prefill test coverage (#2302)
- Enable FP8 backward test (#2276)
- Fix Triton unit tests on gfx950 -- part 1 (#2403) and part 2 (#2491)
- Fix tests not supported in MI355 (#2553)
- Skip test_metadata_redirect.py on archs other than gfx942 (#2456)
- Reduce RoPE tests (#2588)
- Reduce GEMM unit tests (#2584)
- Triton unified attention FP8 cleanup (#2360)
- OPUS device test speed up (#2127)
- OPUS unit tests (#2017, #2040)
- Assert when found duplicated tuned shape (#2503)
- Refine split fused GDR decode test (#2446)
- Fix multigpu test issues (#1995)
- Various unit test updates (#2015, #2025, #2114, #2134, #2147, #2156)

## Compatibility

- **GPU Architectures:** gfx942 (MI300X/MI308/MI325X), gfx950 (MI355X/MI350)
- **Experimental:** gfx1150/1151 (RDNA)
- **ROCm:** 7.0+
- **FlyDSL:** 0.1.2

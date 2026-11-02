# Flash concurrency64: attention post / FFN mHC pre

GPU1, M64 actual batched shapes, hidden4096, four residual streams. Layer0 checkpoint **FFN** hc_ffn_fn/hc_ffn_scale/hc_ffn_base/ffn_norm.weight; these differ from earlier attention-pre weights. R[64,4,4096]BF16 in native shuffled layout, X[64,4096]BF16, post[64,4,1]FP32,comb[64,4,4]FP32. Context1k/1k does not change this region's shapes. Inputs are deterministic synthetic BF16 tensors with checkpoint weights, not a serving activation capture.

| Boundary | Actual native µs | Existing hybrid µs | Launches native/hybrid |
|---|---:|---:|---|
| Attention post + FFN mHC pre + RMSNorm |5.89666|Same native kernels when quant disabled|2/2|
| Above + shared-expert FP8 input quant |8.98222|5.96786|3/2|
| FFN mHC pre + RMSNorm only, starting after attention post |8.70838|No existing M64 standalone candidate|3/—|

Secondround, stable-address128-copy flat HIPgraphs,5warmups,40alternated samples and median GPUeventtime. Round1:5.87897 /8.95499 /5.93202 /8.67238us in nativepostpre, nativepostprequant, hybridpostprequant, preonly order. Rawrounds/samples retained.

Native serving fuses preceding attentionpost into FFNpre already. The earlier full-attention monokernel ended after attentionpost, so comparing the first row requires moving that boundary back; do not charge post twice. Third row is a separate boundary diagnostic, not an additional sequential operation. Concurrency1 monokernel candidate has no batchedM64 entry; it was not looped64times or substituted here.

The existing hybrid is `/work/flash/candidate/dsv4_hybrid.py`, which calls installed `mhc_post_pre_quant_gfx1250` and explicitly supports1..128rows. It performs actual batched work atM64. Parent liveC64trace confirms native2-kernel postpre with FUSE_QUANT0 and later shared-expert dynamic_per_group_scaled_quant; combining those is a legitimate operator grouping. Shared/routedexpert work can overlap in serving, so the isolated quant-fusion gain is not an end-to-end speedup claim. Router and routedMXFP4 paths do not consume this extraFP8output.

Validation:16 changing random M64batches; all six outputs—post,comb,normalizedX,nextresidual,FP8data,E8M0scales—bitwise identical between nativepostpre+separatequant and hybrid. No tolerance relaxation.

Native / hybrid common projection:
`_mhc_post_pre_gemm_sqrsum_gfx1250_kernel_C_4096_BLOCK_M_16_K_LOOP_1_SPLIT_K_64_NUM_STAGES_2_RES_SHUFFLED_1_W_PRESHUFFLED_1`
Grid[4,64,1]=256CTAs,128threads.150VGPR,119488bytesLDS,0scratch.

Native reduction:
`_mhc_pre_reduce_apply_gfx1250_kernel_C_4096_ROWS_1_CHUNK_1024_N_CHUNKS_4_S_64_FUSE_RMSNORM_1_RES_SHUFFLED_1_FUSE_QUANT_0`
Grid[64,1,1],256threads.90VGPR,32868bytesLDS,0scratch.

Hybrid reduction:
`_mhc_pre_reduce_apply_gfx1250_kernel_C_4096_ROWS_1_CHUNK_1024_N_CHUNKS_4_S_64_FUSE_RMSNORM_1_RES_SHUFFLED_1_FUSE_QUANT_1`
Same grid/block/resources; emits sharedinputquant directly.

Native separatequant:
`_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`
Grid[128,1,1],64threads.

The pre-only boundary calls native `aiter.mhc_pre` with residual_preshuffle=True: unshuffle kernel, HIPpreGEMM, HIPpre_big_fuse_rmsnorm. Exact full signatures and grids retained in actual_ffn_pre_norm_only-kernels.json. This is an ablated native pre-only API, not the default serving fusedpostpre dispatch.

`bench.py`, fixture.pt, validation.json,timings.json, traces/kerneljson, `resources.py`,resources.json, generatedGlunISA retained. Run container HIP_VISIBLE_DEVICES=1 python /work/ffn-c64/mhc/bench.py; benchmark overwrites resultfiles. Dependencies are installedAITERN+ATOM, existingflashcandidate, and stablegraph helper in atom-actual-microbench/attention/graphbatch.py.

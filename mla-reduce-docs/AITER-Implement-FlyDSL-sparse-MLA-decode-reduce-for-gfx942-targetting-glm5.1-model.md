MLA decode reduce / combine — gfx942 vs gfx950

Source document: docs/issues/gfx942-vs-gfx950-kernels/01-mla-decode-reduce.md

What this family does

Flash-decoding / split-KV MLA splits the KV sequence across many work tiles so a short decode batch can saturate all CUs. Each stage-1 kernel produces, per split, a partial attention output O_i (fp32) and a log-sum-exp LSE_i. The reduce/combine kernel is the stage-2 epilogue: it merges partials across splits using an LSE-weighted online-softmax merge to produce the final O and optionally the merged LSE.

final_O = sum_i exp(LSE_i - LSE_max) * O_i / sum_i exp(LSE_i - LSE_max)
final_LSE = LSE_max + log(sum_i exp(LSE_i - LSE_max))

This reduce step is shared across ASM, HipKittens/Opus, FlyDSL, Triton, and Gluon producers. It is pure reduction: no MFMA and no matrix core.

Current state

Primary production implementation is kn_mla_reduce_v1 / kn_mla_reduce_v1_ps in aiter/csrc/kernels/mla/reduce.cu. It runs on both gfx942 and gfx950. FlyDSL has no native reduce kernel today; its MLA decode tests call aiter's mla_reduce_v1.

The existing HIP kernel is arch-neutral:

No __gfx942__ / __gfx950__ guards.

Uses generic opus::buffer_load / buffer_store.

LDS is tiny: max_splits*sizeof(int32) + max_splits*sizeof(float) + overflow, typically only a few KB on MI300X.

Runtime guards use hipGetDeviceProperties().maxSharedMemoryPerMultiProcessor.

FlyDSL gfx942 implementation plan

Strategy: translate to FlyDSL. This is low risk and arch-neutral. The reference kernel already runs on gfx942, and FlyDSL already has a structurally similar reduction kernel: compile_moe_reduction in aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py.

The FlyDSL port should add a native MLA reduce kernel so the MLA stack can be fully FlyDSL-native without a separate HIP dependency.

Inputs / outputs

partial_output: fp32 [max_partial_row, H, Dv]

partial_lse: fp32 [max_partial_row, H]

reduce_indptr: int32 [#work+1]

reduce_partial_map: int32

Optional reduce_final_map: int32 x 2

final_output: bf16/fp16 [bs, H, Dv]

Optional final_lse: fp32 [bs, H]

Pipeline

# Constexpr: H, Dv, VEC = Dv // 128
@flyc.kernel(known_block_size=[128, 1, 1])
def mla_reduce_gfx942(partial_output, partial_lse, reduce_indptr,
                      reduce_partial_map, reduce_final_map,
                      final_output, final_lse, softmax_scale: fx.Float32):
    tile = fx.block_idx.x
    head = fx.block_idx.y
    tid = fx.thread_idx.x

    t0 = reduce_indptr[tile]
    t1 = reduce_indptr[tile + 1]
    n_splits = t1 - t0

    o_frag = fx.make_rmem_tensor(make_layout((VEC,)), fx.Float32)

    if n_splits < 4:
        # Simple path: online-softmax merge directly in registers.
        row0 = reduce_partial_map[t0]
        o_frag = load_vec(partial_output, row0, head, tid * VEC)
        max_lse = partial_lse[row0, head]
        sum_e = 1.0
        for s in range_constexpr(1, n_splits):
            rs = reduce_partial_map[t0 + s]
            o_s = load_vec(partial_output, rs, head, tid * VEC)
            lse = partial_lse[rs, head]
            nmax = _fmax(max_lse, lse)
            old = ArithValue(max_lse - nmax).exp2()
            new = ArithValue(lse - nmax).exp2()
            o_frag = Vec(o_frag) * old + Vec(o_s) * new
            max_lse = nmax
            sum_e = sum_e * old + new
        o_frag = Vec(o_frag) * rcp(sum_e)
    else:
        # Massive path: compute normalized per-split scale in LDS, then weighted accumulate.
        smem_scale = smem.alloc((max_splits,), fx.Float32)
        gmax = warp_reduce_max(local_lse)
        gsum = warp_reduce_add(exp2(local_lse - gmax))
        glse = gmax + log(gsum)
        smem_scale[split] = ArithValue(local_lse - glse).exp2()
        s_barrier()
        o_frag = 0
        for s in range_constexpr(0, n_splits):
            rs = reduce_partial_map[t0 + s]
            o_frag = Vec(o_frag) + Vec(load_vec(partial_output, rs, head, tid * VEC)) * smem_scale[s]

    store_vec(final_output, ..., cast(o_frag, out_dtype))
    if final_lse:
        final_lse[...] = final_lse_val

Concrete change list

Add aiter/aiter/ops/flydsl/kernels/mla_reduce.py with compile_mla_reduce(*, H, Dv, out_dtype, persistent, prefetch_depth=2, waves_per_eu=8, use_exp2=True, use_packed_cvt=False, use_packed_f32_fma=False).

Reuse compile_moe_reduction scaffolding for arg marshalling, BufferLoad/BufferStore atoms, and masked gather.

Reuse wave-reduce helpers from topk_gating_softmax_kernel.py and rocdl.exp2 / rocdl.rcp from softmax/rmsnorm kernels.

Add dispatcher wiring for flydsl_mla_reduce, falling back to HIP mla_reduce_v1 if FlyDSL is unavailable.

Verify against torch_mla_reduce_v1 for (H, Dv) in {(128,512), (16,512), (128,128)}, num_kv_splits in {1..16}, and boundary cases n_splits in {1,2,3,4,8,16}.

Benchmark and optimization notes

Before investing heavily, benchmark the current HIP mla_reduce_v1 on gfx942 and check if it is already HBM-bandwidth bound with rocprofv3. If it saturates HBM, the FlyDSL port is primarily a maintenance/toolchain win rather than a performance win.

Optimization knobs to expose and sweep:

prefetch_depth in {2,4,6,8} for deeper software prefetch.

waves_per_eu in {4,6,8} jointly tuned with prefetch depth.

use_exp2 to force hardware exp2 in the LSE-weight computation.

use_packed_cvt for packed fp32-to-bf16/fp16 output conversion.

use_packed_f32_fma to use packed FP32 FMA (V_PK_FMA_F32) on CDNA3 where useful.

Expected performance: memory-bound elementwise reduction, matching HIP mla_reduce_v1 within noise on gfx942. Risk: low. Effort: ~1–2 days.
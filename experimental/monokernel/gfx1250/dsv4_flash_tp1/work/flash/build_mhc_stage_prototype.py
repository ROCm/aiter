from pathlib import Path
root=Path(__file__).parent
s=(root.parent/'candidate/mhc_kernel.py').read_text()
s+='\nfrom gemm_stage_prototype import _gemm_mxfp8_preshuffle_bandwidth_bound_kernel as _qkv_wrapped\n_qkv_stage = _qkv_wrapped.fn\n'
a=s.index('@gluon.jit(repr=_mhc_post_pre_gemm_sqrsum_repr)')
b=s.index('\n\n# ============================================================================',a)
k=s[a:b].replace('_mhc_post_pre_gemm_sqrsum_gfx1250_kernel(', '_mhc_one_launch_kernel(',1)
k=k.replace('@gluon.jit(repr=_mhc_post_pre_gemm_sqrsum_repr)', '@gluon.jit')
k=k.replace('    SMEM_RED2: gl.constexpr,\n):','''    SMEM_RED2: gl.constexpr,
    next_post_ptr, next_comb_ptr, li_ptr, scale_ptr, base_ptr, norm_w_ptr,
    quant_ptr, quant_scale_ptr, counter_ptr, health_ptr,
    RA_APPLY: gl.constexpr, RA_RED4: gl.constexpr, RA_RED16: gl.constexpr,
    RA_SK: gl.constexpr, RA_SMEM_MIX: gl.constexpr, RA_SMEM_RES: gl.constexpr,
    GRID: gl.constexpr, SINKHORN: gl.constexpr, LAST_PRODUCER: gl.constexpr,
    qkv_w_ptr, qkv_ws_ptr, qkv_out_ptr, FUSE_QKV: gl.constexpr,
):''')
k+='''
    # All payload stores (including TDM residual stores) complete before the
    # release publication. Every CTA participates; no partial conditional barrier.
    gl.barrier()
    gl.static_assert(GRID <= 128, "Conservative experimental resident-grid bound")
    gl.static_assert((GRID & (GRID - 1)) == 0, "Counter wrap requires power-of-two grid")
    ticket = gl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if LAST_PRODUCER:
        ready = (ticket % GRID) == (GRID - 1)
        row_start = 0
        row_end = M
    else:
        target = ((ticket // GRID) + 1) * GRID
        seen = gl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")
        spins = 0
        while (seen != target) & (spins < 100000):
            seen = gl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")
            spins += 1
        ready = seen == target
        if not ready:
            gl.atomic_xchg(health_ptr, 1, sem="relaxed", scope="gpu")
        row_start = pid_m * SPLIT_K + pid_k
        row_end = gl.minimum(row_start + 1, M)
    if ready:
        K = _RAConfig(C, HC, SPLIT_K, 1, C, 1024, C // 1024,
                      SINKHORN, RES_SHUFFLED, True, False,
                      RA_APPLY, RA_RED4, RA_RED16, RA_SK,
                      RA_SMEM_MIX, RA_SMEM_RES, KS, True)
        for row in range(row_start, row_end):
            _mono_reduce_apply_main(next_post_ptr, next_comb_ptr, li_ptr,
                                   out_ptr, sqrsum_ptr, scale_ptr, base_ptr,
                                   nres_ptr, norm_w_ptr, quant_ptr, quant_scale_ptr,
                                   M, out_stride, 1.e-6, 1.e-6, 1.e-6, 1.e-6, 2.0,
                                   K, row)

    if FUSE_QKV:
        gl.static_assert(not LAST_PRODUCER, "All CTAs must join the second stage")
        gl.barrier()
        ticket2 = gl.atomic_add(counter_ptr + 1, 1, sem="acq_rel", scope="gpu")
        target2 = ((ticket2 // GRID) + 1) * GRID
        seen2 = gl.atomic_add(counter_ptr + 1, 0, sem="acquire", scope="gpu")
        spins2 = 0
        while (seen2 != target2) & (spins2 < 100000):
            seen2 = gl.atomic_add(counter_ptr + 1, 0, sem="acquire", scope="gpu")
            spins2 += 1
        if seen2 != target2:
            gl.atomic_xchg(health_ptr, 2, sem="relaxed", scope="gpu")
        task = pid_m * SPLIT_K + pid_k
        if (seen2 == target2) & (task < ((M + 15) // 16) * 24):
            _qkv_stage(quant_ptr.to(gl.pointer_type(gl.uint8)), qkv_w_ptr.to(gl.pointer_type(gl.uint8)),
                       qkv_out_ptr, quant_scale_ptr, qkv_ws_ptr, M, 1536, C,
                       C, 1, C * 16, 1, 0, 1536, 1, 1, M, C // 128, 1,
                       task, 16, 64, 256, 1, 128, True, 1, C, True,
                       4, ((0, 1), (1, 0)), ".ca", 3, 1, True, 0, 1, 1, 1)

'''
a=s.index('@gluon.jit\ndef _reduce_apply_main(')
b=s.index('\n\n@gluon.jit\ndef _post_sinkhorn_ws',a)
r=s[a:b].replace('def _reduce_apply_main(', 'def _mono_reduce_apply_main(',1)
r=r.replace('    K,  # _RAConfig\n):','    K,  # _RAConfig\n    row_id,\n):')
r=r.replace('    pid_m = gl.program_id(0)\n    pid_k = gl.program_id(1)','    pid_m = row_id\n    pid_k = 0',1)
(root/'mhc_stage_prototype.py').write_text(s+'\n\n'+r+'\n\n'+k)

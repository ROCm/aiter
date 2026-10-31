// SPDX-License-Identifier: MIT
// r18-put: the PUT kernel of the r17 pipeline (route B). UNCOMPILED source (Evidence builds). Launch order per call: flydsl r17 GEMM (writes S[4][srows][F], dest-major, BF16) -> THIS kernel -> r13's lg_quiet_ready_v2 (unchanged code).
// Derived from lgc::r6_poster_block: the SAME R12 protocol (credit tag + credit release, bounded credit wait with tag check, early arrival tag, ERR-sticky, quietQueue drain, arrival release), minus everything that existed only to overlap the puts with the GEMM:
// NO gdone waits (the GEMM kernel finished before this kernel starts: every S row is ready), ONE put per peer (live rows only), and the SELF slab copy that r13's GEMM epilogue did for free (self tile written straight into R) is done by LGP_SELF_WGS extra workgroups.
// See README-PUT.md for the contract and the order proof.
#pragma once
namespace lgc {
// S layout (r17): slab d = rows [0, srows) x F BF16 at byte offset d * 2 * srows * F of the S window; srows = LGP_SROWS(a) = the GEMM's launch M (>= m_real, <= M: the window is sized for the plan bucket M). Live rows of every slab: [0, m_real).
// R layout (r13 compact): sender s's slab lands at byte offset s * 2 * m_real * F of the DESTINATION's R window (R = [4*m_real, F] contiguous).
__device__ __forceinline__ bool r18_put_cfg_ok(const LgCommArgsV4& a) {
    return a.abi_version == LG_COMM_ABI_VERSION && a.items_per_group == LG_ITEMS_PER_GROUP && a.sdma_queues == 1u && a.chunks <= LG_MAX_CHUNKS && a.chunks == a.M / 256u && a.reserved[0] == LG_R18_PUT_PLAN_CODE && a.reserved[1] != 0ull &&
           a.N == LG_X_N_RAW && a.F == LG_COMM_F && a.err != nullptr && a.step != nullptr && a.qr_wgs == LG_QR_WGS && a.m_real >= 1u && a.m_real <= a.M && a.pad4 == 0u &&
           LGP_SROWS(a) >= a.m_real && LGP_SROWS(a) <= a.M;
}
// ---- self copy: S slab `rank` (rows [0, m_real)) -> R[rank * m_real ..] (16 B grid-stride; contiguous because the slab rows are consecutive); `nwg` workgroups x 512 threads, share of workgroup `wg` ----
__device__ __forceinline__ void r18_self_copy(const LgCommArgsV4& a, uint32_t wg, uint32_t nwg) {
    typedef __attribute__((ext_vector_type(4))) uint32_t u32x4;
    const u32x4* src = reinterpret_cast<const u32x4*>(reinterpret_cast<const char*>(ccoGetLocalPtr(a.Swin, 0)) + static_cast<uint64_t>(a.rank) * 2ull * LGP_SROWS(a) * a.F);
    u32x4* dst = reinterpret_cast<u32x4*>(reinterpret_cast<char*>(ccoGetLocalPtr(a.Rwin, 0)) + static_cast<uint64_t>(a.rank) * 2ull * a.m_real * a.F);
    const uint64_t n16 = (2ull * a.m_real * a.F) / 16ull;                          // 2 * m_real * F is a multiple of 16 (F % 8 == 0)
    for (uint64_t i = static_cast<uint64_t>(wg) * 512ull + opus::thread_id_x(); i < n16; i += static_cast<uint64_t>(nwg) * 512ull) dst[i] = src[i];
}
// ---- the poster: block 0, waves 0..2 active, one issuing lane each (peer = peer_of(rank, wave)); all other threads exit uniformly ----
__device__ __forceinline__ void r18_put_poster(const LgCommArgsV4& a) {
    const int tid = static_cast<int>(opus::thread_id_x()), wave = tid / 64, lane = tid % 64;
    if (wave >= 3 || lane != 0) return;
    const int rank = static_cast<int>(a.rank), peer = peer_of(rank, wave);
    if (!err_clear(a)) return;                                                    // r12 sticky: ERR already set => no credit, no wait, no put, no arrival
    const uint64_t e = ld_u64(cw(a, LGC_EP_OFF));
    const uint64_t E = tag_E(a);                                                  // (STEP << 32) | call_id: rank-uniform by construction
#if LG_X_TEST_BAD_ATAG
    const uint64_t atag_v = (wave == 0) ? (E ^ 1ull) : E;                         // seam (r12 negative control): peer slot 0 presents a wrong ARRIVAL tag
#else
    const uint64_t atag_v = E;
#endif
    const uint64_t budget = r6_budget_ticks(a);
    uint32_t issued = 0;
    auto fail = [&](uint32_t reason) {                                            // F-A: on a timeout after a put was issued, drain that peer's queue before returning poisoned
        set_poison(a, reason);
        if (issued != 0u) ccoSdma{a.dev}.template quietQueue<ccoCoopThread>(peer, 0u);
    };
    {   // r12 D-2: credit TAG (relaxed system store) in program order before this lane's own release of the credit
#if LG_X_TEST_BAD_CTAG
        const uint64_t ctag_v = (wave == 0) ? (E ^ 1ull) : E;                      // seam: peer slot 0 presents a wrong CREDIT tag
#else
        const uint64_t ctag_v = E;
#endif
#if LG_X_TEST_CTAG_AFTER_RELEASE
        st_u64_rel_x(cw_peer(a, peer, LGC_CREDIT_OFF + 8u * a.rank), e);           // seam (negative control): credit released BEFORE the tag store, tag delayed => stale-tag mismatches MUST appear
        for (uint32_t i = 0; i < LG_X_TEST_TAG_DELAY_ITERS; ++i) __builtin_amdgcn_s_sleep(100);
        __hip_atomic_store(reinterpret_cast<uint64_t*>(cw_peer(a, peer, LGC_CTAG_OFF + 8u * a.rank)), ctag_v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        __hip_atomic_store(reinterpret_cast<uint64_t*>(cw_peer(a, peer, LGC_CTAG_OFF + 8u * a.rank)), ctag_v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        st_u64_rel_x(cw_peer(a, peer, LGC_CREDIT_OFF + 8u * a.rank), e);           // credit: "this rank began epoch e" -> the peer may PUT into this rank's R
#endif
    }
    {   // bounded wait for the PEER's credit (peer began epoch e => its R is free); then the credit-tag check BEFORE any put
        const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
        const int crc = r12_poll_u64_ge(a, reinterpret_cast<const uint64_t*>(cw(a, LGC_CREDIT_OFF + 8u * peer)), e, budget);
        add_report(a, 1, __builtin_amdgcn_s_memrealtime() - t0);
        if (crc < 0) return;                                                      // ERR observed: nothing issued yet
        if (crc == 0) { fail(LGC_POISON_W1_TIMEOUT); return; }
        const uint64_t seen = __hip_atomic_load(reinterpret_cast<const uint64_t*>(cw(a, LGC_CTAG_OFF + 8u * peer)), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        if (seen != E) { set_err(a, LGC_POISON_TAG_CREDIT, E, seen, e, e); return; }
#if LG_TAG_EARLY && !LG_X_TEST_ATAG_AFTER_RELEASE
        __hip_atomic_store(reinterpret_cast<uint64_t*>(cw_peer(a, peer, LGC_ATAG_OFF + 8u * a.rank)), atag_v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);   // r12b F1: arrival tag in the batch of the put
#endif
    }
    if (!err_clear(a)) return;                                                    // r12: no put once ERR is set
    {   // ONE put of the peer's slab (live rows only): S[peer] -> R_peer[rank * m_real ..]; queue 0; no signal
        const uint64_t bytes = 2ull * a.m_real * a.F;                             // >= 2 * F > 0: m_real >= 1 is a config-guard condition
        const uint64_t src_off = static_cast<uint64_t>(peer) * 2ull * LGP_SROWS(a) * a.F;
        const uint64_t dst_off = static_cast<uint64_t>(rank) * 2ull * a.m_real * a.F;
        ccoSdma{a.dev}.template put<ccoCoopThread, false, false, ccoSdmaOptFlagsDefault, ccoSdmaThreadIndependent>(
            peer, a.Rwin, static_cast<size_t>(dst_off), a.Swin, static_cast<size_t>(src_off), static_cast<size_t>(bytes), 0u);
        ++issued; add_report(a, 2, 1);                                            // inline_posts: expect 3 per epoch (r13 with n = 1: 3)
    }
    ccoSdma{a.dev}.template quietQueue<ccoCoopThread>(peer, 0u);                 // drain: the put completed at the destination (CCO quiet semantics, claim C6)
    if (!err_clear(a)) return;                                                    // r12: ERR set while the put drained => no arrival
#if LG_X_TEST_ATAG_AFTER_RELEASE
    st_u64_rel_x(cw_peer(a, peer, LGC_ARRIVAL_OFF + 8u * a.rank), e);              // seam (negative control): arrival released BEFORE the tag store, tag delayed
    for (uint32_t i = 0; i < LG_X_TEST_TAG_DELAY_ITERS; ++i) __builtin_amdgcn_s_sleep(100);
    __hip_atomic_store(reinterpret_cast<uint64_t*>(cw_peer(a, peer, LGC_ATAG_OFF + 8u * a.rank)), atag_v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#elif LG_TAG_EARLY
    st_u64_rel_x(cw_peer(a, peer, LGC_ARRIVAL_OFF + 8u * a.rank), e);             // RELEASE arrival (vmcnt(0) drains quietQueue's loads): "all of rank's epoch-e puts into peer's R are complete"
#else
    __hip_atomic_store(reinterpret_cast<uint64_t*>(cw_peer(a, peer, LGC_ATAG_OFF + 8u * a.rank)), atag_v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    st_u64_rel_x(cw_peer(a, peer, LGC_ARRIVAL_OFF + 8u * a.rank), e);
#endif
}
// ---- the entry body: block 0 = poster, blocks 1..LGP_SELF_WGS = self copy (skipped under ERR: quiet_ready fills R with NaN instead) ----
__device__ __forceinline__ void r18_put_body(const LgCommArgsV4& a) {
    if (!r18_put_cfg_ok(a)) {
        if (opus::block_id_x() == 0 && opus::thread_id_x() == 0) set_poison(a, LGC_POISON_CONFIG);
        return;
    }
    const uint32_t wg = static_cast<uint32_t>(opus::block_id_x());
    if (wg == 0u) { r18_put_poster(a); return; }
    if (wg > LGP_SELF_WGS) return;                                                // an oversized grid does nothing
    if (!err_clear(a)) return;
    r18_self_copy(a, wg - 1u, LGP_SELF_WGS);
}
} // namespace lgc
// SPDX-License-Identifier: MIT
// LG-QSDMA-r6 communication slice (r5b L2b + ONE resident poster block (blockIdx 0, 3 waves = 3 peers) posting n big puts per peer; item WGs are blocks 1..; LG_R6_POSTER=0 = r5b control): FP8 quad-subtile GEMM (r1e np math, mma(a,b,c,127,127,0,0)) + chunk-SDMA PUT + quiet_ready.
// UNCOMPILED source (Implementor does not build). GEMM body derives from LG-QSDMA-r1 r1e (np, passed M=2048 correctness);
// r1f persistent mitigation NOT taken. Single receive buffer R: self slab is written DIRECTLY into R[rank]; other slabs go
// to local staging S_d and are PUT (no-signal, queue 0) to peer R[rank] chunk by the last arriver of each (dst,chunk) when the
// peer's credit >= e; otherwise the chunk stays READY and quiet_ready posts it. A GEMM WG never spins on another GPU.
// Schedule (r4): grid = lgc::grid_items(chunks) (504 at M=2048 with LG_REMOTE_HALF=1: 432 remote half-N items then 72 self tiles); r3: grid = chunks*36; REMOTE tiles first (chunk-major among remote), SELF tiles last (see lgc::select_item). r2 was: chunk c = t/36, destination slot (t%36)/9 rotated by rank starting at
// self, n-tile t%9) so a chunk's 9 tiles for one destination finish together and its PUT overlaps later GEMM tiles.
#include <opus/hip_minimal.hpp>
#include "lg_comm_abi_v2.h"
#ifdef __HIP_DEVICE_COMPILE__
#include <opus/opus.hpp>
#include "layout/quad_fp8_traits.hpp"
#include "layout/quad_fp8_fieldmap.hpp"
#include "layout/quad_fp8_layouts.hpp"
#include "epilogue_ptpc_pairstore.hpp"
#include "lg_comm_device.hpp"
#include "lg_r6x.hpp"
// r4b: dense field map is hard-wired to F=2304 / 36 virtual tiles; an indexer slice (F=2560) needs its own FieldSeg table (kFields) first.
static_assert(LG_COMM_F == lgq::kF + (LG_X_INDEXER ? 256u : 0u) && 4 * static_cast<int>(LG_TILES_N_PER_DEST) == lgq::kTilesN && lgq::kFS == static_cast<int>(LG_COMM_F), "r6x: dense Q|K|V field map (F=2304, 36 tiles) [+ one IQ|IK tile column = F 2560]");

namespace lgq {
// Per-half-tile scale + store (identical math/ordering to the full-tile epilogue).
template<class VC>
__device__ __forceinline__ void epi_store(const LgCommArgsV2&, const VC& vc, const HalfScales& ws, const float (&as)[Q::E_M],
                                          opus::bf16_t* c_slab, int row_hm, int packed, int lane_id, int wave_id_m, int wave_id_n, bool bypass) {
    store_c_ptpc(vc, ws, as, c_slab, row_hm, packed, lane_id, wave_id_m, wave_id_n, bypass);
}
__device__ __forceinline__ void load_ws(const float* w_scale, int raw, int wave_id_n, int q, HalfScales& ws) {
    opus::static_for<Q::E_N>([&](auto cen) {
        constexpr int en = cen.value;
        __builtin_memcpy(&ws.w[en], w_scale + raw + (en * Q::T_N + wave_id_n) * 16 + 4 * q, 16);   // 16-B aligned (raw % 128 == 0)
    });
}
__device__ __forceinline__ void load_as(const float* a_scale, int row, int hm, int wave_id_m, int l16, float (&as)[Q::E_M]) {
    opus::static_for<Q::E_M>([&](auto cem) {
        constexpr int em = cem.value;
        as[em] = a_scale[row + hm * Q::HALF_B_M + em * (Q::T_M * Q::W_M) + wave_id_m * Q::W_M + l16];
    });
}
// r7 HALF-N items (256x128 = one raw 128-col segment), revived from LG-GEMM-r2 half_item and specialised per KIND (T_HN / T_HQ / T_HK) so the kind-specific state never shares a register file with a full-tile copy.
// Same per-element MFMA chain as the full tile (K ascending, mma(a,b,c,127,127,0,0)), same epilogue math, one BF16 rounding.
// r7 half-item helpers (kind-specific scalars; everything except `raw` is resolved after the K loop)
template<int KIND> __device__ __forceinline__ int half_dest(const LgCommArgsV2& a, const lgc::XItem& xi) {
    if constexpr (KIND == lgc::T_HK) { (void)a; (void)xi; return 0; }
    else if constexpr (KIND == lgc::T_HQ) { (void)a; return xi.nloc; }
    else { return lgc::dest_of_slot(static_cast<int>(a.rank), xi.dslot); }
}
template<int KIND> __device__ __forceinline__ int half_hv(const LgCommArgsV2& a, const lgc::XItem& xi) {   // T_HN: dense field-table half-tile index
    return (half_dest<KIND>(a, xi) * static_cast<int>(LG_TILES_N_PER_DEST) + xi.nloc) * 2 + xi.nsel;
}
template<int KIND> __device__ __forceinline__ int half_raw(const LgCommArgsV2& a, const lgc::XItem& xi) {
    if constexpr (KIND == lgc::T_HK) { (void)a; (void)xi; return static_cast<int>(LG_X_IK_RAW); }
    else if constexpr (KIND == lgc::T_HQ) { (void)a; return static_cast<int>(LG_X_IQ_RAW) + xi.nloc * 128; }
    else { return lgq::ht_raw(half_hv<KIND>(a, xi)); }
}
template<int KIND> __device__ __forceinline__ int half_packed(const lgc::XItem& xi) {
    if constexpr (KIND == lgc::T_HK) { (void)xi; return static_cast<int>(LG_X_IK_PACKED); }
    else if constexpr (KIND == lgc::T_HQ) { (void)xi; return static_cast<int>(LG_X_IQ_PACKED); }
    else { return -1; }                                                // T_HN: the caller uses the dense table (ht_packed)
}
template<int NP, int HG, int KIND>
__device__ __forceinline__ void half_body(const LgCommArgsV2& args, const lgc::XItem& xi, char* smem_a, char* smem_b) {   // r7: 256x128 half-N item of kind T_HN (normal tile half) / T_HQ (IQ_d) / T_HK (IK, 4 destinations)
    using namespace opus;
    using opus::operator""_I;
    using T = lgq::Q;
    const int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    const int lane_id = opus::thread_id_x() % get_warp_size();
    const int M = static_cast<int>(args.M), K = static_cast<int>(args.K);
    const int wave_id_m = wave_id / T::T_N, wave_id_n = wave_id % T::T_N;
    smem<fp8_t> s_a[2][2] = {
        {make_smem(reinterpret_cast<fp8_t*>(smem_a)), make_smem(reinterpret_cast<fp8_t*>(smem_a + T::smem_a_bytes))},
        {make_smem(reinterpret_cast<fp8_t*>(smem_a + 2 * T::smem_a_bytes)), make_smem(reinterpret_cast<fp8_t*>(smem_a + 3 * T::smem_a_bytes))}};
    smem<fp8_t> s_b[2] = {make_smem(reinterpret_cast<fp8_t*>(smem_b)), make_smem(reinterpret_cast<fp8_t*>(smem_b + T::smem_b_bytes))};
    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(
        seq<T::E_M, T::E_N, T::E_K>{}, seq<T::T_M, T::T_N, T::T_K>{}, seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);
    auto mma_fp8 = [&](const auto& a_, const auto& b_, const auto& c_) { return mma(a_, b_, c_, 127, 127, number<0>{}, number<0>{}); };
    const int stride_band = 16 * K;
    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, static_cast<int>(args.strideA));
    auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb_phys<T>(lane_id, wave_id, stride_band);
    auto u_sb = make_layout_sb_phys<T>(wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb_phys<T>(lane_id, wave_id_n);

    const int chunk = xi.chunk;
    const int row = chunk * T::B_M;                       // 256-row M chunk (== m tile)
    const int raw = half_raw<KIND>(args, xi);             // only `raw` (one scalar) is kind-dependent inside the K loop; packed column / destinations are resolved after it
    auto g_a = make_gmem(reinterpret_cast<const fp8_t*>(args.A) + static_cast<long long>(row) * args.strideA,
                         static_cast<unsigned int>(static_cast<unsigned long long>(M - row) * args.strideA));
    auto g_b = make_gmem(reinterpret_cast<const fp8_t*>(args.W), static_cast<unsigned int>(args.N) * static_cast<unsigned int>(K));
    auto a_offset = [&](int hm, int tile_k) { return hm * T::HALF_B_M * static_cast<int>(args.strideA) + tile_k * T::B_K; };
    auto b_offset = [&](int tile_k) { return (raw / 16) * stride_band + tile_k * T::BAND_BYTES_PER_KTILE; };

    typename MMA::vtype_a v_a[2];
    typename MMA::vtype_b v_b;
    typename MMA::vtype_c v_c[2];
    clear(v_c[0]); clear(v_c[1]);
    const int loops = K / T::B_K;                         // 48
    constexpr int LOADS = T::a_buffer_load_insts * 2 + T::b_buffer_load_insts;   // 6 per K-tile
    auto issue = [&](int buf, int tile_k) {
        async_load<T::VEC_B>(g_b, s_b[buf].ptr, u_gb, u_sb, b_offset(tile_k));
        async_load<T::VEC_A>(g_a, s_a[buf][0].ptr, u_ga, u_sa, a_offset(0, tile_k));
        async_load<T::VEC_A>(g_a, s_a[buf][1].ptr, u_ga, u_sa, a_offset(1, tile_k));
    };
    issue(0, 0);
    issue(1, 1);                                          // loops >= 2
    auto step = [&](auto cbuf, int k) {
        constexpr int buf = decltype(cbuf)::value;
        if (k + 1 < loops) s_waitcnt_vmcnt(number<LOADS>{});   // tile k landed (tile k+1 may still be in flight)
        else               s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();                     // all waves' tile-k LDS writes visible
        v_b = load<T::VEC_B>(s_b[buf], u_rb);
        v_a[0] = load<T::VEC_A>(s_a[buf][0], u_ra);
        v_a[1] = load<T::VEC_A>(s_a[buf][1], u_ra);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();                     // everyone finished reading buf: safe to overwrite
        if (k + 2 < loops) issue(buf, k + 2);
        __builtin_amdgcn_s_setprio(1);
        v_c[0] = mma_fp8(v_a[0], v_b, v_c[0]);
        v_c[1] = mma_fp8(v_a[1], v_b, v_c[1]);
        __builtin_amdgcn_s_setprio(0);
    };
    for (int k = 0; k < loops; k += 2) { step(number<0>{}, k); step(number<1>{}, k + 1); }   // loops even
    // all loads of this item have landed (last step used vmcnt(0)); epilogue
    const float* const w_scale = args.Wscale;
    const float* const a_scale = args.Ascale;
    const int q = lane_id / 16, l16 = lane_id % 16;
    const long long slab_elems = static_cast<long long>(M) * lgq::kFS;
    auto slab_of = [&](int dd_) -> bf16_t* {
        return (dd_ == static_cast<int>(args.rank))
            ? reinterpret_cast<bf16_t*>(mori::cco::ccoGetLocalPtr(args.Rwin, 0)) + static_cast<long long>(args.rank) * slab_elems   // self -> R directly
            : reinterpret_cast<bf16_t*>(mori::cco::ccoGetLocalPtr(args.Swin, 0)) + static_cast<long long>(dd_) * slab_elems;       // remote -> S_dd
    };
    const int rk = static_cast<int>(args.rank);
    const int packed = (KIND == lgc::T_HN) ? lgq::ht_packed(half_hv<KIND>(args, xi)) : half_packed<KIND>(xi);
    const int dd = half_dest<KIND>(args, xi);
    HalfScales ws; load_ws(w_scale, raw, wave_id_n, q, ws);
    static_for<2>([&](auto chm) {
        constexpr int hm = chm.value;
        float as[T::E_M]; load_as(a_scale, row, hm, wave_id_m, l16, as);
        if constexpr (KIND == lgc::T_HK) {                                // IK: one cast, same 16 B to R[rank] + 3 peer S slabs
            bf16_t* const slabs4[4] = {slab_of(rk), slab_of(lgc::peer_of(rk, 0)), slab_of(lgc::peer_of(rk, 1)), slab_of(lgc::peer_of(rk, 2))};
            store_c_ptpc_multi(v_c[hm], ws, as, slabs4, 4, row + hm * T::HALF_B_M, packed, lane_id, wave_id_m, wave_id_n);
        } else {
            store_c_ptpc(v_c[hm], ws, as, slab_of(dd), row + hm * T::HALF_B_M, packed, lane_id, wave_id_m, wave_id_n, dd != rk);
        }
    });
    if constexpr (KIND == lgc::T_HK) {
        lgc::x_publish<NP, HG>(args, lgc::peer_of(rk, 0), lgc::peer_of(rk, 1), lgc::peer_of(rk, 2), 3, static_cast<uint32_t>(chunk));
    } else {
        const int nd = (dd != rk) ? 1 : 0;
        if (nd == 0) lgc::stamp_self_end(args, chunk);
        lgc::x_publish<NP, HG>(args, dd, 0, 0, nd, static_cast<uint32_t>(chunk));
    }
}
// ---- r9 QUARTER item (128 rows x 128 cols = ONE hm half of ONE hn half of a 256x256 tile): the proven half pipeline with ONE A half (msel) and ONE accumulator. Same per-element MFMA chain (K ascending,
// mma(a,b,c,127,127,0,0), swap_ab), same epilogue math (one RNE), same B layout; A half msel lands in s_a[buf][0] (the A layouts are per-half and identical for both halves). 4 loads per K-tile (2 A + 2 B).
template<int NP, int HG>
__device__ __forceinline__ void quarter_body(const LgCommArgsV2& args, const lgc::XItem& xi, char* smem_a, char* smem_b) {
    using namespace opus;
    using opus::operator""_I;
    using T = lgq::Q;
    const int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    const int lane_id = opus::thread_id_x() % get_warp_size();
    const int M = static_cast<int>(args.M), K = static_cast<int>(args.K);
    const int wave_id_m = wave_id / T::T_N, wave_id_n = wave_id % T::T_N;
    smem<fp8_t> s_a[2] = {make_smem(reinterpret_cast<fp8_t*>(smem_a)), make_smem(reinterpret_cast<fp8_t*>(smem_a + T::smem_a_bytes))};
    smem<fp8_t> s_b[2] = {make_smem(reinterpret_cast<fp8_t*>(smem_b)), make_smem(reinterpret_cast<fp8_t*>(smem_b + T::smem_b_bytes))};
    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(
        seq<T::E_M, T::E_N, T::E_K>{}, seq<T::T_M, T::T_N, T::T_K>{}, seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);
    auto mma_fp8 = [&](const auto& a_, const auto& b_, const auto& c_) { return mma(a_, b_, c_, 127, 127, number<0>{}, number<0>{}); };
    const int stride_band = 16 * K;
    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, static_cast<int>(args.strideA));
    auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb_phys<T>(lane_id, wave_id, stride_band);
    auto u_sb = make_layout_sb_phys<T>(wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb_phys<T>(lane_id, wave_id_n);

    const int chunk = xi.chunk;
    const int row = chunk * T::B_M;
    const int hn = xi.nsel & 1, msel = xi.nsel >> 1;                  // column half, row half of the 256x256 tile
    const int raw = lgq::ht_raw((half_dest<lgc::T_HN>(args, xi) * static_cast<int>(LG_TILES_N_PER_DEST) + xi.nloc) * 2 + hn);   // only `raw` is item-dependent inside the K loop
    auto g_a = make_gmem(reinterpret_cast<const fp8_t*>(args.A) + static_cast<long long>(row) * args.strideA,
                         static_cast<unsigned int>(static_cast<unsigned long long>(M - row) * args.strideA));
    auto g_b = make_gmem(reinterpret_cast<const fp8_t*>(args.W), static_cast<unsigned int>(args.N) * static_cast<unsigned int>(K));
    auto a_offset = [&](int tile_k) { return msel * T::HALF_B_M * static_cast<int>(args.strideA) + tile_k * T::B_K; };
    auto b_offset = [&](int tile_k) { return (raw / 16) * stride_band + tile_k * T::BAND_BYTES_PER_KTILE; };

    typename MMA::vtype_a v_a;
    typename MMA::vtype_b v_b;
    typename MMA::vtype_c v_c;
    clear(v_c);
    const int loops = K / T::B_K;                                      // 48
    constexpr int LOADS = T::a_buffer_load_insts + T::b_buffer_load_insts;   // 4 per K-tile (one A half + B)
    auto issue = [&](int buf, int tile_k) {
        async_load<T::VEC_B>(g_b, s_b[buf].ptr, u_gb, u_sb, b_offset(tile_k));
        async_load<T::VEC_A>(g_a, s_a[buf].ptr, u_ga, u_sa, a_offset(tile_k));
    };
    issue(0, 0);
    issue(1, 1);                                                       // loops >= 2
    auto step = [&](auto cbuf, int k) {
        constexpr int buf = decltype(cbuf)::value;
        if (k + 1 < loops) s_waitcnt_vmcnt(number<LOADS>{});           // tile k landed (tile k+1 may still be in flight)
        else               s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();                                  // all waves' tile-k LDS writes visible
        v_b = load<T::VEC_B>(s_b[buf], u_rb);
        v_a = load<T::VEC_A>(s_a[buf], u_ra);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();                                  // everyone finished reading buf: safe to overwrite
        if (k + 2 < loops) issue(buf, k + 2);
        __builtin_amdgcn_s_setprio(1);
        v_c = mma_fp8(v_a, v_b, v_c);
        __builtin_amdgcn_s_setprio(0);
    };
    for (int k = 0; k < loops; k += 2) { step(number<0>{}, k); step(number<1>{}, k + 1); }   // loops even
    const float* const w_scale = args.Wscale;
    const float* const a_scale = args.Ascale;
    const int q = lane_id / 16, l16 = lane_id % 16;
    const long long slab_elems = static_cast<long long>(M) * lgq::kFS;
    bf16_t* const sbase = reinterpret_cast<bf16_t*>(mori::cco::ccoGetLocalPtr(args.Swin, 0));
    const int dd = half_dest<lgc::T_HN>(args, xi);                     // remote destination (quarters are remote-only)
    bf16_t* const c_slab = sbase + static_cast<long long>(dd) * slab_elems;
    const int packed = lgq::ht_packed((dd * static_cast<int>(LG_TILES_N_PER_DEST) + xi.nloc) * 2 + hn);
    HalfScales ws; load_ws(w_scale, raw, wave_id_n, q, ws);
    {
        float as[T::E_M]; load_as(a_scale, row, msel, wave_id_m, l16, as);
        store_c_ptpc(v_c, ws, as, c_slab, row + msel * T::HALF_B_M, packed, lane_id, wave_id_m, wave_id_n, dd != static_cast<int>(args.rank));
    }
    lgc::x_publish<NP, HG>(args, dd, 0, 0, 1, static_cast<uint32_t>(chunk));
}
} // namespace lgq

namespace lgq {
// ---- r6x3: item-kind specialisation. tile_body<NP,KIND> is a separately inlined copy of the tile pipeline per kind (forceinline on purpose: a __noinline__ callee would follow the AMDGPU
// call ABI and save callee-saved high VGPRs to scratch, i.e. private-segment spills). The three copies sit in mutually exclusive branches => disjoint live ranges; the NORMAL copy has NO indexer variable.
template<int KIND> __device__ __forceinline__ int tile_dest(const LgCommArgsV2& a, const lgc::XItem& xi, int hn) {   // destination rank of half hn (IK: unused, returns 0)
    if constexpr (KIND == lgc::T_IK) { (void)a; (void)xi; (void)hn; return 0; }
    else if constexpr (KIND == lgc::T_IQ) { (void)a; (void)hn; return xi.nloc; }                // IQ_d: ONE destination rank for the whole tile
    else { (void)hn; return lgc::dest_of_slot(static_cast<int>(a.rank), xi.dslot); }
}
template<int KIND> __device__ __forceinline__ int tile_raw(const LgCommArgsV2& a, const lgc::XItem& xi, int hn) {
    if constexpr (KIND == lgc::T_IK) { (void)a; (void)xi; (void)hn; return static_cast<int>(LG_X_IK_RAW); }
    else if constexpr (KIND == lgc::T_IQ) { (void)a; (void)hn; return static_cast<int>(LG_X_IQ_RAW) + xi.nloc * 128; }   // hn=1 duplicates hn=0 (not stored), exactly like IK
    else { return lgq::ht_raw((tile_dest<KIND>(a, xi, hn) * static_cast<int>(LG_TILES_N_PER_DEST) + xi.nloc) * 2 + hn); }
}
template<int KIND> __device__ __forceinline__ int tile_packed(const LgCommArgsV2& a, const lgc::XItem& xi, int hn) {
    if constexpr (KIND == lgc::T_IK) { (void)a; (void)xi; (void)hn; return static_cast<int>(LG_X_IK_PACKED); }
    else if constexpr (KIND == lgc::T_IQ) { (void)a; (void)xi; (void)hn; return static_cast<int>(LG_X_IQ_PACKED); }
    else { return lgq::ht_packed((tile_dest<KIND>(a, xi, hn) * static_cast<int>(LG_TILES_N_PER_DEST) + xi.nloc) * 2 + hn); }
}
// tile body: everything of the old gemm_body AFTER the dispatch. smem_a/smem_b are the kernel's ONE LDS allocation (passed in; the three kinds share it, so LDS stays 135168 B).
template<int NP, int HG, int KIND>
__device__ __forceinline__ void tile_body(const LgCommArgsV2& args, const lgc::XItem& xi, char* smem_a, char* smem_b) {
    using namespace opus;
    using namespace lgq;
    using opus::operator""_I;
    using T = lgq::Q;
    const int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    const int lane_id = opus::thread_id_x() % get_warp_size();
    const int M = static_cast<int>(args.M), K = static_cast<int>(args.K);
    const int wave_id_m = wave_id / T::T_N, wave_id_n = wave_id % T::T_N;

    smem<fp8_t> s_a[2][2] = {
        {make_smem(reinterpret_cast<fp8_t*>(smem_a)), make_smem(reinterpret_cast<fp8_t*>(smem_a + T::smem_a_bytes))},
        {make_smem(reinterpret_cast<fp8_t*>(smem_a + 2 * T::smem_a_bytes)), make_smem(reinterpret_cast<fp8_t*>(smem_a + 3 * T::smem_a_bytes))}};
    smem<fp8_t> s_b[2][2] = {
        {make_smem(reinterpret_cast<fp8_t*>(smem_b)), make_smem(reinterpret_cast<fp8_t*>(smem_b + T::smem_b_bytes))},
        {make_smem(reinterpret_cast<fp8_t*>(smem_b + 2 * T::smem_b_bytes)), make_smem(reinterpret_cast<fp8_t*>(smem_b + 3 * T::smem_b_bytes))}};
    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(
        seq<T::E_M, T::E_N, T::E_K>{}, seq<T::T_M, T::T_N, T::T_K>{}, seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);
    static_assert(sizeof(typename MMA::vtype_c) / sizeof(float) == T::E_M * T::E_N * 4, "sub-tile acc = 32 floats");
    // plain mma(a,b,c) has no fp8 16x16x128 dispatch (opus.hpp l.3105-3132 -> unreachable); use the scaled f8f6f4 form, 127 = 2^0.
    auto mma_fp8 = [&](const auto& a_, const auto& b_, const auto& c_) {
        return mma(a_, b_, c_, 127, 127, number<0>{}, number<0>{});
    };
    const int stride_band = 16 * K;
    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, static_cast<int>(args.strideA));
    auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb_phys<T>(lane_id, wave_id, stride_band);
    auto u_sb = make_layout_sb_phys<T>(wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb_phys<T>(lane_id, wave_id_n);

    {   // ---- exactly one tile per WG (non-persistent). KIND is a template parameter: this body is instantiated separately for NORMAL / IQ / IK. ----
        // INSIDE the K loop only `row` and the two raw column origins raw0[2] (B-load addressing) are live; packed columns, destinations and slab pointers are resolved AFTER the loop from xi.
        const int chunk = xi.chunk;
        const int row = chunk * T::B_M;
        const int raw0[2] = {tile_raw<KIND>(args, xi, 0), tile_raw<KIND>(args, xi, 1)};
        auto g_a = make_gmem(reinterpret_cast<const fp8_t*>(args.A) + static_cast<long long>(row) * args.strideA,
                             static_cast<unsigned int>(static_cast<unsigned long long>(M - row) * args.strideA));
        auto g_b = make_gmem(reinterpret_cast<const fp8_t*>(args.W), static_cast<unsigned int>(args.N) * static_cast<unsigned int>(K));
        typename MMA::vtype_a v_a;
        typename MMA::vtype_b v_b[2];
        typename MMA::vtype_c v_c[2][2];
        clear(v_c[0][0]); clear(v_c[0][1]); clear(v_c[1][0]); clear(v_c[1][1]);

        auto a_offset = [&](int hm, int tile_k) { return hm * T::HALF_B_M * static_cast<int>(args.strideA) + tile_k * T::B_K; };
        // B global element offset of half tile hn: band0*16K + tile_k*2048 (band0 = raw_col/16)
        auto b_offset = [&](int hn, int tile_k) { return (raw0[hn] / 16) * stride_band + tile_k * T::BAND_BYTES_PER_KTILE; };

        const int loops = K / T::B_K;                    // 48, even, >= 2
        int tic = 0, toc = 1;

        async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, 0));
        async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, 0));
        async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, 0));
        async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, 0));

        if (wave_id_m == 1) __builtin_amdgcn_s_barrier();
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, 1));
        async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, 1));
        async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, 1));

        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
        __builtin_amdgcn_s_barrier();

        for (int tile = 0; tile < loops - 2; tile += 2) {
            v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
            async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 1));
            s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma_fp8(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
            async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, tile + 2));
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma_fp8(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
            async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, tile + 2));
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma_fp8(v_a, v_b[0], v_c[1][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[0] = load<T::VEC_B>(s_b[toc][0], u_rb);
            async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, tile + 2));
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            v_c[1][1] = mma_fp8(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_a = load<T::VEC_A>(s_a[toc][0], u_ra);
            async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, tile + 2));
            s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma_fp8(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[1] = load<T::VEC_B>(s_b[toc][1], u_rb);
            async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, tile + 3));
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma_fp8(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_a = load<T::VEC_A>(s_a[toc][1], u_ra);
            async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, tile + 3));
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma_fp8(v_a, v_b[0], v_c[1][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
            async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, tile + 3));
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1);
            v_c[1][1] = mma_fp8(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        {
            int tile = loops - 2;
            v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
            async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 1));
            __builtin_amdgcn_s_barrier();
            s_waitcnt_lgkmcnt(0_I);

            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma_fp8(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma_fp8(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma_fp8(v_a, v_b[0], v_c[1][0]);
            v_c[1][1] = mma_fp8(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            tic ^= 1; toc ^= 1;
        }
        {
            v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
            v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma_fp8(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma_fp8(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma_fp8(v_a, v_b[0], v_c[1][0]);
            v_c[1][1] = mma_fp8(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        if (wave_id_m == 0) __builtin_amdgcn_s_barrier();
        // ---- epilogue: scales once per tile, PTPC cast + pair store into R[rank] (self) or S_dst (remote) ----
        const float* const w_scale = args.Wscale;
        const float* const a_scale = args.Ascale;
        const int q = lane_id / 16, l16 = lane_id % 16;
        const long long slab_elems = static_cast<long long>(M) * lgq::kFS;
        auto slab_of = [&](int dd) -> bf16_t* {
            return (dd == static_cast<int>(args.rank))
                ? reinterpret_cast<bf16_t*>(ccoGetLocalPtr(args.Rwin, 0)) + static_cast<long long>(args.rank) * slab_elems   // self -> R directly
                : reinterpret_cast<bf16_t*>(ccoGetLocalPtr(args.Swin, 0)) + static_cast<long long>(dd) * slab_elems;       // remote -> S_dd
        };
        const int rk = static_cast<int>(args.rank);
        // kind-specific packed columns and destination ranks, resolved HERE (after the K loop) from xi
        const int pk0[2] = {tile_packed<KIND>(args, xi, 0), tile_packed<KIND>(args, xi, 1)};
        const int dh[2] = {tile_dest<KIND>(args, xi, 0), tile_dest<KIND>(args, xi, 1)};   // destination rank of half hn (IK: unused)
        static_for<2>([&](auto chn) {
            constexpr int hn = chn.value;
            HalfScales ws;
            static_for<T::E_N>([&](auto cen) {
                constexpr int en = cen.value;
                __builtin_memcpy(&ws.w[en], w_scale + raw0[hn] + (en * T::T_N + wave_id_n) * 16 + 4 * q, 16);
            });
            static_for<2>([&](auto chm) {
                constexpr int hm = chm.value;
                float as[T::E_M];
                static_for<T::E_M>([&](auto cem) {
                    constexpr int em = cem.value;
                    as[em] = a_scale[row + hm * T::HALF_B_M + em * (T::T_M * T::W_M) + wave_id_m * T::W_M + l16];
                });
                if constexpr (KIND == lgc::T_IK) {                         // IK: scale + ONE RNE cast + pack once, same 16 B to R[rank] + 3 peer S slabs; the hn=1 half duplicates hn=0 and is not stored
                    if constexpr (hn == 0) {
                        bf16_t* const slabs4[4] = {slab_of(rk), slab_of(lgc::peer_of(rk, 0)), slab_of(lgc::peer_of(rk, 1)), slab_of(lgc::peer_of(rk, 2))};
                        store_c_ptpc_multi(v_c[hm][hn], ws, as, slabs4, 4, row + hm * T::HALF_B_M, pk0[hn], lane_id, wave_id_m, wave_id_n);
                    }
                } else if constexpr (KIND == lgc::T_IQ) {                  // IQ_d: hn=0 -> destination d (R[rank] if d == rank, else S_d); hn=1 duplicate, not stored
                    if constexpr (hn == 0) store_c_ptpc(v_c[hm][hn], ws, as, slab_of(dh[0]), row + hm * T::HALF_B_M, pk0[hn], lane_id, wave_id_m, wave_id_n, dh[0] != rk);
                } else {                                                    // NORMAL (identical to dense r6b)
                    store_c_ptpc(v_c[hm][hn], ws, as, slab_of(dh[hn]), row + hm * T::HALF_B_M, pk0[hn], lane_id, wave_id_m, wave_id_n, dh[hn] != rk);
                }
            });
        });
        // chunk completion: remote destinations this tile wrote, as scalars
        if constexpr (KIND == lgc::T_NORMAL) {
            const int nd = (dh[0] != rk) ? 1 : 0;
            if (nd == 0) lgc::stamp_self_end(args, chunk);
            lgc::x_publish<NP, HG>(args, dh[0], 0, 0, nd, static_cast<uint32_t>(chunk));
        } else if constexpr (KIND == lgc::T_IQ) {
            const int nd = (dh[0] != rk) ? 1 : 0;                          // IQ_rank is self-only: no arrival
            if (nd == 0) lgc::stamp_self_end(args, chunk);
            lgc::x_publish<NP, HG>(args, dh[0], 0, 0, nd, static_cast<uint32_t>(chunk));
        } else {
            lgc::x_publish<NP, HG>(args, lgc::peer_of(rk, 0), lgc::peer_of(rk, 1), lgc::peer_of(rk, 2), 3, static_cast<uint32_t>(chunk));
        }
    }
}

template<int NP, int HG>
__device__ __forceinline__ void run_item(const LgCommArgsV2& args, const lgc::XItem& xi, char* smem_a, char* smem_b) {
    // uniform (block-uniform, SGPR) dispatch on the item kind: separately inlined tile bodies, the kind-specific state is NOT shared across them
    if constexpr (HG == 2) {                                          // r9 quarter-first plan (dense): quarter items in their OWN copy
        if (xi.type == lgc::T_QN) { quarter_body<NP, HG>(args, xi, smem_a, smem_b); return; }
    }
    if constexpr (HG == 1) {                                          // r7 half-first plan: half-N kinds in their own copies
        if (xi.type == lgc::T_HN) { half_body<NP, HG, lgc::T_HN>(args, xi, smem_a, smem_b); return; }
#if LG_X_INDEXER
        if (xi.type == lgc::T_HK) { half_body<NP, HG, lgc::T_HK>(args, xi, smem_a, smem_b); return; }
        if (xi.type == lgc::T_HQ) { half_body<NP, HG, lgc::T_HQ>(args, xi, smem_a, smem_b); return; }
#endif
    }
#if LG_X_INDEXER
    if constexpr (!HG) {                                              // r6x plan: IK / IQ as duplicate-half full tiles
        if (xi.type == lgc::T_IK) { tile_body<NP, HG, lgc::T_IK>(args, xi, smem_a, smem_b); return; }
        if (xi.type == lgc::T_IQ) { tile_body<NP, HG, lgc::T_IQ>(args, xi, smem_a, smem_b); return; }
    }
#endif
    tile_body<NP, HG, lgc::T_NORMAL>(args, xi, smem_a, smem_b);
}

template<int NP, int HG>
__device__ __forceinline__ void gemm_body(const LgCommArgsV2& args) {
    using namespace opus;
    using T = lgq::Q;
    __shared__ char smem_a[T::smem_a_bytes * 4];
    __shared__ char smem_b[T::smem_b_bytes * 4];
    // config guard (block-uniform, before any barrier): host/device plan must agree, else poison + uniform return (nothing is released: peers' bounded waits poison too).
    // r6x L-G (review r5x): gridDim must equal the plan (items + 1 poster block); an oversized grid would map extra blocks to chunk >= chunks and write out of bounds.
#if LG_NO_GRIDREAD == 0
    const int nblocks = static_cast<int>(opus::grid_size_x()) / 512;                 // r8g: every WG reads the AQL dispatch packet (s_load s[0:1]+0xc) before any work
#endif
    if (args.items_per_group != LG_ITEMS_PER_GROUP || args.abi_version != LG_COMM_ABI_VERSION || args.sdma_queues != 1u || args.chunks > LG_MAX_CHUNKS ||
        args.chunks != args.M / 256u || (HG != 2 && (args.chunks % static_cast<uint32_t>(NP)) != 0u) || args.reserved[0] != LG_R6_PLAN_CODE_N(NP, HG) || (HG == 2 && args.chunks != 8u) || args.reserved[1] == 0ull ||
        args.N != LG_X_N_RAW || args.F != LG_COMM_F
#if LG_NO_GRIDREAD == 0
        || nblocks != lgc::grid_of<NP, HG>(static_cast<int>(args.chunks)) + 1
#endif
        ) {
        if (opus::block_id_x() == 0 && opus::thread_id_x() == 0) lgc::set_poison(args, LGC_POISON_CONFIG);
        return;
    }
#if LG_X_POSTER_LAST
    if (static_cast<int>(opus::block_id_x()) == lgc::grid_of<NP, HG>(static_cast<int>(args.chunks))) { lgc::r6_poster_block<NP, HG>(args); return; }   // X-L hook: poster = LAST block id
    const int item_bid = static_cast<int>(opus::block_id_x());
#else
    if (opus::block_id_x() == 0) { lgc::r6_poster_block<NP, HG>(args); return; }                  // block-uniform: the poster block never reaches the item/barrier code below
    const int item_bid = static_cast<int>(opus::block_id_x()) - 1;
#endif
    lgc::stamp_entry(args);
    const lgc::XItem xi = lgc::select_item_h<NP, HG>(item_bid, static_cast<int>(args.chunks));
    run_item<NP, HG>(args, xi, smem_a, smem_b);
#if LG_NO_GRIDREAD == 1
    // r10 off-path grid check: ONE item block (item 0), thread 0, AFTER its work (nothing is on the entry path): the real dispatch-packet grid must equal the compile-time plan grid; a mismatch poisons CONFIG.
    // (A mismatching oversized grid is also rejected by the host launcher, which computes the same grid; this is the device-side backstop. Extra blocks of an oversized grid would already have run: poison is the signal.)
    if (item_bid == 0 && opus::thread_id_x() == 0 && static_cast<int>(opus::grid_size_x()) / 512 != lgc::grid_of<NP, HG>(static_cast<int>(args.chunks)) + 1) lgc::set_poison(args, LGC_POISON_CONFIG);
#endif
}
} // namespace lgq

// Entries: puts per peer n selected by ENTRY (one HSACO). 512 threads, 1 WG/CU (LDS 135168 B).
extern "C" __global__ __launch_bounds__(512, 1) void lg_r6x_gemm_n1_v1(LgCommArgsV2 args) {
#if defined(__gfx950__)
    lgq::gemm_body<1, 0>(args);
#else
    __builtin_trap();
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void lg_r6x_gemm_n2_v1(LgCommArgsV2 args) {
#if defined(__gfx950__)
    lgq::gemm_body<2, 0>(args);
#else
    __builtin_trap();
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void lg_r6x_gemm_n4_v1(LgCommArgsV2 args) {
#if defined(__gfx950__)
    lgq::gemm_body<4, 0>(args);
#else
    __builtin_trap();
#endif
}

// (3) quiet_ready: ONE WG of 256 threads, stream-after the GEMM.
extern "C" __global__ __launch_bounds__(512, 1) void lg_r7_gemm_n2h_v1(LgCommArgsV2 args) {
#if defined(__gfx950__)
    lgq::gemm_body<2, 1>(args);
#else
    __builtin_trap();
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void lg_r7_gemm_n4h_v1(LgCommArgsV2 args) {
#if defined(__gfx950__)
    lgq::gemm_body<4, 1>(args);
#else
    __builtin_trap();
#endif
}
#if !LG_X_INDEXER
extern "C" __global__ __launch_bounds__(512, 1) void lg_r9_gemm_n3q_v1(LgCommArgsV2 args) {
#if defined(__gfx950__)
    lgq::gemm_body<3, 2>(args);
#else
    __builtin_trap();
#endif
}
#else
extern "C" __global__ __launch_bounds__(512, 1) void lg_r9_gemm_n3q_v1(LgCommArgsV2 args) { __builtin_trap(); }   // quarter-first plan is dense-only: this entry traps in indexer builds (the host rejects it first)
#endif
extern "C" __global__ __launch_bounds__(256)
void lg_quiet_ready_v2(LgCommArgsV2 args) {
#if defined(__gfx950__)
    lgc::r6_quiet_ready(args);
#else
    __builtin_trap();
#endif
}
// r6b host recovery step: drain every peer queue of this rank (see lgc::r6_drain_all)
extern "C" __global__ __launch_bounds__(64)
void lg_drain_all_v2(LgCommArgsV2 args) {
#if defined(__gfx950__)
    lgc::r6_drain_all(args);
#else
    __builtin_trap();
#endif
}
#endif // __HIP_DEVICE_COMPILE__

#ifndef __HIP_DEVICE_COMPILE__
// Host pass: empty __global__ DEFINITIONS (launch stub + registration) as PR695-M4 gemm.cu l.391-396.
extern "C" __global__ __launch_bounds__(512, 1) void lg_r6x_gemm_n1_v1(LgCommArgsV2) {}
extern "C" __global__ __launch_bounds__(512, 1) void lg_r6x_gemm_n2_v1(LgCommArgsV2) {}
extern "C" __global__ __launch_bounds__(512, 1) void lg_r6x_gemm_n4_v1(LgCommArgsV2) {}
extern "C" __global__ __launch_bounds__(512, 1) void lg_r7_gemm_n2h_v1(LgCommArgsV2) {}
extern "C" __global__ __launch_bounds__(512, 1) void lg_r7_gemm_n4h_v1(LgCommArgsV2) {}
extern "C" __global__ __launch_bounds__(512, 1) void lg_r9_gemm_n3q_v1(LgCommArgsV2) {}
extern "C" __global__ __launch_bounds__(256) void lg_quiet_ready_v2(LgCommArgsV2) {}
extern "C" __global__ __launch_bounds__(64) void lg_drain_all_v2(LgCommArgsV2) {}
#endif
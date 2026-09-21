#pragma once

#include <opus/opus.hpp>
#include "defs.h"
#include "global_load.hpp"
#include <bit>

using opus::operator""_I;

namespace opus_mla_decode_a16w16_32mx1_16nx4 {
// ------------------------------------------------------------- instruction scheduling masks

constexpr int VALU_MASK       = 0x002;
constexpr int MFMA_MASK       = 0x008;
constexpr int DS_READ_MASK    = 0x100;
constexpr int VMEM_MASK       = 0x010;
constexpr int VMEM_WRITE_MASK = 0x040;

// ------------------------------------------------------- register budget and operand shapes

constexpr int Q_AGPR_BASE = 0;
template<class T> constexpr int Q_ELEMS          = T::Q_TILE_SIZE * T::D_QK_SIZE / T::WARP_SIZE;
template<class T> constexpr int Q_LOAD_INSTS     = Q_ELEMS<T> / T::VEC_Q;
template<class T> constexpr int Q_AGPR_PER_LOAD  = T::VEC_Q * sizeof(typename T::D_ATTN) / 4;

template<class T> constexpr int K_AGPR_BASE      = Q_AGPR_BASE + Q_LOAD_INSTS<T> * Q_AGPR_PER_LOAD<T>;
template<class T> constexpr int K_CHUNKS         = T::D_128B_SIZE / T::W_K;
template<class T> constexpr int K_AGPR_PER_CHUNK = T::VEC_KV * sizeof(typename T::D_ATTN) / 4;
template<class T> constexpr int K_AGPR_PER_BUF   = K_CHUNKS<T> * K_AGPR_PER_CHUNK<T>;

template<class T> constexpr int V_N_TILES        = T::D_128B_SIZE / T::W_N;
template<class T> constexpr int V_TR_PER_TILE    = T::W_N * T::W_K / T::WARP_SIZE / T::VEC_TR_V;
template<class T> constexpr int V_TR_GRP_K       = (T::WARP_SIZE / 16) / (T::W_N / (4 * T::VEC_TR_V));
template<class T> constexpr int V_TR_PER_STEP    = V_N_TILES<T> * V_TR_PER_TILE<T>;
template<class T, int I> constexpr int v_tr_issue_off = (I / V_TR_PER_TILE<T>) * T::W_N + (I % V_TR_PER_TILE<T>) * V_TR_GRP_K<T> * T::D_128B_SIZE;

constexpr int O_VGPR_BASE = 0;
constexpr int S_VGPR_BASE = 64;
constexpr int P_VGPR_BASE = 72;
template<class T> constexpr int ACC_PER_TILE     = T::W_M * T::W_N / T::WARP_SIZE;
template<class T> constexpr int P_PER_CHUNK      = T::W_M * T::W_K / T::WARP_SIZE;
template<class T> constexpr int S_TILES          = T::GEMM0_E_M * T::GEMM0_E_N;
template<class T> constexpr int O_TILES          = T::GEMM1_E_M * T::GEMM1_E_N;
template<class T> constexpr int P_CHUNKS         = T::GEMM1_E_M * T::GEMM1_E_K;
template<class T> constexpr int S_ELEMS          = S_TILES<T> * ACC_PER_TILE<T>;
template<class T> constexpr int O_ELEMS          = O_TILES<T> * ACC_PER_TILE<T>;
template<class T> constexpr int P_ELEMS          = P_CHUNKS<T> * P_PER_CHUNK<T>;

constexpr int K_DEPTH = 2;
constexpr int V_DEPTH = 2;
static_assert(V_DEPTH >= K_DEPTH);

template<class T> using v_q_t  = opus::array<opus::vector_t<typename T::D_ATTN, T::VEC_Q>, Q_LOAD_INSTS<T>>;
template<class T> using v_s_t  = opus::vector_t<typename T::D_ACC, S_ELEMS<T>>;
template<class T> using v_p_t  = opus::vector_t<typename T::D_ATTN, P_ELEMS<T>>;
template<class T> using v_o_t  = opus::vector_t<typename T::D_ACC, O_ELEMS<T>>;
template<class T> using v_ml_t = opus::vector_t<typename T::D_ACC, T::ML_ELEMS>;
template<class T> using v_k_t  = opus::vector_t<typename T::D_ATTN, T::VEC_KV>;
template<class T> using v_v_t  = opus::vector_t<typename T::D_ATTN, T::W_N * T::W_K / T::WARP_SIZE>;

template<class T, int I> constexpr int kv_sub_tile_off = I * T::NUM_WARPS * T::smem_d_rpt * T::smem_brick;
template<class T, int J> constexpr int kv_d_brick_off  = J * T::NUM_WARPS * T::smem_brick;
template<class T, int SLOT> constexpr int kv_slot_off  = SLOT * T::smem_slot_elems;

template<class T> constexpr int VEC_O2 = 2 * T::VEC_O;
template<class T> constexpr int Q_HEADS         = T::Q_TILE_SIZE * T::T_M;
template<class T> constexpr int Q_SUB_TILE_RPT  = Q_HEADS<T> / T::smem_n_sub_tile;
template<class T> constexpr int Q_ASYNC_INSTS   = Q_SUB_TILE_RPT<T> * T::smem_d_rpt;
constexpr int Q_LDS_SLOT = 1;

// ----------------------------------------------------------------------------------- layouts

template<class T>
__device__ inline auto make_layout_gq(int lane_id, int stride_q_h) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_Q;

    constexpr auto gq_shape = opus::make_tuple(
        opus::number<T::smem_n_per_wave>{},
        opus::number<T::smem_d_rpt>{},
        opus::number<threads_d>{},
        opus::number<T::VEC_Q>{});

    constexpr auto gq_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gq_shape,
        opus::unfold_x_stride(gq_dim, gq_shape, opus::tuple{stride_q_h, 1_I}),
        opus::unfold_p_coord(gq_dim, opus::tuple{lane_id / threads_d, lane_id % threads_d}));
}

template<class T>
__device__ inline auto make_layout_rq(int lane_id) {
    constexpr int heads_per_brick = T::smem_n_per_wave;
    constexpr int bricks_per_m    = T::W_M / heads_per_brick;
    constexpr int lane_groups     = T::WARP_SIZE / T::W_M;
    constexpr int ek_lo           = T::D_128B_SIZE / (lane_groups * T::VEC_Q);

    constexpr auto rq_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<bricks_per_m>{},
        opus::number<heads_per_brick>{},
        opus::number<T::smem_d_rpt>{},
        opus::number<ek_lo>{},
        opus::number<lane_groups>{},
        opus::number<T::VEC_Q>{});

    constexpr auto rq_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_m = lane_id % T::W_M;

    return opus::make_layout(
        rq_shape,
        opus::unfold_x_stride(rq_dim, rq_shape, opus::tuple{opus::number<T::smem_brick>{},
                                                            opus::number<T::D_128B_SIZE>{},
                                                            opus::number<T::NUM_WARPS * T::smem_brick>{},
                                                            1_I}),
        opus::unfold_p_coord(rq_dim, opus::tuple{lane_m / heads_per_brick, lane_m % heads_per_brick,
                                                 lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_kv_indices(int warp_id, int lane_id) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_KV;

    constexpr auto kv_indices_shape = opus::make_tuple(
        opus::number<T::smem_n_sub_tile_rpt>{},
        opus::number<T::smem_n_per_wave>{},
        opus::number<T::NUM_WARPS>{},
        1_I);

    constexpr auto kv_indices_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        kv_indices_shape,
        opus::unfold_x_stride(kv_indices_dim, kv_indices_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(kv_indices_dim, opus::tuple{lane_id / threads_d, warp_id}));
}

template<class T>
__device__ inline auto make_layout_gkv(int lane_id) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_KV;

    constexpr auto gkv_shape = opus::make_tuple(
        opus::number<T::smem_d_rpt>{},
        opus::number<threads_d>{},
        opus::number<T::VEC_KV>{});

    constexpr auto gkv_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gkv_shape,
        opus::unfold_x_stride(gkv_dim, gkv_shape, opus::tuple{opus::number<T::D_128B_SIZE>{}, 1_I}),
        opus::unfold_p_coord(gkv_dim, opus::tuple{lane_id % threads_d}));
}

template<class T>
__device__ inline auto make_layout_skv(int warp_id) {
    constexpr auto skv_shape = opus::make_tuple(
        opus::number<T::smem_d_rpt>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<T::VEC_KV>{});

    constexpr auto skv_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        skv_shape,
        opus::unfold_x_stride(skv_dim, skv_shape, opus::tuple{opus::number<T::smem_brick>{}, 1_I}),
        opus::unfold_p_coord(skv_dim, opus::tuple{warp_id}));
}

template<class T>
__device__ inline auto make_layout_rk(int warp_id, int lane_id) {
    constexpr auto rk_shape = opus::make_tuple(
        opus::number<T::NUM_WARPS>{},
        opus::number<T::smem_n_per_wave / (T::W_N / T::NUM_WARPS)>{},
        opus::number<T::W_N / T::NUM_WARPS>{},
        opus::number<T::D_128B_SIZE / T::W_K>{},
        opus::number<T::WARP_SIZE / T::W_N>{},
        opus::number<T::VEC_KV>{});

    constexpr auto rk_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_n = lane_id % T::W_N;

    return opus::make_layout(
        rk_shape,
        opus::unfold_x_stride(rk_dim, rk_shape, opus::tuple{opus::number<T::smem_brick>{}, opus::number<T::D_128B_SIZE>{}, 1_I}),
        opus::unfold_p_coord(rk_dim, opus::tuple{lane_n % T::NUM_WARPS, warp_id % T::smem_n_sub_tile_rpt,
                                                 lane_n / T::NUM_WARPS, lane_id / T::W_N}));
}

template<class T>
__device__ inline auto make_layout_rv(int lane_id) {
    constexpr int lane_per_grp = 16;
    constexpr int lane_lo = 4;
    constexpr int lane_hi = lane_per_grp / lane_lo;
    constexpr int num_grps = T::WARP_SIZE / lane_per_grp;
    constexpr int grp_n = T::W_N / (lane_lo * T::VEC_TR_V);
    constexpr int grp_k = num_grps / grp_n;

    constexpr auto rv_shape = opus::make_tuple(
        opus::number<V_N_TILES<T>>{},
        opus::number<T::smem_n_sub_tile / T::W_K>{},
        opus::number<lane_hi>{},
        opus::number<T::W_K / (lane_hi * grp_k)>{},
        opus::number<grp_k>{},
        opus::number<grp_n>{},
        opus::number<lane_lo>{},
        opus::number<T::VEC_TR_V>{});

    constexpr auto rv_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int grp_id = lane_id / lane_per_grp;
    const int lane_in_grp = lane_id % lane_per_grp;

    return opus::make_layout(
        rv_shape,
        opus::unfold_x_stride(rv_dim, rv_shape, opus::tuple{opus::number<grp_n * lane_lo * T::VEC_TR_V>{},
                                                            opus::number<T::smem_brick>{},
                                                            opus::number<T::D_128B_SIZE>{},
                                                            1_I}),
        opus::unfold_p_coord(rv_dim, opus::tuple{lane_in_grp / lane_lo, grp_id / grp_n, grp_id % grp_n, lane_in_grp % lane_lo}));
}

template<class T>
__device__ inline auto make_layout_o(int warp_id, int lane_id, int stride_o_h) {
    constexpr auto o_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_M>{},
        opus::number<T::W_M>{},
        opus::number<T::T_N>{},
        opus::number<T::GEMM1_E_N>{},
        opus::number<ACC_PER_TILE<T> / T::VEC_O>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_O>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(o_block_dim, opus::tuple{lane_id % T::W_M, warp_id, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_o_packed(int warp_id, int lane_id, int stride_o_h) {
    constexpr auto o_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_M>{},
        opus::number<T::W_M>{},
        opus::number<T::T_N>{},
        opus::number<T::GEMM1_E_N / 2>{},
        opus::number<2>{},
        opus::number<2>{},
        opus::number<VEC_O2<T>>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_g = lane_id / T::W_M;

    return opus::make_layout(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(o_block_dim, opus::tuple{lane_id % T::W_M, warp_id, lane_g % 2, lane_g / 2}));
}

// ------------------------------------------------------------------ global / lds transfers

template<class T, int SLOT, int IG, class US>
__device__ inline auto kv_lds_sub_tile_base(char* smem_kv, const US& u_skv) {
    auto s_off = opus::layout_to_offsets<T::VEC_KV>(
        u_skv + opus::number<kv_slot_off<T, SLOT> + kv_sub_tile_off<T, IG>>{});
    auto* base = reinterpret_cast<typename T::D_ATTN*>(smem_kv) + s_off[0];
    asm volatile("" : "+s"(base));
    return base;
}

template<class T, int SLOT, class US>
__device__ inline auto kv_lds_bases(char* smem_kv, const US& u_skv) {
    opus::array<typename T::D_ATTN*, T::smem_n_sub_tile_rpt> bases;
    opus::static_for<T::smem_n_sub_tile_rpt>([&](auto ig) {
        bases[ig.value] = kv_lds_sub_tile_base<T, SLOT, ig.value>(smem_kv, u_skv);
    });
    return bases;
}

template<class T, class UG, class VP>
__device__ inline auto kv_gmem_bases(const typename T::D_ATTN* p_kv, const UG& u_gkv,
                                     const VP& kv_pages, int stride_kv_page) {
    opus::array<const typename T::D_ATTN*, T::smem_n_sub_tile_rpt> bases;
    auto g_off = opus::layout_to_offsets<T::VEC_KV>(u_gkv);
    opus::static_for<T::smem_n_sub_tile_rpt>([&](auto ig) {
        bases[ig.value] = p_kv + static_cast<int64_t>(kv_pages[ig.value]) * stride_kv_page + g_off[0];
    });
    return bases;
}

template<class T, int SLOT, int IDX, class GB, class LB, class US>
__device__ inline void async_load_kv_inst(const GB& gmem_bases, const LB& lds_bases, const US& u_skv) {
    constexpr int ig = IDX / T::smem_d_rpt;
    constexpr int id = IDX % T::smem_d_rpt;
    constexpr int ELEM = (int)sizeof(typename T::D_ATTN);
    constexpr auto s_imm = opus::layout_imm_offsets_v<
        opus::remove_cvref_t<decltype(u_skv + opus::number<kv_slot_off<T, SLOT> + kv_sub_tile_off<T, ig>>{})>,
        T::VEC_KV>;
    constexpr int g_imm  = id * T::D_128B_SIZE * ELEM;
    constexpr int m0_imm = (s_imm[id] - s_imm[0]) * ELEM - g_imm;
    global_load_lds<T::VEC_KV, m0_imm, g_imm>(gmem_bases[ig], lds_bases[ig]);
}

template<class T, int SLOT, class GB, class LB, class US>
__device__ inline void async_load_kv_tile(const GB& gmem_bases, const LB& lds_bases, const US& u_skv) {
    opus::static_for<T::kv_async_load_insts>([&](auto i) {
        async_load_kv_inst<T, SLOT, i.value>(gmem_bases, lds_bases, u_skv);
    });
}

template<class T, int IMM, class SM>
__device__ inline auto tr_load_v(SM& s_base, int lane_off) {
    using D_ATTN = typename T::D_ATTN;
    static_assert(IMM >= 0 && IMM < (1 << 16));
    opus::vector_t<opus::i32_t, 2> raw;
    const opus::u32_t addr = static_cast<opus::u32_t>(
        reinterpret_cast<__UINTPTR_TYPE__>(s_base.ptr + lane_off * static_cast<int>(sizeof(D_ATTN))));
    asm volatile("ds_read_b64_tr_b16 %0, %1 offset:%2\n" : "=a"(raw) : "v"(addr), "i"(IMM) : "memory");
    return __builtin_bit_cast(opus::vector_t<D_ATTN, T::VEC_TR_V>, raw);
}

template<class T>
__device__ inline int q_lds_warp_off(int warp_id) {
    const int head_base = (T::T_M > 1) ? warp_id * T::W_M : 0;
    return (head_base / T::smem_n_sub_tile) * kv_sub_tile_off<T, 1>
         + ((head_base % T::smem_n_sub_tile) / T::smem_n_per_wave) * T::smem_brick;
}

template<class T, int IDX, class G, class LB, class UG, class US>
__device__ inline void async_load_q_inst(G& g_q, const LB& lds_bases, const UG& u_gq, const US& u_skv,
                                         int warp_id, int stride_q_h) {
    constexpr int ig = IDX / T::smem_d_rpt;
    constexpr int id = IDX % T::smem_d_rpt;
    auto g_off = opus::layout_to_offsets<T::VEC_Q>(u_gq);
    auto s_off = opus::layout_to_offsets<T::VEC_Q>(
        u_skv + opus::number<kv_slot_off<T, Q_LDS_SLOT> + kv_sub_tile_off<T, ig>>{});
    const int head_base = ig * T::smem_n_sub_tile + warp_id * T::smem_n_per_wave;
    opus::async_load<T::VEC_Q>(g_q, reinterpret_cast<void*>(lds_bases[ig] + (s_off[id] - s_off[0])),
                               g_off[id], head_base * stride_q_h);
}

template<class T, class G, class LB, class UG, class US>
__device__ inline void async_load_q_tile(G& g_q, const LB& lds_bases, const UG& u_gq, const US& u_skv,
                                         int warp_id, int stride_q_h) {
    opus::static_for<Q_ASYNC_INSTS<T>>([&](auto i) {
        async_load_q_inst<T, i.value>(g_q, lds_bases, u_gq, u_skv, warp_id, stride_q_h);
    });
}

template<class T, class UQ>
__device__ inline auto load_q_lds(char* smem_kv, int warp_off, const UQ& u_rq) {
    using namespace opus;
    auto s_q = make_smem(reinterpret_cast<typename T::D_ATTN*>(smem_kv)
                         + kv_slot_off<T, Q_LDS_SLOT> + warp_off);
    auto q_offsets = layout_to_offsets<T::VEC_Q>(u_rq);
    v_q_t<T> v_q;
    static_for<Q_LOAD_INSTS<T>>([&](auto i) {
        [[clang::amdgpu_pin_agpr(Q_AGPR_BASE + i.value * Q_AGPR_PER_LOAD<T>)]]
        v_q[i.value] = load<T::VEC_Q>(s_q, q_offsets[i.value]);
    });
    s_waitcnt_lgkmcnt(0_I);
    return v_q;
}

template<class T, class G, class VO, class UO>
__device__ inline void store_o_packed(G& g_o, const VO& v_o, const UO& u_o) {
    using namespace opus;
    using D_OUT = typename T::D_OUT;
    constexpr int VEC  = VEC_O2<T>;
    constexpr int NC   = O_ELEMS<T> / VEC;
    constexpr int U32C = VEC * (int)sizeof(D_OUT) / (int)sizeof(u32_t);

    auto o_offsets = layout_to_offsets<VEC>(u_o);
    vector_t<D_OUT, VEC> buf[2];

    auto cvt_chunk = [&](auto c) {
        buf[decltype(c)::value % 2] = cast<D_OUT>(
            slice(v_o, number<decltype(c)::value * VEC>{}, number<(decltype(c)::value + 1) * VEC>{}));
    };
    auto pack_store_chunk = [&](auto c) {
        constexpr int s = decltype(c)::value % 2;
        auto* p = reinterpret_cast<u32_t*>(&buf[s]);
        auto r0 = __builtin_amdgcn_permlane16_swap(p[0], p[2], false, true);
        auto r1 = __builtin_amdgcn_permlane16_swap(p[1], p[3], false, true);
        p[0] = r0[0]; p[2] = r0[1];
        p[1] = r1[0]; p[3] = r1[1];
        store<VEC>(g_o, buf[s], o_offsets[decltype(c)::value]);
    };

    __builtin_amdgcn_sched_barrier(0);
    cvt_chunk(number<0>{});
    static_for<NC>([&](auto c) {
        if constexpr (c.value + 1 < NC) cvt_chunk(number<c.value + 1>{});
        pack_store_chunk(c);
        __builtin_amdgcn_sched_group_barrier(VALU_MASK, U32C + 2, 0);
        __builtin_amdgcn_sched_group_barrier(VMEM_WRITE_MASK, 1, 0);
    });
    __builtin_amdgcn_sched_barrier(0);
}

// -------------------------------------------------------------------------------------- gemm

template<class T, int SLOT, int STEP, int BUF, class UK, class VK>
__device__ inline void load_k_step(char* smem_kv, int sub_tile_off, const UK& u_rk, VK& v_k) {
    using namespace opus;
    using D_ATTN = typename T::D_ATTN;

    auto s_sub     = make_smem(reinterpret_cast<D_ATTN*>(smem_kv) + kv_slot_off<T, SLOT> + sub_tile_off);
    auto k_offsets = layout_to_offsets<T::VEC_KV>(u_rk);

    static_for<K_CHUNKS<T>>([&](auto i) {
        [[clang::amdgpu_pin_agpr(K_AGPR_BASE<T> + BUF * K_AGPR_PER_BUF<T> + i.value * K_AGPR_PER_CHUNK<T>)]]
        v_k[BUF][i.value] = load<T::VEC_KV>(s_sub, k_offsets[i.value] + kv_d_brick_off<T, STEP>);
    });
}

template<class T, int SLOT, int STEP, int BUF, class UV, class VV>
__device__ inline void load_v_step(char* smem_kv, int d_brick_off, const UV& u_rv, VV& v_v) {
    using namespace opus;
    using D_ATTN = typename T::D_ATTN;
    constexpr int ns = STEP / T::GEMM1_E_K;
    constexpr int dd = STEP % T::GEMM1_E_K;

    auto s_sub     = make_smem(reinterpret_cast<D_ATTN*>(smem_kv) + kv_slot_off<T, SLOT>
                               + kv_sub_tile_off<T, ns> + d_brick_off);
    auto v_offsets = layout_to_offsets<T::VEC_TR_V>(u_rv);

    static_for<V_N_TILES<T>>([&](auto en) {
        static_for<V_TR_PER_TILE<T>>([&](auto h) {
            constexpr int i = en.value * V_TR_PER_TILE<T> + h.value;
            auto chunk = tr_load_v<T, (kv_d_brick_off<T, dd> + v_tr_issue_off<T, i>) * (int)sizeof(D_ATTN)>(s_sub, v_offsets[0]);
            set_slice(v_v[BUF][en.value], chunk, number<h.value * T::VEC_TR_V>{},
                                                 number<(h.value + 1) * T::VEC_TR_V>{});
        });
    });
}

template<class T, int SLOT, class UK, class VQ, class VS, class VK, class EM>
__device__ inline void compute_qk(char* smem_kv, int k_sub_off, const UK& u_rk,
                                  const VQ& v_q, VS& v_s, VK& v_k, const EM& issue_kv_load) {
    using namespace opus;
    using D_ATTN = typename T::D_ATTN;
    constexpr int QK_STEPS  = T::smem_d_rpt;
    constexpr int K_STEPS   = T::D_128B_SIZE / T::W_K;
    constexpr int QK_MFMA   = T::GEMM0_E_M * K_STEPS;
    constexpr int ASYNC_PER_STEP = T::kv_async_load_insts / QK_STEPS;

    auto mfma_qk = make_mfma<D_ATTN, D_ATTN, typename T::D_ACC>(
        number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{}, mfma_adaptor_swap_ab{});
    using s_tile_t = typename decltype(mfma_qk)::vtype_c;

    auto s_tiles = reinterpret_cast<s_tile_t*>(&v_s);

    static_for<QK_STEPS>([&](auto st) {
        constexpr int step = decltype(st)::value;
        constexpr int b    = step % K_DEPTH;

        constexpr int k_last      = step + K_DEPTH - 2 < QK_STEPS - 1 ? step + K_DEPTH - 2 : QK_STEPS - 1;
        constexpr int k_in_flight = (k_last - step) * K_CHUNKS<T>;
        s_waitcnt_lgkmcnt(number<k_in_flight>{});
        __builtin_amdgcn_sched_barrier(0);

        if constexpr (step + K_DEPTH - 1 < QK_STEPS) {
            load_k_step<T, SLOT, step + K_DEPTH - 1, (step + K_DEPTH - 1) % K_DEPTH>(smem_kv, k_sub_off, u_rk, v_k);
        }
        static_for<ASYNC_PER_STEP>([&](auto j) {
            issue_kv_load(number<step * ASYNC_PER_STEP + j.value>{});
        });

        static_for<T::GEMM0_E_M>([&](auto em) {
            static_for<K_STEPS>([&](auto kk) {
                constexpr int ek = step * K_STEPS + kk.value;
                [[clang::amdgpu_pin_vgpr(S_VGPR_BASE + em.value * ACC_PER_TILE<T>)]]
                s_tiles[em.value] = mfma_qk(v_q[em.value * T::GEMM0_E_K + ek], v_k[b][kk.value], s_tiles[em.value]);
            });
        });

        constexpr int QK_DS = (step + K_DEPTH - 1 < QK_STEPS) ? K_CHUNKS<T> : 0;
        constexpr int QK_DS_HEAD = QK_DS / 2;
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, QK_DS_HEAD, 0);
        static_for<QK_MFMA>([&](auto j) {
            constexpr int lo = (QK_DS - QK_DS_HEAD) * j.value / QK_MFMA;
            constexpr int hi = (QK_DS - QK_DS_HEAD) * (j.value + 1) / QK_MFMA;
            __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
            __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, hi - lo, 0);
            if constexpr (j.value == QK_MFMA - 2) {
                __builtin_amdgcn_sched_group_barrier(VMEM_MASK, ASYNC_PER_STEP, 0);
            }
        });
        __builtin_amdgcn_sched_barrier(0);
    });
}

template<class T, int SLOT, class UK, class UV, class SP, class VP, class VO, class VK, class VV>
__device__ inline void compute_pv(char* smem_kv, int k_sub_off, int v_brick_off,
                                  const UK& u_rk, const UV& u_rv, SP& s_p,
                                  VP& v_p, VO& v_o, VK& v_k, VV& v_v, int lane_id) {
    using namespace opus;
    using D_ATTN = typename T::D_ATTN;
    constexpr int PV_STEPS = T::GEMM1_E_K * T::GEMM1_E_K;
    constexpr int PV_MFMA  = T::GEMM1_E_M * V_N_TILES<T>;
    constexpr int K_PRIME_STEPS  = K_DEPTH - 1;
    constexpr int SLOT_SWAP_STEP = PV_STEPS - 1 - K_PRIME_STEPS;

    auto mfma_pv = make_mfma<D_ATTN, D_ATTN, typename T::D_ACC>(
        number<T::W_M>{}, number<T::W_N>{}, number<T::W_K>{}, mfma_adaptor_swap_ab{});
    using o_tile_t = typename decltype(mfma_pv)::vtype_c;
    using p_tile_t = typename decltype(mfma_pv)::vtype_a;

    auto p_chunks = reinterpret_cast<p_tile_t*>(&v_p);
    auto o_tiles  = reinterpret_cast<o_tile_t*>(&v_o);

    static_for<P_CHUNKS<T>>([&](auto i) {
        [[clang::amdgpu_pin_vgpr(P_VGPR_BASE + i.value * (P_PER_CHUNK<T> * (int)sizeof(D_ATTN) / 4))]]
        p_chunks[i.value] = load<P_PER_CHUNK<T>>(s_p, (i.value * T::WARP_SIZE + lane_id) * P_PER_CHUNK<T>);
    });
    static_for<V_DEPTH>([&](auto i) {
        load_v_step<T, SLOT, i.value, i.value>(smem_kv, v_brick_off, u_rv, v_v);
    });
    s_waitcnt_lgkmcnt(number<V_DEPTH * V_TR_PER_STEP<T>>{});
    __builtin_amdgcn_sched_barrier(0);

    static_for<PV_STEPS>([&](auto st) {
        constexpr int step = decltype(st)::value;
        constexpr int b    = step % V_DEPTH;
        constexpr int ns   = step / T::GEMM1_E_K;
        constexpr int dd   = step % T::GEMM1_E_K;

        if constexpr (step > SLOT_SWAP_STEP) {
            constexpr int kb = step - SLOT_SWAP_STEP - 1;
            load_k_step<T, 1 - SLOT, kb, kb>(smem_kv, k_sub_off, u_rk, v_k);
        }

        constexpr int v_last  = step + V_DEPTH - 1 < PV_STEPS - 1 ? step + V_DEPTH - 1 : PV_STEPS - 1;
        constexpr int v_ahead = v_last - step;
        if constexpr (step < SLOT_SWAP_STEP) {
            s_waitcnt_lgkmcnt(number<v_ahead * V_TR_PER_STEP<T>>{});
        }

        static_for<T::GEMM1_E_M>([&](auto em) {
            static_for<V_N_TILES<T>>([&](auto en) {
                constexpr int ot = em.value * T::GEMM1_E_N + dd * V_N_TILES<T> + en.value;
                [[clang::amdgpu_pin_vgpr(O_VGPR_BASE + ot * ACC_PER_TILE<T>)]]
                o_tiles[ot] = mfma_pv(p_chunks[em.value * T::GEMM1_E_K + ns], v_v[b][en.value], o_tiles[ot]);
            });
        });

        if constexpr (step < SLOT_SWAP_STEP) {
            if constexpr (step + V_DEPTH < PV_STEPS) {
                load_v_step<T, SLOT, step + V_DEPTH, b>(smem_kv, v_brick_off, u_rv, v_v);
            }
            static_for<PV_MFMA>([&](auto) {
                __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
                __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, V_TR_PER_STEP<T> / PV_MFMA, 0);
            });
        } else if constexpr (step == SLOT_SWAP_STEP) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        } else {
            static_for<PV_MFMA>([&](auto) {
                __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
                __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, K_CHUNKS<T> * K_PRIME_STEPS / PV_MFMA, 0);
            });
        }
        __builtin_amdgcn_sched_barrier(0);
    });
}

// ----------------------------------------------------------------------------------- softmax

template<int THR_X, int THR_Y>
__device__ inline void attn_mask_vec2_imm(opus::u32_t rel, opus::u32_t neg_inf_v,
                                          opus::u32_t& x_ref, opus::u32_t& y_ref) {
    opus::u64_t x_mask, y_mask;
    asm volatile(
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
        : "v"(x_ref), "v"(y_ref), "v"(rel),
          "n"(THR_X), "v"(neg_inf_v), "n"(THR_Y)
        : "vcc"
    );
}

template<class T, class VS>
__device__ inline void attn_mask_oob_kv_tile(VS& v_s, int valid_kv_len, int kv_tile_idx,
                                             int warp_id, opus::u32_t neg_inf_v) {
    using D_ACC = typename T::D_ACC;
    using D_ACC_X2 = opus::vector_t<D_ACC, 2>;
    using U32_X2 = opus::vector_t<opus::u32_t, 2>;

    constexpr int c_pack = ACC_PER_TILE<T>;

    const int last_valid_kv_pos = valid_kv_len - 1;
    int lane_id = opus::thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int k_pos = kv_tile_idx * T::KV_TILE_SIZE + warp_id * T::W_N + (lane_id / T::W_N) * c_pack;
    const opus::u32_t rel = static_cast<opus::u32_t>(last_valid_kv_pos - k_pos);

    opus::static_for<S_TILES<T>>([&](auto e) {
        opus::static_for<c_pack / 2>([&](auto i_pair) {
            constexpr int idx = e.value * c_pack + i_pair.value * 2;
            auto pair_bits = __builtin_bit_cast(U32_X2, opus::slice(v_s, opus::number<idx>{}, opus::number<idx + 2>{}));
            opus::u32_t x_ref = pair_bits[0];
            opus::u32_t y_ref = pair_bits[1];
            attn_mask_vec2_imm<i_pair.value * 2, i_pair.value * 2 + 1>(rel, neg_inf_v, x_ref, y_ref);
            pair_bits[0] = x_ref;
            pair_bits[1] = y_ref;
            opus::set_slice(v_s, __builtin_bit_cast(D_ACC_X2, pair_bits), opus::number<idx>{}, opus::number<idx + 2>{});
        });
    });
}

template<class T, class V>
__device__ inline typename T::D_ACC attn_tile_max(const V& s_tile) {
    using D_ACC = typename T::D_ACC;
    D_ACC row_max = opus::numeric_limits<D_ACC>::lowest();
    opus::static_for<ACC_PER_TILE<T>>([&](auto i) { row_max = max(row_max, s_tile[i.value]); });

    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    row_max = max(std::bit_cast<D_ACC>(res32.x), std::bit_cast<D_ACC>(res32.y));
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    return max(std::bit_cast<D_ACC>(res16.x), std::bit_cast<D_ACC>(res16.y));
}

template<class T, class V>
__device__ inline typename T::D_ACC attn_tile_sum(const V& s_tile) {
    using D_ACC = typename T::D_ACC;
    D_ACC row_sum = D_ACC(0.0f);
    opus::static_for<ACC_PER_TILE<T>>([&](auto i) { row_sum += s_tile[i.value]; });

    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    row_sum = std::bit_cast<D_ACC>(res32.x) + std::bit_cast<D_ACC>(res32.y);
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    return std::bit_cast<D_ACC>(res16.x) + std::bit_cast<D_ACC>(res16.y);
}

template<class T, bool IS_MAX, class SM, class VML>
__device__ inline void reduce_ml_across_waves(SM& s_m, VML& row, int warp_id, int lane_id) {
    using D_ACC = typename T::D_ACC;
    const int row_idx = lane_id % T::W_M;

    opus::static_for<T::ML_ELEMS>([&](auto e) {
        const D_ACC v = row[e.value];
        opus::store(s_m, v, (e.value * T::W_M + row_idx) * T::T_N + warp_id);
    });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    opus::static_for<T::ML_ELEMS>([&](auto e) {
        auto lanes = opus::load<T::T_N>(s_m, (e.value * T::W_M + row_idx) * T::T_N);
        D_ACC acc = lanes[0];
        opus::static_for<T::T_N - 1>([&](auto i) {
            if constexpr (IS_MAX) acc = max(acc, lanes[i.value + 1]);
            else                  acc = acc + lanes[i.value + 1];
        });
        row[e.value] = acc;
    });
}

template<class T, class V>
__device__ inline void attn_scale_sub(V& s_tile, typename T::D_ACC scale, typename T::D_ACC row_max) {
    opus::static_for<ACC_PER_TILE<T>>([&](auto i) {
        s_tile[i.value] = __builtin_fmaf(s_tile[i.value], scale, -row_max);
    });
}

template<class T, class V>
__device__ inline void attn_exp2(V& s_tile) {
    opus::static_for<ACC_PER_TILE<T>>([&](auto i) {
        s_tile[i.value] = __builtin_amdgcn_exp2f(s_tile[i.value]);
    });
}

template<class T, int E_M, class VO>
__device__ inline void scale_o_tiles(VO& v_o, typename T::D_ACC scale) {
    using namespace opus;
    using acc_tile_t = vector_t<typename T::D_ACC, ACC_PER_TILE<T>>;
    auto o_tiles = reinterpret_cast<acc_tile_t*>(&v_o);
    static_for<T::GEMM1_E_N>([&](auto i) {
        constexpr int ot = E_M * T::GEMM1_E_N + i.value;
        [[clang::amdgpu_pin_vgpr(O_VGPR_BASE + ot * ACC_PER_TILE<T>)]]
        o_tiles[ot] = o_tiles[ot] * scale;
    });
}

template<class T, class VS, class VO, class SM, class SP>
__device__ inline void softmax_tile(VS& v_s, VO& v_o, SM& s_m, SP& s_p,
                                    v_ml_t<T>& m_row, v_ml_t<T>& l_row,
                                    typename T::D_ACC temperature_scale, int warp_id, int lane_id) {
    using namespace opus;
    using D_ACC = typename T::D_ACC;
    using D_ATTN = typename T::D_ATTN;
    using s_tile_t = vector_t<D_ACC, ACC_PER_TILE<T>>;
    constexpr D_ACC RESCALE_THRESHOLD = D_ACC(8.0f);

    auto s_tiles = reinterpret_cast<s_tile_t*>(&v_s);

    v_ml_t<T> row_max;
    static_for<T::ML_ELEMS>([&](auto e) { row_max[e.value] = attn_tile_max<T>(s_tiles[e.value]); });
    reduce_ml_across_waves<T, true>(s_m, row_max, warp_id, lane_id);

    bool below = true;
    static_for<T::ML_ELEMS>([&](auto e) {
        row_max[e.value] *= temperature_scale;
        below = below && ((row_max[e.value] - m_row[e.value]) <= RESCALE_THRESHOLD);
    });
    const bool all_below = __builtin_amdgcn_ballot_w64(below) == __builtin_amdgcn_read_exec();
    static_for<T::ML_ELEMS>([&](auto e) {
        row_max[e.value] = all_below ? m_row[e.value] : max(m_row[e.value], row_max[e.value]);
    });

    static_for<T::ML_ELEMS>([&](auto e) {
        attn_scale_sub<T>(s_tiles[e.value], temperature_scale, row_max[e.value]);
        attn_exp2<T>(s_tiles[e.value]);
    });

    v_ml_t<T> row_sum;
    static_for<T::ML_ELEMS>([&](auto e) { row_sum[e.value] = attn_tile_sum<T>(s_tiles[e.value]); });

    const int p_half = warp_id % T::GEMM1_E_K;
    const int p_tile = warp_id / T::GEMM1_E_K;
    static_for<T::ML_ELEMS>([&](auto e) {
        auto p_seg = cast<D_ATTN>(s_tiles[e.value]);
        store<ACC_PER_TILE<T>>(s_p, p_seg,
            ((e.value * T::GEMM1_E_K + p_tile) * T::WARP_SIZE + lane_id) * P_PER_CHUNK<T> + p_half * ACC_PER_TILE<T>);
    });

    if (!all_below) {
        static_for<T::ML_ELEMS>([&](auto e) {
            const D_ACC rescale = __builtin_amdgcn_exp2f(m_row[e.value] - row_max[e.value]);
            m_row[e.value] = row_max[e.value];
            l_row[e.value] *= rescale;
            scale_o_tiles<T, e.value>(v_o, rescale);
        });
    }
    static_for<T::ML_ELEMS>([&](auto e) { l_row[e.value] += row_sum[e.value]; });
}

// ------------------------------------------------------------------------------------ driver

template<class Traits>
__device__ void attention_tiles(const opus_mla_decode_kargs& kargs,
                                int page_idx_begin, int valid_kv_len,
                                int tile_begin, int tile_end,
                                char* smem_kv, char* smem_ml, char* smem_p,
                                const v_q_t<Traits>& v_q,
                                v_o_t<Traits>& v_o,
                                v_ml_t<Traits>& m_row,
                                v_ml_t<Traits>& l_row,
                                typename Traits::D_ACC temperature_scale) {
    using namespace opus;
    using T = remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    if (tile_begin >= tile_end) return;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const auto* p_kv = reinterpret_cast<const D_ATTN*>(kargs.kv_ptr);
    auto g_kv_indices = make_gmem(kargs.kv_indices + page_idx_begin, valid_kv_len * sizeof(int));

    auto u_kv_indices = make_layout_kv_indices<T>(warp_id, lane_id);
    auto u_gkv        = make_layout_gkv<T>(lane_id);
    auto u_skv        = make_layout_skv<T>(warp_id);
    auto u_rk         = make_layout_rk<T>(warp_id, lane_id);
    auto u_rv         = make_layout_rv<T>(lane_id);

    const int k_sub_off   = (warp_id / T::smem_n_sub_tile_rpt) * kv_sub_tile_off<T, 1>;
    const int v_brick_off = (warp_id * T::GEMM1_E_K) * kv_d_brick_off<T, 1>;

    const u32_t neg_inf_v = std::bit_cast<u32_t>(-numeric_limits<D_ACC>::infinity());

    v_s_t<T> v_s;
    v_p_t<T> v_p;
    v_k_t<T> v_k[K_DEPTH][K_CHUNKS<T>];
    v_v_t<T> v_v[V_DEPTH][V_N_TILES<T>];

    auto load_pages = [&](int tile_idx) { return load<1>(g_kv_indices, u_kv_indices, tile_idx * T::KV_TILE_SIZE); };
    decltype(load_pages(0)) kv_pages;

    auto run_tile = [&](auto cur, int tile_idx) {
        constexpr int CUR = decltype(cur)::value;
        constexpr int NXT = 1 - CUR;

        auto s_m = make_smem(reinterpret_cast<D_ACC*>(smem_ml) + CUR * T::ML_SLOT_ELEMS);
        auto s_p = make_smem(reinterpret_cast<D_ATTN*>(smem_p) + CUR * T::P_SLOT_ELEMS);

        auto lds_bases  = kv_lds_bases<T, NXT>(smem_kv, u_skv);
        auto gmem_bases = kv_gmem_bases<T>(p_kv, u_gkv, kv_pages, kargs.stride_kv_page);
        auto issue_kv_load = [&](auto idx) {
            async_load_kv_inst<T, NXT, decltype(idx)::value>(gmem_bases, lds_bases, u_skv);
        };

        clear(v_s);
        compute_qk<T, CUR>(smem_kv, k_sub_off, u_rk, v_q, v_s, v_k, issue_kv_load);
        kv_pages = load_pages(tile_idx + 2);

        if ((tile_idx + 1) * T::KV_TILE_SIZE > valid_kv_len) {
            attn_mask_oob_kv_tile<T>(v_s, valid_kv_len, tile_idx, warp_id, neg_inf_v);
        }

        softmax_tile<T>(v_s, v_o, s_m, s_p, m_row, l_row, temperature_scale, warp_id, lane_id);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        compute_pv<T, CUR>(smem_kv, k_sub_off, v_brick_off, u_rk, u_rv, s_p, v_p, v_o, v_k, v_v, lane_id);
    };

    s_waitcnt_vmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    kv_pages = load_pages(tile_begin);
    s_waitcnt_vmcnt(0_I);
    async_load_kv_tile<T, 0>(kv_gmem_bases<T>(p_kv, u_gkv, kv_pages, kargs.stride_kv_page),
                             kv_lds_bases<T, 0>(smem_kv, u_skv), u_skv);
    kv_pages = load_pages(tile_begin + 1);
    s_waitcnt_vmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    static_for<K_DEPTH - 1>([&](auto i) {
        load_k_step<T, 0, i.value, i.value>(smem_kv, k_sub_off, u_rk, v_k);
    });

    int tile_idx = tile_begin;
    #pragma clang loop unroll(disable)
    for (; tile_idx + 1 < tile_end; tile_idx += 2) {
        run_tile(0_I, tile_idx);
        run_tile(1_I, tile_idx + 1);
    }
    if (tile_idx < tile_end) {
        run_tile(0_I, tile_idx);
    }
}

template<class Traits>
__device__ void decode_one_work(const opus_mla_decode_kargs& kargs, int qo_start, int h_block_start,
                                int page_idx_begin, int valid_kv_len,
                                int tile_begin, int tile_end, int slot,
                                char* smem_kv, char* smem_ml, char* smem_p,
                                typename Traits::D_ACC temperature_scale) {
    using namespace opus;
    using T = remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;
    using D_OUT  = typename T::D_OUT;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    const int64_t q_gmem_offset = static_cast<int64_t>(qo_start) * kargs.stride_q_b
                                + static_cast<int64_t>(h_block_start) * kargs.stride_q_h;

    auto g_q = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.q_ptr) + q_gmem_offset,
                         (size_t)(kargs.H - h_block_start) * kargs.stride_q_h * sizeof(D_ATTN));
    auto u_gq  = make_layout_gq<T>(lane_id, kargs.stride_q_h);
    auto u_skv = make_layout_skv<T>(warp_id);
    auto u_rq  = make_layout_rq<T>(lane_id);

    __builtin_amdgcn_s_barrier();
    async_load_q_tile<T>(g_q, kv_lds_bases<T, Q_LDS_SLOT>(smem_kv, u_skv), u_gq, u_skv,
                         warp_id, kargs.stride_q_h);
    s_waitcnt_vmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto v_q = load_q_lds<T>(smem_kv, q_lds_warp_off<T>(warp_id), u_rq);

    v_o_t<T> v_o;
    clear(v_o);
    v_ml_t<T> m_row, l_row;
    static_for<T::ML_ELEMS>([&](auto e) {
        m_row[e.value] = numeric_limits<D_ACC>::lowest();
        l_row[e.value] = D_ACC(0.0f);
    });

    attention_tiles<T>(kargs, page_idx_begin, valid_kv_len, tile_begin, tile_end,
                       smem_kv, smem_ml, smem_p, v_q, v_o, m_row, l_row, temperature_scale);

    auto s_l = make_smem(reinterpret_cast<D_ACC*>(smem_ml));
    __builtin_amdgcn_s_barrier();
    reduce_ml_across_waves<T, false>(s_l, l_row, warp_id, lane_id);

    v_ml_t<T> o_scale;
    static_for<T::ML_ELEMS>([&](auto e) {
        o_scale[e.value] = (l_row[e.value] > D_ACC(0.0f)) ? (D_ACC(1.0f) / l_row[e.value]) : D_ACC(0.0f);
    });
    static_for<T::ML_ELEMS>([&](auto e) { scale_o_tiles<T, e.value>(v_o, o_scale[e.value]); });

    int lane_id_o = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id_o));
    const int warp_id_o = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    if (slot < 0) {
        const int64_t o_gmem_offset = static_cast<int64_t>(qo_start) * kargs.stride_o_b
                                    + static_cast<int64_t>(h_block_start) * kargs.stride_o_h;
        auto g_o = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_gmem_offset,
                             (size_t)(kargs.H - h_block_start) * kargs.stride_o_h * sizeof(D_OUT));
        auto u_o = make_layout_o_packed<T>(warp_id_o, lane_id_o, kargs.stride_o_h);
        store_o_packed<T>(g_o, v_o, u_o);

        if (kargs.lse_ptr != nullptr && warp_id_o == 0 && lane_id_o < T::W_M) {
            const int64_t lse_offset = static_cast<int64_t>(qo_start) * kargs.H + h_block_start;
            auto g_lse = make_gmem(reinterpret_cast<D_ACC*>(kargs.lse_ptr) + lse_offset,
                                   (size_t)(kargs.H - h_block_start) * sizeof(D_ACC));
            static_for<T::ML_ELEMS>([&](auto e) {
                const D_ACC lse = (l_row[e.value] > D_ACC(0.0f))
                                ? (m_row[e.value] + log2f(l_row[e.value])) * D_ACC(MLA_DECODE_LN_2)
                                : numeric_limits<D_ACC>::infinity();
                g_lse.store(lse, e.value * T::W_M + lane_id_o);
            });
        }
    } else {
        const int64_t oa_offset = (static_cast<int64_t>(slot) * kargs.H + h_block_start) * T::D_VO_SIZE;
        auto g_oa = make_gmem(reinterpret_cast<D_ACC*>(kargs.o_accum) + oa_offset,
                              (size_t)(kargs.H - h_block_start) * T::D_VO_SIZE * sizeof(D_ACC));
        auto u_oa = make_layout_o<T>(warp_id_o, lane_id_o, T::D_VO_SIZE);
        store<T::VEC_O>(g_oa, v_o, u_oa);

        if (warp_id_o == 0 && lane_id_o < T::W_M) {
            const int64_t lse_offset = static_cast<int64_t>(slot) * kargs.H + h_block_start;
            auto g_lse = make_gmem(reinterpret_cast<D_ACC*>(kargs.lse_accum) + lse_offset,
                                   (size_t)(kargs.H - h_block_start) * sizeof(D_ACC));
            static_for<T::ML_ELEMS>([&](auto e) {
                const D_ACC lse = (l_row[e.value] > D_ACC(0.0f))
                                ? (m_row[e.value] + log2f(l_row[e.value])) * D_ACC(MLA_DECODE_LN_2)
                                : numeric_limits<D_ACC>::lowest();
                g_lse.store(lse, e.value * T::W_M + lane_id_o);
            });
        }
    }
}

}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void opus_mla_decode_a16w16_32mx1_16nx4_kernel(opus_mla_decode_kargs kargs) {
    using namespace opus;
    using namespace opus_mla_decode_a16w16_32mx1_16nx4;
    using T = opus::remove_cvref_t<Traits>;

    const int part = block_id_x();
    const int h_block_start = block_id_y() * T::Q_TILE_SIZE;

    __shared__ char smem[T::smem_bytes()];
    char* smem_kv = smem;
    char* smem_ml = smem + T::NUM_KV_BUFS * T::smem_kv_bytes;
    char* smem_p  = smem_ml + T::smem_ml_bytes;

    constexpr float LOG2_E = 1.44269504089f;
    const float temperature_scale = kargs.softmax_scale * LOG2_E;

    const int work_begin = __builtin_amdgcn_readfirstlane(kargs.work_indptr[part]);
    const int work_end   = __builtin_amdgcn_readfirstlane(kargs.work_indptr[part + 1]);

    for (int w = work_begin; w < work_end; ++w) {
        const opus_mla_decode_work_info info = kargs.work_info_set[w];
        const int slot         = __builtin_amdgcn_readfirstlane(info.partial_slot);
        const int qo_start     = __builtin_amdgcn_readfirstlane(info.qo_start);
        const int kv_start     = __builtin_amdgcn_readfirstlane(info.kv_start);
        const int valid_kv_len = __builtin_amdgcn_readfirstlane(info.kv_end) - kv_start;

        decode_one_work<T>(kargs, qo_start, h_block_start, kv_start, valid_kv_len,
                           0, ceil_div(valid_kv_len, T::KV_TILE_SIZE), slot,
                           smem_kv, smem_ml, smem_p, temperature_scale);
    }
}

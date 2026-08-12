// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 fp8/e8m0 flatmm split-K pipeline for decode-oriented BMM.
//
// This is the first B_K=128 version: one K iteration maps to one DSv4
// checkpoint scale block, and one consumer-wave M tile maps to one per-row A
// scale. The host launcher keeps this v1 on divisible decode shapes.
#pragma once

#include "opus_gemm_traits_a8w8_scale_gfx950.cuh"
// opus_bmm_splitk_reduce_kernel: the split-K > 1 path's fp32->Y reduce. Lives
// in the shared reduce header so the codegen'd BMM launcher (which #includes
// this pipeline header on the non-fused device pass) can launch it.
#include "splitk_reduce_gfx950.cuh"

#ifdef __HIP_DEVICE_COMPILE__

// ============================================================================
// Layout helpers. Suffixed with _mxsk to avoid ODR collisions with the bf16
// flatmm splitK helpers when both headers are included in a build.
// ============================================================================

template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_group_load_mxsk(int lane_id, int wave_id, int stride) {
    constexpr int threads_k = T::LOAD_GROUP_K / T::VEC_A;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_m = threads_m_per_wave / T::LOAD_GROUP_M_LANE;
    constexpr int repeat_m = T::slots / WAVES;

    constexpr auto g_block_shape = opus::make_tuple(
        opus::number<interlanegroup_m>{},
        opus::number<repeat_m>{},
        opus::number<WAVES>{},
        opus::number<T::LOAD_GROUP_M_LANE>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto g_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        g_block_shape,
        opus::unfold_x_stride(g_block_dim, g_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(g_block_dim,
            opus::tuple{lane_id / threads_k / T::LOAD_GROUP_M_LANE,
                        wave_id % WAVES,
                        (lane_id / threads_k) % T::LOAD_GROUP_M_LANE,
                        lane_id % threads_k}));
}

// B global-load layout for a 16x16-preshuffled weight buffer
// (shuffle_weight(w, layout=(16, 16))), which stores the logical [N, K] plane
// as [N/16][K/32][2][16 n][16 k]. That is a pure address permutation of the
// bytes each thread already loads, so this keeps the thread -> (n, k) mapping
// of make_layout_gmem_group_load_mxsk -- LDS content, u_sb and every
// consumer-side layout stay bit-identical -- and only rewrites the address:
//
//   addr(n, k) = (n/16)*K*16 + (k/32)*512 + ((k%32)/16)*256 + (n%16)*16 + k%16
//
// Only the per-issue (y) dims can carry a stride here. The repeat dim advances
// n by WAVES*LOAD_GROUP_N_LANE while staying inside one 16-row block (see the
// static_assert), so its stride is that times the 16-byte block row; the vector
// dim is the 16 contiguous k bytes of one block row. The per-thread (p) part is
// not linear -- n/16 and n%16 both move -- so it is folded in as a scalar.
template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_group_load_b_preshuffle_mxsk(int lane_id, int wave_id, int stride_b) {
    constexpr int threads_k = T::LOAD_GROUP_K / T::VEC_B;
    constexpr int threads_n_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_n = threads_n_per_wave / T::LOAD_GROUP_N_LANE;
    constexpr int repeat_n = T::slots / WAVES;
    // n span of one lane group, i.e. the n range a single lane walks with y.
    constexpr int n_per_lane_group = repeat_n * WAVES * T::LOAD_GROUP_N_LANE;
    static_assert(n_per_lane_group <= 16 && 16 % n_per_lane_group == 0,
                  "one lane's n range must stay inside a single 16-row preshuffle block");
    static_assert(interlanegroup_n * n_per_lane_group == T::LOAD_GROUP_N);
    constexpr int repeat_n_stride = WAVES * T::LOAD_GROUP_N_LANE * 16;

    constexpr auto g_block_shape = opus::make_tuple(
        opus::number<1>{},
        opus::number<repeat_n>{},
        opus::number<T::VEC_B>{});

    constexpr auto g_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    const int lane_n = lane_id / threads_k;
    const int lane_k = lane_id % threads_k;
    // n of this thread's first issue, relative to the load group's first row
    // (same expression as the row index in make_layout_gmem_group_load_mxsk).
    const int n0 = (lane_n / T::LOAD_GROUP_N_LANE) * n_per_lane_group
                 + (wave_id % WAVES) * T::LOAD_GROUP_N_LANE
                 + lane_n % T::LOAD_GROUP_N_LANE;

    auto u_gb = opus::make_layout<0>(
        g_block_shape,
        opus::unfold_x_stride(g_block_dim, g_block_shape,
            opus::tuple{opus::number<repeat_n_stride>{}, 1_I}),
        opus::unfold_p_coord(g_block_dim, opus::tuple{0}));
    u_gb += (n0 / 16) * stride_b * 16 + (n0 % 16) * 16
          + (lane_k / 2) * 512 + (lane_k % 2) * 256;
    return u_gb;
}

// Chunk grid of one B load group in the preshuffled buffer: it splits into
// LOAD_GROUP_N/16 n blocks, each holding LOAD_GROUP_K*16 contiguous bytes, and
// one wave issue covers warp_size*VEC_B of those bytes.
template<typename T>
inline constexpr int b_preshuffle_chunk_bytes_mxsk = opus::get_warp_size() * T::VEC_B;
template<typename T>
inline constexpr int b_preshuffle_k_chunks_mxsk =
    T::LOAD_GROUP_K * 16 / b_preshuffle_chunk_bytes_mxsk<T>;
template<typename T>
inline constexpr int b_preshuffle_n_blocks_mxsk = T::LOAD_GROUP_N / 16;

// Keeping the row-major thread -> (n, k) mapping (above) costs L1 read tag
// conflicts: a wave issue takes 16 B out of each of 16 cache lines instead of
// filling 8 of them, so the same bytes cost 4x the line touches per
// instruction. Measured on kid325 -> kid177 at n=1024,k=4096,g=16,m=256:
// identical instruction mix, VGPR/LDS/occupancy, TA cycles and L2 traffic (L2
// hit rate 31.6% either way), but TCP_READ_TAGCONFLICT_STALL_CYCLES goes
// 0 -> 1.05M and TA_ADDR_STALLED_BY_TC_CYCLES +166%, for -14% end to end.
//
// So stage the preshuffle order verbatim instead: each issue becomes one
// contiguous warp_size*VEC_B run, the same shape B_DIRECT_REG already fetches.
// LDS then holds B in preshuffle rather than row-major order, which the
// matching consumer layout absorbs.
//
// Gated on !ALL_WAVE so the producer's WAVES is 1 or 2 and the wave index always
// takes the innermost k chunks; ALL_WAVE stages with four waves, which both
// layouts below assume away.
//
// tileN (T_N=2 without ALL_WAVE) is in, and had to be added: the B_M=16 tiles are
// all tileN, so while this also required T_N==1 they were the one part of the
// family reading preshuffled bytes through the row-major mapping, at 8-93% behind
// their own plain kids. Nothing here is actually T_N-dependent. The producer
// layout describes one load group and the group index arrives through
// b_gmem_group_offset_mxsk either way; a tileN consumer owns a contiguous run of
// COM_REP_N*W_N/LOAD_GROUP_N groups from nbc, which is the same base and the same
// n-group stride the staged layout reads from. With LOAD_GROUP_N == W_N there,
// n_blocks and tiles_per_block_n are both 1, so the two layouts also enumerate n
// subtiles in the same order and the scale-side indexing by subtile is unaffected.
template<typename T>
inline constexpr bool b_preshuffle_contig_mxsk =
    T::B_PRESHUFFLE && !T::B_DIRECT_REG && !T::ALL_WAVE
    && T::VEC_B == 16 && T::LOAD_GROUP_N % 16 == 0
    && (T::LOAD_GROUP_K * 16) % b_preshuffle_chunk_bytes_mxsk<T> == 0
    && b_preshuffle_k_chunks_mxsk<T> % 2 == 0
    && b_preshuffle_n_blocks_mxsk<T> * b_preshuffle_k_chunks_mxsk<T> == T::slots
    && T::COM_REP_N % b_preshuffle_n_blocks_mxsk<T> == 0;

// Contiguous-issue B global layout. One issue is chunk c = repeat*WAVES + wave
// of the load group, which is the chunk the smem layout already puts at row c,
// so the staged bytes do not depend on how many waves stage them:
//
//   c    -> n block c / k_chunks, k chunk c % k_chunks
//   addr = (c / k_chunks) * stride_b * 16 + (c % k_chunks) * chunk_bytes
//          + lane * VEC_B
//
// With WAVES dividing k_chunks that separates into a y dim over n blocks, a y
// dim over the wave's own k chunks, and a per-thread scalar. The wave's 64
// lanes then cover chunk_bytes contiguous bytes -- one full cache line per 8
// lanes, which is what removes the tag conflicts.
template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_group_load_b_preshuffle_contig_mxsk(
        int lane_id, int wave_id, int stride_b) {
    constexpr int chunk_bytes = b_preshuffle_chunk_bytes_mxsk<T>;
    constexpr int k_chunks    = b_preshuffle_k_chunks_mxsk<T>;
    constexpr int n_blocks    = b_preshuffle_n_blocks_mxsk<T>;
    static_assert(k_chunks % WAVES == 0,
                  "the staging waves must divide the load group's k chunks");
    constexpr int k_chunks_per_wave = k_chunks / WAVES;
    static_assert(n_blocks * k_chunks_per_wave == T::slots / WAVES,
                  "one wave's issues must cover its share of the load group");

    constexpr auto g_block_shape = opus::make_tuple(
        opus::number<1>{},
        opus::number<n_blocks>{},
        opus::number<k_chunks_per_wave>{},
        opus::number<T::VEC_B>{});

    constexpr auto g_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    auto u_gb = opus::make_layout<0>(
        g_block_shape,
        opus::unfold_x_stride(g_block_dim, g_block_shape,
            opus::tuple{stride_b * 16,
                        opus::number<WAVES * chunk_bytes>{},
                        1_I}),
        opus::unfold_p_coord(g_block_dim, opus::tuple{0}));
    u_gb += (wave_id % WAVES) * chunk_bytes + lane_id * T::VEC_B;
    return u_gb;
}

// B-source dispatchers: plain row-major [N, K] vs the 16x16 preshuffle. Address
// math only -- the A side and everything downstream of LDS is untouched.
template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_b_mxsk(int lane_id, int wave_id, int stride_b) {
    if constexpr (b_preshuffle_contig_mxsk<T>)
        return make_layout_gmem_group_load_b_preshuffle_contig_mxsk<T, WAVES>(lane_id, wave_id, stride_b);
    else if constexpr (T::B_PRESHUFFLE)
        return make_layout_gmem_group_load_b_preshuffle_mxsk<T, WAVES>(lane_id, wave_id, stride_b);
    else
        return make_layout_gmem_group_load_mxsk<T, WAVES>(lane_id, wave_id, stride_b);
}

// Element offset of the WG's B tile origin (column col, K offset k_start).
// col is a multiple of B_N (hence of 16), so the preshuffled n-block term
// (col/16)*K*16 is exactly col*stride_b and only the K term differs.
template<typename T>
inline __device__ size_t b_gmem_tile_base_mxsk(int col, int k_start, int stride_b) {
    const size_t k_term = T::B_PRESHUFFLE ? (size_t)k_start * 16 : (size_t)k_start;
    return (size_t)col * (size_t)stride_b + k_term;
}

// Consumer-side B layout for T::B_DIRECT_REG: the whole register B tile of one
// K iteration, read straight from the preshuffled weight into MFMA registers.
//
// The tiled mma wants v_b indexed as ((n_rep * COM_REP_K + k_rep) * 2 + half)
// * VEC_B + byte (see tiled_mma_adaptor::shape_b -- y dims expd_n, expd_k,
// rept_b, pack_b). In the preshuffle those map to a plain nested address:
//
//   n_rep -> a new 16-column block, i.e. stride_b*16 bytes apart
//   k_rep -> W_K=128 k of one block   = 2048 bytes
//   half  -> the fragment's 64-k half = 1024 bytes
//   lane  -> (lane/16)*256 + (lane%16)*16 == lane*16
//
// so the k side is one row-major chain (2048, 1024, 256, 16, 1) and only the
// n-rep needs the runtime row stride. Every issue is a 16B-aligned dwordx4 and
// the wave's 64 lanes cover 1024 contiguous bytes.
//
// nbc is the wave's first 16-column block within the tile, mirroring what
// smem_b_at(slot, nbc, 0) does on the LDS path: 0 for tileM (both consumers
// share the N range), wave_id_n_cons * COM_REP_N for tileN (each consumer owns
// its own columns). T_N never reaches the register shape -- it partitions
// across waves, which is exactly this base offset.
template<typename T>
inline __device__ auto make_layout_gmem_b_direct_mxsk(int lane_id, int stride_b, int nbc) {
    constexpr int grpk_b = opus::get_warp_size() / T::W_N;
    static_assert(grpk_b * T::W_N * T::VEC_B * 2 == T::W_N * T::W_K,
                  "one 16x16x128 B fragment is 2 halves x 64 lanes x VEC_B bytes");

    constexpr auto b_block_shape = opus::make_tuple(
        opus::number<T::COM_REP_N>{},
        opus::number<T::COM_REP_K>{},
        opus::number<2>{},
        opus::number<grpk_b>{},
        opus::number<T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto b_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto u_gb = opus::make_layout<0>(
        b_block_shape,
        opus::unfold_x_stride(b_block_dim, b_block_shape,
            opus::tuple{stride_b * 16, 1_I}),
        opus::unfold_p_coord(b_block_dim,
            opus::tuple{lane_id / T::W_N, lane_id % T::W_N}));
    u_gb += nbc * stride_b * 16;
    return u_gb;
}

// K-iteration stride of the direct-B tile: B_K columns of one 16-row block.
template<typename T>
inline __device__ int b_direct_iter_offset_mxsk(int loop_k_idx) {
    return loop_k_idx * T::B_K * 16;
}

// An L2/XCD-aware workgroup -> tile mapping was built here and measured; it is
// not kept because it made things slower. For the record, since the memory-side
// result was the opposite of the runtime one: undoing the dispatcher's
// round-robin over the eight XCDs (each XCD taking a contiguous run of logical
// tiles) and then walking those tiles in bands of 4 row tiles x every column
// tile -- so the co-resident workgroups share A rows as well as B columns --
// took kid188's L2 hit rate from 70.1% to 84.3% and its HBM traffic from
// 10.25 GB to 5.39 GB at g16 n1024 k4096 M=32768, better on both counts than
// FlyDSL's 81.2% / 7.48 GB on the same shape. It ran 6% *slower* (3818us against
// 3599us). kid188 moves 10 GB in 3.6 ms, i.e. under 3 TB/s against a ~8 TB/s
// part, so it was never bandwidth-bound and halving the traffic buys it nothing.
// See the kid 188 notes in opus_gemm_common.py.
//
// That last part is about kid188, not about the mapping: read it as "a pipeline
// this far from the matrix pipe cannot spend a cheaper memory system", because
// the same mapping is worth 5-10% to kid205, whose only difference is a schedule
// good enough to feel it (48% MfmaUtil against kid188's 24%). It lives in the
// wave8 pipeline as XCD_WGM; see _BMM_MXSCALE_BPRESHUFFLE_WAVETM1_XCD_TILES.

// Per-(K iter, n group, k group) offset from that origin.
template<typename T>
inline __device__ int b_gmem_group_offset_mxsk(int loop_k_idx, int group_load_idx, int k_group, int stride_b) {
    const int k_off = (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    return group_load_idx * T::LOAD_GROUP_N * stride_b + (T::B_PRESHUFFLE ? k_off * 16 : k_off);
}

template<typename T, int WAVES>
inline __device__ auto make_layout_smem_group_load_mxsk(int lane_id, int wave_id) {
    constexpr int repeat_m = T::slots / WAVES;

    constexpr auto s_block_shape = opus::make_tuple(
        opus::number<repeat_m>{},
        opus::number<WAVES>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto s_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        s_block_shape,
        opus::unfold_x_stride(s_block_dim, s_block_shape,
            opus::tuple{T::smem_linear_wave_per_async_load + T::smem_padding, 1_I}),
        opus::unfold_p_coord(s_block_dim, opus::tuple{wave_id % WAVES, lane_id}));
}

// Element step from one M subtile of the A fragment to the next: a subtile is a
// whole A load group in LDS, since T_M waves split one group's rows.
template<typename T>
inline constexpr int ra_m_block_stride_mxsk =
    T::NUM_LOAD_GROUPS_PER_BK * T::slots
    * (T::smem_linear_wave_per_async_load + T::smem_padding);

// M_REPS is how many of the wave's M subtiles the returned layout covers. It is
// the whole fragment by default; a caller that cannot afford all COM_REP_M of
// them in registers asks for a group at a time and walks the groups with an
// element offset of M_REPS * ra_m_block_stride_mxsk<T>.
template<typename T, int M_REPS = T::COM_REP_M>
inline __device__ auto make_layout_ra_mxsk(int lane_id, int wave_id_m) {
    constexpr int threads_k = opus::get_warp_size() / T::W_M;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_m = threads_m_per_wave / T::LOAD_GROUP_M_LANE;
    constexpr int per_block_load = T::slots * (T::smem_linear_wave_per_async_load + T::smem_padding);
    constexpr int m_block_stride = ra_m_block_stride_mxsk<T>;

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<M_REPS>{},
        opus::number<T::slots>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<T::T_M>{},
        opus::number<interlanegroup_m / T::slots>{},
        opus::number<T::LOAD_GROUP_M_LANE>{},
        opus::number<2>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{},
                         opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_m = lane_id % T::W_M;

    return opus::make_layout<0>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<m_block_stride>{},
                        opus::number<T::smem_linear_wave_per_async_load + T::smem_padding>{},
                        opus::number<per_block_load>{},
                        1_I}),
        opus::unfold_p_coord(ra_block_dim,
            opus::tuple{lane_id_m % T::slots,
                        wave_id_m,
                        lane_id_m / T::slots,
                        lane_id_m % T::LOAD_GROUP_M_LANE,
                        lane_id / T::W_M}));
}

// Consumer B layout for row-major-ordered staging, i.e. whenever the producer
// kept the row-major thread -> (n, k) mapping (plain B, or preshuffled B that
// b_preshuffle_contig_mxsk could not re-map).
template<typename T>
inline __device__ auto make_layout_rb_staged_mxsk(int lane_id) {
    constexpr int grpk_b = opus::get_warp_size() / T::W_N;
    constexpr int interlanegroup_n = T::W_N / T::LOAD_GROUP_N_LANE;
    constexpr int loops_b = interlanegroup_n / T::slots;
    constexpr int tiles_per_block_n = T::LOAD_GROUP_N / T::W_N;
    constexpr int num_blocks_n = T::COM_REP_N / tiles_per_block_n;
    constexpr int per_block_load = T::slots * (T::smem_linear_wave_per_async_load + T::smem_padding);
    constexpr int n_block_stride = T::NUM_LOAD_GROUPS_PER_BK * per_block_load;
    constexpr int n_intra_stride = T::LOAD_GROUP_N_LANE * 2 * grpk_b * T::VEC_B;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<num_blocks_n>{},
        opus::number<T::slots>{},
        opus::number<tiles_per_block_n>{},
        opus::number<loops_b>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<T::LOAD_GROUP_N_LANE>{},
        opus::number<2>{},
        opus::number<grpk_b>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_n = lane_id % T::W_N;

    return opus::make_layout<0>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<n_block_stride>{},
                        opus::number<T::smem_linear_wave_per_async_load + T::smem_padding>{},
                        opus::number<n_intra_stride>{},
                        opus::number<per_block_load>{},
                        1_I}),
        opus::unfold_p_coord(rb_block_dim,
            opus::tuple{lane_id_n % T::slots,
                        lane_id_n / T::slots,
                        lane_id_n % T::LOAD_GROUP_N_LANE,
                        lane_id / T::W_N}));
}

// Consumer B layout for the contiguous-issue staging. LDS holds the preshuffle
// order verbatim, and that order already IS the mfma_16x16x128 B fragment order
// (see make_layout_gmem_b_direct_mxsk: lane -> (lane/16)*256 + (lane%16)*16 ==
// lane*16), so one fragment is a flat lane*VEC_B read of one chunk and the
// whole layout is four nested strides over the staged chunks:
//
//   n group -> NUM_LOAD_GROUPS_PER_BK * per_block_load   (smem_b_at's n stride)
//   n block -> k_chunks * slot                           (chunks of one group)
//   k group -> per_block_load                            (smem_b_at's kg stride)
//   k chunk -> slot                                      (the mma's rept_b)
//
// The y order is (n group, n block, k group, k chunk) so the issues arrive in
// the tiled mma's (expd_n, expd_k, rept_b) order, with the n subtile index
// in == n_group * n_blocks + n_block.
template<typename T>
inline __device__ auto make_layout_rb_preshuffle_contig_mxsk(int lane_id) {
    constexpr int slot_stride    = T::smem_linear_wave_per_async_load + T::smem_padding;
    constexpr int per_block_load = T::slots * slot_stride;
    constexpr int k_chunks       = b_preshuffle_k_chunks_mxsk<T>;
    constexpr int n_blocks       = b_preshuffle_n_blocks_mxsk<T>;
    constexpr int num_groups_n   = T::COM_REP_N / n_blocks;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<num_groups_n>{},
        opus::number<n_blocks>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<k_chunks>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<T::NUM_LOAD_GROUPS_PER_BK * per_block_load>{},
                        opus::number<k_chunks * slot_stride>{},
                        opus::number<per_block_load>{},
                        opus::number<slot_stride>{},
                        1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{lane_id}));
}

template<typename T>
inline __device__ auto make_layout_rb_mxsk(int lane_id) {
    if constexpr (b_preshuffle_contig_mxsk<T>)
        return make_layout_rb_preshuffle_contig_mxsk<T>(lane_id);
    else
        return make_layout_rb_staged_mxsk<T>(lane_id);
}

template<typename T>
inline __device__ auto make_layout_sfa_mxsk(int lane_id, int wave_id_m, int stride_sfa) {
    constexpr auto sfa_block_shape = opus::make_tuple(
        opus::number<T::COM_REP_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::B_K / T::GROUP_K>{});

    constexpr auto sfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sfa_block_shape,
        opus::unfold_x_stride(sfa_block_dim, sfa_block_shape,
            opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(sfa_block_dim,
            opus::tuple{wave_id_m, lane_id % T::W_M}));
}

// pack_e8m0x4 (broadcast e8m0 -> x4 word) is shared via opus_gemm_utils.cuh.

// Per-subtile scaled-MFMA loop -- the shared "else" body used whenever the
// register tile spans more than one MX scale group. The MMA issue pattern is
// identical for every scale layout; only where each subtile's packed scale comes
// from differs, so that is injected via scale_a_of(im, ik) / scale_b_of(in, ik).
// Providers receive compile-time (opus::number<>) subtile indices and return the
// packed-int32 e8m0x4 scale. This lets plain block-scale and shuffled/preloaded
// block-scale reuse one implementation.
// OPSEL == false (default): scale_a_of/scale_b_of return a broadcast-packed x4
//   e8m0 word (all 4 bytes equal) and every MFMA selects byte 0 -- legacy
//   scalar behavior.
// OPSEL == true: scale_a_of/scale_b_of return a K-packed word holding the
//   COM_REP_K distinct K-group e8m0 bytes (byte ik == the ik-th K group) and are
//   K-independent; each MFMA selects its own byte through the compile-time
//   scale_op_sel == ik. This drops the per-subtile broadcast pack and shrinks the
//   K-direction scale register footprint to one word per M / N-scale group.
// Which byte of the packed scale word each MFMA selects, as a compile-time
// immediate. opsel_from_k is the K-packed word every OPSEL kid predating the shuffle_scale
// layout uses: byte ik holds K group ik, on both operands. opsel_shuf is the shuffle_scale
// layout's word, which spends its low bit on the M (resp. N) subtile parity and
// its high bit on K, because one dword there holds two subtiles crossed with two
// K blocks. Both are pure compile-time index arithmetic; nothing is emitted.
struct opsel_from_k {
    template<int IM, int IK> static OPUS_D auto a() { return opus::number<IK>{}; }
    template<int IN, int IK> static OPUS_D auto b() { return opus::number<IK>{}; }
};

struct opsel_shuf {
    template<int IM, int IK> static OPUS_D auto a() {
        return opus::number<(IK << 1) | (IM & 1)>{};
    }
    template<int IN, int IK> static OPUS_D auto b() {
        return opus::number<(IK << 1) | (IN & 1)>{};
    }
};

template<typename T, typename Mma, bool OPSEL = false, typename OpSel = opsel_from_k,
         typename VA, typename VB, typename VC,
         typename ScaleAOf, typename ScaleBOf>
OPUS_D void mma_mxscale_subtile_loop(const VA& v_a, const VB& v_b, VC& v_c,
                                     ScaleAOf&& scale_a_of, ScaleBOf&& scale_b_of) {
    using MMA = typename Mma::MMA;
    constexpr int a_len = Mma::mma_a_len;
    constexpr int b_len = Mma::mma_b_len;
    constexpr int c_len = Mma::mma_c_len;
    opus::static_for<T::COM_REP_M>([&](auto im_c) {
        constexpr int im = decltype(im_c)::value;
        opus::static_for<T::COM_REP_N>([&](auto in_c) {
            constexpr int in = decltype(in_c)::value;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                const int scale_a = scale_a_of(im_c, ik_c);
                const int scale_b = scale_b_of(in_c, ik_c);
                constexpr int i_tile_a = (im * T::COM_REP_K + ik);
                constexpr int i_tile_b = (in * T::COM_REP_K + ik);
                constexpr int i_tile_c = im * T::COM_REP_N + in;
                auto s_a = opus::slice(v_a,
                    opus::number<i_tile_a * a_len>{},
                    opus::number<i_tile_a * a_len + a_len>{});
                auto s_b = opus::slice(v_b,
                    opus::number<i_tile_b * b_len>{},
                    opus::number<i_tile_b * b_len + b_len>{});
                auto s_c = opus::slice(v_c,
                    opus::number<i_tile_c * c_len>{},
                    opus::number<i_tile_c * c_len + c_len>{});
                if constexpr (OPSEL)
                    s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b,
                                OpSel::template a<im, ik>(),
                                OpSel::template b<in, ik>());
                else
                    s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b, 0_I, 0_I);
                opus::set_slice(v_c, s_c,
                    opus::number<i_tile_c * c_len>{},
                    opus::number<i_tile_c * c_len + c_len>{});
            });
        });
    });
}

// How the per-subtile scales are materialized in the multi-scale-group path:
//   preload   -- pack every distinct scale once up front, then index the packed
//                registers in the loop (fewer pack ops when a scale is reused
//                across COM_REP_N / COM_REP_M).
//   on_demand -- pack each subtile's scale inline (lower register pressure).
//   opsel     -- pack the COM_REP_K K-group scales into one word per M / N-scale
//                group (native e8m0x4, no broadcast) and select the K byte per
//                MFMA via the hardware scale_op_sel immediate. Fewest scale ALU
//                ops + smallest K-direction scale register footprint.
//   shuffle_scale       -- the reference kernel's layout: one dword per (M subtile pair,
//                K block pair) already in native e8m0x4 order, so there is
//                nothing to pack at all and the MFMA selects both the subtile
//                parity and the K block through scale_op_sel (see opsel_shuf).
enum class mxscale_pack { preload, on_demand, opsel, shuffle_scale };

template<typename T, mxscale_pack MODE = mxscale_pack::preload,
         typename Mma, typename VA, typename VB, typename VSFA, typename VSFB, typename VC>
OPUS_D void mma_mxscale_tiled(Mma& mma, const VA& v_a, const VB& v_b,
                              const VSFA& v_sfa, const VSFB& v_sfb, VC& v_c) {
    static_assert(std::is_same_v<typename T::D_SF, unsigned char>);
    static_assert((T::COM_REP_M == 1 || T::COM_REP_M == 2 || T::COM_REP_M == 4)
                  && (T::COM_REP_K == 1 || T::COM_REP_K == 2 || T::COM_REP_K == 4));
    static_assert(T::B_K % T::GROUP_K == 0);
    // N-repeats per B scale group. The N-waves read blocked column ranges, so
    // within a wave consecutive N-repeats are consecutive W_N columns and a group
    // spans GROUP_N/W_N of them; the T_N term only applies to the interleaved
    // mapping. The two agree whenever T_N==1 or the wave sits inside one group,
    // which covers every kid that predates SFB_PER_WAVE.
    constexpr int rep_n_per_scale =
        T::SFB_PER_WAVE ? (T::GROUP_N / T::W_N) : (T::GROUP_N / (T::W_N * T::T_N));
    static_assert(rep_n_per_scale > 0 && T::GROUP_N % (T::W_N * T::T_N) == 0);
    // Whole register tile in a single scale group -> one (scale_a, scale_b) pair
    // -> a single tiled-mma call covers the tile.
    if constexpr (T::COM_REP_M == 1 && T::COM_REP_N <= rep_n_per_scale && T::COM_REP_K == 1) {
        const int scale_a = pack_e8m0x4(v_sfa[0]);
        const int scale_b = pack_e8m0x4(v_sfb[0]);
        v_c = mma(v_a, v_b, v_c, scale_a, scale_b, 0_I, 0_I);
    } else if constexpr (MODE == mxscale_pack::shuffle_scale) {
        // v_sfa / v_sfb are already the packed dwords the layout stores, so this
        // path has no pack step: A's word covers subtiles 2p and 2p+1, B's covers
        // one N scale group with each K byte stored twice, and opsel_shuf picks
        // the byte from the compile-time (subtile parity, K block) pair.
        mma_mxscale_subtile_loop<T, Mma, /*OPSEL=*/true, opsel_shuf>(v_a, v_b, v_c,
            [&](auto im_c, auto) {
                return v_sfa[decltype(im_c)::value / 2];
            },
            [&](auto in_c, auto) {
                return v_sfb[decltype(in_c)::value / rep_n_per_scale];
            });
    } else if constexpr (MODE == mxscale_pack::opsel) {
        // One word per M-subtile / N-scale-group holding the COM_REP_K K-group
        // e8m0 bytes; the subtile loop picks byte ik via scale_op_sel == ik.
        // NOTE: reference path only. With the vec-wide (dword) scale load below,
        // the shift/or here folds away, but op_sel packing still measures on par
        // with or slightly slower than preload's broadcast pack across the tuned
        // shapes, so preload stays the default. Kept for experimentation.
        opus::vector_t<int, T::COM_REP_M> packed_sfa;
        opus::vector_t<int, T::SFB_GROUPS> packed_sfb;
        opus::static_for<T::COM_REP_M>([&](auto im_c) {
            constexpr int im = decltype(im_c)::value;
            int w = 0;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                w |= (static_cast<int>(v_sfa[im * T::SCALES_PER_BK + ik]) & 0xFF) << (8 * ik);
            });
            packed_sfa[im] = w;
        });
        opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            int w = 0;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                w |= (static_cast<int>(v_sfb[ng * T::SCALES_PER_BK + ik]) & 0xFF) << (8 * ik);
            });
            packed_sfb[ng] = w;
        });
        mma_mxscale_subtile_loop<T, Mma, /*OPSEL=*/true>(v_a, v_b, v_c,
            [&](auto im_c, auto) {
                return packed_sfa[decltype(im_c)::value];
            },
            [&](auto in_c, auto) {
                return packed_sfb[decltype(in_c)::value / rep_n_per_scale];
            });
    } else if constexpr (MODE == mxscale_pack::preload) {
        opus::vector_t<int, T::COM_REP_M * T::COM_REP_K> packed_sfa;
        opus::vector_t<int, T::SFB_GROUPS * T::COM_REP_K> packed_sfb;
        opus::static_for<T::COM_REP_M>([&](auto im_c) {
            constexpr int im = decltype(im_c)::value;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                packed_sfa[im * T::COM_REP_K + ik] =
                    pack_e8m0x4(v_sfa[im * T::SCALES_PER_BK + ik]);
            });
        });
        opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                packed_sfb[ng * T::COM_REP_K + ik] =
                    pack_e8m0x4(v_sfb[ng * T::SCALES_PER_BK + ik]);
            });
        });
        mma_mxscale_subtile_loop<T, Mma>(v_a, v_b, v_c,
            [&](auto im_c, auto ik_c) {
                return packed_sfa[decltype(im_c)::value * T::COM_REP_K + decltype(ik_c)::value];
            },
            [&](auto in_c, auto ik_c) {
                return packed_sfb[(decltype(in_c)::value / rep_n_per_scale) * T::COM_REP_K
                                  + decltype(ik_c)::value];
            });
    } else {
        mma_mxscale_subtile_loop<T, Mma>(v_a, v_b, v_c,
            [&](auto im_c, auto ik_c) {
                return pack_e8m0x4(
                    v_sfa[decltype(im_c)::value * T::SCALES_PER_BK + decltype(ik_c)::value]);
            },
            [&](auto in_c, auto ik_c) {
                return pack_e8m0x4(
                    v_sfb[(decltype(in_c)::value / rep_n_per_scale) * T::SCALES_PER_BK
                          + decltype(ik_c)::value]);
            });
    }
}

// Scale-packing strategy for the default multi-scale-group accum path. Flip
// between preload and opsel here to A/B the hardware scale_op_sel byte-select.
inline constexpr mxscale_pack MXSCALE_ACCUM_MODE = mxscale_pack::preload;

// Thin wrappers preserving the original entry points / call sites.
template<typename T, typename Mma, typename VA, typename VB, typename VSFA, typename VSFB, typename VC>
OPUS_D void mma_mxscale_flatmm_accum(Mma& mma, const VA& v_a, const VB& v_b,
                                     const VSFA& v_sfa, const VSFB& v_sfb, VC& v_c) {
    // T::SCALE_OPSEL kids opt into the hardware byte-select unconditionally;
    // everyone else follows the global A/B switch.
    constexpr mxscale_pack MODE = T::SCALE_OPSEL ? mxscale_pack::opsel : MXSCALE_ACCUM_MODE;
    mma_mxscale_tiled<T, MODE>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
}

template<typename T, typename Mma, typename VA, typename VB, typename VSFA, typename VSFB, typename VC>
OPUS_D void mma_mxscale_flatmm_accum_on_demand(Mma& mma, const VA& v_a, const VB& v_b,
                                               const VSFA& v_sfa, const VSFB& v_sfb, VC& v_c) {
    mma_mxscale_tiled<T, mxscale_pack::on_demand>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
}

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Main kernel: 4-wave flatmm splitK, fp32 workspace output.
// ============================================================================

template<typename Traits, typename D_OUT = void, bool DIRECT_ONLY = false, bool PREFETCH_SCALE = false,
         bool PRELOAD_SF_LDS = false, bool SHUFFLE_SCALE = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a8w8_mxscale_flatmm_splitk_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;
    static_assert(std::is_same_v<D_C, fp32_t>, "flatmm splitK main writes fp32 workspace");
    static_assert(!DIRECT_ONLY || !std::is_void_v<D_OUT>,
                  "DIRECT_ONLY requires an output dtype for direct Y stores");
    // tileN (T_M=1, T_N=2 consumer N-split) is only wired through the non-DIRECT
    // producer/consumer path (barrier-synced; split_k==1 still stores Y directly
    // via the branch below). The persistent DIRECT_ONLY schedule keeps its
    // original tileM-only consumer mapping, so do not instantiate it for tileN.
    static_assert(!(DIRECT_ONLY && T::IS_TILE_N),
                  "tileN is not supported by the DIRECT_ONLY persistent kernel");
    // PRELOAD_SF_LDS stages the scale panels for the barrier-synced
    // producer/consumer schedule; it is not wired into the DIRECT_ONLY
    // consumer-self-load path.
    static_assert(!(PRELOAD_SF_LDS && DIRECT_ONLY),
                  "PRELOAD_SF_LDS is only supported by the non-DIRECT_ONLY schedule");
    // Direct-to-register B is wired into the barrier-synced producer/consumer
    // schedule only; the persistent DIRECT_ONLY kernel keeps its own B staging.
    static_assert(!(T::B_DIRECT_REG && DIRECT_ONLY),
                  "B_DIRECT_REG is not supported by the DIRECT_ONLY persistent kernel");
    static_assert(!T::B_DIRECT_REG || T::B_PRESHUFFLE,
                  "B_DIRECT_REG requires the 16x16 preshuffled weight layout");
    // Direct B and the LDS scale panels compose: do_scaled_mma waits on vmcnt for
    // B and on lgkmcnt for the panels, rather than folding B's retirement into a
    // single scale wait.
    static_assert(!(T::ALL_WAVE && DIRECT_ONLY),
                  "ALL_WAVE replaces the producer/consumer split, not the persistent kernel");
    // Under ALL_WAVE the staging waves are the computing waves, so the tile's
    // async copies are the only thing left on vmcnt. A per-K-tile global scale
    // load would sit on the same counter and break the fixed in-flight bound
    // stage_barrier waits on, so the scales have to come from the LDS panels.
    static_assert(!T::ALL_WAVE || PRELOAD_SF_LDS,
                  "ALL_WAVE needs PRELOAD_SF_LDS so vmcnt carries only the tile copies");

    int wgid_full = opus::block_id_x();
    int split_id  = 0;
    int wgid      = wgid_full;
    if constexpr (!DIRECT_ONLY) {
        split_id = wgid_full % kargs.split_k;
        wgid = wgid_full / kargs.split_k;
    }
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col = (wgid / num_tiles_m) * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();

    const int total_iters = ceil_div(kargs.k, T::B_K);
    int my_loops = total_iters;
    int k_start = 0;
    int sf_start = 0;
    if constexpr (!DIRECT_ONLY) {
        const int iters_full = ceil_div(total_iters, kargs.split_k);
        my_loops = (split_id < kargs.split_k - 1)
                 ? iters_full
                 : (total_iters - (kargs.split_k - 1) * iters_full);
        k_start = split_id * iters_full * T::B_K;
        sf_start = split_id * iters_full * (T::B_K / T::GROUP_K);
    }
    if (my_loops < T::prefetch_k_iter) return;

    // OOB masking for partial M tiles: bound the A / sfa / C buffers to the
    // valid row window so lanes mapping to rows >= M read 0 and their stores are
    // dropped by the buffer's num_records bound. This lets any M run on a B_M
    // tile without requiring M % B_M == 0 (N/K stay divisible so B and the K
    // axis need no bound).
    //
    // Each WG owns exactly one B_M row tile and the buffer base is already at
    // `row`, so the bound only needs to cover this tile's own rows -- clamp to
    // B_M. Using the full (M - row) span would set num_records to
    // rows_avail*stride_a, and with batch-in-the-middle stride_a = batch*K, so a
    // large-M / high-batch shape would overflow the 32-bit buffer-descriptor
    // num_records (4 GiB) field and silently wrap, corrupting the OOB bound.
    // min(rows_avail, B_M) still masks the partial-M tail correctly.
    // rows_avail >= 1 always (row < M by construction).
    const int rows_left = kargs.m - row;
    const int rows_avail = rows_left < T::B_M ? rows_left : T::B_M;
    const unsigned int a_bytes =
        (unsigned int)rows_avail * (unsigned int)kargs.stride_a * sizeof(D_A);
    // 64-bit base offsets: batch_id*stride_*_batch (= M*K for a batch-in-the-
    // middle A layout) overflows int32 for large M well before the 4 GiB buffer
    // limit, so cast the batch/row products to size_t to keep the base exact.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + (size_t)batch_id * kargs.stride_a_batch + (size_t)row * kargs.stride_a + k_start,
                         a_bytes);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch
                         + b_gmem_tile_base_mxsk<T>(col, k_start, kargs.stride_b));
    const bool direct_store = DIRECT_ONLY || (!std::is_void_v<D_OUT> && kargs.split_k == 1);
    const int stride_c_main = direct_store ? kargs.stride_c : kargs.stride_ws;
    const unsigned int sfa_bytes =
        (unsigned int)rows_avail * (unsigned int)kargs.stride_sfa * sizeof(D_SF);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + (size_t)batch_id * kargs.stride_sfa_batch
                           + (size_t)row * kargs.stride_sfa + sf_start,
                           sfa_bytes);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb + sf_start);

    // Shuffled scale panels, addressed in dwords. One dword holds two M subtiles
    // (SFA_MB rows apart) crossed with two 128-blocks of K, which puts the lane
    // axis at a stride of one dword so a quarter wave's 16 lanes cover a single
    // 64B line instead of sixteen. The address derivation is the wave8
    // pipeline's; see the note there.
    //
    // At B_K=256 a tile owns both of the dword's K blocks and op_sel's high bit
    // is just the K repeat. At B_K=128 the two straddle a K tile boundary, and
    // rather than unroll the loop by two to keep the parity a compile-time
    // immediate (what the wave8 kids do) or shift the high half down on odd tiles
    // (what FlyDSL does), this reads 16 bits instead of 32: the dword is
    // [k0m0, k0m1, k1m0, k1m1], so one K block's two subtile bytes are adjacent
    // and a u16 load at short index 2*word + (k&1) lands them zero-extended in
    // byte 0 and 1. op_sel is then just the subtile parity, which is what
    // opsel_shuf already returns at ik==0. No unroll, no VALU, same line count.
    //
    // The row offset does not go in the base either way: the layout folds the row
    // into the word index, so the panel is taken whole.
    constexpr int SFA_MB = T::T_M * T::W_M;
    static_assert(!SHUFFLE_SCALE || T::COM_REP_K <= 2,
                  "the shuffle_scale dword spans at most two K blocks");
    static_assert(!SHUFFLE_SCALE || T::COM_REP_M % 2 == 0,
                  "the shuffle_scale dword pairs adjacent M subtiles");
    static_assert(!SHUFFLE_SCALE || T::B_M % (2 * SFA_MB) == 0,
                  "the shuffle_scale layout blocks the tile's rows by adjacent subtile pairs");
    static_assert(!SHUFFLE_SCALE || !PRELOAD_SF_LDS,
                  "the shuffle_scale layout replaces the LDS scale panel rather than filling it");
    // K1 counts 128-block pairs over the whole of K, which is the pitch the host
    // padded both panels to.
    const int shuf_k1 = (ceil_div(kargs.k, T::GROUP_K) + 1) / 2;
    const int shuf_k1_start = sf_start / 2;
    auto g_sfa_shuf = make_gmem(
        reinterpret_cast<const int*>(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                                     + (size_t)batch_id * kargs.stride_sfa_batch),
        (unsigned int)kargs.stride_sfa_batch * (unsigned int)sizeof(D_SF));
    auto g_sfb_shuf = make_gmem(
        reinterpret_cast<const int*>(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                                     + (size_t)batch_id * kargs.stride_sfb_batch));
    // The same two panels seen as u16, for the B_K=128 half-dword read above.
    auto g_sfa_shuf_h = make_gmem(
        reinterpret_cast<const unsigned short*>(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                                                + (size_t)batch_id * kargs.stride_sfa_batch),
        (unsigned int)kargs.stride_sfa_batch * (unsigned int)sizeof(D_SF));
    auto g_sfb_shuf_h = make_gmem(
        reinterpret_cast<const unsigned short*>(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                                                + (size_t)batch_id * kargs.stride_sfb_batch));

    // ALL_WAVE has no producers: every wave takes the consumer path and stages
    // its own share of each tile on the way (see stage_barrier below).
    int role = T::ALL_WAVE ? 1 : ((wave_id & 1) ^ ((wgid >> 8) & 1));

    constexpr int smem_slot_factor = DIRECT_ONLY ? 2 : 1;
    // B_DIRECT_REG consumers buffer_load B into registers, so the B staging
    // buffer is dead -- keep a placeholder so smem_b_at() still type-checks.
    constexpr int smem_b_bytes = T::B_DIRECT_REG
        ? 16
        : (smem_slot_factor * T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BN
           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size);
    __shared__ char smem_a[smem_slot_factor * T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[smem_b_bytes];

    // PRELOAD_SF_LDS (kid324): stage the A per-token scale (SFA) and B block
    // scale (SFB) panels for this split's whole K range into LDS once, then read
    // them from LDS (ds_read/lgkmcnt) in the consumer's per-K-tile scale fetch
    // instead of a per-tile global buffer_load (vmcnt) that gates every MMA. The
    // panels are compact byte tiles: SFA is [B_M/GROUP_M rows][loops*SCALES_PER_BK]
    // and SFB is [N_SCALE_GROUPS rows][loops*SCALES_PER_BK], both row-major with a
    // runtime per-row stride == the packed K-scale count. The LDS buffer is sized
    // for a compile-time K upper bound (SFA_K_MAX); the actual packed count is a
    // runtime value so any K<=SFA_K_MAX (K%B_K==0) works. SFA_K_MAX=8192 keeps the
    // combined panel <=~4.2 KiB, well inside the WG_PER_CU=2 LDS headroom.
    constexpr int SFA_K_MAX        = T::SF_PRELOAD_K_MAX;
    constexpr int SFA_K_TILES_MAX  = PRELOAD_SF_LDS ? (SFA_K_MAX / T::B_K) : 1;
    constexpr int SF_SCALES_MAX    = SFA_K_TILES_MAX * T::SCALES_PER_BK;
    constexpr int SFA_ROWS         = T::B_M / T::GROUP_M;
    constexpr int SF_LDS_ELEMS     =
        PRELOAD_SF_LDS ? ((SFA_ROWS + T::N_SCALE_GROUPS) * SF_SCALES_MAX) : 1;
    // 16B-aligned so the panel fill below can land ds_write_b128; a byte array is
    // only byte-aligned as far as the language is concerned.
    alignas(16) __shared__ D_SF smem_sf[SF_LDS_ELEMS];

    auto smem_a_at = [&](int slot_k, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot_k, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };
    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return b_gmem_group_offset_mxsk<T>(loop_k_idx, group_load_idx, k_group, kargs.stride_b);
    };

    const int loops = my_loops;
    // Runtime packed K-scale count per SFA/SFB row (== loops * SCALES_PER_BK) and
    // the two LDS panel base pointers. SFB is packed immediately after SFA using
    // the runtime SFA size so both stay compact regardless of K.
    const int sf_k_scales = loops * T::SCALES_PER_BK;
    D_SF* s_sfa_ptr = smem_sf;
    D_SF* s_sfb_ptr = smem_sf + SFA_ROWS * sf_k_scales;
    constexpr int mb_a = T::a_buffer_load_insts;
    constexpr int mb_b = T::B_DIRECT_REG ? 0 : T::b_buffer_load_insts;
    constexpr int mb = mb_a + mb_b;

    if constexpr (DIRECT_ONLY) {
        __shared__ int b_ready[T::prefetch_k_iter];
        if (opus::thread_id_x() < T::prefetch_k_iter) {
            b_ready[opus::thread_id_x()] = -1;
        }
        s_waitcnt_lgkmcnt(0_I);  // retire the init writes before other waves read them
        __builtin_amdgcn_s_barrier();
        if ((wave_id & 1) == 0) return;

        int wave_id_m = wave_id / 2;
        int wave_id_n_cons = 0;
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 1>(lane_id, 0, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 1>(lane_id, 0);
        auto u_gb = make_layout_gmem_b_mxsk<T, 1>(lane_id, 0, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 1>(lane_id, 0);
        auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_mxsk<T>(lane_id);
        auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        typename decltype(mma)::vtype_a v_a;
        typename decltype(mma)::vtype_b v_b;
        typename decltype(mma)::vtype_c v_c;
        clear(v_c);

        using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
        using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;

        auto issue_a_tile = [&](int loop_k) {
            const int slot = wave_id_m * T::prefetch_k_iter + (loop_k % T::prefetch_k_iter);
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                    constexpr int m = decltype(m_c)::value;
                    async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(loop_k, m, kg));
                });
            });
        };

        auto issue_b_tile = [&](int loop_k) {
            const int slot = loop_k % T::prefetch_k_iter;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(loop_k, n, kg));
                });
            });
        };

        auto load_scales = [&](int loop_k, vtype_sfa& v_sfa, vtype_sfb& v_sfb) {
            const int scale_base = loop_k * T::SCALES_PER_BK;
            v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
            opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
                constexpr int ng = decltype(ng_c)::value;
                auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
                opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                });
            });
            s_waitcnt_vmcnt(0_I);
        };

        auto do_mma = [&](const auto& va, const auto& vb,
                          const vtype_sfa& v_sfa, const vtype_sfb& v_sfb) {
            __builtin_amdgcn_s_setprio(1);
            mma_mxscale_flatmm_accum<T>(mma, va, vb, v_sfa, v_sfb, v_c);
            __builtin_amdgcn_s_setprio(0);
        };

        issue_a_tile(0);
        if (wave_id_m == 0) {
            issue_b_tile(0);
        }
        for (int k = 0; k < loops; ++k) {
            const int a_slot = wave_id_m * T::prefetch_k_iter + (k % T::prefetch_k_iter);
            const int b_slot = k % T::prefetch_k_iter;
            s_waitcnt_vmcnt(0_I);
            if (wave_id_m == 0) {
                reinterpret_cast<volatile int*>(b_ready)[b_slot] = k;
            } else {
                volatile int* ready = reinterpret_cast<volatile int*>(b_ready);
                while (ready[b_slot] != k) {
                }
            }

            auto sa = make_smem(smem_a_at(a_slot, 0, 0));
            auto sb = make_smem(smem_b_at(b_slot, 0, 0));
            v_a = load<T::VEC_A>(sa, u_ra);
            v_b = load<T::VEC_B>(sb, u_rb);
            s_waitcnt_lgkmcnt(0_I);

            vtype_sfa v_sfa;
            vtype_sfb v_sfb;
            load_scales(k, v_sfa, v_sfb);
            if (k + 1 < loops) {
                issue_a_tile(k + 1);
                if (wave_id_m == 0) {
                    issue_b_tile(k + 1);
                }
            }
            do_mma(v_a, v_b, v_sfa, v_sfb);
        }

        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
        D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                       + (size_t)batch_id * kargs.stride_c_batch
                       + (size_t)row * kargs.stride_c
                       + (size_t)col;
        auto g_out = make_gmem(out_ptr,
            (unsigned int)rows_avail * (unsigned int)kargs.stride_c * sizeof(D_OUT));
        store<T::VEC_C>(g_out, v_c, u_gc, 0);
        return;
    }

    // PRELOAD_SF_LDS: cooperative one-shot fill of the SFA + SFB panels into LDS.
    // Executed by all BLOCK_SIZE threads (both producer and consumer waves) before
    // the producer/consumer role split, with a barrier publishing the panels for
    // the consumer reads below. Grid-stride over the compact scalar byte counts.
    // OOB rows (partial-M tail) read 0 via g_sfa's num_records bound and are
    // never consumed. Bail if this split's K exceeds the compile-time LDS bound
    // (the dispatch never selects kid324 for such K, so this only guards misuse).
    if constexpr (PRELOAD_SF_LDS) {
        if (loops > SFA_K_TILES_MAX) return;
        const int tid = opus::thread_id_x();
        auto sm_sfa = make_smem(s_sfa_ptr);
        auto sm_sfb = make_smem(s_sfb_ptr);
        const int sfa_total = SFA_ROWS * sf_k_scales;
        const int sfb_total = T::N_SCALE_GROUPS * sf_k_scales;
        // Copy the widest chunk the panel geometry allows. Byte-at-a-time is 16
        // grid-stride iterations per thread for the 128-row SFA panel at K=4096
        // (kid325/326), and the fill sits in front of a barrier, so its latency is
        // exposed rather than overlapped. A chunk must not span two panel rows and
        // its source offset must stay naturally aligned, so the width has to divide
        // both sf_k_scales and the row stride; hence the short-K fallbacks.
        auto fill = [&](auto vec_c, auto sm, auto g, int stride, int total) {
            constexpr int VEC = decltype(vec_c)::value;
            for (int idx = tid * VEC; idx < total; idx += T::BLOCK_SIZE * VEC) {
                const int r  = idx / sf_k_scales;
                const int kt = idx - r * sf_k_scales;
                sm.template store<VEC>(load<VEC>(g, r * stride + kt), idx);
            }
        };
        auto fill_panel = [&](auto sm, auto g, int stride, int total) {
            const int widths = sf_k_scales | stride;
            if      ((widths & 15) == 0) fill(number<16>{}, sm, g, stride, total);
            else if ((widths & 3) == 0)  fill(number<4>{},  sm, g, stride, total);
            else                         fill(number<1>{},  sm, g, stride, total);
        };
        fill_panel(sm_sfa, g_sfa, kargs.stride_sfa, sfa_total);
        fill_panel(sm_sfb, g_sfb, kargs.stride_sfb, sfb_total);
        // vmcnt retires the global reads feeding the panel; lgkmcnt retires the
        // ds_writes that actually publish it. s_barrier does neither on its own.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    if (role == 0) {
        int wave_id_prod = wave_id / 2;
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);
        auto u_gb = make_layout_gmem_b_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);
        // B_DIRECT_REG: consumers buffer_load B into their own MFMA registers,
        // so producers stage A only (and mb drops the B instruction count).
        auto stage_b = [&](int slot, int issue_k, auto kg_c) {
            if constexpr (!T::B_DIRECT_REG) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb,
                                         b_offset(issue_k, n, kg));
                });
            }
        };

        opus::static_for<T::prefetch_k_iter>([&](auto p_c) {
            constexpr int p = decltype(p_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                    constexpr int m = decltype(m_c)::value;
                    async_load<T::VEC_A>(g_a, smem_a_at(p, m, kg), u_ga, u_sa, a_offset(p, m, kg));
                });
                stage_b(p, p, kg_c);
            });
        });

        opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
            constexpr int p = T::prefetch_k_iter - 1 - decltype(i_c)::value;
            s_waitcnt_vmcnt(number<mb * p>{});
            __builtin_amdgcn_s_barrier();
        });

        if constexpr (T::prefetch_k_iter == 3) {
            s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();
            for (int i = T::prefetch_k_iter - 1; i < loops - 1; i++) {
                int issue_k = i + 1;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    stage_b(slot, issue_k, kg_c);
                });
                s_waitcnt_vmcnt(number<mb>{});
                __builtin_amdgcn_s_barrier();
            }
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        } else {
            for (int i = T::prefetch_k_iter - 2; i < loops - 2; i++) {
                int issue_k = i + 2;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    stage_b(slot, issue_k, kg_c);
                });
                s_waitcnt_vmcnt(number<2 * mb>{});
                __builtin_amdgcn_s_barrier();
            }
            s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
    } else {
        // Consumer waves. tileM: two waves split M (wave_id_m in {0,1}, single
        // N column-block). tileN: two waves split N (single M-wave, wave_id_n
        // in {0,1}); each consumer reads its own 16-col B group from smem, and
        // the C partition / rb layout follow via T_N=2. wave_id_n_cons is 0 for
        // tileM, so the shared `n_block = wave_id_n_cons` smem-B base below is
        // bit-identical for the existing tileM kids.
        // ALL_WAVE: the four waves form a 2x2 (M,N) grid, so both coordinates come
        // from wave_id. The split kinds take one of the two from wave_id/2, since
        // there only every other wave is a consumer.
        int wave_id_m = T::ALL_WAVE ? (wave_id & 1) : (T::IS_TILE_N ? 0 : (wave_id / 2));
        int wave_id_n_cons = T::ALL_WAVE ? (wave_id >> 1) : (T::IS_TILE_N ? (wave_id / 2) : 0);
        // Consumer B smem N-group base. Each consumer N-wave owns COM_REP_N
        // contiguous 16-col load groups (rb reads num_blocks_n=COM_REP_N from
        // this base). tileM: wave_id_n_cons=0 -> nbc=0 (bit-identical).
        // In load groups, not 16-col blocks: a wave owns COM_REP_N*W_N columns,
        // which is COM_REP_N*W_N/LOAD_GROUP_N groups. tileN has LOAD_GROUP_N ==
        // W_N, where that collapses to COM_REP_N and this is unchanged.
        const int nbc = wave_id_n_cons * (T::COM_REP_N * T::W_N / T::LOAD_GROUP_N);
        auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_mxsk<T>(lane_id);
        auto u_gb_direct = make_layout_gmem_b_direct_mxsk<T>(lane_id, kargs.stride_b, nbc);
        auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);
        // LDS read layout for the preloaded SFA panel: same lane/wave mapping as
        // u_sfa but with the compact per-row K-scale count as the row stride.
        auto u_sfa_lds = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, sf_k_scales);

        // ALL_WAVE staging. Each of the four waves owns a quarter of every async
        // copy (LOAD_WAVES=4) and stages it itself, in place of the producer
        // pair. The tile index is clamped to the last valid tile so that every
        // barrier issues exactly one full tile and exactly prefetch_k_iter-1
        // tiles are always in flight -- that keeps the vmcnt bound a single
        // compile-time immediate with no tail special case. The clamped tail
        // copies re-read live bytes into slots nobody reads again.
        auto u_ga_aw = make_layout_gmem_group_load_mxsk<T, T::LOAD_WAVES>(
            lane_id, wave_id, kargs.stride_a);
        auto u_sa_aw = make_layout_smem_group_load_mxsk<T, T::LOAD_WAVES>(lane_id, wave_id);
        auto u_gb_aw = make_layout_gmem_b_mxsk<T, T::LOAD_WAVES>(
            lane_id, wave_id, kargs.stride_b);
        auto u_sb_aw = make_layout_smem_group_load_mxsk<T, T::LOAD_WAVES>(lane_id, wave_id);
        constexpr int aw_mb = T::a_buffer_load_insts + T::b_buffer_load_insts;
        constexpr int aw_inflight = aw_mb * (T::prefetch_k_iter - 1);

        auto issue_tile_aw = [&](int issue_k) {
            if constexpr (T::ALL_WAVE) {
                const int kk = issue_k < loops ? issue_k : loops - 1;
                const int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga_aw, u_sa_aw,
                                             a_offset(kk, m, kg));
                    });
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                        constexpr int n = decltype(n_c)::value;
                        async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb_aw, u_sb_aw,
                                             b_offset(kk, n, kg));
                    });
                });
            }
        };

        // The barrier that publishes tile `pub`. Under ALL_WAVE it first refills
        // the slot freed by tile pub-1 and waits for tile pub to have landed;
        // otherwise it is the plain producer/consumer rendezvous.
        auto stage_barrier = [&](int pub) {
            if constexpr (T::ALL_WAVE) {
                issue_tile_aw(pub + T::prefetch_k_iter - 1);
                s_waitcnt_vmcnt(number<aw_inflight>{});
            }
            __builtin_amdgcn_s_barrier();
        };

        if constexpr (T::ALL_WAVE) {
            opus::static_for<T::prefetch_k_iter - 1>([&](auto p_c) {
                issue_tile_aw(decltype(p_c)::value);
            });
        }

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        typename decltype(mma)::vtype_a v_a0, v_a1;
        typename decltype(mma)::vtype_b v_b0, v_b1;
        typename decltype(mma)::vtype_c v_c;
        clear(v_c);

        using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
        using vtype_sfb = vector_t<D_SF, T::SFB_GROUPS * T::SCALES_PER_BK>;
        // First B scale group this N-wave's columns fall in. Zero unless the wave
        // loads only its own slice of the tile's groups.
        const int sfb_group_base = T::SFB_PER_WAVE
            ? wave_id_n_cons * T::SFB_GROUP_STRIDE : 0;
        // Lane-invariant dword index of this lane's first shuffle_scale word. B_M % 2*SFA_MB
        // == 0 makes `row` a whole number of subtile pairs, and wave_id_m*W_M +
        // lane%W_M is this lane's row inside one subtile, below SFA_MB.
        const int shuf_a_word0 = (row / (2 * SFA_MB)) * shuf_k1 * SFA_MB
                              + wave_id_m * T::W_M + lane_id % T::W_M;
        const int shuf_b_word0 = (col / T::GROUP_N + sfb_group_base) * shuf_k1;
        // A's word covers a subtile pair, B's an N scale group with each K byte
        // stored twice. Both are consumed inside the K tile that reads them,
        // which is what B_K=256 buys; nothing carries across tiles.
        vector_t<int, SHUFFLE_SCALE ? T::COM_REP_M / 2 : 1> v_sfa_shuf;
        vector_t<int, SHUFFLE_SCALE ? T::SFB_GROUPS : 1> v_sfb_shuf;
        constexpr int b_direct_insts = T::B_DIRECT_REG ? T::b_ds_read_insts : 0;
        constexpr int ds_read_insts =
            T::a_ds_read_insts + (T::B_DIRECT_REG ? 0 : T::b_ds_read_insts);

        // B for K tile `loop_k`, straight into the MFMA registers. Always issued
        // so the vmcnt accounting below stays uniform across every K tile; the
        // tail's would-be-past-the-end tile is clamped to the last one, which
        // re-reads live bytes into a register buffer nobody consumes.
        auto issue_b_direct = [&](auto& vb, int loop_k) {
            if constexpr (T::B_DIRECT_REG) {
                const int kk = loop_k < loops ? loop_k : loops - 1;
                vb = load<T::VEC_B>(g_b, u_gb_direct, b_direct_iter_offset_mxsk<T>(kk));
            }
        };
        // LDS-path B read, at the same point in the schedule as the A ds_read.
        auto read_b_lds = [&](auto& vb, int slot) {
            if constexpr (!T::B_DIRECT_REG) {
                auto sb = make_smem(smem_b_at(slot, nbc, 0));
                vb = load<T::VEC_B>(sb, u_rb);
            }
        };

        auto load_scale_regs = [&](int loop_k, vtype_sfa& v_sfa, vtype_sfb& v_sfb) {
            const int scale_base = loop_k * T::SCALES_PER_BK;
            if constexpr (SHUFFLE_SCALE) {
                // One coalesced load per subtile pair, against the plain path's
                // one uncoalesced dword per subtile: same bytes, half the loads,
                // and a quarter wave lands on one line rather than sixteen. At
                // B_K=256 the tile takes the whole dword; at B_K=128 it takes the
                // half its K block owns and the parity rides in the index.
                const int k1 = T::COM_REP_K == 2 ? loop_k : (loop_k >> 1);
                const int kp = T::COM_REP_K == 2 ? 0 : (loop_k & 1);
                opus::static_for<T::COM_REP_M / 2>([&](auto p_c) {
                    constexpr int p = decltype(p_c)::value;
                    const int w = shuf_a_word0 + (p * shuf_k1 + shuf_k1_start + k1) * SFA_MB;
                    if constexpr (T::COM_REP_K == 2)
                        v_sfa_shuf[p] = load<1>(g_sfa_shuf, w)[0];
                    else
                        v_sfa_shuf[p] = (int)load<1>(g_sfa_shuf_h, 2 * w + kp)[0];
                });
                opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
                    constexpr int ng = decltype(ng_c)::value;
                    const int w = shuf_b_word0 + ng * shuf_k1 + shuf_k1_start + k1;
                    if constexpr (T::COM_REP_K == 2)
                        v_sfb_shuf[ng] = load<1>(g_sfb_shuf, w)[0];
                    else
                        v_sfb_shuf[ng] = (int)load<1>(g_sfb_shuf_h, 2 * w + kp)[0];
                });
            } else if constexpr (PRELOAD_SF_LDS) {
                // Read this K-tile's scales from the preloaded LDS panels
                // (ds_read / lgkmcnt) instead of a per-tile global buffer_load.
                // Vec = SCALES_PER_BK so the contiguous per-M-row K bytes come in
                // as one dword (ds_read_b32) instead of SCALES_PER_BK byte reads.
                auto sm_a = make_smem(s_sfa_ptr + scale_base);
                v_sfa = load<T::SCALES_PER_BK>(sm_a, u_sfa_lds);
                opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
                    constexpr int ng = decltype(ng_c)::value;
                    auto sm_b = make_smem(s_sfb_ptr
                                          + (sfb_group_base + ng) * sf_k_scales + scale_base);
                    auto sfb = load<T::SCALES_PER_BK>(sm_b, 0);
                    opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                    });
                });
            } else {
                // Vec = SCALES_PER_BK: the contiguous per-M-row K-scale bytes are
                // read as one dword (buffer_load_b32) rather than SCALES_PER_BK
                // separate buffer_load_ubyte. SFB already loads b32 the same way.
                v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
                opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
                    constexpr int ng = decltype(ng_c)::value;
                    auto sfb = load<T::SCALES_PER_BK>(
                        g_sfb, (sfb_group_base + ng) * kargs.stride_sfb + scale_base);
                    opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                    });
                });
            }
        };

        auto do_scaled_mma = [&](const auto& va, const auto& vb,
                                 const vtype_sfa& v_sfa, const vtype_sfb& v_sfb) {
            // The two counters are independent, and which ones this tile has to
            // wait on depends on where its operands came from:
            //   vmcnt  -- the global scale loads (non-preload) and/or direct B.
            //   lgkmcnt -- the LDS scale panels (preload), plus the A ds_reads.
            // vmcnt is in-order, and every K tile issues its scales before the
            // next tile's B, so leaving exactly b_direct_insts outstanding retires
            // this tile's scales *and* its B while the next tile's B stays in
            // flight. Under preload the scales are no longer vmcnt traffic, so the
            // consumer's only global loads are direct B and the same count still
            // retires B(k) alone. b_direct_insts == 0 restores the LDS path's
            // plain "drain everything" wait.
            if constexpr (T::B_DIRECT_REG || !PRELOAD_SF_LDS) {
                s_waitcnt_vmcnt(number<b_direct_insts>{});
            }
            if constexpr (PRELOAD_SF_LDS) {
                s_waitcnt_lgkmcnt(0_I);
            }
            __builtin_amdgcn_s_setprio(1);
            if constexpr (SHUFFLE_SCALE) {
                mma_mxscale_tiled<T, mxscale_pack::shuffle_scale>(mma, va, vb,
                                                        v_sfa_shuf, v_sfb_shuf, v_c);
            } else {
                mma_mxscale_flatmm_accum<T>(mma, va, vb, v_sfa, v_sfb, v_c);
            }
            __builtin_amdgcn_s_setprio(0);
        };

        // vb_next is the double-buffer half this tile's MMA does not read, i.e.
        // the one the next K tile's B lands in.
        auto scaled_mma = [&](const auto& va, const auto& vb, auto& vb_next, int loop_k) {
            vtype_sfa v_sfa;
            vtype_sfb v_sfb;
            load_scale_regs(loop_k, v_sfa, v_sfb);
            issue_b_direct(vb_next, loop_k + 1);
            do_scaled_mma(va, vb, v_sfa, v_sfb);
        };

        auto wait_lgkm_then_scaled_mma =
            [&](const auto& va, const auto& vb, auto& vb_next, int loop_k, auto lgkm_cnt) {
                if constexpr (PREFETCH_SCALE) {
                    vtype_sfa v_sfa;
                    vtype_sfb v_sfb;
                    load_scale_regs(loop_k, v_sfa, v_sfb);
                    issue_b_direct(vb_next, loop_k + 1);
                    s_waitcnt_lgkmcnt(lgkm_cnt);
                    do_scaled_mma(va, vb, v_sfa, v_sfb);
                } else {
                    s_waitcnt_lgkmcnt(lgkm_cnt);
                    scaled_mma(va, vb, vb_next, loop_k);
                }
            };

        stage_barrier(0);
        {
            auto sa0 = make_smem(smem_a_at(0, 0, 0));
            v_a0 = load<T::VEC_A>(sa0, u_ra);
            read_b_lds(v_b0, 0);
            issue_b_direct(v_b0, 0);
        }

        opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
            constexpr int p = decltype(i_c)::value + 1;
            constexpr int cur = (p - 1) & 1;
            constexpr int nxt = p & 1;
            stage_barrier(p);
            auto sa_p = make_smem(smem_a_at(p, 0, 0));
            if constexpr (nxt == 0) {
                v_a0 = load<T::VEC_A>(sa_p, u_ra);
                read_b_lds(v_b0, p);
            } else {
                v_a1 = load<T::VEC_A>(sa_p, u_ra);
                read_b_lds(v_b1, p);
            }
            if constexpr (cur == 0) {
                wait_lgkm_then_scaled_mma(v_a0, v_b0, v_b1, p - 1, number<ds_read_insts>{});
            } else {
                wait_lgkm_then_scaled_mma(v_a1, v_b1, v_b0, p - 1, number<ds_read_insts>{});
            }
        });

        constexpr int L = (T::prefetch_k_iter - 2) & 1;
        int k = T::prefetch_k_iter - 1;
        for (; k + 1 < loops - 1; k += 2) {
            stage_barrier(k);
            {
                int slot = k % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                if constexpr (L == 0) {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    read_b_lds(v_b1, slot);
                } else {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    read_b_lds(v_b0, slot);
                }
            }
            if constexpr (L == 0) {
                wait_lgkm_then_scaled_mma(v_a0, v_b0, v_b1, k - 1, number<ds_read_insts>{});
            } else {
                wait_lgkm_then_scaled_mma(v_a1, v_b1, v_b0, k - 1, number<ds_read_insts>{});
            }

            stage_barrier(k + 1);
            {
                int slot = (k + 1) % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                if constexpr (L == 0) {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    read_b_lds(v_b0, slot);
                } else {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    read_b_lds(v_b1, slot);
                }
            }
            if constexpr (L == 0) {
                wait_lgkm_then_scaled_mma(v_a1, v_b1, v_b0, k, number<ds_read_insts>{});
            } else {
                wait_lgkm_then_scaled_mma(v_a0, v_b0, v_b1, k, number<ds_read_insts>{});
            }
        }

        bool last_in_buf1 = (L != 0);
        if (k < loops - 1) {
            stage_barrier(k);
            {
                int slot = k % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                if constexpr (L == 0) {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    read_b_lds(v_b1, slot);
                } else {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    read_b_lds(v_b0, slot);
                }
            }
            if constexpr (L == 0) {
                wait_lgkm_then_scaled_mma(v_a0, v_b0, v_b1, k - 1, number<ds_read_insts>{});
            } else {
                wait_lgkm_then_scaled_mma(v_a1, v_b1, v_b0, k - 1, number<ds_read_insts>{});
            }
            last_in_buf1 = (L == 0);
            k++;
        }

        stage_barrier(loops - 1);
        int last_slot = (loops - 1) % T::prefetch_k_iter;
        auto sa_last = make_smem(smem_a_at(last_slot, 0, 0));
        if (last_in_buf1) {
            v_a0 = load<T::VEC_A>(sa_last, u_ra);
            read_b_lds(v_b0, last_slot);
            wait_lgkm_then_scaled_mma(v_a1, v_b1, v_b0, loops - 2, number<ds_read_insts>{});
            wait_lgkm_then_scaled_mma(v_a0, v_b0, v_b1, loops - 1, 0_I);
        } else {
            v_a1 = load<T::VEC_A>(sa_last, u_ra);
            read_b_lds(v_b1, last_slot);
            wait_lgkm_then_scaled_mma(v_a0, v_b0, v_b1, loops - 2, number<ds_read_insts>{});
            wait_lgkm_then_scaled_mma(v_a1, v_b1, v_b0, loops - 1, 0_I);
        }

        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(stride_c_main, 1_I), p_coord_c);
        // tileN with COM_REP_N>1: two consumer waves split B_N and each replicates
        // COM_REP_N N-repeats. The generic swap_ab C partition nests the register
        // N-repeat (expd_n) OUTSIDE the consumer-wave tile (tile_n), which transposes
        // the (wave, n-rep) -> column map (each wave writes a strided, interleaved
        // column set) whenever BOTH T_N>1 and COM_REP_N>1 -- the accumulators are
        // correct but land in swapped output columns. Consumer wave w computes n-rep
        // j from B column-group (w*COM_REP_N + j) (see nbc = wave_id_n_cons*COM_REP_N
        // + the num_blocks_n rb read), so store each n-rep slice to that contiguous
        // column group with a single-N-tile (E_N=1,T_N=1) layout and a scalar column
        // offset. tileM (T_N=1) and tileN COM_REP_N==1 keep the original single store
        // (SPLIT_N_STORE=false), so they stay bit-identical.
        constexpr bool SPLIT_N_STORE = (T::T_N > 1) && (T::COM_REP_N > 1);
        constexpr int C_LEN = decltype(mma)::mma_c_len;
        auto mma_c1 = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, 1, T::COM_REP_K>{}, seq<T::T_M, 1, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
        auto p_coord_c1 = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                           0, lane_id / mma.grpn_c);
        auto u_gc1 = partition_layout_c<T::VEC_C>(mma_c1,
            opus::make_tuple(stride_c_main, 1_I), p_coord_c1);
        auto store_c = [&](auto& g) {
            if constexpr (SPLIT_N_STORE) {
                // The accumulator nests m-repeat outside n-repeat (i_tile_c =
                // im*COM_REP_N + in), so one n-repeat's tiles sit COM_REP_N*C_LEN
                // apart rather than contiguously. Gather them into the layout
                // mma_c1 expects; with COM_REP_M==1 this is the identity, which is
                // why a plain prefix slice held up for tileN.
                //
                // At COM_REP_M>1 this order also costs DRAM write bandwidth: one
                // store covers only 32 bytes of each row, so the two halves of a
                // 64B line come from n-repeats j and j+1 and this leaves COM_REP_M
                // stores between them. The wave8 pipeline's copy of this loop
                // measured 1.30x the necessary DRAM writes and fixed it by making
                // the n-repeat innermost; the same is available here and in the
                // allwave_bdirect epilogue below, neither of which is on a kid
                // fast enough to be worth the churn yet.
                opus::static_for<T::COM_REP_N>([&](auto j_c) {
                    constexpr int j = decltype(j_c)::value;
                    typename decltype(mma_c1)::vtype_c vj;
                    opus::static_for<T::COM_REP_M>([&](auto im_c) {
                        constexpr int im = decltype(im_c)::value;
                        opus::static_for<C_LEN>([&](auto e_c) {
                            constexpr int e = decltype(e_c)::value;
                            vj[im * C_LEN + e] =
                                v_c[(im * T::COM_REP_N + j) * C_LEN + e];
                        });
                    });
                    store<T::VEC_C>(g, vj, u_gc1,
                                    (wave_id_n_cons * T::COM_REP_N + j) * T::W_N);
                });
            } else {
                store<T::VEC_C>(g, v_c, u_gc, 0);
            }
        };
        if constexpr (!std::is_void_v<D_OUT>) {
            if (kargs.split_k == 1) {
                D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                               + (size_t)batch_id * kargs.stride_c_batch
                               + (size_t)row * kargs.stride_c
                               + (size_t)col;
                auto g_out = make_gmem(out_ptr,
                    (unsigned int)rows_avail * (unsigned int)kargs.stride_c * sizeof(D_OUT));
                store_c(g_out);
            } else {
                D_C* ws_c_ptr = reinterpret_cast<D_C*>(kargs.ws_handle->ptr)
                              + (size_t)split_id * kargs.batch * kargs.stride_ws_batch
                              + (size_t)batch_id * kargs.stride_ws_batch
                              + (size_t)row * kargs.stride_ws
                              + (size_t)col;
                auto g_c = make_gmem(ws_c_ptr);
                store_c(g_c);
            }
        } else {
            D_C* ws_c_ptr = reinterpret_cast<D_C*>(kargs.ws_handle->ptr)
                          + (size_t)split_id * kargs.batch * kargs.stride_ws_batch
                          + (size_t)batch_id * kargs.stride_ws_batch
                          + (size_t)row * kargs.stride_ws
                          + (size_t)col;
            auto g_c = make_gmem(ws_c_ptr);
            store_c(g_c);
        }
    }

    if constexpr (!std::is_void_v<D_OUT>) {
        if (kargs.split_k == 1) return;

        __shared__ int fused_do_reduce;
        if (opus::thread_id_x() == 0) {
            fused_do_reduce = 0;
        }
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __builtin_amdgcn_s_barrier();

        int* counters = reinterpret_cast<int*>(
            reinterpret_cast<char*>(kargs.ws_handle->ptr) + kargs.counter_offset_bytes);
        const int num_tiles = num_tiles_m * ceil_div(kargs.n, T::B_N);
        const int tile_id = batch_id * num_tiles + wgid;
        if (opus::thread_id_x() == 0) {
            const int old = __atomic_fetch_add(counters + tile_id, 1, __ATOMIC_ACQ_REL);
            fused_do_reduce = (old == kargs.split_k - 1);
        }
        // Every thread branches on this below, so lane 0's write has to be retired
        // (not merely issued) before the barrier lets the rest read it.
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        if (fused_do_reduce) {
            const D_C* ws_base = reinterpret_cast<const D_C*>(kargs.ws_handle->ptr);
            D_OUT* out = reinterpret_cast<D_OUT*>(kargs.ptr_c);
            const size_t split_stride = (size_t)kargs.batch * (size_t)kargs.stride_ws_batch;
            for (int i = int(opus::thread_id_x()); i < T::B_M * T::B_N; i += T::BLOCK_SIZE) {
                const int mi = i / T::B_N;
                const int ni = i - mi * T::B_N;
                if (row + mi >= kargs.m) continue;  // skip OOB rows of a partial M tile
                float acc = 0.0f;
                const size_t base = (size_t)batch_id * (size_t)kargs.stride_ws_batch
                                  + (size_t)(row + mi) * (size_t)kargs.stride_ws
                                  + (size_t)(col + ni);
                for (int s = 0; s < kargs.split_k; ++s) {
                    acc += static_cast<float>(ws_base[(size_t)s * split_stride + base]);
                }
                const size_t out_idx = (size_t)batch_id * (size_t)kargs.stride_c_batch
                                     + (size_t)(row + mi) * (size_t)kargs.stride_c
                                     + (size_t)(col + ni);
                out[out_idx] = static_cast<D_OUT>(acc);
            }
            __builtin_amdgcn_s_barrier();
            if (opus::thread_id_x() == 0) {
                counters[tile_id] = 0;
            }
        }
    }
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// Direct-store persistent M-outer kernel. Each WG owns one N tile and a small
// run of M tiles, reusing the same B tile stream across the outer loop without
// increasing the per-tile accumulator footprint.
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;

    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    const int m_per_wg = kargs.split_k;
    int bid = opus::block_id_x();
    constexpr int NUM_XCD = 8;
    int xcd_id = __builtin_amdgcn_readfirstlane(bid % NUM_XCD);
    int pos_xcd = __builtin_amdgcn_readfirstlane(bid / NUM_XCD);
    int tile_n_id = __builtin_amdgcn_readfirstlane(pos_xcd % num_tiles_n);
    int m_grp_local = __builtin_amdgcn_readfirstlane(pos_xcd / num_tiles_n);
    int m_grp = __builtin_amdgcn_readfirstlane(xcd_id * kargs.stride_ws_batch + m_grp_local);
    if (m_grp >= kargs.stride_ws) return;
    int tile_m_lo = m_grp * m_per_wg;
    int tile_m_hi = tile_m_lo + m_per_wg;
    int col = tile_n_id * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    int role = ((wave_id & 1) ^ ((bid >> 8) & 1));

    const int loops = kargs.k / T::B_K;
    if (loops < T::prefetch_k_iter) return;

    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch + (size_t)col * kargs.stride_b);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb);

    __shared__ char smem_a[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BN
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int slot_k, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot_k, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    constexpr int mb_a = T::a_buffer_load_insts;
    constexpr int mb_b = T::b_buffer_load_insts;
    constexpr int mb = mb_a + mb_b;

    if (role == 0) {
        int wave_id_prod = wave_id / 2;
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);
        auto u_gb = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);

        for (int tile_m = tile_m_lo; tile_m < tile_m_hi && tile_m < num_tiles_m; ++tile_m) {
            int row = tile_m * T::B_M;
            auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                                 + (size_t)batch_id * kargs.stride_a_batch
                                 + (size_t)row * kargs.stride_a);
            auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
                return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
                     + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
            };

            opus::static_for<T::prefetch_k_iter>([&](auto p_c) {
                constexpr int p = decltype(p_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(p, m, kg), u_ga, u_sa, a_offset(p, m, kg));
                    });
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                        constexpr int n = decltype(n_c)::value;
                        async_load<T::VEC_B>(g_b, smem_b_at(p, n, kg), u_gb, u_sb, b_offset(p, n, kg));
                    });
                });
            });

            opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
                constexpr int p = T::prefetch_k_iter - 1 - decltype(i_c)::value;
                s_waitcnt_vmcnt(number<mb * p>{});
                __builtin_amdgcn_s_barrier();
            });

            if constexpr (T::prefetch_k_iter == 3) {
                s_waitcnt_vmcnt(number<mb>{});
                __builtin_amdgcn_s_barrier();
                for (int i = T::prefetch_k_iter - 1; i < loops - 1; i++) {
                    int issue_k = i + 1;
                    int slot = issue_k % T::prefetch_k_iter;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                            constexpr int m = decltype(m_c)::value;
                            async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                        });
                        opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                            constexpr int n = decltype(n_c)::value;
                            async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                        });
                    });
                    s_waitcnt_vmcnt(number<mb>{});
                    __builtin_amdgcn_s_barrier();
                }
                s_waitcnt_vmcnt(0_I);
                __builtin_amdgcn_s_barrier();
            } else {
                for (int i = T::prefetch_k_iter - 2; i < loops - 2; i++) {
                    int issue_k = i + 2;
                    int slot = issue_k % T::prefetch_k_iter;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                            constexpr int m = decltype(m_c)::value;
                            async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                        });
                        opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                            constexpr int n = decltype(n_c)::value;
                            async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                        });
                    });
                    s_waitcnt_vmcnt(number<2 * mb>{});
                    __builtin_amdgcn_s_barrier();
                }
                s_waitcnt_vmcnt(number<mb>{});
                __builtin_amdgcn_s_barrier();
                s_waitcnt_vmcnt(0_I);
                __builtin_amdgcn_s_barrier();
            }
            __builtin_amdgcn_s_barrier();
        }
    } else {
        int wave_id_m = wave_id / 2;
        int wave_id_n_cons = 0;
        auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_mxsk<T>(lane_id);

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        typename decltype(mma)::vtype_a v_a0, v_a1;
        typename decltype(mma)::vtype_b v_b0, v_b1;
        typename decltype(mma)::vtype_c v_c;

        using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
        using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;
        constexpr int ds_read_insts = T::a_ds_read_insts + T::b_ds_read_insts;

        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);

        for (int tile_m = tile_m_lo; tile_m < tile_m_hi && tile_m < num_tiles_m; ++tile_m) {
            int row = tile_m * T::B_M;
            auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                                   + (size_t)batch_id * kargs.stride_sfa_batch
                                   + (size_t)row * kargs.stride_sfa);
            clear(v_c);

            auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);
            auto scaled_mma = [&](const auto& va, const auto& vb, int loop_k) {
                const int scale_base = loop_k * T::SCALES_PER_BK;
                vtype_sfa v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
                vtype_sfb v_sfb;
                opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
                    constexpr int ng = decltype(ng_c)::value;
                    auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
                    opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                    });
                });
                if constexpr (!SKIP_SCALE_WAIT) {
                    s_waitcnt_vmcnt(0_I);
                }
                __builtin_amdgcn_s_setprio(1);
                mma_mxscale_flatmm_accum<T>(mma, va, vb, v_sfa, v_sfb, v_c);
                __builtin_amdgcn_s_setprio(0);
            };

            __builtin_amdgcn_s_barrier();
            {
                auto sa0 = make_smem(smem_a_at(0, 0, 0));
                auto sb0 = make_smem(smem_b_at(0, 0, 0));
                v_a0 = load<T::VEC_A>(sa0, u_ra);
                v_b0 = load<T::VEC_B>(sb0, u_rb);
            }

            opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
                constexpr int p = decltype(i_c)::value + 1;
                constexpr int cur = (p - 1) & 1;
                constexpr int nxt = p & 1;
                __builtin_amdgcn_s_barrier();
                auto sa_p = make_smem(smem_a_at(p, 0, 0));
                auto sb_p = make_smem(smem_b_at(p, 0, 0));
                if constexpr (nxt == 0) {
                    v_a0 = load<T::VEC_A>(sa_p, u_ra);
                    v_b0 = load<T::VEC_B>(sb_p, u_rb);
                } else {
                    v_a1 = load<T::VEC_A>(sa_p, u_ra);
                    v_b1 = load<T::VEC_B>(sb_p, u_rb);
                }
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                if constexpr (cur == 0) scaled_mma(v_a0, v_b0, p - 1);
                else                    scaled_mma(v_a1, v_b1, p - 1);
            });

            constexpr int L = (T::prefetch_k_iter - 2) & 1;
            int k = T::prefetch_k_iter - 1;
            for (; k + 1 < loops - 1; k += 2) {
                __builtin_amdgcn_s_barrier();
                {
                    int slot = k % T::prefetch_k_iter;
                    auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                    auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                    if constexpr (L == 0) {
                        v_a1 = load<T::VEC_A>(sa_k, u_ra);
                        v_b1 = load<T::VEC_B>(sb_k, u_rb);
                    } else {
                        v_a0 = load<T::VEC_A>(sa_k, u_ra);
                        v_b0 = load<T::VEC_B>(sb_k, u_rb);
                    }
                }
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                if constexpr (L == 0) scaled_mma(v_a0, v_b0, k - 1);
                else                  scaled_mma(v_a1, v_b1, k - 1);

                __builtin_amdgcn_s_barrier();
                {
                    int slot = (k + 1) % T::prefetch_k_iter;
                    auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                    auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                    if constexpr (L == 0) {
                        v_a0 = load<T::VEC_A>(sa_k, u_ra);
                        v_b0 = load<T::VEC_B>(sb_k, u_rb);
                    } else {
                        v_a1 = load<T::VEC_A>(sa_k, u_ra);
                        v_b1 = load<T::VEC_B>(sb_k, u_rb);
                    }
                }
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                if constexpr (L == 0) scaled_mma(v_a1, v_b1, k);
                else                  scaled_mma(v_a0, v_b0, k);
            }

            bool last_in_buf1 = (L != 0);
            if (k < loops - 1) {
                __builtin_amdgcn_s_barrier();
                {
                    int slot = k % T::prefetch_k_iter;
                    auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                    auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                    if constexpr (L == 0) {
                        v_a1 = load<T::VEC_A>(sa_k, u_ra);
                        v_b1 = load<T::VEC_B>(sb_k, u_rb);
                    } else {
                        v_a0 = load<T::VEC_A>(sa_k, u_ra);
                        v_b0 = load<T::VEC_B>(sb_k, u_rb);
                    }
                }
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                if constexpr (L == 0) scaled_mma(v_a0, v_b0, k - 1);
                else                  scaled_mma(v_a1, v_b1, k - 1);
                last_in_buf1 = (L == 0);
                k++;
            }

            __builtin_amdgcn_s_barrier();
            int last_slot = (loops - 1) % T::prefetch_k_iter;
            auto sa_last = make_smem(smem_a_at(last_slot, 0, 0));
            auto sb_last = make_smem(smem_b_at(last_slot, 0, 0));
            if (last_in_buf1) {
                v_a0 = load<T::VEC_A>(sa_last, u_ra);
                v_b0 = load<T::VEC_B>(sb_last, u_rb);
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                scaled_mma(v_a1, v_b1, loops - 2);
                s_waitcnt_lgkmcnt(0_I);
                scaled_mma(v_a0, v_b0, loops - 1);
            } else {
                v_a1 = load<T::VEC_A>(sa_last, u_ra);
                v_b1 = load<T::VEC_B>(sb_last, u_rb);
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                scaled_mma(v_a0, v_b0, loops - 2);
                s_waitcnt_lgkmcnt(0_I);
                scaled_mma(v_a1, v_b1, loops - 1);
            }

            D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                           + (size_t)batch_id * kargs.stride_c_batch
                           + (size_t)row * kargs.stride_c
                           + (size_t)col;
            auto g_out = make_gmem(out_ptr);
            store<T::VEC_C>(g_out, v_c, u_gc, 0);
            __builtin_amdgcn_s_barrier();
        }
    }
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// -- M-tile interleaved direct-store kernel (correctness-first v1) ------------
//
// Each WG owns one N tile and MI=2 consecutive M tiles that share the SAME B
// tile stream. Both M tiles' A operands are resident in LDS simultaneously
// (smem_a0 / smem_a1) while B is loaded once (smem_b). Per K iteration the
// consumer issues MFMA for tile0 then tile1 back-to-back into two independent
// accumulators, so the MFMA instruction stream is ~MIx longer across a single
// prologue/epilogue, and B global traffic is halved.
//
// v1 is single-buffered (load-then-compute, barrier-bracketed) for provable
// deadlock-freedom and correctness; the software-pipelined perf version is a
// follow-up. Producer waves (role 0) load; consumer waves (role 1) MFMA.
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a8w8_mxscale_flatmm_minterleave_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;

    constexpr int MI = 2;                 // M tiles interleaved per WG
    constexpr int SB = 2;                 // double LDS buffer per operand (prefetch k+1)

    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    int bid = opus::block_id_x();
    constexpr int NUM_XCD = 8;
    int xcd_id = __builtin_amdgcn_readfirstlane(bid % NUM_XCD);
    int pos_xcd = __builtin_amdgcn_readfirstlane(bid / NUM_XCD);
    int tile_n_id = __builtin_amdgcn_readfirstlane(pos_xcd % num_tiles_n);
    int m_grp_local = __builtin_amdgcn_readfirstlane(pos_xcd / num_tiles_n);
    int m_grp = __builtin_amdgcn_readfirstlane(xcd_id * kargs.stride_ws_batch + m_grp_local);
    if (m_grp >= kargs.stride_ws) return;
    int tile_m0 = m_grp * MI;
    if (tile_m0 + MI > num_tiles_m) return;  // host guarantees M % (MI*B_M) == 0
    int col = tile_n_id * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    int role = (wave_id & 1);

    const int loops = kargs.k / T::B_K;
    if (loops < T::prefetch_k_iter) return;

    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch + (size_t)col * kargs.stride_b);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb);

    __shared__ char smem_a[MI * SB * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[SB * T::NUM_LOAD_GROUPS_PER_BN
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int mi, int slot_k, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + (((mi * SB + slot_k) * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot_k, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    if (role == 0) {
        int wave_id_prod = wave_id / 2;
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);
        auto u_gb = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);

        auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
            return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
                 + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
        };

        const D_A* base_a = reinterpret_cast<const D_A*>(kargs.ptr_a)
                            + (size_t)batch_id * kargs.stride_a_batch;

        auto issue_loads = [&](int k, int slot) {
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<MI>([&](auto mi_c) {
                    constexpr int mi = decltype(mi_c)::value;
                    auto g_a = make_gmem(base_a + (size_t)(tile_m0 + mi) * T::B_M * kargs.stride_a);
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(mi, slot, m, kg), u_ga, u_sa, a_offset(k, m, kg));
                    });
                });
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(k, n, kg));
                });
            });
        };

        // Prologue: preload slot 0 (K-tile 0), then stream while consumers MFMA.
        issue_loads(0, 0);
        s_waitcnt_vmcnt(0_I);
        for (int k = 0; k < loops; ++k) {
            __builtin_amdgcn_s_barrier();   // R(k): slot k ready
            if (k + 1 < loops) {
                issue_loads(k + 1, (k + 1) % SB);
                s_waitcnt_vmcnt(0_I);       // slot k+1 fully loaded before releasing slot k
            }
            __builtin_amdgcn_s_barrier();   // F(k): consumer done reading slot k
        }
    } else {
        int wave_id_m = wave_id / 2;
        int wave_id_n_cons = 0;
        auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_mxsk<T>(lane_id);

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        typename decltype(mma)::vtype_a v_a[MI];
        typename decltype(mma)::vtype_b v_b;
        typename decltype(mma)::vtype_c v_c[MI];

        using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
        using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;

        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);

        // Per-tile A-scale gmem + layout.
        auto g_sfa = [&](int mi) {
            return make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                             + (size_t)batch_id * kargs.stride_sfa_batch
                             + (size_t)(tile_m0 + mi) * T::B_M * kargs.stride_sfa);
        };
        auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

        opus::static_for<MI>([&](auto mi_c) { clear(v_c[decltype(mi_c)::value]); });

        for (int k = 0; k < loops; ++k) {
            __builtin_amdgcn_s_barrier();   // R(k): wait producer data ready
            const int slot = k % SB;
            auto sb0 = make_smem(smem_b_at(slot, 0, 0));
            v_b = load<T::VEC_B>(sb0, u_rb);
            opus::static_for<MI>([&](auto mi_c) {
                constexpr int mi = decltype(mi_c)::value;
                auto sa = make_smem(smem_a_at(mi, slot, 0, 0));
                v_a[mi] = load<T::VEC_A>(sa, u_ra);
            });
            s_waitcnt_lgkmcnt(0_I);

            const int scale_base = k * T::SCALES_PER_BK;
            vtype_sfb v_sfb;
            opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
                constexpr int ng = decltype(ng_c)::value;
                auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
                opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                });
            });
            opus::static_for<MI>([&](auto mi_c) {
                constexpr int mi = decltype(mi_c)::value;
                auto gsfa = g_sfa(mi);
                vtype_sfa v_sfa = load<T::SCALES_PER_BK>(gsfa, u_sfa, scale_base);
                if constexpr (!SKIP_SCALE_WAIT) s_waitcnt_vmcnt(0_I);
                __builtin_amdgcn_s_setprio(1);
                mma_mxscale_flatmm_accum<T>(mma, v_a[mi], v_b, v_sfa, v_sfb, v_c[mi]);
                __builtin_amdgcn_s_setprio(0);
            });
            __builtin_amdgcn_s_barrier();   // (C2) done reading slot
        }

        opus::static_for<MI>([&](auto mi_c) {
            constexpr int mi = decltype(mi_c)::value;
            int row = (tile_m0 + mi) * T::B_M;
            D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                           + (size_t)batch_id * kargs.stride_c_batch
                           + (size_t)row * kargs.stride_c
                           + (size_t)col;
            auto g_out = make_gmem(out_ptr);
            store<T::VEC_C>(g_out, v_c[mi], u_gc, 0);
        });
    }
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// 8-wave split-accumulator direct-store kernel.
//
// Logical tile: 128x256x128. Internally this is two independent 128x128
// accumulator groups computed by consumer waves {4,5} and {6,7}. Producer
// waves {0,1} load shared A and B phase 0, producer waves {2,3} load B phase 1.
// This keeps each consumer's v_c identical to the proven 128x128 WG1 kernel
// while halving the logical N workgroup count.
template<typename Traits, typename D_OUT>
__global__ __launch_bounds__(512, 1)
void gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;
    static_assert(T::B_M == 128 && T::B_N == 128 && T::B_K == 128,
                  "wave8n2 builds a logical 128x256 tile from 128x128 traits");

    constexpr int N_PHASES = 2;
    int wgid = opus::block_id_x();
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col_base = (wgid / num_tiles_m) * (T::B_N * N_PHASES);
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    const int loops = kargs.k / T::B_K;
    if (loops < 1) return;

    constexpr int WAVE8N2_PREFETCH_SLOTS = 2;
    __shared__ char smem_a[WAVE8N2_PREFETCH_SLOTS * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[WAVE8N2_PREFETCH_SLOTS * N_PHASES * T::NUM_LOAD_GROUPS_PER_BN
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int slot, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot, int phase, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + (((slot * N_PHASES + phase) * T::NUM_LOAD_GROUPS_PER_BN + n_block)
               * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    if (wave_id < 4) {
        const int phase = wave_id / 2;
        const int prod_wave = wave_id & 1;
        auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                             + (size_t)batch_id * kargs.stride_a_batch
                             + (size_t)row * kargs.stride_a);
        auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                             + (size_t)batch_id * kargs.stride_b_batch
                             + (size_t)(col_base + phase * T::B_N) * kargs.stride_b);
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, prod_wave, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, prod_wave);
        auto u_gb = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, prod_wave, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, prod_wave);

        auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
            return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
                 + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
        };
        auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
            return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
                 + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
        };

        auto issue_tile = [&](int loop_k) {
            const int slot = loop_k & 1;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                if (phase == 0) {
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(loop_k, m, kg));
                    });
                }
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, phase, n, kg), u_gb, u_sb, b_offset(loop_k, n, kg));
                });
            });
        };

        issue_tile(0);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        for (int k = 0; k < loops; ++k) {
            if (k + 1 < loops) {
                issue_tile(k + 1);
            }
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        return;
    }

    const int consumer = wave_id - 4;
    const int phase = consumer / 2;
    const int wave_id_m = consumer & 1;
    const int col = col_base + phase * T::B_N;

    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + (size_t)batch_id * kargs.stride_sfa_batch
                           + (size_t)row * kargs.stride_sfa);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb);

    auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
    auto u_rb = make_layout_rb_mxsk<T>(lane_id);
    auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);
    using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
    using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;

    __builtin_amdgcn_s_barrier();
    {
        auto sa = make_smem(smem_a_at(0, 0, 0));
        auto sb = make_smem(smem_b_at(0, phase, 0, 0));
        v_a = load<T::VEC_A>(sa, u_ra);
        v_b = load<T::VEC_B>(sb, u_rb);
        s_waitcnt_lgkmcnt(0_I);
    }

    for (int k = 0; k < loops; ++k) {
        const int scale_base = k * T::SCALES_PER_BK;
        vtype_sfa v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
        vtype_sfb v_sfb;
        opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
            opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
            });
        });
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_mxscale_flatmm_accum<T>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        if (k + 1 < loops) {
            const int slot = (k + 1) & 1;
            auto sa = make_smem(smem_a_at(slot, 0, 0));
            auto sb = make_smem(smem_b_at(slot, phase, 0, 0));
            v_a = load<T::VEC_A>(sa, u_ra);
            v_b = load<T::VEC_B>(sb, u_rb);
            s_waitcnt_lgkmcnt(0_I);
        }
    }

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      0, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma,
        opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                   + (size_t)batch_id * kargs.stride_c_batch
                   + (size_t)row * kargs.stride_c
                   + (size_t)col;
    auto g_out = make_gmem(out_ptr);
    store<T::VEC_C>(g_out, v_c, u_gc, 0);
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// 4-wave self-load split-accumulator direct-store kernel with M reuse.
//
// Logical tile: 256x128x128. Two independent 128x128 accumulator groups cover
// adjacent M tiles and share one B tile. This targets large-M shapes where B
// reuse matters more than reducing N workgroups.
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT = false,
         bool PACK_SCALE_ON_DEMAND = false>
__global__ __launch_bounds__(256, 1)
void gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;
    static_assert(T::B_M == 128 && T::B_N == 128 && T::B_K == 128,
                  "wave4m2 selfload builds a logical 256x128 tile from 128x128 traits");

    constexpr int M_PHASES = 2;
    constexpr int PREFETCH_SLOTS = 2;
    int wgid = opus::block_id_x();
    const int logical_b_m = T::B_M * M_PHASES;
    const int num_tiles_m = ceil_div(kargs.m, logical_b_m);
    int row_base = (wgid % num_tiles_m) * logical_b_m;
    int col = (wgid / num_tiles_m) * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    const int loops = kargs.k / T::B_K;
    if (loops < 1) return;

    const int m_phase = wave_id / 2;
    const int wave_id_m = wave_id & 1;
    const int row = row_base + m_phase * T::B_M;

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + (size_t)batch_id * kargs.stride_a_batch
                         + (size_t)row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch
                         + (size_t)col * kargs.stride_b);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + (size_t)batch_id * kargs.stride_sfa_batch
                           + (size_t)row * kargs.stride_sfa);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb);

    __shared__ char smem_a[PREFETCH_SLOTS * M_PHASES * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[PREFETCH_SLOTS * T::NUM_LOAD_GROUPS_PER_BN
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int slot, int phase, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + (((slot * M_PHASES + phase) * T::NUM_LOAD_GROUPS_PER_BM + m_block)
               * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };
    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_m, kargs.stride_a);
    auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_m);
    auto u_gb = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_m, kargs.stride_b);
    auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_m);
    auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
    auto u_rb = make_layout_rb_mxsk<T>(lane_id);
    auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);
    using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
    using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;

    auto issue_tile = [&](int loop_k) {
        const int slot = loop_k & 1;
        opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
            constexpr int kg = decltype(kg_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                constexpr int m = decltype(m_c)::value;
                async_load<T::VEC_A>(g_a, smem_a_at(slot, m_phase, m, kg), u_ga, u_sa, a_offset(loop_k, m, kg));
            });
            if (m_phase == 0) {
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(loop_k, n, kg));
                });
            }
        });
    };

    auto load_scales = [&](int loop_k, vtype_sfa& v_sfa, vtype_sfb& v_sfb) {
        const int scale_base = loop_k * T::SCALES_PER_BK;
        v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
        opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
            opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
            });
        });
        if constexpr (!SKIP_SCALE_WAIT) {
            s_waitcnt_vmcnt(0_I);
        }
    };

    issue_tile(0);
    s_waitcnt_vmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    {
        auto sa = make_smem(smem_a_at(0, m_phase, 0, 0));
        auto sb = make_smem(smem_b_at(0, 0, 0));
        v_a = load<T::VEC_A>(sa, u_ra);
        v_b = load<T::VEC_B>(sb, u_rb);
        s_waitcnt_lgkmcnt(0_I);
    }

    for (int k = 0; k < loops; ++k) {
        vtype_sfa v_sfa;
        vtype_sfb v_sfb;
        load_scales(k, v_sfa, v_sfb);
        if (k + 1 < loops) {
            issue_tile(k + 1);
        }
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PACK_SCALE_ON_DEMAND) {
            mma_mxscale_flatmm_accum_on_demand<T>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
        } else {
            mma_mxscale_flatmm_accum<T>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
        }
        __builtin_amdgcn_s_setprio(0);
        if (k + 1 < loops) {
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            const int slot = (k + 1) & 1;
            auto sa = make_smem(smem_a_at(slot, m_phase, 0, 0));
            auto sb = make_smem(smem_b_at(slot, 0, 0));
            v_a = load<T::VEC_A>(sa, u_ra);
            v_b = load<T::VEC_B>(sb, u_rb);
            s_waitcnt_lgkmcnt(0_I);
        }
    }

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      0, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma,
        opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                   + (size_t)batch_id * kargs.stride_c_batch
                   + (size_t)row * kargs.stride_c
                   + (size_t)col;
    auto g_out = make_gmem(out_ptr);
    store<T::VEC_C>(g_out, v_c, u_gc, 0);
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// wave4m2_selfload with B read straight into the MFMA registers.
//
// Same logical 256x128x128 tile and wave map as the kernel above: waves 0/1 and
// 2/3 own adjacent 128x128 accumulators over one shared B column range, and
// every wave stages its own half of its own M phase's A tile. The one change is
// where B comes from. The preshuffled weight already is the fragment order, so
// each wave buffer_loads its B operand from global instead of going through
// LDS, which drops the B async copies, the B ds_reads and the B LDS buffers --
// LDS is the two A phases alone, and the tile barrier now only ever protects A.
//
// B is double buffered in registers and tile k+1's copy is issued before the
// MMA that consumes tile k, so its latency hides behind 32 MFMA rather than
// behind the barrier. vmcnt is in-order and each iteration issues, in order:
// the scale loads, this wave's a_buffer_load_insts async copies for k+1, then
// b_direct_load_insts dwordx4s for k+1. Leaving NA+NB outstanding therefore
// retires exactly the scales plus B(k) (issued one iteration earlier), and the
// tail's NB retires the async copies while B(k+1) stays in flight.
//
// The cost is B read amplification: all four waves want the same 128 columns
// and there is no LDS to share them through, so a K tile pulls 4x16 KiB of B
// through L1 instead of staging 16 KiB once. That is the trade this kid exists
// to measure.
// Waves per EU comes from the traits rather than being pinned at 1 like the
// LDS-staged sibling, since dropping B leaves 66 KiB and two workgroups do fit
// the 160 KiB CU. Only WG_PER_CU=1 is shipped: at 2 the wave is capped at half
// the register file, which the 128-register accumulator fills on its own (see
// the note next to the tiles in opus_gemm_common.py).
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT = false,
         bool PACK_SCALE_ON_DEMAND = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a8w8_mxscale_flatmm_splitk_wave4m2_bdirect_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;
    static_assert(T::B_M == 128 && T::B_N == 128 && T::B_K == 128,
                  "wave4m2 bdirect builds a logical 256x128 tile from 128x128 traits");
    static_assert(T::B_DIRECT_REG && T::B_PRESHUFFLE,
                  "wave4m2 bdirect needs the direct-to-register preshuffled-B traits");
    static_assert(T::T_N == 1,
                  "the four waves share one N range, so every wave's B base block is 0");
    static_assert(!SKIP_SCALE_WAIT && !PACK_SCALE_ON_DEMAND,
                  "the explicit vmcnt schedule owns the scale wait; the family's two "
                  "bool axes are carried only to share the wave4m2 launcher");

    constexpr int M_PHASES = 2;
    constexpr int PREFETCH_SLOTS = 2;
    int wgid = opus::block_id_x();
    const int logical_b_m = T::B_M * M_PHASES;
    const int num_tiles_m = ceil_div(kargs.m, logical_b_m);
    int row_base = (wgid % num_tiles_m) * logical_b_m;
    int col = (wgid / num_tiles_m) * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    const int loops = kargs.k / T::B_K;
    if (loops < 1) return;

    const int m_phase = wave_id / 2;
    const int wave_id_m = wave_id & 1;
    const int row = row_base + m_phase * T::B_M;

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + (size_t)batch_id * kargs.stride_a_batch
                         + (size_t)row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch
                         + b_gmem_tile_base_mxsk<T>(col, 0, kargs.stride_b));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + (size_t)batch_id * kargs.stride_sfa_batch
                           + (size_t)row * kargs.stride_sfa);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb);

    __shared__ char smem_a[PREFETCH_SLOTS * M_PHASES * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int slot, int phase, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + (((slot * M_PHASES + phase) * T::NUM_LOAD_GROUPS_PER_BM + m_block)
               * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_m, kargs.stride_a);
    auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_m);
    auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gmem_b_direct_mxsk<T>(lane_id, kargs.stride_b, 0);
    auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b0, v_b1;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);
    using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
    using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;

    // Async copies and direct-B dwordx4s per wave per K tile: the two vmcnt
    // immediates the schedule below is built on.
    constexpr int NA = T::a_buffer_load_insts;
    constexpr int NB = T::b_direct_load_insts;

    auto issue_a_tile = [&](int loop_k) {
        const int slot = loop_k & 1;
        opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
            constexpr int kg = decltype(kg_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                constexpr int m = decltype(m_c)::value;
                async_load<T::VEC_A>(g_a, smem_a_at(slot, m_phase, m, kg), u_ga, u_sa,
                                     a_offset(loop_k, m, kg));
            });
        });
    };

    auto issue_b = [&](auto& vb, int loop_k) {
        vb = load<T::VEC_B>(g_b, u_gb, b_direct_iter_offset_mxsk<T>(loop_k));
    };

    auto load_scales = [&](int loop_k, vtype_sfa& v_sfa, vtype_sfb& v_sfb) {
        const int scale_base = loop_k * T::SCALES_PER_BK;
        v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
        opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
            opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
            });
        });
    };

    auto read_a = [&](int loop_k) {
        auto sa = make_smem(smem_a_at(loop_k & 1, m_phase, 0, 0));
        v_a = load<T::VEC_A>(sa, u_ra);
    };

    issue_a_tile(0);
    issue_b(v_b0, 0);
    s_waitcnt_vmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_a(0);
    s_waitcnt_lgkmcnt(0_I);

    // MMA tile k out of vb_cur while A(k+1) lands in the other LDS slot and
    // B(k+1) lands in vb_nxt. Only ever called with k+1 < loops.
    auto step = [&](int k, auto& vb_cur, auto& vb_nxt) {
        vtype_sfa v_sfa;
        vtype_sfb v_sfb;
        load_scales(k, v_sfa, v_sfb);
        issue_a_tile(k + 1);
        issue_b(vb_nxt, k + 1);
        s_waitcnt_vmcnt(number<NA + NB>{});
        __builtin_amdgcn_s_setprio(1);
        mma_mxscale_flatmm_accum<T>(mma, v_a, vb_cur, v_sfa, v_sfb, v_c);
        __builtin_amdgcn_s_setprio(0);
        s_waitcnt_vmcnt(number<NB>{});
        __builtin_amdgcn_s_barrier();
        read_a(k + 1);
        s_waitcnt_lgkmcnt(0_I);
    };

    auto last_step = [&](int k, auto& vb_cur) {
        vtype_sfa v_sfa;
        vtype_sfb v_sfb;
        load_scales(k, v_sfa, v_sfb);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_mxscale_flatmm_accum<T>(mma, v_a, vb_cur, v_sfa, v_sfb, v_c);
        __builtin_amdgcn_s_setprio(0);
    };

    // Unrolled by two so the register B double buffer alternates at compile
    // time; the odd remainder and the last tile are peeled.
    int k = 0;
    for (; k + 2 <= loops - 1; k += 2) {
        step(k, v_b0, v_b1);
        step(k + 1, v_b1, v_b0);
    }
    if (k < loops - 1) {
        step(k, v_b0, v_b1);
        last_step(k + 1, v_b1);
    } else {
        last_step(k, v_b0);
    }

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      0, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma,
        opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                   + (size_t)batch_id * kargs.stride_c_batch
                   + (size_t)row * kargs.stride_c
                   + (size_t)col;
    auto g_out = make_gmem(out_ptr);
    store<T::VEC_C>(g_out, v_c, u_gc, 0);
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// All-wave 2x2 direct-B: one 128x128x128 tile, four waves arranged as a 2x2
// (M,N) grid, A double-buffered through 32 KiB of LDS and B read straight from
// the preshuffled weight into registers.
//
// This is the wave4m2 direct-B schedule with the wave grid changed, and it is a
// measured negative result -- kept as the control that rules occupancy out. See
// the kid 189/197 notes in opus_gemm_common.py for the numbers.
//
// The premise was that wave4m2 is stuck at one wave per SIMD: it gives each wave
// a 64x128 slice of a doubled M range, so the accumulator is 128 registers and
// each of the two B buffers is 64, it measures 232 VGPRs, and asking for two
// waves per EU makes it spill. Covering 128x128 as a 2x2 grid instead quarters
// the tile per wave, which halves both (64 registers of accumulator, 32 per B
// buffer), and that worked: 120 VGPRs, no scratch, 33 KiB of LDS. It bought no
// performance at all, and the WG_PER_CU=1 and 2 kids land within 0.2% of each
// other, so latency hiding was not the constraint.
//
// What it cost is L2 locality: halving the rows per workgroup doubles how often
// the same B columns are re-read, and HBM traffic nearly doubles with it (19.2 GB
// against wave4m2's 10.2 GB, L2 hit rate 57.9% against 70.1%). The tile also runs
// 16 MFMAs per wave per K tile instead of 32, so the per-tile barrier and scale
// overhead is amortised over half as much work.
//
// vmcnt schedule. One in-order counter carries three streams -- the wave's
// quarter of the A tile (NA copies), its own B tile (NB dwordx4s) and the two
// scale loads -- so the two immediates below follow from issue order, as in
// wave4m2 direct-B. Per iteration the stream is [B(k)] [scales(k)] [A(k+1)]
// [B(k+1)]: vmcnt(NA+NB) retires B(k) and the scales while keeping the next
// tile of both in flight, then vmcnt(NB) retires A(k+1) into LDS before the
// barrier that publishes it. The A lead is one tile, which is why two LDS slots
// are enough.
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT = false,
         bool PACK_SCALE_ON_DEMAND = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a8w8_mxscale_flatmm_splitk_allwave_bdirect_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;
    static_assert(T::B_M == 128 && (T::B_N == 128 || T::B_N == 256) && T::B_K == 128,
                  "all-wave bdirect is built for the 128-row tile at 128 or 256 columns");
    static_assert(T::B_DIRECT_REG && T::B_PRESHUFFLE,
                  "all-wave bdirect needs the direct-to-register preshuffled-B traits");
    static_assert(T::ALL_WAVE && T::T_M == 2 && T::T_N == 2,
                  "the four waves form a 2x2 (M,N) grid over one tile");
    static_assert(!SKIP_SCALE_WAIT && !PACK_SCALE_ON_DEMAND,
                  "the explicit vmcnt schedule owns the scale wait; the family's two "
                  "bool axes are carried only to share the launcher");

    constexpr int PREFETCH_SLOTS = 2;
    int wgid = opus::block_id_x();
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    const int row = (wgid % num_tiles_m) * T::B_M;
    const int col = (wgid / num_tiles_m) * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    const int loops = kargs.k / T::B_K;
    if (loops < 1) return;

    // 2x2 wave grid, M on the low bit so the two waves sharing an N range are
    // adjacent -- the same mapping the LDS all-wave path uses.
    const int wave_id_m = wave_id & 1;
    const int wave_id_n = wave_id >> 1;
    // First B scale group of this N-wave's columns. Zero at 128 columns (both
    // N-waves share group 0, stride 0); wave_id_n at 256 (stride 1).
    const int sfb_group_base = T::SFB_PER_WAVE ? wave_id_n * T::SFB_GROUP_STRIDE : 0;

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + (size_t)batch_id * kargs.stride_a_batch
                         + (size_t)row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch
                         + b_gmem_tile_base_mxsk<T>(col, 0, kargs.stride_b));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + (size_t)batch_id * kargs.stride_sfa_batch
                           + (size_t)row * kargs.stride_sfa);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N + sfb_group_base)
                             * kargs.stride_sfb);

    // A only, and one 128-row tile rather than wave4m2's two M phases: 32 KiB.
    __shared__ char smem_a[PREFETCH_SLOTS * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int slot, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot * T::NUM_LOAD_GROUPS_PER_BM + m_block)
               * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    // All four waves stage every A copy cooperatively (LOAD_WAVES=4), so the
    // group-load layouts are keyed on wave_id, not on the M coordinate.
    auto u_ga = make_layout_gmem_group_load_mxsk<T, T::LOAD_WAVES>(
        lane_id, wave_id, kargs.stride_a);
    auto u_sa = make_layout_smem_group_load_mxsk<T, T::LOAD_WAVES>(lane_id, wave_id);
    auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
    // nbc counts 16-column blocks, i.e. the wave's N-repeat span -- not the
    // load-group index the LDS path's nbc carries.
    auto u_gb = make_layout_gmem_b_direct_mxsk<T>(lane_id, kargs.stride_b,
                                                 wave_id_n * T::COM_REP_N);
    auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b0, v_b1;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);
    using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
    using vtype_sfb = vector_t<D_SF, T::SFB_GROUPS * T::SCALES_PER_BK>;

    constexpr int NA = T::a_buffer_load_insts;
    constexpr int NB = T::b_direct_load_insts;

    auto issue_a_tile = [&](int loop_k) {
        const int slot = loop_k & 1;
        opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
            constexpr int kg = decltype(kg_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                constexpr int m = decltype(m_c)::value;
                async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa,
                                     a_offset(loop_k, m, kg));
            });
        });
    };

    auto issue_b = [&](auto& vb, int loop_k) {
        vb = load<T::VEC_B>(g_b, u_gb, b_direct_iter_offset_mxsk<T>(loop_k));
    };

    // SFB_GROUPS is 1 and both N-waves fall in scale group 0 (see the traits),
    // so the B scale load is wave-invariant here.
    auto load_scales = [&](int loop_k, vtype_sfa& v_sfa, vtype_sfb& v_sfb) {
        const int scale_base = loop_k * T::SCALES_PER_BK;
        v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
        opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
            opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
            });
        });
    };

    auto read_a = [&](int loop_k) {
        auto sa = make_smem(smem_a_at(loop_k & 1, 0, 0));
        v_a = load<T::VEC_A>(sa, u_ra);
    };

    auto step = [&](int k, auto& vb_cur, auto& vb_nxt) {
        vtype_sfa v_sfa;
        vtype_sfb v_sfb;
        load_scales(k, v_sfa, v_sfb);
        issue_a_tile(k + 1);
        issue_b(vb_nxt, k + 1);
        s_waitcnt_vmcnt(number<NA + NB>{});
        __builtin_amdgcn_s_setprio(1);
        mma_mxscale_flatmm_accum<T>(mma, v_a, vb_cur, v_sfa, v_sfb, v_c);
        __builtin_amdgcn_s_setprio(0);
        s_waitcnt_vmcnt(number<NB>{});
        __builtin_amdgcn_s_barrier();
        read_a(k + 1);
        s_waitcnt_lgkmcnt(0_I);
    };

    auto last_step = [&](int k, auto& vb_cur) {
        vtype_sfa v_sfa;
        vtype_sfb v_sfb;
        load_scales(k, v_sfa, v_sfb);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_mxscale_flatmm_accum<T>(mma, v_a, vb_cur, v_sfa, v_sfb, v_c);
        __builtin_amdgcn_s_setprio(0);
    };

    // The 256-column tile has room for one B buffer, not two, so B(k) is issued
    // inside its own step. Order matters: B goes out ahead of A(k+1) so the
    // in-order vmcnt can retire B alone and leave A in flight. What is lost
    // against the double-buffered form is B(k+1) overlapping MMA(k) -- only the
    // scale loads sit in front of B's latency now.
    auto step_single = [&](int k, auto prefetch_a_c) {
        constexpr bool PF_A = decltype(prefetch_a_c)::value;
        vtype_sfa v_sfa;
        vtype_sfb v_sfb;
        load_scales(k, v_sfa, v_sfb);
        issue_b(v_b0, k);
        if constexpr (PF_A) {
            issue_a_tile(k + 1);
            s_waitcnt_vmcnt(number<NA>{});
        } else {
            s_waitcnt_vmcnt(0_I);
        }
        __builtin_amdgcn_s_setprio(1);
        mma_mxscale_flatmm_accum<T>(mma, v_a, v_b0, v_sfa, v_sfb, v_c);
        __builtin_amdgcn_s_setprio(0);
        if constexpr (PF_A) {
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            read_a(k + 1);
            s_waitcnt_lgkmcnt(0_I);
        }
    };

    if constexpr (T::B_REG_BUFS == 1) {
        issue_a_tile(0);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        read_a(0);
        s_waitcnt_lgkmcnt(0_I);
        for (int k = 0; k < loops - 1; ++k) step_single(k, std::true_type{});
        step_single(loops - 1, std::false_type{});
    } else {
        issue_a_tile(0);
        issue_b(v_b0, 0);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        read_a(0);
        s_waitcnt_lgkmcnt(0_I);

        // Unrolled by two so the register B double buffer alternates at compile
        // time; the odd remainder and the last tile are peeled.
        int k = 0;
        for (; k + 2 <= loops - 1; k += 2) {
            step(k, v_b0, v_b1);
            step(k + 1, v_b1, v_b0);
        }
        if (k < loops - 1) {
            step(k, v_b0, v_b1);
            last_step(k + 1, v_b1);
        } else {
            last_step(k, v_b0);
        }
    }

    // T_N=2 with COM_REP_N>1: the generic swap_ab C partition nests the register
    // N-repeat outside the wave's N tile, which interleaves the two N-waves'
    // output columns. Store each N-repeat to its own contiguous W_N columns with
    // a single-N-tile layout instead (the shared kernel's SPLIT_N_STORE path).
    constexpr int C_LEN = decltype(mma)::mma_c_len;
    auto mma_c1 = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::COM_REP_M, 1, T::COM_REP_K>{}, seq<T::T_M, 1, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    auto p_coord_c1 = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                       0, lane_id / mma.grpn_c);
    auto u_gc1 = partition_layout_c<T::VEC_C>(mma_c1,
        opus::make_tuple(kargs.stride_c, 1_I), p_coord_c1);
    D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                   + (size_t)batch_id * kargs.stride_c_batch
                   + (size_t)row * kargs.stride_c
                   + (size_t)col;
    auto g_out = make_gmem(out_ptr);
    opus::static_for<T::COM_REP_N>([&](auto j_c) {
        constexpr int j = decltype(j_c)::value;
        // The accumulator nests the m-repeat outside the n-repeat, so one
        // n-repeat's tiles sit COM_REP_N*C_LEN apart rather than contiguously.
        typename decltype(mma_c1)::vtype_c vj;
        opus::static_for<T::COM_REP_M>([&](auto im_c) {
            constexpr int im = decltype(im_c)::value;
            opus::static_for<C_LEN>([&](auto e_c) {
                constexpr int e = decltype(e_c)::value;
                vj[im * C_LEN + e] = v_c[(im * T::COM_REP_N + j) * C_LEN + e];
            });
        });
        store<T::VEC_C>(g_out, vj, u_gc1, (wave_id_n * T::COM_REP_N + j) * T::W_N);
    });
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

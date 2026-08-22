// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 fp8/e8m0 mxscale split-K pipeline: all-compute waves on a 256x256 tile,
// direct-to-register preshuffled B.
//
// The schedule below is written against T::WAVES rather than a fixed eight, so
// the traits can also aim it at a 4-wave grid on a half-height tile; the argument
// that follows is the 8-wave one it was designed for.
//
// "wave8" in this file's name, in gemm_a8w8_mxscale_bpreshuffle_wave8_kernel, in
// mma_mxscale_wave8_accum and in the base traits struct is therefore historical,
// not a wave count: T::WAVES follows BLOCK_SIZE, and the fastest kid this file
// has -- kid205 on the wavetm1 traits -- runs four waves on a 1x4 grid at
// BLOCK_SIZE=256. Nothing a caller sees says wave8 any more. The tag it was named
// after is gone with kid192 (see opus_gemm_common.py), and the two that remain
// name the grid rather than a count and are honest about it: wave8n4 is T_N=4 on
// eight waves, wavetm1 is T_M=1 on either eight or four. Both are aliases of the
// one base traits struct below, which is the schedule; what parameterizes it is
// the wave grid, T_N = WAVES / T_M.
//
// The 4-wave schedules in this directory all run 64 MFMA per workgroup per K
// tile: two of the four waves compute, 32 MFMA each. The hand-written f4gemm
// assembly runs 256, and every attempt to close that gap without changing the
// wave count has failed for the same reason -- the tile cannot grow. A 128x256
// tile (4 waves, all computing, 128 MFMA) measured +4% at M=8192 and -75% at
// M=1024, and the two constraints that stop it there are hard:
//
//   * Registers. The fp32 accumulator for a BxB tile is B*B/64 registers per
//     lane-slot however many waves hold it; 256x256 is 1024, half of the CU's
//     4 SIMDs x 512. Four waves would need 256 accumulator + 256 of double
//     buffered fp8 fragments per wave -- the entire per-wave file, with nothing
//     left for addressing. (The f4gemm assembly fits the same tile in 4 waves
//     only because fp4 halves every fragment.)
//   * LDS. Staging B costs (B_M + B_N)/32 * 4224 bytes per K tile per slot; at
//     three slots and a scale panel that caps B_M + B_N at 384.
//
// So this file breaks both: eight waves, which cuts the accumulator to 128 per
// wave and makes 2 waves/SIMD mandatory (hence the 256-register ceiling every
// choice below answers to), and direct-to-register B, which removes B from LDS
// entirely and leaves the A ring three slots deep.
//
// What that costs, and why the schedule looks like it does: at 256 registers a
// wave can hold 128 accumulator + 96 of fragments, so neither A nor B is double
// buffered. B's global latency therefore lands inside the K tile it feeds. That
// turned out not to matter -- the second wave on the SIMD covers it, and
// B_STAGED_WAIT below, which waits on B one n-repeat at a time so all but the
// first hide behind MFMAs already issued, measured identical to the single
// conservative wait at both M=8192 and M=32768.
//
// What does matter is that direct B has no LDS to share through, so each of B's
// bytes crosses L1 once per M-wave. The traits pick the wave grid on that basis;
// see the T_M note there.
#pragma once

// Layout helpers (make_layout_*_mxsk), pack_e8m0x4 and the split-K reduce kernel
// are shared verbatim.
#include "opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh"

// SF_SHUF_IN_LDS (the kernel's trailing template bool, per kid) stages the
// shuffle_scale words through the LDS panel instead of reading them from global
// on every K tile. The two do not buy the same thing -- the panel gets the scale
// load off the vmcnt critical path, the layout fills op_sel at COM_REP_M < 4 --
// and the panel holds the shuffled dwords verbatim, so only where the word comes
// from moves. SF_PREFETCH turns off with it, existing to hide a load that no
// longer happens. Per-kid rather than a macro.

#ifdef __HIP_DEVICE_COMPILE__

// Scaled MFMA over the register tile with M-packed A scales and staged B waits.
//
// v_sfa holds, per K group, the COM_REP_M e8m0 bytes of this lane's M subtiles
// in subtile order, so each MFMA picks its byte with the hardware scale_op_sel
// immediate and no pack ALU at all (the shifts fold away: the bytes are already
// adjacent in the register the ds_read landed them in). op_sel indexes one dword,
// so subtile im reads dword im/4 at immediate im%4 -- one dword per K group on
// the 4x2 grid, two on the 2x4 one.
//
// The subtile loop runs n-repeat outermost so that B, which arrives in n-repeat
// order, can be waited on a piece at a time: before n-repeat j only the loads up
// to j need to have landed, and the COM_REP_M MFMAs of every earlier n-repeat
// are already in flight covering them. B_TRAIL is the number of vmcnt entries
// issued after this tile's B (the next A tile's async copies), which the caller
// knows and this loop cannot.
//
// SHUFFLE_SCALE takes the scale words already packed, so
// nothing is packed here: the MFMA picks its byte with
// scale_op_sel = (KP << 1) | (M subtile parity), one immediate for both operands
// since B's dword doubles each byte and spends no bit on M. KP is the K tile's
// parity within its pair, hence a template parameter, and the caller unrolls K
// by two. The M half is (m / SF_SUB) & 1, not im & 1 -- sf_slot/sf_mbit below
// apply SF_GEOM's decomposition of it.
template<typename T, typename Mma, int B_TRAIL, bool B_STAGED_WAIT,
         bool SHUFFLE_SCALE = false, int KP = 0,
         typename VA, typename VB, typename VSFA, typename VSFB, typename VC>
OPUS_D void mma_mxscale_wave8_accum(const VA& v_a, const VB& v_b,
                                    const VSFA& v_sfa, const VSFB& v_sfb, VC& v_c) {
    static_assert(std::is_same_v<typename T::D_SF, unsigned char> || SHUFFLE_SCALE);
    static_assert(T::SCALES_PER_BK == T::COM_REP_K,
                  "one A scale byte per (row, K group), i.e. GROUP_K == W_K");
    static_assert(!SHUFFLE_SCALE || T::COM_REP_K <= 2,
                  "the shuffle_scale layout's dword holds two K blocks, so a tile covers "
                  "one of them (paired across tiles) or both");
    static_assert(!SHUFFLE_SCALE || T::SF_SHUF_OK,
                  "the tile cannot index this layout: SF_MB must divide into SF_SUB "
                  "(paired) or be exactly 2*SF_SUB (wide, one M byte per dword given "
                  "up to the neighbouring wave). SF_MB == 4*SF_SUB straddles two n1 "
                  "blocks and has no wave-uniform fold");
    static_assert(!SHUFFLE_SCALE || T::B_M % (2 * T::SF_SUB) == 0,
                  "the M-subtile parity is compile-time only while the tile's rows block "
                  "evenly by 2*SF_SUB; a narrower tile needs the runtime MP branch");
    // The wave M remap reorders the A fragment and nothing else: make_layout_ra_mxsk
    // splits the m-repeat into a pair-block index and a within-block half, so the y
    // dims flatten as [im/T_M][ik][im%T_M] instead of [im][ik]. Same fragment, same
    // load, same register count -- only i_tile_a below moves. Read off SF_GEOM
    // rather than a template argument so it cannot disagree with the layout call.
    constexpr bool SF_WPAIR = SHUFFLE_SCALE && T::SF_GEOM::WAVE_PAIR;
    using MMA = typename Mma::MMA;
    constexpr int a_len = Mma::mma_a_len;
    constexpr int b_len = Mma::mma_b_len;
    constexpr int c_len = Mma::mma_c_len;
    // N-repeats per B scale group, and the groups this wave's columns span.
    constexpr int rep_n_per_scale = T::GROUP_N / T::W_N;
    static_assert(rep_n_per_scale > 0 && T::GROUP_N % T::W_N == 0);
    static_assert(T::SFB_GROUPS * rep_n_per_scale >= T::COM_REP_N,
                  "the wave's B scale groups must cover its n-repeats");

    // Unused under SHUFFLE_SCALE, where the words come in packed; sized to 1 so the
    // dead vectors fold away rather than needing a second copy of the loop.
    opus::vector_t<int, SHUFFLE_SCALE ? 1 : T::COM_REP_K * T::SFA_WORDS> packed_sfa;
    opus::vector_t<int, SHUFFLE_SCALE ? 1 : T::SFB_GROUPS * T::COM_REP_K> packed_sfb;
    if constexpr (!SHUFFLE_SCALE) {
        opus::static_for<T::COM_REP_K>([&](auto ik_c) {
            constexpr int ik = decltype(ik_c)::value;
            opus::static_for<T::SFA_WORDS>([&](auto iw_c) {
                constexpr int iw = decltype(iw_c)::value;
                int w = 0;
                opus::static_for<4>([&](auto ib_c) {
                    constexpr int ib = decltype(ib_c)::value;
                    w |= (static_cast<int>(v_sfa[ik * T::COM_REP_M + iw * 4 + ib]) & 0xFF)
                         << (8 * ib);
                });
                packed_sfa[ik * T::SFA_WORDS + iw] = w;
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
    }

    opus::static_for<T::COM_REP_N>([&](auto in_c) {
        constexpr int in = decltype(in_c)::value;
        if constexpr (B_STAGED_WAIT) {
            // B was issued one dwordx4 per (n-repeat, k-repeat, half), n-repeat
            // major, so everything from n-repeat in+1 on may still be in flight.
            constexpr int rest = (T::COM_REP_N - 1 - in) * T::COM_REP_K * 2 + B_TRAIL;
            s_waitcnt_vmcnt(opus::number<rest>{});
        }
        opus::static_for<T::COM_REP_M>([&](auto im_c) {
            constexpr int im = decltype(im_c)::value;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                // Subtile im's place in the layout. Both regimes live in
                // opus_sf_shuf_geom's SLOT_OF/MB_BIT_OF, so this and
                // read_scales_shuf cannot drift apart on the register order.
                //
                // At COM_REP_K == 2 the dword's two K blocks are both inside this
                // tile, so its op_sel bit is the K repeat, not the caller's parity.
                constexpr int shuf_kp = T::COM_REP_K == 2 ? ik : KP;
                constexpr int sf_reg  = T::SF_GEOM::REG_OF(im, ik);
                int scale_a, scale_b;
                if constexpr (SHUFFLE_SCALE) {
                    scale_a = v_sfa[sf_reg];
                    scale_b = v_sfb[in / rep_n_per_scale];
                } else {
                    scale_a = packed_sfa[ik * T::SFA_WORDS + im / 4];
                    scale_b = packed_sfb[(in / rep_n_per_scale) * T::COM_REP_K + ik];
                }
                constexpr int op_sel =
                    SHUFFLE_SCALE ? T::SF_GEOM::OPSEL_A(im, shuf_kp) : (im % 4);
                // B's dword stores each K byte twice, so only the K bit matters
                // there and the low bit is free. It carries A's M bit anyway, purely
                // so the emitted immediate does not move.
                constexpr int op_sel_b =
                    SHUFFLE_SCALE ? ((shuf_kp << 1) | T::SF_GEOM::MB_BIT_OF(im)) : 0;
                constexpr int i_tile_a =
                    SF_WPAIR ? ((im / T::T_M) * T::COM_REP_K + ik) * T::T_M
                                   + (im % T::T_M)
                             : (im * T::COM_REP_K + ik);
                constexpr int i_tile_b = in * T::COM_REP_K + ik;
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
                s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b,
                            opus::number<op_sel>{},
                            opus::number<op_sel_b>{});
                opus::set_slice(v_c, s_c,
                    opus::number<i_tile_c * c_len>{},
                    opus::number<i_tile_c * c_len + c_len>{});
            });
        });
    });
}

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Main kernel: 8 all-compute waves in a T_M x T_N grid, direct-to-register
// preshuffled B, LDS-resident scale panels, fp32 workspace or direct output.
// ============================================================================

// DIRECT_ONLY / PREFETCH_SCALE / PRELOAD_SF_LDS are kept in the signature so the
// codegen'd launcher can instantiate this exactly like the shared kernel; only
// the one combination this file implements is accepted.
//
// SFA_MPACK_GLOBAL moves the A panel out of LDS: the host hands it over already
// M-packed (shuffle_scale_mxsk_mpack), so a K tile's scales are one global load
// of COM_REP_M adjacent bytes and there is no panel to stage. It is its own flag
// rather than PRELOAD_SF_LDS=false because that combination means something else
// in the shared kernel -- read the plain layout per K tile -- and a kid handed
// the wrong A-scale layout returns wrong numbers rather than failing.
// XCD_WGM > 0 turns on the L2 rasterization below; it is the band height in M
// tiles. 0 is the plain linear map.
//
// SHUFFLE_SCALE takes both scale panels from global in the reference kernel's layout
// (shuffle_scale_a / _b) and reads them one dword per (subtile pair, K tile
// pair). Against SFA_MPACK_GLOBAL, which packs four M subtiles into a dword and
// leaves the row axis outside at the panel's K pitch, this trades one dwordx2 per
// K tile for one dword per two K tiles per subtile pair -- the same bytes and the
// same instruction count, but the row axis now sits at a stride of one dword, so
// 16 lanes hit one 64B line instead of 16. It implies its own K unroll of two,
// and so needs K % (2*B_K) == 0.
//
// SF_SHUF_IN_LDS stages those shuffled words through the LDS panel instead; see the
// note at the top of this file. It requires SHUFFLE_SCALE and it requires the
// panel to fit, and both are static_asserts rather than a silent downgrade.
template<typename Traits, typename D_OUT = void, bool DIRECT_ONLY = false,
         bool PREFETCH_SCALE = false, bool PRELOAD_SF_LDS = true,
         bool SFA_MPACK_GLOBAL = false, int XCD_WGM = 0, bool SHUFFLE_SCALE = false,
         bool SF_SHUF_IN_LDS = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a8w8_mxscale_bpreshuffle_wave8_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
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
    static_assert(std::is_same_v<D_C, fp32_t>, "split-K main writes an fp32 workspace");
    static_assert(!DIRECT_ONLY && !PREFETCH_SCALE && PRELOAD_SF_LDS,
                  "this pipeline is the direct-B + LDS-scale-panel schedule only");
    static_assert(T::B_PRESHUFFLE && T::B_DIRECT_REG,
                  "B is read from the 16x16 preshuffle straight into registers");
    static_assert(T::ALL_WAVE && (T::WAVES == 8 || T::WAVES == 4),
                  "four or eight all-compute waves");
    static_assert(T::GROUP_M == 1, "one A scale byte per row per K group");

    // Wait on B one n-repeat at a time rather than all at once. Correct either
    // way; false is the conservative single wait for the whole tile.
    //
    // A global A-scale load rules the staged form out. It is issued after B(k)
    // and before the A batch, so vmcnt only reaches it once all of B has retired
    // -- and every MFMA needs it, unlike B where the first n-repeat only needs
    // the first chunk. Staging the wait around it would mean waiting for all of B
    // at the first MFMA regardless, which is what the conservative wait does.
    // Global scale reads add vmcnt entries this loop's piecewise waits do not
    // account for, so they turn the staged B wait off.
    // Shuffled words staged in LDS. Declared here because it decides whether the
    // loop still has a scale load in its vmcnt stream, which B_STAGED_WAIT answers.
    // The test is residency, not capacity. SF_SHUF_K1_MAX
    // makes it fit by construction; the check is for a future edit.
    constexpr int SF_SHUF_LDS_BYTES =
        T::SF_SHUF_RING_LDS + T::SF_SHUF_K1_MAX * T::SF_SHUF_WORDS_PER_K1 * 4 + 256;
    constexpr bool SF_SHUF_FITS = SF_SHUF_LDS_BYTES <= T::SF_PANEL_LDS_CEILING;
    // Hard errors, not downgrades. `&&`-ed into SF_SHUF_IN_LDS, as they used to be, a
    // kid that did not fit compiled as the reg variant under an _lds name --
    // present in the matrix, mislabelled, and reporting "the panel changes
    // nothing", which is how the kid210 regression nearly shipped.
    static_assert(!SF_SHUF_IN_LDS || SHUFFLE_SCALE,
                  "SF_SHUF_IN_LDS stages the shuffled scale words; it means nothing "
                  "without SHUFFLE_SCALE");
    static_assert(!SF_SHUF_IN_LDS || SF_SHUF_FITS,
                  "the shuffled scale panel does not fit under SF_PANEL_LDS_CEILING "
                  "for this tile: emit this kid without SF_SHUF_IN_LDS rather than "
                  "letting it degrade to the register path under an _lds name");
    // With the words in LDS the loop issues no scale load at all, so the plain
    // staged B wait applies again -- there is nothing left for the KP=1 staging to
    // account for.
    constexpr bool B_STAGED_WAIT = !SFA_MPACK_GLOBAL && (!SHUFFLE_SCALE || SF_SHUF_IN_LDS);

    const int split_id = opus::block_id_x() % kargs.split_k;
    const int wgid     = opus::block_id_x() / kargs.split_k;
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col = (wgid / num_tiles_m) * T::B_N;
    // L2 rasterization, the mapping the reference flydsl kernel runs (its
    // xcd_swizzle value is this band height, not a count of XCDs -- it assumes 8
    // of those unconditionally, as here).
    //
    // Two steps. First undo the dispatcher's round-robin over the XCDs, so an XCD
    // owns a contiguous run of logical tiles instead of every 8th one; then walk
    // that run in bands of XCD_WGM row tiles by every column tile, which makes the
    // workgroups co-resident on an XCD share A rows as well as B columns. Only the
    // second step is a reordering -- the first is what makes the second land on
    // one XCD rather than being smeared across all eight.
    //
    // block_id_x is the true dispatch order only when split_k == 1 (otherwise the
    // split_k workgroups of one tile are consecutive ids and sit on different
    // XCDs), so the swizzle is off for split-K. It also assumes gridDim.x is a
    // multiple of 8, or block_id_z shifts which XCD a tile lands on.
    if constexpr (XCD_WGM > 0) {
        constexpr int NUM_XCD = 8;
        if (kargs.split_k == 1) {
            const int num_tiles_n = ceil_div(kargs.n, T::B_N);
            const int total = num_tiles_m * num_tiles_n;
            const int per_xcd = total / NUM_XCD;
            const int rem_xcd = total % NUM_XCD;
            const int xcd = wgid % NUM_XCD;
            const int slot = wgid / NUM_XCD;
            // The first rem_xcd XCDs carry one extra tile, so their runs start
            // one later apiece -- clip at rem_xcd once past them.
            const int lid = xcd * per_xcd + (xcd < rem_xcd ? xcd : rem_xcd) + slot;

            const int per_band = XCD_WGM * num_tiles_n;
            const int band = lid / per_band;
            const int first_m = band * XCD_WGM;
            const int rows_left_m = num_tiles_m - first_m;
            const int band_m = rows_left_m < XCD_WGM ? rows_left_m : XCD_WGM;
            const int loc = lid - band * per_band;
            row = (first_m + loc % band_m) * T::B_M;
            col = (loc / band_m) * T::B_N;
        }
    }
    const int batch_id = opus::block_id_z();
    const int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    const int lane_id = opus::thread_id_x() % get_warp_size();
    // T_M x T_N (M, N) wave grid. wave_id_m is the fast axis so the waves sharing
    // an N range are exactly the ones that split one A load group's rows.
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    const int total_iters = ceil_div(kargs.k, T::B_K);
    const int iters_full = ceil_div(total_iters, kargs.split_k);
    const int loops = (split_id < kargs.split_k - 1)
                    ? iters_full
                    : (total_iters - (kargs.split_k - 1) * iters_full);
    const int k_start = split_id * iters_full * T::B_K;
    const int sf_start = split_id * iters_full * (T::B_K / T::GROUP_K);
    if (loops < T::prefetch_k_iter) return;

    // OOB masking for partial M tiles: bound the A / sfa / C buffers to this
    // tile's valid rows so lanes mapping past M read 0 and their stores are
    // dropped by num_records.
    const int rows_left = kargs.m - row;
    const int rows_avail = rows_left < T::B_M ? rows_left : T::B_M;
    const unsigned int a_bytes =
        (unsigned int)rows_avail * (unsigned int)kargs.stride_a * sizeof(D_A);
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + (size_t)batch_id * kargs.stride_a_batch
                         + (size_t)row * kargs.stride_a + k_start,
                         a_bytes);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch
                         + b_gmem_tile_base_mxsk<T>(col, k_start, kargs.stride_b));
    const bool direct_store = !std::is_void_v<D_OUT> && kargs.split_k == 1;
    const int stride_c_main = direct_store ? kargs.stride_c : kargs.stride_ws;
    // The M-packed panel is a whole B_M tile wide however partial the tile is --
    // the host pads the rows it adds with E8M0 1.0 -- so it needs no row bound,
    // and its K offset scales with the packing. stride_sfa is the per-row scale
    // count either way, which is what makes row*stride_sfa the tile base here.
    const unsigned int sfa_bytes =
        (unsigned int)(SFA_MPACK_GLOBAL ? T::B_M : rows_avail)
        * (unsigned int)kargs.stride_sfa * sizeof(D_SF);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + (size_t)batch_id * kargs.stride_sfa_batch
                           + (size_t)row * kargs.stride_sfa
                           + (SFA_MPACK_GLOBAL ? 0 : sf_start),
                           sfa_bytes);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb + sf_start);

    constexpr int PF = T::prefetch_k_iter;
    // Per-wave vmcnt entries per K tile: A async copies, and direct-B loads.
    constexpr int A_MB = T::a_buffer_load_insts;
    constexpr int B_MB = T::b_direct_load_insts;

    // Shuffled scale words are prefetched one K tile pair ahead, so they are in flight
    // across a whole tile's MFMAs and every wait between has to leave them alone.
    // SF_PF is that many vmcnt entries -- what read_scales_shuf issues -- and it is
    // what the thresholds below add on top of the A and B groups.
    //
    // Only the paired loop (COM_REP_K == 1) prefetches. At COM_REP_K == 2 a tile
    // owns its own word, so the rotate into the consumed register would land while
    // the load is still in flight, and the waitcnt that would fix it is the very
    // exposure the prefetch exists to remove.
    //
    // And only where the wave has the registers for the second set of words,
    // which the register-tile count alone does not decide. Measured over four
    // shapes at K=16384/32768: 128x256 on the 2x4 grid (16 tiles, 152 VGPR with
    // the prefetch, no spill) gains 5-12%, and the same tile on the 1x4 grid (32
    // tiles, and the same five words to hold) goes from 250 to 254 VGPR without
    // spilling and gains 1.5-6%. But 256x256, also 32 tiles and also five words,
    // is already at 256 with 10 spilled before the prefetch: it loses 11-16%, and
    // still 3% when its scale addresses are made cheap enough to bring the
    // prefetched build down to 8 spills. Nothing in the traits separates the two
    // 32-tile shapes -- they agree on every REP -- so the bound carries B_M:
    // past 128 rows per workgroup the wave has no room left.
    constexpr bool SF_PREFETCH = !SF_SHUF_IN_LDS && SHUFFLE_SCALE && T::COM_REP_K == 1
                                 && (T::COM_REP_M * T::COM_REP_N <= 16
                                     || (T::COM_REP_M * T::COM_REP_N <= 32
                                         && T::B_M <= 128));
    // Loads in flight, not registers held. On a WIDE kid this over-counts: the
    // `>> 8*np` fold is a *use* of the loaded word, so it drains vmcnt where it
    // sits and every wait below becomes a no-op -- over-synchronised rather than
    // under-. It also means the prefetch buys a WIDE kid nothing unless the
    // compiler sinks the fold past the rotate; read the ISA before assuming.
    constexpr int SF_PF = SF_PREFETCH ? T::SF_A_SLOTS + T::SFB_GROUPS : 0;

    // B never lands in LDS here, so only A is staged.
    __shared__ char smem_a[PF * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    // Scale panels: the A per-token scales (SFA) and B block scales (SFB) for
    // this split's whole K range, staged once so the per-K-tile fetch is a
    // ds_read instead of a buffer_load. Sized for a compile-time K bound; the
    // packed per-row count is a runtime value, so any K <= SFA_K_MAX works.
    constexpr int SFA_K_MAX       = T::SF_PRELOAD_K_MAX;
    constexpr int SFA_K_TILES_MAX = SFA_K_MAX / T::B_K;
    constexpr int SF_SCALES_MAX   = SFA_K_TILES_MAX * T::SCALES_PER_BK;
    constexpr int SFA_ROWS        = T::B_M / T::GROUP_M;
    // Rows one M subtile spans: the row for subtile im is im*SFA_MB +
    // wave_id_m*W_M + lane%W_M (make_layout_sfa_mxsk), so the subtile index is
    // the high part of the panel row and the lane-dependent part is the low part.
    constexpr int SFA_MB = T::T_M * T::W_M;
    static_assert(SFA_ROWS == SFA_MB * T::COM_REP_M);
    // The shuffled layout's row-pair distance. Equal to SFA_MB only at T_M=2:
    // everything shuffled indexes off SF_SUB, everything M-packed off SFA_MB.
    constexpr int SF_SUB   = T::SF_SUB;
    constexpr int NL_SLOTS = T::SF_NL_SLOTS;
    using SFG = typename T::SF_GEOM;
    // SF_WPAIR is the only thing here that renumbers a row; the A fragment layout,
    // the C store and the shuffled scale address all follow it, and nothing else
    // in the kernel knows a row number (A staging is per load group, the M edge is
    // masked by num_records). Gated on SHUFFLE_SCALE: the M-packed scale panel
    // uses the default map and would have to be re-packed.
    constexpr bool SF_WPAIR = SHUFFLE_SCALE && SFG::WAVE_PAIR;
    constexpr bool SF_WIDE = SFG::WIDE_SPLIT;
    // A read from global needs no panel, which also gives back the SFA_ROWS/
    // N_SCALE_GROUPS share of this array -- at B_M=256 against 2 scale groups
    // that is all but 128 bytes of it. The shuffle_scale layout reads both panels from
    // global and gives the array back entirely.
    constexpr bool SF_LDS_A = !SFA_MPACK_GLOBAL && !SHUFFLE_SCALE;
    constexpr bool SF_LDS_B = !SHUFFLE_SCALE;
    constexpr int SF_LDS_ELEMS =
        ((SF_LDS_A ? SFA_ROWS : 0) + (SF_LDS_B ? T::N_SCALE_GROUPS : 0)) * SF_SCALES_MAX;
    alignas(16) __shared__ D_SF smem_sf[SF_LDS_ELEMS > 0 ? SF_LDS_ELEMS : 1];

    // Shuffled-word panel, in dwords, holding what read_scales_shuf would
    // otherwise fetch from global over the whole split:
    //
    //   A  [block b][k1][nl] -- b < SF_N1_BLOCKS, nl < SF_SUB, k1 < K1
    //   B  [group g][k1]     -- g < N_SCALE_GROUPS, covering every wave's sfb_group0
    //
    // For A the run over (k1, nl) is contiguous in the global layout too, which is
    // what makes the fill a straight copy. K1 counts 128-block *pairs*, and the
    // depth is the shuffled panel's own K bound.
    constexpr int SHUF_K1_MAX = T::SF_SHUF_K1_MAX;
    // shuf_a_word0 and shuf_r_word have no row % SF_SUB term on any arm, so a tile
    // narrower than a subtile pair would read the block's first rows rather than
    // its own -- on the global arm as well as the panel arm. Inert by inventory
    // (every tile here is B_M >= 128); flatmm_splitk carries the term and so hosts
    // the B_M = 16 panel without this restriction.
    static_assert(!SHUFFLE_SCALE || !SFG::SUBTILE_TILE,
                  "a tile narrower than 2*SF_SUB cannot read the shuffled layout here: "
                  "shuf_a_word0 omits its row % SF_SUB offset on every arm");
    constexpr int SHUF_A_WORDS = SF_SHUF_IN_LDS ? T::SF_N1_BLOCKS * SHUF_K1_MAX * SF_SUB : 0;
    constexpr int SHUF_B_WORDS = SF_SHUF_IN_LDS ? T::N_SCALE_GROUPS * SHUF_K1_MAX : 0;
    constexpr int SHUF_WORDS   = SHUF_A_WORDS + SHUF_B_WORDS;
    alignas(16) __shared__ int smem_sf_shuf[SHUF_WORDS > 0 ? SHUF_WORDS : 1];

    auto smem_a_at = [&](int slot_k, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    const int sf_k_scales = loops * T::SCALES_PER_BK;
    D_SF* s_sfa_ptr = smem_sf;
    D_SF* s_sfb_ptr = smem_sf + (SF_LDS_A ? SFA_ROWS * sf_k_scales : 0);

    // One-shot cooperative fill of both panels by all BLOCK_SIZE threads,
    // published by the barrier at the end. OOB rows of a partial M tile read 0
    // through g_sfa's bound and are never consumed. Nothing to do, and no K limit
    // to respect, when both panels are read from global.
    if constexpr (SF_LDS_A || SF_LDS_B) {
        // Overrunning the panels would corrupt LDS, so the bail-out stays, but
        // returning leaves Y untouched and a zeroed output reads like a correct
        // answer. The launcher checks the same bound with AITER_CHECK; reaching
        // this return means a caller went around it.
        if (loops > SFA_K_TILES_MAX) return;
        const int tid = opus::thread_id_x();
        auto sm_sfa = make_smem(s_sfa_ptr);
        auto sm_sfb = make_smem(s_sfb_ptr);
        const int sfa_total = SFA_ROWS * sf_k_scales;
        const int sfb_total = T::N_SCALE_GROUPS * sf_k_scales;
        auto fill_sfb = [&](auto vec_c) {
            constexpr int VEC = decltype(vec_c)::value;
            for (int idx = tid * VEC; idx < sfb_total; idx += T::BLOCK_SIZE * VEC) {
                const int r  = idx / sf_k_scales;
                const int kt = idx - r * sf_k_scales;
                sm_sfb.template store<VEC>(
                    load<VEC>(g_sfb, r * kargs.stride_sfb + kt), idx);
            }
        };
        // SFA goes in M-packed: row r of the logical panel is subtile r/SFA_MB of
        // the lane at row r%SFA_MB, so storing it as [r%SFA_MB][k][r/SFA_MB] puts
        // the COM_REP_M bytes one lane needs for a K group in COM_REP_M adjacent
        // slots, naturally aligned for a single dword read.
        //
        // The subtile index being the fastest destination axis is what makes the
        // fill awkward: adjacent destination bytes come from rows SFA_MB apart. A
        // thread therefore takes SFA_PACK such rows rather than one, which puts
        // SFA_PACK adjacent panel bytes in its registers and lets the panel land
        // in dword stores instead of one ds_write_b8 per byte. Lanes still walk K
        // within a row, so the global reads stay as wide and as coalesced as they
        // were, and the instruction count moves on the LDS side only.
        //
        // Worth 2-3% at large K on every kid that runs this fill, and nothing at
        // K=1024 (g16/m8192/n1024, dword against byte stores: 0.98 / 0.97 / 0.99 /
        // 0.97 / 1.00 / 0.99 / 0.98 for kid168/175/192/194/202/203/205 at K=4096,
        // and 0.99 / 0.97 / 0.99 / 0.98 / 0.97 / 0.98 / 0.98 at K=8192, against
        // 1.00 +-0.006 across all seven at K=1024).
        //
        // That K profile is the panel's own: sf_k_scales is loops*SCALES_PER_BK,
        // so the panel and its fill grow with K and this is a reduction in cost
        // per unit K, not in the fixed part. Worth keeping straight because the
        // panel's *fixed* cost is a separate quantity and kid208 is what measures
        // it -- removing the panel outright buys ~18 us of it and pays 6.9% per
        // unit K back, i.e. exactly the term this store width is what sets.
        constexpr int SFA_PACK = (T::COM_REP_M % 4 == 0) ? 4
                               : ((T::COM_REP_M % 2 == 0) ? 2 : 1);
        constexpr int SFA_PACK_GROUPS = T::COM_REP_M / SFA_PACK;
        auto fill_sfa = [&](auto vec_c) {
            constexpr int VEC = decltype(vec_c)::value;
            // dispatch_width picks VEC to divide sf_k_scales and stride_sfa, so
            // the chunk count is exact and every global read stays aligned.
            const int k_chunks = sf_k_scales / VEC;
            const int items = SFA_MB * SFA_PACK_GROUPS * k_chunks;
            for (int it = tid; it < items; it += T::BLOCK_SIZE) {
                const int kc   = it % k_chunks;
                const int rest = it / k_chunks;
                const int pk   = rest % SFA_PACK_GROUPS;
                const int mb   = rest / SFA_PACK_GROUPS;
                const int kt   = kc * VEC;
                opus::vector_t<D_SF, SFA_PACK * VEC> v;
                opus::static_for<SFA_PACK>([&](auto j_c) {
                    constexpr int j = decltype(j_c)::value;
                    const int r = (pk * SFA_PACK + j) * SFA_MB + mb;
                    auto row = load<VEC>(g_sfa, r * kargs.stride_sfa + kt);
                    opus::static_for<VEC>([&](auto i_c) {
                        constexpr int i = decltype(i_c)::value;
                        v[j * VEC + i] = row[i];
                    });
                });
                const int dst =
                    (mb * sf_k_scales + kt) * T::COM_REP_M + pk * SFA_PACK;
                opus::static_for<VEC>([&](auto i_c) {
                    constexpr int i = decltype(i_c)::value;
                    opus::vector_t<D_SF, SFA_PACK> w;
                    opus::static_for<SFA_PACK>([&](auto j_c) {
                        constexpr int j = decltype(j_c)::value;
                        w[j] = v[j * VEC + i];
                    });
                    sm_sfa.template store<SFA_PACK>(w, dst + i * T::COM_REP_M);
                });
            }
        };
        auto dispatch_width = [&](int stride, auto&& fill) {
            const int widths = sf_k_scales | stride;
            if      ((widths & 15) == 0) fill(number<16>{});
            else if ((widths & 3) == 0)  fill(number<4>{});
            else                         fill(number<1>{});
        };
        if constexpr (SF_LDS_A) dispatch_width(kargs.stride_sfa, fill_sfa);
        if constexpr (SF_LDS_B) dispatch_width(kargs.stride_sfb, fill_sfb);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    // Staging: there are no producer waves, every wave issues its own share of
    // each A tile. A wave takes slot group wave_id%LOAD_WAVES of the A load
    // groups congruent to wave_id/LOAD_WAVES mod LOAD_M_SPLIT, which covers every
    // (group, slot) pair exactly once and gives all eight waves the same count.
    const int wave_id_load = wave_id % T::LOAD_WAVES;
    const int m_block0 = wave_id / T::LOAD_WAVES;
    auto u_ga = make_layout_gmem_group_load_mxsk<T, T::LOAD_WAVES>(
        lane_id, wave_id_load, kargs.stride_a);
    auto u_sa = make_layout_smem_group_load_mxsk<T, T::LOAD_WAVES>(lane_id, wave_id_load);

    auto u_ra = make_layout_ra_mxsk<T, T::COM_REP_M, SF_WPAIR>(lane_id, wave_id_m);
    // Each N-wave owns COM_REP_N contiguous 16-column blocks of the tile.
    auto u_gb_direct = make_layout_gmem_b_direct_mxsk<T>(
        lane_id, kargs.stride_b, wave_id_n * T::COM_REP_N);
    // Lane-invariant part of this lane's M-packed panel row, at the panel's own
    // pitch. The K tile and the subtile index are added at the read. The LDS
    // panel holds this split's K range, so its pitch is sf_k_scales and the split
    // offset is already in the pointer; the host panel holds all of K, so it goes
    // at stride_sfa and carries sf_start here.
    const int sfa_mpack_row = SFA_MPACK_GLOBAL
        ? (wave_id_m * T::W_M + lane_id % T::W_M) * kargs.stride_sfa + sf_start
        : (wave_id_m * T::W_M + lane_id % T::W_M) * sf_k_scales;
    // First B scale group of this wave's column range. A range narrower than
    // GROUP_N makes consecutive N-waves share a group rather than step one.
    const int sfb_group0 = wave_id_n * T::SFB_WAVE_COLS / T::GROUP_N;

    // Shuffled scale addressing, counted in dwords. Splitting a row over blocks of
    // 2*SF_SUB puts the block index high, op_sel's M bit in the middle and the row
    // within the subtile low:
    //
    //     word = ((base/(2*SF_SUB) + b)*K1 + k1)*SF_SUB + nl*SFA_MB + r_lane
    //
    // The lane axis has stride one dword, so a wave's 16 lanes cover one 64B line.
    // `nl` is the second slot axis, live only at T_M=1, where the tile holds two
    // subtiles in one SF_SUB row range as separate dwords rather than bytes of one.
    static_assert(!SHUFFLE_SCALE || T::B_M % (2 * SF_SUB) == 0,
                  "the shuffle_scale layout blocks the tile's rows by 2*SF_SUB, and a "
                  "narrower tile makes op_sel's M bit runtime (needs the MP branch)");
    // K1 counts 128-block pairs over the whole of K, which is the pitch the host
    // padded the panels to -- not the tile count, since a tile covers one block
    // or two depending on B_K.
    const int shuf_k1 = (ceil_div(kargs.k, T::GROUP_K) + 1) / 2;
    const int shuf_k1_start = sf_start / 2;
    auto g_sfa_shuf = make_gmem(
        reinterpret_cast<const int*>(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                                     + (size_t)batch_id * kargs.stride_sfa_batch),
        (unsigned int)kargs.stride_sfa_batch * (unsigned int)sizeof(D_SF));
    auto g_sfb_shuf = make_gmem(
        reinterpret_cast<const int*>(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                                     + (size_t)batch_id * kargs.stride_sfb_batch));
    // Three arms for where wave_id_m goes: below SF_WIDE it
    // stays in the word's low part; at SF_WIDE two waves share a word and differ
    // only in which M *byte* they read, so it leaves the address and reappears as
    // sf_wave_shift; under the remap it owns whole 2*SF_SUB row blocks and so
    // multiplies the block pitch instead. Spelled as whole expressions rather than
    // one arm with a substituted sub-term: substituting re-associates the address
    // and changes instruction selection, so an edit meant to be inert is not.
    const int shuf_a_word0 =
        SF_WPAIR ? (row / (2 * SF_SUB) + wave_id_m) * shuf_k1 * SF_SUB
                       + lane_id % T::W_M
        : SF_WIDE ? (row / (2 * SF_SUB)) * shuf_k1 * SF_SUB
                      + (wave_id_m * T::W_M + lane_id % T::W_M) % SF_SUB
                : (row / (2 * SF_SUB)) * shuf_k1 * SF_SUB
                      + wave_id_m * T::W_M + lane_id % T::W_M;
    const int shuf_b_word0 = (col / T::GROUP_N + sfb_group0) * shuf_k1;

    // Fill the shuffled panel. Both operands are copied for the whole workgroup,
    // not per wave: sfb_group0 is a per-wave offset into the WG's N_SCALE_GROUPS,
    // and the A run spans the layout's whole SF_SUB row range. Copied global -> LDS
    // directly, no word through a register.
    constexpr int SF_FILL_VEC  = (SF_SUB % 4 == 0) ? 4 : 1;   // dwords per lane
    constexpr int SF_FILL_WAVE = 64 * SF_FILL_VEC;            // dwords per instruction
    constexpr int SF_FILL_NW   = T::BLOCK_SIZE / 64;
    if constexpr (SF_SHUF_IN_LDS) {
        // This panel's own bound, smaller than the plain panel's. The launcher
        // checks it with AITER_CHECK, so reaching this return means a caller went
        // around it -- while sf_shuf_in_lds was a bare macro, codegen could not see
        // the panel and emitted no check, and this return handed back zeros.
        if (loops > T::SF_SHUF_K_TILES_MAX) return;
        const int k1n = T::COM_REP_K == 2 ? loops : (loops + 1) / 2;
        const int a_run = k1n * SF_SUB;
        // dst is M0 and must stay wave-uniform; the per-lane part is the source
        // offset only. Lanes past the run are masked off rather than clamped, so
        // no word outside the panel is written.
        opus::static_for<T::SF_N1_BLOCKS>([&](auto b_c) {
            constexpr int b = decltype(b_c)::value;
            const int gbase = ((row / (2 * SF_SUB) + b) * shuf_k1 + shuf_k1_start) * SF_SUB;
            const int dbase = b * (SHUF_K1_MAX * SF_SUB);
            for (int off = wave_id * SF_FILL_WAVE; off < a_run;
                 off += SF_FILL_NW * SF_FILL_WAVE) {
                const int lane_off = off + lane_id * SF_FILL_VEC;
                if (lane_off < a_run) {
                    g_sfa_shuf.template async_load<SF_FILL_VEC>(
                        smem_sf_shuf + dbase + lane_off, gbase + lane_off);
                }
            }
        });
        // B is N_SCALE_GROUPS runs of k1n dwords, and k1n carries no alignment, so
        // this one copies a dword per lane. It is tiny next to A.
        for (int g = 0; g < T::N_SCALE_GROUPS; ++g) {
            const int gbase = (col / T::GROUP_N + g) * shuf_k1 + shuf_k1_start;
            const int dbase = SHUF_A_WORDS + g * SHUF_K1_MAX;
            for (int off = wave_id * 64; off < k1n; off += SF_FILL_NW * 64) {
                const int lane_off = off + lane_id;
                if (lane_off < k1n) {
                    g_sfb_shuf.template async_load<1>(
                        smem_sf_shuf + dbase + lane_off, gbase + lane_off);
                }
            }
        }
        // vmcnt alone. buffer_load ... offen lds is VMEM and its counter only
        // drops once the data is in LDS, so this covers the whole DMA -- same
        // budget the A ring buffer's async_loads sit in (A_MB below). The plain
        // panel needs lgkmcnt too because it fills through ds_write; this path
        // has no ds_write to wait on.
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    // A tile issue, always a whole tile: an index past the end re-reads the last
    // tile's bytes into the slot it would have used -- one nobody reads again.
    // Skipping it would leave fewer copies outstanding than the barrier's vmcnt
    // immediate expects, and the wait would clear with the tile it is meant to
    // publish still in flight.
    auto issue_a_tile = [&](int issue_k) {
        const int kk = issue_k < loops ? issue_k : loops - 1;
        const int slot = issue_k % PF;
        opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
            constexpr int kg = decltype(kg_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BM / T::LOAD_M_SPLIT>([&](auto m_c) {
                const int m = decltype(m_c)::value * T::LOAD_M_SPLIT + m_block0;
                async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa,
                                     a_offset(kk, m, kg));
            });
        });
    };

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    // Neither fragment is double buffered: at 2 waves/SIMD the register budget
    // is 128 accumulator + 96 fragments, and a second copy of either does not
    // fit. The overlap comes from the other wave on the SIMD instead. A is held
    // whole, which is what confines the traits to the (T_M, B_M) pairs whose A
    // fragment fits beside the accumulator.
    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    // v_sfa is indexed [K group][M subtile] -- the panel packs it that way.
    using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
    using vtype_sfb = vector_t<D_SF, T::SFB_GROUPS * T::SCALES_PER_BK>;

    // Shuffled scale words live across the K tile pair that reads them, so they sit outside
    // k_tile: the KP=0 tile loads them and the KP=1 tile reuses the registers.
    //
    // Under SF_PREFETCH there are two sets: the pair being computed reads _shuf
    // while the next pair's words land in _shuf_nx, and the KP=1 tile rotates them
    // down once its staged wait has retired the load. The rotate is SF_A_SLOTS +
    // SFB_GROUPS v_movs per pair, which buys a compile-time register index -- a
    // runtime parity would index the vector and force it to scratch.
    //
    // SF_A_SLOTS_K is A_SLOTS times KD, and KD is 1 throughout this pipeline
    // (COM_REP_K <= 2), so it is the same count the loops below use.
    vector_t<int, SHUFFLE_SCALE ? T::SF_A_SLOTS_K : 1> v_sfa_shuf;
    vector_t<int, SHUFFLE_SCALE ? T::SFB_GROUPS : 1> v_sfb_shuf;
    vector_t<int, SF_PREFETCH ? T::SF_A_SLOTS_K : 1> v_sfa_shuf_nx;
    vector_t<int, SF_PREFETCH ? T::SFB_GROUPS : 1> v_sfb_shuf_nx;

    // The lane's row within the subtile pair, which is the low part of the A word
    // index in both the global layout and the panel.
    const int shuf_r_lane = wave_id_m * T::W_M + lane_id % T::W_M;
    // The M byte this lane's wave owns. op_sel is an MFMA immediate and this bit is
    // runtime, but wave-uniform, so it rides in a plain `word >> 8*bit` -- nothing
    // below SF_WIDE. Under the remap the wave is not in the row at all, so both
    // byte-side terms are dead. shuf_r_word is the panel's low part only (the
    // global side folds the same terms into shuf_a_word0).
    const int shuf_r_word   = SF_WPAIR ? lane_id % T::W_M
                                             + wave_id_m * (SHUF_K1_MAX * SF_SUB)
                            : SF_WIDE  ? (shuf_r_lane % SF_SUB) : shuf_r_lane;
    const int sf_wave_shift = SF_WIDE ? (((shuf_r_lane / SF_SUB) & 1) << 3) : 0;

    auto read_scales_shuf = [&](int k, auto& dst_a, auto& dst_b) {
        const int k1 = T::COM_REP_K == 2 ? k : (k >> 1);
        if constexpr (SF_SHUF_IN_LDS) {
            // Same words, same op_sel, staged base instead of the global one.
            opus::static_for<T::SF_A_SLOTS>([&](auto s_c) {
                constexpr int s  = decltype(s_c)::value;
                // which 2*SF_SUB row block; the remap's slots step two of them,
                // the lane's own, and the wave's offset rides in shuf_r_word
                constexpr int b  = SF_WPAIR ? s * SFG::N1_STEP : s / NL_SLOTS;
                constexpr int nl = s % NL_SLOTS;   // which subtile-wide slot in it
                int w = smem_sf_shuf[b * (SHUF_K1_MAX * SF_SUB) + k1 * SF_SUB
                                     + nl * SFA_MB + shuf_r_word];
                if constexpr (SF_WIDE) w >>= sf_wave_shift;
                dst_a[s] = w;
            });
        } else {
            opus::static_for<T::SF_A_SLOTS>([&](auto s_c) {
                constexpr int s  = decltype(s_c)::value;
                // As in the LDS arm: two blocks per slot under the remap, and the
                // wave's own block is already inside shuf_a_word0.
                constexpr int b  = SF_WPAIR ? s * SFG::N1_STEP : s / NL_SLOTS;
                constexpr int nl = s % NL_SLOTS;
                int w = load<1>(
                    g_sfa_shuf, shuf_a_word0 + (b * shuf_k1 + shuf_k1_start + k1) * SF_SUB
                                             + nl * SFA_MB)[0];
                if constexpr (SF_WIDE) w >>= sf_wave_shift;
                dst_a[s] = w;
            });
        }
        // B needs no M fold at all: its dword stores each K byte twice, so it never
        // spends a byte on M.
        if constexpr (SF_SHUF_IN_LDS) {
            opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
                constexpr int ng = decltype(ng_c)::value;
                dst_b[ng] = smem_sf_shuf[SHUF_A_WORDS
                                         + (sfb_group0 + ng) * SHUF_K1_MAX + k1];
            });
        } else {
            opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
                constexpr int ng = decltype(ng_c)::value;
                dst_b[ng] = load<1>(
                    g_sfb_shuf, shuf_b_word0 + ng * shuf_k1 + shuf_k1_start + k1)[0];
            });
        }
    };

    // Prefetch target for the pair after the one at k. Past the last pair it
    // re-reads the current one: the words go to the rotate and no MFMA consumes
    // them, and unlike a skipped load it keeps SF_PF exact for every wait, and
    // unlike an unclamped index it stays inside the panel the host padded.
    // The if constexpr is not redundant: this lambda is not generic, so its body
    // is compiled for every instantiation, and the _nx vectors collapse to one
    // element unless SF_PREFETCH -- which the static_for would index past.
    auto prefetch_scales_shuf = [&](int k) {
        if constexpr (SF_PREFETCH) {
            const int k_nx = k + 2 < loops ? k + 2 : k;
            read_scales_shuf(k_nx, v_sfa_shuf_nx, v_sfb_shuf_nx);
        }
    };

    auto issue_b_direct = [&](int loop_k) {
        const int kk = loop_k < loops ? loop_k : loops - 1;
        v_b = load<T::VEC_B>(g_b, u_gb_direct, b_direct_iter_offset_mxsk<T>(kk));
    };

    auto read_scales = [&](int loop_k, vtype_sfa& v_sfa, vtype_sfb& v_sfb) {
        const int scale_base = loop_k * T::SCALES_PER_BK;
        // One load for the whole K tile's A scales: this lane's COM_REP_M bytes
        // per K group are adjacent in either panel. Read straight from the host's
        // panel when there is one -- the caller must pass
        // shuffle_scale_mxsk_mpack(x_scale, B_M, T_M*W_M).
        if constexpr (SFA_MPACK_GLOBAL) {
            v_sfa = load<T::COM_REP_M * T::SCALES_PER_BK>(
                g_sfa, (sfa_mpack_row + scale_base) * T::COM_REP_M);
        } else {
            auto sm_a = make_smem(s_sfa_ptr + (sfa_mpack_row + scale_base) * T::COM_REP_M);
            v_sfa = load<T::COM_REP_M * T::SCALES_PER_BK>(sm_a, 0);
        }
        opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            auto sm_b = make_smem(s_sfb_ptr + (sfb_group0 + ng) * sf_k_scales + scale_base);
            auto sfb = load<T::SCALES_PER_BK>(sm_b, 0);
            opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
            });
        });
    };

    // One K tile of the steady state. The order is what keeps both counters
    // exact:
    //   barrier   -- tile k has landed and slot (k-1)%PF is free for everyone,
    //                because every wave consumed tile k-1 before it
    //   ds_read A(k), read scales     (lgkmcnt; under SF_PREFETCH this slot
    //                                  issues the next pair's words instead)
    //   issue A(k+PF-1)               (vmcnt, into the slot just freed)
    //   MFMA(k)                       (waits lgkmcnt 0, and vmcnt down to the
    //                                  A batch just issued plus any prefetch,
    //                                  which retires B(k))
    //   issue B(k+1)                  (vmcnt; earliest point, v_b is now dead)
    auto k_tile = [&](int k, auto kp_c) {
        constexpr int KP = decltype(kp_c)::value;
        // The shuffle_scale layout only loads scales on the first tile of a pair, so the
        // second one issues nothing the staged B wait cannot account for and
        // keeps it. The words it reads landed a pair ago, and the load it does
        // retire is the prefetch, which is what makes the rotate below safe.
        constexpr bool STAGED = SHUFFLE_SCALE ? (KP == 1) : B_STAGED_WAIT;
        // The prefetched words are issued by the KP=0 tile and retired by the KP=1
        // tile's staged wait, so they are outstanding across exactly this barrier
        // and it has to leave them: at the plain A_MB + B_MB it would collect them
        // itself, cutting the cover to the half pair it already had.
        constexpr int SF_HELD = KP == 1 ? SF_PF : 0;
        s_waitcnt_vmcnt(number<A_MB + B_MB + SF_HELD>{});
        __builtin_amdgcn_s_barrier();

        vtype_sfa v_sfa;
        vtype_sfb v_sfb;

        auto sa = make_smem(smem_a_at(k % PF, 0, 0));
        v_a = load<T::VEC_A>(sa, u_ra);
        // read_scales' slot, which is sized for the LDS latency of a panel read.
        // A global load put here is waited on ~A_MB instructions later and so is
        // fully exposed, which is why SF_PREFETCH issues the *next* pair's words
        // here instead and the MFMAs below cover them.
        if constexpr (SF_PREFETCH) {
            if constexpr (KP == 0) prefetch_scales_shuf(k);
        } else if constexpr (SHUFFLE_SCALE) {
            // KP is always 0 on the unpaired COM_REP_K == 2 loop that lands here,
            // so the guard is for whoever forces SF_PREFETCH off: without it the
            // paired loop would load twice and break the KP=1 staged wait.
            if constexpr (KP == 0) read_scales_shuf(k, v_sfa_shuf, v_sfb_shuf);
        } else {
            read_scales(k, v_sfa, v_sfb);
        }

        issue_a_tile(k + PF - 1);

        s_waitcnt_lgkmcnt(0_I);
        // Down to the A batch, plus the prefetch that has to outlive this tile.
        // What this retires is B(k), which the MFMAs consume; the words they read
        // were prefetched a pair ago and are long back.
        if constexpr (!STAGED) s_waitcnt_vmcnt(number<A_MB + SF_PF>{});
        __builtin_amdgcn_s_setprio(1);
        if constexpr (SHUFFLE_SCALE) {
            mma_mxscale_wave8_accum<T, decltype(mma), A_MB, STAGED, true, KP>(
                v_a, v_b, v_sfa_shuf, v_sfb_shuf, v_c);
        } else {
            mma_mxscale_wave8_accum<T, decltype(mma), A_MB, STAGED>(
                v_a, v_b, v_sfa, v_sfb, v_c);
        }
        __builtin_amdgcn_s_setprio(0);

        // Safe here and only here: the staged wait above retired the prefetch, so
        // the source registers hold landed words rather than a load in flight.
        if constexpr (SF_PREFETCH && KP == 1) {
            opus::static_for<T::SF_A_SLOTS_K>([&](auto s_c) {
                v_sfa_shuf[decltype(s_c)::value] = v_sfa_shuf_nx[decltype(s_c)::value];
            });
            opus::static_for<T::SFB_GROUPS>([&](auto ng_c) {
                v_sfb_shuf[decltype(ng_c)::value] = v_sfb_shuf_nx[decltype(ng_c)::value];
            });
        }

        issue_b_direct(k + 1);
    };

    // Prefill: PF-1 A tiles, then B(0).
    //
    // A_MB + B_MB is the right barrier wait for every tile, prologue included.
    // In the steady state it is exactly what is outstanding when a tile starts
    // -- the MFMA of tile k-1 waited vmcnt down to its own A batch, which
    // retired every A copy issued before B(k-1), so A(k) landed an iteration
    // early and the wait is free. At tile 0 the ring is still full: the prefill
    // leaves (PF-1) A batches and B(0) outstanding, and waiting down to
    // A_MB + B_MB retires exactly the A tiles up to and including A(0).
    // Pair 0 has no earlier tile to hide behind, so its words go first and tile
    // 0's barrier wait collects them along with the A prefill it already waits
    // for. Only this one pair pays the latency the steady state now hides.
    if constexpr (SF_PREFETCH) read_scales_shuf(0, v_sfa_shuf, v_sfb_shuf);

    opus::static_for<PF - 1>([&](auto p_c) {
        issue_a_tile(decltype(p_c)::value);
    });
    issue_b_direct(0);

    if constexpr (SHUFFLE_SCALE && T::COM_REP_K == 1) {
        // A tile covers one of the scale word's two K blocks, so K walks in pairs
        // to make the word's K bit a compile-time op_sel, and the loop body
        // exists at both parities. An odd tail runs on the KP=0 body, which is
        // the parity a last unpaired tile has, so it costs no third copy.
        int k = 0;
        for (; k + 1 < loops; k += 2) {
            k_tile(k, number<0>{});
            k_tile(k + 1, number<1>{});
        }
        if (k < loops) k_tile(k, number<0>{});
    } else {
        for (int k = 0; k < loops; ++k) {
            k_tile(k, number<0>{});
        }
    }

    // Store. The generic swap_ab C partition nests the register n-repeat outside
    // the wave's N tile, which transposes the (wave, n-repeat) -> column map
    // whenever both T_N>1 and COM_REP_N>1: the accumulators are right but land in
    // swapped columns. Wave w computes n-repeat j from column group
    // w*COM_REP_N + j (that is what nbc above selects), so each n-repeat is
    // stored to that contiguous column group with a single-N-tile layout.
    constexpr int C_LEN = decltype(mma)::mma_c_len;
    auto p_coord_c1 = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                       0, lane_id / mma.grpn_c);
    // One MFMA tile per store, n-repeat innermost, which is what makes the writes
    // reach DRAM as whole lines.
    //
    // VEC_C packs along N, so a lane holds 4 adjacent columns and lane%16 is its
    // row: one store instruction covers 16 rows x 16 columns, i.e. 32 bytes of
    // each row -- half of a 64B line whatever the per-lane width is. The other
    // half is the next n-repeat. Storing a whole n-repeat at a time (the natural
    // reading of the layout, and what this did originally) therefore leaves
    // COM_REP_M other stores between the two halves of every line, and the L2 does
    // not always still hold the first half when the second arrives: it retired
    // 5.45M DRAM writes for a 268.4 MB C tile against the 4.19M full lines it
    // needs. Pairing the halves back to back is enough to fix that; see the C
    // write note in opus_gemm_common.py.
    auto mma_c2 = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<1, 1, T::COM_REP_K>{}, seq<T::T_M, 1, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    auto u_gc2 = partition_layout_c<T::VEC_C>(mma_c2,
        opus::make_tuple(stride_c_main, 1_I), p_coord_c1);
    auto store_c = [&](auto& g) {
        // The accumulator nests m-repeat outside n-repeat (i_tile_c =
        // im*COM_REP_N + in), so this walks it in order and only has to add the
        // m-repeat's row offset, which u_gc2 no longer carries. Two things about
        // that offset:
        //
        // It is T_M*W_M per m-repeat, not W_M. The generic partition nests the
        // register m-repeat outside the wave's M tile, so a wave's rows are strided
        // by the whole wave grid rather than contiguous. Getting this wrong only
        // shows up at T_M>1 -- it is the identity for the T_M=1 kids.
        //
        // And it goes on the layout, not into store's third argument, which is the
        // buffer soffset: only the layout's offset reaches the bounds-checked
        // voffset, so a row offset passed as soffset addresses past num_records
        // instead of being dropped, and faults on any tile the M bound clips.
        //
        // Under the wave M remap the C store follows the same renumbering: u_gc2's
        // p-coord already moves by W_M, so the layout carries the extra
        // (T_M-1)*W_M.
        opus::static_for<T::COM_REP_M>([&](auto im_c) {
            constexpr int im = decltype(im_c)::value;
            opus::static_for<T::COM_REP_N>([&](auto j_c) {
                constexpr int j = decltype(j_c)::value;
                typename decltype(mma_c2)::vtype_c vj;
                opus::static_for<C_LEN>([&](auto e_c) {
                    constexpr int e = decltype(e_c)::value;
                    vj[e] = v_c[(im * T::COM_REP_N + j) * C_LEN + e];
                });
                // Both arms spelled out whole rather than sharing a named offset
                // term.
                if constexpr (SF_WPAIR) {
                    constexpr int wp_im_row =
                        (im / T::T_M) * T::T_M * T::T_M * T::W_M
                        + (im % T::T_M) * T::W_M;
                    store<T::VEC_C>(g, vj,
                                    u_gc2 + (wp_im_row
                                             + wave_id_m * (T::T_M - 1) * T::W_M)
                                                * stride_c_main,
                                    (wave_id_n * T::COM_REP_N + j) * T::W_N);
                } else {
                    store<T::VEC_C>(g, vj,
                                    u_gc2 + im * T::T_M * T::W_M * stride_c_main,
                                    (wave_id_n * T::COM_REP_N + j) * T::W_N);
                }
            });
        });
    };

    if (direct_store) {
        if constexpr (!std::is_void_v<D_OUT>) {
            D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                           + (size_t)batch_id * kargs.stride_c_batch
                           + (size_t)row * kargs.stride_c
                           + (size_t)col;
            auto g_out = make_gmem(out_ptr,
                (unsigned int)rows_avail * (unsigned int)kargs.stride_c * sizeof(D_OUT));
            store_c(g_out);
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

    // Fused reduce tail: the last workgroup to finish a tile sums the split-K
    // workspace slices into Y itself, so the split_k>1 path needs no second
    // launch. Only compiled for the D_OUT (Y-typed) instantiations.
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
        // Every thread branches on this below, so lane 0's write has to be
        // retired (not merely issued) before the barrier lets the rest read it.
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

// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 BF16 GEMM, C[M, N] = A[M, K] @ B[N, K]^T. Symmetric 4-wave pipeline:
// every wave both TDM-loads and runs WMMA, splitting M 32 rows each and holding
// its own ~256 fp32 VGPR/lane of accumulators live over the whole K loop.
//
// OFFLINE-ONLY HEADER. Unlike every other pipeline under include/gfx1250/, this
// one is NOT compiled by the aiter JIT. It needs three things a release ROCm
// toolchain does not have -- __builtin_amdgcn_pin_vgpr, amdgpu_num_vgpr(1024)
// and -mllvm -amdgpu-expert-scheduling-mode -- so it is compiled ahead of time
// by gen_co/build_co.py into gen_co/gfx1250/*.co and loaded at runtime by
// opus_co_launch_gfx1250.cuh. The generated host launcher includes only the
// TRAITS header, never this one.
//
// The body below is a verbatim port of the standalone kernel
// (gcnasm/opus_gemm_compiler_sample/gemm_a16w16_4wave_compute_kernel_template.hpp)
// with exactly one change: it is a __device__ function instead of a __global__
// one, so the offline stub TU can wrap it in an extern "C" entry point whose
// symbol name is predictable. The register pinning in here is fragile (see the
// OPUS_PIN comments), so nothing else -- naming included -- was touched.
#pragma once

#include "../opus_gemm_utils.cuh"   // opus.hpp + hip_minimal split, ceil_div
#include "opus_gemm_traits_a16w16_gfx1250.cuh"

#include <cstdint>

#if !defined(__gfx1250__)
#error "the 4wave_compute pipeline is gfx1250-only (WMMA 16x16x32 + TDM); build with --offload-arch=gfx1250"
#endif

namespace gemm_4wave_compute_gfx1250 {

using opus::operator""_I;

// The placement request, and the only thing the nopin build drops. It rides on
// the written value, so which writes carry one is visible in the source.
#if OPUS_NO_PIN
#define OPUS_PIN(v, reg) (v)
#else
#define OPUS_PIN(v, reg) __builtin_amdgcn_pin_vgpr(v, reg)
#endif

// A bank of N pinned values at Base, Base+Stride, ... One NAMED member each:
// a pin is as wide as the value using it, and an element-wise write goes through
// a GEP and pins nothing, so every slot must be assigned WHOLE and by name. That
// rules out an array; recursion gives the same codegen as a hand-written list
// while letting N follow the tile.
template <typename V, int Base, int Stride, int N>
struct pinned_bank {
    V                                              head = OPUS_PIN(V{}, Base);
    pinned_bank<V, Base + Stride, Stride, N - 1>   tail;

    template <int I>
    __device__ V get() const {
        if constexpr (I == 0) return head;
        else                  return tail.template get<I - 1>();
    }
    template <int I>
    __device__ void set(V v) {
        if constexpr (I == 0) head = OPUS_PIN(v, Base);
        else                  tail.template set<I - 1>(v);
    }
    // Deliberately unpinned: the declaration above already placed the slot, and
    // re-pinning here measured no different.
    __device__ void zero() { head = V{}; tail.zero(); }
};

template <typename V, int Base, int Stride>
struct pinned_bank<V, Base, Stride, 0> {
    template <int I> __device__ V    get() const { return V{}; }
    template <int I> __device__ void set(V) {}
    __device__ void zero() {}
};

// Kernel-internal derived traits: WMMA decomposition, N-subtiles, TDM windows.
template<typename UT>
struct kernel_traits {
    static constexpr int BLOCK_SIZE = UT::BLOCK_SIZE;

    using DataA   = typename UT::D_A;
    using DataB   = typename UT::D_B;
    using DataC   = typename UT::D_C;
    using DataAcc = typename UT::D_ACC;

    static constexpr int kVecA = UT::VEC_A;
    static constexpr int kVecB = UT::VEC_B;

    static constexpr int kBlockM = UT::B_M;
    static constexpr int kBlockN = UT::B_N;
    static constexpr int kBlockK = UT::B_K;

    // WMMA 16x16x32 (gfx1250 bf16).
    static constexpr int kWmmaM = 16, kWmmaN = 16, kWmmaK = 32;

    static constexpr int kWarp = 32;                       // gfx1250 wave size
    // 32 device / 64 host -- both passes must decompose to the same vtype_c.
    static constexpr int kWarpRt = opus::get_warp_size();
    static constexpr int kNumWaves = BLOCK_SIZE / kWarp;
    static_assert(kNumWaves == 4, "4wave_compute is locked to 4 waves");

    // The 4 waves split M; each owns the full N.
    static constexpr int kTileM = 4;
    static constexpr int kTileN = 1;
    static constexpr int kTileK = 1;

    static constexpr int kExpM = kBlockM / (kWmmaM * kTileM);
    static constexpr int kExpN = kBlockN / (kWmmaN * kTileN);
    static_assert(kExpM * (kWmmaM * kTileM) == kBlockM, "B_M must be a multiple of kWmmaM*kTileM");
    static_assert(kExpN * (kWmmaN * kTileN) == kBlockN, "B_N must be a multiple of kWmmaN*kTileN");

    // N is processed in kNumNSub msb groups; the group WIDTH is the real knob.
    // One msb issues kRElemB = 4 * kExpNPerSub ds_reads and DScnt is 6 bits, so
    // the width must stay <= 15. Widest power of two that divides kExpN, capped
    // at 4 -- the only width with measured numbers behind it. Deriving the SPLIT
    // instead of fixing kNumNSub == 4 is what admits any B_N multiple of 16;
    // fixing it forced B_N to a power-of-two multiple of 64. B_N = 256 still
    // resolves to 4, so the reference config is unchanged.
    static constexpr int kExpNPerSub = (kExpN % 4 == 0) ? 4 : (kExpN % 2 == 0) ? 2 : 1;
    static constexpr int kNumNSub    = kExpN / kExpNPerSub;
    static_assert(4 * kExpNPerSub <= 63, "a msb's ds_read count must fit the 6-bit DScnt");

    static constexpr int kExpKHalf   = 2;                        // K-tiles per ds half
    static constexpr int kKHalfElems = kWmmaK * kExpKHalf;
    static_assert(kBlockK % kKHalfElems == 0, "B_K must be a multiple of kWmmaK*kExpKHalf");
    static constexpr int kHalvesPerSlot = kBlockK / kKHalfElems;
    // A is held in exactly two register buffers (v_a[2]), filled by the only two
    // load_A calls in the K loop, and indexed v_a[tile / kNumNSub]. So a slot may
    // hold exactly two K-halves: B_K == 2 * kWmmaK * kExpKHalf == 128. A wider
    // B_K indexes past v_a and reads whatever the allocator left there.
    static_assert(kHalvesPerSlot == 2,
                  "B_K must be 128: the A register buffer holds exactly two K-halves");

    static constexpr int kTdmK      = kBlockK;                   // one TDM per B_K slot
    static constexpr int kSmemPitch = UT::SMEM_PITCH;

    // Half-row TDM: w0+w1 split the A rows, w2+w3 split the B rows.
    static constexpr int kTdmLoadPart = 2;
    static constexpr int kARowsTdm    = kBlockM / kTdmLoadPart;
    static constexpr int kBRowsTdm    = kBlockN / kTdmLoadPart;
    static constexpr int kSlotBytesTdmA = kARowsTdm * kSmemPitch * (int)sizeof(DataA);
    static constexpr int kSlotBytesTdmB = kBRowsTdm * kSmemPitch * (int)sizeof(DataB);

    static constexpr int kSlotElemsA = kBlockM * kSmemPitch;
    static constexpr int kSlotElemsB = kBlockN * kSmemPitch;
    static constexpr int kSlotBytesA = UT::SLOT_BYTES_A;
    static constexpr int kSlotBytesB = UT::SLOT_BYTES_B;
    static constexpr int kSegBytesA  = UT::SEG_BYTES_A;
    static constexpr int kLdsTotalBytes = UT::LDS_BYTES;

    static constexpr int kNumSlots   = UT::NUM_SLOTS;
    // This pipeline's K loop is hand-unrolled three steps at a time.
    static_assert(kNumSlots == 3, "the unrolled K loop is written out for P == 3");
    static constexpr int kClusterWgM = UT::CLUSTER_WG_M;
    static constexpr int kClusterWgN = UT::CLUSTER_WG_N;

    // WMMA register decomposition (from kWarpRt so dev/host agree).
    static constexpr int kReptA = kWmmaM * kWmmaK / kWarpRt / kVecA;
    static constexpr int kReptB = kWmmaN * kWmmaK / kWarpRt / kVecB;
    static constexpr int kGrpKA = kWarpRt / kWmmaM;
    static constexpr int kGrpKB = kWarpRt / kWmmaN;

    // One b128 read vector of pad per row breaks the power-of-two bank conflict.
    using Padding = opus::tdm_traits::padding_auto<DataA, kBlockK>;
    static_assert(opus::is_same_v<Padding, opus::tdm_traits::padding<DataA, kBlockK, UT::PAD_ELEMS>>,
                  "tdm_traits::padding_auto must agree with the hand-written tier for this dtype and row length");
    // gemm_defs sizes the segments from SMEM_PITCH, the D# is programmed from the
    // policy; they have to be the same number.
    static_assert(Padding::pitch_elements == kSmemPitch, "LDS row pitch must match the padding policy");

    // One window type for both operands; only tile_dim1 differs, and that is
    // patched into the D# at issue time.
    using Window  = opus::tdm<DataA, opus::seq<kTdmK, kARowsTdm>, Padding>;
};

// kExpN-shadowed view of the traits so make_layout_rb builds the B read layout
// for ONE msb (kExpNPerSub N-tiles) instead of the full N.
template<typename T>
struct sub_traits : T { static constexpr int kExpN = T::kExpNPerSub; };

// A operand (M x K) smem read layout; wave_m selects this wave's M sub-tile.
template<typename T>
inline __device__ auto make_layout_ra(int lane_id, int wave_m) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpM>{}, opus::number<T::kTileM>{}, opus::number<T::kWmmaM>{},
        opus::number<T::kExpKHalf>{}, opus::number<T::kTileK>{},
        opus::number<T::kReptA>{}, opus::number<T::kGrpKA>{}, opus::number<T::kVecA>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<0>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSmemPitch, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_m, lane_id % T::kWmmaM, 0, lane_id / T::kWmmaM}));
}

// B operand (N x K) smem read layout. The N-dim order must be (kExpN outer,
// kTileN inner, kWmmaN) to match the C-store layout.
template<typename T>
inline __device__ auto make_layout_rb(int lane_id, int wave_n) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpN>{}, opus::number<T::kTileN>{}, opus::number<T::kWmmaN>{},
        opus::number<T::kExpKHalf>{}, opus::number<T::kTileK>{},
        opus::number<T::kReptB>{}, opus::number<T::kGrpKB>{}, opus::number<T::kVecB>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<0>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSmemPitch, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_n, lane_id % T::kWmmaN, 0, lane_id / T::kWmmaN}));
}

// Everything the pipeline runs on: shapes derived from the block tile, the pinned
// register placement, and the scheduling policy. The kernel body only reads these.
template<typename T>
struct pipeline_traits {
    static constexpr int kSlots  = T::kNumSlots;                   // K-step ring depth
    static constexpr int kNSub   = T::kNumNSub;                    // N sub-tiles (msb)
    static constexpr int kHalves = T::kHalvesPerSlot;               // K-halves per slot
    static constexpr int kTiles  = kHalves * kNSub;                 // msb-tiles per K step
    static constexpr int kExpM   = T::kExpM;
    static constexpr int kExpN   = T::kExpNPerSub;
    static constexpr int kExpK   = T::kExpKHalf;
    static constexpr int kNSubN  = 2;                               // N-tiles per 32x32 sub-tile
    static constexpr int kWmmaPerSub  = kExpM * kExpK * kNSubN;     // WMMAs in one 32x32x64 sub-tile
    static constexpr int kWmmaPerTile = kWmmaPerSub * kNSubN;
    static constexpr int kElemA  = T::kWmmaM * T::kWmmaK / T::kWarp;  // per-WMMA A elements
    static constexpr int kElemC  = T::kWmmaM * T::kWmmaN / T::kWarp;  // per-WMMA C elements
    static constexpr int kSlice  = kExpM * kExpN;                   // C slices per msb
    static constexpr int kNumAcc = kNSub * kSlice;
    static constexpr int kSubRows = kExpN * T::kTileN * T::kWmmaN;  // LDS rows per msb
    static constexpr auto kKStep = opus::number<T::kBlockK>{};

    // ds_read issue counts, from the same layouts the reads use.
    using LayoutRA = decltype(make_layout_ra<T>(0, 0));
    using LayoutRB = decltype(make_layout_rb<sub_traits<T>>(0, 0));
    static constexpr int kRElemA = opus::layout_load_traits<LayoutRA, T::kVecA>::r_elem.value;
    static constexpr int kRElemB = opus::layout_load_traits<LayoutRB, T::kVecB>::r_elem.value;

    // Accumulators start at v256 (v0..v255 holds A, ds offsets and C staging) and
    // B's two banks sit directly above them, so WMMA src0 never has to switch
    // VGPR_MSB. kBPinBase FOLLOWS the accumulator block rather than being a fixed
    // 512: pinned at 512, a tile with fewer accumulators still allocated the full
    // file and no smaller tile could ever cost fewer registers.
    static constexpr int kAccVgpr  = kNumAcc * kElemC;
    static constexpr int kBFrags   = kNSubN * kNSubN * kExpK;  // Bt_get index space
    static constexpr int kBFragVgpr = 8;                       // bf16x16 = 8 VGPR
    static constexpr int kAccBase  = 256;
    static constexpr int kBPinBase = kAccBase + kAccVgpr;
    // The prefetch issues one load_B_sub per WMMA of a 32x32x64 sub-tile and must
    // land exactly on one buffer's fragment set. This is the real reason B_M is
    // 128: it resolves to kExpM == 2. Generalising B_M means kNSubN == kExpM.
    static_assert(kWmmaPerSub == kBFrags,
                  "the prefetch window must span exactly one buffer's B fragments");
    // Bt_get's n_idx spans [0, kNSubN^2), so one msb must hold exactly that many
    // N-tiles. With kNSubN == 2 that is kExpNPerSub == 4, i.e. B_N a multiple of
    // 64 -- a narrower msb would index B fragments belonging to the next group.
    static_assert(kExpN == kNSubN * kNSubN,
                  "one msb must hold kNSubN^2 N-tiles: B_N must be a multiple of 64");
    static_assert(kBPinBase + 2 * kBFrags * kBFragVgpr <= 1024,
                  "the pinned accumulator + B block overruns the VGPR file");

    static constexpr int kBarrierAhead = 3;   // WMMAs between barrier signal and wait
    static constexpr int kTdmBuildW    = 4;   // WMMA in the last tile that builds the D#
    // A group's count is a MAX, not a quota, so kIlSalu stays 1: any more lets the
    // solver pile scalars into one WMMA gap and leave its neighbours empty.
    static constexpr int kIlWmma = 1, kIlDs = 2, kIlSalu = 1;
    // sched_group_barrier masks (AMDGPUIGroupLP).
    static constexpr unsigned kDsRead = 0x100u, kMfma = 0x08u, kSalu = 0x04u,
                              kValu = 0x002u, kDsWrite = 0x200u;

    // C staging: msb m is final once tile kNSub+m retires; the last one only on the
    // very last tile, so kNSub-1 is all the peeled step can fuse.
    static constexpr int kFusedMsb = kNSub - 1;
    static constexpr int kCRowsTdm = T::kBlockM / T::kNumWaves;
};

}  // namespace gemm_4wave_compute_gfx1250

// The pipeline body. __launch_bounds__ / amdgpu_num_vgpr / __cluster_dims__
// describe the kernel ENTRY, not the body, so they live on the extern "C"
// wrapper the offline stub TU emits (see gen_co/build_co.py) rather than here.
template<typename UserTraits>
__device__ void gemm_a16w16_4wave_compute_body_gfx1250(
    opus_gemm_4wave_compute_kargs_gfx1250 kargs) {
    using namespace opus;
    using namespace gemm_4wave_compute_gfx1250;
    using opus::operator""_I;
    using T  = kernel_traits<opus::remove_cvref_t<UserTraits>>;
    using PT = pipeline_traits<T>;
    using DataA   = typename T::DataA;
    using DataB   = typename T::DataB;
    using DataC   = typename T::DataC;
    using DataAcc = typename T::DataAcc;

    const int wave_id = __builtin_amdgcn_readfirstlane((int)opus::waveid_in_workgroup());
    const int lane_id = (int)opus::lane_id();

    // Cluster (CWGM x CWGN): cluster_id = super-tile, cluster_workgroup_id = local tile.
    const int cluster_x = (int)__builtin_amdgcn_cluster_id_x();
    const int cluster_y = (int)__builtin_amdgcn_cluster_id_y();
    const int local_x   = (int)__builtin_amdgcn_cluster_workgroup_id_x();
    const int local_y   = (int)__builtin_amdgcn_cluster_workgroup_id_y();
    const int batch_id  = (int)__builtin_amdgcn_workgroup_id_z();
    const int tile_row  = (cluster_x * T::kClusterWgM + local_x) * T::kBlockM;
    const int tile_col  = (cluster_y * T::kClusterWgN + local_y) * T::kBlockN;

    // Coordinates, not tile counts: no runtime division. Acted on in the prologue,
    // not here -- a tile-less workgroup still owes the cluster barrier its arrival.
    const bool tile_oob = tile_row >= kargs.m || tile_col >= kargs.n;

    // A is multicast to the CWGN peers that fix M, B to the CWGM peers that fix N.
    // Fan-out <=1 folds to "no multicast", which a non-cluster launch needs to be
    // correct, not just fast.
    const auto mask_a = opus::tdm_traits::peers_along_y<T::kClusterWgM, T::kClusterWgN>();
    const auto mask_b = opus::tdm_traits::peers_along_x<T::kClusterWgM, T::kClusterWgN>();

    const int stride_a = kargs.stride_a;
    const int stride_b = kargs.stride_b;
    const int stride_c = kargs.stride_c;

    const DataA* base_a = reinterpret_cast<const DataA*>(kargs.ptr_a)
                        + (size_t)batch_id * (size_t)kargs.stride_a_batch;
    const DataB* base_b = reinterpret_cast<const DataB*>(kargs.ptr_b)
                        + (size_t)batch_id * (size_t)kargs.stride_b_batch;
    // C's batch stride is derived rather than passed: it keeps kargs at 64 B,
    // and m * stride_c stays right for a row-padded C, which the host's old
    // M * N did not. 64-bit because it crosses 2^31 at 4 GiB of bf16.
    DataC*       base_c = reinterpret_cast<DataC*>(kargs.ptr_c)
                        + (size_t)batch_id * (size_t)((int64_t)kargs.m * stride_c);

    const int k_steps = ceil_div(kargs.k, T::kBlockK);
    if (k_steps <= 0) return;

    __shared__ char lds_buf[T::kLdsTotalBytes];
    DataA* smem_a = reinterpret_cast<DataA*>(lds_buf);
    DataB* smem_b = reinterpret_cast<DataB*>(lds_buf + T::kSegBytesA);

    // ONE window type for both operands: two would force an `if (is_operand_a)`
    // around every issue, and that branch ends the scheduling region, stranding
    // the descriptor SALU in a block with no WMMA to pair it with.
    using WindowU = typename T::Window;
    // DataType is the one field not patchable at issue time -- it sets the D#'s
    // element size and scales global_offset_bytes.
    static_assert(opus::is_same_v<DataA, DataB>,
                  "folding the A/B TDM windows patches only tile_dim1; "
                  "mixed A/B element types need two windows");

    // Per-wave TDM stream: all 4 waves load every K step, half the rows each.
    //   w0/w1: A rows [0, kARowsTdm) / [kARowsTdm, B_M)
    //   w2/w3: B rows [0, kBRowsTdm) / [kBRowsTdm, B_N)
    const bool is_operand_a = (wave_id <= 1);
    const int  row_half_id  = wave_id & 1;       // which half of the operand's rows this wave takes

    // Build both and select once, not seven selects over their inputs. The extents
    // are the WHOLE tensor's, so the ragged last tile needs no handling.
    auto window_a = opus::make_tdm<WindowU>(
        (uint32_t)(reinterpret_cast<uintptr_t>(smem_a) + row_half_id * T::kSlotBytesTdmA),
        (const void*)base_a,
        opus::make_layout<-1>(
            opus::make_tuple((uint32_t)kargs.m, (uint32_t)kargs.k),
            opus::make_tuple((uint64_t)stride_a, 1_I),
            opus::make_tuple((uint32_t)(tile_row + row_half_id * T::kARowsTdm), 0u)));
    window_a.set_workgroup_mask(mask_a);
    window_a.set_tile_dim1((uint32_t)T::kARowsTdm);

    auto window_b = opus::make_tdm<WindowU>(
        (uint32_t)(reinterpret_cast<uintptr_t>(smem_b) + row_half_id * T::kSlotBytesTdmB),
        (const void*)base_b,
        opus::make_layout<-1>(
            opus::make_tuple((uint32_t)kargs.n, (uint32_t)kargs.k),
            opus::make_tuple((uint64_t)stride_b, 1_I),
            opus::make_tuple((uint32_t)(tile_col + row_half_id * T::kBRowsTdm), 0u)));
    window_b.set_workgroup_mask(mask_b);
    window_b.set_tile_dim1((uint32_t)T::kBRowsTdm);

    auto tdm_window = is_operand_a ? window_a : window_b;
    // Not window state -- the ring walk scales by it -- so a plain value select.
    const int slot_elems = is_operand_a ? T::kSlotElemsA : T::kSlotElemsB;

    // Ring cursor as separate index and offset: folding them into one cursor that
    // wraps on a runtime P * slot_elems is 16 fewer instructions but 0.5% slower,
    // because the add-compare-select then sits in series.
    int slot_index = 0, slot_offset = 0;
    auto advance_slot = [&]() __attribute__((always_inline)) {
        const bool wrap = (slot_index + 1 == PT::kSlots);
        slot_offset += (wrap ? 1 - PT::kSlots : 1) * slot_elems;
        slot_index   = wrap ? 0 : slot_index + 1;
    };

    // WMMA setup: per-msb sub-tile mma, M-split with wave_m = wave_id.
    auto mma = make_tiled_mma<DataA, DataB, DataAcc>(
        seq<T::kExpM, T::kExpNPerSub, T::kExpKHalf>{},
        seq<T::kTileM, T::kTileN, T::kTileK>{},
        seq<T::kWmmaM, T::kWmmaN, T::kWmmaK>{}, wmma_adaptor_swap_ab{});
    auto u_ra = make_layout_ra<T>(lane_id, wave_id);              // A: wave's M rows
    auto u_rb = make_layout_rb<sub_traits<T>>(lane_id, 0);        // B: one msb

    using VA = typename decltype(mma)::vtype_a;
    // Must lower through the builtin, not asm: asm plus a pinned physreg tuple hits
    // a copyPhysReg subreg assert. swap_ab gives hardware wmma(D, src0=B, src1=A, C).
    using WMMA = typename decltype(mma)::MMA;

    // opus::vector_t throughout: is_vector_v is what slice/set_slice/cast/store
    // dispatch on.
    using bf16x16  = opus::vector_t<DataA, 16>;     // one WMMA's B operand
    using f32x8    = opus::vector_t<DataAcc, 8>;    // one WMMA's C slice
    using cx8      = opus::vector_t<DataC, 8>;      // one C store unit: 8 N, 16 B
    using u32x4    = opus::vector_t<unsigned, 4>;   // the same 16 B, as select-able DWORD
    static_assert(sizeof(cx8) == sizeof(u32x4), "the C store unit must be a whole number of DWORD");

    // Both K-halves live at once; the allocator overlaps them, which is where the K
    // loop's last two va_vdst(0) drains come from. Pinning them apart removes those
    // and measured a wash for 64 extra VGPRs.
    VA v_a[2];

    // Double-buffered B, one pinned value per (buffer, sub-tile). Unpinned, the
    // allocator overlaps the buffers and tile t+1's prefetch lands on registers a
    // tile-t WMMA has not read: a VA_VDST drain. The two buffers are one bank so
    // a single base covers both.
    pinned_bank<bf16x16, PT::kBPinBase, PT::kBFragVgpr, 2 * PT::kBFrags> b_frag;
    auto Bt_set = [&](auto bufN, auto ksubN, bf16x16 v) __attribute__((always_inline)) {
        constexpr int bi = decltype(bufN)::value, ki = decltype(ksubN)::value;
        static_assert(bi < 2 && ki < PT::kBFrags, "B sub-tile out of range");
        b_frag.template set<bi * PT::kBFrags + ki>(v);
    };
    auto Bt_get = [&](auto bufN, auto ksubN) __attribute__((always_inline)) -> bf16x16 {
        constexpr int bi = decltype(bufN)::value, ki = decltype(ksubN)::value;
        return b_frag.template get<bi * PT::kBFrags + ki>();
    };

    // Per-lane ds_read offsets, hoisted: slot/msb/half only move the smem base.
    // Keep the `base_elem + offs[i]` shape -- the address-mode matcher picks its
    // immediates off `uniform + lane + static_i`.
    auto offs_a = opus::layout_to_offsets<T::kVecA>(u_ra);
    auto offs_b = opus::layout_to_offsets<T::kVecB>(u_rb);

    // One K-half of A into v_a[abuf].
    auto load_A = [&](int s, int half, int abuf) __attribute__((always_inline)) {
        const int base_elem = s * T::kSlotElemsA + half * T::kKHalfElems;
        auto sm = make_smem(smem_a);
        VA r;
        opus::static_for<PT::kRElemA>([&](auto iN) __attribute__((always_inline)) {
            constexpr int i = iN.value;
            opus::set_slice(r, sm.template load<T::kVecA>(base_elem + offs_a[i]),
                            opus::number<i * T::kVecA>{}, opus::number<(i + 1) * T::kVecA>{});
        });
        v_a[abuf] = r;
    };

    // One B sub-tile (bf16x16 = one WMMA's B = 2 ds_read) straight into its pinned
    // slot, so the K loop can interleave single prefetches with single WMMAs.
    auto load_B_sub = [&](int s, int half, int msb, auto bufN, auto ksubN) __attribute__((always_inline)) {
        constexpr int ksub = decltype(ksubN)::value;
        const int be = s * T::kSlotElemsB + msb * PT::kSubRows * T::kSmemPitch + half * T::kKHalfElems;
        auto sm = make_smem(smem_b);
        bf16x16 sub;
        opus::static_for<2>([&](auto eN) __attribute__((always_inline)) {
            constexpr int e = eN.value;
            opus::set_slice(sub, sm.template load<T::kVecB>(be + offs_b[2 * ksub + e]),
                            opus::number<e * T::kVecB>{}, opus::number<(e + 1) * T::kVecB>{});
        });
        Bt_set(bufN, ksubN, sub);
    };

    // Persistent accumulators, one pinned value per WMMA C slice: a pin is as wide
    // as the value using it, so one f32x256 would pin its initialiser only.
    pinned_bank<f32x8, PT::kAccBase, PT::kElemC, PT::kNumAcc> acc;
    auto acc_set = [&](auto aiN, f32x8 v) __attribute__((always_inline)) {
        constexpr int ai = decltype(aiN)::value;
        static_assert(ai < PT::kNumAcc, "accumulator index out of range");
        acc.template set<ai>(v);
    };
    auto acc_get = [&](auto aiN) __attribute__((always_inline)) -> f32x8 {
        return acc.template get<decltype(aiN)::value>();
    };

    // C staging, hoisted above the K loop so the peeled step can fuse it. All of it
    // is lane identity; `smem_c` is the one runtime piece, aimed at an idle ring
    // slot just before the step that first writes it.
    static_assert(T::kBlockM * T::kBlockN * (int)sizeof(DataC)
                      <= T::kSlotElemsB * (int)sizeof(DataB),
                  "the C tile must fit in the one B ring slot it stages through");

    // Row pitch 512 B is a multiple of 32 dwords, so M does not move the LDS bank
    // and a whole row group would hit one bank pair -- 3-23% of runtime. Rotating
    // each lane's groups by lane_m spreads them over all 32; a register number
    // cannot be rotated per lane, hence the two v_cndmask stages below.
    // swap_ab transposes the WMMA C layout: grpn_c walks M, grpm_c splits N.
    using mma_t = opus::remove_cvref_t<decltype(mma)>;
    static_assert(mma_t::grpn_c == T::kWmmaM, "swap_ab must put grpn_c on the M axis");
    static_assert(mma_t::grpm_c * (T::kWmmaN / mma_t::grpm_c) == T::kWmmaN, "N must split evenly over grpm_c");
    static_assert((PT::kExpN & (PT::kExpN - 1)) == 0, "the group rotation masks with kExpN - 1");
    static_assert(PT::kExpN * T::kWmmaN == PT::kSubRows, "a msb sub-tile is exactly kExpN groups of kWmmaN");

    const int lane_m = lane_id % mma_t::grpn_c;                              // this lane's row, 0..15
    const int lane_n = (lane_id / mma_t::grpn_c) * (T::kWmmaN / mma_t::grpm_c);
    const int rot    = lane_m & (PT::kExpN - 1);
    const int c_base = (wave_id * T::kWmmaM + lane_m) * T::kBlockN + lane_n;
    DataC* smem_c = nullptr;

    // Cast one msb to bf16 and stage it, rotated. Each store burst reads a register
    // the cvt just wrote, so it is fronted with a va_vdst(0) drain, 10 in all; both
    // ways of buying distance measured worse (-0.4 to -1.6% fragmenting the bursts,
    // -0.1 to -2.1% hoisting the casts).
    auto store_msb = [&](auto mN) __attribute__((always_inline)) {
        constexpr int msb = decltype(mN)::value;
        auto s_c = make_smem(smem_c + msb * PT::kSubRows);
        opus::static_for<PT::kExpM>([&](auto imN) __attribute__((always_inline)) {
            constexpr int im = decltype(imN)::value;
            u32x4 g[PT::kExpN];
            opus::static_for<PT::kExpN>([&](auto inN) __attribute__((always_inline)) {
                constexpr int in = decltype(inN)::value;
                const cx8 v = opus::cast<DataC>(
                    acc_get(opus::number<msb * PT::kSlice + (im * PT::kExpN + in)>{}));
                g[in] = __builtin_bit_cast(u32x4, v);
            });
            // Rotate the four groups left by `rot`, as two v_cndmask stages.
            u32x4 t[PT::kExpN], o[PT::kExpN];
            #pragma unroll
            for (int q = 0; q < PT::kExpN; ++q)
                for (int i = 0; i < 4; ++i)
                    t[q][i] = (rot & 1) ? g[(q + 1) & (PT::kExpN - 1)][i] : g[q][i];
            #pragma unroll
            for (int q = 0; q < PT::kExpN; ++q)
                for (int i = 0; i < 4; ++i)
                    o[q][i] = (rot & 2) ? t[(q + 2) & (PT::kExpN - 1)][i] : t[q][i];
            opus::static_for<PT::kExpN>([&](auto sN) __attribute__((always_inline)) {
                constexpr int s = decltype(sN)::value;
                const int n_off = (rot * T::kWmmaN + s * T::kWmmaN) & (PT::kSubRows - 1);
                s_c.template store<8>(__builtin_bit_cast(cx8, o[s]),
                                      c_base + im * (T::kTileM * T::kWmmaM) * T::kBlockN + n_off);
            });
        });
    };

    // Prologue: prime the kSlots-deep ring. Step 0 is drained and published here,
    // steps 1 and 2 at the end of g = 0 and g = 1, so every TDM gets two full
    // compute bodies to complete in.
    acc.zero();
    // Cluster sync before the first multicast TDM, so peers arrive together and GL1
    // can merge their requests instead of releasing each on its timeout.
    __builtin_amdgcn_s_barrier();
    if (wave_id == 0) {
        __builtin_amdgcn_s_barrier_signal(-3);
        __builtin_amdgcn_s_barrier_wait(-3);
    }
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

    // Tile-less workgroups leave here, not one line earlier: -3 counts one arrival
    // PER WORKGROUP, so skipping the barrier above hangs every peer. Survivors still
    // name the departed in their masks -- harmless, just a GL1 timeout.
    if (tile_oob) return;
    // Unconditional prime: past K the D#'s tensor_dim0 saturates to 0, so a step
    // beyond the last is a zero-extent DMA that only bumps tensorcnt -- which is
    // what lets the hot loop wait tensorcnt(1) unguarded.
    opus::static_for<PT::kSlots>([&](auto iN) __attribute__((always_inline)) {
        if constexpr (iN.value > 0) { tdm_window.move(PT::kKStep); advance_slot(); }
        // Fused: the prologue has no WMMA stream to spread the descriptor across.
        tdm_window.async_load(slot_offset);
    });
    __builtin_amdgcn_s_wait_tensorcnt(PT::kSlots - 1);   // land step 0, rest stay in flight
    __builtin_amdgcn_s_barrier_signal(-1);               // publish step 0
    __builtin_amdgcn_s_barrier_wait(-1);

    // Slot 0's tile-0 dependencies, the same set the last tile issues for the next
    // slot. The sched_barrier forces them to ISSUE before the first body's WMMAs
    // so tile 0's counted wait covers loads already in flight.
    load_A(0, 0, 0);
    opus::static_for<8>([&](auto kN) __attribute__((always_inline)) {
        load_B_sub(0, 0, 0, opus::number<0>{}, kN);
    });
    load_A(0, 1, 1);
    __builtin_amdgcn_sched_barrier(0);

    // Unrolled by kSlots so the ring index stays a literal: a runtime
    // `s * slot_elems` feeds a ds_read address through a VALU and earns an
    // s_wait_alu va_vdst(0) that drains the whole WMMA pipe.
    static_assert(PT::kSlots == 3, "the call sites below are written out for kSlots == 3");
    const int n_body = k_steps - 1;                     // every step but the peeled last one
    const int n_full = n_body - n_body % PT::kSlots;    // those covered by whole kSlots groups

    // What one WMMA's gap may co-issue, in the order the solver must honour.
    auto sched_wmma_gap = [](auto n_ds) __attribute__((always_inline)) {
        __builtin_amdgcn_sched_group_barrier(PT::kSalu, PT::kIlSalu, 0);
        __builtin_amdgcn_sched_group_barrier(PT::kMfma, PT::kIlWmma, 0);
        if constexpr (n_ds.value > 0)
            __builtin_amdgcn_sched_group_barrier(PT::kDsRead, n_ds.value, 0);
    };

    // One K step over kTiles msb-tiles: multiply out of ring slot `cur_slot`, and
    // front-load `next_slot` for the step after. The slots arrive as number<> and
    // every call site passes a constant -- an ordinary int parameter, or a
    // static_for over the remainder, puts the index back in a register and measured
    // 54 extra va_vdst(0) drains, 8% on a K where the remainder is 2 of 5 steps.
    // Every tile but the last prefetches its successor's B; the last one instead
    // carries the barrier handshake, the next slot's tile-0 loads and the next
    // step's TDM, branch-free and spread over its WMMA gaps.
    auto one_step = [&](auto cur_slot, auto next_slot) __attribute__((always_inline)) {
        constexpr int cur = cur_slot.value, next = next_slot.value;
        typename WindowU::descriptor tdm_payload;
        tdm_window.move(PT::kKStep);
        advance_slot();
        opus::static_for<PT::kTiles>([&](auto tileN) __attribute__((always_inline)) {
            constexpr int tile      = tileN.value;
            constexpr int k_half    = tile / PT::kNSub;   // which half of the slot's K
            constexpr int msb       = tile % PT::kNSub;   // which N sub-tile
            constexpr int b_buf     = tile & 1;           // B prefetch double buffer
            constexpr bool is_last  = (tile + 1 == PT::kTiles);
            // Tile 0 is always (k-half 0, msb 0), and waits on the A ds_reads it was
            // front-loaded with; every other tile waits on its B prefetch.
            opus::s_wait_dscnt(opus::number<tile == 0 ? PT::kRElemA : PT::kRElemB>{});
            if constexpr (is_last) {
                // Unguarded, and right for every shape: the prologue primed exactly
                // kSlots and each step issues one. A guard would split the tile from
                // the descriptor SALU below it.
                __builtin_amdgcn_s_wait_tensorcnt(1);
                // WAR: this step refills the slot it reads, and a barrier orders
                // waves, not in-flight LDS traffic -- so retire the reads first.
                opus::s_wait_dscnt(opus::number<0>{});
            }
            opus::static_for<PT::kWmmaPerTile>([&](auto wmmaN) __attribute__((always_inline)) {
                // A tile's WMMAs run sub-tile by sub-tile, each ordered [k][m][n].
                constexpr int wmma    = wmmaN.value;
                constexpr int in_sub  = wmma % PT::kWmmaPerSub;
                constexpr int k_idx   = in_sub / (PT::kExpM * PT::kNSubN);
                constexpr int m_idx   = (in_sub / PT::kNSubN) % PT::kExpM;
                constexpr int n_idx   = (wmma / PT::kWmmaPerSub) * PT::kNSubN + in_sub % PT::kNSubN;
                constexpr int acc_id  = msb * PT::kSlice + m_idx * PT::kExpN + n_idx;
                constexpr int a_first = (m_idx * PT::kExpK + k_idx) * PT::kElemA;
                acc_set(opus::number<acc_id>{},
                        WMMA{}(opus::slice(v_a[k_half], opus::number<a_first>{},
                                           opus::number<a_first + PT::kElemA>{}),
                               Bt_get(opus::number<b_buf>{},
                                      opus::number<n_idx * PT::kExpK + k_idx>{}),
                               acc_get(opus::number<acc_id>{})));
                // At most one event per gap, in gap order. The prefetch feeds the
                // next tile -- on the last one that wraps to the next slot's tile 0,
                // the same eight loads, only placed after the handshake's gap.
                constexpr int pf_tile  = (tile + 1) % PT::kTiles;
                constexpr int pf_first = is_last ? PT::kBarrierAhead : 0;
                constexpr bool do_pf   = wmma >= pf_first && wmma < pf_first + PT::kWmmaPerSub;
                constexpr bool do_hs   = is_last && wmma == PT::kBarrierAhead - 1;

                if constexpr (do_hs) {
                    // Arrive and wait as one: split, a wave that is ahead signals the
                    // next step into this step's count.
                    __builtin_amdgcn_s_barrier();
                    load_A(next, 0, 0);
                }
                // All eight in this tile's first 32x32x64 sub-tile: spreading them
                // over the whole tile cuts the counted s_wait_alu to 27 but measured
                // -0.4 to -3.0%, by flattening load-to-use from 14-22 WMMAs to 14.
                if constexpr (do_pf)
                    load_B_sub(is_last ? next : cur, pf_tile / PT::kNSub, pf_tile % PT::kNSub,
                               opus::number<pf_tile & 1>{}, opus::number<wmma - pf_first>{});
                if constexpr (is_last && wmma == PT::kWmmaPerSub + PT::kTdmBuildW) {
                    // Here, so the last WMMA runs ~18 cycles ahead of the DMA issue;
                    // the D#'s 12 SGPRs stay live until then, of ~36 in use.
                    tdm_payload = tdm_window.make_descriptor(slot_offset);
                    __builtin_amdgcn_sched_barrier(0);
                }
                if constexpr (is_last && wmma == PT::kWmmaPerTile - 1)
                    WindowU::async_load(tdm_payload);

                // The gaps ahead of the handshake carry the WMMA alone.
                if constexpr (is_last && wmma < PT::kBarrierAhead - 1)
                    __builtin_amdgcn_sched_group_barrier(PT::kMfma, PT::kIlWmma, 0);
                else
                    sched_wmma_gap(opus::number<do_pf ? PT::kIlDs
                                              : do_hs ? PT::kRElemA : 0>{});
                // Keeps a prefetch ds_read from sinking to its consumer in the next
                // tile, which is what forced a per-WMMA dscnt(0) drain. Letting SALU
                // cross this boundary instead is ~1.5% slower.
                if constexpr (in_sub == PT::kWmmaPerSub - 1) __builtin_amdgcn_sched_barrier(0);
            });
        });
        __builtin_amdgcn_sched_barrier(0);
        // A half1 last and unguarded: on the final step it reads a slot nobody
        // consumes, but a guard fences those ds_reads into a branch target with no
        // WMMA to hide them behind. SALU does not forward into branches either, so
        // ordering [latch SALU][ds_read] is what covers the latch s_cmp.
        load_A(next, 1, 1);
        __builtin_amdgcn_sched_group_barrier(PT::kSalu, 4, 0);
        __builtin_amdgcn_sched_group_barrier(PT::kDsRead, PT::kRElemA, 0);
    };

    for (int g = 0; g < n_full; g += PT::kSlots) {
        one_step(0_I, 1_I);
        one_step(1_I, 2_I);
        one_step(2_I, 0_I);
    }

    // The at most kSlots-1 leftover steps, dispatched rather than looped: any loop
    // shape holds the main body's accumulators live across them, and only one range
    // can own the pinned tuple. Dispatched, all 640 WMMA stay pinned instead of 444,
    // at 129 fewer instructions. Each step leaves the ring one slot further on.
    int peel_slot = 0;
    const int n_rem = n_body - n_full;
    if (n_rem >= 1) { one_step(0_I, 1_I); peel_slot = 1; }
    if (n_rem >= 2) { one_step(1_I, 2_I); peel_slot = 2; }
    // A step past the last is not a no-op: measured on gfx1250 it zero-fills its
    // whole LDS tile, and C stages in one such slot -- so drain before staging.
    __builtin_amdgcn_s_wait_tensorcnt(0);
    smem_c = reinterpret_cast<DataC*>(
        smem_b + (size_t)((peel_slot + 1 == PT::kSlots) ? 0 : (peel_slot + 1)) * T::kSlotElemsB);

    // The peeled last K step. Nothing follows it, so it drops the front loads, TDM
    // and barrier a body owes the next step, and spends those slots on C staging.
    opus::static_for<PT::kTiles>([&](auto tileN) __attribute__((always_inline)) {
        constexpr int tile     = tileN.value;
        constexpr int k_half   = tile / PT::kNSub;
        constexpr int msb      = tile % PT::kNSub;
        constexpr int b_buf    = tile & 1;
        constexpr bool is_last = (tile + 1 == PT::kTiles);
        constexpr int staged   = tile - PT::kNSub - 1;   // the msb tile tile-1 completed
        opus::s_wait_dscnt(opus::number<tile == 0 ? PT::kRElemA : PT::kRElemB>{});
        opus::static_for<PT::kWmmaPerTile>([&](auto wmmaN) __attribute__((always_inline)) {
            constexpr int wmma    = wmmaN.value;
            constexpr int in_sub  = wmma % PT::kWmmaPerSub;
            constexpr int k_idx   = in_sub / (PT::kExpM * PT::kNSubN);
            constexpr int m_idx   = (in_sub / PT::kNSubN) % PT::kExpM;
            constexpr int n_idx   = (wmma / PT::kWmmaPerSub) * PT::kNSubN + in_sub % PT::kNSubN;
            constexpr int acc_id  = msb * PT::kSlice + m_idx * PT::kExpN + n_idx;
            constexpr int a_first = (m_idx * PT::kExpK + k_idx) * PT::kElemA;
            acc_set(opus::number<acc_id>{},
                    WMMA{}(opus::slice(v_a[k_half], opus::number<a_first>{},
                                       opus::number<a_first + PT::kElemA>{}),
                           Bt_get(opus::number<b_buf>{},
                                  opus::number<n_idx * PT::kExpK + k_idx>{}),
                           acc_get(opus::number<acc_id>{})));
            if constexpr (wmma < PT::kWmmaPerSub && !is_last) {
                load_B_sub(peel_slot, (tile + 1) / PT::kNSub, (tile + 1) % PT::kNSub,
                           opus::number<(tile + 1) & 1>{}, opus::number<wmma>{});
                __builtin_amdgcn_sched_group_barrier(PT::kMfma, PT::kIlWmma, 0);
                __builtin_amdgcn_sched_group_barrier(PT::kDsRead, PT::kIlDs, 0);
            } else if constexpr (staged >= 0 && staged < PT::kFusedMsb) {
                // One msb is 32 cvt and 8 ds_store, spread over the slots this tile
                // has left: kWmmaPerSub, or a whole tile on the last one, which has
                // no prefetch to place.
                constexpr int n_slot = is_last ? PT::kWmmaPerTile : PT::kWmmaPerSub;
                if constexpr (wmma == PT::kWmmaPerTile - n_slot) store_msb(opus::number<staged>{});
                __builtin_amdgcn_sched_group_barrier(PT::kValu, 32 / n_slot, 0);
                __builtin_amdgcn_sched_group_barrier(PT::kMfma, PT::kIlWmma, 0);
                __builtin_amdgcn_sched_group_barrier(PT::kDsWrite, 8 / n_slot, 0);
            } else {
                __builtin_amdgcn_sched_group_barrier(PT::kMfma, PT::kIlWmma, 0);
            }
            if constexpr (in_sub == PT::kWmmaPerSub - 1) __builtin_amdgcn_sched_barrier(0);
        });
    });
    // Epilogue: the B_M x B_N tile goes out through the A/B ring -- dead once the
    // last DMA retired -- as one TDM per wave. It moves whole B_N rows, so the
    // ragged tile needs no guard: the D# clamps origin against the extents.
    using StoreWin = opus::tdm<DataC, opus::seq<T::kBlockN, PT::kCRowsTdm>>;

    // Whatever the peeled step did not stage, one msb at a time: interleaved, every
    // ds_store inherits the VALU->LDS operand hazard.
    opus::static_for<PT::kNSub - PT::kFusedMsb>([&](auto mN) __attribute__((always_inline)) {
        store_msb(opus::number<PT::kFusedMsb + decltype(mN)::value>{});
        __builtin_amdgcn_sched_group_barrier(PT::kValu, 48, 0);
        __builtin_amdgcn_sched_group_barrier(PT::kDsWrite, 8, 0);
    });

    // The barrier orders the waves against each other but not against their own
    // in-flight LDS traffic, so the ds_writes are retired explicitly before a peer's
    // DMA may read the rows they land in.
    opus::s_wait_dscnt(opus::number<0>{});
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

    // The store partitions M contiguously, unlike the compute partition, where a
    // wave's own accumulator rows are two strided 16-row chunks.
    auto window_c = opus::make_tdm<StoreWin>(
        (uint32_t)(reinterpret_cast<uintptr_t>(smem_c)
                   + (size_t)wave_id * PT::kCRowsTdm * T::kBlockN * sizeof(DataC)),
        (void*)base_c,
        opus::make_layout<-1>(
            opus::make_tuple((uint32_t)kargs.m, (uint32_t)kargs.n),
            opus::make_tuple((uint64_t)stride_c, 1_I),
            opus::make_tuple((uint32_t)(tile_row + wave_id * PT::kCRowsTdm), (uint32_t)tile_col)));
    window_c.async_store();
    __builtin_amdgcn_s_wait_tensorcnt(0);
}

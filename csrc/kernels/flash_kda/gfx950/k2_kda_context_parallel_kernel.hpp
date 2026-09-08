// gfx950-private context-parallel KDA implementation.
//
// A production context group is 32, 64, or 128 consecutive C16 chunks.  A
// strict dense-N1 diagnostic may also use 16 chunks after the host proves its
// affine workspace bound.  Two embarrassingly parallel passes form the affine
// map of every group,
//
//     h_out = A_group @ h_in + b_group,                 h: [K,V]
//
// then a short per-sequence scan replaces b_group with the corresponding
// h_in.  A final parallel replay starts from that FP32 h_in, recomputes the
// real recurrence, and writes output directly.  No cs_u/cs_sin replay arena is
// needed.
//
// Outer affine-buffer layout is context-major:
//   affine_a: [total_context_groups, H, K, K] BF16
//   affine_b: [total_context_groups, H, K, V] FP32
// The scan overwrites affine_b in place with h_in.  External state retains the
// public [N,H,V,K] layout and may be BF16 or FP32.
#pragma once

#include <hip/hip_runtime.h>

#include "k2_kda_vsplit_rs_x32_kernel.hpp"
#include "packed_direct_prefixless.hpp"

namespace flashkda_hip::gfx950 {

enum class KdaContextMode : int {
    kAffineB,
    kAffineA,
    kReplay,
};

// The MFMA accumulator returned by INV@V already has the exact B-fragment
// lane mapping that a transpose read of its row-major U publication would
// reconstruct.  Strict-opt-in context kernels can therefore retain that
// rounded fragment in VGPRs and feed it directly to the following MFMAs.
__device__ __forceinline__ f32x4 context_mfma_row_major_a_reg_b(
        const __bf16* __restrict__ a,
        bf16x4 b,
        int lane) {
    const int row = lane & 15;
    const int kb = (lane >> 4) * 4;
    bf16x4 af;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        af[i] = a[row * 16 + kb + i];
    const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
    return mfma_bf16(af, b, zero);
}

__device__ __forceinline__ f32x4 context_mfma_tiled_kr_reg_b(
        const __bf16* __restrict__ kr,
        bf16x4 b,
        int ktile,
        int lane) {
    constexpr int C = 16;
    const bf16x4 af = ds_read_tr16(kr + ktile * C * C, lane);
    const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
    return mfma_bf16(af, b, zero);
}

// Default grid contract for all three modes:
//   block = NW*64 threads
//   grid.x = total_context_groups * H
//   grid.y = 8/NW
// Strict-opt-in DIRECT_TAIL_FIRST with NW=1 instead uses a sequence-major
// one-dimensional grid: grid.x = N*H*8 and grid.y = 1.  The host verifies
// that grid.x is representable by the signed block-index decode.
// Each wave owns one V16 register-state slice.  All NW waves share the
// V-independent C16 operands.  VL maps the context id through context_prefix;
// dense layout derives (batch, local_group) directly from NT.
//
// Mode contracts:
//   kAffineB: h=0, real v, no output; store FP32 b[K,V].
//   kAffineA: h=I, v=0, no output; store BF16 A[K,K].
//   kReplay : h=affine_b (after scan), real v; write output in this pass and,
//             when HO, write external final_state only from the last group.
template <
    int GROUP_CHUNKS,
    KdaContextMode MODE,
    bool HO = false,
    bool SFP32 = false,
    bool VL = false,
    bool DIRECT = false,
    int NW = 4,
    int DIRECT_MAX_CHUNKS = 0,
    bool CACHED_OPERANDS = false,
    bool U_FORWARD = false,
    bool V_FORWARD = false,
    bool LDS_PIPELINE = false,
    bool DIRECT_TAIL_FIRST = false,
    bool PACKED_DIRECT_PREFIXLESS = false,
    bool DENSE_ALL_FULL_C16 = false,
    typename RegBGemm = RegBX32,
    typename KrCarry = TiledKrCarryX16,
    bool PAIRED_STATE_PRODUCTS_X32 = false,
    bool NW1_WAVE_BARRIER = false,
    bool DENSE_N1_H12 = false>
__global__ void __launch_bounds__(NW * 64)
k2_kda_context_parallel_nw4_kernel(
        const __bf16* __restrict__ v_g,       // [T_total,H,D], null in A mode is OK
        const float* __restrict__ beta_g,     // raw [T,H] or activated [n_ht,C]
        __bf16* __restrict__ out_g,           // replay only
        const __bf16* __restrict__ ws_kd,     // [n_ht,C,D]
        const __bf16* __restrict__ ws_qd,     // replay only
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,      // log2 decay or decay, [n_ht,D]
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ ws_mqk,    // replay only
        float* __restrict__ affine_b,          // [G,H,K,V], b output / replay h_in
        __bf16* __restrict__ affine_a,         // [G,H,K,K], A output
        const void* __restrict__ init_state,   // [N,H,V,K], direct replay only
        void* __restrict__ final_state,        // [N,H,V,K], replay/HO only
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ context_prefix,
        int N,
        int total_tiles,
        int T_seq,
        int H,
        int NT) {
    static_assert(!HO || MODE == KdaContextMode::kReplay,
                  "only context replay may publish final state");
    static_assert(!DIRECT || MODE == KdaContextMode::kReplay,
                  "only replay supports the direct sequence route");
    static_assert(!DIRECT_TAIL_FIRST || DIRECT,
                  "tail-first sequence mapping is direct-only");
    static_assert(!PACKED_DIRECT_PREFIXLESS ||
                      (VL && DIRECT && DIRECT_MAX_CHUNKS == 0),
                  "prefixless K2 mapping is packed pure-direct only");
    static_assert(!(DIRECT_TAIL_FIRST && NW == 1) ||
                  (GROUP_CHUNKS == 1 &&
                   MODE == KdaContextMode::kReplay &&
                   DIRECT_MAX_CHUNKS == 0 && CACHED_OPERANDS &&
                   U_FORWARD && V_FORWARD && !LDS_PIPELINE),
                  "NW1 flat tail-first requires pure-direct cached U/V P0");
    static_assert(!LDS_PIPELINE || (U_FORWARD && V_FORWARD),
                  "context LDS pipeline requires U/V register forwarding");
    static_assert(!PAIRED_STATE_PRODUCTS_X32 ||
                      (MODE == KdaContextMode::kReplay && DIRECT &&
                       CACHED_OPERANDS && __is_same(RegBGemm, RegBX32)),
                  "paired x32 state products require cached direct replay "
                  "with RegBX32");
    static_assert(!NW1_WAVE_BARRIER ||
                      (NW == 1 && MODE == KdaContextMode::kReplay &&
                       DIRECT && !LDS_PIPELINE),
                  "NW1 wave barriers require direct replay with the P0 "
                  "single LDS arena");
    static_assert(!DENSE_N1_H12 ||
                      (GROUP_CHUNKS == 1 &&
                       MODE == KdaContextMode::kReplay && !VL && DIRECT &&
                       NW == 1 && DIRECT_MAX_CHUNKS == 0 &&
                       CACHED_OPERANDS && U_FORWARD && V_FORWARD &&
                       !LDS_PIPELINE && DIRECT_TAIL_FIRST &&
                       !PACKED_DIRECT_PREFIXLESS && DENSE_ALL_FULL_C16 &&
                       __is_same(RegBGemm, RegBX32) &&
                       __is_same(KrCarry, TiledKrCarryX16) &&
                       !PAIRED_STATE_PRODUCTS_X32 && !NW1_WAVE_BARRIER),
                  "dense N=1 H=12 replay requires the pure-direct NW1-flat "
                  "cached U/V P0 full-C16 RegBX32/TiledKr recipe");

    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int SD = D + 4;
    constexpr int NKB = D / C;
    constexpr int NTHREADS = NW * 64;
    constexpr int ROW_VECS = (C * D) / 8;
    constexpr bool STRICT_NW1_REPLAY =
        MODE == KdaContextMode::kReplay && DIRECT && DIRECT_TAIL_FIRST &&
        NW == 1 && CACHED_OPERANDS && U_FORWARD && V_FORWARD &&
        !LDS_PIPELINE;
    static_assert(!DENSE_ALL_FULL_C16 ||
                      (STRICT_NW1_REPLAY && !VL &&
                       DIRECT_MAX_CHUNKS == 0),
                  "all-full C16 replay is dense pure-direct NW1 only");
    // NW1/2/4 assign an equal number of common bf16x8 vectors to every
    // thread.  NW8 has twice as many threads as vectors: only the low four
    // waves stage one vector each, while all eight waves retain their own
    // V16 recurrence fragment.
    constexpr int RW = NW <= 4 ? ROW_VECS / NTHREADS : 1;
    constexpr int VR = (C * BV) / 64;
    constexpr int LDS_ARENAS = LDS_PIPELINE ? 2 : 1;
    constexpr int VMAT_ELEMENTS = V_FORWARD ? 1 : NW * C * BV;
    constexpr int UMAT_ELEMENTS = U_FORWARD ? 1 : NW * C * BV;
    static_assert((NW == 1 || NW == 2 || NW == 4 || NW == 8) &&
                  (NW == 8 || ROW_VECS % NTHREADS == 0) && VR == 4,
                  "context kernel requires one, two, four, or eight waves");
    static_assert(NW != 8 || ROW_VECS * 2 == NTHREADS,
                  "NW8 requires exactly one common vector per low-half thread");
    static_assert(NW != 8 ||
                  (CACHED_OPERANDS && U_FORWARD && V_FORWARD),
                  "NW8 is restricted to cached operands with U/V forwarding");

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    int global_context;
    int h;
    int v_group;
    if constexpr (DENSE_N1_H12) {
        // The host contract fixes a dense N=1,H=12 launch to exactly 96
        // sequence-major CTAs.  Constant decoding removes the general H
        // divide/remainder without changing the established flat ordering.
        const int flat = int(blockIdx.x);
        global_context = 0;
        h = flat >> 3;
        v_group = flat & 7;
    } else if constexpr (DIRECT && DIRECT_TAIL_FIRST && NW == 1) {
        // Keeping V16 as the innermost power-of-two axis places all eight
        // slices of one (sequence, head) contiguously in grid.x.
        const int flat = int(blockIdx.x);
        const int sequence_head = flat >> 3;
        v_group = flat & 7;
        global_context = sequence_head / H;
        h = sequence_head - global_context * H;
    } else {
        global_context = int(blockIdx.x) / H;
        h = int(blockIdx.x) - global_context * H;
        v_group = int(blockIdx.y);
    }
    const int v0 = (v_group * NW + wave) * BV;

    int seq;
    int local_group;
    int seq_len;
    int seq_chunks;
    int ht_sequence_base;
    int t_sequence_base;
    if constexpr (DENSE_N1_H12) {
        // N=1 and total_tiles=NT are host-proven.  Keep T_seq/NT runtime so
        // the same symbol covers the two admitted 256/512-token buckets.
        seq = 0;
        local_group = 0;
        seq_len = T_seq;
        seq_chunks = NT;
        ht_sequence_base = h * NT;
        t_sequence_base = 0;
    } else if constexpr (VL) {
        if constexpr (DIRECT) {
            if (N <= 0 || global_context >= N)
                return;
            if constexpr (DIRECT_TAIL_FIRST)
                seq = global_context == 0 ? N - 1 : global_context - 1;
            else
                seq = global_context;
            local_group = 0;
        } else {
            if (N <= 0 || global_context >= context_prefix[N])
                return;
            int lo = 0;
            int hi = N;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (context_prefix[mid] <= global_context)
                    lo = mid;
                else
                    hi = mid;
            }
            seq = lo;
            local_group = global_context - context_prefix[seq];
        }
        if constexpr (PACKED_DIRECT_PREFIXLESS) {
            const PackedC16SequenceMapping mapping =
                packed_c16_sequence_mapping(cu_seqlens, seq);
            seq_len = mapping.token_length;
            seq_chunks = (seq_len + C - 1) / C;
            ht_sequence_base = h * total_tiles + mapping.tile_base;
            t_sequence_base = mapping.token_base;
        } else {
            const int64_t bos = cu_seqlens[seq];
            seq_len = int(cu_seqlens[seq + 1] - bos);
            seq_chunks = (seq_len + C - 1) / C;
            ht_sequence_base = h * total_tiles + tile_prefix[seq];
            t_sequence_base = int(bos);
        }
    } else {
        if (N <= 0 || NT <= 0)
            return;
        const int groups_per_sequence = DIRECT
            ? 1 : (NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
        if (global_context >= N * groups_per_sequence)
            return;
        const int launch_seq = global_context / groups_per_sequence;
        if constexpr (DIRECT_TAIL_FIRST)
            seq = launch_seq == 0 ? N - 1 : launch_seq - 1;
        else
            seq = launch_seq;
        local_group = global_context - launch_seq * groups_per_sequence;
        seq_len = T_seq;
        seq_chunks = NT;
        ht_sequence_base = (seq * H + h) * NT;
        t_sequence_base = seq * T_seq;
    }

    if constexpr (DIRECT && DIRECT_MAX_CHUNKS > 0) {
        // Hybrid packed batches launch this direct pass only for short
        // sequences.  Longer sequences are covered by the filtered affine
        // prefix and must not be advanced twice.
        if (seq_chunks > DIRECT_MAX_CHUNKS)
            return;
    }

    const int first_chunk = DIRECT ? 0 : local_group * GROUP_CHUNKS;
    const int group_chunks = DIRECT
        ? seq_chunks : min(GROUP_CHUNKS, seq_chunks - first_chunk);
    if constexpr (!DENSE_N1_H12) {
        if (group_chunks <= 0) {
            // Packed serving metadata may legally contain an empty sequence.
            // A direct CTA still owns a unique (sequence, head, V16) state
            // slab, so preserve the public recurrence contract without
            // launching a separate copy kernel on every non-empty call.
            if constexpr (DIRECT && MODE == KdaContextMode::kReplay && HO) {
                const int64_t state_slab =
                    (int64_t(seq) * H + h) * D * D;
                #pragma unroll
                for (int ktile = 0; ktile < NKB; ++ktile) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int vv = v0 + (lane & 15);
                        const int kk = ktile * C + (lane >> 4) * 4 + i;
                        const int64_t idx =
                            state_slab + int64_t(vv) * D + kk;
                        float value = 0.0f;
                        if (init_state != nullptr) {
                            value = SFP32
                                ? reinterpret_cast<const float*>(
                                    init_state)[idx]
                                : bf16_to_f32(
                                    reinterpret_cast<const __bf16*>(
                                        init_state)[idx]);
                        }
                        if constexpr (SFP32) {
                            reinterpret_cast<float*>(final_state)[idx] =
                                value;
                        } else {
                            reinterpret_cast<__bf16*>(final_state)[idx] =
                                f32_to_bf16(value);
                        }
                    }
                }
            }
            return;
        }
    }
    const bool is_last_group = DENSE_N1_H12 ||
        first_chunk + group_chunks == seq_chunks;
    const int ht_base = ht_sequence_base + first_chunk;
    const int t0_base = t_sequence_base + first_chunk * C;

    // V-independent operands are loaded once by the CTA.  kr uses the tiled
    // [K16][C,K16] publication expected by TiledKrCarryX16.  The strict-opt-in
    // pipeline gives every chunk-dependent operand an inactive arena: after a
    // wave finishes current compute it may publish the prefetched next chunk
    // without racing a slower wave that still reads the active arena.
    __shared__ __bf16 kd[LDS_ARENAS * C * SD];
    __shared__ __bf16 qd[LDS_ARENAS * C * SD];
    __shared__ __bf16 kr[LDS_ARENAS * C * D];
    // The one-element forward specialization has no usable vmat arena.  As
    // with U forwarding below, all accesses are discarded at compile time.
    __shared__ __bf16 vmat[VMAT_ELEMENTS];
    // The one-element forward specialization has no usable umat arena.  Every
    // access is in a discarded !U_FORWARD branch, allowing the compiler to
    // remove this declaration entirely instead of reserving NW*512 bytes.
    __shared__ __bf16 umat[UMAT_ELEMENTS];
    __shared__ __bf16 inv[LDS_ARENAS * C * C];
    __shared__ __bf16 mqk[LDS_ARENAS * C * C];
    __shared__ float decay[LDS_ARENAS * D];
    __shared__ float beta[LDS_ARENAS * C];

    // sreg is the established register-B mapping: lane&15 selects V, while
    // (lane>>4)*4+i selects K within each K16 tile.  Logically this is h[K,V].
    float sreg[NKB][4];
    const int64_t context_slab =
        (int64_t(global_context) * H + h) * D * D;
    #pragma unroll
    for (int kt = 0; kt < NKB; ++kt) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = kt * C + (lane >> 4) * 4 + i;
            if constexpr (MODE == KdaContextMode::kAffineA) {
                sreg[kt][i] = kk == vv ? 1.0f : 0.0f;
            } else if constexpr (MODE == KdaContextMode::kReplay) {
                if constexpr (DIRECT) {
                    if (init_state != nullptr) {
                        const int64_t state_slab = DENSE_N1_H12
                            ? int64_t(h) * D * D
                            : (int64_t(seq) * H + h) * D * D;
                        const int64_t idx =
                            state_slab + int64_t(vv) * D + kk;
                        if constexpr (SFP32) {
                            sreg[kt][i] =
                                reinterpret_cast<const float*>(init_state)[idx];
                        } else {
                            sreg[kt][i] = bf16_to_f32(
                                reinterpret_cast<const __bf16*>(init_state)[idx]);
                        }
                    } else {
                        sreg[kt][i] = 0.0f;
                    }
                } else {
                    sreg[kt][i] = affine_b[
                        context_slab + int64_t(kk) * D + vv];
                }
            } else {
                sreg[kt][i] = 0.0f;
            }
        }
    }

    // One C16 chunk is prefetched in registers while the current chunk runs.
    bf16x8 kd_r[RW];
    bf16x8 qd_r[RW];
    bf16x8 kr_r[RW];
    bf16x8 inv_r;
    bf16x8 mqk_r;
    f32x4 gt_r;
    __bf16 v_r[VR];
    // `stage(next)` intentionally overlaps the next chunk's global loads with
    // the current recurrence, so it overwrites v_r before current compute.
    // Commit the current chunk into a separate MFMA B-fragment first.  The
    // fragment then remains live while v_r is reused as the prefetch buffer.
    bf16x4 v_fragment;
    float beta_r;

    auto stage = [&](int ht, int t0, int alen) {
        if constexpr (NW <= 4) {
            #pragma unroll
            for (int j = 0; j < RW; ++j) {
                const int vi = tid + j * NTHREADS;
                kd_r[j] = reinterpret_cast<const bf16x8*>(
                    ws_kd + int64_t(ht) * C * D)[vi];
                kr_r[j] = reinterpret_cast<const bf16x8*>(
                    ws_kr + int64_t(ht) * C * D)[vi];
                if constexpr (MODE == KdaContextMode::kReplay) {
                    qd_r[j] = reinterpret_cast<const bf16x8*>(
                        ws_qd + int64_t(ht) * C * D)[vi];
                }
            }
        } else {
            if (tid < ROW_VECS) {
                kd_r[0] = reinterpret_cast<const bf16x8*>(
                    ws_kd + int64_t(ht) * C * D)[tid];
                kr_r[0] = reinterpret_cast<const bf16x8*>(
                    ws_kr + int64_t(ht) * C * D)[tid];
                if constexpr (MODE == KdaContextMode::kReplay) {
                    qd_r[0] = reinterpret_cast<const bf16x8*>(
                        ws_qd + int64_t(ht) * C * D)[tid];
                }
            }
        }
        if (tid < (C * C) / 8) {
            inv_r = reinterpret_cast<const bf16x8*>(
                ws_inv + int64_t(ht) * C * C)[tid];
            if constexpr (MODE == KdaContextMode::kReplay) {
                mqk_r = reinterpret_cast<const bf16x8*>(
                    ws_mqk + int64_t(ht) * C * C)[tid];
            }
        }
        if (tid < D / 4) {
            gt_r = reinterpret_cast<const f32x4*>(
                ws_gt + int64_t(ht) * D)[tid];
        }
        if constexpr (STRICT_NW1_REPLAY) {
            // Almost every K3 prefill chunk is a complete C16.  Hoist the
            // uniform tail decision out of the four lane-varying loads and
            // strength-reduce their token-stride address arithmetic.  The
            // partial tail retains the established predicated loads verbatim.
            const int m0 = (lane >> 4) * 4;
            const int vv = lane & 15;
            if constexpr (DENSE_ALL_FULL_C16) {
                if constexpr (DENSE_N1_H12) {
                    constexpr int64_t token_stride = 12 * D;
                    const int64_t base =
                        int64_t(t0 + m0) * token_stride +
                        int64_t(h) * D + v0 + vv;
                    #pragma unroll
                    for (int j = 0; j < VR; ++j)
                        v_r[j] =
                            v_g[base + int64_t(j) * token_stride];
                } else {
                    const int64_t token_stride = int64_t(H) * D;
                    const int64_t base =
                        (int64_t(t0 + m0) * H + h) * D + v0 + vv;
                    #pragma unroll
                    for (int j = 0; j < VR; ++j)
                        v_r[j] =
                            v_g[base + int64_t(j) * token_stride];
                }
            } else if (alen == C) {
                const int64_t token_stride = int64_t(H) * D;
                const int64_t base =
                    (int64_t(t0 + m0) * H + h) * D + v0 + vv;
                #pragma unroll
                for (int j = 0; j < VR; ++j)
                    v_r[j] = v_g[base + int64_t(j) * token_stride];
            } else {
                #pragma unroll
                for (int j = 0; j < VR; ++j) {
                    const int m = m0 + j;
                    v_r[j] = m < alen
                        ? v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv]
                        : (__bf16)0.0f;
                }
            }
        } else {
            #pragma unroll
            for (int j = 0; j < VR; ++j) {
                int m;
                int vv;
                if constexpr (V_FORWARD) {
                    // Directly stage the fragment reconstructed by the legacy
                    // vmat transpose read: B[(lane>>4)*4+j][lane&15].  The four
                    // 16-lane spans remain contiguous global accesses.
                    m = (lane >> 4) * 4 + j;
                    vv = lane & 15;
                } else {
                    const int idx = lane + j * 64;
                    m = idx / BV;
                    vv = idx - m * BV;
                }
                if constexpr (MODE == KdaContextMode::kAffineA) {
                    v_r[j] = (__bf16)0.0f;
                } else {
                    v_r[j] = m < alen
                        ? v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv]
                        : (__bf16)0.0f;
                }
            }
        }
        if (tid < C) {
            if constexpr (CACHED_OPERANDS) {
                beta_r = beta_g[int64_t(ht) * C + tid];
            } else {
                beta_r = tid < alen
                    ? sigmoid_tanh(beta_g[int64_t(t0 + tid) * H + h])
                    : 0.0f;
            }
        }
    };

    // Keep the established single-arena publication as a separate lambda so
    // every P0 specialization retains its original source and code generation.
    auto commit = [&]() {
        if constexpr (NW <= 4) {
            #pragma unroll
            for (int j = 0; j < RW; ++j) {
                const int vi = tid + j * NTHREADS;
                const int row = vi >> 4;
                const int col8 = vi & 15;
                reinterpret_cast<bf16x8*>(kd + row * SD)[col8] = kd_r[j];
                if constexpr (MODE == KdaContextMode::kReplay) {
                    reinterpret_cast<bf16x8*>(qd + row * SD)[col8] = qd_r[j];
                }

                // Convert the row-major CxD workspace vector to contiguous K16
                // tiles used by the transpose-read carry.
                const int source_element = vi * 8;
                const int c = source_element / D;
                const int k = source_element - c * D;
                const int kt = k / C;
                const int ki = k - kt * C;
                __bf16* kr_dst = kr + kt * C * C + c * C + ki;
                *reinterpret_cast<bf16x8*>(kr_dst) = kr_r[j];
            }
        } else {
            if (tid < ROW_VECS) {
                const int row = tid >> 4;
                const int col8 = tid & 15;
                reinterpret_cast<bf16x8*>(kd + row * SD)[col8] = kd_r[0];
                if constexpr (MODE == KdaContextMode::kReplay) {
                    reinterpret_cast<bf16x8*>(qd + row * SD)[col8] = qd_r[0];
                }

                const int source_element = tid * 8;
                const int c = source_element / D;
                const int k = source_element - c * D;
                const int kt = k / C;
                const int ki = k - kt * C;
                __bf16* kr_dst = kr + kt * C * C + c * C + ki;
                *reinterpret_cast<bf16x8*>(kr_dst) = kr_r[0];
            }
        }

        if (tid < (C * C) / 8) {
            reinterpret_cast<bf16x8*>(inv)[tid] = inv_r;
            if constexpr (MODE == KdaContextMode::kReplay)
                reinterpret_cast<bf16x8*>(mqk)[tid] = mqk_r;
        }
        if (tid < D / 4) {
            f32x4 d = gt_r;
            if constexpr (!CACHED_OPERANDS) {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    d[i] = ex2(d[i]);
            }
            reinterpret_cast<f32x4*>(decay)[tid] = d;
        }
        #pragma unroll
        for (int j = 0; j < VR; ++j) {
            if constexpr (V_FORWARD)
                v_fragment[j] = v_r[j];
            else
                vmat[wave * C * BV + lane + j * 64] = v_r[j];
        }
        if (tid < C)
            beta[tid] = beta_r;
    };

    auto commit_pipeline = [&](int arena) {
        __bf16* const kd_dst = kd + arena * C * SD;
        __bf16* const qd_dst = qd + arena * C * SD;
        __bf16* const kr_dst_base = kr + arena * C * D;
        __bf16* const inv_dst = inv + arena * C * C;
        __bf16* const mqk_dst = mqk + arena * C * C;
        float* const decay_dst = decay + arena * D;
        float* const beta_dst = beta + arena * C;
        if constexpr (NW <= 4) {
            #pragma unroll
            for (int j = 0; j < RW; ++j) {
                const int vi = tid + j * NTHREADS;
                const int row = vi >> 4;
                const int col8 = vi & 15;
                reinterpret_cast<bf16x8*>(kd_dst + row * SD)[col8] = kd_r[j];
                if constexpr (MODE == KdaContextMode::kReplay) {
                    reinterpret_cast<bf16x8*>(qd_dst + row * SD)[col8] = qd_r[j];
                }

                // Convert the row-major CxD workspace vector to contiguous K16
                // tiles used by the transpose-read carry.
                const int source_element = vi * 8;
                const int c = source_element / D;
                const int k = source_element - c * D;
                const int kt = k / C;
                const int ki = k - kt * C;
                __bf16* kr_dst =
                    kr_dst_base + kt * C * C + c * C + ki;
                *reinterpret_cast<bf16x8*>(kr_dst) = kr_r[j];
            }
        } else {
            if (tid < ROW_VECS) {
                const int row = tid >> 4;
                const int col8 = tid & 15;
                reinterpret_cast<bf16x8*>(kd_dst + row * SD)[col8] = kd_r[0];
                if constexpr (MODE == KdaContextMode::kReplay) {
                    reinterpret_cast<bf16x8*>(qd_dst + row * SD)[col8] = qd_r[0];
                }

                const int source_element = tid * 8;
                const int c = source_element / D;
                const int k = source_element - c * D;
                const int kt = k / C;
                const int ki = k - kt * C;
                __bf16* kr_dst =
                    kr_dst_base + kt * C * C + c * C + ki;
                *reinterpret_cast<bf16x8*>(kr_dst) = kr_r[0];
            }
        }

        if (tid < (C * C) / 8) {
            reinterpret_cast<bf16x8*>(inv_dst)[tid] = inv_r;
            if constexpr (MODE == KdaContextMode::kReplay)
                reinterpret_cast<bf16x8*>(mqk_dst)[tid] = mqk_r;
        }
        if (tid < D / 4) {
            f32x4 d = gt_r;
            if constexpr (!CACHED_OPERANDS) {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    d[i] = ex2(d[i]);
            }
            reinterpret_cast<f32x4*>(decay_dst)[tid] = d;
        }
        #pragma unroll
        for (int j = 0; j < VR; ++j) {
            if constexpr (V_FORWARD)
                v_fragment[j] = v_r[j];
            else
                vmat[wave * C * BV + lane + j * 64] = v_r[j];
        }
        if (tid < C)
            beta_dst[tid] = beta_r;
    };

    if constexpr (LDS_PIPELINE) {
        int t0_cur = t0_base;
        int alen_cur = min(C, seq_len - first_chunk * C);
        stage(ht_base, t0_cur, alen_cur);
        commit_pipeline(0);
        __syncthreads();

        for (int chunk = 0; chunk < group_chunks; ++chunk) {
            const int t0 = t0_cur;
            const int alen = alen_cur;
            const bool has_next = chunk + 1 < group_chunks;
            const int current_arena = chunk & 1;
            const __bf16* const kd_current = kd + current_arena * C * SD;
            const __bf16* const qd_current = qd + current_arena * C * SD;
            const __bf16* const kr_current = kr + current_arena * C * D;
            const __bf16* const inv_current = inv + current_arena * C * C;
            const __bf16* const mqk_current = mqk + current_arena * C * C;
            const float* const decay_current = decay + current_arena * D;
            const float* const beta_current = beta + current_arena * C;
            if (has_next) {
                const int next_global_chunk = first_chunk + chunk + 1;
                const int ht_next = ht_base + chunk + 1;
                const int t0_next = t_sequence_base + next_global_chunk * C;
                const int alen_next = min(C, seq_len - next_global_chunk * C);
                stage(ht_next, t0_next, alen_next);
                t0_cur = t0_next;
                alen_cur = alen_next;
            }

            f32x4 residual;
            RegBPairX32 paired_state_products;
            if constexpr (PAIRED_STATE_PRODUCTS_X32) {
                paired_state_products = gemm_regb_even_x32_pair<SD, NKB>(
                    kd_current, qd_current, sreg, lane);
                residual = paired_state_products.first;
            } else {
                residual = RegBGemm::template run<SD, NKB>(
                    kd_current, sreg, lane);
            }
            bf16x4 vnew_bf;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const int vv = lane & 15;
                float source;
                if constexpr (V_FORWARD)
                    source = bf16_to_f32(v_fragment[i]);
                else
                    source = bf16_to_f32(
                        vmat[wave * C * BV + m * BV + vv]);
                const float value =
                    (source - residual[i]) * beta_current[m];
                if constexpr (V_FORWARD)
                    vnew_bf[i] = f32_to_bf16(value);
                else
                    vmat[wave * C * BV + m * BV + vv] = f32_to_bf16(value);
            }

            f32x4 u;
            if constexpr (V_FORWARD) {
                // MFMA output and B fragments use the same lane mapping.  This is
                // the exact BF16 fragment the legacy vmat transpose read returns.
                u = context_mfma_row_major_a_reg_b(
                    inv_current, vnew_bf, lane);
            } else {
                __syncwarp();
                u = mm_std_16_tr(
                    inv_current, vmat + wave * C * BV, lane);
            }
            bf16x4 u_bf;
            if constexpr (U_FORWARD) {
                // This is the same single FP32->BF16 rounding performed by the
                // established umat publication.  Its lane-local fragment is
                // exactly what every subsequent transpose read would return.
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    u_bf[i] = f32_to_bf16(u[i]);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int m = (lane >> 4) * 4 + i;
                    const int vv = lane & 15;
                    umat[wave * C * BV + m * BV + vv] = f32_to_bf16(u[i]);
                }
                __syncwarp();
            }

            if constexpr (MODE == KdaContextMode::kReplay) {
                f32x4 from_state;
                if constexpr (PAIRED_STATE_PRODUCTS_X32)
                    from_state = paired_state_products.second;
                else
                    from_state = RegBGemm::template run<SD, NKB>(
                        qd_current, sreg, lane);
                f32x4 from_local;
                if constexpr (U_FORWARD)
                    from_local = context_mfma_row_major_a_reg_b(
                        mqk_current, u_bf, lane);
                else
                    from_local = mm_std_16_tr(
                        mqk_current, umat + wave * C * BV, lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int m = (lane >> 4) * 4 + i;
                    const int vv = lane & 15;
                    if (m < alen) {
                        const __bf16 a = f32_to_bf16(from_state[i]);
                        const __bf16 b = f32_to_bf16(from_local[i]);
                        out_g[(int64_t(t0 + m) * H + h) * D + v0 + vv] =
                            f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
                    }
                }
            }

            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                f32x4 carry;
                if constexpr (U_FORWARD)
                    carry = context_mfma_tiled_kr_reg_b(
                        kr_current, u_bf, ktile, lane);
                else
                    carry = KrCarry::template run<C, D, BV>(
                        kr_current, umat + wave * C * BV, ktile, 0, lane);
                const int kbase = ktile * C + (lane >> 4) * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sreg[ktile][i] =
                        sreg[ktile][i] * decay_current[kbase + i] + carry[i];
                }
            }
            if (has_next) {
                // The inactive arena is no longer read.  This one barrier both
                // completes every current-arena read and publishes every next
                // operand; the final chunk needs neither commit nor barrier.
                commit_pipeline(current_arena ^ 1);
                __syncthreads();
            }
        }
    } else {
        int t0_cur = t0_base;
        int alen_cur;
        if constexpr (DENSE_ALL_FULL_C16)
            alen_cur = C;
        else
            alen_cur = min(C, seq_len - first_chunk * C);
        stage(ht_base, t0_cur, alen_cur);
        commit();
        if constexpr (NW1_WAVE_BARRIER)
            __syncwarp();
        else
            __syncthreads();

        for (int chunk = 0; chunk < group_chunks; ++chunk) {
            const int t0 = t0_cur;
            const int alen = alen_cur;
            const bool has_next = chunk + 1 < group_chunks;
            if (has_next) {
                const int next_global_chunk = first_chunk + chunk + 1;
                const int ht_next = ht_base + chunk + 1;
                const int t0_next =
                    t_sequence_base + next_global_chunk * C;
                int alen_next;
                if constexpr (DENSE_ALL_FULL_C16)
                    alen_next = C;
                else
                    alen_next = min(C, seq_len - next_global_chunk * C);
                stage(ht_next, t0_next, alen_next);
                t0_cur = t0_next;
                alen_cur = alen_next;
            }

            f32x4 residual;
            RegBPairX32 paired_state_products;
            if constexpr (PAIRED_STATE_PRODUCTS_X32) {
                paired_state_products = gemm_regb_even_x32_pair<SD, NKB>(
                    kd, qd, sreg, lane);
                residual = paired_state_products.first;
            } else {
                residual = RegBGemm::template run<SD, NKB>(
                    kd, sreg, lane);
            }
            bf16x4 vnew_bf;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const int vv = lane & 15;
                float source;
                if constexpr (V_FORWARD)
                    source = bf16_to_f32(v_fragment[i]);
                else
                    source = bf16_to_f32(
                        vmat[wave * C * BV + m * BV + vv]);
                const float value = (source - residual[i]) * beta[m];
                if constexpr (V_FORWARD)
                    vnew_bf[i] = f32_to_bf16(value);
                else
                    vmat[wave * C * BV + m * BV + vv] =
                        f32_to_bf16(value);
            }

            f32x4 u;
            if constexpr (V_FORWARD) {
                // MFMA output and B fragments use the same lane mapping.  This
                // is the exact BF16 fragment the legacy transpose read returns.
                u = context_mfma_row_major_a_reg_b(inv, vnew_bf, lane);
            } else {
                __syncwarp();
                u = mm_std_16_tr(inv, vmat + wave * C * BV, lane);
            }
            bf16x4 u_bf;
            if constexpr (U_FORWARD) {
                // This is the same single FP32->BF16 rounding performed by the
                // established umat publication.
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    u_bf[i] = f32_to_bf16(u[i]);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int m = (lane >> 4) * 4 + i;
                    const int vv = lane & 15;
                    umat[wave * C * BV + m * BV + vv] = f32_to_bf16(u[i]);
                }
                __syncwarp();
            }

            if constexpr (MODE == KdaContextMode::kReplay) {
                f32x4 from_state;
                if constexpr (PAIRED_STATE_PRODUCTS_X32)
                    from_state = paired_state_products.second;
                else
                    from_state = RegBGemm::template run<SD, NKB>(
                        qd, sreg, lane);
                f32x4 from_local;
                if constexpr (U_FORWARD)
                    from_local = context_mfma_row_major_a_reg_b(
                        mqk, u_bf, lane);
                else
                    from_local = mm_std_16_tr(
                        mqk, umat + wave * C * BV, lane);
                if constexpr (STRICT_NW1_REPLAY) {
                    const int m0 = (lane >> 4) * 4;
                    const int vv = lane & 15;
                    if constexpr (DENSE_ALL_FULL_C16) {
                        if constexpr (DENSE_N1_H12) {
                            constexpr int64_t token_stride = 12 * D;
                            const int64_t base =
                                int64_t(t0 + m0) * token_stride +
                                int64_t(h) * D + v0 + vv;
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const __bf16 a =
                                    f32_to_bf16(from_state[i]);
                                const __bf16 b =
                                    f32_to_bf16(from_local[i]);
                                out_g[base + int64_t(i) * token_stride] =
                                    f32_to_bf16(
                                        bf16_to_f32(a) + bf16_to_f32(b));
                            }
                        } else {
                            const int64_t token_stride = int64_t(H) * D;
                            const int64_t base =
                                (int64_t(t0 + m0) * H + h) * D + v0 + vv;
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const __bf16 a =
                                    f32_to_bf16(from_state[i]);
                                const __bf16 b =
                                    f32_to_bf16(from_local[i]);
                                out_g[base + int64_t(i) * token_stride] =
                                    f32_to_bf16(
                                        bf16_to_f32(a) + bf16_to_f32(b));
                            }
                        }
                    } else if (alen == C) {
                        const int64_t token_stride = int64_t(H) * D;
                        const int64_t base =
                            (int64_t(t0 + m0) * H + h) * D + v0 + vv;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const __bf16 a = f32_to_bf16(from_state[i]);
                            const __bf16 b = f32_to_bf16(from_local[i]);
                            out_g[base + int64_t(i) * token_stride] =
                                f32_to_bf16(
                                    bf16_to_f32(a) + bf16_to_f32(b));
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int m = m0 + i;
                            if (m < alen) {
                                const __bf16 a = f32_to_bf16(from_state[i]);
                                const __bf16 b = f32_to_bf16(from_local[i]);
                                out_g[(int64_t(t0 + m) * H + h) * D +
                                      v0 + vv] = f32_to_bf16(
                                    bf16_to_f32(a) + bf16_to_f32(b));
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int m = (lane >> 4) * 4 + i;
                        const int vv = lane & 15;
                        if (m < alen) {
                            const __bf16 a = f32_to_bf16(from_state[i]);
                            const __bf16 b = f32_to_bf16(from_local[i]);
                            out_g[(int64_t(t0 + m) * H + h) * D + v0 + vv] =
                                f32_to_bf16(
                                    bf16_to_f32(a) + bf16_to_f32(b));
                        }
                    }
                }
            }

            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                f32x4 carry;
                if constexpr (U_FORWARD)
                    carry = context_mfma_tiled_kr_reg_b(
                        kr, u_bf, ktile, lane);
                else
                    carry = KrCarry::template run<C, D, BV>(
                        kr, umat + wave * C * BV, ktile, 0, lane);
                const int kbase = ktile * C + (lane >> 4) * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sreg[ktile][i] =
                        sreg[ktile][i] * decay[kbase + i] + carry[i];
                }
            }
            if constexpr (NW1_WAVE_BARRIER) {
                // The next publication reuses this single LDS arena.  First
                // rendezvous after all current reads, then publish it to the
                // following iteration.  The final chunk has no LDS consumer
                // and therefore needs neither wave barrier.
                if (has_next) {
                    __syncwarp();
                    commit();
                    __syncwarp();
                }
            } else {
                __syncthreads();
                if (has_next) {
                    commit();
                    __syncthreads();
                }
            }
        }
    }

    if constexpr (MODE == KdaContextMode::kAffineB) {
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = ktile * C + (lane >> 4) * 4 + i;
                affine_b[context_slab + int64_t(kk) * D + vv] =
                    sreg[ktile][i];
            }
        }
    } else if constexpr (MODE == KdaContextMode::kAffineA) {
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = ktile * C + (lane >> 4) * 4 + i;
                affine_a[context_slab + int64_t(kk) * D + vv] =
                    f32_to_bf16(sreg[ktile][i]);
            }
        }
    } else if constexpr (HO) {
        if (is_last_group) {
            const int64_t state_slab = DENSE_N1_H12
                ? int64_t(h) * D * D
                : (int64_t(seq) * H + h) * D * D;
            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int vv = v0 + (lane & 15);
                    const int kk = ktile * C + (lane >> 4) * 4 + i;
                    const int64_t idx =
                        state_slab + int64_t(vv) * D + kk;
                    if constexpr (SFP32) {
                        reinterpret_cast<float*>(final_state)[idx] =
                            sreg[ktile][i];
                    } else {
                        reinterpret_cast<__bf16*>(final_state)[idx] =
                            f32_to_bf16(sreg[ktile][i]);
                    }
                }
            }
        }
    }
}

// Short affine scan.  The established grid is (N*H, 8/NW); strict-opt-in
// TIGHT_VL_GRID instead launches a conservative (context_upper*H, 8/NW) hybrid
// grid and maps its filtered global-context index back to the owning sequence.
// One CTA owns NW V16 columns and walks only the context groups of one sequence.
// For every group it first preserves b in registers, overwrites b with the
// current h_in, then evaluates h = A@h+b with a FP32 register carry.  A is BF16
// and the MFMA input conversion of h matches the replay recurrence.
template <
    int GROUP_CHUNKS,
    int NW = 4,
    bool HI = false,
    bool HO = false,
    bool SFP32 = false,
    bool VL = false,
    bool TIGHT_VL_GRID = false,
    typename RegBGemm = RegBX32>
__global__ void __launch_bounds__(NW * 64)
k2_kda_context_affine_scan_nw4_kernel(
        const __bf16* __restrict__ affine_a,  // [G,H,K,K]
        float* __restrict__ affine_b,         // b -> h_in, [G,H,K,V]
        const void* __restrict__ init_state,  // [N,H,V,K], HI only
        void* __restrict__ final_state,       // empty packed sequences, HO only
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ context_prefix,
        int T_seq,  // dense length; TIGHT_VL_GRID reuses this slot for N
        int H,
        int NT) {
    static_assert(NW == 1 || NW == 2 || NW == 4,
                  "context affine scan supports one, two, or four waves");
    static_assert(!TIGHT_VL_GRID || VL,
                  "a tight context scan requires the filtered VL prefix");
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int NKB = D / C;
    constexpr int NTHREADS = NW * 64;
    // A row pitch of 128 bf16 values aliases the same LDS banks for every
    // MFMA row fragment.  Match the recurrence kernels' proven pitch-132
    // layout so adjacent rows rotate through the banks instead.
    constexpr int AD = D + 4;

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int bh = int(blockIdx.x);
    const int seq_or_context = bh / H;
    const int h = bh - seq_or_context * H;
    const int v0 = (int(blockIdx.y) * NW + wave) * BV;

    int seq = seq_or_context;
    int context_base;
    int context_count;
    if constexpr (VL) {
        if constexpr (TIGHT_VL_GRID) {
            // The tight hybrid grid is indexed by filtered affine group, not
            // by sequence.  Map that global group back through the same prefix
            // used by affine/replay, then let only group zero own the serial
            // scan.  The host upper is conservative, so trailing CTAs return.
            const int global_context = seq_or_context;
            const int N = T_seq;
            if (N <= 0 || global_context >= context_prefix[N])
                return;
            int lo = 0;
            int hi = N;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (context_prefix[mid] <= global_context)
                    lo = mid;
                else
                    hi = mid;
            }
            seq = lo;
            context_base = context_prefix[seq];
            const int local_group = global_context - context_base;
            if (local_group != 0)
                return;
        } else {
            context_base = context_prefix[seq];
        }
        context_count = context_prefix[seq + 1] - context_base;
    } else {
        (void)T_seq;
        const int groups_per_sequence =
            (NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
        context_base = seq * groups_per_sequence;
        context_count = groups_per_sequence;
    }

    // Hybrid prefixes deliberately omit non-empty short sequences because the
    // preceding direct pass already completed them.  Avoid initializing a full
    // KxV register state in those no-op scan CTAs.  On the established N grid,
    // empty sequences continue so the HO path below can publish their state;
    // the tight grid omits both classes and leaves them to that direct pass.
    if constexpr (VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] != cu_seqlens[seq])
            return;
    }

    float hreg[NKB][4];
    const int64_t state_slab = (int64_t(seq) * H + h) * D * D;
    #pragma unroll
    for (int ktile = 0; ktile < NKB; ++ktile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = ktile * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_slab + int64_t(vv) * D + kk;
            if constexpr (HI) {
                if constexpr (SFP32) {
                    hreg[ktile][i] =
                        reinterpret_cast<const float*>(init_state)[idx];
                } else {
                    hreg[ktile][i] = bf16_to_f32(
                        reinterpret_cast<const __bf16*>(init_state)[idx]);
                }
            } else {
                hreg[ktile][i] = 0.0f;
            }
        }
    }

    __shared__ __bf16 amat[D * AD];
    constexpr int A_ROW_VECS = D / 8;
    constexpr int A_VECS = D * A_ROW_VECS;

    for (int local_group = 0; local_group < context_count; ++local_group) {
        const int global_context = context_base + local_group;
        const int64_t context_slab =
            (int64_t(global_context) * H + h) * D * D;

        float breg[NKB][4];
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = ktile * C + (lane >> 4) * 4 + i;
                const int64_t idx =
                    context_slab + int64_t(kk) * D + vv;
                breg[ktile][i] = affine_b[idx];
                affine_b[idx] = hreg[ktile][i];
            }
        }

        const auto* a_src = reinterpret_cast<const bf16x8*>(
            affine_a + context_slab);
        #pragma unroll
        for (int j = 0; j < A_VECS / NTHREADS; ++j) {
            const int idx = tid + j * NTHREADS;
            const int row = idx / A_ROW_VECS;
            const int col8 = idx - row * A_ROW_VECS;
            reinterpret_cast<bf16x8*>(amat + row * AD)[col8] = a_src[idx];
        }
        __syncthreads();

        float next[NKB][4];
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            const f32x4 product = RegBGemm::template run<AD, NKB>(
                amat + ktile * C * AD, hreg, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                next[ktile][i] = product[i] + breg[ktile][i];
        }

        // All four waves must finish reading A before the next group replaces
        // the shared tile.
        __syncthreads();
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                hreg[ktile][i] = next[ktile][i];
        }
    }

    // A filtered hybrid prefix also gives short sequences zero affine groups,
    // so key this copy on the real packed length rather than context_count
    // alone.  Non-empty short sequences were already completed by the direct
    // pass and must not be overwritten here.
    if constexpr (HO && VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] == cu_seqlens[seq]) {
            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int vv = v0 + (lane & 15);
                    const int kk = ktile * C + (lane >> 4) * 4 + i;
                    const int64_t idx =
                        state_slab + int64_t(vv) * D + kk;
                    if constexpr (SFP32) {
                        reinterpret_cast<float*>(final_state)[idx] =
                            hreg[ktile][i];
                    } else {
                        reinterpret_cast<__bf16*>(final_state)[idx] =
                            f32_to_bf16(hreg[ktile][i]);
                    }
                }
            }
        }
    }
}

// Strict-opt-in affine scan variant that streams each K16 slice of b only
// after the matching output product has consumed the complete register state.
// This shortens b's live range without changing the MFMA order, h publication,
// packed mapping, state handling, or the established scan specialization.
template <
    int GROUP_CHUNKS,
    int NW = 4,
    bool HI = false,
    bool HO = false,
    bool SFP32 = false,
    bool VL = false,
    bool TIGHT_VL_GRID = false,
    typename RegBGemm = RegBX32>
__global__ void __launch_bounds__(NW * 64)
k2_kda_context_affine_scan_b_stream_nw4_kernel(
        const __bf16* __restrict__ affine_a,  // [G,H,K,K]
        float* __restrict__ affine_b,         // b -> h_in, [G,H,K,V]
        const void* __restrict__ init_state,  // [N,H,V,K], HI only
        void* __restrict__ final_state,       // empty packed sequences, HO only
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ context_prefix,
        int T_seq,  // dense length; TIGHT_VL_GRID reuses this slot for N
        int H,
        int NT) {
    static_assert(NW == 1 || NW == 2 || NW == 4,
                  "context affine scan supports one, two, or four waves");
    static_assert(!TIGHT_VL_GRID || VL,
                  "a tight context scan requires the filtered VL prefix");
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int NKB = D / C;
    constexpr int NTHREADS = NW * 64;
    constexpr int AD = D + 4;

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int bh = int(blockIdx.x);
    const int seq_or_context = bh / H;
    const int h = bh - seq_or_context * H;
    const int v0 = (int(blockIdx.y) * NW + wave) * BV;

    int seq = seq_or_context;
    int context_base;
    int context_count;
    if constexpr (VL) {
        if constexpr (TIGHT_VL_GRID) {
            const int global_context = seq_or_context;
            const int N = T_seq;
            if (N <= 0 || global_context >= context_prefix[N])
                return;
            int lo = 0;
            int hi = N;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (context_prefix[mid] <= global_context)
                    lo = mid;
                else
                    hi = mid;
            }
            seq = lo;
            context_base = context_prefix[seq];
            const int local_group = global_context - context_base;
            if (local_group != 0)
                return;
        } else {
            context_base = context_prefix[seq];
        }
        context_count = context_prefix[seq + 1] - context_base;
    } else {
        (void)T_seq;
        const int groups_per_sequence =
            (NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
        context_base = seq * groups_per_sequence;
        context_count = groups_per_sequence;
    }

    if constexpr (VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] != cu_seqlens[seq])
            return;
    }

    float hreg[NKB][4];
    const int64_t state_slab = (int64_t(seq) * H + h) * D * D;
    #pragma unroll
    for (int ktile = 0; ktile < NKB; ++ktile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = ktile * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_slab + int64_t(vv) * D + kk;
            if constexpr (HI) {
                if constexpr (SFP32) {
                    hreg[ktile][i] =
                        reinterpret_cast<const float*>(init_state)[idx];
                } else {
                    hreg[ktile][i] = bf16_to_f32(
                        reinterpret_cast<const __bf16*>(init_state)[idx]);
                }
            } else {
                hreg[ktile][i] = 0.0f;
            }
        }
    }

    __shared__ __bf16 amat[D * AD];
    constexpr int A_ROW_VECS = D / 8;
    constexpr int A_VECS = D * A_ROW_VECS;

    for (int local_group = 0; local_group < context_count; ++local_group) {
        const int global_context = context_base + local_group;
        const int64_t context_slab =
            (int64_t(global_context) * H + h) * D * D;

        const auto* a_src = reinterpret_cast<const bf16x8*>(
            affine_a + context_slab);
        #pragma unroll
        for (int j = 0; j < A_VECS / NTHREADS; ++j) {
            const int idx = tid + j * NTHREADS;
            const int row = idx / A_ROW_VECS;
            const int col8 = idx - row * A_ROW_VECS;
            reinterpret_cast<bf16x8*>(amat + row * AD)[col8] = a_src[idx];
        }
        __syncthreads();

        float next[NKB][4];
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            const f32x4 product = RegBGemm::template run<AD, NKB>(
                amat + ktile * C * AD, hreg, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = ktile * C + (lane >> 4) * 4 + i;
                const int64_t idx =
                    context_slab + int64_t(kk) * D + vv;
                const float b = affine_b[idx];
                affine_b[idx] = hreg[ktile][i];
                next[ktile][i] = product[i] + b;
            }
        }

        // Every product above reads the full hreg array.  Publish next only
        // after all waves have also finished consuming the shared A tile.
        __syncthreads();
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                hreg[ktile][i] = next[ktile][i];
        }
    }

    if constexpr (HO && VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] == cu_seqlens[seq]) {
            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int vv = v0 + (lane & 15);
                    const int kk = ktile * C + (lane >> 4) * 4 + i;
                    const int64_t idx =
                        state_slab + int64_t(vv) * D + kk;
                    if constexpr (SFP32) {
                        reinterpret_cast<float*>(final_state)[idx] =
                            hreg[ktile][i];
                    } else {
                        reinterpret_cast<__bf16*>(final_state)[idx] =
                            f32_to_bf16(hreg[ktile][i]);
                    }
                }
            }
        }
    }
}

// NW2/HI=false experiment that separates the complete A@h product phase from
// every b load/store.  The register-dependent compiler fence prevents LLVM
// from hoisting a b VMEM operation across an unfinished MFMA while emitting no
// device instruction.  The second phase preserves the established operand
// order (product + b), affine_b writeback, CTA barrier, and h publication.
template <
    int GROUP_CHUNKS,
    bool HO = false,
    bool SFP32 = false,
    bool VL = false,
    bool TIGHT_VL_GRID = false,
    typename RegBGemm = RegBX32>
__global__ void __launch_bounds__(2 * 64)
k2_kda_context_affine_scan_b_stream_b_phased_nw2_kernel(
        const __bf16* __restrict__ affine_a,  // [G,H,K,K]
        float* __restrict__ affine_b,         // b -> h_in, [G,H,K,V]
        const void* __restrict__ init_state,  // unused: HI is fixed false
        void* __restrict__ final_state,       // empty packed sequences, HO only
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ context_prefix,
        int T_seq,  // dense length; TIGHT_VL_GRID reuses this slot for N
        int H,
        int NT) {
    static_assert(!TIGHT_VL_GRID || VL,
                  "a tight context scan requires the filtered VL prefix");
    constexpr int NW = 2;
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int NKB = D / C;
    constexpr int NTHREADS = NW * 64;
    constexpr int AD = D + 4;

    (void)init_state;
    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int bh = int(blockIdx.x);
    const int seq_or_context = bh / H;
    const int h = bh - seq_or_context * H;
    const int v0 = (int(blockIdx.y) * NW + wave) * BV;

    int seq = seq_or_context;
    int context_base;
    int context_count;
    if constexpr (VL) {
        if constexpr (TIGHT_VL_GRID) {
            const int global_context = seq_or_context;
            const int N = T_seq;
            if (N <= 0 || global_context >= context_prefix[N])
                return;
            int lo = 0;
            int hi = N;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (context_prefix[mid] <= global_context)
                    lo = mid;
                else
                    hi = mid;
            }
            seq = lo;
            context_base = context_prefix[seq];
            const int local_group = global_context - context_base;
            if (local_group != 0)
                return;
        } else {
            context_base = context_prefix[seq];
        }
        context_count = context_prefix[seq + 1] - context_base;
    } else {
        (void)T_seq;
        const int groups_per_sequence =
            (NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
        context_base = seq * groups_per_sequence;
        context_count = groups_per_sequence;
    }

    if constexpr (VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] != cu_seqlens[seq])
            return;
    }

    float hreg[NKB][4];
    const int64_t state_slab = (int64_t(seq) * H + h) * D * D;
    #pragma unroll
    for (int ktile = 0; ktile < NKB; ++ktile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            hreg[ktile][i] = 0.0f;
    }

    __shared__ __bf16 amat[D * AD];
    constexpr int A_ROW_VECS = D / 8;
    constexpr int A_VECS = D * A_ROW_VECS;

    for (int local_group = 0; local_group < context_count; ++local_group) {
        const int global_context = context_base + local_group;
        const int64_t context_slab =
            (int64_t(global_context) * H + h) * D * D;

        const auto* a_src = reinterpret_cast<const bf16x8*>(
            affine_a + context_slab);
        #pragma unroll
        for (int j = 0; j < A_VECS / NTHREADS; ++j) {
            const int idx = tid + j * NTHREADS;
            const int row = idx / A_ROW_VECS;
            const int col8 = idx - row * A_ROW_VECS;
            reinterpret_cast<bf16x8*>(amat + row * AD)[col8] = a_src[idx];
        }
        __syncthreads();

        f32x4 next[NKB];
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            next[ktile] = RegBGemm::template run<AD, NKB>(
                amat + ktile * C * AD, hreg, lane);
        }

        // Each operand ties the fence to its complete MFMA result.  The
        // memory clobber then keeps every following affine_b VMEM operation
        // on the far side of all eight products without adding an ISA op.
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile)
            asm volatile("" : "+v"(next[ktile]) :: "memory");

        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = ktile * C + (lane >> 4) * 4 + i;
                const int64_t idx =
                    context_slab + int64_t(kk) * D + vv;
                const float b = affine_b[idx];
                affine_b[idx] = hreg[ktile][i];
                next[ktile][i] = next[ktile][i] + b;
            }
        }

        // Preserve the established CTA rendezvous and publish only after all
        // waves have finished consuming both the current A tile and hreg.
        __syncthreads();
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                hreg[ktile][i] = next[ktile][i];
        }
    }

    if constexpr (HO && VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] == cu_seqlens[seq]) {
            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int vv = v0 + (lane & 15);
                    const int kk = ktile * C + (lane >> 4) * 4 + i;
                    const int64_t idx =
                        state_slab + int64_t(vv) * D + kk;
                    if constexpr (SFP32) {
                        reinterpret_cast<float*>(final_state)[idx] =
                            hreg[ktile][i];
                    } else {
                        reinterpret_cast<__bf16*>(final_state)[idx] =
                            f32_to_bf16(hreg[ktile][i]);
                    }
                }
            }
        }
    }
}

// NW2-only A-publication experiment layered on the streamed-b affine scan.
// Each wave owns alternating A rows and issues one dword global-to-LDS load
// per lane.  All requests are fenced before the CTA publishes the complete
// padded tile; the recurrence, MFMA order, and streamed-b lifetime below are
// otherwise identical to k2_kda_context_affine_scan_b_stream_nw4_kernel.
template <
    int GROUP_CHUNKS,
    bool HI = false,
    bool HO = false,
    bool SFP32 = false,
    bool VL = false,
    bool TIGHT_VL_GRID = false,
    typename RegBGemm = RegBX32>
__global__ void __launch_bounds__(2 * 64)
k2_kda_context_affine_scan_b_stream_a_gll_nw2_kernel(
        const __bf16* __restrict__ affine_a,  // [G,H,K,K]
        float* __restrict__ affine_b,         // b -> h_in, [G,H,K,V]
        const void* __restrict__ init_state,  // [N,H,V,K], HI only
        void* __restrict__ final_state,       // empty packed sequences, HO only
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ context_prefix,
        int T_seq,  // dense length; TIGHT_VL_GRID reuses this slot for N
        int H,
        int NT) {
    static_assert(!TIGHT_VL_GRID || VL,
                  "a tight context scan requires the filtered VL prefix");
    constexpr int NW = 2;
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int NKB = D / C;
    constexpr int AD = D + 4;

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int bh = int(blockIdx.x);
    const int seq_or_context = bh / H;
    const int h = bh - seq_or_context * H;
    const int v0 = (int(blockIdx.y) * NW + wave) * BV;

    int seq = seq_or_context;
    int context_base;
    int context_count;
    if constexpr (VL) {
        if constexpr (TIGHT_VL_GRID) {
            const int global_context = seq_or_context;
            const int N = T_seq;
            if (N <= 0 || global_context >= context_prefix[N])
                return;
            int lo = 0;
            int hi = N;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (context_prefix[mid] <= global_context)
                    lo = mid;
                else
                    hi = mid;
            }
            seq = lo;
            context_base = context_prefix[seq];
            const int local_group = global_context - context_base;
            if (local_group != 0)
                return;
        } else {
            context_base = context_prefix[seq];
        }
        context_count = context_prefix[seq + 1] - context_base;
    } else {
        (void)T_seq;
        const int groups_per_sequence =
            (NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
        context_base = seq * groups_per_sequence;
        context_count = groups_per_sequence;
    }

    if constexpr (VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] != cu_seqlens[seq])
            return;
    }

    float hreg[NKB][4];
    const int64_t state_slab = (int64_t(seq) * H + h) * D * D;
    #pragma unroll
    for (int ktile = 0; ktile < NKB; ++ktile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = ktile * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_slab + int64_t(vv) * D + kk;
            if constexpr (HI) {
                if constexpr (SFP32) {
                    hreg[ktile][i] =
                        reinterpret_cast<const float*>(init_state)[idx];
                } else {
                    hreg[ktile][i] = bf16_to_f32(
                        reinterpret_cast<const __bf16*>(init_state)[idx]);
                }
            } else {
                hreg[ktile][i] = 0.0f;
            }
        }
    }

    __shared__ __bf16 amat[D * AD];

    for (int local_group = 0; local_group < context_count; ++local_group) {
        const int global_context = context_base + local_group;
        const int64_t context_slab =
            (int64_t(global_context) * H + h) * D * D;

        #pragma unroll
        for (int j = 0; j < D / NW; ++j) {
            const int row = j * NW + wave;
            global_to_lds_async<4>(
                amat + row * AD,
                affine_a + context_slab + int64_t(row) * D + lane * 2);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();

        float next[NKB][4];
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            const f32x4 product = RegBGemm::template run<AD, NKB>(
                amat + ktile * C * AD, hreg, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = ktile * C + (lane >> 4) * 4 + i;
                const int64_t idx =
                    context_slab + int64_t(kk) * D + vv;
                const float b = affine_b[idx];
                affine_b[idx] = hreg[ktile][i];
                next[ktile][i] = product[i] + b;
            }
        }

        // Every product above reads the full hreg array.  Publish next only
        // after all waves have also finished consuming the shared A tile.
        __syncthreads();
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                hreg[ktile][i] = next[ktile][i];
        }
    }

    if constexpr (HO && VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] == cu_seqlens[seq]) {
            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int vv = v0 + (lane & 15);
                    const int kk = ktile * C + (lane >> 4) * 4 + i;
                    const int64_t idx =
                        state_slab + int64_t(vv) * D + kk;
                    if constexpr (SFP32) {
                        reinterpret_cast<float*>(final_state)[idx] =
                            hreg[ktile][i];
                    } else {
                        reinterpret_cast<__bf16*>(final_state)[idx] =
                            f32_to_bf16(hreg[ktile][i]);
                    }
                }
            }
        }
    }
}

}  // namespace flashkda_hip::gfx950

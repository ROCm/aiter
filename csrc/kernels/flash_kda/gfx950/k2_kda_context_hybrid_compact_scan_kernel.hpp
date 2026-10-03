// gfx950-private packed-hybrid compact affine-scan experiment.
//
// The production hybrid context route filters sequences with at most 64 C16
// chunks out of context_prefix, but its sequence-indexed scan grid still has to
// launch a conservative host-visible upper bound.  In an all-short batch every
// one of those CTAs returns without doing useful work.  This standalone header
// adds the two device-side pieces needed by a strict opt-in comparison without
// changing the established prefix or scan symbols:
//
//   1. one same-stream prefix builder that also publishes a compact, stable,
//      longest-first list of sequences with more than 64 chunks; and
//   2. one packed G64/NW2 P0 scan whose fixed-size 1-D grid walks that compact
//      list deterministically in grid-stride order.
//
// There is deliberately no global or workspace atomic counter.  Every CTA's
// task sequence is a pure function of blockIdx.x and gridDim.x, so graph replay
// needs no reset node and concurrent calls remain isolated by their ordinary
// per-call workspaces.  The caller must launch the builder before all context
// consumers on the same stream and must choose a nonzero scan grid.  A useful
// initial cap on MI355X is one or two CTAs per CU; the device sequence count is
// allowed to be zero, in which case every capped CTA returns immediately.
#pragma once

#include <hip/hip_runtime.h>

#include "k2_kda_context_parallel_kernel.hpp"

namespace flashkda_hip::gfx950 {

inline constexpr int kHybridCompactGroupChunks = 64;
inline constexpr int kHybridCompactDirectMaxChunks = 64;

// Packed prefix ABI for the compact scan candidate.
//
// tile_prefix and pair_prefix retain their architecture-neutral meanings.
// context_prefix is the filtered G64 prefix consumed by the existing hybrid
// affine producers/replay.  sequence_worklist contains exactly the sequences
// represented by that filtered prefix and sequence_count[0] is its device-side
// length.  The list is sorted by descending context-group count, stably for
// ties, so deterministic grid-stride assignment does not leave one very long
// sequence until the tail.
//
// One thread is intentional and matches the established packed-prefix builder:
// serving N is small, and a single same-stream kernel publishes a coherent
// prefix/worklist snapshot without a host synchronization.
__global__ void k1_build_tile_prefix_hybrid_g64_compact_kernel(
        const int32_t* __restrict__ cu_seqlens,
        int N,
        int* __restrict__ tile_prefix,
        int* __restrict__ pair_prefix,
        int* __restrict__ context_prefix,
        int* __restrict__ sequence_worklist,
        int* __restrict__ sequence_count) {
    if (threadIdx.x != 0)
        return;

    int tile_acc = 0;
    int pair_acc = 0;
    int context_acc = 0;
    int long_count = 0;
    tile_prefix[0] = 0;
    pair_prefix[0] = 0;
    context_prefix[0] = 0;

    for (int seq = 0; seq < N; ++seq) {
        const int64_t length64 = int64_t(cu_seqlens[seq + 1]) -
                                 int64_t(cu_seqlens[seq]);
        const int chunks = int((length64 + 15) / 16);
        const int pairs = (chunks + 1) / 2;

        tile_acc += chunks;
        pair_acc += pairs;
        tile_prefix[seq + 1] = tile_acc;
        pair_prefix[seq + 1] = pair_acc;

        if (chunks > kHybridCompactDirectMaxChunks) {
            const int groups =
                (chunks + kHybridCompactGroupChunks - 1) /
                kHybridCompactGroupChunks;
            context_acc += groups;

            // Stable insertion sort by descending group count.  Comparing the
            // already-published context-prefix deltas avoids rereading both
            // cu_seqlen endpoints for every prior worklist entry.
            int position = long_count;
            while (position > 0) {
                const int previous_seq = sequence_worklist[position - 1];
                const int previous_groups =
                    context_prefix[previous_seq + 1] -
                    context_prefix[previous_seq];
                if (previous_groups >= groups)
                    break;
                sequence_worklist[position] = previous_seq;
                --position;
            }
            sequence_worklist[position] = seq;
            ++long_count;
        }
        context_prefix[seq + 1] = context_acc;
    }
    sequence_count[0] = long_count;
}

// Compact packed G64 affine scan, matching the established P0/NW2 arithmetic
// and affine-buffer ABI.  The launch is one-dimensional:
//
//   block = 128 threads
//   grid.x = min(host_task_upper, persistent_block_cap), grid.x >= 1
//
// A logical task owns (compact-long-sequence, head, V32).  Its two waves own
// independent V16 fragments and serially scan only that sequence's affine
// groups.  blockIdx.x, blockIdx.x + gridDim.x, ... are disjoint logical tasks,
// so no atomic counter, reset, or inter-CTA ordering is required.
template <bool HI = false, bool HO = false, bool SFP32 = false,
          typename RegBGemm = RegBX32>
__global__ void __launch_bounds__(128)
k2_kda_context_affine_scan_hybrid_g64_compact_grid_stride_nw2_kernel(
        const __bf16* __restrict__ affine_a,  // [G,H,K,K]
        float* __restrict__ affine_b,         // b -> h_in, [G,H,K,V]
        const void* __restrict__ init_state,  // [N,H,V,K], HI only
        void* __restrict__ final_state,        // direct/replay owns final state
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ context_prefix,
        const int* __restrict__ sequence_worklist,
        const int* __restrict__ sequence_count,
        int N,
        int H) {
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int NW = 2;
    constexpr int NKB = D / C;
    constexpr int V_GROUPS = D / (NW * BV);
    constexpr int NTHREADS = NW * 64;
    constexpr int AD = D + 4;
    constexpr int A_ROW_VECS = D / 8;
    constexpr int A_VECS = D * A_ROW_VECS;
    static_assert(V_GROUPS == 4 && A_VECS % NTHREADS == 0,
                  "compact G64/NW2 scan mapping changed");

    (void)HO;
    (void)final_state;
    (void)cu_seqlens;

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    __shared__ __bf16 amat[D * AD];

    const int long_count = sequence_count[0];
    const int64_t task_count64 =
        int64_t(long_count) * H * V_GROUPS;
    if (long_count <= 0 || N <= 0 || H <= 0 || task_count64 <= 0)
        return;

    for (int64_t task64 = int64_t(blockIdx.x);
         task64 < task_count64;
         task64 += int64_t(gridDim.x)) {
        const int task = int(task64);
        const int compact_seq = task / (H * V_GROUPS);
        const int task_rem = task - compact_seq * H * V_GROUPS;
        const int h = task_rem / V_GROUPS;
        const int v_group = task_rem - h * V_GROUPS;
        const int seq = sequence_worklist[compact_seq];

        // A malformed worklist must not turn into an out-of-bounds state or
        // prefix access.  The builder above only publishes [0,N) entries and
        // every thread in the CTA observes the same value, so this branch is
        // CTA-uniform and cannot strand a barrier.
        if (seq < 0 || seq >= N)
            continue;

        const int context_base = context_prefix[seq];
        const int context_count =
            context_prefix[seq + 1] - context_base;
        if (context_count <= 0)
            continue;

        const int v0 = (v_group * NW + wave) * BV;
        float hreg[NKB][4];
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

        for (int local_group = 0;
             local_group < context_count;
             ++local_group) {
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
                reinterpret_cast<bf16x8*>(amat + row * AD)[col8] =
                    a_src[idx];
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

            // The next group or logical task reuses amat.  This barrier is the
            // same P0 publication barrier as the established scan and keeps the
            // grid-stride loop CTA-uniform across task boundaries.
            __syncthreads();
            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    hreg[ktile][i] = next[ktile][i];
            }
        }
    }
}

}  // namespace flashkda_hip::gfx950

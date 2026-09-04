// Packed-varlen prefix builder for the opt-in persistent gfx942 P3 route.
//
// In addition to the architecture-neutral tile/pair/segment prefixes, this
// publishes one longest-first sequence worklist and resets its device-side
// task queue.  A single thread is intentional: N is small on the measured
// ragged inference shapes, and keeping all publications in one stream-ordered
// kernel avoids a host synchronization before the persistent consumer.
#pragma once

#include <hip/hip_runtime.h>

namespace flashkda_hip::gfx942 {

__global__ void k1_build_prefix_persistent_worklist_kernel(
        const int32_t* __restrict__ cu_seqlens,
        int N,
        int* __restrict__ tile_prefix,
        int* __restrict__ pair_prefix,
        int* __restrict__ segment_prefix,
        int* __restrict__ sequence_worklist,
        int* __restrict__ sequence_count,
        unsigned int* __restrict__ task_counter) {
    if (threadIdx.x != 0)
        return;

    int tile_acc = 0;
    int pair_acc = 0;
    int segment_acc = 0;
    tile_prefix[0] = 0;
    pair_prefix[0] = 0;
    segment_prefix[0] = 0;
    for (int seq = 0; seq < N; ++seq) {
        const int length = int(cu_seqlens[seq + 1] - cu_seqlens[seq]);
        tile_acc += (length + 15) / 16;
        pair_acc += (length + 31) / 32;
        segment_acc += (length + 63) / 64;
        tile_prefix[seq + 1] = tile_acc;
        pair_prefix[seq + 1] = pair_acc;
        segment_prefix[seq + 1] = segment_acc;

        // Insertion sort is stable for equal lengths.  Publishing long
        // sequences first prevents the last persistent wave from becoming a
        // single long-sequence tail on ragged batches.
        int position = seq;
        while (position > 0) {
            const int previous_seq = sequence_worklist[position - 1];
            const int previous_length = int(
                cu_seqlens[previous_seq + 1] - cu_seqlens[previous_seq]);
            if (previous_length >= length)
                break;
            sequence_worklist[position] = previous_seq;
            --position;
        }
        sequence_worklist[position] = seq;
    }
    sequence_count[0] = N;
    task_counter[0] = 0;
}

}  // namespace flashkda_hip::gfx942

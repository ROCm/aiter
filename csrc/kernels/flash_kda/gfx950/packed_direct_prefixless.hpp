#pragma once

#include <cstdint>

namespace flashkda_hip::gfx950 {

// The prefixless route removes a graph node while keeping metadata lookup no
// more expensive than the old prefix consumer: K1 performs a four-step binary
// search for N<=16 and K2 derives its sequence base in O(1).  Keep the limit
// shared by the host guard and both device-side mapping helpers.
constexpr int kPackedDirectPrefixlessMaxSequences = 16;

struct PackedC16SequenceMapping {
    int token_base;
    int token_length;
    int tile_base;
};

struct PackedC16TileMapping {
    int sequence;
    int local_tile;
    int token_base;
    int token_length;
    bool valid;
};

__device__ __forceinline__ int packed_c16_tile_count(int64_t length) {
    return int((length + 15) / 16);
}

// Give every sequence one conservative separator slot on top of the rounded
// global token offset.  For start s and length l,
//
//   ceil(s/16) + ceil(l/16) <= ceil((s+l)/16) + 1,
//
// so adjacent sequence ranges never overlap.  The sentinel base at N is
// exactly ceil(total_tokens/16)+N, i.e. the existing packed total_tiles upper
// bound.  Empty sequences also keep the bases strictly increasing.
__device__ __forceinline__ int packed_c16_tile_base(
        const int32_t* __restrict__ cu_seqlens, int sequence) {
    return packed_c16_tile_count(int64_t(cu_seqlens[sequence])) + sequence;
}

// Resolve one known sequence in O(1); all lanes in a CTA request the same
// values, allowing clang to retain the loads and arithmetic on the uniform
// scalar path on gfx950.
__device__ __forceinline__ PackedC16SequenceMapping
packed_c16_sequence_mapping(
        const int32_t* __restrict__ cu_seqlens, int sequence) {
    const int64_t token_base = cu_seqlens[sequence];
    const int64_t token_end = cu_seqlens[sequence + 1];
    return {
        int(token_base), int(token_end - token_base),
        packed_c16_tile_base(cu_seqlens, sequence)};
}

// Map one global upper-bound tile index through the strictly increasing gapped
// bases.  Separator slots and empty-sequence slots are reported invalid; K1
// returns from those CTAs while K2 never addresses them.
__device__ __forceinline__ PackedC16TileMapping packed_c16_tile_mapping(
        const int32_t* __restrict__ cu_seqlens,
        int N,
        int global_tile) {
    if (N <= 0 || global_tile < 0 ||
        global_tile >= packed_c16_tile_base(cu_seqlens, N))
        return {0, 0, 0, 0, false};

    int lo = 0;
    int hi = N;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (packed_c16_tile_base(cu_seqlens, mid) <= global_tile)
            lo = mid;
        else
            hi = mid;
    }
    const int64_t token_base = cu_seqlens[lo];
    const int64_t token_end = cu_seqlens[lo + 1];
    const int token_length = int(token_end - token_base);
    const int tile_base = packed_c16_tile_base(cu_seqlens, lo);
    const int local_tile = global_tile - tile_base;
    const bool valid = local_tile < packed_c16_tile_count(token_length);
    return {lo, local_tile, int(token_base), token_length, valid};
}

}  // namespace flashkda_hip::gfx950

// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// gfx950 MoE token permute / unpermute: the numbers of the kernels (workgroup, chunk, cache policies, the plan's
// LDS tables). No device code. Kernel arguments: opus_moe_permute_kargs.cuh.
#include "opus_moe_permute.h"   // OPUS_MOE_PERMUTE_GROUP_ALIGN
#include "opus_moe_permute_kargs.cuh"

namespace opus_moe_permute {

// The workgroup of every kernel.
struct workgroup_traits {
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = 4;
    static constexpr int BLOCK_SIZE = OPUS_MOE_PERMUTE_THREADS;
    static_assert(BLOCK_SIZE == NUM_WAVES * WARP_SIZE, "a workgroup is 4 waves");
    static constexpr int LDS_LIMIT = 65536;   // bytes
};
constexpr int THREADS = workgroup_traits::BLOCK_SIZE;   // the launcher's name for it

// K2 .. K5: rows of [rows, H] bf16, moved in 16-byte chunks.
struct row_traits : workgroup_traits {
    using D_ROW = opus::bf16_t;
    using D_ACC = opus::fp32_t;                               // the products of K4 / K5
    static constexpr int VEC = OPUS_MOE_PERMUTE_VEC;          // bf16 elements of a chunk: one load / store per lane
    static constexpr int CHUNK_BYTES = VEC * sizeof(D_ROW);
    static_assert(CHUNK_BYTES == 16, "a chunk is one 16-byte load");

    // Cache policy of a load / store (the aux of the gfx950 buffer instructions; the bytes are the same).
    static constexpr int CPOL_DEFAULT = 0;
    static constexpr int CPOL_NT = 2;                         // nt: non-temporal
    static constexpr int CPOL_SYSTEM = 17;                    // sc0 sc1: system scope
};

// The plan kernels: a workgroup takes a block of TOKENS tokens (one thread per token) and has one thread per
// expert.
struct plan_traits : workgroup_traits {
    static constexpr int TOKENS = OPUS_MOE_PERMUTE_PLAN_TOKENS;     // of a block
    static constexpr int EXPERTS = OPUS_MOE_PERMUTE_PLAN_EXPERTS;   // the most experts
    static_assert(TOKENS == BLOCK_SIZE, "one thread per token");
    static_assert(EXPERTS <= BLOCK_SIZE, "one thread per expert");

    // LDS bitmap: one bit per (expert, token of the block).
    static constexpr int WORD_BITS = 32;
    static constexpr int WORDS = TOKENS / WORD_BITS;          // words of one expert
    static_assert(WORDS * WORD_BITS == TOKENS);

    // Block counts of an expert that are loaded together; the counts workspace holds a multiple of it.
    static constexpr int COUNT_BATCH = OPUS_MOE_PERMUTE_PLAN_COUNT_BATCH;

    static constexpr int GROUP_ALIGN = OPUS_MOE_PERMUTE_GROUP_ALIGN;   // rows every expert's group is padded to
    static_assert((GROUP_ALIGN & (GROUP_ALIGN - 1)) == 0);

    // Fill workgroups: FILL_EXPERTS experts each, FILL_THREADS threads per expert; then FILL_TAIL_ROWS of the
    // rows past the last group.
    static constexpr int FILL_EXPERTS = OPUS_MOE_PERMUTE_PLAN_FILL_EXPERTS;
    static constexpr int FILL_THREADS = BLOCK_SIZE / FILL_EXPERTS;
    static constexpr int FILL_TAIL_ROWS = FILL_EXPERTS * GROUP_ALIGN;

    static constexpr int LDS_BYTES = EXPERTS * WORDS * 4 + 2 * EXPERTS * 4;   // bitmap + total, first
    static_assert(LDS_BYTES <= LDS_LIMIT);
};

} // namespace opus_moe_permute

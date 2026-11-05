// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Offline-only: use the new variable attribute, never the old value builtin.
// A pinned automatic local has no address, so it must stay inside set(), rather
// than becoming a member or a reference-captured variable in a static_for lambda.
//
// Keep an SSA copy between accesses. Re-reading the physical tuple in get()
// makes each loop block read-before-write: scanNamedRegs reserves the whole
// tuple, preventing the WMMA virtual destination from coalescing into it. That
// version emitted low-D -> high-register copies and extra VA_VDST drains.
// set() instead writes the named local and immediately copies it out, preserving
// normal SSA/phi state. Actual WMMA placement is checked in the emitted ISA, not
// assumed from the mere presence of read_register/write_register in the IR.
#if !OPUS_NO_PIN && !__has_attribute(amdgpu_pin_vgpr)
#error "OPUS CO requires LLVM with automatic-local amdgpu_pin_vgpr support"
#endif

namespace opus_co_detail {

template <typename V, int Base, int Stride, int N>
struct named_vgpr_bank {
    static_assert(N > 0 && Base >= 0 && Stride > 0, "invalid named VGPR bank");
    static_assert(sizeof(V) % 4 == 0, "a named tuple must fill whole VGPRs");
    static_assert(Stride >= static_cast<int>(sizeof(V) / 4), "named VGPR tuples must not overlap");
    static_assert(Base + (N - 1) * Stride + sizeof(V) / 4 <= 1024,
                  "named VGPR bank overruns the gfx1250 register file");

    V head{};
    named_vgpr_bank<V, Base + Stride, Stride, N - 1> tail;

    template <int I>
    __device__ __attribute__((always_inline)) V get() const {
        static_assert(I >= 0 && I < N, "named VGPR index out of range");
        if constexpr (I == 0) return head;
        else return tail.template get<I - 1>();
    }

    template <int I>
    __device__ __attribute__((always_inline)) void set(V value) {
        static_assert(I >= 0 && I < N, "named VGPR index out of range");
        if constexpr (I == 0) {
#if OPUS_NO_PIN
            head = value;
#else
            __attribute__((amdgpu_pin_vgpr(Base))) V slot = value;
            head = slot;
#endif
        } else tail.template set<I - 1>(value);
    }

    __device__ __attribute__((always_inline)) void zero() {
        set<0>(V{});
        tail.zero();
    }
};

template <typename V, int Base, int Stride>
struct named_vgpr_bank<V, Base, Stride, 0> {
    template <int I> __device__ V get() const {
        static_assert(I >= 0 && I < 0, "read from an empty named VGPR bank");
        return V{};
    }
    template <int I> __device__ void set(V) {
        static_assert(I >= 0 && I < 0, "write to an empty named VGPR bank");
    }
    __device__ void zero() {}
};

} // namespace opus_co_detail

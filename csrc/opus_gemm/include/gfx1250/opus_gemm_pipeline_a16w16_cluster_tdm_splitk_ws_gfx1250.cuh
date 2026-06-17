// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 bf16 TDM a16w16 GEMM -- 4-WAVE variant, split-K via an fp32
// WORKSPACE + a separate REDUCE kernel (NO atomic_add, NO self-clear, NO
// semaphore). Ported from
//   demon_gcn/wmma_opus_rdna4/gemm_a16w16_cluster_tdm_splitk_reduce_4wave.cc
// and generalized to arbitrary B_M/B_N/B_K and TileN/TileM consumer tiling.
//
//   C[M,N] = A[M,K] @ B[N,K]^T (+ bias[N], folded in the reduce kernel).
//
// NO-CLUSTER plain grid: grid = (M/B_M, N/B_N, split_k); each workgroup owns
// one B_M x B_N tile and loads its own A/B via TDM (no cross-WG multicast).
// 4 waves / workgroup (128 threads): w0 = A producer, w1 = B producer,
// w2,w3 = WMMA consumers (split along N for TileN, along M for TileM).
// grid.z = split_k: each z-layer covers a K chunk and writes its fp32 partial
// into ITS OWN workspace slice ws[split_idx][padded_m][padded_n] via a PLAIN
// store (each (split,m,n) cell written by exactly one WG -> no contention, no
// atomic / clear / handshake). batch is handled by a per-batch host launch.
//
// This file carries NO local constexpr / KernelTraits: every compile-time
// value comes from the traits header (T::...).
#pragma once

#include "opus_gemm_traits_a16w16_gfx1250.cuh"

#ifdef __HIP_DEVICE_COMPILE__
using namespace opus;
using opus::operator""_I;
#endif

__host__ __device__ constexpr inline int opus_ctdm_ws_ceil_div_i(int a, int b) {
    return (a + b - 1) / b;
}
__host__ __device__ constexpr inline int opus_ctdm_ws_min_i(int a, int b) {
    return a < b ? a : b;
}



template <typename UserTraits>
__global__ __launch_bounds__(128, 1)
void gemm_a16w16_cluster_tdm_splitk_ws_kernel_gfx1250(opus_gemm_cluster_tdm_ws_kargs_gfx1250 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx1250__)
    using T = remove_cvref_t<UserTraits>;
    using DataA = typename T::DataA;
    using DataB = typename T::DataB;
    using DataAcc = typename T::DataAcc;
    DECLARE_NAMED_BARRIERS();

    auto bjs = [&](auto IdN) {
        constexpr int id = IdN.value;
        if      constexpr (id == 1) { s_barrier_join_ptr(&__nbar_1); __builtin_amdgcn_s_barrier_signal(1); }
        else if constexpr (id == 2) { s_barrier_join_ptr(&__nbar_2); __builtin_amdgcn_s_barrier_signal(2); }
        else if constexpr (id == 3) { s_barrier_join_ptr(&__nbar_3); __builtin_amdgcn_s_barrier_signal(3); }
        else if constexpr (id == 4) { s_barrier_join_ptr(&__nbar_4); __builtin_amdgcn_s_barrier_signal(4); }
        else if constexpr (id == 5) { s_barrier_join_ptr(&__nbar_5); __builtin_amdgcn_s_barrier_signal(5); }
        else                        { s_barrier_join_ptr(&__nbar_6); __builtin_amdgcn_s_barrier_signal(6); }
    };
    auto bjsw = [&](auto IdN) {
        constexpr int id = IdN.value;
        if      constexpr (id == 1) { s_barrier_join_ptr(&__nbar_1); __builtin_amdgcn_s_barrier_signal(1); __builtin_amdgcn_s_barrier_wait(1); }
        else if constexpr (id == 2) { s_barrier_join_ptr(&__nbar_2); __builtin_amdgcn_s_barrier_signal(2); __builtin_amdgcn_s_barrier_wait(2); }
        else if constexpr (id == 3) { s_barrier_join_ptr(&__nbar_3); __builtin_amdgcn_s_barrier_signal(3); __builtin_amdgcn_s_barrier_wait(3); }
        else if constexpr (id == 4) { s_barrier_join_ptr(&__nbar_4); __builtin_amdgcn_s_barrier_signal(4); __builtin_amdgcn_s_barrier_wait(4); }
        else if constexpr (id == 5) { s_barrier_join_ptr(&__nbar_5); __builtin_amdgcn_s_barrier_signal(5); __builtin_amdgcn_s_barrier_wait(5); }
        else                        { s_barrier_join_ptr(&__nbar_6); __builtin_amdgcn_s_barrier_signal(6); __builtin_amdgcn_s_barrier_wait(6); }
    };

    const int wave_id = __builtin_amdgcn_readfirstlane((int)opus::waveid_in_workgroup());
    const int lane_id = (int)opus::lane_id();
    const bool is_producer = wave_id < T::kNumProducerWaves;

    // No cluster: each workgroup maps directly to one (tile_row, tile_col) tile
    // via blockIdx and loads its own A/B tile (no TDM multicast). The TDM
    // workgroup mask is "self only" (bit 0).
    const int tile_row = (int)__builtin_amdgcn_workgroup_id_x() * T::kBlockM;
    const int tile_col = (int)__builtin_amdgcn_workgroup_id_y() * T::kBlockN;
    const uint16_t mask_a = 1u;
    const uint16_t mask_b = 1u;

    const int stride_a = kargs.stride_a;
    const int stride_b = kargs.stride_b;

    const int split_k     = kargs.split_k < 1 ? 1 : kargs.split_k;
    const int split_idx   = (int)__builtin_amdgcn_workgroup_id_z();
    const int k_steps_tot = opus_ctdm_ws_ceil_div_i(kargs.k, T::kBlockK);
    const int steps_per   = opus_ctdm_ws_ceil_div_i(k_steps_tot, split_k);
    const int k_step_beg  = split_idx * steps_per;
    const int k_step_end  = opus_ctdm_ws_min_i(k_step_beg + steps_per, k_steps_tot);
    const int k_steps     = k_step_end - k_step_beg;
    if (k_steps <= 0) return;   // empty trailing split: launcher clamps split_k so this is rare

    __shared__ char lds_buf[T::kLdsTotalBytes];   // >=160KB tail-pad forces 1 WG/CU when kWgPerCu==1
    DataA* smem_a = reinterpret_cast<DataA*>(lds_buf);
    DataB* smem_b = reinterpret_cast<DataB*>(lds_buf + T::kSegBytesA);
    constexpr int slot_a = T::kSlotElemsA;
    constexpr int slot_b = T::kSlotElemsB;

    using WindowA = typename T::WindowA;
    using WindowB = typename T::WindowB;

    if (!is_producer) {
        s_barrier_init_ptr(&__nbar_1, T::kNumWaves); s_barrier_init_ptr(&__nbar_2, T::kNumWaves); s_barrier_init_ptr(&__nbar_3, T::kNumWaves);
        s_barrier_init_ptr(&__nbar_4, T::kNumWaves); s_barrier_init_ptr(&__nbar_5, T::kNumWaves); s_barrier_init_ptr(&__nbar_6, T::kNumWaves);
    }
    __builtin_amdgcn_s_barrier();

    // ---- Producers: w0 fills A slots, w1 fills B slots (triple buffer). ----
    if (is_producer) {
        const int gk0 = k_step_beg * T::kBlockK;
        const uint32_t k_extent = (uint32_t)(kargs.k - gk0);
        constexpr int slot_a_b = T::kSlotBytesA;
        constexpr int slot_b_b = T::kSlotBytesB;
        constexpr auto KStep = opus::number<T::kBlockK>{};

        // TDM row extent = REMAINING valid rows (kargs.m - tile_row), NOT the
        // full tile height kARows. tensor_dim1 = saturating_sub(extent1,
        // origin1) = saturating_sub(m, tile_row) then bounds the global read to
        // min(B_M, m - tile_row): the TDM never issues OOB global loads for the
        // padded tail rows of the last M-tile (M need not be a multiple of B_M).
        // Padded LDS rows keep stale/garbage data -> their (independent) C rows
        // are written to the padded workspace, which the reduce kernel never
        // reads (it iterates m in [0, M)). Same idea for B/N below.
        const int row_extent_a = kargs.m - tile_row;
        const int row_extent_b = kargs.n - tile_col;
        // Producer body, generic over prefetch depth (T::kNumSlots = 2 or 3).
        // slot_b_bytes is the per-operand LDS slot stride (A: slot_a_b, B: slot_b_b).
        // Compile-time dual specialization keeps every barrier id a literal and
        // the slot ring (slot = k % kNumSlots) a compile-time constant per path.
        // Runtime DATA-signal dispatchers (epilogue announces slots whose id is
        // (k_steps-x) % kNumSlots, a runtime value). s_barrier_signal needs a
        // literal id, so switch the runtime slot -> literal DATA id (1+slot).
        auto bjs_data_rt = [&](int slot) __attribute__((always_inline)) {
            if (slot == 0) bjs(opus::number<1>{});
            else if (slot == 1) bjs(opus::number<2>{});
            else bjs(opus::number<3>{});
        };

        // DECOUPLED run-ahead producer: keep kNumSlots TDMs in flight. Each main-
        // loop step issues ONE prefetch into the just-freed slot, then waits for
        // exactly the OLDEST outstanding TDM (s_wait_tensorcnt(kNumSlots-1), a
        // FIXED count) and signals DATA for THAT completed (older) slot -- not the
        // one just issued. This lets the producer run kNumSlots-1 steps ahead of
        // the consumer (TDM loads complete in-order, SPG p.33). The first
        // kNumSlots loads are issued in the prologue (no FREE wait: first use); the
        // last kNumSlots-1 announces drain in the epilogue (no new issue).
        auto produce = [&](auto& w, int slot_bytes) __attribute__((always_inline)) {
        if constexpr (T::kNumSlots == 3) {
            // prologue: issue steps 0,1,2 -> slots 0,1,2 (3 in flight).
            w.load_to_lds();
            if (k_steps >= 2) { w.move(KStep, 0_I,0_I,0_I,0_I, slot_bytes); w.load_to_lds(); }
            if (k_steps >= 3) { w.move(KStep, 0_I,0_I,0_I,0_I, slot_bytes); w.load_to_lds(); }
            if (k_steps >= 3) {
                __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<1>{});  // drain ALL: partial wait announces a slot whose TDM LDS write is not yet visible to the consumer wave (producer-side visibility gap, run-ahead races); draining all lets it land.
                // issue step p into slot p%3, then drain ALL outstanding TDMs and
                // announce the completed step p-2 (slot (p-2)%3). FREE[p%3] gates
                // overwriting slot p%3 (consumer must be done with step p-3).
                for (int p = 3; p < k_steps; ) {
                    bjsw(opus::number<4>{}); w.move(KStep,0_I,0_I,0_I,0_I, -2*slot_bytes); w.load_to_lds(); __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<2>{}); if (++p >= k_steps) break;  // slot0 loaded; announce step p-2 slot1
                    bjsw(opus::number<5>{}); w.move(KStep,0_I,0_I,0_I,0_I,    slot_bytes); w.load_to_lds(); __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<3>{}); if (++p >= k_steps) break;  // slot1 loaded; announce slot2
                    bjsw(opus::number<6>{}); w.move(KStep,0_I,0_I,0_I,0_I,    slot_bytes); w.load_to_lds(); __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<1>{}); ++p;                          // slot2 loaded; announce slot0
                }
                // epilogue: announce the last 2 still-in-flight steps (no issue).
                __builtin_amdgcn_s_wait_tensorcnt(0); bjs_data_rt((k_steps - 2) % 3);
                __builtin_amdgcn_s_wait_tensorcnt(0); bjs_data_rt((k_steps - 1) % 3);
            } else if (k_steps == 2) {
                __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<1>{});
                __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<2>{});
            } else {  // k_steps == 1
                __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<1>{});
            }
        } else {  // kNumSlots == 2: run 1 step ahead. ids 1,2 / 4,5.
            w.load_to_lds();                                                              // step0 -> slot0
            if (k_steps >= 2) { w.move(KStep, 0_I,0_I,0_I,0_I, slot_bytes); w.load_to_lds(); }  // step1 -> slot1
            if (k_steps >= 2) {
                __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<1>{});  // drain ALL before announce (producer-side LDS visibility; see P=3 path)
                for (int p = 2; p < k_steps; ) {
                    bjsw(opus::number<4>{}); w.move(KStep,0_I,0_I,0_I,0_I, -slot_bytes); w.load_to_lds(); __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<2>{}); if (++p >= k_steps) break;  // slot0 loaded; announce step p-1 slot1
                    bjsw(opus::number<5>{}); w.move(KStep,0_I,0_I,0_I,0_I,  slot_bytes); w.load_to_lds(); __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<1>{}); ++p;                          // slot1 loaded; announce slot0
                }
                // epilogue: 1 step still in flight.
                __builtin_amdgcn_s_wait_tensorcnt(0);
                if (((k_steps - 1) & 1) == 0) bjs(opus::number<1>{});
                else                          bjs(opus::number<2>{});
            } else {  // k_steps == 1
                __builtin_amdgcn_s_wait_tensorcnt(0); bjs(opus::number<1>{});
            }
        }
        };  // produce

        if (wave_id == 0) {
            WindowA w;
            w.make(reinterpret_cast<uintptr_t>(smem_a), kargs.ptr_a, 0,
                   k_extent, (uint32_t)row_extent_a, (uint64_t)stride_a,
                   (uint32_t)gk0, (uint32_t)tile_row);
            w.desc.sg1[0] = (w.desc.sg1[0] & 0xFFFF0000u) | mask_a;
            produce(w, slot_a_b);
        } else {  // wave_id == 1 -> B
            WindowB w;
            w.make(reinterpret_cast<uintptr_t>(smem_b), kargs.ptr_b, 0,
                   k_extent, (uint32_t)row_extent_b, (uint64_t)stride_b,
                   (uint32_t)gk0, (uint32_t)tile_col);
            w.desc.sg1[0] = (w.desc.sg1[0] & 0xFFFF0000u) | mask_b;
            produce(w, slot_b_b);
        }
        return;
    }

    // ---- Consumers (w2,w3): WMMA accumulate, then plain store to workspace. ----
    const int wave_split = wave_id - T::kNumProducerWaves;   // 0..1
    // TileN: consumers split N (wave_n = wave_split, wave_m = 0).
    // TileM: consumers split M (wave_m = wave_split, wave_n = 0).
    const int wave_m = (T::LAYOUT == opus_gfx1250::kCtdmLayoutTileM) ? wave_split : 0;
    const int wave_n = (T::LAYOUT == opus_gfx1250::kCtdmLayoutTileM) ? 0 : wave_split;

    auto mma = make_tiled_mma<DataA, DataB, DataAcc>(
        seq<T::kExpM, T::kExpN, T::kExpKHalf>{},
        seq<T::kTileM, T::kTileN, T::kTileK>{},
        seq<T::kWmmaM, T::kWmmaN, T::kWmmaK>{}, wmma_adaptor_swap_ab{});
    auto u_ra = make_layout_ra_ctdm<T>(lane_id, wave_m);
    auto u_rb = make_layout_rb_ctdm<T>(lane_id, wave_n);

    // Double-buffered WMMA source registers (ping-pong across consume rounds).
    // The single-buffer version had a WAR reg race: round k+1's ds_loads
    // overwrite the SAME VGPRs that round k's (multi-cycle) WMMAs are still
    // reading -- the race window scales with the WMMA count per round
    // (kExpM*kExpN), so high-expansion tiles fail non-deterministically. With
    // two buffers, consecutive rounds use DISTINCT VGPRs, so an overwrite never
    // aliases an in-flight WMMA read.
    typename decltype(mma)::vtype_a v_a[2];   // 2-deep ping-pong (see consume prefetch)
    typename decltype(mma)::vtype_b v_b[2];
    typename decltype(mma)::vtype_c reg_c;
    clear(reg_c);

    auto consume = [&](auto Sn, auto AFirstN) __attribute__((always_inline)) {
        constexpr int s = Sn.value;
        constexpr bool AFirst = AFirstN.value;
        bjsw(opus::number<1 + s>{});
        // Explicit per-round (per-half) schedule, replacing the fragile
        // sched_group_barrier hints. Those mis-schedule on high-expand tiles
        // (e.g. tileM B_N=64 -> kExpN=4) and surface as deterministic NaN. Per
        // half we: (1) sched_barrier(0) wall so the next round's ds_reads are
        // not hoisted (keeps the dscnt accounting exact per round); (2) issue
        // this round's ds_reads (== kSchedDsCount); (3) s_wait_dscnt(0) batch-
        // waits exactly this round's ds batch (nothing else is outstanding);
        // (4) WMMA, now guaranteed to read landed registers.
        // Prefetched (software-pipelined) reads with a 2-deep ping-pong buffer.
        // Loading round i+1 BEFORE issuing round i's WMMA keeps buf[i&1] (read by
        // the WMMA) and buf[(i+1)&1] (just loaded) SIMULTANEOUSLY live -> the
        // compiler cannot coalesce the two buffers into one register set (a plain
        // store-then-use double buffer, OR an inline-asm "+v" pin, DOES get
        // coalesced / is allocation-dependent and unreliable). Distinct VGPRs per
        // adjacent round => round i+1's ds_load never overwrites a register that
        // round i's multi-cycle WMMA is still reading (WAR reg race).
        // do_load issues a round's A+B reads then DRAINS them (round-granular
        // s_wait_dscnt(0)). Draining per round caps the in-flight ds at ONE round
        // (<=63 DScnt even for kExpN=8) instead of two (prefetch overlap = 2
        // rounds = up to 72 > 63). The next round is still loaded BEFORE the
        // current round's WMMA, so both ping-pong buffers hold valid data in
        // DISTINCT VGPRs simultaneously (liveness overlap => no coalescing =>
        // round i+1's ds never overwrites a register round i's multi-pass WMMA is
        // still reading -- the WMMA-source WAR, MI400 SPG 4.6.12.1).
        auto do_load = [&](int half, int buf) __attribute__((always_inline)) {
            auto sa = make_smem(smem_a + s*slot_a + half*T::kKHalfElems);
            auto sb = make_smem(smem_b + s*slot_b + half*T::kKHalfElems);
            if constexpr (AFirst) { v_a[buf] = load<T::kVecA>(sa, u_ra); v_b[buf] = load<T::kVecB>(sb, u_rb); }
            else                  { v_b[buf] = load<T::kVecB>(sb, u_rb); v_a[buf] = load<T::kVecA>(sa, u_ra); }
            __builtin_amdgcn_sched_barrier(0);
            opus::s_wait_dscnt(opus::number<0>{});
            __builtin_amdgcn_sched_barrier(0);
        };
        __builtin_amdgcn_sched_barrier(0);
        do_load(0, 0);                                   // prologue: round 0 -> buf0 (loaded+drained)
        #pragma unroll
        for (int i = 0; i < T::kHalvesPerSlot; ++i) {
            const int cur = i & 1;
            __builtin_amdgcn_sched_barrier(0);
            if (i + 1 < T::kHalvesPerSlot) do_load(i + 1, (i + 1) & 1);  // load+drain next round (distinct buf, 1-round in-flight)
            __builtin_amdgcn_sched_barrier(0);
            reg_c = mma(v_a[cur], v_b[cur], reg_c);
        }
        // FREE: all of this slot's ds_reads are drained per-round above; the
        // wall keeps the FREE signal AFTER the last WMMA so the run-ahead
        // producer cannot overwrite a slot whose reads are still pending.
        __builtin_amdgcn_sched_barrier(0);
        bjs(opus::number<4 + s>{});
    };
    auto run = [&](auto AFirstN) __attribute__((always_inline)) {
        if constexpr (T::kNumSlots == 3) {
            int k = 0;
            for (; k + 3 <= k_steps; k += 3) { consume(0_I, AFirstN); consume(1_I, AFirstN); consume(2_I, AFirstN); }
            const int rem = k_steps - k;
            if (rem >= 1) consume(0_I, AFirstN);
            if (rem >= 2) consume(1_I, AFirstN);
        } else {  // kNumSlots == 2: unroll by 2 (slot = k % 2, ids 1,2 / 4,5).
            int k = 0;
            for (; k + 2 <= k_steps; k += 2) { consume(0_I, AFirstN); consume(1_I, AFirstN); }
            if (k_steps - k >= 1) consume(0_I, AFirstN);
        }
    };
    if (wave_split == 0) run(opus::true_type{});
    else                 run(opus::false_type{});

    // ---- Plain store the fp32 partial into this split's workspace slice. ----
    // ws layout (per host launch) = [split_k][padded_m][padded_n]; each
    // (split,m,n) cell written by exactly one WG -> no contention, no atomic.
    // bias is folded once by the reduce kernel (not here).
    constexpr int kCVec = T::kCVec;   // 4 (fp32 dwordx4)
    DataAcc* ws_ptr = reinterpret_cast<DataAcc*>(kargs.ws_handle->ptr);
    const size_t ws_split = (size_t)split_idx * (size_t)kargs.stride_ws_batch;
    const size_t ws_base  = ws_split + (size_t)tile_row * (size_t)kargs.stride_ws + (size_t)tile_col;
    const unsigned int ws_bytes =
        (unsigned int)(((size_t)kargs.stride_ws_batch
                        - ((size_t)tile_row * kargs.stride_ws + tile_col)) * sizeof(DataAcc));
    auto g_ws = make_gmem<DataAcc>(ws_ptr + ws_base, ws_bytes);
    auto u_gc = partition_layout_c<kCVec>(mma, opus::make_tuple((int)kargs.stride_ws, 1_I),
                    opus::make_tuple(wave_m, lane_id % mma.grpn_c, wave_n, lane_id / mma.grpn_c));
    store<kCVec>(g_ws, reg_c, u_gc, 0);
#else
    (void)kargs;   // non-gfx1250 device pass: empty stub (multi-arch wheel safety)
#endif // __gfx1250__
#endif // __HIP_DEVICE_COMPILE__
}

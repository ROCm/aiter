// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 bf16 TDM a16w16 GEMM, 4-wave split-K via fp32 workspace + separate
// reduce kernel.  C[M,N] = A[M,K] @ B[N,K]^T (+ bias[N], folded in reduce).
//
// Plain grid (no cluster): grid = (M/B_M, N/B_N, split_k); each WG owns one
// B_M x B_N tile and TDM-loads its own A/B. 4 waves: w0=A producer, w1=B
// producer, w2/w3=WMMA consumers (split N for TileN, M for TileM). Each
// (split,m,n) cell is written by exactly one WG -> plain store, no atomics.
// All compile-time values come from the traits header (T::...).
//
// Producer/consumer sync = a PER-SLOT pair of NAMED barriers (ported from the
// 4-wave atomic cluster pipeline's data/compute run-ahead), kNumSlots-deep:
//   DATA[s] (id 1+s):     producers (w0=A, w1=B) join+signal (NO wait ->
//                         run-ahead); consumers join+signal+wait. The slot is
//                         "ready" only once BOTH producers signalled (A and B).
//   FREE[s] (id 1+P+s):   consumers join+signal (no wait); producers
//                         join+signal+wait before reusing slot s.
//   memcnt = kNumWaves (4) = ALL waves. Every barrier needs all 4 signals to
//   complete, so the waiting side is itself a required signaler -> it joined
//   before completion -> no completion-before-join deadlock (MI400 SPG p.84/87).
// Run-ahead: the prologue announces DATA[0..nload-1] up front, then each steady
// reload is gated only by FREE -> the producer runs ~kNumSlots K-steps ahead,
// its TDM DMA overlapping the consumer WMMA. The K loop is unrolled by kNumSlots
// so the slot (and thus every barrier id) is a compile-time constant; the
// K%kNumSlots tail is handled separately.
#pragma once

#include "opus_gemm_traits_a16w16_gfx1250.cuh"

#ifdef __HIP_DEVICE_COMPILE__
using namespace opus;
using opus::operator""_I;
#endif

// Distinct names (opus_ctdm_ws_*) so this header can be included alongside the
// plain-grid pipeline header (which defines opus_ctdm_ws_* with the same bodies).
__host__ __device__ constexpr inline int opus_ctdm_ws_ceil_div_i(int a, int b) {
    return (a + b - 1) / b;
}
__host__ __device__ constexpr inline int opus_ctdm_ws_min_i(int a, int b) {
    return a < b ? a : b;
}
__host__ __device__ constexpr inline int opus_ctdm_ws_max_i(int a, int b) {
    return a > b ? a : b;
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
    DECLARE_NAMED_BARRIERS();   // __nbar_1..__nbar_15 (we use 1..2*kNumSlots <= 6)

    // Compile-time named-barrier helpers (id = compile-time constant).
    // Barrier id layout (P = kNumSlots):
    //   DATA[s]   = 1        + s   (ids 1..P)      memcnt = kNumWaves (2 prod + 2 cons)
    //   FREE_A[s] = 1 +   P  + s   (ids 1+P..2P)   memcnt = 1 + kNumConsumerWaves (prodA + 2 cons)
    //   FREE_B[s] = 1 + 2*P  + s   (ids 1+2P..3P)  memcnt = 1 + kNumConsumerWaves (prodB + 2 cons)
    // PER-PRODUCER FREE barriers: each producer (w0=A, w1=B) reuse-waits on its OWN
    // FREE barrier. The consumer signals BOTH FREE_A[s] and FREE_B[s] when it frees a
    // slot. memcnt = 3 means a FREE_X[s] generation can only complete with that
    // producer's own signal (2 consumer signals < 3), so the producer is always a
    // joined member at completion -> it can never miss the completion broadcast.
    // (A single shared FREE[s] with memcnt=4 let the consumer's extra prologue-slot
    // free substitute for one producer, releasing only the producer that happened to
    // be joined and hanging the other -> the split-K / desynced-producer deadlock.)
    //   binit = init a barrier to a given memcnt
    //   bjs   = signal only             (run-ahead / no-wait side)
    //   bjsw  = join + signal + wait    (waiting side; join sets namedBarID, its own
    //                                    signal is required for completion)
    auto binit = [&](auto IdN, uint32_t mc) __attribute__((always_inline)) {
        constexpr int id = IdN.value;
        if      constexpr (id == 1) s_barrier_init_ptr(&__nbar_1, mc);
        else if constexpr (id == 2) s_barrier_init_ptr(&__nbar_2, mc);
        else if constexpr (id == 3) s_barrier_init_ptr(&__nbar_3, mc);
        else if constexpr (id == 4) s_barrier_init_ptr(&__nbar_4, mc);
        else if constexpr (id == 5) s_barrier_init_ptr(&__nbar_5, mc);
        else if constexpr (id == 6) s_barrier_init_ptr(&__nbar_6, mc);
        else if constexpr (id == 7) s_barrier_init_ptr(&__nbar_7, mc);
        else if constexpr (id == 8) s_barrier_init_ptr(&__nbar_8, mc);
        else                        s_barrier_init_ptr(&__nbar_9, mc);
    };
    auto bjs = [&](auto IdN) __attribute__((always_inline)) {
        constexpr int id = IdN.value;
        if      constexpr (id == 1) { __builtin_amdgcn_s_barrier_signal(1); }
        else if constexpr (id == 2) { __builtin_amdgcn_s_barrier_signal(2); }
        else if constexpr (id == 3) { __builtin_amdgcn_s_barrier_signal(3); }
        else if constexpr (id == 4) { __builtin_amdgcn_s_barrier_signal(4); }
        else if constexpr (id == 5) { __builtin_amdgcn_s_barrier_signal(5); }
        else if constexpr (id == 6) { __builtin_amdgcn_s_barrier_signal(6); }
        else if constexpr (id == 7) { __builtin_amdgcn_s_barrier_signal(7); }
        else if constexpr (id == 8) { __builtin_amdgcn_s_barrier_signal(8); }
        else                        { __builtin_amdgcn_s_barrier_signal(9); }
    };
    auto bjsw = [&](auto IdN) __attribute__((always_inline)) {
        constexpr int id = IdN.value;
        if      constexpr (id == 1) { s_barrier_join_ptr(&__nbar_1); __builtin_amdgcn_s_barrier_signal(1); __builtin_amdgcn_s_barrier_wait(1); }
        else if constexpr (id == 2) { s_barrier_join_ptr(&__nbar_2); __builtin_amdgcn_s_barrier_signal(2); __builtin_amdgcn_s_barrier_wait(2); }
        else if constexpr (id == 3) { s_barrier_join_ptr(&__nbar_3); __builtin_amdgcn_s_barrier_signal(3); __builtin_amdgcn_s_barrier_wait(3); }
        else if constexpr (id == 4) { s_barrier_join_ptr(&__nbar_4); __builtin_amdgcn_s_barrier_signal(4); __builtin_amdgcn_s_barrier_wait(4); }
        else if constexpr (id == 5) { s_barrier_join_ptr(&__nbar_5); __builtin_amdgcn_s_barrier_signal(5); __builtin_amdgcn_s_barrier_wait(5); }
        else if constexpr (id == 6) { s_barrier_join_ptr(&__nbar_6); __builtin_amdgcn_s_barrier_signal(6); __builtin_amdgcn_s_barrier_wait(6); }
        else if constexpr (id == 7) { s_barrier_join_ptr(&__nbar_7); __builtin_amdgcn_s_barrier_signal(7); __builtin_amdgcn_s_barrier_wait(7); }
        else if constexpr (id == 8) { s_barrier_join_ptr(&__nbar_8); __builtin_amdgcn_s_barrier_signal(8); __builtin_amdgcn_s_barrier_wait(8); }
        else                        { s_barrier_join_ptr(&__nbar_9); __builtin_amdgcn_s_barrier_signal(9); __builtin_amdgcn_s_barrier_wait(9); }
    };

    const int wave_id = __builtin_amdgcn_readfirstlane((int)opus::waveid_in_workgroup());
    const int lane_id = (int)opus::lane_id();
    const bool is_producer = wave_id < T::kNumProducerWaves;

    // TDM workgroup_mask MUST be 0 (plain grid, no cluster): a non-zero mask
    // makes TENSOR_LOAD_TO_LDS use CLUSTER_LOAD_ASYNC and wait for cluster peers
    // -> deadlock without a real cluster (MI400 SPG Tbl 80 / §4.10.3). mask=0
    // selects GLOBAL_LOAD_ASYNC.
    const int tile_row = (int)__builtin_amdgcn_workgroup_id_x() * T::kBlockM;
    const int tile_col = (int)__builtin_amdgcn_workgroup_id_y() * T::kBlockN;
    const uint16_t mask_a = 0u;
    const uint16_t mask_b = 0u;

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

    // ONE wave (the first consumer, wave 2) inits the per-slot named barriers
    // (memcnt = kNumWaves = all 4 waves must signal). S_BARRIER_INIT writes the
    // shared barrier-unit state, so a single initializer suffices; the workgroup
    // barrier below then publishes it to every wave before first use.
    if (wave_id == T::kNumProducerWaves) {
        constexpr uint32_t kFreeMemCnt = 1 + T::kNumConsumerWaves;   // prodX + 2 consumers
        opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
            constexpr int s = decltype(sN)::value;
            binit(opus::number<1 + s>{}, T::kNumWaves);                   // DATA[s]
            binit(opus::number<1 + T::kNumSlots + s>{}, kFreeMemCnt);     // FREE_A[s]
            binit(opus::number<1 + 2 * T::kNumSlots + s>{}, kFreeMemCnt); // FREE_B[s]
        });
    }
    __builtin_amdgcn_s_barrier();

    // ---- Producers: w0 fills A slots, w1 fills B slots (kNumSlots ring). ----
    if (is_producer) {
        const int gk0 = k_step_beg * T::kBlockK;
        const uint32_t k_extent = (uint32_t)(kargs.k - gk0);
        constexpr int slot_a_b = T::kSlotBytesA;
        constexpr int slot_b_b = T::kSlotBytesB;
        constexpr auto KStep = opus::number<T::kBlockK>{};

        // TDM row extent = remaining valid rows (clamps OOB global reads on the
        // last M/N tile); padded LDS tail rows are never read by the reduce kernel.
        const int row_extent_a = kargs.m - tile_row;
        const int row_extent_b = kargs.n - tile_col;

        // Producer (per wave, A on w0 / B on w1). Steps stream into a kNumSlots
        // ring (slot = step % kNumSlots; LDS delta +slot_bytes, wrap on slot 0).
        // FreeBaseN selects this producer's own FREE barrier set: FREE_A (1+P) for
        // w0, FREE_B (1+2P) for w1. Each producer only ever waits on its own set.
        auto produce = [&](auto& w, int slot_bytes, auto FreeBaseN) __attribute__((always_inline)) {
            constexpr int kFreeBase = FreeBaseN.value;
            int loaded = 0;
            auto load_next = [&]() __attribute__((always_inline)) {
                if (loaded > 0) {
                    const int delta = (loaded % T::kNumSlots == 0)
                        ? -(T::kNumSlots - 1) * slot_bytes : slot_bytes;
                    w.move(KStep, 0_I, 0_I, 0_I, 0_I, delta);
                }
                w.load_to_lds();
                ++loaded;
            };
            // Steady-state slot step (compile-time slot & barrier ids; runtime K).
            // RUN-AHEAD with a 2-deep TDM overlap + LAGGED DATA signal: issue this
            // step's load for slot s (after FREE[s]), then s_wait_tensorcnt(2) leaves
            // TWO loads in flight (this step's and the previous step's) overlapping the
            // consumer's WMMA, while guaranteeing the load from 2 steps ago (slot
            // prev2 = s-2 mod P) has landed -> signal DATA[prev2]. Every DATA[] still
            // fires only after its slot is fully in LDS (no RAW race). With P=kNumSlots
            // =3 the ring is exactly: 2 slots being written + 1 being read. The last
            // two steps' DATA are signalled by the epilogue drain below.
            auto step_slot = [&](auto sN) __attribute__((always_inline)) {
                constexpr int s = decltype(sN)::value;
                constexpr int prev2 = (s - 2 + T::kNumSlots) % T::kNumSlots;
                bjsw(opus::number<kFreeBase + s>{});              // FREE_{A,B}[s] (wait)
                load_next();                                      // issue load for slot s
                __builtin_amdgcn_s_wait_tensorcnt(2);             // leave 2 in flight; prev2 landed
                bjs(opus::number<1 + prev2>{});                   // DATA[prev2] (signal, landed)
            };
            // Prologue: issue nload = min(kNumSlots, k_steps) TDMs, drain them, then
            // signal DATA[0..nload-1] WITHOUT waiting -> producer starts nload slots
            // ahead. The k_steps >= kNumSlots case (essentially always) is split out
            // so nload == kNumSlots is COMPILE-TIME: the prologue load loop becomes a
            // fully-unrolled static_for and every DATA[s] signal fires unconditionally
            // (no `for(p<nload)` back-edge, no per-slot `if (s<nload)` branch). The
            // rare small-K path keeps the conditional form (over-signalling DATA[s]
            // for slots the consumer never waits on would desync the barrier memcnt).
            if (k_steps >= T::kNumSlots) {
                // Prologue: issue all kNumSlots loads up front (peak in-flight = P),
                // then signal DATA[0..P-3] as each lands (decreasing tensorcnt wait),
                // leaving the LAST TWO prologue loads (slots P-2, P-1) in flight ->
                // their DATA is pending (signalled lagged by the first steady steps).
                // This starts the producer P slots ahead with 2 TDMs already overlapping.
                opus::static_for<T::kNumSlots>([&](auto) __attribute__((always_inline)) {
                    load_next();
                });
                opus::static_for<T::kNumSlots - 2>([&](auto jN) __attribute__((always_inline)) {
                    constexpr int j = decltype(jN)::value;
                    __builtin_amdgcn_s_wait_tensorcnt(T::kNumSlots - 1 - j); // load j landed
                    bjs(opus::number<1 + j>{});                              // DATA[j] (signal)
                });
                // Steady state: full-group main loop + once-run tail. step_slot signals
                // the LAGGED DATA (slot s-2), keeping 2 TDMs in flight per step.
                int k = T::kNumSlots;
                for (; k + T::kNumSlots <= k_steps; k += T::kNumSlots)
                    opus::static_for<T::kNumSlots>(step_slot);
                const int rem = k_steps - k;   // 0..kNumSlots-1
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    if ((int)decltype(sN)::value < rem) step_slot(sN);
                });
                // Epilogue: drain the final TWO in-flight loads and signal their (still
                // pending) DATA -- the last two steps (k_steps-2, k_steps-1) were loaded
                // but step_slot only signalled up to 2 steps back. Drain to 0 (both
                // landed), then signal them in consume order. Runtime-select the
                // compile-time barrier ids.
                __builtin_amdgcn_s_wait_tensorcnt(0);
                const int last2_slot = (k_steps - 2) % T::kNumSlots;
                const int last_slot  = (k_steps - 1) % T::kNumSlots;
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    if ((int)decltype(sN)::value == last2_slot)
                        bjs(opus::number<1 + decltype(sN)::value>{});      // DATA[k_steps-2]
                });
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    if ((int)decltype(sN)::value == last_slot)
                        bjs(opus::number<1 + decltype(sN)::value>{});      // DATA[k_steps-1]
                });
            } else {
                // Rare: k_steps in [1, kNumSlots-1] -> prologue only (no steady loop).
                const int nload = k_steps;
                for (int p = 0; p < nload; ++p) load_next();
                __builtin_amdgcn_s_wait_tensorcnt(0);
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    if ((int)decltype(sN)::value < nload)
                        bjs(opus::number<1 + decltype(sN)::value>{});   // DATA[s] (signal)
                });
            }
        };  // produce

        if (wave_id == 0) {
            WindowA w;
            w.make(reinterpret_cast<uintptr_t>(smem_a), kargs.ptr_a, 0,
                   k_extent, (uint32_t)row_extent_a, (uint64_t)stride_a,
                   (uint32_t)gk0, (uint32_t)tile_row);
            w.desc.set_workgroup_mask(mask_a);   // plain grid (no cluster): mask_a=0 -> multicast off
            produce(w, slot_a_b, opus::number<1 + T::kNumSlots>{});       // FREE_A
        } else {  // wave_id == 1 -> B
            WindowB w;
            w.make(reinterpret_cast<uintptr_t>(smem_b), kargs.ptr_b, 0,
                   k_extent, (uint32_t)row_extent_b, (uint64_t)stride_b,
                   (uint32_t)gk0, (uint32_t)tile_col);
            w.desc.set_workgroup_mask(mask_b);   // plain grid (no cluster): mask_b=0 -> multicast off
            produce(w, slot_b_b, opus::number<1 + 2 * T::kNumSlots>{});   // FREE_B
        }
        // Producer epilogue: rendezvous with the consumers at a workgroup barrier
        // before exiting, so no wave leaves while the per-slot named-barrier
        // handshakes are still in flight (SPG p.83: avoid "signal then exit").
        __builtin_amdgcn_s_barrier();
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

    // WMMA source regs: 3-deep ring so a round's ds_load never overwrites VGPRs a
    // still-running (multi-cycle) WMMA of a recent round reads (WMMA-source WAR,
    // MI400 SPG 4.6.12.1).
    typename decltype(mma)::vtype_a v_a[3];
    typename decltype(mma)::vtype_b v_b[3];
    typename decltype(mma)::vtype_c reg_c;
    clear(reg_c);

    // Per K-step consumer (slot s = compile-time). bjsw(DATA[s]) waits until both
    // producers have signalled the slot ready (A and B landed in LDS); after the
    // ds-reads + WMMA, bjs(FREE[s]) signals the slot is reusable (no wait ->
    // run-ahead). The last ds half's s_wait_dscnt(0) drains the slot's LDS into
    // VGPRs before FREE[s], so the producer's reload never races the ds-reads.
    auto consume_slot = [&](auto Sn, auto AFirstN) __attribute__((always_inline)) {
        constexpr int s = Sn.value;
        constexpr bool AFirst = AFirstN.value;
        bjsw(opus::number<1 + s>{});                // DATA[s] rendezvous (wait)
        asm volatile("" ::: "memory");              // pin ds-reads in the critical region
        // ds-read overlap policy (kSchedDsCount ds ops per half/round):
        // overlap the next round's ds with the current WMMA only when two rounds
        // fit the 6-bit DScnt budget (2*DS <= 56); else fall back to per-round
        // drain. The 3-deep ring gives the WAR cushion for the overlap path.
        constexpr int kDsPerRound = T::kSchedDsCount;
        constexpr bool kDsOverlap = (2 * kDsPerRound <= 56);
        // sched_barrier(0) walls keep ds_reads from being hoisted so the dscnt
        // accounting stays exact per round.
        auto do_load = [&](int half, int buf) __attribute__((always_inline)) {
            auto sa = make_smem(smem_a + s*slot_a + half*T::kKHalfElems);
            auto sb = make_smem(smem_b + s*slot_b + half*T::kKHalfElems);
            if constexpr (AFirst) { v_a[buf] = load<T::kVecA>(sa, u_ra); v_b[buf] = load<T::kVecB>(sb, u_rb); }
            else                  { v_b[buf] = load<T::kVecB>(sb, u_rb); v_a[buf] = load<T::kVecA>(sa, u_ra); }
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (!kDsOverlap) {        // per-round drain (fallback)
                opus::s_wait_dscnt(opus::number<0>{});
                __builtin_amdgcn_sched_barrier(0);
            }
        };
        __builtin_amdgcn_sched_barrier(0);
        do_load(0, 0);
        opus::static_for<T::kHalvesPerSlot>([&](auto i_c) __attribute__((always_inline)) {
            constexpr int i   = decltype(i_c)::value;
            constexpr int cur = i % 3;
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (i + 1 < T::kHalvesPerSlot) do_load(i + 1, (i + 1) % 3);  // prefetch next half
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (kDsOverlap) {
                // drain current round, keep the just-issued next round in flight.
                constexpr int rem = (i + 1 < T::kHalvesPerSlot) ? kDsPerRound : 0;
                opus::s_wait_dscnt(opus::number<rem>{});
            }
            reg_c = mma(v_a[cur], v_b[cur], reg_c);
        });
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("" ::: "memory");
        // Free slot s for BOTH producers: signal FREE_A[s] and FREE_B[s] (run-ahead,
        // no wait). Each producer waits only on its own set, so its reuse-gate can
        // never be completed without its own signal -> no missed-broadcast desync.
        bjs(opus::number<1 + T::kNumSlots + s>{});      // FREE_A[s] signal
        bjs(opus::number<1 + 2 * T::kNumSlots + s>{});  // FREE_B[s] signal
    };
    // Slots consumed in compile-time order 0..kNumSlots-1 per unrolled group;
    // the K%kNumSlots tail consumes slots 0..rem-1. This exactly mirrors the
    // producer's DATA[s]/FREE[s] signal counts (per-slot symmetry -> no hang).
    auto run = [&](auto AFirstN) __attribute__((always_inline)) {
        int k = 0;
        for (; k + T::kNumSlots <= k_steps; k += T::kNumSlots)
            opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                consume_slot(sN, AFirstN);
            });
        const int rem = k_steps - k;   // 0..kNumSlots-1
        opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
            if ((int)decltype(sN)::value < rem) consume_slot(sN, AFirstN);
        });
    };
    if (wave_split == 0) run(opus::true_type{});
    else                 run(opus::false_type{});

    // ---- Plain store the fp32 partial into ws[split_idx][padded_m][padded_n]. ----
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
    __builtin_amdgcn_s_barrier();
    store<kCVec>(g_ws, reg_c, u_gc, 0);

    // Consumer epilogue: rendezvous with the producers (matches the producer's
    // workgroup barrier above) so all 4 waves of the WG exit together.
#else
    (void)kargs;   // non-gfx1250 device pass: empty stub (multi-arch wheel safety)
#endif // __gfx1250__
#endif // __HIP_DEVICE_COMPILE__
}

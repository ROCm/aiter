// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// GPU kernels for generalized top-k paths. Include AFTER block_select_lds / block_gather_topk
// are visible in the translation unit.

template <bool RAGGED, bool WRITE_VALUES, bool NAN_HIGH = false>
__global__ void phase_small_n_topk(const float* __restrict__ input,
                                   int pitch,
                                   RowExtents<RAGGED> extents,
                                   int K,
                                   TopkOut<WRITE_VALUES> dst,
                                   int npasses)
{
    // topk_select, the one NAN_HIGH caller, gathers its values itself.
    static_assert(!(NAN_HIGH && WRITE_VALUES), "NAN_HIGH is instantiated for indices only");
#if TOPK_SAMPLED_DEVICE
    const int row       = blockIdx.x;
    const int row_start = RAGGED ? extents.row_start(row, pitch) : 0;
    int len;
    if constexpr(RAGGED)
        len = extents.row_len(row, pitch);
    else
        len = pitch;
    const float* ri = input + (size_t)row * pitch + row_start;
    float* val      = dst.val_row(row, K);
    if(RAGGED && len <= K)
    {
        emit_identity_row<WRITE_VALUES>(dst.idx_row(row, K), val, ri, row_start, len, K);
        return;
    }
    extern __shared__ uint32_t s_keys[];
    __shared__ __align__(16) uint32_t s_hist[HIST_SLOTS];
    __shared__ uint32_t s_scan[2];
    __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];
    __shared__ unsigned s_wgt, s_weq;

    const int n4 = RAGGED ? n4_cover(len) : (pitch / FP32_EPT);
    for(int u = threadIdx.x; u < n4; u += blockDim.x)
    {
        vfloat4 v        = load_row_f4<RAGGED>(ri, u, len);
        const int base   = u * FP32_EPT;
        s_keys[base + 0] = sortable_key<NAN_HIGH>(v[0]);
        s_keys[base + 1] = sortable_key<NAN_HIGH>(v[1]);
        s_keys[base + 2] = sortable_key<NAN_HIGH>(v[2]);
        s_keys[base + 3] = sortable_key<NAN_HIGH>(v[3]);
    }
    __syncthreads();

    const int k_out = RAGGED ? k_take_dev(K, len) : K;
    uint32_t pivot;
    int eq_needed;
    block_select_lds(s_keys, len, k_out, s_hist, s_scan, s_mm, pivot, eq_needed, npasses, true);

    int* out = dst.idx_row(row, K);
    if(threadIdx.x == 0)
    {
        s_wgt = 0;
        s_weq = 0;
    }
    __syncthreads();
    if constexpr(RAGGED)
    {
        block_gather_topk<WRITE_VALUES>(
            len,
            pivot,
            k_out - eq_needed,
            eq_needed,
            out,
            val,
            &s_wgt,
            &s_weq,
            [&](int i) { return s_keys[i]; },
            [&](int i) { return row_start + i; });
    }
    else
    {
        block_gather_topk<WRITE_VALUES>(
            len,
            pivot,
            k_out - eq_needed,
            eq_needed,
            out,
            val,
            &s_wgt,
            &s_weq,
            [&](int i) { return s_keys[i]; },
            [](int i) { return i; });
    }
    if(RAGGED && k_out < K)
    {
        __syncthreads();
        pad_topk_tail<WRITE_VALUES>(out, val, k_out, K);
    }
#else
    __builtin_trap();
#endif
}

template <bool RAGGED, bool NT, bool NT_STORE = NT, int PB_B = 0>
__global__ void phase_b_filter_coop(const float* __restrict__ input,
                                    int pitch,
                                    RowExtents<RAGGED> extents,
                                    int n4_per_row,
                                    int g_log2,
                                    const float* __restrict__ threshold_f,
                                    uint64_t* __restrict__ cand_pack,
                                    unsigned int* __restrict__ cand_reserved,
                                    unsigned int* __restrict__ cand_bad,
                                    int cap)
{
#if TOPK_SAMPLED_DEVICE
    const int row     = blockIdx.y;
    const int len     = row_len_of<RAGGED>(row, pitch, extents);
    const float* ri   = input + (size_t)row * pitch + (RAGGED ? extents.row_start(row, pitch) : 0);
    const float th    = threshold_f[row];
    const int lane    = threadIdx.x & (WAVE_SIZE - 1);
    const int wid     = threadIdx.x / WAVE_SIZE;
    const int nwaves  = PB_BLOCK / WAVE_SIZE;
    const uint64_t lt = (1ull << lane) - 1ull;

    // Dynamic, not static: the staging buffer needs WSTAGE_CAP_COOP entries per wave,
    // so a 1024-thread block wants sixteen waves' worth and a 512-thread block
    // eight. As a compile-time constant that has to be the larger of the two, and
    // then every 512-thread block reserves 40 KB it cannot use -- measured at 7 to
    // 14% (m=4096 n=131072 goes 540.1 to 615.2us with sixteen waves' staging).
    // Sized at launch instead, each block reserves exactly what its width needs.
    extern __shared__ uint64_t wbuf[];
    uint64_t* buf      = wbuf + (size_t)wid * WSTAGE_CAP_COOP;
    uint64_t* row_base = cand_pack + (size_t)row * cap;
    (void)nwaves;

    // Launched with PB_BLOCK threads and a power-of-two G = 1 << g_log2 blocks per
    // row, so neither needs the hidden dispatch arguments or a division.
    const int G   = 1 << g_log2;
    const int bid = blockIdx.x;
    // Full vectors only: the 1-3 columns past the last one are picked up by the
    // row's last block after the loop, so no load or ballot needs a column bound.
    const int n4       = RAGGED ? len / FP32_EPT : n4_per_row;
    const int chunk_n4 = (n4 + G - 1) >> g_log2;
    const int i0       = bid * chunk_n4;
    const int i1       = min(i0 + chunk_n4, n4);
    const int stride   = PB_BLOCK;
    const int iters    = (i1 > i0) ? ((i1 - i0) + stride - 1) / stride : 0;

    int bcnt                    = 0;
    constexpr int COOP_DRAIN_AT = WSTAGE_CAP_COOP - 4 * WAVE_SIZE;
    // Sticky per wave. Once a drain finds the row's candidate area full the row is
    // marked bad and phase_c falls back for it, so nothing this wave stages after
    // that is ever read. Staging still runs -- bcnt restarts from 0, so every
    // slot it writes stays inside the wave's buffer -- but drains only reset it,
    // and the epilogue reports the wave as overflowed.
    bool wave_bad = false;

// The drain's cost is its reservation atomic, not its data movement: with no
// candidate passing, phase_b ran 100.35 us faster at m=4096 n=131072, while the
// compaction arithmetic, its LDS write and the drain's LDS read and global write
// each priced free. cand_reserved[row] is one address per row that every wave
// of the row's coop_g blocks contends for.
#define COOP_RESERVE(row, n) atomicAdd(&cand_reserved[(size_t)(row) * CTR_STRIDE], (unsigned)(n))
#define COOP_DRAIN_BODY(off)                       \
    for(int _j = lane; _j < bcnt; _j += WAVE_SIZE) \
    row_base[(off) + _j] = buf[_j]

#define COOP_DRAIN_WAVE()                                                \
    do                                                                   \
    {                                                                    \
        if(!wave_bad)                                                    \
        {                                                                \
            unsigned _off = 0;                                           \
            if(lane == 0)                                                \
                _off = COOP_RESERVE(row, bcnt);                          \
            _off = (unsigned)__shfl((int)_off, 0);                       \
            if(_off + (unsigned)bcnt > (unsigned)cap)                    \
            {                                                            \
                if(lane == 0)                                            \
                    atomicExch(&cand_bad[(size_t)row * CTR_STRIDE], 1u); \
                wave_bad = true;                                         \
            }                                                            \
            else                                                         \
            {                                                            \
                COOP_DRAIN_BODY(_off);                                   \
            }                                                            \
        }                                                                \
        bcnt = 0;                                                        \
    } while(0)

    if constexpr(PB_B > 0)
    {
        // With at most two blocks per CU little else hides a filter load, so each
        // thread issues PB_B before filtering any, in the loop's filter order; a
        // slot past i1 loads a live address and is masked by `live`.
        auto filter_v4 = [&](const vfloat4& v, int i, bool live) {
            const int base_idx = i * FP32_EPT;
            const uint64_t b0  = __ballot(live && !(v[0] < th));
            const uint64_t b1  = __ballot(live && !(v[1] < th));
            const uint64_t b2  = __ballot(live && !(v[2] < th));
            const uint64_t b3  = __ballot(live && !(v[3] < th));
            const int t0       = __popcll(b0);
            const int t1       = t0 + __popcll(b1);
            const int t2       = t1 + __popcll(b2);
            const int wtotal   = t2 + __popcll(b3);
            if(wtotal > 0)
            {
                if(b0 & (1ull << lane))
                    buf[bcnt + __popcll(b0 & lt)] =
                        ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
                if(b1 & (1ull << lane))
                    buf[bcnt + t0 + __popcll(b1 & lt)] =
                        ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
                if(b2 & (1ull << lane))
                    buf[bcnt + t1 + __popcll(b2 & lt)] =
                        ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
                if(b3 & (1ull << lane))
                    buf[bcnt + t2 + __popcll(b3 & lt)] =
                        ((uint64_t)__float_as_uint(v[3]) << 32) | (uint32_t)(base_idx + 3);
                bcnt += wtotal;
            }
            if(bcnt > COOP_DRAIN_AT)
            {
                __builtin_amdgcn_wave_barrier();
                COOP_DRAIN_WAVE();
            }
        };
        for(int it0 = 0; it0 < iters; it0 += PB_B)
        {
            vfloat4 vv[PB_B];
#pragma unroll
            for(int t = 0; t < PB_B; t++)
                vv[t] = load_row_f4<false, NT>(
                    ri, min(i0 + (it0 + t) * stride + (int)threadIdx.x, i1 - 1), len);
#pragma unroll
            for(int t = 0; t < PB_B; t++)
            {
                const int i = i0 + (it0 + t) * stride + threadIdx.x;
                filter_v4(vv[t], i, i < i1);
            }
        }
    }
    else
    {
        for(int it = 0; it < iters; it++)
        {
            const int i     = i0 + it * stride + threadIdx.x;
            vfloat4 v       = {0.f, 0.f, 0.f, 0.f};
            const bool live = (i < i1);
            if(live)
                v = load_row_f4<false, NT>(ri, i, len);
            const int base_idx = i * FP32_EPT;
            const uint64_t b0  = __ballot(live && !(v[0] < th));
            const uint64_t b1  = __ballot(live && !(v[1] < th));
            const uint64_t b2  = __ballot(live && !(v[2] < th));
            const uint64_t b3  = __ballot(live && !(v[3] < th));
            const int t0       = __popcll(b0);
            const int t1       = t0 + __popcll(b1);
            const int t2       = t1 + __popcll(b2);
            const int wtotal   = t2 + __popcll(b3);
            if(wtotal > 0)
            {
                if(b0 & (1ull << lane))
                    buf[bcnt + __popcll(b0 & lt)] =
                        ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
                if(b1 & (1ull << lane))
                    buf[bcnt + t0 + __popcll(b1 & lt)] =
                        ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
                if(b2 & (1ull << lane))
                    buf[bcnt + t1 + __popcll(b2 & lt)] =
                        ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
                if(b3 & (1ull << lane))
                    buf[bcnt + t2 + __popcll(b3 & lt)] =
                        ((uint64_t)__float_as_uint(v[3]) << 32) | (uint32_t)(base_idx + 3);
                bcnt += wtotal;
            }
            if(bcnt > COOP_DRAIN_AT)
            {
                __builtin_amdgcn_wave_barrier();
                COOP_DRAIN_WAVE();
            }
        }
    }
    // The vector loop stops at n4 = len / FP32_EPT, which TRUNCATES when the row
    // width is not a multiple of four, so the last one to three columns would
    // never be looked at. A per-element `< len` predicate on every vector (the
    // earlier ragged form) costs the whole row to reach three columns.
    //
    // The block that owns the last chunk picks the tail up instead: at most three
    // columns, one lane each, staged exactly like any other candidate so the
    // epilogue needs no special case. bcnt is at most WSTAGE_CAP_COOP - 4 * WAVE_SIZE
    // here because the drain check runs at the end of every iteration, so the
    // three extra entries cannot overflow the staging buffer.
    //
    // Measured before this existed: at m=4 N=131075 the answer was exactly
    // torch.topk over the first 131072 columns, with the 2049th largest value
    // standing in for the one that fell in the tail.
    {
        const int tail0 = n4 * FP32_EPT;
        const int ncols = len - tail0;
        if(ncols > 0 && bid == G - 1 && !wave_bad)
        {
            const bool mine   = (int)threadIdx.x < ncols;
            const float vv    = mine ? ri[tail0 + (int)threadIdx.x] : 0.f;
            const bool act    = mine && !(vv < th);
            const uint64_t bt = __ballot(act);
            const int t       = __popcll(bt);
            if(t > 0)
            {
                if(act)
                    buf[bcnt + __popcll(bt & lt)] = ((uint64_t)__float_as_uint(vv) << 32) |
                                                    (uint32_t)(tail0 + (int)threadIdx.x);
                bcnt += t;
            }
        }
    }
#undef COOP_DRAIN_WAVE

    // Block epilogue: one reservation for the block's staged total, then each wave
    // writes its own run. A block with nothing staged (total == 0) returns at the
    // guard below without reserving.
    __shared__ int s_local[MAX_WAVES_PER_BLOCK];
    __shared__ int s_off[MAX_WAVES_PER_BLOCK];
    __shared__ unsigned s_base;
    if(lane == 0)
        s_local[wid] = wave_bad ? -1 : bcnt;
    __syncthreads();
    if(threadIdx.x == 0)
    {
        int total = 0;
        bool bad  = false;
        for(int w = 0; w < nwaves; w++)
        {
            if(s_local[w] < 0)
                bad = true;
            else
            {
                s_off[w] = total;
                total += s_local[w];
            }
        }
        if(bad || total > cap)
        {
            if(bad || total > cap)
                atomicExch(&cand_bad[(size_t)row * CTR_STRIDE], 1u);
            s_base = 0xFFFFFFFFu;
        }
        else if(total == 0)
        {
            s_base = 0xFFFFFFFEu;
        }
        else
        {
            s_base = atomicAdd(&cand_reserved[(size_t)row * CTR_STRIDE], (unsigned)total);
            if(s_base + (unsigned)total > (unsigned)cap)
            {
                atomicExch(&cand_bad[(size_t)row * CTR_STRIDE], 1u);
                s_base = 0xFFFFFFFFu;
            }
        }
    }
    __syncthreads();
    if(s_base == 0xFFFFFFFFu || s_base == 0xFFFFFFFEu)
        return;

    // Each wave writes out its own staged run, with a non-temporal store.
    //
    // Both halves were priced against the block-serial walk this replaces, at
    // m=4096 k=2048, gaussian rows, seed 0, phase_b device
    // time, upper three quartiles of 20 launches:
    //
    //                                          N=131072   N=262144
    //   per-wave copy, ordinary store           -12.8us     -8.2us
    //   block-wide walk, non-temporal store     -12.1us     -8.4us
    //   both, which is this                     -22.6us    -21.3us
    //
    // against a run-to-run spread of 1.4us and 3.8us on the unchanged kernel.
    //
    // The serial half is the obvious one: s_off is a prefix sum, so the eight runs
    // are already one contiguous block run, and walking it per wave lets the eight
    // go out at once instead of one after another.
    //
    // The non-temporal half is there because of what the copy turned out to cost.
    // N=131072 and N=262144 plan the same margin, cap and coop_g, so the epilogue
    // writes the same 93.9MB at both -- and it measured 98.3us at one and 153.1us
    // at the other. It is not paying for its own bytes, it is paying to interleave
    // with the read stream, and the bill scales with the reads it interrupts.
    // Bypassing the caches stops the candidate run evicting row data the other
    // blocks are still reading. Three other levers were priced and measured
    // nothing.
    {
        const int cnt = s_local[wid];
        if(cnt > 0)
        {
            uint64_t* dst = row_base + s_base + s_off[wid];
            // The store takes the same M*pitch threshold as the load, for the same reason.
            // Measured three-kernel total, per-wave with an ordinary store against
            // per-wave with a non-temporal one, k=2048, gaussian rows, seed 0:
            //
            //   m=1    n=131072  (2^17)   18.49us   19.72us
            //   m=16   n=1048576 (2^24)   42.02us   43.45us
            //   m=64   n=131072  (2^23)   31.39us   31.75us
            //   m=128  n=262144  (2^25)   46.52us   46.68us
            //   m=1024 n=131072  (2^27)  151.85us  150.72us
            //   m=4096 n=131072  (2^29)  571.96us  552.59us
            //   m=4096 n=1048576 (2^32) 2802.07us 2761.22us
            //
            // It crosses at the same 2^27 the loads do. Writing each wave's own run
            // rather than having the block walk all eight is a win at every size, so
            // only the non-temporal part is conditional.
            if constexpr(NT_STORE)
            {
                for(int j = lane; j < cnt; j += WAVE_SIZE)
                    __builtin_nontemporal_store(buf[j], &dst[j]);
            }
            else
            {
                for(int j = lane; j < cnt; j += WAVE_SIZE)
                {
                    dst[j] = buf[j];
                }
            }
        }
    }
#else
    __builtin_trap();
#endif
}

template <bool RAGGED, bool WRITE_VALUES, bool REUSE_WIDE = false, bool NAN_HIGH = false>
__global__ PHASE_C_OCCUPANCY void
phase_c_select_contig(const float* __restrict__ input,
                      int pitch,
                      RowExtents<RAGGED> extents,
                      const uint64_t* __restrict__ cand_pack,
                      const unsigned int* __restrict__ cand_reserved,
                      const unsigned int* __restrict__ cand_bad,
                      int cap,
                      int K,
                      TopkOut<WRITE_VALUES> dst,
                      int npasses,
                      bool keys_only,
                      int nwide,
                      const uint32_t* __restrict__ threshold)
{
    // topk_select, the one NAN_HIGH caller, gathers its values itself.
    static_assert(!(NAN_HIGH && WRITE_VALUES), "NAN_HIGH is instantiated for indices only");
#if TOPK_SAMPLED_DEVICE
    const int row       = blockIdx.x;
    const int row_start = RAGGED ? extents.row_start(row, pitch) : 0;
    const int len       = row_len_of<RAGGED>(row, pitch, extents);
    // Both counters unconditionally: the ternary compiled to load, wait, branch,
    // load, a second serial round trip in front of every phase_c block.
    const unsigned int bad_raw = cand_bad[(size_t)row * CTR_STRIDE];
    const unsigned int res_raw = cand_reserved[(size_t)row * CTR_STRIDE];
    const unsigned int c_raw   = bad_raw ? 0xFFFFFFFFu : res_raw;

    extern __shared__ uint32_t s_dyn[];
    uint32_t* s_keys_ext = s_dyn;
    int* s_idx           = reinterpret_cast<int*>(s_dyn + cap);
    // nwide * WIDE_WORDS words after the keys (and the indices unless keys_only).
    uint32_t* s_wide = s_dyn + (keys_only ? cap : 2 * cap);
    __shared__ __align__(16) uint32_t s_hist[HIST_SLOTS];
    __shared__ uint32_t s_scan[2];
    __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];
    __shared__ unsigned s_wgt, s_weq;

    int* out             = dst.idx_row(row, K);
    float* val           = dst.val_row(row, K);
    const int k_out      = RAGGED ? k_take_dev(K, len) : K;
    const uint64_t* base = cand_pack + (size_t)row * cap;

    // ATT puts s_waitcnt vmcnt(0) at 20.8% of phase_c's traced latency here, but
    // unrolling this loop is worth nothing: measured at depth 1, 2, 4 and 8,
    // phase_c reads 14.24, 14.27, 14.40 and 14.31us at m=512 n=131072. The trace
    // shows four separate load sites already, and c is about 2867 against a
    // 1024-thread block, so there are three iterations and nothing left to
    // overlap. The wait is the latency of the read itself.
    // Count pass 0's digits while the candidates are being read, the same trade
    // phase_a's sampler takes. Pass 1 of phase_c's select is the expensive one --
    // the only unfiltered scan of all c keys, measured at 4.38us against 1.62 /
    // 1.16 / 1.12 for passes 2, 3 and 4 at m=512 n=131072 -- and the keys are
    // already in registers here.
    const int fold_rep = threadIdx.x & (HIST_REP - 1);
    auto take_cand     = [&](uint64_t p, int i) {
        const uint32_t kk = sortable_key_bits<NAN_HIGH>((uint32_t)(p >> 32));
        s_keys_ext[i]     = kk;
        if(!keys_only)
            s_idx[i] = (int)(uint32_t)p;
        // radix_shift(0) is 24, so pass 0's digit is the top byte.
        atomicAdd(&s_hist[(kk >> 24) * HIST_REP + fold_rep], 1u);
    };
    auto read_cands = [&](int c) {
        clear_hist(s_hist);
        __syncthreads();
        // Published by the barrier after the read. Folding the first wide pass's
        // digits into this read as well, the way pass 0's are, was measured and is
        // SLOWER: phase_c +0.2 to +1.0us over the unfolded wide select at m=64..512,
        // against -0.3 to -0.4us unfolded.
        if(nwide > 0)
            clear_wide(s_wide, wide_buffer_count(nwide, REUSE_WIDE));
        // PC_B candidate loads per thread go out before the first is used; with one
        // block per CU nothing else hides their round trips.
        constexpr int PC_B = 4;
        for(int i0 = threadIdx.x; i0 < c; i0 += PC_B * (int)blockDim.x)
        {
            uint64_t pv[PC_B];
#pragma unroll
            for(int u = 0; u < PC_B; u++)
            {
                const int i = i0 + u * (int)blockDim.x;
                pv[u]       = i < c ? base[i] : 0ull;
            }
#pragma unroll
            for(int u = 0; u < PC_B; u++)
            {
                const int i = i0 + u * (int)blockDim.x;
                if(i < c)
                    take_cand(pv[u], i);
            }
        }
        __syncthreads();
    };

    // len <= K routes unconditionally so the identity emit cannot be diverted by
    // a candidate count the +inf threshold let through.
    int c          = (int)c_raw;
    bool fall_back = (RAGGED && len <= K) || c_raw < (unsigned)k_out || c_raw > (unsigned)cap;
    if(!fall_back)
    {
        read_cands(c);
        // phase_b's `!(v < th)` keeps a NaN of either sign, and a negative one ranks
        // below -INF, so NaNs can pad a row whose candidates at or above the
        // threshold fell short of k_out past the undershoot test above. Pass 0's
        // bucket 0 holds every key below 0x01000000: the negative NaNs, -INF and
        // anything at or below -2^127. If the k-th key is down there the row goes to
        // the whole-row select. No per-candidate work: the bucket was counted by the
        // read, and checking each candidate in phase_b's drain instead cost phase_b
        // 2 to 8% at m <= 256. The fallback comes after this read rather than in a
        // loop around it, which would keep its inputs live through the select.
        unsigned low = 0u;
        for(int r = 0; r < HIST_REP; r++)
            low += s_hist[r];
        if(c - (int)low < k_out)
        {
            fall_back = true;
            __syncthreads(); // everyone has read bucket 0 before the fallback clears s_hist
        }
    }
    if(fall_back)
    {
        const float* rif0 = input + (size_t)row * pitch + row_start;
        if(RAGGED && len <= K)
        {
            emit_identity_row<WRITE_VALUES>(out, val, rif0, row_start, len, K);
            return;
        }
        uint64_t* cand_w = const_cast<uint64_t*>(cand_pack) + (size_t)row * cap;
        // An overflowed row's answer lies at or above phase_a's threshold, where only
        // a few thousand of its keys are.
        const uint32_t tmin = c_raw > (unsigned)cap ? threshold[row] : 0u;
        // Same loader choice as exact_row_select: a plain row whose width is not a
        // multiple of FP32_EPT is read bounds-checked, or its last columns are lost.
        int got;
        if(!RAGGED && (len % FP32_EPT) != 0)
            got = radix_fallback_row<true, WRITE_VALUES, NAN_HIGH>(rif0,
                                                                   n4_cover(len),
                                                                   len,
                                                                   row_start,
                                                                   k_out,
                                                                   cap,
                                                                   tmin,
                                                                   cand_w,
                                                                   out,
                                                                   val,
                                                                   s_hist,
                                                                   s_dyn,
                                                                   s_scan,
                                                                   s_mm,
                                                                   &s_wgt,
                                                                   &s_weq);
        else
            got = radix_fallback_row<RAGGED, WRITE_VALUES, NAN_HIGH>(rif0,
                                                                     RAGGED ? n4_cover(len)
                                                                            : pitch / FP32_EPT,
                                                                     len,
                                                                     row_start,
                                                                     k_out,
                                                                     cap,
                                                                     tmin,
                                                                     cand_w,
                                                                     out,
                                                                     val,
                                                                     s_hist,
                                                                     s_dyn,
                                                                     s_scan,
                                                                     s_mm,
                                                                     &s_wgt,
                                                                     &s_weq);
        if(got < 0)
            return; // emitted; len > K here, so k_out == K and nothing to pad
        c = got;
        read_cands(c);
    }

    uint32_t pivot;
    int eq_needed;
    if(nwide > 0)
        block_select_lds_wide<REUSE_WIDE>(s_keys_ext,
                                          c,
                                          k_out,
                                          s_hist,
                                          s_wide,
                                          s_scan,
                                          s_mm,
                                          pivot,
                                          eq_needed,
                                          nwide,
                                          false,
                                          true);
    else
        block_select_lds(
            s_keys_ext, c, k_out, s_hist, s_scan, s_mm, pivot, eq_needed, npasses, true, true);

    if(threadIdx.x == 0)
    {
        s_wgt = 0;
        s_weq = 0;
    }
    __syncthreads();

    if(keys_only)
    {
        block_gather_topk<WRITE_VALUES>(
            c,
            pivot,
            k_out - eq_needed,
            eq_needed,
            out,
            val,
            &s_wgt,
            &s_weq,
            [&](int i) { return s_keys_ext[i]; },
            [&](int i) { return row_start + (int)(uint32_t)(base[i] & 0xFFFFFFFFull); });
    }
    else
    {
        block_gather_topk<WRITE_VALUES>(
            c,
            pivot,
            k_out - eq_needed,
            eq_needed,
            out,
            val,
            &s_wgt,
            &s_weq,
            [&](int i) { return s_keys_ext[i]; },
            [&](int i) { return row_start + s_idx[i]; });
    }
    if(RAGGED && k_out < K)
    {
        __syncthreads();
        pad_topk_tail<WRITE_VALUES>(out, val, k_out, K);
    }
#else
    __builtin_trap();
#endif
}

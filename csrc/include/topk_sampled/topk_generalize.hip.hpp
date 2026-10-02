// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// GENERATED FILE -- DO NOT EDIT.
//
// Source of truth: the topk-prefill-avo repo. Regenerate with
//   python3 scripts/export_aiter_op.py --aiter <this aiter checkout>
// and verify an existing tree with the same command plus --check.
//
// This is benchmark_topk.hip.cpp up to its AITER_EXPORT_END marker (the kernels
// and their dispatch) followed by csrc/topk_aiter_entry.inc.hip (the aiter op
// entry). The harness half of that file -- CPU/GPU verification oracles, timing,
// CLI -- is deliberately not here. The source repo indents at 2; what you are
// reading was reformatted to aiter's .clang-format on the way in, so this file
// does not line up line-for-line with the source.
//
// Formatted by: AMD clang-format version 22.0.0git

#pragma once

// GPU kernels for generalized top-k paths. Include AFTER block_select_lds / block_gather_topk
// are visible in the translation unit.

template <bool RAGGED, bool WRITE_VALUES>
__global__ void phase_small_n_topk(const float* __restrict__ input,
                                   int pitch,
                                   RowExtents<RAGGED> extents,
                                   int K,
                                   TopkOut<WRITE_VALUES> dst,
                                   int npasses)
{
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
        s_keys[base + 0] = fp32_to_sortable(v[0]);
        s_keys[base + 1] = fp32_to_sortable(v[1]);
        s_keys[base + 2] = fp32_to_sortable(v[2]);
        s_keys[base + 3] = fp32_to_sortable(v[3]);
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

    int bcnt = 0;

// The drain's cost is its reservation atomic, not its data movement: with no
// candidate passing, phase_b ran 100.35 us faster at m=4096 n=131072, while the
// compaction arithmetic, its LDS write and the drain's LDS read and global write
// each priced free. cand_reserved[row] is one address per row that every wave
// of the row's coop_g blocks contends for (knowledge/known_bad.md).
#define COOP_RESERVE(row, n) atomicAdd(&cand_reserved[(size_t)(row) * CTR_STRIDE], (unsigned)(n))
#define COOP_DRAIN_BODY(off)                       \
    for(int _j = lane; _j < bcnt; _j += WAVE_SIZE) \
    row_base[(off) + _j] = buf[_j]

#define COOP_DRAIN_WAVE()                                            \
    do                                                               \
    {                                                                \
        unsigned _off = 0;                                           \
        if(lane == 0)                                                \
            _off = COOP_RESERVE(row, bcnt);                          \
        _off = (unsigned)__shfl((int)_off, 0);                       \
        if(_off + (unsigned)bcnt > (unsigned)cap)                    \
        {                                                            \
            if(lane == 0)                                            \
                atomicExch(&cand_bad[(size_t)row * CTR_STRIDE], 1u); \
            bcnt = -1;                                               \
            break;                                                   \
        }                                                            \
        COOP_DRAIN_BODY(_off);                                       \
        bcnt = 0;                                                    \
    } while(0)

    if constexpr(PB_B > 0)
    {
        // With at most two blocks per CU little else hides a filter load, so each
        // thread issues PB_B before filtering any, in the loop's filter order; a
        // slot past i1 loads a live address and is masked by `live`.
#ifndef COOP_DRAIN_AT
#define COOP_DRAIN_AT (WSTAGE_CAP_COOP - 4 * WAVE_SIZE)
#endif
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
#if PB_BATCH_MUTANT // gate self-test only: never filters a batch's last slot
                filter_v4(vv[t], i, i < i1 && t != PB_B - 1);
#else
                filter_v4(vv[t], i, i < i1);
#endif
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
#ifndef COOP_DRAIN_AT
#define COOP_DRAIN_AT (WSTAGE_CAP_COOP - 4 * WAVE_SIZE)
#endif
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
    // ragged form before v8) costs the whole row to reach three columns.
    //
    // The block that owns the last chunk picks the tail up instead: at most three
    // columns, one lane each, staged exactly like any other candidate so the
    // epilogue needs no special case. bcnt is at most WSTAGE_CAP_COOP - 4 * WAVE_SIZE
    // here because the drain check runs at the end of every iteration, so the
    // three extra entries cannot overflow the staging buffer.
    //
    // Measured before this existed: at m=4 N=131075 the answer was exactly
    // torch.topk over the first 131072 columns, with the 2049th largest value
    // standing in for the one that fell in the tail (scripts/probe72.py).
    {
        const int tail0 = n4 * FP32_EPT;
        const int ncols = len - tail0;
        if(ncols > 0 && bid == G - 1 && bcnt >= 0)
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
    __shared__ int s_tot;
    if(lane == 0)
        s_local[wid] = bcnt;
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
            s_tot  = total;
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
    // m=4096 k=2048 --dist gaussian --seed 0, phase_b device
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
    // blocks are still reading. knowledge/known_bad.md has the full pricing,
    // including the three levers that measured nothing.
    {
        const int cnt = s_local[wid];
        if(cnt > 0)
        {
            uint64_t* dst = row_base + s_base + s_off[wid];
            // The store takes the same gate as the load, and for the same reason.
            // Measured three-kernel total, per-wave with an ordinary store against
            // per-wave with a non-temporal one, k=2048 --dist gaussian --seed 0:
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
            // only the non-temporal part is gated.
            if constexpr(NT_STORE)
            {
                for(int j = lane; j < cnt; j += WAVE_SIZE)
                    __builtin_nontemporal_store(buf[j], &dst[j]);
            }
            else
            {
                for(int j = lane; j < cnt; j += WAVE_SIZE)
                {
#if NT_LOAD_ONLY_MUTANT
                    // Gate self-test only: drops each wave's first candidate in the
                    // NT-load / cached-store instantiation. Must turn the gate red.
                    if(NT && j == 0)
                        continue;
#endif
                    dst[j] = buf[j];
                }
            }
        }
    }
}

template <bool RAGGED, bool WRITE_VALUES, bool REUSE_WIDE = false>
__global__ PHASE_C_OCCUPANCY void
phase_c_select_contig(const float* __restrict__ input,
                      int pitch,
                      RowExtents<RAGGED> extents,
                      const uint64_t* __restrict__ cand_pack,
                      const unsigned int* __restrict__ cand_reserved,
                      const unsigned int* __restrict__ cand_bad,
                      unsigned int* __restrict__ cand_count,
                      int cap,
                      int K,
                      TopkOut<WRITE_VALUES> dst,
                      int* __restrict__ fb_rows,
                      int* __restrict__ fb_count,
                      int npasses,
                      bool keys_only,
                      int nwide,
                      const uint32_t* __restrict__ threshold)
{
    const int row       = blockIdx.x;
    const int row_start = RAGGED ? extents.row_start(row, pitch) : 0;
    const int len       = row_len_of<RAGGED>(row, pitch, extents);
    // Both counters unconditionally: the ternary compiled to load, wait, branch,
    // load, a second serial round trip in front of every phase_c block.
    const unsigned int bad_raw = cand_bad[(size_t)row * CTR_STRIDE];
    const unsigned int res_raw = cand_reserved[(size_t)row * CTR_STRIDE];
    const unsigned int c_raw   = bad_raw ? 0xFFFFFFFFu : res_raw;
    if(threadIdx.x == 0)
        cand_count[row] = c_raw;

    extern __shared__ uint32_t s_dyn[];
    uint32_t* s_keys_ext = s_dyn;
    int* s_idx           = reinterpret_cast<int*>(s_dyn + cap);
    // nwide * WIDE_WORDS words after the keys (and the indices unless keys_only).
    uint32_t* s_wide = s_dyn + (keys_only ? cap : 2 * cap);
    __shared__ __align__(16) uint32_t s_hist[HIST_SLOTS];
    __shared__ uint32_t s_scan[2];
    __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];
    __shared__ unsigned s_wgt, s_weq;

    int* out        = dst.idx_row(row, K);
    float* val      = dst.val_row(row, K);
    const int k_out = RAGGED ? k_take_dev(K, len) : K;
    // len <= K routes unconditionally so the identity emit cannot be diverted by
    // a cand_count the +inf threshold let through.
    int c_band = (int)c_raw;
    if((RAGGED && len <= K) || c_raw < (unsigned)k_out || c_raw > (unsigned)cap)
    {
        if(threadIdx.x == 0)
            fb_rows[atomicAdd(fb_count, 1)] = row;
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
            got = radix_fallback_row<true, WRITE_VALUES>(rif0,
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
            got =
                radix_fallback_row<RAGGED, WRITE_VALUES>(rif0,
                                                         RAGGED ? n4_cover(len) : pitch / FP32_EPT,
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
#if FB_BAND_PROBE
        if(threadIdx.x == 0)
            printf("FBBAND row=%d c_raw=%u got=%d\n", row, c_raw, got);
#endif
        if(got < 0)
            return; // emitted; len > K here, so k_out == K and nothing to pad
        c_band = got;
    }
    const int c          = c_band;
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
    clear_hist(s_hist);
    __syncthreads();
    // Published by the barrier after the read. Folding the first wide pass's
    // digits into this read as well, the way pass 0's are, was measured and is
    // SLOWER: phase_c +0.2 to +1.0us over the unfolded wide select at m=64..512
    // (scripts/wide_ab.py, arms acF against acN), against -0.3 to -0.4us unfolded.
    if(nwide > 0)
        clear_wide(s_wide, wide_buffer_count(nwide, REUSE_WIDE));
    auto take_cand = [&](uint64_t p, int i) {
        const uint32_t kk = fp32_to_sortable_bits((uint32_t)(p >> 32));
        s_keys_ext[i]     = kk;
        if(!keys_only)
            s_idx[i] = (int)(uint32_t)p;
        // radix_shift(0) is 24, so pass 0's digit is the top byte.
        atomicAdd(&s_hist[(kk >> 24) * HIST_REP + fold_rep], 1u);
    };
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
#if PC_BATCH_MUTANT // gate self-test only: never stores a batch's last candidate
            if(i < c && u != PC_B - 1)
                take_cand(pv[u], i);
#else
            if(i < c)
                take_cand(pv[u], i);
#endif
        }
    }
    __syncthreads();

    uint32_t pivot;
    int eq_needed;
    // Prices phase_c the way ABLATE_PA prices phase_a: the candidate read, the LDS
    // fill and the gather all stay, only block_select_lds goes. The gather then
    // works off a pivot that selects nothing in particular, so the results are
    // WRONG -- this exists to say how much of phase_c is the select.
#ifndef ABLATE_PC
#define ABLATE_PC 0
#endif
#if ABLATE_PC
    // Block-UNIFORM and below every key, so block_gather_topk finds its k_out
    // immediately instead of spinning: a per-thread pivot made it loop and the
    // measurement came back at 330-937us, which was the spin, not the select.
    pivot     = 0u;
    eq_needed = 0;
    (void)npasses;
#else
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
#endif

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
}

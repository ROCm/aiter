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
// Formatted by: clang-format version 23.1.1

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
    __shared__ uint32_t s_hist[HIST_SLOTS];
    __shared__ uint32_t s_red[256];
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
    block_select_lds(
        s_keys, len, k_out, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses, true);

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

template <bool RAGGED, bool NT>
__global__ void phase_b_filter_coop(const float* __restrict__ input,
                                    int pitch,
                                    RowExtents<RAGGED> extents,
                                    int n4_per_row,
                                    const float* __restrict__ threshold_f,
                                    uint64_t* __restrict__ cand_pack,
                                    unsigned int* __restrict__ cand_reserved,
                                    unsigned int* __restrict__ cand_bad,
                                    int cap)
{
    const int row   = blockIdx.y;
    const int len   = row_len_of<RAGGED>(row, pitch, extents);
    const float* ri = input + (size_t)row * pitch + (RAGGED ? extents.row_start(row, pitch) : 0);
#if ABLATE_TH == 1
    // Nothing passes, with EVERY host-side parameter left alone. The --margin 0.02
    // reference reached the same branch outcome by moving the threshold from the
    // host, which also moves rank and cap; this moves only the value phase_b
    // compares against, so the two runs are the same code path on the same grid.
    const float th = INFINITY;
#else
    const float th = threshold_f[row];
#endif
    const int lane    = threadIdx.x & (WAVE_SIZE - 1);
    const int wid     = threadIdx.x / WAVE_SIZE;
    const int nwaves  = blockDim.x / WAVE_SIZE;
    const uint64_t lt = (1ull << lane) - 1ull;

    __shared__ uint64_t wbuf[WSTAGE_WAVES * WSTAGE_CAP];
    uint64_t* buf      = wbuf + (size_t)wid * WSTAGE_CAP;
    uint64_t* row_base = cand_pack + (size_t)row * cap;
    (void)nwaves;

    const int G        = gridDim.x;
    const int bid      = blockIdx.x;
    const int n4       = RAGGED ? n4_cover(len) : n4_per_row;
    const int chunk_n4 = (n4 + G - 1) / G;
    const int i0       = bid * chunk_n4;
    const int i1       = min(i0 + chunk_n4, n4);
    const int stride   = blockDim.x;
    const int iters    = (i1 > i0) ? ((i1 - i0) + stride - 1) / stride : 0;

    int bcnt = 0;

// Ceiling-first pricing for the candidate compaction, in the style this repo
// already uses for ABLATE_HIST_ATOMIC: both of these give WRONG results and
// exist only to say what a perfect version of one half could be worth.
//
//   1 = write to a fixed per-lane slot, so the ballot prefix arithmetic
//       (s_and + s_bcnt + the running offsets) disappears but the ds_write and
//       its divergence stay.
//   2 = do all the arithmetic and skip the ds_write itself.
//
// Measured baseline to price against: an ablation that skips the compact block
// entirely (tiny margin, nothing passes the threshold) runs phase_b at 363.52us
// against 463.87 at m=4096 n=131072, so the whole compaction is 100.35us.
#ifndef ABLATE_COMPACT
#define ABLATE_COMPACT 0
#endif

// Separating the drain. ABLATE_COMPACT below showed the compaction's prefix
// arithmetic and its ds_write are both free, which leaves the drain as the whole
// of the 100.35us (m=4096 n=131072) that disappears when nothing passes the
// threshold. The drain does three things at once -- a global atomicAdd, an LDS
// READ of the staged entry, and a global write -- so "it is the write" was an
// inference from an older control kernel, not a measurement. These split it.
//
//   1 = write a constant instead of buf[_j]: the global write stays, the LDS
//       read goes.  (shipped - this) = the LDS read.
//   2 = skip the copy loop entirely: both go, the atomicAdd stays.
//       (1 - this) = the global write.
//
// Both give wrong results and exist only to price a half.
// 3 = replace the reservation atomic with a fixed offset (wrong results). The
//     other three ablations each removed one job of the drain and NONE of them
//     moved the clock: prefix arithmetic free, compaction ds_write free, drain
//     LDS read free, drain global write free -- yet the 100.35us still vanishes
//     when no candidate passes. This is what is left. `cand_reserved[row]` is
//     ONE 4-byte address per row and coop_g=8 blocks x 8 waves = 64 waves
//     contend for it.
#ifndef ABLATE_DRAIN
#define ABLATE_DRAIN 0
#endif
#if ABLATE_DRAIN == 3
#define COOP_RESERVE(row, n) ((unsigned)0)
#else
#define COOP_RESERVE(row, n) atomicAdd(&cand_reserved[row], (unsigned)(n))
#endif
#if ABLATE_DRAIN == 1
#define COOP_DRAIN_BODY(off)                       \
    for(int _j = lane; _j < bcnt; _j += WAVE_SIZE) \
    row_base[(off) + _j] = 1ull
#elif ABLATE_DRAIN == 2
#define COOP_DRAIN_BODY(off) \
    do                       \
    {                        \
    } while(0)
#else
#define COOP_DRAIN_BODY(off)                       \
    for(int _j = lane; _j < bcnt; _j += WAVE_SIZE) \
    row_base[(off) + _j] = buf[_j]
#endif

#define COOP_DRAIN_WAVE()                         \
    do                                            \
    {                                             \
        unsigned _off = 0;                        \
        if(lane == 0)                             \
            _off = COOP_RESERVE(row, bcnt);       \
        _off = (unsigned)__shfl((int)_off, 0);    \
        if(_off + (unsigned)bcnt > (unsigned)cap) \
        {                                         \
            if(lane == 0)                         \
                atomicExch(&cand_bad[row], 1u);   \
            bcnt = -1;                            \
            break;                                \
        }                                         \
        COOP_DRAIN_BODY(_off);                    \
        bcnt = 0;                                 \
    } while(0)

    for(int it = 0; it < iters; it++)
    {
        const int i     = i0 + it * stride + threadIdx.x;
        vfloat4 v       = {0.f, 0.f, 0.f, 0.f};
        const bool live = (i < i1);
        if(live)
            v = load_row_f4<RAGGED, NT>(ri, i, len);
        const int base_idx = i * FP32_EPT;
        const uint64_t b0  = __ballot(live && !(v[0] < th) && (!RAGGED || base_idx + 0 < len));
        const uint64_t b1  = __ballot(live && !(v[1] < th) && (!RAGGED || base_idx + 1 < len));
        const uint64_t b2  = __ballot(live && !(v[2] < th) && (!RAGGED || base_idx + 2 < len));
        const uint64_t b3  = __ballot(live && !(v[3] < th) && (!RAGGED || base_idx + 3 < len));
        const int t0       = __popcll(b0);
        const int t1       = t0 + __popcll(b1);
        const int t2       = t1 + __popcll(b2);
        const int wtotal   = t2 + __popcll(b3);
#if ABLATE_COMPACT == 3
        // Everything the branch guards, gone -- with the REAL threshold and the real
        // shape parameters. The 363.52us reading it is compared against came from
        // --margin 0.02, which changes the threshold to get the same branch outcome;
        // this reproduces the outcome without touching any parameter, so whatever
        // separates 363 from 463 has nowhere else to hide.
        (void)wtotal;
#else
        if(wtotal > 0)
        {
#if ABLATE_COMPACT == 1
            if(b0 & (1ull << lane))
                buf[lane] = ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
            if(b1 & (1ull << lane))
                buf[lane] = ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
            if(b2 & (1ull << lane))
                buf[lane] = ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
            if(b3 & (1ull << lane))
                buf[lane] = ((uint64_t)__float_as_uint(v[3]) << 32) | (uint32_t)(base_idx + 3);
#elif ABLATE_COMPACT == 2
            if(b0 & (1ull << lane))
                (void)(bcnt + __popcll(b0 & lt));
            if(b1 & (1ull << lane))
                (void)(bcnt + t0 + __popcll(b1 & lt));
            if(b2 & (1ull << lane))
                (void)(bcnt + t1 + __popcll(b2 & lt));
            if(b3 & (1ull << lane))
                (void)(bcnt + t2 + __popcll(b3 & lt));
#else
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
#endif
            bcnt += wtotal;
        }
#if ABLATE_DRAIN == 4
        // The drain CHECK itself, gone: no wave_barrier, no drain. The 2x2 leaves
        // 76us unaccounted after both the staging write and the drain copy are
        // removed, and __builtin_amdgcn_wave_barrier() fires every ~11 iterations
        // (bcnt passes 64 at ~5.6 passers per wave-iteration). It is a scheduling
        // barrier, so it stops the compiler hoisting the next loads past it.
        (void)0;
#else
        if(bcnt > WSTAGE_CAP - 4 * WAVE_SIZE)
        {
            __builtin_amdgcn_wave_barrier();
            COOP_DRAIN_WAVE();
        }
#endif
#endif
    }
#undef COOP_DRAIN_WAVE

// The epilogue, priced. Every ABLATE_DRAIN variant above missed this block:
// COOP_DRAIN_WAVE is #undef'd before it, so the epilogue carries its own copy
// loop. It is also the only thing the ABLATE_TH=1 control removes that the
// ablations did not -- with nothing passing, total==0 sets s_base to 0xFFFFFFFE
// and every block returns at the guard below, skipping all of it.
//
//   1 = skip the final LDS->global copy loop: the candidate write goes, the two
//       __syncthreads, the serial prefix over waves and the block atomicAdd stay.
//   2 = return right after the filter loop: the whole epilogue goes.
// Both give wrong results and exist only to price a half.
#ifndef NT_CAND
#define NT_CAND 0
#endif

#ifndef ABLATE_EPI
#define ABLATE_EPI 0
#endif
#if ABLATE_EPI == 2
    return;
#endif
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
                atomicExch(&cand_bad[row], 1u);
            s_base = 0xFFFFFFFFu;
        }
        else if(total == 0)
        {
            s_base = 0xFFFFFFFEu;
        }
        else
        {
            s_tot  = total;
            s_base = atomicAdd(&cand_reserved[row], (unsigned)total);
            if(s_base + (unsigned)total > (unsigned)cap)
            {
                atomicExch(&cand_bad[row], 1u);
                s_base = 0xFFFFFFFFu;
            }
        }
    }
    __syncthreads();
    if(s_base == 0xFFFFFFFFu || s_base == 0xFFFFFFFEu)
        return;

#if ABLATE_EPI == 3
    // Price the misalignment alone: same bytes, same passes, but every wave
    // writes from the row base, which cap=4096 uint64 makes 32 KB aligned.
    // Wrong results (the waves overwrite each other) -- pricing only.
    for(int w = 0; w < nwaves; w++)
    {
        const int cnt = s_local[w];
        if(cnt <= 0)
            continue;
        uint64_t* dst       = row_base;
        const uint64_t* src = wbuf + (size_t)w * WSTAGE_CAP;
        for(int j = threadIdx.x; j < cnt; j += blockDim.x)
            dst[j] = src[j];
    }
#elif ABLATE_EPI == 4
    // CORRECT, not an ablation: each wave copies its own staged run, so the eight
    // runs go out at once instead of the block walking them one after another.
    // Same bytes to the same addresses; only the thread-to-element map changes.
    {
        const int cnt = s_local[wid];
        if(cnt > 0)
        {
            uint64_t* dst = row_base + s_base + s_off[wid];
            for(int j = lane; j < cnt; j += WAVE_SIZE)
                dst[j] = buf[j];
        }
    }
#elif ABLATE_EPI == 90
    // CORRECT. Both halves that measured something, together: each wave copies its
    // own staged run so the eight runs go out at once (-12.8us alone), and the
    // store is non-temporal so the candidate bytes stop evicting the row data the
    // other blocks are still reading (-12.1us alone, on top of the walk form).
    {
        const int cnt = s_local[wid];
        if(cnt > 0)
        {
            uint64_t* dst = row_base + s_base + s_off[wid];
            for(int j = lane; j < cnt; j += WAVE_SIZE)
                __builtin_nontemporal_store(buf[j], &dst[j]);
        }
    }
#elif ABLATE_EPI == 8
    // CORRECT, not an ablation. The pricing says the copy pays to interleave with
    // phase_b's read stream rather than for its own bytes, so the thing to change
    // is not how the write is issued but whether it disturbs the reads. A
    // non-temporal store bypasses the caches, so the candidate run stops evicting
    // the row data the other blocks are still reading. Same bytes, same addresses.
    for(int j = threadIdx.x; j < s_tot; j += blockDim.x)
    {
        int w = 0;
        while(w + 1 < nwaves && j >= s_off[w + 1])
            w++;
        __builtin_nontemporal_store(wbuf[(size_t)w * WSTAGE_CAP + (j - s_off[w])],
                                    &row_base[s_base + j]);
    }
#elif ABLATE_EPI == 7
    // Price halving the candidate record. Alignment, the thread map and the number
    // of passes are all closed, so the only lever left on the epilogue's 98.3us is
    // bytes. This writes the low 32 bits only, which is the footprint a 4-byte
    // record would have. Wrong results -- phase_c reads uint64 -- pricing only.
    {
        uint32_t* row32 = reinterpret_cast<uint32_t*>(row_base);
        for(int j = threadIdx.x; j < s_tot; j += blockDim.x)
        {
            int w = 0;
            while(w + 1 < nwaves && j >= s_off[w + 1])
                w++;
            row32[s_base + j] = (uint32_t)wbuf[(size_t)w * WSTAGE_CAP + (j - s_off[w])];
        }
    }
#elif ABLATE_EPI == 6
    // Alignment ONLY, priced honestly. ABLATE_EPI=3 was not a valid ceiling: it
    // sent all eight waves to row_base, which shrinks the footprint eightfold and
    // lets the stores overwrite each other, so its 56.0us/128.9us measured a
    // smaller write, not an aligned one. This keeps the full footprint and the
    // same number of distinct lines, and only rounds the head down to a 128-byte
    // boundary. Wrong results (it shifts the run) -- pricing only.
    for(int j = threadIdx.x; j < s_tot; j += blockDim.x)
    {
        int w = 0;
        while(w + 1 < nwaves && j >= s_off[w + 1])
            w++;
        row_base[(s_base & ~15u) + j] = wbuf[(size_t)w * WSTAGE_CAP + (j - s_off[w])];
    }
#elif ABLATE_EPI == 5
    // CORRECT, not an ablation. s_off is a prefix sum, so row_base + s_base +
    // s_off[w] for consecutive w is already one contiguous run of s_tot entries.
    // Walking it in eight per-wave chunks restarts the thread-to-address map eight
    // times, and each restart puts a 512-byte wave store on an arbitrary boundary.
    // Aligning dst (ABLATE_EPI=3, wrong results) was worth 56.0us at m=4096
    // n=131072 and 128.9us at n=262144, which is 57% and 84% of the whole copy.
    // One block-wide walk leaves a single unaligned head instead of eight, writes
    // the same bytes to the same addresses, and needs no padding, so phase_c's
    // contiguous [0, cand_reserved[row]) scan is untouched.
    for(int j = threadIdx.x; j < s_tot; j += blockDim.x)
    {
        int w = 0;
        while(w + 1 < nwaves && j >= s_off[w + 1])
            w++;
        row_base[s_base + j] = wbuf[(size_t)w * WSTAGE_CAP + (j - s_off[w])];
    }
#elif ABLATE_EPI == 91
    // The block walking the eight staged runs one after another, which is what
    // this shipped before. Kept so the 22.6us below stays reproducible.
    for(int w = 0; w < nwaves; w++)
    {
        const int cnt = s_local[w];
        if(cnt <= 0)
            continue;
        uint64_t* dst       = row_base + s_base + s_off[w];
        const uint64_t* src = wbuf + (size_t)w * WSTAGE_CAP;
        for(int j = threadIdx.x; j < cnt; j += blockDim.x)
            dst[j] = src[j];
    }
#elif ABLATE_EPI != 1
    // Each wave writes out its own staged run, with a non-temporal store.
    //
    // Both halves were priced against the block-serial walk this replaces
    // (ABLATE_EPI=91), at m=4096 k=2048 --dist gaussian --seed 0, phase_b device
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
            if constexpr(NT)
            {
                for(int j = lane; j < cnt; j += WAVE_SIZE)
                    __builtin_nontemporal_store(buf[j], &dst[j]);
            }
            else
            {
                for(int j = lane; j < cnt; j += WAVE_SIZE)
                    dst[j] = buf[j];
            }
        }
    }
#endif
}

template <bool RAGGED, bool WRITE_VALUES>
__global__ void phase_c_select_contig(const float* __restrict__ input,
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
                                      bool keys_only)
{
    const int row            = blockIdx.x;
    const int row_start      = RAGGED ? extents.row_start(row, pitch) : 0;
    const int len            = row_len_of<RAGGED>(row, pitch, extents);
    const unsigned int c_raw = cand_bad[row] ? 0xFFFFFFFFu : cand_reserved[row];
    if(threadIdx.x == 0)
        cand_count[row] = c_raw;

    extern __shared__ uint32_t s_dyn[];
    uint32_t* s_keys_ext = s_dyn;
    int* s_idx           = reinterpret_cast<int*>(s_dyn + cap);
    __shared__ uint32_t s_hist[HIST_SLOTS];
    __shared__ uint32_t s_red[256];
    __shared__ uint32_t s_scan[2];
    __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];
    __shared__ unsigned s_wgt, s_weq;

    int* out        = dst.idx_row(row, K);
    float* val      = dst.val_row(row, K);
    const int k_out = RAGGED ? k_take_dev(K, len) : K;
    // See phase_c_select_waveseg: len <= K routes unconditionally so the identity
    // emit cannot be diverted by a cand_count the +inf threshold let through.
    if((RAGGED && len <= K) || c_raw < (unsigned)k_out || c_raw > (unsigned)cap)
    {
        if(threadIdx.x == 0)
            fb_rows[atomicAdd(fb_count, 1)] = row;
        exact_row_select<RAGGED, WRITE_VALUES>(
            input, pitch, extents, K, row, out, val, s_hist, s_red, s_scan, &s_wgt, &s_weq);
        return;
    }
    const int c          = (int)c_raw;
    const uint64_t* base = cand_pack + (size_t)row * cap;

    // Prices the other half of fusing phase_b into phase_c: if the candidates were
    // already in this block's LDS, this read would not happen. Wrong results.
#ifndef ABLATE_CREAD
#define ABLATE_CREAD 0
#endif
#if ABLATE_CREAD
    for(int i = threadIdx.x; i < c; i += blockDim.x)
    {
        s_keys_ext[i] = (uint32_t)i;
        if(!keys_only)
            s_idx[i] = i;
    }
#else
    // ATT puts s_waitcnt vmcnt(0) at 20.8% of phase_c's traced latency here, but
    // unrolling this loop is worth nothing: measured at depth 1, 2, 4 and 8,
    // phase_c reads 14.24, 14.27, 14.40 and 14.31us at m=512 n=131072. The trace
    // shows four separate load sites already, and c is about 2867 against a
    // 1024-thread block, so there are three iterations and nothing left to
    // overlap. The wait is the latency of the read itself.
    for(int i = threadIdx.x; i < c; i += blockDim.x)
    {
#if NT_CAND
        // The candidate array is read once here and never again, and phase_b now
        // writes it non-temporally so it is not in cache to begin with. Pricing
        // knob: NT_CAND=0 puts the ordinary load back.
        uint64_t p = __builtin_nontemporal_load(&base[i]);
#else
        uint64_t p = base[i];
#endif
        s_keys_ext[i] = fp32_to_sortable_bits((uint32_t)(p >> 32));
        if(!keys_only)
            s_idx[i] = (int)(uint32_t)p;
    }
#endif
    __syncthreads();

    uint32_t pivot;
    int eq_needed;
    block_select_lds(
        s_keys_ext, c, k_out, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses, true);

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

template <bool RAGGED>
__global__ __launch_bounds__(1024) void phase_ab_fused(const float* __restrict__ input,
                                                       int pitch,
                                                       RowExtents<RAGGED> extents,
                                                       int rank,
                                                       int S,
                                                       int npasses,
                                                       int chunk_stride_host,
                                                       int seg_stride,
                                                       uint64_t* __restrict__ cand_pack,
                                                       int* __restrict__ cand_seg,
                                                       unsigned int* __restrict__ cand_count,
                                                       int* __restrict__ fb_count,
                                                       int K)
{
    const int row   = blockIdx.x;
    const int len   = row_len_of<RAGGED>(row, pitch, extents);
    const float* ri = input + (size_t)row * pitch + (RAGGED ? extents.row_start(row, pitch) : 0);

    if(threadIdx.x == 0 && row == 0)
        *fb_count = 0;

    extern __shared__ uint32_t s_keys[];
    __shared__ uint32_t s_hist[HIST_SLOTS];
    __shared__ uint32_t s_red[256];
    __shared__ uint32_t s_scan[2];
    __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];

    const int chunks       = S / SAMPLE_CHUNK_ELEMS;
    const int chunk_stride = RAGGED ? sample_chunk_stride(len, chunks) : chunk_stride_host;
    const int rank_row = RAGGED && len != pitch ? max(1, (int)((double)rank * pitch / len)) : rank;

    const int v4_per_chunk = SAMPLE_CHUNK_ELEMS / FP32_EPT;
    const int total_v4     = S / FP32_EPT;
    for(int u = threadIdx.x; u < total_v4; u += blockDim.x)
    {
        const int chunk = u / v4_per_chunk;
        const int off4  = u % v4_per_chunk;
        vfloat4 v = *(reinterpret_cast<const vfloat4*>(ri + (size_t)chunk * chunk_stride) + off4);
        const int base   = u * FP32_EPT;
        s_keys[base + 0] = fp32_to_sortable(v[0]);
        s_keys[base + 1] = fp32_to_sortable(v[1]);
        s_keys[base + 2] = fp32_to_sortable(v[2]);
        s_keys[base + 3] = fp32_to_sortable(v[3]);
    }
    __syncthreads();

    uint32_t pivot;
    int eq_needed;
    block_select_lds(s_keys, S, rank_row, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses);
    const float th = sortable_to_fp32(pivot);

    const int lane    = threadIdx.x & (WAVE_SIZE - 1);
    const int wid     = threadIdx.x / WAVE_SIZE;
    const int nwaves  = blockDim.x / WAVE_SIZE;
    const uint64_t lt = (1ull << lane) - 1ull;
    __shared__ uint64_t wbuf[WSTAGE_WAVES * WSTAGE_CAP];
    uint64_t* buf    = wbuf + (size_t)wid * WSTAGE_CAP;
    uint64_t* seg    = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW + (size_t)wid * seg_stride;
    const int n4     = RAGGED ? n4_cover(len) : (pitch / FP32_EPT);
    const int stride = blockDim.x;
    const int iters  = (n4 + stride - 1) / stride;
    int wcnt = 0, bcnt = 0;
    bool overflow = false;
    for(int it = 0; it < iters; it++)
    {
        const int i     = it * stride + threadIdx.x;
        vfloat4 v       = {0.f, 0.f, 0.f, 0.f};
        const bool live = (i < n4);
        if(live)
            v = load_row_f4<RAGGED>(ri, i, len);
        const int base_idx = i * FP32_EPT;
        const uint64_t b0  = __ballot(live && !(v[0] < th) && (!RAGGED || base_idx + 0 < len));
        const uint64_t b1  = __ballot(live && !(v[1] < th) && (!RAGGED || base_idx + 1 < len));
        const uint64_t b2  = __ballot(live && !(v[2] < th) && (!RAGGED || base_idx + 2 < len));
        const uint64_t b3  = __ballot(live && !(v[3] < th) && (!RAGGED || base_idx + 3 < len));
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
        if(bcnt >= WAVE_SIZE)
        {
            __builtin_amdgcn_wave_barrier();
            if(wcnt + bcnt <= seg_stride)
            {
                for(int j = lane; j < bcnt; j += WAVE_SIZE)
                    seg[wcnt + j] = buf[j];
            }
            else
            {
                overflow = true;
            }
            wcnt += bcnt;
            bcnt = 0;
        }
    }
    if(bcnt > 0)
    {
        __builtin_amdgcn_wave_barrier();
        if(wcnt + bcnt <= seg_stride)
        {
            for(int j = lane; j < bcnt; j += WAVE_SIZE)
                seg[wcnt + j] = buf[j];
        }
        else
        {
            overflow = true;
        }
        wcnt += bcnt;
    }
    __shared__ int s_seg[MAX_WAVES_PER_BLOCK];
    if(lane == 0)
        s_seg[wid] = overflow ? -1 : wcnt;
    __syncthreads();
    if(threadIdx.x == 0)
    {
        unsigned total = 0;
        bool bad       = false;
        for(int w = 0; w < nwaves; w++)
        {
            cand_seg[(size_t)row * MAX_WAVES_PER_BLOCK + w] = s_seg[w];
            if(s_seg[w] < 0)
                bad = true;
            else
                total += (unsigned)s_seg[w];
        }
        cand_count[row] = bad ? 0xFFFFFFFFu : total;
    }
    (void)K;
    (void)fb_count;
}

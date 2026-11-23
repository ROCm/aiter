// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused prefill FP8 MQA indexer logits + per-row top-K for gfx950 (GLM-5.x indexer: 32 heads x
// 128).
//
//   logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]   for n in [ks[m], ke[m])
//
// The [M, N] fp32 logits are never stored. Each row gets a threshold tau from a 1/32 strided sample
// of its keys, the logits scan stores only the keys scoring >= tau, and a select ranks those
// candidates:
//
//   1. sample   sample_kernel gathers K[::32], kv_scale[::32] and the windows [ceil(ks/32),
//   ceil(ke/32)); scan_kernel
//               in MODE_DENSE writes the fp32 logits of those sample keys, [M, ceil(N/32)];
//   2. tau      tau_kernel: e = K * ns / nvalid samples are expected above the row's K-th logit,
//   tau <= the
//               (e + 5 sqrt(e) + 8)-th largest sample (-inf for a row of at most K keys);
//   3. emit     scan_kernel in MODE_EMIT: the v35wes MFMA pipeline (csrc/kernels/fp8_mqa_logits.cu,
//   same geometry,
//               schedule and epilogue, so the same fp32 bits) over every key, storing (key, logit)
//               for logit >= tau, up to CAP candidates per (row, split);
//   4. select   select_kernel: the exact top-K of each row's candidates. A row whose candidates
//   cannot prove its
//               top-K (a full segment, or fewer than K candidates) is flagged;
//   5. fallback fb_compute + fb_select (the first 64 flagged rows, each recomputed by the whole
//   grid) and
//               fallback_kernel (the rest, one wave per row): the exact logits of the flagged rows
//               (bit-identical to v35wes) into scratch, then a radix select. Empty launches when
//               nothing was flagged.
//
// Every row's result is therefore a top-K of fp8_mqa_logits' output (exact ties at the K-th value
// broken arbitrarily). Kernels: csrc/kernels/fp8_mqa_topk/. Their hand-placed vmcnt waits and
// inline asm (prefetch drains, permlane wait states) were checked on the ISA this module's flags
// produce (aiter/jit/optCompilerConfig.json).
//
// No torch dependency: every scratch buffer is a slice of the caller's workspace (workspace_layout
// below; the Python wrapper allocates it from fp8_mqa_topk_workspace_size()), and the stream is
// aiter::getCurrentHIPStream().
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "fp8_mqa_topk.h"
#include "fp8_mqa_topk/fmt_scan.cuh"
#include "fp8_mqa_topk/fmt_select.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <tuple>
#include <vector>

namespace {
using mqa_topk::fp8;

constexpr int SAMPLE_STRIDE = 32;              // one sample key per 32
constexpr int CAND_CAP      = 4096;            // candidate slots per (row, split); a fuller row takes the fallback
constexpr int MAX_SPLIT     = 16;              // select_kernel reads at most 16 split segments per row
constexpr int SCAN_ROWS     = 32;              // rows per scan workgroup (8 waves x 4 rows)
constexpr int FB_SCRATCH_ROWS = 2 * mqa_topk::FBR;   // fb_compute's FBR rows + one per fallback_kernel wave (64)

// Makespan model of one scan launch (the v35wes launcher's, fit on MI355X): a workgroup costs GA + GBM per masked
// stage + GBC per clean stage (us); workgroups are dispatched in blockIdx order round-robin over 8 XCDs and greedily
// onto each XCD's CUs, one per CU. It assumes the single causal request of a k-pool prefill chunk (4 tokens per key);
// other layouts only get a worse grid, never a wrong result.
constexpr double GA = 9.87, GBM = 2.962, GBC = 1.975;

double model_makespan(int M, int N, int split, int rev, int cus_per_xcd)
{
    const int rb = MQA_TOPK_CDIV(M, SCAN_ROWS);
    std::vector<std::priority_queue<double, std::vector<double>, std::greater<double>>> xcd(8);
    for(auto& q : xcd)
        for(int i = 0; i < cus_per_xcd; ++i)
            q.push(0.0);
    double makespan = 0.0;
    int idx         = 0;
    for(int y = 0; y < split; ++y)
        for(int x = 0; x < rb; ++x, ++idx)
        {
            const int bx = rev ? rb - 1 - x : x;
            const int r0 = bx * SCAN_ROWS, r1 = std::min(M, r0 + SCAN_ROWS);
            auto ke      = [&](int m) { return std::min((4 * N - M + m + 1) / 4, N); };
            const int hi = ke(r1 - 1), mn = ke(r0);
            const int n   = MQA_TOPK_CDIV(hi, 64);
            const int tc1 = (r0 + SCAN_ROWS <= M) ? std::min(mn / 64, n) : 0;
            auto share    = [&](int n_) {
                const int per = MQA_TOPK_CDIV(n_, split);
                const int b   = y * per;
                return std::max(0, std::min(n_, b + per) - b);
            };
            auto& q        = xcd[idx % 8];
            const double t = q.top() + GA + GBM * share(n - tc1) + GBC * share(tc1);
            q.pop();
            q.push(t);
            makespan = std::max(makespan, t);
        }
    return makespan;
}

// (SplitN, reverse row order) of the smallest modelled makespan, cached per shape.
std::pair<int, int> choose_grid(int M, int N, int total_cu)
{
    static std::mutex mu;
    static std::map<std::tuple<int, int, int>, std::pair<int, int>> cache;
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_tuple(M, N, total_cu);
    const auto it  = cache.find(key);
    if(it != cache.end())
        return it->second;
    const int cus_per_xcd = std::max(1, total_cu / 8);
    double best           = 1e300;
    int split = 1, rev = 1;
    for(int s = 1; s <= MAX_SPLIT; ++s)
        for(int r = 0; r < 2; ++r)
        {
            const double t = model_makespan(M, N, s, r, cus_per_xcd);
            if(t < best * 0.999)
            {
                best  = t;
                split = s;
                rev   = r;
            }
        }
    return cache[key] = {split, rev};
}

template <int MODE>
void launch_scan(const fp8* q, const fp8* k, const float* kv_scale, const float* w, const int* ks, const int* ke,
                 int M, int N, std::pair<int, int> grid_choice, float* out, int out_cols, const float* tau,
                 unsigned long long* cand, int* cnt, int cap, hipStream_t stream)
{
    const auto [split, rev] = grid_choice;
    const dim3 grid(MQA_TOPK_CDIV(M, SCAN_ROWS), split);
    // 8 waves, SUB 2, PF 4: the v35wes configuration
    if(rev)
        mqa_topk::scan_kernel<8, 2, 4, MODE, true><<<grid, 512, 0, stream>>>(
            q, k, kv_scale, w, ks, ke, M, N, split, out, out_cols, tau, cand, cnt, cap);
    else
        mqa_topk::scan_kernel<8, 2, 4, MODE, false><<<grid, 512, 0, stream>>>(
            q, k, kv_scale, w, ks, ke, M, N, split, out, out_cols, tau, cand, cnt, cap);
}

// Step 1's inputs: the sample keys K[j * S] and scales kv_scale[j * S] (j < Ns), and each row's
// sample window [ceil(ks / S), ceil(ke / S)), i.e. floor((x + S - 1) / S) in wrapping int32
// arithmetic.
__global__ __launch_bounds__(256) void sample_kernel(const fp8* __restrict__ k,
                                                     const float* __restrict__ kv_scale,
                                                     const int* __restrict__ ks,
                                                     const int* __restrict__ ke,
                                                     int M,
                                                     int Ns,
                                                     fp8* __restrict__ k_s,
                                                     float* __restrict__ scale_s,
                                                     int* __restrict__ ks_s,
                                                     int* __restrict__ ke_s)
{
    constexpr int S      = SAMPLE_STRIDE;
    constexpr int CHUNKS = mqa_topk::HEAD_SIZE / 16; // 16-B chunks per key
    const int64_t tid    = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t nt     = int64_t(gridDim.x) * blockDim.x;
    for(int64_t i = tid; i < int64_t(Ns) * CHUNKS; i += nt)
    {
        const int64_t j = i / CHUNKS, c = i % CHUNKS;
        reinterpret_cast<uint4*>(k_s)[i] =
            reinterpret_cast<const uint4*>(k + j * S * mqa_topk::HEAD_SIZE)[c];
    }
    for(int64_t j = tid; j < Ns; j += nt)
        scale_s[j] = kv_scale[j * S];
    auto ceil_div = [](int x) {
        const int v = static_cast<int>(static_cast<unsigned>(x) + unsigned(S - 1));
        const int q = v / S;
        return (v % S != 0 && v < 0) ? q - 1 : q; // floor division
    };
    for(int64_t m = tid; m < M; m += nt)
    {
        ks_s[m] = ceil_div(ks[m]);
        ke_s[m] = ceil_div(ke[m]);
    }
}

// Byte offsets of every scratch buffer of one call inside the workspace (each slice 256-B aligned).
struct WorkspaceLayout
{
    int Ns, split, cap;
    int64_t k_s, scale_s, ks_s, ke_s, sample_logits, tau, cand, cnt, flag, flag_count, fb_scratch;
    int64_t bytes;
};

WorkspaceLayout workspace_layout(int M, int N, int total_cu)
{
    WorkspaceLayout l{};
    l.Ns    = MQA_TOPK_CDIV(N, SAMPLE_STRIDE);
    l.split = choose_grid(M, N, total_cu).first;
    // A (row, split) segment never holds more candidates than the row has keys.
    l.cap            = std::min(CAND_CAP, N);
    const int64_t Mp = M + (M & 1);
    int64_t off      = 0;
    auto take        = [&](int64_t bytes) {
        const int64_t at = off;
        off += (bytes + 255) & ~int64_t{255};
        return at;
    };
    l.k_s           = take(int64_t(l.Ns) * mqa_topk::HEAD_SIZE);     // [Ns, 128] fp8
    l.scale_s       = take(int64_t(l.Ns) * sizeof(float));           // [Ns]
    l.ks_s          = take(int64_t(M) * sizeof(int));                // [M]
    l.ke_s          = take(int64_t(M) * sizeof(int));                // [M]
    l.sample_logits = take(int64_t(M) * l.Ns * sizeof(float));       // [M, Ns]
    l.tau           = take(int64_t(M) * sizeof(float));              // [M]
    l.cand          = take(l.split * Mp * l.cap * sizeof(uint64_t)); // [split, Mp, cap]
    l.cnt           = take(int64_t(l.split) * M * sizeof(int));      // [split, M]
    l.flag          = take(int64_t(M) * sizeof(int));                // [M]
    l.flag_count    = take(sizeof(int));                             // [1], zeroed by tau_kernel
    l.fb_scratch    = take(int64_t(FB_SCRATCH_ROWS) * N * sizeof(float)); // [FB_SCRATCH_ROWS, N]
    l.bytes         = off;
    return l;
}
} // namespace

namespace aiter {

int64_t fp8_mqa_topk_workspace_size(int64_t M, int64_t N, int64_t TotalCuCount)
{
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(M >= 0 && M <= INT_MAX && N >= 0, "bad shape: M = ", M, ", N = ", N);
    AITER_CHECK(N * mqa_topk::HEAD_SIZE < (int64_t{1} << 31), "too many keys: ", N);
    AITER_CHECK(TotalCuCount >= 8, "TotalCuCount must be >= 8, got ", TotalCuCount);
    if(M == 0 || N == 0)
        return 0;
    return workspace_layout(
               static_cast<int>(M), static_cast<int>(N), static_cast<int>(TotalCuCount))
        .bytes;
}

void fp8_mqa_topk(const aiter_tensor_t& q_fp8,
                  const aiter_tensor_t& k_fp8,
                  const aiter_tensor_t& kv_scale,
                  const aiter_tensor_t& weights,
                  const aiter_tensor_t& cu_seqlen_ks,
                  const aiter_tensor_t& cu_seqlen_ke,
                  aiter_tensor_t& topk_indices,
                  aiter_tensor_t& workspace,
                  int64_t TotalCuCount,
                  std::optional<aiter_tensor_t> fallback_rows,
                  bool force_fallback)
{
    // Argument failures raise a Python exception instead of aborting the process.
    aiter_detail::AiterThrowGuard throw_guard;
    using mqa_topk::fp8;
    using mqa_topk::HEAD_SIZE;
    using mqa_topk::NUM_HEADS;
    AITER_CHECK(q_fp8.dim() == 3 && q_fp8.size(1) == NUM_HEADS && q_fp8.size(2) == HEAD_SIZE,
                "Only n_heads=32, head_dim=128 supported");
    AITER_CHECK(k_fp8.dim() == 2 && k_fp8.size(1) == HEAD_SIZE, "K must be [N, 128]");
    const int M = q_fp8.size(0);
    const int N = k_fp8.size(0);
    AITER_CHECK(q_fp8.element_size() == 1 && k_fp8.element_size() == 1,
                "q_fp8 and k_fp8 must be fp8");
    AITER_CHECK(q_fp8.is_contiguous() && k_fp8.is_contiguous(),
                "q_fp8 and k_fp8 must be contiguous");
    AITER_CHECK(kv_scale.dtype() == AITER_DTYPE_fp32 && kv_scale.is_contiguous() &&
                    kv_scale.numel() == size_t(N),
                "kv_scale must be a contiguous f32 [N] tensor");
    AITER_CHECK(weights.dtype() == AITER_DTYPE_fp32 && weights.is_contiguous() &&
                    weights.numel() == size_t(M) * NUM_HEADS,
                "weights must be a contiguous f32 [M, 32] tensor");
    AITER_CHECK(cu_seqlen_ks.dtype() == AITER_DTYPE_i32 &&
                    cu_seqlen_ke.dtype() == AITER_DTYPE_i32 && cu_seqlen_ks.is_contiguous() &&
                    cu_seqlen_ke.is_contiguous() && cu_seqlen_ks.numel() == size_t(M) &&
                    cu_seqlen_ke.numel() == size_t(M),
                "cu_seqlen_ks/ke must be contiguous int32 [M] tensors");
    AITER_CHECK(topk_indices.dtype() == AITER_DTYPE_i32 && topk_indices.is_contiguous() &&
                    topk_indices.dim() == 2 && topk_indices.size(0) == M,
                "topk_indices must be a contiguous int32 [M, K] tensor");
    const int K = topk_indices.size(1);
    AITER_CHECK(
        K >= 1 && K <= FP8_MQA_TOPK_MAX_K, "K must be in [1, ", FP8_MQA_TOPK_MAX_K, "], got ", K);
    // The scan addresses K through 32-bit buffer offsets.
    AITER_CHECK(int64_t(N) * HEAD_SIZE < (int64_t{1} << 31), "too many keys: ", N);
    AITER_CHECK(TotalCuCount >= 8, "TotalCuCount must be >= 8, got ", TotalCuCount);
    const int dev = q_fp8.device_id;
    AITER_CHECK(q_fp8.is_gpu() && k_fp8.device_id == dev && kv_scale.device_id == dev &&
                    weights.device_id == dev && cu_seqlen_ks.device_id == dev &&
                    cu_seqlen_ke.device_id == dev && topk_indices.device_id == dev,
                "all tensors must be on q_fp8's device");
    int* nfb = nullptr;
    if(fallback_rows.has_value())
    {
        const auto& f = *fallback_rows;
        AITER_CHECK(f.dtype() == AITER_DTYPE_i32 && f.numel() >= 1 && f.device_id == dev,
                    "fallback_rows must be an int32 tensor of at least one element");
        nfb = static_cast<int*>(f.data_ptr());
    }
    if(M == 0)
        return;

    const HipDeviceGuard guard(dev);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    int* out                 = static_cast<int*>(topk_indices.data_ptr());
    if(N == 0)
    {
        const hipError_t err =
            hipMemsetAsync(out, 0xff, size_t(M) * K * sizeof(int), stream); // every index -1
        AITER_CHECK(
            err == hipSuccess, "fp8_mqa_topk: hipMemsetAsync failed: ", hipGetErrorString(err));
        return;
    }

    const int total_cu        = static_cast<int>(TotalCuCount);
    const WorkspaceLayout lay = workspace_layout(M, N, total_cu);
    AITER_CHECK(workspace.is_contiguous() && workspace.device_id == dev &&
                    int64_t(workspace.numel() * workspace.element_size()) >= lay.bytes,
                "workspace must be a contiguous tensor on q_fp8's device of at least ",
                lay.bytes,
                " bytes (fp8_mqa_topk_workspace_size)");
    char* ws            = static_cast<char*>(workspace.data_ptr());
    auto* k_s           = reinterpret_cast<fp8*>(ws + lay.k_s);
    auto* scale_s       = reinterpret_cast<float*>(ws + lay.scale_s);
    auto* ks_s          = reinterpret_cast<int*>(ws + lay.ks_s);
    auto* ke_s          = reinterpret_cast<int*>(ws + lay.ke_s);
    auto* sample_logits = reinterpret_cast<float*>(ws + lay.sample_logits);
    auto* tau           = reinterpret_cast<float*>(ws + lay.tau);
    auto* cand          = reinterpret_cast<unsigned long long*>(ws + lay.cand);
    auto* cnt           = reinterpret_cast<int*>(ws + lay.cnt);
    auto* fl            = reinterpret_cast<int*>(ws + lay.flag);
    auto* fbc           = reinterpret_cast<int*>(ws + lay.flag_count);
    auto* scratch       = reinterpret_cast<float*>(ws + lay.fb_scratch);

    const auto* q   = static_cast<const fp8*>(q_fp8.data_ptr());
    const auto* k   = static_cast<const fp8*>(k_fp8.data_ptr());
    const float* sc     = static_cast<const float*>(kv_scale.data_ptr());
    const float* w      = static_cast<const float*>(weights.data_ptr());
    const int* ks       = static_cast<const int*>(cu_seqlen_ks.data_ptr());
    const int* ke       = static_cast<const int*>(cu_seqlen_ke.data_ptr());
    constexpr int S     = SAMPLE_STRIDE;
    const int Ns        = lay.Ns;
    const int split     = lay.split;
    const int cap       = lay.cap;
    const int row_waves = MQA_TOPK_CDIV(M, 4); // tau / select: one wave per row, 4 per workgroup

    // 1. sample keys j * S and their row windows [ceil(ks / S), ceil(ke / S)), then their logits
    const int64_t sample_items = std::max<int64_t>(int64_t(Ns) * (HEAD_SIZE / 16), M);
    sample_kernel<<<int(std::min<int64_t>(MQA_TOPK_CDIV(sample_items, 256), 4096)),
                    256,
                    0,
                    stream>>>(k, sc, ks, ke, M, Ns, k_s, scale_s, ks_s, ke_s);
    launch_scan<mqa_topk::MODE_DENSE>(q,
                                      k_s,
                                      scale_s,
                                      w,
                                      ks_s,
                                      ke_s,
                                      M,
                                      Ns,
                                      choose_grid(M, Ns, total_cu),
                                      sample_logits,
                                      Ns,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      0,
                                      stream);
    // 2. tau; rows of at most K keys get -inf (every key a candidate)
    mqa_topk::tau_kernel<<<row_waves, 256, 0, stream>>>(
        sample_logits, Ns, S, ks, ke, M, N, K, K, tau, fbc);
    if(force_fallback)
    {
        // +inf in every row: no logit is >= +inf, so no candidates
        const hipError_t err =
            hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(tau), 0x7f800000, M, stream);
        AITER_CHECK(
            err == hipSuccess, "fp8_mqa_topk: hipMemsetD32Async failed: ", hipGetErrorString(err));
    }
    // 3. candidates
    launch_scan<mqa_topk::MODE_EMIT>(q,
                                     k,
                                     sc,
                                     w,
                                     ks,
                                     ke,
                                     M,
                                     N,
                                     choose_grid(M, N, total_cu),
                                     nullptr,
                                     0,
                                     tau,
                                     cand,
                                     cnt,
                                     cap,
                                     stream);
    // 4. select
    mqa_topk::select_kernel<<<row_waves, 256, 0, stream>>>(
        cand, cnt, split, cap, ks, ke, M, N, K, out, fl, fbc);
    // 5. exact fallback of the flagged rows
    mqa_topk::fb_compute<<<64, 256, 0, stream>>>(fl, fbc, ks, ke, M, N, scratch, q, k, sc, w);
    mqa_topk::fb_select<<<mqa_topk::FBR / 4, 256, 0, stream>>>(fl, fbc, ks, ke, M, N, K, out, scratch, nfb);
    mqa_topk::fallback_kernel<<<16, 256, 0, stream>>>(fl, fbc, ks, ke, M, N, K, out, scratch, q, k, sc, w, nfb);
    const hipError_t err = hipGetLastError();
    AITER_CHECK(err == hipSuccess, "fp8_mqa_topk launch failed: ", hipGetErrorString(err));
}

} // namespace aiter

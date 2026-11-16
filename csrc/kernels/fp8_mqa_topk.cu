// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused prefill FP8 MQA indexer logits + per-row top-K for gfx950 (GLM-5.x indexer: 32 heads x 128).
//
//   logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]   for n in [ks[m], ke[m])
//
// The [M, N] fp32 logits are never stored. Each row gets a threshold tau from a 1/32 strided sample of its keys, the
// logits scan stores only the keys scoring >= tau, and a select ranks those candidates:
//
//   1. sample   K[::32], kv_scale[::32], windows [ceil(ks/32), ceil(ke/32)) (torch); scan_kernel in MODE_DENSE writes
//               the fp32 logits of those sample keys, [M, ceil(N/32)];
//   2. tau      tau_kernel: e = K * ns / nvalid samples are expected above the row's K-th logit, tau <= the
//               (e + 5 sqrt(e) + 8)-th largest sample (-inf for a row of at most K keys);
//   3. emit     scan_kernel in MODE_EMIT: the v35wes MFMA pipeline (csrc/kernels/fp8_mqa_logits.cu, same geometry,
//               schedule and epilogue, so the same fp32 bits) over every key, storing (key, logit) for logit >= tau,
//               up to CAP candidates per (row, split);
//   4. select   select_kernel: the exact top-K of each row's candidates. A row whose candidates cannot prove its
//               top-K (a full segment, or fewer than K candidates) is flagged;
//   5. fallback fb_compute + fb_select (the first 64 flagged rows, each recomputed by the whole grid) and
//               fallback_kernel (the rest, one wave per row): the exact logits of the flagged rows (bit-identical to
//               v35wes) into scratch, then a radix select. Empty launches when nothing was flagged.
//
// Every row's result is therefore a top-K of fp8_mqa_logits' output (exact ties at the K-th value broken
// arbitrarily). Kernels: csrc/kernels/fp8_mqa_topk/. Their hand-placed vmcnt waits and inline asm (prefetch drains,
// permlane wait states) were checked on the ISA this module's flags produce (aiter/jit/optCompilerConfig.json).
#include "fp8_mqa_topk.h"
#include "fp8_mqa_topk/fmt_scan.cuh"
#include "fp8_mqa_topk/fmt_select.cuh"

#include <algorithm>
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
} // namespace

torch::Tensor fp8_mqa_topk(torch::Tensor q_fp8,
                           torch::Tensor k_fp8,
                           torch::Tensor kv_scale,
                           torch::Tensor weights,
                           torch::Tensor cu_seqlen_ks,
                           torch::Tensor cu_seqlen_ke,
                           torch::Tensor topk_indices,
                           int64_t TotalCuCount,
                           std::optional<torch::Tensor> fallback_rows,
                           bool force_fallback)
{
    using mqa_topk::HEAD_SIZE;
    using mqa_topk::NUM_HEADS;
    TORCH_CHECK(q_fp8.dim() == 3 && q_fp8.size(1) == NUM_HEADS && q_fp8.size(2) == HEAD_SIZE,
                "Only n_heads=32, head_dim=128 supported");
    const int M = q_fp8.size(0);
    TORCH_CHECK(k_fp8.dim() == 2 && k_fp8.size(1) == HEAD_SIZE, "K must be [N, 128]");
    const int N = k_fp8.size(0);
    TORCH_CHECK(q_fp8.element_size() == 1 && k_fp8.element_size() == 1, "q_fp8 and k_fp8 must be fp8");
    TORCH_CHECK(q_fp8.is_contiguous() && k_fp8.is_contiguous(), "q_fp8 and k_fp8 must be contiguous");
    TORCH_CHECK(kv_scale.scalar_type() == torch::kFloat32 && kv_scale.is_contiguous() && kv_scale.numel() == N,
                "kv_scale must be a contiguous f32 [N] tensor");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32 && weights.is_contiguous() &&
                    weights.numel() == (int64_t)M * NUM_HEADS,
                "weights must be a contiguous f32 [M, 32] tensor");
    TORCH_CHECK(cu_seqlen_ks.scalar_type() == torch::kInt32 && cu_seqlen_ke.scalar_type() == torch::kInt32 &&
                    cu_seqlen_ks.is_contiguous() && cu_seqlen_ke.is_contiguous() &&
                    cu_seqlen_ks.numel() == M && cu_seqlen_ke.numel() == M,
                "cu_seqlen_ks/ke must be contiguous int32 [M] tensors");
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32 && topk_indices.is_contiguous() &&
                    topk_indices.dim() == 2 && topk_indices.size(0) == M,
                "topk_indices must be a contiguous int32 [M, K] tensor");
    const int K = topk_indices.size(1);
    TORCH_CHECK(K >= 1 && K <= FP8_MQA_TOPK_MAX_K, "K must be in [1, ", FP8_MQA_TOPK_MAX_K, "], got ", K);
    // The scan addresses K through 32-bit buffer offsets.
    TORCH_CHECK((int64_t)N * HEAD_SIZE < (int64_t{1} << 31), "too many keys: ", N);
    TORCH_CHECK(TotalCuCount >= 8, "TotalCuCount must be >= 8, got ", TotalCuCount);
    for(const auto* t : {&k_fp8, &kv_scale, &weights, &cu_seqlen_ks, &cu_seqlen_ke, &topk_indices})
        TORCH_CHECK(t->device() == q_fp8.device(), "all tensors must be on q_fp8's device");
    int* nfb = nullptr;
    if(fallback_rows.has_value())
    {
        const auto& f = fallback_rows.value();
        TORCH_CHECK(f.scalar_type() == torch::kInt32 && f.numel() >= 1 && f.device() == q_fp8.device(),
                    "fallback_rows must be an int32 tensor of at least one element");
        nfb = f.data_ptr<int>();
    }
    if(M == 0)
        return topk_indices;
    if(N == 0)
        return topk_indices.fill_(-1);

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(q_fp8));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    const auto i32           = torch::dtype(torch::kInt32).device(q_fp8.device());
    const auto f32           = torch::dtype(torch::kFloat32).device(q_fp8.device());
    const int total_cu       = static_cast<int>(TotalCuCount);

    // 1. sample keys j * S and their row windows [ceil(ks / S), ceil(ke / S))
    constexpr int S             = SAMPLE_STRIDE;
    const int Ns                = MQA_TOPK_CDIV(N, S);
    const torch::Tensor k_s     = k_fp8.view(torch::kUInt8).slice(0, 0, N, S).contiguous();
    const torch::Tensor scale_s = kv_scale.slice(0, 0, N, S).contiguous();
    const torch::Tensor ks_s    = at::div(cu_seqlen_ks + (S - 1), S, "floor");
    const torch::Tensor ke_s    = at::div(cu_seqlen_ke + (S - 1), S, "floor");
    torch::Tensor sample_logits = torch::empty({M, Ns}, f32);
    torch::Tensor tau           = torch::empty({M}, f32);
    // A (row, split) segment never holds more candidates than the row has keys.
    const int cap                 = std::min(CAND_CAP, N);
    const auto grid_choice        = choose_grid(M, N, total_cu);
    const int split               = grid_choice.first;
    const int64_t Mp              = M + (M & 1);
    torch::Tensor cand            = torch::empty({split * Mp * cap}, torch::dtype(torch::kInt64).device(q_fp8.device()));
    torch::Tensor cnt             = torch::empty({(int64_t)split * M}, i32);
    torch::Tensor flag            = torch::empty({M}, i32);
    torch::Tensor flag_count      = torch::empty({1}, i32);   // zeroed by tau_kernel
    torch::Tensor fb_scratch      = torch::empty({(int64_t)FB_SCRATCH_ROWS * N}, f32);

    const auto* q   = static_cast<const fp8*>(q_fp8.data_ptr());
    const auto* k   = static_cast<const fp8*>(k_fp8.data_ptr());
    const float* sc = kv_scale.data_ptr<float>();
    const float* w  = weights.data_ptr<float>();
    const int* ks   = cu_seqlen_ks.data_ptr<int>();
    const int* ke   = cu_seqlen_ke.data_ptr<int>();
    int* out        = topk_indices.data_ptr<int>();
    int* fl         = flag.data_ptr<int>();
    int* fbc        = flag_count.data_ptr<int>();
    float* scratch  = fb_scratch.data_ptr<float>();
    const int row_waves = MQA_TOPK_CDIV(M, 4);   // tau / select: one wave per row, 4 per workgroup

    launch_scan<mqa_topk::MODE_DENSE>(q, static_cast<const fp8*>(k_s.data_ptr()), scale_s.data_ptr<float>(), w,
                                      ks_s.data_ptr<int>(), ke_s.data_ptr<int>(), M, Ns, choose_grid(M, Ns, total_cu),
                                      sample_logits.data_ptr<float>(), Ns, nullptr, nullptr, nullptr, 0, stream);
    // 2. tau; rows of at most K keys get -inf (every key a candidate)
    mqa_topk::tau_kernel<<<row_waves, 256, 0, stream>>>(
        sample_logits.data_ptr<float>(), Ns, S, ks, ke, M, N, K, K, tau.data_ptr<float>(), fbc);
    if(force_fallback)
        tau.fill_(std::numeric_limits<float>::infinity());   // no logit is >= +inf: no candidates
    // 3. candidates
    launch_scan<mqa_topk::MODE_EMIT>(q, k, sc, w, ks, ke, M, N, grid_choice, nullptr, 0, tau.data_ptr<float>(),
                                     reinterpret_cast<unsigned long long*>(cand.data_ptr<int64_t>()),
                                     cnt.data_ptr<int>(), cap, stream);
    // 4. select
    mqa_topk::select_kernel<<<row_waves, 256, 0, stream>>>(
        reinterpret_cast<const unsigned long long*>(cand.data_ptr<int64_t>()), cnt.data_ptr<int>(), split, cap, ks,
        ke, M, N, K, out, fl, fbc);
    // 5. exact fallback of the flagged rows
    mqa_topk::fb_compute<<<64, 256, 0, stream>>>(fl, fbc, ks, ke, M, N, scratch, q, k, sc, w);
    mqa_topk::fb_select<<<mqa_topk::FBR / 4, 256, 0, stream>>>(fl, fbc, ks, ke, M, N, K, out, scratch, nfb);
    mqa_topk::fallback_kernel<<<16, 256, 0, stream>>>(fl, fbc, ks, ke, M, N, K, out, scratch, q, k, sc, w, nfb);
    const hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "fp8_mqa_topk launch failed: ", hipGetErrorString(err));
    return topk_indices;
}

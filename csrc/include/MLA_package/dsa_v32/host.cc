#include <opus/hip_minimal.hpp>
#include <algorithm>
#include <random>
#include <iostream>
#include <limits>
#include <numeric>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include "defs.h"
#include "parallel.h"

template<class Traits>
__global__ void opus_mla_decode_mxfp8_16mx8_32nx1_kernel(opus_mla_decode_mxfp8_kargs kargs);
template<class Traits>
__global__ void opus_mla_decode_a16w16_16mx4_64nx1_kernel(opus_mla_decode_kargs kargs);
template<class Traits>
__global__ void opus_mla_decode_a16w16_32mx1_16nx4_kernel(opus_mla_decode_kargs kargs);
template<class Traits>
__global__ void opus_mla_decode_a16w16_32mxt_32nx1_kernel(opus_mla_decode_kargs kargs);

__global__ void get_mla_metadata_kernel(opus_mla_decode_metadata_kargs kargs);
template<class Traits, int HEADS_PER_BLOCK = 8>
__global__ void mla_reduce_kernel(opus_mla_decode_reduce_kargs kargs);

static constexpr int MLA_DECODE_REDUCE_HEADS_PER_BLOCK = 8;

template<int Q, int KV, int NW, class DN, class DR, class DO>
inline void mla_decode_launch(opus_mla_decode_mxfp8_16mx8_32nx1_traits<Q, KV, NW, DN, DR, DO>,
                              const opus_mla_decode_mxfp8_kargs& kargs, dim3 grid, dim3 block) {
    using Traits = opus_mla_decode_mxfp8_16mx8_32nx1_traits<Q, KV, NW, DN, DR, DO>;
    opus_mla_decode_mxfp8_16mx8_32nx1_kernel<Traits><<<grid, block>>>(kargs);
}

template<int Q, int KV, int NW, class DA, class DO>
inline void mla_decode_launch(opus_mla_decode_a16w16_16mx4_64nx1_traits<Q, KV, NW, DA, DO>,
                              const opus_mla_decode_kargs& kargs, dim3 grid, dim3 block) {
    using Traits = opus_mla_decode_a16w16_16mx4_64nx1_traits<Q, KV, NW, DA, DO>;
    opus_mla_decode_a16w16_16mx4_64nx1_kernel<Traits><<<grid, block>>>(kargs);
}

template<int Q, int KV, int NW, class DA, class DO>
inline void mla_decode_launch(opus_mla_decode_a16w16_32mx1_16nx4_traits<Q, KV, NW, DA, DO>,
                              const opus_mla_decode_kargs& kargs, dim3 grid, dim3 block) {
    using Traits = opus_mla_decode_a16w16_32mx1_16nx4_traits<Q, KV, NW, DA, DO>;
    opus_mla_decode_a16w16_32mx1_16nx4_kernel<Traits><<<grid, block>>>(kargs);
}

template<int Q, int KV, int NW, class DA, class DO>
inline void mla_decode_launch(opus_mla_decode_a16w16_32mx4_32nx1_traits<Q, KV, NW, DA, DO>,
                              const opus_mla_decode_kargs& kargs, dim3 grid, dim3 block) {
    using Traits = opus_mla_decode_a16w16_32mx4_32nx1_traits<Q, KV, NW, DA, DO>;
    opus_mla_decode_a16w16_32mxt_32nx1_kernel<Traits><<<grid, block>>>(kargs);
}

template<int Q, int KV, int NW, class DA, class DO>
inline void mla_decode_launch(opus_mla_decode_a16w16_32mx3_32nx1_traits<Q, KV, NW, DA, DO>,
                              const opus_mla_decode_kargs& kargs, dim3 grid, dim3 block) {
    using Traits = opus_mla_decode_a16w16_32mx3_32nx1_traits<Q, KV, NW, DA, DO>;
    opus_mla_decode_a16w16_32mxt_32nx1_kernel<Traits><<<grid, block>>>(kargs);
}

// Host-side mirror of aiter's metadata buffer sizing (get_mla_metadata_info_v1),
// plus the device buffers the scheduler fills and the split-KV merge consumes.
struct mla_decode_plan {
    int B = 0;
    int H = 0;
    int seqlen_qo = 1;
    int num_qo_tiles = 1;
    int num_reduce_tiles = 0;
    int max_works = 0;
    int max_split_tiles = 0;
    int num_partial_rows = 0;
    int num_splits = 0;

    int* qo_indptr = nullptr;
    int* work_indptr = nullptr;
    opus_mla_decode_work_info* work_info_set = nullptr;
    int* reduce_indptr = nullptr;
    int* reduce_final_map = nullptr;
    int* reduce_partial_map = nullptr;
    float* o_accum = nullptr;
    float* lse_accum = nullptr;
};

inline opus_mla_decode_metadata_kargs mla_decode_metadata_kargs(const mla_decode_plan& plan,
                                                                const int* kv_indptr) {
    opus_mla_decode_metadata_kargs kargs{};
    kargs.qo_indptr = plan.qo_indptr;
    kargs.kv_indptr = kv_indptr;
    kargs.work_indptr = plan.work_indptr;
    kargs.work_info_set = plan.work_info_set;
    kargs.reduce_indptr = plan.reduce_indptr;
    kargs.reduce_final_map = plan.reduce_final_map;
    kargs.reduce_partial_map = plan.reduce_partial_map;
    kargs.B = plan.B;
    kargs.H = plan.H;
    kargs.num_cu = MLA_DECODE_NUM_CU;
    kargs.num_splits = plan.num_splits;
    kargs.uni_seqlen_qo = plan.seqlen_qo;
    kargs.kv_granularity = MLA_DECODE_KV_GRANULARITY;
    kargs.fixed_overhead = MLA_DECODE_FIXED_OVERHEAD;
    kargs.tail_done_threshold = plan.seqlen_qo;
    kargs.reduce_indptr_size = plan.num_reduce_tiles + 1;
    kargs.auto_split = 0;
    kargs.is_causal = 1;
    return kargs;
}

template<class Traits, class KArgs>
inline void mla_decode_launch_pipeline(Traits, const KArgs& kargs, const mla_decode_plan& plan,
                                       dim3 grid_main, dim3 block_main, bool run_metadata = true) {
    if (run_metadata)
        get_mla_metadata_kernel<<<dim3(1), dim3(Traits::WARP_SIZE)>>>(
            mla_decode_metadata_kargs(plan, kargs.kv_indptr));

    mla_decode_launch(Traits{}, kargs, grid_main, block_main);

    opus_mla_decode_reduce_kargs rargs{};
    rargs.o_accum = plan.o_accum;
    rargs.lse_accum = plan.lse_accum;
    rargs.out_ptr = kargs.out_ptr;
    rargs.lse_ptr = kargs.lse_ptr;
    rargs.reduce_indptr = plan.reduce_indptr;
    rargs.reduce_final_map = plan.reduce_final_map;
    rargs.reduce_partial_map = plan.reduce_partial_map;
    rargs.H = plan.H;
    rargs.stride_o_b = kargs.stride_o_b;
    rargs.stride_o_h = kargs.stride_o_h;

    dim3 grid_reduce(plan.num_reduce_tiles, ceil_div(plan.H, MLA_DECODE_REDUCE_HEADS_PER_BLOCK), 1);
    dim3 block_reduce(MLA_DECODE_REDUCE_HEADS_PER_BLOCK * Traits::WARP_SIZE);
    mla_reduce_kernel<Traits><<<grid_reduce, block_reduce>>>(rargs);
}

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

inline mla_decode_plan mla_decode_plan_create(int B, int H, int seqlen_qo, int D_VO) {
    mla_decode_plan plan;
    plan.B = B;
    plan.H = H;
    plan.seqlen_qo = seqlen_qo;
    plan.num_qo_tiles = mla_decode_num_qo_tiles(seqlen_qo, H);
    plan.num_reduce_tiles = B * plan.num_qo_tiles;
    plan.num_splits = std::min(MLA_DECODE_NUM_CU, MLA_DECODE_MAX_SPLIT_PER_BATCH * B);
    plan.max_works = (B + MLA_DECODE_NUM_CU - 1) * plan.num_qo_tiles;
    plan.max_split_tiles =
        std::max(std::min(B + MLA_DECODE_NUM_CU - 1, (MLA_DECODE_NUM_CU - 1) * 2) * plan.num_qo_tiles,
                 plan.num_reduce_tiles + plan.num_splits);
    plan.num_partial_rows = plan.max_split_tiles * seqlen_qo;

    std::vector<int> host_qo_indptr(B + 1);
    for (int b = 0; b <= B; ++b) host_qo_indptr[b] = b * seqlen_qo;

    CHECK_HIP(hipMalloc(&plan.qo_indptr, host_qo_indptr.size() * sizeof(int)));
    CHECK_HIP(hipMemcpy(plan.qo_indptr, host_qo_indptr.data(), host_qo_indptr.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMalloc(&plan.work_indptr, (MLA_DECODE_NUM_CU + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&plan.work_info_set, (size_t)plan.max_works * sizeof(opus_mla_decode_work_info)));
    CHECK_HIP(hipMalloc(&plan.reduce_indptr, (plan.num_reduce_tiles + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&plan.reduce_final_map, (size_t)plan.num_reduce_tiles * 2 * sizeof(int)));
    CHECK_HIP(hipMalloc(&plan.reduce_partial_map, (size_t)plan.max_split_tiles * sizeof(int)));
    CHECK_HIP(hipMalloc(&plan.o_accum, (size_t)plan.num_partial_rows * H * D_VO * sizeof(float)));
    CHECK_HIP(hipMalloc(&plan.lse_accum, (size_t)plan.num_partial_rows * H * sizeof(float)));
    return plan;
}

inline void mla_decode_plan_destroy(const mla_decode_plan& plan) {
    CHECK_HIP(hipFree(plan.qo_indptr));
    CHECK_HIP(hipFree(plan.work_indptr));
    CHECK_HIP(hipFree(plan.work_info_set));
    CHECK_HIP(hipFree(plan.reduce_indptr));
    CHECK_HIP(hipFree(plan.reduce_final_map));
    CHECK_HIP(hipFree(plan.reduce_partial_map));
    CHECK_HIP(hipFree(plan.o_accum));
    CHECK_HIP(hipFree(plan.lse_accum));
}

static constexpr size_t MLA_DECODE_INIT_GRAIN = 65536;
static constexpr uint32_t MLA_DECODE_INIT_SEED = 2026;

template<typename T>
void rand_vector(T* ptr, size_t size) {
    mla_decode::parallel_for(ceil_div(size, MLA_DECODE_INIT_GRAIN), [&](size_t c) {
        const size_t begin = c * MLA_DECODE_INIT_GRAIN;
        const size_t end = std::min(begin + MLA_DECODE_INIT_GRAIN, size);
        std::mt19937 gen(MLA_DECODE_INIT_SEED + static_cast<uint32_t>(c));
        std::normal_distribution<float> dis(0.0f, 1.0f);
        for (size_t i = begin; i < end; i++) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    });
}

template<class PATraits>
void init_fp8_mla_split(typename PATraits::D_NOPE* nope_ptr,
                        uint8_t* scale_ptr,
                        typename PATraits::D_ROPE* rope_ptr, size_t rows) {
    using D_ROPE = typename PATraits::D_ROPE;
    constexpr int NOPE  = PATraits::D_NOPE_SIZE;
    constexpr int SCALE = PATraits::D_SCALE_SIZE;
    constexpr int ROPE  = PATraits::D_ROPE_SIZE;
    constexpr int BLOCK = NOPE / SCALE;

    mla_decode::parallel_for(ceil_div(rows, MLA_DECODE_INIT_GRAIN), [&](size_t c) {
        const size_t begin = c * MLA_DECODE_INIT_GRAIN;
        const size_t end = std::min(begin + MLA_DECODE_INIT_GRAIN, rows);
        std::mt19937 gen(MLA_DECODE_INIT_SEED + static_cast<uint32_t>(c));
        std::normal_distribution<float> dis(0.0f, 1.0f);
        std::uniform_int_distribution<int> scale_exp_dis(0, 3);
        for (size_t r = begin; r < end; r++) {
            auto* nope = reinterpret_cast<fp8_t*>(nope_ptr) + r * NOPE;
            uint8_t* scale = scale_ptr + r * SCALE;
            for (int b = 0; b < SCALE; b++) {
                const int e = scale_exp_dis(gen);
                scale[b] = static_cast<uint8_t>(e + 127);
                const float inv_scale = std::ldexp(1.0f, -e);
                for (int i = 0; i < BLOCK; i++)
                    nope[b * BLOCK + i] = float_to_fp8_e4m3(dis(gen) * inv_scale);
            }
            D_ROPE* rope = rope_ptr + r * ROPE;
            for (int i = 0; i < ROPE; i++) rope[i] = static_cast<D_ROPE>(dis(gen));
        }
    });
}

void init_kv_page_table(std::vector<int>& kv_indptr,
                        std::vector<int>& kv_indices,
                        int B,
                        int s,
                        bool varlen,
                        uint32_t seed = 1234) {
    assert(B >= 0);
    assert(s > 0);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> len_dis(std::min(5, s), s);

    kv_indptr.assign(B + 1, 0);
    for (int b = 0; b < B; ++b) {
        const size_t end = static_cast<size_t>(kv_indptr[b]) + (varlen ? len_dis(gen) : s);
        assert(end <= static_cast<size_t>(std::numeric_limits<int>::max()));
        kv_indptr[b + 1] = static_cast<int>(end);
    }

    kv_indices.resize(static_cast<size_t>(kv_indptr[B]));
    std::iota(kv_indices.begin(), kv_indices.end(), 0);
    std::shuffle(kv_indices.begin(), kv_indices.end(), gen);
}

template<class Traits, class KArgs>
void benchmark_mla_decode_kernel(const KArgs& kargs, const mla_decode_plan& plan, dim3 grid, dim3 block,
                                 int total_kv, int warmup = 100, int iterations = 50) {
    get_mla_metadata_kernel<<<dim3(1), dim3(Traits::WARP_SIZE)>>>(
        mla_decode_metadata_kargs(plan, kargs.kv_indptr));
    CHECK_HIP_KERNEL_LAUNCH();
    CHECK_HIP(hipDeviceSynchronize());

    for (int i = 0; i < warmup; ++i) {
        mla_decode_launch_pipeline(Traits{}, kargs, plan, grid, block, false);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        mla_decode_launch_pipeline(Traits{}, kargs, plan, grid, block, false);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;

    using D_OUT  = typename Traits::D_OUT;
    constexpr int D_QK = Traits::D_QK_SIZE;
    constexpr int D_VO = Traits::D_VO_SIZE;

    const int total_q = plan.B * plan.seqlen_qo;
    const double flops = 2.0 * plan.seqlen_qo * kargs.H * total_kv * (D_QK + D_VO);
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    constexpr size_t row_bytes = []() -> size_t {
        if constexpr (requires { Traits::D_NOPE_SIZE; })
            return Traits::D_NOPE_SIZE * sizeof(typename Traits::D_NOPE)
                 + Traits::D_SCALE_SIZE * sizeof(uint8_t)
                 + Traits::D_ROPE_SIZE * sizeof(typename Traits::D_ROPE);
        else
            return Traits::D_QK_SIZE * sizeof(typename Traits::D_ATTN);
    }();
    const size_t q_bytes  = (size_t)total_q * kargs.H * row_bytes;
    const size_t o_bytes  = (size_t)total_q * kargs.H * D_VO * sizeof(D_OUT);
    const size_t kv_bytes = (size_t)total_kv * row_bytes;
    const double tbps = double(q_bytes + o_bytes + kv_bytes) / (avg_time * 1e-3) / 1e12;

    printf("MLA decode performance: avg_time=%.3f ms, %.2f TFlops, %.2f TB/s\n",
           avg_time, tflops, tbps);
}

template<typename DType>
bool validate_mla_decode_results(const DType* ref, const DType* gpu,
                                 int total_q, int H, int D, int seqlen_qo,
                                 float rtol = 1e-2f, float atol = 1e-2f,
                                 float tol_err_ratio = 0.05f) {
    const size_t total_elements = (size_t)total_q * H * D;
    constexpr size_t printNum = 10;

    size_t total_errors = 0, printed = 0;
    bool any_nan = false;
    float max_abs_delta = 0.0f, ref_absmax = 0.0f;
    double sq_diff_sum = 0.0, ref_sq_sum = 0.0;

    for (int q = 0; q < total_q; q++) {
        for (int h = 0; h < H; h++) {
            const size_t offset = ((size_t)q * H + h) * D;
            for (int d = 0; d < D; d++) {
                const float ref_val = static_cast<float>(ref[offset + d]);
                const float gpu_val = static_cast<float>(gpu[offset + d]);
                const float delta   = std::abs(gpu_val - ref_val);

                ref_absmax  = std::max(ref_absmax, std::abs(ref_val));
                sq_diff_sum += double(delta) * double(delta);
                ref_sq_sum  += double(ref_val) * double(ref_val);

                const bool nan_inf = std::isnan(gpu_val) || std::isinf(gpu_val);
                any_nan |= nan_inf;
                if (nan_inf || delta > atol + rtol * std::abs(ref_val)) {
                    total_errors++;
                    max_abs_delta = std::max(max_abs_delta, delta);
                    if (printed++ < printNum)
                        printf("  mismatch [q=%d(b=%d,t=%d),h=%d,d=%d] ref=%.6f gpu=%.6f delta=%.6f\n",
                               q, q / seqlen_qo, q % seqlen_qo, h, d, ref_val, gpu_val, delta);
                }
            }
        }
    }

    const double err_ratio    = double(total_errors) / double(total_elements);
    const double nrms         = std::sqrt(sq_diff_sum / std::max(ref_sq_sum, 1e-12));
    const bool   catastrophic = any_nan || max_abs_delta > 0.5f * ref_absmax;
    const bool   all_valid    = !catastrophic && err_ratio <= tol_err_ratio;

    printf("  rtol=%.0e atol=%.0e | max_abs_delta=%.6f nrms=%.3e | mismatch %zu/%zu (%.2f%%)\n",
           rtol, atol, max_abs_delta, nrms, total_errors, total_elements, 100.0 * err_ratio);
    if (all_valid)
        printf("✓ Validation passed (checked %zu elements)\n", total_elements);
    else if (catastrophic)
        printf("✗ Validation failed (catastrophic: %s, max_abs_delta=%.6f)\n",
               any_nan ? "NaN/Inf" : "delta > 0.5*max|ref|", max_abs_delta);
    else
        printf("✗ Validation failed (mismatch ratio %.2f%% > %.2f%%)\n",
               100.0 * err_ratio, 100.0 * tol_err_ratio);

    return all_valid;
}

bool validate_mla_decode_lse(const float* ref, const float* gpu, int total_q, int H, int seqlen_qo,
                             float rtol = 1e-3f, float atol = 1e-3f) {
    const size_t total = (size_t)total_q * H;
    constexpr size_t printNum = 10;

    size_t errors = 0, printed = 0;
    float max_abs_delta = 0.0f;

    for (int q = 0; q < total_q; q++) {
        for (int h = 0; h < H; h++) {
            const float r = ref[(size_t)q * H + h];
            const float g = gpu[(size_t)q * H + h];
            if (std::isinf(r) || std::isinf(g)) {
                if (std::isinf(r) && std::isinf(g) && std::signbit(r) == std::signbit(g)) continue;
                errors++;
                if (printed++ < printNum)
                    printf("  lse mismatch [q=%d(b=%d,t=%d),h=%d] ref=%f gpu=%f\n",
                           q, q / seqlen_qo, q % seqlen_qo, h, r, g);
                continue;
            }
            const float delta = std::abs(g - r);
            max_abs_delta = std::max(max_abs_delta, delta);
            if (std::isnan(g) || delta > atol + rtol * std::abs(r)) {
                errors++;
                if (printed++ < printNum)
                    printf("  lse mismatch [q=%d(b=%d,t=%d),h=%d] ref=%.6f gpu=%.6f delta=%.6f\n",
                           q, q / seqlen_qo, q % seqlen_qo, h, r, g, delta);
            }
        }
    }

    printf("  LSE: rtol=%.0e atol=%.0e | max_abs_delta=%.6f | mismatch %zu/%zu\n",
           rtol, atol, max_abs_delta, errors, total);
    if (errors == 0) printf("✓ LSE validation passed (checked %zu elements)\n", total);
    else             printf("✗ LSE validation failed\n");

    return errors == 0;
}

template<class PATraits>
inline void dequant_mla_row_fp8(const typename PATraits::D_NOPE* nrow,
                                const uint8_t* srow,
                                const typename PATraits::D_ROPE* rrow, float* out) {
    constexpr int NOPE = PATraits::D_NOPE_SIZE;
    constexpr int ROPE = PATraits::D_ROPE_SIZE;
    const auto* nope = reinterpret_cast<const fp8_t*>(nrow);
    for (int d = 0; d < NOPE; d++)
        out[d] = fp8_e4m3_to_float(nope[d]) * std::ldexp(1.0f, int(srow[d / 32]) - 127);
    for (int j = 0; j < ROPE; j++)
        out[NOPE + j] = static_cast<float>(rrow[j]);
}

template<class PATraits>
inline void mla_decode_attention_compute(const float* q_dense, const float* kv_dense, int num_rows,
                                         typename PATraits::D_OUT* o_row, float* lse_row) {
    using O_t = typename PATraits::D_OUT;
    constexpr int D_QK = PATraits::D_QK_SIZE;
    constexpr int D_VO = PATraits::D_VO_SIZE;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D_QK));

    std::vector<float> scores(num_rows);
    for (int p = 0; p < num_rows; p++) {
        const float* k = kv_dense + (size_t)p * D_QK;
        float dot = 0.0f;
        for (int d = 0; d < D_QK; d++) dot += q_dense[d] * k[d];
        scores[p] = dot * softmax_scale;
    }
    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum_exp = 0.0f;
    for (int p = 0; p < num_rows; p++) { scores[p] = std::exp(scores[p] - max_score); sum_exp += scores[p]; }
    *lse_row = std::log(sum_exp) + max_score;
    for (int p = 0; p < num_rows; p++)
        scores[p] = static_cast<float>(static_cast<bf16_t>(scores[p] / sum_exp));
    for (int d = 0; d < D_VO; d++) {
        float acc = 0.0f;
        for (int p = 0; p < num_rows; p++) acc += scores[p] * kv_dense[(size_t)p * D_QK + d];
        o_row[d] = static_cast<O_t>(acc);
    }
}

template<class PATraits>
void mla_decode_attention_ref_fp8(
    const typename PATraits::D_NOPE* Q_nope, const uint8_t* Q_scale, const typename PATraits::D_ROPE* Q_rope,
    const typename PATraits::D_NOPE* KV_nope, const uint8_t* KV_scale, const typename PATraits::D_ROPE* KV_rope,
    typename PATraits::D_OUT* O, float* LSE,
    const int* kv_indptr, const int* kv_indices,
    int B, int H, int seqlen_qo)
{
    using O_t = typename PATraits::D_OUT;
    constexpr int D_QK   = PATraits::D_QK_SIZE;
    constexpr int D_VO   = PATraits::D_VO_SIZE;
    constexpr int NOPE   = PATraits::D_NOPE_SIZE;
    constexpr int SCALE  = PATraits::D_SCALE_SIZE;
    constexpr int ROPE   = PATraits::D_ROPE_SIZE;
    const int total_q    = B * seqlen_qo;
    const int o_stride_n = H * D_VO;
    const int o_stride_h = D_VO;

    mla_decode::parallel_for((size_t)H * total_q, [&](size_t idx) {
        const int h = static_cast<int>(idx / total_q);
        const int i = static_cast<int>(idx % total_q);
        const int b = i / seqlen_qo;
        const int kv_begin = kv_indptr[b];
        const int num_rows = kv_indptr[b + 1] - (seqlen_qo - 1 - i % seqlen_qo) - kv_begin;

        O_t* o_row = O + (size_t)i * o_stride_n + h * o_stride_h;
        float* lse_row = LSE + (size_t)i * H + h;
        if (num_rows <= 0) {
            for (int d = 0; d < D_VO; d++) o_row[d] = static_cast<O_t>(0.0f);
            *lse_row = std::numeric_limits<float>::infinity();
            return;
        }

        std::vector<float> q_dense(D_QK);
        const size_t q_row = (size_t)i * H + h;
        dequant_mla_row_fp8<PATraits>(Q_nope + q_row * NOPE, Q_scale + q_row * SCALE, Q_rope + q_row * ROPE, q_dense.data());

        std::vector<float> kv_dense((size_t)num_rows * D_QK);
        for (int p = 0; p < num_rows; p++) {
            const int kv_row = kv_indices[kv_begin + p];
            dequant_mla_row_fp8<PATraits>(KV_nope + (size_t)kv_row * NOPE, KV_scale + (size_t)kv_row * SCALE,
                                          KV_rope + (size_t)kv_row * ROPE,
                                          kv_dense.data() + (size_t)p * D_QK);
        }

        mla_decode_attention_compute<PATraits>(q_dense.data(), kv_dense.data(), num_rows, o_row, lse_row);
    });
}

template<class PATraits>
void mla_decode_attention_ref_bf16(
    const typename PATraits::D_ATTN* Q, const typename PATraits::D_ATTN* KV,
    typename PATraits::D_OUT* O, float* LSE,
    const int* kv_indptr, const int* kv_indices,
    int B, int H, int seqlen_qo)
{
    using O_t = typename PATraits::D_OUT;
    constexpr int D_VO = PATraits::D_VO_SIZE;
    constexpr int D_QK   = PATraits::D_QK_SIZE;
    const int total_q    = B * seqlen_qo;
    const int o_stride_n = H * D_VO;

    mla_decode::parallel_for((size_t)H * total_q, [&](size_t idx) {
        const int h = static_cast<int>(idx / total_q);
        const int i = static_cast<int>(idx % total_q);
        const int b = i / seqlen_qo;
        const int kv_begin = kv_indptr[b];
        const int num_rows = kv_indptr[b + 1] - (seqlen_qo - 1 - i % seqlen_qo) - kv_begin;

        O_t* o_row = O + (size_t)i * o_stride_n + h * D_VO;
        float* lse_row = LSE + (size_t)i * H + h;
        if (num_rows <= 0) {
            for (int d = 0; d < D_VO; d++) o_row[d] = static_cast<O_t>(0.0f);
            *lse_row = std::numeric_limits<float>::infinity();
            return;
        }

        std::vector<float> q_dense(D_QK);
        const typename PATraits::D_ATTN* q_src = Q + ((size_t)i * H + h) * D_QK;
        for (int d = 0; d < D_QK; d++) q_dense[d] = static_cast<float>(q_src[d]);

        std::vector<float> kv_dense((size_t)num_rows * D_QK);
        for (int p = 0; p < num_rows; p++) {
            const typename PATraits::D_ATTN* k_src = KV + (size_t)kv_indices[kv_begin + p] * D_QK;
            for (int d = 0; d < D_QK; d++) kv_dense[(size_t)p * D_QK + d] = static_cast<float>(k_src[d]);
        }

        mla_decode_attention_compute<PATraits>(q_dense.data(), kv_dense.data(), num_rows, o_row, lse_row);
    });
}

template<class PATraits>
int run_mla_decode_case_fp8(int H, int B, int s, int s_q, bool verify, bool varlen) {
    using OType = typename PATraits::D_OUT;
    constexpr int D_VO = PATraits::D_VO_SIZE;
    const int total_q = B * s_q;
    const size_t o_size = (size_t)total_q * H * D_VO;

    auto host_o_ref = std::make_unique<OType[]>(o_size);
    auto host_o_gpu = std::make_unique<OType[]>(o_size);
    const size_t lse_size = (size_t)total_q * H;
    auto host_lse_ref = std::make_unique<float[]>(lse_size);
    auto host_lse_gpu = std::make_unique<float[]>(lse_size);

    std::vector<int> host_kv_indptr, host_kv_indices;
    init_kv_page_table(host_kv_indptr, host_kv_indices, B, s, varlen);
    const size_t num_page = host_kv_indices.size();
    assert(num_page <= static_cast<size_t>(std::numeric_limits<int>::max()));
    const int total_kv_count = static_cast<int>(num_page);

    printf("MLA decode attention: H_Q=%d, B=%d, S_Q=%d, S=%d%s, total_kv=%d, D_QK=%d, D_VO=%d, %s\n",
           H, B, s_q, s, varlen ? "(max)" : "", total_kv_count,
           PATraits::D_QK_SIZE, D_VO, "NoPE=fp8, RoPE=bf16");

    OType *dev_o;
    float *dev_lse;
    int *dev_kv_indptr, *dev_kv_indices;
    const size_t kv_indices_alloc_size = std::max<size_t>(host_kv_indices.size(), 1);
    CHECK_HIP(hipMalloc(&dev_o, o_size * sizeof(OType)));
    CHECK_HIP(hipMalloc(&dev_lse, lse_size * sizeof(float)));
    CHECK_HIP(hipMalloc(&dev_kv_indptr, host_kv_indptr.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indices, kv_indices_alloc_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(dev_kv_indptr, host_kv_indptr.data(), host_kv_indptr.size() * sizeof(int), hipMemcpyHostToDevice));
    if (!host_kv_indices.empty())
        CHECK_HIP(hipMemcpy(dev_kv_indices, host_kv_indices.data(), host_kv_indices.size() * sizeof(int), hipMemcpyHostToDevice));

    const int num_h_blocks = ceil_div(H, PATraits::Q_TILE_SIZE * PATraits::T_M);
    const auto plan = mla_decode_plan_create(B, H, s_q, PATraits::D_VO_SIZE);

    dim3 grid(MLA_DECODE_NUM_CU, num_h_blocks, 1);
    dim3 block(PATraits::BLOCK_SIZE);
    printf("MLA decode launch config: grid=(%d,%d,%d) block=%d\n",
           grid.x, grid.y, grid.z, block.x);

    int rc = 0;
    auto verify_and_bench = [&](const auto& kargs) {
        mla_decode_launch_pipeline(PATraits{}, kargs, plan, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
        if (verify) {
            printf("\nValidating GPU results against CPU reference...\n");
            CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, o_size * sizeof(OType), hipMemcpyDeviceToHost));
            CHECK_HIP(hipMemcpy(host_lse_gpu.get(), dev_lse, lse_size * sizeof(float), hipMemcpyDeviceToHost));
            bool all_valid = validate_mla_decode_results<OType>(host_o_ref.get(), host_o_gpu.get(), total_q, H, D_VO, s_q);
            all_valid &= validate_mla_decode_lse(host_lse_ref.get(), host_lse_gpu.get(), total_q, H, s_q);
            printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
            if (!all_valid) rc = 1;
        }
        if (!rc) {
            printf("\n");
            benchmark_mla_decode_kernel<PATraits>(kargs, plan, grid, block, total_kv_count);
            printf("\n");
        }
    };

    using D_NOPE = typename PATraits::D_NOPE;
    using D_ROPE = typename PATraits::D_ROPE;
    constexpr int NOPE = PATraits::D_NOPE_SIZE;
    constexpr int SCALE = PATraits::D_SCALE_SIZE;
    constexpr int ROPE = PATraits::D_ROPE_SIZE;
    const size_t q_nope_size = (size_t)total_q * H * NOPE, q_rope_size = (size_t)total_q * H * ROPE;
    const size_t kv_nope_size = num_page * NOPE, kv_rope_size = num_page * ROPE;
    const size_t q_scale_size = (size_t)total_q * H * SCALE, kv_scale_size = num_page * SCALE;

    auto host_q_nope = std::make_unique<D_NOPE[]>(q_nope_size);
    auto host_q_scale = std::make_unique<uint8_t[]>(q_scale_size);
    auto host_q_rope = std::make_unique<D_ROPE[]>(q_rope_size);
    auto host_kv_nope = std::make_unique<D_NOPE[]>(kv_nope_size);
    auto host_kv_scale = std::make_unique<uint8_t[]>(kv_scale_size);
    auto host_kv_rope = std::make_unique<D_ROPE[]>(kv_rope_size);
    init_fp8_mla_split<PATraits>(host_q_nope.get(), host_q_scale.get(), host_q_rope.get(), (size_t)total_q * H);
    init_fp8_mla_split<PATraits>(host_kv_nope.get(), host_kv_scale.get(), host_kv_rope.get(), num_page);

    D_NOPE *dev_q_nope, *dev_kv_nope;
    D_ROPE *dev_q_rope, *dev_kv_rope;
    uint8_t *dev_q_scale, *dev_kv_scale;
    CHECK_HIP(hipMalloc(&dev_q_nope, q_nope_size * sizeof(D_NOPE)));
    CHECK_HIP(hipMalloc(&dev_q_scale, q_scale_size * sizeof(uint8_t)));
    CHECK_HIP(hipMalloc(&dev_q_rope, q_rope_size * sizeof(D_ROPE)));
    CHECK_HIP(hipMalloc(&dev_kv_nope, kv_nope_size * sizeof(D_NOPE)));
    CHECK_HIP(hipMalloc(&dev_kv_scale, kv_scale_size * sizeof(uint8_t)));
    CHECK_HIP(hipMalloc(&dev_kv_rope, kv_rope_size * sizeof(D_ROPE)));
    CHECK_HIP(hipMemcpy(dev_q_nope, host_q_nope.get(), q_nope_size * sizeof(D_NOPE), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_q_scale, host_q_scale.get(), q_scale_size * sizeof(uint8_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_q_rope, host_q_rope.get(), q_rope_size * sizeof(D_ROPE), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_nope, host_kv_nope.get(), kv_nope_size * sizeof(D_NOPE), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_scale, host_kv_scale.get(), kv_scale_size * sizeof(uint8_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv_rope, host_kv_rope.get(), kv_rope_size * sizeof(D_ROPE), hipMemcpyHostToDevice));

    if (verify)
        mla_decode_attention_ref_fp8<PATraits>(host_q_nope.get(), host_q_scale.get(), host_q_rope.get(),
                                               host_kv_nope.get(), host_kv_scale.get(), host_kv_rope.get(),
                                               host_o_ref.get(), host_lse_ref.get(),
                                               host_kv_indptr.data(), host_kv_indices.data(), B, H, s_q);

    opus_mla_decode_mxfp8_kargs kargs{};
    kargs.q_nope_ptr = dev_q_nope;
    kargs.q_scale_ptr = dev_q_scale;
    kargs.q_rope_ptr = dev_q_rope;
    kargs.kv_nope_ptr = dev_kv_nope;
    kargs.kv_scale_ptr = dev_kv_scale;
    kargs.kv_rope_ptr = dev_kv_rope;
    kargs.out_ptr = dev_o;
    kargs.lse_ptr = dev_lse;
    kargs.kv_indptr = dev_kv_indptr;
    kargs.kv_indices = dev_kv_indices;
    kargs.o_accum = plan.o_accum;
    kargs.lse_accum = plan.lse_accum;
    kargs.q_indptr = plan.qo_indptr;
    kargs.work_indptr = plan.work_indptr;
    kargs.work_info_set = plan.work_info_set;
    kargs.H = H;
    kargs.total_tokens = total_kv_count;
    kargs.stride_q_nope_b = H * NOPE;
    kargs.stride_q_nope_h = NOPE;
    kargs.stride_q_scale_b = H * SCALE;
    kargs.stride_q_scale_h = SCALE;
    kargs.stride_q_rope_b = H * ROPE;
    kargs.stride_q_rope_h = ROPE;
    kargs.stride_o_b = H * D_VO;
    kargs.stride_o_h = D_VO;
    kargs.stride_kv_nope_page = NOPE;
    kargs.stride_kv_scale_page = SCALE;
    kargs.stride_kv_rope_page = ROPE;
    kargs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(PATraits::D_QK_SIZE));

    verify_and_bench(kargs);

    CHECK_HIP(hipFree(dev_q_nope));   CHECK_HIP(hipFree(dev_q_rope));   CHECK_HIP(hipFree(dev_q_scale));
    CHECK_HIP(hipFree(dev_kv_nope));  CHECK_HIP(hipFree(dev_kv_rope));  CHECK_HIP(hipFree(dev_kv_scale));

    mla_decode_plan_destroy(plan);

    CHECK_HIP(hipFree(dev_o));
    CHECK_HIP(hipFree(dev_lse));
    CHECK_HIP(hipFree(dev_kv_indptr));
    CHECK_HIP(hipFree(dev_kv_indices));

    return rc;
}

template<class PATraits>
int run_mla_decode_case_bf16(int H, int B, int s, int s_q, bool verify, bool varlen) {
    using OType = typename PATraits::D_OUT;
    constexpr int D_VO = PATraits::D_VO_SIZE;
    const int total_q = B * s_q;
    const size_t o_size = (size_t)total_q * H * D_VO;

    auto host_o_ref = std::make_unique<OType[]>(o_size);
    auto host_o_gpu = std::make_unique<OType[]>(o_size);
    const size_t lse_size = (size_t)total_q * H;
    auto host_lse_ref = std::make_unique<float[]>(lse_size);
    auto host_lse_gpu = std::make_unique<float[]>(lse_size);

    std::vector<int> host_kv_indptr, host_kv_indices;
    init_kv_page_table(host_kv_indptr, host_kv_indices, B, s, varlen);
    const size_t num_page = host_kv_indices.size();
    assert(num_page <= static_cast<size_t>(std::numeric_limits<int>::max()));
    const int total_kv_count = static_cast<int>(num_page);

    printf("MLA decode attention: H_Q=%d, B=%d, S_Q=%d, S=%d%s, total_kv=%d, D_QK=%d, D_VO=%d, %s\n",
           H, B, s_q, s, varlen ? "(max)" : "", total_kv_count,
           PATraits::D_QK_SIZE, D_VO, "dtype=bf16");

    OType *dev_o;
    float *dev_lse;
    int *dev_kv_indptr, *dev_kv_indices;
    const size_t kv_indices_alloc_size = std::max<size_t>(host_kv_indices.size(), 1);
    CHECK_HIP(hipMalloc(&dev_o, o_size * sizeof(OType)));
    CHECK_HIP(hipMalloc(&dev_lse, lse_size * sizeof(float)));
    CHECK_HIP(hipMalloc(&dev_kv_indptr, host_kv_indptr.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&dev_kv_indices, kv_indices_alloc_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(dev_kv_indptr, host_kv_indptr.data(), host_kv_indptr.size() * sizeof(int), hipMemcpyHostToDevice));
    if (!host_kv_indices.empty())
        CHECK_HIP(hipMemcpy(dev_kv_indices, host_kv_indices.data(), host_kv_indices.size() * sizeof(int), hipMemcpyHostToDevice));

    const int num_h_blocks = ceil_div(H, PATraits::Q_TILE_SIZE * PATraits::T_M);
    const auto plan = mla_decode_plan_create(B, H, s_q, PATraits::D_VO_SIZE);

    dim3 grid(MLA_DECODE_NUM_CU, num_h_blocks, 1);
    dim3 block(PATraits::BLOCK_SIZE);
    printf("MLA decode launch config: grid=(%d,%d,%d) block=%d\n",
           grid.x, grid.y, grid.z, block.x);

    int rc = 0;
    auto verify_and_bench = [&](const auto& kargs) {
        mla_decode_launch_pipeline(PATraits{}, kargs, plan, grid, block);
        CHECK_HIP_KERNEL_LAUNCH();
        if (verify) {
            printf("\nValidating GPU results against CPU reference...\n");
            CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, o_size * sizeof(OType), hipMemcpyDeviceToHost));
            CHECK_HIP(hipMemcpy(host_lse_gpu.get(), dev_lse, lse_size * sizeof(float), hipMemcpyDeviceToHost));
            bool all_valid = validate_mla_decode_results<OType>(host_o_ref.get(), host_o_gpu.get(), total_q, H, D_VO, s_q);
            all_valid &= validate_mla_decode_lse(host_lse_ref.get(), host_lse_gpu.get(), total_q, H, s_q);
            printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
            if (!all_valid) rc = 1;
        }
        if (!rc) {
            printf("\n");
            benchmark_mla_decode_kernel<PATraits>(kargs, plan, grid, block, total_kv_count);
            printf("\n");
        }
    };

    using D_ATTN = typename PATraits::D_ATTN;
    constexpr int D_QK = PATraits::D_QK_SIZE;
    const size_t q_size = (size_t)total_q * H * D_QK, kv_size = num_page * D_QK;

    auto host_q  = std::make_unique<D_ATTN[]>(q_size);
    auto host_kv = std::make_unique<D_ATTN[]>(kv_size);
    rand_vector(host_q.get(), q_size);
    rand_vector(host_kv.get(), kv_size);

    D_ATTN *dev_q, *dev_kv;
    CHECK_HIP(hipMalloc(&dev_q, q_size * sizeof(D_ATTN)));
    CHECK_HIP(hipMalloc(&dev_kv, kv_size * sizeof(D_ATTN)));
    CHECK_HIP(hipMemcpy(dev_q, host_q.get(), q_size * sizeof(D_ATTN), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_kv, host_kv.get(), kv_size * sizeof(D_ATTN), hipMemcpyHostToDevice));

    if (verify)
        mla_decode_attention_ref_bf16<PATraits>(host_q.get(), host_kv.get(),
                                                host_o_ref.get(), host_lse_ref.get(),
                                                host_kv_indptr.data(), host_kv_indices.data(), B, H, s_q);

    opus_mla_decode_kargs kargs{};
    kargs.q_ptr = dev_q;
    kargs.kv_ptr = dev_kv;
    kargs.out_ptr = dev_o;
    kargs.lse_ptr = dev_lse;
    kargs.kv_indptr = dev_kv_indptr;
    kargs.kv_indices = dev_kv_indices;
    kargs.o_accum = plan.o_accum;
    kargs.lse_accum = plan.lse_accum;
    kargs.q_indptr = plan.qo_indptr;
    kargs.work_indptr = plan.work_indptr;
    kargs.work_info_set = plan.work_info_set;
    kargs.H = H;
    kargs.total_tokens = total_kv_count;
    kargs.stride_q_b = H * D_QK;
    kargs.stride_q_h = D_QK;
    kargs.stride_o_b = H * D_VO;
    kargs.stride_o_h = D_VO;
    kargs.stride_kv_page = D_QK;
    kargs.softmax_scale = 1.0f / std::sqrt(static_cast<float>(PATraits::D_QK_SIZE));

    verify_and_bench(kargs);

    CHECK_HIP(hipFree(dev_q));
    CHECK_HIP(hipFree(dev_kv));

    mla_decode_plan_destroy(plan);

    CHECK_HIP(hipFree(dev_o));
    CHECK_HIP(hipFree(dev_lse));
    CHECK_HIP(hipFree(dev_kv_indptr));
    CHECK_HIP(hipFree(dev_kv_indices));

    return rc;
}

int main(int argc, char** argv) {
    int H = 128;
    int B = 128;
    int s = 1024;
    int s_q = 1;

    bool verify = false;
    bool varlen = false;
    const char* dtype = "fp8";
    auto parse_val = [](const char* arg, const char* flag) -> const char* {
        size_t len = std::strlen(flag);
        if (std::strncmp(arg, flag, len) == 0) {
            if (arg[len] == '=') return arg + len + 1;
            if (arg[len] == '\0') return reinterpret_cast<const char*>(1);
        }
        return nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* val;
        if (std::strcmp(arg, "--verify") == 0) { verify = true; continue; }
        if (std::strcmp(arg, "--varlen") == 0) { varlen = true; continue; }
        auto try_parse = [&](int& target, const char* flag) {
            if ((val = parse_val(arg, flag))) {
                if (val == reinterpret_cast<const char*>(1)) { if (i + 1 < argc) target = std::atoi(argv[++i]); }
                else target = std::atoi(val);
                return true;
            }
            return false;
        };
        auto try_parse_str = [&](const char*& target, const char* flag) {
            if ((val = parse_val(arg, flag))) {
                if (val == reinterpret_cast<const char*>(1)) { if (i + 1 < argc) target = argv[++i]; }
                else target = val;
                return true;
            }
            return false;
        };
        if (try_parse_str(dtype, "-dtype")) continue;
        if (try_parse(H, "-h_q")) continue;
        if (try_parse(B, "-b")) continue;
        if (try_parse(s, "-s")) continue;
        if (try_parse(s_q, "-s_q")) continue;
        std::cerr << "unknown argument '" << arg << "'\n";
        return 1;
    }

    if (H <= 0 || B <= 0 || s <= 0 || s_q <= 0) {
        std::cerr << "Invalid parameters. H_Q,B,S,S_Q must be positive.\n";
        return 1;
    }

    if (std::strcmp(dtype, "fp8") == 0)
        return run_mla_decode_case_fp8<
            opus_mla_decode_mxfp8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>>(
            H, B, s, s_q, verify, varlen);

    if (std::strcmp(dtype, "bf16") == 0) {
        if (H <= 32)
            return run_mla_decode_case_bf16<
                opus_mla_decode_a16w16_32mx1_16nx4_traits<32, 64, 4, bf16_t, bf16_t>>(
                H, B, s, s_q, verify, varlen);
        if (H % 128 == 0)
            return run_mla_decode_case_bf16<
                opus_mla_decode_a16w16_32mx4_32nx1_traits<32, 32, 4, bf16_t, bf16_t>>(
                H, B, s, s_q, verify, varlen);
        if (H % 96 == 0)
            return run_mla_decode_case_bf16<
                opus_mla_decode_a16w16_32mx3_32nx1_traits<32, 32, 4, bf16_t, bf16_t>>(
                H, B, s, s_q, verify, varlen);
        return run_mla_decode_case_bf16<
            opus_mla_decode_a16w16_16mx4_64nx1_traits<16, 64, 4, bf16_t, bf16_t>>(
            H, B, s, s_q, verify, varlen);
    }

    std::cerr << "unknown -dtype '" << dtype << "'; available: fp8, bf16\n";
    return 1;
}

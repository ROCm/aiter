// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "hipb_groupedgemm.cuh"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t kWorkspaceBytes = 256 * 1024 * 1024;
constexpr size_t kMultiStreamWorkspaceBytes = 64 * 1024 * 1024;

struct HipblasLtContext
{
    int device;
    hipblasLtHandle_t handle = nullptr;
    void* workspace          = nullptr;
    hipblaslt_ext::UserArguments* host_user_args   = nullptr;
    hipblaslt_ext::UserArguments* device_user_args = nullptr;
    size_t user_args_capacity                       = 0;
    hipEvent_t completion                            = nullptr;
    bool has_pending_work                            = false;

    explicit HipblasLtContext(int device_) : device(device_)
    {
        TORCH_CHECK(hipSetDevice(device) == hipSuccess,
                    "selecting device for grouped GEMM context failed");
        TORCH_CHECK(
            hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS,
            "hipblasLtCreate failed for grouped GEMM");
        TORCH_CHECK(
            hipMalloc(&workspace, kWorkspaceBytes) == hipSuccess,
            "hipMalloc failed for grouped GEMM workspace");
        TORCH_CHECK(hipEventCreate(&completion) == hipSuccess,
                    "hipEventCreate failed for grouped GEMM completion");
    }

    ~HipblasLtContext()
    {
        int previous_device = device;
        hipGetDevice(&previous_device);
        hipSetDevice(device);
        if(has_pending_work)
            hipEventSynchronize(completion);
        if(completion != nullptr)
            hipEventDestroy(completion);
        if(workspace != nullptr)
            hipFree(workspace);
        if(host_user_args != nullptr)
            hipHostFree(host_user_args);
        if(device_user_args != nullptr)
            hipFree(device_user_args);
        if(handle != nullptr)
            hipblasLtDestroy(handle);
        hipSetDevice(previous_device);
    }

    void wait_for_completion()
    {
        if(!has_pending_work)
            return;
        TORCH_CHECK(hipEventSynchronize(completion) == hipSuccess,
                    "waiting for grouped GEMM scratch reuse failed");
        has_pending_work = false;
    }

    void mark_pending(hipStream_t stream)
    {
        TORCH_CHECK(hipEventRecord(completion, stream) == hipSuccess,
                    "recording grouped GEMM completion failed");
        has_pending_work = true;
    }

    void reserve_user_args(size_t count)
    {
        if(count <= user_args_capacity)
            return;
        if(host_user_args != nullptr)
            TORCH_CHECK(hipHostFree(host_user_args) == hipSuccess,
                        "hipHostFree failed for grouped GEMM arguments");
        if(device_user_args != nullptr)
            TORCH_CHECK(hipFree(device_user_args) == hipSuccess,
                        "hipFree failed for grouped GEMM arguments");
        const size_t bytes = count * sizeof(hipblaslt_ext::UserArguments);
        TORCH_CHECK(hipHostMalloc(&host_user_args, bytes) == hipSuccess,
                    "hipHostMalloc failed for grouped GEMM arguments");
        TORCH_CHECK(hipMalloc(&device_user_args, bytes) == hipSuccess,
                    "hipMalloc failed for grouped GEMM arguments");
        user_args_capacity = count;
    }
};

thread_local std::unordered_map<int, std::unique_ptr<HipblasLtContext>> contexts;

struct MultiStreamContext
{
    static constexpr int kStreams = 4;
    int device;
    std::array<hipStream_t, kStreams> streams{};
    std::array<hipEvent_t, kStreams> done{};
    std::array<hipblasLtHandle_t, kStreams> handles{};
    std::array<void*, kStreams> workspaces{};
    std::array<hipblasLtMatmulPreference_t, kStreams> preferences{};
    hipEvent_t ready = nullptr;
    std::mutex launch_mutex;

    explicit MultiStreamContext(int device_) : device(device_)
    {
        int stream_priority = 0;
        if(const char* priority_env = std::getenv("SONIC_MOE_MULTISTREAM_PRIORITY"))
        {
            const std::string priority_value(priority_env);
            TORCH_CHECK(
                priority_value == "0" || priority_value == "-1",
                "SONIC_MOE_MULTISTREAM_PRIORITY must be 0 or -1");
            stream_priority = priority_value == "-1" ? -1 : 0;
        }
        TORCH_CHECK(hipSetDevice(device) == hipSuccess,
                    "selecting device for multi-stream hipBLASLt context failed");
        TORCH_CHECK(hipEventCreate(&ready) == hipSuccess,
                    "creating multi-stream ready event failed");
        for(int index = 0; index < kStreams; ++index)
        {
            TORCH_CHECK(
                hipStreamCreateWithPriority(
                    &streams[index], hipStreamNonBlocking, stream_priority) == hipSuccess,
                        "creating multi-stream BLAS stream failed");
            TORCH_CHECK(hipEventCreate(&done[index]) == hipSuccess,
                        "creating multi-stream completion event failed");
            TORCH_CHECK(hipblasLtCreate(&handles[index]) == HIPBLAS_STATUS_SUCCESS,
                        "creating per-stream hipBLASLt handle failed");
            TORCH_CHECK(hipMalloc(&workspaces[index], kMultiStreamWorkspaceBytes) == hipSuccess,
                        "allocating per-stream hipBLASLt workspace failed");
            TORCH_CHECK(
                hipblasLtMatmulPreferenceCreate(&preferences[index]) ==
                    HIPBLAS_STATUS_SUCCESS,
                "creating per-stream hipBLASLt preference failed");
            TORCH_CHECK(
                hipblasLtMatmulPreferenceSetAttribute(
                    preferences[index],
                    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &kMultiStreamWorkspaceBytes,
                    sizeof(kMultiStreamWorkspaceBytes)) == HIPBLAS_STATUS_SUCCESS,
                "setting per-stream hipBLASLt workspace preference failed");
        }
    }

    ~MultiStreamContext()
    {
        int previous_device = device;
        hipGetDevice(&previous_device);
        hipSetDevice(device);
        if(ready != nullptr)
            hipEventDestroy(ready);
        for(int index = 0; index < kStreams; ++index)
        {
            if(streams[index] != nullptr)
                hipStreamDestroy(streams[index]);
            if(done[index] != nullptr)
                hipEventDestroy(done[index]);
            if(preferences[index] != nullptr)
                hipblasLtMatmulPreferenceDestroy(preferences[index]);
            if(workspaces[index] != nullptr)
                hipFree(workspaces[index]);
            if(handles[index] != nullptr)
                hipblasLtDestroy(handles[index]);
        }
        hipSetDevice(previous_device);
    }
};

std::mutex multistream_contexts_mutex;
std::unordered_map<int, std::unique_ptr<MultiStreamContext>> multistream_contexts;

MultiStreamContext& get_multistream_context(int device)
{
    std::lock_guard<std::mutex> lock(multistream_contexts_mutex);
    auto& context_for_device = multistream_contexts[device];
    if(!context_for_device)
        context_for_device = std::make_unique<MultiStreamContext>(device);
    return *context_for_device;
}

struct GemmKey
{
    int device;
    hipblasOperation_t op_a;
    hipblasOperation_t op_b;
    hipDataType input_dtype;
    hipDataType output_dtype;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    bool has_bias;

    bool operator==(const GemmKey& other) const
    {
        return device == other.device && op_a == other.op_a && op_b == other.op_b &&
               input_dtype == other.input_dtype &&
               output_dtype == other.output_dtype && m == other.m && n == other.n &&
               k == other.k && lda == other.lda && ldb == other.ldb &&
               ldc == other.ldc && has_bias == other.has_bias;
    }
};

struct GemmKeyHash
{
    size_t operator()(const GemmKey& key) const
    {
        size_t seed = 0;
        const auto combine = [&seed](auto value) {
            seed ^= std::hash<decltype(value)>{}(value) + 0x9e3779b9 + (seed << 6) +
                    (seed >> 2);
        };
        combine(key.device);
        combine(static_cast<int>(key.op_a));
        combine(static_cast<int>(key.op_b));
        combine(static_cast<int>(key.input_dtype));
        combine(static_cast<int>(key.output_dtype));
        combine(key.m);
        combine(key.n);
        combine(key.k);
        combine(key.lda);
        combine(key.ldb);
        combine(key.ldc);
        combine(key.has_bias);
        return seed;
    }
};

using GemmAlgoCache =
    std::unordered_map<GemmKey, hipblasLtMatmulAlgo_t, GemmKeyHash>;
GemmAlgoCache multistream_algo_cache;
std::mutex multistream_algo_cache_mutex;

struct GemmDescriptors
{
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;
    hipblasLtMatmulDesc_t op   = nullptr;

    ~GemmDescriptors()
    {
        if(op != nullptr)
            hipblasLtMatmulDescDestroy(op);
        if(a != nullptr)
            hipblasLtMatrixLayoutDestroy(a);
        if(b != nullptr)
            hipblasLtMatrixLayoutDestroy(b);
        if(c != nullptr)
            hipblasLtMatrixLayoutDestroy(c);
    }
};

void run_multistream_hipblaslt_gemm(MultiStreamContext& context,
                                    int stream_index,
                                    hipDataType input_dtype,
                                    hipDataType output_dtype,
                                    hipblasOperation_t op_a,
                                    hipblasOperation_t op_b,
                                    int64_t m,
                                    int64_t n,
                                    int64_t k,
                                    const void* a,
                                    int64_t lda,
                                    const void* b,
                                    int64_t ldb,
                                    void* c,
                                    int64_t ldc,
                                    const void* bias)
{
    auto handle     = context.handles[stream_index];
    auto preference = context.preferences[stream_index];
    auto stream     = context.streams[stream_index];
    GemmDescriptors desc;
    const int64_t a_rows = op_a == HIPBLAS_OP_N ? m : k;
    const int64_t a_cols = op_a == HIPBLAS_OP_N ? k : m;
    const int64_t b_rows = op_b == HIPBLAS_OP_N ? k : n;
    const int64_t b_cols = op_b == HIPBLAS_OP_N ? n : k;
    TORCH_CHECK(
        hipblasLtMatrixLayoutCreate(&desc.a, input_dtype, a_rows, a_cols, lda) ==
            HIPBLAS_STATUS_SUCCESS,
        "creating hipBLASLt A layout failed");
    TORCH_CHECK(
        hipblasLtMatrixLayoutCreate(&desc.b, input_dtype, b_rows, b_cols, ldb) ==
            HIPBLAS_STATUS_SUCCESS,
        "creating hipBLASLt B layout failed");
    TORCH_CHECK(
        hipblasLtMatrixLayoutCreate(&desc.c, output_dtype, m, n, ldc) ==
            HIPBLAS_STATUS_SUCCESS,
        "creating hipBLASLt output layout failed");
    TORCH_CHECK(
        hipblasLtMatmulDescCreate(&desc.op, HIPBLAS_COMPUTE_32F, HIP_R_32F) ==
            HIPBLAS_STATUS_SUCCESS,
        "creating hipBLASLt matmul descriptor failed");
    TORCH_CHECK(
        hipblasLtMatmulDescSetAttribute(
            desc.op, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)) ==
            HIPBLAS_STATUS_SUCCESS,
        "setting hipBLASLt TRANSA failed");
    TORCH_CHECK(
        hipblasLtMatmulDescSetAttribute(
            desc.op, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)) ==
            HIPBLAS_STATUS_SUCCESS,
        "setting hipBLASLt TRANSB failed");
    if(bias != nullptr)
    {
        TORCH_CHECK(
            hipblasLtMatmulDescSetAttribute(
                desc.op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)) ==
                HIPBLAS_STATUS_SUCCESS,
            "setting hipBLASLt bias pointer failed");
        auto epilogue = HIPBLASLT_EPILOGUE_BIAS;
        TORCH_CHECK(
            hipblasLtMatmulDescSetAttribute(
                desc.op, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) ==
                HIPBLAS_STATUS_SUCCESS,
            "setting hipBLASLt bias epilogue failed");
    }

    const GemmKey key{context.device,
                      op_a,
                      op_b,
                      input_dtype,
                      output_dtype,
                      m,
                      n,
                      k,
                      lda,
                      ldb,
                      ldc,
                      bias != nullptr};
    hipblasLtMatmulAlgo_t algorithm;
    {
        std::lock_guard<std::mutex> lock(multistream_algo_cache_mutex);
        const auto cached = multistream_algo_cache.find(key);
        if(cached != multistream_algo_cache.end())
        {
            algorithm = cached->second;
        }
        else
        {
            hipblasLtMatmulHeuristicResult_t result{};
            int returned = 0;
            TORCH_CHECK(
                hipblasLtMatmulAlgoGetHeuristic(handle,
                                                desc.op,
                                                desc.a,
                                                desc.b,
                                                desc.c,
                                                desc.c,
                                                preference,
                                                1,
                                                &result,
                                                &returned) == HIPBLAS_STATUS_SUCCESS,
                "querying hipBLASLt algorithm failed");
            TORCH_CHECK(returned > 0,
                        "no hipBLASLt algorithm for GEMM ",
                        m,
                        "x",
                        n,
                        "x",
                        k);
            algorithm = result.algo;
            multistream_algo_cache.emplace(key, algorithm);
        }
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    TORCH_CHECK(
        hipblasLtMatmul(handle,
                       desc.op,
                       &alpha,
                       a,
                       desc.a,
                       b,
                       desc.b,
                       &beta,
                       c,
                       desc.c,
                       c,
                       desc.c,
                       &algorithm,
                       context.workspaces[stream_index],
                       kMultiStreamWorkspaceBytes,
                       stream) == HIPBLAS_STATUS_SUCCESS,
        "launching per-expert hipBLASLt GEMM failed");
}

HipblasLtContext& get_context(int device)
{
    auto& context_for_device = contexts[device];
    if(!context_for_device)
        context_for_device = std::make_unique<HipblasLtContext>(device);
    return *context_for_device;
}

struct GroupedProblem
{
    hipDataType dtype;
    std::vector<int64_t> m;
    std::vector<int64_t> n;
    std::vector<int64_t> setup_n;
    std::vector<int64_t> k;
    std::vector<int64_t> batch;
    std::vector<int64_t> lda;
    std::vector<int64_t> ldb;
    std::vector<int64_t> ldc;
    std::vector<int64_t> ldd;
    std::vector<int64_t> stride_a;
    std::vector<int64_t> stride_b;
    std::vector<int64_t> stride_c;
    std::vector<int64_t> stride_d;
    std::vector<hipblaslt_ext::GemmEpilogue> epilogues;
    std::vector<hipblaslt_ext::GemmInputs> inputs;
    std::vector<float> alphas;
    std::vector<float> betas;
    std::vector<int64_t> empty_experts;
};

GroupedProblem make_problem(const torch::Tensor& a,
                            const torch::Tensor& b,
                            const torch::Tensor& cu_seqlens,
                            const torch::Tensor& out,
                            bool a_is_transposed,
                            const std::optional<torch::Tensor>& bias)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "grouped GEMM tensors must be on GPU");
    TORCH_CHECK(b.get_device() == a.get_device() && out.get_device() == a.get_device(),
                "grouped GEMM tensors must be on the same GPU");
    TORCH_CHECK(!cu_seqlens.is_cuda() || cu_seqlens.get_device() == a.get_device(),
                "grouped GEMM offsets must be on CPU or the same GPU as A");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kHalf,
                "grouped GEMM supports BF16 and FP16");
    TORCH_CHECK(b.scalar_type() == a.scalar_type() && out.scalar_type() == a.scalar_type(),
                "grouped GEMM inputs and output must have the same dtype");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
                "grouped GEMM requires contiguous tensors");
    TORCH_CHECK(a.dim() == 2, "grouped GEMM A must be 2-D");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                "cu_seqlens must contain E+1 offsets");

    const int64_t experts = cu_seqlens.numel() - 1;
    if(a_is_transposed)
    {
        TORCH_CHECK(b.dim() == 2, "wgrad grouped GEMM B must be 2-D");
        TORCH_CHECK(b.size(0) == a.size(0), "wgrad grouped GEMM token dimensions must match");
        TORCH_CHECK(out.dim() == 3 && out.size(0) == experts,
                    "wgrad grouped GEMM output must be [E, K, N]");
        TORCH_CHECK(out.size(1) == a.size(1) && out.size(2) == b.size(1),
                    "wgrad grouped GEMM output shape is invalid");
    }
    else
    {
        TORCH_CHECK(b.dim() == 3 && b.size(0) == experts,
                    "forward grouped GEMM B must be stored as [E, N, K]");
        TORCH_CHECK(a.size(1) == b.size(2), "forward grouped GEMM K dimensions must match");
        TORCH_CHECK(out.dim() == 2 && out.size(0) == a.size(0) && out.size(1) == b.size(1),
                    "forward grouped GEMM output shape is invalid");
    }
    if(bias.has_value())
    {
        TORCH_CHECK(!a_is_transposed, "bias is not supported for wgrad grouped GEMM");
        TORCH_CHECK(bias->is_cuda() && bias->is_contiguous(), "grouped GEMM bias must be contiguous on GPU");
        TORCH_CHECK(bias->get_device() == a.get_device() &&
                        bias->scalar_type() == a.scalar_type(),
                    "grouped GEMM bias must match the input device and dtype");
        TORCH_CHECK(bias->dim() == 2 && bias->size(0) == experts && bias->size(1) == out.size(1),
                    "grouped GEMM bias must be [E, N]");
    }

    auto offsets = cu_seqlens.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64))
                       .contiguous();
    const auto* offset_data = offsets.data_ptr<int64_t>();
    TORCH_CHECK(offset_data[0] == 0 && offset_data[experts] == a.size(0),
                "cu_seqlens must span all rows of A");

    const size_t element_size = a.element_size();
    auto* a_base              = static_cast<char*>(a.data_ptr());
    auto* b_base              = static_cast<char*>(b.data_ptr());
    auto* out_base                   = static_cast<char*>(out.data_ptr());
    auto* bias_base = bias.has_value() ? static_cast<char*>(bias->data_ptr()) : nullptr;
    GroupedProblem problem;
    problem.dtype = a.scalar_type() == at::kBFloat16 ? HIP_R_16BF : HIP_R_16F;
    problem.m.reserve(experts);
    problem.n.reserve(experts);
    problem.k.reserve(experts);
    problem.batch.reserve(experts);
    problem.lda.reserve(experts);
    problem.ldb.reserve(experts);
    problem.ldc.reserve(experts);
    problem.ldd.reserve(experts);
    problem.stride_a.reserve(experts);
    problem.stride_b.reserve(experts);
    problem.stride_c.reserve(experts);
    problem.stride_d.reserve(experts);
    problem.epilogues.reserve(experts);
    problem.inputs.reserve(experts);
    problem.alphas.reserve(experts);
    problem.betas.reserve(experts);
    problem.empty_experts.reserve(experts);
    for(int64_t expert = 0; expert < experts; ++expert)
    {
        const int64_t begin = offset_data[expert];
        const int64_t end   = offset_data[expert + 1];
        TORCH_CHECK(begin <= end, "cu_seqlens must be nondecreasing");
        const int64_t rows = end - begin;
        if(rows == 0)
        {
            problem.empty_experts.push_back(expert);
            continue;
        }

        problem.epilogues.emplace_back();
        problem.epilogues.back().setMode(
            bias.has_value() ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT);
        if(bias.has_value())
            problem.epilogues.back().setBiasDataType(problem.dtype);
        problem.inputs.emplace_back();
        problem.alphas.push_back(1.0f);
        problem.betas.push_back(0.0f);
        auto& input = problem.inputs.back();
        input.setAlpha(&problem.alphas.back());
        input.setBeta(&problem.betas.back());

        if(a_is_transposed)
        {
            const int64_t output_k = a.size(1);
            const int64_t output_n = b.size(1);
            problem.m.push_back(output_n);
            problem.n.push_back(output_k);
            problem.k.push_back(rows);
            problem.lda.push_back(output_n);
            problem.ldb.push_back(output_k);
            problem.ldc.push_back(output_n);
            problem.ldd.push_back(output_n);
            input.setA(b_base + begin * output_n * element_size);
            input.setB(a_base + begin * output_k * element_size);
            input.setC(out_base + expert * output_k * output_n * element_size);
            input.setD(out_base + expert * output_k * output_n * element_size);
        }
        else
        {
            const int64_t input_k  = a.size(1);
            const int64_t output_n = b.size(1);
            problem.m.push_back(output_n);
            problem.n.push_back(rows);
            problem.k.push_back(input_k);
            problem.lda.push_back(input_k);
            problem.ldb.push_back(input_k);
            problem.ldc.push_back(output_n);
            problem.ldd.push_back(output_n);
            input.setA(b_base + expert * input_k * output_n * element_size);
            input.setB(a_base + begin * input_k * element_size);
            input.setC(out_base + begin * output_n * element_size);
            input.setD(out_base + begin * output_n * element_size);
            if(bias.has_value())
                input.setBias(
                    bias_base + expert * output_n * bias->element_size());
        }
        problem.batch.push_back(1);
        problem.stride_a.push_back(0);
        problem.stride_b.push_back(0);
        problem.stride_c.push_back(0);
        problem.stride_d.push_back(0);
    }
    problem.setup_n = problem.n;
    if(!a_is_transposed && !problem.setup_n.empty())
    {
        const int64_t total_n =
            std::accumulate(problem.n.begin(), problem.n.end(), int64_t{0});
        std::fill(problem.setup_n.begin(), problem.setup_n.end(), int64_t{1});
        problem.setup_n.front() = total_n;
    }
    return problem;
}

std::unique_ptr<hipblaslt_ext::GroupedGemm>
make_grouped_gemm(hipblasLtHandle_t handle,
                  GroupedProblem& problem,
                  bool a_is_transposed)
{
    auto grouped = std::make_unique<hipblaslt_ext::GroupedGemm>(
        handle,
        a_is_transposed ? HIPBLAS_OP_N : HIPBLAS_OP_T,
        a_is_transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N,
        problem.dtype,
        problem.dtype,
        problem.dtype,
        problem.dtype,
        HIPBLAS_COMPUTE_32F);
    grouped->setMaxWorkspaceBytes(kWorkspaceBytes);
    hipblaslt_ext::GemmProblemType problem_type(
        a_is_transposed ? HIPBLAS_OP_N : HIPBLAS_OP_T,
        a_is_transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N,
        problem.dtype,
        problem.dtype,
        problem.dtype,
        problem.dtype,
        HIPBLAS_COMPUTE_32F);
    const auto status = grouped->setProblem(
        problem.m,
        problem.setup_n,
        problem.k,
        problem.batch,
        problem.lda,
        problem.ldb,
        problem.ldc,
        problem.ldd,
        problem.stride_a,
        problem.stride_b,
        problem.stride_c,
        problem.stride_d,
        problem.epilogues,
        problem.inputs,
        problem_type);
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS,
                "hipBLASLt grouped setProblem failed: ",
                hipblasStatusToString(status));
    return grouped;
}

std::vector<hipblasLtMatmulHeuristicResult_t>
supported_algorithms(hipblasLtHandle_t handle,
                     hipblaslt_ext::GroupedGemm& grouped,
                     bool a_is_transposed,
                     hipDataType dtype,
                     int requested)
{
    hipblaslt_ext::GemmPreference preference;
    preference.setMaxWorkspaceBytes(kWorkspaceBytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    auto status = grouped.algoGetHeuristic(requested, preference, candidates);
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS,
                "hipBLASLt grouped heuristic query failed: ",
                hipblasStatusToString(status));

    const size_t heuristic_count = candidates.size();
    if(!candidates.empty())
        return candidates;

    auto status_all = hipblaslt_ext::getAllAlgos(
        handle,
        hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
        a_is_transposed ? HIPBLAS_OP_N : HIPBLAS_OP_T,
        a_is_transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N,
        dtype,
        dtype,
        dtype,
        dtype,
        HIPBLAS_COMPUTE_32F,
        candidates);
    TORCH_CHECK(status_all == HIPBLAS_STATUS_SUCCESS,
                "hipBLASLt grouped getAllAlgos failed: ",
                hipblasStatusToString(status_all));
    std::vector<hipblasLtMatmulHeuristicResult_t> supported;
    for(auto& candidate : candidates)
    {
        size_t workspace_bytes = 0;
        if(grouped.isAlgoSupported(candidate.algo, workspace_bytes) == HIPBLAS_STATUS_SUCCESS &&
           workspace_bytes <= kWorkspaceBytes)
        {
            candidate.workspaceSize = workspace_bytes;
            supported.push_back(candidate);
            if(static_cast<int>(supported.size()) >= requested)
                break;
        }
    }
    TORCH_CHECK(!supported.empty(),
                "no hipBLASLt grouped algorithm passed isAlgoSupported (heuristic candidates=",
                heuristic_count,
                ", all candidates=",
                candidates.size(),
                ")");
    return supported;
}

} // namespace

void hipb_grouped_mm(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const torch::Tensor& cu_seqlens,
                     torch::Tensor out,
                     bool a_is_transposed,
                     std::optional<torch::Tensor> bias,
                     int solution_index)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(a));
    auto problem = make_problem(a, b, cu_seqlens, out, a_is_transposed, bias);
    auto stream = torch::hip::getCurrentHIPStream().stream();
    if(a_is_transposed)
    {
        auto* out_data = static_cast<char*>(out.data_ptr());
        const size_t expert_bytes =
            out.size(1) * out.size(2) * out.element_size();
        for(const int64_t expert : problem.empty_experts)
            TORCH_CHECK(
                hipMemsetAsync(
                    out_data + expert * expert_bytes, 0, expert_bytes, stream) ==
                    hipSuccess,
                "zeroing empty-expert grouped wgrad failed");
    }
    if(problem.m.empty())
        return;

    auto& ctx    = get_context(a.get_device());
    ctx.wait_for_completion();
    auto grouped = make_grouped_gemm(ctx.handle, problem, a_is_transposed);
    hipblasLtMatmulAlgo_t algorithm;
    if(solution_index >= 0)
    {
        std::vector<int> indices{solution_index};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        const auto status = hipblaslt_ext::getAlgosFromIndex(ctx.handle, indices, results);
        TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS && !results.empty(),
                    "invalid hipBLASLt grouped solution index ",
                    solution_index);
        algorithm = results.front().algo;
    }
    else
    {
        auto algorithms =
            supported_algorithms(ctx.handle, *grouped, a_is_transposed, problem.dtype, 1);
        TORCH_CHECK(!algorithms.empty(), "no hipBLASLt grouped GEMM algorithm supports this problem");
        algorithm = algorithms.front().algo;
    }

    ctx.reserve_user_args(problem.m.size());
    grouped->getDefaultValueForDeviceUserArguments(ctx.host_user_args);
    for(size_t index = 0; index < problem.n.size(); ++index)
        ctx.host_user_args[index].n = problem.n[index];
    TORCH_CHECK(
        hipMemcpyAsync(ctx.device_user_args,
                       ctx.host_user_args,
                       problem.m.size() * sizeof(hipblaslt_ext::UserArguments),
                       hipMemcpyHostToDevice,
                       stream) == hipSuccess,
        "copying hipBLASLt grouped user arguments failed");
    ctx.mark_pending(stream);
    auto status = grouped->initialize(algorithm, ctx.workspace, true, stream);
    ctx.mark_pending(stream);
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS,
                "hipBLASLt grouped initialize failed: ",
                hipblasStatusToString(status));
    status = grouped->run(ctx.device_user_args, stream);
    ctx.mark_pending(stream);
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS,
                "hipBLASLt grouped run failed: ",
                hipblasStatusToString(status));
}

void hipb_multistream_mm(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& cu_seqlens,
                         torch::Tensor out,
                         bool a_is_transposed,
                         std::optional<torch::Tensor> bias,
                         bool b_is_transposed)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(a));
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(),
                "multi-stream GEMM tensors must be on GPU");
    TORCH_CHECK(b.get_device() == a.get_device() &&
                    out.get_device() == a.get_device(),
                "multi-stream GEMM tensors must be on the same GPU");
    TORCH_CHECK(!cu_seqlens.is_cuda() ||
                    cu_seqlens.get_device() == a.get_device(),
                "multi-stream offsets must be on CPU or the same GPU as A");
    TORCH_CHECK(a.dim() == 2 && a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
                "multi-stream GEMM requires contiguous tensors");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                "multi-stream GEMM input dtypes must match");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kHalf ||
                    a.scalar_type() == at::kFloat,
                "multi-stream GEMM supports BF16, FP16, and FP32");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf ||
                    out.scalar_type() == at::kFloat,
                "multi-stream GEMM output supports BF16, FP16, and FP32");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                "multi-stream offsets must contain E+1 values");
    auto offsets = cu_seqlens.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64))
                       .contiguous();
    const auto* offset_data = offsets.data_ptr<int64_t>();
    const int64_t experts   = offsets.numel() - 1;
    TORCH_CHECK(experts >= 1 && offset_data[0] == 0 && offset_data[experts] == a.size(0),
                "invalid multi-stream GEMM expert offsets");
    for(int64_t expert = 0; expert < experts; ++expert)
        TORCH_CHECK(offset_data[expert] <= offset_data[expert + 1],
                    "multi-stream GEMM offsets must be nondecreasing");
    if(a_is_transposed)
    {
        TORCH_CHECK(!b_is_transposed,
                    "multi-stream wgrad does not support a transposed B");
        TORCH_CHECK(!bias.has_value(),
                    "multi-stream wgrad does not support bias");
        TORCH_CHECK(b.dim() == 2 && b.size(0) == a.size(0),
                    "multi-stream wgrad B shape is invalid");
        TORCH_CHECK(out.dim() == 3 && out.size(0) == experts &&
                        out.size(1) == a.size(1) && out.size(2) == b.size(1),
                    "multi-stream wgrad output shape is invalid");
    }
    else
    {
        TORCH_CHECK(
            b.dim() == 3 && b.size(0) == experts &&
                (b_is_transposed
                     ? (b.size(1) == out.size(1) && b.size(2) == a.size(1))
                     : (b.size(1) == a.size(1) && b.size(2) == out.size(1))),
            "multi-stream forward B must be [E, K, N], or [E, N, K] "
            "when b_is_transposed");
        TORCH_CHECK(out.dim() == 2 && out.size(0) == a.size(0) &&
                        out.size(1) == (b_is_transposed ? b.size(1) : b.size(2)),
                    "multi-stream forward output shape is invalid");
        if(bias.has_value())
        {
            TORCH_CHECK(
                bias->is_cuda() && bias->is_contiguous() &&
                    bias->get_device() == a.get_device() &&
                    bias->scalar_type() == out.scalar_type(),
                "multi-stream bias must be contiguous and match the output device and dtype");
            TORCH_CHECK(bias->dim() == 2 && bias->size(0) == experts &&
                            bias->size(1) == out.size(1),
                        "multi-stream bias must be [E, N]");
        }
    }

    auto& ctx          = get_multistream_context(a.get_device());
    std::lock_guard<std::mutex> launch_lock(ctx.launch_mutex);
    auto current       = torch::hip::getCurrentHIPStream().stream();
    const int streams_used =
        std::min<int64_t>(experts, MultiStreamContext::kStreams);
    if(a_is_transposed)
    {
        auto* out_data = static_cast<char*>(out.data_ptr());
        const size_t expert_bytes =
            out.size(1) * out.size(2) * out.element_size();
        for(int64_t expert = 0; expert < experts; ++expert)
        {
            if(offset_data[expert + 1] == offset_data[expert])
            {
                TORCH_CHECK(
                    hipMemsetAsync(
                        out_data + expert * expert_bytes, 0, expert_bytes, current) ==
                        hipSuccess,
                    "zeroing empty-expert wgrad failed");
            }
        }
    }
    if(offset_data[experts] == 0)
        return;

    TORCH_CHECK(hipEventRecord(ctx.ready, current) == hipSuccess,
                "recording multi-stream ready event failed");
    for(int index = 0; index < streams_used; ++index)
        TORCH_CHECK(
            hipStreamWaitEvent(ctx.streams[index], ctx.ready, 0) == hipSuccess,
            "waiting for multi-stream GEMM inputs failed");

    const size_t input_element_size  = a.element_size();
    const size_t output_element_size = out.element_size();
    auto* a_base                     = static_cast<char*>(a.data_ptr());
    auto* b_base                     = static_cast<char*>(b.data_ptr());
    auto* out_base                   = static_cast<char*>(out.data_ptr());
    auto* bias_base =
        bias.has_value() ? static_cast<char*>(bias->data_ptr()) : nullptr;
    const auto input_dtype = a.scalar_type() == at::kBFloat16
                                 ? HIP_R_16BF
                                 : (a.scalar_type() == at::kHalf ? HIP_R_16F : HIP_R_32F);
    const auto output_dtype = out.scalar_type() == at::kBFloat16
                                  ? HIP_R_16BF
                                  : (out.scalar_type() == at::kHalf ? HIP_R_16F : HIP_R_32F);
    std::array<bool, MultiStreamContext::kStreams> launched{};
    try
    {
        for(int64_t expert = 0; expert < experts; ++expert)
        {
            const int64_t begin = offset_data[expert];
            const int64_t rows  = offset_data[expert + 1] - begin;
            if(rows == 0)
                continue;
            const int stream_index = expert % streams_used;
            if(a_is_transposed)
            {
                const int64_t output_k = a.size(1);
                const int64_t output_n = b.size(1);
                run_multistream_hipblaslt_gemm(
                    ctx,
                    stream_index,
                    input_dtype,
                    output_dtype,
                    HIPBLAS_OP_N,
                    HIPBLAS_OP_T,
                    output_n,
                    output_k,
                    rows,
                    b_base + begin * output_n * input_element_size,
                    output_n,
                    a_base + begin * output_k * input_element_size,
                    output_k,
                    out_base + expert * output_k * output_n * output_element_size,
                    output_n,
                    nullptr);
            }
            else
            {
                const int64_t input_k  = a.size(1);
                const int64_t output_n = b_is_transposed ? b.size(1) : b.size(2);
                run_multistream_hipblaslt_gemm(
                    ctx,
                    stream_index,
                    input_dtype,
                    output_dtype,
                    b_is_transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                    HIPBLAS_OP_N,
                    output_n,
                    rows,
                    input_k,
                    b_base + expert * input_k * output_n * input_element_size,
                    b_is_transposed ? input_k : output_n,
                    a_base + begin * input_k * input_element_size,
                    input_k,
                    out_base + begin * output_n * output_element_size,
                    output_n,
                    bias_base == nullptr
                        ? nullptr
                        : bias_base + expert * output_n * output_element_size);
            }
            launched[stream_index] = true;
        }
    }
    catch(...)
    {
        // A later expert may fail after earlier GEMMs were already queued on
        // side streams. Drain those launches before propagating the error so a
        // Python fallback cannot race them while writing the same output.
        for(int index = 0; index < streams_used; ++index)
            if(launched[index])
                hipStreamSynchronize(ctx.streams[index]);
        throw;
    }

    for(int index = 0; index < streams_used; ++index)
    {
        if(!launched[index])
            continue;
        TORCH_CHECK(hipEventRecord(ctx.done[index], ctx.streams[index]) == hipSuccess,
                    "recording multi-stream completion event failed");
        TORCH_CHECK(hipStreamWaitEvent(current, ctx.done[index], 0) == hipSuccess,
                    "joining multi-stream GEMM onto caller stream failed");
    }
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("hipb_grouped_mm",
               &hipb_grouped_mm,
               "hipBLASLt grouped GEMM",
               py::arg("a"),
               py::arg("b"),
               py::arg("cu_seqlens"),
               py::arg("out"),
               py::arg("a_is_transposed") = false,
               py::arg("bias")             = std::nullopt,
               py::arg("solution_index")   = -1);
    module.def("hipb_multistream_mm",
               &hipb_multistream_mm,
               "multi-stream BLAS GEMM",
               py::arg("a"),
               py::arg("b"),
               py::arg("cu_seqlens"),
               py::arg("out"),
               py::arg("a_is_transposed") = false,
               py::arg("bias")             = std::nullopt,
               py::arg("b_is_transposed")  = false);
}

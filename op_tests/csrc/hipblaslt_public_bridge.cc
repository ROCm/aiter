// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Exposes a public hipBLASLt heuristic as a lightweight pybind module.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void CheckLt(hipblasStatus_t status, const char* expression)
{
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(expression) + ": status " +
                                 std::to_string(status));
    }
}

#define CHECK_LT(expression) CheckLt((expression), #expression)

class PublicGemm
{
public:
    PublicGemm(int m, int n, int k)
    {
        if(m <= 0 || n <= 0 || k <= 0)
        {
            throw std::invalid_argument("dimensions must be positive");
        }

        CHECK_LT(hipblasLtCreate(&handle_));
        CHECK_LT(hipblasLtMatrixLayoutCreate(&layout_a_, HIP_R_8F_E4M3, k, m, k));
        CHECK_LT(hipblasLtMatrixLayoutCreate(&layout_b_, HIP_R_8F_E4M3, k, n, k));
        CHECK_LT(hipblasLtMatrixLayoutCreate(&layout_d_, HIP_R_16BF, m, n, m));
        CHECK_LT(
            hipblasLtMatmulDescCreate(&operation_, HIPBLAS_COMPUTE_32F, HIP_R_32F));

        hipblasOperation_t trans_a = HIPBLAS_OP_T;
        hipblasOperation_t trans_b = HIPBLAS_OP_N;
        CHECK_LT(hipblasLtMatmulDescSetAttribute(
            operation_, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
        CHECK_LT(hipblasLtMatmulDescSetAttribute(
            operation_, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

        hipblasLtMatmulMatrixScale_t scale_mode =
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        CHECK_LT(hipblasLtMatmulDescSetAttribute(operation_,
                                                 HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                 &scale_mode,
                                                 sizeof(scale_mode)));
        CHECK_LT(hipblasLtMatmulDescSetAttribute(operation_,
                                                 HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                 &scale_mode,
                                                 sizeof(scale_mode)));

        CHECK_LT(hipblasLtMatmulPreferenceCreate(&preference_));
        const std::uint64_t max_workspace_size = 128ULL * 1024 * 1024;
        CHECK_LT(hipblasLtMatmulPreferenceSetAttribute(
            preference_,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &max_workspace_size,
            sizeof(max_workspace_size)));

        int returned_count = 0;
        CHECK_LT(hipblasLtMatmulAlgoGetHeuristic(handle_,
                                                 operation_,
                                                 layout_a_,
                                                 layout_b_,
                                                 layout_d_,
                                                 layout_d_,
                                                 preference_,
                                                 1,
                                                 &heuristic_,
                                                 &returned_count));
        if(returned_count != 1)
        {
            throw std::runtime_error("hipBLASLt returned no heuristic solution");
        }

        index_ = hipblaslt_ext::getIndexFromAlgo(heuristic_.algo);
        solution_name_ = hipblaslt_ext::getSolutionNameFromAlgo(handle_, heuristic_.algo);
        kernel_name_ = hipblaslt_ext::getKernelNameFromAlgo(handle_, heuristic_.algo);
    }

    PublicGemm(const PublicGemm&) = delete;
    PublicGemm& operator=(const PublicGemm&) = delete;

    ~PublicGemm()
    {
        if(preference_ != nullptr)
        {
            (void)hipblasLtMatmulPreferenceDestroy(preference_);
        }
        if(operation_ != nullptr)
        {
            (void)hipblasLtMatmulDescDestroy(operation_);
        }
        if(layout_d_ != nullptr)
        {
            (void)hipblasLtMatrixLayoutDestroy(layout_d_);
        }
        if(layout_b_ != nullptr)
        {
            (void)hipblasLtMatrixLayoutDestroy(layout_b_);
        }
        if(layout_a_ != nullptr)
        {
            (void)hipblasLtMatrixLayoutDestroy(layout_a_);
        }
        if(handle_ != nullptr)
        {
            (void)hipblasLtDestroy(handle_);
        }
    }

    int index() const { return index_; }
    const std::string& solution_name() const { return solution_name_; }
    const std::string& kernel_name() const { return kernel_name_; }
    std::size_t workspace_size() const { return heuristic_.workspaceSize; }

    void Launch(std::uintptr_t a,
                std::uintptr_t b,
                std::uintptr_t scale_a,
                std::uintptr_t scale_b,
                std::uintptr_t out,
                std::uintptr_t workspace,
                std::uintptr_t stream)
    {
        void* scale_a_pointer = reinterpret_cast<void*>(scale_a);
        void* scale_b_pointer = reinterpret_cast<void*>(scale_b);
        CHECK_LT(hipblasLtMatmulDescSetAttribute(operation_,
                                                 HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                 &scale_a_pointer,
                                                 sizeof(scale_a_pointer)));
        CHECK_LT(hipblasLtMatmulDescSetAttribute(operation_,
                                                 HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                 &scale_b_pointer,
                                                 sizeof(scale_b_pointer)));

        constexpr float kAlpha = 1.0f;
        constexpr float kBeta = 0.0f;
        CHECK_LT(hipblasLtMatmul(handle_,
                                operation_,
                                &kAlpha,
                                reinterpret_cast<void*>(a),
                                layout_a_,
                                reinterpret_cast<void*>(b),
                                layout_b_,
                                &kBeta,
                                reinterpret_cast<void*>(out),
                                layout_d_,
                                reinterpret_cast<void*>(out),
                                layout_d_,
                                &heuristic_.algo,
                                reinterpret_cast<void*>(workspace),
                                heuristic_.workspaceSize,
                                reinterpret_cast<hipStream_t>(stream)));
    }

private:
    int index_ = -1;
    std::string solution_name_;
    std::string kernel_name_;
    hipblasLtHandle_t handle_ = nullptr;
    hipblasLtMatrixLayout_t layout_a_ = nullptr;
    hipblasLtMatrixLayout_t layout_b_ = nullptr;
    hipblasLtMatrixLayout_t layout_d_ = nullptr;
    hipblasLtMatmulDesc_t operation_ = nullptr;
    hipblasLtMatmulPreference_t preference_ = nullptr;
    hipblasLtMatmulHeuristicResult_t heuristic_{};
};

} // namespace

PYBIND11_MODULE(hipblaslt_public_bridge, module)
{
    py::class_<PublicGemm>(module, "PublicGemm")
        .def(py::init<int, int, int>())
        .def_property_readonly("index", &PublicGemm::index)
        .def_property_readonly("solution_name", &PublicGemm::solution_name)
        .def_property_readonly("kernel_name", &PublicGemm::kernel_name)
        .def_property_readonly("workspace_size", &PublicGemm::workspace_size)
        .def("launch", &PublicGemm::Launch);
}

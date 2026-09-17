// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Queries and times the public hipBLASLt heuristic for MXFP8 TN GEMM.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

namespace {

void CheckHip(hipError_t status, const char* expression) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             hipGetErrorString(status));
  }
}

void CheckLt(hipblasStatus_t status, const char* expression) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(expression) + ": status " +
                             std::to_string(status));
  }
}

#define CHECK_HIP(expression) CheckHip((expression), #expression)
#define CHECK_LT(expression) CheckLt((expression), #expression)

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size) : size_(size) {
    CHECK_HIP(hipMalloc(&data_, size_));
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  void* data() { return data_; }
  std::size_t size() const { return size_; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
};

class PublicGemm {
 public:
  PublicGemm(int m, int n, int k)
      : m_(m),
        n_(n),
        k_(k),
        a_(static_cast<std::size_t>(m) * k),
        b_(static_cast<std::size_t>(n) * k),
        d_(static_cast<std::size_t>(m) * n * sizeof(std::uint16_t)),
        scale_a_(static_cast<std::size_t>(m) * k / 32),
        scale_b_(static_cast<std::size_t>(n) * k / 32) {
    CHECK_HIP(hipMemset(a_.data(), 0, a_.size()));
    CHECK_HIP(hipMemset(b_.data(), 0, b_.size()));
    CHECK_HIP(hipMemset(d_.data(), 0, d_.size()));
    CHECK_HIP(hipMemset(scale_a_.data(), 0x7f, scale_a_.size()));
    CHECK_HIP(hipMemset(scale_b_.data(), 0x7f, scale_b_.size()));

    CHECK_LT(hipblasLtCreate(&handle_));
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layout_a_, HIP_R_8F_E4M3, k, m,
                                         k));
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layout_b_, HIP_R_8F_E4M3, k, n,
                                         k));
    CHECK_LT(hipblasLtMatrixLayoutCreate(&layout_d_, HIP_R_16BF, m, n, m));
    CHECK_LT(hipblasLtMatmulDescCreate(&operation_, HIPBLAS_COMPUTE_32F,
                                       HIP_R_32F));

    hipblasOperation_t trans_a = HIPBLAS_OP_T;
    hipblasOperation_t trans_b = HIPBLAS_OP_N;
    CHECK_LT(hipblasLtMatmulDescSetAttribute(
        operation_, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CHECK_LT(hipblasLtMatmulDescSetAttribute(
        operation_, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    hipblasLtMatmulMatrixScale_t scale_mode =
        HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CHECK_LT(hipblasLtMatmulDescSetAttribute(
        operation_, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode,
        sizeof(scale_mode)));
    CHECK_LT(hipblasLtMatmulDescSetAttribute(
        operation_, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode,
        sizeof(scale_mode)));
    SetScalePointers();

    CHECK_LT(hipblasLtMatmulPreferenceCreate(&preference_));
    std::uint64_t max_workspace_size = 128ULL * 1024 * 1024;
    CHECK_LT(hipblasLtMatmulPreferenceSetAttribute(
        preference_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace_size, sizeof(max_workspace_size)));

    int returned_count = 0;
    CHECK_LT(hipblasLtMatmulAlgoGetHeuristic(
        handle_, operation_, layout_a_, layout_b_, layout_d_, layout_d_,
        preference_, 1, &heuristic_, &returned_count));
    if (returned_count != 1) {
      throw std::runtime_error("hipBLASLt returned no heuristic solution");
    }

    workspace_ = std::make_unique<DeviceBuffer>(
        std::max<std::size_t>(heuristic_.workspaceSize, 1));
    CHECK_HIP(hipStreamCreate(&stream_));
  }

  PublicGemm(const PublicGemm&) = delete;
  PublicGemm& operator=(const PublicGemm&) = delete;

  ~PublicGemm() {
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
    if (preference_ != nullptr) {
      hipblasLtMatmulPreferenceDestroy(preference_);
    }
    if (operation_ != nullptr) {
      hipblasLtMatmulDescDestroy(operation_);
    }
    if (layout_d_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(layout_d_);
    }
    if (layout_b_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(layout_b_);
    }
    if (layout_a_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(layout_a_);
    }
    if (handle_ != nullptr) {
      hipblasLtDestroy(handle_);
    }
  }

  void PrintSelection() {
    std::cout << "shape=" << m_ << "x" << n_ << "x" << k_ << '\n';
    std::cout << "index=" << hipblaslt_ext::getIndexFromAlgo(heuristic_.algo)
              << '\n';
    std::cout << "solution="
              << hipblaslt_ext::getSolutionNameFromAlgo(handle_, heuristic_.algo)
              << '\n';
    std::cout << "kernel="
              << hipblaslt_ext::getKernelNameFromAlgo(handle_, heuristic_.algo)
              << '\n';
    std::cout << "workspace=" << heuristic_.workspaceSize << '\n';
  }

  void Run(int warmups, int iterations, bool graph_mode) {
    for (int i = 0; i < warmups; ++i) {
      Launch();
    }
    CHECK_HIP(hipStreamSynchronize(stream_));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    if (graph_mode) {
      hipGraph_t graph = nullptr;
      hipGraphExec_t graph_exec = nullptr;
      CHECK_HIP(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
      for (int i = 0; i < iterations; ++i) {
        Launch();
      }
      CHECK_HIP(hipStreamEndCapture(stream_, &graph));
      CHECK_HIP(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
      CHECK_HIP(hipEventRecord(start, stream_));
      CHECK_HIP(hipGraphLaunch(graph_exec, stream_));
      CHECK_HIP(hipEventRecord(stop, stream_));
      CHECK_HIP(hipEventSynchronize(stop));
      CHECK_HIP(hipGraphExecDestroy(graph_exec));
      CHECK_HIP(hipGraphDestroy(graph));
    } else {
      CHECK_HIP(hipEventRecord(start, stream_));
      for (int i = 0; i < iterations; ++i) {
        Launch();
      }
      CHECK_HIP(hipEventRecord(stop, stream_));
      CHECK_HIP(hipEventSynchronize(stop));
    }

    float elapsed_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipEventDestroy(start));
    std::cout << "event_us=" << elapsed_ms * 1000.0f / iterations << '\n';
  }

 private:
  void SetScalePointers() {
    void* scale_a = scale_a_.data();
    void* scale_b = scale_b_.data();
    CHECK_LT(hipblasLtMatmulDescSetAttribute(
        operation_, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a,
        sizeof(scale_a)));
    CHECK_LT(hipblasLtMatmulDescSetAttribute(
        operation_, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b,
        sizeof(scale_b)));
  }

  void Launch() {
    constexpr float kAlpha = 1.0f;
    constexpr float kBeta = 0.0f;
    CHECK_LT(hipblasLtMatmul(
        handle_, operation_, &kAlpha, a_.data(), layout_a_, b_.data(), layout_b_,
        &kBeta, d_.data(), layout_d_, d_.data(), layout_d_, &heuristic_.algo,
        workspace_->data(), heuristic_.workspaceSize, stream_));
  }

  int m_;
  int n_;
  int k_;
  DeviceBuffer a_;
  DeviceBuffer b_;
  DeviceBuffer d_;
  DeviceBuffer scale_a_;
  DeviceBuffer scale_b_;
  std::unique_ptr<DeviceBuffer> workspace_;
  hipblasLtHandle_t handle_ = nullptr;
  hipblasLtMatrixLayout_t layout_a_ = nullptr;
  hipblasLtMatrixLayout_t layout_b_ = nullptr;
  hipblasLtMatrixLayout_t layout_d_ = nullptr;
  hipblasLtMatmulDesc_t operation_ = nullptr;
  hipblasLtMatmulPreference_t preference_ = nullptr;
  hipblasLtMatmulHeuristicResult_t heuristic_{};
  hipStream_t stream_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    std::cerr << "usage: " << argv[0]
              << " M N K [iterations=100] [graph=1]\n";
    return 2;
  }

  try {
    const int m = std::stoi(argv[1]);
    const int n = std::stoi(argv[2]);
    const int k = std::stoi(argv[3]);
    const int iterations = argc >= 5 ? std::stoi(argv[4]) : 100;
    const bool graph_mode = argc >= 6 ? std::stoi(argv[5]) != 0 : true;
    if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0) {
      throw std::invalid_argument("dimensions and iterations must be positive");
    }
    PublicGemm gemm(m, n, k);
    gemm.PrintSelection();
    gemm.Run(2, iterations, graph_mode);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

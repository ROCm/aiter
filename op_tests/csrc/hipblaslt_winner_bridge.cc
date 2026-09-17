// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A thin C ABI for an explicit hipBLASLt/Tensile winner, not public heuristics.
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

namespace {
namespace tl = TensileLite;
thread_local std::string last_error;

void CheckHip(hipError_t status) {
  if (status != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(status));
  }
}

struct Winner {
  std::shared_ptr<tl::Hardware> hardware;
  std::shared_ptr<tl::ContractionSolution> solution;
  tl::ContractionProblemGemm problem;
  tl::hip::SolutionAdapter adapter;
  size_t workspace_size = 0;

  Winner(const char* library_path, const char* code_dir, int m, int n, int k)
      : hardware(tl::hip::GetCurrentDevice()),
        problem(tl::ContractionProblemGemm::GEMM_Strides(
            true, false, rocisa::DataType::Float8, rocisa::DataType::Float8,
            rocisa::DataType::BFloat16, rocisa::DataType::BFloat16,
            m, n, k, 1, k, size_t(m) * k, k, size_t(n) * k,
            m, size_t(m) * n, m, size_t(m) * n, 0.0)) {
    auto library = std::dynamic_pointer_cast<
        tl::MasterSolutionLibrary<tl::ContractionProblemGemm>>(
        tl::LoadLibraryFile<tl::ContractionProblemGemm>(library_path));
    if (!library || library->solutions.size() != 1) {
      throw std::runtime_error("Expected exactly one explicit winner");
    }
    solution = library->solutions.begin()->second;
    problem.setHighPrecisionAccumulate(true);
    problem.setStridedBatched(true);
    problem.setCEqualsD(true);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setComputeInputTypeA(rocisa::DataType::Float8);
    problem.setComputeInputTypeB(rocisa::DataType::Float8);
    problem.setMXScaleA(rocisa::DataType::E8, 32, {}, false);
    problem.setMXScaleB(rocisa::DataType::E8, 32, {}, false);
    workspace_size = solution->requiredWorkspaceSize(problem, *hardware);
    problem.setWorkspaceSize(workspace_size);
    int loaded = 0;
    for (const auto& entry : std::filesystem::directory_iterator(code_dir)) {
      if (entry.path().extension() == ".co" ||
          entry.path().extension() == ".hsaco") {
        CheckHip(adapter.loadCodeObjectFile(entry.path().string()));
        ++loaded;
      }
    }
    if (loaded == 0) {
      throw std::runtime_error("No code objects found");
    }
  }
};
}  // namespace

extern "C" const char* WinnerError() { return last_error.c_str(); }

extern "C" void* WinnerCreate(const char* library_path, const char* code_dir,
                              int m, int n, int k) {
  try {
    return std::make_unique<Winner>(library_path, code_dir, m, n, k).release();
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

extern "C" const char* WinnerName(void* handle) {
  return static_cast<Winner*>(handle)->solution->solutionName.c_str();
}

extern "C" size_t WinnerWorkspaceSize(void* handle) {
  return static_cast<Winner*>(handle)->workspace_size;
}

extern "C" int WinnerLaunch(void* handle, const void* a, const void* b,
                            const void* scale_a, const void* scale_b, void* d,
                            void* workspace, void* synchronizer,
                            hipStream_t stream) {
  try {
    auto& winner = *static_cast<Winner*>(handle);
    tl::ContractionInputs inputs(a, b, d, d, 1.0f, 0.0f);
    inputs.mxsa = scale_a;
    inputs.mxsb = scale_b;
    inputs.ws = workspace;
    inputs.Synchronizer = synchronizer;
    inputs.workspaceSize = winner.workspace_size;
    inputs.gpu = true;
    const auto kernels = winner.solution->solve(winner.problem, inputs,
                                                *winner.hardware);
    CheckHip(winner.adapter.launchKernels(kernels, stream, nullptr, nullptr));
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}

extern "C" void WinnerDestroy(void* handle) {
  delete static_cast<Winner*>(handle);
}

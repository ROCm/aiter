// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

namespace {
namespace py = pybind11;
namespace tl = TensileLite;

void CheckHip(hipError_t status) {
  if (status != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(status));
  }
}

class Winner {
 public:
  Winner(const std::string& library_path, const std::string& code_dir, int m,
         int n, int k)
      : hardware_(tl::hip::GetCurrentDevice()),
        problem_(tl::ContractionProblemGemm::GEMM_Strides(
            true, false, rocisa::DataType::Float8, rocisa::DataType::Float8,
            rocisa::DataType::BFloat16, rocisa::DataType::BFloat16, m, n, k, 1,
            k, static_cast<size_t>(m) * k, k, static_cast<size_t>(n) * k, m,
            static_cast<size_t>(m) * n, m, static_cast<size_t>(m) * n, 0.0)) {
    auto library = std::dynamic_pointer_cast<
        tl::MasterSolutionLibrary<tl::ContractionProblemGemm>>(
        tl::LoadLibraryFile<tl::ContractionProblemGemm>(library_path));
    if (!library || library->solutions.size() != 1) {
      throw std::runtime_error("Expected exactly one explicit winner");
    }
    solution_ = library->solutions.begin()->second;
    problem_.setHighPrecisionAccumulate(true);
    problem_.setStridedBatched(true);
    problem_.setCEqualsD(true);
    problem_.setAlphaType(rocisa::DataType::Float);
    problem_.setBetaType(rocisa::DataType::Float);
    problem_.setComputeInputTypeA(rocisa::DataType::Float8);
    problem_.setComputeInputTypeB(rocisa::DataType::Float8);
    problem_.setMXScaleA(rocisa::DataType::E8, 32, {}, false);
    problem_.setMXScaleB(rocisa::DataType::E8, 32, {}, false);
    workspace_size_ = solution_->requiredWorkspaceSize(problem_, *hardware_);
    problem_.setWorkspaceSize(workspace_size_);
    int loaded = 0;
    for (const auto& entry : std::filesystem::directory_iterator(code_dir)) {
      if (entry.path().extension() == ".co" ||
          entry.path().extension() == ".hsaco") {
        CheckHip(adapter_.loadCodeObjectFile(entry.path().string()));
        ++loaded;
      }
    }
    if (loaded == 0) {
      throw std::runtime_error("No code objects found");
    }
  }

  Winner(const Winner&) = delete;
  Winner& operator=(const Winner&) = delete;

  const std::string& solution_name() const { return solution_->solutionName; }

  size_t workspace_size() const { return workspace_size_; }

  void Launch(std::uintptr_t a, std::uintptr_t b, std::uintptr_t scale_a,
              std::uintptr_t scale_b, std::uintptr_t output,
              std::uintptr_t workspace, std::uintptr_t synchronizer,
              std::uintptr_t stream) {
    void* output_pointer = reinterpret_cast<void*>(output);
    tl::ContractionInputs inputs(
        reinterpret_cast<const void*>(a), reinterpret_cast<const void*>(b),
        output_pointer, output_pointer, 1.0f, 0.0f);
    inputs.mxsa = reinterpret_cast<const void*>(scale_a);
    inputs.mxsb = reinterpret_cast<const void*>(scale_b);
    inputs.ws = reinterpret_cast<void*>(workspace);
    inputs.Synchronizer = reinterpret_cast<void*>(synchronizer);
    inputs.workspaceSize = workspace_size_;
    inputs.gpu = true;
    const auto kernels = solution_->solve(problem_, inputs, *hardware_);
    CheckHip(adapter_.launchKernels(
        kernels, reinterpret_cast<hipStream_t>(stream), nullptr, nullptr));
  }

 private:
  std::shared_ptr<tl::Hardware> hardware_;
  std::shared_ptr<tl::ContractionSolution> solution_;
  tl::ContractionProblemGemm problem_;
  tl::hip::SolutionAdapter adapter_;
  size_t workspace_size_ = 0;
};

}  // namespace

PYBIND11_MODULE(hipblaslt_winner_bridge, module) {
  py::class_<Winner>(module, "Winner")
      .def(py::init<const std::string&, const std::string&, int, int, int>())
      .def_property_readonly("solution_name", &Winner::solution_name)
      .def_property_readonly("workspace_size", &Winner::workspace_size)
      .def("launch", &Winner::Launch);
}

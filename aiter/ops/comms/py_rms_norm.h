#ifndef __PYBIND11_RMSNORM_H__
#define __PYBIND11_RMSNORM_H__

#include <c10/hip/HIPCachingAllocator.h>
#include <hip/hip_runtime.h>
#include <omp.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <string>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

#include "dlpack.h"
#include "rmsnorm.h"

namespace py = pybind11;

template <typename T>
struct DLType;

template <>
struct DLType<float>
{
    static constexpr DLDataTypeCode code = kDLFloat;
    static constexpr int bits = 32;
    static constexpr int lanes = 1;
};

template <>
struct DLType<__hip_bfloat16>
{
    static constexpr DLDataTypeCode code = kDLBfloat;
    static constexpr int bits = 16;
    static constexpr int lanes = 1;
};

template <>
struct DLType<__hip_fp8_storage_t>
{
    static constexpr DLDataTypeCode code = kDLFloat8_e4m3;
    static constexpr int bits = 8;
    static constexpr int lanes = 1;
};

template <typename T>
class GPUMemoryResources
{
public:
    GPUMemoryResources()
        : nGPU_(0),
          n_rows_(0),
          n_cols_(0),
          stride_(0),
          rank_(-1),
          initialized_(false),
          external_memory_(false),
          distributed_mode_(false)
    {
    }

    void initialize(int nGPU, int n_rows, int n_cols, int stride);

    // Initialize with external GPU memory (for vLLM workflow)
    void initialize_with_external_memory(
        int nGPU,
        std::vector<uintptr_t>& external_ptrs,
        int n_rows,
        int n_cols,
        int stride);

    // Initialize with distributed external GPU memory (for vLLM
    // workflow)
    void initialize_with_distributed_external_torch(
        at::Tensor& tensor, py::object& py_process_group);

    void compute();

    ~GPUMemoryResources() { cleanup(); }

    void cleanup();

    // Getters that return raw pointers
    std::vector<T*> device_buffers() const;

    std::vector<T**> device_pointers() const;

    // Get shared_ptrs for advanced use cases
    const std::vector<std::shared_ptr<T>>& get_buffer_shared_ptrs()
        const
    {
        return d_buffers_;
    }

    const std::vector<std::shared_ptr<T*>>& get_pointer_shared_ptr()
        const
    {
        return d_ptrs_;
    }

    T** get_map_pointers() { return mapped_ptrs_.get(); }

    int num_gpus() const { return nGPU_; }

    int get_rank() const { return rank_; }

    int num_rows_per_gpu() const { return n_rows_; }

    int num_cols_per_gpu() const { return n_cols_; }

    int stride() const { return stride_; }

    bool is_initialized() const { return initialized_; }

    bool using_external_memory() const { return external_memory_; }

    bool is_distributed() const { return distributed_mode_; }

private:
    T* map_handle(const hipIpcMemHandle_t handle);
    int nGPU_;
    int n_rows_;
    int n_cols_;
    int stride_;
    int rank_;
    bool initialized_;
    bool external_memory_;
    bool distributed_mode_;
    std::vector<std::shared_ptr<T>> d_buffers_;
    std::vector<std::shared_ptr<T*>> d_ptrs_;
    struct SharedData
    {
        hipIpcMemHandle_t handle_;
        size_t offset_bytes_;
    };
    std::vector<SharedData> shared_data_;
    std::vector<hipIpcMemHandle_t> handles_;
    std::shared_ptr<T*> mapped_ptrs_;
};

template <typename T>
struct DLTensorType
{
    static constexpr DLDataTypeCode code = kDLFloat;
};

template <>
struct DLTensorType<__hip_bfloat16>
{
    static constexpr DLDataTypeCode code = kDLBfloat;
};

template <typename T>
class HostTensorDLPacker
{
private:
    DLManagedTensor manager_;
    std::unique_ptr<T[]> host_buffer_;

public:
    static void Deleter(DLManagedTensor* self)
    {
        HostTensorDLPacker<T>* wrapper =
            static_cast<HostTensorDLPacker<T>*>(self->manager_ctx);
        delete[] wrapper->manager_.dl_tensor.shape;
        delete[] wrapper->manager_.dl_tensor.strides;
        delete wrapper;
    }

    HostTensorDLPacker(int n_rows,
                       int n_cols,
                       T* data_ptr,
                       DLDataTypeCode type_code)
        : host_buffer_(data_ptr)
    {
        // --- 1. Initialize DLManagedTensor Management ---
        manager_.manager_ctx = this;
        manager_.deleter = HostTensorDLPacker<T>::Deleter;

        // --- 2. Initialize DLTensor Fields ---
        DLTensor& dlt = manager_.dl_tensor;
        dlt.data = host_buffer_.get();

        // Device Context: CPU
        dlt.device = DLDevice{kDLCPU, 0};

        // Data Type: 16-bit float, with code specified by caller
        dlt.dtype.code = type_code;
        dlt.dtype.bits = 16;
        dlt.dtype.lanes = 1;

        // Shape: 2D tensor
        dlt.ndim = 2;
        dlt.shape = new int64_t[2]{static_cast<int64_t>(n_rows),
                                   static_cast<int64_t>(n_cols)};
        dlt.strides = new int64_t[2]{static_cast<int64_t>(n_cols), 1};
        dlt.byte_offset = 0;
    }

    DLManagedTensor* get_managed_tensor() { return &manager_; }
};

// Function to download tensor fromGPU to CPU and return as
// torch::Tensor
template <typename T>
torch::Tensor download(int gpu_id,
                       const std::vector<T**>& output_ptrs,
                       size_t elements_per_gpu,
                       const std::vector<int64_t>& shape);

// Reusable RMS norm executor with PyTorch integration
template <typename InT, typename OutT = InT>
class RMSNormExecutor
{
public:
    RMSNormExecutor()
        : nGPU_(1),
          rank_(-1),
          n_rows_(1),
          n_cols_(1),
          input_row_stride_(1),
          output_row_stride_(1),
          initialized_(false),
          external_memory_(false),
          peer_access_enabled_(false),
          distributed_mode_(false)
    {
    }

    ~RMSNormExecutor()
    {
        std::cout << "~RMSNormExecutor\n";
        release();
    }

    // Check and enable peer access with better error handling
    void enable_peer_access(int nGPU, int rank = -1);

    // Disable peer access
    void disable_peer_access(int rank = -1);

    // Manuall release resources
    void release();

    // Check if peer access is available between two GPUs
    bool can_access_peer(int from_gpu, int to_gpu) const;

    // Standard initialization with memory allocation
    void initialize(int nGPU,
                    int n_rows,
                    int n_cols,
                    int input_row_stride,
                    int output_row_stride,
                    float epsilon);

    // initialize with external GPU memory (vLLM workflow)
    void initialize_with_external_memory(
        int nGPU,
        int n_rows,
        int n_cols,
        int input_row_stride,
        int output_row_stride,
        std::vector<uintptr_t> input_ptrs,
        std::vector<uintptr_t> output_ptrs,
        std::vector<uintptr_t> g_ptrs,
        std::vector<uintptr_t> rsigma_ptrs,
        float epsilon);

    void recreate_helper();

    // Compute without host-device copies (assumes data is already on
    // GPU)
    void compute();

    void upload(py::array_t<InT, py::array::c_style>& input_arrays);

    py::capsule to_dlpack(int gpu_id);

    //    py::array_t<uint16_t, py::array::c_style> download(int id);
    py::capsule download(int id);

    // Original compute method for backward compatibility
    void compute_with_host_data(
        py::array_t<InT, py::array::c_style>& input_arrays);

    // PyTorch-specific method: compute ith PyTorch tensors
    // (zero-copy)
    void compute_with_pytorch_tensors(
        const std::vector<py::object>& input_tensors,
        std::vector<py::object>& output_tensors,
        const std::vector<py::object>& weight_tensors,
        std::vector<py::object>& rsigma_tensors,
        float epsilon);

    // PyTorch-specific method: compute ith PyTorch tensors
    // (zero-copy)
    void compute_with_distributed_tensors(
        at::Tensor& input_tensor,
        at::Tensor& outtput_tensor,
        at::Tensor& weight_tensor,
        at::Tensor& rsigma_tensor,
        float epsilon,
        py::object& py_process_group);

    bool is_initialized() const { return initialized_; }

    bool using_external_memory() const { return external_memory_; }

    int num_gpus() const { return nGPU_; }

    int num_rows_per_gpu() const { return n_rows_; }

    int num_cols_per_gpu() const { return n_cols_; }

private:
    uintptr_t map_handle(const hipIpcMemHandle_t handle);

    int nGPU_;
    int rank_;
    int n_rows_;
    int n_cols_;
    int input_row_stride_;
    int output_row_stride_;
    bool initialized_;
    bool external_memory_;
    bool peer_access_enabled_;
    bool distributed_mode_;
    float epsilon_;
    GPUMemoryResources<InT> input_resources_;
    GPUMemoryResources<OutT> output_resources_;
    GPUMemoryResources<InT> g_resources_;
    GPUMemoryResources<OutT> rsigma_resources_;
    std::shared_ptr<RMSNormHelper<InT, OutT>> helper_;
    std::vector<std::vector<bool>> peer_access_matrix_;
    std::vector<hipIpcMemHandle_t> remote_handles_;
};

#endif

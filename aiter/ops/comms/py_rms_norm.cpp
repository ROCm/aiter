#include "py_rms_norm.h"

// Initialize
template <typename T>
void GPUMemoryResources<T>::initialize(int nGPU,
                                       int n_rows,
                                       int n_cols,
                                       int stride)
{
    if (initialized_ && (nGPU_ == nGPU) &&
        (n_rows_ == n_rows && n_cols_ == n_cols && stride_ == stride))
    {
        return;
    }

    if (initialized_)
    {
        cleanup();
    }

    nGPU_ = nGPU;
    n_rows_ = n_rows;
    n_cols_ = n_cols;
    stride_ = stride;
    external_memory_ = false;

    d_buffers_.resize(nGPU);
    d_ptrs_.resize(nGPU);

    std::vector<T*> host_ptrs(nGPU);

    for (int dev = 0; dev < nGPU_; dev++)
    {
        HIP_CALL(hipSetDevice(dev));

        d_buffers_[dev] = std::shared_ptr<T>(
            static_cast<T*>(
                hipMallocFunc(n_rows_ * stride_ * sizeof(T))),
            hipDeleteHelper(dev));

        host_ptrs[dev] = d_buffers_[dev].get();

        // Allocate device array of pointers
        d_ptrs_[dev] = std::shared_ptr<T*>(
            static_cast<T**>(hipMallocFunc(nGPU_ * sizeof(T*))),
            hipDeleteHelper(dev));
    }

    // copy host pointer array to each device
    for (int dev = 0; dev < nGPU; dev++)
    {
        HIP_CALL(hipMemcpy(d_ptrs_[dev].get(),
                           host_ptrs.data(),
                           nGPU * sizeof(T*),
                           hipMemcpyHostToDevice));
    }

    initialized_ = true;
}

// Initialize with external GPU memory (for vLLM workflow)
template <typename T>
void GPUMemoryResources<T>::initialize_with_external_memory(
    int nGPU,
    std::vector<uintptr_t>& external_ptrs,
    int n_rows,
    int n_cols,
    int stride)
{
    if (!external_ptrs.empty() && external_ptrs.size() != nGPU)
    {
        throw std::runtime_error(
            "Number of external pointers does not match GPU count");
    }

    if (initialized_ && !external_memory_)
    {
        cleanup();
    }

    if (!external_memory_)
    {
        nGPU_ = nGPU;
        n_rows_ = n_rows;
        n_cols_ = n_cols;
        stride_ = stride;

        external_memory_ = true;
        d_buffers_.resize(nGPU);
        d_ptrs_.resize(nGPU);
        std::vector<T*> host_ptrs(nGPU);

        for (int i = 0; i < nGPU; i++)
        {
            HIP_CALL(hipSetDevice(i));

            // Use external pointer with no-op deleter
            if (!external_ptrs.empty())
            {
                d_buffers_[i] = std::shared_ptr<T>(
                    reinterpret_cast<T*>(external_ptrs[i]),
                    [](T*) {});

                host_ptrs[i] = d_buffers_[i].get();

                d_ptrs_[i] = std::shared_ptr<T*>(
                    static_cast<T**>(
                        hipMallocFunc(nGPU * sizeof(T*))),
                    hipDeleteHelper(i));
            }
        }

        if (!external_ptrs.empty())
        {
            for (int i = 0; i < nGPU; i++)
            {
                HIP_CALL(hipSetDevice(i));
                HIP_CALL(hipMemcpy(d_ptrs_[i].get(),
                                   host_ptrs.data(),
                                   nGPU * sizeof(T*),
                                   hipMemcpyHostToDevice));
            }
        }
        else
        {
            for (int i = 0; i < nGPU; i++)
            {
                d_ptrs_[i] = nullptr;
            }
        }

        initialized_ = true;
    }
}

// Initialize with external GPU memory (for vLLM workflow)
template <typename T>
void GPUMemoryResources<T>::
    initialize_with_distributed_external_torch(
        at::Tensor& tensor, py::object& py_process_group)
{
    if (!distributed_mode_)
    {
        rank_ = py_process_group.attr("rank")().cast<int>();
        nGPU_ = py_process_group.attr("size")().cast<int>();

        if (tensor.data_ptr() != nullptr)
        {
            std::vector<int64_t> shape = tensor.sizes().vec();

            if (shape.size() == 2)
            {
                n_rows_ = shape[0];
                n_cols_ = shape[1];
                stride_ = n_cols_;
            }
            else if (shape.size() == 1)
            {
                n_rows_ = shape[0];
                n_cols_ = 1;
                stride_ = 1;
            }

            void* data_ptr = tensor.data_ptr();
            void* base_ptr =
                c10::hip::HIPCachingAllocator::get()
                    ->getBaseAllocation(data_ptr, nullptr);

            // 1.Get Local Handle
            hipIpcMemHandle_t local_handle;
            HIP_CALL(hipIpcGetMemHandle(&local_handle, base_ptr));

            // 2. offset
            size_t offset_bytes =
                reinterpret_cast<uintptr_t>(data_ptr) -
                reinterpret_cast<uintptr_t>(base_ptr);

            SharedData local_data;
            local_data.handle_ = local_handle;
            local_data.offset_bytes_ = offset_bytes;

            // Serialize handle to byte array
            const size_t shared_data_size = sizeof(SharedData);

#if 0
            std::vector<char> local_buffer(handle_size);
            memcpy(local_buffer.data(), &local_handle, handle_size);
#endif

            // Create CPU tensors for allgahter
            auto options = torch::TensorOptions()
                               .dtype(torch::kByte)
                               .device(at::kCPU);

            // Local tensor on CPU
            /*
               torch::Tensor local_tensor = torch::from_blob(
               local_buffer.data(),
               {static_cast<int64_t>(handle_size)},
               options
               ).clone();
             */
            at::Tensor local_tensor =
                at::empty({shared_data_size}, options);
            /*
               std::memcpy(handle_tensor.data_ptr(), &local_handle,
               handle_size);
             */
            std::memcpy(local_tensor.data_ptr(),
                        &local_data,
                        shared_data_size);

            // Prepare output tensors on GPU
            std::vector<torch::Tensor> gathered_tensors;
            gathered_tensors.resize(nGPU_);
            for (int i = 0; i < nGPU_; ++i)
            {
                /*
                   output_tensors[i] =
                   torch::empty({static_cast<int64_t>(handle_size)},
                   torch::TensorOptions().dtype(torch::kByte)
                   );
                 */
                gathered_tensors[i] = at::empty_like(local_tensor);
            }

            // Convert to Python list
            py::list tensor_list;
            for (auto& t : gathered_tensors)
            {
                tensor_list.append(t);
            }

            py::module torch_dist =
                py::module::import("torch.distributed");

            torch_dist.attr("all_gather")(
                tensor_list,
                local_tensor,
                py::arg("group") = py_process_group,
                py::arg("async_op") = false);

            // 4. Map Handles to pointers
            /*
               handles_.resize(nGPU_);

               for(int i = 0; i < nGPU_; ++i)
               {
               torch::Tensor gathered_tensor = output_tensors[i];
               memcpy(&handles_[i], gathered_tensor.data_ptr(),
               handle_size);
               }
             */
            shared_data_.resize(nGPU_);
            for (int i = 0; i < nGPU_; i++)
            {
                std::memcpy(&shared_data_[i],
                            gathered_tensors[i].data_ptr(),
                            shared_data_size);
            }

            mapped_ptrs_ = std::shared_ptr<T*>(
                static_cast<T**>(hipMallocFunc(nGPU_ * sizeof(T*))),
                hipDeleteHelper(rank_));

            std::vector<T*> mapped_host_ptrs_;
            mapped_host_ptrs_.resize(nGPU_);

            for (int i = 0; i < nGPU_; ++i)
            {
                if (i == rank_)
                {
                    mapped_host_ptrs_[i] =
                        reinterpret_cast<T*>(data_ptr);
                }
                else
                {
                    void* remote_base_ptr =
                        map_handle(shared_data_[i].handle_);
                    void* remote_data_ptr;
                    remote_data_ptr =
                        reinterpret_cast<char*>(remote_base_ptr) +
                        shared_data_[i].offset_bytes_;
                    mapped_host_ptrs_[i] =
                        static_cast<T*>(remote_data_ptr);
                }
            }

            HIP_CALL(hipMemcpy(mapped_ptrs_.get(),
                               mapped_host_ptrs_.data(),
                               nGPU_ * sizeof(T*),
                               hipMemcpyHostToDevice));
        }
        else if (tensor.data_ptr() == nullptr)
        {
            mapped_ptrs_ = std::shared_ptr<T*>(nullptr);
        }

        distributed_mode_ = true;
        initialized_ = true;
    }
}

// clean up memorries and disable P2P
template <typename T>
void GPUMemoryResources<T>::cleanup()
{
    if (!initialized_)
    {
        return;
    }

    // Only free the pointer arrays, not the external buffers
    d_ptrs_.clear();

    // Only clear buffers if we allocatted them
    if (!external_memory_)
    {
        d_buffers_.clear();
    }

    if (distributed_mode_ && rank_ > -1)
    {
        for (int i = 0; i < nGPU_; i++)
        {
            if (i != rank_)
            {
                void* remote_data_ptr = mapped_ptrs_.get()[i];
                void* remote_base_ptr =
                    static_cast<char*>(remote_data_ptr) -
                    shared_data_[i].offset_bytes_;
                HIP_CALL(
                    hipIpcCloseMemHandle((void*) remote_base_ptr));
            }
        }
    }
    initialized_ = false;
    external_memory_ = false;
    distributed_mode_ = false;
    nGPU_ = 1;
    rank_ = -1;
}

// Getters that return raw pointers
template <typename T>
std::vector<T*> GPUMemoryResources<T>::device_buffers() const
{
    std::vector<T*> result;
    for (const auto buf : d_buffers_)
    {
        result.push_back(buf.get());
    }
    return result;
}

template <typename T>
std::vector<T**> GPUMemoryResources<T>::device_pointers() const
{
    std::vector<T**> result;
    for (const auto ptr : d_ptrs_)
    {
        result.push_back(ptr.get());
    }
    return result;
}

template <typename T>
T* GPUMemoryResources<T>::map_handle(const hipIpcMemHandle_t handle)
{
    void* dev_ptr;
    HIP_CALL(hipIpcOpenMemHandle(
        (void**) &dev_ptr, handle, hipIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<T*>(dev_ptr);
}

// Check and enable peer access with better error handling
template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::enable_peer_access(int nGPU,
                                                    int rank)
{
    if (peer_access_enabled_)
    {
        return;
    }

    if (rank == -1)
    {
        peer_access_matrix_.resize(nGPU,
                                   std::vector<bool>(nGPU, false));

        for (int i = 0; i < nGPU; ++i)
        {
            HIP_CALL(hipSetDevice(i));

            peer_access_matrix_[i][i] = true;
            for (int j = 1; j < nGPU; j++)
            {
                int other = (i + j) % nGPU;
                int can_access = 0;

                hipError_t peer_check =
                    hipDeviceCanAccessPeer(&can_access, i, other);
                if (peer_check == hipSuccess && can_access)
                {
                    hipError_t enable_result =
                        hipDeviceEnablePeerAccess(other, 0);
                    if (enable_result == hipSuccess)
                    {
                        peer_access_matrix_[i][other] = true;
#if 0
                        std::cout << "Peer access enabled: GPU " << i
                            << " -> GPU " << other << std::endl;
#endif
                    }
                    else if (enable_result ==
                             hipErrorPeerAccessAlreadyEnabled)
                    {
                        peer_access_matrix_[i][other] = true;
#if 0
                        std::cout << "Peer access already!!! "
                            "enabled: GPU " << i << " -> GPU " << other <<
                            std::endl;
#endif
                    }
                    else
                    {
#if 0
                        peer_access_matrix_[i][other] = false;
                        std::cerr << "Failed to enable perr "
                            "access: GPU " << i << " -> GPU " << other << " ("
                            << hipGetErrorString(enable_result) << ")" << std::endl;
#endif
                    }
                }
                else
                {
                    peer_access_matrix_[i][other] = false;
                    if (peer_check != hipSuccess)
                    {
                        std::cerr << "Peer acdcess check failed: GPU "
                                  << i << " -> GPU " << other << " ("
                                  << hipGetErrorString(peer_check)
                                  << ")" << std::endl;
                    }
                    else
                    {
                        std::cerr << "Peer access not supported: GPU "
                                  << i << " -> GPU " << other
                                  << std::endl;
                    }
                }
            }
        }

        // Check if we have full peer acdcess
        bool full_peer_access = true;
        for (int i = 0; i < nGPU && full_peer_access; ++i)
        {
            for (int j = 0; j < nGPU && full_peer_access; ++j)
            {
                if (!peer_access_matrix_[i][j])
                {
                    full_peer_access = false;
                }
            }
        }

        if (!full_peer_access && nGPU > 1)
        {
            std::cerr
                << "Warning: Not all GPU pairs have peer"
                   " access. Cross-GPU performance may be degraded."
                << std::endl;
        }
        else
        {
            peer_access_enabled_ = true;
        }
    }
    else
    {
        HIP_CALL(hipSetDevice(rank));

        for (int j = 1; j < nGPU; j++)
        {
            int other = (rank + j) % nGPU;
            int can_access = 0;

            hipError_t peer_check =
                hipDeviceCanAccessPeer(&can_access, rank, other);
            if (peer_check == hipSuccess && can_access)
            {
                hipError_t enable_result =
                    hipDeviceEnablePeerAccess(other, 0);
            }
        }
        peer_access_enabled_ = true;
    }
}

// Check and enable peer access with better error handling
template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::disable_peer_access(int rank)
{
    if (!peer_access_enabled_)
    {
        return;
    }

    if (rank == -1)
    {
        for (int i = 0; i < nGPU_; ++i)
        {
            HIP_CALL(hipSetDevice(i));

            peer_access_matrix_[i][i] = true;
            for (int j = 1; j < nGPU_; j++)
            {
                int other = (i + j) % nGPU_;
                int can_access = 0;

                hipError_t peer_check =
                    hipDeviceCanAccessPeer(&can_access, i, other);
                if (peer_check == hipSuccess && can_access)
                {
                    if (peer_access_matrix_[i][other])
                    {
                        HIP_CALL(hipDeviceDisablePeerAccess(other));
                    }
                }
            }
        }
    }
    else
    {
        HIP_CALL(hipSetDevice(rank));

        for (int j = 1; j < nGPU_; j++)
        {
            int other = (rank + j) % nGPU_;
            int can_access = 0;

            hipError_t peer_check =
                hipDeviceCanAccessPeer(&can_access, rank, other);
            if (peer_check == hipSuccess && can_access)
            {
                HIP_CALL(hipDeviceDisablePeerAccess(other));
            }
        }
    }
    peer_access_enabled_ = false;
}

template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::release()
{
    if (is_initialized())
    {
        // Clear memory resources
        input_resources_.cleanup();
        output_resources_.cleanup();
        g_resources_.cleanup();
        rsigma_resources_.cleanup();

        if (peer_access_enabled_)
        {
            disable_peer_access(rank_);
        }

        helper_.reset();

        // Resetstate
        peer_access_enabled_ = false;
        nGPU_ = 0;
        rank_ = -1;
        peer_access_matrix_.clear();
    }
}

// Check if peer access is available between two GPUs
template <typename InT, typename OutT>
bool RMSNormExecutor<InT, OutT>::can_access_peer(int from_gpu,
                                                 int to_gpu) const
{
    if (from_gpu < 0 || from_gpu >= peer_access_matrix_.size() ||
        to_gpu < 0 || to_gpu >= peer_access_matrix_.size())
    {
        return false;
    }
    return peer_access_matrix_[from_gpu][to_gpu];
}

// Standard initialization with memory allocation
template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::initialize(int nGPU,
                                            int n_rows,
                                            int n_cols,
                                            int input_row_stride,
                                            int output_row_stride,
                                            float epsilon)
{
    nGPU_ = nGPU;

    // Enable peer access first
    enable_peer_access(nGPU);

    n_rows_ = n_rows;
    n_cols_ = n_cols;
    input_row_stride_ = input_row_stride;
    output_row_stride_ = output_row_stride;
    epsilon_ = epsilon;

    if (!peer_access_enabled_ && nGPU > 1)
    {
        std::cout << "Proceeding with limited peer access..."
                  << std::endl;
    }

    input_resources_.initialize(
        nGPU, n_rows, n_cols, input_row_stride);
    output_resources_.initialize(
        nGPU, n_rows, n_cols, output_row_stride);
    g_resources_.initialize(nGPU, n_cols, 1, 1);
    rsigma_resources_.initialize(nGPU, n_rows, 1, 1);

    recreate_helper();

    initialized_ = true;
}

// initialize with external GPU memory (vLLM workflow)
template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::initialize_with_external_memory(
    int nGPU,
    int n_rows,
    int n_cols,
    int input_row_stride,
    int output_row_stride,
    std::vector<uintptr_t> input_ptrs,
    std::vector<uintptr_t> output_ptrs,
    std::vector<uintptr_t> g_ptrs,
    std::vector<uintptr_t> rsigma_ptrs,
    float epsilon)
{
    //    enable_peer_access(nGPU);
    if (initialized_ && !external_memory_)
    {
        release();
    }

    if (!external_memory_)
    {
        nGPU_ = nGPU;
        n_rows_ = n_rows;
        n_cols_ = n_cols;
        input_row_stride_ = input_row_stride;
        output_row_stride_ = output_row_stride;
        epsilon_ = epsilon;

        input_resources_.initialize_with_external_memory(
            nGPU, input_ptrs, n_rows, n_cols, input_row_stride);

        output_resources_.initialize_with_external_memory(
            nGPU, output_ptrs, n_rows, n_cols, output_row_stride);

        g_resources_.initialize_with_external_memory(
            nGPU, g_ptrs, n_cols, 1, 1);

        rsigma_resources_.initialize_with_external_memory(
            nGPU, rsigma_ptrs, n_rows, 1, 1);

        if (!initialized_)
        {
            enable_peer_access(nGPU);
            recreate_helper();
            initialized_ = true;
        }

        external_memory_ = true;
    }
}

template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::recreate_helper()
{
    if (!input_resources_.is_distributed())
    {
        auto input_device_ptrs = input_resources_.device_pointers();
        auto output_device_ptrs = output_resources_.device_pointers();
        auto g_device_ptrs = g_resources_.device_pointers();
        auto rsigma_device_ptrs = rsigma_resources_.device_pointers();

        auto nGPU = num_gpus();
        auto n_rows = num_rows_per_gpu();
        auto n_cols = num_cols_per_gpu();
        auto input_row_stride = input_resources_.stride();
        auto output_row_stride = output_resources_.stride();

        helper_ = std::make_shared<RMSNormHelper<InT, OutT>>(
            nGPU,
            n_rows,
            n_cols,
            input_row_stride,
            output_row_stride,
            input_device_ptrs,
            output_device_ptrs,
            g_device_ptrs,
            rsigma_device_ptrs,
            epsilon_);
    }
    else
    {
        auto mapped_input_ptrs = input_resources_.get_map_pointers();
        auto mapped_output_ptrs =
            output_resources_.get_map_pointers();
        auto mapped_g_ptrs = g_resources_.get_map_pointers();
        auto mapped_rsigma_ptrs =
            rsigma_resources_.get_map_pointers();

        auto nGPU = num_gpus();
        auto rank = input_resources_.get_rank();
        auto n_rows = num_rows_per_gpu();
        auto n_cols = num_cols_per_gpu();
        auto input_row_stride = input_resources_.stride();
        auto output_row_stride = output_resources_.stride();

        helper_ = std::make_shared<RMSNormHelper<InT, OutT>>(
            nGPU,
            rank,
            n_rows,
            n_cols,
            input_row_stride,
            output_row_stride,
            mapped_input_ptrs,
            mapped_output_ptrs,
            mapped_g_ptrs,
            mapped_rsigma_ptrs,
            epsilon_);
    }
    initialized_ = true;
}

// Compute without host-device copies (assumes data is already on
// GPU)
template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::compute()
{
    if (!initialized_)
    {
        throw std::runtime_error("RMSNormExecutor not initialized");
    }
    // Just lanuch the kernel - data is already on GPU from previous
    // GEMM
    rms_norm<InT, OutT>(*helper_);
}

template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::upload(
    py::array_t<InT, py::array::c_style>& input_arrays)
{
#if 0
    auto nGPU = input_resources_.num_gpus();
    auto num_rows = num_rows_per_gpu();
    int stride = stride(); 
    int size = num_rows*stride;

    py::buffer_info buf = input_arrays.request();

    T *ptr = static_cast<T *>(buf.ptr);

    // copy input data from host to device
    auto input_device_buffers = input_resources_.device_buffers();
    for (int i = 0; i < nGPU; ++i)
    {
        HIP_CALL(hipMemcpy(input_device_buffers[i], ptr+i*size,
                    size*sizeof(T), hipMemcpyHostToDevice));
    }
#endif
}

// Original compute method for backward compatibility
template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::compute_with_host_data(
    py::array_t<InT, py::array::c_style>& input_arrays)
{
    if (!initialized_)
    {
        throw std::runtime_error("RMSNormExecutor not initialized");
    }

    int nGPU = input_resources_.num_gpus();
    int n_rows = input_resources_.num_rows_per_gpu();
    int stride = input_resources_.stride();

    py::buffer_info buf = input_arrays.request();

    InT* ptr = static_cast<InT*>(buf.ptr);

    // copy input data from host to device
    auto input_device_buffers = input_resources_.device_buffers();

    auto size = n_rows * stride;
    for (int i = 0; i < nGPU; ++i)
    {
        HIP_CALL(hipSetDevice(i));
        HIP_CALL(hipMemcpy(input_device_buffers[i],
                           ptr + i * size,
                           size * sizeof(InT),
                           hipMemcpyHostToDevice));
    }

    // Launch kernel
    rms_norm<InT, OutT>(*helper_);
}

// PyTorch-specific method: compute ith PyTorch tensors
// (zero-copy)
template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::compute_with_pytorch_tensors(
    const std::vector<py::object>& input_tensors,
    std::vector<py::object>& output_tensors,
    const std::vector<py::object>& weight_tensors,
    std::vector<py::object>& rsigma_tensors,
    float epsilon)
{
    /*
       if (!input_resources_.is_initialized())
       {
       throw std::runtime_error("RMSNormExecutor not initialized");
       }
     */
    if (!external_memory_)
    {
        int nGPU = input_tensors.size();

        //    int matrix_size =
        //    input_tensors[0].attr("numel")().cast<int>();
        int n_rows = input_tensors[0]
                         .attr("size")()
                         .cast<py::tuple>()[0]
                         .cast<int>();
        int n_cols = input_tensors[0]
                         .attr("size")()
                         .cast<py::tuple>()[1]
                         .cast<int>();

        //   int g_size =
        //   weight_tensors[0].attr("numel")().cast<int>();
        //  int rsigma_size =
        //  rsigma_tensors[0].attr("numel")().cast<int>();

        // Extract pointers from PyTorch tensors
        std::vector<uintptr_t> input_ptrs;

        for (const auto& tensor_obj : input_tensors)
        {
            py::object data_ptr_attr = tensor_obj.attr("data_ptr");
            uintptr_t ptr = data_ptr_attr().cast<uintptr_t>();
            input_ptrs.push_back(ptr);
        }

        // Extract pointers from PyTorch tensors
        std::vector<uintptr_t> output_ptrs;

        for (const auto& tensor_obj : output_tensors)
        {
            py::object data_ptr_attr = tensor_obj.attr("data_ptr");
            uintptr_t ptr = data_ptr_attr().cast<uintptr_t>();
            output_ptrs.push_back(ptr);
        }

        // Extract pointers from PyTorch tensors
        std::vector<uintptr_t> g_ptrs;

        for (const auto& tensor_obj : weight_tensors)
        {
            py::object data_ptr_attr = tensor_obj.attr("data_ptr");
            uintptr_t ptr = data_ptr_attr().cast<uintptr_t>();
            g_ptrs.push_back(ptr);
        }

        // Extract pointers from PyTorch tensors
        std::vector<uintptr_t> rsigma_ptrs;

        if (!rsigma_tensors.empty())
        {
            for (const auto& tensor_obj : rsigma_tensors)
            {
                py::object data_ptr_attr =
                    tensor_obj.attr("data_ptr");
                uintptr_t ptr = data_ptr_attr().cast<uintptr_t>();
                rsigma_ptrs.push_back(ptr);
            }
        }

        initialize_with_external_memory(nGPU,
                                        n_rows,
                                        n_cols,
                                        n_cols,
                                        n_cols,
                                        input_ptrs,
                                        output_ptrs,
                                        g_ptrs,
                                        rsigma_ptrs,
                                        epsilon);
    }

    // run computation
    compute();
}

// PyTorch-specific method: compute ith PyTorch tensors
// (zero-copy)

template <typename InT, typename OutT>
void RMSNormExecutor<InT, OutT>::compute_with_distributed_tensors(
    at::Tensor& input_tensor,
    at::Tensor& output_tensor,
    at::Tensor& weight_tensor,
    at::Tensor& rsigma_tensor,
    float epsilon,
    py::object& py_process_group)
{
    rank_ = py_process_group.attr("rank")().cast<int>();
    nGPU_ = py_process_group.attr("size")().cast<int>();

    n_rows_ = output_tensor.size(0);
    n_cols_ = output_tensor.size(1);
    input_row_stride_ = n_cols_;
    output_row_stride_ = n_cols_;

    input_resources_.initialize_with_distributed_external_torch(
        input_tensor, py_process_group);

    output_resources_.initialize_with_distributed_external_torch(
        output_tensor, py_process_group);
    g_resources_.initialize_with_distributed_external_torch(
        weight_tensor, py_process_group);
    rsigma_resources_.initialize_with_distributed_external_torch(
        rsigma_tensor, py_process_group);

    initialized_ = true;

    epsilon_ = epsilon;

    recreate_helper();

    // run computation
    compute();
}

template <typename InT, typename OutT>
py::capsule RMSNormExecutor<InT, OutT>::download(int id)
{
    auto num_rows = num_rows_per_gpu();
    auto stride = output_resources_.stride();
    auto size = num_rows * stride;

    // 1. Allocate pinned host memory
    std::unique_ptr<OutT[]> h_pinned =
        std::unique_ptr<OutT[]>(new OutT[size]);
    OutT* raw_host_data = h_pinned.get();
    //   HIP_CALL(hipHostMalloc(&h_pinned, size*sizeof(T)));

    // 2. Async copy GPU->pinned host
    auto output_device_buffers = output_resources_.device_buffers();
    HIP_CALL(hipMemcpy(raw_host_data,
                       output_device_buffers[id],
                       size * sizeof(OutT),
                       hipMemcpyDeviceToHost));

    //   HIP_CALL(hipStreamSynchronize(nullptr));

    // 3. Build DLPack on CPU
    HostTensorDLPacker<OutT>* dl_wrapper;
    if constexpr (std::is_same<OutT, __hip_bfloat16>::value)
    {
        dl_wrapper = new HostTensorDLPacker<OutT>(
            num_rows, stride, h_pinned.release(), kDLBfloat);
    }
    else if constexpr (std::is_same<OutT, __hip_fp8_storage_t>::value)
    {
        dl_wrapper = new HostTensorDLPacker<OutT>(
            num_rows, stride, h_pinned.release(), kDLFloat8_e4m3);
    }

    return py::capsule(
        dl_wrapper->get_managed_tensor(),
        "dltensor",
        [](PyObject* p) {
            DLManagedTensor* t = static_cast<DLManagedTensor*>(
                PyCapsule_GetPointer(p, "dltensor"));

            if (PyErr_Occurred())
            {
                PyErr_Clear();
            }

            if (t && t->deleter)
            {
                t->deleter(t);
            }
        });
}

template <typename InT, typename OutT>
py::capsule RMSNormExecutor<InT, OutT>::to_dlpack(int gpu_id)
{
    if (!output_resources_.is_initialized())
    {
        throw std::runtime_error(
            "RMSNormExecutor output resources not"
            "initialized");
    }

    int nGPU = output_resources_.num_gpus();
    if (nGPU == 0)
    {
        throw std::runtime_error("No GPUs initialzed.");
    }

    auto num_rows = num_rows_per_gpu();
    auto stride = output_resources_.stride();
    auto size = num_rows * stride;

    static std::vector<int64_t> shape = {
        static_cast<int64_t>(num_rows), static_cast<int64_t>(stride)};

    static std::vector<int64_t> strides = {
        static_cast<int64_t>(stride), 1};

    // 1. Synchronize the primary device to ensure the computation is
    // complte
    HIP_CALL(hipSetDevice(gpu_id));
    HIP_CALL(hipDeviceSynchronize());

    // 2. Allocate and populate the DLTensor structure
    // We use std::unique_ptr to manage the DLTensor and
    // DLManagedTensor

    auto dl_managed_tensor = std::make_unique<DLManagedTensor>();
    auto dl_tensor = std::make_unique<DLTensor>();

    // Set the DLTensor fields
    dl_tensor->data = output_resources_.device_buffers()[gpu_id];

    dl_tensor->ndim = 2;
    dl_tensor->shape = shape.data();
    dl_tensor->strides = strides.data();
    dl_tensor->byte_offset = 0;

    // Device information (HIP/ROCm)
    dl_tensor->device.device_type = kDLROCM;
    dl_tensor->device.device_id = gpu_id;  // GPU gpu_id

    // Data Type
    //    std::string type_name = "bfloat16"; //
    //    py::detail::type_cast<T>::name();
    dl_tensor->dtype.code = DLType<OutT>::code;
    dl_tensor->dtype.bits = DLType<OutT>::bits;
    dl_tensor->dtype.lanes = DLType<OutT>::lanes;

    // Set the DLManagedTensor fields
    dl_managed_tensor->dl_tensor = *dl_tensor.release();
    dl_managed_tensor->manager_ctx =
        output_resources_.get_buffer_shared_ptrs()[gpu_id].get();
    dl_managed_tensor->deleter = [](DLManagedTensor* self) {
        delete self;
    };

    return py::capsule(
        dl_managed_tensor.release(), "dltensor", [](PyObject* cap) {
            DLManagedTensor* self = static_cast<DLManagedTensor*>(
                PyCapsule_GetPointer(cap, "dltensor"));

            if (PyErr_Occurred())
            {
                PyErr_Clear();
            }

            if (self && self->deleter)
            {
                self->deleter(self);
            }
        });
}

template class RMSNormExecutor<__hip_bfloat16>;
template class RMSNormExecutor<__hip_bfloat16, __hip_fp8_storage_t>;

py::object create_executor(const std::string& input_type,
                           const std::string& output_type = "")
{
    if (output_type.empty() || output_type == input_type)
    {
        if (input_type == "bf16")
        {
            return py::cast(new RMSNormExecutor<__hip_bfloat16>());
        }
    }
    else
    {
        if (input_type == "bf16" && output_type == "fp8")
        {
            return py::cast(
                new RMSNormExecutor<__hip_bfloat16,
                                    __hip_fp8_storage_t>());
        }
    }
    throw std::runtime_error("Unsupported type combination");
}

template <typename InT, typename OutT>
uintptr_t RMSNormExecutor<InT, OutT>::map_handle(
    const hipIpcMemHandle_t handle)
{
    void* dev_ptr;
    HIP_CALL(hipIpcOpenMemHandle(
        (void**) &dev_ptr, handle, hipIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<uintptr_t>(dev_ptr);
}

PYBIND11_MODULE(_rms, m)
{
    m.doc() = "MultiGPU RMSNorm";

    auto create_executor_name = [](const std::string& input_type,
                                   const std::string& output_type =
                                       "") {
        if (output_type.empty() || output_type == input_type)
        {
            return "RMSNormExecutor_" + input_type;
        }
        else
        {
            return "RMSNormExecutor_" + input_type + "_to_" +
                   output_type;
        }
    };

    // Factory function
    m.def("create_executor",
          &create_executor,
          "Create RMSNormExecutor based on type strings",
          py::arg("input_type"),
          py::arg("output_type") = "");

    // Bind bfloat16 -> bfloat16
    {
        using ExecutorType = RMSNormExecutor<__hip_bfloat16>;
        std::string name = create_executor_name("b16");
        py::class_<ExecutorType>(m, name.c_str())
            .def(py::init<>())
            .def("initialize",
                 &ExecutorType::initialize,
                 "Initialize with GPU count and tensor size",
                 py::arg("n_gpu"),
                 py::arg("n_rows"),
                 py::arg("n_cols"),
                 py::arg("input_row_stride"),
                 py::arg("output_row_stride"),
                 py::arg("epsilon"))
            .def("initialize_with_external_memory",
                 &ExecutorType::initialize_with_external_memory,
                 "Initialize with external GPU memory pointers (vLLM "
                 "workflow)",
                 py::arg("n_gpu"),
                 py::arg("n_rows"),
                 py::arg("n_cols"),
                 py::arg("input_row_stride"),
                 py::arg("output_row_stride"),
                 py::arg("input_ptrs"),
                 py::arg("output_ptrs"),
                 py::arg("g_ptrs"),
                 py::arg("rsigma_ptrs"),
                 py::arg("epsilon"))
            .def(
                "compute",
                &ExecutorType::compute,
                "Run RMS computation (assume data is already on GPU)")
            .def("release",
                 &ExecutorType::release,
                 "Release resources")
            .def("compute_with_host_data",
                 &ExecutorType::compute_with_host_data,
                 "Run RMS computation with host data copying",
                 py::arg("inputs"))
            .def("compute_with_pytorch_tensors",
                 &ExecutorType::compute_with_pytorch_tensors,
                 "Run RMS computation with PyTorch tensors "
                 "(zero-copy)",
                 py::arg("input_tensors"),
                 py::arg("output_tensors"),
                 py::arg("weight_tensors"),
                 py::arg("rsigma_tensors"),
                 py::arg("epsilon"))
            .def("compute_with_distributed_tensors",
                 &ExecutorType::compute_with_distributed_tensors,
                 "Run RMS computation with PyTorch tensors "
                 "(zero-copy)",
                 py::arg("input_tensor"),
                 py::arg("output_tensor"),
                 py::arg("weight_tensor"),
                 py::arg("rsigma_tensor"),
                 py::arg("epsilon"),
                 py::arg("process_group"))
            .def("upload",
                 &ExecutorType::upload,
                 "Upload data to devices")
            .def("download",
                 &ExecutorType::download,
                 "Download data to host)",
                 py::arg("gpu_id"))
            .def("to_dlpack",
                 &ExecutorType::to_dlpack,
                 "return py::capsule)",
                 py::arg("gpu_id"))
            .def_property_readonly("is_initialized",
                                   &ExecutorType::is_initialized)
            .def_property_readonly(
                "using_external_memory",
                &ExecutorType::using_external_memory)
            .def_property_readonly("num_gpus",
                                   &ExecutorType::num_gpus)
            .def_property_readonly("num_rows_per_gpu",
                                   &ExecutorType::num_rows_per_gpu)
            .def_property_readonly("num_cols_per_gpu",
                                   &ExecutorType::num_cols_per_gpu);
    }
    {
        using ExecutorType =
            RMSNormExecutor<__hip_bfloat16, __hip_fp8_storage_t>;
        std::string name = create_executor_name("b16", "fp8");
        py::class_<ExecutorType>(m, name.c_str())
            .def(py::init<>())
            .def("initialize",
                 &ExecutorType::initialize,
                 "Initialize with GPU count and tensor size",
                 py::arg("n_gpu"),
                 py::arg("n_rows"),
                 py::arg("n_cols"),
                 py::arg("input_row_stride"),
                 py::arg("output_row_stride"),
                 py::arg("epsilon"))
            .def("initialize_with_external_memory",
                 &ExecutorType::initialize_with_external_memory,
                 "Initialize with external GPU memory pointers (vLLM "
                 "workflow)",
                 py::arg("n_gpu"),
                 py::arg("n_rows"),
                 py::arg("n_cols"),
                 py::arg("input_row_stride"),
                 py::arg("output_row_stride"),
                 py::arg("input_ptrs"),
                 py::arg("output_ptrs"),
                 py::arg("g_ptrs"),
                 py::arg("rsigma_ptrs"),
                 py::arg("epsilon"))
            .def(
                "compute",
                &ExecutorType::compute,
                "Run RMS computation (assume data is already on GPU)")
            .def("release",
                 &ExecutorType::release,
                 "Release resources")
            .def("compute_with_host_data",
                 &ExecutorType::compute_with_host_data,
                 "Run RMS computation with host data copying",
                 py::arg("inputs"))
            .def("compute_with_pytorch_tensors",
                 &ExecutorType::compute_with_pytorch_tensors,
                 "Run RMS computation with PyTorch tensors "
                 "(zero-copy)",
                 py::arg("input_tensors"),
                 py::arg("output_tensors"),
                 py::arg("weight_tensors"),
                 py::arg("rsigma_tensors"),
                 py::arg("epsilon"))
            .def("compute_with_distributed_tensors",
                 &ExecutorType::compute_with_distributed_tensors,
                 "Run RMS computation with PyTorch tensors "
                 "(zero-copy)",
                 py::arg("input_tensor"),
                 py::arg("output_tensor"),
                 py::arg("weight_tensor"),
                 py::arg("rsigma_tensor"),
                 py::arg("epsilon"),
                 py::arg("process_group"))
            .def("upload",
                 &ExecutorType::upload,
                 "Upload data to devices")
            .def("download",
                 &ExecutorType::download,
                 "Download data to host)",
                 py::arg("gpu_id"))
            .def("to_dlpack",
                 &ExecutorType::to_dlpack,
                 "return py::capsule)",
                 py::arg("gpu_id"))
            .def_property_readonly("is_initialized",
                                   &ExecutorType::is_initialized)
            .def_property_readonly(
                "using_external_memory",
                &ExecutorType::using_external_memory)
            .def_property_readonly("num_gpus",
                                   &ExecutorType::num_gpus)
            .def_property_readonly("num_rows_per_gpu",
                                   &ExecutorType::num_rows_per_gpu)
            .def_property_readonly("num_cols_per_gpu",
                                   &ExecutorType::num_cols_per_gpu);
    }

    // Optional: expose device count
    m.def("get_gpu_count", []() {
        int count = 0;
        HIP_CALL(hipGetDeviceCount(&count));
        return count;
    });
}

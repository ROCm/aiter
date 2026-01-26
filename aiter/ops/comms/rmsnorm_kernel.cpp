#include "rmsnorm.h"

template <typename InT, typename OutT = InT>
__global__ void _all_gather_reduce_scatter(OutT** output_ptr,
                                           InT** input_ptr,
                                           InT** g_ptr,
                                           OutT** rsigma_ptr,
                                           int output_row_stride,
                                           int input_row_stride,
                                           int dev,
                                           int nGPU,
                                           int start,
                                           int n_local_rows,
                                           int n_cols,
                                           int warp_size,
                                           float epsilon)
{
    __shared__ float sh[16];
    __shared__ float fnorm[16];

    int bx = nGPU;
    int by = blockDim.x / nGPU;
    int bz = 1;
    int tx = threadIdx.x % nGPU;
    int ty = threadIdx.x / nGPU;
    int width = blockDim.x / warp_size;

    int row_ind = blockIdx.y * bz;
    int row_inc = gridDim.y;
    int col_ind = blockIdx.x * by + ty;
    int col_inc = by * gridDim.x;

    using vec_type = typename PackData<InT, 8>::vec_type;
    using out_vec_type = typename PackData<OutT, 8>::vec_type;

    out_vec_type* output_vec =
        reinterpret_cast<out_vec_type*>(output_ptr[dev]);

    vec_type* g_vec = reinterpret_cast<vec_type*>(g_ptr[dev]);

    int output_row_stride8 = output_row_stride >> 3;

    float rn_cols = 1.0 / n_cols;
    for (int row = row_ind; row < n_local_rows; row += row_inc)
    {
        float sum = 0.0f;
        for (int col = col_ind; col < n_cols; col += col_inc)
        {
            float input_value = dtype<InT>::to_float(
                input_ptr[tx]
                         [(row + start) * input_row_stride + col]);
            for (int mask = nGPU >> 1; mask > 0; mask = mask >> 1)
            {
                input_value += __shfl_xor(input_value, mask, nGPU);
            }
            sum = fmaf(input_value, input_value, sum);
            if (tx == 0)
            {
                output_ptr[dev]
                          [(row + start) * output_row_stride + col] =
                              dtype<OutT>::from_float(input_value);
            }
        }

        for (int mask = warp_size >> 1; mask >= nGPU;
             mask = mask >> 1)
        {
            sum += __shfl_xor(sum, mask);
        }
        if (threadIdx.x % warp_size == 0)
        {
            sh[threadIdx.x / warp_size] = sum;
        }
        __syncthreads();
        sum = sh[threadIdx.x % width];
        for (int mask = width >> 1; mask > 0; mask = mask >> 1)
        {
            sum += __shfl_xor(sum, mask, width);
        }
        if (threadIdx.x == 0)
        {
            fnorm[row / row_inc] = rsqrt(fmaf(sum, rn_cols, epsilon));
        }
    }

    bz = nGPU;
    bx = blockDim.x / nGPU;
    by = 1;
    tx = threadIdx.x % bx;
    ty = 0;
    int tz = threadIdx.x / bx;

    row_ind = blockIdx.y * by + ty;
    row_inc = gridDim.y;
    col_ind = blockIdx.x * bx + tx;
    col_inc = gridDim.x * bx;

    out_vec_type* dest_vec =
        reinterpret_cast<out_vec_type*>(output_ptr[tz]);

    for (int row = row_ind; row < n_local_rows; row += row_inc)
    {
        float rnorm = fnorm[row / row_inc];

        for (int j = col_ind; j < n_cols >> 3; j += col_inc)
        {
            PackData<OutT, 8> output_data;
            PackData<InT, 8> g_data;
            output_data.vec_ =
                output_vec[(row + start) * output_row_stride8 + j];
            __syncthreads();
            g_data.vec_ = g_vec[j];
#pragma unroll
            for (int s = 0; s < 8; s++)
            {
                float v1 = dtype<OutT>::to_float(output_data[s]);
                float g_value = dtype<InT>::to_float(g_data[s]);
                output_data[s] =
                    dtype<OutT>::from_float(v1 * g_value * rnorm);
            }
            dest_vec[(row + start) * output_row_stride8 + j] =
                output_data.vec_;
        }
        if (tx == 0 && rsigma_ptr != nullptr)
        {
            rsigma_ptr[tz][row + start] =
                dtype<OutT>::from_float(rnorm);
        }
    }
}

template <typename InT, typename OutT = InT>
__global__ void _gather_scatter_workgroup(OutT** output_ptr,
                                          InT** input_ptr,
                                          float* sum_ptr,
                                          int output_row_stride,
                                          int input_row_stride,
                                          int dev,
                                          int nGPU,
                                          int start,
                                          int n_local_rows,
                                          int n_cols,
                                          int warp_size)
{
    __shared__ float sh[16];

    int bx = nGPU;
    int by = blockDim.x / nGPU;
    int bz = 1;
    int tx = threadIdx.x % nGPU;
    int ty = threadIdx.x / nGPU;
    int width = blockDim.x / warp_size;

    int row_ind = blockIdx.y * bz;
    int row_inc = gridDim.y;
    int col_ind = blockIdx.x * by + ty;
    int col_inc = by * gridDim.x;

    for (int row = row_ind; row < n_local_rows; row += row_inc)
    {
        float sum = 0.0f;
        for (int col = col_ind; col < n_cols; col += col_inc)
        {
            float input_value = dtype<InT>::to_float(
                input_ptr[tx]
                         [(row + start) * input_row_stride + col]);

            for (int mask = nGPU >> 1; mask > 0; mask = mask >> 1)
            {
                input_value += __shfl_xor(input_value, mask, nGPU);
            }
            sum = fmaf(input_value, input_value, sum);
            if (tx == 0)
            {
                output_ptr[dev]
                          [(row + start) * output_row_stride + col] =
                              dtype<OutT>::from_float(input_value);
            }
        }

        for (int mask = warp_size >> 1; mask >= nGPU;
             mask = mask >> 1)
        {
            sum += __shfl_xor(sum, mask);
        }
        if (threadIdx.x % warp_size == 0)
        {
            sh[threadIdx.x / warp_size] = sum;
        }
        __syncthreads();
        sum = sh[threadIdx.x % width];
        for (int mask = width >> 1; mask > 0; mask = mask >> 1)
        {
            sum += __shfl_xor(sum, mask, width);
        }
        if (threadIdx.x == 0)
        {
            sum_ptr[row * gridDim.x + blockIdx.x] = sum;
        }
    }
}

template <typename InT, typename OutT = InT>
__global__ void _rms_norm_kernel(OutT** output_ptr,
                                 float* sum_ptr,
                                 InT** g_ptr,
                                 OutT** rsigma_ptr,
                                 int output_row_stride,
                                 int dev,
                                 int nGPU,
                                 int start,
                                 int n_rows,
                                 int n_cols,
                                 float epsilon)
{
    int bz = nGPU;
    int bx = blockDim.x / nGPU;
    int by = 1;
    int tx = threadIdx.x % bx;
    int ty = 0;
    int tz = threadIdx.x / bx;

    int row_ind = blockIdx.y * by + ty;
    int row_inc = gridDim.y * by;
    int col_ind = blockIdx.x * bx + tx;
    int col_inc = gridDim.x * bx;

    using vec_type = typename PackData<InT, 8>::vec_type;
    using out_vec_type = typename PackData<OutT, 8>::vec_type;

    out_vec_type* output_vec =
        reinterpret_cast<out_vec_type*>(output_ptr[dev]);
    out_vec_type* dest_vec = reinterpret_cast<out_vec_type*>(
        output_ptr[(tz + dev) % nGPU]);

    vec_type* g_vec = reinterpret_cast<vec_type*>(g_ptr[dev]);

    int output_row_stride8 = output_row_stride >> 3;
    float rn_cols = 1.0 / n_cols;

    for (int row = row_ind; row < n_rows; row += row_inc)
    {
        float sum = 0.0f;

        if (gridDim.x >= bx)
        {
            for (int j = tx; j < gridDim.x; j += bx)
            {
                sum += sum_ptr[row * gridDim.x + j];
            }

            for (int mask = bx >> 1; mask > 0; mask = mask >> 1)
            {
                sum += __shfl_xor(sum, mask, bx);
            }
        }
        else
        {
            sum += sum_ptr[row * gridDim.x + tx % gridDim.x];

            for (int mask = gridDim.x >> 1; mask > 0;
                 mask = mask >> 1)
            {
                sum += __shfl_xor(sum, mask, gridDim.x);
            }
        }

        float rnorm = rsqrt(fmaf(sum, rn_cols, epsilon));

        for (int j = col_ind; j < n_cols >> 3; j += col_inc)
        {
            PackData<OutT, 8> output_data;
            PackData<InT, 8> g_data;
            output_data.vec_ =
                output_vec[(row + start) * output_row_stride8 + j];
            g_data.vec_ = g_vec[j];
            __syncthreads();
#pragma unroll
            for (int s = 0; s < 8; s++)
            {
                float v1 = dtype<OutT>::to_float(output_data[s]);
                float g_value = dtype<InT>::to_float(g_data[s]);
                output_data[s] =
                    dtype<OutT>::from_float(v1 * g_value * rnorm);
            }
            dest_vec[(row + start) * output_row_stride / 8 + j] =
                output_data.vec_;
        }
        if (rsigma_ptr != nullptr && tx == 0 && blockIdx.x == 0)
        {
            rsigma_ptr[tz][row + start] =
                dtype<OutT>::from_float(rnorm);
        }
    }
}

template <typename InT, typename OutT>
RMSNormHelper<InT, OutT>::RMSNormHelper(
    int nGPU,
    int n_rows,
    int n_cols,
    int input_row_stride,
    int output_row_stride,
    std::vector<InT**> input_ptrs,
    std::vector<OutT**> output_ptrs,
    std::vector<InT**> g_ptrs,
    std::vector<OutT**> rsigma_ptrs,
    float epsilon)
    : nGPU_(nGPU),
      n_rows_(n_rows),
      n_cols_(n_cols),
      input_row_stride_(input_row_stride),
      output_row_stride_(output_row_stride),
      input_ptrs_(input_ptrs),
      output_ptrs_(output_ptrs),
      g_ptrs_(g_ptrs),
      rsigma_ptrs_(rsigma_ptrs),
      epsilon_(epsilon)
{
    HIP_CALL(hipSetDevice(0));
    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));

    rank_ = -1;
    warp_size_ = props.warpSize;
    num_cus_ = props.multiProcessorCount;

    sh_ptr_per_wg_vec_.resize(nGPU_);
    ptr_per_wg_vec_.resize(nGPU_);

    if (nGPU_ >= 1)
    {
        dev_vec_.resize(nGPU_);
        std::iota(dev_vec_.begin(), dev_vec_.end(), 0);

        start_vec_.resize(nGPU_);

        graph_vec_.resize(nGPU_);
        graph_instance_vec_.resize(nGPU_);
        graph_nodeA_vec_.resize(nGPU_);
        graph_nodeB_vec_.resize(nGPU_);
        graph_nodeC_vec_.resize(nGPU_);

        int n_row_block = (n_rows + nGPU_ - 1) / nGPU_;
        n_local_rows_.resize(nGPU_);

        for (int i = 0; i < nGPU_; i++)
        {
            HIP_CALL(hipSetDevice(i));
            HIP_CALL(hipGraphCreate(&graph_vec_[i], 0));

            int start = i * n_row_block;
            int end = start + n_row_block < n_rows
                          ? start + n_row_block
                          : n_rows;

            n_local_rows_[i] = end - start;
            start_vec_[i] = start;

            if (n_rows >= num_cus_ * 4)
            {
                dim3 grid(1, num_cus_ * 4);
                dim3 allreduce_block(512);

                hipKernelNodeParams kpA{};
                OutT** local_output_ptr_ = output_ptrs_[i];
                InT** local_input_ptr_ = input_ptrs_[i];
                InT** local_g_ptr_ = g_ptrs_[i];
                OutT** local_rsigma_ptr_ = rsigma_ptrs_[i];
                void* argsA[] = {(void*) &local_output_ptr_,
                                 (void*) &local_input_ptr_,
                                 (void*) &local_g_ptr_,
                                 (void*) &local_rsigma_ptr_,
                                 (void*) &output_row_stride_,
                                 (void*) &input_row_stride_,
                                 (void*) &dev_vec_[i],
                                 (void*) &nGPU_,
                                 (void*) &start_vec_[i],
                                 (void*) &n_local_rows_[i],
                                 (void*) &n_cols_,
                                 (void*) &warp_size_,
                                 (void*) &epsilon};
                kpA.func =
                    (void*) _all_gather_reduce_scatter<InT, OutT>;
                kpA.gridDim = grid;
                kpA.blockDim = allreduce_block;
                kpA.kernelParams = argsA;
                kpA.sharedMemBytes = 0;
                HIP_CALL(hipGraphAddKernelNode(&graph_nodeA_vec_[i],
                                               graph_vec_[i],
                                               nullptr,
                                               0,
                                               &kpA));
                HIP_CALL(hipGraphInstantiate(&graph_instance_vec_[i],
                                             graph_vec_[i],
                                             nullptr,
                                             nullptr,
                                             0));
            }
            else
            {
                dim3 gather_block(512);
                dim3 gather_grid(num_cus_ / 4, 4);

                // dim3 rms_block(warp_size_, 1, nGPU_);
                dim3 rms_block(512);
                dim3 rms_grid(num_cus_ / 4, 4);

                if (n_row_block >= 4)
                {
                    gather_grid.y = 4;
                    rms_grid.y = 4;
                    gather_grid.x = num_cus_ / gather_grid.y;
                    rms_grid.x = num_cus_ / rms_grid.y;
                }

                sh_ptr_per_wg_vec_[i] = std::shared_ptr<float>(
                    (float*) hipMallocFunc(n_row_block * rms_grid.x *
                                           sizeof(float)),
                    hipDeleteHelper(i));
                ptr_per_wg_vec_[i] = sh_ptr_per_wg_vec_[i].get();

                hipKernelNodeParams kpA{};
                OutT** local_output_ptr_ = output_ptrs_[i];
                InT** local_input_ptr_ = input_ptrs_[i];
                void* argsA[] = {(void*) &local_output_ptr_,
                                 (void*) &local_input_ptr_,
                                 (void*) &ptr_per_wg_vec_[i],
                                 (void*) &output_row_stride_,
                                 (void*) &input_row_stride_,
                                 (void*) &dev_vec_[i],
                                 (void*) &nGPU_,
                                 (void*) &start_vec_[i],
                                 (void*) &n_local_rows_[i],
                                 (void*) &n_cols_,
                                 (void*) &warp_size_};
                kpA.func =
                    (void*) _gather_scatter_workgroup<InT, OutT>;
                kpA.gridDim = gather_grid;
                kpA.blockDim = gather_block;
                kpA.kernelParams = argsA;
                kpA.sharedMemBytes = 0;
                HIP_CALL(hipGraphAddKernelNode(&graph_nodeA_vec_[i],
                                               graph_vec_[i],
                                               nullptr,
                                               0,
                                               &kpA));

                hipKernelNodeParams kpC{};
                InT** local_g_ptr_ = g_ptrs_[i];
                OutT** local_rsigma_ptr_ = rsigma_ptrs_[i];
                void* argsC[] = {(void*) &local_output_ptr_,
                                 (void*) &ptr_per_wg_vec_[i],
                                 (void*) &local_g_ptr_,
                                 (void*) &local_rsigma_ptr_,
                                 (void*) &output_row_stride_,
                                 (void*) &dev_vec_[i],
                                 (void*) &nGPU_,
                                 (void*) &start_vec_[i],
                                 (void*) &n_local_rows_[i],
                                 (void*) &n_cols_,
                                 (void*) &epsilon_};
                kpC.func = (void*) _rms_norm_kernel<InT, OutT>;
                kpC.gridDim = rms_grid;
                kpC.blockDim = rms_block;
                kpC.kernelParams = argsC;
                kpC.sharedMemBytes = 0;
                HIP_CALL(hipGraphAddKernelNode(&graph_nodeC_vec_[i],
                                               graph_vec_[i],
                                               &graph_nodeA_vec_[i],
                                               1,
                                               &kpC));

                HIP_CALL(hipGraphInstantiate(&graph_instance_vec_[i],
                                             graph_vec_[i],
                                             nullptr,
                                             nullptr,
                                             0));
            }
        }
    }
}
template RMSNormHelper<__half>::RMSNormHelper(
    int nGPU,
    int n_rows,
    int n_cols,
    int input_row_stride,
    int output_row_stride,
    std::vector<__half**> input_ptrs,
    std::vector<__half**> output_ptrs,
    std::vector<__half**> g_ptrs,
    std::vector<__half**> rsigma_ptrs,
    float epsilon);

template RMSNormHelper<__hip_bfloat16>::RMSNormHelper(
    int nGPU,
    int n_rows,
    int n_cols,
    int input_row_stride,
    int output_row_stride,
    std::vector<__hip_bfloat16**> input_ptrs,
    std::vector<__hip_bfloat16**> output_ptrs,
    std::vector<__hip_bfloat16**> g_ptrs,
    std::vector<__hip_bfloat16**> rsigma_ptrs,
    float epsilon);

template RMSNormHelper<__hip_bfloat16, __hip_fp8_storage_t>::
    RMSNormHelper(int nGPU,
                  int n_rows,
                  int n_cols,
                  int input_row_stride,
                  int output_row_stride,
                  std::vector<__hip_bfloat16**> input_ptrs,
                  std::vector<__hip_fp8_storage_t**> output_ptrs,
                  std::vector<__hip_bfloat16**> g_ptrs,
                  std::vector<__hip_fp8_storage_t**> rsigma_ptrs,
                  float epsilon);

template <typename InT, typename OutT>
RMSNormHelper<InT, OutT>::RMSNormHelper(int nGPU,
                                        int rank,
                                        int n_rows,
                                        int n_cols,
                                        int input_row_stride,
                                        int output_row_stride,
                                        InT** mapped_input_ptrs,
                                        OutT** mapped_output_ptrs,
                                        InT** mapped_g_ptrs,
                                        OutT** mapped_rsigma_ptrs,
                                        float epsilon)
    : nGPU_(nGPU),
      rank_(rank),
      n_rows_(n_rows),
      n_cols_(n_cols),
      input_row_stride_(input_row_stride),
      output_row_stride_(output_row_stride),
      mapped_input_ptrs_(mapped_input_ptrs),
      mapped_output_ptrs_(mapped_output_ptrs),
      mapped_g_ptrs_(mapped_g_ptrs),
      mapped_rsigma_ptrs_(mapped_rsigma_ptrs)
{
    HIP_CALL(hipSetDevice(rank));
    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));

    warp_size_ = props.warpSize;
    num_cus_ = props.multiProcessorCount;
    epsilon_ = epsilon;

    if (nGPU_ >= 1)
    {
        int n_row_block = (n_rows + nGPU_ - 1) / nGPU_;

        HIP_CALL(hipSetDevice(rank_));
        HIP_CALL(hipGraphCreate(&dist_graph_, 0));

        int dist_start_ = rank_ * n_row_block;
        int end = dist_start_ + n_row_block < n_rows_
                      ? dist_start_ + n_row_block
                      : n_rows_;

        dist_local_rows_ = end - dist_start_;

        if (n_rows >= num_cus_ * 4)
        {
            dim3 grid(1, num_cus_ * 4);
            dim3 allreduce_block(512);

            hipKernelNodeParams kpA{};
            OutT** local_output_ptr_ = mapped_output_ptrs_;
            InT** local_input_ptr_ = mapped_input_ptrs_;
            InT** local_g_ptr_ = mapped_g_ptrs_;
            OutT** local_rsigma_ptr_ = mapped_rsigma_ptrs_;
            void* argsA[] = {&local_output_ptr_,
                             &local_input_ptr_,
                             (void*) &local_g_ptr_,
                             (void*) &local_rsigma_ptr_,
                             (void*) &output_row_stride_,
                             (void*) &input_row_stride_,
                             (void*) &rank_,
                             (void*) &nGPU_,
                             (void*) &dist_start_,
                             (void*) &dist_local_rows_,
                             (void*) &n_cols_,
                             (void*) &warp_size_,
                             (void*) &epsilon_};
            kpA.func = (void*) _all_gather_reduce_scatter<InT, OutT>;
            kpA.gridDim = grid;
            kpA.blockDim = allreduce_block;
            kpA.kernelParams = argsA;
            kpA.sharedMemBytes = 0;
            HIP_CALL(hipGraphAddKernelNode(
                &dist_graph_nodeA_, dist_graph_, nullptr, 0, &kpA));
            HIP_CALL(hipGraphInstantiate(&dist_graph_instance_,
                                         dist_graph_,
                                         nullptr,
                                         nullptr,
                                         0));
        }
        else
        {
            dim3 gather_block(512);
            dim3 gather_grid(num_cus_ / 4, 4);

            // dim3 rms_block(warp_size_, 1, nGPU_);
            dim3 rms_block(512);
            dim3 rms_grid(num_cus_ / 4, 4);

            if (n_row_block >= 4)
            {
                gather_grid.y = 4;
                rms_grid.y = 4;
                gather_grid.x = num_cus_ / gather_grid.y;
                rms_grid.x = num_cus_ / rms_grid.y;
            }

            dist_sh_ptr_per_wg_ = std::shared_ptr<float>(
                (float*) hipMallocFunc(n_row_block * rms_grid.x *
                                       sizeof(float)),
                hipDeleteHelper(rank_));
            dist_ptr_per_wg_ = dist_sh_ptr_per_wg_.get();

            hipKernelNodeParams kpA{};
            OutT** local_output_ptr_ = mapped_output_ptrs_;
            InT** local_input_ptr_ = mapped_input_ptrs_;
            void* argsA[] = {&local_output_ptr_,
                             &local_input_ptr_,
                             &dist_ptr_per_wg_,
                             (void*) &output_row_stride_,
                             (void*) &input_row_stride_,
                             (void*) &rank_,
                             (void*) &nGPU_,
                             (void*) &dist_start_,
                             (void*) &dist_local_rows_,
                             (void*) &n_cols_,
                             (void*) &warp_size_};
            kpA.func = (void*) _gather_scatter_workgroup<InT, OutT>;
            kpA.gridDim = gather_grid;
            kpA.blockDim = gather_block;
            kpA.kernelParams = argsA;
            kpA.sharedMemBytes = 0;
            HIP_CALL(hipGraphAddKernelNode(
                &dist_graph_nodeA_, dist_graph_, nullptr, 0, &kpA));

            hipKernelNodeParams kpC{};
            InT** local_g_ptr_ = mapped_g_ptrs_;
            OutT** local_rsigma_ptr_ = mapped_rsigma_ptrs_;
            void* argsC[] = {&local_output_ptr_,
                             &dist_ptr_per_wg_,
                             &local_g_ptr_,
                             &local_rsigma_ptr_,
                             (void*) &output_row_stride_,
                             (void*) &rank_,
                             (void*) &nGPU_,
                             (void*) &dist_start_,
                             (void*) &dist_local_rows_,
                             (void*) &n_cols_,
                             (void*) &epsilon_};
            kpC.func = (void*) _rms_norm_kernel<InT, OutT>;
            kpC.gridDim = rms_grid;
            kpC.blockDim = rms_block;
            kpC.kernelParams = argsC;
            kpC.sharedMemBytes = 0;
            HIP_CALL(hipGraphAddKernelNode(&dist_graph_nodeC_,
                                           dist_graph_,
                                           &dist_graph_nodeA_,
                                           1,
                                           &kpC));

            HIP_CALL(hipGraphInstantiate(&dist_graph_instance_,
                                         dist_graph_,
                                         nullptr,
                                         nullptr,
                                         0));
        }
    }
}

template RMSNormHelper<__hip_bfloat16>::RMSNormHelper(
    int nGPU,
    int rank,
    int n_rows,
    int n_cols,
    int input_row_stride,
    int output_row_stride,
    __hip_bfloat16** mapped_input_ptrs,
    __hip_bfloat16** mapped_output_ptrs,
    __hip_bfloat16** mapped_g_ptrs,
    __hip_bfloat16** mapped_rsigma_ptrs,
    float epsilon);

template RMSNormHelper<__hip_bfloat16, __hip_fp8_storage_t>::
    RMSNormHelper(int nGPU,
                  int rank,
                  int n_rows,
                  int n_cols,
                  int input_row_stride,
                  int output_row_stride,
                  __hip_bfloat16** mapped_input_ptrs,
                  __hip_fp8_storage_t** mapped_output_ptrs,
                  __hip_bfloat16** mapped_g_ptrs,
                  __hip_fp8_storage_t** mapped_rsigma_ptrs,
                  float epsilon);

template <typename InT, typename OutT>
void rms_norm(RMSNormHelper<InT, OutT>& rms_helper)
{
    int device_id, nGPU;
    hipStream_t stream;

    if (rms_helper.rank_ == -1)
    {
        int nGPU = rms_helper.nGPU_;
#pragma omp parallel for num_threads(nGPU)
        for (int dev = 0; dev < nGPU; dev++)
        {
            rms_helper.launch_graph(dev);
        }
    }
    else
    {
        rms_helper.launch_graph();

#if 0
      dim3 block(512);
      dim3 grid(1, 1024);
      HIP_CALL(hipSetDevice(rms_helper.rank_));
      _all_gather_reduce_scatter<InT, OutT><<<grid,
          block>>>(rms_helper.mapped_output_ptrs_.data(),
		rms_helper.mapped_input_ptrs_.data(),
		rms_helper.mapped_g_ptrs_.data(),
		rms_helper.mapped_rsigma_ptrs_.data(),
		rms_helper.output_row_stride_,
		rms_helper.input_row_stride_,
		rms_helper.rank_,
		rms_helper.nGPU_,
		rms_helper.dist_start_,
		rms_helper.dist_local_rows_,
		rms_helper.n_cols_,
		rms_helper.warp_size_,
		rms_helper.epsilon_);
       ;
#endif
    }
}

#if 0
template <typename T>
void generate_random(std::vector<T> &v)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	
	std::ranges::generate(v, [&]() { return dtype<T>::from_float(dist(gen));});
}	

template
void generate_random<__half>(std::vector<__half> &v);

template
void generate_random<__hip_bfloat16>(std::vector<__hip_bfloat16> &v);
#endif

template void rms_norm<__half>(RMSNormHelper<__half>& rms_helper);

template void rms_norm<__hip_bfloat16>(
    RMSNormHelper<__hip_bfloat16>& rms_helper);

template void rms_norm<__hip_bfloat16, __hip_fp8_storage_t>(
    RMSNormHelper<__hip_bfloat16, __hip_fp8_storage_t>& rms_helper);

template <typename T>
int check(int dev,
          std::shared_ptr<T*>* input_ptr,
          std::shared_ptr<T*>* output_ptr,
          std::shared_ptr<T*>* g_ptr,
          float epsilon,
          int input_row_stride,
          int output_row_stride,
          int n_rows,
          int n_cols)
{
    std::vector<T> vec_h(n_rows * n_cols);
    std::vector<T> g_h(n_cols);
    dim3 block(512);
    dim3 grid;
    grid.x = (n_rows + block.x - 1) / block.x;
    int nGPUs;

    HIP_CALL(hipGetDeviceCount(&nGPUs));
    HIP_CALL(hipSetDevice(dev));

#if 0
	_check<T><<<grid,block>>>(input_ptr, vec_d.get(), g_ptr, epsilon, nGPUs,
														input_row_stride, 1,
														n_rows, n_cols);
#endif

    HIP_CALL(hipMemcpy(vec_h.data(),
                       input_ptr[dev].get()[dev],
                       n_rows * n_cols * sizeof(T),
                       hipMemcpyDeviceToHost));

    HIP_CALL(hipMemcpy(g_h.data(),
                       g_ptr[dev].get()[dev],
                       n_cols * sizeof(T),
                       hipMemcpyDeviceToHost));

    std::vector<T> output_h(n_rows * n_cols);
    HIP_CALL(hipMemcpy(output_h.data(),
                       output_ptr[dev].get()[dev],
                       n_rows * n_cols * sizeof(T),
                       hipMemcpyDeviceToHost));
    int result_OK = 1;
    int row_index, col_index;
    for (int i = 0; i < n_rows; i++)
    {
        float sum = 0.0f;
        ;
        for (int j = 0; j < n_cols; j++)
        {
            float input_value =
                dtype<T>::to_float(vec_h[i * n_cols + j]);
            float output_value =
                dtype<T>::to_float(output_h[i * n_cols + j]);
            /*
            if (fabs(norm_value-output_value) > 1.0e-2)
            {
                    std::cout << "output = " << output_value << "
            input = " << input_value << std::endl; result_OK = 0;
            row_index = i; col_index = j; printf("row = %d, col =
            %d\n", row_index, col_index); break;
            }
            */
            sum += input_value / n_cols * input_value;
        }
        sum *= (nGPUs * nGPUs);
        float rnorm = 1.0 / sqrt(sum + epsilon);
        for (int j = 0; j < n_cols; j++)
        {
            float input_value =
                dtype<T>::to_float(vec_h[i * n_cols + j]);
            float output_value =
                dtype<T>::to_float(output_h[i * n_cols + j]);
            float g = dtype<T>::to_float(g_h[j]);
            float norm_value = nGPUs * input_value * rnorm * g;

            T norm_value_half = dtype<T>::from_float(norm_value);
            norm_value = dtype<T>::to_float(norm_value_half);

            if (fabs(norm_value - output_value) > 1.0e-2)
            {
                std::cout << "output = " << output_value
                          << " input = " << norm_value << std::endl;
                result_OK = 0;
                row_index = i;
                col_index = j;
                printf("row = %d, col = %d\n", row_index, col_index);
                break;
            }
        }
        if (result_OK == 0)
        {
            break;
        }
    }
#if 0
	if (result_OK == 0)
	{
		std::cout << "row = " << row_index << " , col_index = " <<
			col_index << std::endl;
	}
#endif
    return result_OK;
}

template int check<__half>(int dev,
                           std::shared_ptr<__half*>* input_ptr,
                           std::shared_ptr<__half*>* output_ptr,
                           std::shared_ptr<__half*>* g_ptr,
                           float epsilon,
                           int input_row_stride,
                           int output_row_stride,
                           int n_rows,
                           int n_cols);

template int check<__hip_bfloat16>(
    int dev,
    std::shared_ptr<__hip_bfloat16*>* input_ptr,
    std::shared_ptr<__hip_bfloat16*>* output_ptr,
    std::shared_ptr<__hip_bfloat16*>* g_ptr,
    float epsilon,
    int input_row_stride,
    int output_row_stride,
    int n_rows,
    int n_cols);

void setup_multigpu()
{
    int nGPU;

    HIP_CALL(hipGetDeviceCount(&nGPU));
#pragma omp parallel for num_threads(nGPU)
    for (int dev = 0; dev < nGPU; dev++)
    {
        HIP_CALL(hipSetDevice(dev));
        for (int j = 1; j < nGPU; j++)
        {
            int other_dev = (dev + j) % nGPU;
            HIP_CALL(hipDeviceEnablePeerAccess(other_dev, 0));
        }
    }
}

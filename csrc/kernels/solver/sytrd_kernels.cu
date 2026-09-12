#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cmath>
#include <limits>
#include <type_traits>

namespace aiter {

// Template trait for vector types
template<typename T> struct vec4_type {};
template<> struct vec4_type<float> { using type = float4; };
template<> struct vec4_type<double> { using type = double4; };

// Template trait for literal constants
template<typename T> struct literal_constants {
    static constexpr T zero = static_cast<T>(0);
    static constexpr T one = static_cast<T>(1);
    static constexpr T half = static_cast<T>(0.5);
    static constexpr T epsilon = std::numeric_limits<T>::epsilon();
};

// Template wrappers for math functions
template<typename T> __device__ __forceinline__ T sqrt_func(T x);
template<> __device__ __forceinline__ float sqrt_func<float>(float x) { return sqrtf(x); }
template<> __device__ __forceinline__ double sqrt_func<double>(double x) { return sqrt(x); }

template<typename T> __device__ __forceinline__ T abs_func(T x);
template<> __device__ __forceinline__ float abs_func<float>(float x) { return fabsf(x); }
template<> __device__ __forceinline__ double abs_func<double>(double x) { return fabs(x); }

#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

// 2D index calculation for column-major storage
__device__ __host__ __forceinline__ int idx2D(const int i, const int j, const int lda) { return j * lda + i; }

// Warp-level reduction
template<typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction
template<typename scalar_t>
__device__ __forceinline__ void block_reduce_sum(scalar_t val, scalar_t *smem, int tid, int blockDimX) {
    val = warp_reduce_sum(val);

    if (blockDimX > warpSize) {
        int lane = tid & (warpSize - 1);
        int wid = tid / warpSize;
        if (lane == 0) {
            smem[wid] = val;
        }
        __syncthreads();

        if (tid < warpSize) {
            val = tid < CEIL_DIV(blockDimX, warpSize) ? smem[tid] : literal_constants<scalar_t>::zero;
            val = warp_reduce_sum(val);
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }

    // __syncthreads();
    // sync not needed if only thread 0 reads from smem[0]
}

// Kernel to compute Householder reflector (larfg) for a column vector
template<typename scalar_t>
__global__ void larfg_kernel(int m, int i, scalar_t* __restrict__ A, int lda,
                             scalar_t* __restrict__ dE, scalar_t* __restrict__ dTau) {
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    int tid = threadIdx.x;
    scalar_t* col_base = A + idx2D(i + 1, i, lda);

    // Accumulate sum of squares of tail (excluding first element)
    scalar_t local_sum = literal_constants<scalar_t>::zero;
    scalar_t alpha = literal_constants<scalar_t>::zero;

    int m_vec4 = m / 4;
    using vec4_t = typename vec4_type<scalar_t>::type;
    
    #pragma unroll 4
    for (int idx4 = tid; idx4 < m_vec4; idx4 += blockDim.x) {
        vec4_t val4 = reinterpret_cast<const vec4_t*>(&col_base[idx4 * 4])[0];
        
        // Handle first element specially (alpha extraction)
        if (idx4 == 0) {
            alpha = val4.x;
            local_sum += (val4.y * val4.y + val4.z * val4.z + val4.w * val4.w);
        } else {
            local_sum += (val4.x * val4.x + val4.y * val4.y + 
                         val4.z * val4.z + val4.w * val4.w);
        }
    }
    
    for (int idx = 4 * m_vec4 + tid; idx < m; idx += blockDim.x) {
        scalar_t val = col_base[idx];
        if (idx == 0 && m_vec4 == 0) {
            alpha = val;
        } else if (idx > 0) {
            local_sum += val * val;
        }
    }

    // Reduction within block for local_sum
    block_reduce_sum(local_sum, sdata, tid, blockDim.x);
    __syncthreads();
    scalar_t sigma2 = sdata[0];

    // Broadcast alpha to all threads (alpha only known by threads that saw idx==0)
    // Use shared memory position 1 for alpha broadcast if needed.
    if (tid == 0) sdata[1] = alpha;
    __syncthreads();
    alpha = sdata[1];

    scalar_t beta, tau, scale;
    if (sigma2 < literal_constants<scalar_t>::epsilon) {
        tau = literal_constants<scalar_t>::zero;
        beta = alpha;
        scale = literal_constants<scalar_t>::zero;
    } else {
        scalar_t r = sqrt_func(alpha * alpha + sigma2);
        beta = alpha >= literal_constants<scalar_t>::zero ? -r : r;
        tau = (beta - alpha) / beta;
        scale = literal_constants<scalar_t>::one / (alpha - beta);
    }

    // Store tau & beta
    if (tid == 0) {
        dE[i] = beta;
        dTau[i] = tau;
    }

    #pragma unroll 4
    for (int idx4 = tid; idx4 < m_vec4; idx4 += blockDim.x) {
        if (idx4 == 0) {
            // Handle first element specially (set to 1.0) and scale the rest
            vec4_t val4 = reinterpret_cast<const vec4_t*>(&col_base[idx4 * 4])[0];
            val4.x = literal_constants<scalar_t>::one;
            val4.y *= scale;
            val4.z *= scale;
            val4.w *= scale;
            reinterpret_cast<vec4_t*>(&col_base[idx4 * 4])[0] = val4;
        } else {
            vec4_t val4 = reinterpret_cast<const vec4_t*>(&col_base[idx4 * 4])[0];
            val4.x *= scale;
            val4.y *= scale;
            val4.z *= scale;
            val4.w *= scale;
            reinterpret_cast<vec4_t*>(&col_base[idx4 * 4])[0] = val4;
        }
    }
    
    for (int idx = 4 * m_vec4 + tid; idx < m; idx += blockDim.x) {
        if (idx == 0 && m_vec4 == 0) {
            col_base[0] = literal_constants<scalar_t>::one;
        } else if (idx > 0) {
            col_base[idx] *= scale;
        }
    }
}

// Kernel to compute dot product, scale, and axpy in a fused manner
template<typename scalar_t>
__global__ void fused_dot_scale_axpy(int n,
                                     const scalar_t* __restrict__ v,
                                     scalar_t* __restrict__ w,
                                     const scalar_t* __restrict__ tau_ptr) {
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    int tid = threadIdx.x;
    scalar_t tau = *tau_ptr;

    // Compute dot product within single block
    scalar_t local_dot = literal_constants<scalar_t>::zero;
    
    int n_vec4 = n / 4;
    using vec4_t = typename vec4_type<scalar_t>::type;
    
    #pragma unroll 4
    for (int idx4 = tid; idx4 < n_vec4; idx4 += blockDim.x) {
        vec4_t w_val4 = reinterpret_cast<const vec4_t*>(&w[idx4 * 4])[0];
        vec4_t v_val4 = reinterpret_cast<const vec4_t*>(&v[idx4 * 4])[0];
        
        local_dot += (w_val4.x * v_val4.x + w_val4.y * v_val4.y + 
                     w_val4.z * v_val4.z + w_val4.w * v_val4.w);
    }
    
    // Handle remaining elements for dot product
    for (int idx = 4 * n_vec4 + tid; idx < n; idx += blockDim.x) {
        scalar_t w_val = w[idx];
        scalar_t v_val = v[idx];
        local_dot += w_val * v_val;
    }

    // Reduction within block for dot product
    block_reduce_sum(local_dot, sdata, tid, blockDim.x);
    __syncthreads();
    scalar_t dot = sdata[0];

    // Compute correction factor
    scalar_t alpha_corr = -literal_constants<scalar_t>::half * tau * dot;

    // Apply scale and axpy operation
    #pragma unroll 4
    for (int idx4 = tid; idx4 < n_vec4; idx4 += blockDim.x) {
        vec4_t w_val4 = reinterpret_cast<const vec4_t*>(&w[idx4 * 4])[0];
        vec4_t v_val4 = reinterpret_cast<const vec4_t*>(&v[idx4 * 4])[0];
        
        w_val4.x = tau * (w_val4.x + alpha_corr * v_val4.x);
        w_val4.y = tau * (w_val4.y + alpha_corr * v_val4.y);
        w_val4.z = tau * (w_val4.z + alpha_corr * v_val4.z);
        w_val4.w = tau * (w_val4.w + alpha_corr * v_val4.w);
        
        reinterpret_cast<vec4_t*>(&w[idx4 * 4])[0] = w_val4;
    }
    
    for (int idx = 4 * n_vec4 + tid; idx < n; idx += blockDim.x) {
        scalar_t w_val = w[idx];
        scalar_t v_val = v[idx];
        w[idx] = tau * (w_val + alpha_corr * v_val);
    }
}

// Kernel to set up tridiagonal matrix elements D and E from A
template<typename scalar_t>
__global__ void setup_tridiagonal(int n, scalar_t* A, int lda, scalar_t* D, scalar_t* E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        D[i] = A[idx2D(i, i, lda)];

        if (i < n - 2) {
            A[idx2D(i + 1, i, lda)] = E[i];
        } else if (i == n - 2) {
            E[i] =  A[idx2D(i + 1, i, lda)];
        }
    }
}

// Fused kernel for tridiagolization of matrices of size <= warpSize
template<typename scalar_t>
__global__ void small_sytd2_kernel(int m, int j, scalar_t* __restrict__ A, int lda,
                                   scalar_t* __restrict__ dE, scalar_t* __restrict__ dTau) {
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    int tid = threadIdx.x;
    
    // Shared memory for Householder vector v and intermediate vector w
    scalar_t* sv = sdata;
    scalar_t* sw = sdata + m;
    
    scalar_t* col_base = A + idx2D(j + 1, j, lda);
    scalar_t* A22 = A + idx2D(j + 1, j + 1, lda);
    
    // ==================== STEP 1: Compute Householder reflector ====================
    scalar_t sigma2 = literal_constants<scalar_t>::zero;
    scalar_t alpha = literal_constants<scalar_t>::zero;
    
    // Load column vector and compute sum of squares
    for (int idx = tid; idx < m; idx += blockDim.x) {
        scalar_t val = col_base[idx];
        if (idx == 0) alpha = val; else sigma2 += val * val;
        if (idx < m) sv[idx] = val;
    }
    
    // Reduction for sum of squares
    sigma2 = warp_reduce_sum(sigma2);
    sigma2 = __shfl(sigma2, 0);
    alpha = __shfl(alpha, 0);
    
    // Compute Householder parameters
    scalar_t beta, tau, scale;
    if (sigma2 < literal_constants<scalar_t>::epsilon) {
        tau = literal_constants<scalar_t>::zero; 
        beta = alpha; 
        scale = literal_constants<scalar_t>::zero;
    } else {
        scalar_t r = sqrt_func(alpha * alpha + sigma2);
        beta = alpha >= literal_constants<scalar_t>::zero ? -r : r;
        tau = (beta - alpha) / beta;
        scale = literal_constants<scalar_t>::one / (alpha - beta);
    }
    
    // Store tau & beta
    if (tid == 0) {
        dE[j] = beta;
        dTau[j] = tau;
    }
    
    // Update Householder vector in shared memory
    for (int idx = tid; idx < m; idx += blockDim.x) {
        if (idx == 0) sv[0] = literal_constants<scalar_t>::one;
        else sv[idx] *= scale;
    }
    
    // ==================== STEP 2: Matrix-vector multiplication w = A22 * v ====================
    // Each thread computes one element of w
    if (tid < m) {
        scalar_t sum = literal_constants<scalar_t>::zero;
        for (int col = 0; col < m; col++) {
            scalar_t a_val = A22[idx2D(tid, col, lda)];
            sum += a_val * sv[col];
        }
        sw[tid] = sum;
    }
    
    // ==================== STEP 3: Dot product and scaling w = tau * (w - 0.5 * tau * dot(w,v) * v) ====================
    // Compute dot product w^T * v
    scalar_t dot = literal_constants<scalar_t>::zero;
    for (int idx = tid; idx < m; idx += blockDim.x) {
        dot += sw[idx] * sv[idx];
    }
    dot = warp_reduce_sum(dot);
    dot = __shfl(dot, 0);
    
    // Apply scaling: w = tau * (w - 0.5 * tau * dot * v)
    scalar_t alpha_corr = -literal_constants<scalar_t>::half * tau * dot;
    for (int idx = tid; idx < m; idx += blockDim.x) {
        sw[idx] = tau * (sw[idx] + alpha_corr * sv[idx]);
    }
    
    // ==================== STEP 4: Matrix update A22 -= v * w^T + w * v^T ====================
    // Each thread processes one row of the matrix
    if (tid < m) {
        scalar_t v_row = sv[tid];
        scalar_t w_row = sw[tid];
        
        for (int col = 0; col < m; col++) {
            scalar_t a_val = A22[idx2D(tid, col, lda)];
            scalar_t v_col = sv[col];
            scalar_t w_col = sw[col];
            
            // A[row][col] -= v[row] * w[col] + w[row] * v[col]
            A22[idx2D(tid, col, lda)] = a_val - (v_row * w_col + w_row * v_col);
        }
    }
    
    // ==================== STEP 5: Update original column vector ====================
    // Write back the Householder vector to the lower trianglar part of A
    for (int idx = tid; idx < m; idx += blockDim.x) {
        col_base[idx] = sv[idx];
    }
}

// Unblocked symmetric tridiagonal reduction
template<typename scalar_t>
hipError_t hip_sytd2(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    scalar_t* dA,
    int lda,
    scalar_t* dD,
    scalar_t* dE,
    scalar_t* dTau,
    int device_warp_size,
    scalar_t* w_vec,
    const scalar_t* d_scalars)
{
    const int threads = 256;
    size_t shmem_bytes = CEIL_DIV(threads, device_warp_size) * sizeof(scalar_t);

    for (int j = 0; j < n - 2; ++j) {
        int m = n - j - 1;

        if (m <= device_warp_size) {
            // Use fused kernel for small matrices
            size_t sytd2_shmem = 2 * m * sizeof(scalar_t);  // space for v, w vetors
            hipLaunchKernelGGL(small_sytd2_kernel<scalar_t>, dim3(1), dim3(device_warp_size), sytd2_shmem, 0,
                               m, j, dA, lda, dE, dTau);
        } else {
            // Use separate kernels for larger matrices
            hipLaunchKernelGGL(larfg_kernel<scalar_t>, dim3(1), dim3(threads), shmem_bytes, 0,
                               m, j, dA, lda, dE, dTau);

            scalar_t* A22 = dA + idx2D(j + 1, j + 1, lda);
            scalar_t* v_vec = dA + idx2D(j + 1, j, lda);
            
            // w = A[j+1:, j+1:] @ v
            if constexpr (std::is_same_v<scalar_t, float>) {
                rocblas_sgemv(handle, rocblas_operation_none, m, m, d_scalars + 0, A22, lda, v_vec, 1, d_scalars + 2, w_vec, 1);
            } else {
                rocblas_dgemv(handle, rocblas_operation_none, m, m, d_scalars + 0, A22, lda, v_vec, 1, d_scalars + 2, w_vec, 1);                
            }

            // w = tau * (w - 0.5 * tau * dot(w, v) * v)
            hipLaunchKernelGGL(fused_dot_scale_axpy<scalar_t>, dim3(1), dim3(threads), shmem_bytes, 0,
                               m, v_vec, w_vec, dTau + j);
            
            // A[j+1:, j+1:] -= v * w^T + w * v^T
            if constexpr (std::is_same_v<scalar_t, float>) {
                rocblas_sger(handle, m, m, d_scalars + 1, v_vec, 1, w_vec, 1, A22, lda);
                rocblas_sger(handle, m, m, d_scalars + 1, w_vec, 1, v_vec, 1, A22, lda);
            } else {
                rocblas_dger(handle, m, m, d_scalars + 1, v_vec, 1, w_vec, 1, A22, lda);
                rocblas_dger(handle, m, m, d_scalars + 1, w_vec, 1, v_vec, 1, A22, lda);
            }
        }
    }

    return hipSuccess;
}

// Compute the update to column j of A
template<typename scalar_t>
__global__ void accumulate_a_col_updates(int len, int j,
                                         scalar_t* __restrict__ dA, int lda,
                                         const scalar_t* __restrict__ dW, int ldw) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ char shmem_raw[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(shmem_raw);
    scalar_t* sA = shmem;
    scalar_t* sW = shmem + j;

    for (int k = threadIdx.x; k < j; k += blockDim.x) {
        sA[k] = dA[idx2D(j, k, lda)];
        sW[k] = dW[idx2D(j, k, ldw)];
    }
    __syncthreads();

    if (tid >= len) return;

    int row = j + tid;
    scalar_t accum = literal_constants<scalar_t>::zero;
    for (int k = 0; k < j; ++k) {
        scalar_t a_tail = dA[idx2D(row, k, lda)];
        scalar_t w_tail = dW[idx2D(row, k, ldw)];
        accum += sA[k] * w_tail + sW[k] * a_tail;
    }
    dA[idx2D(row, j, lda)] -= accum;
}

template<typename scalar_t>
__global__ void compute_w_col_kernel(int m, int j,
                                     const scalar_t* __restrict__ A22, int lda,
                                     const scalar_t* __restrict__ A,
                                     const scalar_t* __restrict__ W, int ldw,
                                     const scalar_t* __restrict__ v,
                                     scalar_t* __restrict__ w,
                                     scalar_t* __restrict__ tmp_vec) {
    int col = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);

    scalar_t w_sum = literal_constants<scalar_t>::zero;
    scalar_t a_sum = literal_constants<scalar_t>::zero;
    scalar_t symv_sum = literal_constants<scalar_t>::zero;

    int m_vec4 = m / 4;
    using vec4_t = typename vec4_type<scalar_t>::type;
        
    #pragma unroll 4
    for (int row4 = tid; row4 < m_vec4; row4 += blockDim.x) {
        vec4_t v_val4 = reinterpret_cast<const vec4_t*>(&v[row4 * 4])[0];
        vec4_t syma_val4 = reinterpret_cast<const vec4_t*>(&A22[idx2D(row4 * 4, col, lda)])[0];
        
        symv_sum += (syma_val4.x * v_val4.x + syma_val4.y * v_val4.y + 
                    syma_val4.z * v_val4.z + syma_val4.w * v_val4.w);

        if (col < j) {
            vec4_t w_val4 = reinterpret_cast<const vec4_t*>(&W[idx2D(row4 * 4, col, ldw)])[0];
            vec4_t a_val4 = reinterpret_cast<const vec4_t*>(&A[idx2D(row4 * 4, col, lda)])[0];

            w_sum += (w_val4.x * v_val4.x + w_val4.y * v_val4.y + 
                     w_val4.z * v_val4.z + w_val4.w * v_val4.w);
            a_sum += (a_val4.x * v_val4.x + a_val4.y * v_val4.y + 
                     a_val4.z * v_val4.z + a_val4.w * v_val4.w);
        }
    }

    // Handle remaining elements
    for (int row = 4 * m_vec4 + tid; row < m; row += blockDim.x) {
        scalar_t v_val = v[row];
        
        scalar_t syma_val = A22[idx2D(row, col, lda)];
        symv_sum += syma_val * v_val;

        if (col < j) {
            scalar_t w_val = W[idx2D(row, col, ldw)];
            scalar_t a_val = A[idx2D(row, col, lda)];
            w_sum += w_val * v_val;
            a_sum += a_val * v_val;
        }
    }

    if (col < j) {
        // Reduction within block for w_sum
        block_reduce_sum(w_sum, sdata, tid, blockDim.x);
        if (tid == 0) {
            tmp_vec[col] = sdata[0];
        }

        // Reduction within block for a_sum  
        block_reduce_sum(a_sum, sdata, tid, blockDim.x);
        if (tid == 0) {
            tmp_vec[col + j] = sdata[0];
        }
    }

    block_reduce_sum(symv_sum, sdata, tid, blockDim.x);
    if (tid == 0) {
        w[col] = sdata[0];
    }
}

template<typename scalar_t>
__global__ void update_w_col_kernel(int m, int j,
                                    const scalar_t* __restrict__ A, int lda,
                                    const scalar_t* __restrict__ W, int ldw,
                                    const scalar_t* __restrict__ tmp_vec,
                                    scalar_t* __restrict__ w) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    scalar_t* stmp1 = sdata;      // tmp1 = W^T*v
    scalar_t* stmp2 = sdata + j;  // tmp2 = A^T*v
    
    for (int i = threadIdx.x; i < j; i += blockDim.x) {
        stmp1[i] = tmp_vec[i];
        stmp2[i] = tmp_vec[j + i];
    }
    __syncthreads();

    if (row >= m) return;
    
    scalar_t sum = literal_constants<scalar_t>::zero;
    
    for (int col = 0; col < j; ++col) {
        scalar_t a_val = A[idx2D(row, col, lda)];
        scalar_t w_val = W[idx2D(row, col, ldw)];
        sum += a_val * stmp1[col] + w_val * stmp2[col];
    }
    
    w[row] -= sum;
}

// Blocked panel reduction for symmetric tridiagonalization
template<typename scalar_t>
hipError_t hip_latrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    scalar_t* dA,
    int lda,
    scalar_t* dW,
    int ldw,
    scalar_t* dD,
    scalar_t* dE,
    scalar_t* dTau,
    int panel_size,
    int device_warp_size,
    scalar_t* d_tmp_vec,
    const scalar_t* d_scalars)
{
    const int threads = 256;
    size_t shmem_bytes = CEIL_DIV(threads, device_warp_size) * sizeof(scalar_t);

    for (int j = 0; j < panel_size; ++j) {
        int m = n - j - 1;
        
        // A[j:, j] -= A[j:, :j] * W[j, :j]^T + W[j:, :j] * A[j, :j]^T
        if (j > 0) {
            int len = n - j;
            int blocks_updates = CEIL_DIV(len, threads);
            size_t shmem_updates = 2 * j * sizeof(scalar_t);
            hipLaunchKernelGGL(accumulate_a_col_updates<scalar_t>, dim3(blocks_updates), dim3(threads), shmem_updates, 0,
                               len, j, dA, lda, dW, ldw);
        }

        // Single-block optimized path
        hipLaunchKernelGGL(larfg_kernel<scalar_t>, dim3(1), dim3(threads), shmem_bytes, 0,
                           m, j, dA, lda, dE, dTau);

        scalar_t* v_vec = dA + idx2D(j + 1, j, lda);
        scalar_t* w_vec = dW + idx2D(j + 1, j, ldw);

        // w = A[j+1:, j+1:] @ v
        //   - A[j+1:, :j] @ (W[j+1:, :j].T @ v)
        //   - W[j+1:, :j] @ (A[j+1:, :j].T @ v)
        if (j > 0) {            
            hipLaunchKernelGGL(compute_w_col_kernel<scalar_t>, dim3(m), dim3(threads), shmem_bytes, 0,
                               m, j,
                               dA + idx2D(j + 1, j + 1, lda), lda,
                               dA + j + 1,
                               dW + j + 1, ldw,
                               v_vec, w_vec, d_tmp_vec);
            
            hipLaunchKernelGGL(update_w_col_kernel<scalar_t>, dim3(CEIL_DIV(m, threads)), dim3(threads), 2*j*sizeof(scalar_t), 0,
                               m, j, dA + j + 1, lda, dW + j + 1, ldw, d_tmp_vec, w_vec);
        } else {
            if constexpr (std::is_same_v<scalar_t, float>) {
                rocblas_sgemv(handle, rocblas_operation_none, m, m, d_scalars + 0,
                              dA + idx2D(j + 1, j + 1, lda), lda, 
                              v_vec, 1, d_scalars + 2, w_vec, 1);
            } else {
                rocblas_dgemv(handle, rocblas_operation_none, m, m, d_scalars + 0,
                              dA + idx2D(j + 1, j + 1, lda), lda, 
                              v_vec, 1, d_scalars + 2, w_vec, 1);
            }
        }

        // w = tau * (w - 0.5 * tau * dot(w, v) * v)
        hipLaunchKernelGGL(fused_dot_scale_axpy<scalar_t>, dim3(1), dim3(threads), shmem_bytes, 0,
                           m, v_vec, w_vec, dTau + j);
    }
    return hipSuccess;
}

// Main symmetric tridiagonal reduction routine (blocked + unblocked)
template<typename scalar_t>
hipError_t hip_sytrd_template(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    scalar_t* dA,
    int lda,
    scalar_t* dD,
    scalar_t* dE,
    scalar_t* dTau)
{
    if (n < 3) return hipSuccess;

    const int threads = 256;
    const int panel_size = 64;

    // Allocate device memory for scalar constants
    scalar_t h_scalars[3];
    h_scalars[0] = literal_constants<scalar_t>::one;   // 1.0
    h_scalars[1] = -literal_constants<scalar_t>::one;  // -1.0
    h_scalars[2] = literal_constants<scalar_t>::zero;  // 0.0
    
    scalar_t *d_scalars;
    hipMalloc(&d_scalars, 3 * sizeof(scalar_t));
    hipMemcpy(d_scalars, h_scalars, 3 * sizeof(scalar_t), hipMemcpyHostToDevice);
    
    int device_warp_size;
    hipDeviceGetAttribute(&device_warp_size, hipDeviceAttributeWarpSize, 0);
    scalar_t* dW;
    hipMalloc(&dW, n * panel_size * sizeof(scalar_t));
    scalar_t* d_tmp_vec;
    hipMalloc(&d_tmp_vec, 2 * panel_size * sizeof(scalar_t));

    int j = 0;
    while(j < n - panel_size) {
        scalar_t* dA_panel = dA + idx2D(j, j, lda);
        int ldw = n - j;

        hip_latrd<scalar_t>(handle,
                  uplo,
                  n - j,
                  dA + idx2D(j, j, lda),
                  lda,
                  dW,
                  ldw,
                  dD + j,
                  dE + j,
                  dTau + j,
                  panel_size,
                  device_warp_size,
                  d_tmp_vec,
                  d_scalars);

        j += panel_size;

        // A[j:, j:] -= W[j:, :j] * A[j, :j]^T + A[j:, :j] * W[j, :j]^T
        // Use appropriate GEMM function based on scalar type
        if constexpr (std::is_same_v<scalar_t, float>) {
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                          n - j, n - j, panel_size,
                          d_scalars + 1, // -1.0f
                          dA_panel + panel_size, lda,
                          dW + panel_size, ldw,
                          d_scalars + 0, // 1.0f
                          dA + idx2D(j, j, lda), lda);
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                          n - j, n - j, panel_size,
                          d_scalars + 1, // -1.0f
                          dW + panel_size, ldw,
                          dA_panel + panel_size, lda,
                          d_scalars + 0, // 1.0f
                          dA + idx2D(j, j, lda), lda);
        } else {
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                          n - j, n - j, panel_size,
                          d_scalars + 1, // -1.0
                          dA_panel + panel_size, lda,
                          dW + panel_size, ldw,
                          d_scalars + 0, // 1.0
                          dA + idx2D(j, j, lda), lda);
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                          n - j, n - j, panel_size,
                          d_scalars + 1, // -1.0
                          dW + panel_size, ldw,
                          dA_panel + panel_size, lda,
                          d_scalars + 0, // 1.0
                          dA + idx2D(j, j, lda), lda);
        }
    }
    if (j < n - 2) {
        hip_sytd2<scalar_t>(handle,
                  uplo,
                  n - j,
                  dA + idx2D(j, j, lda),
                  lda,
                  dD + j,
                  dE + j,
                  dTau + j,
                  device_warp_size,
                  dW,
                  d_scalars);
    }

    int tridiag_blocks = CEIL_DIV(n, threads);
    hipLaunchKernelGGL(setup_tridiagonal<scalar_t>, dim3(tridiag_blocks), dim3(threads), 0, 0,
                       n, dA, lda, dD, dE);
    hipMemset(dTau + n - 2, 0, sizeof(scalar_t));
    hipDeviceSynchronize();
    hipFree(dW);
    hipFree(d_tmp_vec);
    hipFree(d_scalars);

    return hipSuccess;
}

} // namespace aiter

// Wrapper functions for specific precisions
extern "C" hipError_t hip_ssytrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    float* dA,
    int lda,
    float* dD,
    float* dE,
    float* dTau)
{
    return aiter::hip_sytrd_template<float>(handle, uplo, n, dA, lda, dD, dE, dTau);
}

extern "C" hipError_t hip_dsytrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    double* dA,
    int lda,
    double* dD,
    double* dE,
    double* dTau)
{
    return aiter::hip_sytrd_template<double>(handle, uplo, n, dA, lda, dD, dE, dTau);
}
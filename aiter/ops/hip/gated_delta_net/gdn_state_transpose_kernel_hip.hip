// In-place 128x128 float32 state-tensor transpose kernels for the
// HIP GDN decode path.
//   - state_transpose_128x128_kernel:        legacy single-layer
//   - state_transpose_128x128_multi_layer_kernel + slot_layout_writeback_kernel:
//                                            multi-layer batched, per-slot gated
#include <hip/hip_runtime.h>
#include <cstdint>

// Single-layer transpose: one block per (matrix, row).
constexpr int TRANS_THREADS = 128;

__global__
void state_transpose_128x128_kernel(
    float* __restrict__ state,
    const int* __restrict__ indices,
    int batch_size,
    int num_v_heads
) {
    constexpr int N = 128;
    const int global_idx = blockIdx.x;
    const int mat_idx = global_idx / N;
    const int row = global_idx % N;
    const int batch_idx = mat_idx / num_v_heads;
    const int hv_idx = mat_idx % num_v_heads;
    if (batch_idx >= batch_size) return;

    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    float* mat = state + (int64_t)pool_idx * num_v_heads * N * N + (int64_t)hv_idx * N * N;

    for (int col = row + 1 + threadIdx.x; col < N; col += TRANS_THREADS) {
        int idx1 = row * N + col;
        int idx2 = col * N + row;
        float tmp = mat[idx1];
        mat[idx1] = mat[idx2];
        mat[idx2] = tmp;
    }
}

extern "C" {

void launch_state_transpose(
    void* state, const void* indices,
    int batch_size, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads * 128);
    dim3 block(TRANS_THREADS);
    hipLaunchKernelGGL(state_transpose_128x128_kernel,
        grid, block, 0, stream,
        reinterpret_cast<float*>(state),
        reinterpret_cast<const int*>(indices),
        batch_size, num_v_heads);
}

} // extern "C"

// Multi-layer transpose: one block per 128x128 matrix, batched across
// all GDN layers in one launch. Per-slot ``slot_layout[pool_idx]``
// gates so slots already in ``target_layout`` early-exit.
//
// state_base layout: [num_layers, slots+1, num_v_heads, N, N]
// slot_layout:       int8 array, 0 == KV (FLA prefill), 1 == VK (HIP decode)
constexpr int TRANS_V2_THREADS = 128;
__global__ __launch_bounds__(TRANS_V2_THREADS, 8)
void state_transpose_128x128_multi_layer_kernel(
    float* __restrict__ state_base,
    const int* __restrict__ indices,
    const int8_t* __restrict__ slot_layout,
    int target_layout,
    int num_layers,
    int batch_size,
    int num_v_heads,
    int64_t layer_stride_floats
) {
    constexpr int N = 128;
    const int blocks_per_layer = batch_size * num_v_heads;
    const int gidx = blockIdx.x;
    const int layer_idx = gidx / blocks_per_layer;
    const int within = gidx - layer_idx * blocks_per_layer;
    const int batch_idx = within / num_v_heads;
    const int hv_idx = within - batch_idx * num_v_heads;
    if (layer_idx >= num_layers) return;
    if (batch_idx >= batch_size) return;

    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;
    if ((int)slot_layout[pool_idx] == target_layout) return;

    float* __restrict__ mat = state_base
        + (int64_t)layer_idx * layer_stride_floats
        + (int64_t)pool_idx * num_v_heads * N * N
        + (int64_t)hv_idx * N * N;

    const int r = threadIdx.x;
    if (r >= N) return;
    const int row_off = r * N;

    int c = r + 1;

    // Scalar prefix until c is 4-aligned for the float4 row pointer.
    while (c < N && (c & 3) != 0) {
        float a = mat[row_off + c];
        float b = mat[c * N + r];
        mat[row_off + c] = b;
        mat[c * N + r] = a;
        ++c;
    }

    // Vectorised middle: float4 row load/store + 4 scalar column ops.
    while (c + 4 <= N) {
        float4 row4 = *reinterpret_cast<float4*>(&mat[row_off + c]);
        float c0 = mat[(c + 0) * N + r];
        float c1 = mat[(c + 1) * N + r];
        float c2 = mat[(c + 2) * N + r];
        float c3 = mat[(c + 3) * N + r];
        float4 col4 = make_float4(c0, c1, c2, c3);
        *reinterpret_cast<float4*>(&mat[row_off + c]) = col4;
        mat[(c + 0) * N + r] = row4.x;
        mat[(c + 1) * N + r] = row4.y;
        mat[(c + 2) * N + r] = row4.z;
        mat[(c + 3) * N + r] = row4.w;
        c += 4;
    }

    // Scalar tail.
    while (c < N) {
        float a = mat[row_off + c];
        float b = mat[c * N + r];
        mat[row_off + c] = b;
        mat[c * N + r] = a;
        ++c;
    }
}

// Post-transpose writeback: slot_layout[indices[i]] = target_layout.
__global__ void slot_layout_writeback_kernel(
    int8_t* __restrict__ slot_layout,
    const int* __restrict__ indices,
    int target_layout,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int pool_idx = indices[i];
    if (pool_idx < 0) return;
    slot_layout[pool_idx] = (int8_t)target_layout;
}

extern "C" {

void launch_state_transpose_multi_layer(
    void* state_base,
    const void* indices,
    const void* slot_layout,
    int target_layout,
    int num_layers,
    int batch_size,
    int num_v_heads,
    int64_t layer_stride_floats,
    hipStream_t stream
) {
    const unsigned grid_x =
        (unsigned)num_layers * (unsigned)batch_size * (unsigned)num_v_heads;
    if (grid_x == 0) return;
    dim3 grid(grid_x);
    dim3 block(TRANS_V2_THREADS);
    hipLaunchKernelGGL(state_transpose_128x128_multi_layer_kernel,
        grid, block, 0, stream,
        reinterpret_cast<float*>(state_base),
        reinterpret_cast<const int*>(indices),
        reinterpret_cast<const int8_t*>(slot_layout),
        target_layout, num_layers, batch_size, num_v_heads,
        layer_stride_floats);

    constexpr int WB_THREADS = 256;
    const unsigned wb_blocks = (batch_size + WB_THREADS - 1) / WB_THREADS;
    if (wb_blocks > 0) {
        hipLaunchKernelGGL(slot_layout_writeback_kernel,
            dim3(wb_blocks), dim3(WB_THREADS), 0, stream,
            const_cast<int8_t*>(reinterpret_cast<const int8_t*>(slot_layout)),
            reinterpret_cast<const int*>(indices),
            target_layout, batch_size);
    }
}

} // extern "C"

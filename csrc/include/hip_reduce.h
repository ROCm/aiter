#include "hip_compat.h"

template <typename T, typename F>
__device__ constexpr T wave_reduce(T local, F reduce_f)
{
  constexpr int reduce_stage = 6; // 1<<6=64
  T v_local = local;
#pragma unroll
  for (int i_stage = 0; i_stage < reduce_stage; i_stage++)
  {
    int src_lane = __lane_id() ^ (1 << i_stage);
    int32_t v_remote_tmp =
        __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, v_local));
    T v_remote = __builtin_bit_cast(T, v_remote_tmp);
    v_local = reduce_f(v_local, v_remote);
  }
  return v_local;
}


template <typename T, typename F>
__device__ constexpr T cross_wave_reduce(T local, F reduce_f, T* smem)
{
    int blockSize = blockDim.x;
    int waves     = blockDim.x / WARP_SIZE;
    int wave_size = WARP_SIZE;
    int lane_id   = threadIdx.x % wave_size;
    
    __syncthreads();
    smem[threadIdx.x] = local;
    __syncthreads();

    // the data within single wave is the same
    // but for simplicity, we still use data from each lane.
    T v_local = smem[lane_id];
#pragma unroll
    for(int i_stage = 1; i_stage < waves; i_stage++)
    {
        T v_remote = smem[i_stage * wave_size + lane_id];
        v_local    = reduce_f(v_local, v_remote);
    } 
    return v_local;
}
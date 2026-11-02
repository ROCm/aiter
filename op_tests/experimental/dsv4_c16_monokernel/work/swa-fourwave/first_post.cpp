#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "opus/opus.hpp"
#include "aiter_opus_plus.h"
extern __shared__ unsigned char __dynamic_shared_0[];
namespace aiter { constexpr int warp_size=32;
    template <typename T, typename F>
    __device__ constexpr T reduce_in_4threads(T data, F reduce_op)
    {
        T res = multithread_reduce(data, reduce_op, 4);
        return res;
    }

    template <typename T, typename F>
    __device__ constexpr T reduce_cross_4threads(T data, F reduce_op)
    {
        // DDP_ROW_RR 4 | 8
        data = reduce_op(opus::mov_dpp(data, opus::number<0x124>{}), data);
        data = reduce_op(opus::mov_dpp(data, opus::number<0x128>{}), data);
        return data;
        // float res = data;
        // asm volatile("s_nop 1");
        // asm volatile("v_add_f32 %0, %1, %1 " "row_ror:4 row_mask:0xf bank_mask:0xf bound_ctrl:1" : "=&v"(res) : "v"(res));
        // asm volatile("s_nop 1");
        // asm volatile("v_add_f32 %0, %1, %1 " "row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1" : "=&v"(res) : "v"(res));
        // asm volatile("s_nop 0");
        // return res;
    }


extern "C" __device__ __attribute__((always_inline,used)) void sw4_first_post(uint64_t pa,uint64_t ca,uint64_t ga,uint64_t qa,uint64_t sa,uint64_t ba,unsigned row){
 float*post_mix=(float*)pa;float*comb_mix=(float*)ca;float*gemm=(float*)ga;float*sq=(float*)qa;float*hc_scale=(float*)sa;float*hc_base=(float*)ba;
 constexpr int num_rows=1,hc_mult=4,hc_mult2=16,hc_mult3=24;
 constexpr float hc_sinkhorn_eps=1e-6,hc_post_mult_value=2.;
 int sinkhorn_repeat=20;int m_idx=row,m_oob=1;
 int tid=threadIdx.x,lane_id=tid%32,warp=tid/32;
 float*ss=(float*)__dynamic_shared_0;float*s_hc_mult3=ss+16;
 auto sum_f=[](float a,float b){return a+b;};
 auto sigmoid=[](float x){return 1.0f/(1.0f+__expf(-x));};
 if(tid<16)ss[tid]=0.;__syncthreads();
 float v=sq[tid*16+row];v=wave_reduce<float,decltype(sum_f),32,false>(v,sum_f);
 if(lane_id==31)ss[warp]=v;__syncthreads();
 float r=lane_id<9?ss[lane_id]:0.;r=multithread_reduce(r,sum_f,16);
 r=rsqrtf(r/(4*4096)+1e-6f);r=__builtin_bit_cast(float,__builtin_amdgcn_readlane(__builtin_bit_cast(int,r),0));
 if(tid<24){
  float acc[4]={0.,0.,0.,0.};
  #pragma unroll
  for(int lane=0;lane<48;lane++){
   float v=0.;
   #pragma unroll
   for(int q=0;q<4;q++){int k=lane+q*48;v+=k<128?gemm[(k*16+row)*32+tid]:0.;}
   acc[lane%4]+=v;
  }
  s_hc_mult3[tid]=((acc[0]+acc[1])+(acc[2]+acc[3]))*r;
 }
 __syncthreads();
        if (threadIdx.x < 32){
            // _pre_split_mixes_fwd (post & comb)
            float post_mix_v = 0.0f;
            if (lane_id < num_rows * hc_mult) {
                post_mix_v = s_hc_mult3[lane_id / hc_mult * hc_mult3 + lane_id % hc_mult + hc_mult];
                post_mix_v = sigmoid(post_mix_v * hc_scale[1] + hc_base[lane_id % hc_mult + hc_mult]) * hc_post_mult_value;
                if (lane_id / hc_mult < m_oob) {
                    post_mix[(m_idx + lane_id / hc_mult) * hc_mult + lane_id % hc_mult] = post_mix_v;
                }
            }

            static_assert(num_rows * hc_mult2 <= warp_size, "num_rows * num_rows * hc_mult * hc_mult < warp_size");
            float comb_mix_v = 0.0f;
            if (lane_id < num_rows * hc_mult2) {
                comb_mix_v = s_hc_mult3[lane_id / hc_mult2 * hc_mult3 + lane_id % hc_mult2 + 2 * hc_mult];
                comb_mix_v =comb_mix_v * hc_scale[2] + hc_base[lane_id % hc_mult2 + 2 * hc_mult];
            }

            // comb = comb.softmax(-1) + eps
            float row_max = reduce_in_4threads(comb_mix_v, fmaxf);
            comb_mix_v = expf(comb_mix_v - row_max);
            float row_sum = reduce_in_4threads(comb_mix_v, sum_f);
            comb_mix_v = comb_mix_v / row_sum + hc_sinkhorn_eps;
            // comb = comb / (comb.sum(-2) + eps)
            float col_sum = reduce_cross_4threads(comb_mix_v, sum_f);
            comb_mix_v = comb_mix_v / (col_sum + hc_sinkhorn_eps);

            #pragma clang loop unroll(disable)
            for(int i = 0; i < sinkhorn_repeat - 1; i++) {
                row_sum = reduce_in_4threads(comb_mix_v, sum_f);
                comb_mix_v = comb_mix_v * __builtin_amdgcn_rcpf(row_sum + hc_sinkhorn_eps);
                col_sum = reduce_cross_4threads(comb_mix_v, sum_f);
                comb_mix_v = comb_mix_v * __builtin_amdgcn_rcpf(col_sum + hc_sinkhorn_eps);
            }

            if (lane_id / hc_mult2 < m_oob) {
                comb_mix[(m_idx + lane_id / hc_mult2) * hc_mult2 + lane_id % hc_mult2] = comb_mix_v;
            }
        }
}
}

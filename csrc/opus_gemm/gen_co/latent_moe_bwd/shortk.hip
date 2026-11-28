// SPDX-License-Identifier: MIT
#include "short_pipeline.cuh"
using Base=opus_gemm_a16w16_grouped_traits_gfx950<256,256,true,true,true>;
template<int K> struct Trait:Base {static constexpr int OPT_K=K;};
using A=opus_gemm_a16w16_k_grouped_kargs_gfx950;
using F4=float __attribute__((ext_vector_type(4)));
template<int N,int K> __device__ __forceinline__ void run(A a){
#ifdef __HIP_DEVICE_COMPILE__
    using namespace opus_a16w16_grouped;
    constexpr int nt=N/256,kt=K/256,tiles=nt*kt;
    int block=opus::block_id_x(),e=block/tiles;
    const int *record=a.table+4*e;
    int begin=record[1],count=record[2];
    int mn,kn;
    mixed_band_order(block%tiles,nt,kt,mn,kn);
    int n0=mn*256,k0=kn*256;
    a.m=N; a.n=K; a.stride_a=N; a.stride_b=K; a.stride_c=K; a.stride_c_group=(long long)N*K;
    if(count==0){
        float* base=reinterpret_cast<float*>(a.ptr_c)+((long long)e*N+n0)*K+k0;
        const F4 zero={};
        #pragma unroll 8
        for(int i=opus::thread_id_x()*4;i<256*256;i+=256*4)
            __builtin_nontemporal_store(zero,reinterpret_cast<F4*>(base+(i/256)*K+i%256));
        return;
    }
    tile_t tile={n0,N,e,0,256,begin,count,0,1};
    __shared__ __attribute__((aligned(16))) char smem[Base::LDS_BYTES];
    if(count<=32) gemm_a16w16_grouped_tile_body<Trait<32>>(a,tile,k0,smem);
    else if(count<=64) gemm_a16w16_grouped_tile_body<Trait<64>>(a,tile,k0,smem);
    else gemm_a16w16_grouped_tile_body<Trait<0>>(a,tile,k0,smem);
#endif
}
extern "C" __global__ __launch_bounds__(256, Base::WG_PER_CU)
void opus_gemm_gfx950_bf16_k_grouped_256x256x256x64_2x2_16x16x32_0x0x0_tn_f32_shortk(A a) {
    run<3584, 3072>(a);
}

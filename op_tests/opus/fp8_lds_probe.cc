// SPDX-License-Identifier: MIT
// Approach-A foundation: validate that the 16mx8 traits/layouts instantiate with
// D_ATTN=fp8_t and that async_load(fp8)->fp8 LDS, read via u_rk, reproduces the
// SAME logical K as the bf16 path. Uses small-integer data (exact in fp8 & bf16)
// so this tests the LAYOUT (not quantization). PASS => fp8 LDS round-trip works,
// so Approach A (fp8 in LDS + dequant at read) is feasible by instantiation.
//
// Build (docker gfx950):
//  hipcc -std=c++20 --offload-arch=gfx950 -O2 -I csrc/include op_tests/opus/fp8_lds_probe.cc -o /tmp/flp && /tmp/flp
#define PA_SPARSE_PREFILL_OPUS_IMPL
#include "pa_sparse_prefill_opus.h"
#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#endif

using Tbf  = pa_prefill_16mx8_32nx1_traits<16, 32, 512, 8, bf16_t>;
using Tfp8 = pa_prefill_16mx8_fp8_traits<16, 32, 512, 8, unsigned char, bf16_t>;
constexpr int NROW = 32, D = 512;

// read K slice0 (u_rk, no skv offset) for each lane, convert to float, write [64*ELEM]
__global__ void readK(const bf16_t* g_bf, const unsigned char* g_fp8,
                      const int* kvidx, float* out_bf, float* out_fp8)
{
#if defined(__gfx950__)
    using namespace opus;
    using namespace pa_16mx8_32nx1;
    int lane = thread_id_x() % 64;
    int warp = __builtin_amdgcn_readfirstlane(thread_id_x() / 64);

    // ---- bf16 path ----
    {
        using T = Tbf;
        __shared__ char sm[T::smem_kv_tile_elems * sizeof(bf16_t)];
        auto s = make_smem(reinterpret_cast<bf16_t*>(sm));
        auto g = make_gmem(g_bf, NROW * D * (int)sizeof(bf16_t));
        auto u_g = make_layout_gkv<T>(warp, lane);
        auto u_s = make_layout_skv<T>(warp);
        auto u_kvi = make_layout_kv_indices<T>(warp, lane);
        auto g_kvi = make_gmem(kvidx, NROW * (int)sizeof(int));
        int kv_page = load(g_kvi, u_kvi, 0)[0];
        async_load<T::VEC_KV>(g, s.ptr, u_g + kv_page * D, u_s);
        s_waitcnt_vmcnt(0_I); __builtin_amdgcn_s_barrier();
        auto u_rk = make_layout_rk<T>(lane);
        auto vk = load<T::VEC_KV>(s, u_rk);
        constexpr int N = vector_traits<decltype(vk)>::size();
        for (int e = 0; e < N; ++e) out_bf[thread_id_x() * N + e] = (float)vk[e];
    }
    __builtin_amdgcn_s_barrier();
    // ---- fp8 path ----
    {
        using T = Tfp8;
        using D_KV = typename T::D_KV;
        __shared__ char sm[T::smem_kv_tile_elems * sizeof(D_KV)];
        auto s = make_smem(reinterpret_cast<D_KV*>(sm));
        auto g = make_gmem(g_fp8, NROW * D * (int)sizeof(D_KV));
        auto u_g = make_layout_gkv<T>(warp, lane);
        auto u_s = make_layout_skv<T>(warp);
        auto u_kvi = make_layout_kv_indices<T>(warp, lane);
        auto g_kvi = make_gmem(kvidx, NROW * (int)sizeof(int));
        int kv_page = load(g_kvi, u_kvi, 0)[0];
        async_load<T::VEC_KV>(g, s.ptr, u_g + kv_page * D, u_s);
        s_waitcnt_vmcnt(0_I); __builtin_amdgcn_s_barrier();
        auto u_rk = make_layout_rk<T>(lane);
        auto vk = load<T::VEC_KV>(s, u_rk);
        constexpr int N = vector_traits<decltype(vk)>::size();
        for (int e = 0; e < N; ++e) {
            float f = __builtin_amdgcn_cvt_f32_fp8((int)(unsigned char)vk[e], 0);
            out_fp8[thread_id_x() * N + e] = f;
        }
    }
#endif
}

#ifndef __HIP_DEVICE_COMPILE__
static unsigned char i_to_e4m3(int v){ // exact for small ints in [-7..7]ish
    if(v==0)return 0; int s=v<0?0x80:0; v=v<0?-v:v;
    int e=0,x=v; while(x>1){x>>=1;e++;} int M=(int)(((float)v/(1<<e)-1.f)*8+0.5f);
    return s|((e+7)<<3)|(M&7);
}
int main(){
    srand(3);
    int NN=NROW*D;
    bf16_t* hbf=new bf16_t[NN]; unsigned char* hfp=new unsigned char[NN]; int* kvi=new int[NROW];
    for(int r=0;r<NROW;++r){ kvi[r]=r;
        for(int d=0;d<D;++d){ int v=(rand()%7)-3; hbf[r*D+d]=(bf16_t)(float)v; hfp[r*D+d]=i_to_e4m3(v);} }
    int VEC_bf = 16*32/64; // ELEM_B for K, both traits read same logical count
    int OUTN = 512 * (Tbf::GEMM0_E_N * Tbf::GEMM0_E_K * 16 * 32 / 64 / 1); // generous
    OUTN = 512 * 64; // generous upper bound
    void *dbf,*dfp,*dkvi; float *dob,*dof;
    hipMalloc(&dbf,NN*2); hipMemcpy(dbf,hbf,NN*2,hipMemcpyHostToDevice);
    hipMalloc(&dfp,NN);   hipMemcpy(dfp,hfp,NN,hipMemcpyHostToDevice);
    hipMalloc(&dkvi,NROW*4); hipMemcpy(dkvi,kvi,NROW*4,hipMemcpyHostToDevice);
    hipMalloc(&dob,OUTN*4); hipMalloc(&dof,OUTN*4);
    hipMemset(dob,0,OUTN*4); hipMemset(dof,0,OUTN*4);
    hipLaunchKernelGGL(readK,dim3(1),Tbf::BLOCK_SIZE,0,0,(const bf16_t*)dbf,(const unsigned char*)dfp,(const int*)dkvi,dob,dof);
    hipDeviceSynchronize();
    float* hob=new float[OUTN]; float* hof=new float[OUTN];
    hipMemcpy(hob,dob,OUTN*4,hipMemcpyDeviceToHost); hipMemcpy(hof,dof,OUTN*4,hipMemcpyDeviceToHost);
    int mism=0; float maxd=0;
    for(int i=0;i<OUTN;++i){ float d=hob[i]-hof[i]; if(d<0)d=-d; if(d>0.01f){mism++; if(d>maxd)maxd=d;} }
    printf("u_rk K read: bf16-path vs fp8-path (exact-int data)  mism=%d maxd=%.3f -> %s\n",
        mism,maxd, mism==0?"PASS (fp8 LDS layout works)":"FAIL");
    return mism==0?0:1;
}
#endif

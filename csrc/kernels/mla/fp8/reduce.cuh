// SPDX-License-Identifier: MIT
#pragma once
#include "online_common.cuh"
namespace sparse_mla_fp8::online {
template <int Splits, int Mode, int Cols, bool SingleHead = false, bool OutputFP8 = true>
__global__ __launch_bounds__(64) void reduce_splits(Params p)
{
    constexpr int V = Cols / 16;
    int lane = threadIdx.x, head = SingleHead ? int(blockIdx.y) : int(blockIdx.y) * 16 + lane / 4;
    int col =
        SingleHead ? int(blockIdx.z) * 256 + lane * 4 : int(blockIdx.z) * Cols + (lane % 4) * 4;
    int64_t row = int64_t(blockIdx.x) * 64 + head;
    float accum[V][4]{};
    float z = 0.f, m = -INFINITY;
    if constexpr(Mode == 0)
    {
#pragma unroll
        for(int s = 0; s < Splits; ++s)
        {
            float l     = p.partial_lse[(int64_t(blockIdx.x) * Splits + s) * 64 + head];
            float d     = exp2f(-fabsf((m == -INFINITY && l == -INFINITY) ? 0.f : m - l) * LOG2E);
            float alpha = m == -INFINITY ? 0.f : (m >= l ? 1.f : d),
                  beta  = l == -INFINITY ? 0.f : (l >= m ? 1.f : d);
            z           = z * alpha + beta;
            m           = fmaxf(m, l);
#pragma unroll
            for(int v = 0; v < V; ++v)
            {
                F4 x = partial_load(p.partial_o +
                                    ((int64_t(blockIdx.x) * Splits + s) * 64 + head) * 512 + col +
                                    v * 16);
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    accum[v][j] = fmaf(x[j], beta, accum[v][j] * alpha);
            }
        }
    }
    else if constexpr(Mode == 1)
    {
        float ls[Splits];
#pragma unroll
        for(int s = 0; s < Splits; ++s)
        {
            ls[s] = p.partial_lse[(int64_t(blockIdx.x) * Splits + s) * 64 + head];
            m     = fmaxf(m, ls[s]);
        }
#pragma unroll
        for(int s = 0; s < Splits; ++s)
        {
            float w = m == -INFINITY ? 0.f : exp2f((ls[s] - m) * LOG2E);
            z += w;
#pragma unroll
            for(int v = 0; v < V; ++v)
            {
                F4 x = partial_load(p.partial_o +
                                    ((int64_t(blockIdx.x) * Splits + s) * 64 + head) * 512 + col +
                                    v * 16);
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    accum[v][j] = fmaf(w > 0 ? x[j] : 0.f, w, accum[v][j]);
            }
        }
    }
    if constexpr(Mode >= 2)
    {
#pragma unroll
        for(int s = 0; s < Splits; ++s)
            m = fmaxf(m, p.partial_lse[(int64_t(blockIdx.x) * Splits + s) * 64 + head]);
#pragma unroll(Mode == 3 ? 4 : 1)
        for(int s = 0; s < Splits; ++s)
        {
            float l = p.partial_lse[(int64_t(blockIdx.x) * Splits + s) * 64 + head];
            float w = m == -INFINITY ? 0.f : exp2f((l - m) * LOG2E);
            z += w;
#pragma unroll
            for(int v = 0; v < V; ++v)
            {
                F4 x = partial_load(p.partial_o +
                                    ((int64_t(blockIdx.x) * Splits + s) * 64 + head) * 512 + col +
                                    v * 16);
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    accum[v][j] = fmaf(w > 0 ? x[j] : 0.f, w, accum[v][j]);
            }
        }
    }
    float inv = z > 0 ? 1.f / (z * (OutputFP8 ? 1.f : 1.f)) : 0.f;
#pragma unroll
    for(int v = 0; v < V; ++v)
    {
        F4 x;
        for(int j = 0; j < 4; ++j)
        {
            x[j] = accum[v][j] * inv;
        }
        if constexpr(OutputFP8)
            *reinterpret_cast<uint32_t*>(p.out + row * 512 + col + v * 16) = pack4(x);
        else
        {
            uint32_t lo, hi;
            asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(lo) : "v"(x[0]), "v"(x[1]));
            asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(hi) : "v"(x[2]), "v"(x[3]));
            *reinterpret_cast<I2*>(reinterpret_cast<uint16_t*>(p.out) + row * 512 + col + v * 16) =
                I2{int(lo), int(hi)};
        }
    }
    if(p.lse && blockIdx.z == 0 && (SingleHead ? lane == 0 : lane % 4 == 0))
        p.lse[row] = z > 0 ? m + log2f(z) * LN2 : -INFINITY;
}
} // namespace sparse_mla_fp8::online

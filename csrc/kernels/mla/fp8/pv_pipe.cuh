// SPDX-License-Identifier: MIT
// PV fragment batching only for the selected stage1 template.
#ifndef PV_PIPE_READ
#define PV_PIPE_READ 0
#endif
#ifndef PV_PIPE_WAIT
#define PV_PIPE_WAIT 2
#endif
static_assert(PV_PIPE_READ == 0 || PV_PIPE_READ == 1 || PV_PIPE_READ == 2 || PV_PIPE_READ == 4);
static_assert(PV_PIPE_WAIT == 1 || PV_PIPE_WAIT == 2);
__device__ __forceinline__ I8 pv_pipe_fragment(const uint8_t* base, int col_base)
{
    const int lane = threadIdx.x & 63;
    I8 x;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        int row       = (lane / 16) * 32 + i * 8 + (lane % 16) / 2;
        int col       = col_base + (lane % 2) * 8;
        uint32_t addr = uint32_t(reinterpret_cast<uintptr_t>(base + address<128>(row, col)));
        I2 r;
        asm volatile("ds_read_b64_tr_b8 %0, %1" : "=v"(r) : "v"(addr) : "memory");
        x[i * 2]     = r[0];
        x[i * 2 + 1] = r[1];
    }
    return x;
}
template <int N>
__device__ __forceinline__ void pv_pipe_wait(I8 (&v)[N])
{
#if PV_PIPE_WAIT == 2
#define PV_PIPE_WAIT_TEXT "s_waitcnt lgkmcnt(0)"
#else
#define PV_PIPE_WAIT_TEXT "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"
#endif
    if constexpr(N == 1)
        asm volatile(PV_PIPE_WAIT_TEXT : "+v"(v[0]) : : "memory");
    if constexpr(N == 2)
        asm volatile(PV_PIPE_WAIT_TEXT : "+v"(v[0]), "+v"(v[1]) : : "memory");
    if constexpr(N == 4)
        asm volatile(PV_PIPE_WAIT_TEXT
                     : "+v"(v[0]), "+v"(v[1]), "+v"(v[2]), "+v"(v[3])
                     :
                     : "memory");
#undef PV_PIPE_WAIT_TEXT
}

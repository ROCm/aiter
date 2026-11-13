// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ReplaySSM decode / speculative-verify step of KDA (Kimi Delta Attention), the linear-attention layer of Kimi
// models: the HIP counterpart of the Triton `_replayssm_kda_fwd_kernel` (ATOM fla_ops/replayssm.py), with the same
// inputs, outputs and record semantics.  ReplaySSM keeps one fp32 checkpoint per request plus a short buffer of
// (k, u, g) records instead of a state per speculative token.  Per (request, value head):
//   1. S_h = checkpoint ([V, K], fp32) with the h committed records replayed on top,
//      S_{j+1} = diag_K(e^{g_j}) S_j + u_j (x) k_j            (fp32 forward recurrence)
//   2. on a flush (h + 2 * max_query_len > cache_len) S_h is written back to the checkpoint and the records of this
//      step start again at position 0
//   3. for each of the T tokens of this step: per-K gate g (lower-bounded sigmoid, or -exp(A) softplus), optional
//      L2 norm of q and k, S *= e^g, u = sigmoid(b) (v - S k), S += u (x) k, o = S q; the (k, u, g) of the token is
//      appended to the records at base + t.
//
// One block of 4 * 128 / RPT threads per (request, value head), three phases separated by barriers:
//   * staging: 16 threads per 256-byte row load the T tokens' q, k, v, a rows and the record rows with 16-byte
//     loads and branch-free clamped row indices (every load is in flight at once), compute the gates and all
//     exponentials once per element and park everything in LDS as fp32;
//   * compute: a group of 4 adjacent lanes owns RPT V rows of the 128 x 128 fp32 state, lane kq the 16-byte column
//     chunks kq, kq + 4, ..., kq + 28, so each K-reduction is 32 in-lane FMAs over 4 accumulators plus two DPP
//     quad_perm adds; per-token outputs go to LDS;
//   * write-back: every global store of the step (o and the u/k/g records), coalesced, under block-uniform
//     conditions.
// Nothing touches memory inside divergent control flow: on gfx1250 a wave waits for its outstanding memory
// transactions (s_wait_xcnt) before EXEC changes, so predicated loads/stores would each cost a memory round trip.
// The staging barrier also orders every record read before any record write, so a flush (which rewrites the
// records from position 0) cannot race another thread's replay.
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "kda_replayssm_decode.h"

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace aiter {
namespace kda_replayssm {

#define KDA_DEVI __device__ __forceinline__

constexpr int KD     = 128; // head_k_dim
constexpr int VD     = 128; // head_v_dim
constexpr int CPT    = 32;  // K columns per lane
constexpr int MAXT   = 8;   // max tokens per request in one call (verify window)
constexpr int MAXCAP = 16;  // max record buffer length (cache_len)

KDA_DEVI float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
KDA_DEVI float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
KDA_DEVI uint16_t f2bf(float f)
{
    // round to nearest even, NaN stays NaN; branch-free
    const uint32_t u   = __float_as_uint(f);
    const uint32_t r   = u + 0x7fffu + ((u >> 16) & 1u);
    const bool is_nan  = (u & 0x7fffffffu) > 0x7f800000u;
    return (uint16_t)((is_nan ? (u | 0x400000u) : r) >> 16);
}
KDA_DEVI void unpack8(const uint4 w, float* f)
{
    f[0] = bf_lo(w.x);
    f[1] = bf_hi(w.x);
    f[2] = bf_lo(w.y);
    f[3] = bf_hi(w.y);
    f[4] = bf_lo(w.z);
    f[5] = bf_hi(w.z);
    f[6] = bf_lo(w.w);
    f[7] = bf_hi(w.w);
}
// 8 consecutive g-record values as fp32 (the g records may be fp32 or bf16)
template <typename GT>
KDA_DEVI void ldg8(const GT* p, float* f);
template <>
KDA_DEVI void ldg8<uint16_t>(const uint16_t* p, float* f)
{
    unpack8(*reinterpret_cast<const uint4*>(p), f);
}
template <>
KDA_DEVI void ldg8<float>(const float* p, float* f)
{
    const float4 x = reinterpret_cast<const float4*>(p)[0];
    const float4 y = reinterpret_cast<const float4*>(p)[1];
    f[0] = x.x;
    f[1] = x.y;
    f[2] = x.z;
    f[3] = x.w;
    f[4] = y.x;
    f[5] = y.y;
    f[6] = y.z;
    f[7] = y.w;
}
KDA_DEVI void stg(uint16_t* p, float x) { *p = f2bf(x); }
KDA_DEVI void stg(float* p, float x) { *p = x; }
KDA_DEVI void st8(float* p, const float* f) // 8 fp32 to LDS
{
    reinterpret_cast<float4*>(p)[0] = make_float4(f[0], f[1], f[2], f[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(f[4], f[5], f[6], f[7]);
}
// this lane's 32 columns of an LDS row: 16-byte chunks kq, kq + 4, ..., kq + 28
KDA_DEVI void ldrow(const float* row, int kq, float* out)
{
#pragma unroll
    for(int m = 0; m < CPT / 4; ++m)
    {
        const float4 x  = reinterpret_cast<const float4*>(row)[4 * m + kq];
        out[4 * m + 0] = x.x;
        out[4 * m + 1] = x.y;
        out[4 * m + 2] = x.z;
        out[4 * m + 3] = x.w;
    }
}
// sum over the 4 adjacent lanes of a group: DPP quad_perm [1,0,3,2] then [2,3,0,1]
KDA_DEVI float sum4(float x)
{
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xB1, 0xF, 0xF, false));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4E, 0xF, 0xF, false));
    return x;
}
// sum over the 16 lanes of a DPP row: row_xmask 1, 2, 4, 8 (DPP16); gfx9 falls back to shuffles
KDA_DEVI float sum16(float x)
{
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || \
    defined(__gfx950__)
    x += __shfl_xor(x, 1, 16);
    x += __shfl_xor(x, 2, 16);
    x += __shfl_xor(x, 4, 16);
    x += __shfl_xor(x, 8, 16);
#else
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x161, 0xF, 0xF, false));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x162, 0xF, 0xF, false));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x164, 0xF, 0xF, false));
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x168, 0xF, 0xF, false));
#endif
    return x;
}
KDA_DEVI float fast_rcp(float x) { return __builtin_amdgcn_rcpf(x); }

struct Params
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* a;
    const uint16_t* b;
    const float* A_log;
    const float* dt_bias;
    uint16_t* o;
    float* ckpt;
    uint16_t* buf_k;
    uint16_t* buf_u;
    void* buf_g;
    const int* write_pos;
    const int* slot_idx;
    const int* cu_seqlens;
    float scale;
    float lower_bound;
    int use_lower_bound;
    int use_l2norm;
    int max_query_len;
    int cap;
    int H;
    int HV;
    int64_t s_ck_slot, s_bk_slot, s_bk_pos, s_bk_hv, s_bu_slot, s_bu_pos, s_bu_hv, s_bg_slot, s_bg_pos, s_bg_hv;
    int64_t s_a_tok, s_b_tok; // token strides of the gate input and beta (column views of a wider buffer allowed)
};

template <int RPT, typename GT>
__global__ void __launch_bounds__(4 * VD / RPT) kda_replayssm_decode_kernel(const Params p)
{
    constexpr int NT  = 4 * VD / RPT;
    constexpr int RPP = NT / 16; // staging rows per pass (16 threads x 8 elements per 128-element row)
    __shared__ float s_q[MAXT][KD], s_k[MAXT][KD], s_v[MAXT][VD]; // q (normalized, scaled), k (normalized), v
    __shared__ float s_g[MAXT][KD], s_e[MAXT][KD];                // gates g and e^g
    __shared__ float s_o[MAXT][VD], s_u[MAXT][VD];                // outputs and u records, written back at the end
    __shared__ float s_beta[MAXT];                                // sigmoid(b)
    __shared__ float s_rk[MAXCAP][KD], s_ru[MAXCAP][VD], s_re[MAXCAP][KD]; // records: k, u, e^g

    GT* const buf_g = reinterpret_cast<GT*>(p.buf_g);
    const int tid   = threadIdx.x;
    const int kq    = tid & 3;
    const int rg    = tid >> 2;
    const int r0    = rg * RPT;
    const int nh    = blockIdx.x;
    const int n     = nh / p.HV;
    const int hv    = nh - n * p.HV;
    const int ih    = hv / (p.HV / p.H);
    const int bos   = p.cu_seqlens[n];
    const int T     = min(p.cu_seqlens[n + 1] - bos, MAXT);
    if(T <= 0)
        return;
    const int64_t slot = p.slot_idx[n];
    if(slot < 0)
        return;
    // the prefill sentinel -1 means "no records": fold nothing, append from 0
    const int h           = min(max(p.write_pos[slot], 0), MAXCAP);
    const bool do_flush   = h + 2 * p.max_query_len > p.cap;
    const int64_t base    = do_flush ? 0 : h;
    const int64_t HV      = p.HV;
    float S[RPT][CPT];
    float* ck = p.ckpt + slot * p.s_ck_slot + (int64_t)hv * VD * KD + (int64_t)r0 * KD;

    // ---- staging: 16 threads per row, 8 elements (16 bytes) per thread, rows clamped to valid ones (branch-free).
    // The staging loads are issued before the checkpoint loads: load counters retire in order, so the gate math can
    // start while the 64 KB checkpoint tile is still in flight.
    {
        const int ch = tid & 15, sr = tid >> 4, col = ch * 8;
        constexpr int TP = (MAXT + RPP - 1) / RPP, JP = (MAXCAP + RPP - 1) / RPP;
        uint4 wa[TP], wq[TP], wk[TP], wv[TP];
        int trow[TP];
#pragma unroll
        for(int i = 0; i < TP; ++i)
        {
            const int t       = min(sr + i * RPP, T - 1);
            trow[i]           = t;
            const int64_t tok = bos + t;
            wa[i] = *reinterpret_cast<const uint4*>(p.a + tok * p.s_a_tok + hv * KD + col);
            wq[i] = *reinterpret_cast<const uint4*>(p.q + (tok * p.H + ih) * KD + col);
            wk[i] = *reinterpret_cast<const uint4*>(p.k + (tok * p.H + ih) * KD + col);
            wv[i] = *reinterpret_cast<const uint4*>(p.v + (tok * HV + hv) * VD + col);
        }
        float xg[JP][8];
        uint4 xk[JP], xu[JP];
        int jrow[JP];
        // record rows are clamped to the buffer, not to h: every row of the slot's buffer is valid memory, rows
        // >= h are simply not replayed, and the loads need not wait for write_pos[slot]
        const int jmax = min(p.cap, MAXCAP) - 1;
#pragma unroll
        for(int i = 0; i < JP; ++i)
        {
            const int64_t j = min(sr + i * RPP, jmax);
            jrow[i]         = (int)j;
            ldg8<GT>(buf_g + slot * p.s_bg_slot + hv * p.s_bg_hv + j * p.s_bg_pos + col, xg[i]);
            xk[i] = *reinterpret_cast<const uint4*>(p.buf_k + slot * p.s_bk_slot + hv * p.s_bk_hv +
                                                    j * p.s_bk_pos + col);
            xu[i] = *reinterpret_cast<const uint4*>(p.buf_u + slot * p.s_bu_slot + hv * p.s_bu_hv +
                                                    j * p.s_bu_pos + col);
        }
        const int bt      = min(tid & (MAXT - 1), T - 1);
        const uint16_t rb = p.b[(int64_t)(bos + bt) * p.s_b_tok + hv];
        const float4 db0  = reinterpret_cast<const float4*>(p.dt_bias + (int64_t)hv * KD + col)[0];
        const float4 db1  = reinterpret_cast<const float4*>(p.dt_bias + (int64_t)hv * KD + col)[1];
        // checkpoint tile into registers (consumed only after the staging barrier)
#pragma unroll
        for(int r = 0; r < RPT; ++r)
        {
            const float4* src = reinterpret_cast<const float4*>(ck + r * KD);
#pragma unroll
            for(int m = 0; m < CPT / 4; ++m)
            {
                const float4 x   = src[4 * m + kq];
                S[r][4 * m + 0] = x.x;
                S[r][4 * m + 1] = x.y;
                S[r][4 * m + 2] = x.z;
                S[r][4 * m + 3] = x.w;
            }
        }
        const float dtb[8] = {db0.x, db0.y, db0.z, db0.w, db1.x, db1.y, db1.z, db1.w};
        const float eA     = __expf(p.A_log[hv]);
#pragma unroll
        for(int i = 0; i < TP; ++i)
        {
            float fa[8], f[8], g[8], e[8];
            unpack8(wa[i], fa);
#pragma unroll
            for(int c = 0; c < 8; ++c)
            {
                const float x = fa[c] + dtb[c];
                g[c]          = p.use_lower_bound ? p.lower_bound * fast_rcp(1.f + __expf(-eA * x))
                                                  : -eA * (x <= 20.f ? __logf(1.f + __expf(x)) : x);
                e[c]          = __expf(g[c]);
            }
            const int t = trow[i];
            st8(&s_g[t][col], g);
            st8(&s_e[t][col], e);
            float fq[8], fk[8];
            unpack8(wq[i], fq);
            unpack8(wk[i], fk);
            float iq = p.scale, ik = 1.f;
            if(p.use_l2norm)
            {
                // the row's 16 threads hold its 128 elements: one DPP-row reduction per norm
                float sq = 0.f, sk = 0.f;
#pragma unroll
                for(int c = 0; c < 8; ++c)
                {
                    sq = fmaf(fq[c], fq[c], sq);
                    sk = fmaf(fk[c], fk[c], sk);
                }
                iq *= rsqrtf(sum16(sq) + 1e-6f);
                ik = rsqrtf(sum16(sk) + 1e-6f);
            }
#pragma unroll
            for(int c = 0; c < 8; ++c)
            {
                fq[c] *= iq;
                fk[c] *= ik;
            }
            st8(&s_q[t][col], fq);
            st8(&s_k[t][col], fk);
            unpack8(wv[i], f);
            st8(&s_v[t][col], f);
        }
        s_beta[tid & (MAXT - 1)] = fast_rcp(1.f + __expf(-__uint_as_float(((uint32_t)rb) << 16)));
#pragma unroll
        for(int i = 0; i < JP; ++i)
        {
            float f[8];
            const int j = jrow[i];
#pragma unroll
            for(int c = 0; c < 8; ++c)
                f[c] = __expf(xg[i][c]);
            st8(&s_re[j][col], f);
            unpack8(xk[i], f);
            st8(&s_rk[j][col], f);
            unpack8(xu[i], f);
            st8(&s_ru[j][col], f);
        }
    }
    __syncthreads(); // staging complete; every record read happens before any record write

    // ---- 1. rebuild S_h from LDS, two records per iteration
    {
        int j = 0;
        for(; j + 1 < h; j += 2)
        {
            float k0[CPT], e0[CPT], k1[CPT], e1[CPT], u0[RPT], u1[RPT];
            ldrow(s_rk[j], kq, k0);
            ldrow(s_re[j], kq, e0);
            ldrow(s_rk[j + 1], kq, k1);
            ldrow(s_re[j + 1], kq, e1);
#pragma unroll
            for(int r = 0; r < RPT; ++r)
            {
                u0[r] = s_ru[j][r0 + r];
                u1[r] = s_ru[j + 1][r0 + r];
            }
#pragma unroll
            for(int c = 0; c < CPT; ++c)
            {
#pragma unroll
                for(int r = 0; r < RPT; ++r)
                {
                    const float s1 = fmaf(S[r][c], e0[c], u0[r] * k0[c]);
                    S[r][c]        = fmaf(s1, e1[c], u1[r] * k1[c]);
                }
            }
        }
        if(j < h)
        {
            float k0[CPT], e0[CPT], u0[RPT];
            ldrow(s_rk[j], kq, k0);
            ldrow(s_re[j], kq, e0);
#pragma unroll
            for(int r = 0; r < RPT; ++r)
                u0[r] = s_ru[j][r0 + r];
#pragma unroll
            for(int c = 0; c < CPT; ++c)
            {
#pragma unroll
                for(int r = 0; r < RPT; ++r)
                    S[r][c] = fmaf(S[r][c], e0[c], u0[r] * k0[c]);
            }
        }
    }
    // ---- 2. flush (block-uniform): the checkpoint becomes S_h, records restart at 0
    if(do_flush)
    {
#pragma unroll
        for(int r = 0; r < RPT; ++r)
        {
            float4* dst = reinterpret_cast<float4*>(ck + r * KD);
#pragma unroll
            for(int m = 0; m < CPT / 4; ++m)
                dst[4 * m + kq] =
                    make_float4(S[r][4 * m + 0], S[r][4 * m + 1], S[r][4 * m + 2], S[r][4 * m + 3]);
        }
    }

    // ---- 3. this step's tokens (LDS in, LDS out)
#pragma unroll 2
    for(int t = 0; t < T; ++t)
    {
        float qv[CPT], kv[CPT], ev[CPT];
        ldrow(s_q[t], kq, qv);
        ldrow(s_k[t], kq, kv);
        ldrow(s_e[t], kq, ev);
        const float beta = s_beta[t];
        float dk[RPT][4];
#pragma unroll
        for(int r = 0; r < RPT; ++r)
#pragma unroll
            for(int i = 0; i < 4; ++i)
                dk[r][i] = 0.f;
#pragma unroll
        for(int c = 0; c < CPT; ++c)
        {
#pragma unroll
            for(int r = 0; r < RPT; ++r)
            {
                S[r][c] *= ev[c];
                dk[r][c & 3] = fmaf(S[r][c], kv[c], dk[r][c & 3]);
            }
        }
        float u[RPT];
#pragma unroll
        for(int r = 0; r < RPT; ++r)
            u[r] = beta * (s_v[t][r0 + r] - sum4((dk[r][0] + dk[r][1]) + (dk[r][2] + dk[r][3])));
        float oq[RPT][4];
#pragma unroll
        for(int r = 0; r < RPT; ++r)
#pragma unroll
            for(int i = 0; i < 4; ++i)
                oq[r][i] = 0.f;
#pragma unroll
        for(int c = 0; c < CPT; ++c)
        {
#pragma unroll
            for(int r = 0; r < RPT; ++r)
            {
                S[r][c]      = fmaf(u[r], kv[c], S[r][c]);
                oq[r][c & 3] = fmaf(S[r][c], qv[c], oq[r][c & 3]);
            }
        }
        float ot[RPT];
#pragma unroll
        for(int r = 0; r < RPT; ++r)
            ot[r] = sum4((oq[r][0] + oq[r][1]) + (oq[r][2] + oq[r][3]));
        // lane kq stores row r0 + kq % RPT (lanes beyond RPT duplicate identical values: no divergence)
        const int rr = kq % RPT;
        float my_o = ot[0], my_u = u[0];
#pragma unroll
        for(int r = 1; r < RPT; ++r)
        {
            my_o = rr == r ? ot[r] : my_o;
            my_u = rr == r ? u[r] : my_u;
        }
        s_o[t][r0 + rr] = my_o;
        s_u[t][r0 + rr] = my_u;
    }
    __syncthreads();

    // ---- write-back: the waves of threads 0..127 (thread = V row / K column), one coalesced row per array and token
    if(__builtin_amdgcn_readfirstlane(tid) < 128)
    {
        uint16_t* po = p.o + ((int64_t)bos * HV + hv) * VD + tid;
        uint16_t* pu = p.buf_u + slot * p.s_bu_slot + hv * p.s_bu_hv + base * p.s_bu_pos + tid;
        uint16_t* pk = p.buf_k + slot * p.s_bk_slot + hv * p.s_bk_hv + base * p.s_bk_pos + tid;
        GT* pg       = buf_g + slot * p.s_bg_slot + hv * p.s_bg_hv + base * p.s_bg_pos + tid;
        uint16_t wo[MAXT], wu[MAXT], wk[MAXT];
        float wg[MAXT];
#pragma unroll
        for(int t = 0; t < MAXT; ++t)
        {
            // all LDS reads first (rows >= T are harmless), then the stores
            wo[t] = f2bf(s_o[t][tid]);
            wu[t] = f2bf(s_u[t][tid]);
            wk[t] = f2bf(s_k[t][tid]);
            wg[t] = s_g[t][tid];
        }
#pragma unroll
        for(int t = 0; t < MAXT; ++t)
        {
            if(t < T) // T is block-uniform: a scalar branch, EXEC untouched
            {
                po[(int64_t)t * HV * VD] = wo[t];
                pu[t * p.s_bu_pos]       = wu[t];
                pk[t * p.s_bk_pos]       = wk[t];
                stg(pg + t * p.s_bg_pos, wg[t]);
            }
        }
    }
}

#undef KDA_DEVI

static int device_warp_size(int dev)
{
    static int cached[64] = {0};
    if(dev < 0 || dev >= 64)
        return 64;
    if(cached[dev] == 0)
    {
        int ws = 64;
        HIP_CALL(hipDeviceGetAttribute(&ws, hipDeviceAttributeWarpSize, dev));
        cached[dev] = ws;
    }
    return cached[dev];
}

} // namespace kda_replayssm

void kda_replayssm_decode(aiter_tensor_t& o,
                          aiter_tensor_t& ckpt,
                          aiter_tensor_t& buf_k,
                          aiter_tensor_t& buf_u,
                          aiter_tensor_t& buf_g,
                          const aiter_tensor_t& q,
                          const aiter_tensor_t& k,
                          const aiter_tensor_t& v,
                          const aiter_tensor_t& a,
                          const aiter_tensor_t& b,
                          const aiter_tensor_t& A_log,
                          const aiter_tensor_t& dt_bias,
                          const aiter_tensor_t& write_pos,
                          const aiter_tensor_t& slot_idx,
                          const aiter_tensor_t& cu_seqlens,
                          int max_query_len,
                          float scale,
                          bool use_qk_l2norm,
                          bool use_lower_bound,
                          float lower_bound)
{
    using namespace kda_replayssm;
    const int64_t K   = q.size(-1);
    const int64_t H   = q.size(-2);
    const int64_t V   = v.size(-1);
    const int64_t HV  = v.size(-2);
    const int64_t cap = buf_k.size(-2);
    const int64_t N   = cu_seqlens.numel() - 1;
    AITER_CHECK(K == KD && V == VD, "kda_replayssm_decode: head dims must be 128, got K=", K, " V=", V);
    AITER_CHECK(H > 0 && HV % H == 0, "kda_replayssm_decode: HV=", HV, " must be a multiple of H=", H);
    AITER_CHECK(max_query_len >= 1 && max_query_len <= MAXT,
                "kda_replayssm_decode: max_query_len must be in [1, ", MAXT, "], got ", max_query_len);
    AITER_CHECK(cap <= MAXCAP && cap >= 2 * max_query_len,
                "kda_replayssm_decode: cache_len must be in [2 * max_query_len, ", MAXCAP, "], got ", cap);
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 && k.dtype() == AITER_DTYPE_bf16 && v.dtype() == AITER_DTYPE_bf16 &&
                    a.dtype() == AITER_DTYPE_bf16 && b.dtype() == AITER_DTYPE_bf16 &&
                    o.dtype() == AITER_DTYPE_bf16,
                "kda_replayssm_decode: q, k, v, a, b and o must be bf16");
    AITER_CHECK(ckpt.dtype() == AITER_DTYPE_fp32 && A_log.dtype() == AITER_DTYPE_fp32 &&
                    dt_bias.dtype() == AITER_DTYPE_fp32,
                "kda_replayssm_decode: ckpt, A_log and dt_bias must be fp32");
    AITER_CHECK(buf_k.dtype() == AITER_DTYPE_bf16 && buf_u.dtype() == AITER_DTYPE_bf16 &&
                    (buf_g.dtype() == AITER_DTYPE_fp32 || buf_g.dtype() == AITER_DTYPE_bf16),
                "kda_replayssm_decode: k/u records must be bf16, g records fp32 or bf16");
    AITER_CHECK(write_pos.dtype() == AITER_DTYPE_i32 && slot_idx.dtype() == AITER_DTYPE_i32 &&
                    cu_seqlens.dtype() == AITER_DTYPE_i32,
                "kda_replayssm_decode: write_pos, slot_idx and cu_seqlens must be int32");
    AITER_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && o.is_contiguous(),
                "kda_replayssm_decode: q, k, v and o must be contiguous");
    // a and b may be column views of a wider buffer (Kimi-K3's beta is): only each token's row must be dense
    AITER_CHECK(a.stride(-1) == 1 && a.stride(-2) == K && (a.stride(-3) * 2) % 16 == 0 && b.stride(-1) == 1,
                "kda_replayssm_decode: a needs dense, 16-byte aligned token rows and b unit head stride");
    AITER_CHECK(ckpt.stride(-1) == 1 && ckpt.stride(-2) == K && ckpt.stride(-3) == V * K,
                "kda_replayssm_decode: ckpt must be [slot, HV, V, K] with contiguous heads");
    AITER_CHECK(buf_k.stride(-1) == 1 && buf_u.stride(-1) == 1 && buf_g.stride(-1) == 1 &&
                    buf_k.stride(-2) % 8 == 0 && buf_u.stride(-2) % 8 == 0 && buf_g.stride(-2) % 8 == 0,
                "kda_replayssm_decode: record buffers must be [slot, HV, cache_len, D] with unit stride");
    const aiter_tensor_t* aligned[] = {&q, &k, &v, &a, &o, &dt_bias, &ckpt, &buf_k, &buf_u, &buf_g};
    for(const aiter_tensor_t* t : aligned)
        AITER_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                    "kda_replayssm_decode: tensors must be 16-byte aligned");
    if(N <= 0 || HV <= 0)
        return;

    Params prm;
    prm.q               = static_cast<const uint16_t*>(q.data_ptr());
    prm.k               = static_cast<const uint16_t*>(k.data_ptr());
    prm.v               = static_cast<const uint16_t*>(v.data_ptr());
    prm.a               = static_cast<const uint16_t*>(a.data_ptr());
    prm.b               = static_cast<const uint16_t*>(b.data_ptr());
    prm.A_log           = static_cast<const float*>(A_log.data_ptr());
    prm.dt_bias         = static_cast<const float*>(dt_bias.data_ptr());
    prm.o               = static_cast<uint16_t*>(o.data_ptr());
    prm.ckpt            = static_cast<float*>(ckpt.data_ptr());
    prm.buf_k           = static_cast<uint16_t*>(buf_k.data_ptr());
    prm.buf_u           = static_cast<uint16_t*>(buf_u.data_ptr());
    prm.buf_g           = buf_g.data_ptr();
    prm.write_pos       = static_cast<const int*>(write_pos.data_ptr());
    prm.slot_idx        = static_cast<const int*>(slot_idx.data_ptr());
    prm.cu_seqlens      = static_cast<const int*>(cu_seqlens.data_ptr());
    prm.scale           = scale;
    prm.lower_bound     = lower_bound;
    prm.use_lower_bound = use_lower_bound ? 1 : 0;
    prm.use_l2norm      = use_qk_l2norm ? 1 : 0;
    prm.max_query_len   = max_query_len;
    prm.cap             = (int)cap;
    prm.H               = (int)H;
    prm.HV              = (int)HV;
    prm.s_ck_slot       = ckpt.stride(0);
    prm.s_bk_slot       = buf_k.stride(0);
    prm.s_bk_hv         = buf_k.stride(1);
    prm.s_bk_pos        = buf_k.stride(2);
    prm.s_bu_slot       = buf_u.stride(0);
    prm.s_bu_hv         = buf_u.stride(1);
    prm.s_bu_pos        = buf_u.stride(2);
    prm.s_bg_slot       = buf_g.stride(0);
    prm.s_bg_hv         = buf_g.stride(1);
    prm.s_bg_pos        = buf_g.stride(2);
    prm.s_a_tok         = a.stride(-3);
    prm.s_b_tok         = b.stride(-2);

    HipDeviceGuard device_guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const dim3 grid((unsigned)(N * HV));
    const bool g_fp32 = buf_g.dtype() == AITER_DTYPE_fp32;
    // 4 V rows per lane on wave32 parts (gfx1250: best at every batch measured); 2 rows per lane on wave64 parts,
    // where the 128-thread block is only two waves
    const int rpt = device_warp_size(q.device_id) == 32 ? 4 : 2;
    if(rpt == 4)
    {
        if(g_fp32)
            kda_replayssm_decode_kernel<4, float><<<grid, dim3(4 * VD / 4), 0, stream>>>(prm);
        else
            kda_replayssm_decode_kernel<4, uint16_t><<<grid, dim3(4 * VD / 4), 0, stream>>>(prm);
    }
    else
    {
        if(g_fp32)
            kda_replayssm_decode_kernel<2, float><<<grid, dim3(4 * VD / 2), 0, stream>>>(prm);
        else
            kda_replayssm_decode_kernel<2, uint16_t><<<grid, dim3(4 * VD / 2), 0, stream>>>(prm);
    }
    HIP_CALL(hipGetLastError());
}

} // namespace aiter

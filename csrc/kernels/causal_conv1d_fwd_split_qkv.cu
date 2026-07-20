// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Prefill causal conv1d with fused split q/k/v output (HIP). grid.x indexes a
// flattened (sequence, chunk) schedule decoded through batch_ptr /
// token_chunk_offset_ptr; grid.y is the feature block.

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "causal_conv1d_fwd_split_qkv.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_f32(unsigned short v)
{
    union {
        unsigned int u;
        float f;
    } t;
    t.u = (unsigned int)v << 16;
    return t.f;
}
__device__ __forceinline__ unsigned short f32_to_bf16(float v)
{
    union {
        unsigned int u;
        float f;
    } t;
    t.f = v;
    return (unsigned short)(t.u >> 16);
}

static constexpr int TN  = 64;   // feature tile
static constexpr int KW  = 4;    // kernel width (conv window)
static constexpr int NT  = 256;  // threads per block

// Cooperative-staging load + full conv_state, templated on TM in {8,16,32,64}.
// Fast path covers fully-interior tiles; the slow path applies
// sequence-relative bounds at chunk/sequence boundaries and blends conv_states.
template <int TM>
__global__ __launch_bounds__(256) void conv1d_split_qkv_t(const unsigned short* __restrict__ x,
                                                     const unsigned short* __restrict__ w,
                                                     const unsigned short* __restrict__ bias_ptr,
                                                     unsigned short* __restrict__ q_out,
                                                     unsigned short* __restrict__ k_out,
                                                     unsigned short* __restrict__ v_out,
                                                     unsigned short* __restrict__ conv_states,
                                                     const int* __restrict__ cache_indices,
                                                     const unsigned char* __restrict__ has_initial_state,
                                                     const int* __restrict__ query_start_loc,
                                                     const int* __restrict__ batch_ptr,
                                                     const int* __restrict__ token_chunk_offset_ptr,
                                                     int dim,
                                                     int k_dim,
                                                     int v_dim,
                                                     int T,
                                                     int stride_x_dim,
                                                     int stride_q_tok,
                                                     int stride_q_dim,
                                                     int stride_k_tok,
                                                     int stride_k_dim,
                                                     int stride_v_tok,
                                                     int stride_v_dim,
                                                     int scs0,
                                                     int scs1,
                                                     int scs2,
                                                     int sci,
                                                     int has_bias,
                                                     int do_silu,
                                                     int pad_slot_id)
{
    const int pid     = blockIdx.x;
    const int seq_idx = batch_ptr[pid];
    if(seq_idx == pad_slot_id)
        return;

    const int chunk_idx  = token_chunk_offset_ptr[pid];
    const int seq_start  = query_start_loc[seq_idx];
    const int seqlen     = query_start_loc[seq_idx + 1] - seq_start;
    const int feat_start = (int)blockIdx.y * TN;
    const int tok_start  = chunk_idx * TM;
    const int tid        = threadIdx.x;

    // TM-derived tile constants.
    constexpr int LDS_PAD = TM + KW;      // halo(KW-1) + body(TM) + pad(1)
    constexpr int EPT     = TM / 4;       // outputs per thread (4 token groups)
    constexpr int FG      = NT / TM;      // feat-base groups in cooperative load
    constexpr int ELEMS   = TN * TM / NT; // body features loaded per thread
    constexpr int LOG2_TM = (TM == 8) ? 3 : (TM == 16) ? 4 : (TM == 32) ? 5
                                         : (TM == 64)   ? 6
                                         : (TM == 128)  ? 7
                                                        : 0;

    __shared__ unsigned short shmem[TN * (TM + KW)];

    const int feat_local  = tid >> 2;
    const int tok_group   = tid & 3;
    const int tok_base    = tok_group * EPT;
    const int gfeat       = feat_start + feat_local;
    const bool feat_valid = (gfeat < dim);

    // Issue weight + bias loads early (raw bf16, defer conversion).
    unsigned int w_raw0 = 0, w_raw1 = 0;
    unsigned short b_raw = 0;
    if(feat_valid)
    {
        const unsigned short* wp = w + gfeat * KW;
        w_raw0                   = *(const unsigned int*)wp;
        w_raw1                   = *(const unsigned int*)(wp + 2);
        if(has_bias)
            b_raw = bias_ptr[gfeat];
    }

    // Cooperative load (TM-generic).
    const long long tok_gbase = (long long)seq_start + tok_start - (KW - 1);
    {
        const int t_const = tid & (TM - 1);                  // token in tile (TM pow2)
        const int f_base  = (int)((unsigned)tid >> LOG2_TM); // 0..FG-1
        const int hc      = tid >> 6;                        // halo column 0..3
        const int hf      = tid & 63;                        // halo feature row 0..63

        const long long gt1 = tok_gbase + t_const + (KW - 1);

        const bool all_feat = (feat_start + TN <= dim);
        const bool all_tok1 = (tok_start + TM - 1 < seqlen);
        const bool all_tok2 = (tok_start >= (KW - 1));

        unsigned short vbuf[ELEMS];

        if(all_feat && all_tok1 && all_tok2)
        {
            // Fast path: fully interior, coalesced, no bounds/state.
            long long addr        = (long long)(feat_start + f_base) * stride_x_dim + gt1;
            const long long fstep = (long long)FG * stride_x_dim;
#pragma unroll
            for(int j = 0; j < ELEMS; j++)
            {
                vbuf[j] = x[addr];
                addr += fstep;
            }

            // Issue the halo load early so its latency overlaps the store phase.
            const bool do_halo     = (hc < (KW - 1));
            unsigned short prefix_v = 0;
            if(do_halo)
                prefix_v = x[(long long)(feat_start + hf) * stride_x_dim + (tok_gbase + hc)];

            int lds_off             = f_base * LDS_PAD + t_const + (KW - 1);
            constexpr int LDS_FSTEP = FG * LDS_PAD;
#pragma unroll
            for(int j = 0; j < ELEMS; j++)
            {
                shmem[lds_off] = vbuf[j];
                lds_off += LDS_FSTEP;
            }

            if(do_halo)
                shmem[hf * LDS_PAD + hc] = prefix_v;
        }
        else
        {
            // Slow path: sequence-relative bounds (coalesced load).
            const int body_wp   = tok_start + t_const; // >= 0
            const bool body_ok  = (body_wp < seqlen);
            const long long body_gt =
                (long long)seq_start + (body_ok ? body_wp : (seqlen > 0 ? seqlen - 1 : 0));
#pragma unroll
            for(int j = 0; j < ELEMS; j++)
            {
                int gf      = feat_start + f_base + j * FG;
                int safe_gf = (gf < dim) ? gf : 0;
                vbuf[j]     = x[(long long)safe_gf * stride_x_dim + body_gt];
            }
#pragma unroll
            for(int j = 0; j < ELEMS; j++)
            {
                int gf = feat_start + f_base + j * FG;
                shmem[(f_base + j * FG) * LDS_PAD + t_const + (KW - 1)] =
                    (body_ok && gf < dim) ? vbuf[j] : (unsigned short)0;
            }

            // Halo / prefix column: within-seq pos = tok_start + hc - (KW-1).
            if(hc < (KW - 1))
            {
                int gf               = feat_start + hf;
                int wp               = tok_start + hc - (KW - 1);
                unsigned short pv    = 0;
                if(wp >= 0 && wp < seqlen)
                {
                    pv = (gf < dim) ? x[(long long)gf * stride_x_dim + (long long)(seq_start + wp)]
                                    : (unsigned short)0;
                }
                else if(wp < 0 && gf < dim && (chunk_idx == 0) && has_initial_state[seq_idx] != 0)
                {
                    int in_coord = cache_indices[seq_idx * sci];
                    int slot     = (KW - 1) + wp; // chunk0 -> tok_start+hc, in [0,KW-2]
                    pv           = conv_states[(long long)in_coord * scs0 + (long long)gf * scs1 +
                                     (long long)slot * scs2];
                }
                shmem[hf * LDS_PAD + hc] = pv;
            }
        }
    }
    __syncthreads();

    // Convert weights to f32
    float wr[KW];
    if(feat_valid)
    {
        wr[0] = bf16_to_f32((unsigned short)(w_raw0));
        wr[1] = bf16_to_f32((unsigned short)(w_raw0 >> 16));
        wr[2] = bf16_to_f32((unsigned short)(w_raw1));
        wr[3] = bf16_to_f32((unsigned short)(w_raw1 >> 16));
    }
    else
    {
#pragma unroll
        for(int k = 0; k < KW; k++)
            wr[k] = 0.f;
    }

    float acc[EPT];
    if(has_bias && feat_valid)
    {
        float b = bf16_to_f32(b_raw);
#pragma unroll
        for(int e = 0; e < EPT; e++)
            acc[e] = b;
    }
    else
    {
#pragma unroll
        for(int e = 0; e < EPT; e++)
            acc[e] = 0.f;
    }

    if(feat_valid)
    {
#pragma unroll
        for(int e = 0; e < EPT; e++)
        {
            int base = feat_local * LDS_PAD + tok_base + e;
#pragma unroll
            for(int k = 0; k < KW; k++)
                acc[e] += wr[k] * bf16_to_f32(shmem[base + k]);
        }
    }

    if(do_silu)
    {
#pragma unroll
        for(int e = 0; e < EPT; e++)
        {
            float exp2v = __builtin_amdgcn_exp2f(acc[e] * (-1.4426950408889634f));
            acc[e] *= __builtin_amdgcn_rcpf(1.f + exp2v);
        }
    }

    unsigned short* out_ptr;
    int out_ts, out_ds, fo;
    if(feat_start + TN <= k_dim)
    {
        out_ptr = q_out;
        out_ts  = stride_q_tok;
        out_ds  = stride_q_dim;
        fo      = 0;
    }
    else if(feat_start >= k_dim && feat_start + TN <= 2 * k_dim)
    {
        out_ptr = k_out;
        out_ts  = stride_k_tok;
        out_ds  = stride_k_dim;
        fo      = k_dim;
    }
    else
    {
        out_ptr = v_out;
        out_ts  = stride_v_tok;
        out_ds  = stride_v_dim;
        fo      = 2 * k_dim;
    }

    const bool store_fast = (feat_start + TN <= dim) && (tok_start + TM - 1 < seqlen);
    if(store_fast)
    {
        constexpr int STORE_PAD = TN + 1;
        __syncthreads(); // conv finished reading shmem
#pragma unroll
        for(int e = 0; e < EPT; e++)
            shmem[(tok_base + e) * STORE_PAD + feat_local] = f32_to_bf16(acc[e]);
        __syncthreads();

        const int sf            = tid & (TN - 1);
        const int tg            = tid >> 6;
        const int of            = (feat_start + sf) - fo;
        const long long sbase   = (long long)(seq_start + tok_start + tg * EPT) * out_ts +
                                (long long)of * out_ds;
#pragma unroll
        for(int e = 0; e < EPT; e++)
            out_ptr[sbase + (long long)e * out_ts] = shmem[(tg * EPT + e) * STORE_PAD + sf];
    }
    else if(feat_valid)
    {
        const int of            = gfeat - fo;
        const long long store_base = (long long)(seq_start + tok_start + tok_base) * out_ts +
                                     (long long)of * out_ds;
        if(tok_start + tok_base + EPT - 1 < seqlen)
        {
#pragma unroll
            for(int e = 0; e < EPT; e++)
                out_ptr[store_base + (long long)e * out_ts] = f32_to_bf16(acc[e]);
        }
        else
        {
#pragma unroll
            for(int e = 0; e < EPT; e++)
                if(tok_start + tok_base + e < seqlen)
                    out_ptr[store_base + (long long)e * out_ts] = f32_to_bf16(acc[e]);
        }
    }

    // conv_state writeback (chunk 0): store the sequence's last KW-1 tokens.
    if(chunk_idx == 0)
    {
        const int slot = tok_group; // 0..3, only < KW-1 used
        if(slot < (KW - 1) && gfeat < dim)
        {
            const int in_coord = cache_indices[seq_idx * sci];
            const int pos_x    = seqlen - (KW - 1) + slot;
            float val;
            if(pos_x >= 0)
            {
                val = bf16_to_f32(
                    x[(long long)gfeat * stride_x_dim + (long long)(seq_start + pos_x)]);
            }
            else if(has_initial_state[seq_idx] != 0)
            {
                int src = slot + seqlen; // seqlen < KW-1 edge case
                val     = bf16_to_f32(conv_states[(long long)in_coord * scs0 +
                                              (long long)gfeat * scs1 + (long long)src * scs2]);
            }
            else
            {
                val = 0.f;
            }
            conv_states[(long long)in_coord * scs0 + (long long)gfeat * scs1 +
                        (long long)slot * scs2] = f32_to_bf16(val);
        }
    }
}

void causal_conv1d_fwd_split_qkv_hip_impl(
    aiter_tensor_t x,
    aiter_tensor_t weight,
    aiter_tensor_t bias,
    aiter_tensor_t conv_states,
    aiter_tensor_t cache_indices,
    aiter_tensor_t has_initial_state,
    aiter_tensor_t query_start_loc,
    aiter_tensor_t batch_ptr,
    aiter_tensor_t token_chunk_offset_ptr,
    aiter_tensor_t q,
    aiter_tensor_t k,
    aiter_tensor_t v,
    int64_t k_dim,
    int64_t v_dim,
    int64_t n_programs,
    int64_t block_m,
    bool has_bias,
    bool silu,
    int64_t pad_slot_id)
{
    AITER_CHECK(x.dtype() == AITER_DTYPE_bf16,
                "causal_conv1d HIP kernel requires bfloat16 input.");
    AITER_CHECK(x.dim() == 2, "`x` must be 2-D [dim, cu_seqlen].");
    AITER_CHECK(block_m == 8 || block_m == 16 || block_m == 32 || block_m == 64,
                "`block_m` must be 8, 16, 32, or 64.");
    AITER_CHECK(weight.dtype() == AITER_DTYPE_bf16, "`weight` must be bfloat16.");
    AITER_CHECK(conv_states.dtype() == AITER_DTYPE_bf16, "`conv_states` must be bfloat16.");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 && k.dtype() == AITER_DTYPE_bf16 &&
                    v.dtype() == AITER_DTYPE_bf16,
                "`q`/`k`/`v` outputs must be bfloat16.");
    AITER_CHECK(cache_indices.dtype() == AITER_DTYPE_i32 &&
                    query_start_loc.dtype() == AITER_DTYPE_i32 &&
                    batch_ptr.dtype() == AITER_DTYPE_i32 &&
                    token_chunk_offset_ptr.dtype() == AITER_DTYPE_i32,
                "`cache_indices`/`query_start_loc`/`batch_ptr`/`token_chunk_offset_ptr` "
                "must be int32.");
    AITER_CHECK(has_initial_state.dtype() == AITER_DTYPE_u8,
                "`has_initial_state` must be uint8.");
    if(has_bias)
        AITER_CHECK(bias.dtype() == AITER_DTYPE_bf16, "`bias` must be bfloat16.");

    const int dim       = (int)x.size(0);
    const int cu_seqlen = (int)x.size(1);
    const int BLOCK_N   = TN;
    const int n_feat_blocks = (dim + BLOCK_N - 1) / BLOCK_N;

    if(n_programs == 0)
        return;

    HipDeviceGuard device_guard(x.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    dim3 grid((unsigned)n_programs, (unsigned)n_feat_blocks);
    dim3 block(NT);

    const unsigned short* bias_data =
        has_bias ? (const unsigned short*)bias.data_ptr() : nullptr;

#define CONV1D_ARGS                                                                          \
    (const unsigned short*)x.data_ptr(), (const unsigned short*)weight.data_ptr(),        \
        bias_data, (unsigned short*)q.data_ptr(), (unsigned short*)k.data_ptr(),          \
        (unsigned short*)v.data_ptr(), (unsigned short*)conv_states.data_ptr(),           \
        (const int*)cache_indices.data_ptr(), (const unsigned char*)has_initial_state.data_ptr(), \
        (const int*)query_start_loc.data_ptr(), (const int*)batch_ptr.data_ptr(),         \
        (const int*)token_chunk_offset_ptr.data_ptr(), dim, (int)k_dim, (int)v_dim,       \
        cu_seqlen, (int)x.stride(0), (int)q.stride(0), (int)q.stride(1), (int)k.stride(0), \
        (int)k.stride(1), (int)v.stride(0), (int)v.stride(1), (int)conv_states.stride(0), \
        (int)conv_states.stride(1), (int)conv_states.stride(2), (int)cache_indices.stride(0), \
        has_bias ? 1 : 0, silu ? 1 : 0, (int)pad_slot_id

    if(block_m == 8)
    {
        conv1d_split_qkv_t<8><<<grid, block, 0, stream>>>(CONV1D_ARGS);
    }
    else if(block_m == 16)
    {
        conv1d_split_qkv_t<16><<<grid, block, 0, stream>>>(CONV1D_ARGS);
    }
    else if(block_m == 32)
    {
        conv1d_split_qkv_t<32><<<grid, block, 0, stream>>>(CONV1D_ARGS);
    }
    else
    {
        conv1d_split_qkv_t<64><<<grid, block, 0, stream>>>(CONV1D_ARGS);
    }
#undef CONV1D_ARGS
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace

namespace aiter {

void causal_conv1d_fwd_split_qkv_hip(
    aiter_tensor_t x,
    aiter_tensor_t weight,
    aiter_tensor_t bias,
    aiter_tensor_t conv_states,
    aiter_tensor_t cache_indices,
    aiter_tensor_t has_initial_state,
    aiter_tensor_t query_start_loc,
    aiter_tensor_t batch_ptr,
    aiter_tensor_t token_chunk_offset_ptr,
    aiter_tensor_t q,
    aiter_tensor_t k,
    aiter_tensor_t v,
    int64_t k_dim,
    int64_t v_dim,
    int64_t n_programs,
    int64_t block_m,
    bool has_bias,
    bool silu,
    int64_t pad_slot_id)
{
    causal_conv1d_fwd_split_qkv_hip_impl(x,
                                         weight,
                                         bias,
                                         conv_states,
                                         cache_indices,
                                         has_initial_state,
                                         query_start_loc,
                                         batch_ptr,
                                         token_chunk_offset_ptr,
                                         q,
                                         k,
                                         v,
                                         k_dim,
                                         v_dim,
                                         n_programs,
                                         block_m,
                                         has_bias,
                                         silu,
                                         pad_slot_id);
}

} // namespace aiter

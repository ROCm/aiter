// SPDX-License-Identifier: MIT
#include "online_common.cuh"
#include "reduce.cuh"
namespace sparse_mla_fp8::online {
template <int Splits,
          int HeadRows,
          int Waves,
          int BlockK,
          int QMask,
          int PVMode,
          bool DoubleBuffer,
          bool QBF16,
          bool QUnitScale,
          bool Gradual>
__global__ __launch_bounds__(Waves * 64) void streaming_attention(Params p)
{

    static_assert(DoubleBuffer && !Gradual);
    constexpr int WPH = Waves * 16 / HeadRows, Fragments = BlockK / (WPH * 16), PVTiles = 32 / WPH;
    static_assert((HeadRows == 32 || HeadRows == 64) && (WPH == 1 || WPH == 2));
    static_assert(BlockK == 128 || (BlockK == 64 && (Waves == 4 || Waves == 8)));
    __shared__ __align__(16)
        uint8_t kv_storage[DoubleBuffer ? 2 : 1][(BlockK / 2) * KV_CHUNK_STRIDE];
    __shared__ __align__(16) uint8_t ps[HeadRows * BlockK];
    __shared__ __align__(16) uint8_t qs[QMask ? HeadRows * 128 * __builtin_popcount(QMask) : 1];
    __shared__ float maxs[WPH == 2 ? Waves * 16 : 1], sums[WPH == 2 ? Waves * 16 : 1];
    __shared__ int selected_storage[DoubleBuffer ? 2 : 1][BlockK];
    const int tid = threadIdx.x, lane = tid & 63, wave = tid / 64, head = lane & 15,
              group = lane / 16;
    const auto work = order_ids<Splits, HEAD32_ORDER, 64 / HeadRows>(
        blockIdx.x, blockIdx.y, blockIdx.z, p.queries);
    const int query = work.query, part = work.part, half = wave % WPH, hgroup = wave / WPH;
    const int hbase = work.head * HeadRows + hgroup * 16;
    const int start = (query * p.capacity), end = ((query + 1) * p.capacity);
    const int total_tiles = (end - start + BlockK - 1) / BlockK, quotient = total_tiles / Splits,
              remainder  = total_tiles % Splits;
    const int tile_begin = part * quotient + (part < remainder ? part : remainder),
              tile_count = quotient + (part < remainder ? 1 : 0);
    const int64_t qoff =
        int64_t(query) * p.q_stride_row + int64_t(hbase + head) * p.q_stride_head + group * 32;
    I8 retained[4];
#pragma unroll
    for(int k = 0; k < 4; ++k)
        if(!(QMask & (1 << k)))
            retained[k] = q_load<QBF16, QUnitScale>(p, qoff + k * 128);
    F4 merged[PVTiles]{};
    float merge_max = -INFINITY, merge_sum = 0.f;
    int next_id = -1;
    if constexpr(DoubleBuffer)
    {
        static_assert(BlockK == 128 && HeadRows == 64 && Waves == 8 && QMask == 0 && PVMode <= 2);
        if(tile_count > 0)
        {
            stage_iddirect<Waves, BlockK>(
                p, kv_storage[0], selected_storage[0], start, end, tile_begin);
            asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : : : "memory");
            __syncthreads();
        }
    }
#pragma unroll 1
    for(int iteration = 0; iteration < tile_count; ++iteration)
    {
        const int split = tile_begin + iteration;
        uint8_t* kvs    = kv_storage[DoubleBuffer ? (iteration & 1) : 0];
        int* selected   = selected_storage[DoubleBuffer ? (iteration & 1) : 0];
        if constexpr(DoubleBuffer)
        {
            if(iteration + 1 < tile_count)
                stage_iddirect<Waves, BlockK>(p,
                                              kv_storage[(iteration + 1) & 1],
                                              selected_storage[(iteration + 1) & 1],
                                              start,
                                              end,
                                              split + 1);
        }
        else
        {
            if(tid < BlockK)
            {
                const int at  = start + split * BlockK + tid;
                const int id  = iteration == 0 ? (at < end ? p.indices[at] : -1) : next_id;
                selected[tid] = (id >= 0 && id < p.slots) ? id : -1;
            }
            __syncthreads();
            if(tid < BlockK && iteration + 1 < tile_count)
            {
                const int at = start + (split + 1) * BlockK + tid;
                next_id      = at < end ? p.indices[at] : -1;
            }
            const int id0 = selected[(wave % (BlockK / 16)) * 16 + lane / 4];
            const int id1 =
                (Waves == 4 && BlockK == 128) ? selected[(wave + 4) * 16 + lane / 4] : -1;
#pragma unroll
            for(int i = 0; i < (BlockK / 2) / Waves; ++i)
            {
                const int chunk = wave + i * Waves, r = (chunk % (BlockK / 16)) * 16 + lane / 4;
                const int c  = (chunk / (BlockK / 16)) * 64 + ((lane % 4) * 16 ^ ((r & 3) * 16));
                const int id = (Waves == 4 && BlockK == 128 && (i & 1)) ? id1 : id0;
                if(id >= 0)
                    direct_load(kvs + chunk * KV_CHUNK_STRIDE, p.kv + int64_t(id) * 512 + c);
                else
                    *reinterpret_cast<U4*>(kvs + chunk * KV_CHUNK_STRIDE + lane * 16) = U4{};
            }
            asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : : : "memory");
            __syncthreads();
        }
        F4 score[Fragments]{};
#define TUNE_QK(K)                                                                   \
    qk_step<K, 0, Fragments, BlockK>(                                                \
        kvs,                                                                         \
        half * (BlockK / WPH) + head,                                                \
        group * 32,                                                                  \
        q_fragment<K, QMask, HeadRows, QBF16, QUnitScale>(                           \
            p, retained, qs, qoff, iteration, hgroup * 16 + head, group * 32, half), \
        score)
        TUNE_QK(0);
        TUNE_QK(1);
        TUNE_QK(2);
        TUNE_QK(3);
#undef TUNE_QK
        // Inline MFMA does not expose its result latency to the scheduler.
        // Fence every H32 score operand before non-MFMA consumers; otherwise
        // a different compiler schedule may read it before the 12-cycle delay.
        if constexpr(Fragments == 2)
            asm volatile("s_nop 7\n\ts_nop 3" : "+v"(score[0]), "+v"(score[1]) : : "memory");
        float m           = -INFINITY;
        const float scale = (p.q_scale * p.kv_scale * p.attention_scale) * LOG2E;
#pragma unroll
        for(int n = 0; n < Fragments; ++n)
        {
#pragma unroll
            for(int j = 0; j < 4; ++j)
            {
                const int key  = half * (BlockK / WPH) + n * 16 + group * 4 + j;
                score[n][j]    = selected[key] >= 0 ? score[n][j] * scale : -INFINITY;
                m              = fmaxf(m, score[n][j]);
                const int slot = split * BlockK + key;
            }
        }
        m = fmaxf(m, __shfl_xor(m, 16));
        m = fmaxf(m, __shfl_xor(m, 32));
        if constexpr(WPH == 2)
        {
            if(group == 0)
                maxs[wave * 16 + head] = m;
            __syncthreads();
            m = fmaxf(m, maxs[(wave ^ 1) * 16 + head]);
        }
        const float newmax = fmaxf(merge_max, m);
        const float alpha = merge_max == -INFINITY ? 0.f : domain_exp2<Gradual>(merge_max - newmax);
        merge_max         = newmax;
        m                 = newmax;
        float zz[Fragments / 2]{};
#pragma unroll
        for(int pair = 0; pair < Fragments / 2; ++pair)
        {
#pragma unroll
            for(int j = 0; j < 4; ++j)
            {
                score[2 * pair][j] =
                    m == -INFINITY ? 0.f : domain_exp2<Gradual>(score[2 * pair][j] - m);
                score[2 * pair + 1][j] =
                    m == -INFINITY ? 0.f : domain_exp2<Gradual>(score[2 * pair + 1][j] - m);
                zz[pair] += score[2 * pair][j] + score[2 * pair + 1][j];
            }
            zz[pair] += __shfl_xor(zz[pair], 16);
            zz[pair] += __shfl_xor(zz[pair], 32);
        }
        float local_z = zz[0];
        if constexpr(Fragments >= 4)
            local_z += zz[1];
        if constexpr(Fragments == 8)
            local_z += zz[2] + zz[3];
        if constexpr(WPH == 2)
            if(group == 0)
                sums[wave * 16 + head] = local_z;
// Publish P together with the sum exchange, saving the separate barrier.
#pragma unroll
        for(int n = 0; n < Fragments; ++n)
        {
            const int key = half * (BlockK / WPH) + n * 16 + group * 4;
            uint32_t prob = pack4(score[n] * (1.f / p.p_scale));
            *reinterpret_cast<uint32_t*>(ps + hgroup * 16 * BlockK + p_address<BlockK>(head, key)) =
                prob;
            const int slot = split * BlockK + key;
        }
        float z = local_z;
        if constexpr(WPH == 2)
        {
            __syncthreads();
            const float other = sums[(wave ^ 1) * 16 + head];
            z                 = half == 0 ? local_z + other : other + local_z;
        }
        else
        {
            // Each wave owns sixteen heads; no other wave reads its P region.
            asm volatile("s_waitcnt lgkmcnt(0)" : : : "memory");
        }
        merge_sum = merge_sum * alpha + z;
        if(group == 0 && half == 0 && (BlockK == 128 || (split % 2) == 0))
        {
            const int tile128 = (split * BlockK) / 128;
            const int64_t off = (int64_t(query) * 17 + tile128) * 64 + hbase + head;
        }
        const uint8_t* pvbase = kvs + half * (8 / WPH) * (BlockK / 16) * KV_CHUNK_STRIDE;
        if constexpr(PVMode >= 3)
        {
            static_assert(BlockK == 64 && (PVMode == 3 || PVMode == 4));
            I2 p0 = p32<0>(ps + hgroup * 16 * BlockK, head, group);
            I2 p1 = p32<1>(ps + hgroup * 16 * BlockK, head, group);
            asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(p0), "+v"(p1) : : "memory");
            pv32<0, PVTiles, PVMode == 4>(pvbase, p0, p1, merged, alpha);
        }
        else
        {
            I8 prob = p_fragment<BlockK>(ps + hgroup * 16 * BlockK, head, group * 32);
            asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(prob) : : "memory");
            if constexpr(PVMode == 2)
            {
                static_assert(BlockK == 128);
                I8 first = v_read<0, 128>(pvbase);
                pv_pipeline<0, PVTiles>(pvbase, prob, merged, alpha, first);
            }
            else
                pv<0, PVTiles, BlockK, PVMode>(pvbase, prob, merged, alpha);
        }
        if constexpr(DoubleBuffer)
        {
            // Wait for next KV writes AND this wave's current LDS readers.
            // All waves finish before the old current buffer can be reused.
            asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : : : "memory");
            __syncthreads();
        }
    }
    const float inv = merge_sum > 0 ? p.p_scale * p.kv_scale / merge_sum : 0.f;
#pragma unroll
    for(int tile = 0; tile < PVTiles; ++tile)
    {
        F4 x          = merged[tile] * inv;
        const int col = half * (512 / WPH) + tile * 16 + group * 4;
        if constexpr(Splits == 1)
        {
            const int64_t off = (int64_t(query) * 64 + hbase + head) * 512 + col;

            uint32_t lo, hi;
            asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(lo) : "v"(x[0]), "v"(x[1]));
            asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(hi) : "v"(x[2]), "v"(x[3]));
            *reinterpret_cast<I2*>(reinterpret_cast<uint16_t*>(p.out) + off) = I2{int(lo), int(hi)};
        }
        else
        {
            const int64_t off = ((int64_t(query) * Splits + part) * 64 + hbase + head) * 512 + col;
            partial_store(p.partial_o + off, x);
        }
    }
    if(group == 0 && half == 0)
    {
        const float l = merge_sum > 0 ? (merge_max + log2f(merge_sum)) * LN2 : -INFINITY;
        if constexpr(Splits == 1)
        {
            if(p.lse)
                p.lse[int64_t(query) * 64 + hbase + head] = l;
        }
        else
            p.partial_lse[(int64_t(query) * Splits + part) * 64 + hbase + head] = l;
    }
}

template <int S, bool BF16, bool Unit>
hipError_t streaming_typed(Params p, hipStream_t stream)
{
    hipLaunchKernelGGL((streaming_attention<S, 64, 8, 128, 0, 2, true, BF16, Unit, false>),
                       dim3(p.queries, S, 1),
                       dim3(512),
                       0,
                       stream,
                       p);
    auto status = hipGetLastError();
    if(status != hipSuccess)
        return status;
    if constexpr(S > 1)
        hipLaunchKernelGGL(
            (reduce_splits<S, 1, 16, true, false>), dim3(p.queries, 64, 2), dim3(64), 0, stream, p);
    return hipGetLastError();
}
template <bool BF16, bool Unit>
hipError_t streaming(Params p, hipStream_t stream)
{
    if(p.queries <= 128)
        return streaming_typed<2, BF16, Unit>(p, stream);
    return streaming_typed<1, BF16, Unit>(p, stream);
}
} // namespace sparse_mla_fp8::online
namespace sparse_mla_fp8 {
hipError_t launch_streaming(Params p, hipStream_t stream)
{
    if(!p.bf16_q)
        return online::streaming<false, false>(p, stream);
    if(p.q_scale == 1.f)
        return online::streaming<true, true>(p, stream);
    return online::streaming<true, false>(p, stream);
}
} // namespace sparse_mla_fp8

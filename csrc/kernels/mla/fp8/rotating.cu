// SPDX-License-Identifier: MIT
#include "online_common.cuh"
#include "reduce.cuh"
#define PIPE_MODE 3
namespace sparse_mla_fp8::online {
__device__ __forceinline__ void rotate_stage(Params p, uint8_t* kv, const int* selected)
{
    const int lane = threadIdx.x & 63, wave = threadIdx.x / 64;
    const int id = selected[wave * 16 + lane / 4];
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        const int chunk = wave + i * 8, r = (chunk % 8) * 16 + lane / 4;
        const int col = (chunk / 8) * 64 + ((lane % 4) * 16 ^ ((r & 3) * 16));
        if(id >= 0)
            direct_load(kv + chunk * KV_CHUNK_STRIDE, p.kv + int64_t(id) * 512 + col);
        else
            *reinterpret_cast<U4*>(kv + chunk * KV_CHUNK_STRIDE + lane * 16) = U4{};
    }
}
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
__global__ __launch_bounds__(Waves * 64) void rotating_attention(Params p)
{

    constexpr bool Pipeline    = PIPE_MODE >= 2;
    constexpr bool QAlias      = PIPE_MODE == 1 || PIPE_MODE >= 3 || HeadRows == 64;
    constexpr int Buffers      = Pipeline ? 2 : 1;
    constexpr int InitialTiles = Pipeline ? (QAlias ? 1 : 2) : 0;
    constexpr int StartBuffer  = Pipeline && QAlias ? 1 : 0;
    constexpr int MaxTiles     = (17 + Splits - 1) / Splits;
    constexpr int WPH = Waves * 16 / HeadRows, Fragments = BlockK / (WPH * 16), PVTiles = 32 / WPH;
    static_assert((HeadRows == 32 || HeadRows == 64) && (WPH == 1 || WPH == 2 || WPH == 4));
    static_assert(BlockK == 128 || (BlockK == 64 && (Waves == 4 || Waves == 8)));
    __shared__ __align__(16) uint8_t kv_storage[Buffers][(BlockK / 2) * KV_CHUNK_STRIDE];
    __shared__ __align__(16) uint8_t ps[HeadRows * BlockK];
    __shared__ __align__(16) uint8_t qs[QAlias ? 1 : HeadRows * 512];
    __shared__ float maxs[WPH >= 2 ? Waves * 16 : 1], sums[WPH >= 2 ? Waves * 16 : 1];
    __shared__ int selected_storage[MaxTiles][BlockK];
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
    static_assert(Waves == 8 && BlockK == 128 && QMask == 0 && !DoubleBuffer);
    static_assert(PIPE_MODE >= 1 && PIPE_MODE <= 4);
    // The public serving Call limits each selection row to capacity<=2176 (17 tiles).
    // Load the complete split's IDs once; they stay immutable through the loop.
    for(int i = tid; i < tile_count * BlockK; i += Waves * 64)
    {
        const int at                             = start + tile_begin * BlockK + i;
        const int id                             = at < end ? p.indices[at] : -1;
        selected_storage[i / BlockK][i % BlockK] = (id >= 0 && id < p.slots) ? id : -1;
    }
    __syncthreads();
    if constexpr(Pipeline)
    {
#pragma unroll
        for(int t = 0; t < InitialTiles; ++t)
            if(t < tile_count)
                rotate_stage(p, kv_storage[(t + StartBuffer) & 1], selected_storage[t]);
    }
    // H64 and alias variants use the initially unoccupied KV buffer for Q.
    uint8_t* qstage       = QAlias ? kv_storage[0] : qs;
    constexpr int QChunks = HeadRows / 16 * 8;
#pragma unroll
    for(int chunk = wave; chunk < QChunks; chunk += Waves)
    {
        const int row     = (chunk % (HeadRows / 16)) * 16 + lane / 4;
        const int col     = (chunk / (HeadRows / 16)) * 64 + ((lane % 4) * 16 ^ ((row & 3) * 16));
        const int64_t off = int64_t(query) * p.q_stride_row +
                            int64_t(work.head * HeadRows + row) * p.q_stride_head + col;
        U4 value;
        if constexpr(QBF16)
            value = q_encode::pack16<QUnitScale>(reinterpret_cast<const uint16_t*>(p.q) + off,
                                                 p.q_scale);
        else
            value = *reinterpret_cast<const U4*>(p.q + off);
        *reinterpret_cast<U4*>(qstage + chunk * 1024 + lane * 16) = value;
    }
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : : : "memory");
    __syncthreads();
    I8 retained[4];
#pragma unroll
    for(int k = 0; k < 4; ++k)
    {
        const int col = k * 128 + group * 32, row = hgroup * 16 + head;
        U4 a, b;
        const uint32_t lo =
            uint32_t(reinterpret_cast<uintptr_t>(qstage + kv_address<HeadRows, 0>(row, col)));
        const uint32_t hi =
            uint32_t(reinterpret_cast<uintptr_t>(qstage + kv_address<HeadRows, 0>(row, col + 16)));
        asm volatile("ds_read_b128 %0, %2\n\tds_read_b128 %1, %3"
                     : "=&v"(a), "=&v"(b)
                     : "v"(lo), "v"(hi)
                     : "memory");
#pragma unroll
        for(int j = 0; j < 4; ++j)
        {
            retained[k][j]     = a[j];
            retained[k][j + 4] = b[j];
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)"
                 : "+v"(retained[0]), "+v"(retained[1]), "+v"(retained[2]), "+v"(retained[3])
                 :
                 : "memory");
    if constexpr(QAlias)
        __syncthreads(); // all retained Q reads precede KV reuse
    F4 merged[PVTiles]{};
    float merge_max = -INFINITY, merge_sum = 0.f;
#pragma unroll 1
    for(int iteration = 0; iteration < tile_count; ++iteration)
    {
        const int split     = tile_begin + iteration;
        const int current   = Pipeline ? ((iteration + StartBuffer) & 1) : 0;
        uint8_t* kvs        = kv_storage[current];
        const int* selected = selected_storage[iteration];
        if constexpr(!Pipeline)
        {
            rotate_stage(p, kvs, selected);
            asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : : : "memory");
            __syncthreads();
        }
        if constexpr(Pipeline && PIPE_MODE != 4)
        {
            if(iteration + 1 < tile_count && (QAlias || iteration > 0))
                rotate_stage(p, kv_storage[current ^ 1], selected_storage[iteration + 1]);
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
        if constexpr(Pipeline && PIPE_MODE == 4)
        {
            if(iteration + 1 < tile_count)
                rotate_stage(p, kv_storage[current ^ 1], selected_storage[iteration + 1]);
        }
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
        m = softmax_four_groups<true>(m);
        if constexpr(WPH >= 2)
        {
            if(group == 0)
                maxs[wave * 16 + head] = m;
            __syncthreads();
            if constexpr(WPH == 4)
            {
                const int first = (wave & ~3) * 16 + head;
                m               = fmaxf(fmaxf(maxs[first], maxs[first + 16]),
                          fmaxf(maxs[first + 32], maxs[first + 48]));
            }
            else
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
            zz[pair] = softmax_four_groups<false>(zz[pair]);
        }
        float local_z = zz[0];
        if constexpr(Fragments >= 4)
            local_z += zz[1];
        if constexpr(Fragments == 8)
            local_z += zz[2] + zz[3];
        if constexpr(WPH >= 2)
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
        if constexpr(WPH >= 2)
        {
            __syncthreads();
            if constexpr(WPH == 4)
            {
                // Same tree as two pairs inside each original WPH2 wave,
                // followed by the original cross-wave sum: (z0+z1)+(z2+z3).
                const int first = (wave & ~3) * 16 + head;
                z = (sums[first] + sums[first + 16]) + (sums[first + 32] + sums[first + 48]);
            }
            else
            {
                const float other = sums[(wave ^ 1) * 16 + head];
                z                 = half == 0 ? local_z + other : other + local_z;
            }
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
        if(iteration + 1 < tile_count)
        {
            // Publish incoming KV, finish all current LDS readers, and protect
            // P/max/sum before another wave begins the following iteration.
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

template <int S, int H, bool BF16, bool Unit>
hipError_t rotating_typed(Params p, hipStream_t stream)
{
    hipLaunchKernelGGL((rotating_attention<S, H, 8, 128, 0, 2, false, BF16, Unit, false>),
                       dim3(p.queries, S, 64 / H),
                       dim3(512),
                       0,
                       stream,
                       p);
    auto status = hipGetLastError();
    if(status != hipSuccess)
        return status;
    hipLaunchKernelGGL(
        (reduce_splits<S, 1, 16, true, false>), dim3(p.queries, 64, 2), dim3(64), 0, stream, p);
    return hipGetLastError();
}
template <bool BF16, bool Unit>
hipError_t rotating(Params p, hipStream_t stream)
{
    if(p.queries <= 21)
        return rotating_typed<6, 32, BF16, Unit>(p, stream);
    if(p.queries <= 23)
        return rotating_typed<4, 32, BF16, Unit>(p, stream);
    if(p.queries <= 42)
        return rotating_typed<6, 64, BF16, Unit>(p, stream);
    return rotating_typed<4, 64, BF16, Unit>(p, stream);
}
} // namespace sparse_mla_fp8::online
namespace sparse_mla_fp8 {
hipError_t launch_rotating(Params p, hipStream_t stream)
{
    if(!p.bf16_q)
        return online::rotating<false, false>(p, stream);
    if(p.q_scale == 1.f)
        return online::rotating<true, true>(p, stream);
    return online::rotating<true, false>(p, stream);
}
} // namespace sparse_mla_fp8

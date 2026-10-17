// SPDX-License-Identifier: MIT
#include "online_common.cuh"
#include "reduce.cuh"
#define PF_BUFFERS 2
#define QRAW_MODE 1
#define PACKED_MUL 1
#define REDUCER_FAST_EXP 1
#define LSE_STRIDE 9
#define LSE_VECTOR 0
namespace sparse_mla_fp8::online {
template <int Splits, int Mode, int Cols, bool SingleHead = false, bool OutputFP8 = true>
__global__ __launch_bounds__(64) void reduce_compact(Params p)
{
    constexpr int V = Cols / 16;
    static_assert(!SingleHead || (Cols == 16 || Cols == 32));
    int lane = threadIdx.x, head = SingleHead ? int(blockIdx.y) : int(blockIdx.y) * 16 + lane / 4;
    int col     = SingleHead ? int(blockIdx.z) * (256 * V) + lane * 4
                             : int(blockIdx.z) * Cols + (lane % 4) * 4;
    int64_t row = int64_t(blockIdx.x) * 64 + head;
    float accum[V][4]{};
    float z = 0.f, m = -INFINITY;
    if constexpr(Mode == 0)
    {
#pragma unroll
        for(int s = 0; s < Splits; ++s)
        {
            float l     = p.partial_lse[(int64_t(blockIdx.x) * 64 + head) * LSE_STRIDE + s];
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
                                    v * (SingleHead ? 256 : 16));
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    accum[v][j] = fmaf(x[j], beta, accum[v][j] * alpha);
            }
        }
    }
    else if constexpr(Mode == 1)
    {
        float ls[Splits];
#if LSE_VECTOR
        static_assert(Splits == 9 && LSE_STRIDE == 16);
        const float* lp = p.partial_lse + (int64_t(blockIdx.x) * 64 + head) * LSE_STRIDE;
        const F4 a = *reinterpret_cast<const F4*>(lp), b = *reinterpret_cast<const F4*>(lp + 4);
#pragma unroll
        for(int j = 0; j < 4; ++j)
        {
            ls[j]     = a[j];
            ls[4 + j] = b[j];
        }
        ls[8] = lp[8];
#endif
#pragma unroll
        for(int s = 0; s < Splits; ++s)
        {
#if !LSE_VECTOR
            ls[s] = p.partial_lse[(int64_t(blockIdx.x) * 64 + head) * LSE_STRIDE + s];
#endif
            m = fmaxf(m, ls[s]);
        }
#if REDUCER_FAST_EXP
        // A single head makes every LSE and this decision wave-uniform.
        // For a finite spread <=80, every exp2 argument lies in [-116,0].
        // All other inputs retain the original gradual-underflow implementation.
        float min_l = INFINITY;
#pragma unroll
        for(int s = 0; s < Splits; ++s)
            min_l = fminf(min_l, ls[s]);
        if(__builtin_amdgcn_readfirstlane(int(SingleHead && m - min_l <= 80.f)))
        {
#pragma unroll
            for(int s = 0; s < Splits; ++s)
            {
                float w = __builtin_amdgcn_exp2f((ls[s] - m) * LOG2E);
                z += w;
#pragma unroll
                for(int v = 0; v < V; ++v)
                {
                    F4 x = partial_load(p.partial_o +
                                        ((int64_t(blockIdx.x) * Splits + s) * 64 + head) * 512 +
                                        col + v * (SingleHead ? 256 : 16));
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                        x[j] = w > 0 ? x[j] : 0.f;
                    using Pair = float __attribute__((ext_vector_type(2)));
                    Pair factor{w, w}, lo, hi;
                    Pair xlo{x[0], x[1]}, xhi{x[2], x[3]}, alo{accum[v][0], accum[v][1]},
                        ahi{accum[v][2], accum[v][3]};
                    asm("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(lo) : "v"(xlo), "v"(factor), "v"(alo));
                    asm("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(hi) : "v"(xhi), "v"(factor), "v"(ahi));
                    accum[v][0] = lo[0];
                    accum[v][1] = lo[1];
                    accum[v][2] = hi[0];
                    accum[v][3] = hi[1];
                }
            }
        }
        else
#endif
        {
#pragma unroll
            for(int s = 0; s < Splits; ++s)
            {
                float w = m == -INFINITY ? 0.f : exp2f((ls[s] - m) * LOG2E);
                z += w;
#pragma unroll
                for(int v = 0; v < V; ++v)
                {
                    F4 x = partial_load(p.partial_o +
                                        ((int64_t(blockIdx.x) * Splits + s) * 64 + head) * 512 +
                                        col + v * (SingleHead ? 256 : 16));
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                        x[j] = w > 0 ? x[j] : 0.f;
                    using Pair = float __attribute__((ext_vector_type(2)));
                    Pair factor{w, w}, lo, hi;
                    Pair xlo{x[0], x[1]}, xhi{x[2], x[3]}, alo{accum[v][0], accum[v][1]},
                        ahi{accum[v][2], accum[v][3]};
                    asm("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(lo) : "v"(xlo), "v"(factor), "v"(alo));
                    asm("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(hi) : "v"(xhi), "v"(factor), "v"(ahi));
                    accum[v][0] = lo[0];
                    accum[v][1] = lo[1];
                    accum[v][2] = hi[0];
                    accum[v][3] = hi[1];
                }
            }
        }
    }
    if constexpr(Mode >= 2)
    {
#pragma unroll
        for(int s = 0; s < Splits; ++s)
            m = fmaxf(m, p.partial_lse[(int64_t(blockIdx.x) * 64 + head) * LSE_STRIDE + s]);
#pragma unroll(Mode == 3 ? 4 : 1)
        for(int s = 0; s < Splits; ++s)
        {
            float l = p.partial_lse[(int64_t(blockIdx.x) * 64 + head) * LSE_STRIDE + s];
            float w = m == -INFINITY ? 0.f : exp2f((l - m) * LOG2E);
            z += w;
#pragma unroll
            for(int v = 0; v < V; ++v)
            {
                F4 x = partial_load(p.partial_o +
                                    ((int64_t(blockIdx.x) * Splits + s) * 64 + head) * 512 + col +
                                    v * (SingleHead ? 256 : 16));
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
#if PACKED_MUL
        using Pair = float __attribute__((ext_vector_type(2)));
        Pair lo, hi, factor{inv, inv}, alo{accum[v][0], accum[v][1]}, ahi{accum[v][2], accum[v][3]};
        asm("v_pk_mul_f32 %0, %1, %2" : "=v"(lo) : "v"(alo), "v"(factor));
        asm("v_pk_mul_f32 %0, %1, %2" : "=v"(hi) : "v"(ahi), "v"(factor));
        x = F4{lo[0], lo[1], hi[0], hi[1]};
#else
        for(int j = 0; j < 4; ++j)
            x[j] = accum[v][j] * inv;
#endif
        for(int j = 0; j < 4; ++j) {}
        if constexpr(OutputFP8)
            *reinterpret_cast<uint32_t*>(p.out + row * 512 + col + v * (SingleHead ? 256 : 16)) =
                pack4(x);
        else
        {
            uint32_t lo, hi;
            asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(lo) : "v"(x[0]), "v"(x[1]));
            asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(hi) : "v"(x[2]), "v"(x[3]));
            *reinterpret_cast<I2*>(reinterpret_cast<uint16_t*>(p.out) + row * 512 + col +
                                   v * (SingleHead ? 256 : 16)) = I2{int(lo), int(hi)};
        }
    }
    if(p.lse && blockIdx.z == 0 && (SingleHead ? lane == 0 : lane % 4 == 0))
        p.lse[row] = z > 0 ? m + log2f(z) * LN2 : -INFINITY;
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
          bool Gradual,
          int FixedCapacity,
          bool Compact>
__global__ __launch_bounds__(Waves * 64) void prefetch_attention(Params p)
{

    constexpr int WPH = Waves * 16 / HeadRows, Fragments = BlockK / (WPH * 16), PVTiles = 32 / WPH;
    static_assert((HeadRows == 32 || HeadRows == 64) && (WPH == 1 || WPH == 2 || WPH == 4));
    static_assert(BlockK == 128 || (BlockK == 64 && (Waves == 4 || Waves == 8)));
    __shared__ __align__(16) uint8_t kv_storage[PF_BUFFERS][(BlockK / 2) * KV_CHUNK_STRIDE];
    __shared__ __align__(16) uint8_t ps[HeadRows * BlockK];
    __shared__ __align__(16) uint8_t qs[HeadRows * 512];
    __shared__ float maxs[WPH >= 2 ? Waves * 16 : 1], sums[WPH >= 2 ? Waves * 16 : 1];
    __shared__ int selected_storage[PF_BUFFERS][BlockK];
    const int tid = threadIdx.x, lane = tid & 63, wave = tid / 64, head = lane & 15,
              group = lane / 16;
    const auto work = order_ids<Splits, HEAD32_ORDER, 64 / HeadRows>(
        blockIdx.x, blockIdx.y, blockIdx.z, Compact ? 8 : p.queries);
    const int query = work.query, part = work.part, half = wave % WPH, hgroup = wave / WPH;
    const int hbase       = work.head * HeadRows + hgroup * 16;
    const int start       = FixedCapacity ? query * FixedCapacity : (query * p.capacity),
              end         = FixedCapacity ? start + FixedCapacity : ((query + 1) * p.capacity);
    const int total_tiles = (end - start + BlockK - 1) / BlockK, quotient = total_tiles / Splits,
              remainder  = total_tiles % Splits;
    const int tile_begin = part * quotient + (part < remainder ? part : remainder),
              tile_count = quotient + (part < remainder ? 1 : 0);
    const int64_t qoff =
        int64_t(query) * p.q_stride_row + int64_t(hbase + head) * p.q_stride_head + group * 32;
    // Fixed S9/H32/W8 has at most two K128 tiles for capacity <=2176.
    // Publish selected IDs once, issue the initial KV loads, then encode Q
    // into an independent LDS allocation while the KV requests are in flight.
    static_assert(Splits >= 9 && Splits <= 17 && HeadRows == 32 && Waves == 8 && BlockK == 128 &&
                  QMask == 0 && !DoubleBuffer);
    static_assert(PF_BUFFERS == 1 || PF_BUFFERS == 2);
    int next_id           = -1;
    constexpr int QChunks = HeadRows / 16 * 8;
    static_assert(HeadRows == 32 && Waves == 8 && QChunks / Waves == 2);
    q_encode::Packed4 qraw_lo[2], qraw_hi[2], qpacked[2];
    if constexpr(QBF16 && QUnitScale)
    {
#pragma unroll
        for(int chunk = wave; chunk < QChunks; chunk += Waves)
        {
            const int row = (chunk % (HeadRows / 16)) * 16 + lane / 4;
            const int col = (chunk / (HeadRows / 16)) * 64 + ((lane % 4) * 16 ^ ((row & 3) * 16));
            const int64_t off = int64_t(query) * p.q_stride_row +
                                int64_t(work.head * HeadRows + row) * p.q_stride_head + col;
            const uint16_t* source = reinterpret_cast<const uint16_t*>(p.q) + off;
            qraw_lo[chunk / Waves] = *reinterpret_cast<const q_encode::Packed4*>(source);
            qraw_hi[chunk / Waves] = *reinterpret_cast<const q_encode::Packed4*>(source + 8);
        }
    }
    if(tid < BlockK)
    {
#pragma unroll
        for(int buffer = 0; buffer < PF_BUFFERS; ++buffer)
        {
            const int at                  = start + (tile_begin + buffer) * BlockK + tid;
            const int id                  = buffer < tile_count && at < end ? p.indices[at] : -1;
            selected_storage[buffer][tid] = (id >= 0 && id < p.slots) ? id : -1;
        }
        if constexpr(PF_BUFFERS == 1)
            if(tile_count > 1)
            {
                const int at = start + (tile_begin + 1) * BlockK + tid;
                next_id      = at < end ? p.indices[at] : -1;
            }
    }
    if constexpr(QBF16 && QUnitScale)
    {
        // Consume raw operands before any KV request. Compiler dependency and
        // memory fences require Q loads to finish here without a vmcnt(N) guess.
        asm volatile(""
                     : "+v"(qraw_lo[0]), "+v"(qraw_hi[0]), "+v"(qraw_lo[1]), "+v"(qraw_hi[1])
                     :
                     : "memory");
        if constexpr(QRAW_MODE == 2)
        {
            qpacked[0] = q_encode::pack_loaded16(qraw_lo[0], qraw_hi[0]);
            qpacked[1] = q_encode::pack_loaded16(qraw_lo[1], qraw_hi[1]);
            asm volatile("" : "+v"(qpacked[0]), "+v"(qpacked[1]) : : "memory");
        }
    }
    __syncthreads();
#pragma unroll
    for(int buffer = 0; buffer < PF_BUFFERS; ++buffer)
        if(buffer < tile_count)
        {
            const int id = selected_storage[buffer][wave * 16 + lane / 4];
#pragma unroll
            for(int i = 0; i < (BlockK / 2) / Waves; ++i)
            {
                const int chunk = wave + i * Waves, r = (chunk % (BlockK / 16)) * 16 + lane / 4;
                const int col = (chunk / (BlockK / 16)) * 64 + ((lane % 4) * 16 ^ ((r & 3) * 16));
                if(id >= 0)
                    direct_load(kv_storage[buffer] + chunk * KV_CHUNK_STRIDE,
                                p.kv + int64_t(id) * 512 + col);
                else
                    *reinterpret_cast<U4*>(kv_storage[buffer] + chunk * KV_CHUNK_STRIDE +
                                           lane * 16) = U4{};
            }
        }
    uint8_t* qstage = qs;
    if constexpr(QBF16 && QUnitScale)
    {
        // Define the conversion inputs after KV issue so conversion cannot be
        // hoisted across the volatile direct-load operations.
        if constexpr(QRAW_MODE == 1)
            asm volatile(""
                         : "+v"(qraw_lo[0]), "+v"(qraw_hi[0]), "+v"(qraw_lo[1]), "+v"(qraw_hi[1])
                         :
                         : "memory");
        else
            asm volatile("" : "+v"(qpacked[0]), "+v"(qpacked[1]) : : "memory");
    }
#pragma unroll
    for(int chunk = wave; chunk < QChunks; chunk += Waves)
    {
        const int row     = (chunk % (HeadRows / 16)) * 16 + lane / 4;
        const int col     = (chunk / (HeadRows / 16)) * 64 + ((lane % 4) * 16 ^ ((row & 3) * 16));
        const int64_t off = int64_t(query) * p.q_stride_row +
                            int64_t(work.head * HeadRows + row) * p.q_stride_head + col;
        U4 value;
        if constexpr(QBF16 && QUnitScale)
        {
            if constexpr(QRAW_MODE == 1)
                value = q_encode::pack_loaded16(qraw_lo[chunk / Waves], qraw_hi[chunk / Waves]);
            else
                value = qpacked[chunk / Waves];
        }
        else if constexpr(QBF16)
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
    F4 merged[PVTiles]{};
    float merge_max = -INFINITY, merge_sum = 0.f;
#pragma unroll 1
    for(int iteration = 0; iteration < tile_count; ++iteration)
    {
        const int split = tile_begin + iteration;
        uint8_t* kvs    = kv_storage[PF_BUFFERS == 2 ? iteration : 0];
        int* selected   = selected_storage[PF_BUFFERS == 2 ? iteration : 0];
        if(iteration >= PF_BUFFERS)
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
        if constexpr(PF_BUFFERS == 2)
            if(iteration + 1 < tile_count)
            {
                // Every wave must finish reading P/max/sum before their reuse.
                asm volatile("s_waitcnt lgkmcnt(0)" : : : "memory");
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
        else if constexpr(Compact)
            p.partial_lse[(int64_t(query) * 64 + hbase + head) * 9 + part] = l;
        else
            p.partial_lse[(int64_t(query) * Splits + part) * 64 + hbase + head] = l;
    }
}

template <bool BF16, bool Unit, int Capacity, bool Compact>
hipError_t launch_prefetch_typed(Params p, hipStream_t stream)
{
    hipLaunchKernelGGL(
        (prefetch_attention<9, 32, 8, 128, 0, 2, false, BF16, Unit, false, Capacity, Compact>),
        dim3(p.queries, 9, 2),
        dim3(512),
        0,
        stream,
        p);
    auto status = hipGetLastError();
    if(status != hipSuccess)
        return status;
    if constexpr(Compact)
        hipLaunchKernelGGL((reduce_compact<9, 1, 16, true, false>),
                           dim3(p.queries, 64, 2),
                           dim3(64),
                           0,
                           stream,
                           p);
    else
        hipLaunchKernelGGL(
            (reduce_splits<9, 1, 16, true, false>), dim3(p.queries, 64, 2), dim3(64), 0, stream, p);
    return hipGetLastError();
}
template <bool BF16, bool Unit>
hipError_t prefetch(Params p, hipStream_t stream)
{
    if(p.queries == 8)
    {
        if(p.capacity == 2112)
            return launch_prefetch_typed<BF16, Unit, 2112, true>(p, stream);
        return launch_prefetch_typed<BF16, Unit, 0, true>(p, stream);
    }
    if(p.capacity == 2112)
        return launch_prefetch_typed<BF16, Unit, 2112, false>(p, stream);
    return launch_prefetch_typed<BF16, Unit, 0, false>(p, stream);
}
} // namespace sparse_mla_fp8::online
namespace sparse_mla_fp8 {
hipError_t launch_prefetch(Params p, hipStream_t stream)
{
    if(!p.bf16_q)
        return online::prefetch<false, false>(p, stream);
    if(p.q_scale == 1.f)
        return online::prefetch<true, true>(p, stream);
    return online::prefetch<true, false>(p, stream);
}
} // namespace sparse_mla_fp8

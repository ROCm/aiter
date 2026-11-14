// NVFP4 MoE GEMM kernels for gfx950: shared pieces of the one kernel template.
//
// Orientation: MFMA src A = weights (16 output features), src B = activations (16 tokens), so
// a C lane holds 4 consecutive output features of one token.
//
// Weight decode: NVFP4 exact. fp4 -> f16 via v_cvt_scalef32_pk_f16_fp4, then times the E4M3
// scale in f16 (exact: <= 6 significant bits). Activations f16; with Cfg::inbf16 (fallback
// path, any sort) the kernel reads the bf16 activations and casts them to f16 itself.
//
// Weight tile layout: [E][NTILE][KT][64 lanes][16 B]; in tile i, K-tile j, lane l = 16 g + c holds
// row 16 i + c, k in [128 j + 32 g, +32) (raw packed fp4 bytes).
// Weight scale layout (GS layout, GS = 2 GPS K-tiles per group, GS 2 .. 8): [E][NTILE][KTP /
// GPS][64 lanes][4 GPS B], KTP = ceil(KT / 2) K-tile pairs (odd KT: the last half zero); the
// dword of pair p is dword p % GPS of group p / GPS. GS 2 is the pair layout, GS 6 the SG
// layout. One layout per weight tensor (scale_gs, aiter/jit/utils/moe_nvfp4_common.py).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <type_traits>
#include <utility>

#define DEV static __device__ __forceinline__

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v3i __attribute__((ext_vector_type(3)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

struct Params
{
    const uint8_t* x;    // activations: f16 [rows, K] (bf16 [T, K] with Cfg::inbf16)
    const uint8_t* w;    // weight tiles
    const uint8_t* w_sc; // weight scales (GS layout)
    const float* inv_gs; // stage1: [E][2] (gate, up); stage2: [E]
    const int* sorted_ids;
    const float* sorted_w;
    const int* sorted_eids;
    const int* num_valid;
    // stage1: x_inter f16 [T*topk, I]; stage2: f32 [T, H] (Cfg::outbf16: bf16, Cfg::reduce:
    // f32 [T*topk, H])
    uint8_t* out;
    int T, topk;
    int K;      // reduction length
    int ntiles; // weight row tiles per expert (stage1: 2I/16, stage2: H/16)
    int n_out;  // stage1: I, stage2: H
};

// ---------------------------------------------------------------- kernel config
// X placement (kernel name token): per-wave registers ("xreg"), an LDS ring fed through VGPRs
// ("xstream"), the whole block in LDS ("xres"), an LDS ring fed by DMA ("xdma", with an
// LDS-DMA weight ring).
enum XPlace
{
    X_REG,
    X_STREAM,
    X_RESIDENT,
    X_DMA
};

// One kernel instance. Its name is the config (kernel_name in aiter/jit/utils/moe_nvfp4_common.py);
// gen_instances.py writes the Cfg literal of each name.
struct Cfg
{
    bool s1           = false; // stage 1: gate/up + SwiGLU-OAI; else stage 2: down + output atomics
    int x             = X_REG; // XPlace
    int m_tiles       = 0;     // 16-row m tiles per block (BM = 16 m_tiles)
    int n_tiles       = 0;     // weight tiles per wave (stage 1: half gate, half up)
    int waves         = 1;     // waves along N
    int ksplit        = 1;     // waves along K (partial sums reduced in LDS)
    int chunk_k_tiles = 1;     // K-tiles per weight chunk (X_DMA: 1)
    int wbuf          = 1;     // weight buffers: register chunks, or (X_DMA) LDS K-tile stages
    int xbuf          = 0;     // X LDS ring stages (X_STREAM, X_DMA)
    int k_tiles       = 0;     // compile-time K-tiles (0: K from Params)
    int gs            = 2;     // K-tiles per weight scale group (the tensor's layout)
    bool upfront      = false; // scale groups of the wave's K slice loaded up front
    bool tight        = false; // sched_barrier after every chunk load and compute
    bool inbf16       = false; // X is bf16 (cast in the kernel)
    bool xcdremap     = false; // XCD remap of the block order (+ row-contiguous X DMA lanes)
    bool staged       = false; // stage 2: route weights early, all C reads before the atomics
    int occ           = 1;     // waves per EU hint
    bool outbf16      = false; // stage 2: bf16 output atomics (else f32 workspace atomics)
    bool reduce       = false; // stage 2: f32 [T*topk, H] rows, plain stores (moe_nvfp4_reduce)
};

// Field combinations the kernels implement. Names are checked by parse_kernel
// (aiter/jit/utils/moe_nvfp4_common.py); this also covers a Cfg written by hand.
constexpr bool cfg_valid(const Cfg& c)
{
    // tile sizes: at least one 16-row m tile, one weight tile per wave, one K-tile per chunk
    if(c.m_tiles < 1 || c.n_tiles < 1 || c.chunk_k_tiles < 1)
        return false;
    // counts: at least one wave, K slice, weight buffer, wave per EU
    if(c.waves < 1 || c.ksplit < 1 || c.wbuf < 1 || c.occ < 1)
        return false;
    // the DMA X ring is stage 1 only, the resident X block stage 2 only
    if((c.x == X_DMA && !c.s1) || (c.x == X_RESIDENT && c.s1))
        return false;
    // X LDS ring stages exactly for the ring placements (xstream, xdma)
    if((c.xbuf > 0) != (c.x == X_STREAM || c.x == X_DMA))
        return false;
    // scale group: 2, 4, 6 or 8 K-tiles (the scale layouts)
    if(c.gs < 2 || c.gs > 8 || c.gs % 2)
        return false;
    // bf16 output only in stage 2 (stage 1 writes f16 x_inter)
    if(c.outbf16 && c.s1)
        return false;
    // reduce rows: stage 2, f32
    if(c.reduce && (c.s1 || c.outbf16))
        return false;
    return true;
}

// Chunks loaded ahead of the one being computed: register X loads the next chunk into a free
// buffer before the compute (pre), LDS X refills the buffer just computed (post), the DMA ring
// refills the stage just read (mid).
constexpr int lookahead(const Cfg& c) { return c.x == X_REG ? c.wbuf - 1 : c.wbuf; }

// Runtime K walks run in units of the fewest chunks that hold whole scale groups (K must be
// whole units) and in cycles of the fewest units that are also a multiple of the buffer count,
// so buffer slots and scale rows are compile-time.
constexpr int reg_unit(const Cfg& c)
{
    int u = 1;
    while(u * c.chunk_k_tiles % c.gs)
        u++;
    return u;
}
constexpr int reg_cycle(const Cfg& c)
{
    int m = reg_unit(c);
    while(m % c.wbuf)
        m += reg_unit(c);
    return m;
}

// ---------------------------------------------------------------- inline asm
// The wave-uniform v is needed in an SGPR here: issues its load before a following branch.
DEV void require_sgpr(int v) { asm volatile("" ::"s"(v)); }
DEV void lds_wait() { asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory"); }
// Own LDS ops done, then s_barrier (no vmcnt wait, unlike __syncthreads).
DEV void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
// At most N own vector memory ops in flight and own LDS ops done, then s_barrier.
template <int N>
DEV void vmem_lds_barrier()
{
    asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)\n\ts_barrier" ::"n"(N) : "memory");
}
// Compiler-only scheduling fence: no instruction is moved across it.
DEV void sched_fence() { __builtin_amdgcn_sched_barrier(0); }
// A wave-uniform load offset in its own SGPR. The offsets of a load group are computed first,
// then sched_fence(), then the loads: otherwise the compiler recomputes each offset into a
// reused SGPR just before its load, and a SALU write to the SGPR of a load still queued stalls
// the wave.
DEV void own_sgpr(int& v) { asm volatile("" : "+s"(v)); }

// Buffer-resource loads: wave-uniform byte offsets go to SGPR soffset and the per-lane part to
// a 32-bit voffset. Bounded resource: loads at byte offsets >= n return 0 (still issued).
typedef __amdgpu_buffer_rsrc_t buffer_desc_t;
DEV buffer_desc_t make_buffer_desc(const void* p, int n = -1)
{
    return __builtin_amdgcn_make_buffer_rsrc((void*)p, (short)0, n, 0x00020000);
}
// Weights and weight scales load nt (aux 2): best measured in both stages. X loads stay
// plain: X lines are reused by the WGs of all N groups.
DEV v4i load_w16(buffer_desc_t d, int voff, int soff)
{
    return __builtin_amdgcn_raw_buffer_load_b128(d, voff, soff, 2);
}
DEV v4i load_x16(buffer_desc_t d, int voff, int soff)
{
    return __builtin_amdgcn_raw_buffer_load_b128(d, voff, soff, 0);
}

// Compile-time int argument.
template <int N>
using IntConst = std::integral_constant<int, N>;
// f(IntConst<0>{}), .., f(IntConst<N - 1>{}) in order
template <int N, class F>
DEV void static_for(F&& f)
{
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(IntConst<I>{}), ...); }(
        std::make_integer_sequence<int, N>{});
}

// ---------------------------------------------------------------- block start
// N block of the WG when Cfg::xcdremap or Cfg::staged (set by the kernel).
__shared__ int lds_n_block;
// The hardware dispatches the WGs of a grid round-robin over the XCDs: WG i runs on XCD
// i % NUM_XCDS.
constexpr int NUM_XCDS_LOG2 = 3;
constexpr int NUM_XCDS      = 1 << NUM_XCDS_LOG2;
// AITER's gemm1 XCD remap (mxfp4_gemm1.py). The valid tiles are m_blocks (valid M blocks at run
// time) x n_blocks. Each XCD gets a contiguous run of tiles / NUM_XCDS tile ids (the first
// tiles % NUM_XCDS XCDs one more), so WGs that share weights share an L2. Tile ids are walked in
// groups of GM M blocks x all N blocks, M fastest. Returns the M block (-1: idle WG) and stores
// the N block to lds_n_block (every thread writes the same value and reads its own write).
template <int GM>
DEV int xcd_remap(int BM, int nvalid)
{
    const int n_blocks = gridDim.x;
    const int wg_id    = blockIdx.x + blockIdx.y * n_blocks;
    const int m_blocks = (nvalid + BM - 1) / BM;
    const int tiles    = m_blocks * n_blocks;
    if(wg_id >= tiles)
        return -1;
    const int current_xcd = wg_id & (NUM_XCDS - 1);
    const int per_xcd     = tiles >> NUM_XCDS_LOG2;
    const int extra       = tiles & (NUM_XCDS - 1);
    const int xcd_slot    = wg_id >> NUM_XCDS_LOG2;
    const int tile        = current_xcd * per_xcd + min(current_xcd, extra) + xcd_slot;
    const int group_tiles = GM * n_blocks;
    const int group_m0    = tile / group_tiles * GM;
    const int group_m     = min(m_blocks - group_m0, GM);
    const int in_group    = tile % group_tiles;
    lds_n_block           = in_group / group_m;
    return group_m0 + in_group % group_m;
}

// Number of leading 16-row tiles of block mblk holding a valid row (WG-uniform); MT if the
// valid tiles are not a prefix. Both sorts put the valid rows of an expert before its padding.
template <int MT>
DEV int active_tiles(const Params& p, int mblk)
{
    const int lane_col = threadIdx.x & 15;
    uint32_t tile_mask = 0;
#pragma unroll
    for(int m = 0; m < MT; m++)
    {
        const int s = p.sorted_ids[mblk * 16 * MT + m * 16 + lane_col];
        tile_mask |= (__builtin_amdgcn_ballot_w64((s & 0xffffff) < p.T) != 0 ? 1u : 0u) << m;
    }
    const int n_active = __builtin_popcount(tile_mask);
    return __builtin_amdgcn_readfirstlane(tile_mask == (1u << n_active) - 1 ? n_active : MT);
}

// f(IntConst<MA>{}) with MA = n_active for 1 <= n_active < MT, else MT.
template <int MT, class F>
DEV void dispatch_active_tiles(int n_active, F&& f)
{
    if constexpr(MT >= 2)
        if(n_active == 1)
            return f(IntConst<1>{});
    if constexpr(MT >= 3)
        if(n_active == 2)
            return f(IntConst<2>{});
    if constexpr(MT >= 4)
        if(n_active == 3)
            return f(IntConst<3>{});
    f(IntConst<MT>{});
}

// ---------------------------------------------------------------- decode + MFMA
DEV v2h cvt_fp4x2_f16(uint32_t d, int byte)
{
    switch(byte)
    {
    case 0: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, 1.0f, 0);
    case 1: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, 1.0f, 1);
    case 2: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, 1.0f, 2);
    default: return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(d, 1.0f, 3);
    }
}
// the 8 fp4 of d as f16, times sc (one block scale in both halves)
DEV v8h decode_fp4x8(uint32_t d, v2h sc)
{
    const v2h e0 = cvt_fp4x2_f16(d, 0) * sc;
    const v2h e1 = cvt_fp4x2_f16(d, 1) * sc;
    const v2h e2 = cvt_fp4x2_f16(d, 2) * sc;
    const v2h e3 = cvt_fp4x2_f16(d, 3) * sc;
    return v8h{e0[0], e0[1], e1[0], e1[1], e2[0], e2[1], e3[0], e3[1]};
}

// 8 bf16 -> 8 f16, rounded to nearest even (the fused sort's cast and the inbf16 kernels').
DEV v4i cast_bf16x8_f16(v4i bf16x8)
{
    v4i f16x8;
#pragma unroll
    for(int k = 0; k < 4; k++)
    {
        const float lo = __uint_as_float((uint32_t)bf16x8[k] << 16);
        const float hi = __uint_as_float((uint32_t)bf16x8[k] & 0xffff0000u);
        const v2h pair = {(_Float16)lo, (_Float16)hi};
        f16x8[k]       = __builtin_bit_cast(int, pair);
    }
    return f16x8;
}

// One K-tile: the NT weight tiles [JB, JE) (16 B per lane each), decoded with half PAR of
// their E4M3 scale pairs sc, times the MA X tiles (4 x 16 B per lane each). X_CAST: x holds
// bf16, cast to f16 in place before its first use (weight tile 0).
template <int MT, int NT, int MA, int PAR, bool X_CAST = false, int JB = 0, int JE = NT>
DEV void mma_ktile(const v4i (&w)[NT], const uint32_t (&sc)[NT], v8h (&x)[MT][4],
                   v4f (&acc)[NT][MT])
{
#pragma unroll
    for(int j = JB; j < JE; j++)
    {
        const v2h sp = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(sc[j], 1.0f, PAR != 0);
        const v2h s0 = {sp[0], sp[0]};
        const v2h s1 = {sp[1], sp[1]};
#pragma unroll
        for(int q = 0; q < 4; q++)
        {
            const v8h wv = decode_fp4x8(w[j][q], q < 2 ? s0 : s1);
#pragma unroll
            for(int m = 0; m < MA; m++)
            {
                if constexpr(X_CAST)
                    if(j == 0)
                        x[m][q] = __builtin_bit_cast(
                            v8h, cast_bf16x8_f16(__builtin_bit_cast(v4i, x[m][q])));
                acc[j][m] = __builtin_amdgcn_mfma_f32_16x16x32_f16(wv, x[m][q], acc[j][m], 0, 0, 0);
            }
        }
    }
}

// GPT-OSS style SwiGLU: gate clamped above at LIMIT, up clamped to [-LIMIT, LIMIT],
// sigmoid(ALPHA * gate).
constexpr float SWIGLU_OAI_LIMIT = 7.0f;
constexpr float SWIGLU_OAI_ALPHA = 1.702f;

DEV float swiglu_oai(float gate, float up)
{
    gate = fminf(gate, SWIGLU_OAI_LIMIT);
    up   = fminf(fmaxf(up, -SWIGLU_OAI_LIMIT), SWIGLU_OAI_LIMIT);
    return gate / (1.0f + __expf(-SWIGLU_OAI_ALPHA * gate)) * (up + 1.0f);
}

// ---------------------------------------------------------------- wave, routing, sources
struct WaveIds
{
    int lane, wave_n, wave_k, lane_group, lane_col, nblock;
};
template <Cfg C>
DEV WaveIds wave_ids()
{
    WaveIds w;
    w.lane         = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    w.wave_k       = wave % C.ksplit;
    w.wave_n       = wave / C.ksplit;
    w.lane_group   = w.lane >> 4;
    w.lane_col     = w.lane & 15;
    w.nblock       = C.xcdremap || C.staged ? lds_n_block : (int)blockIdx.x;
    return w;
}

// Weight tile j of the wave: stage 1 pairs gate tile g with up tile g of the same columns.
template <Cfg C>
DEV int weight_tile(const Params& p, const WaveIds& w, int j)
{
    if constexpr(C.s1)
    {
        constexpr int GATE_TILES = C.n_tiles / 2;
        const int gate           = (w.nblock * C.waves + w.wave_n) * GATE_TILES + j % GATE_TILES;
        return j < GATE_TILES ? gate : p.n_out / 16 + gate;
    }
    else
        return (w.nblock * C.waves + w.wave_n) * C.n_tiles + j;
}

// Routing of the lane's rows 16 m + lane_col of block mblk.
template <int MT>
struct Routing
{
    int token[MT], slot[MT];
    bool valid[MT];
};
// The lane's sorted ids (load) and their routing (decode): split so the loads can be issued
// early and decoded after other loads.
template <int MT>
DEV void read_sorted_ids(int (&sid)[MT], const Params& p, int mblk, int lane_col)
{
#pragma unroll
    for(int m = 0; m < MT; m++)
        sid[m] = p.sorted_ids[mblk * 16 * MT + m * 16 + lane_col];
}
template <int MT>
DEV Routing<MT> routing_of(const int (&sid)[MT], const Params& p)
{
    Routing<MT> r;
#pragma unroll
    for(int m = 0; m < MT; m++)
    {
        r.token[m] = sid[m] & 0xffffff;
        r.slot[m]  = sid[m] >> 24;
        r.valid[m] = r.token[m] < p.T;
    }
    return r;
}
template <int MT>
DEV Routing<MT> read_routing(const Params& p, int mblk, int lane_col)
{
    int sid[MT];
    read_sorted_ids<MT>(sid, p, mblk, lane_col);
    return routing_of<MT>(sid, p);
}
// X row of sorted id s (stage 1: the token, stage 2: its x_inter row), x_rows when padding
template <bool S1>
DEV int x_row(const Params& p, int s, int x_rows)
{
    const int t = s & 0xffffff;
    return t < p.T ? (S1 ? t : t * p.topk + (s >> 24)) : x_rows;
}

// Weight / scale sources of one wave: expert descriptors and wave-uniform tile byte offsets.
template <int NT>
struct Sources
{
    buffer_desc_t w_desc, s_desc, x_desc;
    int w_off[NT], s_off[NT];
};
template <Cfg C>
DEV Sources<C.n_tiles> make_sources(const Params& p, int e, const WaveIds& w, int k_tiles)
{
    const int ktp = (k_tiles + 1) / 2;
    const int x_rows = C.s1 ? p.T : p.T * p.topk;
    Sources<C.n_tiles> s;
    s.w_desc = make_buffer_desc(p.w + (size_t)e * p.ntiles * k_tiles * 1024);
    s.s_desc = make_buffer_desc(p.w_sc + (size_t)e * p.ntiles * ktp * 256);
    // padding rows load from row x_rows (out of range: zeros)
    s.x_desc = make_buffer_desc(p.x, x_rows * p.K * 2);
#pragma unroll
    for(int j = 0; j < C.n_tiles; j++)
    {
        const int tile = weight_tile<C>(p, w, j);
        s.w_off[j]     = tile * k_tiles * 1024;
        s.s_off[j]     = tile * ktp * 256;
    }
    return s;
}

// ---------------------------------------------------------------- scale groups
// Byte offsets (own SGPRs) of scale group g of the NT tiles.
template <int GPS, int NT>
DEV void scale_group_offsets(int (&off)[NT], const Sources<NT>& s, int g)
{
#pragma unroll
    for(int j = 0; j < NT; j++)
    {
        off[j] = s.s_off[j] + g * GPS * 256;
        own_sgpr(off[j]);
    }
}
// The scale group at byte offsets offs (scale_group_offsets) of the NT tiles (one b32 .. b128
// per tile and lane) into pairs p0 .. p0 + GPS - 1.
template <int GPS, int NT, int N_PAIRS>
DEV void
load_scale_group(uint32_t (&dst)[N_PAIRS][NT], const Sources<NT>& s, const int (&offs)[NT], int p0)
{
    const int lane_off = (threadIdx.x & 63) * 4 * GPS;
#pragma unroll
    for(int j = 0; j < NT; j++)
    {
        const int off = offs[j];
        if constexpr(GPS == 1)
            dst[p0][j] = __builtin_amdgcn_raw_buffer_load_b32(s.s_desc, lane_off, off, 2);
        else if constexpr(GPS == 2)
        {
            const v2i t = __builtin_amdgcn_raw_buffer_load_b64(s.s_desc, lane_off, off, 2);
            dst[p0][j]     = t[0];
            dst[p0 + 1][j] = t[1];
        }
        else if constexpr(GPS == 3)
        {
            const v3i t = __builtin_amdgcn_raw_buffer_load_b96(s.s_desc, lane_off, off, 2);
#pragma unroll
            for(int q = 0; q < 3; q++)
                dst[p0 + q][j] = t[q];
        }
        else
        {
            static_assert(GPS == 4, "GS 2, 4, 6 or 8");
            const v4i t = __builtin_amdgcn_raw_buffer_load_b128(s.s_desc, lane_off, off, 2);
#pragma unroll
            for(int q = 0; q < 4; q++)
                dst[p0 + q][j] = t[q];
        }
    }
}

// ---------------------------------------------------------------- K steps
// Steps 0 .. n - 1 of one wave in cycles of CYC steps: f(c, IntConst<POS>, IntConst<LEFT>) with
// POS = c % CYC (compile-time) and LEFT the number of steps after c when c is in the peeled end
// (compile-time), -1 in the rolled loop (then at least MIN_END steps follow). HEAD cycles are
// peeled before the loop; the peeled end starts at a cycle boundary and is the fewest steps
// >= MIN_END (one compiled end per remainder n % CYC, a multiple of U). n >= HEAD CYC +
// MIN_END. A compile-time K walk is one peeled end (CYC = U = MIN_END = n).
template <int CYC, int U, int HEAD, int MIN_END, class F>
DEV void k_steps(int n, F&& f)
{
    static_assert(CYC % U == 0, "cycle of whole units");
    auto end_steps = [](int rem) {
        return rem + (MIN_END > rem ? (MIN_END - rem + CYC - 1) / CYC * CYC : 0);
    };
    static_for<HEAD * CYC>([&](auto s) { f(s.value, IntConst<s.value % CYC>{}, IntConst<-1>{}); });
    const int rem = CYC == U ? 0 : n % CYC;
#pragma unroll 1
    for(int c = HEAD * CYC; c < n - end_steps(rem); c += CYC)
        static_for<CYC>([&](auto s) { f(c + s.value, s, IntConst<-1>{}); });
    static_for<CYC / U>([&](auto v) {
        constexpr int END = end_steps(v.value * U);
        if(rem == v.value * U)
        {
            const int c0 = n - END;
            static_for<END>([&](auto s) {
                f(c0 + s.value, IntConst<s.value % CYC>{}, IntConst<END - 1 - s.value>{});
            });
        }
    });
}

// ---------------------------------------------------------------- K reduction, epilogues
// ksplit > 1: partial sums of waves wave_k > 0 go through LDS (partial, (ksplit - 1) waves
// n_tiles m_tiles 64 v4f) to wave_k == 0. Returns false for the waves that are done.
template <Cfg C, int MA>
DEV bool k_reduce(v4f (&acc)[C.n_tiles][C.m_tiles], const WaveIds& w, v4f* partial)
{
    if constexpr(C.ksplit == 1)
        return true;
    auto at = [&](int k, int j, int m) {
        return ((((k - 1) * C.waves + w.wave_n) * C.n_tiles + j) * C.m_tiles + m) * 64 + w.lane;
    };
    if(w.wave_k > 0)
    {
#pragma unroll
        for(int j = 0; j < C.n_tiles; j++)
#pragma unroll
            for(int m = 0; m < MA; m++)
                partial[at(w.wave_k, j, m)] = acc[j][m];
    }
    __syncthreads();
    if(w.wave_k > 0)
        return false;
    for(int k = 1; k < C.ksplit; k++)
#pragma unroll
        for(int j = 0; j < C.n_tiles; j++)
#pragma unroll
            for(int m = 0; m < MA; m++)
                acc[j][m] += partial[at(k, j, m)];
    return true;
}

// Stage 1: SwiGLU-OAI of the gate / up tiles, f16 x_inter rows.
template <Cfg C, int MA>
DEV void s1_epilogue(const Params& p,
                     const v4f (&acc)[C.n_tiles][C.m_tiles],
                     int e,
                     const Routing<C.m_tiles>& rt,
                     const WaveIds& w)
{
    constexpr int GATE_TILES = C.n_tiles / 2;
    const float inv_gs_gate  = p.inv_gs[2 * e];
    const float inv_gs_up    = p.inv_gs[2 * e + 1];
    const int I              = p.n_out;
    const int n_begin        = (w.nblock * C.waves + w.wave_n) * 16 * GATE_TILES;
#pragma unroll
    for(int m = 0; m < MA; m++)
    {
        const size_t out_row = (size_t)rt.token[m] * p.topk + rt.slot[m];
        float o[GATE_TILES][4];
#pragma unroll
        for(int g = 0; g < GATE_TILES; g++)
#pragma unroll
            for(int v = 0; v < 4; v++)
                o[g][v] = swiglu_oai(acc[g][m][v] * inv_gs_gate,
                                     acc[GATE_TILES + g][m][v] * inv_gs_up);
        if(rt.valid[m])
        {
#pragma unroll
            for(int g = 0; g < GATE_TILES; g++)
            {
                const v2h a0 = {(_Float16)o[g][0], (_Float16)o[g][1]};
                const v2h a1 = {(_Float16)o[g][2], (_Float16)o[g][3]};
                const v2i st = {__builtin_bit_cast(int, a0), __builtin_bit_cast(int, a1)};
                *(v2i*)(p.out + (out_row * I + n_begin + 16 * g + 4 * w.lane_group) * 2) = st;
            }
        }
    }
}

// Stage 2 C tile row stride in floats (row pad 4 as AITER gemm1: 0 loses 0.2-2.4 us at TP4)
constexpr int c_row_floats(int nt) { return 16 * nt + 4; }
// LDS v4i of the stage-2 C tiles of all waves
constexpr int c_tile_v4(const Cfg& c)
{
    return c.s1 ? 0 : (c.waves * 16 * c.m_tiles * c_row_floats(c.n_tiles) + 3) / 4;
}

// Stage 2 route weights of the lane's rows, times the expert's inverse global scale
template <int MT, int MA>
DEV void read_route_weights(float (&rw)[MT], const Params& p, int e, int mblk, int lane_col)
{
    const float inv_gs = p.inv_gs[e];
#pragma unroll
    for(int m = 0; m < MA; m++)
        rw[m] = p.sorted_w[mblk * 16 * MT + m * 16 + lane_col] * inv_gs;
}

// Stage 2: acc * route weight -> the wave's LDS C rows (c_lds), wave-local transpose, then bf16 x2
// atomics (C.outbf16), two f32 atomics per column pair (f32 workspace of f16 output) or (C.reduce)
// an f32 pair store to the (token, slot) row of the [T*topk, H] reduce workspace: one
// instruction covers 64 consecutive column pairs of 64 / (8 NT) rows instead of 16 rows x 16 B.
// C.staged: every C value and row token of the wave is read before the first atomic. Other NT
// (not 1, 2, 4 or 8k): an instruction covers the largest divisor of the 8 NT column pairs that
// is <= 64, whole rows; the lanes past the last whole row store nothing.
constexpr int s2_inst_pairs(int col_pairs)
{
    int d = col_pairs < 64 ? col_pairs : 64;
    while(col_pairs % d)
        d--;
    return d;
}
template <Cfg C, int MA>
DEV void s2_epilogue(const Params& p,
                     const v4f (&acc)[C.n_tiles][C.m_tiles],
                     const float (&rw)[C.m_tiles],
                     const Routing<C.m_tiles>& rt,
                     float* c_lds,
                     const WaveIds& w)
{
    constexpr int NT         = C.n_tiles;
    constexpr int C_STRIDE   = c_row_floats(NT);
    constexpr int COL_PAIRS  = 8 * NT;
    constexpr int INST_PAIRS = s2_inst_pairs(COL_PAIRS);    // column pairs per instruction
    constexpr int INST_ROWS  = 64 / INST_PAIRS;             // rows per instruction
    constexpr bool IDLE      = INST_ROWS * INST_PAIRS < 64; // lanes past the last whole row
    constexpr int ROW_STEPS  = 16 * MA / INST_ROWS;
    constexpr int COL_STEPS  = COL_PAIRS / INST_PAIRS;
    static_assert(16 % INST_ROWS == 0, "the rows of an instruction are in one 16-row tile");
    // token (C.reduce: token * topk + slot) of row 16 m + lane_col, -1: padding (no store)
    int out_token[C.m_tiles];
#pragma unroll
    for(int m = 0; m < MA; m++)
    {
        if constexpr(C.reduce)
            out_token[m] = rt.valid[m] ? rt.token[m] * p.topk + rt.slot[m] : -1;
        else
            out_token[m] = rt.valid[m] ? rt.token[m] : -1;
#pragma unroll
        for(int j = 0; j < NT; j++)
            *(v4f*)(c_lds + (16 * m + w.lane_col) * C_STRIDE + 16 * j + 4 * w.lane_group) =
                acc[j][m] * rw[m];
    }
    __syncwarp(); // lanes read C rows written by other lanes of the wave
    const int col_pair = w.lane % INST_PAIRS;
    const int n_begin  = (w.nblock * C.waves + w.wave_n) * NT * 16;
    // packed: bf16 x2 (C.outbf16), else the f32 pair
    using Packed = std::conditional_t<C.outbf16, int, float2>;
    int row_token[ROW_STEPS];
    Packed packed[ROW_STEPS][COL_STEPS];
    auto read_row = [&](int r) {
        // IDLE: the lanes past the last whole row read that row and store nothing
        const int row = r * INST_ROWS + (IDLE ? min(w.lane / INST_PAIRS, INST_ROWS - 1)
                                              : w.lane / INST_PAIRS);
        row_token[r]  = __shfl(out_token[(r * INST_ROWS) / 16], row % 16);
        if constexpr(IDLE)
            if(w.lane >= INST_ROWS * INST_PAIRS)
                row_token[r] = -1;
#pragma unroll
        for(int h = 0; h < COL_STEPS; h++)
        {
            const int col  = 2 * (col_pair + h * INST_PAIRS);
            const float2 f = *(const float2*)(c_lds + row * C_STRIDE + col);
            if constexpr(C.outbf16)
                packed[r][h] = __builtin_bit_cast(int, v2bf{(__bf16)f.x, (__bf16)f.y});
            else
                packed[r][h] = f;
        }
    };
    auto store_row = [&](int r) {
        if(row_token[r] >= 0)
#pragma unroll
            for(int h = 0; h < COL_STEPS; h++)
            {
                const uint32_t col = n_begin + 2 * (col_pair + h * INST_PAIRS);
                const uint32_t idx = (uint32_t)row_token[r] * p.n_out + col;
                if constexpr(C.reduce)
                    *(float2*)(p.out + ((size_t)row_token[r] * p.n_out + col) * 4) = packed[r][h];
                else if constexpr(C.outbf16)
                    __builtin_amdgcn_global_atomic_fadd_v2bf16(
                        (v2bf*)(p.out + idx * 2), __builtin_bit_cast(v2bf, packed[r][h]));
                else
                {
                    float* dst = (float*)(p.out + idx * 4);
                    __builtin_amdgcn_global_atomic_fadd_f32(dst, packed[r][h].x);
                    __builtin_amdgcn_global_atomic_fadd_f32(dst + 1, packed[r][h].y);
                }
            }
    };
    if constexpr(C.staged)
    {
#pragma unroll
        for(int r = 0; r < ROW_STEPS; r++)
            read_row(r);
        sched_fence();
#pragma unroll
        for(int r = 0; r < ROW_STEPS; r++)
            store_row(r);
    }
    else
    {
#pragma unroll
        for(int r = 0; r < ROW_STEPS; r++)
        {
            read_row(r);
            store_row(r);
        }
    }
}

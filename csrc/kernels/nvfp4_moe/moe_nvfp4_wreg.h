// NVFP4 MoE GEMM: register weight ring with X in registers (X_REG), streamed through an LDS
// ring (X_STREAM) or resident in LDS (X_RESIDENT).
#pragma once
#include "moe_nvfp4_common.h"

// One weight chunk in registers: KC K-tiles of NT weight tiles, and the X tiles of its K-tiles
// (loaded with the chunk for X_REG, read from LDS before the compute otherwise).
template <int MT, int NT, int KC>
struct Chunk
{
    v4i w[KC][NT];
    v8h x[KC][MT][4];
};

// X in LDS (X_STREAM, X_RESIDENT): a stage of XK K-tiles holds XK MT rows (i, m) of 4 units of
// 64 lanes x 16 B; unit q of row (i, m) holds, for lane l, X[row(m, l % 16)][128 i + 32 (l / 16)
// + 8 q, +8), the MFMA B operand layout. Global loads are row-coalesced: load q of a row is the 8
// X rows 8 (q % 2) + l % 8 x 128 B (column half q / 2), scattered to the B-operand order by the
// LDS store. The XK MA active rows of a stage go to the waves as in AITER gemm2: UW = min(WN, 4)
// waves share the 4 loads of a row (wave quarter wave_n % UW), WN / UW row groups split the rows
// (rows wave_n / UW + WN / UW t). bf16 X (stage-1 X_STREAM _xbf16): cast to f16 at the LDS store.
template <Cfg C, int MA, int XK>
struct XLdsUnits
{
    static constexpr int MT       = C.mt;
    static constexpr int UW       = C.wn < 4 ? C.wn : 4;
    static constexpr int GROUPS   = C.wn / UW;
    static constexpr int ROWS     = (XK * MT + GROUPS - 1) / GROUPS; // rows per wave (at MA MT)
    static constexpr int PER_WAVE = ROWS * (4 / UW);
    static constexpr int STAGE_V4 = XK * MT * 4 * 64; // LDS v4i of a stage
    static_assert(C.wn % UW == 0 && 4 % UW == 0, "X LDS loads: waves");
    int row_off[MT][2]; // X byte offset of the lane's row 8 h + l % 8 of m tile m, + its column
    int lds_lane;       // LDS v4i index of the lane within a row
    int quarter, group; // wave_n % UW, wave_n / UW

    __device__ __forceinline__ XLdsUnits(const Params& p, int mblk, const WaveIds& w)
    {
        const int x_rows = C.s1 ? p.T : p.T * p.topk;
#pragma unroll
        for(int m = 0; m < MT; m++)
#pragma unroll
            for(int h = 0; h < 2; h++)
            {
                const int s   = p.sorted_ids[mblk * 16 * MT + m * 16 + 8 * h + (w.lane & 7)];
                row_off[m][h] = x_row<C.s1>(p, s, x_rows) * p.K * 2 + 16 * (w.lane >> 3);
            }
        lds_lane = (w.lane >> 3 & 3) * 64 + (w.lane & 7) + 16 * (w.lane >> 5);
        // WN <= 4: one row group, rows compile-time (wave_n < WN is unknown to the compiler)
        quarter = C.wn == UW ? w.wave_n : w.wave_n % UW;
        group   = GROUPS == 1 ? 0 : w.wave_n / UW;
    }
    // load j of the wave: row t = j / (4 / UW) of the wave, quarter a = j % (4 / UW)
    struct Unit
    {
        int i, m, q;
        bool on;
    };
    __device__ __forceinline__ Unit unit(int j) const
    {
        const int r = group + GROUPS * (j / (4 / UW));
        return {r / MA, r % MA, quarter + UW * (j % (4 / UW)), r < XK * MA};
    }
    // the stage from K-tile kt0 -> r
    __device__ __forceinline__ void load(v4i (&r)[PER_WAVE], buffer_desc_t x_desc, int kt0) const
    {
#pragma unroll
        for(int j = 0; j < PER_WAVE; j++)
            if(const Unit a = unit(j); a.on)
            {
                // compile-time indices only (a runtime index would put row_off in memory)
                int voff = 0;
#pragma unroll
                for(int m = 0; m < MT; m++)
#pragma unroll
                    for(int h = 0; h < 2; h++)
                        voff = a.m == m && (a.q & 1) == h ? row_off[m][h] : voff;
                r[j] = load_x16(x_desc, voff, (kt0 + a.i) * 256 + 128 * (a.q >> 1));
            }
    }
    __device__ __forceinline__ void store(const v4i (&r)[PER_WAVE], v4i* stage) const
    {
#pragma unroll
        for(int j = 0; j < PER_WAVE; j++)
            if(const Unit a = unit(j); a.on)
                stage[(a.i * MT + a.m) * 256 + 8 * (a.q & 1) + 32 * (a.q >> 1) + lds_lane] =
                    C.xbf16 ? cast_bf16x8_f16(r[j]) : r[j];
    }
};

// No X in LDS (X_REG)
struct NoXUnits
{
    static constexpr int STAGE_V4 = 0, PER_WAVE = 1;
    __device__ __forceinline__ NoXUnits(const Params&, int, const WaveIds&) {}
    __device__ __forceinline__ void load(v4i (&)[1], buffer_desc_t, int) const {}
    __device__ __forceinline__ void store(const v4i (&)[1], v4i*) const {}
};

// Number of distinct group phases (kt_begin % GS) of the WK wave slices of SLICE K-tiles.
constexpr bool slice_phase_used(int wk, int slice, int gs, int ph)
{
    for(int k = 0; k < wk; k++)
        if(k * slice % gs == ph)
            return true;
    return false;
}
constexpr int slice_phases(int wk, int slice, int gs)
{
    int n = 0;
    for(int ph = 0; ph < gs; ph++)
        n += slice_phase_used(wk, slice, gs, ph);
    return n;
}

// Runtime K: one buffer at GS 2 peels the first cycle (its first chunk is loaded ahead of the
// rolled loop).
constexpr int reg_head(const Cfg& c) { return c.wbuf == 1 && c.gs == 2 ? 1 : 0; }
// Runtime K is whole units of this many chunks: whole cycles for X_STREAM (one compiled end),
// whole scale groups otherwise (one compiled end per remainder).
constexpr int reg_k_unit(const Cfg& c) { return c.x == X_STREAM ? reg_cycle(c) : reg_unit(c); }

// LDS v4i of the X ring / block (X_STREAM, X_RESIDENT)
constexpr int x_lds_v4(const Cfg& c)
{
    return c.x == X_STREAM ? c.xbuf * c.kc * c.mt * 4 * 64
                           : c.x == X_RESIDENT ? c.kt * c.mt * 4 * 64 : 0;
}

// Waves per EU hint: X_RESIDENT blocks of 2+ m tiles in WGs of at most 4 waves ask for the waves
// per SIMD their LDS allows (160 KB per CU), at most 3 (168 VGPRs); left free, the scheduler
// trades occupancy for ILP (GLM 32x256 KT 4: 184 VGPRs, 2 waves per SIMD, +10%). 1-tile blocks
// (small T, latency-bound) and 8-wave WGs keep their hint (16x128: +5-9% at T 1-2 with the
// hint; 32x512 waves8 would spill).
constexpr int waves_hint(const Cfg& c)
{
    if(c.x != X_RESIDENT || c.mt < 2 || c.wn > 4)
        return c.occ;
    const int lds   = 16 * (x_lds_v4(c) > c_tile_v4(c) ? x_lds_v4(c) : c_tile_v4(c)) + 16;
    const int waves = 163840 / lds * c.wn / 4;
    const int w     = waves < 3 ? waves : 3;
    return w > c.occ ? w : c.occ;
}

// Load offsets in own SGPRs and load groups fenced, except in X_RESIDENT kernels whose weight
// ring, X chunk and accumulators exceed the VGPR cap of their waves per EU hint (there the
// fenced groups spill: 32x256 KT 6 at occupancy 3, 1.08-1.15x).
constexpr bool fence_offsets(const Cfg& c)
{
    const int vgprs = 4 * c.wbuf * c.kc * c.nt + 16 * c.kc * c.mt + 4 * c.nt * c.mt;
    return c.x != X_RESIDENT || vgprs <= 512 / waves_hint(c);
}

// K loop of one wave on the register weight ring. The wave reduces the K slice kt_begin ..
// kt_begin + slice - 1 (slice = K-tiles / WK) in chunks of KC K-tiles; chunk c is issued
// P = lookahead(C) chunks ahead: weights, the scale groups whose first K-tile it holds (unless
// C.up: all groups of the slice loaded up front, before the routing) and, for X_REG, its X.
// Compile-time K (C.kt): one fully unrolled walk per slice start phase kt_begin % GS (the last
// chunk may be shorter); runtime K: cycles of reg_cycle(C) chunks, whole groups per wave.
// Load order: X first with bf16 X (X lands and is cast while the weights are in flight); else
// at GS 2 weights, scales, X; else weights, X, scales.
template <Cfg C, int MA>
DEV void reg_k_loop(const Params& p, int mblk, int e, const WaveIds& w, const Sources<C.nt>& src,
                    Routing<C.mt>& rt, v4f (&acc)[C.nt][C.mt], v4i* x_lds, float (&rw)[C.mt])
{
    constexpr int MT = C.mt, NT = C.nt, KC = C.kc, R = C.wbuf, P = lookahead(C);
    constexpr int GS = C.gs, GPS = GS / 2;
    constexpr bool CT    = C.kt > 0;
    constexpr int SLICE  = CT ? C.kt / C.wk : 0;
    constexpr bool LDS_X = C.x != X_REG;
    constexpr bool CAST  = C.xbf16 && C.x == X_REG;
    static_assert(C.x != X_DMA && (!C.xbf16 || C.x == X_REG || (C.x == X_STREAM && C.s1)),
                  "register ring X placement; bf16 X: X_REG or stage-1 X_STREAM");
    static_assert(!CT || C.kt % C.wk == 0, "compile-time K: whole wave slices");
    static_assert(CT || KC % 2 == 0, "runtime K: whole scale pairs per chunk");
    static_assert(!C.up || (CT && (SLICE % GS == 0 || C.wk == 1)),
                  "up front: compile-time K, slices of whole groups");
    static_assert(!LDS_X || C.wk == 1, "X in LDS: no K split");
    static_assert(C.x != X_RESIDENT || (CT && C.kt % KC == 0), "resident X: whole chunks");
    static_assert(C.x != X_STREAM || C.xbuf >= 2, "X stream: at least 2 LDS stages");
    const int k_tiles  = CT ? C.kt : p.K / 128;
    const int slice    = CT ? SLICE : k_tiles / C.wk;
    const int kt_begin = w.wave_k * slice;

    constexpr int NUP = (SLICE + GS - 1) / GS; // up front: groups of the slice
    uint32_t up[C.up ? NUP * GPS : 1][NT];
    // X_REG: the sorted id loads go first (the X rows gate the first X loads)
    int sid[MT];
    if constexpr(C.x == X_REG)
        read_sorted_ids<MT>(sid, p, mblk, w.lane_col);
    if constexpr(C.up)
    {
        int up_off[NUP][NT];
        static_for<NUP>([&](auto q) {
            scale_group_offsets<GPS>(up_off[q.value], src, kt_begin / GS + q.value);
        });
        if constexpr(fence_offsets(C))
            sched_fence();
        static_for<NUP>([&](auto q) {
            load_scale_group<GPS>(up, src, up_off[q.value], q.value * GPS);
        });
    }
    if constexpr(C.x == X_REG)
        rt = routing_of<MT>(sid, p);
    else
        rt = read_routing<MT>(p, mblk, w.lane_col);
    int x_lane_off[MT]; // X_REG: per-lane X row offsets (padding: out of range, zeros)
    if constexpr(C.x == X_REG)
    {
        const int x_rows = C.s1 ? p.T : p.T * p.topk;
#pragma unroll
        for(int m = 0; m < MT; m++)
        {
            const int row = rt.valid[m] ? (C.s1 ? rt.token[m] : rt.token[m] * p.topk + rt.slot[m])
                                        : x_rows;
            x_lane_off[m] = (row * p.K + 32 * w.lane_group) * 2;
        }
    }
    constexpr int XK = C.x == X_RESIDENT ? C.kt : KC; // K-tiles per X LDS stage
    using Units = std::conditional_t<LDS_X, XLdsUnits<C, MA, XK>, NoXUnits>;
    const Units units(p, mblk, w);
    v4i x_regs[C.x == X_STREAM ? R : 1][Units::PER_WAVE];

    auto walk = [&](auto ph) {
        constexpr int PH   = ph.value;
        constexpr int NC   = CT ? (SLICE + KC - 1) / KC : 0;
        constexpr int CYC  = CT ? NC : reg_cycle(C);
        constexpr int HEAD = CT ? 0 : reg_head(C);
        constexpr int END  = CT ? NC : P; // fewest peeled end steps
        constexpr int NG   = CT ? (PH + SLICE + GS - 1) / GS : 0; // groups of the slice
        constexpr int BANK = CT ? (C.up ? 1 : NG * GPS) : CYC * KC / 2;
        constexpr int NB   = !CT && P > 0 ? 2 : 1; // banks: this cycle's pairs (+ the next's)
        uint32_t bank[NB][BANK][NT];
        Chunk<MT, NT, KC> ring[R];
        int kt0 = kt_begin;
        // a phase-specific copy of kt_begin: otherwise the identical loads of the phase walks
        // are hoisted above the phase branch
        if constexpr(CT && slice_phases(C.wk, SLICE, GS) > 1)
            asm volatile("; phase %1" : "+s"(kt0) : "n"(PH));
        const int g0 = (int)((uint32_t)kt0 / GS);
        auto len     = [](int pos) { return CT && SLICE - pos * KC < KC ? SLICE - pos * KC : KC; };

        // issue chunk c (cycle position POS) into its buffer, its groups into bank B
        auto issue = [&](int c, auto pos, auto b) {
            constexpr int POS = pos.value, LEN = len(POS);
            Chunk<MT, NT, KC>& r = ring[POS % R];
            const int kt         = kt0 + c * KC;
            // f(u, g, p0) for each scale group g of the chunk (slot u, first bank pair p0)
            constexpr int NGU = CT ? (C.up ? 1 : NG) : KC / 2;
            auto groups       = [&](auto f) {
                if constexpr(CT && !C.up)
                    static_for<NG>([&](auto q) {
                        constexpr int FIRST = q.value == 0 ? 0 : q.value * GS - PH;
                        if constexpr(FIRST / KC == POS)
                            f(q.value, g0 + q.value, q.value * GPS);
                    });
                else if constexpr(!CT)
                    static_for<KC / 2>([&](auto i) {
                        constexpr int PAIR = POS * KC / 2 + i.value;
                        if constexpr(PAIR % GPS == 0)
                            f(i.value, g0 + (c * KC + 2 * i.value) / GS, PAIR);
                    });
            };
            // the chunk's wave-uniform byte offsets before its first load (own_sgpr): weights one
            // per K-tile and tile, X one per K-tile (its 16 B units in the immediate offset)
            int w_soff[LEN][NT], x_soff[LEN], g_off[NGU][NT];
#pragma unroll
            for(int i = 0; i < LEN; i++)
            {
#pragma unroll
                for(int j = 0; j < NT; j++)
                {
                    w_soff[i][j] = src.w_off[j] + (kt + i) * 1024;
                    if constexpr(fence_offsets(C))
                        own_sgpr(w_soff[i][j]);
                }
                x_soff[i] = (kt + i) * 256;
                if constexpr(C.x == X_REG)
                    own_sgpr(x_soff[i]);
            }
            groups([&](int u, int g, int) { scale_group_offsets<GPS>(g_off[u], src, g); });
            // not in the runtime K walk at one buffer: there the loads of chunk c + 1 overlap
            // the compute of chunk c (a fence here sits right after that compute)
            if constexpr((CT || R > 1) && fence_offsets(C))
                sched_fence();
            auto load_w = [&]() {
#pragma unroll
                for(int i = 0; i < LEN; i++)
#pragma unroll
                    for(int j = 0; j < NT; j++)
                        r.w[i][j] = load_w16(src.w_desc, w.lane * 16, w_soff[i][j]);
            };
            auto load_x = [&]() {
                if constexpr(C.x == X_REG)
#pragma unroll
                    for(int i = 0; i < LEN; i++)
#pragma unroll
                        for(int m = 0; m < MA; m++)
#pragma unroll
                            for(int q = 0; q < 4; q++)
                                r.x[i][m][q] = __builtin_bit_cast(
                                    v8h, load_x16(src.x_desc, x_lane_off[m] + 16 * q, x_soff[i]));
            };
            auto load_groups = [&]() {
                groups([&](int u, int, int p0) {
                    load_scale_group<GPS>(bank[CT ? 0 : b.value], src, g_off[u], p0);
                });
            };
            if constexpr(CAST)
                load_x(), load_w(), load_groups();
            else if constexpr(GS == 2)
                load_w(), load_groups(), load_x();
            else
                load_w(), load_x(), load_groups();
        };

        // compute chunk c (cycle position POS)
        auto compute = [&](int c, auto pos) {
            constexpr int POS = pos.value, LEN = len(POS);
            Chunk<MT, NT, KC>& r = ring[POS % R];
            if constexpr(LDS_X)
            {
                const v4i* stage = x_lds + (C.x == X_STREAM ? c % C.xbuf * Units::STAGE_V4 : 0);
                constexpr int I0 = C.x == X_RESIDENT ? POS * KC : 0; // stage K-tile of i = 0
#pragma unroll
                for(int i = 0; i < LEN; i++)
#pragma unroll
                    for(int m = 0; m < MA; m++)
#pragma unroll
                        for(int q = 0; q < 4; q++)
                            r.x[i][m][q] = __builtin_bit_cast(
                                v8h, stage[(((I0 + i) * MT + m) * 4 + q) * 64 + w.lane]);
            }
            static_for<LEN>([&](auto i) {
                constexpr int T = PH + POS * KC + i.value; // K-tile in the bank
                uint32_t sc[NT];
#pragma unroll
                for(int j = 0; j < NT; j++)
                    if constexpr(C.up)
                        sc[j] = up[T / 2][j];
                    else
                        sc[j] = bank[0][T / 2][j];
                mma_ktile<MT, NT, MA, T & 1, CAST>(r.w[i.value], sc, r.x[i.value], acc);
            });
        };

        auto step = [&](int c, auto pos, auto left) {
            constexpr int POS = pos.value, LEFT = left.value;
            constexpr bool AHEAD = LEFT < 0 || P <= LEFT; // chunk c + P exists
            constexpr int PA = (POS + P) % CYC, BA = (POS + P) / CYC;
            if constexpr(C.x == X_STREAM && AHEAD)
                units.load(x_regs[PA % R], src.x_desc, (c + P) * KC);
            if constexpr(C.x == X_REG && AHEAD)
            {
                issue(c + P, IntConst<PA>{}, IntConst<BA>{});
                // the next chunk goes out before the compute: fenced in the runtime K cycle
                // walk (else it sinks into the compute), and when tight or with bf16 X; only
                // the compile-time K walk also fences the compute when tight
                if constexpr(C.tight || CAST || !CT)
                    sched_fence();
            }
            compute(c, pos);
            if constexpr(C.tight && CT)
                sched_fence();
            if constexpr(LDS_X && AHEAD)
                issue(c + P, IntConst<PA>{}, IntConst<BA>{});
            if constexpr(C.x == X_STREAM && LEFT != 0)
            {
                units.store(x_regs[(POS + 1) % R], x_lds + (c + 1) % C.xbuf * Units::STAGE_V4);
                lds_barrier();
            }
            if constexpr(NB > 1 && POS == CYC - 1)
#pragma unroll
                for(int i = 0; i < BANK; i++)
#pragma unroll
                    for(int j = 0; j < NT; j++)
                        bank[0][i][j] = bank[1][i][j];
        };

        // prologue: X block (X_RESIDENT), chunks 0 .. P - 1, X stage 0 (X_STREAM)
        if constexpr(C.x == X_RESIDENT)
            units.load(x_regs[0], src.x_desc, 0);
        if constexpr(C.up && C.x == X_REG)
            asm volatile("" ::"v"(x_lane_off[0])); // the first weights wait for the routing
        static_for<P>([&](auto u) {
            if constexpr(!CT || u.value < NC)
            {
                if constexpr(C.x == X_STREAM)
                    units.load(x_regs[u.value % R], src.x_desc, u.value * KC);
                issue(u.value, IntConst<u.value % CYC>{}, IntConst<u.value / CYC>{});
                // X_STREAM: chunk u (X stage first) goes out before chunk u + 1 (else weights
                // are hoisted above the X stage 0 the first barrier waits for)
                if constexpr(C.x == X_REG || C.x == X_STREAM)
                    sched_fence();
            }
        });
        if constexpr(LDS_X)
        {
            units.store(x_regs[0], x_lds);
            lds_barrier();
        }
        // after the prologue: no accumulator is live while its loads are issued
#pragma unroll
        for(int j = 0; j < NT; j++)
#pragma unroll
            for(int m = 0; m < MT; m++)
                acc[j][m] = v4f{0, 0, 0, 0};
        // C.staged: the stage-2 route weights are read before the main loop
        if constexpr(!C.s1 && C.staged)
            read_route_weights<MT, MA>(rw, p, e, mblk, w.lane_col);
        k_steps<CYC, CT ? NC : reg_k_unit(C), HEAD, END>(CT ? NC : slice / KC, step);
    };

    if constexpr(CT)
    {
        const int phase = (int)((uint32_t)kt_begin % GS);
        static_for<GS>([&](auto ph) {
            if constexpr(slice_phase_used(C.wk, SLICE, GS, ph.value))
                if(slice_phases(C.wk, SLICE, GS) == 1 || phase == ph.value)
                    walk(ph);
        });
    }
    else
        walk(IntConst<0>{});
}

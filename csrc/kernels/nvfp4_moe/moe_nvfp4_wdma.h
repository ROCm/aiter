// NVFP4 MoE GEMM: LDS-DMA weight ring with X in an LDS ring fed by DMA (X_DMA, stage 1 only).
//
// Weights and weight scales go global -> LDS by buffer_load ... lds (nt) into a wave-private ring
// of NS = Cfg::wbuf K-tile stages. The DMA is inline asm, so the compiler's waitcnt pass does not
// tie the ds_reads to it; each stage waits with a constant vmcnt (dma_vmcnt). Stage s = K-tile s:
// NT W blocks (1 KB); GS 2 (pair layout): even stages also carry the NT scale pair blocks (256 B,
// one pair per two K-tiles), so the ring is NS / 2 pair slots of NT 2304 B per wave; GS > 2: the
// pair slots hold only W (NT 2048 B) and one group slot of NT KB per wave (GS 4: NT 512 B) takes
// the scale group (one buffer_load_dword{x3,x4} ... lds per tile and group, read once per group).
// Stage c: wait for c, ds_read it, refill its slot with stage c + NS, compute. The gathered X rows
// go by DMA into a WG-shared ring of XS = Cfg::xbuf K-tiles, synced by s_barrier per stage.
// Cfg::xg: row-contiguous X DMA lanes (8 lanes per 128 B line, XOR-swizzled LDS slots), with the
// XCD remap of the kernel. Cfg::xbf16: X(s) goes global -> VGPR where its DMA would be issued;
// stage s - 1, after the W refill and the MFMAs of weight tile 0, waits for it, casts it to f16
// and writes the own blocks of the X ring (the X ring holds f16 as with the f16 X DMA).
#pragma once
#include "moe_nvfp4_common.h"

constexpr int DMA_X_SLOTS = 64; // 16 B slots per X ring block

// buffer resource as 4 SGPRs (asm operand), n bytes (default: no range limit)
DEV v4i dma_desc(const void* p, int n = -1)
{
    const uint64_t a = (uint64_t)p;
    return v4i{(int)(uint32_t)a, (int)((a >> 32) & 0xffff), n, 0x00020000};
}

// LDS bytes per wave of the W ring: NS / 2 pair slots of 2 K-tiles (NT KB of W each, + NT 256 B
// of scale pairs at GS 2), GS > 2: + the scale group slot (NT KB, GS 4: NT / 2 KB)
constexpr int dma_wave_ring_bytes(const Cfg& c)
{
    return c.wbuf / 2 * c.nt * (c.gs > 2 ? 2048 : 2304) +
           (c.gs > 2 ? c.nt * (c.gs == 4 ? 512 : 1024) : 0);
}
// X DMA blocks per wave and stage (4 MT blocks of 1 KB per K-tile over the WN waves)
constexpr int dma_x_blocks(const Cfg& c) { return (4 * c.mt + c.wn - 1) / c.wn; }
// LDS v4i of a kernel: the W rings of all waves, the X ring and a spare X block (WN does not
// divide 4 MT: the waves without a block of their own DMA into it)
constexpr int dma_lds_v4(const Cfg& c)
{
    return (c.wn * dma_wave_ring_bytes(c) + c.mt * 4 * 1024 * c.xbuf +
            ((4 * c.mt) % c.wn ? DMA_X_SLOTS * 16 : 0)) /
           16;
}
// Stages at the end that issue no W or X refill past the last K-tile (even)
constexpr int dma_end_stages(const Cfg& c)
{
    return (c.wbuf > c.xbuf - 1 ? c.wbuf : c.xbuf - 1) + 1 & ~1;
}
// Stage cycle of the K walk: one scale group (GS 2: the stage pair), K-tiles whole cycles
constexpr int dma_cycle(const Cfg& c) { return c.gs; }
// Stages of the peeled end: the whole cycles holding the end stages
constexpr int dma_peeled(const Cfg& c)
{
    return (dma_end_stages(c) + c.gs - 1) / c.gs * c.gs;
}

// LDS DMA of BYTES (16, 12 or 4) per lane to LDS byte address m0 + 16 lane (4 B: m0 + 4 lane;
// 12 B: the 4th dword is skipped). NONTEMP: nontemporal (only 16 B may be plain, for X rows).
// HALF (16 B nt): lanes 0 .. 31 only, a GS-4 scale group block (512 B, 8 B per lane) copied as
// is (there is no 8 B LDS DMA).
#define NVFP4_MOE_DMA_ASM(LOAD, MOD)                                                               \
    asm volatile("s_mov_b32 m0, %3\n\ts_nop 0\n\t" LOAD " %0, %1, %2 offen" MOD " lds"             \
                 ::"v"(voff), "s"(desc), "s"(soff), "s"(m0)                                        \
                 : "memory", "m0")
template <int BYTES, bool NONTEMP = true, bool HALF = false>
DEV void lds_dma(const v4i& desc, int voff, int soff, uint32_t m0)
{
    static_assert((BYTES == 16 || (NONTEMP && (BYTES == 12 || BYTES == 4))) &&
                      (!HALF || (BYTES == 16 && NONTEMP)),
                  "bad lds_dma");
    if constexpr(HALF)
    {
        uint64_t exec_save;
        asm volatile("s_mov_b64 %[sv], exec\n\ts_mov_b32 exec_hi, 0\n\ts_mov_b32 m0, %[m0]\n\t"
                     "s_nop 0\n\tbuffer_load_dwordx4 %[v], %[d], %[so] offen nt lds\n\t"
                     "s_mov_b64 exec, %[sv]"
                     : [sv] "=&s"(exec_save)
                     : [v] "v"(voff), [d] "s"(desc), [so] "s"(soff), [m0] "s"(m0)
                     : "memory", "m0");
    }
    else if constexpr(BYTES == 16 && NONTEMP)
        NVFP4_MOE_DMA_ASM("buffer_load_dwordx4", " nt");
    else if constexpr(BYTES == 16)
        NVFP4_MOE_DMA_ASM("buffer_load_dwordx4", "");
    else if constexpr(BYTES == 12)
        NVFP4_MOE_DMA_ASM("buffer_load_dwordx3", " nt");
    else
        NVFP4_MOE_DMA_ASM("buffer_load_dword", " nt");
}
#undef NVFP4_MOE_DMA_ASM

// a mod n in [0, n), also for negative a
constexpr int nonneg_mod(int a, int n) { return (a % n + n) % n; }

// vmcnt once iterations <= last are issued: the DMAs issued after X(r) and (unless x_only) W(r).
// Iteration k issues X(k + XS - 1) (X_DMAS per wave) and then W(k + NS) (NT blocks, + NT scale
// blocks if W(k + NS) starts a scale group; GS 2: every even stage). Stages relative to K-tiles
// - NS (steady state: r < 0), X(s) issued for s < NS and W(s) for k < 0; K-tiles % GS == 0, so
// W(k + NS) starts a group iff k % GS == 0. PROLOGUE: absolute stages, X(s) and W(s) issued for
// s >= 0.
template <int NT, int X_DMAS, int NS, int XS, int GS, bool PROLOGUE = false>
constexpr int dma_vmcnt(int last, int r, bool x_only = false)
{
    int after = 0;
    for(int k = last; k >= last - NS - XS - 2; k--)
    {
        if(PROLOGUE ? k + NS >= 0 : k < 0)
        {
            if(!x_only && k + NS == r)
                return after;
            after += NT * (1 + (nonneg_mod(PROLOGUE ? k + NS : k, GS) == 0));
        }
        if(PROLOGUE ? k + XS - 1 >= 0 : k + XS - 1 < NS)
        {
            if(k + XS - 1 == r)
                return after;
            after += X_DMAS;
        }
    }
    return -1;
}

// Whether a kernel takes reduction length K: whole scale groups (GS 2: K-tile pairs), at least
// the W ring and the peeled end cycles.
template <Cfg C>
static bool dma_k_fits(int K)
{
    const int k_tiles = K / 128;
    return K % 128 == 0 && k_tiles % dma_cycle(C) == 0 && k_tiles >= C.wbuf &&
           k_tiles >= dma_peeled(C);
}

// K loop of one wave on the LDS-DMA weight ring (X_DMA): the whole K, stage = K-tile, in cycles
// of GS stages (compile-time scale group phase and pair parity), the end cycles peeled.
template <Cfg C, int MA>
DEV void dma_k_loop(const Params& p, int mblk, int e, const WaveIds& w, const Sources<C.nt>& src,
                    Routing<C.mt>& rt, v4f (&acc)[C.nt][C.mt], v4i* ring)
{
    constexpr int MT = C.mt, NT = C.nt, WN = C.wn, NS = C.wbuf, XS = C.xbuf, GS = C.gs;
    constexpr bool XG = C.xg, XBF16 = C.xbf16;
    static_assert(C.x == X_DMA && C.s1 && C.wk == 1 && C.kc == 1 && C.kt == 0, "DMA ring");
    static_assert(NS % 2 == 0 && NS >= 2, "LDS ring: even NS");
    // GS > 2: one scale group slot per wave (after the pair slots), refilled with W(GS g) at stage
    // GS g - NS, after the read of group g - 1 (at its first stage GS (g - 1))
    constexpr bool SG = GS > 2;
    static_assert(GS == 2 || NS <= GS, "scale group slot: NS <= GS");
    constexpr int BM              = 16 * MT;
    constexpr int PAIR_SLOT_BYTES = NT * (SG ? 2048 : 2304);  // W (+ scales) of 2 K-tiles
    constexpr int SG_SLOT_OFF     = NS / 2 * PAIR_SLOT_BYTES; // SG: scale group slot offset
    constexpr int SG_TILE_BYTES   = GS == 4 ? 512 : 1024;     // SG: group slot bytes per tile
    constexpr int WAVE_RING_BYTES = dma_wave_ring_bytes(C);
    constexpr int X_DMAS          = dma_x_blocks(C);
    constexpr int X_STAGE_V4      = 4 * MT * DMA_X_SLOTS; // one X ring stage
    static_assert(XS >= 2 && XS <= NS + 3, "X ring");
    // XBF16: register ring of X(s), slot s % X_REGS (X(s) is loaded at stage s - XS + 1 and cast
    // at stage s - 1). The slot is compile-time in the cycle: X_REGS divides GS.
    static_assert(!XBF16 || (XS >= 3 && XS <= NS + 2 && GS % (XS - 1) == 0), "bad XBF16");
    constexpr int X_REGS = XBF16 ? XS - 1 : 1;
    const int k_tiles    = p.K / 128;
    const int lane       = w.lane;
    const int wave_n     = w.wave_n;

    const v4i w_desc = dma_desc(p.w + (size_t)e * p.ntiles * k_tiles * 1024);
    const v4i s_desc = dma_desc(p.w_sc + (size_t)e * p.ntiles * (k_tiles / 2) * 256);
    rt               = read_routing<MT>(p, mblk, w.lane_col);
    // LDS byte addresses (DMA m0) and pointers of the wave's W ring and the shared X ring
    const uint32_t ring_addr = (uint32_t)(uintptr_t)(__attribute__((address_space(3))) v4i*)ring;
    const uint32_t wave_ring_addr =
        __builtin_amdgcn_readfirstlane(ring_addr + wave_n * WAVE_RING_BYTES);
    const uint32_t x_ring_addr = __builtin_amdgcn_readfirstlane(ring_addr + WN * WAVE_RING_BYTES);
    const v4i* w_ring          = ring + wave_n * WAVE_RING_BYTES / 16;
    v4i* const x_ring          = ring + WN * WAVE_RING_BYTES / 16;

    // W(s) DMA (pair half PAR = s & 1); group_start (SG): stage s starts a scale group
    auto dma_w = [&](int s, auto par, bool group_start) {
        constexpr int PAR = par.value;
        const uint32_t slot_addr =
            wave_ring_addr + (uint32_t)((s >> 1) % (NS / 2)) * PAIR_SLOT_BYTES;
#pragma unroll
        for(int j = 0; j < NT; j++)
            lds_dma<16>(w_desc, lane * 16, src.w_off[j] + s * 1024,
                        slot_addr + (PAR * NT + j) * 1024);
        if constexpr(SG)
        {
            if(group_start)
#pragma unroll
                for(int j = 0; j < NT; j++)
                {
                    const int group_off = src.s_off[j] + s / GS * GS * 128;
                    const uint32_t dst  = wave_ring_addr + SG_SLOT_OFF + j * SG_TILE_BYTES;
                    if constexpr(GS == 6)
                        lds_dma<12>(s_desc, lane * 12, group_off, dst);
                    else if constexpr(GS == 8)
                        lds_dma<16>(s_desc, lane * 16, group_off, dst);
                    else
                        lds_dma<16, true, true>(s_desc, lane * 16, group_off, dst);
                }
        }
        else if constexpr(PAR == 0)
        {
#pragma unroll
            for(int j = 0; j < NT; j++)
                lds_dma<4>(s_desc, lane * 4, src.s_off[j] + (s >> 1) * 256,
                           slot_addr + 2 * NT * 1024 + j * 256);
        }
    };

    // X: per K-tile 4 MT blocks of 1 KB: block unit = (m * 2 + col_half) * 2 + row_half holds
    // rows 16 m + 8 row_half + (0..7), bytes 128 col_half + (0..127); slot 8 s + r = row r,
    // column s.
    const int x_rows = p.T;
    const v4i x_desc = dma_desc(p.x, x_rows * p.K * 2);
    int x_dma_voff[X_DMAS];     // global byte offset (row, column) of the lane
    uint32_t x_dma_dst[X_DMAS]; // LDS byte offset of the block within an X ring stage
#pragma unroll
    for(int j = 0; j < X_DMAS; j++)
    {
        // unit >= 4 MT (WN does not divide 4 MT): out-of-range load into the spare 1 KB block,
        // so every wave issues X_DMAS X loads per stage (uniform vmcnt)
        const int unit        = wave_n + WN * j;
        const bool unit_valid = unit < 4 * MT;
        const int row_half = unit & 1, col_half = unit >> 1 & 1, m = unit_valid ? unit >> 2 : 0;
        // XG: lanes 8 r .. 8 r + 7 load the 128 B of row r (one line per 8 lanes), column
        // (lane & 7) ^ r, so slot 8 r + (s ^ r) = row r, column s
        const int dma_row = XG ? lane >> 3 : lane & 7;
        const int t       = p.sorted_ids[mblk * BM + m * 16 + 8 * row_half + dma_row] & 0xffffff;
        x_dma_dst[j]      = unit_valid ? unit * DMA_X_SLOTS * 16 : XS * X_STAGE_V4 * 16;
        // padding rows / unused tiles load an out-of-range row (zeros)
        const int row = t < p.T && m < MA && unit_valid ? t : x_rows;
        x_dma_voff[j] =
            row * p.K * 2 + 128 * col_half + 16 * (XG ? (lane & 7) ^ (lane >> 3) : lane >> 3);
    }
    // XBF16: x_cast(s, slot, vmcnt) waits for X(s) in register slot slot (the wait asm takes the
    // registers, so the casts cannot move above it), casts it and writes the own X ring blocks
    v4i x_regs[X_REGS][X_DMAS];
    auto x_cast = [&](int s, int slot, auto vmcnt) {
#pragma unroll
        for(int j = 0; j < X_DMAS; j++)
            asm volatile("s_waitcnt vmcnt(%1)" : "+v"(x_regs[slot][j]) : "n"(vmcnt.value));
        // units of the active tiles only (the MFMAs read no other); every load is still waited
        // for above: a load in flight into a dead register would clobber its reuse
#pragma unroll
        for(int j = 0; j < X_DMAS; j++)
            if((4 * MA) % WN == 0 || wave_n + WN * j < 4 * MA)
                x_ring[x_dma_dst[j] / 16 + s % XS * X_STAGE_V4 + lane] =
                    cast_bf16x8_f16(x_regs[slot][j]);
    };
    // X(s) DMA into the X ring; XBF16: X(s) load into register slot slot
    auto dma_x = [&](int s, int slot) {
#pragma unroll
        for(int j = 0; j < X_DMAS; j++)
        {
            if constexpr(XBF16)
                asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen"
                             : "=v"(x_regs[slot][j])
                             : "v"(x_dma_voff[j]), "s"(x_desc), "s"(s * 256));
            else
                lds_dma<16, false>(x_desc, x_dma_voff[j], s * 256,
                                   x_ring_addr + x_dma_dst[j] +
                                       (wave_n + WN * j < 4 * MT ? s % XS * X_STAGE_V4 * 16 : 0));
        }
    };
    // X read offsets (16 B slots) of the lane within a stage per q: row r = lane_col & 7, column
    // s = 4 (lane_group & 1) + q in slot 8 s + r (XG: 8 r + (s ^ r))
    int x_read_off[4];
#pragma unroll
    for(int q = 0; q < 4; q++)
    {
        const int r = w.lane_col & 7, s = 4 * (w.lane_group & 1) + q;
        x_read_off[q] = (2 * (w.lane_group >> 1) + (w.lane_col >> 3)) * DMA_X_SLOTS +
                        (XG ? 8 * r + (s ^ r) : 8 * s + r);
    }

#pragma unroll
    for(int j = 0; j < NT; j++)
#pragma unroll
        for(int m = 0; m < MT; m++)
            acc[j][m] = v4f{0, 0, 0, 0};

    // prologue = iterations -END .. -1: X(k + XS - 1) if >= 0, then W(k + NS) if >= 0
    constexpr int END = dma_end_stages(C);
    static_for<END / 2>([&](auto i) {
        constexpr int K = 2 * i.value - END;
        if constexpr(K + XS - 1 >= 0)
            dma_x(K + XS - 1, (K + XS - 1) % X_REGS);
        if constexpr(K + NS >= 0)
            dma_w(K + NS, IntConst<0>{}, nonneg_mod(K + NS, GS) == 0);
        if constexpr(K + XS >= 0)
            dma_x(K + XS, (K + XS) % X_REGS);
        if constexpr(K + NS + 1 >= 0)
            dma_w(K + NS + 1, IntConst<1>{}, false);
    });
    if constexpr(XBF16)
        x_cast(0, 0, IntConst<dma_vmcnt<NT, X_DMAS, NS, XS, GS, true>(-1, 0, true)>{});

    uint32_t w_scale[NT];                    // scale pairs of the current K-tile pair
    uint32_t sg_scales[SG ? GS / 2 : 1][NT]; // SG: the scale pairs of the current group
    // a main-loop stage relative to K-tiles - NS (see dma_vmcnt), phase 0
    constexpr int MAIN_STAGE = -GS * (NS + XS + 4);
    // stage c, POS = c % GS (scale group phase; pair half POS & 1), LEFT stages after c (-1: at
    // least the peeled end): wait + barrier, X refill, read, W refill, compute. XBF16: X(c + 1)
    // is cast after the MFMAs of weight tile 0.
    auto stage = [&](int c, auto pos, auto left) {
        constexpr int POS = pos.value, LEFT = left.value, PAR = POS & 1;
        constexpr bool W_REFILL = LEFT < 0 || LEFT >= NS;     // W(c + NS) exists
        constexpr bool X_REFILL = LEFT < 0 || LEFT >= XS - 1; // X(c + XS - 1) exists
        constexpr int R = LEFT < 0 ? MAIN_STAGE + nonneg_mod(POS + NS, GS) : NS - 1 - LEFT;
        constexpr int X_CAST_VMCNT =
            XBF16 && LEFT != 0 ? dma_vmcnt<NT, X_DMAS, NS, XS, GS>(R, R + 1, true) : -1;
        constexpr int NWAIT = dma_vmcnt<NT, X_DMAS, NS, XS, GS>(R - 1, R);
        static_assert(NWAIT >= 0, "vmcnt");
        // own X / W DMAs of stage c and own reads of stage c - 1 done; after the barrier those
        // of all waves (the X(c - 1) slot then takes X(c + XS - 1))
        vmem_lds_barrier<NWAIT>();
        if constexpr(X_REFILL)
            dma_x(c + XS - 1, (POS + XS - 1) % X_REGS);
        const int pair_slot_off = ((c >> 1) % (NS / 2)) * PAIR_SLOT_BYTES / 16;
        v4i wt[NT];
#pragma unroll
        for(int j = 0; j < NT; j++)
            wt[j] = w_ring[pair_slot_off + (PAR * NT + j) * 64 + lane];
        if constexpr(SG && PAR == 0)
        {
            // the group's pairs are read once, at its first stage
            if constexpr(POS == 0)
            {
#pragma unroll
                for(int j = 0; j < NT; j++)
                {
                    if constexpr(GS == 4)
                    {
                        const v2i t = ((const v2i*)w_ring)[(SG_SLOT_OFF + j * 512) / 8 + lane];
                        sg_scales[0][j] = t[0];
                        sg_scales[1][j] = t[1];
                        continue;
                    }
                    const v4i t = w_ring[SG_SLOT_OFF / 16 + j * 64 + lane];
#pragma unroll
                    for(int q = 0; q < GS / 2; q++)
                        sg_scales[q][j] = t[q];
                }
            }
#pragma unroll
            for(int j = 0; j < NT; j++)
                w_scale[j] = sg_scales[POS / 2][j];
        }
        else if constexpr(PAR == 0)
        {
#pragma unroll
            for(int j = 0; j < NT; j++)
                w_scale[j] =
                    ((const uint32_t*)(w_ring + pair_slot_off + 2 * NT * 64))[j * 64 + lane];
        }
        v8h x[MT][4];
        const v4i* x_stage = x_ring + c % XS * X_STAGE_V4;
#pragma unroll
        for(int m = 0; m < MA; m++)
#pragma unroll
            for(int q = 0; q < 4; q++)
                x[m][q] = __builtin_bit_cast(v8h, x_stage[m * 4 * DMA_X_SLOTS + x_read_off[q]]);
        if constexpr(W_REFILL)
        {
            lds_wait();
            dma_w(c + NS, IntConst<PAR>{}, SG && nonneg_mod(POS + NS, GS) == 0);
        }
        if constexpr(X_CAST_VMCNT >= 0)
        {
            mma_ktile<MT, NT, MA, PAR, false, 0, 1>(wt, w_scale, x, acc);
            x_cast(c + 1, (POS + 1) % X_REGS, IntConst<X_CAST_VMCNT>{});
            mma_ktile<MT, NT, MA, PAR, false, 1, NT>(wt, w_scale, x, acc);
        }
        else
            mma_ktile<MT, NT, MA, PAR>(wt, w_scale, x, acc);
    };
    k_steps<dma_cycle(C), dma_cycle(C), 0, dma_end_stages(C)>(k_tiles, stage);
}

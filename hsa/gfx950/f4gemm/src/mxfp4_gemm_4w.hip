// MXFP4 GEMM for gfx950 (MI355X): C = alpha A B^T in bf16 from aiter's f4gemm inputs (A row-major, B shuffle_weight
// (16, 16), scales e8m0_shuffle), K % 64 == 0. Persistent 4-wave CTAs compute 256 x 256 C tiles (wave w: columns
// 64w.., in all 256 AGPRs); A -> LDS via DMA, B -> registers. A K step (256) is 16 units, one per 16-row m-block:
// 8 MFMAs (4 B tiles x 2 K halves) with loads beside them. A tile runs plain steps, a woven tail, a trailing block.

#include <hip/hip_runtime.h>
#ifndef MXG4W_NO_ENTRY
#include "mxfp4_gemm.h"                                                // C API
#endif
#include <algorithm>
#include <utility>

#define INLINE __attribute__((always_inline))
#define DEVICE __device__ __forceinline__

namespace mxg4w {

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef __amdgpu_buffer_rsrc_t Buf;

// ---- shape and loads ------------------------------------------------------------------------------------------------
constexpr int WaveSize = 64, Waves = 4, Threads = Waves * WaveSize;
constexpr int TileM = 256, TileN = 256, KStep = 256, Units = 16, Slots = 8, BTiles = 4;
// Per step and wave: 10 B loads (8 x 16 B, 2 scale dwords) and an A window of 10 DMAs (8 x 1 KB, 2 x 256 B).
constexpr int ARowBytes = KStep / 2, ABlockBytes = 16 * ARowBytes, DmaBytes = 1024, ScaleBytes = 256, Loads = 10;
constexpr int BHalfBytes = 1024, BStepBytes = 2 * BHalfBytes;        // B per tile and K step: 64 lanes x 16 B x 2
enum Tensor { TensorA, TensorAScale, TensorB, TensorBScale, Tensors };
// Window DMAs 0-3 load A rows 0-127 (DMA d: rows 32d..32d+31); DMA 4 their scales; 5-8 rows 128-255; 9 their scales.
constexpr int dma_half(int dma) { return dma / 5; }
constexpr bool dma_is_scale(int dma) { return dma % 5 == 4; }
constexpr int dma_chunk(int dma) { return dma - dma / 5; }             // its 32-row chunk (not a scale)
constexpr Tensor dma_tensor(int dma) { return dma_is_scale(dma) ? TensorAScale : TensorA; }
constexpr uint32_t Oob = 0x80000000u;                                 // an offset past every buffer
constexpr int64_t MallBudget = 128 << 20;                             // half of MALL (256 MB)

// ---- LDS: two A stages, two A scale stages, a C ring per wave -------------------------------------------------------
// The ring: 8 m-blocks of 16 rows x 64 bf16 columns, two planes of 64 B rows (columns 8k..8k+3 | 8k+4..8k+7).
constexpr int RingRowBytes = BTiles * 16, RingBlockBytes = 8 * RingRowBytes, PlaneBytes = 16 * RingBlockBytes;
constexpr int AStage = TileM * ARowBytes, ScaleStage = TileM / 32 * ScaleBytes;   // 32 KB, 2 KB
constexpr int ScaleLds = 2 * AStage, RingLds = ScaleLds + 2 * ScaleStage, LdsBytes = RingLds + Waves * 2 * PlaneBytes;
constexpr int swizzle(int row) { return row / 2 % 4; }                 // 16 B chunks: so a park hits 8 bank groups
static_assert(LdsBytes <= 160 * 1024);

// ---- what a step's 16 units do beside their MFMAs (the load pattern) ------------------------------------------------
// Slot 0 waits. Units 0-4 load B (slots 2, 4); unit 5 waits for the window and syncs; units 5-14 DMA the next window
// (slot 6). A lane moves on a unit after its load; moving sooner costs up to 18%. Each unit reads the A of the unit
// ReadAhead later (slots 1, 3) and, if new, its scale vector (slot 5).
constexpr int ReadAhead = 3, BarrierUnit = Loads / 2, AReads = 2, ScaleSlot = 5, DmaSlot = 6;
constexpr bool reads_a(int slot) { return slot == 1 || slot == 3; }   // K half = slot / 2
// vmcnt: unit 0 waits for its B (the window's 10 DMAs younger), the barrier unit for the window (10 B loads younger).
constexpr int vm_wait(int unit) { return unit == 0 || unit == BarrierUnit ? Loads : 63; }
// lgkmcnt: the barrier unit drains LDS; even units wait for this and the next unit's A (the newest AReads LDS ops).
constexpr int lgkm_wait(int unit) { return unit == BarrierUnit ? 0 : unit % 2 ? 15 : AReads; }

// ---- the tail: the tile's last two steps a, b in the order a0 .. a7 | b0 a8 b1 a9 .. b7 a15 | b8 .. b15 -------------
constexpr int TailUnits = 2 * Units, NextBUnit = 24, WindowAUnit = Units + BarrierUnit, CBlocksAtWindowA = 7;
constexpr int tail_step(int unit) { return unit >= 24 || (unit >= 8 && unit % 2 == 0); }   // 0: a, 1: b
constexpr int tail_block(int unit)                                    // units 8-23: b_j (even), a_j+8 (odd)
{ return unit < 8 || unit >= 24 ? unit % Units : (unit - 8) / 2 + 8 * (unit % 2); }
// The tail's epilogue of m-block p < 11, where `since` = units after unit 9 + 2p (p <= 8 is right after its MFMA):
//   since 0-2   slot s: read r = 8 since + s < 16 (AGPR r of the m-block; B tile n: 4n..4n+3), op r - 2: op 4n + k
//               scales pair k by alpha (k < 2) or converts pair k - 2 to bf16; ops 8, 16 park tile pairs 0, 1
//   since 3, 4  slot 4: drain 8-row C block 2p, 2p + 1 from the ring
//   since 5, 6  slot 7: store C block 2p, 2p + 1
constexpr int PackedInTail = 11, ConvertLag = 2, DrainSlot = 4, StoreSlot = 7;

// ---- one-instruction helpers (inline asm: LLVM would add waits around VMEM and move the accumulators) ---------------
// Accumulator tile acc += B x A; op_sel picks the scale bytes: B tile n % 2 (bn), m-block % 2 (am), K half (kh).
DEVICE void mma(int acc, u32x4 b, u32x4 a, uint32_t sb, uint32_t sa, int bn, int am, int kh)
{ asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 a[%0:%1], %2, %3, a[%0:%1], %4, %5 "
               "op_sel:[%6,%7,0] op_sel_hi:[%8,%8,0] cbsz:4 blgp:4"
               :: "i"(4 * acc), "i"(4 * acc + 3), "v"(b), "v"(a), "v"(sb), "v"(sa), "i"(bn), "i"(am), "i"(kh)); }
DEVICE void acc_read(f32x2& x, int i, int reg) { asm volatile("v_accvgpr_read_b32 %0, a[%1]" : "=v"(x[i]) : "i"(reg)); }
DEVICE void acc_zero(int i, float z)                                   // AGPRs 32i..32i+31 = z (0), on the MFMA pipe
{ asm volatile("v_mfma_f32_32x32x1_2b_f32 a[%0:%1], %2, %2, 0" :: "i"(32 * i), "i"(32 * i + 31), "v"(z)); }

DEVICE void ds_read(u32x4& x, uint32_t addr, int off)
{ asm volatile("ds_read_b128 %0, %1 offset:%2" : "=v"(x) : "v"(addr), "i"(off)); }
DEVICE void ds_read_block(u32x4& x, uint32_t addr, int off)           // 8 B from each ring plane (offsets / 512)
{ asm volatile("ds_read2st64_b64 %0, %1 offset0:%2 offset1:%3" : "=v"(x) : "v"(addr), "i"(off / 512),
               "i"((off + PlaneBytes) / 512)); }
DEVICE void ds_write(uint32_t addr, u32x4 x, int off)
{ asm volatile("ds_write_b128 %0, %1 offset:%2" :: "v"(addr), "v"(x), "i"(off) : "memory"); }

DEVICE int uniform(int x) { return __builtin_amdgcn_readfirstlane(x); }
DEVICE Buf rsrc(const void* base, uint32_t bytes) {                   // a buffer, in SGPRs
    const uint64_t at = (uint64_t)base, lo = (uint32_t)uniform((int)at), hi = (uint32_t)uniform((int)(at >> 32));
    return __builtin_amdgcn_make_buffer_rsrc((void*)(hi << 32 | lo), 0, uniform((int)bytes), 0x20000);
}
DEVICE void load(u32x4& x, Buf r, uint32_t lane, int off)             // soffset 0: an SGPR costs ~4 issue cycles
{ asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3" : "=v"(x) : "v"(lane), "s"(r), "i"(off)); }
DEVICE void load(uint32_t& x, Buf r, uint32_t lane)
{ asm volatile("buffer_load_dword %0, %1, %2, 0 offen" : "=v"(x) : "v"(lane), "s"(r)); }
// Window DMA: M0 (LLVM leaves unused) is its LDS address two slots early (in the DMA's slot: +0.5%), 16 or 4 B a lane.
DEVICE void window_m0(int dma, uint32_t rows_lds, uint32_t scales_lds) {
    const int scale_off = ScaleStage / 2 * dma_half(dma), rows_off = Waves * DmaBytes * dma_chunk(dma);
    if (dma_is_scale(dma)) asm volatile("s_add_u32 m0, %0, %1" :: "s"(scales_lds), "i"(scale_off) : "scc");
    else asm volatile("s_add_u32 m0, %0, %1" :: "s"(rows_lds), "i"(rows_off) : "scc");
}
DEVICE void window_dma(int dma, Buf rows, Buf scales, uint32_t lane) {
    if (dma_is_scale(dma)) asm volatile("buffer_load_dword %0, %1, 0 offen lds" :: "v"(lane), "s"(scales) : "memory");
    else asm volatile("buffer_load_dwordx4 %0, %1, 0 offen lds" :: "v"(lane), "s"(rows) : "memory");
}

// C block q (nt: past L2 and MALL) at row offset q * block, computed here so LLVM doesn't keep 32 SGPR offsets.
DEVICE void store16(u32x4 x, Buf r, uint32_t lane, uint32_t block, int q) {
    uint32_t row;
    asm volatile("s_mul_i32 %0, %1, %2\n\tbuffer_store_dwordx4 %3, %4, %5, %0 offen nt"
                 : "=&s"(row) : "s"(block), "i"(q), "v"(x), "v"(lane), "s"(r) : "memory");
}
// Split K adds a block into C (host-zeroed) as 4 atomics of 2 rows, transposing register j and 16-lane group g.
DEVICE void add16(u32x4 x, Buf r, uint32_t lane, uint32_t rows2, int q) {
    const auto a = __builtin_amdgcn_permlane32_swap(x.x, x.z, false, false);
    const auto b = __builtin_amdgcn_permlane32_swap(x.y, x.w, false, false);
    const auto lo = __builtin_amdgcn_permlane16_swap(a[0], b[0], false, false);
    const auto hi = __builtin_amdgcn_permlane16_swap(a[1], b[1], false, false);
    const uint32_t y[4] = {lo[0], lo[1], hi[0], hi[1]};
    uint32_t row;
    for (int j = 0; j < 4; ++j)
        asm volatile("s_mul_i32 %0, %1, %2\n\tbuffer_atomic_pk_add_bf16 %3, %4, %5, %0 offen"
                     : "=&s"(row) : "s"(rows2), "i"(4 * q + j), "v"(y[j]), "v"(lane), "s"(r) : "memory");
}

// Every barrier clobbers all 256 AGPRs: LLVM can't see the asm MFMAs' accumulators and would park values there.
#define A10(p) p "0", p "1", p "2", p "3", p "4", p "5", p "6", p "7", p "8", p "9"   // p0 .. p9
#define AGPRS_ALL_256 A10("a"), A10("a1"), A10("a2"), A10("a3"), A10("a4"), A10("a5"), A10("a6"), A10("a7"), \
    A10("a8"), A10("a9"), A10("a10"), A10("a11"), A10("a12"), A10("a13"), A10("a14"), A10("a15"), A10("a16"), \
    A10("a17"), A10("a18"), A10("a19"), A10("a20"), A10("a21"), A10("a22"), A10("a23"), A10("a24"), \
    "a250", "a251", "a252", "a253", "a254", "a255"
DEVICE void barrier() {
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory", AGPRS_ALL_256);
}
DEVICE void fence() { __builtin_amdgcn_sched_barrier(0); }
DEVICE void wait(int vm, int lgkm)                                      // vmcnt (63: none), lgkmcnt (15: none)
{ if (vm < 63 || lgkm < 15) asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(%1)" :: "i"(vm), "i"(lgkm)); }
template <class T> DEVICE void pin(T& x) { asm volatile("" : "+v"(x)); }   // computed here (a VGPR), not at its use
template <class... T> DEVICE void pin_all(T&... x) { (pin(x), ...); }
DEVICE void move(uint32_t& x, uint32_t delta) { pin(x += delta); }      // a lane offset, VALU in its slot
DEVICE uint32_t to_bf16(f32x2 x) {
    uint32_t y = __builtin_bit_cast(uint32_t, __builtin_convertvector(x, bf16x2));
    pin(y);
    return y;
}
DEVICE bool at_least(int x, int y) {          // uniform x >= y that LLVM keeps (a plain `if` merges the code copies)
    asm goto("s_cmp_ge_i32 %0, %1\n\ts_cbranch_scc1 %l2" :: "s"(x), "s"(y) : "scc" : yes);
    return false;
yes:
    return true;
}

struct Args {
    const void *a, *b, *sa, *sb;
    __bf16* c;
    int m, n, k, cus, a_stride, sa_stride, sb_stride, c_stride;   // strides: bytes, bytes, bytes, elements
    float alpha;                                     // C = alpha A B^T
    int log2_split;                                  // split K into 2^log2_split slices (aiter's log2_k_split)
};

// ---- the kernel, per wave -------------------------------------------------------------------------------------------
DEVICE void run(const Args& args, uint8_t* lds, int wave, int lane) {
    asm volatile("" :: "s"(args.a), "s"(args.b), "s"(args.sa), "s"(args.sb), "s"(args.c), "s"(args.m), "s"(args.n),
                 "s"(args.k), "s"(args.cus), "s"(args.a_stride), "s"(args.sa_stride), "s"(args.sb_stride),
                 "s"(args.c_stride), "s"(args.alpha), "s"(args.log2_split));   // all in SGPRs: one SMEM round trip
    const int tiles_m = (args.m + TileM - 1) / TileM, tiles_n = (args.n + TileN - 1) / TileN, tiles = tiles_m * tiles_n;

    // Split K as aiter's: work items = tiles x K slices, while each still gets its own CTA (else fewer slices, exact).
    int log2_split = args.log2_split, k_slice, slices;
    for (;; --log2_split) {
        k_slice = ((args.k >> log2_split) + KStep - 1) / KStep * KStep;
        slices = (args.k + k_slice - 1) / k_slice;
        if (log2_split == 0 || tiles * slices <= args.cus) break;
    }
    const int work = tiles * slices, cta = blockIdx.x, ctas = min(work, args.cus);   // ctas: the grid
    const bool split = uniform(slices > 1);

    // ---- tile order: groups of group_size tiles on one axis x the whole other axis, group axis fastest -----------
    // An XCD (cta % 8) walks its run, each round dealing positions to its CTAs: groups of 8 M tiles (64 or 32 N tiles
    // if B per group exceeds half of MALL, dealt round-robin). K < 12288: XCD x starts x / 8 in. Under two rounds whose
    // panels fit half of MALL, a CTA's two tiles are neighbours.
    const int64_t panel = (int64_t)TileM * args.k / 2;                 // bytes of one 256-row A or B panel
    const bool spills = tiles_m > 8 && (min(tiles_m, 8 * 8) + tiles_n) * panel > MallBudget;
    const bool n_groups_64 = spills && ctas % 64 == 0 && tiles_n % 64 == 0 && 64 * panel <= MallBudget;
    const bool n_groups_32 = spills && ctas % 32 == 0 && tiles_n % 32 == 0 && 32 * panel <= MallBudget;
    const int log2_group = n_groups_64 ? 6 : n_groups_32 ? 5 : 3, xcd = cta % 8;
    // tile_of's values in VGPRs; group axis M, or N if by_n. x / n = umulhi(x, 2^32 / n + 1) (exact for small x).
    uint32_t group_size = 1 << log2_group, by_n = group_size > 8, axis_tiles = by_n ? tiles_n : tiles_m;
    uint32_t group_tiles = group_size * (by_n ? tiles_m : tiles_n), group_magic = 0xFFFFFFFFu / group_tiles + 1;
    uint32_t size_magic = 1u << (32 - log2_group), last_size = axis_tiles & (group_size - 1);
    uint32_t last_magic = last_size ? 0xFFFFFFFFu / last_size + 1 : 0, all_tiles = tiles;
    uint32_t run_first = by_n ? xcd : xcd * (work / 8) + min(xcd, work % 8), run_stride = by_n ? 8 : 1;
    uint32_t tiles_magic = split ? 0xFFFFFFFFu / tiles + 1 : 0, end_m = tiles_m * TileM, end_n = tiles_n * TileN;
    pin_all(group_size, by_n, axis_tiles, all_tiles, group_tiles, group_magic, size_magic, last_size, last_magic);
    pin_all(run_first, run_stride, tiles_magic, end_m, end_n);

    const int in_xcd = cta / 8, xcd_ctas = (ctas - xcd + 7) / 8, run = work / 8 + (xcd < work % 8);   // run >= xcd_ctas
    const int extra = run - xcd_ctas, run_start = args.k < 12288 ? xcd * run / 8 : 0;
    const bool pairs = extra < xcd_ctas && (tiles_m + tiles_n) * panel <= MallBudget;
    const int first = pairs ? in_xcd + min(in_xcd, extra) : in_xcd, stride = pairs ? 1 : xcd_ctas;
    const int rounds = pairs ? 1 + (in_xcd < extra) : run;          // pairs: CTAs below extra get two tiles
    // The CTA's tile in a round (C row, column; past the end if none) and K slice; item uniform (else LLVM copies B).
    auto tile_of = [&](int round, int& row0, int& col0) INLINE {
        const int in_run = first + round * stride, rotated = in_run + run_start;
        const int item = uniform(round < rounds && in_run < run ? (rotated >= run ? rotated - run : rotated) : -1);
        const uint32_t work_item = run_first + run_stride * item;
        const uint32_t slice = __umulhi(work_item, tiles_magic), tile = work_item - slice * all_tiles;   // split K only
        const uint32_t group = __umulhi(tile, group_magic), in_group = tile - group * group_tiles;
        const bool short_group = group_size * (group + 1) > axis_tiles;
        const uint32_t size = short_group ? last_size : group_size, magic = short_group ? last_magic : size_magic;
        const uint32_t across = __umulhi(in_group, magic) + (size == 1 ? in_group : 0);   // the other axis
        const uint32_t along = group_size * group + in_group - across * size;            // the group axis
        row0 = item >= 0 ? (by_n ? across : along) * TileM : end_m;
        col0 = item >= 0 ? (by_n ? along : across) * TileN : end_n;
        return slice;
    };
    int round = 0, next_m, next_n;
    const int k_slice_index = uniform(tile_of(0, next_m, next_n));
    int tile_m = uniform(next_m), tile_n = uniform(next_n);           // the tile in the MFMAs; next_*: the next tile

    // ---- K: panels k0 .. k0 + steps - 1, at least 2 (masked past K); split K: one work item per CTA -----------------
    const int k0 = k_slice_index * k_slice / KStep, k_end = min(k0 * KStep + k_slice, args.k);
    const int steps = max((k_end + KStep - 1) / KStep - k0, 2);
    const bool k_tail = steps * KStep != k_end - k0 * KStep;
    uint32_t k_scale_mask[2];                        // steps a, b: lane / 16 = g keeps K blocks g, g + 4 if < k_end
    for (int step = 0; step < 2; ++step) {
        const int k_blocks = std::clamp(k_end / 32 - 8 * (k0 + steps - 2 + step), 0, 8), group = lane / 16;
        k_scale_mask[step] = (group < k_blocks ? 0xFFFFu : 0u) | (group + 4 < k_blocks ? 0xFFFF0000u : 0u);
    }

    // ---- global loads: one buffer per tensor; lane offsets (tile, K panel, lane) move on after each load ------------
    const uint32_t row_bytes[Tensors] = {(uint32_t)args.a_stride, (uint32_t)args.sa_stride, (uint32_t)args.k / 2,
                                         (uint32_t)args.sb_stride};   // uint32_t: buffers reach 4 GiB (no int overflow)
    const Buf a_rsrc = rsrc(args.a, args.m * row_bytes[TensorA]);
    const Buf a_scale_rsrc = rsrc(args.sa, tiles_m * TileM * row_bytes[TensorAScale]);
    const Buf b_rsrc = rsrc(args.b, args.n * row_bytes[TensorB]);
    const Buf b_scale_rsrc = rsrc(args.sb, tiles_n * TileN * row_bytes[TensorBScale]);
    const int panel_bytes[Tensors] = {ARowBytes, ScaleBytes, BStepBytes, ScaleBytes};             // per K step
    uint32_t dma_lane[Loads], b_lane[BTiles], b_scale_lane[2];       // [window DMA], [B tile], [B tile pair]
    for (int d = 0; d < Loads; ++d) {                // A rows: 32 chunk + 8 wave + lane / 8, swizzled 16 B pieces
        const int chunk = dma_chunk(d), half = dma_half(d);
        if (dma_is_scale(d)) dma_lane[d] = (tile_m + 32 * (4 * half + lane % 4)) * row_bytes[TensorAScale] + 64 * wave
                                           + 4 * (lane / 4) + k0 * ScaleBytes;
        else dma_lane[d] = (tile_m + 8 * (Waves * chunk + wave) + lane / 8) * row_bytes[TensorA]
                           + 16 * ((lane % 8) ^ (lane / 8 & 6)) + k0 * ARowBytes;
    }
    for (int tile = 0; tile < BTiles; ++tile)
        b_lane[tile] = (tile_n + 16 * (BTiles * wave + tile)) * row_bytes[TensorB] + 16 * lane + k0 * BStepBytes;
    for (int pair = 0; pair < 2; ++pair)
        b_scale_lane[pair] = (tile_n + 64 * wave + 32 * pair) * row_bytes[TensorBScale] + 4 * lane + k0 * ScaleBytes;
    // Per tensor: K step now and next round (odd rounds walk back from L2-warm panels; K tail: forward), tile jump.
    int step_now[Tensors], step_next[Tensors], jump[Tensors];
    for (int tensor = 0; tensor < Tensors; ++tensor) step_now[tensor] = panel_bytes[tensor];
    for (int tensor = 0; tensor < Tensors; ++tensor) step_next[tensor] = k_tail ? step_now[tensor] : -step_now[tensor];

    // ---- LDS: lane 16g + r reads row r of an A tile, K 32g of the K half; scales: 16 B per lane ---------------------
    const uint32_t lds_base = (uint32_t)(uintptr_t)(__attribute__((address_space(3))) uint8_t*)lds;
    uint32_t a_read[2][2], scale_read[2], dma_lds[2], scale_dma_lds[2];   // [stage][K half], [stage]
    for (int stage = 0; stage < 2; ++stage) {
        for (int half = 0; half < 2; ++half)
            a_read[stage][half] = lds_base + AStage * stage + (lane % 16) * ARowBytes
                                  + 16 * ((4 * half + lane / 16) ^ (lane % 16 & 6));
        scale_read[stage] = lds_base + ScaleLds + ScaleStage * stage + 16 * lane;
        dma_lds[stage] = lds_base + AStage * stage + DmaBytes * wave;   // + 4 KB per chunk
        scale_dma_lds[stage] = lds_base + ScaleLds + ScaleStage * stage + ScaleBytes * wave;   // + 1 KB per half
    }
    // The ring (m-block m in slot m % 8). Park: lane 16g + r writes row r, columns 4g..4g+3 of each tile of a pair.
    // Drain: lane 8r + k reads row r of an 8-row C block, columns 8k..8k+7 (8 B per plane), stored as 16 B.
    const uint32_t ring = lds_base + RingLds + 2 * PlaneBytes * wave, col_group = lane % 8, block_row = lane / 8;
    uint32_t park_lane[2];
    for (int pair = 0; pair < 2; ++pair)
        park_lane[pair] = ring + (lane / 16 % 2) * PlaneBytes + (lane % 16) * RingRowBytes
                          + 16 * ((2 * pair + lane / 32) ^ swizzle(lane % 16));
    const uint32_t block_read = ring + block_row * RingRowBytes + 16 * ((2 * (col_group / 4) + col_group % 2)
                                ^ swizzle(block_row)) + 8 * (col_group / 2 % 2);

    // ---- C: a buffer per tile at its row 0 (rows past M fall out), lane offset Oob past N -----------------
    // Split K's atomics (add16): lane L adds row (L % 16) / 8 and columns + 2 (L / 16) of the store lane's 8.
    const int c_stride = uniform(args.c_stride), m_rows = uniform(args.m), m_past = m_rows - 64;
    const uint32_t c_block_bytes = 8 * c_stride * 2, c_col = 64 * wave + 8 * col_group;
    const uint32_t c_lane0 = (block_row * c_stride + c_col) * 2, add_shift = 4 * (lane / 16) * (c_stride - 1);
    int c_rows;                                                        // this tile's rows in C
    uint32_t c_lane, add_lane;
    Buf c_buffer;
    f32x2 alpha = {args.alpha, args.alpha};
    float zero = 0.0f;
    pin_all(alpha, zero);                                              // VGPRs

    u32x4 a_regs[4][2], a_scales[4];                 // A of units in flight [unit % 4][K half]; scale vectors
    u32x4 b_tile[2][BTiles][2];                      // B buffers 0, 1 (= step parity) [B tile][K half]
    uint32_t b_scale[2][2];                          // [buffer][B tile pair]
    f32x2 acc_pairs[Units][BTiles][2];               // AGPRs read out, as pairs
    u32x4 packed[Units][2], drained[2 * Units];      // bf16 per m-block tile pair; per 8-row C block

    // ---- a plain step s of parity P: B(s) in buffer P, A(s) in stage P ----------------------------------------------
    // Short tiles (<= 64 rows) run units 4-15 without MFMAs or A reads, and DMA only A rows 0-63 and the scales.
    auto plain_units = [&](int parity, int first, int last, bool short_tile) INLINE {
#pragma unroll
        for (int unit = first; unit < last; ++unit) {
            const int ahead = unit + ReadAhead, ahead_stage = ahead < Units ? parity : 1 - parity;
            const int d = unit - BarrierUnit, dma_stage = dma_half(d) ? 1 - parity : parity;   // units 5-14: DMA d
#pragma unroll
            for (int slot = 0; slot < Slots; ++slot) {
                // Snaking (tiles 0 0 1 1 .. 3 3, odd units reversed; K halves 0 1 1 0 ..): shared operands save energy.
                const int tile = unit % 2 ? BTiles - 1 - slot / 2 : slot / 2, half = (slot ^ slot / 2) % 2;
                if (slot == 0) wait(vm_wait(unit), lgkm_wait(unit));
                if (!short_tile)
                    mma(BTiles * unit + tile, b_tile[parity][tile][half], a_regs[unit % 4][half],
                        b_scale[parity][tile / 2], a_scales[unit / 8][unit % 8 / 2], tile % 2, unit % 2, half);
                if (slot == 0 && unit == BarrierUnit) barrier();      // the last window landed; rows 0-127 were read

                // The A of unit + 3 (13-15: the next step's 0-2, in stage 1 - P) and, if new, its scale vector.
                if (reads_a(slot) && (!short_tile || ahead >= Units))
                    ds_read(a_regs[ahead % 4][slot / 2], a_read[ahead_stage][slot / 2], ABlockBytes * (ahead % Units));
                if (slot == ScaleSlot && ahead % 8 == 0)
                    ds_read(a_scales[ahead % Units / 8], scale_read[ahead_stage], ScaleStage / 2 * (ahead % Units / 8));

                // B(s + 1) into buffer 1 - P: B load i = 2 unit + slot / 2 - 1 (slots 2, 4) is tile i % 4 of K half
                // i / 4, then scale pair i - 8. Lanes of loads 4-9 move a unit later, two slots earlier (load i - 1).
                const int b_load = 2 * unit + slot / 2 - 1, b_prev = b_load - 1, b_half = b_load / 4;
                if ((slot == 2 || slot == 4) && b_load < 8)
                    load(b_tile[1 - parity][b_load % 4][b_half], b_rsrc, b_lane[b_load % 4], BHalfBytes * b_half);
                if ((slot == 2 || slot == 4) && (b_load == 8 || b_load == 9))
                    load(b_scale[1 - parity][b_load - 8], b_scale_rsrc, b_scale_lane[b_load - 8]);
                if ((slot == 0 || slot == 2) && b_prev >= 4 && b_prev < 8) move(b_lane[b_prev % 4], step_now[TensorB]);
                if ((slot == 0 || slot == 2) && (b_prev == 8 || b_prev == 9))
                    move(b_scale_lane[b_prev - 8], step_now[TensorBScale]);

                // The window: A(s + 2) rows 0-127 into stage P (read: the barrier), A(s + 1) rows 128-255 into 1 - P.
                const bool dma = d >= 0 && d < Loads && (!short_tile || d < 2 || d == 4);
                if (slot == 4 && dma) window_m0(d, dma_lds[dma_stage], scale_dma_lds[dma_stage]);
                if (slot == 5 && dma && short_tile) asm volatile("s_nop 0");   // M0 -> DMA (no MFMA between)
                if (slot == DmaSlot && dma) window_dma(d, a_rsrc, a_scale_rsrc, dma_lane[d]);
                if (slot == 4 && d >= 1 && d <= Loads) move(dma_lane[d - 1], step_now[dma_tensor(d - 1)]);
                fence();
            }
            if (short_tile && unit == Units - 1) wait(3, 15);   // B(s + 1) landed (DMAs 0, 1, 4 younger)
        }
    };

    // ---- the tail: steps a (B buffer 0, stage 0) and b (buffer 1, stage 1) in tail order ----------------------------
    // Units 0-15 load as a plain parity-0 step: B(b); window a = next tile's A(0) rows 0-127 + A(b) rows 128-255.
    // Units 16-31 as parity 1: window b = next tile's A(1) rows 0-127 + A(0) rows 128-255, and its B(0) 8 units late
    // (after a15, buffer 0's last user). Unit 0 plans the next tile; the epilogue runs from unit 9. Short tiles run
    // units 16-31 without MFMAs, A reads and packs of m-blocks 4-10.
    auto tail_units = [&](int first, int last, bool short_tile) INLINE {
#pragma unroll
        for (int unit = first; unit < last; ++unit) {
            const int step = tail_step(unit), block = tail_block(unit);
            const int ahead = unit + ReadAhead, ahead_step = tail_step(ahead), ahead_block = tail_block(ahead);
            const int parity = unit / Units, u = unit % Units, d = u - BarrierUnit;   // load pattern: parity, unit, DMA
            const int b_unit = parity ? unit - NextBUnit : unit, dma_stage = dma_half(d) ? 1 - parity : parity;
            const int* b_delta = parity ? step_next : jump;              // B(b) lanes jump to the next tile
            const int* dma_delta = parity == 0 && d >= 6 ? jump : step_next;   // so do A(b)'s rows 128-255 (DMAs 5-9)
#pragma unroll
            for (int slot = 0; slot < Slots; ++slot) {
                // vmcnt by tail unit: 0, 5 as in a plain step; 8: B(b) (3 DMAs younger); 21: window a, behind the 7 C
                // blocks stored since (5 VM ops each with split K); others none. lgkmcnt by load-pattern unit u.
                if (slot == 0 && unit != WindowAUnit) wait(unit == 8 ? 3 : vm_wait(unit), lgkm_wait(u));
                if (slot == 0 && unit == WindowAUnit) wait(5 * CBlocksAtWindowA, 0);
                if (slot == 0 && unit == WindowAUnit && __builtin_expect(!split, 1)) wait(CBlocksAtWindowA, 0);
                if (slot == 0 && block % 8 == 0) {          // scales past K: e8m0 0 in A and B, so the products vanish
                    for (int pair = 0; pair < 2 && block == 0; ++pair) b_scale[step][pair] &= k_scale_mask[step];
                    a_scales[2 * step + block / 8] &= k_scale_mask[step];
                    fence();
                    asm volatile("s_nop 4");                                // VALU write -> MFMA read
                }
                if (!short_tile)
                    mma(BTiles * block + slot / 2, b_tile[step][slot / 2][slot % 2], a_regs[unit % 4][slot % 2],
                        b_scale[step][slot / 4], a_scales[2 * step + block / 8][block % 8 / 2], slot / 2 % 2,
                        block % 2, slot % 2);
                if (slot == 0 && u == BarrierUnit) barrier();

                // The A of tail unit + 3 and, if new, its scale vector (the next tile's: in the trailing block).
                if (reads_a(slot) && ahead < TailUnits && !short_tile)
                    ds_read(a_regs[ahead % 4][slot / 2], a_read[ahead_step][slot / 2], ABlockBytes * ahead_block);
                if (slot == ScaleSlot && ahead < TailUnits && ahead_block % 8 == 0)
                    ds_read(a_scales[2 * ahead_step + ahead_block / 8], scale_read[ahead_step],
                            ScaleStage / 2 * (ahead_block / 8));

                // B(b) into buffer 1 (units 0-4), the next tile's B(0) into buffer 0 (24-28): as in a plain step.
                const int b_load = 2 * b_unit + slot / 2 - 1, b_prev = b_load - 1, b_half = b_load / 4;
                if ((slot == 2 || slot == 4) && b_load >= 0 && b_load < 8)
                    load(b_tile[1 - parity][b_load % 4][b_half], b_rsrc, b_lane[b_load % 4], BHalfBytes * b_half);
                if ((slot == 2 || slot == 4) && (b_load == 8 || b_load == 9))
                    load(b_scale[1 - parity][b_load - 8], b_scale_rsrc, b_scale_lane[b_load - 8]);
                if ((slot == 0 || slot == 2) && b_prev >= 4 && b_prev < 8) move(b_lane[b_prev % 4], b_delta[TensorB]);
                if ((slot == 0 || slot == 2) && (b_prev == 8 || b_prev == 9))
                    move(b_scale_lane[b_prev - 8], b_delta[TensorBScale]);

                // Windows a (5-14) and b (21-30), as in plain steps of parity 0, 1.
                const bool dma = d >= 0 && d < Loads;
                if (slot == 4 && dma) window_m0(d, dma_lds[dma_stage], scale_dma_lds[dma_stage]);
                if (slot == 5 && dma && short_tile) asm volatile("s_nop 0");
                if (slot == DmaSlot && dma) window_dma(d, a_rsrc, a_scale_rsrc, dma_lane[d]);
                if (slot == 4 && d >= 1 && d <= Loads) move(dma_lane[d - 1], dma_delta[dma_tensor(d - 1)]);

                // Unit 0 plans the next tile: the jump, this tile's C; window a's rows 0-127 lanes move to its step 0.
                if (unit == 0 && slot == 3) {
                    tile_of(round + 1, next_m, next_n);
                    for (int tensor = 0; tensor < Tensors; ++tensor)   // K: back to k0 (K tail), else stay on the panel
                        jump[tensor] = (tensor <= TensorAScale ? next_m - tile_m : next_n - tile_n) * row_bytes[tensor]
                                       + (k_tail ? 1 - steps : 0) * panel_bytes[tensor];
                    c_rows = max(min(m_rows - tile_m, TileM), 0);
                    c_lane = tile_n + c_col < args.n ? c_lane0 : c_lane0 | Oob;
                    add_lane = c_lane - add_shift;
                    c_lane |= split ? Oob : 0;
                    c_buffer = rsrc(args.c + (size_t)tile_m * c_stride + tile_n, c_rows * c_stride * 2);
                    for (int i = 0; i < 5; ++i) move(dma_lane[i], jump[dma_tensor(i)] - step_now[dma_tensor(i)]);
                }

                // The epilogue of m-blocks 0-10 (timeline above): AGPR read, op, C block drained and stored.
#pragma unroll
                for (int p = 0; p < PackedInTail; ++p) {
                    const int since = unit - 9 - 2 * p, read = Slots * since + slot, op = read - ConvertLag;
                    const int c_block = 2 * p + (since - 1) % 2;
                    const bool packs = !short_tile || p < 4;                     // short: m-blocks 4-10 have no rows
                    if (packs && read >= 0 && read < 16)
                        acc_read(acc_pairs[p][read / 4][read % 4 / 2], read % 2, 16 * p + read);
                    if (packs && op >= 0 && op < 16 && op % 4 < 2) acc_pairs[p][op / 4][op % 2] *= alpha;
                    if (packs && op >= 0 && op < 16 && op % 4 >= 2)
                        packed[p][op / 8][op / 4 % 2 * 2 + op % 2] = to_bf16(acc_pairs[p][op / 4][op % 2]);
                    if (packs && (op == 8 || op == 16))
                        ds_write(park_lane[op / 8 - 1], packed[p][op / 8 - 1], p % 8 * 2 * RingBlockBytes);
                    if (slot == DrainSlot && (since == 3 || since == 4))
                        ds_read_block(drained[c_block], block_read, c_block % 16 * RingBlockBytes);
                    if (slot == StoreSlot && (since == 5 || since == 6)) {
                        wait(63, short_tile ? 0 : 4);                            // its drain (4 LDS ops since)
                        store16(drained[c_block], c_buffer, c_lane, c_block_bytes, c_block);
                        if (__builtin_expect(split, 0) && packs)                 // short: m-blocks 0-3 only
                            add16(drained[c_block], c_buffer, add_lane, c_block_bytes / 4, c_block);
                    }
                }
                fence();
            }
        }
    };

    // ---- prologue: A(0) rows 0-127, B(0), A(0) rows 128-255 and A(1) rows 0-127 (stage 1); zeroed AGPRs ------------
    for (int i = 0; i < Loads + 5; ++i) {
        const int d = i % Loads, stage = i >= Loads;
        window_m0(d, dma_lds[stage], scale_dma_lds[stage]);
        asm volatile("s_nop 0");                                       // M0 -> DMA
        window_dma(d, a_rsrc, a_scale_rsrc, dma_lane[d]);
        move(dma_lane[d], step_now[dma_tensor(d)]);
        for (int j = 0; i == 4 && j < 8; ++j) {     // B(0): tile j % 4, K half j / 4
            load(b_tile[0][j % BTiles][j / BTiles], b_rsrc, b_lane[j % BTiles], BHalfBytes * (j / BTiles));
            if (j >= BTiles) move(b_lane[j % BTiles], step_now[TensorB]);
        }
        for (int pair = 0; i == 4 && pair < 2; ++pair) {
            load(b_scale[0][pair], b_scale_rsrc, b_scale_lane[pair]);
            move(b_scale_lane[pair], step_now[TensorBScale]);
        }
    }
    for (int i = 0; i < 8; ++i) acc_zero(i, zero);
    asm volatile("s_nop 7\n\ts_nop 7");                                // zeroing MFMA -> MFMA
    wait(2 * Loads, 0);                                                // A(0) rows 0-127 landed (20 loads younger)
    barrier();
    ds_read(a_scales[0], scale_read[0], 0);                            // step 0's first A: scale vector 0, units 0-2
    for (int i = 0; i < 2 * ReadAhead; ++i) ds_read(a_regs[i / 2][i % 2], a_read[0][i % 2], ABlockBytes * (i / 2));

    // ---- per tile: its plain steps (parities 0, 1, 0, ..), the tail, the trailing block -----------------------------
    while (tile_m < tiles_m * TileM) {
        for (int plain_left = steps - 2; plain_left > 0; plain_left -= 2) {
#pragma unroll
            for (int parity = 0; parity < 2; ++parity) {
                if (parity == 1 && plain_left == 1) break;
                plain_units(parity, 0, 4, false);
                if (at_least(tile_m, m_past)) plain_units(parity, 4, Units, true);   // a short tile
                else plain_units(parity, 4, Units, false);
            }
        }

        // The tail's step a has parity 0: after odd plain steps, copy B(a) and swap the stages (one tail code copy).
        wait(Loads, 15);                                 // B(a) landed (the window's DMAs younger): LLVM may copy it
        if (steps % 2) {
            for (int i = 0; i < 2 * BTiles; ++i) b_tile[0][i / 2][i % 2] = b_tile[1][i / 2][i % 2];
            for (int pair = 0; pair < 2; ++pair) b_scale[0][pair] = b_scale[1][pair];
            std::swap(a_read[0], a_read[1]);
            std::swap(scale_read[0], scale_read[1]);
            std::swap(dma_lds[0], dma_lds[1]);
            std::swap(scale_dma_lds[0], scale_dma_lds[1]);
        }
        tail_units(0, Units, false);
        if (at_least(tile_m, m_past)) tail_units(Units, TailUnits, true);
        else tail_units(Units, TailUnits, false);

        // The trailing block: the next tile's first A reads, the tail's leftovers (m-blocks 9, 10) and 11-15: pack
        // (block), drain (block - 1), store (block - 2). AGPR group i (m-blocks 2i, 2i + 1): zeroed after block 11 + i.
        ds_read(a_scales[0], scale_read[0], 0);
        for (int i = 0; i < 2 * ReadAhead; ++i) ds_read(a_regs[i / 2][i % 2], a_read[0][i % 2], ABlockBytes * (i / 2));
#pragma unroll
        for (int block = PackedInTail; block < Units + 2; ++block) {   // + 2: the last drains and stores
            if (block < Units && __builtin_expect(16 * block < c_rows, 1)) {   // rows past M: no pack
                for (int tile = 0; tile <= BTiles; ++tile) {           // read tile n, convert tile n - 1
                    for (int i = 0; tile < BTiles && i < 4; ++i)
                        acc_read(acc_pairs[block][tile][i / 2], i % 2, 16 * block + 4 * tile + i);
                    for (int i = 0; tile > 0 && i < 2; ++i) acc_pairs[block][tile - 1][i] *= alpha;
                    for (int i = 0; tile > 0 && i < 2; ++i)
                        packed[block][(tile - 1) / 2][(tile - 1) % 2 * 2 + i] = to_bf16(acc_pairs[block][tile - 1][i]);
                    if (tile == 2 || tile == 4)
                        ds_write(park_lane[tile / 2 - 1], packed[block][tile / 2 - 1], block % 8 * 2 * RingBlockBytes);
                }
            }
            acc_zero(block - PackedInTail, zero);
            for (int c_block = 2 * block - 2; block - 1 < Units && c_block < 2 * block; ++c_block)
                ds_read_block(drained[c_block], block_read, c_block % 16 * RingBlockBytes);
            if (block - 2 < Units) wait(63, block < Units ? 2 : 0);    // block - 2's drains (the pack may be skipped)
            for (int c_block = 2 * block - 4; block - 2 < Units && c_block < 2 * block - 2; ++c_block) {
                store16(drained[c_block], c_buffer, c_lane, c_block_bytes, c_block);
                if (__builtin_expect(split, 0)) add16(drained[c_block], c_buffer, add_lane, c_block_bytes / 4, c_block);
            }
            fence();
        }
        acc_zero(7, zero);                                             // m-blocks 14, 15
        fence();

        ++round;
        tile_m = uniform(next_m);
        tile_n = uniform(next_n);
        if (!k_tail) std::swap(step_now, step_next);                   // the next round walks K the other way
    }
}

#ifndef MXG4W_NO_ENTRY   // other entries (aiter's f4gemm wrapper) include only run()
__global__ __launch_bounds__(Threads, 1) void gemm(Args p) {
    __shared__ __attribute__((aligned(1024))) uint8_t lds[LdsBytes];
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WaveSize), lane = threadIdx.x % WaveSize;
    asm volatile("" ::: "a255");                                         // the accumulators: all 256 AGPRs
    run(p, lds, wave, lane);
}

extern "C" int mxfp4_gemm(const void* a, const void* b, const void* sa, const void* sb, void* c, int m, int n, int k,
                          int a_stride, int sa_stride, int sb_stride, int c_stride, hipStream_t stream) {
    const int64_t tiles_m = (m + 255) / 256, tiles_n = (n + 255) / 256, scale_k = (k / 32 + 7) / 8 * 8;
    static int cus = 0, dev = 0;
    if (m <= 0 || n <= 0 || k < 64 || n % 16 || k % 64 || a_stride < k / 2 || sa_stride < scale_k
        || sb_stride < scale_k || c_stride < n || (int64_t)m * a_stride >= Oob || (int64_t)n * k / 2 >= Oob
        || tiles_m * 256 * sa_stride >= Oob || tiles_n * 256 * sb_stride >= Oob || (int64_t)256 * c_stride * 2 >= Oob
        || (!cus && (hipGetDevice(&dev) || hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev))))
        return -1;
    const Args args{a, b, sa, sb, (__bf16*)c, m, n, k, cus, a_stride, sa_stride, sb_stride, c_stride, 1.0f, 0};
    gemm<<<std::min<int64_t>(tiles_m * tiles_n, cus), Threads, 0, stream>>>(args);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
#endif

}  // namespace mxg4w

// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_tdm_gfx1250.cu
 * @brief Functional coverage for opus::tdm on gfx1250: every feature of the descriptor
 *        and its wrapper, checked byte-exactly.
 *
 * Each kernel isolates one feature and the host checks the bytes it moved, against a
 * position hash rather than a reference kernel -- a DMA either put the bytes where it was
 * asked to or it did not, so there is no tolerance to pick. Untouched memory is checked
 * just as hard: the destination and the LDS dumps are pre-filled with sentinels and the
 * sentinel has to survive wherever the transfer was supposed to stop, which is the only
 * way clamping, a short gather list, the tile_dim1 patch and padding are visible at all.
 *
 * What is covered, and where gcnasm/opus_tdm_example does not reach:
 *
 *   element sizes    1/2/4/8 bytes plus the sub-byte pack array<fp4_t,2>   (example: bf16 only)
 *   rank             2, 3, 4, 5, each with a move along its OUTERMOST dim  (example: 2 and 3)
 *   gather           <16> and <32>, at the list maximum and short of it    (example: none)
 *   scatter          a gather-mode async_store                             (example: none)
 *   set_tile_dim1    runtime row-count patch                               (example: none)
 *   multicast        compile-time tag, and peers_along_y                   (example: runtime peers_along_x)
 *   padding          on a 4-byte element, so the tag's byte scaling moves  (example: bf16, where it does not)
 *   make_from_layout called directly, in D# order                          (example: only via make_tdm)
 *   move             backwards as well as forwards                         (example: forwards only)
 *   clamp            ragged, and wholly past the end                       (example: ragged only)
 *   cache policy     load and store spellings on live windows              (example: same)
 *
 * Deliberately small: every kernel runs on one wave and a handful of workgroups. The
 * gather hang investigated separately needs thousands of workgroups' worth of tensor DMAs
 * outstanding machine-wide to reproduce, so nothing here goes near it.
 *
 * Entry point is run_tdm_feature_suite(), which allocates its own buffers, prints a line
 * per check and returns the number that failed. The suite validates on this side rather
 * than in the Python harness because the expectations are byte-exact against tile
 * geometry, and restating that geometry in torch would give two descriptions of one
 * layout that have to be kept in agreement.
 *
 * gcnasm/opus_tdm_test links this same file against a main(), and adds the compile-time
 * half of the coverage: the diagnostics that must NOT compile.
 */

#include <opus/hip_minimal.hpp>
#include <cstddef>

// ──────────────────────── geometry, needed by both passes ─────────────────────────

__host__ __device__ constexpr inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct tdm_test_config {
    // ── the source arena ────────────────────────────────────────────────────────
    // Row-major, ROW_BYTES per row. An element type of size S sees ROW_BYTES / S
    // elements per row, so the same arena is a valid tensor for every S under test.
    static constexpr int ROWS      = 128;
    static constexpr int ROW_BYTES = 512;
    static constexpr int SRC_BYTES = ROWS * ROW_BYTES;   // 64 KiB

    // ── the tile, stated in BYTES so LDS sizes are the same for every element type ──
    static constexpr int TILE_ROW_BYTES = 256;
    static constexpr int TILE_ROWS      = 8;
    static constexpr int TILE_BYTES     = TILE_ROW_BYTES * TILE_ROWS;   // 2 KiB

    // Raw LDS image written back by the kernels that are checked through LDS rather
    // than through a round trip. Sized for the largest of them (gather<16>, 16 rows).
    static constexpr int DUMP_BYTES = 8192;

    // One wave per workgroup: the tensor DMA opcodes are wave-level, so a second wave
    // would only issue the same transfer again.
    static constexpr int BLOCK_SIZE = 32;

    // ── gather / scatter ────────────────────────────────────────────────────────
    // Groups 2/3 are eight dwords, so a 32-bit index list tops out at 8 rows and a
    // 16-bit one at 16. Both maxima are exercised.
    //
    // The indices must be STRICTLY INCREASING -- a hardware requirement for gather
    // mode, not an opus one, so nothing in the header will catch a list that is not.
    static constexpr int GATHER32_N = 8;
    static constexpr int GATHER16_N = 16;
    // Fewer indices than the list can hold, to pin down that tile_dim1 follows the
    // count handed to set_indices() rather than the TileShape's second extent.
    static constexpr int GATHER_PARTIAL_N = 5;
    static constexpr int GATHER_MAX_N     = GATHER16_N;

    // ── padding, on a 4-byte element ────────────────────────────────────────────
    // The example covers bf16. Repeating it at 4 bytes is what actually exercises the
    // tag's byte scaling: the same "one 16-byte read vector per row" recipe has to come
    // out as a different element count and a different pair of encoded D# fields.
    static constexpr int PAD_ELEM_BYTES = 4;
    static constexpr int PAD_TILE0      = TILE_ROW_BYTES / PAD_ELEM_BYTES;   // 64 u32 = 256 B
    static constexpr int PAD_ELEMS      = 16 / PAD_ELEM_BYTES;               // 4 u32 = one b128
    static constexpr int PAD_PITCH      = PAD_TILE0 + PAD_ELEMS;             // 68 elements
    static constexpr int PAD_LDS_BYTES  = PAD_PITCH * TILE_ROWS * PAD_ELEM_BYTES;   // 2176

    // ── rank 2..5, on a 2-byte element ──────────────────────────────────────────
    // The strides are deliberately coprime-ish multiples of a row so that pairing a
    // delta with the wrong dimension's stride lands somewhere that cannot be mistaken
    // for the right answer. Every extent past dim0 is small, because what is under test
    // is the stride programming, not the volume.
    static constexpr int RK_ELEM_BYTES = 2;
    static constexpr int RK_ROW_ELEMS  = ROW_BYTES / RK_ELEM_BYTES;   // 256 u16 per row
    static constexpr int RK_D0 = 8;    // contiguous elements per tile row
    static constexpr int RK_DN = 2;    // extent of every dimension past dim0
    // Rows between successive steps of dimensions 1..4.
    static constexpr int RK_S1 = 1;
    static constexpr int RK_S2 = 3;
    static constexpr int RK_S3 = 12;
    static constexpr int RK_S4 = 28;
    // Tensor extent of every dimension past dim0; only has to leave room for the tile
    // plus the one move, so nothing clamps.
    static constexpr int RK_EXTENT = 4;
    static constexpr int RK_STEPS  = 2;   // load, move one step along the OUTERMOST dim, load again

    // Elements in a rank-R tile: D0 * DN^(R-1).
    static constexpr int RK_ELEMS_2 = RK_D0 * RK_DN;
    static constexpr int RK_ELEMS_3 = RK_ELEMS_2 * RK_DN;
    static constexpr int RK_ELEMS_4 = RK_ELEMS_3 * RK_DN;
    static constexpr int RK_ELEMS_5 = RK_ELEMS_4 * RK_DN;
    static constexpr int RK_ELEMS_MAX = RK_ELEMS_5;   // 128 u16 = 256 B

    // ── runtime tile_dim1 patch ─────────────────────────────────────────────────
    static constexpr int TILE_DIM1_OVERRIDE = 3;   // < TILE_ROWS, so the tail must stay untouched

    // ── multicast ───────────────────────────────────────────────────────────────
    // Two peers is the smallest fan-out that is a multicast at all: one peer has to be
    // folded to a zero mask and is rejected by the compile-time tag.
    static constexpr int CLUSTER_PEERS = 2;

    // ── descriptor split / move ─────────────────────────────────────────────────
    // Three LDS slots: tile 0, tile 1, then tile 0 again reached by moving BACKWARDS.
    static constexpr int SPLIT_SLOTS = 3;

    // Written into LDS before a load so a dump can distinguish "the DMA skipped this"
    // from "the DMA wrote this". Both nibbles equal, so a byte fill produces it.
    static constexpr unsigned char LDS_FILL = 0xA5;
    // Same idea for the destination arena.
    static constexpr unsigned char DST_FILL = 0x3C;
};

// One kargs for every kernel; each uses the subset it needs. Extents and strides are in
// ELEMENTS of whatever type that kernel instantiates, matching every offset opus takes.
struct tdm_test_kargs {
    const void* __restrict__ src;
    void* __restrict__       dst;
    void* __restrict__       dump;      // raw LDS image, for the checks that read LDS directly
    const int* __restrict__  indices;   // gather/scatter row-index list, strictly increasing

    int shape0;    // whole-tensor contiguous extent, in elements
    int shape1;    // whole-tensor row count
    int stride0;   // row pitch, in elements
    int origin0;   // tile origin along the contiguous axis, in elements
    int origin1;   // tile origin along rows
    int n_indices;

    int replica_stride;   // elements between output replicas, multicast kernels only
};

// The kernel roster, in one place. Three separate builds need the same list spelled
// differently -- real definitions in the device pass, empty stubs so a host pass can
// form launch symbols, and declarations for the driver -- and a name that reached only
// two of the three would show up as a link error rather than a missing check. The
// includer supplies X.
//
//     #define X(name) __global__ void name(tdm_test_kargs);
//     TDM_FEATURE_KERNELS(X)
//     #undef X
#define TDM_FEATURE_KERNELS(X) \
    X(test_elem_b1) X(test_elem_b2) X(test_elem_b4) X(test_elem_b8) X(test_elem_fp4) \
    X(test_rank2) X(test_rank3) X(test_rank4) X(test_rank5) \
    X(test_gather16) X(test_gather16_pair) X(test_gather16_bit30) X(test_gather16_bit31) \
    X(test_gather32) X(test_gather32_pair) X(test_gather_partial) X(test_scatter) \
    X(test_tile_dim1) X(test_multicast_static) X(test_multicast_peers_y) \
    X(test_padded) X(test_split_move)

// Returned by run_tdm_feature_suite() instead of a failure count when the suite did not
// run at all. Negative so it cannot be confused with one, and so a caller that ignores
// the distinction still does not read it as "everything passed".
static constexpr int TDM_FEATURE_SKIPPED = -1;

#ifdef __HIP_DEVICE_COMPILE__
// ──────────────────────────────── Device pass ─────────────────────────────────────
#if defined(__gfx1250__)

#include <opus/opus.hpp>

namespace tdm_test {

using C = tdm_test_config;
namespace tt = opus::tdm_traits;
using opus::operator""_I;

// ── LDS helpers ─────────────────────────────────────────────────────────────────
// Both are the test's own instrumentation; nothing about the DMA needs them. Every
// size below is a multiple of 4, so a dword-strided lane loop covers them exactly.
__device__ inline void fill_lds(void* lds, int bytes, unsigned char v) {
    unsigned int* p = reinterpret_cast<unsigned int*>(lds);
    const unsigned int word = 0x01010101u * (unsigned int)v;
    for (int i = (int)opus::lane_id(); i < bytes / 4; i += C::BLOCK_SIZE) p[i] = word;
    opus::s_wait_dscnt(opus::number<0>{});   // must land before the DMA overwrites part of it
}

__device__ inline void dump_lds(void* dst, const void* lds, int bytes) {
    if (dst == nullptr) return;
    unsigned int* d = reinterpret_cast<unsigned int*>(dst);
    const unsigned int* s = reinterpret_cast<const unsigned int*>(lds);
    for (int i = (int)opus::lane_id(); i < bytes / 4; i += C::BLOCK_SIZE) d[i] = s[i];
}

// ── 1. element sizes ────────────────────────────────────────────────────────────
// D#.data_size is log2 of the element size in bytes and encodes only 1/2/4/8, so those
// four are the whole legal set; the fifth instantiation is the sub-byte pack, which is
// legal only because array<fp4_t,2> is one whole byte.
//
// The tile is stated in bytes and divided by sizeof(T), so every instantiation moves the
// same 2 KiB and the host compares the same byte range. Anything that scaled by the
// wrong element size would land the tile somewhere else.
template<typename T>
__device__ void elem_copy_body(tdm_test_kargs k) {
    constexpr int kTile0 = C::TILE_ROW_BYTES / (int)sizeof(T);
    static_assert(kTile0 * (int)sizeof(T) == C::TILE_ROW_BYTES, "tile row must divide evenly into elements");
    using Win = opus::tdm<T, opus::seq<kTile0, C::TILE_ROWS>>;

    __shared__ char lds[C::TILE_BYTES];
    const opus::u32_t base = (opus::u32_t)reinterpret_cast<__UINTPTR_TYPE__>(lds);

    // Flat 2D factory. Both windows are told the WHOLE tensor's shape plus this tile's
    // origin and clamp internally, so the ragged and past-the-end cases the host drives
    // through kargs need nothing here.
    auto in = opus::make_tdm<Win>(base, k.src, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                  (opus::u64_t)k.stride0, (opus::u32_t)k.origin0, (opus::u32_t)k.origin1);
    auto out = opus::make_tdm<Win>(base, k.dst, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                   (opus::u64_t)k.stride0, (opus::u32_t)k.origin0, (opus::u32_t)k.origin1);

    in.async_load();
    opus::s_wait_tensorcnt<0>();
    out.async_store();
    opus::s_wait_tensorcnt<0>();
}

// One byte, and the packed pair that is also one byte. The pack is the interesting one:
// it is the only spelling of a sub-byte element the window accepts, and it has to stride
// by the byte rather than by the logical 4-bit value.
using fp4x2_t = opus::array<opus::fp4_t, 2>;
static_assert(sizeof(fp4x2_t) == 1, "array<fp4_t,2> must be exactly the one byte the D# strides by");
static_assert(opus::sizeof_bits_v<opus::fp4_t> == 4, "fp4_t is one 4-bit logical element");

// ── 2. rank 2..5 ────────────────────────────────────────────────────────────────
// Rank comes from TileShape alone. Each dimension past dim0 is reached by its own
// stride, and only a >2D descriptor programs the second and later ones -- a dimension
// whose stride is missing or paired with the wrong delta lands on top of its neighbour,
// which the extents alone would never reveal.
//
// make_from_layout() is used directly here rather than through make_tdm(): it takes
// shape/pitch/coord already in D# ORDER (index 0 = fastest), which is the one entry
// point that does no dimension reversal. pitch[N] is the stride that advances
// dimension N+1.
using rk_t = unsigned short;

template<int Rank> struct rank_window;
template<> struct rank_window<2> { using type = opus::tdm<rk_t, opus::seq<C::RK_D0, C::RK_DN>>; };
template<> struct rank_window<3> { using type = opus::tdm<rk_t, opus::seq<C::RK_D0, C::RK_DN, C::RK_DN>>; };
template<> struct rank_window<4> { using type = opus::tdm<rk_t, opus::seq<C::RK_D0, C::RK_DN, C::RK_DN, C::RK_DN>>; };
template<> struct rank_window<5> { using type = opus::tdm<rk_t, opus::seq<C::RK_D0, C::RK_DN, C::RK_DN, C::RK_DN, C::RK_DN>>; };

template<int Rank>
__device__ void rank_body(tdm_test_kargs k) {
    using Win = typename rank_window<Rank>::type;
    static_assert(Win::ndim == (opus::u32_t)Rank, "TileShape rank must be the window's rank");
    constexpr int kElems = C::RK_D0 * (Rank >= 2 ? C::RK_DN : 1) * (Rank >= 3 ? C::RK_DN : 1) *
                                      (Rank >= 4 ? C::RK_DN : 1) * (Rank >= 5 ? C::RK_DN : 1);

    __shared__ char lds[C::RK_STEPS * C::RK_ELEMS_MAX * C::RK_ELEM_BYTES];
    const opus::u32_t base = (opus::u32_t)reinterpret_cast<__UINTPTR_TYPE__>(lds);

    // Rows between successive steps of dimensions 1..4, turned into element pitches.
    constexpr int kRowStep[4] = {C::RK_S1, C::RK_S2, C::RK_S3, C::RK_S4};

    opus::u32_t shape[Rank], coord[Rank];
    opus::u64_t pitch[Rank - 1];
    shape[0] = (opus::u32_t)C::RK_ROW_ELEMS;
    coord[0] = 0;
    opus::static_for<Rank - 1>([&](auto N) {
        shape[N.value + 1] = (opus::u32_t)C::RK_EXTENT;
        coord[N.value + 1] = 0;
        pitch[N.value] = (opus::u64_t)(kRowStep[N.value] * C::RK_ROW_ELEMS);
    });

    Win w;
    w.make_from_layout(base, k.src, shape, pitch, coord);

    // Load, step ONE unit along the outermost dimension, load again. The pair is what
    // pins the outermost stride: a resting tile only shows that the dimension is spread
    // correctly, and only the move shows that a delta on it advances by that same pitch.
    w.async_load(0);
    if constexpr (Rank == 2)      w.move(0_I, 1_I);
    else if constexpr (Rank == 3) w.move(0_I, 0_I, 1_I);
    else if constexpr (Rank == 4) w.move(0_I, 0_I, 0_I, 1_I);
    else                          w.move(0_I, 0_I, 0_I, 0_I, 1_I);
    w.async_load(kElems);
    opus::s_wait_tensorcnt<0>();

    dump_lds(k.dump, lds, C::RK_STEPS * kElems * C::RK_ELEM_BYTES);
}

// ── 3. gather ───────────────────────────────────────────────────────────────────
// In gather mode groups 2/3 stop carrying dim2..4 and carry the row-index list instead
// (which is why opus static_asserts gather against ndim > 2), and tile_dim1 stops
// meaning "rows in the tile" and becomes "how many indices are valid". The TileShape's
// second extent is therefore only a placeholder that make_descriptor() overwrites.
//
// The index IS the dim1 coordinate, so origin1 must stay 0: a row origin would bias
// every gathered row by it. The contiguous origin is an ordinary dim0 offset.
using gath_t = unsigned short;
constexpr int kGatherTile0 = C::TILE_ROW_BYTES / (int)sizeof(gath_t);

// The two mode bits, pinned to their literals. group0 dword0 is count[1:0]=1 (Valid
// Tensor) plus index_size at bit 30 and gather_mode at bit 31, which is what the
// hardware honours -- established by the encoding probe below, not by the MLIR lowering,
// which has the two the other way round.
//
// Nothing downstream complains about a transposition: it turns gather<16> into a plain
// contiguous copy that still moves the right number of bytes, just from the wrong rows.
// And gather<32> is 0xC0000001 whichever way round the pair goes, so only gather<16>
// can catch it.
template<int Bits> using gather_probe = opus::tdm<gath_t, opus::seq<kGatherTile0, 8>, tt::gather<Bits>>;
static_assert(gather_probe<16>::descriptor::group0_dword0_const == 0x80000001u,
              "gather<16> must set gather_mode (bit 31) and leave index_size (bit 30) clear");
static_assert(gather_probe<32>::descriptor::group0_dword0_const == 0xC0000001u,
              "gather<32> must set both gather_mode (bit 31) and index_size (bit 30)");
static_assert(opus::tdm<gath_t, opus::seq<kGatherTile0, 8>>::descriptor::group0_dword0_const == 0x00000001u,
              "a non-gather tile must leave both mode bits clear");

template<int IndexBits, int MaxN>
__device__ void gather_body(tdm_test_kargs k, int n) {
    using Win = opus::tdm<gath_t, opus::seq<kGatherTile0, MaxN>, tt::gather<IndexBits>>;
    static_assert(Win::is_gather, "the gather tag must reach the window");
    static_assert(Win::needs_groups23, "gather always needs groups 2/3 for the index list");

    __shared__ char lds[C::GATHER_MAX_N * C::TILE_ROW_BYTES];
    fill_lds(lds, C::GATHER_MAX_N * C::TILE_ROW_BYTES, C::LDS_FILL);

    const opus::u32_t base = (opus::u32_t)reinterpret_cast<__UINTPTR_TYPE__>(lds);
    auto w = opus::make_tdm<Win>(base, k.src, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                 (opus::u64_t)k.stride0, (opus::u32_t)k.origin0, 0u);

    // The list is uniform (one pointer, compile-time subscripts), so these stay scalar
    // loads and the descriptor stays in SGPRs.
    int idx[MaxN];
    opus::static_for<MaxN>([&](auto I) { idx[I.value] = k.indices[I.value]; });
    w.set_indices(idx, n);

    w.async_load();
    opus::s_wait_tensorcnt<0>();
    dump_lds(k.dump, lds, C::GATHER_MAX_N * C::TILE_ROW_BYTES);
}

// ── 4. scatter: a gather-mode store ─────────────────────────────────────────────
// The same descriptor drives both directions, so a gather window's async_store writes
// LDS row j out to the global row named by index j. Rows are staged into LDS by a plain
// consecutive load first, so what the scatter moves is known independently of it.
__device__ void scatter_body(tdm_test_kargs k) {
    constexpr int N = C::GATHER32_N;
    using LoadWin = opus::tdm<gath_t, opus::seq<kGatherTile0, N>>;
    // Store-side cache policy: th=7 (TH_STORE_NT_WB) exists on the store side alone, so
    // this also pins that the store spelling is the one that reaches the instruction.
    constexpr int kStorePolicy = tt::make_cache_policy(tt::store_temporal_hint::non_temporal_write_back,
                                                       tt::scope::dev);
    using ScatterWin = opus::tdm<gath_t, opus::seq<kGatherTile0, N>, tt::gather<32>, tt::cache<kStorePolicy>>;

    __shared__ char lds[N * C::TILE_ROW_BYTES];
    const opus::u32_t base = (opus::u32_t)reinterpret_cast<__UINTPTR_TYPE__>(lds);

    // Stage rows 0..N-1, consecutive, from the source.
    auto in = opus::make_tdm<LoadWin>(base, k.src, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                      (opus::u64_t)k.stride0, 0u, 0u);
    in.async_load();
    opus::s_wait_tensorcnt<0>();

    auto out = opus::make_tdm<ScatterWin>(base, k.dst, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                          (opus::u64_t)k.stride0, 0u, 0u);
    int idx[N];
    opus::static_for<N>([&](auto I) { idx[I.value] = k.indices[I.value]; });
    out.set_indices(idx, N);

    out.async_store();
    opus::s_wait_tensorcnt<0>();
}

// ── 5. runtime tile_dim1 ────────────────────────────────────────────────────────
// The one tile extent worth patching at runtime: it lets a single window type serve two
// operands that differ only in row count. Under test is that the patch actually shortens
// the transfer -- the rows past the override must be left as the fill found them.
__device__ void tile_dim1_body(tdm_test_kargs k) {
    using Win = opus::tdm<gath_t, opus::seq<kGatherTile0, C::TILE_ROWS>>;

    __shared__ char lds[C::TILE_BYTES];
    fill_lds(lds, C::TILE_BYTES, C::LDS_FILL);

    const opus::u32_t base = (opus::u32_t)reinterpret_cast<__UINTPTR_TYPE__>(lds);
    auto w = opus::make_tdm<Win>(base, k.src, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                 (opus::u64_t)k.stride0, 0u, 0u);
    w.set_tile_dim1((opus::u32_t)C::TILE_DIM1_OVERRIDE);

    w.async_load();
    opus::s_wait_tensorcnt<0>();
    dump_lds(k.dump, lds, C::TILE_BYTES);
}

// ── 6. multicast ────────────────────────────────────────────────────────────────
// The peers that share a tile issue the same load with the same mask and GL1 merges the
// requests into one data return that lands in each of their LDS. Each peer then writes
// the tile to its own replica of the output and the host compares the replicas, which is
// what makes a merged return observable without a profiler.
//
// Two spellings, and they reach the D# by different routes: the compile-time tag bakes
// the mask into group1's initializer, while peers_along_* computes it from this
// workgroup's position and goes in through set_workgroup_mask().
using mc_t = unsigned short;
constexpr int kMcTile0 = C::TILE_ROW_BYTES / (int)sizeof(mc_t);

// Compile-time peer set. Mask bit index is y * ClusterDimX + x, so with a 2x1 cluster the
// two peers are bits 0 and 1.
using McPlainWin  = opus::tdm<mc_t, opus::seq<kMcTile0, C::TILE_ROWS>>;
using McStaticWin = opus::tdm<mc_t, opus::seq<kMcTile0, C::TILE_ROWS>, tt::multicast<0, 1>>;

// UseStatic picks which of the two routes to the mask is under test; it is a template
// parameter rather than an argument so neither path carries a branch the other would
// have to be uniform across.
template<bool UseStatic>
__device__ void multicast_body(tdm_test_kargs k, int replica, tt::mask runtime_mask) {
    __shared__ char lds[C::TILE_BYTES];
    const opus::u32_t base = (opus::u32_t)reinterpret_cast<__UINTPTR_TYPE__>(lds);

    // Cluster-wide barrier (-3; -1 is workgroup scope) before the first multicast, so no
    // peer issues its half of a merged request before the others exist.
    __builtin_amdgcn_s_barrier_signal(-3);
    __builtin_amdgcn_s_barrier_wait(-3);

    if constexpr (UseStatic) {
        auto w = opus::make_tdm<McStaticWin>(base, k.src, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                             (opus::u64_t)k.stride0, 0u, 0u);
        w.async_load();
    } else {
        auto w = opus::make_tdm<McPlainWin>(base, k.src, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                            (opus::u64_t)k.stride0, 0u, 0u);
        w.set_workgroup_mask(runtime_mask);
        w.async_load();
    }
    opus::s_wait_tensorcnt<0>();

    // Each peer writes the tile back through a plain window into its own replica of the
    // output, so what the host compares is what actually reached that peer's LDS.
    mc_t* dst = reinterpret_cast<mc_t*>(k.dst) + (size_t)replica * k.replica_stride;
    auto o = opus::make_tdm<McPlainWin>(base, dst, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                        (opus::u64_t)k.stride0, 0u, 0u);
    o.async_store();
    opus::s_wait_tensorcnt<0>();
}

// ── 7. padding on a 4-byte element ──────────────────────────────────────────────
// "After every PAD_TILE0 elements written, skip PAD_ELEMS elements", both ELEMENT
// counts. The D# wants a log-scale DWORD interval and a DWORD count minus one; the tag
// does that arithmetic, and doing it against a 4-byte element rather than bf16 is what
// makes the encoded pair differ from the example's.
using pad_t = unsigned int;
using PaddingManual = tt::padding<pad_t, C::PAD_TILE0, C::PAD_ELEMS>;
using PaddingAuto   = tt::padding_auto<pad_t, C::PAD_TILE0>;
static_assert(opus::is_same_v<PaddingManual, PaddingAuto>,
              "the auto tier is shorthand for the manual one whenever the pad is one 16-byte vector");
static_assert(PaddingManual::pitch_elements == C::PAD_PITCH,
              "the LDS row pitch the host places elements with must be the one the D# programs");
// 256-byte row, 16-byte pad: interval encodes as ctz(256)-3 = 5, amount as 16/4-1 = 3.
// The example's bf16 tile encodes as (4, 3) for the same recipe, so this pins that the
// tag scaled by the element size rather than by a constant.
static_assert(PaddingManual::encoded_interval == 5 && PaddingManual::encoded_amount == 3,
              "u32 64-element row + 4-element pad must encode as pad_interval=5, pad_amount=3");

// Load-side cache policy, so the issued instruction carries th:TH_LOAD_NT scope:SCOPE_SYS.
constexpr int kLoadPolicy = tt::make_cache_policy(tt::load_temporal_hint::non_temporal, tt::scope::sys);
using PaddedWin = opus::tdm<pad_t, opus::seq<C::PAD_TILE0, C::TILE_ROWS>, PaddingAuto, tt::cache<kLoadPolicy>>;

__device__ void padded_body(tdm_test_kargs k) {
    __shared__ char lds[C::PAD_LDS_BYTES];
    fill_lds(lds, C::PAD_LDS_BYTES, C::LDS_FILL);

    const opus::u32_t base = (opus::u32_t)reinterpret_cast<__UINTPTR_TYPE__>(lds);
    auto w = opus::make_tdm<PaddedWin>(base, k.src, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                       (opus::u64_t)k.stride0, 0u, 0u);
    w.async_load();
    opus::s_wait_tensorcnt<0>();
    dump_lds(k.dump, lds, C::PAD_LDS_BYTES);
}

// ── 8. descriptor split, and moving backwards ───────────────────────────────────
// make_descriptor() and async_load() are separate calls so a caller can put unrelated
// work between them. Three tiles are taken: tile 0, tile 1 reached by a forward move,
// and tile 0 again reached by moving BACK. The third is the point -- a move that only
// ever accumulated forwards, or that dropped the sign, still passes the first two.
__device__ void split_move_body(tdm_test_kargs k) {
    using Win = opus::tdm<gath_t, opus::seq<kGatherTile0, C::TILE_ROWS>>;
    constexpr int kSlotElems = kGatherTile0 * C::TILE_ROWS;

    __shared__ char lds[C::SPLIT_SLOTS * C::TILE_BYTES];
    const opus::u32_t base = (opus::u32_t)reinterpret_cast<__UINTPTR_TYPE__>(lds);

    auto w = opus::make_tdm<Win>(base, k.src, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                 (opus::u64_t)k.stride0, 0u, 0u);

    // Slot 0: tile 0, built and issued as two separate steps.
    typename Win::descriptor d0 = w.make_descriptor(0);
    Win::async_load(d0);

    // Slot 1: one tile forward along the contiguous axis.
    w.move(kGatherTile0);
    typename Win::descriptor d1 = w.make_descriptor(kSlotElems);
    Win::async_load(d1);

    // Slot 2: back where we started. A runtime negative delta, so nothing about it is
    // folded away at compile time.
    w.move(-kGatherTile0);
    typename Win::descriptor d2 = w.make_descriptor(2 * kSlotElems);
    Win::async_load(d2);

    opus::s_wait_tensorcnt<0>();
    dump_lds(k.dump, lds, C::SPLIT_SLOTS * C::TILE_BYTES);
}

// ── entry points ────────────────────────────────────────────────────────────────
#define TDM_TEST_ELEM(name, type)                     \
    __global__ __launch_bounds__(C::BLOCK_SIZE)       \
    void name(tdm_test_kargs k) { elem_copy_body<type>(k); }

TDM_TEST_ELEM(test_elem_b1,  unsigned char)
TDM_TEST_ELEM(test_elem_b2,  unsigned short)
TDM_TEST_ELEM(test_elem_b4,  unsigned int)
TDM_TEST_ELEM(test_elem_b8,  unsigned long long)
TDM_TEST_ELEM(test_elem_fp4, fp4x2_t)
#undef TDM_TEST_ELEM

#define TDM_TEST_RANK(name, rank)               \
    __global__ __launch_bounds__(C::BLOCK_SIZE) \
    void name(tdm_test_kargs k) { rank_body<rank>(k); }

TDM_TEST_RANK(test_rank2, 2)
TDM_TEST_RANK(test_rank3, 3)
TDM_TEST_RANK(test_rank4, 4)
TDM_TEST_RANK(test_rank5, 5)
#undef TDM_TEST_RANK

__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_gather16(tdm_test_kargs k) { gather_body<16, C::GATHER16_N>(k, C::GATHER16_N); }

// Two indices: one dword of group 2, the smallest 16-bit list there is.
__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_gather16_pair(tdm_test_kargs k) { gather_body<16, C::GATHER16_N>(k, 2); }

// ── which bit is gather_mode? ───────────────────────────────────────────────────
// The two candidate orderings of group0 dword0 differ ONLY for a 16-bit gather:
//
//     index_size[30]  | gather_mode[31]  ->  gather<16> 0x80000001, gather<32> 0xC0000001   <- what the hardware honours
//     gather_mode[30] | index_size[31]   ->  gather<16> 0x40000001, gather<32> 0xC0000001   <- what MLIR emits
//
// gather<32> is the same constant either way, which is why a 32-bit-only workload can
// never tell them apart, and why the question stayed open until this suite ran a 16-bit
// list. The probe builds the descriptor through the ordinary opus path and overrides
// that one dword just before the issue, so the two encodings are compared with every
// other field, the index list and the LDS layout held identical.
//
// Kept as a live check rather than a note, because a transposition is silent: it costs
// gather<16> its rows and leaves gather<32> working, so nothing else in this suite --
// or in any 32-bit workload -- would go red.
template<opus::u32_t Group0Dword0, int N>
__device__ void gather_bit_probe_body(tdm_test_kargs k) {
    using Win = opus::tdm<gath_t, opus::seq<kGatherTile0, C::GATHER16_N>, tt::gather<16>>;

    __shared__ char lds[C::GATHER_MAX_N * C::TILE_ROW_BYTES];
    fill_lds(lds, C::GATHER_MAX_N * C::TILE_ROW_BYTES, C::LDS_FILL);

    const opus::u32_t base = (opus::u32_t)reinterpret_cast<__UINTPTR_TYPE__>(lds);
    auto w = opus::make_tdm<Win>(base, k.src, (opus::u32_t)k.shape0, (opus::u32_t)k.shape1,
                                 (opus::u64_t)k.stride0, 0u, 0u);
    int idx[N];
    opus::static_for<N>([&](auto I) { idx[I.value] = k.indices[I.value]; });
    w.set_indices(idx, N);

    typename Win::descriptor d = w.make_descriptor(0);
    d.group0[0] = (opus::i32_t)Group0Dword0;
    Win::async_load(d);
    opus::s_wait_tensorcnt<0>();
    dump_lds(k.dump, lds, C::GATHER_MAX_N * C::TILE_ROW_BYTES);
}

// The ordering opus emits; must gather the listed rows.
__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_gather16_bit31(tdm_test_kargs k) { gather_bit_probe_body<0x80000001u, C::GATHER32_N>(k); }

// The transposed ordering; must NOT gather. Asserting this half too is what keeps the
// probe honest: if a future part honoured both, the pair would stop discriminating and
// the check above would no longer be evidence of anything.
__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_gather16_bit30(tdm_test_kargs k) { gather_bit_probe_body<0x40000001u, C::GATHER32_N>(k); }

// The 32-bit control at the same length.
__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_gather32_pair(tdm_test_kargs k) { gather_body<32, C::GATHER32_N>(k, 2); }

__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_gather32(tdm_test_kargs k) { gather_body<32, C::GATHER32_N>(k, C::GATHER32_N); }

// Same window as test_gather32, but told that only the first few indices are valid.
__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_gather_partial(tdm_test_kargs k) { gather_body<32, C::GATHER32_N>(k, C::GATHER_PARTIAL_N); }

__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_scatter(tdm_test_kargs k) { scatter_body(k); }

__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_tile_dim1(tdm_test_kargs k) { tile_dim1_body(k); }

__global__ __launch_bounds__(C::BLOCK_SIZE)
__cluster_dims__(C::CLUSTER_PEERS, 1, 1)
void test_multicast_static(tdm_test_kargs k) {
    multicast_body<true>(k, (int)__builtin_amdgcn_cluster_workgroup_id_x(), tt::mask{0});
}

__global__ __launch_bounds__(C::BLOCK_SIZE)
__cluster_dims__(1, C::CLUSTER_PEERS, 1)
void test_multicast_peers_y(tdm_test_kargs k) {
    multicast_body<false>(k, (int)__builtin_amdgcn_cluster_workgroup_id_y(),
                          tt::peers_along_y<1, C::CLUSTER_PEERS>());
}

__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_padded(tdm_test_kargs k) { padded_body(k); }

__global__ __launch_bounds__(C::BLOCK_SIZE)
void test_split_move(tdm_test_kargs k) { split_move_body(k); }

// Reports which device pass was actually compiled. See the host-side gate for why the
// arch is established this way rather than by asking the runtime.
__global__ void probe_gfx1250(int* out) { *out = 1; }

}  // namespace tdm_test

#else   // the tensor DMA opcodes exist only on gfx1250
namespace tdm_test {
#define X(name) __global__ void name(tdm_test_kargs) {}
TDM_FEATURE_KERNELS(X)
#undef X
__global__ void probe_gfx1250(int* out) { *out = 0; }
}  // namespace tdm_test
#endif  // __gfx1250__

#else
// ───────────────────────────────── Host pass ──────────────────────────────────────
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

// Empty bodies so the host pass has a launch symbol per kernel; the real ones are
// emitted by the device pass above.
namespace tdm_test {
#define X(name) __global__ void name(tdm_test_kargs) {}
TDM_FEATURE_KERNELS(X)
#undef X
__global__ void probe_gfx1250(int*) {}
}  // namespace tdm_test

// Reports and gives up rather than returning a code: every allocation here happens before
// the first check, so a failure means the suite never ran, which is not a result to fold
// into a pass/fail count.
#define CHECK_HIP(call)                                                                                 \
    do {                                                                                                \
        hipError_t status_ = call;                                                                      \
        if (status_ != hipSuccess) {                                                                    \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_)); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

using C = tdm_test_config;

// A position hash rather than a counter: two different offsets holding the same byte is
// then a 1-in-256 accident instead of a systematic aliasing of every 256th byte, so a
// whole row landing at the wrong address cannot go unnoticed.
static inline unsigned char pat(size_t off) {
    return (unsigned char)(((unsigned int)off * 2654435761u) >> 16);
}

// ── the reporting harness ───────────────────────────────────────────────────────
static int g_failures = 0;
static int g_passes   = 0;

struct cmp_result {
    int           bad   = 0;
    size_t        first = 0;
    unsigned char got   = 0;
    unsigned char want  = 0;
};

// `want(i)` is the byte that offset i of `got` must hold.
template <typename Fn>
static cmp_result compare(const unsigned char* got, size_t n, Fn want) {
    cmp_result r;
    for (size_t i = 0; i < n; ++i) {
        const unsigned char w = want(i);
        if (got[i] != w) {
            if (r.bad == 0) { r.first = i; r.got = got[i]; r.want = w; }
            ++r.bad;
        }
    }
    return r;
}

static void report(const char* name, const cmp_result& r, const char* note = "") {
    if (r.bad == 0) {
        printf("  %-36s PASS  %s\n", name, note);
        ++g_passes;
    } else {
        printf("  %-36s FAIL  %d bad bytes, first at 0x%05zx: got 0x%02x want 0x%02x\n",
               name, r.bad, r.first, r.got, r.want);
        ++g_failures;
    }
}

// ── device state ────────────────────────────────────────────────────────────────
static unsigned char* dev_src  = nullptr;
static unsigned char* dev_dst  = nullptr;
static unsigned char* dev_dump = nullptr;
static int*           dev_idx  = nullptr;

static std::vector<unsigned char> host_src;
static std::vector<unsigned char> host_dst;
static std::vector<unsigned char> host_dump;

static constexpr size_t kDstBytes = (size_t)C::SRC_BYTES * C::CLUSTER_PEERS;

static void reset_dst() { CHECK_HIP(hipMemset(dev_dst, C::DST_FILL, kDstBytes)); }
static void reset_dump() { CHECK_HIP(hipMemset(dev_dump, C::LDS_FILL, C::DUMP_BYTES)); }

static void read_dst() {
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(host_dst.data(), dev_dst, kDstBytes, hipMemcpyDeviceToHost));
}
static void read_dump() {
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(host_dump.data(), dev_dump, C::DUMP_BYTES, hipMemcpyDeviceToHost));
}

// ── 1. element sizes ────────────────────────────────────────────────────────────
// A 2 KiB tile round-tripped global -> LDS -> global at each legal element size. The
// tile is placed at the same BYTE region for every size (one tile along each axis from
// the origin), so all five instantiations must produce a bit-identical destination
// however differently the geometry is spelled in elements.
struct elem_case {
    const char* name;
    void (*kernel)(tdm_test_kargs);
    int elem_size;
};

static void run_elem_case(const elem_case& ec, tdm_test_kargs kargs) {
    const int tile0   = C::TILE_ROW_BYTES / ec.elem_size;
    const int stride0 = C::ROW_BYTES / ec.elem_size;

    kargs.shape0  = stride0;
    kargs.shape1  = C::ROWS;
    kargs.stride0 = stride0;
    kargs.origin0 = tile0;             // one tile along the contiguous axis -> byte 256
    kargs.origin1 = C::TILE_ROWS;      // one tile down -> row 8

    reset_dst();
    ec.kernel<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(kargs);
    CHECK_HIP(hipGetLastError());
    read_dst();

    // Rows [8, 16) and bytes [256, 512) of each: the same region for every element size.
    const int r0 = C::TILE_ROWS, r1 = C::TILE_ROWS + C::TILE_ROWS;
    const int b0 = C::TILE_ROW_BYTES, b1 = C::TILE_ROW_BYTES + C::TILE_ROW_BYTES;
    auto want = [&](size_t off) -> unsigned char {
        const int row = (int)(off / C::ROW_BYTES), col = (int)(off % C::ROW_BYTES);
        const bool inside = row >= r0 && row < r1 && col >= b0 && col < b1;
        return inside ? pat(off) : C::DST_FILL;
    };
    report(ec.name, compare(host_dst.data(), C::SRC_BYTES, want));
}

// ── 2. rank 2..5 ────────────────────────────────────────────────────────────────
// The tile arrives packed with dim0 fastest, so LDS element
// i0 + D0*(i1 + DN*(i2 + DN*(i3 + DN*i4))) must hold the source element that the
// per-dimension strides place it at. Step 1 is the same tile after one move along the
// OUTERMOST dimension, which is the only thing that reads that dimension's stride.
static void run_rank_case(const char* name, void (*kernel)(tdm_test_kargs), int rank,
                          tdm_test_kargs kargs) {
    const int row_step[4] = {C::RK_S1, C::RK_S2, C::RK_S3, C::RK_S4};
    int elems = C::RK_D0;
    for (int n = 1; n < rank; ++n) elems *= C::RK_DN;

    reset_dump();
    kargs.dump = dev_dump;
    kernel<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(kargs);
    CHECK_HIP(hipGetLastError());
    read_dump();

    const size_t checked = (size_t)C::RK_STEPS * elems * C::RK_ELEM_BYTES;
    auto want = [&](size_t off) -> unsigned char {
        const int step = (int)(off / ((size_t)elems * C::RK_ELEM_BYTES));
        size_t within  = off % ((size_t)elems * C::RK_ELEM_BYTES);
        const int lane = (int)(within % C::RK_ELEM_BYTES);   // byte within the element
        int flat       = (int)(within / C::RK_ELEM_BYTES);   // packed element index

        const int i0 = flat % C::RK_D0;
        flat /= C::RK_D0;
        int src_row = 0;
        for (int n = 1; n < rank; ++n) {
            const int in = flat % C::RK_DN;
            flat /= C::RK_DN;
            // Only the outermost dimension carries the move.
            const int origin = (n == rank - 1) ? step : 0;
            src_row += (in + origin) * row_step[n - 1];
        }
        const size_t src_elem = (size_t)src_row * C::RK_ROW_ELEMS + i0;
        return pat(src_elem * C::RK_ELEM_BYTES + lane);
    };
    char note[96];
    snprintf(note, sizeof(note), "(rank %d, %d elems/tile, +1 on dim%d)", rank, elems, rank - 1);
    report(name, compare(host_dump.data(), checked, want), note);
}

// ── 3. gather ───────────────────────────────────────────────────────────────────
// Row j of LDS must hold source row indices[j]; rows past the valid count must be
// untouched, which is what pins tile_dim1 to the count set_indices() was given rather
// than to the TileShape's second extent.
static void run_gather_case(const char* name, void (*kernel)(tdm_test_kargs), int n,
                            const std::vector<int>& idx, tdm_test_kargs kargs) {
    reset_dump();
    kargs.dump      = dev_dump;
    kargs.n_indices = n;
    kernel<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(kargs);
    CHECK_HIP(hipGetLastError());
    read_dump();

    const size_t checked = (size_t)C::GATHER_MAX_N * C::TILE_ROW_BYTES;
    auto want = [&](size_t off) -> unsigned char {
        const int row = (int)(off / C::TILE_ROW_BYTES);
        const int col = (int)(off % C::TILE_ROW_BYTES);
        if (row >= n) return C::LDS_FILL;   // past the valid index count
        return pat((size_t)idx[row] * C::ROW_BYTES + col);
    };
    char note[96];
    snprintf(note, sizeof(note), "(%d rows gathered, %d left as fill)", n, C::GATHER_MAX_N - n);
    const cmp_result r = compare(host_dump.data(), checked, want);
    report(name, r, note);

    // A gather that fetched the wrong rows and one that fetched nothing look the same in
    // a byte count, and they have completely different causes. Naming the source row each
    // LDS row actually holds separates them at a glance.
    if (r.bad != 0) {
        printf("        LDS row -> source row:");
        for (int j = 0; j < C::GATHER_MAX_N; ++j) {
            const unsigned char* row = host_dump.data() + (size_t)j * C::TILE_ROW_BYTES;
            int found = -1;
            for (int s = 0; s < C::ROWS && found < 0; ++s) {
                if (memcmp(row, host_src.data() + (size_t)s * C::ROW_BYTES, C::TILE_ROW_BYTES) == 0) found = s;
            }
            bool all_fill = true;
            for (int b = 0; b < C::TILE_ROW_BYTES && all_fill; ++b) all_fill = row[b] == C::LDS_FILL;
            if (all_fill)          printf(" [%d]fill", j);
            else if (found >= 0)   printf(" [%d]%d", j, found);
            else                   printf(" [%d]?", j);
        }
        printf("\n        expected            :");
        for (int j = 0; j < C::GATHER_MAX_N; ++j) {
            if (j < n) printf(" [%d]%d", j, idx[j]);
            else       printf(" [%d]fill", j);
        }
        printf("\n");
    }
}

// Whether the kernels above are the real ones. The tensor DMA opcodes exist only on
// gfx1250, so anywhere else the device pass compiled empty stubs and every check would
// come back as a byte mismatch -- twenty-six failures that say nothing about the arch
// being wrong. Asking the probe rather than the runtime for the arch name is what keeps
// this honest: it is compiled by the same #if that chose between real kernels and stubs,
// so the gate cannot disagree with what was actually built, which a string comparison
// against a list of arch names eventually would.
static bool tdm_device_is_gfx1250() {
    int* flag = nullptr;
    CHECK_HIP(hipMalloc(&flag, sizeof(int)));
    CHECK_HIP(hipMemset(flag, 0, sizeof(int)));
    tdm_test::probe_gfx1250<<<dim3(1), dim3(1)>>>(flag);
    CHECK_HIP(hipGetLastError());
    int on_gfx1250 = 0;
    CHECK_HIP(hipMemcpy(&on_gfx1250, flag, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(flag));
    return on_gfx1250 != 0;
}

// Returns the number of failed checks, or TDM_FEATURE_SKIPPED if the device is not
// gfx1250. Buffers are allocated and freed inside, and the counters reset on entry, so
// the harness may call this more than once per process.
inline int tdm_feature_suite_run() {
    g_failures = 0;
    g_passes   = 0;

    if (!tdm_device_is_gfx1250()) {
        printf("opus TDM functional coverage: SKIP, requires gfx1250\n");
        fflush(stdout);
        return TDM_FEATURE_SKIPPED;
    }

    printf("opus TDM functional coverage (gfx1250)\n");
    printf("  arena  : %d rows x %d B  (%d KiB)\n", C::ROWS, C::ROW_BYTES, C::SRC_BYTES / 1024);
    printf("  tile   : %d B x %d rows (%d KiB)\n", C::TILE_ROW_BYTES, C::TILE_ROWS, C::TILE_BYTES / 1024);

    // ── buffers ─────────────────────────────────────────────────────────────────
    host_src.resize(C::SRC_BYTES);
    for (size_t i = 0; i < host_src.size(); ++i) host_src[i] = pat(i);
    host_dst.resize(kDstBytes);
    host_dump.resize(C::DUMP_BYTES);

    // Strictly increasing, which gather mode requires of the hardware and opus does not
    // check. Spread out and irregular so a list read with the wrong stride, or reversed,
    // or off by one, lands on rows that are nothing like the right ones.
    std::vector<int> idx16 = {1, 3, 4, 9, 13, 20, 26, 31, 44, 52, 63, 77, 88, 99, 110, 127};
    std::vector<int> idx32(idx16.begin(), idx16.begin() + C::GATHER32_N);
    for (int v : idx16) {
        if (v >= C::ROWS) { fprintf(stderr, "gather index %d is past the arena\n", v); return 1; }
    }

    CHECK_HIP(hipMalloc(&dev_src, C::SRC_BYTES));
    CHECK_HIP(hipMalloc(&dev_dst, kDstBytes));
    CHECK_HIP(hipMalloc(&dev_dump, C::DUMP_BYTES));
    CHECK_HIP(hipMalloc(&dev_idx, (size_t)C::GATHER_MAX_N * sizeof(int)));
    CHECK_HIP(hipMemcpy(dev_src, host_src.data(), C::SRC_BYTES, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_idx, idx16.data(), (size_t)C::GATHER_MAX_N * sizeof(int),
                        hipMemcpyHostToDevice));

    tdm_test_kargs base{};
    base.src            = dev_src;
    base.dst            = dev_dst;
    base.dump           = nullptr;
    base.indices        = dev_idx;
    base.replica_stride = 0;

    // ── element sizes ───────────────────────────────────────────────────────────
    printf("\nelement sizes (D#.data_size, plus the sub-byte pack)\n");
    const elem_case elem_cases[] = {
        {"1 B  (unsigned char)", tdm_test::test_elem_b1, 1},
        {"2 B  (unsigned short)", tdm_test::test_elem_b2, 2},
        {"4 B  (unsigned int)", tdm_test::test_elem_b4, 4},
        {"8 B  (unsigned long long)", tdm_test::test_elem_b8, 8},
        {"1 B  (array<fp4_t, 2>)", tdm_test::test_elem_fp4, 1},
    };
    for (const auto& ec : elem_cases) run_elem_case(ec, base);

    // ── rank ────────────────────────────────────────────────────────────────────
    printf("\ntile rank, each with a move along its outermost dimension\n");
    {
        tdm_test_kargs k = base;
        run_rank_case("rank 2", tdm_test::test_rank2, 2, k);
        run_rank_case("rank 3", tdm_test::test_rank3, 3, k);
        run_rank_case("rank 4", tdm_test::test_rank4, 4, k);
        run_rank_case("rank 5", tdm_test::test_rank5, 5, k);
    }

    // ── gather ──────────────────────────────────────────────────────────────────
    printf("\ngather (row-index list in D# groups 2/3, 2D only)\n");
    {
        tdm_test_kargs k = base;
        k.shape0  = C::ROW_BYTES / 2;   // u16 elements per row
        k.shape1  = C::ROWS;
        k.stride0 = C::ROW_BYTES / 2;
        k.origin0 = 0;
        k.origin1 = 0;   // must stay 0: in gather mode the index IS the dim1 coordinate
        // A second list of the same length, so a case can be run twice changing nothing
        // but which rows were asked for. That is the only way to tell a list that is read
        // and misused from a list that is not read at all.
        const std::vector<int> alt = {37, 82};
        std::vector<int> alt_padded(C::GATHER_MAX_N, 0);
        for (size_t i = 0; i < alt.size(); ++i) alt_padded[i] = alt[i];
        int* dev_alt = nullptr;
        CHECK_HIP(hipMalloc(&dev_alt, (size_t)C::GATHER_MAX_N * sizeof(int)));
        CHECK_HIP(hipMemcpy(dev_alt, alt_padded.data(), (size_t)C::GATHER_MAX_N * sizeof(int),
                            hipMemcpyHostToDevice));
        tdm_test_kargs k_alt = k;
        k_alt.indices = dev_alt;

        // 32-bit indices: the working width. Three list lengths and two different lists,
        // so what lands in LDS is pinned to the list rather than to the origin.
        run_gather_case("gather<32>, 8 indices", tdm_test::test_gather32, C::GATHER32_N, idx32, k);
        run_gather_case("gather<32>, short list", tdm_test::test_gather_partial,
                        C::GATHER_PARTIAL_N, idx32, k);
        run_gather_case("gather<32>, second list", tdm_test::test_gather32_pair, 2, alt, k_alt);

        // 16-bit indices: two per dword, so a full list is twice as long as a 32-bit one.
        run_gather_case("gather<16>, 16 indices", tdm_test::test_gather16, C::GATHER16_N, idx16, k);
        run_gather_case("gather<16>, 2 indices", tdm_test::test_gather16_pair, 2, idx16, k);

        // Which bit is gather_mode. The same descriptor issued twice, differing in
        // nothing but group0 dword0: the ordering opus emits must fetch the listed rows,
        // and the transposed one must not. Both halves are asserted, because a pair that
        // stopped discriminating would leave the first half proving nothing.
        //
        // This is the only check in the suite that would catch the transposition. It is
        // silent everywhere else: it costs gather<16> its rows while leaving gather<32>
        // -- which is 0xC0000001 either way round -- working perfectly.
        {
            auto probe = [&](const char* name, void (*kernel)(tdm_test_kargs), bool should_gather) {
                reset_dump();
                tdm_test_kargs kp = k;
                kp.dump = dev_dump;
                kernel<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(kp);
                CHECK_HIP(hipGetLastError());
                read_dump();
                auto listed = [&](size_t off) -> unsigned char {
                    const int row = (int)(off / C::TILE_ROW_BYTES);
                    const int col = (int)(off % C::TILE_ROW_BYTES);
                    if (row >= C::GATHER32_N) return C::LDS_FILL;
                    return pat((size_t)idx32[row] * C::ROW_BYTES + col);
                };
                const cmp_result r = compare(host_dump.data(),
                                             (size_t)C::GATHER_MAX_N * C::TILE_ROW_BYTES, listed);
                // Inverted for the negative half: "did not fetch the listed rows" is the
                // pass, so a zero mismatch count there is the failure.
                cmp_result verdict;
                if (!should_gather && r.bad != 0) verdict = cmp_result{};
                else if (!should_gather)          verdict = cmp_result{1, 0, 0, 0};
                else                              verdict = r;
                report(name, verdict,
                       should_gather ? "(fetches the listed rows)" : "(does not gather, as it must not)");
            };
            probe("group0 0x80000001, as emitted", tdm_test::test_gather16_bit31, true);
            probe("group0 0x40000001, transposed", tdm_test::test_gather16_bit30, false);
        }

        CHECK_HIP(hipFree(dev_alt));
    }

    // ── scatter ─────────────────────────────────────────────────────────────────
    // Rows 0..N-1 are staged into LDS by a plain consecutive load, then written out by a
    // gather-mode store, so they must land on the INDEXED rows of the destination and
    // nowhere else.
    printf("\nscatter (a gather-mode async_store)\n");
    {
        tdm_test_kargs k = base;
        k.shape0  = C::ROW_BYTES / 2;
        k.shape1  = C::ROWS;
        k.stride0 = C::ROW_BYTES / 2;
        k.origin0 = 0;
        k.origin1 = 0;
        reset_dst();
        tdm_test::test_scatter<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(k);
        CHECK_HIP(hipGetLastError());
        read_dst();

        // Which staged row, if any, a destination row must hold.
        std::vector<int> src_of_row(C::ROWS, -1);
        for (int j = 0; j < C::GATHER32_N; ++j) src_of_row[idx32[j]] = j;

        auto want = [&](size_t off) -> unsigned char {
            const int row = (int)(off / C::ROW_BYTES), col = (int)(off % C::ROW_BYTES);
            if (col >= C::TILE_ROW_BYTES || src_of_row[row] < 0) return C::DST_FILL;
            return pat((size_t)src_of_row[row] * C::ROW_BYTES + col);
        };
        char note[96];
        snprintf(note, sizeof(note), "(%d rows placed by index)", C::GATHER32_N);
        report("scatter<32>, 8 rows", compare(host_dst.data(), C::SRC_BYTES, want), note);
    }

    // ── runtime tile_dim1 ───────────────────────────────────────────────────────
    printf("\nruntime state overrides\n");
    {
        tdm_test_kargs k = base;
        k.shape0  = C::ROW_BYTES / 2;
        k.shape1  = C::ROWS;
        k.stride0 = C::ROW_BYTES / 2;
        k.dump    = dev_dump;
        reset_dump();
        tdm_test::test_tile_dim1<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(k);
        CHECK_HIP(hipGetLastError());
        read_dump();

        auto want = [&](size_t off) -> unsigned char {
            const int row = (int)(off / C::TILE_ROW_BYTES);
            const int col = (int)(off % C::TILE_ROW_BYTES);
            if (row >= C::TILE_DIM1_OVERRIDE) return C::LDS_FILL;
            return pat((size_t)row * C::ROW_BYTES + col);
        };
        char note[96];
        snprintf(note, sizeof(note), "(%d of %d rows moved)", C::TILE_DIM1_OVERRIDE, C::TILE_ROWS);
        report("set_tile_dim1", compare(host_dump.data(), C::TILE_BYTES, want), note);
    }

    // ── descriptor split and move ───────────────────────────────────────────────
    // Slots 0 and 2 are the same tile, the second reached by moving forward one tile and
    // then back. Slot 1 is the tile in between. Only slot 2 can fail a move that dropped
    // the sign or that only ever accumulated forwards.
    {
        const int tile0_u16 = C::TILE_ROW_BYTES / 2;
        tdm_test_kargs k = base;
        k.shape0  = C::ROW_BYTES / 2;
        k.shape1  = C::ROWS;
        k.stride0 = C::ROW_BYTES / 2;
        k.dump    = dev_dump;
        reset_dump();
        tdm_test::test_split_move<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(k);
        CHECK_HIP(hipGetLastError());
        read_dump();

        auto want = [&](size_t off) -> unsigned char {
            const int slot = (int)(off / C::TILE_BYTES);
            const size_t within = off % C::TILE_BYTES;
            const int row = (int)(within / C::TILE_ROW_BYTES);
            const int col = (int)(within % C::TILE_ROW_BYTES);
            const int col_origin = (slot == 1) ? tile0_u16 * 2 : 0;   // slot 1 is one tile along
            return pat((size_t)row * C::ROW_BYTES + col_origin + col);
        };
        report("make_descriptor + move back/forth",
               compare(host_dump.data(), (size_t)C::SPLIT_SLOTS * C::TILE_BYTES, want),
               "(tile 0, tile 1, tile 0 again via a negative delta)");
    }

    // ── padding ─────────────────────────────────────────────────────────────────
    // Row r must start at r * PAD_PITCH elements, and the PAD_ELEMS slots after each row
    // must still hold the fill the kernel wrote before the load.
    printf("\npadding (LDS write side, 4-byte element)\n");
    {
        tdm_test_kargs k = base;
        k.shape0  = C::ROW_BYTES / C::PAD_ELEM_BYTES;
        k.shape1  = C::ROWS;
        k.stride0 = C::ROW_BYTES / C::PAD_ELEM_BYTES;
        k.dump    = dev_dump;
        reset_dump();
        tdm_test::test_padded<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(k);
        CHECK_HIP(hipGetLastError());
        read_dump();

        const int pitch_bytes = C::PAD_PITCH * C::PAD_ELEM_BYTES;   // 272
        auto want = [&](size_t off) -> unsigned char {
            const int row = (int)(off / pitch_bytes);
            const int col = (int)(off % pitch_bytes);
            if (col >= C::TILE_ROW_BYTES) return C::LDS_FILL;   // the pad slots
            return pat((size_t)row * C::ROW_BYTES + col);
        };
        char note[96];
        snprintf(note, sizeof(note), "(pitch %d elems, %d pad elems/row untouched)",
                 C::PAD_PITCH, C::PAD_ELEMS);
        report("padding_auto<u32>", compare(host_dump.data(), C::PAD_LDS_BYTES, want), note);
    }

    // ── multicast ───────────────────────────────────────────────────────────────
    // Both peers issue the same load with the same mask; GL1 merges the requests into one
    // data return that lands in each of their LDS. Each writes its own replica of the
    // output and both replicas must hold the tile.
    printf("\nmulticast (2 peers, each writing its own replica)\n");
    {
        tdm_test_kargs k = base;
        k.shape0         = C::ROW_BYTES / 2;
        k.shape1         = C::ROWS;
        k.stride0        = C::ROW_BYTES / 2;
        k.replica_stride = C::SRC_BYTES / 2;   // in u16 elements

        auto check_replicas = [&](const char* name, const char* note) {
            for (int rep = 0; rep < C::CLUSTER_PEERS; ++rep) {
                const unsigned char* buf = host_dst.data() + (size_t)rep * C::SRC_BYTES;
                auto want = [&](size_t off) -> unsigned char {
                    const int row = (int)(off / C::ROW_BYTES), col = (int)(off % C::ROW_BYTES);
                    const bool inside = row < C::TILE_ROWS && col < C::TILE_ROW_BYTES;
                    return inside ? pat(off) : C::DST_FILL;
                };
                char full[96];
                snprintf(full, sizeof(full), "%s, replica %d", name, rep);
                report(full, compare(buf, C::SRC_BYTES, want), note);
            }
        };

        reset_dst();
        tdm_test::test_multicast_static<<<dim3(C::CLUSTER_PEERS, 1, 1), dim3(C::BLOCK_SIZE)>>>(k);
        CHECK_HIP(hipGetLastError());
        read_dst();
        check_replicas("multicast<0,1>", "(compile-time peer set)");

        reset_dst();
        tdm_test::test_multicast_peers_y<<<dim3(1, C::CLUSTER_PEERS, 1), dim3(C::BLOCK_SIZE)>>>(k);
        CHECK_HIP(hipGetLastError());
        read_dst();
        check_replicas("peers_along_y", "(runtime mask via set_workgroup_mask)");
    }

    // ── clamping ────────────────────────────────────────────────────────────────
    // Nothing in the kernels guards the edges. make() is told the WHOLE tensor's shape
    // plus the tile origin and clamps internally, so a tile hanging off the edge becomes
    // a short transfer and a tile entirely past the end becomes a zero-extent DMA that
    // touches no memory. Both windows in the round trip clamp identically.
    printf("\nclamping (saturating_sub on the tensor extents)\n");
    {
        // Ragged: the logical shape ends mid-tile on both axes while the allocation and
        // the row pitch stay whole, which is how you would view a sub-matrix.
        const int stride0 = C::ROW_BYTES / 2;
        tdm_test_kargs k  = base;
        k.stride0 = stride0;
        k.shape0  = C::TILE_ROW_BYTES / 2 + 37;   // ends 37 elements into the second tile
        k.shape1  = C::TILE_ROWS + 3;             // ends 3 rows into the second tile
        k.origin0 = C::TILE_ROW_BYTES / 2;        // the tile that hangs off both edges
        k.origin1 = C::TILE_ROWS;

        reset_dst();
        tdm_test::test_elem_b2<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(k);
        CHECK_HIP(hipGetLastError());
        read_dst();
        auto want_ragged = [&](size_t off) -> unsigned char {
            const int row = (int)(off / C::ROW_BYTES), col_b = (int)(off % C::ROW_BYTES);
            const int elem = col_b / 2;
            const bool inside = row >= k.origin1 && row < k.shape1 &&
                                elem >= k.origin0 && elem < k.shape0;
            return inside ? pat(off) : C::DST_FILL;
        };
        report("ragged tile", compare(host_dst.data(), C::SRC_BYTES, want_ragged),
               "(3 of 8 rows, 37 of 128 elements)");

        // Wholly past the end on both axes: a zero-extent DMA. It still costs a tensorcnt
        // tick -- the kernel waits to zero and returns, which is the other half of the
        // claim -- but it must touch no memory at all.
        k.origin0 = k.shape0 + 64;
        k.origin1 = k.shape1 + 10;
        reset_dst();
        tdm_test::test_elem_b2<<<dim3(1), dim3(C::BLOCK_SIZE)>>>(k);
        CHECK_HIP(hipGetLastError());
        read_dst();
        report("origin wholly past the end",
               compare(host_dst.data(), C::SRC_BYTES, [](size_t) { return C::DST_FILL; }),
               "(zero-extent DMA, still retires)");
    }

    CHECK_HIP(hipFree(dev_src));
    CHECK_HIP(hipFree(dev_dst));
    CHECK_HIP(hipFree(dev_dump));
    CHECK_HIP(hipFree(dev_idx));

    printf("\n%d passed, %d failed\n", g_passes, g_failures);
    if (g_failures == 0) {
        printf("all checks passed\n");
        printf("note: the compile-time coverage (gather mode bits, padding encodings, tag\n"
               "      rejections) is checked by building -- see gcnasm/opus_tdm_test's\n"
               "      `make negative` for the diagnostics that must NOT compile.\n");
    }
    fflush(stdout);
    return g_failures;
}

extern "C" int run_tdm_feature_suite(void) { return tdm_feature_suite_run(); }
#endif  // __HIP_DEVICE_COMPILE__

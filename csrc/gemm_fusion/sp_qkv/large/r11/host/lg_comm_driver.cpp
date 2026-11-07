// SPDX-License-Identifier: MIT
// LG-QSDMA-r6 host driver (derived from r3..r5b) (UNCOMPILED source): CCO comm/window/devcomm setup + kernel launch, C-linkage for ctypes.
// Mirrors PR695-M4 host/driver.cpp l.569, 636-770 (ccoGetUniqueId / ccoCommCreate / size-only managed ccoWindowRegister /
// ccoDevCommCreate with GDA_NONE + sdmaQueueCount=1). Windows are registered ONCE, BEFORE ccoDevCommCreate.
// Caller contract: all 4 ranks call lg_comm_create, then a cross-rank barrier, then lg_comm_init_control, then a SECOND
// barrier, before the first lg_comm_launch (peers must not PUT/credit into an uninitialised control window).
#include <hip/hip_runtime.h>
#include <mori/cco/cco.hpp>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "lg_comm_abi_v2.h"

// r4 grid (must equal lgc::grid_items in device/lg_comm_device.hpp; same LG_REMOTE_HALF build flag on host and device):
// remote tiles = LG_NUM_PEERS*NPD*chunks (x2 half-N items when LG_REMOTE_HALF) + self tiles NPD*chunks, NPD = LG_COMM_F/256.
// r6x default puts-per-peer policy by M (documented default only; the TIMED choice is made by Evidence per T): from the r6b dense measurements B/C T8192 n1 1.126, T16384 n2 1.312, T32768 n4 1.378.
static inline uint32_t default_puts_per_peer(uint32_t M) { return M <= 2048u ? 1u : (M <= 4096u ? 2u : 4u); }
// r7: grid = 1 poster + items; half-first plan (hg) adds 3*9*(chunks/n) items (group-0 remote tiles split into halves). Must equal lgc::grid_items_h + 1 (device guard checks gridDim).
// r9: hg == 2 = quarter-first (dense M=2048, n=2): 3*9*(2*4 + 6) + 9*8 = 450 items. Must equal lgc::grid_items_q + 1 (device guard checks gridDim).
static inline uint32_t grid_items(uint32_t chunks, uint32_t npp, uint32_t hg) {
    if (hg == 2u) return LG_NUM_PEERS * LG_TILES_N_PER_DEST * (LG_R9_QC * 4u + (chunks - LG_R9_QC)) + LG_TILES_N_PER_DEST * chunks + (LG_R6_POSTER ? 1u : 0u);
    return LG_X_ITEMS_PER_CHUNK * chunks + (hg ? LG_NUM_PEERS * LG_TILES_N_PER_DEST * (chunks / npp) : 0u) + (LG_R6_POSTER ? 1u : 0u);
}

struct LgCommCtx {
    mori::cco::ccoComm* comm = nullptr;
    mori::cco::ccoDevComm dev{};
    mori::cco::ccoWindow_t S = nullptr, R = nullptr, C = nullptr;
    void *Sva = nullptr, *Rva = nullptr, *Cva = nullptr;
    uint32_t rank = 0, M = 0, Q = 1;
    hipModule_t mod = nullptr; hipFunction_t fgemm[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}, fquiet = nullptr, fdrain = nullptr;   // r6x: n1/n2/n4 entries (puts per peer) in one HSACO
};

extern "C" {
int32_t lg_comm_uid(uint8_t* out128) {            // rank 0 only; other ranks pass zeros
    mori::cco::ccoUniqueId uid{};
    if (mori::cco::ccoGetUniqueId(&uid) != 0) return -1;
    static_assert(sizeof(uid) == 128, "UID 128B");
    std::memcpy(out128, &uid, 128);
    return 0;
}

// r8f diagnostics: with env LG_DIAG_CREATE=1 every create stage prints its rc/size to stderr (rank-tagged). No behaviour change otherwise.
static bool lg_diag() { const char* e = std::getenv("LG_DIAG_CREATE"); return e && e[0] == '1'; }
#define LGD(rank, ...) do { if (lg_diag()) { std::fprintf(stderr, "[LG_DIAG_CREATE r%u] ", static_cast<unsigned>(rank)); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

void* lg_comm_create(uint32_t rank, const uint8_t* uid128, uint32_t M, uint64_t per_rank_vmm_bytes, const char* hsaco_path, uint32_t sdma_queues) {
    LGD(rank, "enter M=%u per_rank_vmm_bytes=%llu Q=%u LG_COMM_F=%u N=%u indexer=%d fold=%d pad=%u", M, static_cast<unsigned long long>(per_rank_vmm_bytes), sdma_queues,
        static_cast<unsigned>(LG_COMM_F), static_cast<unsigned>(LG_X_N_RAW), static_cast<int>(LG_X_INDEXER), static_cast<int>(LG_R8_FOLD), static_cast<unsigned>(LG_R_PAD_BYTES));
    if (rank > 3u || M == 0 || M % 256u || M > 8192u) { LGD(rank, "REJECT: rank/M"); return nullptr; }
    if (sdma_queues != 1u && sdma_queues != 2u && sdma_queues != 4u) { LGD(rank, "REJECT: Q"); return nullptr; }   // r4: Q in {1,2,4}
    const hipError_t se = hipSetDevice(static_cast<int>(rank));
    LGD(rank, "hipSetDevice rc=%d", static_cast<int>(se));
    if (se != hipSuccess) return nullptr;
    auto* c = new LgCommCtx();
    c->rank = rank; c->M = M; c->Q = sdma_queues;
    const int rc_cc = mori::cco::ccoCommCreate(*reinterpret_cast<const mori::cco::ccoUniqueId*>(uid128), 4, static_cast<int>(rank), per_rank_vmm_bytes, &c->comm);
    LGD(rank, "ccoCommCreate rc=%d comm=%p", rc_cc, static_cast<void*>(c->comm));
    if (rc_cc != 0 || !c->comm) { delete c; return nullptr; }
    LGD(rank, "world=%d lsaSize=%d lsaRank=%d myNodeStart=%d vmmGranularity=%llu perRankSize=%llu", c->comm->worldSize, c->comm->lsaSize, c->comm->lsaRank, c->comm->myNodeStart,
        static_cast<unsigned long long>(c->comm->vmmGranularity), static_cast<unsigned long long>(c->comm->perRankSize));
    if (c->comm->worldSize != 4 || c->comm->lsaSize != 4 || static_cast<uint32_t>(c->comm->lsaRank) != rank ||
        c->comm->myNodeStart != 0) { LGD(rank, "REJECT: topology guard"); delete c; return nullptr; }                       // topology guard as M4
    const uint64_t P = 8ull * M * LG_COMM_F;                                           // 4*M*F BF16 bytes
    const uint64_t stamp_bytes = LGC_STAMP_OFF + 64ull * 4096ull;                   // r5: per-WG stamp records (grid <= 4096 blocks)
    const uint64_t cap = (P > stamp_bytes ? P : stamp_bytes);                     // Cwin capacity: >= payload size and >= stamp area (control content still LGC_BYTES)
    // r8g window sizing (create failure analysis, see the note): the allocator (cco_init.cpp l.759-850) does AlignUp(size, vmmGranularity), takes a first-fit slot, then hipMemCreate/hipMemMap/hipMemSetAccess
    // on THAT window only (no cross-window size coupling in the code). But `vmmGranularity` (4096 here) is queried (l.653-668) with an allocProp WITHOUT allocFlags.gpuDirectRDMACapable, while ccoMemAlloc
    // allocates WITH it (l.800-806; the code's own comment says this flag can make hipMemSetAccess fail with "invalid argument"). So 4096 is not the effective allocation granule. Every window size that
    // ever worked (36/40/72/80/144/160 MiB: gcd 4 MiB) is a multiple of 4 MiB; the failing 38 MiB (r8e R) and 36 MiB + 4 KiB (r8d R) are not => the effective granule is 4 MiB (inference).
    // Fix (robust under BOTH hypotheses): ALL THREE windows get the SAME size = roundup(max(P + pad, cap), G) with G = max(comm granularity, 4 MiB, env LG_WINDOW_GRAN_MIB). Cost: <= 2 x (G) extra HBM.
    uint64_t gran = static_cast<uint64_t>(c->comm->vmmGranularity);
    if (gran < (4ull << 20)) gran = (4ull << 20);
    if (const char* ge = std::getenv("LG_WINDOW_GRAN_MIB")) { const uint64_t m = std::strtoull(ge, nullptr, 10); if (m > 0 && (m << 20) > gran) gran = m << 20; }   // experiment knob (8, 16, ...)
    auto up = [gran](uint64_t x) { return (x + gran - 1) / gran * gran; };
    const uint64_t need = (P + LG_R_PAD_BYTES) > cap ? (P + LG_R_PAD_BYTES) : cap;
    const uint64_t Ws = up(need);
    const uint64_t Ps = Ws, Rs = Ws, Cs = Ws;                                          // equal sizes (S == R == C)
    LGD(rank, "window sizing: P=%llu P+pad=%llu cap=%llu need=%llu gran=%llu -> every window = %llu B (%.1f MiB)", static_cast<unsigned long long>(P), static_cast<unsigned long long>(P + LG_R_PAD_BYTES),
        static_cast<unsigned long long>(cap), static_cast<unsigned long long>(need), static_cast<unsigned long long>(gran), static_cast<unsigned long long>(Ws), static_cast<double>(Ws) / 1048576.0);
    if (lg_diag()) {                                                                    // probe the REAL granule with and without gpuDirectRDMACapable (decides the hypothesis on the next run)
        for (int gdr = 0; gdr < 2; ++gdr) {
            hipMemAllocationProp pr = {};
            pr.type = hipMemAllocationTypePinned;
            pr.requestedHandleType = static_cast<hipMemAllocationHandleType>(c->comm->handleType);
            pr.location.type = hipMemLocationTypeDevice;
            pr.location.id = static_cast<int>(rank);
            pr.allocFlags.gpuDirectRDMACapable = gdr;
            size_t gmin = 0, grec = 0;
            const hipError_t e1 = hipMemGetAllocationGranularity(&gmin, &pr, hipMemAllocationGranularityMinimum);
            const hipError_t e2 = hipMemGetAllocationGranularity(&grec, &pr, hipMemAllocationGranularityRecommended);
            LGD(rank, "granule probe gpuDirectRDMACapable=%d: minimum=%zu (rc=%d) recommended=%zu (rc=%d)", gdr, gmin, static_cast<int>(e1), grec, static_cast<int>(e2));
        }
    }    const int rc_s = mori::cco::ccoWindowRegister(c->comm, static_cast<size_t>(Ps), &c->S, &c->Sva);
    LGD(rank, "ccoWindowRegister S size=%llu rc=%d va=%p", static_cast<unsigned long long>(Ps), rc_s, c->Sva);
    if (rc_s != 0) { delete c; return nullptr; }
    const int rc_r = mori::cco::ccoWindowRegister(c->comm, static_cast<size_t>(Rs), &c->R, &c->Rva);
    LGD(rank, "ccoWindowRegister R size=%llu rc=%d va=%p", static_cast<unsigned long long>(Rs), rc_r, c->Rva);
    if (rc_r != 0) { delete c; return nullptr; }
    const int rc_c = mori::cco::ccoWindowRegister(c->comm, static_cast<size_t>(Cs), &c->C, &c->Cva);
    LGD(rank, "ccoWindowRegister C size=%llu rc=%d va=%p", static_cast<unsigned long long>(Cs), rc_c, c->Cva);
    if (rc_c != 0) { delete c; return nullptr; }
    mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_NONE;
    reqs.sdmaQueueCount = static_cast<int>(sdma_queues);          // r4: Q queues per peer (chunk c -> queue c%Q)
    const int rc_dc = mori::cco::ccoDevCommCreate(c->comm, &reqs, &c->dev);
    LGD(rank, "ccoDevCommCreate rc=%d sdmaNumQueue=%d (want %u)", rc_dc, static_cast<int>(c->dev.sdma.sdmaNumQueue), sdma_queues);
    if (rc_dc != 0 || static_cast<uint32_t>(c->dev.sdma.sdmaNumQueue) != sdma_queues) { delete c; return nullptr; }   // CCO may cap Q: failure is reported, never silently reduced
    const hipError_t e_ml = hipModuleLoad(&c->mod, hsaco_path);
    LGD(rank, "hipModuleLoad rc=%d path=%s", static_cast<int>(e_ml), hsaco_path);
    if (e_ml != hipSuccess) { delete c; return nullptr; }
    struct { hipFunction_t* f; const char* name; } fns[] = {{&c->fgemm[0], LG_ENTRY_GEMM_N1}, {&c->fgemm[1], LG_ENTRY_GEMM_N2}, {&c->fgemm[2], LG_ENTRY_GEMM_N4}, {&c->fgemm[3], LG_ENTRY_GEMM_N2H},
                                                 {&c->fgemm[4], LG_ENTRY_GEMM_N4H}, {&c->fgemm[5], LG_ENTRY_GEMM_N3Q}, {&c->fquiet, LG_ENTRY_QUIET}, {&c->fdrain, LG_ENTRY_DRAIN}};
    for (auto& fn : fns) {
        const hipError_t e_gf = hipModuleGetFunction(fn.f, c->mod, fn.name);
        LGD(rank, "hipModuleGetFunction %s rc=%d", fn.name, static_cast<int>(e_gf));
        if (e_gf != hipSuccess) { delete c; return nullptr; }
    }
    LGD(rank, "create OK");
    return c;
}
// Device pointers (MORI-managed, local rank). R is the single receive buffer [4M, F] BF16 token-major.
void* lg_comm_R_ptr(void* h) { return static_cast<LgCommCtx*>(h)->Rva; }
// r8 build constants (so the bridge can validate that the HSACO/lib it loads matches its plan): out[0]=F, out[1]=N, out[2]=indexer, out[3]=fold, out[4]=R pad bytes
int32_t lg_comm_build_info(uint32_t* out5) { out5[0] = LG_COMM_F; out5[1] = LG_X_N_RAW; out5[2] = LG_X_INDEXER; out5[3] = LG_R8_FOLD; out5[4] = LG_R_PAD_BYTES; return 0; }
void* lg_comm_S_ptr(void* h) { return static_cast<LgCommCtx*>(h)->Sva; }

// Control window initial image: magic, ep=1 (first epoch is 1), everything else 0. Also clears R/S bookkeeping-free.
int32_t lg_comm_init_control(void* h) {
    auto* c = static_cast<LgCommCtx*>(h);
    uint8_t img[LGC_BYTES]; std::memset(img, 0, sizeof img);
    *reinterpret_cast<uint64_t*>(img + LGC_EP_OFF) = 1ull;
    *reinterpret_cast<uint64_t*>(img + LGC_MAGIC_OFF) = LGC_MAGIC;
    if (hipMemcpy(c->Cva, img, sizeof img, hipMemcpyHostToDevice) != hipSuccess) return -1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : -2;
}

// One epoch = GEMM(grid = chunks*36 WGs x 512) then quiet_ready(1 WG x 256), same stream. Returns hip error code.
int32_t lg_comm_launch(void* h, const void* A, const void* W, const float* Ascale, const float* Wscale,
                       uint32_t strideA, uint64_t poll_iterations, uint64_t poll_budget_us, uint32_t puts_per_peer, uint32_t half_first, void* stream) {   // half_first (r7): 1 = group-0 remote tiles as half-N items (needs puts_per_peer 2 or 4); 0 = r6x plan   // puts_per_peer 0 => default_puts_per_peer(M); else 1, 2 or 4 (selects the ENTRY)
    auto* c = static_cast<LgCommCtx*>(h);
    LgCommArgsV2 a; std::memset(&a, 0, sizeof a);
    a.dev = c->dev; a.Swin = c->S; a.Rwin = c->R; a.Cwin = c->C;
    a.A = A; a.W = W; a.Ascale = Ascale; a.Wscale = Wscale;
    a.M = c->M; a.N = LG_X_N_RAW; a.K = 6144; a.strideA = strideA; a.F = LG_COMM_F; a.rank = c->rank;
    a.chunks = c->M / LG_CHUNK_ROWS; a.abi_version = LG_COMM_ABI_VERSION; a.poll_iterations = poll_iterations;
    a.sdma_queues = c->Q; a.items_per_group = LG_ITEMS_PER_GROUP;
    a.reserved[1] = poll_budget_us ? poll_budget_us : LG_POLL_BUDGET_US_DEFAULT;               // r6b: wait budget in MICROSECONDS (s_memrealtime); 0 => default 50 ms
    const uint32_t npp = puts_per_peer ? puts_per_peer : (half_first == 2u ? 3u : default_puts_per_peer(a.M));
    if (npp != 1u && npp != 2u && npp != 4u && !(half_first == 2u && npp == 3u)) return -101;
    const uint32_t hg = half_first == 0u ? 0u : (half_first == 2u ? 2u : 1u);   // half_first: 0 = r6x plan, 1 = half-N first group (r7), 2 = QUARTER-first group 0 (r9: dense, M=2048, n=2 only)
    if (hg == 1u && npp == 1u) return -102;                                // half-first needs >= 2 groups
    if (hg == 2u && (npp != 3u || a.chunks != 8u || LG_X_INDEXER)) return -103;   // quarter-first: THREE groups (npp must be 3), M=2048 (8 chunks), dense build only
    a.reserved[0] = LG_R6_PLAN_CODE_N(npp, hg);                                                        // r6: host/device build agreement (poster, puts per peer, half)
    if (half_first != 2u && a.chunks % npp != 0u) return -100;   // chunks must be divisible by n (the quarter-first plan has uneven groups)
    size_t sz = sizeof a;
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    hipError_t r = hipModuleLaunchKernel(c->fgemm[hg == 2u ? 5 : (hg ? (npp == 2u ? 3 : 4) : (npp == 1u ? 0 : (npp == 2u ? 1 : 2)))], grid_items(a.chunks, npp, hg), 1, 1, 512, 1, 1, 0, static_cast<hipStream_t>(stream), nullptr, cfg);
#if LG_R8_FOLD
    return static_cast<int32_t>(r);                                          // r8 A2: no quiet_ready launch: the poster block folded W2 + reset + epoch advance (the GEMM kernel end implies R complete)
#else
    if (r != hipSuccess) return static_cast<int32_t>(r);
    r = hipModuleLaunchKernel(c->fquiet, 1, 1, 1, 256, 1, 1, 0, static_cast<hipStream_t>(stream), nullptr, cfg);
    return static_cast<int32_t>(r);
#endif
}

// Control/report readback (call after stream sync): out[0]=ep, out[1]=poison, out[2..9]=report[0..7].
int32_t lg_comm_read_report(void* h, uint64_t* out10) {
    auto* c = static_cast<LgCommCtx*>(h);
    uint8_t img[LGC_BYTES];
    if (hipMemcpy(img, c->Cva, sizeof img, hipMemcpyDeviceToHost) != hipSuccess) return -1;
    out10[0] = *reinterpret_cast<uint64_t*>(img + LGC_EP_OFF);
    out10[1] = *reinterpret_cast<uint32_t*>(img + LGC_POISON_OFF);
    for (int i = 0; i < 8; ++i) out10[2 + i] = *reinterpret_cast<uint64_t*>(img + LGC_REPORT_OFF + 8 * i);
    return 0;
}
// DEBUG (build with -DLG_DEBUG_POSTCOUNT=1 device side): read post_count[4][32] (cumulative since the last clear); zero_after=1 clears it.
int32_t lg_comm_read_postcount(void* h, uint32_t* out128, int32_t zero_after) {
    auto* c = static_cast<LgCommCtx*>(h);
    if (hipMemcpy(out128, static_cast<uint8_t*>(c->Cva) + LGC_POSTCNT_OFF, 4 * 32 * 4, hipMemcpyDeviceToHost) != hipSuccess) return -1;
    if (zero_after) {
        uint32_t z[128]; std::memset(z, 0, sizeof z);
        if (hipMemcpy(static_cast<uint8_t*>(c->Cva) + LGC_POSTCNT_OFF, z, sizeof z, hipMemcpyHostToDevice) != hipSuccess) return -2;
    }
    return 0;
}
// r5 stamps (device built with -DLG_STAMPS=1): out = [grid blocks][8] per-WG records, [3][8] quiet stamps, [8] xcc queue counters. clear=1 zeroes the per-WG area.
int32_t lg_comm_read_stamps(void* h, uint32_t nblocks, uint64_t* wg_out, uint64_t* q_out, int32_t clear) {
    auto* c = static_cast<LgCommCtx*>(h); if (nblocks > 4096u) return -3;
    if (hipMemcpy(wg_out, static_cast<uint8_t*>(c->Cva) + LGC_STAMP_OFF, 64ull * nblocks, hipMemcpyDeviceToHost) != hipSuccess) return -1;
    if (hipMemcpy(q_out, static_cast<uint8_t*>(c->Cva) + LGC_QSTAMP_OFF, 3 * 64, hipMemcpyDeviceToHost) != hipSuccess) return -2;
    if (clear) { void* z = calloc(64ull * nblocks, 1); hipMemcpy(static_cast<uint8_t*>(c->Cva) + LGC_STAMP_OFF, z, 64ull * nblocks, hipMemcpyHostToDevice); free(z); }
    return 0;
}
// r6b recovery (F-A): quietQueue on every peer queue of this rank. Call after ANY poisoned epoch, then hipStreamSynchronize, BEFORE re-init / the next epoch.
int32_t lg_comm_drain_all(void* h, void* stream) {
    auto* c = static_cast<LgCommCtx*>(h);
    LgCommArgsV2 a; std::memset(&a, 0, sizeof a);
    a.dev = c->dev; a.Swin = c->S; a.Rwin = c->R; a.Cwin = c->C; a.rank = c->rank; a.sdma_queues = c->Q; a.abi_version = LG_COMM_ABI_VERSION;
    size_t sz = sizeof a; void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    return static_cast<int32_t>(hipModuleLaunchKernel(c->fdrain, 1, 1, 1, 64, 1, 1, 0, static_cast<hipStream_t>(stream), nullptr, cfg));
}
uint32_t lg_comm_grid_items(uint32_t chunks, uint32_t npp, uint32_t hg) { return grid_items(chunks, npp, hg); }
// N-C2 / W4-C3: record the allocation properties of the S, R and C windows (same query M4 driver.cpp l.712-729 uses). out[3*i+0]=hipMemGetAllocationPropertiesFromHandle type,
// [+1]=location type, [+2]=allocFlags/handle type word; out[9]=range extent S. Return 0 only if all queries succeeded. Uncached (type 0x40000000 on affected ROCm builds) => NOWB has NO power.
int32_t lg_comm_window_props(void* h, int64_t* out10) {
    auto* c = static_cast<LgCommCtx*>(h); void* vas[3] = {c->Sva, c->Rva, c->Cva}; int bad = 0;
    for (int i = 0; i < 3; ++i) {
        hipDeviceptr_t base = nullptr; size_t extent = 0; hipMemGenericAllocationHandle_t qh = nullptr; hipMemAllocationProp prop{};
        const hipError_t q1 = hipMemGetAddressRange(&base, &extent, static_cast<hipDeviceptr_t>(vas[i]));
        const hipError_t q2 = hipMemRetainAllocationHandle(&qh, vas[i]);
        const hipError_t q3 = (q2 == hipSuccess && qh) ? hipMemGetAllocationPropertiesFromHandle(&prop, qh) : hipErrorInvalidValue;
        out10[3 * i + 0] = static_cast<int64_t>(prop.type); out10[3 * i + 1] = static_cast<int64_t>(prop.location.type);
        out10[3 * i + 2] = static_cast<int64_t>(prop.requestedHandleType);
        if (i == 0) out10[9] = static_cast<int64_t>(extent);
        if (q1 != hipSuccess || q2 != hipSuccess || q3 != hipSuccess) bad = 1;
    }
    return bad ? -1 : 0;
}
}  // extern "C"

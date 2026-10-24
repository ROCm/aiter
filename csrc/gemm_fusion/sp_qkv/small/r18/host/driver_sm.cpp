// SPDX-License-Identifier: MIT
// SM-SK2-RAN-r1 host driver (UNCOMPILED). One process per rank. Window/CCO setup mirrors PR695-M4 host/driver.cpp:
// ccoCommCreate -> ccoWindowRegister (managed 4-arg overload) for Rx/C -> ccoDevCommCreate (GDA_NONE) -> hipModuleLoad
// of two hash-gated HSACOs. Launches sk_gemm then RAN on the caller stream; captures BOTH in a HIP graph.
// Plain host C++ (clang++ -x c++); never define __HIPCC__.
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mori/cco/cco.hpp>
#include "sm_abi_v2.h"
#include "sm_expected_hashes.h"
#include "sha256_image.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

namespace {
constexpr int OK = 0, E_ARG = -1, E_HIP = -2, E_CCO = -3, E_HASH = -4, E_STATE = -5,
          E_CCO_COMM = -10, E_CCO_TOPO = -11, E_CCO_WIN_RX = -12, E_CCO_WIN_C = -13, E_CCO_DEVCOMM = -14, E_STREAM_NULL = -15;
void sm_log(const char* stage, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::fprintf(stderr, "SM_SETUP stage=%s ", stage); std::vfprintf(stderr, fmt, ap); std::fputc(10, stderr);
    va_end(ap);
}
constexpr uint64_t RX_BYTES = 2ull * 4ull * SM_MP * SM_F * 2ull;        // [par][src][Mp][F] bf16
constexpr uint32_t RAN_GRID = SM_T1_GRID;                                // T1: exactly 53 static-role WGs (no tickets)
//                                       // tickets make co-residency unnecessary (S4 grid 1/4096 passed in r1); 128 kept

// Caller tensors/pointers (device addresses on this rank). Mirrors the dest-owned fields of SmArgsV1.
struct SmBindingV1 {
    const void* A; const void* W; const float* Ascale; const float* Wscale;
    const float* qw; const float* kw; const uint16_t* cos_cache; const uint16_t* sin_cache; int64_t cos_stride_pos;
    const int64_t* positions; const int64_t* slot_mapping;
    void* k_cache; void* v_cache; float* k_scale; float* v_scale;
    int64_t kc_sb, kc_sh, kc_sdx, kc_ss, kc_sx, vc_sb, vc_sh, vc_ssc, vc_sd, vc_sx, ks_sb, ks_sh, vs_sb, vs_sh;
    uint16_t* q_out; uint16_t* k_out; uint16_t* v_out; const float* iqw; const float* ikw; uint16_t* index_q_out; void* index_cache; const int64_t* index_slot_mapping;
    uint32_t M, T, block_size, x_size; uint32_t strideA; uint32_t pad0; float eps; float fp8_amax; uint64_t poll_iterations;
};
struct Ctx {
    int rank = -1; mori::cco::ccoComm* comm = nullptr; mori::cco::ccoDevComm dev{};
    mori::cco::ccoWindow_t rwin = nullptr, cwin = nullptr; void* rlocal = nullptr; void* clocal = nullptr;
    float* P = nullptr; hipModule_t mg = nullptr, mr = nullptr; hipFunction_t fg = nullptr, fr = nullptr;
    SmBindingV1 b{}; bool bound = false; SmDev* dblk = nullptr;
    uint8_t* cpeer[4] = {}; uint8_t* rpeer[4] = {}; uint64_t* dbg = nullptr; hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr; int last_hip = 0;
};
}

namespace { int32_t build_dev_block(Ctx* c); }
extern "C" {
int32_t sm_uid_v1(uint32_t rank, uint8_t* out128) {
    if (!out128 || rank > 3u) return E_ARG;
    std::memset(out128, 0, 128);
    if (rank != 0u) return OK;
    mori::cco::ccoUniqueId uid{};
    if (mori::cco::ccoGetUniqueId(&uid) != 0) return E_CCO;
    std::memcpy(out128, &uid, sizeof(uid) < 128 ? sizeof(uid) : 128);
    return OK;
}

int32_t sm_create_v1(uint32_t rank, const uint8_t* uid128, uint64_t per_rank_vmm_bytes,
                     const char* gemm_hsaco, const char* ran_hsaco, void** out_ctx) {
    if (!out_ctx) return E_ARG;
    *out_ctx = nullptr;
    if (!uid128 || !gemm_hsaco || !ran_hsaco || rank > 3u) { sm_log("args", "rc=FAIL"); return E_ARG; }
    auto* c = new Ctx(); *out_ctx = c; c->rank = static_cast<int>(rank);   // retained even on failure (quarantine, no cleanup)
    int cur = -1;
    if (hipSetDevice(c->rank) != hipSuccess || hipGetDevice(&cur) != hipSuccess || cur != c->rank) {
        sm_log("device", "requested=%d current=%d rc=FAIL", c->rank, cur); return E_HIP; }
    sm_log("device", "requested=%d current=%d rc=OK", c->rank, cur);
    int rc = mori::cco::ccoCommCreate(*reinterpret_cast<const mori::cco::ccoUniqueId*>(uid128), 4, c->rank, per_rank_vmm_bytes, &c->comm);
    if (rc != 0 || !c->comm) { sm_log("comm", "rc=FAIL ccoCommCreate=%d comm=%p rank=%d vmm=%llu", rc, (void*)c->comm, c->rank, (unsigned long long)per_rank_vmm_bytes); return E_CCO_COMM; }
    sm_log("comm", "rc=OK world=%d lsa=%d rank=%d lsaRank=%d myNodeStart=%d hipDev=%d vmmGran=%llu flatBase=%p perRank=%llu",
           (int)c->comm->worldSize, (int)c->comm->lsaSize, (int)c->comm->rank, (int)c->comm->lsaRank, (int)c->comm->myNodeStart,
           (int)c->comm->hipDev, (unsigned long long)c->comm->vmmGranularity, c->comm->flatBase, (unsigned long long)per_rank_vmm_bytes);
    if (c->comm->worldSize != 4 || c->comm->lsaSize != 4 || static_cast<int>(c->comm->rank) != c->rank
        || static_cast<int>(c->comm->lsaRank) != c->rank || c->comm->myNodeStart != 0) {
        sm_log("topology", "rc=FAIL (world/lsa/rank/lsaRank/myNodeStart mismatch)"); return E_CCO_TOPO; }
    // collective, same order/size on all ranks; BEFORE DevCommCreate. M4: C capacity = max(payload, 8192) via the managed
    // 4-arg overload (tiny-window registration was failing there); logical control content stays SM_C_BYTES.
    const uint64_t c_capacity = (RX_BYTES > 8192ull) ? RX_BYTES : 8192ull;
    rc = mori::cco::ccoWindowRegister(c->comm, static_cast<size_t>(RX_BYTES), &c->rwin, &c->rlocal);
    sm_log("alloc", "window=Rx capacity=%llu managed=1 rc=%d localPtr=%p win=%p", (unsigned long long)RX_BYTES, rc, c->rlocal, (void*)c->rwin);
    if (rc != 0 || !c->rlocal || !c->rwin) return E_CCO_WIN_RX;
    rc = mori::cco::ccoWindowRegister(c->comm, static_cast<size_t>(c_capacity), &c->cwin, &c->clocal);
    sm_log("alloc", "window=C capacity=%llu logical=%u managed=1 rc=%d localPtr=%p win=%p", (unsigned long long)c_capacity, SM_C_BYTES, rc, c->clocal, (void*)c->cwin);
    if (rc != 0 || !c->clocal || !c->cwin) return E_CCO_WIN_C;
    // U2: resolve the flat peer addresses ONCE on the host (windows are immutable for the graph lifetime); never dereferenced here
    for (int p = 0; p < 4; ++p) {
        c->cpeer[p] = static_cast<uint8_t*>(mori::cco::ccoGetPeerPtr(c->comm, c->clocal, p));
        c->rpeer[p] = static_cast<uint8_t*>(mori::cco::ccoGetPeerPtr(c->comm, c->rlocal, p));
        sm_log("peer", "p=%d cwin=%p rwin=%p", p, (void*)c->cpeer[p], (void*)c->rpeer[p]);
        if (!c->cpeer[p] || !c->rpeer[p]) { sm_log("peer", "rc=FAIL null peer pointer"); return E_CCO_WIN_C; }
        // r7: the in-kernel cross-check against ccoGetLsaPeerPtr is gone (SmDev holds raw pointers only): the self slot must equal the local window base
        if (p == c->rank && (c->cpeer[p] != static_cast<uint8_t*>(c->clocal) || c->rpeer[p] != static_cast<uint8_t*>(c->rlocal))) { sm_log("peer", "rc=FAIL self peer pointer != local window base (p=%d)", p); return E_CCO_WIN_C; }
    }
    mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_NONE;
    reqs.sdmaQueueCount = 1;
    rc = mori::cco::ccoDevCommCreate(c->comm, &reqs, &c->dev);
    sm_log("devcomm", "rc=%d sdmaNumQueue=%d", rc, (int)c->dev.sdma.sdmaNumQueue);
    if (rc != 0) return E_CCO_DEVCOMM;
    if (hipMalloc(reinterpret_cast<void**>(&c->P), sizeof(float) * SM_SK * SM_MP * SM_N) != hipSuccess) { sm_log("P", "hipMalloc rc=FAIL"); return E_HIP; }   // >=256B aligned
    // exact-image gate BEFORE any module action
    if (!sm_image_matches(gemm_hsaco, SM_EXPECTED_GEMM_SHA256) || !sm_image_matches(ran_hsaco, SM_EXPECTED_RAN_SHA256)) { sm_log("bundle", "rc=FAIL image hash mismatch"); return E_HASH; }
    if (hipModuleLoad(&c->mg, gemm_hsaco) != hipSuccess || hipModuleLoad(&c->mr, ran_hsaco) != hipSuccess) { sm_log("module", "rc=FAIL gemm=%s ran=%s", gemm_hsaco, ran_hsaco); return E_HIP; }
    if (hipModuleGetFunction(&c->fg, c->mg, "sm_gemm_v1") != hipSuccess || hipModuleGetFunction(&c->fr, c->mr, "sm_ran_v1") != hipSuccess) { sm_log("resolve", "rc=FAIL"); return E_HIP; }
#ifdef SM_STAMPS_HOST
    // SM_STAMPS diagnostic build only: rotating debug buffer, zero-initialised (dbg[0] = replay counter)
    if (hipMalloc(reinterpret_cast<void**>(&c->dbg), sizeof(uint64_t) * SM_DBG_TOTAL_WORDS) != hipSuccess || hipMemset(c->dbg, 0, sizeof(uint64_t) * SM_DBG_TOTAL_WORDS) != hipSuccess) { sm_log("dbg", "rc=FAIL"); return E_HIP; }
    sm_log("dbg", "rc=OK words=%u", (unsigned)SM_DBG_TOTAL_WORDS);
#endif
    sm_log("create", "rc=OK");
    return OK;
}
// Zero Rx + control window (epoch/tickets/flags/credits/poison). Run once after create, and again after ANY poison.
// ALL ranks must call it and then pass a cross-rank barrier BEFORE the first launch (peers write into our windows).
// All ranks must run identical warmup/replay counts: epochs are device-owned and advance once per RAN launch.
int32_t sm_reset_control_v1(void* ctx) {
    auto* c = static_cast<Ctx*>(ctx); if (!c) return E_ARG;
    if (hipSetDevice(c->rank) != hipSuccess) return E_HIP;
    if (hipMemset(c->clocal, 0, SM_C_BYTES) != hipSuccess || hipMemset(c->rlocal, 0, RX_BYTES) != hipSuccess) return E_HIP;
    return hipDeviceSynchronize() == hipSuccess ? OK : E_HIP;
}
int32_t sm_bind_v1(void* ctx, const SmBindingV1* b) {
    auto* c = static_cast<Ctx*>(ctx); if (!c || !b) return E_ARG;
    #if SM_B_ROWSPLIT
    if (b->M <= 16u) { sm_log("bind", "rc=FAIL SM_B_ROWSPLIT build supports M>16 only (M=%u)", b->M); return E_ARG; }          // r18 R4
#endif
#if SM_PUSH8
    if (b->M > 16u) { sm_log("bind", "rc=FAIL SM_PUSH8 build supports M<=16 only (M=%u)", b->M); return E_ARG; }          // r15 Q1: one row pass per half
#endif
    if (b->M == 0u || b->M > SM_MP || b->T != 4u * b->M) { sm_log("bind", "rc=FAIL M=%u T=%u (need 1<=M<=%u, T==4*M)", b->M, b->T, SM_MP); return E_ARG; }
    {   // r6: the kernel issues UNCONDITIONAL 16 B vector loads (no runtime alignment branch): every such pointer/stride must be 16 B aligned
        auto al = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u; };
        const bool okal = al(b->Wscale) && al(b->qw) && al(b->kw) && al(b->cos_cache) && al(b->sin_cache) && al(b->iqw) && al(b->ikw) && al(c->P) && al(c->rlocal)
                          && (b->cos_stride_pos % 8 == 0) && (b->cos_stride_pos > 0);
        if (!okal) { sm_log("bind", "rc=FAIL 16 B alignment (Wscale/qw/kw/cos/sin/P/Rx) or cos_stride_pos %% 8 != 0"); return E_ARG; }
    }
    c->b = *b; c->bound = true;
    return build_dev_block(c);
}
}

namespace {
// r7 (K1): build the device-resident immutable SmDev block ONCE per bind (outside capture): hipMalloc (256 B aligned, kept for the context lifetime), one H2D copy.
int32_t build_dev_block(Ctx* c) {
    if (hipSetDevice(c->rank) != hipSuccess) return E_HIP;
    if (!c->dblk && hipMalloc(reinterpret_cast<void**>(&c->dblk), sizeof(SmDev)) != hipSuccess) { sm_log("dblk", "hipMalloc rc=FAIL"); return E_HIP; }
    SmDev h{};
    h.P = c->P; h.Wscale = c->b.Wscale; h.Ascale = c->b.Ascale; h.cwin_loc = static_cast<uint8_t*>(c->clocal);
    h.rwin_pr0 = c->rpeer[0]; h.rwin_pr1 = c->rpeer[1]; h.rwin_pr2 = c->rpeer[2]; h.rwin_pr3 = c->rpeer[3];
    h.cwin_pr0 = c->cpeer[0]; h.cwin_pr1 = c->cpeer[1]; h.cwin_pr2 = c->cpeer[2]; h.cwin_pr3 = c->cpeer[3];
    h.poll_iterations = c->b.poll_iterations; h.dbg = c->dbg; h.rwin_loc = static_cast<uint8_t*>(c->rlocal); h.cos_stride_pos = c->b.cos_stride_pos;
    h.qw = c->b.qw; h.kw = c->b.kw; h.cos_cache = c->b.cos_cache; h.sin_cache = c->b.sin_cache;
    h.positions = c->b.positions; h.slot_mapping = c->b.slot_mapping; h.q_out = c->b.q_out; h.k_out = c->b.k_out;
    h.v_out = c->b.v_out; h.k_cache = c->b.k_cache; h.v_cache = c->b.v_cache; h.k_scale = c->b.k_scale; h.v_scale = c->b.v_scale;
    h.kc_sb = c->b.kc_sb; h.kc_sdx = c->b.kc_sdx; h.kc_ss = c->b.kc_ss; h.kc_sx = c->b.kc_sx;
    h.vc_sb = c->b.vc_sb; h.vc_ssc = c->b.vc_ssc; h.vc_sd = c->b.vc_sd; h.vc_sx = c->b.vc_sx; h.ks_sb = c->b.ks_sb; h.vs_sb = c->b.vs_sb;
    h.iqw = c->b.iqw; h.ikw = c->b.ikw; h.index_q_out = c->b.index_q_out; h.index_cache = c->b.index_cache; h.index_slot_mapping = c->b.index_slot_mapping;
    h.block_size = c->b.block_size; h.x_size = c->b.x_size; h.T = c->b.T; h.eps = c->b.eps; h.fp8_amax = c->b.fp8_amax;
    if (hipMemcpy(c->dblk, &h, sizeof(h), hipMemcpyHostToDevice) != hipSuccess) { sm_log("dblk", "H2D rc=FAIL"); return E_HIP; }
    return OK;
}
hipError_t enqueue(Ctx* c, hipStream_t s) {
    SmGemmArgsV1 g{}; g.A = c->b.A; g.W = c->b.W; g.P = c->P; g.M = c->b.M; g.N = SM_N; g.K = 6144u; g.strideA = c->b.strideA; g.sk = SM_SK; g.Cwin = c->cwin; g.dbg = c->dbg;
    // r7 fix: hipModuleLaunchKernel(kernelParams) needs ONE pointer PER kernel parameter. sm_ran_v1(const SmDev* blk, uint32_t M, uint32_t rank) has THREE parameters
    // (the earlier single `&ranargs` was one pointer for three parameters -> the kernel read garbage pointers/values -> GPU memory access fault). The GEMM has ONE by-value struct
    // parameter, so its single `&g` is correct. Both stay in the kernelParams convention (never mixed with the `extra` buffer mode).
    const SmDev* ran_blk = c->dblk; uint32_t ran_M = c->b.M; uint32_t ran_rank = static_cast<uint32_t>(c->rank);
    static_assert(sizeof(const SmDev*) == 8 && sizeof(uint32_t) == 4, "sm_ran_v1 parameter sizes (const SmDev*, uint32_t, uint32_t) = 8 + 4 + 4 = 16 B kernarg");
    static_assert(sizeof(SmRanArgs) == sizeof(const SmDev*) + 2 * sizeof(uint32_t), "SmRanArgs mirrors the three kernel parameters (documentation struct only; not passed)");
    void* ga[] = {&g};
    void* ra[] = {&ran_blk, &ran_M, &ran_rank};
    hipError_t e = hipModuleLaunchKernel(c->fg, SM_GEMM_NTILES, SM_SK, 1, 256, 1, 1, 0, s, ga, nullptr);
    if (e != hipSuccess) return e;
    return hipModuleLaunchKernel(c->fr, RAN_GRID, 1, 1, SM_RAN_THREADS, 1, 1, 0, s, ra, nullptr);          // r15: 256 (default) or 512 (PUSH8 / PUSH_HALVES)
}
}

extern "C" {
int32_t sm_launch_eager_v1(void* ctx, void* stream) {
    auto* c = static_cast<Ctx*>(ctx); if (!c || !c->bound) return E_STATE;
    const hipError_t e = enqueue(c, static_cast<hipStream_t>(stream));
    if (e != hipSuccess) { c->last_hip = static_cast<int>(e); std::fprintf(stderr, "SM_LAUNCH stage=enqueue err=%d (%s)\n", (int)e, hipGetErrorString(e)); return E_HIP; }
    return OK;
}
// Capture exactly TWO kernels (sk_gemm, RAN) into a graph. Call OUTSIDE any other capture, after warmup; no allocation inside.
// The stream MUST be a dedicated non-default stream (null/legacy default stream cannot be captured).
int32_t sm_capture_v1(void* ctx, void* stream) {
    auto* c = static_cast<Ctx*>(ctx); if (!c || !c->bound) return E_STATE;
    if (!stream) { std::fprintf(stderr, "SM_CAPTURE stage=args err=stream==nullptr (default stream cannot be captured)\n"); return E_STREAM_NULL; }
    hipStream_t s = static_cast<hipStream_t>(stream);
    hipError_t e = hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal);
    if (e != hipSuccess) { c->last_hip = static_cast<int>(e); std::fprintf(stderr, "SM_CAPTURE stage=begin err=%d (%s)\n", (int)e, hipGetErrorString(e)); return E_HIP; }
    const hipError_t ee = enqueue(c, s);
    if (ee != hipSuccess) { c->last_hip = static_cast<int>(ee); std::fprintf(stderr, "SM_CAPTURE stage=enqueue err=%d (%s)\n", (int)ee, hipGetErrorString(ee)); }
    e = hipStreamEndCapture(s, &c->graph);    // always end the capture, even after an enqueue failure
    if (e != hipSuccess) { c->last_hip = static_cast<int>(e); std::fprintf(stderr, "SM_CAPTURE stage=end err=%d (%s)\n", (int)e, hipGetErrorString(e)); return E_HIP; }
    if (ee != hipSuccess) return E_HIP;
    size_t n = 0;
    e = hipGraphGetNodes(c->graph, nullptr, &n);
    if (e != hipSuccess || n != 2) { c->last_hip = static_cast<int>(e); std::fprintf(stderr, "SM_CAPTURE stage=nodes err=%d nodes=%zu (expect 2)\n", (int)e, n); return E_STATE; }
    e = hipGraphInstantiate(&c->exec, c->graph, nullptr, nullptr, 0);
    if (e != hipSuccess) { c->last_hip = static_cast<int>(e); std::fprintf(stderr, "SM_CAPTURE stage=instantiate err=%d (%s)\n", (int)e, hipGetErrorString(e)); return E_HIP; }
    return OK;
}
int32_t sm_replay_v1(void* ctx, void* stream) {
    auto* c = static_cast<Ctx*>(ctx); if (!c || !c->exec) return E_STATE;
    const hipError_t e = hipGraphLaunch(c->exec, static_cast<hipStream_t>(stream));
    if (e != hipSuccess) { c->last_hip = static_cast<int>(e); std::fprintf(stderr, "SM_REPLAY stage=launch err=%d (%s)\n", (int)e, hipGetErrorString(e)); return E_HIP; }
    return OK;
}
// SM_STAMPS: copy the whole debug buffer to host (out must hold SM_DBG_TOTAL_WORDS u64); returns E_STATE if not a stamps build.
int32_t sm_dbg_read_v1(void* ctx, uint64_t* out) {
    auto* c = static_cast<Ctx*>(ctx); if (!c || !out) return E_ARG;
    if (!c->dbg) return E_STATE;
    return hipMemcpy(out, c->dbg, sizeof(uint64_t) * SM_DBG_TOTAL_WORDS, hipMemcpyDeviceToHost) == hipSuccess ? OK : E_HIP;
}
// Last failing hipError_t (as int) recorded by launch/capture/replay; 0 if none.
int32_t sm_last_hip_error_v1(void* ctx) { auto* c = static_cast<Ctx*>(ctx); return c ? c->last_hip : -1; }// Poison word (0 = healthy). Host must reset_control on ALL ranks after nonzero.
int32_t sm_poison_v1(void* ctx, uint32_t* out) {
    auto* c = static_cast<Ctx*>(ctx); if (!c || !out) return E_ARG;
    return hipMemcpy(out, static_cast<uint8_t*>(c->clocal) + SM_C_POISON, 4, hipMemcpyDeviceToHost) == hipSuccess ? OK : E_HIP;
}
}

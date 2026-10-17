// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gemm_fusion small-M host driver = the accepted Phase-1 driver_sm.cpp (included UNMODIFIED from the per-source vendored
// host/ directory selected by -I) plus HOST-ONLY additions. Kernels, device code, ABI headers and the Phase-1 create/bind/
// capture/replay entry points are untouched. Deviations (recorded in aiter-integration-notes.md):
//   D1 sm_attach_v1: uses a process-shared comm and already-registered Rx/C windows instead of creating its own comm
//      (no devcomm: the small-M kernels only dereference the Cwin handle through ccoGetLocalPtr; the shared devcomm is created by gf_comm).
//   D2 sm_bind_new_v1 / sm_launch_blk_v1 / sm_blk_free_v1: several bindings (one per site/layer) can share one context
//      (windows+epoch); each owns its device-resident SmDev block. A and W are passed per launch (as kernargs, like Phase-1).
//   D3 sm_info_v1: exports Rx/control byte sizes so the runtime sizes windows from the compiled geometry.
#include "driver_sm.cpp"

namespace {
struct Blk { SmDev* dblk = nullptr; SmBindingV1 b{}; };
bool bind_ok(Ctx* c, const SmBindingV1* b) {
    // identical checks to sm_bind_v1
#if SM_B_ROWSPLIT
    if (b->M <= 16u) { sm_log("bind", "rc=FAIL SM_B_ROWSPLIT build supports M>16 only (M=%u)", b->M); return false; }
#endif
#if SM_PUSH8
    if (b->M > 16u) { sm_log("bind", "rc=FAIL SM_PUSH8 build supports M<=16 only (M=%u)", b->M); return false; }
#endif
    if (b->M == 0u || b->M > SM_MP || b->T != 4u * b->M) { sm_log("bind", "rc=FAIL M=%u T=%u", b->M, b->T); return false; }
    auto al = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u; };
    const bool okal = al(b->Wscale) && al(b->qw) && al(b->kw) && al(b->cos_cache) && al(b->sin_cache) && al(b->iqw) && al(b->ikw) && al(c->P) && al(c->rlocal)
                      && (b->cos_stride_pos % 8 == 0) && (b->cos_stride_pos > 0);
    if (!okal) { sm_log("bind", "rc=FAIL 16 B alignment or cos_stride_pos %% 8 != 0"); return false; }
    return true;
}
}  // namespace

extern "C" {
int32_t sm_attach_v1(uint32_t rank, void* comm, void* rwin, void* rlocal, void* cwin, void* clocal,
                     const char* gemm_hsaco, const char* ran_hsaco, void** out_ctx) {
    if (!out_ctx) return E_ARG;
    *out_ctx = nullptr;
    if (!comm || !rwin || !rlocal || !cwin || !clocal || !gemm_hsaco || !ran_hsaco || rank > 3u) return E_ARG;
    auto* c = new Ctx(); *out_ctx = c; c->rank = static_cast<int>(rank);
    if (hipSetDevice(c->rank) != hipSuccess) return E_HIP;
    c->comm = static_cast<mori::cco::ccoComm*>(comm);
    c->rwin = reinterpret_cast<mori::cco::ccoWindow_t>(rwin); c->rlocal = rlocal;
    c->cwin = reinterpret_cast<mori::cco::ccoWindow_t>(cwin); c->clocal = clocal;
    for (int p = 0; p < 4; ++p) {   // same peer resolution + self check as sm_create_v1
        c->cpeer[p] = static_cast<uint8_t*>(mori::cco::ccoGetPeerPtr(c->comm, c->clocal, p));
        c->rpeer[p] = static_cast<uint8_t*>(mori::cco::ccoGetPeerPtr(c->comm, c->rlocal, p));
        if (!c->cpeer[p] || !c->rpeer[p]) { sm_log("attach", "rc=FAIL null peer pointer p=%d", p); return E_CCO_WIN_C; }
        if (p == c->rank && (c->cpeer[p] != static_cast<uint8_t*>(c->clocal) || c->rpeer[p] != static_cast<uint8_t*>(c->rlocal))) {
            sm_log("attach", "rc=FAIL self peer pointer != local window base"); return E_CCO_WIN_C; }
    }
    if (hipMalloc(reinterpret_cast<void**>(&c->P), sizeof(float) * SM_SK * SM_MP * SM_N) != hipSuccess) { sm_log("P", "hipMalloc rc=FAIL"); return E_HIP; }
    if (!sm_image_matches(gemm_hsaco, SM_EXPECTED_GEMM_SHA256) || !sm_image_matches(ran_hsaco, SM_EXPECTED_RAN_SHA256)) { sm_log("bundle", "rc=FAIL image hash mismatch"); return E_HASH; }
    if (hipModuleLoad(&c->mg, gemm_hsaco) != hipSuccess || hipModuleLoad(&c->mr, ran_hsaco) != hipSuccess) { sm_log("module", "rc=FAIL"); return E_HIP; }
    if (hipModuleGetFunction(&c->fg, c->mg, "sm_gemm_v1") != hipSuccess || hipModuleGetFunction(&c->fr, c->mr, "sm_ran_v1") != hipSuccess) { sm_log("resolve", "rc=FAIL"); return E_HIP; }
    sm_log("attach", "rc=OK");
    return OK;
}
// Create a binding (device-resident SmDev block). OUTSIDE graph capture only (hipMalloc + H2D). Pointers must stay valid for the binding's lifetime.
int32_t sm_bind_new_v1(void* ctx, const SmBindingV1* b, void** out_blk) {
    auto* c = static_cast<Ctx*>(ctx); if (!c || !b || !out_blk) return E_ARG;
    *out_blk = nullptr;
    if (!bind_ok(c, b)) return E_ARG;
    SmBindingV1 saved_b = c->b; bool saved_bound = c->bound; SmDev* saved_dblk = c->dblk;
    c->b = *b; c->dblk = nullptr;
    const int32_t rc = build_dev_block(c);
    auto* blk = new Blk(); blk->dblk = c->dblk; blk->b = *b;
    c->b = saved_b; c->bound = saved_bound; c->dblk = saved_dblk;
    if (rc != OK) { delete blk; return rc; }
    *out_blk = blk;
    return OK;
}
// Enqueue sm_gemm then RAN on `stream` for this binding (capturable by torch.cuda.graph: plain kernel launches).
int32_t sm_launch_blk_v1(void* ctx, void* blk_, const void* A, const void* W, void* stream) {
    auto* c = static_cast<Ctx*>(ctx); auto* blk = static_cast<Blk*>(blk_); if (!c || !blk || !blk->dblk) return E_STATE;
    SmBindingV1 saved_b = c->b; SmDev* saved_dblk = c->dblk;
    c->b = blk->b; c->b.A = A; c->b.W = W; c->dblk = blk->dblk;
    const hipError_t e = enqueue(c, static_cast<hipStream_t>(stream));
    c->b = saved_b; c->dblk = saved_dblk;
    if (e != hipSuccess) { c->last_hip = static_cast<int>(e); std::fprintf(stderr, "SM_LAUNCH stage=enqueue err=%d (%s)\n", (int)e, hipGetErrorString(e)); return E_HIP; }
    return OK;
}
int32_t sm_blk_free_v1(void* blk_) {   // call only after quiesce (device sync)
    auto* blk = static_cast<Blk*>(blk_); if (!blk) return E_ARG;
    if (blk->dblk) hipFree(blk->dblk);
    delete blk; return OK;
}
// Compiled geometry: out[0]=Rx bytes, out[1]=control logical bytes, out[2]=N, out[3]=F, out[4]=ran grid, out[5]=ran threads, out[6]=gemm ntiles.
int32_t sm_info_v1(uint64_t* out7) {
    if (!out7) return E_ARG;
    out7[0] = RX_BYTES; out7[1] = SM_C_BYTES; out7[2] = SM_N; out7[3] = SM_F; out7[4] = RAN_GRID; out7[5] = SM_RAN_THREADS; out7[6] = SM_GEMM_NTILES;
    return OK;
}
}

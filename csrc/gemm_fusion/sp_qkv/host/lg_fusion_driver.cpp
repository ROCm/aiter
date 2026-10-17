// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gemm_fusion large-M host driver = the accepted Phase-1 LG-QSDMA-r11 lg_comm_driver.cpp (included UNMODIFIED via -I) plus
// HOST-ONLY additions. Kernels, device code, ABI header and the r11 entry points (lg_comm_launch, lg_comm_init_control,
// lg_comm_drain_all, lg_comm_read_report, ...) are untouched. Deviations (recorded in aiter-integration-notes.md):
//   D1 lg_attach_v1: uses the process-shared comm + devcomm (gf_comm) and three already-registered equal-size windows
//      S/R/C instead of creating its own comm (lg_comm_create). Same checks as lg_comm_create after window creation.
//      H-3: exact code-object image gate (sha256 of the HSACO must equal the caller-supplied expected hex, as small-M's
//      sm_image_matches gate does) BEFORE hipModuleLoad.
//   D2 lg_info_v1 / lg_window_bytes_v1: build constants and the window size the runtime must register.
//   D3 lg_free_v1 (L-5): unload the module and delete the context (call after a device sync, before comm teardown).
//   D4 lg_report_fast_v1: poison word only (4-byte D2H on the given stream), for the per-step host check.
#include "lg_comm_driver.cpp"
#include "sha256_image.hpp"

extern "C" {
// Window bytes every one of S, R, C must have for this build and M: roundup(max(P + pad, stamp area), 4 MiB) (r8g rule).
uint64_t lg_window_bytes_v1(uint32_t M) {
    const uint64_t P = 8ull * M * LG_COMM_F;
    const uint64_t stamp_bytes = LGC_STAMP_OFF + 64ull * 4096ull;
    const uint64_t cap = (P > stamp_bytes ? P : stamp_bytes);
    const uint64_t need = (P + LG_R_PAD_BYTES) > cap ? (P + LG_R_PAD_BYTES) : cap;
    const uint64_t gran = 4ull << 20;
    return (need + gran - 1) / gran * gran;
}
// out[0]=F, out[1]=N, out[2]=indexer, out[3]=fold, out[4]=R pad, out[5]=sizeof(ccoDevComm)
int32_t lg_info_v1(uint64_t* out6) {
    if (!out6) return -1;
    out6[0] = LG_COMM_F; out6[1] = LG_X_N_RAW; out6[2] = LG_X_INDEXER; out6[3] = LG_R8_FOLD; out6[4] = LG_R_PAD_BYTES; out6[5] = sizeof(mori::cco::ccoDevComm);
    return 0;
}
void* lg_attach_v1(uint32_t rank, uint32_t M, void* comm, const void* devcomm, void* S, void* Sva, void* R, void* Rva, void* C, void* Cva,
                   const char* hsaco_path, const char* expected_hsaco_sha256_hex, uint32_t sdma_queues) {
    if (rank > 3u || M == 0 || M % 256u || M > 8192u || sdma_queues != 1u || !comm || !devcomm || !S || !R || !C || !Sva || !Rva || !Cva || !hsaco_path || !expected_hsaco_sha256_hex) {
        std::fprintf(stderr, "LG_ATTACH rc=FAIL stage=args rank=%u M=%u Q=%u comm=%p dev=%p S=%p R=%p C=%p Sva=%p Rva=%p Cva=%p hsaco=%p sha=%p\n", rank, M, sdma_queues, comm, devcomm, S, R, C, Sva, Rva, Cva,
                     static_cast<const void*>(hsaco_path), static_cast<const void*>(expected_hsaco_sha256_hex));
        return nullptr;
    }
    if (!sm_image_matches(hsaco_path, expected_hsaco_sha256_hex)) { std::fprintf(stderr, "LG_ATTACH rc=FAIL image hash mismatch (%s)\n", hsaco_path); return nullptr; }   // H-3: exact-image gate BEFORE any module action
    if (const hipError_t e = hipSetDevice(static_cast<int>(rank)); e != hipSuccess) { std::fprintf(stderr, "LG_ATTACH rc=FAIL stage=hipSetDevice rank=%u err=%d (%s)\n", rank, static_cast<int>(e), hipGetErrorString(e)); return nullptr; }
    auto* c = new LgCommCtx();
    c->rank = rank; c->M = M; c->Q = sdma_queues;
    c->comm = static_cast<mori::cco::ccoComm*>(comm);
    c->dev = *static_cast<const mori::cco::ccoDevComm*>(devcomm);
    c->S = reinterpret_cast<mori::cco::ccoWindow_t>(S); c->R = reinterpret_cast<mori::cco::ccoWindow_t>(R); c->C = reinterpret_cast<mori::cco::ccoWindow_t>(C);
    c->Sva = Sva; c->Rva = Rva; c->Cva = Cva;
    if (c->comm->worldSize != 4 || c->comm->lsaSize != 4 || static_cast<uint32_t>(c->comm->lsaRank) != rank || c->comm->myNodeStart != 0) {
        std::fprintf(stderr, "LG_ATTACH rc=FAIL stage=topology rank=%u world=%d lsaSize=%d lsaRank=%d myNodeStart=%d (need 4/4/rank/0)\n", rank,
                     static_cast<int>(c->comm->worldSize), static_cast<int>(c->comm->lsaSize), static_cast<int>(c->comm->lsaRank), static_cast<int>(c->comm->myNodeStart));
        delete c; return nullptr;
    }
    if (static_cast<uint32_t>(c->dev.sdma.sdmaNumQueue) != sdma_queues) {       // CCO may cap Q: report, never silently reduce
        std::fprintf(stderr, "LG_ATTACH rc=FAIL stage=sdma_queues rank=%u devcomm sdmaNumQueue=%d but the kernels need %u\n", rank, static_cast<int>(c->dev.sdma.sdmaNumQueue), sdma_queues);
        delete c; return nullptr;
    }
    if (const hipError_t e = hipModuleLoad(&c->mod, hsaco_path); e != hipSuccess) { std::fprintf(stderr, "LG_ATTACH rc=FAIL stage=hipModuleLoad rank=%u err=%d (%s) path=%s\n", rank, static_cast<int>(e), hipGetErrorString(e), hsaco_path); delete c; return nullptr; }
    struct { hipFunction_t* f; const char* name; } fns[] = {{&c->fgemm[0], LG_ENTRY_GEMM_N1}, {&c->fgemm[1], LG_ENTRY_GEMM_N2}, {&c->fgemm[2], LG_ENTRY_GEMM_N4}, {&c->fgemm[3], LG_ENTRY_GEMM_N2H},
                                                 {&c->fgemm[4], LG_ENTRY_GEMM_N4H}, {&c->fgemm[5], LG_ENTRY_GEMM_N3Q}, {&c->fquiet, LG_ENTRY_QUIET}, {&c->fdrain, LG_ENTRY_DRAIN}};
    for (auto& fn : fns) if (const hipError_t e = hipModuleGetFunction(fn.f, c->mod, fn.name); e != hipSuccess) {
        std::fprintf(stderr, "LG_ATTACH rc=FAIL stage=entry rank=%u symbol=%s err=%d (%s)\n", rank, fn.name, static_cast<int>(e), hipGetErrorString(e));
        hipModuleUnload(c->mod); delete c; return nullptr;
    }
    return c;
}
// L-5: call after hipDeviceSynchronize and BEFORE ccoCommDestroy. Windows are owned by the runtime (FusionResources), not freed here.
int32_t lg_free_v1(void* h) {
    auto* c = static_cast<LgCommCtx*>(h);
    if (!c) return -1;
    if (hipSetDevice(static_cast<int>(c->rank)) != hipSuccess) return -2;
    if (c->mod) hipModuleUnload(c->mod);
    delete c;
    return 0;
}
// Poison word (u32 at LGC_POISON_OFF of the control window): 4-byte D2H on `stream`, then stream sync. Cheap: no 8 KiB image copy.
int32_t lg_poison_v1(void* h, void* stream, uint32_t* out) {
    auto* c = static_cast<LgCommCtx*>(h);
    if (!c || !out) return -1;
    const hipStream_t s = static_cast<hipStream_t>(stream);
    if (hipMemcpyAsync(out, static_cast<uint8_t*>(c->Cva) + LGC_POISON_OFF, 4, hipMemcpyDeviceToHost, s) != hipSuccess) return -2;
    return hipStreamSynchronize(s) == hipSuccess ? 0 : -3;
}
}

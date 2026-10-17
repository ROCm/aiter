// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gemm_fusion large-M host driver = the accepted Phase-1 LG-QSDMA-r11 lg_comm_driver.cpp (included UNMODIFIED via -I) plus
// HOST-ONLY additions. Kernels, device code, ABI header and the r11 entry points (lg_comm_launch, lg_comm_init_control,
// lg_comm_drain_all, lg_comm_read_report, ...) are untouched. Deviations (recorded in aiter-integration-notes.md):
//   D1 lg_attach_v1: uses the process-shared comm + devcomm (gf_comm) and three already-registered equal-size windows
//      S/R/C instead of creating its own comm (lg_comm_create). Same checks as lg_comm_create after window creation.
//   D2 lg_info_v1: exports build constants and the window size the runtime must register (same formula as lg_comm_create).
#include "lg_comm_driver.cpp"

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
                   const char* hsaco_path, uint32_t sdma_queues) {
    if (rank > 3u || M == 0 || M % 256u || M > 8192u || sdma_queues != 1u || !comm || !devcomm || !S || !R || !C || !Sva || !Rva || !Cva || !hsaco_path) return nullptr;
    if (hipSetDevice(static_cast<int>(rank)) != hipSuccess) return nullptr;
    auto* c = new LgCommCtx();
    c->rank = rank; c->M = M; c->Q = sdma_queues;
    c->comm = static_cast<mori::cco::ccoComm*>(comm);
    c->dev = *static_cast<const mori::cco::ccoDevComm*>(devcomm);
    c->S = reinterpret_cast<mori::cco::ccoWindow_t>(S); c->R = reinterpret_cast<mori::cco::ccoWindow_t>(R); c->C = reinterpret_cast<mori::cco::ccoWindow_t>(C);
    c->Sva = Sva; c->Rva = Rva; c->Cva = Cva;
    if (c->comm->worldSize != 4 || c->comm->lsaSize != 4 || static_cast<uint32_t>(c->comm->lsaRank) != rank || c->comm->myNodeStart != 0) { delete c; return nullptr; }
    if (static_cast<uint32_t>(c->dev.sdma.sdmaNumQueue) != sdma_queues) { delete c; return nullptr; }       // CCO may cap Q: report, never silently reduce
    if (hipModuleLoad(&c->mod, hsaco_path) != hipSuccess) { delete c; return nullptr; }
    struct { hipFunction_t* f; const char* name; } fns[] = {{&c->fgemm[0], LG_ENTRY_GEMM_N1}, {&c->fgemm[1], LG_ENTRY_GEMM_N2}, {&c->fgemm[2], LG_ENTRY_GEMM_N4}, {&c->fgemm[3], LG_ENTRY_GEMM_N2H},
                                                 {&c->fgemm[4], LG_ENTRY_GEMM_N4H}, {&c->fgemm[5], LG_ENTRY_GEMM_N3Q}, {&c->fquiet, LG_ENTRY_QUIET}, {&c->fdrain, LG_ENTRY_DRAIN}};
    for (auto& fn : fns) if (hipModuleGetFunction(fn.f, c->mod, fn.name) != hipSuccess) { delete c; return nullptr; }
    return c;
}
}

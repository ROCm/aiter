// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SDMA device functions of the big-M MegaMoE TP kernel, on MORI's header-only CCO device API
// (ccoSdma: an installed MORI built with BUILD_CCO_SDMA=ON; no change to MORI). sdma.py compiles
// this file to LLVM bitcode with MORI's own JIT toolchain (mori.jit: the same hipcc flags /
// include dirs as MORI's device wrapper) and the FlyDSL kernels link it (link_extern).

#include "mori/cco/cco_scale_out.hpp"

using namespace mori::cco;

#define AITER_CCO_DEV extern "C" __device__ __attribute__((always_inline, visibility("default")))

// Remote-signalled put (thread scope) on raw addresses: dst a peer's buffer mapped into this
// process (hipIpc), src a local one (the data need not live in a MORI window, see sdma.py); once
// the copy landed, the peer's signalBuf[myLsaRank * sdmaNumQueue + qid] += 1 (polled through
// aiter_cco_sdma_signal_slot).
AITER_CCO_DEV int aiter_cco_sdma_put_ptr_rsig__thread(uint64_t dc, int peer, uint64_t dst, uint64_t src,
                                                      uint64_t n, int qid) {
  const ccoDevComm& comm = *reinterpret_cast<const ccoDevComm*>(dc);
  const ccoSdmaContext& s = comm.sdma;
  ccoSdmaDispatchChunked<ccoCoopThread, false, true, ccoSdmaOptFlagsDefault, ccoSdmaThreadIndependent>(
      s, peer, s.sdmaNumQueue, reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), n, qid,
      static_cast<uint32_t>(comm.lsaRank), static_cast<uint32_t>(comm.lsaSize));
  return 0;
}

// Address of this rank's signal slot counting src's remote-signalled puts on queue qid.
AITER_CCO_DEV uint64_t aiter_cco_sdma_signal_slot(uint64_t dc, int src, int qid) {
  const ccoSdmaContext& s = reinterpret_cast<const ccoDevComm*>(dc)->sdma;
  return reinterpret_cast<uint64_t>(s.signalBuf + static_cast<uint32_t>(src) * s.sdmaNumQueue +
                                    static_cast<uint32_t>(qid));
}

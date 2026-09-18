// SPDX-License-Identifier: MIT
// Minimal device shim over MORI's public ccoGda API, for the two things the
// public FlyDSL binding (mori.cco.device.flydsl) cannot express:
//
//  1. THE RAIL TEAM. Every GDA entry point in MORI's wrapper takes the
//     `ccoTeamMode` template default, CCO_TEAM_WORLD (cco_scale_out.hpp:212),
//     so `peer` is resolved as a world rank. The cross-node return path
//     addresses a NODE index through CCO_TEAM_GDA, which no exported symbol
//     reaches.
//  2. DEFERRED DOORBELL. `ccoGdaOptFlagsAggregateRequests` posts the WQE
//     without ringing the doorbell, so a whole chunk batch can be queued and
//     then released by one put_value + flush per QP. MORI's wrapper never
//     passes optFlags on the GDA path (it does for SDMA), and this is a
//     DIFFERENT axis from `ccoGdaThreadAggregate` (the "at" symbol), which
//     coalesces a warp's lanes into one transfer and does not defer anything.
//
// Everything else -- lsa_ptr, the waits, window management, the whole host
// side -- goes through the public MORI API directly. Delete this file the day
// MORI's wrapper exposes a team argument and optFlags on GDA.
//
// Only the (rail, warp) instantiation exists because that is the only one the
// operator uses: kernel2 rejects any other team (stage2_node_combine.py), and
// no call site asks for block coop.

#include "mori/cco/cco_scale_out.hpp"

namespace {
using namespace mori::cco;
using Gda = ccoGda<CCO_GDA_BUILD_PROVIDER>;

inline __device__ const ccoDevComm* AsDevComm(uint64_t h) {
  return reinterpret_cast<const ccoDevComm*>(h);
}
inline __device__ ccoWindow_t AsWindow(uint64_t h) {
  return reinterpret_cast<ccoWindow_t>(h);
}
inline __device__ uint32_t OptFlags(int aggregate) {
  return aggregate ? ccoGdaOptFlagsAggregateRequests : ccoGdaOptFlagsDefault;
}
}  // namespace

#define MEGAMOE_GDA_DEV \
  extern "C" __device__ __attribute__((always_inline, visibility("default")))

MEGAMOE_GDA_DEV void megamoe_gda_put_rail_warp(
    uint64_t dc, int ctx, int peer, uint64_t dst, uint64_t dstOff, uint64_t src,
    uint64_t srcOff, uint64_t bytes, int aggregate) {
  Gda gda{*AsDevComm(dc), ctx};
  gda.put<CCO_TEAM_GDA, ccoGdaThreadIndependent>(
      peer, AsWindow(dst), dstOff, AsWindow(src), srcOff, bytes,
      ccoGda_NoSignal{}, ccoCoopWarp{}, OptFlags(aggregate));
}

MEGAMOE_GDA_DEV void megamoe_gda_put_value_rail_warp(
    uint64_t dc, int ctx, int peer, uint64_t dst, uint64_t dstOff,
    uint64_t value, int aggregate) {
  Gda gda{*AsDevComm(dc), ctx};
  gda.putValue<CCO_TEAM_GDA, ccoGdaThreadIndependent>(
      peer, AsWindow(dst), dstOff, value, ccoGda_NoSignal{}, ccoCoopWarp{},
      OptFlags(aggregate));
}

// Rings the doorbell for this ctx/peer and polls the CQ until the batch posted
// above has completed. Replaces the old flushAsync + wait(request) pair: every
// call site waited on the request immediately, so the split bought nothing.
MEGAMOE_GDA_DEV void megamoe_gda_flush_peer_rail_warp(
    uint64_t dc, int ctx, int peer) {
  Gda gda{*AsDevComm(dc), ctx};
  gda.flush<CCO_TEAM_GDA, ccoCoopWarp>(peer, ccoCoopWarp{});
}


// flushAsync/wait: stage1 posts a batch, rings the doorbell, then goes and does
// other work before waiting on that specific request (it interleaves a credit
// request with the original one). MORI's wrapper exports neither -- only the
// blocking cco_gda_flush{,_peer}__<coop> -- so collapsing these into the
// blocking flush would serialize an overlap stage1 deliberately builds.
// The request is packed into one u64 so it travels as a single FlyDSL scalar.
MEGAMOE_GDA_DEV uint64_t megamoe_gda_flush_async_rail_warp(
    uint64_t dc, int ctx, int peer) {
  Gda gda{*AsDevComm(dc), ctx};
  ccoGdaRequest_t req{};
  gda.flushAsync<CCO_TEAM_GDA, ccoCoopWarp>(peer, &req, ccoCoopWarp{});
  return (static_cast<uint64_t>(static_cast<uint32_t>(req.qpIdx)) << 32) |
         static_cast<uint32_t>(req.postIdx);
}

MEGAMOE_GDA_DEV void megamoe_gda_wait_request_warp(
    uint64_t dc, int ctx, uint64_t packed) {
  Gda gda{*AsDevComm(dc), ctx};
  ccoGdaRequest_t req{};
  req.qpIdx = static_cast<int>(packed >> 32);
  req.postIdx = static_cast<uint32_t>(packed);
  gda.wait(req, ccoCoopWarp{});
}

#undef MEGAMOE_GDA_DEV

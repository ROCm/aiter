// SPDX-License-Identifier: MIT
// AITER-private scalar ABI over MORI's public ccoGda device API.

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

template <ccoTeamMode Team, typename Coop>
__device__ void Put(uint64_t dc, int ctx, int peer, uint64_t dst, uint64_t dstOff, uint64_t src,
                    uint64_t srcOff, uint64_t bytes, int aggregate) {
  Gda gda{*AsDevComm(dc), ctx};
  uint32_t flags =
      aggregate ? ccoGdaOptFlagsAggregateRequests : ccoGdaOptFlagsDefault;
  gda.put<Team, ccoGdaThreadIndependent>(peer, AsWindow(dst), dstOff, AsWindow(src), srcOff,
                                         bytes, ccoGda_NoSignal{}, Coop{}, flags);
}

template <ccoTeamMode Team, typename Coop>
__device__ void Get(uint64_t dc, int ctx, int peer, uint64_t remote, uint64_t remoteOff,
                    uint64_t local, uint64_t localOff, uint64_t bytes, int aggregate) {
  Gda gda{*AsDevComm(dc), ctx};
  uint32_t flags =
      aggregate ? ccoGdaOptFlagsAggregateRequests : ccoGdaOptFlagsDefault;
  gda.get<Team, ccoGdaThreadIndependent>(peer, AsWindow(remote), remoteOff, AsWindow(local),
                                         localOff, bytes, Coop{}, flags);
}

template <ccoTeamMode Team, typename Coop>
__device__ void PutValue(uint64_t dc, int ctx, int peer, uint64_t dst, uint64_t dstOff,
                         uint64_t value, int aggregate) {
  Gda gda{*AsDevComm(dc), ctx};
  uint32_t flags =
      aggregate ? ccoGdaOptFlagsAggregateRequests : ccoGdaOptFlagsDefault;
  gda.putValue<Team, ccoGdaThreadIndependent>(peer, AsWindow(dst), dstOff, value,
                                              ccoGda_NoSignal{}, Coop{}, flags);
}

template <ccoTeamMode Team, typename Coop>
__device__ uint64_t FlushAsync(uint64_t dc, int ctx, int peer) {
  Gda gda{*AsDevComm(dc), ctx};
  ccoGdaRequest_t req{};
  gda.flushAsync<Team>(peer, &req, Coop{});
  return (static_cast<uint64_t>(static_cast<uint32_t>(req.qpIdx)) << 32) |
         static_cast<uint32_t>(req.postIdx);
}

template <typename Coop>
__device__ void WaitRequest(uint64_t dc, int ctx, uint64_t packed) {
  Gda gda{*AsDevComm(dc), ctx};
  ccoGdaRequest_t req{};
  req.qpIdx = static_cast<int>(packed >> 32);
  req.postIdx = static_cast<uint32_t>(packed);
  gda.wait(req, Coop{});
}

template <ccoTeamMode Team, typename Coop>
__device__ void FlushPeer(uint64_t dc, int ctx, int peer) {
  Gda gda{*AsDevComm(dc), ctx};
  gda.flush<Team>(peer, Coop{});
}

// A plain LLVM/global load may be hoisted out of a FlyDSL spin loop.  Use a
// system-scope atomic acquire load so observing the monotonically increasing
// generation also publishes the preceding same-QP RDMA writes to this GPU.
__device__ uint64_t WaitU64GeSystem(uint64_t address, uint64_t expected) {
  auto* ptr = reinterpret_cast<uint64_t*>(address);
  uint64_t observed = 0;
  do {
    observed = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    asm volatile("" ::: "memory");
  } while (observed < expected);
  return observed;
}
}  // namespace

#define AITER_CCO_DEV \
  extern "C" __device__ __attribute__((always_inline, visibility("default")))

// Public CCO LSA addressing: peer is an intra-node LSA rank and offset is a
// byte offset inside the registered window. The returned VA is directly
// load/store accessible over the node fabric.
AITER_CCO_DEV uint64_t aiter_cco_lsa_ptr(
    uint64_t window, int peer_lsa_rank, uint64_t offset) {
  return reinterpret_cast<uint64_t>(
      ccoGetLsaPeerPtr(AsWindow(window), peer_lsa_rank, offset));
}

#define AITER_CCO_TEAM_OPS(TEAM_TAG, TEAM, SCOPE_TAG, COOP)                                \
  AITER_CCO_DEV void aiter_cco_put_##TEAM_TAG##_##SCOPE_TAG(                               \
      uint64_t dc, int ctx, int peer, uint64_t dst, uint64_t dstOff, uint64_t src,          \
      uint64_t srcOff, uint64_t bytes, int aggregate) {                                     \
    Put<TEAM, COOP>(dc, ctx, peer, dst, dstOff, src, srcOff, bytes, aggregate);              \
  }                                                                                         \
  AITER_CCO_DEV void aiter_cco_get_##TEAM_TAG##_##SCOPE_TAG(                               \
      uint64_t dc, int ctx, int peer, uint64_t remote, uint64_t remoteOff, uint64_t local,  \
      uint64_t localOff, uint64_t bytes, int aggregate) {                                   \
    Get<TEAM, COOP>(dc, ctx, peer, remote, remoteOff, local, localOff, bytes, aggregate);    \
  }                                                                                         \
  AITER_CCO_DEV void aiter_cco_put_value_##TEAM_TAG##_##SCOPE_TAG(                         \
      uint64_t dc, int ctx, int peer, uint64_t dst, uint64_t dstOff, uint64_t value,         \
      int aggregate) {                                                                       \
    PutValue<TEAM, COOP>(dc, ctx, peer, dst, dstOff, value, aggregate);                      \
  }                                                                                         \
  AITER_CCO_DEV uint64_t aiter_cco_flush_async_##TEAM_TAG##_##SCOPE_TAG(                   \
      uint64_t dc, int ctx, int peer) {                                                      \
    return FlushAsync<TEAM, COOP>(dc, ctx, peer);                                            \
  }                                                                                         \
  AITER_CCO_DEV void aiter_cco_flush_peer_##TEAM_TAG##_##SCOPE_TAG(                        \
      uint64_t dc, int ctx, int peer) {                                                      \
    FlushPeer<TEAM, COOP>(dc, ctx, peer);                                                    \
  }

// WORLD takes a world rank.  RAIL is the GDA-team view and therefore takes a
// node index when the DevComm was created with CCO_GDA_CONNECTION_RAIL.
AITER_CCO_TEAM_OPS(world, CCO_TEAM_WORLD, warp, ccoCoopWarp)
AITER_CCO_TEAM_OPS(world, CCO_TEAM_WORLD, block, ccoCoopBlock)
AITER_CCO_TEAM_OPS(rail, CCO_TEAM_GDA, warp, ccoCoopWarp)
AITER_CCO_TEAM_OPS(rail, CCO_TEAM_GDA, block, ccoCoopBlock)

#define AITER_CCO_WAIT_OP(SCOPE_TAG, COOP)                                                   \
  AITER_CCO_DEV void aiter_cco_wait_request_##SCOPE_TAG(                                    \
      uint64_t dc, int ctx, uint64_t req) {                                                  \
    WaitRequest<COOP>(dc, ctx, req);                                                         \
  }

AITER_CCO_WAIT_OP(warp, ccoCoopWarp)
AITER_CCO_WAIT_OP(block, ccoCoopBlock)

AITER_CCO_DEV uint64_t aiter_cco_wait_u64_ge_system(uint64_t address, uint64_t expected) {
  return WaitU64GeSystem(address, expected);
}

#undef AITER_CCO_WAIT_OP
#undef AITER_CCO_TEAM_OPS
#undef AITER_CCO_DEV

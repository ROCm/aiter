// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Raw peer-write/peer-read bandwidth primitive for the MI350P all-reduce
// investigation (op_tests/dump_data/docs/mi350p_flydsl_regression_2026-09-03.md,
// section 11 step 2).
//
// WHY THIS EXISTS
// ---------------
// The 2026-08-31 fix that made the FlyDSL qr_int4 kernels fast on PCIe was built
// on a microbenchmark measurement, not on a kernel measurement:
//
//     peers            1        2        3
//     uncached      55.4     4.45     1.44  GB/s
//     finegrained   52.3    53.64    33.45  GB/s
//     coarse        52.4    52.96    32.65  GB/s
//
// and, at 3 peers, a granularity sweep:
//
//     per-peer run    64 B    256 B   1 KiB   4 KiB   64 KiB
//     cached         36.25    54.03   54.37   55.00   54.88  GB/s
//     uncached        1.58     2.27    2.29    2.42    2.41  GB/s
//
// The original binary was standalone and was never committed. Section 10 of the
// regression report shows the *kernel-level* version of both gaps has collapsed
// to 0.90x and 1.02x respectively. This file re-measures the primitive itself,
// so we can say whether the fabric still rewards allocation type and write
// granularity at all, independently of anything the FlyDSL kernels do.
//
// WHAT IT REPRODUCES
// ------------------
// `_fanout_nt` in aiter/ops/flydsl/kernels/qr_int4_kernel.py issues one
// `global_store_dwordx4` (16 B) per lane; four lanes form a 64 B sector; whole
// sectors are handed to peers in round-robin. `--chunk` is the contiguous run
// handed to one destination before switching, so `--chunk 64` is the kernel's
// "sector" fanout order and `--chunk 512` is its "peer" order (8 sectors of one
// 1152 B rank-tile). The stores go out through inline asm with the same cache
// policy string the kernel emits, so `--policy nt` is byte-for-byte the
// instruction the kernel runs.
//
// WHAT IT ADDS OVER THE 08-31 MEASUREMENT
// ---------------------------------------
//   --mode read      pull instead of push. `cdr` is pull-based and never
//                    regressed (report section 9), so read-vs-write separates
//                    "the PCIe path is degraded" from "the write path is".
//   --writers all    every GPU fans out at once, which is what the two-shot
//                    kernel actually does. A single writer cannot reproduce an
//                    incast/credit-exhaustion effect; the report's EA
//                    write-credit stalls were measured with four ranks live.
//   --ipc            exchange the destination buffers through hipIpcGetMemHandle
//                    across forked processes, exactly as QRInt4 does, instead of
//                    same-process hipDeviceEnablePeerAccess. If the two disagree
//                    the mapping attributes differ, which is itself the answer.
//   --dst local      write to the writer's own HBM. Control for "is this ALU or
//                    fabric" -- it should read in the TB/s, not the GB/s.
//   --publish N      drain, buffer_wbl2 and flag every N payload bytes per
//                    block, as _publish does. The kernel's own cadence is 6912 B
//                    at ST=1 and 55296 B at ST=8. Streaming bandwidth prices
//                    none of this.
//   --handshake 1    also spin until every peer's flag for the round has landed,
//                    as _wait_release does. Implies --ipc --writers all, since a
//                    wait is only meaningful when every rank publishes. With it,
//                    this is a complete model of one super-tile of the two-shot
//                    schedule, minus the codec.
//   --mode lat       store-to-retire latency rather than bandwidth. A fabric can
//                    keep its streaming rate and still have got slower here, and
//                    a drained publish pays latency, not bandwidth.
//
// BUILD
// -----
//   hipcc -O3 --offload-arch=gfx950 -o /tmp/peer_write_bw \
//       op_tests/multigpu_tests/peer_write_bw.cpp
//
// RUN
// ---
//   /tmp/peer_write_bw --sweep            # the full table set, CSV on stdout
//   /tmp/peer_write_bw --ipc --sweep      # the same, mapped through HIP IPC
//   /tmp/peer_write_bw --peers 3 --chunk 64 --alloc uncached
//   /tmp/peer_write_bw --handshake 1 --chunk 512 --publish 55296  # ST=8 model
//
// Findings from the 2026-09-07 run of this file, and what they settle, are in
// op_tests/dump_data/docs/mi350p_peer_write_primitive_2026-09-07.md; the raw
// output is op_tests/dump_data/mi350p_peer_write_bw_2026-09-07.csv.
//
// ONE WARNING, FROM GETTING IT WRONG FIRST
// ----------------------------------------
// The first version of this file read ~1 GB/s on fine-grained peer writes --
// indistinguishable from the effect under investigation. The cause was a
// runtime-indexed `dsts[peer]` that clang sank into the store loop as a VMEM
// load; the `s_waitcnt vmcnt(0)` guarding it also drained every peer store in
// flight, so the loop issued one 16 B PCIe write at a time and measured
// round-trip latency. See `pick` below. Any change to the store loop should be
// followed by
//
//   hipcc -O3 --offload-arch=gfx950 -S --cuda-device-only -o - <this file>
//
// and a check that the inner loop still contains no `s_waitcnt`.

#include <hip/hip_runtime.h>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                       \
  do {                                                                        \
    hipError_t _e = (expr);                                                   \
    if (_e != hipSuccess) {                                                   \
      fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,         \
              hipGetErrorString(_e));                                         \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

// hipExtMallocWithFlags modes, mirroring UncachedIpcHeap in
// aiter/ops/flydsl/kernels/qr_int4_ipc.py.
static const unsigned MALLOC_DEFAULT = 0x0;      // coarse-grained, cacheable
static const unsigned MALLOC_FINEGRAINED = 0x1;  // fine-grained
static const unsigned MALLOC_UNCACHED = 0x3;     // uncached

// TransferBench-comparable default: 32 MiB moved per measured iteration, the
// same total the 08-31 table used, split across however many peers are active.
static const size_t DEFAULT_TOTAL_BYTES = 32ull << 20;

// Every buffer is allocated `total + FLAG_TAIL_BYTES` so a --publish run has
// somewhere to put its 64 B-per-block handshake sectors without overwriting
// payload. 4096 blocks x 64 B is more than any geometry here asks for.
// Flag area: one 64 B sector per (writer rank, block). --handshake needs a slot
// per rank; without it only slot 0 is used.
static const unsigned MAX_BLOCKS = 1024;
static const size_t FLAG_TAIL_BYTES = 8ull * MAX_BLOCKS * 64ull;

static const int MAX_PEERS = 8;

struct DstList {
  void* p[MAX_PEERS];
};

// ---------------------------------------------------------------------------
// Store / load with an explicit cache policy
// ---------------------------------------------------------------------------
//
// Inline asm rather than a plain `*(int4*)a = v` because the cache policy is
// the variable under test: the kernel emits `nt` today, section 10 of the report
// tried dropping it, and the coarse-grained inbox mode needs `sc0 sc1`. Letting
// the compiler choose would make the sweep meaningless.

enum Pol { POL_NT = 0, POL_PLAIN = 1, POL_SC = 2 };

// Native 4-wide vector rather than HIP's `int4` struct: only the former lands in
// a VGPR tuple that a "v" asm constraint can name. A struct is passed
// indirectly and clang rejects it outright.
typedef int v4i __attribute__((ext_vector_type(4)));

template <int POL>
__device__ __forceinline__ void st16(void* a, v4i v) {
  if constexpr (POL == POL_NT) {
    asm volatile("global_store_dwordx4 %0, %1, off nt" ::"v"(a), "v"(v)
                 : "memory");
  } else if constexpr (POL == POL_PLAIN) {
    asm volatile("global_store_dwordx4 %0, %1, off" ::"v"(a), "v"(v)
                 : "memory");
  } else {
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(a), "v"(v)
                 : "memory");
  }
}

// Loads deliberately do NOT go through inline asm. A value produced by an asm
// block carries no vmcnt dependency the compiler knows about, so it would have
// to be consumed under an `s_waitcnt vmcnt(0)` written into the asm itself --
// which allows exactly one load in flight per lane and measures the round-trip
// latency of PCIe rather than its bandwidth. Builtins keep the scheduler in
// charge and let the loads pipeline, which is what a bandwidth number needs.
// `sc0 sc1` is unavailable this way; the read table varies allocation type and
// peer count, not the reader's cache policy, so nothing in it needs that arm.
template <int POL>
__device__ __forceinline__ v4i ld16(const v4i* a) {
  if constexpr (POL == POL_NT) return __builtin_nontemporal_load(a);
  return *a;
}

// ---------------------------------------------------------------------------
// The fanout itself
// ---------------------------------------------------------------------------
//
// Index math, in 16 B store units:
//
//   run          = s >> run_shift          which contiguous chunk this store is in
//   peer         = run % npeers            round-robin destinations, as _fanout_nt does
//   off_in_peer  = (run / npeers) << run_shift  +  (s & (run_size - 1))
//
// With run_shift = 2 (64 B) consecutive quads of four lanes target consecutive
// peers, which is the kernel's "sector" order. With run_shift = 5 (512 B) a quad
// walks the sectors of one peer first, which is its "peer" order.

// Destination pointer by index, kept entirely in registers.
//
// This is fiddly for a reason worth recording. `dsts.p[peer]` with a runtime
// `peer` is a dynamically-indexed load out of the kernarg segment. Clang will
// happily sink it into the loop -- even when written as a chain of compares, it
// re-folds the chain into "select the *offset*, then load" -- and because the
// address is lane-varying that load is VMEM, not SMEM. The `s_waitcnt vmcnt(0)`
// guarding it then also waits on every peer store already in flight, so the loop
// issues exactly one 16 B PCIe write at a time and the benchmark measures
// round-trip latency instead of bandwidth. On a first build that read as
// ~1 GB/s and looked exactly like the effect under investigation.
//
// `pin` launders each pointer through an empty asm block, which makes it an
// opaque VGPR-resident value the optimizer cannot rematerialize from kernarg.
// The select is then a cndmask chain over NPEERS values and the store loop is
// waitcnt-free.
__device__ __forceinline__ void* pin(void* p) {
  asm("" : "+v"(p));
  return p;
}

template <int NPEERS>
__device__ __forceinline__ void* pick(const DstList& d, unsigned peer) {
  void* b[NPEERS];
#pragma unroll
  for (int i = 0; i < NPEERS; ++i) b[i] = pin(d.p[i]);
  void* r = b[0];
#pragma unroll
  for (int i = 1; i < NPEERS; ++i)
    if (peer == (unsigned)i) r = b[i];
  return r;
}

// The kernel's release sequence, from `_publish` in qr_int4_kernel.py:
//
//     s_waitcnt vmcnt(0)      this wave's payload stores have retired
//     s_barrier               the other waves' too -- vmcnt is per-wave
//     buffer_wbl2 sc1         push them out of this XCD's L2 (cacheable inbox)
//     s_waitcnt vmcnt(0)      wait for the writeback to land
//     one 64 B flag sector per peer, sc0 sc1 nt
//
// This is the one structural thing the kernel does that a straight bandwidth
// sweep does not, and it is not free: each `s_waitcnt vmcnt(0)` drains every
// peer write still in flight, so the write pipeline empties twice per publish.
// Streaming bandwidth says nothing about what that costs -- the cost is set by
// write *latency* and by how much payload the kernel can put in flight between
// drains. Sweeping `--publish` against `--publish 0` prices it directly.
// The other half is `_wait_release`: after publishing, a block spins until every
// peer's flag for this round has appeared in its *own* inbox, so no block runs
// ahead of the slowest peer. `--handshake` adds it, which makes the primitive a
// complete model of one super-tile of the two-shot schedule minus the codec.
// Splitting publish from wait matters: they fail differently, and the report's
// section 9 counters cannot tell them apart.
struct Handshake {
  void* mine = nullptr;   // this rank's own inbox, where peers drop their flags
  int myrank = 0;         // slot this rank writes into, in every peer's inbox
  int nranks = 1;
  int enable = 0;
};

// Flag slot for (writer rank, block), in 16 B units past the payload.
__device__ __forceinline__ unsigned flag_slot16(unsigned flag_off16, int src,
                                                unsigned blk) {
  return flag_off16 + ((unsigned)src * MAX_BLOCKS + blk) * 4u;
}

__device__ __forceinline__ void st16_flag(void* a, v4i v) {
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1 nt" ::"v"(a), "v"(v)
               : "memory");
}

// `_load_i32_uncached` + `_invalidate_l1` from the kernel: an sc1 load drained
// immediately, then a buffer_inv so the next trip cannot be answered from L1.
__device__ __forceinline__ int ld_flag(const int* a) {
  int v;
  asm volatile("global_load_dword %0, %1, off sc1\n\ts_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(a)
               : "memory");
  return v;
}

__device__ __forceinline__ void inv_l1() {
  asm volatile("buffer_inv sc1" ::: "memory");
}

template <int NPEERS>
__device__ __forceinline__ void publish(const DstList& d, unsigned flag_off16,
                                        int do_wbl2, const Handshake& hs,
                                        int color) {
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (do_wbl2) {
    asm volatile("buffer_wbl2 sc1" ::: "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  }
  // One quad (4 lanes x 16 B = 64 B) per peer, as _publish does.
  const unsigned quad = threadIdx.x / 4, lane = threadIdx.x % 4;
  if (quad < (unsigned)NPEERS) {
    const v4i col = {color, color, color, color};
    st16_flag((v4i*)pick<NPEERS>(d, quad) +
                  flag_slot16(flag_off16, hs.enable ? hs.myrank : 0, blockIdx.x) +
                  lane,
              col);
  }
  if (!hs.enable) return;
  // One lane per peer spins on that peer's slot in our own inbox.
  //
  // The wait is `>= color`, not `== color`. A peer only needs to see *our* flag
  // for round g to move on to g+1, and it can overwrite its own slot with g+1
  // before we have read g -- an equality spin then waits for a colour that will
  // never be there again. Colours are monotone within and across launches, so
  // the signed difference is the safe test.
  //
  // The trip cap turns a protocol mistake into a wrong number instead of a hung
  // GPU. It is far above any real wait; if it is ever hit the run is invalid,
  // which `--handshake` reports by way of an implausibly fast result.
  if ((int)threadIdx.x < hs.nranks && (int)threadIdx.x != hs.myrank) {
    const int* f = (const int*)((v4i*)hs.mine +
                               flag_slot16(flag_off16, threadIdx.x, blockIdx.x));
    for (unsigned t = 0; t < 200000000u && (ld_flag(f) - color) < 0; ++t)
      inv_l1();
  }
  __syncthreads();
}

// NPEERS is a template parameter so the round-robin `% NPEERS` compiles to a
// constant-divisor sequence rather than the runtime reciprocal expansion, which
// would put a dozen VALU ops in front of every store.
//
// Blocks own a contiguous partition of the store index space rather than
// grid-striding it, because a publish is per-block and has to sit between two
// runs of that block's own payload -- which is also how the real kernel is
// structured, one block per super-tile. The set of bytes written is identical
// either way; only the order changes.
// LDS staging for --src lds. The real kernel does not hold its payload in
// registers: `_fanout_nt` reads each 16 B store out of the LDS packet the codec
// wrote, and the disassembly shows every `global_store_dwordx4 ... nt` preceded
// by its own `ds_read_b128` and `s_waitcnt lgkmcnt(0)`. If the compiler does not
// hoist those reads, the store issue rate is bounded by LDS latency rather than
// by the fabric -- and nothing in a register-sourced bandwidth sweep would show
// it. 9216 B is the kernel's own group_segment_fixed_size.
__shared__ v4i g_lds[9216 / 16];

template <int POL, int NPEERS>
__global__ __launch_bounds__(256) void fanout_write(DstList dsts,
                                                    unsigned total16,
                                                    unsigned run_shift,
                                                    unsigned pub16,
                                                    unsigned flag_off16,
                                                    int do_wbl2, Handshake hs,
                                                    int color_base,
                                                    int from_lds) {
  const unsigned run_mask = (1u << run_shift) - 1u;
  const unsigned spb = (total16 + gridDim.x - 1u) / gridDim.x;
  const unsigned begin = blockIdx.x * spb;
  const unsigned end = begin + spb < total16 ? begin + spb : total16;
  const unsigned group = pub16 ? pub16 : spb;
  // Trip count from `spb`, not from this block's own `end`. The last block owns
  // fewer stores when total16 is not a multiple of the grid, and if it published
  // fewer times its peers would spin for a colour that never arrives.
  const unsigned ngroups = (spb + group - 1u) / group;
  for (unsigned g = 0; g < ngroups; ++g) {
    const unsigned g0 = begin + g * group;
    const unsigned g1 = g0 + group < end ? g0 + group : end;
    for (unsigned s = g0 + threadIdx.x; s < g1; s += blockDim.x) {
      const unsigned run = s >> run_shift;
      const unsigned peer = run % (unsigned)NPEERS;
      const unsigned off16 =
          ((run / (unsigned)NPEERS) << run_shift) + (s & run_mask);
      v4i v = {(int)s, (int)(s ^ 0x5a5a5a5au), (int)(s + 7u), (int)~s};
      if (from_lds) {
        // Read-then-store with the LDS read pinned in front of it, as the
        // kernel's codegen has it.
        v = g_lds[s & (9216u / 16u - 1u)];
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      }
      st16<POL>((v4i*)pick<NPEERS>(dsts, peer) + off16, v);
    }
    if (pub16) publish<NPEERS>(dsts, flag_off16, do_wbl2, hs, color_base + (int)g + 1);
  }
}

template <int POL, int NPEERS>
__global__ __launch_bounds__(256) void fanin_read(DstList srcs,
                                                  unsigned total16,
                                                  unsigned run_shift,
                                                  v4i* sink) {
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned run_mask = (1u << run_shift) - 1u;
  v4i acc = {0, 0, 0, 0};
  for (unsigned s = blockIdx.x * blockDim.x + threadIdx.x; s < total16;
       s += stride) {
    const unsigned run = s >> run_shift;
    const unsigned peer = run % (unsigned)NPEERS;
    const unsigned off16 = ((run / (unsigned)NPEERS) << run_shift) + (s & run_mask);
    acc += ld16<POL>((const v4i*)pick<NPEERS>(srcs, peer) + off16);
  }
  // Never taken -- the buffers are zeroed, so the accumulator is zero -- but the
  // compiler cannot prove it, so the loads survive DCE without any store
  // bandwidth being charged to the measurement.
  if (acc[0] == 0x7f3ff7f3 && acc[1] == 0x13579bdf) sink[blockIdx.x] = acc;
}

// Store-to-retire latency: one wave, one 64 B sector per iteration, drained
// every time. This is the quantity the publish sequence actually pays and the
// one the 08-31 measurement never took -- a fabric can keep its streaming
// bandwidth and still have got slower here, and every symptom in section 9 of
// the report (5137-cycle average write latency, IO-credit stalls at a traffic
// level RCCL sustains) is what that would look like. Recording it gives any
// future container/driver comparison a single scalar to diff.
template <int POL, int NPEERS>
__global__ __launch_bounds__(64) void write_latency(DstList dsts,
                                                    unsigned loops) {
  const v4i v = {1, 2, 3, 4};
  for (unsigned i = 0; i < loops; ++i) {
    // Walk destinations and addresses so nothing is answered from a hot line.
    const unsigned peer = i % (unsigned)NPEERS;
    st16<POL>((v4i*)pick<NPEERS>(dsts, peer) + (i & 4095u) * 4u + threadIdx.x,
              v);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  }
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

static unsigned alloc_flags(const std::string& name) {
  if (name == "coarse" || name == "default") return MALLOC_DEFAULT;
  if (name == "fine" || name == "finegrained") return MALLOC_FINEGRAINED;
  if (name == "uncached") return MALLOC_UNCACHED;
  fprintf(stderr, "unknown --alloc %s (coarse|fine|uncached)\n", name.c_str());
  exit(2);
}

static void* alloc_on(int dev, size_t bytes, unsigned flags) {
  HIP_CHECK(hipSetDevice(dev));
  void* p = nullptr;
  // hipExtMallocWithFlags(0x0) and hipMalloc are the same allocation; going
  // through one entry point keeps the three arms comparable.
  HIP_CHECK(hipExtMallocWithFlags(&p, bytes, flags));
  HIP_CHECK(hipMemset(p, 0, bytes));
  return p;
}

// ---------------------------------------------------------------------------
// One measurement
// ---------------------------------------------------------------------------

struct Cfg {
  std::string mode = "write";    // write | read | lat
  std::string alloc = "fine";    // coarse | fine | uncached
  std::string policy = "nt";     // nt | plain | sc
  std::string dst = "peer";      // peer | local
  int npeers = 3;
  int chunk = 64;                // contiguous bytes per destination
  size_t total = DEFAULT_TOTAL_BYTES;
  int from_lds = 0;              // source each store from LDS, as _fanout_nt does
  int handshake = 0;             // also wait on peers' flags, as _wait_release does
  int lat_loops = 4000;          // drained stores per --mode lat kernel
  int publish = 0;               // payload bytes per block between publishes
  int wbl2 = 1;                  // include buffer_wbl2 in the publish
  int blocks = 512;
  int threads = 256;
  int iters = 50;
  int warmup = 10;
  int src_dev = 0;
  std::vector<int> devs;         // participating devices, src first
  bool all_writers = false;
  bool ipc = false;
  bool header = true;
};

static int pol_of(const std::string& s) {
  if (s == "nt") return POL_NT;
  if (s == "plain" || s == "") return POL_PLAIN;
  if (s == "sc" || s == "sc0sc1") return POL_SC;
  fprintf(stderr, "unknown --policy %s (nt|plain|sc)\n", s.c_str());
  exit(2);
}

static unsigned shift_of(int chunk) {
  if (chunk < 16 || (chunk & (chunk - 1)) != 0) {
    fprintf(stderr, "--chunk must be a power of two >= 16 (got %d)\n", chunk);
    exit(2);
  }
  unsigned sh = 0;
  for (int c = chunk / 16; c > 1; c >>= 1) sh++;
  return sh;
}

// Both the cache policy and the peer count are template parameters, so the
// dispatch is a 3 x 8 table. Written out rather than generated so the set of
// instantiations is visible at the call site.
#define PWB_WRITE_CASE(P, N)                                              \
  case N:                                                                 \
    hipLaunchKernelGGL((fanout_write<P, N>), g, b, 0, stream, dsts,       \
                       total16, sh, pub16, flag_off16, c.wbl2, hs,         \
                       color_base, c.from_lds);                           \
    return
#define PWB_LAT_CASE(P, N)                                                \
  case N:                                                                 \
    hipLaunchKernelGGL((write_latency<P, N>), dim3(1), dim3(64), 0,       \
                       stream, dsts, (unsigned)c.lat_loops);              \
    return
#define PWB_READ_CASE(P, N)                                               \
  case N:                                                                 \
    hipLaunchKernelGGL((fanin_read<P, N>), g, b, 0, stream, dsts, total16, \
                       sh, sink);                                         \
    return
#define PWB_BY_PEERS(MAC, P)                                              \
  switch (npeers) {                                                       \
    MAC(P, 1); MAC(P, 2); MAC(P, 3); MAC(P, 4);                           \
    MAC(P, 5); MAC(P, 6); MAC(P, 7); MAC(P, 8);                           \
    default: break;                                                       \
  }

static void launch(const Cfg& c, DstList dsts, unsigned total16, unsigned sh,
                   v4i* sink, hipStream_t stream, Handshake hs = Handshake(),
                   int color_base = 0) {
  const int pol = pol_of(c.policy);
  const int npeers = c.npeers;
  // A publish interval is quoted per block in payload bytes; the kernel counts
  // in 16 B stores. The flag sectors live past the payload, in the tail the
  // allocation reserves for them.
  const unsigned pub16 = (unsigned)(c.publish / 16);
  const unsigned flag_off16 = (unsigned)(c.total / 16);
  dim3 g(c.blocks), b(c.threads);
  if (c.mode == "lat") {
    if (pol == POL_NT) PWB_BY_PEERS(PWB_LAT_CASE, POL_NT)
    else if (pol == POL_PLAIN) PWB_BY_PEERS(PWB_LAT_CASE, POL_PLAIN)
    else PWB_BY_PEERS(PWB_LAT_CASE, POL_SC)
  } else if (c.mode == "write") {
    if (pol == POL_NT) PWB_BY_PEERS(PWB_WRITE_CASE, POL_NT)
    else if (pol == POL_PLAIN) PWB_BY_PEERS(PWB_WRITE_CASE, POL_PLAIN)
    else PWB_BY_PEERS(PWB_WRITE_CASE, POL_SC)
  } else {
    if (pol == POL_NT) PWB_BY_PEERS(PWB_READ_CASE, POL_NT)
    else PWB_BY_PEERS(PWB_READ_CASE, POL_PLAIN)
  }
  fprintf(stderr, "unsupported (mode=%s policy=%s peers=%d)\n", c.mode.c_str(),
          c.policy.c_str(), npeers);
  exit(2);
}

// ---------------------------------------------------------------------------
// Process-shared state, for the --ipc path
// ---------------------------------------------------------------------------
//
// A HIP context cannot be inherited across fork: the child ends up sharing its
// parent's KFD file descriptors and doorbells and the first allocation it makes
// fails with "out of memory". So the fork happens at the very top of main,
// before any HIP entry point is touched, and each process initialises the
// runtime for itself. That is also closer to what QRInt4 does, since aiter's
// ranks are separate processes.

struct Shared {
  volatile int count;
  volatile int sense;
  hipIpcMemHandle_t handles[MAX_PEERS];
  double gbps[MAX_PEERS];
  double us[MAX_PEERS];
};

static Shared* g_shared = nullptr;
static std::vector<pid_t> g_kids;
static int g_rank = 0;
static int g_nproc = 1;
static int g_sense = 0;

static void barrier() {
  if (g_nproc <= 1) return;
  g_sense = !g_sense;
  const int arrived = __sync_add_and_fetch((int*)&g_shared->count, 1);
  if (arrived == g_nproc) {
    g_shared->count = 0;
    g_shared->sense = g_sense;
  } else {
    while (g_shared->sense != g_sense) sched_yield();
  }
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

struct Writer {
  int dev;
  DstList dsts{};
  Handshake hs{};
  v4i* sink = nullptr;
  hipStream_t stream{};
  hipEvent_t e0{}, e1{};
  double us = 0.0;
};

static void writer_init(Writer& w, const Cfg& c) {
  HIP_CHECK(hipSetDevice(w.dev));
  HIP_CHECK(hipMalloc(&w.sink, sizeof(v4i) * (size_t)c.blocks));
  HIP_CHECK(hipStreamCreate(&w.stream));
  HIP_CHECK(hipEventCreate(&w.e0));
  HIP_CHECK(hipEventCreate(&w.e1));
}

static void writer_free(Writer& w) {
  HIP_CHECK(hipSetDevice(w.dev));
  HIP_CHECK(hipEventDestroy(w.e0));
  HIP_CHECK(hipEventDestroy(w.e1));
  HIP_CHECK(hipStreamDestroy(w.stream));
  HIP_CHECK(hipFree(w.sink));
}

// Enqueue warmup, then the timed loop, on every writer. Streams are per-device
// and independent, so the launches overlap; with iters x ~1 ms of work each the
// few microseconds of launch skew across four devices do not matter.
static void time_writers(std::vector<Writer>& ws, const Cfg& c, unsigned total16,
                         unsigned sh) {
  // Every launch uses a fresh colour range so a flag left over from the previous
  // launch cannot satisfy this one's spin. `stride` is an upper bound on the
  // publishes any single launch performs.
  const int stride = c.publish ? (int)(c.total / (size_t)c.blocks /
                                       (size_t)c.publish) + 2 : 1;
  int color = 0;
  for (Writer& w : ws) {
    HIP_CHECK(hipSetDevice(w.dev));
    for (int i = 0; i < c.warmup; ++i)
      launch(c, w.dsts, total16, sh, w.sink, w.stream, w.hs, (color + i) * stride);
  }
  color += c.warmup;
  for (Writer& w : ws) {
    HIP_CHECK(hipSetDevice(w.dev));
    HIP_CHECK(hipStreamSynchronize(w.stream));
  }
  barrier();

  for (Writer& w : ws) {
    HIP_CHECK(hipSetDevice(w.dev));
    HIP_CHECK(hipEventRecord(w.e0, w.stream));
    for (int i = 0; i < c.iters; ++i)
      launch(c, w.dsts, total16, sh, w.sink, w.stream, w.hs,
             (color + i) * stride);
    HIP_CHECK(hipEventRecord(w.e1, w.stream));
  }
  for (Writer& w : ws) {
    HIP_CHECK(hipSetDevice(w.dev));
    HIP_CHECK(hipEventSynchronize(w.e1));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, w.e0, w.e1));
    w.us = (double)ms * 1e3 / c.iters;
  }
  barrier();
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

static void emit_header() {
  printf("mode,map,alloc,policy,dst,writers,peers,chunk_B,publish_B,wbl2,"
         "total_B,blocks,threads,us,gbps_per_writer,gbps_aggregate,"
         "ns_per_drain\n");
}

static void emit_row(const Cfg& c, double worst_us, double per_writer,
                     double agg) {
  // In --mode lat the kernel is one wave doing `lat_loops` drained stores, so
  // bandwidth is meaningless and the useful number is the per-drain time.
  const bool lat = c.mode == "lat";
  const double ns = lat ? worst_us * 1e3 / (double)c.lat_loops : 0.0;
  printf("%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%zu,%d,%d,%.3f,%.3f,%.3f,%.1f\n",
         c.mode.c_str(), c.ipc ? "ipc" : "p2p", c.alloc.c_str(),
         c.policy.c_str(), c.dst.c_str(), c.all_writers ? "all" : "one",
         c.npeers, c.chunk, c.publish, c.publish ? c.wbl2 : 0, c.total,
         lat ? 1 : c.blocks, lat ? 64 : c.threads, worst_us,
         lat ? 0.0 : per_writer, lat ? 0.0 : agg, ns);
  fflush(stdout);
}

// ---------------------------------------------------------------------------
// The two mapping models
// ---------------------------------------------------------------------------

// Same-process peer access. Every buffer is allocated by this process on its own
// device and reached through hipDeviceEnablePeerAccess. Everything is allocated
// and freed per measurement, so a sweep cannot silently reuse the previous
// configuration's allocation flags.
static void measure_p2p(const Cfg& c) {
  const int nranks = (int)c.devs.size();
  const unsigned flags = alloc_flags(c.alloc);
  const unsigned sh = shift_of(c.chunk);
  const unsigned total16 = (unsigned)(c.total / 16);

  std::vector<void*> bufs(nranks, nullptr);
  for (int r = 0; r < nranks; ++r) bufs[r] = alloc_on(c.devs[r], c.total + FLAG_TAIL_BYTES, flags);
  for (int r = 0; r < nranks; ++r) {
    HIP_CHECK(hipSetDevice(c.devs[r]));
    for (int q = 0; q < nranks; ++q) {
      if (q == r) continue;
      hipError_t e = hipDeviceEnablePeerAccess(c.devs[q], 0);
      if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e);
    }
  }

  std::vector<Writer> ws;
  for (int r = 0; r < nranks; ++r) {
    if (!c.all_writers && r != 0) continue;
    Writer w;
    w.dev = c.devs[r];
    for (int i = 0; i < MAX_PEERS; ++i) {
      // Destinations are the ranks after this one, cyclically, so with every
      // rank writing, each drives a different set of links.
      const int peer = (r + 1 + (i % (nranks - 1))) % nranks;
      w.dsts.p[i] = (c.dst == "local") ? bufs[r] : bufs[peer];
    }
    writer_init(w, c);
    ws.push_back(w);
  }

  time_writers(ws, c, total16, sh);

  double agg = 0.0, worst = 0.0;
  for (Writer& w : ws) {
    const double gbps = (double)c.total / (w.us * 1e3);
    agg += gbps;
    if (w.us > worst) worst = w.us;
    writer_free(w);
  }
  emit_row(c, worst, agg / (double)ws.size(), agg);

  for (int r = 0; r < nranks; ++r) {
    HIP_CHECK(hipSetDevice(c.devs[r]));
    HIP_CHECK(hipFree(bufs[r]));
  }
}

// HIP IPC, one process per rank, exactly as QRInt4's UncachedIpcHeap does it.
// Worth measuring separately because the mapping attributes an IPC import gets
// need not match those of a same-process peer mapping, and this whole
// investigation is about page attributes.
static void measure_ipc(const Cfg& c) {
  const int nranks = g_nproc;
  const unsigned flags = alloc_flags(c.alloc);
  const unsigned sh = shift_of(c.chunk);
  const unsigned total16 = (unsigned)(c.total / 16);
  const int r = g_rank;

  void* mine = alloc_on(c.devs[r], c.total + FLAG_TAIL_BYTES, flags);
  HIP_CHECK(hipIpcGetMemHandle(&g_shared->handles[r], mine));
  barrier();

  std::vector<void*> opened;
  Writer w;
  w.dev = c.devs[r];
  for (int i = 0; i < MAX_PEERS; ++i) {
    if (c.dst == "local") {
      w.dsts.p[i] = mine;
      continue;
    }
    const int peer = (r + 1 + (i % (nranks - 1))) % nranks;
    void* p = nullptr;
    HIP_CHECK(hipIpcOpenMemHandle(&p, g_shared->handles[peer],
                                  hipIpcMemLazyEnablePeerAccess));
    w.dsts.p[i] = p;
    opened.push_back(p);
  }
  barrier();

  w.hs.mine = mine;
  w.hs.myrank = r;
  w.hs.nranks = nranks;
  w.hs.enable = c.handshake;
  std::vector<Writer> ws;
  if (c.all_writers || r == 0) {
    writer_init(w, c);
    ws.push_back(w);
  }
  time_writers(ws, c, total16, sh);

  g_shared->us[r] = ws.empty() ? 0.0 : ws[0].us;
  g_shared->gbps[r] = ws.empty() ? 0.0 : (double)c.total / (ws[0].us * 1e3);
  barrier();

  if (r == 0) {
    double agg = 0.0, worst = 0.0;
    int n = 0;
    for (int q = 0; q < nranks; ++q) {
      if (g_shared->gbps[q] <= 0.0) continue;
      agg += g_shared->gbps[q];
      if (g_shared->us[q] > worst) worst = g_shared->us[q];
      n++;
    }
    emit_row(c, worst, n ? agg / n : 0.0, agg);
  }

  for (Writer& x : ws) writer_free(x);
  HIP_CHECK(hipSetDevice(c.devs[r]));
  for (void* p : opened) HIP_CHECK(hipIpcCloseMemHandle(p));
  barrier();
  HIP_CHECK(hipFree(mine));
  barrier();
}

static void measure(const Cfg& c) {
  if (c.ipc) measure_ipc(c);
  else measure_p2p(c);
}

// ---------------------------------------------------------------------------
// Sweeps
// ---------------------------------------------------------------------------

static const char* const ALLOCS[] = {"coarse", "fine", "uncached"};

static void sweep(Cfg base) {
  const int nranks = (int)base.devs.size();
  const int max_peers = nranks - 1;
  const int chunks[] = {64, 128, 256, 512, 1024, 4096, 65536};

  // Table A -- the 08-31 allocation table: 64 B destination interleave, one
  // writer, 1..3 peers. Row for row, the measurement the fine-grained fix was
  // justified by.
  for (const char* a : ALLOCS) {
    for (int p = 1; p <= max_peers && p <= 3; ++p) {
      Cfg c = base; c.alloc = a; c.npeers = p; c.chunk = 64;
      measure(c);
    }
  }

  // Table B -- the 08-31 granularity table: 3 peers, per-destination run length
  // from 64 B to 64 KiB. "Does the fabric still reward contiguity."
  for (const char* a : ALLOCS) {
    for (int ch : chunks) {
      Cfg c = base; c.alloc = a; c.npeers = (max_peers < 3 ? max_peers : 3);
      c.chunk = ch;
      measure(c);
    }
  }

  // Table C -- pull instead of push, same axes as A. `cdr` reads from peers and
  // did not regress, so this separates "the PCIe path is degraded" from "the
  // write path is".
  for (const char* a : ALLOCS) {
    for (int p = 1; p <= max_peers && p <= 3; ++p) {
      Cfg c = base; c.mode = "read"; c.alloc = a; c.npeers = p; c.chunk = 64;
      measure(c);
    }
  }

  // Table D -- every rank fanning out at once, which is what the two-shot kernel
  // does and the traffic under which the report's EA write-credit stalls were
  // counted. A single writer cannot show an incast effect.
  for (const char* a : ALLOCS) {
    for (int ch : {64, 512, 4096}) {
      Cfg c = base; c.alloc = a; c.all_writers = true; c.npeers = max_peers;
      c.chunk = ch;
      measure(c);
    }
  }

  // Table F -- store-to-retire latency, the cost a publish actually pays.
  for (const char* a : ALLOCS) {
    for (int p = 1; p <= max_peers && p <= 3; ++p) {
      Cfg c = base; c.mode = "lat"; c.alloc = a; c.npeers = p; c.chunk = 64;
      measure(c);
    }
  }
  {
    Cfg c = base; c.mode = "lat"; c.alloc = "fine"; c.dst = "local";
    c.npeers = 1; c.chunk = 64;
    measure(c);
  }

  // Table E -- local HBM control. If this is not in the TB/s then the index
  // math or the launch geometry is the bottleneck and nothing above means
  // anything.
  for (const char* a : ALLOCS) {
    Cfg c = base; c.alloc = a; c.dst = "local"; c.npeers = 1; c.chunk = 64;
    measure(c);
  }
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Cfg c;
  bool do_sweep = false;
  std::vector<int> devs;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s needs a value\n", a.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--sweep") do_sweep = true;
    else if (a == "--mode") c.mode = next();
    else if (a == "--alloc") c.alloc = next();
    else if (a == "--policy") c.policy = next();
    else if (a == "--dst") c.dst = next();
    else if (a == "--peers") c.npeers = atoi(next().c_str());
    else if (a == "--chunk") c.chunk = atoi(next().c_str());
    else if (a == "--total") c.total = (size_t)atoll(next().c_str());
    else if (a == "--src") c.from_lds = (next() == "lds");
    else if (a == "--handshake") c.handshake = atoi(next().c_str());
    else if (a == "--lat-loops") c.lat_loops = atoi(next().c_str());
    else if (a == "--publish") c.publish = atoi(next().c_str());
    else if (a == "--wbl2") c.wbl2 = atoi(next().c_str());
    else if (a == "--blocks") c.blocks = atoi(next().c_str());
    else if (a == "--threads") c.threads = atoi(next().c_str());
    else if (a == "--iters") c.iters = atoi(next().c_str());
    else if (a == "--warmup") c.warmup = atoi(next().c_str());
    else if (a == "--writers") c.all_writers = (next() == "all");
    else if (a == "--ipc") c.ipc = true;
    else if (a == "--no-header") c.header = false;
    else if (a == "--devs") {
      std::string s = next();
      size_t pos = 0;
      while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        devs.push_back(atoi(s.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
      }
    } else {
      fprintf(stderr, "unknown argument %s\n", a.c_str());
      exit(2);
    }
  }

  // Device discovery without HIP: hipGetDeviceCount would initialise the
  // runtime, and the --ipc fork below has to happen before that.
  if (devs.empty()) {
    const char* vis = getenv("HIP_VISIBLE_DEVICES");
    int n = 4;
    if (vis) {
      n = 1;
      for (const char* p = vis; *p; ++p)
        if (*p == ',') n++;
    }
    for (int i = 0; i < n && i < 4; ++i) devs.push_back(i);
  }
  if ((int)devs.size() > MAX_PEERS) devs.resize(MAX_PEERS);
  c.devs = devs;

  if (c.handshake) {
    // A wait is only meaningful when every rank publishes, and the flags have
    // to be in IPC-shared inboxes.
    c.ipc = true;
    c.all_writers = true;
  }
  if (c.blocks > (int)MAX_BLOCKS) {
    fprintf(stderr, "--blocks capped at %u by the flag area\n", MAX_BLOCKS);
    return 2;
  }
  if (c.ipc) {
    g_nproc = (int)devs.size();
    g_shared = (Shared*)mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (g_shared == MAP_FAILED) {
      perror("mmap");
      return 1;
    }
    memset(g_shared, 0, sizeof(Shared));
    for (int r = 1; r < g_nproc; ++r) {
      const pid_t pid = fork();
      if (pid == 0) {
        g_rank = r;
        g_kids.clear();
        break;
      }
      g_kids.push_back(pid);
    }
    // Every process, the parent included, falls through and initialises HIP for
    // itself below. Children are reaped after the sweep.
  }

  if (c.npeers > (int)devs.size() - 1 && c.dst != "local")
    c.npeers = (int)devs.size() - 1;

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  for (int d : devs) {
    if (d >= ndev) {
      fprintf(stderr, "device %d not visible (%d present)\n", d, ndev);
      return 1;
    }
  }

  if (g_rank == 0) {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, devs[0]));
    fprintf(stderr, "# %s  %d CU  arch=%s  devs=", prop.name,
            prop.multiProcessorCount, prop.gcnArchName);
    for (size_t i = 0; i < devs.size(); ++i)
      fprintf(stderr, "%s%d", i ? "," : "", devs[i]);
    fprintf(stderr, "  map=%s\n", c.ipc ? "ipc" : "p2p");
    if (c.header) emit_header();
  }

  if (do_sweep) sweep(c);
  else measure(c);

  for (pid_t p : g_kids) {
    int st = 0;
    waitpid(p, &st, 0);
  }
  return 0;
}

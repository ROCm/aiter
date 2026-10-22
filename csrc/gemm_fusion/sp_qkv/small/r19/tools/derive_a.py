import os
import re
import sys

d = sys.argv[1]


def rd(p):
    with open(p, encoding="utf-8", newline="") as _f:
        t = _f.read()
    nl = "\r\n" if "\r\n" in t else "\n"
    return t.replace("\r\n", "\n"), nl


def wr(p, t, nl):
    with open(p, "w", encoding="utf-8", newline="") as _f:
        _f.write(t.replace("\n", nl))


def rep(t, old, new, cnt=1):
    assert t.count(old) == cnt, (f"count {t.count(old):d} != {cnt:d}: ") + old[:110]
    return t.replace(old, new)


# ===================== ABI: v2 -> v3 =====================
t, nl = rd(d + "/interface/sm_abi_v2.h")
t = rep(
    t,
    "#define SM_ABI_VERSION 2u",
    "#define SM_ABI_VERSION 3u     // r19 (R12 card r2 D2.3): kernarg v3 (SmRanArgsV3, 64 B), per-comm sticky ERR/STEP, per-launch tags; SM_C_POISON retired",
)
t = rep(
    t,
    "#define SM_C_POISON  20u            // u32 (sticky)\n",
    "// SM_C_POISON (offset 20) is RETIRED in r19: no reads or writes anywhere; the sticky error is the per-comm ERR word (SmRanArgsV3::err).\n",
)
t = rep(
    t,
    "#define SM_C_BYTES   (SM_C_FLAGS + 8u*2u*4u*SM_NA)",
    """#define SM_C_TAG(par,src) (SM_C_FLAGS + 64u + 8u*((par)*4u + (src)))   // r19: u64 tag[par][src] in the DESTINATION ctl window (bytes 320-383, inside SM_C_BYTES, unused by r18: T1 uses SM_T1_FLAG(src) = bytes 256-287 only)
#define SM_C_BYTES   (SM_C_FLAGS + 8u*2u*4u*SM_NA)
// ---- r19 per-comm ERR word (layout shared with the large path, lg_comm_abi_v3.h): reason[7:0] | origin_rank[9:8] | kind[11:10] (1 small, 2 large) | propagated[12]; the first CAS 0 -> code wins. Reasons stay < 0x100 (bit 8 is the origin rank LSB).
#define SM_ERR_REASON_MASK 0xFFu
#define SM_ERR_ORIGIN_SHIFT 8u
#define SM_ERR_KIND_SHIFT 10u
#define SM_ERR_KIND_SMALL 1u
#define SM_ERR_PROPAGATED (1u << 12)
#define SM_ERR_CODE(reason, rank) ((static_cast<uint32_t>(reason) & SM_ERR_REASON_MASK) | (static_cast<uint32_t>(rank) << SM_ERR_ORIGIN_SHIFT) | (SM_ERR_KIND_SMALL << SM_ERR_KIND_SHIFT))
#define SM_ERRR_FLAG_TIMEOUT   0x20u    // B-WG flag poll exhausted its budget
#define SM_ERRR_FLAG_OVERSHOOT 0x21u    // flag > K*(e+1) (r18: 0x20|0x100, which collided with origin_rank bit 8)
#define SM_ERRR_TAG_MISMATCH   0x30u    // producer tag != E = (step << 32) | call_id
#define SM_ERRR_BAD_LAUNCH     0x40u    // M / T geometry or shape-build refusal
#define SM_ERRR_BAD_GRID       0x41u    // RAN grid != SM_T1_GRID (WG0, end of launch)
#define SM_ERRR_NO_GEMM        0x42u    // epoch word 0: the GEMM did not run / wrong pairing""",
)
t = rep(
    t,
    "struct SmDev {",
    """// r19: peer ERR words (device-resident, filled by the host from gpage_local / ccoGetPeerPtr); err[own rank] is ignored; the SmDev pointer may be null (no propagation).
struct SmGfPeers { uint32_t* err[4]; };
struct SmDev {""",
)
t = rep(t, "int64_t kc_sh_unused;", "const SmGfPeers* gpeers;")
i = t.index("struct SmRanArgs {")
j = t.index('static_assert(sizeof(SmRanArgs) == 16, "kernarg = 16 B");') + len(
    'static_assert(sizeof(SmRanArgs) == 16, "kernarg = 16 B");'
)
t = (
    t[:i]
    + """// RAN kernarg v3: one by-value 64 B block (one s_load_dwordx16). Ascale / positions / slot_mapping are PER-LAUNCH inputs (r19): SmDev::Ascale / positions / slot_mapping / index_slot_mapping are written as nullptr by the host and NEVER read by the v3 kernel.
// err / step point into the per-comm global page (u32 ERR at +0, u64 STEP at +8; DIAG u64[8] at +64). Rows t >= t_valid are padded in-kernel (position 0, slot -1: no cache writes).
struct SmRanArgsV3 {
    const SmDev* blk; const float* Ascale; const int64_t* positions; const int64_t* slot_mapping; const uint32_t* err; const uint64_t* step;
    uint32_t M; uint32_t rank; uint32_t t_valid; uint32_t call_id;
};
static_assert(sizeof(SmRanArgsV3) == 64 && alignof(SmRanArgsV3) == 8, "kernarg v3 = 64 B");"""
    + t[j:]
)
wr(d + "/interface/sm_abi_v3.h", t, nl)
os.remove(d + "/interface/sm_abi_v2.h")
# includes
for rel in (
    "device/ran_kernel.cu",
    "device/sk_gemm_fp8.cu",
    "device/stamps.hpp",
    "device/layout/ran_layouts.hpp",
    "device/layout/sk_traits.hpp",
    "device/layout/t1_roles.hpp",
    "host/driver_sm.cpp",
):
    p = d + "/" + rel
    t, nl = rd(p)
    if "sm_abi_v2.h" in t:
        t = t.replace("sm_abi_v2.h", "sm_abi_v3.h")
        wr(p, t, nl)
# ===================== RAN kernel =====================
p = d + "/device/ran_kernel.cu"
t, nl = rd(p)
# flags: NO_CREDIT is required, new seams
t = rep(
    t,
    "#ifndef SM_NO_CREDIT\n#define SM_NO_CREDIT 1\n#endif",
    """#ifndef SM_NO_CREDIT
#define SM_NO_CREDIT 1
#endif
#if !SM_NO_CREDIT
#error "r19 (R12) is built on the credit-free protocol: SM_NO_CREDIT must be 1 (the credit slow path used the retired SM_C_POISON word)"
#endif
// r19 seams (default 0, test-only, never candidates): SM_TEST_DELAY_RELEASE=<n>: the releasing thread sleeps n x s_sleep(127) BEFORE its tag store + release (a delayed source, G10a). SM_TEST_TAG_AFTER_RELEASE=<n>: the releasing thread does the flag RMW FIRST, sleeps
// n x s_sleep(127), and stores the tag AFTER it (violates D-2): consumers can then observe the flag with a stale tag => 0x30 mismatches MUST appear (G10b).
#ifndef SM_TEST_DELAY_RELEASE
#define SM_TEST_DELAY_RELEASE 0
#endif
#ifndef SM_TEST_TAG_AFTER_RELEASE
#define SM_TEST_TAG_AFTER_RELEASE 0
#endif""",
)
# poison() + poll64 + poll_range64 -> set_err + new poll_range64
i = t.index("SM_INL void poison(const SmDev& a, uint32_t why) {")
j = t.index("SM_INL uint16_t bf16_bits(float f)")
new = """// r19 set_err (cold path): CAS(ERR, 0 -> code) at system scope; the winner writes DIAG (E, seen tag, flag value, plan code) and propagates code|PROPAGATED with a relaxed system CAS on the 3 peers' ERR words (when SmDev::gpeers is set).
SM_INL void set_err(const SmDev& a, const uint32_t* err, uint32_t rank, uint32_t reason, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint32_t* const e = const_cast<uint32_t*>(err);
    const uint32_t code = SM_ERR_CODE(reason, rank);
    uint32_t z = 0u;
    if (!__hip_atomic_compare_exchange_strong(e, &z, code, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) return;       // not the first: nothing more to do
    uint64_t* const dg = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(e) + 64);
    __hip_atomic_store(dg + 0, d0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(dg + 1, d1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(dg + 2, d2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(dg + 3, static_cast<uint64_t>(SM_T1_GRID), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);        // plan code word: the grid size of this build (diagnostic only)
    if (a.gpeers) {
        for (uint32_t r = 0; r < 4u; ++r) {
            if (r == rank) continue;
            uint32_t* const pe = a.gpeers->err[r];
            if (!pe) continue;
            uint32_t pz = 0u;
            __hip_atomic_compare_exchange_strong(pe, &pz, code | SM_ERR_PROPAGATED, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}
// Bounded poll, thread 0 only: relaxed system spin; ONE system acquire fence on success (caller: barrier, then reads). r19: every iteration also reads the per-comm ERR word (a set ERR returns false WITHOUT a new set_err: the first CAS already won);
// overshoot (> hi) and timeout call set_err with reasons < 0x100 (0x21 / 0x20); the diagnostic words carry E, 0 and the flag value.
SM_INL bool poll_range64(const SmDev& a, const uint32_t* err, uint32_t rank, uint64_t E, const void* p, uint64_t lo, uint64_t hi, uint64_t* iters) {
    for (uint64_t i = 0; i < a.poll_iterations; ++i) {
        const uint64_t v = ld_rlx64(p);
        if (v > hi) { set_err(a, err, rank, SM_ERRR_FLAG_OVERSHOOT, E, 0ull, v); return false; }
        if (v >= lo) {
#if SM_RX_BYPASS
            asm volatile("" ::: "memory");                  // r18 R2: NO buffer_inv: the Rx data loads after this poll are system-coherent (sc0 sc1) loads; only the compiler may not hoist them above the poll
#elif !SM_TEST_NO_READER_ACQUIRE
            acquire_fence_system();
#endif
            if (iters) *iters = i;
            return true;
        }
        if (__hip_atomic_load(err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0u) { if (iters) *iters = i; return false; }      // PEER_ERR
        __builtin_amdgcn_s_sleep(2);
    }
    set_err(a, err, rank, SM_ERRR_FLAG_TIMEOUT, E, 0ull, ld_rlx64(p));
    return false;
}
"""
t = t[:i] + new + t[j:]
# pre_a with the per-launch pointers and the pad mask
i = t.index("SM_INL void pre_a(const SmDev& a, uint32_t t, RowPre& p)")
j = t.index("SM_INL void pre_b(", i)
t = (
    t[:i]
    + """SM_INL void pre_a(const SmDev& a, const int64_t* pos, const int64_t* slot, uint32_t tv, uint32_t t, RowPre& p) {   // r19: per-launch inputs; rows t >= t_valid are padded IN-KERNEL (pos 0, slot -1) and their loads are not issued
    const bool v = t < tv;
    p.pos = v ? pos[t] : 0; p.slot = v ? slot[t] : -1;
#if SM_R5X_INDEXER
    p.islot = v ? slot[t] : -1;                      // the index slot is slot_mapping (golden passes slot_mapping)
#else
    p.islot = -1;                                    // dense/reuse: no index cache (r10: no load at all)
#endif
}   // round 1
"""
    + t[j:]
)
# head_row fallbacks (pre is never null at any call site): constants
t = rep(
    t,
    "const int64_t slot = pre ? pre->slot : a.slot_mapping[t];",
    "const int64_t slot = pre ? pre->slot : -1;",
    2,
)
t = rep(
    t,
    "const int64_t pos = pre ? pre->pos : (live ? a.positions[t] : 0);",
    "const int64_t pos = pre ? pre->pos : 0;",
)
t = rep(
    t,
    "const int64_t is = pre ? pre->islot : (a.index_slot_mapping ? a.index_slot_mapping[t] : -1);",
    "const int64_t is = pre ? pre->islot : -1;",
)
# decide comment
t = rep(
    t,
    "if (pz != 0u) return 0xFFFFFFFFu;",
    "if (pz != 0u) return 0xFFFFFFFFu;                                                // ERR already set: NANMODE (r19: outputs become NaN; the r18 silent skip is gone)",
)
# kernel header -> device body + wrapper
t = rep(
    t,
    """extern "C" __global__ __launch_bounds__(SM_RAN_THREADS)
void sm_ran_v1(const SmDev* __restrict__ blk, uint32_t M, uint32_t rank) {
#if defined(__gfx950__)
    using namespace sm;
    const SmDev& a = *blk;""",
    """// r19: the former kernel body; `blk` stays a restrict const pointer so the SmDev loads remain invariant scalar loads after inlining into sm_ran_v3.
__device__ __forceinline__ void ran_body(const SmDev* __restrict__ blk, const SmRanArgsV3& ka) {
#if defined(__gfx950__)
    using namespace sm;
    const uint32_t M = ka.M, rank = ka.rank;
    const SmDev& a = *blk;""",
)
t = rep(
    t,
    """#ifndef __HIP_DEVICE_COMPILE__
extern "C" __global__ void sm_ran_v1(const SmDev*, uint32_t, uint32_t) {}
#endif""",
    """#ifdef __HIP_DEVICE_COMPILE__
extern "C" __global__ __launch_bounds__(SM_RAN_THREADS)
void sm_ran_v3(SmRanArgsV3 ka) { ran_body(ka.blk, ka); }
#else
extern "C" __global__ void sm_ran_v3(SmRanArgsV3) {}
#endif""",
)
# per-launch inputs in the body
t = rep(
    t, "as_[pass] = a.Ascale[rc];", "as_[pass] = ka.Ascale[rc];", 2
)  # IK owner + self
t = rep(t, "as_v = a.Ascale[rcl];", "as_v = ka.Ascale[rcl];")
t = rep(
    t,
    "as_[pass] = a.Ascale[rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1)];",
    "as_[pass] = ka.Ascale[rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1)];",
)
n = len(re.findall(r"pre_a\(a, (?!ka\.)", t))
print("pre_a call sites", n)
t = re.sub(
    r"pre_a\(a, (?!const)(?=[a-z])",
    "pre_a(a, ka.positions, ka.slot_mapping, ka.t_valid, ",
    t,
)
# entry round: ERR + STEP instead of the poison word
t = rep(
    t,
    "const uint32_t pz = ld_ep32(cwin_local(a, SM_C_POISON));",
    "const uint32_t pz = ld_ep32(ka.err);                                         // r19: per-comm ERR (nonzero => NANMODE)\n        const uint64_t E = (ld_ep64(ka.step) << 32) | static_cast<uint64_t>(ka.call_id);   // expected tag, same single entry round trip",
    5,
)
# releasing thread stores the tag first (D-2)
n2 = len(
    re.findall(
        r"sys_add_release\(cwin_peer\(a, static_cast<int>\((.*?)\), SM_T1_FLAG\(rank\)\), 1ull\)",
        t,
    )
)
print("release sites", n2)
t = re.sub(
    r"sys_add_release\(cwin_peer\(a, static_cast<int>\((.*?)\), SM_T1_FLAG\(rank\)\), 1ull\)",
    r"release_tagged(a, static_cast<int>(\1), rank, par, E)",
    t,
)
t = rep(
    t,
    "// Entry reads (U2):",
    """// r19 D-2: the SAME thread that executes fence(release) -> vmcnt(0) -> RMW first stores the destination's tag (relaxed system store, sc0 sc1 flavour like push16), in program order before its own fence, so the release sequence publishes it.
SM_INL void release_tagged(const SmDev& a, int d, uint32_t rank, uint32_t par, uint64_t E) {
#if SM_TEST_DELAY_RELEASE
    for (int dly = 0; dly < SM_TEST_DELAY_RELEASE; ++dly) __builtin_amdgcn_s_sleep(127);
#endif
#if SM_TEST_TAG_AFTER_RELEASE
    sys_add_release(cwin_peer(a, d, SM_T1_FLAG(rank)), 1ull);                                   // G10b NEGATIVE CONTROL: tag AFTER the release (illegal)
    for (int dly = 0; dly < SM_TEST_TAG_AFTER_RELEASE; ++dly) __builtin_amdgcn_s_sleep(127);
    __hip_atomic_store(reinterpret_cast<uint64_t*>(cwin_peer(a, d, SM_C_TAG(par, rank))), E, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    __hip_atomic_store(reinterpret_cast<uint64_t*>(cwin_peer(a, d, SM_C_TAG(par, rank))), E, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    sys_add_release(cwin_peer(a, d, SM_T1_FLAG(rank)), 1ull);
#endif
}
// Entry reads (U2):""",
)
wr(p, t, nl)
print("ran stage 1 ok")

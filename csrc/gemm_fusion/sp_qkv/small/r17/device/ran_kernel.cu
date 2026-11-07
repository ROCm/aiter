// SPDX-License-Identifier: MIT
// SM-T1-r7x = r7 (K1) + r5x indexer/reuse roles + numerics-fidelity switches SM_ZSIGN / SM_RMS_ORDER / SM_ROPE_FMA (see below). Based on r7 (D7 K1): kernarg = { const SmDev* blk, M, rank } only; every other field is read from the device-resident SmDev block (role-first 64 B groups). Derived from r6 (D6 V1/V2): every push/self/B WG does ONE memory round trip with unconditional clamped loads and a single vmcnt(0); poison CAS deferred; see note. Derived from r5 (derived from r4: U1 24 push WGs, U2 host window pointers + agent-scope entry loads + late poison check, U3 B heads flag) RAN kernel (Track 1, dense, M<=32, SK=2). UNCOMPILED / UNTESTED. Numerics identical to r3: ascending-split FP32 reduce ->
// (acc*Ws)*As -> ONE BF16 RNE -> push (16 B SC0|SC1 stores) / fused self -> Gemma RMS(1+w), NeoX partial RoPE64, Q/K BF16 (16 B stores, P1),
// K cache FP32 post-RoPE amax, V raw BF16 + FP8 cache.
// Static roles, grid SM_T1_GRID = 24 push + 18 self + 3*B_PER_PAIR B (69 with B_HEADS=2, 96 with B_HEADS=1), no tickets, no DONE/SRCDONE/tail counter. r5 role map:
//   WG 0-23 PUSH (3 per XCC), each 2..3 units of ONE destination (K=8 pushers per (src,dst), t1r static_asserts); WG 24-41 SELF; WG 42.. B (NH heads each).
//   (the r4 text below describes the 8-WG version; the role bodies in this file are the r5 ones)
//   WG 0-7   PUSH (one per XCC, XCC = wgid%8 measured): units [54*w/8, 54*(w+1)/8) of the 54 remote units (3 dest x 18 heads); loads software-
//            pipelined one unit ahead, all stores, per-wave vmcnt(0), barrier, then thread 0 does ONE system-scope RELEASE fetch_add(1) per touched
//            destination on flag[par][src] (release-sequence counter; K_sd = number of pushers touching (src,dst), static, identical both sides).
//   WG 8-25  SELF (one head each, no sync).
//   WG 26-52 B (27): 2 heads of one remote source; positions/slots/cos/sin/norm weights prefetched BEFORE the single acquire (relaxed system
//            spin on flag[par][src] == K_sd*((e+1)>>1), one system acquire fence, barrier, reads); poison if the count overshoots.
//   Credit (Q6): each B WG does vmcnt(0)+barrier then fetch_add(1, RELEASE, system) on credit[me] in the SOURCE window (9 per source per epoch);
//   the pusher requires credit[d] >= 9*(e-2) for e>2 (e=1,2 skipped explicitly, no underflow).
//   Epoch (Q7): relaxed AGENT store of e by the B WGs (deviation: not every WG, see note). Host zeroes the control window once and after poison.
// Review conditions kept: system-scope acquire after the relaxed spin, per-wave vmcnt(0) before every release barrier, release scope system.
// Flags SM_P0 (push16 16B SC0|SC1) / SM_P1 (16 B output stores) default 1/1 (r3 variant B); r3 R1-R4/P2/P3 flags are not used by this kernel.#include "sm_abi_v2.h"
#include "stamps.hpp"
#include "layout/t1_roles.hpp"
#include <opus/hip_minimal.hpp>

#ifndef SM_R1_RELAXED_SPIN
#define SM_R1_RELAXED_SPIN 1
#endif
#ifndef SM_R2_LIGHT_RELEASE
#define SM_R2_LIGHT_RELEASE 0
#endif
#ifndef SM_R3_SRC_CREDIT
#define SM_R3_SRC_CREDIT 1
#endif
#ifndef SM_R4_LAST_WG_FENCE
#define SM_R4_LAST_WG_FENCE 1
#endif
#ifndef SM_P0
#define SM_P0 1
#endif
#ifndef SM_P1
#define SM_P1 1
#endif
#ifndef SM_P2
#define SM_P2 0
#endif
#ifndef SM_P3
#define SM_P3 0
#endif

// Test-only negative controls / stress (default 0; never candidates): SM_TEST_NO_READER_ACQUIRE drops the system acquire fence after the spin;
// SM_TEST_ADD_BEFORE_RETIRE issues the push release adds BEFORE vmcnt(0)+barrier; SM_TEST_DELAY_PUSH_WG=<wg>+1 sleeps that push WG before its adds.
// SM_MICRO_STAMP (default 0, stamps build only): push WG thread 0 times ONE lone epoch load (word 62) to separate per-load first-touch latency from chaining.
#ifndef SM_MICRO_STAMP
#define SM_MICRO_STAMP 0
#endif
// B-F2 negative control (indexer build only): K cache uses the DENSE formula fp8(x/s) instead of the golden indexer fp8(float(bf16(x))/s).
#ifndef SM_TEST_K_DENSE_ROUND
#define SM_TEST_K_DENSE_ROUND 0
#endif
// r7x numerics-fidelity switches (default = golden-matching best derivation; each can be turned off for A/B by Evidence):
//  SM_ZSIGN (default 1): non-rotary output dims (dd >= 64) are computed exactly like the Triton golden: nrm*1.0 + (0.0*0.0) (+0.0 forced via asm so -0 becomes +0, as in golden_k: no -0 in dims >= 64, measured).
//  SM_RMS_ORDER (default 0 = r6 serial-16 sum; 1 = balanced adjacent binary tree over the 128 squares; 2 = far-first xor butterfly): order of the variance sum. UNPROVEN which one matches
//   Triton's lowering of tl.sum on gfx950 (see tests/fit_k_scale_order.py: Evidence picks by exact k_scale match rate on saved data).
//  SM_ROPE_FMA (default 0 = compiler default contraction; 1 = fmaf(nrm, cs, rot*sn); 2 = fmaf(rot, sn, nrm*cs); 3 = no contraction: separate mul/mul/add) for the RoPE sum.
// r7x4: MODE-SPECIFIC receiver numerics. REUSE (and dense) golden = Triton (derived orders below, defaults 3/3/1/1). INDEXER golden = AITER HIP fused_qknorm_idxrqknorm.cu (read-only source):
//   32 lanes x 4 elems; sumsq = per-lane serial 4-term fma chain then xor butterfly 16,8,4,2,1 (warpReduceSum l.26-34) => SM_RMS_ORDER=4;
//   RoPE = e*c - p*s (first half) / e*c + p*s (second half) with the POST-norm partner, contracted by the compiler (hipcc default fast contraction) => SM_ROPE_FMA=4: fma(e, c, -(p*s)) / fma(e, c, p*s);
//   dims >= 64 are NOT touched (no +0) and no `0 - g`: SM_ZSIGN=0, SM_ROT_NEG=0; K cache = fp8(float(bf16(elem))/scale) (already kq()). rsqrt = v_rsq_f32 (rsqrtf -> same instruction as Triton's).
// Evidence: CPU fit on 40 retained rows (tests/fit_hip_indexer.py): this model reproduces golden k_scale bit-exactly on 36/40 rows, all-128 FP8 codes on 40/40; the 4 misses need r -1 ulp (hardware rsqrt).
// FMA contraction choice (fmaA vs fmaB) and the per-lane sumsq fma-vs-round are INDISTINGUISHABLE on this data (identical 36/40): fmaA/fma chosen as the LLVM DAG-combine default; UNPROVEN without the aiter kernel ISA.
#if SM_R5X_INDEXER
#define SM_DEFAULT_ZSIGN 0
#define SM_DEFAULT_ROT_NEG 0
#define SM_DEFAULT_RMS_ORDER 4
#define SM_DEFAULT_ROPE_FMA 4
#else
#define SM_DEFAULT_ZSIGN 1
#define SM_DEFAULT_ROT_NEG 1
#define SM_DEFAULT_RMS_ORDER 3
#define SM_DEFAULT_ROPE_FMA 3
#endif
#ifndef SM_ZSIGN
#define SM_ZSIGN SM_DEFAULT_ZSIGN
#endif
#ifndef SM_RMS_ORDER
#define SM_RMS_ORDER SM_DEFAULT_RMS_ORDER   // reuse/dense: 3 = Triton DPP scan (derived from AMDGCN); indexer: 4 = AITER HIP 32-lane xor tree; 0 = r6/r5x serial16, 1/2 = modelled trees (A/B)
#endif
//  SM_RMS_ORDER=3 (DERIVED from the Triton 3.6.0+rocm7.2.4 gfx950 AMDGCN of _fused_qkv_norm_rope_cache_kernel, K path l.56-200; num_warps=4, 2 waves hold the 128 elements, waves 2-3 duplicate):
//    per wave of 64 lanes (element e <-> lane e&63, rows of 16 lanes = e>>4&3):  s0 = k*k (rounded);  step1: lane l>=8 in its 16-row: fma(k_l, k_l, s0[l-8]) (own square UNROUNDED), l<8: s0;
//    then row_shr 4,2,1 plain adds (inclusive scan; shifted-in zeros at row starts): s2=s1+s1[l-4], s3=s2+s2[l-2], s4=s3+s3[l-1]; row total T_r = s4[row lane 15];
//    row_bcast15: W = (T1+T0) on row 1 / (T3+T2) on row 3;  row_bcast31: wave total = (T3+T2)+(T1+T0);  cross-wave: W0 + W1 (LDS, then one add).
//    var+eps = fma(sum, 2^-7, 1e-6) (== sum/128 + eps: the product is exact), r = v_rsq_f32 (no refinement), nrm = (k*r)*(1+w).
//  SM_ROT_NEG (default 1, DERIVED): first-half rotated operand is `-(g) + 0` (v_fma_f32 -g*(1+w) + 0 in the golden) = (0 - g): +0 for g = +-0, i.e. never -0; second half keeps g.
//  SM_RSQ_BUILTIN (default 1): use __builtin_amdgcn_rsqf (v_rsq_f32) instead of rsqrtf (token47 dump: the CPU model with an exactly rounded rsqrt misses 4/40 rows by 1-2 ulp of r).
//  SM_FP8_SCALEF32 (default 0): FP8 convert via v_cvt_scalef32_pk_fp8_f32 (golden uses it with scale 1.0) instead of opus::fp32_to_fp8 (v_cvt_pk_fp8_f32). UNVERIFIED equivalence: A/B.
#ifndef SM_ROT_NEG
#define SM_ROT_NEG SM_DEFAULT_ROT_NEG
#endif
#ifndef SM_RSQ_BUILTIN
#define SM_RSQ_BUILTIN 1    // default ON (token47 dump fit): the golden rsqrt is the plain v_rsq_f32 (~1 ulp, NOT correctly rounded); 4 of 40 dumped rows only match with the hardware value, never with an exactly rounded 1/sqrt
#endif
#ifndef SM_FP8_SCALEF32
#define SM_FP8_SCALEF32 0
#endif
#ifndef SM_ROPE_FMA
#define SM_ROPE_FMA SM_DEFAULT_ROPE_FMA   // reuse/dense: 3 = Triton unfused (pk_mul + add); indexer: 4 = HIP contracted fma(e,c,-+(p*s)); 0 = compiler default, 1/2 = fma orders (A/B)
#endif
// r8 (D8 D1): SM_NO_GRIDREAD=1 (default) removes the AQL dispatch-packet read (grid_size_x) from the critical path: G = compile-time SM_T1_GRID; WG0 checks the real grid OFF-path at the end of the kernel (volatile packet read) and poisons 0x41 on mismatch. 0 restores the r7 read (A/B).
#ifndef SM_NO_GRIDREAD
#define SM_NO_GRIDREAD 1
#endif
// r11: head_row LDS diet. SM_AMAX_REG (default 1): the per-row amax (K and V) is a register xor-max over the 16 lanes (max is exact and order independent => bit-identical); the RoPE partner exchange is ALWAYS an in-register
// ds_swizzle xor 4 (partner dim d+-32 lives on lane L^4 at the same element index); SM_LDS_LEGACY = the variance orders that still stage through LDS (0,1,2,13,14).
// With SM_RMS_ORDER 3/4 and SM_AMAX_REG=1 head_row touches NO LDS and has NO barrier. tests/check_lds_protection.py proves hazard-freedom of every remaining LDS use per configuration (and bites on a negative control).
#ifndef SM_AMAX_REG
#define SM_AMAX_REG 1
#endif
#if SM_RMS_ORDER == 3 || SM_RMS_ORDER == 4
#define SM_LDS_LEGACY 0
#else
#define SM_LDS_LEGACY 1
#endif
#define SM_HEAD_TAIL_SYNC (SM_LDS_LEGACY || !SM_AMAX_REG)   // trailing barrier protects s_part/s_row WAR between consecutive head_row calls
// r12 (D9 P1): SM_P1 (default 1) = B WG credit release BEFORE the head math and output stores: Rx loads -> vmcnt(0) -> barrier -> system release credit -> math + stores. 0 = r11 placement (A/B).
// Test seams (default 0, never candidates): SM_TEST_DELAY_CREDIT=<n> thread 0 sleeps n x s_sleep(127) before the credit release (widens the WAR window: sources must wait, credit-poll iterations rise, outputs stay correct);
// SM_TEST_EARLY_CREDIT=1 NEGATIVE CONTROL: credit released right after the flag acquire, BEFORE the Rx loads; with SM_TEST_DELAY_B_LOADS=<n> (sleep before the Rx loads) a source can overwrite Rx[par] before it is read -> wrong output must be DETECTED.
#ifndef SM_P1
#define SM_P1 1
#endif
#ifndef SM_TEST_DELAY_CREDIT
#define SM_TEST_DELAY_CREDIT 0
#endif
#ifndef SM_TEST_EARLY_CREDIT
#define SM_TEST_EARLY_CREDIT 0
#endif
#ifndef SM_TEST_DELAY_B_LOADS
#define SM_TEST_DELAY_B_LOADS 0
#endif
#if SM_TEST_EARLY_CREDIT && !SM_P1
#error "SM_TEST_EARLY_CREDIT is a negative control of the P1 path (needs SM_P1=1)"
#endif
// r13: head_row for row pass 1 is skipped when M <= 16 (`pass < passes`, passes = ceil(M/16) is WG-uniform, so legacy LDS barriers inside head_row stay uniform). The loads stay unconditional (clamped); every store/amax/scale in head_row was already guarded by live = (m < M), so outputs are byte-identical.
// r17 (D12 R1): SM_NO_CREDIT (default 1; 0 = r15b behaviour). No WAR credit: B WGs do not release it (no vmcnt/barrier/system release for it), push WGs/IK owner do not load or check the credit words (no acquire fence). The flag release/acquire chain,
// poison, bounded polls and epoch words are UNCHANGED. Safety argument (note): D reads Rx_D[par] in D's RAN e; D's push release for e+1 is stream-ordered after it; S's RAN e+1 acquires flag_S[D] (>= K*(e+1)) in its B WGs; S's RAN e+2 (the next writer of Rx_D[par]) is
// stream-ordered after S's RAN e+1 => reads(e) hb flag release(e+1) -> acquire(e+1) hb writes(e+2). Precondition: every source performs at least one acquire of flag_S[D] per destination D in every epoch (B WGs exist for every (src,dst) pair).
// Test seams (default 0, never candidates): SM_TEST_SKIP_FLAG_ACQ_FOR_SRC=<s+1>: B WGs for source s skip the flag poll/acquire (break the chain); SM_TEST_DELAY_B_RX=<n>: B thread 0 sleeps n x s_sleep(127) after the flag acquire and BEFORE the Rx loads (delayed reader, works without any credit).
#ifndef SM_NO_CREDIT
#define SM_NO_CREDIT 1
#endif
#ifndef SM_TEST_SKIP_FLAG_ACQ_FOR_SRC
#define SM_TEST_SKIP_FLAG_ACQ_FOR_SRC 0
#endif
#ifndef SM_TEST_DELAY_B_RX
#define SM_TEST_DELAY_B_RX 0
#endif
#if SM_NO_CREDIT && (SM_TEST_EARLY_CREDIT || SM_TEST_DELAY_CREDIT)
#error "SM_TEST_EARLY_CREDIT / SM_TEST_DELAY_CREDIT test the credit; build them with SM_NO_CREDIT=0"
#endif
#ifndef SM_TEST_NO_READER_ACQUIRE
#define SM_TEST_NO_READER_ACQUIRE 0
#endif
#ifndef SM_TEST_ADD_BEFORE_RETIRE
#define SM_TEST_ADD_BEFORE_RETIRE 0
#endif
#ifndef SM_TEST_DELAY_PUSH_WG
#define SM_TEST_DELAY_PUSH_WG 0
#endif
#ifdef __HIP_DEVICE_COMPILE__
#include <opus/opus.hpp>
// Resource gate (Evidence task-218: private 416 B scratch): helpers MUST inline so local arrays never have their address escape to a call,
// and every local-array index must be a compile-time constant (pass/k loops use constant trip counts + full unroll with a uniform guard).
#define SM_INL inline __device__ __attribute__((always_inline))
#include "layout/ran_layouts.hpp"
namespace sm {
using namespace mori::cco;
struct alignas(16) U32x4 { uint32_t w[4]; };
struct alignas(8)  U32x2 { uint32_t w[2]; };
struct alignas(16) F32x4 { float v[4]; };
// r7 (K1): window bases come from the device-resident SmDev block (written once per bind). The kernel never calls ccoGet*Ptr.
#if !SM_U2_HOST_PTRS
#error "SM-T1-r7 requires SM_U2_HOST_PTRS=1 (host-resolved window pointers live in SmDev)"
#endif
SM_INL uint8_t* pick4(uint8_t* p0, uint8_t* p1, uint8_t* p2, uint8_t* p3, int i) { return i == 0 ? p0 : (i == 1 ? p1 : (i == 2 ? p2 : p3)); }   // scalar select
SM_INL uint8_t* cwin_local(const SmDev& a, uint32_t off) { return a.cwin_loc + off; }
SM_INL uint8_t* cwin_peer(const SmDev& a, int peer, uint32_t off) { return pick4(a.cwin_pr0, a.cwin_pr1, a.cwin_pr2, a.cwin_pr3, peer) + off; }
SM_INL uint8_t* rwin_local(const SmDev& a) { return a.rwin_loc; }
SM_INL uint8_t* rwin_peer(const SmDev& a, int peer) { return pick4(a.rwin_pr0, a.rwin_pr1, a.rwin_pr2, a.rwin_pr3, peer); }SM_INL uint64_t ld_acq64(const void* p) { return __hip_atomic_load(reinterpret_cast<const uint64_t*>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); }
SM_INL uint64_t ld_rlx64(const void* p) { return __hip_atomic_load(reinterpret_cast<const uint64_t*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
SM_INL void st_rel64(void* p, uint64_t v) {   // same W4-C1 pattern (unused by the T1/T2 kernel bodies; kept consistent)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __hip_atomic_store(reinterpret_cast<uint64_t*>(p), v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
SM_INL uint32_t ld_acq32(const void* p) { return __hip_atomic_load(reinterpret_cast<const uint32_t*>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); }
SM_INL uint32_t ld_rlx32(const void* p) { return __hip_atomic_load(reinterpret_cast<const uint32_t*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
SM_INL void acquire_fence_system() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); }   // "" = system scope
SM_INL void poison(const SmDev& a, uint32_t why) {
    uint32_t z = 0u;
    __hip_atomic_compare_exchange_strong(reinterpret_cast<uint32_t*>(cwin_local(a, SM_C_POISON)), &z, why,
        __ATOMIC_ACQ_REL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// Bounded poll, thread 0 only (see r2): R1 relaxed system spin + ONE system acquire fence on success (caller: barrier, then reads).
SM_INL bool poll64(const SmDev& a, const void* p, uint64_t expect, bool eq, uint32_t why, uint64_t* iters) {
    for (uint64_t i = 0; i < a.poll_iterations; ++i) {
#if SM_R1_RELAXED_SPIN
        const uint64_t v = ld_rlx64(p);
#else
        const uint64_t v = ld_acq64(p);
#endif
        if (eq ? (v == expect) : (v >= expect)) {
#if SM_R1_RELAXED_SPIN && !SM_TEST_NO_READER_ACQUIRE
            acquire_fence_system();
#endif
            if (iters) *iters = i;
            return true;
        }
        if (eq && v > expect) { poison(a, why | 0x100u); return false; }
        __builtin_amdgcn_s_sleep(2);
    }
    poison(a, why);
    return false;
}
// flag wait: relaxed system spin until v >= lo; poison if v > hi or on timeout; one system acquire fence on success.
SM_INL bool poll_range64(const SmDev& a, const void* p, uint64_t lo, uint64_t hi, uint32_t why, uint64_t* iters) {
    for (uint64_t i = 0; i < a.poll_iterations; ++i) {
        const uint64_t v = ld_rlx64(p);
        if (v > hi) { poison(a, why | 0x100u); return false; }
        if (v >= lo) {
#if !SM_TEST_NO_READER_ACQUIRE
            acquire_fence_system();
#endif
            if (iters) *iters = i;
            return true;
        }
        __builtin_amdgcn_s_sleep(2);
    }
    poison(a, why);
    return false;
}
SM_INL uint16_t bf16_bits(float f) { return __builtin_bit_cast(unsigned short, opus::fp32_to_bf16(f)); }
// r9x: in-register xor exchange inside a 16-lane row (ds_swizzle bitmask mode: and_mask 0x1f, or_mask 0, xor_mask M -> lane L reads lane L ^ M; the offset has no LDS memory traffic, only the LDS crossbar).
template <int M> SM_INL float swz_f(float v) {
    static_assert(M >= 1 && M <= 8, "xor within a 16-lane row");
    return __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), (M << 10) | 0x1f));
}
SM_INL float bf16_val(uint16_t b) { return opus::bf16_to_fp32(__builtin_bit_cast(opus::bf16_t, b)); }
SM_INL uint32_t pack2(float lo, float hi) { return static_cast<uint32_t>(bf16_bits(lo)) | (static_cast<uint32_t>(bf16_bits(hi)) << 16); }

SM_INL void unit_dest(uint32_t n0, int& d, uint32_t& fc) {
    if (n0 < 8192u)      { d = n0 / 2048u;            fc = n0 % 2048u; }
    else if (n0 < 8704u) { d = (n0 - 8192u) / 128u;   fc = 2048u; }
    else                 { d = (n0 - 8704u) / 128u;   fc = 2176u; }
}
SM_INL uint32_t unit_head(uint32_t n0) { if (n0 < 8192u) return (n0 % 2048u) / 128u; return (n0 < 8704u) ? 16u : 17u; }
// raw column of (destination d, head h) and its packed column
SM_INL uint32_t head_n0(uint32_t d, uint32_t h) { return (h < 16u) ? d * 2048u + h * 128u : ((h == 16u) ? 8192u + d * 128u : ((h == 17u) ? 8704u + d * 128u : ((h == 18u) ? 9216u + d * 128u : 9728u))); }   // 19 (IK): destination-independent
SM_INL uint32_t head_fc(uint32_t h) { return (h < 16u) ? h * 128u : 2048u + (h - 16u) * 128u; }

SM_INL void ld8f(const float* p, float* o) {
    if ((reinterpret_cast<__UINTPTR_TYPE__>(p) & 15u) == 0u) {
        F32x4 a = *reinterpret_cast<const F32x4*>(p), b = *reinterpret_cast<const F32x4*>(p + 4);
        o[0]=a.v[0]; o[1]=a.v[1]; o[2]=a.v[2]; o[3]=a.v[3]; o[4]=b.v[0]; o[5]=b.v[1]; o[6]=b.v[2]; o[7]=b.v[3];
    } else { _Pragma("unroll") for (int i = 0; i < 8; ++i) o[i] = p[i]; }
}
SM_INL void ld8bf(const uint16_t* p, float* o) {
    uint16_t t[8];
    if ((reinterpret_cast<__UINTPTR_TYPE__>(p) & 15u) == 0u) { U32x4 w = *reinterpret_cast<const U32x4*>(p); __builtin_memcpy(t, &w, 16); }
    else { _Pragma("unroll") for (int i = 0; i < 8; ++i) t[i] = p[i]; }
    _Pragma("unroll") for (int i = 0; i < 8; ++i) o[i] = bf16_val(t[i]);
}
SM_INL void st8bf(uint16_t* p, const float* v) {                      // r2 helper (kept for the P1-off output path)
    uint16_t t[8];
    _Pragma("unroll") for (int i = 0; i < 8; ++i) t[i] = bf16_bits(v[i]);
    if ((reinterpret_cast<__UINTPTR_TYPE__>(p) & 15u) == 0u) { U32x4 w; __builtin_memcpy(&w, t, 16); *reinterpret_cast<U32x4*>(p) = w; }
    else { _Pragma("unroll") for (int i = 0; i < 8; ++i) p[i] = t[i]; }
}
// P1 output store: ONE 16 B aligned-type store, no runtime alignment branch (q/k/v rows are 16 B aligned by construction: 8 bf16 per lane,
// row strides 2048/128 elements; base pointers are torch allocations).
// r15 Q2: SM_NT_STORES=1 emits non-temporal (global_store ... nt) stores for the B/self outputs, K/V caches and scales (values and bytes unchanged); the push Rx stores (SC0|SC1) and the index cache are NOT touched.
// 8-byte K-cache store of a U32x2 (struct): a vector type is required by __builtin_nontemporal_store
SM_INL void st_nt8(uint8_t* p, const U32x2& w) {
#if SM_NT_STORES
    typedef unsigned int v2u32 __attribute__((ext_vector_type(2)));
    v2u32 vv; vv[0] = w.w[0]; vv[1] = w.w[1];
    __builtin_nontemporal_store(vv, reinterpret_cast<v2u32*>(p));
#else
    *reinterpret_cast<U32x2*>(p) = w;
#endif
}
template <class T> SM_INL void st_nt(T* p, T v) {
#if SM_NT_STORES
    __builtin_nontemporal_store(v, p);
#else
    *p = v;
#endif
}
SM_INL void store_out8(uint16_t* p, const float* v) {
#if SM_P1
    U32x4 w; w.w[0] = pack2(v[0], v[1]); w.w[1] = pack2(v[2], v[3]); w.w[2] = pack2(v[4], v[5]); w.w[3] = pack2(v[6], v[7]);
#if SM_NT_STORES
    typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));
    v4u32 vv; vv[0] = w.w[0]; vv[1] = w.w[1]; vv[2] = w.w[2]; vv[3] = w.w[3];
    __builtin_nontemporal_store(vv, reinterpret_cast<v4u32*>(p));
#else
    *reinterpret_cast<U32x4*>(p) = w;
#endif
#else
    st8bf(p, v);
#endif
}
// P0 push: one 16 B system-scope (SC0|SC1 = 17) store through a buffer resource whose base is the wave-uniform peer window base.
// elem_off is in bf16 elements (multiple of 8 -> 16 B aligned inside the Rx layout [par][src][32][2304]).
SM_INL void push16(uint16_t* rx_peer_base, int elem_off, const float* v) {
#if SM_P0
    opus::vector_t<opus::u32_t, 4> x;
    x[0] = pack2(v[0], v[1]); x[1] = pack2(v[2], v[3]); x[2] = pack2(v[4], v[5]); x[3] = pack2(v[6], v[7]);
    auto g = opus::make_gmem(reinterpret_cast<const opus::u32_t*>(rx_peer_base), static_cast<unsigned int>(SM_RX_BYTES));
    g.template store<4>(x, elem_off >> 1, 0, opus::number<17>{});     // u32 units; aux = SC0|SC1
#else
    st8bf(rx_peer_base + elem_off, v);
#endif
}

struct RowPre { int64_t pos; int64_t slot; int64_t islot; float cs[8]; float sn[8]; };
SM_INL uint8_t f8(float v) {
#if SM_FP8_SCALEF32
    // golden (Triton) lowering: v_cvt_scalef32_pk_fp8_f32 dst, src0, src1, scale(1.0)  - low byte holds fp8(src0)
    uint32_t w; asm volatile("v_cvt_scalef32_pk_fp8_f32 %0, %1, %2, 1.0" : "=v"(w) : "v"(v), "v"(0.0f));
    return static_cast<uint8_t>(w & 0xFFu);
#else
    return __builtin_bit_cast(uint8_t, opus::fp32_to_fp8(v));
#endif
}
// K-cache quant operand. Golden AITER indexer (storeCacheElems l.99-100): fp8( float(bf16(elem)) / scale ), scale from the FP32 (pre-rounding) amax.
// Dense/reuse (Triton norm_rope_cache.py l.250): fp8( k_roped_fp32 / scale ).
SM_INL float kq(float v, float ks) {
#if SM_R5X_INDEXER && !SM_TEST_K_DENSE_ROUND
    return bf16_val(bf16_bits(v)) / ks;
#else
    return v / ks;
#endif
}   // per (row,pass): prefetched before the flag poll (T1 Q5)
// Head math for ONE 16-row pass; ALL threads of the WG call it (contains WG barriers). h: 0..15 Q, 16 K, 17 V.
SM_INL void head_row(const SmDev& a, uint32_t h, uint32_t t, bool live, int row16, int lane16, const float* x,
                     float* s_row, float* s_part, const RowPre* pre = nullptr, const float* wpre = nullptr) {
    if (h == 17u) {
        float am = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) am = fmaxf(am, fabsf(x[i]));
#if SM_AMAX_REG
        am = fmaxf(am, swz_f<8>(am)); am = fmaxf(am, swz_f<4>(am)); am = fmaxf(am, swz_f<2>(am)); am = fmaxf(am, swz_f<1>(am));   // r11: exact register max over the 16 lanes of the row
        const float amax = am;
#else
        s_part[row16 * 16 + lane16] = am;
        mori::cco::impl::syncThreads();
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, s_part[row16 * 16 + i]);
        mori::cco::impl::syncThreads();
#endif
        if (live) {
            if (a.v_out) store_out8(a.v_out + static_cast<size_t>(t) * 128u + lane16 * 8, x);   // indexer golden returns no v: host may pass nullptr
            const int64_t slot = pre ? pre->slot : a.slot_mapping[t];
            if (slot >= 0) {
                const int64_t blk = slot / a.block_size, sib = slot % a.block_size;
                float vs = amax / a.fp8_amax; if (vs == 0.0f) vs = 1.0f;
                if (lane16 == 0) st_nt(&a.v_scale[blk * a.vs_sb + sib], vs);
                uint8_t* vc = static_cast<uint8_t*>(a.v_cache);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int64_t dd = lane16 * 8 + i;
                    st_nt<uint8_t>(&vc[blk * a.vc_sb + (sib / a.x_size) * a.vc_ssc + dd * a.vc_sd + (sib % a.x_size) * a.vc_sx], __builtin_bit_cast(uint8_t, opus::fp32_to_fp8(x[i] / vs)));
                }
            }
        }
        return;
    }
    const float* nw = (h < 16u) ? a.qw : ((h == 16u) ? a.kw : ((h == 18u) ? a.iqw : a.ikw));
#if SM_RMS_ORDER <= 2
    float ss = 0.0f;                                                   // r11: only the LDS-staged variance orders 0/1/2 read these partials
    #pragma unroll
    for (int i = 0; i < 8; ++i) ss += x[i] * x[i];
    s_part[row16 * 16 + lane16] = ss;
    mori::cco::impl::syncThreads();
#endif
    float var;
#if SM_RMS_ORDER == 1
    {   // balanced ADJACENT binary tree over the 16 lane partials (each lane partial = serial sum of its 8 squares): ((p0+p1)+(p2+p3))+...  (models a DPP/adjacent-pair warp tree)
        float t0 = s_part[row16 * 16 + 0] + s_part[row16 * 16 + 1], t1 = s_part[row16 * 16 + 2] + s_part[row16 * 16 + 3];
        float t2 = s_part[row16 * 16 + 4] + s_part[row16 * 16 + 5], t3 = s_part[row16 * 16 + 6] + s_part[row16 * 16 + 7];
        float t4 = s_part[row16 * 16 + 8] + s_part[row16 * 16 + 9], t5 = s_part[row16 * 16 + 10] + s_part[row16 * 16 + 11];
        float t6 = s_part[row16 * 16 + 12] + s_part[row16 * 16 + 13], t7 = s_part[row16 * 16 + 14] + s_part[row16 * 16 + 15];
        var = (((t0 + t1) + (t2 + t3)) + ((t4 + t5) + (t6 + t7)));
    }
#elif SM_RMS_ORDER == 4
    {
        // r9x: AITER HIP 32 lanes x 4 elements, REGISTER exchange (no LDS, no barrier, no serial loop). Our lane L (16 per token row) holds x[8L..8L+7] = HIP lanes 2L (elements 0..3) and 2L+1 (4..7):
        // two serial fma chains c0, c1; HIP butterfly masks 16,8,4,2 act on each chain across our lanes (xor 8,4,2,1 within the 16-lane row, ds_swizzle), the last mask (1) is c0 + c1.
        // Bit-identical to the golden xor tree by construction of the pairing (tests/check_rms_mapping.py).
        #pragma clang fp contract(off)
        float c0 = 0.0f, c1 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { c0 = __builtin_fmaf(x[i], x[i], c0); c1 = __builtin_fmaf(x[4 + i], x[4 + i], c1); }
        c0 = c0 + swz_f<8>(c0); c1 = c1 + swz_f<8>(c1);
        c0 = c0 + swz_f<4>(c0); c1 = c1 + swz_f<4>(c1);
        c0 = c0 + swz_f<2>(c0); c1 = c1 + swz_f<2>(c1);
        c0 = c0 + swz_f<1>(c0); c1 = c1 + swz_f<1>(c1);
        var = c0 + c1;
    }
#elif SM_RMS_ORDER == 14
    {
        // AITER HIP: 32 lanes x 4 elements (lane L holds x[4L..4L+3]); per-lane serial fma chain sumsq += e*e (hipcc fast contraction), then warpReduceSum xor 16,8,4,2,1.
        // Stage x exactly; lane16 == 0 of each token row emulates the whole warp (a pairwise tree of identical operands: every lane ends with the same value).
        #pragma unroll
        for (int i = 0; i < 8; ++i) s_row[row16 * 128 + lane16 * 8 + i] = x[i];
        mori::cco::impl::syncThreads();
        if (lane16 == 0) {
            #pragma clang fp contract(off)
            float pl[32];
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                float s = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) { const float e = s_row[row16 * 128 + 4 * l + i]; s = __builtin_fmaf(e, e, s); }
                pl[l] = s;
            }
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                float nx[32];
                #pragma unroll
                for (int l = 0; l < 32; ++l) nx[l] = pl[l] + pl[l ^ m];
                #pragma unroll
                for (int l = 0; l < 32; ++l) pl[l] = nx[l];
            }
            s_part[row16 * 16 + 0] = pl[0];
        }
        mori::cco::impl::syncThreads();
        var = s_part[row16 * 16 + 0];
    }
#elif SM_RMS_ORDER == 3
    {
        // r9x: Triton DPP scan, REGISTER exchange (no LDS, no barrier, no serial loop). Triton wave row r of 16 lanes = 16 consecutive elements = OUR TWO lanes (L, L^1): A (bit0 = 0) holds Triton lanes 0..7 of the row, B (bit0 = 1) lanes 8..15.
        // step1 (row_shr:8 + v_fmac): B-lane element i = fma(k_B, k_B, round(k_A^2)); A-lane element = round(k^2)  [we compute the 8 B-type values redundantly in both lanes after ONE xor-1 swizzle of x]
        // then row_shr 4/2/1 scan, row total T = s4[15]: (((s1[7]+s1[3])+(s1[5]+s1[1]))+((s1[6]+s1[2])+(s1[4]+s1[0])))  (the scan tree of the last element, evaluated directly), row_bcast15/31 -> (T3+T2)+(T1+T0) per wave,
        // cross-wave W0 + W1: rows are our lane bits 1..2 (xor 2, 4), the wave is bit 3 (xor 8): pairwise sums commute exactly, so every lane ends with the golden total. Verified bit-exact: tests/check_rms_mapping.py.
        #pragma clang fp contract(off)
        const bool hi = (lane16 & 1) != 0;
        float s1[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float po = swz_f<1>(x[i]);
            const float kA = hi ? po : x[i];
            const float kB = hi ? x[i] : po;
            const float sq = kA * kA;
            s1[i] = __builtin_fmaf(kB, kB, sq);
        }
        float T = ((s1[7] + s1[3]) + (s1[5] + s1[1])) + ((s1[6] + s1[2]) + (s1[4] + s1[0]));
        T = T + swz_f<2>(T);
        T = T + swz_f<4>(T);
        T = T + swz_f<8>(T);
        var = T;
    }
#elif SM_RMS_ORDER == 13
    {
        // x staged exactly (the later normalised write to s_row happens after another barrier); lane16 == 0 of each token row emulates the two 64-lane waves.
        #pragma unroll
        for (int i = 0; i < 8; ++i) s_row[row16 * 128 + lane16 * 8 + i] = x[i];
        mori::cco::impl::syncThreads();
        if (lane16 == 0) {
            // One 16-lane ROW at a time (small register footprint). Contraction is disabled here: the only fma allowed is the explicit step-1 fma below (as in the golden DPP sequence).
            #pragma clang fp contract(off)
            float wave_tot[2];
            #pragma unroll
            for (int w = 0; w < 2; ++w) {
                float T[4];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float* kr = &s_row[row16 * 128 + w * 64 + r * 16];
                    float s1[16];
                    #pragma unroll
                    for (int l = 0; l < 16; ++l) {                                      // step 1: row_shr:8 then v_fmac: lanes >= 8 fma(k_l, k_l, round(k_{l-8}^2)); lanes < 8: round(k_l^2)
                        const float k = kr[l];
                        if (l >= 8) { const float km = kr[l - 8]; const float sqm = km * km; s1[l] = __builtin_fmaf(k, k, sqm); }
                        else { s1[l] = k * k; }
                    }
                    float s2[16], s3[16], s4[16];
                    #pragma unroll
                    for (int l = 0; l < 16; ++l) s2[l] = (l >= 4) ? (s1[l] + s1[l - 4]) : s1[l];   // row_shr:4 (shifted-in zero is an identity for non-negative squares)
                    #pragma unroll
                    for (int l = 0; l < 16; ++l) s3[l] = (l >= 2) ? (s2[l] + s2[l - 2]) : s2[l];   // row_shr:2
                    #pragma unroll
                    for (int l = 0; l < 16; ++l) s4[l] = (l >= 1) ? (s3[l] + s3[l - 1]) : s3[l];   // row_shr:1
                    T[r] = s4[15];                                                                  // row total at its last lane
                }
                const float b1 = T[1] + T[0], b3 = T[3] + T[2];                         // row_bcast:15 (rows 1 and 3 add the previous row's total)
                wave_tot[w] = b3 + b1;                                                  // row_bcast:31 -> lane 63
            }
            s_part[row16 * 16 + 0] = wave_tot[0] + wave_tot[1];
        }
        mori::cco::impl::syncThreads();
        var = s_part[row16 * 16 + 0];
    }
#elif SM_RMS_ORDER == 2
    {   // FAR-FIRST butterfly (xor 8,4,2,1): lane i pairs with i^8 first: (p0+p8) ... then ^4, ^2, ^1
        float u0 = s_part[row16 * 16 + 0] + s_part[row16 * 16 + 8], u1 = s_part[row16 * 16 + 1] + s_part[row16 * 16 + 9];
        float u2 = s_part[row16 * 16 + 2] + s_part[row16 * 16 + 10], u3 = s_part[row16 * 16 + 3] + s_part[row16 * 16 + 11];
        float u4 = s_part[row16 * 16 + 4] + s_part[row16 * 16 + 12], u5 = s_part[row16 * 16 + 5] + s_part[row16 * 16 + 13];
        float u6 = s_part[row16 * 16 + 6] + s_part[row16 * 16 + 14], u7 = s_part[row16 * 16 + 7] + s_part[row16 * 16 + 15];
        float v0 = u0 + u4, v1 = u1 + u5, v2 = u2 + u6, v3 = u3 + u7;
        var = ((v0 + v2) + (v1 + v3));
    }
#else
    var = 0.0f;
    for (int i = 0; i < 16; ++i) var += s_part[row16 * 16 + i];                 // r6: serial over the 16 lane partials
#endif
    var = var / 128.0f;
#if SM_RSQ_BUILTIN || SM_RMS_ORDER == 3 || SM_RMS_ORDER == 4 || SM_RMS_ORDER == 13 || SM_RMS_ORDER == 14
    const float rs = __builtin_amdgcn_rsqf(var + a.eps);       // v_rsq_f32 (golden Triton and AITER HIP both lower to the plain instruction)
#else
    const float rs = rsqrtf(var + a.eps);
#endif
    float w8[8];
    if (wpre) { _Pragma("unroll") for (int i = 0; i < 8; ++i) w8[i] = wpre[i]; } else ld8f(nw + lane16 * 8, w8);
    float nrm[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) nrm[i] = x[i] * rs * (1.0f + w8[i]);
    float pnv[8];                                                      // r11: RoPE partner (dim d+-32 = lane L^4, same element index i) by in-register ds_swizzle xor 4; every lane executes it (uniform)
    #pragma unroll
    for (int i = 0; i < 8; ++i) pnv[i] = swz_f<4>(nrm[i]);
    const int64_t pos = pre ? pre->pos : (live ? a.positions[t] : 0);
    float outv[8];
    if (lane16 < 8) {
        const int dd0 = lane16 * 8;
        const int ci0 = (dd0 < 32) ? dd0 : dd0 - 32;
        float cs[8], sn[8];
        if (pre) { _Pragma("unroll") for (int i = 0; i < 8; ++i) { cs[i] = pre->cs[i]; sn[i] = pre->sn[i]; } }
        else { ld8bf(a.cos_cache + pos * a.cos_stride_pos + ci0, cs); ld8bf(a.sin_cache + pos * a.cos_stride_pos + ci0, sn); }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int dd = dd0 + i;
            const float pn = pnv[i];                                   // == post-norm value of dim (dd < 32 ? dd + 32 : dd - 32)
#if SM_ROT_NEG
            float rot;
            if (dd < 32) { asm volatile("v_sub_f32 %0, 0, %1" : "=v"(rot) : "v"(pn)); }   // 0 - g: never -0 (golden: v_fma_f32 -g*(1+w) + 0)
            else rot = pn;
#else
            const float rot = (dd < 32) ? -pn : pn;
#endif
#if SM_ROPE_FMA == 4
            // AITER HIP normAndRope l.223-224: first half e*c - p*s, second half e*c + p*s; contraction fuses the FIRST product: fma(e, c, -+(p*s)), p*s rounded.
            outv[i] = (dd < 32) ? __builtin_fmaf(nrm[i], cs[i], -(pn * sn[i])) : __builtin_fmaf(nrm[i], cs[i], pn * sn[i]);
#elif SM_ROPE_FMA == 1
            outv[i] = __builtin_fmaf(nrm[i], cs[i], rot * sn[i]);
#elif SM_ROPE_FMA == 2
            outv[i] = __builtin_fmaf(rot, sn[i], nrm[i] * cs[i]);
#elif SM_ROPE_FMA == 3
            { float m0, m1; asm volatile("v_mul_f32 %0, %1, %2" : "=v"(m0) : "v"(nrm[i]), "v"(cs[i])); asm volatile("v_mul_f32 %0, %1, %2" : "=v"(m1) : "v"(rot), "v"(sn[i])); asm volatile("v_add_f32 %0, %1, %2" : "=v"(outv[i]) : "v"(m0), "v"(m1)); }
#else
            outv[i] = nrm[i] * cs[i] + rot * sn[i];
#endif
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
#if SM_ZSIGN
            // golden (Triton): q_roped = q_normed*cos + q_rot*sin with cos=1.0 and q_rot=where(rot_mask,..,+0.0) for dd >= 64  =>  nrm*1.0 + (+0.0*+0.0): -0 becomes +0.
            float zz; asm volatile("v_add_f32 %0, 0, %1" : "=v"(zz) : "v"(nrm[i])); outv[i] = zz;      // asm: the compiler may not fold x + 0.0f -> x
#else
            outv[i] = nrm[i];
#endif
        }
    }
    if (live) {
        if (h < 16u)       store_out8(a.q_out + static_cast<size_t>(t) * 2048u + h * 128u + lane16 * 8, outv);
        else if (h == 16u) { if (a.k_out) store_out8(a.k_out + static_cast<size_t>(t) * 128u + lane16 * 8, outv); }   // indexer golden returns no k
        else if (h == 18u) store_out8(a.index_q_out + static_cast<size_t>(t) * 128u + lane16 * 8, outv);              // IQ: BF16 of FP32 post-RoPE
    }
    if (h == 19u) {                                                      // IK: index_cache[slot*128 + d] = fp8( float(bf16(x)) / 1.0 ) (golden storeCacheElems scale 1.0, l.479-483)
        if (live) {
            const int64_t is = pre ? pre->islot : (a.index_slot_mapping ? a.index_slot_mapping[t] : -1);
            if (is >= 0) {
                uint8_t q8[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) q8[i] = f8(bf16_val(bf16_bits(outv[i])) / 1.0f);
                U32x2 w; __builtin_memcpy(&w, q8, 8);
                *reinterpret_cast<U32x2*>(static_cast<uint8_t*>(a.index_cache) + is * 128 + lane16 * 8) = w;
            }
        }
#if SM_HEAD_TAIL_SYNC
        mori::cco::impl::syncThreads();
#endif
        return;
    }
    if (h == 16u) {
        float am = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) am = fmaxf(am, fabsf(outv[i]));
#if SM_AMAX_REG
        am = fmaxf(am, swz_f<8>(am)); am = fmaxf(am, swz_f<4>(am)); am = fmaxf(am, swz_f<2>(am)); am = fmaxf(am, swz_f<1>(am));   // r11: exact register max over the 16 lanes of the row
        const float amax = am;
#else
        mori::cco::impl::syncThreads();
        s_part[row16 * 16 + lane16] = am;
        mori::cco::impl::syncThreads();
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, s_part[row16 * 16 + i]);
#endif
        if (live) {
            const int64_t slot = pre ? pre->slot : a.slot_mapping[t];
            if (slot >= 0) {
                const int64_t blk = slot / a.block_size, sib = slot % a.block_size;
                float ks = amax / a.fp8_amax; if (ks == 0.0f) ks = 1.0f;
                if (lane16 == 0) st_nt(&a.k_scale[blk * a.ks_sb + sib], ks);
                uint8_t* kc = static_cast<uint8_t*>(a.k_cache);
#if SM_P1
                // 8 contiguous bytes per lane (dd..dd+7 inside one x_size group) -> ONE 8 B store; uniform runtime check, else scalar.
                const bool k8 = (a.kc_sx == 1) && (a.x_size % 8 == 0) && (((a.kc_sb | a.kc_sdx | a.kc_ss) & 7) == 0);
                if (k8) {
                    const int64_t dd = lane16 * 8;
                    uint8_t q8[8];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) q8[i] = f8(kq(outv[i], ks));
                    U32x2 w; __builtin_memcpy(&w, q8, 8);
                    st_nt8(kc + blk * a.kc_sb + (dd / a.x_size) * a.kc_sdx + sib * a.kc_ss + (dd % a.x_size), w);
                } else
#endif
                {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int64_t dd = lane16 * 8 + i;
                        st_nt<uint8_t>(&kc[blk * a.kc_sb + (dd / a.x_size) * a.kc_sdx + sib * a.kc_ss + (dd % a.x_size) * a.kc_sx], f8(kq(outv[i], ks)));
                    }
                }
            }
        }
    }
#if SM_HEAD_TAIL_SYNC
    mori::cco::impl::syncThreads();
#endif
}

// ---------------- SM-T1-r6 helpers (D6 V1/V2: ONE memory round trip per WG role) ----------------
// All loads below are UNCONDITIONAL with clamped row indices (rows >= M re-read row M-1 and are discarded later); alignment of every vector
// load is guaranteed by construction + host validation (sm_bind_v1), so there is no runtime alignment branch either.
SM_INL float bf_lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
SM_INL float bf_hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xFFFF0000u); }
SM_INL void ld8f_a(const float* p, float* o) {
    const F32x4 a = *reinterpret_cast<const F32x4*>(p), b = *reinterpret_cast<const F32x4*>(p + 4);
    o[0]=a.v[0]; o[1]=a.v[1]; o[2]=a.v[2]; o[3]=a.v[3]; o[4]=b.v[0]; o[5]=b.v[1]; o[6]=b.v[2]; o[7]=b.v[3];
}
SM_INL void ld8bf_a(const uint16_t* p, float* o) {                                   // 8 BF16 -> fp32 (exact: bits << 16, same as opus::bf16_to_fp32)
    const U32x4 w = *reinterpret_cast<const U32x4*>(p);
    o[0]=bf_lo(w.w[0]); o[1]=bf_hi(w.w[0]); o[2]=bf_lo(w.w[1]); o[3]=bf_hi(w.w[1]); o[4]=bf_lo(w.w[2]); o[5]=bf_hi(w.w[2]); o[6]=bf_lo(w.w[3]); o[7]=bf_hi(w.w[3]);
}
SM_INL uint32_t rowc(uint32_t m, uint32_t Mm1) { return m < Mm1 ? m : Mm1; }       // clamp to the last live row
SM_INL void pre_a(const SmDev& a, uint32_t t, RowPre& p) { p.pos = a.positions[t]; p.slot = a.slot_mapping[t]; 
#if SM_R5X_INDEXER
    p.islot = a.index_slot_mapping ? a.index_slot_mapping[t] : -1;
#else
    p.islot = -1;                                    // dense/reuse: no index cache (r10: no load at all)
#endif
 }   // unconditional loads: the SmDev pointer is host-validated non-null in indexer builds          // round 1
SM_INL void pre_b(const SmDev& a, int lane16, RowPre& p) {                                                              // round 2 (address depends on pos)
    const int dd0 = (lane16 & 7) * 8, ci0 = (dd0 < 32) ? dd0 : dd0 - 32;                                                  // lanes >= 8 load a valid row; results unused
    ld8bf_a(a.cos_cache + p.pos * a.cos_stride_pos + ci0, p.cs);
    ld8bf_a(a.sin_cache + p.pos * a.cos_stride_pos + ci0, p.sn);
}
// r14: SK partials (SM_SK = 2|3), ascending split order: (((0 + p[0]) + p[1]) + p[2]) * ws * as, ONE BF16 RNE. SK=2 is the r13 arithmetic exactly.
SM_INL float red_u(const float (&p)[SM_SK_CFG], float ws, float as) {
    float acc = 0.0f;
    _Pragma("unroll") for (int q = 0; q < SM_SK_CFG; ++q) acc = acc + p[q];
    const float sw = acc * ws; const float sa = sw * as;
    return bf16_val(bf16_bits(sa));
}
// Release RMW (review W4-C1 pattern): system-scope release fence (wbl2) -> explicit s_waitcnt vmcnt(0) -> RELAXED system RMW. Callers keep the per-wave vmcnt(0) + barrier before thread 0 gets here.
SM_INL void sys_add_release(void* p, uint64_t v) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __hip_atomic_fetch_add(reinterpret_cast<uint64_t*>(p), v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// Entry reads (U2): agent-scope relaxed (ep_g is GEMM-written earlier in this stream; poison only local/host). U2=0 keeps r4 system-scope loads.
SM_INL uint64_t ld_ep64(const void* p) {
#if SM_U2_HOST_PTRS
    return __hip_atomic_load(reinterpret_cast<const uint64_t*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    return ld_rlx64(p);
#endif
}
SM_INL uint32_t ld_ep32(const void* p) {
#if SM_U2_HOST_PTRS
    return __hip_atomic_load(reinterpret_cast<const uint32_t*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    return ld_rlx32(p);
#endif
}
// Pure decision on already-loaded values (no memory access, no atomics): 0 = ok, 0xFFFFFFFF = already poisoned (silent skip), else the poison code to record LATER
// (the cold CAS is deferred to the end of the kernel so no atomic/wait sits between the first load and the first push store).
SM_INL uint32_t decide(const SmDev& a, uint32_t M, uint32_t G, uint64_t e, uint32_t pz) {
    if (pz != 0u) return 0xFFFFFFFFu;
    if (e == 0ull) return 0x42u;                                                     // GEMM did not run / wrong pairing
    if (M == 0u || M > SM_MP || a.T != 4u * M) return 0x40u;
#if SM_PUSH8
    if (M > 16u) return 0x40u;                                                       // G-1: SM_PUSH8 handles one row pass per half (M <= 16 only); device-side guard in addition to the host refusal
#endif
    if (G != SM_T1_GRID) return 0x41u;                                               // static roles need exactly SM_T1_GRID WGs
    return 0u;
}
} // namespace sm

extern "C" __global__ __launch_bounds__(SM_RAN_THREADS)
void sm_ran_v1(const SmDev* __restrict__ blk, uint32_t M, uint32_t rank) {
#if defined(__gfx950__)
    using namespace sm;
    const SmDev& a = *blk;      // restrict const kernarg pointer: invariant loads, the compiler may emit scalar (s_load) loads lazily per field
    __shared__ uint32_t s_ticket;
    __shared__ uint32_t s_ok;
    __shared__ float s_row[16 * 128];
    __shared__ float s_part[16 * 16];
    const int tid = opus::thread_id_x();
    const uint32_t wgid = static_cast<uint32_t>(opus::block_id_x());
#if SM_NO_GRIDREAD
    const uint32_t G = SM_T1_GRID;                       // static roles require exactly this grid; verified off-path by WG0 at the end
#else
    const uint32_t G = static_cast<uint32_t>(__builtin_amdgcn_grid_size_x()) / SM_RAN_THREADS;
#if SM_STAMPS
    asm volatile("s_waitcnt lgkmcnt(0)" :: "s"(G));       // stamp 31: dispatch-packet latency alone (before the M/rank wait)
#endif
#endif
    uint64_t* const rec = SM_STAMP_REC(a.dbg, 0u, wgid < SM_DBG_RAN_MAXWG ? wgid : SM_DBG_RAN_MAXWG - 1u);
    if (tid == 0) { SM_STAMP(rec, 0); SM_STAMP_HWID(rec, 59); SM_STAMP_XCC(rec, 58); }
#if SM_STAMPS
#if !SM_NO_GRIDREAD
    if (tid == 0) SM_STAMP(rec, 31);
#endif
    asm volatile("s_waitcnt lgkmcnt(0)" :: "s"(M), "s"(rank));   // stamp 32: kernarg (M/rank) latency alone
    if (tid == 0) SM_STAMP(rec, 32);
#endif
#if SM_RAN_THREADS > 256
    if (wgid >= SM_T1_PUSH_WGS && __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(tid) >> 6) >= 4u) return;   // I-1: wave-uniform (SGPR) condition -> scalar branch to s_endpgm, no s_barrier reachable on this path       // r15: 512-thread launch; IK/self/B WGs use waves 0-3 only. A terminated wave leaves the workgroup barrier count (s_endpgm), so the remaining waves' s_barrier still completes. RISK documented in the note.
#endif
    const int lane16 = tid & 15, row16 = tid >> 4;
    const uint32_t passes = (M + 15u) / 16u;
    uint16_t* const rx_local = reinterpret_cast<uint16_t*>(rwin_local(a));
    constexpr uint32_t NH = SM_T1_B_HEADS;
    const uint32_t Mm1 = (M != 0u && M <= SM_MP) ? M - 1u : 0u;              // last live row (clamp target; invalid launches are rejected by decide())
    uint64_t max_cr = 0, sum_fl = 0;
    uint32_t nA = 0, nB = 0, errc = 0;


#if SM_R5X_INDEXER
    if (wgid == SM_T1_PUSH_WGS) {
        // ======== IK OWNER WG (exactly ONE per source rank): ONE memory round trip; reduce/scale/BF16-round IK ONCE, push the byte-identical row to the 3 remote destinations,
        //          release once per destination, then fuse the SELF destination from the same registers (head_row h = 19). ========
        if (tid == 0) SM_STAMP(rec, 24);
        const uint32_t n0 = head_n0(0u, 19u);
        float pk[SM_SK_CFG][2][8], ws[8], as_[2];
        ld8f_a(a.Wscale + n0 + lane16 * 8, ws);
        _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
            const uint32_t rc = rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1);
            _Pragma("unroll") for (int q = 0; q < SM_SK_CFG; ++q) ld8f_a(a.P + sml::p_view()(q, static_cast<int>(rc), static_cast<int>(n0) + lane16 * 8), pk[q][pass]);
            as_[pass] = a.Ascale[rc];
        }
        const uint64_t e = ld_ep64(cwin_local(a, SM_C_EPOCH_G));
        const uint32_t pz = ld_ep32(cwin_local(a, SM_C_POISON));
#if !SM_NO_CREDIT
        uint64_t cr3[3];
        _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) cr3[rr] = ld_rlx64(cwin_local(a, SM_T1_CREDIT((rr >= rank) ? rr + 1u : rr)));   // credit words in the same round trip
#endif
        asm volatile("" ::: "memory");
        __builtin_amdgcn_s_waitcnt(0x0);                                           // THE single wait
        if (tid == 0) SM_STAMP(rec, 29);
        if (tid == 0) {
            errc = decide(a, M, G, e, pz);
            bool okk = (errc == 0u);
#if !SM_NO_CREDIT
            if (okk && e > 2ull) {
                const uint64_t need = static_cast<uint64_t>(t1r::B_PER_PAIR) * (e - 2ull);
                bool all = true;
                _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) all = all && (cr3[rr] >= need);
                if (all) acquire_fence_system();
                else {                                                              // slow path: bounded poll per destination
                    _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) { uint64_t it = 0; if (okk) okk = poll64(a, cwin_local(a, SM_T1_CREDIT((rr >= rank) ? rr + 1u : rr)), need, false, 0x10u, &it); if (it > max_cr) max_cr = it; }
                }
            }
#endif
            s_ok = okk ? 1u : 0u;
        }
        mori::cco::impl::syncThreads();
        const bool ok = (s_ok == 1u);
        const uint32_t par = static_cast<uint32_t>(e & 1ull);
        if (tid == 0) SM_STAMP(rec, 30);
        float vv[2][8];
        _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) { _Pragma("unroll") for (int i = 0; i < 8; ++i) vv[pass][i] = 0.0f; }
        if (ok) {
            _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
                const uint32_t m = static_cast<uint32_t>(row16) + 16u * pass;
                _Pragma("unroll") for (int i = 0; i < 8; ++i) { float pp[SM_SK_CFG]; _Pragma("unroll") for (int q = 0; q < SM_SK_CFG; ++q) pp[q] = pk[q][pass][i]; vv[pass][i] = red_u(pp, ws[i], as_[pass]); }   // the ONE reduction/scale/BF16 rounding of IK
                if (m < M) {
                    _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) {
                        const uint32_t d = (rr >= rank) ? rr + 1u : rr;
                        push16(reinterpret_cast<uint16_t*>(rwin_peer(a, static_cast<int>(d))), sml::rx_view()(static_cast<int>(par), static_cast<int>(rank), static_cast<int>(m), static_cast<int>(head_fc(19u)) + lane16 * 8), vv[pass]);
                    }
                }
            }
            nA = 1;
        }
        __builtin_amdgcn_s_waitcnt(0x0);                                           // EVERY wave: vmcnt(0) before the release barrier, all exec paths
        if (tid == 0) SM_STAMP(rec, 25);
        mori::cco::impl::syncThreads();
        if (tid == 0 && ok) {                                                      // ONE release per destination (fence(release,system) -> vmcnt(0) -> relaxed RMW)
            _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) { const uint32_t d = (rr >= rank) ? rr + 1u : rr; sys_add_release(cwin_peer(a, static_cast<int>(d), SM_T1_FLAG(rank)), 1ull); }
            SM_STAMP(rec, 26);
        }
        if (ok) {                                                                  // self fusion AFTER the remote release (does not delay the other ranks)
            RowPre pre[2];
            _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) { const uint32_t rc = rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1); pre_a(a, rank * M + rc, pre[pass]); }
            _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) pre_b(a, lane16, pre[pass]);
            float wq[8]; ld8f_a(a.ikw + lane16 * 8, wq);
            _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
                const uint32_t m = static_cast<uint32_t>(row16) + 16u * pass;
                if (pass < passes) head_row(a, 19u, rank * M + m, m < M, row16, lane16, vv[pass], s_row, s_part, &pre[pass], wq);
            }
        }
        if (tid == 0) SM_STAMP(rec, 28);
    } else
#endif
    if (wgid < SM_T1_PUSH_WGS) {
#if SM_PUSH8 || SM_PUSH_HALVES
        // ======== r15 PUSH WG, 512 threads = two 256-thread halves. ONE memory round trip, ONE wait. Slot (half, s) -> (unit index uidx within the WG, row pass):
        //   Q1 (SM_PUSH8, M <= 16): the WG owns nu = 6..8 units (touching 1 or 2 destinations); half hf takes units [hf*H, hf*H + H) with H = ceil(nu/2) <= 4 (slots beyond nu duplicate the last unit, discarded); pass 0 only.
        //   Q3 (SM_PUSH_HALVES): the WG owns nu = 2..3 units of ONE destination; half hf owns row pass hf; all nu units.
        const uint32_t g0 = t1r::g0(wgid), g1 = t1r::g0(wgid + 1u), nu = g1 - g0;
        const uint32_t hf = static_cast<uint32_t>(tid) >> 8, t8 = static_cast<uint32_t>(tid) & 255u;
        const int prow = static_cast<int>(t8 >> 4), plane = static_cast<int>(t8 & 15u);
#if SM_PUSH8
        constexpr uint32_t NS = 4u;
        const uint32_t H = (nu + 1u) >> 1;
#else
        constexpr uint32_t NS = 3u;
#endif
        if (tid == 0) SM_STAMP(rec, 24);
        float pk[SM_SK_CFG][NS][8], ws[NS][8], as_v;
        uint32_t sl_u[NS]; bool sl_real[NS];
        _Pragma("unroll") for (uint32_t kk = 0; kk < NS; ++kk) {
#if SM_PUSH8
            const uint32_t raw = hf * H + kk;                                         // unit index within the WG
            sl_real[kk] = (kk < H) && (raw < nu);
            sl_u[kk] = (raw < nu) ? raw : (nu - 1u);
#else
            sl_real[kk] = (kk < nu);
            sl_u[kk] = (kk < nu) ? kk : (nu - 1u);
#endif
        }
        const uint32_t pass_h =
#if SM_PUSH8
            0u;
#else
            hf;
#endif
        const uint32_t rcl = rowc(static_cast<uint32_t>(prow) + 16u * pass_h, Mm1);
        _Pragma("unroll") for (uint32_t kk = 0; kk < NS; ++kk) {
            const uint32_t gu = g0 + sl_u[kk];
            const uint32_t rr_ = gu / t1r::HEADS, dd_ = (rr_ >= rank) ? rr_ + 1u : rr_;
            const uint32_t n0 = head_n0(dd_, gu % t1r::HEADS);
            ld8f_a(a.Wscale + n0 + plane * 8, ws[kk]);
            _Pragma("unroll") for (int q = 0; q < SM_SK_CFG; ++q) ld8f_a(a.P + sml::p_view()(q, static_cast<int>(rcl), static_cast<int>(n0) + plane * 8), pk[q][kk]);
        }
        as_v = a.Ascale[rcl];
        const uint64_t e = ld_ep64(cwin_local(a, SM_C_EPOCH_G));
        const uint32_t pz = ld_ep32(cwin_local(a, SM_C_POISON));
#if !SM_NO_CREDIT
        // credit words of the (<= 2) touched destinations, same round trip (loaded for every WG: cheap, system relaxed like the poll)
        uint64_t crw[3];
        _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) crw[rr] = ld_rlx64(cwin_local(a, SM_T1_CREDIT((rr >= rank) ? rr + 1u : rr)));
#endif
        asm volatile("" ::: "memory");
        __builtin_amdgcn_s_waitcnt(0x0);                                           // THE single wait
        if (tid == 0) SM_STAMP(rec, 29);
        if (tid == 0) {
            errc = decide(a, M, G, e, pz);
            bool okk = (errc == 0u);
#if !SM_NO_CREDIT
            if (okk && e > 2ull) {
                const uint64_t need = static_cast<uint64_t>(t1r::B_PER_PAIR) * (e - 2ull);
                bool all = true;
                _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) if (t1r::touches(wgid, rr)) all = all && (crw[rr] >= need);
                if (all) acquire_fence_system();
                else {
                    _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) if (t1r::touches(wgid, rr) && okk) { uint64_t it = 0; okk = poll64(a, cwin_local(a, SM_T1_CREDIT((rr >= rank) ? rr + 1u : rr)), need, false, 0x10u, &it); if (it > max_cr) max_cr = it; }
                }
            }
#endif
            s_ok = okk ? 1u : 0u;
        }
        mori::cco::impl::syncThreads();
        const bool ok = (s_ok == 1u);
        const uint32_t par = static_cast<uint32_t>(e & 1ull);
        if (tid == 0) SM_STAMP(rec, 30);
        if (ok) {
            const uint32_t m = static_cast<uint32_t>(prow) + 16u * pass_h;
            _Pragma("unroll") for (uint32_t kk = 0; kk < NS; ++kk) if (sl_real[kk] && m < M) {
                const uint32_t gu = g0 + sl_u[kk];
                const uint32_t rr_ = gu / t1r::HEADS, dd_ = (rr_ >= rank) ? rr_ + 1u : rr_;
                uint16_t* const rx_dst = reinterpret_cast<uint16_t*>(rwin_peer(a, static_cast<int>(dd_)));
                float v[8];
                _Pragma("unroll") for (int i = 0; i < 8; ++i) { float pp[SM_SK_CFG]; _Pragma("unroll") for (int q = 0; q < SM_SK_CFG; ++q) pp[q] = pk[q][kk][i]; v[i] = red_u(pp, ws[kk][i], as_v); }
                push16(rx_dst, sml::rx_view()(static_cast<int>(par), static_cast<int>(rank), static_cast<int>(m), static_cast<int>(head_fc(gu % t1r::HEADS)) + plane * 8), v);
            }
            nA = nu;
        }
        __builtin_amdgcn_s_waitcnt(0x0);                                    // EVERY wave: vmcnt(0) (push stores retired) BEFORE the barrier
        if (tid == 0) SM_STAMP(rec, 25);
        mori::cco::impl::syncThreads();
        if (ok) {
            // one system RELEASE per touched destination (fence(release,system) -> vmcnt(0) -> relaxed RMW on flag[src]); the first by thread 0, the second (Q1 straddlers) by thread 256 = a different wave, so the two releases overlap
            if (tid == 0) {
                _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) if (t1r::touches(wgid, rr)) { sys_add_release(cwin_peer(a, static_cast<int>((rr >= rank) ? rr + 1u : rr), SM_T1_FLAG(rank)), 1ull); break; }
                SM_STAMP(rec, 26);
            }
#if SM_PUSH8
            if (tid == 256) {
                uint32_t seen = 0;
                _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) if (t1r::touches(wgid, rr)) { if (seen++ == 1u) sys_add_release(cwin_peer(a, static_cast<int>((rr >= rank) ? rr + 1u : rr), SM_T1_FLAG(rank)), 1ull); }
            }
#endif
        }
#else

        if (tid == 0) SM_STAMP(rec, 33);                                // first statement of the push block (far-branch i-fetch probe)
        // ======== PUSH WG (24 = 3 per XCC): 2..3 units of ONE destination. ONE memory round trip: all loads -> ONE vmcnt(0) -> decide -> compute -> stores -> release ========
        const uint32_t g0 = t1r::g0(wgid), g1 = t1r::g0(wgid + 1u), nu = g1 - g0;       // nu in {2,3}; the third slot of a 2-unit WG is a discarded duplicate of unit 1
        const uint32_t r = g0 / t1r::HEADS, d = (r >= rank) ? r + 1u : r;
        if (tid == 0) SM_STAMP(rec, 24);
#if SM_MICRO_STAMP
        if (tid == 0) { const uint64_t v0 = ld_rlx64(cwin_local(a, SM_C_EPOCH_G)); asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); SM_STAMP(rec, 62); asm volatile("" :: "v"(v0)); }   // lone-load latency falsifier
#endif
        float pk[SM_SK_CFG][3][2][8], ws[3][8], as_[2];
        _Pragma("unroll") for (uint32_t kk = 0; kk < 3u; ++kk) {
            const uint32_t kc = (kk < nu) ? kk : (nu - 1u);
            const uint32_t n0 = head_n0(d, (g0 + kc) % t1r::HEADS);
            ld8f_a(a.Wscale + n0 + lane16 * 8, ws[kk]);
            _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
                const uint32_t rc = rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1);
                _Pragma("unroll") for (int q = 0; q < SM_SK_CFG; ++q) ld8f_a(a.P + sml::p_view()(q, static_cast<int>(rc), static_cast<int>(n0) + lane16 * 8), pk[q][kk][pass]);
            }
        }
        _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) as_[pass] = a.Ascale[rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1)];
        const uint64_t e = ld_ep64(cwin_local(a, SM_C_EPOCH_G));
        const uint32_t pz = ld_ep32(cwin_local(a, SM_C_POISON));
#if !SM_NO_CREDIT
        const uint64_t cr = ld_rlx64(cwin_local(a, SM_T1_CREDIT(d)));              // credit word loaded in the same round trip (system-scope relaxed, like the poll)
#endif
        asm volatile("" ::: "memory");                                             // compiler barrier: nothing above may sink below, nothing below may hoist above
        __builtin_amdgcn_s_waitcnt(0x0);                                           // THE single wait (vmcnt(0)); the self-check script gates on exactly one
        if (tid == 0) SM_STAMP(rec, 29);                                           // loads COMPLETE
        if (tid == 0) {
            errc = decide(a, M, G, e, pz);
            bool okk = (errc == 0u);
#if !SM_NO_CREDIT
            if (okk && e > 2ull) {                                                 // WAR credit: fast path = value already loaded; slow path = bounded poll
                const uint64_t need = static_cast<uint64_t>(t1r::B_PER_PAIR) * (e - 2ull);
                if (cr >= need) acquire_fence_system();
                else { uint64_t it = 0; okk = poll64(a, cwin_local(a, SM_T1_CREDIT(d)), need, false, 0x10u, &it); max_cr = it; }
            }
#endif
            s_ok = okk ? 1u : 0u;
        }
        mori::cco::impl::syncThreads();
        const bool ok = (s_ok == 1u);
        const uint32_t par = static_cast<uint32_t>(e & 1ull);
        if (tid == 0) SM_STAMP(rec, 30);
        if (ok) {
            uint16_t* const rx_dst = reinterpret_cast<uint16_t*>(rwin_peer(a, static_cast<int>(d)));
            _Pragma("unroll") for (uint32_t kk = 0; kk < 3u; ++kk) if (kk < nu) {
                const uint32_t h = (g0 + kk) % t1r::HEADS;
                _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
                    const uint32_t m = static_cast<uint32_t>(row16) + 16u * pass;
                    if (m < M) {
                        float v[8];
                        _Pragma("unroll") for (int i = 0; i < 8; ++i) { float pp[SM_SK_CFG]; _Pragma("unroll") for (int q = 0; q < SM_SK_CFG; ++q) pp[q] = pk[q][kk][pass][i]; v[i] = red_u(pp, ws[kk][i], as_[pass]); }
                        push16(rx_dst, sml::rx_view()(static_cast<int>(par), static_cast<int>(rank), static_cast<int>(m), static_cast<int>(head_fc(h)) + lane16 * 8), v);
                    }
                }
            }
            nA = nu;
        }
#if SM_TEST_ADD_BEFORE_RETIRE
        if (tid == 0 && ok) sys_add_release(cwin_peer(a, static_cast<int>(d), SM_T1_FLAG(rank)), 1ull);   // NEGATIVE CONTROL
#endif
        __builtin_amdgcn_s_waitcnt(0x0);                                    // EVERY wave: vmcnt(0) (push stores retired) BEFORE the barrier, all exec paths
        if (tid == 0) SM_STAMP(rec, 25);
        mori::cco::impl::syncThreads();
#if SM_TEST_DELAY_PUSH_WG
        if (wgid + 1u == SM_TEST_DELAY_PUSH_WG && tid == 0) { for (int i = 0; i < 400; ++i) __builtin_amdgcn_s_sleep(127); }   // late-push-WG stale detector
#endif
#if !SM_TEST_ADD_BEFORE_RETIRE
        if (tid == 0 && ok) {                                               // ONE system RELEASE (fence(release,system) -> vmcnt(0) -> relaxed RMW) on flag[src] of the single destination
            sys_add_release(cwin_peer(a, static_cast<int>(d), SM_T1_FLAG(rank)), 1ull);
            SM_STAMP(rec, 26);
        }
#endif
#endif
    } else if (wgid < SM_T1_B0) {
        // ======== SELF WG: one head of the local destination. Round 1: P/Wscale/Ascale/positions/slots/weights/epoch/poison; round 2 (cos/sin, address depends on pos) overlaps compute ========
        const uint32_t h = wgid - SM_T1_S0;                                  // self heads 0..UPD-1 (IK self is fused by the IK owner WG)
        if (tid == 0) SM_STAMP(rec, 27);
        const uint32_t n0 = head_n0(rank, h);
        float pk[SM_SK_CFG][2][8], ws[8], as_[2], wq[8];
        RowPre pre[2];
        ld8f_a(a.Wscale + n0 + lane16 * 8, ws);
        const float* const wsel = (h < 16u) ? a.qw : ((h == 16u) ? a.kw : a.iqw);   // scalar select; h==17 (V) unused (harmless); h==18 IQ
        ld8f_a(wsel + lane16 * 8, wq);
        _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
            const uint32_t rc = rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1);
            _Pragma("unroll") for (int q = 0; q < SM_SK_CFG; ++q) ld8f_a(a.P + sml::p_view()(q, static_cast<int>(rc), static_cast<int>(n0) + lane16 * 8), pk[q][pass]);
            as_[pass] = a.Ascale[rc];
            pre_a(a, rank * M + rc, pre[pass]);
        }
        const uint64_t e = ld_ep64(cwin_local(a, SM_C_EPOCH_G));
        const uint32_t pz = ld_ep32(cwin_local(a, SM_C_POISON));
        asm volatile("" ::: "memory");
        __builtin_amdgcn_s_waitcnt(0x0);                                           // round 1 complete
        _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) pre_b(a, lane16, pre[pass]);   // round 2 issued now, consumed only inside head_row
        if (tid == 0) { errc = decide(a, M, G, e, pz); s_ok = (errc == 0u) ? 1u : 0u; }
        mori::cco::impl::syncThreads();
        const bool ok = (s_ok == 1u);
        if (ok) {
            float v[2][8];
            _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) { _Pragma("unroll") for (int i = 0; i < 8; ++i) { float pp[SM_SK_CFG]; _Pragma("unroll") for (int q = 0; q < SM_SK_CFG; ++q) pp[q] = pk[q][pass][i]; v[pass][i] = red_u(pp, ws[i], as_[pass]); } }
            _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
                const uint32_t m = static_cast<uint32_t>(row16) + 16u * pass;
                if (pass < passes) head_row(a, h, rank * M + m, m < M, row16, lane16, v[pass], s_row, s_part, &pre[pass], wq);
            }
        }
        if (tid == 0) SM_STAMP(rec, 28);
        nA = 1;
    } else if (wgid < SM_T1_GRID) {
        // ======== B WG: NH heads of ONE remote source. Round 1 = positions/slots/weights/epoch/poison (overlaps nothing it needs later); cos/sin issued after; all Rx rows (all heads, both passes) loaded in ONE batch after the single acquire ========
        const uint32_t bi = wgid - SM_T1_B0, r = bi / t1r::B_PER_PAIR, pair = bi % t1r::B_PER_PAIR;
        const uint32_t s = (r >= rank) ? r + 1u : r;
        const uint32_t h0 = pair * NH;
        RowPre pre[2];
        float wq[NH][8];
        _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) pre_a(a, s * M + rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1), pre[pass]);
        _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) ld8f_a((((h0 + kk) < 16u) ? a.qw : (((h0 + kk) == 16u) ? a.kw : (((h0 + kk) == 18u) ? a.iqw : a.ikw))) + lane16 * 8, wq[kk]);     // V head: weights unused
        const uint64_t e = ld_ep64(cwin_local(a, SM_C_EPOCH_G));
        const uint32_t pz = ld_ep32(cwin_local(a, SM_C_POISON));
        if (tid == 0) SM_STAMP(rec, 7);
        asm volatile("" ::: "memory");
        __builtin_amdgcn_s_waitcnt(0x0);                                           // round 1 complete
        _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) pre_b(a, lane16, pre[pass]);   // round 2 (cos/sin), overlaps the decision + flag poll
        if (tid == 0) { errc = decide(a, M, G, e, pz); s_ok = (errc == 0u) ? 1u : 0u; }
        mori::cco::impl::syncThreads();
        bool ok = (s_ok == 1u);
        const uint32_t par = static_cast<uint32_t>(e & 1ull);
#if SM_TEST_SKIP_FLAG_ACQ_FOR_SRC
        if (ok && s != SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)                 // NEGATIVE CONTROL: B WGs of source (SKIP-1) skip the flag poll + acquire fence (broken chain)
#else
        if (ok)
#endif
        {
            if (tid == 0) {
                uint64_t it = 0;
                const uint32_t rr = (rank > s) ? rank - 1u : rank;
                const uint64_t K = t1r::kcount(rr);
                s_ticket = poll_range64(a, cwin_local(a, SM_T1_FLAG(s)), K * e, K * (e + 1ull), 0x20u, &it) ? 1u : 0u;   // >= K*e, poison if > K*(e+1); own system acquire fence on exit
                sum_fl = it; SM_STAMP(rec, 8); SM_STAMP_VAL(rec, 15, it);
            }
            mori::cco::impl::syncThreads();                                // fence (thread 0) -> barrier -> reads
            if (s_ticket != 1u) ok = false;
            mori::cco::impl::syncThreads();
        }
#if SM_TEST_DELAY_B_RX
        if (ok) { if (tid == 0) { for (int dly = 0; dly < SM_TEST_DELAY_B_RX; ++dly) __builtin_amdgcn_s_sleep(127); } mori::cco::impl::syncThreads(); }   // delayed reader: after the flag acquire, before the Rx loads
#endif
#if SM_TEST_EARLY_CREDIT
        if (ok) {                                                          // NEGATIVE CONTROL: credit BEFORE the reads (illegal), optionally followed by a delay before the loads
            if (tid == 0) sys_add_release(cwin_peer(a, static_cast<int>(s), SM_T1_CREDIT(rank)), 1ull);
#if SM_TEST_DELAY_B_LOADS
            if (tid == 0) { for (int dly = 0; dly < SM_TEST_DELAY_B_LOADS; ++dly) __builtin_amdgcn_s_sleep(127); }
#endif
            mori::cco::impl::syncThreads();
        }
#endif
#if SM_P1
        // r12 / D9 P1: the credit only has to order OUR Rx READS before the source's next-epoch writes into Rx[par]. Sequence: ONE batch of Rx loads -> every wave vmcnt(0) (the loads have RETURNED: values bound in VGPRs)
        // -> barrier -> thread 0: system release credit (fence(release) -> vmcnt(0) -> relaxed RMW) -> only then math and ALL output stores (q/k/v/caches/scales never touch Rx or the control window).
        if (ok) {
            float x[NH][2][8];
            _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
                const uint32_t rc = rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1);
                ld8bf_a(rx_local + sml::rx_view()(static_cast<int>(par), static_cast<int>(s), static_cast<int>(rc), static_cast<int>(head_fc(h0 + kk)) + lane16 * 8), x[kk][pass]);
            }
#if SM_NO_CREDIT
            if (tid == 0) { SM_STAMP_VAL(rec, 34, ~0ull); SM_STAMP_VAL(rec, 35, ~0ull); }                  // R-1: credit stamps N/A (all-ones) under SM_NO_CREDIT
#else
            asm volatile("" ::: "memory");                                 // the loads may not sink below the wait / barrier
            __builtin_amdgcn_s_waitcnt(0x0);                               // EVERY wave: all Rx loads (and the earlier cos/sin loads) complete; no store has been issued yet in this WG
            if (tid == 0) SM_STAMP(rec, 35);
            mori::cco::impl::syncThreads();                                // all waves' loads complete before the release
#endif
#if !SM_TEST_EARLY_CREDIT && !SM_NO_CREDIT
            if (tid == 0) {
#if SM_TEST_DELAY_CREDIT
                for (int dly = 0; dly < SM_TEST_DELAY_CREDIT; ++dly) __builtin_amdgcn_s_sleep(127);
#endif
                sys_add_release(cwin_peer(a, static_cast<int>(s), SM_T1_CREDIT(rank)), 1ull);                 // release-sequence credit (B_PER_PAIR per source per epoch)
                SM_STAMP(rec, 34);
            }
#endif
            _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
                const uint32_t m = static_cast<uint32_t>(row16) + 16u * pass;
                if (pass < passes) head_row(a, h0 + kk, s * M + m, m < M, row16, lane16, x[kk][pass], s_row, s_part, &pre[pass], wq[kk]);
            }
        }
        if (tid == 0) SM_STAMP(rec, 9);
        if (tid == 0 && ok) {
            // split epoch words: RAN reads ep_g (GEMM-written) and writes ep_r = e here; nothing in this launch reads ep_r (next launch's GEMM reads it after kernel completion).
            __hip_atomic_store(reinterpret_cast<uint64_t*>(cwin_local(a, SM_C_EPOCH)), e, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
#else
        if (ok) {
            float x[NH][2][8];                                             // ONE batch of unconditional clamped Rx loads (all heads, both passes), then math
            _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
                const uint32_t rc = rowc(static_cast<uint32_t>(row16) + 16u * pass, Mm1);
                ld8bf_a(rx_local + sml::rx_view()(static_cast<int>(par), static_cast<int>(s), static_cast<int>(rc), static_cast<int>(head_fc(h0 + kk)) + lane16 * 8), x[kk][pass]);
            }
            _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {
                const uint32_t m = static_cast<uint32_t>(row16) + 16u * pass;
                if (pass < passes) head_row(a, h0 + kk, s * M + m, m < M, row16, lane16, x[kk][pass], s_row, s_part, &pre[pass], wq[kk]);
            }
        }
        if (tid == 0) SM_STAMP(rec, 9);
#if !SM_NO_CREDIT
        __builtin_amdgcn_s_waitcnt(0x0);                                   // every wave: vmcnt(0) before the credit barrier
        mori::cco::impl::syncThreads();
#endif
        if (tid == 0 && ok) {
#if SM_TEST_DELAY_CREDIT
            for (int dly = 0; dly < SM_TEST_DELAY_CREDIT; ++dly) __builtin_amdgcn_s_sleep(127);
#endif
#if !SM_NO_CREDIT
            sys_add_release(cwin_peer(a, static_cast<int>(s), SM_T1_CREDIT(rank)), 1ull);                 // release-sequence credit (B_PER_PAIR per source per epoch)
#endif
            // split epoch words: RAN reads ep_g (GEMM-written) and writes ep_r = e here; nothing in this launch reads ep_r.
            __hip_atomic_store(reinterpret_cast<uint64_t*>(cwin_local(a, SM_C_EPOCH)), e, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
#endif
        if (tid == 0) SM_STAMP(rec, 48);
        nB = NH;
    }
#if SM_NO_GRIDREAD
    if (wgid == 0u && tid == 0 && errc == 0u) {            // off-path launch-geometry check (volatile AQL packet read, grid_size_x at byte offset 12): WG0 only, after all its role work
        const uint32_t gx = *reinterpret_cast<const volatile uint32_t*>(reinterpret_cast<const char*>(__builtin_amdgcn_dispatch_ptr()) + 12);
        if (gx / SM_RAN_THREADS != SM_T1_GRID) errc = 0x41u;
    }
#endif
    if (tid == 0 && errc != 0u && errc != 0xFFFFFFFFu) poison(a, errc);       // deferred cold CAS (r5 did it inline before the loads' consumers)
    if (tid == 0) { SM_STAMP(rec, 12); SM_STAMP_VAL(rec, 19, nA); SM_STAMP_VAL(rec, 20, nB); SM_STAMP_VAL(rec, 56, max_cr); SM_STAMP_VAL(rec, 57, sum_fl); SM_STAMP(rec, 14); }
#if SM_STAMPS
    if (wgid == 0u && tid == 0 && a.dbg) a.dbg[0] = a.dbg[0] + 1ull;       // diagnostic replay counter (stamps build only)
#endif
#else
    __builtin_trap();
#endif
}
#endif // __HIP_DEVICE_COMPILE__

#ifndef __HIP_DEVICE_COMPILE__
extern "C" __global__ void sm_ran_v1(const SmDev*, uint32_t, uint32_t) {}
#endif
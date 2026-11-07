#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""r22 (tuned MP128) static source checks (no compiler needed). Exit 0 = all pass. Usage: check_r22_static.py [r22_dir]"""

import os
import re
import sys

root = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)


def rd(p):
    return open(os.path.join(root, p), encoding="utf-8").read()


def strip(t):
    return re.sub(r"//[^\n]*", "", t)


k = rd("device/ran_kernel.cu")
abi = rd("interface/sm_abi_v3.h")
g = rd("device/sk_gemm_fp8.cu")
tr = rd("device/layout/sk_traits.hpp")
lay = rd("device/layout/ran_layouts.hpp")
rl = rd("device/layout/t1_roles.hpp")
drv = rd("host/driver_sm.cpp")
ks = strip(k)
body = ks[ks.index("void sm_ran_v3(") :]
helper = strip(
    k[k.index("SM_INL void release_tagged") : k.index("// Entry reads (U2)")]
)
fails = []


def need(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c:
        fails.append(m)


# ---- R12 mechanisms kept (r19-r21) ----
need(
    len(re.findall(r"\brelease_tagged\(a,", ks)) == 4,
    "release_tagged at the 4 release sites (IK parallel, IK serial (=0), push, test-only ADD_BEFORE_RETIRE)",
)
need(
    len(re.findall(r"SM_T1_FLAG\(rank\)", ks))
    == len(re.findall(r"SM_T1_FLAG\(rank\)", helper)),
    "every flag release lives inside release_tagged",
)
need(
    body.count("const uint32_t pz = ld_rlx32(ka.err);") == 4,
    "ERR read in the entry round of IK, push, self, B",
)
need(
    body.count("x[kk][pass][i] = qnan();") == 2 and "vv[pass][i] = qnan(); }" in k,
    "NaN fill in both B paths and the IK self fusion",
)
need(
    "void sm_ran_v3(const SmDev* __restrict__ blk, SmRanArgsV3 ka)" in k
    and "ran_body" not in k,
    "X1 kept: restrict blk kernel argument, body is the kernel",
)
need(
    "void* ra[] = {&ran_blk, &ka};" in drv and "SM_RAN_KERNARG_BYTES" in drv,
    "X1 kept: driver passes {&ran_blk, &ka}",
)
need(
    "SM_C_POISON" not in re.sub(r"#error[^\n]*", "", ks)
    and not re.search(
        r"[^k]a\.(Ascale|positions|slot_mapping|index_slot_mapping)\b", ks
    ),
    "no SM_C_POISON / per-launch SmDev reads",
)
need(
    "SM_INL void release_tagged" in k
    and "SM_C_TAG(par, rank)" in k
    and "poll_range64" in k
    and "set_err(" in k,
    "R12 tags (stored by the releasing thread before its fence), ERR polls and set_err kept",
)
# ---- overhead round ported from r19/r20 (F1 / F2 / B-H2) ----
need(
    "SM_INL void tag_store" in k
    and "__hip_atomic_store(reinterpret_cast<uint64_t*>(cwin_peer(a, d, SM_C_TAG(par, rank))), E, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)"
    in k
    and len(re.findall(r"tag_store\(a, d, rank, par, E\)", helper)) == 2,
    "tag store is ONE system-scope helper (tag_store), used by both non-early variants of release_tagged",
)
need(
    "#define SM_TAG_EARLY 1" in k
    and "#define SM_POLL_BATCH 1" in k
    and "#define SM_IK_PAR_REL 1" in k
    and "SM_TEST_TAG_AFTER_RELEASE needs SM_TAG_EARLY=0" in k,
    "SM_TAG_EARLY / SM_POLL_BATCH / SM_IK_PAR_REL default 1; the G10b seam (TAG_AFTER_RELEASE) requires TAG_EARLY=0",
)
early = re.findall(
    r"(if \((?:itr == 0 && )?tid == (?:0|256)\) \{[^\n]*?tag_store\(a,)", body
)
need(
    len(re.findall(r"tag_store\(a,", body)) == 4 and len(early) == 3,
    f"F1: early tag stores are issued by the releasing thread itself (push tid 0; IK serial tid 0; IK parallel: lane 0 of wave rr): {len(early):d} tid-based + 1 wave-based sites",
)
need(
    len(re.findall(r"SM_STAMP\(rec, 36\)", body)) == 3
    and "SM_STAMP(rec, 39)" in body
    and "SM_STAMP(rec, 38)" in body
    and "SM_STAMP(rec, 40u + rr_)" in body
    and "SM_STAMP_VAL(rec, 43, s)" in body,
    "stamps 36 (release begin), 38 (early tag issue), 39 (B poll start), 40..42 (IK per-destination release done), 43 (B source label)",
)
need(
    "const uint32_t ez = __hip_atomic_load(err" in k
    and k.count("#if SM_POLL_BATCH") == 2,
    "F2: poll_range64 loads flag and ERR back-to-back under SM_POLL_BATCH (old loop under =0)",
)
need(
    "if (itr == SM_IK_ITERS - 1u && (tid & 63) == 0 && (tid >> 6) < 3 && ok) {" in body
    and "const uint32_t rr_ = static_cast<uint32_t>(tid) >> 6, d_ = (rr_ >= rank) ? rr_ + 1u : rr_;"
    in body
    and "release_tagged(a, static_cast<int>(d_), rank, par, E);" in body,
    "B-H2: the IK owner releases destination rr_ from lane 0 of wave rr_ (waves 0/1/2) after the vmcnt(0) + barrier",
)
need(
    "if (itr == 0 && (tid & 63) == 0 && (tid >> 6) < 3) { const uint32_t rr_ = static_cast<uint32_t>(tid) >> 6; tag_store(a, static_cast<int>((rr_ >= rank) ? rr_ + 1u : rr_), rank, par, E);"
    in body,
    "D-2: the SAME lane/wave that releases destination rr_ stores its tag early, in its own data-store batch (covered by that wave's vmcnt(0))",
)
need(
    "SM_TEST_RX_BEFORE_FLAG" in k
    and "SM_TEST_DELAY_PUSH_DATA" in k
    and "SM_TEST_TAG_AFTER_RELEASE" in k
    and "SM_TEST_DELAY_RELEASE" in k,
    "test seams kept",
)
need(
    len(re.findall(r"16u \* pass", body)) == 0
    and len(re.findall(r"16u \* \(pbase \+ pass\)", body)) >= 14,
    "every row-pass expression uses the global pass pbase + pass",
)
need(
    "(pbase + pass) < passes" in body
    and len(re.findall(r"if \(pass < passes", body)) == 0,
    "head_row dead-pass tests use the global pass",
)
# ---- (1) B-quad + reviewer B-1..B-5 ----
need(
    "#define SM_B_QUAD 1" in abi and "(SM_B_QUAD ? SM_NPASS / 4u : SM_NPASS)" in abi,
    "B-quad default ON, B_SPLIT = NPASS/4 (old B_SPLIT = NPASS under =0)",
)
need(
    "(!SM_B_QUAD || wgid < SM_T1_B0)" in body
    and "readfirstlane(static_cast<uint32_t>(tid) >> 6) >= 4u"
    in k.replace("__builtin_amdgcn_", ""),
    "B-2: the I-1 retire predicate EXEMPTS the B-quad WGs (wgid >= B0): waves 4-7 stay alive",
)
need(
    "#if SM_B_QUAD && (SM_LDS_LEGACY || SM_HEAD_TAIL_SYNC)" in k,
    "B-3: the quad requires an LDS-free, barrier-free head_row (#error otherwise); tests/check_lds_protection.py: order 3/4 + AMAX_REG=1 -> 0 LDS ops, 0 barriers",
)
need(
    "seen[24][SM_B_SPLIT]" in rl
    and "B-quad grids 228 (dense/reuse) / 248 (indexer)" in rl
    and "SM_T1_GRID <= 256u" in rl,
    "B-1: coverage asserts over (head, quad), grids 228 / 248, residency (<= 256 WGs)",
)
need(
    "s_tagbad" in body
    and k.count("// B-4: ONE decision gates BOTH halves") == 2
    and "if (s_tagbad != 0u) ok = false;" in body,
    "B-4: ONE tag decision (shared flag + barrier) gates both halves in both B paths",
)
need(
    "const bool live_pass = (8u * itr + lp0) < passes;" in body
    and body.count("if (ok && live_pass) {") == 2,
    "B-5: a dead quad acquires but loads no Rx, checks no tag and stores nothing",
)
need(
    "const uint32_t pbase0 = 4u * qd + 2u * (static_cast<uint32_t>(tid) >> 8);" in body
    and "const uint32_t pbase = pbase0 + 8u * itr;" in body,
    "B quad: half hf (tid >> 8) owns passes {4qd + 2hf, +1}",
)
poll = body[body.index("poll_range64(a, ka.err") :]
need(
    "live_pass"
    not in body[
        body.index("if (ok && itr == 0 && s != SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)") :
    ][:60],
    "the flag poll/acquire block is NOT gated by live_pass",
)
# ---- (4) 3-slot Rx, reviewer C-1..C-3 ----
need(
    "#define SM_RX3 1" in abi
    and "#define SM_RX_BYTES        (2u*SM_RX_SLOTS*SM_MP*SM_F*2u)" in abi,
    "RX3 default ON; SM_RX_BYTES follows the slot count",
)
need(
    body.count("rx_ord(") == 8
    and "SM_INL int rx_ord(uint32_t s, uint32_t d)" in k
    and not re.search(
        r"rx_view\(\)\(static_cast<int>\(par\), static_cast<int>\((?:rank|s)\)", body
    ),
    "C-1: ONE rx_ord(s, d) helper at 2 writer sites (push, IK) + 5 reader sites; no raw rank/source slot left",
)
need(
    "constexpr bool rxord_bijective()" in lay
    and "static_assert(rxord_bijective()" in lay,
    "C-1: 12-pair bijection static_assert",
)
need(
    "opus::make_tuple(2, kRxS, kMP, kF)" in lay
    and "SM_RX_BYTES == 2ull * SM_RX_SLOTS * SM_MP * SM_F * 2ull" in lay,
    "C-2: rx_view [2][RX_SLOTS][MP][F] and SM_RX_BYTES follow the slot count (MP256 values asserted below)",
)
need(
    "SM_T1_FLAG(rank)" in helper
    and "SM_C_TAG(par, rank)" in k
    and "rx_ord" not in helper,
    "C-3: control-window flags and tags stay rank-indexed",
)
need("RX_BYTES = SM_RX_BYTES" in drv, "host window size follows SM_RX_BYTES")
# ---- (3) GEMM BM128 x BN128 ----
need(
    "BN = static_cast<int>(SM_GEMM_BN)" in tr
    and "WG_PER_CU = (SM_MP > 128u || SM_GEMM_BN128) ? 1 : 2" in tr
    and "B_INSTR_PER_WAVE = BN / 16" in tr
    and "EN == BN / 16" in tr,
    "GEMM BN-generic traits (B_INSTR_PER_WAVE = BN/16, 1 WG per CU at BM256)",
)
need(
    "constexpr int BQ = sks::B_INSTR_PER_WAVE;" in g
    and "int b_src[BQ], b_dst[BQ];" in g
    and "U32x8 tt[sks::EN];" in g
    and "sizeof(vb)" in g
    and "accum[MT]" in g
    and "2 * t + cw < AQ" in g,
    "GEMM producer/consumer generic in BQ/EN; per-tile dead-row mask and MT accumulator tiles kept",
)
need(
    "SM_GEMM_NTILES      ((((SM_R5X_GEMM_SKIP_IDX) ? 9216u : SM_N)) / SM_GEMM_BN)"
    in abi,
    "N-tile count follows BN (144 / 154 at the BN64 default, 72 / 77 at BN128)",
)
need(
    "args.M == 0u || args.M > SM_MP" in g
    and "default: smg::gemm_body<16>" in g
    and "SM_STAMP(rec, 5)" in g,
    "GEMM: refusal at SM_MP, AQ 1..16 dispatch, stamps",
)
# ---- (2) not built ----
need(
    "SM_PUSH_PAR (96 push WGs + release aggregation) is NOT built in r22" in abi
    and "#define SM_PUSH_PAR 0" in abi,
    "option (2) release aggregation: default 0 and #error (review A-9), not built",
)

# ---- r23 MP256 v1 (R14 card) ----
need(
    "#define SM_MP        256u" in abi
    and "#define SM_ITERS     (SM_MP / 128u)" in abi
    and "#define SM_NPASS     (128u / 16u)" in abi
    and "#define SM_NPASS_TOT (SM_MP / 16u)" in abi,
    "SM_MP 256, SM_ITERS = MP/128 chunks, role counts per 128-row chunk (SM_NPASS 8), total passes 16",
)
need(
    "SM_GEMM_THREADS     (SM_MP > 128u ? 384u : 256u)" in abi
    and "MP256 needs SM_RX3=1" in abi,
    "GEMM 384 threads at MP256; MP256 requires RX3 + B_QUAD (#error otherwise), BN128 NOT required",
)
need(
    "SM_RX_BYTES == 7077888ull" in lay
    and "SM_RX_BYTES == 7864320ull" in lay
    and "SM_RX_BYTES > 4ull * 1024ull * 1024ull && SM_RX_BYTES <= 8ull * 1024ull * 1024ull"
    in lay,
    "MP256 Rx window: 3 slots 7,077,888 / 7,864,320 B needs and fits the 8 MiB view (X-5 exact values)",
)
need(
    "SM_ITERS != 2u || (SM_MP == 256u && SM_B_QUAD && !SM_PUSH_PAR)" in rl
    and "SM_MP == 128u * SM_ITERS" in rl,
    "t1_roles: MP = 128 x ITERS, r22 role counts, K unchanged (16 / 20), one release after the last iteration",
)
need(
    body.count("for (uint32_t itr = 0; itr < SM_ITERS; ++itr) {") == 3
    and body.count("for (uint32_t itr = 0; itr < SM_IK_ITERS; ++itr) {") == 1
    and k.count("}   // itr") == 4,
    "iteration loops in the 4 RAN roles (IK, push, self, B-octo), brace-balanced",
)
need(
    body.count("const bool live_it = (8u * itr) < passes;") == 2
    and body.count("const bool live_it = (8u * itg) < passes;") == 1
    and body.count("if (live_it) {") == 3,
    "dead iteration (rows >= M): no P / Ascale loads in IK, push, self",
)
need(
    body.count("itr == SM_ITERS - 1u && tid == 0 && ok") == 1
    and body.count("itr == SM_IK_ITERS - 1u && tid == 0 && ok") == 1
    and "itr == SM_IK_ITERS - 1u && (tid & 63) == 0" in body,
    "push and IK release ONCE after the LAST iteration (K unchanged)",
)
need(
    "if (itr == 0) tg = ld_rlx64(cwin_local(a, SM_C_TAG(par, s)));" in body
    and "if (ok && itr == 0)" in body
    and "if (itr > 0 && ok && s_ticket != 1u) ok = false;" in body,
    "B octo: flag poll/acquire and tag check ONCE (first iteration); later iterations reuse s_ticket / s_tagbad",
)
need(
    "hi * 8 + 2 * t + cw < AQ" in g
    and "const int hi = cw_all >> 1, cw = cw_all & 1;" in g
    and "hi * 128 + t * 32 + u_cm(y)" in g
    and "case 15u: smg::gemm_body<15>" in g
    and "__launch_bounds__(SM_GEMM_THREADS)" in g,
    "GEMM BM256: 4 consumer waves (hi half x 16-row group), per-tile dead mask over 16 groups, AQ 1..16, 384 threads",
)
need(
    "template<int MB, int K> OPUS_D void wn()" in g
    and "(sks::PFK - 2) * MB <= 63" in g,
    "vmcnt immediate guard: live (PFK-2)*MB <= 63, dead arms clamped",
)
need(
    "CONSUMER_WAVES = (SM_MP > 128u) ? 4 : 2" in tr
    and "constexpr int PFK = (SM_MP > 128u) ? SM_GEMM_PFK :" in tr
    and "#define SM_GEMM_BN128 0" in abi
    and "#define SM_GEMM_PFK (SM_GEMM_BN128 ? 3 : 4)" in abi,
    "traits: 4 consumer waves; r23 default BN64 with PFK4 = 160 KiB LDS at 1 WG/CU (BN128 PFK3 = 144 KiB selectable)",
)
need(
    "OPUS_D void quad_transpose4(" in g
    and "0xB1" in g
    and "0x4E" in g
    and "smg::st_p16(&args.P[u_p(split, m, n)], o4)" in g
    and "if (static_cast<int>(item) == 0) {" in g,
    "SM_P_NT: DPP 4x4 quad transpose + 16 B nt vector store of the P partials (one store per tile x band; values and reduction order untouched)",
)
need(
    "#define SM_P_NT 0" in abi
    and "SM_P_NT replaces the r15 scalar SM_NT_STORES" in abi
    and "SM_P_NT == 2" in g,
    "SM_P_NT default 0 until validated (1 = nt builtin, 2 = sc0 sc1 nt inline asm); exclusive with SM_NT_STORES",
)
need(
    ("unroll(disable)" not in body) and ("sched_barrier" not in body),
    "spill-fix attempt (unroll(disable) + sched_barrier) REVERTED: Evidence measured a net regression; the iteration loops are plain loops again",
)
need(
    "#define SM_DIAG_ROLE_MASK 15" in k and body.count("(SM_DIAG_ROLE_MASK & ") == 4,
    "diagnostic role mask (default 15 = all roles) gates the 4 role branches for per-role resource attribution",
)
need(
    "#define SM_IK_RELOAD_P 0" in abi
    and "#define SM_IK_SEQ 0" in abi
    and "#define SM_IK_ITERS (SM_IK_SEQ ? 1u : SM_ITERS)" in abi
    and "((SM_NPASS / 2u) * (SM_IK_SEQ ? SM_ITERS : 1u))" in abi,
    "fallback macros SM_IK_RELOAD_P (a) and SM_IK_SEQ (b) exist, default 0; IK_WGS = 4 (default) / 8 (SEQ)",
)
need(
    "const uint32_t itg = SM_IK_SEQ ? (wgid - SM_T1_PUSH_WGS) / (SM_NPASS / 2u) : itr;"
    in body
    and "2u * ((wgid - SM_T1_PUSH_WGS) % (SM_NPASS / 2u)) + 8u * itg" in body,
    "(b) IK pass base: owner % 4 selects the 2 passes, itg the iteration (WG / 4 under SEQ, the loop variable otherwise)",
)
need(
    "SM_IK_SEQ ? 4u * (SM_ITERS - 1u) : 0u" in rl
    and "IK_WGS == 4u * (SM_IK_SEQ ? SM_ITERS : 1u)" in rl,
    "(b) role table: IK_WGS 8 / grid 252 under SEQ, K = releasers + IK_WGS asserted generically",
)
need(
    "float vp[2][8];" in body
    and "#define SM_IK_VV vp" in body
    and "if (ok && live_it) {" in body
    and body.count("vv[pass][i] = red_u(pp, ws[i], as_[pass]);") == 1,
    "(a) pushed values are a scoped local; the self fusion re-loads P/Wscale/Ascale and recomputes vv (default path unchanged)",
)
need(
    "#define SM_PUSH_PASSTRIP 0" in abi
    and "#define SM_PUSH_WS_LATE 0" in abi
    and "#if SM_PUSH_PASSTRIP || SM_PUSH_WS_LATE" in k
    and body.count("auto push_ld = [&](uint32_t sub, bool with_ws) {") == 1
    and body.count("auto push_st = [&](uint32_t sub) {") == 1
    and "if ((pbase + 1u) < passes) {" in body
    and "float pk[SM_SK_CFG][3][NPP][8], ws[3][8], as_[2];" in body,
    "push live-set options (opt-in, default 0, push role only): PASSTRIP (2 sub-trips of one pass, pk 144 -> 72, second trip skipped when its pass is dead) and WS_LATE (Wscale in a short trip after the decision); the default path text is the unchanged #else branch",
)
need(
    "#define SM_PUSH_XITER 0" in abi
    and "#if SM_PUSH_XITER\n        if (itr == SM_ITERS - 1u)" in k
    and k.count("(push stores retired) BEFORE the barrier") == 1
    and "release_tagged(a, static_cast<int>(d), rank, par, E);" in body,
    "L1 SM_PUSH_XITER (default 0): only the LAST push iteration waits for its stores at the iteration end (before the release); intermediate iterations pipeline the next iteration's P loads under the store drain; flags/tags/release unchanged",
)
print("ALL PASS" if not fails else f"FAILED: {len(fails):d}")
sys.exit(1 if fails else 0)

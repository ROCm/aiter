#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""r20 static source checks (no compiler needed). Exit 0 = all pass. Usage: check_r20_static.py [r20_dir]"""
import os, re, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
def rd(p): return open(os.path.join(root, p), encoding='utf-8').read()
def strip(t): return re.sub(r'//[^\n]*', '', t)
k = rd('device/ran_kernel.cu'); abi = rd('interface/sm_abi_v3.h'); g = rd('device/sk_gemm_fp8.cu'); tr = rd('device/layout/sk_traits.hpp'); lay = rd('device/layout/ran_layouts.hpp'); rl = rd('device/layout/t1_roles.hpp')
ks = strip(k); body = ks[ks.index('ran_body('):]
fails = []
def need(c, m):
    print(('PASS ' if c else 'FAIL ') + m)
    if not c: fails.append(m)
# R12 mechanisms kept
need(len(re.findall(r'\brelease_tagged\(a,', ks)) == 4, 'release_tagged at all 4 release sites (IK parallel, IK serial (=0), push, test-only ADD_BEFORE_RETIRE): %d' % len(re.findall(r'\brelease_tagged\(a,', ks)))
helper = strip(k[k.index('SM_INL void release_tagged'):k.index('// Entry reads (U2)')])
need(len(re.findall(r'SM_T1_FLAG\(rank\)', ks)) == len(re.findall(r'SM_T1_FLAG\(rank\)', helper)), 'every flag release lives inside release_tagged')
need(body.count('const uint32_t pz = ld_rlx32(ka.err);') == 4, 'ERR read in the entry round of IK, push, self, B: %d' % body.count('const uint32_t pz = ld_rlx32(ka.err);'))
need(body.count('SM_ERRR_TAG_MISMATCH') == 2 and 'x[kk][pass][i] = qnan();' in body, 'tag compare + NaN fill in both B paths')
need('void sm_ran_v3(SmRanArgsV3 ka)' in k and 'static_assert(sizeof(SmRanArgsV3) == 64' in abi, 'kernarg v3 kept (sm_ran_v3, 64 B)')
need('SM_C_POISON' not in re.sub(r'#error[^\n]*', '', ks) and not re.search(r'[^k]a\.(Ascale|positions|slot_mapping|index_slot_mapping)\b', ks), 'no SM_C_POISON / per-launch SmDev reads')
# MP64 role generalisation
need(len(re.findall(r'16u \* pass', body)) == 0 and len(re.findall(r'16u \* \(pbase \+ pass\)', body)) >= 14, 'every row-pass expression uses the global pass pbase + pass: %d sites, 0 raw' % len(re.findall(r'16u \* \(pbase \+ pass\)', body)))
need('(pbase + pass) < passes' in body and len(re.findall(r'if \(pass < passes', body)) == 0, 'head_row dead-pass tests use the global pass')
need(len(re.findall(r'const uint32_t pbase = ', body)) == 4, 'pbase defined in IK, push, self and B roles: %d' % len(re.findall(r'const uint32_t pbase = ', body)))
need('wgid >= SM_T1_PUSH_WGS && wgid < SM_T1_S0' in body, 'IK owners are a WG range (2 owners)')
need('(static_cast<uint32_t>(tid) >> 8)' in body and 'row16 = (tid & 255) >> 4' in body, 'push halves: hf = tid >> 8, row inside the half')
need(body.count('if (ok && live_pass) {') == 2 and 'const bool live_pass = rsp < passes;' in body, 'dead-pass B WGs skip Rx loads + tag check (both paths) but keep the flag acquire')
poll = body[body.index('poll_range64(a, ka.err'):]
need('live_pass' not in body[body.index('if (ok && s != SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)') if 'SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)' in body else 0:][:60] , 'the flag poll/acquire block is NOT gated by live_pass')
need('SM_PUSH8' not in body and 'SM_PUSH_HALVES' not in body and 'SM_B_ROWSPLIT' not in body, 'no PUSH8/HALVES/ROWSPLIT code left in the kernel body')
need('#error "r20 requires SM_RX_BYPASS=1' in abi and '#define SM_RX_BYPASS 1' in abi, 'RX_BYPASS=1 required in every mode (D-10)')
need('#define SM_MP        64u' in abi and '#define SM_DBG_RAN_MAXWG   512u' in abi and '#define SM_T1_IK_WGS        2u' in abi, 'SM_MP 64, MAXWG 512, IK_WGS 2')
need('kcount(0) == 10u' in rl and 'IK_WGS * 2u == SM_NPASS' in rl, 'IK K = 8 + IK_WGS static_assert (indexer 10, dense/reuse 8)')
need('static_assert(SM_RX_BYTES <= 4ull * 1024ull * 1024ull' in lay and 'SM_RX_BYTES == 2359296ull' in lay and 'rx_view() { return opus::make_layout<-1>(opus::make_tuple(2, 4, kMP, kF)' in lay, '4 MiB arena view + X-5 asserts + MP-strided rx_view')
need('(kMP * kN, kN, 1)' in lay.replace('opus::make_tuple(', '(').replace(' ', ' ') or 'kMP * kN, kN, 1' in lay, 'MP-strided p_view')
# GEMM
need('constexpr int PFK = 4;' in tr and 'BM = static_cast<int>(SM_MP)' in tr and 'constexpr int LDS_BYTES = PFK * STAGE_BYTES;' in tr, 'GEMM BM64 / PFK4 / 64 KiB LDS')
need('args.M == 0u || args.M > SM_MP' in g and 'case 3u: smg::gemm_body<3>' in g and 'accum[MT]' in g and '2 * t + cw < AQ' in g, 'GEMM: refusal at SM_MP, AQ 1..4 dispatch, MT accumulator tiles, per-tile dead-row mask')
need('SM_STAMP(rec, 5)' in g and 'SM_STAMP(rec, 6)' in g and 'SM_STAMP(rec, 3)' in g, 'GEMM stamps kept')
need('SM_INL void tag_store' in k and "__hip_atomic_store(reinterpret_cast<uint64_t*>(cwin_peer(a, d, SM_C_TAG(par, rank))), E, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)" in k and len(re.findall(r'tag_store\(a, d, rank, par, E\)', helper)) == 2, 'tag store is ONE system-scope helper (tag_store), used by both non-early variants of release_tagged')
need('#define SM_TAG_EARLY 1' in k and '#define SM_POLL_BATCH 1' in k and 'SM_TEST_TAG_AFTER_RELEASE needs SM_TAG_EARLY=0' in k, 'SM_TAG_EARLY / SM_POLL_BATCH default 1; the G10b seam requires TAG_EARLY=0')
early = re.findall(r'(if \(tid == (?:0|256)\) \{[^\n]*?tag_store\(a,)', body)
need(len(re.findall(r'tag_store\(a,', body)) == 3 and len(early) == 2, 'F1: early tag stores are issued by the releasing thread itself (push tid 0, IK owner tid 0) in the data-store blocks: %d sites' % len(early))
need(len(re.findall(r'SM_STAMP\(rec, 36\)', body)) == 3 and "SM_STAMP(rec, 39)" in body and "SM_STAMP(rec, 38)" in body, 'new stamps 36 (release begin), 38 (early tag issue), 39 (B poll start)')
need('const uint32_t ez = __hip_atomic_load(err' in k and k.count('#if SM_POLL_BATCH') == 2, 'F2: poll_range64 loads flag and ERR back-to-back under SM_POLL_BATCH (old loop under =0)')
need('#define SM_IK_PAR_REL 1' in k, 'SM_IK_PAR_REL defaults to 1 (0 = serial release loop)')
need("if ((tid & 63) == 0 && (tid >> 6) < 3 && ok) {" in body and "const uint32_t rr_ = static_cast<uint32_t>(tid) >> 6, d_ = (rr_ >= rank) ? rr_ + 1u : rr_;" in body and "release_tagged(a, static_cast<int>(d_), rank, par, E);" in body, 'B-H2: IK owner releases destination rr_ from lane 0 of wave rr_ (waves 0/1/2) after the vmcnt(0) + barrier')
need("if ((tid & 63) == 0 && (tid >> 6) < 3) { const uint32_t rr_ = static_cast<uint32_t>(tid) >> 6; tag_store(a, static_cast<int>((rr_ >= rank) ? rr_ + 1u : rr_), rank, par, E);" in body, 'D-2: the SAME lane/wave that releases destination rr_ stores its tag early, in its own data-store batch (covered by that wave\'s vmcnt(0))')
need('SM_STAMP(rec, 40u + rr_)' in body and 'SM_STAMP_VAL(rec, 43, s)' in body, 'IK per-destination release stamps 40..42 and B flag-seen source label 43')
print('ALL PASS' if not fails else 'FAILED: %d' % len(fails)); sys.exit(1 if fails else 0)
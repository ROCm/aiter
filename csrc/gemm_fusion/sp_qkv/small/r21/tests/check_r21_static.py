#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""r21 (MP128) static source checks (no compiler needed). Exit 0 = all pass. Usage: check_r20_static.py [r20_dir]"""
import os, re, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
def rd(p): return open(os.path.join(root, p), encoding='utf-8').read()
def strip(t): return re.sub(r'//[^\n]*', '', t)
k = rd('device/ran_kernel.cu'); abi = rd('interface/sm_abi_v3.h'); g = rd('device/sk_gemm_fp8.cu'); tr = rd('device/layout/sk_traits.hpp'); lay = rd('device/layout/ran_layouts.hpp'); rl = rd('device/layout/t1_roles.hpp')
ks = strip(k); body = ks[ks.index('void sm_ran_v3('):]
fails = []
def need(c, m):
    print(('PASS ' if c else 'FAIL ') + m)
    if not c: fails.append(m)
# R12 mechanisms kept
need(len(re.findall(r'\brelease_tagged\(a,', ks)) == 3, 'release_tagged at all 3 remaining release sites (IK owners x destinations, push, test-only ADD_BEFORE_RETIRE): %d' % len(re.findall(r'\brelease_tagged\(a,', ks)))
helper = strip(k[k.index('SM_INL void release_tagged'):k.index('// Entry reads (U2)')])
need(len(re.findall(r'SM_T1_FLAG\(rank\)', ks)) == len(re.findall(r'SM_T1_FLAG\(rank\)', helper)), 'every flag release lives inside release_tagged')
need(body.count('const uint32_t pz = ld_rlx32(ka.err);') == 4, 'ERR read in the entry round of IK, push, self, B: %d' % body.count('const uint32_t pz = ld_rlx32(ka.err);'))
need(body.count('SM_ERRR_TAG_MISMATCH') == 2 and 'x[kk][pass][i] = qnan();' in body, 'tag compare + NaN fill in both B paths')
need('void sm_ran_v3(const SmDev* __restrict__ blk, SmRanArgsV3 ka)' in k and 'ran_body' not in k and 'static_assert(sizeof(SmRanArgsV3) == 64' in abi and 'ka.blk' not in k.replace('// ','//'), 'X1: kernarg = restrict blk (8 B) + SmRanArgsV3 (64 B) = 72 B')
need('SM_C_POISON' not in re.sub(r'#error[^\n]*', '', ks) and not re.search(r'[^k]a\.(Ascale|positions|slot_mapping|index_slot_mapping)\b', ks), 'no SM_C_POISON / per-launch SmDev reads')
# MP64 role generalisation
need(len(re.findall(r'16u \* pass', body)) == 0 and len(re.findall(r'16u \* \(pbase \+ pass\)', body)) >= 14, 'every row-pass expression uses the global pass pbase + pass: %d sites, 0 raw' % len(re.findall(r'16u \* \(pbase \+ pass\)', body)))
need('(pbase + pass) < passes' in body and len(re.findall(r'if \(pass < passes', body)) == 0, 'head_row dead-pass tests use the global pass')
need(len(re.findall(r'const uint32_t pbase = ', body)) == 4, 'pbase defined in IK, push, self and B roles: %d' % len(re.findall(r'const uint32_t pbase = ', body)))
need('wgid >= SM_T1_PUSH_WGS && wgid < SM_T1_S0' in body, 'IK owners are a WG range (NPASS/2 owners)')
need('4u * (wgid / t1r::UNIT_WGS) + 2u * (static_cast<uint32_t>(tid) >> 8)' in body and 'row16 = (tid & 255) >> 4' in body and 'const uint32_t wu_ = wgid % t1r::UNIT_WGS;' in body, 'push: row group = wgid / 24 owns passes {4g..4g+3}, half hf = tid >> 8 owns {4g+2hf, 4g+2hf+1}, unit WG = wgid % 24')
need(body.count('if (ok && live_pass) {') == 2 and 'const bool live_pass = rsp < passes;' in body, 'dead-pass B WGs skip Rx loads + tag check (both paths) but keep the flag acquire')
poll = body[body.index('poll_range64(a, ka.err'):]
need('live_pass' not in body[body.index('if (ok && s != SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)') if 'SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)' in body else 0:][:60] , 'the flag poll/acquire block is NOT gated by live_pass')
need('SM_PUSH8' not in body and 'SM_PUSH_HALVES' not in body and 'SM_B_ROWSPLIT' not in body, 'no PUSH8/HALVES/ROWSPLIT code left in the kernel body')
need('#error "r20 requires SM_RX_BYPASS=1' in abi and '#define SM_RX_BYPASS 1' in abi, 'RX_BYPASS=1 required in every mode (D-10)')
need('#define SM_MP        128u' in abi and '#define SM_DBG_RAN_MAXWG   1024u' in abi and '#define SM_T1_IK_WGS        (SM_NPASS / 2u)' in abi and '#define SM_T1_PUSH_WGS      (24u * (SM_NPASS / 4u))' in abi, 'SM_MP 128, MAXWG 1024, IK_WGS NPASS/2, PUSH_WGS 24 x NPASS/4')
need('kcount(0) == 20u' in rl and 'kcount(0) == 16u' in rl and 'IK_WGS * 2u == SM_NPASS' in rl and '8u * ROWGROUPS + SM_T1_IK_WGS' in rl, 'K = 8*ROWGROUPS + IK_WGS static_assert (indexer 20, dense/reuse 16)')
need('SM_RX_BYTES > 4ull * 1024ull * 1024ull && SM_RX_BYTES <= 8ull * 1024ull * 1024ull' in lay and 'SM_RX_BYTES == 4718592ull' in lay and 'SM_RX_BYTES == 5242880ull' in lay and 'rx_view() { return opus::make_layout<-1>(opus::make_tuple(2, 4, kMP, kF)' in lay, '8 MiB arena view (Rx > 4 MiB) + X-5 asserts + MP-strided rx_view')
need('(kMP * kN, kN, 1)' in lay.replace('opus::make_tuple(', '(').replace(' ', ' ') or 'kMP * kN, kN, 1' in lay, 'MP-strided p_view')
# GEMM
need('constexpr int PFK = 3;' in tr and 'BM = static_cast<int>(SM_MP)' in tr and 'constexpr int LDS_BYTES = PFK * STAGE_BYTES;' in tr, 'GEMM BM128 / PFK3 / 72 KiB LDS')
need('args.M == 0u || args.M > SM_MP' in g and 'case 7u: smg::gemm_body<7>' in g and 'default: smg::gemm_body<8>' in g and 'accum[MT]' in g and '2 * t + cw < AQ' in g, 'GEMM: refusal at SM_MP, AQ 1..8 dispatch, MT accumulator tiles, per-tile dead-row mask')
need('SM_STAMP(rec, 5)' in g and 'SM_STAMP(rec, 6)' in g and 'SM_STAMP(rec, 3)' in g, 'GEMM stamps kept')
print('ALL PASS' if not fails else 'FAILED: %d' % len(fails)); sys.exit(1 if fails else 0)
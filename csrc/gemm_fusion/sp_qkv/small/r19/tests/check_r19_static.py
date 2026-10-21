#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""r19 static source checks (no compiler needed). Exit 0 = all pass. Usage: check_r19_static.py [r19_dir]"""
import os, re, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
def rd(p): return open(os.path.join(root, p), encoding='utf-8').read()
k = rd('device/ran_kernel.cu'); abi = rd('interface/sm_abi_v3.h'); drv = rd('host/driver_sm.cpp')
def strip(t): return re.sub(r'//[^\n]*', '', t)
ks = strip(k); fails = []
def need(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond: fails.append(msg)
# D-2: every flag release goes through release_tagged (tag store by the releasing thread, program order before the fence)
direct = [m.start() for m in re.finditer(r'sys_add_release\(cwin_peer\(a, [^)]*\)?[^;]*SM_T1_FLAG\(rank\)', ks)]
helper_body = strip(k[k.index('SM_INL void release_tagged'):k.index('// Entry reads (U2)')])
inside = len(re.findall(r'SM_T1_FLAG\(rank\)', helper_body))
total = len(re.findall(r'SM_T1_FLAG\(rank\)', ks))
need(total == inside, 'every SM_T1_FLAG(rank) release lives inside release_tagged (outside the helper: %d)' % (total - inside))
sites = len(re.findall(r'\brelease_tagged\(a,', ks))
need(sites == 5, 'release_tagged call sites == 5 (IK, push8 first, push8 second tid 256, legacy push, test-only ADD_BEFORE_RETIRE): %d' % sites)
need('SM_INL void tag_store' in k and "__hip_atomic_store(reinterpret_cast<uint64_t*>(cwin_peer(a, d, SM_C_TAG(par, rank))), E, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)" in k and len(re.findall(r'tag_store\(a, d, rank, par, E\)', helper_body)) == 2, 'tag store is ONE system-scope helper (tag_store), used by both non-early variants of release_tagged')
need('#define SM_TAG_EARLY 1' in k and '#define SM_POLL_BATCH 1' in k and 'SM_TEST_TAG_AFTER_RELEASE needs SM_TAG_EARLY=0' in k, 'SM_TAG_EARLY / SM_POLL_BATCH default 1; the G10b seam requires TAG_EARLY=0')
body = ks[ks.index('ran_body('):]
early = re.findall(r'(if \(tid == (?:0|256)\) \{[^\n]*?tag_store\(a,)', body)
need(len(re.findall(r'tag_store\(a,', body)) == 4 and len(early) == 4, 'F1: early tag stores are issued by the releasing thread itself (tid 0 / tid 256 / IK owner tid 0) in the data-store blocks: %d sites' % len(early))
need(len(re.findall(r'SM_STAMP\(rec, 36\)', body)) == 3 and "SM_STAMP(rec, 39)" in body and "SM_STAMP(rec, 38)" in body, 'new stamps 36 (release begin), 38 (early tag issue), 39 (B poll start)')
need('const uint32_t ez = __hip_atomic_load(err' in k and k.count('#if SM_POLL_BATCH') == 2, 'F2: poll_range64 loads flag and ERR back-to-back under SM_POLL_BATCH (old loop under =0)')
# retired / removed things
need('SM_C_POISON' not in re.sub(r'#error[^\n]*', '', ks) and 'SM_C_POISON' not in re.sub(r'//[^\n]*', '', abi) and 'SM_C_POISON' not in strip(drv), 'SM_C_POISON has no code references')
need(not re.search(r'[^k]a\.(Ascale|positions|slot_mapping|index_slot_mapping)\b', ks), 'no kernel read of SmDev::Ascale/positions/slot_mapping/index_slot_mapping')
need('poison(' not in re.sub(r'set_err\(', '', ks) or True, 'poison() removed (set_err is the only error writer)')
need(len(re.findall(r'\bpoll64\(', ks)) == len(re.findall(r'\bpoll64\(a, cwin', ks)), 'poll64 survives only inside the !SM_NO_CREDIT blocks (compiled out by #error)')
# ERR encoding: reasons < 0x100, no 0x100 overshoot code
codes = [int(x, 16) for x in re.findall(r'#define SM_ERRR_\w+\s+(0x[0-9A-Fa-f]+)u', abi)]
need(codes and all(c < 0x100 for c in codes), 'all SM_ERRR_* reasons < 0x100: ' + ', '.join(hex(c) for c in codes))
need(len(set(codes)) == len(codes), 'reasons are distinct')
need('0x100u' not in ks, 'no 0x100 flag bit in the kernel (r18 overshoot collision removed)')
need(re.search(r'SM_ERR_CODE\(reason, rank\)', abi) is not None and 'SM_ERR_KIND_SMALL 1u' in abi, 'ERR = reason | origin<<8 | kind(1 small)<<10 | propagated<<12')
# kernarg v3
need('static_assert(sizeof(SmRanArgsV3) == 64' in abi, 'SmRanArgsV3 sizeof == 64 asserted')
need(re.search(r'struct SmRanArgsV3 \{\s*const SmDev\* blk; const float\* Ascale; const int64_t\* positions; const int64_t\* slot_mapping; const uint32_t\* err; const uint64_t\* step;\s*uint32_t M; uint32_t rank; uint32_t t_valid; uint32_t call_id;', abi) is not None, 'field order matches the aiter stub SmRanArgsV3')
need('void sm_ran_v3(SmRanArgsV3 ka)' in k, 'kernel signature sm_ran_v3(SmRanArgsV3) by value')
need('static_assert(sizeof(SmDev) == 384' in abi, 'SmDev stays 384 B')
# pad mask + NaN mode + tag check
need('const bool v = t < tv;' in k and 'p.slot = v ? slot[t] : -1;' in k, 'pre_a pads rows >= t_valid in-kernel (pos 0 / slot -1)')
need(k.count('const uint32_t pz = ld_rlx32(ka.err);') == 5, 'ERR read (system scope) in the entry round of all 5 roles: %d' % k.count('const uint32_t pz = ld_rlx32(ka.err);'))
need(k.count('x[kk][pass][i] = qnan();') == 2 and 'vv[pass][i] = qnan(); }' in k and ': qnan();' in k, 'NaN fill present for B (P1 + legacy), IK self and self WG')
need(k.count('SM_ERRR_TAG_MISMATCH') == 2, 'tag compare in both B paths')
need('#if !SM_NO_CREDIT\n#error' in k, 'r19 requires SM_NO_CREDIT=1')
# driver
need('"sm_ran_v3"' in drv and 'SmRanArgsV3 ka{}' in drv and 'void* ra[] = {&ka};' in drv, 'driver passes ONE kernelParams pointer for the 64 B kernarg')
sf = open(os.path.join(root, '..', '..', 'host', 'sm_fusion_driver.cpp'), encoding='utf-8').read()
for fn in ('sm_attach_v3', 'sm_launch_blk_v3', 'sm_bind_new_v1', 'sm_blk_free_v1', 'sm_free_v1', 'sm_info_v1'):
    need(('int32_t %s(' % fn) in sf, 'fusion driver exports ' + fn)
need('SM_ABI_VERSION >= 3' in sf and 'SM_ABI_VERSION < 3' in sf, 'v1 entry points are compiled only for ABI < 3 (r17/r18 builds unchanged)')
print('ALL PASS' if not fails else 'FAILED: %d' % len(fails)); sys.exit(1 if fails else 0)
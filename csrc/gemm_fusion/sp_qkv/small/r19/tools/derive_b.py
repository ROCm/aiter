import sys,re
p=sys.argv[1]+'/device/ran_kernel.cu'
t=open(p,encoding='utf-8',newline='').read(); nl='\r\n' if '\r\n' in t else '\n'; t=t.replace('\r\n','\n')
def rep(old,new,cnt=1):
    global t
    assert t.count(old)==cnt,('count %d != %d: '%(t.count(old),cnt))+old[:110]
    t=t.replace(old,new)
# qnan helper (before decide)
rep('SM_INL uint32_t decide(','SM_INL float qnan() { return __builtin_bit_cast(float, 0x7FC00000u); }      // r19 NANMODE value (bf16 0x7FC0 pattern after rounding)\nSM_INL uint32_t decide(')
# B poll call
rep('s_ticket = poll_range64(a, cwin_local(a, SM_T1_FLAG(s)), K * e, K * (e + 1ull), 0x20u, &it) ? 1u : 0u;','s_ticket = poll_range64(a, ka.err, rank, E, cwin_local(a, SM_T1_FLAG(s)), K * e, K * (e + 1ull), &it) ? 1u : 0u;')
# deferred CAS
rep('poison(a, errc);       // deferred cold CAS (r5 did it inline before the loads\' consumers)','set_err(a, ka.err, rank, errc & SM_ERR_REASON_MASK, 0ull, 0ull, 0ull);       // deferred cold CAS (r19: the sticky per-comm ERR word); codes are the 0x40-0x42 contract reasons')
# IK owner: NaN input for the self fusion in NANMODE
rep('vv[pass][i] = 0.0f; }','vv[pass][i] = qnan(); }')
rep('        if (ok) {                                                                  // self fusion AFTER the remote release (does not delay the other ranks)','        {                                                                          // self fusion AFTER the remote release (does not delay the other ranks); r19: also in NANMODE (vv stays NaN)')
# self WG
rep('        if (ok) {\n            float v[2][8];','        {                                                                          // r19: always runs; NANMODE feeds NaN\n            float v[2][8];')
n=len(re.findall(r'\bv\[pass\]\[i\] = red_u\(pp, ws\[i\], as_\[pass\]\);',t)); assert n==1,n
t=re.sub(r'\bv\[pass\]\[i\] = red_u\(pp, ws\[i\], as_\[pass\]\);','v[pass][i] = ok ? red_u(pp, ws[i], as_[pass]) : qnan();',t)
# B WG: top of both load blocks
top_old='        if (ok) {\n            float x[NH][2][8];\n            _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {\n                const uint32_t rc = rowc('
top_new='        {                                                                          // r19: always runs; !ok (ERR set / poll failure / tag mismatch) feeds NaN into head_row\n            float x[NH][2][8];\n            if (ok) {\n            _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {\n                const uint32_t rc = rowc('
pat=re.compile(r'        if \(ok\) \{\n            float x\[NH\]\[2\]\[8\];([^\n]*)\n            _Pragma\("unroll"\) for \(uint32_t kk = 0; kk < NH; \+\+kk\) _Pragma\("unroll"\) for \(uint32_t pass = 0; pass < 2u; \+\+pass\) \{\n                const uint32_t rc = rowc\(')
assert len(pat.findall(t))==2,len(pat.findall(t))
t=pat.sub(lambda m: top_new,t)
tag='''            // r19: the producers' tag (D-2: stored by the releasing thread before its release), loaded in the SAME batch as the Rx loads after the acquire (system-scope relaxed load = sc0 sc1, like the bypass Rx loads); compared once the batch has completed
            const uint64_t tg = ld_rlx64(cwin_local(a, SM_C_TAG(par, s)));
            if (__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(tg != E)) != 0u) {
                ok = false;
                if ((tid & 63) == 0) set_err(a, ka.err, rank, SM_ERRR_TAG_MISMATCH, E, tg, 0ull);
            }
            }   // if (ok)
            if (!ok) {
                _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) _Pragma("unroll") for (int i = 0; i < 8; ++i) x[kk][pass][i] = qnan();
            }
'''
# P1 path: before the NO_CREDIT stamps
rep('            }\n#if SM_NO_CREDIT\n            if (tid == 0) { SM_STAMP_VAL(rec, 34,','            }\n'+tag+'#if SM_NO_CREDIT\n            if (tid == 0) { SM_STAMP_VAL(rec, 34,')
# legacy path (P1 == 0): between the loads loop and the head_row loop
leg_old='#endif\n            }\n            _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {\n                const uint32_t m = static_cast<uint32_t>(row16) + 16u * pass;\n                if (pass < passes && SM_BPASS_ON(pass)) head_row('
rep(leg_old,'#endif\n            }\n'+tag+'            _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {\n                const uint32_t m = static_cast<uint32_t>(row16) + 16u * pass;\n                if (pass < passes && SM_BPASS_ON(pass)) head_row(')
open(p,'w',encoding='utf-8',newline='').write(t.replace('\n',nl)); print('stage2 ok')
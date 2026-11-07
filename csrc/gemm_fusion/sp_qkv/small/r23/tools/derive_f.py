import sys

p = sys.argv[1] + "/tests/check_r23_static.py"
with open(p, encoding="utf-8") as _f:
    t = _f.read()


def rep(a, b):
    global t
    assert t.count(a) == 1, a[:90]
    t = t.replace(a, b)


rep(
    'r"(if \\(tid == (?:0|256)\\) \\{[^\\n]*?tag_store\\(a,)"',
    'r"(if \\((?:itr == 0 && )?tid == (?:0|256)\\) \\{[^\\n]*?tag_store\\(a,)"',
)
rep(
    '"if ((tid & 63) == 0 && (tid >> 6) < 3 && ok) {" in body',
    '"if (itr == SM_ITERS - 1u && (tid & 63) == 0 && (tid >> 6) < 3 && ok) {" in body',
)
rep(
    '"if ((tid & 63) == 0 && (tid >> 6) < 3) { const uint32_t rr_ = static_cast<uint32_t>(tid) >> 6; tag_store(',
    '"if (itr == 0 && (tid & 63) == 0 && (tid >> 6) < 3) { const uint32_t rr_ = static_cast<uint32_t>(tid) >> 6; tag_store(',
)
rep(
    '"const bool live_pass = (4u * qd) < passes;" in body',
    '"const bool live_pass = (8u * itr + lp0) < passes;" in body',
)
rep(
    '"const uint32_t pbase = 4u * qd + 2u * (static_cast<uint32_t>(tid) >> 8);" in body',
    '"const uint32_t pbase0 = 4u * qd + 2u * (static_cast<uint32_t>(tid) >> 8);" in body and "const uint32_t pbase = pbase0 + 8u * itr;" in body',
)
rep(
    'body.index("if (ok && s != SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)")',
    'body.index("if (ok && itr == 0 && s != SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)")',
)
new = """
# ---- r23 MP256 v1 (R14 card) ----
need("#define SM_MP        256u" in abi and "#define SM_ITERS     (SM_MP / 128u)" in abi and "#define SM_NPASS     (128u / 16u)" in abi and "#define SM_NPASS_TOT (SM_MP / 16u)" in abi, "SM_MP 256, SM_ITERS = MP/128 chunks, role counts per 128-row chunk (SM_NPASS 8), total passes 16")
need("SM_GEMM_THREADS     (SM_MP > 128u ? 384u : 256u)" in abi and "MP256 needs SM_RX3=1" in abi, "GEMM 384 threads at MP256; MP256 requires RX3 + B_QUAD + BN128 (#error otherwise)")
need("SM_RX_BYTES == 7077888ull" in lay and "SM_RX_BYTES == 7864320ull" in lay and "SM_RX_BYTES > 4ull * 1024ull * 1024ull && SM_RX_BYTES <= 8ull * 1024ull * 1024ull" in lay, "MP256 Rx window: 3 slots 7,077,888 / 7,864,320 B needs and fits the 8 MiB view (X-5 exact values)")
need("SM_ITERS != 2u || (SM_MP == 256u && SM_B_QUAD && !SM_PUSH_PAR)" in rl and "SM_MP == 128u * SM_ITERS" in rl, "t1_roles: MP = 128 x ITERS, r22 role counts, K unchanged (16 / 20), one release after the last iteration")
need(body.count("for (uint32_t itr = 0; itr < SM_ITERS; ++itr) {") == 4 and k.count("}   // itr") == 4, "iteration loops in the 4 RAN roles (IK, push, self, B-octo), brace-balanced")
need(body.count("const bool live_it = (8u * itr) < passes;") == 3 and body.count("if (live_it) {") == 3, "dead iteration (rows >= M): no P / Ascale loads in IK, push, self")
need(body.count("itr == SM_ITERS - 1u && tid == 0 && ok") == 2 and "itr == SM_ITERS - 1u && (tid & 63) == 0" in body, "push and IK release ONCE after the LAST iteration (K unchanged)")
need("if (itr == 0) tg = ld_rlx64(cwin_local(a, SM_C_TAG(par, s)));" in body and "if (ok && itr == 0)" in body and "if (itr > 0 && ok && s_ticket != 1u) ok = false;" in body, "B octo: flag poll/acquire and tag check ONCE (first iteration); later iterations reuse s_ticket / s_tagbad")
need("hi * 8 + 2 * t + cw < AQ" in g and "const int hi = cw_all >> 1, cw = cw_all & 1;" in g and "hi * 128 + t * 32 + u_cm(y)" in g and "case 15u: smg::gemm_body<15>" in g and "__launch_bounds__(SM_GEMM_THREADS)" in g, "GEMM BM256: 4 consumer waves (hi half x 16-row group), per-tile dead mask over 16 groups, AQ 1..16, 384 threads")
need("template<int MB, int K> OPUS_D void wn()" in g and "(sks::PFK - 2) * MB <= 63" in g, "vmcnt immediate guard: live (PFK-2)*MB <= 63, dead arms clamped")
need("CONSUMER_WAVES = (SM_MP > 128u) ? 4 : 2" in tr and "constexpr int PFK = (SM_MP > 128u) ? 3 :" in tr, "traits: 4 consumer waves, PFK3 at BM256 (144 KiB LDS, 1 WG/CU)")
"""
m = 'print("ALL PASS" if not fails'
assert t.count(m) == 1
t = t.replace(m, new + m)
with open(p, "w", encoding="utf-8", newline="") as _f:
    _f.write(t)
print("ok")

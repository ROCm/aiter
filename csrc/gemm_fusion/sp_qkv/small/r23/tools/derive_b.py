import sys

r = sys.argv[1]


def rd(p):
    with open(p, encoding="utf-8", newline="") as _f:
        t = _f.read()
    nl = "\r\n" if "\r\n" in t else "\n"
    return t.replace("\r\n", "\n"), nl


def wr(p, t, nl):
    with open(p, "w", encoding="utf-8", newline="") as _f:
        _f.write(t.replace("\n", nl))


def rep(t, old, new, cnt=1):
    assert t.count(old) == cnt, (f"count {t.count(old):d} != {cnt:d}: ") + old[:100]
    return t.replace(old, new)


p = r + "/device/ran_kernel.cu"
t, nl = rd(p)
# B quad: loop + per-iteration pbase / live_pass
t = rep(
    t,
    "        const uint32_t pbase = 4u * qd + 2u * (static_cast<uint32_t>(tid) >> 8);",
    """#if SM_ITERS > 1 && !SM_B_QUAD
#error "SM_ITERS > 1 (MP256) needs SM_B_QUAD=1 (the B octo = 2 iterations of the quad)"
#endif
        for (uint32_t itr = 0; itr < SM_ITERS; ++itr) {                              // r23 MP256 B-octo: 2 iterations over 128-row chunks; poll/acquire and tag check ONCE (first iteration)
        const uint32_t pbase = 4u * qd + 2u * (static_cast<uint32_t>(tid) >> 8) + 8u * itr;""",
)
t = rep(
    t,
    "        const bool live_pass = (4u * qd) < passes;",
    "        const bool live_pass = (8u * itr + 4u * qd) < passes;",
)
# poll once
t = rep(
    t,
    "        if (ok && s != SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)",
    "        if (ok && itr == 0 && s != SM_TEST_SKIP_FLAG_ACQ_FOR_SRC - 1u)",
)
t = rep(
    t,
    "#else\n        if (ok)\n#endif\n        {\n            if (tid == 0) {\n                uint64_t it = 0;",
    "#else\n        if (ok && itr == 0)\n#endif\n        {\n            if (tid == 0) {\n                uint64_t it = 0;",
)
a = t.index(
    "            if (s_ticket != 1u) ok = false;\n            mori::cco::impl::syncThreads();\n        }\n"
)
a += len(
    "            if (s_ticket != 1u) ok = false;\n            mori::cco::impl::syncThreads();\n        }\n"
)
t = (
    t[:a]
    + "        if (itr > 0 && ok && s_ticket != 1u) ok = false;                       // r23: later iterations reuse the first iteration's poll result (LDS), the acquire happened once\n"
    + t[a:]
)
# tag check once
t = rep(
    t,
    "            const uint64_t tg = ld_rlx64(cwin_local(a, SM_C_TAG(par, s)));",
    "            uint64_t tg = E;                                                         // r23: later iterations skip the tag load (checked once, first iteration; s_tagbad persists)\n            if (itr == 0) tg = ld_rlx64(cwin_local(a, SM_C_TAG(par, s)));",
    2,
)
t = rep(
    t,
    "            mori::cco::impl::syncThreads();                                          // B-4: ONE decision gates BOTH halves",
    "            if (itr == 0) mori::cco::impl::syncThreads();                            // B-4: ONE decision gates BOTH halves",
    2,
)
# loop close
t = rep(t, "        nB = NH;\n    }", "        }   // itr\n        nB = NH;\n    }")
# SM_TEST_RX_BEFORE_FLAG seam: valid for the first iteration only
wr(p, t, nl)
print("B ok")

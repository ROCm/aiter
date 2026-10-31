import sys

p = sys.argv[1] + "/device/ran_kernel.cu"
with open(p, encoding="utf-8", newline="") as _f:
    t = _f.read()
nl = "\r\n" if "\r\n" in t else "\n"
t = t.replace("\r\n", "\n")


def rep(old, new):
    global t
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)


rep(
    """#if SM_ITERS > 1 && !SM_B_QUAD
#error "SM_ITERS > 1 (MP256) needs SM_B_QUAD=1 (the B octo = 2 iterations of the quad)"
#endif
        for (uint32_t itr = 0; itr < SM_ITERS; ++itr) {                              // r23 MP256 B-octo: 2 iterations over 128-row chunks; poll/acquire and tag check ONCE (first iteration)
        const uint32_t pbase = 4u * qd + 2u * (static_cast<uint32_t>(tid) >> 8) + 8u * itr;""",
    """#if SM_ITERS > 1 && !SM_B_QUAD
#error "SM_ITERS > 1 (MP256) needs SM_B_QUAD=1 (the B octo = 2 iterations of the quad)"
#endif
        const uint32_t pbase0 = 4u * qd + 2u * (static_cast<uint32_t>(tid) >> 8);""",
)
rep(
    "        const bool live_pass = (8u * itr + 4u * qd) < passes;",
    "        const uint32_t lp0 = 4u * qd;",
)
rep(
    "        const uint32_t pbase = 2u * (rsp >> 1);",
    "        const uint32_t pbase0 = 2u * (rsp >> 1);",
)
rep(
    "        const bool live_pass = rsp < passes; ",
    "        const uint32_t lp0 = rsp;\n        const bool live_pass_unused_ = rsp < passes; ",
)
rep(
    "#define SM_BPASS_ON(pp) ((pbase + (pp)) == rsp)\n#endif\n",
    "#define SM_BPASS_ON(pp) ((pbase + (pp)) == rsp)\n#endif\n        for (uint32_t itr = 0; itr < SM_ITERS; ++itr) {                              // r23 MP256 B-octo: 2 iterations over 128-row chunks; poll/acquire and tag check ONCE (first iteration)\n        const uint32_t pbase = pbase0 + 8u * itr;\n        const bool live_pass = (8u * itr + lp0) < passes;\n",
)
with open(p, "w", encoding="utf-8", newline="") as _f:
    _f.write(t.replace("\n", nl))
print("ok")

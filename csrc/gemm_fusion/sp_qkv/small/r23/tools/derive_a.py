import re
import sys

r = sys.argv[1]
p = r + "/device/ran_kernel.cu"
with open(p, encoding="utf-8", newline="") as _f:
    t = _f.read()
nl = "\r\n" if "\r\n" in t else "\n"
t = t.replace("\r\n", "\n")


def rep(old, new, cnt=1):
    global t
    assert t.count(old) == cnt, (f"count {t.count(old):d} != {cnt:d}: ") + old[:100]
    t = t.replace(old, new)


LIVE = "        const bool live_it = (8u * itr) < passes;                                       // r23: a dead iteration (all its 16-row passes >= M) loads nothing and stores nothing, but keeps the one-time duties (release / acquire)\n"
OPEN = "        for (uint32_t itr = 0; itr < SM_ITERS; ++itr) {                                   // r23 MP256: SM_ITERS = SM_MP / 128 iterations over 128-row chunks; pass base + 8 * it; one-time duties (early tag, release, poll, tag check) at the first / last iteration\n"
# ---- pbase lines (+ loop opening + live_it) ----
rep(
    "        const uint32_t pbase = 2u * (wgid - SM_T1_PUSH_WGS);",
    """%s        const uint32_t pbase = 2u * (wgid - SM_T1_PUSH_WGS) + 8u * itr;""".replace(
        "%s", OPEN
    ),
)
t = t.replace(
    "first global row pass of this IK owner\n",
    "first global row pass of this IK owner\n" + LIVE,
    1,
)
rep(
    "        const uint32_t pbase = 4u * (wgid / t1r::UNIT_WGS) + 2u * (static_cast<uint32_t>(tid) >> 8);",
    "%s        const uint32_t pbase = 4u * (wgid / t1r::UNIT_WGS) + 2u * (static_cast<uint32_t>(tid) >> 8) + 8u * itr;".replace(
        "%s", OPEN
    ),
)
i = t.index("2u * (static_cast<uint32_t>(tid) >> 8) + 8u * itr;")
j = t.index("\n", i)
t = t[: j + 1] + LIVE + t[j + 1 :]
rep(
    "        const uint32_t pbase = 2u * (sk_ % SM_SELF_SPLIT);",
    "%s        const uint32_t pbase = 2u * (sk_ % SM_SELF_SPLIT) + 8u * itr;".replace(
        "%s", OPEN
    ),
)
i = t.index("2u * (sk_ % SM_SELF_SPLIT) + 8u * itr;")
j = t.index("\n", i)
t = t[: j + 1] + LIVE + t[j + 1 :]
# ---- loads guarded by live_it (IK + self) ----
pat = re.compile(
    r'(        )(_Pragma\("unroll"\) for \(uint32_t pass = 0; pass < 2u; \+\+pass\) \{\n            const uint32_t rc = rowc\(static_cast<uint32_t>\(row16\) \+ 16u \* \(pbase \+ pass\), Mm1\);\n            _Pragma\("unroll"\) for \(int q = 0; q < SM_SK_CFG; \+\+q\) ld8f_a\(a\.P[^\n]*\n            as_\[pass\] = ka\.Ascale\[rc\];\n(?:            pre_a[^\n]*\n)?        \}\n)'
)
t, n = pat.subn(
    lambda m: m.group(1)
    + "if (live_it) {\n"
    + m.group(1)
    + m.group(2)
    + m.group(1)
    + "}\n",
    t,
)
assert n == 2, n
# push loads
a = t.index(
    '        _Pragma("unroll") for (uint32_t kk = 0; kk < 3u; ++kk) {\n            const uint32_t kc = (kk < nu) ? kk : (nu - 1u);'
)
b = t.index(
    "as_[pass] = ka.Ascale[rowc(static_cast<uint32_t>(row16) + 16u * (pbase + pass), Mm1)];\n",
    a,
)
b = t.index("\n", b) + 1
t = t[:a] + "        if (live_it) {\n" + t[a:b] + "        }\n" + t[b:]
# ---- one-time duties ----
rep(
    "if ((tid & 63) == 0 && (tid >> 6) < 3) { const uint32_t rr_ = static_cast<uint32_t>(tid) >> 6; tag_store(",
    "if (itr == 0 && (tid & 63) == 0 && (tid >> 6) < 3) { const uint32_t rr_ = static_cast<uint32_t>(tid) >> 6; tag_store(",
)
rep(
    '            if (tid == 0) { _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) tag_store(',
    '            if (itr == 0 && tid == 0) { _Pragma("unroll") for (uint32_t rr = 0; rr < 3u; ++rr) tag_store(',
)
rep(
    "            if (tid == 0) { tag_store(a, static_cast<int>(d), rank, par, E); SM_STAMP(rec, 38); }",
    "            if (itr == 0 && tid == 0) { tag_store(a, static_cast<int>(d), rank, par, E); SM_STAMP(rec, 38); }",
)
rep(
    "        if ((tid & 63) == 0 && (tid >> 6) < 3 && ok) {                             // B-H2",
    "        if (itr == SM_ITERS - 1u && (tid & 63) == 0 && (tid >> 6) < 3 && ok) {      // B-H2",
)
rep(
    "        if (tid == 0 && ok) {                                                      // ONE release per destination",
    "        if (itr == SM_ITERS - 1u && tid == 0 && ok) {                               // ONE release per destination",
)
rep(
    "        if (tid == 0 && ok) {                                               // ONE system RELEASE",
    "        if (itr == SM_ITERS - 1u && tid == 0 && ok) {                        // ONE system RELEASE (r23: after the LAST iteration)",
)
# ---- loop closers ----
rep(
    "        if (tid == 0) SM_STAMP(rec, 28);\n    } else\n#endif",
    "        if (tid == 0) SM_STAMP(rec, 28);\n        }   // itr\n    } else\n#endif",
)
rep(
    "#endif\n    } else if (wgid < SM_T1_B0) {",
    "#endif\n        }   // itr\n    } else if (wgid < SM_T1_B0) {",
)
rep(
    "        if (tid == 0) SM_STAMP(rec, 28);\n        nA = 1;\n    } else if (wgid < SM_T1_GRID) {",
    "        if (tid == 0) SM_STAMP(rec, 28);\n        }   // itr\n        nA = 1;\n    } else if (wgid < SM_T1_GRID) {",
)
with open(p, "w", encoding="utf-8", newline="") as _f:
    _f.write(t.replace("\n", nl))
print("ik/push/self ok")
